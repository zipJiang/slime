"""Collect on-policy search, transport actor records, retain separate critic targets."""
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import subprocess
import shutil
import time

from batches import training_data
from advantage_diagnostics import summarize as summarize_advantages
from balanced_data import test_case_keys, validate_batch

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[2]


def read_rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def convert_actor_data(args, samples):
    rows = [s.metadata['prepared_record'] for s in samples]
    return training_data(rows, lane='actor', expected_groups=range(args.rollout_batch_size))


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    collection_started = time.time()
    from slime.rollout.base_types import RolloutFnTrainOutput, RolloutFnEvalOutput
    from slime.utils.types import Sample
    run = Path(args.save).parent
    freeze = json.loads((run/'collection-freeze.json').read_text())
    if not evaluation and freeze['rollout_id'] != rollout_id:
        raise ValueError('Collector freeze does not match current round')
    if evaluation:
        case_keys = test_case_keys()
    else:
        originals = data_source.get_samples(args.rollout_batch_size)
        case_keys = [group[0].metadata['case_key'] for group in originals]
        validate_batch(case_keys)
    directory = run/'rollouts'/f'{"eval" if evaluation else "train"}-{rollout_id:04d}'
    if (directory/'contract.json').exists():
        raise ValueError('Refusing stale or partial rollout reuse; start a fresh attempt directory')
    replay = warmup_replay = retry_replay = None
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'collection-freeze.json').write_text(json.dumps(freeze, indent=2)+'\n')
    questions = directory/'questions.json'
    questions.write_text(json.dumps(case_keys)+'\n')
    command = [str(ROOT/'rollout-controller/.venv/bin/python'), str(EXPERIMENT/'scripts/collect_rollouts.py'),
        '--checkpoint', args.hf_checkpoint,
        '--url', f'http://{args.sglang_router_ip}:{args.sglang_router_port}',
        '--value-version', freeze['value_version'], '--group-size', str(args.vine_group_size),
        '--value-rollouts-per-state', str(args.vine_value_rollouts_per_state),
        '--policy-version', freeze['policy_version'], '--server-weight-version', freeze['server_weight_version'],
        '--output', str(directory), '--questions', str(questions),
        '--split', 'val' if evaluation else 'train', '--pass-tokens', str(args.ppo_pass_tokens),
        '--concurrency', str(args.ppo_search_concurrency), '--prior-strength', str(args.ppo_prior_strength),
        '--estimator', freeze['estimator']]
    if getattr(args, 'ppo_seed_namespace', None):
        command += ['--seed-namespace', f'{args.ppo_seed_namespace}/{rollout_id:04d}']
    if evaluation:
        command += ['--evaluation', '--eval-branches', str(args.n_samples_per_eval_prompt)]
    recovery = getattr(args, 'vine_replay_batch', None)
    if recovery and not evaluation and rollout_id == args.start_rollout_id:
        source = Path(recovery)
        source_run = source.parent.parent
        source_init = json.loads((source_run/'initialization.json').read_text())
        source_contract = json.loads((source/'contract.json').read_text())
        source_summary = json.loads((source/'summary.json').read_text())
        iteration = int((Path(args.load)/'latest_checkpointed_iteration.txt').read_text())
        if Path(source_init['resume']['checkpoint']).resolve() != (Path(args.load)/f'iter_{iteration:07d}').resolve():
            raise ValueError('Replay behavior checkpoint differs from restored actor')
        if (source/'training-complete.json').exists():
            raise ValueError('Recovery replay must be an untrained batch')
        if json.loads((source/'questions.json').read_text()) != case_keys:
            raise ValueError('Recovery replay differs from restored question cursor')
        if source_contract['group_size'] != args.vine_group_size:
            raise ValueError('Recovery replay has a different group size')
        for key in ('recipe_id', 'estimator', 'policy_version', 'server_weight_version'):
            if source_contract[key] != freeze[key]:
                raise ValueError('Recovery replay provenance mismatch: '+key)
        if len(source_summary['results']) != args.rollout_batch_size:
            raise ValueError('Recovery replay is incomplete')
        shutil.copytree(source, directory, dirs_exist_ok=True)
        (directory/'collection-freeze.json').write_text(json.dumps(freeze, indent=2)+'\n')
        (directory/'recovery-replay.json').write_text(json.dumps(dict(source=str(source.resolve()),
            checkpoint=source_init['resume']['checkpoint'], retained_behavior_logprobs=True),indent=2)+'\n')
        retry_replay = source
    if not replay and warmup_replay is None and retry_replay is None:
        with (directory/'collector.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                cwd=EXPERIMENT/'snapshots/harness', env=dict(os.environ, PYTHONPATH=str(EXPERIMENT/'snapshots/harness')))
    summary = json.loads((directory/'summary.json').read_text())
    (directory/'collection-timing.json').write_text(json.dumps(dict(
        started_unix=collection_started, finished_unix=time.time(),
        seconds=time.time()-collection_started), indent=2)+'\n')
    if summary['server_weight_version'] != freeze['server_weight_version']:
        raise ValueError('Rollout behavior version changed')
    metrics = {f'{"replayed" if warmup_replay or retry_replay else "collection"}_{key}': value for key, value in summary['cost'].items()}
    metrics['warmup_replay'] = int(warmup_replay is not None)
    metrics['retry_batch'] = int(retry_replay is not None)
    metrics.update(critic_requests=summary['critic_requests'], critic_contexts=summary['critic_contexts'])
    if evaluation:
        domains = defaultdict(list)
        for result in summary['results']:
            index = result['group_index']*args.n_samples_per_eval_prompt+result['branch']
            domains[result['case_key'].split('/')[0]].append(Sample(index=index, rollout_id=index,
                group_index=result['group_index'], prompt=result['case_key'], response=result['answer'] or '',
                reward=result['outcome'], response_length=result['cost']['output_tokens'], status=Sample.Status.COMPLETED))
        metrics['outcome_success'] = summary['correct']/summary['branches']
        return RolloutFnEvalOutput(data={d: dict(samples=s, rewards=[x.reward for x in s]) for d,s in domains.items()}, metrics=metrics)
    samples = []
    for group, case_key in enumerate(case_keys):
        for row in read_rows(directory/f'group-{group:03d}.actor.jsonl.gz'):
            sample = Sample(index=len(samples), rollout_id=group, group_index=group,
                prompt=case_key, tokens=row['tokens'], response_length=row['response_length'],
                loss_mask=row['loss_mask'], rollout_log_probs=row['rollout_log_probs'], reward=row['reward'],
                metadata=dict(prepared_record=row), status=Sample.Status.COMPLETED)
            sample.weight_versions = [freeze['server_weight_version']]
            samples.append(sample)
    diagnostics = summarize_advantages([s.metadata['prepared_record'] for s in samples])
    (directory/'advantage-diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    metrics.update({f'advantage_{k}': v for k, v in diagnostics.items()})
    count = sum(r['terminal_count'] for r in summary['results'])
    metrics['search_terminal_success'] = sum(r['terminal_correct'] for r in summary['results'])/count
    metrics['search_terminals'] = count
    # This is adaptively selected search success, never root-started accuracy.
    return RolloutFnTrainOutput(samples=samples, metrics=metrics)
