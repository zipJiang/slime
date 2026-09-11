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
        manifest = json.loads((EXPERIMENT/'data/question-manifest.json').read_text())
        families = manifest['heldout_families']
    else:
        originals = data_source.get_samples(args.rollout_batch_size)
        families = [group[0].metadata['family'] for group in originals]
    directory = run/'rollouts'/f'{"eval" if evaluation else "train"}-{rollout_id:04d}'
    if (directory/'contract.json').exists():
        raise ValueError('Refusing stale or partial rollout reuse; start a fresh attempt directory')
    replay = not evaluation and rollout_id == 0 and args.ppo_replay_initial_batch
    if replay:
        verified = json.loads((run/'replay-initial-batch-verified.json').read_text())
        source = Path(verified['source'])
        if json.loads((source/'contract.json').read_text())['families'] != families:
            raise ValueError('Replay questions do not match the fresh data cursor')
        shutil.copytree(source, directory, ignore=shutil.ignore_patterns('*audit.json', 'cost-reference.json', 'training-complete.json'))
        (directory/'replay-source.json').write_text(json.dumps(verified, indent=2)+'\n')
    else:
        directory.mkdir(parents=True, exist_ok=True)
    (directory/'collection-freeze.json').write_text(json.dumps(freeze, indent=2)+'\n')
    questions = directory/'questions.json'
    questions.write_text(json.dumps(families)+'\n')
    command = [str(ROOT/'rollout-controller/.venv/bin/python'), str(EXPERIMENT/'scripts/collect_rollouts.py'),
        '--checkpoint', args.hf_checkpoint,
        '--url', f'http://{args.sglang_router_ip}:{args.sglang_router_port}',
        '--value-url', freeze['value_url'], '--value-version', freeze['value_version'],
        '--policy-version', freeze['policy_version'], '--server-weight-version', freeze['server_weight_version'],
        '--output', str(directory), '--questions', str(questions),
        '--split', 'val' if evaluation else 'train', '--pass-tokens', str(args.ppo_pass_tokens),
        '--concurrency', str(args.ppo_search_concurrency), '--prior-strength', str(args.ppo_prior_strength),
        '--estimator', freeze['estimator']]
    if getattr(args, 'ppo_seed_namespace', None):
        command += ['--seed-namespace', f'{args.ppo_seed_namespace}/{rollout_id:04d}']
    if evaluation:
        command += ['--evaluation', '--eval-branches', str(args.n_samples_per_eval_prompt)]
    if not replay:
        with (directory/'collector.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                cwd=EXPERIMENT/'snapshots/harness', env=dict(os.environ, PYTHONPATH=str(EXPERIMENT/'snapshots/harness')))
    summary = json.loads((directory/'summary.json').read_text())
    (directory/'collection-timing.json').write_text(json.dumps(dict(
        started_unix=collection_started, finished_unix=time.time(),
        seconds=time.time()-collection_started), indent=2)+'\n')
    if summary['server_weight_version'] != freeze['server_weight_version']:
        raise ValueError('Rollout behavior version changed')
    metrics = {f'collection_{key}': value for key, value in summary['cost'].items()}
    metrics.update(critic_requests=summary['critic_requests'], critic_contexts=summary['critic_contexts'])
    if evaluation:
        domains = defaultdict(list)
        for result in summary['results']:
            index = result['group_index']*args.n_samples_per_eval_prompt+result['branch']
            domains[result['family'].split('/')[0]].append(Sample(index=index, rollout_id=index,
                group_index=result['group_index'], prompt=result['family'], response=result['answer'] or '',
                reward=result['outcome'], response_length=result['cost']['output_tokens'], status=Sample.Status.COMPLETED))
        metrics['outcome_success'] = summary['correct']/summary['branches']
        return RolloutFnEvalOutput(data={d: dict(samples=s, rewards=[x.reward for x in s]) for d,s in domains.items()}, metrics=metrics)
    samples = []
    for group, family in enumerate(families):
        for row in read_rows(directory/f'group-{group:03d}.actor.jsonl.gz'):
            sample = Sample(index=len(samples), rollout_id=group, group_index=group,
                prompt=family, tokens=row['tokens'], response_length=row['response_length'],
                loss_mask=row['loss_mask'], rollout_log_probs=row['rollout_log_probs'], reward=row['reward'],
                metadata=dict(prepared_record=row), status=Sample.Status.COMPLETED)
            sample.weight_versions = [freeze['server_weight_version']]
            samples.append(sample)
    count = sum(r['terminal_count'] for r in summary['results'])
    metrics['search_terminal_success'] = sum(r['terminal_correct'] for r in summary['results'])/count
    metrics['search_terminals'] = count
    # This is adaptively selected search success, never root-started accuracy.
    return RolloutFnTrainOutput(samples=samples, metrics=metrics)
