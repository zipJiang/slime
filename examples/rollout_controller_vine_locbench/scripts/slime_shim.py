"""Bridge Slime batches to the frozen LocBench VinePPO collector."""
import gzip
import json
import os
from pathlib import Path
import subprocess
import time

from advantage_diagnostics import summarize as summarize_advantages
from batches import align_actor_records, training_data

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[2]


def read_rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def convert_actor_data(args, samples):
    rows = [sample.metadata['prepared_record'] for sample in samples]
    vpp = args.virtual_pipeline_model_parallel_size or 1
    mb_group = args.microbatch_group_size_per_vp_stage or 1
    alignment = args.data_parallel_size * (mb_group if vpp > 1 else 1)
    rows = align_actor_records(rows, alignment)
    return training_data(rows, lane='actor', expected_groups=range(args.rollout_batch_size))


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        raise ValueError('This one-epoch training driver does not run inline evaluation')
    from slime.rollout.base_types import RolloutFnTrainOutput
    from slime.utils.types import Sample

    started = time.time()
    run = Path(args.save).parent
    freeze = json.loads((run / 'collection-freeze.json').read_text())
    if freeze['rollout_id'] != rollout_id:
        raise ValueError('Collector freeze does not match current round')
    originals = data_source.get_samples(args.rollout_batch_size)
    case_ids = [group[0].metadata['query_id'] for group in originals]
    if len(case_ids) != args.rollout_batch_size or len(set(case_ids)) != len(case_ids):
        raise ValueError('Schedule produced an incomplete or duplicate LocBench batch')

    directory = run / 'rollouts' / f'train-{rollout_id:04d}'
    if directory.exists():
        raise ValueError('Refusing stale or partial rollout reuse')
    directory.mkdir(parents=True)
    (directory / 'collection-freeze.json').write_text(json.dumps(freeze, indent=2) + '\n')
    questions = directory / 'questions.json'
    questions.write_text(json.dumps(case_ids) + '\n')
    command = [
        str(ROOT / 'rollout-controller/.venv/bin/python'),
        str(EXPERIMENT / 'scripts/collect_rollouts.py'),
        '--checkpoint', args.hf_checkpoint,
        '--url', f'http://{args.sglang_router_ip}:{args.sglang_router_port}',
        '--value-version', freeze['value_version'],
        '--group-size', str(args.vine_group_size),
        '--policy-version', freeze['policy_version'],
        '--server-weight-version', freeze['server_weight_version'],
        '--output', str(directory), '--questions', str(questions),
        '--batch-size', str(args.rollout_batch_size),
        '--concurrency', str(args.ppo_search_concurrency),
        '--collection-profile', args.loc_collection_profile,
        '--estimator', freeze['estimator'],
        '--seed-namespace', f'{args.ppo_seed_namespace}/{rollout_id:04d}',
    ]
    with (directory / 'collector.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                       cwd=EXPERIMENT / 'snapshots/harness-vine',
                       env=dict(os.environ, PYTHONPATH=str(EXPERIMENT / 'snapshots/harness-vine')))
    summary = json.loads((directory / 'summary.json').read_text())
    (directory / 'collection-timing.json').write_text(json.dumps(dict(
        started_unix=started, finished_unix=time.time(), seconds=time.time() - started),
        indent=2) + '\n')
    if summary['server_weight_version'] != freeze['server_weight_version']:
        raise ValueError('Rollout behavior version changed')

    samples = []
    for group, case_id in enumerate(case_ids):
        for row in read_rows(directory / f'group-{group:03d}.actor.jsonl.gz'):
            sample = Sample(
                index=len(samples), rollout_id=group, group_index=group,
                prompt=case_id, tokens=row['tokens'],
                response_length=row['response_length'], loss_mask=row['loss_mask'],
                rollout_log_probs=row['rollout_log_probs'], reward=row['reward'],
                metadata=dict(prepared_record=row), status=Sample.Status.COMPLETED)
            sample.weight_versions = [freeze['server_weight_version']]
            samples.append(sample)
    diagnostics = summarize_advantages([s.metadata['prepared_record'] for s in samples])
    (directory / 'advantage-diagnostics.json').write_text(
        json.dumps(diagnostics, indent=2) + '\n')
    metrics = {f'collection_{key}': value for key, value in summary['cost'].items()}
    metrics.update({f'advantage_{key}': value for key, value in diagnostics.items()})
    terminal_count = sum(result['terminal_count'] for result in summary['results'])
    metrics['search_terminal_success'] = (
        sum(result['terminal_correct'] for result in summary['results']) / terminal_count)
    metrics['search_terminals'] = terminal_count
    return RolloutFnTrainOutput(samples=samples, metrics=metrics)
