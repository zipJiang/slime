"""Audit and compare paired end-to-end runs; no model or GPU dependency."""
import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def measure(run):
    start = load(run/'throughput-start.json')
    rounds = list(range(start['start_round'], start['last_round']+1))
    if len(rounds) < 4:
        raise ValueError('At least four joint updates are required for two interior cycles')
    recipe = load(run/'recipe.json')['arguments']
    if rounds[0] < recipe['num_critic_only_steps']:
        raise ValueError('Warmup cannot substitute for joint training timing')
    rows = []
    for round_id in rounds:
        directory = run/'rollouts'/f'train-{round_id:04d}'
        status = load(directory/'training-complete.json')
        summary = load(directory/'summary.json')
        contract = load(directory/'contract.json')
        replay = load(directory/'target-replay-audit.json')
        if not replay['passed'] or replay['scope'] != 'whole_batch':
            raise ValueError('Every measured batch needs full target replay')
        gates = [load(p) for p in sorted((run/'on-policy-audit').glob(f'round-{round_id:04d}-rank-*.json'))]
        if len(gates) != 4 or any(g['mean_abs_difference'] > .05 or g['p99_abs_difference'] > .5 for g in gates):
            raise ValueError('Missing or failing native behavior-probability audit')
        rows.append(dict(round=round_id, output_tokens=summary['cost']['output_tokens'],
            input_tokens=summary['cost']['input_tokens'], families=contract['families'],
            pass_tokens=contract['pass_tokens'], seed_namespace=contract['seed_namespace'],
            seconds=status['seconds'], elapsed=status['throughput_elapsed'],
            lineage=status['lineage'], optimizer_window=status['optimizer_window'],
            collection_window=load(directory/'collection-timing.json')))
    overlap = start['execution'] == 'overlap'
    if overlap:
        publications = [load(p) for p in (run/'critic-publication').glob('*.json')]
        if len(publications) != len(rounds)+1 or any(
                p['max_abs_error'] > p['tolerance'] or not p['deterministic'] for p in publications):
            raise ValueError('Missing or failing critic publication equivalence checks')
        if not any(row['lineage']['actor_lag'] == 1 for row in rows):
            raise ValueError('No actual overlapped batch was trained')
        overlap_seconds = sum(max(0., min(a['optimizer_window']['finished_unix'], b['collection_window']['finished_unix'])
            - max(a['optimizer_window']['started_unix'], b['collection_window']['started_unix']))
            for a,b in zip(rows, rows[1:]))
        if overlap_seconds <= 0:
            raise ValueError('Collection did not actually overlap optimization')
    else:
        overlap_seconds = 0.
    final_iteration = rounds[-1]
    for role in ('actor', 'critic'):
        audit = load(run/role/f'iter_{final_iteration:07d}-readback.json')
        steps = final_iteration+1-(recipe['num_critic_only_steps'] if role == 'actor' else 0)
        if not audit['full_storage_read'] or not audit['finite_tensors'] or audit['optimizer_steps'] != [steps]:
            raise ValueError('Final paired checkpoint is not fully verified')
    if not (run/'actor/rollout'/f'global_dataset_state_dict_{final_iteration}.pt').is_file():
        raise ValueError('Missing exact resume cursor')
    elapsed = rows[-1]['elapsed']
    output_tokens = sum(row['output_tokens'] for row in rows)
    interior = rows[1:-1]
    steady_seconds = sum(row['seconds']['total'] for row in interior)
    return dict(run=str(run.resolve()), execution=start['execution'], rows=rows,
        elapsed_seconds=elapsed, startup_seconds=start['startup_seconds'], output_tokens=output_tokens,
        tokens_per_second=output_tokens/elapsed, updates_per_hour=len(rows)*3600/elapsed,
        including_startup_tokens_per_second=output_tokens/(elapsed+start['startup_seconds']),
        interior_updates=len(interior), interior_updates_per_hour=len(interior)*3600/steady_seconds,
        interior_tokens_per_second=sum(row['output_tokens'] for row in interior)/steady_seconds,
        measured_collection_optimizer_overlap_seconds=overlap_seconds,
        total_gpus=recipe['actor_num_nodes']*recipe['actor_num_gpus_per_node']+recipe['rollout_num_gpus']+int(overlap),
        resume=load(run/'resume.json'), recipe_arguments=recipe)


def compare(sync, overlap):
    if sync['execution'] != 'sync' or overlap['execution'] != 'overlap':
        raise ValueError('Expected synchronous baseline and overlap candidate')
    if sync['resume'] != overlap['resume'] or sync['total_gpus'] != overlap['total_gpus']:
        raise ValueError('Different initial checkpoint/cursor or total GPU allocation')
    for key in ('lr', 'ppo_critic_lr', 'ppo_pass_tokens', 'ppo_search_concurrency',
                'ppo_prior_strength', 'rollout_batch_size', 'ppo_seed_namespace',
                'max_tokens_per_gpu', 'seq_length', 'eps_clip', 'kl_loss_coef'):
        if sync['recipe_arguments'][key] != overlap['recipe_arguments'][key]:
            raise ValueError(f'Unmatched recipe setting: {key}')
    for a,b in zip(sync['rows'], overlap['rows'], strict=True):
        if any(a[k] != b[k] for k in ('round', 'families', 'pass_tokens', 'seed_namespace')):
            raise ValueError('Unmatched questions, budgets, or random seed namespace')
    gain = overlap['tokens_per_second']/sync['tokens_per_second']-1
    steady_gain = overlap['interior_tokens_per_second']/sync['interior_tokens_per_second']-1
    output_ratio = overlap['output_tokens']/sync['output_tokens']
    comparable = .9 <= output_ratio <= 1.1
    promote = comparable and steady_gain >= .2 and gain > 0
    return dict(sync=sync, overlap=overlap, throughput_gain=gain,
        steady_throughput_gain=steady_gain, output_token_ratio=output_ratio, comparable_realized_budget=comparable,
        threshold=.2, prioritize_overlap=promote,
        decision='overlap_plus_algorithm' if promote else 'synchronous_algorithm_only',
        note='Throughput benchmark only; accuracy requires the matched heldout training comparison.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sync', type=Path, required=True)
    parser.add_argument('--overlap', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = compare(measure(args.sync), measure(args.overlap))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('sync', 'overlap')}, indent=2))
