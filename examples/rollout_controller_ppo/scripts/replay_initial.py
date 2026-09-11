"""Recover a validated initial warmup batch after a post-optimizer logging failure."""
import gzip
import hashlib
import json
from pathlib import Path


def verify_initial_batch(args, scorer, frozen):
    source = Path(args.ppo_replay_initial_batch).resolve()
    previous_run = source.parent.parent
    if args.start_rollout_id != 0 or args.ppo_critic_load or args.num_critic_only_steps <= 0:
        raise ValueError('Initial-batch replay only supports fresh-base critic warmup')
    if (previous_run/'scientific-rejection.json').exists():
        raise ValueError('Cannot replay a scientifically rejected batch')
    contract = json.loads((source/'contract.json').read_text())
    if contract.get('critic_supervision') != 'nonterminal_only':
        raise ValueError('Replay cannot cross the terminal-supervision correction')
    old_args = json.loads((previous_run/'recipe.json').read_text())['arguments']
    if args.load != args.hf_checkpoint or old_args['load'] != args.hf_checkpoint:
        raise ValueError('Replay must use the same original base actor, not a trained checkpoint')
    for key in ('hf_checkpoint', 'seed', 'ppo_prior_strength', 'ppo_pass_tokens',
                'ppo_search_concurrency', 'num_critic_only_steps', 'rollout_batch_size'):
        if old_args[key] != getattr(args, key):
            raise ValueError(f'Replay initialization/recipe mismatch: {key}')
    for key in ('policy_version', 'value_version', 'server_weight_version'):
        if contract[key] != frozen[key]:
            raise ValueError(f'Replay frozen version mismatch: {key}')
    if contract['horizon'] != 'shared_task_turns_plus_one_forced_submission' or contract['evaluation']:
        raise ValueError('Replay requires a valid shared-horizon training collection')
    audit = json.loads((source/'target-replay-audit.json').read_text())
    horizon = json.loads((source/'horizon-audit.json').read_text())
    if not audit['passed'] or audit['scope'] != 'whole_batch' or not horizon['passed'] or horizon['scope'] != 'whole_batch':
        raise ValueError('Replay requires complete target and horizon audits')
    records = []
    for path in sorted(source.glob('group-*.critic.jsonl.gz')):
        with gzip.open(path, 'rt') as stream:
            records.extend(json.loads(line) for line in stream)
    # Terminal checkpoints have an exact zero prior supplied by the estimator,
    # not the network. Every learned prior must match the reinitialized critic.
    probes = [r for r in records if r['metadata']['diagnostics']['prior'] != 0.]
    if not probes:
        raise ValueError('No initial critic priors to verify')
    differences = []
    for offset in range(0, len(probes), 16):
        rows = probes[offset:offset+16]
        actual = scorer.score([r['context'] for r in rows], frozen['value_version'])['scores']
        differences.extend(abs(v-r['metadata']['diagnostics']['prior']) for v,r in zip(actual,rows,strict=True))
    if max(differences) > 1e-5:
        raise ValueError(f'Reinitialized critic differs from replay priors: {max(differences)}')
    return dict(source=str(source), validated_priors=len(probes), max_prior_difference=max(differences),
        initial_actor='Same explicit base HF initialization, before any actor update',
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(source.glob('group-*.jsonl.gz'))})
