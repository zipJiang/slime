"""Reuse a fully audited first batch after restoring its exact paired checkpoint."""
import hashlib
import json
from pathlib import Path
import shutil

from recipe import RECIPE_ID


def read(path):
    return json.loads(path.read_text())


def restore_batch(args, rollout_id, case_keys, frozen, directory):
    source_run = getattr(args, 'ppo_retry_batch_run', None)
    if not source_run or rollout_id != args.start_rollout_id:
        return None
    old = Path(source_run).resolve()
    run = Path(args.save).parent.resolve()
    source = old/'rollouts'/f'train-{rollout_id:04d}'
    if old == run or not (old/'failed.json').exists() or (source/'training-complete.json').exists():
        raise ValueError('Retry requires an incomplete batch from a failed separate attempt')
    if read(old/'resume.json') != read(run/'resume.json'):
        raise ValueError('Retry must restore the exact same paired checkpoint and cursor')
    for root in (old, run):
        if not read(root/'initial-native-restore-audit.json')['passed']:
            raise ValueError('Retry requires verified native optimizer restoration')
    recipe = read(old/'recipe.json')
    if recipe['recipe_id'] != RECIPE_ID:
        raise ValueError('Retry recipe differs')
    for key in ('hf_checkpoint', 'seed', 'ppo_prior_strength', 'ppo_pass_tokens',
                'ppo_search_concurrency', 'rollout_batch_size', 'num_critic_only_steps',
                'rollout_temperature', 'rollout_top_p', 'rollout_top_k', 'seq_length'):
        if recipe['arguments'][key] != getattr(args, key):
            raise ValueError('Retry collection settings differ: '+key)
    lineage = read(source/'training-lineage.json')
    if (lineage['collection_round'] != rollout_id or lineage['behavior_round'] != rollout_id
            or lineage['actor_lag'] != 0 or lineage['critic_lag'] != 0):
        raise ValueError('Retry accepts only the first batch from the restored behavior')
    contract = read(source/'contract.json')
    if contract['case_keys'] != case_keys or contract['evaluation'] or contract['recipe_id'] != RECIPE_ID:
        raise ValueError('Retry question cursor or contract differs')
    for key in ('policy_version', 'value_version', 'server_weight_version', 'estimator'):
        if contract[key] != frozen[key]:
            raise ValueError('Retry behavior differs: '+key)
    audit = read(source/'target-replay-audit.json')
    if not audit['passed'] or audit['scope'] != 'whole_batch':
        raise ValueError('Retry requires a full target replay audit')
    for key in ('policy_version', 'value_version', 'recipe_id'):
        if audit[key] != contract[key]:
            raise ValueError('Retry audit does not describe the batch')
    version = frozen['value_version']
    publications = [read(root/'critic-publication'/f'{version}.json') for root in (old, run)]
    if (not all(p['passed'] for p in publications) or
            publications[0]['publication']['manifest']['sha256'] !=
            publications[1]['publication']['manifest']['sha256']):
        raise ValueError('Retry critic weights differ from the collected behavior')
    for group in range(args.rollout_batch_size):
        for suffix in ('.json', '.native.pkl.gz', '.actor.jsonl.gz', '.critic.jsonl.gz'):
            if not (source/f'group-{group:03d}{suffix}').is_file():
                raise ValueError('Retry batch is incomplete')
    hashes = {}
    for path in sorted(source.glob('group-*')):
        if path.is_file() and not path.name.endswith('audit.json'):
            with path.open('rb') as stream:
                hashes[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    staging = directory.with_name(directory.name+'.retry-copy')
    shutil.copytree(source, staging, ignore=shutil.ignore_patterns('*audit.json',
        'training-complete.json', 'training-lineage.json', 'critic-supervision.json'))
    for name, expected in hashes.items():
        with (staging/name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError('Retry batch changed while copying: '+name)
    provenance = dict(source=str(source), restored_checkpoint=read(run/'resume.json'),
        source_files=hashes, source_contract_sha256=hashlib.sha256((source/'contract.json').read_bytes()).hexdigest(),
        collection_reused=True, optimizer_updates_reused=False)
    (staging/'retry-batch-source.json').write_text(json.dumps(provenance, indent=2)+'\n')
    staging.rename(directory)
    return provenance
