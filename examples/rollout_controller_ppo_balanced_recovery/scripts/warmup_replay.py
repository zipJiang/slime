"""Reuse audited empirical warmup data while restoring optimizer and cursor state."""
import hashlib
import json
import math
from pathlib import Path
import shutil
from recipe import RECIPE_ID


def restore_batch(args, rollout_id, case_keys, frozen, directory):
    source_run=getattr(args,'ppo_replay_warmup_run',None)
    if not source_run: return None
    source=Path(source_run)/'rollouts'/f'train-{rollout_id:04d}'
    if not (source/'summary.json').exists(): return None
    if not args.start_rollout_id <= rollout_id < args.num_critic_only_steps:
        raise ValueError('Replay is restricted to critic-only warmup rounds')
    old_recipe=json.loads((Path(source_run)/'recipe.json').read_text())
    if old_recipe['recipe_id']!=RECIPE_ID:
        raise ValueError('Unsupported warmup source recipe')
    old_args=old_recipe['arguments']
    for key in ['hf_checkpoint','seed','ppo_prior_strength','ppo_pass_tokens',
                'ppo_search_concurrency','rollout_batch_size']:
        if old_args[key]!=getattr(args,key): raise ValueError('Warmup replay recipe mismatch: '+key)
    if old_args['load']!=args.hf_checkpoint:
        raise ValueError('Warmup source actor was not initialized from the same base')
    contract=json.loads((source/'contract.json').read_text())
    if (contract['recipe_id']!=RECIPE_ID or contract['case_keys']!=case_keys
            or contract['estimator']!='refined_td' or contract['critic_supervision']!='nonterminal_only'
            or contract['horizon']!='shared_task_turns_plus_one_forced_submission'
            or contract['evaluation'] or frozen['policy_version']!='actor-0000'):
        raise ValueError('Warmup replay contract or question cursor mismatch')
    for key in ['policy_version','value_version','server_weight_version']:
        if contract[key]!=frozen[key]: raise ValueError('Warmup replay version mismatch: '+key)
    audit=json.loads((source/'target-replay-audit.json').read_text())
    if not audit['passed'] or audit['scope']!='whole_batch':
        raise ValueError('Warmup replay requires a whole-batch target/horizon audit')
    # Preserve the original data and contract exactly; only runtime audits
    # and training completion records are regenerated for this attempt.
    hashes={}
    for path in sorted(source.glob('group-*')):
        if path.is_file():
            with path.open('rb') as stream: hashes[path.name]=hashlib.file_digest(stream,'sha256').hexdigest()
    shutil.copytree(source,directory,ignore=shutil.ignore_patterns('*audit.json',
        'training-complete.json','training-lineage.json','critic-supervision.json'))
    (directory/'original-contract.json').write_bytes((source/'contract.json').read_bytes())
    provenance=dict(source=str(source.resolve()),source_recipe=RECIPE_ID,
        source_contract_sha256=hashlib.sha256((source/'contract.json').read_bytes()).hexdigest(),
        source_files=hashes,actor_unchanged=True,
        supervision='Empirical continuation outcomes; stored critic priors are not used as warmup targets')
    (directory/'warmup-replay-source.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return provenance


def compare_optimizer_replay(expected, restored, *, tolerance):
    """Bound restart drift while rejecting invalid probabilities or lineage."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Invalid optimizer replay tolerance')
    if (expected['version'] != restored['version'] or not expected['scores']
            or len(expected['scores']) != len(restored['scores'])):
        raise ValueError('Optimizer replay version or context count mismatch')
    if any(not math.isfinite(v) or not 0 <= v <= 1
           for result in (expected, restored) for v in result['scores']):
        raise ValueError('Invalid optimizer replay probability')
    errors = [abs(a-b) for a,b in zip(expected['scores'], restored['scores'], strict=True)]
    return dict(passed=max(errors) <= tolerance, tolerance=tolerance,
        max_abs_error=max(errors), mean_abs_error=sum(errors)/len(errors),
        abs_errors=errors, failed_context_indices=[i for i,e in enumerate(errors) if e > tolerance])
