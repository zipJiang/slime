"""Role arguments for a fresh actor plus an independently pretrained critic."""
import copy
from pathlib import Path

from provenance import function_sha256
from warmstart_candidate import digest, require_pilot_candidate


def zero_warmup_role_arguments(args, candidate_path, context_source):
    candidate_path=Path(candidate_path).resolve()
    candidate=require_pilot_candidate(candidate_path)
    if args.num_critic_only_steps != 0:
        raise ValueError('Pretrained-critic pilot must configure zero critic warmup rounds')
    if args.start_rollout_id not in (None,0):
        raise ValueError('Fresh actor and pretrained critic must start at rollout zero')
    if Path(args.load).resolve()!=Path(candidate['base_actor']).resolve():
        raise ValueError('Actor must start from the base snapshot named by the critic candidate')
    if args.ckpt_step is not None:
        raise ValueError('Actor cannot inherit the critic checkpoint iteration')
    context_hash=function_sha256(context_source,'context')
    if context_hash!=candidate['context_function_sha256']:
        raise ValueError('PPO context function differs from critic pretraining')
    actor,critic=copy.deepcopy(args),copy.deepcopy(args)
    actor.start_rollout_id=critic.start_rollout_id=0
    flags=candidate['critic']
    critic.load=flags['load']
    critic.ckpt_step=flags['ckpt_step']
    critic.finetune=flags['finetune']
    critic.no_load_optim=flags['no_load_optim']
    critic.no_load_rng=flags['no_load_rng']
    lineage=dict(mode='base-actor-pretrained-critic-zero-warmup-pilot',
        candidate=str(candidate_path),candidate_sha256=digest(candidate_path),
        actor_base=str(Path(actor.load).resolve()),critic_checkpoint=flags['load'],
        critic_iteration=flags['ckpt_step'],start_rollout_id=0,
        actor_class='fresh_actor.FreshStartActor',
        critic_class='critic_actor.CheckpointCriticActor',
        context_source=str(Path(context_source).resolve()),
        context_function_sha256=context_hash)
    return actor,critic,lineage
