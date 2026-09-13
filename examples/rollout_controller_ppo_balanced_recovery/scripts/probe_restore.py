"""Restore both native PPO roles on replacement GPUs without training or saving."""
import json
import os
from pathlib import Path
import traceback

import ray
from slime.utils.arguments import parse_args
from slime.ray.placement_group import create_actor_model, allocate_train_group
from slime.observability.logging_utils import configure_logger

from critic_actor import CheckpointCriticActor
from optimizer_restore import inspect_optimizer, validate_restore
from placement import pinned_placement
from resume import role_arguments
from train_slime import custom_args, write_json


def report_rank(actor):
    import torch
    return dict(optimizer=inspect_optimizer(actor),
        gpu=torch.cuda.get_device_name(),
        total_bytes=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved())


def run(args):
    configure_logger()
    out=Path(args.save).parent
    out.mkdir(parents=True,exist_ok=True)
    if (out/'probe.json').exists(): raise ValueError('Use a fresh probe output')
    args.use_wandb=False
    args.rollout_num_gpus=0
    args.ppo_execution='sync'
    args.save_hf=None
    args.num_gpus_per_node=args.actor_num_gpus_per_node
    actor_args,critic_args,resume=role_arguments(args)
    if resume is None or resume['actor_updates']<=0:
        raise ValueError('Probe requires a trained, fully audited paired checkpoint')
    write_json(out/'probe.json',dict(resume=resume,training_nodes=args.actor_num_nodes,
        training_gpus_per_node=args.actor_num_gpus_per_node,
        optimizer_updates_performed=0,checkpoint_writes=False))
    pgs=pinned_placement(args)
    groups=[]
    try:
        actor,starts=create_actor_model(actor_args,pgs,None)
        groups.append(actor)
        if starts!=[resume['start_rollout_id']]*4: raise ValueError('Actor cursor mismatch')
        actor_reports=ray.get([h.__ray_call__.remote(report_rank) for h in actor._actor_handlers])
        write_json(out/'actor-restored.json',dict(starts=starts,ranks=actor_reports))
        critic_args.save=str(out/'critic')
        critic_args.save_hf=None
        critic_args.lr=args.ppo_critic_lr
        critic_args.use_kl_loss=False
        critic_args.kl_coef=0
        critic_args.use_opd=False
        critic_args.disable_param_buffers_cpu_backup=False
        critic_args.custom_advantage_function_path=None
        critic_args.rollout_data_postprocess_path=None
        critic_args.custom_tis_function_path=None
        critic_args.init_method_std=.001
        critic=allocate_train_group(critic_args,args.actor_num_nodes,args.actor_num_gpus_per_node,
            pgs['critic'],role='critic',actor_cls=CheckpointCriticActor)
        groups.append(critic)
        starts=critic.create()
        if starts!=[resume['start_rollout_id']]*4: raise ValueError('Critic cursor mismatch')
        critic_reports=ray.get([h.__ray_call__.remote(report_rank) for h in critic._actor_handlers])
        restored={role:[r['optimizer'] for r in reports] for role,reports in
            [('actor',actor_reports),('critic',critic_reports)]}
        audit=validate_restore(resume,restored,args.global_batch_size)
        write_json(out/'complete.json',dict(**audit,memory=dict(actor=actor_reports,
            critic=critic_reports),optimizer_updates_performed=0,checkpoint_writes=False))
    except BaseException:
        write_json(out/'failed.json',dict(traceback=traceback.format_exc()))
        raise
    finally:
        for group in reversed(groups): group.release()


if __name__=='__main__':
    args=parse_args(custom_args)
    ray.init(address=os.environ['RAY_ADDRESS'],runtime_env={'env_vars':{
        'GLOO_SOCKET_IFNAME':'ens0','NCCL_SOCKET_IFNAME':'ens0','NCCL_IB_DISABLE':'1',
        'PYTHONPATH':os.environ['PYTHONPATH']}})
    run(args)
