"""One pass of exact-token task and compaction SFT; full native and HF exports."""
import hashlib
import json
import math
import os
from pathlib import Path
import time

import ray
from slime.ray.placement_group import allocate_train_group
from slime.utils.arguments import parse_args
from slime.observability.logging_utils import configure_logger
from runtime_v2 import EXPERIMENT as _E
import sys
sys.path.append(str(_E/'snapshots/native-support-v1'))
from batches import partition_data,put_packets
from imitation_actor import ImitationActor
from imitation_data import load_dataset,question_packet,digest,read,training_order
from train_critic import placement,write

def status(out, stage, **fields):
    write(out/'status.json', dict(stage=stage, time=time.time(), **fields))
from audit_checkpoint import audit as audit_checkpoint

EXPERIMENT=Path(__file__).resolve().parents[1]


def custom_args(parser):
    parser.add_argument('--imitation-source',required=True)
    parser.add_argument('--imitation-smoke',action='store_true')
    parser.add_argument('--imitation-preflight-only',action='store_true')
    return parser


def train(args):
    configure_logger();out=Path(args.save).parent;out.mkdir(parents=True,exist_ok=True)
    if (out/'recipe.json').exists():raise ValueError('Use a fresh imitation output')
    source=Path(args.imitation_source)
    data,inventory=load_dataset(source,read(EXPERIMENT/'data/split.json'),require_complete=not args.imitation_smoke)
    if args.loss_type!='sft_loss' or args.compute_advantages_and_returns or args.use_critic or args.use_kl_loss:
        raise ValueError('Imitation requires pure supervised NLL with no PPO advantages or critic')
    if args.calculate_per_token_loss:raise ValueError('SFT averages target tokens per question, then questions')
    args.save_hf=str(out/'hf'/'iter_{rollout_id:07d}')
    args.finetune=True;args.no_load_optim=True;args.no_load_rng=True;args.start_rollout_id=0
    args.custom_advantage_function_path=None;args.rollout_data_postprocess_path=None
    args.custom_tis_function_path=None;args.kl_coef=0;args.use_opd=False
    groups=data['train'];order=training_order(groups,args.global_batch_size,smoke=args.imitation_smoke)
    updates=len(order)//args.global_batch_size
    args.num_rollout=updates
    write(out/'recipe.json',dict(protocol='locbench-iterative-imitation-v1',objective='supervised_imitation',
        source=str(source.resolve()),source_index_sha256=digest(source/'training-index.json') if not args.imitation_smoke else None,
        train_questions=order,development_questions=sorted(data['development']),epochs=1,updates=updates,
        weighting='Mean target-token NLL per question, averaged across questions; both task and fold',
        smoke=args.imitation_smoke,arguments={k:v for k,v in vars(args).items() if 'key' not in k.lower()},
        scripts={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}))
    write(out/'inventory.json',inventory)
    status(out,'initialization',updates=0,total_updates=updates)
    pg,physical=placement(args.actor_num_nodes*2);write(out/'placement.json',physical)
    actor=allocate_train_group(args,args.actor_num_nodes,2,pg,role='actor',actor_cls=ImitationActor)
    cursors=actor.create()
    if cursors!=[0]*(args.actor_num_nodes*2):raise ValueError('SFT must initialize from base at zero')
    initial_optim=ray.get([a.audit_optimizer_start.remote() for a in actor._actor_handlers])
    if not all(v['fresh'] for v in initial_optim):raise ValueError('SFT optimizer not fresh')
    parallel=dict(dp_size=args.actor_num_nodes,cp_size=1,vpp_size=1,microbatch_group_size_per_vp_stage=1)
    def packets(question_rows):
        import copy
        local=copy.copy(args);local.global_batch_size=len(question_rows)
        packet=question_packet(question_rows)
        return packet,put_packets(partition_data(local,parallel,packet))
    def evaluate(label):
        totals={tag:dict(nll=0.,tokens=0) for tag in ('task','fold')}
        questions=sorted(data['development'])
        if args.imitation_smoke:questions=questions[:2]
        for start in range(0,len(questions),args.actor_num_nodes):
            rows=[data['development'][q] for q in questions[start:start+args.actor_num_nodes]]
            real_rows=sum(map(len,rows))
            while len(rows)<args.actor_num_nodes:rows.append(rows[-1])
            packet,refs=packets(rows)
            reports=ray.get([a.evaluate_targets.remote(refs) for a in actor._actor_handlers])
            unique={}
            for report in reports:
                for row in report:
                    if row['index'] in unique and row!=unique[row['index']]:raise ValueError('TP evaluation disagreement')
                    unique[row['index']]=row
            flat=[r for question in rows for r in question]
            if sorted(unique)!=list(range(len(flat))):raise ValueError('Evaluation dropped target spans')
            for i,row in enumerate(flat[:real_rows]):
                result=unique[i];tag=row['metadata']['tag']
                if result['tokens']!=sum(row['loss_mask']) or not math.isfinite(result['nll']):
                    raise ValueError('Invalid held-out likelihood')
                totals[tag]['nll']+=result['nll'];totals[tag]['tokens']+=result['tokens']
        report={tag:dict(**v,mean_nll=v['nll']/v['tokens']) for tag,v in totals.items()}
        write(out/'validation'/f'{label}.json',report);return report
    status(out,'initial-evaluation',updates=0);initial=evaluate('base')
    for step,start in enumerate(range(0,len(order),args.global_batch_size)):
        _,refs=packets([groups[q] for q in order[start:start+args.global_batch_size]])
        from memory_qualification import reset_peak,memory_report
        ray.get([h.__ray_call__.remote(reset_peak) for h in actor._actor_handlers])
        before=time.time();ray.get(actor.async_train(step,refs))
        write(out/'training-memory'/f'update-{step+1:04d}.json',ray.get([h.__ray_call__.remote(memory_report) for h in actor._actor_handlers]))
        status(out,'training',updates=step+1,total_updates=updates,seconds=time.time()-before)
        if args.save_interval and (step+1)%args.save_interval==0 and step+1<updates:
            status(out,'intermediate-save',updates=step+1,total_updates=updates)
            actor.save_model(step,force_sync=True)
            write(out/'checkpoints'/f'update-{step+1:04d}.json',dict(
                updates=step+1,next_question_offset=start+args.global_batch_size,
                native=str((Path(args.save)/f'iter_{step:07d}').resolve()),
                recipe_sha256=digest(out/'recipe.json'),readback_pending=True))
    status(out,'trained-evaluation',updates=updates);final=evaluate('trained')
    status(out,'save',updates=updates);actor.save_model(updates-1,force_sync=True);actor.release()
    checkpoint=Path(args.save)/f'iter_{updates-1:07d}'
    audit_checkpoint(checkpoint,updates,'actor')
    hf=out/'hf'/f'iter_{updates-1:07d}'
    # Finalizer reads every exported tensor and copies the base tokenizer/config.
    import subprocess
    finalize=EXPERIMENT.parent/'rollout_controller_ppo_balanced_recovery/scripts/finalize_hf.py'
    subprocess.run([os.sys.executable,str(finalize),'--source',args.hf_checkpoint,'--target',str(hf)],check=True)
    write(out/'complete.json',dict(protocol='locbench-iterative-imitation-v1',updates=updates,
        native=str(checkpoint.resolve()),model=str(hf.resolve()),initial=initial,final=final,
        smoke=args.imitation_smoke,source_index_sha256=digest(source/'training-index.json') if not args.imitation_smoke else None))
    status(out,'complete',updates=updates,model=str(hf.resolve()))


if __name__=='__main__':
    args=parse_args(custom_args)
    if args.imitation_preflight_only:
        from megatron.core.num_microbatches_calculator import ConstantNumMicroBatchesCalculator
        initialization=ConstantNumMicroBatchesCalculator(args.global_batch_size,args.micro_batch_size,
            args.actor_num_nodes,args.decrease_batch_size_if_needed,0)
        data,inventory=load_dataset(args.imitation_source,read(EXPERIMENT/'data/split.json'),require_complete=False)
        parallel=dict(dp_size=args.actor_num_nodes,cp_size=1,vpp_size=1,microbatch_group_size_per_vp_stage=1)
        reports=[]
        import copy
        for lane,width in [('train',args.global_batch_size),('development',args.actor_num_nodes)]:
            if lane=='train' and len(data[lane])%width==0:
                order=training_order(data[lane],width)
            else:
                order=sorted(data[lane])
            groups=[data[lane][q] for q in order]
            for start in range(0,len(groups),width):
                rows=groups[start:start+width]
                real=len(rows)
                while len(rows)<width:rows.append(rows[-1])
                local=copy.copy(args);local.global_batch_size=len(rows)
                packet=question_packet(rows);packets=partition_data(local,parallel,packet)
                if any(sum(p['global_batch_sizes'])!=len(rows) for p in packets):
                    raise ValueError('Native packing changed the question batch denominator')
                reports.append(dict(lane=lane,start=start,real_questions=real,packets=len(packets),sequences=len(packet['tokens'])))
        if not reports:raise ValueError('No samples to preflight')
        print(json.dumps(dict(preflight_passed=True,initialization_batch=initialization.current_running_global_batch_size,
            training_question_batch=args.global_batch_size,batches=reports,loss_type=args.loss_type,
            use_critic=args.use_critic,compute_advantages=args.compute_advantages_and_returns)))
    else:
        import traceback
        try:
            ray.init(address=os.environ['RAY_ADDRESS'],runtime_env={'env_vars':{
                'GLOO_SOCKET_IFNAME':'ens0','NCCL_SOCKET_IFNAME':'ens0','NCCL_IB_DISABLE':'1',
                'PYTHONPATH':os.environ['PYTHONPATH']}})
            train(args)
        except Exception:
            write(Path(args.save).parent/'failed.json',dict(traceback=traceback.format_exc(),unix_time=time.time()))
            raise
