"""Native TP=2 critic pretraining, validation, and weight-only reload audit."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from slime.ray.placement_group import InfoActor, allocate_train_group
from slime.utils.arguments import parse_args
from slime.observability.logging_utils import configure_logger
from transformers import AutoTokenizer

from batches import training_data, partition_data, put_packets
from audit_sampling import audit as audit_sampling
from audit_checkpoint import audit as audit_checkpoint
from critic_actor import CheckpointCriticActor
from critic_replica import FrozenCriticReplica
from critic_equivalence import compare_scores
from dataset import load_dataset, metrics, constant_baseline
from inference_probe import probe_indices
from provenance import dataset_inventory, function_sha256, validate_collection_provenance
from targets import checkpoint_fields
from value_service import CriticScorer
from warmstart_candidate import build_candidate

EXPERIMENT=Path(__file__).resolve().parents[1]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n');temp.replace(path)


def status(out,stage,**details):
    write(out/'status.json',dict(stage=stage,unix_time=time.time(),**details))


def custom_args(parser):
    parser.add_argument('--critic-collection',required=True)
    parser.add_argument('--critic-epochs',type=int,default=1)
    parser.add_argument('--critic-preflight-only',action='store_true')
    parser.add_argument('--critic-reload-tolerance',type=float,default=1e-5)
    parser.add_argument('--critic-eval-interval',type=int,default=0)
    return parser


def placement(num_hosts):
    # Reserve two GPUs on each host explicitly; every adjacent TP pair stays local.
    nodes=[n for n in ray.nodes() if n['Alive'] and n['Resources'].get('browsecomp_critic_train',0)>=2]
    if len(nodes)!=num_hosts: raise ValueError(f'Expected exactly {num_hosts} dedicated critic hosts')
    world_size = num_hosts * 2
    hosts=sorted(n['NodeManagerAddress'] for n in nodes)
    bundles=[dict(CPU=1,GPU=1,**{f'node:{host}':.001}) for host in hosts for _ in range(2)]
    pg=placement_group(bundles,strategy='PACK')
    ray.get(pg.ready(),timeout=180)
    actors=[InfoActor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,placement_group_bundle_index=i)).remote() for i in range(world_size)]
    physical=ray.get([a.get_ip_and_gpu_id.remote() for a in actors])
    for actor in actors: ray.kill(actor)
    order=sorted(range(world_size),key=lambda i:(physical[i][0],int(physical[i][1])))
    for offset in range(0,world_size,2):
        pair=[physical[i] for i in order[offset:offset+2]]
        if pair[0][0]!=pair[1][0] or len({int(p[1]) for p in pair})!=2:
            raise ValueError('TP pair crosses hosts or repeats a GPU')
    return (pg,order,[physical[i][1] for i in order]),physical


def run(args):
    configure_logger()
    if not math.isfinite(args.critic_reload_tolerance) or args.critic_reload_tolerance<0:
        raise ValueError('Reload tolerance must be finite and nonnegative')
    if args.critic_eval_interval<0:
        raise ValueError('Evaluation interval cannot be negative')
    out=Path(args.save).parent
    out.mkdir(parents=True,exist_ok=True)
    if (out/'recipe.json').exists(): raise ValueError('Use a fresh training output for each attempt')
    required_free=300*1024**3
    usage=shutil.disk_usage(out)
    storage=dict(path=str(out.resolve()),total_bytes=usage.total,used_bytes=usage.used,
                 free_bytes=usage.free,required_free_bytes=required_free,
                 passed=usage.free>=required_free)
    write(out/'storage-preflight.json',storage)
    if not storage['passed']:
        raise RuntimeError('Insufficient free storage for native critic checkpoint and export')
    split_path=EXPERIMENT/'data/split.json'
    provenance=validate_collection_provenance(EXPERIMENT,args.critic_collection,split_path)
    sampling=audit_sampling(Path(args.critic_collection)/'manifest.json')
    split=json.loads(split_path.read_text())
    data=load_dataset(args.critic_collection,split)
    if args.critic_epochs!=1: raise ValueError('Initial experiment is one pass; review validation before adding epochs')
    args.num_gpus_per_node=2
    args.save_hf=None
    args.use_kl_loss=False
    args.kl_coef=0
    args.use_opd=False
    args.disable_param_buffers_cpu_backup=False
    args.custom_advantage_function_path=None
    args.rollout_data_postprocess_path=None
    args.custom_tis_function_path=None
    args.init_method_std=.001
    if not args.use_critic or not args.offload_train or args.normalize_advantages:
        raise ValueError('Expected native unnormalized critic-only PPO loss path')
    write(out/'recipe.json',dict(protocol='browsecomp-base-critic-v1',
        collection=str(Path(args.critic_collection).resolve()),
        collection_provenance=provenance,
        collection_sampling={k:v for k,v in sampling.items() if k!='streams'},
        storage_preflight=storage,
        arguments={k:v for k,v in vars(args).items() if 'key' not in k.lower()},
        scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}))
    write(out/'sampling-audit.json',sampling)
    write(out/'dataset-inventory.json',dataset_inventory(data))
    status(out,'native-initialization',train_questions=len({r['group_index'] for r in data['train']}),
           validation_questions=len({r['group_index'] for r in data['validation']}))
    num_hosts=args.actor_num_nodes
    world_size=num_hosts*2
    pg,physical=placement(num_hosts)
    write(out/'placement.json',physical)
    critic=allocate_train_group(args,num_hosts,2,pg,role='critic',actor_cls=CheckpointCriticActor)
    starts=critic.create()
    if starts != [0]*world_size: raise ValueError('Fresh critic must start at rollout zero on all ranks')
    tokenizer=AutoTokenizer.from_pretrained(args.hf_checkpoint,local_files_only=True)
    parallel=dict(dp_size=num_hosts,cp_size=1,vpp_size=1,microbatch_group_size_per_vp_stage=1)
    def scorer_for(group):
        return CriticScorer(group._actor_handlers,args,parallel,tokenizer,tokenizer.eos_token_id)
    scorer=scorer_for(critic)
    baseline=constant_baseline(data['train'])
    def evaluate(version):
        predictions=[]
        scorer.begin(version)
        try:
            for start in range(0,len(data['validation']),8):
                contexts=[r['context'] for r in data['validation'][start:start+8]]
                predictions.extend(scorer.score(contexts,version)['scores'])
        finally: scorer.end()
        report=metrics(data['validation'],predictions,baseline)
        report['by_checkpoint']={}
        for name,is_root in [('root',True),('fold',False)]:
            selected=[(r,p) for r,p in zip(data['validation'],predictions,strict=True)
                      if (r['turn']==0)==is_root]
            if selected:
                report['by_checkpoint'][name]=metrics([r for r,p in selected],[p for r,p in selected],baseline)
        write(out/'validation'/f'{version}.json',dict(**report,predictions=predictions))
        return report,predictions
    status(out,'initial-evaluation')
    initial,_=evaluate('initial')
    groups={q:[r for r in data['train'] if r['group_index']==q] for q in sorted({r['group_index'] for r in data['train']})}
    order=sorted(groups,key=lambda q:hashlib.sha256(('critic-epoch0/'+q).encode()).hexdigest())
    if len(order)%args.global_batch_size: raise ValueError('Question batch would be incomplete')
    step=0
    for start in range(0,len(order),args.global_batch_size):
        question_ids=order[start:start+args.global_batch_size]
        records=[]
        for index,q in enumerate(question_ids):
            for record in groups[q]:
                record=dict(record,group_index=index)
                records.append(checkpoint_fields(record,tokenizer,sentinel_token_id=tokenizer.eos_token_id,
                    max_sequence_length=args.seq_length,warmup=True))
        packet=training_data(records,lane='critic',expected_groups=range(len(question_ids)))
        started=time.time()
        ray.get(critic.async_train(step,put_packets(partition_data(args,parallel,packet))))
        step+=1
        status(out,'training',updates=step,total_updates=len(order)//args.global_batch_size,
               last_update_seconds=time.time()-started)
        if (args.critic_eval_interval and step%args.critic_eval_interval==0
                and start+args.global_batch_size<len(order)):
            status(out,'intermediate-evaluation',updates=step)
            evaluate(f'update-{step:04d}')
    status(out,'trained-evaluation',updates=step)
    final,predictions=evaluate('trained')
    status(out,'native-save',updates=step,iteration=step-1)
    critic.save_model(step-1,force_sync=True)
    # Export before releasing trainer ranks; all TP ranks join the gather.
    snapshot=out/'inference'
    version='browsecomp-base-critic-v1'
    status(out,'portable-export',updates=step,iteration=step-1)
    ray.get([a.export_snapshot.remote(str(snapshot),version) for a in critic._actor_handlers])
    critic.release()
    # Read every model and optimizer tensor from storage independently of the
    # native loader. The model-only warmstart below intentionally skips these
    # optimizer tensors, so it cannot establish that they were saved intact.
    checkpoint=Path(args.save)/f'iter_{step-1:07d}'
    status(out,'checkpoint-readback',updates=step,iteration=step-1)
    audit_checkpoint(checkpoint,step,'critic')
    # A future fresh PPO run loads only critic model weights and resets its
    # optimizer/RNG/cursor independently of the base actor.
    warm=copy.deepcopy(args)
    warm.load=args.save;warm.ckpt_step=step-1
    warm.finetune=True;warm.no_load_optim=True;warm.no_load_rng=True
    status(out,'weight-only-reload',updates=step,iteration=step-1)
    loaded=allocate_train_group(warm,num_hosts,2,pg,role='critic',actor_cls=CheckpointCriticActor)
    cursors=loaded.create()
    if cursors != [0]*world_size: raise ValueError('Weight-only warmstart retained the pretraining cursor')
    optimizers=ray.get([a.audit_optimizer_start.remote() for a in loaded._actor_handlers])
    if len(optimizers)!=world_size or not all(report['fresh'] for report in optimizers):
        raise ValueError('Weight-only warmstart restored optimizer/scheduler history')
    scorer=scorer_for(loaded)
    reloaded,reloaded_predictions=evaluate('weight-only-reload')
    errors=[abs(a-b) for a,b in zip(predictions,reloaded_predictions,strict=True)]
    error=max(errors)
    write(out/'reload-audit.json',dict(max_abs_error=error,passed=error<=args.critic_reload_tolerance,
        tolerance=args.critic_reload_tolerance,abs_errors=errors,
        world_size=world_size,cursors=cursors,optimizers=optimizers,finetune=True,no_load_optim=True,no_load_rng=True))
    if error>args.critic_reload_tolerance: raise ValueError('Weight-only critic reload changed predictions beyond tolerance')
    loaded.release()
    write(out/'native-validated.json',dict(updates=step,checkpoint=args.save,iteration=step-1,
        initial=initial,trained=final,reload_max_abs_error=error,
        full_checkpoint_readback=str(Path(args.save)/f'iter_{step-1:07d}-readback.json'),
        unix_time=time.time()))
    # Verify the portable inference artifact at actual held-out prefixes.
    status(out,'portable-equivalence',updates=step,iteration=step-1)
    replica=ray.remote(num_gpus=1)(FrozenCriticReplica).options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg[0],
            placement_group_bundle_index=pg[1][0])).remote(args.hf_checkpoint,args.seq_length)
    publication=ray.get(replica.publish.remote(str(snapshot),version))
    ray.get(replica.begin.remote(version))
    lengths=[len(tokenizer.encode(r['context'],add_special_tokens=False)) for r in data['validation']]
    indices=probe_indices(data['validation'],lengths)
    contexts=[data['validation'][i]['context'] for i in indices]
    hf=ray.get(replica.score.remote(contexts,version))
    repeat=ray.get(replica.score.remote(contexts,version))
    expected=dict(version=version,scores=[predictions[i] for i in indices])
    comparison=compare_scores(expected,hf,repeat,
                              version=version,count=len(contexts))
    write(out/'inference-audit.json',dict(publication=publication,comparison=comparison,
        native=expected,replica=hf,
        contexts=[dict(validation_index=i,question=data['validation'][i]['group_index'],
            turn=data['validation'][i]['turn'],tokens=lengths[i],
            sha256=hashlib.sha256(data['validation'][i]['context'].encode()).hexdigest()) for i in indices]))
    ray.get(replica.end.remote(version));ray.kill(replica)
    write(out/'complete.json',dict(updates=step,initial=initial,trained=final,
        better_than_initial=final['mse']<initial['mse'],
        better_than_constant=final['mse']<final['baseline_mse'],
        native_checkpoint=args.save,native_iteration=step-1,
        inference=str(snapshot) if comparison['passed'] else None,
        portable_inference_passed=comparison['passed'],
        readiness='Requires focused audit and zero-warmup PPO pilot before assuming warmup can be removed',
        unix_time=time.time()))
    status(out,'candidate-finalization',updates=step,iteration=step-1,
           portable_inference_passed=comparison['passed'])
    write(out/'warmstart-candidate.json',build_candidate(out,
        base_model=args.hf_checkpoint,
        context_source_file_sha256=provenance['sources']['scripts/collect.py'],
        context_function_sha256=function_sha256(EXPERIMENT/'scripts/collect.py','context')))
    status(out,'complete',updates=step,iteration=step-1,
           portable_inference_passed=comparison['passed'])


def main():
    args=parse_args(custom_args)
    if args.critic_preflight_only:
        from types import SimpleNamespace
        records=[dict(tokens=[1,2,3],response_length=1,reward=float(i%2),loss_mask=[1],
            group_index=i,metadata=dict(lane='critic',node_id=i)) for i in range(args.global_batch_size)]
        parallel=dict(dp_size=args.actor_num_nodes,cp_size=1,vpp_size=1,microbatch_group_size_per_vp_stage=1)
        packets=partition_data(args,parallel,training_data(records,lane='critic',
            expected_groups=range(args.global_batch_size)))
        print(json.dumps(dict(preflight_passed=True,dp_packets=len(packets),
            sizes=[p['global_batch_sizes'] for p in packets],offload_train=args.offload_train,
            use_critic=args.use_critic,tf32=args.disable_tf32 if hasattr(args,'disable_tf32') else None)))
        return
    out=Path(args.save).parent
    try:
        ray.init(address=os.environ['RAY_ADDRESS'],runtime_env={'env_vars':{
            'GLOO_SOCKET_IFNAME':'ens0','NCCL_SOCKET_IFNAME':'ens0','NCCL_IB_DISABLE':'1',
            'PYTHONPATH':os.environ['PYTHONPATH']}})
        run(args)
    except Exception:
        if not (out/'failed.json').exists():
            write(out/'failed.json',dict(traceback=traceback.format_exc(),unix_time=time.time()))
        raise


if __name__=='__main__':
    main()
