"""Native scalar critic warmup, selected on disjoint LocBench development questions."""
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback

from runtime_v2 import EXPERIMENT,MODEL,CONTEXT_LIMIT,contract,digest,verify_harness
sys.path.append(str(EXPERIMENT/'snapshots/native-support-v1'))
from warmup_data import load


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n');tmp.replace(path)


def custom_args(parser):
    parser.add_argument('--loc-collection',type=Path,required=True)
    parser.add_argument('--loc-epochs',type=int,default=2)
    parser.add_argument('--loc-eval-interval',type=int,default=2)
    parser.add_argument('--loc-root-mass',type=float,default=.25)
    parser.add_argument('--loc-patience',type=int,default=2)
    parser.add_argument('--loc-preflight',action='store_true')
    return parser


def placement(world):
    import ray
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from slime.ray.placement_group import InfoActor
    nodes=sorted((n for n in ray.nodes() if n['Alive'] and n['Resources'].get('locbench_critic_train',0)),key=lambda n:n['NodeManagerAddress'])
    slots=[n['NodeManagerAddress'] for n in nodes for _ in range(int(n['Resources']['locbench_critic_train']))]
    if len(slots)!=world:raise ValueError('Native trainer slot count differs from request')
    pg=placement_group([dict(CPU=1,GPU=1,**{f'node:{host}':.001}) for host in slots],strategy='PACK')
    ray.get(pg.ready(),timeout=180)
    probes=[InfoActor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,placement_group_bundle_index=i)).remote() for i in range(world)]
    physical=ray.get([a.get_ip_and_gpu_id.remote() for a in probes])
    for a in probes:ray.kill(a)
    order=sorted(range(world),key=lambda i:(physical[i][0],int(physical[i][1])))
    for i in range(0,world,2):
        pair=[physical[j] for j in order[i:i+2]]
        if pair[0][0]!=pair[1][0] or pair[0][1]==pair[1][1]:raise ValueError('Invalid TP pair')
    return (pg,order,[physical[i][1] for i in order]),physical


def run(args, *, model=MODEL, environment=None, actor_identity=None):
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from transformers import AutoTokenizer
    from slime.ray.placement_group import allocate_train_group
    from critic_actor import CheckpointCriticActor
    from critic_initialization import audit_initial_snapshot
    from trace_validation import TraceValidationReplica
    from trace_warmup import packet,baseline,choose_best,report
    from targets import checkpoint_fields
    from batches import partition_data,put_packets
    from value_service import CriticScorer
    from inference_probe import probe_indices
    from audit_checkpoint import audit as audit_checkpoint
    from critic_equivalence import compare_scores

    environment=contract() if environment is None else environment
    manifest=json.loads((args.loc_collection/'manifest.json').read_text())
    if manifest['environment']!=environment:raise ValueError('Critic environment differs from fresh collection')
    if actor_identity is not None and manifest.get('actor_identity')!=actor_identity:raise ValueError('Critic actor identity differs from collection')
    verify_harness();out=Path(args.save).parent;out.mkdir(parents=True,exist_ok=True)
    world=args.actor_num_nodes*args.actor_num_gpus_per_node
    if (world%2 or args.tensor_model_parallel_size!=2 or args.context_parallel_size!=1
        or args.pipeline_model_parallel_size!=1 or args.global_batch_size!=8
        or not args.use_critic or not args.offload_train or args.normalize_advantages
        or args.calculate_per_token_loss or args.lr!=1e-6 or args.seq_length!=CONTEXT_LIMIT):
        raise ValueError('Unsupported native critic recipe')
    data=load(args.loc_collection);constant=baseline(data['train'],args.loc_root_mass)
    args.load=model;args.ckpt_step=None;args.finetune=True;args.no_load_optim=True;args.no_load_rng=True
    args.start_rollout_id=0;args.critic_initial_value_probability=constant;args.init_method_std=.001
    args.save_hf=None;args.use_kl_loss=False;args.kl_coef=0;args.use_opd=False
    args.disable_param_buffers_cpu_backup=False;args.custom_advantage_function_path=None
    args.rollout_data_postprocess_path=None;args.custom_tis_function_path=None
    if (out/'recipe.json').exists():raise ValueError('Fresh training attempt required')
    write(out/'recipe.json',dict(protocol='locbench-native-critic-v1',environment=environment,actor_identity=actor_identity,model_initialization=model,
        constant=constant,collection_manifest_sha256=digest(args.loc_collection/'manifest.json'),
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if 'key' not in k.lower()},
        sources={p.name:digest(p) for p in (EXPERIMENT/'scripts').glob('*.py')}))
    write(out/'dataset-inventory.json',data)
    pg,physical=placement(world);write(out/'placement.json',physical)
    validators=[n for n in ray.nodes() if n['Alive'] and n['Resources'].get('locbench_critic_validator')==1]
    if len(validators)!=1 or validators[0]['NodeManagerAddress'] in {x[0] for x in physical}:
        raise ValueError('Dedicated standalone validation device required')
    validator=ray.remote(num_gpus=1)(TraceValidationReplica).options(scheduling_strategy=
        NodeAffinitySchedulingStrategy(validators[0]['NodeID'],soft=False)).remote(model,CONTEXT_LIMIT)
    critic=allocate_train_group(args,args.actor_num_nodes,args.actor_num_gpus_per_node,pg,
        role='critic',actor_cls=CheckpointCriticActor)
    starts=critic.create()
    optimizers=ray.get([a.audit_optimizer_start.remote() for a in critic._actor_handlers])
    if starts!=[0]*world or not all(r['fresh'] for r in optimizers):raise ValueError('Nonfresh critic initialization')
    write(out/'initialization-audit.json',dict(cursors=starts,optimizers=optimizers))
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    parallel=dict(dp_size=world//2,cp_size=1,vpp_size=1,microbatch_group_size_per_vp_stage=1)
    scorer=CriticScorer(critic._actor_handlers,args,parallel,tokenizer,tokenizer.eos_token_id)
    lengths=[len(tokenizer.encode(r['context'],add_special_tokens=False)) for r in data['validation']]
    indices=probe_indices(data['validation'],lengths);rows_ref=ray.put(data['validation'])
    def native(contexts,version):
        scorer.begin(version)
        try:
            values=[]
            for i in range(0,len(contexts),8):values.extend(scorer.score(contexts[i:i+8],version)['scores'])
            return dict(version=version,scores=values)
        finally:scorer.end()
    def evaluate(step):
        version=f'locbench-warmup-{step:04d}';directory=out/'exports'/version
        ray.get([a.export_snapshot.remote(str(directory),version) for a in critic._actor_handlers])
        if step==0:
            audit=audit_initial_snapshot(directory,model,constant);write(out/'initial-weights-audit.json',audit)
            if not audit['passed']:raise ValueError('Initial critic backbone or intercept mismatch')
        expected=native([data['validation'][i]['context'] for i in indices],version)
        future=validator.evaluate.remote(str(directory),version,rows_ref,indices,expected,constant,args.loc_root_mass,.03,.005)
        return dict(step=step,version=version,directory=str(directory),future=future)
    reports=[]
    def settle(pending):
        result=ray.get(pending['future']);meta={k:v for k,v in pending.items() if k!='future'}
        write(out/'validation'/f"update-{pending['step']:04d}.json",dict(**meta,**result))
        if result.get('rejected'):raise ValueError('Native/portable prediction mismatch')
        reports.append(dict(**meta,**result['report']))
        best=choose_best(reports)
        write(out/'selection.json',dict(best=reports[best],evaluated=reports))
        return len(reports)-best-1>=args.loc_patience
    settle(evaluate(0));initial=reports[0]
    groups={q:[r for r in data['train'] if r['group_index']==q] for q in sorted({r['group_index'] for r in data['train']})}
    if len(groups)%8:raise ValueError('Warmup question count must fill whole optimizer batches')
    step=0;pending=None;stop=False
    import hashlib
    for epoch in range(args.loc_epochs):
        order=sorted(groups,key=lambda q:hashlib.sha256(f'locbench-warmup/{epoch}/{q}'.encode()).hexdigest())
        for i in range(0,len(order),8):
            records=[]
            for group,q in enumerate(order[i:i+8]):
                for row in groups[q]:
                    prepared=checkpoint_fields(dict(row,group_index=group),tokenizer,
                        sentinel_token_id=tokenizer.eos_token_id,max_sequence_length=CONTEXT_LIMIT,warmup=True)
                    records.append(dict(prepared,turn=row['turn']))
            start=time.time();packets=partition_data(args,parallel,packet(records,args.loc_root_mass))
            ray.get(critic.async_train(step,put_packets(packets)));step+=1
            write(out/'status.json',dict(stage='critic-warmup',updates=step,epoch=epoch+1,
                last_update_seconds=time.time()-start,world_size=world,records=len(records),time=time.time()))
            if step%args.loc_eval_interval==0:
                if pending:stop=settle(pending)
                critic.save_model(step-1,force_sync=True);pending=evaluate(step)
                if stop:break
        if stop:break
    if pending:settle(pending)
    best=reports[choose_best(reports)];critic.release()
    write(out/'warmup-finished.json',dict(updates=step,selected=best,stopped_early=stop))
    if best['step']==0:
        ray.kill(validator)
        write(out/'complete.json',dict(ready=False,reason='Starting critic selected; no held-out improvement'))
        return
    selected=best['step'];iteration=selected-1
    audit_checkpoint(Path(args.save)/f'iter_{iteration:07d}',selected,'critic')
    warm=copy.deepcopy(args);warm.load=args.save;warm.ckpt_step=iteration
    loaded=allocate_train_group(warm,args.actor_num_nodes,args.actor_num_gpus_per_node,pg,
        role='critic',actor_cls=CheckpointCriticActor)
    cursors=loaded.create();opts=ray.get([a.audit_optimizer_start.remote() for a in loaded._actor_handlers])
    if cursors!=[0]*world or not all(r['fresh'] for r in opts):raise ValueError('Model-only reload kept training state')
    scorer=CriticScorer(loaded._actor_handlers,warm,parallel,tokenizer,tokenizer.eos_token_id)
    actual=native([r['context'] for r in data['validation']],best['version'])
    expected=json.loads((out/'validation'/f'update-{selected:04d}.json').read_text())
    repeat=native([r['context'] for r in data['validation']],best['version'])
    comparison=compare_scores(dict(version=best['version'],scores=expected['predictions']),actual,repeat,
        version=best['version'],count=len(data['validation']),tolerance=.03,mean_tolerance=.005)
    write(out/'reload-audit.json',dict(**comparison,cursors=cursors,optimizers=opts,
        reference='full portable validation predictions',repeated='model-only native reload'))
    loaded.release();ray.kill(validator)
    if not comparison['passed']:raise ValueError('Selected native model-only reload mismatch')
    ready=(best['selection_mse']<initial['selection_mse'] and best['selection_mse']<best['selection_baseline_mse'])
    result=dict(ready=ready,selected_update=selected,optimizer_updates_executed=step,
        native_checkpoint=args.save,native_iteration=iteration,inference=best['directory'],
        environment=environment,actor_identity=actor_identity,model_initialization=model,context_source_sha256=digest(EXPERIMENT/'scripts/runtime_v2.py'),
        initial=initial,selected=best,validation_audit_sha256=digest(out/'reload-audit.json'))
    write(out/'complete.json',result)
    if ready:write(out/'warmstart-candidate.json',result)


def main():
    import ray
    from slime.utils.arguments import parse_args
    from slime.observability.logging_utils import configure_logger
    args=parse_args(custom_args);configure_logger()
    if args.loc_preflight:
        from batches import partition_data
        from trace_warmup import packet
        from megatron.core.num_microbatches_calculator import ConstantNumMicroBatchesCalculator
        dp=args.actor_num_nodes*args.actor_num_gpus_per_node//2
        calc=ConstantNumMicroBatchesCalculator(8,args.micro_batch_size,dp,args.decrease_batch_size_if_needed,0)
        records=[dict(tokens=[1,2,3],response_length=1,reward=.5,loss_mask=[1],group_index=i//2,
            turn=i%2,metadata=dict(lane='critic',node_id=i)) for i in range(16)]
        parts=partition_data(args,dict(dp_size=dp,cp_size=1,vpp_size=1,microbatch_group_size_per_vp_stage=1),packet(records))
        print(json.dumps(dict(passed=True,dp_size=dp,initialization_batch=calc.current_running_global_batch_size,
            packet_sizes=[p['global_batch_sizes'] for p in parts])),flush=True)
        return
    try:
        ray.init(address=os.environ['RAY_ADDRESS'],runtime_env={'env_vars':{
            'PYTHONPATH':os.environ['PYTHONPATH'],'NCCL_IB_DISABLE':'1','NCCL_SOCKET_IFNAME':'ens0','GLOO_SOCKET_IFNAME':'ens0'}})
        run(args)
    except BaseException:
        write(Path(args.save).parent/'failed.json',dict(traceback=traceback.format_exc(),time=time.time()));raise


if __name__=='__main__':main()
