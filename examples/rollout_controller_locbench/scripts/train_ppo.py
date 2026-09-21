"""LocBench PPO: frozen one-batch overlap, measured filtering and paired recovery."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoTokenizer
from slime.observability.logging_utils import configure_logger,finish_tracking,init_tracking
from slime.ray.placement_group import allocate_train_group,create_actor_model,create_rollout_manager
from slime.utils.arguments import parse_args
from runtime_active import EXPERIMENT,HARNESS,MODEL,CONTEXT_LIMIT,REPLY_LIMIT,contract,digest,verify_harness
import ppo_runtime
from train_critic import write
from ppo_protocol import RECIPE_ID,initial_roles,resume_roles,lineage,boundary,prefetch,candidate
from prepare_ppo import prepare
from ppo_placement import pinned_placement
from ppo_collection import Collection,read_rows,audit_collection,materialize,router_address,replay_benchmark,PY
from ppo_batches import actor_packets
from batches import training_data,partition_data,put_packets
from targets import checkpoint_fields
from critic_selection import learned_critic_rows
from critic_actor import CheckpointCriticActor
from fresh_actor import FreshStartActor
from critic_replica import FrozenCriticReplica,ReplicaScorer
from critic_equivalence import compare_scores
from rollout_version import engine_version
from value_service import CriticScorer,serve
from audit_checkpoint import audit as audit_checkpoint
from optimizer_restore import inspect_optimizer,validate_restore


def custom_args(parser):
    parser.add_argument('--loc-candidate',type=Path,required=True)
    parser.add_argument('--loc-critic-replica-host',required=True)
    parser.add_argument('--loc-critic-lr',type=float,default=1e-6)
    parser.add_argument('--loc-pass-tokens',type=int,default=32768)
    parser.add_argument('--loc-max-pass-attempts',type=int,default=32)
    parser.add_argument('--loc-search-concurrency',type=int,default=4)
    parser.add_argument('--loc-prior-strength',type=float,default=1.)
    parser.add_argument('--loc-seed-namespace',default=RECIPE_ID)
    parser.add_argument('--loc-resume-run',type=Path)
    parser.add_argument('--loc-allow-dp-reshard',action='store_true')
    parser.add_argument('--loc-stop-after-round',type=int)
    parser.add_argument('--loc-preflight-only',action='store_true')
    parser.add_argument('--loc-arguments-only',action='store_true')
    parser.add_argument('--loc-dataset-preflight-only',action='store_true')
    parser.add_argument('--loc-benchmark-only',action='store_true')
    parser.add_argument('--loc-benchmark-source',type=Path)
    parser.add_argument('--loc-efficiency-plan',type=Path)
    parser.add_argument('--loc-memory-preflight-source',type=Path)
    parser.add_argument('--loc-memory-stress-source',type=Path)
    parser.add_argument('--loc-collection-profile',choices=ppo_runtime.COLLECTION_PROFILES,default='original')
    parser.add_argument('--loc-allow-profile-transition',action='store_true')
    parser.add_argument('--loc-profile-comparison-only',action='store_true')
    return parser


def validate(args):
    if args.rollout_temperature!=1 or args.rollout_top_p!=1 or args.rollout_top_k!=-1:
        raise ValueError('Native likelihoods use per-span temperatures and full-softmax normalization')
    if (args.seq_length!=CONTEXT_LIMIT or args.sglang_context_length<CONTEXT_LIMIT
        or args.rollout_max_response_len<REPLY_LIMIT or args.tensor_model_parallel_size!=2
        or args.pipeline_model_parallel_size!=1 or args.context_parallel_size!=1):
        raise ValueError('PPO requires the tested 98K context and native TP2 model sharding')
    if (args.actor_num_nodes*args.actor_num_gpus_per_node not in (2,4)
        or args.rollout_num_gpus_per_engine!=1 or args.rollout_num_gpus<2
        or not args.use_critic or not args.offload_train or args.release_train
        or args.normalize_advantages or args.calculate_per_token_loss
        or args.num_critic_only_steps!=0 or args.rollout_batch_size!=args.global_batch_size
        or args.n_samples_per_prompt!=1 or args.num_steps_per_rollout!=1
        or args.num_rollout<2 or args.save_interval<1 or args.lr!=1e-6 or args.loc_critic_lr!=1e-6):
        raise ValueError('Unsupported LocBench native PPO recipe')


def fresh_report(group,cursors,ranks):
    reports=ray.get([h.audit_optimizer_start.remote() for h in group._actor_handlers])
    if cursors!=[0]*ranks or len(reports)!=ranks or not all(r['fresh'] for r in reports):
        raise ValueError('Native role did not start with fresh optimizer/cursor')
    return dict(cursors=cursors,optimizers=reports)


def train(args):
    configure_logger();verify_harness();validate(args)
    run=Path(args.save).parent;run.mkdir(parents=True,exist_ok=True)
    if (run/'recipe.json').exists():raise ValueError('Each PPO attempt, including resume, needs a new directory')
    data=prepare(args.loc_candidate,run/'data',args.num_rollout,args.rollout_batch_size)
    plan=json.loads(Path(data['audit']).read_text());schedule_hash=digest(data['audit'])
    if Path(args.prompt_data).resolve()!=Path(data['schedule']).resolve():raise ValueError('Wrong question schedule')
    if args.loc_benchmark_only and (args.loc_resume_run or args.loc_efficiency_plan):
        raise ValueError('Benchmark mode must start from fresh base/critic weights')
    if args.loc_memory_stress_source and not args.loc_memory_preflight_source:
        raise ValueError('Historical shape stress is restricted to isolated memory qualification')
    if args.loc_memory_preflight_source and (not args.loc_resume_run or args.loc_benchmark_only):
        raise ValueError('Memory qualification requires an isolated restored run')
    if args.loc_profile_comparison_only and (not args.loc_resume_run or args.loc_benchmark_only or args.loc_memory_preflight_source or args.loc_collection_profile!='original'):
        raise ValueError('Profile comparison requires isolated original-profile restoration')
    if args.loc_collection_profile not in ('original','compact24-reply8-read60','robust-null-v3') and not args.loc_resume_run:
        raise ValueError('New profile currently requires explicit paired-checkpoint transition')
    if args.loc_resume_run:
        actor_args,critic_args,origin,resume=resume_roles(args,schedule_sha256=schedule_hash)
    else:
        actor_args,critic_args,origin=initial_roles(args,args.loc_candidate);resume=None
    start=resume['start_rollout_id'] if resume else 0
    last=min(args.num_rollout-1,args.loc_stop_after_round) if args.loc_stop_after_round is not None else args.num_rollout-1
    if last<start:raise ValueError('Stop boundary precedes the restored cursor')
    args.start_rollout_id=start;ranks=args.actor_num_nodes*args.actor_num_gpus_per_node
    for role in (actor_args,critic_args):role.num_gpus_per_node=args.actor_num_gpus_per_node
    actor_args.save_hf=str(run/'hf/iter_{rollout_id:07d}')
    critic_args.save=str(run/'critic');critic_args.save_hf=None;critic_args.lr=args.loc_critic_lr
    critic_args.use_kl_loss=False;critic_args.kl_coef=0;critic_args.use_opd=False
    critic_args.disable_param_buffers_cpu_backup=False;critic_args.custom_advantage_function_path=None
    critic_args.rollout_data_postprocess_path=None;critic_args.custom_tis_function_path=None
    recipe=dict(recipe_id=RECIPE_ID,environment=ppo_runtime.environment(args.loc_collection_profile),
        collection_profile_sha256=digest(ppo_runtime.__file__),lineage=origin,resume=resume,
        native_backend=json.loads((EXPERIMENT/'operations/native-temperature-patch.json').read_text()),
        candidate_sha256=digest(args.loc_candidate),schedule_sha256=schedule_hash,
        arguments={k:v for k,v in vars(args).items() if not (k.endswith('_api_key') or k in ('api_key','hf_token','access_token'))},
        sources={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        likelihood_contract=dict(normalization='full-softmax at each span temperature',task_temperature=.6,fold_temperature=.7),
        efficiency=resume['efficiency'] if resume else None)
    write(run/'recipe.json',recipe)
    archive=run/'runtime-sources';archive.mkdir()
    for path in Path(__file__).parent.glob('*.py'):shutil.copyfile(path,archive/path.name)
    if args.loc_preflight_only:return
    write(run/'status.json',dict(stage='initializing-native-models',time=time.time()))
    pgs=pinned_placement(args);init_tracking(args)
    actor_args.wandb_run_id=critic_args.wandb_run_id=getattr(args,'wandb_run_id',None)
    manager,_=create_rollout_manager(args,pgs['rollout'])
    def create_roles(a_args,c_args,*,restored=False,tag='initialization'):
        actor,cursors=create_actor_model(a_args,pgs,manager,actor_cls=FreshStartActor)
        critic=allocate_train_group(c_args,args.actor_num_nodes,args.actor_num_gpus_per_node,pgs['critic'],
            role='critic',actor_cls=CheckpointCriticActor)
        critic_cursors=critic.create(rollout_manager=manager)
        if restored:
            if cursors!=[start]*ranks or critic_cursors!=[start]*ranks:raise ValueError('Native resume cursor mismatch')
            reports={role:ray.get([h.__ray_call__.remote(inspect_optimizer) for h in model._actor_handlers])
                for role,model in [('actor',actor),('critic',critic)]}
            write(run/f'{tag}.json',validate_restore(resume,reports,args.global_batch_size,world_size=ranks))
        else:
            write(run/f'{tag}.json',dict(actor=fresh_report(actor,cursors,ranks),critic=fresh_report(critic,critic_cursors,ranks)))
            from actor_initialization import validate_reports
            reports=ray.get([h.audit_initial_models.remote() for h in actor._actor_handlers])
            write(run/f'{tag}-model-readback.json',validate_reports(reports,ranks))
        return actor,critic
    actor,critic=create_roles(actor_args,critic_args,restored=bool(resume))
    endpoint=ray.get(manager.__ray_call__.remote(router_address))
    args.sglang_router_ip=endpoint['host'];args.sglang_router_port=endpoint['port']
    recipe['router']=endpoint;write(run/'recipe.json',recipe)
    actor_publication_started=time.monotonic()
    actor.update_weights();behavior_version=engine_version(manager,expected_engines=args.rollout_num_gpus)
    initial_actor_publication_seconds=time.monotonic()-actor_publication_started
    behavior_round=start
    tokenizer=AutoTokenizer.from_pretrained(args.hf_checkpoint,local_files_only=True);sentinel=tokenizer.eos_token_id
    parallel=dict(dp_size=ranks//2,cp_size=1,vpp_size=1,microbatch_group_size_per_vp_stage=1)
    pg,bundle=pgs['critic_inference']
    replica=ray.remote(num_gpus=1,num_cpus=1)(FrozenCriticReplica).options(scheduling_strategy=
        PlacementGroupSchedulingStrategy(placement_group=pg,placement_group_bundle_index=bundle)).remote(args.hf_checkpoint,CONTEXT_LIMIT)
    scorer=ReplicaScorer(replica);service=serve(scorer)
    value_url=f'http://{ray.util.get_node_ip_address()}:{service.server_port}'
    critic_recipe=json.loads((args.loc_candidate.parent/'recipe.json').read_text())
    from warmup_data import load
    from inference_probe import probe_indices
    validation=load(Path(critic_recipe['arguments']['loc_collection']))['validation']
    indices=probe_indices(validation,[len(tokenizer.encode(r['context'],add_special_tokens=False)) for r in validation])[:16]
    contexts=[validation[i]['context'] for i in indices]
    previous_snapshot=None;publication_index=0;pending=None
    def publish(completed):
        nonlocal previous_snapshot,publication_index
        began=time.monotonic();version=f'critic-{completed:04d}'
        snapshot=run/'critic-snapshots'/f'{version}-{publication_index:04d}';publication_index+=1
        exports=ray.get([h.export_snapshot.remote(str(snapshot),version) for h in critic._actor_handlers])
        if len(exports)!=ranks or {int(r['rank']) for r in exports}!=set(range(ranks)) or any(r['version']!=version for r in exports):
            raise ValueError('Missing critic export rank/version')
        published=ray.get(replica.publish.remote(str(snapshot),version))
        native=CriticScorer(critic._actor_handlers,critic_args,parallel,tokenizer,sentinel)
        native.begin(version)
        try:expected=native.score(contexts,version)
        finally:native.end()
        scorer.begin(version)
        try:actual=scorer.score(contexts,version);repeated=scorer.score(contexts,version)
        finally:scorer.end()
        comparison=compare_scores(expected,actual,repeated,version=version,count=len(contexts),tolerance=.03,mean_tolerance=.005)
        write(run/'critic-publication'/f'{snapshot.name}.json',dict(**comparison,exports=exports,publication=published,
            native=expected,replica=actual,context_sha256=[hashlib.sha256(c.encode()).hexdigest() for c in contexts]))
        if not comparison['passed']:raise ValueError('Published critic differs from native trainer')
        if previous_snapshot is not None:shutil.rmtree(previous_snapshot)
        previous_snapshot=snapshot
        return time.monotonic()-began
    def begin(round_id,cutoff,pass_tokens):
        nonlocal pending
        frozen=dict(rollout_id=round_id,behavior_round=behavior_round,policy_version=f'actor-{behavior_round:04d}',
            server_weight_version=behavior_version,value_version=f'critic-{behavior_round:04d}',value_url=value_url,
            seed_namespace=f'{args.loc_seed_namespace}/{round_id:04d}')
        scorer.begin(frozen['value_version'])
        try:pending=Collection(args,run,round_id,plan['batches'][round_id],frozen,cutoff=cutoff,pass_tokens=pass_tokens)
        except BaseException:scorer.end();raise
        return pending
    def finish():
        nonlocal pending
        try:
            result=pending.finish()
            if engine_version(manager,expected_engines=args.rollout_num_gpus)!=result['frozen']['server_weight_version']:
                raise ValueError('Actor changed during collection')
            return result
        finally:scorer.end();pending=None
    def critic_batch(directory):
        rows=[r for g in range(args.rollout_batch_size) for r in read_rows(directory/f'group-{g:03d}.critic.jsonl.gz')]
        rows,supervision=learned_critic_rows(rows)
        records=[checkpoint_fields(r,tokenizer,sentinel_token_id=sentinel,max_sequence_length=CONTEXT_LIMIT) for r in rows]
        packets=partition_data(critic_args,parallel,training_data(records,lane='critic',expected_groups=range(args.rollout_batch_size)))
        return put_packets(packets),supervision
    def actor_batch(records):
        packets,selection=actor_packets(actor_args,parallel,records,expected_groups=range(args.rollout_batch_size))
        return (put_packets(packets) if packets is not None else None),selection
    actor_updates=resume['actor_updates'] if resume else 0
    critic_updates=resume['critic_updates'] if resume else 0
    completed=start;ready=None
    try:
        write(run/'status.json',dict(stage='publishing-initial-critic',time=time.time()))
        initial_critic_publication_seconds=publish(start)
        if args.loc_profile_comparison_only:
            from profile_comparison import compare
            compare(args,run,manager,behavior_version,start)
            return
        if args.loc_memory_preflight_source:
            from memory_qualification import qualify
            qualify(args,run,resume,plan,schedule_hash,actor,critic,actor_batch,critic_batch,ranks)
            return
        if not resume and args.loc_efficiency_plan is None:
            # One measured optimizer step per candidate, restored from original
            # actor and critic weights before actual PPO. No benchmark publication.
            write(run/'status.json',dict(stage='collecting-efficiency-batch',time=time.time()))
            if args.loc_benchmark_source:
                ready=replay_benchmark(args,run,plan,schedule_hash=schedule_hash,
                    behavior_version=behavior_version,value_url=value_url)
            else:begin(0,None,args.loc_pass_tokens);ready=finish()
            directory=ready['directory']
            audit_collection(directory);write(directory/'training-lineage.json',lineage(0,0,overlap=False))
            subprocess.run([PY,str(EXPERIMENT/'scripts/efficiency.py'),'--collection',str(directory),
                '--output',str(run/'efficiency-analysis.json')],check=True,env=dict(os.environ,PYTHONPATH=str(HARNESS)))
            analysis=json.loads((run/'efficiency-analysis.json').read_text());baseline=analysis['candidates'][0]
            feasible=[c for c in analysis['candidates'][1:] if c['edges'] and c['retained_mass_fraction'] is not None
                and c['retained_mass_fraction']>=.95 and c['sequence_tokens']<baseline['sequence_tokens']
                and all(c['by_kind'][k]['weighted_absolute_advantage_mass']>=.9*v['weighted_absolute_advantage_mass']
                    for k,v in baseline['by_kind'].items())]
            crop=min(feasible,key=lambda c:c['sequence_tokens']) if feasible else None
            timings=[]
            for index,cutoff in enumerate([None]+([crop['cutoff']] if crop else [])):
                write(run/'status.json',dict(stage='benchmarking-native-cutoff',cutoff=cutoff,time=time.time()))
                records,export=materialize(directory,cutoff);a_refs,selection=actor_batch(records)
                if a_refs is None:raise ValueError('Benchmark candidate has no actor supervision')
                from memory_qualification import reset_peak,memory_report
                c_refs,_=critic_batch(directory)
                ray.get([h.__ray_call__.remote(reset_peak) for h in critic._actor_handlers])
                began=time.monotonic()
                ray.get(critic.async_train(0,c_refs));critic_seconds=time.monotonic()-began
                critic_memory=ray.get([h.__ray_call__.remote(memory_report) for h in critic._actor_handlers])
                ray.get([h.__ray_call__.remote(reset_peak) for h in actor._actor_handlers])
                began=time.monotonic();ray.get(actor.async_train(0,a_refs));actor_seconds=time.monotonic()-began
                actor_memory=ray.get([h.__ray_call__.remote(memory_report) for h in actor._actor_handlers])
                memory=dict(actor=actor_memory,critic=critic_memory)
                headroom=min(r['total_bytes']-r['peak_reserved_bytes'] for role in memory.values() for r in role)
                write(run/'benchmark-memory'/f'candidate-{index:02d}.json',dict(memory=memory,
                    min_allocator_headroom_bytes=headroom,cutoff=cutoff))
                audit_reports=[json.loads((run/'on-policy-audit'/f'round-0000-rank-{r}.json').read_text()) for r in range(ranks)]
                if any(not r['passed'] for r in audit_reports):raise ValueError('Benchmark behavior audit failed')
                timings.append(dict(cutoff=cutoff,actor_seconds=actor_seconds,critic_seconds=critic_seconds,
                    collection_seconds=ready['seconds'],collection_replayed=ready.get('replayed',False),
                    collection_rollout_gpus=ready.get('collection_rollout_gpus',args.rollout_num_gpus),overlapped_seconds=max(ready['seconds'],actor_seconds+critic_seconds),
                    export=export,selection=selection,behavior_audits=audit_reports,
                    min_allocator_headroom_bytes=headroom))
                actor.release();critic.release()
                actor,critic=create_roles(actor_args,critic_args,tag=f'benchmark-reset-{index}')
            from efficiency import choose_timing
            # Leave room for CUDA contexts, the other offloaded role, and
            # batch variation; this measured batch alone is not a hard cap.
            minimum_headroom=12*2**30 if args.loc_collection_profile=='compact24-reply8-read60' else 0
            chosen=choose_timing(timings,minimum_headroom_bytes=minimum_headroom)
            efficiency=dict(min_abs_advantage=chosen['cutoff'],pass_tokens=args.loc_pass_tokens,
                timings=timings,training_gpus=ranks,rollout_gpus=args.rollout_num_gpus,critic_inference_gpus=1,
                initial_publication_seconds=dict(actor=initial_actor_publication_seconds,critic=initial_critic_publication_seconds),
                mass_floor=.95,per_kind_mass_floor=.9,
                minimum_allocator_headroom_bytes=minimum_headroom,
                selection='Least filtering within 5% of measured feasible overlap cycle; publication excluded because it is shared',
                limitations='One real batch at one token budget/topology; remaining budget/allocation comparison required')
            recipe['efficiency']=efficiency;write(run/'recipe.json',recipe);write(run/'efficiency-selection.json',efficiency)
            if args.loc_benchmark_only:
                write(run/'benchmark-complete.json',dict(passed=True,efficiency=efficiency,
                    candidate_sha256=digest(args.loc_candidate),schedule_sha256=schedule_hash,environment=ppo_runtime.environment(args.loc_collection_profile),
                    actor_updates_committed=0,critic_updates_committed=0))
                return
            actor.update_weights();behavior_version=engine_version(manager,expected_engines=args.rollout_num_gpus)
            # Fresh actor recreation restarts its publication counter. Verify the
            # ready batch still names that same immutable base behavior.
            if behavior_version!=ready['frozen']['server_weight_version']:raise ValueError('Benchmark reset changed behavior identity')
            publish(0)
        elif resume:efficiency=resume['efficiency']
        else:
            efficiency=json.loads(args.loc_efficiency_plan.read_text())
            if (not efficiency.get('passed') or efficiency['environment']!=ppo_runtime.environment(args.loc_collection_profile)
                or efficiency['candidate_sha256']!=digest(args.loc_candidate) or efficiency['schedule_sha256']!=schedule_hash
                or efficiency['training_gpus']!=ranks or efficiency['rollout_gpus']!=args.rollout_num_gpus):
                raise ValueError('Efficiency plan differs from this native experiment')
            for name,sha in efficiency['benchmark_evidence_sha256'].items():
                if digest(name)!=sha:raise ValueError('Efficiency benchmark evidence changed')
            if args.loc_collection_profile=='compact24-reply8-read60' and (
                efficiency.get('minimum_allocator_headroom_bytes',0)<12*2**30
                or efficiency.get('measured_selection',{}).get('min_allocator_headroom_bytes',0)<12*2**30):
                raise ValueError('Imitation PPO plan lacks measured allocator headroom')
            recipe['efficiency']=efficiency;write(run/'recipe.json',recipe)
        cutoff=efficiency['min_abs_advantage'];pass_tokens=efficiency['pass_tokens']
        for round_id in range(start,last+1):
            began=time.monotonic()
            if ready is None:begin(round_id,cutoff,pass_tokens);ready=finish()
            directory=ready['directory'];frozen=ready['frozen'];collection_seconds=ready['seconds'];summary=ready['summary'];ready=None
            stamp=lineage(round_id,frozen['behavior_round'],overlap=round_id>=2)
            write(directory/'training-lineage.json',stamp)
            if not (directory/'target-replay-audit.json').exists():audit_collection(directory)
            # First benchmark source was unfiltered; re-export chosen records.
            if not resume and args.loc_efficiency_plan is None and round_id==0:
                output=directory/'exports'/('none' if cutoff is None else str(cutoff))
                records=[r for g in range(args.rollout_batch_size) for r in read_rows(output/f'group-{g:03d}.actor.jsonl.gz')]
            else:records=[r for g in range(args.rollout_batch_size) for r in read_rows(directory/f'group-{g:03d}.actor.jsonl.gz')]
            a_refs,selection=actor_batch(records);c_refs,supervision=critic_batch(directory)
            stop=(run/'STOP').exists();save_now=(round_id==start or boundary(round_id,save_interval=args.save_interval,last_round=last,stop=stop))
            if prefetch(round_id,last_round=last,save_now=save_now,stop=stop):begin(round_id+1,cutoff,pass_tokens)
            from memory_qualification import reset_peak,memory_report
            ray.get([h.__ray_call__.remote(reset_peak) for h in critic._actor_handlers])
            t0=time.monotonic();ray.get(critic.async_train(round_id,c_refs));t1=time.monotonic();critic_updates+=1
            write(run/'training-memory'/f'round-{round_id:04d}-critic.json',
                ray.get([h.__ray_call__.remote(memory_report) for h in critic._actor_handlers]))
            if a_refs is not None:
                ray.get([h.__ray_call__.remote(reset_peak) for h in actor._actor_handlers])
                ray.get(actor.async_train(round_id,a_refs));actor_updates+=1
                write(run/'training-memory'/f'round-{round_id:04d}-actor.json',
                    ray.get([h.__ray_call__.remote(memory_report) for h in actor._actor_handlers]))
                reports=[json.loads((run/'on-policy-audit'/f'round-{round_id:04d}-rank-{r}.json').read_text()) for r in range(ranks)]
                if any(not r['passed'] or r['lineage']!=stamp for r in reports):raise ValueError('PPO behavior audit failed')
            t2=time.monotonic()
            if pending is not None:ready=finish()
            drained=time.monotonic();completed=round_id+1
            if save_now:
                if ready is not None:raise ValueError('Cannot checkpoint untrained prefetched questions')
                actor.save_model(round_id,force_sync=True);critic.save_model(round_id,force_sync=True)
                for role,updates in [('actor',actor_updates),('critic',critic_updates)]:
                    audit_checkpoint(run/role/f'iter_{round_id:07d}',updates,role)
                cursor=run/'checkpoints'/f'cursor-{round_id:04d}.json'
                write(cursor,dict(batches_completed=completed,questions_consumed=completed*args.rollout_batch_size,
                    schedule_sha256=schedule_hash,prefetched_untrained_questions=0))
                evidence=[run/role/f'iter_{round_id:07d}-readback.json' for role in ('actor','critic')]+[cursor]
                write(run/'checkpoints'/f'round-{round_id:04d}.json',dict(passed=True,iteration=round_id,batches_completed=completed,
                    actor_updates=actor_updates,critic_updates=critic_updates,
                    evidence_sha256={str(p.relative_to(run)):digest(p) for p in evidence}))
            publication_started=time.monotonic();previous=behavior_version
            actor.update_weights();behavior_version=engine_version(manager,expected_engines=args.rollout_num_gpus)
            if previous==behavior_version:raise ValueError('Actor publication did not advance')
            behavior_round=completed;publish(completed)
            status=dict(stage='ppo',batches_completed=completed,actor_updates=actor_updates,critic_updates=critic_updates,
                lineage=stamp,checkpoint_saved=save_now,prefetched_next=ready is not None,actor_selection=selection,
                critic_supervision=supervision,terminal_mean_recall=sum(r['terminal_correct'] for r in summary['results'])/sum(r['terminal_count'] for r in summary['results']),
                seconds=dict(collection=collection_seconds,critic=t1-t0,actor=t2-t1,prefetch_tail=drained-t2,
                    publication=time.monotonic()-publication_started,total=time.monotonic()-began))
            write(run/'status.json',status);write(directory/'training-complete.json',status)
            if stop:break
        finalizer=EXPERIMENT.parent/'rollout_controller_ppo_balanced_recovery/scripts/finalize_hf.py'
        hf=run/'hf'/f'iter_{completed-1:07d}'
        subprocess.run([os.sys.executable,str(finalizer),'--source',args.hf_checkpoint,'--target',str(hf)],check=True)
        write(run/('completed.json' if completed==args.num_rollout else 'paused.json'),dict(**status,model=str(hf)))
    except BaseException:
        write(run/'failed.json',dict(traceback=traceback.format_exc(),batches_completed=completed,
            actor_updates=actor_updates,critic_updates=critic_updates));raise
    finally:
        if pending is not None:pending.cancel()
        service.shutdown();service.server_close()
        try:ray.get(manager.dispose.remote(),timeout=60)
        except Exception as exc:write(run/'cleanup-warning.json',dict(error=repr(exc)))
        ray.kill(replica);actor.release();critic.release();finish_tracking(args)


if __name__=='__main__':
    args=parse_args(custom_args)
    if args.loc_arguments_only:
        validate(args)
        print(json.dumps(dict(passed=True,scope='Native argument validation; no model or optimizer allocation',
            training_gpus=args.actor_num_nodes*args.actor_num_gpus_per_node,rollout_gpus=args.rollout_num_gpus,
            use_critic=args.use_critic,offload_train=args.offload_train,seq_length=args.seq_length,
            critic_only_steps=args.num_critic_only_steps,global_batch_size=args.global_batch_size,
            actor_logprob_audit=args.rollout_data_postprocess_path)),flush=True)
        raise SystemExit(0)
    if args.loc_dataset_preflight_only:
        validate(args)
        from slime.rollout.data_source import RolloutDataSource
        source=RolloutDataSource(args)
        expected=args.num_rollout*args.rollout_batch_size
        if len(source.dataset)!=expected or source.sample_offset!=0:
            raise ValueError('Native dataset construction changed schedule size or cursor')
        write(Path(args.save).parent/'native-dataset-preflight.json',dict(passed=True,questions=expected,
            cursor=source.sample_offset,scope='Real native Qwen processor and Slime dataset constructor; no GPU model allocation'))
        raise SystemExit(0)
    try:
        if not args.loc_preflight_only:
            ray.init(address=os.environ['RAY_ADDRESS'],runtime_env={'env_vars':{
                'GLOO_SOCKET_IFNAME':'ens0','NCCL_SOCKET_IFNAME':'ens0','NCCL_IB_DISABLE':'1','PYTHONPATH':os.environ['PYTHONPATH']}})
        train(args)
    except BaseException:
        path=Path(args.save).parent/'failed.json'
        if not path.exists():write(path,dict(traceback=traceback.format_exc()))
        raise
