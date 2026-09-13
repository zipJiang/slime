"""Run and audit two joint BrowserComp PPO updates with a pretrained critic."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoTokenizer
from slime.observability.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.ray.placement_group import allocate_train_group, create_actor_model, create_rollout_manager
from slime.utils.arguments import parse_args

from audit_checkpoint import audit as audit_checkpoint
from batches import partition_data, put_packets, training_data
from critic_actor import CheckpointCriticActor
from critic_equivalence import compare_scores
from critic_replica import FrozenCriticReplica, ReplicaScorer
from critic_selection import learned_critic_rows
from dataset import load_dataset
from fresh_actor import FreshStartActor
from inference_probe import probe_indices
from pilot_gate import promote
from pilot_preflight import build as build_preflight
from pilot_placement import pinned_placement
from pilot_slime_shim import read_rows
from ppo_initialization import zero_warmup_role_arguments
from provenance import function_sha256
from targets import checkpoint_fields
from value_service import CriticScorer, serve
from warmstart_candidate import digest, require_pilot_candidate


EXPERIMENT=Path(__file__).resolve().parents[1]
ROOT=EXPERIMENT.parents[2]
from pilot_runtime import HARNESS, PROFILE, TRACE, CONTEXT_LIMIT, ACTOR_REPLY_LIMIT, environment_contract, verify_harness
RECIPE_ID='browsecomp-zero-warmup-pilot-v1'


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n');temporary.replace(path)


def custom_args(parser):
    parser.add_argument('--pilot-candidate',type=Path,required=True)
    parser.add_argument('--pilot-context-source',type=Path,required=True)
    parser.add_argument('--pilot-schedule-audit',type=Path,required=True)
    parser.add_argument('--pilot-cases',required=True)
    parser.add_argument('--pilot-retriever-code',required=True)
    parser.add_argument('--pilot-retriever-url',required=True)
    parser.add_argument('--pilot-judge-url',required=True)
    parser.add_argument('--pilot-infrastructure-manifest',type=Path,required=True)
    parser.add_argument('--pilot-pass-tokens',type=int,default=140000)
    parser.add_argument('--pilot-max-pass-attempts',type=int,default=64)
    parser.add_argument('--pilot-search-concurrency',type=int,default=4)
    parser.add_argument('--pilot-prior-strength',type=float,default=1.)
    parser.add_argument('--pilot-critic-lr',type=float,default=5e-6)
    parser.add_argument('--pilot-critic-replica-host',required=True)
    parser.add_argument('--pilot-critic-equivalence-tolerance',type=float,default=.01)
    parser.add_argument('--pilot-seed-namespace',default='browsecomp-zero-warmup-pilot-v1')
    parser.add_argument('--pilot-preflight-only',action='store_true')
    return parser


def engine_version(manager,expected_engines=2):
    engines,*_=ray.get(manager.get_updatable_engines_and_lock.remote())
    versions=[str(v) for v in ray.get([engine.get_weight_version.remote() for engine in engines])]
    if len(versions)!=expected_engines or len(set(versions))!=1 or versions[0]=='None':
        raise RuntimeError(f'Rollout engines disagree on weight version: {versions}')
    return versions[0]


def equivalence_contexts(candidate_path,tokenizer):
    training=Path(candidate_path).resolve().parent
    recipe=json.loads((training/'recipe.json').read_text())
    data=load_dataset(recipe['collection'],json.loads((EXPERIMENT/'data/split.json').read_text()))
    lengths=[len(tokenizer.encode(row['context'],add_special_tokens=False)) for row in data['validation']]
    indices=probe_indices(data['validation'],lengths)
    # Eight questions, each with its root and longest observed fold where present.
    selected=indices[:16]
    return [data['validation'][i]['context'] for i in selected]


def initialization_report(group,cursors,expected_ranks):
    reports=ray.get([actor.audit_optimizer_start.remote() for actor in group._actor_handlers])
    if (cursors!=[0]*expected_ranks or len(reports)!=expected_ranks
            or not all(report['fresh'] for report in reports)):
        raise ValueError('Pilot role did not start at cursor zero with a fresh optimizer')
    return dict(cursors=cursors,optimizers=reports)


def on_policy_reports(run,round_id,ranks):
    reports=[]
    for rank in range(ranks):
        path=run/'on-policy-audit'/f'round-{round_id:04d}-rank-{rank}.json'
        if not path.is_file(): raise ValueError(f'Missing actor on-policy audit rank {rank}')
        report=json.loads(path.read_text())
        if (report['rollout_id']!=round_id or report['mean_abs_difference']>.05
                or report['p99_abs_difference']>.5):
            raise ValueError('Actor on-policy audit exceeded pilot tolerance')
        reports.append(report)
    return reports


def replay_audit(directory):
    log=Path(directory)/'target-replay-audit.log'
    command=[str(ROOT/'rollout-controller/.venv/bin/python'),
        str(EXPERIMENT/'scripts/pilot_audit_batch.py'),str(directory)]
    with log.open('x') as stream:
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,
            cwd=HARNESS,env=dict(os.environ,PYTHONPATH=str(HARNESS)))
    return json.loads((Path(directory)/'target-replay-audit.json').read_text())


def train(args):
    verify_harness()
    if TRACE and (args.seq_length < CONTEXT_LIMIT or args.sglang_context_length < CONTEXT_LIMIT
            or args.rollout_max_response_len < ACTOR_REPLY_LIMIT):
        raise ValueError('TRACE requires 96K trainer/server context and 16384 actor replies')
    configure_logger();run=Path(args.save).parent;run.mkdir(parents=True,exist_ok=True)
    training_ranks=args.actor_num_nodes*args.actor_num_gpus_per_node
    if (training_ranks not in (2,4) or args.tensor_model_parallel_size!=2
            or args.pipeline_model_parallel_size!=1 or args.context_parallel_size!=1):
        raise ValueError('Pilot requires two or four ranks with TP=2, PP=1, CP=1')
    if (run/'recipe.json').exists(): raise ValueError('Use a fresh output for every pilot attempt')
    if (not args.use_critic or not args.offload_train or args.release_train
            or args.normalize_advantages or args.calculate_per_token_loss):
        raise ValueError('Pilot requires persistent offloaded native PPO and fixed prepared weighting')
    if (args.num_rollout!=2 or args.num_critic_only_steps!=0 or args.rollout_batch_size!=6
            or args.global_batch_size!=6 or args.n_samples_per_prompt!=1
            or args.num_steps_per_rollout!=1):
        raise ValueError('Pilot is exactly two zero-warmup six-question joint updates')
    require_pilot_candidate(args.pilot_candidate)
    actor_args,critic_args,lineage=zero_warmup_role_arguments(
        args,args.pilot_candidate,args.pilot_context_source)
    preflight=build_preflight(candidate_path=args.pilot_candidate,
        context_source=args.pilot_context_source,
        schedule_audit=args.pilot_schedule_audit,cases=args.pilot_cases,
        retriever_code=args.pilot_retriever_code,base_actor=args.load,
        updates=args.num_rollout,batch_size=args.rollout_batch_size,
        critic_only_steps=args.num_critic_only_steps,
        train_gpus=args.actor_num_nodes*args.actor_num_gpus_per_node,
        rollout_gpus=args.rollout_num_gpus,
        critic_replica_host=args.pilot_critic_replica_host,
        retriever_url=args.pilot_retriever_url,judge_url=args.pilot_judge_url)
    preflight_path=run/'preflight.json'
    if preflight_path.exists() and json.loads(preflight_path.read_text())!=preflight:
        raise ValueError('GPU pilot arguments differ from the completed CPU preflight')
    write(preflight_path,preflight)
    actor_args.num_gpus_per_node=critic_args.num_gpus_per_node=args.actor_num_gpus_per_node
    critic_args.save=str(run/'critic');critic_args.save_hf=None;critic_args.lr=args.pilot_critic_lr
    critic_args.use_kl_loss=False;critic_args.kl_coef=0;critic_args.use_opd=False
    critic_args.disable_param_buffers_cpu_backup=False
    critic_args.custom_advantage_function_path=None
    critic_args.rollout_data_postprocess_path=None;critic_args.custom_tis_function_path=None
    write(run/'recipe.json',dict(recipe_id=RECIPE_ID,lineage=lineage,
        environment=environment_contract(),harness=str(HARNESS),
        harness_manifest_sha256=digest(HARNESS/'source-manifest.json'),
        schedule_sha256=digest(args.pilot_schedule_audit),
        arguments={k:v for k,v in vars(args).items() if 'key' not in k.lower()},
        scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in Path(__file__).parent.glob('*.py')},
        driver_step=f"{os.environ.get('SLURM_JOB_ID')}.{os.environ.get('SLURM_STEP_ID')}"))
    archive=run/'runtime-sources';archive.mkdir()
    for source in Path(__file__).parent.glob('*.py'): (archive/source.name).write_bytes(source.read_bytes())
    if args.pilot_preflight_only:
        return
    pgs=pinned_placement(args);init_tracking(args)
    actor_args.wandb_run_id=critic_args.wandb_run_id=getattr(args,'wandb_run_id',None)
    manager,_=create_rollout_manager(args,pgs['rollout'])
    actor,actor_cursors=create_actor_model(actor_args,pgs,manager,actor_cls=FreshStartActor)
    critic=allocate_train_group(critic_args,args.actor_num_nodes,args.actor_num_gpus_per_node,
        pgs['critic'],role='critic',actor_cls=CheckpointCriticActor)
    critic_cursors=critic.create(rollout_manager=manager)
    initialization=dict(actor=dict(load=str(Path(actor_args.load).resolve()),
        **initialization_report(actor,actor_cursors,training_ranks)),
        critic=dict(load=str(Path(critic_args.load).resolve()),ckpt_step=critic_args.ckpt_step,
        **initialization_report(critic,critic_cursors,training_ranks)))
    actor.update_weights();behavior_version=engine_version(manager)
    tokenizer=AutoTokenizer.from_pretrained(args.hf_checkpoint,local_files_only=True)
    sentinel=tokenizer.eos_token_id
    if not isinstance(sentinel,int): raise ValueError('Expected one EOS sentinel token')
    parallel=dict(dp_size=training_ranks//2,cp_size=1,vpp_size=1,microbatch_group_size_per_vp_stage=1)
    native=CriticScorer(critic._actor_handlers,critic_args,parallel,tokenizer,sentinel)
    pg,bundle=pgs['critic_inference']
    replica=ray.remote(num_gpus=1,num_cpus=1)(FrozenCriticReplica).options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg,
            placement_group_bundle_index=bundle)).remote(args.hf_checkpoint,args.seq_length)
    scorer=ReplicaScorer(replica);service=serve(scorer)
    value_url=f'http://{ray.util.get_node_ip_address()}:{service.server_port}'
    contexts=equivalence_contexts(args.pilot_candidate,tokenizer)
    publications=[];previous_snapshot=None
    def publish(completed):
        nonlocal previous_snapshot
        version=f'critic-{completed:04d}';snapshot=run/'critic-snapshots'/version
        exports=ray.get([a.export_snapshot.remote(str(snapshot),version) for a in critic._actor_handlers])
        if (len(exports)!=training_ranks or sorted(int(report['rank']) for report in exports)!=list(range(training_ranks))
                or any(report['version']!=version for report in exports)):
            raise RuntimeError('Critic snapshot did not involve every native rank')
        published=ray.get(replica.publish.remote(str(snapshot),version))
        native.begin(version)
        try: expected=native.score(contexts,version)
        finally: native.end()
        scorer.begin(version)
        try:
            actual=scorer.score(contexts,version);repeated=scorer.score(contexts,version)
        finally: scorer.end()
        comparison=compare_scores(expected,actual,repeated,version=version,count=len(contexts),
            tolerance=args.pilot_critic_equivalence_tolerance)
        report=dict(version=version,**comparison,exports=exports,publication=published,
            context_sha256=[hashlib.sha256(c.encode()).hexdigest() for c in contexts])
        write(run/'critic-publication'/f'{version}.json',report)
        if not comparison['passed']: raise ValueError('Portable critic publication differs from native')
        if previous_snapshot is not None: shutil.rmtree(previous_snapshot)
        previous_snapshot=snapshot;publications.append(report)
    publish(0)
    batches=[]
    try:
        for round_id in range(2):
            frozen=dict(rollout_id=round_id,value_version=f'critic-{round_id:04d}',
                policy_version=f'actor-{round_id:04d}',server_weight_version=behavior_version,
                value_url=value_url,seed_namespace=f'{args.pilot_seed_namespace}/{round_id:04d}',
                recipe_id=RECIPE_ID,behavior_round=round_id,execution='sequential',
                estimator='direct_branch_td')
            write(run/'collection-freeze.json',frozen)
            scorer.begin(frozen['value_version'])
            try: actor_refs=ray.get(manager.generate.remote(round_id))
            finally: scorer.end()
            if engine_version(manager)!=frozen['server_weight_version']:
                raise RuntimeError('Actor changed during pilot collection')
            directory=run/'pilot-rollouts'/f'train-{round_id:04d}'
            batch_lineage=dict(collection_round=round_id,behavior_round=round_id,
                learner_round=round_id,actor_lag=0,critic_lag=0,
                denominator='stored_behavior_logprobs',execution='sequential')
            write(directory/'training-lineage.json',batch_lineage)
            contract=json.loads((directory/'contract.json').read_text())
            if any(contract[key]!=frozen[key] for key in
                    ('recipe_id','policy_version','server_weight_version','value_version')):
                raise ValueError('Pilot collector contract differs from frozen versions')
            replay=replay_audit(directory)
            critic_rows=[]
            for group in range(args.rollout_batch_size):
                critic_rows.extend(read_rows(directory/f'group-{group:03d}.critic.jsonl.gz'))
            if any(row['metadata']['value_version']!=frozen['value_version'] or
                   row['metadata'].get('target_source')!='direct_branch_mean' for row in critic_rows):
                raise ValueError('Stale or wrong pilot critic target')
            critic_rows,supervision=learned_critic_rows(critic_rows)
            write(directory/'critic-supervision.json',supervision)
            prepared=[checkpoint_fields(row,tokenizer,sentinel_token_id=sentinel,
                max_sequence_length=args.seq_length,warmup=False) for row in critic_rows]
            critic_packet=training_data(prepared,lane='critic',
                expected_groups=range(args.rollout_batch_size))
            critic_refs=put_packets(partition_data(critic_args,parallel,critic_packet))
            ray.get(critic.async_train(round_id,critic_refs))
            ray.get(actor.async_train(round_id,actor_refs))
            policy_reports=on_policy_reports(run,round_id,len(actor._actor_handlers))
            previous=behavior_version;actor.update_weights();behavior_version=engine_version(manager)
            if behavior_version==previous: raise RuntimeError('Actor serving version did not advance')
            publish(round_id+1)
            summary=json.loads((directory/'summary.json').read_text())
            terminals=sum(row['terminal_count'] for row in summary['results'])
            successes=sum(row['terminal_correct'] for row in summary['results'])
            batch=dict(target_replay_passed=replay['passed'],on_policy_passed=True,
                context_contract_passed=replay['context_contract_passed'],
                infrastructure_failures=replay['infrastructure_failures'],
                terminals=terminals,successes=successes,on_policy=policy_reports,
                critic_supervision=supervision)
            batches.append(batch);write(directory/'training-complete.json',batch)
            write(run/'status.json',dict(stage='zero_warmup_pilot',completed_joint_updates=round_id+1,
                server_weight_version=behavior_version))
        actor.save_model(1,force_sync=True);critic.save_model(1,force_sync=True)
    except Exception:
        write(run/'failed.json',dict(traceback=traceback.format_exc(),completed_joint_updates=len(batches)))
        raise
    finally:
        service.shutdown();service.server_close()
    ray.get(manager.dispose.remote());ray.kill(replica);actor.release();critic.release()
    actor_audit=audit_checkpoint(run/'actor/iter_0000001',2,'actor')
    critic_audit=audit_checkpoint(run/'critic/iter_0000001',2,'critic')
    terminals=sum(batch['terminals'] for batch in batches);successes=sum(batch['successes'] for batch in batches)
    rejection=[]
    if successes==0: rejection.append('No successful terminal was observed')
    if successes==terminals: rejection.append('No failed terminal was observed')
    pilot=dict(schema='browsecomp-zero-warmup-pilot-v1',
        training_ranks=training_ranks,environment=environment_contract(),
        collection_context_function_sha256=function_sha256(EXPERIMENT/'scripts/pilot_runtime.py','context'),
        candidate_sha256=digest(args.pilot_candidate),
        context_function_sha256=function_sha256(args.pilot_context_source,'context'),
        num_critic_only_steps=0,start_rollout_id=0,completed_joint_updates=2,
        initialization=initialization,batches=batches,
        critic_publications=publications,
        checkpoint=dict(actor=actor_audit,critic=critic_audit),
        scientific_rejection=bool(rejection),scientific_rejection_reasons=rejection)
    write(run/'pilot.json',pilot)
    if not rejection:
        write(run/'long-run-warmstart.json',promote(args.pilot_candidate,run/'pilot.json'))
    write(run/'completed.json',dict(joint_updates=2,scientific_rejection_reasons=rejection,
        long_run_authorized=not rejection))
    finish_tracking(args)


def main():
    args=parse_args(custom_args)
    run=Path(args.save).parent
    try:
        if args.pilot_preflight_only:
            train(args);return
        ray.init(address=os.environ['RAY_ADDRESS'],runtime_env={'env_vars':{
            'GLOO_SOCKET_IFNAME':'ens0','NCCL_SOCKET_IFNAME':'ens0','NCCL_IB_DISABLE':'1',
            'PYTHONPATH':os.environ['PYTHONPATH'], 'BROWSECOMP_PROFILE':PROFILE}})
        train(args)
    except Exception:
        if not (run/'failed.json').exists():
            write(run/'failed.json',dict(traceback=traceback.format_exc(),completed_joint_updates=0))
        raise


if __name__=='__main__':
    main()
