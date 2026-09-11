"""Synchronous Slime PPO with separate controller actor/checkpoint batches."""
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import shutil
import time
import traceback

import ray
from transformers import AutoTokenizer
from slime.observability.logging_utils import configure_logger, init_tracking, finish_tracking
from slime.ray.placement_group import create_actor_model, allocate_train_group, create_rollout_manager
from slime.utils.arguments import parse_args

from batches import partition_data, put_packets, training_data
from critic_actor import CheckpointCriticActor
from placement import pinned_placement
from slime_shim import read_rows
from targets import checkpoint_fields
from value_service import CriticScorer, serve
from resume import role_arguments
from recipe import RECIPE_ID, estimator_for_round, target_source
from replay_initial import verify_initial_batch
from pipeline import BatchStamp, save_boundary, may_prefetch


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str)+'\n')
    temporary.replace(path)


def custom_args(parser):
    parser.add_argument('--ppo-pass-tokens', type=int, default=140000)
    parser.add_argument('--ppo-search-concurrency', type=int, default=4)
    parser.add_argument('--ppo-prior-strength', type=float, default=1.)
    parser.add_argument('--ppo-critic-lr', type=float, default=5e-6)
    parser.add_argument('--ppo-critic-load')
    parser.add_argument('--ppo-replay-initial-batch')
    parser.add_argument('--ppo-execution', choices=['sync', 'overlap'], default='sync')
    parser.add_argument('--ppo-critic-replica-host', default='172.16.203.30')
    parser.add_argument('--ppo-critic-equivalence-contexts')
    parser.add_argument('--ppo-seed-namespace')
    parser.add_argument('--ppo-stop-after-round', type=int)
    parser.add_argument('--ppo-benchmark', action='store_true', help='Defer boundary evaluation to measure training throughput')
    return parser


def engine_version(manager):
    engines, *_ = ray.get(manager.get_updatable_engines_and_lock.remote())
    versions = [str(v) for v in ray.get([e.get_weight_version.remote() for e in engines])]
    if not versions or len(set(versions)) != 1 or versions[0] == 'None':
        raise RuntimeError(f'Rollout engines disagree on weight version: {versions}')
    return versions[0]


def train(args):
    driver_started = time.monotonic()
    configure_logger()
    if not args.use_critic or not args.offload_train or args.release_train:
        raise ValueError('Expected native PPO with train offload and persistent ranks')
    if args.normalize_advantages or args.calculate_per_token_loss:
        raise ValueError('Prepared edge weights require fixed question/edge normalization')
    if args.n_samples_per_prompt != 1 or args.global_batch_size != args.rollout_batch_size or args.num_steps_per_rollout != 1:
        raise ValueError('One search per question, one fresh question batch per optimizer update')
    if args.ppo_replay_initial_batch:
        raise ValueError('This ablation collects fresh warm-up data; replay is disabled')
    run = Path(args.save).parent
    run.mkdir(parents=True, exist_ok=True)
    if (run/'recipe.json').exists():
        raise ValueError('Use a fresh run directory for every attempt/resume')
    actor_args, critic_args, resume = role_arguments(args)
    actor_args.num_gpus_per_node = critic_args.num_gpus_per_node = args.actor_num_gpus_per_node
    if resume is not None:
        args.start_rollout_id = resume['start_rollout_id']
        write_json(run/'resume.json', resume)
    write_json(run/'recipe.json', dict(recipe_id=RECIPE_ID, arguments={k:v for k,v in vars(args).items() if k != 'wandb_key'},
        script_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        driver_step=f"{os.environ.get('SLURM_JOB_ID')}.{os.environ.get('SLURM_STEP_ID')}"))
    archive = run/'runtime-sources'
    archive.mkdir()
    for source in Path(__file__).parent.glob('*.py'):
        (archive/source.name).write_bytes(source.read_bytes())
    pgs = pinned_placement(args)
    init_tracking(args)
    # The primary assigns this ID during initialization, after role arguments
    # were copied for resume preflight. Both native workers must receive it.
    if args.use_wandb and not getattr(args, 'wandb_run_id', None):
        raise RuntimeError('Primary experiment tracking did not initialize')
    actor_args.wandb_run_id = critic_args.wandb_run_id = args.wandb_run_id
    manager, _ = create_rollout_manager(args, pgs['rollout'])
    actor, actor_starts = create_actor_model(actor_args, pgs, manager)
    critic_args.save = str(run/'critic')
    critic_args.save_hf = None
    critic_args.load = args.ppo_critic_load or args.hf_checkpoint
    critic_args.lr = args.ppo_critic_lr
    critic_args.use_kl_loss = False
    critic_args.kl_coef = 0
    critic_args.use_opd = False
    critic_args.disable_param_buffers_cpu_backup = False
    critic_args.custom_advantage_function_path = None
    critic_args.rollout_data_postprocess_path = None
    critic_args.custom_tis_function_path = None
    # A small scalar head starts near probability .5, without saturating sigmoid.
    critic_args.init_method_std = .001
    critic = allocate_train_group(critic_args, args.actor_num_nodes, args.actor_num_gpus_per_node,
        pgs['critic'], role='critic', actor_cls=CheckpointCriticActor)
    critic_starts = critic.create(rollout_manager=manager)
    if len(set(actor_starts)) != 1 or len(set(critic_starts)) != 1:
        raise RuntimeError('Ranks disagree on checkpoint iteration')
    if args.start_rollout_id is None:
        args.start_rollout_id = critic_starts[0]
    if args.ppo_critic_load and critic_starts != [args.start_rollout_id]*len(critic_starts):
        raise RuntimeError('Critic resume iteration mismatch')
    if args.start_rollout_id and actor_starts != [args.start_rollout_id]*len(actor_starts):
        raise RuntimeError('Actor resume iteration mismatch')
    if args.rollout_global_dataset:
        ray.get(manager.load.remote(args.start_rollout_id-1))
    actor.update_weights()
    behavior_version = engine_version(manager)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, local_files_only=True)
    sentinel = tokenizer.eos_token_id
    if not isinstance(sentinel, int):
        raise ValueError('Expected one EOS id for critic-only sentinel')
    parallel = dict(dp_size=2, cp_size=1, vpp_size=1, microbatch_group_size_per_vp_stage=1)
    native_scorer = CriticScorer(critic._actor_handlers, critic_args, parallel, tokenizer, sentinel)
    overlap = args.ppo_execution == 'overlap'
    replica = None
    if overlap:
        from critic_replica import FrozenCriticReplica, ReplicaScorer
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
        pg, bundle = pgs['critic_inference']
        replica = ray.remote(num_gpus=1, num_cpus=1)(FrozenCriticReplica).options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg,
                placement_group_bundle_index=bundle)).remote(args.hf_checkpoint, args.seq_length)
        scorer = ReplicaScorer(replica)
    else:
        scorer = native_scorer
    service = serve(scorer)
    value_url = f'http://{ray.util.get_node_ip_address()}:{service.server_port}'
    behavior_round = args.start_rollout_id

    contexts = [tokenizer.apply_chat_template([dict(role='user', content=s)],
        tokenize=False, add_generation_prompt=True) for s in
        ('Checkpoint scoring readiness.', 'A different checkpoint.')]
    equivalence_contexts = list(contexts)
    if args.ppo_critic_equivalence_contexts:
        source = Path(args.ppo_critic_equivalence_contexts)
        equivalence_contexts += [row['context'] for row in json.loads(source.read_text())['contexts']]
        write_json(run/'critic-equivalence-input.json', dict(source=str(source),
            sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    # Exercise longer inputs as well as real root contexts before expensive search.
    equivalence_contexts += [tokenizer.apply_chat_template([dict(role='user',
        content='A passenger has requested a change. '*n)], tokenize=False,
        add_generation_prompt=True) for n in (1000, 3000)]
    previous_snapshot = None

    def publish_critic(completed_rounds, *, initial=False):
        nonlocal previous_snapshot
        start = time.monotonic()
        version = f'critic-{completed_rounds:04d}'
        snapshot = run/'critic-snapshots'/version
        exports = ray.get([a.export_snapshot.remote(str(snapshot), version)
                           for a in critic._actor_handlers])
        published = ray.get(replica.publish.remote(str(snapshot), version))
        check_contexts = equivalence_contexts if initial else contexts
        native_scorer.begin(version)
        try:
            expected = native_scorer.score(check_contexts, version)
        finally:
            native_scorer.end()
        scorer.begin(version)
        try:
            actual = scorer.score(check_contexts, version)
            repeated = scorer.score(check_contexts, version)
        finally:
            scorer.end()
        error = max(abs(a-b) for a,b in zip(expected['scores'], actual['scores'], strict=True))
        report = dict(version=version, native=expected, replica=actual,
            context_sha256=[hashlib.sha256(c.encode()).hexdigest() for c in check_contexts],
            context_tokens=[len(tokenizer.encode(c, add_special_tokens=False)) for c in check_contexts],
            max_abs_error=error, tolerance=.005, deterministic=actual == repeated,
            exports=exports, publication=published, seconds=time.monotonic()-start)
        audit_path = run/'critic-publication'/f'{version}.json'
        audit_path.parent.mkdir(exist_ok=True)
        write_json(audit_path, report)
        if error > .005 or actual != repeated:
            raise RuntimeError('Standalone critic differs from native final-context inference')
        if previous_snapshot is not None:
            shutil.rmtree(previous_snapshot)
        previous_snapshot = snapshot
        return report['seconds']

    if overlap:
        publish_critic(behavior_round, initial=True)

    def freeze(round_id):
        value_version = f'critic-{behavior_round:04d}'
        result = dict(rollout_id=round_id, value_version=value_version,
            policy_version=f'actor-{max(0, behavior_round-args.num_critic_only_steps):04d}',
            server_weight_version=behavior_version, value_url=value_url, recipe_id=RECIPE_ID,
            behavior_round=behavior_round, execution=args.ppo_execution,
            estimator=estimator_for_round(round_id, args.num_critic_only_steps))
        write_json(run/'collection-freeze.json', result)
        return result

    initial = freeze(args.start_rollout_id)
    # Exercise distributed packing, native final-context inference, TP agreement,
    # and offload lifecycle before the expensive first search.
    scorer.begin(initial['value_version'])
    try:
        ready = scorer.score(contexts, initial['value_version'])
        if ready != scorer.score(contexts, initial['value_version']):
            raise RuntimeError('Frozen critic scoring is nondeterministic')
        write_json(run/'critic-readiness.json', ready)
    finally:
        scorer.end()
    write_json(run/'status.json', dict(stage='critic_warmup' if args.start_rollout_id < args.num_critic_only_steps else 'ppo',
                                    completed_actor_updates=max(0, args.start_rollout_id-args.num_critic_only_steps),
                                    completed_collection_rounds=args.start_rollout_id))

    def begin_collection(round_id):
        frozen = freeze(round_id)
        scorer.begin(frozen['value_version'])
        return dict(frozen=frozen, started=time.monotonic(),
                    ref=manager.generate.remote(round_id))

    def finish_collection(pending):
        try:
            refs = ray.get(pending['ref'])
        finally:
            scorer.end()
        if engine_version(manager) != pending['frozen']['server_weight_version']:
            raise RuntimeError('Actor changed during collection')
        timing = json.loads((run/'rollouts'/f"train-{pending['frozen']['rollout_id']:04d}"/
                             'collection-timing.json').read_text())
        return dict(refs=refs, frozen=pending['frozen'], seconds=timing['seconds'])

    try:
        if args.start_rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(manager.eval.remote(0))
        if not args.ppo_benchmark and resume is not None and (args.start_rollout_id == args.num_critic_only_steps or
                (resume['actor_updates'] > 0 and resume['actor_updates'] % args.eval_interval == 0)):
            # A checkpoint is saved before its evaluation. Repeat that boundary
            # evaluation in the fresh attempt so interruption cannot omit it.
            ray.get(manager.eval.remote(args.start_rollout_id))
        last_round = args.num_rollout-1
        if args.ppo_stop_after_round is not None:
            last_round = min(last_round, args.ppo_stop_after_round)
        if last_round < args.start_rollout_id:
            raise ValueError('Stop round precedes resume cursor')
        ready = None
        wall_start = time.monotonic()
        write_json(run/'throughput-start.json', dict(unix_time=time.time(), startup_seconds=time.monotonic()-driver_started,
            start_round=args.start_rollout_id, last_round=last_round, execution=args.ppo_execution))
        for round_id in range(args.start_rollout_id, last_round+1):
            start = time.monotonic()
            if ready is None:
                ready = finish_collection(begin_collection(round_id))
            actor_refs, frozen = ready['refs'], ready['frozen']
            collection_seconds = ready['seconds']
            ready = None
            collected = time.monotonic()
            directory = run/'rollouts'/f'train-{round_id:04d}'
            lineage = BatchStamp(round_id, frozen['behavior_round'], args.num_critic_only_steps).lineage(
                round_id, overlap=overlap)
            write_json(directory/'training-lineage.json', lineage)
            contract = json.loads((directory/'contract.json').read_text())
            if (contract['recipe_id'] != RECIPE_ID
                    or contract['estimator'] != frozen['estimator']
                    or contract['policy_version'] != frozen['policy_version']
                    or contract['value_version'] != frozen['value_version']
                    or contract['server_weight_version'] != frozen['server_weight_version']):
                raise ValueError('Collection preparation recipe mismatch')
            rows = []
            warmup = round_id < args.num_critic_only_steps
            for group in range(args.rollout_batch_size):
                for row in read_rows(directory/f'group-{group:03d}.critic.jsonl.gz'):
                    if row['metadata'].get('target_source') != target_source(frozen['estimator']):
                        raise ValueError('Critic training target source mismatch')
                    if row['metadata']['value_version'] != frozen['value_version']:
                        raise RuntimeError('Stale critic target version')
                    rows.append(row)
            from critic_selection import learned_critic_rows
            rows, supervision = learned_critic_rows(rows)
            write_json(directory/'critic-supervision.json', supervision)
            rows = [checkpoint_fields(row, tokenizer, sentinel_token_id=sentinel,
                max_sequence_length=args.seq_length, warmup=warmup) for row in rows]
            data = training_data(rows, lane='critic', expected_groups=range(args.rollout_batch_size))
            critic_refs = put_packets(partition_data(critic_args, parallel, data))
            actor_updates = max(0, round_id+1-args.num_critic_only_steps)
            stop_requested = (run/'STOP').exists()
            save_now = save_boundary(round_id, args.num_critic_only_steps,
                                     args.save_interval, last_round) or stop_requested
            eval_now = not args.ppo_benchmark and (
                (round_id+1 == args.num_critic_only_steps and args.skip_eval_before_train) or
                (not warmup and actor_updates % args.eval_interval == 0))
            pending = None
            if overlap and may_prefetch(round_id, warmup_rounds=args.num_critic_only_steps,
                    last_round=last_round, save_now=save_now, eval_now=eval_now,
                    stop_requested=stop_requested):
                pending = begin_collection(round_id+1)
            train_started = time.monotonic()
            train_started_unix = time.time()
            ray.get(critic.async_train(round_id, critic_refs))
            critic_done = time.monotonic()
            if not warmup:
                # No foreign critic values: targets were frozen before either update.
                ray.get(actor.async_train(round_id, actor_refs))
            trained = time.monotonic()
            trained_unix = time.time()
            if pending is not None:
                ready = finish_collection(pending)
            drained = time.monotonic()
            if save_now:
                if ready is not None:
                    raise RuntimeError('Checkpoint cannot skip a prefetched question batch')
                actor.save_model(round_id, force_sync=True)
                critic.save_model(round_id, force_sync=True)
                if args.rollout_global_dataset:
                    ray.get(manager.save.remote(round_id))
            if not warmup:
                previous_version = behavior_version
                actor.update_weights()
                behavior_version = engine_version(manager)
                if previous_version == behavior_version:
                    raise RuntimeError('Slime did not advance actor weight version')
            elif engine_version(manager) != behavior_version:
                raise RuntimeError('Actor inference weights changed during critic-only warm-up')
            behavior_round = round_id+1
            publish_seconds = publish_critic(behavior_round) if overlap else 0.
            status = dict(stage='critic_warmup' if warmup else 'ppo',
                completed_actor_updates=actor_updates, completed_collection_rounds=round_id+1,
                server_weight_version=behavior_version, execution=args.ppo_execution,
                lineage=lineage, prefetched_next=ready is not None,
                optimizer_window=dict(started_unix=train_started_unix, finished_unix=trained_unix),
                throughput_elapsed=time.monotonic()-wall_start,
                seconds=dict(collection=collection_seconds, initial_collection_wait=collected-start,
                    critic=critic_done-train_started, actor=trained-critic_done,
                    prefetch_tail=drained-trained, critic_publication=publish_seconds,
                    total=time.monotonic()-start))
            write_json(run/'status.json', status)
            write_json(directory/'training-complete.json', status)
            if eval_now:
                freeze(round_id+1)
                ray.get(manager.eval.remote(round_id+1))
            if stop_requested:
                break
        write_json(run/('completed.json' if round_id+1 == args.num_rollout else 'paused.json'), status)
    except Exception:
        write_json(run/'failed.json', dict(rollout_id=round_id if 'round_id' in locals() else None,
            traceback=traceback.format_exc(), last_completed=json.loads((run/'status.json').read_text())))
        raise
    finally:
        service.shutdown()
        service.server_close()
    ray.get(manager.dispose.remote())
    if replica is not None:
        ray.kill(replica)
    finish_tracking(args)


if __name__ == '__main__':
    args = parse_args(custom_args)
    ray.init(address=os.environ['RAY_ADDRESS'], runtime_env={'env_vars':{
        'GLOO_SOCKET_IFNAME':'ens0', 'NCCL_SOCKET_IFNAME':'ens0', 'NCCL_IB_DISABLE':'1',
        'PYTHONPATH':os.environ['PYTHONPATH']}})
    train(args)
