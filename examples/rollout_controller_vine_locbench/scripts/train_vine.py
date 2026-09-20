"""Actor-only VinePPO with one-batch asynchronous rollout overlap and drained saves."""
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
import time
import traceback

import ray
from slime.observability.logging_utils import configure_logger, init_tracking, finish_tracking
from slime.ray.placement_group import create_actor_model, create_rollout_manager
from slime.utils.arguments import parse_args
from placement import pinned_placement
from pipeline import BatchStamp, save_boundary, may_prefetch
from recipe import RECIPE_ID
from resume_vine import prepare_resume, validate_actor_restore
from optimizer_restore import inspect_optimizer
from vine_actor import VineActor


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str)+'\n')
    tmp.replace(path)


def custom_args(parser):
    parser.add_argument('--vine-group-size', type=int, default=4)
    parser.add_argument('--resume-vine-group-size-from', type=int)
    parser.add_argument('--vine-replay-batch', type=str)
    parser.add_argument('--rollout-engine-base-port', type=int, default=22000)
    parser.add_argument('--ppo-search-concurrency', type=int, default=4)
    parser.add_argument('--ppo-pass-tokens', type=int, default=140000)
    parser.add_argument('--ppo-prior-strength', type=float, default=1.)
    parser.add_argument('--ppo-execution', choices=['sync','overlap'], default='overlap')
    parser.add_argument('--ppo-seed-namespace', default='locbench-vine-imitation-sep17')
    parser.add_argument('--loc-collection-profile', default='compact24-reply8-read60')
    parser.add_argument('--vine-calibration-only', action='store_true')
    return parser


def engine_version(manager):
    engines, *_ = ray.get(manager.get_updatable_engines_and_lock.remote())
    versions = [str(v) for v in ray.get([e.get_weight_version.remote() for e in engines])]
    if not versions or len(set(versions)) != 1 or versions[0] == 'None':
        raise ValueError(f'Inconsistent engine versions: {versions}')
    return versions[0]


def train(args):
    configure_logger()
    if args.colocate or args.release_train or args.num_critic_only_steps:
        raise ValueError('Vine uses dedicated persistent actor ranks and no critic warmup')
    if args.normalize_advantages or args.calculate_per_token_loss:
        raise ValueError('Keep prepared edge credit and question/edge normalization')
    if args.n_samples_per_prompt != 1 or args.global_batch_size != args.rollout_batch_size or args.num_steps_per_rollout != 1:
        raise ValueError('Expected one optimizer step per fresh question batch')
    if args.rollout_temperature != 1.0 or args.log_probs_chunk_size <= 0 or not args.bf16:
        raise ValueError('Chunked Vine logits require BF16, temperature1, and positive chunk size')
    args.vine_chunked_logits = True
    args.use_critic = False  # native PPO policy loss, supplied MC advantages, no critic actor
    # Native argument parsing enables offload for PPO because it normally shares
    # training GPUs with a learned critic. Vine has no second model on those GPUs.
    args.offload_train = False
    args.disable_param_buffers_cpu_backup = False
    args.disable_grad_buffers_cpu_backup = False
    args.train_env_vars = dict(args.train_env_vars, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    run = Path(args.save).parent
    if (run/'recipe.json').exists(): raise ValueError('Use a fresh attempt directory')
    run.mkdir(parents=True, exist_ok=True)
    resume = prepare_resume(args)
    start = args.start_rollout_id or 0
    if start >= args.num_rollout:
        raise ValueError('Resume has already reached the requested update count')
    write(run/'recipe.json', dict(recipe_id=RECIPE_ID, arguments={k:v for k,v in vars(args).items() if k not in ('wandb_key','wandb_api_key')},
        script_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        driver_step=f"{os.environ.get('SLURM_JOB_ID')}.{os.environ.get('SLURM_STEP_ID')}"))
    archive = run/'runtime-sources'
    archive.mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(path, archive/path.name)
    args.start_rollout_id = start
    if resume is not None:
        write(run/'resume.json', resume)
    # This placement allocates no replica when execution is sync; overlap is owned
    # here, with no learned critic or inference reservation.
    place_args = copy.deepcopy(args)
    place_args.ppo_execution = 'sync'
    pgs = pinned_placement(place_args)
    init_tracking(args)
    manager, _ = create_rollout_manager(args, pgs['rollout'])
    actor_args = copy.deepcopy(args)
    actor_args.num_gpus_per_node = args.actor_num_gpus_per_node
    actor, starts = create_actor_model(actor_args, pgs, manager, actor_cls=VineActor)
    # Slime's HF loader reports iteration zero and its actor returns zero+one.
    # This is an initialization convention, not a completed optimizer update.
    expected_native_start = start if resume is not None else 1
    if len(set(starts)) != 1 or starts[0] != expected_native_start:
        raise ValueError(f'Unexpected actor iterations: {starts}, expected {expected_native_start}')
    reports = ray.get([h.__ray_call__.remote(inspect_optimizer) for h in actor._actor_handlers])
    if resume is not None:
        write(run/'initialization.json', validate_actor_restore(resume, reports,
            args.global_batch_size, args.actor_num_nodes*args.actor_num_gpus_per_node))
        ray.get(manager.load.remote(start-1))
    else:
        if any(r['scheduler_samples'] != 0 or any(step != 0 for step in r['steps']) for r in reports):
            raise ValueError('Base initialization unexpectedly has training history')
        write(run/'initialization.json', dict(passed=True, fresh=True,
            native_cursors=starts, start_rollout_id=start, reports=reports))
    actor.update_weights()
    version = engine_version(manager)
    behavior_round = start
    overlap = args.ppo_execution == 'overlap'

    def begin(round_id):
        frozen = dict(rollout_id=round_id, policy_version=f'actor-{behavior_round:04d}',
            server_weight_version=version, behavior_round=behavior_round,
            value_version='mc-outcome', value_url='', estimator='vine_ppo',
            recipe_id=RECIPE_ID, execution=args.ppo_execution)
        write(run/'collection-freeze.json', frozen)
        return dict(ref=manager.generate.remote(round_id), frozen=frozen, started=time.monotonic())

    def finish(pending):
        refs = ray.get(pending['ref'])
        if engine_version(manager) != pending['frozen']['server_weight_version']:
            raise ValueError('Actor changed during collection')
        return dict(refs=refs, frozen=pending['frozen'], seconds=time.monotonic()-pending['started'])

    ready = None
    try:
        if args.vine_calibration_only:
            batches = []
            for round_id in range(start, args.num_rollout):
                result = finish(begin(round_id))
                summary = json.loads((run/'rollouts'/f'train-{round_id:04d}'/'summary.json').read_text())
                batches.append(dict(round_id=round_id, seconds=result['seconds'],
                    cost=summary['cost'], results=summary['results']))
            report = dict(group_size=args.vine_group_size, batches=batches,
                questions=sum(len(batch['results']) for batch in batches),
                cost={key:sum(batch['cost'][key] for batch in batches)
                      for key in ('input_tokens','output_tokens','generations')})
            write(run/'calibration.json', report)
            return
        for round_id in range(start, args.num_rollout):
            started = time.monotonic()
            if ready is None: ready = finish(begin(round_id))
            data, frozen, collection_seconds = ready['refs'], ready['frozen'], ready['seconds']
            ready = None
            directory = run/'rollouts'/f'train-{round_id:04d}'
            lineage = BatchStamp(round_id, frozen['behavior_round'], 0).lineage(round_id, overlap=overlap)
            write(directory/'training-lineage.json', lineage)
            contract = json.loads((directory/'contract.json').read_text())
            if any(contract[k] != frozen[k] for k in ('recipe_id','estimator','policy_version','server_weight_version')):
                raise ValueError('Frozen collection provenance mismatch')
            stop = (run/'STOP').exists()
            save = round_id == start or save_boundary(round_id, 0, args.save_interval, args.num_rollout-1) or stop
            pending = begin(round_id+1) if overlap and may_prefetch(round_id, warmup_rounds=0,
                last_round=args.num_rollout-1, save_now=save, eval_now=False, stop_requested=stop) else None
            train_started = time.monotonic()
            ray.get(actor.async_train(round_id, data))
            trained = time.monotonic()
            if pending is not None: ready = finish(pending)
            if save:
                if ready is not None: raise ValueError('Cannot save beyond an untrained dataset cursor')
                actor.save_model(round_id, force_sync=True)
                if args.rollout_global_dataset: ray.get(manager.save.remote(round_id))
                write(run/'checkpoints'/f'round-{round_id:04d}.json',dict(round_id=round_id,
                    actor_updates=round_id+1, native=str(run/'actor'/f'iter_{round_id:07d}'),
                    cursor=str(run/'actor/rollout'/f'global_dataset_state_dict_{round_id}.pt')))
            old_version = version
            actor.update_weights()
            version = engine_version(manager)
            if version == old_version: raise ValueError('Actor publication did not advance')
            behavior_round = round_id+1
            status = dict(stage='vine_ppo', completed_actor_updates=round_id+1,
                execution=args.ppo_execution, lineage=lineage, prefetched_next=ready is not None,
                group_size=args.vine_group_size, checkpoint_saved=save,
                seconds=dict(collection=collection_seconds,actor=trained-train_started,total=time.monotonic()-started))
            write(run/'status.json', status)
            write(directory/'training-complete.json', status)
            if stop: break
        write(run/('paused.json' if stop else 'completed.json'), status)
    except BaseException:
        write(run/'failed.json',dict(traceback=traceback.format_exc()))
        raise
    finally:
        ray.get(manager.dispose.remote())
        finish_tracking(args)


if __name__ == '__main__':
    args = parse_args(custom_args)
    ray.init(address=os.environ['RAY_ADDRESS'], runtime_env={'env_vars':{
        'GLOO_SOCKET_IFNAME':'ens0', 'NCCL_SOCKET_IFNAME':'ens0', 'NCCL_IB_DISABLE':'1',
        'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True', 'PYTHONPATH':os.environ['PYTHONPATH']}})
    train(args)
