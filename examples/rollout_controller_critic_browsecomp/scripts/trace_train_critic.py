"""Six-GPU critic-only warmup, with overlapped validation on a seventh GPU."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from transformers import AutoTokenizer
from slime.utils.arguments import parse_args
from slime.observability.logging_utils import configure_logger
from slime.ray.placement_group import allocate_train_group

from train_critic import placement, write, status, custom_args as base_args
from audit_checkpoint import audit as audit_checkpoint
from audit_sampling import audit as audit_sampling
from batches import partition_data, put_packets
from critic_actor import CheckpointCriticActor
from dataset import load_dataset
from inference_probe import probe_indices
from pilot_runtime import CONTEXT_LIMIT, environment_contract, verify_harness
from provenance import validate_collection_provenance, dataset_inventory, function_sha256, sha256
from targets import checkpoint_fields
from trace_validation import TraceValidationReplica
from trace_warmup import packet, baseline, report, choose_best
from value_service import CriticScorer
from warmstart_candidate import build_candidate, require_pilot_candidate

EXPERIMENT = Path(__file__).resolve().parents[1]


def custom_args(parser):
    base_args(parser)
    parser.add_argument('--trace-source-candidate', type=Path, required=True)
    parser.add_argument('--trace-root-mass', type=float, default=.25)
    parser.add_argument('--trace-patience', type=int, default=2)
    parser.add_argument('--trace-equivalence-tolerance', type=float, default=.01)
    return parser


def run(args):
    configure_logger()
    verify_harness()
    if environment_contract()['profile'] != 'trace96k':
        raise ValueError('TRACE warmup requires its versioned environment')
    if (args.actor_num_nodes != 3 or args.actor_num_gpus_per_node != 2
            or args.tensor_model_parallel_size != 2 or args.context_parallel_size != 1
            or args.pipeline_model_parallel_size != 1 or args.global_batch_size != 8
            or not args.use_critic or not args.offload_train or args.release_train
            or args.normalize_advantages or args.calculate_per_token_loss
            or args.critic_epochs not in (1, 2) or args.critic_eval_interval != 4
            or args.seq_length != CONTEXT_LIMIT or args.num_critic_only_steps < 32
            or args.trace_patience < 1 or args.lr > 1e-6 or args.lr <= 0
            or args.trace_equivalence_tolerance != .01):
        raise ValueError('Unsupported warmup recipe')
    source = require_pilot_candidate(args.trace_source_candidate)
    out = Path(args.save).parent
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'recipe.json').exists():
        raise ValueError('Warmup output must be fresh')
    usage = shutil.disk_usage(out)
    storage = dict(free_bytes=usage.free, required_free_bytes=2 * 1024**4,
                   passed=usage.free >= 2 * 1024**4)
    write(out / 'storage-preflight.json', storage)
    if not storage['passed']:
        raise ValueError('Need 2 TiB free for validation-boundary recovery checkpoints')
    collection = Path(args.critic_collection)
    manifest = json.loads((collection / 'manifest.json').read_text())
    if manifest.get('environment') != environment_contract():
        raise ValueError('Fresh warmup data must match the new environment')
    provenance = validate_collection_provenance(EXPERIMENT, collection, EXPERIMENT / 'data/split.json')
    audit = json.loads((collection.parent / 'collection-audit.json').read_text())
    if not audit.get('passed') or not audit.get('full_collection') or not audit.get('identities_exact'):
        raise ValueError('Full fresh collection readback is required')
    sampling = audit_sampling(collection / 'manifest.json', max_calls=256)
    data = load_dataset(collection, json.loads((EXPERIMENT / 'data/split.json').read_text()))
    constant = baseline(data['train'], args.trace_root_mass)
    args.load = source['critic']['load']; args.ckpt_step = source['critic']['ckpt_step']
    args.finetune = True; args.no_load_optim = True; args.no_load_rng = True
    args.start_rollout_id = 0
    args.save_hf = None; args.use_kl_loss = False; args.kl_coef = 0; args.use_opd = False
    args.disable_param_buffers_cpu_backup = False
    args.custom_advantage_function_path = None; args.rollout_data_postprocess_path = None
    args.custom_tis_function_path = None; args.init_method_std = .001
    write(out / 'recipe.json', dict(protocol='browsecomp-trace-warmup-v1',
        environment=environment_contract(), collection=str(collection.resolve()),
        source_candidate=str(args.trace_source_candidate.resolve()), source_candidate_sha256=sha256(args.trace_source_candidate),
        collection_provenance=provenance, root_mass=args.trace_root_mass,
        maximum_epochs=args.critic_epochs, validation_interval=4,
        validation_overlap='One immutable checkpoint at a time; up to one four-update block in flight',
        selection='Earliest balanced validation MSE improvement of at least 1e-4, including starting critic',
        arguments=vars(args), scripts={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}))
    write(out / 'sampling-audit.json', sampling)
    write(out / 'dataset-inventory.json', dataset_inventory(data))
    pg, physical = placement(3)
    write(out / 'placement.json', physical)
    validators = [n for n in ray.nodes() if n['Alive'] and n['Resources'].get('browsecomp_trace_validator', 0) == 1]
    if len(validators) != 1 or validators[0]['NodeManagerAddress'] in {p[0] for p in physical}:
        raise ValueError('Need a separate dedicated validation GPU')
    validator = ray.remote(num_gpus=1)(TraceValidationReplica).options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(validators[0]['NodeID'], soft=False)
    ).remote(args.hf_checkpoint, args.seq_length)
    critic = allocate_train_group(args, 3, 2, pg, role='critic', actor_cls=CheckpointCriticActor)
    status(out, 'native-initialization')
    cursors = critic.create()
    optimizers = ray.get([a.audit_optimizer_start.remote() for a in critic._actor_handlers])
    if cursors != [0]*6 or not all(r['fresh'] for r in optimizers):
        raise ValueError('Warmup must reset source optimizer/cursor state')
    write(out / 'initialization-audit.json', dict(cursors=cursors, optimizers=optimizers, source=source['critic']))
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, local_files_only=True)
    parallel = dict(dp_size=3, cp_size=1, vpp_size=1, microbatch_group_size_per_vp_stage=1)
    def scorer_for(group):
        return CriticScorer(group._actor_handlers, args, parallel, tokenizer, tokenizer.eos_token_id)
    scorer = scorer_for(critic)
    lengths = [len(tokenizer.encode(r['context'], add_special_tokens=False)) for r in data['validation']]
    indices = probe_indices(data['validation'], lengths)
    rows_ref = ray.put(data['validation'])
    def native_scores(contexts, version):
        scorer.begin(version)
        try:
            values = []
            for start in range(0, len(contexts), 8):
                values.extend(scorer.score(contexts[start:start+8], version)['scores'])
            return dict(version=version, scores=values)
        finally:
            scorer.end()
    def evaluate_async(step):
        version = f'trace-warmup-{step:04d}'
        directory = out / 'exports' / version
        ray.get([a.export_snapshot.remote(str(directory), version) for a in critic._actor_handlers])
        native = native_scores([data['validation'][i]['context'] for i in indices], version)
        future = validator.evaluate.remote(str(directory), version, rows_ref, indices, native,
                                          constant, args.trace_root_mass, args.trace_equivalence_tolerance)
        return dict(step=step, version=version, directory=str(directory), future=future)
    reports = []
    def finish_evaluation(pending):
        result = ray.get(pending['future'])
        metadata = {k: v for k, v in pending.items() if k != 'future'}
        write(out / 'validation' / f"update-{pending['step']:04d}.json", dict(**metadata, **result))
        if result.get('rejected'):
            raise ValueError('Native/portable validation equivalence failed')
        reports.append(dict(**metadata, **result['report']))
        best = choose_best(reports)
        write(out / 'selection.json', dict(best=reports[best], evaluated=reports,
              updates_without_improvement=len(reports)-best-1))
        return len(reports)-best-1 >= args.trace_patience
    status(out, 'initial-evaluation')
    finish_evaluation(evaluate_async(0))
    initial = reports[0]
    groups = {q: [r for r in data['train'] if r['group_index'] == q]
              for q in sorted({r['group_index'] for r in data['train']})}
    if len(groups) != 128:
        raise ValueError('Warmup recipe expects all 128 training questions')
    step = 0; pending = None; stop = False
    for epoch in range(args.critic_epochs):
        order = sorted(groups, key=lambda q: hashlib.sha256(f'trace-warmup/{epoch}/{q}'.encode()).hexdigest())
        for start in range(0, len(order), args.global_batch_size):
            records = []
            for group_id, q in enumerate(order[start:start+args.global_batch_size]):
                for original in groups[q]:
                    row = dict(original, group_index=group_id)
                    prepared = checkpoint_fields(row, tokenizer, sentinel_token_id=tokenizer.eos_token_id,
                                                 max_sequence_length=args.seq_length, warmup=True)
                    records.append(dict(prepared, turn=row['turn']))
            started = time.time()
            ray.get(critic.async_train(step, put_packets(partition_data(args, parallel,
                packet(records, args.trace_root_mass)))))
            step += 1
            status(out, 'critic-only-warmup', updates=step, maximum_updates=16*args.critic_epochs,
                   epoch=epoch+1, last_update_seconds=time.time()-started,
                   best_validated_update=reports[choose_best(reports)]['step'])
            if step % 4 == 0:
                if pending is not None:
                    stop = finish_evaluation(pending)
                critic.save_model(step-1, force_sync=True)
                pending = evaluate_async(step)
                if stop:
                    break
        if stop:
            break
    if pending is not None:
        finish_evaluation(pending)
    best = reports[choose_best(reports)]
    critic.release()
    selected = best['step']
    write(out / 'warmup-finished.json', dict(updates=step, selected_update=selected,
                                           selected=best, stopped_early=stop, unix_time=time.time()))
    if selected == 0:
        ray.kill(validator)
        write(out / 'complete.json', dict(updates=step, selected_update=0,
            ready_for_joint_training=False, reason='No checkpoint improved the starting critic on fresh held-out data',
            source_candidate=str(args.trace_source_candidate), unix_time=time.time()))
        status(out, 'complete-without-improvement', updates=step)
        return
    iteration = selected-1
    checkpoint = Path(args.save) / f'iter_{iteration:07d}'
    status(out, 'selected-checkpoint-readback', selected_update=selected)
    audit_checkpoint(checkpoint, selected, 'critic')
    warm = copy.deepcopy(args); warm.load=args.save; warm.ckpt_step=iteration
    loaded = allocate_train_group(warm, 3, 2, pg, role='critic', actor_cls=CheckpointCriticActor)
    cursors = loaded.create()
    optimizers = ray.get([a.audit_optimizer_start.remote() for a in loaded._actor_handlers])
    if cursors != [0]*6 or len(optimizers) != 6 or not all(r['fresh'] for r in optimizers):
        raise ValueError('Selected critic model-only reload retained training state')
    scorer = scorer_for(loaded)
    actual = native_scores([r['context'] for r in data['validation']], best['version'])
    expected_result = json.loads((out / 'validation' / f'update-{selected:04d}.json').read_text())
    expected = dict(version=best['version'], scores=expected_result['predictions'])
    # Compare every held-out context, not merely the publication probes.
    errors = [abs(a-b) for a,b in zip(actual['scores'], expected['scores'], strict=True)]
    reload_ok = max(errors) <= args.trace_equivalence_tolerance
    write(out / 'reload-audit.json', dict(passed=reload_ok, max_abs_error=max(errors),
        abs_errors=errors, tolerance=args.trace_equivalence_tolerance, world_size=6,
        cursors=cursors, optimizers=optimizers, finetune=True, no_load_optim=True, no_load_rng=True))
    write(out / 'validation' / 'weight-only-reload.json',
          dict(**report(data['validation'], actual['scores'], constant, args.trace_root_mass), predictions=actual['scores']))
    loaded.release(); ray.kill(validator)
    if not reload_ok:
        raise ValueError('Full selected checkpoint reload differs beyond numerical allowance')
    snapshot = out / 'inference'
    snapshot.symlink_to(Path(best['directory']).resolve(), target_is_directory=True)
    write(out / 'native-validated.json', dict(updates=selected, checkpoint=args.save, iteration=iteration,
        initial=initial, trained=best, reload_max_abs_error=max(errors)))
    write(out / 'inference-audit.json', dict(publication=expected_result['publication'],
        comparison=expected_result['comparison'], context_indices=indices))
    write(out / 'complete.json', dict(updates=selected, optimizer_updates_executed=step,
        initial=initial, trained=best, better_than_initial=best['mse'] < initial['mse'],
        better_than_constant=best['mse'] < best['baseline_mse'],
        trace_quality=dict(balanced_improvement=best['selection_mse'] < initial['selection_mse'],
            balanced_better_than_constant=best['selection_mse'] < best['selection_baseline_mse'],
            root_improvement=best['by_checkpoint']['root']['mse'] < initial['by_checkpoint']['root']['mse']),
        native_checkpoint=args.save, native_iteration=iteration, inference=str(snapshot),
        portable_inference_passed=True, unix_time=time.time()))
    write(out / 'warmstart-candidate.json', build_candidate(out, base_model=args.hf_checkpoint,
        context_source_file_sha256=sha256(EXPERIMENT/'scripts/pilot_runtime.py'),
        context_function_sha256=function_sha256(EXPERIMENT/'scripts/pilot_runtime.py', 'context')))
    status(out, 'complete', updates=step, selected_update=selected)


def main():
    args = parse_args(custom_args)
    out = Path(args.save).parent
    if args.critic_preflight_only:
        rows = [dict(tokens=[1,2,3], response_length=1, reward=0., loss_mask=[1],
            group_index=i//2, turn=i%2, metadata=dict(lane='critic', node_id=i)) for i in range(16)]
        parallel = dict(dp_size=3, cp_size=1, vpp_size=1, microbatch_group_size_per_vp_stage=1)
        packets = partition_data(args, parallel, packet(rows, args.trace_root_mass))
        source = require_pilot_candidate(args.trace_source_candidate)
        print(json.dumps(dict(preflight_passed=True, dp_packets=len(packets),
            sizes=[p['global_batch_sizes'] for p in packets], source=source['critic'],
            seq_length=args.seq_length, lr=args.lr, epochs=args.critic_epochs,
            root_mass=args.trace_root_mass, tolerance=args.trace_equivalence_tolerance)))
        return
    try:
        ray.init(address=os.environ['RAY_ADDRESS'], runtime_env={'env_vars':{
            'GLOO_SOCKET_IFNAME':'ens0', 'NCCL_SOCKET_IFNAME':'ens0', 'NCCL_IB_DISABLE':'1',
            'PYTHONPATH':os.environ['PYTHONPATH'], 'BROWSECOMP_PROFILE':'trace96k'}})
        run(args)
    except Exception:
        write(out/'failed.json', dict(traceback=traceback.format_exc(), unix_time=time.time()))
        raise


if __name__ == '__main__':
    main()
