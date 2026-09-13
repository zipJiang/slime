"""Build and validate the critic handoff consumed by a future PPO pilot."""
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def validate_calibration(report, name):
    calibration=report.get('calibration')
    if not isinstance(calibration,dict) or not isinstance(calibration.get('bins'),list):
        raise ValueError(f'{name} validation is missing calibration evidence')
    bins=calibration['bins']
    if not bins:
        raise ValueError(f'{name} validation has no populated calibration bins')
    weights=[];gaps=[];lowers=[]
    for bucket in bins:
        values=[bucket.get(key) for key in ('lower','upper','weight','predicted_mean',
                                             'target_mean','absolute_gap')]
        if (any(not isinstance(value,(int,float)) or not math.isfinite(value) for value in values)
                or not 0<=bucket['lower']<bucket['upper']<=1
                or not 0<bucket['weight']<=1
                or not 0<=bucket['predicted_mean']<=1
                or not 0<=bucket['target_mean']<=1
                or not math.isclose(bucket['absolute_gap'],
                                    abs(bucket['predicted_mean']-bucket['target_mean']),
                                    abs_tol=1e-12,rel_tol=1e-12)
                or not isinstance(bucket.get('contexts'),int) or bucket['contexts']<=0
                or not isinstance(bucket.get('questions'),int) or bucket['questions']<=0):
            raise ValueError(f'{name} validation has malformed calibration bins')
        weights.append(bucket['weight']);gaps.append(bucket['absolute_gap'])
        lowers.append(bucket['lower'])
    expected=sum(weight*gap for weight,gap in zip(weights,gaps,strict=True))
    reported_expected=calibration.get('expected_absolute_gap')
    reported_maximum=calibration.get('maximum_absolute_gap')
    if (any(not math.isclose(bucket['upper']-bucket['lower'],.1,
                             abs_tol=1e-12,rel_tol=1e-12) for bucket in bins)
            or lowers!=sorted(set(lowers))
            or not isinstance(reported_expected,(int,float))
            or not isinstance(reported_maximum,(int,float))
            or not math.isfinite(reported_expected) or not math.isfinite(reported_maximum)
            or not math.isclose(sum(weights),1.,abs_tol=1e-9,rel_tol=1e-9)
            or not math.isclose(reported_expected,expected,
                                abs_tol=1e-12,rel_tol=1e-12)
            or not math.isclose(reported_maximum,max(gaps),
                                abs_tol=1e-12,rel_tol=1e-12)):
        raise ValueError(f'{name} validation calibration summary is inconsistent')


def build_candidate(training, *, base_model, context_source_file_sha256,
                    context_function_sha256):
    training=Path(training).resolve()
    complete=read(training/'complete.json')
    validate_calibration(complete.get('initial',{}),'initial')
    validate_calibration(complete.get('trained',{}),'trained')
    native=read(training/'native-validated.json')
    reload=read(training/'reload-audit.json')
    inference=read(training/'inference-audit.json')
    sampling=read(training/'sampling-audit.json')
    storage=read(training/'storage-preflight.json')
    iteration=int(native['native_iteration'] if 'native_iteration' in native else native['iteration'])
    checkpoint_root=Path(native['checkpoint']).resolve()
    checkpoint=checkpoint_root/f'iter_{iteration:07d}'
    readback_path=checkpoint_root/f'iter_{iteration:07d}-readback.json'
    readback=read(readback_path)
    if not checkpoint.is_dir():
        raise ValueError('Native critic checkpoint directory is missing')
    if (Path(complete['native_checkpoint']).resolve()!=checkpoint_root
            or complete['native_iteration']!=iteration):
        raise ValueError('Completion record points at a different native critic')
    if (Path(readback['checkpoint']).resolve()!=checkpoint
            or readback['role']!='critic'
            or readback['common_state']['iteration']!=iteration
            or readback['expected_optimizer_steps']!=native['updates']
            or readback['optimizer_steps']!=[native['updates']]
            or not readback['full_storage_read'] or not readback['finite_tensors']):
        raise ValueError('Native checkpoint readback does not match the trained critic')
    if (reload['cursors']!=[0,0,0,0]
            or not reload['passed']
            or not all(item['fresh'] for item in reload['optimizers'])
            or not reload['finetune'] or not reload['no_load_optim'] or not reload['no_load_rng']):
        raise ValueError('Weight-only reload is not a fresh optimizer/cursor initialization')
    portable=Path(complete['inference']).resolve() if complete.get('inference') else None
    comparison=inference['comparison']
    if complete['portable_inference_passed'] != comparison['passed']:
        raise ValueError('Portable inference evidence disagrees with completion record')
    if portable is not None and not (portable/'manifest.json').is_file():
        raise ValueError('Portable critic snapshot is missing')
    if (not sampling.get('passed') or sampling.get('traces')!=640
            or sampling.get('samples_per_question')!=4
            or not sampling.get('distinct_within_question_at_every_call')):
        raise ValueError('Collection sampling streams lack a complete independence audit')
    if not storage.get('passed'):
        raise ValueError('Critic training did not pass its storage preflight')
    checks=dict(
        better_than_untrained=bool(complete['better_than_initial']),
        better_than_train_fitted_constant=bool(complete['better_than_constant']),
        portable_inference=bool(comparison['passed']),
        native_weight_only_reload=True,
        native_checkpoint_full_readback=True,
        independent_collection_streams=True,
        heldout_calibration_reported=True,
    )
    reasons=[name for name,passed in checks.items() if not passed]
    evidence=[training/name for name in ['complete.json','native-validated.json','reload-audit.json',
                                         'inference-audit.json','dataset-inventory.json','recipe.json',
                                         'sampling-audit.json','storage-preflight.json']]
    evidence.append(readback_path)
    if portable is not None: evidence.append(portable/'manifest.json')
    return dict(schema='browsecomp-critic-warmstart-v1',
        ready_for_zero_warmup_pilot=not reasons,
        ready_for_long_run=False,
        long_run_gate='A successful on-policy zero-warmup PPO pilot is still required',
        failed_candidate_checks=reasons,
        checks=checks,
        base_actor=str(Path(base_model).resolve()),
        critic=dict(load=str(checkpoint_root),ckpt_step=iteration,finetune=True,
                    no_load_optim=True,no_load_rng=True,expected_rollout_cursor=0),
        portable_inference=str(portable) if portable is not None else None,
        context_source_file_sha256=context_source_file_sha256,
        context_function_sha256=context_function_sha256,
        evidence_sha256={str(path.relative_to(training)):digest(path) for path in evidence})


def require_pilot_candidate(path):
    path=Path(path).resolve()
    candidate=read(path)
    if candidate.get('schema')!='browsecomp-critic-warmstart-v1':
        raise ValueError('Unknown critic warmstart schema')
    if not candidate.get('ready_for_zero_warmup_pilot'):
        raise ValueError('Critic candidate did not pass its offline quality gates')
    if candidate.get('ready_for_long_run'):
        raise ValueError('Pretraining cannot authorize a long zero-warmup PPO run')
    flags=candidate['critic']
    if not (flags['finetune'] and flags['no_load_optim'] and flags['no_load_rng']
            and flags['expected_rollout_cursor']==0):
        raise ValueError('Critic warmstart flags would retain training history')
    if not Path(candidate['base_actor']).is_dir():
        raise ValueError('Base actor snapshot is missing')
    checkpoint=Path(flags['load'])/f"iter_{int(flags['ckpt_step']):07d}"
    if not checkpoint.is_dir():
        raise ValueError('Native critic warmstart checkpoint is missing')
    for relative,expected in candidate['evidence_sha256'].items():
        evidence=(path.parent/relative).resolve()
        if not evidence.is_relative_to(path.parent) or digest(evidence)!=expected:
            raise ValueError(f'Critic warmstart evidence changed: {relative}')
    return candidate


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('candidate',type=Path)
    args=parser.parse_args()
    print(json.dumps(require_pilot_candidate(args.candidate),indent=2))
