import json
from pathlib import Path
import pytest
from warmstart_candidate import build_candidate, require_pilot_candidate


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def calibration():
    return dict(calibration=dict(bins=[dict(lower=0.,upper=.1,weight=1.,contexts=1,
        questions=1,predicted_mean=.05,target_mean=0.,absolute_gap=.05)],
        expected_absolute_gap=.05,maximum_absolute_gap=.05))


def fixture(tmp_path, *, better=True):
    (tmp_path/'base').mkdir()
    training=tmp_path/'training';native=training/'native';checkpoint=native/'iter_0000015'
    checkpoint.mkdir(parents=True)
    write(training/'complete.json',dict(better_than_initial=better,better_than_constant=better,
        portable_inference_passed=True,inference=str(training/'inference'),
        native_checkpoint=str(native),native_iteration=15,
        initial=calibration(),trained=calibration()))
    write(training/'native-validated.json',dict(updates=16,checkpoint=str(native),iteration=15))
    write(training/'reload-audit.json',dict(passed=True,cursors=[0]*4,
        optimizers=[dict(fresh=True)]*4,finetune=True,no_load_optim=True,no_load_rng=True))
    write(training/'inference-audit.json',dict(comparison=dict(passed=True)))
    write(native/'iter_0000015-readback.json',dict(checkpoint=str(checkpoint),role='critic',
        common_state=dict(iteration=15),expected_optimizer_steps=16,optimizer_steps=[16],
        full_storage_read=True,finite_tensors=True))
    write(training/'inference/manifest.json',dict(version='v1'))
    write(training/'dataset-inventory.json',{})
    write(training/'recipe.json',{})
    write(training/'sampling-audit.json',dict(passed=True,traces=640,samples_per_question=4,
        distinct_within_question_at_every_call=True))
    write(training/'storage-preflight.json',dict(passed=True))
    return training


def test_candidate_encodes_fresh_model_only_load_and_requires_pilot(tmp_path):
    training=fixture(tmp_path)
    candidate=build_candidate(training,base_model=tmp_path/'base',context_source_file_sha256='a'*64,
                              context_function_sha256='e'*64)
    assert candidate['ready_for_zero_warmup_pilot']
    assert not candidate['ready_for_long_run']
    assert candidate['critic']['ckpt_step']==15
    path=training/'warmstart-candidate.json';write(path,candidate)
    assert require_pilot_candidate(path)['critic']['expected_rollout_cursor']==0


def test_offline_quality_failure_cannot_start_zero_warmup_pilot(tmp_path):
    training=fixture(tmp_path,better=False)
    candidate=build_candidate(training,base_model=tmp_path/'base',context_source_file_sha256='b'*64,
                              context_function_sha256='f'*64)
    assert not candidate['ready_for_zero_warmup_pilot']
    path=training/'warmstart-candidate.json';write(path,candidate)
    with pytest.raises(ValueError,match='quality gates'):
        require_pilot_candidate(path)


def test_inconsistent_checkpoint_readback_is_rejected(tmp_path):
    training=fixture(tmp_path)
    readback=training/'native/iter_0000015-readback.json'
    value=json.loads(readback.read_text());value['optimizer_steps']=[15];write(readback,value)
    with pytest.raises(ValueError,match='readback'):
        build_candidate(training,base_model=tmp_path/'base',context_source_file_sha256='c'*64,
                        context_function_sha256='0'*64)


def test_missing_calibration_cannot_create_candidate(tmp_path):
    training=fixture(tmp_path)
    complete=json.loads((training/'complete.json').read_text())
    complete['trained'].pop('calibration');write(training/'complete.json',complete)
    with pytest.raises(ValueError,match='calibration evidence'):
        build_candidate(training,base_model=tmp_path/'base',context_source_file_sha256='c'*64,
                        context_function_sha256='0'*64)


def test_consumer_rechecks_evidence_hashes(tmp_path):
    training=fixture(tmp_path)
    candidate=build_candidate(training,base_model=tmp_path/'base',context_source_file_sha256='d'*64,
                              context_function_sha256='1'*64)
    path=training/'warmstart-candidate.json';write(path,candidate)
    write(training/'recipe.json',dict(changed=True))
    with pytest.raises(ValueError,match='evidence changed'):
        require_pilot_candidate(path)
