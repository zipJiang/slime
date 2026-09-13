import json
from pathlib import Path
import pytest

from pilot_gate import promote, require_long_run
from warmstart_candidate import build_candidate, digest


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def calibration():
    return dict(calibration=dict(bins=[dict(lower=0.,upper=.1,weight=1.,contexts=1,
        questions=1,predicted_mean=.05,target_mean=0.,absolute_gap=.05)],
        expected_absolute_gap=.05,maximum_absolute_gap=.05))


def fixture(tmp_path):
    base=tmp_path/'base';base.mkdir();training=tmp_path/'training';native=training/'native'
    checkpoint=native/'iter_0000015';checkpoint.mkdir(parents=True)
    write(training/'complete.json',dict(better_than_initial=True,better_than_constant=True,
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
    write(training/'inference/manifest.json',{});write(training/'dataset-inventory.json',{});write(training/'recipe.json',{})
    write(training/'sampling-audit.json',dict(passed=True,traces=640,samples_per_question=4,
        distinct_within_question_at_every_call=True))
    write(training/'storage-preflight.json',dict(passed=True))
    candidate=build_candidate(training,base_model=base,context_source_file_sha256='a'*64,
                              context_function_sha256='b'*64)
    candidate_path=training/'warmstart-candidate.json';write(candidate_path,candidate)
    fresh=dict(cursors=[0]*4,optimizers=[dict(fresh=True)]*4)
    pilot=dict(schema='browsecomp-zero-warmup-pilot-v1',candidate_sha256=digest(candidate_path),
        context_function_sha256='b'*64,num_critic_only_steps=0,start_rollout_id=0,
        completed_joint_updates=2,initialization=dict(
            actor=dict(load=str(base),**fresh),critic=dict(load=str(native),ckpt_step=15,**fresh)),
        batches=[dict(target_replay_passed=True,on_policy_passed=True,
            context_contract_passed=True,infrastructure_failures=0,terminals=4,successes=successes)
            for successes in (1,0)],
        critic_publications=[dict(passed=True,deterministic=True,max_abs_error=.001,tolerance=.005)
                             for _ in range(3)],
        checkpoint={role:dict(full_storage_read=True,finite_tensors=True,optimizer_steps=[2])
                    for role in ('actor','critic')},scientific_rejection=False)
    pilot_path=tmp_path/'pilot.json';write(pilot_path,pilot)
    return candidate_path,pilot_path


def test_two_update_pilot_promotes_and_remains_verifiable(tmp_path):
    candidate,pilot=fixture(tmp_path)
    result=promote(candidate,pilot)
    assert result['ready_for_long_run'] and result['joint_updates']==2
    path=tmp_path/'promoted.json';write(path,result)
    assert require_long_run(path)['critic']['ckpt_step']==15


def test_two_rank_pilot_requires_complete_optimizer_evidence(tmp_path):
    candidate,pilot=fixture(tmp_path);value=json.loads(pilot.read_text())
    value['training_ranks']=2
    for state in value['initialization'].values():
        state.update(cursors=[0]*2,optimizers=[dict(fresh=True)]*2)
    write(pilot,value)
    assert promote(candidate,pilot)['ready_for_long_run']
    value['initialization']['critic']['optimizers'].pop();write(pilot,value)
    with pytest.raises(ValueError,match='fresh optimizer'):
        promote(candidate,pilot)


@pytest.mark.parametrize('mutation,match',[
    (lambda p:p.update(completed_joint_updates=1),'at least two'),
    (lambda p:p['initialization']['critic'].update(cursors=[1]*4),'cursor zero'),
    (lambda p:p['batches'][0].update(on_policy_passed=False),'Pilot batches'),
    (lambda p:p['critic_publications'][0].update(max_abs_error=.01),'equivalence'),
    (lambda p:p['checkpoint']['actor'].update(optimizer_steps=[1]),'checkpoint'),
    (lambda p:[row.update(successes=0) for row in p['batches']],'both successful'),
    (lambda p:[row.update(successes=row['terminals']) for row in p['batches']],'both successful'),
    (lambda p:p.pop('scientific_rejection'),'Scientifically rejected'),
])
def test_incomplete_pilot_cannot_promote(tmp_path,mutation,match):
    candidate,pilot=fixture(tmp_path);value=json.loads(pilot.read_text());mutation(value);write(pilot,value)
    with pytest.raises(ValueError,match=match): promote(candidate,pilot)


def test_handwritten_long_run_approval_is_rejected(tmp_path):
    candidate,pilot=fixture(tmp_path);result=promote(candidate,pilot)
    result['joint_updates']=999
    path=tmp_path/'promoted.json';write(path,result)
    with pytest.raises(ValueError,match='exactly reproduce'):
        require_long_run(path)
