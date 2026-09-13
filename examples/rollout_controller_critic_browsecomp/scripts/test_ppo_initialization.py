import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from ppo_initialization import zero_warmup_role_arguments
from provenance import function_sha256
from warmstart_candidate import build_candidate


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def candidate_fixture(tmp_path):
    base=tmp_path/'base';base.mkdir()
    context_source=tmp_path/'adapter.py';context_source.write_text('def context(state):\n    return str(state)\n')
    training=tmp_path/'training';native=training/'native';checkpoint=native/'iter_0000015'
    checkpoint.mkdir(parents=True)
    write(training/'complete.json',dict(better_than_initial=True,better_than_constant=True,
        portable_inference_passed=True,inference=str(training/'inference'),
        native_checkpoint=str(native),native_iteration=15))
    write(training/'native-validated.json',dict(updates=16,checkpoint=str(native),iteration=15))
    write(training/'reload-audit.json',dict(passed=True,cursors=[0]*4,
        optimizers=[dict(fresh=True)]*4,finetune=True,no_load_optim=True,no_load_rng=True))
    write(training/'inference-audit.json',dict(comparison=dict(passed=True)))
    write(native/'iter_0000015-readback.json',dict(checkpoint=str(checkpoint),role='critic',
        common_state=dict(iteration=15),expected_optimizer_steps=16,optimizer_steps=[16],
        full_storage_read=True,finite_tensors=True))
    write(training/'inference/manifest.json',dict(version='v1'))
    write(training/'dataset-inventory.json',{});write(training/'recipe.json',{})
    write(training/'sampling-audit.json',dict(passed=True,traces=640,samples_per_question=4,
        distinct_within_question_at_every_call=True))
    write(training/'storage-preflight.json',dict(passed=True))
    candidate=build_candidate(training,base_model=base,context_source_file_sha256='a'*64,
                              context_function_sha256=function_sha256(context_source,'context'))
    path=training/'warmstart-candidate.json';write(path,candidate)
    return base,path,native,context_source


def args(base, **changes):
    values=dict(load=str(base),ckpt_step=None,start_rollout_id=None,
                num_critic_only_steps=0,finetune=True,no_load_optim=True,no_load_rng=True)
    values.update(changes)
    return SimpleNamespace(**values)


def test_separates_fresh_actor_from_pretrained_critic(tmp_path):
    base,candidate,native,context_source=candidate_fixture(tmp_path)
    actor,critic,lineage=zero_warmup_role_arguments(args(base),candidate,context_source)
    assert Path(actor.load)==base and actor.ckpt_step is None
    assert Path(critic.load)==native and critic.ckpt_step==15
    assert critic.finetune and critic.no_load_optim and critic.no_load_rng
    assert actor.start_rollout_id==critic.start_rollout_id==0
    assert lineage['actor_class']=='fresh_actor.FreshStartActor'
    assert lineage['context_function_sha256']==function_sha256(context_source,'context')


@pytest.mark.parametrize('changes,match',[
    ({'num_critic_only_steps':1},'zero critic warmup'),
    ({'start_rollout_id':1},'rollout zero'),
    ({'ckpt_step':15},'cannot inherit'),
])
def test_rejects_ambiguous_or_nonzero_start(tmp_path,changes,match):
    base,candidate,_,context_source=candidate_fixture(tmp_path)
    with pytest.raises(ValueError,match=match):
        zero_warmup_role_arguments(args(base,**changes),candidate,context_source)


def test_rejects_actor_from_a_different_base(tmp_path):
    base,candidate,_,context_source=candidate_fixture(tmp_path)
    other=tmp_path/'other';other.mkdir()
    with pytest.raises(ValueError,match='base snapshot'):
        zero_warmup_role_arguments(args(other),candidate,context_source)


def test_rejects_different_context_implementation(tmp_path):
    base,candidate,_,context_source=candidate_fixture(tmp_path)
    context_source.write_text('def context(state):\n    return repr(state)\n')
    with pytest.raises(ValueError,match='context function differs'):
        zero_warmup_role_arguments(args(base),candidate,context_source)
