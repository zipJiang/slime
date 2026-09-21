from types import SimpleNamespace
import pytest
import ppo_protocol
import ppo_runtime
from runtime_v2 import MODEL,contract
from synthetic_runtime import environment


def test_fresh_imitation_ppo_requires_matched_actor_reference_and_critic_profile(tmp_path,monkeypatch):
    path=tmp_path/'candidate.json';path.write_text('{}')
    model=str(tmp_path/'imitation-model')
    value=dict(actor_identity=dict(model=model),environment=environment(),
        native_checkpoint=str(tmp_path/'critic-native'),native_iteration=5)
    monkeypatch.setattr(ppo_protocol,'candidate',lambda _:value)
    args=SimpleNamespace(hf_checkpoint=model,load=model,ref_load=model,
        loc_collection_profile='compact24-reply8-read60',num_critic_only_steps=0,
        start_rollout_id=None,ckpt_step=None)
    actor,critic,lineage=ppo_protocol.initial_roles(args,path)
    assert actor.load==model and critic.load==value['native_checkpoint'] and critic.ckpt_step==5
    assert actor.no_load_optim and critic.no_load_optim and actor.start_rollout_id==critic.start_rollout_id==0
    assert lineage['actor_base']==model
    args.ref_load=MODEL
    with pytest.raises(ValueError,match='Actor and reference'):ppo_protocol.initial_roles(args,path)
    args.ref_load=model;args.loc_collection_profile='original'
    with pytest.raises(ValueError,match='profile'):ppo_protocol.initial_roles(args,path)
    value.pop('actor_identity');value['environment']=contract()
    args.hf_checkpoint=args.load=args.ref_load=MODEL
    args.loc_collection_profile='compact24-reply8-read60'
    with pytest.raises(ValueError,match='profile'):ppo_protocol.initial_roles(args,path)


def test_new_ppo_world_uses_exact_imitation_environment(monkeypatch):
    import synthetic_runtime
    result=(object(),object(),object(),object())
    monkeypatch.setattr(synthetic_runtime,'make_world',lambda *args:result)
    assert ppo_runtime.environment('compact24-reply8-read60')==environment()
    assert ppo_runtime.identify(environment())=='compact24-reply8-read60'
    assert ppo_runtime.make_world(None,None,None,profile='compact24-reply8-read60') is result
    assert ppo_runtime.environment('original')==contract()
