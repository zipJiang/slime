import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from recipe import RECIPE_ID
from warmup_replay import restore_batch


def fixture(tmp_path):
    old=tmp_path/'old'; old.mkdir()
    knobs=dict(hf_checkpoint='base',seed=1,ppo_prior_strength=1,ppo_pass_tokens=140000,
               ppo_search_concurrency=4,rollout_batch_size=6)
    (old/'recipe.json').write_text(json.dumps(dict(recipe_id=RECIPE_ID,
                                                arguments=dict(**knobs,load='base'))))
    source=old/'rollouts/train-0001';source.mkdir(parents=True)
    frozen=dict(policy_version='actor-0000',value_version='critic-0001',server_weight_version='1')
    contract=dict(**frozen,recipe_id=RECIPE_ID,case_keys=['a','b'],estimator='refined_td',
        critic_supervision='nonterminal_only',horizon='shared_task_turns_plus_one_forced_submission',evaluation=False)
    (source/'contract.json').write_text(json.dumps(contract))
    (source/'summary.json').write_text('{}')
    (source/'target-replay-audit.json').write_text(json.dumps(dict(passed=True,scope='whole_batch')))
    args=SimpleNamespace(**knobs,ppo_replay_warmup_run=str(old),start_rollout_id=1,num_critic_only_steps=10)
    return args,frozen,source


def test_restore_keeps_source_provenance_and_contract(tmp_path):
    args,frozen,source=fixture(tmp_path)
    out=tmp_path/'restored'
    result=restore_batch(args,1,['a','b'],frozen,out)
    assert result['actor_unchanged']
    assert json.loads((out/'contract.json').read_text())['recipe_id']==RECIPE_ID
    assert (out/'original-contract.json').read_bytes()==(source/'contract.json').read_bytes()
    assert (out/'contract.json').read_bytes()==(source/'contract.json').read_bytes()
    assert not (out/'target-replay-audit.json').exists()


def test_wrong_cursor_and_unaudited_data_are_rejected(tmp_path):
    args,frozen,source=fixture(tmp_path)
    with pytest.raises(ValueError,match='cursor'):
        restore_batch(args,1,['b','a'],frozen,tmp_path/'bad')
    (source/'target-replay-audit.json').write_text(json.dumps(dict(passed=False,scope='whole_batch')))
    with pytest.raises(ValueError,match='audit'):
        restore_batch(args,1,['a','b'],frozen,tmp_path/'bad')


def test_replay_cannot_cross_actor_training_boundary(tmp_path):
    args,frozen,source=fixture(tmp_path)
    args.num_critic_only_steps=1
    with pytest.raises(ValueError,match='critic-only'):
        restore_batch(args,1,['a','b'],frozen,tmp_path/'bad')


def test_optimizer_replay_allowance_keeps_lineage_and_probability_checks():
    from warmup_replay import compare_optimizer_replay
    good=dict(version='critic-0002',scores=[.5,.3])
    close=dict(version='critic-0002',scores=[.502,.3])
    assert compare_optimizer_replay(good,close,tolerance=.01)['passed']
    assert not compare_optimizer_replay(good,close,tolerance=1e-5)['passed']
    assert not compare_optimizer_replay(good,dict(version='critic-0002',scores=[.52,.3]),tolerance=.01)['passed']
    for bad in [dict(version='critic-0001',scores=[.5,.3]),
                dict(version='critic-0002',scores=[.5]),
                dict(version='critic-0002',scores=[float('nan'),.3]),
                dict(version='critic-0002',scores=[1.1,.3])]:
        with pytest.raises(ValueError): compare_optimizer_replay(good,bad,tolerance=.01)
    for tolerance in [-1,float('nan'),float('inf')]:
        with pytest.raises(ValueError): compare_optimizer_replay(good,close,tolerance=tolerance)
