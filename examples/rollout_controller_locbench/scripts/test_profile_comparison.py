from copy import deepcopy
from types import SimpleNamespace
import pytest
import ppo_runtime as runtime
from profile_pilot import repeated_calls
from profile_gate import evaluate


def test_profile_changes_only_requested_limits_and_requires_transition():
    baseline=runtime.environment()
    for name,limit in [('reply8',32768),('compact16-reply8',16384)]:
        value=runtime.environment(name)
        assert set(k for k in value if value[k]!=baseline[k])==({'profile','actor_reply_limit'} if name=='reply8' else {'profile','actor_reply_limit','prompt_limit'})
        assert value['actor_reply_limit']==8192 and value['prompt_limit']==limit
        assert value['fold_reply_limit']==16384 and value['summary_target_tokens']==2048
        with pytest.raises(ValueError):runtime.transition(baseline,name)
        assert runtime.transition(baseline,name,allow=True)['current']==value
    altered=deepcopy(baseline);altered['call_budget']=10
    with pytest.raises(ValueError):runtime.identify(altered)
    with pytest.raises(ValueError):runtime.transition(runtime.environment('reply8'),'compact16-reply8',allow=True)


def test_trigger_replaced_per_world_without_changing_fold_parameters(monkeypatch):
    from step_controller.harness.compaction.triggers import AnyOf,PromptTokens,StateCounter
    runners=[]
    def original(*_):
        runner=SimpleNamespace(_compactor=SimpleNamespace(trigger=AnyOf(PromptTokens(32768),StateCounter()),fold_reply_limit=16384))
        runners.append(runner);return runner,None,None,None
    monkeypatch.setattr(runtime,'original_world',original)
    old=runtime.make_world(None,None,None)[0];new=runtime.make_world(None,None,None,profile='compact16-reply8')[0]
    state=SimpleNamespace(calls_remaining=20,calls_max=20)
    assert not old._compactor.trigger.fires(range(20000),state)
    assert new._compactor.trigger.fires(range(20000),state)
    assert not new._compactor.trigger.fires(range(15000),state)
    assert new._compactor.trigger.fires([],SimpleNamespace(calls_remaining=0,calls_max=20))
    assert new._compactor.fold_reply_limit==old._compactor.fold_reply_limit


def test_repetition_tracks_pre_fold_evidence_and_ignores_submit():
    a=dict(name='read',arguments={'path':'a','start':1,'end':10})
    b=dict(name='read',arguments={'path':'a','start':11,'end':20})
    event=lambda n,tag,calls:dict(turn=n,tag=tag,calls=calls)
    m=repeated_calls([event(0,'task',[a]),event(1,'fold',[]),event(2,'task',[b,a]),event(3,'task',[b]),event(4,'task',[dict(name='submit',arguments={})])])
    assert (m['tool_calls'],m['repeated_calls'],m['post_fold_repeated_calls'])==(4,2,1)


def pilot_rows():
    rows=[]
    for profile in runtime.PROFILES:
        for q in range(4):
            for sample in range(2):
                short=profile=='compact16-reply8'
                rows.append(dict(profile=profile,query_id=str(q),sample=sample,seconds=90 if short else 100,
                    reward=.5,total_turns=30,tool_calls=29,repeated_calls=1,post_fold_repeated_calls=1,
                    output_tokens=1000,input_tokens=300000,total_region_tokens=20000 if short else 40000,
                    max_region_tokens=20000 if short else 40000,fold_prompt_tokens=[18000] if short else [35000],
                    horizon_hit=False,max_task_reply=1000,threshold_folds=2 if short else 0))
    return rows


def test_gate_rejects_regressions_and_incomplete_evidence():
    rows=pilot_rows();assert evaluate(rows)['selected_profile']=='compact16-reply8'
    for key,value in [('reward',0),('seconds',200),('repeated_calls',10),('post_fold_repeated_calls',10),('total_turns',80),('max_region_tokens',45000),('horizon_hit',True)]:
        bad=deepcopy(rows)
        for r in bad:
            if r['profile']=='compact16-reply8':r[key]=value
        assert not evaluate(bad)['passed'],key
    with pytest.raises(ValueError):evaluate(rows[:-1])


@pytest.mark.asyncio
async def test_task_cap_does_not_shrink_fold_allowance(monkeypatch):
    from collect_ppo import BoundedSlimePolicy
    from step_controller.generation.slime import SlimePolicy
    from step_controller.generation import SamplingParams
    observed=[]
    async def fake(self,prefix_tokens,sampling_params=None):
        observed.append(sampling_params.max_tokens)
    monkeypatch.setattr(SlimePolicy,'agenerate_tokens',fake)
    policy=object.__new__(BoundedSlimePolicy)
    policy.default_params=SamplingParams(max_tokens=8192,temperature=.6)
    await policy.agenerate_tokens([0]*100)
    await policy.agenerate_tokens([0]*20000,SamplingParams(max_tokens=16384,temperature=.7))
    await policy.agenerate_tokens([0]*98000)
    assert observed==[8192,16384,304]
