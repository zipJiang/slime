from collections import Counter
import json
import pickle
import pytest
from test_runtime import world
from collect_ppo import search, BoundedSlimePolicy, SlimePolicy, BatchedValueClient
from step_controller import SamplingParams, DirectBranchTdEstimator
from step_controller.loop import Runtime
from step_controller.config import RolloutConfig
from step_controller.reward import AsyncRewardModel, RewardResult
from step_controller.reward.config import RewardConfig
from step_controller.preparation import prepare_samples
from step_controller.export import to_samples


@pytest.mark.asyncio
async def test_two_pass_native_tree_is_trainable_and_excludes_terminal_critic(world):
    runner,workspace,prompt,tools,tokenizer=world
    from tests.helpers import ScriptedGenerator
    text='<tool_call><function=submit><parameter=locations>["a.py"]</parameter></function></tool_call>'
    spend=Counter(input_tokens=0,output_tokens=0,generations=0)
    async def complete(prefix,params):
        generated=await ScriptedGenerator([text],logprobs=-.1).agenerate_tokens(prefix,params)
        spend.update(input_tokens=len(prefix),output_tokens=len(generated.tokens),generations=1)
        return generated
    runner.policy._complete=complete
    class Value(AsyncRewardModel):
        async def ascore(self,context):
            assert context=='live root'
            return RewardResult(score=.5)
    rc=RewardConfig(value_version='test-value')
    runtime=Runtime(runner=runner,value_model=Value(),value_serialize=lambda _: 'live root',
                    config=RolloutConfig(reward_config=rc))
    snapshots=[]
    tree,passes=await search(prompt,workspace,runtime,spend,pass_tokens=1,max_attempts=2,concurrency=1,
        save_pass=lambda index,state,passes:snapshots.append(pickle.dumps(state)))
    assert len(passes)==2 and len(snapshots)==2
    assert all(p['cost']['generations']>=1 for p in passes)
    prepared=prepare_samples(pickle.loads(pickle.dumps(tree)),estimator=DirectBranchTdEstimator(),
        reward_config=rc,behavior_version='test')
    records=to_samples(prepared,group_index=0)
    assert prepared.actor and prepared.critic
    assert all(len(r['tokens'])==len(r['loss_mask'])==len(r['logprobs'])
               for r in records if r['metadata']['lane']=='actor')


@pytest.mark.asyncio
async def test_slime_context_bound_only_changes_reply(monkeypatch):
    import collect_ppo
    monkeypatch.setattr(collect_ppo,'CONTEXT_LIMIT',10)
    seen=[]
    async def generate(self,prefix,params):seen.append((prefix,params));return 'result'
    monkeypatch.setattr(SlimePolicy,'agenerate_tokens',generate)
    policy=object.__new__(BoundedSlimePolicy)
    policy.default_params=SamplingParams(max_tokens=20)
    assert await policy.agenerate_tokens((1,)*7)=='result'
    assert seen[0][0]==(1,)*7 and seen[0][1].max_tokens==3
    with pytest.raises(ValueError,match='Input exceeds'):
        await policy.agenerate_tokens((1,)*10)


@pytest.mark.asyncio
async def test_value_client_dedup_and_version_failure_are_explicit():
    class Response:
        def raise_for_status(self):pass
        def json(self):return dict(version='wrong',scores=[.4])
    class Http:
        calls=0
        async def post(self,*args,**kwargs):self.calls+=1;return Response()
    http=Http();client=BatchedValueClient(http,'http://critic','v1')
    with pytest.raises(ValueError,match='version mismatch'):
        await client.ascore_batch(['same','same'])
    assert http.calls==1


@pytest.mark.asyncio
async def test_model_compaction_failure_is_terminal_with_exact_tokens_but_transport_raises(world,monkeypatch):
    from runtime_v2 import make_world, compaction_terminal
    from step_controller.harness import Turn
    from step_controller.harness.compaction.base import IncompleteCompactionError
    from examples.locbench.env import LocBenchRunner
    runner,workspace,prompt,tools,tokenizer=world
    root=await runner.start(prompt,workspace=workspace)
    rejected=Turn(prefix=(1,2),tokens=(3,4),logprobs={'test':(-.2,-.3)},tag='fold')
    terminal=compaction_terminal(root,(rejected,),('unclosed_prefix_reasoning',),'test')
    assert terminal.done and terminal.reward==0 and not terminal.state.submitted
    assert terminal.messages==root.messages and terminal.workspace==root.workspace
    assert terminal.turns[-2]==rejected and not terminal.turns[-1].tokens
    assert terminal.turns[-2].transition is None
    assert terminal.turns[-1].transition.done
    # Exercise the actual v2 runner factory with the original tiny repository.
    from examples.locbench.dataset import Case
    from examples.locbench.metrics import Gold
    repository=next(t for t in runner._env.tools if t.name=='list').repo
    case=Case('test__repo-1','test/repo',repository.commit,'Locate issue',runner._env._gold)
    live,_,_,_=make_world(case,repository,runner.policy)
    async def reject(self,rs):
        raise IncompleteCompactionError(turns=(rejected,),reasons=('unclosed_prefix_reasoning',))
    monkeypatch.setattr(LocBenchRunner,'compact',reject)
    assert (await live.compact(root)).turns==terminal.turns
    async def transport(self,rs):raise ConnectionError('server lost')
    monkeypatch.setattr(LocBenchRunner,'compact',transport)
    with pytest.raises(ConnectionError):await live.compact(root)
