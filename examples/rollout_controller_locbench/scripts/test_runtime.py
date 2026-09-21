import asyncio
from dataclasses import replace
import json
from pathlib import Path
import pickle
import subprocess
import sys

import pytest
from runtime import context, make_world, TASK_LIMIT
from collect_warmup import MonteCarloPolicy, RecordedPolicy, DRAW, REQUEST_SEED
from examples.locbench.dataset import Case
from examples.locbench.metrics import Gold
from examples.locbench.repository import Repository
from step_controller import Policy, PolicyFormat, ChatCodec, QwenXMLToolCallParser, SamplingParams
from step_controller.harness import Turn, StepResult
sys.path.append('/weka/scratch/jhu/bvandur1/zjiang31/rollout-controller')
from tests.helpers import FakeChatTokenizer, ScriptedGenerator


@pytest.fixture
def world(tmp_path):
    def git(*args):return subprocess.check_output(['git','-C',str(tmp_path),*args],text=True).strip()
    git('init','-q');git('config','user.email','test@example.invalid');git('config','user.name','Test')
    (tmp_path/'a.py').write_text('def f(): pass\n');git('add','.');git('commit','-qm','base')
    repo=Repository(tmp_path,git('rev-parse','HEAD'))
    case=Case('test__repo-1','test/repo',repo.commit,'Locate the issue.',Gold(frozenset({'PRIVATE_GOLD'}),frozenset()))
    text='<tool_call><function=submit><parameter=locations>["a.py"]</parameter></function></tool_call>'
    gen=ScriptedGenerator([text],logprobs=-.1)
    tokenizer=FakeChatTokenizer()
    policy=Policy(format=PolicyFormat('test',ChatCodec(tokenizer),QwenXMLToolCallParser()),
        complete=gen.agenerate_tokens,version='test')
    return (*make_world(case,repo,policy),tokenizer)


@pytest.mark.asyncio
async def test_context_private_labels_and_global_branch_horizon(world):
    runner,workspace,prompt,tools,tokenizer=world
    root=await runner.start(prompt,workspace=workspace)
    class Serializer:
        def apply_chat_template(self,messages,**kwargs):return json.dumps(messages)
    text=context(root,tools,Serializer())
    assert 'PRIVATE_GOLD' not in text and 'expected final file recall@5' in text
    assert 'remaining_steps' in text and len(tools)==4
    # A resumed search must not obtain another full task budget.
    transition=StepResult(next_state=root.state)
    boundary=replace(root,turns=tuple(Turn(prefix=(),transition=transition) for _ in range(TASK_LIMIT)))
    final=await runner.advance(boundary)
    assert final.done and final.truncated and final.state.submitted
    assert final.state.calls==()
    assert pickle.loads(pickle.dumps(root)).state==root.state


@pytest.mark.asyncio
async def test_sampling_streams_are_per_episode_and_bound_only_reply(monkeypatch):
    import collect_warmup as module
    monkeypatch.setattr(module,'CONTEXT_LIMIT',40)
    seen=[]
    async def record(self,prefix,params):
        await asyncio.sleep(0)
        seen.append((REQUEST_SEED.get(),prefix,params.max_tokens))
        return None
    monkeypatch.setattr(RecordedPolicy,'agenerate_tokens',record)
    policy=MonteCarloPolicy(model='test',format=PolicyFormat('test',ChatCodec(FakeChatTokenizer()),QwenXMLToolCallParser()),
        base_url='http://127.0.0.1:1/v1',api_key='EMPTY',default_params=SamplingParams(max_tokens=16))
    async def episode(namespace):
        token=DRAW.set((namespace,0))
        try:
            await policy.agenerate_tokens((1,)*36)
            await policy.agenerate_tokens((1,)*36)
        finally:DRAW.reset(token)
    await asyncio.gather(episode('q/0'),episode('q/1'))
    assert len({seed for seed,_,_ in seen})==4
    assert all(prefix==(1,)*36 and maximum==4 for _,prefix,maximum in seen)
    await episode('q/0')
    assert seen[-2][0]==seen[0][0]
    with pytest.raises(ValueError,match='Input exceeds'):
        await policy.agenerate_tokens((1,)*40)
    policy.close()


def test_repository_disjoint_split_and_tuning_exclusion():
    from runtime import EXPERIMENT
    split=json.loads((EXPERIMENT/'data/split.json').read_text())
    groups=[set(split[lane]) for lane in ['train','development','test']]
    repos=[set(split['repositories'][lane]) for lane in ['train','development','test']]
    assert sum(map(len,groups))==560
    assert all(not a&b for i,a in enumerate(groups) for b in groups[i+1:])
    assert all(not a&b for i,a in enumerate(repos) for b in repos[i+1:])
    assert set(split['tuning_ids'])<=groups[0]
