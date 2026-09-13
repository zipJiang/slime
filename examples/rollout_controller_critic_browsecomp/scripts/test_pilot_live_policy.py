"""Exercise live policy provenance through search and sample preparation."""
import os
from pathlib import Path
import subprocess
import sys


def test_live_policy_versions_survive_both_search_passes():
    experiment=Path(__file__).resolve().parents[1]
    # A fresh interpreter avoids the outer Slime examples package shadowing the
    # pinned harness's examples package, just as the real collector does.
    code=r'''
import asyncio
import gzip
import pickle
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from transformers import AutoTokenizer
from pilot_collect import (make_live_runner, two_pass_search, serialize_context,
    ChatCodec, PolicyFormat, SamplingParams, SlimePolicy, Runtime, RolloutConfig,
    RewardConfig, AsyncRewardModel, RewardResult, prepare_samples, DirectBranchTdEstimator,
    to_samples, split_targets, RetrievalArchive, save_tree, collect_questions)

checkpoint='/weka/projects/bvandur1/zjiang31/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a'
tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)

class Value(AsyncRewardModel):
    async def ascore(self,context): return RewardResult(score=.5)

async def exercise(version):
    spend=Counter(input_tokens=0,output_tokens=0,generations=0)
    async def post(url,payload):
        answer='Paris' if spend['generations']%2==0 else 'London'
        reply='<tool_call>\n<function=submit>\n<parameter=answer>'+answer+'</parameter>\n</function>\n</tool_call>'
        tokens=tokenizer.encode(reply,add_special_tokens=False)
        spend.update(input_tokens=len(payload['input_ids']),output_tokens=len(tokens),generations=1)
        return dict(text=reply,meta_info=dict(output_token_logprobs=[[-.1,t,None] for t in tokens],
            finish_reason={'type':'stop'},weight_version='1'))
    codec=ChatCodec(tokenizer)
    policy=SlimePolicy(post,'http://fixture/generate',model='Qwen/Qwen3.5-9B',
        format=PolicyFormat.resolve('Qwen/Qwen3.5-9B',tokenizer=tokenizer,profile='qwen_xml'),
        version=version,default_params=SamplingParams(max_tokens=128,temperature=1.,logprobs=1))
    runner,workspace,prompt,tools=make_live_runner(SimpleNamespace(question='Name a city.'),
        codec,policy,archive=RetrievalArchive(None))
    assert runner.policy is policy and runner.policy.trainable
    nested=runner.derive(env=runner._env,system_prompt='Summarize.',max_steps=1,finish_prompt=None)
    assert nested.policy is policy
    rc=RewardConfig(value_version='critic-test',value_prior_strength=1.,config_id='smoke')
    runtime=Runtime(runner=runner,value_model=Value(),
        value_serialize=lambda p:serialize_context(p,tools,tokenizer),
        config=RolloutConfig(reward_config=rc))
    async def judge(answer): return float(answer=='Paris')
    state,passes=await two_pass_search(prompt,workspace,runtime,spend,judge,
        pass_tokens=1,max_attempts=2,concurrency=1)
    assert len(passes)==2 and not state.stats.get('failures',0)
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'tree.pkl.gz'
        save_tree(path,state)
        with gzip.open(path,'rb') as stream: state=pickle.load(stream)
    prepared=prepare_samples(state,estimator=DirectBranchTdEstimator(),
        reward_config=rc,behavior_version=version)
    actor,critic=split_targets(to_samples(prepared,group_index=0))
    assert actor and critic
    assert sum(sum(r['loss_mask']) for r in actor)==spend['output_tokens']
    for node in state.nodes.values():
        for turn in node.payload.turns:
            if turn.tokens:
                assert turn.exact and set(turn.logprobs)=={version}

async def main():
    for version in ['actor-0000','actor-0001']:
        await exercise(version)
    finished=[]
    async def fail(): raise ValueError('preparation failed')
    async def finish():
        await asyncio.sleep(.01)
        finished.append(True)
        return {'saved':True}
    try:
        await collect_questions([fail(),finish()])
    except ExceptionGroup as exc:
        assert len(exc.exceptions)==1 and isinstance(exc.exceptions[0],ValueError)
    else:
        raise AssertionError('failure was swallowed')
    assert finished==[True]
    assert await collect_questions([finish()])==[{'saved':True}]
asyncio.run(main())
print('live policy search/preparation passed for both actor versions')
'''
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([
        str(experiment/'snapshots/harness'),str(experiment/'scripts')]))
    result=subprocess.run([sys.executable,'-c',code],cwd=experiment/'snapshots/harness',
        env=env,capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
