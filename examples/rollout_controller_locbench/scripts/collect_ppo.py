"""Fresh LocBench two-pass search with exact behavior records and saved cutoff evidence."""
import argparse
import asyncio
from collections import Counter
from dataclasses import replace
import gzip
import hashlib
import inspect
import json
import math
from pathlib import Path
import pickle
import sys
import time

from runtime_active import (EXPERIMENT, MODEL, CONTEXT_LIMIT, REPLY_LIMIT, contract,
                     context, digest, verify_harness)
import ppo_runtime
from ppo_runtime import make_world
# Legacy-compatible helper modules above adjust import search paths.  Establish
# one coherent robust package tree now, before binding any harness classes.
from runtime_v3 import activate as activate_robust
activate_robust(force=True)
sys.path.append(str(EXPERIMENT/'snapshots/native-support-v1'))
from examples.locbench.dataset import load_cases
from examples.locbench.repository import prepare
from examples.locbench.metrics import score
from step_controller import ConcurrencyGating, DirectBranchTdEstimator, TokenEntropyAllocator, ValueRefinement
from step_controller.allocation import RolloutLedger
from step_controller.config import RolloutConfig
from step_controller.export import to_samples
from step_controller.generation import SamplingParams
from step_controller.generation.policy import PolicyFormat
from step_controller.generation.slime import SlimePolicy
from step_controller.generation.interfaces import merge_sampling_params
from step_controller.loop import Runtime, _expander, _score_root, _value_head, rescore_anchor
from step_controller.preparation import prepare_samples
from step_controller.reward import AsyncRewardModel, RewardResult
from step_controller.reward.config import RewardConfig
from step_controller.scheduler.core.policies import Budget
from step_controller.scheduler.core.scheduler import Scheduler
from step_controller.scheduler.core.tree import SchedulerState, SchedulerView
from remaining_value import RemainingValueScorer
from targets import split_targets
from efficiency import analyze


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def save(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_bytes(gzip.compress(pickle.dumps(value),compresslevel=1,mtime=0));tmp.replace(path)


def rows(path, values):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_bytes(gzip.compress(''.join(json.dumps(v,allow_nan=False)+'\n' for v in values).encode(),mtime=0))
    tmp.replace(path)


def scheduler_identity(state=None):
    module=sys.modules.get(SchedulerState.__module__)
    current=getattr(module,SchedulerState.__qualname__,None) if module is not None else None
    actual=type(state) if state is not None else SchedulerState
    return dict(matches=actual is current, imported_class_id=id(SchedulerState),
        actual_class_id=id(actual),current_class_id=id(current),module_id=id(module),
        imported_source=inspect.getsourcefile(SchedulerState),actual_source=inspect.getsourcefile(actual),
        current_source=inspect.getsourcefile(current) if current is not None else None,
        module_source=getattr(module,'__file__',None),
        owners=sorted(name for name,value in tuple(sys.modules.items())
            if name.startswith('step_controller')
            and vars(value).get('SchedulerState') is actual),
        sys_path=sys.path[:12])


class BoundedSlimePolicy(SlimePolicy):
    async def agenerate_tokens(self,prefix_tokens,sampling_params=None):
        params=merge_sampling_params(self.default_params,sampling_params)
        remaining=CONTEXT_LIMIT-len(prefix_tokens)
        if remaining<=0:raise ValueError('Input exceeds context; no conditioning truncation')
        return await super().agenerate_tokens(prefix_tokens,replace(params,max_tokens=min(params.max_tokens,remaining)))


class BatchedValueClient(AsyncRewardModel):
    def __init__(self,client,url,version,batch_size=16):
        self.client,self.url,self.version=client,url,version
        self.batch_size=batch_size;self.pending=[];self.cache={};self.worker=None
        self.requests=0;self.contexts_scored=0

    async def ascore(self,context):
        return (await self.ascore_batch([context]))[0]

    async def ascore_batch(self,contexts):
        loop=asyncio.get_running_loop();futures=[]
        for context in contexts:
            if context not in self.cache:
                future=loop.create_future();self.cache[context]=future
                self.pending.append((context,future))
            futures.append(self.cache[context])
        if self.worker is None or self.worker.done(): self.worker=asyncio.create_task(self._drain())
        scores=await asyncio.gather(*(asyncio.shield(f) for f in futures))
        return [RewardResult(score=s) for s in scores]

    async def _drain(self):
        await asyncio.sleep(.005)
        while self.pending:
            batch,self.pending=self.pending[:self.batch_size],self.pending[self.batch_size:]
            try:
                response=await self.client.post(self.url.rstrip('/')+'/score',
                    json=dict(contexts=[c for c,_ in batch],version=self.version))
                response.raise_for_status();result=response.json()
                if result['version']!=self.version or len(result['scores'])!=len(batch):
                    raise ValueError('Critic result count/version mismatch')
                scores=[float(s) for s in result['scores']]
                if any(not math.isfinite(s) or not 0<=s<=1 for s in scores):
                    raise ValueError('Critic prediction is not a finite probability')
                self.requests+=1;self.contexts_scored+=len(batch)
                for (_,future),score in zip(batch,scores,strict=True):
                    if not future.done(): future.set_result(score)
            except Exception as exc:
                for _,future in batch:
                    if not future.done(): future.set_exception(exc)


async def search(prompt, workspace, runtime, spend, *, pass_tokens, max_attempts, concurrency, save_pass):
    root=await runtime.runner.start(prompt,workspace=workspace)
    state=SchedulerState.root(root,gating=ConcurrencyGating(concurrency))
    identity=scheduler_identity(state)
    if not identity['matches']:
        raise RuntimeError('Scheduler identity changed before search: '+json.dumps(identity,sort_keys=True))
    scorer=RemainingValueScorer(_value_head(runtime))
    await _score_root(scorer,state)
    if state.stats.get('score_failures',0):raise ValueError('Root critic score failed')
    view=SchedulerView(state);ledger=RolloutLedger(runtime.config.reward_config);ledger.bind(view)
    passes=[]
    for index,allocator in enumerate((TokenEntropyAllocator(window=8,version=runtime.runner.policy.version),ValueRefinement())):
        started=time.monotonic();before=dict(spend);old_stats=dict(state.stats)
        cap=spend['output_tokens']+pass_tokens
        attempts=int(state.stats.get('rollouts',0)+state.stats.get('failures',0))
        async with state.lock:state.regate(ConcurrencyGating(concurrency))
        allocation=allocator.build(view,ledger)
        if allocation is None:raise ValueError('Search pass has no allocation session')
        await Scheduler(expander=_expander(runtime,f'pass-{index}'),allocation=allocation,
            termination=Budget(max_rollouts=attempts+max_attempts,goal=lambda _:spend['output_tokens']>=cap),
            scorer=scorer,max_concurrency=concurrency).run_on(state)
        cost={key:value-before.get(key,0) for key,value in spend.items()}
        passes.append(dict(pass_id=index,allocator=allocator.name,token_target=pass_tokens,cost=cost,
            output_overshoot=max(0,cost['output_tokens']-pass_tokens),seconds=time.monotonic()-started,
            stats={key:value-old_stats.get(key,0) for key,value in state.stats.items()}))
        save_pass(index,state,passes)
        if state.stats.get('failures',0) or state.stats.get('score_failures',0):
            raise RuntimeError(f'Search failure: {state.failures}; stats={state.stats}')
    await rescore_anchor(runtime,view)
    return state,passes


async def run(args):
    verify_harness()
    import httpx
    from transformers import AutoTokenizer
    split=json.loads((EXPERIMENT/'data/split.json').read_text())
    selected=json.loads(args.questions.read_text())
    if (not isinstance(selected,list) or len(selected)!=args.batch_size or len(set(selected))!=len(selected)
            or not set(selected)<=set(split['train']) or set(selected)&(set(split['test'])|set(split['development']))):
        raise ValueError('Search requires a complete, unique training-question batch')
    cases={c.id:c for c in load_cases(EXPERIMENT/'data/train.jsonl')}
    args.output.mkdir(parents=True,exist_ok=False)
    collection_profile=getattr(args,'collection_profile','original')
    env_contract=ppo_runtime.environment(collection_profile)
    recipe=dict(collection_profile_sha256=digest(ppo_runtime.__file__),protocol='locbench-ppo-search-v1',environment=env_contract,checkpoint=str(args.checkpoint.resolve()),
        policy_version=args.policy_version,server_weight_version=args.server_weight_version,
        value_version=args.value_version,case_ids=selected,pass_tokens=args.pass_tokens,
        max_pass_attempts=args.max_pass_attempts,concurrency=args.concurrency,prior_strength=args.prior_strength,
        actor_min_abs_advantage=args.min_abs_advantage,estimator='direct_branch_td',seed_namespace=args.seed_namespace,
        split_sha256=digest(EXPERIMENT/'data/split.json'),cases_sha256=digest(EXPERIMENT/'data/train.jsonl'),
        context_source_sha256=digest(EXPERIMENT/'scripts/runtime_v2.py'),collector_sha256=digest(__file__))
    write(args.output/'contract.json',recipe)
    if args.prepare_only:return
    tokenizer=AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True)
    profile=PolicyFormat.resolve(str(args.checkpoint),tokenizer=tokenizer,profile='qwen_xml')
    params=SamplingParams(max_tokens=env_contract['actor_reply_limit'],temperature=.6,top_p=.95,top_k=20,logprobs=1)
    started=time.monotonic()
    async with httpx.AsyncClient(timeout=1800.,limits=httpx.Limits(max_connections=128)) as http:
        value=BatchedValueClient(http,args.value_url,args.value_version)
        async def one(group,question):
            begin=time.monotonic();spend=Counter(input_tokens=0,output_tokens=0,generations=0);draw=0
            stem=f'group-{group:03d}';case=cases[question]
            async def post(url,payload):
                nonlocal draw
                draw_id,draw=draw,draw+1
                key=f'{args.seed_namespace}/{group}/{question}/{draw_id}'
                payload['sampling_params']['sampling_seed']=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%2**31
                response=await http.post(url,json=payload);response.raise_for_status();output=response.json()
                meta=output['meta_info'];pairs=meta.get('output_token_logprobs')
                if not pairs or any(p[0] is None or not math.isfinite(float(p[0])) or float(p[0])>1e-5 for p in pairs):
                    raise ValueError('Missing or invalid exact behavior logprobs')
                if str(meta.get('weight_version'))!=args.server_weight_version:
                    raise ValueError('Behavior version changed during collection')
                spend.update(input_tokens=len(payload['input_ids']),output_tokens=len(pairs),generations=1)
                return output
            policy=BoundedSlimePolicy(post,args.url.rstrip('/')+'/generate',model=str(args.checkpoint),
                format=profile,version=args.policy_version,default_params=params)
            repository=await asyncio.to_thread(prepare,args.cache,case.repo,case.base_commit)
            runner,workspace,prompt,tools=make_world(case,repository,policy,profile=collection_profile)
            rc=RewardConfig(value_version=args.value_version,value_prior_strength=args.prior_strength,
                config_id=f'locbench-ppo-kappa-{args.prior_strength:g}')
            runtime=Runtime(runner=runner,value_model=value,score_timeout=1800,
                value_serialize=lambda payload:context(payload,tools,tokenizer),config=RolloutConfig(reward_config=rc))
            def save_pass(index,state,passes):
                identity=scheduler_identity(state)
                write(args.output/f'{stem}.pass-{index}.identity.json',identity)
                if not identity['matches']:
                    raise RuntimeError('Scheduler identity changed during search: '+json.dumps(identity,sort_keys=True))
                save(args.output/f'{stem}.pass-{index}.native.pkl.gz',state)
                write(args.output/f'{stem}.progress.json',dict(passes=passes,time=time.time()))
            state,passes=await search(prompt,workspace,runtime,spend,pass_tokens=args.pass_tokens,
                max_attempts=args.max_pass_attempts,concurrency=args.concurrency,save_pass=save_pass)
            save(args.output/f'{stem}.native.pkl.gz',state)
            prepared=prepare_samples(state,estimator=DirectBranchTdEstimator(),reward_config=rc,behavior_version=args.policy_version)
            save(args.output/f'{stem}.prepared.pkl.gz',prepared)
            actor,critic=split_targets(to_samples(prepared,group_index=group,min_abs_advantage=args.min_abs_advantage))
            if not prepared.actor or not critic:raise ValueError('Search produced no trainable source evidence')
            for records,lane in ((actor,'actor'),(critic,'critic')):
                for row in records:
                    row['metadata'].update(query_id=question,policy_version=args.policy_version,
                        server_weight_version=args.server_weight_version)
                    if lane=='critic':
                        row['metadata'].update(terminal_boundary=state.nodes[row['metadata']['node_id']].payload.done,
                            target_source='direct_branch_mean')
                rows(args.output/f'{stem}.{lane}.jsonl.gz',records)
            terminals=[n.payload for n in state.nodes.values() if n.payload.done]
            rewards=[float(p.reward_outcome) for p in terminals]
            for terminal,reward in zip(terminals,rewards,strict=True):
                if score(terminal.state.locations,case.gold)['reward']!=reward:
                    raise ValueError('Search terminal reward differs from deterministic file recall')
            if not terminals:raise ValueError('Search produced no terminal outcomes')
            immediate=[rc.configured_return(n.payload.edge(state.nodes[n.parent_id].payload))
                for n in state.nodes.values() if n.parent_id is not None]
            result=dict(group_index=group,query_id=question,passes=passes,cost=dict(spend),
                actor_edges_before_filter=len(prepared.actor),actor_edges_retained=len({r['metadata']['node_id'] for r in actor}),
                actor_empty=not actor,actor_spans=len(actor),actor_tokens=sum(sum(r['loss_mask']) for r in actor),
                critic_checkpoints=len(critic),terminal_count=len(terminals),terminal_rewards=rewards,
                terminal_correct=sum(rewards),terminal_exact=sum(v==1 for v in rewards),
                zero_immediate_reward_edges=sum(v==0 for v in immediate),stats=dict(state.stats),seconds=time.monotonic()-begin,
                native_sha256=digest(args.output/f'{stem}.native.pkl.gz'),prepared_sha256=digest(args.output/f'{stem}.prepared.pkl.gz'))
            write(args.output/f'{stem}.json',result)
            write(args.output/f'{stem}.cutoffs.json',analyze([prepared]))
            print(json.dumps(result,allow_nan=False),flush=True)
            return result
        results=await asyncio.gather(*(one(g,q) for g,q in enumerate(selected)),return_exceptions=True)
        errors=[r for r in results if isinstance(r,BaseException)]
        if errors:
            write(args.output/'failed.json',dict(errors=[repr(e) for e in errors],time=time.time()))
            raise BaseExceptionGroup('Questions failed after draining in-flight work',errors)
        write(args.output/'summary.json',dict(protocol=recipe['protocol'],policy_version=args.policy_version,
            server_weight_version=args.server_weight_version,value_version=args.value_version,results=results,
            cost={key:sum(r['cost'][key] for r in results) for key in ('input_tokens','output_tokens','generations')},
            critic_requests=value.requests,critic_contexts=value.contexts_scored,seconds=time.monotonic()-started))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=Path(MODEL))
    for name in ('questions','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('/weka/scratch/jhu/bvandur1/zjiang31/rollout-controller/data/locbench/repositories'))
    for name in ('url','value-url','policy-version','server-weight-version','value-version','seed-namespace'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--pass-tokens',type=int,default=32768)
    p.add_argument('--max-pass-attempts',type=int,default=32)
    p.add_argument('--concurrency',type=int,default=4)
    p.add_argument('--prior-strength',type=float,default=1.)
    p.add_argument('--min-abs-advantage',type=float,default=None)
    p.add_argument('--collection-profile',choices=ppo_runtime.COLLECTION_PROFILES,default='original')
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    if (args.batch_size<1 or args.pass_tokens<=0 or args.concurrency<1 or args.max_pass_attempts<args.concurrency
        or not math.isfinite(args.prior_strength) or args.prior_strength<=0
        or (args.min_abs_advantage is not None and (not math.isfinite(args.min_abs_advantage) or args.min_abs_advantage<0))):
        p.error('Invalid search budget, prior, concurrency or cutoff')
    asyncio.run(run(args))


if __name__=='__main__':main()
