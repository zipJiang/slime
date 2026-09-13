"""Collect one on-policy BrowserComp tree batch for a zero-warmup PPO pilot."""
import argparse
import asyncio
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from urllib.parse import urlparse

EXPERIMENT=Path(__file__).resolve().parents[1]
ROOT=EXPERIMENT.parents[2]
HARNESS=EXPERIMENT/'snapshots/harness'
inherited=set(os.environ.get('PYTHONPATH','').split(os.pathsep))
sys.path[:]=[str(HARNESS),str(EXPERIMENT/'scripts')]+[
    p for p in sys.path if p not in inherited and p not in (str(ROOT/'slime'),str(ROOT/'rollout-controller'))]

from examples.browsercomp_plus.env import RetrievalArchive, grade, load_cases, make_runner
from examples.browsercomp_plus.pipeline import client_class
from step_controller import ConcurrencyGating, DirectBranchTdEstimator, TokenEntropyAllocator, ValueRefinement
from step_controller.allocation import RolloutLedger
from step_controller.codec import ChatCodec
from step_controller.config import RolloutConfig
from step_controller.export import to_samples
from step_controller.generation import SamplingParams
from step_controller.generation.policy import PolicyFormat
from step_controller.generation.slime import SlimePolicy
from step_controller.loop import Runtime, _expander, _score_root, _value_head, rescore_anchor
from step_controller.preparation import prepare_samples
from step_controller.reward import AsyncRewardModel, RewardResult
from step_controller.reward.config import RewardConfig
from step_controller.scheduler.core.policies import Budget
from step_controller.scheduler.core.scheduler import Scheduler
from step_controller.scheduler.core.tree import SchedulerState, SchedulerView

from collect import context as serialize_context
from context_bound import ContextBoundedCompactor
from semantic_judge import SemanticJudge, contract as judge_contract
from semantic_reward import SemanticRewardExpander
from targets import split_targets
from provenance import function_sha256


RECIPE_ID='browsecomp-zero-warmup-pilot-v1'
JUDGE_CHECKPOINT=Path('/weka/projects/bvandur1/zjiang31/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654')


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def write_rows(path,rows):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with gzip.open(temporary,'wt') as stream:
        for row in rows: stream.write(json.dumps(row,allow_nan=False)+'\n')
    temporary.replace(path)


def validate_infrastructure(args,infrastructure):
    if infrastructure.get('schema')!='browsecomp-zero-warmup-pilot-infrastructure-v1':
        raise ValueError('Unknown pilot infrastructure contract')
    retriever=args.retriever_code.resolve();index=(retriever/'indexes/qwen3-embedding-0.6b').resolve()
    commit=subprocess.check_output(['git','-C',str(retriever),'rev-parse','HEAD'],text=True).strip()
    clean=not bool(subprocess.check_output(
        ['git','-C',str(retriever),'status','--porcelain'],text=True).strip())
    hashes={}
    for path in sorted(index.glob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                hashes[path.name]=hashlib.file_digest(stream,'sha256').hexdigest()
    endpoints={urlparse(args.retriever_url).hostname,urlparse(args.judge_url).hostname}
    gpus=infrastructure.get('gpus',{});required=infrastructure.get('required_gpus',{})
    if (infrastructure.get('retriever_commit')!=commit
            or infrastructure.get('retriever_tree_clean') is not True or not clean
            or infrastructure.get('retriever_client_sha256')!=hashlib.sha256(
                (retriever/'retriever/serve/client.py').read_bytes()).hexdigest()
            or Path(infrastructure.get('retriever_index','')).resolve()!=index
            or infrastructure.get('retriever_index_sha256')!=hashes
            or Path(infrastructure.get('judge_checkpoint','')).resolve()!=JUDGE_CHECKPOINT.resolve()
            or endpoints!={infrastructure.get('ips',{}).get('aux')}
            or required!=dict(train=4,inference=3,aux=2)
            or any(int(gpus.get(role,0))<count for role,count in required.items())):
        raise ValueError('Pilot infrastructure differs from the recorded services or resources')


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


async def two_pass_search(prompt,workspace,runtime,spend,judge,*,pass_tokens,max_attempts,concurrency):
    root=await runtime.runner.start(prompt,workspace=workspace)
    state=SchedulerState.root(root,gating=ConcurrencyGating(concurrency))
    critic=_value_head(runtime);await _score_root(critic,state)
    if state.stats.get('score_failures',0):
        raise RuntimeError('Root prior failed; refusing unscored refinement')
    view=SchedulerView(state);ledger=RolloutLedger(runtime.config.reward_config);ledger.bind(view)
    passes=[]
    for index,allocator in enumerate((TokenEntropyAllocator(window=8,
            version=runtime.runner.policy.version),ValueRefinement())):
        before=dict(spend);old_stats=dict(state.stats);cap=spend['output_tokens']+pass_tokens
        attempts=int(state.stats.get('rollouts',0)+state.stats.get('failures',0))
        async with state.lock: state.regate(ConcurrencyGating(concurrency))
        allocation=allocator.build(view,ledger)
        if allocation is None: raise RuntimeError('Required search pass has no allocation session')
        expander=SemanticRewardExpander(_expander(runtime,f'pass-{index}'),judge)
        await Scheduler(expander=expander,allocation=allocation,
            termination=Budget(max_rollouts=attempts+max_attempts,
                goal=lambda _:spend['output_tokens']>=cap),
            scorer=critic,max_concurrency=concurrency).run_on(state)
        consumed={key:value-before.get(key,0) for key,value in spend.items()}
        passes.append(dict(pass_id=index,allocator=allocator.name,token_target=pass_tokens,
            cost=consumed,output_overshoot=max(0,consumed['output_tokens']-pass_tokens),
            stats={key:value-old_stats.get(key,0) for key,value in state.stats.items()}))
        if state.stats.get('failures',0) or state.stats.get('score_failures',0):
            raise RuntimeError(f'Search failure: {state.failures}; stats={state.stats}')
    await rescore_anchor(runtime,view)
    return state,passes


def request_seed(namespace,group,question,draw):
    key=f'{namespace}/{group}/{question}/{draw}'
    return int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%2**31


async def main(args):
    import httpx
    from transformers import AutoTokenizer
    split=json.loads((EXPERIMENT/'data/split.json').read_text())
    if hashlib.sha256(args.cases.read_bytes()).hexdigest()!=split['cases_sha256']:
        raise ValueError('Private case file differs from frozen split')
    cases=load_cases(args.cases)
    infrastructure=json.loads(args.infrastructure_manifest.read_text())
    validate_infrastructure(args,infrastructure)
    selected=json.loads(args.questions.read_text())
    if (not isinstance(selected,list) or len(selected)!=args.batch_size
            or len(set(selected))!=len(selected) or any(q not in split['train'] for q in selected)
            or any(q in split['test'] for q in selected)):
        raise ValueError('Pilot batch is not a fresh complete training selection')
    args.output.mkdir(parents=True,exist_ok=False)
    contract=dict(recipe_id=RECIPE_ID,checkpoint=str(args.checkpoint.resolve()),model=args.model,
        policy_version=args.policy_version,
        server_weight_version=args.server_weight_version,value_version=args.value_version,
        case_ids=selected,split_sha256=hashlib.sha256((EXPERIMENT/'data/split.json').read_bytes()).hexdigest(),
        cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        pass_tokens=args.pass_tokens,max_pass_attempts=args.max_pass_attempts,
        concurrency=args.concurrency,prior_strength=args.prior_strength,
        estimator='direct_branch_td',temperature=1.,top_p=1.,top_k=-1,
        max_steps=48,prompt_limit=14336,actor_reply_limit=6144,fold_reply_limit=4096,
        call_budget=10,top_k_documents=5,snippet_chars=700,read_chars=6000,
        checkpoint_context_function_sha256=function_sha256(
            EXPERIMENT/'scripts/collect.py','context'),
        judge=judge_contract(),collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        harness_manifest_sha256=hashlib.sha256((HARNESS/'source-manifest.json').read_bytes()).hexdigest(),
        infrastructure_manifest=str(args.infrastructure_manifest.resolve()),
        infrastructure_manifest_sha256=hashlib.sha256(args.infrastructure_manifest.read_bytes()).hexdigest(),
        seed_namespace=args.seed_namespace)
    write_json(args.output/'contract.json',contract)
    if args.prepare_only:
        print(json.dumps(contract,indent=2));return
    tokenizer=AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True)
    codec=ChatCodec(tokenizer);profile=PolicyFormat.resolve(args.model,tokenizer=tokenizer,profile='qwen_xml')
    params=SamplingParams(max_tokens=6144,temperature=1.,top_p=1.,top_k=-1,
        repetition_penalty=1.,logprobs=1)
    judge_sem=asyncio.Semaphore(8);Client=client_class(args.retriever_code)
    async with httpx.AsyncClient(timeout=600.,limits=httpx.Limits(max_connections=128)) as http, \
            Client(args.retriever_url,max_inflight=16,timeout=30,max_retries=3) as retrieval:
        value=BatchedValueClient(http,args.value_url,args.value_version)
        async def one(group,question):
            started=time.monotonic();spend=Counter(input_tokens=0,output_tokens=0,generations=0)
            draw=0;case=cases[question];archive=RetrievalArchive(retrieval)
            async def post(url,payload):
                nonlocal draw
                draw_id,draw=draw,draw+1
                payload['sampling_params']['sampling_seed']=request_seed(args.seed_namespace,group,question,draw_id)
                response=await http.post(url,json=payload);response.raise_for_status();output=response.json()
                meta=output['meta_info'];pairs=meta.get('output_token_logprobs')
                if not pairs or any(p[0] is None or not math.isfinite(float(p[0])) or float(p[0])>1e-5 for p in pairs):
                    raise ValueError('Missing or invalid exact behavior logprobs')
                if str(meta.get('weight_version'))!=args.server_weight_version:
                    raise ValueError('Behavior weight version differs from driver freeze')
                spend.update(input_tokens=len(payload['input_ids']),output_tokens=len(pairs),generations=1)
                return output
            live=SlimePolicy(post,args.url.rstrip('/')+'/generate',model=args.model,
                format=profile,version=args.policy_version,default_params=params)
            runner,replay,workspace,prompt,compactor,tools=make_runner(case,codec,live,
                archive=archive,max_steps=48,compactor_temperature=1.,compactor_top_p=1.)
            runner._compactor=ContextBoundedCompactor(compactor);replay.allow_live=True
            judge=SemanticJudge(http,args.judge_url,case.question,case.answers,semaphore=judge_sem)
            rc=RewardConfig(value_version=args.value_version,value_prior_strength=args.prior_strength,
                config_id=f'browsecomp-ppo-kappa-{args.prior_strength:g}')
            runtime=Runtime(runner=runner,value_model=value,score_timeout=600,
                value_serialize=lambda p:serialize_context(p,tools,tokenizer),
                config=RolloutConfig(reward_config=rc))
            state,passes=await two_pass_search(prompt,workspace,runtime,spend,judge,
                pass_tokens=args.pass_tokens,max_attempts=args.max_pass_attempts,
                concurrency=args.concurrency)
            if archive.failure is not None: raise RuntimeError('Retrieval infrastructure failed')
            prepared=prepare_samples(state,estimator=DirectBranchTdEstimator(),
                reward_config=rc,behavior_version=args.policy_version)
            actor,critic=split_targets(to_samples(prepared,group_index=group))
            if not actor or not critic: raise ValueError('Search produced no actor or critic targets')
            for rows,lane in ((actor,'actor'),(critic,'critic')):
                for row in rows:
                    row['metadata'].update(query_id=question,policy_version=args.policy_version,
                        server_weight_version=args.server_weight_version)
                    if lane=='critic':
                        row['metadata']['terminal_boundary']=state.nodes[row['metadata']['node_id']].payload.done
                        row['metadata']['target_source']='direct_branch_mean'
                write_rows(args.output/f'group-{group:03d}.{lane}.jsonl.gz',rows)
            archive.save(args.output/f'group-{group:03d}.retrieval.json.gz')
            write_json(args.output/f'group-{group:03d}.judge.json',dict(records=judge.records))
            with gzip.open(args.output/f'group-{group:03d}.native.pkl.gz','wb') as stream: pickle.dump(state,stream)
            terminals=[node.payload for node in state.nodes.values() if node.payload.done]
            result=dict(group_index=group,query_id=question,passes=passes,cost=dict(spend),
                actor_spans=len(actor),actor_tokens=sum(sum(r['loss_mask']) for r in actor),
                critic_checkpoints=len(critic),terminal_count=len(terminals),
                terminal_correct=sum(p.reward_outcome for p in terminals),
                terminal_exact=sum(grade(p.state.answer or '',case) for p in terminals),
                judge_calls=len(judge.records),stats=dict(state.stats),seconds=time.monotonic()-started)
            write_json(args.output/f'group-{group:03d}.json',result)
            print(json.dumps(result,allow_nan=False),flush=True);return result
        results=await asyncio.gather(*(one(group,q) for group,q in enumerate(selected)))
        write_json(args.output/'summary.json',dict(recipe_id=RECIPE_ID,
            policy_version=args.policy_version,server_weight_version=args.server_weight_version,
            value_version=args.value_version,results=results,
            cost={key:sum(r['cost'][key] for r in results) for key in ('input_tokens','output_tokens','generations')},
            critic_requests=value.requests,critic_contexts=value.contexts_scored))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--questions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--retriever-code',type=Path,required=True)
    parser.add_argument('--infrastructure-manifest',type=Path,required=True)
    for name in ('url','value-url','judge-url','retriever-url','policy-version','server-weight-version','value-version','seed-namespace'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--model',default='Qwen/Qwen3.5-9B')
    parser.add_argument('--batch-size',type=int,default=6)
    parser.add_argument('--pass-tokens',type=int,default=140000)
    parser.add_argument('--max-pass-attempts',type=int,default=64)
    parser.add_argument('--concurrency',type=int,default=4)
    parser.add_argument('--prior-strength',type=float,default=1.)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.pass_tokens<=0 or args.max_pass_attempts<args.concurrency or args.concurrency<1:
        parser.error('Positive token budget and valid concurrency/attempt cap required')
    asyncio.run(main(args))
