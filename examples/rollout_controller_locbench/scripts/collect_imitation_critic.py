"""Fresh LocBench Monte Carlo outcomes with resumable native checkpoints."""
import argparse
import asyncio
from contextvars import ContextVar
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import time

from runtime_v2 import EXPERIMENT, MODEL, CONTEXT_LIMIT, TASK_LIMIT, context, digest, verify_harness
from synthetic_runtime import environment as contract,make_world
from imitation_lineage import require_model
REPLY_LIMIT=8192
from examples.locbench.dataset import load_cases
from examples.locbench.metrics import score
from examples.locbench.pilot import RecordedPolicy, JOURNAL
from examples.locbench.repository import prepare
from step_controller import SamplingParams
from step_controller.generation import PolicyFormat
from step_controller.generation.interfaces import merge_sampling_params

DRAW=ContextVar('locbench_mc_draw')
REQUEST_SEED=ContextVar('locbench_mc_request_seed',default=0)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def save_native(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_bytes(gzip.compress(pickle.dumps(value),compresslevel=1,mtime=0));tmp.replace(path)


class MonteCarloPolicy(RecordedPolicy):
    def _completion_kwargs(self,model,prefix_tokens,params):
        kwargs=super()._completion_kwargs(model,prefix_tokens,params)
        kwargs['seed']=REQUEST_SEED.get()
        return kwargs

    async def agenerate_tokens(self,prefix_tokens,sampling_params=None):
        params=merge_sampling_params(self.default_params,sampling_params)
        remaining=CONTEXT_LIMIT-len(prefix_tokens)
        if remaining<=0:
            raise ValueError('Input exceeds server context; no conditioning truncation')
        if params.max_tokens>remaining:
            params=replace(params,max_tokens=remaining)
        namespace,index=DRAW.get()
        seed=int(hashlib.sha256(f'{namespace}/{index}'.encode()).hexdigest()[:8],16)%2**31
        token=REQUEST_SEED.set(seed)
        DRAW.set((namespace,index+1))
        journal=JOURNAL.get()
        if journal:
            with journal.open('a') as stream:
                stream.write(json.dumps(dict(event='draw',index=index,seed=seed,namespace=namespace))+'\n')
        try:
            return await super().agenerate_tokens(prefix_tokens,params)
        finally:
            REQUEST_SEED.reset(token)


async def run(args):
    verify_harness()
    identity=require_model(args.imitation_training);model=identity['model']
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    split=json.loads((EXPERIMENT/'data/split.json').read_text())
    servers=json.loads(args.servers.read_text())
    manifest=dict(protocol='locbench-imitation-mc-critic-v1',environment=contract(),actor_identity=identity,
        split_sha256=digest(EXPERIMENT/'data/split.json'),
        train_ids=split['train'][:args.train_questions],validation_ids=split['development'][:args.dev_questions],
        samples_per_question=args.samples,seed_namespace='locbench-imitation-critic-mc-v1',
        conditioning='root and every nonterminal fold; exact native snapshot',
        target='observed terminal file_recall@5; failures of infrastructure are not labels',
        sources={str(p):digest(p) for p in [Path(__file__).resolve(),EXPERIMENT/'scripts/runtime_v2.py',EXPERIMENT/'scripts/synthetic_runtime.py',EXPERIMENT/'scripts/imitation_lineage.py']})
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())!=manifest:
            raise ValueError('Collection identity changed; use a new attempt')
    else:write(out/'manifest.json',manifest)
    format=PolicyFormat.resolve(model,profile='qwen_xml')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    cases={c.id:c for lane in ['train','development'] for c in load_cases(EXPERIMENT/'data'/f'{lane}.jsonl')}
    if Path(servers['model']).resolve()!=Path(model).resolve():raise ValueError('Collector servers do not name the verified SFT model')
    policies=[MonteCarloPolicy(model=model,served_model='locbench-qwen35-9b',format=format,
        base_url=url,api_key='EMPTY',timeout=1800,version='locbench-imitation-'+identity['training_complete_sha256'][:12],
        default_params=SamplingParams(temperature=.6,top_p=.95,top_k=20,max_tokens=REPLY_LIMIT,logprobs=1))
        for url in servers['urls']]
    for policy in policies:await asyncio.to_thread(policy.startup_check)
    gates=[asyncio.Semaphore(args.concurrency) for _ in policies]
    pending=[(lane,q,s) for lane,ids in [('train',manifest['train_ids']),('validation',manifest['validation_ids'])]
        for q in ids for s in range(args.samples)]
    pending.sort(key=lambda row:hashlib.sha256('/'.join(map(str,row)).encode()).hexdigest())
    if args.pilot:pending=pending[:args.pilot]

    async def one(index,lane,q,sample):
        path=out/lane/q/f'sample-{sample}.json'
        if path.exists():return
        endpoint=index%len(policies)
        async with gates[endpoint]:
            path.parent.mkdir(parents=True,exist_ok=True)
            partial=path.with_suffix('.partial.pkl.gz')
            saved=pickle.loads(gzip.decompress(partial.read_bytes())) if partial.exists() else None
            state=None;checkpoints=[];advances=0
            draw_token=DRAW.set(saved['draw'] if saved else (f"{manifest['seed_namespace']}/{lane}/{q}/{sample}",0))
            journal_token=JOURNAL.set(path.with_suffix('.generations.jsonl'))
            started=time.time();previous_seconds=saved['seconds'] if saved else 0
            try:
                case=cases[q]
                repo=await asyncio.to_thread(prepare,args.cache,case.repo,case.base_commit)
                runner,workspace,prompt,tools=make_world(case,repo,policies[endpoint])
                runner._sampling_params=SamplingParams(max_tokens=REPLY_LIMIT,temperature=.6,top_p=.95,top_k=20)
                state=saved['state'] if saved else await runner.start(prompt,workspace=workspace)
                checkpoints=saved['checkpoints'] if saved else [state.snapshot()]
                advances=saved['advances'] if saved else 0
                while not state.done and advances<TASK_LIMIT:
                    before=len(state.turns)
                    state=await runner.advance(state)
                    advances+=1
                    if not state.done and state.turns[-1].tag=='fold':checkpoints.append(state.snapshot())
                    save_native(partial,dict(state=state.snapshot(),checkpoints=checkpoints,advances=advances,
                        draw=DRAW.get(),seconds=time.time()-started+previous_seconds))
                    write(path.with_suffix('.progress.json'),dict(turns=state.turns_taken,folds=state.folds,
                        advances=advances,updated=time.time(),endpoint=endpoint))
                    if len(state.turns)==before:break
                if not state.done:state=await runner.finish(state)
                if not state.done:raise ValueError('No terminal outcome')
                metrics=score(state.state.locations,case.gold)
                if state.reward!=metrics['reward']:raise ValueError('Native reward disagrees with submission')
                if not all(t.exact for t in state.turns):raise ValueError('Inexact generation')
                rows=[]
                for i,snapshot in enumerate(checkpoints):
                    text=context(snapshot,tools,tokenizer)
                    tokens=len(tokenizer.encode(text,add_special_tokens=False))+1
                    if snapshot.done or tokens>CONTEXT_LIMIT:raise ValueError('Invalid critic context')
                    rows.append(dict(context=text,target=metrics['reward'],group_index=q,
                        turn=snapshot.turns_taken,folds=snapshot.folds,tokens=tokens,
                        metadata=dict(lane='critic',node_id=f'{sample}/{i}',target_source='monte_carlo_suffix',
                            diagnostics=dict(observations=1,mean_return=metrics['reward']))))
                native=path.with_suffix('.pkl.gz');save_native(native,dict(checkpoints=checkpoints,final=state.snapshot()))
                contexts=path.with_suffix('.contexts.jsonl.gz')
                contexts.write_bytes(gzip.compress(''.join(json.dumps(r)+'\n' for r in rows).encode(),mtime=0))
                write(path,dict(case_id=q,lane=lane,sample=sample,done=True,metrics=metrics,
                    locations=state.state.locations,turns=state.turns_taken,folds=state.folds,
                    checkpoints=len(rows),source_sha256=digest(native),contexts_sha256=digest(contexts),
                    manifest_sha256=digest(out/'manifest.json'),seconds=time.time()-started+previous_seconds))
                partial.unlink(missing_ok=True);path.with_suffix('.failure.json').unlink(missing_ok=True)
                print(json.dumps(dict(event='complete',case=q,lane=lane,sample=sample,reward=metrics['reward'],turns=state.turns_taken)),flush=True)
            except Exception as exc:
                write(path.with_suffix('.failure.json'),dict(error=repr(exc),time=time.time()))
                raise
            finally:
                DRAW.reset(draw_token);JOURNAL.reset(journal_token)

    results=await asyncio.gather(*(one(i,*row) for i,row in enumerate(pending)),return_exceptions=True)
    errors=[x for x in results if isinstance(x,BaseException)]
    if errors:
        write(out/'failed.json',dict(errors=[repr(x) for x in errors],time=time.time()))
        raise RuntimeError(f'{len(errors)} episodes failed after other episodes drained')
    write(out/('pilot-complete.json' if args.pilot else 'collection-complete.json'),dict(traces=len(pending),time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--servers',type=Path,required=True)
    p.add_argument('--imitation-training',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('/weka/scratch/jhu/bvandur1/zjiang31/rollout-controller/data/locbench/repositories'))
    p.add_argument('--train-questions',type=int,default=64)
    p.add_argument('--dev-questions',type=int,default=16)
    p.add_argument('--samples',type=int,default=4)
    p.add_argument('--concurrency',type=int,default=4)
    p.add_argument('--pilot',type=int,default=0)
    asyncio.run(run(p.parse_args()))
