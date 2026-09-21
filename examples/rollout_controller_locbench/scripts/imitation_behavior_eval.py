"""Fresh held-out actor rollouts with self-compaction, before or after SFT."""
import argparse,asyncio,hashlib,json,time
from pathlib import Path
from synthetic_data import E,MODEL,CACHE,read,write,save,LoggedPolicy,load_cases,prepare,PolicyFormat,SamplingParams,metrics,score,QUESTIONS,digest
from synthetic_runtime import make_world,environment
async def main(a):
 from transformers import AutoTokenizer
 a.output.mkdir(exist_ok=False,parents=True);infra=read(a.infra);split=read(E/'data/split.json')
 tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);fmt=PolicyFormat.resolve(MODEL,tokenizer=tokenizer,profile='qwen_xml')
 cases={c.id:c for c in load_cases(E/'data/development.jsonl')}
 extra=sorted(set(split['development'])-set(QUESTIONS),key=lambda q:hashlib.sha256(('loc-sft-behavior-v1/'+q).encode()).hexdigest())[:4];questions=QUESTIONS+extra
 urls=infra['actor_urls'];sems=[asyncio.Semaphore(2) for _ in urls]
 manifest=dict(protocol='locbench-sft-behavior-v1',model=a.model,questions=questions,samples=2,environment=environment(),seed_namespace='loc-sft-behavior-v1',split_sha256=digest(E/'data/split.json'),sources={str(p):digest(p) for p in (Path(__file__),E/'scripts/synthetic_runtime.py',E/'scripts/synthetic_data.py')},scope='Eight development questions, including four previously difficult cases; screening, not test-set performance')
 write(a.output/'manifest.json',manifest)
 async def one(q,sample,index):
  async with sems[index%len(sems)]:
   out=a.output/q/f'sample-{sample}';policy=LoggedPolicy(output=out,key=f'loc-sft-behavior-v1/{q}/{sample}',model=a.model,served_model=a.served_model,format=fmt,base_url=urls[index%len(urls)],api_key='EMPTY',timeout=1800,version=a.label,default_params=SamplingParams(max_tokens=8192,temperature=.6,top_p=.95,top_k=20,logprobs=1));state=None;start=time.monotonic()
   try:
    case=cases[q];repo=await asyncio.to_thread(prepare,CACHE,case.repo,case.base_commit);runner,ws,prompt,tools=make_world(case,repo,policy);runner._sampling_params=SamplingParams(max_tokens=8192,temperature=.6,top_p=.95,top_k=20);state=await runner.start(prompt,workspace=ws)
    while not state.done and not state.truncated:
     before=len(state.turns);state=await runner.advance(state)
     if len(state.turns)==before:raise ValueError('No rollout progress')
     save(out/'partial.pkl.gz',state.snapshot());write(out/'progress.json',dict(turns=state.turns_taken,folds=state.folds,time=time.time()))
    if not state.done:state=await runner.finish(state)
    data=metrics(state,policy);result=dict(query_id=q,sample=sample,label=a.label,reward=score(state.state.locations,case.gold)['reward'],submitted=state.state.submitted,horizon_hit=state.turns_taken+state.folds>=80,seconds=time.monotonic()-start,**{k:v for k,v in data.items() if k!='events'})
    save(out/'trace.pkl.gz',state.snapshot());write(out/'events.json',data['events']);write(out/'complete.json',result);print(json.dumps({k:result[k] for k in ('query_id','sample','reward','total_turns','folds','repeated_calls','seconds')}),flush=True);return result
   except Exception as e:
    result=dict(query_id=q,sample=sample,error=repr(e));write(out/'failure.json',result)
    if state is not None:save(out/'partial.pkl.gz',state.snapshot())
    return result
   finally:await policy.aclose()
 results=await asyncio.gather(*(one(q,s,i) for i,(q,s) in enumerate((q,s) for s in range(2) for q in questions)))
 write(a.output/'results.json',results);write(a.output/'complete.json',dict(time=time.time(),trajectories=len(results),failures=sum('error' in r for r in results)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--infra',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--model',default=MODEL);p.add_argument('--served-model',default='locbench-synthetic-9b');asyncio.run(main(p.parse_args()))
