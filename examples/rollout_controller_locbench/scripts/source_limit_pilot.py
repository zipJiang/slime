"""Fresh paired no-compaction runs: determine whether shorter reads exhaust budgets."""
import argparse,asyncio,json,time
from pathlib import Path
from synthetic_data import E,MODEL,CACHE,read,write,save,Workflow,load_cases,prepare,PolicyFormat,SamplingParams,metrics,score,QUESTIONS,digest
from synthetic_runtime import make_world,environment
async def main(a):
 from transformers import AutoTokenizer
 a.output.mkdir(exist_ok=False,parents=True);infra=read(a.infra);split=read(E/'data/split.json')
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);fmt=PolicyFormat.resolve(MODEL,tokenizer=tok,profile='qwen_xml')
 cases={c.id:c for lane in ('train','development') for c in load_cases(E/'data'/f'{lane}.jsonl')}
 # Eight pre-existing long training sources and four fixed held-out stress cases.
 train=sorted({r['case_id'] for r in read(E/'operations/synthetic-source-audit-v1.json') if r['lane']=='train'})[:8]
 questions=train+QUESTIONS;flow=Workflow(a.output,fmt,infra,cases,split);sems=[asyncio.Semaphore(3) for _ in infra['actor_urls']]
 write(a.output/'manifest.json',dict(questions=questions,train_questions=train,development_questions=QUESTIONS,read_limits=[120,60],compaction=False,task_limit=80,reply_limit=8192,context_limit=98304,candidate=environment(),selection='Eight known long successful training questions plus four difficult development cases; diagnostic, not an unbiased benchmark',sources={str(p):digest(p) for p in (Path(__file__),E/'scripts/synthetic_runtime.py',E/'scripts/synthetic_data.py')}))
 async def one(q,limit,index):
  async with sems[index%len(sems)]:
   out=a.output/str(limit)/q;policy=flow.policy('read-limit-pilot/'+q,out,index);state=None;started=time.monotonic()
   try:
    case=cases[q];repo=await asyncio.to_thread(prepare,CACHE,case.repo,case.base_commit);runner,ws,prompt,tools=make_world(case,repo,policy,compact=False,read_limit=limit)
    runner._sampling_params=SamplingParams(max_tokens=8192,temperature=.6,top_p=.95,top_k=20);state=await runner.start(prompt,workspace=ws);context_stop=False
    while not state.done and not state.truncated:
     if len(policy.prepare(state.messages,tools=tools).tokens)>88000:
      context_stop=True;state=await runner.finish(state);break
     state=await runner.advance(state);write(out/'progress.json',dict(turns=state.turns_taken,seconds=time.monotonic()-started,time=time.time()))
    if not state.done:state=await runner.finish(state)
    data=metrics(state,policy);stats={k:v for k,v in data.items() if k!='events'}
    result=dict(query_id=q,lane='train' if q in train else 'development',read_limit=limit,seconds=time.monotonic()-started,reward=score(state.state.locations,case.gold)['reward'],reward_ceiling=min(5,len(case.gold.files))/len(case.gold.files),submitted=state.state.submitted,horizon_hit=state.turns_taken>=80,context_stop=context_stop,tool_error_count=sum(str(m.get('content','')).lower().startswith('error:') for t in state.turns if t.transition for m in t.transition.messages),**stats)
    save(out/'trace.pkl.gz',state.snapshot());write(out/'events.json',data['events']);write(out/'complete.json',result);print(json.dumps(result),flush=True);return result
   except Exception as e:
    result=dict(query_id=q,read_limit=limit,error=repr(e),time=time.time());write(out/'failure.json',result)
    if state is not None:save(out/'partial.pkl.gz',state.snapshot())
    return result
   finally:await policy.aclose()
 rows=await asyncio.gather(*(one(q,limit,i) for i,q in enumerate(questions) for limit in (120,60)))
 write(a.output/'results.json',rows);write(a.output/'complete.json',dict(time=time.time(),trajectories=len(rows),failures=sum('error' in r for r in rows)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--infra',type=Path,required=True);p.add_argument('--output',type=Path,required=True);asyncio.run(main(p.parse_args()))
