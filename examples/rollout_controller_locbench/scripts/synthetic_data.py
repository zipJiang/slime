"""Prefix-only 27B summaries with successful 9B action replay; explicit off-policy SFT."""
import argparse,asyncio,gzip,hashlib,json,pickle,re,time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from runtime_v2 import EXPERIMENT as E,MODEL,digest,verify_harness
from synthetic_runtime import make_world,environment
from collect_warmup import write
from examples.locbench.dataset import load_cases
from examples.locbench.env import build_world,RunConfig
from examples.locbench.metrics import score
from examples.locbench.repository import prepare
from step_controller.generation import Policy,VLLMPolicy,PolicyFormat,SamplingParams,GenerateResult
from step_controller.export import to_samples
from step_controller.preparation.records import ActorSample,PreparedBatch,spans_of
from profile_pilot import metrics,QUESTIONS
PROFILE='compact24-reply8-read60'
CACHE=Path('/weka/scratch/jhu/bvandur1/zjiang31/rollout-controller/data/locbench/repositories')
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_bytes(gzip.compress(pickle.dumps(v),compresslevel=1,mtime=0));tmp.replace(p)
def unpack(p):return pickle.loads(gzip.decompress(Path(p).read_bytes()))
class LoggedPolicy(VLLMPolicy):
 def __init__(self,*,output,key,teacher=False,**kw):
  super().__init__(**kw);self.output=output;self.key=key;self.teacher=teacher;self.index=0;self.records=[]
 def _completion_kwargs(self,model,prefix_tokens,params):
  kw=super()._completion_kwargs(model,prefix_tokens,params);kw['seed']=int(hashlib.sha256(f'{self.key}/{self.index}'.encode()).hexdigest()[:8],16)%2**31;return kw
 async def agenerate_tokens(self,tokens,sampling_params=None):
  if self.teacher:sampling_params=SamplingParams(max_tokens=16384,temperature=.2,top_p=.95,top_k=20,logprobs=1)
  allowance=16384 if self.teacher else 8192
  if len(tokens)+allowance>98304:raise ValueError('Context reserve exceeded; no truncation')
  start=time.monotonic();result=await super().agenerate_tokens(tokens,sampling_params)
  item=dict(input_tokens=len(tokens),output_tokens=len(result.tokens),stop_reason=result.stop_reason,seconds=time.monotonic()-start,text=self.decode(result.tokens,skip_special_tokens=True))
  write(self.output/'calls'/f'{self.index:04d}.json',item);self.records.append(item);self.index+=1
  return result
class Replay(Policy):
 def __init__(self,live):super().__init__(format=live.format,exact=False);self.live=live;self.next_turn=None;self.allow_live=False
 async def agenerate_tokens(self,tokens,sampling_params=None):
  if self.next_turn is not None:
   if self.allow_live:raise ValueError('Replay target entered compactor')
   t,self.next_turn=self.next_turn,None
   return GenerateResult(tokens=t.tokens,prefix_tokens=tuple(tokens),exact_generation=False)
  if not self.allow_live:raise ValueError('Unplanned live execution')
  return await self.live.agenerate_tokens(tokens,sampling_params)
def export(turns,q,source,first_fold):
 selected=tuple(replace(t,logprobs={},exact=False) for t in turns[first_fold:])
 if not selected or selected[0].tag!='fold':raise ValueError('Missing first fold')
 record=ActorSample(node_id=1,group_id=0,spans=spans_of(selected),reward_config_id='synthetic_sft',estimator='synthetic_sft',weight=1.,importance=1.)
 rows=to_samples(PreparedBatch(actor=(record,),behavior_version='none'),group_index=0,apply_importance=False)
 for r in rows:
  r['group_index']=q;r['metadata'].update(case_id=q,synthetic=True,on_policy=False,objective='supervised_imitation',source=source,teacher_model='Qwen/Qwen3.5-27B',execution_model='Qwen/Qwen3.5-9B')
  if r['logprobs'] or len(r['tokens'])>98304:raise ValueError('Invalid SFT export')
 if [t for r in rows for t,m in zip(r['tokens'],r['loss_mask'],strict=True) if m]!=[t for turn in selected for t in turn.tokens]:raise ValueError('Export changed targets')
 if {r['metadata']['tag'] for r in rows}!={'fold','task'}:raise ValueError('Need both target kinds')
 return rows
class Workflow:
 def __init__(self,run,fmt,infra,cases,split):
  self.run=run;self.fmt=fmt;self.infra=infra;self.cases=cases;self.split=split;self.targets={'train':48,'development':8};self.selected={l:{} for l in self.targets};self.done=asyncio.Event();self.attempts=0
 def policy(self,key,out,index=0,teacher=False):
  urls=self.infra['teacher_urls' if teacher else 'actor_urls']
  return LoggedPolicy(output=out,key=key,teacher=teacher,model=self.infra['teacher_model'] if teacher else MODEL,served_model='locbench-synthetic-27b' if teacher else 'locbench-synthetic-9b',format=self.fmt,base_url=urls[index%len(urls)],api_key='EMPTY',timeout=1800,version='locbench-synthetic-source-v1',default_params=SamplingParams(max_tokens=16384 if teacher else 8192,temperature=.2 if teacher else .6,top_p=.95,top_k=20,logprobs=1))
 def qualify(self,state,case,policy):
  m=metrics(state,policy);ceiling=min(5,len(case.gold.files))/len(case.gold.files);reward=score(state.state.locations,case.gold)['reward']
  # Selection only: preserve the benchmark reward unchanged, including its top-5 ceiling.
  useful=any(len(t.prefix)>=24576 for t in state.turns) or m['tool_calls']>20
  oversized_reads=0
  for event in m['events']:
   for call in event['calls']:
    if call['name']=='read' and isinstance(call['arguments'],dict):
     try:oversized_reads+=int(call['arguments'].get('lines',60))>60
     except (TypeError,ValueError):oversized_reads+=1
  good=(not oversized_reads and state.done and state.state.submitted and not state.folds and reward+1e-9>=ceiling and useful and m['max_task_reply']<=8192 and m['task_turns']<=60 and m['repeated_calls']<=max(1,int(.10*m['tool_calls'])))
  return dict(qualified=good,oversized_read_requests=oversized_reads,reward=reward,reward_ceiling=ceiling,useful=useful,**{k:v for k,v in m.items() if k!='events'})
 def select(self,lane,q,path,summary):
  if not summary['qualified'] or q in self.selected[lane] or len(self.selected[lane])>=self.targets[lane]:return
  if q not in self.split[lane] or q in QUESTIONS:raise ValueError('Split or held-out behavioral cohort violation')
  item=dict(lane=lane,case_id=q,path=str(path),source_sha256=digest(path),summary=summary)
  self.selected[lane][q]=item;write(self.run/'selected'/lane/(q+'.json'),item)
 def progress(self):write(self.run/'collection-progress.json',dict(targets=self.targets,selected={l:len(v) for l,v in self.selected.items()},attempts=self.attempts,time=time.time()))
 async def adopt(self):
  for lane in self.targets:
   for p in (self.run/'selected'/lane).glob('*.json'):
    d=read(p)
    if digest(d['path'])!=d['source_sha256']:raise ValueError('Selected source changed')
    if d['case_id'] not in self.split[lane] or d['case_id'] in QUESTIONS:raise ValueError('Split changed')
    self.selected[lane][d['case_id']]=d
  # Only fresh traces using the exact 60-line tool contract can be adopted.
  pilot=E/'runs/read60-uncompacted-pilot-v1'
  approved=read(pilot/'review.json')
  if approved.get('proceed_synthetic') is not True:raise ValueError('Read-limit baseline must pass review first')
  for complete in sorted((pilot/'60').glob('*/complete.json')):
   original=read(complete);q=original['query_id'];lane=original['lane']
   if q in QUESTIONS or q in self.selected[lane]:continue
   source=complete.parent/'trace.pkl.gz';state=unpack(source);policy=self.policy(q,self.run/'adoption')
   try:summary=self.qualify(state,self.cases[q],policy)
   finally:await policy.aclose()
   if not summary['qualified']:continue
   target=self.run/'collection'/lane/q/'adopted.pkl.gz';save(target,state)
   summary.update(adopted_from=str(source),adopted_sha256=digest(source));write(target.with_suffix('.json'),summary);self.select(lane,q,target,summary)
  self.progress()
 async def collect_one(self,lane,q,attempt,worker):
  out=self.run/'collection'/lane/q/f'sample-{attempt}';complete=out/'complete.json'
  if complete.exists():self.select(lane,q,out/'trace.pkl.gz',read(complete));return
  policy=self.policy(f'{q}/{attempt}',out,worker);state=None;start=time.monotonic()
  try:
   case=self.cases[q];repo=await asyncio.to_thread(prepare,CACHE,case.repo,case.base_commit)
   runner,ws,prompt,_=make_world(case,repo,policy,compact=False)
   runner._sampling_params=SamplingParams(max_tokens=8192,temperature=.6,top_p=.95,top_k=20)
   state=await runner.start(prompt,workspace=ws)
   while not state.done and not state.truncated:
    # Reserve space for finalization without truncating any conditioning.
    if len(policy.prepare(state.messages,tools=runner._tools).tokens)>88000:state=await runner.finish(state);break
    state=await runner.advance(state);write(out/'progress.json',dict(turns=state.turns_taken,time=time.time()));save(out/'partial.pkl.gz',state.snapshot())
   if not state.done:state=await runner.finish(state)
   summary=dict(**self.qualify(state,case,policy),seconds=time.monotonic()-start)
   save(out/'trace.pkl.gz',state.snapshot());write(complete,summary);self.select(lane,q,out/'trace.pkl.gz',summary)
  except Exception as e:write(out/'failure.json',dict(error=repr(e),time=time.time()))
  finally:
   if state is not None:save(out/'partial.pkl.gz',state.snapshot())
   await policy.aclose();self.attempts+=1;self.progress()
 async def collect(self):
  jobs=[(lane,q,a) for a in range(3) for lane in self.targets for q in self.split[lane] if q not in QUESTIONS]
  jobs.sort(key=lambda x:(x[2],hashlib.sha256('/'.join(map(str,x)).encode()).hexdigest()));queue=iter(jobs)
  async def worker(i):
   for lane,q,a in queue:
    if q in self.selected[lane] or len(self.selected[lane])>=self.targets[lane]:continue
    await self.collect_one(lane,q,a,i)
  try:await asyncio.gather(*(worker(i) for i in range(3*len(self.infra['actor_urls']))))
  finally:self.done.set()
  if any(len(self.selected[l])!=n for l,n in self.targets.items()):raise ValueError('Source attempts exhausted')
  write(self.run/'collection-complete.json',dict(targets=self.targets,time=time.time()))
 async def prepare_one(self,item,index):
  lane,q=item['lane'],item['case_id'];out=self.run/'synthetic'/lane/q
  if (out/'complete.json').exists():return
  if digest(item['path'])!=item['source_sha256']:raise ValueError('Source changed')
  source=unpack(item['path']);teacher=self.policy('fold/'+q,out/'teacher',index,True);replay=Replay(teacher);state=None
  try:
   case=self.cases[q];repo=await asyncio.to_thread(prepare,CACHE,case.repo,case.base_commit)
   runner,ws,prompt,tools=make_world(case,repo,replay);inner=runner._compactor
   state=await runner.start(prompt,workspace=ws);first=None
   for i,target in enumerate(source.turns):
    if not target.tokens:raise ValueError('Source has non-generated marker')
    if inner.trigger.fires(replay.prepare(state.messages,tools=tools).tokens,state.state):
     before=state.snapshot();replay.allow_live=True
     try:state=await runner.compact(state)
     finally:replay.allow_live=False
     if state.done:raise ValueError('Teacher compaction rejected')
     if first is None:first=len(before.turns)
     record=teacher.records[-1];visible=record['text'].split('</think>')[-1]
     headings=['Candidate locations','Evidence and completed exploration','Remaining uncertainty','Next steps']
     if record['stop_reason']!='stop' or not all('## '+h in visible for h in headings) or not re.search(r'## Next steps\s+None\.?\s*$',visible):raise ValueError('Invalid completed no-verification note')
     if inner.trigger.fires(replay.prepare(state.messages,tools=tools).tokens,state.state):raise ValueError('Fold failed to relieve trigger')
     save(out/'folds'/f'{state.folds:03d}.pkl.gz',dict(before=before,after=state.snapshot()))
    if state.turns_taken+state.folds>=80:raise ValueError('Replay exceeds target horizon')
    replay.next_turn=target
    if source.truncated and i==len(source.turns)-1:state=await runner.finish(state)
    else:state=await runner.advance(state)
    observed=state.turns[-1]
    if replay.next_turn is not None or observed.tokens!=target.tokens or observed.transition.messages!=target.transition.messages or observed.transition.done!=target.transition.done or observed.transition.reward_outcome!=target.transition.reward_outcome:raise ValueError('Execution replay changed source action/observation')
    write(out/'progress.json',dict(source_turn=i+1,source_turns=len(source.turns),folds=state.folds,time=time.time()))
   if not state.done or state.state.locations!=source.state.locations or first is None:raise ValueError('Incomplete synthetic trajectory')
   rows=export(state.turns,q,item['path'],first);out.mkdir(parents=True,exist_ok=True)
   blob=gzip.compress((''.join(json.dumps(r)+'\n' for r in rows)).encode(),mtime=0);(out/'samples.jsonl.gz').write_bytes(blob)
   if [json.loads(l) for l in gzip.decompress(blob).splitlines()]!=rows:raise ValueError('Serialization mismatch')
   save(out/'trace.pkl.gz',state.snapshot());counts=Counter();tokens=Counter()
   for r in rows:counts[r['metadata']['tag']]+=1;tokens[r['metadata']['tag']]+=sum(r['loss_mask'])
   write(out/'complete.json',dict(source=item,rows=dict(counts),trainable_tokens=dict(tokens),folds=state.folds,first_fold_turn=first,source_turns=len(source.turns),samples_sha256=digest(out/'samples.jsonl.gz'),trace_sha256=digest(out/'trace.pkl.gz'),exact_execution_target_copy=True))
  except Exception as e:
   write(out/'failure.json',dict(error=repr(e),source=item,time=time.time()))
   if state is not None:save(out/'partial.pkl.gz',state.snapshot())
  finally:await teacher.aclose()
 async def prepare(self):
  attempted=set();pending=set();sem=asyncio.Semaphore(8)
  async def one(item,index):
   async with sem:await self.prepare_one(item,index)
  while True:
   for lane,items in self.selected.items():
    for q,item in list(items.items()):
     if (lane,q) not in attempted:attempted.add((lane,q));pending.add(asyncio.create_task(one(item,len(attempted))))
   completed={t for t in pending if t.done()}
   for t in completed:t.result()
   pending-=completed;index={l:[] for l in self.targets}
   for p in sorted((self.run/'synthetic').glob('*/*/complete.json')):
    d=read(p);index[d['source']['lane']].append(dict(case_id=d['source']['case_id'],samples=str(p.parent/'samples.jsonl.gz'),sha256=d['samples_sha256'],rows=d['rows'],trainable_tokens=d['trainable_tokens']))
   write(self.run/'training-index.json',dict(protocol='locbench-iterative-27b-9b-sft-v1',on_policy=False,environment=environment(),lanes=index,targets=self.targets))
   write(self.run/'preparation-progress.json',dict(ready={l:len(v) for l,v in index.items()},pending=len(pending),time=time.time()))
   if self.done.is_set() and not pending:break
   await asyncio.sleep(3)
  if {l:len(v) for l,v in index.items()}!=self.targets:raise ValueError('Synthetic preparation incomplete; inspect quarantined traces')
  write(self.run/'dataset-complete.json',dict(targets=self.targets,index_sha256=digest(self.run/'training-index.json'),time=time.time()))
async def main(a):
 from transformers import AutoTokenizer
 verify_harness();infra=read(a.run/'ready.json');split=read(E/'data/split.json')
 tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);teacher=AutoTokenizer.from_pretrained(infra['teacher_model'],local_files_only=True)
 if tokenizer.get_vocab()!=teacher.get_vocab() or tokenizer.chat_template!=teacher.chat_template:raise ValueError('Teacher/student token formats differ')
 fmt=PolicyFormat.resolve(MODEL,tokenizer=tokenizer,profile='qwen_xml')
 cases={c.id:c for lane in ('train','development') for c in load_cases(E/'data'/f'{lane}.jsonl')}
 manifest=dict(protocol='locbench-iterative-27b-9b-sft-v1',environment=environment(),split_sha256=digest(E/'data/split.json'),source_sha256=digest(__file__),base_actor=MODEL,teacher=infra['teacher_model'],target_train=48,target_development=8,selection='Observed recall reaches top-5 attainable ceiling; No oversized read requests; <=10% duplicate calls, <=60 task turns, useful compaction; standard reward unchanged',on_policy=False,behavior_evaluation='Fresh post-SFT continuations; offline action replay is not evidence of fewer research turns')
 if (a.run/'manifest.json').exists() and read(a.run/'manifest.json')!=manifest:raise ValueError('Dataset recipe changed')
 write(a.run/'manifest.json',manifest);flow=Workflow(a.run,fmt,infra,cases,split);await flow.adopt();await asyncio.gather(flow.collect(),flow.prepare())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);asyncio.run(main(p.parse_args()))
