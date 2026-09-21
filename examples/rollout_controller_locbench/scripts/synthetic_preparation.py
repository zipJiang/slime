"""Prepare successful low-repeat sources, masking rejected read requests, while collection continues."""
import argparse,asyncio,json,time,subprocess
from pathlib import Path
from synthetic_data import *
from synthetic_targets import export
class Preparation(Workflow):
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

async def main(a):
 from transformers import AutoTokenizer
 infra=read(a.sources/'ready.json');split=read(E/'data/split.json');a.run.mkdir(exist_ok=False,parents=True)
 tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);teacher=AutoTokenizer.from_pretrained(infra['teacher_model'],local_files_only=True)
 if tokenizer.get_vocab()!=teacher.get_vocab() or tokenizer.chat_template!=teacher.chat_template:raise ValueError('Token format mismatch')
 fmt=PolicyFormat.resolve(MODEL,tokenizer=tokenizer,profile='qwen_xml');cases={c.id:c for l in ('train','development') for c in load_cases(E/'data'/f'{l}.jsonl')}
 flow=Preparation(a.run,fmt,infra,cases,split)
 manifest=read(a.sources/'manifest.json');manifest.update(protocol='locbench-iterative-27b-9b-masked-read-sft-v1',source_collection=str(a.sources),source_collection_manifest_sha256=digest(a.sources/'manifest.json'),source_sha256=digest(__file__),target_mask_sha256=digest(E/'scripts/synthetic_targets.py'),selection='Observed recall reaches attainable top-5 ceiling; useful compaction; <=60 task turns; <=10% exact repeated calls; no execution reply reaches 8192 tokens. Rejected read requests retain history but have zero SFT target loss.')
 if a.reuse:
  import shutil
  previous=read(a.reuse/'manifest.json')
  for key in ('source_collection_manifest_sha256','target_mask_sha256'):
   if previous[key]!=manifest[key]:raise ValueError('Reusable dataset contract changed')
  for p in sorted(a.reuse.glob('synthetic/*/*/complete.json')):
   d=read(p);item=d['source'];lane=item['lane'];q=item['case_id']
   if digest(item['path'])!=item['source_sha256']:raise ValueError('Reusable source changed')
   if any(len(t.tokens)>=8192 for t in unpack(item['path']).turns):continue
   if q not in split[lane] or q in QUESTIONS or q in flow.selected[lane]:raise ValueError('Reusable split mismatch')
   if digest(p.parent/'trace.pkl.gz')!=d['trace_sha256'] or digest(p.parent/'samples.jsonl.gz')!=d['samples_sha256']:raise ValueError('Reusable targets changed')
   shutil.copytree(p.parent,a.run/'synthetic'/lane/q)
   flow.selected[lane][q]=item;write(a.run/'selected'/lane/(q+'.json'),item)
  manifest['reused_from']=str(a.reuse);manifest['reused_questions']={l:sorted(v) for l,v in flow.selected.items()}
 write(a.run/'manifest.json',manifest);write(a.run/'ready.json',infra)
 sem=asyncio.Semaphore(8);pending=set();attempted=set();seen=set()
 async def one(item,i):
  async with sem:await flow.prepare_one(item,i)
 while True:
  for p in sorted(a.sources.glob('collection/*/*/*/complete.json')):
   if str(p) in seen:continue
   seen.add(str(p));d=read(p);lane=p.parents[2].name;q=p.parents[1].name
   if q not in split[lane] or q in QUESTIONS:continue
   if len(flow.selected[lane])>=flow.targets[lane] or q in flow.selected[lane]:continue
   if (d['reward']+1e-9<d['reward_ceiling'] or not d['useful'] or d['task_turns']>60 or d['repeated_calls']>max(1,int(.1*d['tool_calls']))):continue
   native=p.parent/'trace.pkl.gz';state=unpack(native)
   if not state.done or not state.state.submitted or state.folds:raise ValueError('Invalid source terminal')
   if any(len(t.tokens)>=8192 for t in state.turns):continue
   item=dict(lane=lane,case_id=q,path=str(native),source_sha256=digest(native),source_summary_sha256=digest(p),summary=d)
   flow.selected[lane][q]=item;write(a.run/'selected'/lane/(q+'.json'),item)
   pending.add(asyncio.create_task(one(item,len(seen))))
  finished={t for t in pending if t.done()}
  for t in finished:t.result()
  pending-=finished
  index={l:[] for l in flow.targets}
  for p in sorted((a.run/'synthetic').glob('*/*/complete.json')):
   d=read(p);index[d['source']['lane']].append(dict(case_id=d['source']['case_id'],samples=str(p.parent/'samples.jsonl.gz'),sha256=d['samples_sha256'],rows=d['rows'],trainable_tokens=d['trainable_tokens']))
  write(a.run/'training-index.json',dict(protocol=manifest['protocol'],on_policy=False,environment=environment(),lanes=index,targets=flow.targets))
  write(a.run/'preparation-progress.json',dict(ready={l:len(v) for l,v in index.items()},selected={l:len(v) for l,v in flow.selected.items()},pending=len(pending),source_traces_examined=len(seen),time=time.time()))
  if {l:len(v) for l,v in index.items()}==flow.targets:break
  failures=list((a.run/'synthetic').glob('*/*/failure.json'))
  if failures and not pending:raise ValueError('Teacher preparation failed; inspect quarantined examples')
  # A successful scheduler query is authoritative; a transient query failure is not.
  check=subprocess.run(['squeue','--steps','-j','413877','-h','-o','%i'],text=True,capture_output=True)
  if check.returncode==0 and a.collector_step not in check.stdout.split() and not a.sources.joinpath('collection-complete.json').exists():raise RuntimeError('Source collector step is no longer live')
  if a.sources.joinpath('collection-complete.json').exists() and not pending:raise ValueError('Source collection ended before masked dataset filled')
  await asyncio.sleep(5)
 write(a.run/'dataset-complete.json',dict(targets=flow.targets,index_sha256=digest(a.run/'training-index.json'),time=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--collector-step',required=True);p.add_argument('--reuse',type=Path);asyncio.run(main(p.parse_args()))
