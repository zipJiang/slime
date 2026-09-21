"""Independently validate source identity, split membership, replay targets, and masks."""
import gzip,hashlib,json,pickle,sys
from collections import Counter
from pathlib import Path
from runtime_v2 import EXPERIMENT as E,digest
sys.path.append(str(E/'snapshots/native-support-v1'))
def read(p):return json.loads(Path(p).read_text())
def unpack(p):return pickle.loads(gzip.decompress(Path(p).read_bytes()))
def training_order(groups,batch_size,*,smoke=False):
 order=sorted(groups,key=lambda q:hashlib.sha256(('locbench-sft-v1/'+q).encode()).hexdigest())
 if smoke:order=sorted(order,key=lambda q:max(len(r['tokens']) for r in groups[q]),reverse=True)[:batch_size]*2
 if batch_size<=0 or not order or len(order)%batch_size:raise ValueError('Need complete question batches')
 return order
def validate_rows(rows,question,*,max_length=98304):
 counts=Counter();tokens=Counter()
 for r in rows:
  m=r['metadata'];ids=r['tokens'];mask=r['loss_mask']
  if r['group_index']!=question or m.get('synthetic') is not True or m.get('on_policy') is not False or m.get('objective')!='supervised_imitation' or m.get('tag') not in ('task','fold') or r.get('logprobs')!=[]:raise ValueError('Invalid SFT provenance')
  if not 1<len(ids)<=max_length or len(ids)!=len(mask) or any(type(t) is not int or t<0 for t in ids) or any(m not in (0,1) for m in mask) or not any(mask) or mask[0]:raise ValueError('Invalid tokens/masks')
  counts[m['tag']]+=1;tokens[m['tag']]+=sum(mask)
 if set(counts)!={'task','fold'} or any(r['metadata']['edge_tokens']!=sum(tokens.values()) for r in rows):raise ValueError('Missing target type or wrong question denominator')
 return dict(rows=dict(counts),trainable_tokens=dict(tokens))
def load_dataset(run,split,*,require_complete=True):
 from examples.locbench.dataset import load_cases
 from examples.locbench.metrics import score
 from synthetic_runtime import environment
 run=Path(run);manifest=read(run/'manifest.json')
 if manifest['environment']!=environment():raise ValueError('Imitation collection environment changed')
 if manifest['split_sha256']!=digest(E/'data/split.json'):raise ValueError('Split changed')
 cases={c.id:c for lane in ('train','development') for c in load_cases(E/'data'/f'{lane}.jsonl')}
 targets=dict(train=manifest['target_train'],development=manifest['target_development'])
 if require_complete:
  complete=read(run/'dataset-complete.json')
  if complete['targets']!=targets or complete['index_sha256']!=digest(run/'training-index.json'):raise ValueError('Completion/index mismatch')
 data={l:{} for l in targets};seen=set();inventory=[]
 for p in sorted((run/'synthetic').glob('*/*/complete.json')):
  d=read(p);src=d['source'];lane=src['lane'];q=src['case_id']
  if lane not in data or q not in split[lane] or q in seen or p.parent.name!=q or p.parent.parent.name!=lane:raise ValueError('Split overlap/identity mismatch')
  seen.add(q)
  if digest(src['path'])!=src['source_sha256']:raise ValueError('Source changed')
  source=unpack(src['path']);state=unpack(p.parent/'trace.pkl.gz');case=cases[q]
  if source.folds or not source.done or state.state.locations!=source.state.locations or not state.done:raise ValueError('Wrong source or replay terminal')
  reward=score(source.state.locations,case.gold)['reward'];ceiling=min(5,len(case.gold.files))/len(case.gold.files)
  if reward+1e-9<ceiling:raise ValueError('Source below attainable recall ceiling')
  if any(len(t.tokens)>=8192 for t in source.turns):raise ValueError('Source reached generation cap; review truncated reasoning before SFT')
  original=[t.tokens for t in source.turns];replayed=[t.tokens for t in state.turns if t.tag=='task']
  if replayed!=original:raise ValueError('Synthetic execution targets changed')
  rows=[json.loads(l) for l in gzip.decompress((p.parent/'samples.jsonl.gz').read_bytes()).splitlines()]
  if digest(p.parent/'samples.jsonl.gz')!=d['samples_sha256'] or digest(p.parent/'trace.pkl.gz')!=d['trace_sha256']:raise ValueError('Synthetic artifact changed')
  stats=validate_rows(rows,q)
  if any(stats[k]!=d[k] for k in stats):raise ValueError('Counts differ')
  selected=state.turns[d['first_fold_turn']:]
  if manifest['protocol']=='locbench-iterative-27b-9b-masked-read-sft-v1':
   from synthetic_targets import rejected_read
   if digest(E/'scripts/synthetic_targets.py')!=manifest['target_mask_sha256']:raise ValueError('Target mask implementation changed')
   ignored=[i for i,t in enumerate(state.turns) if i>=d['first_fold_turn'] and rejected_read(t)]
   if any(r['metadata'].get('ignored_rejected_read_turns')!=ignored for r in rows):raise ValueError('Rejected-read mask provenance differs')
   selected=tuple(t for t in selected if not rejected_read(t))
  if selected[0].tag!='fold' or [t for turn in selected for t in turn.tokens]!=[t for r in rows for t,m in zip(r['tokens'],r['loss_mask'],strict=True) if m]:raise ValueError('Target mask duplicates/drops tokens')
  data[lane][q]=rows;inventory.append(dict(lane=lane,case_id=q,sha256=d['samples_sha256'],max_length=max(len(r['tokens']) for r in rows),**stats))
 if require_complete:
  if {l:len(v) for l,v in data.items()}!=targets:raise ValueError('Incomplete question counts')
  index=read(run/'training-index.json');expected={(l,r['case_id']):r['sha256'] for l,rows in index['lanes'].items() for r in rows}
  if {(r['lane'],r['case_id']):r['sha256'] for r in inventory}!=expected:raise ValueError('Index differs')
 return data,inventory
def question_packet(groups):
 from batches import training_data
 records=[]
 for group,rows in enumerate(groups):
  total=sum(sum(r['loss_mask']) for r in rows)
  for r in rows:
   start=next(i for i,m in enumerate(r['loss_mask']) if m)
   records.append(dict(tokens=r['tokens'],response_length=len(r['tokens'])-start,loss_mask=r['loss_mask'][start:],reward=0.,group_index=group,rollout_log_probs=[],metadata=dict(lane='actor',node_id=0,edge_tokens=total)))
 packet=training_data(records,lane='actor',expected_groups=range(len(groups)))
 del packet['rollout_log_probs']
 # Temperature-one SFT; explicit values also select the qualified BF16 streamed-logit path.
 packet['sampling_temperatures']=[1.]*len(records)
 return packet
