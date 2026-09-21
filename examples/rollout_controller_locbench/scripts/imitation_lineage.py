"""Require the audited SFT model identity before critic collection or fresh PPO."""
import hashlib,json,math
from pathlib import Path
from runtime_v2 import EXPERIMENT as E

def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require_model(training):
 training=Path(training).resolve();complete=read(training/'complete.json');recipe=read(training/'recipe.json')
 model=Path(complete['model']).resolve();source=Path(recipe['source']);source_manifest=read(source/'manifest.json')
 batch=recipe['arguments']['global_batch_size'];train=recipe['train_questions'];dev=recipe['development_questions']
 if (complete.get('protocol')!='locbench-iterative-imitation-v1' or complete.get('smoke') is not False
  or recipe.get('epochs')!=1 or len(train)!=len(set(train)) or len(train)!=source_manifest['target_train']
  or len(dev)!=source_manifest['target_development'] or set(train)&set(dev) or complete['updates']!=len(train)//batch):raise ValueError('Incomplete or mismatched imitation training')
 split=read(E/'data/split.json')
 if not set(train)<=set(split['train']) or not set(dev)<=set(split['development']):raise ValueError('Imitation split violation')
 if sha(source/'training-index.json')!=complete['source_index_sha256'] or complete['source_index_sha256']!=recipe['source_index_sha256']:raise ValueError('Imitation dataset identity changed')
 native=Path(complete['native']);audit=read(native.with_name(native.name+'-readback.json'))
 if (Path(audit['checkpoint']).resolve()!=native.resolve() or audit['role']!='actor' or not audit['full_storage_read']
  or not audit['finite_tensors'] or audit['optimizer_steps']!=[complete['updates']]):raise ValueError('Missing matching native SFT readback')
 export=read(model/'export-audit.json');index=read(model/'model.safetensors.index.json')
 if Path(export['target']).resolve()!=model or not export['changed_keys'] or export['expect_unchanged'] is not False:raise ValueError('Unverified trained HF export')
 if set(index['weight_map'].values())!=set(export['shard_sha256']):raise ValueError('HF shard coverage mismatch')
 for name,expected in export['shard_sha256'].items():
  path=(model/name).resolve()
  if not path.is_relative_to(model) or sha(path)!=expected:raise ValueError('HF model shard changed')
 for stage in ('initial','final'):
  for tag in ('task','fold'):
   entry=complete[stage][tag]
   if entry['tokens']<=0 or not math.isfinite(entry['mean_nll']):raise ValueError('Invalid held-out SFT likelihood')
 return dict(training=str(training),model=str(model),training_complete_sha256=sha(training/'complete.json'),recipe_sha256=sha(training/'recipe.json'),export_audit_sha256=sha(model/'export-audit.json'),model_index_sha256=sha(model/'model.safetensors.index.json'),shards=export['shard_sha256'],source_index_sha256=complete['source_index_sha256'],actor_updates=complete['updates'])
