"""Fresh scalar critic from the verified imitation actor and matching MC targets."""
import os,json,traceback,hashlib
from pathlib import Path
import train_critic as training
from synthetic_runtime import environment
from imitation_lineage import require_model

def custom_args(parser):
 training.custom_args(parser);parser.add_argument('--loc-imitation-training',type=Path,required=True);return parser

def main():
 import ray
 from slime.utils.arguments import parse_args
 from slime.observability.logging_utils import configure_logger
 args=parse_args(custom_args);configure_logger();identity=require_model(args.loc_imitation_training)
 if Path(args.hf_checkpoint).resolve()!=Path(identity['model']).resolve():raise ValueError('Critic structure/source must name the verified imitation checkpoint')
 if args.loc_preflight:
  from batches import partition_data
  from trace_warmup import packet
  from targets import checkpoint_fields
  data=training.load(args.loc_collection)
  world=args.actor_num_nodes*args.actor_num_gpus_per_node
  if world%2 or args.tensor_model_parallel_size!=2:raise ValueError('Critic preflight requires TP2')
  from megatron.core.num_microbatches_calculator import ConstantNumMicroBatchesCalculator
  calc=ConstantNumMicroBatchesCalculator(args.global_batch_size,args.micro_batch_size,
   world//2,args.decrease_batch_size_if_needed,0)
  if not args.use_critic or not args.offload_train:raise ValueError('Native critic role/offload missing')
  manifest=json.loads((args.loc_collection/'manifest.json').read_text())
  if manifest['environment']!=environment() or manifest['actor_identity']!=identity:raise ValueError('Critic collection identity mismatch')
  parallel=dict(dp_size=world//2,cp_size=1,vpp_size=1,microbatch_group_size_per_vp_stage=1)
  from transformers import AutoTokenizer
  tok=AutoTokenizer.from_pretrained(identity['model'],local_files_only=True)
  groups={q:[r for r in data['train'] if r['group_index']==q] for q in sorted({r['group_index'] for r in data['train']})}
  if not groups or len(groups)%args.global_batch_size:raise ValueError('Incomplete critic question batch')
  reports=[]
  for epoch in range(args.loc_epochs):
   order=sorted(groups,key=lambda q:hashlib.sha256(f'locbench-warmup/{epoch}/{q}'.encode()).hexdigest())
   for start in range(0,len(order),args.global_batch_size):
    rows=[]
    for group,q in enumerate(order[start:start+args.global_batch_size]):
     for row in groups[q]:
      prepared=checkpoint_fields(dict(row,group_index=group),tok,sentinel_token_id=tok.eos_token_id,
       max_sequence_length=training.CONTEXT_LIMIT,warmup=True)
      rows.append(dict(prepared,turn=row['turn']))
    pieces=partition_data(args,parallel,packet(rows,args.loc_root_mass))
    if any(sum(p['global_batch_sizes'])!=args.global_batch_size for p in pieces):raise ValueError('Critic question denominator changed')
    reports.append(dict(epoch=epoch,start=start,records=len(rows),packets=len(pieces)))
  print(json.dumps(dict(preflight_passed=True,batches=reports,initialization_batch=calc.current_running_global_batch_size,
   actor_identity=identity['training_complete_sha256'])))
  return
 ray.init(address=os.environ['RAY_ADDRESS'],runtime_env={'env_vars':{
  'GLOO_SOCKET_IFNAME':'ens0','NCCL_SOCKET_IFNAME':'ens0','NCCL_IB_DISABLE':'1','PYTHONPATH':os.environ['PYTHONPATH']}})
 training.run(args,model=identity['model'],environment=environment(),actor_identity=identity)
if __name__=='__main__':
 try:main()
 except BaseException:
  # Slurm driver log remains authoritative even before argument parsing completes.
  traceback.print_exc();raise
