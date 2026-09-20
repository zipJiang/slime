"""Persist token-level evidence and stop before gradients or optimizer updates."""
import json
from pathlib import Path
import torch
import torch.distributed as dist
from audit_on_policy import check as original_check

def check(args,rollout_id,data):
    directory=Path(args.save).parent/'diagnostic-logprobs'
    directory.mkdir(exist_ok=True)
    current=data['log_probs']; behavior=data['rollout_log_probs']
    references=data.get('ref_log_probs',[None]*len(current))
    rows=[]; tensors=[]
    for i,(new,old,ref,mask,tokens) in enumerate(zip(current,behavior,references,data['loss_masks'],data['tokens'],strict=True)):
        mask=mask.to(new.device,dtype=torch.bool)
        delta=(new-old.to(new.device))[mask].detach().float()
        row=dict(index=i,context_tokens=len(tokens),response_tokens=len(mask),trainable_tokens=int(mask.sum()),
            mean_abs=float(delta.abs().mean()),p99_abs=float(delta.abs().quantile(.99)),
            max_abs=float(delta.abs().max()),mean_ratio=float(delta.exp().mean()))
        if ref is not None:
            row['actor_ref_mean_abs']=float((new-ref.to(new.device))[mask].abs().mean())
            row['behavior_ref_mean_abs']=float((old.to(new.device)-ref.to(new.device))[mask].abs().mean())
        rows.append(row)
        tensors.append(dict(tokens=tokens.cpu(),mask=mask.cpu(),current=new.detach().cpu(),
            behavior=old.detach().cpu(),reference=ref.detach().cpu() if ref is not None else None))
    rank=dist.get_rank()
    torch.save(tensors,directory/f'rank-{rank}.pt')
    (directory/f'rank-{rank}.json').write_text(json.dumps(rows,indent=2))
    try:
        original_check(args,rollout_id,data)
    finally:
        raise RuntimeError('DIAGNOSTIC COMPLETE: deliberately stopped before optimizer update')
