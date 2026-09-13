"""Audit behavior/current logprob agreement before each pilot actor update."""
import json
from pathlib import Path


def metrics(args,*,pg_loss,train_log_probs,rollout_log_probs,loss_masks,**kwargs):
    import torch
    ratio=(torch.cat(train_log_probs)-torch.cat(rollout_log_probs)).detach().exp()
    return pg_loss,loss_masks,dict(tis=ratio,tis_abs=(ratio-1).abs(),
        tis_clipfrac=torch.zeros_like(ratio))


def check(args,rollout_id,data):
    import torch
    import torch.distributed as dist
    current,behavior=data.get('log_probs'),data.get('rollout_log_probs')
    if current is None: return
    run=Path(args.save).parent
    lineage=json.loads((run/'pilot-rollouts'/f'train-{rollout_id:04d}'/'training-lineage.json').read_text())
    expected=dict(collection_round=rollout_id,behavior_round=rollout_id,
        learner_round=rollout_id,actor_lag=0,critic_lag=0,
        denominator='stored_behavior_logprobs',execution='sequential')
    if lineage!=expected: raise ValueError('Pilot training lineage changed before native PPO')
    differences=[]
    for new,old,mask in zip(current,behavior,data['loss_masks'],strict=True):
        mask=mask.to(device=new.device,dtype=torch.bool)
        differences.append((new-old.to(new.device))[mask].detach().float())
    diff=torch.cat(differences)
    if not diff.numel() or not torch.isfinite(diff).all():
        raise ValueError('Missing or nonfinite current/behavior token log probabilities')
    report=dict(rollout_id=rollout_id,rank=dist.get_rank(),tokens=diff.numel(),lineage=lineage,
        mean_abs_difference=diff.abs().mean().item(),
        p99_abs_difference=diff.abs().quantile(.99).item(),mean_ratio=diff.exp().mean().item())
    output=run/'on-policy-audit';output.mkdir(parents=True,exist_ok=True)
    (output/f'round-{rollout_id:04d}-rank-{dist.get_rank()}.json').write_text(
        json.dumps(report,indent=2)+'\n')
    bad=torch.tensor(int(report['mean_abs_difference']>.05 or
        report['p99_abs_difference']>.5),device=diff.device)
    dist.all_reduce(bad,op=dist.ReduceOp.MAX)
    if bad.item(): raise ValueError('Current/behavior logprob mismatch exceeds pilot tolerance')
