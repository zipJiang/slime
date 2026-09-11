"""Observe current/behavior agreement before Slime applies its native PPO loss."""
import json
from pathlib import Path
import torch
import torch.distributed as dist


def metrics(args, *, pg_loss, train_log_probs, rollout_log_probs, loss_masks, **kwargs):
    """Slime's mismatch-metric hook: no TIS weighting or rejection sampling."""
    ratio = (torch.cat(train_log_probs)-torch.cat(rollout_log_probs)).detach().exp()
    return pg_loss, loss_masks, {'tis':ratio, 'tis_abs':(ratio-1).abs(),
                                'tis_clipfrac':torch.zeros_like(ratio)}


def check(args, rollout_id, data):
    current, behavior = data.get('log_probs'), data.get('rollout_log_probs')
    if current is None:  # Non-last pipeline stage (PP=1 in this experiment).
        return
    differences = []
    for new, old, mask in zip(current, behavior, data['loss_masks'], strict=True):
        mask = mask.to(device=new.device,dtype=torch.bool)
        differences.append((new-old.to(new.device))[mask].detach().float())
    diff = torch.cat(differences)
    if not diff.numel() or not torch.isfinite(diff).all():
        raise ValueError('Missing or nonfinite current/behavior token log probabilities')
    report = dict(rollout_id=rollout_id, rank=dist.get_rank(), tokens=diff.numel(),
                  mean_abs_difference=diff.abs().mean().item(),
                  p99_abs_difference=diff.abs().quantile(.99).item(),
                  mean_ratio=diff.exp().mean().item())
    directory = Path(args.save).parent/'on-policy-audit'
    directory.mkdir(parents=True,exist_ok=True)
    (directory/f'round-{rollout_id:04d}-rank-{dist.get_rank()}.json').write_text(json.dumps(report,indent=2)+'\n')
    # Small BF16/kernel differences are expected across inference and training.
    # Large disagreement needs diagnosis before any optimizer update.
    bad = torch.tensor(int(report['mean_abs_difference'] > .05 or
                           report['p99_abs_difference'] > .5), device=diff.device)
    dist.all_reduce(bad,op=dist.ReduceOp.MAX)
    if bad.item():
        raise ValueError('Current/behavior logprob mismatch exceeds the initial audit tolerance')
