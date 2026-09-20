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
    directory = Path(args.save).parent/'rollouts'/f'train-{rollout_id:04d}'
    lineage = json.loads((directory/'training-lineage.json').read_text())
    from pipeline import BatchStamp
    expected = BatchStamp(rollout_id, lineage['behavior_round'], args.num_critic_only_steps).lineage(
        rollout_id, overlap=args.ppo_execution == 'overlap')
    if lineage != expected:
        raise ValueError('Training lineage changed before native PPO')
    differences = []
    for new, old, mask in zip(current, behavior, data['loss_masks'], strict=True):
        mask = mask.to(device=new.device,dtype=torch.bool)
        differences.append((new-old.to(new.device))[mask].detach().float())
    diff = torch.cat(differences)
    if diff.numel() and not torch.isfinite(diff).all():
        raise ValueError('Nonfinite current/behavior token log probabilities')
    # A whole batch may legitimately contain only fully masked placeholders
    # after uninformative edges are filtered.  There is no drift to measure in
    # that case; report a neutral audit and preserve the optimizer/checkpoint
    # step contract instead of rejecting the batch before training.
    all_masked = not diff.numel()
    report = dict(
        rollout_id=rollout_id, rank=dist.get_rank(), tokens=diff.numel(),
        all_masked=all_masked, lineage=lineage,
        mean_abs_difference=0.0 if all_masked else diff.abs().mean().item(),
        p99_abs_difference=0.0 if all_masked else diff.abs().quantile(.99).item(),
        mean_ratio=1.0 if all_masked else diff.exp().mean().item(),
    )
    directory = Path(args.save).parent/'on-policy-audit'
    directory.mkdir(parents=True,exist_ok=True)
    (directory/f'round-{rollout_id:04d}-rank-{dist.get_rank()}.json').write_text(json.dumps(report,indent=2)+'\n')
    # Deterministic SGLang removed the batch-dependent Qwen3.5 drift that reached
    # mean .0285 / p99 .342 in the rejected round 27.  Normal native-versus-
    # serving BF16 drift is about mean .010 / p99 .13, so keep enough headroom for
    # kernels while rejecting a recurrence before the optimizer changes weights.
    bad = torch.tensor(int(report['mean_abs_difference'] > .02 or
                           report['p99_abs_difference'] > .25), device=diff.device)
    dist.all_reduce(bad,op=dist.ReduceOp.MAX)
    if bad.item():
        raise ValueError('Current/behavior logprob mismatch exceeds the initial audit tolerance')
