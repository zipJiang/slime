"""Checkpoint success-probability regression, shared by inference and training."""
import torch


def regression_errors(values, targets, old_values, *, warmup, clip):
    if not torch.isfinite(targets).all() or ((targets < 0) | (targets > 1)).any():
        raise ValueError('Success-probability targets must be finite and in [0,1]')
    squared = (values - targets).square()
    if warmup:
        return squared, torch.zeros_like(values), squared
    old = old_values.detach()
    clipped = old + (values-old).clamp(-clip, clip)
    return torch.maximum(squared, (clipped-targets).square()), ((values-old).abs() > clip).float(), squared


def probability_values(logits, **kwargs):
    from slime.backends.megatron_utils.loss import get_values
    empty, result = get_values(logits, **kwargs)
    result['values'] = [v.sigmoid() for v in result['values']]
    return empty, result


def checkpoint_loss(args, batch, logits, reducer):
    if any(n != 1 for n in batch['response_lengths']):
        raise ValueError('Checkpoint loss cannot supervise actor response sequences')
    _, output = probability_values(logits, args=args,
                                  unconcat_tokens=batch['unconcat_tokens'],
                                  total_lengths=batch['total_lengths'],
                                  response_lengths=batch['response_lengths'])
    values = torch.cat(output['values']).flatten()
    targets = torch.cat(batch['returns']).flatten()
    errors, clipped_fraction, squared = regression_errors(
        values, targets, torch.cat(batch['values']).flatten(),
        warmup=getattr(args, 'ppo_critic_warmup', False), clip=args.value_clip)
    loss = reducer(errors)
    return loss, dict(value_loss=loss.detach(), value_clipfrac=reducer(clipped_fraction).detach(),
                      value_mse=reducer(squared).detach())
