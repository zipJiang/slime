"""Validate controller exports without importing either framework runtime.

Actor rewards here are prepared advantages, never environment rewards. Critic
contexts remain separate: no response-token value targets are manufactured.
"""
import math
import hashlib
from collections import defaultdict


def split_targets(rows):
    actors, critics = [], []
    edges = defaultdict(list)
    checkpoints = set()
    value_versions = set()
    for row in rows:
        meta = dict(row['metadata'])
        lane = meta['lane']
        identity = (row['group_index'], meta['node_id'])
        if lane == 'critic':
            if identity in checkpoints:
                raise ValueError('Duplicate checkpoint target')
            checkpoints.add(identity)
            if not isinstance(row['context'], str) or not row['context']:
                raise ValueError('Critic context must be nonempty serialized text')
            target = float(row['target'])
            if not math.isfinite(target):
                raise ValueError('Nonfinite critic target')
            value_versions.add(meta['value_version'])
            critics.append(dict(context=row['context'], target=target,
                                group_index=row['group_index'], metadata=meta))
        elif lane == 'actor':
            tokens, mask, logps = row['tokens'], row['loss_mask'], row['logprobs']
            if not len(tokens) == len(mask) == len(logps):
                raise ValueError('Actor token/mask/logprob length mismatch')
            if any(m not in (0, 1) for m in mask) or not any(mask):
                raise ValueError('Invalid or empty actor mask')
            start = next(i for i, m in enumerate(mask) if m)
            if start == 0:
                raise ValueError('Actor action requires a conditioning prefix')
            if any(not math.isfinite(lp) or lp > 1e-5
                   for lp, m in zip(logps, mask) if m):
                raise ValueError('Invalid behavior logprob')
            advantage = float(row['reward'])
            if not math.isfinite(advantage):
                raise ValueError('Nonfinite actor advantage')
            if row.get('importance', 1.0) != 1.0:
                raise ValueError('Initial PPO recipe requires anchor-only collection')
            if meta['estimator'] not in ('refined_td', 'direct_branch_td'):
                raise ValueError('Expected a supported controller TD advantage')
            item = dict(tokens=list(tokens), response_length=len(tokens)-start,
                        loss_mask=list(mask[start:]),
                        rollout_log_probs=[lp if m else 0.0
                                           for lp, m in zip(logps[start:], mask[start:])],
                        reward=advantage, group_index=row['group_index'], metadata=meta)
            actors.append(item)
            edges[identity].append(item)
        else:
            raise ValueError(f'Unknown training lane: {lane}')
    if len(value_versions) > 1:
        raise ValueError('Mixed critic versions in settled batch')
    if len({r['metadata']['estimator'] for r in actors}) > 1:
        raise ValueError('Mixed preparation estimators in one training batch')
    for spans in edges.values():
        meta = spans[0]['metadata']
        n = meta['span_count']
        if len(spans) != n or sorted(s['metadata']['span_index'] for s in spans) != list(range(n)):
            raise ValueError('Missing or duplicate actor span')
        total = sum(sum(s['loss_mask']) for s in spans)
        if any(s['metadata']['edge_tokens'] != total for s in spans):
            raise ValueError('Edge token count mismatch')
        if any(s['reward'] != spans[0]['reward'] for s in spans):
            raise ValueError('Conflicting advantages for one edge')
        for s in spans:
            # For a sum of span-mean losses, this produces one edge-mean loss.
            s['metadata']['edge_loss_scale'] = sum(s['loss_mask']) / total
    return actors, critics


def checkpoint_fields(record, tokenizer, *, sentinel_token_id, max_sequence_length, warmup=False):
    """One scalar value at the final context token using Slime's causal shift.

    Slime slices logits [prompt_length-1:total_length-1]. For a one-token
    response this is precisely the last context position. Causal attention
    prevents that position from reading the appended sentinel. These records
    MUST be routed only to the critic; the sentinel is never a policy sample.
    """
    if record['metadata']['lane'] != 'critic':
        raise ValueError('Expected checkpoint critic record')
    context = record['context']
    tokens = tokenizer.encode(context, add_special_tokens=False)
    if not tokens:
        raise ValueError('Empty tokenized critic context')
    if len(tokens) + 1 > max_sequence_length:
        raise ValueError('Critic context exceeds limit; silent truncation is forbidden')
    target = float(record['target'])
    if not math.isfinite(target):
        raise ValueError('Nonfinite checkpoint target')
    metadata = dict(record['metadata'])
    if warmup and metadata.get('target_source') == 'direct_branch_mean':
        raise ValueError('Warm-up requires empirical suffix targets')
    if warmup:
        diagnostics = metadata['diagnostics']
        if diagnostics['observations'] <= 0:
            raise ValueError('Warm-up needs observed suffix outcomes')
        target = float(diagnostics['mean_return'])
        if not math.isfinite(target) or not 0 <= target <= 1:
            raise ValueError('Invalid empirical warm-up target')
        metadata['target_source'] = 'warmup_empirical_suffix_mean'
    else:
        if metadata.get('target_source') == 'direct_branch_mean':
            diagnostics = metadata['diagnostics']
            if (diagnostics['direct_branches'] <= 0
                    or target != float(diagnostics['mean_return'])
                    or not 0 <= target <= 1):
                raise ValueError('Invalid unshrunk direct-branch critic target')
        else:
            metadata['target_source'] = 'refined_checkpoint_value'
    metadata.update(context_sha256=hashlib.sha256(context.encode()).hexdigest(),
                    context_tokens=len(tokens), value_position=len(tokens)-1)
    return dict(tokens=[*tokens, sentinel_token_id], response_length=1,
                loss_mask=[1], reward=target, group_index=record['group_index'],
                metadata=metadata)


def prepared_advantages(args, rollout_data):
    """Slime custom advantage hook; preserve precomputed scalar targets.

    On critic batches each response contains one checkpoint value target. On
    actor batches rewards are edge advantages, broadcast across response tails.
    Generated-token masks, applied by the loss, exclude all conditioning tokens.
    Role-specific batch routing is mandatory outside this hook.
    """
    import torch
    if args.normalize_advantages:
        raise ValueError('Prepared refined-TD advantages must not be whitened')
    templates = rollout_data.get('values')
    if templates is None:
        templates = rollout_data.get('log_probs')
    if templates is None:
        templates = rollout_data.get('rollout_log_probs')
    if templates is None:
        raise ValueError('No value/logprob tensors available for target placement')
    rewards = rollout_data['rewards']
    if len(rewards) != len(templates):
        raise ValueError('Prepared target/tensor count mismatch')
    targets = []
    for reward, template in zip(rewards, templates):
        if not math.isfinite(float(reward)):
            raise ValueError('Nonfinite prepared target')
        targets.append(torch.full_like(template, float(reward), dtype=torch.float32))
    rollout_data['advantages'] = targets
    rollout_data['returns'] = [t.clone() for t in targets]
