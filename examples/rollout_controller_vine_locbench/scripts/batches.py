"""Role-specific batches using Slime's native packing and loss denominator.

Each optimizer batch contains all questions. Actor loss is averaged over
generated tokens within an edge, then edges within a question, then questions.
Critic loss is averaged over checkpoints within each question, then questions.
Packing boundaries and the number of serialized spans do not change weighting.
"""
import copy
from collections import defaultdict


def align_actor_records(records, alignment):
    """Return actor records whose count is divisible by the DP alignment.

    Dynamic packing cannot create more non-empty microbatches than records.  If
    every record already occupies its own microbatch and the record count is
    just below the next DP multiple, Slime otherwise aborts while trying to
    give every rank the same number of microbatches.  Split a record's loss
    mask over repeated conditioning to supply the missing microbatch.  The two
    masks are disjoint and retain the original edge metadata, so this changes
    packing only: no trainable token is duplicated and the question/edge loss
    denominator is unchanged.
    """
    if not isinstance(alignment, int) or alignment < 1:
        raise ValueError('Actor record alignment must be a positive integer')
    rows = copy.deepcopy(records)
    if not rows:
        return rows
    target = ((len(rows) + alignment - 1) // alignment) * alignment
    while len(rows) < target:
        candidates = [
            (sum(bool(value) for value in row['loss_mask']), index)
            for index, row in enumerate(rows)
        ]
        trainable, index = max(candidates)
        if trainable < 2:
            raise ValueError('Not enough trainable tokens to align a sparse actor batch')
        original = rows[index]
        positions = [i for i, value in enumerate(original['loss_mask']) if value]
        midpoint = len(positions) // 2
        left, right = copy.deepcopy(original), copy.deepcopy(original)
        for i in positions[midpoint:]:
            left['loss_mask'][i] = 0
        for i in positions[:midpoint]:
            right['loss_mask'][i] = 0
        for row in (left, right):
            row['metadata']['native_mask_partition'] = True
        rows[index:index + 1] = [left, right]
    return rows


def training_data(records, *, lane, expected_groups):
    if lane not in ('actor', 'critic'):
        raise ValueError('Unknown role')
    if not records or {r['group_index'] for r in records} != set(expected_groups):
        raise ValueError('Every question must contribute to the role batch')
    units = defaultdict(set)
    for row in records:
        if row['metadata']['lane'] != lane:
            raise ValueError('Mixed actor/critic batch')
        units[row['group_index']].add(row['metadata']['node_id'])
    original_counts = {}
    for row in records:
        group = row['group_index']
        count = (row['metadata'].get('group_edge_count', len(units[group]))
                 if lane == 'actor' else len(units[group]))
        if type(count) is not int or count < len(units[group]):
            raise ValueError('Invalid original actor edge count')
        if original_counts.setdefault(group, count) != count:
            raise ValueError('Conflicting original actor edge counts')
    denominators = []
    for row in records:
        group = row['group_index']
        size = row['metadata']['edge_tokens'] if lane == 'actor' else 1
        if size <= 0:
            raise ValueError('Empty loss unit')
        denominators.append(size * original_counts[group])
    data = dict(
        tokens=[r['tokens'] for r in records],
        response_lengths=[r['response_length'] for r in records],
        rewards=[r['reward'] for r in records],
        # Retain for transport; these are prepared coefficients/targets, not
        # task accuracy. Outcome metrics must come from terminal branch files.
        raw_reward=[r['reward'] for r in records],
        truncated=[0] * len(records),
        sample_indices=list(range(len(records))),
        rollout_ids=[r['group_index'] for r in records],
        loss_masks=[r['loss_mask'] for r in records],
        rollout_mask_sums=denominators,
    )
    if lane == 'actor':
        data['rollout_log_probs'] = [r['rollout_log_probs'] for r in records]
    return data


def partition_data(args, parallel_config, data):
    """Build native DP packets, rejecting dropped samples or extra updates."""
    from slime.utils.dp_schedule import build_dp_schedule
    from slime.observability.rollout_data_utils import tensorize_rollout_data_for_training
    if args.calculate_per_token_loss:
        raise ValueError('Per-token global reduction overrides question/edge weighting')
    lengths = [len(t) for t in data['tokens']]
    groups = set(data['rollout_ids'])
    if len(groups) != args.global_batch_size:
        raise ValueError('One fresh question batch must produce one optimizer update')
    parts, indices, microbatches, sizes = build_dp_schedule(
        args, parallel_config, lengths, global_batch_size=len(groups),
        rollout_indices=data['rollout_ids'])
    if sizes != [len(groups)]:
        raise ValueError('Expected exactly one optimizer update')
    placed = [i for part in parts for i in part]
    if sorted(placed) != list(range(len(lengths))):
        raise ValueError('Slime packing dropped or duplicated a training record')
    result = []
    for rank, part in enumerate(parts):
        packet = {k: [v[i] for i in part] for k, v in data.items() if k != 'raw_reward'}
        packet.update(partition=part, total_lengths=lengths, raw_reward=data['raw_reward'],
                      global_batch_sizes=sizes, num_microbatches=microbatches,
                      micro_batch_indices=indices[rank])
        tensorize_rollout_data_for_training(packet)
        result.append(packet)
    return result


def put_packets(packets):
    import ray
    from slime.utils.misc import Box
    return [Box(ray.put(packet)) for packet in packets]
