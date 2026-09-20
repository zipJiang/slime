"""Regression tests for sparse VinePPO actor-batch DP alignment."""

from batches import align_actor_records, training_data


def record(index):
    return {
        'group_index': index % 4,
        'tokens': [10, 11, 12, 13, 14, 15],
        'response_length': 5,
        'reward': 0.25,
        'rollout_log_probs': [0.0] * 5,
        'loss_mask': [0, 1, 1, 1, 1, 1],
        'metadata': {
            'lane': 'actor',
            'node_id': f'edge-{index}',
            'edge_tokens': 5,
        },
    }


def test_odd_singleton_batch_is_mask_partitioned_for_dp2():
    original = [record(i) for i in range(51)]
    aligned = align_actor_records(original, 2)

    assert len(aligned) == 52
    assert len(original) == 51
    split = [row for row in aligned if row['metadata'].get('native_mask_partition')]
    assert len(split) == 2
    assert split[0]['metadata']['node_id'] == split[1]['metadata']['node_id']
    assert all(not (left and right) for left, right in zip(split[0]['loss_mask'], split[1]['loss_mask']))
    assert [bool(left or right) for left, right in zip(split[0]['loss_mask'], split[1]['loss_mask'])] == [
        bool(value) for value in original[-1]['loss_mask']
    ]

    packet = training_data(aligned, lane='actor', expected_groups=range(4))
    assert len(packet['tokens']) == 52
    # The split copies retain one node id, so the edge-count denominator stays
    # at the original number of unique edges within each question.
    assert packet['rollout_mask_sums'][-1] == packet['rollout_mask_sums'][-2] == 5 * 13


def test_aligned_batch_is_unchanged():
    original = [record(i) for i in range(52)]
    assert align_actor_records(original, 2) == original


def test_alignment_requires_splittable_trainable_mask():
    rows = [record(i) for i in range(51)]
    for row in rows:
        row['loss_mask'] = [0, 0, 0, 0, 0, 1]
    try:
        align_actor_records(rows, 2)
    except ValueError as exc:
        assert 'trainable tokens' in str(exc)
    else:
        raise AssertionError('Expected sparse unsplittable batch to fail')


def test_filtered_edges_keep_original_question_denominator():
    rows = [record(i) for i in range(4)]
    for row in rows:
        row['metadata']['group_edge_count'] = 8
    packet = training_data(rows, lane='actor', expected_groups=range(4))
    assert packet['rollout_mask_sums'] == [5 * 8] * 4
