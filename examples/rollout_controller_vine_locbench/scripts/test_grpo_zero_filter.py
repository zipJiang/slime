from advantage_diagnostics import summarize
from targets import zero_advantage_placeholder


def actor(node, *, tokens, reward=0.0, group=0):
    return {
        'group_index': group,
        'tokens': list(range(tokens + 1)),
        'response_length': tokens,
        'reward': reward,
        'rollout_log_probs': [0.0] * tokens,
        'loss_mask': [1] * tokens,
        'metadata': {
            'lane': 'actor',
            'node_id': node,
            'edge_tokens': tokens,
            'group_edge_count': 3,
        },
    }


def test_placeholder_is_shortest_and_fully_masked():
    row = zero_advantage_placeholder([
        actor('long', tokens=7), actor('short', tokens=3), actor('middle', tokens=5)
    ])
    assert row['metadata']['node_id'] == 'short'
    assert row['metadata']['filtered_zero_edges'] == 3
    assert row['metadata']['zero_advantage_placeholder'] is True
    assert not any(row['loss_mask'])
    assert row['metadata']['group_edge_count'] == 3


def test_placeholder_rejects_nonzero_group():
    try:
        zero_advantage_placeholder([actor('zero', tokens=3), actor('live', tokens=3, reward=.5)])
    except ValueError as exc:
        assert 'all-zero' in str(exc)
    else:
        raise AssertionError('Expected a mixed group to be rejected')


def test_all_placeholder_diagnostics_are_finite():
    rows = []
    for group in range(4):
        row = zero_advantage_placeholder([actor(f'e{group}', tokens=3, group=group)])
        rows.append(row)
    report = summarize(rows)
    assert report['edges'] == report['tokens'] == 0
    assert report['zero_advantage_placeholder_groups'] == 4
    assert all(value == 0.0 for key, value in report.items() if key.endswith('_fraction'))
