import pytest

from trace_warmup import baseline, checkpoint_weights, choose_best, packet, report


def row(question, turn, target, node):
    return dict(group_index=question, turn=turn, target=target,
        tokens=[1, 2], response_length=1, reward=target, loss_mask=[1],
        metadata=dict(lane='critic', node_id=node))


def test_long_failed_traces_do_not_dilute_roots_or_other_questions():
    rows = [row('a', 0, 1., 'a0')] + [row('a', i, 0., f'a{i}') for i in range(1, 101)]
    rows += [row('b', 0, 0., 'b0'), row('b', 1, 1., 'b1')]
    weights = checkpoint_weights(rows)
    assert weights[0] == weights[-2] == .25
    assert sum(weights[:101]) == pytest.approx(1.)
    assert sum(weights[101:]) == pytest.approx(1.)
    data = packet(rows)
    assert [1 / d for d in data['rollout_mask_sums']] == pytest.approx(weights)
    assert baseline(rows) == pytest.approx(.5)
    # This reproduces the actual question-averaged regression loss.
    predictions = [.3] * len(rows)
    loss = sum((p-r['target'])**2/d for p,r,d in zip(predictions,rows,data['rollout_mask_sums'])) / 2
    assert report(rows, predictions, .5)['selection_mse'] == pytest.approx(loss)


def test_root_only_and_fold_only_questions_keep_unit_weight():
    rows = [row('a', 0, 1., 'a'), row('b', 1, 0., 'b'), row('b', 2, 1., 'c')]
    assert checkpoint_weights(rows) == [1., .5, .5]


def test_best_selection_retains_starting_critic_and_earliest_tie():
    assert choose_best([{'selection_mse': x} for x in [.1, .2, .12]]) == 0
    assert choose_best([{'selection_mse': x} for x in [.1, .08, .08001, .09]]) == 1
    with pytest.raises(ValueError):
        choose_best([{'selection_mse': float('nan')}])


@pytest.mark.parametrize('mass', [0., 1., -.1, float('nan')])
def test_invalid_weighting_rejected(mass):
    with pytest.raises(ValueError):
        checkpoint_weights([row('a', 0, 0., 'a')], mass)
