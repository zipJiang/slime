from types import SimpleNamespace

import pytest

from collect_grpo_rollouts import IndependentRootGating


def node(parent_id, children=()):
    return SimpleNamespace(parent_id=parent_id, children=list(children))


def test_independent_root_gating_fans_out_only_at_root():
    gating = IndependentRootGating(8)
    root = node(None, range(7))
    compacted = node(0, ())

    assert gating.gate(root).available(root)
    assert gating.gate(node(None, range(8))).available(node(None, range(8))) is False
    assert gating.gate(compacted).available(compacted)
    assert gating.gate(node(0, [2])).available(node(0, [2])) is False


def test_independent_root_gating_rejects_nonpositive_width():
    with pytest.raises(ValueError, match='positive'):
        IndependentRootGating(0)
