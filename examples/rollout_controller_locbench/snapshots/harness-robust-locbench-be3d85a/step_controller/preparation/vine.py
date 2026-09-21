"""Independent k=1 outcome-value differences on retained VinePPO edges."""

from __future__ import annotations

import math
from collections.abc import Sequence

from step_controller.harness import AnyRollout
from step_controller.preparation.base import (
    AdvantageEstimator,
    _resolved,
    register_estimator,
)
from step_controller.preparation.records import ActorSample, CriticSample
from step_controller.reward.config import RewardConfig
from step_controller.scheduler.core.tree import SchedulerView
from step_controller.vine import VINE_READY, VINE_VALUE


@register_estimator("vine_ppo")
class VinePpoEstimator(AdvantageEstimator):
    """Outcome-only credit: V(after) - V(before), no normalization or critic."""

    requires_completed_paths = True

    def critic_targets(
        self, *, view: SchedulerView[AnyRollout], reward_config: RewardConfig
    ) -> Sequence[CriticSample]:
        return ()

    def assign(
        self,
        records: Sequence[ActorSample],
        *,
        view: SchedulerView[AnyRollout],
        reward_config: RewardConfig,
    ) -> None:
        if view.root.metadata.get(VINE_READY) is not True:
            raise ValueError("vine_ppo requires completed independent VinePPO probes")
        if reward_config.gamma != 1.0:
            raise ValueError("VinePPO outcome differences require gamma=1")
        for record, node in _resolved(records, view):
            parent = view.node(record.group_id)
            if parent is None or node.parent_id != parent.id:
                raise ValueError("VinePPO edge has no matching parent")
            before, after = (
                parent.metadata.get(VINE_VALUE),
                node.metadata.get(VINE_VALUE),
            )
            if not isinstance(before, (int, float)) or not isinstance(
                after, (int, float)
            ):
                raise ValueError("VinePPO edge is missing a Monte Carlo value")
            if not math.isfinite(before) or not math.isfinite(after):
                raise ValueError("VinePPO edge requires finite Monte Carlo values")
            record.weight = float(after) - float(before)
            record.diagnostics = {
                "advantage": record.weight,
                "parent_value": float(before),
                "successor_value": float(after),
                "value_rollouts_per_state": 1.0,
            }
