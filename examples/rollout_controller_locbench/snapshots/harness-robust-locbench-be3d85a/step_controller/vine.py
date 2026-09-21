"""VinePPO collection: retained root paths and disposable Monte Carlo probes."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING

from step_controller.allocation import RolloutLedger
from step_controller.harness import AnyRollout
from step_controller.scheduler.core.execution import EdgeProvenance, stamp_provenance
from step_controller.scheduler.core.gating import ConcurrencyGating
from step_controller.scheduler.core.tree import SchedulerState, SchedulerView

if TYPE_CHECKING:
    from step_controller.loop import Runtime

VINE_VALUE = "vine_outcome_value"
VINE_READY = "vine_complete"


async def _bounded_map[T, R](
    items: Sequence[T], operation: Callable[[T], Awaitable[R]], concurrency: int
) -> list[R]:
    # TaskGroup cancels siblings on failure; no detached sampling after an error.
    semaphore = asyncio.Semaphore(concurrency)

    async def one(item: T) -> R:
        async with semaphore:
            return await operation(item)

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(one(item)) for item in items]
    return [task.result() for task in tasks]


async def run_vine_search(
    root: AnyRollout, runtime: Runtime
) -> SchedulerState[AnyRollout]:
    """Collect all actor paths first, then one independent probe per live state.

    The root probe is shared across root trajectories, independent of every actor
    action. Intermediate estimates are shared by the incoming and outgoing edge.
    Only scalar outcomes and compute counters survive the probe phase.
    """
    config = runtime.config
    vine = config.vine
    assert vine is not None
    if not root.forkable:
        raise ValueError("VinePPO requires a live forkable root")
    state: SchedulerState[AnyRollout] = SchedulerState.root(
        root, gating=ConcurrencyGating(1)
    )
    RolloutLedger(config.reward_config).bind(SchedulerView(state))
    state.nodes[0].metadata[VINE_READY] = False
    for kind in ("actor", "value"):
        for metric in ("rollouts", "turns", "generated_tokens", "prompt_tokens"):
            state.stats[f"vine_{kind}_{metric}"] = 0.0

    def account(before: AnyRollout, after: AnyRollout, kind: str) -> None:
        turns = after.edge(before)
        cost = {
            "rollouts": 1,
            "turns": sum(t.transition is not None for t in turns),
            "generated_tokens": sum(len(t.tokens) for t in turns),
            "prompt_tokens": sum(len(t.prefix) for t in turns),
        }
        for metric, value in cost.items():
            state.stats[f"vine_{kind}_{metric}"] += value
        state.stats["rollouts"] += 1
        state.stats["turns"] += cost["turns"]

    def outcome(leaf: AnyRollout) -> float:
        # Do not silently treat censored continuations or backend failures as zero.
        if not leaf.done:
            raise ValueError(
                "VinePPO requires completed outcomes, not truncated rollouts"
            )
        value = leaf.reward_outcome
        if not math.isfinite(value):
            raise ValueError("VinePPO outcome must be finite")
        return value

    async def actor(index: int) -> list[AnyRollout]:
        del index
        _, params = config.proposal.draw()
        chain = [
            rs
            async for rs in runtime.runner.iter_run(root.fork(), sampling_params=params)
        ]
        if not chain or len(chain[-1].turns) <= len(root.turns):
            raise ValueError("VinePPO actor rollout made no progress")
        outcome(chain[-1])
        account(root, chain[-1], "actor")
        return chain

    chains = await _bounded_map(
        list(range(vine.group_size)), actor, config.max_concurrency
    )
    stamp = EdgeProvenance(
        proposal_id=config.proposal.proposal_id,
        continuation_id=config.continuation_id,
        behavior_version=runtime.runner.policy.version,
        reward_config_id=config.reward_config.config_id,
        pass_id="vine-actor",
    )
    remaining_steps = {state.root_id: runtime.runner.max_steps}
    for chain in chains:
        for node in state.attach_chain(state.root_id, chain):
            assert node.parent_id is not None
            parent = state.nodes[node.parent_id]
            remaining = remaining_steps[parent.id]
            # iter_run yields each fold separately. Each task transition and each
            # fold-only advance consumes one unit of its local horizon. A probe
            # must not receive a fresh horizon merely because it starts mid-path.
            consumed = sum(
                t.transition is not None for t in node.payload.edge(parent.payload)
            )
            if not node.payload.done:
                consumed += 1
            remaining_steps[node.id] = (
                max(0, remaining - consumed) if remaining >= 0 else -1
            )
            stamp_provenance(node, stamp)
            if node.payload.done:
                node.metadata[VINE_VALUE] = outcome(node.payload)
    state.stats["nodes"] = float(len(state.nodes) - 1)
    del chains
    live = [node for node in state.nodes.values() if not node.payload.done]

    async def probe(index: int) -> float:
        node = live[index]
        _, params = config.proposal.draw()
        leaf = await runtime.runner.run(
            node.payload.fork(),
            sampling_params=params,
            max_steps=remaining_steps[node.id],
        )
        value = outcome(leaf)
        account(node.payload, leaf, "value")
        return value  # discard the entire auxiliary trajectory here

    values = await _bounded_map(list(range(len(live))), probe, config.max_concurrency)
    for node, value in zip(live, values, strict=True):
        node.metadata[VINE_VALUE] = value
    state.stats["vine_trainable_edges"] = float(len(state.nodes) - 1)
    state.nodes[0].metadata[VINE_READY] = True
    return state
