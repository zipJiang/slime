"""Apply an asynchronous semantic judge to terminal rollout transitions.

BrowseComp's tool environment records a submitted answer but deliberately has no
embedded answer key.  Tree backup still needs the frozen judge outcome before a
terminal child is attached.  This expander wrapper performs that one task without
putting answers or labels into nonterminal critic contexts.
"""
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
import math
from typing import Any

from step_controller.harness import RolloutState
from step_controller.scheduler.core.execution import Expander


Judge = Callable[[str], Awaitable[float]]


def _with_outcome(payload: RolloutState[Any], outcome: float) -> RolloutState[Any]:
    if not payload.done:
        raise ValueError('Semantic outcome can only be attached to a terminal rollout')
    if not math.isfinite(outcome) or outcome not in (0.0, 1.0):
        raise ValueError('Semantic judge must return a binary finite outcome')
    turns = list(payload.turns)
    terminal = [i for i, turn in enumerate(turns)
                if turn.transition is not None and turn.transition.done]
    if terminal != [len(turns)-1]:
        raise ValueError('Expected exactly one final terminal task transition')
    index = terminal[0]
    transition = turns[index].transition
    assert transition is not None
    turns[index] = replace(turns[index], transition=replace(
        transition, reward_outcome=outcome))
    return replace(payload, turns=tuple(turns))


class SemanticRewardExpander(Expander[RolloutState[Any]]):
    """Judge the terminal child returned by an ordinary rollout expander.

    Judge errors propagate through the scheduler's normal failed-expansion path.
    The enclosing collection contract must reject any such failure before export.
    """

    def __init__(self, inner: Expander[RolloutState[Any]], judge: Judge):
        self.inner = inner
        self.judge = judge

    async def expand(self, payload: RolloutState[Any]) -> tuple[
        list[RolloutState[Any]], Mapping[str, object]
    ]:
        chain, metadata = await self.inner.expand(payload)
        terminal = [i for i, child in enumerate(chain) if child.done]
        if terminal and terminal != [len(chain)-1]:
            raise ValueError('Rollout expansion continued beyond a terminal child')
        if terminal:
            answer = getattr(chain[-1].state, 'answer', None)
            outcome = 0.0 if answer is None or not str(answer).strip() else float(
                await self.judge(str(answer)))
            chain[-1] = _with_outcome(chain[-1], outcome)
        return chain, metadata


__all__ = ['SemanticRewardExpander']
