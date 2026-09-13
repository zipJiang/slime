"""Adapt a node scorer to predict remaining value only where it exists."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class RemainingValueScorer:
    """Skip terminal payloads while preserving scheduler score/node alignment.

    A terminal has zero remaining return by definition: its realized outcome belongs
    to the incoming edge.  ``Scheduler`` scores a complete expansion chain and later
    hands the returned sequence back to ``write`` alongside the created nodes, so this
    adapter keeps explicit ``None`` placeholders rather than losing their positions.
    The placeholders live only between those two calls and make concurrent expansion
    calls independent; no per-call mask is stored on the shared scorer.
    """

    def __init__(self, scorer: Any) -> None:
        self._scorer = scorer
        self.version = scorer.version

    async def score(self, payloads: Sequence[Any]) -> list[Any | None]:
        live = [payload for payload in payloads if not payload.done]
        predictions = iter(await self._scorer.score(live))
        aligned = [None if payload.done else next(predictions) for payload in payloads]
        try:
            next(predictions)
        except StopIteration:
            return aligned
        raise ValueError("remaining-value scorer returned excess predictions")

    def write(self, nodes: Sequence[Any], scores: Sequence[Any | None]) -> None:
        pairs = [(node, score) for node, score in zip(nodes, scores, strict=True)
                 if score is not None]
        self._scorer.write([node for node, _ in pairs],
                           [score for _, score in pairs])
