import asyncio
from types import SimpleNamespace

import pytest

from remaining_value import RemainingValueScorer


class RecordingScorer:
    version = "critic-v1"

    def __init__(self):
        self.calls = []
        self.writes = []

    async def score(self, payloads):
        self.calls.append([payload.label for payload in payloads])
        await asyncio.sleep(0)
        return [f"prediction:{payload.label}" for payload in payloads]

    def write(self, nodes, scores):
        self.writes.append(([node.label for node in nodes], list(scores)))


def payload(label, *, done=False):
    return SimpleNamespace(label=label, done=done)


@pytest.mark.asyncio
async def test_terminal_contexts_are_neither_scored_nor_written():
    inner = RecordingScorer()
    scorer = RemainingValueScorer(inner)
    chain = [payload("fold"), payload("answer", done=True)]
    scores = await scorer.score(chain)

    scorer.write(chain, scores)

    assert scorer.version == "critic-v1"
    assert inner.calls == [["fold"]]
    assert scores == ["prediction:fold", None]
    assert inner.writes == [(["fold"], ["prediction:fold"])]


@pytest.mark.asyncio
async def test_concurrent_chains_keep_their_own_alignment():
    inner = RecordingScorer()
    scorer = RemainingValueScorer(inner)
    first = [payload("first"), payload("first-terminal", done=True)]
    second = [payload("second-terminal", done=True), payload("second")]

    first_scores, second_scores = await asyncio.gather(
        scorer.score(first), scorer.score(second)
    )
    scorer.write(first, first_scores)
    scorer.write(second, second_scores)

    assert first_scores == ["prediction:first", None]
    assert second_scores == [None, "prediction:second"]
    assert inner.writes == [
        (["first"], ["prediction:first"]),
        (["second"], ["prediction:second"]),
    ]
