from dataclasses import dataclass

import pytest

from semantic_reward import SemanticRewardExpander
from step_controller.harness import NullWorkspace, RolloutState
from step_controller.harness.env import StepResult
from step_controller.harness.turn import Turn
from step_controller.scheduler.core.execution import Expander


@dataclass(frozen=True)
class State:
    answer: str | None = None


def rollout(answer=None, *, done=False):
    transition = StepResult(State(answer), done=done)
    turn = Turn(prefix=(1,), tokens=(2,), logprobs={'actor': (-.1,)},
                transition=transition)
    return RolloutState(messages=(), seed=(), state=State(answer),
                        workspace=NullWorkspace(), turns=(turn,))


class StaticExpander(Expander):
    def __init__(self, chain):
        self.chain = chain

    async def expand(self, payload):
        return list(self.chain), {'source': 'test'}


@pytest.mark.asyncio
async def test_terminal_answer_is_judged_before_attachment():
    calls = []
    async def judge(answer):
        calls.append(answer)
        return 1.0
    expander = SemanticRewardExpander(
        StaticExpander([rollout(), rollout('Paris', done=True)]), judge)
    chain, metadata = await expander.expand(rollout())
    assert calls == ['Paris']
    assert chain[0].reward_outcome == 0
    assert chain[-1].reward_outcome == 1
    assert metadata == {'source': 'test'}


@pytest.mark.asyncio
async def test_empty_answer_is_a_failure_without_calling_judge():
    async def judge(answer):
        raise AssertionError('empty answer must not reach the judge')
    expander = SemanticRewardExpander(StaticExpander([rollout('', done=True)]), judge)
    chain, _ = await expander.expand(rollout())
    assert chain[-1].reward_outcome == 0


@pytest.mark.asyncio
async def test_judge_failure_propagates_fail_closed():
    async def judge(answer):
        raise RuntimeError('judge unavailable')
    expander = SemanticRewardExpander(StaticExpander([rollout('Paris', done=True)]), judge)
    with pytest.raises(RuntimeError, match='judge unavailable'):
        await expander.expand(rollout())


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', [-1, .2, 2, float('nan')])
async def test_nonbinary_judge_outcomes_are_rejected(outcome):
    async def judge(answer):
        return outcome
    expander = SemanticRewardExpander(StaticExpander([rollout('Paris', done=True)]), judge)
    with pytest.raises(ValueError, match='binary finite'):
        await expander.expand(rollout())
