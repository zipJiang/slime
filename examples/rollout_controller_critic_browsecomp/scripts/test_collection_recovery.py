import asyncio
from types import SimpleNamespace

import pytest

from collect import collect_all
from context_bound import BoundedFoldRunner, CONTEXT_LIMIT, FOLD_OVERFLOW_LIMIT
from step_controller.generation import SamplingParams
from step_controller.harness.runner import Runner


@pytest.mark.parametrize('count,ending,expected', [
    (1000, False, 4096),
    (CONTEXT_LIMIT - 100, False, 100),
    (CONTEXT_LIMIT + 100, True, 4096),
    (FOLD_OVERFLOW_LIMIT - 100, True, 100),
])
def test_fold_generation_preserves_prompt_and_respects_window(monkeypatch, count, ending, expected):
    runner = object.__new__(BoundedFoldRunner)
    params = SamplingParams(max_tokens=4096)
    monkeypatch.setattr(runner, '_params', lambda _: params)
    cond = SimpleNamespace(tokens=tuple(range(count)))
    original = cond.tokens

    async def generate(self, rs, messages, prepared, *, sampling_params, ending):
        assert prepared is cond and prepared.tokens is original
        assert sampling_params.max_tokens == expected
        assert count + expected <= (FOLD_OVERFLOW_LIMIT if ending else CONTEXT_LIMIT)
        return 'generated'

    monkeypatch.setattr(Runner, '_generate_turn', generate)
    assert asyncio.run(runner._generate_turn(None, [], cond, sampling_params=params, ending=ending)) == 'generated'


@pytest.mark.parametrize('count,ending', [(CONTEXT_LIMIT, False), (FOLD_OVERFLOW_LIMIT, True)])
def test_fold_rejects_inputs_that_cannot_fit(monkeypatch, count, ending):
    runner = object.__new__(BoundedFoldRunner)
    monkeypatch.setattr(runner, '_params', lambda _: SamplingParams(max_tokens=4096))
    with pytest.raises(ValueError, match='refusing prompt truncation'):
        asyncio.run(runner._generate_turn(None, [], SimpleNamespace(tokens=range(count)),
            sampling_params=None, ending=ending))


def test_failed_episode_drains_other_work_before_raising():
    finished = []

    async def fail():
        raise ValueError('one episode failed')

    async def succeed(i):
        await asyncio.sleep(.01)
        finished.append(i)

    async def run():
        await collect_all([succeed(1), fail(), succeed(2)])

    with pytest.raises(ExceptionGroup) as error:
        asyncio.run(run())
    assert sorted(finished) == [1, 2]
    assert len(error.value.exceptions) == 1
    assert isinstance(error.value.exceptions[0], ValueError)
