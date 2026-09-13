"""Bound the inner compaction episode without discarding conditioning tokens."""
from dataclasses import replace
import logging

from step_controller.harness.compaction.base import Compactor
from step_controller.harness.runner import Runner

CONTEXT_LIMIT = 32768


class BoundedFoldRunner(Runner):
    async def advance(self, rs, *, sampling_params=None):
        if rs.done or rs.truncated:
            return rs
        count = len(self.policy.prepare(rs.messages, tools=self._tools or ()).tokens)
        reply = self._params(sampling_params).max_tokens
        # Leave room for this reply, tool framing, and the forced final summary.
        # The ordinary interaction budget is still enforced by Runner.iter_run.
        if count + 2 * reply + 2048 >= CONTEXT_LIMIT:
            logging.getLogger(__name__).info('Finalizing fold near context limit: %d tokens', count)
            return await self.finish(rs, sampling_params=sampling_params)
        return await super().advance(rs, sampling_params=sampling_params)

    async def _generate_turn(self, rs, messages, cond, *, sampling_params, ending):
        params = self._params(sampling_params)
        available = CONTEXT_LIMIT - len(cond.tokens)
        if available <= 0:
            raise ValueError('Fold input exceeds context window; refusing prompt truncation')
        if available < params.max_tokens:
            logging.getLogger(__name__).info('Bounding fold reply to %d tokens', available)
            params = replace(params, max_tokens=available)
        return await super()._generate_turn(rs, messages, cond,
            sampling_params=params, ending=ending)


class _Derivation:
    def __init__(self, runner):
        self.runner = runner

    def derive(self, **kwargs):
        return BoundedFoldRunner(policy=self.runner.policy, compactor=None,
            generate_timeout=self.runner._generate_timeout, **kwargs)


class ContextBoundedCompactor(Compactor):
    def __init__(self, inner):
        self.inner = inner
        self.trigger = inner.trigger

    async def compact(self, compaction):
        return await self.inner.compact(replace(compaction, runner=_Derivation(compaction.runner)))
