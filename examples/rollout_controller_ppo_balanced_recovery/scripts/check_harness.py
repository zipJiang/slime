"""Python 3.13 integration checks against the pinned controller, no GPUs."""
import asyncio
from collections import Counter
import pickle

from collect_rollouts import BatchedValueClient, two_pass_search
from tests.helpers import ScriptedGenerator, SubmitEnv, tool_call, toy_runner
from step_controller import RefinedTdEstimator, DirectBranchTdEstimator
from step_controller.config import RolloutConfig
from step_controller.export import to_samples
from step_controller.generation.parsing import HermesToolCallParser
from step_controller.loop import Runtime
from step_controller.preparation import prepare_samples
from step_controller.reward import AsyncRewardModel, RewardResult
from step_controller.reward.config import RewardConfig
from targets import split_targets
from context_bound import BoundedFoldRunner, ContextBoundedCompactor, CONTEXT_LIMIT
from step_controller.generation import SamplingParams
from step_controller.harness import AgenticCompactor, Compaction, FoldEnv, InMemoryWorkspace, Compactor, CompactionResult, PromptTokens
from episode_horizon import iter_episode, with_episode_horizon
from step_controller.scheduler.core.execution import RolloutExpander


class Value(AsyncRewardModel):
    async def ascore(self, context):
        return RewardResult(.5)


class Client:
    calls = 0
    async def post(self, url, json):
        self.calls += 1
        assert json['version'] == 'critic-0'
        scores = [.25] * len(json['contexts'])
        class Response:
            def raise_for_status(self):
                pass
            def json(self):
                return dict(version='critic-0', scores=scores)
        return Response()


async def main():
    # A resumed suffix gets only its remaining task turns. Compare the native
    # search expander and root-evaluation loop, including the final submission.
    gen = ScriptedGenerator(['continue']*10, logprobs=-.5)
    class KeepCompactor(Compactor):
        trigger = PromptTokens(999999)
        async def compact(self, compaction):
            return CompactionResult(compaction.messages, compaction.state)
    runner = toy_runner(gen, max_steps=3, compactor=KeepCompactor())
    root = await runner.start('q')
    prefix = await runner.advance(root)
    prefix = await runner.advance(prefix)
    prefix = await runner.compact(prefix)
    chain, _ = await with_episode_horizon(RolloutExpander(runner), 3).expand(prefix)
    assert chain[-1].turns_taken == 4 and chain[-1].truncated
    assert gen.calls == 4  # Two shared-prefix turns, one continuation, one ending.
    at_limit = await runner.advance(prefix.fork())
    at_limit = await runner.compact(at_limit)
    before = gen.calls
    chain, _ = await with_episode_horizon(RolloutExpander(runner), 3).expand(at_limit)
    assert gen.calls == before+1 and chain[-1].turns_taken == 4
    async for terminal in iter_episode(runner, root.fork(), task_limit=3):
        pass
    assert terminal.turns_taken == 4 and terminal.truncated
    # A long fold finishes through the native environment, retaining its exact
    # prefix and sampled final reply, rather than issuing another tool round.
    gen = ScriptedGenerator(['r</think>kept'], logprobs=-.5)
    parent = toy_runner(gen, parser=HermesToolCallParser())
    fold = BoundedFoldRunner(policy=parent.policy, env=FoldEnv(), system_prompt='fold',
        sampling_params=SamplingParams(max_tokens=4096, temperature=1.),
        max_steps=8, finish_prompt='Finish now.')
    root = await fold.root((dict(role='user', content='x'*24500),))
    done = await fold.run(root)
    assert done.done and done.truncated and len(done.turns) == 1
    assert done.state.kept == 'kept' and len(gen.prefixes[0]) + gen.params[0].max_tokens <= CONTEXT_LIMIT
    # Regression for a prompt + requested completion just over the server limit.
    root = await fold.root((dict(role='user', content='x'*29000),))
    done = await fold.run(root)
    assert done.done and gen.params[-1].max_tokens < 4096
    assert len(gen.prefixes[-1]) + gen.params[-1].max_tokens == CONTEXT_LIMIT
    # Exercise the compactor's derivation and fold-turn retagging as used live.
    compact = ContextBoundedCompactor(AgenticCompactor(max_reply_tokens=4096,
        sampling_params=SamplingParams(temperature=1.)))
    result = await compact.compact(Compaction(messages=(dict(role='user', content='x'*24500),),
        seed=(dict(role='user', content='q'),), state=None,
        workspace=InMemoryWorkspace(), runner=parent))
    assert len(result.turns) == 1 and result.turns[0].tag == 'fold'
    client = Client()
    value = BatchedValueClient(client, 'http://unused', 'critic-0')
    a, b = await asyncio.gather(value.ascore_batch(['a', 'b']), value.ascore('a'))
    assert client.calls == 1 and len(a) == 2 and b.score == .25
    assert (await value.ascore('a')).score == .25 and client.calls == 1
    generator = ScriptedGenerator(['r</think>'+tool_call('submit', answer='42')]*8, logprobs=-.5)
    runner = toy_runner(generator, env=SubmitEnv([]), parser=HermesToolCallParser(), version='v1')
    rc = RewardConfig(value_version='value1')
    runtime = Runtime(runner=runner, value_model=Value(), config=RolloutConfig(reward_config=rc))
    state, passes = await two_pass_search('Question?', None, runtime,
        Counter(input_tokens=0, output_tokens=0, generations=0),
        pass_tokens=1, max_attempts=2, concurrency=2)
    prepared = prepare_samples(state, estimator=RefinedTdEstimator(), reward_config=rc, behavior_version='v1')
    actor, critic = split_targets(to_samples(prepared, 0))
    assert len(passes) == 2 and len(actor) == 4 and len(critic) == 5
    assert state.stats['failures'] == state.stats['score_failures'] == 0
    assert len(pickle.loads(pickle.dumps(state)).nodes) == len(state.nodes)
    assert all(r['target'] == 0 for r in critic if r['metadata']['node_id'] != 0)
    before = pickle.dumps(state)
    direct = prepare_samples(state, estimator=DirectBranchTdEstimator(), reward_config=rc, behavior_version='v1')
    actor, critic = split_targets(to_samples(direct, 0))
    assert len(actor) == 4 and len(critic) == 1
    assert critic[0]['metadata']['node_id'] == 0
    assert critic[0]['target'] == critic[0]['metadata']['diagnostics']['mean_return'] == 1.
    assert critic[0]['metadata']['diagnostics']['direct_branches'] == 4
    assert pickle.dumps(state) == before
    print('PASS: direct-branch export, frozen targets, two-pass search, role exports, terminal zero values, tree persistence, batched value RPC.')


if __name__ == '__main__':
    asyncio.run(main())
