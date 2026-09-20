"""Pinned Python 3.13 search collector; the trainer owns both model versions."""
import argparse
import asyncio
from collections import Counter
import gzip
import hashlib
import httpx
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[2]
HARNESS = EXPERIMENT / 'snapshots/harness-vine-k3-sep20'
inherited = set(os.environ.get('PYTHONPATH', '').split(os.pathsep))
sys.path[:] = [str(HARNESS), str(EXPERIMENT/'scripts')] + [
    p for p in sys.path if p not in inherited and p not in (str(ROOT/'slime'), str(ROOT/'rollout-controller'))]

from examples.deontic.bench import load_cases, grade
from examples.deontic.env import RunConfig, build_world
from step_controller import VinePpoConfig, VinePpoEstimator, run_search, ConcurrencyGating, RefinedTdEstimator, DirectBranchTdEstimator, TokenEntropyAllocator, ValueRefinement
from step_controller.allocation import RolloutLedger
from step_controller.config import RolloutConfig
from step_controller.export import to_samples
from step_controller.generation import SamplingParams
from step_controller.generation.policy import PolicyFormat
from step_controller.generation.slime import SlimePolicy
from step_controller.harness import NullWorkspace, build_budget
from step_controller.loop import Runtime, _expander, _value_head, _score_root, rescore_anchor
from step_controller.preparation import prepare_samples
from step_controller.reward import AsyncRewardModel, RewardResult
from step_controller.reward.config import RewardConfig
from step_controller.scheduler.core.policies import Budget
from step_controller.scheduler.core.scheduler import Scheduler
from step_controller.scheduler.core.tree import SchedulerState, SchedulerView
from targets import split_targets, zero_advantage_placeholder
from recipe import RECIPE_ID, target_source
from context_bound import CONTEXT_LIMIT, ContextBoundedCompactor
from episode_horizon import iter_episode, with_episode_horizon


async def post_with_retries(client, url, payload, *, attempts=5, initial_delay=.25):
    """Retry transient transport/server failures with the frozen request."""
    if attempts < 1:
        raise ValueError('attempts must be positive')
    last = None
    for attempt in range(attempts):
        try:
            response = await client.post(url, json=payload)
            if response.status_code == 429 or response.status_code >= 500:
                last = RuntimeError(
                    f'Transient HTTP {response.status_code} from {url}: '
                    f'{response.text[:1000]}')
            else:
                response.raise_for_status()
                return response
        except httpx.TransportError as exc:
            last = exc
        if attempt + 1 < attempts:
            await asyncio.sleep(initial_delay * (2 ** attempt))
    raise RuntimeError(f'POST {url} failed after {attempts} attempts: {last}') from last


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_rows(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with gzip.open(temporary, 'wt') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    temporary.replace(path)


class BatchedValueClient(AsyncRewardModel):
    """Combine concurrent frontier requests before trainer TP collectives.

    Cache only within this frozen version's collection batch. Identical contexts
    have identical checkpoint semantics; labels never enter this cache.
    """
    def __init__(self, client, url, version, batch_size=16):
        self.client, self.url, self.version = client, url, version
        self.batch_size = batch_size
        self.pending = []
        self.cache = {}
        self.worker = None
        self.requests = 0
        self.contexts_scored = 0

    async def ascore(self, context):
        return (await self.ascore_batch([context]))[0]

    async def ascore_batch(self, contexts):
        futures = []
        loop = asyncio.get_running_loop()
        for context in contexts:
            if context not in self.cache:
                future = loop.create_future()
                self.cache[context] = future
                self.pending.append((context, future))
            futures.append(self.cache[context])
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self._drain())
        scores = await asyncio.gather(*(asyncio.shield(f) for f in futures))
        return [RewardResult(score=s) for s in scores]

    async def _drain(self):
        await asyncio.sleep(.005)
        while self.pending:
            batch, self.pending = self.pending[:self.batch_size], self.pending[self.batch_size:]
            try:
                response = await self.client.post(self.url.rstrip('/')+'/score',
                    json=dict(contexts=[c for c, _ in batch], version=self.version))
                response.raise_for_status()
                result = response.json()
                if result['version'] != self.version or len(result['scores']) != len(batch):
                    raise ValueError('Critic result count/version mismatch')
                scores = [float(s) for s in result['scores']]
                if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
                    raise ValueError('Critic prediction is not a finite probability')
                self.requests += 1
                self.contexts_scored += len(batch)
                for (_, future), score in zip(batch, scores, strict=True):
                    if not future.done():
                        future.set_result(score)
            except Exception as exc:
                for _, future in batch:
                    if not future.done():
                        future.set_exception(exc)


def serialize_checkpoint(payload, *, tools, tokenizer, max_steps):
    # Explicit sufficient resumption state, without answer keys or search outcomes.
    # The exact returned string is stored by NodeScorer and reused for training.
    state = dict(messages=list(payload.messages), tools=tools,
                 workspace=dict(payload.workspace.snapshot()),
                 remaining_task_steps=max(0, max_steps-payload.turns_taken),
                 calls_remaining=payload.state.calls_remaining,
                 calls_max=payload.state.calls_max, calls=list(payload.state.calls),
                 done=payload.done, truncated=payload.truncated)
    content = 'Predict the probability that a continuation from this checkpoint succeeds.\n'
    content += json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return tokenizer.apply_chat_template([dict(role='user', content=content)],
        tokenize=False, add_generation_prompt=True)


async def two_pass_search(prompt, workspace, runtime, spend, *, pass_tokens, max_attempts, concurrency):
    root = await runtime.runner.start(prompt, workspace=workspace)
    state = SchedulerState.root(root, gating=ConcurrencyGating(concurrency))
    critic = _value_head(runtime)
    await _score_root(critic, state)
    if state.stats.get('score_failures', 0):
        raise RuntimeError('Root prior failed; refusing unscored refinement')
    view = SchedulerView(state)
    ledger = RolloutLedger(runtime.config.reward_config)
    ledger.bind(view)
    passes = []
    for index, allocator in enumerate((
        TokenEntropyAllocator(window=8, version=runtime.runner.policy.version), ValueRefinement())):
        before = dict(spend)
        old_stats = dict(state.stats)
        cap = spend['output_tokens'] + pass_tokens
        attempts = int(state.stats.get('rollouts', 0) + state.stats.get('failures', 0))
        async with state.lock:
            state.regate(ConcurrencyGating(concurrency))
        allocation = allocator.build(view, ledger)
        if allocation is None:
            raise RuntimeError('Required search pass has no allocation session')
        await Scheduler(expander=with_episode_horizon(_expander(runtime, f'pass-{index}'), runtime.runner.max_steps), allocation=allocation,
            termination=Budget(max_rollouts=attempts+max_attempts,
                               goal=lambda _: spend['output_tokens'] >= cap),
            scorer=critic, max_concurrency=concurrency).run_on(state)
        consumed = {key: value-before.get(key, 0) for key, value in spend.items()}
        passes.append(dict(pass_id=index, allocator=allocator.name, token_target=pass_tokens,
            cost=consumed, output_overshoot=max(0, consumed['output_tokens']-pass_tokens),
            stats={key: value-old_stats.get(key, 0) for key, value in state.stats.items()}))
        if state.stats.get('failures', 0) or state.stats.get('score_failures', 0):
            raise RuntimeError(f'Search failure: {state.failures}; stats={state.stats}')
    await rescore_anchor(runtime, view)
    return state, passes


from balanced_data import corpus_and_split, metadata_for, split_digest, EVAL_SEED, request_seed


class EpisodeRunner:
    max_steps = -1
    def __init__(self, runner): self.runner, self.policy = runner, runner.policy
    async def start(self, *args, **kwargs): return await self.runner.start(*args, **kwargs)
    def iter_run(self, state, *, sampling_params=None, max_steps=None):
        return iter_episode(self.runner, state, task_limit=48, sampling_params=sampling_params)
    async def run(self, state, **kwargs):
        async for state in self.iter_run(state, **kwargs): pass
        return state


async def main(args):
    from transformers import AutoTokenizer
    cases, selected, split_hash = corpus_and_split(args)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'contract.json').exists():
        raise ValueError('Refusing to overwrite/reuse a collection directory')
    seed_namespace = (EVAL_SEED if args.evaluation else
                      (args.seed_namespace or f'{args.policy_version}/{args.value_version}/{args.output.name}'))
    contract = dict(policy_version=args.policy_version, server_weight_version=args.server_weight_version,
        value_version=args.value_version, case_keys=selected,
        case_families=[metadata_for(k)['family'] for k in selected],
        sampling_cells=[[metadata_for(k)['domain'], metadata_for(k)['hard']] for k in selected],
        split_bundle=str(EXPERIMENT/'data/split'), split=args.split, split_sha256=split_hash,
        evaluation=args.evaluation, pass_tokens=args.pass_tokens, max_pass_attempts=args.max_pass_attempts,
        per_question_concurrency=args.concurrency, value_prior_strength=args.prior_strength,
        temperature=1., top_p=1., top_k=-1, submissions=1, max_steps=48,
        prompt_limit=14336, task_reply_limit=6144, fold_reply_limit=4096,
        context_limit=CONTEXT_LIMIT, fold_context_reserve=2048,
        actor_workspace='NullWorkspace', fold_workspace='NullWorkspace',
        workspace_tools=[], fold_tools=[],
        horizon='shared_task_turns_plus_one_forced_submission',
        critic_supervision='none',
        estimator=args.estimator, recipe_id=RECIPE_ID, group_size=args.group_size,
        value_rollouts_per_state=args.value_rollouts_per_state,
        critic_target_source=target_source(args.estimator),
        harness_manifest_sha256=hashlib.sha256((HARNESS/'source-manifest.json').read_bytes()).hexdigest(),
        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seed_namespace=seed_namespace)
    (args.output/'collector-source.py').write_bytes(Path(__file__).read_bytes())
    write_json(args.output/'contract.json', contract)
    if args.prepare_only:
        print(json.dumps(contract, indent=2)); return
    if not args.evaluation and args.estimator != 'vine_ppo' and not args.value_url:
        raise ValueError('Search requires the trainer-owned critic endpoint')
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    profile = PolicyFormat.resolve(args.model, tokenizer=tokenizer, profile='qwen_xml')
    params = SamplingParams(max_tokens=6144, temperature=1., top_p=1., top_k=-1,
                            repetition_penalty=1., logprobs=1)
    async with httpx.AsyncClient(timeout=600., limits=httpx.Limits(max_connections=128)) as client:
        value = BatchedValueClient(client, args.value_url, args.value_version)

        async def one(group, case_key, branch=None):
            start = time.monotonic()
            spend = Counter(input_tokens=0, output_tokens=0, generations=0)
            case = cases[case_key]
            identity = metadata_for(case_key)
            draw = 0
            # Unique per-request seeds; a fork cannot copy and reuse a contextvar counter.
            async def post(url, payload):
                nonlocal draw
                draw_id, draw = draw, draw+1
                payload['sampling_params']['sampling_seed'] = request_seed(seed_namespace, group, case_key, branch, draw_id)
                response = await post_with_retries(client, url, payload)
                output = response.json()
                meta = output['meta_info']
                pairs = meta.get('output_token_logprobs')
                if not pairs or any(p[0] is None or not math.isfinite(float(p[0])) or float(p[0]) > 1e-5 for p in pairs):
                    raise ValueError('Missing/invalid exact behavior logprobs')
                if str(meta.get('weight_version')) != args.server_weight_version:
                    raise ValueError('Behavior weight version differs from driver freeze')
                spend.update(input_tokens=len(payload['input_ids']), output_tokens=len(pairs), generations=1)
                return output
            policy = SlimePolicy(post, args.url.rstrip('/')+'/generate', model=args.model,
                                 format=profile, version=args.policy_version, default_params=params)
            cfg = RunConfig(policy=policy, tools=('list', 'grep', 'read', 'python'),
                max_steps=48, call_budget=10, max_prompt_tokens=14336, python_timeout=10,
                no_workspace=True, no_compact=False, read_max_chars=3000,
                fold_reply_tokens=4096, scorer=lambda answer: float(grade(answer, case)))
            world = build_world(case, cfg)
            if not isinstance(world.workspace, NullWorkspace) or world.workspace.tools():
                raise ValueError('Both-NullWorkspace Vine recipe forbids actor storage')
            # The default fold overlay is temperature .7. Both action types must
            # use the same temperature-1 behavior law as Slime's logprob audit.
            # Folding is a summarization operation, not another task episode.  Keeping
            # the task tools here let a fold issue calls until its own prompt exceeded
            # the model context window.  A NullWorkspace makes the fold tool-free, so
            # the 14K trigger plus 4K reply remains safely inside the 32K backend
            # window.  Deontic summaries are free-form; do not impose BrowserComp's
            # structured-summary schema on them.
            world.runner._compactor = ContextBoundedCompactor(build_budget(
                max_prompt_tokens=14336,
                max_reply_tokens=4096,
                fold_workspace=NullWorkspace(),
                sampling_params=SamplingParams(temperature=1., top_p=1., top_k=-1,
                                               repetition_penalty=1., logprobs=1)))
            if not isinstance(world.runner._compactor.inner._fold_workspace, NullWorkspace):
                raise ValueError('Both-NullWorkspace Vine recipe forbids inherited fold storage')
            if args.evaluation:
                state = await world.runner.start(world.prompt, workspace=world.workspace)
                async for checkpoint in iter_episode(world.runner, state, task_limit=48):
                    state = checkpoint
                result = dict(group_index=group, case_key=case_key, family=identity['family'],
                    domain=identity['domain'], hard=identity['hard'], branch=branch,
                    outcome=float(grade(state.state.answer or '', case)), answer=state.state.answer,
                    task_turns=state.turns_taken, folds=state.folds, truncated=state.truncated,
                    cost=dict(spend), seconds=time.monotonic()-start)
                artifact = args.output/f'group-{group:03d}-branch-{branch}'
                native = state
            else:
                rc = RewardConfig(value_version='mc-outcome', beta_step=0., config_id='vine-outcome-v1')
                runtime = Runtime(runner=EpisodeRunner(world.runner),
                    config=RolloutConfig(reward_config=rc, vine=VinePpoConfig(
                        group_size=args.group_size,
                        value_rollouts_per_state=args.value_rollouts_per_state),
                        max_concurrency=args.concurrency))
                state = await run_search(world.prompt, runtime, workspace=world.workspace)
                passes = []
                if any(n.payload.turns_taken > 49 for n in state.nodes.values()):
                    raise ValueError('Vine tree exceeded shared task horizon')
                prepared = prepare_samples(state, estimator=VinePpoEstimator(), reward_config=rc,
                    behavior_version=args.policy_version)
                # Exact-zero Vine advantages have exactly zero policy gradient.
                # Remove their full edges before token batching; retaining them
                # previously consumed about 80% of actor tokens in round 27.
                actor, critic = split_targets(to_samples(
                    prepared, group_index=group, min_abs_advantage=0.0
                ))
                if critic:
                    raise ValueError('Vine must export actor edges only')
                placeholder = False
                if not actor:
                    unfiltered, unfiltered_critic = split_targets(
                        to_samples(prepared, group_index=group)
                    )
                    if unfiltered_critic:
                        raise ValueError('Vine must export actor edges only')
                    actor = [zero_advantage_placeholder(unfiltered)]
                    placeholder = True
                artifact = args.output/f'group-{group:03d}'
                for rows, lane in ((actor, 'actor'), (critic, 'critic')):
                    for row in rows:
                        row['metadata'].update(case_key=case_key, family=identity['family'],
                            policy_version=args.policy_version,
                            server_weight_version=args.server_weight_version)
                        if lane == 'critic':
                            row['metadata']['terminal_boundary'] = state.nodes[row['metadata']['node_id']].payload.done
                            row['metadata']['target_source'] = target_source(args.estimator)
                    write_rows(artifact.with_suffix(f'.{lane}.jsonl.gz'), rows)
                terminals = [node.payload for node in state.nodes.values() if node.payload.done]
                result = dict(group_index=group, case_key=case_key, family=identity['family'],
                    domain=identity['domain'], hard=identity['hard'], passes=passes, cost=dict(spend),
                    actor_spans=len(actor), actor_tokens=sum(sum(r['loss_mask']) for r in actor),
                    zero_advantage_placeholder=placeholder,
                    critic_checkpoints=len(critic), terminal_count=len(terminals),
                    terminal_correct=sum(p.reward_outcome for p in terminals),
                    stats=dict(state.stats), seconds=time.monotonic()-start)
                native = state
            with gzip.open(artifact.with_suffix('.native.pkl.gz'), 'wb') as stream:
                pickle.dump(native, stream)
            write_json(artifact.with_suffix('.json'), result)
            print(json.dumps(result, allow_nan=False), flush=True)
            return result

        requests = [one(g, case_key, branch) for g, case_key in enumerate(selected)
                    for branch in (range(args.eval_branches) if args.evaluation else (None,))]
        results = await asyncio.gather(*requests)
        summary = dict(policy_version=args.policy_version, server_weight_version=args.server_weight_version,
            value_version=args.value_version, results=results,
            cost={key: sum(r['cost'][key] for r in results) for key in ('input_tokens','output_tokens','generations')},
            critic_requests=value.requests, critic_contexts=value.contexts_scored)
        if args.evaluation:
            summary.update(correct=sum(r['outcome'] for r in results), branches=len(results))
        write_json(args.output/'summary.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--value-url', default='')
    parser.add_argument('--model', default='Qwen/Qwen3.5-9B')
    parser.add_argument('--policy-version', required=True)
    parser.add_argument('--seed-namespace', help='Paired benchmark seeds; heldout seeds remain fixed')
    parser.add_argument('--server-weight-version', required=True)
    parser.add_argument('--value-version', required=True)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['train','val'], default='train')
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--eval-branches', type=int, default=8)
    parser.add_argument('--pass-tokens', type=int, required=True)
    parser.add_argument('--max-pass-attempts', type=int, default=64)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--prior-strength', type=float, default=1.)
    parser.add_argument('--group-size', type=int, default=4)
    parser.add_argument('--value-rollouts-per-state', type=int, default=1)
    parser.add_argument('--estimator', choices=['vine_ppo'], default='vine_ppo')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if (args.pass_tokens <= 0 or args.max_pass_attempts < args.concurrency
            or args.concurrency < 1 or args.group_size < 1
            or args.value_rollouts_per_state < 1):
        parser.error('Positive token budget and valid concurrency/attempt cap required')
    asyncio.run(main(args))
