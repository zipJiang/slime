"""Collect independent LocBench root trajectories for step-wise GRPO."""
import argparse
import asyncio
from collections import Counter
from dataclasses import replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
import time

from runtime_v2 import EXPERIMENT, MODEL, CONTEXT_LIMIT, digest, verify_harness
import ppo_runtime
from ppo_runtime import make_world

sys.path.append(str(EXPERIMENT / 'snapshots/native-support-v1'))
from examples.locbench.dataset import load_cases
from examples.locbench.repository import prepare
from examples.locbench.metrics import score
from step_controller import (
    FifoAllocator,
    GatingPolicy,
    GrpoEstimator,
    SearchPass,
    run_search,
)
from step_controller.config import RolloutConfig
from step_controller.export import to_samples
from step_controller.generation import SamplingParams
from step_controller.generation.interfaces import merge_sampling_params
from step_controller.generation.policy import PolicyFormat
from step_controller.generation.slime import SlimePolicy
from step_controller.loop import Runtime
from step_controller.preparation import prepare_samples
from step_controller.reward.config import RewardConfig
from step_controller.scheduler import Gate
from grpo_recipe import RECIPE_ID, target_source
from targets import split_targets, zero_advantage_placeholder


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_rows(path, values):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(gzip.compress(
        ''.join(json.dumps(v, allow_nan=False) + '\n' for v in values).encode(), mtime=0))
    temporary.replace(path)


class BoundedSlimePolicy(SlimePolicy):
    async def agenerate_tokens(self, prefix_tokens, sampling_params=None):
        params = merge_sampling_params(self.default_params, sampling_params)
        remaining = CONTEXT_LIMIT - len(prefix_tokens)
        if remaining <= 0:
            raise ValueError('Input exceeds context; no conditioning truncation')
        return await super().agenerate_tokens(
            prefix_tokens, replace(params, max_tokens=min(params.max_tokens, remaining)))


class IndependentRootGating(GatingPolicy):
    """Fan out at the prompt while keeping each compacted trajectory linear."""

    def __init__(self, width):
        if width < 1:
            raise ValueError('Root width must be positive')
        self.width = width

    def gate(self, node):
        return Gate(self.width if node.parent_id is None else 1, count_children=True)


async def run(args):
    verify_harness()
    import httpx
    from transformers import AutoTokenizer

    split = json.loads((EXPERIMENT / 'data/split.json').read_text())
    selected = json.loads(args.questions.read_text())
    if (not isinstance(selected, list) or len(selected) != args.batch_size
            or len(set(selected)) != len(selected) or not set(selected) <= set(split['train'])
            or set(selected) & (set(split['test']) | set(split['development']))):
        raise ValueError('GRPO collection requires a complete unique training batch')
    cases = {case.id: case for case in load_cases(EXPERIMENT / 'data/train.jsonl')}
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'contract.json').exists():
        raise ValueError('Refusing stale or partial rollout reuse')
    environment = ppo_runtime.environment(args.collection_profile)
    contract = dict(
        protocol='locbench-root-grpo-v1', recipe_id=RECIPE_ID,
        environment=environment, checkpoint=str(args.checkpoint.resolve()),
        policy_version=args.policy_version, server_weight_version=args.server_weight_version,
        value_version=args.value_version, estimator=args.estimator,
        group_size=args.group_size,
        case_ids=selected, batch_size=args.batch_size, concurrency=args.concurrency,
        seed_namespace=args.seed_namespace, temperature=.6, top_p=.95, top_k=20,
        target_source=target_source(args.estimator),
        actor_export=dict(
            min_abs_advantage=0.0,
            zero_group='shortest_fully_masked_placeholder',
            normalization='original_group_edge_count',
        ),
        split_sha256=digest(EXPERIMENT / 'data/split.json'),
        cases_sha256=digest(EXPERIMENT / 'data/train.jsonl'),
        collector_sha256=digest(__file__))
    write(args.output / 'contract.json', contract)
    if args.prepare_only:
        print(json.dumps(contract, indent=2))
        return

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    profile = PolicyFormat.resolve(str(args.checkpoint), tokenizer=tokenizer, profile='qwen_xml')
    params = SamplingParams(max_tokens=environment['actor_reply_limit'], temperature=.6,
                            top_p=.95, top_k=20, logprobs=1)
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=1800., limits=httpx.Limits(max_connections=256)) as http:
        async def one(group, question):
            begin = time.monotonic()
            spend = Counter(input_tokens=0, output_tokens=0, generations=0)
            draw = 0
            case = cases[question]

            async def post(url, payload):
                nonlocal draw
                draw_id, draw = draw, draw + 1
                key = f'{args.seed_namespace}/{group}/{question}/{draw_id}'
                payload['sampling_params']['sampling_seed'] = int(
                    hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 2**31
                response = await http.post(url, json=payload)
                response.raise_for_status()
                output = response.json()
                meta = output['meta_info']
                pairs = meta.get('output_token_logprobs')
                if (not pairs or any(p[0] is None or not math.isfinite(float(p[0]))
                                     or float(p[0]) > 1e-5 for p in pairs)):
                    raise ValueError('Missing or invalid exact behavior logprobs')
                if str(meta.get('weight_version')) != args.server_weight_version:
                    raise ValueError('Behavior version changed during collection')
                spend.update(input_tokens=len(payload['input_ids']),
                             output_tokens=len(pairs), generations=1)
                return output

            policy = BoundedSlimePolicy(
                post, args.url.rstrip('/') + '/generate', model=str(args.checkpoint),
                format=profile, version=args.policy_version, default_params=params)
            repository = await asyncio.to_thread(prepare, args.cache, case.repo, case.base_commit)
            runner, workspace, prompt, _ = make_world(
                case, repository, policy, profile=args.collection_profile)
            reward = RewardConfig(value_version='none', beta_step=0.,
                                  config_id='locbench-grpo-outcome-v1')
            runtime = Runtime(runner=runner, config=RolloutConfig(
                reward_config=reward,
                passes=(SearchPass(FifoAllocator(), budget=args.group_size,
                                   gating=IndependentRootGating(args.group_size)),),
                max_concurrency=min(args.concurrency, args.group_size)))
            state = await run_search(prompt, runtime, workspace=workspace)
            prepared = prepare_samples(state, estimator=GrpoEstimator(),
                                       reward_config=reward,
                                       behavior_version=args.policy_version)
            original_actor_edges = len(prepared.actor)
            original_actor_tokens = sum(edge.edge_tokens for edge in prepared.actor)
            actor, critic = split_targets(to_samples(
                prepared, group_index=group, min_abs_advantage=0.0
            ))
            if critic:
                raise ValueError('GRPO must export actor edges only')
            placeholder = False
            if not actor:
                unfiltered, unfiltered_critic = split_targets(
                    to_samples(prepared, group_index=group)
                )
                if unfiltered_critic:
                    raise ValueError('GRPO must export actor edges only')
                actor = [zero_advantage_placeholder(unfiltered)]
                placeholder = True
            for row in actor:
                row['metadata'].update(
                    query_id=question, policy_version=args.policy_version,
                    server_weight_version=args.server_weight_version,
                    target_source=target_source(args.estimator))
            stem = args.output / f'group-{group:03d}'
            write_rows(stem.with_suffix('.actor.jsonl.gz'), actor)
            terminals = [node.payload for node in state.nodes.values() if node.payload.done]
            rewards = [float(payload.reward_outcome) for payload in terminals]
            for terminal, outcome in zip(terminals, rewards, strict=True):
                if score(terminal.state.locations, case.gold)['reward'] != outcome:
                    raise ValueError('GRPO terminal reward differs from file recall')
            if len(terminals) != args.group_size:
                raise ValueError(
                    f'GRPO requires {args.group_size} complete roots, got {len(terminals)}')
            if state.stats.get('failures', 0):
                raise ValueError('GRPO root generation had a failed branch')
            with gzip.open(stem.with_suffix('.native.pkl.gz'), 'wb') as stream:
                pickle.dump(state, stream)
            result = dict(
                group_index=group, query_id=question, cost=dict(spend),
                actor_spans=len(actor), actor_edges=len({r['metadata']['node_id'] for r in actor}),
                actor_tokens=sum(sum(r['loss_mask']) for r in actor),
                zero_advantage_placeholder=placeholder,
                original_actor_edges=original_actor_edges,
                original_actor_tokens=original_actor_tokens,
                filtered_zero_edges=original_actor_edges - (
                    0 if placeholder else len({r['metadata']['node_id'] for r in actor})
                ),
                filtered_zero_tokens=original_actor_tokens - sum(
                    sum(r['loss_mask']) for r in actor
                ),
                terminal_count=len(terminals), terminal_rewards=rewards,
                terminal_correct=sum(rewards), terminal_exact=sum(v == 1 for v in rewards),
                stats=dict(state.stats), seconds=time.monotonic() - begin)
            write(stem.with_suffix('.json'), result)
            print(json.dumps(result, allow_nan=False), flush=True)
            return result

        results = await asyncio.gather(
            *(one(group, question) for group, question in enumerate(selected)),
            return_exceptions=True)
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            write(args.output / 'failed.json',
                  dict(errors=[repr(error) for error in errors], time=time.time()))
            raise BaseExceptionGroup('GRPO questions failed after draining in-flight work', errors)
        write(args.output / 'summary.json', dict(
            protocol=contract['protocol'], policy_version=args.policy_version,
            server_weight_version=args.server_weight_version,
            value_version=args.value_version, results=results,
            cost={key: sum(result['cost'][key] for result in results)
                  for key in ('input_tokens', 'output_tokens', 'generations')},
            critic_requests=0, critic_contexts=0, seconds=time.monotonic() - started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path(MODEL))
    for name in ('questions', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path(
        '/weka/scratch/jhu/bvandur1/zjiang31/rollout-controller/data/locbench/repositories'))
    for name in ('url', 'policy-version', 'server-weight-version', 'value-version', 'seed-namespace'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--concurrency', type=int, default=12)
    parser.add_argument('--group-size', type=int, default=8)
    parser.add_argument('--collection-profile', choices=ppo_runtime.COLLECTION_PROFILES,
                        default='compact24-reply8-read60')
    parser.add_argument('--estimator', choices=['grpo'], default='grpo')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.concurrency < 1 or args.group_size < 2:
        parser.error('Positive batch/concurrency and group size >= 2 required')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
