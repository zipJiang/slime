"""Evaluate saved PPO critics against checkpoint-matched oversampled MCTS targets."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextvars import ContextVar
from dataclasses import replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
import time

from runtime_v2 import CONTEXT_LIMIT, EXPERIMENT, digest, verify_harness
import ppo_runtime
from ppo_runtime import make_world

sys.path.append(str(EXPERIMENT / "snapshots/native-support-v1"))
from examples.locbench.dataset import load_cases
from examples.locbench.repository import prepare
from step_controller import ConcurrencyGating, TokenEntropyAllocator, ValueRefinement
from step_controller.allocation import RolloutLedger
from step_controller.allocation.evidence import evidence_of
from step_controller.config import RolloutConfig
from step_controller.generation import SamplingParams
from step_controller.generation.interfaces import merge_sampling_params
from step_controller.generation.policy import PolicyFormat
from step_controller.generation.vllm import VLLMPolicy
from step_controller.loop import Runtime, _expander, _score_root, _value_head
from step_controller.preparation.direct_branch import direct_branch_values
from step_controller.reward import AsyncRewardModel, RewardResult
from step_controller.reward.config import RewardConfig
from step_controller.scheduler.core.policies import Budget
from step_controller.scheduler.core.scheduler import Scheduler
from step_controller.scheduler.core.tree import SchedulerState, SchedulerView


NEUTRAL_VALUE_VERSION = "critic-blind-neutral-0.5-v1"
REQUEST_SEED = ContextVar("locbench_critic_mcts_seed", default=0)


def read(path: Path) -> object:
    return json.loads(path.read_text())


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(gzip.compress(pickle.dumps(value), compresslevel=1, mtime=0))
    temporary.replace(path)


def load(path: Path) -> object:
    return pickle.loads(gzip.decompress(path.read_bytes()))


class MeasuredVLLMPolicy(VLLMPolicy):
    def __init__(self, *, seed_namespace, draw, spend, semaphore, **kwargs):
        super().__init__(**kwargs)
        self.seed_namespace = seed_namespace
        self.draw = draw
        self.spend = spend
        self.semaphore = semaphore
    def _completion_kwargs(self, model, prefix_tokens, params):
        kwargs = super()._completion_kwargs(model, prefix_tokens, params)
        kwargs["seed"] = REQUEST_SEED.get()
        return kwargs

    async def agenerate_tokens(self, prefix_tokens, sampling_params=None):
        params = merge_sampling_params(self.default_params, sampling_params)
        remaining = CONTEXT_LIMIT - len(prefix_tokens)
        if remaining <= 0:
            raise ValueError("Input exceeds context; no conditioning truncation")
        key = f"{self.seed_namespace}/{self.draw}"
        seed = (
            int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 2**31
        )
        self.draw += 1
        async with self.semaphore:
            token = REQUEST_SEED.set(seed)
            try:
                result = await super().agenerate_tokens(
                    prefix_tokens,
                    replace(params, max_tokens=min(params.max_tokens, remaining)),
                )
            finally:
                REQUEST_SEED.reset(token)
        if (
            not result.logprobs
            or len(result.logprobs) != len(result.tokens)
            or any(not math.isfinite(value) or value > 1e-5 for value in result.logprobs)
        ):
            raise ValueError("Missing or invalid exact behavior logprobs")
        self.spend.update(
            input_tokens=len(prefix_tokens),
            output_tokens=len(result.tokens),
            generations=1,
        )
        return result


class NeutralValue(AsyncRewardModel):
    """A checkpoint-independent prior keeps evaluated critics out of their targets."""

    async def ascore(self, context: str) -> RewardResult:
        del context
        return RewardResult(score=0.5)

    async def ascore_batch(self, contexts):
        return [RewardResult(score=0.5) for _ in contexts]


def source_snapshot(spec: dict[str, object]):
    native = load(Path(str(spec["native_file"])))
    if not isinstance(native, dict) or "checkpoints" not in native:
        raise ValueError(f"Malformed native continuation: {spec['native_file']}")
    snapshot = native["checkpoints"][int(spec["checkpoint_index"])]
    if (
        snapshot.done
        or not snapshot.forkable
        or snapshot.turns_taken != int(spec["turn"])
        or snapshot.folds != int(spec["folds"])
    ):
        raise ValueError(f"Native continuation changed for probe {spec['probe_position']}")
    return snapshot


async def run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    verify_harness()
    manifest = read(args.manifest)
    assert isinstance(manifest, dict)
    checkpoints = {
        row["iteration"]: row for row in manifest["checkpoints"]
    }
    checkpoint = checkpoints.get(args.iteration)
    if checkpoint is None:
        raise ValueError(f"Iteration {args.iteration} is absent from the fixed manifest")
    if Path(str(checkpoint["actor_model"])).resolve() != args.model.resolve():
        raise ValueError("Actor model differs from the frozen checkpoint manifest")
    if digest(args.model / "model.safetensors.index.json") != checkpoint["actor_model_index_sha256"]:
        raise ValueError("Actor model index changed after the fixed manifest was written")
    predictions = {
        int(row["probe_position"]): row for row in checkpoint["predictions"]
    }

    args.output.mkdir(parents=True, exist_ok=True)
    contract = {
        "protocol": "locbench-ppo-critic-oversampled-mcts-v1",
        "iteration": args.iteration,
        "model": str(args.model.resolve()),
        "model_index_sha256": checkpoint["actor_model_index_sha256"],
        "subset_manifest": str(args.manifest.resolve()),
        "subset_manifest_sha256": digest(args.manifest),
        "state_positions": [row["probe_position"] for row in manifest["states"]],
        "profile": "compact24-reply8-read60",
        "pass_tokens": args.pass_tokens,
        "max_pass_attempts": args.max_pass_attempts,
        "concurrency": args.concurrency,
        "allocation": ["windowed_token_entropy", "value_refinement"],
        "allocation_prior": 0.5,
        "allocation_prior_strength": 1.0,
        "target": (
            "direct-branch MCTS mean: each root child contributes its edge return plus "
            "the recursively refined neutral-prior value of that child"
        ),
        "critic_independence": (
            "The evaluated critic is never loaded by search; a fixed 0.5 prior is used "
            "for scoring, allocation, and unresolved frontier values."
        ),
        "seed_namespace": "locbench-ppo-critic-oversampled-mcts-v1",
        "environment": ppo_runtime.environment("compact24-reply8-read60"),
        "source_sha256": digest(__file__),
    }
    contract_path = args.output / "contract.json"
    if contract_path.exists() and read(contract_path) != contract:
        raise ValueError("Critic MCTS evaluation contract changed")
    write(contract_path, contract)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    policy_format = PolicyFormat.resolve(str(args.model), tokenizer=tokenizer, profile="qwen_xml")
    cases = {
        case.id: case
        for lane in ("train", "development", "test")
        for case in load_cases(EXPERIMENT / "data" / f"{lane}.jsonl")
    }
    states = list(manifest["states"])
    semaphore = asyncio.Semaphore(args.state_concurrency)
    generation_semaphore = asyncio.Semaphore(args.concurrency)

    async def one(spec: dict[str, object]):
        position = int(spec["probe_position"])
        if predictions[position]["context_sha256"] != spec["context_sha256"]:
            raise ValueError(f"Prediction/state mismatch at probe {position}")
        output = args.output / f"state-{position:02d}"
        complete = output / "complete.json"
        if complete.exists():
            return read(complete)
        async with semaphore:
            output.mkdir(parents=True, exist_ok=True)
            case = cases[str(spec["question"])]
            repository = await asyncio.to_thread(
                prepare, args.cache, case.repo, case.base_commit
            )
            spend = Counter(input_tokens=0, output_tokens=0, generations=0)
            progress_path = output / "progress.json"
            progress = read(progress_path) if progress_path.exists() else {"draw": 0, "passes": []}
            policy = MeasuredVLLMPolicy(
                seed_namespace=(
                    f"{contract['seed_namespace']}/{args.iteration}/{position}"
                ),
                draw=int(progress["draw"]),
                spend=spend,
                semaphore=generation_semaphore,
                model=str(args.model),
                format=policy_format,
                version=args.policy_version,
                base_url=args.url,
                api_key="EMPTY",
                served_model="locbench-qwen35-9b",
                timeout=1800,
                default_params=SamplingParams(
                    max_tokens=8192,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    logprobs=1,
                ),
            )
            runner, _, _, _ = make_world(
                case, repository, policy, profile="compact24-reply8-read60"
            )
            reward_config = RewardConfig(
                value_version=NEUTRAL_VALUE_VERSION,
                anchor_version="anchor",
                value_prior_strength=1.0,
                config_id="locbench-critic-blind-mcts-v1",
            )
            runtime = Runtime(
                runner=runner,
                value_model=NeutralValue(),
                value_serialize=lambda payload: "neutral",
                config=RolloutConfig(reward_config=reward_config),
            )
            allocators = (
                TokenEntropyAllocator(window=8, version=args.policy_version),
                ValueRefinement(),
            )
            completed_passes = list(progress["passes"])
            if completed_passes:
                last = int(completed_passes[-1]["pass_id"])
                state = load(output / f"pass-{last}.native.pkl.gz")
                start_pass = last + 1
            else:
                state = SchedulerState.root(
                    source_snapshot(spec), gating=ConcurrencyGating(args.concurrency)
                )
                scorer = _value_head(runtime)
                await _score_root(scorer, state)
                if state.stats.get("score_failures", 0):
                    raise RuntimeError("Neutral root scoring failed")
                start_pass = 0
            view = SchedulerView(state)
            ledger = RolloutLedger(reward_config)
            ledger.bind(view)
            scorer = _value_head(runtime)
            started = time.monotonic()
            for pass_id in range(start_pass, len(allocators)):
                allocator = allocators[pass_id]
                before_spend = dict(spend)
                before_stats = dict(state.stats)
                token_cap = spend["output_tokens"] + args.pass_tokens
                attempts = int(
                    state.stats.get("rollouts", 0) + state.stats.get("failures", 0)
                )
                async with state.lock:
                    state.regate(ConcurrencyGating(args.concurrency))
                allocation = allocator.build(view, ledger)
                if allocation is None:
                    raise RuntimeError(f"Allocator {allocator.name} produced no session")
                await Scheduler(
                    expander=_expander(runtime, f"pass-{pass_id}"),
                    allocation=allocation,
                    termination=Budget(
                        max_rollouts=attempts + args.max_pass_attempts,
                        goal=lambda _, cap=token_cap: spend["output_tokens"] >= cap,
                    ),
                    scorer=scorer,
                    max_concurrency=args.concurrency,
                ).run_on(state)
                if state.stats.get("failures", 0) > before_stats.get("failures", 0):
                    raise RuntimeError(f"Generation failure in pass {pass_id}: {state.failures}")
                if state.stats.get("score_failures", 0) > before_stats.get("score_failures", 0):
                    raise RuntimeError(f"Neutral scoring failure in pass {pass_id}")
                pass_record = {
                    "pass_id": pass_id,
                    "allocator": allocator.name,
                    "token_target": args.pass_tokens,
                    "token_target_reached": spend["output_tokens"] >= token_cap,
                    "cost": {
                        key: value - before_spend.get(key, 0)
                        for key, value in spend.items()
                    },
                    "stats": {
                        key: value - before_stats.get(key, 0)
                        for key, value in state.stats.items()
                    },
                    "seconds": time.monotonic() - started,
                }
                completed_passes.append(pass_record)
                save(output / f"pass-{pass_id}.native.pkl.gz", state)
                write(progress_path, {
                    "draw": policy.draw,
                    "passes": completed_passes,
                    "time": time.time(),
                })

            values = direct_branch_values(view, reward_config)
            root_value = values[state.root_id]
            root_counts = evidence_of(view, reward_config).observations.get(state.root_id)
            if root_value.mean is None or root_value.branches == 0:
                raise RuntimeError("Oversampled tree produced no root target")
            terminals = state.terminals()
            result = {
                "iteration": args.iteration,
                "probe_position": position,
                "question": spec["question"],
                "turn": spec["turn"],
                "folds": spec["folds"],
                "context_sha256": spec["context_sha256"],
                "critic_prediction": predictions[position]["native"],
                "mcts_target": root_value.mean,
                "root_refined_value": root_value.value,
                "root_branches": root_value.branches,
                "root_direct_rollout_mean": (
                    root_counts.returns.mean if root_counts and root_counts.returns.n else None
                ),
                "root_direct_rollouts": (
                    root_counts.returns.n if root_counts is not None else 0
                ),
                "terminal_nodes": len(terminals),
                "terminal_reward_mean": (
                    sum(float(node.payload.reward_outcome) for node in terminals)
                    / len(terminals)
                ),
                "cost": {
                    key: sum(record["cost"].get(key, 0) for record in completed_passes)
                    for key in ("input_tokens", "output_tokens", "generations")
                },
                "passes": completed_passes,
                "stats": dict(state.stats),
                "seconds": time.monotonic() - started,
                "tree_sha256": digest(output / "pass-1.native.pkl.gz"),
                "time": time.time(),
            }
            write(complete, result)
            print(json.dumps(result, allow_nan=False), flush=True)
            await policy.aclose()
            return result

    results = await asyncio.gather(
        *(one(spec) for spec in states), return_exceptions=True
    )
    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        write(
            args.output / "failed.json",
            {"errors": [repr(error) for error in errors], "time": time.time()},
        )
        raise RuntimeError(
            "Critic MCTS state failures: " + "; ".join(repr(error) for error in errors)
        )
    write(args.output / "results.json", results)
    write(
        args.output / "complete.json",
        {"iteration": args.iteration, "states": len(results), "time": time.time()},
    )
    (args.output / "failed.json").unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--policy-version", required=True)
    parser.add_argument("--cache", type=Path, default=Path(
        "/weka/scratch/jhu/bvandur1/zjiang31/rollout-controller/data/locbench/repositories"
    ))
    parser.add_argument("--pass-tokens", type=int, default=131072)
    parser.add_argument("--max-pass-attempts", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--state-concurrency", type=int, default=8)
    arguments = parser.parse_args()
    if (
        arguments.pass_tokens <= 0
        or arguments.max_pass_attempts < arguments.concurrency
        or arguments.concurrency < 1
        or arguments.state_concurrency < 1
    ):
        parser.error("Invalid search budget or concurrency")
    asyncio.run(run(arguments))
