"""Paired multi-endpoint LocBench pilot with per-generation diagnostics."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import pickle
import time
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import asdict
from pathlib import Path
from typing import Any

from step_controller import SamplingParams, VLLMPolicy
from step_controller.generation import GenerateResult, PolicyFormat, TokenId
from step_controller.generation.interfaces import merge_sampling_params

from .analysis import analyze_trace
from .bench import TOOL_SETTINGS, run_one, score_predictions, write_json
from .dataset import load_cases
from .env import RunConfig
from .metrics import score
from .repository import prepare

JOURNAL: ContextVar[Path | None] = ContextVar("locbench_journal", default=None)


class RecordedPolicy(VLLMPolicy):
    async def agenerate_tokens(
        self,
        prefix_tokens: Sequence[TokenId],
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        started = time.time()
        entry: dict[str, Any] = {
            "started": started,
            "prefix_tokens": len(prefix_tokens),
            "sampling": asdict(
                merge_sampling_params(self.default_params, sampling_params)
            ),
        }
        try:
            result = await super().agenerate_tokens(prefix_tokens, sampling_params)
            entry.update(
                text=result.text,
                completion_tokens=len(result.tokens),
                stop_reason=result.stop_reason,
                exact=result.exact_generation,
            )
            return result
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            entry["elapsed_seconds"] = time.time() - started
            path = JOURNAL.get()
            if path is not None:
                with path.open("a") as stream:
                    stream.write(json.dumps(entry, ensure_ascii=False) + "\n")


async def run(args: argparse.Namespace) -> None:
    cases = load_cases(args.data)
    servers = json.loads(args.servers.read_text())
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=False)
    format = PolicyFormat.resolve(servers["model"], profile="qwen_xml")
    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        logprobs=0,
    )
    policies = [
        RecordedPolicy(
            model=servers["model"],
            served_model="locbench-qwen35-9b",
            format=format,
            base_url=url,
            api_key="EMPTY",
            default_params=params,
            timeout=1200,
            version="locbench-base-9b-pilot",
        )
        for url in servers["urls"]
    ]
    for policy in policies:
        await asyncio.to_thread(policy.startup_check)
    write_json(
        output / "manifest.json",
        {
            "cases": [c.id for c in cases],
            "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "servers": servers,
            "sampling": asdict(params),
            "settings": {
                **TOOL_SETTINGS,
                **{
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
            },
            "assignment": (
                "question index modulo endpoint count; paired arms share endpoint"
            ),
            "budget_semantics": (
                f"{args.max_steps} runner advances, including fold-only advances; "
                "finalization outside budget"
            ),
            "sources": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path("examples/locbench").glob("*.py"))
            },
        },
    )
    gates = [asyncio.Semaphore(args.per_server) for _ in policies]
    for arm in args.arms:
        (output / arm / "traces").mkdir(parents=True)

    async def one(index: int, arm: str) -> dict[str, Any]:
        case = cases[index]
        endpoint = index % len(policies)
        directory = output / arm
        trace = directory / "traces" / f"{index:04d}.pkl.gz"
        async with gates[endpoint]:
            token = JOURNAL.set(directory / f"generation-{index:04d}.jsonl")
            started = time.time()
            try:
                repo = await asyncio.to_thread(
                    prepare, args.cache, case.repo, case.base_commit
                )
                cfg = RunConfig(
                    policy=policies[endpoint],
                    compact=arm == "compact",
                    max_steps=args.max_steps,
                    call_budget=args.call_budget,
                    max_prompt_tokens=args.max_prompt_tokens,
                    fold_reply_tokens=args.max_tokens,
                    focused_guidance=args.focused_guidance,
                    repeat_limit=args.repeat_limit,
                )
                record = await run_one(case, repo, cfg, trace_path=trace)
            except Exception as exc:
                record = {
                    "instance_id": case.id,
                    "locations": [],
                    "submitted": False,
                    "metrics": score([], case.gold),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                JOURNAL.reset(token)
            if trace.exists():
                # Diagnostics must never overwrite a completed trajectory's score.
                try:
                    result = pickle.loads(gzip.decompress(trace.read_bytes()))
                    analysis = analyze_trace(result, format)
                    write_json(directory / f"analysis-{index:04d}.json", analysis)
                    record["repetition"] = {
                        k: v for k, v in analysis.items() if not k.endswith("events")
                    }
                except Exception as exc:
                    record["analysis_error"] = f"{type(exc).__name__}: {exc}"
            record.update(
                arm=arm, endpoint=endpoint, started=started, finished=time.time()
            )
            write_json(directory / f"case-{index:04d}.json", record)
            print(json.dumps(record), flush=True)
            return record

    # Alternate admission order; every question uses the same GPU in both arms.
    jobs = [
        (i, arm)
        for i in range(len(cases))
        for arm in (args.arms if i % 2 == 0 else list(reversed(args.arms)))
    ]
    records = await asyncio.gather(*(one(i, arm) for i, arm in jobs))
    for arm in args.arms:
        selected = [r for r in records if r["arm"] == arm]
        by_id = {r["instance_id"]: r for r in selected}
        ordered = [by_id[c.id] for c in cases]
        (output / arm / "predictions.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in ordered)
        )
        write_json(output / arm / "summary.json", score_predictions(cases, ordered))
    write_json(
        output / "complete.json",
        {"finished": time.time(), "trajectories": len(records)},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--servers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache", type=Path, default=Path("data/locbench/repositories")
    )
    parser.add_argument(
        "--arms", choices=["plain", "compact"], nargs="+", default=["plain", "compact"]
    )
    parser.add_argument("--per-server", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--call-budget", type=int, default=20)
    parser.add_argument("--max-prompt-tokens", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--focused-guidance", action="store_true")
    parser.add_argument("--repeat-limit", type=int, default=0)
    args = parser.parse_args()
    if len(args.arms) != len(set(args.arms)) or args.per_server < 1:
        parser.error("arms must be unique and per-server concurrency positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
