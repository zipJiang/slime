"""Dataset preparation, fixed-page probe, evaluation, and training integration."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import pickle
import random
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from step_controller import (
    AdvantageEstimator,
    PreparedBatch,
    RewardModel,
    RolloutConfig,
    Runtime,
    SamplingParams,
    VLLMPolicy,
    prepare_samples,
    run_search,
)

from .dataset import Case, download, load_cases
from .env import RunConfig, build_world
from .metrics import aggregate, score, submission
from .repository import (
    LIST_DEPTH,
    LIST_LIMIT,
    MAX_HITS,
    MAX_LINE_CHARS,
    READ_LINES,
    Repository,
    prepare,
)

TOOL_SETTINGS = {
    "tools": ["list", "grep", "read", "submit"],
    "list_limit": LIST_LIMIT,
    "subtree_depth": LIST_DEPTH,
    "read_page_lines": READ_LINES,
    "grep_max_hits": MAX_HITS,
    "max_line_chars": MAX_LINE_CHARS,
    "regex_dialect": "POSIX extended",
    "reward": "file_recall@5",
}


def write_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


async def search_one(
    case: Case,
    repo: Repository,
    cfg: RunConfig,
    search: RolloutConfig,
    estimator: AdvantageEstimator,
    *,
    value_model: RewardModel | None = None,
) -> PreparedBatch:
    """Use the library's search/preparation path, preserving task and fold tokens."""
    world = build_world(case, repo, cfg)
    runtime = Runtime(runner=world.runner, config=search, value_model=value_model)
    tree = await run_search(world.prompt, runtime, workspace=world.workspace)
    return prepare_samples(
        tree,
        estimator=estimator,
        reward_config=search.reward_config,
        behavior_version=runtime.runner.policy.version,
    )


async def run_one(
    case: Case, repo: Repository, cfg: RunConfig, *, trace_path: Path | None = None
) -> dict[str, Any]:
    """A trajectory and native checkpoint, scored from its recorded submission."""
    start = time.monotonic()
    world = build_world(case, repo, cfg)
    result = await world.runner.run_from(world.prompt, workspace=world.workspace)
    metrics = score(result.state.locations, case.gold)
    if result.reward != metrics["reward"]:
        raise ValueError("recorded submission and trajectory reward disagree")
    if trace_path is not None:
        # This is a local trusted checkpoint, not a pickle accepted from a model.
        data = gzip.compress(pickle.dumps(result.snapshot()), compresslevel=1, mtime=0)
        temp = trace_path.with_suffix(trace_path.suffix + ".tmp")
        temp.write_bytes(data)
        temp.replace(trace_path)
    return {
        "instance_id": case.id,
        "repo": case.repo,
        "base_commit": case.base_commit,
        "locations": list(result.state.locations),
        "submitted": result.state.submitted,
        "metrics": metrics,
        "turns": result.turns_taken,
        "folds": result.folds,
        "tool_calls": {
            name: result.state.calls.count(name) for name in ("list", "grep", "read")
        },
        "prompt_tokens": sum(len(turn.prefix) for turn in result.turns),
        "max_prompt_tokens": max(
            (len(turn.prefix) for turn in result.turns), default=0
        ),
        "completion_tokens": sum(len(turn.tokens) for turn in result.turns),
        "fold_completion_tokens": sum(
            len(turn.tokens) for turn in result.turns if turn.tag == "fold"
        ),
        "forced": result.truncated,
        "exact_tokens": all(turn.exact for turn in result.turns),
        "elapsed_seconds": time.monotonic() - start,
        "trace": str(trace_path) if trace_path is not None else None,
        "error": None,
    }


def probe(repo: Repository, *, sample_files: int = 64) -> dict[str, object]:
    """Deterministic gold-independent source-length probe; does not alter page size."""
    paths = sorted(
        p for p, entry in repo.entries.items() if entry.readable and p.endswith(".py")
    )
    chosen = random.Random(0).sample(paths, min(sample_files, len(paths)))
    lengths: list[int] = []
    errors: list[dict[str, str]] = []
    for path in chosen:
        try:
            text = repo.text(path)
            lengths.append(
                text.count("\n") + int(bool(text) and not text.endswith("\n"))
            )
        except ValueError as exc:
            errors.append({"path": path, "error": str(exc)})
    ordered = sorted(lengths)
    return {
        "base_commit": repo.commit,
        "repository_files": len(repo.entries),
        "python_files": len(paths),
        "probed_files": len(lengths),
        "sample_seed": 0,
        "line_quantiles": {
            str(q): ordered[round((len(ordered) - 1) * q)] if ordered else None
            for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "files_larger_than_page": sum(n > READ_LINES for n in lengths),
        "read_page_lines": READ_LINES,
        "errors": errors,
    }


def selected_cases(
    path: Path, ids: Sequence[str] | None, limit: int | None
) -> list[Case]:
    cases = load_cases(path)
    if ids:
        missing = set(ids) - {case.id for case in cases}
        if missing:
            raise ValueError(f"unknown instance IDs: {sorted(missing)}")
        cases = [case for case in cases if case.id in ids]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        cases = cases[:limit]
    if not cases:
        raise ValueError("no cases selected")
    return cases


def score_predictions(
    cases: Sequence[Case], rows: Sequence[dict[str, Any]], *, maximum: int = 10
) -> dict[str, Any]:
    """Missing or malformed submissions score zero, without dropping their cases."""
    by_id: dict[str, dict[str, Any]] = {}
    allowed = {case.id for case in cases}
    for row in rows:
        id_ = row["instance_id"]
        if id_ in by_id or id_ not in allowed:
            raise ValueError(f"duplicate or unselected prediction ID: {id_}")
        by_id[id_] = row
    records = []
    invalid = 0
    for case in cases:
        prediction = by_id.get(case.id)
        try:
            locations = submission(
                prediction.get("locations", []) if prediction else [], maximum=maximum
            )
        except ValueError:
            locations = ()
            invalid += 1
        records.append(score(locations, case.gold))
    return {
        **aggregate(records),
        "missing_submissions": len(cases) - len(by_id),
        "invalid_submissions": invalid,
        "errored_cases": sum(bool(row.get("error")) for row in rows),
    }


async def run_batch(args: argparse.Namespace, cases: Sequence[Case]) -> None:
    output: Path = args.output
    # A new directory makes each evaluation cohort/configuration unambiguous.
    output.mkdir(parents=True, exist_ok=False)
    trace_dir = output / "traces"
    trace_dir.mkdir()
    params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    policy = VLLMPolicy(
        model=args.model,
        base_url=args.vllm_url,
        served_model=args.served_model,
        profile=args.profile,
        default_params=params,
        api_key="EMPTY",
        timeout=args.request_timeout,
    )
    policy.startup_check()
    cfg = RunConfig(
        policy=policy,
        compact=args.compact,
        max_steps=args.max_steps,
        call_budget=args.call_budget,
        max_prompt_tokens=args.max_prompt_tokens,
        fold_reply_tokens=args.fold_reply_tokens,
        max_locations=args.max_locations,
        focused_guidance=args.focused_guidance,
        repeat_limit=args.repeat_limit,
    )
    settings = dict(TOOL_SETTINGS)
    settings.update(
        {
            str(k): v
            for k, v in vars(args).items()
            if k not in {"data", "cache", "output"}
        }
    )
    write_json(
        output / "manifest.json",
        {
            "dataset_path": str(args.data.resolve()),
            "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "cases": [
                {"instance_id": c.id, "repo": c.repo, "base_commit": c.base_commit}
                for c in cases
            ],
            "settings": settings,
            "policy_version": policy.version,
            "sampling": asdict(params),
        },
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(index: int, case: Case) -> dict[str, Any]:
        async with semaphore:
            try:
                repository = await asyncio.to_thread(
                    prepare, args.cache, case.repo, case.base_commit
                )
                record = await run_one(
                    case, repository, cfg, trace_path=trace_dir / f"{index:04d}.pkl.gz"
                )
            except Exception as exc:
                record = {
                    "instance_id": case.id,
                    "locations": [],
                    "submitted": False,
                    "metrics": score([], case.gold),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            write_json(output / f"case-{index:04d}.json", record)
            print(
                json.dumps(
                    {
                        "instance_id": case.id,
                        "reward": record["metrics"]["reward"],
                        "error": record["error"],
                    }
                ),
                flush=True,
            )
            return record

    records = await asyncio.gather(*(one(i, case) for i, case in enumerate(cases)))
    # Stable dataset order regardless of concurrent completion order.
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    write_json(
        output / "summary.json",
        score_predictions(cases, records, maximum=args.max_locations),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser(
        "download", help="download and validate the pinned dataset"
    )
    fetch.add_argument("--directory", type=Path, default=Path("data/locbench/source"))
    for name in ("prepare", "probe", "run", "score"):
        command = commands.add_parser(name)
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--instances", nargs="+")
        command.add_argument("--limit", type=int)
        command.add_argument(
            "--cache", type=Path, default=Path("data/locbench/repositories")
        )
        if name == "score":
            command.add_argument("--predictions", type=Path, required=True)
            command.add_argument("--max-locations", type=int, default=10)
        if name == "run":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--model", required=True)
            command.add_argument("--vllm-url", default="http://localhost:8000/v1")
            command.add_argument("--served-model")
            command.add_argument("--profile")
            command.add_argument("--compact", action="store_true")
            command.add_argument("--focused-guidance", action="store_true")
            command.add_argument("--repeat-limit", type=int, default=0)
            command.add_argument("--max-steps", type=int, default=80)
            command.add_argument("--call-budget", type=int, default=20)
            command.add_argument("--max-prompt-tokens", type=int, default=32768)
            command.add_argument("--fold-reply-tokens", type=int, default=8192)
            command.add_argument("--max-tokens", type=int, default=8192)
            command.add_argument("--max-locations", type=int, default=10)
            command.add_argument("--temperature", type=float, default=0.0)
            command.add_argument("--top-p", type=float, default=1.0)
            command.add_argument("--top-k", type=int)
            command.add_argument("--request-timeout", type=float, default=300)
            command.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "download":
        print(download(args.directory))
        return
    cases = selected_cases(args.data, args.instances, args.limit)
    if args.command == "run":
        if args.concurrency < 1:
            parser.error("concurrency must be positive")
        asyncio.run(run_batch(args, cases))
    elif args.command == "score":
        rows = [
            json.loads(line)
            for line in args.predictions.read_text().splitlines()
            if line.strip()
        ]
        print(
            json.dumps(
                score_predictions(cases, rows, maximum=args.max_locations), indent=2
            )
        )
    else:
        for case in cases:
            repo = prepare(args.cache, case.repo, case.base_commit)
            result = (
                probe(repo) if args.command == "probe" else {"files": len(repo.entries)}
            )
            print(json.dumps({"instance_id": case.id, **result}), flush=True)


if __name__ == "__main__":
    main()
