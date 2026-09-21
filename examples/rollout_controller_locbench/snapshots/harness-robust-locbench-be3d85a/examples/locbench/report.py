"""Summarize completed paired pilots, re-reading their native traces consistently."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import statistics
from pathlib import Path
from typing import Any

from step_controller.generation import PolicyFormat

from .analysis import analyze_trace
from .bench import write_json
from .metrics import aggregate


def summarize(directory: Path, format: PolicyFormat[Any]) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if not (directory / "complete.json").exists():
        raise ValueError(f"pilot has not completed: {directory}")
    result: dict[str, Any] = {}
    by_arm: dict[str, dict[str, Any]] = {}
    for arm in manifest["settings"]["arms"]:
        records = [
            json.loads((directory / arm / f"case-{i:04d}.json").read_text())
            for i in range(len(manifest["cases"]))
        ]
        if [r["instance_id"] for r in records] != manifest["cases"]:
            raise ValueError("case records differ from the frozen cohort")
        diagnostics = {}
        for i, record in enumerate(records):
            path = directory / arm / "traces" / f"{i:04d}.pkl.gz"
            if path.exists():
                trace = pickle.loads(gzip.decompress(path.read_bytes()))
                if tuple(record["locations"]) != trace.state.locations:
                    raise ValueError("saved prediction differs from native submission")
                diagnostics[record["instance_id"]] = analyze_trace(trace, format)
        completed = [r for r in records if "turns" in r]
        journals = [
            json.loads(line)
            for path in (directory / arm).glob("generation-*.jsonl")
            for line in path.read_text().splitlines()
        ]
        stats = {
            **aggregate([r["metrics"] for r in records]),
            "full_file_successes": sum(r["metrics"]["file_acc@5"] for r in records),
            "any_file_hits": sum(r["metrics"]["file_recall@5"] > 0 for r in records),
            "errors": sum(bool(r.get("error")) for r in records),
            "length_statistics_n": len(completed),
            "native_traces_n": len(diagnostics),
            "forced": sum(bool(r.get("forced")) for r in records),
            "folded_trajectories": sum(r.get("folds", 0) > 0 for r in records),
            "generation_length_stops": sum(
                r.get("stop_reason") == "length" for r in journals
            ),
            "generation_errors": sum(bool(r.get("error")) for r in journals),
            # Include work spent on failed episodes; native-only length averages
            # would otherwise hide their generation cost.
            "total_generation_calls": len(journals),
            "total_generated_tokens": sum(
                r.get("completion_tokens", 0) for r in journals
            ),
            "mean_generated_tokens_per_case": (
                sum(r.get("completion_tokens", 0) for r in journals) / len(records)
            ),
        }
        for metric in (
            "turns",
            "folds",
            "prompt_tokens",
            "max_prompt_tokens",
            "completion_tokens",
            "fold_completion_tokens",
            "elapsed_seconds",
        ):
            values = [r[metric] for r in completed]
            stats[f"mean_{metric}"] = statistics.mean(values) if values else None
            stats[f"median_{metric}"] = statistics.median(values) if values else None
        for metric in (
            "tool_calls",
            "exact_repeat_calls",
            "repeats_from_before_fold",
            "tool_errors",
            "read_lines",
            "previously_read_lines",
            "invalid_submissions",
            "no_tool_turns",
            "fold_replies",
            "tool_shaped_fold_replies",
        ):
            stats[f"total_{metric}"] = sum(d[metric] for d in diagnostics.values())
        result[arm] = {"summary": stats, "records": records, "diagnostics": diagnostics}
        by_arm[arm] = {r["instance_id"]: r for r in records}
    if set(by_arm) == {"plain", "compact"}:
        paired = []
        for id_ in manifest["cases"]:
            a, b = by_arm["plain"][id_], by_arm["compact"][id_]
            paired.append(
                {
                    "instance_id": id_,
                    "file_acc@5_delta": b["metrics"]["file_acc@5"]
                    - a["metrics"]["file_acc@5"],
                    "file_recall@5_delta": b["metrics"]["file_recall@5"]
                    - a["metrics"]["file_recall@5"],
                    "turns_delta": b["turns"] - a["turns"]
                    if "turns" in a and "turns" in b
                    else None,
                }
            )
        result["paired_compact_minus_plain"] = paired
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    first = json.loads((args.runs[0] / "manifest.json").read_text())
    format = PolicyFormat.resolve(first["servers"]["model"], profile="qwen_xml")
    results = {str(path): summarize(path, format) for path in args.runs}
    write_json(
        args.output,
        {
            "analysis_sources": {
                name: hashlib.sha256(
                    Path(__file__).with_name(name).read_bytes()
                ).hexdigest()
                for name in ("analysis.py", "calls.py", "report.py")
            },
            "analysis_sha256": hashlib.sha256(
                Path(__file__).with_name("analysis.py").read_bytes()
            ).hexdigest(),
            "runs": results,
        },
    )


if __name__ == "__main__":
    main()
