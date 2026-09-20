"""Audit and summarize the aligned Deontic learning-curve evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import time


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def metric(rows: list[dict]) -> dict:
    return {
        "correct": sum(float(row["outcome"]) for row in rows),
        "total": len(rows),
        "rate": sum(float(row["outcome"]) for row in rows) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", type=Path, required=True)
    args = parser.parse_args()
    operation = args.operation.resolve()
    manifest = read(operation / "manifest.json")
    questions = read(operation / "questions.json")
    expected_keys = [row if isinstance(row, str) else row["case_key"]
                     for row in questions]
    if len(expected_keys) != 338 or len(set(expected_keys)) != 338:
        raise ValueError("Frozen question list is not 338 unique case keys")
    digest = hashlib.sha256((operation / "questions.json").read_bytes()).hexdigest()
    if digest != manifest["question_sha256"]:
        raise ValueError("Question-list digest changed")

    results = []
    failures = []
    for item in manifest["checkpoints"]:
        label = item["checkpoint"]
        marker = operation / "results" / label / "complete.json"
        if not marker.is_file():
            failures.append({"checkpoint": label, "state": "pending"})
            continue
        attempt = Path(read(marker)["attempt_output"])
        summary = read(attempt / "summary.json")
        rows = summary.get("results", [])
        keys = [row["case_key"] for row in rows]
        if keys != expected_keys:
            raise ValueError(f"{label}: ordered case keys differ from frozen questions")
        if summary.get("policy_version") != label:
            # Reused results retain their original label, which is also the
            # manifest label. New evaluations must do the same.
            raise ValueError(f"{label}: policy-version mismatch")
        cells: dict[tuple[str, bool], list[dict]] = defaultdict(list)
        for row in rows:
            cells[(row["domain"], bool(row["hard"]))].append(row)
        result = {
            **item,
            "overall": metric(rows),
            "hard": metric([row for row in rows if row["hard"]]),
            "normal": metric([row for row in rows if not row["hard"]]),
            "cells": {
                f"{domain}/{'hard' if hard else 'normal'}": metric(values)
                for (domain, hard), values in sorted(cells.items())
            },
            "mean_task_turns": sum(row["task_turns"] for row in rows) / len(rows),
            "mean_folds": sum(row["folds"] for row in rows) / len(rows),
            "truncated": sum(bool(row["truncated"]) for row in rows),
            "input_tokens": sum(row["cost"]["input_tokens"] for row in rows),
            "output_tokens": sum(row["cost"]["output_tokens"] for row in rows),
        }
        results.append(result)

    results.sort(key=lambda row: (row["actor_update"], row["method"]))
    state = {
        "protocol": manifest["protocol"],
        "question_sha256": digest,
        "completed": len(results),
        "available": len(manifest["checkpoints"]),
        "expected": manifest["expected_distinct_evaluations"],
        "pending_training": manifest["pending"],
        "pending_evaluation": failures,
        "results": results,
        "time": time.time(),
    }
    atomic_text(operation / "report.json", json.dumps(state, indent=2) + "\n")

    columns = ["method", "actor_update", "native_iteration", "correct", "total",
               "accuracy", "hard_correct", "hard_total", "hard_accuracy",
               "mean_task_turns", "mean_folds", "truncated", "input_tokens",
               "output_tokens", "source_run", "checkpoint"]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row in results:
        writer.writerow({
            "method": row["method"], "actor_update": row["actor_update"],
            "native_iteration": row["native_iteration"],
            "correct": row["overall"]["correct"], "total": row["overall"]["total"],
            "accuracy": row["overall"]["rate"],
            "hard_correct": row["hard"]["correct"], "hard_total": row["hard"]["total"],
            "hard_accuracy": row["hard"]["rate"],
            "mean_task_turns": row["mean_task_turns"], "mean_folds": row["mean_folds"],
            "truncated": row["truncated"], "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"], "source_run": row["source_run"],
            "checkpoint": row["checkpoint"],
        })
    atomic_text(operation / "accuracy.csv", stream.getvalue())

    lines = ["# Aligned DeonticBench learning curves", "",
             "All rows use the same 338 ordered cases and deterministic evaluation contract.", "",
             "| Method | Actor update | Native iteration | Overall | Hard | Mean turns | Folds | Truncated |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in results:
        overall, hard = row["overall"], row["hard"]
        lines.append(
            f"| {row['method']} | {row['actor_update']} | {row['native_iteration']} "
            f"| {overall['correct']:g}/{overall['total']} ({100*overall['rate']:.2f}%) "
            f"| {hard['correct']:g}/{hard['total']} ({100*hard['rate']:.2f}%) "
            f"| {row['mean_task_turns']:.2f} | {row['mean_folds']:.2f} | {row['truncated']} |"
        )
    lines += ["", f"Completed: {len(results)}/{manifest['expected_distinct_evaluations']} distinct evaluations."]
    atomic_text(operation / "report.md", "\n".join(lines) + "\n")
    atomic_text(operation / "status.json", json.dumps({
        "stage": "complete" if len(results) == manifest["expected_distinct_evaluations"] else "evaluating",
        "completed": len(results), "available": len(manifest["checkpoints"]),
        "expected": manifest["expected_distinct_evaluations"], "time": time.time(),
    }, indent=2) + "\n")
    print(json.dumps({"completed": len(results), "available": len(manifest["checkpoints"]),
                      "expected": manifest["expected_distinct_evaluations"]}))


if __name__ == "__main__":
    main()
