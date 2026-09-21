"""Frozen robust LocBench runtime with actor and compactor NullWorkspace."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
HARNESS = EXPERIMENT / "snapshots/harness-robust-locbench-be3d85a"
MODEL = "/weka/projects/bvandur1/zjiang31/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
CONTEXT_LIMIT = 98304
TASK_LIMIT = 80
REPLY_LIMIT = 16384


def activate() -> None:
    """Select this harness even after another frozen harness was imported.

    Moving this snapshot to the front cannot change an existing package's
    ``__path__``.  Evict only stale harness-owned packages so every subsequent
    import resolves from one coherent immutable tree.
    """
    value = str(HARNESS)
    sys.path[:] = [entry for entry in sys.path if entry != value]
    sys.path.insert(0, value)
    harness = HARNESS.resolve()

    def relevant(name: str, family: str) -> bool:
        if family == "step_controller":
            return name == family or name.startswith(family + ".")
        return (
            name == "examples"
            or name == "examples.locbench"
            or name.startswith("examples.locbench.")
            or name == "examples.toolkit"
            or name.startswith("examples.toolkit.")
        )

    def belongs(module) -> bool:
        source = getattr(module, "__file__", None)
        if source is not None:
            return Path(source).resolve().is_relative_to(harness)
        paths = getattr(module, "__path__", ())
        return any(Path(path).resolve().is_relative_to(harness) for path in paths)

    # A package tree is one import-identity unit.  Keeping robust children while
    # replacing a stale parent (or vice versa) can leave two class objects with
    # the same qualified name, which makes completed scheduler states
    # unpicklable.  If any member is stale, discard the whole relevant family
    # before callers import classes from the selected immutable snapshot.
    for family in ("examples", "step_controller"):
        loaded = [
            (name, module)
            for name, module in tuple(sys.modules.items())
            if relevant(name, family)
        ]
        if any(not belongs(module) for _, module in loaded):
            for name, _ in sorted(loaded, key=lambda item: item[0].count("."), reverse=True):
                sys.modules.pop(name, None)
    importlib.invalidate_caches()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract():
    return dict(
        profile="locbench-robust-null-v3",
        compaction_failure="terminal_zero",
        model=MODEL,
        tools=["list", "grep", "read", "submit"],
        reward="file_recall@5",
        task_limit=TASK_LIMIT,
        horizon="global task turns plus folds; one final submission outside budget",
        server_context_limit=CONTEXT_LIMIT,
        actor_reply_limit=REPLY_LIMIT,
        fold_reply_limit=REPLY_LIMIT,
        prompt_limit=32768,
        call_budget=20,
        call_counter="independent rollover; never a compaction trigger",
        summary_target_tokens=2048,
        actor_temperature=0.6,
        actor_top_p=0.95,
        fold_temperature=0.2,
        fold_top_p=0.95,
        top_k=20,
        repeat_limit=0,
        actor_workspace="null",
        compactor_workspace="null",
        compaction_memory_role="assistant",
        compaction_schema="structured-state-evidence-open-questions-next-steps-v1",
        context_schema="expected-file-recall-v2",
        harness_commit=json.loads((HARNESS / "source-manifest.json").read_text())[
            "source_commit"
        ],
        harness_sha256=digest(HARNESS / "source-manifest.json"),
    )


def verify_harness():
    activate()
    manifest = json.loads((HARNESS / "source-manifest.json").read_text())
    if manifest["source_commit"] != "be3d85ace36f10f3b4ae71804ec2c3929ed7ce68":
        raise ValueError("Unexpected robust harness commit")
    for relative, expected in manifest["files"].items():
        if digest(HARNESS / relative) != expected:
            raise ValueError(f"Frozen robust harness differs: {relative}")


def context(payload, tools, tokenizer):
    """Serialize only observable state; future reward and gold remain private."""
    state = dict(
        messages=list(payload.messages),
        tools=tools,
        workspace=dict(payload.workspace.snapshot()),
        remaining_steps=max(0, TASK_LIMIT - payload.turns_taken - payload.folds),
        calls_remaining=payload.state.calls_remaining,
        calls_max=payload.state.calls_max,
        last_request=payload.state.last_request,
        consecutive_requests=payload.state.consecutive_requests,
        compaction_failure=list(payload.state.compaction_failure),
        done=payload.done,
        truncated=payload.truncated,
    )
    content = (
        "Predict the expected final file recall@5 from continuing this localization "
        "checkpoint, between 0 and 1.\n"
    )
    content += json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return tokenizer.apply_chat_template(
        [dict(role="user", content=content)], tokenize=False, add_generation_prompt=True
    )


def make_world(case, repo, policy):
    activate()
    from examples.locbench.env import RunConfig, build_world
    from step_controller.harness import NullWorkspace

    world = build_world(
        case,
        repo,
        RunConfig(
            policy=policy,
            compact=True,
            robust_compaction=True,
            max_steps=TASK_LIMIT,
            call_budget=20,
            max_prompt_tokens=32768,
            fold_reply_tokens=REPLY_LIMIT,
            focused_guidance=True,
            repeat_limit=0,
            compaction_failure="terminal_zero",
            count_folds_in_horizon=True,
        ),
    )
    if not isinstance(world.workspace, NullWorkspace):
        raise ValueError("Robust LocBench actor must use NullWorkspace")
    compactor = world.runner._compactor
    if not isinstance(compactor._fold_workspace, NullWorkspace):
        raise ValueError("Robust LocBench compactor must use NullWorkspace")
    return world.runner, world.workspace, world.prompt, world.runner.tools
