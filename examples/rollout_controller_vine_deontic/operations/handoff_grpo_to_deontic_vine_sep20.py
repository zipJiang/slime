"""Launch the queued Deontic VinePPO stages after LocBench GRPO completes.

The handoff is intentionally strict: a paused or failed GRPO run does not
satisfy the dependency.  The watcher requires the full actor-update target and
a successful full readback of the final native checkpoint before it starts the
one-round synchronous group-size transition.  Only after that stage and its
checkpoint audit complete does it launch the asynchronous continuation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


E = Path(__file__).resolve().parents[1]
OPERATIONS = E / "operations"
GRPO_RUN = Path(
    "/weka/projects/bvandur1/zjiang31/locbench-grpo-9b/runs/"
    "loc-grpo-g8-c12-sep20-f-48k-8g"
)
GRPO_OPERATION = (
    E.parent / "rollout_controller_vine_locbench" / "operations"
    / "loc-grpo-g8-c12-sep20-f-48k-8g"
)
REQUIRED_GRPO_UPDATES = 97
SOURCE_VINE_RUN = Path(
    "/weka/projects/bvandur1/zjiang31/deontic-vine-9b/runs/"
    "vine-both-null-g5-c12-sep19-replacement-633715"
)
SYNC_NAME = "vine-both-null-g4-c12-sep20-sync-r27"
ASYNC_NAME = "vine-both-null-g4-c12-sep20-async-r28"
ASSIGNMENTS = (
    "562163=4:0,1,2,3",
    "486600=2:0,1",
    "633715=2:0,1",
)
HEAD = "562163"
STATE = OPERATIONS / "queued-after-loc-grpo-sep20-state.json"


def write_state(stage: str, **extra: object) -> None:
    value = {"stage": stage, "time": time.time(), **extra}
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(STATE)


def completed_updates(run: Path) -> int:
    values = []
    for marker in (run / "rollouts").glob("train-*/training-complete.json"):
        try:
            values.append(int(json.loads(marker.read_text())["completed_actor_updates"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return max(values, default=0)


def require_final_readback(run: Path, expected_optimizer_step: int) -> Path:
    latest_file = run / "actor/latest_checkpointed_iteration.txt"
    if not latest_file.is_file():
        raise RuntimeError("Completed GRPO run has no latest checkpoint marker")
    iteration = int(latest_file.read_text().strip())
    readback = run / f"actor/iter_{iteration:07d}-readback.json"
    if not readback.is_file():
        raise RuntimeError(f"Final checkpoint has no full readback: {readback}")
    value = json.loads(readback.read_text())
    if value.get("full_storage_read") is not True:
        raise RuntimeError(f"Final checkpoint was not fully read: {readback}")
    if value.get("finite_tensors") is not True:
        raise RuntimeError(f"Final checkpoint contains nonfinite tensors: {readback}")
    steps = value.get("optimizer_steps")
    if steps != [expected_optimizer_step]:
        raise RuntimeError(
            f"Final optimizer step mismatch: expected {expected_optimizer_step}, got {steps}"
        )
    return readback


def wait_for_grpo() -> Path:
    while True:
        if (GRPO_OPERATION / "failed.json").exists():
            raise RuntimeError("LocBench GRPO predecessor failed")
        if (GRPO_RUN / "completed.json").exists():
            updates = completed_updates(GRPO_RUN)
            if updates < REQUIRED_GRPO_UPDATES:
                raise RuntimeError(
                    f"GRPO completed below target: {updates} < {REQUIRED_GRPO_UPDATES}"
                )
            readback = require_final_readback(GRPO_RUN, updates)
            write_state("grpo_complete", updates=updates, readback=str(readback))
            return readback
        if (GRPO_RUN / "paused.json").exists() or (GRPO_OPERATION / "complete.json").exists():
            # A drained predecessor must be resumed to the full target.  Do not
            # consume its GPUs with the queued baseline.
            updates = completed_updates(GRPO_RUN)
            write_state("waiting_for_grpo_resume", updates=updates)
        else:
            write_state("waiting_for_grpo", updates=completed_updates(GRPO_RUN))
        time.sleep(30)


def supervisor_live(operation: Path) -> bool:
    marker = operation / "supervisor.json"
    if not marker.is_file():
        return False
    try:
        pid = int(json.loads(marker.read_text())["pid"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return Path(f"/proc/{pid}").exists()


def wait_existing(operation: Path) -> None:
    while supervisor_live(operation):
        write_state("waiting_for_existing_stage", operation=str(operation))
        time.sleep(30)


def run_stage(
    *, name: str, run_name: str, resume_run: Path, execution: str,
    num_rollout: int, transition: bool,
) -> Path:
    operation = OPERATIONS / name
    run = E / "runs" / run_name
    if (operation / "failed.json").exists():
        raise RuntimeError(f"Queued Vine stage previously failed: {operation}")
    if operation.exists() and not (operation / "complete.json").exists():
        wait_existing(operation)
    if not (operation / "complete.json").exists():
        if operation.exists() or run.exists():
            raise RuntimeError(f"Refusing ambiguous existing stage: {operation}, {run}")
        command = [
            "/usr/bin/python3", str(OPERATIONS / "supervise_vine_group4_sep20.py"),
            "--name", name,
            "--run-name", run_name,
            "--resume-run", str(resume_run),
            "--group-size", "4",
            "--value-rollouts-per-state", "1",
            "--search-concurrency", "12",
            "--execution", execution,
            "--num-rollout", str(num_rollout),
            "--head", HEAD,
        ]
        for assignment in ASSIGNMENTS:
            command.extend(("--assign", assignment))
        if transition:
            command.extend(("--resume-group-size-from", "5"))
        write_state("launching_vine_stage", name=name, command=command)
        log_path = OPERATIONS / f"{name}-handoff.log"
        with log_path.open("x") as stream:
            result = subprocess.run(
                command, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT,
                env={k: v for k, v in os.environ.items()
                     if not k.startswith(("SLURM_", "SRUN_", "SBATCH_"))},
            )
        if result.returncode:
            raise RuntimeError(f"Vine stage {name} exited {result.returncode}")
    complete = json.loads((operation / "complete.json").read_text())
    result_path = Path(complete["result"])
    if result_path.name != "completed.json":
        raise RuntimeError(f"Vine stage drained before its target: {result_path}")
    updates = completed_updates(run)
    require_final_readback(run, updates)
    write_state("vine_stage_complete", name=name, updates=updates)
    return run


def main() -> None:
    wait_for_grpo()
    sync_run = run_stage(
        name=SYNC_NAME, run_name=SYNC_NAME, resume_run=SOURCE_VINE_RUN,
        execution="sync", num_rollout=28, transition=True,
    )
    run_stage(
        name=ASYNC_NAME, run_name=ASYNC_NAME, resume_run=sync_run,
        execution="overlap", num_rollout=120, transition=False,
    )
    write_state("complete")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write_state("failed", error=repr(exc))
        raise
