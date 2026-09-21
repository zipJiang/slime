"""Launch robust both-null LocBench Improved PPO after GRPO update 97."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
LOC = ROOT / "examples/rollout_controller_locbench"
VINE = ROOT / "examples/rollout_controller_vine_locbench"
GRPO_OPERATION = VINE / "operations/loc-grpo-g8-c12-sep20-h-48k-8g"
GRPO_RUN = VINE / "runs/loc-grpo-g8-c12-sep20-h-48k-8g"
OUT = LOC / "operations/handoff-grpo-to-robust-ppo-sep21"
PPO_NAME = "loc-improved-ppo-robust-null-sep21-v1"
ASSIGNMENTS = {
    "562163": [0, 1, 2, 3],
    "423716": [0, 1, 2, 3],
    "486600": [0, 1],
    "633715": [0],
}
EXPECTED_UPDATES = 97
PYTHON = "/weka/scratch/jhu/bvandur1/zjiang31/rollout-controller/.venv/bin/python"


def write(name: str, value: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def process_live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def completed_updates() -> int:
    status = GRPO_RUN / "status.json"
    if not status.exists():
        return 0
    return int(read(status).get("completed_actor_updates", 0))


def wait_for_grpo() -> None:
    supervisor = read(GRPO_OPERATION / "supervisor.json")
    pid = int(supervisor["pid"])
    while True:
        updates = completed_updates()
        if (GRPO_OPERATION / "failed.json").exists():
            raise RuntimeError("GRPO supervisor failed before the full epoch")
        stopped = (GRPO_OPERATION / "stopped.json").exists()
        if stopped:
            if updates < EXPECTED_UPDATES or not (GRPO_RUN / "completed.json").exists():
                raise RuntimeError(
                    f"GRPO stopped at {updates}/{EXPECTED_UPDATES}; resume it before PPO"
                )
            return
        if not process_live(pid):
            raise RuntimeError("GRPO supervisor disappeared without terminal evidence")
        write(
            "status.json",
            {
                "stage": "waiting_for_grpo",
                "time": time.time(),
                "updates": updates,
                "expected_updates": EXPECTED_UPDATES,
                "grpo_supervisor_pid": pid,
            },
        )
        time.sleep(30)


def main() -> None:
    if (LOC / "operations" / PPO_NAME).exists() or (LOC / "runs" / PPO_NAME).exists():
        raise ValueError(f"Refusing existing PPO run {PPO_NAME}")
    assignments = OUT / "assignments.json"
    write("assignments.json", ASSIGNMENTS)
    write(
        "plan.json",
        {
            "created": time.time(),
            "grpo_operation": str(GRPO_OPERATION),
            "grpo_run": str(GRPO_RUN),
            "required_grpo_updates": EXPECTED_UPDATES,
            "ppo_name": PPO_NAME,
            "profile": "robust-null-v3",
            "actor_workspace": "null",
            "compactor_workspace": "null",
            "harness_commit": "be3d85ace36f10f3b4ae71804ec2c3929ed7ce68",
            "assignments": ASSIGNMENTS,
            "train_job": "562163",
            "critic_replica_job": "633715",
            "optional_rollout_jobs": ["423716", "486600"],
            "critic_transfer": (
                "Explicit transfer from imitation-critic-expanded-v1; the first "
                "on-policy rounds recalibrate the changed context distribution"
            ),
        },
    )
    wait_for_grpo()
    command = [
        PYTHON,
        str(VINE / "operations/slurm_retry.py"),
        "srun",
        "--jobid=562163",
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=2",
        "--cpu-bind=none",
        "--mem=0",
        "--gres=none",
        "--job-name=loc-robust-ppo-supervisor",
        "--chdir=" + str(LOC),
        PYTHON,
        "-u",
        str(LOC / "operations/ppo_supervisor.py"),
        "--name",
        PPO_NAME,
        "--candidate",
        str(LOC / "runs/imitation-critic-expanded-v1/warmstart-candidate.json"),
        "--critic-operations",
        str(LOC / "operations/imitation-critic-expanded-v1"),
        "--assignments",
        str(assignments),
        "--train-job",
        "562163",
        "--train-gpus",
        "4",
        "--replica-job",
        "633715",
        "--collection-profile",
        "robust-null-v3",
        "--allow-profile-transition",
        "--pass-tokens",
        "32768",
        "--train-allocator-conf",
        "expandable_segments:True",
    ]
    write("launch.json", {"time": time.time(), "command": command})
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SLURM_", "SRUN_", "SBATCH_"))
    }
    with (OUT / "ppo-supervisor.log").open("x") as log:
        child = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write("launched.json", {"time": time.time(), "pid": child.pid, "command": command})
    result = child.wait()
    if result:
        raise RuntimeError(f"Robust PPO supervisor exited {result}")
    write("complete.json", {"time": time.time(), "ppo_name": PPO_NAME})


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write("failed.json", {"time": time.time(), "error": repr(exc)})
        raise
