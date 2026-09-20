"""Run the corrected group-4 Deontic VinePPO continuation.

The caller supplies physical reservations.  A first invocation can run exactly
one synchronous update from the last accepted group-5/k=1 checkpoint; a second
invocation resumes its audited checkpoint with one-batch overlap.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from slurm_retry import wrap


E = Path(__file__).resolve().parents[1]


def assignment(text):
    job, separator, values = text.partition("=")
    width, separator2, raw_devices = values.partition(":")
    devices = [int(value) for value in raw_devices.split(",") if value]
    if not separator or not separator2 or not job or not devices:
        raise argparse.ArgumentTypeError(
            f"Expected JOB=ALLOCATION_WIDTH:DEVICES, got {text!r}"
        )
    allocation_width = int(width)
    if min(devices) < 0 or max(devices) >= allocation_width:
        raise argparse.ArgumentTypeError(f"Invalid device set: {text!r}")
    return job, allocation_width, devices


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume-run", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--value-rollouts-per-state", type=int, default=1)
    parser.add_argument("--resume-group-size-from", type=int)
    parser.add_argument("--resume-value-rollouts-from", type=int)
    parser.add_argument("--search-concurrency", type=int, default=12)
    parser.add_argument("--execution", choices=("sync", "overlap"), required=True)
    parser.add_argument("--num-rollout", type=int, default=120)
    parser.add_argument("--drain-seconds", type=int, default=7200)
    parser.add_argument("--minimum-lease-seconds", type=int, default=21600)
    parser.add_argument("--assign", type=assignment, action="append", required=True)
    parser.add_argument("--head", required=True)
    args = parser.parse_args()
    assignments = {job: devices for job, _, devices in args.assign}
    widths = {job: width for job, width, _ in args.assign}
    if len(assignments) != len(args.assign):
        raise ValueError("Duplicate reservation assignment")
    if args.head not in assignments or len(assignments[args.head]) != 4:
        raise ValueError("The selected head must provide four trainer GPUs")
    if min(args.group_size, args.value_rollouts_per_state,
           args.search_concurrency, args.num_rollout) < 1:
        raise ValueError("Training counts must be positive")
    if not (args.resume_run / "actor/latest_checkpointed_iteration.txt").is_file():
        raise ValueError("Resume run has no committed actor checkpoint")
    run = E / "runs" / args.run_name
    if run.exists():
        raise ValueError(f"Refusing existing run: {run}")
    out = E / "operations" / args.name
    out.mkdir(exist_ok=False)
    children = {}
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("SLURM_", "SRUN_", "SBATCH_"))
    }

    def step(job, label, cpus, command, gpu=False):
        return wrap([
            "srun", "--jobid=" + job, "--overlap", "--nodes=1", "--ntasks=1",
            "--cpus-per-task=" + str(cpus), "--cpu-bind=none", "--mem=0",
            "--gres=" + ("gpu:" + str(widths[job]) if gpu else "none"),
            "--job-name=" + label, "--chdir=" + str(E), *command,
        ])

    def start(label, command):
        stream = (out / f"{label}.log").open("x")
        child = subprocess.Popen(
            command, env=env, stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        stream.close()
        children[label] = child
        write(out / f"{label}.json", {"pid": child.pid, "command": command})
        return child

    def interrupted(signum, _frame):
        raise InterruptedError(signum)

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    write(out / "supervisor.json", {
        "pid": os.getpid(), "time": time.time(), "assignments": assignments,
        "allocation_widths": widths, "head": args.head,
        "resume_run": str(args.resume_run), "run_name": args.run_name,
        "execution": args.execution, "group_size": args.group_size,
        "value_rollouts_per_state": args.value_rollouts_per_state,
        "search_concurrency": args.search_concurrency,
    })
    try:
        hosts, ends = {}, {}
        for job, devices in assignments.items():
            info = subprocess.check_output(
                ["scontrol", "show", "job", job, "-o"], env=env, text=True
            )
            if "JobState=RUNNING" not in info:
                raise ValueError("Reservation is not running: " + job)
            ends[job] = datetime.datetime.fromisoformat(
                re.search(r"EndTime=(\S+)", info)[1]
            ).timestamp()
            if ends[job] - time.time() < args.minimum_lease_seconds:
                raise ValueError("Lease too short: " + job)
            host_text = subprocess.check_output(
                step(job, "deon-vine-host", 1, ["hostname", "-I"]),
                env=env, text=True,
            )
            hosts[job] = next(ip for ip in host_text.split() if ip.startswith("172."))

        deadline = time.monotonic() + 14400
        while True:
            occupied = {}
            for job, devices in assignments.items():
                query = subprocess.check_output(step(job, "deon-vine-free", 1, [
                    "nvidia-smi", "--query-gpu=index,memory.used",
                    "--format=csv,noheader,nounits",
                ], gpu=True), env=env, text=True)
                memory = {
                    int(match[1]): int(match[2])
                    for row in query.splitlines()
                    if (match := re.fullmatch(r"\s*(\d+)\s*,\s*(\d+)\s*", row))
                }
                if not set(devices) <= memory:
                    raise ValueError(f"Missing selected GPU {job}: {memory}")
                busy = {device: memory[device] for device in devices
                        if memory[device] > 100}
                if busy:
                    occupied[job] = busy
                write(out / f"free-{job}.json", {
                    "time": time.time(), "selected": devices, "memory": memory,
                })
            if not occupied:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"GPUs did not drain: {occupied}")
            write(out / "status.json", {
                "stage": "waiting_for_handoff", "time": time.time(),
                "occupied": occupied,
            })
            time.sleep(30)

        address = hosts[args.head] + ":6428"
        drain_at = min(ends.values()) - args.drain_seconds
        write(out / "layout.json", {
            "assignments": assignments, "allocation_widths": widths,
            "hosts": hosts, "ends": ends, "address": address,
            "drain_at": drain_at,
        })
        for job, devices in assignments.items():
            role = "train" if job == args.head else "rollout"
            start("ray-" + job, step(job, "deon-vine-" + role, 24, [
                "bash", str(E / "scripts/ray_node.sh"), role, address,
                ",".join(map(str, devices)),
            ], gpu=True))
            if job == args.head:
                deadline = time.monotonic() + 360
                head_log = out / f"ray-{job}.log"
                while "Ray runtime started." not in head_log.read_text(errors="replace"):
                    if children["ray-" + job].poll() is not None:
                        raise RuntimeError("Ray head exited")
                    if time.monotonic() > deadline:
                        raise TimeoutError("Ray head startup")
                    time.sleep(5)

        total_gpus = sum(map(len, assignments.values()))
        deadline = time.monotonic() + 600
        while True:
            if any(child.poll() is not None for child in children.values()):
                raise RuntimeError("Ray service exited during startup")
            check = subprocess.run(step(args.head, "deon-vine-ready", 1, [
                "bash", str(E / "scripts/sif.sh"), "ray", "status",
                "--address=" + address,
            ]), env=env, capture_output=True, text=True)
            (out / "readiness.log").write_text(check.stdout + check.stderr)
            if check.returncode == 0 and f"/{total_gpus}.0 GPU" in check.stdout:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Ray cluster readiness")
            time.sleep(5)

        transition = []
        if args.resume_group_size_from is not None:
            transition += ["--resume-vine-group-size-from",
                           str(args.resume_group_size_from)]
        if args.resume_value_rollouts_from is not None:
            transition += ["--resume-vine-value-rollouts-from",
                           str(args.resume_value_rollouts_from)]
        rollout_gpus = total_gpus - len(assignments[args.head])
        driver = start("driver", step(args.head, "deon-vine-driver", 8, [
            "env", "RAY_ADDRESS=" + address, "PPO_RUN_NAME=" + args.run_name,
            "PPO_ROLLOUT_GPUS=" + str(rollout_gpus),
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "bash", str(E / "scripts/run_ppo.sh"),
            "--load", str((args.resume_run / "actor").resolve()),
            "--vine-group-size", str(args.group_size),
            "--vine-value-rollouts-per-state", str(args.value_rollouts_per_state),
            "--ppo-search-concurrency", str(args.search_concurrency),
            "--ppo-execution", args.execution,
            "--rollout-engine-base-port", "24000",
            "--num-rollout", str(args.num_rollout), *transition,
        ], gpu=True))
        audit = start("checkpoint-audit", step(args.head, "deon-vine-audit", 4, [
            "/usr/bin/python3", str(E / "operations/watch_actor_checkpoints_sep20.py"),
            str(run),
        ]))
        while driver.poll() is None:
            if any(child.poll() is not None for label, child in children.items()
                   if label.startswith("ray-")):
                raise RuntimeError("Ray service exited")
            if audit.poll() not in (None, 0):
                (run / "STOP").touch()
                raise RuntimeError("Checkpoint audit failed")
            if time.time() >= drain_at or (out / "STOP").exists():
                if run.exists():
                    (run / "STOP").touch()
            status = {
                "stage": "training", "run": str(run), "time": time.time(),
                "drain_at": drain_at,
            }
            if (run / "status.json").exists():
                status["training"] = json.loads((run / "status.json").read_text())
            write(out / "status.json", status)
            time.sleep(30)
        if driver.returncode:
            raise RuntimeError(f"Vine driver failed with {driver.returncode}")
        if audit.wait(timeout=1800):
            raise RuntimeError("Checkpoint audit failed")
        result = run / ("completed.json" if (run / "completed.json").exists()
                        else "paused.json")
        if not result.exists():
            raise RuntimeError("Vine exited without a durable result")
        write(out / "complete.json", {
            "time": time.time(), "result": str(result),
            "training": json.loads(result.read_text()),
        })
    except BaseException as exc:
        write(out / "failed.json", {"time": time.time(), "error": repr(exc)})
        raise
    finally:
        for child in reversed(list(children.values())):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children.values():
            try:
                child.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        write(out / "stopped.json", {
            "time": time.time(),
            "children": {label: child.returncode for label, child in children.items()},
        })


if __name__ == "__main__":
    main()
