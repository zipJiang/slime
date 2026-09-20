"""Resume an audited LocBench GRPO checkpoint on pinned reservations.

This is the recovery counterpart to :mod:`supervise_grpo_sep20`.  It keeps the
scientific recipe and question cursor, but permits a different physical layout
and token-packing bounds at a checkpoint boundary.
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
    job, separator, raw = text.partition("=")
    devices = [int(value) for value in raw.split(",") if value]
    if not separator or not job or not devices:
        raise argparse.ArgumentTypeError(f"Expected JOB=DEVICES, got {text!r}")
    return job, devices


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def step(job, name, cpus, command, devices=()):
    gres = "gpu:" + str(max(devices) + 1) if devices else "none"
    return wrap([
        "srun", "--jobid=" + job, "--overlap", "--nodes=1", "--ntasks=1",
        "--cpus-per-task=" + str(cpus), "--cpu-bind=none", "--mem=0",
        "--gres=" + gres, "--job-name=" + name, "--chdir=" + str(E), *command,
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--load", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--search-concurrency", type=int, default=12)
    parser.add_argument("--num-rollout", type=int, default=97)
    parser.add_argument("--drain-seconds", type=int, default=7200)
    parser.add_argument("--assign", type=assignment, action="append", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--driver-arg", action="append", default=[])
    args = parser.parse_args()
    assignments = dict(args.assign)
    if len(assignments) != len(args.assign):
        raise ValueError("Duplicate reservation assignment")
    if args.head not in assignments:
        raise ValueError("Head reservation is not assigned")
    if not (args.load / "latest_checkpointed_iteration.txt").is_file():
        raise ValueError(f"No native checkpoint to resume: {args.load}")
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

    def start(name, command):
        with (out / f"{name}.log").open("x") as log:
            child = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        children[name] = child
        write(out / f"{name}.json", {"pid": child.pid, "command": command})
        return child

    def interrupted(signum, _frame):
        raise InterruptedError(signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    write(out / "supervisor.json", {
        "pid": os.getpid(), "time": time.time(), "assignments": assignments,
        "head": args.head, "group_size": args.group_size,
        "search_concurrency": args.search_concurrency, "run_name": args.run_name,
        "resume_from": str(args.load), "driver_args": args.driver_arg,
    })
    try:
        hosts, ends = {}, {}
        for job, devices in assignments.items():
            info = subprocess.check_output(
                ["scontrol", "show", "job", job, "-o"], env=env, text=True
            )
            ends[job] = datetime.datetime.fromisoformat(
                re.search(r"EndTime=(\S+)", info)[1]
            ).timestamp()
            if ends[job] - time.time() < 21600:
                raise ValueError(f"Lease too short: {job}")
            host_text = subprocess.check_output(
                step(job, "loc-grpo-host", 1, ["hostname", "-I"]), env=env,
                text=True,
            )
            hosts[job] = next(ip for ip in host_text.split() if ip.startswith("172."))

        # The source GRPO, final Improved-PPO evaluation, and this recovery may
        # share reservations during handoff.  Wait for every selected device to
        # become genuinely free before starting a new Ray runtime.
        deadline = time.monotonic() + 14400
        while True:
            occupied = {}
            for job, devices in assignments.items():
                query = subprocess.check_output(step(job, "loc-grpo-free", 1, [
                    "nvidia-smi", "--query-gpu=index,memory.used",
                    "--format=csv,noheader,nounits",
                ], devices), env=env, text=True)
                memory = {
                    int(match[1]): int(match[2])
                    for row in query.splitlines()
                    if (match := re.fullmatch(r"\s*(\d+)\s*,\s*(\d+)\s*", row))
                }
                busy = {
                    device: memory.get(device)
                    for device in devices if memory.get(device, 999999) > 100
                }
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

        address = hosts[args.head] + ":6438"
        drain_at = min(ends.values()) - args.drain_seconds
        write(out / "layout.json", {
            "assignments": assignments, "hosts": hosts, "ends": ends,
            "drain_at": drain_at, "address": address,
        })
        for job, devices in assignments.items():
            role = "train" if job == args.head else "rollout"
            start("ray-" + job, step(job, "loc-grpo-" + role, 24, [
                "bash", str(E / "scripts/ray_node.sh"), role, address,
                ",".join(map(str, devices)),
            ], devices))
            if job == args.head:
                deadline = time.monotonic() + 360
                log = out / f"ray-{job}.log"
                while "Ray runtime started." not in log.read_text(errors="replace"):
                    if children["ray-" + job].poll() is not None:
                        raise RuntimeError("Ray head exited")
                    if time.monotonic() > deadline:
                        raise TimeoutError("Ray head startup")
                    time.sleep(5)

        deadline = time.monotonic() + 600
        total_gpus = sum(map(len, assignments.values()))
        while True:
            if any(child.poll() is not None for child in children.values()):
                raise RuntimeError("Ray service exited during startup")
            check = subprocess.run(
                step(args.head, "loc-grpo-ready", 1, [
                    "bash", str(E / "scripts/sif.sh"), "ray", "status",
                    "--address=" + address,
                ]), env=env, capture_output=True, text=True,
            )
            (out / "readiness.log").write_text(check.stdout + check.stderr)
            if check.returncode == 0 and f"/{total_gpus}.0 GPU" in check.stdout:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Ray cluster readiness")
            time.sleep(5)

        rollout_gpus = total_gpus - len(assignments[args.head])
        driver = start("driver", step(args.head, "loc-grpo-driver", 8, [
            "env", "RAY_ADDRESS=" + address, "GRPO_RUN_NAME=" + args.run_name,
            "GRPO_ROLLOUT_GPUS=" + str(rollout_gpus),
            "GRPO_NUM_ROLLOUT=" + str(args.num_rollout),
            "GRPO_GROUP_SIZE=" + str(args.group_size),
            "GRPO_SEARCH_CONCURRENCY=" + str(args.search_concurrency),
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "bash", str(E / "scripts/run_grpo_publication.sh"),
            # Last --load wins over the script's imitation checkpoint default.
            "--load", str(args.load.resolve()), *args.driver_arg,
        ], assignments[args.head]))
        audit = start("checkpoint-audit", step(args.head, "loc-grpo-audit", 4, [
            "/usr/bin/python3", str(E / "operations/watch_actor_checkpoints_sep18.py"),
            str(run),
        ]))
        while driver.poll() is None:
            if any(child.poll() is not None for name, child in children.items()
                   if name.startswith("ray-")):
                raise RuntimeError("Ray service exited")
            if audit.poll() not in (None, 0):
                (run / "STOP").touch()
                raise RuntimeError("Checkpoint audit failed")
            if time.time() > drain_at and run.exists():
                (run / "STOP").touch()
            status = {
                "stage": "training", "time": time.time(), "run": str(run),
                "drain_at": drain_at,
            }
            if (run / "status.json").exists():
                status["training"] = json.loads((run / "status.json").read_text())
            write(out / "status.json", status)
            time.sleep(30)
        if driver.returncode:
            raise RuntimeError(f"Training driver failed with {driver.returncode}")
        if audit.wait(timeout=1800):
            raise RuntimeError("Checkpoint audit failed")
        result = run / ("completed.json" if (run / "completed.json").exists()
                        else "paused.json")
        write(out / "complete.json", {"time": time.time(), "result": str(result)})
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
            "children": {name: child.returncode for name, child in children.items()},
        })


if __name__ == "__main__":
    main()
