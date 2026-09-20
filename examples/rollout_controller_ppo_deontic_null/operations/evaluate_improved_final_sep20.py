#!/usr/bin/env python3
"""Wait for the final Improved-PPO checkpoint, evaluate it, and audit every trace."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request


EXPERIMENT = Path(__file__).resolve().parents[1]
TRAIN_RUN = Path(
    "/weka/projects/bvandur1/zjiang31/deontic-ppo-null-9b/runs/"
    "both-null-base-warmup10-c12-sep20-v24-async-regex-bounded"
)
ROOT = Path(
    "/weka/projects/bvandur1/zjiang31/deontic-ppo-null-9b/evaluations/"
    "improved-iter89-sep20-v1"
)
ITERATION = 89
ACTOR_UPDATES = 80
POLICY_VERSION = "actor-0080"
PORTS = (35800, 35801, 35802)
SIF = EXPERIMENT.parent / "rollout_controller_vine_deontic/scripts/sif.sh"
COLLECTOR = EXPERIMENT / "scripts/collect_rollouts.py"
AUDITOR = EXPERIMENT / "scripts/audit_evaluation.py"
PYTHON = Path(
    "/weka/scratch/jhu/bvandur1/zjiang31/rollout-controller/.venv/bin/python"
)


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=45)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def wait_http(url: str, processes: list[subprocess.Popen], seconds: int = 1200) -> None:
    deadline = time.monotonic() + seconds
    while True:
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(
                    f"process {process.pid} exited {process.returncode} before {url} was ready"
                )
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status < 500:
                    return
        except Exception:
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {url}")
        time.sleep(5)


def breakdown(rows: list[dict]) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        outcome = float(row["outcome"])
        groups["overall"].append(outcome)
        groups["hard" if row["hard"] else "normal"].append(outcome)
        groups[f"{row['domain']}/{'hard' if row['hard'] else 'normal'}"].append(outcome)
    return {
        label: {
            "correct": sum(values),
            "total": len(values),
            "rate": sum(values) / len(values),
        }
        for label, values in sorted(groups.items())
    }


def wait_checkpoint() -> None:
    while True:
        paused = TRAIN_RUN / "paused.json"
        watch = TRAIN_RUN / "checkpoint-watch.json"
        try:
            state = json.loads(watch.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            state = {}
        ready = paused.exists() and state.get(f"iter_{ITERATION:07d}", {}).get("passed")
        write(ROOT / "status.json", {
            "stage": "waiting-for-final-checkpoint",
            "paused": paused.exists(),
            "checkpoint_audited": bool(
                state.get(f"iter_{ITERATION:07d}", {}).get("passed")
            ),
            "time": time.time(),
        })
        if ready:
            return
        if (TRAIN_RUN / "failed.json").exists():
            raise RuntimeError("Improved PPO failed before its final checkpoint")
        time.sleep(30)


def prepare_model() -> tuple[Path, Path]:
    source = TRAIN_RUN / "hf" / f"iter_{ITERATION:07d}"
    index = source / "model.safetensors.index.json"
    if not index.is_file():
        raise FileNotFoundError(index)
    serving = ROOT / "serving-model"
    serving.mkdir()
    for path in source.iterdir():
        if path.name == index.name:
            continue
        (serving / path.name).symlink_to(path.resolve())
    # A local regular index avoids model loaders rewriting through a symlink.
    parsed = json.loads(index.read_text())
    (serving / index.name).write_text(json.dumps(parsed, indent=2, sort_keys=True) + "\n")
    questions = ROOT / "questions.json"
    prior_questions = Path(
        "/weka/projects/bvandur1/zjiang31/deontic-ppo-null-9b/evaluations/"
        "improved-iter81-sep20-v1/questions.json"
    )
    questions.write_bytes(prior_questions.read_bytes())
    write(ROOT / "manifest.json", {
        "protocol": "improved-ppo-heldout-actor80-fixed-protocol-v1",
        "created": time.time(),
        "native_iteration": ITERATION,
        "actor_optimizer_updates": ACTOR_UPDATES,
        "policy_version": POLICY_VERSION,
        "model": str(serving),
        "source_model": str(source),
        "source_model_index_sha256": sha256(index),
        "serving_index_sha256": sha256(serving / index.name),
        "questions": str(questions),
        "questions_sha256": sha256(questions),
        "runtime_sources": str(TRAIN_RUN / "runtime-sources"),
    })
    return serving, questions


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=False)
    children: list[tuple[subprocess.Popen, object]] = []
    try:
        wait_checkpoint()
        model, questions = prepare_model()
        common = dict(
            os.environ,
            HF_HOME="/weka/projects/bvandur1/zjiang31/.cache/huggingface",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            OMP_NUM_THREADS="8",
            NCCL_IB_DISABLE="1",
            PYTHONUNBUFFERED="1",
            PYTHONPATH=str(TRAIN_RUN / "runtime-sources"),
        )
        servers = []
        for slot, port in enumerate(PORTS[:2]):
            command = [
                "bash", str(SIF), "python", "-m", "sglang.launch_server",
                "--model-path", str(model), "--served-model-name", "Qwen/Qwen3.5-9B",
                "--weight-version", str(ITERATION), "--host", "127.0.0.1",
                "--port", str(port), "--tp-size", "1", "--dtype", "bfloat16",
                "--context-length", "32768", "--mem-fraction-static", "0.75",
                "--max-running-requests", "32", "--chunked-prefill-size", "8192",
                "--max-prefill-tokens", "16384", "--cuda-graph-max-bs", "32",
                "--nccl-port", str(port + 100), "--skip-server-warmup",
            ]
            log = (ROOT / f"server-{slot}.log").open("x")
            process = subprocess.Popen(
                command,
                env=dict(common, CUDA_VISIBLE_DEVICES=str(slot)),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            children.append((process, log))
            servers.append(process)
        write(ROOT / "status.json", {
            "stage": "starting-servers", "host": socket.gethostname(),
            "server_pids": [process.pid for process in servers], "time": time.time(),
        })
        for port in PORTS[:2]:
            wait_http(f"http://127.0.0.1:{port}/get_model_info", servers)

        router_command = [
            "bash", str(SIF), "python", "-m", "sglang_router.launch_router",
            "--host", "127.0.0.1", "--port", str(PORTS[2]), "--worker-urls",
            f"http://127.0.0.1:{PORTS[0]}", f"http://127.0.0.1:{PORTS[1]}",
            "--policy", "round_robin", "--request-timeout-secs", "900",
            "--max-concurrent-requests", "512", "--queue-size", "1024",
        ]
        router_log = (ROOT / "router.log").open("x")
        router = subprocess.Popen(
            router_command,
            env=dict(common, CUDA_VISIBLE_DEVICES=""),
            stdin=subprocess.DEVNULL,
            stdout=router_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children.append((router, router_log))
        deadline = time.monotonic() + 300
        while True:
            if router.poll() is not None:
                raise RuntimeError(f"router exited {router.returncode}")
            try:
                with socket.create_connection(("127.0.0.1", PORTS[2]), timeout=3):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("router readiness timeout")
                time.sleep(2)

        output = ROOT / f"eval-{ITERATION:04d}"
        command = [
            "bash", str(SIF), "python", "-u", str(COLLECTOR),
            "--checkpoint", str(model), "--url", f"http://127.0.0.1:{PORTS[2]}",
            "--model", "Qwen/Qwen3.5-9B", "--policy-version", POLICY_VERSION,
            "--server-weight-version", str(ITERATION), "--value-version", "none",
            "--questions", str(questions), "--output", str(output), "--split", "val",
            "--evaluation", "--eval-branches", "1", "--pass-tokens", "1",
            "--max-pass-attempts", "12", "--concurrency", "12",
        ]
        write(ROOT / "status.json", {
            "stage": "evaluation", "output": str(output), "time": time.time(),
        })
        with (ROOT / "collector.log").open("x") as log:
            result = subprocess.run(
                command, env=common, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"collector exited {result.returncode}")
        # The standalone evaluator owns questions outside the collector output;
        # copy them in so the immutable whole-trace auditor can verify identity.
        (output / "questions.json").write_bytes(questions.read_bytes())
        with (ROOT / "audit.log").open("x") as log:
            result = subprocess.run(
                [str(PYTHON), str(AUDITOR), "--batch", str(output)],
                env=common, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"evaluation audit exited {result.returncode}")
        summary = json.loads((output / "summary.json").read_text())
        if len(summary["results"]) != 338:
            raise ValueError("Final evaluation is incomplete")
        complete = {
            "native_iteration": ITERATION,
            "actor_optimizer_updates": ACTOR_UPDATES,
            "policy_version": POLICY_VERSION,
            "correct": float(summary["correct"]),
            "total": int(summary["branches"]),
            "accuracy": float(summary["correct"]) / int(summary["branches"]),
            "breakdown": breakdown(summary["results"]),
            "evaluation_audit": str(output / "evaluation-audit.json"),
            "finished": time.time(),
        }
        write(ROOT / "complete.json", complete)
        write(ROOT / "status.json", {
            "stage": "complete", "accuracy": complete["accuracy"], "time": time.time(),
        })
    except BaseException as exc:
        write(ROOT / "failed.json", {"error": repr(exc), "time": time.time()})
        raise
    finally:
        for process, log in reversed(children):
            stop(process)
            log.close()


if __name__ == "__main__":
    main()
