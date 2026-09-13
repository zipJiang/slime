#!/usr/bin/env bash
set -euo pipefail
experiment_root=$(cd "$(dirname "$0")/.." && pwd)
# Slurm copies batch scripts, so the submitted path is supplied explicitly.
experiment_root=${PPO_EXPERIMENT_ROOT:?}
exec /weka/scratch/jhu/bvandur1/zjiang31/rollout-controller/.venv/bin/python "$experiment_root/operations/supervise.py" "$@"
