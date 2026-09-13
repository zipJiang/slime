#!/usr/bin/env bash
set -euo pipefail
exec /weka/scratch/jhu/bvandur1/zjiang31/rollout-controller/.venv/bin/python "${CRITIC_EXPERIMENT_ROOT:?}/operations/collect_supervisor.py"
