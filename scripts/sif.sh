#!/usr/bin/env bash
# Execute in the pinned CUDA image with this checkout's uv overlay first on PATH.
set -euo pipefail
slime_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
slime_image=${SLIME_SIF:-/weka/projects/bvandur1/zjiang31/slime/images/slime-latest.sif}
[[ -f "$slime_image" ]] || { echo "SIF missing: $slime_image; set SLIME_SIF" >&2; exit 1; }
slime_uv=$(command -v uv)
slime_uv=$(readlink -f "$slime_uv")
slime_workspace=$(dirname "$slime_root")
slime_bind=(--bind "$slime_workspace:$slime_workspace" --bind "$slime_uv:/opt/slime-uv:ro")
[[ ! -d /weka ]] || slime_bind+=(--bind /weka:/weka)
slime_env=(--env "VIRTUAL_ENV=$slime_root/.venv"
  --env "PATH=$slime_root/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  --env "PYTHONPATH=$slime_root:/root/Megatron-LM${SLIME_EXTRA_PYTHONPATH:+:$SLIME_EXTRA_PYTHONPATH}"
  --env "UV_CACHE_DIR=$slime_root/.cache/uv"
  --env "PYTHONUNBUFFERED=1")
# Preserve explicit job/device and communication settings without importing the
# host Python/CUDA library paths into the container.
for slime_key in CUDA_VISIBLE_DEVICES CUDA_DEVICE_MAX_CONNECTIONS OMP_NUM_THREADS \
  NCCL_IB_DISABLE NCCL_SOCKET_IFNAME NCCL_CUMEM_ENABLE NCCL_NVLS_ENABLE \
  GLOO_SOCKET_IFNAME MASTER_ADDR MASTER_PORT RAY_ADDRESS RAY_TMPDIR \
  HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE WANDB_MODE WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY \
  SLURM_JOB_ID SLURM_JOBID SLURM_STEP_ID SLURM_NODELIST SLURM_PROCID SLURM_LOCALID; do
  [[ ! -v "$slime_key" ]] || slime_env+=(--env "$slime_key=${!slime_key}")
done
exec apptainer exec --nv --cleanenv "${slime_bind[@]}" "${slime_env[@]}" \
  --pwd "$slime_root" "$slime_image" "$@"
