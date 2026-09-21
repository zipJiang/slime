#!/usr/bin/env bash
# Execute in the pinned CUDA image with this checkout's uv overlay first on PATH.
set -euo pipefail
# Ray workers plus concurrent collectors can exceed the login soft task limit.
# Raise only the inherited soft limit, bounded by the administrator's hard limit.
slime_task_soft=$(ulimit -Su)
slime_task_hard=$(ulimit -Hu)
slime_task_target=16384
if [[ "$slime_task_hard" != unlimited && "$slime_task_hard" -lt "$slime_task_target" ]]; then
  slime_task_target=$slime_task_hard
fi
if [[ "$slime_task_soft" != unlimited && "$slime_task_soft" -lt "$slime_task_target" ]]; then
  ulimit -Su "$slime_task_target"
fi
experiment_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
slime_workspace=$(cd "$experiment_root/../../.." && pwd)
slime_root="$slime_workspace/slime"
slime_snapshot="$experiment_root/snapshots/slime-streamed-logprobs-v4"
slime_image=${SLIME_SIF:-/weka/projects/bvandur1/zjiang31/slime/images/slime-latest.sif}
[[ -f "$slime_image" ]] || { echo "SIF missing: $slime_image; set SLIME_SIF" >&2; exit 1; }
slime_uv=$(command -v uv)
slime_uv=$(readlink -f "$slime_uv")
slime_workspace=$(dirname "$slime_root")
slime_bind=(--bind "$slime_workspace:$slime_workspace" --bind "$slime_uv:/opt/slime-uv:ro")
[[ ! -d /weka ]] || slime_bind+=(--bind /weka:/weka)
slime_env=(--env "VIRTUAL_ENV=$slime_root/.venv"
  --env "PATH=$slime_root/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  --env "PYTHONPATH=$experiment_root/scripts:$experiment_root/snapshots/native-support-v1:$slime_snapshot:/root/Megatron-LM"
  --env "UV_CACHE_DIR=$slime_root/.cache/uv"
  --env "PYTHONUNBUFFERED=1")
# Preserve explicit job/device and communication settings without importing the
# host Python/CUDA library paths into the container.
for slime_key in BROWSECOMP_PROFILE CUDA_VISIBLE_DEVICES CUDA_DEVICE_MAX_CONNECTIONS OMP_NUM_THREADS \
  NCCL_IB_DISABLE NCCL_SOCKET_IFNAME NCCL_CUMEM_ENABLE NCCL_NVLS_ENABLE \
  GLOO_SOCKET_IFNAME MASTER_ADDR MASTER_PORT RAY_ADDRESS RAY_TMPDIR \
  HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE WANDB_MODE WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY \
  SLURM_JOB_ID SLURM_JOBID SLURM_STEP_ID SLURM_NODELIST SLURM_PROCID SLURM_LOCALID; do
  [[ ! -v "$slime_key" ]] || slime_env+=(--env "$slime_key=${!slime_key}")
done
exec apptainer exec --nv --cleanenv "${slime_bind[@]}" "${slime_env[@]}" \
  --pwd "$slime_snapshot" "$slime_image" "$@"
