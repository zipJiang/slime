#!/usr/bin/env bash
set -euo pipefail
experiment_root=$(cd "$(dirname "$0")/.." && pwd)
workspace_root=$(cd "$experiment_root/../../.." && pwd)
role=${1:?train or rollout}
address=${2:-}
rollout_gpus=${3:-4}
export RAY_TMPDIR=/tmp/deontic-direct-branch-ray
export SLIME_EXTRA_PYTHONPATH="$experiment_root/scripts"
export OMP_NUM_THREADS=4 NCCL_IB_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_SOCKET_IFNAME=ens0 NCCL_SOCKET_IFNAME=ens0
if [[ "$role" == train ]]; then
  ray_args=(--head --port=6405 --dashboard-port=8285 --num-gpus=4 --num-cpus=32
    --resources='{"deontic_direct_branch_train":4}')
else
  ray_args=(--address="${address:?head address}" --num-gpus="$rollout_gpus" --num-cpus=32
    --resources="{\"deontic_direct_branch_rollout\":$rollout_gpus}")
fi
exec bash "$experiment_root/scripts/sif.sh" ray start "${ray_args[@]}" \
  --object-store-memory=4294967296 --disable-usage-stats --block
