#!/usr/bin/env bash
set -euo pipefail
experiment_root=$(cd "$(dirname "$0")/.." && pwd)
role=${1:?}; address=${2:?}; devices=${3:?}
export CUDA_VISIBLE_DEVICES=$devices RAY_TMPDIR=/tmp/deontic-vine-ray-v1
export OMP_NUM_THREADS=4 NCCL_IB_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_SOCKET_IFNAME=ens0 NCCL_SOCKET_IFNAME=ens0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
count=$(awk -F, '{print NF}' <<< "$devices")
if [[ $role == train ]]; then
  ray_args=(--head --port=6428 --dashboard-port=8308)
else
  ray_args=(--address="$address")
fi
exec bash "$experiment_root/scripts/sif.sh" python "$experiment_root/scripts/with_torch_cudnn.py" \
  ray start "${ray_args[@]}" --num-gpus="$count" --num-cpus=24 \
  --dashboard-agent-listen-port=55331 --runtime-env-agent-port=55332 \
  --resources="{\"deontic_direct_branch_$role\":$count}" \
  --object-store-memory=4294967296 --disable-usage-stats --block
