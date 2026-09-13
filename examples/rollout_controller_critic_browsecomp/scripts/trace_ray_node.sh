#!/usr/bin/env bash
set -euo pipefail
experiment_root=$(cd "$(dirname "$0")/.." && pwd)
export RAY_TMPDIR=/tmp/browsecomp-trace-warmup-ray
export OMP_NUM_THREADS=4 NCCL_IB_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_SOCKET_IFNAME=ens0 NCCL_SOCKET_IFNAME=ens0 BROWSECOMP_PROFILE=trace96k
role=${1:?head, trainer, or validator}
if [[ $role == head ]]; then
  ray_args=(--head --port=6575 --dashboard-port=8475)
else
  ray_args=(--address="${2:?head address}")
fi
if [[ $role == validator ]]; then
  export CUDA_VISIBLE_DEVICES=${3:?singleton GPU}
  resources='{"browsecomp_trace_validator":1}'
  gpus=1
elif [[ $role == head || $role == trainer ]]; then
  export CUDA_VISIBLE_DEVICES=0,1
  resources='{"browsecomp_critic_train":2}'
  gpus=2
else
  echo "Unknown warmup Ray role: $role" >&2
  exit 2
fi
exec bash "$experiment_root/scripts/sif.sh" python "$experiment_root/scripts/with_torch_cudnn.py" \
  ray start "${ray_args[@]}" --num-gpus="$gpus" --num-cpus=24 \
  --resources="$resources" --object-store-memory=4294967296 --disable-usage-stats --block
