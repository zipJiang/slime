#!/usr/bin/env bash
set -euo pipefail
experiment_root=$(cd "$(dirname "$0")/.." && pwd)
export RAY_TMPDIR=/tmp/browsecomp-critic-ray
export OMP_NUM_THREADS=4 NCCL_IB_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_SOCKET_IFNAME=ens0 NCCL_SOCKET_IFNAME=ens0
if [[ ${1:?head or worker} == head ]]; then
  ray_args=(--head --port=6475 --dashboard-port=8375)
else
  ray_args=(--address="${2:?head address}")
fi
exec bash "$experiment_root/scripts/sif.sh" python "$experiment_root/scripts/with_torch_cudnn.py" \
  ray start "${ray_args[@]}" --num-gpus=2 --num-cpus=24 \
  --resources='{"browsecomp_critic_train":2}' \
  --object-store-memory=4294967296 --disable-usage-stats --block
