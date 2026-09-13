#!/usr/bin/env bash
set -euo pipefail
experiment_root=$(cd "$(dirname "$0")/.." && pwd)
role=${1:?train or inference}
address=${2:-}
export RAY_TMPDIR=/tmp/browsecomp-pilot-ray
export SLIME_EXTRA_PYTHONPATH="$experiment_root/scripts"
export OMP_NUM_THREADS=4 NCCL_IB_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_SOCKET_IFNAME=ens0 NCCL_SOCKET_IFNAME=ens0
if [[ "$role" == train ]]; then
  ray_args=(--head --port=6485 --dashboard-port=8385 --num-gpus=4 --num-cpus=32
    --resources='{"browsecomp_pilot_train":4}')
elif [[ "$role" == inference ]]; then
  ray_args=(--address="${address:?head address}" --num-gpus=3 --num-cpus=32
    --resources='{"browsecomp_pilot_rollout":2,"browsecomp_pilot_replica":1}')
else
  echo "unknown pilot Ray role: $role" >&2; exit 2
fi
exec bash "$experiment_root/scripts/sif.sh" python "$experiment_root/scripts/with_torch_cudnn.py" \
  ray start "${ray_args[@]}" --object-store-memory=4294967296 --disable-usage-stats --block
