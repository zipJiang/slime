#!/usr/bin/env bash
set -euo pipefail
experiment_root=$(cd "$(dirname "$0")/.." && pwd)
role=${1:?head, trainer, validator}
address=${2:?head address}
devices=${3:?device list}
export CUDA_VISIBLE_DEVICES=$devices
export RAY_TMPDIR=/tmp/locbench-critic-ray-v1
export OMP_NUM_THREADS=4 NCCL_IB_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_SOCKET_IFNAME=ens0 NCCL_SOCKET_IFNAME=ens0
if [[ $role == head ]]; then
  ray_args=(--head --port=6975 --dashboard-port=8875)
else
  ray_args=(--address="$address")
fi
count=$(awk -F, '{print NF}' <<< "$devices")
if [[ $role == validator ]]; then
  resources='{"locbench_critic_validator":1}'
elif [[ $role == head || $role == trainer ]]; then
  resources="{\"locbench_critic_train\":$count}"
else
  exit 2
fi
exec bash "$experiment_root/scripts/sif.sh" python "$experiment_root/snapshots/native-support-v1/with_torch_cudnn.py" \
  ray start "${ray_args[@]}" --num-gpus="$count" --num-cpus=24 \
  --dashboard-agent-listen-port=52775 --runtime-env-agent-port=52776 \
  --resources="$resources" --object-store-memory=4294967296 --disable-usage-stats --block
