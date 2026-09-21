#!/usr/bin/env bash
set -euo pipefail
experiment_root=$(cd "$(dirname "$0")/.." && pwd)
role=${1:?head, train, rollout, replica}
address=${2:?head address}
devices=${3:?physical CUDA device list}
export CUDA_VISIBLE_DEVICES=$devices
export RAY_TMPDIR=/tmp/locbench-ppo-ray-v1
export OMP_NUM_THREADS=4 NCCL_IB_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_SOCKET_IFNAME=ens0 NCCL_SOCKET_IFNAME=ens0
count=$(awk -F, '{print NF}' <<< "$devices")
if [[ $role == head || $role == train ]]; then
  if [[ $role == head ]]; then
    ray_args=(--head --port=6976 --dashboard-port=8876)
  else
    ray_args=(--address="$address")
  fi
  train_count=${LOC_TRAIN_GPUS_PER_NODE:-$count}
  rollout_count=$((count-train_count))
  if [[ $rollout_count -gt 0 ]]; then
    resources="{\"locbench_ppo_train\":$train_count,\"locbench_ppo_rollout\":$rollout_count}"
  else
    resources="{\"locbench_ppo_train\":$train_count}"
  fi
elif [[ $role == rollout ]]; then
  ray_args=(--address="$address")
  resources="{\"locbench_ppo_rollout\":$count}"
elif [[ $role == replica ]]; then
  ray_args=(--address="$address")
  resources="{\"locbench_ppo_replica\":$count}"
else
  exit 2
fi
exec bash "$experiment_root/scripts/sif.sh" python "$experiment_root/snapshots/native-support-v1/with_torch_cudnn.py" \
  ray start "${ray_args[@]}" --num-gpus="$count" --num-cpus=24 \
  --dashboard-agent-listen-port=52777 --runtime-env-agent-port=52778 \
  --resources="$resources" --object-store-memory=4294967296 --disable-usage-stats --block
