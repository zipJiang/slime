#!/usr/bin/env bash
set -euo pipefail
experiment_root=$(cd "$(dirname "$0")/.." && pwd)
checkpoint=/weka/projects/bvandur1/zjiang31/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
collection_run=${CRITIC_RUN_ROOT:-$experiment_root/runs/base-v1}
run_root=${CRITIC_TRAIN_OUTPUT:-$collection_run/training}
train_nodes=${CRITIC_TRAIN_NODES:-2}
export RAY_ADDRESS=${RAY_ADDRESS:-172.16.203.1:6475}
export OMP_NUM_THREADS=4 NCCL_IB_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_SOCKET_IFNAME=ens0 NCCL_SOCKET_IFNAME=ens0
source "$experiment_root/snapshots/slime/scripts/models/qwen3.5-9B.sh"
exec bash "$experiment_root/scripts/sif.sh" python "$experiment_root/scripts/with_torch_cudnn.py" \
  python "$experiment_root/scripts/train_critic.py" "${MODEL_ARGS[@]}" \
  --actor-num-nodes "$train_nodes" --actor-num-gpus-per-node 2 --num-gpus-per-node 2 --rollout-num-gpus 0 \
  --hf-checkpoint "$checkpoint" --load "$checkpoint" --save "$run_root/native" --save-interval 16 \
  --rollout-function-path slime.rollout.sglang_rollout.generate_rollout \
  --num-rollout 16 --num-critic-only-steps 16 --rollout-batch-size 8 --n-samples-per-prompt 1 \
  --num-steps-per-rollout 1 --global-batch-size 8 --skip-eval-before-train \
  --advantage-estimator ppo --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.2 \
  --optimizer adam --lr 5e-6 --lr-decay-style constant --weight-decay 0 \
  --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1 \
  --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --sequence-parallel --use-distributed-optimizer \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --seq-length 32768 --use-dynamic-batch-size --max-tokens-per-gpu 24576 --balance-data \
  --attention-dropout 0 --hidden-dropout 0 --attention-backend flash \
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --critic-collection "$collection_run/collection" "$@"
