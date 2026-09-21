#!/usr/bin/env bash
set -euo pipefail
experiment_root=$(cd "$(dirname "$0")/.." && pwd)
checkpoint=/weka/projects/bvandur1/zjiang31/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
run_root=${IMITATION_OUTPUT:?}
source_run=${IMITATION_SOURCE:?}
train_nodes=${IMITATION_TRAIN_NODES:-2}
export OMP_NUM_THREADS=4 NCCL_IB_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_SOCKET_IFNAME=ens0 NCCL_SOCKET_IFNAME=ens0
source "$experiment_root/snapshots/slime/scripts/models/qwen3.5-9B.sh"
exec bash "$experiment_root/scripts/sif.sh" python "$experiment_root/snapshots/native-support-v1/with_torch_cudnn.py" \
  python "$experiment_root/scripts/train_imitation.py" "${MODEL_ARGS[@]}" \
  --actor-num-nodes "$train_nodes" --actor-num-gpus-per-node 2 --num-gpus-per-node 2 --rollout-num-gpus 0 \
  --hf-checkpoint "$checkpoint" --load "$checkpoint" --save "$run_root/native" --save-interval 4 \
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout \
  --num-rollout 30 --num-critic-only-steps 0 --rollout-batch-size 8 --n-samples-per-prompt 1 \
  --num-steps-per-rollout 1 --global-batch-size 8 --decrease-batch-size-if-needed --skip-eval-before-train \
  --advantage-estimator grpo --loss-type sft_loss --disable-compute-advantages-and-returns \
  --optimizer adam --lr 5e-6 --lr-decay-style constant --weight-decay 0 \
  --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1 \
  --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --sequence-parallel --use-distributed-optimizer --offload-train \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --seq-length 98304 --max-position-embeddings 98304 --use-dynamic-batch-size --max-tokens-per-gpu 32768 --balance-data \
  --log-probs-chunk-size 1024 --rollout-temperature 1 \
  --attention-dropout 0 --hidden-dropout 0 --attention-backend flash \
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --imitation-source "$source_run" "$@"
