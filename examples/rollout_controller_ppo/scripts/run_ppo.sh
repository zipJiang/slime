#!/usr/bin/env bash
set -euo pipefail
experiment_root=$(cd "$(dirname "$0")/.." && pwd)
workspace_root=$(cd "$experiment_root/../../.." && pwd)
slime_root="$workspace_root/slime"
checkpoint="/weka/projects/bvandur1/zjiang31/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
run_root="$experiment_root/runs/${PPO_RUN_NAME:?Set a fresh PPO_RUN_NAME}"
storage_root=${PPO_STORAGE_ROOT:-/weka/projects/bvandur1/zjiang31/deontic-ppo-overlap-9b/runs}
mkdir -p "$experiment_root/runs" "$storage_root/$PPO_RUN_NAME"
if [[ ! -e "$run_root" ]]; then
  ln -s "$storage_root/$PPO_RUN_NAME" "$run_root"
fi
export SLIME_EXTRA_PYTHONPATH="$experiment_root/scripts"
export RAY_ADDRESS=${RAY_ADDRESS:-172.16.204.2:6405}
export OMP_NUM_THREADS=4 NCCL_IB_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_SOCKET_IFNAME=ens0 NCCL_SOCKET_IFNAME=ens0
source "$experiment_root/snapshots/slime/scripts/models/qwen3.5-9B.sh"
exec bash "$experiment_root/scripts/sif.sh" python "$experiment_root/scripts/with_torch_cudnn.py" \
  python "$experiment_root/scripts/train_slime.py" \
  "${MODEL_ARGS[@]}" \
  --actor-num-nodes 1 --actor-num-gpus-per-node 4 --rollout-num-gpus "${PPO_ROLLOUT_GPUS:-6}" \
  --rollout-num-gpus-per-engine 1 --num-gpus-per-node 2 \
  --hf-checkpoint "$checkpoint" --load "$checkpoint" --ref-load "$checkpoint" \
  --save "$run_root/actor" --save-hf "$run_root/hf/iter_{rollout_id:07d}" --save-interval 12 \
  --rollout-function-path slime_shim.generate_rollout \
  --eval-function-path slime_shim.generate_rollout \
  --custom-convert-samples-to-train-data-path slime_shim.convert_actor_data \
  --rollout-data-postprocess-path audit_on_policy.check \
  --prompt-data "$experiment_root/data/train-questions-v2.jsonl" --input-key prompt \
  --num-rollout 126 --num-critic-only-steps 6 --rollout-batch-size 6 --n-samples-per-prompt 1 \
  --num-steps-per-rollout 1 --global-batch-size 6 --eval-interval 12 --skip-eval-before-train \
  --eval-prompt-data airline "$experiment_root/data/val-airline.jsonl" \
    sara_numeric "$experiment_root/data/val-sara_numeric.jsonl" \
    sara_binary "$experiment_root/data/val-sara_binary.jsonl" \
  --n-samples-per-eval-prompt 8 \
  --rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1 \
  --rollout-max-response-len 6144 \
  --advantage-estimator ppo --custom-advantage-function-path targets.prepared_advantages \
  --use-rollout-logprobs --get-mismatch-metrics \
  --custom-tis-function-path audit_on_policy.metrics \
  --use-kl-loss --kl-loss-coef 0.01 --kl-loss-type low_var_kl \
  --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.2 \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0 \
  --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1 \
  --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 --sequence-parallel --use-distributed-optimizer \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --seq-length 32768 --use-dynamic-batch-size --max-tokens-per-gpu 24576 --balance-data \
  --attention-dropout 0 --hidden-dropout 0 --attention-backend flash \
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --sglang-mem-fraction-static 0.75 --sglang-context-length 32768 \
  --sglang-max-running-requests 32 --sglang-cuda-graph-max-bs 32 \
  --router-balance-abs-threshold "${PPO_ROUTER_BALANCE_ABS_THRESHOLD:-1}" \
  --use-wandb --wandb-mode online --wandb-project deontic-compaction-ppo \
  --disable-wandb-random-suffix \
  --wandb-group "${PPO_RUN_NAME:-direct-branch-v1}" --wandb-dir "$run_root/wandb" \
  "$@"
