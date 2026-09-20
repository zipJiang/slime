#!/usr/bin/env bash
set -euo pipefail
experiment_root=$(cd "$(dirname "$0")/.." && pwd)
checkpoint=${LOC_VINE_ACTOR_CHECKPOINT:-/weka/projects/bvandur1/zjiang31/locbench-9b/runs/imitation-24k-read60-lr1e6-v1/hf/iter_0000005}
run_root="$experiment_root/runs/${PPO_RUN_NAME:?Set a fresh PPO_RUN_NAME}"
storage_root=${PPO_STORAGE_ROOT:-/weka/projects/bvandur1/zjiang31/locbench-vine-9b/runs}
mkdir -p "$experiment_root/runs" "$storage_root/$PPO_RUN_NAME"
if [[ ! -e "$run_root" ]]; then
  ln -s "$storage_root/$PPO_RUN_NAME" "$run_root"
fi
export SLIME_EXTRA_PYTHONPATH="$experiment_root/scripts"
export RAY_ADDRESS=${RAY_ADDRESS:?Set the dedicated LocBench VinePPO Ray address}
export OMP_NUM_THREADS=4 NCCL_IB_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_SOCKET_IFNAME=ens0 NCCL_SOCKET_IFNAME=ens0
source "$experiment_root/snapshots/slime/scripts/models/qwen3.5-9B.sh"
exec bash "$experiment_root/scripts/sif.sh" python "$experiment_root/scripts/with_torch_cudnn.py" \
  python "$experiment_root/scripts/train_vine_publication.py" \
  "${MODEL_ARGS[@]}" \
  --actor-num-nodes 1 --actor-num-gpus-per-node 4 --rollout-num-gpus "${PPO_ROLLOUT_GPUS:-5}" \
  --rollout-num-gpus-per-engine 1 --num-gpus-per-node 4 \
  --hf-checkpoint "$checkpoint" --load "$checkpoint" --ref-load "$checkpoint" \
  --save "$run_root/actor" --save-hf "$run_root/hf/iter_{rollout_id:07d}" --save-interval 6 \
  --rollout-function-path slime_shim.generate_rollout \
  --custom-convert-samples-to-train-data-path slime_shim.convert_actor_data \
  --rollout-data-postprocess-path audit_on_policy.check \
  --prompt-data "$experiment_root/data/train-schedule.jsonl" --input-key prompt \
  --num-rollout "${PPO_NUM_ROLLOUT:-97}" --num-critic-only-steps 0 \
  --rollout-batch-size 4 --n-samples-per-prompt 1 --num-steps-per-rollout 1 --global-batch-size 4 \
  --skip-eval-before-train \
  --rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1 --rollout-max-response-len 16384 \
  --advantage-estimator ppo --custom-advantage-function-path targets.prepared_advantages \
  --use-rollout-logprobs --get-mismatch-metrics --log-probs-chunk-size 1024 \
  --custom-tis-function-path audit_on_policy.metrics \
  --use-kl-loss --kl-loss-coef .01 --kl-loss-type low_var_kl \
  --entropy-coef 0 --eps-clip .2 --eps-clip-high .2 \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0 \
  --adam-beta1 .9 --adam-beta2 .95 --clip-grad 1 \
  --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --sequence-parallel --use-distributed-optimizer \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --max-position-embeddings 98304 --seq-length 98304 --use-dynamic-batch-size \
  --max-tokens-per-gpu 24576 --log-probs-max-tokens-per-gpu 24576 --balance-data \
  --attention-dropout 0 --hidden-dropout 0 --attention-backend flash \
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --sglang-mem-fraction-static .80 --sglang-context-length 98304 \
  --sglang-max-running-requests 16 --sglang-cuda-graph-max-bs 16 \
  --router-balance-abs-threshold 1 \
  --use-wandb --wandb-mode online --wandb-project locbench-vine-ppo \
  --disable-wandb-random-suffix --wandb-group "$PPO_RUN_NAME" --wandb-dir "$run_root/wandb" \
  "$@"
