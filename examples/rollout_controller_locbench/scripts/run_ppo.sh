#!/usr/bin/env bash
set -euo pipefail
experiment_root=$(cd "$(dirname "$0")/.." && pwd)
checkpoint=/weka/projects/bvandur1/zjiang31/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
checkpoint=${LOC_ACTOR_CHECKPOINT:-$checkpoint}
run_root=${LOC_PPO_OUTPUT:?Set a fresh LocBench PPO output directory}
candidate=${LOC_CRITIC_CANDIDATE:-$experiment_root/runs/base-critic-v2/training-v2/warmstart-candidate.json}
updates=${LOC_PPO_UPDATES:-120}
batch_size=${LOC_PPO_BATCH_SIZE:-4}
if [[ ${LOC_ARGUMENT_PREFLIGHT:-0} != 1 ]]; then
/weka/scratch/jhu/bvandur1/zjiang31/rollout-controller/.venv/bin/python \
  "$experiment_root/scripts/prepare_ppo.py" --candidate "$candidate" --output "$run_root/data" \
  --updates "$updates" --batch-size "$batch_size"
fi
resume_args=(--loc-collection-profile "${LOC_COLLECTION_PROFILE:-original}")
if [[ ${LOC_ALLOW_PROFILE_TRANSITION:-0} == 1 ]]; then
  resume_args+=(--loc-allow-profile-transition)
fi
if [[ ${LOC_PROFILE_COMPARISON_ONLY:-0} == 1 ]]; then
  resume_args+=(--loc-profile-comparison-only)
fi
if [[ -n ${LOC_MEMORY_STRESS_SOURCE:-} ]]; then
  resume_args+=(--loc-memory-stress-source "$LOC_MEMORY_STRESS_SOURCE")
fi
if [[ -n ${LOC_MEMORY_PREFLIGHT_SOURCE:-} ]]; then
  resume_args+=(--loc-memory-preflight-source "$LOC_MEMORY_PREFLIGHT_SOURCE")
fi
if [[ -n ${LOC_TRAIN_ALLOCATOR_CONF:-} ]]; then
  allocator_json=$(python -c 'import json,os; print(json.dumps({"PYTORCH_CUDA_ALLOC_CONF":os.environ["LOC_TRAIN_ALLOCATOR_CONF"]}))')
  resume_args+=(--train-env-vars "$allocator_json")
fi
if [[ -n ${LOC_BENCHMARK_SOURCE:-} ]]; then
  resume_args+=(--loc-benchmark-source "$LOC_BENCHMARK_SOURCE")
fi
if [[ ${LOC_PPO_BENCHMARK_ONLY:-0} == 1 ]]; then
  resume_args+=(--loc-benchmark-only)
fi
if [[ -n ${LOC_EFFICIENCY_PLAN:-} ]]; then
  resume_args+=(--loc-efficiency-plan "$LOC_EFFICIENCY_PLAN")
fi
if [[ -n ${LOC_PPO_RESUME_RUN:-} ]]; then
  resume_args+=(--loc-resume-run "$LOC_PPO_RESUME_RUN")
fi
if [[ ${LOC_ALLOW_DP_RESHARD:-0} == 1 ]]; then
  resume_args+=(--loc-allow-dp-reshard)
fi
if [[ -n ${LOC_PPO_STOP_AFTER_ROUND:-} ]]; then
  resume_args+=(--loc-stop-after-round "$LOC_PPO_STOP_AFTER_ROUND")
fi
export RAY_ADDRESS=${RAY_ADDRESS:?Set the dedicated LocBench PPO Ray address}
export OMP_NUM_THREADS=4 NCCL_IB_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_SOCKET_IFNAME=ens0 NCCL_SOCKET_IFNAME=ens0
source "$experiment_root/snapshots/slime/scripts/models/qwen3.5-9B.sh"
exec bash "$experiment_root/scripts/sif.sh" python "$experiment_root/snapshots/native-support-v1/with_torch_cudnn.py" \
  python "$experiment_root/scripts/train_ppo.py" "${MODEL_ARGS[@]}" \
  --actor-num-nodes "${LOC_TRAIN_NODES:-1}" --actor-num-gpus-per-node "${LOC_TRAIN_GPUS_PER_NODE:-4}" \
  --rollout-num-gpus "${LOC_ROLLOUT_GPUS:-10}" --rollout-num-gpus-per-engine 1 --num-gpus-per-node 2 \
  --hf-checkpoint "$checkpoint" --load "$checkpoint" --ref-load "$checkpoint" \
  --save "$run_root/actor" --save-interval 6 \
  --rollout-function-path ppo_collection.unused_rollout --prompt-data "$run_root/data/schedule.jsonl" --input-key prompt \
  --rollout-data-postprocess-path ppo_on_policy.check --custom-advantage-function-path targets.prepared_advantages \
  --num-rollout "$updates" --num-critic-only-steps 0 --rollout-batch-size "$batch_size" \
  --n-samples-per-prompt 1 --num-steps-per-rollout 1 --global-batch-size "$batch_size" \
  --skip-eval-before-train --rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1 \
  --rollout-max-response-len 16384 --advantage-estimator ppo \
  --use-rollout-logprobs --get-mismatch-metrics --custom-tis-function-path ppo_on_policy.metrics \
  --use-kl-loss --kl-loss-coef .01 --kl-loss-type low_var_kl --entropy-coef 0 --eps-clip .2 --eps-clip-high .2 \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0 \
  --adam-beta1 .9 --adam-beta2 .95 --clip-grad 1 \
  --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --sequence-parallel --use-distributed-optimizer \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --max-position-embeddings 98304 --seq-length 98304 --use-dynamic-batch-size --max-tokens-per-gpu 24576 --balance-data \
  --log-probs-chunk-size 1024 --attention-dropout 0 --hidden-dropout 0 --attention-backend flash \
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --sglang-mem-fraction-static .80 --sglang-context-length 98304 \
  --sglang-max-running-requests 16 --sglang-cuda-graph-max-bs 16 \
  --loc-candidate "$candidate" --loc-critic-replica-host "${LOC_CRITIC_REPLICA_HOST:-172.16.203.8}" \
  --loc-critic-lr 1e-6 --loc-pass-tokens "${LOC_PASS_TOKENS:-32768}" \
  --loc-search-concurrency "${LOC_SEARCH_CONCURRENCY:-4}" --loc-max-pass-attempts 32 \
  "${resume_args[@]}" "$@"
