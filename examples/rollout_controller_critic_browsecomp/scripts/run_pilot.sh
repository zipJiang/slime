#!/usr/bin/env bash
set -euo pipefail
experiment_root=$(cd "$(dirname "$0")/.." && pwd)
checkpoint=/weka/projects/bvandur1/zjiang31/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
export BROWSECOMP_PROFILE=${BROWSECOMP_PROFILE:-trace96k}
case "$BROWSECOMP_PROFILE" in
  trace96k) context_limit=98304; server_context=98304; reply_limit=16384 ;;
  legacy) context_limit=32768; server_context=65536; reply_limit=6144 ;;
  *) echo "Unknown BROWSECOMP_PROFILE" >&2; exit 1 ;;
esac
run_name=${PILOT_RUN_NAME:?Set a fresh PILOT_RUN_NAME}
run_root="$experiment_root/runs/$run_name"
storage_root=${PILOT_STORAGE_ROOT:-/weka/projects/bvandur1/zjiang31/browsecomp-critic-ppo/runs}
mkdir -p "$experiment_root/runs" "$storage_root"
if [[ ! -e "$run_root" && ! -L "$run_root" ]]; then
  mkdir "$storage_root/$run_name"
  ln -s "$storage_root/$run_name" "$run_root"
fi
candidate=${PILOT_CANDIDATE:-$experiment_root/runs/base-v2/training/warmstart-candidate.json}
train_nodes=${PILOT_TRAIN_NODES:-1}
train_gpus_per_node=${PILOT_TRAIN_GPUS_PER_NODE:-4}
for argument in "$@"; do
  if [[ "$argument" == --pilot-preflight-only ]]; then
    exec /weka/scratch/jhu/bvandur1/zjiang31/rollout-controller/.venv/bin/python \
      "$experiment_root/scripts/pilot_preflight.py" \
      --run "$run_root" --candidate "$candidate" \
      --context-source "${PILOT_CONTEXT_SOURCE:-$experiment_root/scripts/collect.py}" \
      --schedule-audit "$experiment_root/data/pilot-schedule-audit.json" \
      --cases /weka/scratch/jhu/bvandur1/zjiang31/rollout-controller/data/browsercomp-plus/cases.private.jsonl \
      --retriever-code /weka/projects/bvandur1/zjiang31/browsecomp-plus-retriever \
      --base-actor "$checkpoint" --updates 2 --batch-size 6 --critic-only-steps 0 \
      --train-gpus "$((train_nodes * train_gpus_per_node))" --rollout-gpus "${PILOT_ROLLOUT_GPUS:-2}" \
      --critic-replica-host "${PILOT_CRITIC_REPLICA_HOST:?}" \
      --retriever-url "${PILOT_RETRIEVER_URL:?}" --judge-url "${PILOT_JUDGE_URL:?}"
  fi
done
export SLIME_EXTRA_PYTHONPATH="$experiment_root/scripts"
export RAY_ADDRESS=${RAY_ADDRESS:?Set the pilot Ray head address}
export OMP_NUM_THREADS=4 NCCL_IB_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_SOCKET_IFNAME=ens0 NCCL_SOCKET_IFNAME=ens0
source "$experiment_root/snapshots/slime/scripts/models/qwen3.5-9B.sh"
exec bash "$experiment_root/scripts/sif.sh" python "$experiment_root/scripts/with_torch_cudnn.py" \
  python "$experiment_root/scripts/train_pilot.py" "${MODEL_ARGS[@]}" \
  --actor-num-nodes "$train_nodes" --actor-num-gpus-per-node "$train_gpus_per_node" \
  --rollout-num-gpus "${PILOT_ROLLOUT_GPUS:-2}" --rollout-num-gpus-per-engine 1 \
  --num-gpus-per-node 2 --hf-checkpoint "$checkpoint" --load "$checkpoint" --ref-load "$checkpoint" \
  --save "$run_root/actor" --save-interval 2 \
  --rollout-function-path pilot_slime_shim.generate_rollout \
  --custom-convert-samples-to-train-data-path pilot_slime_shim.convert_actor_data \
  --rollout-data-postprocess-path pilot_on_policy.check \
  --prompt-data "$experiment_root/data/pilot-schedule.jsonl" --input-key prompt \
  --num-rollout 2 --num-critic-only-steps 0 --rollout-batch-size 6 \
  --n-samples-per-prompt 1 --num-steps-per-rollout 1 --global-batch-size 6 \
  --skip-eval-before-train --rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1 \
  --rollout-max-response-len "$reply_limit" --advantage-estimator ppo \
  --custom-advantage-function-path targets.prepared_advantages \
  --use-rollout-logprobs --get-mismatch-metrics --custom-tis-function-path pilot_on_policy.metrics \
  --use-kl-loss --kl-loss-coef 0.01 --kl-loss-type low_var_kl \
  --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.2 \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0 \
  --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1 \
  --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 --sequence-parallel --use-distributed-optimizer \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --max-position-embeddings "$context_limit" --seq-length "$context_limit" --use-dynamic-batch-size --max-tokens-per-gpu 24576 --balance-data \
  --log-probs-chunk-size 1024 \
  --attention-dropout 0 --hidden-dropout 0 --attention-backend flash \
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --sglang-mem-fraction-static 0.75 --sglang-context-length "$server_context" \
  --sglang-max-running-requests 24 --sglang-cuda-graph-max-bs 24 \
  --pilot-candidate "$candidate" --pilot-context-source "${PILOT_CONTEXT_SOURCE:-$experiment_root/scripts/collect.py}" \
  --pilot-critic-lr "${PILOT_CRITIC_LR:-1e-6}" \
  --pilot-critic-equivalence-tolerance "${PILOT_CRITIC_EQUIVALENCE_TOLERANCE:-0.01}" \
  --pilot-schedule-audit "$experiment_root/data/pilot-schedule-audit.json" \
  --pilot-cases /weka/scratch/jhu/bvandur1/zjiang31/rollout-controller/data/browsercomp-plus/cases.private.jsonl \
  --pilot-retriever-code /weka/projects/bvandur1/zjiang31/browsecomp-plus-retriever \
  --pilot-infrastructure-manifest "$run_root/pilot-operations/infrastructure-manifest.json" \
  --pilot-retriever-url "${PILOT_RETRIEVER_URL:?}" --pilot-judge-url "${PILOT_JUDGE_URL:?}" \
  --pilot-critic-replica-host "${PILOT_CRITIC_REPLICA_HOST:?}" \
  --pilot-pass-tokens "${PILOT_PASS_TOKENS:-140000}" \
  --pilot-search-concurrency "${PILOT_SEARCH_CONCURRENCY:-4}" \
  --pilot-seed-namespace browsecomp-zero-warmup-pilot-v1 \
  "$@"
