#!/usr/bin/env bash
# Resume native model/optimizer/RNG state and the exact saved question cursor.
set -euo pipefail
experiment_root=$(cd "$(dirname "$0")/.." && pwd)
: "${PPO_RESUME_RUN:?Set PPO_RESUME_RUN to the previous run directory}"
: "${PPO_RUN_NAME:?Use a fresh PPO_RUN_NAME for the recovery attempt}"
resume_root=$(cd "$PPO_RESUME_RUN" && pwd)
[[ ! -f "$experiment_root/runs/$PPO_RUN_NAME/recipe.json" ]]
[[ -f "$resume_root/actor/latest_checkpointed_iteration.txt" ]]
[[ -f "$resume_root/critic/latest_checkpointed_iteration.txt" ]]
actor_iteration=$(<"$resume_root/actor/latest_checkpointed_iteration.txt")
critic_iteration=$(<"$resume_root/critic/latest_checkpointed_iteration.txt")
[[ "$actor_iteration" =~ ^[0-9]+$ && "$actor_iteration" == "$critic_iteration" ]]
start_round=$((10#$actor_iteration + 1))
exec bash "$experiment_root/scripts/run_ppo.sh" \
  --load "$resume_root/actor" --ppo-critic-load "$resume_root/critic" \
  --start-rollout-id "$start_round" "$@"
