#!/usr/bin/env bash
#SBATCH --job-name=loc-grpo-expand8
#SBATCH --partition=med
#SBATCH --account=bvandur1
#SBATCH --qos=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=1-00:00:00
#SBATCH --output=/weka/scratch/jhu/bvandur1/zjiang31/slime-ppo-worktree/examples/rollout_controller_vine_locbench/operations/grpo-expand8-sep20-%j.log
set -euo pipefail
experiment=/weka/scratch/jhu/bvandur1/zjiang31/slime-ppo-worktree/examples/rollout_controller_vine_locbench
source_run=/weka/projects/bvandur1/zjiang31/locbench-grpo-9b/runs/loc-grpo-g8-c12-sep20-c-48k
source_operation="$experiment/operations/loc-grpo-g8-c12-sep20-c-48k"
cd "$experiment"

# Resume only after the current batch is committed, its native checkpoint has
# passed a full tensor readback, and the old Ray cluster has released its GPUs.
while true; do
  if [[ -f "$source_run/failed.json" ]]; then
    echo "source GRPO failed" >&2
    exit 1
  fi
  if [[ -f "$source_run/paused.json" && -f "$source_operation/stopped.json" && -f "$source_run/actor/latest_checkpointed_iteration.txt" ]]; then
    iteration=$(<"$source_run/actor/latest_checkpointed_iteration.txt")
    audit="$source_run/actor/iter_$(printf '%07d' "$iteration")-readback.json"
    [[ -f "$audit" ]] && break
  fi
  sleep 30
done

exec /usr/bin/python3 operations/supervise_grpo_recovery_sep20.py \
  --name loc-grpo-g8-c12-sep20-d-48k-8g \
  --run-name loc-grpo-g8-c12-sep20-d-48k-8g \
  --load "$source_run/actor" \
  --group-size 8 \
  --search-concurrency 12 \
  --num-rollout 97 \
  --drain-seconds 7200 \
  --assign 562163=0,1,2,3 \
  --assign 423716=0,1,2,3 \
  --assign 633715=0,1 \
  --assign 486600=0,1 \
  --head 562163 \
  --driver-arg=--max-tokens-per-gpu --driver-arg=49152 \
  --driver-arg=--log-probs-max-tokens-per-gpu --driver-arg=49152
