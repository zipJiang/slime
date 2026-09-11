#!/usr/bin/env bash
# Apply the same task-limit policy as sif.sh to a separately pinned launcher.
set -euo pipefail
slime_task_soft=$(ulimit -Su)
slime_task_hard=$(ulimit -Hu)
slime_task_target=16384
if [[ "$slime_task_hard" != unlimited && "$slime_task_hard" -lt "$slime_task_target" ]]; then
  slime_task_target=$slime_task_hard
fi
if [[ "$slime_task_soft" != unlimited && "$slime_task_soft" -lt "$slime_task_target" ]]; then
  ulimit -Su "$slime_task_target"
fi
exec "$@"
