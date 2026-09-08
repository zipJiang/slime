#!/usr/bin/env bash
# Idempotent uv overlay. CUDA, torch and compiled kernels stay in the immutable SIF.
set -euo pipefail
slime_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
slime_image=${SLIME_SIF:-/weka/projects/bvandur1/zjiang31/slime/images/slime-latest.sif}
slime_expected=$(cut -d ' ' -f 1 "$slime_root/environment/sif-image.sha256")
slime_actual=$(sha256sum "$slime_image")
slime_actual=${slime_actual%% *}
[[ "$slime_actual" == "$slime_expected" ]] || {
  echo "SIF digest differs from environment/sif-image.sha256; select the pinned image." >&2
  exit 1
}
if [[ ! -d "$slime_root/.venv" ]]; then
  bash "$slime_root/scripts/sif.sh" /opt/slime-uv venv --python /usr/bin/python \
    --system-site-packages "$slime_root/.venv"
fi
bash "$slime_root/scripts/sif.sh" /opt/slime-uv pip install \
  --python "$slime_root/.venv/bin/python" --no-deps --no-build-isolation -e "$slime_root"
bash "$slime_root/scripts/sif.sh" python environment/check_env.py \
  --output "$slime_root/.venv/environment.json"
