#!/usr/bin/env bash
set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
exec python -u scripts/train/run_que_comprehensive_reconstruction.py \
  --run-tag que_comprehensive_reconstruction_shared_20260908 \
  --gpu-id 7 --allow-shared-gpu \
  --shared-memory-limit-gib 6 --shared-headroom-gib 2 "$@"
