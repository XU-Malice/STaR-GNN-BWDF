#!/usr/bin/env bash
# Exec preserves the launcher's PID; Python owns and terminates each child group.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python -u scripts/train/run_que_protocol_audit.py --gpu-id "${GPU_ID:-6}" "$@"
