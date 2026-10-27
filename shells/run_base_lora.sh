#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
exec "$DATA_PYTHON" -m synthetic_joint_training.cli base "${RUN_ARGS[@]}" "$@"
