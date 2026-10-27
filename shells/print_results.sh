#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
# Discover all saved runs by default; active model/data selections do not hide results.
exec "$DATA_PYTHON" -m synthetic_joint_training.results --output-root "$OUTPUT_ROOT" "$@"
