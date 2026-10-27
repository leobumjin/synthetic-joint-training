#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
[[ -z "$BASE_LORA_REPO" ]] || RUN_ARGS+=(--base-lora-repo "$BASE_LORA_REPO" --base-lora-revision "$BASE_LORA_REVISION")
(( ${#SYNTHETIC[@]} == 0 )) || RUN_ARGS+=(--synthetic "${SYNTHETIC[@]}")
exec "$DATA_PYTHON" -m synthetic_joint_training.cli synthetic "${RUN_ARGS[@]}" "$@"
