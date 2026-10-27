#!/usr/bin/env bash
# Upload completed adapter parameters; spelling retained as requested.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
PUSH_ARGS=(--output-root "$OUTPUT_ROOT" --base-output-root "$BASE_OUTPUT_ROOT" --registry "$BASE_LORA_REGISTRY"
           --hf-repo "$HF_REPO" --hf-synthetic-repo "$HF_SYNTHETIC_REPO")
[[ -z "$HF_NAMESPACE" ]] || PUSH_ARGS+=(--hf-namespace "$HF_NAMESPACE")
exec "$DATA_PYTHON" -m synthetic_joint_training.push "${PUSH_ARGS[@]}" "$@"
