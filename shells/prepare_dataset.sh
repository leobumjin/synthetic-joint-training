#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
PREPARE_ARGS=(--data-root "$DATA_ROOT" --benchmarks "${BENCHMARKS[@]}" --data-seed "$DATA_SEED"
              --mmlu-train-samples "$MMLU_TRAIN_SAMPLES" --split-policy "$SPLIT_POLICY"
              --train-fraction "$TRAIN_FRACTION" --max-train "$MAX_TRAIN" --max-test "$MAX_TEST")
SYNTHETIC_OVERRIDE=0
for argument in "$@"; do
  case "$argument" in
    --synthetic|--synthetic=*|--synthetic-file|--synthetic-file=*) SYNTHETIC_OVERRIDE=1 ;;
  esac
done
if (( SYNTHETIC_OVERRIDE )); then
  : # Explicit CLI selection takes precedence over the configured array.
elif (( ${#SYNTHETIC[@]} )); then
  PREPARE_ARGS+=(--synthetic "${SYNTHETIC[@]}")
elif [[ -n "${SYNTHETIC_FILE:-}" ]]; then
  PREPARE_ARGS+=(--synthetic-file "$SYNTHETIC_FILE")
fi
# Dataset download requires the data extra; offline normalized input is stdlib-only.
exec "$DATA_PYTHON" -m synthetic_joint_training.data "${PREPARE_ARGS[@]}" "$@"
