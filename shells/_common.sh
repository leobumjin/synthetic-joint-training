#!/usr/bin/env bash
# Internal shared arguments; shared by the user-facing scripts.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
RUN_ARGS=(
  --models "${MODELS[@]}" --benchmarks "${BENCHMARKS[@]}" --seeds "${TRAIN_SEEDS[@]}"
  --revision "$MODEL_REVISION" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --base-output-root "$BASE_OUTPUT_ROOT"
  --registry "$BASE_LORA_REGISTRY" --hf-repo "$HF_REPO" --train-python "$TRAIN_PYTHON" --inference-python "$INFERENCE_PYTHON"
  --epochs "$EPOCHS" --learning-rate "$LEARNING_RATE"
  --batch-size "$BATCH_SIZE" --gradient-accumulation "$GRADIENT_ACCUMULATION" --max-length "$MAX_LENGTH"
  --lora-ranks "${LORA_RANKS[@]}" --lora-dropout "$LORA_DROPOUT"
  --target-modules "$TARGET_MODULES" --dtype "$DTYPE" --warmup-ratio "$WARMUP_RATIO" --weight-decay "$WEIGHT_DECAY"
  --eval-split "$EVAL_SPLIT" --prompt-styles "${PROMPT_STYLES[@]}"
  --direct-max-tokens "$DIRECT_MAX_TOKENS" --cot-max-tokens "$COT_MAX_TOKENS" --eval-seed "$EVAL_SEED" --eval-batch-size "$EVAL_BATCH_SIZE"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
)
[[ -z "$HF_NAMESPACE" ]] || RUN_ARGS+=(--hf-namespace "$HF_NAMESPACE")

[[ -z "${LORA_RANK:-}" ]] || RUN_ARGS+=(--lora-rank "$LORA_RANK")
[[ -z "$LORA_ALPHA" ]] || RUN_ARGS+=(--lora-alpha "$LORA_ALPHA")

[[ -z "$JOINT_EPOCHS" ]] || RUN_ARGS+=(--joint-epochs "$JOINT_EPOCHS")
