# Central experiment selection. CLI arguments override these defaults.
SJT_SHELL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SJT_SHELL_ROOT}/.." && pwd)"
MODELS=(
  "Qwen/Qwen3-4B-Instruct-2507"
  # "Qwen/Qwen3-14B"
  # "google/gemma-3-4b-it"
  # "google/gemma-3-12b-it"
  # "google/gemma-3-27b-it"
  # "meta-llama/Llama-3.1-8B-Instruct"
)
BENCHMARKS=(
  gsm8k
  # mmlu
  # folio
)
# name=/path/to/train.jsonl; each entry is a separate synthetic experiment.
SYNTHETIC=(
  # "edit=${PROJECT_ROOT}/data/input/edit.jsonl"
  # "rebuild=${PROJECT_ROOT}/data/input/rebuild.jsonl"
)
TRAIN_SEEDS=(0 1 2)
PROMPT_STYLES=(Direct CoT)
MODEL_REVISION="${MODEL_REVISION:-main}" # Resolved to a commit SHA for actual runs.
DATA_SEED="${DATA_SEED:-42}"
SPLIT_POLICY="${SPLIT_POLICY:-custom}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.25}"
MAX_TRAIN="${MAX_TRAIN:-1000}"
MAX_TEST="${MAX_TEST:-3000}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
BASE_LORA_REGISTRY="${BASE_LORA_REGISTRY:-${PROJECT_ROOT}/configs/base_lora_registry.json}"
MMLU_TRAIN_SAMPLES="${MMLU_TRAIN_SAMPLES:-3000}"
SJT_VENV="${SJT_VENV:-${PROJECT_ROOT}/.venv}"
DATA_PYTHON="${DATA_PYTHON:-python3}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${SJT_VENV}/bin/python}"
INFERENCE_PYTHON="${INFERENCE_PYTHON:-${SJT_VENV}/bin/python}"
EPOCHS="${EPOCHS:-1}"
SYNTHETIC_EPOCHS="${SYNTHETIC_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-16}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
# Both ranks use alpha/rank=2 unless LORA_ALPHA is explicitly overridden.
LORA_RANKS=(16 32)
LORA_ALPHA="${LORA_ALPHA:-}"
LORA_DROPOUT="${LORA_DROPOUT:-0}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"
DTYPE="${DTYPE:-bfloat16}"
WARMUP_RATIO="${WARMUP_RATIO:-0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
EVAL_SPLIT="${EVAL_SPLIT:-test}" # Fixed custom heldout split shared by every condition.
DIRECT_MAX_TOKENS="${DIRECT_MAX_TOKENS:-256}"
COT_MAX_TOKENS="${COT_MAX_TOKENS:-2048}"
EVAL_SEED="${EVAL_SEED:-42}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
# HF username or organization used for model repositories; no account is assumed.
HF_NAMESPACE="${HF_NAMESPACE:-}"

# Shared HF repositories; checkpoints are stored in per-run subfolders.
HF_REPO="${HF_REPO:-synthetic-joint-training-base-loras}"
HF_SYNTHETIC_REPO="${HF_SYNTHETIC_REPO:-synthetic-joint-training-synthetic-loras}"
# Optional: namespace/repo to fetch the public base registry before synthetic runs.
BASE_LORA_REPO="${BASE_LORA_REPO:-}"
BASE_LORA_REVISION="${BASE_LORA_REVISION:-main}"
