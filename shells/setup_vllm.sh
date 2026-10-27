#!/usr/bin/env bash
# Pinned shared Python 3.11/3.12 environment for SFT and vLLM evaluation.
set -euo pipefail
SJT_SETUP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SJT_VENV="${SJT_VENV:-${SJT_SETUP_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CHECK=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --check) CHECK=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) echo 'Usage: setup_vllm.sh [--check] [--dry-run]'; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done
if (( DRY_RUN )); then
  printf 'Create/reuse Python environment: %s\nInstall editable project extras: data,train,vllm,dev\n' "$SJT_VENV"
  exit 0
fi
if (( ! CHECK )); then
  "$PYTHON_BIN" -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,13), "Use Python 3.11 or 3.12"'
  if [[ ! -e "$SJT_VENV" ]]; then
    "$PYTHON_BIN" -m venv "$SJT_VENV"
  fi
  "$SJT_VENV/bin/python" -m pip install --upgrade pip
  "$SJT_VENV/bin/python" -m pip install -e "${SJT_SETUP_ROOT}[data,train,vllm,dev]"
fi
"$SJT_VENV/bin/python" -m pip check
"$SJT_VENV/bin/python" - <<'PY'
import importlib.metadata as m
import torch, transformers, peft, vllm
for name in ('torch', 'transformers', 'peft', 'vllm', 'datasets', 'accelerate'):
    print(name, m.version(name))
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable; environment installed but GPU execution is not ready')
print('CUDA', torch.version.cuda, '| devices', torch.cuda.device_count())
PY
