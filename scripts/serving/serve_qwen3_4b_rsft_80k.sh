#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen3-4B-Instruct-2507}"
LORA_PATH="${LORA_PATH:-/root/autodl-tmp/sft_models/qwen3-4b-rsft-80k-lora}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"
VENV_PATH="${VENV_PATH:-/root/autodl-tmp/venvs/vllm-py312}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-base}"
LORA_NAME="${LORA_NAME:-qwen3-4b-rsft}"

[[ -d "${BASE_MODEL}" ]] || { echo "Base model does not exist: ${BASE_MODEL}" >&2; exit 1; }
[[ -d "${LORA_PATH}" ]] || { echo "LoRA adapter does not exist: ${LORA_PATH}" >&2; exit 1; }
source "${VENV_PATH}/bin/activate"

exec env CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$BASE_MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 81920 \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --enable-lora \
  --max-lora-rank 16 \
  --lora-modules "${LORA_NAME}=${LORA_PATH}"
