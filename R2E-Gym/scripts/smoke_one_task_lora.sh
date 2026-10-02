#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:18000/v1}"
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"

START_IDX="${START_IDX:-0}"
TASK_COUNT="${TASK_COUNT:-1}"
EXP_NAME="${EXP_NAME:-astropy-paper-128k-150step-lora}"
TRAJ_DIR="${TRAJ_DIR:-./results/astropy-paper-128k-lora}"
MODEL_NAME="${MODEL_NAME:-hosted_vllm/swe-master-smoke}"

exec uv run python src/r2egym/agenthub/run/edit.py runagent_multiple \
  --traj_dir "$TRAJ_DIR" \
  --max_workers 1 \
  --start_idx "$START_IDX" \
  --k "$TASK_COUNT" \
  --dataset "R2E-Gym/SWE-Bench-Verified" \
  --split "test" \
  --llm_name "$MODEL_NAME" \
  --use_fn_calling False \
  --exp_name "$EXP_NAME" \
  --temperature 0.7 \
  --max_steps 150 \
  --max_steps_absolute 150 \
  --context_window 131072 \
  --max_output_tokens 16384 \
  --context_safety_margin 0 \
  --max_trajectory_output_tokens 1000000 \
  --backend "docker" \
  --prepull_images True \
  --prepull_workers 4 \
  --scaffold "openhands" \
  --used_yaml "./src/r2egym/agenthub/config/openhands/openhands_sp_non_fn_calling.yaml" \
  --enable_compression False
