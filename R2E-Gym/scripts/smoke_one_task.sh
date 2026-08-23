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
EXP_NAME="${EXP_NAME:-swe-master-4b-smoke}"

exec uv run python src/r2egym/agenthub/run/edit.py runagent_multiple \
  --traj_dir "./results/one-task" \
  --max_workers 1 \
  --start_idx "$START_IDX" \
  --k "$TASK_COUNT" \
  --dataset "R2E-Gym/SWE-Bench-Verified" \
  --split "test" \
  --llm_name "hosted_vllm/swe-master-sft" \
  --use_fn_calling False \
  --exp_name "$EXP_NAME" \
  --temperature 0 \
  --max_steps 40 \
  --max_steps_absolute 40 \
  --context_window 32768 \
  --max_output_tokens 2048 \
  --context_safety_margin 1024 \
  --max_trajectory_output_tokens 32768 \
  --backend "docker" \
  --prepull_images True \
  --prepull_workers 4 \
  --scaffold "openhands" \
  --used_yaml "./src/r2egym/agenthub/config/openhands/openhands_sp_non_fn_calling.yaml"
