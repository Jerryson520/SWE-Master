#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R2E_GYM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${R2E_GYM_DIR}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:18000/v1}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
export no_proxy="${no_proxy:-${NO_PROXY}}"

START_IDX="${START_IDX:-1}"
TASK_COUNT="${TASK_COUNT:-50}"
MAX_WORKERS="${MAX_WORKERS:-1}"
EXP_NAME="${EXP_NAME:-swe-master-4b-rl-50tasks-128k-150step}"
TRAJ_DIR="${TRAJ_DIR:-./results/50-tasks-rl}"
MODEL_NAME="${MODEL_NAME:-hosted_vllm/swe-master-rl}"

echo "Starting SWE-Master evaluation"
echo "tasks=${TASK_COUNT} start_idx=${START_IDX} max_workers=${MAX_WORKERS}"
echo "experiment=${EXP_NAME} trajectory_dir=${TRAJ_DIR}"
echo "context_window=131072 max_output_tokens=16384 max_steps=150 temperature=0.7"

exec ./.venv/bin/python src/r2egym/agenthub/run/edit.py runagent_multiple \
  --traj_dir "${TRAJ_DIR}" \
  --max_workers "${MAX_WORKERS}" \
  --start_idx "${START_IDX}" \
  --k "${TASK_COUNT}" \
  --dataset "R2E-Gym/SWE-Bench-Verified" \
  --split "test" \
  --llm_name "${MODEL_NAME}" \
  --use_fn_calling False \
  --exp_name "${EXP_NAME}" \
  --temperature 0.7 \
  --max_steps 150 \
  --max_steps_absolute 150 \
  --context_window 131072 \
  --max_output_tokens 16384 \
  --context_safety_margin 4096 \
  --max_trajectory_output_tokens 1000000 \
  --backend "docker" \
  --prepull_images True \
  --prepull_workers 4 \
  --scaffold "openhands" \
  --used_yaml "./src/r2egym/agenthub/config/openhands/openhands_sp_non_fn_calling.yaml" \
  --enable_compression False
