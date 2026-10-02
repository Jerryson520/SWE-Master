#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

export OPENAI_API_KEY=EMPTY
export OPENAI_API_BASE=http://127.0.0.1:18000/v1
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy="$NO_PROXY"

TASK_COUNT="${1:-1}"

exec ./.venv/bin/python src/r2egym/agenthub/run/edit.py runagent_multiple \
  --traj_dir "./results/qwen3-base-sweb50-80k" \
  --max_workers 4 \
  --start_idx 1 \
  --k "${TASK_COUNT}" \
  --dataset "R2E-Gym/SWE-Bench-Verified" \
  --split "test" \
  --llm_name "hosted_vllm/qwen3-4b-base" \
  --use_fn_calling False \
  --exp_name "qwen3-4b-base-sweb50-80k" \
  --temperature 0.7 \
  --max_steps 150 \
  --max_steps_absolute 150 \
  --context_window 81920 \
  --max_output_tokens 8192 \
  --context_safety_margin 8192 \
  --max_trajectory_output_tokens 1000000 \
  --backend "docker" \
  --prepull_images True \
  --prepull_workers 4 \
  --scaffold "openhands" \
  --used_yaml "./src/r2egym/agenthub/config/openhands/openhands_sp_non_fn_calling.yaml" \
  --enable_compression False
