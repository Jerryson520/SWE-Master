#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R2E_GYM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${R2E_GYM_DIR}"


export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:18000/v1}"

export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
export no_proxy="${no_proxy:-${NO_PROXY}}"


TASK_INDICES=(23 27 28 39 46)


# 每次执行脚本使用新的实验名，避免覆盖或混入旧 JSONL。
RUN_ID="$(date +%Y%m%d-%H%M%S)"
TRAJ_DIR="./results/context-retry-5"
EXP_NAME="swe-master-4b-context-retry-5-${RUN_ID}"

echo "Experiment: ${EXP_NAME}"
echo "Output: ${TRAJ_DIR}/${EXP_NAME}.jsonl"
echo "Indices: ${TASK_INDICES[*]}"
echo

for index in "${TASK_INDICES[@]}"; do
    echo "============================================================"
    echo "Running dataset index: ${index}"
    echo "============================================================"

    ./.venv/bin/python \
        src/r2egym/agenthub/run/edit.py \
        runagent_multiple \
        --traj_dir "${TRAJ_DIR}" \
        --max_workers 1 \
        --start_idx "${index}" \
        --k 1 \
        --dataset "R2E-Gym/SWE-Bench-Verified" \
        --split "test" \
        --llm_name "hosted_vllm/swe-master-sft" \
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
        --used_yaml \
          "./src/r2egym/agenthub/config/openhands/openhands_sp_non_fn_calling.yaml" \
        --enable_compression False

    echo "Finished dataset index: ${index}"
    echo
done

RESULT_FILE="${TRAJ_DIR}/${EXP_NAME}.jsonl"

echo "All retry tasks finished."
echo "Result file: ${RESULT_FILE}"
wc -l "${RESULT_FILE}"
