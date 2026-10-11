#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON="${PYTHON:-${REPO_ROOT}/R2E-Gym/.venv/bin/python}"
INPUT="${INPUT:-${HOME}/swe-smith-teacher-deepseek-flash-2000x3/selected-tasks.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/leaf-sft/raw}"
OUTPUT="${OUTPUT:-${OUTPUT_ROOT}/swesmith-2000x3.jsonl}"
LOG="${LOG:-${OUTPUT_ROOT}/swesmith-2000x3.log}"
LOCK="${LOCK:-${OUTPUT}.lock}"

SEEDS="${SEEDS:-0,1,2}"
WORKERS="${WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-150}"
MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN:-8192}"
STEP_TIMEOUT="${STEP_TIMEOUT:-180}"
REWARD_TIMEOUT="${REWARD_TIMEOUT:-1800}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-7200}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"

export LEAF_TEACHER_BASE_URL="${LEAF_TEACHER_BASE_URL:-https://api.deepseek.com/v1}"
export LEAF_TEACHER_MODEL="${LEAF_TEACHER_MODEL:-deepseek-flash}"

[[ -x "${PYTHON}" ]] || { echo "Python environment missing: ${PYTHON}" >&2; exit 1; }
[[ -f "${INPUT}" ]] || { echo "SWE-smith task list missing: ${INPUT}" >&2; exit 1; }
docker info >/dev/null
mkdir -p "${OUTPUT_ROOT}"

if [[ -z "${LEAF_TEACHER_API_KEY:-}" ]]; then
  read -rsp 'DeepSeek API key: ' LEAF_TEACHER_API_KEY
  echo
  export LEAF_TEACHER_API_KEY
fi

# Keep one writer per output file. The Python collector itself resumes by
# skipping every completed (instance_id, seed) pair already present in OUTPUT.
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "Another Leaf rollout process already owns ${LOCK}" >&2
  exit 1
fi

completed=0
if [[ -f "${OUTPUT}" ]]; then
  completed="$(wc -l < "${OUTPUT}")"
fi
echo "Resuming Leaf teacher rollout: completed=${completed}, target=6000"
echo "Output: ${OUTPUT}"
echo "Log: ${LOG}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/R2E-Gym/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" -u R2E-Gym/SWE-smith/scripts/collect_leaf_teacher_rollouts.py \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --seeds "${SEEDS}" \
  --workers "${WORKERS}" \
  --max-steps "${MAX_STEPS}" \
  --max-tokens-per-turn "${MAX_TOKENS_PER_TURN}" \
  --step-timeout "${STEP_TIMEOUT}" \
  --reward-timeout "${REWARD_TIMEOUT}" \
  --request-timeout "${REQUEST_TIMEOUT}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  2>&1 | tee -a "${LOG}"
