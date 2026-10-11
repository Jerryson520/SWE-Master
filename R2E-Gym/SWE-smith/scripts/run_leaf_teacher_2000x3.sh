#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON="${PYTHON:-${REPO_ROOT}/R2E-Gym/.venv/bin/python}"
FULL_INPUT="${INPUT:-${HOME}/swe-smith-teacher-deepseek-flash-2000x3/selected-tasks.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/leaf-sft/raw}"
OUTPUT="${OUTPUT:-${OUTPUT_ROOT}/swesmith-2000x3.jsonl}"
LOG="${LOG:-${OUTPUT_ROOT}/swesmith-2000x3.log}"
LOCK="${LOCK:-${OUTPUT}.lock}"

TASK_COUNT="${TASK_COUNT:-2000}"
SEEDS="${SEEDS:-0,1,2}"
WORKERS="${WORKERS:-12}"
TEMPERATURE="${TEMPERATURE:-1.0}"

export LEAF_TEACHER_BASE_URL="${LEAF_TEACHER_BASE_URL:-https://api.deepseek.com/v1}"
export LEAF_TEACHER_MODEL="${LEAF_TEACHER_MODEL:-deepseek-flash}"

[[ -x "${PYTHON}" ]] || { echo "Python environment missing: ${PYTHON}" >&2; exit 1; }
[[ -f "${FULL_INPUT}" ]] || { echo "SWE-smith task list missing: ${FULL_INPUT}" >&2; exit 1; }
[[ "${TASK_COUNT}" =~ ^[0-9]+$ ]] && (( TASK_COUNT >= 1 && TASK_COUNT <= 2000 )) || {
  echo "TASK_COUNT must be an integer from 1 to 2000: ${TASK_COUNT}" >&2
  exit 1
}
docker info >/dev/null
mkdir -p "${OUTPUT_ROOT}"

host_cpus="$(nproc)"
if (( WORKERS > host_cpus )); then
  echo "Warning: WORKERS=${WORKERS} exceeds the current host CPU count (${host_cpus})." >&2
fi

INPUT="${FULL_INPUT}"
if (( TASK_COUNT < 2000 )); then
  subset_dir="${OUTPUT_ROOT}/task-subsets"
  INPUT="${subset_dir}/swesmith-first-${TASK_COUNT}.jsonl"
  mkdir -p "${subset_dir}"
  "${PYTHON}" - "${FULL_INPUT}" "${INPUT}" "${TASK_COUNT}" <<'PY'
import os
import sys
from itertools import islice
from pathlib import Path

source, output, count = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
with source.open(encoding="utf-8") as stream:
    lines = list(islice((line.rstrip("\n") for line in stream if line.strip()), count))
if len(lines) < count:
    raise SystemExit(f"task list has only {len(lines)} records; requested {count}")
pending = output.with_suffix(output.suffix + ".tmp")
pending.write_text("\n".join(lines[:count]) + "\n", encoding="utf-8")
os.replace(pending, output)
print(f"Prepared deterministic {count}-task prefix: {output}")
PY
fi

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
seed_count="$(awk -F, '{print NF}' <<<"${SEEDS}")"
target=$((TASK_COUNT * seed_count))
echo "Resuming Leaf teacher rollout: records_present=${completed}, current_target=${target}"
echo "Tasks: ${TASK_COUNT}; seeds: ${SEEDS}; input: ${INPUT}"
echo "Output: ${OUTPUT}"
echo "Log: ${LOG}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/R2E-Gym/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" -u R2E-Gym/SWE-smith/scripts/collect_leaf_teacher_rollouts.py \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --seeds "${SEEDS}" \
  --workers "${WORKERS}" \
  --temperature "${TEMPERATURE}" \
  2>&1 | tee -a "${LOG}"
