#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON="${PYTHON:-${REPO_ROOT}/R2E-Gym/.venv/bin/python}"
FULL_INPUT="${INPUT:-${HOME}/swe-smith-teacher-deepseek-flash-2000x3/selected-tasks.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/leaf-sft/raw}"
TASK_COUNT="${TASK_COUNT:-2000}"
SEEDS="${SEEDS:-0,1,2}"
SEED_COUNT="$(awk -F, '{print NF}' <<<"${SEEDS}")"
RUN_NAME="${RUN_NAME:-swesmith-first${TASK_COUNT}x${SEED_COUNT}-leaf-teacher-offline-v2}"
OUTPUT="${OUTPUT:-${OUTPUT_ROOT}/${RUN_NAME}.jsonl}"
LOG="${LOG:-${OUTPUT_ROOT}/${RUN_NAME}.log}"
LOCK="${LOCK:-${OUTPUT}.lock}"
ANTIHACK="${ANTIHACK:-1}"
ANTIHACK_EVENTS="${ANTIHACK_EVENTS:-${OUTPUT%.jsonl}.antihack.jsonl}"
ANTIHACK_RUN_ID="${ANTIHACK_RUN_ID:-${RUN_NAME}}"

WORKERS="${WORKERS:-12}"
TEMPERATURE="${TEMPERATURE:-1.0}"
LOG_TURNS="${LOG_TURNS:-1}"
PREPARED_IMAGE_MANIFEST="${PREPARED_IMAGE_MANIFEST:-${REPO_ROOT}/R2E-Gym/image_manifests/swesmith-first500-offline-v2.json}"
REQUIRE_PREPARED_IMAGES="${REQUIRE_PREPARED_IMAGES:-1}"
NETWORK_MODE="${NETWORK_MODE:-none}"
PREPULL_PREPARED_IMAGES="${PREPULL_PREPARED_IMAGES:-1}"

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

if [[ "${REQUIRE_PREPARED_IMAGES}" == "1" ]]; then
  [[ "${NETWORK_MODE}" == "none" ]] || { echo "Prepared images require NETWORK_MODE=none" >&2; exit 1; }
  [[ -f "${PREPARED_IMAGE_MANIFEST}" ]] || { echo "Prepared image manifest missing: ${PREPARED_IMAGE_MANIFEST}" >&2; exit 1; }
  "${PYTHON}" - "${PREPARED_IMAGE_MANIFEST}" "${FULL_INPUT}" "${TASK_COUNT}" "${PREPULL_PREPARED_IMAGES}" <<'PY'
import json
import re
import sys
from pathlib import Path

import docker

manifest = json.loads(Path(sys.argv[1]).read_text())
if manifest.get("schema_version") != "swesmith-offline-image-v1":
    raise SystemExit("unsupported prepared image manifest schema")
rows = []
with Path(sys.argv[2]).open() as stream:
    for line in stream:
        if line.strip():
            row = json.loads(line)
            value = row.get("extra_info", row)
            rows.append(json.loads(value) if isinstance(value, str) else value)
        if len(rows) >= int(sys.argv[3]):
            break
references = {}
for row in rows:
    source = row.get("docker_image") or row.get("image_name")
    record = manifest.get("images", {}).get(source)
    if not record or row.get("instance_id") not in record.get("instance_ids", []):
        raise SystemExit(f"prepared image coverage missing: {row.get('instance_id')} {source}")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(record.get("prepared_digest", ""))):
        raise SystemExit(f"invalid prepared digest: {source}")
    verify = record.get("offline_verification", {})
    if not verify.get("passed") or not verify.get("network_blocked"):
        raise SystemExit(f"image is not offline verified: {source}")
    references[record["prepared_tag_reference"]] = record["prepared_digest"]
registry_host = manifest["registry"].split("/", 1)[0]
docker_config = json.loads((Path.home() / ".docker/config.json").read_text())
if not (
    registry_host in docker_config.get("auths", {})
    or registry_host in docker_config.get("credHelpers", {})
    or docker_config.get("credsStore")
):
    raise SystemExit(f"Docker is not logged in to prepared registry: {registry_host}")
if sys.argv[4] == "1":
    client = docker.from_env(timeout=900)
    try:
        for tag, digest in sorted(references.items()):
            reference = f"{tag}@{digest}"
            try:
                client.images.get(reference)
            except docker.errors.ImageNotFound:
                client.images.pull(tag)
                client.images.get(reference)
    finally:
        client.close()
print(f"Prepared-image preflight passed for {len(rows)} tasks")
PY
fi

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
target=$((TASK_COUNT * SEED_COUNT))
echo "Resuming Leaf teacher rollout: records_present=${completed}, current_target=${target}"
echo "Tasks: ${TASK_COUNT}; seeds: ${SEEDS}; input: ${INPUT}"
echo "Output: ${OUTPUT}"
echo "Log: ${LOG}"
echo "Anti-hack: ${ANTIHACK}; events: ${ANTIHACK_EVENTS}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/R2E-Gym/src${PYTHONPATH:+:${PYTHONPATH}}"

PROGRESS_ARGS=()
if [[ "${LOG_TURNS}" == "1" ]]; then
  PROGRESS_ARGS+=(--log-turns)
fi
if [[ "${ANTIHACK}" == "1" ]]; then
  PROGRESS_ARGS+=(--antihack --antihack-events "${ANTIHACK_EVENTS}" --antihack-run-id "${ANTIHACK_RUN_ID}")
fi
if [[ "${REQUIRE_PREPARED_IMAGES}" == "1" ]]; then
  PROGRESS_ARGS+=(--prepared-image-manifest "${PREPARED_IMAGE_MANIFEST}" --require-prepared-images)
fi

"${PYTHON}" -u R2E-Gym/scripts/collect_leaf_teacher_rollouts.py \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --seeds "${SEEDS}" \
  --workers "${WORKERS}" \
  --temperature "${TEMPERATURE}" \
  --network-mode "${NETWORK_MODE}" \
  "${PROGRESS_ARGS[@]}" \
  2>&1 | tee -a "${LOG}"
