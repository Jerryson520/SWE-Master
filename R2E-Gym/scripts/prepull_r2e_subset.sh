#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &&
  pwd
)"

R2E_GYM_DIR="$(
  cd -- "${SCRIPT_DIR}/.." &&
  pwd
)"

if [[ -x "${R2E_GYM_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${R2E_GYM_DIR}/.venv/bin/python"
else
  PYTHON_BIN="python"
fi

cd "${R2E_GYM_DIR}"

echo "R2E-Gym directory: ${R2E_GYM_DIR}"
echo "Python: ${PYTHON_BIN}"

exec "${PYTHON_BIN}" \
  "${SCRIPT_DIR}/prepull_r2e_subset.py" \
  "$@"
