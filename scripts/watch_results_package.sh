#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL="${1:-60}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/results/titan_logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/results-package-watcher.log}"
STATE_FILE="${STATE_FILE:-${LOG_DIR}/results-package-watcher.state}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BUILD_SCRIPT="${ROOT_DIR}/paper/figures/build_results_package.py"

mkdir -p "${LOG_DIR}"

if [ ! -f "${BUILD_SCRIPT}" ]; then
  echo "Build script not found: ${BUILD_SCRIPT}" >&2
  exit 1
fi

fingerprint_results() {
  (
    cd "${ROOT_DIR}"
    find results -maxdepth 2 -type f \
      \( -path "results/distill_t*_seed*/results.json" -o -path "results/tuned_er_seed*/results.json" \) \
      -print0 \
      | sort -z \
      | xargs -0 stat -f "%N %z %m" 2>/dev/null || true
  ) | shasum -a 256 | awk '{print $1}'
}

log_line() {
  echo "[$(date -Iseconds)] $*" | tee -a "${LOG_FILE}"
}

rebuild_package() {
  log_line "rebuild-start reason=$1"
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" "${BUILD_SCRIPT}"
  )
  log_line "rebuild-complete"
}

current_state="$(fingerprint_results)"
previous_state=""

if [ -f "${STATE_FILE}" ]; then
  previous_state="$(cat "${STATE_FILE}")"
fi

log_line "watcher-start interval=${INTERVAL}s state=${current_state}"

if [ "${current_state}" != "${previous_state}" ]; then
  rebuild_package "initial-change-detected"
  printf '%s\n' "${current_state}" > "${STATE_FILE}"
fi

while true; do
  sleep "${INTERVAL}"
  current_state="$(fingerprint_results)"
  previous_state="$(cat "${STATE_FILE}" 2>/dev/null || true)"

  if [ "${current_state}" = "${previous_state}" ]; then
    log_line "no-change"
    continue
  fi

  rebuild_package "results-fingerprint-changed"
  printf '%s\n' "${current_state}" > "${STATE_FILE}"
done
