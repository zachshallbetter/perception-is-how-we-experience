#!/usr/bin/env bash
set -euo pipefail

HOST="${TITAN_HOST:-titan}"
REMOTE_ROOT="${TITAN_REMOTE_ROOT:-/ai/projects/Perception}"
REMOTE_PYTHON="${TITAN_PYTHON:-/ai/envs/perception/bin/python}"
WATCH_INTERVAL=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --watch)
      WATCH_INTERVAL="${2:-60}"
      shift 2
      ;;
    *)
      echo "Usage: titan_remote_dashboard.sh [--watch <seconds>]" >&2
      exit 1
      ;;
  esac
done

render_once() {
  ssh "${HOST}" "cd '${REMOTE_ROOT}' && '${REMOTE_PYTHON}' scripts/render_titan_dashboard.py --results-root results --log-dir results/titan_logs --md-out results/titan_dashboard.md --json-out results/titan_dashboard.json"
}

if [ -n "${WATCH_INTERVAL}" ]; then
  while true; do
    clear
    render_once
    sleep "${WATCH_INTERVAL}"
  done
else
  render_once
fi
