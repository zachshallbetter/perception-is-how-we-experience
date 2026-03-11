#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATUS_FILE="${STATUS_FILE:-${ROOT_DIR}/paper/figures/generated/results_package/distillation_campaign_status.json}"
DISTILL_LAUNCHER="${DISTILL_LAUNCHER:-${ROOT_DIR}/scripts/launch_distill_confirmation_campaign.sh}"
CONTROL_LAUNCHER="${CONTROL_LAUNCHER:-${ROOT_DIR}/scripts/launch_tuned_core_expansion_campaign.sh}"
MIN_DISTILL_RUNS="${MIN_DISTILL_RUNS:-4}"
MIN_CONTROL_RUNS="${MIN_CONTROL_RUNS:-2}"
APPLY=0

usage() {
  cat <<EOF
Usage: decide_distill_followup.sh [--apply]

Reads ${STATUS_FILE} and chooses the next campaign once the distillation replay comparison
has enough completed runs to evaluate.

Default thresholds:
  MIN_DISTILL_RUNS=${MIN_DISTILL_RUNS}
  MIN_CONTROL_RUNS=${MIN_CONTROL_RUNS}

Environment overrides:
  STATUS_FILE=/path/to/distillation_campaign_status.json
  DISTILL_LAUNCHER=/path/to/launch_distill_confirmation_campaign.sh
  CONTROL_LAUNCHER=/path/to/launch_tuned_core_expansion_campaign.sh
  MIN_DISTILL_RUNS=4
  MIN_CONTROL_RUNS=2

Behavior:
  - If the required run counts are not complete, print a pending decision and exit 2.
  - If mean distillation BWT is better (higher) than mean control BWT, choose the
    distillation confirmation launcher.
  - Otherwise choose the tuned-core expansion launcher.
  - With --apply, execute the chosen launcher from the repo root.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --apply)
      APPLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ ! -f "${STATUS_FILE}" ]; then
  echo "Status file not found: ${STATUS_FILE}" >&2
  exit 1
fi

decision_payload="$(
  python3 - "${STATUS_FILE}" "${MIN_DISTILL_RUNS}" "${MIN_CONTROL_RUNS}" <<'PY'
import json
import statistics
import sys

status_path = sys.argv[1]
min_distill = int(sys.argv[2])
min_control = int(sys.argv[3])

with open(status_path, "r", encoding="utf-8") as fh:
    data = json.load(fh)

distill = data.get("completed_distill_runs", [])
control = data.get("completed_control_runs", [])

if len(distill) < min_distill or len(control) < min_control:
    print("state=pending")
    print(f"status={data.get('status', 'unknown')}")
    print(f"distill_count={len(distill)}")
    print(f"control_count={len(control)}")
    print(f"distill_required={min_distill}")
    print(f"control_required={min_control}")
    raise SystemExit(0)

distill_mean_bwt = statistics.mean(float(row["BWT"]) for row in distill)
control_mean_bwt = statistics.mean(float(row["BWT"]) for row in control)

decision = "distill-confirm" if distill_mean_bwt > control_mean_bwt else "tuned-core-expand"

print("state=ready")
print(f"status={data.get('status', 'unknown')}")
print(f"distill_count={len(distill)}")
print(f"control_count={len(control)}")
print(f"distill_mean_bwt={distill_mean_bwt:.6f}")
print(f"control_mean_bwt={control_mean_bwt:.6f}")
print(f"decision={decision}")
PY
)"

eval "${decision_payload}"

if [ "${state}" = "pending" ]; then
  echo "Decision pending."
  echo "Package status: ${status}"
  echo "Completed distill runs: ${distill_count}/${distill_required}"
  echo "Completed control runs: ${control_count}/${control_required}"
  exit 2
fi

if [ "${decision}" = "distill-confirm" ]; then
  chosen_launcher="${DISTILL_LAUNCHER}"
  decision_label="distillation confirmation"
else
  chosen_launcher="${CONTROL_LAUNCHER}"
  decision_label="tuned-core expansion"
fi

echo "Decision ready."
echo "Package status: ${status}"
echo "Distill mean BWT: ${distill_mean_bwt}"
echo "Control mean BWT: ${control_mean_bwt}"
echo "Chosen next step: ${decision_label}"
echo "Launcher: ${chosen_launcher}"

if [ "${APPLY}" -eq 1 ]; then
  if [ ! -x "${chosen_launcher}" ] && [ ! -f "${chosen_launcher}" ]; then
    echo "Chosen launcher not found: ${chosen_launcher}" >&2
    exit 1
  fi
  echo "Applying decision now..."
  (
    cd "${ROOT_DIR}"
    bash "${chosen_launcher}"
  )
fi
