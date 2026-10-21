#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load untracked local settings while preserving per-command overrides.
settings="${DMC_BLIND_CONFIG:-${REPO_ROOT}/local_data/dmc_blind.env}"
if [[ -f "${settings}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${settings}"
  set +a
elif [[ -n "${DMC_BLIND_CONFIG:-}" ]]; then
  echo "DMC_BLIND_CONFIG does not exist: ${settings}" >&2
  exit 2
fi

source "${REPO_ROOT}/experiments/_runtime_defaults.sh"

require_fixed_selection() {
  local name="$1" expected="$2" actual
  actual="${!name:-}"
  if [[ -n "${actual}" && "${actual,,}" != "${expected}" ]]; then
    echo "${name} is fixed to ${expected} by the blind launcher; got ${actual}." >&2
    exit 2
  fi
  export "${name}=${expected}"
}
require_fixed_selection ALGORITHM p2e
require_fixed_selection STRICT_TASK_BLIND_PRETRAINING true
require_fixed_selection DMC_TASK_SET control_suite
export DMC_LAUNCHER_DISPLAY=launch_dmc_blind.sh
# Select a GPU type with GRES if required by your cluster.
export GRES="${GRES:-gpu:1}"
# Set GPU allocation policy before MJX initializes.
export JAX_PREALLOCATE="${JAX_PREALLOCATE:-false}"

check_setup() {
  bash "${SCRIPT_DIR}/launch_dmc_p2e.sh" --list >/dev/null
  # Check the same explicitly selected or active environment as the worker.
  source "${SCRIPT_DIR}/_activate_runtime.sh"
  for required in sbatch git rsync flock; do
    command -v "${required}" >/dev/null || {
      echo "Missing ${required}; run this launcher on a configured Slurm login node." >&2
      exit 1
    }
  done
  python3 "${SCRIPT_DIR}/check_dmc_runtime.py"
}

case "${1:-}" in
  --check) check_setup; exit ;;
  --dry-run) export DRY_RUN=true; shift ;;
  --help|-h)
    cat <<EOF
Usage: bash experiments/launch_dmc_blind.sh [--check|--dry-run|--list|--help]

With no option, check setup and submit 25 DMC tasks using the saved settings.
  --check    Check runtime dependencies and W&B configuration; submit nothing.
  --dry-run  Preview submission commands without checking the training runtime.
  --list     Show task indices for ARRAY_TASKS selection.

Local settings: ${settings}
Setup and logging instructions: README.md

Examples:
  IDM_CONFIG=simple_curiosity bash experiments/launch_dmc_blind.sh
  IDM_CONFIG=rnd SEEDS=0:1:2 bash experiments/launch_dmc_blind.sh --dry-run
  ARRAY_TASKS=24 bash experiments/launch_dmc_blind.sh

Defaults: vic, seed 0, dmc_proprio, dreamerv3, size12M, offline W&B logging.
GPU default: GRES=gpu:1; set GRES to select a specific GPU type.
Allocation default: JAX_PREALLOCATE=false; set true for a controlled comparison.
Other selectors: ENV_CONFIG, BACKEND_AGENT, MODEL_CONFIG, WANDB_PROJECT,
MEMORY, GRES, PARTITION, QOS, MAX_CONCURRENT_GPUS.
This launcher fixes ALGORITHM=p2e and STRICT_TASK_BLIND_PRETRAINING=true.
EOF
    exit 0
    ;;
  "")
    if [[ "${DRY_RUN:-false}" != "true" ]]; then
      check_setup
    fi
    ;;
esac

exec bash "${SCRIPT_DIR}/launch_dmc_p2e.sh" "$@"
