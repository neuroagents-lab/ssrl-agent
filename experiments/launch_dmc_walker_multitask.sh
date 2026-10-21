#!/usr/bin/env bash

# Concurrent detached-head Walker benchmark, with stand as the source task.
# Reuse the DMC submission/worker path for frozen source bundles and recovery.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SSRL_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DMC_LAUNCHER="${REPO_ROOT}/experiments/launch_dmc_p2e.sh"

usage() {
  cat <<'EOF'
Usage:
  bash experiments/launch_dmc_walker_multitask.sh --list
  bash experiments/launch_dmc_walker_multitask.sh --dry-run
  bash experiments/launch_dmc_walker_multitask.sh

Source: dmc_walker_stand. Evaluation: walker_stand, walker_walk, walker_run.
Dreamer uses its source-trained base policy for stand and detached target heads
for walk/run. P2E collects with intrinsic reward and evaluates three detached
target heads. All evaluations use greedy actions and the held-out reset bank.

Selections:
  ENV_CONFIG=dmc_walker_multitask|dmc_walker_multitask_vision
                                   # default: dmc_walker_multitask (proprio)
  ALGORITHM=all|dreamer|p2e         # default: all (Dreamer + the 23 blind IDMs)
  IDM_CONFIG=all|<configs/idm name> # default: all; used by P2E
  BACKEND_AGENT=dreamerv3|ne_dreamer|r2dreamer
                                   # default: dreamerv3; aliases: v3, ne, r2
  MODEL_CONFIG=size12M             # default; other existing sizes supported
  SEEDS=0:1:2                      # default: 0; SEED also supported
  ARRAY_TASKS=0:2                  # seed-list indices, applied to each method
  WANDB_PROJECT=dmc_walker_multitask

Both environment profiles use native dm_control, 1.01M environment steps,
and concurrent detached task heads. The vision profile uses 64x64 pixels only.
P2E requires strict task blindness. Training settings come from Hydra configs.
Slurm, runtime, and storage options are forwarded to launch_dmc_p2e.sh.
Each method gets its own seed array; --list and --dry-run never submit jobs.

Examples:
  SEEDS=1:2 bash experiments/launch_dmc_walker_multitask.sh
  SEEDS=1:2 ENV_CONFIG=dmc_walker_multitask_vision \
    bash experiments/launch_dmc_walker_multitask.sh
  ENV_CONFIG=dmc_walker_multitask_vision bash experiments/launch_dmc_walker_multitask.sh
  ALGORITHM=dreamer bash experiments/launch_dmc_walker_multitask.sh
  ALGORITHM=p2e IDM_CONFIG=simple_curiosity bash experiments/launch_dmc_walker_multitask.sh
  ALGORITHM=all IDM_CONFIG=disagreement SEEDS=0:1:2 \
    bash experiments/launch_dmc_walker_multitask.sh --dry-run
EOF
}

if (( $# > 1 )); then
  echo "Expected at most one argument; use --help for usage." >&2
  exit 2
fi
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  --dry-run) export DRY_RUN=true; shift ;;
  ""|--list|--worker) ;;
  *) echo "Unknown argument: ${1}" >&2; usage >&2; exit 2 ;;
esac

if [[ -n "${DMC_TASK_SET:-}" && "${DMC_TASK_SET}" != "walker_multitask" ]]; then
  echo "DMC_TASK_SET is fixed to walker_multitask; got ${DMC_TASK_SET}." >&2
  exit 2
fi
: "${ENV_CONFIG:=dmc_walker_multitask}"
if [[ "${ENV_CONFIG}" != "dmc_walker_multitask" &&
      "${ENV_CONFIG}" != "dmc_walker_multitask_vision" ]]; then
  echo "ENV_CONFIG must be dmc_walker_multitask or dmc_walker_multitask_vision; got ${ENV_CONFIG}." >&2
  exit 2
fi
export DMC_TASK_SET=walker_multitask
export ENV_CONFIG
export DMC_LAUNCHER_DISPLAY=launch_dmc_walker_multitask.sh
: "${ALGORITHM:=all}"
: "${IDM_CONFIG:=all}"
: "${BACKEND_AGENT:=dreamerv3}"
: "${MODEL_CONFIG:=size12M}"
: "${WANDB_PROJECT:=dmc_walker_multitask}"
export MODEL_CONFIG WANDB_PROJECT

case "${ALGORITHM}" in
  all|dreamer|p2e) ;;
  *) echo "Invalid ALGORITHM=${ALGORITHM}. Use all, dreamer, or p2e." >&2; exit 2 ;;
esac
blind_selection="${STRICT_TASK_BLIND_PRETRAINING:-true}"
if [[ "${ALGORITHM}" != "dreamer" && "${blind_selection,,}" != "true" ]]; then
  echo "Walker multitask P2E requires STRICT_TASK_BLIND_PRETRAINING=true." >&2
  exit 2
fi
case "${BACKEND_AGENT}" in
  dreamer|dreamerv3|v3) BACKEND_AGENT=dreamerv3 ;;
  ne|ne_dreamer|nedreamer) BACKEND_AGENT=ne_dreamer ;;
  r2|r2dreamer|r2_dreamer) BACKEND_AGENT=r2dreamer ;;
  *) echo "Invalid BACKEND_AGENT=${BACKEND_AGENT}. Select one backend." >&2; exit 2 ;;
esac

RUN_ALGORITHMS=()
RUN_IDMS=()
if [[ "${ALGORITHM}" != "p2e" ]]; then
  RUN_ALGORITHMS+=(dreamer)
  RUN_IDMS+=(none)
fi
if [[ "${ALGORITHM}" != "dreamer" ]]; then
  if [[ "${IDM_CONFIG}" == "all" ]]; then
    source "${REPO_ROOT}/experiments/_craftax_blind_idms.sh"
    # Walker includes all four empowerment objectives.
    P2E_IDMS=("${CRAFTAX_BLIND_IDMS[@]}" empowerment_forward empowerment_gain)
  else
    P2E_IDMS=("${IDM_CONFIG}")
  fi
  for idm in "${P2E_IDMS[@]}"; do
    RUN_ALGORITHMS+=(p2e)
    RUN_IDMS+=("${idm}")
  done
fi

run_selection() {
  local index="$1"
  shift
  local algorithm="${RUN_ALGORITHMS[${index}]}" blind=false
  local label="walker_multitask_${BACKEND_AGENT}"
  if [[ "${algorithm}" == "p2e" ]]; then
    blind=true
    label="walker_multitask_p2e_${RUN_IDMS[${index}]}_${BACKEND_AGENT}"
  fi
  ALGORITHM="${algorithm}" IDM_CONFIG="${RUN_IDMS[${index}]}" \
    BACKEND_AGENT="${BACKEND_AGENT}" DMC_JOB_LABEL="${label}" \
    STRICT_TASK_BLIND_PRETRAINING="${blind}" \
    bash "${DMC_LAUNCHER}" "$@"
}

if [[ "${1:-}" == "--worker" ]]; then
  if (( ${#RUN_ALGORITHMS[@]} != 1 )); then
    echo "--worker requires one ALGORITHM and one IDM_CONFIG for P2E." >&2
    exit 2
  fi
  run_selection 0 --worker
  exit 0
fi

# Validate every selection before any submission or source-bundle publication.
for index in "${!RUN_ALGORITHMS[@]}"; do
  run_selection "${index}" --list >/dev/null
done

echo "Walker multitask: ${#RUN_ALGORITHMS[@]} methods per seed"
echo "  source=dmc_walker_stand eval=walker_stand,walker_walk,walker_run"
echo "  envs=${ENV_CONFIG} backend=dm_control model=${MODEL_CONFIG} seeds=${SEEDS:-${SEED:-0}}"
echo "  wandb.project=${WANDB_PROJECT}"
if [[ "${1:-}" == "--list" ]]; then
  printf '%-10s %-12s %s\n' 'Algorithm' 'Backend' 'IDM'
  for index in "${!RUN_ALGORITHMS[@]}"; do
    printf '%-10s %-12s %s\n' "${RUN_ALGORITHMS[${index}]}" \
      "${BACKEND_AGENT}" "${RUN_IDMS[${index}]}"
  done
  run_selection 0 --list
else
  for index in "${!RUN_ALGORITHMS[@]}"; do
    run_selection "${index}"
  done
fi
