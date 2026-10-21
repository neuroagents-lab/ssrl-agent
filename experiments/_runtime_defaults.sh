#!/usr/bin/env bash
# Shared runtime defaults. Source after REPO_ROOT (or SSRL_RUN_REPO_ROOT) is set.
# The submission checkout must be visible on compute nodes; temporary training
# files live on node-local storage and completed artifacts are published to runs/.

_SSRL_REPO_ROOT="${REPO_ROOT:-${SSRL_RUN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${SCRATCH_LOG_ROOT:=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/ssrl-${USER:-$(id -un)}/training_logs}"
: "${EXPERIMENT_ROOT:=${SCRATCH_LOG_ROOT}}"
: "${WORK_LOG_ROOT:=${_SSRL_REPO_ROOT}/runs}"
: "${SSRL_SOURCE_BUNDLE_ROOT:=${_SSRL_REPO_ROOT}/.runtime/source_bundles}"
: "${WANDB_ENTITY:=}"
: "${WANDB_MODE:=offline}"
export SCRATCH_LOG_ROOT EXPERIMENT_ROOT WORK_LOG_ROOT WANDB_ENTITY WANDB_MODE
unset _SSRL_REPO_ROOT

# Site-specific scheduler choices are optional. Empty selections use the
# cluster defaults; callers may set these variables before running a launcher.
ssrl_validate_slurm_options() {
  local name value
  for name in PARTITION QOS; do
    value="${!name:-}"
    [[ -z "${value}" || "${value}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ ]] || {
      echo "Invalid ${name}=${value}; use a Slurm partition or QoS name." >&2
      return 2
    }
  done
  [[ "${GRES:-gpu:1}" =~ ^gpu(:[A-Za-z0-9_.-]+)?:1$ ]] || {
    echo "Invalid GRES=${GRES}; use gpu:1 or gpu:<type>:1." >&2
    return 2
  }
  [[ -z "${EXCLUDE_NODES:-}" || "${EXCLUDE_NODES}" != *$'\n'* ]] || {
    echo "EXCLUDE_NODES cannot contain a newline." >&2
    return 2
  }
}
