#!/usr/bin/env bash
# Shared certified transfer, offline adaptation, and Slurm recovery workflow.
set -euo pipefail
ADAPTATION_SUITE=walker
: "${PARENT_ALGORITHM:=all}"
: "${SEEDS:=0:1:2}"
: "${TASKS:=run}"
if [[ -n "${SSRL_REPO_ROOT:-}" ]]; then
  ADAPTATION_HELPER_ROOT="${SSRL_REPO_ROOT}"
else
  ADAPTATION_HELPER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
# Default to the complete Walker cohort, including both controls. Explicit
# subsets remain available for individual preparations and targeted reruns.
if [[ -z "${IDM_SUBSET:-}" ]]; then
  source "${ADAPTATION_HELPER_ROOT}/experiments/_craftax_blind_idms.sh"
  WALKER_ADAPTATION_IDMS=()
  if [[ "${PARENT_ALGORITHM}" != "dreamer" ]]; then
    WALKER_ADAPTATION_IDMS+=("${CRAFTAX_BLIND_IDMS[@]}" empowerment_forward empowerment_gain random)
  fi
  if [[ "${PARENT_ALGORITHM}" != "p2e" ]]; then
    WALKER_ADAPTATION_IDMS+=(none)
  fi
  IDM_SUBSET="$(IFS=:; echo "${WALKER_ADAPTATION_IDMS[*]}")"
fi
source "${ADAPTATION_HELPER_ROOT}/experiments/_launch_task_adaptation.sh"
