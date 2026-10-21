#!/usr/bin/env bash
# Shared certified transfer, offline adaptation, and Slurm recovery workflow.
set -euo pipefail
ADAPTATION_SUITE=craftax
if [[ -n "${SSRL_REPO_ROOT:-}" ]]; then
  ADAPTATION_HELPER_ROOT="${SSRL_REPO_ROOT}"
else
  ADAPTATION_HELPER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
source "${ADAPTATION_HELPER_ROOT}/experiments/_launch_task_adaptation.sh"
