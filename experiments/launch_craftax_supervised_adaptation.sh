#!/usr/bin/env bash
# Supervised DreamerV3 control for the Craftax blind-parent adaptation study.
# Uses the same replay/offline/online protocol with freshly initialized heads.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BACKEND_AGENT=dreamerv3 PARENT_ALGORITHM=dreamer IDM_SUBSET=none
export MODEL_CONFIG=size200M
export WANDB_PROJECT="${WANDB_PROJECT:-craftax_blind_adaptation}"
export ADAPTATION_JOB_NAME="${ADAPTATION_JOB_NAME:-craftax_adapt_supervised_dreamerv3}"
export MAX_CONCURRENT_GPUS=1
exec bash "${SCRIPT_DIR}/launch_craftax_adaptation.sh" "$@"
