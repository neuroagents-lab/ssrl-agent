#!/usr/bin/env bash

set -euo pipefail

# Set the worker group before creating source and runtime files.
ensure_shared_worker_group() {
  [[ -n "${SSRL_SHARED_GROUP:-}" ]] || return 0
  [[ "${SSRL_SHARED_GROUP}" =~ ^[a-z_][a-z0-9_-]*$ ]] || {
    echo "Invalid SSRL_SHARED_GROUP." >&2; exit 2
  }
  [[ "${1:-}" == "--worker" ]] || return 0
  [[ "$(id -gn)" != "${SSRL_SHARED_GROUP}" ]] || return 0
  [[ "${SSRL_SHARED_GROUP_REEXEC:-}" != "1" ]] || {
    echo "Could not activate shared group ${SSRL_SHARED_GROUP}." >&2; exit 1
  }
  # Python must not inherit a vanished shared working directory.
  cd -- / || exit 1
  local launcher command
  case "${ADAPTATION_SUITE}" in
    craftax) launcher=launch_craftax_adaptation.sh ;;
    walker) launcher=launch_dmc_walker_adaptation.sh ;;
    *) echo "Unknown adaptation suite." >&2; exit 2 ;;
  esac
  # sg evaluates its command with /bin/sh; shlex quoting preserves arbitrary
  # source paths and the original arguments without shell interpretation.
  command="$(/usr/bin/python3 -c 'import shlex,sys; print(shlex.join(sys.argv[1:]))' \
    bash "${ADAPTATION_HELPER_ROOT}/experiments/${launcher}" "$@")"
  export SSRL_SHARED_GROUP_REEXEC=1
  exec sg "${SSRL_SHARED_GROUP}" -c "exec ${command}"
}

ensure_shared_worker_group "$@"

stage_submitted_source() {
  [[ "${DRY_RUN:-false}" != "true" ]] || return 0
  [[ -n "${SLURM_JOB_ID:-}" && -n "${SSRL_SOURCE_BUNDLE_DIR:-}" ]] || return 0
  # Use a local working directory while verifying the shared source bundle.
  cd -- / || {
    echo "Could not enter the stable source-staging working directory." >&2
    exit 1
  }
  local source_id="${SSRL_SUBMISSION_SOURCE_ID:?Missing submitted source ID}"
  local shared_tree="${SSRL_SOURCE_BUNDLE_DIR}/tree"
  local helper="${shared_tree}/experiments/run_storage.py"
  /usr/bin/python3 "${helper}" source-bundle verify \
    --bundle-dir "${SSRL_SOURCE_BUNDLE_DIR}" --source-id "${source_id}" \
    >/dev/null

  # Stage source and the working directory locally to isolate training from shared
  # filesystem outages.
  local scratch_root="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
  local job_key="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID:-0}_${SLURM_RESTART_COUNT:-0}"
  local destination="${scratch_root}/ssrl-${USER}/source/${source_id}/${job_key}"
  local stage="${destination}.tmp.${BASHPID}.${RANDOM}"
  mkdir -p "$(dirname "${destination}")"
  if [[ ! -d "${destination}" || -L "${destination}" ]]; then
    [[ ! -e "${destination}" && ! -L "${destination}" ]] || {
      echo "Refusing unsafe staged source path: ${destination}" >&2
      exit 1
    }
    cp -a -- "${shared_tree}" "${stage}"
    local staged_id
    staged_id="$(/usr/bin/python3 "${stage}/experiments/run_storage.py" \
      source-id --repo "${stage}")"
    if [[ "${staged_id}" != "${source_id}" ]]; then
      echo "Staged source ID mismatch: ${staged_id} != ${source_id}" >&2
      rm -rf -- "${stage}"
      exit 1
    fi
    mv -- "${stage}" "${destination}"
  fi
  local destination_id
  destination_id="$(/usr/bin/python3 "${destination}/experiments/run_storage.py" \
    source-id --repo "${destination}")"
  if [[ "${destination_id}" != "${source_id}" ]]; then
    echo "Existing staged source ID mismatch: ${destination_id} != ${source_id}" >&2
    exit 1
  fi
  SSRL_REPO_ROOT="${destination}"
  export SSRL_REPO_ROOT
  cd "${destination}"
}

stage_submitted_source

if [[ -n "${SSRL_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${SSRL_REPO_ROOT}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

source "${REPO_ROOT}/experiments/_runtime_defaults.sh"

: "${BACKEND_AGENT:=dreamerv3}"
: "${ADAPTATION_PROTOCOL:=fresh_heads}"
: "${PARENT_ALGORITHM:=p2e}"
: "${PARENT_COHORT_MANIFEST:=}"
: "${PARENT_COHORT_MANIFEST_SHA256:=}"
: "${PARENT_COHORT_RECEIPT:=}"
: "${PARENT_COHORT_RECEIPT_SHA256:=}"
: "${IDM_SUBSET:=}"
: "${PARENT_SEED:=0}"
: "${ADAPTATION_SEED:=0}"
if [[ "${ADAPTATION_SUITE}" == "walker" ]]; then
  : "${ADAPTATION_STEPS:=200000}"
  : "${MODEL_CONFIG:=size12M}"
  : "${ENV_CONFIG:=dmc_walker_adaptation}"
  case "${ENV_CONFIG}" in
    dmc_proprio|dmc_walker_multitask) ENV_CONFIG=dmc_walker_adaptation ;;
    dmc_vision|dmc_walker_multitask_vision) ENV_CONFIG=dmc_walker_adaptation_vision ;;
  esac
  : "${TASKS:=run}"
  : "${WANDB_PROJECT:=dmc_walker_adaptation}"
  ADAPTATION_DISPLAY=Walker
  ADAPTATION_LAUNCHER=launch_dmc_walker_adaptation.sh
  PREFLIGHT="${REPO_ROOT}/experiments/walker_adaptation.py"
  ADAPTATION_EVAL_EVERY=20000
  : "${ADAPTATION_JOB_NAME:=walker_adapt_${BACKEND_AGENT}_${ENV_CONFIG}}"
else
  : "${ADAPTATION_STEPS:=5000000}"
  : "${MODEL_CONFIG:=size200M}"
  : "${WANDB_PROJECT:=craftax_adaptation}"
  : "${ADAPTATION_JOB_NAME:=craftax_adapt_${BACKEND_AGENT}}"
  ADAPTATION_DISPLAY=Craftax
  ADAPTATION_LAUNCHER=launch_craftax_adaptation.sh
  PREFLIGHT="${REPO_ROOT}/experiments/craftax_adaptation.py"
  ADAPTATION_EVAL_EVERY=100000
fi
case "${ADAPTATION_PROTOCOL}" in
  fresh_heads) EXPECTED_OFFLINE_UPDATES=4688; STEP0_POLICY=post_algorithm2_offline ;;
  fresh_heads_10x) EXPECTED_OFFLINE_UPDATES=46880; STEP0_POLICY=post_algorithm2_offline ;;
  pretrained_heads) EXPECTED_OFFLINE_UPDATES=0; STEP0_POLICY=pretrained_task_policy ;;
  *) echo "Invalid ADAPTATION_PROTOCOL=${ADAPTATION_PROTOCOL}." >&2; exit 2 ;;
esac
: "${ADAPTATION_OFFLINE_UPDATES:=${EXPECTED_OFFLINE_UPDATES}}"
: "${ADAPTATION_OFFLINE_SAVE_EVERY:=1000}"
: "${ARRAY_TASKS:=}"
: "${MAX_CONCURRENT_GPUS:=20}"
: "${PARTITION:=}"
: "${QOS:=}"
: "${GRES:=gpu:1}"
: "${MEMORY:=128G}"
: "${TIME_LIMIT:=2-00:00:00}"
# Resume checkpoints with a new W&B attempt in the same logical group.
: "${WANDB_RELAUNCH_MODE:=fresh}"
: "${EXCLUDE_NODES:=}"
: "${WANDB_ENTITY:=}"
PARENT_TASK="craftax_classic_pixels"

resolve_preflight_python() {
  local -a candidates=()
  local candidate
  [[ -z "${ADAPTATION_PYTHON:-}" ]] || candidates+=("${ADAPTATION_PYTHON}")
  candidate="$(command -v python3 2>/dev/null || true)"
  [[ -z "${candidate}" ]] || candidates+=("${candidate}")
  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]] && \
       "${candidate}" -c 'import numpy, zstandard' >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  echo "Could not find a preflight Python with NumPy and zstandard." >&2
  return 1
}

usage() {
  if [[ "${ADAPTATION_SUITE}" == "walker" ]]; then
    cat <<'EOF'
Usage:
  PARENT_COHORT_MANIFEST=/shared/parents.json \
  PARENT_COHORT_MANIFEST_SHA256=<sha256> \
  PARENT_COHORT_RECEIPT=/shared/validation.json \
  PARENT_COHORT_RECEIPT_SHA256=<sha256> \
    bash experiments/launch_dmc_walker_adaptation.sh [--dry-run|--list]

Selectors:
  TASKS=run                      # default downstream target: Walker Run
  ENV_CONFIG=dmc_walker_adaptation|dmc_walker_adaptation_vision
                                 # dmc_proprio/dmc_vision aliases supported
  BACKEND_AGENT=dreamerv3|r2dreamer|ne_dreamer
  MODEL_CONFIG=size12M
  ADAPTATION_PROTOCOL=fresh_heads|fresh_heads_10x|pretrained_heads
                                 # offline updates: 4688 / 46880 / 0
  IDM_SUBSET=apt:disagreement:rnd # default: 23 IDMs + random + supervised none
  SEEDS=0:1:2                    # each child uses its parent's seed
  PARENT_ALGORITHM=all|p2e|dreamer # all methods in the certified cohort
  ARRAY_TASKS=0:2:4              # indices in --list; zero based
  WANDB_PROJECT=dmc_walker_adaptation
  DRY_RUN=true

Stand-source multitask parents, using the matching modality/backend/seed.
By default, children use ordinary Dreamer with fresh task heads and the final
500k replay relabeled with the target's native reward, followed by 4,688 offline
updates. fresh_heads_10x uses 46,880 updates; pretrained_heads imports the task
heads and uses zero offline updates. Collect exactly 200,000 additional steps;
evaluate 10 greedy episodes at step 0 and every 20,000 steps. Native DMC train
ratio 512, 16 training environments, and action repeat/chunk 1.
See README.md for checkpoint preparation and launch instructions.
EOF
    return
  fi
  cat <<'EOF'
Usage:
  PARENT_COHORT_MANIFEST=/shared/parents.json \
  PARENT_COHORT_MANIFEST_SHA256=<sha256> \
  PARENT_COHORT_RECEIPT=/shared/parents.validation.json \
  PARENT_COHORT_RECEIPT_SHA256=<sha256> \
    bash experiments/launch_craftax_adaptation.sh

Options:
  BACKEND_AGENT=dreamerv3|r2dreamer|ne_dreamer
  PARENT_ALGORITHM=p2e|dreamer     # dreamer is the explicit supervised control
  IDM_SUBSET=apt:disagreement:rnd   # empty selects every certified match
  PARENT_SEED=0
  ADAPTATION_SEED=0
  ADAPTATION_STEPS=1000000|5000000 # default 5M; explicit 1M study supported
  ADAPTATION_PROTOCOL=fresh_heads  # or fresh_heads_10x / pretrained_heads
  ADAPTATION_OFFLINE_UPDATES=4688  # automatic: 46880 for fresh_heads_10x, 0 for pretrained_heads
  ADAPTATION_OFFLINE_SAVE_EVERY=1000
  MODEL_CONFIG=size200M
  WANDB_PROJECT=craftax_adaptation # override for a separate adaptation cohort
  ARRAY_TASKS=1:3:8                  # optional original cohort indices
  GRES=gpu:A6000:1                # optional compatible-GPU override
  DRY_RUN=true

By default, each child uses algo=dreamer, a fresh task actor/critic/reward model,
the parent's final 500k replay, 4,688 learner-only Algorithm-2 updates, then
greedy 10-episode evaluation at environment step 0 and each 100k boundary, and
adaptation-relative logging through the selected 1M or 5M environment steps. Each manifest
entry must pin a trusted, non-executable compatibility certificate.
fresh_heads_10x uses 46,880 offline updates; pretrained_heads imports the task
heads and uses zero offline updates.
EOF
}

validate_scalar_selections() {
  if [[ "${ADAPTATION_SUITE}" == "walker" ]]; then
    case "${PARENT_ALGORITHM}" in
      all|p2e|dreamer) ;;
      *) echo "Invalid PARENT_ALGORITHM=${PARENT_ALGORITHM}." >&2; exit 2 ;;
    esac
    [[ "${SEEDS}" =~ ^(0|[1-9][0-9]*)(:(0|[1-9][0-9]*))*$ ]] || {
      echo "SEEDS must be colon-delimited nonnegative integers." >&2; exit 2;
    }
  else
    case "${PARENT_ALGORITHM}/${BACKEND_AGENT}" in
      p2e/*|dreamer/dreamerv3) ;;
      *) echo "Invalid parent algorithm/backend." >&2; exit 2 ;;
    esac
  fi
  [[ "${ADAPTATION_JOB_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "Invalid ADAPTATION_JOB_NAME." >&2; exit 2
  }
  case "${BACKEND_AGENT}" in
    dreamerv3|r2dreamer|ne_dreamer) ;;
    *) echo "Invalid BACKEND_AGENT=${BACKEND_AGENT}." >&2; exit 2 ;;
  esac
  for pair in "PARENT_SEED=${PARENT_SEED}" "ADAPTATION_SEED=${ADAPTATION_SEED}"; do
    [[ "${pair#*=}" =~ ^(0|[1-9][0-9]*)$ ]] || {
      echo "Invalid ${pair}; seeds must be nonnegative integers." >&2
      exit 2
    }
  done
  if [[ "${ADAPTATION_SUITE}" == "walker" ]]; then
    [[ "${ADAPTATION_STEPS}" == "200000" ]] || {
      echo "ADAPTATION_STEPS must be exactly 200000 for Walker." >&2; exit 2;
    }
    case "${ENV_CONFIG}" in
      dmc_walker_adaptation|dmc_walker_adaptation_vision) ;;
      *) echo "ENV_CONFIG must select Walker proprio or vision adaptation." >&2; exit 2 ;;
    esac
    [[ "${TASKS}" =~ ^(stand|walk|run)(:(stand|walk|run))*$ ]] || {
      echo "TASKS must be colon-delimited stand, walk, run selections." >&2; exit 2;
    }
  else
    [[ "${ADAPTATION_STEPS}" == "1000000" || "${ADAPTATION_STEPS}" == "5000000" ]] || {
      echo "ADAPTATION_STEPS must be exactly 1000000 or 5000000." >&2
      exit 2
    }
  fi
  [[ "${ADAPTATION_OFFLINE_UPDATES}" == "${EXPECTED_OFFLINE_UPDATES}" ]] || {
    echo "ADAPTATION_OFFLINE_UPDATES must be exactly ${EXPECTED_OFFLINE_UPDATES} for ${ADAPTATION_PROTOCOL}." >&2
    exit 2
  }
  [[ "${ADAPTATION_OFFLINE_SAVE_EVERY}" == "1000" ]] || {
    echo "ADAPTATION_OFFLINE_SAVE_EVERY must be exactly 1000." >&2
    exit 2
  }
  [[ "${MODEL_CONFIG}" =~ ^size[0-9]+[KMG]$ && \
     -f "${REPO_ROOT}/configs/model/${MODEL_CONFIG}.yaml" ]] || {
    echo "Invalid MODEL_CONFIG=${MODEL_CONFIG}." >&2
    exit 2
  }
  [[ -n "${PARENT_COHORT_MANIFEST}" ]] || {
    echo "PARENT_COHORT_MANIFEST is required." >&2
    exit 2
  }
  [[ "${WANDB_PROJECT}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "Invalid WANDB_PROJECT=${WANDB_PROJECT}." >&2
    exit 2
  }
  [[ "${PARENT_COHORT_MANIFEST}${PARENT_COHORT_RECEIPT}${IDM_SUBSET}" != *","* ]] || {
    echo "Manifest/receipt paths and IDM_SUBSET may not contain commas (Slurm export delimiter)." >&2
    exit 2
  }
  [[ "${PARENT_COHORT_MANIFEST}${PARENT_COHORT_RECEIPT}${IDM_SUBSET}" != *$'\n'* ]] || {
    echo "Manifest/receipt paths and IDM_SUBSET may not contain newlines." >&2
    exit 2
  }
  [[ "${PARENT_COHORT_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "PARENT_COHORT_MANIFEST_SHA256 must pin the immutable manifest." >&2
    exit 2
  }
  [[ -n "${PARENT_COHORT_RECEIPT}" ]] || {
    echo "PARENT_COHORT_RECEIPT is required." >&2
    exit 2
  }
  [[ "${PARENT_COHORT_RECEIPT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "PARENT_COHORT_RECEIPT_SHA256 must pin the validation receipt." >&2
    exit 2
  }
  # Workers enter the frozen source tree; relative submit-side paths must not
  # change their meaning when Slurm stages the job on another node.
  PARENT_COHORT_MANIFEST="$(/usr/bin/python3 -c \
    'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' \
    "${PARENT_COHORT_MANIFEST}")"
  PARENT_COHORT_RECEIPT="$(/usr/bin/python3 -c \
    'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' \
    "${PARENT_COHORT_RECEIPT}")"
  [[ "${MAX_CONCURRENT_GPUS}" =~ ^([1-9]|1[0-9]|2[0-4])$ ]] || {
    echo "MAX_CONCURRENT_GPUS must be 1-24." >&2
    exit 2
  }
  ssrl_validate_slurm_options

}

preflight_command() {
  local -a command
  local preflight_python
  preflight_python="$(resolve_preflight_python)"
  command=(
    "${preflight_python}" "${PREFLIGHT}" validate-parents
    --manifest "${PARENT_COHORT_MANIFEST}"
    --manifest-sha256 "${PARENT_COHORT_MANIFEST_SHA256}"
    --receipt "${PARENT_COHORT_RECEIPT}"
    --receipt-sha256 "${PARENT_COHORT_RECEIPT_SHA256}"
    --backend "${BACKEND_AGENT}"
    --protocol "${ADAPTATION_PROTOCOL}"
    --parent-algorithm "${PARENT_ALGORITHM}"
    --model "${MODEL_CONFIG}"
  )
  if [[ "${ADAPTATION_SUITE}" == "walker" ]]; then
    command+=(--tasks "${TASKS}" --env-config "${ENV_CONFIG}" --seeds "${SEEDS}")
  else
    command+=(--task "${PARENT_TASK}" --parent-seed "${PARENT_SEED}")
  fi
  [[ -z "${IDM_SUBSET}" ]] || command+=(--idms "${IDM_SUBSET}")
  printf '%s\0' "${command[@]}"
}

validate_parents() {
  local -a command
  mapfile -d '' -t command < <(preflight_command)
  "${command[@]}"
}

parent_at_index() {
  local index="$1"
  local -a command
  mapfile -d '' -t command < <(preflight_command)
  command+=(--index "${index}")
  "${command[@]}"
}

run_array_task() {
  local index parent_index parent_json
  local -a fields
  index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required in worker mode}"
  [[ "${index}" =~ ^(0|[1-9][0-9]*)$ ]] || {
    echo "Invalid SLURM_ARRAY_TASK_ID=${index}." >&2
    exit 2
  }
  # Keep the Slurm index for run identity; select the parent within the verified
  # manifest independently.
  parent_index="${PARENT_COHORT_INDEX:-${index}}"
  [[ "${parent_index}" =~ ^(0|[1-9][0-9]*)$ ]] || {
    echo "Invalid PARENT_COHORT_INDEX=${parent_index}." >&2
    exit 2
  }
  parent_json="$(parent_at_index "${parent_index}")"
  mapfile -t fields < <(/usr/bin/python3 -c '
import json, sys
x = json.loads(sys.argv[1]); r = x["replay_binding"]; c = x["compatibility_certificate"]
for value in (x["idm"], x["run_uid"], x["source_digest"],
              x["checkpoint_generation"], x["checkpoint_sha256"], r["directory"],
              r["inventory_sha256"], r["sequence_length"], c["path"],
              c["sha256"]): print(value)
print("true" if x["strict_task_blind_pretraining"] is True else "false")
print(x["task"])
print(x["parent_step"])
print(x.get("parent_algorithm", "p2e"))
print(x["parent_seed"])
print(x.get("parent_task", x["task"]))
print(x.get("parent_env_config", ""))
' "${parent_json}")
  [[ "${#fields[@]}" -eq 17 ]] || {
    echo "Validated parent record is incomplete." >&2
    exit 1
  }

  local parent_idm="${fields[0]}"
  CFG="craftax_classic"
  MODEL_SIZE="${MODEL_CONFIG}"
  AGENT="${BACKEND_AGENT}"
  DOMAIN_NAME="craftax"
  TASK_NAME="classic_pixels"
  local parent_step="${fields[12]}"
  if [[ "${ADAPTATION_SUITE}" == "walker" ]]; then
    CFG="${ENV_CONFIG}"
    DOMAIN_NAME=walker
    TASK_NAME="${fields[11]#dmc_walker_}"
    DMC_BACKEND=dm_control
    PARENT_ALGORITHM="${fields[13]}"
    PARENT_SEED="${fields[14]}"
    ADAPTATION_SEED="${PARENT_SEED}"
    ADAPTATION_PARENT_TASK="${fields[15]}"
    ADAPTATION_PARENT_ENV_CONFIG="${fields[16]}"
    export ADAPTATION_PARENT_TASK ADAPTATION_PARENT_ENV_CONFIG
    unset MJX_ACTOR_CHUNK
    # Use 16 parallel environments to align evaluation cadence and the step budget.
    ENV_NUM=16
    unset DEVICE_ACTOR_CHUNK
    TRAIN_RATIO=512
    export DMC_BACKEND ENV_NUM TRAIN_RATIO
  fi
  ALGO_PRESET="dreamer"
  IDM_TYPE="none"
  SEED="${ADAPTATION_SEED}"
  STEPS="${ADAPTATION_STEPS}"
  EVAL_EVERY="${ADAPTATION_EVAL_EVERY}"
  EVAL_EPISODES=10
  EVAL_ENV_NUM=10
  EVAL_ACTION_SELECTIONS="[greedy]"
  ADAPTATION_ENABLED=true
  ADAPTATION_BACKEND="${BACKEND_AGENT}"
  ADAPTATION_PARENT_ALGORITHM="${PARENT_ALGORITHM}"
  ADAPTATION_PARENT_IDM="${parent_idm}"
  ADAPTATION_PARENT_SEED="${PARENT_SEED}"
  ADAPTATION_MODEL="${MODEL_CONFIG}"
  ADAPTATION_PARENT_RUN_UID="${fields[1]}"
  ADAPTATION_PARENT_SOURCE_DIGEST="${fields[2]}"
  ADAPTATION_PARENT_CHECKPOINT="${fields[3]}"
  ADAPTATION_PARENT_CHECKPOINT_SHA256="${fields[4]}"
  ADAPTATION_PARENT_REPLAY="${fields[5]}"
  ADAPTATION_REPLAY_INVENTORY_SHA256="${fields[6]}"
  ADAPTATION_PARENT_REPLAY_LENGTH="${fields[7]}"
  ADAPTATION_COMPATIBILITY_CERTIFICATE="${fields[8]}"
  ADAPTATION_COMPATIBILITY_CERTIFICATE_SHA256="${fields[9]}"
  ADAPTATION_COHORT_VALIDATION_RECEIPT="${PARENT_COHORT_RECEIPT}"
  ADAPTATION_COHORT_VALIDATION_RECEIPT_SHA256="${PARENT_COHORT_RECEIPT_SHA256}"
  PRETRAINING_ENVIRONMENT_STEPS="${parent_step}"
  STRICT_TASK_BLIND_PRETRAINING="${fields[10]}"
  [[ "${PARENT_ALGORITHM}/${STRICT_TASK_BLIND_PRETRAINING}" == "p2e/true" ||
     "${PARENT_ALGORITHM}/${STRICT_TASK_BLIND_PRETRAINING}" == "dreamer/false" ]] || {
    echo "Validated adaptation parent algorithm/blindness mismatch." >&2
    exit 1
  }
  EXPERIMENT_NAME="${WANDB_PROJECT}_${BACKEND_AGENT}_${MODEL_CONFIG}"
  if [[ "${ADAPTATION_SUITE}" == "walker" ]]; then
    EXPERIMENT_NAME="${WANDB_PROJECT}_${ENV_CONFIG}_${BACKEND_AGENT}_${parent_idm}_${MODEL_CONFIG}_${DOMAIN_NAME}_${TASK_NAME}"
  fi
  if [[ "${ADAPTATION_PROTOCOL}" != "fresh_heads" ]]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_${ADAPTATION_PROTOCOL}"
  fi
  WANDB_NOTES_EXTRA="adaptation_protocol=${ADAPTATION_PROTOCOL}; adaptation_suite=${ADAPTATION_SUITE}; parent_algorithm=${PARENT_ALGORITHM}; algorithm2_offline_updates=${ADAPTATION_OFFLINE_UPDATES}; step0_policy=${STEP0_POLICY}; parent_idm=${parent_idm}; parent_seed=${PARENT_SEED}; adaptation_seed=${ADAPTATION_SEED}; pretraining_environment_steps=${parent_step}; strict_task_blind_pretraining=${STRICT_TASK_BLIND_PRETRAINING}; compatibility_certificate_sha256=${ADAPTATION_COMPATIBILITY_CERTIFICATE_SHA256}; cohort_validation_receipt_sha256=${ADAPTATION_COHORT_VALIDATION_RECEIPT_SHA256}"
  HELPER_DIR="${REPO_ROOT}/experiments"
  export STEPS EVAL_EVERY EVAL_EPISODES EVAL_ENV_NUM EVAL_ACTION_SELECTIONS
  export ADAPTATION_ENABLED ADAPTATION_BACKEND ADAPTATION_PARENT_IDM ADAPTATION_PROTOCOL
  export ADAPTATION_PARENT_ALGORITHM
  export ADAPTATION_PARENT_SEED ADAPTATION_MODEL ADAPTATION_PARENT_RUN_UID
  export ADAPTATION_PARENT_SOURCE_DIGEST ADAPTATION_PARENT_CHECKPOINT
  export ADAPTATION_PARENT_CHECKPOINT_SHA256
  export ADAPTATION_PARENT_REPLAY ADAPTATION_REPLAY_INVENTORY_SHA256
  export ADAPTATION_PARENT_REPLAY_LENGTH
  export ADAPTATION_COMPATIBILITY_CERTIFICATE
  export ADAPTATION_COMPATIBILITY_CERTIFICATE_SHA256
  export ADAPTATION_COHORT_VALIDATION_RECEIPT
  export ADAPTATION_COHORT_VALIDATION_RECEIPT_SHA256
  export ADAPTATION_OFFLINE_UPDATES ADAPTATION_OFFLINE_SAVE_EVERY
  export PRETRAINING_ENVIRONMENT_STEPS STRICT_TASK_BLIND_PRETRAINING
  export EXPERIMENT_NAME WANDB_NOTES_EXTRA

  echo "Adaptation array ${index}: backend=${BACKEND_AGENT} parent_idm=${parent_idm} parent_seed=${PARENT_SEED} adaptation_seed=${ADAPTATION_SEED}"
  echo "Parent checkpoint: ${ADAPTATION_PARENT_CHECKPOINT}"
  if [[ "${ADAPTATION_OFFLINE_UPDATES}" -gt 0 ]]; then
    echo "Objective: Algorithm-2 offline reward/behavior adaptation (${ADAPTATION_OFFLINE_UPDATES} updates), then ordinary ${BACKEND_AGENT} Dreamer; environment steps=0..${ADAPTATION_STEPS}"
  else
    echo "Objective: pretrained task heads, then ordinary ${BACKEND_AGENT} Dreamer; environment steps=0..${ADAPTATION_STEPS}"
  fi
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/experiments/_run_common.sh"
}

submit_array() {
  local parents_json count last_index array_spec submission_count source_id source_bundle source_tree export_vars
  local selected_index
  local -a selected_indices
  local -A seen_indices=()
  local slurm_workdir log_root
  local -a command
  # Validate inputs before capturing source or submitting jobs.
  parents_json="$(validate_parents)"
  count="$(/usr/bin/python3 -c 'import json,sys; print(json.loads(sys.argv[1])["count"])' "${parents_json}")"
  (( count > 0 )) || { echo "No parents selected." >&2; exit 2; }
  last_index=$((count - 1))
  submission_count="${count}"
  if [[ -n "${ARRAY_TASKS}" ]]; then
    [[ "${ARRAY_TASKS}" =~ ^(0|[1-9][0-9]*)(:(0|[1-9][0-9]*))*$ ]] || {
      echo "Invalid ARRAY_TASKS=${ARRAY_TASKS}; use unique colon-delimited indices." >&2
      exit 2
    }
    IFS=: read -r -a selected_indices <<< "${ARRAY_TASKS}"
    for selected_index in "${selected_indices[@]}"; do
      (( selected_index <= last_index )) || {
        echo "Invalid ARRAY_TASKS=${ARRAY_TASKS}: index ${selected_index} is outside 0-${last_index}." >&2
        exit 2
      }
      [[ -z "${seen_indices[${selected_index}]:-}" ]] || {
        echo "Invalid ARRAY_TASKS=${ARRAY_TASKS}: duplicate index ${selected_index}." >&2
        exit 2
      }
      seen_indices["${selected_index}"]=1
    done
    array_spec="${ARRAY_TASKS//:/,}%${MAX_CONCURRENT_GPUS}"
    submission_count="${#selected_indices[@]}"
  else
    array_spec="0-${last_index}%${MAX_CONCURRENT_GPUS}"
  fi
  slurm_workdir="${SSRL_SLURM_WORKDIR:-/}"
  [[ -d "${slurm_workdir}" ]] || { echo "Invalid Slurm workdir." >&2; exit 2; }
  slurm_workdir="$(cd "${slurm_workdir}" && pwd -P)"
  # Slurm opens these before the worker starts; keep both streams node-local.
  # The storage publisher preserves them with the other run artifacts.
  log_root="/tmp"

  source_id="$(/usr/bin/python3 "${REPO_ROOT}/experiments/run_storage.py" source-id --repo "${REPO_ROOT}")"
  source_bundle="${SSRL_SOURCE_BUNDLE_ROOT}/${source_id}"
  source_tree="${source_bundle}/tree"
  if [[ "${DRY_RUN:-false}" != "true" ]]; then
    mkdir -p "${SSRL_SOURCE_BUNDLE_ROOT}"
    /usr/bin/python3 "${REPO_ROOT}/experiments/run_storage.py" source-bundle ensure \
      --repo "${REPO_ROOT}" --bundle-dir "${source_bundle}" \
      --source-id "${source_id}" >/dev/null
  fi
  export_vars="SSRL_REPO_ROOT=${source_tree},SSRL_SOURCE_BUNDLE_DIR=${source_bundle},SSRL_SUBMISSION_SOURCE_ID=${source_id},BACKEND_AGENT=${BACKEND_AGENT},PARENT_COHORT_MANIFEST=${PARENT_COHORT_MANIFEST},PARENT_COHORT_MANIFEST_SHA256=${PARENT_COHORT_MANIFEST_SHA256},PARENT_COHORT_RECEIPT=${PARENT_COHORT_RECEIPT},PARENT_COHORT_RECEIPT_SHA256=${PARENT_COHORT_RECEIPT_SHA256},IDM_SUBSET=${IDM_SUBSET},PARENT_SEED=${PARENT_SEED},ADAPTATION_SEED=${ADAPTATION_SEED},ADAPTATION_STEPS=${ADAPTATION_STEPS},ADAPTATION_PROTOCOL=${ADAPTATION_PROTOCOL},ADAPTATION_OFFLINE_UPDATES=${ADAPTATION_OFFLINE_UPDATES},ADAPTATION_OFFLINE_SAVE_EVERY=${ADAPTATION_OFFLINE_SAVE_EVERY},MODEL_CONFIG=${MODEL_CONFIG},WANDB_RELAUNCH_MODE=${WANDB_RELAUNCH_MODE},USER=${USER},HOME=${HOME},SLURM_PREEMPTABLE=true"
  export_vars+=",SSRL_REQUIRE_NODE_LOCAL_RUNTIME=true"
  export_vars+=",WANDB_PROJECT=${WANDB_PROJECT}"
  export_vars+=",PARENT_ALGORITHM=${PARENT_ALGORITHM},ADAPTATION_JOB_NAME=${ADAPTATION_JOB_NAME}"
  if [[ "${ADAPTATION_SUITE}" == "walker" ]]; then
    export_vars+=",ENV_CONFIG=${ENV_CONFIG},TASKS=${TASKS},SEEDS=${SEEDS}"
  fi
  local runtime_key runtime_value
  for runtime_key in PATH VIRTUAL_ENV CONDA_PREFIX LD_LIBRARY_PATH SSRL_SHARED_GROUP CONDA_EXE CONDA_ENV WANDB_ENTITY WANDB_MODE WANDB_BASE_URL \
      SCRATCH_LOG_ROOT EXPERIMENT_ROOT WORK_LOG_ROOT JAX_PREALLOCATE; do
    [[ -v "${runtime_key}" ]] || continue
    runtime_value="${!runtime_key}"
    [[ "${runtime_value}" != *","* && "${runtime_value}" != *$'\n'* ]] || {
      echo "Invalid ${runtime_key}: Slurm exports cannot contain commas/newlines." >&2; exit 2;
    }
    export_vars+=",${runtime_key}=${runtime_value}"
  done
  [[ -z "${WANDB_API_KEY:-}" ]] || export_vars+=",WANDB_API_KEY"
  command=(
    sbatch
    "--job-name=${ADAPTATION_JOB_NAME}"
    "--array=${array_spec}"
    "--time=${TIME_LIMIT}" "--cpus-per-task=4" "--mem=${MEMORY}"
    "--gres=${GRES}" "--chdir=${slurm_workdir}"
    "--output=${log_root}/ssrl-slurm-%u-%x-%A_%a.out"
    "--error=${log_root}/ssrl-slurm-%u-%x-%A_%a.err"
    "--signal=B:USR1@300" --requeue --open-mode=append
    "--export=${export_vars}"
  )
  [[ -z "${PARTITION}" ]] || command+=("--partition=${PARTITION}")
  [[ -z "${QOS}" ]] || command+=("--qos=${QOS}")
  if [[ -n "${EXCLUDE_NODES}" ]]; then
    command+=("--exclude=${EXCLUDE_NODES}")
  fi
  command+=("${source_tree}/experiments/${ADAPTATION_LAUNCHER}" --worker)
  echo "Submitting ${submission_count} certified ${ADAPTATION_DISPLAY} adaptation jobs as array ${array_spec}:"
  if [[ "${ADAPTATION_SUITE}" == "walker" ]]; then
    echo "  backend=${BACKEND_AGENT} model=${MODEL_CONFIG} paired_seeds=${SEEDS} tasks=${TASKS}"
  else
    echo "  backend=${BACKEND_AGENT} model=${MODEL_CONFIG} parent_seed=${PARENT_SEED} adaptation_seed=${ADAPTATION_SEED}"
  fi
  echo "  offline_updates=${ADAPTATION_OFFLINE_UPDATES} adaptation_steps=${ADAPTATION_STEPS} protocol=${ADAPTATION_PROTOCOL} evaluations=${STEP0_POLICY}-step0+${ADAPTATION_EVAL_EVERY} greedy_episodes=10"
  echo "  manifest_sha256=${PARENT_COHORT_MANIFEST_SHA256}"
  echo "  validation_receipt_sha256=${PARENT_COHORT_RECEIPT_SHA256}"
  echo "  gres=${GRES} exclude_nodes=${EXCLUDE_NODES}"
  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    printf '%q ' "${command[@]}"; echo
  else
    "${command[@]}"
  fi
}

if [[ "${1:-}" == "--dry-run" ]]; then
  export DRY_RUN=true
  shift
fi
case "${1:-}" in
  -h|--help) usage ;;
  --list) validate_scalar_selections; validate_parents ;;
  --worker) validate_scalar_selections; run_array_task ;;
  "") validate_scalar_selections; submit_array ;;
  *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac
