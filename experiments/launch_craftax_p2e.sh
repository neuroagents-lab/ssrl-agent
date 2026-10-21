#!/usr/bin/env bash

set -euo pipefail

if [[ "${1:-}" == "--dry-run" ]]; then
  export DRY_RUN=true
  shift
fi

stage_submitted_source() {
  [[ "${DRY_RUN:-false}" != "true" ]] || return 0
  [[ -n "${SLURM_JOB_ID:-}" && -n "${SSRL_SOURCE_BUNDLE_DIR:-}" ]] || return 0
  local source_id="${SSRL_SUBMISSION_SOURCE_ID:?Missing submitted source ID}"
  local shared_tree="${SSRL_SOURCE_BUNDLE_DIR}/tree"
  local helper="${shared_tree}/experiments/run_storage.py"
  /usr/bin/python3 "${helper}" source-bundle verify \
    --bundle-dir "${SSRL_SOURCE_BUNDLE_DIR}" --source-id "${source_id}" \
    >/dev/null

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
  cd -- "${SSRL_REPO_ROOT}" || {
    echo "Could not enter staged source tree: ${SSRL_REPO_ROOT}" >&2
    exit 1
  }
}

stage_submitted_source

# sbatch executes a spool copy of this file. The submitter exports an immutable
# source bundle, which each worker verifies and stages onto node-local scratch.
if [[ -n "${SSRL_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${SSRL_REPO_ROOT}"
  SCRIPT_DIR="${REPO_ROOT}/experiments"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

source "${REPO_ROOT}/experiments/_runtime_defaults.sh"

# Suite-wide experiment selections

# Default to pixel-based Craftax Classic; other settings come from the selected config.
: "${ENV_CONFIG:=craftax_classic}"
: "${ALGORITHM:=p2e}"
case "${ALGORITHM}" in
  p2e|dreamer) ;;
  *) echo "ALGORITHM must be p2e or dreamer." >&2; exit 2 ;;
esac

# Choose the representation/backend agent used for every IDM.
# Allowed values: dreamerv3, r2dreamer, ne_dreamer, dreamerpro.
: "${BACKEND_AGENT:=ne_dreamer}"

# Run every IDM with seed 0 by default. Colons are used because commas delimit
# values in Slurm's --export argument. SEED remains a one-run shorthand.
SEEDS="${SEEDS:-${SEED:-0}}"
: "${ARRAY_TASKS:=}"
# Resume local training state with a new W&B attempt in the same logical group.
: "${WANDB_RELAUNCH_MODE:=fresh}"
: "${FORK_FROM_ARRAY_JOB_ID:=}"
: "${FORK_FROM_SOURCE_ID:=}"
: "${FORK_FROM_CLUSTER:=${SLURM_CLUSTER_NAME:-slurm}}"

: "${MODEL_CONFIG:=size200M}"
: "${LEARNER_BATCH_SIZE:=16}"
: "${LEARNER_RSSM_UNROLL:=auto}"
: "${EXPL_GRAD_MODE:=}"
: "${TOTAL_STEPS:=}"
# P2E parents are task blind; supervised baselines optimize task reward.
if [[ "${ALGORITHM}" == "p2e" ]]; then
  : "${STRICT_TASK_BLIND_PRETRAINING:=true}"
else
  : "${STRICT_TASK_BLIND_PRETRAINING:=false}"
fi
: "${ADAPTIVE_CRITIC_ENABLED:=}"
: "${ADAPTIVE_CRITIC_CRITICS:=}"
: "${ADAPTIVE_CRITIC_SCOPE:=}"
: "${ADAPTIVE_CRITIC_GRANULARITY:=}"
: "${ADAPTIVE_CRITIC_META_LR:=}"
: "${ADAPTIVE_CRITIC_TRACE_DECAY:=}"
: "${ADAPTIVE_CRITIC_INIT_SCALE:=}"
: "${ADAPTIVE_CRITIC_MIN_SCALE:=}"
: "${ADAPTIVE_CRITIC_MAX_SCALE:=}"
: "${ADAPTIVE_CRITIC_WARMUP:=}"
: "${ADAPTIVE_CRITIC_EPS:=}"
if [[ "${ALGORITHM}" == "p2e" ]]; then
  : "${WANDB_PROJECT:=craftax_blind}"
else
  : "${WANDB_PROJECT:=craftax_supervised}"
fi
: "${WANDB_ENTITY:=}"
# Advance shared progress only after publishing the matching checkpoint and replay.
: "${SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_MODE:=published}"

# P2E IDM inventory

# shellcheck source=experiments/_craftax_blind_idms.sh
source "${REPO_ROOT}/experiments/_craftax_blind_idms.sh"
CRAFTAX_METHODS=("${CRAFTAX_BLIND_IDMS[@]}")
# Allow selection of a single method from the cohort.
: "${IDM_CONFIG:=all}"
if [[ "${IDM_CONFIG}" != "all" ]]; then
  [[ "${IDM_CONFIG}" =~ ^[a-z0-9_]+$ ]] || { echo "Invalid IDM_CONFIG=${IDM_CONFIG}." >&2; exit 2; }
  CRAFTAX_METHODS=("${IDM_CONFIG}")
fi
if [[ "${ALGORITHM}" == "dreamer" ]]; then
  [[ "${IDM_CONFIG}" == "all" || "${IDM_CONFIG}" == "none" ]] || {
    echo "ALGORITHM=dreamer requires IDM_CONFIG=all or none." >&2; exit 2;
  }
  IDM_CONFIG=none
  CRAFTAX_METHODS=(none)
fi

# Slurm resources (not training hyperparameters)

: "${PARTITION:=}"
: "${QOS:=}"
: "${MAX_CONCURRENT_GPUS:=24}"
: "${GRES:=gpu:1}"
: "${CPUS_PER_TASK:=4}"
: "${MEMORY:=128G}"
: "${TIME_LIMIT:=2-00:00:00}"
SLURM_SIGNAL="B:USR1@300"
SLURM_PREEMPTABLE="true"
# Keep source bundles on storage visible to both login and compute nodes.

: "${EXCLUDE_NODES:=}"

usage() {
  cat <<'EOF'
Usage:
  bash experiments/launch_craftax_p2e.sh
  bash experiments/launch_craftax_p2e.sh --list
  bash experiments/launch_craftax_p2e.sh --dry-run

Optional selection overrides:
  ALGORITHM=p2e|dreamer            # default p2e; dreamer runs supervised baselines
  ENV_CONFIG=craftax_classic|craftax_classic_symbolic|craftax_symbolic|craftax_full
  IDM_CONFIG=all|<configs/idm name> # use random for the random-data control
  MODEL_CONFIG=size100M|size200M       # model-size preset
  LEARNER_BATCH_SIZE=auto|16|32|64     # model/IDM-aware default or explicit override
  LEARNER_RSSM_UNROLL=auto|1|2|4|8    # model/IDM-aware default or explicit override
  EXPL_GRAD_MODE=detached|head_only|encoder_only|full
                                      # full lets IDM loss shape encoder/RSSM
  TOTAL_STEPS=10010000                # optional total environment-step target
  STRICT_TASK_BLIND_PRETRAINING=true # P2E only; false for supervised Dreamer
  WANDB_PROJECT=craftax_blind         # defaults to the task-blind cohort project
  ADAPTIVE_CRITIC_ENABLED=true        # enable critic update-rate adaptation
  ADAPTIVE_CRITIC_CRITICS=value|expl_value|both
  ADAPTIVE_CRITIC_SCOPE=output|critic
  ADAPTIVE_CRITIC_GRANULARITY=tensor|feature
  ADAPTIVE_CRITIC_META_LR=0.001       # 0 gives a fixed-multiplier control
  ADAPTIVE_CRITIC_INIT_SCALE=1.0
  BACKEND_AGENT=dreamerv3|r2dreamer|ne_dreamer|dreamerpro
  SEEDS=0:1:2                        # colon-delimited, unique nonnegative ints
  SEED=7                             # shorthand for a single seed
  FORK_FROM_ARRAY_JOB_ID=<parent-array-id>     # checkpoint-fork every matching task
  FORK_FROM_SOURCE_ID=<sha256>       # immutable source ID of the parent array
  FORK_FROM_CLUSTER=slurm            # parent RUN_ID cluster prefix
  MAX_CONCURRENT_GPUS=19             # submission throttle (1-24)
  MEMORY=128G                        # host RAM per array task
  GRES=gpu:1                        # optionally select gpu:<type>:1
  ARRAY_TASKS=5:18:19                # submit only these array indices
  WANDB_RELAUNCH_MODE=fresh|continue # fresh attempts are safe across preemption
  PARTITION=gpu QOS=normal           # optional site-specific scheduler choices
  SSRL_SLURM_WORKDIR=/shared/ssrl-agent  # compute-visible checkout path

Examples:
  ENV_CONFIG=craftax_classic_symbolic bash experiments/launch_craftax_p2e.sh
  BACKEND_AGENT=r2dreamer SEEDS=0:1 bash experiments/launch_craftax_p2e.sh
  SEEDS=0 bash experiments/launch_craftax_p2e.sh    # launch only seed 0
  FORK_FROM_ARRAY_JOB_ID=<parent-array-id> FORK_FROM_SOURCE_ID=<sha256> \
    MAX_CONCURRENT_GPUS=19 bash experiments/launch_craftax_p2e.sh

The seed-0 default submits 21 P2E IDMs as 21 Slurm array elements using the
selected BACKEND_AGENT. ALGORITHM=dreamer instead submits one supervised run
per seed. Craftax Classic pixels trains to 5.01M steps with a
two-day job limit. The `%24` array concurrency cap allows at most 24 concurrent
array elements; configure MAX_CONCURRENT_GPUS for your allocation.
Set video_policy_log: True in the selected environment config to store gameplay
videos; symbolic Craftax configs cap each completed-episode video at 1000 frames.
EOF
}

print_command() {
  printf '%q ' "$@"
  echo
}

resolve_slurm_workdir() {
  local requested="${SSRL_SLURM_WORKDIR:-${REPO_ROOT}}"
  local resolved

  if [[ ! -d "${requested}" ]]; then
    echo "Slurm working directory is not accessible: ${requested}" >&2
    return 1
  fi
  resolved="$(cd -- "${requested}" && pwd -P)" || return 1
  [[ "${resolved}" == /* ]] || {
    echo "Slurm working directory must resolve to an absolute path: ${resolved}" >&2
    return 1
  }
  printf '%s\n' "${resolved}"
}

list_algorithms() {
  local index=0 method seed
  printf '%-7s %-24s %s\n' 'Index' 'Method' 'Seed'
  for method in "${CRAFTAX_METHODS[@]}"; do
    for seed in "${CRAFTAX_SEEDS[@]}"; do
      printf '%-7d %-24s %s\n' "${index}" "${method}" "${seed}"
      index=$((index + 1))
    done
  done
  printf 'Environment config: %s\n' "${ENV_CONFIG}"
}

parse_seeds() {
  local seed
  local -A seen=()

  if [[ ! "${SEEDS}" =~ ^(0|[1-9][0-9]*)(:(0|[1-9][0-9]*))*$ ]]; then
    echo "Invalid SEEDS=${SEEDS}. Use unique nonnegative integers separated by colons, for example 0:1:2." >&2
    exit 2
  fi
  IFS=: read -r -a CRAFTAX_SEEDS <<< "${SEEDS}"
  for seed in "${CRAFTAX_SEEDS[@]}"; do
    if [[ -n "${seen[${seed}]:-}" ]]; then
      echo "Invalid SEEDS=${SEEDS}: duplicate seed ${seed}." >&2
      exit 2
    fi
    seen["${seed}"]=1
  done
}

parse_array_tasks() {
  local index total_jobs
  local -A seen=()
  SELECTED_ARRAY_TASKS=()
  [[ -n "${ARRAY_TASKS}" ]] || return 0
  if [[ ! "${ARRAY_TASKS}" =~ ^(0|[1-9][0-9]*)(:(0|[1-9][0-9]*))*$ ]]; then
    echo "Invalid ARRAY_TASKS=${ARRAY_TASKS}. Use unique array indices separated by colons, for example 5:18:19." >&2
    exit 2
  fi
  IFS=: read -r -a SELECTED_ARRAY_TASKS <<< "${ARRAY_TASKS}"
  total_jobs=$((${#CRAFTAX_METHODS[@]} * ${#CRAFTAX_SEEDS[@]}))
  for index in "${SELECTED_ARRAY_TASKS[@]}"; do
    if [[ -n "${seen[${index}]:-}" ]]; then
      echo "Invalid ARRAY_TASKS=${ARRAY_TASKS}: duplicate index ${index}." >&2
      exit 2
    fi
    if (( index >= total_jobs )); then
      echo "Invalid ARRAY_TASKS=${ARRAY_TASKS}: index ${index} is outside the 0-$((total_jobs - 1)) grid." >&2
      exit 2
    fi
    seen["${index}"]=1
  done
}

validate_selections() {
  local idm
  parse_seeds
  parse_array_tasks

  local expected_blind=false
  [[ "${ALGORITHM}" != "p2e" ]] || expected_blind=true
  if [[ "${STRICT_TASK_BLIND_PRETRAINING,,}" != "${expected_blind}" ]]; then
    echo "${ALGORITHM} requires STRICT_TASK_BLIND_PRETRAINING=${expected_blind}." >&2
    exit 2
  fi
  STRICT_TASK_BLIND_PRETRAINING="${expected_blind}"

  WANDB_RELAUNCH_MODE="${WANDB_RELAUNCH_MODE,,}"
  case "${WANDB_RELAUNCH_MODE}" in
    continue|fresh) ;;
    *)
      echo "Invalid WANDB_RELAUNCH_MODE=${WANDB_RELAUNCH_MODE}; use continue or fresh." >&2
      exit 2
      ;;
  esac

  case "${ENV_CONFIG}" in
    craftax_classic|craftax_classic_symbolic|craftax_symbolic|craftax_full)
      ;;
    *)
      echo "Invalid ENV_CONFIG=${ENV_CONFIG}. Use craftax_classic, craftax_classic_symbolic, craftax_symbolic, or craftax_full." >&2
      exit 2
      ;;
  esac
  if [[ ! -f "${REPO_ROOT}/configs/envs/${ENV_CONFIG}.yaml" ]]; then
    echo "Missing environment config: configs/envs/${ENV_CONFIG}.yaml" >&2
    exit 2
  fi
  if [[ ! "${MODEL_CONFIG}" =~ ^size[0-9]+[KMG]$ ||
        ! -f "${REPO_ROOT}/configs/model/${MODEL_CONFIG}.yaml" ]]; then
    echo "Invalid MODEL_CONFIG=${MODEL_CONFIG}. Select an existing configs/model/size*.yaml preset." >&2
    exit 2
  fi
  if [[ ! "${LEARNER_BATCH_SIZE}" =~ ^(auto|16|32|64)$ ]]; then
    echo "Invalid LEARNER_BATCH_SIZE=${LEARNER_BATCH_SIZE}; use auto, 16, 32, or 64." >&2
    exit 2
  fi
  if [[ ! "${LEARNER_RSSM_UNROLL}" =~ ^(auto|1|2|4|8)$ ]]; then
    echo "Invalid LEARNER_RSSM_UNROLL=${LEARNER_RSSM_UNROLL}; use auto, 1, 2, 4, or 8." >&2
    exit 2
  fi
  if [[ -n "${TOTAL_STEPS}" &&
        ! "${TOTAL_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid TOTAL_STEPS=${TOTAL_STEPS}; use a positive integer." >&2
    exit 2
  fi
  case "${EXPL_GRAD_MODE}" in
    ""|detached|detach|stopgrad|stop_grad|head_only|head|encoder_only|encoder|enc|full|shared) ;;
    *)
      echo "Invalid EXPL_GRAD_MODE=${EXPL_GRAD_MODE}." >&2
      exit 2
      ;;
  esac
  case "${ADAPTIVE_CRITIC_CRITICS}" in
    ""|value|expl_value|both) ;;
    *)
      echo "Invalid ADAPTIVE_CRITIC_CRITICS=${ADAPTIVE_CRITIC_CRITICS}." >&2
      exit 2
      ;;
  esac
  case "${ADAPTIVE_CRITIC_SCOPE}" in
    ""|output|critic) ;;
    *)
      echo "Invalid ADAPTIVE_CRITIC_SCOPE=${ADAPTIVE_CRITIC_SCOPE}." >&2
      exit 2
      ;;
  esac
  case "${ADAPTIVE_CRITIC_GRANULARITY}" in
    ""|tensor|feature) ;;
    *)
      echo "Invalid ADAPTIVE_CRITIC_GRANULARITY=${ADAPTIVE_CRITIC_GRANULARITY}." >&2
      exit 2
      ;;
  esac

  for idm in "${CRAFTAX_METHODS[@]}"; do
    [[ "${ALGORITHM}" != "dreamer" ]] || continue
    if [[ ! -f "${REPO_ROOT}/configs/idm/${idm}.yaml" ]]; then
      echo "Missing IDM config: configs/idm/${idm}.yaml" >&2
      exit 2
    fi
  done

  case "${BACKEND_AGENT}" in
    dreamer|dreamerv3|v3|r2dreamer|r2_dreamer|ne_dreamer|nedreamer|dreamerpro|dreamer_pro)
      ;;
    *)
      echo "Invalid BACKEND_AGENT=${BACKEND_AGENT}." >&2
      exit 2
      ;;
  esac
  if [[ ! "${MAX_CONCURRENT_GPUS}" =~ ^([1-9]|1[0-9]|2[0-4])$ ]]; then
    echo "Invalid MAX_CONCURRENT_GPUS=${MAX_CONCURRENT_GPUS}; use an integer from 1 through 24." >&2
    exit 2
  fi
  if [[ ! "${MEMORY}" =~ ^[1-9][0-9]*[GM]$ ]]; then
    echo "Invalid MEMORY=${MEMORY}; use a positive Slurm memory value such as 128G." >&2
    exit 2
  fi
  ssrl_validate_slurm_options
  if [[ -n "${FORK_FROM_ARRAY_JOB_ID}" || -n "${FORK_FROM_SOURCE_ID}" ]]; then
    if [[ ! "${FORK_FROM_ARRAY_JOB_ID}" =~ ^[1-9][0-9]*$ ]]; then
      echo "FORK_FROM_ARRAY_JOB_ID must be a positive Slurm array job ID." >&2
      exit 2
    fi
    if [[ ! "${FORK_FROM_SOURCE_ID}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "FORK_FROM_SOURCE_ID must be the parent's lowercase SHA-256 source ID." >&2
      exit 2
    fi
    if [[ ! "${FORK_FROM_CLUSTER}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
      echo "FORK_FROM_CLUSTER is not a safe run-ID component." >&2
      exit 2
    fi
  fi
}

resolve_learner_profile() {
  local memory_class="normal"
  local auto_batch auto_unroll

  case "${IDM_TYPE}" in
    max) memory_class="ultra" ;;
    disagreement|cig|dads|rdads|empowerment_process) memory_class="heavy" ;;
  esac

  case "${MODEL_CONFIG}" in
    size1M|size12M|size25M|size50M|size100M)
      # Use batch size 32 for smaller models while retaining the probabilistic ensemble
      # update cadence.
      if [[ "${memory_class}" == ultra ]]; then
        auto_batch=16
        auto_unroll=1
      else
        auto_batch=32
        auto_unroll=8
      fi
      ;;
    size200M)
      case "${memory_class}" in
        ultra)
          auto_batch=16
          auto_unroll=1
          ;;
        heavy)
          auto_batch=32
          auto_unroll=2
          ;;
        normal)
          auto_batch=32
          auto_unroll=4
          ;;
      esac
      ;;
    *)
      # Use a smaller batch for models at or above size400M.
      auto_batch=16
      auto_unroll=1
      ;;
  esac

  BATCH_SIZE="${auto_batch}"
  RSSM_UNROLL="${auto_unroll}"
  [[ "${LEARNER_BATCH_SIZE}" == auto ]] || BATCH_SIZE="${LEARNER_BATCH_SIZE}"
  [[ "${LEARNER_RSSM_UNROLL}" == auto ]] || RSSM_UNROLL="${LEARNER_RSSM_UNROLL}"
  export BATCH_SIZE RSSM_UNROLL
  echo "Learner profile: memory_class=${memory_class} batch_size=${BATCH_SIZE} rssm_unroll=${RSSM_UNROLL}"
}

run_array_task() {
  local index idm_index seed_index seed_count total_jobs
  index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required in worker mode}"
  if [[ ! "${index}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "SLURM_ARRAY_TASK_ID=${index} is not a nonnegative integer." >&2
    exit 2
  fi
  seed_count=${#CRAFTAX_SEEDS[@]}
  total_jobs=$((${#CRAFTAX_METHODS[@]} * seed_count))
  if (( index >= total_jobs )); then
    echo "SLURM_ARRAY_TASK_ID=${index} is outside the ${total_jobs}-run IDM/seed grid." >&2
    exit 2
  fi

  idm_index=$((index / seed_count))
  seed_index=$((index % seed_count))
  SEED="${CRAFTAX_SEEDS[${seed_index}]}"
  IDM_TYPE="${CRAFTAX_METHODS[${idm_index}]}"

  CFG="${ENV_CONFIG}"
  MODEL_SIZE="${MODEL_CONFIG}"
  AGENT="${BACKEND_AGENT}"
  DOMAIN_NAME="craftax"
  case "${ENV_CONFIG}" in
    craftax_classic) TASK_NAME="classic_pixels" ;;
    craftax_classic_symbolic) TASK_NAME="classic_symbolic" ;;
    craftax_symbolic) TASK_NAME="symbolic" ;;
    craftax_full) TASK_NAME="full" ;;
  esac
  ALGO_PRESET="${ALGORITHM}"
  HELPER_DIR="${REPO_ROOT}/experiments"
  if [[ -n "${TOTAL_STEPS}" ]]; then
    STEPS="${TOTAL_STEPS}"
    export STEPS
  fi
  resolve_learner_profile

  echo "Array run ${index}/$((total_jobs - 1)): idm=${IDM_TYPE} seed=${SEED} env=${ENV_CONFIG}"
  echo "Configs: envs=${CFG} algo=${ALGO_PRESET} idm=${IDM_TYPE} model=${MODEL_SIZE} agent=${AGENT}"

  if [[ -n "${FORK_FROM_ARRAY_JOB_ID}" ]]; then
    RUN_STORAGE_MODE="fork"
    RUN_FORK_SOURCE_RUN_UID="${FORK_FROM_CLUSTER}_job${FORK_FROM_ARRAY_JOB_ID}_task${index}"
    RUN_FORK_PARENT_SOURCE_ID="${FORK_FROM_SOURCE_ID}"
    export RUN_STORAGE_MODE RUN_FORK_SOURCE_RUN_UID RUN_FORK_PARENT_SOURCE_ID
  fi

  # shellcheck disable=SC1091
  source "${REPO_ROOT}/experiments/_run_common.sh"
}

submit_array() {
  local seed_count total_jobs submission_jobs last_index array_spec job_name export_vars
  local selected_spec selected_index selected_idm_index selected_idm job_idm
  local submission_source_id storage_helper source_bundle_dir source_tree
  local slurm_workdir slurm_log_root
  local -a command

  if [[ -n "${SLURM_JOB_ID:-}" && "${DRY_RUN:-false}" != "true" ]]; then
    echo "Submit this launcher from a login shell with 'bash', not from inside sbatch." >&2
    exit 2
  fi
  if ! command -v sbatch >/dev/null 2>&1 && [[ "${DRY_RUN:-false}" != "true" ]]; then
    echo "sbatch is unavailable. Run this launcher on a Slurm login node." >&2
    exit 127
  fi

  slurm_workdir="$(resolve_slurm_workdir)" || exit $?
  # Store scheduler output locally and publish it with the run artifacts.
  slurm_log_root="/tmp"
  seed_count=${#CRAFTAX_SEEDS[@]}
  total_jobs=$((${#CRAFTAX_METHODS[@]} * seed_count))
  last_index=$((total_jobs - 1))
  submission_jobs=${total_jobs}
  if [[ "${#SELECTED_ARRAY_TASKS[@]}" -gt 0 ]]; then
    selected_spec="${ARRAY_TASKS//:/,}"
    submission_jobs=${#SELECTED_ARRAY_TASKS[@]}
    array_spec="${selected_spec}%${MAX_CONCURRENT_GPUS}"
  else
    array_spec="0-${last_index}%${MAX_CONCURRENT_GPUS}"
  fi
  job_name="craftax_${ALGORITHM}_${ENV_CONFIG}"
  # Use a distinct Slurm and log prefix for single-method submissions.
  if [[ "${#SELECTED_ARRAY_TASKS[@]}" -gt 0 ]]; then
    job_idm=""
    for selected_index in "${SELECTED_ARRAY_TASKS[@]}"; do
      selected_idm_index=$((selected_index / seed_count))
      selected_idm="${CRAFTAX_METHODS[${selected_idm_index}]}"
      if [[ -z "${job_idm}" ]]; then
        job_idm="${selected_idm}"
      elif [[ "${job_idm}" != "${selected_idm}" ]]; then
        job_idm=""
        break
      fi
    done
    if [[ -n "${job_idm}" ]]; then
      job_name="craftax_${ALGORITHM}_${job_idm}_${ENV_CONFIG}"
    fi
  fi
  storage_helper="${REPO_ROOT}/experiments/run_storage.py"
  if [[ ! -f "${storage_helper}" ]]; then
    echo "Missing source provenance helper: ${storage_helper}" >&2
    exit 1
  fi
  submission_source_id="$(/usr/bin/python3 "${storage_helper}" source-id --repo "${REPO_ROOT}")"
  source_bundle_dir="${SSRL_SOURCE_BUNDLE_ROOT}/${submission_source_id}"
  source_tree="${source_bundle_dir}/tree"
  if [[ "${SSRL_SOURCE_BUNDLE_ROOT}" == *","* ]]; then
    echo "SSRL_SOURCE_BUNDLE_ROOT cannot contain a comma (Slurm --export delimiter)." >&2
    exit 2
  fi
  if [[ "${DRY_RUN:-false}" != "true" ]]; then
    mkdir -p "${SSRL_SOURCE_BUNDLE_ROOT}"
    /usr/bin/python3 "${storage_helper}" source-bundle ensure \
      --repo "${REPO_ROOT}" --bundle-dir "${source_bundle_dir}" \
      --source-id "${submission_source_id}" >/dev/null
  fi

  # Export only experiment selectors so ambient training hyperparameters cannot
  # silently override the selected Hydra configs in _run_common.sh.
  export_vars="SSRL_REPO_ROOT=${source_tree},SSRL_SOURCE_BUNDLE_DIR=${source_bundle_dir},SSRL_SUBMISSION_SOURCE_ID=${submission_source_id},SSRL_REQUIRE_NODE_LOCAL_RUNTIME=true,ALGORITHM=${ALGORITHM},IDM_CONFIG=${IDM_CONFIG},ENV_CONFIG=${ENV_CONFIG},MODEL_CONFIG=${MODEL_CONFIG},BACKEND_AGENT=${BACKEND_AGENT},LEARNER_BATCH_SIZE=${LEARNER_BATCH_SIZE},LEARNER_RSSM_UNROLL=${LEARNER_RSSM_UNROLL},TOTAL_STEPS=${TOTAL_STEPS},SEEDS=${SEEDS},STRICT_TASK_BLIND_PRETRAINING=${STRICT_TASK_BLIND_PRETRAINING},WANDB_PROJECT=${WANDB_PROJECT},WANDB_ENTITY=${WANDB_ENTITY},WANDB_RELAUNCH_MODE=${WANDB_RELAUNCH_MODE},FORK_FROM_ARRAY_JOB_ID=${FORK_FROM_ARRAY_JOB_ID},FORK_FROM_SOURCE_ID=${FORK_FROM_SOURCE_ID},FORK_FROM_CLUSTER=${FORK_FROM_CLUSTER},SLURM_PREEMPTABLE=${SLURM_PREEMPTABLE},SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_MODE=${SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_MODE},USER=${USER},HOME=${HOME}"
  local runtime_key runtime_value
  for runtime_key in PATH VIRTUAL_ENV CONDA_PREFIX LD_LIBRARY_PATH CONDA_EXE CONDA_ENV \
      WANDB_MODE WANDB_BASE_URL SCRATCH_LOG_ROOT EXPERIMENT_ROOT WORK_LOG_ROOT JAX_PREALLOCATE; do
    [[ -v "${runtime_key}" ]] || continue
    runtime_value="${!runtime_key}"
    [[ "${runtime_value}" != *","* && "${runtime_value}" != *$'\n'* ]] || {
      echo "Invalid ${runtime_key}: Slurm exports cannot contain commas/newlines." >&2
      exit 2
    }
    export_vars+=",${runtime_key}=${runtime_value}"
  done
  [[ -z "${WANDB_API_KEY:-}" ]] || export_vars+=",WANDB_API_KEY"
  if [[ -n "${EXPL_GRAD_MODE}" ]]; then
    export_vars+=",EXPL_GRAD_MODE=${EXPL_GRAD_MODE}"
  fi
  local adaptive_var
  for adaptive_var in \
      ADAPTIVE_CRITIC_ENABLED ADAPTIVE_CRITIC_CRITICS \
      ADAPTIVE_CRITIC_SCOPE ADAPTIVE_CRITIC_GRANULARITY \
      ADAPTIVE_CRITIC_META_LR ADAPTIVE_CRITIC_TRACE_DECAY \
      ADAPTIVE_CRITIC_INIT_SCALE ADAPTIVE_CRITIC_MIN_SCALE \
      ADAPTIVE_CRITIC_MAX_SCALE ADAPTIVE_CRITIC_WARMUP \
      ADAPTIVE_CRITIC_EPS; do
    if [[ -n "${!adaptive_var}" ]]; then
      export_vars+=",${adaptive_var}=${!adaptive_var}"
    fi
  done

  command=(
    sbatch
    "--job-name=${job_name}"
    "--array=${array_spec}"
    "--time=${TIME_LIMIT}"
    "--cpus-per-task=${CPUS_PER_TASK}"
    "--mem=${MEMORY}"
    "--gres=${GRES}"
    "--chdir=${slurm_workdir}"
    "--output=${slurm_log_root}/ssrl-slurm-%u-%x-%A_%a.out"
    "--error=${slurm_log_root}/ssrl-slurm-%u-%x-%A_%a.err"
    "--signal=${SLURM_SIGNAL}"
    "--requeue"
    "--open-mode=append"
    "--export=${export_vars}"
  )
  [[ -z "${PARTITION}" ]] || command+=("--partition=${PARTITION}")
  [[ -z "${QOS}" ]] || command+=("--qos=${QOS}")
  if [[ -n "${EXCLUDE_NODES}" ]]; then
    command+=("--exclude=${EXCLUDE_NODES}")
  fi
  # Slurm must spool the launcher from the same frozen tree used by workers;
  # otherwise an edit between source capture and sbatch could mix versions.
  command+=("${source_tree}/experiments/launch_craftax_p2e.sh" --worker)

  echo "Submitting ${submission_jobs} Craftax ${ALGORITHM} jobs as array ${array_spec}:"
  echo "  envs=${ENV_CONFIG} agent=${BACKEND_AGENT} model=${MODEL_CONFIG}"
  echo "  learner_batch_size=${LEARNER_BATCH_SIZE} learner_rssm_unroll=${LEARNER_RSSM_UNROLL}"
  echo "  strict_task_blind_pretraining=${STRICT_TASK_BLIND_PRETRAINING}"
  if [[ -n "${TOTAL_STEPS}" ]]; then
    echo "  total_steps=${TOTAL_STEPS}"
  fi
  if [[ -n "${EXPL_GRAD_MODE}" ]]; then
    echo "  expl_grad_mode=${EXPL_GRAD_MODE}"
  fi
  if [[ -n "${ADAPTIVE_CRITIC_ENABLED}" ]]; then
    echo "  adaptive_critic=${ADAPTIVE_CRITIC_ENABLED} critics=${ADAPTIVE_CRITIC_CRITICS:-config} scope=${ADAPTIVE_CRITIC_SCOPE:-config} granularity=${ADAPTIVE_CRITIC_GRANULARITY:-config}"
  fi
  echo "  seeds=${CRAFTAX_SEEDS[*]} (${#CRAFTAX_METHODS[@]} methods x ${seed_count} seeds)"
  echo "  partition=${PARTITION} qos=${QOS} max_concurrent_gpus=${MAX_CONCURRENT_GPUS}"
  echo "  gres=${GRES} node_local_slurm_logs=${slurm_log_root}"
  echo "  slurm_workdir=${slurm_workdir}"
  echo "  source_id=${submission_source_id}"
  echo "  source_bundle=${source_bundle_dir}"
  if [[ -n "${FORK_FROM_ARRAY_JOB_ID}" ]]; then
    echo "  checkpoint_fork=${FORK_FROM_CLUSTER}_job${FORK_FROM_ARRAY_JOB_ID}_task<index>"
    echo "  parent_source_id=${FORK_FROM_SOURCE_ID}"
  fi
  echo "  wandb.project=${WANDB_PROJECT} wandb.entity=${WANDB_ENTITY}"
  echo "  wandb.relaunch_mode=${WANDB_RELAUNCH_MODE}"
  list_algorithms
  echo

  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    print_command "${command[@]}"
  else
    "${command[@]}"
  fi
}

case "${1:-}" in
  -h|--help)
    usage
    ;;
  --list)
    validate_selections
    list_algorithms
    ;;
  --worker)
    validate_selections
    run_array_task
    ;;
  "")
    validate_selections
    submit_array
    ;;
  *)
    echo "Unknown argument: ${1}" >&2
    usage >&2
    exit 2
    ;;
esac
