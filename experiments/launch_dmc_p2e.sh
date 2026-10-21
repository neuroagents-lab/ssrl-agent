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
}

stage_submitted_source

# Workers use the submitted source bundle because sbatch executes a spool copy of this
# launcher.
if [[ -n "${SSRL_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${SSRL_REPO_ROOT}"
  SCRIPT_DIR="${REPO_ROOT}/experiments"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

source "${REPO_ROOT}/experiments/_runtime_defaults.sh"

# Suite-wide experiment selections
# These are the only experiment selections made by this launcher. Training
# schedules, batch sizes, replay settings, and all other hyperparameters come
# from configs/{envs,algo,idm,model}.

: "${DMC_TASK_SET:=control_suite}"
: "${DMC_LAUNCHER_DISPLAY:=launch_dmc_p2e.sh}"

# Swap among: dmc_proprio, dmc_vision, dmc_full.
# dmc_proprio selects MuJoCo Playground/MJX in its config. Vision and full use
# dm_control because the pinned Playground version only supports cartpole
# vision and does not expose combined image + proprio observations suite-wide.
: "${ENV_CONFIG:=dmc_proprio}"

# Default to Plan2Explore; ALGORITHM=dreamer selects the supervised baseline.
: "${ALGORITHM:=p2e}"

# Strict task blindness isolates task reward from the learned representation
# and exploration policy; observations and detached task evaluation stay intact.
if [[ "${ALGORITHM}" == "p2e" ]]; then
  : "${STRICT_TASK_BLIND_PRETRAINING:=true}"
else
  : "${STRICT_TASK_BLIND_PRETRAINING:=false}"
fi

# Used only when ALGORITHM=p2e. This must name configs/idm/<name>.yaml, for
# example: disagreement, cig, rnd, simple_curiosity, icm, apt, re3, csf, or ctec.
: "${IDM_CONFIG:=vic}"

# Choose the representation/backend agent for every task.
# Allowed values: dreamerv3, r2dreamer, ne_dreamer, dreamerpro.
: "${BACKEND_AGENT:=dreamerv3}"

# Use colon-delimited SEEDS; commas delimit Slurm export values.
SEEDS="${SEEDS:-${SEED:-0}}"
: "${ARRAY_TASKS:=}"

# Model preset is part of the immutable run identity and worker selection.
: "${MODEL_CONFIG:=size12M}"
if [[ "${ALGORITHM}" == "p2e" ]]; then
  if [[ "${STRICT_TASK_BLIND_PRETRAINING,,}" == "true" ]]; then
    : "${WANDB_PROJECT:=dmc_${IDM_CONFIG}_blind}"
  else
    : "${WANDB_PROJECT:=${ENV_CONFIG}_${IDM_CONFIG}}"
  fi
else
  : "${WANDB_PROJECT:=${ENV_CONFIG}}"
fi
# Resume local training state with a new W&B attempt in the same logical group.
: "${WANDB_RELAUNCH_MODE:=fresh}"
: "${WANDB_ENTITY:=}"
: "${WANDB_MODE:=offline}"
# Freeze runtime/account/path choices alongside experiment selectors. Do not
# export ALL: ambient training hyperparameters must not enter the workers.
DMC_RUNTIME_EXPORTS=(
  PATH VIRTUAL_ENV CONDA_PREFIX CONDA_EXE CONDA_ENV LD_LIBRARY_PATH WANDB_ENTITY WANDB_MODE WANDB_BASE_URL
  SCRATCH_LOG_ROOT EXPERIMENT_ROOT WORK_LOG_ROOT JAX_PREALLOCATE
)
# Advance the shared progress marker only after checkpoint payload publication.
: "${SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_MODE:=published}"
: "${RUN_STORAGE_MODE:=auto}"
: "${RUN_FORK_SOURCE_RUN_UID:=}"
: "${RUN_FORK_PARENT_SOURCE_ID:=}"
: "${DMC_JOB_LABEL:=}"
: "${DMC_VIDEO_POLICY_LOG:=}"
: "${DMC_VIDEO_PRED_LOG:=}"
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

# Explicit DMC task inventories
# Dedicated launchers select smaller dm_control task sets; the regular DMC
# launcher retains the 25-task control-suite inventory.

case "${DMC_TASK_SET}" in
  control_suite)
    DMC_TASKS=(
      "acrobot swingup"
      "acrobot swingup_sparse"
      "ball_in_cup catch"
      "cartpole balance"
      "cartpole balance_sparse"
      "cartpole swingup"
      "cartpole swingup_sparse"
      "cheetah run"
      "finger spin"
      "finger turn_easy"
      "finger turn_hard"
      "fish swim"
      "hopper hop"
      "hopper stand"
      "humanoid stand"
      "humanoid walk"
      "humanoid run"
      "pendulum swingup"
      "point_mass easy"
      "reacher easy"
      "reacher hard"
      "swimmer swimmer6"
      "walker run"
      "walker stand"
      "walker walk"
    )
    ;;
  manipulation)
    DMC_TASKS=(
      "manip lift_brick"
      "manip place_cradle"
      "manip stack_3_bricks"
    )
    ;;
  place_cradle_vision)
    DMC_TASKS=(
      "manip place_cradle"
    )
    ;;
  quadruped_escape)
    DMC_TASKS=(
      "quadruped escape"
    )
    ;;
  walker_multitask)
    DMC_TASKS=(
      "walker stand"
    )
    ;;
  *)
    echo "Invalid DMC_TASK_SET=${DMC_TASK_SET}." >&2
    exit 2
    ;;
esac

# Slurm resources (not training hyperparameters)

# MAX_CONCURRENT_GPUS limits simultaneous elements in each task-specific array.
: "${PARTITION:=}"
: "${QOS:=}"
: "${MAX_CONCURRENT_GPUS:=24}"
: "${GRES:=gpu:1}"
: "${CPUS_PER_TASK:=16}"
: "${MEMORY:=64G}"
: "${TIME_LIMIT:=2-00:00:00}"
SLURM_SIGNAL="B:USR1@300"
: "${SLURM_PREEMPTABLE:=true}"
# Keep source bundles on storage visible to both login and compute nodes.

# Optional exclusions are supplied by the cluster administrator or caller.
: "${EXCLUDE_NODES:=}"

usage() {
  cat <<EOF
Usage:
  bash experiments/${DMC_LAUNCHER_DISPLAY}
  bash experiments/${DMC_LAUNCHER_DISPLAY} --list
  bash experiments/${DMC_LAUNCHER_DISPLAY} --dry-run

Optional selection overrides:
  ENV_CONFIG=dmc_proprio|dmc_vision|dmc_full
  ALGORITHM=dreamer|p2e
  STRICT_TASK_BLIND_PRETRAINING=true|false  # default true for P2E; false for Dreamer
  IDM_CONFIG=<configs/idm name>       # used only for p2e
  BACKEND_AGENT=dreamerv3|r2dreamer|ne_dreamer|dreamerpro
  MODEL_CONFIG=size12M|size200M       # existing model-size preset
  SEEDS=0:1:2                        # colon-delimited, unique nonnegative ints
  SEED=7                             # shorthand for a single seed
  ARRAY_TASKS=5:8:22                 # optional sparse replacement indices
  MAX_CONCURRENT_GPUS=24             # per-task-array throttle
  MEMORY=32G                         # host RAM requested per task
  GRES=gpu:1                        # optionally select gpu:<type>:1
  JAX_PREALLOCATE=false              # optional JAX GPU allocation override
  ADAPTIVE_CRITIC_ENABLED=true       # enable critic update-rate adaptation
  ADAPTIVE_CRITIC_CRITICS=value      # Dreamer exploitation critic
  ADAPTIVE_CRITIC_SCOPE=output|critic
  ADAPTIVE_CRITIC_GRANULARITY=feature|tensor
  ADAPTIVE_CRITIC_META_LR=0.001      # 0 gives a fixed-multiplier control
  ADAPTIVE_CRITIC_INIT_SCALE=1.0
  DMC_VIDEO_POLICY_LOG=true|false    # optional explicit Hydra override
  DMC_VIDEO_PRED_LOG=true|false      # optional explicit Hydra override
  WANDB_PROJECT=<project>            # optional explicit project override
  WANDB_ENTITY=your-team             # optional; authenticate with wandb login
  WANDB_MODE=online|offline|disabled
  CONDA_EXE=/path/to/miniconda3/bin/conda CONDA_ENV=your-environment
  WORK_LOG_ROOT=/path/to/shared/training_logs
  SCRATCH_LOG_ROOT=/path/to/node-local/training_logs
  WANDB_RELAUNCH_MODE=fresh|continue # fresh attempts are safe across preemption
  PARTITION=gpu QOS=normal           # optional site-specific scheduler choices
  SSRL_SLURM_WORKDIR=/shared/ssrl-agent  # compute-visible checkout path

Examples:
  IDM_CONFIG=cig bash experiments/${DMC_LAUNCHER_DISPLAY}
  STRICT_TASK_BLIND_PRETRAINING=true IDM_CONFIG=disagreement \\
    bash experiments/${DMC_LAUNCHER_DISPLAY}
  ALGORITHM=dreamer bash experiments/${DMC_LAUNCHER_DISPLAY}
  ENV_CONFIG=dmc_vision bash experiments/${DMC_LAUNCHER_DISPLAY}
  ENV_CONFIG=dmc_full BACKEND_AGENT=r2dreamer bash experiments/${DMC_LAUNCHER_DISPLAY}

The default submits Plan2Explore on ${#DMC_TASKS[@]} tasks x 1 seed =
${#DMC_TASKS[@]} Slurm array elements. Slurm starts pending elements as GPUs
become available.
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

list_tasks() {
  local index=0 task domain name seed
  printf '%-7s %-32s %s\n' 'Index' 'Task' 'Seed'
  for task in "${DMC_TASKS[@]}"; do
    read -r domain name <<< "${task}"
    for seed in "${DMC_SEEDS[@]}"; do
      printf '%-7d %-32s %s\n' "${index}" "dmc_${domain}_${name}" "${seed}"
      index=$((index + 1))
    done
  done
}

parse_seeds() {
  local seed
  local -A seen=()

  if [[ ! "${SEEDS}" =~ ^(0|[1-9][0-9]*)(:(0|[1-9][0-9]*))*$ ]]; then
    echo "Invalid SEEDS=${SEEDS}. Use unique nonnegative integers separated by colons, for example 0:1:2." >&2
    exit 2
  fi
  IFS=: read -r -a DMC_SEEDS <<< "${SEEDS}"
  for seed in "${DMC_SEEDS[@]}"; do
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
    echo "Invalid ARRAY_TASKS=${ARRAY_TASKS}. Use unique array indices separated by colons, for example 5:8:22." >&2
    exit 2
  fi
  IFS=: read -r -a SELECTED_ARRAY_TASKS <<< "${ARRAY_TASKS}"
  total_jobs=$((${#DMC_TASKS[@]} * ${#DMC_SEEDS[@]}))
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
  local runtime_var runtime_value
  for runtime_var in "${DMC_RUNTIME_EXPORTS[@]}"; do
    runtime_value="${!runtime_var:-}"
    if [[ "${runtime_value}" == *","* || "${runtime_value}" == *$'\n'* ||
          "${runtime_value}" == *$'\r'* ]]; then
      echo "${runtime_var} cannot contain commas or newlines (Slurm --export delimiter)." >&2
      exit 2
    fi
  done
  case "${JAX_PREALLOCATE:-}" in
    ""|true|false) ;;
    *)
      echo "Invalid JAX_PREALLOCATE=${JAX_PREALLOCATE}; use true or false." >&2
      exit 2
      ;;
  esac
  parse_seeds
  parse_array_tasks

  STRICT_TASK_BLIND_PRETRAINING="${STRICT_TASK_BLIND_PRETRAINING,,}"
  case "${STRICT_TASK_BLIND_PRETRAINING}" in
    true|false) ;;
    *)
      echo "Invalid STRICT_TASK_BLIND_PRETRAINING=${STRICT_TASK_BLIND_PRETRAINING}; use true or false." >&2
      exit 2
      ;;
  esac
  if [[ "${STRICT_TASK_BLIND_PRETRAINING}" == "true" &&
        "${ALGORITHM}" != "p2e" ]]; then
    echo "STRICT_TASK_BLIND_PRETRAINING=true requires ALGORITHM=p2e." >&2
    exit 2
  fi

  if [[ "${DMC_TASK_SET}" == "walker_multitask" ]]; then
    if [[ "${ENV_CONFIG}" != "dmc_walker_multitask" &&
          "${ENV_CONFIG}" != "dmc_walker_multitask_vision" ]]; then
      echo "DMC_TASK_SET=walker_multitask requires ENV_CONFIG=dmc_walker_multitask or dmc_walker_multitask_vision." >&2
      exit 2
    fi
    if [[ "${ALGORITHM}" == "p2e" && "${STRICT_TASK_BLIND_PRETRAINING}" != "true" ]]; then
      echo "Walker multitask P2E requires STRICT_TASK_BLIND_PRETRAINING=true." >&2
      exit 2
    fi
  elif [[ ! "${ENV_CONFIG}" =~ ^dmc_(proprio|vision|full|walker_scratch_(proprio|vision))$ ]]; then
    echo "Invalid ENV_CONFIG=${ENV_CONFIG}. Use dmc_proprio, dmc_vision, dmc_full, dmc_walker_scratch_proprio, or dmc_walker_scratch_vision." >&2
    exit 2
  fi
  if [[ ! -f "${REPO_ROOT}/configs/envs/${ENV_CONFIG}.yaml" ]]; then
    echo "Missing environment config: configs/envs/${ENV_CONFIG}.yaml" >&2
    exit 2
  fi
  if [[ ! "${MODEL_CONFIG}" =~ ^size[0-9]+[KMG]$ ||
        ! -f "${REPO_ROOT}/configs/model/${MODEL_CONFIG}.yaml" ]]; then
    echo "Invalid MODEL_CONFIG=${MODEL_CONFIG}; select an existing size preset." >&2
    exit 2
  fi

  case "${ALGORITHM}" in
    dreamer)
      ;;
    p2e)
      if [[ ! "${IDM_CONFIG}" =~ ^[a-z0-9_]+$ ]]; then
        echo "Invalid IDM_CONFIG=${IDM_CONFIG}." >&2
        exit 2
      fi
      if [[ ! -f "${REPO_ROOT}/configs/idm/${IDM_CONFIG}.yaml" ]]; then
        echo "Missing IDM config: configs/idm/${IDM_CONFIG}.yaml" >&2
        exit 2
      fi
      ;;
    *)
      echo "Invalid ALGORITHM=${ALGORITHM}. Use dreamer or p2e." >&2
      exit 2
      ;;
  esac

  case "${BACKEND_AGENT}" in
    dreamer|dreamerv3|v3|r2dreamer|r2_dreamer|ne_dreamer|nedreamer|dreamerpro|dreamer_pro)
      ;;
    *)
      echo "Invalid BACKEND_AGENT=${BACKEND_AGENT}." >&2
      exit 2
      ;;
  esac

  if [[ ! "${MEMORY}" =~ ^[1-9][0-9]*[GM]$ ]]; then
    echo "Invalid MEMORY=${MEMORY}; use a positive Slurm memory value such as 32G." >&2
    exit 2
  fi
  case "${SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_MODE}" in
    synchronous|published) ;;
    *)
      echo "Invalid SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_MODE=${SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_MODE}." >&2
      exit 2
      ;;
  esac
  if [[ ! "${MAX_CONCURRENT_GPUS}" =~ ^([1-9]|1[0-9]|2[0-4])$ ]]; then
    echo "Invalid MAX_CONCURRENT_GPUS=${MAX_CONCURRENT_GPUS}; use an integer from 1 through 24." >&2
    exit 2
  fi
  ssrl_validate_slurm_options

  case "${ADAPTIVE_CRITIC_ENABLED,,}" in
    ""|true|false) ;;
    *)
      echo "Invalid ADAPTIVE_CRITIC_ENABLED=${ADAPTIVE_CRITIC_ENABLED}; use true or false." >&2
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
    ""|feature|tensor) ;;
    *)
      echo "Invalid ADAPTIVE_CRITIC_GRANULARITY=${ADAPTIVE_CRITIC_GRANULARITY}." >&2
      exit 2
      ;;
  esac
  local video_setting video_value
  for video_setting in DMC_VIDEO_POLICY_LOG DMC_VIDEO_PRED_LOG; do
    video_value="${!video_setting,,}"
    case "${video_value}" in
      ""|true|false) ;;
      *)
        echo "Invalid ${video_setting}=${!video_setting}; use true or false." >&2
        exit 2
        ;;
    esac
  done
  if [[ -n "${DMC_JOB_LABEL}" &&
        ! "${DMC_JOB_LABEL}" =~ ^[a-z0-9][a-z0-9_]{0,63}$ ]]; then
    echo "Invalid DMC_JOB_LABEL=${DMC_JOB_LABEL}." >&2
    exit 2
  fi

  WANDB_RELAUNCH_MODE="${WANDB_RELAUNCH_MODE,,}"
  case "${WANDB_RELAUNCH_MODE}" in
    fresh|continue) ;;
    *)
      echo "Invalid WANDB_RELAUNCH_MODE=${WANDB_RELAUNCH_MODE}. Use fresh or continue." >&2
      exit 2
      ;;
  esac

  RUN_STORAGE_MODE="${RUN_STORAGE_MODE,,}"
  case "${RUN_STORAGE_MODE}" in
    auto|new|fresh|resume)
      if [[ -n "${RUN_FORK_SOURCE_RUN_UID}" ||
            -n "${RUN_FORK_PARENT_SOURCE_ID}" ]]; then
        echo "RUN_FORK_SOURCE_RUN_UID and RUN_FORK_PARENT_SOURCE_ID require RUN_STORAGE_MODE=fork." >&2
        exit 2
      fi
      ;;
    fork)
      if [[ ! "${RUN_FORK_SOURCE_RUN_UID}" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$ ||
            ! "${RUN_FORK_PARENT_SOURCE_ID}" =~ ^[0-9a-f]{64}$ ]]; then
        echo "RUN_STORAGE_MODE=fork requires a safe RUN_FORK_SOURCE_RUN_UID and 64-character RUN_FORK_PARENT_SOURCE_ID." >&2
        exit 2
      fi
      ;;
    *)
      echo "Invalid RUN_STORAGE_MODE=${RUN_STORAGE_MODE}. Use auto, new, fresh, resume, or fork." >&2
      exit 2
      ;;
  esac
}

run_array_task() {
  local index task_index seed_index seed_count total_jobs domain_name task_name
  index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required in worker mode}"
  if [[ ! "${index}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "SLURM_ARRAY_TASK_ID=${index} is not a nonnegative integer." >&2
    exit 2
  fi
  seed_count=${#DMC_SEEDS[@]}
  total_jobs=$((${#DMC_TASKS[@]} * seed_count))
  if (( index >= total_jobs )); then
    echo "SLURM_ARRAY_TASK_ID=${index} is outside the ${total_jobs}-run task/seed grid." >&2
    exit 2
  fi
  task_index=$((index / seed_count))
  seed_index=$((index % seed_count))
  SEED="${DMC_SEEDS[${seed_index}]}"
  read -r domain_name task_name <<< "${DMC_TASKS[${task_index}]}"

  # _run_common.sh performs MJX CUDA-library and Madrona heap setup before
  # Hydra is composed, so the backend must be known before it is sourced.
  case "${DMC_TASK_SET}" in
    manipulation|place_cradle_vision|quadruped_escape|walker_multitask)
      # These task sets require native dm_control. Walker multitask records
      # native stand/walk/run rewards as learner-only replay labels.
      DMC_BACKEND="dm_control"
      ;;
    *)
      case "${ENV_CONFIG}" in
        dmc_proprio) DMC_BACKEND="mjx" ;;
        dmc_vision|dmc_full|dmc_walker_scratch_proprio|dmc_walker_scratch_vision) DMC_BACKEND="dm_control" ;;
      esac
      ;;
  esac

  # Select Hydra config groups and experiment identity.
  CFG="${ENV_CONFIG}"
  MODEL_SIZE="${MODEL_CONFIG}"
  AGENT="${BACKEND_AGENT}"
  DOMAIN_NAME="${domain_name}"
  TASK_NAME="${task_name}"
  HELPER_DIR="${REPO_ROOT}/experiments"

  if [[ "${DMC_TASK_SET}" == "walker_multitask" ]]; then
    # Use greedy evaluation for the concurrent Walker task heads.
    EVAL_ACTION_SELECTIONS="[greedy]"
  fi

  if [[ "${ALGORITHM}" == "dreamer" ]]; then
    ALGO_PRESET="dreamer"
    IDM_TYPE="none"
  else
    ALGO_PRESET="p2e"
    IDM_TYPE="${IDM_CONFIG}"
  fi
  if [[ -n "${DMC_VIDEO_POLICY_LOG}" ]]; then
    VIDEO_POLICY_LOG="${DMC_VIDEO_POLICY_LOG,,}"
  fi
  if [[ -n "${DMC_VIDEO_PRED_LOG}" ]]; then
    VIDEO_PRED_LOG="${DMC_VIDEO_PRED_LOG,,}"
  fi

  echo "Array run ${index}/$((total_jobs - 1)): task=${task_index} dmc_${domain_name}_${task_name} seed=${SEED}"
  echo "Configs: envs=${CFG} backend=${DMC_BACKEND} algo=${ALGO_PRESET} idm=${IDM_TYPE} model=${MODEL_SIZE} agent=${AGENT}"
  echo "Strict task-blind pretraining: ${STRICT_TASK_BLIND_PRETRAINING}"

  # shellcheck disable=SC1091
  source "${REPO_ROOT}/experiments/_run_common.sh"
}

submit_array() {
  local seed_count total_jobs submission_jobs submission_arrays
  local index task_index first_index last_index array_spec job_name export_vars resolved_idm job_label
  local domain_name task_name
  local submission_source_id storage_helper source_bundle_dir source_tree
  local slurm_workdir slurm_log_root
  local -A task_array_specs=()
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
  slurm_log_root="${slurm_workdir}/local_data/slurm"
  mkdir -p "${slurm_log_root}/out" "${slurm_log_root}/err"
  seed_count=${#DMC_SEEDS[@]}
  total_jobs=$((${#DMC_TASKS[@]} * seed_count))
  submission_jobs=${total_jobs}
  if [[ "${#SELECTED_ARRAY_TASKS[@]}" -gt 0 ]]; then
    submission_jobs=${#SELECTED_ARRAY_TASKS[@]}
    for index in "${SELECTED_ARRAY_TASKS[@]}"; do
      task_index=$((index / seed_count))
      if [[ -n "${task_array_specs[${task_index}]:-}" ]]; then
        task_array_specs["${task_index}"]+=",${index}"
      else
        task_array_specs["${task_index}"]="${index}"
      fi
    done
  else
    for task_index in "${!DMC_TASKS[@]}"; do
      first_index=$((task_index * seed_count))
      last_index=$((first_index + seed_count - 1))
      if (( first_index == last_index )); then
        task_array_specs["${task_index}"]="${first_index}"
      else
        task_array_specs["${task_index}"]="${first_index}-${last_index}"
      fi
    done
  fi
  submission_arrays=${#task_array_specs[@]}
  resolved_idm="${IDM_CONFIG}"
  if [[ "${ALGORITHM}" == "dreamer" ]]; then
    resolved_idm="none"
  fi
  job_label="${resolved_idm}"
  if [[ -n "${DMC_JOB_LABEL}" ]]; then
    job_label="${DMC_JOB_LABEL}"
  fi
  if [[ "${STRICT_TASK_BLIND_PRETRAINING}" == "true" ]]; then
    job_label="blind_${job_label}"
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
  # Export only experiment selectors. In particular, ambient STEPS,
  # BATCH_SIZE, TRAIN_RATIO, etc. cannot leak into _run_common.sh and override
  # the Hydra config files.
  export_vars="SSRL_REPO_ROOT=${source_tree},SSRL_SOURCE_BUNDLE_DIR=${source_bundle_dir},SSRL_SUBMISSION_SOURCE_ID=${submission_source_id},DMC_TASK_SET=${DMC_TASK_SET},ENV_CONFIG=${ENV_CONFIG},MODEL_CONFIG=${MODEL_CONFIG},ALGORITHM=${ALGORITHM},IDM_CONFIG=${IDM_CONFIG},BACKEND_AGENT=${BACKEND_AGENT},SEEDS=${SEEDS},WANDB_PROJECT=${WANDB_PROJECT},WANDB_RELAUNCH_MODE=${WANDB_RELAUNCH_MODE},RUN_STORAGE_MODE=${RUN_STORAGE_MODE},RUN_FORK_SOURCE_RUN_UID=${RUN_FORK_SOURCE_RUN_UID},RUN_FORK_PARENT_SOURCE_ID=${RUN_FORK_PARENT_SOURCE_ID},SLURM_PREEMPTABLE=${SLURM_PREEMPTABLE},SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_MODE=${SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_MODE},USER=${USER},HOME=${HOME}"
  export_vars+=",STRICT_TASK_BLIND_PRETRAINING=${STRICT_TASK_BLIND_PRETRAINING}"
  local runtime_var
  for runtime_var in "${DMC_RUNTIME_EXPORTS[@]}"; do
    if [[ -n "${!runtime_var:-}" ]]; then
      export_vars+=",${runtime_var}=${!runtime_var}"
    fi
  done
  # Slurm copies a named variable from sbatch's environment. Never put the key
  # itself in the command line, dry-run output, or frozen source bundle.
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY
    export_vars+=",WANDB_API_KEY"
  fi
  if [[ -n "${DMC_JOB_LABEL}" ]]; then
    export_vars+=",DMC_JOB_LABEL=${DMC_JOB_LABEL}"
  fi
  if [[ -n "${DMC_VIDEO_POLICY_LOG}" ]]; then
    export_vars+=",DMC_VIDEO_POLICY_LOG=${DMC_VIDEO_POLICY_LOG}"
  fi
  if [[ -n "${DMC_VIDEO_PRED_LOG}" ]]; then
    export_vars+=",DMC_VIDEO_PRED_LOG=${DMC_VIDEO_PRED_LOG}"
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

  echo "Submitting ${submission_jobs} DMC jobs as ${submission_arrays} task-specific arrays:"
  echo "  envs=${ENV_CONFIG} algorithm=${ALGORITHM} idm=${resolved_idm} agent=${BACKEND_AGENT} model=${MODEL_CONFIG}"
  echo "  strict_task_blind_pretraining=${STRICT_TASK_BLIND_PRETRAINING}"
  echo "  seeds=${DMC_SEEDS[*]} (${#DMC_TASKS[@]} tasks x ${seed_count} seeds)"
  echo "  partition=${PARTITION} qos=${QOS} per_task_array_throttle=${MAX_CONCURRENT_GPUS}"
  echo "  resources=${GRES} cpus=${CPUS_PER_TASK} memory=${MEMORY} time=${TIME_LIMIT}"
  echo "  slurm_workdir=${slurm_workdir}"
  echo "  source_id=${submission_source_id}"
  echo "  source_bundle=${source_bundle_dir}"
  echo "  wandb.project=${WANDB_PROJECT}"
  echo "  wandb.entity=${WANDB_ENTITY} mode=${WANDB_MODE}"
  echo "  wandb.relaunch_mode=${WANDB_RELAUNCH_MODE}"
  echo "  conda.env=${CONDA_ENV:-active environment}"
  echo "  scratch_logs=${SCRATCH_LOG_ROOT}"
  echo "  training_logs=${WORK_LOG_ROOT}"
  if [[ -n "${ADAPTIVE_CRITIC_ENABLED}" ]]; then
    echo "  adaptive_critic=${ADAPTIVE_CRITIC_ENABLED} critics=${ADAPTIVE_CRITIC_CRITICS:-config} scope=${ADAPTIVE_CRITIC_SCOPE:-config} granularity=${ADAPTIVE_CRITIC_GRANULARITY:-config}"
  fi
  list_tasks
  echo

  for task_index in "${!DMC_TASKS[@]}"; do
    [[ -n "${task_array_specs[${task_index}]:-}" ]] || continue
    read -r domain_name task_name <<< "${DMC_TASKS[${task_index}]}"
    job_name="dmc_${job_label}_${domain_name}_${task_name}"
    array_spec="${task_array_specs[${task_index}]}%${MAX_CONCURRENT_GPUS}"
    command=(
      sbatch
      "--job-name=${job_name}"
      "--array=${array_spec}"
      "--time=${TIME_LIMIT}"
      "--cpus-per-task=${CPUS_PER_TASK}"
      "--mem=${MEMORY}"
      "--gres=${GRES}"
      "--chdir=${slurm_workdir}"
      "--output=${slurm_log_root}/out/slurm-%x-%A_%a.out"
      "--error=${slurm_log_root}/err/slurm-%x-%A_%a.err"
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
    command+=("${source_tree}/experiments/launch_dmc_p2e.sh" --worker)

    if [[ "${DRY_RUN:-false}" == "true" ]]; then
      print_command "${command[@]}"
    else
      "${command[@]}"
    fi
  done
}

case "${1:-}" in
  -h|--help)
    usage
    ;;
  --list)
    validate_selections
    list_tasks
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
