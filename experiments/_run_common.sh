#!/bin/bash

# Shared launcher for experiment presets.
# Representation defaults live in configs/algo/p2e.yaml. Wrappers can still
# override PLAN_REPR_SOURCE / EXPL_REPR_MODE / EXPL_GRAD_MODE directly, and may
# set EXPL_IMAG_REPR_MODE when using expl_repr_mode=branch/parallel.

set -euo pipefail
umask 0007

SSRL_RUN_REPO_ROOT="${SSRL_REPO_ROOT:-${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
source "${SSRL_RUN_REPO_ROOT}/experiments/_runtime_defaults.sh"

# Use the staged source tree as the working directory to avoid shared filesystem
# outages.
if [[ -n "${SLURM_JOB_ID:-}" && -n "${SSRL_REPO_ROOT:-}" ]]; then
  cd -- "${SSRL_RUN_REPO_ROOT}" || {
    echo "Could not enter staged source tree: ${SSRL_RUN_REPO_ROOT}" >&2
    exit 1
  }
fi

# Store disposable caches and temporary files on node-local storage.
if [[ -n "${SLURM_JOB_ID:-}" && "${DRY_RUN:-false}" != "true" ]]; then
  _ssrl_local_base="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
  _ssrl_local_key="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID:-0}"
  : "${SSRL_LOCAL_RUNTIME_ROOT:=${_ssrl_local_base}/ssrl-${USER}/runtime/${_ssrl_local_key}}"
  mkdir -p -- \
    "${SSRL_LOCAL_RUNTIME_ROOT}/tmp" \
    "${SSRL_LOCAL_RUNTIME_ROOT}/cache" \
    "${SSRL_LOCAL_RUNTIME_ROOT}/wandb" \
    "${SSRL_LOCAL_RUNTIME_ROOT}/wandb/cache" \
    "${SSRL_LOCAL_RUNTIME_ROOT}/wandb/data" \
    "${SSRL_LOCAL_RUNTIME_ROOT}/wandb/artifacts" \
    "${SSRL_LOCAL_RUNTIME_ROOT}/jax_compilation_cache" \
    "${SSRL_LOCAL_RUNTIME_ROOT}/matplotlib" \
    "${SSRL_LOCAL_RUNTIME_ROOT}/numba"
  # Compiler caches must be private because they contain executable artifacts.
  chmod 0700 -- \
    "${SSRL_LOCAL_RUNTIME_ROOT}" \
    "${SSRL_LOCAL_RUNTIME_ROOT}/tmp" \
    "${SSRL_LOCAL_RUNTIME_ROOT}/cache" \
    "${SSRL_LOCAL_RUNTIME_ROOT}/wandb" \
    "${SSRL_LOCAL_RUNTIME_ROOT}/wandb/cache" \
    "${SSRL_LOCAL_RUNTIME_ROOT}/wandb/data" \
    "${SSRL_LOCAL_RUNTIME_ROOT}/wandb/artifacts" \
    "${SSRL_LOCAL_RUNTIME_ROOT}/jax_compilation_cache" \
    "${SSRL_LOCAL_RUNTIME_ROOT}/matplotlib" \
    "${SSRL_LOCAL_RUNTIME_ROOT}/numba"
  export SSRL_LOCAL_RUNTIME_ROOT
  export TMPDIR="${SSRL_LOCAL_RUNTIME_ROOT}/tmp"
  export XDG_CACHE_HOME="${SSRL_LOCAL_RUNTIME_ROOT}/cache"
  export WANDB_CACHE_DIR="${SSRL_LOCAL_RUNTIME_ROOT}/wandb/cache"
  export WANDB_DATA_DIR="${SSRL_LOCAL_RUNTIME_ROOT}/wandb/data"
  export WANDB_ARTIFACT_DIR="${SSRL_LOCAL_RUNTIME_ROOT}/wandb/artifacts"
  export JAX_COMPILATION_CACHE_ROOT="${SSRL_LOCAL_RUNTIME_ROOT}/jax_compilation_cache"
  SSRL_AUTO_LOCAL_JAX_CACHE=1
  export SSRL_AUTO_LOCAL_JAX_CACHE
  export MPLCONFIGDIR="${SSRL_LOCAL_RUNTIME_ROOT}/matplotlib"
  export NUMBA_CACHE_DIR="${SSRL_LOCAL_RUNTIME_ROOT}/numba"
  unset _ssrl_local_base _ssrl_local_key
fi

ssrl_runtime_filesystem_type() {
  stat -f -c '%T' -- "$1" 2>/dev/null || return 1
}

ssrl_require_node_local_path() {
  local label="$1"
  local path="$2"
  local filesystem
  filesystem="$(ssrl_runtime_filesystem_type "${path}")" || {
    echo "Node-local runtime preflight could not inspect ${label}: ${path}" >&2
    return 1
  }
  case "${filesystem,,}" in
    nfs|nfs4|cifs|smb*|fuse.sshfs)
      echo "Node-local runtime preflight rejected ${label} on ${filesystem}: ${path}" >&2
      return 1
      ;;
  esac
}

ssrl_node_local_runtime_preflight() {
  [[ "${SSRL_REQUIRE_NODE_LOCAL_RUNTIME:-false}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]] || return 0
  [[ -n "${SLURM_JOB_ID:-}" ]] || return 0
  [[ "${DRY_RUN:-false}" != "true" ]] || return 0

  local current_dir stdout_path stderr_path
  current_dir="$(pwd -P)" || return 1
  # $$ remains the batch-shell PID inside command substitutions; BASHPID would
  # identify the short-lived readlink subshell and inspect its pipe instead.
  stdout_path="$(readlink -f "/proc/$$/fd/1" 2>/dev/null || true)"
  stderr_path="$(readlink -f "/proc/$$/fd/2" 2>/dev/null || true)"

  ssrl_require_node_local_path "working directory" "${current_dir}" || return 1
  ssrl_require_node_local_path "staged source" "${SSRL_RUN_REPO_ROOT}" || return 1
  ssrl_require_node_local_path "runtime root" "${SSRL_LOCAL_RUNTIME_ROOT}" || return 1
  if [[ -z "${stdout_path}" || ! -f "${stdout_path}" ]]; then
    echo "Node-local runtime preflight requires stdout to be a regular file." >&2
    return 1
  fi
  if [[ -z "${stderr_path}" || ! -f "${stderr_path}" ]]; then
    echo "Node-local runtime preflight requires stderr to be a regular file." >&2
    return 1
  fi
  ssrl_require_node_local_path "stdout" "${stdout_path}" || return 1
  ssrl_require_node_local_path "stderr" "${stderr_path}" || return 1

  echo "Node-local runtime preflight passed: cwd=${current_dir} runtime=${SSRL_LOCAL_RUNTIME_ROOT}"
}

ssrl_node_local_runtime_preflight

export PYTHONPATH="${SSRL_RUN_REPO_ROOT}:${PYTHONPATH:-}"
# shellcheck disable=SC1091
source "${SSRL_RUN_REPO_ROOT}/experiments/_activate_runtime.sh"

# Load Madrona and JAX from the same pip-provided CUDA libraries.
_dmc_backend="${DMC_BACKEND:-}"
if [[ "${_dmc_backend,,}" == "mjx" || "${_dmc_backend,,}" == "mujoco_playground" || "${_dmc_backend,,}" == "playground" ]]; then
  NVIDIA_PYPI_ROOT="$(python3 -c 'import pathlib, site; print(next((pathlib.Path(path) / "nvidia" for path in site.getsitepackages() if (pathlib.Path(path) / "nvidia").is_dir()), ""))')"
  if [[ -n "${NVIDIA_PYPI_ROOT}" ]]; then
    NVIDIA_CUDA_LIBRARY_DIRS=()
    for component in cuda_runtime cuda_nvrtc nvjitlink cublas cusolver cusparse cudnn cufft nccl cuda_cupti; do
      if [[ -d "${NVIDIA_PYPI_ROOT}/${component}/lib" ]]; then
        NVIDIA_CUDA_LIBRARY_DIRS+=("${NVIDIA_PYPI_ROOT}/${component}/lib")
      fi
    done
    if (( ${#NVIDIA_CUDA_LIBRARY_DIRS[@]} )); then
      NVIDIA_CUDA_LIBRARY_PATH="$(IFS=:; printf '%s' "${NVIDIA_CUDA_LIBRARY_DIRS[*]}")"
      export LD_LIBRARY_PATH="${NVIDIA_CUDA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      export JAX_CUDA_LIBRARY_ROOT="${NVIDIA_PYPI_ROOT}"
    fi
  fi
fi
unset _dmc_backend

echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
echo "Python executable: $(command -v python3)"
if [[ -n "${JAX_CUDA_LIBRARY_ROOT:-}" ]]; then
  echo "JAX/Madrona CUDA libraries: ${JAX_CUDA_LIBRARY_ROOT}"
fi

export CUDA_LAUNCH_BLOCKING=0
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

skill_idm() {
  case "$1" in
    csf|vic|diayn|dads|rdads|valor) return 0 ;;
    *) return 1 ;;
  esac
}

declare -A EXPLICIT_VARS=()
for _explicit_var_name in $(compgen -v); do
  EXPLICIT_VARS["${_explicit_var_name}"]=1
done
unset _explicit_var_name
# Exclude allocation-specific cache paths from the immutable Hydra config.
if [[ "${SSRL_AUTO_LOCAL_JAX_CACHE:-0}" -eq 1 ]]; then
  unset 'EXPLICIT_VARS[JAX_COMPILATION_CACHE_ROOT]'
fi

was_explicit() {
  local var_name="$1"
  [[ -n "${EXPLICIT_VARS[${var_name}]:-}" ]]
}

mark_explicit() {
  local var_name="$1"
  EXPLICIT_VARS["${var_name}"]=1
}

append_explicit_override() {
  local key="$1"
  local var_name="$2"
  local value="${!var_name:-}"
  if was_explicit "${var_name}" && [[ -n "${value}" ]]; then
    CMD+=("${key}=${value}")
  fi
}

if [[ -n "${batch_size:-}" && -z "${BATCH_SIZE:-}" ]]; then
  BATCH_SIZE="${batch_size}"
  mark_explicit BATCH_SIZE
fi
if [[ -n "${batch_length:-}" && -z "${BATCH_LENGTH:-}" ]]; then
  BATCH_LENGTH="${batch_length}"
  mark_explicit BATCH_LENGTH
fi

quote_hydra_string() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "${value}"
}

append_explicit_string_override() {
  local key="$1"
  local var_name="$2"
  local value="${!var_name:-}"
  if was_explicit "${var_name}" && [[ -n "${value}" ]]; then
    CMD+=("${key}=$(quote_hydra_string "${value}")")
  fi
}

validate_run_path_component() {
  local label="$1"
  local value="$2"
  if [[ -z "${value}" || "${value}" == "." || "${value}" == ".." ||
        ! "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid ${label}=${value@Q}; use one path component containing only letters, numbers, dot, underscore, or hyphen." >&2
    exit 2
  fi
  if (( ${#value} > 180 )); then
    echo "Invalid ${label}: value is longer than 180 characters." >&2
    exit 2
  fi
}

note_pair() {
  local key="$1"
  local var_name="$2"
  local value="${!var_name:-}"
  if [[ -n "${value}" ]]; then
    NOTES+=("${key}=${value}")
  fi
}

print_config_value() {
  local label="$1"
  local var_name="$2"
  local value="${!var_name:-}"
  if was_explicit "${var_name}" && [[ -n "${value}" ]]; then
    echo "  ${label}: ${value}"
  else
    echo "  ${label}: loaded from Hydra config"
  fi
}

normalize_plan_repr_source() {
  local value="${1,,}"
  case "${value}" in
    feat|deter|stoch|embed) printf '%s\n' "${value}" ;;
    *)
      echo "Unknown plan repr source: ${1}" >&2
      exit 1
      ;;
  esac
}

normalize_expl_repr_mode() {
  local value="${1,,}"
  case "${value}" in
    raw|raw_plan) printf '%s\n' "raw_plan" ;;
    shared|shared_head) printf '%s\n' "shared_head" ;;
    branch|branch_bridge) printf '%s\n' "branch" ;;
    parallel|parallel_bridge) printf '%s\n' "parallel" ;;
    *)
      echo "Unknown exploration repr mode: ${1}" >&2
      exit 1
      ;;
  esac
}

normalize_expl_imag_repr_mode() {
  local value="${1,,}"
  case "${value}" in
    bridge|decode) printf '%s\n' "${value}" ;;
    *)
      echo "Unknown exploration imagination repr mode: ${1}" >&2
      exit 1
      ;;
  esac
}

normalize_expl_grad_mode() {
  local value="${1,,}"
  case "${value}" in
    detached|detach|stopgrad|stop_grad) printf '%s\n' "detached" ;;
    head_only|head) printf '%s\n' "head_only" ;;
    encoder_only|encoder|enc) printf '%s\n' "encoder_only" ;;
    full|shared) printf '%s\n' "full" ;;
    *)
      echo "Unknown exploration grad mode: ${1}" >&2
      exit 1
      ;;
  esac
}

normalize_agent_name() {
  local value="${1,,}"
  value="${value//-/_}"
  case "${value}" in
    dreamer|dreamerv3|v3) printf '%s\n' "dreamerv3" ;;
    r2dreamer|r2_dreamer) printf '%s\n' "r2dreamer" ;;
    nedreamer|ne_dreamer) printf '%s\n' "ne_dreamer" ;;
    dreamerpro|dreamer_pro) printf '%s\n' "dreamerpro" ;;
    *)
      echo "Unknown AGENT: ${1}. Use dreamerv3, r2dreamer, ne_dreamer, or dreamerpro." >&2
      exit 1
      ;;
  esac
}

infer_parallel_replay_obs_mode() {
  case "${CFG}" in
    dmc_proprio|manip_proprio) printf '%s\n' "raw_vector" ;;
    dmc_vision) printf '%s\n' "dreamer_cnn" ;;
    dmc_full) printf '%s\n' "dreamer_cnn+raw_vector" ;;
    dmlab) printf '%s\n' "dreamer_cnn" ;;
    *)
      case "${OBS_MODE}" in
        proprio) printf '%s\n' "raw_vector" ;;
        vision) printf '%s\n' "dreamer_cnn" ;;
        dmlab) printf '%s\n' "dreamer_cnn" ;;
        full) printf '%s\n' "dreamer_cnn+raw_vector" ;;
        *) printf '%s\n' "auto" ;;
      esac
      ;;
  esac
}

resolve_representation_config() {
  RESOLVED_EXPL_REPR_MODE="$(normalize_expl_repr_mode "${EXPL_REPR_MODE}")"
  RESOLVED_PLAN_REPR_SOURCE="$(normalize_plan_repr_source "${PLAN_REPR_SOURCE}")"
  RESOLVED_EXPL_GRAD_MODE="$(normalize_expl_grad_mode "${EXPL_GRAD_MODE}")"
  RESOLVED_EXPL_IMAG_REPR_MODE="$(normalize_expl_imag_repr_mode "${EXPL_IMAG_REPR_MODE}")"
  RESOLVED_EXPL_REPR_DIM="${EXPL_REPR_DIM}"
}

# Shared defaults

: "${DOMAIN_NAME:=quadruped}"
: "${TASK_NAME:=escape}"
: "${CFG:=dmc_proprio}"
if [[ "${CFG}" == dmc_* ]]; then
  ENV_SUITE="dmc"
  OBS_MODE="${CFG#dmc_}"
elif [[ "${CFG}" == atari* ]]; then
  ENV_SUITE="atari"
  OBS_MODE="${CFG}"
elif [[ "${CFG}" == craftax* ]]; then
  ENV_SUITE="craftax"
  if [[ "${CFG}" == "craftax" ]]; then
    OBS_MODE="pixels"
  else
    OBS_MODE="${CFG#craftax_}"
  fi
else
  ENV_SUITE="${CFG}"
  OBS_MODE="${CFG}"
fi
if [[ -n "${ENV_SUITE_OVERRIDE:-}" ]]; then
  ENV_SUITE="${ENV_SUITE_OVERRIDE}"
fi

: "${DMLAB_RUNFILES_PATH:=}"
: "${DMC_BACKEND:=}"
if [[ "${DMC_BACKEND,,}" == "mjx" || "${DMC_BACKEND,,}" == "mujoco_playground" || "${DMC_BACKEND,,}" == "playground" ]]; then
  # Use a smaller Madrona CUDA heap for cartpole vision; allow a per-job override.
  : "${MADRONA_MWGPU_DEVICE_HEAP_SIZE:=1073741824}"
  export MADRONA_MWGPU_DEVICE_HEAP_SIZE
fi

CALLER_SCRIPT="$(basename "${BASH_SOURCE[1]:-$0}")"
if [[ -z "${IDM_TYPE:-}" && "${CALLER_SCRIPT}" == "run_dreamer.sh" ]]; then
  IDM_TYPE="none"
fi
if [[ -z "${EXPL_BEHAVIOR:-}" ]]; then
  if [[ "${IDM_TYPE:-}" == "none" ]]; then
    EXPL_BEHAVIOR="greedy"
  else
    EXPL_BEHAVIOR="plan2explore"
  fi
fi
IDM_REWARD_NORMALIZE_WAS_SET="${IDM_REWARD_NORMALIZE+x}"
EXPL_UNTIL_WAS_SET="${EXPL_UNTIL+x}"
EXPL_INTR_SCALE_WAS_SET="${EXPL_INTR_SCALE+x}"

# Mirror config defaults for launcher validation/display only. These are
# forwarded as Hydra overrides only when a wrapper or caller set them explicitly.
: "${IDM_REWARD_NORMALIZE:=false}"
: "${EXPL_UNTIL:=0}"
if [[ "${EXPL_BEHAVIOR}" == "greedy" || "${IDM_TYPE:-}" == "none" ]]; then
  : "${EXPL_INTR_SCALE:=0.0}"
else
  : "${EXPL_INTR_SCALE:=1.0}"
fi
: "${EXPL_EXTR_SCALE:=0.0}"

: "${SEED:=5}"
: "${USE_WANDB:=true}"
: "${WANDB_PROJECT:=ssrl}"
: "${WANDB_ENTITY:=}"
: "${WANDB_TAG:=}"
: "${WANDB_ID:=}"
: "${WANDB_NOTES_EXTRA:=}"
: "${WANDB_RELAUNCH_MODE:=fresh}"
: "${RUN_STORAGE_MODE:=auto}"
: "${RUN_FORK_SOURCE_RUN_UID:=}"
: "${RUN_FORK_PARENT_SOURCE_ID:=}"
: "${RUN_FORK_SOURCE_DIR:=}"
: "${DEVICE:=cuda:0}"
: "${DRY_RUN:=false}"
: "${MODEL_SIZE:=size12M}"
: "${AGENT:=dreamerv3}"
RESOLVED_AGENT_NAME="$(normalize_agent_name "${AGENT}")"

if [[ -z "${JOB_TAG:-}" ]]; then
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    JOB_TAG="${SLURM_CLUSTER_NAME:-slurm}_job${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"
  elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    JOB_TAG="${SLURM_CLUSTER_NAME:-slurm}_job${SLURM_JOB_ID}"
  else
    _local_host="$(hostname -s 2>/dev/null || hostname)"
    _local_host="${_local_host//[^A-Za-z0-9._-]/_}"
    JOB_TAG="local_${_local_host}_$(date -u +%Y%m%dT%H%M%S%N)_${BASHPID}_${RANDOM}"
    unset _local_host
  fi
fi
validate_run_path_component JOB_TAG "${JOB_TAG}"
RUN_UID="${JOB_TAG}"

case "${RUN_STORAGE_MODE,,}" in
  auto)
    if [[ "${SLURM_RESTART_COUNT:-0}" =~ ^[0-9]+$ ]] &&
       (( ${SLURM_RESTART_COUNT:-0} > 0 )); then
      RUN_STORAGE_MODE_RESOLVED="resume"
    else
      RUN_STORAGE_MODE_RESOLVED="new"
    fi
    ;;
  new|fresh)
    # Resume the same logical run after Slurm requeue, even if the initial submission
    # requested a fresh run.
    if [[ "${SLURM_RESTART_COUNT:-0}" =~ ^[1-9][0-9]*$ ]]; then
      echo "Slurm restart ${SLURM_RESTART_COUNT}: overriding initial RUN_STORAGE_MODE=${RUN_STORAGE_MODE} with resume."
      RUN_STORAGE_MODE_RESOLVED="resume"
    else
      RUN_STORAGE_MODE_RESOLVED="new"
    fi
    ;;
  resume) RUN_STORAGE_MODE_RESOLVED="resume" ;;
  fork) RUN_STORAGE_MODE_RESOLVED="fork" ;;
  *)
    echo "Invalid RUN_STORAGE_MODE=${RUN_STORAGE_MODE}; use auto, new, resume, or fork." >&2
    exit 2
    ;;
esac
if [[ "${RUN_STORAGE_MODE_RESOLVED}" == "fork" ]]; then
  validate_run_path_component RUN_FORK_SOURCE_RUN_UID "${RUN_FORK_SOURCE_RUN_UID}"
  if [[ ! "${RUN_FORK_PARENT_SOURCE_ID}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "RUN_FORK_PARENT_SOURCE_ID must be the parent run's SHA-256 source ID." >&2
    exit 2
  fi
elif [[ -n "${RUN_FORK_SOURCE_RUN_UID}" || -n "${RUN_FORK_PARENT_SOURCE_ID}" ]]; then
  echo "RUN_FORK_SOURCE_RUN_UID and RUN_FORK_PARENT_SOURCE_ID require RUN_STORAGE_MODE=fork." >&2
  exit 2
fi

case "${WANDB_RELAUNCH_MODE,,}" in
  continue|fresh) WANDB_RELAUNCH_MODE="${WANDB_RELAUNCH_MODE,,}" ;;
  *)
    echo "Invalid WANDB_RELAUNCH_MODE=${WANDB_RELAUNCH_MODE}; use continue or fresh." >&2
    exit 2
    ;;
esac
if was_explicit WANDB_RESUME && [[ "${WANDB_RESUME,,}" != "never" ]]; then
    echo "WANDB_RESUME=${WANDB_RESUME} is unsupported; use WANDB_RELAUNCH_MODE=continue or fresh." >&2
    exit 2
fi
WANDB_RESUME="never"
WANDB_RESUME_FROM=""
RUN_ATTEMPT_ID="restart${SLURM_RESTART_COUNT:-0}_$(date -u +%Y%m%dT%H%M%S%N)_${BASHPID}"
: "${HELPER_DIR:=${SSRL_RUN_REPO_ROOT}/experiments}"

if [[ -z "${IDM_TYPE:-}" ]]; then
  echo "IDM_TYPE must be set by the wrapper script." >&2
  exit 1
fi
IDM_CONFIG_NAME=""
if [[ "${IDM_TYPE}" != "none" ]]; then
  IDM_CONFIG_PATH="${SSRL_RUN_REPO_ROOT}/configs/idm/${IDM_TYPE}.yaml"
  if [[ ! -f "${IDM_CONFIG_PATH}" ]]; then
    echo "No Hydra IDM config found for IDM_TYPE=${IDM_TYPE}: ${IDM_CONFIG_PATH}" >&2
    exit 1
  fi
  IDM_CONFIG_NAME="${IDM_TYPE}"
fi

TRAIN_TASK_NAME="${TASK_NAME}"
EVAL_TASKS_HYDRA=""
EVAL_TASKS_DISPLAY=""
if [[ "${TASK_NAME}" == "multi" ]]; then
  if [[ "${ENV_SUITE}" != "dmc" ]]; then
    echo "TASK_NAME=multi is currently defined for DMC domains only." >&2
    exit 1
  fi
  case "${DOMAIN_NAME}" in
    humanoid|walker)
      TRAIN_TASK_NAME="stand"
      EVAL_TASKS_HYDRA="[dmc_${DOMAIN_NAME}_stand,dmc_${DOMAIN_NAME}_walk,dmc_${DOMAIN_NAME}_run]"
      EVAL_TASKS_DISPLAY="${DOMAIN_NAME}_stand, ${DOMAIN_NAME}_walk, ${DOMAIN_NAME}_run"
      ;;
    *)
      echo "TASK_NAME=multi is currently defined for DMC DOMAIN_NAME=humanoid or walker only." >&2
      exit 1
      ;;
  esac
fi

PLAN_REPR_SOURCE_WAS_SET="${PLAN_REPR_SOURCE+x}"
EXPL_REPR_MODE_WAS_SET="${EXPL_REPR_MODE+x}"
EXPL_IMAG_REPR_MODE_WAS_SET="${EXPL_IMAG_REPR_MODE+x}"
EXPL_GRAD_MODE_WAS_SET="${EXPL_GRAD_MODE+x}"
EXPL_HEAD_TYPE_WAS_SET="${EXPL_HEAD_TYPE+x}"

# Mirror config defaults for launcher validation/display only. These are
# forwarded as Hydra overrides only when a wrapper or caller set them explicitly.
: "${PLAN_REPR_SOURCE:=deter}"
: "${EXPL_REPR_MODE:=shared_head}"
: "${EXPL_IMAG_REPR_MODE:=bridge}"
if [[ "${EXPL_BEHAVIOR}" == "greedy" || "${IDM_TYPE}" == "none" ]]; then
  : "${EXPL_GRAD_MODE:=detached}"
else
  : "${EXPL_GRAD_MODE:=head_only}"
fi
: "${EXPL_REPR_DIM:=0}"
: "${EXPL_HEAD_TYPE:=identity}"
: "${EXPL_HEAD_HIDDEN_DIM:=256}"
: "${EXPL_HEAD_LAYERS:=2}"
: "${EXPL_OBS_ENCODER_TYPE:=null}"
: "${EXPL_OBS_HIDDEN_DIM:=256}"
: "${EXPL_OBS_LAYERS:=2}"
: "${EXPL_RNN_HIDDEN_DIM:=0}"
: "${EXPL_RNN_BATCH_CHUNK_SIZE:=256}"
: "${EXPL_OBS_CNN_DEPTH:=0}"
: "${EXPL_OBS_CNN_KERNEL_SIZE:=0}"
: "${EXPL_OBS_CNN_MINRES:=0}"
: "${EXPL_OBS_CNN_CHUNK_SIZE:=64}"
: "${EXPL_BRIDGE_HIDDEN_DIM:=256}"
: "${EXPL_BRIDGE_LAYERS:=2}"
: "${EXPL_BRIDGE_LOSS_WEIGHT:=1.0}"
: "${EXPL_BRIDGE_CONSISTENCY_WEIGHT:=1.0}"
: "${EXPL_BRIDGE_TARGET_EMA:=false}"
: "${EXPL_BRIDGE_TARGET_EMA_MOMENTUM:=0.99}"
: "${IDM_REWARD_NORM_SOURCE:=replay_only}"

: "${IDM_HIDDEN_DIM:=400}"
: "${IDM_OUTPUT_DIM:=400}"
: "${IDM_N_LAYERS:=2}"
: "${RND_OUTPUT_DIM:=400}"
: "${RND_NORMALIZE:=true}"

: "${DISAG_MODELS:=5}"
: "${DISAG_ACTION_COND:=true}"
: "${DISAG_BOOTSTRAP:=true}"
: "${DISAG_TARGET:=embed}"
: "${DISAG_NORMALIZE:=true}"
: "${DISAG_LOG:=false}"
: "${DISAG_OFFSET:=1}"

SHOW_SKILLS=false
if skill_idm "${IDM_TYPE}" || [[ "${USE_SKILLS:-}" == "true" ]]; then
  SHOW_SKILLS=true
fi

# Validation

if [[ "${IDM_TYPE}" == "csf" ]]; then
  if [[ -z "${CSF_PRIOR:-}" && -n "${SKILL_SPACE:-}" ]]; then
    if [[ "${SKILL_SPACE:-}" == "sphere" ]]; then
      CSF_PRIOR="uniform_hypersphere"
    elif [[ "${SKILL_SPACE:-}" == "box" ]]; then
      CSF_PRIOR="uniform_box"
    else
      CSF_PRIOR="uniform_categorical"
    fi
    if was_explicit SKILL_SPACE; then
      mark_explicit CSF_PRIOR
    fi
  fi
fi

resolve_representation_config

PARALLEL_REPLAY_INPUT_MODE=""
if [[ "${RESOLVED_EXPL_REPR_MODE}" == "parallel" ]]; then
  PARALLEL_REPLAY_INPUT_MODE="$(infer_parallel_replay_obs_mode)"
fi
if [[ "${RESOLVED_EXPL_REPR_MODE}" != "branch" && "${RESOLVED_EXPL_REPR_MODE}" != "parallel" && "${RESOLVED_EXPL_IMAG_REPR_MODE}" != "bridge" ]]; then
  echo "Invalid combo: EXPL_IMAG_REPR_MODE only applies to EXPL_REPR_MODE=branch or parallel." >&2
  exit 1
fi
if [[ "${RESOLVED_PLAN_REPR_SOURCE}" == "embed" ]]; then
  if [[ "${RESOLVED_EXPL_REPR_MODE}" != "branch" && "${RESOLVED_EXPL_REPR_MODE}" != "parallel" ]]; then
    echo "Invalid combo: plan_repr_source=embed is replay-only unless expl_repr_mode is branch/parallel with decode imagination." >&2
    exit 1
  fi
  if [[ "${RESOLVED_EXPL_IMAG_REPR_MODE}" != "decode" ]]; then
    echo "Invalid combo: plan_repr_source=embed requires expl_imag_repr_mode=decode for branch/parallel imagination." >&2
    exit 1
  fi
fi
# Names and paths

if [[ -z "${ALGO_PRESET:-}" ]]; then
  if [[ "${EXPL_BEHAVIOR}" == "greedy" || "${IDM_TYPE}" == "none" ]]; then
    ALGO_PRESET="dreamer"
  else
    ALGO_PRESET="p2e"
  fi
fi

RUN_STRICT_TASK_BLIND_PRETRAINING=false
if [[ "${ALGO_PRESET}" == "p2e" ]]; then
  case "${STRICT_TASK_BLIND_PRETRAINING:-false}" in
    true|True|TRUE|1|yes|Yes|YES|on|On|ON)
      RUN_STRICT_TASK_BLIND_PRETRAINING=true
      ;;
    false|False|FALSE|0|no|No|NO|off|Off|OFF)
      ;;
    *)
      echo "STRICT_TASK_BLIND_PRETRAINING must be a boolean." >&2
      exit 2
      ;;
  esac
fi

if [[ "${ALGO_PRESET}" == "dreamer" ]]; then
  if [[ "${RESOLVED_AGENT_NAME}" == "dreamerv3" ]]; then
    COMBO_TAG="dreamer"
    RUN_ALGO_TAG="dreamer"
  else
    COMBO_TAG="${RESOLVED_AGENT_NAME}"
    RUN_ALGO_TAG="${RESOLVED_AGENT_NAME}"
  fi
else
  RUN_ALGO_TAG="${IDM_TYPE}"
  COMBO_TAG="${IDM_TYPE}"
fi

if [[ "${ENV_SUITE}" == "dmlab" && "${DOMAIN_NAME}" == "dmlab" ]]; then
  RUN_TASK_TAG="${TASK_NAME}"
else
  RUN_TASK_TAG="${DOMAIN_NAME}_${TASK_NAME}"
fi

if [[ "${ALGO_PRESET}" == "dreamer" ]]; then
  RUN_FAMILY_PARTS=("${RUN_ALGO_TAG}" "${CFG}")
else
  RUN_FAMILY_PARTS=("${ALGO_PRESET}" "${CFG}")
fi
if [[ "${ALGO_PRESET}" != "dreamer" && "${RUN_ALGO_TAG}" != "${ALGO_PRESET}" ]]; then
  RUN_FAMILY_PARTS+=("${RUN_ALGO_TAG}")
fi
if [[ "${ALGO_PRESET}" != "dreamer" && "${RESOLVED_AGENT_NAME}" != "dreamerv3" ]]; then
  RUN_FAMILY_PARTS+=("${RESOLVED_AGENT_NAME}")
fi
RUN_FAMILY_PARTS+=("${MODEL_SIZE}")
if [[ "${RUN_TASK_TAG}" != "${CFG}" ]]; then
  RUN_FAMILY_PARTS+=("${RUN_TASK_TAG}")
fi
RUN_FAMILY_NAME="$(IFS=_; echo "${RUN_FAMILY_PARTS[*]}")"

NAME_PARTS=(
  "${RUN_FAMILY_NAME}"
  "seed${SEED}"
  "${JOB_TAG}"
)
DEFAULT_RUN_NAME="$(IFS=_; echo "${NAME_PARTS[*]}")"

: "${EXPERIMENT_NAME:=${RUN_FAMILY_NAME}}"
: "${RUN_NAME:=${DEFAULT_RUN_NAME}}"
: "${WANDB_NAME:=${RUN_NAME}}"
: "${WANDB_ID:=${RUN_NAME}}"

validate_run_path_component EXPERIMENT_NAME "${EXPERIMENT_NAME}"
validate_run_path_component RUN_NAME "${RUN_NAME}"

# Assign a new W&B attempt unless verified continuation selects an existing run.
WANDB_LOGICAL_ID="${WANDB_ID}"
WANDB_LOGICAL_NAME="${WANDB_NAME}"
_wandb_attempt_hash="$(printf '%s\0%s' "${WANDB_LOGICAL_ID}" "${RUN_ATTEMPT_ID}" | sha256sum | awk '{print $1}')"
_wandb_id_prefix="${WANDB_LOGICAL_ID//[^A-Za-z0-9_-]/_}"
_wandb_id_prefix="${_wandb_id_prefix:0:80}"
[[ -n "${_wandb_id_prefix}" ]] || _wandb_id_prefix="run"
WANDB_ATTEMPT_ID="${_wandb_id_prefix}_a${_wandb_attempt_hash:0:20}"
WANDB_ID="${WANDB_ATTEMPT_ID}"
: "${WANDB_GROUP:=${WANDB_LOGICAL_ID}}"
unset _wandb_attempt_hash _wandb_id_prefix

LOG_DIR="${EXPERIMENT_ROOT}/${EXPERIMENT_NAME}/${RUN_NAME}"
TRAIN_DIR="${LOG_DIR}/train_eps"
EVAL_DIR="${LOG_DIR}/eval_eps"

if [[ ! -f "${HELPER_DIR}/scratch_sync.sh" || ! -f "${HELPER_DIR}/slurm_preemptable.sh" ]]; then
  echo "Required storage/preemption helpers are missing from HELPER_DIR=${HELPER_DIR}." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${HELPER_DIR}/scratch_sync.sh"
# shellcheck disable=SC1090
source "${HELPER_DIR}/slurm_preemptable.sh"
scratch_sync_require_commands
scratch_sync_configure_paths
# Keep reusable compiler caches outside the checkpointed run tree.
: "${JAX_COMPILATION_CACHE_ROOT:=${WORK_LOG_ROOT}/.jax_compilation_cache/uid-${UID}}"
RUN_STORAGE_HELPER="${HELPER_DIR}/run_storage.py"
if [[ ! -f "${RUN_STORAGE_HELPER}" ]]; then
  echo "Required run provenance helper is missing: ${RUN_STORAGE_HELPER}" >&2
  exit 1
fi
if [[ "${RUN_STORAGE_MODE_RESOLVED}" == "fork" ]]; then
  RUN_FORK_SOURCE_RUN_NAME="${RUN_FAMILY_NAME}_seed${SEED}_${RUN_FORK_SOURCE_RUN_UID}"
  validate_run_path_component RUN_FORK_SOURCE_RUN_NAME "${RUN_FORK_SOURCE_RUN_NAME}"
  # Allow recovery inputs and new publications to use separate storage roots.
  : "${RUN_FORK_SOURCE_ROOT:=${WORK_LOG_ROOT}}"
  RUN_FORK_SOURCE_DIR="${RUN_FORK_SOURCE_ROOT}/${EXPERIMENT_NAME}/${RUN_FORK_SOURCE_RUN_NAME}"
  # Match the parent's group when either continuing its exact remote run or
  # intentionally creating a separate fresh attempt.
  if ! was_explicit WANDB_GROUP; then
    WANDB_GROUP="${RUN_FORK_SOURCE_RUN_NAME}"
  fi
  if [[ "${RUN_FORK_SOURCE_DIR}" == "${WORK_LOG_DIR}" ]]; then
    echo "Checkpoint fork source and destination must be distinct runs." >&2
    exit 2
  fi
fi

# Resume W&B at the checkpoint independently verified by storage initialization.
# A preemption before the first checkpoint starts a new attempt in the same group.
if [[ "${WANDB_RELAUNCH_MODE}" == "continue" &&
      "${RUN_STORAGE_MODE_RESOLVED}" != "new" &&
      "${USE_WANDB,,}" == "true" &&
      "${DRY_RUN,,}" != "true" ]]; then
  _wandb_continue_json=""
  if [[ "${RUN_STORAGE_MODE_RESOLVED}" == "fork" ]]; then
    # Resume a verified child after requeue; otherwise initialize the fork from its
    # parent.
    if scratch_sync_dir_has_entries "${WORK_LOG_DIR}" ||
       scratch_sync_dir_has_entries "${LOG_DIR}" ||
       [[ -e "${SCRATCH_SYNC_TOMBSTONE}" ||
          -L "${SCRATCH_SYNC_TOMBSTONE}" ||
          -e "${SCRATCH_SYNC_RECOVERY_DIR}" ||
          -L "${SCRATCH_SYNC_RECOVERY_DIR}" ]]; then
      if _wandb_continue_json="$(
        python3 "${RUN_STORAGE_HELPER}" wandb-continuation \
          --run-dir "${WORK_LOG_DIR}" \
          --expected-run-id "${RUN_UID}" \
          --live-run-dir "${LOG_DIR}" \
          --live-run-dir "${SCRATCH_SYNC_TOMBSTONE}" 2>/dev/null
      )"; then
        _wandb_continue_dir="${WORK_LOG_DIR}"
        _wandb_continue_run_uid="${RUN_UID}"
        _wandb_continue_source_args=(
          --live-run-dir "${LOG_DIR}"
          --live-run-dir "${SCRATCH_SYNC_TOMBSTONE}"
        )
      else
        _wandb_continue_json=""
        _wandb_continue_dir="${RUN_FORK_SOURCE_DIR}"
        _wandb_continue_run_uid="${RUN_FORK_SOURCE_RUN_UID}"
        _wandb_continue_source_args=(--expected-source-id "${RUN_FORK_PARENT_SOURCE_ID}")
      fi
    else
      _wandb_continue_dir="${RUN_FORK_SOURCE_DIR}"
      _wandb_continue_run_uid="${RUN_FORK_SOURCE_RUN_UID}"
      _wandb_continue_source_args=(--expected-source-id "${RUN_FORK_PARENT_SOURCE_ID}")
    fi
  elif [[ "${RUN_STORAGE_MODE_RESOLVED}" == "resume" ]]; then
    _wandb_continue_dir="${WORK_LOG_DIR}"
    _wandb_continue_run_uid="${RUN_UID}"
    # Include verified scratch checkpoints when selecting the most advanced recovery
    # state.
    _wandb_continue_source_args=(
      --live-run-dir "${LOG_DIR}"
      --live-run-dir "${SCRATCH_SYNC_TOMBSTONE}"
    )
    if scratch_sync_is_uninitialized_wandb_empty_preempt \
         "${WORK_LOG_DIR}" "${RUN_UID}"; then
      # Create a fresh remote attempt only when every checkpoint location is empty.
      # Verification failures do not indicate absence.
      _wandb_checkpoint_state_present=0
      for _wandb_candidate_dir in "${LOG_DIR}" "${SCRATCH_SYNC_TOMBSTONE}"; do
        if [[ -e "${_wandb_candidate_dir}/checkpoint.pkl" ||
              -L "${_wandb_candidate_dir}/checkpoint.pkl" ||
              -e "${_wandb_candidate_dir}/checkpoint.pkl.incomplete" ||
              -L "${_wandb_candidate_dir}/checkpoint.pkl.incomplete" ||
              -e "${_wandb_candidate_dir}/latest.pt" ||
              -L "${_wandb_candidate_dir}/latest.pt" ]]; then
          _wandb_checkpoint_state_present=1
          break
        fi
      done
      if [[ -e "${SCRATCH_SYNC_RECOVERY_DIR}" ||
            -L "${SCRATCH_SYNC_RECOVERY_DIR}" ]]; then
        _wandb_checkpoint_state_present=1
      fi
      if [[ "${_wandb_checkpoint_state_present}" -eq 0 ]]; then
        WANDB_RELAUNCH_MODE="fresh"
        echo "Certified pre-checkpoint restart has no initialized W&B run; using a fresh remote attempt in logical group ${WANDB_GROUP}." >&2
      fi
      unset _wandb_checkpoint_state_present _wandb_candidate_dir
    fi
  else
    echo "Unsupported W&B continuation storage mode: ${RUN_STORAGE_MODE_RESOLVED}." >&2
    exit 1
  fi
  if [[ "${WANDB_RELAUNCH_MODE}" == "continue" ]]; then
    if [[ -z "${_wandb_continue_json}" ]]; then
      if ! _wandb_continue_json="$(
        python3 "${RUN_STORAGE_HELPER}" wandb-continuation \
          --run-dir "${_wandb_continue_dir}" \
          --expected-run-id "${_wandb_continue_run_uid}" \
          "${_wandb_continue_source_args[@]}"
      )"; then
        echo "Cannot safely continue the prior W&B run; training was not started." >&2
        echo "Use WANDB_RELAUNCH_MODE=fresh only if a separate W&B run is intentional." >&2
        exit 1
      fi
    fi
    mapfile -t _wandb_continue_fields < <(
      python3 -c \
        'import json, sys; value = json.loads(sys.argv[1]); print(value["wandb_id"]); print(value["resume_from"]); print(value["parent_run_name"]); print(value["checkpoint_step"]); print(value["checkpoint"])' \
        "${_wandb_continue_json}"
    )
    if [[ "${#_wandb_continue_fields[@]}" -ne 5 ||
          -z "${_wandb_continue_fields[0]}" ||
          -z "${_wandb_continue_fields[1]}" ||
          -z "${_wandb_continue_fields[2]}" ||
          ! "${_wandb_continue_fields[3]}" =~ ^[0-9]+$ ||
          ! "${_wandb_continue_fields[4]}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
      echo "W&B continuation metadata is incomplete or unsafe; training was not started." >&2
      exit 1
    fi
    WANDB_ATTEMPT_ID="${_wandb_continue_fields[0]}"
    WANDB_ID="${WANDB_ATTEMPT_ID}"
    WANDB_RESUME_FROM="${_wandb_continue_fields[1]}"
    WANDB_NAME="${_wandb_continue_fields[2]}"
    WANDB_CHECKPOINT_STEP="${_wandb_continue_fields[3]}"
    WANDB_CHECKPOINT_NAME="${_wandb_continue_fields[4]}"
    unset _wandb_continue_dir _wandb_continue_run_uid _wandb_continue_source_args
    unset _wandb_continue_json _wandb_continue_fields
  fi
fi
if ! declare -F slurm_run_if_preempt >/dev/null 2>&1; then
  _slurm_forward_to_child() {
    local signal_name="$1"
    local mark_preempt="${2:-0}"
    if [[ "${mark_preempt}" == "1" ]]; then
      SLURM_PREEMPTED=1
      export SLURM_PREEMPTED
    fi
    echo "Received ${signal_name}; forwarding to training process ${SLURM_RUN_CHILD_PID:-unknown}."
    if [[ -n "${SLURM_RUN_CHILD_PID:-}" ]]; then
      kill "-${signal_name}" "${SLURM_RUN_CHILD_PID}" 2>/dev/null || true
    fi
  }

  slurm_run_if_preempt() {
    local status
    "$@" &
    SLURM_RUN_CHILD_PID=$!

    trap '_slurm_forward_to_child USR1 1' USR1
    trap '_slurm_forward_to_child TERM 0' TERM
    trap '_slurm_forward_to_child INT 0' INT

    while true; do
      if wait "${SLURM_RUN_CHILD_PID}"; then
        status=0
        break
      else
        status=$?
      fi
      if [[ "${status}" -ge 128 ]] && kill -0 "${SLURM_RUN_CHILD_PID}" 2>/dev/null; then
        continue
      fi
      break
    done

    trap - USR1 TERM INT
    SLURM_RUN_CHILD_PID=""
    if [[ "${SLURM_PREEMPTED:-0}" -eq 1 ]]; then
      return 143
    fi
    return "${status}"
  }
fi

NOTES=(
  "suite=${ENV_SUITE}"
  "combo=${COMBO_TAG}"
  "agent=${RESOLVED_AGENT_NAME}"
  "idm=${IDM_TYPE}"
  "seed=${SEED}"
)
if [[ "${ALGO_PRESET}" == "p2e" ]]; then
  NOTES+=(
    "strict_task_blind_pretraining=${RUN_STRICT_TASK_BLIND_PRETRAINING}"
  )
fi
if was_explicit ADAPTIVE_CRITIC_ENABLED &&
    [[ -n "${ADAPTIVE_CRITIC_ENABLED:-}" ]]; then
  note_pair "adaptive_critic" "ADAPTIVE_CRITIC_ENABLED"
  note_pair "adaptive_critics" "ADAPTIVE_CRITIC_CRITICS"
  note_pair "adaptive_scope" "ADAPTIVE_CRITIC_SCOPE"
  note_pair "adaptive_granularity" "ADAPTIVE_CRITIC_GRANULARITY"
  note_pair "adaptive_meta_lr" "ADAPTIVE_CRITIC_META_LR"
  note_pair "adaptive_init_scale" "ADAPTIVE_CRITIC_INIT_SCALE"
fi
if [[ "${RUN_STORAGE_MODE_RESOLVED}" == "fork" ]]; then
  NOTES+=(
    "fork_parent=${RUN_FORK_SOURCE_RUN_UID}"
    "fork_parent_source=${RUN_FORK_PARENT_SOURCE_ID}"
  )
fi
if [[ -n "${PLAN_REPR_SOURCE_WAS_SET}" ]]; then
  NOTES+=("plan_repr=${RESOLVED_PLAN_REPR_SOURCE}")
fi
if [[ -n "${EXPL_REPR_MODE_WAS_SET}" ]]; then
  NOTES+=("expl_repr=${RESOLVED_EXPL_REPR_MODE}")
fi
if [[ -n "${EXPL_GRAD_MODE_WAS_SET}" ]]; then
  NOTES+=("expl_grad=${RESOLVED_EXPL_GRAD_MODE}")
fi
if [[ -n "${EVAL_TASKS_DISPLAY}" ]]; then
  NOTES+=("train_task=${DOMAIN_NAME}_${TRAIN_TASK_NAME}")
  NOTES+=("eval_tasks=${EVAL_TASKS_DISPLAY}")
fi
if [[ "${ENV_SUITE}" == "dmlab" ]]; then
  note_pair "dmlab_runfiles" "DMLAB_RUNFILES_PATH"
fi
note_pair "expl_dim" "RESOLVED_EXPL_REPR_DIM"
case "${RESOLVED_EXPL_REPR_MODE}" in
  shared_head)
    if [[ -n "${EXPL_HEAD_TYPE_WAS_SET}" ]]; then
      note_pair "expl_head" "EXPL_HEAD_TYPE"
    fi
    ;;
  branch)
    if [[ -n "${EXPL_IMAG_REPR_MODE_WAS_SET}" ]]; then
      NOTES+=("expl_imag=${RESOLVED_EXPL_IMAG_REPR_MODE}")
    fi
    note_pair "expl_obs_encoder" "EXPL_OBS_ENCODER_TYPE"
    ;;
  parallel)
    if [[ -n "${EXPL_IMAG_REPR_MODE_WAS_SET}" ]]; then
      NOTES+=("expl_imag=${RESOLVED_EXPL_IMAG_REPR_MODE}")
    fi
    note_pair "expl_rnn_dim" "EXPL_RNN_HIDDEN_DIM"
    ;;
esac
if [[ -n "${PARALLEL_REPLAY_INPUT_MODE}" ]]; then
  NOTES+=("parallel_obs=${PARALLEL_REPLAY_INPUT_MODE}")
fi
if [[ "${IDM_TYPE}" == "ctec" ]]; then
  note_pair "ctec_sim" "CTEC_SIMILARITY"
  note_pair "ctec_gamma" "CTEC_GAMMA"
  note_pair "ctec_loss" "CTEC_LOSS_TYPE"
fi
if [[ "${IDM_TYPE}" == "cip" ]]; then
  note_pair "cip_state" "CIP_STATE_SOURCE"
  note_pair "cip_imag_last" "CIP_IMAG_LAST"
  note_pair "cip_probes" "CIP_PROBES"
  note_pair "cip_scale" "CIP_SCALE"
fi
if [[ "${IDM_TYPE}" == "csf" ]]; then
  if was_explicit CSF_PROJ_DIM; then
    note_pair "csf_proj" "CSF_PROJ_DIM"
  fi
fi
if [[ "${IDM_TYPE}" == "re3" ]]; then
  note_pair "re3_rep" "RE3_REP_DIM"
fi
if [[ "${IDM_TYPE}" == "ngu" ]]; then
  note_pair "ngu_rep" "NGU_REP_DIM"
fi
WANDB_NOTES="$(IFS='; '; echo "${NOTES[*]}")"
if [[ -n "${WANDB_NOTES_EXTRA}" ]]; then
  WANDB_NOTES="${WANDB_NOTES}; ${WANDB_NOTES_EXTRA}"
fi

# Command

case "${ENV_SUITE}" in
  dmc)
    ENV_TASK="dmc_${DOMAIN_NAME}_${TRAIN_TASK_NAME}"
    ;;
  dmlab)
    if [[ "${TASK_NAME}" == dmlab_* ]]; then
      ENV_TASK="${TASK_NAME}"
    else
      ENV_TASK="dmlab_${TASK_NAME}"
    fi
    ;;
  *)
    ENV_TASK="${ENV_SUITE}_${TASK_NAME}"
    ;;
esac

CMD=(
  python3 "${SSRL_RUN_REPO_ROOT}/train.py"
  "envs=${CFG}"
  "algo=${ALGO_PRESET}"
  "model=${MODEL_SIZE}"
  "logdir=${LOG_DIR}"
  "envs.task=${ENV_TASK}"
)
if [[ "${RESOLVED_AGENT_NAME}" != "dreamerv3" ]]; then
  CMD+=("agent_name=${RESOLVED_AGENT_NAME}")
fi
if [[ -n "${IDM_CONFIG_NAME}" ]]; then
  CMD+=("idm=${IDM_CONFIG_NAME}")
fi

append_explicit_override "seed" SEED
append_explicit_override "deterministic_run" DETERMINISTIC_RUN
append_explicit_override "jax_autotune" JAX_AUTOTUNE
append_explicit_override "jax_profiler" JAX_PROFILER
append_explicit_override "jax_donate_policy_params" JAX_DONATE_POLICY_PARAMS
append_explicit_override "jax_preallocate" JAX_PREALLOCATE
append_explicit_override "jax_compilation_cache" JAX_COMPILATION_CACHE
append_explicit_string_override "jax_compilation_cache_root" JAX_COMPILATION_CACHE_ROOT
append_explicit_string_override "jax_compilation_cache_dir" JAX_COMPILATION_CACHE_DIR
append_explicit_override "jax_compilation_cache_min_compile_time_secs" JAX_COMPILATION_CACHE_MIN_COMPILE_TIME_SECS
append_explicit_override "jax_compilation_cache_min_entry_size_bytes" JAX_COMPILATION_CACHE_MIN_ENTRY_SIZE_BYTES
append_explicit_override "device" DEVICE
append_explicit_override "model.expl_behavior" EXPL_BEHAVIOR
append_explicit_override "model.expl_extr_scale" EXPL_EXTR_SCALE
append_explicit_override "model.imag_last" IMAG_LAST
append_explicit_override "model.imag_start_ratio" IMAG_START_RATIO
append_explicit_string_override "model.imag_start_strategy" IMAG_START_STRATEGY
append_explicit_override "model.recon_last" RECON_LAST
append_explicit_override "model.recon_start_ratio" RECON_START_RATIO
append_explicit_string_override "model.recon_start_strategy" RECON_START_STRATEGY
append_explicit_override "model.log_grads" LOG_GRADS
append_explicit_override "model.rssm.unroll" RSSM_UNROLL
append_explicit_override "model.policy_checks" POLICY_CHECKS
if [[ "${ALGO_PRESET}" == "p2e" ]]; then
  append_explicit_override "model.strict_task_blind_pretraining" STRICT_TASK_BLIND_PRETRAINING
fi
append_explicit_override "model.adaptive_critic.enabled" ADAPTIVE_CRITIC_ENABLED
append_explicit_override "model.adaptive_critic.critics" ADAPTIVE_CRITIC_CRITICS
append_explicit_override "model.adaptive_critic.scope" ADAPTIVE_CRITIC_SCOPE
append_explicit_override "model.adaptive_critic.granularity" ADAPTIVE_CRITIC_GRANULARITY
append_explicit_override "model.adaptive_critic.meta_lr" ADAPTIVE_CRITIC_META_LR
append_explicit_override "model.adaptive_critic.trace_decay" ADAPTIVE_CRITIC_TRACE_DECAY
append_explicit_override "model.adaptive_critic.init_scale" ADAPTIVE_CRITIC_INIT_SCALE
append_explicit_override "model.adaptive_critic.min_scale" ADAPTIVE_CRITIC_MIN_SCALE
append_explicit_override "model.adaptive_critic.max_scale" ADAPTIVE_CRITIC_MAX_SCALE
append_explicit_override "model.adaptive_critic.warmup" ADAPTIVE_CRITIC_WARMUP
append_explicit_override "model.adaptive_critic.eps" ADAPTIVE_CRITIC_EPS
if [[ -z "${IDM_CONFIG_NAME}" ]]; then
  append_explicit_override "model.idm_type" IDM_TYPE
fi
append_explicit_override "model.idm_hidden_dim" IDM_HIDDEN_DIM
append_explicit_override "model.idm_output_dim" IDM_OUTPUT_DIM
append_explicit_override "model.rnd_output_dim" RND_OUTPUT_DIM
append_explicit_override "model.idm_n_layers" IDM_N_LAYERS
append_explicit_override "model.rnd_normalize" RND_NORMALIZE
append_explicit_override "model.idm_reward_norm_source" IDM_REWARD_NORM_SOURCE
append_explicit_override "model.expl_repr_dim" EXPL_REPR_DIM
append_explicit_override "model.expl_head_hidden_dim" EXPL_HEAD_HIDDEN_DIM
append_explicit_override "model.expl_head_layers" EXPL_HEAD_LAYERS
append_explicit_override "model.expl_obs_hidden_dim" EXPL_OBS_HIDDEN_DIM
append_explicit_override "model.expl_obs_layers" EXPL_OBS_LAYERS
append_explicit_override "model.expl_rnn_hidden_dim" EXPL_RNN_HIDDEN_DIM
append_explicit_override "model.expl_rnn_batch_chunk_size" EXPL_RNN_BATCH_CHUNK_SIZE
append_explicit_override "model.expl_obs_cnn_depth" EXPL_OBS_CNN_DEPTH
append_explicit_override "model.expl_obs_cnn_kernel_size" EXPL_OBS_CNN_KERNEL_SIZE
append_explicit_override "model.expl_obs_cnn_minres" EXPL_OBS_CNN_MINRES
append_explicit_override "model.expl_obs_cnn_chunk_size" EXPL_OBS_CNN_CHUNK_SIZE
append_explicit_override "model.expl_bridge_hidden_dim" EXPL_BRIDGE_HIDDEN_DIM
append_explicit_override "model.expl_bridge_layers" EXPL_BRIDGE_LAYERS
append_explicit_override "model.expl_bridge_loss_weight" EXPL_BRIDGE_LOSS_WEIGHT
append_explicit_override "model.expl_bridge_consistency_weight" EXPL_BRIDGE_CONSISTENCY_WEIGHT
append_explicit_override "model.expl_bridge_target_ema" EXPL_BRIDGE_TARGET_EMA
append_explicit_override "model.expl_bridge_target_ema_momentum" EXPL_BRIDGE_TARGET_EMA_MOMENTUM
append_explicit_override "model.disag_models" DISAG_MODELS
append_explicit_override "model.disag_action_cond" DISAG_ACTION_COND
append_explicit_override "model.disag_bootstrap" DISAG_BOOTSTRAP
append_explicit_override "model.disag_target" DISAG_TARGET
append_explicit_override "model.disag_normalize" DISAG_NORMALIZE
append_explicit_override "model.disag_log" DISAG_LOG
append_explicit_override "model.disag_offset" DISAG_OFFSET
append_explicit_override "wandb.enabled" USE_WANDB

if was_explicit TASK_NAME && [[ -n "${EVAL_TASKS_HYDRA}" ]]; then
  CMD+=("envs.eval_tasks=${EVAL_TASKS_HYDRA}")
fi
if [[ "${ENV_SUITE}" == "dmlab" ]]; then
  append_explicit_override "envs.dmlab_runfiles_path" DMLAB_RUNFILES_PATH
fi
if [[ "${ENV_SUITE}" == "atari" ]]; then
  append_explicit_override "envs.sticky" ATARI_STICKY
  append_explicit_override "envs.noops" ATARI_NOOPS
  append_explicit_override "envs.actions" ATARI_ACTIONS
  append_explicit_override "envs.gray" ATARI_GRAY
  append_explicit_override "envs.clip_reward" ATARI_CLIP_REWARD
fi
append_explicit_override "envs.backend" DMC_BACKEND

append_explicit_override "envs.eval_episode_num" EVAL_EPISODES
append_explicit_override "envs.env_num" ENV_NUM
append_explicit_override "envs.steps" STEPS
append_explicit_override "envs.train_ratio" TRAIN_RATIO
append_explicit_override "batch_size" BATCH_SIZE
append_explicit_override "batch_length" BATCH_LENGTH
append_explicit_override "envs.mjx_actor_chunk" MJX_ACTOR_CHUNK
append_explicit_override "envs.device_actor_chunk" DEVICE_ACTOR_CHUNK
append_explicit_override "trainer.eval_every" EVAL_EVERY
append_explicit_override "trainer.eval_action_selections" EVAL_ACTION_SELECTIONS
append_explicit_override "trainer.eval_episode_num" EVAL_EPISODES
append_explicit_override "trainer.eval_env_num" EVAL_ENV_NUM
append_explicit_override "trainer.replay_context" REPLAY_CONTEXT
append_explicit_override "trainer.save_every" SAVE_EVERY
append_explicit_override "trainer.update_log_every" LOG_EVERY
append_explicit_override "trainer.video_pred_log" VIDEO_PRED_LOG
append_explicit_override "trainer.video_policy_log" VIDEO_POLICY_LOG
append_explicit_override "adaptation.enabled" ADAPTATION_ENABLED
append_explicit_string_override "adaptation.protocol" ADAPTATION_PROTOCOL
append_explicit_string_override "adaptation.backend" ADAPTATION_BACKEND
append_explicit_string_override "adaptation.parent_algorithm" ADAPTATION_PARENT_ALGORITHM
append_explicit_string_override "adaptation.parent_task" ADAPTATION_PARENT_TASK
append_explicit_string_override "adaptation.parent_env_config" ADAPTATION_PARENT_ENV_CONFIG
append_explicit_string_override "adaptation.parent_idm" ADAPTATION_PARENT_IDM
append_explicit_override "adaptation.parent_seed" ADAPTATION_PARENT_SEED
append_explicit_string_override "adaptation.model" ADAPTATION_MODEL
append_explicit_string_override "adaptation.parent_run_uid" ADAPTATION_PARENT_RUN_UID
append_explicit_string_override "adaptation.parent_source_digest" ADAPTATION_PARENT_SOURCE_DIGEST
append_explicit_string_override "adaptation.parent_checkpoint" ADAPTATION_PARENT_CHECKPOINT
append_explicit_string_override "adaptation.parent_checkpoint_sha256" ADAPTATION_PARENT_CHECKPOINT_SHA256
append_explicit_string_override "adaptation.parent_replay" ADAPTATION_PARENT_REPLAY
append_explicit_string_override "adaptation.replay_inventory_sha256" ADAPTATION_REPLAY_INVENTORY_SHA256
append_explicit_string_override "adaptation.compatibility_certificate" ADAPTATION_COMPATIBILITY_CERTIFICATE
append_explicit_string_override "adaptation.compatibility_certificate_sha256" ADAPTATION_COMPATIBILITY_CERTIFICATE_SHA256
append_explicit_string_override "adaptation.cohort_validation_receipt" ADAPTATION_COHORT_VALIDATION_RECEIPT
append_explicit_string_override "adaptation.cohort_validation_receipt_sha256" ADAPTATION_COHORT_VALIDATION_RECEIPT_SHA256
append_explicit_override "adaptation.parent_replay_length" ADAPTATION_PARENT_REPLAY_LENGTH
append_explicit_override "adaptation.pretraining_environment_steps" PRETRAINING_ENVIRONMENT_STEPS
append_explicit_override "adaptation.offline_updates" ADAPTATION_OFFLINE_UPDATES
append_explicit_override "adaptation.offline_save_every" ADAPTATION_OFFLINE_SAVE_EVERY
if [[ "${ALGO_PRESET}" != "p2e" ]]; then
  append_explicit_override "adaptation.strict_task_blind_pretraining" STRICT_TASK_BLIND_PRETRAINING
fi

CMD+=("wandb.project=${WANDB_PROJECT}")
CMD+=("wandb.entity=${WANDB_ENTITY:-null}" "wandb.mode=${WANDB_MODE}")
CMD+=("wandb.name=$(quote_hydra_string "${WANDB_NAME}")")
CMD+=("wandb.id=$(quote_hydra_string "${WANDB_ID}")")
CMD+=("wandb.group=$(quote_hydra_string "${WANDB_GROUP}")")
if [[ -n "${WANDB_RESUME_FROM}" ]]; then
  CMD+=("wandb.resume=null")
  CMD+=("wandb.resume_from=$(quote_hydra_string "${WANDB_RESUME_FROM}")")
else
  CMD+=("wandb.resume=${WANDB_RESUME}")
  CMD+=("wandb.resume_from=null")
fi
CMD+=("wandb.notes=$(quote_hydra_string "${WANDB_NOTES}")")
if was_explicit WANDB_TAG && [[ -n "${WANDB_TAG}" ]]; then
  CMD+=("wandb.tags=[$(quote_hydra_string "${WANDB_TAG}")]")
fi
if [[ -n "${IDM_REWARD_NORMALIZE_WAS_SET}" ]]; then
  CMD+=("model.idm_reward_normalize=${IDM_REWARD_NORMALIZE}")
fi
if [[ -n "${EXPL_UNTIL_WAS_SET}" ]]; then
  CMD+=("model.expl_until=${EXPL_UNTIL}")
fi
if [[ -n "${EXPL_INTR_SCALE_WAS_SET}" ]]; then
  CMD+=("model.expl_intr_scale=${EXPL_INTR_SCALE}")
fi
if [[ -n "${PLAN_REPR_SOURCE_WAS_SET}" ]]; then
  CMD+=("model.plan_repr_source=${RESOLVED_PLAN_REPR_SOURCE}")
fi
if [[ -n "${EXPL_REPR_MODE_WAS_SET}" ]]; then
  CMD+=("model.expl_repr_mode=${RESOLVED_EXPL_REPR_MODE}")
fi
if [[ -n "${EXPL_GRAD_MODE_WAS_SET}" ]]; then
  CMD+=("model.expl_grad_mode=${RESOLVED_EXPL_GRAD_MODE}")
fi
if [[ -n "${EXPL_IMAG_REPR_MODE_WAS_SET}" ]]; then
  CMD+=("model.expl_imag_repr_mode=${RESOLVED_EXPL_IMAG_REPR_MODE}")
fi
if [[ "${RESOLVED_EXPL_REPR_MODE}" == "shared_head" && -n "${EXPL_HEAD_TYPE_WAS_SET}" ]]; then
  append_explicit_string_override "model.expl_head_type" EXPL_HEAD_TYPE
fi
if [[ "${RESOLVED_EXPL_REPR_MODE}" == "branch" && "${EXPL_OBS_ENCODER_TYPE}" != "null" ]]; then
  append_explicit_override "model.expl_obs_encoder_type" EXPL_OBS_ENCODER_TYPE
fi
case "${RESOLVED_EXPL_REPR_MODE}" in
  shared_head)
    ;;
  branch)
    ;;
  parallel)
    ;;
esac

if [[ "${SHOW_SKILLS}" == "true" ]]; then
  append_explicit_override "model.use_skills" USE_SKILLS
  append_explicit_override "model.skill_space" SKILL_SPACE
  append_explicit_override "model.num_skills" NUM_SKILLS
  append_explicit_override "model.skill_dim" SKILL_DIM
  if [[ "${IDM_TYPE}" != "csf" ]]; then
    append_explicit_override "model.skill_horizon" SKILL_HORIZON
    append_explicit_override "model.skill_embed_dim" SKILL_EMBED_DIM
  fi
fi

case "${IDM_TYPE}" in
  disagreement)
    append_explicit_override "model.disag_model_width_factor" DISAG_MODEL_WIDTH_FACTOR
    append_explicit_override "model.disag_units" DISAG_UNITS
    append_explicit_override "model.disag_layers" DISAG_LAYERS
    ;;
  rnd)
    ;;
  vime|vime_disagreement|vime_surprise)
    append_explicit_override "model.vime_prior_sigma" VIME_PRIOR_SIGMA
    append_explicit_override "model.vime_posterior_rho_init" VIME_POSTERIOR_RHO_INIT
    append_explicit_override "model.vime_likelihood_std" VIME_LIKELIHOOD_STD
    append_explicit_override "model.vime_kl_scale" VIME_KL_SCALE
    append_explicit_override "model.vime_info_gain_step_size" VIME_INFO_GAIN_STEP_SIZE
    append_explicit_override "model.vime_reward_history_size" VIME_REWARD_HISTORY_SIZE
    append_explicit_override "model.vime_reward_chunk_size" VIME_REWARD_CHUNK_SIZE
    append_explicit_override "model.vime_reward_samples" VIME_REWARD_SAMPLES
    append_explicit_override "model.vime_train_mc_samples" VIME_TRAIN_MC_SAMPLES
    ;;
  gamma_progress)
    append_explicit_override "model.gamma_progress_gamma" GAMMA_PROGRESS_GAMMA
    ;;
  empowerment_forward)
    append_explicit_override "model.empowerment_action_concat" ACTION_CONCAT
    ;;
  empowerment_gain)
    append_explicit_override "model.empowerment_gain_alpha" EMPOWERMENT_GAIN_ALPHA
    append_explicit_string_override "model.empowerment_gain_estimator" EMPOWERMENT_GAIN_ESTIMATOR
    if [[ "${EMPOWERMENT_GAIN_ESTIMATOR:-forward}" == "forward" ]]; then
      append_explicit_override "model.empowerment_action_concat" ACTION_CONCAT
    fi
    ;;
  empowerment_potential)
    append_explicit_override "model.empowerment_particles" EMPOWERMENT_PARTICLES
    append_explicit_override "model.empowerment_next_samples" EMPOWERMENT_NEXT_SAMPLES
    append_explicit_override "model.empowerment_q_loss_weight" EMPOWERMENT_Q_LOSS_WEIGHT
    append_explicit_override "model.empowerment_action_concat" ACTION_CONCAT
    append_explicit_override "model.empowerment_reward_chunk_size" EMPOWERMENT_REWARD_CHUNK_SIZE
    ;;
  empowerment_process)
    append_explicit_override "model.empowerment_particles" EMPOWERMENT_PARTICLES
    append_explicit_override "model.empowerment_next_samples" EMPOWERMENT_NEXT_SAMPLES
    append_explicit_override "model.empowerment_action_concat" ACTION_CONCAT
    append_explicit_override "model.empowerment_reward_chunk_size" EMPOWERMENT_REWARD_CHUNK_SIZE
    ;;
  diayn)
    append_explicit_string_override "model.diayn_prior" DIAYN_PRIOR
    append_explicit_string_override "model.diayn_prior_kwargs" DIAYN_PRIOR_KWARGS
    append_explicit_override "model.diayn_entropy_coef" DIAYN_ENTROPY_COEF
    append_explicit_override "model.diayn_replay_batch_size" DIAYN_REPLAY_BATCH_SIZE
    append_explicit_string_override "model.continuous_skill_posterior" CONTINUOUS_SKILL_POSTERIOR
    append_explicit_override "model.continuous_skill_temperature" CONTINUOUS_SKILL_TEMPERATURE
    ;;
  dads)
    append_explicit_string_override "model.dads_prior" DADS_PRIOR
    append_explicit_string_override "model.dads_prior_kwargs" DADS_PRIOR_KWARGS
    append_explicit_override "model.dads_entropy_coef" DADS_ENTROPY_COEF
    append_explicit_override "model.dads_use_marginal_model" DADS_USE_MARGINAL_MODEL
    append_explicit_override "model.dads_marginal_samples" DADS_MARGINAL_SAMPLES
    append_explicit_override "model.dads_replay_batch_size" DADS_REPLAY_BATCH_SIZE
    ;;
  rdads)
    append_explicit_string_override "model.rdads_prior" RDADS_PRIOR
    append_explicit_string_override "model.rdads_prior_kwargs" RDADS_PRIOR_KWARGS
    append_explicit_override "model.rdads_entropy_coef" RDADS_ENTROPY_COEF
    append_explicit_override "model.rdads_replay_batch_size" RDADS_REPLAY_BATCH_SIZE
    append_explicit_string_override "model.continuous_skill_posterior" CONTINUOUS_SKILL_POSTERIOR
    append_explicit_override "model.continuous_skill_temperature" CONTINUOUS_SKILL_TEMPERATURE
    ;;
  vic)
    append_explicit_string_override "model.vic_prior" VIC_PRIOR
    append_explicit_string_override "model.vic_prior_kwargs" VIC_PRIOR_KWARGS
    append_explicit_override "model.vic_entropy_coef" VIC_ENTROPY_COEF
    append_explicit_override "model.vic_condition_on_start" VIC_CONDITION_ON_START
    append_explicit_override "model.vic_reward_at_end" VIC_REWARD_AT_END
    append_explicit_override "model.vic_replay_batch_size" VIC_REPLAY_BATCH_SIZE
    append_explicit_string_override "model.continuous_skill_posterior" CONTINUOUS_SKILL_POSTERIOR
    append_explicit_override "model.continuous_skill_temperature" CONTINUOUS_SKILL_TEMPERATURE
    ;;
  valor)
    append_explicit_string_override "model.valor_prior" VALOR_PRIOR
    append_explicit_string_override "model.valor_prior_kwargs" VALOR_PRIOR_KWARGS
    append_explicit_override "model.valor_entropy_coef" VALOR_ENTROPY_COEF
    append_explicit_override "model.valor_mlp_layers" VALOR_MLP_LAYERS
    append_explicit_override "model.valor_num_subseq" VALOR_NUM_SUBSEQ
    append_explicit_override "model.valor_reward_at_end" VALOR_REWARD_AT_END
    append_explicit_override "model.valor_subtract_log_prior" VALOR_SUBTRACT_LOG_PRIOR
    append_explicit_override "model.valor_head_hidden_dim" VALOR_HEAD_HIDDEN_DIM
    append_explicit_override "model.valor_replay_batch_size" VALOR_REPLAY_BATCH_SIZE
    append_explicit_string_override "model.continuous_skill_posterior" CONTINUOUS_SKILL_POSTERIOR
    append_explicit_override "model.continuous_skill_temperature" CONTINUOUS_SKILL_TEMPERATURE
    ;;
  apt)
    append_explicit_override "model.apt_rep_dim" APT_REP_DIM
    append_explicit_override "model.apt_k" APT_K
    append_explicit_override "model.apt_normalize_phi" APT_NORMALIZE_PHI
    append_explicit_override "model.apt_train_phi" APT_TRAIN_PHI
    append_explicit_override "model.apt_reward_normalize" APT_REWARD_NORMALIZE
    append_explicit_override "model.apt_knn_chunk_size" APT_KNN_CHUNK_SIZE
    append_explicit_override "model.apt_memory_size" APT_MEMORY_SIZE
    append_explicit_override "model.apt_memory_sample_size" APT_MEMORY_SAMPLE_SIZE
    ;;
  re3)
    append_explicit_override "model.re3_rep_dim" RE3_REP_DIM
    append_explicit_override "model.re3_k" RE3_K
    append_explicit_override "model.re3_log_reward" RE3_LOG_REWARD
    append_explicit_override "model.re3_reward_normalize" RE3_REWARD_NORMALIZE
    append_explicit_override "model.re3_knn_chunk_size" RE3_KNN_CHUNK_SIZE
    append_explicit_override "model.re3_memory_size" RE3_MEMORY_SIZE
    append_explicit_override "model.re3_memory_sample_size" RE3_MEMORY_SAMPLE_SIZE
    ;;
  ngu)
    append_explicit_override "model.ngu_rep_dim" NGU_REP_DIM
    append_explicit_override "model.ngu_k" NGU_K
    append_explicit_override "model.ngu_kernel_epsilon" NGU_KERNEL_EPSILON
    append_explicit_override "model.ngu_cluster_distance" NGU_CLUSTER_DISTANCE
    append_explicit_override "model.ngu_pseudo_count_c" NGU_PSEUDO_COUNT_C
    append_explicit_override "model.ngu_max_similarity" NGU_MAX_SIMILARITY
    append_explicit_override "model.ngu_lifelong_clip" NGU_LIFELONG_CLIP
    append_explicit_override "model.ngu_use_rnd" NGU_USE_RND
    append_explicit_override "model.ngu_rnd_hidden_dim" NGU_RND_HIDDEN_DIM
    append_explicit_override "model.ngu_rnd_output_dim" NGU_RND_OUTPUT_DIM
    append_explicit_override "model.ngu_rnd_layers" NGU_RND_LAYERS
    append_explicit_override "model.ngu_inverse_coef" NGU_INVERSE_COEF
    append_explicit_override "model.ngu_rnd_coef" NGU_RND_COEF
    append_explicit_override "model.ngu_vectorized_scan" NGU_VECTORIZED_SCAN
    append_explicit_override "model.ngu_update_tail" NGU_UPDATE_TAIL
    ;;
  proto_rl)
    append_explicit_override "model.proto_rl_rep_dim" PROTO_RL_REP_DIM
    append_explicit_override "model.proto_rl_num_prototypes" PROTO_RL_NUM_PROTOTYPES
    append_explicit_override "model.proto_rl_candidates_per_prototype" PROTO_RL_CANDIDATES_PER_PROTOTYPE
    append_explicit_override "model.proto_rl_temperature" PROTO_RL_TEMPERATURE
    append_explicit_override "model.proto_rl_sinkhorn_iters" PROTO_RL_SINKHORN_ITERS
    append_explicit_override "model.proto_rl_target_momentum" PROTO_RL_TARGET_MOMENTUM
    append_explicit_override "model.proto_rl_knn_k" PROTO_RL_KNN_K
    append_explicit_override "model.proto_rl_reward_normalize" PROTO_RL_REWARD_NORMALIZE
    ;;
  ctec)
    append_explicit_override "model.ctec_rep_dim" CTEC_REP_DIM
    append_explicit_override "model.ctec_hidden_dim" CTEC_HIDDEN_DIM
    append_explicit_override "model.ctec_layers" CTEC_LAYERS
    append_explicit_override "model.ctec_gamma" CTEC_GAMMA
    append_explicit_override "model.ctec_temperature" CTEC_TEMPERATURE
    append_explicit_override "model.ctec_learn_temperature" CTEC_LEARN_TEMPERATURE
    append_explicit_string_override "model.ctec_similarity" CTEC_SIMILARITY
    append_explicit_string_override "model.ctec_loss_type" CTEC_LOSS_TYPE
    append_explicit_override "model.ctec_normalize_embeddings" CTEC_NORMALIZE_EMBEDDINGS
    append_explicit_override "model.ctec_replay_batch_size" CTEC_REPLAY_BATCH_SIZE
    append_explicit_string_override "model.ctec_reward_estimator" CTEC_REWARD_ESTIMATOR
    append_explicit_override "model.ctec_reward_normalize_truncation" CTEC_REWARD_NORMALIZE_TRUNCATION
    ;;
  cip)
    append_explicit_string_override "model.cip_state_source" CIP_STATE_SOURCE
    append_explicit_override "model.cip_imag_last" CIP_IMAG_LAST
    append_explicit_override "model.cip_probes" CIP_PROBES
    append_explicit_override "model.cip_scale" CIP_SCALE
    append_explicit_override "model.cip_eps" CIP_EPS
    append_explicit_override "model.cip_positive" CIP_POSITIVE
    append_explicit_override "model.cip_normalize" CIP_NORMALIZE
    append_explicit_override "model.cip_seed" CIP_SEED
    append_explicit_override "model.cip_stop_reward_grad" CIP_STOP_REWARD_GRAD
    append_explicit_override "model.cip_replay_metric" CIP_REPLAY_METRIC
    ;;
esac

if [[ "${IDM_TYPE}" == "csf" ]]; then
  append_explicit_override "model.csf_prior" CSF_PRIOR
  append_explicit_override "model.csf_proj_dim" CSF_PROJ_DIM
  append_explicit_override "model.csf_proj_hidden_dim" CSF_PROJ_HIDDEN_DIM
  append_explicit_override "model.csf_proj_layers" CSF_PROJ_LAYERS
  append_explicit_override "model.csf_reward_mode" CSF_REWARD_MODE
  append_explicit_override "model.csf_posterior" CSF_POSTERIOR
  append_explicit_override "model.csf_input_norm" CSF_INPUT_NORM
  append_explicit_override "model.csf_num_negatives" CSF_NUM_NEGATIVES
  append_explicit_override "model.csf_replay_batch_size" CSF_REPLAY_BATCH_SIZE
fi

# Immutable run identity and storage initialization

if [[ ! -f "${RUN_STORAGE_HELPER}" ]]; then
  echo "Required run provenance helper is missing: ${RUN_STORAGE_HELPER}" >&2
  exit 1
fi

RUN_LAUNCHER_PATH="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
if [[ ! -f "${RUN_LAUNCHER_PATH}" || -L "${RUN_LAUNCHER_PATH}" ]]; then
  echo "Refusing unsafe launcher source: ${RUN_LAUNCHER_PATH}" >&2
  exit 1
fi
RUN_LAUNCHER_SHA256="$(sha256sum -- "${RUN_LAUNCHER_PATH}" | awk '{print $1}')"
RUN_SOURCE_ID="$(python3 "${RUN_STORAGE_HELPER}" source-id --repo "${SSRL_RUN_REPO_ROOT}")"
RUN_SOURCE_COMMIT="$(git -C "${SSRL_RUN_REPO_ROOT}" rev-parse HEAD 2>/dev/null || printf unknown)"
RUN_RUNTIME_DESCRIPTION="$(
  python3 -c 'import importlib.metadata as m, platform, sys; print("python=" + platform.python_implementation() + "-" + platform.python_version()); print("executable=" + sys.executable); print("packages:"); print("\n".join(sorted({((d.metadata.get("Name") or "unknown").lower() + "==" + d.version) for d in m.distributions()})))'
  if command -v nvidia-smi >/dev/null 2>&1; then
    printf 'gpu_drivers:\n'
    nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | sort -u || true
  fi
)"
RUN_RUNTIME_SHA256="$(printf '%s\n' "${RUN_RUNTIME_DESCRIPTION}" | sha256sum | awk '{print $1}')"
if [[ "${DEVICE,,}" == cuda* ]] && command -v nvidia-smi >/dev/null 2>&1; then
  if ! _jax_cache_hardware_description="$(
    nvidia-smi \
      --query-gpu=name,compute_cap,driver_version \
      --format=csv,noheader,nounits 2>/dev/null | sort -u
  )" || [[ -z "${_jax_cache_hardware_description}" ]]; then
    _jax_cache_hardware_description="$(
      nvidia-smi \
        --query-gpu=name,driver_version \
        --format=csv,noheader,nounits 2>/dev/null | sort -u || true
    )"
  fi
else
  _jax_cache_hardware_description="device=${DEVICE};machine=$(uname -m)"
fi
SSRL_JAX_CACHE_HARDWARE_ID="$(
  printf '%s\n' "${_jax_cache_hardware_description}" | sha256sum | awk '{print $1}'
)"
unset _jax_cache_hardware_description
if [[ -n "${SSRL_SUBMISSION_SOURCE_ID:-}" &&
      "${SSRL_SUBMISSION_SOURCE_ID}" != "${RUN_SOURCE_ID}" ]]; then
  cat >&2 <<EOF
Refusing to start because the checkout changed after array submission.
  submitted=${SSRL_SUBMISSION_SOURCE_ID}
  current=${RUN_SOURCE_ID}

Submit a new array from the current checkout instead of mixing source versions.
EOF
  exit 1
fi

RUN_COMMAND_DISPLAY=""
for _run_arg in "${CMD[@]}"; do
  [[ "${_run_arg}" == wandb.resume=* ||
     "${_run_arg}" == wandb.resume_from=* ||
     "${_run_arg}" == wandb.id=* ]] && continue
  printf -v _run_quoted_arg '%q' "${_run_arg}"
  RUN_COMMAND_DISPLAY+="${_run_quoted_arg} "
done
RUN_COMMAND_DISPLAY="${RUN_COMMAND_DISPLAY% }"
RUN_COMMAND_DISPLAY+=" launcher_sha256=${RUN_LAUNCHER_SHA256}"
RUN_COMMAND_SHA256="$(
  python3 "${RUN_STORAGE_HELPER}" config-id \
    --launcher-sha256 "${RUN_LAUNCHER_SHA256}" -- "${CMD[@]}"
)"
unset _run_arg _run_quoted_arg

RUN_ID="${RUN_UID}"
RUN_FAMILY="${RUN_FAMILY_NAME}"
RUN_EXPERIMENT="${EXPERIMENT_NAME}"
RUN_SEED="${SEED}"
RUN_ENV_CONFIG="${CFG}"
RUN_TASK="${ENV_TASK}"
RUN_ALGORITHM="${ALGO_PRESET}"
RUN_IDM="${IDM_TYPE}"
RUN_AGENT="${RESOLVED_AGENT_NAME}"
RUN_MODEL="${MODEL_SIZE}"
RUN_WANDB_ID="${WANDB_LOGICAL_ID}"
RUN_WANDB_ATTEMPT_ID="${WANDB_ATTEMPT_ID}"
RUN_WANDB_PROJECT="${WANDB_PROJECT}"
RUN_CONFIG_ID="${RUN_COMMAND_SHA256}"
RUN_SLURM_JOB_ID="${SLURM_JOB_ID:-}"
RUN_SLURM_ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-}"
RUN_SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
RUN_SLURM_RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"
RUN_HOSTNAME="$(hostname)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  _slurm_log_root="${SLURM_SUBMIT_DIR:-${SSRL_RUN_REPO_ROOT}}/local_data/slurm"
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    _slurm_log_stem="slurm-${SLURM_JOB_NAME:-job}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  else
    _slurm_log_stem="slurm-${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID}"
  fi
  RUN_SLURM_STDOUT="${_slurm_log_root}/out/${_slurm_log_stem}.out"
  RUN_SLURM_STDERR="${_slurm_log_root}/err/${_slurm_log_stem}.err"
  unset _slurm_log_root _slurm_log_stem
  _fd_stdout="$(readlink -f "/proc/$$/fd/1" 2>/dev/null || true)"
  _fd_stderr="$(readlink -f "/proc/$$/fd/2" 2>/dev/null || true)"
  [[ -n "${_fd_stdout}" && -f "${_fd_stdout}" ]] && RUN_SLURM_STDOUT="${_fd_stdout}"
  [[ -n "${_fd_stderr}" && -f "${_fd_stderr}" ]] && RUN_SLURM_STDERR="${_fd_stderr}"
  unset _fd_stdout _fd_stderr
fi

export RUN_UID RUN_NAME EXPERIMENT_NAME RUN_FAMILY_NAME SEED MODEL_SIZE
export RESOLVED_AGENT_NAME ALGO_PRESET IDM_TYPE CFG ENV_TASK WANDB_ID
export WANDB_PROJECT WANDB_ENTITY RUN_STORAGE_MODE_RESOLVED
export RUN_SOURCE_ID RUN_SOURCE_COMMIT RUN_COMMAND_DISPLAY RUN_COMMAND_SHA256
export RUN_LAUNCHER_SHA256
export RUN_RUNTIME_SHA256
export JAX_COMPILATION_CACHE_ROOT SSRL_JAX_CACHE_HARDWARE_ID
export RUN_ATTEMPT_ID RUN_STORAGE_HELPER WORK_LOG_DIR LOG_DIR
SCRATCH_SYNC_REQUIRE_SOURCE_SNAPSHOT=1
export SCRATCH_SYNC_REQUIRE_SOURCE_SNAPSHOT
export RUN_SLURM_STDOUT RUN_SLURM_STDERR SSRL_RUN_REPO_ROOT
export SLURM_JOB_ID SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_RESTART_COUNT
export SLURM_CLUSTER_NAME SLURM_JOB_NAME
export RUN_ID RUN_FAMILY RUN_EXPERIMENT RUN_SEED RUN_ENV_CONFIG RUN_TASK
export RUN_ALGORITHM RUN_IDM RUN_AGENT RUN_MODEL RUN_WANDB_ID
export RUN_STRICT_TASK_BLIND_PRETRAINING
export RUN_WANDB_ATTEMPT_ID RUN_WANDB_PROJECT RUN_CONFIG_ID RUN_HOSTNAME
export RUN_SLURM_JOB_ID RUN_SLURM_ARRAY_JOB_ID RUN_SLURM_ARRAY_TASK_ID
export RUN_SLURM_RESTART_COUNT
export RUN_FORK_SOURCE_RUN_UID RUN_FORK_PARENT_SOURCE_ID RUN_FORK_SOURCE_DIR

if ! enable_scratch_log_sync; then
  echo "Run storage initialization failed; training was not started." >&2
  exit 1
fi
if [[ "${WANDB_RELAUNCH_MODE}" == "continue" &&
      -n "${WANDB_CHECKPOINT_STEP:-}" ]]; then
  _selected_checkpoint="$(scratch_sync_checkpoint_latest \
    "${LOG_DIR}/checkpoint.pkl")" || {
      echo "W&B continuation selected a checkpoint but restored storage has none." >&2
      scratch_sync_stop_recovery_monitor || true
      trap - EXIT
      scratch_sync_release_lock
      exit 1
    }
  _selected_checkpoint_step="$(scratch_sync_checkpoint_step \
    "${LOG_DIR}" "${_selected_checkpoint}")" || {
      echo "Could not authenticate the restored checkpoint step for W&B continuation." >&2
      scratch_sync_stop_recovery_monitor || true
      trap - EXIT
      scratch_sync_release_lock
      exit 1
    }
  if [[ "${_selected_checkpoint}" != "${WANDB_CHECKPOINT_NAME}" ||
        "${_selected_checkpoint_step}" != "${WANDB_CHECKPOINT_STEP}" ]]; then
    echo "W&B continuation/storage selector mismatch; training was not started." >&2
    echo "  W&B checkpoint=${WANDB_CHECKPOINT_NAME} step=${WANDB_CHECKPOINT_STEP}" >&2
    echo "  restored checkpoint=${_selected_checkpoint} step=${_selected_checkpoint_step}" >&2
    scratch_sync_stop_recovery_monitor || true
    trap - EXIT
    scratch_sync_release_lock
    exit 1
  fi
  unset _selected_checkpoint _selected_checkpoint_step
fi

snapshot_effective_launcher() {
  local parent attempt_dir destination temporary actual_sha
  parent="${LOG_DIR}/attempts"
  attempt_dir="${parent}/${RUN_ATTEMPT_ID}"
  destination="${attempt_dir}/launcher.sh"
  if [[ ( -e "${parent}" || -L "${parent}" ) &&
        ( ! -d "${parent}" || -L "${parent}" ) ]]; then
    echo "Unsafe launcher snapshot directory: ${parent}" >&2
    return 1
  fi
  mkdir -p -- "${attempt_dir}" || return 1
  if [[ ! -d "${attempt_dir}" || -L "${attempt_dir}" ]]; then
    echo "Unsafe launcher attempt directory: ${attempt_dir}" >&2
    return 1
  fi
  if [[ -e "${destination}" || -L "${destination}" ]]; then
    if [[ ! -f "${destination}" || -L "${destination}" ]] ||
       ! cmp -s -- "${RUN_LAUNCHER_PATH}" "${destination}"; then
      echo "Refusing to replace a differing launcher snapshot: ${destination}" >&2
      return 1
    fi
    return 0
  fi
  temporary="$(mktemp "${attempt_dir}/.launcher.tmp.XXXXXX")" || return 1
  if ! cp -- "${RUN_LAUNCHER_PATH}" "${temporary}" ||
     ! chmod 0440 "${temporary}" ||
     ! sync -f "${temporary}"; then
    rm -f -- "${temporary}" 2>/dev/null || true
    return 1
  fi
  actual_sha="$(sha256sum -- "${temporary}" | awk '{print $1}')"
  if [[ "${actual_sha}" != "${RUN_LAUNCHER_SHA256}" ]] ||
     ! ln -- "${temporary}" "${destination}" ||
     ! rm -f -- "${temporary}" ||
     ! sync -f "${attempt_dir}"; then
    rm -f -- "${temporary}" 2>/dev/null || true
    echo "Failed to atomically preserve the effective launcher." >&2
    return 1
  fi
}

snapshot_runtime_environment() {
  local attempt_dir destination temporary actual_sha
  attempt_dir="${LOG_DIR}/attempts/${RUN_ATTEMPT_ID}"
  destination="${attempt_dir}/runtime.txt"
  if [[ ! -d "${attempt_dir}" || -L "${attempt_dir}" ]]; then
    echo "Unsafe runtime snapshot directory: ${attempt_dir}" >&2
    return 1
  fi
  if [[ -e "${destination}" || -L "${destination}" ]]; then
    if [[ ! -f "${destination}" || -L "${destination}" ]] ||
       [[ "$(sha256sum -- "${destination}" | awk '{print $1}')" != "${RUN_RUNTIME_SHA256}" ]]; then
      echo "Refusing to replace a differing runtime snapshot: ${destination}" >&2
      return 1
    fi
    return 0
  fi
  temporary="$(mktemp "${attempt_dir}/.runtime.tmp.XXXXXX")" || return 1
  if ! printf '%s\n' "${RUN_RUNTIME_DESCRIPTION}" > "${temporary}" ||
     ! chmod 0440 "${temporary}" ||
     ! sync -f "${temporary}"; then
    rm -f -- "${temporary}" 2>/dev/null || true
    return 1
  fi
  actual_sha="$(sha256sum -- "${temporary}" | awk '{print $1}')"
  if [[ "${actual_sha}" != "${RUN_RUNTIME_SHA256}" ]] ||
     ! ln -- "${temporary}" "${destination}" ||
     ! rm -f -- "${temporary}" ||
     ! sync -f "${attempt_dir}"; then
    rm -f -- "${temporary}" 2>/dev/null || true
    echo "Failed to atomically preserve the runtime environment." >&2
    return 1
  fi
}

if [[ "${DRY_RUN,,}" != "true" ]]; then
  if ! snapshot_effective_launcher; then
    echo "Launcher provenance capture failed; training was not started." >&2
    exit 1
  fi
  if ! snapshot_runtime_environment; then
    echo "Runtime provenance capture failed; training was not started." >&2
    exit 1
  fi

  SOURCE_SNAPSHOT_DIR="${LOG_DIR}/source_provenance"
  if [[ -e "${SOURCE_SNAPSHOT_DIR}" || -L "${SOURCE_SNAPSHOT_DIR}" ]]; then
    if [[ ! -d "${SOURCE_SNAPSHOT_DIR}" || -L "${SOURCE_SNAPSHOT_DIR}" ]] ||
       ! python3 "${RUN_STORAGE_HELPER}" source-snapshot verify \
          --snapshot-dir "${SOURCE_SNAPSHOT_DIR}" --source-id "${RUN_SOURCE_ID}" \
          >/dev/null; then
      echo "Immutable source snapshot verification failed; training was not started." >&2
      exit 1
    fi
  elif [[ "${RUN_STORAGE_MODE_RESOLVED}" == "new" ||
          "${SCRATCH_SYNC_FORKED:-0}" -eq 1 ||
          "${RUN_ALLOW_LEGACY_RESUME:-false}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    if ! python3 "${RUN_STORAGE_HELPER}" source-snapshot create \
        --repo "${SSRL_RUN_REPO_ROOT}" --snapshot-dir "${SOURCE_SNAPSHOT_DIR}" \
        --source-id "${RUN_SOURCE_ID}" >/dev/null; then
      echo "Immutable source snapshot creation failed; training was not started." >&2
      exit 1
    fi
  else
    echo "Resume source snapshot is missing: ${SOURCE_SNAPSHOT_DIR}" >&2
    echo "Refusing an unverifiable resume; use a new run directory." >&2
    exit 1
  fi
fi

# Display

case "${ENV_SUITE}" in
  dmc) DISPLAY_SUITE="DMC" ;;
  dmlab) DISPLAY_SUITE="DMLab" ;;
  atari) DISPLAY_SUITE="Atari" ;;
  craftax) DISPLAY_SUITE="Craftax" ;;
  *) DISPLAY_SUITE="${ENV_SUITE}" ;;
esac

echo "================================================================================"
echo "  ${DISPLAY_SUITE} ${OBS_MODE} Task Training (${ALGO_PRESET})"
echo "================================================================================"
echo ""
echo "Date: $(date)"
echo "Hostname: $(hostname)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "SLURM Job ID: ${SLURM_JOB_ID}"
  echo "SLURM Node: ${SLURM_NODELIST:-unknown}"
fi
echo ""
echo "--- Environment ---"
echo "  Task: ${ENV_TASK}"
if [[ "${TRAIN_TASK_NAME}" != "${TASK_NAME}" ]]; then
  echo "  Train Task: ${DOMAIN_NAME}_${TRAIN_TASK_NAME}"
fi
if [[ -n "${EVAL_TASKS_DISPLAY}" ]]; then
  echo "  Eval Tasks: ${EVAL_TASKS_DISPLAY}"
fi
if [[ "${ENV_SUITE}" == "dmlab" ]]; then
  if [[ -n "${DMLAB_RUNFILES_PATH}" ]]; then
    echo "  DMLab Runfiles Path: ${DMLAB_RUNFILES_PATH}"
  fi
elif [[ "${ENV_SUITE}" == "atari" ]]; then
  print_config_value "Sticky Actions" ATARI_STICKY
  print_config_value "No-op Reset Max" ATARI_NOOPS
  print_config_value "Action Set" ATARI_ACTIONS
  print_config_value "Gray Observations" ATARI_GRAY
  print_config_value "Clip Reward" ATARI_CLIP_REWARD
else
  echo "  Rendering: ${MUJOCO_GL} (PyOpenGL: ${PYOPENGL_PLATFORM})"
fi
echo ""
echo "--- Training ---"
echo "  Algo Preset: ${ALGO_PRESET}"
echo "  Model Preset: ${MODEL_SIZE}"
if [[ "${ALGO_PRESET}" == "p2e" ]]; then
  echo "  Strict Task-Blind Pretraining: ${RUN_STRICT_TASK_BLIND_PRETRAINING}"
fi
echo "  Schedule: loaded from Hydra config"
if [[ -n "${ENV_NUM:-}" ]]; then
  echo "  Env Num Override: ${ENV_NUM}"
fi
if [[ -n "${STEPS:-}" ]]; then
  echo "  Steps Override: ${STEPS}"
fi
if [[ -n "${TRAIN_RATIO:-}" ]]; then
  echo "  Train Ratio Override: ${TRAIN_RATIO}"
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  echo "  Batch Size Override: ${BATCH_SIZE}"
fi
if [[ -n "${BATCH_LENGTH:-}" ]]; then
  echo "  Batch Length Override: ${BATCH_LENGTH}"
fi
if [[ -n "${MJX_ACTOR_CHUNK:-}" ]]; then
  echo "  MJX Actor Chunk Override: ${MJX_ACTOR_CHUNK}"
fi
if [[ -n "${DEVICE_ACTOR_CHUNK:-}" ]]; then
  echo "  Device Actor Chunk Override: ${DEVICE_ACTOR_CHUNK}"
fi
if [[ -n "${REPLAY_CONTEXT:-}" ]]; then
  echo "  Replay Context Override: ${REPLAY_CONTEXT}"
fi
if [[ -n "${IMAG_LAST:-}" ]]; then
  echo "  Imag Last Override: ${IMAG_LAST}"
fi
if [[ -n "${JAX_AUTOTUNE:-}" ]]; then
  echo "  JAX Autotune Override: ${JAX_AUTOTUNE}"
fi
if [[ -n "${JAX_PROFILER:-}" ]]; then
  echo "  JAX Profiler Override: ${JAX_PROFILER}"
fi
if [[ -n "${JAX_DONATE_POLICY_PARAMS:-}" ]]; then
  echo "  Donate Policy Parameters Override: ${JAX_DONATE_POLICY_PARAMS}"
fi
if [[ -n "${LOG_GRADS:-}" ]]; then
  echo "  Gradient Diagnostics Override: ${LOG_GRADS}"
fi
if [[ -n "${RSSM_UNROLL:-}" ]]; then
  echo "  RSSM Scan Unroll Override: ${RSSM_UNROLL}"
fi
if was_explicit ADAPTIVE_CRITIC_ENABLED &&
    [[ -n "${ADAPTIVE_CRITIC_ENABLED:-}" ]]; then
  echo "  Adaptive Critic: ${ADAPTIVE_CRITIC_ENABLED}"
  print_config_value "Adaptive Critics" ADAPTIVE_CRITIC_CRITICS
  print_config_value "Adaptive Scope" ADAPTIVE_CRITIC_SCOPE
  print_config_value "Adaptive Granularity" ADAPTIVE_CRITIC_GRANULARITY
  print_config_value "Adaptive Meta LR" ADAPTIVE_CRITIC_META_LR
  print_config_value "Adaptive Initial Scale" ADAPTIVE_CRITIC_INIT_SCALE
fi
if [[ -n "${JAX_PREALLOCATE:-}" ]]; then
  echo "  JAX Preallocate Override: ${JAX_PREALLOCATE}"
fi
if was_explicit JAX_COMPILATION_CACHE; then
  echo "  JAX Compilation Cache Override: ${JAX_COMPILATION_CACHE}"
else
  echo "  JAX Compilation Cache: loaded from Hydra config"
fi
if [[ -n "${JAX_COMPILATION_CACHE_DIR:-}" ]]; then
  echo "  JAX Compilation Cache Directory: ${JAX_COMPILATION_CACHE_DIR}"
else
  echo "  JAX Compilation Cache Root: ${JAX_COMPILATION_CACHE_ROOT}"
fi
echo "  Device: ${DEVICE}"
echo "  Seed: ${SEED}"
echo ""
echo "--- Logging ---"
echo "  Use Wandb: ${USE_WANDB}"
echo "  Wandb Project: ${WANDB_PROJECT}"
if was_explicit WANDB_NAME; then
  echo "  Wandb Run Name Override: ${WANDB_NAME}"
else
  echo "  Wandb Run Name: ${WANDB_NAME}"
fi
echo "  Wandb Tag: ${WANDB_TAG}"
if was_explicit WANDB_ID && [[ -n "${WANDB_ID}" ]]; then
  echo "  Wandb Logical ID Override: ${WANDB_LOGICAL_ID}"
elif [[ -n "${WANDB_LOGICAL_ID}" ]]; then
  echo "  Wandb Logical ID: ${WANDB_LOGICAL_ID}"
fi
echo "  Wandb Attempt ID: ${WANDB_ATTEMPT_ID}"
echo "  Wandb Group: ${WANDB_GROUP}"
echo "  Wandb Relaunch Mode: ${WANDB_RELAUNCH_MODE}"
if [[ -n "${WANDB_RESUME_FROM}" ]]; then
  echo "  Wandb Resume From: ${WANDB_RESUME_FROM}"
  echo "  Durable Checkpoint Step: ${WANDB_CHECKPOINT_STEP}"
fi
if [[ -n "${LOG_EVERY:-}" ]]; then
  echo "  Update Log Every: ${LOG_EVERY}"
fi
if [[ -n "${EVAL_EVERY:-}" ]]; then
  echo "  Eval Every: ${EVAL_EVERY}"
fi
if [[ -n "${EVAL_EPISODES:-}" ]]; then
  echo "  Eval Episodes: ${EVAL_EPISODES}"
fi
if [[ -n "${VIDEO_PRED_LOG:-}" ]]; then
  echo "  Video Pred Log: ${VIDEO_PRED_LOG}"
fi
if [[ -n "${VIDEO_POLICY_LOG:-}" ]]; then
  echo "  Video Policy Log: ${VIDEO_POLICY_LOG}"
fi
echo "  Wandb Notes: ${WANDB_NOTES}"
echo ""
echo "--- Exploration (${IDM_TYPE}) ---"
echo "  Behavior: ${EXPL_BEHAVIOR}"
echo "  Plan Repr Source: ${RESOLVED_PLAN_REPR_SOURCE}"
echo "  Replay Repr Mode: ${RESOLVED_EXPL_REPR_MODE}"
if [[ "${RESOLVED_EXPL_REPR_MODE}" == "branch" || "${RESOLVED_EXPL_REPR_MODE}" == "parallel" ]]; then
  echo "  Imag Repr Mode: ${RESOLVED_EXPL_IMAG_REPR_MODE}"
fi
echo "  Expl Grad Mode: ${RESOLVED_EXPL_GRAD_MODE}"
echo "  Expl Repr Dim: ${RESOLVED_EXPL_REPR_DIM}"
print_config_value "Intrinsic Scale" EXPL_INTR_SCALE
print_config_value "IDM Reward Normalize" IDM_REWARD_NORMALIZE
print_config_value "IDM Reward Norm Source" IDM_REWARD_NORM_SOURCE
if was_explicit IDM_HIDDEN_DIM; then
  echo "  IDM Hidden Dim Override: ${IDM_HIDDEN_DIM}"
fi
if was_explicit IDM_N_LAYERS; then
  echo "  IDM Layers Override: ${IDM_N_LAYERS}"
fi
if ! was_explicit IDM_HIDDEN_DIM && ! was_explicit IDM_N_LAYERS; then
  echo "  IDM Architecture: loaded from Hydra config"
fi
echo ""

if [[ "${SHOW_SKILLS}" == "true" ]]; then
  echo "--- Skills ---"
  print_config_value "Use Skills" USE_SKILLS
  print_config_value "Skill Space" SKILL_SPACE
  print_config_value "Num Skills" NUM_SKILLS
  print_config_value "Skill Dim" SKILL_DIM
  print_config_value "Skill Horizon" SKILL_HORIZON
  print_config_value "Skill Embed Dim" SKILL_EMBED_DIM
  echo ""
fi

if [[ "${IDM_TYPE}" == "csf" ]]; then
  echo "--- CSF ---"
  print_config_value "CSF Prior" CSF_PRIOR
  print_config_value "CSF Proj Dim" CSF_PROJ_DIM
  print_config_value "CSF Reward Mode" CSF_REWARD_MODE
  echo ""
fi

if [[ "${IDM_TYPE}" == "re3" ]]; then
  echo "--- RE3 ---"
  print_config_value "RE3 Rep Dim" RE3_REP_DIM
  print_config_value "RE3 k-NN k" RE3_K
  print_config_value "RE3 Log Reward" RE3_LOG_REWARD
  echo ""
fi

if [[ "${IDM_TYPE}" == "ngu" ]]; then
  echo "--- NGU ---"
  print_config_value "NGU Rep Dim" NGU_REP_DIM
  print_config_value "NGU k-NN k" NGU_K
  print_config_value "NGU Use RND" NGU_USE_RND
  print_config_value "NGU Vectorized Scan" NGU_VECTORIZED_SCAN
  print_config_value "NGU Lifelong Clip" NGU_LIFELONG_CLIP
  echo ""
fi

SHOW_REPR_MODULES="false"
case "${RESOLVED_EXPL_REPR_MODE}" in
  shared_head)
    [[ -n "${EXPL_HEAD_TYPE:-}" || -n "${EXPL_HEAD_HIDDEN_DIM:-}" || -n "${EXPL_HEAD_LAYERS:-}" ]] && SHOW_REPR_MODULES="true"
    ;;
  branch)
    [[ -n "${EXPL_OBS_ENCODER_TYPE:-}" || -n "${EXPL_OBS_HIDDEN_DIM:-}" || -n "${EXPL_OBS_LAYERS:-}" || -n "${EXPL_BRIDGE_HIDDEN_DIM:-}" || -n "${EXPL_BRIDGE_LAYERS:-}" || -n "${EXPL_BRIDGE_LOSS_WEIGHT:-}" ]] && SHOW_REPR_MODULES="true"
    ;;
  parallel)
    [[ -n "${PARALLEL_REPLAY_INPUT_MODE:-}" || -n "${EXPL_RNN_HIDDEN_DIM:-}" || -n "${EXPL_OBS_CNN_DEPTH:-}" || -n "${EXPL_OBS_CNN_KERNEL_SIZE:-}" || -n "${EXPL_OBS_CNN_MINRES:-}" || -n "${EXPL_BRIDGE_HIDDEN_DIM:-}" || -n "${EXPL_BRIDGE_LAYERS:-}" || -n "${EXPL_BRIDGE_LOSS_WEIGHT:-}" ]] && SHOW_REPR_MODULES="true"
    ;;
esac

if [[ "${SHOW_REPR_MODULES}" == "true" ]]; then
  echo "--- Representation Modules ---"
  if [[ "${RESOLVED_EXPL_REPR_MODE}" == "shared_head" ]]; then
    [[ -n "${EXPL_HEAD_TYPE:-}" ]] && echo "  Shared Head Type: ${EXPL_HEAD_TYPE}"
  fi
  if [[ "${RESOLVED_EXPL_REPR_MODE}" == "branch" ]]; then
    echo "  Imag Entry: ${RESOLVED_EXPL_IMAG_REPR_MODE}"
    [[ -n "${EXPL_OBS_ENCODER_TYPE:-}" ]] && echo "  Obs Encoder Override: ${EXPL_OBS_ENCODER_TYPE}"
  fi
  if [[ "${RESOLVED_EXPL_REPR_MODE}" == "parallel" ]]; then
    echo "  Imag Entry: ${RESOLVED_EXPL_IMAG_REPR_MODE}"
    [[ -n "${PARALLEL_REPLAY_INPUT_MODE:-}" ]] && echo "  Parallel Replay Input: ${PARALLEL_REPLAY_INPUT_MODE} (runtime auto-select)"
    [[ -n "${EXPL_RNN_HIDDEN_DIM:-}" ]] && echo "  Replay GRU Hidden Dim: ${EXPL_RNN_HIDDEN_DIM}"
  fi
  [[ -n "${EXPL_BRIDGE_HIDDEN_DIM:-}" ]] && echo "  Bridge Hidden Dim: ${EXPL_BRIDGE_HIDDEN_DIM}"
  [[ -n "${EXPL_BRIDGE_LAYERS:-}" ]] && echo "  Bridge Layers: ${EXPL_BRIDGE_LAYERS}"
  [[ -n "${EXPL_BRIDGE_LOSS_WEIGHT:-}" ]] && echo "  Bridge Loss Weight: ${EXPL_BRIDGE_LOSS_WEIGHT}"
  echo ""
fi

if [[ "${IDM_TYPE}" == "ctec" ]]; then
  echo "--- C-TeC ---"
  print_config_value "C-TeC Rep Dim" CTEC_REP_DIM
  print_config_value "C-TeC Hidden Dim" CTEC_HIDDEN_DIM
  print_config_value "C-TeC Gamma" CTEC_GAMMA
  print_config_value "C-TeC Similarity" CTEC_SIMILARITY
  echo ""
fi

if [[ "${IDM_TYPE}" == "cip" ]]; then
  echo "--- CIP ---"
  print_config_value "CIP State Source" CIP_STATE_SOURCE
  print_config_value "CIP Imag Last" CIP_IMAG_LAST
  print_config_value "CIP Probes" CIP_PROBES
  print_config_value "CIP Scale" CIP_SCALE
  print_config_value "CIP Positive Only" CIP_POSITIVE
  print_config_value "CIP Normalize" CIP_NORMALIZE
  print_config_value "CIP Replay Metric" CIP_REPLAY_METRIC
  echo ""
fi

if [[ "${IDM_TYPE}" == "proto_rl" ]]; then
  echo "--- Proto-RL ---"
  print_config_value "Proto-RL Rep Dim" PROTO_RL_REP_DIM
  print_config_value "Proto-RL Num Prototypes" PROTO_RL_NUM_PROTOTYPES
  print_config_value "Proto-RL Queue Depth" PROTO_RL_CANDIDATES_PER_PROTOTYPE
  print_config_value "Proto-RL Temperature" PROTO_RL_TEMPERATURE
  print_config_value "Proto-RL k-NN k" PROTO_RL_KNN_K
  echo ""
fi

echo "--- Output ---"
echo "  Log Directory: ${LOG_DIR}"
echo "  Run Name: ${RUN_NAME}"
echo "  Storage Mode: ${RUN_STORAGE_MODE_RESOLVED}"
if [[ -n "${RUN_FORK_SOURCE_RUN_UID}" ]]; then
  echo "  Fork Parent: ${RUN_FORK_SOURCE_RUN_UID}"
  echo "  Fork Source: ${RUN_FORK_SOURCE_DIR}"
fi
echo ""
echo "================================================================================"
echo ""

if command -v nvidia-smi &> /dev/null; then
  echo "--- GPU Information ---"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || true
  echo ""
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  printf '%q ' "${CMD[@]}"
  echo
  # _run_common.sh is normally sourced by wrapper scripts, so the dry-run path
  # must return cleanly when sourced and only exit when executed directly.
  return 0 2>/dev/null || exit 0
fi

slurm_run_if_preempt "${CMD[@]}"

echo ""
echo "================================================================================"
echo "  Training completed successfully at $(date)"
echo "================================================================================"
