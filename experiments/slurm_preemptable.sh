#!/usr/bin/env bash

# Forward Slurm warning signals to training and publish verified recovery before
# requeue.
# Slurm handles preemption; wall-time warnings require one explicit requeue.

SLURM_PREEMPTED=0
_SLURM_CHILD_PID=""
_SLURM_SIGNAL_NAME=""
_SLURM_SIGNAL_REASON=""

_slurm_lowercase() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

slurm_preemptable_enabled_for_partition() {
  local partition="$1"
  local mode
  mode="$(_slurm_lowercase "${SLURM_PREEMPTABLE:-auto}")"
  case "${mode}" in
    1|true|yes|on) return 0 ;;
    0|false|no|off) return 1 ;;
    auto)
      [[ "${partition}" == *preempt* ]]
      ;;
    *)
      echo "Invalid SLURM_PREEMPTABLE=${SLURM_PREEMPTABLE}; use auto, true, or false." >&2
      return 2
      ;;
  esac
}

slurm_preemptable_enabled() {
  slurm_preemptable_enabled_for_partition "${SLURM_JOB_PARTITION:-}"
}

_slurm_usr1_warning_seconds() {
  local value="${SLURM_SIGNAL_WARNING_SECONDS:-}"
  local signal_spec="${SLURM_SIGNAL:-}"

  if [[ -z "${value}" && "${signal_spec}" =~ @([0-9]+)$ ]]; then
    value="${BASH_REMATCH[1]}"
  fi
  [[ "${value}" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "${value}"
}

_slurm_usr1_is_walltime_warning() {
  local end_time="${SLURM_JOB_END_TIME:-}"
  local warning_seconds
  local now
  local remaining
  local tolerance="${SLURM_SIGNAL_EARLY_TOLERANCE_SECONDS:-60}"

  [[ "${end_time}" =~ ^[0-9]+$ && "${end_time}" -gt 0 ]] || return 1
  [[ "${tolerance}" =~ ^[0-9]+$ ]] || return 1
  warning_seconds="$(_slurm_usr1_warning_seconds)" || return 1
  now="$(date +%s)" || return 1
  remaining=$((end_time - now))

  # Slurm documents that the end-time warning may arrive up to 60 seconds
  # earlier than requested. A USR1 outside that bounded window is the
  # send_user_signal notification for controller-owned preemption.
  [[ "${remaining}" -ge 0 &&
     "${remaining}" -le $((warning_seconds + tolerance)) ]]
}

_slurm_requeue_target() {
  local job_id="${SLURM_JOB_ID:-}"
  local array_job_id="${SLURM_ARRAY_JOB_ID:-}"
  local array_task_id="${SLURM_ARRAY_TASK_ID:-}"

  if [[ -n "${array_job_id}" || -n "${array_task_id}" ]]; then
    if [[ "${array_job_id}" =~ ^[0-9]+$ &&
          "${array_task_id}" =~ ^[0-9]+$ ]]; then
      printf '%s_%s\n' "${array_job_id}" "${array_task_id}"
      return 0
    fi
    echo "[Slurm] Refusing an unsafe or incomplete array requeue target." >&2
    return 1
  fi
  if [[ "${job_id}" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "${job_id}"
    return 0
  fi
  echo "[Slurm] Refusing an unsafe wall-time requeue target: ${job_id}" >&2
  return 1
}

_slurm_request_walltime_requeue() {
  local target
  target="$(_slurm_requeue_target)" || return 1
  if ! command -v scontrol >/dev/null 2>&1; then
    echo "[Slurm] scontrol is unavailable; wall-time continuation could not be requested." >&2
    return 127
  fi
  echo "[Slurm] Requesting wall-time continuation for ${target}."
  scontrol requeue "${target}"
}

_slurm_forward_signal() {
  local signal_name="$1"
  if [[ -n "${_SLURM_CHILD_PID}" ]] && kill -0 "${_SLURM_CHILD_PID}" 2>/dev/null; then
    echo "[Slurm] Forwarding SIG${signal_name} to training process ${_SLURM_CHILD_PID}."
    kill -s "${signal_name}" "${_SLURM_CHILD_PID}" 2>/dev/null || true
  fi
}

_slurm_on_usr1() {
  if [[ "${SLURM_PREEMPTED}" -eq 1 ]]; then
    if [[ "${_SLURM_SIGNAL_NAME}" == "USR1" ]]; then
      echo "[Slurm] Ignoring repeated SIGUSR1; preemption shutdown is already active."
    else
      echo "[Slurm] Ignoring SIGUSR1; cancellation shutdown is already active."
    fi
    return
  fi

  SLURM_PREEMPTED=1
  _SLURM_SIGNAL_NAME="USR1"
  if _slurm_usr1_is_walltime_warning; then
    _SLURM_SIGNAL_REASON="WALLTIME"
  else
    _SLURM_SIGNAL_REASON="PREEMPTION"
  fi
  # The EXIT trap must never turn any USR1 failure into an unbounded full-tree
  # publish during the warning/grace window.
  SCRATCH_SYNC_PREEMPT_NO_FINAL_PUBLISH=1
  export SCRATCH_SYNC_PREEMPT_NO_FINAL_PUBLISH
  echo ""
  echo "================================================================================"
  if [[ "${_SLURM_SIGNAL_REASON}" == "WALLTIME" ]]; then
    echo "[Slurm] Received the wall-time SIGUSR1 warning; requesting a checkpoint before explicit continuation."
  else
    echo "[Slurm] Received preemption SIGUSR1; requesting a checkpoint before native Slurm requeue."
  fi
  echo "================================================================================"
  echo ""
  _slurm_forward_signal USR1
}

_slurm_on_term() {
  if [[ "${SLURM_PREEMPTED}" -eq 1 ]]; then
    if [[ "${_SLURM_SIGNAL_NAME}" == "USR1" ]]; then
      # Preserve the first signal classification if TERM follows a preemption warning.
      if [[ "${_SLURM_SIGNAL_REASON}" == "WALLTIME" ]]; then
        echo "[Slurm] Received SIGTERM after SIGUSR1; wall-time continuation remains active."
      else
        echo "[Slurm] Received SIGTERM after SIGUSR1; native preemption shutdown remains active."
      fi
    else
      echo "[Slurm] Ignoring repeated SIGTERM; cancellation shutdown is already active."
    fi
    return
  fi

  # A TERM that was not preceded by USR1 is an explicit cancellation. Slurm
  # still owns all requeue decisions; this path only checkpoints and cleans up.
  SLURM_PREEMPTED=1
  _SLURM_SIGNAL_NAME="TERM"
  echo ""
  echo "================================================================================"
  echo "[Slurm] Received SIGTERM; requesting a checkpoint before cancellation."
  echo "================================================================================"
  echo ""
  _slurm_forward_signal TERM
}

_slurm_sync_scratch() {
  local child_status="${1:-0}"
  local event_status=0
  local status=0

  # Use bounded recovery during the preemption grace period; a full copy may exceed
  # KillWait.
  if [[ "${_SLURM_SIGNAL_NAME:-}" == "USR1" ]]; then
    if ! declare -F scratch_sync_prepare_preempt_requeue >/dev/null 2>&1; then
      echo "[Slurm] No bounded recovery helper is available; native Slurm remains the requeue authority."
      return 0
    fi
    echo "[Slurm] Preparing certified durable recovery for native Slurm requeue."
    if scratch_sync_prepare_preempt_requeue "${child_status}"; then
      return 0
    else
      status=$?
    fi
    echo "[Slurm] Bounded recovery preparation failed (exit ${status}); native Slurm remains the requeue authority." >&2
    return "${status}"
  fi

  if ! declare -F scratch_sync_move_logs >/dev/null 2>&1; then
    return 0
  fi

  # Record terminal state before publication and any requeue request.
  if declare -F scratch_sync_record_finish >/dev/null 2>&1; then
    scratch_sync_record_finish "${child_status}" || event_status=$?
  fi

  echo "[Slurm] Syncing complete run directory to durable storage."
  # Retry publication before choosing the supervisor exit status.
  local -a publish_command=(scratch_sync_move_logs)
  if declare -F scratch_sync_retry_publication >/dev/null 2>&1; then
    publish_command=(scratch_sync_retry_publication scratch_sync_move_logs)
  fi
  if "${publish_command[@]}"; then
    if declare -F scratch_sync_copy_scheduler_logs_to >/dev/null 2>&1; then
      scratch_sync_copy_scheduler_logs_to \
        "${WORK_LOG_DIR:-${LOG_DIR:-}}" || status=$?
    fi
    if [[ "${event_status}" -ne 0 ]]; then
      return "${event_status}"
    fi
    return "${status}"
  else
    status=$?
  fi

  echo "[Slurm] Full scratch sync failed (exit ${status}); scratch data was preserved." >&2
  return "${status}"
}

slurm_run_preemptable() {
  local exit_code=0
  local sync_status=0

  SLURM_PREEMPTED=0
  _SLURM_SIGNAL_NAME=""
  _SLURM_SIGNAL_REASON=""
  SCRATCH_SYNC_PREEMPT_NO_FINAL_PUBLISH=0
  export SCRATCH_SYNC_PREEMPT_NO_FINAL_PUBLISH

  trap _slurm_on_usr1 USR1
  trap _slurm_on_term TERM

  "$@" &
  _SLURM_CHILD_PID="$!"
  local restore_errexit=0
  if [[ $- == *e* ]]; then
    restore_errexit=1
  fi

  while true; do
    # Resume interrupted waits without letting set -e terminate checkpoint shutdown.
    set +e
    wait "${_SLURM_CHILD_PID}"
    exit_code=$?
    if [[ "${restore_errexit}" == "1" ]]; then
      set -e
    fi

    if [[ "${exit_code}" -ge 128 ]] && kill -0 "${_SLURM_CHILD_PID}" 2>/dev/null; then
      continue
    fi
    break
  done

  _SLURM_CHILD_PID=""

  # Keep the signal traps active while storage finalization runs. A TERM after
  # USR1 must leave the preemption classification and bounded path unchanged.
  if _slurm_sync_scratch "${exit_code}"; then
    sync_status=0
  else
    sync_status=$?
  fi

  if [[ "${sync_status}" -eq 0 &&
        "${_SLURM_SIGNAL_REASON}" == "WALLTIME" ]]; then
    if _slurm_request_walltime_requeue; then
      :
    else
      sync_status=$?
      echo "[Slurm] Wall-time requeue request failed (exit ${sync_status})." >&2
    fi
  fi

  trap - USR1 TERM

  if [[ "${sync_status}" -ne 0 ]]; then
    return "${sync_status}"
  fi

  if [[ "${SLURM_PREEMPTED}" -eq 1 ]]; then
    if [[ "${_SLURM_SIGNAL_NAME}" == "USR1" ]]; then
      if [[ "${_SLURM_SIGNAL_REASON}" == "WALLTIME" ]]; then
        echo "[Slurm] Exiting after SIGUSR1; durable recovery and wall-time requeue are complete."
      else
        echo "[Slurm] Exiting after SIGUSR1; checkpoint shutdown is complete and native Slurm owns requeue."
      fi
      exit 0
    fi
    echo "[Slurm] Exiting after SIG${_SLURM_SIGNAL_NAME}; checkpoint shutdown is complete."
    # Preserve signal exit codes so canceled or truncated runs are not reported as
    # complete.
    case "${_SLURM_SIGNAL_NAME}" in
      TERM) exit 143 ;;
      USR1) exit 138 ;;
      *) exit 1 ;;
    esac
  fi

  return "${exit_code}"
}

slurm_run_if_preempt() {
  local enabled_status=0
  if slurm_preemptable_enabled; then
    slurm_run_preemptable "$@"
    return $?
  else
    enabled_status=$?
  fi
  if [[ "${enabled_status}" -eq 2 ]]; then
    return 2
  fi
  # Even jobs outside preemptible partitions use the same signal supervisor:
  # Slurm owns requeue policy, while TERM-only cancellation still gets cleanup.
  slurm_run_preemptable "$@"
}
