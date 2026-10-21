#!/usr/bin/env bash

# Publish scratch runs to WORK_LOG_ROOT through an atomic staging directory.
# A durable flock prevents concurrent writers.

# Resolve the helper while sourcing, before a background publisher or EXIT trap
# changes directory. BASH_SOURCE can be relative when this library is sourced.
SCRATCH_SYNC_HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" || return 1

scratch_sync_enter_storage_cwd() {
    # Use a stable local working directory for publication if the submission mount
    # becomes unavailable.
    builtin cd -- /
}

scratch_sync_dir_has_entries() {
    local dir="$1"
    local path

    [[ -d "${dir}" ]] || return 1
    for path in "${dir}"/* "${dir}"/.[!.]* "${dir}"/..?*; do
        if [[ -e "${path}" || -L "${path}" ]]; then
            return 0
        fi
    done
    return 1
}

scratch_sync_require_commands() {
    local command_name
    for command_name in rsync flock realpath python3; do
        if ! command -v "${command_name}" >/dev/null 2>&1; then
            echo "Error: ${command_name} is required for safe run storage." >&2
            return 1
        fi
    done
}

scratch_sync_canonical_path() {
    realpath -m -- "$1"
}

scratch_sync_lexical_path() {
    # Normalize paths without resolving links so symlink checks can inspect every
    # component.
    python3 -c 'import os, sys; print(os.path.abspath(os.path.normpath(sys.argv[1])))' "$1"
}

scratch_sync_is_strict_child() {
    local child="$1"
    local parent="$2"
    [[ "${child}" == "${parent}/"* && "${child}" != "${parent}" ]]
}

scratch_sync_reject_symlink_components() {
    local root="$1"
    local target="$2"
    python3 - "${root}" "${target}" <<'PY'
import os
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
try:
    relative = target.relative_to(root)
except ValueError:
    raise SystemExit(2)
if not relative.parts:
    raise SystemExit(2)

cursor = root
for part in relative.parts:
    cursor = cursor / part
    try:
        metadata = os.lstat(cursor)
    except FileNotFoundError:
        # A missing ancestor means no deeper component can exist yet.
        break
    if stat.S_ISLNK(metadata.st_mode):
        print(f"Error: storage path contains a symlink below its approved root: {cursor}", file=sys.stderr)
        raise SystemExit(1)
PY
}

scratch_sync_roots_overlap() {
    local first="$1"
    local second="$2"
    [[ "${first}" == "${second}" ]] ||
        scratch_sync_is_strict_child "${first}" "${second}" ||
        scratch_sync_is_strict_child "${second}" "${first}"
}

scratch_sync_validate_run_suffix() {
    local suffix="$1"
    local expected

    if [[ -n "${EXPERIMENT_NAME:-}" && -n "${RUN_NAME:-}" ]]; then
        expected="${EXPERIMENT_NAME}/${RUN_NAME}"
        if [[ "${suffix}" != "${expected}" ]]; then
            echo "Error: storage suffix does not match EXPERIMENT_NAME/RUN_NAME." >&2
            echo "  suffix=${suffix}" >&2
            echo "  expected=${expected}" >&2
            return 1
        fi
    elif [[ -n "${RUN_NAME:-}" && "$(basename "${suffix}")" != "${RUN_NAME}" ]]; then
        echo "Error: storage run basename does not match RUN_NAME." >&2
        echo "  suffix=${suffix}" >&2
        echo "  RUN_NAME=${RUN_NAME}" >&2
        return 1
    elif [[ -n "${EXPERIMENT_NAME:-}" && "${suffix%%/*}" != "${EXPERIMENT_NAME}" ]]; then
        echo "Error: storage path does not begin with EXPERIMENT_NAME." >&2
        echo "  suffix=${suffix}" >&2
        echo "  EXPERIMENT_NAME=${EXPERIMENT_NAME}" >&2
        return 1
    fi
}

scratch_sync_configure_paths() {
    source "$(dirname "${BASH_SOURCE[0]}")/_runtime_defaults.sh"

    local canonical_log
    local canonical_scratch
    local canonical_work
    local lexical_log
    local lexical_scratch
    local lexical_work
    local lexical_work_dir
    local suffix

    lexical_log="$(scratch_sync_lexical_path "${LOG_DIR}")" || return 1
    lexical_scratch="$(scratch_sync_lexical_path "${SCRATCH_LOG_ROOT}")" || return 1
    lexical_work="$(scratch_sync_lexical_path "${WORK_LOG_ROOT}")" || return 1

    if [[ "${lexical_scratch}" == "/" || "${lexical_work}" == "/" ]]; then
        echo "Error: storage roots may not be /." >&2
        return 1
    fi
    if scratch_sync_roots_overlap "${lexical_scratch}" "${lexical_work}"; then
        echo "Error: scratch and durable roots may not overlap." >&2
        return 1
    fi

    if scratch_sync_is_strict_child "${lexical_log}" "${lexical_scratch}"; then
        suffix="${lexical_log#${lexical_scratch}/}"
        lexical_work_dir="${lexical_work}/${suffix}"
        scratch_sync_reject_symlink_components \
            "${lexical_scratch}" "${lexical_log}" || return 1
        scratch_sync_reject_symlink_components \
            "${lexical_work}" "${lexical_work_dir}" || return 1
        SCRATCH_SYNC_DIRECT_DURABLE=0
    elif scratch_sync_is_strict_child "${lexical_log}" "${lexical_work}"; then
        suffix="${lexical_log#${lexical_work}/}"
        lexical_work_dir="${lexical_log}"
        scratch_sync_reject_symlink_components \
            "${lexical_work}" "${lexical_log}" || return 1
        SCRATCH_SYNC_DIRECT_DURABLE=1
    else
        cat >&2 <<EOF
Error: LOG_DIR is outside both approved storage roots.
  LOG_DIR=${LOG_DIR}
  SCRATCH_LOG_ROOT=${SCRATCH_LOG_ROOT}
  WORK_LOG_ROOT=${WORK_LOG_ROOT}

Set SCRATCH_LOG_ROOT and WORK_LOG_ROOT explicitly for a custom scratch tree,
or place EXPERIMENT_ROOT directly below WORK_LOG_ROOT.
EOF
        return 1
    fi

    scratch_sync_validate_run_suffix "${suffix}" || return 1

    canonical_log="$(scratch_sync_canonical_path "${lexical_log}")" || return 1
    canonical_scratch="$(scratch_sync_canonical_path "${lexical_scratch}")" || return 1
    canonical_work="$(scratch_sync_canonical_path "${lexical_work}")" || return 1
    if scratch_sync_roots_overlap "${canonical_scratch}" "${canonical_work}"; then
        echo "Error: scratch and durable roots resolve to overlapping paths." >&2
        return 1
    fi
    SCRATCH_LOG_ROOT="${canonical_scratch}"
    WORK_LOG_ROOT="${canonical_work}"
    LOG_DIR="${canonical_log}"
    if [[ "${SCRATCH_SYNC_DIRECT_DURABLE}" -eq 0 ]]; then
        if ! scratch_sync_is_strict_child "${LOG_DIR}" "${SCRATCH_LOG_ROOT}" ||
           [[ "${LOG_DIR#${SCRATCH_LOG_ROOT}/}" != "${suffix}" ]]; then
            echo "Error: scratch run path changed while resolving storage components." >&2
            return 1
        fi
        WORK_LOG_DIR="${WORK_LOG_ROOT}/${suffix}"
    else
        if ! scratch_sync_is_strict_child "${LOG_DIR}" "${WORK_LOG_ROOT}" ||
           [[ "${LOG_DIR#${WORK_LOG_ROOT}/}" != "${suffix}" ]]; then
            echo "Error: durable run path changed while resolving storage components." >&2
            return 1
        fi
        WORK_LOG_DIR="${LOG_DIR}"
    fi

    WORK_LOG_DIR="$(scratch_sync_canonical_path "${WORK_LOG_DIR}")" || return 1
    if ! scratch_sync_is_strict_child "${WORK_LOG_DIR}" "${WORK_LOG_ROOT}"; then
        echo "Error: resolved durable run path escaped WORK_LOG_ROOT: ${WORK_LOG_DIR}" >&2
        return 1
    fi

    local work_parent
    local work_base
    work_parent="$(dirname "${WORK_LOG_DIR}")"
    work_base="$(basename "${WORK_LOG_DIR}")"
    SCRATCH_SYNC_LOCK_PATH="${work_parent}/.${work_base}.run.lock"
    SCRATCH_SYNC_STAGE_DIR="${work_parent}/.${work_base}.publish-new"
    SCRATCH_SYNC_PREVIOUS_DIR="${work_parent}/.${work_base}.publish-previous"
    SCRATCH_SYNC_RECOVERY_DIR="${work_parent}/.${work_base}.recovery"
    SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR="${SCRATCH_SYNC_RECOVERY_DIR}/generations"
    SCRATCH_SYNC_RECOVERY_CURRENT="${SCRATCH_SYNC_RECOVERY_DIR}/current"
    SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_PATH="${SCRATCH_SYNC_RECOVERY_DIR}/checkpoint-high-water.json"
    SCRATCH_SYNC_RECOVERY_INDEX_STAGE="${work_parent}/.${work_base}.recovery-index-new"
    SCRATCH_SYNC_TOMBSTONE="$(dirname "${LOG_DIR}")/.${work_base}.cleanup"
    : "${SCRATCH_SYNC_MAX_RESUME_GAP_STEPS:=${SAVE_EVERY:-10000}}"
    if [[ ! "${SCRATCH_SYNC_MAX_RESUME_GAP_STEPS}" =~ ^[0-9]+$ ]]; then
        echo "Error: SCRATCH_SYNC_MAX_RESUME_GAP_STEPS must be a nonnegative integer." >&2
        return 1
    fi
    if [[ "${SCRATCH_SYNC_DIRECT_DURABLE}" -eq 0 ]]; then
        case "${SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_MODE:-synchronous}" in
            synchronous)
                SCRATCH_SYNC_CHECKPOINT_HIGH_WATER="${SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_PATH}"
                ;;
            published)
                # Advance durable progress only after publishing the checkpoint and
                # replay together.
                unset SCRATCH_SYNC_CHECKPOINT_HIGH_WATER
                ;;
            *)
                echo "Error: SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_MODE must be synchronous or published." >&2
                return 1
                ;;
        esac
        export SCRATCH_SYNC_MAX_RESUME_GAP_STEPS
        if [[ -n "${SCRATCH_SYNC_CHECKPOINT_HIGH_WATER:-}" ]]; then
            export SCRATCH_SYNC_CHECKPOINT_HIGH_WATER
        fi
    else
        unset SCRATCH_SYNC_CHECKPOINT_HIGH_WATER
        export SCRATCH_SYNC_MAX_RESUME_GAP_STEPS
    fi
}

scratch_sync_ensure_runtime_paths() {
    if [[ -n "${SCRATCH_SYNC_STAGE_DIR:-}" ]]; then
        return 0
    fi
    if [[ -z "${LOG_DIR:-}" || -z "${WORK_LOG_DIR:-}" ]]; then
        echo "Error: LOG_DIR and WORK_LOG_DIR are required for storage sync." >&2
        return 1
    fi

    LOG_DIR="$(scratch_sync_canonical_path "${LOG_DIR}")" || return 1
    WORK_LOG_DIR="$(scratch_sync_canonical_path "${WORK_LOG_DIR}")" || return 1
    if [[ "${LOG_DIR}" == "${WORK_LOG_DIR}" ]]; then
        SCRATCH_SYNC_DIRECT_DURABLE=1
    else
        SCRATCH_SYNC_DIRECT_DURABLE=0
    fi
    local work_parent="$(dirname "${WORK_LOG_DIR}")"
    local work_base="$(basename "${WORK_LOG_DIR}")"
    SCRATCH_SYNC_LOCK_PATH="${work_parent}/.${work_base}.run.lock"
    SCRATCH_SYNC_STAGE_DIR="${work_parent}/.${work_base}.publish-new"
    SCRATCH_SYNC_PREVIOUS_DIR="${work_parent}/.${work_base}.publish-previous"
    SCRATCH_SYNC_RECOVERY_DIR="${work_parent}/.${work_base}.recovery"
    SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR="${SCRATCH_SYNC_RECOVERY_DIR}/generations"
    SCRATCH_SYNC_RECOVERY_CURRENT="${SCRATCH_SYNC_RECOVERY_DIR}/current"
    SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_PATH="${SCRATCH_SYNC_RECOVERY_DIR}/checkpoint-high-water.json"
    SCRATCH_SYNC_RECOVERY_INDEX_STAGE="${work_parent}/.${work_base}.recovery-index-new"
    SCRATCH_SYNC_TOMBSTONE="$(dirname "${LOG_DIR}")/.${work_base}.cleanup"
    : "${WORK_LOG_ROOT:=$(dirname "${WORK_LOG_DIR}")}"
    : "${SCRATCH_SYNC_MAX_RESUME_GAP_STEPS:=${SAVE_EVERY:-10000}}"
    if [[ ! "${SCRATCH_SYNC_MAX_RESUME_GAP_STEPS}" =~ ^[0-9]+$ ]]; then
        echo "Error: SCRATCH_SYNC_MAX_RESUME_GAP_STEPS must be a nonnegative integer." >&2
        return 1
    fi
    if [[ "${SCRATCH_SYNC_DIRECT_DURABLE}" -eq 0 ]]; then
        case "${SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_MODE:-synchronous}" in
            synchronous)
                SCRATCH_SYNC_CHECKPOINT_HIGH_WATER="${SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_PATH}"
                ;;
            published)
                unset SCRATCH_SYNC_CHECKPOINT_HIGH_WATER
                ;;
            *)
                echo "Error: SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_MODE must be synchronous or published." >&2
                return 1
                ;;
        esac
        export SCRATCH_SYNC_MAX_RESUME_GAP_STEPS
        if [[ -n "${SCRATCH_SYNC_CHECKPOINT_HIGH_WATER:-}" ]]; then
            export SCRATCH_SYNC_CHECKPOINT_HIGH_WATER
        fi
    else
        unset SCRATCH_SYNC_CHECKPOINT_HIGH_WATER
        export SCRATCH_SYNC_MAX_RESUME_GAP_STEPS
    fi
}

scratch_sync_acquire_lock() {
    mkdir -p "$(dirname "${SCRATCH_SYNC_LOCK_PATH}")" || return 1
    if [[ -e "${SCRATCH_SYNC_LOCK_PATH}" || -L "${SCRATCH_SYNC_LOCK_PATH}" ]]; then
        if [[ ! -f "${SCRATCH_SYNC_LOCK_PATH}" || -L "${SCRATCH_SYNC_LOCK_PATH}" ]]; then
            echo "Error: run storage lock path is not a regular file: ${SCRATCH_SYNC_LOCK_PATH}" >&2
            return 1
        fi
    fi
    # Open without O_TRUNC to prevent accidental truncation if the lock path changes.
    exec {SCRATCH_SYNC_LOCK_FD}<>"${SCRATCH_SYNC_LOCK_PATH}" || return 1
    if [[ ! -f "${SCRATCH_SYNC_LOCK_PATH}" || -L "${SCRATCH_SYNC_LOCK_PATH}" ]]; then
        echo "Error: run storage lock changed while opening: ${SCRATCH_SYNC_LOCK_PATH}" >&2
        eval "exec ${SCRATCH_SYNC_LOCK_FD}>&-"
        unset SCRATCH_SYNC_LOCK_FD
        return 1
    fi
    if ! flock -n "${SCRATCH_SYNC_LOCK_FD}"; then
        echo "Error: another process owns run storage lock ${SCRATCH_SYNC_LOCK_PATH}." >&2
        eval "exec ${SCRATCH_SYNC_LOCK_FD}>&-"
        unset SCRATCH_SYNC_LOCK_FD
        return 1
    fi
}

scratch_sync_release_lock() {
    if [[ -n "${SCRATCH_SYNC_LOCK_FD:-}" ]]; then
        flock -u "${SCRATCH_SYNC_LOCK_FD}" 2>/dev/null || true
        eval "exec ${SCRATCH_SYNC_LOCK_FD}>&-"
        unset SCRATCH_SYNC_LOCK_FD
    fi
}

scratch_sync_copy_dir() {
    local src="$1"
    local dst="$2"

    mkdir -p "${dst}" || return 1
    rsync -a --delete --no-owner --no-group \
        --chmod=Dug+rwx,Do-rwx,Fug+rw,Fo-rwx \
        "${src}/" "${dst}/"
}

scratch_sync_copy_file() {
    local src="$1"
    local dst="$2"

    mkdir -p "$(dirname "${dst}")" || return 1
    cp -a -- "${src}" "${dst}"
}

scratch_sync_fsync_tree() {
    local root="$1"
    python3 - "${root}" <<'PY'
import os
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1])
metadata = root.lstat()
if not stat.S_ISDIR(metadata.st_mode) or root.is_symlink():
    raise SystemExit("fsync root is not a real directory")

directories: list[pathlib.Path] = []
for walk_root, names, files in os.walk(root, topdown=True, followlinks=False):
    directory = pathlib.Path(walk_root)
    directories.append(directory)
    for name in names:
        path = directory / name
        child = path.lstat()
        if stat.S_ISLNK(child.st_mode):
            # Internal relative links are intentional catalog artifacts. Their
            # directory entry is persisted by fsyncing the parent directory.
            continue
        if not stat.S_ISDIR(child.st_mode):
            raise SystemExit(f"unsafe non-directory in directory list: {path}")
    for name in files:
        path = directory / name
        child = path.lstat()
        if stat.S_ISLNK(child.st_mode):
            continue
        if not stat.S_ISREG(child.st_mode):
            raise SystemExit(f"unsafe staged artifact for fsync: {path}")
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

for directory in reversed(directories):
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
PY
}

scratch_sync_fsync_directory() {
    local directory="$1"
    python3 - "${directory}" <<'PY'
import os
import pathlib
import stat
import sys

path = pathlib.Path(sys.argv[1])
metadata = path.lstat()
if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
    raise SystemExit("fsync target is not a real directory")
descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
}

scratch_sync_checkpoint_name_is_safe() {
    local name="$1"
    [[ "${name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] &&
        [[ "${name}" != "." && "${name}" != ".." ]]
}

scratch_sync_checkpoint_name_is_newer() {
    local candidate="$1"
    local current="$2"

    if [[ "${candidate}" =~ ^[0-9]+$ && "${current}" =~ ^[0-9]+$ ]]; then
        python3 -c '
import sys
raise SystemExit(0 if int(sys.argv[1]) > int(sys.argv[2]) else 1)
' "${candidate}" "${current}"
    else
        [[ "${candidate}" > "${current}" ]]
    fi
}

scratch_sync_checkpoint_step() {
    local run_dir="$1"
    local checkpoint="$2"

    scratch_sync_checkpoint_name_is_safe "${checkpoint}" || return 1
    python3 - "${run_dir}/checkpoint.pkl/${checkpoint}/step.pkl" \
        "${checkpoint}" <<'PY'
import os
import pickletools
import stat
import sys

path, checkpoint = sys.argv[1:]
try:
    metadata = os.lstat(path)
except FileNotFoundError:
    # A numeric generation name provides the step when step.pkl is absent.
    if checkpoint.isascii() and checkpoint.isdecimal():
        print(int(checkpoint))
        raise SystemExit(0)
    raise SystemExit(1)
if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 1024:
    raise SystemExit(1)
data = open(path, "rb").read()

# Accept scalar steps encoded as decimal text.
try:
    text = data.decode("ascii").strip()
except UnicodeDecodeError:
    text = ""
if text and text.isdecimal():
    print(int(text))
    raise SystemExit(0)

allowed = {
    "PROTO", "FRAME", "BININT", "BININT1", "BININT2", "INT", "LONG",
    "LONG1", "LONG4", "STOP",
}
try:
    operations = list(pickletools.genops(data))
except ValueError:
    raise SystemExit(1)
if not operations or operations[-1][0].name != "STOP":
    raise SystemExit(1)
integer = None
for operation, argument, _ in operations:
    if operation.name not in allowed:
        raise SystemExit(1)
    if operation.name in {
        "BININT", "BININT1", "BININT2", "INT", "LONG", "LONG1", "LONG4",
    }:
        if integer is not None or not isinstance(argument, int) or isinstance(argument, bool):
            raise SystemExit(1)
        integer = argument
if integer is None or integer < 0:
    raise SystemExit(1)
print(integer)
PY
}

scratch_sync_checkpoint_step_is_newer() {
    local candidate="$1"
    local current="$2"

    python3 -c '
import sys
raise SystemExit(0 if int(sys.argv[1]) > int(sys.argv[2]) else 1)
' "${candidate}" "${current}"
}

scratch_sync_checkpoint_point_is_newer() {
    local candidate_step="$1"
    local candidate_checkpoint="$2"
    local current_step="$3"
    local current_checkpoint="$4"

    python3 -c '
import sys
import re

def generation_key(value):
    match = re.search(r"-u([0-9]{12})$", value)
    if match:
        return (1, int(match.group(1)), 1, value)
    if value.isdecimal():
        return (0, 0, 0, int(value))
    return (0, 0, 1, value)

candidate = (int(sys.argv[1]), generation_key(sys.argv[2]))
current = (int(sys.argv[3]), generation_key(sys.argv[4]))
raise SystemExit(0 if candidate > current else 1)
' "${candidate_step}" "${candidate_checkpoint}" \
  "${current_step}" "${current_checkpoint}"
}

scratch_sync_checkpoint_high_water_command() {
    local helper="${RUN_STORAGE_HELPER:-${HELPER_DIR:-${SCRATCH_SYNC_HELPER_DIR}}/run_storage.py}"
    if [[ ! -f "${helper}" || -L "${helper}" ]]; then
        echo "Error: missing or unsafe checkpoint high-water helper ${helper}." >&2
        return 1
    fi
    python3 "${helper}" checkpoint-high-water "$@"
}

scratch_sync_checkpoint_high_water_check() {
    local run_dir="$1"
    local checkpoint="$2"
    local minimum="${3:-floor}"
    local marker="${4:-${SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_PATH:-}}"
    local expected_run_id="${5:-${RUN_UID:-}}"

    [[ -n "${marker}" ]] || return 0
    if [[ ! -e "${marker}" && ! -L "${marker}" ]]; then
        return 0
    fi
    local -a command=(check --marker "${marker}" --run-dir "${run_dir}"
        --checkpoint "${checkpoint}" --minimum "${minimum}")
    [[ -n "${expected_run_id}" ]] &&
        command+=(--expected-run-id "${expected_run_id}")
    if ! scratch_sync_checkpoint_high_water_command "${command[@]}" >/dev/null; then
        echo "Error: checkpoint ${checkpoint} in ${run_dir} failed the durable progress fence." >&2
        return 1
    fi
}

scratch_sync_checkpoint_high_water_seed() {
    local run_dir="$1"
    local checkpoint="$2"
    local marker="${3:-${SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_PATH:-}}"

    [[ -n "${marker}" ]] || return 0
    if [[ -e "${marker}" || -L "${marker}" ]]; then
        scratch_sync_checkpoint_high_water_check \
            "${run_dir}" "${checkpoint}" floor "${marker}"
        return $?
    fi
    if ! scratch_sync_checkpoint_high_water_command seed \
        --marker "${marker}" --run-dir "${run_dir}" \
        --checkpoint "${checkpoint}" \
        --max-gap-steps "${SCRATCH_SYNC_MAX_RESUME_GAP_STEPS:-10000}" \
        >/dev/null; then
        echo "Error: could not seed the durable checkpoint progress fence." >&2
        return 1
    fi
}

scratch_sync_repair_first_checkpoint_latest() {
    local run_dir="$1"
    local marker="${2:-${SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_PATH:-}}"

    [[ -n "${marker}" && ( -e "${marker}" || -L "${marker}" ) ]] || return 0
    [[ -d "${run_dir}/checkpoint.pkl" &&
       ! -L "${run_dir}/checkpoint.pkl" ]] || return 0
    local -a command=(repair-latest --marker "${marker}" --run-dir "${run_dir}")
    [[ -n "${RUN_UID:-}" ]] && command+=(--expected-run-id "${RUN_UID}")
    if ! scratch_sync_checkpoint_high_water_command \
        "${command[@]}" >/dev/null; then
        echo "Error: could not repair the first-checkpoint latest transaction." >&2
        return 1
    fi
}

scratch_sync_checkpoint_latest() {
    local dir="$1"
    local latest=""
    local candidate=""
    local path
    local name

    if [[ -f "${dir}/latest" && ! -L "${dir}/latest" ]]; then
        candidate="$(<"${dir}/latest")"
        candidate="${candidate//$'\n'/}"
        if scratch_sync_checkpoint_name_is_safe "${candidate}" &&
            [[ -d "${dir}/${candidate}" && ! -L "${dir}/${candidate}" ]] &&
            [[ -f "${dir}/${candidate}/done" && ! -L "${dir}/${candidate}/done" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    fi

    for path in "${dir}"/*; do
        [[ -d "${path}" && ! -L "${path}" ]] || continue
        [[ -f "${path}/done" && ! -L "${path}/done" ]] || continue
        name="$(basename "${path}")"
        scratch_sync_checkpoint_name_is_safe "${name}" || continue
        if [[ -z "${latest}" ]] ||
           scratch_sync_checkpoint_name_is_newer "${name}" "${latest}"; then
            latest="${name}"
        fi
    done

    if [[ -n "${latest}" ]]; then
        printf '%s\n' "${latest}"
        return 0
    fi
    return 1
}

scratch_sync_checkpoint_is_valid() {
    local dir="$1"
    local latest

    latest="$(scratch_sync_checkpoint_latest "${dir}")" || return 1
    [[ -f "${dir}/${latest}/done" && ! -L "${dir}/${latest}/done" ]]
}

scratch_sync_cleanup_checkpoint_dir() {
    local dir="$1"
    local keep_latest="$2"
    local path
    local name

    for path in "${dir}"/*; do
        [[ -d "${path}" && ! -L "${path}" ]] || continue
        name="$(basename "${path}")"
        [[ "${name}" == "${keep_latest}" ]] && continue
        rm -rf -- "${path}" || return 1
    done
}

scratch_sync_copy_checkpoint_dir() {
    local src="$1"
    local dst="$2"
    local requested="${3:-}"
    local latest
    local tmp_latest

    if [[ -n "${requested}" ]]; then
        if ! scratch_sync_checkpoint_name_is_safe "${requested}" ||
           [[ ! -d "${src}/${requested}" || -L "${src}/${requested}" ||
              ! -f "${src}/${requested}/done" || -L "${src}/${requested}/done" ]]; then
            echo "Invalid checkpoint: requested generation ${requested} is unavailable in ${src}." >&2
            return 1
        fi
        latest="${requested}"
    else
        latest="$(scratch_sync_checkpoint_latest "${src}")" || {
            echo "Invalid checkpoint: ${src} has no completed checkpoint generation." >&2
            return 1
        }
    fi

    mkdir -p "${dst}/${latest}" || return 1
    rm -f -- "${dst}/${latest}/done" || return 1
    rsync -a --delete --no-owner --no-group --exclude='/done' \
        --chmod=Dug+rwx,Do-rwx,Fug+rw,Fo-rwx \
        "${src}/${latest}/" "${dst}/${latest}/" || return 1

    # Publish the checkpoint's completion marker last inside the private stage.
    cp -p -- "${src}/${latest}/done" "${dst}/${latest}/.done.tmp.$$" || return 1
    mv -f -- "${dst}/${latest}/.done.tmp.$$" "${dst}/${latest}/done" || return 1

    tmp_latest="${dst}/.latest.tmp.$$"
    printf '%s\n' "${latest}" > "${tmp_latest}" || return 1
    mv -f -- "${tmp_latest}" "${dst}/latest" || return 1
    scratch_sync_cleanup_checkpoint_dir "${dst}" "${latest}" || return 1
}

scratch_sync_pin_checkpoint_generation() {
    local checkpoint_root="$1"
    local checkpoint="$2"
    local source="${checkpoint_root}/${checkpoint}"
    local pin_root="$(dirname "${checkpoint_root}")/.checkpoint-pins"
    local pin="${pin_root}/pin-${checkpoint}-${BASHPID}-${RANDOM}"

    scratch_sync_checkpoint_name_is_safe "${checkpoint}" || return 1
    [[ -d "${source}" && ! -L "${source}" &&
       -f "${source}/done" && ! -L "${source}/done" ]] || return 1
    python3 - "${source}" <<'PY' || return 1
import os
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1])
for walk_root, directories, filenames in os.walk(root, followlinks=False):
    directory = pathlib.Path(walk_root)
    for name in directories:
        metadata = (directory / name).lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise SystemExit(1)
    for name in filenames:
        metadata = (directory / name).lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise SystemExit(1)
PY

    mkdir -p "${pin_root}" || return 1
    [[ ! -e "${pin}" && ! -L "${pin}" ]] || return 1
    mkdir -p "${pin}" || return 1
    if ! cp -al -- "${source}" "${pin}/${checkpoint}" ||
       ! printf '%s\n' "${checkpoint}" > "${pin}/latest" ||
       [[ "$(scratch_sync_checkpoint_latest "${pin}" 2>/dev/null || true)" != "${checkpoint}" ]]; then
        rm -rf -- "${pin}" || true
        return 1
    fi
    printf '%s\n' "${pin}"
}

scratch_sync_release_checkpoint_pin() {
    local pin="$1"
    local pin_root="${LOG_DIR}/.checkpoint-pins"
    [[ -n "${pin}" && "${pin}" == "${pin_root}/"* &&
       -d "${pin}" && ! -L "${pin}" ]] || return 1
    rm -rf -- "${pin}" || return 1
    rmdir -- "${pin_root}" 2>/dev/null || true
}

scratch_sync_cleanup_stale_checkpoint_pins() {
    local pin_root="${LOG_DIR}/.checkpoint-pins"
    local path
    local name
    if [[ ! -e "${pin_root}" && ! -L "${pin_root}" ]]; then
        return 0
    fi
    [[ -d "${pin_root}" && ! -L "${pin_root}" ]] || {
        echo "Error: stale checkpoint pin root is unsafe: ${pin_root}" >&2
        return 1
    }
    for path in "${pin_root}"/* "${pin_root}"/.[!.]* "${pin_root}"/..?*; do
        [[ -e "${path}" || -L "${path}" ]] || continue
        name="$(basename "${path}")"
        if [[ ! "${name}" =~ ^pin-[A-Za-z0-9][A-Za-z0-9._-]*-[0-9]+-[0-9]+$ ||
              ! -d "${path}" || -L "${path}" ]]; then
            echo "Error: stale checkpoint pin is unsafe: ${path}" >&2
            return 1
        fi
        rm -rf -- "${path}" || return 1
    done
    rmdir -- "${pin_root}"
}

scratch_sync_checkpoint_replay_inventory() {
    local operation="$1"
    local source_root="$2"
    local checkpoint_root="$3"
    local destination_root="${4:-}"

    python3 - "${operation}" "${source_root}" "${checkpoint_root}" \
        "${destination_root}" <<'PY'
import hashlib
import os
import pathlib
import pickle
import re
import stat
import sys

operation, source_text, checkpoint_text, destination_text = sys.argv[1:]
if operation not in {"authenticate", "copy", "repair", "verify"}:
    raise SystemExit("unsupported replay inventory operation")

source_root = pathlib.Path(source_text)
checkpoint_root = pathlib.Path(checkpoint_text)
destination_root = pathlib.Path(destination_text) if destination_text else None
safe_generation = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
safe_chunk = re.compile(r"^[^/\\]+\.npz(?:\.zst)?$")
# Both supported runners store their directory-backed training replay in
# LOG_DIR/replay. They differ only in the checkpoint saveable name.
replay_pickles = ("replay.pkl", "replay_train.pkl")


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(
            f"global object {module}.{name} is not allowed in replay inventory"
        )

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("persistent IDs are not allowed")


def require_directory(path, label):
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{label} is not a real directory: {path}")


def regular_file(path, label):
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is not a regular file: {path}")
    return metadata


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_checkpoint_directory():
    require_directory(checkpoint_root, "checkpoint root")
    latest = checkpoint_root / "latest"
    regular_file(latest, "checkpoint latest pointer")
    value = latest.read_text(encoding="utf-8").strip()
    if not safe_generation.fullmatch(value):
        raise ValueError("checkpoint latest pointer is unsafe")
    selected = checkpoint_root / value
    require_directory(selected, "selected checkpoint")
    regular_file(selected / "done", "checkpoint completion marker")
    return selected


def validate_inventory(value, label):
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or value.get("format") != "embodied-replay-chunks-v1"
        or set(value) != {"format", "chunks"}
        or not isinstance(value["chunks"], list)
    ):
        raise ValueError(f"{label} has an unsupported replay inventory")
    entries = {}
    ordered = []
    for entry in value["chunks"]:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"filename", "size", "sha256"}
        ):
            raise ValueError(f"{label} contains a malformed replay entry")
        name = entry["filename"]
        size = entry["size"]
        digest = entry["sha256"]
        if (
            not isinstance(name, str)
            or not safe_chunk.fullmatch(name)
            or pathlib.PurePath(name).name != name
            or name in entries
        ):
            raise ValueError(f"{label} contains an unsafe replay filename")
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError(f"{label} contains an invalid replay size")
        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError(f"{label} contains an invalid replay digest")
        entries[name] = (size, digest)
        ordered.append(name)
    if ordered != sorted(ordered):
        raise ValueError(f"{label} replay inventory is not canonically sorted")
    return entries


def load_inventories(selected):
    present = []
    for pickle_name in replay_pickles:
        path = selected / pickle_name
        try:
            regular_file(path, pickle_name)
        except FileNotFoundError:
            continue
        present.append((pickle_name, path))
    if len(present) > 1:
        raise ValueError(
            "checkpoint contains ambiguous replay.pkl and replay_train.pkl"
        )
    if not present:
        return {"replay": None}
    pickle_name, path = present[0]
    with path.open("rb") as handle:
        value = RestrictedUnpickler(handle).load()
        if handle.read(1):
            raise ValueError(f"{pickle_name} contains trailing data")
    return {"replay": validate_inventory(value, pickle_name)}


def scan_chunks(directory, *, allow_missing):
    try:
        require_directory(directory, "replay directory")
    except FileNotFoundError:
        if allow_missing:
            return {}
        raise
    result = {}
    for child in directory.iterdir():
        metadata = child.lstat()
        if not stat.S_ISREG(metadata.st_mode) or not safe_chunk.fullmatch(child.name):
            raise ValueError(f"unsafe entry in replay directory: {child}")
        result[child.name] = metadata
    return result


def verify_source(directory_name, inventory):
    directory = source_root / directory_name
    actual = scan_chunks(directory, allow_missing=True)
    if inventory is None:
        if actual:
            raise ValueError(
                f"{directory_name} contains checkpoint-unbound replay chunks"
            )
        return {}
    for name, (size, expected_digest) in inventory.items():
        metadata = actual.get(name)
        if metadata is None:
            raise ValueError(f"checkpoint replay chunk is missing: {directory / name}")
        if metadata.st_size != size:
            raise ValueError(f"checkpoint replay chunk size changed: {directory / name}")
        if digest_file(directory / name) != expected_digest:
            raise ValueError(f"checkpoint replay chunk hash changed: {directory / name}")
    return actual


def ensure_destination_directory(path):
    try:
        path.mkdir(mode=0o770)
    except FileExistsError:
        pass
    require_directory(path, "destination replay directory")


def copy_inventory(directory_name, inventory):
    source_directory = source_root / directory_name
    destination_directory = destination_root / directory_name
    actual = verify_source(directory_name, inventory)
    expected = inventory or {}
    if not expected and not destination_directory.exists():
        return
    ensure_destination_directory(destination_directory)
    existing = scan_chunks(destination_directory, allow_missing=False)

    for name in sorted(existing):
        if name not in expected:
            (destination_directory / name).unlink()

    for name, (size, expected_digest) in sorted(expected.items()):
        source = source_directory / name
        destination = destination_directory / name
        keep = False
        try:
            metadata = regular_file(destination, "destination replay chunk")
        except FileNotFoundError:
            pass
        else:
            keep = (
                metadata.st_size == size
                and digest_file(destination) == expected_digest
            )
        if keep:
            continue

        temporary = destination_directory / f".{name}.tmp.{os.getpid()}"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o660,
        )
        try:
            with source.open("rb") as reader, os.fdopen(
                descriptor, "wb", closefd=False
            ) as writer:
                for block in iter(lambda: reader.read(1024 * 1024), b""):
                    writer.write(block)
                writer.flush()
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if temporary.stat().st_size != size or digest_file(temporary) != expected_digest:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"copied replay chunk failed validation: {source}")
        os.replace(temporary, destination)

    descriptor = os.open(
        destination_directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


try:
    require_directory(source_root, "source root")
    selected = read_checkpoint_directory()
    inventories = load_inventories(selected)
    if operation in {"authenticate", "repair", "verify"}:
        for directory_name, inventory in inventories.items():
            actual = verify_source(directory_name, inventory)
            expected_names = set((inventory or {}).keys())
            if operation == "verify" and set(actual) != expected_names:
                extras = sorted(set(actual) - expected_names)
                raise ValueError(
                    f"{directory_name} contains checkpoint-unbound replay chunks: "
                    + ", ".join(extras[:5])
                )
            if operation == "repair":
                extras = sorted(set(actual) - expected_names)
                directory = source_root / directory_name
                for name in extras:
                    (directory / name).unlink()
                if extras:
                    descriptor = os.open(
                        directory,
                        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                    )
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
    else:
        if destination_root is None:
            raise ValueError("copy operation requires a destination")
        require_directory(destination_root, "destination root")
        for directory_name, inventory in inventories.items():
            copy_inventory(directory_name, inventory)
except (
    FileNotFoundError,
    OSError,
    UnicodeError,
    ValueError,
    pickle.PickleError,
) as exc:
    print(f"Replay inventory error: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
}

scratch_sync_replay_directories_are_empty() {
    local root="$1"
    local directory
    local entry
    for directory in "${root}/replay" "${root}/train_replay"; do
        [[ -e "${directory}" || -L "${directory}" ]] || continue
        if [[ ! -d "${directory}" || -L "${directory}" ]]; then
            echo "Replay state path is unsafe: ${directory}" >&2
            return 1
        fi
        for entry in "${directory}"/* "${directory}"/.[!.]* "${directory}"/..?*; do
            [[ -e "${entry}" || -L "${entry}" ]] || continue
            echo "Replay state is not bound to a completed checkpoint: ${entry}" >&2
            return 1
        done
    done
    return 0
}

scratch_sync_copy_log_dir() {
    local src="$1"
    local dst="$2"
    local checkpoint_snapshot="${3:-}"
    local checkpoint_source="${4:-${src}/checkpoint.pkl}"

    mkdir -p "${dst}" || return 1
    rsync -a --delete --no-owner --no-group \
        --chmod=Dug+rwx,Do-rwx,Fug+rw,Fo-rwx \
        --exclude='/checkpoint.pkl/' --exclude='/latest.pt' \
        --exclude='/checkpoint.pkl.archive/' \
        --exclude='/replay/' --exclude='/train_replay/' \
        --exclude='/.checkpoint-pins/' \
        --exclude='/.sync_complete.json' \
        "${src}/" "${dst}/" || return 1
    # Remove checkpoint archives explicitly because rsync exclusions also protect them
    # from deletion.
    rm -rf -- "${dst}/checkpoint.pkl.archive" || return 1

    if [[ -L "${src}/checkpoint.pkl" ]]; then
        echo "Invalid checkpoint: checkpoint.pkl may not be a symlink." >&2
        return 1
    elif [[ -d "${checkpoint_source}" && ! -L "${checkpoint_source}" ]]; then
        if scratch_sync_checkpoint_is_valid "${checkpoint_source}"; then
            scratch_sync_copy_checkpoint_dir \
                "${checkpoint_source}" "${dst}/checkpoint.pkl" \
                "${checkpoint_snapshot}" || return 1
            scratch_sync_checkpoint_replay_inventory \
                copy "${src}" "${dst}/checkpoint.pkl" "${dst}" || return 1
            rm -rf -- "${dst}/checkpoint.pkl.incomplete" || return 1
            rm -f -- "${dst}/latest.pt" || return 1
        else
            scratch_sync_replay_directories_are_empty "${src}" || return 1
            scratch_sync_copy_dir \
                "${src}/checkpoint.pkl" \
                "${dst}/checkpoint.pkl.incomplete" || return 1
            rm -rf -- "${dst}/checkpoint.pkl" || return 1
            rm -rf -- "${dst}/checkpoint.pkl.archive" || return 1
            rm -rf -- "${dst}/replay" "${dst}/train_replay" || return 1
            rm -f -- "${dst}/latest.pt" || return 1
            echo "Backed up incomplete checkpoint as ${dst}/checkpoint.pkl.incomplete" >&2
        fi
    elif [[ -f "${src}/latest.pt" && ! -L "${src}/latest.pt" ]]; then
        scratch_sync_replay_directories_are_empty "${src}" || return 1
        scratch_sync_copy_file "${src}/latest.pt" "${dst}/latest.pt" || return 1
        rm -rf -- "${dst}/checkpoint.pkl" "${dst}/checkpoint.pkl.incomplete" \
            "${dst}/checkpoint.pkl.archive" || return 1
        rm -rf -- "${dst}/replay" "${dst}/train_replay" || return 1
    else
        scratch_sync_replay_directories_are_empty "${src}" || return 1
        rm -rf -- "${dst}/checkpoint.pkl" "${dst}/checkpoint.pkl.incomplete" \
            "${dst}/checkpoint.pkl.archive" || return 1
        rm -rf -- "${dst}/replay" "${dst}/train_replay" || return 1
        rm -f -- "${dst}/latest.pt" || return 1
    fi
}

scratch_sync_compact_durable_checkpoint_state() {
    local run_dir="$1"
    local checkpoint_root="${run_dir}/checkpoint.pkl"
    local latest

    # Keep two scratch checkpoints so the recovery monitor can pin one during rotation.
    # A quiescent run needs only the selected generation.
    if [[ -e "${checkpoint_root}" || -L "${checkpoint_root}" ]]; then
        if [[ ! -d "${checkpoint_root}" || -L "${checkpoint_root}" ]]; then
            echo "Error: checkpoint.pkl is not a real directory: ${checkpoint_root}" >&2
            return 1
        fi
        latest="$(scratch_sync_checkpoint_latest "${checkpoint_root}")" || {
            echo "Error: ${checkpoint_root} has no completed checkpoint generation." >&2
            return 1
        }
        # Validate the checkpoint and its replay inventory before removing unreferenced
        # data.
        scratch_sync_checkpoint_replay_inventory \
            authenticate "${run_dir}" "${checkpoint_root}" || return 1
        scratch_sync_cleanup_checkpoint_dir "${checkpoint_root}" "${latest}" ||
            return 1
        # Filter replay in place for runs that do not use a private copy stage.
        scratch_sync_checkpoint_replay_inventory \
            copy "${run_dir}" "${checkpoint_root}" "${run_dir}" || return 1
    fi
    rm -rf -- "${run_dir}/checkpoint.pkl.archive"
}

scratch_sync_has_valid_checkpoint() {
    if [[ -n "${SCRATCH_SYNC_RESUME_SOURCE_DIR:-}" &&
          -d "${SCRATCH_SYNC_RESUME_SOURCE_DIR}/checkpoint.pkl" &&
          ! -L "${SCRATCH_SYNC_RESUME_SOURCE_DIR}/checkpoint.pkl" ]] &&
        scratch_sync_checkpoint_is_valid \
            "${SCRATCH_SYNC_RESUME_SOURCE_DIR}/checkpoint.pkl"; then
        return 0
    fi
    if [[ -n "${LOG_DIR:-}" && -d "${LOG_DIR}/checkpoint.pkl" &&
          ! -L "${LOG_DIR}/checkpoint.pkl" ]] &&
        scratch_sync_checkpoint_is_valid "${LOG_DIR}/checkpoint.pkl"; then
        return 0
    fi
    if [[ -n "${WORK_LOG_DIR:-}" && -d "${WORK_LOG_DIR}/checkpoint.pkl" &&
          ! -L "${WORK_LOG_DIR}/checkpoint.pkl" ]] &&
        scratch_sync_checkpoint_is_valid "${WORK_LOG_DIR}/checkpoint.pkl"; then
        return 0
    fi
    return 1
}

scratch_sync_has_prior_run_state() {
    local dir

    for dir in "${LOG_DIR:-}" "${WORK_LOG_DIR:-}"; do
        [[ -n "${dir}" ]] || continue
        scratch_sync_dir_has_entries "${dir}" && return 0
    done
    return 1
}

scratch_sync_resume_requires_checkpoint() {
    [[ "${RUN_STORAGE_MODE_RESOLVED:-new}" == "resume" ]]
}

scratch_sync_empty_preempt_payload_is_empty() {
    local directory="$1"
    python3 - "${directory}" <<'PY'
import os
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1])
try:
    root_metadata = root.lstat()
except OSError:
    raise SystemExit(1)
if not stat.S_ISDIR(root_metadata.st_mode) or stat.S_ISLNK(root_metadata.st_mode):
    raise SystemExit(1)


def entry_metadata(path):
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        raise SystemExit(1)


# Preserve incomplete checkpoints for inspection; they cannot authorize a fresh start.
for name in (
    "checkpoint.pkl",
    "checkpoint.pkl.incomplete",
    "checkpoint.pkl.archive",
    "latest.pt",
):
    if entry_metadata(root / name) is not None:
        raise SystemExit(1)

# Log writers may create empty files while the agent is being initialized, but
# a non-empty record proves that the attempt advanced beyond an empty launch.
for name in (
    "metrics.jsonl",
    "scores.jsonl",
    "train_episodes.jsonl",
    "eval_episodes.jsonl",
):
    metadata = entry_metadata(root / name)
    if metadata is None:
        continue
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size:
        raise SystemExit(1)

# Empty output directories may precede the first checkpoint; populated directories
# require one.
try:
    children = list(root.iterdir())
except OSError:
    raise SystemExit(1)
for child in children:
    name = child.name
    state_directory = (
        name in {"replay", "train_replay", "analysis"}
        or name.startswith("eval_replay")
        or name.startswith("analysis-replica-")
    )
    if not state_directory:
        continue
    metadata = entry_metadata(child)
    if metadata is None or not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(
        metadata.st_mode
    ):
        raise SystemExit(1)
    try:
        with os.scandir(child) as entries:
            if next(entries, None) is not None:
                raise SystemExit(1)
    except OSError:
        raise SystemExit(1)
PY
}

scratch_sync_is_certified_empty_preempt() {
    local directory="$1"

    [[ -d "${directory}" && ! -L "${directory}" &&
       -f "${directory}/.sync_complete.json" &&
       ! -L "${directory}/.sync_complete.json" ]] || return 1
    scratch_sync_verify_integrity_marker \
        "${directory}" ".sync_complete.json" || return 1
    scratch_sync_empty_preempt_payload_is_empty "${directory}" || return 1

    # Verify run identity, attempt history, and terminal state before permitting a
    # restart without a checkpoint.
    python3 - "${directory}" <<'PY'
import json
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1])


def read_json(name):
    path = root / name
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"unsafe metadata: {name}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(name):
    path = root / name
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"unsafe metadata: {name}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object record: {name}")
                records.append(value)
    if not records:
        raise ValueError(f"empty metadata: {name}")
    return records


try:
    marker = read_json(".sync_complete.json")
    manifest = read_json("run_manifest.json")
    status = read_json("run_status.json")
    attempts = read_jsonl("run_attempts.jsonl")
    events = read_jsonl("run_events.jsonl")

    identity = manifest.get("identity")
    if manifest.get("schema_version") != 1 or not isinstance(identity, dict):
        raise ValueError("invalid manifest")
    run_id = identity.get("RUN_ID")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("invalid run ID")
    if (
        marker.get("schema_version") != 2
        or marker.get("marker_kind") != "final-publish"
        or marker.get("checkpoint") is not None
        or marker.get("run_uid") != run_id
    ):
        raise ValueError("invalid empty final publish")

    last_attempt = attempts[-1]
    attempt = last_attempt.get("attempt")
    attempt_id = attempt.get("RUN_ATTEMPT_ID") if isinstance(attempt, dict) else None
    if (
        last_attempt.get("schema_version") != 1
        or last_attempt.get("run_id") != run_id
        or last_attempt.get("mode") not in {"new", "resume"}
        or not isinstance(attempt_id, str)
        or not attempt_id
    ):
        raise ValueError("invalid final attempt")

    last_event = events[-1]
    status_fields = status.get("fields")
    event_fields = last_event.get("fields")
    if (
        status.get("schema_version") != 1
        or status.get("run_id") != run_id
        or status.get("attempt_id") != attempt_id
        or status.get("event") != "terminal"
        or status.get("status") != "PREEMPTED"
        or not isinstance(status_fields, dict)
        or status_fields.get("signal") != "USR1"
        or last_event.get("schema_version") != 1
        or last_event.get("run_id") != run_id
        or last_event.get("attempt_id") != attempt_id
        or last_event.get("event") != "terminal"
        or last_event.get("status") != "PREEMPTED"
        or not isinstance(event_fields, dict)
        or event_fields.get("signal") != "USR1"
        or last_event.get("sequence") != status.get("sequence")
    ):
        raise ValueError("run did not end in a matching preemption event")
except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
    raise SystemExit(1)
PY
}

scratch_sync_is_uninitialized_wandb_empty_preempt() {
    local directory="$1"
    local expected_run_id="$2"

    [[ -n "${expected_run_id}" ]] || return 1
    scratch_sync_is_certified_empty_preempt "${directory}" || return 1

    # A remote-run receipt requires a checkpoint before the W&B run can be reused.
    if [[ -e "${directory}/wandb_run.json" ||
          -L "${directory}/wandb_run.json" ]]; then
        return 1
    fi

    # The completion envelope authenticates the manifest; bind the exception
    # to the exact logical run requested by this requeued Slurm allocation.
    python3 - "${directory}/run_manifest.json" "${expected_run_id}" <<'PY'
import json
import pathlib
import stat
import sys

path = pathlib.Path(sys.argv[1])
expected_run_id = sys.argv[2]
try:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("unsafe manifest")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    identity = manifest.get("identity")
    if (
        manifest.get("schema_version") != 1
        or not isinstance(identity, dict)
        or identity.get("RUN_ID") != expected_run_id
    ):
        raise ValueError("run identity mismatch")
except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
    raise SystemExit(1)
PY
}

scratch_sync_has_resumable_durable_state() {
    if scratch_sync_has_valid_checkpoint; then
        return 0
    fi

    # Revalidate completed preemptions without a checkpoint before restoring storage.
    local directory
    for directory in \
        "${WORK_LOG_DIR:-}" \
        "${LOG_DIR:-}" \
        "${SCRATCH_SYNC_RESUME_SOURCE_DIR:-}"; do
        [[ -n "${directory}" ]] || continue
        if scratch_sync_is_certified_empty_preempt "${directory}"; then
            return 0
        fi
    done
    return 1
}

scratch_sync_validate_resume_checkpoint() {
    local selected_dir="${1:-${SCRATCH_SYNC_RESUME_SOURCE_DIR:-${LOG_DIR:-}}}"
    scratch_sync_resume_requires_checkpoint || return 0
    if [[ -n "${selected_dir}" &&
          -d "${selected_dir}/checkpoint.pkl" &&
          ! -L "${selected_dir}/checkpoint.pkl" ]] &&
       scratch_sync_checkpoint_is_valid "${selected_dir}/checkpoint.pkl"; then
        if ! scratch_sync_checkpoint_replay_inventory \
            verify "${selected_dir}" "${selected_dir}/checkpoint.pkl"; then
            echo "Refusing to resume checkpoint-unbound or corrupt replay state." >&2
            echo "  SELECTED_SOURCE=${selected_dir}" >&2
            echo "The source remains preserved for inspection or explicit repair." >&2
            return 1
        fi
        return 0
    fi

    # Allow initialization at step zero only after verifying a preemption before the
    # first checkpoint.
    # Revalidate after restore; callers cannot set this flag.
    if [[ "${SCRATCH_SYNC_EMPTY_PREEMPT_RESTART:-0}" -eq 1 &&
          -n "${LOG_DIR:-}" && "${selected_dir}" == "${LOG_DIR}" ]]; then
        if scratch_sync_empty_preempt_payload_is_empty "${selected_dir}"; then
            return 0
        fi
        echo "Refusing empty-preemption restart because restored training state is no longer empty." >&2
    elif scratch_sync_is_certified_empty_preempt "${selected_dir}"; then
        SCRATCH_SYNC_EMPTY_PREEMPT_RESTART=1
        echo "Continuing a certified pre-checkpoint preemption from fresh learner state." >&2
        echo "  PREEMPTED_SOURCE=${selected_dir}" >&2
        return 0
    fi

    cat >&2 <<EOF
Refusing to resume the selected run source without a valid completed checkpoint.
  WANDB_ID=${WANDB_ID:-}
  WANDB_RESUME=${WANDB_RESUME:-}
  SELECTED_SOURCE=${selected_dir}
  LOG_DIR=${LOG_DIR:-}
  WORK_LOG_DIR=${WORK_LOG_DIR:-}
EOF
    return 1
}

scratch_sync_directory_has_exact_checkpoint_replay() {
    local directory="$1"
    [[ -d "${directory}" && ! -L "${directory}" &&
       -d "${directory}/checkpoint.pkl" &&
       ! -L "${directory}/checkpoint.pkl" ]] || return 1
    scratch_sync_checkpoint_is_valid "${directory}/checkpoint.pkl" || return 1
    scratch_sync_checkpoint_replay_inventory \
        verify "${directory}" "${directory}/checkpoint.pkl"
}

scratch_sync_directory_is_deletion_backing() {
    local directory="$1"
    if scratch_sync_directory_has_exact_checkpoint_replay "${directory}"; then
        return 0
    fi
    # Only a verified preemption without learner state can authorize cleanup without a
    # checkpoint.
    scratch_sync_is_certified_empty_preempt "${directory}"
}

scratch_sync_directory_meets_checkpoint_high_water() {
    local directory="$1"
    local minimum="${2:-floor}"
    local marker="${3:-${SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_PATH:-}}"
    [[ -n "${marker}" ]] || return 0
    if [[ ! -e "${marker}" && ! -L "${marker}" ]]; then
        return 0
    fi
    if [[ ! -d "${directory}/checkpoint.pkl" ||
          -L "${directory}/checkpoint.pkl" ]]; then
        echo "Error: durable progress exists but candidate has no completed checkpoint." >&2
        return 1
    fi
    local checkpoint
    checkpoint="$(scratch_sync_checkpoint_latest \
        "${directory}/checkpoint.pkl")" || return 1
    scratch_sync_checkpoint_high_water_check \
        "${directory}" "${checkpoint}" "${minimum}" "${marker}"
}

scratch_sync_discard_incomplete_publish_stage() {
    # Remove incomplete publication stages to bound storage use; they cannot be resumed.
    local parent

    if [[ ! -d "${SCRATCH_SYNC_STAGE_DIR}" ||
          -L "${SCRATCH_SYNC_STAGE_DIR}" ]]; then
        echo "Error: unsafe publish stage requires inspection: ${SCRATCH_SYNC_STAGE_DIR}" >&2
        return 1
    fi
    parent="$(dirname "${WORK_LOG_DIR}")"
    rm -rf -- "${SCRATCH_SYNC_STAGE_DIR}" || return 1
    scratch_sync_fsync_directory "${parent}" || return 1
    echo "Discarded uncommitted publish stage ${SCRATCH_SYNC_STAGE_DIR}." >&2
}

scratch_sync_recover_publish_transaction() {
    # Called only while holding the per-run durable lock.
    local stage_time=""
    local durable_time=""
    local stage_checkpoint=""
    local durable_checkpoint=""
    local stage_step=""
    local durable_step=""

    if [[ -e "${SCRATCH_SYNC_STAGE_DIR}" || -L "${SCRATCH_SYNC_STAGE_DIR}" ]]; then
        if [[ ! -d "${SCRATCH_SYNC_STAGE_DIR}" ||
              -L "${SCRATCH_SYNC_STAGE_DIR}" ]]; then
            echo "Error: unsafe publish stage requires inspection: ${SCRATCH_SYNC_STAGE_DIR}" >&2
            return 1
        fi
        if [[ ! -e "${SCRATCH_SYNC_STAGE_DIR}/.sync_complete.json" &&
              ! -L "${SCRATCH_SYNC_STAGE_DIR}/.sync_complete.json" ]]; then
            scratch_sync_discard_incomplete_publish_stage || return 1
        elif [[ ! -f "${SCRATCH_SYNC_STAGE_DIR}/.sync_complete.json" ||
                -L "${SCRATCH_SYNC_STAGE_DIR}/.sync_complete.json" ]]; then
            echo "Error: unsafe publish completion marker requires inspection: ${SCRATCH_SYNC_STAGE_DIR}" >&2
            return 1
        fi
    fi

    if [[ -e "${SCRATCH_SYNC_STAGE_DIR}" || -L "${SCRATCH_SYNC_STAGE_DIR}" ]]; then
        if ! scratch_sync_validate_manifest "${SCRATCH_SYNC_STAGE_DIR}"; then
            echo "Warning: discarding uncommitted publish stage with invalid provenance." >&2
            scratch_sync_discard_incomplete_publish_stage || return 1
        fi
    fi
    if [[ -e "${SCRATCH_SYNC_STAGE_DIR}" || -L "${SCRATCH_SYNC_STAGE_DIR}" ]]; then
        if ! stage_time="$(scratch_sync_marker_time \
            "${SCRATCH_SYNC_STAGE_DIR}/.sync_complete.json")"; then
            echo "Warning: discarding publish stage with an invalid completion marker." >&2
            scratch_sync_discard_incomplete_publish_stage || return 1
            stage_time=""
        fi
    fi

    if [[ -e "${SCRATCH_SYNC_STAGE_DIR}" || -L "${SCRATCH_SYNC_STAGE_DIR}" ]]; then
        if ! scratch_sync_directory_meets_checkpoint_high_water \
            "${SCRATCH_SYNC_STAGE_DIR}" high; then
            echo "Error: recovered final publish stage is below the durable high-water checkpoint." >&2
            return 1
        fi
        if [[ -d "${SCRATCH_SYNC_STAGE_DIR}/checkpoint.pkl" &&
              ! -L "${SCRATCH_SYNC_STAGE_DIR}/checkpoint.pkl" ]]; then
            stage_checkpoint="$(scratch_sync_checkpoint_latest \
                "${SCRATCH_SYNC_STAGE_DIR}/checkpoint.pkl")" || return 1
            stage_step="$(scratch_sync_checkpoint_step \
                "${SCRATCH_SYNC_STAGE_DIR}" "${stage_checkpoint}")" || return 1
        fi
    fi

    if [[ -e "${SCRATCH_SYNC_STAGE_DIR}" || -L "${SCRATCH_SYNC_STAGE_DIR}" ]]; then
        if ! scratch_sync_fsync_tree "${SCRATCH_SYNC_STAGE_DIR}"; then
            echo "Error: recovered publish stage could not be flushed safely." >&2
            return 1
        fi
    fi

    if [[ -d "${WORK_LOG_DIR}" &&
          -f "${SCRATCH_SYNC_STAGE_DIR}/.sync_complete.json" ]]; then
        if ! scratch_sync_validate_manifest "${WORK_LOG_DIR}"; then
            echo "Error: canonical run manifest validation failed; see the validator diagnostic above." >&2
            return 1
        fi
        if [[ -f "${WORK_LOG_DIR}/.sync_complete.json" &&
              ! -L "${WORK_LOG_DIR}/.sync_complete.json" ]]; then
            durable_time="$(scratch_sync_marker_time "${WORK_LOG_DIR}/.sync_complete.json")" || {
                echo "Error: canonical run has an invalid completion marker: ${WORK_LOG_DIR}" >&2
                return 1
            }
            if [[ -d "${WORK_LOG_DIR}/checkpoint.pkl" &&
                  ! -L "${WORK_LOG_DIR}/checkpoint.pkl" ]]; then
                durable_checkpoint="$(scratch_sync_checkpoint_latest \
                    "${WORK_LOG_DIR}/checkpoint.pkl")" || return 1
                durable_step="$(scratch_sync_checkpoint_step \
                    "${WORK_LOG_DIR}" "${durable_checkpoint}")" || return 1
            fi
            if [[ -n "${durable_step}" && -z "${stage_step}" ]] ||
               { [[ -n "${durable_step}" && -n "${stage_step}" ]] &&
                 scratch_sync_checkpoint_point_is_newer \
                    "${durable_step}" "${durable_checkpoint}" \
                    "${stage_step}" "${stage_checkpoint}"; } ||
               { [[ "${durable_step}" == "${stage_step}" &&
                     "${durable_checkpoint}" == "${stage_checkpoint}" ]] &&
                 [[ "${stage_time}" < "${durable_time}" ||
                    "${stage_time}" == "${durable_time}" ]]; }; then
                echo "Error: refusing to roll back canonical run from a stale publish stage." >&2
                return 1
            fi
        fi
    fi

    if [[ ! -e "${WORK_LOG_DIR}" && ! -L "${WORK_LOG_DIR}" ]]; then
        if [[ -f "${SCRATCH_SYNC_STAGE_DIR}/.sync_complete.json" ]]; then
            mv -- "${SCRATCH_SYNC_STAGE_DIR}" "${WORK_LOG_DIR}" || return 1
            scratch_sync_fsync_directory "$(dirname "${WORK_LOG_DIR}")" || return 1
        elif [[ -d "${SCRATCH_SYNC_PREVIOUS_DIR}" ]]; then
            if ! scratch_sync_validate_manifest "${SCRATCH_SYNC_PREVIOUS_DIR}"; then
                echo "Error: previous durable generation failed manifest validation." >&2
                return 1
            fi
            mv -- "${SCRATCH_SYNC_PREVIOUS_DIR}" "${WORK_LOG_DIR}" || return 1
            scratch_sync_fsync_directory "$(dirname "${WORK_LOG_DIR}")" || return 1
        fi
    fi

    if [[ -d "${WORK_LOG_DIR}" && -f "${SCRATCH_SYNC_STAGE_DIR}/.sync_complete.json" ]]; then
        rm -rf -- "${SCRATCH_SYNC_PREVIOUS_DIR}" || return 1
        mv -- "${WORK_LOG_DIR}" "${SCRATCH_SYNC_PREVIOUS_DIR}" || return 1
        if ! mv -- "${SCRATCH_SYNC_STAGE_DIR}" "${WORK_LOG_DIR}"; then
            mv -- "${SCRATCH_SYNC_PREVIOUS_DIR}" "${WORK_LOG_DIR}" || true
            scratch_sync_fsync_directory "$(dirname "${WORK_LOG_DIR}")" ||
                echo "Warning: recovered publish rollback could not be flushed." >&2
            return 1
        fi
        scratch_sync_fsync_directory "$(dirname "${WORK_LOG_DIR}")" || return 1
    fi
}

scratch_sync_marker_time() {
    local marker="$1"
    local marker_directory
    local marker_name
    marker_directory="$(dirname "${marker}")"
    marker_name="$(basename "${marker}")"
    if ! scratch_sync_verify_integrity_marker \
        "${marker_directory}" "${marker_name}"; then
        echo "Error: critical artifact integrity validation failed for ${marker}." >&2
        return 1
    fi
    python3 -c '
import json, os, stat, sys
path = sys.argv[1]
metadata = os.lstat(path)
if not stat.S_ISREG(metadata.st_mode):
    raise SystemExit(2)
with open(path, "r", encoding="utf-8") as handle:
    value = json.load(handle)
timestamp = value.get("completed_at")
completed_ns = value.get("completed_ns")
run_uid = value.get("run_uid")
if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
    raise SystemExit(2)
if not isinstance(completed_ns, str) or not completed_ns.isdigit() or len(completed_ns) != 19:
    raise SystemExit(2)
expected = os.environ.get("RUN_UID")
if expected and run_uid != expected:
    raise SystemExit(2)
print(completed_ns)
' "${marker}"
}

scratch_sync_archive_residual() {
    local source="$1"
    local label="$2"
    local stamp
    local recovery
    local stage

    stamp="$(date -u +%Y%m%dT%H%M%S%N)_${BASHPID}"
    recovery="${WORK_LOG_DIR}.${label}-${stamp}"
    stage="${recovery}.publish-new"
    if [[ -e "${recovery}" || -L "${recovery}" ||
          -e "${stage}" || -L "${stage}" ]]; then
        echo "Error: residual recovery destination collision: ${recovery}" >&2
        return 1
    fi
    scratch_sync_copy_dir "${source}" "${stage}" || return 1
    if ! scratch_sync_materialize_links "${source}" "${stage}"; then
        rm -rf -- "${stage}" || true
        return 1
    fi
    printf '{"recovered_at":"%s","label":"%s","source":"%s"}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${label}" "${source}" \
        > "${stage}/recovery_metadata.json" || return 1
    scratch_sync_fsync_tree "${stage}" || return 1
    mv -- "${stage}" "${recovery}" || return 1
    scratch_sync_fsync_directory "$(dirname "${recovery}")" || return 1
    rm -rf -- "${source}" || return 1
    echo "Archived residual scratch data to ${recovery}"
}

scratch_sync_preserve_incomplete_fork_attempt() {
    # Preserve incomplete child state before retrying an explicit fork from its verified
    # parent.
    if [[ "${SCRATCH_SYNC_RECOVERY_INCOMPLETE_ONLY:-0}" -ne 1 ]]; then
        echo "Error: incomplete fork preservation was requested without a certified fallback condition." >&2
        return 1
    fi

    local stamp
    local destination
    stamp="$(date -u +%Y%m%dT%H%M%S%N)_${BASHPID}"

    if [[ "${LOG_DIR}" != "${WORK_LOG_DIR}" &&
          ( -e "${LOG_DIR}" || -L "${LOG_DIR}" ) ]]; then
        if [[ ! -d "${LOG_DIR}" || -L "${LOG_DIR}" ]]; then
            echo "Error: incomplete fork scratch path is unsafe: ${LOG_DIR}" >&2
            return 1
        fi
        scratch_sync_archive_residual \
            "${LOG_DIR}" "incomplete-fork-scratch" || return 1
    fi

    if [[ -e "${WORK_LOG_DIR}" || -L "${WORK_LOG_DIR}" ]]; then
        if [[ ! -d "${WORK_LOG_DIR}" || -L "${WORK_LOG_DIR}" ]]; then
            echo "Error: incomplete fork canonical path is unsafe: ${WORK_LOG_DIR}" >&2
            return 1
        fi
        destination="${WORK_LOG_DIR}.incomplete-fork-canonical-${stamp}"
        if [[ -e "${destination}" || -L "${destination}" ]]; then
            echo "Error: incomplete fork canonical quarantine collision: ${destination}" >&2
            return 1
        fi
        mv -- "${WORK_LOG_DIR}" "${destination}" || return 1
        scratch_sync_fsync_directory "$(dirname "${WORK_LOG_DIR}")" || return 1
        echo "Preserved incomplete fork canonical data at ${destination}." >&2
    fi

    if [[ ! -d "${SCRATCH_SYNC_RECOVERY_DIR}" ||
          -L "${SCRATCH_SYNC_RECOVERY_DIR}" ]]; then
        echo "Error: incomplete fork recovery path is unsafe: ${SCRATCH_SYNC_RECOVERY_DIR}" >&2
        return 1
    fi
    destination="${SCRATCH_SYNC_RECOVERY_DIR}.fork-incomplete-${stamp}"
    if [[ -e "${destination}" || -L "${destination}" ]]; then
        echo "Error: incomplete fork recovery quarantine collision: ${destination}" >&2
        return 1
    fi
    mv -- "${SCRATCH_SYNC_RECOVERY_DIR}" "${destination}" || return 1
    scratch_sync_fsync_directory "$(dirname "${SCRATCH_SYNC_RECOVERY_DIR}")" || return 1
    echo "Preserved incomplete fork recovery data at ${destination}." >&2

    SCRATCH_SYNC_RECOVERY_INCOMPLETE_ONLY=0
    SCRATCH_SYNC_RECOVERY_SOURCE_DIR=""
    SCRATCH_SYNC_RECOVERY_CHECKPOINT=""
    SCRATCH_SYNC_RECOVERY_PUBLISHED_NS=""
    export SCRATCH_SYNC_RECOVERY_SOURCE_DIR SCRATCH_SYNC_RECOVERY_CHECKPOINT \
        SCRATCH_SYNC_RECOVERY_PUBLISHED_NS
}

scratch_sync_recover_newer_tombstone() {
    local backing="$1"
    local tombstone_checkpoint=""
    local backing_checkpoint=""
    local tombstone_step=""
    local backing_step=""
    SCRATCH_SYNC_TOMBSTONE_RESTORED=0

    # Require recovery to reach both the live checkpoint step and the durable progress
    # floor before cleanup.
    if [[ -d "${SCRATCH_SYNC_TOMBSTONE}/checkpoint.pkl" &&
          ! -L "${SCRATCH_SYNC_TOMBSTONE}/checkpoint.pkl" ]]; then
        tombstone_checkpoint="$(scratch_sync_checkpoint_latest \
            "${SCRATCH_SYNC_TOMBSTONE}/checkpoint.pkl")" || {
            echo "Error: scratch tombstone has unsafe checkpoint residue; preserving it." >&2
            return 1
        }
        tombstone_step="$(scratch_sync_checkpoint_step \
            "${SCRATCH_SYNC_TOMBSTONE}" "${tombstone_checkpoint}")" || {
            echo "Error: scratch tombstone has no safe checkpoint step; preserving it." >&2
            return 1
        }
    elif ! scratch_sync_empty_preempt_payload_is_empty \
        "${SCRATCH_SYNC_TOMBSTONE}"; then
        echo "Error: scratch tombstone has unverifiable state; preserving it." >&2
        return 1
    fi

    if [[ -d "${backing}/checkpoint.pkl" &&
          ! -L "${backing}/checkpoint.pkl" ]]; then
        backing_checkpoint="$(scratch_sync_checkpoint_latest \
            "${backing}/checkpoint.pkl")" || return 1
        backing_step="$(scratch_sync_checkpoint_step \
            "${backing}" "${backing_checkpoint}")" || return 1
    fi

    if [[ -n "${tombstone_step}" ]] &&
       { [[ -z "${backing_step}" ]] ||
         scratch_sync_checkpoint_point_is_newer \
            "${tombstone_step}" "${tombstone_checkpoint}" \
            "${backing_step}" "${backing_checkpoint}"; }; then
        if ! scratch_sync_directory_has_exact_checkpoint_replay \
            "${SCRATCH_SYNC_TOMBSTONE}"; then
            echo "Error: newer scratch tombstone is not checkpoint-exact; preserving it for repair." >&2
            return 1
        fi
        if ! scratch_sync_directory_meets_checkpoint_high_water \
            "${SCRATCH_SYNC_TOMBSTONE}" floor; then
            echo "Error: newer scratch tombstone does not meet the durable progress floor; preserving it for repair." >&2
            return 1
        fi
        if [[ -e "${LOG_DIR}" || -L "${LOG_DIR}" ]]; then
            echo "Error: cannot restore newer markerless tombstone over existing scratch state." >&2
            return 1
        fi
        mv -- "${SCRATCH_SYNC_TOMBSTONE}" "${LOG_DIR}" || return 1
        scratch_sync_fsync_directory "$(dirname "${LOG_DIR}")" || return 1
        SCRATCH_SYNC_TOMBSTONE_RESTORED=1
        export SCRATCH_SYNC_TOMBSTONE_RESTORED
        echo "Restored newer scratch checkpoint ${tombstone_checkpoint} instead of deleting its tombstone." >&2
    fi
}

scratch_sync_recover_cleanup_tombstone() {
    local mode="$1"
    local durable_backing=0
    local durable_backing_dir=""

    [[ -e "${SCRATCH_SYNC_TOMBSTONE}" || -L "${SCRATCH_SYNC_TOMBSTONE}" ]] || return 0
    # A requeued fork resumes from its child once that child has verified durable state.
    if [[ "${mode}" != "resume" && "${mode}" != "fork" ]]; then
        echo "Error: stale scratch cleanup data exists for this run: ${SCRATCH_SYNC_TOMBSTONE}" >&2
        return 1
    fi
    if [[ ! -d "${SCRATCH_SYNC_TOMBSTONE}" ||
          -L "${SCRATCH_SYNC_TOMBSTONE}" ]]; then
        echo "Error: scratch cleanup residue is unsafe." >&2
        return 1
    fi

    # Defer scratch deletion until restart after a complete publication or verified
    # preemption recovery.
    if [[ -d "${WORK_LOG_DIR}" &&
          -f "${WORK_LOG_DIR}/.sync_complete.json" ]] &&
       scratch_sync_validate_manifest "${WORK_LOG_DIR}" &&
       scratch_sync_verify_integrity_marker \
            "${WORK_LOG_DIR}" ".sync_complete.json" &&
       scratch_sync_directory_is_deletion_backing "${WORK_LOG_DIR}" &&
       scratch_sync_directory_meets_checkpoint_high_water \
            "${WORK_LOG_DIR}" floor; then
        durable_backing=1
        durable_backing_dir="${WORK_LOG_DIR}"
    elif [[ -d "${WORK_LOG_DIR}" &&
            -f "${WORK_LOG_DIR}/.recovery_available.json" ]] &&
         scratch_sync_validate_recovery_index "${WORK_LOG_DIR}" &&
         scratch_sync_select_recovery_snapshot &&
         scratch_sync_directory_meets_checkpoint_high_water \
            "${SCRATCH_SYNC_RECOVERY_SOURCE_DIR}" floor; then
        durable_backing=1
        durable_backing_dir="${SCRATCH_SYNC_RECOVERY_SOURCE_DIR}"
    elif scratch_sync_select_recovery_snapshot &&
         scratch_sync_directory_meets_checkpoint_high_water \
            "${SCRATCH_SYNC_RECOVERY_SOURCE_DIR}" floor; then
        # A prior canonical generation can coexist with a newer hidden
        # recovery generation, in which case no index stub is needed.
        durable_backing=1
        durable_backing_dir="${SCRATCH_SYNC_RECOVERY_SOURCE_DIR}"
    fi
    if [[ "${durable_backing}" -ne 1 ]]; then
        echo "Error: scratch cleanup residue is not backed by certified durable state." >&2
        return 1
    fi
    if [[ -f "${SCRATCH_SYNC_TOMBSTONE}/run_manifest.json" ]] &&
       scratch_sync_validate_manifest "${SCRATCH_SYNC_TOMBSTONE}"; then
        scratch_sync_recover_newer_tombstone \
            "${durable_backing_dir}" || return 1
        if [[ "${SCRATCH_SYNC_TOMBSTONE_RESTORED:-0}" -eq 1 ]]; then
            return 0
        fi
        rm -rf -- "${SCRATCH_SYNC_TOMBSTONE}" || return 1
        echo "Removed verified scratch cleanup residue ${SCRATCH_SYNC_TOMBSTONE}"
        return 0
    fi
    # Preserve unverifiable cleanup remnants in durable storage before deleting them.
    scratch_sync_archive_residual "${SCRATCH_SYNC_TOMBSTONE}" "scratch-recovery"
}

scratch_sync_recovery_is_newer_than_canonical() {
    local canonical="$1"
    [[ -n "${SCRATCH_SYNC_RECOVERY_SOURCE_DIR:-}" ]] || return 1
    if [[ ! -d "${canonical}/checkpoint.pkl" ||
          -L "${canonical}/checkpoint.pkl" ]]; then
        return 0
    fi
    local canonical_checkpoint
    canonical_checkpoint="$(scratch_sync_checkpoint_latest "${canonical}/checkpoint.pkl")" || return 0
    if [[ "${canonical_checkpoint}" == "${SCRATCH_SYNC_RECOVERY_CHECKPOINT}" ]]; then
        return 1
    fi
    local canonical_step
    local recovery_step
    canonical_step="$(scratch_sync_checkpoint_step \
        "${canonical}" "${canonical_checkpoint}")" || return 0
    recovery_step="$(scratch_sync_checkpoint_step \
        "${SCRATCH_SYNC_RECOVERY_SOURCE_DIR}" \
        "${SCRATCH_SYNC_RECOVERY_CHECKPOINT}")" || return 1
    scratch_sync_checkpoint_point_is_newer \
        "${recovery_step}" "${SCRATCH_SYNC_RECOVERY_CHECKPOINT}" \
        "${canonical_step}" "${canonical_checkpoint}"
}

scratch_sync_promote_recovery_to_direct_durable() {
    local source="$1"
    if ! scratch_sync_directory_meets_checkpoint_high_water \
        "${source}" high; then
        echo "Error: refusing final recovery promotion below the recorded high-water checkpoint." >&2
        return 1
    fi
    if [[ -e "${SCRATCH_SYNC_STAGE_DIR}" || -L "${SCRATCH_SYNC_STAGE_DIR}" ]]; then
        echo "Error: direct recovery promotion stage already exists: ${SCRATCH_SYNC_STAGE_DIR}" >&2
        return 1
    fi
    mkdir -p "${SCRATCH_SYNC_STAGE_DIR}" || return 1
    if ! scratch_sync_copy_log_dir "${source}" "${SCRATCH_SYNC_STAGE_DIR}" ||
       ! scratch_sync_materialize_links "${source}" "${SCRATCH_SYNC_STAGE_DIR}"; then
        rm -rf -- "${SCRATCH_SYNC_STAGE_DIR}" || true
        return 1
    fi
    rm -f -- "${SCRATCH_SYNC_STAGE_DIR}/.recovery_complete.json" || return 1
    scratch_sync_write_complete_marker "${SCRATCH_SYNC_STAGE_DIR}" || return 1
    scratch_sync_verify_integrity_marker \
        "${SCRATCH_SYNC_STAGE_DIR}" ".sync_complete.json" || return 1
    scratch_sync_fsync_tree "${SCRATCH_SYNC_STAGE_DIR}" || return 1
    rm -rf -- "${SCRATCH_SYNC_PREVIOUS_DIR}" || return 1
    if [[ -e "${WORK_LOG_DIR}" || -L "${WORK_LOG_DIR}" ]]; then
        mv -- "${WORK_LOG_DIR}" "${SCRATCH_SYNC_PREVIOUS_DIR}" || return 1
    fi
    if ! mv -- "${SCRATCH_SYNC_STAGE_DIR}" "${WORK_LOG_DIR}"; then
        [[ -d "${SCRATCH_SYNC_PREVIOUS_DIR}" ]] &&
            mv -- "${SCRATCH_SYNC_PREVIOUS_DIR}" "${WORK_LOG_DIR}" || true
        scratch_sync_fsync_directory "$(dirname "${WORK_LOG_DIR}")" ||
            echo "Warning: direct recovery rollback could not be flushed." >&2
        return 1
    fi
    scratch_sync_fsync_directory "$(dirname "${WORK_LOG_DIR}")" || return 1
    rm -rf -- "${SCRATCH_SYNC_PREVIOUS_DIR}" ||
        echo "Warning: previous direct durable state remains at ${SCRATCH_SYNC_PREVIOUS_DIR}." >&2
    SCRATCH_SYNC_RESUME_SOURCE_DIR="${WORK_LOG_DIR}"
    echo "Promoted newer recovery snapshot to direct durable run ${WORK_LOG_DIR}"
}

scratch_sync_restore_logs() {
    local source="${SCRATCH_SYNC_RESUME_SOURCE_DIR:-${WORK_LOG_DIR}}"
    scratch_sync_dir_has_entries "${source}" || return 0
    if [[ "${SCRATCH_SYNC_DIRECT_DURABLE:-0}" -eq 1 ]]; then
        if [[ "${source}" != "${WORK_LOG_DIR}" ]]; then
            scratch_sync_promote_recovery_to_direct_durable "${source}"
        fi
        return $?
    fi

    if scratch_sync_dir_has_entries "${LOG_DIR}"; then
        echo "Error: refusing to merge residual scratch data with durable state: ${LOG_DIR}" >&2
        return 1
    fi

    local restore_stage="${LOG_DIR}.restore"
    if [[ -e "${restore_stage}" || -L "${restore_stage}" ]]; then
        echo "Error: stale restore stage requires inspection: ${restore_stage}" >&2
        return 1
    fi

    mkdir -p "$(dirname "${LOG_DIR}")" || return 1
    if ! scratch_sync_copy_dir "${source}" "${restore_stage}"; then
        rm -rf -- "${restore_stage}" || true
        return 1
    fi
    rm -rf -- "${LOG_DIR}" || return 1
    mv -- "${restore_stage}" "${LOG_DIR}" || return 1
    rm -f -- "${LOG_DIR}/.recovery_complete.json" || return 1
    echo "Restored complete run from ${source} to ${LOG_DIR}"
}

scratch_sync_copy_scheduler_logs_to() {
    local target_dir="$1"
    local source_path
    local target_name
    local temporary

    [[ -d "${target_dir}" && ! -L "${target_dir}" ]] || return 0
    if [[ -e "${target_dir}/slurm" || -L "${target_dir}/slurm" ]]; then
        if [[ ! -d "${target_dir}/slurm" || -L "${target_dir}/slurm" ]]; then
            echo "Error: scheduler log destination is unsafe: ${target_dir}/slurm" >&2
            return 1
        fi
    else
        mkdir -p "${target_dir}/slurm" || return 1
    fi
    for source_path in "${RUN_SLURM_STDOUT:-}" "${RUN_SLURM_STDERR:-}"; do
        [[ -n "${source_path}" && -f "${source_path}" &&
           ! -L "${source_path}" ]] || continue
        target_name="$(basename "${source_path}")"
        if [[ ! "${target_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
            echo "Error: unsafe scheduler log basename: ${target_name}" >&2
            return 1
        fi
        temporary="${target_dir}/slurm/.${target_name}.tmp.${BASHPID}"
        cp -p -- "${source_path}" "${temporary}" || return 1
        mv -f -- "${temporary}" "${target_dir}/slurm/${target_name}" || return 1
    done
}

scratch_sync_stage_scheduler_logs() {
    scratch_sync_copy_scheduler_logs_to "${LOG_DIR}"
}

scratch_sync_materialize_links() {
    local source="$1"
    local stage="$2"
    local helper="${RUN_STORAGE_HELPER:-${HELPER_DIR:-${SCRATCH_SYNC_HELPER_DIR}}/run_storage.py}"
    local -a command

    [[ -f "${helper}" ]] || {
        echo "Error: missing storage integrity helper ${helper}." >&2
        return 1
    }
    command=(python3 "${helper}" materialize-links --source "${source}" --stage "${stage}")
    command+=(--allowed-root "${source}")
    [[ -n "${SSRL_RUN_REPO_ROOT:-}" ]] && command+=(--allowed-root "${SSRL_RUN_REPO_ROOT}")
    # Materialize W&B log links only from this run's runtime root or approved cache
    # paths.
    [[ -n "${SSRL_LOCAL_RUNTIME_ROOT:-}" &&
       -d "${SSRL_LOCAL_RUNTIME_ROOT}" &&
       ! -L "${SSRL_LOCAL_RUNTIME_ROOT}" ]] &&
        command+=(--allowed-root "${SSRL_LOCAL_RUNTIME_ROOT}")
    [[ -d "${HOME:-}/.cache/wandb/logs" && ! -L "${HOME:-}/.cache/wandb/logs" ]] &&
        command+=(--allowed-root "${HOME}/.cache/wandb/logs")
    # A link may refer back to this exact run's older durable generation, but
    # never to an arbitrary sibling run merely because it shares the group
    # storage root.
    [[ -d "${WORK_LOG_DIR:-}" && ! -L "${WORK_LOG_DIR:-}" ]] &&
        command+=(--allowed-root "${WORK_LOG_DIR}")
    "${command[@]}"
}

scratch_sync_integrity_marker() {
    # Hash run metadata, results, and the completed checkpoint for publication
    # verification.
    # Replay integrity is checked separately.
    local operation="$1"
    local directory="$2"
    local marker_name="$3"
    shift 3
    python3 - "${operation}" "${directory}" "${marker_name}" "$@" <<'PY'
import datetime
import hashlib
import json
import os
import pathlib
import re
import stat
import sys
import uuid

operation, directory_text, marker_name, *extra = sys.argv[1:]
root = pathlib.Path(directory_text)
if not root.is_dir() or root.is_symlink():
    raise SystemExit("integrity root is not a real directory")
if not re.fullmatch(r"\.[A-Za-z0-9_.-]+\.json", marker_name):
    raise SystemExit("unsafe integrity marker name")
marker_path = root / marker_name
scope_v1 = "critical-run-state-v1"
scope_v2 = "critical-run-state-v2"
current_scope = scope_v2
supported_scopes = {scope_v1, scope_v2}
write_scope = os.environ.get(
    "SCRATCH_SYNC_WRITE_INTEGRITY_SCOPE", current_scope
)
if write_scope not in supported_scopes:
    raise SystemExit("unsupported requested integrity scope")


def regular_file(path: pathlib.Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISREG(metadata.st_mode)


def digest(path: pathlib.Path) -> tuple[int, str]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"critical artifact is not a regular file: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    after = path.lstat()
    if (
        not stat.S_ISREG(after.st_mode)
        or metadata.st_dev != after.st_dev
        or metadata.st_ino != after.st_ino
        or metadata.st_size != after.st_size
        or metadata.st_mtime_ns != after.st_mtime_ns
    ):
        raise ValueError(f"critical artifact changed while hashing: {path}")
    return metadata.st_size, hasher.hexdigest()


def add_tree(paths: set[str], relative: pathlib.PurePosixPath) -> None:
    base = root.joinpath(*relative.parts)
    if not base.exists() and not base.is_symlink():
        return
    if not base.is_dir() or base.is_symlink():
        raise ValueError(f"critical artifact tree is unsafe: {base}")
    for walk_root, directories, files in os.walk(base, followlinks=False):
        walk_path = pathlib.Path(walk_root)
        for name in directories:
            child = walk_path / name
            if child.is_symlink():
                raise ValueError(f"critical artifact tree contains a link: {child}")
        for name in files:
            child = walk_path / name
            if not regular_file(child):
                raise ValueError(f"critical artifact tree contains a non-file: {child}")
            paths.add(child.relative_to(root).as_posix())


def inventory(scope: str) -> tuple[list[dict[str, object]], str | None]:
    if not isinstance(scope, str) or scope not in supported_scopes:
        raise ValueError(f"unsupported critical artifact integrity scope: {scope!r}")
    manifest = root / "run_manifest.json"
    paths: set[str] = set()
    if manifest.exists() or manifest.is_symlink():
        if not regular_file(manifest):
            raise ValueError("run_manifest.json is unsafe")
        paths.add("run_manifest.json")
    optional_artifacts = ["run_attempts.jsonl"]
    # Version 2 also authenticates the initialization receipt.
    if scope == scope_v2:
        optional_artifacts.append("wandb_run.json")
    optional_artifacts.extend(
        (
            "dreamer_config.yaml",
            "config.yaml",
            "config.yml",
            "config.json",
            "metrics.jsonl",
            "scores.jsonl",
            "train_episodes.jsonl",
            "eval_episodes.jsonl",
            "run_events.jsonl",
            "run_status.json",
            "source_snapshot.json",
            "source_provenance/source_snapshot.json",
        )
    )
    for relative in optional_artifacts:
        candidate = root.joinpath(*pathlib.PurePosixPath(relative).parts)
        if candidate.exists() or candidate.is_symlink():
            if not regular_file(candidate):
                raise ValueError(f"critical artifact is unsafe: {candidate}")
            paths.add(relative)
    add_tree(paths, pathlib.PurePosixPath(".hydra"))
    add_tree(paths, pathlib.PurePosixPath("analysis"))

    checkpoint_name = None
    checkpoint_root = root / "checkpoint.pkl"
    if checkpoint_root.exists() or checkpoint_root.is_symlink():
        if not checkpoint_root.is_dir() or checkpoint_root.is_symlink():
            raise ValueError("checkpoint.pkl is not a real directory")
        latest_path = checkpoint_root / "latest"
        if not regular_file(latest_path):
            raise ValueError("checkpoint.pkl/latest is missing or unsafe")
        checkpoint_name = latest_path.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", checkpoint_name):
            raise ValueError("checkpoint.pkl/latest contains an unsafe generation")
        generation = checkpoint_root / checkpoint_name
        if not generation.is_dir() or generation.is_symlink():
            raise ValueError("completed checkpoint generation is missing or unsafe")
        if not regular_file(generation / "done"):
            raise ValueError("completed checkpoint has no safe done marker")
        paths.add("checkpoint.pkl/latest")
        add_tree(
            paths,
            pathlib.PurePosixPath("checkpoint.pkl") / checkpoint_name,
        )
    elif regular_file(root / "latest.pt"):
        # Protect archived PyTorch checkpoints against corruption without allowing
        # resume.
        paths.add("latest.pt")
    add_tree(paths, pathlib.PurePosixPath("checkpoint.pkl.incomplete"))
    add_tree(paths, pathlib.PurePosixPath("checkpoint.pkl.archive"))

    artifacts = []
    for relative in sorted(paths):
        size, sha256 = digest(root.joinpath(*pathlib.PurePosixPath(relative).parts))
        artifacts.append(
            {"path": relative, "size_bytes": size, "sha256": sha256}
        )
    return artifacts, checkpoint_name


def envelope(kind: str, fields: dict[str, object]) -> dict[str, object]:
    artifacts, checkpoint = inventory(write_scope)
    return {
        "schema_version": 2,
        "marker_kind": kind,
        **fields,
        "checkpoint": checkpoint,
        "integrity": {
            "algorithm": "sha256",
            "scope": write_scope,
            "artifacts": artifacts,
        },
    }


def install(value: dict[str, object]) -> None:
    temporary = root / (marker_name + ".tmp." + uuid.uuid4().hex)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        data = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
        view = memoryview(data)
        while view:
            view = view[os.write(descriptor, view):]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, marker_path)
    try:
        descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


if operation == "write-sync":
    if len(extra) != 3:
        raise SystemExit("write-sync requires completed_at, completed_ns, run_uid")
    completed_at, completed_ns, run_uid = extra
    install(
        envelope(
            "final-publish",
            {
                "completed_at": completed_at,
                "completed_ns": completed_ns,
                "run_uid": run_uid,
            },
        )
    )
elif operation == "write-recovery":
    if len(extra) != 5:
        raise SystemExit(
            "write-recovery requires generation, checkpoint, published_at, "
            "published_ns, run_uid"
        )
    generation, checkpoint, published_at, published_ns, run_uid = extra
    value = envelope(
        "periodic-recovery",
        {
            "published_at": published_at,
            "published_ns": published_ns,
            "generation": generation,
            "run_uid": run_uid,
        },
    )
    if value["checkpoint"] != checkpoint:
        raise SystemExit("recovery checkpoint changed before marker publication")
    install(value)
elif operation == "verify":
    if extra:
        raise SystemExit("verify takes no extra arguments")
    metadata = marker_path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise SystemExit("integrity marker is not a regular file")
    value = json.loads(marker_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 2:
        raise SystemExit("unsupported integrity marker schema")
    if value.get("marker_kind") not in {"final-publish", "periodic-recovery"}:
        raise SystemExit("invalid integrity marker kind")
    integrity = value.get("integrity")
    if not isinstance(integrity, dict):
        raise SystemExit("integrity inventory is missing")
    if integrity.get("algorithm") != "sha256":
        raise SystemExit("unsupported integrity algorithm")
    scope = integrity.get("scope")
    if not isinstance(scope, str) or scope not in supported_scopes:
        raise SystemExit("unsupported integrity scope")
    expected, checkpoint = inventory(scope)
    if integrity.get("artifacts") != expected:
        raise SystemExit("critical artifact integrity check failed")
    if value.get("checkpoint") != checkpoint:
        raise SystemExit("integrity marker checkpoint does not match the tree")
else:
    raise SystemExit("unknown integrity marker operation")
PY
}

scratch_sync_verify_integrity_marker() {
    local directory="$1"
    local marker_name="$2"
    scratch_sync_integrity_marker verify "${directory}" "${marker_name}"
}

scratch_sync_write_complete_marker() {
    local directory="$1"
    scratch_sync_integrity_marker write-sync "${directory}" ".sync_complete.json" \
        "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$(date -u +%s%N)" \
        "${RUN_UID:-unknown}"
}

scratch_sync_recovery_generation_name_is_safe() {
    local name="$1"
    [[ "${name}" =~ ^snapshot-[0-9]{19}-[0-9]+-[0-9]+$ ]]
}

scratch_sync_validate_recovery_layout() {
    # Return 1 for an absent store and 2 for an unsafe store.
    [[ -e "${SCRATCH_SYNC_RECOVERY_DIR}" ||
       -L "${SCRATCH_SYNC_RECOVERY_DIR}" ]] || return 1
    if [[ ! -d "${SCRATCH_SYNC_RECOVERY_DIR}" ||
          -L "${SCRATCH_SYNC_RECOVERY_DIR}" ]]; then
        echo "Error: recovery path is not a real directory: ${SCRATCH_SYNC_RECOVERY_DIR}" >&2
        return 2
    fi

    local entry
    for entry in "${SCRATCH_SYNC_RECOVERY_DIR}"/* \
                 "${SCRATCH_SYNC_RECOVERY_DIR}"/.[!.]* \
                 "${SCRATCH_SYNC_RECOVERY_DIR}"/..?*; do
        [[ -e "${entry}" || -L "${entry}" ]] || continue
        case "$(basename "${entry}")" in
            generations)
                if [[ ! -d "${entry}" || -L "${entry}" ]]; then
                    echo "Error: recovery generations path is unsafe: ${entry}" >&2
                    return 2
                fi
                ;;
            quarantine)
                if [[ ! -d "${entry}" || -L "${entry}" ]]; then
                    echo "Error: recovery quarantine path is unsafe: ${entry}" >&2
                    return 2
                fi
                local quarantined
                for quarantined in "${entry}"/* "${entry}"/.[!.]* "${entry}"/..?*; do
                    [[ -e "${quarantined}" || -L "${quarantined}" ]] || continue
                    if [[ -L "${quarantined}" ||
                          ( ! -d "${quarantined}" && ! -f "${quarantined}" ) ]]; then
                        echo "Error: unsafe recovery quarantine entry: ${quarantined}" >&2
                        return 2
                    fi
                done
                ;;
            current)
                if [[ ! -f "${entry}" || -L "${entry}" ]]; then
                    echo "Error: recovery current pointer is unsafe: ${entry}" >&2
                    return 2
                fi
                ;;
            checkpoint-high-water.json)
                if [[ ! -f "${entry}" || -L "${entry}" ]]; then
                    echo "Error: checkpoint high-water marker is unsafe: ${entry}" >&2
                    return 2
                fi
                ;;
            .current.tmp.*)
                if [[ ! -f "${entry}" || -L "${entry}" ]]; then
                    echo "Error: incomplete recovery pointer is unsafe: ${entry}" >&2
                    return 2
                fi
                ;;
            .checkpoint-high-water.json.tmp.*)
                if [[ ! -f "${entry}" || -L "${entry}" ]]; then
                    echo "Error: incomplete checkpoint high-water marker is unsafe: ${entry}" >&2
                    return 2
                fi
                ;;
            *)
                echo "Error: unexpected entry in recovery store: ${entry}" >&2
                return 2
                ;;
        esac
    done

    if [[ -e "${SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR}" ||
          -L "${SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR}" ]]; then
        local entry_name
        for entry in "${SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR}"/* \
                     "${SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR}"/.[!.]* \
                     "${SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR}"/..?*; do
            [[ -e "${entry}" || -L "${entry}" ]] || continue
            if [[ ! -d "${entry}" || -L "${entry}" ]]; then
                echo "Error: recovery generation entry is unsafe: ${entry}" >&2
                return 2
            fi
            entry_name="$(basename "${entry}")"
            if scratch_sync_recovery_generation_name_is_safe "${entry_name}"; then
                continue
            fi
            if [[ "${entry_name}" =~ ^\.stage-[0-9]+-[0-9]+-[0-9]{19}$ ]]; then
                continue
            fi
            echo "Error: unexpected recovery generation name: ${entry}" >&2
            return 2
        done
    fi
    return 0
}

scratch_sync_read_recovery_marker() {
    local generation_dir="$1"
    local generation_name="$2"
    local marker="${generation_dir}/.recovery_complete.json"
    local values

    if ! scratch_sync_verify_integrity_marker \
        "${generation_dir}" ".recovery_complete.json"; then
        echo "Error: recovery critical artifact integrity check failed: ${marker}" >&2
        return 1
    fi
    values="$(python3 -c '
import json, os, stat, sys
path, expected_generation = sys.argv[1:]
metadata = os.lstat(path)
if not stat.S_ISREG(metadata.st_mode):
    raise SystemExit(2)
with open(path, "r", encoding="utf-8") as handle:
    value = json.load(handle)
if value.get("schema_version") != 2:
    raise SystemExit(2)
if value.get("marker_kind") != "periodic-recovery":
    raise SystemExit(2)
published_ns = value.get("published_ns")
checkpoint = value.get("checkpoint")
generation = value.get("generation")
run_uid = value.get("run_uid")
if not isinstance(published_ns, str) or len(published_ns) != 19 or not published_ns.isdigit():
    raise SystemExit(2)
if not isinstance(checkpoint, str) or not checkpoint:
    raise SystemExit(2)
if generation != expected_generation:
    raise SystemExit(2)
expected_uid = os.environ.get("RUN_UID")
if expected_uid and run_uid != expected_uid:
    raise SystemExit(2)
if any(character in checkpoint for character in "\r\n\t"):
    raise SystemExit(2)
print(published_ns, checkpoint, sep="\t")
' "${marker}" "${generation_name}")" || {
        echo "Error: invalid recovery completion marker: ${marker}" >&2
        return 1
    }
    IFS=$'\t' read -r SCRATCH_SYNC_RECOVERY_PUBLISHED_NS \
        SCRATCH_SYNC_RECOVERY_CHECKPOINT <<< "${values}"
    export SCRATCH_SYNC_RECOVERY_PUBLISHED_NS SCRATCH_SYNC_RECOVERY_CHECKPOINT
}

scratch_sync_recovery_current_generation() {
    python3 -c '
import os, re, stat, sys
path = sys.argv[1]
metadata = os.lstat(path)
if not stat.S_ISREG(metadata.st_mode):
    raise SystemExit(2)
with open(path, "r", encoding="ascii", newline="") as handle:
    value = handle.read()
match = re.fullmatch(r"(snapshot-[0-9]{19}-[0-9]+-[0-9]+)\n?", value)
if not match:
    raise SystemExit(2)
print(match.group(1))
' "${SCRATCH_SYNC_RECOVERY_CURRENT}"
}

scratch_sync_recovery_marker_generation() {
    local marker="$1"
    python3 -c '
import json, os, re, stat, sys
path = sys.argv[1]
metadata = os.lstat(path)
if not stat.S_ISREG(metadata.st_mode):
    raise SystemExit(2)
with open(path, "r", encoding="utf-8") as handle:
    value = json.load(handle)
generation = value.get("generation")
if not isinstance(generation, str) or not re.fullmatch(
    r"snapshot-[0-9]{19}-[0-9]+-[0-9]+", generation
):
    raise SystemExit(2)
print(generation)
' "${marker}"
}

scratch_sync_directory_is_resumable_copy() {
    local directory="$1"
    [[ -d "${directory}" && ! -L "${directory}" &&
       -f "${directory}/run_manifest.json" &&
       ! -L "${directory}/run_manifest.json" &&
       -d "${directory}/checkpoint.pkl" &&
       ! -L "${directory}/checkpoint.pkl" ]] || return 1
    scratch_sync_checkpoint_is_valid "${directory}/checkpoint.pkl" || return 1
    scratch_sync_validate_manifest "${directory}" >/dev/null 2>&1 || return 1
    scratch_sync_checkpoint_replay_inventory \
        verify "${directory}" "${directory}/checkpoint.pkl" >/dev/null 2>&1
}

scratch_sync_quarantine_recovery_entry() {
    local source="$1"
    local label="$2"
    local quarantine="${SCRATCH_SYNC_RECOVERY_DIR}/quarantine"
    local stamp
    local destination

    stamp="$(date -u +%s%N)-${BASHPID}-${RANDOM}"
    destination="${quarantine}/${label}-${stamp}"
    mkdir -p "${quarantine}" || return 1
    [[ ! -e "${destination}" && ! -L "${destination}" ]] || return 1
    mv -- "${source}" "${destination}" || return 1
    scratch_sync_fsync_directory "$(dirname "${source}")" || return 1
    scratch_sync_fsync_directory "${quarantine}" || return 1
    echo "Quarantined unsafe recovery state at ${destination}." >&2
}

scratch_sync_reclaim_recovery_entry() {
    # Atomically move obsolete generations out of the recovery namespace before
    # recursive deletion.
    local source="$1"
    local label="$2"
    local quarantine="${SCRATCH_SYNC_RECOVERY_DIR}/quarantine"
    local stamp
    local destination

    [[ -d "${source}" && ! -L "${source}" ]] || return 1
    stamp="$(date -u +%s%N)-${BASHPID}-${RANDOM}"
    destination="${quarantine}/reclaim-${label}-${stamp}"
    mkdir -p "${quarantine}" || return 1
    [[ ! -e "${destination}" && ! -L "${destination}" ]] || return 1
    mv -- "${source}" "${destination}" || return 1
    scratch_sync_fsync_directory "$(dirname "${source}")" || return 1
    scratch_sync_fsync_directory "${quarantine}" || return 1
    if rm -rf -- "${destination}"; then
        scratch_sync_fsync_directory "${quarantine}" || return 1
    else
        echo "Warning: recovery cleanup residue remains at ${destination}." >&2
    fi
}

scratch_sync_cleanup_recovery_reclaims() {
    local residue
    local removed=0
    for residue in "${SCRATCH_SYNC_RECOVERY_DIR}/quarantine"/reclaim-*; do
        [[ -e "${residue}" || -L "${residue}" ]] || continue
        if [[ -L "${residue}" ||
              ( ! -d "${residue}" && ! -f "${residue}" ) ]]; then
            echo "Error: unsafe recovery cleanup residue: ${residue}" >&2
            return 1
        fi
        if rm -rf -- "${residue}"; then
            removed=1
        else
            echo "Warning: could not remove recovery cleanup residue ${residue}." >&2
        fi
    done
    if [[ "${removed}" -eq 1 ]]; then
        scratch_sync_fsync_directory \
            "${SCRATCH_SYNC_RECOVERY_DIR}/quarantine" || return 1
    fi
}

scratch_sync_recovery_has_reclaim_residue() {
    local residue
    for residue in "${SCRATCH_SYNC_RECOVERY_DIR}/quarantine"/reclaim-*; do
        [[ -e "${residue}" || -L "${residue}" ]] && return 0
    done
    return 1
}

scratch_sync_retry_recovery_reclaims_for_publication() {
    # Remove only deletion remnants already outside the resumable namespace.
    [[ -n "${SCRATCH_SYNC_LOCK_FD:-}" ]] &&
        flock -n "${SCRATCH_SYNC_LOCK_FD}" || return 1
    scratch_sync_validate_manifest "${LOG_DIR}" || return 1
    local checkpoint
    local pin
    local status=0
    checkpoint="$(scratch_sync_checkpoint_latest \
        "${LOG_DIR}/checkpoint.pkl")" || return 1
    pin="$(scratch_sync_pin_checkpoint_generation \
        "${LOG_DIR}/checkpoint.pkl" "${checkpoint}")" || return 1
    if scratch_sync_checkpoint_replay_inventory verify "${LOG_DIR}" "${pin}" &&
       scratch_sync_checkpoint_high_water_check \
            "${LOG_DIR}" "${checkpoint}" floor; then
        scratch_sync_cleanup_recovery_reclaims || status=$?
    else
        echo "Error: no validated live checkpoint authorizes recovery cleanup retry." >&2
        status=1
    fi
    scratch_sync_release_checkpoint_pin "${pin}" || status=1
    return "${status}"
}

scratch_sync_reconcile_recovery_store() {
    # Recover completed stages and update stale pointers under the run lock.
    # Reclaim partial stages only when another verified checkpoint exists.
    local layout_status=0
    SCRATCH_SYNC_RECOVERY_INCOMPLETE_ONLY=0
    scratch_sync_validate_recovery_layout || layout_status=$?
    [[ "${layout_status}" -eq 0 ]] || return "${layout_status}"

    local -a incomplete_stages=()
    local stage
    local generation
    local final
    for stage in "${SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR}"/.stage-*; do
        [[ -e "${stage}" || -L "${stage}" ]] || continue
        if [[ ! -d "${stage}" || -L "${stage}" ]]; then
            echo "Error: unsafe recovery stage requires inspection: ${stage}" >&2
            return 2
        fi
        if [[ -f "${stage}/.recovery_complete.json" &&
              ! -L "${stage}/.recovery_complete.json" ]]; then
            generation="$(scratch_sync_recovery_marker_generation \
                "${stage}/.recovery_complete.json")" || generation=""
            if [[ -n "${generation}" ]] &&
               scratch_sync_read_recovery_marker "${stage}" "${generation}" &&
               scratch_sync_validate_manifest "${stage}" &&
               scratch_sync_directory_has_exact_checkpoint_replay "${stage}" &&
               scratch_sync_directory_meets_checkpoint_high_water \
                    "${stage}" floor; then
                final="${SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR}/${generation}"
                if [[ -e "${final}" || -L "${final}" ]]; then
                    echo "Error: completed recovery stage collides with ${final}." >&2
                    return 2
                fi
                scratch_sync_fsync_tree "${stage}" || return 2
                mv -- "${stage}" "${final}" || return 2
                scratch_sync_fsync_directory \
                    "${SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR}" || return 2
                echo "Recovered completed uncommitted recovery stage ${generation}." >&2
                continue
            fi
        fi
        incomplete_stages+=("${stage}")
    done

    local current=""
    local current_pointer_invalid=0
    if [[ -f "${SCRATCH_SYNC_RECOVERY_CURRENT}" &&
          ! -L "${SCRATCH_SYNC_RECOVERY_CURRENT}" ]]; then
        current="$(scratch_sync_recovery_current_generation)" || {
            current=""
            current_pointer_invalid=1
        }
    fi

    local -a valid_names=()
    local -a valid_times=()
    local -a valid_steps=()
    local -a valid_checkpoints=()
    local -a invalid_generations=()
    local path
    local name
    local published_ns
    local checkpoint
    local checkpoint_step
    for path in "${SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR}"/snapshot-*; do
        [[ -e "${path}" || -L "${path}" ]] || continue
        name="$(basename "${path}")"
        if [[ -d "${path}" && ! -L "${path}" ]] &&
           scratch_sync_read_recovery_marker "${path}" "${name}" &&
           scratch_sync_validate_manifest "${path}" &&
           scratch_sync_directory_has_exact_checkpoint_replay "${path}"; then
            published_ns="${SCRATCH_SYNC_RECOVERY_PUBLISHED_NS}"
            checkpoint="${SCRATCH_SYNC_RECOVERY_CHECKPOINT}"
            if ! scratch_sync_checkpoint_name_is_safe "${checkpoint}"; then
                invalid_generations+=("${path}")
                continue
            fi
            checkpoint_step="$(scratch_sync_checkpoint_step \
                "${path}" "${checkpoint}")" || {
                invalid_generations+=("${path}")
                continue
            }
            valid_names+=("${name}")
            valid_times+=("${published_ns}")
            valid_steps+=("${checkpoint_step}")
            valid_checkpoints+=("${checkpoint}")
        else
            invalid_generations+=("${path}")
        fi
    done

    local best=""
    local best_time=""
    local best_step=""
    local best_checkpoint=""
    local index
    local current_valid=0
    for index in "${!valid_names[@]}"; do
        name="${valid_names[${index}]}"
        published_ns="${valid_times[${index}]}"
        checkpoint_step="${valid_steps[${index}]}"
        checkpoint="${valid_checkpoints[${index}]}"
        [[ "${name}" == "${current}" ]] && current_valid=1
        # Choose by checkpoint step; use publication time only to break ties.
        if [[ -z "${best}" ]] ||
           scratch_sync_checkpoint_point_is_newer \
                "${checkpoint_step}" "${checkpoint}" \
                "${best_step}" "${best_checkpoint}" ||
           { [[ "${checkpoint_step}" == "${best_step}" &&
                 "${checkpoint}" == "${best_checkpoint}" ]] &&
             [[ "${published_ns}" > "${best_time}" ]]; }; then
            best="${name}"
            best_time="${published_ns}"
            best_step="${checkpoint_step}"
            best_checkpoint="${checkpoint}"
        fi
    done
    if [[ "${current_pointer_invalid}" -eq 1 && -z "${best}" ]]; then
        echo "Error: invalid recovery current pointer and no verified fallback generation; preserving it for inspection." >&2
        return 2
    fi
    if [[ -n "${current}" && "${current_valid}" -eq 0 && -z "${best}" ]]; then
        echo "Error: recovery snapshot provenance does not match this run or failed integrity validation." >&2
        echo "Error: current recovery generation is missing or failed validation: ${current}" >&2
        return 2
    fi

    # Require the selected checkpoint to meet the durable progress floor before
    # reclaiming recovery state.
    if [[ -e "${SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_PATH}" ||
          -L "${SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_PATH}" ]]; then
        if [[ -n "${best}" ]]; then
            path="${SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR}/${best}"
            checkpoint="$(scratch_sync_checkpoint_latest \
                "${path}/checkpoint.pkl")" || return 2
            if ! scratch_sync_checkpoint_high_water_check \
                "${path}" "${checkpoint}" floor; then
                echo "Warning: preserving recovery transaction state because every completed snapshot is below the durable progress floor." >&2
                return 0
            fi
        elif [[ "${#incomplete_stages[@]}" -gt 0 ||
                "${#invalid_generations[@]}" -gt 0 ]]; then
            echo "Warning: preserving recovery transaction state until a checkpoint meets the durable progress floor." >&2
            return 0
        fi
    fi
    scratch_sync_cleanup_recovery_reclaims || return 2

    if [[ "${current_pointer_invalid}" -eq 1 ]]; then
        scratch_sync_quarantine_recovery_entry \
            "${SCRATCH_SYNC_RECOVERY_CURRENT}" "invalid-current-pointer" || return 2
    fi
    if [[ -n "${best}" && "${best}" != "${current}" ]]; then
        if [[ -n "${current}" && "${current_valid}" -eq 0 ]]; then
            echo "Warning: falling back from invalid current recovery generation ${current} to ${best}." >&2
        fi
        scratch_sync_publish_recovery_pointer "${best}" || return 2
        echo "Recovered unpointed durable recovery generation ${best}." >&2
        current="${best}"
    fi

    local safe_copy=0
    if [[ -n "${best}" ]] ||
       scratch_sync_directory_is_resumable_copy "${LOG_DIR:-/nonexistent}" ||
       scratch_sync_directory_is_resumable_copy "${WORK_LOG_DIR:-/nonexistent}"; then
        safe_copy=1
    fi
    if [[ "${safe_copy}" -eq 0 &&
          ( "${#incomplete_stages[@]}" -gt 0 ||
            "${#invalid_generations[@]}" -gt 0 ) ]]; then
        SCRATCH_SYNC_RECOVERY_INCOMPLETE_ONLY=1
        echo "Error: incomplete recovery data is the only remaining copy; preserving it for inspection." >&2
        return 2
    fi

    for path in "${incomplete_stages[@]}"; do
        [[ -n "${path}" ]] || continue
        scratch_sync_reclaim_recovery_entry \
            "${path}" "incomplete-stage-$(basename "${path}")" || return 2
    done
    for path in "${invalid_generations[@]}"; do
        [[ -n "${path}" ]] || continue
        [[ "$(basename "${path}")" != "${current}" ]] || return 2
        scratch_sync_quarantine_recovery_entry \
            "${path}" "invalid-generation-$(basename "${path}")" || return 2
    done
    if [[ "${#incomplete_stages[@]}" -gt 0 ||
          "${#invalid_generations[@]}" -gt 0 ]]; then
        scratch_sync_fsync_directory \
            "${SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR}" || return 2
    fi

    # Reclaim redundant generations only after verifying the current pointer.
    local removed_old=0
    for index in "${!valid_names[@]}"; do
        name="${valid_names[${index}]}"
        [[ "${name}" == "${current}" ]] && continue
        scratch_sync_reclaim_recovery_entry \
            "${SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR}/${name}" \
            "old-generation-${name}" || return 2
        removed_old=1
    done
    if [[ "${removed_old}" -eq 1 ]]; then
        scratch_sync_fsync_directory \
            "${SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR}" || return 2
    fi
    return 0
}

scratch_sync_select_recovery_snapshot() {
    # Return 0 when a completed snapshot was selected, 1 when the store is
    # absent/incomplete, and 2 when an existing store is unsafe or incompatible.
    SCRATCH_SYNC_RECOVERY_SOURCE_DIR=""
    SCRATCH_SYNC_RECOVERY_CHECKPOINT=""
    SCRATCH_SYNC_RECOVERY_PUBLISHED_NS=""
    SCRATCH_SYNC_RECOVERY_INCOMPLETE_ONLY=0

    local layout_status=0
    scratch_sync_validate_recovery_layout || layout_status=$?
    if [[ "${layout_status}" -eq 1 ]]; then
        return 1
    elif [[ "${layout_status}" -ne 0 ]]; then
        return 2
    fi
    if ! scratch_sync_reconcile_recovery_store; then
        return 2
    fi
    if [[ ! -f "${SCRATCH_SYNC_RECOVERY_CURRENT}" ||
          -L "${SCRATCH_SYNC_RECOVERY_CURRENT}" ]]; then
        echo "Warning: ignoring incomplete recovery stages with no published snapshot in ${SCRATCH_SYNC_RECOVERY_DIR}." >&2
        return 1
    fi

    local generation
    generation="$(scratch_sync_recovery_current_generation)" || {
        echo "Error: invalid recovery current pointer in ${SCRATCH_SYNC_RECOVERY_CURRENT}." >&2
        return 2
    }
    local source="${SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR}/${generation}"
    if [[ ! -d "${source}" || -L "${source}" ]]; then
        echo "Error: published recovery generation is missing or unsafe: ${source}" >&2
        return 2
    fi
    if ! scratch_sync_read_recovery_marker "${source}" "${generation}"; then
        return 2
    fi
    if ! scratch_sync_checkpoint_name_is_safe "${SCRATCH_SYNC_RECOVERY_CHECKPOINT}" ||
       [[ ! -d "${source}/checkpoint.pkl" ||
          -L "${source}/checkpoint.pkl" ]]; then
        echo "Error: recovery snapshot has no safe completed checkpoint." >&2
        return 2
    fi
    local actual_checkpoint
    actual_checkpoint="$(scratch_sync_checkpoint_latest "${source}/checkpoint.pkl")" || {
        echo "Error: recovery snapshot has no completed checkpoint." >&2
        return 2
    }
    if [[ "${actual_checkpoint}" != "${SCRATCH_SYNC_RECOVERY_CHECKPOINT}" ]]; then
        echo "Error: recovery marker/checkpoint mismatch in ${source}." >&2
        return 2
    fi
    if ! scratch_sync_validate_manifest "${source}"; then
        echo "Error: recovery snapshot provenance does not match this run." >&2
        return 2
    fi
    if ! scratch_sync_directory_has_exact_checkpoint_replay "${source}"; then
        echo "Error: recovery snapshot replay does not exactly match its checkpoint." >&2
        return 2
    fi
    SCRATCH_SYNC_RECOVERY_SOURCE_DIR="${source}"
    export SCRATCH_SYNC_RECOVERY_SOURCE_DIR
    return 0
}

scratch_sync_write_recovery_metadata() {
    local stage="$1"
    local generation="$2"
    local checkpoint="$3"
    local published_ns="$4"

    scratch_sync_integrity_marker write-recovery "${stage}" \
        ".recovery_complete.json" "${generation}" "${checkpoint}" \
        "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "${published_ns}" \
        "${RUN_UID:-unknown}"
}

scratch_sync_publish_recovery_pointer() {
    local generation="$1"
    mkdir -p "${SCRATCH_SYNC_RECOVERY_DIR}" || return 1
    python3 -c '
import os, pathlib, sys, uuid
root, generation = map(pathlib.Path, (sys.argv[1], sys.argv[2]))
path = root / "current"
temporary = root / (".current.tmp." + uuid.uuid4().hex)
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
try:
    view = memoryview((generation.name + "\n").encode())
    while view:
        view = view[os.write(descriptor, view):]
    os.fsync(descriptor)
finally:
    os.close(descriptor)
os.replace(temporary, path)
try:
    directory_descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
except OSError:
    pass
' "${SCRATCH_SYNC_RECOVERY_DIR}" "${generation}"
}

scratch_sync_validate_recovery_index() {
    local index_dir="$1"
    [[ -d "${index_dir}" && ! -L "${index_dir}" ]] || {
        echo "Error: recovery index is not a real directory: ${index_dir}" >&2
        return 1
    }
    [[ -f "${index_dir}/run_manifest.json" &&
       ! -L "${index_dir}/run_manifest.json" &&
       -f "${index_dir}/.recovery_available.json" &&
       ! -L "${index_dir}/.recovery_available.json" ]] || {
        echo "Error: recovery index is incomplete or unsafe: ${index_dir}" >&2
        return 1
    }
    scratch_sync_validate_manifest "${index_dir}" || return 1
    python3 -c '
import json, os, stat, sys
path, recovery_name, current_name, expected_uid = sys.argv[1:]
metadata = os.lstat(path)
if not stat.S_ISREG(metadata.st_mode):
    raise SystemExit(2)
with open(path, "r", encoding="utf-8") as handle:
    value = json.load(handle)
expected = {
    "schema_version": 1,
    "state": "DURABLE_RECOVERY_AVAILABLE",
    "recovery_directory": recovery_name,
    "current_pointer": current_name,
    "run_uid": expected_uid,
}
if value != expected:
    raise SystemExit(2)
' "${index_dir}/.recovery_available.json" \
      "$(basename "${SCRATCH_SYNC_RECOVERY_DIR}")" \
      "$(basename "${SCRATCH_SYNC_RECOVERY_DIR}")/current" \
      "${RUN_UID:-unknown}" || {
        echo "Error: recovery index metadata does not match this run." >&2
        return 1
    }
}

scratch_sync_publish_recovery_index() {
    local source="$1"

    # Create an index only when recovery is the sole durable copy.
    if [[ -e "${WORK_LOG_DIR}" || -L "${WORK_LOG_DIR}" ]]; then
        if [[ -f "${WORK_LOG_DIR}/.recovery_available.json" ||
              -L "${WORK_LOG_DIR}/.recovery_available.json" ]]; then
            scratch_sync_validate_recovery_index "${WORK_LOG_DIR}"
        fi
        return $?
    fi
    if [[ -e "${SCRATCH_SYNC_RECOVERY_INDEX_STAGE}" ||
          -L "${SCRATCH_SYNC_RECOVERY_INDEX_STAGE}" ]]; then
        echo "Error: incomplete recovery-index stage requires inspection: ${SCRATCH_SYNC_RECOVERY_INDEX_STAGE}" >&2
        return 1
    fi

    mkdir -p "${SCRATCH_SYNC_RECOVERY_INDEX_STAGE}" || return 1
    if ! scratch_sync_copy_file \
        "${source}/run_manifest.json" \
        "${SCRATCH_SYNC_RECOVERY_INDEX_STAGE}/run_manifest.json"; then
        rm -rf -- "${SCRATCH_SYNC_RECOVERY_INDEX_STAGE}" || true
        return 1
    fi
    python3 -c '
import json, os, pathlib, sys, uuid
directory_text, recovery_name, current_name, run_uid = sys.argv[1:]
directory = pathlib.Path(directory_text)
path = directory / ".recovery_available.json"
temporary = directory / (".recovery_available.json.tmp." + uuid.uuid4().hex)
value = {
    "schema_version": 1,
    "state": "DURABLE_RECOVERY_AVAILABLE",
    "recovery_directory": recovery_name,
    "current_pointer": current_name,
    "run_uid": run_uid,
}
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
try:
    data = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    view = memoryview(data)
    while view:
        view = view[os.write(descriptor, view):]
    os.fsync(descriptor)
finally:
    os.close(descriptor)
os.replace(temporary, path)
' "${SCRATCH_SYNC_RECOVERY_INDEX_STAGE}" \
      "$(basename "${SCRATCH_SYNC_RECOVERY_DIR}")" \
      "$(basename "${SCRATCH_SYNC_RECOVERY_DIR}")/current" \
      "${RUN_UID:-unknown}" || {
        rm -rf -- "${SCRATCH_SYNC_RECOVERY_INDEX_STAGE}" || true
        return 1
    }
    if ! scratch_sync_validate_recovery_index \
        "${SCRATCH_SYNC_RECOVERY_INDEX_STAGE}"; then
        rm -rf -- "${SCRATCH_SYNC_RECOVERY_INDEX_STAGE}" || true
        return 1
    fi
    scratch_sync_fsync_tree "${SCRATCH_SYNC_RECOVERY_INDEX_STAGE}" || return 1
    mv -- "${SCRATCH_SYNC_RECOVERY_INDEX_STAGE}" "${WORK_LOG_DIR}" || return 1
    scratch_sync_fsync_directory "$(dirname "${WORK_LOG_DIR}")"
}

scratch_sync_recover_recovery_index_stage() {
    [[ -e "${SCRATCH_SYNC_RECOVERY_INDEX_STAGE}" ||
       -L "${SCRATCH_SYNC_RECOVERY_INDEX_STAGE}" ]] || return 0
    if ! scratch_sync_validate_recovery_index \
        "${SCRATCH_SYNC_RECOVERY_INDEX_STAGE}"; then
        echo "Error: incomplete recovery-index stage requires inspection: ${SCRATCH_SYNC_RECOVERY_INDEX_STAGE}" >&2
        return 1
    fi
    if [[ -e "${WORK_LOG_DIR}" || -L "${WORK_LOG_DIR}" ]]; then
        echo "Error: both canonical run and recovery-index stage exist; refusing to merge them." >&2
        return 1
    fi
    scratch_sync_fsync_tree "${SCRATCH_SYNC_RECOVERY_INDEX_STAGE}" || return 1
    mv -- "${SCRATCH_SYNC_RECOVERY_INDEX_STAGE}" "${WORK_LOG_DIR}" || return 1
    scratch_sync_fsync_directory "$(dirname "${WORK_LOG_DIR}")"
}

scratch_sync_publish_recovery_snapshot() {
    scratch_sync_enter_storage_cwd || return 1
    [[ "${SCRATCH_SYNC_DIRECT_DURABLE:-0}" -eq 0 ]] || return 0
    [[ -d "${LOG_DIR}" && ! -L "${LOG_DIR}" ]] || return 1
    local seed_previous="${SCRATCH_SYNC_RECOVERY_SEED_PREVIOUS:-1}"
    case "${seed_previous}" in
        0|1) ;;
        *) echo "Error: SCRATCH_SYNC_RECOVERY_SEED_PREVIOUS must be 0 or 1." >&2; return 1 ;;
    esac
    if [[ "${seed_previous}" == 0 && "${SCRATCH_SYNC_PRESERVE_SOURCE:-0}" != 1 ]]; then
        echo "Error: fresh recovery stages require source-preserving maintenance mode." >&2
        return 1
    fi

    # Wait for the child source snapshot before publishing recovery from a fork.
    if [[ "${SCRATCH_SYNC_REQUIRE_SOURCE_SNAPSHOT:-0}" -eq 1 ]]; then
        local source_artifact
        for source_artifact in source_snapshot.json tracked.patch untracked.tar; do
            if [[ ! -f "${LOG_DIR}/source_provenance/${source_artifact}" ||
                  -L "${LOG_DIR}/source_provenance/${source_artifact}" ]]; then
                echo "Warning: source provenance is not ready; deferring durable recovery snapshot." >&2
                return 2
            fi
        done
    fi

    if ! scratch_sync_validate_manifest "${LOG_DIR}"; then
        echo "Error: scratch manifest validation failed; see the validator diagnostic above." >&2
        return 1
    fi

    local layout_status=0
    scratch_sync_validate_recovery_layout || layout_status=$?
    if [[ "${layout_status}" -gt 1 ]]; then
        return 1
    fi
    mkdir -p "${SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR}" || return 1
    if ! scratch_sync_validate_recovery_layout; then
        echo "Error: recovery layout changed while it was being created." >&2
        return 1
    fi
    if ! scratch_sync_reconcile_recovery_store; then
        echo "Error: recovery store could not be reconciled before publication." >&2
        return 1
    fi
    if scratch_sync_recovery_has_reclaim_residue; then
        scratch_sync_retry_recovery_reclaims_for_publication || return 1
    fi
    if scratch_sync_recovery_has_reclaim_residue; then
        echo "Error: recovery cleanup residue remains; refusing to grow the recovery store." >&2
        return 1
    fi

    # Stop publication if failed cleanup would allow unbounded storage growth.
    local recovery_generation_count=0
    local existing_generation
    for existing_generation in \
        "${SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR}"/snapshot-*; do
        [[ -d "${existing_generation}" &&
           ! -L "${existing_generation}" ]] || continue
        recovery_generation_count=$((recovery_generation_count + 1))
    done
    if [[ "${recovery_generation_count}" -gt 1 ]]; then
        echo "Error: recovery store exceeds its current-only retention bound; refusing to add another snapshot." >&2
        return 1
    fi

    # Keep the verified previous generation until its successor is committed.
    local previous_generation=""
    local previous_source=""
    local previous_checkpoint=""
    local previous_checkpoint_step=""
    if [[ -f "${SCRATCH_SYNC_RECOVERY_CURRENT}" &&
          ! -L "${SCRATCH_SYNC_RECOVERY_CURRENT}" ]]; then
        previous_generation="$(scratch_sync_recovery_current_generation)" || {
            echo "Error: refusing to advance an invalid recovery pointer." >&2
            return 1
        }
        previous_source="${SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR}/${previous_generation}"
        if [[ ! -d "${previous_source}" || -L "${previous_source}" ]] ||
           ! scratch_sync_read_recovery_marker \
                "${previous_source}" "${previous_generation}" ||
           ! scratch_sync_validate_manifest "${previous_source}"; then
            echo "Error: prior recovery generation is not known-good; refusing to replace it." >&2
            return 1
        fi
        previous_checkpoint="$(scratch_sync_checkpoint_latest \
            "${previous_source}/checkpoint.pkl")" || return 1
        if [[ "${previous_checkpoint}" != "${SCRATCH_SYNC_RECOVERY_CHECKPOINT}" ]]; then
            echo "Error: prior recovery marker/checkpoint mismatch." >&2
            return 1
        fi
        previous_checkpoint_step="$(scratch_sync_checkpoint_step \
            "${previous_source}" "${previous_checkpoint}")" || {
            echo "Error: prior certified recovery has no safe checkpoint step." >&2
            return 1
        }
    fi

    local published_ns
    local generation
    local stage
    local checkpoint
    local checkpoint_step
    local checkpoint_pin
    published_ns="$(date -u +%s%N)"
    generation="snapshot-${published_ns}-${BASHPID}-${RANDOM}"
    stage="${SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR}/.stage-${BASHPID}-${RANDOM}-${published_ns}"
    if [[ -e "${stage}" || -L "${stage}" ||
          -e "${SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR}/${generation}" ||
          -L "${SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR}/${generation}" ]]; then
        echo "Error: recovery snapshot generation collision." >&2
        return 1
    fi
    mkdir -p "${stage}" || return 1

    # Pin the selected checkpoint after validation and before copying to prevent
    # retention from reclaiming it.
    checkpoint="$(scratch_sync_checkpoint_latest \
        "${LOG_DIR}/checkpoint.pkl")" || {
        rm -rf -- "${stage}" || true
        return 2
    }
    checkpoint_step="$(scratch_sync_checkpoint_step \
        "${LOG_DIR}" "${checkpoint}")" || {
        echo "Error: live checkpoint ${checkpoint} has no safe checkpoint step." >&2
        rm -rf -- "${stage}" || true
        return 1
    }
    if ! scratch_sync_checkpoint_high_water_check \
        "${LOG_DIR}" "${checkpoint}" floor; then
        echo "Error: refusing to publish recovery below the durable progress floor." >&2
        rm -rf -- "${stage}" || true
        return 1
    fi
    if [[ -n "${previous_checkpoint_step}" ]] &&
       scratch_sync_checkpoint_point_is_newer \
            "${previous_checkpoint_step}" "${previous_checkpoint}" \
            "${checkpoint_step}" "${checkpoint}"; then
        echo "Error: refusing to replace certified recovery step ${previous_checkpoint_step} with lower checkpoint step ${checkpoint_step} (${checkpoint})." >&2
        rm -rf -- "${stage}" || true
        return 1
    fi
    checkpoint_pin="$(scratch_sync_pin_checkpoint_generation \
        "${LOG_DIR}/checkpoint.pkl" "${checkpoint}")" || {
        echo "Error: could not pin completed checkpoint ${checkpoint}; will retry." >&2
        rm -rf -- "${stage}" || true
        return 1
    }

    # Seed the stage with hardlinks so rsync copies only changed files.
    # The previous generation stays valid until the successor commits, including when
    # seeding is disabled.
    if [[ -n "${previous_source}" && "${seed_previous}" == 1 ]]; then
        if ! cp -al -- "${previous_source}/." "${stage}/"; then
            scratch_sync_release_checkpoint_pin "${checkpoint_pin}" || true
            echo "Error: could not seed incremental recovery stage; prior recovery was preserved." >&2
            rm -rf -- "${stage}" || true
            return 1
        fi
    elif [[ -n "${previous_source}" ]]; then
        echo "Maintenance recovery: copying into a fresh stage; prior generation preserved until commit."
    fi
    if ! scratch_sync_copy_log_dir \
            "${LOG_DIR}" "${stage}" "${checkpoint}" "${checkpoint_pin}" ||
       ! scratch_sync_materialize_links "${LOG_DIR}" "${stage}"; then
        scratch_sync_release_checkpoint_pin "${checkpoint_pin}" || true
        echo "Error: recovery snapshot copy failed; scratch and prior recovery were preserved." >&2
        rm -rf -- "${stage}" ||
            echo "Warning: incomplete recovery stage remains at ${stage}." >&2
        return 1
    fi
    if ! scratch_sync_release_checkpoint_pin "${checkpoint_pin}"; then
        echo "Warning: copied checkpoint pin could not be reclaimed: ${checkpoint_pin}" >&2
    fi

    # Use the checkpoint selected before replay copying so later saves cannot change its
    # replay inventory.
    local staged_checkpoint
    staged_checkpoint="$(scratch_sync_checkpoint_latest "${stage}/checkpoint.pkl")" || {
        echo "Error: recovery stage contains no completed checkpoint." >&2
        rm -rf -- "${stage}" || true
        return 1
    }
    if [[ "${staged_checkpoint}" != "${checkpoint}" ]]; then
        echo "Error: recovery stage checkpoint drifted during snapshot creation." >&2
        rm -rf -- "${stage}" || true
        return 1
    fi
    if ! scratch_sync_validate_manifest "${stage}" ||
       ! scratch_sync_directory_has_exact_checkpoint_replay "${stage}" ||
       ! scratch_sync_write_recovery_metadata \
            "${stage}" "${generation}" "${checkpoint}" "${published_ns}"; then
        echo "Error: recovery snapshot validation failed; prior recovery was preserved." >&2
        rm -rf -- "${stage}" || true
        return 1
    fi
    if ! scratch_sync_fsync_tree "${stage}"; then
        echo "Error: recovery snapshot payload could not be flushed; prior recovery was preserved." >&2
        rm -rf -- "${stage}" || true
        return 1
    fi

    local final="${SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR}/${generation}"
    if ! mv -- "${stage}" "${final}"; then
        echo "Error: could not publish recovery generation; prior recovery was preserved." >&2
        return 1
    fi
    if ! scratch_sync_fsync_directory "${SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR}"; then
        echo "Error: recovery generation directory entry could not be flushed; prior pointer remains current." >&2
        return 1
    fi
    if [[ "${SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_MODE:-synchronous}" == "published" ]]; then
        # Persist the progress floor before advancing the pointer; failed updates remain
        # available for retry.
        if ! scratch_sync_checkpoint_high_water_command commit \
            --marker "${SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_PATH}" \
            --run-dir "${final}" --checkpoint "${checkpoint}" \
            --max-gap-steps "${SCRATCH_SYNC_MAX_RESUME_GAP_STEPS:-10000}" \
            >/dev/null; then
            echo "Error: recovery generation committed, but its shared checkpoint fence could not be advanced." >&2
            return 1
        fi
    fi
    if ! scratch_sync_validate_recovery_layout; then
        echo "Error: recovery layout changed before pointer publication." >&2
        return 1
    fi
    if ! scratch_sync_publish_recovery_pointer "${generation}"; then
        echo "Error: could not advance recovery pointer; prior recovery remains current." >&2
        return 1
    fi

    # Retry index publication separately; failure does not invalidate committed
    # recovery.
    if ! scratch_sync_publish_recovery_index "${final}"; then
        echo "Warning: recovery checkpoint is durable, but its canonical-path index could not be published." >&2
    fi

    # Reclaim the previous generation after the new pointer commits.
    local old
    for old in "${SCRATCH_SYNC_RECOVERY_GENERATIONS_DIR}"/snapshot-*; do
        [[ -d "${old}" && ! -L "${old}" ]] || continue
        [[ "$(basename "${old}")" == "${generation}" ]] && continue
        scratch_sync_reclaim_recovery_entry \
            "${old}" "old-generation-$(basename "${old}")" ||
            echo "Warning: superseded recovery generation remains at ${old}." >&2
    done
    SCRATCH_SYNC_RECOVERY_SOURCE_DIR="${final}"
    SCRATCH_SYNC_RECOVERY_CHECKPOINT="${checkpoint}"
    SCRATCH_SYNC_RECOVERY_PUBLISHED_NS="${published_ns}"
    export SCRATCH_SYNC_RECOVERY_SOURCE_DIR SCRATCH_SYNC_RECOVERY_CHECKPOINT \
        SCRATCH_SYNC_RECOVERY_PUBLISHED_NS
    echo "Published durable recovery checkpoint ${checkpoint} to ${final}"
}

scratch_sync_recovery_monitor_loop() {
    local parent_pid="$1"
    local interval="$2"
    local last_checkpoint="${SCRATCH_SYNC_RECOVERY_CHECKPOINT:-}"
    local snapshot_pid=""
    local snapshot_status=0
    local timer_pid=""
    local waited
    local retry_delay="${interval}"
    local next_retry=0

    trap - EXIT
    scratch_sync_enter_storage_cwd || return 1
    trap '
        [[ -n "${timer_pid}" ]] && kill "${timer_pid}" 2>/dev/null || true
        if [[ -n "${snapshot_pid}" ]] && kill -0 "${snapshot_pid}" 2>/dev/null; then
            scratch_sync_signal_process_tree "${snapshot_pid}" TERM
            wait "${snapshot_pid}" 2>/dev/null || true
        fi
        exit 0
    ' TERM INT HUP
    while kill -0 "${parent_pid}" 2>/dev/null; do
        local checkpoint=""
        checkpoint="$(scratch_sync_checkpoint_latest "${LOG_DIR}/checkpoint.pkl" 2>/dev/null)" || checkpoint=""
        if [[ -n "${checkpoint}" && "${checkpoint}" != "${last_checkpoint}" &&
              "${SECONDS}" -ge "${next_retry}" ]]; then
            # Run copies as children so signal handlers can stop them before final
            # publication.
            scratch_sync_publish_recovery_snapshot &
            snapshot_pid=$!
            if wait "${snapshot_pid}"; then
                last_checkpoint="${checkpoint}"
                retry_delay="${interval}"
                next_retry=0
            else
                snapshot_status=$?
                if [[ "${snapshot_status}" -eq 2 ]]; then
                    # Retry deferred checkpoints on the next poll.
                    echo "Durable recovery snapshot deferred for checkpoint ${checkpoint}; will retry." >&2
                else
                    # Retry failed copies with backoff while retaining the verified
                    # previous generation.
                    next_retry=$((SECONDS + retry_delay))
                    echo "Error: durable recovery snapshot failed for checkpoint ${checkpoint}; retrying in ${retry_delay}s." >&2
                    retry_delay=$((retry_delay * 2))
                    [[ "${retry_delay}" -le 900 ]] || retry_delay=900
                fi
            fi
            snapshot_pid=""
        fi
        # Poll in one-second intervals so deferred Bash traps can stop the monitor
        # promptly.
        waited=0
        while [[ "${waited}" -lt "${interval}" ]]; do
            sleep 1 &
            timer_pid=$!
            wait "${timer_pid}" 2>/dev/null || true
            timer_pid=""
            kill -0 "${parent_pid}" 2>/dev/null || exit 0
            waited=$((waited + 1))
        done
    done
}

scratch_sync_signal_process_tree() {
    # Signal copier descendants before their parent to prevent races with final
    # publication.
    local process_pid="$1"
    local signal_name="${2:-TERM}"
    local child
    local children=""

    [[ "${process_pid}" =~ ^[0-9]+$ ]] || return 1
    if [[ -r "/proc/${process_pid}/task/${process_pid}/children" ]]; then
        children="$(<"/proc/${process_pid}/task/${process_pid}/children")"
        for child in ${children}; do
            scratch_sync_signal_process_tree "${child}" "${signal_name}" || true
        done
    fi
    kill -s "${signal_name}" "${process_pid}" 2>/dev/null || true
}

scratch_sync_start_recovery_monitor() {
    [[ "${SCRATCH_SYNC_DIRECT_DURABLE:-0}" -eq 0 ]] || return 0
    local interval="${SCRATCH_SYNC_RECOVERY_INTERVAL_SECONDS:-60}"
    local owner_pid="${BASHPID}"
    if [[ ! "${interval}" =~ ^[0-9]+$ ]] ||
       [[ "${interval}" -lt 1 || "${interval}" -gt 86400 ]]; then
        echo "Error: SCRATCH_SYNC_RECOVERY_INTERVAL_SECONDS must be 1..86400." >&2
        return 1
    fi
    if [[ -n "${SCRATCH_SYNC_MONITOR_PID:-}" ]] &&
       kill -0 "${SCRATCH_SYNC_MONITOR_PID}" 2>/dev/null; then
        echo "Error: recovery monitor is already running." >&2
        return 1
    fi
    scratch_sync_recovery_monitor_loop "${owner_pid}" "${interval}" &
    SCRATCH_SYNC_MONITOR_PID=$!
    export SCRATCH_SYNC_MONITOR_PID
}

scratch_sync_stop_recovery_monitor() {
    local monitor_pid="${SCRATCH_SYNC_MONITOR_PID:-}"
    [[ -n "${monitor_pid}" ]] || return 0
    if kill -0 "${monitor_pid}" 2>/dev/null; then
        kill -TERM "${monitor_pid}" 2>/dev/null || true
    fi
    wait "${monitor_pid}" 2>/dev/null || true
    unset SCRATCH_SYNC_MONITOR_PID
}

scratch_sync_cleanup_recovery_after_publish() {
    [[ -e "${SCRATCH_SYNC_RECOVERY_DIR}" ||
       -L "${SCRATCH_SYNC_RECOVERY_DIR}" ]] || return 0
    if [[ ! -d "${SCRATCH_SYNC_RECOVERY_DIR}" ||
          -L "${SCRATCH_SYNC_RECOVERY_DIR}" ]]; then
        echo "Error: refusing to remove unsafe recovery path ${SCRATCH_SYNC_RECOVERY_DIR}." >&2
        return 1
    fi
    rm -rf -- "${SCRATCH_SYNC_RECOVERY_DIR}"
}

scratch_sync_cleanup_source() {
    [[ "${SCRATCH_SYNC_DIRECT_DURABLE:-0}" -eq 0 ]] || return 0

    # The preservation option skips scratch cleanup after a verified publication.
    if [[ "${SCRATCH_SYNC_PRESERVE_SOURCE:-0}" == "1" ]]; then
        echo "Preserved scratch source after durable publish: ${LOG_DIR}"
        return 0
    fi

    if [[ -d "${LOG_DIR}" ]]; then
        if [[ -e "${SCRATCH_SYNC_TOMBSTONE}" || -L "${SCRATCH_SYNC_TOMBSTONE}" ]]; then
            echo "Error: scratch cleanup tombstone already exists: ${SCRATCH_SYNC_TOMBSTONE}" >&2
            return 1
        fi
        mv -- "${LOG_DIR}" "${SCRATCH_SYNC_TOMBSTONE}" || return 1
    fi
    if [[ -e "${SCRATCH_SYNC_TOMBSTONE}" || -L "${SCRATCH_SYNC_TOMBSTONE}" ]]; then
        rm -rf -- "${SCRATCH_SYNC_TOMBSTONE}" || return 1
    fi
}

scratch_sync_prepare_preempt_requeue() {
    local child_status="${1:-0}"
    local event_status=0
    local recovery_status=1

    scratch_sync_enter_storage_cwd || return 1
    scratch_sync_ensure_runtime_paths || return 1
    scratch_sync_stop_recovery_monitor || {
        echo "Error: failed to stop durable recovery monitor before requeue." >&2
        return 1
    }
    scratch_sync_record_finish "${child_status}" || event_status=$?
    if [[ "${event_status}" -ne 0 ]]; then
        return "${event_status}"
    fi

    # Direct-durable runs do not need a data transfer. The regular publisher
    # still writes and verifies their completion envelope before requeue.
    if [[ "${SCRATCH_SYNC_DIRECT_DURABLE:-0}" -eq 1 ]]; then
        scratch_sync_move_logs || return $?
        scratch_sync_copy_scheduler_logs_to \
            "${WORK_LOG_DIR:-${LOG_DIR:-}}" || return $?
        SCRATCH_SYNC_PREEMPT_PREPARED=1
        export SCRATCH_SYNC_PREEMPT_PREPARED
        scratch_sync_release_lock
        return 0
    fi

    # Prefer the monitor's already-committed generation. It is deliberately
    # bounded and independently checksummed, unlike the full mutable run tree.
    if scratch_sync_select_recovery_snapshot; then
        if scratch_sync_directory_meets_checkpoint_high_water \
            "${SCRATCH_SYNC_RECOVERY_SOURCE_DIR}" floor; then
            recovery_status=0
        else
            recovery_status=2
        fi
    else
        recovery_status=$?
    fi
    if [[ "${recovery_status}" -gt 1 ]]; then
        echo "Error: durable recovery storage is present but unsafe; refusing requeue." >&2
        return 1
    fi

    if [[ "${recovery_status}" -ne 0 ]]; then
        if scratch_sync_checkpoint_is_valid "${LOG_DIR}/checkpoint.pkl"; then
            echo "[Slurm] No prior recovery generation; publishing the completed checkpoint before requeue."
            scratch_sync_publish_recovery_snapshot || return $?
            scratch_sync_select_recovery_snapshot || {
                echo "Error: the preemption checkpoint was not durably certified." >&2
                return 1
            }
        else
            # Verify preemptions that occur before the first checkpoint before allowing
            # requeue.
            echo "[Slurm] No checkpoint exists yet; certifying the pre-checkpoint run state."
            scratch_sync_move_logs || return $?
            if ! scratch_sync_is_certified_empty_preempt "${WORK_LOG_DIR}"; then
                echo "Error: refusing to requeue uncertified checkpoint-less state." >&2
                return 1
            fi
            scratch_sync_copy_scheduler_logs_to "${WORK_LOG_DIR}" || return $?
            SCRATCH_SYNC_PREEMPT_PREPARED=1
            export SCRATCH_SYNC_PREEMPT_PREPARED
            scratch_sync_release_lock
            return 0
        fi
    fi

    # Hide scratch atomically before releasing the lock so restart selects verified
    # recovery.
    # Defer deletion until after the preemption deadline.
    if [[ -e "${SCRATCH_SYNC_TOMBSTONE}" ||
          -L "${SCRATCH_SYNC_TOMBSTONE}" ]]; then
        echo "Error: scratch preemption tombstone already exists: ${SCRATCH_SYNC_TOMBSTONE}" >&2
        return 1
    fi
    if [[ -d "${LOG_DIR}" && ! -L "${LOG_DIR}" ]]; then
        mv -- "${LOG_DIR}" "${SCRATCH_SYNC_TOMBSTONE}" || return 1
        scratch_sync_fsync_directory "$(dirname "${LOG_DIR}")" || return 1
    else
        echo "Error: live scratch run disappeared before preemption commit." >&2
        return 1
    fi

    # Keep the lock until the durable pointer is revalidated.
    SCRATCH_SYNC_PREEMPT_PREPARED=1
    export SCRATCH_SYNC_PREEMPT_PREPARED

    # Revalidate committed recovery before releasing the run lock; avoid new copies
    # during preemption.
    scratch_sync_select_recovery_snapshot &&
    scratch_sync_directory_meets_checkpoint_high_water \
        "${SCRATCH_SYNC_RECOVERY_SOURCE_DIR}" floor || {
        echo "Error: durable recovery changed before preemption commit." >&2
        return 1
    }
    SCRATCH_SYNC_RESUME_SOURCE_DIR="${SCRATCH_SYNC_RECOVERY_SOURCE_DIR}"
    export SCRATCH_SYNC_RESUME_SOURCE_DIR
    scratch_sync_release_lock
    echo "[Slurm] Certified recovery is ready; deferred scratch cleanup to the replacement allocation."
}

scratch_sync_move_logs() {
    scratch_sync_enter_storage_cwd || return 1
    scratch_sync_ensure_runtime_paths || return 1
    # Stop the monitor before final publication, including direct calls outside the EXIT
    # trap.
    scratch_sync_stop_recovery_monitor || return 1
    if [[ "${SCRATCH_SYNC_COMMITTED:-0}" -eq 1 ]]; then
        scratch_sync_cleanup_recovery_after_publish || return 1
        scratch_sync_cleanup_source
        return $?
    fi
    if [[ "${SCRATCH_SYNC_DONE:-0}" -eq 1 ]]; then
        return 0
    fi
    if [[ -z "${LOG_DIR:-}" || ! -d "${LOG_DIR}" ]]; then
        return 0
    fi
    if [[ "${SCRATCH_SYNC_MANAGED_PIPELINE:-0}" -eq 1 ]]; then
       if [[ ! -f "${LOG_DIR}/run_manifest.json" ||
              -L "${LOG_DIR}/run_manifest.json" ]] ||
           ! scratch_sync_validate_manifest "${LOG_DIR}"; then
            echo "Error: final publish manifest validation failed; see the validator diagnostic above." >&2
            return 1
        fi
    fi
    if ! scratch_sync_directory_meets_checkpoint_high_water \
        "${LOG_DIR}" high; then
        echo "Error: refusing final publish below the recorded high-water checkpoint." >&2
        return 1
    fi
    scratch_sync_stage_scheduler_logs || return 1
    if [[ "${SCRATCH_SYNC_DIRECT_DURABLE:-0}" -eq 1 ]]; then
        if ! scratch_sync_compact_durable_checkpoint_state "${LOG_DIR}" ||
           ! scratch_sync_write_complete_marker "${LOG_DIR}" ||
           ! scratch_sync_verify_integrity_marker \
                "${LOG_DIR}" ".sync_complete.json"; then
            echo "Error: direct durable completion marker failed integrity validation." >&2
            return 1
        fi
        if ! scratch_sync_fsync_tree "${LOG_DIR}" ||
           ! scratch_sync_fsync_directory "$(dirname "${LOG_DIR}")"; then
            echo "Error: direct durable run could not be flushed to storage." >&2
            return 1
        fi
        scratch_sync_cleanup_recovery_after_publish || return 1
        SCRATCH_SYNC_COMMITTED=1
        SCRATCH_SYNC_DONE=1
        echo "Run data is already on durable storage: ${LOG_DIR}"
        return 0
    fi

    scratch_sync_recover_publish_transaction || return 1

    if [[ -e "${SCRATCH_SYNC_STAGE_DIR}" || -L "${SCRATCH_SYNC_STAGE_DIR}" ]]; then
        echo "Error: refusing to overwrite incomplete publish stage ${SCRATCH_SYNC_STAGE_DIR}." >&2
        return 1
    fi
    mkdir -p "${SCRATCH_SYNC_STAGE_DIR}" || return 1
    if ! scratch_sync_copy_log_dir "${LOG_DIR}" "${SCRATCH_SYNC_STAGE_DIR}"; then
        echo "Error: staged durable copy failed; canonical run was not changed." >&2
        rm -rf -- "${SCRATCH_SYNC_STAGE_DIR}" || \
            echo "Warning: could not remove incomplete publish stage ${SCRATCH_SYNC_STAGE_DIR}." >&2
        return 1
    fi
    if ! scratch_sync_materialize_links "${LOG_DIR}" "${SCRATCH_SYNC_STAGE_DIR}"; then
        rm -rf -- "${SCRATCH_SYNC_STAGE_DIR}" || \
            echo "Warning: could not remove incomplete publish stage ${SCRATCH_SYNC_STAGE_DIR}." >&2
        return 1
    fi
    if ! scratch_sync_write_complete_marker "${SCRATCH_SYNC_STAGE_DIR}"; then
        rm -rf -- "${SCRATCH_SYNC_STAGE_DIR}" || \
            echo "Warning: could not remove incomplete publish stage ${SCRATCH_SYNC_STAGE_DIR}." >&2
        return 1
    fi
    if ! scratch_sync_verify_integrity_marker \
        "${SCRATCH_SYNC_STAGE_DIR}" ".sync_complete.json"; then
        echo "Error: staged durable copy failed critical artifact verification." >&2
        rm -rf -- "${SCRATCH_SYNC_STAGE_DIR}" || \
            echo "Warning: could not remove invalid publish stage ${SCRATCH_SYNC_STAGE_DIR}." >&2
        return 1
    fi
    if ! scratch_sync_fsync_tree "${SCRATCH_SYNC_STAGE_DIR}"; then
        echo "Error: staged durable payload could not be flushed; source data was preserved." >&2
        rm -rf -- "${SCRATCH_SYNC_STAGE_DIR}" || true
        return 1
    fi

    rm -rf -- "${SCRATCH_SYNC_PREVIOUS_DIR}" || return 1
    if [[ -e "${WORK_LOG_DIR}" || -L "${WORK_LOG_DIR}" ]]; then
        mv -- "${WORK_LOG_DIR}" "${SCRATCH_SYNC_PREVIOUS_DIR}" || return 1
    fi
    if ! mv -- "${SCRATCH_SYNC_STAGE_DIR}" "${WORK_LOG_DIR}"; then
        if [[ -d "${SCRATCH_SYNC_PREVIOUS_DIR}" ]]; then
            mv -- "${SCRATCH_SYNC_PREVIOUS_DIR}" "${WORK_LOG_DIR}" || true
            scratch_sync_fsync_directory "$(dirname "${WORK_LOG_DIR}")" ||
                echo "Warning: durable rollback directory entry could not be flushed." >&2
        fi
        return 1
    fi
    if ! scratch_sync_fsync_directory "$(dirname "${WORK_LOG_DIR}")"; then
        echo "Error: durable commit directory entry could not be flushed; scratch was preserved." >&2
        return 1
    fi

    # Commit is final before scratch deletion. A cleanup retry must never
    # republish a partially deleted source over this complete durable snapshot.
    SCRATCH_SYNC_COMMITTED=1
    SCRATCH_SYNC_DONE=1
    if ! scratch_sync_cleanup_recovery_after_publish; then
        echo "Error: durable publish succeeded but recovery cleanup failed; source data was preserved." >&2
        return 1
    fi
    if ! scratch_sync_cleanup_source; then
        echo "Error: durable publish succeeded but scratch cleanup failed; only cleanup will be retried." >&2
        return 1
    fi
    rm -rf -- "${SCRATCH_SYNC_PREVIOUS_DIR}" || \
        echo "Warning: previous durable generation remains at ${SCRATCH_SYNC_PREVIOUS_DIR}." >&2
    echo "Published logs transactionally from ${LOG_DIR} to ${WORK_LOG_DIR}"
}

scratch_sync_manifest_command() {
    local helper="${RUN_STORAGE_HELPER:-${HELPER_DIR:-${SCRATCH_SYNC_HELPER_DIR}}/run_storage.py}"
    [[ -f "${helper}" ]] || {
        echo "Error: missing run manifest helper ${helper}." >&2
        return 1
    }
    python3 "${helper}" "$@"
}

scratch_sync_initialize_manifest() {
    local run_dir="$1"
    local mode="$2"

    mkdir -p "${run_dir}" || return 1
    if [[ -n "${SSRL_RUN_REPO_ROOT:-}" ]]; then
        scratch_sync_manifest_command manifest init --run-dir "${run_dir}" \
            --mode "${mode}" --repo "${SSRL_RUN_REPO_ROOT}"
    elif [[ -n "${RUN_SOURCE_ID:-}" ]]; then
        scratch_sync_manifest_command manifest init --run-dir "${run_dir}" \
            --mode "${mode}" --source-id "${RUN_SOURCE_ID}"
    else
        echo "Error: manifest initialization requires SSRL_RUN_REPO_ROOT or RUN_SOURCE_ID." >&2
        return 1
    fi
}

scratch_sync_prepare_manifest() {
    scratch_sync_initialize_manifest "${LOG_DIR}" "$1"
}

scratch_sync_validate_manifest() {
    local run_dir="$1"

    if [[ -n "${SSRL_RUN_REPO_ROOT:-}" ]]; then
        scratch_sync_manifest_command manifest validate --run-dir "${run_dir}" \
            --repo "${SSRL_RUN_REPO_ROOT}"
    elif [[ -n "${RUN_SOURCE_ID:-}" ]]; then
        scratch_sync_manifest_command manifest validate --run-dir "${run_dir}" \
            --source-id "${RUN_SOURCE_ID}"
    else
        echo "Error: manifest validation requires SSRL_RUN_REPO_ROOT or RUN_SOURCE_ID." >&2
        return 1
    fi
}

scratch_sync_read_fork_marker() {
    local marker="$1"
    local expected_uid="$2"
    python3 - "${marker}" "${expected_uid}" <<'PY'
import json
import os
import re
import stat
import sys

path, expected_uid = sys.argv[1:]
metadata = os.lstat(path)
if not stat.S_ISREG(metadata.st_mode):
    raise SystemExit(2)
with open(path, "r", encoding="utf-8") as handle:
    value = json.load(handle)
kind = value.get("marker_kind")
if value.get("schema_version") != 2 or kind not in {
    "final-publish", "periodic-recovery"
}:
    raise SystemExit(2)
if value.get("run_uid") != expected_uid:
    raise SystemExit(2)
checkpoint = value.get("checkpoint")
if not isinstance(checkpoint, str) or not re.fullmatch(
    r"[A-Za-z0-9][A-Za-z0-9._-]*", checkpoint
):
    raise SystemExit(2)
timestamp = value.get(
    "completed_ns" if kind == "final-publish" else "published_ns"
)
if not isinstance(timestamp, str) or not re.fullmatch(r"[0-9]{19}", timestamp):
    raise SystemExit(2)
print(timestamp, checkpoint, kind, sep="\t")
PY
}

scratch_sync_select_fork_source() {
    local requested="${RUN_FORK_SOURCE_DIR:-}"
    local source_root="${RUN_FORK_SOURCE_ROOT:-${WORK_LOG_ROOT}}"
    local parent_uid="${RUN_FORK_SOURCE_RUN_UID:-}"
    local parent_source_id="${RUN_FORK_PARENT_SOURCE_ID:-}"
    local lexical_source
    local canonical_source
    local source_parent
    local source_base
    local recovery
    local current
    local generation
    local candidate
    local marker
    local values
    local timestamp
    local checkpoint
    local kind
    local checkpoint_step
    local parent_high_water
    local selected=""
    local selected_marker=""
    local selected_checkpoint=""
    local selected_timestamp="0"
    local selected_step=""

    if [[ -z "${requested}" || -z "${parent_uid}" ||
          ! "${parent_uid}" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$ ||
          ! "${parent_source_id}" =~ ^[0-9a-f]{64}$ ]]; then
        echo "Error: checkpoint fork requires a safe source path, parent run UID, and parent source digest." >&2
        return 1
    fi
    lexical_source="$(scratch_sync_lexical_path "${requested}")" || return 1
    local lexical_root
    lexical_root="$(scratch_sync_lexical_path "${source_root}")" || return 1
    if [[ "${source_root}" != /* || ! -d "${lexical_root}" ||
          "$(scratch_sync_canonical_path "${lexical_root}")" != "${lexical_root}" ]]; then
        echo "Error: checkpoint fork source root must be a real absolute directory: ${source_root}" >&2
        return 1
    fi
    if ! scratch_sync_is_strict_child "${lexical_source}" "${lexical_root}"; then
        echo "Error: checkpoint fork source is outside its approved root: ${requested}" >&2
        return 1
    fi
    scratch_sync_reject_symlink_components \
        "${lexical_root}" "${lexical_source}" || return 1
    canonical_source="$(scratch_sync_canonical_path "${lexical_source}")" || return 1
    if [[ "${canonical_source}" != "${lexical_source}" ||
          "${canonical_source}" == "${WORK_LOG_DIR}" ]]; then
        echo "Error: checkpoint fork source resolved unsafely: ${requested}" >&2
        return 1
    fi
    source_parent="$(dirname "${canonical_source}")"
    source_base="$(basename "${canonical_source}")"
    recovery="${source_parent}/.${source_base}.recovery"
    parent_high_water="${recovery}/checkpoint-high-water.json"

    if [[ -e "${canonical_source}" || -L "${canonical_source}" ]]; then
        if [[ ! -d "${canonical_source}" || -L "${canonical_source}" ]]; then
            echo "Error: checkpoint fork canonical source is unsafe: ${canonical_source}" >&2
            return 1
        fi
        marker="${canonical_source}/.sync_complete.json"
        if [[ -e "${marker}" || -L "${marker}" ]]; then
            if [[ ! -f "${marker}" || -L "${marker}" ]] ||
               ! scratch_sync_verify_integrity_marker \
                    "${canonical_source}" ".sync_complete.json"; then
                echo "Error: checkpoint fork canonical source failed completion integrity." >&2
                return 1
            fi
            if values="$(scratch_sync_read_fork_marker \
                "${marker}" "${parent_uid}")"; then
                IFS=$'\t' read -r timestamp checkpoint kind <<< "${values}"
                checkpoint_step="$(scratch_sync_checkpoint_step \
                    "${canonical_source}" "${checkpoint}")" || checkpoint_step=""
                if [[ -n "${checkpoint_step}" ]] &&
                   scratch_sync_checkpoint_high_water_check \
                    "${canonical_source}" "${checkpoint}" floor \
                    "${parent_high_water}" "${parent_uid}"; then
                    selected="${canonical_source}"
                    selected_marker=".sync_complete.json"
                    selected_checkpoint="${checkpoint}"
                    selected_timestamp="${timestamp}"
                    selected_step="${checkpoint_step}"
                fi
            elif scratch_sync_is_certified_empty_preempt \
                    "${canonical_source}"; then
                # A published preemption without a checkpoint must not hide a newer
                # recovery checkpoint.
                echo "Ignoring certified checkpoint-less canonical fork source; checking periodic recovery." >&2
            else
                echo "Error: checkpoint fork canonical marker has no valid checkpoint." >&2
                return 1
            fi
        fi
    fi

    if [[ -e "${recovery}" || -L "${recovery}" ]]; then
        if [[ ! -d "${recovery}" || -L "${recovery}" ||
              ! -d "${recovery}/generations" ||
              -L "${recovery}/generations" ||
              ! -f "${recovery}/current" ||
              -L "${recovery}/current" ]]; then
            echo "Error: checkpoint fork recovery store is incomplete or unsafe: ${recovery}" >&2
            return 1
        fi
        current="$(python3 -c '
import os, re, stat, sys
path = sys.argv[1]
metadata = os.lstat(path)
if not stat.S_ISREG(metadata.st_mode):
    raise SystemExit(2)
with open(path, "r", encoding="ascii", newline="") as handle:
    value = handle.read()
match = re.fullmatch(r"(snapshot-[0-9]{19}-[0-9]+-[0-9]+)\n?", value)
if not match:
    raise SystemExit(2)
print(match.group(1))
' "${recovery}/current")" || return 1
        local pointed_generation_seen=0
        for candidate in "${recovery}/generations"/snapshot-*; do
            [[ -e "${candidate}" || -L "${candidate}" ]] || continue
            generation="${candidate}"
            [[ "$(basename "${generation}")" == "${current}" ]] &&
                pointed_generation_seen=1
            marker="${generation}/.recovery_complete.json"
            if [[ ! -d "${generation}" || -L "${generation}" ||
                  ! -f "${marker}" || -L "${marker}" ]] ||
               ! scratch_sync_verify_integrity_marker \
                    "${generation}" ".recovery_complete.json"; then
                if [[ "$(basename "${generation}")" == "${current}" ]]; then
                    echo "Error: checkpoint fork current recovery generation failed integrity: ${generation}" >&2
                    return 1
                fi
                echo "Warning: ignoring invalid unpointed fork recovery generation ${generation}." >&2
                continue
            fi
            values="$(scratch_sync_read_fork_marker \
                "${marker}" "${parent_uid}")" || {
                [[ "$(basename "${generation}")" == "${current}" ]] && return 1
                continue
            }
            IFS=$'\t' read -r timestamp checkpoint kind <<< "${values}"
            checkpoint_step="$(scratch_sync_checkpoint_step \
                "${generation}" "${checkpoint}")" || {
                [[ "$(basename "${generation}")" == "${current}" ]] && return 1
                continue
            }
            if scratch_sync_checkpoint_high_water_check \
               "${generation}" "${checkpoint}" floor \
                "${parent_high_water}" "${parent_uid}" &&
               { [[ -z "${selected_step}" ]] ||
                 scratch_sync_checkpoint_point_is_newer \
                    "${checkpoint_step}" "${checkpoint}" \
                    "${selected_step}" "${selected_checkpoint}" ||
                 { [[ "${checkpoint_step}" == "${selected_step}" &&
                       "${checkpoint}" == "${selected_checkpoint}" ]] &&
                   [[ "${timestamp}" -gt "${selected_timestamp}" ]]; }; }; then
                selected="${generation}"
                selected_marker=".recovery_complete.json"
                selected_checkpoint="${checkpoint}"
                selected_timestamp="${timestamp}"
                selected_step="${checkpoint_step}"
            fi
        done
        if [[ "${pointed_generation_seen}" -ne 1 ]]; then
            echo "Error: checkpoint fork recovery pointer names a missing generation." >&2
            return 1
        fi
    fi

    if [[ -z "${selected}" ]]; then
        echo "Error: checkpoint fork source has no certified completed artifact: ${canonical_source}" >&2
        return 1
    fi
    if ! scratch_sync_manifest_command manifest fork-validate \
        --run-dir "${selected}" \
        --parent-run-id "${parent_uid}" \
        --parent-source-id "${parent_source_id}" \
        --parent-marker "${selected_marker}" >/dev/null; then
        echo "Error: checkpoint fork parent is not compatible with this experiment." >&2
        return 1
    fi

    local previous_mode="${RUN_STORAGE_MODE_RESOLVED:-fork}"
    local previous_source="${SCRATCH_SYNC_RESUME_SOURCE_DIR:-}"
    RUN_STORAGE_MODE_RESOLVED=resume
    SCRATCH_SYNC_RESUME_SOURCE_DIR="${selected}"
    if ! scratch_sync_validate_resume_checkpoint "${selected}"; then
        RUN_STORAGE_MODE_RESOLVED="${previous_mode}"
        SCRATCH_SYNC_RESUME_SOURCE_DIR="${previous_source}"
        return 1
    fi
    RUN_STORAGE_MODE_RESOLVED="${previous_mode}"
    SCRATCH_SYNC_RESUME_SOURCE_DIR="${previous_source}"

    RUN_FORK_SELECTED_SOURCE="${selected}"
    RUN_FORK_SELECTED_MARKER="${selected_marker}"
    RUN_FORK_SELECTED_CHECKPOINT="${selected_checkpoint}"
    export RUN_FORK_SELECTED_SOURCE RUN_FORK_SELECTED_MARKER \
        RUN_FORK_SELECTED_CHECKPOINT
}

scratch_sync_fork_logs() {
    local source="${RUN_FORK_SELECTED_SOURCE:?Missing selected fork source}"
    local marker="${RUN_FORK_SELECTED_MARKER:?Missing selected fork marker}"
    local stage="${LOG_DIR}.fork-new"
    local -a manifest_command

    if [[ -e "${LOG_DIR}" || -L "${LOG_DIR}" ]]; then
        echo "Error: checkpoint fork destination already exists: ${LOG_DIR}" >&2
        return 1
    fi
    if [[ -e "${stage}" || -L "${stage}" ]]; then
        if [[ ! -d "${stage}" || -L "${stage}" ]]; then
            echo "Error: stale checkpoint fork stage is unsafe: ${stage}" >&2
            return 1
        fi
        rm -rf -- "${stage}" || return 1
    fi
    mkdir -p "$(dirname "${LOG_DIR}")" || return 1
    if ! scratch_sync_copy_dir "${source}" "${stage}"; then
        rm -rf -- "${stage}" || true
        return 1
    fi
    if ! scratch_sync_verify_integrity_marker "${stage}" "${marker}" ||
       ! scratch_sync_materialize_links "${source}" "${stage}" ||
       ! scratch_sync_verify_integrity_marker "${stage}" "${marker}"; then
        echo "Error: copied checkpoint fork source failed integrity verification." >&2
        rm -rf -- "${stage}" || true
        return 1
    fi

    manifest_command=(
        manifest fork --run-dir "${stage}"
        --parent-run-id "${RUN_FORK_SOURCE_RUN_UID}"
        --parent-source-id "${RUN_FORK_PARENT_SOURCE_ID}"
        --parent-marker "${marker}"
        --parent-artifact "${source}"
    )
    if [[ -n "${SSRL_RUN_REPO_ROOT:-}" ]]; then
        manifest_command+=(--repo "${SSRL_RUN_REPO_ROOT}")
    else
        manifest_command+=(--source-id "${RUN_SOURCE_ID}")
    fi
    if ! scratch_sync_manifest_command "${manifest_command[@]}" >/dev/null ||
       ! scratch_sync_validate_manifest "${stage}"; then
        echo "Error: checkpoint fork manifest creation failed." >&2
        rm -rf -- "${stage}" || true
        return 1
    fi
    SCRATCH_SYNC_CERTIFIED_FORK=1
    if ! scratch_sync_validate_resume_checkpoint "${stage}"; then
        rm -rf -- "${stage}" || true
        return 1
    fi
    mv -- "${stage}" "${LOG_DIR}" || return 1
    SCRATCH_SYNC_RESUME_SOURCE_DIR="${LOG_DIR}"
    SCRATCH_SYNC_FORKED=1
    export SCRATCH_SYNC_CERTIFIED_FORK SCRATCH_SYNC_RESUME_SOURCE_DIR \
        SCRATCH_SYNC_FORKED
    echo "Forked checkpoint ${RUN_FORK_SELECTED_CHECKPOINT} from ${source} to ${LOG_DIR}"
}

scratch_sync_record_finish() {
    local status="$1"
    local state="COMPLETED"
    local run_dir="${LOG_DIR:-}"

    [[ "${SCRATCH_SYNC_FINISH_RECORDED:-0}" -eq 0 ]] || return 0
    if [[ "${_SLURM_SIGNAL_NAME:-}" == "USR1" ]]; then
        state="PREEMPTED"
    elif [[ -n "${_SLURM_SIGNAL_NAME:-}" ]]; then
        state="CANCELED"
    elif [[ "${status}" -ne 0 ]]; then
        state="FAILED"
    fi
    if [[ -n "${run_dir}" && -f "${run_dir}/run_manifest.json" ]]; then
        scratch_sync_manifest_command event append --run-dir "${run_dir}" \
            --event terminal --status "${state}" \
            --field "exit_status=${status}" \
            --field "signal=${_SLURM_SIGNAL_NAME:-}" || return 1
        SCRATCH_SYNC_FINISH_RECORDED=1
    fi
}

scratch_sync_retry_publication() {
    local operation="$1"
    local attempts="${SCRATCH_SYNC_PUBLISH_MAX_ATTEMPTS:-3}"
    local delay="${SCRATCH_SYNC_PUBLISH_RETRY_SECONDS:-15}"
    local attempt status=1

    case "${operation}" in
        scratch_sync_move_logs|scratch_sync_publish_recovery_snapshot) ;;
        *) echo "Error: unsupported publication operation ${operation}." >&2; return 1 ;;
    esac
    if [[ ! "${attempts}" =~ ^[0-9]+$ || "${attempts}" -lt 1 || "${attempts}" -gt 10 ||
          ! "${delay}" =~ ^[0-9]+$ || "${delay}" -gt 60 ]]; then
        echo "Error: publication retries require 1..10 attempts and a 0..60 second initial delay." >&2
        return 1
    fi
    for ((attempt=1; attempt<=attempts; attempt++)); do
        if "${operation}"; then
            return 0
        else
            status=$?
        fi
        [[ "${attempt}" -lt "${attempts}" ]] || break
        echo "Warning: ${operation} failed (attempt ${attempt}/${attempts}); retrying in ${delay}s with source data preserved." >&2
        sleep "${delay}" || return "${status}"
        delay=$((delay * 2))
        [[ "${delay}" -le 60 ]] || delay=60
    done
    echo "Error: ${operation} exhausted ${attempts} attempts; source data was preserved." >&2
    return "${status}"
}

scratch_sync_on_exit() {
    local status="${1:-0}"
    local sync_status=0
    local event_status=0

    trap - EXIT
    scratch_sync_enter_storage_cwd || return 1
    if [[ "${SCRATCH_SYNC_PREEMPT_PREPARED:-0}" -eq 1 ||
          "${SCRATCH_SYNC_PREEMPT_NO_FINAL_PUBLISH:-0}" -eq 1 ]]; then
        # Preserve scratch if bounded preemption recovery fails; do not start a full
        # copy.
        scratch_sync_stop_recovery_monitor ||
            echo "Warning: recovery monitor did not stop cleanly during USR1 exit." >&2
        scratch_sync_release_lock
        return "${status}"
    fi
    scratch_sync_stop_recovery_monitor || sync_status=$?
    if [[ "${sync_status}" -ne 0 ]]; then
        echo "Error: failed to stop durable recovery monitor; refusing a racing final publish." >&2
    else
        # Record terminal state before staging so publication authenticates it with the
        # run results.
        scratch_sync_record_finish "${status}" || event_status=$?
        if scratch_sync_retry_publication scratch_sync_move_logs; then
            sync_status=0
            # Scheduler output can still grow while this EXIT trap reports its
            # own cleanup progress, so it is refreshed outside the immutable
            # result envelope while the run lock remains held.
            scratch_sync_copy_scheduler_logs_to "${WORK_LOG_DIR:-${LOG_DIR:-}}" || \
                event_status=$?
        else
            sync_status=$?
            echo "Error: failed to publish ${LOG_DIR:-run logs} to ${WORK_LOG_DIR:-durable storage}; source data was preserved." >&2
        fi
    fi
    scratch_sync_release_lock

    if [[ "${status}" -ne 0 ]]; then
        return "${status}"
    fi
    if [[ "${sync_status}" -ne 0 ]]; then
        return "${sync_status}"
    fi
    return "${event_status}"
}

enable_scratch_log_sync() {
    local mode="${RUN_STORAGE_MODE_RESOLVED:-new}"
    local resume_attempt_recorded=0
    local scratch_owned_by_enable=0

    scratch_sync_require_commands || return 1
    scratch_sync_configure_paths || return 1
    SCRATCH_SYNC_DONE=0
    SCRATCH_SYNC_COMMITTED=0
    SCRATCH_SYNC_FINISH_RECORDED=0
    SCRATCH_SYNC_PREEMPT_PREPARED=0
    SCRATCH_SYNC_PREEMPT_NO_FINAL_PUBLISH=0
    # Internal authorization for the one safe checkpoint-less restart path.
    # Always reset inherited/user-provided values before inspecting storage.
    SCRATCH_SYNC_EMPTY_PREEMPT_RESTART=0
    SCRATCH_SYNC_CERTIFIED_FORK=0
    SCRATCH_SYNC_FORKED=0
    SCRATCH_SYNC_RECOVERY_INCOMPLETE_ONLY=0
    SCRATCH_SYNC_REPLAY_FALLBACK_TOMBSTONE=0

    # Dry runs must validate naming/mapping but never create locks, restore data,
    # or install a transfer trap.
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        return 0
    fi

    scratch_sync_acquire_lock || return 1
    if ! scratch_sync_recover_publish_transaction; then
        scratch_sync_release_lock
        return 1
    fi
    if ! scratch_sync_recover_cleanup_tombstone "${mode}"; then
        scratch_sync_release_lock
        return 1
    fi
    if ! scratch_sync_repair_first_checkpoint_latest "${LOG_DIR}"; then
        scratch_sync_release_lock
        return 1
    fi
    if ! scratch_sync_recover_recovery_index_stage; then
        scratch_sync_release_lock
        return 1
    fi

    SCRATCH_SYNC_RESUME_SOURCE_DIR=""
    local recovery_status=1
    if [[ "${mode}" == "resume" || "${mode}" == "fork" ]]; then
        if scratch_sync_select_recovery_snapshot; then
            recovery_status=0
        else
            recovery_status=$?
        fi
        if [[ "${recovery_status}" -gt 1 ]]; then
            if [[ "${mode}" == "fork" &&
                  "${SCRATCH_SYNC_RECOVERY_INCOMPLETE_ONLY}" -eq 1 ]]; then
                echo "Warning: no certified child checkpoint exists; preserving the incomplete fork attempt and retrying from its certified parent." >&2
                if ! scratch_sync_preserve_incomplete_fork_attempt; then
                    scratch_sync_release_lock
                    return 1
                fi
                recovery_status=1
            else
                scratch_sync_release_lock
                return 1
            fi
        fi
    fi

    case "${mode}" in
        new)
            if [[ -e "${LOG_DIR}" || -L "${LOG_DIR}" ||
                  -e "${WORK_LOG_DIR}" || -L "${WORK_LOG_DIR}" ||
                  -e "${SCRATCH_SYNC_RECOVERY_DIR}" ||
                  -L "${SCRATCH_SYNC_RECOVERY_DIR}" ]]; then
                echo "Error: fresh run destination already exists; refusing to overwrite or auto-resume." >&2
                echo "  LOG_DIR=${LOG_DIR}" >&2
                echo "  WORK_LOG_DIR=${WORK_LOG_DIR}" >&2
                echo "  RECOVERY_DIR=${SCRATCH_SYNC_RECOVERY_DIR}" >&2
                scratch_sync_release_lock
                return 1
            fi
            ;;
        resume)
            if ! scratch_sync_dir_has_entries "${WORK_LOG_DIR}" &&
                ! scratch_sync_dir_has_entries "${LOG_DIR}" &&
                [[ -z "${SCRATCH_SYNC_RECOVERY_SOURCE_DIR}" ]]; then
                # Allow first initialization after requeue only if every possible
                # storage location is absent or empty.
                if [[ "${RUN_SLURM_RESTART_COUNT:-0}" =~ ^[1-9][0-9]*$ &&
                      ! -e "${SCRATCH_SYNC_RECOVERY_DIR}" &&
                      ! -L "${SCRATCH_SYNC_RECOVERY_DIR}" &&
                      ( ( ! -e "${WORK_LOG_DIR}" && ! -L "${WORK_LOG_DIR}" ) ||
                        ( -d "${WORK_LOG_DIR}" && ! -L "${WORK_LOG_DIR}" ) ) &&
                      ( ( ! -e "${LOG_DIR}" && ! -L "${LOG_DIR}" ) ||
                        ( -d "${LOG_DIR}" && ! -L "${LOG_DIR}" ) ) ]]; then
                    echo "Slurm requeue has no persisted run data; initializing the logical run for the first time." >&2
                    mode="new"
                    RUN_STORAGE_MODE_RESOLVED="new"
                    export RUN_STORAGE_MODE_RESOLVED
                else
                    echo "Error: resume requested but no prior run data exists." >&2
                    scratch_sync_release_lock
                    return 1
                fi
            fi
            ;;
        fork)
            if scratch_sync_dir_has_entries "${WORK_LOG_DIR}" ||
               scratch_sync_dir_has_entries "${LOG_DIR}" ||
               [[ -n "${SCRATCH_SYNC_RECOVERY_SOURCE_DIR}" ]]; then
                mode="resume"
                RUN_STORAGE_MODE_RESOLVED="resume"
            else
                if ! scratch_sync_select_fork_source ||
                   ! scratch_sync_fork_logs; then
                    scratch_sync_release_lock
                    return 1
                fi
                mode="resume"
                RUN_STORAGE_MODE_RESOLVED="resume"
                resume_attempt_recorded=1
                scratch_owned_by_enable=1
            fi
            export RUN_STORAGE_MODE_RESOLVED
            ;;
        *)
            echo "Error: RUN_STORAGE_MODE_RESOLVED must be new, resume, or fork, got ${mode}." >&2
            scratch_sync_release_lock
            return 1
            ;;
    esac

    if [[ "${mode}" == "resume" ]]; then
        local work_has_state=0
        local scratch_has_state=0
        scratch_sync_dir_has_entries "${WORK_LOG_DIR}" && work_has_state=1
        scratch_sync_dir_has_entries "${LOG_DIR}" && scratch_has_state=1

        if [[ "${work_has_state}" -eq 1 &&
              ( -f "${WORK_LOG_DIR}/.recovery_available.json" ||
                -L "${WORK_LOG_DIR}/.recovery_available.json" ) ]]; then
            if ! scratch_sync_validate_recovery_index "${WORK_LOG_DIR}"; then
                scratch_sync_release_lock
                return 1
            fi
        fi

        # Quarantine a corrupt canonical copy only when independently verified recovery
        # has the same run identity.
        if [[ "${work_has_state}" -eq 1 &&
              -n "${SCRATCH_SYNC_RECOVERY_SOURCE_DIR}" &&
              ! -f "${WORK_LOG_DIR}/.recovery_available.json" ]]; then
            if [[ ! -f "${WORK_LOG_DIR}/run_manifest.json" ||
                  -L "${WORK_LOG_DIR}/run_manifest.json" ]] ||
               ! scratch_sync_validate_manifest "${WORK_LOG_DIR}"; then
                scratch_sync_release_lock
                return 1
            fi
            local canonical_integrity_ok=0
            if [[ -f "${WORK_LOG_DIR}/.sync_complete.json" &&
                  ! -L "${WORK_LOG_DIR}/.sync_complete.json" ]] &&
               scratch_sync_verify_integrity_marker \
                    "${WORK_LOG_DIR}" ".sync_complete.json"; then
                canonical_integrity_ok=1
            fi
            if [[ "${canonical_integrity_ok}" -eq 0 ]]; then
                echo "Warning: canonical run failed completion integrity; falling back to verified recovery." >&2
                if ! scratch_sync_archive_residual \
                    "${WORK_LOG_DIR}" "corrupt-canonical"; then
                    scratch_sync_release_lock
                    return 1
                fi
                work_has_state=0
            fi
        fi

        # Validate scratch and durable copies independently after interruption; never
        # merge their trees.
        if [[ "${SCRATCH_SYNC_DIRECT_DURABLE}" -eq 0 &&
              "${work_has_state}" -eq 1 && "${scratch_has_state}" -eq 1 &&
              ( ! -f "${LOG_DIR}/run_manifest.json" ||
                -L "${LOG_DIR}/run_manifest.json" ) ]]; then
            echo "Error: refusing to merge residual scratch data without a safe manifest: ${LOG_DIR}" >&2
            scratch_sync_release_lock
            return 1
        fi

        if [[ "${scratch_has_state}" -eq 1 &&
              "${SCRATCH_SYNC_DIRECT_DURABLE}" -eq 0 ]]; then
            if [[ -n "${SCRATCH_SYNC_RECOVERY_SOURCE_DIR}" ]] &&
               scratch_sync_recovery_is_newer_than_canonical "${LOG_DIR}"; then
                echo "Verified recovery is ahead of residual scratch; preserving the lower-progress scratch tree." >&2
                if ! scratch_sync_archive_residual \
                    "${LOG_DIR}" "lower-progress-scratch"; then
                    scratch_sync_release_lock
                    return 1
                fi
                scratch_has_state=0
                SCRATCH_SYNC_RESUME_SOURCE_DIR="${SCRATCH_SYNC_RECOVERY_SOURCE_DIR}"
            else
                SCRATCH_SYNC_RESUME_SOURCE_DIR="${LOG_DIR}"
            fi
        elif [[ -n "${SCRATCH_SYNC_RECOVERY_SOURCE_DIR}" ]]; then
            if [[ "${work_has_state}" -eq 0 ]] ||
               scratch_sync_recovery_is_newer_than_canonical "${WORK_LOG_DIR}"; then
                SCRATCH_SYNC_RESUME_SOURCE_DIR="${SCRATCH_SYNC_RECOVERY_SOURCE_DIR}"
            else
                SCRATCH_SYNC_RESUME_SOURCE_DIR="${WORK_LOG_DIR}"
            fi
        else
            SCRATCH_SYNC_RESUME_SOURCE_DIR="${WORK_LOG_DIR}"
        fi

        # Select the most advanced checkpoint across all verified copies, regardless of
        # filesystem recency.
        if [[ "${SCRATCH_SYNC_DIRECT_DURABLE}" -eq 0 &&
              "${work_has_state}" -eq 1 &&
              "${SCRATCH_SYNC_RESUME_SOURCE_DIR}" != "${WORK_LOG_DIR}" &&
              -f "${WORK_LOG_DIR}/.sync_complete.json" &&
              ! -L "${WORK_LOG_DIR}/.sync_complete.json" ]]; then
            if ! scratch_sync_validate_manifest "${WORK_LOG_DIR}" ||
               ! scratch_sync_verify_integrity_marker \
                    "${WORK_LOG_DIR}" ".sync_complete.json"; then
                echo "Error: canonical continuation candidate is not safely certified." >&2
                scratch_sync_release_lock
                return 1
            fi
            # A verified preemption without a checkpoint ranks below any completed
            # recovery checkpoint.
            if ! scratch_sync_is_certified_empty_preempt "${WORK_LOG_DIR}"; then
                if ! scratch_sync_directory_has_exact_checkpoint_replay \
                        "${WORK_LOG_DIR}" ||
                   ! scratch_sync_directory_meets_checkpoint_high_water \
                        "${WORK_LOG_DIR}" floor; then
                    echo "Error: canonical continuation candidate is not safely certified." >&2
                    scratch_sync_release_lock
                    return 1
                fi
                local canonical_checkpoint
                local canonical_step
                local selected_checkpoint=""
                local selected_step=""
                local selected_is_certified_empty=0
                canonical_checkpoint="$(scratch_sync_checkpoint_latest \
                    "${WORK_LOG_DIR}/checkpoint.pkl")" || {
                    scratch_sync_release_lock
                    return 1
                }
                canonical_step="$(scratch_sync_checkpoint_step \
                    "${WORK_LOG_DIR}" "${canonical_checkpoint}")" || {
                    scratch_sync_release_lock
                    return 1
                }
                if [[ -d "${SCRATCH_SYNC_RESUME_SOURCE_DIR}/checkpoint.pkl" &&
                      ! -L "${SCRATCH_SYNC_RESUME_SOURCE_DIR}/checkpoint.pkl" ]]; then
                    selected_checkpoint="$(scratch_sync_checkpoint_latest \
                        "${SCRATCH_SYNC_RESUME_SOURCE_DIR}/checkpoint.pkl")" || {
                        scratch_sync_release_lock
                        return 1
                    }
                    selected_step="$(scratch_sync_checkpoint_step \
                        "${SCRATCH_SYNC_RESUME_SOURCE_DIR}" \
                        "${selected_checkpoint}")" || {
                        scratch_sync_release_lock
                        return 1
                    }
                fi
                if scratch_sync_is_certified_empty_preempt \
                    "${SCRATCH_SYNC_RESUME_SOURCE_DIR}"; then
                    selected_is_certified_empty=1
                fi
                if [[ "${selected_is_certified_empty}" -eq 1 ]] ||
                   { [[ -n "${selected_step}" ]] &&
                     scratch_sync_checkpoint_point_is_newer \
                        "${canonical_step}" "${canonical_checkpoint}" \
                        "${selected_step}" "${selected_checkpoint}"; }; then
                    if scratch_sync_dir_has_entries "${LOG_DIR}"; then
                        echo "Certified canonical checkpoint is ahead of residual scratch; preserving the lower-progress scratch tree." >&2
                        if ! scratch_sync_archive_residual \
                            "${LOG_DIR}" "lower-progress-scratch"; then
                            scratch_sync_release_lock
                            return 1
                        fi
                        scratch_has_state=0
                    fi
                    SCRATCH_SYNC_RESUME_SOURCE_DIR="${WORK_LOG_DIR}"
                fi
            fi
        fi

        # Replace interrupted scratch replay only when its run identity matches
        # independently verified recovery.
        if [[ "${SCRATCH_SYNC_DIRECT_DURABLE}" -eq 0 &&
              "${SCRATCH_SYNC_RESUME_SOURCE_DIR}" == "${LOG_DIR}" &&
              -n "${SCRATCH_SYNC_RECOVERY_SOURCE_DIR}" &&
              -d "${LOG_DIR}/checkpoint.pkl" &&
              ! -L "${LOG_DIR}/checkpoint.pkl" ]] &&
           scratch_sync_checkpoint_is_valid "${LOG_DIR}/checkpoint.pkl" &&
           scratch_sync_validate_manifest "${LOG_DIR}" &&
           ! scratch_sync_checkpoint_replay_inventory \
                verify "${LOG_DIR}" "${LOG_DIR}/checkpoint.pkl"; then
            if [[ -e "${SCRATCH_SYNC_TOMBSTONE}" ||
                  -L "${SCRATCH_SYNC_TOMBSTONE}" ]]; then
                echo "Error: scratch replay fallback tombstone already exists: ${SCRATCH_SYNC_TOMBSTONE}" >&2
                scratch_sync_release_lock
                return 1
            fi
            mv -- "${LOG_DIR}" "${SCRATCH_SYNC_TOMBSTONE}" || {
                scratch_sync_release_lock
                return 1
            }
            if ! scratch_sync_fsync_directory "$(dirname "${LOG_DIR}")"; then
                scratch_sync_release_lock
                return 1
            fi
            SCRATCH_SYNC_REPLAY_FALLBACK_TOMBSTONE=1
            SCRATCH_SYNC_RESUME_SOURCE_DIR="${SCRATCH_SYNC_RECOVERY_SOURCE_DIR}"
            scratch_has_state=0
            export SCRATCH_SYNC_REPLAY_FALLBACK_TOMBSTONE \
                SCRATCH_SYNC_RESUME_SOURCE_DIR
            echo "Mutable scratch replay failed checkpoint binding; falling back to exact durable recovery." >&2
        fi

        # Validate the canonical run identity even when a newer recovery checkpoint
        # exists.
        if [[ "${work_has_state}" -eq 1 &&
              "${SCRATCH_SYNC_RESUME_SOURCE_DIR}" != "${WORK_LOG_DIR}" ]]; then
            if [[ ! -f "${WORK_LOG_DIR}/run_manifest.json" ||
                  -L "${WORK_LOG_DIR}/run_manifest.json" ]]; then
                echo "Error: canonical run beside recovery has no safe manifest." >&2
                scratch_sync_release_lock
                return 1
            fi
            if ! scratch_sync_validate_manifest "${WORK_LOG_DIR}"; then
                scratch_sync_release_lock
                return 1
            fi
        fi

        # Reject non-resumable artifacts before copying or adopting anything.
        if ! scratch_sync_validate_resume_checkpoint \
            "${SCRATCH_SYNC_RESUME_SOURCE_DIR}"; then
            scratch_sync_release_lock
            return 1
        fi
        # Validate immutable identity before copying any durable bytes to scratch.
        local validate_dir="${SCRATCH_SYNC_RESUME_SOURCE_DIR}"
        if [[ -f "${validate_dir}/run_manifest.json" ||
              -L "${validate_dir}/run_manifest.json" ]]; then
            if ! scratch_sync_validate_manifest "${validate_dir}"; then
                scratch_sync_release_lock
                return 1
            fi
        else
            # The helper refuses this unless RUN_ALLOW_LEGACY_RESUME is an
            # explicit true value; on approval it adopts the legacy identity
            # atomically and records this attempt before any restore.
            if ! scratch_sync_initialize_manifest "${validate_dir}" resume; then
                scratch_sync_release_lock
                return 1
            fi
            resume_attempt_recorded=1
        fi
        if [[ -f "${validate_dir}/.sync_complete.json" ||
              -L "${validate_dir}/.sync_complete.json" ]]; then
            if ! scratch_sync_verify_integrity_marker \
                "${validate_dir}" ".sync_complete.json"; then
                echo "Error: refusing to resume a durable run with failed critical artifact integrity." >&2
                scratch_sync_release_lock
                return 1
            fi
        fi
        if ! scratch_sync_directory_meets_checkpoint_high_water \
            "${validate_dir}" floor; then
            scratch_sync_release_lock
            return 1
        fi
        if [[ -d "${validate_dir}/checkpoint.pkl" &&
              ! -L "${validate_dir}/checkpoint.pkl" ]]; then
            local resume_checkpoint
            resume_checkpoint="$(scratch_sync_checkpoint_latest \
                "${validate_dir}/checkpoint.pkl")" || {
                scratch_sync_release_lock
                return 1
            }
            if [[ "${SCRATCH_SYNC_DIRECT_DURABLE}" -eq 0 &&
                  "${SCRATCH_SYNC_CHECKPOINT_HIGH_WATER_MODE:-synchronous}" != "published" ]] &&
               ! scratch_sync_checkpoint_high_water_seed \
                    "${validate_dir}" "${resume_checkpoint}"; then
                scratch_sync_release_lock
                return 1
            fi
        fi
        if [[ "${validate_dir}" != "${LOG_DIR}" ]]; then
            if ! scratch_sync_restore_logs; then
                scratch_sync_release_lock
                return 1
            fi
            [[ "${SCRATCH_SYNC_DIRECT_DURABLE}" -eq 1 ]] || scratch_owned_by_enable=1
        fi

        # Remove the verified prior publication marker before recording a new attempt in
        # the mutable run tree.
        if [[ -e "${LOG_DIR}/.sync_complete.json" ||
              -L "${LOG_DIR}/.sync_complete.json" ]]; then
            if [[ ! -f "${LOG_DIR}/.sync_complete.json" ||
                  -L "${LOG_DIR}/.sync_complete.json" ]]; then
                echo "Error: restored completion marker is unsafe: ${LOG_DIR}/.sync_complete.json" >&2
                scratch_sync_release_lock
                return 1
            fi
            rm -f -- "${LOG_DIR}/.sync_complete.json" || {
                scratch_sync_release_lock
                return 1
            }
        fi
    fi

    if [[ "${mode}" == "new" && "${SCRATCH_SYNC_DIRECT_DURABLE}" -eq 0 ]]; then
        scratch_owned_by_enable=1
    fi
    if ! scratch_sync_cleanup_stale_checkpoint_pins; then
        scratch_sync_release_lock
        return 1
    fi
    if [[ "${mode}" == "new" ]] ||
       [[ "${resume_attempt_recorded}" -eq 0 ]]; then
      if ! scratch_sync_prepare_manifest "${mode}"; then
        if [[ "${scratch_owned_by_enable}" -eq 1 ]]; then
            rm -rf -- "${LOG_DIR}" || true
        fi
        scratch_sync_release_lock
        return 1
      fi
    fi
    if [[ "${mode}" == "resume" ]]; then
        if ! scratch_sync_validate_resume_checkpoint "${LOG_DIR}"; then
            if [[ "${scratch_owned_by_enable}" -eq 1 ]]; then
                rm -rf -- "${LOG_DIR}" || true
            fi
            scratch_sync_release_lock
            return 1
        fi
    fi
    if [[ "${SCRATCH_SYNC_REPLAY_FALLBACK_TOMBSTONE:-0}" -eq 1 ]]; then
        if [[ ! -d "${SCRATCH_SYNC_TOMBSTONE}" ||
              -L "${SCRATCH_SYNC_TOMBSTONE}" ]]; then
            echo "Error: scratch replay fallback tombstone changed before cleanup." >&2
            scratch_sync_release_lock
            return 1
        fi
        rm -rf -- "${SCRATCH_SYNC_TOMBSTONE}" || {
            scratch_sync_release_lock
            return 1
        }
        scratch_sync_fsync_directory "$(dirname "${LOG_DIR}")" || {
            scratch_sync_release_lock
            return 1
        }
        SCRATCH_SYNC_REPLAY_FALLBACK_TOMBSTONE=0
        export SCRATCH_SYNC_REPLAY_FALLBACK_TOMBSTONE
    fi
    SCRATCH_SYNC_MANAGED_PIPELINE=1
    if ! scratch_sync_start_recovery_monitor; then
        if [[ "${scratch_owned_by_enable}" -eq 1 ]]; then
            rm -rf -- "${LOG_DIR}" || true
        fi
        scratch_sync_release_lock
        return 1
    fi
    trap 'status=$?; scratch_sync_on_exit "${status}"; exit $?' EXIT
}
