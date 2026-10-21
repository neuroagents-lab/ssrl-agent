#!/usr/bin/env python3
"""Small, dependency-free helpers for durable experiment storage.

The module intentionally does not know about a particular training framework.  It
provides narrowly scoped facilities that shell launchers can use before
moving a run between scratch and durable storage:

* deterministic source-tree identities;
* immutable, offline-verifiable snapshots and runnable source bundles;
* immutable run manifests and append-only attempt records;
* append-only lifecycle events plus an atomically replaced status summary; and
* safe materialization of symlinks that would otherwise break after scratch is
  removed, plus a read-only catalog of canonical and recovery runs.

All JSON written by this module is UTF-8 and sorted to keep it inspectable and
stable across invocations.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import errno
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import pickletools
import re
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import time
import uuid
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
MANIFEST_NAME = "run_manifest.json"
ATTEMPTS_NAME = "run_attempts.jsonl"
WANDB_RECEIPT_NAME = "wandb_run.json"
EVENTS_NAME = "run_events.jsonl"
STATUS_NAME = "run_status.json"
CRITICAL_INTEGRITY_SCOPE_V1 = "critical-run-state-v1"
CRITICAL_INTEGRITY_SCOPE_V2 = "critical-run-state-v2"
CRITICAL_INTEGRITY_SCOPE_CURRENT = CRITICAL_INTEGRITY_SCOPE_V2
CRITICAL_INTEGRITY_SCOPES = frozenset(
    {CRITICAL_INTEGRITY_SCOPE_V1, CRITICAL_INTEGRITY_SCOPE_V2}
)
SOURCE_SNAPSHOT_SCHEMA_VERSION = 1
SOURCE_SNAPSHOT_METADATA_NAME = "source_snapshot.json"
SOURCE_SNAPSHOT_DIFF_NAME = "tracked.patch"
SOURCE_SNAPSHOT_UNTRACKED_NAME = "untracked.tar"
SOURCE_SNAPSHOT_FORMAT = "ssrl-source-snapshot-v1"
SOURCE_BUNDLE_SCHEMA_VERSION = 1
SOURCE_BUNDLE_METADATA_NAME = "source_bundle.json"
SOURCE_BUNDLE_PROVENANCE_DIR = "source_provenance"
SOURCE_BUNDLE_TREE_DIR = "tree"
SOURCE_BUNDLE_FORMAT = "ssrl-runnable-source-bundle-v1"
CHECKPOINT_HIGH_WATER_NAME = "checkpoint-high-water.json"
CHECKPOINT_HIGH_WATER_SCHEMA_VERSION = 1
CHECKPOINT_HIGH_WATER_KIND = "checkpoint-high-water"
CHECKPOINT_HIGH_WATER_MAX_BYTES = 16 * 1024

# Bound source snapshots to detect accidentally included training artifacts.
# Git-ignored files are excluded.
MAX_SOURCE_UNTRACKED_ENTRIES = 50_000
MAX_SOURCE_UNTRACKED_FILE_BYTES = 128 * 1024 * 1024
MAX_SOURCE_UNTRACKED_TOTAL_BYTES = 256 * 1024 * 1024

# Allowlist run identity fields to exclude credentials and unrelated scheduler state.
IDENTITY_ENV_ALLOWLIST = (
    "RUN_ID",
    "RUN_SET_ID",
    "RUN_FAMILY",
    "RUN_NAME",
    "RUN_EXPERIMENT",
    "RUN_SEED",
    "RUN_ENV_CONFIG",
    "RUN_TASK",
    "RUN_ALGORITHM",
    "RUN_IDM",
    "RUN_AGENT",
    "RUN_MODEL",
    "RUN_STRICT_TASK_BLIND_PRETRAINING",
    "RUN_WANDB_ID",
    "RUN_WANDB_PROJECT",
    "RUN_CONFIG_ID",
    "RUN_LAUNCHER_SHA256",
    "RUN_RUNTIME_SHA256",
)

# A fork changes run and source identity while preserving experiment coordinates.
# Its W&B project may differ from the recorded parent project.
FORK_STABLE_IDENTITY_FIELDS = (
    "RUN_FAMILY",
    "RUN_EXPERIMENT",
    "RUN_SEED",
    "RUN_ENV_CONFIG",
    "RUN_TASK",
    "RUN_ALGORITHM",
    "RUN_IDM",
    "RUN_AGENT",
    "RUN_MODEL",
    "RUN_STRICT_TASK_BLIND_PRETRAINING",
)

ATTEMPT_ENV_ALLOWLIST = (
    "RUN_ATTEMPT_ID",
    "RUN_WANDB_ATTEMPT_ID",
    "RUN_SLURM_JOB_ID",
    "RUN_SLURM_ARRAY_JOB_ID",
    "RUN_SLURM_ARRAY_TASK_ID",
    "RUN_SLURM_RESTART_COUNT",
    "RUN_HOSTNAME",
)

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_WANDB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_FIELD_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_CHECKPOINT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class StorageError(RuntimeError):
    """An expected safety or validation failure."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _canonical_json_bytes(value: Any, *, pretty: bool = True) -> bytes:
    if pretty:
        text = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False)
    else:
        text = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
    return (text + "\n").encode("utf-8")


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            # Some network filesystems do not implement directory fsync.  File
            # fsync plus atomic rename/link is still the strongest operation
            # available there.
            if exc.errno not in (errno.EINVAL, errno.ENOTSUP):
                raise
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise StorageError("short write while persisting storage metadata")
        view = view[written:]


def _atomic_write_bytes(
    path: Path, data: bytes, *, mode: int = 0o640, exclusive: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, mode)
    try:
        _write_all(descriptor, data)
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)

    try:
        if exclusive:
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError as exc:
                raise StorageError(f"refusing to replace existing file: {path}") from exc
            temporary.unlink()
        else:
            os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(
    path: Path, value: Any, *, mode: int = 0o640, exclusive: bool = False
) -> None:
    _atomic_write_bytes(
        path, _canonical_json_bytes(value), mode=mode, exclusive=exclusive
    )


def _stage_write_bytes(path: Path, data: bytes, *, mode: int = 0o640) -> None:
    """Write a new file directly inside an unpublished staging directory.

    The containing directory is the atomic publication boundary, so a sibling
    temporary file adds no durability or visibility guarantee here. Avoiding
    one also prevents NFS from exposing an unlinked, externally opened
    temporary as a transient ``.nfs*`` entry that fails strict inventory
    verification.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        _write_all(descriptor, data)
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _stage_write_json(path: Path, value: Any, *, mode: int = 0o640) -> None:
    _stage_write_bytes(path, _canonical_json_bytes(value), mode=mode)


def _read_json_file(path: Path) -> Any:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise StorageError(f"metadata path is not a regular file: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StorageError(f"cannot read valid JSON from {path}: {exc}") from exc


def _append_json_line(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_json_bytes(value, pretty=False)
    flags = os.O_RDWR | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o640)
    except OSError as exc:
        raise StorageError(f"cannot open append-only metadata {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise StorageError(f"append-only metadata is not a regular file: {path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _truncate_torn_jsonl_tail(descriptor, path)
        _write_all(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _truncate_torn_jsonl_tail(descriptor: int, path: Path) -> bool:
    """Drop only a non-newline-terminated tail from a locked JSONL file."""

    size = os.fstat(descriptor).st_size
    if not size:
        return False
    os.lseek(descriptor, size - 1, os.SEEK_SET)
    if os.read(descriptor, 1) == b"\n":
        os.lseek(descriptor, 0, os.SEEK_END)
        return False

    position = size
    boundary = 0
    while position:
        chunk_start = max(0, position - 64 * 1024)
        os.lseek(descriptor, chunk_start, os.SEEK_SET)
        chunk = os.read(descriptor, position - chunk_start)
        newline = chunk.rfind(b"\n")
        if newline >= 0:
            boundary = chunk_start + newline + 1
            break
        position = chunk_start
    os.ftruncate(descriptor, boundary)
    os.fsync(descriptor)
    os.lseek(descriptor, 0, os.SEEK_END)
    return True


def _repair_jsonl_tail(path: Path) -> bool:
    flags = os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise StorageError(f"append-only metadata is not a regular file: {path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return _truncate_torn_jsonl_tail(descriptor, path)
    finally:
        os.close(descriptor)


def _git(repo: Path, arguments: Sequence[str], *, check: bool = True) -> bytes:
    environment = dict(os.environ)
    environment.update(
        {"LC_ALL": "C", "LANG": "C", "GIT_OPTIONAL_LOCKS": "0"}
    )
    process = subprocess.run(
        ["git", "-C", os.fspath(repo), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        check=False,
    )
    if check and process.returncode:
        message = process.stderr.decode("utf-8", "replace").strip()
        raise StorageError(
            f"git {' '.join(arguments)} failed in {repo}: {message or process.returncode}"
        )
    return process.stdout


def _hash_part(digest: "hashlib._Hash", label: bytes, value: bytes) -> None:
    digest.update(len(label).to_bytes(8, "big"))
    digest.update(label)
    digest.update(len(value).to_bytes(16, "big"))
    digest.update(value)


def _validated_git_relative_path(raw_path: bytes) -> tuple[bytes, ...]:
    """Validate a path emitted by Git before using it below another tree."""

    if not raw_path or b"\0" in raw_path or raw_path.startswith(b"/"):
        raise StorageError(f"Git emitted an unsafe source path: {raw_path!r}")
    components = tuple(raw_path.split(b"/"))
    if any(component in {b"", b".", b".."} for component in components):
        raise StorageError(f"Git emitted an unsafe source path: {raw_path!r}")
    return components


def _capture_untracked_entry(top_level: Path, raw_path: bytes) -> dict[str, Any]:
    """Capture one untracked path without following any symlink component."""

    components = _validated_git_relative_path(raw_path)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    descriptor = os.open(top_level, directory_flags)
    try:
        for component in components[:-1]:
            next_descriptor = os.open(
                component, directory_flags, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = next_descriptor

        name = components[-1]
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISREG(metadata.st_mode):
            if metadata.st_size > MAX_SOURCE_UNTRACKED_FILE_BYTES:
                raise StorageError(
                    "untracked source file exceeds the provenance size limit "
                    f"({MAX_SOURCE_UNTRACKED_FILE_BYTES} bytes): "
                    f"{os.fsdecode(raw_path)}"
                )
            file_flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                file_flags |= os.O_NOFOLLOW
            file_descriptor = os.open(name, file_flags, dir_fd=descriptor)
            try:
                opened = os.fstat(file_descriptor)
                if not stat.S_ISREG(opened.st_mode):
                    raise StorageError(
                        f"untracked source path changed type while reading: "
                        f"{os.fsdecode(raw_path)}"
                    )
                chunks: list[bytes] = []
                captured_bytes = 0
                while True:
                    chunk = os.read(file_descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    captured_bytes += len(chunk)
                    if captured_bytes > MAX_SOURCE_UNTRACKED_FILE_BYTES:
                        raise StorageError(
                            "untracked source file grew beyond the provenance "
                            f"size limit while reading: {os.fsdecode(raw_path)}"
                        )
                    chunks.append(chunk)
                finished = os.fstat(file_descriptor)
                if (
                    opened.st_dev != metadata.st_dev
                    or opened.st_ino != metadata.st_ino
                    or finished.st_size != opened.st_size
                    or finished.st_mtime_ns != opened.st_mtime_ns
                ):
                    raise StorageError(
                        f"untracked source file changed while reading: "
                        f"{os.fsdecode(raw_path)}"
                    )
            finally:
                os.close(file_descriptor)
            return {
                "path": raw_path,
                "mode": mode,
                "type": "file",
                "contents": b"".join(chunks),
            }
        if stat.S_ISLNK(metadata.st_mode):
            target = os.readlink(name, dir_fd=descriptor)
            target_bytes = target if isinstance(target, bytes) else os.fsencode(target)
            finished = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if (
                finished.st_dev != metadata.st_dev
                or finished.st_ino != metadata.st_ino
                or not stat.S_ISLNK(finished.st_mode)
            ):
                raise StorageError(
                    f"untracked source symlink changed while reading: "
                    f"{os.fsdecode(raw_path)}"
                )
            return {
                "path": raw_path,
                "mode": mode,
                "type": "symlink",
                "target": target_bytes,
            }
        raise StorageError(
            "unsupported untracked source type (not file/symlink): "
            f"{os.fsdecode(raw_path)}"
        )
    except OSError as exc:
        raise StorageError(
            f"untracked source path changed while reading "
            f"{os.fsdecode(raw_path)}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)


def _capture_effective_source(
    repo: os.PathLike[str] | str,
) -> tuple[Path, bytes, bytes, list[dict[str, Any]]]:
    """Capture every byte that participates in a source identity."""

    requested = Path(repo).expanduser()
    try:
        top_level_text = _git(requested, ["rev-parse", "--show-toplevel"])
    except (OSError, StorageError) as exc:
        raise StorageError(f"not a usable Git repository: {requested}: {exc}") from exc
    top_level = Path(os.fsdecode(top_level_text.strip())).resolve(strict=True)

    commit_bytes = _git(
        top_level, ["rev-parse", "--verify", "HEAD"], check=False
    ).strip()
    if commit_bytes:
        tracked_diff = _git(
            top_level,
            [
                "diff",
                "--binary",
                "--full-index",
                "--no-ext-diff",
                "--no-textconv",
                "--no-renames",
                "HEAD",
                "--",
            ],
        )
    else:
        # Include staged and unstaged content separately, including repositories without
        # a commit.
        tracked_diff = _git(
            top_level,
            ["diff", "--cached", "--binary", "--full-index", "--no-ext-diff", "--"],
        ) + _git(
            top_level,
            ["diff", "--binary", "--full-index", "--no-ext-diff", "--"],
        )

    untracked_output = _git(
        top_level, ["ls-files", "--others", "--exclude-standard", "-z"]
    )
    untracked_paths = sorted(path for path in untracked_output.split(b"\0") if path)
    if len(untracked_paths) > MAX_SOURCE_UNTRACKED_ENTRIES:
        raise StorageError(
            "too many untracked source entries for provenance capture "
            f"({len(untracked_paths)} > {MAX_SOURCE_UNTRACKED_ENTRIES}); "
            "ignore generated artifacts before launching"
        )
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for raw_path in untracked_paths:
        entry = _capture_untracked_entry(top_level, raw_path)
        if entry["type"] == "file":
            total_bytes += len(entry["contents"])
            if total_bytes > MAX_SOURCE_UNTRACKED_TOTAL_BYTES:
                raise StorageError(
                    "untracked source contents exceed the provenance size limit "
                    f"({MAX_SOURCE_UNTRACKED_TOTAL_BYTES} bytes); ignore generated "
                    "artifacts before launching"
                )
        entries.append(entry)
    return top_level, commit_bytes, tracked_diff, entries


def _source_identity_from_capture(
    commit_bytes: bytes,
    tracked_diff: bytes,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute the public v1 source identity from captured components."""

    digest = hashlib.sha256()
    _hash_part(digest, b"format", b"ssrl-source-id-v1")
    _hash_part(digest, b"commit", commit_bytes)
    _hash_part(digest, b"tracked-diff", tracked_diff)
    for entry in entries:
        raw_path = entry["path"]
        _hash_part(digest, b"untracked-path", raw_path)
        _hash_part(
            digest,
            b"untracked-mode",
            f"{entry['mode']:04o}".encode("ascii"),
        )
        if entry["type"] == "file":
            _hash_part(digest, b"untracked-file", entry["contents"])
        elif entry["type"] == "symlink":
            _hash_part(digest, b"untracked-symlink", entry["target"])
        else:
            raise StorageError(f"unsupported captured source type: {entry['type']!r}")

    return {
        "commit": commit_bytes.decode("ascii") if commit_bytes else None,
        "dirty": bool(tracked_diff or entries),
        "source_id": digest.hexdigest(),
    }


def source_identity(repo: os.PathLike[str] | str) -> dict[str, Any]:
    """Return a deterministic identity for a Git checkout's effective source.

    A clean tree is represented by its HEAD.  For a dirty tree, the binary
    tracked diff and every untracked, non-ignored path and its contents are
    framed into the digest as well.  Ignored output directories therefore do
    not make otherwise identical source trees look different.
    """

    _, commit_bytes, tracked_diff, entries = _capture_effective_source(repo)
    return _source_identity_from_capture(commit_bytes, tracked_diff, entries)


def _validated_source_id(value: str, *, label: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise StorageError(f"{label} must be a lowercase 64-character SHA-256 digest")
    return value


def _source_path_lstat(
    top_level: Path, raw_path: bytes
) -> tuple[os.stat_result | None, bytes | None]:
    """Inspect a Git path without following a symlink in any component."""

    components = _validated_git_relative_path(raw_path)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    descriptor = os.open(top_level, directory_flags)
    try:
        for component in components[:-1]:
            try:
                next_descriptor = os.open(
                    component, directory_flags, dir_fd=descriptor
                )
            except FileNotFoundError:
                return None, None
            os.close(descriptor)
            descriptor = next_descriptor
        try:
            metadata = os.stat(
                components[-1], dir_fd=descriptor, follow_symlinks=False
            )
        except FileNotFoundError:
            return None, None
        target: bytes | None = None
        if stat.S_ISLNK(metadata.st_mode):
            raw_target = os.readlink(components[-1], dir_fd=descriptor)
            target = (
                raw_target if isinstance(raw_target, bytes) else os.fsencode(raw_target)
            )
        return metadata, target
    except OSError as exc:
        raise StorageError(
            f"source path is unsafe or changed while inspecting "
            f"{os.fsdecode(raw_path)}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)


def _validated_internal_symlink_target(
    top_level: Path, raw_path: bytes, raw_target: bytes
) -> tuple[tuple[bytes, ...], Path]:
    """Require a source symlink to resolve inside the repository."""

    if not raw_target or b"\0" in raw_target or raw_target.startswith(b"/"):
        raise StorageError(
            f"source snapshot refuses absolute/empty symlink: "
            f"{os.fsdecode(raw_path)} -> {os.fsdecode(raw_target)}"
        )
    components = list(_validated_git_relative_path(raw_path)[:-1])
    for component in raw_target.split(b"/"):
        if component in {b"", b"."}:
            continue
        if component == b"..":
            if not components:
                raise StorageError(
                    f"source snapshot refuses escaping symlink: "
                    f"{os.fsdecode(raw_path)} -> {os.fsdecode(raw_target)}"
                )
            components.pop()
        else:
            components.append(component)
    candidate = top_level.joinpath(*(os.fsdecode(item) for item in components))
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise StorageError(
            f"source snapshot refuses broken/cyclic symlink: "
            f"{os.fsdecode(raw_path)} -> {os.fsdecode(raw_target)}"
        ) from exc
    if not _is_within(resolved, top_level):
        raise StorageError(
            f"source snapshot refuses escaping symlink: "
            f"{os.fsdecode(raw_path)} -> {os.fsdecode(raw_target)}"
        )
    return tuple(components), resolved


def _validate_snapshot_source_tree(
    top_level: Path, entries: Sequence[Mapping[str, Any]]
) -> None:
    """Reject source entries that cannot be represented safely and portably."""

    stage_output = _git(top_level, ["ls-files", "--stage", "-z"])
    tracked: dict[bytes, bytes] = {}
    for record in stage_output.split(b"\0"):
        if not record:
            continue
        try:
            prefix, raw_path = record.split(b"\t", 1)
            mode, _object_id, stage_number = prefix.split(b" ", 2)
        except ValueError as exc:
            raise StorageError("Git emitted an invalid staged source record") from exc
        # A conflicted index has multiple effective versions and cannot be
        # faithfully represented by `git diff HEAD` alone.
        if stage_number != b"0":
            raise StorageError(
                f"source snapshot refuses an unmerged index entry: "
                f"{os.fsdecode(raw_path)}"
            )
        if mode == b"160000":
            raise StorageError(
                f"source snapshot does not support Git submodules: "
                f"{os.fsdecode(raw_path)}"
            )
        tracked[raw_path] = mode

    untracked = {entry["path"]: entry for entry in entries}
    available_paths: set[bytes] = set()
    path_states: dict[bytes, tuple[os.stat_result | None, bytes | None]] = {}
    for raw_path in sorted(set(tracked) | set(untracked)):
        metadata, target = _source_path_lstat(top_level, raw_path)
        path_states[raw_path] = (metadata, target)
        if metadata is None:
            # A tracked path may be intentionally deleted by the dirty diff.
            if raw_path in untracked:
                raise StorageError(
                    f"untracked source path vanished: {os.fsdecode(raw_path)}"
                )
            continue
        available_paths.add(raw_path)
        if stat.S_ISREG(metadata.st_mode):
            continue
        if stat.S_ISLNK(metadata.st_mode):
            continue
        raise StorageError(
            "source snapshot contains an unsupported special/directory entry: "
            f"{os.fsdecode(raw_path)}"
        )

    available_directories: set[bytes] = {b""}
    for raw_path in available_paths:
        components = _validated_git_relative_path(raw_path)
        for length in range(1, len(components)):
            available_directories.add(b"/".join(components[:length]))

    for raw_path, (metadata, target) in path_states.items():
        if metadata is None or not stat.S_ISLNK(metadata.st_mode):
            continue
        assert target is not None
        lexical_components, resolved = _validated_internal_symlink_target(
            top_level, raw_path, target
        )
        lexical = b"/".join(lexical_components)
        try:
            relative_path = resolved.relative_to(top_level)
            resolved_relative = (
                b"" if relative_path == Path(".") else os.fsencode(os.fspath(relative_path))
            )
        except ValueError as exc:
            raise StorageError(
                f"source symlink escaped while validating: {os.fsdecode(raw_path)}"
            ) from exc
        if lexical not in available_paths | available_directories:
            raise StorageError(
                "source snapshot symlink target is ignored/not captured: "
                f"{os.fsdecode(raw_path)} -> {os.fsdecode(target)}"
            )
        if resolved_relative not in available_paths | available_directories:
            raise StorageError(
                "source snapshot resolved symlink target is ignored/not captured: "
                f"{os.fsdecode(raw_path)} -> {resolved}"
            )


def _snapshot_entry_record(entry: Mapping[str, Any]) -> dict[str, Any]:
    common = {
        "path_b64": base64.b64encode(entry["path"]).decode("ascii"),
        "mode": f"{entry['mode']:04o}",
        "type": entry["type"],
    }
    if entry["type"] == "file":
        contents = entry["contents"]
        return {
            **common,
            "bytes": len(contents),
            "sha256": hashlib.sha256(contents).hexdigest(),
        }
    if entry["type"] == "symlink":
        return {
            **common,
            "target_b64": base64.b64encode(entry["target"]).decode("ascii"),
        }
    raise StorageError(f"unsupported captured source type: {entry['type']!r}")


def _build_untracked_tar(entries: Sequence[Mapping[str, Any]]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(
        fileobj=output,
        mode="w:",
        format=tarfile.PAX_FORMAT,
        encoding=sys.getfilesystemencoding(),
        errors="surrogateescape",
    ) as archive:
        for entry in entries:
            raw_path = entry["path"]
            _validated_git_relative_path(raw_path)
            member = tarfile.TarInfo(os.fsdecode(raw_path))
            member.mode = int(entry["mode"])
            member.uid = member.gid = 0
            member.uname = member.gname = ""
            member.mtime = 0
            if entry["type"] == "file":
                contents = entry["contents"]
                member.type = tarfile.REGTYPE
                member.size = len(contents)
                archive.addfile(member, io.BytesIO(contents))
            elif entry["type"] == "symlink":
                member.type = tarfile.SYMTYPE
                member.size = 0
                member.linkname = os.fsdecode(entry["target"])
                archive.addfile(member)
            else:
                raise StorageError(
                    f"unsupported captured source type: {entry['type']!r}"
                )
    return output.getvalue()


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise StorageError(f"cannot read source snapshot {label}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise StorageError(f"source snapshot {label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise StorageError(f"source snapshot {label} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_untracked_tar(data: bytes) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    total_bytes = 0
    try:
        with tarfile.open(
            fileobj=io.BytesIO(data),
            mode="r:",
            encoding=sys.getfilesystemencoding(),
            errors="surrogateescape",
        ) as archive:
            for member in archive:
                raw_path = os.fsencode(member.name)
                _validated_git_relative_path(raw_path)
                if raw_path in seen:
                    raise StorageError(
                        f"source snapshot tar has duplicate path: {member.name}"
                    )
                seen.add(raw_path)
                if len(seen) > MAX_SOURCE_UNTRACKED_ENTRIES:
                    raise StorageError("source snapshot tar has too many entries")
                if (
                    member.uid != 0
                    or member.gid != 0
                    or member.uname
                    or member.gname
                    or member.mtime != 0
                ):
                    raise StorageError(
                        f"source snapshot tar has non-canonical metadata: {member.name}"
                    )
                mode = int(member.mode) & 0o7777
                if member.isreg():
                    if member.size < 0 or member.size > MAX_SOURCE_UNTRACKED_FILE_BYTES:
                        raise StorageError(
                            f"source snapshot tar file exceeds size limit: {member.name}"
                        )
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise StorageError(
                            f"source snapshot tar cannot read file: {member.name}"
                        )
                    contents = extracted.read(MAX_SOURCE_UNTRACKED_FILE_BYTES + 1)
                    if len(contents) != member.size:
                        raise StorageError(
                            f"source snapshot tar file has inconsistent size: {member.name}"
                        )
                    total_bytes += len(contents)
                    if total_bytes > MAX_SOURCE_UNTRACKED_TOTAL_BYTES:
                        raise StorageError(
                            "source snapshot tar exceeds total content size limit"
                        )
                    entries.append(
                        {
                            "path": raw_path,
                            "mode": mode,
                            "type": "file",
                            "contents": contents,
                        }
                    )
                elif member.issym():
                    raw_target = os.fsencode(member.linkname)
                    # Reject symlink targets outside the reconstructed tree.
                    lexical_components = list(
                        _validated_git_relative_path(raw_path)[:-1]
                    )
                    if (
                        not raw_target
                        or b"\0" in raw_target
                        or raw_target.startswith(b"/")
                    ):
                        raise StorageError(
                            f"source snapshot tar has unsafe symlink: {member.name}"
                        )
                    for component in raw_target.split(b"/"):
                        if component in {b"", b"."}:
                            continue
                        if component == b"..":
                            if not lexical_components:
                                raise StorageError(
                                    "source snapshot tar has escaping symlink: "
                                    f"{member.name}"
                                )
                            lexical_components.pop()
                        else:
                            lexical_components.append(component)
                    entries.append(
                        {
                            "path": raw_path,
                            "mode": mode,
                            "type": "symlink",
                            "target": raw_target,
                        }
                    )
                else:
                    raise StorageError(
                        "source snapshot tar contains a directory, hardlink, or "
                        f"special file: {member.name}"
                    )
    except (tarfile.TarError, OSError, UnicodeError) as exc:
        raise StorageError(f"cannot read source snapshot tar: {exc}") from exc

    if [entry["path"] for entry in entries] != sorted(
        entry["path"] for entry in entries
    ):
        raise StorageError("source snapshot tar entries are not in canonical order")
    return entries


def _snapshot_file_descriptor(path: str, data: bytes) -> dict[str, Any]:
    return {
        "path": path,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _expected_snapshot_files(directory: Path) -> None:
    expected = {
        SOURCE_SNAPSHOT_METADATA_NAME,
        SOURCE_SNAPSHOT_DIFF_NAME,
        SOURCE_SNAPSHOT_UNTRACKED_NAME,
    }
    try:
        entries = list(os.scandir(directory))
    except OSError as exc:
        raise StorageError(f"cannot inspect source snapshot directory: {exc}") from exc
    actual = {entry.name for entry in entries}
    if actual != expected:
        raise StorageError(
            "source snapshot file inventory mismatch: "
            f"unexpected={sorted(actual - expected)!r}, "
            f"missing={sorted(expected - actual)!r}"
        )
    for entry in entries:
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise StorageError(
                f"cannot inspect source snapshot entry {entry.name}: {exc}"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise StorageError(
                f"source snapshot entry is not a regular file: {entry.name}"
            )


def verify_source_snapshot(
    snapshot_dir: os.PathLike[str] | str, *, source_id: str
) -> dict[str, Any]:
    """Verify a snapshot and recompute its effective source identity offline."""

    expected_source_id = _validated_source_id(source_id, label="--source-id")
    directory = _real_directory(snapshot_dir, label="source snapshot")
    _expected_snapshot_files(directory)
    metadata_path = directory / SOURCE_SNAPSHOT_METADATA_NAME
    metadata = _read_json_file(metadata_path)
    if not isinstance(metadata, dict):
        raise StorageError("source snapshot metadata must be a JSON object")
    if set(metadata) != {
        "schema_version",
        "format",
        "created_at",
        "source",
        "tracked_patch",
        "untracked_archive",
        "untracked_entries",
    }:
        raise StorageError("source snapshot metadata has unsupported/missing fields")
    if metadata.get("schema_version") != SOURCE_SNAPSHOT_SCHEMA_VERSION:
        raise StorageError("source snapshot metadata has an unsupported schema")
    if metadata.get("format") != SOURCE_SNAPSHOT_FORMAT:
        raise StorageError("source snapshot metadata has an unsupported format")
    if not isinstance(metadata.get("created_at"), str):
        raise StorageError("source snapshot created_at must be a string")

    source = metadata.get("source")
    if not isinstance(source, dict) or set(source) != {"commit", "dirty", "source_id"}:
        raise StorageError("source snapshot has invalid source metadata")
    commit = source.get("commit")
    if commit is not None and (
        not isinstance(commit, str)
        or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit)
    ):
        raise StorageError("source snapshot commit is not a Git object ID")
    if not isinstance(source.get("dirty"), bool):
        raise StorageError("source snapshot dirty flag must be boolean")
    stored_source_id = source.get("source_id")
    if not isinstance(stored_source_id, str):
        raise StorageError("source snapshot source_id must be a string")
    _validated_source_id(stored_source_id, label="snapshot source_id")
    if stored_source_id != expected_source_id:
        raise StorageError(
            "source snapshot does not match the expected source ID: "
            f"stored={stored_source_id}, expected={expected_source_id}"
        )

    tracked_patch = _read_regular_bytes(
        directory / SOURCE_SNAPSHOT_DIFF_NAME, label=SOURCE_SNAPSHOT_DIFF_NAME
    )
    untracked_tar = _read_regular_bytes(
        directory / SOURCE_SNAPSHOT_UNTRACKED_NAME,
        label=SOURCE_SNAPSHOT_UNTRACKED_NAME,
    )
    expected_patch_descriptor = _snapshot_file_descriptor(
        SOURCE_SNAPSHOT_DIFF_NAME, tracked_patch
    )
    expected_tar_descriptor = _snapshot_file_descriptor(
        SOURCE_SNAPSHOT_UNTRACKED_NAME, untracked_tar
    )
    if metadata.get("tracked_patch") != expected_patch_descriptor:
        raise StorageError("tracked source patch digest/size does not match metadata")
    if metadata.get("untracked_archive") != expected_tar_descriptor:
        raise StorageError("untracked source archive digest/size does not match metadata")

    entries = _read_untracked_tar(untracked_tar)
    if _build_untracked_tar(entries) != untracked_tar:
        raise StorageError(
            "untracked source archive is not in canonical form or has trailing data"
        )
    expected_entry_records = [_snapshot_entry_record(entry) for entry in entries]
    if metadata.get("untracked_entries") != expected_entry_records:
        raise StorageError("untracked source inventory does not match its archive")
    computed = _source_identity_from_capture(
        commit.encode("ascii") if commit is not None else b"",
        tracked_patch,
        entries,
    )
    if computed != source:
        raise StorageError(
            "source snapshot contents do not reproduce the recorded source identity"
        )
    if computed["source_id"] != expected_source_id:
        raise StorageError("source snapshot contents do not match --source-id")
    return {
        "snapshot_dir": os.fspath(directory),
        "source_id": computed["source_id"],
        "commit": computed["commit"],
        "dirty": computed["dirty"],
        "tracked_patch_bytes": len(tracked_patch),
        "untracked_archive_bytes": len(untracked_tar),
        "untracked_entries": len(entries),
        "valid": True,
    }


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a sibling directory without replacing a winner.

    Some cluster home filesystems reject Linux ``RENAME_NOREPLACE`` with
    ``EINVAL``.  Serialize publishers by locking their common parent instead,
    check for a winner while holding that lock, and use portable same-directory
    rename for the atomic visibility boundary.  Every publisher in this module
    follows this protocol, including concurrent submissions of the same source.
    """

    if source.parent != destination.parent:
        raise StorageError(
            "source snapshot staging and destination directories must be siblings"
        )
    parent = destination.parent
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        parent_descriptor = os.open(parent, flags)
    except OSError as exc:
        raise StorageError(
            f"cannot open source snapshot publication directory {parent}: {exc}"
        ) from exc
    try:
        try:
            fcntl.flock(parent_descriptor, fcntl.LOCK_EX)
        except OSError as exc:
            raise StorageError(
                f"cannot lock source snapshot publication directory {parent}: {exc}"
            ) from exc

        try:
            os.stat(
                destination.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise StorageError(
                f"refusing to replace existing source snapshot: {destination}"
            )

        try:
            os.rename(
                source.name,
                destination.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
        except OSError as exc:
            if exc.errno in {errno.EEXIST, errno.ENOTEMPTY}:
                raise StorageError(
                    f"refusing to replace existing source snapshot: {destination}"
                ) from exc
            raise StorageError(
                f"cannot atomically publish source snapshot {destination}: {exc}"
            ) from exc
    finally:
        os.close(parent_descriptor)


def create_source_snapshot(
    repo: os.PathLike[str] | str,
    snapshot_dir: os.PathLike[str] | str,
    *,
    source_id: str,
) -> dict[str, Any]:
    """Create and atomically publish an immutable effective-source snapshot."""

    expected_source_id = _validated_source_id(source_id, label="--source-id")
    requested_destination = Path(snapshot_dir).expanduser().absolute()
    parent = _real_directory(
        requested_destination.parent, label="source snapshot parent"
    )
    destination = parent / requested_destination.name
    if destination.name in {"", ".", ".."}:
        raise StorageError("source snapshot destination must name a child directory")
    try:
        destination.lstat()
    except FileNotFoundError:
        pass
    else:
        raise StorageError(f"refusing to replace existing source snapshot: {destination}")

    top_level, commit_bytes, tracked_diff, entries = _capture_effective_source(repo)
    # A snapshot created inside its own repository immediately changes the
    # effective source and can recursively capture prior snapshot artifacts.
    if _is_within(destination, top_level):
        raise StorageError(
            "source snapshot destination must be outside the source repository"
        )
    captured_identity = _source_identity_from_capture(
        commit_bytes, tracked_diff, entries
    )
    if captured_identity["source_id"] != expected_source_id:
        raise StorageError(
            "current repository does not match the expected source ID: "
            f"current={captured_identity['source_id']}, "
            f"expected={expected_source_id}"
        )
    _validate_snapshot_source_tree(top_level, entries)

    # Capture source twice to detect concurrent edits before creating any staging files.
    _, check_commit, check_diff, check_entries = _capture_effective_source(top_level)
    check_identity = _source_identity_from_capture(
        check_commit, check_diff, check_entries
    )
    if check_identity["source_id"] != captured_identity["source_id"]:
        raise StorageError("source repository changed while creating its snapshot")

    untracked_tar = _build_untracked_tar(entries)
    metadata = {
        "schema_version": SOURCE_SNAPSHOT_SCHEMA_VERSION,
        "format": SOURCE_SNAPSHOT_FORMAT,
        "created_at": _utc_now(),
        "source": captured_identity,
        "tracked_patch": _snapshot_file_descriptor(
            SOURCE_SNAPSHOT_DIFF_NAME, tracked_diff
        ),
        "untracked_archive": _snapshot_file_descriptor(
            SOURCE_SNAPSHOT_UNTRACKED_NAME, untracked_tar
        ),
        "untracked_entries": [_snapshot_entry_record(entry) for entry in entries],
    }
    stage = parent / (
        f".{destination.name}.source-snapshot.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        os.mkdir(stage, 0o750)
        os.chmod(stage, 0o750)
        _stage_write_bytes(
            stage / SOURCE_SNAPSHOT_DIFF_NAME,
            tracked_diff,
            mode=0o440,
        )
        _stage_write_bytes(
            stage / SOURCE_SNAPSHOT_UNTRACKED_NAME,
            untracked_tar,
            mode=0o440,
        )
        _stage_write_json(
            stage / SOURCE_SNAPSHOT_METADATA_NAME,
            metadata,
            mode=0o440,
        )
        result = verify_source_snapshot(stage, source_id=expected_source_id)
        _fsync_directory(stage)
        _rename_noreplace(stage, destination)
        _fsync_directory(parent)
        result["snapshot_dir"] = os.fspath(destination)
        return result
    except BaseException:
        if stage.exists() and not stage.is_symlink():
            shutil.rmtree(stage, ignore_errors=True)
        raise


def _source_tree_destination(tree: Path, raw_path: bytes) -> Path:
    components = _validated_git_relative_path(raw_path)
    if b".git" in components:
        raise StorageError(
            f"source bundle refuses reserved Git path: {os.fsdecode(raw_path)}"
        )
    parent = tree
    for raw_component in components[:-1]:
        component = os.fsdecode(raw_component)
        candidate = parent / component
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            candidate.mkdir(mode=0o750)
        else:
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise StorageError(
                    "source bundle path has an unsafe parent: "
                    f"{os.fsdecode(raw_path)}"
                )
        parent = candidate
    return parent / os.fsdecode(components[-1])


def _materialize_untracked_source(
    tree: Path, entries: Sequence[Mapping[str, Any]]
) -> None:
    for entry in entries:
        destination = _source_tree_destination(tree, entry["path"])
        if os.path.lexists(destination):
            raise StorageError(
                "untracked source collides with the reconstructed tracked tree: "
                f"{os.fsdecode(entry['path'])}"
            )
        if entry["type"] == "file":
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(destination, flags, int(entry["mode"]) & 0o777)
            try:
                _write_all(descriptor, entry["contents"])
                os.fchmod(descriptor, int(entry["mode"]) & 0o777)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        elif entry["type"] == "symlink":
            os.symlink(os.fsdecode(entry["target"]), destination)
        else:
            raise StorageError(
                f"unsupported captured source type: {entry['type']!r}"
            )


def _materialize_source_snapshot(
    repo: os.PathLike[str] | str,
    snapshot_dir: os.PathLike[str] | str,
    tree: Path,
    *,
    source_id: str,
) -> dict[str, Any]:
    snapshot = verify_source_snapshot(snapshot_dir, source_id=source_id)
    commit = snapshot.get("commit")
    if not isinstance(commit, str):
        raise StorageError("runnable source bundles require a committed Git HEAD")

    repository = Path(repo).expanduser().resolve(strict=True)
    tree.mkdir(mode=0o750)
    _git(tree, ["init", "-q"])
    _git(
        tree,
        [
            "fetch",
            "-q",
            "--depth=1",
            "--no-tags",
            os.fspath(repository),
            commit,
        ],
    )
    _git(tree, ["checkout", "-q", "--detach", "FETCH_HEAD"])

    snapshot_path = Path(snapshot_dir).expanduser().resolve(strict=True)
    tracked_patch = _read_regular_bytes(
        snapshot_path / SOURCE_SNAPSHOT_DIFF_NAME,
        label=SOURCE_SNAPSHOT_DIFF_NAME,
    )
    if tracked_patch:
        _git(
            tree,
            [
                "apply",
                "--index",
                "--binary",
                "--whitespace=nowarn",
                os.fspath(snapshot_path / SOURCE_SNAPSHOT_DIFF_NAME),
            ],
        )
    untracked_tar = _read_regular_bytes(
        snapshot_path / SOURCE_SNAPSHOT_UNTRACKED_NAME,
        label=SOURCE_SNAPSHOT_UNTRACKED_NAME,
    )
    _materialize_untracked_source(tree, _read_untracked_tar(untracked_tar))

    actual = source_identity(tree)
    if actual["source_id"] != source_id:
        raise StorageError(
            "materialized source tree does not match its snapshot: "
            f"actual={actual['source_id']}, expected={source_id}"
        )
    return actual


def _source_snapshot_untracked_paths(snapshot_dir: Path) -> set[bytes]:
    payload = _read_regular_bytes(
        snapshot_dir / SOURCE_SNAPSHOT_UNTRACKED_NAME,
        label=SOURCE_SNAPSHOT_UNTRACKED_NAME,
    )
    return {entry["path"] for entry in _read_untracked_tar(payload)}


def _make_source_tree_readonly(tree: Path, untracked_paths: set[bytes]) -> None:
    for directory, directories, filenames in os.walk(tree, topdown=False):
        root = Path(directory)
        for filename in filenames:
            path = root / filename
            metadata = path.lstat()
            relative = os.fsencode(os.fspath(path.relative_to(tree)))
            if stat.S_ISREG(metadata.st_mode) and relative not in untracked_paths:
                path.chmod(stat.S_IMODE(metadata.st_mode) & ~0o222)
        for dirname in directories:
            path = root / dirname
            metadata = path.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                path.chmod(stat.S_IMODE(metadata.st_mode) & ~0o222)
        metadata = root.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            root.chmod(stat.S_IMODE(metadata.st_mode) & ~0o222)


def _fsync_source_tree(tree: Path) -> None:
    directories: list[Path] = []
    for directory, child_directories, filenames in os.walk(
        tree, followlinks=False
    ):
        root = Path(directory)
        directories.append(root)
        for name in child_directories:
            metadata = (root / name).lstat()
            if not (
                stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)
            ):
                raise StorageError(f"unsafe source bundle entry: {root / name}")
        for name in filenames:
            path = root / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise StorageError(f"unsafe source bundle entry: {path}")
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for directory in reversed(directories):
        _fsync_directory(directory)


def _remove_source_bundle_stage(stage: Path) -> None:
    if not stage.exists() or stage.is_symlink():
        return
    for directory, directories, _filenames in os.walk(stage, topdown=False):
        root = Path(directory)
        for dirname in directories:
            path = root / dirname
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o750)
        root.chmod(0o750)
    shutil.rmtree(stage, ignore_errors=True)


def _verify_source_tree_readonly(tree: Path, untracked_paths: set[bytes]) -> None:
    for directory, directories, filenames in os.walk(tree):
        root = Path(directory)
        for name in [*directories, *filenames]:
            path = root / name
            metadata = path.lstat()
            relative = os.fsencode(os.fspath(path.relative_to(tree)))
            allowed_writable = (
                stat.S_ISREG(metadata.st_mode) and relative in untracked_paths
            )
            if (
                not stat.S_ISLNK(metadata.st_mode)
                and not allowed_writable
                and stat.S_IMODE(metadata.st_mode) & 0o222
            ):
                raise StorageError(f"source bundle tree is writable: {path}")


def verify_source_bundle(
    bundle_dir: os.PathLike[str] | str, *, source_id: str
) -> dict[str, Any]:
    expected_source_id = _validated_source_id(source_id, label="--source-id")
    bundle = _real_directory(bundle_dir, label="source bundle")
    expected = {
        SOURCE_BUNDLE_METADATA_NAME,
        SOURCE_BUNDLE_PROVENANCE_DIR,
        SOURCE_BUNDLE_TREE_DIR,
    }
    try:
        entries = {entry.name: entry for entry in os.scandir(bundle)}
    except OSError as exc:
        raise StorageError(f"cannot inspect source bundle: {exc}") from exc
    if set(entries) != expected:
        raise StorageError(
            "source bundle inventory mismatch: "
            f"unexpected={sorted(set(entries) - expected)!r}, "
            f"missing={sorted(expected - set(entries))!r}"
        )
    for name in (SOURCE_BUNDLE_PROVENANCE_DIR, SOURCE_BUNDLE_TREE_DIR):
        metadata = entries[name].stat(follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode):
            raise StorageError(f"source bundle {name} is not a real directory")
    metadata_entry = entries[SOURCE_BUNDLE_METADATA_NAME].stat(
        follow_symlinks=False
    )
    if not stat.S_ISREG(metadata_entry.st_mode):
        raise StorageError("source bundle metadata is not a regular file")

    metadata = _read_json_file(bundle / SOURCE_BUNDLE_METADATA_NAME)
    if not isinstance(metadata, dict) or set(metadata) != {
        "schema_version",
        "format",
        "created_at",
        "source",
        "provenance_directory",
        "tree_directory",
    }:
        raise StorageError("source bundle metadata has unsupported/missing fields")
    if metadata.get("schema_version") != SOURCE_BUNDLE_SCHEMA_VERSION:
        raise StorageError("source bundle metadata has an unsupported schema")
    if metadata.get("format") != SOURCE_BUNDLE_FORMAT:
        raise StorageError("source bundle metadata has an unsupported format")
    if not isinstance(metadata.get("created_at"), str):
        raise StorageError("source bundle created_at must be a string")
    if metadata.get("provenance_directory") != SOURCE_BUNDLE_PROVENANCE_DIR:
        raise StorageError("source bundle has an invalid provenance directory")
    if metadata.get("tree_directory") != SOURCE_BUNDLE_TREE_DIR:
        raise StorageError("source bundle has an invalid tree directory")
    source = metadata.get("source")
    if not isinstance(source, dict) or source.get("source_id") != expected_source_id:
        raise StorageError("source bundle metadata does not match --source-id")

    snapshot = verify_source_snapshot(
        bundle / SOURCE_BUNDLE_PROVENANCE_DIR, source_id=expected_source_id
    )
    untracked_paths = _source_snapshot_untracked_paths(
        bundle / SOURCE_BUNDLE_PROVENANCE_DIR
    )
    tree = bundle / SOURCE_BUNDLE_TREE_DIR
    if not (tree / ".git").is_dir() or (tree / ".git").is_symlink():
        raise StorageError("source bundle tree is not a standalone Git checkout")
    actual = source_identity(tree)
    if actual != source:
        raise StorageError("source bundle tree does not reproduce its source metadata")
    _verify_source_tree_readonly(tree, untracked_paths)
    return {
        "bundle_dir": os.fspath(bundle),
        "tree_dir": os.fspath(tree),
        "source_id": expected_source_id,
        "commit": snapshot["commit"],
        "dirty": snapshot["dirty"],
        "valid": True,
    }


def create_source_bundle(
    repo: os.PathLike[str] | str,
    bundle_dir: os.PathLike[str] | str,
    *,
    source_id: str,
) -> dict[str, Any]:
    expected_source_id = _validated_source_id(source_id, label="--source-id")
    requested_destination = Path(bundle_dir).expanduser().absolute()
    parent = _real_directory(
        requested_destination.parent, label="source bundle parent"
    )
    destination = parent / requested_destination.name
    repository = Path(repo).expanduser().resolve(strict=True)
    if _is_within(destination, repository):
        raise StorageError(
            "source bundle destination must be outside the source repository"
        )
    if os.path.lexists(destination):
        raise StorageError(
            f"refusing to replace existing source bundle: {destination}"
        )

    stage = parent / (
        f".{destination.name}.source-bundle.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        stage.mkdir(mode=0o750)
        provenance = stage / SOURCE_BUNDLE_PROVENANCE_DIR
        create_source_snapshot(
            repository, provenance, source_id=expected_source_id
        )
        tree = stage / SOURCE_BUNDLE_TREE_DIR
        source = _materialize_source_snapshot(
            repository,
            provenance,
            tree,
            source_id=expected_source_id,
        )
        untracked_paths = _source_snapshot_untracked_paths(provenance)
        _make_source_tree_readonly(tree, untracked_paths)
        _fsync_source_tree(tree)
        metadata = {
            "schema_version": SOURCE_BUNDLE_SCHEMA_VERSION,
            "format": SOURCE_BUNDLE_FORMAT,
            "created_at": _utc_now(),
            "source": source,
            "provenance_directory": SOURCE_BUNDLE_PROVENANCE_DIR,
            "tree_directory": SOURCE_BUNDLE_TREE_DIR,
        }
        _stage_write_json(
            stage / SOURCE_BUNDLE_METADATA_NAME,
            metadata,
            mode=0o440,
        )
        result = verify_source_bundle(stage, source_id=expected_source_id)
        _fsync_directory(stage)
        _rename_noreplace(stage, destination)
        _fsync_directory(parent)
        result["bundle_dir"] = os.fspath(destination)
        result["tree_dir"] = os.fspath(destination / SOURCE_BUNDLE_TREE_DIR)
        return result
    except BaseException:
        _remove_source_bundle_stage(stage)
        raise


def ensure_source_bundle(
    repo: os.PathLike[str] | str,
    bundle_dir: os.PathLike[str] | str,
    *,
    source_id: str,
) -> dict[str, Any]:
    destination = Path(bundle_dir).expanduser().absolute()
    if os.path.lexists(destination):
        return verify_source_bundle(destination, source_id=source_id)
    try:
        return create_source_bundle(repo, destination, source_id=source_id)
    except StorageError:
        # Concurrent submitters of the same effective source may race the final
        # no-replace rename. Accept only the fully verified winning bundle.
        if os.path.lexists(destination):
            return verify_source_bundle(destination, source_id=source_id)
        raise


def _parse_bool(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise StorageError(f"{name} must be a boolean, got {value!r}")


def _allowlisted_environment(
    names: Iterable[str], environment: Mapping[str, str]
) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        if name not in environment:
            continue
        value = environment[name]
        if "\x00" in value:
            raise StorageError(f"{name} contains a NUL byte")
        if len(value) > 16_384:
            raise StorageError(f"{name} is unexpectedly large")
        result[name] = value
    return result


def run_config_id(
    command: Sequence[str], *, launcher_sha256: str
) -> str:
    """Return a restart-stable identity for the effective training config.

    The interpreter and training-script paths are execution locations, not
    configuration. Slurm source bundles are copied into a restart-specific
    scratch directory, so including those first two argv entries would change
    the identity after every requeue. Source, launcher, and runtime identities
    are authenticated independently in the run manifest.
    """

    if not re.fullmatch(r"[0-9a-f]{64}", launcher_sha256):
        raise StorageError("launcher SHA-256 must be a lowercase 64-character digest")
    arguments = list(command)
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if len(arguments) < 2:
        raise StorageError(
            "run config identity requires an interpreter and training script"
        )
    stable_arguments = [
        argument
        for argument in arguments[2:]
        if not argument.startswith(
            ("wandb.id=", "wandb.resume=", "wandb.resume_from=")
        )
    ]
    if any("\x00" in argument for argument in stable_arguments):
        raise StorageError("run config argument contains a NUL byte")
    payload = {
        "arguments": stable_arguments,
        "launcher_sha256": launcher_sha256,
        "schema": "ssrl-run-config-v1",
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def run_identity(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = os.environ if environment is None else environment
    identity = _allowlisted_environment(IDENTITY_ENV_ALLOWLIST, environment)
    run_id = identity.get("RUN_ID", "")
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise StorageError(
            "RUN_ID is required and must match "
            "[A-Za-z0-9][A-Za-z0-9._:-]{0,255}"
        )
    return identity


def _source_provenance(
    *,
    repo: os.PathLike[str] | str | None,
    source_id: str | None,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    computed: dict[str, Any] | None = source_identity(repo) if repo else None
    supplied_id = source_id or environment.get("RUN_SOURCE_ID")
    if computed is not None:
        if supplied_id and supplied_id != computed["source_id"]:
            raise StorageError(
                "supplied RUN_SOURCE_ID/--source-id does not match the current repository"
            )
        return computed

    if not supplied_id or not re.fullmatch(r"[0-9a-f]{64}", supplied_id):
        raise StorageError(
            "source provenance requires --repo or a 64-character --source-id/RUN_SOURCE_ID"
        )
    commit = environment.get("RUN_SOURCE_COMMIT") or None
    dirty_value = environment.get("RUN_SOURCE_DIRTY")
    dirty = (
        _parse_bool(dirty_value, name="RUN_SOURCE_DIRTY")
        if dirty_value is not None
        else None
    )
    return {"commit": commit, "dirty": dirty, "source_id": supplied_id}


def _new_manifest(
    identity: Mapping[str, str],
    provenance: Mapping[str, Any],
    *,
    legacy_adopted: bool,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "identity": dict(identity),
        "provenance": dict(provenance),
        "legacy_adopted": bool(legacy_adopted),
    }


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / MANIFEST_NAME
    try:
        manifest = _read_json_file(path)
    except FileNotFoundError as exc:
        raise StorageError(f"run manifest does not exist: {path}") from exc
    if not isinstance(manifest, dict):
        raise StorageError(f"run manifest is not a JSON object: {path}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise StorageError(
            f"unsupported run manifest schema in {path}: "
            f"{manifest.get('schema_version')!r}"
        )
    if not isinstance(manifest.get("identity"), dict) or not isinstance(
        manifest.get("provenance"), dict
    ):
        raise StorageError(f"invalid identity/provenance in {path}")
    return manifest


def _describe_mapping_difference(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> str:
    differences: list[str] = []
    for key in sorted(set(expected) | set(actual)):
        if expected.get(key) != actual.get(key) or (key in expected) != (key in actual):
            differences.append(
                f"{key}: stored={expected.get(key)!r}, current={actual.get(key)!r}"
            )
    return "; ".join(differences) or "unknown mismatch"


def validate_manifest(
    run_dir: os.PathLike[str] | str,
    *,
    repo: os.PathLike[str] | str | None = None,
    source_id: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    environment = os.environ if environment is None else environment
    directory = Path(run_dir).expanduser()
    identity = run_identity(environment)
    provenance = _source_provenance(
        repo=repo, source_id=source_id, environment=environment
    )
    manifest = _load_manifest(directory)
    if manifest["identity"] != identity:
        raise StorageError(
            "run identity does not match immutable manifest: "
            + _describe_mapping_difference(manifest["identity"], identity)
        )
    if manifest["provenance"] != provenance:
        raise StorageError(
            "source provenance does not match immutable manifest: "
            + _describe_mapping_difference(manifest["provenance"], provenance)
        )
    return manifest


def validate_fork_parent(
    run_dir: os.PathLike[str] | str,
    *,
    parent_run_id: str,
    parent_source_id: str,
    parent_marker_name: str,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate a certified parent artifact against the child experiment.

    The completion envelope itself is verified by ``scratch_sync.sh`` before
    and after the artifact copy. This helper validates the immutable manifest,
    checkpoint pointer, explicit parent identity, and semantic compatibility.
    """

    environment = os.environ if environment is None else environment
    directory = Path(run_dir).expanduser()
    if not directory.is_dir() or directory.is_symlink():
        raise StorageError(f"fork parent is not a real directory: {directory}")
    if not _RUN_ID_PATTERN.fullmatch(parent_run_id):
        raise StorageError("fork parent run ID is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", parent_source_id):
        raise StorageError("fork parent source ID must be a SHA-256 digest")
    expected_marker_kind = {
        ".sync_complete.json": "final-publish",
        ".recovery_complete.json": "periodic-recovery",
    }.get(parent_marker_name)
    if expected_marker_kind is None:
        raise StorageError("fork parent marker name is not supported")

    parent_manifest = _load_manifest(directory)
    parent_identity = parent_manifest["identity"]
    parent_provenance = parent_manifest["provenance"]
    if parent_identity.get("RUN_ID") != parent_run_id:
        raise StorageError(
            "fork parent run ID does not match immutable manifest: "
            f"stored={parent_identity.get('RUN_ID')!r}, expected={parent_run_id!r}"
        )
    if parent_provenance.get("source_id") != parent_source_id:
        raise StorageError(
            "fork parent source does not match immutable manifest: "
            f"stored={parent_provenance.get('source_id')!r}, "
            f"expected={parent_source_id!r}"
        )

    child_identity = run_identity(environment)
    if child_identity["RUN_ID"] == parent_run_id:
        raise StorageError("checkpoint fork requires a distinct child RUN_ID")
    differences: list[str] = []
    for field in FORK_STABLE_IDENTITY_FIELDS:
        parent_present = field in parent_identity
        child_present = field in child_identity
        if (
            parent_present != child_present
            or parent_identity.get(field) != child_identity.get(field)
        ):
            differences.append(
                f"{field}: parent={parent_identity.get(field)!r}, "
                f"child={child_identity.get(field)!r}"
            )
    if differences:
        raise StorageError(
            "fork parent has different experiment coordinates: "
            + "; ".join(differences)
        )

    marker_path = directory / parent_marker_name
    marker = _read_json_file(marker_path)
    if not isinstance(marker, dict):
        raise StorageError(f"fork parent marker is not a JSON object: {marker_path}")
    if marker.get("schema_version") != 2:
        raise StorageError("fork parent marker has an unsupported schema")
    if marker.get("marker_kind") != expected_marker_kind:
        raise StorageError("fork parent marker kind does not match its filename")
    if marker.get("run_uid") != parent_run_id:
        raise StorageError("fork parent marker run UID does not match the parent")
    checkpoint = marker.get("checkpoint")
    if not isinstance(checkpoint, str) or not _CHECKPOINT_NAME_PATTERN.fullmatch(
        checkpoint
    ):
        raise StorageError("fork parent marker has an invalid checkpoint")
    latest_path = directory / "checkpoint.pkl" / "latest"
    try:
        latest_metadata = latest_path.lstat()
        latest = latest_path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError, UnicodeError) as exc:
        raise StorageError(f"fork parent checkpoint pointer cannot be read: {exc}") from exc
    if not stat.S_ISREG(latest_metadata.st_mode) or latest_path.is_symlink():
        raise StorageError("fork parent checkpoint pointer is not a regular file")
    if latest != checkpoint:
        raise StorageError("fork parent marker/checkpoint pointer mismatch")
    generation = directory / "checkpoint.pkl" / checkpoint
    if not generation.is_dir() or generation.is_symlink():
        raise StorageError("fork parent checkpoint generation is missing or unsafe")
    for required in ("agent.pkl", "done"):
        path = generation / required
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise StorageError(
                f"fork parent checkpoint is missing required artifact: {required}"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise StorageError(
                f"fork parent checkpoint artifact is unsafe: {required}"
            )

    _, manifest_sha256 = _catalog_hash_regular(directory / MANIFEST_NAME)
    _, marker_sha256 = _catalog_hash_regular(marker_path)
    return {
        "child_identity": child_identity,
        "parent_checkpoint": checkpoint,
        "parent_identity": parent_identity,
        "parent_manifest": parent_manifest,
        "parent_manifest_sha256": manifest_sha256,
        "parent_marker": marker,
        "parent_marker_sha256": marker_sha256,
        "parent_provenance": parent_provenance,
    }


def fork_manifest(
    run_dir: os.PathLike[str] | str,
    *,
    parent_run_id: str,
    parent_source_id: str,
    parent_marker_name: str,
    parent_artifact: str,
    repo: os.PathLike[str] | str | None = None,
    source_id: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Rekey a copied, certified checkpoint as a provenance-linked child run."""

    environment = os.environ if environment is None else environment
    directory = Path(run_dir).expanduser()
    if (
        not parent_artifact
        or "\x00" in parent_artifact
        or "\n" in parent_artifact
        or "\r" in parent_artifact
    ):
        raise StorageError("fork parent artifact path is invalid")
    validated = validate_fork_parent(
        directory,
        parent_run_id=parent_run_id,
        parent_source_id=parent_source_id,
        parent_marker_name=parent_marker_name,
        environment=environment,
    )
    child_identity = validated["child_identity"]
    child_provenance = _source_provenance(
        repo=repo, source_id=source_id, environment=environment
    )

    temporary = directory / f".fork-parent-metadata-{uuid.uuid4().hex}"
    lineage_root = directory / "lineage"
    if temporary.exists() or temporary.is_symlink():
        raise StorageError(f"fork metadata staging collision: {temporary}")
    temporary.mkdir(mode=0o750)
    metadata_entries = (
        MANIFEST_NAME,
        ATTEMPTS_NAME,
        WANDB_RECEIPT_NAME,
        EVENTS_NAME,
        STATUS_NAME,
        ".run_status.lock",
        # Retain the parent config as provenance; the child writes its own log
        # directory.
        "dreamer_config.yaml",
        "config.yaml",
        "source_provenance",
        "attempts",
        ".sync_complete.json",
        ".recovery_complete.json",
    )
    if lineage_root.exists() or lineage_root.is_symlink():
        metadata_entries += ("lineage",)
    try:
        for name in metadata_entries:
            source = directory / name
            if not source.exists() and not source.is_symlink():
                continue
            metadata = source.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not (
                stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)
            ):
                raise StorageError(f"unsafe fork parent metadata entry: {source}")
            archived_name = "prior_lineage" if name == "lineage" else name
            os.replace(source, temporary / archived_name)

        lineage_root.mkdir(mode=0o750)
        parent_metadata = lineage_root / "parent_metadata"
        os.replace(temporary, parent_metadata)
        _fsync_directory(lineage_root)

        manifest = _new_manifest(
            child_identity, child_provenance, legacy_adopted=False
        )
        manifest["lineage"] = {
            "forked_at": _utc_now(),
            "kind": "checkpoint-fork",
            "parent_artifact": parent_artifact,
            "parent_checkpoint": validated["parent_checkpoint"],
            "parent_identity": validated["parent_identity"],
            "parent_manifest_sha256": validated["parent_manifest_sha256"],
            "parent_marker_name": parent_marker_name,
            "parent_marker_sha256": validated["parent_marker_sha256"],
            "parent_provenance": validated["parent_provenance"],
        }
        _atomic_write_json(
            directory / MANIFEST_NAME, manifest, mode=0o440, exclusive=True
        )
        _append_json_line(
            directory / ATTEMPTS_NAME,
            _attempt_record(
                mode="fork", identity=child_identity, environment=environment
            ),
        )
        return manifest
    except Exception:
        # The caller operates on a disposable copy stage and will remove it.
        # Avoid attempting an in-place rollback that could itself be partial.
        raise


def _attempt_record(
    *, mode: str, identity: Mapping[str, str], environment: Mapping[str, str]
) -> dict[str, Any]:
    attempt = _allowlisted_environment(ATTEMPT_ENV_ALLOWLIST, environment)
    attempt.setdefault("RUN_ATTEMPT_ID", uuid.uuid4().hex)
    attempt.setdefault("RUN_HOSTNAME", socket.gethostname())
    return {
        "schema_version": SCHEMA_VERSION,
        "started_at": _utc_now(),
        "mode": mode,
        "run_id": identity["RUN_ID"],
        "attempt": attempt,
    }


def initialize_manifest(
    run_dir: os.PathLike[str] | str,
    *,
    mode: str,
    repo: os.PathLike[str] | str | None = None,
    source_id: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if mode not in {"new", "resume"}:
        raise StorageError(f"manifest mode must be new or resume, got {mode!r}")
    environment = os.environ if environment is None else environment
    directory = Path(run_dir).expanduser()
    identity = run_identity(environment)
    provenance = _source_provenance(
        repo=repo, source_id=source_id, environment=environment
    )
    manifest_path = directory / MANIFEST_NAME

    if mode == "new":
        if directory.exists():
            if directory.is_symlink() or not directory.is_dir():
                raise StorageError(f"run directory is not a real directory: {directory}")
            entries = list(directory.iterdir())
            if entries:
                raise StorageError(
                    f"new run refuses non-empty directory collision: {directory}"
                )
        else:
            directory.mkdir(parents=True)
        manifest = _new_manifest(identity, provenance, legacy_adopted=False)
        _atomic_write_json(manifest_path, manifest, mode=0o440, exclusive=True)
    else:
        if not directory.exists() or directory.is_symlink() or not directory.is_dir():
            raise StorageError(f"resume requires an existing run directory: {directory}")
        if manifest_path.exists() or manifest_path.is_symlink():
            manifest = validate_manifest(
                directory,
                repo=repo,
                source_id=source_id,
                environment=environment,
            )
        else:
            entries = list(directory.iterdir())
            if not entries:
                raise StorageError(
                    f"resume refuses empty directory without a manifest: {directory}"
                )
            allow_legacy = _parse_bool(
                environment.get("RUN_ALLOW_LEGACY_RESUME", "false"),
                name="RUN_ALLOW_LEGACY_RESUME",
            )
            if not allow_legacy:
                raise StorageError(
                    "legacy run has no run_manifest.json; set "
                    "RUN_ALLOW_LEGACY_RESUME=true for an explicit one-time adoption"
                )
            manifest = _new_manifest(identity, provenance, legacy_adopted=True)
            _atomic_write_json(manifest_path, manifest, mode=0o440, exclusive=True)

    _append_json_line(
        directory / ATTEMPTS_NAME,
        _attempt_record(mode=mode, identity=identity, environment=environment),
    )
    return manifest


def _event_lock(run_dir: Path):
    lock_path = run_dir / ".run_status.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o640)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise StorageError(f"event lock is not a regular file: {lock_path}")
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    return descriptor


def _load_status(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / STATUS_NAME
    try:
        value = _read_json_file(path)
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise StorageError(f"run status is not a JSON object: {path}")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise StorageError(f"unsupported run status schema: {path}")
    return value


def _last_event_sequence(path: Path) -> int:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return 0
    if not stat.S_ISREG(metadata.st_mode):
        raise StorageError(f"event log is not a regular file: {path}")
    last = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    last = max(last, int(record.get("sequence", 0)))
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    raise StorageError(f"invalid append-only event record in {path}") from exc
    except (OSError, UnicodeError) as exc:
        raise StorageError(f"cannot read event log {path}: {exc}") from exc
    return last


def _event_context(
    run_dir: Path, environment: Mapping[str, str]
) -> tuple[dict[str, Any], str, str | None]:
    manifest = _load_manifest(run_dir)
    manifest_run_id = manifest["identity"].get("RUN_ID")
    environment_run_id = environment.get("RUN_ID")
    if environment_run_id and environment_run_id != manifest_run_id:
        raise StorageError(
            f"event RUN_ID {environment_run_id!r} does not match manifest "
            f"{manifest_run_id!r}"
        )
    attempt_id = environment.get("RUN_ATTEMPT_ID")
    return manifest, manifest_run_id, attempt_id


def _validated_fields(items: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise StorageError(f"event field must use KEY=VALUE syntax: {item!r}")
        key, value = item.split("=", 1)
        if not _FIELD_PATTERN.fullmatch(key):
            raise StorageError(f"invalid event field name: {key!r}")
        if key in result:
            raise StorageError(f"duplicate event field: {key}")
        result[key] = value
    return result


def append_event(
    run_dir: os.PathLike[str] | str,
    *,
    event: str,
    status_value: str | None = None,
    fields: Mapping[str, str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if not _NAME_PATTERN.fullmatch(event):
        raise StorageError(f"invalid event name: {event!r}")
    if status_value is not None and not _NAME_PATTERN.fullmatch(status_value):
        raise StorageError(f"invalid status value: {status_value!r}")
    environment = os.environ if environment is None else environment
    directory = Path(run_dir).expanduser()
    _, run_id, attempt_id = _event_context(directory, environment)
    descriptor = _event_lock(directory)
    try:
        _repair_jsonl_tail(directory / EVENTS_NAME)
        previous = _load_status(directory)
        previous_sequence = int(previous.get("sequence", 0)) if previous else 0
        sequence = max(
            previous_sequence, _last_event_sequence(directory / EVENTS_NAME)
        ) + 1
        timestamp = _utc_now()
        delta_fields = dict(fields or {})
        event_record = {
            "schema_version": SCHEMA_VERSION,
            "sequence": sequence,
            "timestamp": timestamp,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "event": event,
            "status": status_value,
            "fields": delta_fields,
        }
        _append_json_line(directory / EVENTS_NAME, event_record)

        merged_fields = dict(previous.get("fields", {})) if previous else {}
        merged_fields.update(delta_fields)
        status_record = {
            "schema_version": SCHEMA_VERSION,
            "created_at": previous.get("created_at", timestamp) if previous else timestamp,
            "updated_at": timestamp,
            "sequence": sequence,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "event": event,
            "status": status_value
            if status_value is not None
            else (previous.get("status") if previous else event),
            "fields": merged_fields,
        }
        _atomic_write_json(directory / STATUS_NAME, status_record)
        return status_record
    finally:
        os.close(descriptor)


def update_status(
    run_dir: os.PathLike[str] | str,
    *,
    status_value: str,
    fields: Mapping[str, str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if not _NAME_PATTERN.fullmatch(status_value):
        raise StorageError(f"invalid status value: {status_value!r}")
    environment = os.environ if environment is None else environment
    directory = Path(run_dir).expanduser()
    _, run_id, attempt_id = _event_context(directory, environment)
    descriptor = _event_lock(directory)
    try:
        _repair_jsonl_tail(directory / EVENTS_NAME)
        previous = _load_status(directory)
        timestamp = _utc_now()
        sequence = max(
            int(previous.get("sequence", 0)) if previous else 0,
            _last_event_sequence(directory / EVENTS_NAME),
        ) + 1
        merged_fields = dict(previous.get("fields", {})) if previous else {}
        merged_fields.update(dict(fields or {}))
        status_record = {
            "schema_version": SCHEMA_VERSION,
            "created_at": previous.get("created_at", timestamp) if previous else timestamp,
            "updated_at": timestamp,
            "sequence": sequence,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "event": previous.get("event") if previous else None,
            "status": status_value,
            "fields": merged_fields,
        }
        _atomic_write_json(directory / STATUS_NAME, status_record)
        return status_record
    finally:
        os.close(descriptor)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _real_directory(path: os.PathLike[str] | str, *, label: str) -> Path:
    requested = Path(path).expanduser().absolute()
    try:
        metadata = requested.lstat()
    except OSError as exc:
        raise StorageError(f"{label} does not exist: {requested}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise StorageError(f"{label} must be a real directory, not a symlink: {requested}")
    return requested.resolve(strict=True)


def _symlinks_under(root: Path) -> dict[Path, str]:
    result: dict[Path, str] = {}
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        names = list(directory_names) + list(file_names)
        for name in names:
            path = current / name
            try:
                if path.is_symlink():
                    relative = path.relative_to(root)
                    result[relative] = os.readlink(path)
                    if name in directory_names:
                        directory_names.remove(name)
            except OSError as exc:
                raise StorageError(f"cannot inspect symlink {path}: {exc}") from exc
    return result


def _validate_tree_entry_types(root: Path, *, label: str) -> None:
    """Reject sockets, devices, FIFOs, and other non-portable run entries."""

    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in list(directory_names) + list(file_names):
            path = current / name
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise StorageError(f"cannot inspect {label} entry {path}: {exc}") from exc
            if (
                stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISREG(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
            ):
                continue
            raise StorageError(
                f"{label} contains unsupported special file: {path}"
            )


def _resolved_symlink(path: Path) -> Path:
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise StorageError(f"broken or cyclic symlink: {path} -> {os.readlink(path)!r}") from exc


def _atomic_materialize_file(source: Path, destination: Path) -> None:
    temporary = destination.parent / (
        f".{destination.name}.materialize.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        with source.open("rb") as source_handle:
            source_metadata = os.fstat(source_handle.fileno())
            if not stat.S_ISREG(source_metadata.st_mode):
                raise StorageError(f"symlink target is not a regular file: {source}")
            with temporary.open("xb") as destination_handle:
                shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
        shutil.copystat(source, temporary, follow_symlinks=True)
        source_mode = stat.S_IMODE(source_metadata.st_mode)
        durable_mode = (source_mode | 0o660) & ~0o007
        if source_mode & 0o111:
            durable_mode |= 0o110
        temporary.chmod(durable_mode)
        # copystat changes mode/timestamps after the data fsync above.
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def materialize_links(
    source: os.PathLike[str] | str,
    stage: os.PathLike[str] | str,
    *,
    allowed_roots: Sequence[os.PathLike[str] | str] = (),
) -> dict[str, int]:
    """Make a staged run tree self-contained without flattening internal links."""

    source_root = _real_directory(source, label="source")
    stage_root = _real_directory(stage, label="stage")
    if (
        source_root == stage_root
        or os.path.samefile(source_root, stage_root)
        or _is_within(source_root, stage_root)
        or _is_within(stage_root, source_root)
    ):
        raise StorageError("source and stage must be separate, non-nested trees")
    allowed = [
        _real_directory(root, label="allowed root") for root in allowed_roots
    ]

    _validate_tree_entry_types(source_root, label="source")
    _validate_tree_entry_types(stage_root, label="stage")
    source_links = _symlinks_under(source_root)
    stage_links = _symlinks_under(stage_root)
    preserved: set[Path] = set()
    materialized = 0

    for relative, link_text in sorted(source_links.items(), key=lambda item: str(item[0])):
        source_link = source_root / relative
        stage_link = stage_root / relative
        if relative not in stage_links:
            raise StorageError(f"staged copy omitted source symlink: {relative}")
        if stage_links[relative] != link_text:
            raise StorageError(
                f"staged symlink differs from source: {relative}: "
                f"{stage_links[relative]!r} != {link_text!r}"
            )

        target = _resolved_symlink(source_link)
        is_relative_internal = not os.path.isabs(link_text) and _is_within(
            target, source_root
        )
        if is_relative_internal:
            staged_target = _resolved_symlink(stage_link)
            if not _is_within(staged_target, stage_root):
                raise StorageError(f"relative staged symlink escapes stage: {relative}")
            preserved.add(relative)
            continue

        authorized = _is_within(target, source_root) or any(
            _is_within(target, root) for root in allowed
        )
        if not authorized:
            raise StorageError(
                f"external symlink target is outside allowed roots: "
                f"{source_link} -> {target}"
            )
        try:
            target_metadata = target.stat()
        except OSError as exc:
            raise StorageError(f"cannot stat symlink target {target}: {exc}") from exc
        if stat.S_ISDIR(target_metadata.st_mode):
            raise StorageError(
                f"external or absolute directory symlink is forbidden: "
                f"{source_link} -> {target}"
            )
        if not stat.S_ISREG(target_metadata.st_mode):
            raise StorageError(
                f"external symlink target is not a regular file: {source_link} -> {target}"
            )
        _atomic_materialize_file(target, stage_link)
        materialized += 1

    remaining_stage_links = _symlinks_under(stage_root)
    if set(remaining_stage_links) != preserved:
        unexpected = sorted(set(remaining_stage_links) - preserved, key=str)
        missing = sorted(preserved - set(remaining_stage_links), key=str)
        raise StorageError(
            "staged symlink inventory mismatch after materialization: "
            f"unexpected={[str(x) for x in unexpected]}, "
            f"missing={[str(x) for x in missing]}"
        )
    for relative, link_text in remaining_stage_links.items():
        if os.path.isabs(link_text):
            raise StorageError(f"absolute symlink remains in stage: {relative}")
        resolved = _resolved_symlink(stage_root / relative)
        if not _is_within(resolved, stage_root):
            raise StorageError(f"symlink escapes staged tree: {relative}")

    return {
        "source_symlinks": len(source_links),
        "preserved_internal": len(preserved),
        "materialized_files": materialized,
    }


_CATALOG_ARTIFACTS = (
    "dreamer_config.yaml",
    "config.yaml",
    "metrics.jsonl",
    "scores.jsonl",
    "analysis/manifest.json",
    f"source_provenance/{SOURCE_SNAPSHOT_METADATA_NAME}",
    f"source_provenance/{SOURCE_SNAPSHOT_DIFF_NAME}",
    f"source_provenance/{SOURCE_SNAPSHOT_UNTRACKED_NAME}",
    ATTEMPTS_NAME,
    WANDB_RECEIPT_NAME,
    EVENTS_NAME,
    STATUS_NAME,
    ".sync_complete.json",
    "stdout.log",
    "stderr.log",
)


def _entry_type(path: Path) -> tuple[str, int | None]:
    """Return a path's lstat type and regular-file size without following links."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return "missing", None
    except OSError:
        return "unreadable", None
    if stat.S_ISREG(metadata.st_mode):
        return "regular", metadata.st_size
    if stat.S_ISDIR(metadata.st_mode):
        return "directory", None
    if stat.S_ISLNK(metadata.st_mode):
        return "symlink", None
    return "special", None


def _catalog_manifest(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Inspect a manifest's shape without comparing it to the current process."""

    path = run_dir / MANIFEST_NAME
    errors: list[str] = []
    identity: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    manifest: Any = None
    try:
        manifest = _read_json_file(path)
    except (OSError, StorageError) as exc:
        errors.append(str(exc))

    if manifest is not None:
        if not isinstance(manifest, dict):
            errors.append("manifest must be a JSON object")
        else:
            if manifest.get("schema_version") != SCHEMA_VERSION:
                errors.append(
                    f"unsupported schema_version {manifest.get('schema_version')!r}"
                )
            if not isinstance(manifest.get("created_at"), str):
                errors.append("created_at must be a string")
            if not isinstance(manifest.get("legacy_adopted"), bool):
                errors.append("legacy_adopted must be a boolean")

            raw_identity = manifest.get("identity")
            if not isinstance(raw_identity, dict):
                errors.append("identity must be an object")
            else:
                identity = dict(raw_identity)
                unknown = sorted(set(identity) - set(IDENTITY_ENV_ALLOWLIST))
                if unknown:
                    errors.append(f"identity contains unsupported keys: {unknown!r}")
                if any(not isinstance(key, str) or not isinstance(value, str)
                       for key, value in identity.items()):
                    errors.append("identity keys and values must be strings")
                run_id = identity.get("RUN_ID")
                if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
                    errors.append("identity has an invalid or missing RUN_ID")

            raw_provenance = manifest.get("provenance")
            if not isinstance(raw_provenance, dict):
                errors.append("provenance must be an object")
            else:
                provenance = dict(raw_provenance)
                if set(provenance) != {"commit", "dirty", "source_id"}:
                    errors.append(
                        "provenance must contain exactly commit, dirty, and source_id"
                    )
                source_id = provenance.get("source_id")
                if not isinstance(source_id, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", source_id
                ):
                    errors.append("provenance has an invalid or missing source_id")
                commit = provenance.get("commit")
                if commit is not None and (
                    not isinstance(commit, str)
                    or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit)
                ):
                    errors.append("provenance commit must be a Git object ID or null")
                dirty = provenance.get("dirty")
                if dirty is not None and not isinstance(dirty, bool):
                    errors.append("provenance dirty must be boolean or null")

            config_sha = manifest.get("resolved_config_sha256")
            if config_sha is not None and (
                not isinstance(config_sha, str)
                or not re.fullmatch(r"[0-9a-f]{64}", config_sha)
            ):
                errors.append("resolved_config_sha256 must be a SHA-256 digest")

    summary = {
        "present": True,
        "valid": not errors,
        "schema_version": manifest.get("schema_version")
        if isinstance(manifest, dict)
        else None,
        "created_at": manifest.get("created_at")
        if isinstance(manifest, dict)
        else None,
        "legacy_adopted": manifest.get("legacy_adopted")
        if isinstance(manifest, dict)
        else None,
        "resolved_config_sha256": manifest.get("resolved_config_sha256")
        if isinstance(manifest, dict)
        else None,
        "errors": errors,
    }
    return summary, identity, provenance


_RECOVERY_STORE_PATTERN = re.compile(r"^\.(?P<run_name>.+)\.recovery$")
_RECOVERY_GENERATION_PATTERN = re.compile(
    r"^snapshot-(?P<published_ns>[0-9]{19})-[0-9]+-[0-9]+$"
)
_RECOVERY_STAGE_PATTERN = re.compile(r"^\.stage-[0-9]+-[0-9]+-[0-9]{19}$")
_TRANSACTION_RESIDUE_PATTERNS = (
    (re.compile(r"^\.(?P<run_name>.+)\.publish-new$"), "publish_stage"),
    (re.compile(r"^\.(?P<run_name>.+)\.publish-previous$"), "publish_previous"),
    (
        re.compile(
            r"^\.(?P<run_name>.+)\.publish-incomplete-"
            r"[0-9]{8}T[0-9]{15,32}_[0-9]+_[0-9]+$"
        ),
        "publish_incomplete",
    ),
    (
        re.compile(r"^\.(?P<run_name>.+)\.recovery-index-new$"),
        "recovery_index_stage",
    ),
)
_SCRATCH_RECOVERY_ARCHIVE_PATTERN = re.compile(
    r"^(?P<run_name>.+)\.scratch-recovery-[0-9]{8}T[0-9]{15,32}_[0-9]+"
    r"(?P<stage>\.publish-new)?$"
)


def _catalog_hash_regular(path: Path) -> tuple[int, str]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise StorageError(f"critical artifact is not a regular file: {path}")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise StorageError(f"critical artifact changed while hashing: {path}")
        return before.st_size, digest.hexdigest()
    finally:
        os.close(descriptor)


def _catalog_add_critical_tree(root: Path, relative: str, paths: set[str]) -> None:
    base = root.joinpath(*Path(relative).parts)
    entry_type, _ = _entry_type(base)
    if entry_type == "missing":
        return
    if entry_type != "directory":
        raise StorageError(f"critical artifact tree is unsafe: {relative}")
    for directory, directory_names, file_names in os.walk(base, followlinks=False):
        current = Path(directory)
        for name in directory_names:
            if _entry_type(current / name)[0] != "directory":
                raise StorageError(
                    f"critical artifact tree contains an unsafe directory: "
                    f"{current / name}"
                )
        for name in file_names:
            child = current / name
            if _entry_type(child)[0] != "regular":
                raise StorageError(
                    f"critical artifact tree contains a non-file: {child}"
                )
            paths.add(child.relative_to(root).as_posix())


def _catalog_critical_inventory(
    root: Path, *, scope: str = CRITICAL_INTEGRITY_SCOPE_CURRENT
) -> tuple[list[dict[str, Any]], str | None]:
    if not isinstance(scope, str) or scope not in CRITICAL_INTEGRITY_SCOPES:
        raise StorageError(
            f"unsupported critical artifact integrity scope: {scope!r}"
        )
    if _entry_type(root / MANIFEST_NAME)[0] != "regular":
        raise StorageError("recovery run_manifest.json is missing or unsafe")
    paths = {MANIFEST_NAME}
    optional_artifacts = [ATTEMPTS_NAME]
    # Version 2 also authenticates the initialization receipt.
    if scope == CRITICAL_INTEGRITY_SCOPE_V2:
        optional_artifacts.append(WANDB_RECEIPT_NAME)
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
            EVENTS_NAME,
            STATUS_NAME,
            SOURCE_SNAPSHOT_METADATA_NAME,
            f"source_provenance/{SOURCE_SNAPSHOT_METADATA_NAME}",
        )
    )
    for relative in optional_artifacts:
        entry_type, _ = _entry_type(root.joinpath(*Path(relative).parts))
        if entry_type == "missing":
            continue
        if entry_type != "regular":
            raise StorageError(f"critical recovery artifact is unsafe: {relative}")
        paths.add(relative)
    _catalog_add_critical_tree(root, ".hydra", paths)
    _catalog_add_critical_tree(root, "analysis", paths)

    checkpoint_name: str | None = None
    checkpoint_root = root / "checkpoint.pkl"
    checkpoint_type, _ = _entry_type(checkpoint_root)
    if checkpoint_type != "missing":
        if checkpoint_type != "directory":
            raise StorageError("recovery checkpoint.pkl is not a real directory")
        latest = checkpoint_root / "latest"
        latest_type, latest_size = _entry_type(latest)
        if latest_type != "regular" or latest_size is None or latest_size > 4096:
            raise StorageError("recovery checkpoint latest pointer is missing/unsafe")
        checkpoint_name = latest.read_text(encoding="utf-8").strip()
        if not _CHECKPOINT_NAME_PATTERN.fullmatch(checkpoint_name):
            raise StorageError("recovery checkpoint latest pointer is invalid")
        generation = checkpoint_root / checkpoint_name
        if _entry_type(generation)[0] != "directory":
            raise StorageError("recovery checkpoint generation is missing/unsafe")
        if _entry_type(generation / "done")[0] != "regular":
            raise StorageError("recovery checkpoint generation has no done marker")
        paths.add("checkpoint.pkl/latest")
        _catalog_add_critical_tree(
            root, f"checkpoint.pkl/{checkpoint_name}", paths
        )
    elif _entry_type(root / "latest.pt")[0] == "regular":
        paths.add("latest.pt")
    _catalog_add_critical_tree(root, "checkpoint.pkl.incomplete", paths)
    # Verify archived checkpoint inventories when present.
    _catalog_add_critical_tree(root, "checkpoint.pkl.archive", paths)

    artifacts: list[dict[str, Any]] = []
    for relative in sorted(paths):
        size, sha256 = _catalog_hash_regular(root.joinpath(*Path(relative).parts))
        artifacts.append(
            {"path": relative, "size_bytes": size, "sha256": sha256}
        )
    return artifacts, checkpoint_name


def _catalog_integrity_marker(
    root: Path,
    marker_name: str,
    *,
    marker_kind: str,
    expected_generation: str | None = None,
) -> dict[str, Any]:
    """Offline-verify a final-publish or periodic-recovery integrity marker."""

    marker_path = root / marker_name
    marker_type, _ = _entry_type(marker_path)
    errors: list[str] = []
    marker: Any = None
    if marker_type == "missing":
        errors.append(f"{marker_name} is missing")
    elif marker_type != "regular":
        errors.append(f"{marker_name} is {marker_type}, not a regular file")
    else:
        try:
            marker = _read_json_file(marker_path)
        except (OSError, StorageError) as exc:
            errors.append(str(exc))

    common_fields = {
        "schema_version",
        "marker_kind",
        "checkpoint",
        "run_uid",
        "integrity",
    }
    kind_fields = (
        {"completed_at", "completed_ns"}
        if marker_kind == "final-publish"
        else {"published_at", "published_ns", "generation"}
    )
    if marker is not None:
        if not isinstance(marker, dict):
            errors.append(f"{marker_name} must contain a JSON object")
            marker = None
        else:
            if set(marker) != common_fields | kind_fields:
                errors.append(f"{marker_name} has unsupported or missing fields")
            if marker.get("schema_version") != 2:
                errors.append(f"{marker_name} has an unsupported schema")
            if marker.get("marker_kind") != marker_kind:
                errors.append(f"{marker_name} has the wrong marker kind")
            run_uid = marker.get("run_uid")
            if not isinstance(run_uid, str) or not run_uid:
                errors.append(f"{marker_name} has an invalid run_uid")
            checkpoint = marker.get("checkpoint")
            if checkpoint is not None and (
                not isinstance(checkpoint, str)
                or not _CHECKPOINT_NAME_PATTERN.fullmatch(checkpoint)
            ):
                errors.append(f"{marker_name} has an invalid checkpoint")

            timestamp_key = (
                "completed_at" if marker_kind == "final-publish" else "published_at"
            )
            nanoseconds_key = (
                "completed_ns" if marker_kind == "final-publish" else "published_ns"
            )
            timestamp = marker.get(timestamp_key)
            if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
                errors.append(f"{marker_name} has an invalid {timestamp_key}")
            nanoseconds = marker.get(nanoseconds_key)
            if not isinstance(nanoseconds, str) or not re.fullmatch(
                r"[0-9]{19}", nanoseconds
            ):
                errors.append(f"{marker_name} has an invalid {nanoseconds_key}")

            if marker_kind == "periodic-recovery":
                generation = marker.get("generation")
                generation_match = (
                    _RECOVERY_GENERATION_PATTERN.fullmatch(generation)
                    if isinstance(generation, str)
                    else None
                )
                if generation_match is None:
                    errors.append(f"{marker_name} has an invalid generation")
                else:
                    if expected_generation is not None and generation != expected_generation:
                        errors.append(
                            "recovery marker generation does not match its directory"
                        )
                    if nanoseconds != generation_match.group("published_ns"):
                        errors.append(
                            "recovery marker timestamp does not match its generation"
                        )

            integrity = marker.get("integrity")
            if not isinstance(integrity, dict):
                errors.append(f"{marker_name} integrity inventory is missing")
            else:
                scope = integrity.get("scope")
                if (
                    not isinstance(scope, str)
                    or scope not in CRITICAL_INTEGRITY_SCOPES
                ):
                    errors.append(f"{marker_name} has an unsupported integrity scope")
                else:
                    try:
                        expected_artifacts, actual_checkpoint = (
                            _catalog_critical_inventory(root, scope=scope)
                        )
                    except (OSError, StorageError, UnicodeError) as exc:
                        errors.append(str(exc))
                    else:
                        expected_integrity = {
                            "algorithm": "sha256",
                            "scope": scope,
                            "artifacts": expected_artifacts,
                        }
                        if integrity != expected_integrity:
                            errors.append("critical artifact integrity check failed")
                        if marker.get("checkpoint") != actual_checkpoint:
                            errors.append(
                                f"{marker_name} checkpoint does not match the artifact tree"
                            )

    return {
        "state": "valid" if marker is not None and not errors else (
            "missing" if marker_type == "missing" else "invalid"
        ),
        "present": marker_type != "missing",
        "valid": marker is not None and not errors,
        "path": os.fspath(marker_path),
        "schema_version": marker.get("schema_version")
        if isinstance(marker, dict)
        else None,
        "marker_kind": marker.get("marker_kind")
        if isinstance(marker, dict)
        else None,
        "checkpoint": marker.get("checkpoint")
        if isinstance(marker, dict)
        else None,
        "run_uid": marker.get("run_uid") if isinstance(marker, dict) else None,
        "completed_at": marker.get("completed_at")
        if isinstance(marker, dict)
        else None,
        "completed_ns": marker.get("completed_ns")
        if isinstance(marker, dict)
        else None,
        "published_at": marker.get("published_at")
        if isinstance(marker, dict)
        else None,
        "published_ns": marker.get("published_ns")
        if isinstance(marker, dict)
        else None,
        "generation": marker.get("generation")
        if isinstance(marker, dict)
        else None,
        "errors": errors,
    }


def _catalog_sync_complete(run_dir: Path) -> dict[str, Any]:
    return _catalog_integrity_marker(
        run_dir, ".sync_complete.json", marker_kind="final-publish"
    )


def _catalog_recovery_generation(path: Path, *, stage: bool) -> dict[str, Any]:
    expected_generation = None if stage else path.name
    marker = _catalog_integrity_marker(
        path,
        ".recovery_complete.json",
        marker_kind="periodic-recovery",
        expected_generation=expected_generation,
    )
    checkpoint_step: int | None = None
    if marker["valid"] and isinstance(marker["checkpoint"], str):
        try:
            checkpoint_step = _checkpoint_step_without_unpickling(
                path, marker["checkpoint"]
            )
        except (OSError, StorageError):
            # Runs without a scalar step may be cataloged but cannot outrank a decoded
            # checkpoint for resume.
            checkpoint_step = None
    return {
        "kind": "recovery_stage" if stage else "recovery_generation",
        "directory_name": path.name,
        "path": os.fspath(path),
        "valid": marker["valid"],
        "generation": marker["generation"],
        "checkpoint": marker["checkpoint"],
        "checkpoint_step": checkpoint_step,
        "published_at": marker["published_at"],
        "published_ns": marker["published_ns"],
        "run_uid": marker["run_uid"],
        "errors": marker["errors"],
    }


def _catalog_recovery_index(
    run_dir: Path, expected_store: Path, *, expected_run_uid: str | None
) -> dict[str, Any]:
    path = run_dir / ".recovery_available.json"
    entry_type, _ = _entry_type(path)
    errors: list[str] = []
    value: Any = None
    if entry_type == "missing":
        return {
            "present": False,
            "valid": None,
            "path": os.fspath(path),
            "run_uid": None,
            "errors": [],
        }
    if entry_type != "regular":
        errors.append("recovery index is not a regular file")
    else:
        try:
            value = _read_json_file(path)
        except (OSError, StorageError) as exc:
            errors.append(str(exc))
    if value is not None:
        if not isinstance(value, dict):
            errors.append("recovery index must contain a JSON object")
            value = None
        else:
            if set(value) != {
                "schema_version",
                "state",
                "recovery_directory",
                "current_pointer",
                "run_uid",
            }:
                errors.append("recovery index has unsupported or missing fields")
            if value.get("schema_version") != 1:
                errors.append("recovery index has an unsupported schema")
            if value.get("state") != "DURABLE_RECOVERY_AVAILABLE":
                errors.append("recovery index has an invalid state")
            if value.get("recovery_directory") != expected_store.name:
                errors.append("recovery index names the wrong recovery directory")
            if value.get("current_pointer") != f"{expected_store.name}/current":
                errors.append("recovery index names the wrong current pointer")
            run_uid = value.get("run_uid")
            if not isinstance(run_uid, str) or not run_uid:
                errors.append("recovery index has an invalid run_uid")
            elif expected_run_uid is not None and run_uid != expected_run_uid:
                errors.append("recovery index run_uid does not match its snapshot")
    return {
        "present": entry_type != "missing",
        "valid": value is not None and not errors,
        "path": os.fspath(path),
        "run_uid": value.get("run_uid") if isinstance(value, dict) else None,
        "errors": errors,
    }


def _empty_recovery(run_dir: Path) -> dict[str, Any]:
    expected_store = run_dir.parent / f".{run_dir.name}.recovery"
    return {
        "state": "none",
        "valid": True,
        "recoverable": False,
        "store_path": None,
        "canonical_path": os.fspath(run_dir),
        "pointer_generation": None,
        "pointer_valid": None,
        "fallback_used": False,
        "generation": None,
        "snapshot_path": None,
        "checkpoint": None,
        "published_at": None,
        "published_ns": None,
        "run_uid": None,
        "indexed": False,
        "index": _catalog_recovery_index(
            run_dir, expected_store, expected_run_uid=None
        ),
        "high_water": {
            "present": False,
            "valid": None,
            "path": os.fspath(expected_store / CHECKPOINT_HIGH_WATER_NAME),
            "high_step": None,
            "resume_floor_step": None,
            "resume_floor_checkpoint": None,
            "run_uid": None,
            "errors": [],
        },
        "generations": [],
        "stages": [],
        "store_residues": [],
        "errors": [],
    }


def _recovery_generation_sort_key(
    value: Mapping[str, Any],
) -> tuple[int, tuple[int, int, int, Any], int, str]:
    checkpoint_step = value.get("checkpoint_step")
    checkpoint = value.get("checkpoint")
    published_ns = value.get("published_ns")
    return (
        checkpoint_step if isinstance(checkpoint_step, int) else -1,
        (
            _checkpoint_identity_order_key(checkpoint)
            if isinstance(checkpoint, str)
            else (-1, -1, -1, "")
        ),
        int(published_ns) if isinstance(published_ns, str) and published_ns.isdigit() else -1,
        str(value.get("directory_name", "")),
    )


def _catalog_recovery_store(store: Path) -> tuple[dict[str, Any], Path | None]:
    """Catalog a recovery store and retain a valid fallback if `current` broke."""

    errors: list[str] = []
    generations: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    store_residues: list[dict[str, Any]] = []
    pointer_generation: str | None = None
    pointer_valid = False
    match = _RECOVERY_STORE_PATTERN.fullmatch(store.name)
    if not match or _entry_type(store)[0] != "directory":
        canonical = store.parent / store.name
        result = _empty_recovery(canonical)
        result.update(
            {
                "state": "invalid",
                "valid": False,
                "store_path": os.fspath(store),
                "canonical_path": None,
                "errors": ["recovery store path is unsafe or misnamed"],
            }
        )
        return result, None

    canonical = store.parent / match.group("run_name")
    generation_root = store / "generations"
    generation_root_type, _ = _entry_type(generation_root)
    if generation_root_type == "directory":
        try:
            children = sorted(generation_root.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            children = []
            errors.append(f"cannot enumerate recovery generations: {exc}")
        for child in children:
            child_type, _ = _entry_type(child)
            generation_match = _RECOVERY_GENERATION_PATTERN.fullmatch(child.name)
            stage_match = _RECOVERY_STAGE_PATTERN.fullmatch(child.name)
            if child_type == "directory" and generation_match:
                generations.append(_catalog_recovery_generation(child, stage=False))
            elif child_type == "directory" and stage_match:
                stages.append(_catalog_recovery_generation(child, stage=True))
            else:
                store_residues.append(
                    {
                        "kind": "recovery_unexpected_entry",
                        "path": os.fspath(child),
                        "entry_type": child_type,
                        "valid": False,
                        "errors": ["unexpected or unsafe recovery-generation entry"],
                    }
                )
    elif generation_root_type == "missing":
        errors.append("recovery generations directory is missing")
    else:
        errors.append("recovery generations path is unsafe")

    try:
        store_children = sorted(store.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        store_children = []
        errors.append(f"cannot enumerate recovery store: {exc}")
    for child in store_children:
        if child.name in {"current", "generations", CHECKPOINT_HIGH_WATER_NAME}:
            continue
        child_type, _ = _entry_type(child)
        if child.name == "quarantine" and child_type == "directory":
            try:
                quarantined = sorted(child.iterdir(), key=lambda item: item.name)
            except OSError as exc:
                errors.append(f"cannot enumerate recovery quarantine: {exc}")
                quarantined = []
            for entry in quarantined:
                entry_type, _ = _entry_type(entry)
                safe_entry = entry_type in {"directory", "regular"}
                store_residues.append(
                    {
                        "kind": "recovery_quarantine",
                        "path": os.fspath(entry),
                        "entry_type": entry_type,
                        "valid": safe_entry,
                        "errors": []
                        if safe_entry
                        else ["unsafe recovery quarantine entry"],
                    }
                )
            continue
        is_pointer_stage = bool(re.fullmatch(r"\.current\.tmp\..+", child.name))
        is_high_water_stage = bool(
            re.fullmatch(r"\.checkpoint-high-water\.json\.tmp\..+", child.name)
        )
        store_residues.append(
            {
                "kind": (
                    "recovery_pointer_stage"
                    if is_pointer_stage
                    else "checkpoint_high_water_stage"
                    if is_high_water_stage
                    else "recovery_unexpected_entry"
                ),
                "path": os.fspath(child),
                "entry_type": child_type,
                "valid": (is_pointer_stage or is_high_water_stage)
                and child_type == "regular",
                "errors": []
                if (is_pointer_stage or is_high_water_stage)
                and child_type == "regular"
                else ["unexpected or unsafe recovery-store entry"],
            }
        )

    pointer = store / "current"
    pointer_type, pointer_size = _entry_type(pointer)
    if pointer_type != "regular" or pointer_size is None or pointer_size > 256:
        errors.append("recovery current pointer is missing or unsafe")
    else:
        try:
            pointer_text = pointer.read_text(encoding="ascii")
        except (OSError, UnicodeError) as exc:
            errors.append(f"cannot read recovery current pointer: {exc}")
        else:
            pointer_match = re.fullmatch(
                r"(snapshot-[0-9]{19}-[0-9]+-[0-9]+)\n?", pointer_text
            )
            if pointer_match is None:
                errors.append("recovery current pointer has an invalid generation")
            else:
                pointer_generation = pointer_match.group(1)

    selected: dict[str, Any] | None = None
    if pointer_generation is not None:
        pointed = next(
            (
                generation
                for generation in generations
                if generation["directory_name"] == pointer_generation
            ),
            None,
        )
        if pointed is None:
            errors.append("recovery current generation is missing or unsafe")
        elif not pointed["valid"]:
            errors.extend(
                f"current generation: {message}" for message in pointed["errors"]
            )
        else:
            pointer_valid = True
            selected = pointed

    fallback_used = False
    valid_generations = [value for value in generations if value["valid"]]
    if selected is None:
        if valid_generations:
            selected = max(valid_generations, key=_recovery_generation_sort_key)
            fallback_used = True
    elif valid_generations:
        # Select by checkpoint progress because an interrupted publication can leave a
        # stale current pointer.
        semantic_best = max(valid_generations, key=_recovery_generation_sort_key)
        selected_step = selected.get("checkpoint_step")
        semantic_best_step = semantic_best.get("checkpoint_step")
        selected_checkpoint = selected.get("checkpoint")
        semantic_best_checkpoint = semantic_best.get("checkpoint")
        if (
            isinstance(semantic_best_step, int)
            and (
                not isinstance(selected_step, int)
                or not isinstance(selected_checkpoint, str)
                or (
                    isinstance(semantic_best_checkpoint, str)
                    and _checkpoint_point_key(
                        semantic_best_step, semantic_best_checkpoint
                    )
                    > _checkpoint_point_key(selected_step, selected_checkpoint)
                )
            )
        ):
            errors.append(
                "recovery current pointer is behind the highest checkpoint progress"
            )
            selected = semantic_best
            fallback_used = True

    if store_residues:
        errors.append("recovery store contains transaction residue or unsafe entries")

    selected_path = Path(selected["path"]) if selected is not None else None
    high_water_path = store / CHECKPOINT_HIGH_WATER_NAME
    high_water: dict[str, Any] = {
        "present": _entry_type(high_water_path)[0] != "missing",
        "valid": None,
        "path": os.fspath(high_water_path),
        "high_step": None,
        "resume_floor_step": None,
        "resume_floor_checkpoint": None,
        "run_uid": None,
        "errors": [],
    }
    if high_water["present"]:
        try:
            high_water_value = load_checkpoint_high_water(
                high_water_path,
                run_dir=selected_path,
                expected_run_id=selected.get("run_uid") if selected else None,
            )
        except (OSError, StorageError) as exc:
            high_water["valid"] = False
            high_water["errors"].append(str(exc))
            errors.append(f"checkpoint high-water marker: {exc}")
        else:
            assert high_water_value is not None
            high_water.update(
                {
                    "valid": True,
                    "high_step": high_water_value["high_step"],
                    "resume_floor_step": high_water_value["resume_floor_step"],
                    "resume_floor_checkpoint": _checkpoint_floor_identity(
                        high_water_value
                    ),
                    "run_uid": high_water_value["run_uid"],
                }
            )
    selected_usable = selected is not None
    if high_water["present"] and high_water["valid"] is not True:
        selected_usable = False
    if selected is not None and high_water["valid"] is True:
        selected_step = selected.get("checkpoint_step")
        if not isinstance(selected_step, int):
            errors.append(
                "selected recovery checkpoint has no authenticated semantic step"
            )
            selected_usable = False
        else:
            selected_checkpoint = selected.get("checkpoint")
            floor_checkpoint = high_water["resume_floor_checkpoint"]
            below_floor = selected_step < high_water["resume_floor_step"] or (
                selected_step == high_water["resume_floor_step"]
                and isinstance(floor_checkpoint, str)
                and (
                    not isinstance(selected_checkpoint, str)
                    or _checkpoint_point_key(selected_step, selected_checkpoint)
                    < _checkpoint_point_key(
                        high_water["resume_floor_step"], floor_checkpoint
                    )
                )
            )
            if below_floor:
                errors.append(
                    "selected recovery checkpoint is below the durable progress floor"
                )
                selected_usable = False

    if not selected_usable:
        state = "invalid"
    elif fallback_used:
        state = "fallback_available"
    elif errors:
        state = "available_with_errors"
    else:
        state = "available"
    result = {
        "state": state,
        "valid": state == "available",
        "recoverable": selected_usable,
        "store_path": os.fspath(store),
        "canonical_path": os.fspath(canonical),
        "pointer_generation": pointer_generation,
        "pointer_valid": pointer_valid,
        "fallback_used": fallback_used,
        "generation": selected.get("directory_name") if selected else None,
        "snapshot_path": os.fspath(selected_path) if selected_path is not None else None,
        "checkpoint": selected.get("checkpoint") if selected else None,
        "published_at": selected.get("published_at") if selected else None,
        "published_ns": selected.get("published_ns") if selected else None,
        "run_uid": selected.get("run_uid") if selected else None,
        "indexed": False,
        "index": {
            "present": False,
            "valid": None,
            "path": os.fspath(canonical / ".recovery_available.json"),
            "run_uid": None,
            "errors": [],
        },
        "high_water": high_water,
        "generations": generations,
        "stages": stages,
        "store_residues": store_residues,
        "errors": errors,
    }
    return result, selected_path if selected_usable else None


def _catalog_recovery_for_run(run_dir: Path) -> tuple[dict[str, Any], Path | None]:
    index_path = run_dir / ".recovery_available.json"
    expected_store = run_dir.parent / f".{run_dir.name}.recovery"
    if _entry_type(index_path)[0] == "missing" and _entry_type(expected_store)[0] == "missing":
        return _empty_recovery(run_dir), None

    recovery, snapshot = _catalog_recovery_store(expected_store)
    index = _catalog_recovery_index(
        run_dir, expected_store, expected_run_uid=recovery.get("run_uid")
    )
    recovery["index"] = index
    recovery["indexed"] = index["present"]
    recovery["errors"].extend(f"recovery index: {error}" for error in index["errors"])

    snapshot_matches = False
    if snapshot is not None:
        try:
            canonical_manifest = _read_json_file(run_dir / MANIFEST_NAME)
            snapshot_manifest = _read_json_file(snapshot / MANIFEST_NAME)
            snapshot_matches = canonical_manifest == snapshot_manifest
            if not snapshot_matches:
                recovery["errors"].append(
                    "recovery snapshot manifest does not match canonical run"
                )
        except (OSError, StorageError) as exc:
            recovery["errors"].append(str(exc))

    usable_snapshot = snapshot if snapshot_matches else None
    if usable_snapshot is None:
        recovery["state"] = "invalid"
    elif recovery["fallback_used"]:
        recovery["state"] = "fallback_available"
    elif recovery["errors"]:
        recovery["state"] = "available_with_errors"
    else:
        recovery["state"] = "available"
    recovery["recoverable"] = usable_snapshot is not None
    recovery["valid"] = recovery["state"] == "available"
    return recovery, usable_snapshot


def _checkpoint_step_without_unpickling(run_dir: Path, checkpoint: str) -> int:
    """Read the scalar checkpoint step without executing pickle opcodes."""

    if not _CHECKPOINT_NAME_PATTERN.fullmatch(checkpoint):
        raise StorageError("checkpoint name is unsafe")
    path = run_dir / "checkpoint.pkl" / checkpoint / "step.pkl"
    entry_type, size = _entry_type(path)
    if entry_type == "missing" and checkpoint.isascii() and checkpoint.isdecimal():
        # A numeric generation name encodes the checkpoint step.
        return int(checkpoint)
    if entry_type != "regular" or size is None or size > 1024:
        raise StorageError("checkpoint step.pkl is missing, unsafe, or unexpectedly large")
    data = path.read_bytes()
    try:
        decimal = data.decode("ascii").strip()
    except UnicodeDecodeError:
        decimal = ""
    if decimal and decimal.isdecimal():
        return int(decimal)
    integer: int | None = None
    allowed = {
        "PROTO",
        "FRAME",
        "BININT",
        "BININT1",
        "BININT2",
        "INT",
        "LONG",
        "LONG1",
        "LONG4",
        "STOP",
    }
    try:
        operations = list(pickletools.genops(data))
    except ValueError as exc:
        raise StorageError(f"checkpoint step.pkl is malformed: {exc}") from exc
    if not operations or operations[-1][0].name != "STOP":
        raise StorageError("checkpoint step.pkl has no terminal STOP opcode")
    for operation, argument, _ in operations:
        if operation.name not in allowed:
            raise StorageError(
                f"checkpoint step.pkl contains unsafe opcode {operation.name}"
            )
        if operation.name in {"BININT", "BININT1", "BININT2", "INT", "LONG", "LONG1", "LONG4"}:
            if integer is not None or not isinstance(argument, int) or isinstance(argument, bool):
                raise StorageError("checkpoint step.pkl is not a single integer")
            integer = argument
    if integer is None or integer < 0:
        raise StorageError("checkpoint step.pkl does not contain a nonnegative step")
    return integer


def _manifest_binding(run_dir: Path) -> tuple[str, str, frozenset[str]]:
    """Return the run ID, current digest, and valid manifest bindings.

    The launcher adds ``resolved_config_sha256`` exactly once after resolving
    Hydra, which normally happens before the first learner checkpoint. A fork
    is different: it already has a parent checkpoint, so storage must seed its
    progress fence before the launcher can resolve and persist that field. The
    pre-field manifest is therefore also an authentic binding for that narrow
    transition. No other manifest change is accepted.
    """

    manifest = _load_manifest(run_dir)
    run_uid = manifest["identity"].get("RUN_ID")
    if not isinstance(run_uid, str) or not _RUN_ID_PATTERN.fullmatch(run_uid):
        raise StorageError("run manifest has an invalid RUN_ID")
    path = run_dir / MANIFEST_NAME
    entry_type, size = _entry_type(path)
    if entry_type != "regular" or size is None or size > 16 * 1024 * 1024:
        raise StorageError("run manifest is unsafe or unexpectedly large")
    raw_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    config_digest = manifest.get("resolved_config_sha256")
    if config_digest is not None:
        if not isinstance(config_digest, str) or not re.fullmatch(
            r"[0-9a-f]{64}", config_digest
        ):
            raise StorageError(
                "run manifest has an invalid resolved_config_sha256"
            )
        before_config = dict(manifest)
        del before_config["resolved_config_sha256"]
    else:
        before_config = manifest
    pre_config_digest = hashlib.sha256(
        _canonical_json_bytes(before_config)
    ).hexdigest()
    # Accept version 1 digests and migrate them to the complete manifest digest on the
    # next advancing commit.
    return run_uid, raw_digest, frozenset({pre_config_digest, raw_digest})


def _validate_checkpoint_high_water_value(
    value: Any,
    *,
    marker_path: Path,
    run_dir: Path | None = None,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "marker_kind",
        "run_uid",
        "manifest_sha256",
        "high_checkpoint",
        "high_step",
        "previous_checkpoint",
        "previous_step",
        "resume_floor_step",
        "max_gap_steps",
        "attempt_id",
        "hostname",
        "committed_at",
        "committed_ns",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise StorageError(
            f"checkpoint high-water marker has unsupported or missing fields: {marker_path}"
        )
    if value.get("schema_version") != CHECKPOINT_HIGH_WATER_SCHEMA_VERSION:
        raise StorageError("checkpoint high-water marker has an unsupported schema")
    if value.get("marker_kind") != CHECKPOINT_HIGH_WATER_KIND:
        raise StorageError("checkpoint high-water marker has an invalid kind")

    run_uid = value.get("run_uid")
    if not isinstance(run_uid, str) or not _RUN_ID_PATTERN.fullmatch(run_uid):
        raise StorageError("checkpoint high-water marker has an invalid run_uid")
    if expected_run_id is not None and run_uid != expected_run_id:
        raise StorageError("checkpoint high-water marker belongs to a different run")
    digest = value.get("manifest_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise StorageError("checkpoint high-water marker has an invalid manifest digest")

    high_checkpoint = value.get("high_checkpoint")
    high_step = value.get("high_step")
    previous_checkpoint = value.get("previous_checkpoint")
    previous_step = value.get("previous_step")
    floor = value.get("resume_floor_step")
    gap = value.get("max_gap_steps")
    if (
        not isinstance(high_checkpoint, str)
        or not _CHECKPOINT_NAME_PATTERN.fullmatch(high_checkpoint)
        or not isinstance(high_step, int)
        or isinstance(high_step, bool)
        or high_step < 0
        or not isinstance(floor, int)
        or isinstance(floor, bool)
        or floor < 0
        or floor > high_step
        or not isinstance(gap, int)
        or isinstance(gap, bool)
        or gap < 0
        or gap > 10**12
    ):
        raise StorageError("checkpoint high-water marker has invalid progress fields")
    if (previous_checkpoint is None) != (previous_step is None):
        raise StorageError("checkpoint high-water marker has an incomplete previous point")
    if previous_checkpoint is not None:
        if (
            not isinstance(previous_checkpoint, str)
            or not _CHECKPOINT_NAME_PATTERN.fullmatch(previous_checkpoint)
            or not isinstance(previous_step, int)
            or isinstance(previous_step, bool)
            or previous_step < 0
            or previous_step > high_step
        ):
            raise StorageError("checkpoint high-water marker has an invalid previous point")
    attempt_id = value.get("attempt_id")
    if attempt_id is not None and (
        not isinstance(attempt_id, str) or not _RUN_ID_PATTERN.fullmatch(attempt_id)
    ):
        raise StorageError("checkpoint high-water marker has an invalid attempt_id")
    hostname = value.get("hostname")
    if not isinstance(hostname, str) or not hostname or len(hostname) > 255:
        raise StorageError("checkpoint high-water marker has an invalid hostname")
    committed_at = value.get("committed_at")
    committed_ns = value.get("committed_ns")
    if not isinstance(committed_at, str) or len(committed_at) > 64:
        raise StorageError("checkpoint high-water marker has an invalid timestamp")
    if not isinstance(committed_ns, str) or not re.fullmatch(r"[0-9]{19}", committed_ns):
        raise StorageError("checkpoint high-water marker has invalid nanoseconds")

    if run_dir is not None:
        actual_uid, _, valid_digests = _manifest_binding(run_dir)
        if run_uid != actual_uid or digest not in valid_digests:
            raise StorageError(
                "checkpoint high-water marker does not match the candidate run manifest"
            )
    return dict(value)


def load_checkpoint_high_water(
    marker: os.PathLike[str] | str,
    *,
    run_dir: os.PathLike[str] | str | None = None,
    expected_run_id: str | None = None,
) -> dict[str, Any] | None:
    """Load and authenticate a high-water marker without following a symlink."""

    path = Path(marker).expanduser()
    entry_type, size = _entry_type(path)
    if entry_type == "missing":
        return None
    if entry_type != "regular" or size is None or size > CHECKPOINT_HIGH_WATER_MAX_BYTES:
        raise StorageError("checkpoint high-water marker is unsafe or unexpectedly large")
    try:
        value = _read_json_file(path)
    except (OSError, StorageError) as exc:
        raise StorageError(f"cannot read checkpoint high-water marker: {exc}") from exc
    return _validate_checkpoint_high_water_value(
        value,
        marker_path=path,
        run_dir=Path(run_dir).expanduser() if run_dir is not None else None,
        expected_run_id=expected_run_id,
    )


def _checkpoint_high_water_payload(
    *,
    run_uid: str,
    manifest_sha256: str,
    checkpoint: str,
    step: int,
    previous_checkpoint: str | None,
    previous_step: int | None,
    resume_floor_step: int,
    max_gap_steps: int,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_HIGH_WATER_SCHEMA_VERSION,
        "marker_kind": CHECKPOINT_HIGH_WATER_KIND,
        "run_uid": run_uid,
        "manifest_sha256": manifest_sha256,
        "high_checkpoint": checkpoint,
        "high_step": step,
        "previous_checkpoint": previous_checkpoint,
        "previous_step": previous_step,
        "resume_floor_step": resume_floor_step,
        "max_gap_steps": max_gap_steps,
        "attempt_id": environment.get("RUN_ATTEMPT_ID") or None,
        "hostname": environment.get("RUN_HOSTNAME") or socket.gethostname(),
        "committed_at": _utc_now(),
        "committed_ns": f"{time.time_ns():019d}",
    }


def _checkpoint_identity_order_key(checkpoint: str) -> tuple[int, int, int, Any]:
    """Order generations, prioritizing explicit offline-update progress."""

    offline = re.search(r"-u([0-9]{12})$", checkpoint)
    if offline is not None:
        return 1, int(offline.group(1)), 1, checkpoint
    if checkpoint.isdecimal():
        return 0, 0, 0, int(checkpoint)
    return 0, 0, 1, checkpoint


def _checkpoint_point_key(
    step: int, checkpoint: str
) -> tuple[int, tuple[int, int, int, Any]]:
    """Order saves at the same environment step by their generation identity."""

    return step, _checkpoint_identity_order_key(checkpoint)


def _checkpoint_floor_identity(value: Mapping[str, Any]) -> str | None:
    """Return the checkpoint identity attached to a numeric resume floor."""

    if (
        value.get("previous_checkpoint") is not None
        and value.get("previous_step") == value.get("resume_floor_step")
    ):
        return str(value["previous_checkpoint"])
    if value.get("resume_floor_step") == value.get("high_step"):
        return str(value["high_checkpoint"])
    return None


def _advertised_checkpoint_point(run_dir: Path) -> tuple[str, int] | None:
    """Authenticate the checkpoint currently advertised through `latest`."""

    root = run_dir / "checkpoint.pkl"
    root_type, _ = _entry_type(root)
    if root_type == "missing":
        return None
    if root_type != "directory":
        raise StorageError("checkpoint root is unsafe")
    latest = root / "latest"
    latest_type, latest_size = _entry_type(latest)
    if latest_type == "missing":
        return None
    if latest_type != "regular" or latest_size is None or latest_size > 256:
        raise StorageError("checkpoint latest pointer is unsafe")
    try:
        checkpoint = latest.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise StorageError(f"cannot read checkpoint latest pointer: {exc}") from exc
    if not _CHECKPOINT_NAME_PATTERN.fullmatch(checkpoint):
        raise StorageError("checkpoint latest pointer has an unsafe generation")
    generation_type, _ = _entry_type(root / checkpoint)
    done_type, done_size = _entry_type(root / checkpoint / "done")
    if generation_type != "directory" or done_type != "regular" or done_size != 0:
        raise StorageError("advertised checkpoint generation is incomplete or unsafe")
    return checkpoint, _checkpoint_step_without_unpickling(run_dir, checkpoint)


def _ensure_directory_entry_durable(directory: Path) -> os.stat_result:
    """Create real directory ancestors and fsync every new parent entry."""

    missing: list[Path] = []
    current = directory
    while True:
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if current == current.parent:
                raise StorageError(f"cannot create directory hierarchy: {directory}")
            missing.append(current)
            current = current.parent
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise StorageError(f"metadata parent is not a real directory: {current}")
        break

    for child in reversed(missing):
        try:
            child.mkdir(mode=0o750)
        except FileExistsError:
            pass
        metadata = child.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise StorageError(f"metadata parent is not a real directory: {child}")
        _fsync_directory(child.parent)

    metadata = directory.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise StorageError(f"metadata parent is not a real directory: {directory}")
    # Persist the parent directory entry when repairing an existing directory.
    _fsync_directory(directory.parent)
    return metadata


def commit_checkpoint_high_water(
    marker: os.PathLike[str] | str,
    run_dir: os.PathLike[str] | str,
    checkpoint: str,
    *,
    max_gap_steps: int,
    seed: bool = False,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Persist progress before a local checkpoint is advertised through latest."""

    if (
        not isinstance(max_gap_steps, int)
        or isinstance(max_gap_steps, bool)
        or max_gap_steps < 0
        or max_gap_steps > 10**12
    ):
        raise StorageError("max checkpoint resume gap must be a nonnegative integer")
    path = Path(marker).expanduser()
    directory = Path(run_dir).expanduser()
    run_uid, manifest_sha256, _ = _manifest_binding(directory)
    step = _checkpoint_step_without_unpickling(directory, checkpoint)
    # Establish (or repair) durability of the marker's containing directory
    # before any catch-up fast path can authorize replacement of `latest`.
    parent_before = _ensure_directory_entry_durable(path.parent)
    current = load_checkpoint_high_water(
        path, run_dir=directory, expected_run_id=run_uid
    )
    if current is not None and current["max_gap_steps"] != max_gap_steps:
        raise StorageError(
            "max checkpoint resume gap changed for an existing run: "
            f"stored={current['max_gap_steps']}, requested={max_gap_steps}"
        )
    # Seed only without an existing marker; subsequent initialization must preserve the
    # recorded checkpoint floor.
    if seed and current is not None:
        seed = False
    if current is not None:
        floor_checkpoint = _checkpoint_floor_identity(current)
        below_floor = step < current["resume_floor_step"] or (
            step == current["resume_floor_step"]
            and floor_checkpoint is not None
            and _checkpoint_point_key(step, checkpoint)
            < _checkpoint_point_key(current["resume_floor_step"], floor_checkpoint)
        )
        if below_floor:
            raise StorageError(
                "checkpoint would continue below the durable progress floor: "
                f"candidate={step}/{checkpoint}, "
                f"floor={current['resume_floor_step']}/{floor_checkpoint or '*'}, "
                f"high={current['high_step']}/{current['high_checkpoint']}"
            )
    advertised = None if seed else _advertised_checkpoint_point(directory)
    if current is not None and not seed:
        if advertised is None:
            # Allow retry if the first checkpoint marker commits before latest is
            # created.
            # A missing latest pointer after any previous checkpoint is corruption.
            if current["previous_checkpoint"] is not None or (
                _checkpoint_point_key(step, checkpoint)
                < _checkpoint_point_key(
                    current["high_step"], current["high_checkpoint"]
                )
            ):
                raise StorageError(
                    "durable progress marker exists but no checkpoint is currently advertised"
                )
        else:
            current_floor_checkpoint = _checkpoint_floor_identity(current)
            advertised_below_floor = advertised[1] < current["resume_floor_step"] or (
                advertised[1] == current["resume_floor_step"]
                and current_floor_checkpoint is not None
                and _checkpoint_point_key(advertised[1], advertised[0])
                < _checkpoint_point_key(
                    current["resume_floor_step"], current_floor_checkpoint
                )
            )
            if advertised_below_floor:
                raise StorageError(
                    "currently advertised checkpoint is below the durable progress floor"
                )
    if advertised is not None and _checkpoint_point_key(step, checkpoint) < (
        _checkpoint_point_key(advertised[1], advertised[0])
    ):
        raise StorageError(
            "checkpoint would roll back the currently advertised progress: "
            f"candidate={step}/{checkpoint}, "
            f"advertised={advertised[1]}/{advertised[0]}"
        )
    if current is not None and _checkpoint_point_key(step, checkpoint) <= (
        _checkpoint_point_key(current["high_step"], current["high_checkpoint"])
    ):
        # A bounded rollback may legitimately replay toward the prior high.
        # Never lower or refresh the durable marker while catching up.
        parent_after = path.parent.lstat()
        if (
            stat.S_ISLNK(parent_after.st_mode)
            or not stat.S_ISDIR(parent_after.st_mode)
            or (parent_before.st_dev, parent_before.st_ino)
            != (parent_after.st_dev, parent_after.st_ino)
        ):
            raise StorageError("checkpoint high-water parent changed during commit")
        return current

    if current is None:
        previous_checkpoint = advertised[0] if advertised is not None else None
        previous_step = advertised[1] if advertised is not None else None
        floor = (
            step
            if seed
            else (
                previous_step
                if previous_step is not None
                else max(0, step - max_gap_steps)
            )
        )
    else:
        previous_checkpoint = advertised[0] if advertised is not None else None
        previous_step = advertised[1] if advertised is not None else None
        floor = max(
            current["resume_floor_step"],
            (
                previous_step
                if previous_step is not None
                else step - max_gap_steps
            ),
        )
    value = _checkpoint_high_water_payload(
        run_uid=run_uid,
        manifest_sha256=manifest_sha256,
        checkpoint=checkpoint,
        step=step,
        previous_checkpoint=previous_checkpoint,
        previous_step=previous_step,
        resume_floor_step=floor,
        max_gap_steps=max_gap_steps,
        environment=os.environ if environment is None else environment,
    )

    existing_type, _ = _entry_type(path)
    if existing_type not in {"missing", "regular"}:
        raise StorageError("refusing to replace unsafe checkpoint high-water marker")
    _atomic_write_json(path, value)
    parent_after = path.parent.lstat()
    if (
        stat.S_ISLNK(parent_after.st_mode)
        or not stat.S_ISDIR(parent_after.st_mode)
        or (parent_before.st_dev, parent_before.st_ino)
        != (parent_after.st_dev, parent_after.st_ino)
    ):
        raise StorageError("checkpoint high-water parent changed during commit")
    return value


def check_checkpoint_high_water(
    marker: os.PathLike[str] | str,
    run_dir: os.PathLike[str] | str,
    checkpoint: str,
    *,
    minimum: str = "floor",
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    """Require a candidate checkpoint to meet the recorded progress fence."""

    if minimum not in {"floor", "high"}:
        raise StorageError("checkpoint high-water minimum must be floor or high")
    directory = Path(run_dir).expanduser()
    value = load_checkpoint_high_water(
        marker, run_dir=directory, expected_run_id=expected_run_id
    )
    step = _checkpoint_step_without_unpickling(directory, checkpoint)
    if value is None:
        return {
            "present": False,
            "checkpoint": checkpoint,
            "checkpoint_step": step,
            "minimum": minimum,
        }
    required = value["resume_floor_step"] if minimum == "floor" else value["high_step"]
    required_checkpoint = (
        _checkpoint_floor_identity(value)
        if minimum == "floor"
        else value["high_checkpoint"]
    )
    below_required = step < required or (
        step == required
        and required_checkpoint is not None
        and _checkpoint_point_key(step, checkpoint)
        < _checkpoint_point_key(required, required_checkpoint)
    )
    if below_required:
        raise StorageError(
            "checkpoint is below the durable progress fence: "
            f"candidate={step}/{checkpoint}, "
            f"required={required}/{required_checkpoint or '*'}, "
            f"high={value['high_step']}/{value['high_checkpoint']}"
        )
    return {
        "present": True,
        "checkpoint": checkpoint,
        "checkpoint_step": step,
        "minimum": minimum,
        "required_step": required,
        "high_step": value["high_step"],
        "resume_floor_step": value["resume_floor_step"],
    }


def repair_first_checkpoint_latest(
    marker: os.PathLike[str] | str,
    run_dir: os.PathLike[str] | str,
    *,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    """Finish the sole safe marker-before-first-`latest` crash transaction."""

    directory = Path(run_dir).expanduser()
    value = load_checkpoint_high_water(
        marker, run_dir=directory, expected_run_id=expected_run_id
    )
    if value is None:
        return {"present": False, "repaired": False}
    root = directory / "checkpoint.pkl"
    root_type, _ = _entry_type(root)
    if root_type != "directory":
        raise StorageError("checkpoint root is missing or unsafe")
    latest = root / "latest"
    latest_type, latest_size = _entry_type(latest)
    if latest_type == "regular":
        if latest_size is None or latest_size > 256:
            raise StorageError("checkpoint latest pointer is unexpectedly large")
        return {"present": True, "repaired": False}
    if latest_type != "missing":
        raise StorageError("checkpoint latest pointer is unsafe")
    if value["previous_checkpoint"] is not None or value["previous_step"] is not None:
        raise StorageError(
            "established checkpoint high-water marker cannot repair a missing latest"
        )

    completed: list[str] = []
    try:
        children = list(root.iterdir())
    except OSError as exc:
        raise StorageError(f"cannot enumerate checkpoint root: {exc}") from exc
    for child in children:
        child_type, _ = _entry_type(child)
        if child_type != "directory" or not _CHECKPOINT_NAME_PATTERN.fullmatch(
            child.name
        ):
            continue
        done_type, _ = _entry_type(child / "done")
        if done_type == "regular":
            completed.append(child.name)
    if completed != [value["high_checkpoint"]]:
        raise StorageError(
            "missing latest is not backed by exactly one marker-owned checkpoint"
        )
    checkpoint = completed[0]
    check_checkpoint_high_water(
        marker,
        directory,
        checkpoint,
        minimum="high",
        expected_run_id=expected_run_id,
    )
    _atomic_write_bytes(latest, (checkpoint + "\n").encode("utf-8"))
    return {
        "present": True,
        "repaired": True,
        "checkpoint": checkpoint,
        "checkpoint_step": value["high_step"],
    }


def _last_wandb_attempt_id(run_dir: Path) -> str:
    receipt_path = run_dir / WANDB_RECEIPT_NAME
    receipt_type, receipt_size = _entry_type(receipt_path)
    if receipt_type != "missing":
        if (
            receipt_type != "regular"
            or receipt_size is None
            or receipt_size > 16 * 1024
        ):
            raise StorageError("W&B initialization receipt is unsafe")
        try:
            receipt = _read_json_file(receipt_path)
        except (OSError, StorageError) as exc:
            raise StorageError(f"cannot read W&B initialization receipt: {exc}") from exc
        if not isinstance(receipt, dict) or receipt.get("schema_version") != 1:
            raise StorageError("W&B initialization receipt has an unsupported schema")
        receipt_id = receipt.get("wandb_id")
        if not isinstance(receipt_id, str) or not _WANDB_ID_PATTERN.fullmatch(receipt_id):
            raise StorageError("W&B initialization receipt has an unsafe run ID")
        return receipt_id

    path = run_dir / ATTEMPTS_NAME
    entry_type, size = _entry_type(path)
    if entry_type != "regular" or size is None or size > 16 * 1024 * 1024:
        raise StorageError("run attempt history is missing, unsafe, or unexpectedly large")
    latest: str | None = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise StorageError(
                        f"run attempt history line {line_number} is not an object"
                    )
                attempt = value.get("attempt")
                candidate = (
                    attempt.get("RUN_WANDB_ATTEMPT_ID")
                    if isinstance(attempt, dict)
                    else None
                )
                if candidate is not None:
                    if not isinstance(candidate, str) or not _WANDB_ID_PATTERN.fullmatch(candidate):
                        raise StorageError(
                            f"run attempt history line {line_number} has an unsafe W&B run ID"
                        )
                    latest = candidate
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StorageError(f"cannot read run attempt history: {exc}") from exc
    if latest is None:
        raise StorageError("run attempt history has no recorded W&B run ID")
    return latest


def wandb_continuation(
    run_dir: os.PathLike[str] | str,
    *,
    expected_run_id: str,
    expected_source_id: str | None = None,
    live_run_dirs: Sequence[os.PathLike[str] | str] = (),
) -> dict[str, Any]:
    """Resolve the highest authenticated W&B rewind point for a stopped run."""

    canonical = Path(run_dir).expanduser()
    # Select by checkpoint progress even when delayed publishers commit older snapshots
    # later.
    candidates: list[tuple[int, int, Path, str, str]] = []
    candidate_errors: list[StorageError] = []
    store = canonical.parent / f".{canonical.name}.recovery"
    high_water_path = store / CHECKPOINT_HIGH_WATER_NAME
    high_water = load_checkpoint_high_water(
        high_water_path, expected_run_id=expected_run_id
    )

    def add_candidate(
        *, artifact: Path, checkpoint: str, marker_kind: str, timestamp: int
    ) -> None:
        try:
            step = _checkpoint_step_without_unpickling(artifact, checkpoint)
            if high_water is not None:
                check_checkpoint_high_water(
                    high_water_path,
                    artifact,
                    checkpoint,
                    minimum="floor",
                    expected_run_id=expected_run_id,
                )
        except StorageError as exc:
            candidate_errors.append(exc)
            return
        except OSError as exc:
            candidate_errors.append(StorageError(str(exc)))
            return
        candidates.append((step, timestamp, artifact, checkpoint, marker_kind))

    for live_run_dir in live_run_dirs:
        live = Path(live_run_dir).expanduser()
        live_type, _ = _entry_type(live)
        if live_type not in {"missing", "directory"}:
            raise StorageError("live W&B continuation candidate is unsafe")
        if live_type == "directory":
            live_checkpoint = _catalog_checkpoint(live)
            checkpoint = live_checkpoint.get("latest")
            if live_checkpoint.get("valid") and isinstance(checkpoint, str):
                latest_path = live / "checkpoint.pkl" / "latest"
                try:
                    timestamp = latest_path.lstat().st_mtime_ns
                except OSError:
                    timestamp = 0
                add_candidate(
                    artifact=live,
                    checkpoint=checkpoint,
                    marker_kind="live-checkpoint",
                    timestamp=timestamp,
                )
            elif (
                high_water is not None
                and high_water["previous_checkpoint"] is None
                and live_checkpoint.get("state") == "jax_complete_pointer_invalid"
                and live_checkpoint.get("completed_generations") == 1
                and live_checkpoint.get("latest") == high_water["high_checkpoint"]
            ):
                # Without latest, allow W&B continuation only from the sole generation
                # authenticated by the progress marker.
                add_candidate(
                    artifact=live,
                    checkpoint=high_water["high_checkpoint"],
                    marker_kind="live-first-checkpoint-recovery",
                    timestamp=0,
                )

    final = _catalog_sync_complete(canonical)
    if (
        final["valid"]
        and final["run_uid"] == expected_run_id
        and final["checkpoint"] is not None
    ):
        completed_ns = final.get("completed_ns")
        if isinstance(completed_ns, str) and completed_ns.isdigit():
            add_candidate(
                artifact=canonical,
                checkpoint=str(final["checkpoint"]),
                marker_kind="final-publish",
                timestamp=int(completed_ns),
            )

    recovery, _snapshot = _catalog_recovery_store(store)
    for generation in recovery.get("generations", []):
        published_ns = generation.get("published_ns")
        checkpoint = generation.get("checkpoint")
        artifact_value = generation.get("path")
        if (
            generation.get("valid")
            and generation.get("run_uid") == expected_run_id
            and isinstance(artifact_value, str)
            and isinstance(published_ns, str)
            and published_ns.isdigit()
            and isinstance(checkpoint, str)
        ):
            add_candidate(
                artifact=Path(artifact_value),
                checkpoint=checkpoint,
                marker_kind="periodic-recovery",
                timestamp=int(published_ns),
            )

    if not candidates:
        if candidate_errors:
            raise candidate_errors[0]
        suffix = " meeting its durable progress floor" if high_water is not None else ""
        raise StorageError(
            "run has no integrity-certified checkpoint for W&B continuation" + suffix
        )
    step, _, artifact, checkpoint, marker_kind = max(
        candidates,
        key=lambda item: (
            item[0],
            _checkpoint_identity_order_key(item[3]),
            item[1],
        ),
    )

    manifest = _load_manifest(artifact)
    identity = manifest["identity"]
    provenance = manifest["provenance"]
    if identity.get("RUN_ID") != expected_run_id:
        raise StorageError(
            "W&B continuation run ID does not match immutable manifest: "
            f"stored={identity.get('RUN_ID')!r}, expected={expected_run_id!r}"
        )
    if expected_source_id is not None and provenance.get("source_id") != expected_source_id:
        raise StorageError(
            "W&B continuation source does not match immutable manifest: "
            f"stored={provenance.get('source_id')!r}, expected={expected_source_id!r}"
        )

    actual_checkpoint = _catalog_checkpoint(artifact)
    marker_owned_pointerless = (
        marker_kind == "live-first-checkpoint-recovery"
        and actual_checkpoint.get("state") == "jax_complete_pointer_invalid"
        and actual_checkpoint.get("completed_generations") == 1
        and actual_checkpoint.get("latest") == checkpoint
    )
    if (
        not marker_owned_pointerless
        and (
            not actual_checkpoint.get("valid")
            or actual_checkpoint.get("latest") != checkpoint
        )
    ):
        raise StorageError("W&B continuation checkpoint failed validation")
    wandb_id = _last_wandb_attempt_id(artifact)
    parent_run_name = identity.get("RUN_NAME")
    if (
        not isinstance(parent_run_name, str)
        or len(parent_run_name) > 180
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", parent_run_name)
    ):
        raise StorageError("W&B continuation manifest has an unsafe run name")
    return {
        "artifact": os.fspath(artifact),
        "checkpoint": checkpoint,
        "checkpoint_step": step,
        "marker_kind": marker_kind,
        "parent_run_name": parent_run_name,
        "resume_from": f"{wandb_id}?_step={step}",
        "wandb_id": wandb_id,
    }


def _catalog_last_event(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    last: dict[str, Any] | None = None
    entry_type, _ = _entry_type(path)
    if entry_type == "missing":
        return None, errors
    if entry_type != "regular":
        return None, [f"{path.name} is {entry_type}, not a regular file"]
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    errors.append(f"invalid JSON at line {line_number}")
                    continue
                if not isinstance(value, dict):
                    errors.append(f"non-object event at line {line_number}")
                    continue
                last = value
    except (OSError, UnicodeError) as exc:
        errors.append(f"cannot read {path.name}: {exc}")
    return last, errors


def _catalog_lifecycle(run_dir: Path, run_id: Any) -> dict[str, Any]:
    errors: list[str] = []
    status_record: dict[str, Any] | None = None
    status_path = run_dir / STATUS_NAME
    status_type, _ = _entry_type(status_path)
    if status_type == "regular":
        try:
            value = _read_json_file(status_path)
            if not isinstance(value, dict):
                errors.append("run_status.json must contain an object")
            else:
                status_record = value
                if value.get("schema_version") != SCHEMA_VERSION:
                    errors.append("run_status.json has an unsupported schema")
                sequence = value.get("sequence")
                if (
                    not isinstance(sequence, int)
                    or isinstance(sequence, bool)
                    or sequence < 0
                ):
                    errors.append("run_status.json has an invalid sequence")
                if value.get("status") is not None and not isinstance(
                    value.get("status"), str
                ):
                    errors.append("run_status.json has a non-string status")
                if value.get("event") is not None and not isinstance(
                    value.get("event"), str
                ):
                    errors.append("run_status.json has a non-string event")
                if run_id is not None and value.get("run_id") != run_id:
                    errors.append("run_status.json RUN_ID does not match the manifest")
        except (OSError, StorageError) as exc:
            errors.append(str(exc))
    elif status_type != "missing":
        errors.append(f"run_status.json is {status_type}, not a regular file")

    last_event, event_errors = _catalog_last_event(run_dir / EVENTS_NAME)
    errors.extend(event_errors)
    if last_event is not None:
        if last_event.get("schema_version") != SCHEMA_VERSION:
            errors.append("latest event has an unsupported schema")
        if not isinstance(last_event.get("sequence"), int):
            errors.append("latest event has an invalid sequence")
        if run_id is not None and last_event.get("run_id") != run_id:
            errors.append("latest event RUN_ID does not match the manifest")

    status_sequence = (
        status_record.get("sequence", -1) if status_record is not None else -1
    )
    event_sequence = last_event.get("sequence", -1) if last_event is not None else -1
    if (
        status_record is not None
        and last_event is not None
        and isinstance(status_sequence, int)
        and isinstance(event_sequence, int)
        and event_sequence > status_sequence
    ):
        errors.append("run_status.json lags the append-only event log")
        source = "run_events.jsonl"
        selected = last_event
    elif status_record is not None:
        source = "run_status.json"
        selected = status_record
    elif last_event is not None:
        source = "run_events.jsonl"
        selected = last_event
    else:
        source = None
        selected = {}
    return {
        "valid": not errors,
        "source": source,
        "status": selected.get("status"),
        "event": selected.get("event"),
        "sequence": selected.get("sequence"),
        "updated_at": selected.get("updated_at") or selected.get("timestamp"),
        "errors": errors,
    }


def _safe_checkpoint_generation(path: Path) -> bool:
    entry_type, _ = _entry_type(path)
    if entry_type != "directory" or not _CHECKPOINT_NAME_PATTERN.fullmatch(path.name):
        return False
    done_type, _ = _entry_type(path / "done")
    return done_type == "regular"


def _checkpoint_generation_sort_key(name: str) -> tuple[int, Any]:
    """Sort decimal step names numerically while keeping other names stable."""

    if name.isdecimal():
        return 1, int(name)
    return 0, name


def _catalog_checkpoint(run_dir: Path) -> dict[str, Any]:
    jax_path = run_dir / "checkpoint.pkl"
    incomplete_path = run_dir / "checkpoint.pkl.incomplete"
    legacy_path = run_dir / "latest.pt"
    jax_type, _ = _entry_type(jax_path)
    incomplete_type, _ = _entry_type(incomplete_path)
    legacy_type, legacy_size = _entry_type(legacy_path)
    errors: list[str] = []
    completed: list[str] = []
    pointer: str | None = None
    pointer_valid = False

    if jax_type == "directory":
        try:
            for child in jax_path.iterdir():
                if _safe_checkpoint_generation(child):
                    completed.append(child.name)
        except OSError as exc:
            errors.append(f"cannot inspect checkpoint.pkl: {exc}")
        completed.sort(key=_checkpoint_generation_sort_key)
        latest_path = jax_path / "latest"
        latest_type, latest_size = _entry_type(latest_path)
        if latest_type == "regular":
            if latest_size is not None and latest_size <= 4096:
                try:
                    pointer = latest_path.read_text(encoding="utf-8").replace("\n", "")
                except (OSError, UnicodeError) as exc:
                    errors.append(f"cannot read checkpoint latest pointer: {exc}")
            else:
                errors.append("checkpoint latest pointer is unexpectedly large")
            pointer_valid = bool(
                pointer
                and _CHECKPOINT_NAME_PATTERN.fullmatch(pointer)
                and pointer in completed
            )
            if not pointer_valid:
                errors.append("checkpoint latest pointer is invalid or incomplete")
        elif latest_type == "missing":
            errors.append("checkpoint latest pointer is missing")
        else:
            errors.append(f"checkpoint latest pointer is {latest_type}")
        if not completed:
            errors.append("checkpoint.pkl has no completed generation")
    elif jax_type != "missing":
        errors.append(f"checkpoint.pkl is {jax_type}, not a real directory")

    legacy_valid = legacy_type == "regular" and bool(legacy_size)
    if legacy_type not in {"missing", "regular"}:
        errors.append(f"latest.pt is {legacy_type}, not a regular file")
    elif legacy_type == "regular" and not legacy_valid:
        errors.append("latest.pt is empty")

    incomplete_present = incomplete_type != "missing"
    if incomplete_present and incomplete_type != "directory":
        errors.append(
            f"checkpoint.pkl.incomplete is {incomplete_type}, not a directory"
        )

    formats = sum(
        (
            jax_type != "missing",
            incomplete_present,
            legacy_type != "missing",
        )
    )
    if formats > 1:
        state = "mixed"
        errors.append("multiple checkpoint formats are present")
    elif jax_type != "missing":
        if completed and pointer_valid:
            state = "jax_complete"
        elif completed:
            state = "jax_complete_pointer_invalid"
        else:
            state = "jax_incomplete"
    elif incomplete_present:
        state = "jax_incomplete"
    elif legacy_type != "missing":
        state = "legacy_complete" if legacy_valid else "invalid"
    else:
        state = "missing"

    valid = state in {"jax_complete", "legacy_complete"} and not errors
    return {
        "state": state,
        "valid": valid,
        "recoverable": bool(completed) or legacy_valid,
        "latest": pointer if pointer_valid else (completed[-1] if completed else None),
        "latest_pointer_valid": pointer_valid if jax_type == "directory" else None,
        "completed_generations": len(completed),
        "errors": errors,
    }


def _catalog_symlinks(run_dir: Path) -> dict[str, Any]:
    total = broken = absolute = escaping = 0
    errors: list[str] = []
    try:
        root = run_dir.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return {
            "total": 0,
            "broken": 0,
            "absolute": 0,
            "escaping": 0,
            "errors": [f"cannot inspect run symlinks: {exc}"],
        }

    def walk_error(exc: OSError) -> None:
        errors.append(str(exc))

    for directory, directory_names, file_names in os.walk(
        root, followlinks=False, onerror=walk_error
    ):
        current = Path(directory)
        for name in list(directory_names) + list(file_names):
            path = current / name
            try:
                if not path.is_symlink():
                    continue
                total += 1
                link_text = os.readlink(path)
                if os.path.isabs(link_text):
                    absolute += 1
                try:
                    lexical_target = (
                        Path(link_text)
                        if os.path.isabs(link_text)
                        else path.parent / link_text
                    ).resolve(strict=False)
                    if not _is_within(lexical_target, root):
                        escaping += 1
                except (OSError, RuntimeError):
                    escaping += 1
                try:
                    path.resolve(strict=True)
                except (OSError, RuntimeError):
                    broken += 1
                if name in directory_names:
                    directory_names.remove(name)
            except OSError as exc:
                errors.append(f"cannot inspect symlink {path}: {exc}")
    return {
        "total": total,
        "broken": broken,
        "absolute": absolute,
        "escaping": escaping,
        "errors": errors,
    }


def _looks_like_legacy_run(directory: Path) -> bool:
    has_config = any(
        _entry_type(directory / name)[0] == "regular"
        for name in ("dreamer_config.yaml", "config.yaml")
    )
    has_output = any(
        _entry_type(directory / name)[0] != "missing"
        for name in ("checkpoint.pkl", "checkpoint.pkl.incomplete", "latest.pt")
    ) or any(
        _entry_type(directory / name)[0] == "regular"
        for name in ("metrics.jsonl", "scores.jsonl")
    )
    # Identify partial runs by run output; a config file alone is insufficient.
    return has_output or (
        has_config and _entry_type(directory / "analysis")[0] == "directory"
    )


def _catalog_artifacts(run_dir: Path) -> tuple[dict[str, bool], dict[str, int]]:
    presence: dict[str, bool] = {}
    sizes: dict[str, int] = {}
    for name in _CATALOG_ARTIFACTS:
        entry_type, size = _entry_type(run_dir / name)
        presence[name] = entry_type == "regular"
        if entry_type == "regular" and size is not None:
            sizes[name] = size
    return presence, sizes


def _catalog_config(run_dir: Path, expected_sha: Any) -> dict[str, Any]:
    errors: list[str] = []
    selected: Path | None = None
    for name in ("dreamer_config.yaml", "config.yaml"):
        if _entry_type(run_dir / name)[0] == "regular":
            selected = run_dir / name
            break

    actual_sha: str | None = None
    if selected is not None:
        digest = hashlib.sha256()
        try:
            with selected.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            actual_sha = digest.hexdigest()
        except OSError as exc:
            errors.append(f"cannot hash configuration artifact: {exc}")
    elif expected_sha is not None:
        errors.append("manifest config digest exists but configuration artifact is missing")

    matches = (
        actual_sha == expected_sha
        if isinstance(expected_sha, str) and actual_sha is not None
        else None
    )
    if matches is False:
        errors.append("configuration artifact does not match the manifest digest")
    return {
        "path": selected.name if selected is not None else None,
        "sha256": actual_sha,
        "manifest_sha256": expected_sha,
        "matches_manifest": matches,
        "valid": not errors,
        "errors": errors,
    }


def _catalog_source_snapshot(run_dir: Path, expected_source_id: Any) -> dict[str, Any]:
    directory = run_dir / "source_provenance"
    entry_type, _ = _entry_type(directory)
    if entry_type == "missing":
        return {
            "present": False,
            "valid": None,
            "path": None,
            "source_id": None,
            "errors": [],
        }
    if entry_type != "directory":
        return {
            "present": True,
            "valid": False,
            "path": os.fspath(directory),
            "source_id": None,
            "errors": ["source_provenance is not a real directory"],
        }
    if not isinstance(expected_source_id, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_source_id
    ):
        return {
            "present": True,
            "valid": False,
            "path": os.fspath(directory),
            "source_id": None,
            "errors": ["manifest has no valid source ID for snapshot verification"],
        }
    try:
        verified = verify_source_snapshot(directory, source_id=expected_source_id)
    except (OSError, StorageError) as exc:
        return {
            "present": True,
            "valid": False,
            "path": os.fspath(directory),
            "source_id": expected_source_id,
            "errors": [str(exc)],
        }
    return {
        "present": True,
        "valid": True,
        "path": os.fspath(directory),
        "source_id": verified["source_id"],
        "errors": [],
    }


def _catalog_attempt_file_snapshots(
    run_dir: Path,
    expected_sha: Any,
    *,
    filename: str,
    label: str,
) -> dict[str, Any]:
    """Verify immutable per-attempt files named by the run identity."""
    attempts = run_dir / "attempts"
    attempts_type, _ = _entry_type(attempts)
    if not isinstance(expected_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha
    ):
        return {
            "expected_sha256": expected_sha,
            "present": False,
            "valid": None,
            "artifacts": [],
            "errors": [],
        }
    if attempts_type == "missing":
        return {
            "expected_sha256": expected_sha,
            "present": False,
            "valid": False,
            "artifacts": [],
            "errors": [f"{label} snapshot directory is missing"],
        }
    if attempts_type != "directory":
        return {
            "expected_sha256": expected_sha,
            "present": True,
            "valid": False,
            "artifacts": [],
            "errors": [f"{label} snapshot directory is not a real directory"],
        }

    artifacts: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        children = sorted(attempts.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        children = []
        errors.append(f"cannot enumerate {label} snapshots: {exc}")
    for child in children:
        child_type, _ = _entry_type(child)
        if child_type != "directory":
            errors.append(f"unsafe attempt entry: {child.name} is {child_type}")
            continue
        artifact = child / filename
        artifact_type, artifact_size = _entry_type(artifact)
        if artifact_type == "missing":
            # Record attempts without a verified source snapshot, but do not follow
            # their paths.
            continue
        if artifact_type != "regular":
            errors.append(
                f"unsafe {label} snapshot: {child.name}/{filename} is {artifact_type}"
            )
            continue
        try:
            verified_size, actual_sha = _catalog_hash_regular(artifact)
        except (OSError, StorageError) as exc:
            errors.append(f"cannot hash {child.name}/{filename}: {exc}")
            continue
        matches = actual_sha == expected_sha
        artifacts.append(
            {
                "attempt_id": child.name,
                "path": os.fspath(artifact.relative_to(run_dir)),
                "size_bytes": verified_size if verified_size is not None else artifact_size,
                "sha256": actual_sha,
                "matches_manifest": matches,
            }
        )
        if not matches:
            errors.append(
                f"{label} snapshot for attempt {child.name} does not match the manifest"
            )
    if not artifacts:
        errors.append(f"no {label} snapshot matches the manifest attempt history")
    return {
        "expected_sha256": expected_sha,
        "present": bool(artifacts),
        "valid": not errors,
        "artifacts": artifacts,
        "errors": errors,
    }


def _catalog_launcher_snapshots(run_dir: Path, expected_sha: Any) -> dict[str, Any]:
    """Verify per-attempt copies of the exact wrapper bytes Slurm executed."""

    return _catalog_attempt_file_snapshots(
        run_dir, expected_sha, filename="launcher.sh", label="launcher"
    )


def _catalog_runtime_snapshots(run_dir: Path, expected_sha: Any) -> dict[str, Any]:
    """Verify per-attempt Python/package/GPU runtime inventories."""

    return _catalog_attempt_file_snapshots(
        run_dir, expected_sha, filename="runtime.txt", label="runtime"
    )


def _catalog_layout(
    run_dir: Path,
    identity: Mapping[str, Any],
    recovery: Mapping[str, Any],
    *,
    legacy: bool,
) -> dict[str, Any]:
    """Cross-check durable path components against immutable naming metadata."""

    canonical_text = recovery.get("canonical_path")
    canonical = Path(str(canonical_text)) if canonical_text else run_dir
    expected_run_name = identity.get("RUN_NAME")
    expected_experiment = identity.get("RUN_EXPERIMENT")
    run_name_matches = (
        canonical.name == expected_run_name
        if not legacy and isinstance(expected_run_name, str) and expected_run_name
        else None
    )
    experiment_matches = (
        canonical.parent.name == expected_experiment
        if not legacy
        and isinstance(expected_experiment, str)
        and expected_experiment
        else None
    )
    errors: list[str] = []
    if run_name_matches is False:
        errors.append(
            "canonical directory basename does not match manifest RUN_NAME"
        )
    if experiment_matches is False:
        errors.append(
            "canonical parent directory basename does not match manifest RUN_EXPERIMENT"
        )
    return {
        "canonical_path": os.fspath(canonical),
        "actual_run_name": canonical.name,
        "expected_run_name": expected_run_name,
        "run_name_matches": run_name_matches,
        "actual_experiment": canonical.parent.name,
        "expected_experiment": expected_experiment,
        "experiment_matches": experiment_matches,
        "valid": not errors,
        "errors": errors,
    }


def _catalog_record(
    run_dir: Path,
    root: Path,
    *,
    legacy: bool,
    kind_override: str | None = None,
    recovery_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if legacy:
        manifest = {
            "present": False,
            "valid": None,
            "schema_version": None,
            "created_at": None,
            "legacy_adopted": None,
            "resolved_config_sha256": None,
            "errors": [],
        }
        identity: dict[str, Any] = {}
        provenance: dict[str, Any] = {}
    else:
        manifest, identity, provenance = _catalog_manifest(run_dir)

    if recovery_override is not None:
        recovery = dict(recovery_override)
        recovery_snapshot = run_dir
    elif legacy:
        recovery = _empty_recovery(run_dir)
        recovery_snapshot = None
    else:
        recovery, recovery_snapshot = _catalog_recovery_for_run(run_dir)
    artifact_dir = recovery_snapshot if recovery_snapshot is not None else run_dir
    record_kind = kind_override or (
        "legacy"
        if legacy
        else ("recovery_index" if recovery.get("indexed") else "manifest")
    )
    layout = _catalog_layout(run_dir, identity, recovery, legacy=legacy)

    run_id = identity.get("RUN_ID")
    lifecycle = _catalog_lifecycle(artifact_dir, run_id)
    checkpoint = _catalog_checkpoint(artifact_dir)
    sync_complete = _catalog_sync_complete(run_dir)
    symlinks = _catalog_symlinks(artifact_dir)
    artifacts, artifact_bytes = _catalog_artifacts(artifact_dir)
    config = _catalog_config(artifact_dir, manifest["resolved_config_sha256"])
    source_snapshot = _catalog_source_snapshot(
        artifact_dir, provenance.get("source_id")
    )
    launcher_snapshots = _catalog_launcher_snapshots(
        artifact_dir, identity.get("RUN_LAUNCHER_SHA256")
    )
    runtime_snapshots = _catalog_runtime_snapshots(
        artifact_dir, identity.get("RUN_RUNTIME_SHA256")
    )
    warnings: list[str] = []
    if legacy:
        warnings.append("legacy run has no immutable manifest")
    elif not manifest["valid"]:
        warnings.append("manifest is invalid")
    if not legacy and record_kind == "manifest":
        if sync_complete["state"] == "missing":
            warnings.append(
                "final publish integrity marker is missing (run may still be active)"
            )
        elif not sync_complete["valid"]:
            warnings.append("final publish critical artifact integrity check failed")
    if checkpoint["state"] == "missing":
        warnings.append("checkpoint is missing")
    elif not checkpoint["valid"]:
        warnings.append(f"checkpoint state is {checkpoint['state']}")
    if not (artifacts["dreamer_config.yaml"] or artifacts["config.yaml"]):
        warnings.append("configuration artifact is missing")
    if not config["valid"]:
        warnings.append("configuration integrity check failed")
    elif (
        not legacy
        and config["sha256"] is not None
        and config["manifest_sha256"] is None
    ):
        warnings.append("configuration artifact has no manifest digest")
    if not legacy and source_snapshot["present"] is False:
        warnings.append("effective source snapshot is missing")
    elif source_snapshot["valid"] is False:
        warnings.append("effective source snapshot failed verification")
    if launcher_snapshots["valid"] is False:
        warnings.append("effective launcher snapshot failed verification")
    if runtime_snapshots["valid"] is False:
        warnings.append("runtime environment snapshot failed verification")
    if not layout["valid"]:
        warnings.append("durable directory naming does not match the run manifest")
    if (
        symlinks["broken"]
        or symlinks["absolute"]
        or symlinks["escaping"]
        or symlinks["errors"]
    ):
        warnings.append("run contains non-portable or broken symlinks")
    if recovery["state"] in {"available", "available_with_errors"}:
        warnings.append("latest durable artifacts are in a periodic recovery snapshot")
    elif recovery["state"] == "fallback_available":
        warnings.append(
            "recovery current pointer is invalid; a previous verified generation is available"
        )
    elif recovery["state"] == "invalid":
        warnings.append("periodic recovery metadata or artifacts are invalid")
    recovery_index = recovery.get("index", {})
    if recovery_index.get("present") and recovery_index.get("valid") is False:
        warnings.append("canonical recovery index failed verification")

    return {
        "path": os.fspath(run_dir),
        "relative_path": os.fspath(run_dir.relative_to(root)),
        "kind": record_kind,
        "identity": identity,
        "provenance": provenance,
        "manifest": manifest,
        "lifecycle": lifecycle,
        "checkpoint": checkpoint,
        "sync_complete": sync_complete,
        "config": config,
        "source_snapshot": source_snapshot,
        "launcher_snapshots": launcher_snapshots,
        "runtime_snapshots": runtime_snapshots,
        "layout": layout,
        "artifacts": artifacts,
        "artifact_bytes": artifact_bytes,
        "symlinks": symlinks,
        "recovery": recovery,
        "warnings": warnings,
    }


def _catalog_transaction_residue(
    path: Path,
    root: Path,
    *,
    kind: str,
    canonical_path: Path,
) -> dict[str, Any]:
    entry_type, _ = _entry_type(path)
    errors: list[str] = []
    manifest: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    sync_complete: dict[str, Any] | None = None
    recovery_index: dict[str, Any] | None = None
    archive_metadata: Any = None
    if entry_type != "directory":
        errors.append("transaction residue is not a real directory")
    else:
        if _entry_type(path / MANIFEST_NAME)[0] != "missing":
            manifest, _identity, _provenance = _catalog_manifest(path)
            if not manifest["valid"]:
                errors.append("transaction residue manifest is invalid")
        checkpoint = _catalog_checkpoint(path)
        sync_complete = _catalog_sync_complete(path)

        if kind == "recovery_index_stage":
            expected_store = path.parent / f".{canonical_path.name}.recovery"
            recovery_index = _catalog_recovery_index(
                path, expected_store, expected_run_uid=None
            )
            if recovery_index["valid"] is not True:
                errors.append("staged recovery index is invalid")

        if kind in {"scratch_recovery_archive", "scratch_recovery_publish_stage"}:
            metadata_path = path / "recovery_metadata.json"
            try:
                archive_metadata = _read_json_file(metadata_path)
            except (OSError, StorageError) as exc:
                errors.append(str(exc))
            if not isinstance(archive_metadata, dict):
                errors.append("scratch recovery metadata must be a JSON object")

    if kind in {"publish_stage", "publish_previous"}:
        if sync_complete is None or not sync_complete["valid"]:
            errors.append("publish transaction has no valid final integrity marker")
        if manifest is None:
            errors.append("publish transaction has no run manifest")
    elif kind == "recovery_index_stage" and manifest is None:
        errors.append("staged recovery index has no run manifest")

    recoverable = bool(
        checkpoint is not None and checkpoint.get("recoverable")
    ) or bool(sync_complete is not None and sync_complete.get("valid"))
    return {
        "path": os.fspath(path),
        "relative_path": os.fspath(path.relative_to(root)),
        "kind": kind,
        "canonical_path": os.fspath(canonical_path),
        "entry_type": entry_type,
        "valid": not errors,
        "recoverable": recoverable,
        "manifest": manifest,
        "checkpoint": checkpoint,
        "sync_complete": sync_complete,
        "recovery_index": recovery_index,
        "archive_metadata": archive_metadata,
        "errors": errors,
    }


def _catalog_recovery_residues(
    recovery: Mapping[str, Any], root: Path
) -> list[dict[str, Any]]:
    residues: list[dict[str, Any]] = []
    pointer_generation = recovery.get("pointer_generation")
    selected_generation = recovery.get("generation")
    canonical_path = recovery.get("canonical_path")
    for generation in recovery.get("generations", []):
        directory_name = generation.get("directory_name")
        if directory_name == pointer_generation and generation.get("valid"):
            continue
        if directory_name == selected_generation and recovery.get("fallback_used"):
            kind = "recovery_generation_fallback"
        elif directory_name == pointer_generation:
            kind = "recovery_generation_current_invalid"
        elif generation.get("valid"):
            kind = "recovery_generation_unpointed"
        else:
            kind = "recovery_generation_invalid"
        path = Path(str(generation["path"]))
        residues.append(
            {
                "path": os.fspath(path),
                "relative_path": os.fspath(path.relative_to(root)),
                "kind": kind,
                "canonical_path": canonical_path,
                "entry_type": "directory",
                "valid": bool(generation.get("valid")),
                "recoverable": bool(generation.get("valid")),
                "details": generation,
                "errors": list(generation.get("errors", [])),
            }
        )
    for stage in recovery.get("stages", []):
        path = Path(str(stage["path"]))
        residues.append(
            {
                "path": os.fspath(path),
                "relative_path": os.fspath(path.relative_to(root)),
                "kind": "recovery_stage",
                "canonical_path": canonical_path,
                "entry_type": "directory",
                "valid": bool(stage.get("valid")),
                "recoverable": bool(stage.get("valid")),
                "details": stage,
                "errors": list(stage.get("errors", [])),
            }
        )
    for residue in recovery.get("store_residues", []):
        path = Path(str(residue["path"]))
        residues.append(
            {
                "path": os.fspath(path),
                "relative_path": os.fspath(path.relative_to(root)),
                "kind": residue["kind"],
                "canonical_path": canonical_path,
                "entry_type": residue["entry_type"],
                "valid": bool(residue["valid"]),
                "recoverable": False,
                "details": residue,
                "errors": list(residue.get("errors", [])),
            }
        )
    if recovery.get("state") == "invalid" and not residues:
        store_path = recovery.get("store_path")
        if store_path:
            path = Path(str(store_path))
            residues.append(
                {
                    "path": os.fspath(path),
                    "relative_path": os.fspath(path.relative_to(root)),
                    "kind": "recovery_store_invalid",
                    "canonical_path": canonical_path,
                    "entry_type": _entry_type(path)[0],
                    "valid": False,
                    "recoverable": bool(recovery.get("recoverable")),
                    "details": None,
                    "errors": list(recovery.get("errors", [])),
                }
            )
    return residues


def _duplicate_values(
    records: Sequence[Mapping[str, Any]], identity_key: str
) -> list[dict[str, Any]]:
    paths_by_value: dict[str, list[str]] = {}
    for record in records:
        identity = record.get("identity", {})
        value = identity.get(identity_key) if isinstance(identity, dict) else None
        if not isinstance(value, str) or not value:
            continue
        paths_by_value.setdefault(value, []).append(str(record["path"]))
    return [
        {"value": value, "paths": sorted(paths)}
        for value, paths in sorted(paths_by_value.items())
        if len(paths) > 1
    ]


def catalog_runs(
    root: os.PathLike[str] | str, *, include_legacy: bool = False
) -> dict[str, Any]:
    """Return a stable, read-only inventory of durable experiment runs."""

    catalog_root = _real_directory(root, label="catalog root")
    candidates: dict[Path, bool] = {}
    recovery_stores: set[Path] = set()
    transaction_residue_specs: dict[Path, tuple[str, Path]] = {}
    scan_errors: list[str] = []

    def walk_error(exc: OSError) -> None:
        scan_errors.append(str(exc))

    for directory, directory_names, file_names in os.walk(
        catalog_root, followlinks=False, onerror=walk_error
    ):
        directory_names.sort()
        file_names.sort()
        current = Path(directory)

        excluded_directories: set[str] = set()
        for name in [*directory_names, *file_names]:
            child = current / name
            recovery_match = _RECOVERY_STORE_PATTERN.fullmatch(name)
            if recovery_match:
                recovery_stores.add(child)
                if name in directory_names:
                    excluded_directories.add(name)
                continue
            matched_residue = False
            for pattern, kind in _TRANSACTION_RESIDUE_PATTERNS:
                match = pattern.fullmatch(name)
                if match is None:
                    continue
                canonical = current / match.group("run_name")
                transaction_residue_specs[child] = (kind, canonical)
                matched_residue = True
                break
            if not matched_residue:
                archive_match = _SCRATCH_RECOVERY_ARCHIVE_PATTERN.fullmatch(name)
                if archive_match is not None:
                    kind = (
                        "scratch_recovery_publish_stage"
                        if archive_match.group("stage")
                        else "scratch_recovery_archive"
                    )
                    transaction_residue_specs[child] = (
                        kind,
                        current / archive_match.group("run_name"),
                    )
                    matched_residue = True
            if matched_residue and name in directory_names:
                excluded_directories.add(name)

        modern_candidate = (
            MANIFEST_NAME in file_names or MANIFEST_NAME in directory_names
        )
        if modern_candidate:
            candidates[current] = False
        elif include_legacy and _looks_like_legacy_run(current):
            candidates[current] = True

        # A manifest defines the run root; nested artifacts cannot define additional
        # runs.
        if modern_candidate:
            directory_names[:] = []
            continue

        # Checkpoint/replay trees can contain very many directories but cannot
        # contain separate run roots. Avoid traversing those artifact internals.
        directory_names[:] = [
            name
            for name in directory_names
            if name not in {"checkpoint.pkl", "checkpoint.pkl.incomplete", "replay"}
            and name not in excluded_directories
            and _entry_type(current / name)[0] != "symlink"
        ]

    records = [
        _catalog_record(path, catalog_root, legacy=candidates[path])
        for path in sorted(candidates, key=lambda candidate: os.fspath(candidate))
    ]
    represented_stores = {
        record["recovery"]["store_path"]: record["recovery"]
        for record in records
        if record["recovery"]["state"] != "none"
    }
    residues = [
        _catalog_transaction_residue(
            path,
            catalog_root,
            kind=kind,
            canonical_path=canonical,
        )
        for path, (kind, canonical) in sorted(
            transaction_residue_specs.items(), key=lambda item: os.fspath(item[0])
        )
    ]
    for store in sorted(recovery_stores, key=os.fspath):
        recovery = represented_stores.get(os.fspath(store))
        if recovery is None:
            recovery, snapshot = _catalog_recovery_store(store)
        else:
            snapshot_path = recovery.get("snapshot_path")
            snapshot = Path(str(snapshot_path)) if snapshot_path else None
        residues.extend(_catalog_recovery_residues(recovery, catalog_root))
        if os.fspath(store) in represented_stores:
            continue
        if snapshot is None:
            scan_errors.extend(
                f"{store}: {message}" for message in recovery["errors"]
            )
            continue
        if recovery["fallback_used"]:
            recovery["state"] = "orphan_fallback_available"
        elif recovery["state"] == "available":
            recovery["state"] = "orphan_available"
        else:
            recovery["state"] = "orphan_available_with_errors"
        records.append(
            _catalog_record(
                snapshot,
                catalog_root,
                legacy=False,
                kind_override="orphan_recovery",
                recovery_override=recovery,
            )
        )
    records.sort(key=lambda record: str(record["path"]))
    residues.sort(key=lambda residue: (str(residue["path"]), str(residue["kind"])))
    checkpoint_counts: dict[str, int] = {}
    for record in records:
        state = str(record["checkpoint"]["state"])
        checkpoint_counts[state] = checkpoint_counts.get(state, 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "root": os.fspath(catalog_root),
        "include_legacy": include_legacy,
        "runs": records,
        "residues": residues,
        "scan_errors": scan_errors,
        "summary": {
            "total": len(records),
            "manifest_runs": sum(
                record["kind"] in {"manifest", "recovery_index"}
                for record in records
            ),
            "legacy_runs": sum(record["kind"] == "legacy" for record in records),
            "recovery_index_runs": sum(
                record["kind"] == "recovery_index" for record in records
            ),
            "orphan_recoveries": sum(
                record["kind"] == "orphan_recovery" for record in records
            ),
            "valid_final_publishes": sum(
                record["kind"] == "manifest"
                and record["sync_complete"]["valid"] is True
                for record in records
            ),
            "missing_final_publish_markers": sum(
                record["kind"] == "manifest"
                and record["sync_complete"]["state"] == "missing"
                for record in records
            ),
            "invalid_final_publish_markers": sum(
                record["kind"] == "manifest"
                and record["sync_complete"]["state"] == "invalid"
                for record in records
            ),
            "invalid_recovery_indexes": sum(
                record["recovery"].get("index", {}).get("present")
                and record["recovery"].get("index", {}).get("valid") is False
                for record in records
            ),
            "transaction_residues": len(residues),
            "recoverable_residues": sum(
                bool(residue["recoverable"]) for residue in residues
            ),
            "invalid_manifests": sum(
                record["manifest"]["valid"] is False for record in records
            ),
            "layout_mismatches": sum(
                record["layout"]["valid"] is False for record in records
            ),
            "symlink_hazards": sum(
                bool(
                    record["symlinks"]["broken"]
                    or record["symlinks"]["absolute"]
                    or record["symlinks"]["escaping"]
                    or record["symlinks"]["errors"]
                )
                for record in records
            ),
            "config_mismatches": sum(
                record["config"]["matches_manifest"] is False
                for record in records
            ),
            "checkpoint_states": dict(sorted(checkpoint_counts.items())),
            "duplicate_run_ids": _duplicate_values(records, "RUN_ID"),
            "duplicate_run_names": _duplicate_values(records, "RUN_NAME"),
            "duplicate_wandb_ids": _duplicate_values(records, "RUN_WANDB_ID"),
        },
    }


def _tsv_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).replace("\t", "\\t").replace("\r", "\\r").replace("\n", "\\n")


def _emit_catalog_tsv(catalog: Mapping[str, Any]) -> None:
    columns = (
        "path",
        "kind",
        "manifest_valid",
        "layout_valid",
        "run_id",
        "run_name",
        "seed",
        "agent",
        "model",
        "status",
        "event",
        "checkpoint",
        "sync_complete",
        "source_id",
        "commit",
        "dirty",
        "broken_links",
        "absolute_links",
        "escaping_links",
        "recovery_state",
        "recovery_snapshot",
    )
    print("\t".join(columns))
    for record in catalog["runs"]:
        identity = record["identity"]
        provenance = record["provenance"]
        lifecycle = record["lifecycle"]
        symlinks = record["symlinks"]
        recovery = record["recovery"]
        row = (
            record["path"],
            record["kind"],
            record["manifest"]["valid"],
            record["layout"]["valid"],
            identity.get("RUN_ID"),
            identity.get("RUN_NAME"),
            identity.get("RUN_SEED"),
            identity.get("RUN_AGENT"),
            identity.get("RUN_MODEL"),
            lifecycle.get("status"),
            lifecycle.get("event"),
            record["checkpoint"]["state"],
            record["sync_complete"]["state"],
            provenance.get("source_id"),
            provenance.get("commit"),
            provenance.get("dirty"),
            symlinks["broken"],
            symlinks["absolute"],
            symlinks["escaping"],
            recovery["state"],
            recovery["snapshot_path"],
        )
        print("\t".join(_tsv_text(value) for value in row))
    for residue in catalog.get("residues", []):
        checkpoint = residue.get("checkpoint")
        checkpoint_state = (
            checkpoint.get("state") if isinstance(checkpoint, dict) else None
        )
        row = (
            residue.get("path"),
            residue.get("kind"),
            None,
            residue.get("valid"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            checkpoint_state,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "recoverable_residue"
            if residue.get("recoverable")
            else ("valid_residue" if residue.get("valid") else "invalid_residue"),
            None,
        )
        print("\t".join(_tsv_text(value) for value in row))


def _add_source_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--source-id")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    config_parser = commands.add_parser(
        "config-id", help="compute a restart-stable effective run config identity"
    )
    config_parser.add_argument("--launcher-sha256", required=True)
    config_parser.add_argument("run_command", nargs=argparse.REMAINDER)

    source_parser = commands.add_parser(
        "source-id", help="compute the effective Git source-tree identity"
    )
    source_parser.add_argument("--repo", type=Path, required=True)
    source_parser.add_argument("--json", action="store_true")

    snapshot_parser = commands.add_parser(
        "source-snapshot",
        help="create or verify an immutable effective-source snapshot",
    )
    snapshot_commands = snapshot_parser.add_subparsers(
        dest="snapshot_command", required=True
    )
    snapshot_create = snapshot_commands.add_parser("create")
    snapshot_create.add_argument("--repo", type=Path, required=True)
    snapshot_create.add_argument("--snapshot-dir", type=Path, required=True)
    snapshot_create.add_argument("--source-id", required=True)
    snapshot_verify = snapshot_commands.add_parser("verify")
    snapshot_verify.add_argument("--snapshot-dir", type=Path, required=True)
    snapshot_verify.add_argument("--source-id", required=True)

    bundle_parser = commands.add_parser(
        "source-bundle",
        help="create, ensure, or verify a runnable immutable source checkout",
    )
    bundle_commands = bundle_parser.add_subparsers(
        dest="bundle_command", required=True
    )
    for name in ("create", "ensure"):
        bundle_create = bundle_commands.add_parser(name)
        bundle_create.add_argument("--repo", type=Path, required=True)
        bundle_create.add_argument("--bundle-dir", type=Path, required=True)
        bundle_create.add_argument("--source-id", required=True)
    bundle_verify = bundle_commands.add_parser("verify")
    bundle_verify.add_argument("--bundle-dir", type=Path, required=True)
    bundle_verify.add_argument("--source-id", required=True)

    manifest_parser = commands.add_parser(
        "manifest", help="initialize or validate immutable run identity"
    )
    manifest_commands = manifest_parser.add_subparsers(
        dest="manifest_command", required=True
    )
    manifest_init = manifest_commands.add_parser("init")
    manifest_init.add_argument("--run-dir", type=Path, required=True)
    manifest_init.add_argument("--mode", choices=("new", "resume"), required=True)
    _add_source_options(manifest_init)
    manifest_validate = manifest_commands.add_parser("validate")
    manifest_validate.add_argument("--run-dir", type=Path, required=True)
    _add_source_options(manifest_validate)
    manifest_fork_validate = manifest_commands.add_parser("fork-validate")
    manifest_fork_validate.add_argument("--run-dir", type=Path, required=True)
    manifest_fork_validate.add_argument("--parent-run-id", required=True)
    manifest_fork_validate.add_argument("--parent-source-id", required=True)
    manifest_fork_validate.add_argument("--parent-marker", required=True)
    manifest_fork = manifest_commands.add_parser("fork")
    manifest_fork.add_argument("--run-dir", type=Path, required=True)
    manifest_fork.add_argument("--parent-run-id", required=True)
    manifest_fork.add_argument("--parent-source-id", required=True)
    manifest_fork.add_argument("--parent-marker", required=True)
    manifest_fork.add_argument("--parent-artifact", required=True)
    _add_source_options(manifest_fork)

    event_parser = commands.add_parser(
        "event", help="append lifecycle events or atomically update run status"
    )
    event_commands = event_parser.add_subparsers(dest="event_command", required=True)
    event_append = event_commands.add_parser("append")
    event_append.add_argument("--run-dir", type=Path, required=True)
    event_append.add_argument("--event", required=True)
    event_append.add_argument("--status")
    event_append.add_argument("--field", action="append", default=[])
    event_update = event_commands.add_parser("update")
    event_update.add_argument("--run-dir", type=Path, required=True)
    event_update.add_argument("--status", required=True)
    event_update.add_argument("--field", action="append", default=[])

    links_parser = commands.add_parser(
        "materialize-links", help="make staged external symlinks self-contained"
    )
    links_parser.add_argument("--source", type=Path, required=True)
    links_parser.add_argument("--stage", type=Path, required=True)
    links_parser.add_argument("--allowed-root", type=Path, action="append", default=[])

    catalog_parser = commands.add_parser(
        "catalog", help="read-only inventory and health audit of durable runs"
    )
    catalog_parser.add_argument("--root", type=Path, required=True)
    catalog_parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="also report config/output directories that predate run manifests",
    )
    catalog_parser.add_argument("--json", action="store_true")

    wandb_parser = commands.add_parser(
        "wandb-continuation",
        help="resolve an integrity-certified remote W&B run and checkpoint step",
    )
    wandb_parser.add_argument("--run-dir", type=Path, required=True)
    wandb_parser.add_argument("--expected-run-id", required=True)
    wandb_parser.add_argument("--expected-source-id")
    wandb_parser.add_argument("--live-run-dir", type=Path, action="append", default=[])

    high_water_parser = commands.add_parser(
        "checkpoint-high-water",
        help="commit or validate the durable checkpoint progress fence",
    )
    high_water_commands = high_water_parser.add_subparsers(
        dest="high_water_command", required=True
    )
    for name in ("commit", "seed"):
        high_water_commit = high_water_commands.add_parser(name)
        high_water_commit.add_argument("--marker", type=Path, required=True)
        high_water_commit.add_argument("--run-dir", type=Path, required=True)
        high_water_commit.add_argument("--checkpoint", required=True)
        high_water_commit.add_argument(
            "--max-gap-steps", type=int, required=True
        )
    high_water_check = high_water_commands.add_parser("check")
    high_water_check.add_argument("--marker", type=Path, required=True)
    high_water_check.add_argument("--run-dir", type=Path, required=True)
    high_water_check.add_argument("--checkpoint", required=True)
    high_water_check.add_argument(
        "--minimum", choices=("floor", "high"), default="floor"
    )
    high_water_check.add_argument("--expected-run-id")
    high_water_repair = high_water_commands.add_parser("repair-latest")
    high_water_repair.add_argument("--marker", type=Path, required=True)
    high_water_repair.add_argument("--run-dir", type=Path, required=True)
    high_water_repair.add_argument("--expected-run-id")

    return parser


def _emit_json(value: Any) -> None:
    sys.stdout.buffer.write(_canonical_json_bytes(value))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "config-id":
            print(
                run_config_id(
                    arguments.run_command,
                    launcher_sha256=arguments.launcher_sha256,
                )
            )
            return 0

        if arguments.command == "source-id":
            value = source_identity(arguments.repo)
            if arguments.json:
                _emit_json(value)
            else:
                print(value["source_id"])
            return 0

        if arguments.command == "source-snapshot":
            if arguments.snapshot_command == "create":
                value = create_source_snapshot(
                    arguments.repo,
                    arguments.snapshot_dir,
                    source_id=arguments.source_id,
                )
            else:
                value = verify_source_snapshot(
                    arguments.snapshot_dir, source_id=arguments.source_id
                )
            _emit_json(value)
            return 0

        if arguments.command == "source-bundle":
            if arguments.bundle_command == "create":
                value = create_source_bundle(
                    arguments.repo,
                    arguments.bundle_dir,
                    source_id=arguments.source_id,
                )
            elif arguments.bundle_command == "ensure":
                value = ensure_source_bundle(
                    arguments.repo,
                    arguments.bundle_dir,
                    source_id=arguments.source_id,
                )
            else:
                value = verify_source_bundle(
                    arguments.bundle_dir, source_id=arguments.source_id
                )
            _emit_json(value)
            return 0

        if arguments.command == "manifest":
            if arguments.manifest_command == "init":
                value = initialize_manifest(
                    arguments.run_dir,
                    mode=arguments.mode,
                    repo=arguments.repo,
                    source_id=arguments.source_id,
                )
                _emit_json(
                    {
                        "manifest": os.fspath(arguments.run_dir / MANIFEST_NAME),
                        "mode": arguments.mode,
                        "run_id": value["identity"]["RUN_ID"],
                    }
                )
            elif arguments.manifest_command == "validate":
                value = validate_manifest(
                    arguments.run_dir,
                    repo=arguments.repo,
                    source_id=arguments.source_id,
                )
                _emit_json(
                    {
                        "manifest": os.fspath(arguments.run_dir / MANIFEST_NAME),
                        "valid": True,
                        "run_id": value["identity"]["RUN_ID"],
                    }
                )
            elif arguments.manifest_command == "fork-validate":
                value = validate_fork_parent(
                    arguments.run_dir,
                    parent_run_id=arguments.parent_run_id,
                    parent_source_id=arguments.parent_source_id,
                    parent_marker_name=arguments.parent_marker,
                )
                _emit_json(
                    {
                        "checkpoint": value["parent_checkpoint"],
                        "manifest": os.fspath(arguments.run_dir / MANIFEST_NAME),
                        "parent_run_id": value["parent_identity"]["RUN_ID"],
                        "valid": True,
                    }
                )
            else:
                value = fork_manifest(
                    arguments.run_dir,
                    parent_run_id=arguments.parent_run_id,
                    parent_source_id=arguments.parent_source_id,
                    parent_marker_name=arguments.parent_marker,
                    parent_artifact=arguments.parent_artifact,
                    repo=arguments.repo,
                    source_id=arguments.source_id,
                )
                _emit_json(
                    {
                        "checkpoint": value["lineage"]["parent_checkpoint"],
                        "manifest": os.fspath(arguments.run_dir / MANIFEST_NAME),
                        "parent_run_id": value["lineage"]["parent_identity"]["RUN_ID"],
                        "run_id": value["identity"]["RUN_ID"],
                    }
                )
            return 0

        if arguments.command == "event":
            fields = _validated_fields(arguments.field)
            if arguments.event_command == "append":
                value = append_event(
                    arguments.run_dir,
                    event=arguments.event,
                    status_value=arguments.status,
                    fields=fields,
                )
            else:
                value = update_status(
                    arguments.run_dir,
                    status_value=arguments.status,
                    fields=fields,
                )
            _emit_json(value)
            return 0

        if arguments.command == "materialize-links":
            value = materialize_links(
                arguments.source,
                arguments.stage,
                allowed_roots=arguments.allowed_root,
            )
            _emit_json(value)
            return 0

        if arguments.command == "catalog":
            value = catalog_runs(
                arguments.root, include_legacy=arguments.include_legacy
            )
            if arguments.json:
                _emit_json(value)
            else:
                _emit_catalog_tsv(value)
            return 0

        if arguments.command == "wandb-continuation":
            _emit_json(
                wandb_continuation(
                    arguments.run_dir,
                    expected_run_id=arguments.expected_run_id,
                    expected_source_id=arguments.expected_source_id,
                    live_run_dirs=arguments.live_run_dir,
                )
            )
            return 0

        if arguments.command == "checkpoint-high-water":
            if arguments.high_water_command in {"commit", "seed"}:
                value = commit_checkpoint_high_water(
                    arguments.marker,
                    arguments.run_dir,
                    arguments.checkpoint,
                    max_gap_steps=arguments.max_gap_steps,
                    seed=arguments.high_water_command == "seed",
                )
            elif arguments.high_water_command == "check":
                value = check_checkpoint_high_water(
                    arguments.marker,
                    arguments.run_dir,
                    arguments.checkpoint,
                    minimum=arguments.minimum,
                    expected_run_id=arguments.expected_run_id,
                )
            else:
                value = repair_first_checkpoint_latest(
                    arguments.marker,
                    arguments.run_dir,
                    expected_run_id=arguments.expected_run_id,
                )
            _emit_json(value)
            return 0

        parser.error(f"unknown command: {arguments.command}")
    except (OSError, StorageError) as exc:
        print(f"run_storage: error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
