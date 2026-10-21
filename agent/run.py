from __future__ import annotations

import collections
import contextlib
import hashlib
import io
import inspect
import json
import os
import pathlib
import pickle
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from functools import partial as bind

import embodied
import elements
import numpy as np

from .adaptation_manifest import (
    ALGORITHM2_OFFLINE_SAVE_EVERY,
    ALGORITHM2_OFFLINE_UPDATES,
    ALGORITHM2_REFERENCE_TASK_STEPS,
    checkpoint_sha256,
    file_sha256,
    _safe_pickle_integer,
    _load_compatibility_certificate,
)
from .adaptation_protocol import (
    WALKER_REWARD_LABELS, is_walker_task, is_walker_source_transfer,
    validate_env_config, validate_parent_step, validate_walker_source,
    TRANSFER_COMPONENTS, validate_protocol, offline_updates as protocol_offline_updates,
    transfer_components, source_components,
)


class _PreemptionHandler:
    def __init__(self):
        self.requested = False
        self.signum = None
        self._previous = {}

    def install(self):
        for name in ("SIGUSR1", "SIGTERM"):
            signum = getattr(signal, name, None)
            if signum is None:
                continue
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)
        return self

    def restore(self):
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous.clear()

    def _handle(self, signum, frame):
        del frame
        self.requested = True
        self.signum = signum
        print(f"Received signal {signum}; will save JAX checkpoint and exit.")


def _fsync_directory(path):
    descriptor = os.open(
        os.fspath(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_fsynced_file(path, payload, *, exclusive=True):
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    flags |= os.O_EXCL if exclusive else os.O_TRUNC
    descriptor = os.open(os.fspath(path), flags, 0o660)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view):]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _commit_checkpoint_high_water(run_dir, checkpoint):
    """Persist progress before publishing a local checkpoint."""

    marker = os.environ.get("SCRATCH_SYNC_CHECKPOINT_HIGH_WATER")
    if not marker:
        return
    helper = os.environ.get("RUN_STORAGE_HELPER")
    if not helper:
        raise RuntimeError(
            "SCRATCH_SYNC_CHECKPOINT_HIGH_WATER requires RUN_STORAGE_HELPER"
        )
    gap = os.environ.get("SCRATCH_SYNC_MAX_RESUME_GAP_STEPS", "10000")
    command = (
        sys.executable,
        helper,
        "checkpoint-high-water",
        "commit",
        "--marker",
        marker,
        "--run-dir",
        os.fspath(run_dir),
        "--checkpoint",
        checkpoint,
        "--max-gap-steps",
        gap,
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        diagnostic = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            "Could not commit durable checkpoint high-water fence"
            + (f": {diagnostic}" if diagnostic else "")
        )


class _DurableCheckpoint(elements.Checkpoint):
    """Elements checkpoint writer with an atomic, fsynced commit boundary."""

    @elements.timer.section("checkpoint_save")
    def save(self, path=None, keys=None):
        assert self._directory or path
        if keys is None:
            savefns = {key: value.save for key, value in self._saveables.items()}
        else:
            assert all(not key.startswith("_") for key in keys), keys
            savefns = {key: self._saveables[key].save for key in keys}

        folder = None
        if path:
            target = pathlib.Path(os.fspath(path))
        else:
            folder = elements.timestamp(millis=True)
            if self._step is not None:
                folder += f"-{int(self._step):012d}"
            # Offline adaptation leaves the environment step at zero. Include its
            # update counter so generations can be ordered without loading them.
            offline = self._saveables.get("adaptation_offline")
            if offline is not None:
                updates = int(offline.completed_updates)
                if updates < 0 or updates >= 10**12:
                    raise RuntimeError(
                        f"Offline adaptation update counter is out of range: {updates}"
                    )
                folder += f"-u{updates:012d}"
            target = pathlib.Path(os.fspath(self._directory)) / folder
        print(f"Saving checkpoint: {target}")

        if not self._write:
            for savefn in savefns.values():
                value = savefn()
                if inspect.isgenerator(value):
                    for index, _ in enumerate(value):
                        assert index < 100_000, index
            print("Saved checkpoint.")
            return

        root = target.parent
        target_created = False
        try:
            try:
                root_before = root.lstat()
            except FileNotFoundError:
                root_before = None
                root.mkdir(parents=True, exist_ok=False)
            else:
                if stat.S_ISLNK(root_before.st_mode) or not stat.S_ISDIR(
                    root_before.st_mode
                ):
                    raise RuntimeError(
                        "Unsafe checkpoint root; expected a real directory: "
                        f"{root}"
                    )
            root_after = root.lstat()
            if stat.S_ISLNK(root_after.st_mode) or not stat.S_ISDIR(
                root_after.st_mode
            ):
                raise RuntimeError(
                    f"Unsafe checkpoint root after creation: {root}"
                )
            if root_before is not None and (
                root_before.st_dev,
                root_before.st_ino,
            ) != (
                root_after.st_dev,
                root_after.st_ino,
            ):
                raise RuntimeError(f"Checkpoint root changed while saving: {root}")

            # Flush the parent even when retrying an existing checkpoint root.
            _fsync_directory(root.parent)
            target.mkdir(exist_ok=False)
            target_created = True
            target_info = target.lstat()
            if stat.S_ISLNK(target_info.st_mode) or not stat.S_ISDIR(
                target_info.st_mode
            ):
                raise RuntimeError(
                    f"Unsafe checkpoint generation directory: {target}"
                )
            _fsync_directory(root)

            for name, savefn in savefns.items():
                try:
                    value = savefn()
                    if inspect.isgenerator(value):
                        for index, shard in enumerate(value):
                            assert index < 100_000, index
                            payload = pickle.dumps(shard)
                            _write_fsynced_file(
                                target / f"{name}-{index:04d}.pkl", payload
                            )
                    else:
                        _write_fsynced_file(
                            target / f"{name}.pkl", pickle.dumps(value)
                        )
                except Exception:
                    print(f"Error save {name!r} to checkpoint.")
                    raise

            # The private generation and its payloads become durable before
            # the completion marker.
            _fsync_directory(target)
            _write_fsynced_file(target / "done", b"")
            _fsync_directory(target)
            if folder:
                # Commit the durable marker before publishing the local generation
                # through `latest`; failure leaves the prior checkpoint active.
                _commit_checkpoint_high_water(root.parent, folder)
        except Exception:
            # Incomplete generations must not consume the rollback budget.
            if target_created:
                shutil.rmtree(target, ignore_errors=True)
                try:
                    _fsync_directory(root)
                except OSError as exc:
                    print(
                        "Could not fsync checkpoint cleanup after failed save: "
                        f"{exc}"
                    )
            raise

        # The root pointer is replaced atomically and fsynced, making it the
        # transaction's sole commit point.
        if folder:
            temporary = (
                root / f".latest.tmp.{os.getpid()}.{secrets.token_hex(16)}"
            )
            try:
                _write_fsynced_file(temporary, folder.encode("utf-8"))
                os.replace(temporary, root / "latest")
                _fsync_directory(root)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
            # Retention errors must not invalidate a committed checkpoint or
            # prevent replay from recording the commit.
            try:
                self._cleanup_completed()
                _fsync_directory(root)
            except Exception as exc:
                print(f"Checkpoint committed with retention cleanup debt: {exc}")
        print("Saved checkpoint.")

    def _cleanup_completed(self):
        if not self._keep:
            return
        root = pathlib.Path(os.fspath(self._directory))
        latest_payload = _read_regular_file(
            root / "latest", "checkpoint latest pointer"
        )
        try:
            latest = latest_payload.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise RuntimeError("Checkpoint latest pointer is not UTF-8.") from exc
        if not _OWN_CHECKPOINT_GENERATION.fullmatch(latest):
            raise RuntimeError(
                f"Checkpoint latest pointer is not writer-owned: {latest!r}"
            )
        completed = []
        incomplete = []
        for child in root.iterdir():
            if child.name == "latest":
                continue
            try:
                metadata = child.lstat()
            except FileNotFoundError:
                continue
            if _CHECKPOINT_LATEST_TEMP.fullmatch(child.name):
                if not stat.S_ISREG(metadata.st_mode):
                    print(
                        "Preserving unsafe checkpoint latest-pointer residue: "
                        f"{child}"
                    )
                else:
                    child.unlink()
                continue
            current_name = bool(
                _CURRENT_CHECKPOINT_GENERATION.fullmatch(child.name)
            )
            legacy_name = bool(
                _LEGACY_CHECKPOINT_GENERATION.fullmatch(child.name)
            )
            if not current_name and not legacy_name:
                print(
                    "Preserving unowned checkpoint-root entry during "
                    f"retention: {child}"
                )
                continue
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                print(f"Preserving unsafe checkpoint-root entry: {child}")
                continue
            done = child / "done"
            try:
                done_metadata = done.lstat()
            except FileNotFoundError:
                if legacy_name:
                    print(
                        "Preserving incomplete legacy checkpoint directory "
                        f"without current-writer provenance: {child}"
                    )
                else:
                    incomplete.append(child)
                continue
            if not stat.S_ISREG(done_metadata.st_mode):
                print(f"Preserving unsafe checkpoint completion marker: {done}")
                continue
            completed.append(child)
        current = root / latest
        if current not in completed:
            raise RuntimeError(
                f"Checkpoint latest pointer is not completed: {latest!r}"
            )
        others = sorted(child for child in completed if child != current)
        keep_others = max(0, self._keep - 1)
        old_completed = others[:-keep_others] if keep_others else others
        for child in incomplete + old_completed:
            shutil.rmtree(child)


def _save_checkpoint(checkpoint, after_save=None, *, retries=1):
    for attempt in range(retries + 1):
        try:
            checkpoint.save()
        except (OSError, RuntimeError):
            if attempt >= retries:
                raise
            print("Checkpoint save failed; retrying once after cleanup.")
        else:
            if after_save is not None:
                after_save()
            return


def _save_preempt_checkpoint_if_needed(
    handler, checkpoint, after_save=None
):
    if not handler.requested:
        return False
    print("Saving JAX checkpoint before Slurm preemption exit.")
    # Attempt one save before termination. Reset failed chunk futures so a
    # subsequent run can retry.
    _save_checkpoint(checkpoint, after_save, retries=0)
    print("Saved JAX checkpoint for Slurm preemption.")
    return True


def _save_final_checkpoint_if_complete(
    handler, checkpoint, step, target_steps, after_save=None
):
    if handler.requested or int(step) < int(target_steps):
        return False
    print("Saving final JAX checkpoint after successful completion.")
    _save_checkpoint(checkpoint, after_save)
    print("Saved final JAX checkpoint.")
    return True


def _numeric_timer_stats():
    stats = elements.timer.stats()
    stats.pop("summary", None)
    return {
        key: value for key, value in stats.items()
        if isinstance(value, (int, float, np.number)) and np.isfinite(value)
    }


class _AbsoluteStepSchedule:
    """Triggers once when an absolute step interval is crossed.

    Unlike a process-local relative clock, priming this schedule from a
    restored checkpoint preserves the original global boundaries after a
    preemption at an arbitrary step.
    """

    def __init__(self, every):
        self.every = int(every)
        if self.every <= 0:
            raise ValueError(
                f"Step schedule interval must be positive, got {self.every}."
            )
        self._bucket = None

    def prime(self, step):
        self._bucket = int(step) // self.every

    def __call__(self, step):
        bucket = int(step) // self.every
        if self._bucket is None:
            self._bucket = bucket
            return False
        if bucket <= self._bucket:
            return False
        self._bucket = bucket
        return True


def _step_interval_bounds(step, every):
    every = int(every)
    if every <= 0:
        raise ValueError(f"Step interval must be positive, got {every}.")
    start = (int(step) // every) * every
    return start, start + every - 1


def _require_real_directory(path, description):
    path = pathlib.Path(path)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing {description}: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(
            f"Unsafe {description} at {path}: expected a real directory."
        )
    return info


_SAFE_CHECKPOINT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CHECKPOINT_LATEST_TEMP = re.compile(
    r"^\.latest\.tmp\.[0-9]+(?:\.(?:[0-9]+|[0-9a-f]{32}))?$"
)
_CURRENT_CHECKPOINT_GENERATION = re.compile(
    r"^[0-9]{8}T[0-9]{6}F[0-9]{6}(?:-[0-9]{12})?(?:-u[0-9]{12})?$"
)
_LEGACY_CHECKPOINT_GENERATION = re.compile(
    r"^(?:[0-9]{1,20}|[0-9]{8}T[0-9]{6}(?:-[0-9]{12})?)$"
)
_OWN_CHECKPOINT_GENERATION = re.compile(
    rf"(?:{_CURRENT_CHECKPOINT_GENERATION.pattern}|"
    rf"{_LEGACY_CHECKPOINT_GENERATION.pattern})"
)


class _RestrictedReplayInventoryUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(
            f"Replay inventory cannot load global {module}.{name}."
        )

    def persistent_load(self, pid):
        del pid
        raise pickle.UnpicklingError(
            "Replay inventory cannot contain persistent IDs."
        )


def _stat_signature(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _read_regular_file(path, description):
    path = pathlib.Path(path)
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing {description}: {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(
            f"Unsafe {description} at {path}: expected a regular file."
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"Could not safely open {description}: {path}") from exc
    try:
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or _stat_signature(opened) != _stat_signature(before)
            ):
                raise RuntimeError(f"{description} changed while opening: {path}")
            payload = stream.read()
            after = os.fstat(stream.fileno())
            if _stat_signature(after) != _stat_signature(opened):
                raise RuntimeError(f"{description} changed while reading: {path}")
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    try:
        current = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"{description} disappeared while reading: {path}") from exc
    if _stat_signature(current) != _stat_signature(after):
        raise RuntimeError(f"{description} path changed while reading: {path}")
    return payload


def _read_replay_inventory(path):
    payload = _read_regular_file(path, "checkpoint replay inventory")
    stream = io.BytesIO(payload)
    value = _RestrictedReplayInventoryUnpickler(stream).load()
    if stream.read(1):
        raise RuntimeError(f"Replay inventory contains trailing data: {path}")
    return value


_ADAPTATION_COMPONENTS = TRANSFER_COMPONENTS


class _AdaptationLineage:
    """Checkpointed provenance for a task adaptation run."""

    def __init__(self, value):
        self.value = json.loads(json.dumps(value, sort_keys=True))

    def save(self):
        return self.value

    def load(self, value):
        if value != self.value:
            raise RuntimeError(
                "Adaptation checkpoint lineage does not match this launch."
            )


class _AdaptationOfflineState:
    """Checkpointed progress for the pre-execution Algorithm-2 phase."""

    _VERSION = 1

    def __init__(self, target_updates, save_every):
        self.target_updates = int(target_updates)
        self.save_every = int(save_every)
        if self.target_updates <= 0:
            raise ValueError("Offline adaptation target must be positive.")
        if self.save_every <= 0:
            raise ValueError("Offline adaptation save cadence must be positive.")
        self.completed_updates = 0
        self.complete = False

    def advance(self):
        if self.complete or self.completed_updates >= self.target_updates:
            raise RuntimeError("Offline adaptation advanced past completion.")
        self.completed_updates += 1

    def finish(self):
        if self.completed_updates != self.target_updates:
            raise RuntimeError(
                "Offline adaptation cannot finish before its exact target: "
                f"{self.completed_updates} != {self.target_updates}."
            )
        self.complete = True

    def save(self):
        return {
            "schema_version": self._VERSION,
            "target_updates": self.target_updates,
            "save_every": self.save_every,
            "completed_updates": self.completed_updates,
            "complete": self.complete,
        }

    def load(self, value):
        expected_fields = {
            "schema_version",
            "target_updates",
            "save_every",
            "completed_updates",
            "complete",
        }
        if not isinstance(value, dict) or set(value) != expected_fields:
            raise RuntimeError("Malformed offline adaptation checkpoint state.")
        if (
            not isinstance(value["schema_version"], int)
            or isinstance(value["schema_version"], bool)
            or value["schema_version"] != self._VERSION
        ):
            raise RuntimeError("Unsupported offline adaptation checkpoint state.")
        target = value["target_updates"]
        cadence = value["save_every"]
        if (
            not isinstance(target, int)
            or isinstance(target, bool)
            or target != self.target_updates
        ):
            raise RuntimeError(
                "Offline adaptation target changed across resume: "
                f"{value['target_updates']} != {self.target_updates}."
            )
        if (
            not isinstance(cadence, int)
            or isinstance(cadence, bool)
            or cadence != self.save_every
        ):
            raise RuntimeError(
                "Offline adaptation save cadence changed across resume: "
                f"{value['save_every']} != {self.save_every}."
            )
        completed = value["completed_updates"]
        complete = value["complete"]
        if (
            not isinstance(completed, int)
            or isinstance(completed, bool)
            or completed < 0
            or completed > self.target_updates
            or not isinstance(complete, bool)
            or complete != (completed == self.target_updates)
        ):
            raise RuntimeError("Inconsistent offline adaptation checkpoint state.")
        self.completed_updates = completed
        self.complete = complete


def _adaptation_enabled(args):
    return bool(_getattr(args, "adaptation_enabled", False))


def _adaptation_child_checkpoint_exists(checkpoint_root):
    root = pathlib.Path(checkpoint_root)
    if not root.exists() and not root.is_symlink():
        return False
    _require_real_directory(root, "adaptation child checkpoint root")
    latest = root / "latest"
    if latest.exists() or latest.is_symlink():
        _completed_checkpoint_generations(root)
        return True
    # Generations not published through `latest` cannot be resumed, even if
    # they contain a completion marker.
    for child in root.iterdir():
        info = child.lstat()
        if _CHECKPOINT_LATEST_TEMP.fullmatch(child.name):
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise RuntimeError(
                    f"Unsafe checkpoint latest-pointer residue: {child}"
                )
            continue
        if (
            not _CURRENT_CHECKPOINT_GENERATION.fullmatch(child.name)
            or stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
        ):
            raise RuntimeError(
                f"Unsafe pre-bootstrap child checkpoint entry: {child}"
            )
    return False


def _adaptation_lineage(args):
    protocol = validate_protocol(_getattr(args, "adaptation_protocol", "fresh_heads"))
    backend = str(_getattr(args, "adaptation_backend", "")).lower()
    if backend not in _ADAPTATION_COMPONENTS:
        raise ValueError(
            "Adaptation backend must be dreamerv3, r2dreamer, or ne_dreamer; "
            f"got {backend!r}."
        )
    parent_step = int(_getattr(args, "pretraining_environment_steps", 0))
    task = str(_getattr(args, "train_task", ""))
    walker = is_walker_task(task)
    validate_parent_step(task, parent_step)
    env_config = str(_getattr(args, "adaptation_env_config", "")) if walker else None
    validate_env_config(task, env_config)
    source_transfer = is_walker_source_transfer(env_config)
    parent_task = str(_getattr(args, "adaptation_parent_task", ""))
    parent_env_config = str(_getattr(args, "adaptation_parent_env_config", ""))
    if source_transfer:
        validate_walker_source(task, env_config, parent_task, parent_env_config, parent_step)
    adaptation_steps = int(args.steps)
    allowed_steps = (200_000,) if walker else (1_000_000, 5_000_000)
    if adaptation_steps not in allowed_steps:
        expected_steps = " or ".join(f"{value:,}" for value in allowed_steps)
        raise ValueError(
            f"Scientific adaptation must run exactly {expected_steps} "
            f"child steps; got {adaptation_steps}."
        )
    replay_length = int(
        _getattr(args, "adaptation_parent_replay_length", 0)
    )
    if replay_length < 1:
        raise ValueError(
            "Scientific adaptation requires a positive certified replay "
            f"sequence length; got {replay_length}."
        )
    offline_updates = int(_getattr(args, "adaptation_offline_updates", 0))
    offline_save_every = int(
        _getattr(args, "adaptation_offline_save_every", 0)
    )
    if offline_updates != protocol_offline_updates(protocol):
        raise ValueError(
            "Scientific adaptation requires exactly "
            f"{protocol_offline_updates(protocol)} pre-execution updates for {protocol}; got "
            f"{offline_updates}."
        )
    if offline_save_every != ALGORITHM2_OFFLINE_SAVE_EVERY:
        raise ValueError(
            "Scientific adaptation requires offline save cadence "
            f"{ALGORITHM2_OFFLINE_SAVE_EVERY}; got {offline_save_every}."
        )
    parent_source_digest = str(
        _getattr(args, "adaptation_parent_source_digest", "")
    )
    if not re.fullmatch(r"[0-9a-f]{64}", parent_source_digest):
        raise ValueError(
            "Scientific adaptation requires a lowercase parent source "
            "SHA-256."
        )
    strict_task_blind_pretraining = bool(
        _getattr(args, "adaptation_strict_task_blind_pretraining", False)
    )
    parent_algorithm = str(_getattr(args, "adaptation_parent_algorithm", "p2e"))
    if parent_algorithm not in ("p2e", "dreamer"):
        raise ValueError(f"Invalid adaptation parent algorithm: {parent_algorithm!r}")
    if parent_algorithm == "dreamer" and (
        (not source_transfer and backend != "dreamerv3") or str(_getattr(args, "adaptation_parent_idm", "")) != "none"
        or strict_task_blind_pretraining
    ):
        raise ValueError("Supervised control requires DreamerV3, IDM none, and strict blindness false.")
    if parent_algorithm == "p2e" and not strict_task_blind_pretraining:
        raise ValueError(
            "Scientific adaptation requires strict task-blind parent "
            "pretraining."
        )
    lineage = {
        "schema_version": 2,
        "study": ("walker_multitask_fresh_head_adaptation" if source_transfer
                  else "supervised_dreamer_fresh_head_adaptation" if parent_algorithm == "dreamer"
                  else "walker_few_shot_task_adaptation" if walker
                  else "extended_plan2explore_task_adaptation"),
        **({"parent_algorithm": "dreamer"} if parent_algorithm == "dreamer" else {}),
        "backend": backend,
        "idm": str(_getattr(args, "adaptation_parent_idm", "")),
        "parent_seed": int(_getattr(args, "adaptation_parent_seed", 0)),
        "adaptation_seed": int(_getattr(args, "train_seed", 0)),
        "task": task,
        **({
            "parent_algorithm": parent_algorithm,
            "parent_task": parent_task,
            "parent_env_config": parent_env_config,
            "replay_reward_key": f"label/reward/{task.removeprefix('dmc_')}",
            "replay_drop_keys": ["skill", "skill_start", *WALKER_REWARD_LABELS],
        } if source_transfer else {}),
        **({
            "env_config": env_config,
            "evaluation_every": 20_000,
            # The fixed learner-only control is shared with Craftax; it is not
            # rescaled using Walker's larger online training ratio.
            "algorithm2_offline_budget_basis": "craftax_4688_update_control",
            "algorithm2_reference_train_ratio": 32,
        } if walker else {}),
        "model": str(_getattr(args, "adaptation_model", "")),
        "parent_run_uid": str(
            _getattr(args, "adaptation_parent_run_uid", "")
        ),
        "parent_source_digest": parent_source_digest,
        "parent_checkpoint_generation": str(
            _getattr(args, "adaptation_parent_checkpoint", "")
        ),
        "parent_checkpoint_sha256": str(
            _getattr(args, "adaptation_parent_checkpoint_sha256", "")
        ),
        "parent_replay_directory": str(
            _getattr(args, "adaptation_parent_replay", "")
        ),
        "parent_replay_inventory_sha256": str(
            _getattr(args, "adaptation_replay_inventory_sha256", "")
        ),
        "compatibility_certificate": str(
            _getattr(args, "adaptation_compatibility_certificate", "")
        ),
        "compatibility_certificate_sha256": str(
            _getattr(
                args,
                "adaptation_compatibility_certificate_sha256",
                "",
            )
        ),
        "cohort_validation_receipt": str(
            _getattr(args, "adaptation_cohort_validation_receipt", "")
        ),
        "cohort_validation_receipt_sha256": str(
            _getattr(
                args,
                "adaptation_cohort_validation_receipt_sha256",
                "",
            )
        ),
        "parent_replay_length": replay_length,
        "pretraining_environment_steps": parent_step,
        "adaptation_environment_steps": adaptation_steps,
        "algorithm2_reference_task_steps": ALGORITHM2_REFERENCE_TASK_STEPS,
        "algorithm2_offline_updates": offline_updates,
        "algorithm2_offline_save_every": offline_save_every,
        "algorithm2_offline_environment_steps": 0,
        "algorithm2_update_order": ["reward", "imagined_policy_value"],
        "algorithm2_imported_model_frozen": True,
        "first_evaluation": "post_algorithm2_offline_at_adaptation_step_0",
        "strict_task_blind_pretraining": strict_task_blind_pretraining,
        "transfer_components": list(_ADAPTATION_COMPONENTS[backend]),
        "fresh_state": [
            "rew",
            "pol",
            "val",
            "slowval",
            "retnorm",
            "valnorm",
            "advnorm",
            "opt",
            "adapt_rew_opt",
            "adapt_behavior_opt",
            "agent_counters",
        ],
    }

    if protocol == "fresh_heads_10x":
        lineage.update(
            protocol=protocol,
            algorithm2_offline_update_multiplier=10,
            algorithm2_offline_budget_basis="craftax_4688_update_control_10x",
        )
    elif protocol == "pretrained_heads":
        lineage.update(
            protocol=protocol,
            study=("walker_multitask_pretrained_head_adaptation" if source_transfer
                   else "pretrained_task_head_adaptation"),
            transfer_components=list(transfer_components(backend, protocol)),
            source_components=source_components(backend, protocol, task=task,
                env_config=env_config, parent_algorithm=parent_algorithm, parent_task=parent_task),
            fresh_state=["opt", "agent_counters"],
            algorithm2_reference_task_steps=0,
            algorithm2_update_order=[],
            algorithm2_imported_model_frozen=False,
            first_evaluation="pretrained_task_policy_at_adaptation_step_0",
        )
        if walker:
            lineage.pop("algorithm2_offline_budget_basis", None)
            lineage.pop("algorithm2_reference_train_ratio", None)
    return lineage


def _validate_adaptation_parent(lineage):
    generation = pathlib.Path(lineage["parent_checkpoint_generation"])
    _require_real_directory(generation, "adaptation parent generation")
    done = generation / "done"
    done_info = done.lstat()
    if not stat.S_ISREG(done_info.st_mode) or stat.S_ISLNK(done_info.st_mode):
        raise RuntimeError(
            f"Unsafe adaptation parent completion marker: {done}"
        )
    if not _CURRENT_CHECKPOINT_GENERATION.fullmatch(generation.name):
        raise RuntimeError(
            f"Unsafe adaptation parent generation name: {generation.name!r}"
        )
    suffix = re.search(r"-([0-9]{12})$", generation.name)
    if suffix is not None and int(suffix.group(1)) != lineage[
        "pretraining_environment_steps"
    ]:
        raise RuntimeError(
            "Parent checkpoint generation suffix step mismatch: "
            f"{suffix.group(1)} != "
            f"{lineage['pretraining_environment_steps']}"
        )
    generation_step = _safe_pickle_integer(generation / "step.pkl")
    if generation_step != lineage["pretraining_environment_steps"]:
        raise RuntimeError(
            "Parent checkpoint generation step mismatch: "
            f"{generation_step} != {lineage['pretraining_environment_steps']}"
        )
    if not re.fullmatch(
        r"[0-9a-f]{64}", lineage["parent_checkpoint_sha256"]
    ):
        raise RuntimeError(
            "Adaptation checkpoint binding must provide a lowercase SHA-256."
        )
    actual_checkpoint_digest = checkpoint_sha256(generation)
    if actual_checkpoint_digest != lineage["parent_checkpoint_sha256"]:
        raise RuntimeError(
            "Adaptation parent checkpoint digest mismatch: "
            f"{actual_checkpoint_digest} != "
            f"{lineage['parent_checkpoint_sha256']}"
        )
    replay_pickle = generation / "replay_train.pkl"
    expected_digest = lineage["parent_replay_inventory_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        raise RuntimeError(
            "Adaptation replay binding must provide a lowercase SHA-256."
        )
    payload = _read_regular_file(
        replay_pickle, "adaptation parent replay inventory"
    )
    actual_digest = hashlib.sha256(payload).hexdigest()
    if actual_digest != expected_digest:
        raise RuntimeError(
            "Adaptation parent replay inventory digest mismatch: "
            f"{actual_digest} != {expected_digest}"
        )
    _require_real_directory(
        pathlib.Path(lineage["parent_replay_directory"]),
        "adaptation parent replay directory",
    )
    certificate_digest = lineage["compatibility_certificate_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", certificate_digest):
        raise RuntimeError(
            "Adaptation compatibility certificate must provide a lowercase "
            "SHA-256."
        )
    actual_certificate_digest = file_sha256(
        pathlib.Path(lineage["compatibility_certificate"])
    )
    if actual_certificate_digest != certificate_digest:
        raise RuntimeError(
            "Adaptation compatibility certificate digest mismatch: "
            f"{actual_certificate_digest} != {certificate_digest}"
        )
    if lineage.get("protocol", "fresh_heads") != "fresh_heads":
        entry = {
            "protocol": lineage["protocol"],
            "backend": lineage["backend"], "idm": lineage["idm"],
            "parent_seed": lineage["parent_seed"], "task": lineage["task"],
            "model": lineage["model"], "run_uid": lineage["parent_run_uid"],
            "source_digest": lineage["parent_source_digest"],
            "checkpoint_sha256": lineage["parent_checkpoint_sha256"],
            "parent_step": lineage["pretraining_environment_steps"],
            "replay_binding": {"inventory_sha256": lineage["parent_replay_inventory_sha256"]},
            **{key: lineage[key] for key in
               ("env_config", "parent_task", "parent_env_config", "parent_algorithm") if key in lineage},
        }
        _load_compatibility_certificate(
            pathlib.Path(lineage["compatibility_certificate"]),
            expected_sha256=certificate_digest, entry=entry)
    receipt_digest = lineage["cohort_validation_receipt_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", receipt_digest):
        raise RuntimeError(
            "Adaptation cohort validation receipt must provide a lowercase "
            "SHA-256."
        )
    actual_receipt_digest = file_sha256(
        pathlib.Path(lineage["cohort_validation_receipt"])
    )
    if actual_receipt_digest != receipt_digest:
        raise RuntimeError(
            "Adaptation cohort validation receipt digest mismatch: "
            f"{actual_receipt_digest} != {receipt_digest}"
        )
    return generation


def _bootstrap_adaptation(
    agent, replay, checkpoint, checkpoint_committed, args, logdir
):
    lineage = _adaptation_lineage(args)
    generation = _validate_adaptation_parent(lineage)
    if int(replay.length) != lineage["parent_replay_length"]:
        raise RuntimeError(
            "Adaptation replay sequence length does not match its certified "
            f"parent binding: {replay.length} != "
            f"{lineage['parent_replay_length']}"
        )
    schema = {
        key: space
        for key, space in agent.spaces.items()
        if key not in {"consec", "stepid"}
    }
    report_path = pathlib.Path(str(logdir)) / "transfer_report.json"
    transfer = {
        "backend": lineage["backend"],
        "components": tuple(lineage["transfer_components"]),
        "report_path": str(report_path),
        "source": lineage,
        **({"source_components": lineage["source_components"]}
           if "source_components" in lineage else {}),
    }
    replay_result = {}

    def load_replay(inventory):
        options = {}
        if "replay_reward_key" in lineage:
            options = {
                "reward_key": lineage["replay_reward_key"],
                "drop_keys": tuple(lineage["replay_drop_keys"]),
            }
        replay_result.update(
            replay.load_transfer(
                inventory,
                lineage["parent_replay_directory"],
                schema,
                amount=500_000,
                **options,
            )
        )

    print(
        "Bootstrapping strict adaptation from parent generation:",
        generation,
    )
    # Stream the validated pickle files to avoid an extra checkpoint-sized
    # allocation, then release parent state before importing replay.
    for name, loadfn in {
        "agent": bind(agent.load, strict_transfer=transfer),
        "replay_train": load_replay,
    }.items():
        with (generation / f"{name}.pkl").open("rb") as stream:
            payload = pickle.load(stream)
        try:
            loadfn(payload)
        finally:
            del payload
    if any(
        int(counter) != 0
        for counter in (agent.n_updates, agent.n_batches, agent.n_actions)
    ):
        raise RuntimeError("Adaptation bootstrap restored nonzero counters.")
    if len(replay) != min(500_000, int(replay.capacity)):
        raise RuntimeError(
            "Adaptation parent did not provide the required final replay "
            f"window: loaded {len(replay)} items."
        )
    _save_checkpoint(checkpoint, checkpoint_committed)
    print(
        "Committed transfer-only adaptation child checkpoint at step 0;",
        ("Pretrained task policy is ready for evaluation and online fine-tuning."
         if lineage.get("protocol") == "pretrained_heads"
         else "Algorithm-2 offline task training remains pending.")
    )
    return {**lineage, "replay_import": replay_result}


def _run_algorithm2_offline_adaptation(
    *,
    agent,
    replay,
    stream,
    carry,
    state,
    checkpoint,
    checkpoint_committed,
    preemption,
    step,
):
    """Fits reward then imagined task behavior before the first task action."""
    if state.complete:
        updates = int(agent.n_updates)
        invalid = (
            updates != state.target_updates
            if int(step) == 0
            else updates < state.target_updates
        )
        if invalid:
            raise RuntimeError(
                "Completed offline adaptation checkpoint has an invalid agent "
                f"update counter at child step {int(step)}: {updates}; offline "
                f"target is {state.target_updates}."
            )
        if int(step) == 0 and int(agent.n_actions) != 0:
            raise RuntimeError(
                "Post-offline step-0 checkpoint contains task-collection "
                f"actions: {int(agent.n_actions)}."
            )
        sync = getattr(agent, "sync_policy_if_needed", None)
        if callable(sync):
            sync()
        return True
    if int(step) != 0:
        raise RuntimeError(
            "Algorithm-2 offline adaptation must complete before any task "
            f"environment step; child step is {int(step)}."
        )
    if int(agent.n_actions) != 0:
        raise RuntimeError(
            "Algorithm-2 offline adaptation found prior policy execution."
        )
    if int(agent.n_updates) != state.completed_updates:
        raise RuntimeError(
            "Offline adaptation progress/counter mismatch: "
            f"{state.completed_updates} != {int(agent.n_updates)}."
        )
    train_adaptation = getattr(agent, "train_adaptation", None)
    if not callable(train_adaptation):
        raise RuntimeError(
            "Adaptation agent does not expose Algorithm-2 offline training."
        )

    print(
        "Algorithm-2 offline task adaptation:",
        f"updates {state.completed_updates}..{state.target_updates},",
        "environment step remains 0.",
    )
    begin_burst = getattr(agent, "begin_train_burst", None)
    end_burst = getattr(agent, "end_train_burst", None)
    abort_burst = getattr(agent, "abort_train_burst", None)
    burst = callable(begin_burst) and callable(end_burst)
    if burst:
        begin_burst()
    last_saved = state.completed_updates
    try:
        while state.completed_updates < state.target_updates:
            if preemption.requested:
                break
            batch = next(stream)
            carry[0], outs, _ = train_adaptation(carry[0], batch)
            if outs:
                raise RuntimeError(
                    "Algorithm-2 offline updates may not mutate replay; got "
                    f"outputs {sorted(outs)}."
                )
            state.advance()
            if (
                state.completed_updates < state.target_updates
                and state.completed_updates % state.save_every == 0
            ):
                _save_checkpoint(checkpoint, checkpoint_committed)
                last_saved = state.completed_updates
                print(
                    "Saved Algorithm-2 offline progress:",
                    f"{state.completed_updates}/{state.target_updates} updates.",
                )
                if preemption.requested:
                    break
    except BaseException:
        if burst and callable(abort_burst):
            abort_burst()
        raise
    else:
        if burst:
            end_burst()

    if state.completed_updates == state.target_updates:
        state.finish()
        sync = getattr(agent, "sync_policy_if_needed", None)
        if callable(sync):
            sync()
        _save_checkpoint(checkpoint, checkpoint_committed)
        print(
            "Completed Algorithm-2 offline task adaptation and committed the "
            f"post-training step-0 policy after {state.target_updates} updates."
        )
        return True

    if preemption.requested:
        if last_saved != state.completed_updates:
            print("Saving Algorithm-2 progress before preemption exit.")
            _save_checkpoint(checkpoint, checkpoint_committed, retries=0)
        else:
            print(
                "Algorithm-2 progress was already committed after the "
                "preemption signal."
            )
        return False

    raise RuntimeError("Offline adaptation stopped without completion or preemption.")


def _completed_checkpoint_generations(root):
    root = pathlib.Path(root)
    _require_real_directory(root, "checkpoint root")
    latest_payload = _read_regular_file(
        root / "latest", "checkpoint latest pointer"
    )
    try:
        latest = latest_payload.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise RuntimeError("Checkpoint latest pointer is not UTF-8.") from exc
    if not _SAFE_CHECKPOINT_NAME.fullmatch(latest):
        raise RuntimeError(f"Unsafe checkpoint generation name: {latest!r}")

    completed = {}
    try:
        children = list(root.iterdir())
    except OSError as exc:
        raise RuntimeError(f"Could not enumerate checkpoint root: {root}") from exc
    for child in children:
        if child.name == "latest":
            continue
        if _CHECKPOINT_LATEST_TEMP.fullmatch(child.name):
            try:
                temp_info = child.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(temp_info.st_mode):
                raise RuntimeError(
                    f"Unsafe checkpoint latest-pointer residue: {child}"
                )
            # An interrupted pointer write leaves no restorable generation and
            # protects no replay files.
            continue
        if not _SAFE_CHECKPOINT_NAME.fullmatch(child.name):
            raise RuntimeError(f"Unsafe checkpoint entry name: {child}")
        try:
            info = child.lstat()
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Checkpoint entry disappeared while enumerating: {child}"
            ) from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"Unsafe checkpoint entry: {child}")
        done = child / "done"
        try:
            done_info = done.lstat()
        except FileNotFoundError:
            # Failed saves have no completion marker and cannot be restored.
            continue
        if not stat.S_ISREG(done_info.st_mode):
            raise RuntimeError(f"Unsafe checkpoint completion marker: {done}")
        completed[child.name] = child
    if latest not in completed:
        raise RuntimeError(
            f"Checkpoint latest pointer does not name a completed generation: "
            f"{latest!r}"
        )
    return [completed[name] for name in sorted(completed)]


def _startup_preemption_should_exit(handler, logdir):
    if not handler.requested:
        return False
    checkpoint_root = pathlib.Path(str(logdir)) / "checkpoint.pkl"
    try:
        _completed_checkpoint_generations(checkpoint_root)
    except Exception as exc:
        print(
            "Preemption arrived before training began and no verified "
            "completed checkpoint is available. Exiting immediately so the "
            "storage supervisor can certify a fresh pre-checkpoint restart: "
            f"{exc}"
        )
        return True
    print(
        "Preemption arrived during startup; preserving the existing completed "
        "checkpoint without loading and rewriting it."
    )
    return True


def _close_startup_resources(*, logger=None, drivers=()):
    for resource in (*drivers, logger):
        close = getattr(resource, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except Exception as exc:
            print(f"Could not close startup resource after preemption: {exc}")


def _pinned_checkpoint_generations(checkpoint_root):
    pin_root = pathlib.Path(checkpoint_root).parent / ".checkpoint-pins"
    try:
        info = pin_root.lstat()
    except FileNotFoundError:
        return []
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"Unsafe checkpoint pin root: {pin_root}")
    try:
        pins = list(pin_root.iterdir())
    except OSError as exc:
        raise RuntimeError(f"Could not enumerate checkpoint pins: {pin_root}") from exc
    generations = []
    for pin in pins:
        if not _SAFE_CHECKPOINT_NAME.fullmatch(pin.name):
            raise RuntimeError(f"Unsafe checkpoint pin name: {pin}")
        try:
            pin_info = pin.lstat()
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Checkpoint pin disappeared while enumerating: {pin}"
            ) from exc
        if stat.S_ISLNK(pin_info.st_mode) or not stat.S_ISDIR(pin_info.st_mode):
            raise RuntimeError(f"Unsafe checkpoint pin: {pin}")
        generations.extend(_completed_checkpoint_generations(pin))
    return generations


def _checkpoint_replay_inventories(
    checkpoint_root, saveable, *, cancelled=None
):
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", saveable):
        raise ValueError(f"Unsafe replay saveable name: {saveable!r}")
    generations = _completed_checkpoint_generations(checkpoint_root)
    generations.extend(_pinned_checkpoint_generations(checkpoint_root))
    inventories = []
    for generation in generations:
        if cancelled is not None and cancelled():
            return None
        inventories.append(
            _read_replay_inventory(generation / f"{saveable}.pkl")
        )
    return inventories


def _checkpoint_replays_committed(
    checkpoint_root, replays, *, collect=True, cancelled=None
):
    ready = set()
    for replay in replays.values():
        committed = getattr(replay, "checkpoint_committed", None)
        if not callable(committed):
            continue
        try:
            committed()
        except Exception as exc:
            print(f"Could not record committed replay checkpoint: {exc}")
        else:
            ready.add(id(replay))

    if not collect:
        return
    if cancelled is not None and cancelled():
        return

    for saveable, replay in replays.items():
        collect_replay = getattr(replay, "checkpoint_gc", None)
        if not callable(collect_replay) or id(replay) not in ready:
            continue
        try:
            inventories = _checkpoint_replay_inventories(
                checkpoint_root, saveable, cancelled=cancelled
            )
            if inventories is None:
                return
            result = collect_replay(inventories, cancelled=cancelled)
        except Exception as exc:
            # Garbage collection errors must not invalidate a committed checkpoint.
            print(
                f"Skipping replay GC for {saveable}; could not prove the "
                f"complete protected inventory set: {exc}"
            )
            continue
        if isinstance(result, Mapping) and result.get("errors"):
            print(
                f"Replay GC for {saveable} left cleanup debt: "
                f"{result['errors']} error(s)."
            )


def _require_regular_jsonl(path, description):
    path = pathlib.Path(path)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing {description}: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(
            f"Unsafe {description} at {path}: expected a regular file, not a symlink."
        )
    return info


def _atomic_rewrite_jsonl(path, chunks, mode):
    path = pathlib.Path(path)
    _require_real_directory(path.parent, "JSONL parent directory")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.rewrite-", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.writelines(chunks)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)


def _open_jsonl_append(path):
    path = pathlib.Path(path)
    _require_real_directory(path.parent, "JSONL parent directory")
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o660)
    except OSError as exc:
        raise RuntimeError(f"Could not safely open JSONL output {path}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"Unsafe JSONL output at {path}: expected a regular file.")
        return os.fdopen(descriptor, "a", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise


def _truncate_logger_jsonl_after(logdir, maximum):
    """Atomically removes scalar log rows newer than a loaded checkpoint.

    JSONL outputs are not part of the checkpoint. After an ungraceful restart,
    they can therefore contain rows beyond the restored counter. Only complete,
    valid mapping records with an integer ``step`` at or below the checkpoint
    are retained without changing numeric values or key ordering.
    """
    logdir = pathlib.Path(str(logdir)).expanduser()
    _require_real_directory(logdir, "run log directory")
    removed = {}
    for filename in ("metrics.jsonl", "scores.jsonl"):
        path = logdir / filename
        if not os.path.lexists(path):
            continue
        info = _require_regular_jsonl(path, f"scalar log {filename}")

        retained = []
        dropped = 0
        with path.open("rb") as handle:
            for line in handle:
                # JSONLOutput commits one newline-terminated record at a time.
                # A non-terminated final line may be a torn write, even if the
                # bytes happen to form parseable JSON.
                if not line.endswith(b"\n"):
                    dropped += 1
                    continue
                try:
                    record = json.loads(line.decode("utf-8"))
                    value = record["step"]
                    if not isinstance(record, Mapping):
                        raise TypeError("JSONL record is not a mapping")
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise TypeError("JSONL step is not numeric")
                    if isinstance(value, float):
                        if not np.isfinite(value) or not value.is_integer():
                            raise ValueError("JSONL step is not an integer")
                        step = int(value)
                    else:
                        step = value
                except (KeyError, TypeError, ValueError, UnicodeDecodeError):
                    dropped += 1
                    continue
                if step > int(maximum):
                    dropped += 1
                    continue
                retained.append(line)

        if not dropped:
            continue

        _atomic_rewrite_jsonl(path, retained, stat.S_IMODE(info.st_mode))
        removed[filename] = dropped
    return removed


def _logger_checkpoint_step(logger, step):
    """Converts the checkpoint counter to the logger's displayed step scale."""
    return int(step) * int(getattr(logger, "multiplier", 1))


class _EpisodeJSONLWriter:
    """Append-only episode records with restart-safe identity de-duplication."""

    def __init__(self, path, key_fields):
        self.path = pathlib.Path(str(path)).expanduser()
        self.key_fields = tuple(key_fields)
        self.seen = set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _require_real_directory(self.path.parent, "episode JSONL parent directory")
        if os.path.lexists(self.path):
            info = _require_regular_jsonl(self.path, "episode JSONL output")
            retained = []
            records = []
            repair = False
            with self.path.open("rb") as handle:
                for line in handle:
                    if not line.endswith(b"\n"):
                        repair = True
                        continue
                    try:
                        record = json.loads(line.decode("utf-8"))
                    except (ValueError, TypeError, UnicodeDecodeError):
                        repair = True
                        continue
                    if not isinstance(record, Mapping):
                        repair = True
                        continue
                    retained.append(line)
                    records.append(record)
            if repair:
                _atomic_rewrite_jsonl(
                    self.path, retained, stat.S_IMODE(info.st_mode)
                )
            self.seen = {self._key(record) for record in records}

    def write(self, records):
        pending = []
        for record in records:
            record = _json_value(record)
            key = self._key(record)
            if key in self.seen:
                continue
            self.seen.add(key)
            pending.append(record)
        if pending:
            with _open_jsonl_append(self.path) as handle:
                for record in pending:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
        return len(pending)

    def truncate_after(self, field, maximum):
        """Atomically drops records newer than the loaded checkpoint."""
        if not os.path.lexists(self.path):
            return 0
        info = _require_regular_jsonl(self.path, "episode JSONL output")
        retained = []
        removed = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except (ValueError, TypeError):
                    removed += 1
                    continue
                if not isinstance(record, Mapping):
                    removed += 1
                    continue
                value = record.get(field)
                try:
                    is_newer = value is not None and int(value) > int(maximum)
                except (TypeError, ValueError, OverflowError):
                    removed += 1
                    continue
                if is_newer:
                    removed += 1
                else:
                    retained.append(record)
        if removed:
            chunks = [
                (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
                for record in retained
            ]
            _atomic_rewrite_jsonl(
                self.path, chunks, stat.S_IMODE(info.st_mode)
            )
        self.seen = {self._key(record) for record in retained}
        return removed

    def next_indices(self, group_field, index_field):
        result = collections.defaultdict(int)
        if not os.path.lexists(self.path):
            return result
        _require_regular_jsonl(self.path, "episode JSONL output")
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                    if not isinstance(record, Mapping):
                        continue
                    group = int(record[group_field])
                    index = int(record[index_field])
                except (KeyError, ValueError, TypeError):
                    continue
                result[group] = max(result[group], index + 1)
        return result

    def count_matching(self, **expected):
        if not os.path.lexists(self.path):
            return 0
        _require_regular_jsonl(self.path, "episode JSONL output")
        count = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if not isinstance(record, Mapping):
                    continue
                if all(record.get(key) == value for key, value in expected.items()):
                    count += 1
        return count

    def count_matching_between(self, field, minimum, maximum, **expected):
        """Counts matching records whose integer field lies in a closed range."""
        if not os.path.lexists(self.path):
            return 0
        _require_regular_jsonl(self.path, "episode JSONL output")
        minimum = int(minimum)
        maximum = int(maximum)
        if maximum < minimum:
            raise ValueError(
                f"Invalid record range for {field!r}: {minimum}..{maximum}."
            )
        count = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                    value = int(record[field])
                except (KeyError, ValueError, TypeError, OverflowError):
                    continue
                if not isinstance(record, Mapping):
                    continue
                if not minimum <= value <= maximum:
                    continue
                if all(record.get(key) == value for key, value in expected.items()):
                    count += 1
        return count

    def _key(self, record):
        if not isinstance(record, Mapping):
            raise TypeError(f"Episode record must be a mapping, got {type(record)!r}.")
        return tuple(record.get(field) for field in self.key_fields)


class _EpisodeLogAccumulator:
    """Low-overhead scalar episode reductions for environment diagnostics."""

    def __init__(self):
        self.values = {}

    def reset(self):
        self.values.clear()

    def add(self, key, value):
        value = np.asarray(value)
        assert value.ndim == 0, (key, value.shape, value.dtype)
        number = float(value)
        if key in self.values:
            total, maximum, count, _ = self.values[key]
            self.values[key] = (
                total + number, max(maximum, number), count + 1, number
            )
        else:
            self.values[key] = (number, number, 1, number)

    def result(self, reset=True):
        result = {}
        for key, (total, maximum, count, last) in self.values.items():
            result[f"{key}/avg"] = total / count
            result[f"{key}/max"] = maximum
            result[f"{key}/sum"] = total
            result[f"{key}/last"] = last
        if reset:
            self.reset()
        return result


def _json_value(value):
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class BatchedDriver:
    def __init__(self, make_env, **kwargs):
        self.env = make_env()
        self.kwargs = kwargs
        self.length = len(self.env)
        self.act_space = self.env.act_space
        self.callbacks = []
        self.acts = None
        self.carry = None
        self.reset()

    def reset(self, init_policy=None):
        self.acts = {
            key: np.zeros((self.length,) + space.shape, space.dtype)
            for key, space in self.act_space.items()
        }
        self.acts["reset"] = np.ones(self.length, bool)
        self.carry = init_policy and init_policy(self.length)

    def close(self):
        self.env.close()

    def on_step(self, callback):
        self.callbacks.append(callback)

    def __call__(self, policy, steps=0, episodes=0):
        step, episode = 0, 0
        while step < steps or episode < episodes:
            step, episode = self._step(policy, step, episode)

    def _step(self, policy, step, episode):
        obs = self.env.step(self.acts)
        logs = {key: value for key, value in obs.items() if key.startswith("log/")}
        labels = {
            key: value for key, value in obs.items() if key.startswith("label/")
        }
        obs = {
            key: value
            for key, value in obs.items()
            if not key.startswith(("log/", "label/"))
        }
        assert all(len(value) == self.length for value in obs.values()), obs
        self.carry, acts, outs = policy(self.carry, obs, **self.kwargs)
        assert all(key not in acts for key in outs), (list(outs.keys()), list(acts.keys()))
        if obs["is_last"].any():
            mask = ~obs["is_last"]
            acts = {key: self._mask(value, mask) for key, value in acts.items()}
        self.acts = {**acts, "reset": obs["is_last"].copy()}
        trans = {**obs, **acts, **outs, **labels, **logs}
        for index in range(self.length):
            trn = elements.tree.map(lambda value: value[index], trans)
            for callback in self.callbacks:
                callback(trn, index, **self.kwargs)
        step += len(obs["is_first"])
        episode += int(obs["is_last"].sum())
        return step, episode

    def _mask(self, value, mask):
        while mask.ndim < value.ndim:
            mask = mask[..., None]
        return value * mask.astype(value.dtype)


class DeviceChunkedBatchedDriver:
    def __init__(
        self,
        make_env,
        agent,
        *,
        chunk=32,
        mode="train",
        policy_seed=None,
        **kwargs,
    ):
        import jax
        import jax.numpy as jnp

        self.env = make_env()
        self.agent = agent
        self.chunk = max(1, int(chunk))
        self.mode = mode
        self.policy_seed = None if policy_seed is None else int(policy_seed)
        self.kwargs = kwargs
        self.length = len(self.env)
        self.act_space = self.env.act_space
        self.callbacks = []
        self.chunk_callbacks = []
        self._jax = jax
        self._jnp = jnp
        self._rollout = self._make_rollout()
        self.reset()

    def reset(self, init_policy=None, reset_seed_bank=None):
        del init_policy
        self.agent.sync_policy_if_needed()
        self.carry = self.agent.init_policy_device(self.length)
        fixed_reset_keys = getattr(self.env, "fixed_reset_keys_device", None)
        fixed_reset_keys = fixed_reset_keys() if fixed_reset_keys else None
        if reset_seed_bank is not None:
            reset_seed_bank = tuple(int(seed) for seed in reset_seed_bank)
            if len(reset_seed_bank) != self.length:
                raise ValueError(
                    "reset_seed_bank must contain one seed per environment; "
                    f"got {len(reset_seed_bank)} for {self.length} environments."
                )
            self.rng = self.env.rng_device()
            with self.env.device_put_scope():
                keys = self._jnp.stack([
                    self._jax.random.PRNGKey(seed) for seed in reset_seed_bank
                ])
        elif fixed_reset_keys is not None:
            # Evaluation uses common random numbers: every checkpoint starts
            # from the exact same held-out reset-key bank.
            self.rng = self.env.rng_device()
            keys = fixed_reset_keys
        else:
            # Advance the reset stream when no fixed evaluation bank is configured.
            if not hasattr(self, "rng"):
                self.rng = self.env.rng_device()
            self.rng, keys = self.env.split_keys_device(self.rng)
        self.state = self.env.reset_device(keys)
        self._policy_seed_offset = 0
        with self.env.device_put_scope():
            jnp = self._jnp
            self.reward = jnp.zeros((self.length,), dtype=jnp.float32)
            self.is_first = jnp.ones((self.length,), dtype=bool)
            self.is_last = jnp.zeros((self.length,), dtype=bool)
            self.is_terminal = jnp.zeros((self.length,), dtype=bool)
            self.step_count = jnp.zeros((self.length,), dtype=jnp.int32)

    def close(self):
        self.env.close()

    def on_step(self, callback):
        self.callbacks.append(callback)

    def on_chunk(self, callback):
        self.chunk_callbacks.append(callback)

    def __call__(self, policy=None, steps=0, episodes=0):
        del policy
        step, episode = 0, 0
        while step < steps or episode < episodes:
            step, episode = self._chunk(step, episode, episodes)

    def _make_rollout(self):
        jax = self._jax
        jnp = self._jnp
        env = self.env
        agent = self.agent
        chunk = self.chunk
        mode = self.mode
        finite_state = getattr(env, "finite_state_device", None)
        reset_from_rng = getattr(env, "reset_from_rng_device", None)
        diagnostics_from_state = getattr(
            env, "diagnostics_from_state_device", None)
        terminal_on_time_limit = bool(
            getattr(env, "terminal_on_time_limit", False)
        )

        def mask_done(value, is_last):
            mask = is_last
            while mask.ndim < value.ndim:
                mask = mask[..., None]
            return jnp.where(mask, jnp.zeros_like(value), value)

        def finite_reward(value):
            return jnp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

        @jax.jit
        def rollout(
            policy_params,
            carry,
            state,
            reward,
            is_first,
            is_last,
            is_terminal,
            step_count,
            rng,
            seed,
        ):
            def body(inputs, index):
                carry, state, reward, is_first, is_last, is_terminal, step_count, rng = inputs
                obs = env.obs_from_state_device(
                    state, reward, is_first, is_last, is_terminal)
                step_seed = jax.random.fold_in(seed, index)
                carry, acts, outs = agent.policy_device(
                    carry, obs, step_seed, mode=mode, params=policy_params)
                outs = {key: value for key, value in outs.items() if key != "finite"}
                acts = {key: mask_done(value, is_last) for key, value in acts.items()}

                stepped_state = env.step_device(state, acts["action"])
                if reset_from_rng is None:
                    reset_state = env.first_state_device(state)
                    next_rng = rng
                else:
                    def fresh_reset(current_rng):
                        return reset_from_rng(current_rng)

                    def keep_state(current_rng):
                        return current_rng, state

                    next_rng, reset_state = jax.lax.cond(
                        jnp.any(is_last),
                        fresh_reset,
                        keep_state,
                        rng,
                    )
                next_state = env.select_reset_device(is_last, reset_state, stepped_state)
                env_done = stepped_state.done.astype(bool)
                if finite_state is None:
                    invalid_state = jnp.zeros_like(env_done, dtype=bool)
                else:
                    invalid_state = ~finite_state(stepped_state)
                stepped_count = step_count + env.action_repeat
                if env.time_limit > 0:
                    time_done = stepped_count >= env.time_limit
                else:
                    time_done = jnp.zeros_like(env_done, dtype=bool)
                active_done = env_done | time_done | invalid_state
                next_reward = jnp.where(
                    is_last, jnp.zeros_like(stepped_state.reward), stepped_state.reward)
                next_reward = finite_reward(next_reward)
                next_is_first = is_last
                next_is_last = jnp.where(
                    is_last, jnp.zeros_like(active_done, dtype=bool), active_done)
                next_is_terminal = jnp.where(
                    is_last,
                    jnp.zeros_like(env_done, dtype=bool),
                    (
                        env_done
                        if terminal_on_time_limit
                        else env_done & ~time_done
                    )
                    | invalid_state,
                )
                next_step_count = jnp.where(
                    is_last,
                    jnp.zeros_like(stepped_count),
                    stepped_count,
                )
                diagnostics = (
                    diagnostics_from_state(state, is_last, is_terminal)
                    if diagnostics_from_state is not None else {})
                trans = {**obs, **acts, **outs, **diagnostics}
                return (
                    carry,
                    next_state,
                    next_reward,
                    next_is_first,
                    next_is_last,
                    next_is_terminal,
                    next_step_count,
                    next_rng,
                ), trans

            indices = jnp.arange(chunk, dtype=jnp.uint32)
            inputs = (
                carry,
                state,
                reward,
                is_first,
                is_last,
                is_terminal,
                step_count,
                rng,
            )
            outputs, data = jax.lax.scan(body, inputs, indices)
            return outputs, data

        return rollout

    def _chunk(self, step, episode, target_episodes):
        self.agent.sync_policy_if_needed()
        if self.policy_seed is None:
            seed = self.agent.reserve_policy_seed_device(self.chunk)
        else:
            seed = self.agent.policy_seed_device(
                self.policy_seed + self._policy_seed_offset)
            self._policy_seed_offset += self.chunk
        with elements.timer.section("device_actor_rollout"):
            outputs, data = self._rollout(
                self.agent.policy_params,
                self.carry,
                self.state,
                self.reward,
                self.is_first,
                self.is_last,
                self.is_terminal,
                self.step_count,
                self.rng,
                seed,
            )
        (
            self.carry,
            self.state,
            self.reward,
            self.is_first,
            self.is_last,
            self.is_terminal,
            self.step_count,
            self.rng,
        ) = outputs
        with elements.timer.section("device_actor_fetch"):
            with self.env.device_get_scope():
                trans = self.agent._take_outs(data)
        if not target_episodes:
            for callback in self.chunk_callbacks:
                callback(trans, **self.kwargs)
        with elements.timer.section("device_actor_drain"):
            return self._drain(trans, step, episode, target_episodes)

    def _drain(self, trans, step, episode, target_episodes):
        if not self.callbacks and not target_episodes:
            return (
                step + _chunk_amount(trans),
                episode + int(np.asarray(trans["is_last"]).sum()),
            )
        limit_episodes = bool(target_episodes)
        for time_index in range(next(iter(trans.values())).shape[0]):
            for env_index in range(self.length):
                trn = elements.tree.map(lambda value: value[time_index, env_index], trans)
                for callback in self.callbacks:
                    callback(trn, env_index, **self.kwargs)
                step += 1
                episode += int(bool(trn["is_last"]))
                if limit_episodes and episode >= target_episodes:
                    return step, episode
        return step, episode


@contextlib.contextmanager
def _eval_driver_scope(persistent_drivers, eval_key, make_driver):
    """Reuses device eval drivers and closes freshly constructed host drivers."""
    driver = persistent_drivers.get(eval_key)
    temporary = driver is None
    if temporary:
        driver = make_driver()
    try:
        yield driver
    finally:
        if temporary:
            driver.close()


def train(make_agent, make_replay, make_env, make_stream, make_logger, args):
    preemption = _PreemptionHandler().install()
    try:
        return _train(
            make_agent,
            make_replay,
            make_env,
            make_stream,
            make_logger,
            args,
            preemption,
        )
    finally:
        preemption.restore()


def _train(
    make_agent,
    make_replay,
    make_env,
    make_stream,
    make_logger,
    args,
    preemption,
):
    logdir = elements.Path(args.logdir)
    if _startup_preemption_should_exit(preemption, logdir):
        return
    agent = make_agent()
    if _startup_preemption_should_exit(preemption, logdir):
        return
    replay = make_replay()
    if _startup_preemption_should_exit(preemption, logdir):
        return
    logger = make_logger()
    if _startup_preemption_should_exit(preemption, logdir):
        _close_startup_resources(logger=logger)
        return
    step = logger.step
    usage = elements.Usage(**args.usage)
    train_agg = elements.Agg()
    epstats = elements.Agg()
    episodes = collections.defaultdict(elements.Agg)
    episode_logs = collections.defaultdict(_EpisodeLogAccumulator)
    episode_video_steps = collections.defaultdict(int)
    video_policy_stride = max(
        1, int(_getattr(args, "video_policy_stride", 1))
    )
    policy_fps = elements.FPS()
    train_fps = elements.FPS()

    batch_steps = args.batch_size * args.batch_length
    should_train = elements.when.Ratio(args.train_ratio / batch_steps)
    should_log = elements.when.Every(_getattr(args, "log_every_steps", args.log_every))
    report_every_steps = int(
        _getattr(args, "report_every_steps", args.report_every)
    )
    save_every_steps = int(_getattr(args, "save_every_steps", args.save_every))
    should_report = _AbsoluteStepSchedule(report_every_steps)
    should_save = _AbsoluteStepSchedule(save_every_steps)

    @elements.timer.section("logfn")
    def logfn(tran, worker):
        episode = episodes[worker]
        scalar_logs = episode_logs[worker]
        if tran["is_first"]:
            episode.reset()
            scalar_logs.reset()
            episode_video_steps[worker] = 0
        episode.add("score", tran["reward"], agg="sum")
        episode.add("length", 0 if tran["is_first"] else 1, agg="sum")
        if not tran["is_first"]:
            episode.add("rewards", tran["reward"], agg="stack")
        _add_episode_logs(
            episode,
            scalar_logs,
            tran,
            worker,
            video_policy_log=_getattr(args, "video_policy_log", True),
            video_policy_stride=video_policy_stride,
            video_policy_step=episode_video_steps[worker],
        )
        episode_video_steps[worker] += 1
        if tran["is_last"]:
            result = episode.result()
            result.update(scalar_logs.result())
            logger.add(
                {"score": result.pop("score"), "length": result.pop("length")},
                prefix="episode",
            )
            _finish_episode_stats(result, epstats)

    fns = [bind(make_env, i) for i in range(args.envs)]
    driver = embodied.Driver(fns, parallel=not args.debug)
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(lambda tran, _: policy_fps.step())
    driver.on_step(replay.add)
    driver.on_step(logfn)

    stream_train = iter(agent.stream(make_stream(replay, "train")))
    stream_report = iter(agent.stream(make_stream(replay, "report")))
    carry_train = [agent.init_train(args.batch_size)]
    carry_report = agent.init_report(args.batch_size)

    def trainfn(tran, worker):
        del tran, worker
        if len(replay) < args.batch_size * args.batch_length:
            return
        for _ in range(should_train(step)):
            with elements.timer.section("stream_next"):
                batch = next(stream_train)
            carry_train[0], outs, mets = agent.train(carry_train[0], batch)
            train_fps.step(batch_steps)
            if "replay" in outs:
                replay.update(outs["replay"])
            train_agg.add(mets, prefix="train")

    driver.on_step(trainfn)

    # Retain generations while the recovery monitor copies checkpoints.
    if _startup_preemption_should_exit(preemption, logdir):
        _close_startup_resources(
            logger=logger, drivers=(driver,)
        )
        return
    cp = _DurableCheckpoint(logdir / "checkpoint.pkl", keep=2)
    cp.step = step
    cp.agent = agent
    cp.replay = replay
    checkpoint_root = pathlib.Path(str(logdir)) / "checkpoint.pkl"

    def checkpoint_committed():
        _checkpoint_replays_committed(
            checkpoint_root,
            {"replay": replay},
            collect=not preemption.requested,
            cancelled=lambda: preemption.requested,
        )
    if args.from_checkpoint:
        elements.checkpoint.load(
            args.from_checkpoint,
            dict(agent=bind(agent.load, regex=args.from_checkpoint_regex)),
        )
    cp.load_or_save()
    checkpoint_committed()
    if preemption.requested:
        print(
            "Checkpoint startup transaction completed after preemption; "
            "exiting without a redundant save."
        )
        _close_startup_resources(
            logger=logger, drivers=(driver,)
        )
        return
    last_checkpoint_step = [int(step)]
    removed_logger_records = _truncate_logger_jsonl_after(
        logdir, _logger_checkpoint_step(logger, step)
    )
    if removed_logger_records:
        print(
            "Restored checkpoint rollback: removed scalar log records:",
            ", ".join(
                f"{filename}={count}"
                for filename, count in sorted(removed_logger_records.items())
            ),
        )

    try:
        print("Start training loop")
        policy = lambda *items: agent.policy(*items, mode="train")
        driver.reset(agent.init_policy)
        while step < args.steps:
            if _save_preempt_checkpoint_if_needed(
                preemption, cp, checkpoint_committed
            ):
                break
            driver(policy, steps=10)
            if _save_preempt_checkpoint_if_needed(
                preemption, cp, checkpoint_committed
            ):
                break

            if should_report(step) and len(replay):
                agg = elements.Agg()
                for _ in range(args.consec_report * args.report_batches):
                    carry_report, mets = agent.report(carry_report, next(stream_report))
                    mets = _filter_report_metrics(
                        mets, _getattr(args, "video_pred_log", True)
                    )
                    agg.add(mets)
                logger.add(agg.result(), prefix="report")

            if should_log(step):
                logger.add(train_agg.result())
                logger.add(epstats.result(), prefix="epstats")
                logger.add(replay.stats(), prefix="replay")
                logger.add(usage.stats(), prefix="usage")
                logger.add(_fps_metrics(
                    policy_fps, train_fps, args.train_ratio))
                logger.add(_numeric_timer_stats(), prefix="timer")
                logger.write()

            if should_save(step):
                _save_checkpoint(cp, checkpoint_committed)
                last_checkpoint_step[0] = int(step)
                if preemption.requested:
                    print(
                        "Checkpoint committed after preemption signal; "
                        "skipping a duplicate save."
                    )
                    break
        completed_naturally = (
            not preemption.requested and int(step) >= int(args.steps)
        )
        if (
            completed_naturally
            and last_checkpoint_step[0] != int(step)
        ):
            _save_final_checkpoint_if_complete(
                preemption,
                cp,
                step,
                args.steps,
                checkpoint_committed,
            )
    finally:
        logger.close()


def train_eval_multi(
    make_agent,
    make_replay_train,
    make_replay_eval,
    make_env_train,
    make_env_eval,
    make_stream,
    make_logger,
    args,
    eval_tasks,
    make_env_train_batch=None,
    make_env_eval_batch=None,
):
    preemption = _PreemptionHandler().install()
    try:
        return _train_eval_multi(
            make_agent,
            make_replay_train,
            make_replay_eval,
            make_env_train,
            make_env_eval,
            make_stream,
            make_logger,
            args,
            eval_tasks,
            make_env_train_batch,
            make_env_eval_batch,
            preemption,
        )
    finally:
        preemption.restore()


def _train_eval_multi(
    make_agent,
    make_replay_train,
    make_replay_eval,
    make_env_train,
    make_env_eval,
    make_stream,
    make_logger,
    args,
    eval_tasks,
    make_env_train_batch,
    make_env_eval_batch,
    preemption,
):
    logdir = elements.Path(args.logdir)
    if _startup_preemption_should_exit(preemption, logdir):
        return
    agent = make_agent()
    if _startup_preemption_should_exit(preemption, logdir):
        return
    replay_train = make_replay_train()
    if _startup_preemption_should_exit(preemption, logdir):
        return
    logger = make_logger()
    if _startup_preemption_should_exit(preemption, logdir):
        _close_startup_resources(logger=logger)
        return
    eval_runs = _evaluation_runs(
        eval_tasks,
        _getattr(args, "eval_action_selections", None),
        _getattr(args, "eval_policy_roles", None),
    )

    print("Logdir", logdir)
    step = logger.step
    usage = elements.Usage(**args.usage)
    agg = elements.Agg()
    train_episodes = collections.defaultdict(elements.Agg)
    train_episode_logs = collections.defaultdict(_EpisodeLogAccumulator)
    train_video_steps = collections.defaultdict(int)
    train_epstats = elements.Agg()
    train_completed = []
    train_worker_episode_counts = collections.defaultdict(int)
    eval_episodes = {
        eval_key: collections.defaultdict(elements.Agg)
        for eval_key, _, _, _, _ in eval_runs
    }
    eval_episode_logs = {
        eval_key: collections.defaultdict(_EpisodeLogAccumulator)
        for eval_key, _, _, _, _ in eval_runs
    }
    eval_video_steps = {
        eval_key: collections.defaultdict(int)
        for eval_key, _, _, _, _ in eval_runs
    }
    eval_epstats = {
        eval_key: elements.Agg() for eval_key, _, _, _, _ in eval_runs
    }
    eval_completed = {eval_key: [] for eval_key, _, _, _, _ in eval_runs}
    eval_worker_episode_counts = {
        eval_key: collections.defaultdict(int)
        for eval_key, _, _, _, _ in eval_runs
    }
    eval_worker_targets = {
        eval_key: _worker_episode_targets(args.eval_eps, args.eval_envs)
        for eval_key, _, _, _, _ in eval_runs
    }
    eval_worker_offsets = {
        eval_key: 0 for eval_key, _, _, _, _ in eval_runs
    }
    active_eval_replay = {
        eval_key: None for eval_key, _, _, _, _ in eval_runs
    }
    train_record_writer = _EpisodeJSONLWriter(
        pathlib.Path(str(logdir)) / "train_episodes.jsonl",
        ("step", "env_index", "episode_index"),
    )
    eval_record_writer = _EpisodeJSONLWriter(
        pathlib.Path(str(logdir)) / "eval_episodes.jsonl",
        (
            "checkpoint_step",
            "task",
            "eval_label",
            "action_selection",
            "policy_role",
            "env_seed",
            "env_index",
            "worker_episode",
        ),
    )
    policy_fps = elements.FPS()
    eval_policy_fps = elements.FPS()
    train_fps = elements.FPS()
    video_policy_stride = max(
        1, int(_getattr(args, "video_policy_stride", 1))
    )

    batch_steps = args.batch_size * args.batch_length
    should_train = elements.when.Ratio(args.train_ratio / batch_steps)
    should_log = elements.when.Every(_getattr(args, "log_every_steps", args.log_every))
    report_every_steps = int(
        _getattr(args, "report_every_steps", args.report_every)
    )
    save_every_steps = int(_getattr(args, "save_every_steps", args.save_every))
    should_report = _AbsoluteStepSchedule(report_every_steps)
    should_save = _AbsoluteStepSchedule(save_every_steps)

    @elements.timer.section("logfn")
    def logfn(tran, worker, mode, eval_key=None):
        if mode == "train":
            episode = train_episodes[worker]
            scalar_logs = train_episode_logs[worker]
            video_steps = train_video_steps
            epstats = train_epstats
        else:
            episode = eval_episodes[eval_key][worker]
            scalar_logs = eval_episode_logs[eval_key][worker]
            video_steps = eval_video_steps[eval_key]
            epstats = eval_epstats[eval_key]
        if tran["is_first"]:
            episode.reset()
            scalar_logs.reset()
            video_steps[worker] = 0
        episode.add("score", tran["reward"], agg="sum")
        episode.add("length", 0 if tran["is_first"] else 1, agg="sum")
        if not tran["is_first"]:
            episode.add("rewards", tran["reward"], agg="stack")
        episode.add("terminal", tran["is_terminal"], agg="max")
        _add_episode_logs(
            episode,
            scalar_logs,
            tran,
            worker,
            video_policy_log=_getattr(args, "video_policy_log", True),
            video_policy_stride=video_policy_stride,
            video_policy_step=video_steps[worker],
        )
        video_steps[worker] += 1
        if tran["is_last"]:
            result = episode.result()
            result.update(scalar_logs.result())
            score = result.pop("score")
            length = result.pop("length")
            if mode == "train":
                diagnostics = _finish_episode_stats(result, epstats)
                episode_index = train_worker_episode_counts[worker]
                train_worker_episode_counts[worker] += 1
                completed = {
                    "score": float(score),
                    "length": int(length),
                    "env_index": int(worker),
                    "episode_index": int(episode_index),
                    "diagnostics": diagnostics,
                }
                train_completed.append(completed)
                train_record_writer.write([{
                    "version": 1,
                    "step": int(step),
                    "task": str(_getattr(args, "train_task", "")),
                    "env_index": int(worker),
                    "episode_index": int(episode_index),
                    "return": float(score),
                    "length": int(length),
                    "diagnostics": diagnostics,
                }])
            else:
                worker_episode = eval_worker_episode_counts[eval_key][worker]
                if worker_episode >= eval_worker_targets[eval_key][worker]:
                    return
                diagnostics = _finish_episode_stats(result, epstats)
                eval_worker_episode_counts[eval_key][worker] += 1
                eval_completed[eval_key].append({
                    "score": float(score),
                    "length": int(length),
                    "env_index": int(worker),
                    "worker_episode": int(worker_episode),
                    "diagnostics": diagnostics,
                })

    @elements.timer.section("chunk_logfn")
    def chunk_logfn(batch, mode, label=None):
        assert mode == "train", mode
        length, envs = next(iter(batch.values())).shape[:2]
        records = []
        for worker in range(envs):
            episode = train_episodes[worker]
            scalar_logs = train_episode_logs[worker]
            for time_index in range(length):
                if bool(batch["is_first"][time_index, worker]):
                    episode.reset()
                    scalar_logs.reset()
                    train_video_steps[worker] = 0
                reward = batch["reward"][time_index, worker]
                episode.add("score", reward, agg="sum")
                episode.add(
                    "length",
                    0 if bool(batch["is_first"][time_index, worker]) else 1,
                    agg="sum",
                )
                if not bool(batch["is_first"][time_index, worker]):
                    episode.add("rewards", reward, agg="stack")
                episode.add(
                    "terminal",
                    batch["is_terminal"][time_index, worker],
                    agg="max",
                )
                _add_episode_logs_from_batch(
                    episode,
                    scalar_logs,
                    batch,
                    time_index,
                    worker,
                    video_policy_log=_getattr(args, "video_policy_log", True),
                    video_policy_stride=video_policy_stride,
                    video_policy_step=train_video_steps[worker],
                )
                train_video_steps[worker] += 1
                if bool(batch["is_last"][time_index, worker]):
                    result = episode.result()
                    result.update(scalar_logs.result())
                    score = result.pop("score")
                    length_ = result.pop("length")
                    diagnostics = _finish_episode_stats(result, train_epstats)
                    episode_index = train_worker_episode_counts[worker]
                    train_worker_episode_counts[worker] += 1
                    completed = {
                        "score": float(score),
                        "length": int(length_),
                        "env_index": int(worker),
                        "episode_index": int(episode_index),
                        "diagnostics": diagnostics,
                    }
                    train_completed.append(completed)
                    records.append({
                        "version": 1,
                        "step": int(step),
                        "task": str(_getattr(args, "train_task", "")),
                        "env_index": int(worker),
                        "episode_index": int(episode_index),
                        "return": float(score),
                        "length": int(length_),
                        "diagnostics": diagnostics,
                    })
        train_record_writer.write(records)

    batched_train = _getattr(args, "batched_env", False)
    host_batched_train = (
        batched_train and _getattr(args, "host_batched_env", False)
    )
    chunked_train = batched_train and not host_batched_train
    if chunked_train:
        assert make_env_train_batch is not None
        driver_train = DeviceChunkedBatchedDriver(
            bind(make_env_train_batch, args.envs),
            agent,
            chunk=_getattr(
                args,
                "device_actor_chunk",
                _getattr(args, "mjx_actor_chunk", 32),
            ),
            mode="train",
        )
    elif host_batched_train:
        assert make_env_train_batch is not None
        driver_train = BatchedDriver(bind(make_env_train_batch, args.envs))
    else:
        fns = [bind(make_env_train, i) for i in range(args.envs)]
        driver_train = embodied.Driver(fns, parallel=(not args.debug))
    if chunked_train:
        driver_train.on_chunk(lambda batch: step.increment(_chunk_amount(batch)))
        driver_train.on_chunk(lambda batch: policy_fps.step(_chunk_amount(batch)))
        driver_train.on_chunk(replay_train.add_batch)
        driver_train.on_chunk(bind(chunk_logfn, mode="train"))
    else:
        driver_train.on_step(lambda tran, _: step.increment())
        driver_train.on_step(lambda tran, _: policy_fps.step())
        driver_train.on_step(replay_train.add)
        driver_train.on_step(bind(logfn, mode="train"))

    run_eval = args.eval_envs > 0 and args.eval_eps > 0
    persistent_eval_drivers = {}

    def add_eval_replay(tran, worker, eval_key):
        replay = active_eval_replay[eval_key]
        within_target = (
            eval_worker_episode_counts[eval_key][worker]
            < eval_worker_targets[eval_key][worker]
        )
        if replay is not None and within_target:
            replay.add(tran, worker)

    def build_eval_driver(
        eval_key,
        label,
        task,
        action_selection,
        policy_role,
        *,
        env_count=None,
        worker_start=0,
    ):
        env_count = int(args.eval_envs if env_count is None else env_count)
        if chunked_train:
            assert make_env_eval_batch is not None
            driver = DeviceChunkedBatchedDriver(
                bind(make_env_eval_batch, env_count, task),
                agent,
                chunk=_getattr(
                    args,
                    "device_actor_chunk",
                    _getattr(args, "mjx_actor_chunk", 32),
                ),
                mode=_eval_policy_mode(
                    label, action_selection, policy_role
                ),
                policy_seed=_getattr(args, "eval_policy_seed", 2_000_000),
            )
        elif host_batched_train:
            assert make_env_eval_batch is not None
            driver = BatchedDriver(
                bind(
                    make_env_eval_batch,
                    env_count,
                    task,
                    seed_start=worker_start,
                )
            )
        else:
            fns = [
                bind(make_env_eval, worker_start + i, task)
                for i in range(env_count)
            ]
            # Keep evaluation lanes parallel when training uses one environment.
            driver = embodied.Driver(
                fns,
                parallel=bool(_getattr(args, "eval_parallel", not args.debug)),
            )

        def global_worker_callback(callback):
            def wrapped(tran, worker, **kwargs):
                offset = (
                    eval_worker_offsets[eval_key]
                    if chunked_train else worker_start
                )
                worker = offset + int(worker)
                if worker >= int(args.eval_eps):
                    return
                callback(tran, worker, **kwargs)
            return wrapped

        driver.on_step(global_worker_callback(
            bind(add_eval_replay, eval_key=eval_key)))
        driver.on_step(global_worker_callback(
            bind(logfn, mode="eval", eval_key=eval_key)))
        driver.on_step(global_worker_callback(
            lambda tran, worker: eval_policy_fps.step()))
        return driver

    if run_eval and chunked_train:
        for eval_key, label, task, action_selection, policy_role in eval_runs:
            persistent_eval_drivers[eval_key] = build_eval_driver(
                eval_key, label, task, action_selection, policy_role
            )

    stream_train = [None]
    stream_report = [None]

    carry_train = [agent.init_train(args.batch_size)]
    carry_report = agent.init_report(args.batch_size)

    def stream_for(slot, replay, mode):
        if slot[0] is None:
            slot[0] = iter(agent.stream(make_stream(replay, mode)))
        return slot[0]

    def trainfn(tran, worker):
        del tran, worker
        if len(replay_train) < args.batch_size * args.batch_length:
            return
        updates = should_train(step)
        begin_burst = getattr(agent, "begin_train_burst", None)
        end_burst = getattr(agent, "end_train_burst", None)
        burst = (
            chunked_train and updates and
            callable(begin_burst) and callable(end_burst))
        if burst:
            begin_burst()
        try:
            for _ in range(updates):
                with elements.timer.section("stream_next"):
                    batch = next(stream_for(stream_train, replay_train, "train"))
                carry_train[0], outs, mets = agent.train(carry_train[0], batch)
                train_fps.step(batch_steps)
                if "replay" in outs:
                    replay_train.update(
                        _sanitize_replay_context_update(outs["replay"]))
                agg.add(mets, prefix="train")
        except BaseException:
            abort_burst = getattr(agent, "abort_train_burst", None)
            if burst and callable(abort_burst):
                abort_burst()
            raise
        else:
            if burst:
                end_burst()

    if chunked_train:
        driver_train.on_chunk(lambda batch: trainfn(None, None))
    else:
        driver_train.on_step(trainfn)

    last_train_log_step = [None]

    def write_train_metrics():
        logger.add(agg.result())
        logger.add(_summary_stats(train_completed), prefix="episode")
        train_completed.clear()
        logger.add(train_epstats.result(), prefix="epstats")
        logger.add(replay_train.stats(), prefix="replay")
        logger.add(usage.stats(), prefix="usage")
        logger.add(_fps_metrics(
            policy_fps,
            train_fps,
            args.train_ratio,
            eval_policy_fps=eval_policy_fps,
        ))
        logger.add(_numeric_timer_stats(), prefix="timer")
        if _adaptation_enabled(args):
            logger.add(_adaptation_step_metrics(args, step), prefix="adaptation")
        logger.write()
        last_train_log_step[0] = int(step)

    def reportfn(carry, stream):
        report_agg = elements.Agg()
        for _ in range(args.report_batches):
            batch = next(stream)
            carry, mets = agent.report(carry, batch)
            mets = _filter_report_metrics(
                mets, _getattr(args, "video_pred_log", True)
            )
            report_agg.add(mets)
        return carry, report_agg.result()

    def eval_reportfn(replay):
        """Reports only on trajectories from the current evaluation pass."""
        raw_stream = iter(make_stream(replay, "eval"))
        carry = agent.init_report(args.batch_size)
        report_agg = elements.Agg()
        prepare = getattr(agent, "prepare_stream_batch", None)
        if not callable(prepare):
            raise RuntimeError(
                "Current-checkpoint eval/report requires synchronous batch "
                "preparation support on the JAX agent.")
        seed_base = int(_getattr(args, "eval_report_seed", 3_000_000))
        for batch_index in range(args.report_batches):
            batch = prepare(
                next(raw_stream), seed_counter=seed_base + batch_index)
            carry, mets = agent.report(carry, batch)
            mets = _filter_report_metrics(
                mets, _getattr(args, "video_pred_log", True)
            )
            report_agg.add(mets)
        return report_agg.result()

    # Retain generations while the recovery monitor copies checkpoints.
    if _startup_preemption_should_exit(preemption, logdir):
        _close_startup_resources(
            logger=logger,
            drivers=(driver_train, *persistent_eval_drivers.values()),
        )
        return
    cp = _DurableCheckpoint(logdir / "checkpoint.pkl", keep=2)
    cp.step = step
    cp.agent = agent
    cp.replay_train = replay_train
    checkpoint_root = pathlib.Path(str(logdir)) / "checkpoint.pkl"
    adaptation = _adaptation_enabled(args)
    adaptation_offline = None
    if adaptation:
        if args.from_checkpoint:
            raise ValueError(
                "adaptation bootstrap and from_checkpoint are mutually exclusive."
            )
        cp.adaptation = _AdaptationLineage(_adaptation_lineage(args))
        if protocol_offline_updates(
            _getattr(args, "adaptation_protocol", "fresh_heads")
        ) > 0:
            adaptation_offline = _AdaptationOfflineState(
                _getattr(args, "adaptation_offline_updates", 0),
                _getattr(args, "adaptation_offline_save_every", 0),
            )
            cp.adaptation_offline = adaptation_offline

    def checkpoint_committed():
        _checkpoint_replays_committed(
            checkpoint_root,
            {"replay_train": replay_train},
            collect=not preemption.requested,
            cancelled=lambda: preemption.requested,
        )
    if args.from_checkpoint:
        elements.checkpoint.load(
            args.from_checkpoint,
            dict(agent=bind(agent.load, regex=args.from_checkpoint_regex)),
        )
    if adaptation and not _adaptation_child_checkpoint_exists(checkpoint_root):
        _bootstrap_adaptation(
            agent,
            replay_train,
            cp,
            checkpoint_committed,
            args,
            logdir,
        )
    else:
        cp.load_or_save()
        checkpoint_committed()
    if preemption.requested:
        print(
            "Checkpoint startup transaction completed after preemption; "
            "exiting without a redundant save."
        )
        _close_startup_resources(
            logger=logger,
            drivers=(driver_train, *persistent_eval_drivers.values()),
        )
        return
    if adaptation_offline is not None:
        offline_stream = (
            None
            if adaptation_offline.complete
            else stream_for(stream_train, replay_train, "train")
        )
        if not _run_algorithm2_offline_adaptation(
            agent=agent,
            replay=replay_train,
            stream=offline_stream,
            carry=carry_train,
            state=adaptation_offline,
            checkpoint=cp,
            checkpoint_committed=checkpoint_committed,
            preemption=preemption,
            step=step,
        ):
            _close_startup_resources(
                logger=logger,
                drivers=(driver_train, *persistent_eval_drivers.values()),
            )
            return
        if preemption.requested:
            print(
                "Post-offline step-0 checkpoint completed after preemption; "
                "exiting before task-policy evaluation."
            )
            _close_startup_resources(
                logger=logger,
                drivers=(driver_train, *persistent_eval_drivers.values()),
            )
            return
    removed_logger_records = _truncate_logger_jsonl_after(
        logdir, _logger_checkpoint_step(logger, step)
    )
    last_checkpoint_step = [int(step)]
    removed_train_records = train_record_writer.truncate_after("step", int(step))
    removed_eval_records = eval_record_writer.truncate_after(
        "checkpoint_step", int(step))
    train_worker_episode_counts.update(
        train_record_writer.next_indices("env_index", "episode_index"))
    if (
        removed_logger_records
        or removed_train_records
        or removed_eval_records
    ):
        print(
            "Restored checkpoint rollback:",
            f"scalar_log_records={sum(removed_logger_records.values())},",
            f"train_records={removed_train_records},",
            f"eval_records={removed_eval_records}.",
        )
    # Prime schedules from the restored global step to preserve absolute
    # evaluation and reporting intervals across restarts.
    should_report.prime(step)
    should_save.prime(step)
    interval_start, interval_end = _step_interval_bounds(
        step, report_every_steps
    )
    current_eval_complete = run_eval and _evaluation_pass_complete(
        eval_record_writer,
        eval_runs,
        args.eval_eps,
        minimum_step=interval_start,
        maximum_step=min(int(step), interval_end),
    )
    # Resume an incomplete evaluation in the current interval before training.
    initial_eval_pending = (
        (run_eval and not current_eval_complete)
        or (not run_eval and int(step) == 0)
    )

    try:
        print("Start training loop")
        train_policy = lambda *items: agent.policy(*items, mode="train")

        def eval_policy(label, action_selection, policy_role):
            counter = [int(_getattr(args, "eval_policy_seed", 2_000_000))]

            def policy(*items):
                seed_counter = counter[0]
                counter[0] += 1
                return agent.policy(
                    *items,
                    mode=_eval_policy_mode(
                        label, action_selection, policy_role
                    ),
                    seed_counter=seed_counter,
                )

            return policy

        driver_train.reset(agent.init_policy)
        while True:
            if _save_preempt_checkpoint_if_needed(
                preemption, cp, checkpoint_committed
            ):
                break
            target_reached = int(step) >= int(args.steps)
            if target_reached:
                # Episode identities prevent duplicate final evaluations
                # after a restart.
                report_now = run_eval and not _evaluation_pass_complete(
                    eval_record_writer,
                    eval_runs,
                    args.eval_eps,
                    minimum_step=int(step),
                    maximum_step=int(step),
                )
            else:
                report_now = should_report(step)
                if initial_eval_pending:
                    report_now = True
                    initial_eval_pending = False
            if report_now:
                if run_eval:
                    for (
                        eval_key,
                        label,
                        task,
                        action_selection,
                        policy_role,
                    ) in eval_runs:
                        if _evaluation_run_complete(
                            eval_record_writer,
                            args.eval_eps,
                            task,
                            label,
                            action_selection,
                            policy_role,
                            minimum_step=int(step),
                            maximum_step=int(step),
                        ):
                            print(
                                "Evaluation already complete",
                                _eval_prefix(label, action_selection),
                                f"policy_role={policy_role}",
                                f"at step {int(step)}; skipping.",
                            )
                            continue
                        eval_prefix = _eval_prefix(label, action_selection)
                        print(
                            "Evaluation",
                            eval_prefix,
                            f"policy_role={policy_role}",
                        )
                        eval_completed[eval_key].clear()
                        eval_worker_episode_counts[eval_key].clear()
                        replay_label = (
                            f"{action_selection}_{policy_role}_"
                            f"{label or 'eval'}"
                        )
                        current_replay = make_replay_eval(replay_label)
                        seed_bank = _eval_seed_bank(args, task, label)
                        if len(seed_bank) < int(args.eval_eps):
                            raise ValueError(
                                "Evaluation seed bank is too short for the "
                                f"requested episode budget: {len(seed_bank)} "
                                f"seeds for eval_eps={args.eval_eps} and task "
                                f"{task!r}."
                            )
                        policy = eval_policy(
                            label, action_selection, policy_role
                        )
                        for worker_start, env_count in _evaluation_waves(
                            args.eval_eps, args.eval_envs
                        ):
                            eval_worker_offsets[eval_key] = worker_start
                            make_driver = bind(
                                build_eval_driver,
                                eval_key,
                                label,
                                task,
                                action_selection,
                                policy_role,
                                env_count=env_count,
                                worker_start=worker_start,
                            )
                            with _eval_driver_scope(
                                persistent_eval_drivers, eval_key, make_driver
                            ) as driver:
                                active_eval_replay[eval_key] = current_replay
                                # Synchronize before the next host action.
                                agent.sync_policy_if_needed()
                                if chunked_train:
                                    reset_seeds = list(seed_bank[
                                        worker_start: worker_start + env_count
                                    ])
                                    reset_seeds.extend(
                                        [reset_seeds[-1]]
                                        * (driver.length - len(reset_seeds))
                                    )
                                    driver.reset(
                                        agent.init_policy,
                                        reset_seed_bank=reset_seeds,
                                    )
                                else:
                                    driver.reset(agent.init_policy)
                                try:
                                    target = worker_start + env_count
                                    while len(eval_completed[eval_key]) < target:
                                        driver(
                                            policy,
                                            steps=max(1, int(driver.length)),
                                        )
                                        if preemption.requested:
                                            break
                                finally:
                                    active_eval_replay[eval_key] = None
                            if preemption.requested:
                                break
                        if preemption.requested:
                            break
                        completed = sorted(
                            eval_completed[eval_key],
                            key=lambda item: (
                                item["env_index"],
                                item["worker_episode"],
                            ),
                        )
                        if len(completed) != int(args.eval_eps):
                            raise RuntimeError(
                                f"Expected {args.eval_eps} evaluation episodes "
                                f"for {task!r}, received {len(completed)}.")
                        logger.add(_summary_stats(completed), prefix=eval_prefix)
                        records = [
                            _eval_episode_record(
                                episode,
                                args=args,
                                checkpoint_step=int(step),
                                task=task,
                                label=label,
                                action_selection=action_selection,
                                policy_role=policy_role,
                                episode_index=index,
                            )
                            for index, episode in enumerate(completed)
                        ]
                        eval_record_writer.write(records)
                        eval_completed[eval_key].clear()
                        logger.add(
                            eval_epstats[eval_key].result(),
                            prefix=f"{eval_prefix}/epstats",
                        )
                        if len(current_replay):
                            mets = eval_reportfn(current_replay)
                            logger.add(mets, prefix=f"{eval_prefix}/report")
                if _save_preempt_checkpoint_if_needed(
                    preemption, cp, checkpoint_committed
                ):
                    break
                if len(replay_train):
                    stream = stream_for(stream_report, replay_train, "report")
                    carry_report, mets = reportfn(carry_report, stream)
                    logger.add(mets, prefix="report")
                # Persist step-zero evaluation before the first training log.
                if _adaptation_enabled(args):
                    logger.add(
                        _adaptation_step_metrics(args, step),
                        prefix="adaptation",
                    )
                logger.write()

            if target_reached:
                break

            driver_train(train_policy, steps=10)
            if _save_preempt_checkpoint_if_needed(
                preemption, cp, checkpoint_committed
            ):
                break

            if should_log(step):
                write_train_metrics()

            if should_save(step):
                _save_checkpoint(cp, checkpoint_committed)
                last_checkpoint_step[0] = int(step)
                if preemption.requested:
                    print(
                        "Checkpoint committed after preemption signal; "
                        "skipping a duplicate save."
                    )
                    break
        completed_naturally = (
            not preemption.requested and int(step) >= int(args.steps)
        )
        if (
            completed_naturally
            and last_checkpoint_step[0] != int(step)
        ):
            # Preserve the exact final learner/replay state even when a short
            # run never reaches its first periodic checkpoint boundary.
            _save_final_checkpoint_if_complete(
                preemption,
                cp,
                step,
                args.steps,
                checkpoint_committed,
            )
        if completed_naturally and last_train_log_step[0] != int(step):
            write_train_metrics()
    finally:
        logger.close()


def _add_episode_logs(
    episode,
    scalar_logs,
    tran,
    worker,
    *,
    video_policy_log=True,
    video_policy_stride=1,
    video_policy_step=0,
):
    for key, value in tran.items():
        if value.dtype == np.uint8 and value.ndim == 3:
            if (
                video_policy_log
                and worker == 0
                and _keep_policy_video_frame(
                    video_policy_step,
                    video_policy_stride,
                    tran.get("is_last", False),
                )
            ):
                episode.add(_policy_image_key(key), value, agg="stack")
        elif key.startswith("log/"):
            scalar_logs.add(key, value)


def _add_episode_logs_from_batch(
    episode,
    scalar_logs,
    batch,
    time_index,
    worker,
    *,
    video_policy_log=True,
    video_policy_stride=1,
    video_policy_step=0,
):
    for key, values in batch.items():
        value = values[time_index, worker]
        if value.dtype == np.uint8 and value.ndim == 3:
            if (
                video_policy_log
                and worker == 0
                and _keep_policy_video_frame(
                    video_policy_step,
                    video_policy_stride,
                    batch.get("is_last", False)[time_index, worker],
                )
            ):
                episode.add(_policy_image_key(key), value, agg="stack")
        elif key.startswith("log/"):
            scalar_logs.add(key, value)


def _keep_policy_video_frame(step, stride, is_last=False):
    stride = max(1, int(stride))
    return int(step) % stride == 0 or bool(np.asarray(is_last))


def _policy_image_key(key):
    key = key.removeprefix("log/")
    return f"policy_{key}"


def _filter_report_metrics(metrics, video_pred_log=True):
    if video_pred_log:
        return metrics
    return {
        key: value
        for key, value in metrics.items()
        if not key.startswith("openloop/")
    }


def _finish_episode_stats(result, epstats):
    rew = np.asarray(result.pop("rewards", ()), dtype=np.float32)
    if rew.size:
        result["reward_mean"] = float(rew.mean())
        result["reward_std"] = float(rew.std())
        result["reward_min"] = float(rew.min())
        result["reward_max"] = float(rew.max())
        result["reward_nonzero_rate"] = float((np.abs(rew) >= 0.01).mean())
    if rew.size > 1:
        result["reward_rate"] = float(
            (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean())
    epstats.add(result)
    return {
        key: _json_value(value)
        for key, value in result.items()
        if not isinstance(value, str) and np.asarray(value).ndim == 0
    }


def _summary_stats(episodes, keys=("score", "length")):
    if not episodes:
        return {}
    result = {}
    for key in keys:
        values = np.asarray([episode[key] for episode in episodes], dtype=np.float32)
        result[key] = float(values.mean())
        if values.size > 1:
            std = float(values.std(ddof=1))
            stderr = float(std / np.sqrt(values.size))
        else:
            std = 0.0
            stderr = 0.0
        result[f"{key}_std"] = std
        result[f"{key}_stderr"] = stderr
        result[f"{key}_count"] = int(values.size)
        result[f"{key}_min"] = float(values.min())
        result[f"{key}_max"] = float(values.max())
        for suffix, quantile in (
            ("p10", 0.10),
            ("p25", 0.25),
            ("median", 0.50),
            ("p75", 0.75),
            ("p90", 0.90),
        ):
            result[f"{key}_{suffix}"] = float(np.quantile(values, quantile))
    return result


def _fps_metrics(policy_fps, train_fps, train_ratio, eval_policy_fps=None):
    train_fps_value = train_fps.result()
    result = {
        "fps/policy": policy_fps.result(),
        "fps/train": train_fps_value,
    }
    if eval_policy_fps is not None:
        result["fps/eval_policy"] = eval_policy_fps.result()
    if train_ratio:
        result["fps/policy_train_cap"] = train_fps_value / train_ratio
    return result


def _worker_episode_targets(episodes, envs):
    """Give every global fixed-seed lane exactly one evaluation episode."""
    episodes = max(0, int(episodes))
    envs = max(0, int(envs))
    if not episodes or not envs:
        return ()
    return (True,) * episodes


def _evaluation_waves(episodes, envs):
    """Partition unique fixed-seed lanes into bounded-concurrency waves."""
    episodes = max(0, int(episodes))
    envs = max(0, int(envs))
    if not episodes or not envs:
        return ()
    return tuple(
        (start, min(envs, episodes - start))
        for start in range(0, episodes, envs)
    )


def _evaluation_run_complete(
    writer,
    episodes,
    task,
    label,
    action_selection,
    policy_role,
    *,
    minimum_step,
    maximum_step,
):
    episodes = int(episodes)
    if episodes <= 0:
        return True
    return writer.count_matching_between(
        "checkpoint_step",
        minimum_step,
        maximum_step,
        task=str(task),
        eval_label=str(label or "eval"),
        action_selection=action_selection,
        policy_role=_eval_policy_role(policy_role),
    ) >= episodes


def _evaluation_pass_complete(
    writer, eval_runs, episodes, *, minimum_step, maximum_step
):
    return all(
        _evaluation_run_complete(
            writer,
            episodes,
            task,
            label,
            action_selection,
            policy_role,
            minimum_step=minimum_step,
            maximum_step=maximum_step,
        )
        for _, label, task, action_selection, policy_role in eval_runs
    )


def _eval_seed_bank(args, task, label):
    value = _getattr(args, "eval_seed_bank", None)
    if isinstance(value, Mapping):
        value = value.get(task, value.get(label, value.get("default")))
    if value is not None:
        bank = [int(seed) for seed in value]
        if bank:
            return bank
    base = (
        int(_getattr(args, "train_seed", 0))
        + int(_getattr(args, "eval_seed_offset", 1_000_000))
    )
    return [base + index for index in range(int(args.eval_eps))]


def _eval_episode_record(
    episode,
    *,
    args,
    checkpoint_step,
    task,
    label,
    action_selection,
    policy_role,
    episode_index,
):
    action_selection = _eval_action_selection(action_selection)
    policy_role = _eval_policy_role(policy_role)
    worker = int(episode["env_index"])
    worker_episode = int(episode["worker_episode"])
    seed_bank = _eval_seed_bank(args, task, label)
    if worker >= len(seed_bank):
        raise ValueError(
            "Evaluation seed bank is too short for completed episode "
            f"lane {worker}: {len(seed_bank)} seeds for task {task!r}.")
    action_repeat = int(_getattr(args, "action_repeat", 1))
    record = {
        "version": 3,
        "checkpoint_step": int(checkpoint_step),
        "environment_samples": int(checkpoint_step) * action_repeat,
        "collector_task": str(_getattr(args, "train_task", "")),
        "train_seed": int(_getattr(args, "train_seed", 0)),
        "task": str(task),
        "eval_label": str(label or "eval"),
        "policy_role": policy_role,
        "protocol": str(_getattr(args, "protocol", "standard")),
        "task_registry_hash": _getattr(args, "task_registry_hash", None),
        "port_version": _getattr(args, "port_version", None),
        "action_selection": action_selection,
        "episode_index": int(episode_index),
        "env_index": worker,
        "worker_episode": worker_episode,
        "env_seed": int(seed_bank[worker]),
        "policy_seed": int(_getattr(args, "eval_policy_seed", 2_000_000)),
        "deterministic_actions": action_selection == "greedy",
        "return": float(episode["score"]),
        "length": int(episode["length"]),
        "diagnostics": dict(episode.get("diagnostics", {})),
    }
    if _adaptation_enabled(args):
        lineage = _adaptation_lineage(args)
        lineage_payload = json.dumps(
            lineage,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        lineage_sha256 = hashlib.sha256(lineage_payload).hexdigest()
        pretraining = lineage["pretraining_environment_steps"]
        adaptation_step = int(checkpoint_step)
        record.update(
            adaptation_step=adaptation_step,
            pretraining_environment_steps=pretraining,
            lifetime_environment_steps=pretraining + adaptation_step,
            strict_task_blind_pretraining=lineage[
                "strict_task_blind_pretraining"
            ],
            parent_idm=lineage["idm"],
            parent_seed=lineage["parent_seed"],
            parent_run_uid=lineage["parent_run_uid"],
            parent_source_digest=lineage["parent_source_digest"],
            parent_checkpoint_sha256=lineage["parent_checkpoint_sha256"],
            parent_replay_inventory_sha256=lineage[
                "parent_replay_inventory_sha256"
            ],
            representation_backend=lineage["backend"],
            adaptation_model=lineage["model"],
            adaptation_lineage=lineage,
            adaptation_lineage_sha256=lineage_sha256,
            algorithm2_offline_updates=lineage[
                "algorithm2_offline_updates"
            ],
            algorithm2_offline_environment_steps=0,
            algorithm2_offline_complete=True,
            evaluation_phase=(
                ("pretrained_task_policy" if lineage.get("protocol") == "pretrained_heads"
                 else "post_algorithm2_offline")
                if adaptation_step == 0
                else "online_task_adaptation"
            ),
        )
    return record


def _adaptation_step_metrics(args, step):
    adaptation_step = int(step)
    pretraining = int(_getattr(args, "pretraining_environment_steps", 0))
    return {
        "environment_steps": adaptation_step,
        "pretraining_environment_steps": pretraining,
        "lifetime_environment_steps": pretraining + adaptation_step,
        "algorithm2_offline_updates": int(
            _getattr(args, "adaptation_offline_updates", 0)
        ),
        "algorithm2_offline_environment_steps": 0,
        "algorithm2_offline_complete": 1,
        "strict_task_blind_pretraining": int(
            bool(
                _getattr(
                    args,
                    "adaptation_strict_task_blind_pretraining",
                    False,
                )
            )
        ),
    }


def _chunk_amount(batch):
    return int(np.prod(next(iter(batch.values())).shape[:2]))


_REPLAY_CONTEXT_WARNING_KEYS = set()


def _sanitize_replay_context_update(data):
    result = {}
    for key, value in data.items():
        if (
            not key.startswith(("enc/", "dyn/", "dec/"))
            or not np.issubdtype(value.dtype, np.floating)
        ):
            result[key] = value
            continue
        finite = np.isfinite(value)
        if finite.all():
            result[key] = value
            continue
        if key not in _REPLAY_CONTEXT_WARNING_KEYS:
            _REPLAY_CONTEXT_WARNING_KEYS.add(key)
            print(
                "Warning: replacing non-finite replay context update values: "
                f"{_nonfinite_summary(key, value, finite)}"
            )
        result[key] = np.nan_to_num(
            value, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    return result


def _nonfinite_summary(key, value, finite):
    bad = ~finite
    indices = np.argwhere(bad)
    first = tuple(int(x) for x in indices[0]) if len(indices) else None
    good = value[finite]
    if good.size:
        minmax = f"min={float(good.min()):.6g}, max={float(good.max()):.6g}"
    else:
        minmax = "no finite values"
    return (
        f"key={key!r}, shape={value.shape}, dtype={value.dtype}, "
        f"nonfinite={int(bad.sum())}, first_index={first}, {minmax}"
    )


_EVAL_ACTION_SELECTIONS = ("sampled", "greedy")
_EVAL_POLICY_ROLES = ("base", "task_head")


def _eval_action_selection(value):
    value = str(value).strip().lower()
    if value not in _EVAL_ACTION_SELECTIONS:
        raise ValueError(
            "Evaluation action selection must be one of "
            f"{_EVAL_ACTION_SELECTIONS}, got {value!r}."
        )
    return value


def _eval_policy_role(value):
    value = str(value).strip().lower()
    if value not in _EVAL_POLICY_ROLES:
        raise ValueError(
            "Evaluation policy role must be one of "
            f"{_EVAL_POLICY_ROLES}, got {value!r}."
        )
    return value


def _parse_eval_policy_roles(value):
    if value is None:
        return {}
    if isinstance(value, Mapping):
        items = value.items()
    else:
        if isinstance(value, str):
            value = [item for item in value.strip("[]").split(",") if item]
        parsed = []
        for spec in value:
            spec = str(spec).strip()
            if "=" not in spec:
                raise ValueError(
                    "Evaluation policy role specs must use label=role syntax; "
                    f"got {spec!r}."
                )
            parsed.append(spec.split("=", 1))
        items = parsed
    result = {}
    for label, role in items:
        label = str(label).strip()
        if not label:
            raise ValueError("Evaluation policy role labels must not be empty.")
        if label in result:
            raise ValueError(f"Duplicate evaluation policy role label: {label!r}.")
        result[label] = _eval_policy_role(role)
    return result


def _evaluation_runs(eval_tasks, action_selections=None, policy_roles=None):
    """Expands evaluation tasks over action selections and policy roles."""
    eval_tasks = tuple(eval_tasks)
    selections = (
        _EVAL_ACTION_SELECTIONS
        if action_selections is None
        else tuple(_eval_action_selection(item) for item in action_selections)
    )
    if not selections:
        raise ValueError("Evaluation action selections must not be empty.")
    if len(set(selections)) != len(selections):
        raise ValueError(
            f"Evaluation action selections contain duplicates: {selections!r}."
        )
    roles = _parse_eval_policy_roles(policy_roles)
    labels = {str(label or "").strip() for label, _ in eval_tasks}
    unknown = set(roles) - labels
    if unknown:
        raise ValueError(
            "Evaluation policy roles reference unknown labels: "
            f"{sorted(unknown)!r}."
        )
    result = []
    for label, task in eval_tasks:
        label = str(label or "").strip()
        default_role = "task_head" if label else "base"
        policy_role = roles.get(label, default_role)
        if policy_role == "task_head" and not label:
            raise ValueError(
                "Task-head evaluation requires a nonempty task label."
            )
        for action_selection in selections:
            eval_key = (label, action_selection, policy_role)
            result.append(
                (eval_key, label, task, action_selection, policy_role)
            )
    return tuple(result)


def _eval_prefix(label, action_selection="sampled"):
    action_selection = _eval_action_selection(action_selection)
    label = str(label or "").strip()
    root = "eval" if action_selection == "sampled" else "eval_greedy"
    return f"{root}/{label}" if label else root


def _eval_policy_mode(label, action_selection="sampled", policy_role=None):
    action_selection = _eval_action_selection(action_selection)
    label = str(label or "").strip()
    if policy_role is None:
        policy_role = "task_head" if label else "base"
    policy_role = _eval_policy_role(policy_role)
    if policy_role == "base":
        label = ""
    elif not label:
        raise ValueError("Task-head evaluation requires a nonempty task label.")
    root = "eval" if action_selection == "sampled" else "eval_greedy"
    return f"{root}:{label}" if label else root


def _getattr(obj, name, default):
    return getattr(obj, name) if hasattr(obj, name) else default
