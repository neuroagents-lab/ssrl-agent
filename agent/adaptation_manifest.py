"""Validation for immutable Plan2Explore adaptation parent cohorts."""

from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import os
import pathlib
import pickle
import pickletools
import re
import stat
import tempfile
import zipfile
from typing import Iterable, Mapping

if __package__:
    from .adaptation_protocol import (
        is_walker_source_transfer, validate_env_config, validate_parent_step,
        validate_walker_source, PROTOCOLS, TRANSFER_COMPONENTS, STATELESS_COMPONENTS,
        validate_protocol, transfer_components, source_components, remap_source_parameters,
    )
else:
    # Resolve helpers from this source tree when loaded directly by file path.
    import runpy

    _protocol = runpy.run_path(str(pathlib.Path(__file__).with_name("adaptation_protocol.py")))
    validate_env_config = _protocol["validate_env_config"]
    validate_parent_step = _protocol["validate_parent_step"]
    is_walker_source_transfer = _protocol["is_walker_source_transfer"]
    validate_walker_source = _protocol["validate_walker_source"]
    for _name in ("PROTOCOLS", "TRANSFER_COMPONENTS", "STATELESS_COMPONENTS",
                  "validate_protocol", "transfer_components", "source_components",
                  "remap_source_parameters"):
        globals()[_name] = _protocol[_name]


SCHEMA_VERSION = 2
CERTIFICATE_SCHEMA_VERSION = 1
CERTIFICATE_TYPE = "plan2explore-dreamer-transfer-compatibility"
VALIDATION_RECEIPT_SCHEMA_VERSION = 1
VALIDATION_RECEIPT_TYPE = "plan2explore-cohort-validation-receipt"
PARENT_STEP = 5_010_000
ADAPTATION_STEPS = 5_000_000
ALGORITHM2_REFERENCE_TASK_STEPS = 150_000
ALGORITHM2_OFFLINE_UPDATES = 4_688
ALGORITHM2_OFFLINE_SAVE_EVERY = 1_000
SUPPORTED_BACKENDS = ("dreamerv3", "r2dreamer", "ne_dreamer")
_GENERATION = re.compile(
    r"^(?P<timestamp>[0-9]{8}T[0-9]{6}F[0-9]{6})"
    r"(?:-(?P<step>[0-9]{12}))?$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_TEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
_CHUNK_NAME = re.compile(
    r"^(?P<time>[0-9]{8}T[0-9]{6}F[0-9]{6})-"
    r"(?P<uuid>[0-9A-Za-z]{22})-(?P<succ>[0-9A-Za-z]{22})-"
    r"(?P<length>[1-9][0-9]*)\.npz(?:\.zst)?$"
)
_REPLAY_FORMAT = "embodied-replay-chunks-v1"
_MAX_CHUNK_BYTES = 1024 * 1024 * 1024
_MAX_CERTIFICATE_BYTES = 64 * 1024 * 1024
_BASE62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_PARAMETER_KEY = re.compile(
    r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$"
)


class ManifestError(ValueError):
    pass


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(
            f"Replay inventory cannot load global {module}.{name}."
        )

    def persistent_load(self, pid):
        del pid
        raise pickle.UnpicklingError(
            "Replay inventory cannot contain persistent IDs."
        )


def _read_regular_bytes(path, *, label: str, max_bytes: int | None = None):
    """Reads one stable regular-file identity without following symlinks."""
    path = pathlib.Path(path)
    try:
        path_before = path.lstat()
    except OSError as exc:
        raise ManifestError(f"Could not inspect {label}: {path}") from exc
    if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(
        path_before.st_mode
    ):
        raise ManifestError(f"Expected a regular {label}: {path}")
    if max_bytes is not None and path_before.st_size > max_bytes:
        raise ManifestError(f"{label.capitalize()} exceeds the size limit.")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ManifestError(f"Could not open {label}: {path}") from exc
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino) != (
            path_before.st_dev,
            path_before.st_ino,
        ):
            raise ManifestError(f"{label.capitalize()} changed before read: {path}")
        payload = (
            stream.read() if max_bytes is None else stream.read(max_bytes + 1)
        )
        after = os.fstat(stream.fileno())
    if max_bytes is not None and len(payload) > max_bytes:
        raise ManifestError(f"{label.capitalize()} exceeds the size limit.")
    if _stat_signature(before) != _stat_signature(after):
        raise ManifestError(f"{label.capitalize()} changed while reading: {path}")
    try:
        path_after = path.lstat()
    except OSError as exc:
        raise ManifestError(f"{label.capitalize()} disappeared: {path}") from exc
    if _stat_signature(path_before) != _stat_signature(path_after):
        raise ManifestError(f"{label.capitalize()} path changed while reading: {path}")
    return payload


def file_sha256(path: os.PathLike[str] | str) -> str:
    path = pathlib.Path(path)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ManifestError(f"Expected a regular file: {path}")
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        before = os.fstat(stream.fileno())
        while block := stream.read(1024 * 1024):
            digest.update(block)
        after = os.fstat(stream.fileno())
    if _stat_signature(before) != _stat_signature(after):
        raise ManifestError(f"File changed while hashing: {path}")
    return digest.hexdigest()


def checkpoint_sha256(
    generation: os.PathLike[str] | str,
    *,
    evidence: list[dict] | None = None,
) -> str:
    generation = pathlib.Path(generation)
    _real_directory(generation, "checkpoint generation")
    digest = hashlib.sha256()
    children = sorted(generation.iterdir(), key=lambda path: path.name)
    if not children:
        raise ManifestError(f"Empty checkpoint generation: {generation}")
    for child in children:
        info = child.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ManifestError(f"Unsafe checkpoint entry: {child}")
        name = child.name.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(child, flags), "rb") as stream:
            before = os.fstat(stream.fileno())
            if _stat_signature(info) != _stat_signature(before):
                raise ManifestError(
                    f"Checkpoint entry changed before hashing: {child}"
                )
            while block := stream.read(1024 * 1024):
                digest.update(block)
            after = os.fstat(stream.fileno())
        if _stat_signature(before) != _stat_signature(after):
            raise ManifestError(f"Checkpoint entry changed while hashing: {child}")
        path_after = child.lstat()
        if _stat_signature(after) != _stat_signature(path_after):
            raise ManifestError(f"Checkpoint entry changed after hashing: {child}")
        if evidence is not None:
            evidence.append(
                {"name": child.name, "stat": _stat_record(path_after)}
            )
    return digest.hexdigest()


def schema_sha256(records: Iterable[Mapping]) -> str:
    """Return the canonical fingerprint of key/shape/dtype schema records."""
    normalized = _validate_schema_records(records, label="schema")
    return _json_sha256(normalized)


def parameter_records(params: Mapping, components: Iterable[str]) -> list[dict]:
    """Describe selected checkpoint arrays without serializing executable data."""
    try:
        import numpy as np
    except ImportError as exc:
        raise ManifestError(
            "Compatibility certification requires NumPy."
        ) from exc
    components = _validate_components(tuple(components))
    if not isinstance(params, Mapping):
        raise ManifestError("Agent checkpoint params must be a mapping.")
    records = []
    for raw_key, value in params.items():
        if not isinstance(raw_key, str) or not _PARAMETER_KEY.fullmatch(raw_key):
            raise ManifestError(f"Unsafe checkpoint parameter key: {raw_key!r}")
        if raw_key.split("/", 1)[0] not in components:
            continue
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ManifestError(
                f"Selected parameter {raw_key!r} has an object dtype."
            )
        contiguous = np.ascontiguousarray(array)
        digest = hashlib.sha256()
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(tuple(array.shape)).encode("ascii"))
        digest.update(contiguous.view(np.uint8))
        records.append(
            {
                "key": raw_key,
                "shape": list(array.shape),
                "dtype": array.dtype.str,
                "sha256": digest.hexdigest(),
            }
        )
    records.sort(key=lambda item: item["key"])
    _validate_parameter_records(records, components)
    return records


def create_compatibility_certificate(
    *,
    source_agent_pickle: os.PathLike[str] | str,
    target_agent_pickle: os.PathLike[str] | str,
    target_replay_schema: Mapping,
    backend: str,
    idm: str,
    parent_seed: int,
    task: str,
    model: str,
    run_uid: str,
    source_digest: str,
    checkpoint_sha256_value: str,
    replay_inventory_sha256: str,
    parent_step: int = PARENT_STEP,
    env_config: str | None = None,
    parent_task: str | None = None,
    parent_env_config: str | None = None,
    parent_algorithm: str | None = None,
    protocol: str = "fresh_heads",
) -> dict:
    """Create a safe JSON certificate from explicitly trusted pickle inputs.

    This function intentionally unpickles both agent files. It must only be run
    as an explicit certification operation on trusted local checkpoints. Normal
    launcher preflight never calls it and only reads the resulting strict JSON.
    """
    backend = str(backend)
    if backend not in SUPPORTED_BACKENDS:
        raise ManifestError(f"Unsupported adaptation backend: {backend!r}")
    components = transfer_components(backend, protocol)
    mapping = source_components(backend, protocol, task=task, env_config=env_config,
                                parent_algorithm=parent_algorithm or "p2e", parent_task=parent_task)
    identity = {
        "backend": backend,
        "idm": idm,
        "parent_seed": parent_seed,
        "task": task,
        "model": model,
        "run_uid": run_uid,
        "source_digest": source_digest,
        "checkpoint_sha256": checkpoint_sha256_value,
        "replay_inventory_sha256": replay_inventory_sha256,
        "parent_step": parent_step,
    }
    if env_config is not None:
        identity["env_config"] = env_config
    if parent_task is not None or parent_env_config is not None or parent_algorithm is not None:
        identity.update(parent_task=parent_task, parent_env_config=parent_env_config,
                        parent_algorithm=parent_algorithm)
    if protocol != "fresh_heads":
        identity["protocol"] = protocol
    _validate_certificate_identity(identity)
    source_params = _trusted_agent_params(source_agent_pickle)
    target_params = _trusted_agent_params(target_agent_pickle)
    source = parameter_records(remap_source_parameters(source_params, mapping), components)
    child_with_hashes = parameter_records(target_params, components)
    child = [
        {key: record[key] for key in ("key", "shape", "dtype")}
        for record in child_with_hashes
    ]
    source_schema = [
        {key: record[key] for key in ("key", "shape", "dtype")}
        for record in source
    ]
    if source_schema != child:
        source_map = {record["key"]: record for record in source_schema}
        child_map = {record["key"]: record for record in child}
        raise ManifestError(
            "Selected parent parameters are incompatible with the fresh child: "
            f"missing={sorted(set(child_map) - set(source_map))}, "
            f"unexpected={sorted(set(source_map) - set(child_map))}, "
            f"incompatible={sorted(key for key in set(source_map) & set(child_map) if source_map[key] != child_map[key])}"
        )
    replay = _validate_target_replay_schema(
        target_replay_schema,
        backend=backend,
        model=model,
        task=task,
        env_config=env_config,
    )
    return {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "certificate_type": CERTIFICATE_TYPE,
        "identity": identity,
        "transfer": {
            "components": list(components),
            **({"source_components": mapping} if protocol == "pretrained_heads" else {}),
            "source_parameters": source,
            "source_schema_sha256": _json_sha256(source_schema),
            "source_content_sha256": _parameter_content_sha256(source),
            "expected_child_parameters": child,
            "expected_child_schema_sha256": _json_sha256(child),
        },
        "replay": {
            "normalized_target_schema": replay,
            "normalized_target_schema_sha256": _json_sha256(replay),
        },
    }


def write_compatibility_certificate(
    path: os.PathLike[str] | str, certificate: Mapping
) -> str:
    """Atomically write a new read-only certificate and return its SHA-256."""
    path = pathlib.Path(path).expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise ManifestError(f"Refusing to replace compatibility certificate: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(certificate, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(payload) > _MAX_CERTIFICATE_BYTES:
        raise ManifestError("Compatibility certificate exceeds the size limit.")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o440)
        os.replace(temporary, path)
        directory = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return file_sha256(path)


def create_validation_receipt(
    manifest: Mapping,
    *,
    backend: str,
    idms: Iterable[str] | None,
    parent_seed: int,
    task: str,
    model: str,
    env_config: str | None = None,
) -> dict:
    """Run full byte validation once and return reusable stat-bound evidence."""
    parents = select_and_validate(
        dict(manifest),
        backend=backend,
        idms=idms,
        parent_seed=parent_seed,
        task=task,
        model=model,
        env_config=env_config,
        collect_evidence=True,
    )
    entries = []
    for parent in parents:
        evidence = parent.pop("_validation_evidence")
        entries.append(
            {
                "backend": parent["backend"],
                "idm": parent["idm"],
                "parent_seed": parent["parent_seed"],
                "task": parent["task"],
                "model": parent["model"],
                "run_uid": parent["run_uid"],
                **({"env_config": parent["env_config"]} if "env_config" in parent else {}),
                "evidence": evidence,
            }
        )
    entries.sort(
        key=lambda entry: (
            entry["backend"], entry["parent_seed"], entry["idm"]
        )
    )
    return {
        "schema_version": VALIDATION_RECEIPT_SCHEMA_VERSION,
        "receipt_type": VALIDATION_RECEIPT_TYPE,
        "manifest_sha256": manifest["_sha256"],
        "entries": entries,
    }


def write_validation_receipt(
    path: os.PathLike[str] | str, receipt: Mapping
) -> str:
    """Atomically write a new read-only validation receipt."""
    return _write_readonly_json(
        path,
        receipt,
        label="validation receipt",
    )


def load_validation_receipt(
    path: os.PathLike[str] | str,
    *,
    expected_sha256: str,
) -> dict:
    path = pathlib.Path(path).expanduser().absolute()
    if not _SHA256.fullmatch(str(expected_sha256)):
        raise ManifestError(
            "An immutable validation receipt requires its lowercase SHA-256."
        )
    payload = _read_regular_bytes(
        path, label="validation receipt", max_bytes=_MAX_CERTIFICATE_BYTES
    )
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ManifestError(
            f"Validation receipt digest mismatch: {actual} != "
            f"{expected_sha256}"
        )
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_unique_json_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(
            f"Could not read validation receipt: {exc}"
        ) from exc
    required = {
        "schema_version", "receipt_type", "manifest_sha256", "entries"
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ManifestError("Malformed validation receipt top level.")
    if value["schema_version"] != VALIDATION_RECEIPT_SCHEMA_VERSION:
        raise ManifestError("Unsupported validation receipt schema.")
    if value["receipt_type"] != VALIDATION_RECEIPT_TYPE:
        raise ManifestError("Unexpected validation receipt type.")
    if not isinstance(value["manifest_sha256"], str) or not _SHA256.fullmatch(
        value["manifest_sha256"]
    ):
        raise ManifestError("Invalid validation receipt manifest digest.")
    if not isinstance(value["entries"], list) or not value["entries"]:
        raise ManifestError("Validation receipt entries must be non-empty.")
    return {**value, "_path": str(path), "_sha256": actual}


def _write_readonly_json(path, value: Mapping, *, label: str) -> str:
    path = pathlib.Path(path).expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise ManifestError(f"Refusing to replace {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(payload) > _MAX_CERTIFICATE_BYTES:
        raise ManifestError(f"{label.capitalize()} exceeds the size limit.")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o440)
        os.replace(temporary, path)
        directory = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return file_sha256(path)


def load_manifest(
    path: os.PathLike[str] | str, *, expected_sha256: str
) -> dict:
    path = pathlib.Path(path).expanduser().absolute()
    if not _SHA256.fullmatch(str(expected_sha256)):
        raise ManifestError(
            "An immutable cohort manifest requires its lowercase SHA-256."
        )
    payload = _read_regular_bytes(
        path, label="parent cohort manifest", max_bytes=_MAX_CERTIFICATE_BYTES
    )
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ManifestError(
            f"Parent cohort manifest digest mismatch: {actual} != "
            f"{expected_sha256}"
        )
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_unique_json_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Could not read parent cohort manifest: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError("Unsupported parent cohort manifest schema.")
    if set(value) != {"schema_version", "entries"}:
        raise ManifestError("Parent cohort manifest has unexpected top-level fields.")
    if not isinstance(value["entries"], list):
        raise ManifestError("Parent cohort entries must be a list.")
    return {**value, "_path": str(path), "_sha256": actual}


def select_and_validate(
    manifest: dict,
    *,
    backend: str,
    idms: Iterable[str] | None,
    parent_seed: int,
    task: str,
    model: str,
    env_config: str | None = None,
    collect_evidence: bool = False,
    validation_receipt: Mapping | None = None,
) -> list[dict]:
    backend = str(backend)
    if backend not in SUPPORTED_BACKENDS:
        raise ManifestError(f"Unsupported adaptation backend: {backend!r}")
    try:
        validate_env_config(task, env_config)
    except ValueError as exc:
        raise ManifestError(str(exc)) from exc
    selected = None if idms is None else tuple(str(item) for item in idms)
    if selected is not None and (not selected or len(set(selected)) != len(selected)):
        raise ManifestError("IDM subset must be non-empty and unique.")
    base = pathlib.Path(manifest["_path"]).parent
    raw_matches = [
        raw
        for raw in manifest["entries"]
        if isinstance(raw, dict)
        and raw.get("backend") == backend
        and raw.get("parent_seed") == int(parent_seed)
        and raw.get("task") == task
        and raw.get("model") == model
        and raw.get("env_config") == env_config
        and (selected is None or raw.get("idm") in selected)
    ]
    receipt_evidence = None
    if validation_receipt is not None:
        receipt_evidence = _validation_receipt_evidence(
            validation_receipt,
            manifest=manifest,
            backend=backend,
            parent_seed=int(parent_seed),
            task=task,
            model=model,
            raw_matches=raw_matches,
        )
        collect_evidence = False
    matches = {}
    identities = set()
    # Bound parallel checkpoint and replay reads to limit shared-storage load.
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(4, max(1, len(raw_matches)))
    ) as executor:
        validated = list(
            executor.map(
                lambda raw: _validate_entry(
                    raw,
                    base,
                    collect_evidence=collect_evidence,
                    validation_evidence=(
                        receipt_evidence[
                            _receipt_identity(raw)
                        ]
                        if receipt_evidence is not None
                        else None
                    ),
                ),
                raw_matches,
            )
        )
    for entry in validated:
        identity = (entry["backend"], entry["idm"], entry["parent_seed"])
        if identity in identities:
            raise ManifestError(f"Duplicate parent cohort identity: {identity}")
        identities.add(identity)
        if (
            entry["backend"] == backend
            and entry["parent_seed"] == int(parent_seed)
            and entry["task"] == task
            and entry["model"] == model
            and (selected is None or entry["idm"] in selected)
        ):
            matches[entry["idm"]] = entry
    if selected is not None:
        missing = [idm for idm in selected if idm not in matches]
        if missing:
            raise ManifestError(
                f"Missing certified {backend} parents for IDMs: {missing}"
            )
        order = selected
    else:
        order = tuple(sorted(matches))
    if not order:
        raise ManifestError(
            f"No certified parents match backend={backend}, seed={parent_seed}, "
            f"task={task}, model={model}."
        )
    return [matches[idm] for idm in order]


def _validation_receipt_evidence(
    receipt: Mapping,
    *,
    manifest: Mapping,
    backend: str,
    parent_seed: int,
    task: str,
    model: str,
    raw_matches: Iterable[Mapping],
) -> dict:
    if not isinstance(receipt, Mapping):
        raise ManifestError("Validation receipt must be a mapping.")
    if receipt.get("schema_version") != VALIDATION_RECEIPT_SCHEMA_VERSION or (
        receipt.get("receipt_type") != VALIDATION_RECEIPT_TYPE
    ):
        raise ManifestError("Unsupported validation receipt.")
    if receipt.get("manifest_sha256") != manifest.get("_sha256"):
        raise ManifestError(
            "Validation receipt does not bind the selected cohort manifest."
        )
    records = receipt.get("entries")
    if not isinstance(records, list):
        raise ManifestError("Validation receipt entries must be a list.")
    evidence = {}
    fields = {
        "backend", "idm", "parent_seed", "task", "model", "run_uid",
        "evidence",
    }
    for record in records:
        if (
            not isinstance(record, Mapping)
            or not fields <= set(record)
            or set(record) - fields - {"env_config"}
        ):
            raise ManifestError("Malformed validation receipt entry.")
        key = _receipt_identity(record)
        if key in evidence:
            raise ManifestError(
                f"Duplicate validation receipt identity: {key}"
            )
        evidence[key] = record
    selected = {}
    for raw in raw_matches:
        key = _receipt_identity(raw)
        record = evidence.get(key)
        if record is None:
            raise ManifestError(
                f"Validation receipt is missing selected parent: {key}"
            )
        expected_identity = {
            "backend": backend,
            "idm": raw["idm"],
            "parent_seed": parent_seed,
            "task": task,
            "model": model,
            "run_uid": raw.get("run_uid"),
            "env_config": raw.get("env_config"),
        }
        mismatches = {
            name: (record.get(name), value)
            for name, value in expected_identity.items()
            if record.get(name) != value
        }
        if mismatches:
            raise ManifestError(
                f"Validation receipt parent identity mismatch: {mismatches}"
            )
        item = record["evidence"]
        if not isinstance(item, Mapping) or set(item) != {
            "manifest_entry_sha256", "checkpoint_files", "replay"
        }:
            raise ManifestError("Malformed parent validation evidence.")
        digest = item["manifest_entry_sha256"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ManifestError("Invalid validation receipt entry digest.")
        selected[key] = item
    return selected


def _receipt_identity(entry):
    # Include task and modality so a receipt can certify several Walker tasks
    # and both observation configurations for one backend/IDM/seed.
    return (
        entry["backend"], entry["idm"], entry["parent_seed"],
        entry["task"], entry["model"], entry.get("env_config"),
    )


def _validate_entry(
    raw,
    base: pathlib.Path,
    *,
    collect_evidence: bool = False,
    validation_evidence: Mapping | None = None,
) -> dict:
    required = {
        "backend",
        "idm",
        "parent_seed",
        "task",
        "model",
        "run_uid",
        "source_digest",
        "checkpoint_generation",
        "checkpoint_sha256",
        "compatibility_certificate",
        "replay_binding",
        "parent_step",
        "certified",
        "complete",
        "strict_task_blind_pretraining",
    }
    optional = {"parent_algorithm", "env_config", "parent_task", "parent_env_config", "protocol"}
    if not isinstance(raw, dict) or not required <= set(raw) or set(raw) - required - optional:
        missing = sorted(required - set(raw) if isinstance(raw, dict) else required)
        extra = sorted(set(raw) - required - optional) if isinstance(raw, dict) else []
        raise ManifestError(
            f"Malformed parent cohort entry: missing={missing}, extra={extra}"
        )
    entry = dict(raw)
    validate_protocol(entry.get("protocol", "fresh_heads"))
    if validation_evidence is not None:
        expected_entry_digest = validation_evidence.get(
            "manifest_entry_sha256"
        )
        if expected_entry_digest != _json_sha256(raw):
            raise ManifestError(
                "Validation receipt does not bind the selected manifest entry."
            )
    if entry["backend"] not in SUPPORTED_BACKENDS:
        raise ManifestError(f"Invalid parent backend: {entry['backend']!r}")
    for field in ("idm", "task", "model", "run_uid"):
        if not isinstance(entry[field], str) or not _SAFE_TEXT.fullmatch(entry[field]):
            raise ManifestError(f"Unsafe parent {field}: {entry[field]!r}")
    if not isinstance(entry["parent_seed"], int) or isinstance(
        entry["parent_seed"], bool
    ) or entry["parent_seed"] < 0:
        raise ManifestError("Parent seed must be a nonnegative integer.")
    try:
        validate_parent_step(entry["task"], entry["parent_step"])
        validate_env_config(entry["task"], entry.get("env_config"))
        _validate_source_binding(entry)
    except ValueError as exc:
        raise ManifestError(str(exc)) from exc
    parent_step = entry["parent_step"]
    if entry["certified"] is not True or entry["complete"] is not True:
        raise ManifestError(f"Parent is uncertified or incomplete: {entry['run_uid']}")
    parent_algorithm = entry.get("parent_algorithm", "p2e")
    if parent_algorithm not in ("p2e", "dreamer"):
        raise ManifestError(f"Invalid parent algorithm: {parent_algorithm!r}")
    if parent_algorithm == "dreamer" and (
        (entry["backend"] != "dreamerv3" and not is_walker_source_transfer(entry.get("env_config")))
        or entry["idm"] != "none"
        or entry["strict_task_blind_pretraining"] is not False
    ):
        raise ManifestError("Supervised control requires DreamerV3, IDM none, and strict blindness false.")
    if parent_algorithm == "p2e" and entry["strict_task_blind_pretraining"] is not True:
        raise ManifestError(
            "Adaptation parents must certify "
            "strict_task_blind_pretraining=true."
        )
    for field in ("source_digest", "checkpoint_sha256"):
        if not isinstance(entry[field], str) or not _SHA256.fullmatch(entry[field]):
            raise ManifestError(f"Invalid {field} for {entry['run_uid']}.")

    generation = _resolve_path(base, entry["checkpoint_generation"])
    _real_directory(generation, "parent checkpoint generation")
    generation_match = _GENERATION.fullmatch(generation.name)
    if generation_match is None:
        raise ManifestError(f"Unsafe checkpoint generation name: {generation.name}")
    name_step = generation_match.group("step")
    if name_step is not None and int(name_step) != parent_step:
        raise ManifestError(
            f"Checkpoint generation is not exact step {parent_step}: {generation}"
        )
    done = generation / "done"
    if file_sha256(done) != hashlib.sha256(b"").hexdigest():
        raise ManifestError(f"Invalid checkpoint completion marker: {done}")
    step = _safe_pickle_integer(generation / "step.pkl")
    if step != parent_step:
        raise ManifestError(f"Checkpoint scalar step mismatch: {step} != {parent_step}")
    checkpoint_evidence = [] if collect_evidence else None
    if validation_evidence is None:
        actual_checkpoint = checkpoint_sha256(
            generation, evidence=checkpoint_evidence
        )
    else:
        _validate_checkpoint_evidence(
            generation,
            validation_evidence.get("checkpoint_files"),
        )
        actual_checkpoint = entry["checkpoint_sha256"]
    if actual_checkpoint != entry["checkpoint_sha256"]:
        raise ManifestError(
            f"Checkpoint digest mismatch for {entry['run_uid']}: "
            f"{actual_checkpoint} != {entry['checkpoint_sha256']}"
        )

    run_manifest_path = generation.parent.parent / "run_manifest.json"
    try:
        run_manifest = json.loads(
            _read_regular_bytes(
                run_manifest_path,
                label="parent run manifest",
                max_bytes=_MAX_CERTIFICATE_BYTES,
            ).decode("utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(
            f"Parent is uncertified without a readable run manifest: {exc}"
        ) from exc
    identity = run_manifest.get("identity", {}) if isinstance(run_manifest, dict) else {}
    provenance = run_manifest.get("provenance", {}) if isinstance(run_manifest, dict) else {}
    expected_identity = {
        "RUN_ID": entry["run_uid"],
        "RUN_SEED": str(entry["parent_seed"]),
        "RUN_TASK": entry.get("parent_task", entry["task"]),
        "RUN_ALGORITHM": parent_algorithm,
        "RUN_IDM": entry["idm"],
        "RUN_AGENT": entry["backend"],
        "RUN_MODEL": entry["model"],
        "RUN_STRICT_TASK_BLIND_PRETRAINING": "true" if parent_algorithm == "p2e" else "false",
    }
    if "env_config" in entry:
        expected_identity["RUN_ENV_CONFIG"] = entry.get("parent_env_config", entry["env_config"])
    # Only the Dreamer control may omit the strict-blind identity field.
    checked_identity = dict(identity)
    if parent_algorithm == "dreamer":
        checked_identity.setdefault("RUN_STRICT_TASK_BLIND_PRETRAINING", "false")
    mismatches = {
        key: (identity.get(key), expected)
        for key, expected in expected_identity.items()
        if str(checked_identity.get(key)) != expected
    }
    if mismatches:
        raise ManifestError(
            f"Parent run/checkpoint identity mismatch for {entry['run_uid']}: "
            f"{mismatches}"
        )
    if provenance.get("source_id") != entry["source_digest"]:
        raise ManifestError(
            f"Parent source digest mismatch for {entry['run_uid']}: "
            f"{provenance.get('source_id')!r} != {entry['source_digest']!r}"
        )

    binding = entry["replay_binding"]
    binding_fields = {
        "saveable", "directory", "inventory_sha256", "capacity",
        "sequence_length",
    }
    if not isinstance(binding, dict) or set(binding) != binding_fields:
        raise ManifestError(f"Malformed replay binding for {entry['run_uid']}")
    if binding["saveable"] != "replay_train" or binding["capacity"] != 500_000:
        raise ManifestError(f"Parent replay binding is not the final 500k train replay.")
    if (
        not isinstance(binding["sequence_length"], int)
        or isinstance(binding["sequence_length"], bool)
        or binding["sequence_length"] < 1
    ):
        raise ManifestError("Parent replay binding has an invalid sequence length.")
    if not isinstance(binding["inventory_sha256"], str) or not _SHA256.fullmatch(
        binding["inventory_sha256"]
    ):
        raise ManifestError("Invalid replay inventory digest.")
    replay_directory = _resolve_path(base, binding["directory"])
    _real_directory(replay_directory, "parent replay directory")
    inventory = generation / "replay_train.pkl"
    actual_inventory = file_sha256(inventory)
    if actual_inventory != binding["inventory_sha256"]:
        raise ManifestError(
            f"Replay binding mismatch for {entry['run_uid']}: "
            f"{actual_inventory} != {binding['inventory_sha256']}"
        )
    certificate_reference = entry["compatibility_certificate"]
    if not isinstance(certificate_reference, dict) or set(
        certificate_reference
    ) != {"path", "sha256"}:
        raise ManifestError(
            f"Malformed compatibility certificate binding for {entry['run_uid']}"
        )
    if not isinstance(certificate_reference["sha256"], str) or not _SHA256.fullmatch(
        certificate_reference["sha256"]
    ):
        raise ManifestError("Invalid compatibility certificate digest.")
    certificate_path = _resolve_path(base, certificate_reference["path"])
    certificate = _load_compatibility_certificate(
        certificate_path,
        expected_sha256=certificate_reference["sha256"],
        entry=entry,
    )
    replay_evidence = [] if collect_evidence else None
    replay_result = _validate_replay_inventory(
        inventory,
        replay_directory,
        binding["sequence_length"],
        expected_schema=certificate["replay"]["normalized_target_schema"],
        reward_key=(f"label/reward/{entry['task'].removeprefix('dmc_')}"
                    if is_walker_source_transfer(entry.get("env_config")) else None),
        evidence=replay_evidence,
        trusted_evidence=(
            validation_evidence.get("replay")
            if validation_evidence is not None
            else None
        ),
    )
    entry["checkpoint_generation"] = str(generation)
    entry["compatibility_certificate"] = {
        **certificate_reference,
        "path": str(certificate_path),
    }
    entry["replay_binding"] = {
        **binding,
        "directory": str(replay_directory),
    }
    if collect_evidence:
        entry["_validation_evidence"] = {
            "manifest_entry_sha256": _json_sha256(raw),
            "checkpoint_files": checkpoint_evidence,
            "replay": {
                "chunks": replay_evidence,
                "usable_items": replay_result["usable_items"],
                "normalized_target_schema_sha256": certificate["replay"][
                    "normalized_target_schema_sha256"
                ],
            },
        }
    return entry


def _load_compatibility_certificate(
    path: pathlib.Path, *, expected_sha256: str, entry: Mapping
) -> dict:
    payload = _read_regular_bytes(
        path,
        label="compatibility certificate",
        max_bytes=_MAX_CERTIFICATE_BYTES,
    )
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ManifestError(
            f"Compatibility certificate digest mismatch for {entry['run_uid']}"
        )
    try:
        certificate = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_unique_json_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(
            f"Could not safely read compatibility certificate: {exc}"
        ) from exc
    required = {
        "schema_version",
        "certificate_type",
        "identity",
        "transfer",
        "replay",
    }
    if not isinstance(certificate, dict) or set(certificate) != required:
        raise ManifestError("Malformed compatibility certificate top level.")
    if certificate["schema_version"] != CERTIFICATE_SCHEMA_VERSION:
        raise ManifestError("Unsupported compatibility certificate schema.")
    if certificate["certificate_type"] != CERTIFICATE_TYPE:
        raise ManifestError("Unexpected compatibility certificate type.")

    identity = certificate["identity"]
    _validate_certificate_identity(identity)
    expected_identity = {
        "backend": entry["backend"],
        "idm": entry["idm"],
        "parent_seed": entry["parent_seed"],
        "task": entry["task"],
        "model": entry["model"],
        "run_uid": entry["run_uid"],
        "source_digest": entry["source_digest"],
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "replay_inventory_sha256": entry["replay_binding"]["inventory_sha256"],
        "parent_step": entry["parent_step"],
    }
    if "env_config" in entry:
        expected_identity["env_config"] = entry["env_config"]
    if is_walker_source_transfer(entry.get("env_config")):
        expected_identity.update({key: entry[key] for key in
                                  ("parent_task", "parent_env_config", "parent_algorithm")})
    if "protocol" in entry:
        expected_identity["protocol"] = entry["protocol"]
    if identity != expected_identity:
        mismatches = {
            key: (identity.get(key), value)
            for key, value in expected_identity.items()
            if identity.get(key) != value
        }
        raise ManifestError(
            f"Compatibility certificate identity mismatch for "
            f"{entry['run_uid']}: {mismatches}"
        )

    transfer = certificate["transfer"]
    transfer_fields = {
        "components",
        "source_parameters",
        "source_schema_sha256",
        "source_content_sha256",
        "expected_child_parameters",
        "expected_child_schema_sha256",
    }
    protocol = entry.get("protocol", "fresh_heads")
    if protocol == "pretrained_heads":
        transfer_fields.add("source_components")
    if not isinstance(transfer, dict) or set(transfer) != transfer_fields:
        raise ManifestError("Malformed compatibility certificate transfer section.")
    components = _validate_components(transfer["components"])
    expected_components = transfer_components(entry["backend"], protocol)
    if components != expected_components:
        raise ManifestError(
            "Compatibility certificate selects the wrong backend components: "
            f"{components} != {expected_components}"
        )
    if protocol == "pretrained_heads":
        expected_mapping = source_components(entry["backend"], protocol,
            task=entry["task"], env_config=entry.get("env_config"),
            parent_algorithm=entry.get("parent_algorithm", "p2e"), parent_task=entry.get("parent_task"))
        if transfer["source_components"] != expected_mapping:
            raise ManifestError("Compatibility certificate task-head source mapping mismatch.")
    source = _validate_parameter_records(
        transfer["source_parameters"], components
    )
    child = _validate_schema_records(
        transfer["expected_child_parameters"],
        label="expected child selected parameter schema",
    )
    source_schema = [
        {key: record[key] for key in ("key", "shape", "dtype")}
        for record in source
    ]
    if source_schema != child:
        raise ManifestError(
            "Certified source parameters do not exactly cover the expected child "
            "selected schema."
        )
    checks = {
        "source_schema_sha256": _json_sha256(source_schema),
        "source_content_sha256": _parameter_content_sha256(source),
        "expected_child_schema_sha256": _json_sha256(child),
    }
    for field, actual in checks.items():
        declared = transfer[field]
        if not isinstance(declared, str) or not _SHA256.fullmatch(declared):
            raise ManifestError(f"Invalid certificate {field}.")
        if declared != actual:
            raise ManifestError(
                f"Compatibility certificate {field} mismatch: "
                f"{declared} != {actual}"
            )

    replay = certificate["replay"]
    if not isinstance(replay, dict) or set(replay) != {
        "normalized_target_schema",
        "normalized_target_schema_sha256",
    }:
        raise ManifestError("Malformed compatibility certificate replay section.")
    replay_schema = _validate_schema_records(
        replay["normalized_target_schema"],
        label="normalized replay target schema",
    )
    _validate_required_replay_schema(replay_schema, env_config=entry.get("env_config"))
    replay_digest = replay["normalized_target_schema_sha256"]
    if not isinstance(replay_digest, str) or not _SHA256.fullmatch(replay_digest):
        raise ManifestError("Invalid normalized replay target schema fingerprint.")
    actual_replay_digest = _json_sha256(replay_schema)
    if replay_digest != actual_replay_digest:
        raise ManifestError(
            "Normalized replay target schema fingerprint mismatch: "
            f"{replay_digest} != {actual_replay_digest}"
        )
    return certificate


def _validate_certificate_identity(identity) -> dict:
    fields = {
        "backend",
        "idm",
        "parent_seed",
        "task",
        "model",
        "run_uid",
        "source_digest",
        "checkpoint_sha256",
        "replay_inventory_sha256",
        "parent_step",
    }
    if (
        not isinstance(identity, dict)
        or not fields <= set(identity)
        or set(identity) - fields - {"env_config", "parent_task", "parent_env_config", "parent_algorithm", "protocol"}
    ):
        raise ManifestError("Malformed compatibility certificate identity.")
    validate_protocol(identity.get("protocol", "fresh_heads"))
    if identity["backend"] not in SUPPORTED_BACKENDS:
        raise ManifestError("Invalid certificate backend.")
    for field in ("idm", "task", "model", "run_uid"):
        if not isinstance(identity[field], str) or not _SAFE_TEXT.fullmatch(
            identity[field]
        ):
            raise ManifestError(f"Unsafe certificate {field}.")
    if (
        not isinstance(identity["parent_seed"], int)
        or isinstance(identity["parent_seed"], bool)
        or identity["parent_seed"] < 0
    ):
        raise ManifestError("Invalid certificate parent seed.")
    try:
        validate_parent_step(identity["task"], identity["parent_step"])
        validate_env_config(identity["task"], identity.get("env_config"))
        _validate_source_binding(identity)
    except ValueError as exc:
        raise ManifestError(f"Compatibility certificate: {exc}") from exc
    for field in (
        "source_digest",
        "checkpoint_sha256",
        "replay_inventory_sha256",
    ):
        if not isinstance(identity[field], str) or not _SHA256.fullmatch(
            identity[field]
        ):
            raise ManifestError(f"Invalid certificate {field}.")
    return identity


def _validate_source_binding(value):
    """Bind stand-source multitask provenance separately from a child target."""
    if is_walker_source_transfer(value.get("env_config")):
        validate_walker_source(value["task"], value["env_config"],
                               value.get("parent_task"), value.get("parent_env_config"),
                               value["parent_step"])
        if value.get("parent_algorithm") not in ("dreamer", "p2e"):
            raise ValueError("Walker source transfer requires parent_algorithm dreamer or p2e.")
        if value["parent_algorithm"] == "dreamer" and value["idm"] != "none":
            raise ValueError("Supervised Walker source must use IDM none.")
        if value["parent_algorithm"] == "p2e" and value["idm"] == "none":
            raise ValueError("P2E Walker source must identify its intrinsic reward method.")
    elif "parent_task" in value or "parent_env_config" in value:
        raise ValueError("Explicit parent task/config fields require a Walker adaptation profile.")


def _validate_components(components) -> tuple[str, ...]:
    if not isinstance(components, (list, tuple)):
        raise ManifestError("Transfer components must be a list or tuple.")
    components = tuple(components)
    if (
        not components
        or len(set(components)) != len(components)
        or any(
            not isinstance(component, str)
            or not re.fullmatch(r"[A-Za-z0-9_]+", component)
            for component in components
        )
    ):
        raise ManifestError("Transfer components must be safe, non-empty, and unique.")
    return components


def _validate_schema_records(records, *, label: str) -> list[dict]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ManifestError(f"Validating {label} requires NumPy.") from exc
    if not isinstance(records, (list, tuple)) or not records:
        raise ManifestError(f"{label.capitalize()} must be a non-empty list.")
    normalized = []
    keys = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "key", "shape", "dtype"
        }:
            raise ManifestError(f"Malformed {label} record.")
        key = record["key"]
        if (
            not isinstance(key, str)
            or len(key) > 1024
            or not _PARAMETER_KEY.fullmatch(key)
            or key in keys
        ):
            raise ManifestError(f"Unsafe or duplicate {label} key: {key!r}")
        shape = record["shape"]
        if not isinstance(shape, (list, tuple)) or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in shape
        ):
            raise ManifestError(f"Invalid {label} shape for {key!r}.")
        try:
            dtype = np.dtype(record["dtype"])
        except (TypeError, ValueError) as exc:
            raise ManifestError(
                f"Invalid {label} dtype for {key!r}: {record['dtype']!r}"
            ) from exc
        if dtype.hasobject:
            raise ManifestError(f"Object dtype is forbidden in {label}.")
        keys.add(key)
        normalized.append(
            {"key": key, "shape": list(shape), "dtype": dtype.str}
        )
    if [record["key"] for record in normalized] != sorted(keys):
        raise ManifestError(f"{label.capitalize()} records are not canonically sorted.")
    return normalized


def _validate_parameter_records(records, components) -> list[dict]:
    if not isinstance(records, (list, tuple)) or not records:
        raise ManifestError("Selected source parameters must be a non-empty list.")
    schemas = []
    normalized = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "key", "shape", "dtype", "sha256"
        }:
            raise ManifestError("Malformed selected source parameter record.")
        schema = {key: record[key] for key in ("key", "shape", "dtype")}
        schemas.append(schema)
        digest = record["sha256"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ManifestError(
                f"Invalid parameter content hash for {record.get('key')!r}."
            )
        normalized.append({**schema, "sha256": digest})
    schemas = _validate_schema_records(
        schemas, label="selected source parameter schema"
    )
    for index, schema in enumerate(schemas):
        normalized[index].update(schema)
    namespaces = {record["key"].split("/", 1)[0] for record in normalized}
    components = set(_validate_components(components))
    if namespaces - components or components - namespaces - STATELESS_COMPONENTS:
        raise ManifestError(
            "Selected source parameter namespaces do not exactly cover the "
            f"backend: {sorted(namespaces)} != {sorted(components)}"
        )
    return normalized


def _validate_target_replay_schema(
    value: Mapping, *, backend: str, model: str, task: str,
    env_config: str | None = None,
) -> list[dict]:
    fields = {"schema_version", "backend", "model", "task", "fields"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ManifestError("Malformed target replay schema document.")
    if value["schema_version"] != 1:
        raise ManifestError("Unsupported target replay schema document.")
    expected = {"backend": backend, "model": model, "task": task}
    mismatches = {
        key: (value.get(key), target)
        for key, target in expected.items()
        if value.get(key) != target
    }
    if mismatches:
        raise ManifestError(f"Target replay schema identity mismatch: {mismatches}")
    records = _validate_schema_records(
        value["fields"], label="normalized replay target schema"
    )
    _validate_required_replay_schema(records, env_config=env_config)
    return records


def _validate_required_replay_schema(
    records: Iterable[Mapping], *, env_config: str | None = None
) -> None:
    records = list(records)
    by_key = {record["key"]: record for record in records}
    required = {
        "action",
        "reward",
        "is_first",
        "is_last",
        "is_terminal",
        "stepid",
        "dyn/deter",
        "dyn/stoch",
    }
    if env_config not in ("dmc_proprio", "dmc_walker_adaptation"):
        required.add("image")
    else:
        observations = {
            key for key in by_key
            if key not in required | {"skill", "skill_start", "consec"}
            and not key.startswith(("enc/", "dyn/", "dec/", "log/", "label/"))
        }
        if not observations or "image" in observations:
            raise ManifestError("Walker proprio replay requires non-image observations.")
    missing = required - set(by_key)
    if missing:
        raise ManifestError(
            f"Normalized replay target schema is missing fields: {sorted(missing)}"
        )
    forbidden = {"skill", "skill_start", "consec"} & set(by_key)
    if forbidden:
        raise ManifestError(
            f"Normalized replay target schema retains forbidden fields: "
            f"{sorted(forbidden)}"
        )
    scalar_dtypes = {
        "reward": "<f4",
        "is_first": "|b1",
        "is_last": "|b1",
        "is_terminal": "|b1",
    }
    for key, dtype in scalar_dtypes.items():
        if by_key[key]["shape"] != [] or by_key[key]["dtype"] != dtype:
            raise ManifestError(
                f"Normalized replay target schema has invalid {key}."
            )
    if by_key["stepid"] != {"key": "stepid", "shape": [20], "dtype": "|u1"}:
        raise ManifestError("Normalized replay target schema has invalid stepid.")


def _parameter_content_sha256(records: Iterable[Mapping]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["key"].encode("utf-8"))
        digest.update(bytes.fromhex(record["sha256"]))
    return digest.hexdigest()


def _json_sha256(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _unique_json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ManifestError(f"Duplicate JSON field: {key!r}")
        value[key] = item
    return value


def _trusted_agent_params(path: os.PathLike[str] | str) -> Mapping:
    path = pathlib.Path(path).expanduser().absolute()
    if path.is_dir():
        path = path / "agent.pkl"
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ManifestError(f"Expected a regular trusted agent pickle: {path}")
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
            if stream.read(1):
                raise ManifestError(f"Trusted agent pickle has trailing data: {path}")
    except (OSError, EOFError, pickle.UnpicklingError) as exc:
        raise ManifestError(f"Could not load trusted agent pickle {path}: {exc}") from exc
    params = payload.get("params") if isinstance(payload, Mapping) else None
    if not isinstance(params, Mapping) or not params:
        raise ManifestError(f"Trusted agent pickle has no params mapping: {path}")
    return params


def _validate_replay_inventory(
    inventory_path: pathlib.Path,
    replay_directory: pathlib.Path,
    sequence_length: int,
    *,
    expected_schema: Iterable[Mapping],
    reward_key: str | None = None,
    evidence: list[dict] | None = None,
    trusted_evidence: Mapping | None = None,
) -> dict:
    if evidence is not None and trusted_evidence is not None:
        raise ManifestError(
            "Replay validation cannot collect and consume evidence together."
        )
    expected_schema = _validate_schema_records(
        expected_schema, label="normalized replay target schema"
    )
    expected_by_key = {record["key"]: record for record in expected_schema}
    trusted_chunks = None
    if trusted_evidence is not None:
        if not isinstance(trusted_evidence, Mapping) or set(
            trusted_evidence
        ) != {
            "chunks",
            "usable_items",
            "normalized_target_schema_sha256",
        }:
            raise ManifestError("Malformed trusted replay validation evidence.")
        trusted_records = trusted_evidence["chunks"]
        if not isinstance(trusted_records, list):
            raise ManifestError("Trusted replay chunk evidence must be a list.")
        trusted_chunks = {}
        for record in trusted_records:
            if not isinstance(record, Mapping) or set(record) != {
                "filename", "size", "sha256", "stat"
            }:
                raise ManifestError("Malformed trusted replay chunk evidence.")
            name = record["filename"]
            if name in trusted_chunks:
                raise ManifestError(
                    f"Duplicate trusted replay chunk evidence: {name}"
                )
            trusted_chunks[name] = record
        expected_schema_digest = _json_sha256(expected_schema)
        if (
            trusted_evidence["normalized_target_schema_sha256"]
            != expected_schema_digest
        ):
            raise ManifestError(
                "Trusted replay evidence binds a different target schema."
            )
    try:
        payload = inventory_path.read_bytes()
        stream = io.BytesIO(payload)
        inventory = _RestrictedUnpickler(stream).load()
        if stream.read(1):
            raise ManifestError("Replay inventory contains trailing data.")
    except (OSError, EOFError, pickle.UnpicklingError) as exc:
        raise ManifestError(f"Could not safely read replay inventory: {exc}") from exc
    if (
        not isinstance(inventory, dict)
        or set(inventory) != {"format", "chunks"}
        or inventory.get("format") != _REPLAY_FORMAT
        or not isinstance(inventory.get("chunks"), list)
    ):
        raise ManifestError("Malformed replay checkpoint inventory.")

    entries = inventory["chunks"]
    names = []
    uuids = set()
    chunks = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "filename", "size", "sha256"
        }:
            raise ManifestError("Malformed replay checkpoint chunk entry.")
        name = entry["filename"]
        match = _CHUNK_NAME.fullmatch(name) if isinstance(name, str) else None
        if match is None or pathlib.PurePath(name).name != name:
            raise ManifestError(f"Unsafe replay checkpoint filename: {name!r}")
        if name in names:
            raise ManifestError(f"Duplicate replay checkpoint filename: {name}")
        if match["uuid"] in uuids:
            raise ManifestError(f"Duplicate replay chunk UUID: {match['uuid']}")
        size = entry["size"]
        digest = entry["sha256"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ManifestError(f"Invalid replay checkpoint size for {name}.")
        if size > _MAX_CHUNK_BYTES:
            raise ManifestError(f"Replay checkpoint chunk is too large: {name}.")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ManifestError(f"Invalid replay checkpoint SHA-256 for {name}.")

        filename = replay_directory / name
        try:
            before = filename.lstat()
        except FileNotFoundError as exc:
            raise ManifestError(f"Missing replay chunk: {filename}") from exc
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ManifestError(f"Unsafe replay chunk: {filename}")
        if trusted_chunks is None:
            actual_digest = file_sha256(filename)
            try:
                after = filename.lstat()
            except FileNotFoundError as exc:
                raise ManifestError(
                    f"Replay chunk disappeared during validation: {filename}"
                ) from exc
            if _stat_signature(before) != _stat_signature(after):
                raise ManifestError(
                    f"Replay chunk changed during validation: {filename}"
                )
            if after.st_size != size or actual_digest != digest:
                raise ManifestError(
                    f"Replay chunk binding mismatch for {name}: "
                    f"size={after.st_size}/{size}, "
                    f"sha256={actual_digest}/{digest}"
                )
            if evidence is not None:
                evidence.append(
                    {
                        "filename": name,
                        "size": size,
                        "sha256": digest,
                        "stat": _stat_record(after),
                    }
                )
        else:
            record = trusted_chunks.pop(name, None)
            if record is None:
                raise ManifestError(
                    f"Validation receipt is missing replay chunk: {name}"
                )
            if record["size"] != size or record["sha256"] != digest:
                raise ManifestError(
                    f"Validation receipt replay binding mismatch: {name}"
                )
            _validate_stat_record(record["stat"], before, label=str(filename))
            after = before
        names.append(name)
        uuids.add(match["uuid"])
        length = int(match["length"])
        if trusted_chunks is None:
            arrays = _load_replay_chunk(filename, name, length)
            if reward_key is not None:
                if __package__:
                    from .walker_replay import normalize_walker_replay_arrays
                else:
                    normalize_walker_replay_arrays = runpy.run_path(
                        str(pathlib.Path(__file__).with_name("walker_replay.py"))
                    )["normalize_walker_replay_arrays"]
                try:
                    arrays = normalize_walker_replay_arrays(arrays, reward_key)
                except (ValueError, TypeError) as exc:
                    raise ManifestError(f"Walker replay relabeling failed at {name}: {exc}") from exc
            _validate_replay_chunk_arrays(
                arrays, match["uuid"], length, name,
                expected_keys=expected_by_key,
            )
            schema = _normalized_replay_schema(arrays, expected_by_key, name)
            if schema != expected_schema:
                actual = {record["key"]: record for record in schema}
                missing = sorted(set(expected_by_key) - set(actual))
                unexpected = sorted(set(actual) - set(expected_by_key))
                incompatible = sorted(
                    key
                    for key in set(actual) & set(expected_by_key)
                    if actual[key] != expected_by_key[key]
                )
                raise ManifestError(
                    f"Replay target schema mismatch at {name}: "
                    f"missing={missing}, unexpected={unexpected}, "
                    f"incompatible={incompatible}"
                )
        chunks.append(
            (
                name[: -len(".npz.zst")] if name.endswith(".npz.zst")
                else name[: -len(".npz")],
                match["uuid"],
                match["succ"],
                length,
            )
        )

    if names != sorted(names):
        raise ManifestError("Replay checkpoint inventory is not canonically sorted.")
    usable_items = _usable_replay_items(chunks, sequence_length)
    if usable_items < 500_000:
        raise ManifestError(
            "Parent replay is incomplete: inventory provides only "
            f"{usable_items} usable items, fewer than 500000."
        )
    if trusted_chunks:
        raise ManifestError(
            "Validation receipt contains replay chunks outside the inventory: "
            f"{sorted(trusted_chunks)}"
        )
    if trusted_evidence is not None and (
        trusted_evidence["usable_items"] != usable_items
    ):
        raise ManifestError(
            "Validation receipt usable replay count mismatch: "
            f"{trusted_evidence['usable_items']} != {usable_items}"
        )
    return {"usable_items": usable_items}


def _load_replay_chunk(filename: pathlib.Path, name: str, length: int):
    try:
        import numpy as np
    except ImportError as exc:
        raise ManifestError(
            "Replay certification requires NumPy in the preflight Python."
        ) from exc
    payload = filename.read_bytes()
    if len(payload) > _MAX_CHUNK_BYTES:
        raise ManifestError(f"Replay chunk exceeds safety limit: {name}")
    if name.endswith(".npz.zst"):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise ManifestError(
                "Compressed replay certification requires zstandard in the "
                "preflight Python."
            ) from exc
        try:
            params = zstd.get_frame_parameters(payload)
            declared = params.content_size
            if declared in (zstd.CONTENTSIZE_UNKNOWN, zstd.CONTENTSIZE_ERROR):
                raise ManifestError(
                    f"Replay chunk has no bounded Zstandard size: {name}"
                )
            if declared > _MAX_CHUNK_BYTES or params.window_size > _MAX_CHUNK_BYTES:
                raise ManifestError(f"Replay chunk exceeds safety limit: {name}")
            payload = zstd.ZstdDecompressor(
                max_window_size=max(1024, _MAX_CHUNK_BYTES // 1024)
            ).decompress(
                payload,
                max_output_size=_MAX_CHUNK_BYTES,
                allow_extra_data=False,
            )
            if len(payload) != declared:
                raise ManifestError(
                    f"Replay chunk decompressed size mismatch: {name}"
                )
        except zstd.ZstdError as exc:
            raise ManifestError(f"Invalid Zstandard replay chunk {name}: {exc}") from exc
    archive_stream = io.BytesIO(payload)
    try:
        with zipfile.ZipFile(archive_stream) as archive:
            members = archive.infolist()
            if not members or any(
                member.file_size > _MAX_CHUNK_BYTES for member in members
            ):
                raise ManifestError(f"Unsafe replay NPZ members in {name}.")
            if name.endswith(".npz.zst") and any(
                member.compress_type != zipfile.ZIP_STORED for member in members
            ):
                raise ManifestError(
                    f"Zstandard replay chunk has compressed NPZ members: {name}"
                )
        archive_stream.seek(0)
        with np.load(archive_stream, allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
        raise ManifestError(f"Invalid replay chunk payload {name}: {exc}") from exc
    if not arrays or any(
        value.ndim < 1 or len(value) != length for value in arrays.values()
    ):
        shapes = {key: value.shape for key, value in arrays.items()}
        raise ManifestError(f"Replay chunk payload length mismatch {name}: {shapes}")
    return arrays


def _validate_replay_chunk_arrays(
    arrays, uuid: str, length: int, name: str, *, expected_keys=None
):
    import numpy as np

    required = {
        "action", "reward", "is_first", "is_last", "is_terminal",
        "stepid",
    }
    context_required = {"dyn/deter", "dyn/stoch"}
    missing = (required | context_required) - set(arrays)
    if missing:
        raise ManifestError(f"Replay chunk {name} is missing fields: {sorted(missing)}")
    if ("skill" in arrays) != ("skill_start" in arrays):
        raise ManifestError(f"Replay chunk {name} has an incomplete skill schema.")
    allowed = required | {"skill", "skill_start"} | set(expected_keys or ("image",))
    unexpected = {
        key for key in arrays
        if key not in allowed
        and not key.startswith(("enc/", "dyn/", "dec/"))
    }
    if unexpected:
        raise ManifestError(
            f"Replay chunk {name} has non-target fields: {sorted(unexpected)}"
        )
    if arrays["reward"].shape != (length,) or arrays["reward"].dtype != np.float32:
        raise ManifestError(
            f"Replay chunk {name} must contain scalar float32 true reward."
        )
    for key in ("is_first", "is_last", "is_terminal"):
        if arrays[key].shape != (length,) or arrays[key].dtype != np.bool_:
            raise ManifestError(f"Replay chunk {name} has invalid {key}.")
    if "skill_start" in arrays and (
        arrays["skill_start"].shape != (length,)
        or arrays["skill_start"].dtype != np.bool_
    ):
        raise ManifestError(f"Replay chunk {name} has invalid skill_start.")
    stepids = arrays["stepid"]
    if stepids.dtype != np.uint8 or stepids.shape != (length, 20):
        raise ManifestError(f"Replay chunk {name} has invalid stepid wire format.")
    if length >= 2 ** 32:
        raise ManifestError(f"Replay chunk index space is too large: {name}")
    expected = np.empty((length, 20), np.uint8)
    expected[:, :16] = np.frombuffer(_base62_bytes(uuid), np.uint8)
    expected[:, 16:] = np.arange(length, dtype=">u4").view(np.uint8).reshape(
        length, 4
    )
    if not np.array_equal(stepids, expected):
        raise ManifestError(f"Replay chunk {name} has invalid step identity.")


def _normalized_replay_schema(arrays, expected_by_key, name: str) -> list[dict]:
    """Mirror replay-context compaction at schema level without copying arrays."""
    import numpy as np

    records = []
    for key, value in arrays.items():
        if key in {"skill", "skill_start"}:
            continue
        shape = tuple(value.shape[1:])
        dtype = value.dtype
        if key == "dyn/deter":
            target = expected_by_key.get(key)
            target_dtype = np.dtype(target["dtype"]) if target else None
            if target_dtype not in (np.dtype(np.uint16), np.dtype(np.uint32)):
                raise ManifestError(
                    "Normalized replay target dyn/deter must use uint16 or uint32."
                )
            if not (
                np.issubdtype(dtype, np.floating)
                or dtype in (np.dtype(np.uint16), np.dtype(np.uint32))
            ):
                raise ManifestError(
                    f"Replay chunk {name} has unsupported dyn/deter dtype: {dtype}"
                )
            dtype = target_dtype
        elif key == "dyn/stoch":
            if value.ndim >= 3:
                shape = tuple(value.shape[1:-1])
                dtype = np.dtype(np.uint8 if value.shape[-1] <= 256 else np.uint16)
            elif dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
                dtype = np.dtype(np.uint16)
        records.append(
            {"key": key, "shape": list(shape), "dtype": np.dtype(dtype).str}
        )
    records.sort(key=lambda record: record["key"])
    return _validate_schema_records(records, label="normalized replay chunk schema")


def _usable_replay_items(chunks, sequence_length: int) -> int:
    lengths = {uuid: length for _, uuid, _, length in chunks}
    if len(lengths) != len(chunks):
        raise ManifestError("Replay inventory contains duplicate chunk UUIDs.")
    future = {}
    total = 0
    for _, uuid, succ, length in reversed(sorted(chunks)):
        if succ in lengths and succ not in future:
            raise ManifestError(
                f"Replay successor graph is not chronological: {uuid} -> {succ}"
            )
        successor_rows = future.get(succ, 0)
        future[uuid] = length + successor_rows
        total += max(0, min(length, length + 1 - sequence_length + successor_rows))
    return total


def _base62_bytes(value: str) -> bytes:
    integer = 0
    for char in value:
        try:
            digit = _BASE62.index(char)
        except ValueError as exc:
            raise ManifestError(f"Invalid replay chunk UUID: {value!r}") from exc
        integer = integer * 62 + digit
    try:
        return integer.to_bytes(16, "big")
    except OverflowError as exc:
        raise ManifestError(f"Replay chunk UUID is out of range: {value!r}") from exc


def _safe_pickle_integer(path: pathlib.Path) -> int:
    if path.lstat().st_size > 1024:
        raise ManifestError(f"Checkpoint scalar is unexpectedly large: {path}")
    data = path.read_bytes()
    integer = None
    allowed = {
        "PROTO", "FRAME", "BININT", "BININT1", "BININT2", "INT", "LONG",
        "LONG1", "LONG4", "STOP",
    }
    try:
        operations = list(pickletools.genops(data))
    except ValueError as exc:
        raise ManifestError(f"Malformed checkpoint step: {exc}") from exc
    if not operations or operations[-1][0].name != "STOP":
        raise ManifestError("Checkpoint step has no terminal STOP opcode.")
    for operation, argument, _ in operations:
        if operation.name not in allowed:
            raise ManifestError(f"Unsafe checkpoint step opcode: {operation.name}")
        if operation.name in {
            "BININT", "BININT1", "BININT2", "INT", "LONG", "LONG1", "LONG4"
        }:
            if integer is not None or not isinstance(argument, int) or isinstance(
                argument, bool
            ):
                raise ManifestError("Checkpoint step is not one integer.")
            integer = argument
    if integer is None or integer < 0:
        raise ManifestError("Checkpoint step is not a nonnegative integer.")
    return integer


def _resolve_path(base: pathlib.Path, value) -> pathlib.Path:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"Missing parent path: {value!r}")
    path = pathlib.Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.absolute()


def _validate_checkpoint_evidence(generation: pathlib.Path, records) -> None:
    if not isinstance(records, list) or not records:
        raise ManifestError("Checkpoint validation evidence must be non-empty.")
    expected = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"name", "stat"}:
            raise ManifestError("Malformed checkpoint validation evidence.")
        name = record["name"]
        if (
            not isinstance(name, str)
            or pathlib.PurePath(name).name != name
            or name in expected
        ):
            raise ManifestError(
                f"Unsafe or duplicate checkpoint evidence name: {name!r}"
            )
        expected[name] = record["stat"]
    actual_names = sorted(path.name for path in generation.iterdir())
    if actual_names != sorted(expected):
        raise ManifestError(
            "Checkpoint generation membership changed after certification: "
            f"{actual_names} != {sorted(expected)}"
        )
    for name in actual_names:
        child = generation / name
        info = child.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ManifestError(f"Unsafe checkpoint entry: {child}")
        _validate_stat_record(expected[name], info, label=str(child))


def _real_directory(path: pathlib.Path, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ManifestError(f"Missing {label}: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ManifestError(f"Unsafe {label}: {path}")


def _stat_record(info) -> dict:
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": int(info.st_mode),
        "size": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def _validate_stat_record(record, info, *, label: str) -> None:
    fields = {"device", "inode", "mode", "size", "mtime_ns", "ctime_ns"}
    if (
        not isinstance(record, Mapping)
        or set(record) != fields
        or any(
            not isinstance(record[field], int)
            or isinstance(record[field], bool)
            or record[field] < 0
            for field in fields
        )
    ):
        raise ManifestError(f"Malformed file identity evidence for {label}.")
    actual = _stat_record(info)
    # Device numbers vary across network filesystem mounts. Match the inode,
    # mode, size, mtime, and ctime across hosts; retain st_dev for diagnostics.
    stable_fields = fields - {"device"}
    expected_stable = {field: record[field] for field in stable_fields}
    actual_stable = {field: actual[field] for field in stable_fields}
    if expected_stable != actual_stable:
        raise ManifestError(
            f"Certified file identity changed for {label}: "
            f"{actual_stable} != {expected_stable}"
        )


def _stat_signature(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )
