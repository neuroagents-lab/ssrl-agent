#!/usr/bin/env python3
"""Certify one Walker multitask parent for a selected fresh-head child target.

The reference directory must contain an independently initialized ordinary
Dreamer agent.pkl, replay_schema.json, and reference_metadata.json. Metadata
binds both artifact SHA-256 hashes, backend/model/task/env_config, zero training
counters, and checkpoint_loaded/replay_imported/adaptation_enabled=false.
Only this explicit preparation command reads the trusted agent pickles;
launcher preflight uses the resulting immutable JSON certificates.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import adaptation_manifest as api
from agent.adaptation_protocol import (
    WALKER_ADAPTATION_ENV_CONFIGS, WALKER_ENV_CONFIGS, WALKER_PARENT_ENV_CONFIGS, WALKER_TASKS,
    is_walker_source_transfer, validate_parent_step, validate_walker_source,
)
from experiments import run_storage


def _json(path):
    value = json.loads(api._read_regular_bytes(
        path, label=path.name, max_bytes=api._MAX_CERTIFICATE_BYTES
    ), object_pairs_hook=api._unique_json_object)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _parent_identity(manifest, env_config, task=None):
    identity = manifest.get("identity", {})
    if (not isinstance(identity, dict)
            or identity.get("RUN_ALGORITHM") not in ("dreamer", "p2e")
            or identity.get("RUN_AGENT") not in api.SUPPORTED_BACKENDS
            or identity.get("RUN_TASK") not in WALKER_TASKS):
        raise ValueError("Parent must be a Dreamer or strict-blind p2e Walker checkpoint.")
    for name in ("RUN_ID", "RUN_MODEL", "RUN_IDM"):
        if not isinstance(identity.get(name), str) or not api._SAFE_TEXT.fullmatch(identity[name]):
            raise ValueError(f"Missing or unsafe parent identity: {name}")
    algorithm = identity["RUN_ALGORITHM"]
    expected_blind = "true" if algorithm == "p2e" else "false"
    if identity.get("RUN_STRICT_TASK_BLIND_PRETRAINING") != expected_blind:
        raise ValueError("Parent must identify strict-blind p2e or supervised Dreamer pretraining.")
    if algorithm == "p2e" and identity["RUN_IDM"] == "none":
        raise ValueError("Blind p2e parent must identify its intrinsic reward method.")
    if algorithm == "dreamer" and identity["RUN_IDM"] != "none":
        raise ValueError("Supervised Dreamer parent must use IDM none.")
    seed = identity.get("RUN_SEED")
    if not isinstance(seed, str) or not seed.isdecimal():
        raise ValueError("Parent RUN_SEED must be a nonnegative integer string.")
    recorded = identity.get("RUN_ENV_CONFIG")
    if task in ("stand", "walk", "run"):
        task = f"dmc_walker_{task}"
    source_fields = {}
    child_configs = {parent: child for child, parent in WALKER_PARENT_ENV_CONFIGS.items()}
    if recorded in child_configs:
        if task not in WALKER_TASKS:
            raise ValueError("Multitask parent preparation requires --task stand, walk, or run.")
        inferred = child_configs[recorded]
        if env_config and env_config != inferred:
            raise ValueError("--env-config conflicts with the parent RUN_ENV_CONFIG modality.")
        env_config = inferred
        source_fields = dict(parent_task=identity["RUN_TASK"], parent_env_config=recorded,
                             parent_algorithm=algorithm)
    else:
        if recorded not in ("dmc_proprio", "dmc_vision"):
            raise ValueError("Parent must identify a supported Walker RUN_ENV_CONFIG.")
        if algorithm != "p2e":
            raise ValueError("Legacy parent preparation requires strict-blind p2e.")
        if env_config and recorded != env_config:
            raise ValueError("--env-config conflicts with the parent RUN_ENV_CONFIG identity.")
        if task and task != identity["RUN_TASK"]:
            raise ValueError("Cross-task adaptation requires a Walker multitask parent with reward labels.")
        env_config, task = recorded, identity["RUN_TASK"]
    provenance = manifest.get("provenance", {})
    source_digest = provenance.get("source_id") if isinstance(provenance, dict) else None
    if not isinstance(source_digest, str) or not api._SHA256.fullmatch(source_digest):
        raise ValueError("Parent must have a valid source provenance digest.")
    return dict(backend=identity["RUN_AGENT"], idm=identity["RUN_IDM"],
                parent_seed=int(seed), task=task,
                model=identity["RUN_MODEL"], env_config=env_config,
                run_uid=identity["RUN_ID"], source_digest=source_digest, **source_fields)


def _reference(reference, parent_identity, generation):
    api._real_directory(reference, "fresh reference directory")
    metadata_path = reference / "reference_metadata.json"
    metadata_stat = api._stat_record(metadata_path.lstat())
    metadata_hash = api.file_sha256(metadata_path)
    metadata = _json(metadata_path)
    backend = parent_identity["backend"]
    kind_backend = "ne-dreamer" if backend == "ne_dreamer" else backend
    expected = {"schema_version": 1, "kind": f"ordinary-{kind_backend}-initialization",
                "environment_steps": 0, "idm_type": "none", "expl_behavior": "greedy",
                **{key: parent_identity[key] for key in ("backend", "model", "task", "env_config")}}
    if (any(metadata.get(key) != value for key, value in expected.items())
            or metadata.get("counters") != {"actions": 0, "batches": 0, "updates": 0}
            or any(metadata.get(key) is not False for key in
                   ("checkpoint_loaded", "replay_imported", "adaptation_enabled"))):
        raise ValueError("Reference must be a fresh ordinary zero-step agent matching backend, model, task and env_config.")
    if not api._SHA256.fullmatch(str(metadata.get("source_id", ""))):
        raise ValueError("Reference metadata must identify its source digest.")
    if (reference / "agent.pkl").samefile(generation / "agent.pkl"):
        raise ValueError("Reference agent must be initialized independently of the parent.")
    declared_artifacts = metadata.get("artifacts")
    if not isinstance(declared_artifacts, dict):
        raise ValueError("Reference metadata must bind agent and replay schema artifacts.")
    artifacts = {}
    for name in ("agent.pkl", "replay_schema.json"):
        path = reference / name
        declared = declared_artifacts.get(name, {})
        if not isinstance(declared, dict):
            raise ValueError(f"Malformed reference artifact metadata: {name}")
        before = api._stat_record(path.lstat())
        digest = api.file_sha256(path)
        api._validate_stat_record(before, path.lstat(), label=str(path))
        if digest != declared.get("sha256"):
            raise ValueError(f"Reference artifact digest mismatch: {name}")
        if "size" in declared and declared["size"] != path.stat().st_size:
            raise ValueError(f"Reference artifact size mismatch: {name}")
        artifacts[name] = {"path": str(path), "sha256": digest,
                           "stat": api._stat_record(path.lstat())}
    schema = _json(reference / "replay_schema.json")
    api._validate_target_replay_schema(schema, **{
        key: parent_identity[key] for key in ("backend", "model", "task", "env_config")})
    api._validate_stat_record(metadata_stat, metadata_path.lstat(), label=str(metadata_path))
    return schema, {"path": str(metadata_path), "sha256": metadata_hash,
                    "source_id": metadata["source_id"], "artifacts": artifacts}


def prepare(parent, generation_name, reference, output, *, env_config=None, task=None, protocol="fresh_heads"):
    api.validate_protocol(protocol)
    if not isinstance(generation_name, str) or not api._GENERATION.fullmatch(generation_name):
        raise ValueError("Unsafe checkpoint generation name")
    parent, reference, output = (
        pathlib.Path(path).expanduser().absolute() for path in (parent, reference, output))
    api._real_directory(parent, "Walker parent")
    api._real_directory(parent / "checkpoint.pkl", "parent checkpoint directory")
    generation = parent / "checkpoint.pkl" / generation_name
    api._real_directory(generation, "parent checkpoint generation")
    manifest_path = parent / "run_manifest.json"
    identity = _parent_identity(_json(manifest_path), env_config, task)
    parent_step = api._safe_pickle_integer(generation / "step.pkl")
    validate_parent_step(identity["task"], parent_step)
    if is_walker_source_transfer(identity["env_config"]):
        validate_walker_source(identity["task"], identity["env_config"],
            identity["parent_task"], identity["parent_env_config"], parent_step)
    parent_algorithm = identity.get("parent_algorithm", "p2e")
    marker_path = parent / ".sync_complete.json"
    marker = _json(marker_path)
    if (marker.get("checkpoint") != generation_name
            or marker.get("run_uid") != identity["run_uid"]
            or marker.get("marker_kind") != "final-publish"):
        raise ValueError("Parent must have a committed final publication for this exact checkpoint.")
    checkpoint_stats = {str(path): api._stat_record(path.lstat()) for path in generation.iterdir()}
    print("Verifying parent publication integrity", flush=True)
    verified = run_storage._catalog_sync_complete(parent)
    if not verified.get("valid"):
        raise ValueError(f"Parent publication failed verification: {verified.get('errors')}")
    pinned_parent = {str(path): api.file_sha256(path) for path in (manifest_path, marker_path)}
    schema, reference_binding = _reference(reference, identity, generation)
    selection = {key: identity[key] for key in ("backend", "parent_seed", "task", "model", "env_config")}
    selection["idms"] = (identity["idm"],)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve incomplete stages for inspection and publish readiness only
    # after full replay validation. Concurrent preparation never replaces data.
    lock_path = output.parent / f".{output.name}.prepare.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists() or output.is_symlink():
            raise ValueError(f"Refusing to replace existing preparation: {output}")
        stage = pathlib.Path(tempfile.mkdtemp(prefix=f".{output.name}.prepare-", dir=output.parent))
        checkpoint_hash = api.checkpoint_sha256(generation)
        replay_hash = api.file_sha256(generation / "replay_train.pkl")
        print(f"Certifying {protocol} transfer and complete final 500k replay", flush=True)
        certificate = api.create_compatibility_certificate(
            source_agent_pickle=generation / "agent.pkl", target_agent_pickle=reference / "agent.pkl",
            target_replay_schema=schema, **identity, parent_step=parent_step, protocol=protocol,
            checkpoint_sha256_value=checkpoint_hash, replay_inventory_sha256=replay_hash)
        cert_hash = api.write_compatibility_certificate(stage / "certificate.json", certificate)
        entry = {**identity, **({"protocol": protocol} if protocol != "fresh_heads" else {}), "parent_algorithm": parent_algorithm, "parent_step": parent_step,
                 "checkpoint_generation": str(generation), "checkpoint_sha256": checkpoint_hash,
                 "compatibility_certificate": {"path": "certificate.json", "sha256": cert_hash},
                 "replay_binding": {"saveable": "replay_train", "directory": str(parent / "replay"),
                                    "inventory_sha256": replay_hash, "capacity": 500_000,
                                    "sequence_length": 65},
                 "certified": True, "complete": True,
                 "strict_task_blind_pretraining": parent_algorithm == "p2e"}
        manifest_hash = api._write_readonly_json(stage / "parents.json",
            {"schema_version": 2, "entries": [entry]}, label="Walker parent manifest")
        loaded = api.load_manifest(stage / "parents.json", expected_sha256=manifest_hash)
        receipt_hash = api.write_validation_receipt(stage / "validation.json",
            api.create_validation_receipt(loaded, **selection))
        receipt = api.load_validation_receipt(stage / "validation.json", expected_sha256=receipt_hash)
        api.select_and_validate(loaded, **selection, validation_receipt=receipt)
        for path, digest in pinned_parent.items():
            if api.file_sha256(path) != digest:
                raise ValueError("Parent publication changed during certification.")
        if {str(path) for path in generation.iterdir()} != set(checkpoint_stats):
            raise ValueError("Parent checkpoint membership changed during certification.")
        for path, before in checkpoint_stats.items():
            api._validate_stat_record(before, pathlib.Path(path).lstat(), label=path)
        if api.file_sha256(reference_binding["path"]) != reference_binding["sha256"]:
            raise ValueError("Reference metadata changed during certification.")
        for artifact in reference_binding["artifacts"].values():
            api._validate_stat_record(artifact["stat"], pathlib.Path(artifact["path"]).lstat(),
                                      label=artifact["path"])
        result = {**identity, "protocol": protocol, "parent_run_uid": identity["run_uid"], "parent_algorithm": parent_algorithm,
                  "parent_step": parent_step, "strict_task_blind_pretraining": parent_algorithm == "p2e",
                  "manifest": str(output / "parents.json"), "manifest_sha256": manifest_hash,
                  "receipt": str(output / "validation.json"), "receipt_sha256": receipt_hash,
                  "certificate": str(output / "certificate.json"), "certificate_sha256": cert_hash,
                  "reference": reference_binding, "parent_publication": pinned_parent,
                  "full_validation_passed": True}
        api._write_readonly_json(stage / "readiness.json", result, label="Walker preparation readiness")
        os.rename(stage, output)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", required=True, type=pathlib.Path)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--reference", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--task", choices=("stand", "walk", "run"), required=True,
                        help="Child reward/behavior target; prepare each desired target separately.")
    parser.add_argument("--env-config", choices=sorted(WALKER_ENV_CONFIGS | WALKER_ADAPTATION_ENV_CONFIGS),
                        help="Optional child-profile assertion; modality is inferred from the parent.")
    parser.add_argument("--protocol", choices=api.PROTOCOLS, default="fresh_heads")
    args = parser.parse_args(argv)
    try:
        result = prepare(args.parent_run, args.generation, args.reference, args.output,
                         env_config=args.env_config, task=args.task, protocol=args.protocol)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
