#!/usr/bin/env python3
"""Certify an explicitly selected, trusted strict-blind Craftax P2E parent.

The independent ordinary reference is produced by
create_craftax_adaptation_reference.py. Publication/reference integrity is
checked before agent pickles are loaded. The immutable outputs include a
representation-only compatibility certificate and full final-500k replay
validation receipt. A completed preparation is published atomically.
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
from agent.adaptation_protocol import validate_parent_step
from experiments import run_storage
from experiments.prepare_dmc_walker_adaptation import _json, _reference


def _parent_identity(manifest):
    identity = manifest.get("identity", {})
    if (not isinstance(identity, dict)
            or identity.get("RUN_ALGORITHM") != "p2e"
            or identity.get("RUN_AGENT") not in api.SUPPORTED_BACKENDS
            or identity.get("RUN_TASK") != "craftax_classic_pixels"
            or identity.get("RUN_MODEL") != "size200M"
            or identity.get("RUN_STRICT_TASK_BLIND_PRETRAINING") != "true"
            or identity.get("RUN_IDM") == "none"):
        raise ValueError("Parent must be strict-blind P2E size200M Craftax Classic pixels.")
    if identity.get("RUN_ENV_CONFIG", "craftax_classic") != "craftax_classic":
        raise ValueError("Parent must use the Craftax environment config.")
    for name in ("RUN_ID", "RUN_IDM"):
        if not isinstance(identity.get(name), str) or not api._SAFE_TEXT.fullmatch(identity[name]):
            raise ValueError(f"Missing or unsafe parent identity: {name}")
    seed = identity.get("RUN_SEED")
    if not isinstance(seed, str) or not seed.isdecimal():
        raise ValueError("Parent RUN_SEED must be a nonnegative integer string.")
    provenance = manifest.get("provenance", {})
    source_digest = provenance.get("source_id") if isinstance(provenance, dict) else None
    if not isinstance(source_digest, str) or not api._SHA256.fullmatch(source_digest):
        raise ValueError("Parent must have a valid source provenance digest.")
    return dict(backend=identity["RUN_AGENT"], idm=identity["RUN_IDM"],
                parent_seed=int(seed), task=identity["RUN_TASK"], model=identity["RUN_MODEL"],
                run_uid=identity["RUN_ID"], source_digest=source_digest)


def prepare(parent, generation_name, reference, output, *, protocol="fresh_heads"):
    api.validate_protocol(protocol)
    if not isinstance(generation_name, str) or not api._GENERATION.fullmatch(generation_name):
        raise ValueError("Unsafe checkpoint generation name")
    parent, reference, output = (
        pathlib.Path(path).expanduser().absolute() for path in (parent, reference, output))
    api._real_directory(parent, "Craftax parent")
    api._real_directory(parent / "checkpoint.pkl", "parent checkpoint directory")
    generation = parent / "checkpoint.pkl" / generation_name
    api._real_directory(generation, "parent checkpoint generation")
    manifest_path = parent / "run_manifest.json"
    identity = _parent_identity(_json(manifest_path))
    parent_step = api._safe_pickle_integer(generation / "step.pkl")
    validate_parent_step(identity["task"], parent_step)
    parent_algorithm = "p2e"
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
    schema, reference_binding = _reference(reference, {**identity, "env_config": "craftax_classic"}, generation)
    selection = {key: identity[key] for key in ("backend", "parent_seed", "task", "model")}
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
            {"schema_version": 2, "entries": [entry]}, label="Craftax parent manifest")
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
        api._write_readonly_json(stage / "readiness.json", result, label="Craftax preparation readiness")
        # Grant the inherited group read and traverse access before publication.
        stage.chmod(0o750)
        os.rename(stage, output)
        descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", required=True, type=pathlib.Path)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--reference", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--protocol", choices=api.PROTOCOLS, default="fresh_heads")
    args = parser.parse_args(argv)
    try:
        result = prepare(args.parent_run, args.generation, args.reference, args.output, protocol=args.protocol)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
