#!/usr/bin/env python3
"""Certify an explicitly selected, trusted supervised Dreamer adaptation parent.

Reads agent pickles only after checking the publication/reference hashes.
Outputs an immutable manifest and a full checkpoint/replay validation receipt.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import adaptation_manifest as api
from experiments import run_storage


def prepare(parent, generation_name, reference, output, *, protocol="fresh_heads"):
    api.validate_protocol(protocol)
    parent = pathlib.Path(parent).absolute()
    reference = pathlib.Path(reference).absolute()
    output = pathlib.Path(output).absolute()
    generation = parent / "checkpoint.pkl" / generation_name
    api._real_directory(parent, "supervised parent")
    api._real_directory(generation, "supervised checkpoint")
    manifest = json.loads((parent / "run_manifest.json").read_text())
    identity = manifest["identity"]
    expected = {"RUN_ALGORITHM": "dreamer", "RUN_AGENT": "dreamerv3",
                "RUN_IDM": "none", "RUN_MODEL": "size200M",
                "RUN_TASK": "craftax_classic_pixels", "RUN_SEED": "0"}
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError("Parent must be supervised DreamerV3 size200M Craftax seed 0.")
    if identity.get("RUN_STRICT_TASK_BLIND_PRETRAINING", "false") != "false":
        raise ValueError("Supervised control must not claim strict-blind pretraining.")
    if api._safe_pickle_integer(generation / "step.pkl") != api.PARENT_STEP:
        raise ValueError("Parent checkpoint must be exactly 5,010,000 steps.")
    marker = json.loads((parent / ".sync_complete.json").read_text())
    if (marker.get("checkpoint") != generation_name
            or marker.get("run_uid") != identity["RUN_ID"]
            or marker.get("marker_kind") != "final-publish"):
        raise ValueError("Parent must have a committed final publication for this checkpoint.")
    print("Verifying parent publication integrity", flush=True)
    verified = run_storage._catalog_sync_complete(parent)
    if not verified.get("valid"):
        raise ValueError(f"Parent publication failed verification: {verified.get('errors')}")

    metadata = json.loads((reference / "reference_metadata.json").read_text())
    expected_reference = {"kind": "ordinary-dreamerv3-initialization",
                          "backend": "dreamerv3", "model": "size200M",
                          "task": "craftax_classic_pixels", "environment_steps": 0,
                          "checkpoint_loaded": False, "replay_imported": False,
                          "adaptation_enabled": False, "idm_type": "none",
                          "expl_behavior": "greedy"}
    if (any(metadata.get(k) != v for k, v in expected_reference.items())
            or metadata.get("counters") != {"actions": 0, "batches": 0, "updates": 0}):
        raise ValueError("Reference must be a fresh ordinary DreamerV3 agent.")
    for name in ("agent.pkl", "replay_schema.json"):
        if api.file_sha256(reference / name) != metadata["artifacts"][name]["sha256"]:
            raise ValueError(f"Reference artifact digest mismatch: {name}")
    output.mkdir(parents=True, exist_ok=True)
    selection = dict(backend="dreamerv3", idms=("none",), parent_seed=0,
                     task="craftax_classic_pixels", model="size200M")
    checkpoint_hash = api.checkpoint_sha256(generation)
    replay_hash = api.file_sha256(generation / "replay_train.pkl")
    print(f"Certifying {protocol} transfer against fresh reference", flush=True)
    certificate = api.create_compatibility_certificate(
        source_agent_pickle=generation / "agent.pkl", protocol=protocol,
        target_agent_pickle=reference / "agent.pkl",
        target_replay_schema=json.loads((reference / "replay_schema.json").read_text()),
        backend="dreamerv3", idm="none", parent_seed=0,
        task="craftax_classic_pixels", model="size200M", run_uid=identity["RUN_ID"],
        source_digest=manifest["provenance"]["source_id"],
        checkpoint_sha256_value=checkpoint_hash, replay_inventory_sha256=replay_hash)
    cert_hash = api.write_compatibility_certificate(output / "certificate.json", certificate)
    entry = dict(backend="dreamerv3", parent_algorithm="dreamer", idm="none", parent_seed=0,
                 task="craftax_classic_pixels", model="size200M", run_uid=identity["RUN_ID"],
                 source_digest=manifest["provenance"]["source_id"],
                 checkpoint_generation=str(generation), checkpoint_sha256=checkpoint_hash,
                 compatibility_certificate={"path": "certificate.json", "sha256": cert_hash},
                 replay_binding={"saveable": "replay_train", "directory": str(parent / "replay"),
                                 "inventory_sha256": replay_hash, "capacity": 500_000,
                                 "sequence_length": 65},
                 parent_step=api.PARENT_STEP, certified=True, complete=True,
                 strict_task_blind_pretraining=False)
    if protocol != "fresh_heads":
        entry["protocol"] = protocol
    manifest_hash = api._write_readonly_json(
        output / "parents.json", {"schema_version": 2, "entries": [entry]}, label="supervised parent manifest")
    loaded = api.load_manifest(output / "parents.json", expected_sha256=manifest_hash)
    print("Certifying complete final 500k replay", flush=True)
    receipt = api.create_validation_receipt(loaded, **selection)
    receipt_hash = api.write_validation_receipt(output / "validation.json", receipt)
    receipt = api.load_validation_receipt(output / "validation.json", expected_sha256=receipt_hash)
    api.select_and_validate(loaded, **selection, validation_receipt=receipt)
    result = {"protocol": protocol, "parent_run_uid": identity["RUN_ID"], "parent_algorithm": "dreamer",
              "parent_step": api.PARENT_STEP, "strict_task_blind_pretraining": False,
              "manifest": str(output / "parents.json"), "manifest_sha256": manifest_hash,
              "receipt": str(output / "validation.json"), "receipt_sha256": receipt_hash,
              "reference_metadata_sha256": api.file_sha256(reference / "reference_metadata.json"),
              "full_validation_passed": True}
    api._write_readonly_json(output / "readiness.json", result, label="supervised preparation readiness")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", required=True, type=pathlib.Path)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--reference", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--protocol", choices=api.PROTOCOLS, default="fresh_heads")
    args = parser.parse_args()
    if not api._GENERATION.fullmatch(args.generation):
        parser.error("Unsafe checkpoint generation name")
    print(json.dumps(prepare(args.parent_run, args.generation, args.reference, args.output, protocol=args.protocol), indent=2))


if __name__ == "__main__":
    main()
