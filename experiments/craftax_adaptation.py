#!/usr/bin/env python3
"""Preflight helpers for the extended Craftax task-adaptation study."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.adaptation_manifest import (
    ManifestError, PROTOCOLS,
    checkpoint_sha256,
    create_validation_receipt,
    create_compatibility_certificate,
    file_sha256,
    load_manifest,
    load_validation_receipt,
    select_and_validate,
    write_compatibility_certificate,
    write_validation_receipt,
)


def _idms(value: str):
    return tuple(item for item in value.replace(",", ":").split(":") if item)


def _parents(arguments) -> list[dict]:
    manifest = load_manifest(
        arguments.manifest, expected_sha256=arguments.manifest_sha256
    )
    idms = _idms(arguments.idms) if arguments.idms else None
    if arguments.index is not None:
        selected = set(idms) if idms is not None else None
        matches = {}
        for raw in manifest["entries"]:
            if not isinstance(raw, dict):
                continue
            if (
                raw.get("backend") == arguments.backend
                and raw.get("parent_seed") == arguments.parent_seed
                and raw.get("task") == arguments.task
                and raw.get("model") == arguments.model
                and (selected is None or raw.get("idm") in selected)
            ):
                idm = raw.get("idm")
                if not isinstance(idm, str) or idm in matches:
                    raise ManifestError(
                        f"Malformed or duplicate indexed parent IDM: {idm!r}"
                    )
                matches[idm] = raw
        order = idms if idms is not None else tuple(sorted(matches))
        missing = [idm for idm in order if idm not in matches]
        if missing:
            raise ManifestError(
                f"Missing indexed {arguments.backend} parents for IDMs: {missing}"
            )
        if arguments.index < 0 or arguments.index >= len(order):
            raise ManifestError(
                f"parent index {arguments.index} is outside 0-{len(order) - 1}"
            )
        idms = (order[arguments.index],)
    receipt = None
    if arguments.receipt or arguments.receipt_sha256:
        if not arguments.receipt or not arguments.receipt_sha256:
            raise ManifestError(
                "Validation receipt path and SHA-256 must be provided together."
            )
        receipt = load_validation_receipt(
            arguments.receipt,
            expected_sha256=arguments.receipt_sha256,
        )
    parents = select_and_validate(
        manifest,
        backend=arguments.backend,
        idms=idms,
        parent_seed=arguments.parent_seed,
        task=arguments.task,
        model=arguments.model,
        validation_receipt=receipt,
    )
    protocol = getattr(arguments, "protocol", "fresh_heads")
    if any(entry.get("protocol", "fresh_heads") != protocol for entry in parents):
        raise ManifestError(f"Selected certificates do not match adaptation protocol {protocol!r}.")
    expected = getattr(arguments, "parent_algorithm", "p2e")
    if any(entry.get("parent_algorithm", "p2e") != expected for entry in parents):
        raise ManifestError(f"Selected parents do not match requested algorithm {expected!r}.")
    return parents


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-parents")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--manifest-sha256", required=True)
    validate.add_argument("--backend", required=True)
    validate.add_argument("--protocol", choices=PROTOCOLS, default="fresh_heads")
    validate.add_argument("--parent-algorithm", choices=("p2e", "dreamer"), default="p2e")
    validate.add_argument("--idms", default="")
    validate.add_argument("--parent-seed", type=int, required=True)
    validate.add_argument("--task", default="craftax_classic_pixels")
    validate.add_argument("--model", required=True)
    validate.add_argument("--index", type=int)
    validate.add_argument("--receipt", default="")
    validate.add_argument("--receipt-sha256", default="")
    checkpoint_digest = subparsers.add_parser("checkpoint-sha256")
    checkpoint_digest.add_argument("generation")
    file_digest = subparsers.add_parser("file-sha256")
    file_digest.add_argument("path")
    certificate = subparsers.add_parser(
        "create-certificate",
        description=(
            "Explicit trusted operation: unpickle a certified parent and a "
            "fresh ordinary-Dreamer reference child, then emit non-executable "
            "compatibility metadata. Never use this on untrusted pickle files."
        ),
    )
    certificate.add_argument("--source-generation", required=True)
    certificate.add_argument("--target-agent-pickle", required=True)
    certificate.add_argument("--target-replay-schema", required=True)
    certificate.add_argument("--backend", required=True)
    certificate.add_argument("--protocol", choices=PROTOCOLS, default="fresh_heads")
    certificate.add_argument("--idm", required=True)
    certificate.add_argument("--parent-seed", type=int, required=True)
    certificate.add_argument("--task", default="craftax_classic_pixels")
    certificate.add_argument("--model", required=True)
    certificate.add_argument("--run-uid", required=True)
    certificate.add_argument("--source-digest", required=True)
    certificate.add_argument("--output", required=True)
    receipt = subparsers.add_parser(
        "certify-cohort",
        description=(
            "Run the expensive full checkpoint/replay validation once and "
            "emit a SHA-pinned stat-bound receipt for fast launches."
        ),
    )
    receipt.add_argument("--manifest", required=True)
    receipt.add_argument("--manifest-sha256", required=True)
    receipt.add_argument("--backend", required=True)
    receipt.add_argument("--idms", default="")
    receipt.add_argument("--parent-seed", type=int, required=True)
    receipt.add_argument("--task", default="craftax_classic_pixels")
    receipt.add_argument("--model", required=True)
    receipt.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "checkpoint-sha256":
        print(checkpoint_sha256(arguments.generation))
        return 0
    if arguments.command == "file-sha256":
        print(file_sha256(arguments.path))
        return 0
    if arguments.command == "create-certificate":
        generation = pathlib.Path(arguments.source_generation).expanduser().absolute()
        try:
            target_schema = json.loads(
                pathlib.Path(arguments.target_replay_schema).read_text(
                    encoding="utf-8"
                )
            )
            certificate_value = create_compatibility_certificate(
                source_agent_pickle=generation / "agent.pkl",
                protocol=arguments.protocol,
                target_agent_pickle=arguments.target_agent_pickle,
                target_replay_schema=target_schema,
                backend=arguments.backend,
                idm=arguments.idm,
                parent_seed=arguments.parent_seed,
                task=arguments.task,
                model=arguments.model,
                run_uid=arguments.run_uid,
                source_digest=arguments.source_digest,
                checkpoint_sha256_value=checkpoint_sha256(generation),
                replay_inventory_sha256=file_sha256(
                    generation / "replay_train.pkl"
                ),
            )
            digest = write_compatibility_certificate(
                arguments.output, certificate_value
            )
        except (
            ManifestError,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            parser.error(str(exc))
        json.dump(
            {
                "path": str(pathlib.Path(arguments.output).expanduser().absolute()),
                "sha256": digest,
            },
            sys.stdout,
            sort_keys=True,
        )
        sys.stdout.write("\n")
        return 0
    if arguments.command == "certify-cohort":
        try:
            manifest = load_manifest(
                arguments.manifest,
                expected_sha256=arguments.manifest_sha256,
            )
            value = create_validation_receipt(
                manifest,
                backend=arguments.backend,
                idms=_idms(arguments.idms) if arguments.idms else None,
                parent_seed=arguments.parent_seed,
                task=arguments.task,
                model=arguments.model,
            )
            digest = write_validation_receipt(arguments.output, value)
        except (ManifestError, OSError, ValueError) as exc:
            parser.error(str(exc))
        json.dump(
            {
                "path": str(pathlib.Path(arguments.output).expanduser().absolute()),
                "sha256": digest,
                "count": len(value["entries"]),
            },
            sys.stdout,
            sort_keys=True,
        )
        sys.stdout.write("\n")
        return 0
    try:
        parents = _parents(arguments)
    except (ManifestError, OSError) as exc:
        parser.error(str(exc))
    if arguments.index is not None:
        value = parents[0]
    else:
        value = {
            "schema_version": 1,
            "count": len(parents),
            "parents": parents,
        }
    json.dump(value, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
