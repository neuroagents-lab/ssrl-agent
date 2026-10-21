#!/usr/bin/env python3
"""Validate stand-source Walker parents for separate fresh-head target children."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import adaptation_manifest as api


def _selection(value, *, allowed=None):
    items = tuple(value.split(":"))
    if not all(items) or len(set(items)) != len(items):
        raise api.ManifestError("Selections must be nonempty, unique, colon-delimited values.")
    if allowed is not None and set(items) - set(allowed):
        raise api.ManifestError(f"Unsupported selection {value!r}; choose from {allowed}.")
    return items


def _selections(args, manifest):
    tasks = _selection(args.tasks, allowed=("stand", "walk", "run"))
    seeds_value = getattr(args, "seeds", None)
    if seeds_value is None:
        seeds_value = str(getattr(args, "parent_seed", 0))
    raw_seeds = _selection(seeds_value)
    if any(not seed.isdecimal() or str(int(seed)) != seed for seed in raw_seeds):
        raise api.ManifestError("SEEDS must contain unique nonnegative integers, e.g. 0:1:2.")
    seeds = tuple(int(seed) for seed in raw_seeds)
    algorithm = getattr(args, "parent_algorithm", "all")
    candidate_entries = [entry for entry in manifest["entries"]
        if isinstance(entry, dict) and isinstance(entry.get("idm"), str)
        and entry.get("backend") == args.backend
        and entry.get("parent_seed") in seeds
        and entry.get("task") in {f"dmc_walker_{task}" for task in tasks}
        and entry.get("model") == args.model
        and entry.get("env_config") == args.env_config
        and (algorithm == "all" or entry.get("parent_algorithm", "p2e") == algorithm)]
    requested = _selection(args.idms) if args.idms else None
    if requested is None:
        # Require the complete requested method/task/seed grid.
        requested = tuple(sorted({entry["idm"] for entry in candidate_entries}))
    selections = []
    for seed in seeds:
        for task_name in tasks:
            task = f"dmc_walker_{task_name}"
            matches = [entry for entry in candidate_entries
                       if entry.get("parent_seed") == seed and entry.get("task") == task]
            idms = [entry["idm"] for entry in matches]
            if len(set(idms)) != len(idms):
                raise api.ManifestError(f"Duplicate parent IDM for {task}, seed {seed}.")
            missing = set(requested) - set(idms)
            if not requested or missing:
                raise api.ManifestError(
                    f"Missing certified parents: backend={args.backend}, "
                    f"env_config={args.env_config}, task={task}, "
                    f"seed={seed}, model={args.model}, IDMs={sorted(missing)}."
                )
            selections.extend(dict(backend=args.backend, idms=(idm,),
                                   parent_seed=seed, task=task,
                                   model=args.model, env_config=args.env_config)
                              for idm in requested)
    return selections


def validate(args):
    manifest = api.load_manifest(args.manifest, expected_sha256=args.manifest_sha256)
    selections = _selections(args, manifest)
    if args.index is not None:
        if not 0 <= args.index < len(selections):
            raise api.ManifestError(f"Parent index {args.index} outside 0-{len(selections) - 1}.")
        selections = [selections[args.index]]
    receipt = api.load_validation_receipt(args.receipt, expected_sha256=args.receipt_sha256)
    parents = []
    for selection in selections:
        parents.extend(api.select_and_validate(manifest, **selection,
                                               validation_receipt=receipt))
    protocol = getattr(args, "protocol", "fresh_heads")
    if any(entry.get("protocol", "fresh_heads") != protocol for entry in parents):
        raise api.ManifestError(f"Selected certificates do not match adaptation protocol {protocol!r}.")
    expected = getattr(args, "parent_algorithm", "all")
    if expected != "all" and any(entry.get("parent_algorithm", "p2e") != expected for entry in parents):
        raise api.ManifestError(f"Walker parent algorithm does not match {expected}.")
    return parents[0] if args.index is not None else {
        "schema_version": 1, "count": len(parents), "parents": parents,
    }


def certify(args):
    manifest = api.load_manifest(args.manifest, expected_sha256=args.manifest_sha256)
    selections = _selections(args, manifest)
    receipt = None
    for selection in selections:
        part = api.create_validation_receipt(manifest, **selection)
        if receipt is None:
            receipt = part
        else:
            receipt["entries"].extend(part["entries"])
    digest = api.write_validation_receipt(args.output, receipt)
    return {"path": str(pathlib.Path(args.output).expanduser().absolute()),
            "sha256": digest, "count": len(receipt["entries"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-parents", "certify-cohort"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", required=True)
        command.add_argument("--manifest-sha256", required=True)
        command.add_argument("--backend", choices=api.SUPPORTED_BACKENDS, required=True)
        command.add_argument("--parent-algorithm", choices=("all", "p2e", "dreamer"), default="all")
        command.add_argument("--seeds", default="0:1:2")
        command.add_argument("--tasks", default="run")
        command.add_argument("--env-config", choices=("dmc_walker_adaptation", "dmc_walker_adaptation_vision"),
                             default="dmc_walker_adaptation")
        command.add_argument("--model", default="size12M")
        command.add_argument("--idms", default="")
        if name == "validate-parents":
            command.add_argument("--protocol", choices=api.PROTOCOLS, default="fresh_heads")
            command.add_argument("--receipt", required=True)
            command.add_argument("--receipt-sha256", required=True)
            command.add_argument("--index", type=int)
        else:
            command.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = validate(args) if args.command == "validate-parents" else certify(args)
    except (api.ManifestError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
