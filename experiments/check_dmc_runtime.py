#!/usr/bin/env python3
"""Read-only login-node check; never initialize CUDA or create a W&B run."""

from __future__ import annotations

import importlib.metadata
import netrc
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlsplit


PYTHON_MINOR = (3, 11)
REQUIREMENTS_PATH = Path(__file__).resolve().parents[1] / "requirements.txt"


def read_requirements(path):
    """Read the installable runtime pins, rejecting ambiguous requirements."""
    try:
        from packaging.requirements import InvalidRequirement, Requirement
        from packaging.utils import canonicalize_name
    except ImportError as exc:
        raise ValueError(
            "The packaging package is missing; install the runtime with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    try:
        lines = Path(path).read_text().splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read runtime requirements {path}: {exc}") from exc
    requirements = {}
    for number, line in enumerate(lines, start=1):
        line = re.split(r"\s+#", line, maxsplit=1)[0].strip()
        if not line or line.startswith("#"):
            continue
        location = f"{path}:{number}"
        try:
            requirement = Requirement(line)
        except InvalidRequirement as exc:
            raise ValueError(f"{location}: invalid runtime requirement: {line}") from exc
        specifiers = list(requirement.specifier)
        if (
            requirement.url is not None
            or len(specifiers) != 1
            or specifiers[0].operator != "=="
            or "*" in specifiers[0].version
        ):
            raise ValueError(f"{location}: runtime requirements must use one exact == pin: {line}")
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        name = canonicalize_name(requirement.name)
        if name in requirements:
            raise ValueError(f"{location}: duplicate runtime requirement: {name}")
        requirements[name] = requirement
    if not requirements:
        raise ValueError(f"{path}: no active runtime requirements found")
    return requirements


def check_runtime(requirements):
    errors = []
    if sys.version_info[:2] != PYTHON_MINOR:
        errors.append(
            f"Python {PYTHON_MINOR[0]}.{PYTHON_MINOR[1]} required; "
            f"found {sys.version.split()[0]}"
        )
    for name, requirement in requirements.items():
        expected = next(iter(requirement.specifier)).version
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing runtime package: {name}=={expected}")
        else:
            if not requirement.specifier.contains(actual, prereleases=True):
                errors.append(f"Runtime mismatch: {name}=={actual}; expected {expected}")
    return errors


def check_dependencies(requirements):
    """Check installed metadata, including extras that pip check omits."""
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    errors = []
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True,
    )
    if result.returncode:
        details = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
        errors.append(f"Installed dependency check failed: {details or 'pip check failed'}")

    # Follow extras transitively: jax[cuda12] requests the CUDA plugin's own
    # [with-cuda] extra, which in turn supplies the NVIDIA runtime libraries.
    pending = [requirement for requirement in requirements.values() if requirement.extras]
    checked = set()
    while pending:
        parent = pending.pop()
        key = (canonicalize_name(parent.name), tuple(sorted(parent.extras)))
        if key in checked:
            continue
        checked.add(key)
        try:
            dependencies = importlib.metadata.requires(parent.name) or []
        except importlib.metadata.PackageNotFoundError:
            # The direct version check or the requesting dependency reports it.
            continue
        label = f"{key[0]}[{','.join(key[1])}]"
        for raw in dependencies:
            try:
                dependency = Requirement(raw)
            except InvalidRequirement:
                errors.append(f"Invalid installed dependency metadata for {label}: {raw}")
                continue
            if dependency.marker is not None and not any(
                dependency.marker.evaluate({"extra": extra})
                for extra in ("", *parent.extras)
            ):
                continue
            name = canonicalize_name(dependency.name)
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                errors.append(
                    f"Missing runtime dependency: {name}{dependency.specifier} (required by {label})"
                )
                continue
            if not dependency.specifier.contains(actual, prereleases=True):
                errors.append(
                    f"Runtime dependency mismatch: {name}=={actual}; "
                    f"{label} requires {name}{dependency.specifier}"
                )
            if dependency.extras:
                pending.append(dependency)
    return list(dict.fromkeys(errors))


def check_wandb(environ):
    mode = environ.get("WANDB_MODE", "offline")
    if mode not in {"online", "offline", "disabled"}:
        return ["WANDB_MODE must be online, offline, or disabled"]
    if mode != "online" or environ.get("WANDB_API_KEY"):
        return []
    host = urlsplit(environ.get("WANDB_BASE_URL", "https://api.wandb.ai")).hostname
    try:
        auth = netrc.netrc(str(Path(environ["HOME"]) / ".netrc")).authenticators(host)
    except (OSError, netrc.NetrcParseError):
        auth = None
    if auth and auth[2]:
        return []
    return [
        "No W&B credentials found. Activate the training environment and run "
        "`wandb login` for your account, or export WANDB_API_KEY."
    ]


def main():
    print(f"Checking DMC runtime: {sys.executable}")
    try:
        requirements = read_requirements(REQUIREMENTS_PATH)
    except ValueError as exc:
        errors = [str(exc)]
    else:
        errors = check_runtime(requirements) + check_dependencies(requirements)
    errors += check_wandb(os.environ)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print("See README.md for installation and runtime instructions.", file=sys.stderr)
        return 1
    print("requirements.txt versions, installed dependencies, and local W&B credentials checked.")
    print("GPU execution, team authorization, and shared-storage access are checked on the worker.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
