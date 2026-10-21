"""Shared configuration and CPU fixtures for the experiment test suite."""

import os
from pathlib import Path
import sys

# Tests use small CPU workloads and never contact experiment tracking services.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("WANDB_MODE", "disabled")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "agent")]

import hydra
import pytest


@pytest.fixture
def compose_config():
    def compose(*overrides):
        with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
            return hydra.compose(
                config_name="configs",
                overrides=["device=cpu", "wandb.enabled=false", *overrides],
            )
    return compose
