"""Paper launchers preview jobs without requiring a scheduler or submitting work."""

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("idm", ("simple_curiosity", "icm"))
@pytest.mark.parametrize("launcher,algorithm", [
    ("launch_dmc_p2e.sh", "p2e"),
    ("launch_dmc_p2e.sh", "dreamer"),
    ("launch_dmc_blind.sh", "p2e"),
    ("launch_craftax_p2e.sh", "p2e"),
    ("launch_craftax_p2e.sh", "dreamer"),
    ("launch_dmc_walker_multitask.sh", "p2e"),
])
def test_dry_run_does_not_submit_jobs(tmp_path, launcher, algorithm, idm):
    marker = tmp_path / "submitted"
    scheduler = tmp_path / "sbatch"
    scheduler.write_text('#!/bin/sh\ntouch "$SUBMISSION_MARKER"\nexit 99\n')
    scheduler.chmod(0o755)
    env = dict(os.environ)
    env.update(
        PATH=f"{tmp_path}:{env['PATH']}", SUBMISSION_MARKER=str(marker),
        ALGORITHM=algorithm, IDM_CONFIG="none" if algorithm == "dreamer" else idm,
        BACKEND_AGENT="dreamerv3", SEEDS="0",
        PARTITION="test_gpu", QOS="test_qos", WANDB_MODE="offline",
        WORK_LOG_ROOT=str(tmp_path / "runs"),
        SSRL_SOURCE_BUNDLE_ROOT=str(tmp_path / "source_bundles"),
    )
    result = subprocess.run(
        ["bash", str(ROOT / "experiments" / launcher), "--dry-run"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "sbatch" in result.stdout
    assert f"ALGORITHM={algorithm}" in result.stdout
    assert not marker.exists()


@pytest.mark.parametrize("launcher", [
    "launch_craftax_adaptation.sh",
    "launch_craftax_supervised_adaptation.sh",
    "launch_dmc_walker_adaptation.sh",
])
def test_adaptation_help_does_not_require_parent_checkpoints(launcher):
    result = subprocess.run(
        ["bash", str(ROOT / "experiments" / launcher), "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "PARENT_COHORT_MANIFEST" in result.stdout
