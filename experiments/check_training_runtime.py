#!/usr/bin/env python3
"""Exercise the installed runtime with a small, isolated DMC P2E training run."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import pickle
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
STEPS = 64


def training_command(logdir: Path, device: str, run_id: str) -> list[str]:
    # Exercise MJX, JIT, learner updates, and evaluation in a short isolated run.
    return [
        sys.executable, "-u", str(ROOT / "train.py"),
        "envs=dmc_proprio", "algo=p2e", "idm=simple_curiosity", "model=size1M",
        "model.strict_task_blind_pretraining=true",
        "envs.task=dmc_cartpole_balance", "envs.backend=mjx",
        "envs.env_num=2", f"envs.steps={STEPS}", "envs.time_limit=16",
        "envs.eval_episode_num=1", "trainer.eval_env_num=1",
        "envs.eval_seed_bank=[1000000]", "trainer.eval_action_selections=[greedy]",
        "batch_size=2", "batch_length=4", "envs.train_ratio=1",
        "envs.eval_every=32", "envs.update_log_every=16",
        "trainer.save_every=32", "buffer.max_size=256", "buffer.chunksize=16",
        "envs.video_policy_log=false", "envs.video_pred_log=false",
        "model.idm_hidden_dim=32", "model.imag_horizon=2", "model.warmup=0",
        "jax_compilation_cache=false", "jax_preallocate=false",
        "wandb.enabled=true", "wandb.mode=offline",
        "wandb.project=ssrl-runtime-check", "wandb.entity=null",
        "wandb.name=dmc-p2e-runtime-check", "wandb.resume=never",
        f"wandb.id={run_id}", f"device={device}",
        # JSON string quoting is understood by Hydra and protects spaces and
        # special characters here; the command is executed without a shell.
        f"logdir={json.dumps(str(logdir))}",
    ]


def training_environment(logdir: Path, device: str) -> dict[str, str]:
    # A check invoked from inside a worker must never inherit that run's W&B
    # identity or commit a checkpoint to its durable high-water marker.
    env = {
        name: value for name, value in os.environ.items()
        if not name.startswith(("WANDB_", "RUN_", "SCRATCH_SYNC_"))
    }
    env.update(
        WANDB_MODE="offline",
        WANDB_DIR=str(logdir),
        WANDB_CACHE_DIR=str(logdir / "wandb-cache"),
        WANDB_CONFIG_DIR=str(logdir / "wandb-config"),
        WANDB_DATA_DIR=str(logdir / "wandb-data"),
        WANDB_DISABLE_CODE="true",
        PYTHONUNBUFFERED="1",
        # The observation-space probe creates MJX arrays before agent setup.
        # Select the backend and allocation policy before JAX is imported.
        JAX_PLATFORMS="cpu" if device == "cpu" else "cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    return env


def stop_process(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def run_training(command: list[str], logdir: Path, timeout: int, device: str) -> None:
    transcript = logdir / "training.log"
    print(f"Training output: {transcript}", flush=True)
    print(shlex.join(command), flush=True)
    started = time.monotonic()
    with transcript.open("x") as output:
        process = subprocess.Popen(
            command, cwd=ROOT, env=training_environment(logdir, device),
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError(f"Training exceeded {timeout} seconds")
                try:
                    returncode = process.wait(timeout=min(30, remaining))
                    break
                except subprocess.TimeoutExpired:
                    elapsed = int(time.monotonic() - started)
                    print(f"Training check running ({elapsed}s); output: {transcript}", flush=True)
            if returncode:
                raise RuntimeError(f"Training exited with status {returncode}")
        except BaseException:
            stop_process(process)
            raise


def read_records(path: Path) -> list[dict]:
    with path.open() as source:
        records = [json.loads(line) for line in source if line.strip()]
    if not records or any(not isinstance(record, dict) for record in records):
        raise RuntimeError(f"Missing or invalid records in {path}")
    return records


def finite(value) -> bool:
    return isinstance(value, (float, int)) and math.isfinite(value)


def verify_outputs(logdir: Path, run_id: str) -> dict:
    metrics = read_records(logdir / "metrics.jsonl")
    update_key = "train/p2e_opt/updates"
    updates = [record[update_key] for record in metrics if update_key in record]
    if not updates or not all(finite(value) for value in updates) or max(updates) <= 0:
        raise RuntimeError("No completed P2E learner updates in metrics.jsonl")
    losses = [
        value for record in metrics for key, value in record.items()
        if key.startswith("train/loss/") or key == "train/p2e_opt/loss"
    ]
    if not losses or not all(finite(value) for value in losses):
        raise RuntimeError("Missing or nonfinite learner losses in metrics.jsonl")
    final_step = max(int(record["step"]) for record in metrics)
    if final_step < STEPS:
        raise RuntimeError(f"Training stopped at step {final_step}, expected at least {STEPS}")

    evaluations = read_records(logdir / "eval_episodes.jsonl")
    final_eval = [record for record in evaluations if record["checkpoint_step"] == final_step]
    if not final_eval or not all(
        finite(record.get("return")) and record.get("length", 0) > 0
        and record.get("task") == "dmc_cartpole_balance"
        for record in final_eval
    ):
        raise RuntimeError("No finite completed evaluation at the final learner step")

    checkpoint_root = logdir / "checkpoint.pkl"
    generation = (checkpoint_root / "latest").read_text().strip()
    if Path(generation).name != generation or generation in {".", ".."}:
        raise RuntimeError("Invalid checkpoint latest pointer")
    checkpoint = checkpoint_root / generation
    if not (checkpoint / "done").is_file():
        raise RuntimeError("Latest checkpoint has no completed-generation marker")
    for filename in ("step.pkl", "replay_train.pkl"):
        if not (checkpoint / filename).is_file() or not (checkpoint / filename).stat().st_size:
            raise RuntimeError(f"Missing checkpoint payload: {filename}")
    # Read the saved scalar step; checkpoint names do not always encode progress.
    try:
        with (checkpoint / "step.pkl").open("rb") as source:
            checkpoint_step = pickle.load(source)
    except (EOFError, pickle.UnpicklingError) as exc:
        raise RuntimeError("Unreadable checkpoint step payload") from exc
    if not isinstance(checkpoint_step, int) or checkpoint_step != final_step:
        raise RuntimeError("Latest checkpoint does not match the final learner step")
    agent_payloads = list(checkpoint.glob("agent*.pkl"))
    if not agent_payloads or not all(path.stat().st_size for path in agent_payloads):
        raise RuntimeError("Missing agent checkpoint payloads")
    replay_chunks = [
        path for path in (logdir / "replay").glob("*.npz*")
        if path.is_file() and path.stat().st_size
    ]
    if not replay_chunks:
        raise RuntimeError("No persisted training replay chunks")

    receipt = json.loads((logdir / "wandb_run.json").read_text())
    if receipt.get("wandb_id") != run_id:
        raise RuntimeError("W&B receipt does not identify this isolated check")
    wandb_logs = [
        path for path in (logdir / "wandb").glob(f"offline-run-*-{run_id}/run-{run_id}.wandb")
        if path.is_file() and path.stat().st_size
    ]
    if not wandb_logs:
        raise RuntimeError("No persisted offline W&B run log")
    return dict(
        status="passed", environment_steps=final_step,
        max_logged_update_counter=max(updates), final_evaluation_episodes=len(final_eval),
        checkpoint=str(checkpoint), replay_chunks=len(replay_chunks),
        wandb_mode="offline", wandb_run_id=run_id,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a 64-step DMC/MJX strict-blind P2E simple curiosity check using the current "
            "Python environment. Checks training, evaluation, checkpoint/replay "
            "and offline W&B output. Run on an allocated NVIDIA GPU by default."
        ),
        epilog=(
            "Output is retained for inspection. This check submits no jobs, "
            "uploads no W&B run, and does not verify shared-storage publication "
            "or team authorization."
        ),
    )
    parser.add_argument("--device", default="cuda:0", help="cuda:N or cpu (default: cuda:0)")
    parser.add_argument("--logdir", type=Path, help="new output directory; must not already exist")
    parser.add_argument("--timeout", type=int, default=900, metavar="SECONDS", help="training time limit (default: 900)")
    args = parser.parse_args()
    if not re.fullmatch(r"cuda:\d+|cpu", args.device):
        parser.error("--device must be cuda:N or cpu")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.logdir is None:
        logdir = Path(tempfile.mkdtemp(prefix="ssrl-runtime-check-"))
    else:
        logdir = args.logdir.expanduser().absolute()
        if "${" in str(logdir):
            parser.error("--logdir cannot contain Hydra interpolation syntax '${'")
        try:
            logdir.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            parser.error(f"Cannot create fresh --logdir {logdir}: {exc}")
    run_id = uuid.uuid4().hex[:12]
    command = training_command(logdir, args.device, run_id)
    (logdir / "check_command.json").write_text(json.dumps(command, indent=2) + "\n")
    print(f"Runtime check: {sys.executable}; device={args.device}; logs={logdir}", flush=True)
    try:
        run_training(command, logdir, args.timeout, args.device)
        result = verify_outputs(logdir, run_id)
    except (OSError, ValueError, KeyError, RuntimeError, TimeoutError, KeyboardInterrupt) as exc:
        result = dict(status="failed", error=str(exc) or type(exc).__name__)
        transcript = logdir / "training.log"
        if transcript.is_file():
            with transcript.open(errors="replace") as source:
                print("".join(deque(source, maxlen=40)), file=sys.stderr, end="")
        print(f"Runtime check FAILED: {result['error']}. Logs: {logdir}", file=sys.stderr)
    result.update(python=sys.executable, device=args.device, logdir=str(logdir))
    (logdir / "check_result.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "passed":
        return 1
    print(
        f"Runtime check PASSED: {result['environment_steps']} environment steps, "
        "completed learner updates, final evaluation, "
        f"checkpoint/replay and offline W&B output. Logs: {logdir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
