#!/usr/bin/env python3
"""Initialize a fresh ordinary Dreamer reference for Walker parent certification.

Only model initialization and serialization run: no training, checkpoint load,
replay import, environment actions, or W&B session. The output is consumed by
prepare_dmc_walker_adaptation.py and must be a new directory.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import pickle
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import adaptation_manifest as api
from agent.adaptation_protocol import WALKER_TASKS
from experiments import run_storage


REFERENCE_ENV_CONFIGS = (
    "dmc_walker_adaptation", "dmc_walker_adaptation_vision",
    "dmc_proprio", "dmc_vision",
)


def compose_config(*, task, env_config, backend, model, seed, device, output):
    """Compose the same ordinary agent and observation configuration as a child."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    task = f"dmc_walker_{task}" if task in ("stand", "walk", "run") else task
    if task not in WALKER_TASKS or env_config not in REFERENCE_ENV_CONFIGS:
        raise ValueError("Reference requires a Walker stand/walk/run task and proprio/vision config.")
    if backend not in api.SUPPORTED_BACKENDS or not api._SAFE_TEXT.fullmatch(str(model)):
        raise ValueError("Invalid reference backend or model.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Reference seed must be a nonnegative integer.")
    device = "cuda:0" if device == "cuda0" else device
    if device not in ("cpu", "cuda:0"):
        raise ValueError("Reference device must be cpu or cuda:0.")
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        config = compose(config_name="configs", overrides=[
            f"envs={env_config}", f"envs.task={task}", f"model={model}", "algo=dreamer",
            f"agent_name={backend}", f"seed={seed}", f"device={device}",
            f"logdir={json.dumps(str(output))}", "wandb.enabled=false",
            "adaptation.enabled=false", "jax_compilation_cache=false", "jax_preallocate=false",
            "envs.steps=200000", "trainer.steps=200000", "envs.eval_every=20000",
            "trainer.eval_every=20000", "trainer.eval_action_selections=[greedy]",
            "model.eval_task_heads=false", "model.train_base_task_behavior=true",
            "model.strict_task_blind_pretraining=false",
        ])
    OmegaConf.resolve(config)
    if (config.model.idm_type != "none" or config.model.expl_behavior != "greedy"
            or config.adaptation.enabled or config.wandb.enabled):
        raise ValueError("Reference must initialize ordinary Dreamer without adaptation or W&B.")
    return config


def create_reference(*, task, env_config, backend, model="size12M", seed=0,
                     device="cuda:0", output):
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"Refusing to reuse or overwrite output: {output}")
    # Configure these before importing the launcher or initializing any device.
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["WANDB_MODE"] = "disabled"
    from omegaconf import OmegaConf
    import numpy as np
    from agent import launcher

    config = compose_config(task=task, env_config=env_config, backend=backend,
        model=model, seed=seed, device=device, output=output)
    launcher._validate_agent_selection(config)
    identity = run_storage.source_identity(ROOT)
    deps = launcher._load_reference_deps()
    reference = launcher.build_reference_config(config, deps)
    # Initialize parameters and spaces without compiling training or reporting.
    values = dict(reference.items())
    values["jax"] = {**dict(reference.jax.items()), "precompile": False}
    reference = deps.elements.Config(values)
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    reference.save(str(output / "dreamer_config.yaml"))
    print(f"Initializing ordinary {backend} reference for {config.envs.task} ({env_config})", flush=True)
    agent = launcher._make_agent(reference, config, deps, config.envs.task)
    saved = agent.save()
    if saved.get("counters") != {"updates": 0, "batches": 0, "actions": 0}:
        raise ValueError(f"Fresh reference has nonzero or missing counters: {saved.get('counters')}")
    fields = [{"key": key, "dtype": np.dtype(space.dtype).str, "shape": list(space.shape)}
              for key, space in sorted(agent.spaces.items()) if key != "consec"]
    schema = {"schema_version": 1, "backend": backend, "model": model,
              "task": config.envs.task, "fields": fields}
    api._validate_target_replay_schema(schema, backend=backend, model=model,
                                      task=config.envs.task, env_config=env_config)
    records = api.parameter_records(saved["params"], api.TRANSFER_COMPONENTS[backend])
    with (output / "agent.pkl").open("xb") as stream:
        pickle.dump(saved, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    (output / "replay_schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    kind_backend = "ne-dreamer" if backend == "ne_dreamer" else backend
    metadata = {
        "schema_version": 1, "kind": f"ordinary-{kind_backend}-initialization",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_id": identity["source_id"], "source_path": str(ROOT), "source_identity": identity,
        "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "device": config.device, "seed": seed, "backend": backend,
        "task": config.envs.task, "model": model, "env_config": env_config,
        "environment_steps": 0, "counters": saved["counters"],
        "checkpoint_loaded": False, "replay_imported": False, "adaptation_enabled": False,
        "idm_type": "none", "expl_behavior": "greedy", "precompile_train_report": False,
        "transfer_components": list(api.TRANSFER_COMPONENTS[backend]),
        "transfer_parameter_count": len(records),
        "artifacts": {name: {"sha256": api.file_sha256(output / name),
                             "size": (output / name).stat().st_size}
                      for name in ("agent.pkl", "replay_schema.json", "config.yaml", "dreamer_config.yaml")},
        "initializer_sha256": api.file_sha256(__file__),
    }
    api._write_readonly_json(output / "reference_metadata.json", metadata, label="Walker reference metadata")
    for name in metadata["artifacts"]:
        (output / name).chmod(0o440)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("stand", "walk", "run"), required=True)
    parser.add_argument("--env-config", choices=REFERENCE_ENV_CONFIGS,
                        default="dmc_walker_adaptation")
    parser.add_argument("--backend", choices=api.SUPPORTED_BACKENDS, default="dreamerv3")
    parser.add_argument("--model", default="size12M")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda:0", "cuda0"), default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = create_reference(**vars(args))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(args.output.expanduser().absolute()),
                      "source_id": result["source_id"], "counters": result["counters"],
                      "artifacts": result["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
