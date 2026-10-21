from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib.metadata
import importlib.resources
import json
import os
import pathlib
import platform
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from functools import partial as bind
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from . import dinov2 as dinov2_utils
from . import envs as local_envs
from ssrl.idms import validate_supported_idm
from .train_budget import resolve_start_budget
from .adaptation_protocol import (
    is_walker_task, is_walker_source_transfer, validate_parent_step,
    validate_walker_source, validate_protocol, offline_updates as protocol_offline_updates,
)


DREAMER_SOURCE_COMMIT = "b65cf81a6fb13625af8722127459283f899a35d9"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
AGENT_SOURCE_ROOT = REPO_ROOT / "agent"
IDM_SOURCE_ROOT = REPO_ROOT / "ssrl"
LOCAL_DREAMER = AGENT_SOURCE_ROOT / "dreamer"
LOCAL_EMBODIED = AGENT_SOURCE_ROOT / "embodied"
LOCAL_DREAMER_SETUP = (
    "git clone https://github.com/danijar/dreamerv3.git /tmp/dreamerv3\n"
    f"git -C /tmp/dreamerv3 checkout {DREAMER_SOURCE_COMMIT}\n"
    "cp -a /tmp/dreamerv3/dreamerv3 agent/dreamer\n"
    "cp -a /tmp/dreamerv3/embodied agent/embodied"
)

JAX_CACHE_SCHEMA = "v1"
JAX_CACHE_SOURCE_SUFFIXES = frozenset({".py", ".yaml", ".yml"})


@dataclass(frozen=True)
class ReferenceDeps:
    elements: Any
    embodied: Any
    portal: Any
    yaml: Any
    dreamer_agent_cls: Any
    plan2explore_agent_cls: Any
    random_control_agent_cls: Any = None


class _LazyOutput:
    """Construct a side-effecting logger output only on the first write."""

    def __init__(self, factory):
        self._factory = factory
        self._output = None

    def __call__(self, summaries):
        if self._output is None:
            self._output = self._factory()
        return self._output(summaries)

    def wait(self):
        if self._output is not None and hasattr(self._output, "wait"):
            return self._output.wait()


def _write_wandb_receipt(logdir: str, output: Any, expected_id: str | None) -> None:
    """Atomically record the remote run that W&B actually initialized."""

    run = getattr(getattr(output, "_wandb", None), "run", None)
    actual_id = getattr(run, "id", None)
    if not isinstance(actual_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", actual_id
    ):
        raise RuntimeError("W&B initialized without a safe remote run ID")
    if expected_id and actual_id != expected_id:
        raise RuntimeError(
            f"W&B initialized unexpected run ID {actual_id!r}; expected {expected_id!r}"
        )

    directory = pathlib.Path(logdir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "wandb_run.json"
    receipt = {
        "attempt_id": os.environ.get("RUN_ATTEMPT_ID"),
        "schema_version": 1,
        "wandb_id": actual_id,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".wandb_run.json.tmp.", dir=directory
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o440)
        os.replace(temporary, destination)
        directory_descriptor = os.open(
            directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_jax_compilation_cache(config: DictConfig) -> pathlib.Path | None:
    """Configure a private persistent JAX cache before JAX is imported."""

    enabled = _bool(_get(config, "jax_compilation_cache", True))
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = str(enabled).lower()
    if not enabled:
        os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)
        os.environ["SSRL_JAX_COMPILATION_CACHE_EFFECTIVE"] = "false"
        print("JAX compilation cache: disabled")
        return None

    minimum_seconds = _float(
        _get(config, "jax_compilation_cache_min_compile_time_secs", 1.0)
    )
    minimum_bytes = _int(
        _get(config, "jax_compilation_cache_min_entry_size_bytes", 0)
    )
    if minimum_seconds < 0:
        raise ValueError(
            "jax_compilation_cache_min_compile_time_secs must be nonnegative, "
            f"got {minimum_seconds}"
        )
    if minimum_bytes < 0:
        raise ValueError(
            "jax_compilation_cache_min_entry_size_bytes must be nonnegative, "
            f"got {minimum_bytes}"
        )

    configured_directory = str(
        _get(config, "jax_compilation_cache_dir", "") or ""
    ).strip()
    inherited_directory = os.environ.get("JAX_COMPILATION_CACHE_DIR", "").strip()
    configured_root = str(
        _get(config, "jax_compilation_cache_root", "") or ""
    ).strip()
    inherited_root = os.environ.get(
        "JAX_COMPILATION_CACHE_ROOT", ""
    ).strip()

    if configured_directory:
        directory = pathlib.Path(configured_directory).expanduser()
        privacy_root = directory
    elif configured_root:
        privacy_root = pathlib.Path(configured_root).expanduser()
        directory = (
            privacy_root
            / JAX_CACHE_SCHEMA
            / _jax_compilation_cache_namespace(config)
        )
    elif inherited_directory:
        # Use JAX's explicit cache directory without adding a namespace.
        directory = pathlib.Path(inherited_directory).expanduser()
        privacy_root = directory
    else:
        root_text = inherited_root
        if root_text:
            root = pathlib.Path(root_text).expanduser()
        else:
            xdg_root = os.environ.get("XDG_CACHE_HOME", "").strip()
            root = (
                pathlib.Path(xdg_root).expanduser()
                if xdg_root
                else pathlib.Path.home() / ".cache"
            ) / "ssrl_agent" / "jax_compilation_cache"
        privacy_root = root
        directory = root / JAX_CACHE_SCHEMA / _jax_compilation_cache_namespace(config)

    directory = directory.absolute()
    privacy_root = privacy_root.absolute()
    _validate_jax_cache_location(directory, config)
    try:
        _ensure_private_cache_directory(privacy_root)
        _ensure_private_cache_directory(directory)
    except OSError as exc:
        # Training can continue without a persistent compilation cache.
        print(
            "Warning: disabling JAX compilation cache because its directory "
            f"could not be prepared ({directory}: {exc})."
        )
        os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
        os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)
        os.environ["SSRL_JAX_COMPILATION_CACHE_EFFECTIVE"] = "false"
        return None

    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(directory)
    os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = str(
        minimum_seconds
    )
    os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = str(minimum_bytes)
    os.environ["SSRL_JAX_COMPILATION_CACHE_EFFECTIVE"] = "true"
    print(f"JAX compilation cache: {directory}")
    return directory


def _jax_compilation_cache_namespace(config: DictConfig) -> str:
    identities = {
        "schema": JAX_CACHE_SCHEMA,
        "source": _jax_cache_source_identity(),
        "config": _jax_cache_config_identity(config),
        "runtime": _jax_cache_runtime_identity(),
        "hardware": _jax_cache_hardware_identity(config),
    }
    payload = json.dumps(
        identities, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _jax_cache_source_identity() -> str:
    supplied = os.environ.get("RUN_SOURCE_ID", "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", supplied):
        return supplied

    # Hash local sources when no launcher-provided source identity is available.
    digest = hashlib.sha256()
    for path in sorted([*AGENT_SOURCE_ROOT.rglob("*"), *IDM_SOURCE_ROOT.rglob("*")]):
        if (
            path.suffix not in JAX_CACHE_SOURCE_SUFFIXES
            or path.is_symlink()
            or not path.is_file()
        ):
            continue
        relative = path.relative_to(REPO_ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _jax_cache_config_identity(config: DictConfig) -> str:
    # Include tracing and shape settings; exclude run identities so compatible
    # jobs share compiled executables. JAX also keys the compiled computation.
    payload = {
        key: _jax_cache_plain_value(_get(config, key, None))
        for key in (
            "agent_name",
            "device",
            "deterministic_run",
            "batch_size",
            "batch_length",
            "buffer",
            "envs",
            "model",
        )
    }
    trainer = _get(config, "trainer", {})
    payload["trainer"] = {
        key: _jax_cache_plain_value(_get(trainer, key, None))
        for key in ("batch_size", "batch_length", "replay_context")
    }
    payload["jax"] = {
        key: _jax_cache_plain_value(_get(config, key, None))
        for key in (
            "jax_autotune",
            "jax_donate_policy_params",
            "jax_preallocate",
        )
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jax_cache_plain_value(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {
            str(key): _jax_cache_plain_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jax_cache_plain_value(item) for item in value]
    return value


def _jax_cache_runtime_identity() -> str:
    supplied = os.environ.get("RUN_RUNTIME_SHA256", "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", supplied):
        return supplied

    def package_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "missing"

    payload = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "jax": package_version("jax"),
        "jaxlib": package_version("jaxlib"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _jax_cache_hardware_identity(config: DictConfig) -> str:
    supplied = os.environ.get("SSRL_JAX_CACHE_HARDWARE_ID", "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", supplied):
        return supplied
    payload = {
        "device": str(_get(config, "device", "unknown")).split(":", 1)[0],
        "machine": platform.machine(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _validate_jax_cache_location(directory: pathlib.Path, config: DictConfig) -> None:
    resolved = directory.resolve(strict=False)
    if resolved == pathlib.Path(resolved.anchor):
        raise ValueError("JAX compilation cache directory may not be a filesystem root")
    logdir_value = _get(config, "logdir", None)
    if logdir_value is None:
        return
    logdir = pathlib.Path(str(logdir_value)).expanduser().absolute().resolve(strict=False)
    if resolved == logdir or logdir in resolved.parents:
        raise ValueError(
            "JAX compilation cache must be outside logdir so checkpoint/recovery "
            f"snapshots do not copy it: cache={resolved}, logdir={logdir}"
        )


def _ensure_private_cache_directory(directory: pathlib.Path) -> None:
    if directory.exists() or directory.is_symlink():
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(
                f"JAX compilation cache path is not a real directory: {directory}"
            )
    else:
        # Allow concurrent creation, then validate ownership and type.
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)

    status = directory.stat()
    if hasattr(os, "getuid") and status.st_uid != os.getuid():
        raise ValueError(
            "JAX compilation cache directory must be owned by the current user: "
            f"{directory}"
        )
    if stat.S_IMODE(status.st_mode) & 0o022:
        raise ValueError(
            "JAX compilation cache directory may not be group/world writable: "
            f"{directory}"
        )


def _jax_compilation_cache_reference_settings(
    config: DictConfig,
) -> dict[str, Any]:
    """Return only restart-stable cache controls for the Dreamer config."""

    return {
        "compilation_cache": _bool(
            _get(config, "jax_compilation_cache", True)
        ),
        # Cache location and availability can change when a job is requeued.
        "compilation_cache_dir": "",
        "persistent_cache_min_compile_time_secs": _float(
            _get(
                config,
                "jax_compilation_cache_min_compile_time_secs",
                1.0,
            )
        ),
        "persistent_cache_min_entry_size_bytes": _int(
            _get(
                config,
                "jax_compilation_cache_min_entry_size_bytes",
                0,
            )
        ),
    }


def run(config: DictConfig) -> None:
    agent_name = _agent_name(config)
    _validate_agent_selection(config)
    validate_supported_idm(config.model)
    _validate_adaptation_config(config)

    logdir = pathlib.Path(str(config.logdir)).expanduser()
    if (logdir / "latest.pt").exists():
        print(
            "Warning: found legacy PyTorch checkpoint latest.pt in "
            f"{logdir}. JAX Dreamer will ignore it and use only JAX "
            "checkpoints for resume."
        )

    # Set the persistent-cache environment before importing Dreamer, Embodied,
    # or JAX. JAX reads these values while its config module is initialized.
    _prepare_jax_compilation_cache(config)
    deps = _load_reference_deps()
    ref_config = build_reference_config(config, deps)
    batched_env = _uses_batched_env(
        config.envs,
        _bool(_get(config.envs, "include_image", True)),
        config.model,
    )
    _configure_process(ref_config)
    _save_translated_config(ref_config, deps.elements)
    _configure_wandb(config)

    _setup_portal(ref_config, deps)

    args = _make_run_args(ref_config, config, deps, batched_env)

    train_task = _train_task(config.envs.task)
    eval_tasks = _eval_tasks(config.envs)
    labelled_eval_tasks = [(_task_label(task), task) for task in eval_tasks]
    if not labelled_eval_tasks:
        labelled_eval_tasks = [("", train_task)]

    def make_agent():
        return _make_agent(ref_config, config, deps, train_task)

    def make_replay_train():
        return _make_replay(ref_config, deps, "replay", "train")

    def make_replay_eval(label="eval"):
        return _make_replay(ref_config, deps, f"eval_replay_{_task_label(label)}", "eval")

    def make_env_train(index):
        return local_envs.make_env(
            config,
            deps.elements,
            deps.embodied,
            index,
            mode="train",
            task=train_task,
        )

    def make_env_eval(index, task):
        return local_envs.make_env(
            config,
            deps.elements,
            deps.embodied,
            index,
            mode="eval",
            task=task,
        )

    def make_env_train_batch(num_envs):
        return local_envs.make_batched_env(
            config,
            deps.elements,
            num_envs,
            mode="train",
            task=train_task,
        )

    def make_env_eval_batch(num_envs, task, seed_start=0):
        return local_envs.make_batched_env(
            config,
            deps.elements,
            num_envs,
            mode="eval",
            task=task,
            seed_start=seed_start,
        )

    def make_stream(replay, mode):
        return _make_stream(ref_config, deps, replay, mode)

    def make_logger():
        return _make_logger(ref_config, config, deps)

    from .run import train_eval_multi

    print(
        f"Launching JAX {_display_agent_name(agent_name)} train+eval:",
        f"train_task={train_task}",
        "eval_tasks=" + ",".join(task for _, task in labelled_eval_tasks),
    )
    train_eval_multi(
        make_agent,
        make_replay_train,
        make_replay_eval,
        make_env_train,
        make_env_eval,
        make_stream,
        make_logger,
        args,
        labelled_eval_tasks,
        make_env_train_batch=make_env_train_batch if batched_env else None,
        make_env_eval_batch=make_env_eval_batch if batched_env else None,
    )


def _make_run_args(ref_config, config: DictConfig, deps: ReferenceDeps, batched_env: bool):
    run_args = dict(ref_config.run)
    run_args["video_pred_log"] = _bool(run_args.get("video_pred_log", False))
    run_args["video_policy_log"] = _bool(run_args.get("video_policy_log", True))
    eval_seed_bank = _get(config.envs, "eval_seed_bank", None)
    if OmegaConf.is_config(eval_seed_bank):
        eval_seed_bank = OmegaConf.to_container(eval_seed_bank, resolve=True)
    train_task = _train_task(_get(config.envs, "task", ""))
    adaptation = _get(config, "adaptation", {})
    environment_metadata = local_envs.environment_metadata(train_task)
    task_head_specs = _eval_task_head_specs(config.envs, config.model)
    eval_policy_roles = _eval_policy_role_specs(config.envs, config.model)
    eval_action_selections = _get(
        config.trainer, "eval_action_selections", None
    )
    if OmegaConf.is_config(eval_action_selections):
        eval_action_selections = OmegaConf.to_container(
            eval_action_selections, resolve=True
        )
    if isinstance(eval_action_selections, str):
        eval_action_selections = tuple(
            item.strip()
            for item in eval_action_selections.strip("[]").split(",")
            if item.strip()
        )
    elif eval_action_selections is not None:
        eval_action_selections = tuple(str(item) for item in eval_action_selections)
    else:
        eval_action_selections = ("sampled", "greedy")
    eval_task_head_labels = tuple(
        spec.split("=", 1)[0] for spec in task_head_specs
    )
    optional_args = {}
    if eval_task_head_labels:
        # elements.Config cannot infer an empty sequence's type. Omit unused
        # task-head fields; the training loop treats them as empty.
        optional_args["eval_task_head_labels"] = eval_task_head_labels
    if eval_policy_roles:
        optional_args["eval_policy_roles"] = tuple(eval_policy_roles)
    return deps.elements.Config(
        **run_args,
        **optional_args,
        replica=ref_config.replica,
        replicas=ref_config.replicas,
        logdir=ref_config.logdir,
        batch_size=ref_config.batch_size,
        batch_length=ref_config.batch_length,
        report_length=ref_config.report_length,
        consec_train=ref_config.consec_train,
        consec_report=ref_config.consec_report,
        replay_context=ref_config.replay_context,
        train_task=train_task,
        protocol=str(_get(config.envs, "protocol", "standard")),
        task_registry_hash=environment_metadata.get("task_registry_hash", ""),
        port_version=environment_metadata.get("port_version", ""),
        eval_action_selections=eval_action_selections,
        train_seed=_int(_get(config, "seed", 0)),
        env_backend=str(_get(config.envs, "backend", "unknown")),
        action_repeat=_int(
            local_envs.task_env_value(
                config.envs, "action_repeat", train_task, 1
            )
        ),
        eval_seed_bank=eval_seed_bank,
        eval_seed_offset=_int(_get(config.envs, "eval_seed_offset", 1_000_000)),
        eval_policy_seed=_int(
            _get(config.trainer, "eval_policy_seed", 2_000_000)
        ),
        eval_report_seed=_int(
            _get(config.trainer, "eval_report_seed", 3_000_000)
        ),
        eval_parallel=(
            _bool(_get(config.trainer, "eval_parallel", True))
            and int(run_args.get("eval_envs", 0)) > 1
        ),
        from_checkpoint_regex=".*",
        log_every_steps=_int(config.trainer.update_log_every),
        report_every_steps=_int(config.trainer.eval_every),
        save_every_steps=_int(
            _get(config.trainer, "save_every", config.trainer.eval_every)
        ),
        batched_env=batched_env,
        host_batched_env=(
            batched_env and dinov2_utils.encoder_enabled(config.model)
        ),
        device_actor_chunk=_int(
            _get(
                config.envs,
                "device_actor_chunk",
                _get(config.envs, "mjx_actor_chunk", 32),
            )
        ),
        adaptation_enabled=_bool(_get(adaptation, "enabled", False)),
        adaptation_protocol=str(_get(adaptation, "protocol", "fresh_heads")),
        adaptation_backend=str(_get(adaptation, "backend", "")),
        adaptation_env_config=_adaptation_env_config(config),
        adaptation_parent_task=str(_get(adaptation, "parent_task", "")),
        adaptation_parent_env_config=str(_get(adaptation, "parent_env_config", "")),
        adaptation_parent_algorithm=str(_get(adaptation, "parent_algorithm", "p2e")),
        adaptation_parent_idm=str(_get(adaptation, "parent_idm", "")),
        adaptation_parent_seed=_int(_get(adaptation, "parent_seed", 0)),
        adaptation_model=str(_get(adaptation, "model", "")),
        adaptation_parent_run_uid=str(
            _get(adaptation, "parent_run_uid", "")
        ),
        adaptation_parent_source_digest=str(
            _get(adaptation, "parent_source_digest", "")
        ),
        adaptation_parent_checkpoint=str(
            _get(adaptation, "parent_checkpoint", "")
        ),
        adaptation_parent_checkpoint_sha256=str(
            _get(adaptation, "parent_checkpoint_sha256", "")
        ),
        adaptation_parent_replay=str(
            _get(adaptation, "parent_replay", "")
        ),
        adaptation_replay_inventory_sha256=str(
            _get(adaptation, "replay_inventory_sha256", "")
        ),
        adaptation_compatibility_certificate=str(
            _get(adaptation, "compatibility_certificate", "")
        ),
        adaptation_compatibility_certificate_sha256=str(
            _get(adaptation, "compatibility_certificate_sha256", "")
        ),
        adaptation_cohort_validation_receipt=str(
            _get(adaptation, "cohort_validation_receipt", "")
        ),
        adaptation_cohort_validation_receipt_sha256=str(
            _get(adaptation, "cohort_validation_receipt_sha256", "")
        ),
        adaptation_parent_replay_length=_int(
            _get(adaptation, "parent_replay_length", 0)
        ),
        pretraining_environment_steps=_int(
            _get(adaptation, "pretraining_environment_steps", 0)
        ),
        adaptation_offline_updates=_int(
            _get(adaptation, "offline_updates", 0)
        ),
        adaptation_offline_save_every=_int(
            _get(adaptation, "offline_save_every", 0)
        ),
        adaptation_strict_task_blind_pretraining=_bool(
            _get(adaptation, "strict_task_blind_pretraining", False)
        ),
    )


def build_reference_config(config: DictConfig, deps: ReferenceDeps):
    configs = _load_reference_configs(deps)
    defaults = copy.deepcopy(configs["defaults"])
    _extend_reference_jax_defaults(defaults)
    _extend_reference_defaults(defaults, config.model, _agent_name(config))
    ref = deps.elements.Config(defaults)
    ref = ref.update(_translation(config))
    return ref


def _extend_reference_jax_defaults(defaults: dict[str, Any]) -> None:
    defaults.setdefault("jax", {}).setdefault("transfer_guard", True)
    defaults.setdefault("jax", {}).setdefault("deterministic", False)
    defaults.setdefault("jax", {}).setdefault("autotune", 1)
    defaults.setdefault("jax", {}).setdefault("profiler", False)
    defaults.setdefault("jax", {}).setdefault("donate_policy_params", False)
    defaults.setdefault("jax", {}).setdefault("compilation_cache", True)
    # elements.Config uses the default value's type for later updates, so the
    # optional path must use an empty string rather than None.
    defaults.setdefault("jax", {}).setdefault("compilation_cache_dir", "")
    defaults.setdefault("jax", {}).setdefault(
        "persistent_cache_min_compile_time_secs", 1.0
    )
    defaults.setdefault("jax", {}).setdefault(
        "persistent_cache_min_entry_size_bytes", 0
    )


def _extend_reference_defaults(defaults: dict[str, Any], model: DictConfig, agent_name: str) -> None:
    env = defaults.setdefault("env", {})
    env.setdefault("craftax", {"size": [64, 64], "repeat": 1})

    run = defaults.setdefault("run", {})
    run.setdefault("video_pred_log", False)
    run.setdefault("video_policy_log", True)
    run.setdefault("video_policy_max_frames", 0)
    run.setdefault("video_policy_stride", 1)

    agent = defaults.setdefault("agent", {})
    _deep_setdefault(agent, _agent_translation(model, agent_name))
    if (
        not _bool(_get(model, "repval_loss", True))
        or (
            _is_plan2explore(model)
            and _bool(_get(model, "strict_task_blind_pretraining", False))
        )
    ):
        # Remove the replay-value default before the recursive config merge.
        agent.setdefault("loss_scales", {}).pop("repval", None)
    agent.setdefault("eval_task_heads", ("none",))
    agent.setdefault("train_base_task_behavior", True)
    agent.setdefault("task_head_loss_scales", {
        "rew": 1.0,
        "policy": 1.0,
        "value": 1.0,
        "repval": 0.3,
    })

    if not _is_plan2explore(model) or _p2e_idm_type(model) == "random":
        return
    loss_scales = agent.setdefault("loss_scales", {})
    idm_loss_key = _p2e_idm_loss_key(model)
    for key, value in {
        idm_loss_key: _float(_get(_get(model, "loss_scales", {}), idm_loss_key, _get(_get(model, "loss_scales", {}), "idm", 1.0))),
        "expl_policy": _float(_get(_get(model, "loss_scales", {}), "policy", 1.0)),
        "expl_value": _float(_get(_get(model, "loss_scales", {}), "value", 1.0)),
    }.items():
        loss_scales.setdefault(key, value)
    p2e = agent.setdefault("p2e", {})
    _deep_setdefault(p2e, _p2e_translation(model))


def _deep_setdefault(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict):
            child = target.setdefault(key, {})
            _deep_setdefault(child, value)
        else:
            target.setdefault(key, value)


def _load_reference_deps() -> ReferenceDeps:
    _require_local_reference_checkout()
    try:
        import embodied
        import elements
        import portal
        import ruamel.yaml as yaml
        from .dreamer.agent import DreamerAgent
        from .p2e.agent import Plan2ExploreAgent
        from .p2e.random_control import RandomControlAgent
    except ImportError as exc:
        raise RuntimeError(
            "JAX Dreamer dependencies are not installed. Install the JAX "
            "dependency set from requirements.txt. The local Dreamer source "
            f"folders must be present under {AGENT_SOURCE_ROOT}."
        ) from exc
    return ReferenceDeps(
        elements, embodied, portal, yaml, DreamerAgent, Plan2ExploreAgent,
        RandomControlAgent,
    )


def _require_local_reference_checkout() -> None:
    missing = [path for path in (LOCAL_DREAMER, LOCAL_EMBODIED) if not path.exists()]
    if missing:
        raise RuntimeError(
            "Local Dreamer and Embodied source folders are required for the "
            "JAX IDM migration.\n"
            "Missing:\n"
            + "\n".join(f"  {path}" for path in missing)
            + "\n"
            "Create it with:\n"
            f"{LOCAL_DREAMER_SETUP}"
        )
    path = str(AGENT_SOURCE_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_reference_configs(deps: ReferenceDeps):
    try:
        raw = importlib.resources.files("dreamer").joinpath("configs.yaml").read_text()
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Could not load dreamer/configs.yaml from the local reference "
            f"checkout ({DREAMER_SOURCE_COMMIT})."
        ) from exc
    return deps.yaml.YAML(typ="safe").load(raw)


def _translation(config: DictConfig) -> dict[str, Any]:
    env_config = config.envs
    model = config.model
    task = _train_task(env_config.task)
    platform, devices = _jax_platform_and_devices(config.device)
    env_size = _list(_get(env_config, "size", [64, 64]))
    action_repeat = _int(
        local_envs.task_env_value(env_config, "action_repeat", task, 1)
    )
    include_image = _bool(_get(env_config, "include_image", True))
    replay_size = _int(_get(config.buffer, "max_size", 5e6))
    replay_chunksize = _int(
        _get(
            env_config,
            "replay_chunksize",
            _get(config.buffer, "chunksize", 1024),
        )
    )
    if replay_chunksize < 1:
        raise ValueError(
            f"buffer.chunksize must be positive, got {replay_chunksize}"
        )
    batched_env = _uses_batched_env(env_config, include_image, model)
    jax_preallocate = (
        platform == "cuda"
        and _bool(_get(config, "jax_preallocate", True))
        and not dinov2_utils.may_use_cuda(model)
        and not (batched_env and include_image)
    )
    eval_episodes = _int(config.trainer.eval_episode_num)
    if eval_episodes < 0:
        raise ValueError(
            "trainer.eval_episode_num must be nonnegative, "
            f"got {eval_episodes}"
        )
    requested_eval_envs = _int(
        _get(
            env_config,
            "eval_env_num",
            _get(config.trainer, "eval_env_num", 10),
        )
    )
    if requested_eval_envs < 0:
        raise ValueError(
            "eval_env_num must be nonnegative, "
            f"got {requested_eval_envs}"
        )
    eval_envs = min(requested_eval_envs, eval_episodes)
    if batched_env and include_image:
        # Madrona-MJX allows one renderer per process, so separate batched
        # evaluation environments cannot render alongside training.
        eval_envs = 0
    replay_context = _int(_get(config.trainer, "replay_context", 1))
    if replay_context < 0:
        raise ValueError(f"trainer.replay_context must be >= 0, got {replay_context}")
    video_policy_max_frames = _int(
        _get(env_config, "video_policy_max_frames", 0)
    )
    if video_policy_max_frames < 0 or video_policy_max_frames == 1:
        raise ValueError(
            "envs.video_policy_max_frames must be 0 (unlimited) or at least 2, "
            f"got {video_policy_max_frames}"
        )
    video_policy_stride = 1
    if video_policy_max_frames:
        horizon = max(
            0,
            _int(local_envs.task_env_value(env_config, "time_limit", task, 0)),
        )
        if horizon:
            # Include the reset and terminal frames while keeping the total at
            # or below the configured cap across a full-length episode.
            video_policy_stride = max(
                1,
                (horizon + video_policy_max_frames - 2)
                // (video_policy_max_frames - 1),
            )
    log_every = 120
    report_every = 300
    save_every = 900
    donate_policy_params = _get(config, "jax_donate_policy_params", None)
    if donate_policy_params is None:
        donate_policy_params = (
            batched_env and not dinov2_utils.encoder_enabled(model)
        )

    update = {
        "logdir": str(config.logdir),
        "task": str(task),
        "seed": _int(config.seed),
        "script": "train_eval",
        "batch_size": _int(config.batch_size),
        "batch_length": _int(config.batch_length),
        "report_length": min(32, _int(config.batch_length)),
        "replay_context": replay_context,
        "random_agent": False,
        "logger": {
            "outputs": _logger_outputs(config),
            "filter": (
                "score|length|fps|ratio|train/loss/|train/rand/|"
                "train/disag|train/expl|train/(p2e_)?opt/adaptive/|"
                "episode|eval|epstats|env/|termination/|"
                "nonfinite/|reset/|replacement/"
            ),
            "timer": True,
        },
        "env": _env_translation(env_config, env_size, action_repeat, include_image),
        "replay": {
            "size": replay_size,
            "online": True,
            "fracs": {"uniform": 1.0, "priority": 0.0, "recency": 0.0},
            "chunksize": replay_chunksize,
        },
        "run": {
            "steps": _int(config.trainer.steps),
            "train_ratio": _float(config.trainer.train_ratio),
            "log_every": log_every,
            "report_every": report_every,
            "save_every": save_every,
            "envs": _int(env_config.env_num),
            "eval_envs": eval_envs,
            "eval_eps": eval_episodes,
            "report_batches": 1,
            "from_checkpoint": "",
            "video_pred_log": _bool(config.trainer.video_pred_log),
            "video_policy_log": _bool(config.trainer.video_policy_log),
            "video_policy_max_frames": video_policy_max_frames,
            "video_policy_stride": video_policy_stride,
            "debug": platform == "cpu" or _int(env_config.env_num) <= 1 or batched_env,
            # Disable synchronous GPU telemetry, which can stall the learner.
            "usage": {
                "psutil": True,
                "nvsmi": False,
                "gputil": False,
                "malloc": False,
                "gc": False,
            },
        },
        "jax": {
            "platform": platform,
            "compute_dtype": _compute_dtype(config),
            "policy_devices": devices,
            "train_devices": devices,
            "mock_devices": 0,
            "prealloc": jax_preallocate,
            "jit": _bool(_get(model, "compile", True)),
            "debug": False,
            "deterministic": _bool(_get(config, "deterministic_run", False)),
            "autotune": _int(_get(config, "jax_autotune", 1)),
            "profiler": _bool(_get(config, "jax_profiler", False)),
            "donate_policy_params": _bool(donate_policy_params),
            **_jax_compilation_cache_reference_settings(config),
            "transfer_guard": not batched_env,
            "expect_devices": 0,
            "enable_policy": True,
            "coordinator_address": "",
        },
        "agent": _agent_translation(
            model,
            _agent_name(config),
            env_config,
            train_ratio=_float(config.trainer.train_ratio),
            batch_length=_int(config.batch_length),
            algorithm2_adaptation=_bool(
                _get(_get(config, "adaptation", {}), "enabled", False)
            ) and protocol_offline_updates(
                _get(_get(config, "adaptation", {}), "protocol", "fresh_heads")
            ) > 0,
        ),
    }
    return update


def _agent_translation(
    model: DictConfig,
    agent_name: str = "dreamerv3",
    env_config: DictConfig | None = None,
    *,
    train_ratio: float | None = None,
    batch_length: int | None = None,
    algorithm2_adaptation: bool = False,
) -> dict[str, Any]:
    agent_name = _normalize_agent_name(agent_name)
    plan2explore = _is_plan2explore(model)
    strict_task_blind_pretraining = _bool(
        _get(model, "strict_task_blind_pretraining", False)
    )
    if strict_task_blind_pretraining and not plan2explore:
        raise ValueError(
            "model.strict_task_blind_pretraining is only valid for "
            "Plan2Explore training."
        )
    if strict_task_blind_pretraining and _float(
        _get(model, "expl_extr_scale", 0.0)
    ):
        raise ValueError(
            "Strict task-blind Plan2Explore requires expl_extr_scale=0."
        )
    units = _int(_get(model, "units", 1024))
    depth = _int(_get(model, "depth", 64))
    encoder = _encoder_translation(model, units, depth)
    # Replay TD(lambda) consumes scalar task reward directly, bypassing the
    # detached reward model. Strict P2E therefore trains task behavior only from
    # reward-model predictions in imagination.
    repval_loss = (
        _bool(_get(model, "repval_loss", True))
        and not strict_task_blind_pretraining
    )
    imag_last, imag_start_strategy = resolve_start_budget(
        manual_count=_int(_get(model, "imag_last", 0)),
        target_ratio=_float(_get(model, "imag_start_ratio", 0.0)),
        train_ratio=train_ratio,
        batch_length=batch_length,
        strategy=_get(model, "imag_start_strategy", None),
        automatic_strategy="random_contiguous",
        minimum_count=2 if repval_loss else 1,
        align_automatic_divisor=True,
    )
    recon_last, recon_start_strategy = resolve_start_budget(
        manual_count=_int(_get(model, "recon_last", 0)),
        target_ratio=_float(_get(model, "recon_start_ratio", 0.0)),
        train_ratio=train_ratio,
        batch_length=batch_length,
        strategy=_get(model, "recon_start_strategy", None),
        automatic_strategy="stratified",
    )
    configured_base_behavior = _get(model, "train_base_task_behavior", None)
    concurrent_multitask_p2e = (
        env_config is not None
        and _concurrent_multitask_protocol(env_config)
        and _is_plan2explore(model)
    )
    train_base_task_behavior = (
        False
        if concurrent_multitask_p2e
        else _bool(
            True if configured_base_behavior is None else configured_base_behavior
        )
    )
    task_head_loss_scales = {
        "rew": 1.0,
        "policy": 1.0,
        "value": 1.0,
        "repval": 0.3,
        **_prior_kwargs(model, "task_head_loss_scales"),
    }
    agent = {
        "rep_loss": _agent_rep_loss(agent_name),
        "algorithm2_adaptation": bool(algorithm2_adaptation),
        "train_base_task_behavior": train_base_task_behavior,
        "strict_task_blind_pretraining": strict_task_blind_pretraining,
        "task_head_loss_scales": task_head_loss_scales,
        "walker_task_reward_labels": (
            env_config is not None
            and _concurrent_multitask_protocol(env_config)
            and str(_get(env_config, "task", "")).startswith("dmc_walker_")
        ),
        "policy_checks": _bool(_get(model, "policy_checks", False)),
        "log_grads": _bool(_get(model, "log_grads", False)),
        "loss_scales": _loss_scales(model, agent_name),
        "opt": {
            "lr": _float(_get(model, "lr", 4e-5)),
            "agc": _float(_get(model, "agc", 0.3)),
            "eps": _float(_get(model, "eps", _get(model, "opt_eps", 1e-20))),
            "beta1": _float(_get(model, "beta1", 0.9)),
            "beta2": _float(_get(model, "beta2", 0.999)),
            "momentum": True,
            "wd": _float(_get(model, "weight_decay", 0.0)),
            "schedule": "const",
            "warmup": _int(_get(model, "warmup", 1000)),
            "anneal": 0,
        },
        "adaptive_critic": _adaptive_critic_translation(model),
        "dyn": {
            "typ": "rssm",
            "rssm": {
                "deter": _int(_get(model, "deter", 8192)),
                "hidden": _int(_get(model, "hidden", 1024)),
                "stoch": 32,
                "classes": _int(_get(model, "discrete", 64)),
                "act": _jax_act(_get(model, "act", "SiLU")),
                "norm": "rms",
                "unroll": _rssm_unroll(
                    _get(_get(model, "rssm", {}), "unroll", 1)
                ),
                "unimix": 0.01,
                "outscale": 1.0,
                "winit": "trunc_normal_in",
                "imglayers": 2,
                "obslayers": 1,
                "dynlayers": 1,
                "absolute": False,
                "blocks": 8,
                "free_nats": _float(_get(model, "kl_free", 1.0)),
            },
        },
        "enc": encoder,
        "dec": {
            "typ": "simple",
            "simple": {
                "depth": depth,
                "mults": [2, 3, 4, 4],
                "layers": 3,
                "units": units,
                "act": _jax_act(_get(model, "act", "SiLU")),
                "norm": "rms",
                "outscale": 1.0,
                "winit": "trunc_normal_in",
                "outer": False,
                "kernel": 5,
                "bspace": 8,
                "strided": False,
            },
        },
        "rewhead": _mlp_head(units, output="symexp_twohot", outscale=0.0),
        "conhead": _mlp_head(units, output="binary", outscale=1.0),
        "policy": {
            "layers": 3,
            "units": units,
            "act": _jax_act(_get(model, "act", "SiLU")),
            "norm": "rms",
            "minstd": 0.1,
            "maxstd": 1.0,
            "outscale": 0.01,
            "unimix": 0.01,
            "winit": "trunc_normal_in",
        },
        "value": _mlp_head(units, layers=3, output="symexp_twohot", outscale=0.0),
        "policy_dist_disc": "categorical",
        "policy_dist_cont": "bounded_normal",
        "imag_last": imag_last,
        "imag_start_ratio": _float(_get(model, "imag_start_ratio", 0.0)),
        "imag_start_strategy": imag_start_strategy,
        "recon_last": recon_last,
        "recon_start_ratio": _float(_get(model, "recon_start_ratio", 0.0)),
        "recon_start_strategy": recon_start_strategy,
        "imag_length": _int(_get(model, "imag_horizon", 15)),
        "horizon": _int(_get(model, "horizon", 333)),
        "contdisc": True,
        "imag_loss": {
            "slowtar": False,
            "lam": _float(_get(model, "lamb", 0.95)),
            "actent": _float(_get(model, "act_entropy", 3e-4)),
            "slowreg": 1.0,
        },
        "repl_loss": {
            "slowtar": False,
            "lam": _float(_get(model, "lamb", 0.95)),
            "slowreg": 1.0,
        },
        "slowvalue": {
            "rate": _float(_get(model, "slow_target_fraction", 0.02)),
            "every": _int(_get(model, "slow_target_update", 1)),
        },
        "retnorm": {
            "impl": "perc",
            "rate": 0.01,
            "limit": 1.0,
            "perclo": 5.0,
            "perchi": 95.0,
            "debias": False,
        },
        "valnorm": {"impl": "none", "rate": 0.01, "limit": 1e-8},
        "advnorm": {"impl": "none", "rate": 0.01, "limit": 1e-8},
        "ac_grads": False,
        "reward_grad": (
            _bool(_get(model, "reward_grad", True))
            and not strict_task_blind_pretraining
        ),
        "repval_loss": repval_loss,
        "repval_grad": (
            _bool(_get(model, "repval_grad", True))
            and not strict_task_blind_pretraining
        ),
        "report": True,
        "report_gradnorms": False,
        "r2dreamer": _r2dreamer_translation(model),
        "ne_dreamer": _ne_dreamer_translation(model),
        "dreamer_pro": _dreamerpro_translation(model),
    }
    if _is_plan2explore(model):
        agent["p2e"] = _p2e_translation(model)
    if dinov2_utils.encoder_enabled(model):
        agent["dinov2"] = dinov2_utils.config_from_model(model)
    if env_config is not None:
        eval_task_heads = _eval_task_head_specs(env_config, model)
        if eval_task_heads:
            agent["eval_task_heads"] = eval_task_heads
    return agent


def _encoder_translation(model: DictConfig, units: int, depth: int) -> dict[str, Any]:
    if dinov2_utils.encoder_enabled(model):
        cfg = dinov2_utils.config_from_model(model)
        return {
            "typ": "identity",
            "identity": {
                "keys": [cfg["feature_key"]],
            },
        }
    encoder = _get(model, "encoder", {})
    typ = str(_get(encoder, "typ", "simple")).lower()
    if typ not in {"simple", "learned", "cnn"}:
        raise ValueError(
            "model.encoder.typ must be simple or dinov2; "
            f"got {typ!r}."
        )
    return {
        "typ": "simple",
        "simple": {
            "depth": depth,
            "mults": [2, 3, 4, 4],
            "layers": 3,
            "units": units,
            "act": _jax_act(_get(model, "act", "SiLU")),
            "norm": "rms",
            "winit": "trunc_normal_in",
            "symlog": True,
            "outer": False,
            "kernel": 5,
            "strided": False,
        },
    }


def _loss_scales(model: DictConfig, agent_name: str = "dreamerv3") -> dict[str, float]:
    source = _get(model, "loss_scales", {})
    agent_name = _normalize_agent_name(agent_name)
    scales = {
        "rew": _float(_get(source, "rew", 1.0)),
        "con": _float(_get(source, "con", 1.0)),
        "dyn": _float(_get(source, "dyn", 1.0)),
        "rep": _float(_get(source, "rep", 0.1)),
        "policy": _float(_get(source, "policy", 1.0)),
        "value": _float(_get(source, "value", 1.0)),
        "repval": _float(_get(source, "repval", 0.3)),
    }
    if (
        not _bool(_get(model, "repval_loss", True))
        or (
            _is_plan2explore(model)
            and _bool(_get(model, "strict_task_blind_pretraining", False))
        )
    ):
        scales.pop("repval")
    if agent_name == "dreamerv3":
        scales["rec"] = _float(_get(source, "recon", _get(source, "rec", 1.0)))
    elif agent_name == "r2dreamer":
        scales["barlow"] = _float(_get(source, "barlow", 0.05))
    elif agent_name == "ne_dreamer":
        scales["ne_dreamer"] = _float(_get(source, "ne_dreamer", 1.0))
    elif agent_name == "dreamerpro":
        scales["swav"] = _float(_get(source, "swav", 1.0))
        scales["temp"] = _float(_get(source, "temp", 1.0))
        scales["norm"] = _float(_get(source, "norm", 1.0))
    if _is_plan2explore(model) and _p2e_idm_type(model) != "random":
        idm_loss_key = _p2e_idm_loss_key(model)
        scales.update(
            **{idm_loss_key: _float(_get(source, idm_loss_key, _get(source, "idm", 1.0)))},
            expl_policy=_float(_get(source, "expl_policy", _get(source, "policy", 1.0))),
            expl_value=_float(_get(source, "expl_value", _get(source, "value", 1.0))),
        )
    return scales


def _adaptive_critic_translation(model: DictConfig) -> dict[str, Any]:
    source = _get(model, "adaptive_critic", {})
    default_critics = "expl_value" if _is_plan2explore(model) else "value"
    return {
        "enabled": _bool(_get(source, "enabled", False)),
        "critics": str(_get(source, "critics", default_critics)),
        "scope": str(_get(source, "scope", "output")),
        "granularity": str(_get(source, "granularity", "feature")),
        "meta_lr": _float(_get(source, "meta_lr", 1e-3)),
        "trace_decay": _float(_get(source, "trace_decay", 0.9)),
        "init_scale": _float(_get(source, "init_scale", 1.0)),
        "min_scale": _float(_get(source, "min_scale", 0.25)),
        "max_scale": _float(_get(source, "max_scale", 4.0)),
        "warmup": _int(_get(source, "warmup", _get(model, "warmup", 1000))),
        "eps": _float(_get(source, "eps", 1e-8)),
    }


def _r2dreamer_translation(model: DictConfig) -> dict[str, Any]:
    source = _get(model, "r2dreamer", {})
    return {"lambd": _float(_get(source, "lambd", 5e-4))}


def _ne_dreamer_translation(model: DictConfig) -> dict[str, Any]:
    source = _get(model, "ne_dreamer", {})
    return {
        "hidden_dim": _int(_get(source, "hidden_dim", 256)),
        "num_layers": _int(_get(source, "num_layers", 2)),
        "num_heads": _int(_get(source, "num_heads", 4)),
        "dropout": _float(_get(source, "dropout", 0.0)),
        "use_actions": _bool(_get(source, "use_actions", True)),
        "use_projector": _bool(_get(source, "use_projector", True)),
        "use_same": _bool(_get(source, "use_same", False)),
        "use_next": _bool(_get(source, "use_next", True)),
        "predict_horizon": _int(_get(source, "predict_horizon", 1)),
        "horizon_discount": _float(_get(source, "horizon_discount", 0.9)),
        "loss_type": str(_get(source, "loss_type", "barlow")),
        "lambd": _float(_get(source, "lambd", 5e-4)),
        "weight_same": _float(_get(source, "weight_same", 1.0)),
        "weight_next": _float(_get(source, "weight_next", 1.0)),
        "max_seq_len": _int(_get(source, "max_seq_len", 128)),
    }


def _dreamerpro_translation(model: DictConfig) -> dict[str, Any]:
    source = _get(model, "dreamer_pro", {})
    aug = _get(source, "aug", {})
    return {
        "warm_up": _int(_get(source, "warm_up", 1)),
        "num_prototypes": _int(_get(source, "num_prototypes", 2500)),
        "proto_dim": _int(_get(source, "proto_dim", 32)),
        "temperature": _float(_get(source, "temperature", 0.1)),
        "sinkhorn_eps": _float(_get(source, "sinkhorn_eps", 0.05)),
        "sinkhorn_iters": _int(_get(source, "sinkhorn_iters", 3)),
        "ema_update_every": _int(_get(source, "ema_update_every", 1)),
        "ema_update_fraction": _float(_get(source, "ema_update_fraction", 0.05)),
        "freeze_prototypes_iters": _int(_get(source, "freeze_prototypes_iters", 10000)),
        "aug": {
            "max_delta": _float(_get(aug, "max_delta", 3.0)),
            "same_across_time": _bool(_get(aug, "same_across_time", True)),
            "bilinear": _bool(_get(aug, "bilinear", True)),
        },
    }


def _p2e_idm_type(model: DictConfig) -> str:
    return str(_get(model, "idm_type", "disagreement")).lower()


def _p2e_idm_loss_key(model: DictConfig) -> str:
    idm_type = _p2e_idm_type(model)
    if idm_type == "disagreement":
        return "disag"
    return idm_type


def _p2e_translation(model: DictConfig) -> dict[str, Any]:
    idm_type = _p2e_idm_type(model)
    idm_loss_key = _p2e_idm_loss_key(model)
    hidden = _int(_get(model, "disag_units", 0))
    if hidden <= 0:
        width = _float(_get(model, "disag_model_width_factor", 1.0))
        hidden = max(1, int(round(_int(_get(model, "idm_hidden_dim", 256)) * width)))
    layers = _int(_get(model, "disag_layers", 0))
    if layers <= 0:
        layers = _int(_get(model, "idm_n_layers", 2))
    normalize = _bool(_get(model, "disag_normalize", False))
    rnd_hidden = _int(_get(model, "rnd_hidden_dim", _get(model, "idm_hidden_dim", 256)))
    if rnd_hidden <= 0:
        rnd_hidden = _int(_get(model, "idm_hidden_dim", 256)) or 256
    rnd_output = _int(_get(model, "rnd_output_dim", 512))
    if rnd_output <= 0:
        rnd_output = rnd_hidden
    ngu_rnd_hidden = _int(_get(model, "ngu_rnd_hidden_dim", _get(model, "idm_hidden_dim", 256)))
    if ngu_rnd_hidden <= 0:
        ngu_rnd_hidden = _int(_get(model, "idm_hidden_dim", 256)) or 256
    ngu_rnd_output = _int(_get(model, "ngu_rnd_output_dim", ngu_rnd_hidden))
    if ngu_rnd_output <= 0:
        ngu_rnd_output = ngu_rnd_hidden
    skill_space = str(_get(model, "skill_space", "categorical")).lower()
    if skill_space in {"continuous", "sphere", "hypersphere"}:
        default_skill_prior = "uniform_hypersphere"
    elif skill_space in {"box", "cube", "euclidean"}:
        default_skill_prior = "uniform_box"
    else:
        default_skill_prior = "uniform_categorical"
    common = {
        "enabled": _is_plan2explore(model),
        "idm_type": idm_type,
        "idm_loss_key": idm_loss_key,
        "random_action_distribution": str(_get(model, "random_action_distribution", "p2e")),
        "random_action_noise": _float(_get(model, "random_action_noise", 1.0)),
        "use_skills": _bool(_get(model, "use_skills", False)),
        "skill_space": str(_get(model, "skill_space", "categorical")),
        "num_skills": _int(_get(model, "num_skills", 10)),
        "skill_dim": _int(_get(model, "skill_dim", 0)),
        "skill_embed_dim": _int(_get(model, "skill_embed_dim", 0)),
        "skill_horizon": _int(_get(model, "skill_horizon", 0)),
        "continuous_skill_posterior": str(_get(model, "continuous_skill_posterior", "dot_product")),
        "continuous_skill_temperature": _float(_get(model, "continuous_skill_temperature", 0.1)),
        "plan_repr_source": str(_get(model, "plan_repr_source", "stoch")),
        "expl_repr_mode": str(_get(model, "expl_repr_mode", "raw_plan")),
        "expl_grad_mode": str(_get(model, "expl_grad_mode", "head_only")),
        "expl_head_type": str(_get(model, "expl_head_type", "mlp")),
        "expl_intr_scale": _float(_get(model, "expl_intr_scale", 1.0)),
        "expl_extr_scale": _float(_get(model, "expl_extr_scale", 0.0)),
        "idm_reward_normalize": _bool(
            _get(model, "idm_reward_normalize", False)
        ),
        "idm_reward_normalize_eps": _float(
            _get(model, "idm_reward_normalize_eps", 1e-8)
        ),
        "idm_reward_norm_source": str(
            _get(model, "idm_reward_norm_source", "replay_only")
        ),
        "idm_hidden_dim": _int(_get(model, "idm_hidden_dim", 256)),
        "idm_layers": _int(_get(model, "idm_n_layers", 2)),
        "idm_act": str(_get(model, "idm_act", _get(model, "disag_act", "relu"))),
        "disag_target": str(_get(model, "disag_target", "same")),
        "disag_offset": _int(_get(model, "disag_offset", 1)),
        "disag_models": _int(_get(model, "disag_models", 5)),
        "disag_hidden_dim": hidden,
        "disag_layers": layers,
        "disag_action_cond": _bool(_get(model, "disag_action_cond", True)),
        "disag_bootstrap": _bool(_get(model, "disag_bootstrap", True)),
        "disag_scale": _float(_get(model, "disag_scale", 10000.0)),
        "disag_log": _bool(_get(model, "disag_log", False)),
        "disag_normalize": normalize,
        "disag_eps": _float(_get(model, "disag_eps", _get(model, "idm_reward_normalize_eps", 1e-8))),
        "disag_act": str(_get(model, "disag_act", "relu")),
    }
    common.update(
        icm_proj_dim=_int(
            _get(
                model,
                "icm_proj_dim",
                _get(model, "idm_output_dim", _get(model, "idm_hidden_dim", 256)),
            )
        ),
        icm_proj_hidden_dim=_int(
            _get(model, "icm_proj_hidden_dim", _get(model, "idm_hidden_dim", 256))
        ),
        icm_proj_layers=_int(_get(model, "icm_proj_layers", 1)),
        icm_forward_loss_weight=_float(
            _get(model, "icm_forward_loss_weight", 0.2)
        ),
        simple_curiosity_scale=_float(_get(model, "simple_curiosity_scale", 1.0)),
        icm_scale=_float(_get(model, "icm_scale", 0.01)),
        max_models=_int(_get(model, "max_models", 5)),
        max_hidden_dim=_int(_get(model, "max_hidden_dim", 256)),
        max_layers=_int(_get(model, "max_layers", 2)),
        max_chunk_size=_int(_get(model, "max_chunk_size", 0)),
        max_scale=_float(_get(model, "max_scale", 1.0)),
        max_lambda=_float(_get(model, "max_lambda", 0.1)),
        max_min_log_var=_float(_get(model, "max_min_log_var", -10.0)),
        max_max_log_var=_float(_get(model, "max_max_log_var", 2.0)),
        max_normalize=_bool(_get(model, "max_normalize", False)),
        max_eps=_float(_get(model, "max_eps", _get(model, "idm_reward_normalize_eps", 1e-8))),
        cig_models=_int(_get(model, "cig_models", 5)),
        cig_hidden_dim=_int(_get(model, "cig_hidden_dim", 400)),
        cig_layers=_int(_get(model, "cig_layers", 4)),
        cig_scale=_float(_get(model, "cig_scale", 1.0)),
        cig_ridge_scale=_float(_get(model, "cig_ridge_scale", 1.0)),
        cig_aleatoric_ema=_float(_get(model, "cig_aleatoric_ema", 0.99)),
        cig_normalize=_bool(_get(model, "cig_normalize", True)),
        cig_reward_ema=_float(_get(model, "cig_reward_ema", 0.99)),
        cig_eps=_float(
            _get(model, "cig_eps", _get(model, "idm_reward_normalize_eps", 1e-8))
        ),
        cig_act=str(_get(model, "cig_act", "elu")),
        smm_latent_dim=_int(_get(model, "smm_latent_dim", 128)),
        smm_hidden_dim=_int(_get(model, "smm_hidden_dim", 150)),
        smm_layers=_int(_get(model, "smm_layers", 2)),
        smm_scale=_float(_get(model, "smm_scale", 1.0)),
        smm_kl_scale=_float(_get(model, "smm_kl_scale", 0.5)),
        smm_target=str(_get(model, "smm_target", "uniform")),
        smm_log_target_const=_float(_get(model, "smm_log_target_const", 0.0)),
        smm_min_log_var=_float(_get(model, "smm_min_log_var", -7.0)),
        smm_max_log_var=_float(_get(model, "smm_max_log_var", 2.0)),
        smm_reward_normalize=_bool(_get(model, "smm_reward_normalize", False)),
        smm_eps=_float(_get(model, "smm_eps", _get(model, "idm_reward_normalize_eps", 1e-8))),
        rnd_hidden_dim=rnd_hidden,
        rnd_output_dim=rnd_output,
        rnd_layers=_int(_get(model, "rnd_layers", _get(model, "idm_n_layers", 2))),
        rnd_scale=_float(_get(model, "rnd_scale", 1.0)),
        rnd_normalize=_bool(_get(model, "rnd_normalize", True)),
        rnd_eps=_float(_get(model, "rnd_eps", _get(model, "idm_reward_normalize_eps", 1e-8))),
        apt_rep_dim=_int(_get(model, "apt_rep_dim", 256)),
        apt_k=_int(_get(model, "apt_k", 16)),
        apt_knn_clip=_float(_get(model, "apt_knn_clip", 0.0005)),
        apt_normalize_phi=_bool(_get(model, "apt_normalize_phi", True)),
        apt_train_phi=_bool(_get(model, "apt_train_phi", True)),
        apt_reward_normalize=_bool(_get(model, "apt_reward_normalize", True)),
        apt_knn_chunk_size=_int(_get(model, "apt_knn_chunk_size", 1024)),
        apt_memory_size=_int(_get(model, "apt_memory_size", 2048)),
        apt_memory_sample_size=_int(_get(model, "apt_memory_sample_size", 2048)),
        apt_eps=_float(_get(model, "apt_eps", _get(model, "idm_reward_normalize_eps", 1e-8))),
        re3_rep_dim=_int(_get(model, "re3_rep_dim", 50)),
        re3_k=_int(_get(model, "re3_k", 53)),
        re3_scale=_float(_get(model, "re3_scale", 0.1)),
        re3_log_reward=_bool(_get(model, "re3_log_reward", False)),
        re3_reward_normalize=_bool(_get(model, "re3_reward_normalize", True)),
        re3_knn_chunk_size=_int(_get(model, "re3_knn_chunk_size", 1024)),
        re3_memory_size=_int(_get(model, "re3_memory_size", 2048)),
        re3_memory_sample_size=_int(_get(model, "re3_memory_sample_size", 2048)),
        re3_eps=_float(_get(model, "re3_eps", _get(model, "idm_reward_normalize_eps", 1e-8))),
        proto_rl_rep_dim=_int(_get(model, "proto_rl_rep_dim", 128)),
        proto_rl_pred_dim=_int(_get(model, "proto_rl_pred_dim", 512)),
        proto_rl_num_prototypes=_int(_get(model, "proto_rl_num_prototypes", 512)),
        proto_rl_candidates_per_prototype=_int(_get(model, "proto_rl_candidates_per_prototype", 4)),
        proto_rl_temperature=_float(_get(model, "proto_rl_temperature", 0.1)),
        proto_rl_sinkhorn_iters=_int(_get(model, "proto_rl_sinkhorn_iters", 3)),
        proto_rl_target_momentum=_float(_get(model, "proto_rl_target_momentum", 0.95)),
        proto_rl_knn_k=_int(_get(model, "proto_rl_knn_k", 3)),
        proto_rl_reward_normalize=_bool(_get(model, "proto_rl_reward_normalize", False)),
        proto_rl_eps=_float(_get(model, "proto_rl_eps", _get(model, "idm_reward_normalize_eps", 1e-8))),
        ngu_rep_dim=_int(_get(model, "ngu_rep_dim", 32)),
        ngu_k=_int(_get(model, "ngu_k", 10)),
        ngu_scale=_float(_get(model, "ngu_scale", 0.3)),
        ngu_kernel_epsilon=_float(_get(model, "ngu_kernel_epsilon", 1e-4)),
        ngu_cluster_distance=_float(_get(model, "ngu_cluster_distance", 8e-3)),
        ngu_pseudo_count_c=_float(_get(model, "ngu_pseudo_count_c", 1e-3)),
        ngu_max_similarity=_float(_get(model, "ngu_max_similarity", 8.0)),
        ngu_lifelong_clip=_float(_get(model, "ngu_lifelong_clip", 5.0)),
        ngu_use_rnd=_bool(_get(model, "ngu_use_rnd", True)),
        ngu_rnd_hidden_dim=ngu_rnd_hidden,
        ngu_rnd_output_dim=ngu_rnd_output,
        ngu_rnd_layers=_int(_get(model, "ngu_rnd_layers", 2)),
        ngu_inverse_coef=_float(_get(model, "ngu_inverse_coef", 1.0)),
        ngu_rnd_coef=_float(_get(model, "ngu_rnd_coef", 1.0)),
        ngu_vectorized_scan=_bool(_get(model, "ngu_vectorized_scan", True)),
        ngu_update_tail=_int(_get(model, "ngu_update_tail", 0)),
        ngu_eps=_float(_get(model, "ngu_eps", 1e-8)),
        progress_scale=_float(_get(model, "progress_scale", 1.0)),
        gamma_progress_gamma=_float(_get(model, "gamma_progress_gamma", 0.9995)),
        ctec_rep_dim=_int(_get(model, "ctec_rep_dim", 64)),
        ctec_hidden_dim=_int(_get(model, "ctec_hidden_dim", _get(model, "idm_hidden_dim", 1024))),
        ctec_layers=_int(_get(model, "ctec_layers", 2)),
        ctec_scale=_float(_get(model, "ctec_scale", 1.0)),
        ctec_gamma=_float(_get(model, "ctec_gamma", 0.99)),
        ctec_temperature=_float(_get(model, "ctec_temperature", 1.0)),
        ctec_learn_temperature=_bool(_get(model, "ctec_learn_temperature", True)),
        ctec_similarity=str(_get(model, "ctec_similarity", "l1")),
        ctec_loss_type=str(_get(model, "ctec_loss_type", "infonce")),
        ctec_normalize_embeddings=_bool(_get(model, "ctec_normalize_embeddings", True)),
        ctec_reward_estimator=str(_get(model, "ctec_reward_estimator", "single")),
        ctec_reward_normalize_truncation=_bool(_get(model, "ctec_reward_normalize_truncation", False)),
        ctec_replay_batch_size=_int(_get(model, "ctec_replay_batch_size", 256)),
        cip_state_source=str(_get(model, "cip_state_source", "deter")),
        cip_imag_last=_int(_get(model, "cip_imag_last", 0)),
        cip_imag_start_strategy=str(
            _get(model, "cip_imag_start_strategy", "last") or "last"
        ),
        cip_probes=_int(_get(model, "cip_probes", 8)),
        cip_scale=_float(_get(model, "cip_scale", 1.0)),
        cip_eps=_float(_get(model, "cip_eps", 1e-6)),
        cip_positive=_bool(_get(model, "cip_positive", True)),
        cip_normalize=_bool(_get(model, "cip_normalize", False)),
        cip_seed=_int(_get(model, "cip_seed", 0)),
        cip_stop_reward_grad=_bool(_get(model, "cip_stop_reward_grad", True)),
        cip_replay_metric=_bool(_get(model, "cip_replay_metric", False)),
        empowerment_gain_gamma=_float(_get(model, "empowerment_gain_gamma", 0.9995)),
        empowerment_particles=_int(_get(model, "empowerment_particles", 16)),
        empowerment_next_samples=_int(_get(model, "empowerment_next_samples", 1)),
        empowerment_q_loss_weight=_float(_get(model, "empowerment_q_loss_weight", 1.0)),
        empowerment_action_concat=_bool(_get(model, "empowerment_action_concat", False)),
        empowerment_reward_chunk_size=_int(_get(model, "empowerment_reward_chunk_size", 0)),
        empowerment_estimator=str(_get(model, "empowerment_estimator", "gce")),
        empowerment_gce_latent_dim=_int(_get(model, "empowerment_gce_latent_dim", 64)),
        empowerment_gce_power=_float(_get(model, "empowerment_gce_power", 1.0)),
        empowerment_gce_waterfill_iters=_int(_get(model, "empowerment_gce_waterfill_iters", 32)),
        empowerment_gce_recon_scale=_float(_get(model, "empowerment_gce_recon_scale", 1.0)),
        empowerment_gce_state_noise=_bool(_get(model, "empowerment_gce_state_noise", False)),
        diayn_prior=str(_get(model, "diayn_prior", None) or default_skill_prior),
        diayn_prior_kwargs=_prior_kwargs(model, "diayn_prior_kwargs"),
        diayn_entropy_coef=_float(_get(model, "diayn_entropy_coef", 0.0)),
        diayn_replay_batch_size=_int(_get(model, "diayn_replay_batch_size", 128)),
        vic_prior=str(_get(model, "vic_prior", None) or default_skill_prior),
        vic_prior_kwargs=_prior_kwargs(model, "vic_prior_kwargs"),
        vic_entropy_coef=_float(_get(model, "vic_entropy_coef", 0.0)),
        vic_condition_on_start=_bool(_get(model, "vic_condition_on_start", True)),
        vic_reward_at_end=_bool(_get(model, "vic_reward_at_end", True)),
        dads_prior=str(_get(model, "dads_prior", None) or default_skill_prior),
        dads_prior_kwargs=_prior_kwargs(model, "dads_prior_kwargs"),
        dads_entropy_coef=_float(_get(model, "dads_entropy_coef", 0.0)),
        dads_use_marginal_model=_bool(_get(model, "dads_use_marginal_model", False)),
        dads_marginal_samples=_int(_get(model, "dads_marginal_samples", 100)),
        dads_replay_batch_size=_int(_get(model, "dads_replay_batch_size", 256)),
        rdads_prior=str(_get(model, "rdads_prior", None) or default_skill_prior),
        rdads_prior_kwargs=_prior_kwargs(model, "rdads_prior_kwargs"),
        rdads_entropy_coef=_float(_get(model, "rdads_entropy_coef", 0.0)),
        rdads_replay_batch_size=_int(_get(model, "rdads_replay_batch_size", 256)),
        valor_prior=str(_get(model, "valor_prior", None) or default_skill_prior),
        valor_prior_kwargs=_prior_kwargs(model, "valor_prior_kwargs"),
        valor_entropy_coef=_float(_get(model, "valor_entropy_coef", 0.0)),
        valor_mlp_layers=_int(_get(model, "valor_mlp_layers", 1)),
        valor_num_subseq=_int(_get(model, "valor_num_subseq", 11)),
        valor_reward_at_end=_bool(_get(model, "valor_reward_at_end", True)),
        valor_subtract_log_prior=_bool(_get(model, "valor_subtract_log_prior", True)),
        valor_head_hidden_dim=_int(_get(model, "valor_head_hidden_dim", 0)),
        csf_prior=str(_get(model, "csf_prior", None) or default_skill_prior),
        csf_prior_kwargs=_prior_kwargs(model, "csf_prior_kwargs"),
        csf_proj_dim=_int(_get(model, "csf_proj_dim", 0)),
        csf_proj_hidden_dim=_int(_get(model, "csf_proj_hidden_dim", 1024)),
        csf_proj_layers=_int(_get(model, "csf_proj_layers", 2)),
        csf_reward_mode=str(_get(model, "csf_reward_mode", "inner")),
        csf_posterior=str(_get(model, "csf_posterior", "dot_product")),
        csf_temperature=_float(_get(model, "csf_temperature", 1.0)),
        csf_normalize_features=_bool(_get(model, "csf_normalize_features", False)),
        csf_normalize_skills=_bool(_get(model, "csf_normalize_skills", False)),
        csf_input_norm=str(_get(model, "csf_input_norm", "none")),
        csf_input_norm_eps=_float(_get(model, "csf_input_norm_eps", 1e-3)),
        csf_xi=_float(_get(model, "csf_xi", 5.0)),
        csf_num_negatives=_int(_get(model, "csf_num_negatives", 256)),
        csf_sample_prior_negatives=_bool(
            _get(model, "csf_sample_prior_negatives", True)
        ),
        csf_exclude_positive_from_negatives=_bool(_get(model, "csf_exclude_positive_from_negatives", True)),
        csf_replay_batch_size=_int(_get(model, "csf_replay_batch_size", 256)),
        csf_offset=_int(_get(model, "csf_offset", 1)),
    )
    return common


def _env_translation(env_config: DictConfig, size, action_repeat, include_image):
    suite = str(env_config.task).split("_", 1)[0]
    if suite == "dmc":
        return {
            "dmc": {
                "size": size,
                "repeat": action_repeat,
                "proprio": _bool(_get(env_config, "include_proprio", not include_image)),
                "image": include_image,
                "camera": -1,
            }
        }
    if suite == "dmlab":
        return {
            "dmlab": {
                "size": size,
                "repeat": action_repeat,
                "episodic": True,
                "use_seed": True,
            }
        }
    if suite == "atari":
        return {
            "atari": {
                "size": size,
                "repeat": action_repeat,
                "sticky": _bool(_get(env_config, "sticky", True)),
                "gray": _bool(_get(env_config, "gray", False)),
                "actions": _get(env_config, "actions", "needed"),
                "lives": _get(env_config, "lives", "unused"),
                "noops": _int(_get(env_config, "noops", 30)),
                "autostart": _bool(_get(env_config, "autostart", False)),
                "pooling": _int(_get(env_config, "pooling", 2)),
                "aggregate": _get(env_config, "aggregate", "max"),
                "resize": _get(env_config, "resize", "pillow"),
                "clip_reward": _bool(_get(env_config, "clip_reward", False)),
            }
        }
    if suite == "crafter":
        return {"crafter": {"size": size, "logs": False}}
    if suite == "craftax":
        return {"craftax": {"size": size, "repeat": action_repeat}}
    return {}


def _uses_mjx_env(env_config: DictConfig, include_image: bool) -> bool:
    backend = str(_get(env_config, "backend", "auto")).lower()
    task = str(env_config.task)
    if "_" not in task:
        return False
    suite, task_name = task.split("_", 1)
    if suite not in local_envs.PLAYGROUND_TASKS_BY_SUITE:
        return False
    if backend in {"mjx", "mujoco_playground", "playground"}:
        return True
    if backend != "auto":
        return False
    if suite == "dmc":
        return not include_image and local_envs.is_playground_task(suite, task_name)
    return local_envs.is_playground_task(suite, task_name)


def _uses_batched_env(
    env_config: DictConfig, include_image: bool, model: DictConfig
) -> bool:
    del model
    task = str(env_config.task)
    craftax = (
        task.startswith("craftax_")
        and not include_image
        and _int(_get(env_config, "env_num", 1)) > 1
        and _bool(_get(env_config, "device_batch", True))
    )
    return craftax or _uses_mjx_env(env_config, include_image)


def _make_agent(ref_config, local_config, deps: ReferenceDeps, task: str):
    _validate_agent_selection(local_config)
    obs_space, act_space = _agent_spaces(local_config, deps, task)
    jax_config = dict(ref_config.jax.items())
    if _is_plan2explore(local_config.model):
        agent_cls = (
            deps.random_control_agent_cls
            if _p2e_idm_type(local_config.model) == "random"
            else deps.plan2explore_agent_cls
        )
    else:
        agent_cls = deps.dreamer_agent_cls
    return agent_cls(
        obs_space,
        act_space,
        deps.elements.Config(
            **ref_config.agent,
            logdir=ref_config.logdir,
            seed=ref_config.seed,
            jax=jax_config,
            batch_size=ref_config.batch_size,
            batch_length=ref_config.batch_length,
            replay_context=ref_config.replay_context,
            report_length=ref_config.report_length,
            replica=ref_config.replica,
            replicas=ref_config.replicas,
        ),
    )


def _agent_spaces(local_config, deps: ReferenceDeps, task: str):
    include_image = _bool(_get(local_config.envs, "include_image", True))
    model_config = _get(local_config, "model", None)
    if _uses_mjx_env(local_config.envs, include_image) and include_image:
        probe_config = copy.deepcopy(local_config)
        probe_config.envs.include_image = False
        _disable_dinov2_for_probe(probe_config)
        env = local_envs.make_env(
            probe_config,
            deps.elements,
            deps.embodied,
            0,
            mode="train",
            task=task,
        )
        try:
            base_obs = {
                key: value
                for key, value in env.obs_space.items()
                if key in {"reward", "is_first", "is_last", "is_terminal"}
                or key.startswith("log/")
            }
            size = _image_size(local_config.envs)
            obs_space = {
                "image": deps.elements.Space(np.uint8, size + (3,), 0, 255),
                **base_obs,
            }
            act_space = dict(env.act_space)
        finally:
            env.close()
    else:
        env = local_envs.make_env(
            local_config,
            deps.elements,
            deps.embodied,
            0,
            mode="train",
            task=task,
        )
        try:
            obs_space = dict(env.obs_space)
            act_space = dict(env.act_space)
        finally:
            env.close()

    if dinov2_utils.encoder_enabled(model_config):
        obs_space = _dinov2_obs_space(obs_space, deps.elements, model_config)

    obs_space = {
        key: value
        for key, value in obs_space.items()
        if not key.startswith(("log/", "label/"))
    }
    act_space = {key: value for key, value in act_space.items() if key != "reset"}
    return obs_space, act_space


def _disable_dinov2_for_probe(config) -> None:
    model = _get(config, "model", None)
    if model is None:
        return
    encoder = _get(model, "encoder", None)
    if encoder is None:
        return
    if isinstance(encoder, dict):
        encoder["typ"] = "simple"
    else:
        setattr(encoder, "typ", "simple")


def _dinov2_obs_space(
    obs_space: dict[str, Any], elements, model: DictConfig
) -> dict[str, Any]:
    cfg = dinov2_utils.config_from_model(model)
    image_key = cfg["image_key"]
    feature_key = cfg["feature_key"]
    if image_key not in obs_space and feature_key in obs_space:
        return dict(obs_space)
    if image_key not in obs_space:
        raise ValueError(
            "model.encoder.typ=dinov2 requires image observations: missing "
            f"{image_key!r} in obs_space {sorted(obs_space)}."
        )
    spaces = dict(obs_space)
    spaces.pop(image_key)
    spaces[feature_key] = elements.Space(
        np.float32, (int(cfg["output_dim"]),), -np.inf, np.inf
    )
    return spaces


def _image_size(env_config):
    value = _get(env_config, "size", (64, 64))
    if value is None:
        value = (64, 64)
    return tuple(int(x) for x in value)


def _make_replay(ref_config, deps: ReferenceDeps, folder: str, mode: str):
    batlen = ref_config.batch_length if mode == "train" else ref_config.report_length
    consec = ref_config.consec_train if mode == "train" else ref_config.consec_report
    capacity = ref_config.replay.size if mode == "train" else ref_config.replay.size / 10
    length = consec * batlen + ref_config.replay_context
    assert ref_config.batch_size * length <= capacity
    # Evaluation replay is temporary; episode records are persisted separately.
    directory = (
        deps.elements.Path(ref_config.logdir) / folder
        if mode == "train" else None
    )
    kwargs = dict(
        length=length,
        capacity=int(capacity),
        online=ref_config.replay.online,
        chunksize=ref_config.replay.chunksize,
        directory=directory,
        replay_context_dtype=(
            str(ref_config.jax.compute_dtype)
            if int(ref_config.replay_context) else None
        ),
        # Flush replay writes before publishing a completed checkpoint.
        save_wait=(mode == "train"),
    )
    if ref_config.replay.fracs.uniform < 1 and mode == "train":
        recency = 1.0 / np.arange(1, int(capacity) + 1) ** ref_config.replay.recexp
        selectors = deps.embodied.replay.selectors
        kwargs["selector"] = selectors.Mixture(
            dict(
                uniform=selectors.Uniform(),
                priority=selectors.Prioritized(**ref_config.replay.prio),
                recency=selectors.Recency(recency),
            ),
            ref_config.replay.fracs,
        )
    return deps.embodied.replay.Replay(**kwargs)


def _make_stream(ref_config, deps: ReferenceDeps, replay, mode: str):
    fn = bind(replay.sample, ref_config.batch_size, mode)
    stream = deps.embodied.streams.Stateless(fn)
    stream = deps.embodied.streams.Consec(
        stream,
        length=ref_config.batch_length if mode == "train" else ref_config.report_length,
        consec=ref_config.consec_train if mode == "train" else ref_config.consec_report,
        prefix=ref_config.replay_context,
        strict=(mode == "train"),
        contiguous=True,
    )
    return stream


def _make_logger(ref_config, local_config, deps: ReferenceDeps):
    step = deps.elements.Counter()
    logdir = ref_config.logdir
    suite = ref_config.task.split("_", 1)[0]
    multiplier = ref_config.env.get(suite, {}).get("repeat", 1)
    outputs = [deps.elements.logger.TerminalOutput(ref_config.logger.filter, "Agent")]
    for output in ref_config.logger.outputs:
        if output == "jsonl":
            outputs.append(deps.elements.logger.JSONLOutput(logdir, "metrics.jsonl"))
            outputs.append(deps.elements.logger.JSONLOutput(logdir, "scores.jsonl", "episode/score"))
        elif output == "tensorboard":
            outputs.append(deps.elements.logger.TensorBoardOutput(logdir, ref_config.logger.fps))
        elif output == "wandb":
            # Initialize W&B on the first write, after checkpoint validation.
            def create_wandb_output():
                kwargs = _wandb_kwargs(local_config, ref_config, logdir)
                wandb_output = deps.elements.logger.WandBOutput(
                    _wandb_run_name(local_config, logdir), **kwargs
                )
                _write_wandb_receipt(logdir, wandb_output, kwargs.get("id"))
                return wandb_output

            outputs.append(_LazyOutput(create_wandb_output))
        elif output == "scope":
            outputs.append(deps.elements.logger.ScopeOutput(deps.elements.Path(logdir)))
        else:
            raise NotImplementedError(output)
    return deps.elements.Logger(step, outputs, multiplier)


def _setup_portal(ref_config, deps: ReferenceDeps) -> None:
    def init():
        deps.elements.timer.global_timer.enabled = ref_config.logger.timer

    deps.portal.setup(
        errfile=ref_config.errfile and deps.elements.Path(ref_config.logdir) / "error",
        clientkw=dict(logging_color="cyan"),
        serverkw=dict(logging_color="cyan"),
        initfns=[init],
        ipv6=ref_config.ipv6,
    )


def _configure_process(ref_config) -> None:
    if not ref_config.jax.prealloc:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _configure_wandb(config: DictConfig) -> None:
    wandb = _get(config, "wandb", {})
    if not _bool(_get(wandb, "enabled", False)):
        return
    os.environ["WANDB_DIR"] = str(config.logdir)
    mapping = {
        "project": "WANDB_PROJECT",
        "entity": "WANDB_ENTITY",
        "name": "WANDB_NAME",
        "group": "WANDB_GROUP",
        "job_type": "WANDB_JOB_TYPE",
        "notes": "WANDB_NOTES",
        "mode": "WANDB_MODE",
        "id": "WANDB_RUN_ID",
        "resume": "WANDB_RESUME",
    }
    for key, env_key in mapping.items():
        value = _get(wandb, key, None)
        if value is not None and str(value):
            os.environ[env_key] = str(value)
    tags = _get(wandb, "tags", None)
    if tags:
        os.environ["WANDB_TAGS"] = ",".join(str(tag) for tag in tags)


def _wandb_kwargs(config: DictConfig, ref_config, logdir: str) -> dict:
    wandb = _get(config, "wandb", {})
    resume_from = _get(wandb, "resume_from", None)
    keys = (
        "project",
        "entity",
        "group",
        "job_type",
        "notes",
        "mode",
        "id",
        "resume",
        "resume_from",
    )
    kwargs = {
        "dir": str(logdir),
        "config": _wandb_config_payload(config, ref_config),
    }
    for key in keys:
        if key == "resume" and resume_from is not None and str(resume_from):
            continue
        value = _get(wandb, key, None)
        if value is not None and str(value):
            kwargs[key] = value
    tags = _get(wandb, "tags", None)
    if tags:
        kwargs["tags"] = [str(tag) for tag in tags]
    return kwargs


def _wandb_config_payload(config: DictConfig, ref_config) -> dict:
    local = OmegaConf.to_container(config, resolve=True)
    payload = {
        "migration/dreamer_source_commit": DREAMER_SOURCE_COMMIT,
        **_flatten_config(local, "local"),
        **_flatten_config(dict(ref_config), "dreamerv3"),
    }
    return {
        key: value
        for key, value in payload.items()
        if (
            isinstance(value, (str, int, float, bool, list))
            or value is None
        )
    }


def _flatten_config(value, prefix: str) -> dict:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(_flatten_config(child, f"{prefix}/{key}"))
        return result
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, (str, int, float, bool)) or item is None for item in value):
            return {prefix: list(value)}
        return {prefix: str(value)}
    return {prefix: value}


def _save_translated_config(ref_config, elements) -> None:
    logdir = pathlib.Path(str(ref_config.logdir)).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    _require_real_directory(logdir, description="run log directory")

    descriptor, candidate_name = tempfile.mkstemp(
        prefix=".dreamer_config.candidate-", suffix=".yaml", dir=logdir
    )
    os.close(descriptor)
    candidate = pathlib.Path(candidate_name)
    try:
        ref_config.save(elements.Path(candidate))
        # Make the completed configuration readable by the run's group.
        candidate.chmod(0o640)
        candidate_bytes = candidate.read_bytes()
        config_sha = hashlib.sha256(candidate_bytes).hexdigest()

        lock_path = logdir / ".run_config.lock"
        with _open_nofollow_lock(lock_path) as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

            attempt_id = os.environ.get("RUN_ATTEMPT_ID", "").strip()
            if attempt_id and (
                attempt_id in {".", ".."}
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", attempt_id)
            ):
                raise ValueError(
                    "RUN_ATTEMPT_ID may contain only letters, digits, '.', "
                    f"'_', and '-'; got {attempt_id!r}."
                )

            if attempt_id:
                attempts_dir = logdir / "attempts"
                attempt_dir = attempts_dir / attempt_id
                if os.path.lexists(attempts_dir):
                    _require_real_directory(
                        attempts_dir, description="config attempts directory"
                    )
                if os.path.lexists(attempt_dir):
                    _require_real_directory(
                        attempt_dir,
                        description=f"attempt {attempt_id!r} directory",
                    )

            canonical = logdir / "dreamer_config.yaml"
            if canonical.exists():
                _require_regular_file(
                    canonical, description="initial translated config"
                )
                _require_config_bytes_match(
                    canonical,
                    candidate_bytes,
                    description="initial translated config",
                )

            _validate_manifest_config_sha(logdir, config_sha)
            _install_immutable_config(
                canonical,
                candidate_bytes,
                description="initial translated config",
            )

            # Record an attempt only after config and manifest validation.
            if attempt_id:
                attempts_dir.mkdir(exist_ok=True)
                _require_real_directory(
                    attempts_dir, description="config attempts directory"
                )
                attempt_dir.mkdir(exist_ok=True)
                _require_real_directory(
                    attempt_dir, description=f"attempt {attempt_id!r} directory"
                )
                _install_immutable_config(
                    attempt_dir / "dreamer_config.yaml",
                    candidate_bytes,
                    description=f"attempt {attempt_id!r} translated config",
                )
    finally:
        candidate.unlink(missing_ok=True)


def _install_immutable_config(destination, candidate_bytes, *, description) -> None:
    destination = pathlib.Path(destination)
    _require_real_directory(
        destination.parent, description=f"{description} parent directory"
    )
    if os.path.lexists(destination):
        _require_regular_file(destination, description=description)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.candidate-", dir=destination.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(candidate_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o640)
        try:
            # A hard link atomically publishes the file without replacing it.
            os.link(temporary, destination)
        except FileExistsError:
            _require_config_bytes_match(
                destination, candidate_bytes, description=description
            )
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)


def _require_config_bytes_match(path, candidate_bytes, *, description) -> None:
    path = pathlib.Path(path)
    _require_regular_file(path, description=description)
    existing_bytes = path.read_bytes()
    if existing_bytes == candidate_bytes:
        return
    existing_sha = hashlib.sha256(existing_bytes).hexdigest()
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
    raise RuntimeError(
        f"Refusing to overwrite differing {description} at {path}. "
        f"Existing SHA256={existing_sha}; resolved SHA256={candidate_sha}. "
        "Use a new run ID/directory for a new configuration."
    )


def _validate_manifest_config_sha(logdir, config_sha) -> None:
    manifest_path = pathlib.Path(logdir) / "run_manifest.json"
    if not os.path.lexists(manifest_path):
        return
    _require_regular_file(manifest_path, description="run manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"Invalid run manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"Invalid run manifest {manifest_path}: expected an object.")

    key = "resolved_config_sha256"
    existing_sha = manifest.get(key)
    if existing_sha is not None and existing_sha != config_sha:
        raise RuntimeError(
            f"Run manifest config mismatch at {manifest_path}: "
            f"existing SHA256={existing_sha}; resolved SHA256={config_sha}. "
            "Refusing to contaminate the existing run."
        )
    if existing_sha == config_sha:
        return

    manifest[key] = config_sha
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".run_manifest.", suffix=".tmp", dir=manifest_path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        mode = manifest_path.stat().st_mode & 0o777
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        temporary.replace(manifest_path)
    finally:
        # os.fdopen owns the descriptor after it succeeds. If it failed before
        # taking ownership, close the original descriptor defensively.
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)


def _require_real_directory(path, *, description) -> None:
    path = pathlib.Path(path)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing {description}: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(
            f"Unsafe {description} at {path}: expected a real directory, not a symlink."
        )


def _require_regular_file(path, *, description) -> None:
    path = pathlib.Path(path)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing {description}: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(
            f"Unsafe {description} at {path}: expected a regular file, not a symlink."
        )


def _open_nofollow_lock(path):
    """Open a regular lock file without truncation or symlink traversal."""
    path = pathlib.Path(path)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o660)
    except OSError as exc:
        raise RuntimeError(f"Could not safely open config lock {path}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(
                f"Unsafe config lock at {path}: expected a regular file."
            )
        return os.fdopen(descriptor, "r+b")
    except BaseException:
        os.close(descriptor)
        raise


def _train_task(task) -> str:
    task = str(task)
    if task in {"dmc_humanoid_multi", "dmc_walker_multi"}:
        domain = task.removeprefix("dmc_").removesuffix("_multi")
        return f"dmc_{domain}_stand"
    return task


def _eval_tasks(env_config: DictConfig) -> list[str]:
    tasks = _as_task_list(_get(env_config, "eval_tasks", None))
    if tasks:
        return tasks
    task = str(env_config.task)
    if task in {"dmc_humanoid_multi", "dmc_walker_multi"}:
        domain = task.removeprefix("dmc_").removesuffix("_multi")
        return [f"dmc_{domain}_stand", f"dmc_{domain}_walk", f"dmc_{domain}_run"]
    return []


def _eval_task_head_specs(env_config: DictConfig, model: DictConfig) -> list[str]:
    tasks = _eval_tasks(env_config)
    if _concurrent_multitask_protocol(env_config):
        source_task = _train_task(_get(env_config, "task", ""))
        if source_task.startswith("robokitchen_"):
            family, inventory = "RoboKitchen", "six singleton"
            expected = {
                "robokitchen_bottom_burner",
                "robokitchen_light_switch",
                "robokitchen_slide_cabinet",
                "robokitchen_hinge_cabinet",
                "robokitchen_microwave",
                "robokitchen_kettle",
            }
        elif source_task.startswith("dmc_walker_"):
            family, inventory = "DMC Walker", "three stand/walk/run"
            expected = {"dmc_walker_stand", "dmc_walker_walk", "dmc_walker_run"}
            if str(_get(env_config, "backend", "")) != "dm_control":
                raise ValueError(
                    "DMC Walker concurrent_detached_heads requires "
                    "envs.backend=dm_control for native reward relabeling."
                )
            if source_task not in expected:
                raise ValueError(
                    "DMC Walker concurrent_detached_heads requires a singleton "
                    f"source task; got {source_task!r}."
                )
        else:
            raise ValueError(
                "concurrent_detached_heads supports RoboKitchen and DMC Walker; "
                f"got source task {source_task!r}."
            )
        if len(tasks) != len(expected) or set(tasks) != expected:
            raise ValueError(
                f"{family} concurrent_detached_heads requires all {inventory} "
                f"eval_tasks; got {tasks!r}."
            )
        head_setting = _get(model, "eval_task_heads", None)
        if head_setting is not None and not _bool(head_setting):
            raise ValueError(
                f"{family} concurrent_detached_heads requires "
                "model.eval_task_heads=true."
            )
        if not _is_plan2explore(model):
            if source_task not in expected:
                raise ValueError(
                    f"{family} concurrent_detached_heads Dreamer runs require "
                    "a singleton source task; got "
                    f"{source_task!r}."
                )
            # Reuse the base policy for the source task; other tasks use
            # detached reward, policy, and value heads.
            tasks = [task for task in tasks if task != source_task]
    if len(tasks) <= 1:
        return []
    setting = _get(model, "eval_task_heads", None)
    supported = all(_supports_eval_task_head(task) for task in tasks)
    enabled = supported if setting is None else _bool(setting)
    if not enabled:
        return []
    if not supported:
        unsupported = ", ".join(task for task in tasks if not _supports_eval_task_head(task))
        raise ValueError(
            "model.eval_task_heads supports DMC walker/humanoid stand/walk/run "
            "and the six RoboKitchen singleton tasks; unsupported eval "
            f"task(s): {unsupported}"
        )
    return [f"{_task_label(task)}={task}" for task in tasks]


def _eval_policy_role_specs(
    env_config: DictConfig, model: DictConfig
) -> list[str]:
    """Returns explicit ``label=role`` routing for labelled evaluations."""
    tasks = _eval_tasks(env_config)
    if not tasks:
        return []
    source_task = _train_task(_get(env_config, "task", ""))
    source_uses_base = (
        _concurrent_multitask_protocol(env_config)
        and not _is_plan2explore(model)
    )
    return [
        f"{_task_label(task)}="
        f"{'base' if source_uses_base and task == source_task else 'task_head'}"
        for task in tasks
    ]


def _concurrent_multitask_protocol(env_config: DictConfig) -> bool:
    return (
        str(_get(env_config, "protocol", "standard"))
        == "concurrent_detached_heads"
    )


def _supports_eval_task_head(task: str) -> bool:
    task = str(task)
    if task in {
        "dmc_walker_stand",
        "dmc_walker_walk",
        "dmc_walker_run",
        "dmc_humanoid_stand",
        "dmc_humanoid_walk",
        "dmc_humanoid_run",
    }:
        return True
    return task in {
        "robokitchen_bottom_burner",
        "robokitchen_light_switch",
        "robokitchen_slide_cabinet",
        "robokitchen_hinge_cabinet",
        "robokitchen_microwave",
        "robokitchen_kettle",
    }


def _as_task_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null", "[]"}:
            return []
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return [item.strip() for item in text.split(",") if item.strip()]
    return [str(item) for item in value if str(item)]


def _task_label(task) -> str:
    label = str(task)
    if "_" in label:
        suite, rest = label.split("_", 1)
        if suite in {
            "dmc",
            "locomotion",
            "manipulation",
            "dmlab",
            "atari",
            "memorymaze",
            "crafter",
            "craftax",
            "metaworld",
            "robokitchen",
        }:
            label = rest
    return re.sub(r"[^0-9A-Za-z_]+", "_", label).strip("_") or "eval"


def _logger_outputs(config: DictConfig) -> list[str]:
    outputs = ["jsonl", "scope"]
    if _bool(_get(config.wandb, "enabled", False)):
        outputs.append("wandb")
    return outputs


def _jax_platform_and_devices(device) -> tuple[str, list[int]]:
    text = str(device).lower()
    if text.startswith("cpu"):
        return "cpu", [0]
    match = re.search(r":(\d+)", text)
    return "cuda", [int(match.group(1)) if match else 0]


def _compute_dtype(config: DictConfig) -> str:
    if str(config.device).lower().startswith("cpu"):
        return "float32"
    precision = _int(_get(config.model, "precision", 16))
    return "float32" if precision >= 32 else "bfloat16"


def _mlp_head(units, *, layers=1, output, outscale):
    config = {
        "layers": layers,
        "units": units,
        "act": "silu",
        "norm": "rms",
        "output": output,
        "outscale": outscale,
        "winit": "trunc_normal_in",
    }
    if "twohot" in output:
        config["bins"] = 255
    return config


def _jax_act(value) -> str:
    value = str(value).lower()
    return {"silu": "silu", "swish": "silu", "relu": "relu", "elu": "elu"}.get(value, value)


def _is_plan2explore(model: DictConfig) -> bool:
    return str(_get(model, "expl_behavior", "greedy")).lower() == "plan2explore"


def _agent_name(config: DictConfig) -> str:
    return _normalize_agent_name(_get(config, "agent_name", "dreamerv3"))


def _validate_agent_selection(config: DictConfig) -> None:
    _agent_name(config)


def _adaptation_env_config(config: DictConfig) -> str:
    if not is_walker_task(_train_task(_get(config.envs, "task", ""))):
        return ""
    adaptation = _get(config, "adaptation", {})
    explicit = str(_get(adaptation, "env_config", "") or _get(config.envs, "adaptation_env_config", ""))
    if explicit:
        return explicit
    vision = _bool(_get(config.envs, "include_image", False))
    if str(_get(adaptation, "parent_env_config", "")):
        return "dmc_walker_adaptation_vision" if vision else "dmc_walker_adaptation"
    return "dmc_vision" if vision else "dmc_proprio"


def _validate_adaptation_config(config: DictConfig) -> None:
    adaptation = _get(config, "adaptation", {})
    if not _bool(_get(adaptation, "enabled", False)):
        return
    protocol = validate_protocol(_get(adaptation, "protocol", "fresh_heads"))
    if not str(_get(adaptation, "backend", "")):
        raise ValueError("Adaptation backend is required.")
    backend = _normalize_agent_name(_get(adaptation, "backend", ""))
    if backend == "dreamerpro":
        raise ValueError(
            "Scientific adaptation supports dreamerv3, r2dreamer, and "
            "ne_dreamer only."
        )
    if backend != _agent_name(config):
        raise ValueError(
            f"Adaptation backend {backend!r} does not match agent_name "
            f"{_agent_name(config)!r}."
        )
    if _is_plan2explore(config.model):
        raise ValueError(
            "Adaptation children must use the ordinary Dreamer objective, "
            "not Plan2Explore."
        )
    if str(_get(config.model, "idm_type", "none")).lower() != "none":
        raise ValueError("Adaptation children may not configure an IDM.")
    if not _bool(_get(config.model, "train_base_task_behavior", True)):
        raise ValueError("Adaptation must train the ordinary base task behavior.")
    if _eval_task_head_specs(config.envs, config.model):
        raise ValueError(
            "Adaptation children may not configure detached evaluation task heads."
        )
    task = _train_task(_get(config.envs, "task", ""))
    walker = is_walker_task(task)
    env_config = _adaptation_env_config(config)
    source_transfer = is_walker_source_transfer(env_config)
    validate_parent_step(task, _int(_get(adaptation, "pretraining_environment_steps", 0)))
    if source_transfer:
        validate_walker_source(
            task, env_config, str(_get(adaptation, "parent_task", "")),
            str(_get(adaptation, "parent_env_config", "")),
            _int(_get(adaptation, "pretraining_environment_steps", 0)),
        )
    adaptation_steps = _int(config.trainer.steps)
    allowed_steps = (200_000,) if walker else (1_000_000, 5_000_000)
    if adaptation_steps not in allowed_steps:
        expected_steps = " or ".join(str(value) for value in allowed_steps)
        raise ValueError(
            f"trainer.steps must be {expected_steps} for scientific "
            f"adaptation; got {adaptation_steps}."
        )
    checks = {
        "trainer.eval_every": (
            _int(config.trainer.eval_every), 20_000 if walker else 100_000
        ),
        "trainer.eval_episode_num": (
            _int(config.trainer.eval_episode_num), 10
        ),
        "trainer.eval_env_num": (
            _int(config.trainer.eval_env_num), 10
        ),
        "trainer.batch_length": (_int(config.trainer.batch_length), 64),
        "batch_size": (_int(config.batch_size), 16),
        "trainer.train_ratio": (
            _float(config.trainer.train_ratio), 512.0 if walker else 32.0
        ),
        "trainer.replay_context": (
            _int(config.trainer.replay_context), 1
        ),
        "buffer.max_size": (_int(config.buffer.max_size), 500_000),
        "adaptation.parent_replay_length": (
            _int(_get(adaptation, "parent_replay_length", 0)), 65
        ),
        "adaptation.offline_updates": (
            _int(_get(adaptation, "offline_updates", 0)), protocol_offline_updates(protocol)
        ),
        "adaptation.offline_save_every": (
            _int(_get(adaptation, "offline_save_every", 0)), 1_000
        ),
    }
    if walker:
        checks.update({
            "envs.env_num": (_int(config.envs.env_num), 16),
            "envs.action_repeat": (
                _int(local_envs.task_env_value(config.envs, "action_repeat", task, 1)), 1
            ),
            "envs.mjx_actor_chunk": (_int(_get(config.envs, "mjx_actor_chunk", 1)), 1),
        })
        if str(_get(config.envs, "backend", "")) == "mjx":
            checks["envs.device_actor_chunk"] = (
                _int(_get(config.envs, "device_actor_chunk", _get(config.envs, "mjx_actor_chunk", 1))),
                1,
            )
    if source_transfer:
        vision = env_config.endswith("_vision")
        checks.update({
            "envs.backend": (str(_get(config.envs, "backend", "")), "dm_control"),
            "envs.include_image": (_bool(_get(config.envs, "include_image", False)), vision),
            "envs.include_proprio": (_bool(_get(config.envs, "include_proprio", True)), not vision),
            "model.reward_grad": (_bool(_get(config.model, "reward_grad", True)), True),
            "model.repval_grad": (_bool(_get(config.model, "repval_grad", True)), True),
            "model.repval_loss": (_bool(_get(config.model, "repval_loss", True)), True),
            "model.strict_task_blind_pretraining": (
                _bool(_get(config.model, "strict_task_blind_pretraining", False)), False
            ),
        })
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(
                f"{label} must be {expected} for scientific adaptation; "
                f"got {actual}."
            )
    selections = _get(config.trainer, "eval_action_selections", ("greedy",))
    if isinstance(selections, str):
        selections = tuple(
            item.strip()
            for item in selections.strip("[]").split(",")
            if item.strip()
        )
    if tuple(selections) != ("greedy",):
        raise ValueError(
            "Adaptation evaluation must use deterministic greedy actions only."
        )
    required = (
        "parent_idm",
        "model",
        "parent_run_uid",
        "parent_source_digest",
        "parent_checkpoint",
        "parent_checkpoint_sha256",
        "parent_replay",
        "replay_inventory_sha256",
        "compatibility_certificate",
        "compatibility_certificate_sha256",
        "cohort_validation_receipt",
        "cohort_validation_receipt_sha256",
    )
    missing = [key for key in required if not str(_get(adaptation, key, ""))]
    if missing:
        raise ValueError(
            f"Adaptation parent binding is incomplete: {missing}."
        )
    for key in (
        "parent_source_digest",
        "parent_checkpoint_sha256",
        "replay_inventory_sha256",
        "compatibility_certificate_sha256",
        "cohort_validation_receipt_sha256",
    ):
        value = str(_get(adaptation, key, ""))
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(
                f"adaptation.{key} must be a lowercase SHA-256; got "
                f"{value!r}."
            )
    if _int(_get(adaptation, "parent_replay_length", 0)) <= 0:
        raise ValueError("Adaptation parent replay length must be positive.")
    parent_algorithm = str(_get(adaptation, "parent_algorithm", "p2e"))
    if parent_algorithm not in ("p2e", "dreamer"):
        raise ValueError(f"Invalid adaptation parent algorithm: {parent_algorithm!r}")
    strict_parent = _bool(_get(adaptation, "strict_task_blind_pretraining", False))
    if parent_algorithm == "dreamer" and (
        (not source_transfer and backend != "dreamerv3") or str(_get(adaptation, "parent_idm", "")) != "none"
        or strict_parent
    ):
        raise ValueError("Supervised control requires DreamerV3, IDM none, and strict blindness false.")
    if parent_algorithm == "p2e" and not strict_parent:
        raise ValueError(
            "Scientific adaptation requires a strictly task-blind parent; "
            "adaptation.strict_task_blind_pretraining must be true."
        )


def _normalize_agent_name(value) -> str:
    value = str(value or "dreamerv3").lower().replace("-", "_")
    aliases = {
        "dreamer": "dreamerv3",
        "dreamerv3": "dreamerv3",
        "v3": "dreamerv3",
        "r2dreamer": "r2dreamer",
        "r2_dreamer": "r2dreamer",
        "nedreamer": "ne_dreamer",
        "ne_dreamer": "ne_dreamer",
        "dreamerpro": "dreamerpro",
        "dreamer_pro": "dreamerpro",
    }
    if value not in aliases:
        raise ValueError(
            "agent_name must be one of dreamerv3, r2dreamer, ne_dreamer, or dreamerpro; "
            f"got {value!r}."
        )
    return aliases[value]


def _agent_rep_loss(agent_name: str) -> str:
    return {
        "dreamerv3": "dreamer",
        "r2dreamer": "r2dreamer",
        "ne_dreamer": "ne_dreamer",
        "dreamerpro": "dreamerpro",
    }[_normalize_agent_name(agent_name)]


def _display_agent_name(agent_name: str) -> str:
    return {
        "dreamerv3": "DreamerV3",
        "r2dreamer": "R2-Dreamer",
        "ne_dreamer": "NE-Dreamer",
        "dreamerpro": "DreamerPro",
    }[agent_name]


def _wandb_run_name(config: DictConfig, logdir: str) -> str:
    name = _get(config.wandb, "name", None)
    if name:
        return str(name)
    parts = pathlib.Path(str(logdir)).parts
    return "/".join(parts[-4:]) if len(parts) >= 4 else str(logdir)


def _prior_kwargs(model: DictConfig, key: str) -> dict[str, Any]:
    value = _get(model, key, None)
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        import ruamel.yaml as yaml

        parsed = yaml.YAML(typ="safe", pure=True).load(text)
        if parsed is None:
            return {}
        if not isinstance(parsed, dict):
            raise TypeError(f"{key} must parse to a mapping, got {type(parsed).__name__}.")
        return dict(parsed)
    raise TypeError(f"{key} must be a mapping, YAML mapping string, or null.")


def _get(obj, key, default=None):
    if isinstance(obj, DictConfig):
        return obj.get(key, default)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _int(value) -> int:
    return int(float(value))


def _rssm_unroll(value) -> int | bool:
    # Booleans select full or no unrolling; integers select partial unrolling.
    return value if isinstance(value, bool) else _int(value)


def _float(value) -> float:
    return float(value)


def _bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _list(value):
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    return list(value)
