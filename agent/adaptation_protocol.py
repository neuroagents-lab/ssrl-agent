"""Shared contracts for certified task adaptation protocols."""

OFFLINE_UPDATES = {
    "fresh_heads": 4688,
    "fresh_heads_10x": 46880,
    "pretrained_heads": 0,
}
PROTOCOLS = tuple(OFFLINE_UPDATES)
TRANSFER_COMPONENTS = {
    "dreamerv3": ("enc", "dyn", "con", "dec"),
    "r2dreamer": ("enc", "dyn", "con", "r2_projector"),
    "ne_dreamer": ("enc", "dyn", "con", "ne"),
}
TASK_COMPONENTS = (
    "rew", "pol", "val", "slowval", "slowval_count",
    "retnorm", "valnorm", "advnorm",
)
# These normalizers use impl=none in the supported profiles.
STATELESS_COMPONENTS = frozenset(("valnorm", "advnorm"))


def validate_protocol(protocol):
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown adaptation protocol: {protocol!r}; choose from {PROTOCOLS}.")
    return protocol


def offline_updates(protocol):
    return OFFLINE_UPDATES[validate_protocol(protocol)]


def transfer_components(backend, protocol="fresh_heads"):
    validate_protocol(protocol)
    return TRANSFER_COMPONENTS[backend] + (
        TASK_COMPONENTS if protocol == "pretrained_heads" else ())


def source_components(backend, protocol="fresh_heads", *, task=None,
                      env_config=None, parent_algorithm="p2e", parent_task=None):
    """Map child namespaces to the parent's corresponding task namespaces."""
    mapping = {name: name for name in transfer_components(backend, protocol)}
    if protocol == "pretrained_heads" and is_walker_source_transfer(env_config):
        if task not in WALKER_TASKS or parent_algorithm not in ("p2e", "dreamer"):
            raise ValueError("Pretrained Walker transfer requires a valid task and parent algorithm.")
        # P2E parents train detached heads for all targets. Supervised parents
        # use base heads only for their collection/source task.
        if parent_algorithm == "p2e" or task != parent_task:
            label = task.removeprefix("dmc_")
            for name in TASK_COMPONENTS:
                mapping[name] = (f"task_slowval_{label}_count" if name == "slowval_count"
                                 else f"task_{name}_{label}")
    return mapping


def remap_source_parameters(params, mapping):
    """Select and rename namespaces, without falling back to base heads."""
    inverse = {source: target for target, source in mapping.items()}
    if len(inverse) != len(mapping):
        raise ValueError("Source component mapping must be one-to-one.")
    return {
        inverse[key.split("/", 1)[0]] + "/" + key.split("/", 1)[1]: value
        for key, value in params.items()
        if "/" in key and key.split("/", 1)[0] in inverse
    }


WALKER_TASKS = frozenset(
    f"dmc_walker_{name}" for name in ("stand", "walk", "run")
)
WALKER_ENV_CONFIGS = frozenset(("dmc_proprio", "dmc_vision"))
WALKER_ADAPTATION_ENV_CONFIGS = frozenset((
    "dmc_walker_adaptation", "dmc_walker_adaptation_vision",
))
WALKER_PARENT_ENV_CONFIGS = {
    "dmc_walker_adaptation": "dmc_walker_multitask",
    "dmc_walker_adaptation_vision": "dmc_walker_multitask_vision",
}
WALKER_REWARD_LABELS = tuple(
    f"label/reward/walker_{name}" for name in ("stand", "walk", "run")
)


def is_walker_source_transfer(env_config: str | None) -> bool:
    return env_config in WALKER_ADAPTATION_ENV_CONFIGS


def validate_walker_source(
    task: str, env_config: str, parent_task: str,
    parent_env_config: str, parent_step: int,
) -> None:
    if task not in WALKER_TASKS or not is_walker_source_transfer(env_config):
        raise ValueError("Walker source transfer requires a Walker adaptation profile and target.")
    if parent_task != "dmc_walker_stand":
        raise ValueError("Walker source transfer requires parent_task=dmc_walker_stand.")
    if parent_env_config != WALKER_PARENT_ENV_CONFIGS[env_config]:
        raise ValueError("Walker source and child observation profiles must match.")
    if isinstance(parent_step, bool) or not isinstance(parent_step, int) or parent_step != 1_010_000:
        raise ValueError("Walker multitask source requires exact parent step 1,010,000.")


def is_walker_task(task: str) -> bool:
    return task in WALKER_TASKS


def allowed_parent_steps(task: str) -> tuple[int, ...]:
    if is_walker_task(task):
        # Allow the collection budget rounded up to a complete environment batch.
        return (1_010_000, 1_010_048)
    if task == "craftax_classic_pixels":
        return (5_010_000,)
    raise ValueError(f"Unsupported adaptation task: {task!r}.")


def validate_parent_step(task: str, step: int) -> None:
    allowed = allowed_parent_steps(task)
    if isinstance(step, bool) or not isinstance(step, int) or step not in allowed:
        expected = " or ".join(f"{value:,}" for value in allowed)
        raise ValueError(
            f"Scientific adaptation requires parent step {expected} for "
            f"{task}; got {step}."
        )


def validate_env_config(task: str, env_config: str | None) -> None:
    if is_walker_task(task):
        if env_config not in WALKER_ENV_CONFIGS | WALKER_ADAPTATION_ENV_CONFIGS:
            raise ValueError(
                "Walker adaptation requires env_config=dmc_proprio, dmc_vision, "
                "dmc_walker_adaptation or dmc_walker_adaptation_vision; "
                f"got {env_config!r}."
            )
    elif env_config is not None:
        raise ValueError("env_config is only supported for Walker adaptation.")
