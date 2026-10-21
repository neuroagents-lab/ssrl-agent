"""Intrinsic-dynamics module interface and configuration validation."""

from __future__ import annotations

import math
from importlib import import_module
from dataclasses import dataclass
from typing import Any, Protocol


class IDM(Protocol):
    def intrinsic_reward(
        self,
        carry: Any,
        world_state: Any,
        action: Any,
        skill: Any,
        mode: str,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Return intrinsic rewards, updated carry, and metrics."""

    def loss(
        self,
        carry: Any,
        batch: Any,
        world_outputs: Any,
    ) -> tuple[dict[str, Any], Any, dict[str, Any]]:
        """Return named losses, updated carry, and metrics."""


@dataclass(frozen=True)
class NoIDM:
    """No-op IDM used by plain DreamerV3."""

    def intrinsic_reward(self, carry, world_state, action, skill, mode):
        del world_state, action, skill, mode
        return 0.0, carry, {}

    def loss(self, carry, batch, world_outputs):
        del batch, world_outputs
        return {}, carry, {}


def _field(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _normalize(value: Any, default: str = "") -> str:
    return str(value if value is not None else default).lower()


def _enabled(value: Any) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"}


SUPPORTED_CURIOSITY_IDMS = {
    "apt",
    "disagreement",
    "icm",
    "simple_curiosity",
    "rnd",
    "ngu",
    "proto_rl",
    "gamma_progress",
    "re3",
    "ctec",
    "cip",
    "cig",
    "max",
    "smm",
}

SUPPORTED_COMPETENCY_IDMS = {
    "diayn",
    "vic",
    "dads",
    "rdads",
    "valor",
    "csf",
    "empowerment_forward",
    "empowerment_reverse",
    "empowerment_gain",
    "empowerment_potential",
    "empowerment_process",
}

SKILL_COMPETENCY_IDMS = {
    "diayn",
    "vic",
    "dads",
    "rdads",
    "valor",
    "csf",
}

SUPPORTED_CONTROL_IDMS = {"random"}

SUPPORTED_IDMS = (
    SUPPORTED_CURIOSITY_IDMS | SUPPORTED_COMPETENCY_IDMS | SUPPORTED_CONTROL_IDMS
)


def validate_supported_idm(model_or_idm: Any) -> None:
    if isinstance(model_or_idm, str) or model_or_idm is None:
        behavior = "greedy"
        idm_type = _normalize(model_or_idm, "none")
        model = None
    else:
        model = model_or_idm
        behavior = _normalize(_field(model, "expl_behavior", "greedy"), "greedy")
        idm_type = _normalize(_field(model, "idm_type", "none"), "none")

    if behavior == "greedy":
        if idm_type in {"", "none"}:
            return
        raise NotImplementedError(
            f"JAX DreamerV3 supports model.idm_type=none when expl_behavior=greedy, "
            f"got idm_type={idm_type!r}."
        )

    if behavior != "plan2explore":
        raise NotImplementedError(
            f"JAX expl_behavior={behavior!r} is not supported. "
            "Use greedy or plan2explore with idm_type=disagreement."
        )

    if idm_type.startswith("vime"):
        raise NotImplementedError(
            "JAX Plan2Explore does not support VIME."
        )
    if idm_type not in SUPPORTED_IDMS:
        available = ", ".join(sorted(SUPPORTED_IDMS))
        raise NotImplementedError(
            f"JAX Plan2Explore supports idm_type in {{{available}}}, got {idm_type!r}."
        )
    if model is None:
        return

    reward_norm_source = _normalize(
        _field(model, "idm_reward_norm_source", "replay_only"),
        "replay_only",
    )
    if reward_norm_source not in {
        "replay_only",
        "replay",
        "replay_and_imagination",
        "both",
    }:
        raise NotImplementedError(
            "model.idm_reward_norm_source must be replay_only or "
            f"replay_and_imagination, got {reward_norm_source!r}."
        )

    expl_until = int(_field(model, "expl_until", 0) or 0)
    if expl_until != 0:
        raise NotImplementedError(
            "JAX Plan2Explore requires model.expl_until=0 so training always uses "
            f"the exploration policy; got {expl_until}."
        )
    uses_skills = _enabled(_field(model, "use_skills", False))
    if uses_skills and idm_type not in SKILL_COMPETENCY_IDMS:
        raise NotImplementedError(
            "JAX Plan2Explore supports skill-conditioned exploration only for "
            f"skill competency IDMs, got idm_type={idm_type!r}."
        )
    if idm_type in SKILL_COMPETENCY_IDMS and not uses_skills:
        raise NotImplementedError(
            f"JAX Plan2Explore idm_type={idm_type!r} requires model.use_skills=True."
        )
    if idm_type == "random":
        distribution = _normalize(_field(model, "random_action_distribution", "p2e"))
        if distribution not in {"p2e", "uniform"}:
            raise ValueError("model.random_action_distribution must be p2e or uniform.")
        noise = float(_field(model, "random_action_noise", 1.0))
        if not math.isfinite(noise) or noise < 0:
            raise ValueError("model.random_action_noise must be finite and nonnegative.")
        return
    skill_space = _normalize(
        _field(model, "skill_space", "categorical"), "categorical"
    )
    if idm_type in {"vic", "valor"}:
        if skill_space != "categorical":
            raise NotImplementedError(
                f"JAX Plan2Explore {idm_type.upper()} requires categorical skills, "
                f"got skill_space={skill_space!r}."
            )
        skill_horizon = int(_field(model, "skill_horizon", 0) or 0)
        if skill_horizon <= 0:
            raise NotImplementedError(
                f"JAX Plan2Explore {idm_type.upper()} requires model.skill_horizon > 0 "
                "to define complete option trajectories."
            )
        imag_horizon = int(_field(model, "imag_horizon", skill_horizon) or skill_horizon)
        if skill_horizon != imag_horizon:
            raise NotImplementedError(
                f"JAX Plan2Explore {idm_type.upper()} requires model.skill_horizon "
                f"({skill_horizon}) to equal model.imag_horizon ({imag_horizon})."
            )
    if idm_type == "rdads":
        posterior_type = _normalize(
            _field(model, "continuous_skill_posterior", "dot_product"),
            "dot_product",
        )
        categorical = skill_space in {"categorical", "discrete"}
        if not categorical and posterior_type != "gaussian":
            raise NotImplementedError(
                "Continuous rDADS requires "
                "model.continuous_skill_posterior=gaussian. A dot-product "
                "InfoNCE score is not a normalized q(z | context) density."
            )

    plan_source = _normalize(_field(model, "plan_repr_source", "stoch"), "stoch")
    if plan_source not in {"feat", "deter", "stoch"}:
        raise NotImplementedError(
            "JAX Plan2Explore supports plan_repr_source in {feat,deter,stoch}, "
            f"got {plan_source!r}."
        )

    repr_mode = _normalize(_field(model, "expl_repr_mode", "raw_plan"), "raw_plan")
    head_type = _normalize(_field(model, "expl_head_type", "mlp"), "mlp")
    if repr_mode in {"branch", "parallel", "branch_bridge", "parallel_bridge"}:
        raise NotImplementedError("JAX Plan2Explore does not support branch or parallel representations.")
    if repr_mode == "shared_head" and head_type != "identity":
        raise NotImplementedError(
            "JAX Plan2Explore supports expl_repr_mode=shared_head only with "
            f"expl_head_type=identity, got {head_type!r}."
        )
    if repr_mode not in {"raw_plan", "shared_head"}:
        raise NotImplementedError(
            f"JAX Plan2Explore does not support expl_repr_mode={repr_mode!r}."
        )

    target = _normalize(_field(model, "disag_target", "same"), "same")
    if idm_type == "disagreement" and target not in {"same", "feat", "deter", "stoch"}:
        raise NotImplementedError(
            "JAX Plan2Explore disagreement supports disag_target in {same,feat,deter,stoch}, "
            f"got {target!r}."
        )
    offset = int(_field(model, "disag_offset", 1) or 1)
    if offset != 1:
        raise NotImplementedError(
            f"JAX Plan2Explore requires model.disag_offset=1, got {offset}."
        )


_EXPORTS = {
    "FixedCategoricalPrior": ".competency",
    "FixedHyperspherePrior": ".competency",
    "APT": ".entropy",
    "CIG": ".curiosity",
    "CIP": ".curiosity",
    "CSF": ".competency",
    "CTEC": ".curiosity",
    "DADS": ".competency",
    "DIAYN": ".competency",
    "Disagreement": ".curiosity",
    "EmpowermentGain": ".competency",
    "ForwardEmpowerment": ".competency",
    "GammaLearningProgress": ".curiosity",
    "ICM": ".curiosity",
    "MAX": ".curiosity",
    "NGU": ".curiosity",
    "PotentialEmpowerment": ".competency",
    "ProcessEmpowerment": ".competency",
    "ProtoRL": ".entropy",
    "RDADS": ".competency",
    "RE3": ".entropy",
    "RND": ".curiosity",
    "ReverseEmpowerment": ".competency",
    "SMM": ".entropy",
    "SimpleCuriosity": ".curiosity",
    "VALOR": ".competency",
    "VIC": ".competency",
    "SkillPrior": ".competency",
    "UniformBoxPrior": ".competency",
    "UniformCategoricalPrior": ".competency",
    "UniformHyperspherePrior": ".competency",
    "conditional_information_gain": ".curiosity.cig",
    "get_skill_prior": ".competency",
    "normalize_skill_space": ".competency",
    "prior_log_prob_for_skill": ".competency",
    "skill_event_dim": ".competency",
    "skill_to_indices": ".competency",
    "skill_to_vectors": ".competency",
}

__all__ = [
    "IDM",
    "NoIDM",
    "SUPPORTED_CURIOSITY_IDMS",
    "SUPPORTED_COMPETENCY_IDMS",
    "SKILL_COMPETENCY_IDMS",
    "SUPPORTED_CONTROL_IDMS",
    "SUPPORTED_IDMS",
    "validate_supported_idm",
    *_EXPORTS,
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
