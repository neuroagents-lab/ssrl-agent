"""JAX skill priors for competency-based exploration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import ninjax as nj


def _normalize_prior_kwargs(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            import ruamel.yaml as yaml
        except ImportError as exc:
            raise RuntimeError("Parsing YAML prior kwargs requires ruamel.yaml.") from exc
        parsed = yaml.YAML(typ="safe", pure=True).load(text)
        if parsed is None:
            return {}
        if not isinstance(parsed, dict):
            raise TypeError(f"Expected prior kwargs YAML to parse to dict, got {type(parsed)}.")
        return dict(parsed)
    raise TypeError(f"Expected prior kwargs dict, YAML string, or None, got {type(raw)}.")


@dataclass(frozen=True)
class SkillPrior:
    num_skills: int | None = None
    skill_dim: int | None = None
    skill_space: str = "categorical"

    @property
    def is_categorical(self) -> bool:
        return str(self.skill_space).lower() == "categorical"

    @property
    def event_dim(self) -> int:
        if self.is_categorical:
            if self.num_skills is None:
                raise ValueError("Categorical skill prior is missing num_skills.")
            return int(self.num_skills)
        if self.skill_dim is None:
            raise ValueError("Continuous skill prior is missing skill_dim.")
        return int(self.skill_dim)

    def sample(self, batch_shape: tuple[int, ...] = ()):
        raise NotImplementedError

    def log_prob(self, z):
        raise NotImplementedError


@dataclass(frozen=True)
class UniformCategoricalPrior(SkillPrior):
    num_skills: int = 0
    skill_dim: int | None = None
    skill_space: str = "categorical"

    def sample(self, batch_shape: tuple[int, ...] = ()):
        return jax.random.randint(nj.seed(), batch_shape, 0, int(self.num_skills), dtype=jnp.int32)

    def log_prob(self, z):
        z = jnp.asarray(z)
        return jnp.full((*z.shape, 1), -math.log(float(self.num_skills)), jnp.float32)


@dataclass(frozen=True)
class FixedCategoricalPrior(SkillPrior):
    num_skills: int = 0
    skill_dim: int | None = None
    logits: Any = None
    probs: Any = None
    skill_space: str = "categorical"

    def _logits(self):
        if self.probs is not None:
            logits = jnp.log(jnp.asarray(self.probs, jnp.float32) + 1e-8)
        elif self.logits is not None:
            logits = jnp.asarray(self.logits, jnp.float32)
        else:
            logits = jnp.zeros((int(self.num_skills),), jnp.float32)
        if logits.shape[-1] != int(self.num_skills):
            raise ValueError(
                f"FixedCategoricalPrior expected last dim {self.num_skills}, got {logits.shape}."
            )
        return logits

    def sample(self, batch_shape: tuple[int, ...] = ()):
        logits = self._logits()
        if logits.ndim == 1:
            logits = jnp.broadcast_to(logits, (*batch_shape, logits.shape[-1]))
        return jax.random.categorical(nj.seed(), logits, axis=-1).astype(jnp.int32)

    def log_prob(self, z):
        z = jnp.asarray(z).astype(jnp.int32)
        logits = jax.nn.log_softmax(self._logits(), axis=-1)
        return jnp.take(logits, z, axis=-1)[..., None]


@dataclass(frozen=True)
class UniformHyperspherePrior(SkillPrior):
    skill_dim: int = 0
    num_skills: int | None = None
    skill_space: str = "sphere"

    def sample(self, batch_shape: tuple[int, ...] = ()):
        z = jax.random.normal(nj.seed(), (*batch_shape, int(self.skill_dim)), jnp.float32)
        return z / jnp.maximum(jnp.linalg.norm(z, axis=-1, keepdims=True), 1e-8)

    def log_prob(self, z):
        z = jnp.asarray(z)
        dim = float(self.skill_dim)
        log_area = math.log(2.0) + 0.5 * dim * math.log(math.pi) - math.lgamma(0.5 * dim)
        return jnp.full((*z.shape[:-1], 1), -log_area, jnp.float32)


@dataclass(frozen=True)
class UniformBoxPrior(SkillPrior):
    skill_dim: int = 0
    num_skills: int | None = None
    low: float = -1.0
    high: float = 1.0
    skill_space: str = "box"

    def __post_init__(self):
        if not float(self.low) < float(self.high):
            raise ValueError("uniform_box requires low < high.")

    def sample(self, batch_shape: tuple[int, ...] = ()):
        return jax.random.uniform(
            nj.seed(),
            (*batch_shape, int(self.skill_dim)),
            jnp.float32,
            float(self.low),
            float(self.high),
        )

    def log_prob(self, z):
        z = jnp.asarray(z, jnp.float32)
        inside = jnp.all(
            (z >= float(self.low)) & (z <= float(self.high)), axis=-1
        )
        value = -float(self.skill_dim) * math.log(float(self.high) - float(self.low))
        return jnp.where(
            inside[..., None],
            jnp.full((*z.shape[:-1], 1), value, jnp.float32),
            jnp.full((*z.shape[:-1], 1), -jnp.inf, jnp.float32),
        )


@dataclass(frozen=True)
class FixedHyperspherePrior(SkillPrior):
    skill_dim: int = 0
    skills: Any = None
    num_skills: int | None = None
    skill_space: str = "sphere"

    def _skills(self):
        skills = jnp.asarray(self.skills, jnp.float32)
        if skills.ndim != 2 or skills.shape[-1] != int(self.skill_dim):
            raise ValueError(
                f"fixed_hypersphere expected skills [N,{self.skill_dim}], got {skills.shape}."
            )
        return skills / jnp.maximum(jnp.linalg.norm(skills, axis=-1, keepdims=True), 1e-8)

    def sample(self, batch_shape: tuple[int, ...] = ()):
        skills = self._skills()
        idx = jax.random.randint(nj.seed(), batch_shape, 0, skills.shape[0], dtype=jnp.int32)
        return skills[idx]

    def log_prob(self, z):
        z = jnp.asarray(z)
        n = int(self._skills().shape[0])
        return jnp.full((*z.shape[:-1], 1), -math.log(float(n)), jnp.float32)


def normalize_skill_space(skill_space: str) -> str:
    value = str(skill_space).lower()
    if value in {"continuous", "sphere", "hypersphere"}:
        return "sphere"
    if value in {"box", "cube", "euclidean"}:
        return "box"
    if value == "categorical":
        return "categorical"
    raise ValueError(f"Unknown skill space {skill_space!r}.")


def get_skill_prior(
    name: str,
    num_skills: int | None = None,
    skill_dim: int | None = None,
    **kwargs,
) -> SkillPrior:
    aliases = {
        "uniform": "uniform_categorical",
        "fixed": "fixed_categorical",
        "uniform_sphere": "uniform_hypersphere",
        "uniform_cube": "uniform_box",
        "uniform_euclidean": "uniform_box",
        "fixed_sphere": "fixed_hypersphere",
    }
    key = aliases.get(str(name).lower(), str(name).lower())
    kwargs = _normalize_prior_kwargs(kwargs)
    if "prior_kwargs" in kwargs:
        nested = _normalize_prior_kwargs(kwargs.pop("prior_kwargs"))
        kwargs = {**nested, **kwargs}
    if key == "uniform_categorical":
        if num_skills is None:
            raise ValueError("uniform_categorical requires num_skills.")
        return UniformCategoricalPrior(num_skills=int(num_skills))
    if key == "fixed_categorical":
        if num_skills is None:
            raise ValueError("fixed_categorical requires num_skills.")
        return FixedCategoricalPrior(num_skills=int(num_skills), **kwargs)
    if skill_dim is not None and int(skill_dim) <= 0:
        skill_dim = None
    if skill_dim is None:
        if num_skills is None:
            raise ValueError(f"{key} requires skill_dim or num_skills.")
        skill_dim = int(num_skills)
    if key == "uniform_hypersphere":
        return UniformHyperspherePrior(skill_dim=int(skill_dim), num_skills=num_skills)
    if key == "uniform_box":
        return UniformBoxPrior(
            skill_dim=int(skill_dim), num_skills=num_skills, **kwargs
        )
    if key == "fixed_hypersphere":
        return FixedHyperspherePrior(skill_dim=int(skill_dim), num_skills=num_skills, **kwargs)
    raise ValueError(
        "Unknown skill prior "
        f"{name!r}. Available: uniform_categorical, fixed_categorical, "
        "uniform_hypersphere, uniform_box, fixed_hypersphere."
    )


def skill_event_dim(prior: SkillPrior, num_skills: int, skill_dim: int | None = None) -> int:
    if prior.is_categorical:
        return int(num_skills)
    if skill_dim is not None and int(skill_dim) > 0:
        return int(skill_dim)
    return int(prior.event_dim)


def skill_to_indices(skill, num_skills: int):
    skill = jnp.asarray(skill)
    if skill.ndim > 0 and skill.shape[-1] == int(num_skills):
        return jnp.argmax(skill, axis=-1).astype(jnp.int32)
    if skill.ndim > 0 and skill.shape[-1] == 1:
        skill = jnp.squeeze(skill, axis=-1)
    return skill.astype(jnp.int32)


def skill_to_vectors(
    skill,
    ref,
    skill_space: str,
    num_skills: int,
    skill_dim: int,
    normalize: bool = False,
):
    ref = jnp.asarray(ref)
    skill_space = normalize_skill_space(skill_space)
    skill = jnp.asarray(skill)
    if skill_space == "categorical":
        if skill.ndim > 0 and skill.shape[-1] == int(num_skills):
            out = skill.astype(jnp.float32)
        else:
            out = jax.nn.one_hot(skill_to_indices(skill, num_skills), int(num_skills)).astype(jnp.float32)
    else:
        if skill.ndim == 0:
            skill = skill.reshape((1,))
        if skill.shape[-1] != int(skill_dim):
            raise ValueError(f"Expected skill last dim {skill_dim}, got {skill.shape}.")
        out = skill.astype(jnp.float32)
        if normalize:
            out = out / jnp.maximum(jnp.linalg.norm(out, axis=-1, keepdims=True), 1e-8)
    while out.ndim < ref.ndim:
        if out.shape[:-1] == ref.shape[: out.ndim - 1]:
            out = jnp.expand_dims(out, axis=-2)
        else:
            out = out[None]
    if out.shape[:-1] != ref.shape[:-1]:
        out = jnp.broadcast_to(out, (*ref.shape[:-1], out.shape[-1]))
    return out.astype(ref.dtype)


def prior_log_prob_for_skill(prior: SkillPrior, skill, num_skills: int):
    if prior.is_categorical:
        return prior.log_prob(skill_to_indices(skill, num_skills))
    return prior.log_prob(skill)
