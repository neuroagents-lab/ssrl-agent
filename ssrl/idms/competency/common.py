"""Shared JAX competency IDM helpers."""

import math

import jax
import jax.numpy as jnp
import ninjax as nj

from .._runtime import act, f32
from ..curiosity.common import MLP, masked_mean, softmax_cross_entropy, xavier_uniform
from .skill_priors import (
    SkillPrior,
    get_skill_prior,
    normalize_skill_space,
    prior_log_prob_for_skill,
    skill_event_dim,
    skill_to_indices,
    skill_to_vectors,
)

LOG_2PI = math.log(2.0 * math.pi)


def infer_skill_starts(skill, is_first, explicit_starts=None, atol=1e-6):
    """Return exact option-start markers for batch-major skill sequences.

    Explicit markers are authoritative and preserve boundaries when a resampled
    categorical skill happens to equal the previous one. Skill changes and
    episode starts also mark boundaries.
    """
    skill = f32(skill)
    is_first = jnp.asarray(is_first, bool)
    if skill.ndim != 3:
        raise ValueError(f"Skill starts expect skill [B,T,Z], got {skill.shape}.")
    if is_first.shape != skill.shape[:2]:
        raise ValueError(
            f"is_first must match skill leading shape {skill.shape[:2]}, got {is_first.shape}."
        )
    changed = jnp.any(jnp.abs(skill[:, 1:] - skill[:, :-1]) > float(atol), axis=-1)
    changed = jnp.concatenate([jnp.zeros_like(is_first[:, :1]), changed], axis=1)
    starts = is_first | changed
    if explicit_starts is not None:
        explicit_starts = jnp.asarray(explicit_starts, bool)
        if explicit_starts.shape != is_first.shape:
            raise ValueError(
                f"explicit_starts must match {is_first.shape}, got {explicit_starts.shape}."
            )
        starts = starts | explicit_starts
    return starts


def fixed_horizon_skill_segments(
    states,
    skill,
    skill_starts,
    is_first,
    is_last,
    horizon,
):
    """Extract all complete fixed-horizon option trajectories from replay.

    The result has a static leading size ``B * max(1, (T - 1) // horizon)`` for
    JAX compilation. Only complete, non-overlapping segments can fit in that
    bound. ``valid`` distinguishes real segments from padding when a replay
    sequence contains fewer complete options.
    """
    states = f32(states)
    skill = f32(skill)
    is_first = jnp.asarray(is_first, bool)
    is_last = jnp.asarray(is_last, bool)
    horizon = int(horizon)
    if states.ndim != 3 or skill.ndim != 3:
        raise ValueError(
            f"Skill segments expect states [B,T,D] and skill [B,T,Z], got "
            f"{states.shape} and {skill.shape}."
        )
    if states.shape[:2] != skill.shape[:2]:
        raise ValueError("State and skill sequences must have matching leading shapes.")
    if is_first.shape != states.shape[:2] or is_last.shape != states.shape[:2]:
        raise ValueError("Episode flags must match the state sequence leading shape.")
    if horizon <= 0:
        raise ValueError(f"Skill segment horizon must be positive, got {horizon}.")

    batch, length = states.shape[:2]
    if length <= 0:
        raise ValueError("Skill segment sequences must contain at least one state.")
    starts = infer_skill_starts(skill, is_first, skill_starts)
    candidates = jnp.arange(length, dtype=jnp.int32)
    state_offsets = jnp.arange(horizon + 1, dtype=jnp.int32)
    state_indices = candidates[:, None] + state_offsets[None]
    clipped_state_indices = jnp.minimum(state_indices, max(length - 1, 0))
    batch_indices = jnp.arange(batch, dtype=jnp.int32)[:, None, None]

    depart_offsets = jnp.arange(horizon, dtype=jnp.int32)
    depart_indices = candidates[:, None] + depart_offsets[None]
    clipped_depart_indices = jnp.minimum(depart_indices, max(length - 1, 0))
    depart_skill = skill[batch_indices, clipped_depart_indices[None]]
    labels = skill

    gathered_first = is_first[batch_indices, clipped_state_indices[None]]
    gathered_last = is_last[batch_indices, clipped_state_indices[None]]
    gathered_starts = starts[batch_indices, clipped_state_indices[None]]
    stable_skill = jnp.all(
        jnp.abs(depart_skill - labels[:, :, None]) <= 1e-6,
        axis=(-1, -2),
    )
    within = (candidates + horizon < length)[None]
    crosses_episode = jnp.any(gathered_first[..., 1:], axis=-1)
    ends_early = jnp.any(gathered_last[..., :-1], axis=-1)
    has_inner_start = jnp.any(gathered_starts[..., 1:-1], axis=-1)
    has_complete_end = gathered_starts[..., -1] | gathered_last[..., -1]
    valid = (
        starts
        & within
        & stable_skill
        & ~crosses_episode
        & ~ends_early
        & ~has_inner_start
        & has_complete_end
    )
    max_segments = max(1, (length - 1) // horizon)
    selected_starts = jax.vmap(
        lambda row: jnp.nonzero(row, size=max_segments, fill_value=0)[0]
    )(valid)
    selected_valid = jnp.take_along_axis(valid, selected_starts, axis=1)
    selected_state_indices = jnp.minimum(
        selected_starts[..., None] + state_offsets,
        length - 1,
    )
    selected_batch = jnp.arange(batch, dtype=jnp.int32)[:, None]
    trajectories = states[selected_batch[..., None], selected_state_indices]
    labels = skill[selected_batch, selected_starts]
    return (
        trajectories.reshape((-1, horizon + 1, states.shape[-1])),
        labels.reshape((-1, skill.shape[-1])),
        selected_valid.reshape((-1,)),
    )


def diag_gaussian_log_prob(target, mean, log_std):
    target, mean, log_std = f32(target), f32(mean), f32(log_std)
    inv_var = jnp.exp(-2.0 * log_std)
    log_prob = -0.5 * (jnp.square(target - mean) * inv_var + 2.0 * log_std + LOG_2PI)
    return log_prob.sum(axis=-1, keepdims=True)


def normalize_rows(x, eps=1e-8):
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


class Linear(nj.Module):
    input_dim: int = 0
    output_dim: int = 0
    bias: bool = True

    def __init__(self):
        self._input_dim = int(self.input_dim)
        self._output_dim = int(self.output_dim)
        if self._input_dim <= 0 or self._output_dim <= 0:
            raise ValueError("Linear input_dim and output_dim must be positive.")

    def __call__(self, x, stop_params: bool = False):
        weight = self.value(
            "w",
            lambda shape: xavier_uniform(shape, self._input_dim, self._output_dim),
            (self._input_dim, self._output_dim),
        )
        if stop_params:
            weight = jax.lax.stop_gradient(weight)
        out = jnp.einsum("...i,io->...o", f32(x), weight)
        if self.bias:
            bias = self.value("b", jnp.zeros, (self._output_dim,), jnp.float32)
            if stop_params:
                bias = jax.lax.stop_gradient(bias)
            out = out + bias
        return f32(out)


class ConditionalDiagonalGaussian(nj.Module):
    input_dim: int = 0
    target_dim: int = 0
    hidden_dim: int = 256
    n_layers: int = 2
    min_log_std: float = -5.0
    max_log_std: float = 2.0
    hidden_condition_dim: int = 0
    act: str = "relu"

    def __init__(self):
        self._input_dim = int(self.input_dim)
        self._target_dim = int(self.target_dim)
        self._hidden_dim = int(self.hidden_dim)
        self._layers = max(1, int(self.n_layers))
        self._hidden_condition_dim = int(self.hidden_condition_dim)
        if self._input_dim <= 0 or self._target_dim <= 0:
            raise ValueError("Conditional Gaussian dimensions must be positive.")

    def _weight(self, name, in_dim, out_dim):
        return self.value(
            name,
            lambda shape: xavier_uniform(shape, in_dim, out_dim),
            (in_dim, out_dim),
        )

    def _bias(self, name, out_dim):
        return self.value(name, jnp.zeros, (out_dim,), jnp.float32)

    def __call__(self, x, hidden_condition=None, stop_params: bool = False):
        x = f32(x)
        if self._hidden_condition_dim > 0:
            if hidden_condition is None:
                raise ValueError("hidden_condition is required for repeated conditioning.")
            hidden_condition = f32(hidden_condition)
            if hidden_condition.shape[-1] != self._hidden_condition_dim:
                raise ValueError(
                    "hidden_condition has wrong last dim: "
                    f"expected {self._hidden_condition_dim}, got {hidden_condition.shape[-1]}."
                )
        in_dim = self._input_dim
        for index in range(self._layers):
            weight = self._weight(f"w{index}", in_dim, self._hidden_dim)
            bias = self._bias(f"b{index}", self._hidden_dim)
            if stop_params:
                weight = jax.lax.stop_gradient(weight)
                bias = jax.lax.stop_gradient(bias)
            hidden = act(self.act)(jnp.einsum("...i,io->...o", x, weight) + bias)
            if self._hidden_condition_dim > 0:
                x = jnp.concatenate([hidden, hidden_condition], axis=-1)
                in_dim = self._hidden_dim + self._hidden_condition_dim
            else:
                x = hidden
                in_dim = self._hidden_dim
        weight = self._weight("wout", in_dim, 2 * self._target_dim)
        bias = self._bias("bout", 2 * self._target_dim)
        if stop_params:
            weight = jax.lax.stop_gradient(weight)
            bias = jax.lax.stop_gradient(bias)
        params = jnp.einsum("...i,io->...o", x, weight) + bias
        mean, log_std = jnp.split(params, 2, axis=-1)
        log_std = jnp.clip(log_std, float(self.min_log_std), float(self.max_log_std))
        return f32(mean), f32(log_std)

    def log_prob(self, target, context, hidden_condition=None, stop_params: bool = False):
        mean, log_std = self(context, hidden_condition=hidden_condition, stop_params=stop_params)
        return diag_gaussian_log_prob(target, mean, log_std)


class PotentialActionHead(nj.Module):
    obs_dim: int = 0
    action_dim: int = 0
    particles: int = 16
    hidden_dim: int = 256
    n_layers: int = 2
    act: str = "relu"

    def __init__(self):
        self._obs_dim = int(self.obs_dim)
        self._action_dim = int(self.action_dim)
        self._particles = int(self.particles)
        if self._obs_dim <= 0 or self._action_dim <= 0 or self._particles <= 0:
            raise ValueError("PotentialActionHead requires positive dimensions.")
        self.net = MLP(
            input_dim=self._obs_dim,
            output_dim=self._particles * self._action_dim + self._particles,
            hidden_dim=int(self.hidden_dim),
            layers=int(self.n_layers),
            act=self.act,
            name="net",
        )

    def __call__(self, obs):
        out = self.net(f32(obs))
        raw, logits = jnp.split(out, [self._particles * self._action_dim], axis=-1)
        actions = jnp.tanh(raw.reshape((*obs.shape[:-1], self._particles, self._action_dim)))
        return f32(actions), f32(logits)


class SkillPosteriorHead(nj.Module):
    input_dim: int = 0
    num_skills: int = 0
    skill_dim: int = 0
    prior_name: str = "uniform_categorical"
    prior_kwargs: dict | None = None
    hidden_dim: int = 256
    n_layers: int = 2
    posterior_type: str = "dot_product"
    temperature: float = 0.1
    normalize_features: bool = True
    normalize_skills: bool = True
    act: str = "relu"

    def __init__(self):
        self._input_dim = int(self.input_dim)
        self._num_skills = int(self.num_skills)
        self.prior: SkillPrior = get_skill_prior(
            self.prior_name,
            num_skills=self._num_skills,
            skill_dim=self.skill_dim,
            **(self.prior_kwargs or {}),
        )
        self._skill_space = normalize_skill_space(self.prior.skill_space)
        self._skill_dim = skill_event_dim(self.prior, self._num_skills, self.skill_dim)
        self._posterior_type = str(self.posterior_type).lower()
        self._temperature = float(max(self.temperature, 1e-6))
        if self._skill_space == "categorical":
            out_dim = self._num_skills
        elif self._posterior_type == "dot_product":
            out_dim = self._skill_dim
        elif self._posterior_type == "gaussian":
            out_dim = 2 * self._skill_dim
        else:
            raise ValueError(f"Unknown posterior_type {self._posterior_type!r}.")
        self.net = MLP(
            input_dim=self._input_dim,
            output_dim=out_dim,
            hidden_dim=int(self.hidden_dim),
            layers=int(self.n_layers),
            act=self.act,
            name="net",
        )

    @property
    def skill_dim_value(self) -> int:
        return self._skill_dim

    def __call__(self, context):
        out = self.net(context)
        if self._skill_space != "categorical" and self._posterior_type == "gaussian":
            mean, log_std = jnp.split(out, 2, axis=-1)
            return mean, jnp.clip(log_std, -5.0, 2.0)
        return out

    def log_prob(self, context, skill, return_aux: bool = True):
        if self._skill_space == "categorical":
            return self._categorical(context, skill, return_aux)
        if self._posterior_type == "dot_product":
            return self._dot_product(context, skill, return_aux)
        return self._gaussian(context, skill, return_aux)

    def _categorical(self, context, skill, return_aux):
        logits = self.net(context)
        idx = skill_to_indices(skill, self._num_skills)
        while idx.ndim < logits.ndim - 1:
            idx = idx[None]
        if idx.shape != logits.shape[:-1]:
            idx = jnp.broadcast_to(idx, logits.shape[:-1])
        log_q = jax.nn.log_softmax(logits, axis=-1)
        log_q_z = jnp.take_along_axis(log_q, idx[..., None], axis=-1)
        if not return_aux:
            return log_q_z, {}
        pred = jnp.argmax(logits, axis=-1)
        correct = (pred == idx).astype(jnp.float32)
        return log_q_z, {
            "logits": logits,
            "skill_idx": idx,
            "acc": correct.mean(),
            "correct": correct,
            "pos_score": jnp.take_along_axis(logits, idx[..., None], axis=-1),
        }

    def _dot_product(self, context, skill, return_aux):
        pred = self.net(context)
        if self.normalize_features:
            pred = normalize_rows(pred)
        skill_vec = skill_to_vectors(
            skill,
            pred,
            self._skill_space,
            self._num_skills,
            self._skill_dim,
            normalize=bool(self.normalize_skills),
        )
        flat_pred = pred.reshape((-1, pred.shape[-1]))
        flat_skill = skill_vec.reshape((-1, skill_vec.shape[-1]))
        scores = (flat_pred @ flat_skill.T) / self._temperature
        log_q = jax.nn.log_softmax(scores, axis=-1)
        diag = jnp.arange(flat_pred.shape[0], dtype=jnp.int32)
        log_q_z = log_q[diag, diag].reshape((*pred.shape[:-1], 1))
        if not return_aux:
            return log_q_z, {}
        pos = scores[diag, diag].reshape((*pred.shape[:-1], 1))
        return log_q_z, {
            "pred_skill": pred,
            "skill_vec": skill_vec,
            "pos_score": pos,
            "cosine": (pred * skill_vec).sum(axis=-1).mean(),
        }

    def _gaussian(self, context, skill, return_aux):
        mean, log_std = self(context)
        skill_vec = skill_to_vectors(
            skill,
            mean,
            self._skill_space,
            self._num_skills,
            self._skill_dim,
            normalize=False,
        )
        log_q_z = diag_gaussian_log_prob(skill_vec, mean, log_std)
        if not return_aux:
            return log_q_z, {}
        return log_q_z, {
            "pred_skill": mean,
            "skill_vec": skill_vec,
            "pos_score": (mean * skill_vec).sum(axis=-1, keepdims=True),
            "mse": jnp.square(mean - skill_vec).mean(),
        }

    def positive_score(self, context, skill):
        _, aux = self.log_prob(context, skill)
        return aux["pos_score"]

    def loss(self, context, skill, mask=None, return_log_prob=False):
        log_q_z, aux = self.log_prob(context, skill)
        loss = -masked_mean(log_q_z[..., 0], mask)
        metrics = {"loss": loss}
        if "correct" in aux:
            metrics["acc"] = masked_mean(aux["correct"], mask)
        elif "acc" in aux:
            metrics["acc"] = aux["acc"]
        if "cosine" in aux:
            metrics["cosine"] = aux["cosine"]
        if "mse" in aux:
            metrics["mse"] = aux["mse"]
        if return_log_prob:
            return loss, metrics, log_q_z
        return loss, metrics


__all__ = [
    "ConditionalDiagonalGaussian",
    "Linear",
    "PotentialActionHead",
    "SkillPosteriorHead",
    "diag_gaussian_log_prob",
    "fixed_horizon_skill_segments",
    "get_skill_prior",
    "infer_skill_starts",
    "masked_mean",
    "normalize_rows",
    "normalize_skill_space",
    "prior_log_prob_for_skill",
    "skill_event_dim",
    "skill_to_indices",
    "skill_to_vectors",
    "softmax_cross_entropy",
]
