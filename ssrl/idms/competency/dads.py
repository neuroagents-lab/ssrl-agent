"""JAX DADS and reverse-DADS competency modules."""

import math

import jax
import jax.numpy as jnp
import ninjax as nj

from .._runtime import f32
from ..chunking import chunked_rows

from .common import (
    ConditionalDiagonalGaussian,
    SkillPosteriorHead,
    masked_mean,
    skill_event_dim,
    skill_to_vectors,
)
from .skill_priors import get_skill_prior


class DADS(nj.Module):

    obs_dim: int = 0
    num_skills: int = 0
    hidden_dim: int = 512
    n_layers: int = 2
    scale: float = 1.0
    entropy_coef: float = 0.0
    prior_name: str = "uniform_categorical"
    prior_kwargs: dict | None = None
    skill_dim: int | None = None
    use_marginal_model: bool = False
    marginal_samples: int = 100
    min_log_std: float = -5.0
    max_log_std: float = 2.0
    act: str = "relu"
    # Execution bound only; all marginal samples and transitions are retained.
    chunk_size: int = 32

    def __init__(self):
        self._obs_dim = int(self.obs_dim)
        self._num_skills = int(self.num_skills)
        self._scale = float(self.scale)
        self._entropy_coef = float(self.entropy_coef)
        self._marginal_samples = max(1, int(self.marginal_samples))
        self.prior = get_skill_prior(
            self.prior_name,
            num_skills=self._num_skills,
            skill_dim=self.skill_dim,
            **(self.prior_kwargs or {}),
        )
        self._skill_dim = skill_event_dim(self.prior, self._num_skills, self.skill_dim)
        self.dynamics = ConditionalDiagonalGaussian(
            input_dim=self._obs_dim + self._skill_dim,
            target_dim=self._obs_dim,
            hidden_dim=int(self.hidden_dim),
            n_layers=int(self.n_layers),
            min_log_std=float(self.min_log_std),
            max_log_std=float(self.max_log_std),
            act=self.act,
            name="dynamics",
        )
        self.marginal = (
            ConditionalDiagonalGaussian(
                input_dim=self._obs_dim,
                target_dim=self._obs_dim,
                hidden_dim=int(self.hidden_dim),
                n_layers=int(self.n_layers),
                min_log_std=float(self.min_log_std),
                max_log_std=float(self.max_log_std),
                act=self.act,
                name="marginal",
            )
            if bool(self.use_marginal_model)
            else None
        )

    def sample_skills(self, batch_shape=()):
        return self.prior.sample(tuple(batch_shape))

    def _skill_vectors(self, skill, ref):
        if skill is None:
            raise ValueError("Skill must be provided for DADS.")
        return skill_to_vectors(
            skill,
            ref,
            self.prior.skill_space,
            self._num_skills,
            self._skill_dim,
            normalize=self.prior.skill_space == "sphere",
        )

    def _conditional_context(self, obs, skill):
        return jnp.concatenate([f32(obs), self._skill_vectors(skill, obs)], axis=-1)

    def _exact_categorical_log_marginal(self, obs, next_obs):
        obs_flat = f32(obs).reshape((-1, self._obs_dim))
        next_flat = f32(next_obs).reshape((-1, self._obs_dim))
        eye = jnp.eye(self._num_skills, dtype=obs_flat.dtype)
        log_prior = self.prior.log_prob(jnp.arange(self._num_skills, dtype=jnp.int32))[..., 0]

        def evaluate(chunk):
            current, following = chunk
            batch = current.shape[0]
            obs_exp = jnp.broadcast_to(current[:, None], (batch, self._num_skills, self._obs_dim))
            skill_exp = jnp.broadcast_to(eye[None], (batch, self._num_skills, self._skill_dim))
            context = jnp.concatenate([obs_exp, skill_exp], axis=-1)
            target = jnp.broadcast_to(following[:, None], obs_exp.shape)
            log_p = self.dynamics.log_prob(target, context)[..., 0]
            return jax.nn.logsumexp(log_p + log_prior[None], axis=-1, keepdims=True)

        log_marg = chunked_rows(evaluate, (obs_flat, next_flat), self.chunk_size)
        return log_marg.reshape((*obs.shape[:-1], 1))

    def _sampled_log_marginal(self, obs, next_obs):
        obs_flat = f32(obs).reshape((-1, self._obs_dim))
        next_flat = f32(next_obs).reshape((-1, self._obs_dim))
        batch = obs_flat.shape[0]
        samples = self.prior.sample((batch, self._marginal_samples))

        def evaluate(chunk):
            current, following, skills = chunk
            shape = (current.shape[0], self._marginal_samples, self._obs_dim)
            obs_exp = jnp.broadcast_to(current[:, None], shape)
            skill_vec = self._skill_vectors(skills, obs_exp)
            context = jnp.concatenate([obs_exp, skill_vec], axis=-1)
            target = jnp.broadcast_to(following[:, None], shape)
            log_p = self.dynamics.log_prob(target, context)[..., 0]
            return jax.nn.logsumexp(log_p, axis=-1, keepdims=True) - math.log(float(self._marginal_samples))

        log_marg = chunked_rows(
            evaluate, (obs_flat, next_flat, samples), self.chunk_size
        )
        return log_marg.reshape((*obs.shape[:-1], 1))

    def _log_marginal(self, obs, next_obs):
        return (
            self._exact_categorical_log_marginal(obs, next_obs)
            if self.prior.is_categorical
            else self._sampled_log_marginal(obs, next_obs)
        )

    def __call__(self, obs, next_obs, skill):
        return self.dynamics.log_prob(f32(next_obs), self._conditional_context(obs, skill))

    def compute_reward(self, obs, action=None, next_obs=None, skill=None, policy_entropy=None, **kwargs):
        del action, kwargs
        if next_obs is None:
            raise ValueError("next_obs is required to compute DADS reward.")
        log_cond = self(obs, next_obs, skill)
        log_marg = self.marginal.log_prob(next_obs, obs) if self.marginal is not None else self._log_marginal(obs, next_obs)
        reward = self._scale * (log_cond - log_marg)
        if policy_entropy is not None and self._entropy_coef:
            ent = f32(policy_entropy)
            if ent.shape != reward.shape:
                ent = jnp.reshape(ent, reward.shape)
            reward = reward + self._entropy_coef * ent
        return reward

    def loss(self, obs, next_obs, skill, mask=None):
        log_cond = self(obs, next_obs, skill)
        loss_cond = -masked_mean(log_cond[..., 0], mask)
        loss = loss_cond
        if self.marginal is not None:
            log_marg = self.marginal.log_prob(next_obs, obs)
            loss_marg = -masked_mean(log_marg[..., 0], mask)
            loss = loss + loss_marg
        else:
            log_marg = self._log_marginal(obs, next_obs)
            loss_marg = None
        metrics = {
            "dads_loss": loss,
            "dads_cond_nll": loss_cond,
            "dads_mi": masked_mean((log_cond - log_marg)[..., 0], mask),
        }
        if loss_marg is not None:
            metrics["dads_marg_nll"] = loss_marg
        return loss, metrics

    def update(self, obs=None, action=None, next_obs=None, skill=None, optimizer=None, return_loss=False, **kwargs):
        del action
        if optimizer is not None:
            raise NotImplementedError("JAX DADS is updated by the enclosing Ninjax optimizer.")
        loss, metrics = self.loss(obs, next_obs, skill, **kwargs)
        return (loss, metrics) if return_loss else metrics

    def bridge_consistency_targets(self, obs, action=None, next_obs=None, skill=None, **kwargs):
        del action, kwargs
        return {
            "log_p_cond": self(obs, next_obs, skill),
            "log_p_marg": self.marginal.log_prob(next_obs, obs) if self.marginal is not None else self._log_marginal(obs, next_obs),
        }

    def get_modules(self):
        modules = {"dads_dynamics": self.dynamics}
        if self.marginal is not None:
            modules["dads_marginal"] = self.marginal
        return modules


class RDADS(nj.Module):
    """Reverse-DADS estimator for ``I(Z; S' | S)``.

    The reverse conditional-MI identity is

        I(Z; S' | S) = E[log p(z | s, s') - log p(z | s)].

    Both posterior terms must be learned. Replacing the state-conditional
    denominator with the unconditional skill prior would instead optimize
    ``I(Z; (S, S'))`` and reward information about the current state alone.
    """

    obs_dim: int = 0
    num_skills: int = 0
    hidden_dim: int = 256
    n_layers: int = 2
    scale: float = 1.0
    entropy_coef: float = 0.0
    prior_name: str = "uniform_categorical"
    prior_kwargs: dict | None = None
    skill_dim: int | None = None
    posterior_type: str = "dot_product"
    temperature: float = 0.1
    act: str = "relu"

    def __init__(self):
        self._obs_dim = int(self.obs_dim)
        self._num_skills = int(self.num_skills)
        self._scale = float(self.scale)
        self._entropy_coef = float(self.entropy_coef)
        self.prior = get_skill_prior(
            self.prior_name,
            num_skills=self._num_skills,
            skill_dim=self.skill_dim,
            **(self.prior_kwargs or {}),
        )
        posterior_type = str(self.posterior_type).lower()
        if not self.prior.is_categorical and posterior_type != "gaussian":
            raise ValueError(
                "Continuous rDADS requires posterior_type='gaussian' so both "
                "reverse-MI terms are normalized densities. The dot-product "
                "head is a batch-contrastive probability, not q(z | context)."
            )
        normalize_dot = posterior_type == "dot_product" and self.prior.skill_space == "sphere"
        posterior_kwargs = dict(
            num_skills=self._num_skills,
            skill_dim=int(self.skill_dim or 0),
            prior_name=self.prior_name,
            prior_kwargs=self.prior_kwargs,
            hidden_dim=int(self.hidden_dim),
            n_layers=int(self.n_layers),
            posterior_type=posterior_type,
            temperature=float(self.temperature),
            normalize_features=normalize_dot,
            normalize_skills=self.prior.skill_space == "sphere",
            act=self.act,
        )
        self.posterior = SkillPosteriorHead(
            input_dim=2 * self._obs_dim,
            **posterior_kwargs,
            name="posterior",
        )
        self.state_posterior = SkillPosteriorHead(
            input_dim=self._obs_dim,
            **posterior_kwargs,
            name="state_posterior",
        )

    def _context(self, obs, next_obs):
        if next_obs is None:
            raise ValueError("next_obs is required for rDADS.")
        return jnp.concatenate([f32(obs), f32(next_obs)], axis=-1)

    def __call__(self, obs, next_obs):
        return self.posterior(self._context(obs, next_obs))

    def sample_skills(self, batch_shape=()):
        return self.prior.sample(tuple(batch_shape))

    def loss(self, obs, next_obs, skill, mask=None):
        joint_loss, joint_metrics, log_joint = self.posterior.loss(
            self._context(obs, next_obs),
            skill,
            mask,
            return_log_prob=True,
        )
        state_loss, state_metrics, log_state = self.state_posterior.loss(
            f32(obs),
            skill,
            mask,
            return_log_prob=True,
        )
        loss = joint_loss + state_loss
        metrics = {
            "rdads_loss": loss,
            "rdads_joint_nll": joint_loss,
            "rdads_state_nll": state_loss,
            "rdads_mi": masked_mean((log_joint - log_state)[..., 0], mask),
        }
        for key, value in joint_metrics.items():
            if key != "loss":
                metrics[f"rdads_joint_{key}"] = value
        for key, value in state_metrics.items():
            if key != "loss":
                metrics[f"rdads_state_{key}"] = value
        if "acc" in joint_metrics:
            metrics["rdads_acc"] = joint_metrics["acc"]
        return loss, metrics

    def compute_reward(self, obs, action=None, next_obs=None, skill=None, policy_entropy=None, **kwargs):
        del action, kwargs
        if skill is None:
            raise ValueError("Skill must be provided to compute rDADS reward.")
        context = self._context(obs, next_obs)
        log_joint, _ = self.posterior.log_prob(context, skill, return_aux=False)
        log_state, _ = self.state_posterior.log_prob(
            f32(obs), skill, return_aux=False
        )
        reward = self._scale * (log_joint - log_state)
        if policy_entropy is not None and self._entropy_coef:
            ent = f32(policy_entropy)
            if ent.shape != reward.shape:
                ent = jnp.reshape(ent, reward.shape)
            reward = reward + self._entropy_coef * ent
        return reward

    def update(self, obs=None, action=None, next_obs=None, skill=None, optimizer=None, return_loss=False, **kwargs):
        del action
        if optimizer is not None:
            raise NotImplementedError("JAX rDADS is updated by the enclosing Ninjax optimizer.")
        loss, metrics = self.loss(obs, next_obs, skill, **kwargs)
        return (loss, metrics) if return_loss else metrics

    def bridge_consistency_targets(self, obs, action=None, next_obs=None, **kwargs):
        del action, kwargs
        return {
            "joint_posterior": self(obs, next_obs),
            "state_posterior": self.state_posterior(f32(obs)),
        }

    def get_modules(self):
        return {
            "rdads_posterior": self.posterior,
            "rdads_state_posterior": self.state_posterior,
        }
