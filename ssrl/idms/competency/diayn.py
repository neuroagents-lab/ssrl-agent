"""JAX DIAYN competency module."""

import jax.numpy as jnp
import ninjax as nj

from .._runtime import f32
from .common import SkillPosteriorHead, prior_log_prob_for_skill
from .skill_priors import get_skill_prior


class DIAYN(nj.Module):

    obs_dim: int = 0
    num_skills: int = 0
    hidden_dim: int = 300
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
        normalize_dot = str(self.posterior_type).lower() == "dot_product" and self.prior.skill_space == "sphere"
        self.posterior = SkillPosteriorHead(
            input_dim=self._obs_dim,
            num_skills=self._num_skills,
            skill_dim=int(self.skill_dim or 0),
            prior_name=self.prior_name,
            prior_kwargs=self.prior_kwargs,
            hidden_dim=int(self.hidden_dim),
            n_layers=int(self.n_layers),
            posterior_type=self.posterior_type,
            temperature=float(self.temperature),
            normalize_features=normalize_dot,
            normalize_skills=self.prior.skill_space == "sphere",
            act=self.act,
            name="posterior",
        )

    def __call__(self, obs):
        return self.posterior(f32(obs))

    def sample_skills(self, batch_shape=()):
        return self.prior.sample(tuple(batch_shape))

    def loss(self, obs, skill, mask=None):
        loss, metrics = self.posterior.loss(f32(obs), skill, mask)
        return loss, {f"diayn_{key}": value for key, value in metrics.items()}

    def compute_reward(
        self,
        obs,
        action=None,
        next_obs=None,
        skill=None,
        policy_entropy=None,
        **kwargs,
    ):
        del action, next_obs, kwargs
        if skill is None:
            raise ValueError("Skill must be provided to compute DIAYN reward.")
        log_q, _ = self.posterior.log_prob(f32(obs), skill, return_aux=False)
        log_p = prior_log_prob_for_skill(self.prior, skill, self._num_skills).astype(log_q.dtype)
        reward = self._scale * (log_q - log_p)
        if policy_entropy is not None and self._entropy_coef:
            ent = f32(policy_entropy)
            if ent.shape != reward.shape:
                ent = jnp.reshape(ent, reward.shape)
            reward = reward + self._entropy_coef * ent
        return reward

    def update(self, obs=None, action=None, next_obs=None, skill=None, optimizer=None, return_loss=False, **kwargs):
        del action, next_obs, kwargs
        if optimizer is not None:
            raise NotImplementedError("JAX DIAYN is updated by the enclosing Ninjax optimizer.")
        if skill is None:
            raise ValueError("Skill must be provided for DIAYN update.")
        loss, metrics = self.loss(obs, skill)
        return (loss, metrics) if return_loss else metrics

    def bridge_consistency_targets(self, obs, **kwargs):
        del kwargs
        return self(obs)

    def get_modules(self):
        return {"diayn_posterior": self.posterior}
