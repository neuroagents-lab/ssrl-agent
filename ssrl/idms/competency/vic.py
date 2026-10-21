"""JAX Variational Intrinsic Control module."""

import jax.numpy as jnp
import ninjax as nj

from .._runtime import f32
from .common import SkillPosteriorHead, masked_mean, prior_log_prob_for_skill
from .skill_priors import get_skill_prior


class VIC(nj.Module):

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
    condition_on_start: bool = True
    reward_at_end: bool = True
    act: str = "relu"

    def __init__(self):
        self._obs_dim = int(self.obs_dim)
        self._num_skills = int(self.num_skills)
        self._scale = float(self.scale)
        self._entropy_coef = float(self.entropy_coef)
        self._condition_on_start = bool(self.condition_on_start)
        self._reward_at_end = bool(self.reward_at_end)
        self.prior = get_skill_prior(
            self.prior_name,
            num_skills=self._num_skills,
            skill_dim=self.skill_dim,
            **(self.prior_kwargs or {}),
        )
        normalize_dot = str(self.posterior_type).lower() == "dot_product" and self.prior.skill_space == "sphere"
        self.posterior = SkillPosteriorHead(
            input_dim=self._obs_dim * (2 if self._condition_on_start else 1),
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

    def _context(self, start_obs, final_obs):
        final_obs = f32(final_obs)
        if not self._condition_on_start:
            return final_obs
        if start_obs is None:
            raise ValueError("start_obs is required when condition_on_start=True.")
        return jnp.concatenate([f32(start_obs), final_obs], axis=-1)

    def __call__(self, start_obs, final_obs):
        return self.posterior(self._context(start_obs, final_obs))

    def sample_skills(self, batch_shape=()):
        return self.prior.sample(tuple(batch_shape))

    def _trajectory_labels(self, skill, trajectory):
        skill = jnp.asarray(skill)
        if skill.ndim == trajectory.ndim and skill.shape[:2] == trajectory.shape[:2]:
            return skill[:, 0]
        if skill.ndim >= 2 and skill.shape[0] == trajectory.shape[0]:
            return skill
        raise ValueError(
            f"VIC skill labels must be [B,Z] or [B,T,Z], got {skill.shape} "
            f"for trajectory {trajectory.shape}."
        )

    def segment_reward(self, start_obs, final_obs, skill):
        context = self._context(start_obs, final_obs)
        log_q, _ = self.posterior.log_prob(context, skill, return_aux=False)
        log_p = prior_log_prob_for_skill(self.prior, skill, self._num_skills).astype(log_q.dtype)
        return self._scale * (log_q - log_p)

    def loss(self, start_obs, final_obs, skill, mask=None):
        context = self._context(start_obs, final_obs)
        loss, metrics, log_q = self.posterior.loss(
            context,
            skill,
            mask,
            return_log_prob=True,
        )
        log_p = prior_log_prob_for_skill(self.prior, skill, self._num_skills).astype(log_q.dtype)
        metrics["segment_reward"] = masked_mean(
            (self._scale * (log_q - log_p))[..., 0],
            mask,
        )
        return loss, {f"vic_{key}": value for key, value in metrics.items()}

    def compute_reward(
        self,
        trajectory,
        action=None,
        next_obs=None,
        skill=None,
        policy_entropy=None,
        **kwargs,
    ):
        del action, next_obs, kwargs
        if skill is None:
            raise ValueError("Skill must be provided to compute VIC reward.")
        trajectory = f32(trajectory)
        if trajectory.ndim != 3 or trajectory.shape[1] < 2:
            raise ValueError(
                f"VIC reward requires complete trajectories [B,T,D] with T>=2, got {trajectory.shape}."
            )
        labels = self._trajectory_labels(skill, trajectory)
        segment_reward = self.segment_reward(trajectory[:, 0], trajectory[:, -1], labels)
        if self._reward_at_end:
            reward = jnp.zeros((*trajectory.shape[:2], 1), trajectory.dtype)
            reward = reward.at[:, -1].set(segment_reward.astype(trajectory.dtype))
        else:
            reward = jnp.broadcast_to(
                segment_reward[:, None] / float(trajectory.shape[1] - 1),
                (*trajectory.shape[:2], 1),
            )
        if policy_entropy is not None and self._entropy_coef:
            ent = f32(policy_entropy)
            if ent.shape != reward.shape:
                ent = jnp.reshape(ent, reward.shape)
            reward = reward + self._entropy_coef * ent
        return reward

    def update(self, obs=None, action=None, next_obs=None, skill=None, optimizer=None, return_loss=False, **kwargs):
        del action
        if optimizer is not None:
            raise NotImplementedError("JAX VIC is updated by the enclosing Ninjax optimizer.")
        if skill is None:
            raise ValueError("Skill must be provided for VIC update.")
        if obs is None or next_obs is None:
            raise ValueError("VIC update requires option start and final observations.")
        loss, metrics = self.loss(obs, next_obs, skill, **kwargs)
        return (loss, metrics) if return_loss else metrics

    def bridge_consistency_targets(self, start_obs, final_obs, **kwargs):
        del kwargs
        return self(start_obs, final_obs)

    def get_modules(self):
        return {"vic_posterior": self.posterior}
