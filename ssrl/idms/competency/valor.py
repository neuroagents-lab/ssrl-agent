"""JAX VALOR competency module."""

import jax.numpy as jnp
import ninjax as nj

from .._runtime import f32
from .common import MLP, SkillPosteriorHead, masked_mean, prior_log_prob_for_skill
from .skill_priors import get_skill_prior


class VALOR(nj.Module):
    """Whole-trajectory skill posterior with an ordered MLP decoder.

    Encode long-step deltas between equally spaced trajectory observations,
    then decode the ordered features with an MLP. VALOR uses a bidirectional
    LSTM decoder; this implementation uses the Dreamer RSSM representation.
    """

    obs_dim: int = 0
    num_skills: int = 0
    hidden_dim: int = 256
    n_layers: int = 1
    num_subseq: int = 11
    scale: float = 1.0
    entropy_coef: float = 0.0
    prior_name: str = "uniform_categorical"
    prior_kwargs: dict | None = None
    skill_dim: int | None = None
    posterior_type: str = "dot_product"
    temperature: float = 0.1
    subtract_log_prior: bool = True
    reward_at_end: bool = True
    head_hidden_dim: int = 0
    max_segments: int = 0
    act: str = "relu"

    def __init__(self):
        self._obs_dim = int(self.obs_dim)
        self._num_skills = int(self.num_skills)
        self._num_subseq = max(2, int(self.num_subseq))
        self._scale = float(self.scale)
        self._entropy_coef = float(self.entropy_coef)
        self._subtract_log_prior = bool(self.subtract_log_prior)
        self._reward_at_end = bool(self.reward_at_end)
        self.prior = get_skill_prior(
            self.prior_name,
            num_skills=self._num_skills,
            skill_dim=self.skill_dim,
            **(self.prior_kwargs or {}),
        )
        self.delta_encoder = MLP(
            input_dim=self._obs_dim,
            output_dim=int(self.hidden_dim),
            hidden_dim=int(self.hidden_dim),
            layers=1,
            act=self.act,
            name="delta_encoder",
        )
        self.encoder = MLP(
            input_dim=(self._num_subseq - 1) * int(self.hidden_dim),
            output_dim=2 * int(self.hidden_dim),
            hidden_dim=int(self.hidden_dim),
            layers=max(1, int(self.n_layers)),
            act=self.act,
            name="encoder",
        )
        posterior_hidden = int(self.head_hidden_dim or self.hidden_dim)
        normalize_dot = str(self.posterior_type).lower() == "dot_product" and self.prior.skill_space == "sphere"
        self.posterior = SkillPosteriorHead(
            input_dim=2 * int(self.hidden_dim),
            num_skills=self._num_skills,
            skill_dim=int(self.skill_dim or 0),
            prior_name=self.prior_name,
            prior_kwargs=self.prior_kwargs,
            hidden_dim=posterior_hidden,
            n_layers=1,
            posterior_type=self.posterior_type,
            temperature=float(self.temperature),
            normalize_features=normalize_dot,
            normalize_skills=self.prior.skill_space == "sphere",
            act=self.act,
            name="posterior",
        )

    def sample_skills(self, batch_shape=()):
        return self.prior.sample(tuple(batch_shape))

    def _as_batch_major(self, obs):
        obs = f32(obs)
        if obs.ndim == 2:
            obs = obs[:, None, :]
        if obs.ndim != 3:
            raise ValueError(f"VALOR expects obs [B,T,D] or [B,D], got {obs.shape}.")
        return obs

    def _sequence_labels(self, skill, obs):
        skill = jnp.asarray(skill)
        if skill.ndim >= 3 and skill.shape[:2] == obs.shape[:2]:
            return skill[:, 0]
        if skill.ndim >= 2 and skill.shape[0] == obs.shape[0]:
            return skill
        raise ValueError(
            f"VALOR skill labels must be [B,Z] or [B,T,Z], got {skill.shape} "
            f"for trajectories {obs.shape}."
        )

    def encode(self, obs):
        obs = self._as_batch_major(obs)
        b, t, _ = obs.shape
        if t <= 1:
            flat = jnp.zeros(
                (b, (self._num_subseq - 1) * int(self.hidden_dim)),
                obs.dtype,
            )
            return self.encoder(flat)
        idx = jnp.rint(jnp.linspace(0, t - 1, self._num_subseq)).astype(jnp.int32)
        sampled = obs[:, idx]
        deltas = sampled[:, 1:] - sampled[:, :-1]
        encoded_deltas = self.delta_encoder(deltas)
        return self.encoder(encoded_deltas.reshape((b, -1)))

    def __call__(self, obs):
        return self.encode(obs)

    def segment_reward(self, context, labels):
        log_q, _ = self.posterior.log_prob(context, labels, return_aux=False)
        if not self._subtract_log_prior:
            return self._scale * log_q
        log_p = prior_log_prob_for_skill(self.prior, labels, self._num_skills).astype(log_q.dtype)
        return self._scale * (log_q - log_p)

    def trajectory_reward(self, obs, skill):
        obs = self._as_batch_major(obs)
        labels = self._sequence_labels(skill, obs)
        return self.segment_reward(self.encode(obs), labels)

    def loss(self, obs, skill, mask=None):
        obs = self._as_batch_major(obs)
        labels = self._sequence_labels(skill, obs)
        context = self.encode(obs)
        loss, metrics, log_q = self.posterior.loss(
            context,
            labels,
            mask,
            return_log_prob=True,
        )
        log_p = prior_log_prob_for_skill(self.prior, labels, self._num_skills).astype(log_q.dtype)
        if self._subtract_log_prior:
            reward = self._scale * (log_q - log_p)
        else:
            reward = self._scale * log_q
        metrics["segment_reward"] = masked_mean(reward[..., 0], mask)
        segment_count = (
            jnp.asarray(obs.shape[0], jnp.float32)
            if mask is None
            else f32(mask).sum()
        )
        metrics = {
            "valor_num_segments": segment_count,
            "valor_total_segments": jnp.asarray(obs.shape[0], jnp.float32),
            **{f"valor_{key}": value for key, value in metrics.items()},
        }
        return loss, metrics

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
            raise ValueError("Skill must be provided to compute VALOR reward.")
        obs = self._as_batch_major(obs)
        labels = self._sequence_labels(skill, obs)
        seg_reward = self.segment_reward(self.encode(obs), labels)
        if self._reward_at_end:
            zeros = jnp.zeros((*obs.shape[:2], 1), obs.dtype)
            reward = zeros.at[:, -1, :].set(seg_reward.astype(obs.dtype))
        else:
            reward = jnp.broadcast_to(
                seg_reward[:, None] / float(max(obs.shape[1] - 1, 1)),
                (*obs.shape[:2], 1),
            )
        if policy_entropy is not None and self._entropy_coef:
            ent = f32(policy_entropy)
            if ent.shape != reward.shape:
                ent = jnp.reshape(ent, reward.shape)
            reward = reward + self._entropy_coef * ent
        return reward

    def update(self, obs=None, action=None, next_obs=None, skill=None, optimizer=None, return_loss=False, **kwargs):
        del action, next_obs
        if optimizer is not None:
            raise NotImplementedError("JAX VALOR is updated by the enclosing Ninjax optimizer.")
        if skill is None:
            raise ValueError("Skill must be provided for VALOR update.")
        loss, metrics = self.loss(obs, skill, **kwargs)
        return (loss, metrics) if return_loss else metrics

    def get_modules(self):
        return {
            "valor_delta_encoder": self.delta_encoder,
            "valor_encoder": self.encoder,
            "valor_posterior": self.posterior,
        }
