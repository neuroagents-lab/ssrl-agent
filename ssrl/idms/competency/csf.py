"""JAX Contrastive Skill Features module."""

import math

import jax
import jax.numpy as jnp
import ninjax as nj

from .._runtime import f32
from ..curiosity.common import MLP, masked_mean, xavier_uniform
from .common import Linear, diag_gaussian_log_prob, normalize_rows, prior_log_prob_for_skill
from .skill_priors import get_skill_prior, skill_to_indices, skill_to_vectors


def _normalize_input_norm_type(value):
    norm = str(value).lower()
    if norm in {"", "none", "identity", "false"}:
        return "none"
    if norm in {"layernorm", "layer_norm", "ln"}:
        return "layernorm"
    raise ValueError(f"Unknown CSF input norm {value!r}.")


class CSF(nj.Module):

    obs_dim: int = 0
    num_skills: int = 0
    proj_dim: int = 0
    proj_hidden_dim: int = 1024
    proj_layers: int = 2
    scale: float = 1.0
    entropy_coef: float = 0.0
    prior_name: str = "uniform_categorical"
    prior_kwargs: dict | None = None
    skill_dim: int | None = None
    reward_mode: str = "inner"
    posterior_type: str = "dot_product"
    temperature: float = 1.0
    normalize_features: bool = False
    normalize_skills: bool = False
    input_norm_type: str = "none"
    input_norm_eps: float = 1e-3
    xi: float = 5.0
    num_negatives: int = 256
    exclude_positive: bool = True
    learn_projector: bool = True
    sample_prior_negatives: bool = False
    subsample_batch_loss: bool = True
    act: str = "relu"

    def __init__(self):
        self._obs_dim = int(self.obs_dim)
        self._num_skills = int(self.num_skills)
        self._scale = float(self.scale)
        self._entropy_coef = float(self.entropy_coef)
        self._reward_mode = str(self.reward_mode).lower()
        self._posterior_type = str(self.posterior_type).lower()
        self._temperature = float(max(self.temperature, 1e-6))
        self._normalize_features = bool(self.normalize_features)
        self._normalize_skills = bool(self.normalize_skills)
        self._input_norm_type = _normalize_input_norm_type(self.input_norm_type)
        self._input_norm_eps = float(max(self.input_norm_eps, 1e-8))
        self._xi = float(self.xi)
        self._exclude_positive = bool(self.exclude_positive)
        self._sample_prior_negatives = bool(self.sample_prior_negatives)
        if self._reward_mode not in {"inner", "posterior"}:
            raise ValueError("CSF reward_mode must be 'inner' or 'posterior'.")
        self.prior = get_skill_prior(
            self.prior_name,
            num_skills=self._num_skills,
            skill_dim=self.skill_dim,
            **(self.prior_kwargs or {}),
        )
        self._skill_dim = int(self.prior.event_dim)
        self._proj_dim = int(self.proj_dim or (self._num_skills if self.prior.is_categorical else self._skill_dim))
        if not bool(self.learn_projector) and self._proj_dim != self._obs_dim:
            raise ValueError("CSF learn_projector=False requires obs_dim == proj_dim.")
        self.projector = (
            MLP(
                input_dim=self._obs_dim,
                output_dim=self._proj_dim,
                hidden_dim=int(self.proj_hidden_dim),
                layers=max(1, int(self.proj_layers)),
                act=self.act,
                name="projector",
            )
            if bool(self.learn_projector)
            else None
        )
        self.skill_projection = None
        self.gaussian_head = None
        if not self.prior.is_categorical:
            if self._posterior_type == "dot_product":
                self.skill_projection = (
                    None
                    if self._skill_dim == self._proj_dim
                    else Linear(
                        input_dim=self._skill_dim,
                        output_dim=self._proj_dim,
                        bias=False,
                        name="skill_projection",
                    )
                )
            elif self._posterior_type == "gaussian":
                self.gaussian_head = MLP(
                    input_dim=self._proj_dim,
                    output_dim=2 * self._skill_dim,
                    hidden_dim=int(self.proj_hidden_dim),
                    layers=max(1, int(self.proj_layers)),
                    act=self.act,
                    name="gaussian_head",
                )
            else:
                raise ValueError("CSF posterior_type must be 'dot_product' or 'gaussian'.")

    @property
    def proj_dim_value(self):
        return self._proj_dim

    def sample_skills(self, batch_shape=()):
        return self.prior.sample(tuple(batch_shape))

    def _normalize_inputs(self, obs):
        obs = f32(obs)
        if self._input_norm_type != "layernorm":
            return obs
        mean = obs.mean(axis=-1, keepdims=True)
        var = jnp.square(obs - mean).mean(axis=-1, keepdims=True)
        return (obs - mean) / jnp.sqrt(var + self._input_norm_eps)

    def project(self, obs, stop_params=False):
        obs = self._normalize_inputs(obs)
        proj = obs if self.projector is None else self.projector(obs, stop_params=stop_params)
        if self._normalize_features:
            proj = normalize_rows(proj)
        return f32(proj)

    def project_pair(self, obs, next_obs):
        if obs.shape == next_obs.shape:
            pair = jnp.concatenate([f32(obs), f32(next_obs)], axis=0)
            phi = self.project(pair)
            return jnp.split(phi, 2, axis=0)
        return self.project(obs), self.project(next_obs)

    def transition_features(self, obs, next_obs):
        phi, next_phi = self.project_pair(obs, next_obs)
        delta = next_phi - phi
        if self._normalize_features:
            delta = normalize_rows(delta)
        return f32(delta)

    def _skill_embeddings(self):
        return self.value(
            "skill_embeddings",
            lambda shape: 0.02 * jax.random.normal(nj.seed(), shape, jnp.float32),
            (self._num_skills, self._proj_dim),
        )

    def _categorical_logits_and_idx(self, transition, skill):
        embeddings = self._skill_embeddings()
        logits = (f32(transition) @ embeddings.T) / self._temperature
        idx = skill_to_indices(skill, self._num_skills)
        while idx.ndim < logits.ndim - 1:
            idx = idx[None]
        if idx.shape != logits.shape[:-1]:
            idx = jnp.broadcast_to(idx, logits.shape[:-1])
        return logits, idx

    def _categorical_positive_score(self, transition, skill):
        logits, idx = self._categorical_logits_and_idx(transition, skill)
        return jnp.take_along_axis(logits, idx[..., None], axis=-1)

    def _categorical_log_prob_and_aux(self, transition, skill, return_aux=True):
        logits, idx = self._categorical_logits_and_idx(transition, skill)
        pos = jnp.take_along_axis(logits, idx[..., None], axis=-1)
        log_prior = self.prior.log_prob(jnp.arange(self._num_skills, dtype=jnp.int32))[..., 0]
        prior_logits = jnp.broadcast_to(log_prior, logits.shape)
        denom = jax.nn.logsumexp(logits + prior_logits, axis=-1, keepdims=True)
        picked_prior = jnp.take_along_axis(prior_logits, idx[..., None], axis=-1)
        log_q = pos + picked_prior - denom
        if self._exclude_positive and self._num_skills > 1:
            neg_mask = jnp.arange(self._num_skills, dtype=idx.dtype) != idx[..., None]
            masked_scores = jnp.where(neg_mask, logits + prior_logits, -jnp.inf)
            masked_prior = jnp.where(neg_mask, prior_logits, -jnp.inf)
            neg_term = (
                jax.nn.logsumexp(masked_scores, axis=-1, keepdims=True)
                - jax.nn.logsumexp(masked_prior, axis=-1, keepdims=True)
            )
        else:
            neg_term = denom
        if not return_aux:
            return log_q, {}
        objective = pos - self._xi * neg_term
        return log_q, {
            "logits": logits,
            "pos_score": pos,
            "neg_term": neg_term,
            "objective": objective,
            "acc": (jnp.argmax(logits, axis=-1) == idx).astype(jnp.float32).mean(),
        }

    def _continuous_dot_terms(self, transition, skill):
        skill_vec = skill_to_vectors(
            skill,
            transition,
            self.prior.skill_space,
            self._num_skills,
            self._skill_dim,
            normalize=self._normalize_skills,
        )
        skill_feat = skill_vec if self.skill_projection is None else self.skill_projection(skill_vec)
        if self._normalize_skills:
            skill_feat = normalize_rows(skill_feat)
        flat_transition = transition.reshape((-1, transition.shape[-1]))
        flat_skill = skill_feat.reshape((-1, skill_feat.shape[-1]))
        pos = (flat_transition * flat_skill).sum(axis=-1, keepdims=True) / self._temperature
        return pos, skill_feat, flat_transition, flat_skill

    def _continuous_dot_positive_score(self, transition, skill):
        pos, _, _, _ = self._continuous_dot_terms(transition, skill)
        return pos.reshape((*transition.shape[:-1], 1))

    def _sample_prior_negative_features(self, ref):
        if int(self.num_negatives) <= 0:
            return jnp.zeros((0, self._proj_dim), ref.dtype)
        sampled = self.prior.sample((int(self.num_negatives),))
        sampled_vec = skill_to_vectors(
            sampled,
            jnp.zeros((int(self.num_negatives), self._skill_dim), ref.dtype),
            self.prior.skill_space,
            self._num_skills,
            self._skill_dim,
            normalize=self._normalize_skills,
        )
        feat = sampled_vec if self.skill_projection is None else self.skill_projection(sampled_vec)
        return normalize_rows(feat) if self._normalize_skills else feat

    def _continuous_dot_log_prob_and_aux(self, transition, skill, return_aux=True):
        pos, skill_feat, flat_transition, flat_skill = self._continuous_dot_terms(transition, skill)
        if self._sample_prior_negatives:
            neg_feat = self._sample_prior_negative_features(flat_transition)
            neg_scores = (flat_transition @ neg_feat.T) / self._temperature
            neg_term = jax.nn.logsumexp(neg_scores, axis=-1, keepdims=True)
        else:
            scores = (flat_transition @ flat_skill.T) / self._temperature
            n = scores.shape[-1]
            if self._exclude_positive and n > 1:
                diag = jnp.eye(n, dtype=bool)
                neg_scores = jnp.where(diag, -jnp.inf, scores)
                neg_term = jax.nn.logsumexp(neg_scores, axis=-1, keepdims=True) - math.log(float(n - 1))
            else:
                neg_term = jax.nn.logsumexp(scores, axis=-1, keepdims=True) - math.log(float(n))
        log_p = prior_log_prob_for_skill(self.prior, skill, self._num_skills).astype(transition.dtype).reshape((-1, 1))
        log_q = pos + log_p - neg_term
        if not return_aux:
            return log_q.reshape((*transition.shape[:-1], 1)), {}
        objective = pos - self._xi * neg_term
        return log_q.reshape((*transition.shape[:-1], 1)), {
            "pos_score": pos.reshape((*transition.shape[:-1], 1)),
            "neg_term": neg_term.reshape((*transition.shape[:-1], 1)),
            "objective": objective.reshape((*transition.shape[:-1], 1)),
            "cosine": (transition * skill_feat).sum(axis=-1).mean(),
            "skill_feat": skill_feat,
        }

    def _continuous_gaussian_positive_score(self, transition, skill):
        params = self.gaussian_head(transition)
        mean, _ = jnp.split(params, 2, axis=-1)
        skill_vec = skill_to_vectors(skill, mean, self.prior.skill_space, self._num_skills, self._skill_dim)
        return (mean * skill_vec).sum(axis=-1, keepdims=True)

    def _continuous_gaussian_log_prob_and_aux(self, transition, skill, return_aux=True):
        params = self.gaussian_head(transition)
        mean, log_std = jnp.split(params, 2, axis=-1)
        log_std = jnp.clip(log_std, -5.0, 2.0)
        skill_vec = skill_to_vectors(skill, mean, self.prior.skill_space, self._num_skills, self._skill_dim)
        log_q = diag_gaussian_log_prob(skill_vec, mean, log_std)
        if not return_aux:
            return log_q, {}
        return log_q, {
            "pos_score": (mean * skill_vec).sum(axis=-1, keepdims=True),
            "mse": jnp.square(mean - skill_vec).mean(),
        }

    def _log_prob_and_aux_from_transition(self, transition, skill, return_aux=True):
        if self.prior.is_categorical:
            return self._categorical_log_prob_and_aux(transition, skill, return_aux)
        if self._posterior_type == "dot_product":
            return self._continuous_dot_log_prob_and_aux(transition, skill, return_aux)
        return self._continuous_gaussian_log_prob_and_aux(transition, skill, return_aux)

    def _positive_score_from_transition(self, transition, skill):
        if self.prior.is_categorical:
            return self._categorical_positive_score(transition, skill)
        if self._posterior_type == "dot_product":
            return self._continuous_dot_positive_score(transition, skill)
        return self._continuous_gaussian_positive_score(transition, skill)

    def log_prob(self, obs, next_obs, skill):
        log_q, _ = self._log_prob_and_aux_from_transition(
            self.transition_features(obs, next_obs),
            skill,
            return_aux=False,
        )
        return log_q

    def positive_score(self, obs, next_obs, skill):
        return self._positive_score_from_transition(self.transition_features(obs, next_obs), skill)

    def reward_from_transition(self, transition, skill, policy_entropy=None):
        if self._reward_mode == "inner":
            reward = self._positive_score_from_transition(transition, skill)
        else:
            log_q, _ = self._log_prob_and_aux_from_transition(transition, skill, return_aux=False)
            log_p = prior_log_prob_for_skill(self.prior, skill, self._num_skills).astype(log_q.dtype)
            reward = log_q - log_p
        reward = self._scale * reward
        if policy_entropy is not None and self._entropy_coef:
            ent = f32(policy_entropy)
            if ent.shape != reward.shape:
                ent = jnp.reshape(ent, reward.shape)
            reward = reward + self._entropy_coef * ent
        return reward

    def loss(self, obs, next_obs, skill, mask=None):
        transition = self.transition_features(obs, next_obs)
        log_q, aux = self._log_prob_and_aux_from_transition(transition, skill)
        objective = aux.get("objective", log_q)
        loss = -masked_mean(objective[..., 0], mask)
        metrics = {
            "csf_loss": loss,
            "csf_log_q": masked_mean(log_q[..., 0], mask),
            "csf_pos_score": masked_mean(aux["pos_score"][..., 0], mask),
        }
        if "neg_term" in aux:
            metrics["csf_neg_term"] = masked_mean(aux["neg_term"][..., 0], mask)
        if "objective" in aux:
            metrics["csf_objective"] = masked_mean(aux["objective"][..., 0], mask)
        if "acc" in aux:
            metrics["csf_acc"] = aux["acc"]
        if "cosine" in aux:
            metrics["csf_cosine"] = aux["cosine"]
        if "mse" in aux:
            metrics["csf_mse"] = aux["mse"]
        return loss, metrics

    def compute_reward(self, obs, action=None, next_obs=None, skill=None, policy_entropy=None, **kwargs):
        del action, kwargs
        if next_obs is None:
            raise ValueError("next_obs is required to compute CSF reward.")
        if skill is None:
            raise ValueError("Skill must be provided to compute CSF reward.")
        return self.reward_from_transition(
            self.transition_features(obs, next_obs),
            skill,
            policy_entropy=policy_entropy,
        )

    def update(self, obs=None, action=None, next_obs=None, skill=None, optimizer=None, return_loss=False, **kwargs):
        del action, kwargs
        if optimizer is not None:
            raise NotImplementedError("JAX CSF is updated by the enclosing Ninjax optimizer.")
        loss, metrics = self.loss(obs, next_obs, skill)
        return (loss, metrics) if return_loss else metrics

    def bridge_consistency_targets(self, obs, action=None, next_obs=None, skill=None, **kwargs):
        del action, skill, kwargs
        return self.transition_features(obs, next_obs)

    def get_modules(self):
        modules = {"csf": self}
        if self.projector is not None:
            modules["csf_projector"] = self.projector
        if self.skill_projection is not None:
            modules["csf_skill_projection"] = self.skill_projection
        if self.gaussian_head is not None:
            modules["csf_gaussian_head"] = self.gaussian_head
        return modules
