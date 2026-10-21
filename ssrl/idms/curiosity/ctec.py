"""JAX C-TeC curiosity module."""

import jax
import jax.numpy as jnp
import ninjax as nj

from .._runtime import f32
from .common import MLP, masked_mean


def _normalize_similarity(name: str) -> str:
    value = str(name).lower()
    aliases = {
        "l1": "l1",
        "l2": "l2",
        "sq_l2": "sq_l2",
        "l2_sq": "sq_l2",
        "squared_l2": "sq_l2",
        "dot": "dot",
    }
    if value not in aliases:
        raise ValueError(f"Unknown C-TeC similarity {name!r}.")
    return aliases[value]


def _normalize_loss_type(name: str) -> str:
    value = str(name).lower()
    aliases = {
        "infonce": "infonce",
        "info_nce": "infonce",
        "symmetric_infonce": "symmetric_infonce",
        "symmetric_info_nce": "symmetric_infonce",
        "symmetric": "symmetric_infonce",
    }
    if value not in aliases:
        raise ValueError(f"Unknown C-TeC loss {name!r}.")
    return aliases[value]


def _normalize_reward_estimator(name: str) -> str:
    value = str(name).lower()
    aliases = {
        "single": "single",
        "sample": "single",
        "truncated": "truncated",
        "truncated_expectation": "truncated",
        "all_future": "truncated",
    }
    if value not in aliases:
        raise ValueError(f"Unknown C-TeC reward estimator {name!r}.")
    return aliases[value]


class CTEC(nj.Module):

    obs_dim: int = 0
    action_dim: int = 0
    rep_dim: int = 64
    hidden_dim: int = 1024
    n_layers: int = 2
    scale: float = 1.0
    discount: float = 0.99
    temperature: float = 1.0
    learn_temperature: bool = True
    similarity: str = "l1"
    loss_type: str = "infonce"
    normalize_embeddings: bool = True
    reward_estimator: str = "single"
    reward_normalize_truncation: bool = False
    replay_batch_size: int = 256
    act: str = "relu"

    def __init__(self):
        self._obs_dim = int(self.obs_dim)
        self._action_dim = int(self.action_dim)
        self._rep_dim = int(self.rep_dim)
        if min(self._obs_dim, self._action_dim, self._rep_dim) <= 0:
            raise ValueError("C-TeC obs_dim, action_dim, and rep_dim must be positive.")
        self._scale = float(self.scale)
        self._discount = float(min(max(float(self.discount), 0.0), 0.999999))
        self._learn_temperature = bool(self.learn_temperature)
        self._similarity = _normalize_similarity(self.similarity)
        self._loss_type = _normalize_loss_type(self.loss_type)
        self._reward_estimator = _normalize_reward_estimator(self.reward_estimator)
        self._normalize_truncation = bool(self.reward_normalize_truncation)
        self.query_encoder = MLP(
            input_dim=self._obs_dim + self._action_dim,
            output_dim=self._rep_dim,
            hidden_dim=int(self.hidden_dim),
            layers=int(self.n_layers),
            act=self.act,
            name="query_encoder",
        )
        self.future_encoder = MLP(
            input_dim=self._obs_dim,
            output_dim=self._rep_dim,
            hidden_dim=int(self.hidden_dim),
            layers=int(self.n_layers),
            act=self.act,
            name="future_encoder",
        )

    def _log_temperature(self):
        init = lambda shape: jnp.full(shape, jnp.log(max(float(self.temperature), 1e-6)), jnp.float32)
        value = self.value("log_temperature", init, ())
        return value if self._learn_temperature else jax.lax.stop_gradient(value)

    @property
    def temperature_value(self):
        return jnp.maximum(jnp.exp(self._log_temperature()), 1e-6)

    def _normalize(self, z):
        if not bool(self.normalize_embeddings):
            return z
        norm = jnp.linalg.norm(z, axis=-1, keepdims=True)
        return z / jnp.maximum(norm, 1e-8)

    def encode_query(self, obs, action):
        z = self.query_encoder(jnp.concatenate([f32(obs), f32(action)], axis=-1))
        return self._normalize(z)

    def encode_future(self, future_obs):
        z = self.future_encoder(f32(future_obs))
        return self._normalize(z)

    def _pairwise_logits(self, query, future):
        if self._similarity == "dot":
            scores = query @ future.T
        elif self._similarity == "l1":
            scores = -jnp.abs(query[:, None] - future[None]).sum(axis=-1)
        elif self._similarity == "l2":
            dist_sq = jnp.square(query[:, None] - future[None]).sum(axis=-1)
            scores = -jnp.sqrt(jnp.maximum(dist_sq, 0.0))
        else:
            scores = -jnp.square(query[:, None] - future[None]).sum(axis=-1)
        return scores / self.temperature_value

    def _pairwise_reward(self, query, future):
        if self._similarity == "dot":
            return -(query * future).sum(axis=-1, keepdims=True)
        diff = query - future
        if self._similarity == "l1":
            return jnp.abs(diff).sum(axis=-1, keepdims=True)
        dist_sq = jnp.square(diff).sum(axis=-1, keepdims=True)
        if self._similarity == "l2":
            return jnp.sqrt(jnp.maximum(dist_sq, 0.0))
        return dist_sq

    def _sequence_pairwise_rewards(self, query, future):
        if self._similarity == "dot":
            return -(query[:, None] * future[None]).sum(axis=-1, keepdims=True)
        diff = query[:, None] - future[None]
        if self._similarity == "l1":
            return jnp.abs(diff).sum(axis=-1, keepdims=True)
        dist_sq = jnp.square(diff).sum(axis=-1, keepdims=True)
        if self._similarity == "l2":
            return jnp.sqrt(jnp.maximum(dist_sq, 0.0))
        return dist_sq

    def loss(self, obs, action, future_obs, mask=None):
        query = self.encode_query(obs, action).reshape((-1, self._rep_dim))
        future = self.encode_future(future_obs).reshape((-1, self._rep_dim))
        logits = self._pairwise_logits(query, future)
        labels = jnp.arange(logits.shape[0], dtype=jnp.int32)
        row_loss = -jax.nn.log_softmax(logits, axis=-1)[labels, labels]
        loss_fwd = masked_mean(row_loss, mask)
        pred = jnp.argmax(logits, axis=-1)
        acc_fwd = masked_mean((pred == labels).astype(jnp.float32), mask)
        if self._loss_type == "symmetric_infonce":
            row_loss_bwd = -jax.nn.log_softmax(logits.T, axis=-1)[labels, labels]
            loss_bwd = masked_mean(row_loss_bwd, mask)
            pred_bwd = jnp.argmax(logits.T, axis=-1)
            acc_bwd = masked_mean((pred_bwd == labels).astype(jnp.float32), mask)
            loss = 0.5 * (loss_fwd + loss_bwd)
            acc = 0.5 * (acc_fwd + acc_bwd)
        else:
            loss = loss_fwd
            acc = acc_fwd
        diag = jnp.diag(logits)
        neg_count = jnp.maximum(logits.size - logits.shape[0], 1)
        neg_logits = (logits.sum() - diag.sum()) / neg_count
        metrics = {
            "ctec_loss": loss,
            "ctec_acc": acc,
            "ctec_temperature": self.temperature_value,
            "ctec_pos_logit": diag.mean(),
            "ctec_neg_logit": neg_logits,
            "ctec_batch": jnp.asarray(logits.shape[0], jnp.float32),
        }
        return loss, metrics

    def compute_reward(self, obs, action, next_obs=None, future_obs=None, **kwargs):
        del kwargs
        target = future_obs if future_obs is not None else next_obs
        if target is None:
            raise ValueError("C-TeC reward requires future_obs or next_obs.")
        query = self.encode_query(obs, action)
        future = self.encode_future(target)
        return self._pairwise_reward(query, future) * self._scale

    def compute_sequence_reward(
        self,
        obs,
        action,
        estimator=None,
        normalize_truncation=None,
        discount=None,
    ):
        if obs.ndim != 3 or action.ndim != 3:
            raise ValueError(f"C-TeC sequence rewards expect [T,B,D], got {obs.shape}.")
        estimator = _normalize_reward_estimator(
            estimator if estimator is not None else self._reward_estimator
        )
        gamma = self._discount if discount is None else float(min(max(float(discount), 0.0), 0.999999))
        normalize = self._normalize_truncation if normalize_truncation is None else bool(normalize_truncation)
        query = self.encode_query(obs, action)
        future = self.encode_future(obs)
        horizon = obs.shape[0]
        if horizon <= 1:
            return jnp.zeros((*obs.shape[:2], 1), dtype=f32(obs).dtype)
        distances = self._sequence_pairwise_rewards(query, future)
        step = jnp.arange(horizon)
        future_offset = step[None, :] - step[:, None] - 1
        valid_future = future_offset >= 0
        weights = jnp.where(
            valid_future,
            jnp.power(jnp.asarray(gamma, obs.dtype), jnp.maximum(future_offset, 0).astype(obs.dtype)),
            jnp.zeros((horizon, horizon), dtype=obs.dtype),
        )
        if estimator == "single":
            probs = weights / jnp.maximum(weights.sum(axis=-1, keepdims=True), 1e-8)
            sampled = jax.random.categorical(
                nj.seed(), jnp.log(jnp.maximum(probs[:-1], 1e-30)), axis=-1
            )
            rewards = jnp.zeros((*obs.shape[:2], 1), dtype=obs.dtype)
            chosen = distances[jnp.arange(horizon - 1), sampled]
            rewards = rewards.at[:-1].set(chosen)
            return rewards * self._scale
        weighted = (distances * weights[:, :, None, None]).sum(axis=1)
        if normalize:
            weighted = weighted / jnp.maximum(weights.sum(axis=-1)[:, None, None], 1e-8)
        return weighted * self._scale

    def sample_future_loss(self, obs, action, is_first=None, sample_size=None):
        if obs.ndim != 3:
            raise ValueError(f"C-TeC replay sampling expects [B,T,D], got {obs.shape}.")
        batch, horizon = obs.shape[:2]
        sample_size = int(sample_size or self.replay_batch_size or (batch * horizon))
        if is_first is None:
            is_first = jnp.zeros((batch, horizon), bool)
        elif is_first.ndim > 2 and is_first.shape[-1] == 1:
            is_first = is_first[..., 0]
        episode_id = jnp.cumsum(is_first.astype(jnp.int32), axis=1)
        step = jnp.arange(horizon)
        future_offset = step[None, :] - step[:, None] - 1
        valid_future = future_offset >= 0
        same_episode = episode_id[:, :, None] == episode_id[:, None, :]
        valid = same_episode & valid_future[None]
        weights = jnp.where(
            valid,
            jnp.power(jnp.asarray(self._discount, obs.dtype), jnp.maximum(future_offset, 0).astype(obs.dtype))[None],
            jnp.zeros((batch, horizon, horizon), dtype=obs.dtype),
        )
        flat_weights = weights.reshape((-1,))
        total = flat_weights.sum()
        probs = jnp.where(
            total > 0,
            flat_weights / jnp.maximum(total, 1e-8),
            jnp.ones_like(flat_weights) / flat_weights.size,
        )
        idx = jax.random.categorical(
            nj.seed(), jnp.log(jnp.maximum(probs, 1e-30)), shape=(sample_size,)
        )
        valid_flat = valid.reshape((-1,))
        sample_mask = valid_flat[idx] & (total > 0)
        start = idx // horizon
        b_idx = start // horizon
        t_idx = start % horizon
        f_idx = idx % horizon
        obs_batch = obs[b_idx, t_idx]
        action_batch = action[b_idx, t_idx]
        future_batch = obs[b_idx, f_idx]
        loss, metrics = self.loss(obs_batch, action_batch, future_batch, sample_mask)
        metrics["ctec_transition_batch"] = sample_mask.astype(jnp.float32).sum()
        return loss, metrics

    def update(
        self,
        obs,
        action,
        next_obs=None,
        future_obs=None,
        optimizer=None,
        return_loss=False,
        **kwargs,
    ):
        del kwargs
        if optimizer is not None:
            raise NotImplementedError("JAX C-TeC is updated by the enclosing Ninjax optimizer.")
        target = future_obs if future_obs is not None else next_obs
        if target is None:
            raise ValueError("C-TeC update requires future_obs or next_obs.")
        loss, metrics = self.loss(obs, action, target)
        return (loss, metrics) if return_loss else metrics

    def get_modules(self):
        return {
            "ctec_query_encoder": self.query_encoder,
            "ctec_future_encoder": self.future_encoder,
        }
