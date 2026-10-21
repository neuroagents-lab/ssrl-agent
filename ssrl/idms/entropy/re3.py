"""JAX RE3 entropy module."""

import jax
import jax.numpy as jnp
import ninjax as nj

from .._runtime import f32
from .common import (
    MLP,
    RunningRMSNormalizer,
    masked_kth_l2_distance,
    ring_update,
    sample_memory,
    validate_at_least,
    validate_positive,
)


class RE3(nj.Module):

    obs_dim: int = 0
    rep_dim: int = 50
    hidden_dim: int = 256
    n_layers: int = 2
    k: int = 53
    scale: float = 0.1
    log_reward: bool = False
    reward_normalize: bool = True
    eps: float = 1e-8
    knn_chunk_size: int = 1024
    memory_size: int = 2048
    memory_sample_size: int = 2048
    act: str = "relu"

    def __init__(self):
        self._obs_dim = int(self.obs_dim)
        self._rep_dim = int(self.rep_dim)
        self._k = int(self.k)
        self._scale = float(self.scale)
        self._log_reward = bool(self.log_reward)
        self._reward_normalize = bool(self.reward_normalize)
        self._eps = float(self.eps)
        self._knn_chunk_size = int(self.knn_chunk_size)
        self._memory_size = int(self.memory_size)
        self._memory_sample_size = int(self.memory_sample_size)
        self.updates_reward_normalizer_in_update = True

        validate_positive("RE3 obs_dim", self._obs_dim)
        validate_positive("RE3 rep_dim", self._rep_dim)
        validate_positive("RE3 hidden_dim", self.hidden_dim)
        validate_at_least("RE3 n_layers", self.n_layers, 1)
        validate_positive("RE3 k", self._k)
        validate_positive("RE3 knn_chunk_size", self._knn_chunk_size)
        validate_positive("RE3 memory_size", self._memory_size)
        validate_positive("RE3 memory_sample_size", self._memory_sample_size)
        if self._memory_sample_size > self._memory_size:
            raise ValueError("RE3 memory_sample_size must be <= memory_size.")

        self.phi = MLP(
            input_dim=self._obs_dim,
            output_dim=self._rep_dim,
            hidden_dim=int(self.hidden_dim),
            layers=int(self.n_layers),
            act=self.act,
            name="phi",
        )
        self.reward_rms = (
            RunningRMSNormalizer(eps=self._eps, name="reward_rms")
            if self._reward_normalize
            else None
        )
        self._mem = nj.Variable(
            jnp.zeros,
            (self._memory_size, self._rep_dim),
            jnp.float32,
            name="mem",
        )
        self._mem_ptr = nj.Variable(jnp.zeros, (), jnp.int32, name="mem_ptr")
        self._mem_count = nj.Variable(jnp.zeros, (), jnp.int32, name="mem_count")

    def embed(self, obs):
        return self.phi(f32(obs), stop_params=True)

    def reset_memory(self):
        self._mem.write(jnp.zeros_like(self._mem.read()))
        self._mem_ptr.write(jnp.zeros((), jnp.int32))
        self._mem_count.write(jnp.zeros((), jnp.int32))

    def memory_count(self):
        return self._mem_count.read()

    def update_memory(self, obs):
        self._update_memory_from_embeddings(self.embed(obs))

    def _update_memory_from_embeddings(self, h):
        mem, ptr, count = ring_update(
            self._mem.read(),
            self._mem_ptr.read(),
            self._mem_count.read(),
            jax.lax.stop_gradient(h),
        )
        self._mem.write(mem)
        self._mem_ptr.write(ptr)
        self._mem_count.write(count)

    def _sample_particles(self):
        return sample_memory(
            self._mem.read(),
            self._mem_count.read(),
            self._memory_sample_size,
        )

    def _compute_bonus_from_embeddings(self, h, update_reward_norm=True):
        query = h.reshape((-1, self._rep_dim))
        particles, mask = self._sample_particles()
        kth_dist = masked_kth_l2_distance(query, particles, mask, self._k)
        bonus = kth_dist
        if self._log_reward:
            bonus = jnp.log1p(bonus)
        if self.reward_rms is not None:
            bonus = self.reward_rms.normalize(bonus, update=update_reward_norm)
        return (
            bonus.reshape((*h.shape[:-1], 1)),
            kth_dist.reshape((*h.shape[:-1], 1)),
        )

    def compute_reward(
        self,
        obs,
        action=None,
        next_obs=None,
        update_reward_norm=True,
        **kwargs,
    ):
        del action, next_obs, kwargs
        bonus, _ = self._compute_bonus_from_embeddings(
            self.embed(obs),
            update_reward_norm=update_reward_norm,
        )
        return self._scale * bonus

    def loss(self, obs, mask=None, update_memory=True, update_reward_norm=True):
        del mask
        if obs.size == 0:
            zero = jnp.zeros((), jnp.float32)
            metrics = {
                "re3_bonus_mean": zero,
                "re3_kth_dist_mean": zero,
                "re3_mem_count": self.memory_count().astype(jnp.float32),
            }
            return zero, metrics
        h = self.embed(obs)
        bonus, kth_dist = self._compute_bonus_from_embeddings(
            h,
            update_reward_norm=update_reward_norm,
        )
        if update_memory:
            self._update_memory_from_embeddings(h)
        zero = bonus.mean() * 0.0
        metrics = {
            "re3_bonus_mean": bonus.mean(),
            "re3_kth_dist_mean": kth_dist.mean(),
            "re3_mem_count": self.memory_count().astype(jnp.float32),
        }
        if self.reward_rms is not None:
            metrics["re3_reward_rms"] = jnp.sqrt(
                jnp.maximum(self.reward_rms.mean_sq.read(), self._eps)
            )
        return zero, metrics

    def update(
        self,
        obs=None,
        action=None,
        next_obs=None,
        optimizer=None,
        return_loss=False,
        **kwargs,
    ):
        del action, next_obs
        if optimizer is not None:
            raise NotImplementedError("JAX RE3 is updated by the enclosing Ninjax optimizer.")
        if obs is None:
            raise ValueError("obs is required to update RE3.")
        loss, metrics = self.loss(obs, **kwargs)
        return (loss, metrics) if return_loss else metrics

    def bridge_consistency_targets(self, obs, action=None, next_obs=None, **kwargs):
        del action, next_obs, kwargs
        return self.embed(obs)

    def reset(self):
        self.reset_memory()
        if self.reward_rms is not None:
            self.reward_rms.reset()

    def get_modules(self):
        return {"re3_phi": self.phi}
