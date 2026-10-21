"""JAX Active Pretraining entropy module."""

import jax
import jax.numpy as jnp
import ninjax as nj

from .._runtime import f32
from .common import (
    MLP,
    RunningMeanNormalizer,
    masked_knn_mean_lp_power,
    ring_update,
    sample_memory,
    validate_at_least,
    validate_positive,
)


class APT(nj.Module):

    obs_dim: int = 0
    rep_dim: int = 256
    hidden_dim: int = 256
    n_layers: int = 2
    k: int = 16
    knn_clip: float = 0.0005
    scale: float = 1.0
    normalize_phi: bool = True
    train_phi: bool = True
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
        self._knn_clip = float(self.knn_clip)
        self._scale = float(self.scale)
        self._normalize_phi = bool(self.normalize_phi)
        self._train_phi = bool(self.train_phi)
        self._reward_normalize = bool(self.reward_normalize)
        self._eps = float(self.eps)
        self._knn_chunk_size = int(self.knn_chunk_size)
        self._memory_size = int(self.memory_size)
        self._memory_sample_size = int(self.memory_sample_size)
        self.updates_reward_normalizer_in_update = True

        validate_positive("APT obs_dim", self._obs_dim)
        validate_positive("APT rep_dim", self._rep_dim)
        validate_at_least("APT n_layers", self.n_layers, 1)
        validate_positive("APT k", self._k)
        validate_positive("APT knn_chunk_size", self._knn_chunk_size)
        validate_positive("APT memory_size", self._memory_size)
        validate_positive("APT memory_sample_size", self._memory_sample_size)
        if self._memory_sample_size > self._memory_size:
            raise ValueError("APT memory_sample_size must be <= memory_size.")

        self.phi = MLP(
            input_dim=self._obs_dim,
            output_dim=self._rep_dim,
            hidden_dim=int(self.hidden_dim),
            layers=int(self.n_layers),
            act=self.act,
            name="phi",
        )
        self.reward_mean = (
            RunningMeanNormalizer(eps=self._eps, name="reward_mean")
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

    def embed(self, obs, stop_params=False):
        h = self.phi(f32(obs), stop_params=stop_params)
        if self._normalize_phi:
            # Clamp the squared norm to keep zero-vector gradients finite.
            norm_sq = jnp.square(h).sum(axis=-1, keepdims=True)
            inv_norm = jax.lax.rsqrt(
                jnp.maximum(norm_sq, jnp.square(jnp.asarray(self._eps, h.dtype)))
            )
            h = h * inv_norm
        return f32(h)

    def reset_memory(self):
        self._mem.write(jnp.zeros_like(self._mem.read()))
        self._mem_ptr.write(jnp.zeros((), jnp.int32))
        self._mem_count.write(jnp.zeros((), jnp.int32))

    def memory_count(self):
        return self._mem_count.read()

    def update_memory(self, obs):
        h = self.embed(obs, stop_params=not self._train_phi)
        self._update_memory_from_embeddings(h)

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

    def _compute_bonus_from_embeddings(self, h_next, update_stats=True):
        query = h_next.reshape((-1, self._rep_dim))
        particles, mask = self._sample_particles()
        # APT uses mean Euclidean k-NN distance for a stable particle-entropy estimate.
        mean_knn = masked_knn_mean_lp_power(
            query=query,
            particles=particles,
            mask=mask,
            k=self._k,
            power=1.0,
        )
        raw_bonus = mean_knn.reshape((*h_next.shape[:-1], 1))
        if self.reward_mean is not None:
            raw_bonus = self.reward_mean.normalize(raw_bonus, update=update_stats)
        if self._knn_clip >= 0.0:
            raw_bonus = jnp.maximum(raw_bonus - self._knn_clip, 0.0)
        return jnp.log1p(raw_bonus)

    def compute_bonus(self, next_obs, update_stats=True):
        h_next = self.embed(next_obs, stop_params=not self._train_phi)
        return self._compute_bonus_from_embeddings(h_next, update_stats=update_stats)

    def compute_reward(
        self,
        obs=None,
        action=None,
        next_obs=None,
        update_reward_norm=True,
        **kwargs,
    ):
        del obs, action, kwargs
        if next_obs is None:
            raise ValueError("next_obs is required to compute APT reward.")
        return self._scale * self.compute_bonus(
            next_obs,
            update_stats=update_reward_norm,
        )

    def loss(
        self,
        next_obs,
        mask=None,
        train_phi=None,
        update_memory=True,
        update_reward_norm=True,
    ):
        del mask
        train_phi = self._train_phi if train_phi is None else bool(train_phi)
        if next_obs.size == 0:
            zero = jnp.zeros((), jnp.float32)
            metrics = {
                "apt_bonus_mean": zero,
                "apt_mem_count": self.memory_count().astype(jnp.float32),
            }
            if train_phi:
                metrics["apt_phi_loss"] = zero
            return zero, metrics
        h_next = self.embed(next_obs, stop_params=not train_phi)
        bonus = self._compute_bonus_from_embeddings(
            h_next,
            update_stats=update_reward_norm,
        )
        if update_memory:
            self._update_memory_from_embeddings(h_next)
        bonus_mean = bonus.mean()
        loss = -bonus_mean if train_phi else jnp.zeros((), bonus.dtype)
        metrics = {
            "apt_bonus_mean": bonus_mean,
            "apt_mem_count": self.memory_count().astype(jnp.float32),
        }
        if train_phi:
            metrics["apt_phi_loss"] = loss
        return loss, metrics

    def update(
        self,
        obs=None,
        action=None,
        next_obs=None,
        optimizer=None,
        return_loss=False,
        **kwargs,
    ):
        del obs, action
        if optimizer is not None:
            raise NotImplementedError("JAX APT is updated by the enclosing Ninjax optimizer.")
        if next_obs is None:
            raise ValueError("next_obs is required to update APT.")
        loss, metrics = self.loss(next_obs, **kwargs)
        return (loss, metrics) if return_loss else metrics

    def bridge_consistency_targets(self, obs, action=None, next_obs=None, **kwargs):
        del action, next_obs, kwargs
        return self.embed(obs)

    def reset(self):
        self.reset_memory()
        if self.reward_mean is not None:
            self.reward_mean.reset()

    def get_modules(self):
        return {"apt_phi": self.phi}
