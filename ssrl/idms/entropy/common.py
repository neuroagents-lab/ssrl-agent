"""Shared JAX entropy IDM helpers."""

import jax
import jax.numpy as jnp
import ninjax as nj

from .._runtime import f32
from ..curiosity.common import MLP, RunningMeanNormalizer


class RunningRMSNormalizer(nj.Module):

    eps: float = 1e-8

    def __init__(self):
        self.mean_sq = nj.Variable(jnp.zeros, (), jnp.float32, name="mean_sq")
        self.count = nj.Variable(jnp.zeros, (), jnp.float32, name="count")
        self._eps = float(self.eps)

    def reset(self):
        self.mean_sq.write(jnp.zeros((), jnp.float32))
        self.count.write(jnp.zeros((), jnp.float32))

    def update(self, x):
        x = jax.lax.stop_gradient(f32(x))
        batch_count = jnp.asarray(x.size, jnp.float32)
        batch_mean_sq = jnp.square(x).mean()
        total = self.count.read() + batch_count
        total_safe = jnp.maximum(total, 1.0)
        mean_sq = (
            self.mean_sq.read() * self.count.read()
            + batch_mean_sq * batch_count
        ) / total_safe
        self.mean_sq.write(mean_sq)
        self.count.write(total)

    def normalize(self, x, update=True):
        if update:
            self.update(x)
        rms = jnp.sqrt(jnp.maximum(self.mean_sq.read().astype(x.dtype), self._eps))
        return x / rms


def validate_positive(name, value):
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def validate_at_least(name, value, minimum):
    if int(value) < int(minimum):
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")


def _safe_l2_distance_from_squared(dist_sq):
    """Square root with a finite zero-distance gradient.

    Use a positive differentiation denominator at zero while preserving
    an exact zero distance.
    """
    dist_sq = f32(dist_sq)
    is_zero = dist_sq == 0
    safe_sq = jnp.where(is_zero, jnp.ones_like(dist_sq), dist_sq)
    safe = jnp.sqrt(safe_sq)
    return jnp.where(is_zero, jnp.zeros_like(safe), safe)


def ring_update(memory, ptr, count, values):
    values = f32(values).reshape((-1, values.shape[-1]))
    size = int(memory.shape[0])
    n = int(values.shape[0])
    if n == 0:
        return memory, ptr, count
    if n >= size:
        return (
            values[-size:].astype(jnp.float32),
            jnp.zeros((), jnp.int32),
            jnp.asarray(size, jnp.int32),
        )
    idx = (ptr.astype(jnp.int32) + jnp.arange(n, dtype=jnp.int32)) % size
    memory = memory.at[idx].set(values.astype(jnp.float32))
    ptr = (ptr.astype(jnp.int32) + n) % size
    count = jnp.minimum(jnp.asarray(size, jnp.int32), count.astype(jnp.int32) + n)
    return memory, ptr, count


def sample_memory(memory, count, sample_size):
    sample_size = int(sample_size)
    total = jnp.clip(count.astype(jnp.int32), 0, int(memory.shape[0]))
    sample_count = jnp.minimum(total, jnp.asarray(sample_size, jnp.int32))
    arange = jnp.arange(sample_size, dtype=jnp.int32)
    random_idx = jax.random.randint(
        nj.seed(),
        (sample_size,),
        0,
        jnp.maximum(total, 1),
        dtype=jnp.int32,
    )
    idx = jnp.where(total > sample_size, random_idx, arange)
    mask = arange < sample_count
    return memory[idx], mask


def masked_kth_l2_distance(query, particles, mask, k):
    query = f32(query).reshape((-1, query.shape[-1]))
    particles = f32(particles)
    mask = mask.astype(bool)
    dist_sq = jnp.square(query[:, None, :] - particles[None]).sum(axis=-1)
    dist = _safe_l2_distance_from_squared(dist_sq)
    dist = jnp.where(mask[None], dist, jnp.inf)
    sorted_dist = jnp.sort(dist, axis=-1)
    valid_count = mask.astype(jnp.int32).sum()
    k_eff = jnp.minimum(jnp.asarray(int(k), jnp.int32), valid_count)
    kth_index = jnp.maximum(k_eff, 1) - 1
    kth = jnp.take_along_axis(
        sorted_dist,
        jnp.full((query.shape[0], 1), kth_index, jnp.int32),
        axis=-1,
    )
    return jnp.where(valid_count > 0, kth, jnp.zeros_like(kth))


def masked_knn_mean_lp_power(query, particles, mask, k, power):
    """Mean of the k smallest Euclidean distances, each raised to ``power``.

    Apply ``power`` to the Euclidean norm, after summing squared coordinate
    differences. APT uses ``power=1`` for numerical stability.
    """
    query = f32(query).reshape((-1, query.shape[-1]))
    particles = f32(particles)
    mask = mask.astype(bool)
    dist_sq = jnp.square(query[:, None, :] - particles[None]).sum(axis=-1)
    dist_power = _safe_l2_distance_from_squared(dist_sq)
    if float(power) != 1.0:
        dist_power = jnp.power(dist_power, float(power))
    dist_power = jnp.where(mask[None], dist_power, jnp.inf)
    sorted_dist = jnp.sort(dist_power, axis=-1)
    top_width = min(int(k), int(particles.shape[0]))
    top = sorted_dist[:, :top_width]
    valid_count = mask.astype(jnp.int32).sum()
    k_eff = jnp.minimum(jnp.asarray(int(k), jnp.int32), valid_count)
    valid_knn = jnp.arange(top_width, dtype=jnp.int32) < k_eff
    total = jnp.where(valid_knn[None], top, jnp.zeros_like(top)).sum(axis=-1, keepdims=True)
    mean = total / jnp.maximum(k_eff.astype(query.dtype), 1.0)
    return jnp.where(valid_count > 0, mean, jnp.zeros_like(mean))
