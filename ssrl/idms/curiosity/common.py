"""Shared JAX curiosity helpers."""

import math

import jax
import jax.numpy as jnp
import ninjax as nj

from .._runtime import act, f32


def xavier_uniform(shape, in_dim: int, out_dim: int):
    limit = math.sqrt(6.0 / float(in_dim + out_dim))
    return jax.random.uniform(nj.seed(), shape, jnp.float32, -limit, limit)


def masked_mean(values, mask=None, eps: float = 1e-8):
    values = f32(values)
    if mask is None:
        return values.mean()
    mask = f32(mask)
    while mask.ndim < values.ndim:
        mask = mask[..., None]
    total = (values * mask).sum()
    count = jnp.maximum(mask.sum(), eps)
    return total / count


def padded_next_sequence(seq):
    return jnp.concatenate([seq[:, 1:], seq[:, -1:]], axis=1)


def softmax_cross_entropy(logits, labels):
    labels = labels.astype(jnp.int32)
    onehot = jax.nn.one_hot(labels, logits.shape[-1])
    return -(onehot * jax.nn.log_softmax(logits, axis=-1)).sum(axis=-1)


class RunningMeanNormalizer(nj.Module):

    eps: float = 1e-8

    def __init__(self):
        self.mean = nj.Variable(jnp.zeros, (), jnp.float32, name="mean")
        self.count = nj.Variable(jnp.zeros, (), jnp.float32, name="count")
        self._eps = float(self.eps)

    def reset(self):
        self.mean.write(jnp.zeros((), jnp.float32))
        self.count.write(jnp.zeros((), jnp.float32))

    def update(self, x, mask=None):
        x = jax.lax.stop_gradient(f32(x))
        if mask is None:
            batch_count = jnp.asarray(x.size, jnp.float32)
            batch_mean = jnp.abs(x).mean()
        else:
            mask = f32(mask)
            while mask.ndim < x.ndim:
                mask = mask[..., None]
            mask = jnp.broadcast_to(mask, x.shape)
            batch_count = mask.sum()
            batch_mean = (jnp.abs(x) * mask).sum() / jnp.maximum(
                batch_count, 1.0
            )
        total = self.count.read() + batch_count
        total_safe = jnp.maximum(total, 1.0)
        mean = (
            self.mean.read() * self.count.read() + batch_mean * batch_count
        ) / total_safe
        has_values = batch_count > 0
        self.mean.write(jnp.where(has_values, mean, self.mean.read()))
        self.count.write(jnp.where(has_values, total, self.count.read()))

    def normalize(self, x, update=True, mask=None):
        if update:
            self.update(x, mask)
        denom = jnp.maximum(self.mean.read().astype(x.dtype), self._eps)
        return x / denom


class RunningMeanStd(nj.Module):

    eps: float = 1e-8

    def __init__(self):
        self.mean = nj.Variable(jnp.zeros, (), jnp.float32, name="mean")
        self.var = nj.Variable(jnp.ones, (), jnp.float32, name="var")
        self.count = nj.Variable(jnp.zeros, (), jnp.float32, name="count")
        self._eps = float(self.eps)

    def reset(self):
        self.mean.write(jnp.zeros((), jnp.float32))
        self.var.write(jnp.ones((), jnp.float32))
        self.count.write(jnp.zeros((), jnp.float32))

    def update(self, x):
        x = jax.lax.stop_gradient(f32(x).reshape((-1,)))
        batch_count = jnp.asarray(x.size, jnp.float32)
        batch_mean = x.mean()
        batch_var = jnp.square(x - batch_mean).mean()
        old_count = self.count.read()
        total = old_count + batch_count
        delta = batch_mean - self.mean.read()
        new_mean = self.mean.read() + delta * batch_count / jnp.maximum(total, 1.0)
        m_a = self.var.read() * old_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + jnp.square(delta) * old_count * batch_count / jnp.maximum(total, 1.0)
        new_var = m2 / jnp.maximum(total, 1.0)
        first = old_count <= 0
        self.mean.write(jnp.where(first, batch_mean, new_mean))
        self.var.write(jnp.where(first, batch_var, new_var))
        self.count.write(total)

    def normalize(self, x, update=True):
        if update:
            self.update(x)
        mean = self.mean.read().astype(x.dtype)
        std = jnp.sqrt(jnp.maximum(self.var.read().astype(x.dtype), self._eps))
        normalized = (x - mean) / std
        return jnp.where(self.count.read() <= 0, jnp.zeros_like(x), normalized)


class MLP(nj.Module):

    input_dim: int = 0
    output_dim: int = 0
    hidden_dim: int = 256
    layers: int = 2
    act: str = "relu"

    def __init__(self):
        self._input_dim = int(self.input_dim)
        self._output_dim = int(self.output_dim)
        self._hidden_dim = int(self.hidden_dim)
        self._layers = int(self.layers)
        if self._input_dim <= 0 or self._output_dim <= 0:
            raise ValueError("MLP input_dim and output_dim must be positive.")
        if self._layers < 1:
            self._dims = [self._input_dim, self._output_dim]
        else:
            self._dims = [
                self._input_dim,
                *([self._hidden_dim] * self._layers),
                self._output_dim,
            ]

    def _weight(self, name, in_dim, out_dim):
        shape = (in_dim, out_dim)
        return self.value(name, lambda shape: xavier_uniform(shape, in_dim, out_dim), shape)

    def _bias(self, name, out_dim):
        return self.value(name, jnp.zeros, (out_dim,), jnp.float32)

    def __call__(self, x, stop_params: bool = False):
        x = f32(x)
        last = len(self._dims) - 2
        for index, (in_dim, out_dim) in enumerate(zip(self._dims[:-1], self._dims[1:])):
            weight = self._weight(f"w{index}", in_dim, out_dim)
            bias = self._bias(f"b{index}", out_dim)
            if stop_params:
                weight = jax.lax.stop_gradient(weight)
                bias = jax.lax.stop_gradient(bias)
            x = jnp.einsum("...i,io->...o", x, weight) + bias
            if index != last:
                x = act(self.act)(x)
        return f32(x)
