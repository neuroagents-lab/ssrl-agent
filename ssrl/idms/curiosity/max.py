"""Model-Based Active Exploration (MAX) over world-model features.

Shyam, Jaskowski & Gomez (2019), "Model-Based Active Exploration".

An ensemble of diagonal-Gaussian forward models is trained with bootstrapped
Gaussian likelihood on replay transitions. Imagined transitions receive the
Jensen-Renyi divergence between predictive distributions, including both
mean disagreement and predicted variance.
"""

import math

import jax
import jax.numpy as jnp
import ninjax as nj

from .._runtime import act, f32
from ..chunking import chunked_rows


class MAX(nj.Module):

    obs_dim: int = 0
    input_dim: int = 0
    action_dim: int = 0
    target_dim: int | None = None
    hidden_dim: int = 256
    mlp_layers: int = 2
    layers: int = 0
    n_models: int = 5
    models: int = 0
    scale: float = 1.0
    jrd_lambda: float = 0.1
    min_log_var: float = -10.0
    max_log_var: float = 2.0
    normalize_rewards: bool = False
    eps: float = 1e-8
    act: str = "relu"
    # Zero evaluates all model pairs and transitions in one batch.
    chunk_size: int = 0

    def __init__(self):
        self._input_dim = int(self.input_dim or self.obs_dim)
        self._target_dim = int(self.target_dim if self.target_dim is not None else self._input_dim)
        self._layers = int(self.layers or self.mlp_layers)
        self._models = int(self.models or self.n_models)
        self._scale = float(self.scale)
        self._lambda = float(min(max(self.jrd_lambda, 0.0), 1.0))
        self._min_log_var = float(self.min_log_var)
        self._max_log_var = float(self.max_log_var)
        self._normalize_rewards = bool(self.normalize_rewards)
        self._eps = float(self.eps)
        if self._input_dim <= 0 or self._target_dim <= 0:
            raise ValueError("MAX input and target dimensions must be positive.")
        if self._models < 2:
            raise ValueError("MAX requires at least two ensemble models.")
        if self.chunk_size < 0:
            raise ValueError("MAX chunk_size must be nonnegative (zero disables chunking).")
        if self._min_log_var >= self._max_log_var:
            raise ValueError("MAX min_log_var must be < max_log_var.")
        if self._normalize_rewards:
            self.reward_mean = nj.Variable(jnp.zeros, (), jnp.float32, name="reward_mean")
            self.reward_count = nj.Variable(jnp.zeros, (), jnp.float32, name="reward_count")

    @property
    def _dims(self):
        hidden = [self.hidden_dim] * self._layers
        # Each member outputs a diagonal Gaussian: mean and log-variance.
        return [self._input_dim + self.action_dim, *hidden, 2 * self._target_dim]

    def _weight(self, name, in_dim, out_dim):
        shape = (self._models, in_dim, out_dim)

        def init(shape):
            limit = math.sqrt(6.0 / (in_dim + out_dim))
            return jax.random.uniform(nj.seed(), shape, jnp.float32, -limit, limit)

        return self.value(name, init, shape)

    def _bias(self, name, out_dim):
        return self.value(name, jnp.zeros, (self._models, out_dim), jnp.float32)

    def _inputs(self, obs, action=None):
        obs = f32(obs)
        if self.action_dim:
            if action is None:
                raise ValueError("Action-conditioned MAX requires action inputs.")
            obs = jnp.concatenate([obs, f32(action)], -1)
        return obs

    def _forward_models(self, x):
        x = f32(x)
        for index, (in_dim, out_dim) in enumerate(zip(self._dims[:-1], self._dims[1:])):
            weight = self._weight(f"w{index}", in_dim, out_dim)
            bias = self._bias(f"b{index}", out_dim)
            x = jnp.einsum("m...i,mio->m...o", x, weight)
            x = x + bias.reshape((self._models,) + (1,) * (x.ndim - 2) + (out_dim,))
            if index < len(self._dims) - 2:
                x = act(self.act)(x)
        return f32(x)

    def _split(self, out):
        mean, log_var = jnp.split(out, 2, axis=-1)
        log_var = jnp.clip(log_var, self._min_log_var, self._max_log_var)
        return mean, log_var

    def predict(self, obs, action=None):
        inputs = self._inputs(obs, action)
        inputs = jnp.broadcast_to(inputs[None], (self._models, *inputs.shape))
        out = self._forward_models(inputs)
        return self._split(out)

    def _rescaled_var(self, log_var):
        # Optional temperature on variances (paper Sec 2.4): shrink toward the
        # upper bound so the divergence is dominated by mean disagreement.
        # var_hat = var_U - lambda * (var_U - var) = (1 - lambda) var_U + lambda var.
        var = jnp.exp(log_var)
        var_upper = math.exp(self._max_log_var)
        var_hat = (1.0 - self._lambda) * var_upper + self._lambda * var
        return jnp.maximum(var_hat, self._eps)

    def _jrd(self, mean, log_var):
        """Jensen-Renyi Divergence (quadratic Renyi entropy), MAX Eq. 9.

        mean, log_var: [N, ..., D]. Returns [..., 1] per-transition utility.
        """
        var = self._rescaled_var(log_var)  # rescaled Sigma_hat, [N, ..., D]
        log_var_hat = jnp.log(var)         # used consistently in both JRD terms
        dim = mean.shape[-1]
        if self.chunk_size == 0:
            omega = var[:, None] + var[None, :]
            delta_sq = jnp.square(mean[None, :] - mean[:, None])
            log_d = -0.5 * jnp.sum(jnp.log(omega), axis=-1) \
                - 0.5 * jnp.sum(delta_sq / omega, axis=-1)
            log_mix = jax.nn.logsumexp(log_d, axis=(0, 1))
            avg_log_det = jnp.mean(jnp.sum(log_var_hat, axis=-1), axis=0)
            jrd = (2.0 * math.log(self._models) - log_mix
                   - 0.5 * avg_log_det - 0.5 * dim * math.log(2.0))
            return jnp.maximum(jrd, 0.0)[..., None]
        # Reduce model-pair rows without allocating an [N,N,...,D] tensor.
        def pair_row(_, pair):
            mean_i, var_i = pair
            omega = var_i[None] + var
            delta_sq = jnp.square(mean - mean_i[None])
            log_d = -0.5 * jnp.sum(jnp.log(omega), axis=-1) \
                - 0.5 * jnp.sum(delta_sq / omega, axis=-1)
            return None, jax.nn.logsumexp(log_d, axis=0)

        _, log_rows = jax.lax.scan(jax.checkpoint(pair_row), None, (mean, var))
        # H2(mixture) term: -ln( (1/N^2) sum_ij D_ij ) = 2 ln N - logsumexp(log_d).
        log_mix = jax.nn.logsumexp(log_rows, axis=0)    # [...]
        h2_mixture = 2.0 * math.log(self._models) - log_mix
        # Average per-model H2(N_i): (1/N) sum_i (1/2) ln|Sigma_hat_i| (+ const).
        avg_log_det = jnp.mean(jnp.sum(log_var_hat, axis=-1), axis=0)  # [...]
        const = 0.5 * dim * math.log(2.0)
        jrd = h2_mixture - 0.5 * avg_log_det - const
        # JRD is a (non-negative) divergence; clamp tiny negatives from numerics.
        jrd = jnp.maximum(jrd, 0.0)
        return jrd[..., None]

    def get_disagreement(self, obs, action=None, next_obs=None):
        del next_obs
        if self.chunk_size == 0:
            return self._jrd(*self.predict(obs, action))
        inputs = self._inputs(obs, action)
        flat = inputs.reshape((-1, inputs.shape[-1]))

        def evaluate(chunk):
            expanded = jnp.broadcast_to(chunk[None], (self._models, *chunk.shape))
            return self._jrd(*self._split(self._forward_models(expanded)))

        reward = chunked_rows(evaluate, flat, self.chunk_size)
        return reward.reshape((*inputs.shape[:-1], 1))

    def compute_reward(self, obs, action=None, next_obs=None, update_reward_norm=True, **kwargs):
        del next_obs, kwargs
        bonus = self.get_disagreement(obs, action)
        if self._normalize_rewards:
            bonus = self._normalize_reward(bonus, update_reward_norm)
        return bonus * self._scale

    def loss(self, obs, action, target, mask, bootstrap=True):
        inputs = self._inputs(obs, action)
        target = f32(target)
        mask = f32(mask)
        if self.chunk_size == 0:
            return self._dense_loss(inputs, target, mask, bootstrap)
        batch = inputs.shape[0]
        sequence = math.prod(inputs.shape[1:-1])
        if bootstrap:
            indices = jax.random.randint(nj.seed(), (self._models, batch), 0, batch)
        else:
            indices = jnp.broadcast_to(jnp.arange(batch), (self._models, batch))
        inputs = inputs.reshape((-1, inputs.shape[-1]))
        target = target.reshape((-1, target.shape[-1]))
        mask = mask.reshape(-1)

        def evaluate(rows):
            selected = indices[:, rows // sequence] * sequence + rows[None] % sequence
            mean, log_var = self._split(self._forward_models(inputs[selected]))
            inv_var = jnp.exp(-log_var)
            nll = 0.5 * (jnp.square(mean - target[selected]) * inv_var + log_var).sum(-1)
            # Row first, model second, so chunking stacks only scalar statistics.
            return nll.T, log_var.mean(-1).T, mask[selected].T

        nll, log_var, weights = chunked_rows(
            evaluate, jnp.arange(batch * sequence), self.chunk_size
        )
        denom = jnp.maximum(weights.sum(0), 1.0)
        per_model = (nll * weights).sum(0) / denom
        loss = per_model.mean()
        metrics = {
            "loss": loss,
            "nll": nll.mean(),
            "mean_log_var": log_var.mean(),
        }
        for index in range(self._models):
            metrics[f"model{index}_loss"] = per_model[index]
        return loss, metrics

    def _dense_loss(self, inputs, target, mask, bootstrap):
        """Full-batch bootstrap and Gaussian likelihood."""
        if bootstrap:
            batch = inputs.shape[0]
            indices = jax.random.randint(nj.seed(), (self._models, batch), 0, batch)
            inputs, target, mask = inputs[indices], target[indices], mask[indices]
        else:
            inputs, target, mask = [
                jnp.broadcast_to(value[None], (self._models, *value.shape))
                for value in (inputs, target, mask)
            ]
        mean, log_var = self._split(self._forward_models(inputs))
        nll = 0.5 * (jnp.square(mean - target) * jnp.exp(-log_var) + log_var).sum(-1)
        axes = tuple(range(1, mask.ndim))
        per_model = (nll * mask).sum(axes) / jnp.maximum(mask.sum(axes), 1.0)
        loss = per_model.mean()
        metrics = {"loss": loss, "nll": nll.mean(), "mean_log_var": log_var.mean()}
        metrics.update({f"model{index}_loss": value for index, value in enumerate(per_model)})
        return loss, metrics

    def update(self, obs, action=None, next_obs=None, optimizer=None, **kwargs):
        del kwargs
        if optimizer is not None:
            raise NotImplementedError(
                "JAX MAX is updated by the enclosing Ninjax train transform.")
        if next_obs is None:
            return {}
        mask = jnp.ones(next_obs.shape[:-1], bool)
        loss, metrics = self.loss(obs, action, next_obs, mask, bootstrap=False)
        return {"max_loss": loss, "max_nll": metrics["nll"]}

    def get_modules(self):
        return {"max_ensemble": self}

    def _normalize_reward(self, reward, update):
        if update:
            x = jax.lax.stop_gradient(f32(reward))
            batch_mean = jnp.abs(x).mean()
            batch_count = jnp.asarray(x.size, jnp.float32)
            total = self.reward_count.read() + batch_count
            total_safe = jnp.maximum(total, 1.0)
            mean = (
                self.reward_mean.read() * self.reward_count.read()
                + batch_mean * batch_count
            ) / total_safe
            self.reward_mean.write(mean)
            self.reward_count.write(total)
        denom = jnp.maximum(self.reward_mean.read().astype(reward.dtype), self._eps)
        return reward / denom
