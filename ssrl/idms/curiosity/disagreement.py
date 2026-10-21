"""JAX disagreement curiosity module for Plan2Explore."""

import math

import jax
import jax.numpy as jnp
import ninjax as nj

from .._runtime import act, f32


class Disagreement(nj.Module):

    obs_dim: int = 0
    input_dim: int = 0
    action_dim: int = 0
    target_dim: int | None = None
    hidden_dim: int = 1024
    mlp_layers: int = 2
    layers: int = 0
    n_models: int = 5
    models: int = 0
    scale: float = 10000.0
    use_log_reward: bool = False
    normalize_rewards: bool = False
    eps: float = 1e-8
    act: str = "relu"

    def __init__(self):
        self._input_dim = int(self.input_dim or self.obs_dim)
        self._target_dim = int(self.target_dim if self.target_dim is not None else self._input_dim)
        self._layers = int(self.layers or self.mlp_layers)
        self._models = int(self.models or self.n_models)
        self._scale = float(self.scale)
        self._use_log_reward = bool(self.use_log_reward)
        self._normalize_rewards = bool(self.normalize_rewards)
        self._eps = float(self.eps)
        if self._input_dim <= 0 or self._target_dim <= 0:
            raise ValueError("Disagreement input and target dimensions must be positive.")
        if self._models < 2:
            raise ValueError("Disagreement requires at least two ensemble models.")
        if self._normalize_rewards:
            self.reward_mean = nj.Variable(jnp.zeros, (), jnp.float32, name="reward_mean")
            self.reward_count = nj.Variable(jnp.zeros, (), jnp.float32, name="reward_count")

    @property
    def _dims(self):
        hidden = [self.hidden_dim] * self._layers
        return [self._input_dim + self.action_dim, *hidden, self._target_dim]

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
                raise ValueError("Action-conditioned disagreement requires action inputs.")
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

    def predict(self, obs, action=None):
        inputs = self._inputs(obs, action)
        inputs = jnp.broadcast_to(inputs[None], (self._models, *inputs.shape))
        return self._forward_models(inputs)

    def predict_model(self, model_idx, obs, action=None):
        return self.predict(obs, action)[int(model_idx)]

    def model_error(self, model_idx, obs, action=None, next_obs=None):
        pred = self.predict_model(model_idx, obs, action)
        if next_obs is None:
            error = jnp.square(pred)
        else:
            error = jnp.square(f32(next_obs) - pred)
        return error.mean(-1, keepdims=True)

    def forward(self, obs, action=None, next_obs=None):
        pred = self.predict(obs, action)
        if next_obs is None:
            error = jnp.square(pred)
        else:
            error = jnp.square(f32(next_obs)[None] - pred)
        return error.mean(-1, keepdims=True)

    def __call__(self, obs, action=None, next_obs=None):
        return self.forward(obs, action, next_obs)

    def loss(self, obs, action, target, mask, bootstrap=True):
        inputs = self._inputs(obs, action)
        target = f32(target)
        mask = f32(mask)
        if bootstrap:
            batch = inputs.shape[0]
            indices = jax.random.randint(nj.seed(), (self._models, batch), 0, batch)
            inputs = inputs[indices]
            target = target[indices]
            mask = mask[indices]
        else:
            inputs = jnp.broadcast_to(inputs[None], (self._models, *inputs.shape))
            target = jnp.broadcast_to(target[None], (self._models, *target.shape))
            mask = jnp.broadcast_to(mask[None], (self._models, *mask.shape))
        pred = self._forward_models(inputs)
        error = jnp.square(pred - target).mean(-1)
        denom = jnp.maximum(mask.sum(tuple(range(1, mask.ndim))), 1.0)
        per_model = (error * mask).sum(tuple(range(1, error.ndim))) / denom
        loss = per_model.mean()
        metrics = {
            "loss": loss,
            "error": error.mean(),
        }
        for index in range(self._models):
            metrics[f"model{index}_loss"] = per_model[index]
        return loss, metrics

    def get_disagreement(self, obs, action=None, next_obs=None):
        del next_obs
        pred = self.predict(obs, action)
        return jnp.var(f32(pred), axis=0).mean(-1, keepdims=True)

    def raw_reward(self, obs, action=None):
        return self.get_disagreement(obs, action)

    def compute_reward(self, obs, action=None, next_obs=None, update_reward_norm=True, **kwargs):
        del kwargs
        bonus = self.get_disagreement(obs, action, next_obs)
        if self._use_log_reward:
            bonus = jnp.log1p(jnp.maximum(bonus, 0))
        if self._normalize_rewards:
            bonus = self._normalize_reward(bonus, update_reward_norm)
        return bonus * self._scale

    def update(self, obs, action=None, next_obs=None, optimizer=None, **kwargs):
        del kwargs
        if optimizer is not None:
            raise NotImplementedError(
                "JAX disagreement is updated by the enclosing Ninjax train transform, "
                "not by passing an optimizer to update()."
            )
        if next_obs is None:
            return {}
        mask = jnp.ones(next_obs.shape[:-1], bool)
        loss, metrics = self.loss(obs, action, next_obs, mask, bootstrap=False)
        result = {
            "disagreement_loss": loss,
            "disagreement_mean_error": metrics["error"],
        }
        for index in range(self._models):
            result[f"disagreement_model_{index}_error"] = metrics[f"model{index}_loss"]
        return result

    def get_modules(self):
        return {"disagreement_ensemble": self}

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
