"""JAX Random Network Distillation curiosity module."""

import jax
import jax.numpy as jnp
import ninjax as nj

from .._runtime import f32
from .common import MLP, RunningMeanNormalizer, masked_mean


class RND(nj.Module):

    obs_dim: int = 0
    hidden_dim: int = 256
    output_dim: int = 512
    mlp_layers: int = 2
    layers: int = 0
    scale: float = 1.0
    normalize_rewards: bool = True
    eps: float = 1e-8
    act: str = "relu"

    def __init__(self):
        self._obs_dim = int(self.obs_dim)
        self._layers = int(self.layers or self.mlp_layers)
        self._scale = float(self.scale)
        self._normalize_rewards = bool(self.normalize_rewards)
        if self._obs_dim <= 0:
            raise ValueError("RND obs_dim must be positive.")
        self.target = MLP(
            input_dim=self._obs_dim,
            output_dim=int(self.output_dim),
            hidden_dim=int(self.hidden_dim),
            layers=self._layers,
            act=self.act,
            name="target",
        )
        self.predictor = MLP(
            input_dim=self._obs_dim,
            output_dim=int(self.output_dim),
            hidden_dim=int(self.hidden_dim),
            layers=self._layers,
            act=self.act,
            name="predictor",
        )
        self.reward_rms = (
            RunningMeanNormalizer(eps=float(self.eps), name="reward_rms")
            if self._normalize_rewards
            else None
        )

    def features(self, obs):
        target = self.target(obs, stop_params=True)
        pred = self.predictor(obs)
        return jax.lax.stop_gradient(target), pred

    def forward(self, obs):
        target, pred = self.features(obs)
        return jnp.square(target - pred).mean(axis=-1, keepdims=True)

    def compute_reward(self, obs, action=None, next_obs=None, update_reward_norm=True, **kwargs):
        del action, next_obs, kwargs
        reward = self.forward(obs)
        if self.reward_rms is not None:
            reward = self.reward_rms.normalize(reward, update=update_reward_norm)
        return reward * self._scale

    def loss(self, obs, mask=None):
        error = self.forward(obs)
        loss = masked_mean(error[..., 0], mask)
        metrics = {
            "rnd_loss": loss,
            "rnd_prediction_error": loss,
        }
        if self.reward_rms is not None:
            metrics["rnd_reward_mean"] = self.reward_rms.mean.read()
            metrics["rnd_reward_count"] = self.reward_rms.count.read()
        return loss, metrics

    def update(self, obs, action=None, next_obs=None, optimizer=None, **kwargs):
        del action, next_obs, kwargs
        if optimizer is not None:
            raise NotImplementedError("JAX RND is updated by the enclosing Ninjax optimizer.")
        _, metrics = self.loss(obs)
        return metrics

    def get_modules(self):
        return {"rnd_target": self.target, "rnd_predictor": self.predictor}
