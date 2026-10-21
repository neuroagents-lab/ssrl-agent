"""JAX gamma-learning-progress curiosity module."""

import jax.numpy as jnp
import ninjax as nj

from .._runtime import SlowModel, f32
from .common import MLP, masked_mean


class GammaLearningProgress(nj.Module):

    obs_dim: int = 0
    action_dim: int = 0
    hidden_dim: int = 256
    n_layers: int = 2
    mlp_layers: int = 0
    scale: float = 1.0
    gamma: float = 0.9995
    act: str = "relu"

    def __init__(self):
        self._obs_dim = int(self.obs_dim)
        self._action_dim = int(self.action_dim)
        self._layers = int(self.mlp_layers or self.n_layers)
        self._scale = float(self.scale)
        if self._obs_dim <= 0:
            raise ValueError("Learning progress obs_dim must be positive.")
        input_dim = self._obs_dim + max(0, self._action_dim)
        self.dynamics = MLP(
            input_dim=input_dim,
            output_dim=self._obs_dim,
            hidden_dim=int(self.hidden_dim),
            layers=self._layers,
            act=self.act,
            name="dynamics",
        )
        rate = 1.0 - float(self.gamma)
        self.trailing_dynamics = SlowModel(
            MLP(
                input_dim=input_dim,
                output_dim=self._obs_dim,
                hidden_dim=int(self.hidden_dim),
                layers=self._layers,
                act=self.act,
                name="trailing_dynamics",
            ),
            source=self.dynamics,
            rate=rate,
            every=1,
        )

    def _input(self, obs, action):
        obs = f32(obs)
        if self._action_dim > 0:
            if action is None:
                raise ValueError("Learning progress requires action when action_dim > 0.")
            return jnp.concatenate([obs, f32(action)], axis=-1)
        return obs

    def forward(self, obs, action, next_obs):
        model_input = self._input(obs, action)
        current_pred = self.dynamics(model_input)
        trailing_pred = self.trailing_dynamics(model_input)
        target = f32(next_obs)
        current_loss = jnp.square(current_pred - target).mean(axis=-1)
        trailing_loss = jnp.square(trailing_pred - target).mean(axis=-1)
        return current_loss, trailing_loss

    def compute_reward(self, obs, action, next_obs, **kwargs):
        del kwargs
        current_loss, trailing_loss = self.forward(obs, action, next_obs)
        return (trailing_loss - current_loss)[..., None] * self._scale

    def loss(self, obs, action, next_obs, mask=None):
        current_loss, trailing_loss = self.forward(obs, action, next_obs)
        loss = masked_mean(current_loss, mask)
        metrics = {
            "progress_loss": loss,
            "progress_current_error": masked_mean(current_loss, mask),
            "progress_trailing_error": masked_mean(trailing_loss, mask),
        }
        metrics["gamma_progress_updated"] = jnp.ones((), jnp.float32)
        return loss, metrics

    def update(self, obs=None, action=None, next_obs=None, optimizer=None, return_loss=False, **kwargs):
        del kwargs
        if optimizer is not None:
            raise NotImplementedError(
                "JAX learning progress is updated by the enclosing Ninjax optimizer."
            )
        if obs is None or next_obs is None:
            metrics = {"gamma_progress_updated": jnp.ones((), jnp.float32)}
            return (None, metrics) if return_loss else metrics
        loss, metrics = self.loss(obs, action, next_obs)
        return (loss, metrics) if return_loss else metrics

    def post_update(self):
        self.trailing_dynamics.update()

    def get_modules(self):
        return {
            "progress_dynamics": self.dynamics,
            "progress_trailing_model": self.trailing_dynamics.model,
        }
