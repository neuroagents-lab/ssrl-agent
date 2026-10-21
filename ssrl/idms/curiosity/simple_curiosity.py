"""Forward and inverse prediction curiosity."""

import jax.numpy as jnp
import ninjax as nj

from .._runtime import f32
from .common import MLP, masked_mean, softmax_cross_entropy


class SimpleCuriosity(nj.Module):
    """Forward and inverse prediction over shared world-model features."""

    obs_dim: int = 0
    action_dim: int = 0
    hidden_dim: int = 256
    n_layers: int = 1
    mlp_layers: int = 0
    scale: float = 1.0
    discrete_actions: bool = False
    act: str = "relu"

    def __init__(self):
        self._obs_dim = int(self.obs_dim)
        self._action_dim = int(self.action_dim)
        self._layers = int(self.mlp_layers or self.n_layers)
        self._scale = float(self.scale)
        self._discrete_actions = bool(self.discrete_actions)
        if self._obs_dim <= 0 or self._action_dim <= 0:
            raise ValueError("Simple curiosity obs_dim and action_dim must be positive.")
        self.forward_net = MLP(
            input_dim=self._obs_dim + self._action_dim,
            output_dim=self._obs_dim,
            hidden_dim=int(self.hidden_dim),
            layers=self._layers,
            act=self.act,
            name="forward_net",
        )
        self.inverse_net = MLP(
            input_dim=2 * self._obs_dim,
            output_dim=self._action_dim,
            hidden_dim=int(self.hidden_dim),
            layers=self._layers,
            act=self.act,
            name="inverse_net",
        )

    def predict_next(self, obs, action):
        return self.forward_net(
            jnp.concatenate([f32(obs), f32(action)], axis=-1)
        )

    def predict_action(self, obs, next_obs):
        return self.inverse_net(
            jnp.concatenate([f32(obs), f32(next_obs)], axis=-1)
        )

    def forward(self, obs, action, next_obs):
        pred = self.predict_next(obs, action)
        return jnp.linalg.norm(
            f32(next_obs) - pred, axis=-1, keepdims=True
        )

    def loss_terms(self, obs, action, next_obs, mask=None):
        pred_next = self.predict_next(obs, action)
        forward_error = jnp.square(pred_next - f32(next_obs)).mean(axis=-1)
        forward_loss = masked_mean(forward_error, mask)
        pred_action = self.predict_action(obs, next_obs)
        if self._discrete_actions:
            if action.shape[-1] == self._action_dim:
                target = jnp.argmax(action, axis=-1)
            else:
                target = jnp.squeeze(action, axis=-1).astype(jnp.int32)
            inverse_error = softmax_cross_entropy(pred_action, target)
        else:
            inverse_error = jnp.square(pred_action - f32(action)).mean(axis=-1)
        inverse_loss = masked_mean(inverse_error, mask)
        reward_error = jnp.linalg.norm(
            f32(next_obs) - pred_next, axis=-1, keepdims=True
        )
        return forward_loss, inverse_loss, reward_error

    def compute_reward(self, obs, action, next_obs, **kwargs):
        del kwargs
        return self.forward(obs, action, next_obs) * self._scale

    def loss(self, obs, action, next_obs, mask=None):
        forward_loss, inverse_loss, forward_error = self.loss_terms(
            obs, action, next_obs, mask
        )
        total = forward_loss + inverse_loss
        metrics = {
            "simple_curiosity_loss": total,
            "simple_curiosity_total_loss": total,
            "simple_curiosity_forward_loss": forward_loss,
            "simple_curiosity_inverse_loss": inverse_loss,
            "simple_curiosity_forward_error": masked_mean(forward_error[..., 0], mask),
        }
        return total, metrics

    def update(
        self, obs, action, next_obs, optimizer=None, return_loss=False, **kwargs
    ):
        del kwargs
        if optimizer is not None:
            raise NotImplementedError(
                "JAX simple curiosity is updated by the enclosing Ninjax optimizer."
            )
        loss, metrics = self.loss(obs, action, next_obs)
        if return_loss:
            return loss, metrics
        return metrics

    def get_modules(self):
        return {"simple_curiosity": self}
