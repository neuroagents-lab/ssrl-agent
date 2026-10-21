"""JAX Intrinsic Curiosity Module."""

import jax.numpy as jnp
import ninjax as nj

from .._runtime import f32
from .common import MLP, masked_mean, softmax_cross_entropy


class ICM(nj.Module):
    """Pathak et al. ICM with a shared learned feature projector."""

    obs_dim: int = 0
    action_dim: int = 0
    proj_dim: int = 0
    proj_hidden_dim: int = 0
    proj_layers: int = 1
    hidden_dim: int = 256
    n_layers: int = 1
    mlp_layers: int = 0
    forward_loss_weight: float = 0.2
    scale: float = 0.01
    discrete_actions: bool = False
    act: str = "relu"

    def __init__(self):
        self._obs_dim = int(self.obs_dim)
        self._action_dim = int(self.action_dim)
        self._proj_dim = int(self.proj_dim or self._obs_dim)
        self._proj_hidden_dim = int(self.proj_hidden_dim or self.hidden_dim)
        self._proj_layers = int(self.proj_layers)
        self._layers = int(self.mlp_layers or self.n_layers)
        self._forward_loss_weight = float(self.forward_loss_weight)
        self._scale = float(self.scale)
        self._discrete_actions = bool(self.discrete_actions)
        if self._obs_dim <= 0 or self._action_dim <= 0:
            raise ValueError("ICM obs_dim and action_dim must be positive.")
        if (
            self._proj_dim <= 0
            or self._proj_hidden_dim <= 0
            or self._proj_layers < 1
        ):
            raise ValueError(
                "ICM proj_dim and proj_hidden_dim must be positive and "
                "proj_layers must be at least one."
            )
        if not 0.0 <= self._forward_loss_weight <= 1.0:
            raise ValueError("ICM forward_loss_weight must be in [0, 1].")
        self.projector = MLP(
            input_dim=self._obs_dim,
            output_dim=self._proj_dim,
            hidden_dim=self._proj_hidden_dim,
            layers=self._proj_layers,
            act=self.act,
            name="projector",
        )
        self.forward_net = MLP(
            input_dim=self._proj_dim + self._action_dim,
            output_dim=self._proj_dim,
            hidden_dim=int(self.hidden_dim),
            layers=self._layers,
            act=self.act,
            name="forward_net",
        )
        self.inverse_net = MLP(
            input_dim=2 * self._proj_dim,
            output_dim=self._action_dim,
            hidden_dim=int(self.hidden_dim),
            layers=self._layers,
            act=self.act,
            name="inverse_net",
        )

    @property
    def proj_dim_value(self):
        return self._proj_dim

    def project(self, obs):
        """Encode observations into the shared ICM feature space."""
        return f32(self.projector(f32(obs)))

    def project_pair(self, obs, next_obs):
        if obs.shape == next_obs.shape and obs.ndim > 1:
            pair = jnp.concatenate([f32(obs), f32(next_obs)], axis=0)
            features = self.project(pair)
            return jnp.split(features, 2, axis=0)
        return self.project(obs), self.project(next_obs)

    def predict_next_features(self, features, action):
        return self.forward_net(
            jnp.concatenate([f32(features), f32(action)], axis=-1)
        )

    def predict_action_features(self, features, next_features):
        return self.inverse_net(
            jnp.concatenate([f32(features), f32(next_features)], axis=-1)
        )

    def predict_next(self, obs, action):
        return self.predict_next_features(self.project(obs), action)

    def predict_action(self, obs, next_obs):
        features, next_features = self.project_pair(obs, next_obs)
        return self.predict_action_features(features, next_features)

    @staticmethod
    def prediction_error(prediction, target):
        # Pathak et al. (2017), Equation (6):
        # 1/2 ||f(phi(s), a) - phi(s')||^2.
        return 0.5 * jnp.square(f32(prediction) - f32(target)).sum(
            axis=-1, keepdims=True
        )

    def forward(self, obs, action, next_obs):
        features, next_features = self.project_pair(obs, next_obs)
        pred = self.predict_next_features(features, action)
        return self.prediction_error(pred, next_features)

    def loss_terms(self, obs, action, next_obs, mask=None):
        features, next_features = self.project_pair(obs, next_obs)
        pred_next = self.predict_next_features(features, action)
        forward_error = self.prediction_error(pred_next, next_features)
        forward_loss = masked_mean(forward_error[..., 0], mask)
        pred_action = self.predict_action_features(features, next_features)
        if self._discrete_actions:
            if action.shape[-1] == self._action_dim:
                target = jnp.argmax(action, axis=-1)
            else:
                target = jnp.squeeze(action, axis=-1).astype(jnp.int32)
            inverse_error = softmax_cross_entropy(pred_action, target)
        else:
            inverse_error = jnp.square(pred_action - f32(action)).mean(axis=-1)
        inverse_loss = masked_mean(inverse_error, mask)
        return forward_loss, inverse_loss, forward_error

    def compute_reward(self, obs, action, next_obs, **kwargs):
        del kwargs
        return self.forward(obs, action, next_obs) * self._scale

    def loss(self, obs, action, next_obs, mask=None):
        forward_loss, inverse_loss, forward_error = self.loss_terms(
            obs, action, next_obs, mask
        )
        beta = self._forward_loss_weight
        total = beta * forward_loss + (1.0 - beta) * inverse_loss
        metrics = {
            "icm_loss": total,
            "icm_total_loss": total,
            "icm_forward_loss": forward_loss,
            "icm_inverse_loss": inverse_loss,
            "icm_forward_error": masked_mean(forward_error[..., 0], mask),
        }
        return total, metrics

    def update(self, obs, action, next_obs, optimizer=None, return_loss=False, **kwargs):
        del kwargs
        if optimizer is not None:
            raise NotImplementedError("JAX ICM is updated by the enclosing Ninjax optimizer.")
        loss, metrics = self.loss(obs, action, next_obs)
        if return_loss:
            return loss, metrics
        return metrics

    def get_modules(self):
        return {
            "icm": self,
            "icm_projector": self.projector,
            "icm_forward_net": self.forward_net,
            "icm_inverse_net": self.inverse_net,
        }
