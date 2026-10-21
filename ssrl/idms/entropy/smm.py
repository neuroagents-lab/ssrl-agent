"""State Marginal Matching (SMM) with a variational density model.

Lee, Eysenbach, Parisotto, Xing, Levine & Salakhutdinov (2019),
"Efficient Exploration via State Marginal Matching".

A VAE estimates the RSSM state density from replay transitions. The reward
is log p*(s) - ELBO(s), where p* is a uniform target. This implements the
single-policy SMM objective over world-model features.
"""

import math

import jax
import jax.numpy as jnp
import ninjax as nj

from .._runtime import f32
from ..curiosity.common import MLP, RunningMeanStd, masked_mean
from .common import validate_at_least, validate_positive

LOG_2PI = math.log(2.0 * math.pi)


def _diag_gaussian_log_prob(target, mean, log_var):
    inv_var = jnp.exp(-log_var)
    log_prob = -0.5 * (jnp.square(target - mean) * inv_var + log_var + LOG_2PI)
    return log_prob.sum(axis=-1, keepdims=True)


class SMM(nj.Module):

    obs_dim: int = 0
    latent_dim: int = 128
    hidden_dim: int = 150
    n_layers: int = 2
    scale: float = 1.0
    kl_scale: float = 0.5
    target: str = "uniform"
    log_target_const: float = 0.0
    min_log_var: float = -7.0
    max_log_var: float = 2.0
    reward_normalize: bool = False
    eps: float = 1e-8
    act: str = "relu"

    def __init__(self):
        self._obs_dim = int(self.obs_dim)
        self._latent_dim = int(self.latent_dim)
        self._scale = float(self.scale)
        self._kl_scale = float(self.kl_scale)
        self._target = str(self.target).lower()
        self._log_target_const = float(self.log_target_const)
        self._min_log_var = float(self.min_log_var)
        self._max_log_var = float(self.max_log_var)
        self._reward_normalize = bool(self.reward_normalize)
        self._eps = float(self.eps)
        # SMM reward is computed on imagined states; the density model is fit on
        # replay states. Update the reward normalizer from the replay path only.
        self.updates_reward_normalizer_in_update = False

        validate_positive("SMM obs_dim", self._obs_dim)
        validate_positive("SMM latent_dim", self._latent_dim)
        validate_at_least("SMM n_layers", self.n_layers, 1)
        if self._target not in {"uniform"}:
            raise ValueError(
                "SMM supports target='uniform' (constant log p*); "
                f"got {self.target!r}."
            )

        self.encoder = MLP(
            input_dim=self._obs_dim,
            output_dim=2 * self._latent_dim,
            hidden_dim=int(self.hidden_dim),
            layers=int(self.n_layers),
            act=self.act,
            name="encoder",
        )
        self.decoder = MLP(
            input_dim=self._latent_dim,
            output_dim=2 * self._obs_dim,
            hidden_dim=int(self.hidden_dim),
            layers=int(self.n_layers),
            act=self.act,
            name="decoder",
        )
        self.reward_stats = (
            RunningMeanStd(eps=self._eps, name="reward_stats")
            if self._reward_normalize
            else None
        )

    def _encode(self, obs):
        mean, log_var = jnp.split(self.encoder(f32(obs)), 2, axis=-1)
        log_var = jnp.clip(log_var, self._min_log_var, self._max_log_var)
        return mean, log_var

    def _decode(self, z):
        mean, log_var = jnp.split(self.decoder(z), 2, axis=-1)
        log_var = jnp.clip(log_var, self._min_log_var, self._max_log_var)
        return mean, log_var

    def _elbo(self, obs, sample=True):
        """Evidence lower bound on log q(s); shape [..., 1]."""
        obs = f32(obs)
        q_mean, q_log_var = self._encode(obs)
        if sample:
            noise = jax.random.normal(nj.seed(), q_mean.shape, q_mean.dtype)
            z = q_mean + jnp.exp(0.5 * q_log_var) * noise
        else:
            z = q_mean
        dec_mean, dec_log_var = self._decode(z)
        log_p = _diag_gaussian_log_prob(obs, dec_mean, dec_log_var)
        # KL(q(z|s) || N(0, I)) for diagonal Gaussians.
        kl = 0.5 * jnp.sum(
            jnp.exp(q_log_var) + jnp.square(q_mean) - 1.0 - q_log_var,
            axis=-1,
            keepdims=True,
        )
        elbo = log_p - self._kl_scale * kl
        return elbo, log_p, kl

    def _log_target(self, obs):
        # A uniform target has constant log density.
        return jnp.full((*f32(obs).shape[:-1], 1), self._log_target_const, jnp.float32)

    def compute_reward(self, obs, action=None, next_obs=None, update_reward_norm=True, **kwargs):
        del action, next_obs, kwargs
        elbo, _, _ = self._elbo(obs, sample=False)
        log_q = elbo  # ELBO is a lower bound on log q(s).
        reward = self._log_target(obs) - log_q
        if self.reward_stats is not None:
            reward = self.reward_stats.normalize(reward, update=update_reward_norm)
        return self._scale * reward

    def loss(self, obs, mask=None, update_reward_norm=False):
        obs = f32(obs)
        if obs.size == 0:
            zero = jnp.zeros((), jnp.float32)
            return zero, {"smm_elbo": zero, "smm_recon": zero, "smm_kl": zero}
        elbo, log_p, kl = self._elbo(obs, sample=True)
        # Maximize ELBO (fit q to the visited-state marginal) => minimize -ELBO.
        loss = -masked_mean(elbo[..., 0], mask)
        metrics = {
            "smm_loss": loss,
            "smm_elbo": masked_mean(elbo[..., 0], mask),
            "smm_recon": masked_mean(log_p[..., 0], mask),
            "smm_kl": masked_mean(kl[..., 0], mask),
        }
        if update_reward_norm and self.reward_stats is not None:
            reward = self._log_target(obs) - elbo
            self.reward_stats.update(reward)
        return loss, metrics

    def update(self, obs=None, action=None, next_obs=None, optimizer=None, return_loss=False, **kwargs):
        del action, next_obs
        if optimizer is not None:
            raise NotImplementedError("JAX SMM is updated by the enclosing Ninjax optimizer.")
        if obs is None:
            raise ValueError("obs is required to update SMM.")
        loss, metrics = self.loss(obs, **kwargs)
        return (loss, metrics) if return_loss else metrics

    def reset(self):
        if self.reward_stats is not None:
            self.reward_stats.reset()

    def get_modules(self):
        return {"smm_encoder": self.encoder, "smm_decoder": self.decoder}
