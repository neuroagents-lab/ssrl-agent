"""JAX empowerment competency modules."""

import math

import jax
import jax.numpy as jnp
import ninjax as nj

from .._runtime import SlowModel, f32
from ..curiosity.common import MLP, masked_mean
from .common import ConditionalDiagonalGaussian, PotentialActionHead, diag_gaussian_log_prob


class _ChannelCapacityMixin:
    obs_dim: int
    action_dim: int
    particles: int
    next_samples: int
    reward_chunk_size: int
    action_concat: bool
    state_conditional: ConditionalDiagonalGaussian

    def _action_dim_value(self):
        return int(getattr(self, "_action_dim", self.action_dim))

    def _next_samples_value(self):
        return int(getattr(self, "_next_samples", self.next_samples))

    def _action_concat_value(self):
        return bool(getattr(self, "_action_concat", self.action_concat))

    def _conditional_input(self, obs, action):
        obs = f32(obs)
        action_dim = self._action_dim_value()
        if action_dim <= 0:
            return obs
        if action is None:
            raise ValueError("Channel-capacity empowerment requires action particles.")
        action = f32(action)
        obs_exp = obs
        while obs_exp.ndim < action.ndim:
            obs_exp = jnp.expand_dims(obs_exp, axis=-2)
        obs_exp = jnp.broadcast_to(obs_exp, (*action.shape[:-1], obs.shape[-1]))
        return jnp.concatenate([obs_exp, action], axis=-1)

    def _conditional_hidden_action(self, action):
        if not self._action_concat_value():
            return None
        if action is None:
            raise ValueError("Action concatenation requires actions.")
        return f32(action)

    def channel_log_prob(self, obs, action, next_obs, stop_params=False):
        cond_in = self._conditional_input(obs, action)
        target = f32(next_obs)
        while target.ndim < cond_in.ndim:
            target = jnp.expand_dims(target, axis=-2)
        target = jnp.broadcast_to(target, (*cond_in.shape[:-1], target.shape[-1]))
        return self.state_conditional.log_prob(
            target,
            cond_in,
            hidden_condition=self._conditional_hidden_action(action),
            stop_params=stop_params,
        )

    def _channel_params(self, obs, action, stop_params=False):
        return self.state_conditional(
            self._conditional_input(obs, action),
            hidden_condition=self._conditional_hidden_action(action),
            stop_params=stop_params,
        )

    def _channel_nll(self, obs, action, next_obs):
        return -self.channel_log_prob(obs, action, next_obs).mean()

    def estimate_mi(
        self,
        obs,
        action_particles,
        action_logits=None,
        next_samples=None,
        reward_chunk_size=None,
        stop_params=False,
    ):
        del reward_chunk_size
        action_particles = f32(action_particles)
        if action_particles.ndim != f32(obs).ndim + 1:
            raise ValueError(
                "action_particles must have shape obs.shape[:-1] + "
                f"(K, action_dim); got obs={f32(obs).shape}, actions={action_particles.shape}."
            )
        action_dim = self._action_dim_value()
        if action_particles.shape[-1] != action_dim:
            raise ValueError(f"Expected action_dim={action_dim}, got {action_particles.shape[-1]}.")
        particles = action_particles.shape[-2]
        if action_logits is None:
            log_weights = jnp.full(action_particles.shape[:-1], -math.log(float(particles)), action_particles.dtype)
        else:
            if action_logits.shape != action_particles.shape[:-1]:
                raise ValueError("action_logits must have shape action_particles.shape[:-1].")
            log_weights = jax.nn.log_softmax(f32(action_logits), axis=-1)
        samples = int(next_samples if next_samples is not None else self._next_samples_value())
        mean, log_std = self._channel_params(obs, action_particles, stop_params=stop_params)
        eps = jax.random.normal(nj.seed(), (*mean.shape[:-1], samples, mean.shape[-1]), mean.dtype)
        next_draws = mean[..., :, None, :] + eps * jnp.exp(log_std[..., :, None, :])
        log_source = diag_gaussian_log_prob(
            next_draws,
            mean[..., :, None, :],
            log_std[..., :, None, :],
        )
        log_all = diag_gaussian_log_prob(
            next_draws[..., :, :, None, :],
            mean[..., None, None, :, :],
            log_std[..., None, None, :, :],
        )
        log_marg = jax.nn.logsumexp(log_all + log_weights[..., None, None, :, None], axis=-2)
        mi_by_action = (log_source - log_marg).mean(axis=-2)[..., 0]
        weights = jnp.exp(log_weights)
        return (weights * mi_by_action).sum(axis=-1, keepdims=True)


class ForwardEmpowerment(nj.Module):

    obs_dim: int = 0
    action_dim: int = 0
    hidden_dim: int = 256
    n_layers: int = 2
    scale: float = 1.0
    action_concat: bool = False
    act: str = "relu"

    def __init__(self):
        self._obs_dim = int(self.obs_dim)
        self._action_dim = int(self.action_dim)
        self._scale = float(self.scale)
        self._action_concat = bool(self.action_concat and self._action_dim > 0)
        self.state_prior = ConditionalDiagonalGaussian(
            input_dim=self._obs_dim,
            target_dim=self._obs_dim,
            hidden_dim=int(self.hidden_dim),
            n_layers=int(self.n_layers),
            act=self.act,
            name="state_prior",
        )
        self.state_conditional = ConditionalDiagonalGaussian(
            input_dim=self._obs_dim + self._action_dim if self._action_dim > 0 else self._obs_dim,
            target_dim=self._obs_dim,
            hidden_dim=int(self.hidden_dim),
            n_layers=int(self.n_layers),
            hidden_condition_dim=self._action_dim if self._action_concat else 0,
            act=self.act,
            name="state_conditional",
        )

    def _conditional_input(self, obs, action):
        if self._action_dim <= 0:
            return f32(obs)
        if action is None:
            raise ValueError("Forward empowerment requires actions.")
        return jnp.concatenate([f32(obs), f32(action)], axis=-1)

    def _hidden_action(self, action):
        return f32(action) if self._action_concat else None

    def _log_terms(self, obs, action, next_obs):
        log_cond = self.state_conditional.log_prob(
            next_obs,
            self._conditional_input(obs, action),
            hidden_condition=self._hidden_action(action),
        )
        log_prior = self.state_prior.log_prob(next_obs, obs)
        return log_cond, log_prior

    def compute_reward(self, obs, action=None, next_obs=None, **kwargs):
        del kwargs
        log_cond, log_prior = self._log_terms(obs, action, next_obs)
        return self._scale * (log_cond - log_prior)

    def loss(self, obs, action, next_obs, mask=None):
        log_cond, log_prior = self._log_terms(obs, action, next_obs)
        loss_cond = -log_cond[..., 0].mean() if mask is None else -(log_cond[..., 0] * f32(mask)).sum() / jnp.maximum(f32(mask).sum(), 1.0)
        loss_prior = -log_prior[..., 0].mean() if mask is None else -(log_prior[..., 0] * f32(mask)).sum() / jnp.maximum(f32(mask).sum(), 1.0)
        loss = loss_cond + loss_prior
        return loss, {
            "empowerment_forward_loss": loss,
            "empowerment_forward_prior_nll": loss_prior,
            "empowerment_forward_conditional_nll": loss_cond,
            "empowerment_forward_mi": (log_cond - log_prior).mean(),
        }

    def update(self, obs=None, action=None, next_obs=None, optimizer=None, return_loss=False, **kwargs):
        if optimizer is not None:
            raise NotImplementedError("JAX empowerment is updated by the enclosing Ninjax optimizer.")
        loss, metrics = self.loss(obs, action, next_obs, **kwargs)
        return (loss, metrics) if return_loss else metrics

    def get_modules(self):
        return {
            "empowerment_forward_prior": self.state_prior,
            "empowerment_forward_conditional": self.state_conditional,
        }


class ReverseEmpowerment(nj.Module):

    obs_dim: int = 0
    action_dim: int = 0
    hidden_dim: int = 256
    n_layers: int = 2
    scale: float = 1.0
    act: str = "relu"

    def __init__(self):
        self._obs_dim = int(self.obs_dim)
        self._action_dim = int(self.action_dim)
        self._scale = float(self.scale)
        self.action_prior = ConditionalDiagonalGaussian(
            input_dim=self._obs_dim,
            target_dim=self._action_dim,
            hidden_dim=int(self.hidden_dim),
            n_layers=int(self.n_layers),
            act=self.act,
            name="action_prior",
        )
        self.action_posterior = ConditionalDiagonalGaussian(
            input_dim=2 * self._obs_dim,
            target_dim=self._action_dim,
            hidden_dim=int(self.hidden_dim),
            n_layers=int(self.n_layers),
            act=self.act,
            name="action_posterior",
        )

    def _posterior_input(self, obs, next_obs):
        return jnp.concatenate([f32(obs), f32(next_obs)], axis=-1)

    def _log_terms(self, obs, action, next_obs):
        log_prior = self.action_prior.log_prob(action, obs)
        log_post = self.action_posterior.log_prob(action, self._posterior_input(obs, next_obs))
        return log_post, log_prior

    def compute_reward(self, obs, action=None, next_obs=None, **kwargs):
        del kwargs
        log_post, log_prior = self._log_terms(obs, action, next_obs)
        return self._scale * (log_post - log_prior)

    def loss(self, obs, action, next_obs, mask=None):
        del mask
        log_post, log_prior = self._log_terms(obs, action, next_obs)
        loss_prior = -log_prior.mean()
        loss_post = -log_post.mean()
        loss = loss_prior + loss_post
        return loss, {
            "empowerment_reverse_loss": loss,
            "empowerment_reverse_prior_nll": loss_prior,
            "empowerment_reverse_posterior_nll": loss_post,
            "empowerment_reverse_mi": (log_post - log_prior).mean(),
        }

    def update(self, obs=None, action=None, next_obs=None, optimizer=None, return_loss=False, **kwargs):
        if optimizer is not None:
            raise NotImplementedError("JAX empowerment is updated by the enclosing Ninjax optimizer.")
        loss, metrics = self.loss(obs, action, next_obs, **kwargs)
        return (loss, metrics) if return_loss else metrics

    def get_modules(self):
        return {
            "empowerment_reverse_prior": self.action_prior,
            "empowerment_reverse_posterior": self.action_posterior,
        }


class EmpowermentGain(nj.Module):

    obs_dim: int = 0
    action_dim: int = 0
    hidden_dim: int = 256
    n_layers: int = 2
    scale: float = 1.0
    gamma: float = 0.9995
    gce_latent_dim: int = 64
    gce_power: float = 1.0
    gce_waterfill_iters: int = 32
    gce_recon_scale: float = 1.0
    gce_state_noise: bool = False
    act: str = "relu"

    def __init__(self):
        self._obs_dim = int(self.obs_dim)
        self._action_dim = int(self.action_dim)
        self._scale = float(self.scale)
        self._gamma = float(self.gamma)
        if not 0.5 < self._gamma < 1.0:
            raise ValueError("Empowerment gain gamma must be between 0.5 and 1.0.")

        channel_kwargs = dict(
            obs_dim=self._obs_dim,
            action_dim=self._action_dim,
            latent_dim=int(self.gce_latent_dim),
            hidden_dim=int(self.hidden_dim),
            n_layers=int(self.n_layers),
            power=float(self.gce_power),
            waterfill_iters=int(self.gce_waterfill_iters),
            recon_scale=float(self.gce_recon_scale),
            state_noise=bool(self.gce_state_noise),
            act=self.act,
        )
        self.gce = _GaussianChannel(**channel_kwargs, name="gce")
        self.trailing_gce = SlowModel(
            _GaussianChannel(**channel_kwargs, name="trailing_gce"),
            source=self.gce,
            rate=1.0 - self._gamma,
            every=1,
        )

    def _ensure_initialized(self, obs, action, next_obs):
        # Capacity does not touch the decoder or drift network. During Ninjax
        # initialization, materialize the full fast channel before SlowModel
        # clones its parameter tree.
        if nj.creating():
            if action is None or next_obs is None:
                raise ValueError(
                    "Empowerment gain initialization requires action and next_obs."
                )
            self.gce.prediction_loss(obs, action, next_obs)

    def capacities(self, obs):
        current = self.gce.capacity(obs)
        trailing = self.trailing_gce.capacity(obs)
        return current, trailing

    def compute_reward(self, obs, action=None, next_obs=None, **kwargs):
        del kwargs
        self._ensure_initialized(obs, action, next_obs)
        current, trailing = self.capacities(obs)
        return self._scale * (current - trailing)

    def loss(self, obs, action, next_obs, mask=None):
        loss, mets = self.gce.prediction_loss(obs, action, next_obs, mask)
        current, trailing = self.capacities(obs)
        gain = current - trailing
        return loss, {
            "empowerment_gain_loss": loss,
            "empowerment_gain_channel_nll": mets["channel_nll"],
            "empowerment_gain_recon": mets["channel_recon"],
            "empowerment_gain_log_var": mets["channel_log_var"],
            "empowerment_gain_current_capacity": current.mean(),
            "empowerment_gain_trailing_capacity": trailing.mean(),
            "empowerment_gain_mean": gain.mean(),
            "empowerment_gain_gamma": jnp.asarray(self._gamma, jnp.float32),
        }

    def update(
        self,
        obs=None,
        action=None,
        next_obs=None,
        optimizer=None,
        return_loss=False,
        **kwargs,
    ):
        del kwargs
        if optimizer is not None:
            raise NotImplementedError(
                "JAX empowerment is updated by the enclosing Ninjax optimizer."
            )
        loss, metrics = self.loss(obs, action, next_obs)
        return (loss, metrics) if return_loss else metrics

    def post_update(self):
        self.trailing_gce.update()

    def get_trainable_modules(self):
        return {"empowerment_gain_gce": self.gce}

    def get_modules(self):
        return {
            "empowerment_gain_gce": self.gce,
            "empowerment_gain_trailing_gce": self.trailing_gce.model,
        }


class _GaussianChannel(nj.Module):
    """Latent Gaussian channel for empowerment estimation.

    Zhao, Lu, Abbeel & Tiomkin (ICLR 2021), "Efficient Empowerment Estimation
    for Unsupervised Stabilization".

    Learn f(s') ~ N(drift(f(s)) + G(f(s)) * a, Sigma_eta) using reconstruction
    and channel prediction losses (Eq. 5-6). The action encoder is identity.

    Potential empowerment maximizes 1/2 log|I + Sigma_eta^-1 G Sigma_a G^T|
    over action covariances with trace <= power, using water-filling
    (Eq. 11-12). Process empowerment evaluates the same expression under
    the policy's action covariance. Sylvester's identity reduces both
    determinants to action_dim x action_dim matrices.
    """

    obs_dim: int = 0
    action_dim: int = 0
    latent_dim: int = 64
    hidden_dim: int = 256
    n_layers: int = 2
    power: float = 1.0
    waterfill_iters: int = 32
    recon_scale: float = 1.0
    state_noise: bool = False
    min_log_var: float = -10.0
    max_log_var: float = 2.0
    ridge: float = 1e-4
    eps: float = 1e-8
    act: str = "relu"

    def __init__(self):
        self._obs_dim = int(self.obs_dim)
        self._action_dim = int(self.action_dim)
        self._latent_dim = int(self.latent_dim)
        self._power = float(self.power)
        self._iters = int(self.waterfill_iters)
        self._recon_scale = float(self.recon_scale)
        self._state_noise = bool(self.state_noise)
        self._min_log_var = float(self.min_log_var)
        self._max_log_var = float(self.max_log_var)
        self._ridge = float(self.ridge)
        self._eps = float(self.eps)
        if self._obs_dim <= 0 or self._action_dim <= 0 or self._latent_dim <= 0:
            raise ValueError("Gaussian channel requires positive obs_dim, action_dim, latent_dim.")
        mlp = lambda name, din, dout: MLP(
            input_dim=din, output_dim=dout, hidden_dim=int(self.hidden_dim),
            layers=int(self.n_layers), act=self.act, name=name)
        self.state_enc = mlp("state_enc", self._obs_dim, self._latent_dim)
        self.state_dec = mlp("state_dec", self._latent_dim, self._obs_dim)
        self.drift = mlp("drift", self._latent_dim, self._latent_dim)
        self.channel = mlp("channel", self._latent_dim, self._latent_dim * self._action_dim)
        if self._state_noise:
            self.log_noise = mlp("log_noise", self._latent_dim, self._latent_dim)
        else:
            self.log_noise = nj.Variable(jnp.zeros, (self._latent_dim,), jnp.float32, name="log_noise")

    def encode(self, obs):
        return self.state_enc(f32(obs))

    def _matrix(self, z):
        flat = self.channel(z)
        return flat.reshape((*z.shape[:-1], self._latent_dim, self._action_dim))

    def _log_noise_var(self, z):
        if self._state_noise:
            lv = self.log_noise(z)
        else:
            lv = jnp.broadcast_to(self.log_noise.read(), (*z.shape[:-1], self._latent_dim))
        return jnp.clip(lv, self._min_log_var, self._max_log_var)

    def _predict_latent(self, z, action):
        G = self._matrix(z)
        return self.drift(z) + jnp.einsum("...ij,...j->...i", G, f32(action))

    def prediction_loss(self, obs, action, next_obs, mask=None):
        z = self.encode(obs)
        z_next = self.encode(next_obs)
        recon = self.state_dec(z)
        recon_loss = jnp.square(recon - f32(obs)).mean(-1)
        pred = self._predict_latent(z, action)
        log_var = self._log_noise_var(z)
        nll = 0.5 * (jnp.square(z_next - pred) * jnp.exp(-log_var) + log_var).mean(-1)
        loss = masked_mean(nll, mask) + self._recon_scale * masked_mean(recon_loss, mask)
        metrics = {
            "channel_nll": masked_mean(nll, mask),
            "channel_recon": masked_mean(recon_loss, mask),
            "channel_log_var": log_var.mean(),
        }
        return loss, metrics

    def _gram(self, obs):
        # G^T Sigma_eta^{-1} G  = (Sigma_eta^{-1/2} G)^T (Sigma_eta^{-1/2} G), [..., a, a]
        z = self.encode(obs)
        G = self._matrix(z)
        inv_std = jnp.exp(-0.5 * self._log_noise_var(z))
        gw = inv_std[..., :, None] * G                       # [..., df, a]
        return jnp.einsum("...ij,...ik->...jk", gw, gw)      # [..., a, a]

    def capacity(self, obs):
        gram = self._gram(obs)
        gains = jnp.clip(jnp.linalg.eigvalsh(gram), 0.0, None)   # [..., a] (>=0)
        power = self._waterfill(gains)
        cap = 0.5 * jnp.sum(jnp.log1p(gains * power), axis=-1, keepdims=True)
        return jnp.maximum(cap, 0.0)

    def policy_mi(self, obs, action_cov):
        gram = self._gram(obs)                                # [..., a, a]
        cov = f32(action_cov) + self._ridge * jnp.eye(self._action_dim, dtype=jnp.float32)
        mat = jnp.eye(self._action_dim, dtype=jnp.float32) + jnp.einsum("...ij,...jk->...ik", cov, gram)
        _, logdet = jnp.linalg.slogdet(mat)                   # log|I + Sigma_a G^T Sigma_eta^-1 G|
        return jnp.maximum(0.5 * logdet[..., None], 0.0)

    def _waterfill(self, gains):
        # p_i = max(w - 1/gain_i, 0), sum_i p_i = P, solved by bisection on water level w.
        inv_gain = 1.0 / jnp.maximum(gains, self._eps)        # [..., a]
        base = jnp.min(inv_gain, axis=-1)                     # [...] tight bracket
        w_lo = base
        w_hi = base + self._power
        for _ in range(self._iters):
            w_mid = 0.5 * (w_lo + w_hi)
            total = jnp.sum(jnp.maximum(w_mid[..., None] - inv_gain, 0.0), axis=-1)
            over = total > self._power
            w_hi = jnp.where(over, w_mid, w_hi)
            w_lo = jnp.where(over, w_lo, w_mid)
        w = 0.5 * (w_lo + w_hi)
        return jnp.maximum(w[..., None] - inv_gain, 0.0)

    def get_modules(self):
        mods = {
            "gce_state_enc": self.state_enc,
            "gce_state_dec": self.state_dec,
            "gce_drift": self.drift,
            "gce_channel": self.channel,
        }
        if self._state_noise:
            mods["gce_log_noise"] = self.log_noise
        return mods


class PotentialEmpowerment(_ChannelCapacityMixin, nj.Module):

    obs_dim: int = 0
    action_dim: int = 0
    hidden_dim: int = 256
    n_layers: int = 2
    scale: float = 1.0
    particles: int = 16
    next_samples: int = 1
    q_loss_weight: float = 1.0
    action_concat: bool = False
    reward_chunk_size: int = 0
    estimator: str = "gce"
    gce_latent_dim: int = 64
    gce_power: float = 1.0
    gce_waterfill_iters: int = 32
    gce_recon_scale: float = 1.0
    gce_state_noise: bool = False
    act: str = "relu"

    def __init__(self):
        self._obs_dim = int(self.obs_dim)
        self._action_dim = int(self.action_dim)
        self._particles = int(self.particles)
        self._next_samples = int(self.next_samples)
        self._reward_chunk_size = int(self.reward_chunk_size)
        self._action_concat = bool(self.action_concat and self._action_dim > 0)
        self._scale = float(self.scale)
        self._q_loss_weight = float(self.q_loss_weight)
        self._estimator = str(self.estimator).lower()
        if self._estimator not in {"gce", "vlb"}:
            raise ValueError("Potential empowerment estimator must be 'gce' or 'vlb'.")
        if self._estimator == "gce":
            self.gce = _GaussianChannel(
                obs_dim=self._obs_dim,
                action_dim=self._action_dim,
                latent_dim=int(self.gce_latent_dim),
                hidden_dim=int(self.hidden_dim),
                n_layers=int(self.n_layers),
                power=float(self.gce_power),
                waterfill_iters=int(self.gce_waterfill_iters),
                recon_scale=float(self.gce_recon_scale),
                state_noise=bool(self.gce_state_noise),
                act=self.act,
                name="gce",
            )
            return
        self.state_conditional = ConditionalDiagonalGaussian(
            input_dim=self._obs_dim + self._action_dim if self._action_dim > 0 else self._obs_dim,
            target_dim=self._obs_dim,
            hidden_dim=int(self.hidden_dim),
            n_layers=int(self.n_layers),
            hidden_condition_dim=self._action_dim if self._action_concat else 0,
            act=self.act,
            name="state_conditional",
        )
        self.action_head = PotentialActionHead(
            obs_dim=self._obs_dim,
            action_dim=self._action_dim,
            particles=self._particles,
            hidden_dim=int(self.hidden_dim),
            n_layers=int(self.n_layers),
            act=self.act,
            name="action_head",
        )

    def _potential_actions(self, obs):
        return self.action_head(obs)

    def compute_reward(self, obs, action=None, next_obs=None, **kwargs):
        del action, next_obs
        if self._estimator == "gce":
            # Channel capacity by water-filling (potential empowerment).
            return self._scale * self.gce.capacity(obs)
        actions, logits = self._potential_actions(obs)
        mi = self.estimate_mi(obs, actions, action_logits=logits, reward_chunk_size=kwargs.pop("reward_chunk_size", None))
        return self._scale * mi

    def q_loss(self, obs):
        actions, logits = self._potential_actions(obs)
        mi = self.estimate_mi(obs, actions, action_logits=logits, stop_params=True)
        return -mi.mean()

    def loss(self, obs, action, next_obs, mask=None):
        if self._estimator == "gce":
            loss, mets = self.gce.prediction_loss(obs, action, next_obs, mask)
            metrics = {
                "empowerment_potential_loss": loss,
                "empowerment_potential_channel_nll": mets["channel_nll"],
                "empowerment_potential_recon": mets["channel_recon"],
                "empowerment_potential_log_var": mets["channel_log_var"],
            }
            return loss, metrics
        del mask
        channel_loss = self._channel_nll(obs, action, next_obs)
        q_loss = self.q_loss(obs)
        loss = channel_loss + self._q_loss_weight * q_loss
        return loss, {
            "empowerment_potential_loss": loss,
            "empowerment_potential_channel_nll": channel_loss,
            "empowerment_potential_q_loss": q_loss,
            "empowerment_potential_mi": -q_loss,
            "empowerment_potential_particles": jnp.asarray(self._particles, jnp.float32),
            "empowerment_potential_next_samples": jnp.asarray(self._next_samples, jnp.float32),
        }

    def update(self, obs=None, action=None, next_obs=None, optimizer=None, return_loss=False, **kwargs):
        if optimizer is not None:
            raise NotImplementedError("JAX empowerment is updated by the enclosing Ninjax optimizer.")
        loss, metrics = self.loss(obs, action, next_obs, **kwargs)
        return (loss, metrics) if return_loss else metrics

    def get_modules(self):
        if self._estimator == "gce":
            return {f"empowerment_potential_{k}": v for k, v in self.gce.get_modules().items()}
        return {
            "empowerment_potential_channel": self.state_conditional,
            "empowerment_potential_action_head": self.action_head,
        }


class ProcessEmpowerment(_ChannelCapacityMixin, nj.Module):

    obs_dim: int = 0
    action_dim: int = 0
    hidden_dim: int = 256
    n_layers: int = 2
    scale: float = 1.0
    particles: int = 16
    next_samples: int = 1
    action_concat: bool = False
    reward_chunk_size: int = 0
    estimator: str = "gce"
    gce_latent_dim: int = 64
    gce_waterfill_iters: int = 32
    gce_recon_scale: float = 1.0
    gce_state_noise: bool = False
    ridge: float = 1e-4
    act: str = "relu"

    def __init__(self):
        self._obs_dim = int(self.obs_dim)
        self._action_dim = int(self.action_dim)
        self._particles = int(self.particles)
        self._next_samples = int(self.next_samples)
        self._reward_chunk_size = int(self.reward_chunk_size)
        self._action_concat = bool(self.action_concat and self._action_dim > 0)
        self._scale = float(self.scale)
        self._ridge = float(self.ridge)
        self._estimator = str(self.estimator).lower()
        if self._estimator not in {"gce", "vlb"}:
            raise ValueError("Process empowerment estimator must be 'gce' or 'vlb'.")
        if self._estimator == "gce":
            self.gce = _GaussianChannel(
                obs_dim=self._obs_dim,
                action_dim=self._action_dim,
                latent_dim=int(self.gce_latent_dim),
                hidden_dim=int(self.hidden_dim),
                n_layers=int(self.n_layers),
                power=1.0,
                waterfill_iters=int(self.gce_waterfill_iters),
                recon_scale=float(self.gce_recon_scale),
                state_noise=bool(self.gce_state_noise),
                ridge=self._ridge,
                act=self.act,
                name="gce",
            )
            return
        self.state_conditional = ConditionalDiagonalGaussian(
            input_dim=self._obs_dim + self._action_dim if self._action_dim > 0 else self._obs_dim,
            target_dim=self._obs_dim,
            hidden_dim=int(self.hidden_dim),
            n_layers=int(self.n_layers),
            hidden_condition_dim=self._action_dim if self._action_concat else 0,
            act=self.act,
            name="state_conditional",
        )

    def _action_covariance(self, particles):
        # Empirical covariance of the policy's action distribution at each state.
        particles = f32(particles)                              # [..., K, A]
        mean = particles.mean(axis=-2, keepdims=True)
        centered = particles - mean
        count = max(int(particles.shape[-2]), 1)
        return jnp.einsum("...ka,...kb->...ab", centered, centered) / float(count)

    def compute_reward(self, obs, action=None, next_obs=None, action_particles=None, process_action_particles=None, action_logits=None, **kwargs):
        del action, next_obs
        particles = action_particles if action_particles is not None else process_action_particles
        if particles is None:
            raise ValueError("Process empowerment reward requires action_particles.")
        if self._estimator == "gce":
            # One-step MI under the policy's action covariance (feedback-capacity
            # per step); no inner max — the RL optimizer maximizes accumulated MI.
            cov = self._action_covariance(particles)
            return self._scale * self.gce.policy_mi(obs, cov)
        mi = self.estimate_mi(
            obs,
            particles,
            action_logits=action_logits,
            reward_chunk_size=kwargs.pop("reward_chunk_size", None),
        )
        return self._scale * mi

    def loss(self, obs, action, next_obs, mask=None):
        if self._estimator == "gce":
            loss, mets = self.gce.prediction_loss(obs, action, next_obs, mask)
            return loss, {
                "empowerment_process_loss": loss,
                "empowerment_process_channel_nll": mets["channel_nll"],
                "empowerment_process_recon": mets["channel_recon"],
                "empowerment_process_log_var": mets["channel_log_var"],
            }
        del mask
        loss = self._channel_nll(obs, action, next_obs)
        return loss, {
            "empowerment_process_loss": loss,
            "empowerment_process_channel_nll": loss,
            "empowerment_process_particles": jnp.asarray(self._particles, jnp.float32),
            "empowerment_process_next_samples": jnp.asarray(self._next_samples, jnp.float32),
        }

    def update(self, obs=None, action=None, next_obs=None, optimizer=None, return_loss=False, **kwargs):
        if optimizer is not None:
            raise NotImplementedError("JAX empowerment is updated by the enclosing Ninjax optimizer.")
        loss, metrics = self.loss(obs, action, next_obs, **kwargs)
        return (loss, metrics) if return_loss else metrics

    def get_modules(self):
        if self._estimator == "gce":
            return {f"empowerment_process_{k}": v for k, v in self.gce.get_modules().items()}
        return {"empowerment_process_channel": self.state_conditional}
