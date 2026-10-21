"""Conditional Information Gain (CIG) curiosity.

Joseph et al. (2026), "CIG: Exploration via Conditional Information Gain",
https://arxiv.org/abs/2605.20878v1.

CIG reuses a Plan2Explore ensemble of deterministic one-step predictors. For
each imagined rollout it forms the trace-reduced Gram matrix of centered
ensemble predictions (paper Eq. 6), adds a scalar aleatoric ridge estimated
from replay residuals (Eq. 9), and reads causal per-step rewards from the
Cholesky diagonal (Eq. 8 and Algorithm 1).
"""

import math

import jax
import jax.numpy as jnp
import ninjax as nj

from .._runtime import f32, get_data_axes
from .disagreement import Disagreement


def _sequence_mask(value, name, batch, horizon):
    if value is None:
        return None
    value = jnp.asarray(value)
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.shape != (batch, horizon):
        raise ValueError(
            f"CIG {name} must have shape {(batch, horizon)}, got {value.shape}."
        )
    return value.astype(bool)


def _global_sum(value):
    """Sum replicated state updates across data axes."""
    axes = get_data_axes()
    return jax.lax.psum(value, axes) if axes else value


def _global_weighted_moments(value, weight):
    """Return a shard-consistent weighted mean, variance, and count."""
    value = f32(value)
    weight = f32(weight)
    while weight.ndim < value.ndim:
        weight = weight[..., None]
    weight = jnp.broadcast_to(weight, value.shape)
    selected = weight.astype(bool)
    value = jnp.where(selected, value, jnp.zeros((), value.dtype))
    count = _global_sum(weight.sum())
    total = _global_sum((value * weight).sum())
    mean = total / jnp.maximum(count, 1.0)
    squared = _global_sum((jnp.square(value - mean) * weight).sum())
    variance = squared / jnp.maximum(count, 1.0)
    return mean, variance, count


def _segmented_cholesky_diagonal(features, ridge, is_first, valid):
    """Causal reorthogonalized QR for replay sequences with resets."""
    batch, horizon, feature_dim = features.shape
    feature_basis = jnp.zeros(
        (batch, horizon, feature_dim), features.dtype
    )
    noise_basis = jnp.zeros((batch, horizon, horizon), features.dtype)
    units = jnp.eye(horizon, dtype=features.dtype)

    def step(carry, inputs):
        feature_basis, noise_basis = carry
        feature, reset, keep, unit, index = inputs
        retain = (~reset)[:, None, None]
        feature_basis = jnp.where(retain, feature_basis, 0.0)
        noise_basis = jnp.where(retain, noise_basis, 0.0)
        feature = jnp.where(keep[:, None], feature, 0.0)
        noise = jnp.where(
            keep[:, None], jnp.sqrt(ridge) * unit[None], 0.0
        )

        # Two modified Gram--Schmidt passes retain the tiny independent-noise
        # component even after nearly collinear epistemic directions.
        for _ in range(2):
            coefficients = jnp.einsum(
                "bkp,bp->bk",
                feature_basis,
                feature,
                precision=jax.lax.Precision.HIGHEST,
            )
            coefficients += jnp.einsum(
                "bkt,bt->bk",
                noise_basis,
                noise,
                precision=jax.lax.Precision.HIGHEST,
            )
            feature -= jnp.einsum(
                "bk,bkp->bp",
                coefficients,
                feature_basis,
                precision=jax.lax.Precision.HIGHEST,
            )
            noise -= jnp.einsum(
                "bk,bkt->bt",
                coefficients,
                noise_basis,
                precision=jax.lax.Precision.HIGHEST,
            )

        squared = jnp.square(feature).sum(-1) + jnp.square(noise).sum(-1)
        inverse = jax.lax.rsqrt(
            jnp.maximum(squared, jnp.finfo(features.dtype).tiny)
        )
        new_feature = feature * inverse[:, None]
        new_noise = noise * inverse[:, None]
        feature_basis = feature_basis.at[:, index].set(
            jnp.where(keep[:, None], new_feature, 0.0)
        )
        noise_basis = noise_basis.at[:, index].set(
            jnp.where(keep[:, None], new_noise, 0.0)
        )
        return (feature_basis, noise_basis), jnp.sqrt(squared)

    inputs = (
        jnp.swapaxes(features, 0, 1),
        jnp.swapaxes(is_first, 0, 1),
        jnp.swapaxes(valid, 0, 1),
        units,
        jnp.arange(horizon),
    )
    _, diagonal = jax.lax.scan(
        step, (feature_basis, noise_basis), inputs
    )
    return jnp.swapaxes(diagonal, 0, 1)


def _regularized_gram_qr_diagonal(
    features, ridge, *, is_first=None, valid=None
):
    """QR fallback for ``diag(chol(FF' + ridge * I))`` in float32.

    Forming the regularized Gram matrix directly can discard a small paper
    ridge when the epistemic Gram diagonal is much larger. Instead, let
    ``A = [F, sqrt(ridge) I]``. QR of ``A'`` gives
    ``FF' + ridge I = R'R``, so ``abs(diag(R))`` is the requested Cholesky
    diagonal without squaring the condition number. Householder QR also keeps
    the chronological columns prefix-causal. Replay resets use the equivalent
    causal, reorthogonalized incremental QR above.
    """
    batch, horizon, _ = features.shape
    if is_first is not None:
        if valid is None:
            valid = jnp.ones((batch, horizon), bool)
        return _segmented_cholesky_diagonal(
            features, ridge, is_first, valid
        )
    eye = jnp.broadcast_to(
        jnp.eye(horizon, dtype=features.dtype),
        (batch, horizon, horizon),
    )
    augmented = jnp.concatenate(
        [features, jnp.sqrt(ridge).astype(features.dtype) * eye], axis=-1
    )
    triangular = jnp.linalg.qr(
        jnp.swapaxes(augmented, -1, -2), mode="r"
    )
    return jnp.abs(jnp.diagonal(triangular, axis1=-2, axis2=-1))


def conditional_information_gain(
    predictions,
    aleatoric_var,
    *,
    is_first=None,
    valid=None,
    ridge_scale=1.0,
    eps=1e-8,
    detach=True,
):
    r"""Compute CIG rewards for predictions shaped ``[M, B, T, D]``.

    ``aleatoric_var`` is the scalar :math:`\hat\sigma^2` from paper Eq. 9.
    The returned rewards have shape ``[B, T, 1]`` and sum along each valid,
    uninterrupted rollout to ``logdet(K + aleatoric_var * D * I)``.
    Optional masks exclude padding and separate episode boundaries.
    """
    predictions = f32(predictions)
    if predictions.ndim != 4:
        raise ValueError(
            "CIG predictions must have shape [models,batch,time,features], "
            f"got {predictions.shape}."
        )
    models, batch, horizon, feature_dim = predictions.shape
    if models < 2 or feature_dim <= 0:
        raise ValueError("CIG requires at least two models and one feature.")
    if jnp.shape(aleatoric_var) != ():
        raise ValueError(
            "CIG aleatoric_var must be a scalar, "
            f"got shape {jnp.shape(aleatoric_var)}."
        )
    ridge_scale = float(ridge_scale)
    eps = float(eps)
    if ridge_scale <= 0:
        raise ValueError("CIG ridge_scale must be positive.")
    if eps < 0:
        raise ValueError("CIG eps must be non-negative.")
    if horizon == 0:
        return jnp.zeros((batch, 0, 1), predictions.dtype)

    # Appendix A.2 detaches ensemble outputs on the reward path. The ensemble
    # is trained only by its replay MSE below.
    if detach:
        predictions = jax.lax.stop_gradient(predictions)

    deviations = predictions - predictions.mean(axis=0, keepdims=True)
    # F has rows delta^(t) / sqrt(M), concatenated over ensemble members and
    # target dimensions. Therefore F F^T is exactly Eq. 6 with population 1/M.
    features = jnp.transpose(deviations, (1, 2, 0, 3))
    features = features.reshape((batch, horizon, models * feature_dim))
    features = features / math.sqrt(models)

    valid = _sequence_mask(valid, "valid", batch, horizon)
    if valid is None:
        valid = jnp.ones((batch, horizon), bool)
    features = jnp.where(
        valid[..., None], features, jnp.zeros((), features.dtype)
    )

    # Paper Eq. 6: K[j,t] = 1/M sum_k delta[k,j]' delta[k,t].
    kernel = jnp.einsum(
        "bti,bsi->bts",
        features,
        features,
        precision=jax.lax.Precision.HIGHEST,
    )
    is_first = _sequence_mask(is_first, "is_first", batch, horizon)
    if is_first is not None:
        segment = jnp.cumsum(is_first.astype(jnp.int32), axis=1)
        same_segment = segment[:, :, None] == segment[:, None, :]
        kernel = jnp.where(
            same_segment, kernel, jnp.zeros((), kernel.dtype)
        )
    kernel = 0.5 * (kernel + jnp.swapaxes(kernel, -1, -2))

    # Algorithm 1 uses sigma_hat^2 * d, with eps as a numerical floor.
    aleatoric_var = jax.lax.stop_gradient(f32(aleatoric_var))
    ridge = ridge_scale * aleatoric_var * feature_dim
    ridge = jnp.maximum(ridge, jnp.asarray(eps, predictions.dtype))

    # Use augmented QR when float32 cannot resolve the ridge in the Gram matrix.
    trace_scale = jnp.max(
        jnp.diagonal(kernel, axis1=-2, axis2=-1).sum(-1)
    )
    stability_limit = (
        32.0 * horizon * jnp.finfo(kernel.dtype).eps * trace_scale
    )
    eye = jnp.eye(horizon, dtype=kernel.dtype)[None]

    def paper_cholesky(_):
        chol = jnp.linalg.cholesky(kernel + ridge * eye)
        return jnp.diagonal(chol, axis1=-2, axis2=-1)

    def stable_qr(_):
        return _regularized_gram_qr_diagonal(
            features, ridge, is_first=is_first, valid=valid
        )

    diagonal = jax.lax.cond(
        ridge <= stability_limit, stable_qr, paper_cholesky, operand=None
    )
    diagonal = jnp.where(valid, diagonal, jnp.ones_like(diagonal))
    reward = 2.0 * jnp.log(diagonal)
    reward = jnp.where(valid, reward, jnp.zeros_like(reward))
    return reward[..., None]


class CIG(nj.Module):
    """Plan2Explore ensemble with the paper's trajectory-level CIG reward."""

    obs_dim: int = 0
    input_dim: int = 0
    action_dim: int = 0
    target_dim: int | None = None
    hidden_dim: int = 400
    mlp_layers: int = 4
    layers: int = 0
    n_models: int = 5
    models: int = 0
    scale: float = 1.0
    ridge_scale: float = 1.0
    aleatoric_ema: float = 0.99
    normalize_rewards: bool = True
    reward_ema: float = 0.99
    eps: float = 1e-8
    act: str = "elu"

    def __init__(self):
        self._input_dim = int(self.input_dim or self.obs_dim)
        self._target_dim = int(
            self.target_dim if self.target_dim is not None else self._input_dim
        )
        self._layers = int(self.layers or self.mlp_layers)
        self._models = int(self.models or self.n_models)
        self._scale = float(self.scale)
        self._ridge_scale = float(self.ridge_scale)
        self._aleatoric_ema = float(self.aleatoric_ema)
        self._normalize_rewards = bool(self.normalize_rewards)
        self._reward_ema = float(self.reward_ema)
        self._eps = float(self.eps)
        if self._input_dim <= 0 or self._target_dim <= 0:
            raise ValueError("CIG input and target dimensions must be positive.")
        if self._models < 2:
            raise ValueError("CIG requires at least two ensemble models.")
        if self._layers < 1 or int(self.hidden_dim) <= 0:
            raise ValueError("CIG requires positive hidden dimensions and layers.")
        if self._ridge_scale <= 0 or self._eps <= 0:
            raise ValueError("CIG ridge_scale and eps must be positive.")
        if not 0.0 <= self._aleatoric_ema < 1.0:
            raise ValueError("CIG aleatoric_ema must be in [0, 1).")
        if not 0.0 <= self._reward_ema < 1.0:
            raise ValueError("CIG reward_ema must be in [0, 1).")

        # Separate predictor weights from EMA state to exclude EMAs from optimization.
        self.ensemble = Disagreement(
            input_dim=self._input_dim,
            action_dim=int(self.action_dim),
            target_dim=self._target_dim,
            hidden_dim=int(self.hidden_dim),
            mlp_layers=self._layers,
            n_models=self._models,
            scale=1.0,
            use_log_reward=False,
            normalize_rewards=False,
            eps=self._eps,
            act=self.act,
            name="ensemble",
        )
        self.aleatoric_var = nj.Variable(
            jnp.zeros, (), jnp.float32, name="aleatoric_var"
        )
        self.aleatoric_initialized = nj.Variable(
            jnp.zeros, (), jnp.float32, name="aleatoric_initialized"
        )
        if self._normalize_rewards:
            self.reward_mean = nj.Variable(
                jnp.zeros, (), jnp.float32, name="reward_mean"
            )
            self.reward_var = nj.Variable(
                jnp.ones, (), jnp.float32, name="reward_var"
            )
            self.reward_initialized = nj.Variable(
                jnp.zeros, (), jnp.float32, name="reward_initialized"
            )

    def _canonical_inputs(self, obs, action, sequence_layout):
        obs = f32(obs)
        if sequence_layout not in {"batch_major", "time_major"}:
            raise ValueError(
                f"Unsupported CIG sequence_layout={sequence_layout!r}."
            )
        squeezed = obs.ndim == 2
        if squeezed:
            obs = obs[:, None]
            if action is not None:
                action = f32(action)[:, None]
        elif obs.ndim != 3:
            raise ValueError(
                "CIG rewards expect [B,D], [B,T,D], or [T,B,D] inputs, "
                f"got {obs.shape}."
            )
        elif action is not None:
            action = f32(action)
        if not squeezed and sequence_layout == "time_major":
            obs = jnp.swapaxes(obs, 0, 1)
            if action is not None:
                action = jnp.swapaxes(action, 0, 1)
        return obs, action, squeezed

    def _canonical_mask(self, value, squeezed, sequence_layout):
        if value is None:
            return None
        value = jnp.asarray(value)
        if squeezed:
            if value.ndim == 2 and value.shape[-1] == 1:
                value = value[..., 0]
            if value.ndim != 1:
                raise ValueError(
                    f"CIG one-step masks must have shape [B], got {value.shape}."
                )
            return value[:, None]
        if sequence_layout == "time_major":
            value = jnp.swapaxes(value, 0, 1)
        return value

    def predict(self, obs, action=None):
        return self.ensemble.predict(obs, action)

    def get_disagreement(
        self,
        obs,
        action=None,
        next_obs=None,
        *,
        sequence_layout="batch_major",
        is_first=None,
        valid=None,
    ):
        del next_obs
        obs, action, squeezed = self._canonical_inputs(
            obs, action, sequence_layout
        )
        is_first = self._canonical_mask(is_first, squeezed, sequence_layout)
        valid = self._canonical_mask(valid, squeezed, sequence_layout)
        predictions = self.predict(obs, action)
        reward = conditional_information_gain(
            predictions,
            self.aleatoric_var.read(),
            is_first=is_first,
            valid=valid,
            ridge_scale=self._ridge_scale,
            eps=self._eps,
            detach=True,
        )
        if squeezed:
            return reward[:, 0]
        if sequence_layout == "time_major":
            reward = jnp.swapaxes(reward, 0, 1)
        return reward

    def raw_reward(self, obs, action=None, **kwargs):
        return self.get_disagreement(obs, action, **kwargs)

    def compute_reward(
        self,
        obs,
        action=None,
        next_obs=None,
        update_reward_norm=True,
        *,
        sequence_layout="batch_major",
        is_first=None,
        valid=None,
        **kwargs,
    ):
        del kwargs
        reward = self.get_disagreement(
            obs,
            action,
            next_obs,
            sequence_layout=sequence_layout,
            is_first=is_first,
            valid=valid,
        )
        if self._normalize_rewards:
            reward = self._normalize_reward(reward, update_reward_norm, valid)
        return reward * self._scale

    def loss(
        self,
        obs,
        action,
        target,
        mask,
        *,
        update_aleatoric=True,
    ):
        """Train every ensemble member on the same replay transitions by MSE."""
        inputs = self.ensemble._inputs(obs, action)
        target = f32(target)
        mask = f32(mask)
        # Sanitize padding before the MLP to prevent NaNs in masked gradients.
        active = mask.astype(bool)
        inputs = jnp.where(
            active[..., None], inputs, jnp.zeros((), inputs.dtype)
        )
        target = jnp.where(
            active[..., None], target, jnp.zeros((), target.dtype)
        )
        inputs = jnp.broadcast_to(inputs[None], (self._models, *inputs.shape))
        targets = jnp.broadcast_to(target[None], (self._models, *target.shape))
        predictions = self.ensemble._forward_models(inputs)

        error = jnp.square(predictions - targets).mean(axis=-1)
        reduce_axes = tuple(range(1, error.ndim))
        model_mask = jnp.broadcast_to(mask[None], error.shape)
        error = jnp.where(
            model_mask.astype(bool), error, jnp.zeros((), error.dtype)
        )
        denom = jnp.maximum(model_mask.sum(reduce_axes), 1.0)
        per_model = (error * model_mask).sum(reduce_axes) / denom
        loss = per_model.mean()

        ensemble_mean = predictions.mean(axis=0)
        residual = jnp.square(target - ensemble_mean).mean(axis=-1)
        batch_aleatoric, _, valid_count = _global_weighted_moments(
            residual, mask
        )
        if update_aleatoric:
            self._update_aleatoric(batch_aleatoric, valid_count > 0)

        metrics = {
            "loss": loss,
            "error": (error * model_mask).sum() / jnp.maximum(
                model_mask.sum(), 1.0
            ),
            "aleatoric_batch": batch_aleatoric,
            "aleatoric_var": self.aleatoric_var.read(),
        }
        for index in range(self._models):
            metrics[f"model{index}_loss"] = per_model[index]
        return loss, metrics

    def _update_aleatoric(self, batch_value, has_valid):
        batch_value = jax.lax.stop_gradient(f32(batch_value))
        old = self.aleatoric_var.read()
        initialized = self.aleatoric_initialized.read() > 0
        ema = self._aleatoric_ema * old + (1.0 - self._aleatoric_ema) * batch_value
        updated = jnp.where(initialized, ema, batch_value)
        self.aleatoric_var.write(jnp.where(has_valid, updated, old))
        self.aleatoric_initialized.write(
            jnp.maximum(
                self.aleatoric_initialized.read(), f32(has_valid)
            )
        )

    def _normalize_reward(self, reward, update, valid=None):
        if valid is None:
            weight = jnp.ones_like(reward)
        else:
            weight = f32(valid)
            while weight.ndim < reward.ndim:
                weight = weight[..., None]
            weight = jnp.broadcast_to(weight, reward.shape)

        if update:
            value = jax.lax.stop_gradient(f32(reward))
            weight = jax.lax.stop_gradient(weight)
            batch_mean, batch_var, count = _global_weighted_moments(
                value, weight
            )
            has_valid = count > 0
            initialized = self.reward_initialized.read() > 0
            old_mean = self.reward_mean.read()
            old_var = self.reward_var.read()
            mean = self._reward_ema * old_mean + (
                1.0 - self._reward_ema
            ) * batch_mean
            # Include between-mean variance when merging the EMA and batch distributions.
            var = (
                self._reward_ema * old_var
                + (1.0 - self._reward_ema) * batch_var
                + self._reward_ema
                * (1.0 - self._reward_ema)
                * jnp.square(old_mean - batch_mean)
            )
            mean = jnp.where(initialized, mean, batch_mean)
            var = jnp.where(initialized, var, batch_var)
            self.reward_mean.write(jnp.where(has_valid, mean, old_mean))
            self.reward_var.write(jnp.where(has_valid, var, old_var))
            self.reward_initialized.write(
                jnp.maximum(self.reward_initialized.read(), f32(has_valid))
            )

        mean = self.reward_mean.read().astype(reward.dtype)
        var = self.reward_var.read().astype(reward.dtype)
        normalized = (reward - mean) / jnp.sqrt(jnp.maximum(var, self._eps))
        normalized = jnp.where(
            self.reward_initialized.read() > 0, normalized, jnp.zeros_like(reward)
        )
        return jnp.where(weight.astype(bool), normalized, jnp.zeros_like(reward))

    def update(self, obs, action=None, next_obs=None, optimizer=None, **kwargs):
        if optimizer is not None:
            raise NotImplementedError(
                "JAX CIG is updated by the enclosing Ninjax train transform."
            )
        if next_obs is None:
            return {}
        mask = kwargs.pop("mask", jnp.ones(next_obs.shape[:-1], bool))
        update_aleatoric = bool(kwargs.pop("update_aleatoric", True))
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected CIG update arguments: {unknown}.")
        loss, metrics = self.loss(
            obs,
            action,
            next_obs,
            mask,
            update_aleatoric=update_aleatoric,
        )
        return {
            "cig_loss": loss,
            "cig_error": metrics["error"],
            "cig_aleatoric_var": metrics["aleatoric_var"],
        }

    def get_modules(self):
        return {"cig_ensemble": self.ensemble}
