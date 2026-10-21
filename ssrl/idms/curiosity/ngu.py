"""JAX Never Give Up curiosity module."""

import jax
import jax.numpy as jnp
import ninjax as nj

from .._runtime import f32
from .common import MLP, RunningMeanStd, masked_mean, softmax_cross_entropy


class NGU(nj.Module):

    obs_dim: int = 0
    action_dim: int = 0
    rep_dim: int = 32
    hidden_dim: int = 256
    n_layers: int = 2
    k: int = 10
    scale: float = 0.3
    kernel_epsilon: float = 1e-4
    cluster_distance: float = 8e-3
    pseudo_count_c: float = 1e-3
    max_similarity: float = 8.0
    lifelong_clip: float = 5.0
    use_rnd: bool = True
    rnd_hidden_dim: int = 256
    rnd_output_dim: int = 256
    rnd_layers: int = 2
    inverse_coef: float = 1.0
    rnd_coef: float = 1.0
    vectorized_scan: bool = True
    discrete_actions: bool = False
    eps: float = 1e-8
    act: str = "relu"

    def __init__(self):
        self._obs_dim = int(self.obs_dim)
        self._action_dim = int(self.action_dim)
        self._rep_dim = int(self.rep_dim)
        self._k = int(self.k)
        self._scale = float(self.scale)
        self._use_rnd = bool(self.use_rnd)
        self._discrete_actions = bool(self.discrete_actions)
        self._eps = float(self.eps)
        if min(self._obs_dim, self._action_dim, self._rep_dim, self._k) <= 0:
            raise ValueError("NGU obs_dim, action_dim, rep_dim, and k must be positive.")
        self.encoder = MLP(
            input_dim=self._obs_dim,
            output_dim=self._rep_dim,
            hidden_dim=int(self.hidden_dim),
            layers=int(self.n_layers),
            act=self.act,
            name="encoder",
        )
        self.inverse_head = MLP(
            input_dim=2 * self._rep_dim,
            output_dim=self._action_dim,
            hidden_dim=int(self.hidden_dim),
            layers=int(self.n_layers),
            act=self.act,
            name="inverse_head",
        )
        if self._use_rnd:
            self.rnd_target = MLP(
                input_dim=self._obs_dim,
                output_dim=int(self.rnd_output_dim),
                hidden_dim=int(self.rnd_hidden_dim),
                layers=int(self.rnd_layers),
                act=self.act,
                name="rnd_target",
            )
            self.rnd_predictor = MLP(
                input_dim=self._obs_dim,
                output_dim=int(self.rnd_output_dim),
                hidden_dim=int(self.rnd_hidden_dim),
                layers=int(self.rnd_layers),
                act=self.act,
                name="rnd_predictor",
            )
            self.rnd_error_stats = RunningMeanStd(eps=self._eps, name="rnd_error_stats")
        else:
            self.rnd_target = None
            self.rnd_predictor = None
            self.rnd_error_stats = None

    def embed(self, obs):
        return self.encoder(obs)

    def embed_pair(self, obs, next_obs):
        if obs.shape == next_obs.shape:
            pair = jnp.concatenate([f32(obs), f32(next_obs)], axis=0)
            z = self.embed(pair)
            return jnp.split(z, 2, axis=0)
        return self.embed(obs), self.embed(next_obs)

    def predict_action(self, obs, next_obs):
        return self.inverse_head(jnp.concatenate([f32(obs), f32(next_obs)], axis=-1))

    def _rnd_error(self, obs):
        if not self._use_rnd:
            return jnp.zeros((*obs.shape[:-1], 1), dtype=f32(obs).dtype)
        target = self.rnd_target(obs, stop_params=True)
        pred = self.rnd_predictor(obs)
        return jnp.square(jax.lax.stop_gradient(target) - pred).mean(axis=-1, keepdims=True)

    def _inverse_loss(self, z, z_next, action, mask=None):
        pred = self.predict_action(z, z_next)
        if self._discrete_actions:
            if action.shape[-1] == self._action_dim:
                target = jnp.argmax(action, axis=-1)
            else:
                target = jnp.squeeze(action, axis=-1).astype(jnp.int32)
            error = softmax_cross_entropy(pred, target)
            acc = masked_mean((jnp.argmax(pred, axis=-1) == target).astype(jnp.float32), mask)
            return masked_mean(error, mask), {"ngu_inverse_acc": acc}
        target = f32(action)
        error = jnp.square(pred - target).mean(axis=-1)
        return masked_mean(error, mask), {"ngu_inverse_error": masked_mean(error, mask)}

    def _align_is_first(self, is_first, z):
        if is_first is None:
            return jnp.zeros(z.shape[:2], bool)
        is_first = jnp.asarray(is_first).astype(bool)
        if is_first.ndim > 2 and is_first.shape[-1] == 1:
            is_first = is_first[..., 0]
        if is_first.shape == z.shape[:2]:
            return is_first
        if is_first.ndim == 2 and is_first.shape == (z.shape[1], z.shape[0]):
            return is_first.T
        raise ValueError(f"is_first must match {z.shape[:2]}, got {is_first.shape}.")

    def _episodic_reward_scan(self, z, is_first=None, alpha=None):
        if z.ndim != 3:
            raise ValueError(f"Expected NGU embeddings [T,B,D], got {z.shape}.")
        t_len, batch, _ = z.shape
        dtype = z.dtype
        is_first = self._align_is_first(is_first, z)
        z_bt = jnp.swapaxes(z, 0, 1)
        memory_bt = jax.lax.stop_gradient(z_bt)
        is_first_bt = jnp.swapaxes(is_first, 0, 1)
        pairwise_sq = jnp.square(z_bt[:, :, None, :] - memory_bt[:, None, :, :]).sum(axis=-1)
        episode_id = jnp.cumsum(is_first_bt.astype(jnp.int32), axis=1)
        step = jnp.arange(t_len)
        prev_mask = step[None, :] < step[:, None]
        valid_prev = (episode_id[:, :, None] == episode_id[:, None, :]) & prev_mask[None]
        prev_count = valid_prev.sum(axis=-1)

        k_max = max(1, min(self._k, max(1, t_len - 1)))
        masked_sq = jnp.where(valid_prev, pairwise_sq, jnp.inf)
        knn_sq_all = jnp.sort(masked_sq, axis=-1)[..., :k_max]
        kth_index = jnp.clip(prev_count, 1, k_max).astype(jnp.int32) - 1
        kth_sq = jnp.take_along_axis(knn_sq_all, kth_index[..., None], axis=-1)[..., 0]

        knn_index = jnp.arange(k_max)
        pseudo_count = jnp.asarray(self.pseudo_count_c, dtype)
        first_reward = jax.lax.rsqrt(pseudo_count)
        eps = jnp.asarray(self._eps, dtype)
        cluster_distance = jnp.asarray(self.cluster_distance, dtype)
        kernel_epsilon = jnp.asarray(self.kernel_epsilon, dtype)
        max_similarity = jnp.asarray(self.max_similarity, dtype)

        def step_fn(carry, inputs):
            dm2_mean, dm2_count = carry
            knn_t, kth_t, prev_count_t, reset_t = inputs
            dm2_mean = jnp.where(reset_t, jnp.zeros_like(dm2_mean), dm2_mean)
            dm2_count = jnp.where(reset_t, jnp.zeros_like(dm2_count), dm2_count)
            has_memory = prev_count_t > 0
            updated_dm2 = jnp.where(
                dm2_count == 0,
                jax.lax.stop_gradient(kth_t),
                (dm2_mean * dm2_count.astype(dtype) + jax.lax.stop_gradient(kth_t))
                / (dm2_count + 1).astype(dtype),
            )
            valid_knn = knn_index[None, :] < jnp.clip(prev_count_t[:, None], 0, k_max)
            normalized = knn_t / jnp.maximum(updated_dm2[:, None], eps)
            clustered = jnp.maximum(normalized - cluster_distance, 0.0)
            kernel = jnp.where(
                valid_knn,
                kernel_epsilon / (clustered + kernel_epsilon),
                jnp.zeros_like(clustered),
            )
            similarity = jnp.sqrt(jnp.maximum(kernel.sum(axis=-1), 0.0)) + pseudo_count
            reward_values = jnp.where(
                similarity > max_similarity,
                jnp.zeros_like(similarity),
                1.0 / jnp.maximum(similarity, eps),
            )
            reward_t = jnp.where(has_memory, reward_values, jnp.full_like(reward_values, first_reward))
            dm2_t = jnp.where(has_memory, updated_dm2, dm2_mean)
            dm2_mean = jnp.where(has_memory, updated_dm2, dm2_mean)
            dm2_count = jnp.where(has_memory, dm2_count + 1, dm2_count)
            return (dm2_mean, dm2_count), (reward_t, dm2_t)

        init = (
            jnp.zeros((batch,), dtype),
            jnp.zeros((batch,), jnp.int32),
        )
        inputs = (
            jnp.swapaxes(knn_sq_all, 0, 1),
            jnp.swapaxes(kth_sq, 0, 1),
            jnp.swapaxes(prev_count, 0, 1),
            jnp.swapaxes(is_first_bt, 0, 1),
        )
        _, (episodic_reward, dm2_values) = jax.lax.scan(step_fn, init, inputs)
        episodic_reward = episodic_reward[..., None]
        dm2_values = dm2_values[..., None]
        if alpha is None:
            alpha = jnp.ones_like(episodic_reward)
        reward = episodic_reward * alpha
        metrics = {
            "ngu_reward_mean": reward.mean(),
            "ngu_episodic_reward_mean": episodic_reward.mean(),
            "ngu_dm2": dm2_values.mean(),
        }
        return reward, metrics

    def _reward_with_metrics(self, obs, is_first=None, update_reward_norm=True):
        squeezed = obs.ndim == 2
        obs_seq = obs[:, None, :] if squeezed else obs
        z = self.embed(obs_seq)
        alpha = None
        alpha_metrics = {
            "ngu_alpha_mean": jnp.ones((), jnp.float32),
            "ngu_rnd_error_mean": jnp.zeros((), jnp.float32),
            "ngu_rnd_error_std": jnp.zeros((), jnp.float32),
        }
        if self._use_rnd:
            rnd_error = self._rnd_error(obs_seq)
            normalized = self.rnd_error_stats.normalize(
                rnd_error, update=update_reward_norm
            )
            alpha = jnp.clip(1.0 + normalized, 1.0, float(self.lifelong_clip))
            alpha_metrics = {
                "ngu_alpha_mean": alpha.mean(),
                "ngu_rnd_error_mean": rnd_error.mean(),
                "ngu_rnd_error_std": jnp.sqrt(jnp.maximum(self.rnd_error_stats.var.read(), self._eps)),
            }
        reward, metrics = self._episodic_reward_scan(z, is_first=is_first, alpha=alpha)
        metrics = {**metrics, **alpha_metrics}
        return (reward[:, 0] if squeezed else reward), metrics

    def compute_reward(
        self,
        obs,
        action=None,
        next_obs=None,
        is_first=None,
        update_reward_norm=True,
        **kwargs,
    ):
        del action, next_obs, kwargs
        reward, _ = self._reward_with_metrics(obs, is_first, update_reward_norm)
        return reward * self._scale

    def loss(self, obs, action, next_obs, mask=None):
        z, z_next = self.embed_pair(f32(obs), f32(next_obs))
        inverse_loss, inverse_metrics = self._inverse_loss(z, z_next, action, mask)
        rnd_loss = jnp.zeros((), jnp.float32)
        rnd_metrics = {}
        if self._use_rnd:
            rnd_error = self._rnd_error(obs)
            rnd_loss = masked_mean(rnd_error[..., 0], mask)
            rnd_metrics = {
                "ngu_rnd_loss": rnd_loss,
                "ngu_rnd_error_mean": masked_mean(rnd_error[..., 0], mask),
                "ngu_rnd_error_std": jnp.sqrt(jnp.maximum(self.rnd_error_stats.var.read(), self._eps)),
            }
        total = float(self.inverse_coef) * inverse_loss + float(self.rnd_coef) * rnd_loss
        metrics = {
            "ngu_loss": total,
            "ngu_inverse_loss": inverse_loss,
            "ngu_reward_mean": jnp.zeros((), jnp.float32),
            "ngu_episodic_reward_mean": jnp.zeros((), jnp.float32),
            "ngu_dm2": jnp.zeros((), jnp.float32),
            "ngu_alpha_mean": jnp.ones((), jnp.float32),
            **inverse_metrics,
            **rnd_metrics,
        }
        return total, metrics

    def update(self, obs, action, next_obs, optimizer=None, return_loss=False, **kwargs):
        del kwargs
        if optimizer is not None:
            raise NotImplementedError("JAX NGU is updated by the enclosing Ninjax optimizer.")
        loss, metrics = self.loss(obs, action, next_obs)
        return (loss, metrics) if return_loss else metrics

    def bridge_consistency_targets(self, obs, action=None, next_obs=None, **kwargs):
        del action, next_obs, kwargs
        return self.embed(obs)

    def reset(self):
        if self.rnd_error_stats is not None:
            self.rnd_error_stats.reset()

    def get_modules(self):
        modules = {
            "ngu_encoder": self.encoder,
            "ngu_inverse_head": self.inverse_head,
        }
        if self.rnd_target is not None:
            modules["ngu_rnd_target"] = self.rnd_target
        if self.rnd_predictor is not None:
            modules["ngu_rnd_predictor"] = self.rnd_predictor
        if self.rnd_error_stats is not None:
            modules["ngu_rnd_error_stats"] = self.rnd_error_stats
        return modules
