"""JAX Proto-RL entropy module."""

import jax
import jax.numpy as jnp
import ninjax as nj

from .._runtime import f32
from .common import (
    MLP,
    RunningMeanNormalizer,
    masked_kth_l2_distance,
    validate_at_least,
    validate_positive,
)


class _PrototypeBank(nj.Module):

    num_prototypes: int = 0
    rep_dim: int = 0

    def __init__(self):
        self._num_prototypes = int(self.num_prototypes)
        self._rep_dim = int(self.rep_dim)

    def __call__(self):
        def init(shape):
            return 0.02 * jax.random.normal(nj.seed(), shape, jnp.float32)

        return self.value(
            "prototypes",
            init,
            (self._num_prototypes, self._rep_dim),
        )


class ProtoRL(nj.Module):

    obs_dim: int = 0
    rep_dim: int = 128
    hidden_dim: int = 256
    pred_dim: int = 512
    n_layers: int = 2
    scale: float = 1.0
    num_prototypes: int = 512
    candidates_per_prototype: int = 4
    temperature: float = 0.1
    sinkhorn_iters: int = 3
    target_momentum: float = 0.95
    k: int = 3
    reward_normalize: bool = False
    eps: float = 1e-8
    act: str = "relu"

    def __init__(self):
        self._obs_dim = int(self.obs_dim)
        self._rep_dim = int(self.rep_dim)
        self._hidden_dim = int(self.hidden_dim)
        self._pred_dim = int(self.pred_dim)
        self._layers = int(self.n_layers)
        self._scale = float(self.scale)
        self._num_prototypes = int(self.num_prototypes)
        self._candidates_per_prototype = int(self.candidates_per_prototype)
        self._temperature = float(self.temperature)
        self._sinkhorn_iters = int(self.sinkhorn_iters)
        self._target_momentum = float(self.target_momentum)
        self._k = int(self.k)
        self._reward_normalize = bool(self.reward_normalize)
        self._eps = float(self.eps)

        validate_positive("Proto-RL obs_dim", self._obs_dim)
        validate_positive("Proto-RL rep_dim", self._rep_dim)
        validate_positive("Proto-RL hidden_dim", self._hidden_dim)
        validate_positive("Proto-RL pred_dim", self._pred_dim)
        validate_at_least("Proto-RL n_layers", self._layers, 1)
        validate_at_least("Proto-RL num_prototypes", self._num_prototypes, 2)
        validate_positive(
            "Proto-RL candidates_per_prototype",
            self._candidates_per_prototype,
        )
        validate_positive("Proto-RL k", self._k)
        if self._temperature <= 0.0:
            raise ValueError("Proto-RL temperature must be positive.")
        if self._sinkhorn_iters < 1:
            raise ValueError("Proto-RL sinkhorn_iters must be >= 1.")
        if not 0.0 <= self._target_momentum <= 1.0:
            raise ValueError("Proto-RL target_momentum must be in [0, 1].")

        self.online_encoder = MLP(
            input_dim=self._obs_dim,
            output_dim=self._hidden_dim,
            hidden_dim=self._hidden_dim,
            layers=self._layers,
            act=self.act,
            name="online_encoder",
        )
        self.online_projector = MLP(
            input_dim=self._hidden_dim,
            output_dim=self._rep_dim,
            hidden_dim=self._hidden_dim,
            layers=1,
            act=self.act,
            name="online_projector",
        )
        self.predictor = MLP(
            input_dim=self._rep_dim,
            output_dim=self._rep_dim,
            hidden_dim=self._pred_dim,
            layers=1,
            act=self.act,
            name="predictor",
        )
        self.target_encoder = MLP(
            input_dim=self._obs_dim,
            output_dim=self._hidden_dim,
            hidden_dim=self._hidden_dim,
            layers=self._layers,
            act=self.act,
            name="target_encoder",
        )
        self.target_projector = MLP(
            input_dim=self._hidden_dim,
            output_dim=self._rep_dim,
            hidden_dim=self._hidden_dim,
            layers=1,
            act=self.act,
            name="target_projector",
        )
        self.prototype_bank = _PrototypeBank(
            num_prototypes=self._num_prototypes,
            rep_dim=self._rep_dim,
            name="prototype_bank",
        )
        self.reward_mean = (
            RunningMeanNormalizer(eps=self._eps, name="reward_mean")
            if self._reward_normalize
            else None
        )
        self._queue = nj.Variable(
            jnp.zeros,
            (
                self._num_prototypes,
                self._candidates_per_prototype,
                self._rep_dim,
            ),
            jnp.float32,
            name="queue",
        )
        self._queue_ptr = nj.Variable(
            jnp.zeros,
            (self._num_prototypes,),
            jnp.int32,
            name="queue_ptr",
        )
        self._queue_count = nj.Variable(
            jnp.zeros,
            (self._num_prototypes,),
            jnp.int32,
            name="queue_count",
        )

    def _prototypes(self):
        return self.prototype_bank()

    def _normalized_prototypes(self):
        return self._normalize(self._prototypes())

    def _normalize(self, z):
        # Clamp the squared norm to keep zero-vector gradients finite.
        norm_sq = jnp.square(z).sum(axis=-1, keepdims=True)
        inv_norm = jax.lax.rsqrt(
            jnp.maximum(norm_sq, jnp.square(jnp.asarray(self._eps, z.dtype)))
        )
        return z * inv_norm

    def _ensure_target_copy(self, source, target):
        if not source.values:
            return
        if target.values:
            return
        prefix = target.path + "/"
        nj.context().update({prefix + key: value for key, value in source.values.items()})

    def _ensure_targets(self):
        self._ensure_target_copy(self.online_encoder, self.target_encoder)
        self._ensure_target_copy(self.online_projector, self.target_projector)

    def encode_online(self, obs):
        y = self.online_encoder(f32(obs))
        z = self.online_projector(y)
        return self._normalize(z)

    def encode_target(self, obs):
        self._ensure_targets()
        y = self.target_encoder(f32(obs), stop_params=True)
        z = self.target_projector(y, stop_params=True)
        return jax.lax.stop_gradient(self._normalize(z))

    def _predict_online(self, obs):
        z = self.encode_online(obs)
        pred = self.predictor(z)
        return self._normalize(pred)

    def _prototype_logits(self, z, prototypes=None):
        prototypes = self._normalized_prototypes() if prototypes is None else prototypes
        return (z @ prototypes.T) / self._temperature

    def _sinkhorn(self, logits):
        logits = logits - logits.max(axis=-1, keepdims=True)
        q = jnp.exp(logits).T
        q = q / jnp.maximum(q.sum(), self._eps)
        k = q.shape[0]
        b = q.shape[1]
        for _ in range(self._sinkhorn_iters):
            q = q / jnp.maximum(q.sum(axis=1, keepdims=True), self._eps)
            q = q / float(k)
            q = q / jnp.maximum(q.sum(axis=0, keepdims=True), self._eps)
            q = q / float(b)
        return (q * float(b)).T

    def _enqueue_candidates(self, z):
        if z.size == 0:
            return
        z = jax.lax.stop_gradient(f32(z).reshape((-1, self._rep_dim)))
        logits = self._prototype_logits(z)
        sampled = jax.random.categorical(nj.seed(), logits.T, axis=-1)
        candidates = z[sampled].astype(jnp.float32)
        proto_ids = jnp.arange(self._num_prototypes, dtype=jnp.int32)
        ptr = self._queue_ptr.read()
        queue = self._queue.read().at[proto_ids, ptr].set(candidates)
        self._queue.write(queue)
        self._queue_ptr.write((ptr + 1) % self._candidates_per_prototype)
        self._queue_count.write(
            jnp.minimum(
                self._queue_count.read() + 1,
                jnp.asarray(self._candidates_per_prototype, jnp.int32),
            )
        )

    def queue_count(self):
        return self._queue_count.read().sum()

    def _flatten_queue(self):
        queue = self._queue.read().reshape((-1, self._rep_dim))
        slots = jnp.arange(self._candidates_per_prototype, dtype=jnp.int32)
        valid = slots[None] < self._queue_count.read()[:, None]
        return queue, valid.reshape((-1,))

    def _reward_from_latents(self, z, update_stats=True):
        query = z.reshape((-1, self._rep_dim))
        candidates, mask = self._flatten_queue()
        reward = masked_kth_l2_distance(query, candidates, mask, self._k)
        reward = reward.reshape((*z.shape[:-1], 1))
        if self.reward_mean is not None:
            reward = self.reward_mean.normalize(reward, update=update_stats)
        return reward

    def compute_reward(
        self,
        obs=None,
        action=None,
        next_obs=None,
        update_reward_norm=True,
        **kwargs,
    ):
        del obs, action, kwargs
        if next_obs is None:
            raise ValueError("next_obs is required to compute Proto-RL reward.")
        z_next = self.encode_online(next_obs)
        return self._scale * self._reward_from_latents(
            z_next,
            update_stats=update_reward_norm,
        )

    def loss(self, obs, next_obs, mask=None, update_queue=False):
        del mask
        obs = f32(obs).reshape((-1, self._obs_dim))
        next_obs = f32(next_obs).reshape((-1, self._obs_dim))
        if obs.size == 0 or next_obs.size == 0:
            zero = self._prototypes().sum() * 0.0
            metrics = {
                "proto_rl_loss": zero,
                "proto_rl_assignment_entropy": zero,
                "proto_rl_queue_count": self.queue_count().astype(jnp.float32),
                "proto_rl_proto_norm": jnp.linalg.norm(
                    self._normalized_prototypes(),
                    axis=-1,
                ).mean(),
            }
            return zero, metrics

        pred = self._predict_online(obs)
        prototypes = self._normalized_prototypes()
        target = self.encode_target(next_obs)
        assignments = jax.lax.stop_gradient(
            self._sinkhorn(self._prototype_logits(target, prototypes=prototypes))
        )
        logits = self._prototype_logits(pred, prototypes=prototypes)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        loss = -(assignments * log_probs).sum(axis=-1).mean()
        entropy = -(
            assignments * jnp.log(jnp.maximum(assignments, self._eps))
        ).sum(axis=-1)
        metrics = {
            "proto_rl_loss": loss,
            "proto_rl_assignment_entropy": entropy.mean(),
            "proto_rl_queue_count": self.queue_count().astype(jnp.float32),
            "proto_rl_proto_norm": jnp.linalg.norm(prototypes, axis=-1).mean(),
        }
        if update_queue:
            self._enqueue_candidates(self.encode_online(next_obs))
            metrics["proto_rl_queue_count"] = self.queue_count().astype(jnp.float32)
        return loss, metrics

    def post_update(self):
        self._ensure_targets()
        momentum = self._target_momentum

        def update_pair(source, target):
            if not source.values:
                return
            values = jax.tree.map(
                lambda src, dst: momentum * dst + (1.0 - momentum) * src,
                source.values,
                target.values,
            )
            [target.write(key, value) for key, value in values.items()]

        update_pair(self.online_encoder, self.target_encoder)
        update_pair(self.online_projector, self.target_projector)

    def update(
        self,
        obs=None,
        action=None,
        next_obs=None,
        optimizer=None,
        return_loss=False,
        **kwargs,
    ):
        del action
        if optimizer is not None:
            raise NotImplementedError("JAX Proto-RL is updated by the enclosing Ninjax optimizer.")
        if next_obs is None:
            raise ValueError("next_obs is required to update Proto-RL.")
        loss_obs = next_obs if obs is None else obs
        loss, metrics = self.loss(loss_obs, next_obs, update_queue=bool(kwargs.pop("update_queue", True)))
        return (loss, metrics) if return_loss else metrics

    def reset(self):
        self._queue.write(jnp.zeros_like(self._queue.read()))
        self._queue_ptr.write(jnp.zeros_like(self._queue_ptr.read()))
        self._queue_count.write(jnp.zeros_like(self._queue_count.read()))
        if self.reward_mean is not None:
            self.reward_mean.reset()

    def get_modules(self):
        return {
            "proto_rl_online_encoder": self.online_encoder,
            "proto_rl_online_projector": self.online_projector,
            "proto_rl_predictor": self.predictor,
            "proto_rl_prototypes": self.prototype_bank,
            "proto_rl_target_encoder": self.target_encoder,
            "proto_rl_target_projector": self.target_projector,
        }
