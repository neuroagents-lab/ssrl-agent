"""Unified JAX Dreamer agent.

``dreamer.base`` and ``dreamer.rssm`` contain the DreamerV3
training flow and world model. This module adds representation objectives
selected by ``config.rep_loss``.
"""

from copy import deepcopy

import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import ninjax as nj
import numpy as np
import optax

from . import rssm
from .base import (
    Agent as BaseDreamerAgent,
    concat,
    f32,
    i32,
    imag_loss,
    isimage,
    prefix,
    sample,
    scatter_sequence,
    sg,
    stratified_sequence_weights,
)


REP_LOSS_ALIASES = {
    "dreamer": "dreamer",
    "dreamerv3": "dreamer",
    "v3": "dreamer",
    "r2dreamer": "r2dreamer",
    "r2_dreamer": "r2dreamer",
    "ne_dreamer": "ne_dreamer",
    "nedreamer": "ne_dreamer",
    "dreamerpro": "dreamerpro",
    "dreamer_pro": "dreamerpro",
}


def normalize_rep_loss(value) -> str:
    value = str(value or "dreamer").lower().replace("-", "_")
    if value not in REP_LOSS_ALIASES:
        raise ValueError(
            "rep_loss must be one of dreamer, r2dreamer, ne_dreamer, or "
            f"dreamerpro; got {value!r}."
        )
    return REP_LOSS_ALIASES[value]


def _get(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _ensure_base_rec_scale(config, rep_loss):
    if rep_loss == "dreamer" or "rec" in _get(config, "loss_scales", {}):
        return config
    data = deepcopy(dict(config))
    data.setdefault("loss_scales", {})
    data["loss_scales"] = dict(data["loss_scales"])
    data["loss_scales"]["rec"] = 0.0
    return type(config)(data)


def _pos_init(shape, dtype=jnp.float32):
    return jax.random.normal(nj.seed(), shape, dtype) * 0.02


def _normalize(x, eps=1e-8):
    x = f32(x)
    norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True) + eps)
    return x / norm


def cosine_prediction_loss(pred, target, mask=None):
    pred = f32(pred)
    target = f32(target)
    loss = -(_normalize(pred) * _normalize(target)).sum(-1)
    if mask is None:
        return loss.mean()
    mask = jnp.asarray(mask, bool)
    assert mask.shape == loss.shape, (mask.shape, loss.shape)
    count = mask.astype(f32).sum()
    total = jnp.where(mask, loss, 0.0).sum()
    return jnp.where(count > 0, total / jnp.maximum(count, 1.0), 0.0)


def barlow_prediction_loss(pred, target, lambd=5e-4, eps=1e-8, mask=None):
    pred = f32(pred)
    target = f32(target)
    assert pred.shape[-1] == target.shape[-1], (pred.shape, target.shape)
    lead_shape = pred.shape[:-1]
    pred = pred.reshape((-1, pred.shape[-1]))
    target = target.reshape((-1, target.shape[-1]))
    assert pred.shape[0] == target.shape[0], (pred.shape, target.shape)
    if mask is None:
        mask = jnp.ones((pred.shape[0],), bool)
    else:
        mask = jnp.asarray(mask, bool)
        assert mask.shape == lead_shape, (mask.shape, lead_shape)
        mask = mask.reshape((-1,))
    count = mask.astype(f32).sum()
    safe_count = jnp.maximum(count, 1.0)

    def standardize(x):
        valid = mask[:, None]
        mean = jnp.where(valid, x, 0.0).sum(0) / safe_count
        centered = jnp.where(valid, x - mean, 0.0)
        var = jnp.square(centered).sum(0) / safe_count
        return centered / (jnp.sqrt(var) + eps)

    pred = standardize(pred)
    target = standardize(target)
    corr = pred.T @ target / safe_count
    diag = jnp.diag(corr)
    inv = jnp.square(diag - 1.0).sum()
    offdiag = corr - jnp.diag(diag)
    red = jnp.square(offdiag).sum()
    loss = inv + lambd * red
    return jnp.where(count > 0, loss, 0.0)


def ne_valid_transitions(is_first, is_last):
    """Returns whether each transition t -> t + 1 stays within an episode."""
    is_first = jnp.asarray(is_first, bool)
    is_last = jnp.asarray(is_last, bool)
    assert is_first.shape == is_last.shape, (is_first.shape, is_last.shape)
    assert is_first.ndim == 2, is_first.shape
    return ~(is_last[:, :-1] | is_first[:, 1:])


def _ne_horizon_mask(valid_transitions, horizon, length):
    valid_transitions = jnp.asarray(valid_transitions, bool)
    assert valid_transitions.ndim == 2, valid_transitions.shape
    assert horizon >= 1, horizon
    assert length + horizon - 1 <= valid_transitions.shape[1], (
        length, horizon, valid_transitions.shape)
    valid = jnp.ones((valid_transitions.shape[0], length), bool)
    for offset in range(horizon):
        valid &= valid_transitions[:, offset: offset + length]
    return valid


def ne_prediction_loss(
    pred_same,
    pred_nexts,
    target,
    *,
    use_same=False,
    use_next=True,
    loss_type="barlow",
    lambd=5e-4,
    horizon_discount=0.9,
    weight_same=1.0,
    weight_next=1.0,
    valid_transitions=None,
):
    target = sg(target)
    loss_type = str(loss_type).lower()

    def loss_fn(pred, tgt, mask=None):
        if loss_type == "cosine":
            return cosine_prediction_loss(pred, tgt, mask=mask)
        if loss_type == "barlow":
            return barlow_prediction_loss(pred, tgt, lambd=lambd, mask=mask)
        raise ValueError(f"Unknown NE-Dreamer loss_type: {loss_type!r}")

    zero = jnp.zeros((), f32)
    same = zero
    if use_same:
        same = loss_fn(pred_same, target)

    next_loss = zero
    if use_next:
        total = zero
        total_weight = zero
        for index, pred in enumerate(pred_nexts):
            if pred is None:
                continue
            horizon = index + 1
            if horizon >= target.shape[1]:
                continue
            tgt = target[:, horizon:]
            length = min(pred.shape[1], tgt.shape[1])
            if length <= 0:
                continue
            weight = float(horizon_discount) ** index
            mask = None
            pair_count = jnp.asarray(pred.shape[0] * length, f32)
            if valid_transitions is not None:
                mask = _ne_horizon_mask(valid_transitions, horizon, length)
                pair_count = mask.astype(f32).sum()
            min_pairs = 2 if loss_type == "barlow" else 1
            active = (pair_count >= min_pairs).astype(f32)
            total = total + weight * active * loss_fn(
                pred[:, :length], tgt[:, :length], mask=mask)
            total_weight = total_weight + weight * active
        safe_total_weight = jnp.where(total_weight > 0, total_weight, 1.0)
        next_loss = jnp.where(total_weight > 0, total / safe_total_weight, zero)

    if use_same and use_next:
        combined = weight_same * same + weight_next * next_loss
    elif use_same:
        combined = weight_same * same
    else:
        combined = weight_next * next_loss

    metrics = {
        "ne_dreamer/same": same,
        "ne_dreamer/next": next_loss,
        "ne_dreamer/total": combined,
    }
    return combined, metrics


def _ne_episode_starts(is_first, is_last=None):
    is_first = jnp.asarray(is_first, bool)
    assert is_first.ndim == 2, is_first.shape
    starts = is_first
    if is_last is not None:
        is_last = jnp.asarray(is_last, bool)
        assert is_last.shape == is_first.shape, (is_last.shape, is_first.shape)
        starts = starts.at[:, 1:].set(starts[:, 1:] | is_last[:, :-1])
    return starts


def _ne_causal_mask(is_first, is_last=None, use_actions=True):
    """Builds a causal attention mask that cannot cross episode boundaries."""
    starts = _ne_episode_starts(is_first, is_last)
    segments = jnp.cumsum(starts.astype(jnp.int32), axis=1)
    if use_actions:
        segments = jnp.repeat(segments, 2, axis=1)
    length = segments.shape[1]
    causal = jnp.tril(jnp.ones((length, length), bool))
    same_episode = segments[:, :, None] == segments[:, None, :]
    return same_episode & causal[None]


def _ne_position_indices(is_first, is_last=None, use_actions=True):
    """Returns episode-local positional indices for feature and action tokens."""
    starts = _ne_episode_starts(is_first, is_last)
    length = starts.shape[1]
    steps = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), starts.shape)
    start_indices = jnp.where(starts, steps, 0)
    latest_start = jax.lax.associative_scan(jnp.maximum, start_indices, axis=1)
    local_steps = steps - latest_start
    if use_actions:
        local_steps = jnp.stack([2 * local_steps, 2 * local_steps + 1], axis=2)
        local_steps = local_steps.reshape((starts.shape[0], 2 * length))
    return local_steps


class FeatureProjector(nj.Module):

    output_dim: int
    bias: bool = False
    winit: str = "trunc_normal_in"

    def __call__(self, x):
        x = nn.cast(x)
        return self.sub(
            "linear",
            nn.Linear,
            int(self.output_dim),
            bias=bool(self.bias),
            winit=self.winit,
        )(x)


class NEDreamerTransformer(nj.Module):
    """Causal transformer that predicts encoder embeddings from RSSM features."""

    def __init__(
        self,
        act_space,
        embed_dim,
        *,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        use_actions=True,
        use_projector=True,
        use_same=False,
        use_next=True,
        predict_horizon=1,
        max_seq_len=128,
        act="gelu",
        norm="layer",
        winit="trunc_normal_in",
    ):
        self.act_space = act_space
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.use_actions = bool(use_actions)
        self.use_projector = bool(use_projector)
        self.use_same = bool(use_same)
        self.use_next = bool(use_next)
        self.predict_horizon = int(predict_horizon)
        self.max_seq_len = int(max_seq_len)
        self.act = str(act).lower()
        self.norm = str(norm).lower()
        self.kw = dict(winit=winit)
        if not (self.use_same or self.use_next):
            raise ValueError("NE-Dreamer requires use_same or use_next.")
        if self.predict_horizon < 1:
            raise ValueError("NE-Dreamer predict_horizon must be >= 1.")

    def __call__(
        self,
        feat,
        actions=None,
        training=True,
        is_first=None,
        is_last=None,
    ):
        feat = nn.cast(feat)
        B, T = feat.shape[:2]
        if T > self.max_seq_len:
            raise ValueError(
                f"NE-Dreamer sequence length {T} exceeds max_seq_len={self.max_seq_len}."
            )
        if self.use_projector:
            feat = self.sub("projector", nn.Linear, self.embed_dim, **self.kw)(feat)
            feat = nn.act(self.act)(self.sub("projector_norm", nn.Norm, self.norm)(feat))
        tok_f = self.sub("feat_embed", nn.Linear, self.hidden_dim, **self.kw)(feat)

        if self.use_actions:
            if actions is None:
                raise ValueError("NE-Dreamer action tokens are enabled but no actions were provided.")
            action = nn.DictConcat(self.act_space, 1)(actions)
            tok_a = self.sub("action_embed", nn.Linear, self.hidden_dim, **self.kw)(action)
            tok_a = self.sub("action_norm", nn.Norm, self.norm)(tok_a)
            tok_a = nn.act(self.act)(tok_a)
            tokens = jnp.stack([tok_f, tok_a], 2).reshape((B, 2 * T, self.hidden_dim))
        else:
            tokens = tok_f

        pos_len = 2 * self.max_seq_len if self.use_actions else self.max_seq_len
        pos = self.value("pos_embed", _pos_init, (pos_len, self.hidden_dim))
        S = tokens.shape[1]
        if is_first is None:
            if is_last is not None:
                raise ValueError("NE-Dreamer received is_last without is_first.")
            tokens = tokens + pos[:S].astype(tokens.dtype)
            mask = jnp.tril(jnp.ones((S, S), bool))
            mask = jnp.broadcast_to(mask, (B, S, S))
        else:
            if jnp.shape(is_first) != (B, T):
                raise ValueError(
                    f"NE-Dreamer is_first shape {jnp.shape(is_first)} != {(B, T)}.")
            if is_last is not None and jnp.shape(is_last) != (B, T):
                raise ValueError(
                    f"NE-Dreamer is_last shape {jnp.shape(is_last)} != {(B, T)}.")
            position = _ne_position_indices(is_first, is_last, self.use_actions)
            tokens = tokens + pos[position].astype(tokens.dtype)
            mask = _ne_causal_mask(is_first, is_last, self.use_actions)
        h = self.sub(
            "transformer",
            nn.Transformer,
            units=self.hidden_dim,
            layers=self.num_layers,
            heads=self.num_heads,
            ffup=4,
            act=self.act,
            norm=self.norm,
            rope=False,
            dropout=self.dropout,
            **self.kw,
        )(tokens, mask=mask, training=training)

        if self.use_actions:
            h_same = h[:, 0::2]
            h_next = h[:, 1 : 2 * T - 1 : 2]
        else:
            h_same = h
            h_next = h[:, :-1]

        same = self._head("same", h_same) if self.use_same else None
        nexts = []
        if self.use_next:
            for index in range(self.predict_horizon):
                if index < h_next.shape[1]:
                    nexts.append(self._head(f"next{index}", h_next[:, : h_next.shape[1] - index]))
                else:
                    nexts.append(None)
        return same, nexts

    def _head(self, name, x):
        with nj.scope(name):
            x = self.sub("norm", nn.Norm, self.norm)(x)
            x = self.sub("hidden", nn.Linear, self.hidden_dim, **self.kw)(x)
            x = nn.act(self.act)(x)
            x = self.sub("out", nn.Linear, self.embed_dim, **self.kw)(x)
            return x


class _PrototypeBank(nj.Module):

    num_prototypes: int = 0
    proto_dim: int = 0

    def __call__(self):
        def init(shape):
            return jax.random.normal(nj.seed(), shape, jnp.float32)

        return self.value(
            "prototypes",
            init,
            (int(self.num_prototypes), int(self.proto_dim)),
        )

    def normalized(self, eps=1e-8, freeze=False):
        prototypes = self()
        norm = jnp.sqrt(jnp.sum(jnp.square(f32(prototypes)), axis=-1, keepdims=True) + eps)
        prototypes = prototypes / norm
        return jnp.where(freeze, sg(prototypes), prototypes)

    def normalize_in_place(self, eps=1e-8):
        prototypes = self.normalized(eps=eps, freeze=False)
        self.write("prototypes", prototypes)


class DreamerProObjective:

    def __init__(
        self,
        config,
        embed_dim,
        feat_dim,
        enc_space,
        enc_config,
        enc_cls=rssm.Encoder,
    ):
        self.warm_up = int(_get(config, "warm_up", 1))
        self.num_prototypes = int(_get(config, "num_prototypes", 2500))
        self.proto_dim = int(_get(config, "proto_dim", 32))
        self.temperature = float(_get(config, "temperature", 0.1))
        self.sinkhorn_eps = float(_get(config, "sinkhorn_eps", 0.05))
        self.sinkhorn_iters = int(_get(config, "sinkhorn_iters", 3))
        self.ema_update_every = int(_get(config, "ema_update_every", 1))
        self.ema_update_fraction = float(_get(config, "ema_update_fraction", 0.05))
        self.freeze_prototypes_iters = int(_get(config, "freeze_prototypes_iters", 10000))
        aug = _get(config, "aug", {})
        self.aug_max_delta = float(_get(aug, "max_delta", 3.0))
        self.aug_same_across_time = bool(_get(aug, "same_across_time", True))
        self.aug_bilinear = bool(_get(aug, "bilinear", True))
        self.eps = 1e-8

        self.prototype_bank = _PrototypeBank(
            num_prototypes=self.num_prototypes,
            proto_dim=self.proto_dim,
            name="dreamerpro_prototypes",
        )
        self.obs_proj = FeatureProjector(output_dim=self.proto_dim, name="dreamerpro_obs_proj")
        self.feat_proj = FeatureProjector(output_dim=self.proto_dim, name="dreamerpro_feat_proj")
        self.ema_obs_proj = FeatureProjector(
            output_dim=self.proto_dim, name="dreamerpro_ema_obs_proj")
        self.slow_obs_proj = embodied.jax.SlowModel(
            self.ema_obs_proj,
            source=self.obs_proj,
            rate=self.ema_update_fraction,
            every=self.ema_update_every,
        )
        self.ema_enc = embodied.jax.SlowModel(
            enc_cls(enc_space, **enc_config, name="dreamerpro_ema_enc"),
            source=None,
            rate=self.ema_update_fraction,
            every=self.ema_update_every,
        )
        self.embed_dim = int(embed_dim)
        self.feat_dim = int(feat_dim)

    @property
    def modules(self):
        return [self.prototype_bank, self.obs_proj, self.feat_proj]

    def bind_encoder(self, enc):
        self.ema_enc.source = enc

    def update_targets(self):
        self.prototype_bank.normalize_in_place(self.eps)
        self.ema_enc.update()
        self.slow_obs_proj.update()

    def augment(self, obs, imgkeys):
        obs = {k: jnp.concatenate([v, v], 0) for k, v in obs.items()}
        for key in imgkeys:
            obs[key] = self._random_translate(
                obs[key],
                self.aug_max_delta,
                same_across_time=self.aug_same_across_time,
            )
        return obs

    def target_embed(self, obs, reset, training):
        _, _, embed = self.ema_enc({}, obs, reset, training)
        return sg(embed)

    def target_project(self, embed):
        proj = self.slow_obs_proj(embed)
        return sg(self._normalize(proj))

    def loss(self, feat, embed, target_embed, freeze_prototypes=False):
        if embed.shape[1] <= self.warm_up:
            raise ValueError(
                "DreamerPro requires batch_length > dreamer_pro.warm_up; "
                f"got sequence length {embed.shape[1]} and warm_up={self.warm_up}."
            )
        prototypes = self.prototype_bank.normalized(self.eps, freeze_prototypes)

        obs_proj = self.obs_proj(embed)
        obs_norm = self._norm(obs_proj, keepdims=False)
        obs_proj = self._normalize(obs_proj)
        ema_proj = self.target_project(target_embed)

        obs_scores = self._scores(obs_proj, prototypes)
        obs_logits = jax.nn.log_softmax(obs_scores / self.temperature, axis=0)
        obs_logits = obs_logits[:, :, self.warm_up :]
        obs_logits_1, obs_logits_2 = self._split_views(obs_logits)

        ema_scores = self._scores(ema_proj, prototypes)
        ema_scores = ema_scores[:, :, self.warm_up :]
        ema_scores_1, ema_scores_2 = self._split_views(ema_scores)
        ema_targets_1 = sg(self.sinkhorn(ema_scores_1))
        ema_targets_2 = sg(self.sinkhorn(ema_scores_2))
        ema_targets = jnp.concatenate([ema_targets_1, ema_targets_2], axis=1)

        feat_proj = self.feat_proj(feat)
        feat_norm = self._norm(feat_proj, keepdims=False)
        feat_proj = self._normalize(feat_proj)
        feat_scores = self._scores(feat_proj, prototypes)
        feat_logits = jax.nn.log_softmax(feat_scores / self.temperature, axis=0)
        feat_logits = feat_logits[:, :, self.warm_up :]

        swav_loss = -0.5 * jnp.mean(jnp.sum(ema_targets_2 * obs_logits_1, axis=0))
        swav_loss += -0.5 * jnp.mean(jnp.sum(ema_targets_1 * obs_logits_2, axis=0))
        temp_loss = -jnp.mean(jnp.sum(ema_targets * feat_logits, axis=0))
        norm_loss = jnp.square(obs_norm - 1).mean() + jnp.square(feat_norm - 1).mean()
        losses = {"swav": swav_loss, "temp": temp_loss, "norm": norm_loss}
        metrics = {
            "dreamerpro/prototype_norm": self._norm(prototypes, keepdims=False).mean(),
            "dreamerpro/obs_norm": obs_norm.mean(),
            "dreamerpro/feat_norm": feat_norm.mean(),
        }
        return losses, metrics

    def sinkhorn(self, scores):
        shape = scores.shape
        k = shape[0]
        scores = scores.reshape((k, -1))
        scores = scores - scores.max()
        log_q = jax.nn.log_softmax((scores / self.sinkhorn_eps).reshape(-1), axis=0)
        log_q = log_q.reshape((k, -1))
        n = log_q.shape[1]
        for _ in range(self.sinkhorn_iters):
            log_q = log_q - jsp.logsumexp(log_q, axis=1, keepdims=True) - jnp.log(k)
            log_q = log_q - jsp.logsumexp(log_q, axis=0, keepdims=True) - jnp.log(n)
        return jnp.exp(log_q + jnp.log(n)).reshape(shape)

    def _scores(self, x, prototypes):
        B, T = x.shape[:2]
        x = x.reshape((B * T, -1))
        scores = x @ prototypes.T
        return scores.reshape((B, T, -1)).transpose((2, 0, 1))

    def _split_views(self, x):
        half = x.shape[1] // 2
        return x[:, :half], x[:, half:]

    def _normalize(self, x):
        return f32(x) / self._norm(x, keepdims=True)

    def _norm(self, x, keepdims):
        x = f32(x)
        return jnp.sqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=keepdims) + self.eps)

    def _random_translate(self, x, max_delta, same_across_time=False):
        pad = int(max_delta)
        if pad <= 0:
            return x
        B, T, H, W, C = x.shape
        padded = jnp.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge")
        flat = padded.reshape((B * T, H + 2 * pad, W + 2 * pad, C))
        if same_across_time:
            shifts = jax.random.randint(nj.seed(), (B, 2), 0, 2 * pad + 1, jnp.int32)
            shifts = jnp.repeat(shifts[:, None], T, axis=1).reshape((B * T, 2))
        else:
            shifts = jax.random.randint(nj.seed(), (B * T, 2), 0, 2 * pad + 1, jnp.int32)

        def crop(img, shift):
            return jax.lax.dynamic_slice(img, (shift[0], shift[1], 0), (H, W, C))

        out = jax.vmap(crop)(flat, shifts)
        return out.reshape((B, T, H, W, C))


class DreamerAgent(BaseDreamerAgent):

    banner = [
        r"---  ___                           ___              _ _       ---",
        r"--- |   \ _ _ ___ __ _ _ __  ___ _| __|_ _ _ __  _ | | |_  _ ---",
        r"--- | |) | '_/ -_) _` | '  \/ -_) | _/ _` | '  \| || | | || | ---",
        r"--- |___/|_| \___\__,_|_|_|_\___|_|_|\__,_|_|_|_|\__|_|_|\_, | ---",
        r"---                                                   |__/ ---",
    ]

    def __init__(self, obs_space, act_space, config):
        rep_loss = normalize_rep_loss(_get(config, "rep_loss", "dreamer"))
        config = _ensure_base_rec_scale(config, rep_loss)
        super().__init__(obs_space, act_space, config)
        self.rep_loss = rep_loss
        exclude = ("is_first", "is_last", "is_terminal", "reward")
        enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
        dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
        self._embed_dim = _encoder_output_dim(self.enc, enc_space)
        self._feat_dim = int(self.dyn.deter + self.dyn.stoch * self.dyn.classes)

        self.r2_projector = None
        self.ne = None
        self.dreamerpro = None
        rep_modules = []
        if self.rep_loss == "dreamer":
            rep_modules.append(self.dec)
        elif self.rep_loss == "r2dreamer":
            self.r2_projector = FeatureProjector(
                output_dim=self._embed_dim, name="r2_projector")
            rep_modules.append(self.r2_projector)
        elif self.rep_loss == "ne_dreamer":
            self.ne = self._make_ne_transformer(act_space)
            rep_modules.append(self.ne)
        elif self.rep_loss == "dreamerpro":
            dp = config.dreamer_pro
            if int(config.batch_length) <= int(_get(dp, "warm_up", 1)):
                raise ValueError(
                    "DreamerPro requires batch_length > dreamer_pro.warm_up; "
                    f"got batch_length={config.batch_length}, warm_up={_get(dp, 'warm_up', 1)}."
                )
            self.dreamerpro = DreamerProObjective(
                dp,
                self._embed_dim,
                self._feat_dim,
                enc_space,
                dict(config.enc[config.enc.typ]),
                rssm.ENCODERS[config.enc.typ],
            )
            self.dreamerpro.bind_encoder(self.enc)
            rep_modules.extend(self.dreamerpro.modules)
        else:
            raise NotImplementedError(f"Unsupported rep_loss: {self.rep_loss}")

        base_behavior = (
            [self.rew, self.pol, self.val]
            if self.train_base_task_behavior else []
        )
        self.modules = [
            self.dyn,
            *self._encoder_modules,
            self.con,
            *base_behavior,
            *rep_modules,
        ]
        # The shared optimizer uses the ``opt`` checkpoint namespace.
        self.main_opt = embodied.jax.Optimizer(
            self.modules,
            self._make_opt(**config.opt),
            adaptive=embodied.jax.opt.adaptive_critic(
                **(_get(config, "adaptive_critic", {}) or {})),
            summary_depth=1,
            log_grads=_get(config, "log_grads", True),
            name="opt",
        )
        self.opt = self.main_opt
        self.scales = self._make_scales(dec_space)
        self.algorithm2_adaptation = bool(
            _get(config, "algorithm2_adaptation", False)
        )
        self.adapt_rew_opt = None
        self.adapt_behavior_opt = None
        if self.algorithm2_adaptation:
            # Algorithm 2 treats the imported world model as fixed while it
            # first distills the downstream reward and derives a task policy in
            # imagination. Separate optimizers keep those offline moments out
            # of the ordinary online Dreamer optimizer that follows.
            self.adapt_rew_opt = embodied.jax.Optimizer(
                [self.rew],
                self._make_opt(**config.opt),
                summary_depth=1,
                log_grads=_get(config, "log_grads", True),
                name="adapt_rew_opt",
            )
            self.adapt_behavior_opt = embodied.jax.Optimizer(
                [self.pol, self.val],
                self._make_opt(**config.opt),
                summary_depth=1,
                log_grads=_get(config, "log_grads", True),
                name="adapt_behavior_opt",
            )

    def _make_ne_transformer(self, act_space):
        ne = self.config.ne_dreamer
        return NEDreamerTransformer(
            act_space,
            self._embed_dim,
            hidden_dim=int(_get(ne, "hidden_dim", 256)),
            num_layers=int(_get(ne, "num_layers", 2)),
            num_heads=int(_get(ne, "num_heads", 4)),
            dropout=float(_get(ne, "dropout", 0.0)),
            use_actions=bool(_get(ne, "use_actions", True)),
            use_projector=bool(_get(ne, "use_projector", True)),
            use_same=bool(_get(ne, "use_same", False)),
            use_next=bool(_get(ne, "use_next", True)),
            predict_horizon=int(_get(ne, "predict_horizon", 1)),
            max_seq_len=int(_get(ne, "max_seq_len", 128)),
            name="ne",
        )

    def _make_scales(self, dec_space):
        scales = dict(self.config.loss_scales.copy())
        if self.rep_loss == "dreamer":
            rec = scales.pop("rec")
            scales.update({k: rec for k in dec_space})
        else:
            scales.pop("rec", None)
            scales.pop("recon", None)
        if not self.config.repval_loss:
            scales.pop("repval", None)
        if not self.train_base_task_behavior:
            for key in ("rew", "policy", "value", "repval"):
                scales.pop(key, None)
        return scales

    def train(self, carry, data):
        carry, obs, prevact, stepid, task_labels = self._apply_replay_context(
            carry, data
        )
        actions = self._current_actions(data)
        self._reset_optimizer_state_on_create()
        metrics, (carry, entries, outs, mets) = self.main_opt(
            self.loss, carry, obs, prevact, actions, training=True, has_aux=True)
        metrics.update(mets)
        if self.train_base_task_behavior:
            self.slowval.update()
        if self.task_opt is not None:
            task_metrics, task_mets = self.task_opt(
                self.task_loss,
                outs["repfeat"],
                entries[1],
                carry[1],
                obs,
                prevact,
                task_labels,
                training=True,
                has_aux=True,
            )
            metrics.update(task_metrics)
            metrics.update(task_mets)
            for slowval in self.task_heads["slowvals"].values():
                slowval.update()
        if self.rep_loss == "dreamerpro":
            self.dreamerpro.update_targets()
        outs = {}
        if self.config.replay_context:
            updates = elements.tree.flatdict(dict(
                stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
            B, T = obs["is_first"].shape
            assert all(x.shape[:2] == (B, T) for x in updates.values()), (
                (B, T), {k: v.shape for k, v in updates.items()})
            outs["replay"] = updates
        carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
        return carry, outs, metrics

    def train_adaptation(self, carry, data):
        """Runs one paper Algorithm 2 update with the world model frozen."""
        if not self.algorithm2_adaptation:
            raise RuntimeError(
                "Algorithm 2 adaptation was not enabled for this agent."
            )
        carry, obs, prevact, _, _ = self._apply_replay_context(carry, data)
        actions = self._current_actions(data)
        optimizers = (self.adapt_rew_opt, self.adapt_behavior_opt)
        self._reset_optimizer_state_on_create(optimizers)
        initialize_only = bool(nj.creating())

        reward_metrics, (carry, dyn_entries, repfeat, reward_mets) = (
            self.adapt_rew_opt(
                self.adaptation_reward_loss,
                carry,
                obs,
                prevact,
                training=True,
                initialize_only=initialize_only,
                has_aux=True,
            )
        )
        behavior_metrics, behavior_mets = self.adapt_behavior_opt(
            self.adaptation_behavior_loss,
            repfeat,
            dyn_entries,
            carry[1],
            obs,
            training=True,
            initialize_only=initialize_only,
            has_aux=True,
        )
        if initialize_only:
            # transform.init executes the function to discover optimizer state.
            # State initialization must not advance the adaptation update counter.
            self.adapt_rew_opt.step.write(jnp.asarray(0, i32))
            self.adapt_behavior_opt.step.write(jnp.asarray(0, i32))
        else:
            self.slowval.update()

        metrics = {}
        metrics.update(prefix(reward_metrics, "offline"))
        metrics.update(prefix(behavior_metrics, "offline"))
        metrics.update(reward_mets)
        metrics.update(behavior_mets)
        carry = (*carry, {k: actions[k][:, -1] for k in self.act_space})
        return carry, {}, metrics

    def adaptation_reward_loss(
        self, carry, obs, prevact, training, initialize_only=False
    ):
        """Distills scalar replay rewards without changing the imported model."""
        enc_carry, dyn_carry, dec_carry = carry
        reset = obs["is_first"]
        enc_carry, _, tokens = self.enc(
            enc_carry, obs, reset, training=False
        )
        dyn_carry, dyn_entries, repfeat = self.dyn.observe(
            dyn_carry, tokens, prevact, reset, training=False
        )
        inp = sg(self.feat2tensor(repfeat))
        reward_loss = self.rew(inp, 2).loss(obs["reward"])
        loss = reward_loss.mean() * self.scales["rew"]
        if initialize_only:
            loss = loss * jnp.asarray(0.0, f32)
        metrics = {
            "offline_adaptation/loss/rew": reward_loss.mean(),
        }
        return loss, (
            (enc_carry, dyn_carry, dec_carry),
            dyn_entries,
            repfeat,
            metrics,
        )

    def adaptation_behavior_loss(
        self,
        repfeat,
        dyn_entries,
        dyn_carry,
        obs,
        training,
        initialize_only=False,
    ):
        """Trains the task actor/value from predicted reward in imagination."""
        B, T = obs["is_first"].shape
        K = self._imag_start_count(T)
        H = self.config.imag_length
        imag_indices = self._sequence_selection(
            B,
            T,
            K,
            _get(self.config, "imag_start_strategy", "last"),
        )
        starts, first = self._imag_start_data(
            repfeat, dyn_entries, dyn_carry, K, imag_indices
        )
        policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
        _, imgfeat, imgprevact = self.dyn.imagine(
            starts, policyfn, H, training=False
        )
        imgfeat = concat(
            [sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1
        )
        lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
        lastact = jax.tree.map(lambda x: x[:, None], lastact)
        imgact = concat([imgprevact, lastact], 1)
        inp = self.feat2tensor(imgfeat)
        losses, _, metrics = imag_loss(
            imgact,
            self.rew(inp, 2).pred(),
            self.con(inp, 2).prob(1),
            self.pol(inp, 2),
            self.val(inp, 2),
            self.slowval(inp, 2),
            self.retnorm,
            self.valnorm,
            self.advnorm,
            update=bool(training and not initialize_only),
            contdisc=self.config.contdisc,
            horizon=self.config.horizon,
            **self.config.imag_loss,
        )
        expected = {"policy", "value"}
        if set(losses) != expected:
            raise ValueError(
                "Algorithm 2 imagination returned unexpected losses: "
                f"{sorted(losses)} != {sorted(expected)}"
            )
        loss = sum(
            losses[key].mean() * self.scales[key] for key in sorted(expected)
        )
        if initialize_only:
            loss = loss * jnp.asarray(0.0, f32)
        metrics.update(
            {
                f"offline_adaptation/loss/{key}": value.mean()
                for key, value in losses.items()
            }
        )
        return loss, metrics

    def loss(self, carry, obs, prevact, actions, training):
        enc_carry, dyn_carry, dec_carry = carry
        start_enc_carry, start_dyn_carry = enc_carry, dyn_carry
        reset = obs["is_first"]
        B, T = reset.shape
        losses = {}
        metrics = {}

        if self.rep_loss == "dreamer":
            reconK = min(int(_get(self.config, "recon_last", 0)) or T, T)
            recon_indices = self._sequence_selection(
                B,
                T,
                reconK,
                _get(self.config, "recon_start_strategy", "stratified"),
            )
        else:
            reconK, recon_indices = T, None

        enc_carry, enc_entries, tokens = self.enc(enc_carry, obs, reset, training)
        dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
            dyn_carry, tokens, prevact, reset, training)
        losses.update(los)
        metrics.update(mets)

        dec_carry, dec_entries, replosses, repmetrics = self._representation_loss(
            dec_carry,
            start_enc_carry,
            start_dyn_carry,
            obs,
            prevact,
            actions,
            tokens,
            repfeat,
            training,
            reconK,
            recon_indices,
        )
        for key, value in replosses.items():
            losses[key] = value if value.shape == (B, T) else jnp.ones((B, T), f32) * value
        metrics.update(repmetrics)

        inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
        if self.train_base_task_behavior:
            losses["rew"] = self.rew(inp, 2).loss(obs["reward"])
        con = f32(~obs["is_terminal"])
        if self.config.contdisc:
            con *= 1 - 1 / self.config.horizon
        losses["con"] = self.con(self.feat2tensor(repfeat), 2).loss(con)

        shapes = {k: v.shape for k, v in losses.items()}
        assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

        K = self._imag_start_count(T)
        if self.train_base_task_behavior:
            H = self.config.imag_length
            imag_indices = self._sequence_selection(
                B, T, K, _get(self.config, "imag_start_strategy", "last")
            )
            starts, first = self._imag_start_data(
                repfeat, dyn_entries, dyn_carry, K, imag_indices
            )
            policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
            _, imgfeat, imgprevact = self.dyn.imagine(
                starts, policyfn, H, training
            )
            imgfeat = concat(
                [sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1
            )
            lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
            lastact = jax.tree.map(lambda x: x[:, None], lastact)
            imgact = concat([imgprevact, lastact], 1)
            assert all(
                x.shape[:2] == (B * K, H + 1)
                for x in jax.tree.leaves(imgfeat)
            )
            assert all(
                x.shape[:2] == (B * K, H + 1)
                for x in jax.tree.leaves(imgact)
            )
            inp = self.feat2tensor(imgfeat)
            los, _, mets = imag_loss(
                imgact,
                self.rew(inp, 2).pred(),
                self.con(inp, 2).prob(1),
                self.pol(inp, 2),
                self.val(inp, 2),
                self.slowval(inp, 2),
                self.retnorm,
                self.valnorm,
                self.advnorm,
                update=training,
                contdisc=self.config.contdisc,
                horizon=self.config.horizon,
                **self.config.imag_loss,
            )
            losses.update(
                {k: v.mean(1).reshape((B, K)) for k, v in los.items()}
            )
            metrics.update(mets)

            if self.config.repval_loss:
                last, term, rew = [
                    obs[k] for k in ("is_last", "is_terminal", "reward")
                ]
                los, _, mets = self._replay_value_loss(
                    repfeat,
                    last,
                    term,
                    rew,
                    self.val,
                    self.slowval,
                    self.valnorm,
                    K,
                    training,
                    repval_grad=self.config.repval_grad,
                )
                losses.update(los)
                metrics.update(prefix(mets, "reploss"))

        metrics["imag/start_count"] = jnp.asarray(K, f32)
        metrics["imag/start_fraction"] = jnp.asarray(K / T, f32)
        if self.rep_loss == "dreamer":
            metrics["recon/start_count"] = jnp.asarray(reconK, f32)
            metrics["recon/start_fraction"] = jnp.asarray(reconK / T, f32)

        assert set(losses.keys()) == set(self.scales.keys()), (
            sorted(losses.keys()), sorted(self.scales.keys()))
        metrics.update({f"loss/{k}": v.mean() for k, v in losses.items()})
        loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

        carry = (enc_carry, dyn_carry, dec_carry)
        entries = (enc_entries, dyn_entries, dec_entries)
        outs = {"tokens": tokens, "repfeat": repfeat, "losses": losses}
        return loss, (carry, entries, outs, metrics)

    def _representation_loss(
        self,
        dec_carry,
        start_enc_carry,
        start_dyn_carry,
        obs,
        prevact,
        actions,
        tokens,
        repfeat,
        training,
        recon_count=None,
        recon_indices=None,
    ):
        reset = obs["is_first"]
        if self.rep_loss == "dreamer":
            length = reset.shape[1]
            recon_count = int(recon_count or length)
            dec_feat = self._select_sequence(repfeat, recon_indices)
            dec_reset = self._select_sequence(reset, recon_indices)
            dec_obs = self._select_sequence(obs, recon_indices)
            dec_carry, dec_entries, recons = self.dec(
                dec_carry, dec_feat, dec_reset, training
            )
            losses = {}
            for key, recon in recons.items():
                space, value = self.obs_space[key], dec_obs[key]
                assert value.dtype == space.dtype, (key, space, value.dtype)
                target = f32(value) / 255 if isimage(space) else value
                losses[key] = recon.loss(sg(target))
            if recon_indices is not None:
                if jax.tree.leaves(dec_entries):
                    raise ValueError(
                        "Sampled reconstruction requires a stateless decoder."
                    )
                strategy = str(
                    _get(self.config, "recon_start_strategy", "stratified")
                ).lower()
                weights = (
                    stratified_sequence_weights(length, recon_count)
                    if strategy == "stratified"
                    else None
                )
                losses = scatter_sequence(
                    losses, recon_indices, length, weights=weights
                )
            return dec_carry, dec_entries, losses, {}

        if self.rep_loss == "r2dreamer":
            pred = self.r2_projector(self.feat2tensor(repfeat))
            loss = barlow_prediction_loss(pred, sg(tokens), lambd=float(self.config.r2dreamer.lambd))
            metrics = {"r2dreamer/barlow": loss}
            return {}, {}, {"barlow": loss}, metrics

        if self.rep_loss == "ne_dreamer":
            valid_transitions = ne_valid_transitions(obs["is_first"], obs["is_last"])
            same, nexts = self.ne(
                self.feat2tensor(repfeat),
                actions,
                training=training,
                is_first=obs["is_first"],
                is_last=obs["is_last"],
            )
            loss, metrics = ne_prediction_loss(
                same,
                nexts,
                tokens,
                use_same=bool(_get(self.config.ne_dreamer, "use_same", False)),
                use_next=bool(_get(self.config.ne_dreamer, "use_next", True)),
                loss_type=str(_get(self.config.ne_dreamer, "loss_type", "barlow")),
                lambd=float(_get(self.config.ne_dreamer, "lambd", 5e-4)),
                horizon_discount=float(_get(self.config.ne_dreamer, "horizon_discount", 0.9)),
                weight_same=float(_get(self.config.ne_dreamer, "weight_same", 1.0)),
                weight_next=float(_get(self.config.ne_dreamer, "weight_next", 1.0)),
                valid_transitions=valid_transitions,
            )
            return {}, {}, {"ne_dreamer": loss}, metrics

        if self.rep_loss == "dreamerpro":
            obs_aug = self.dreamerpro.augment(obs, self.enc.imgkeys)
            reset_aug = obs_aug["is_first"]
            prevact_aug = jax.tree.map(lambda x: jnp.concatenate([x, x], 0), prevact)
            enc_carry_aug = jax.tree.map(lambda x: jnp.concatenate([x, x], 0), start_enc_carry)
            dyn_carry_aug = jax.tree.map(lambda x: jnp.concatenate([x, x], 0), start_dyn_carry)
            _, _, embed_aug = self.enc(enc_carry_aug, obs_aug, reset_aug, training)
            _, _, aug_feat = self.dyn.observe(
                dyn_carry_aug, embed_aug, prevact_aug, reset_aug, training)
            target_embed = self.dreamerpro.target_embed(obs_aug, reset_aug, training)
            freeze = self.opt.step.read() < self.dreamerpro.freeze_prototypes_iters
            losses, metrics = self.dreamerpro.loss(
                self.feat2tensor(aug_feat), embed_aug, target_embed, freeze)
            return {}, {}, losses, metrics

        raise NotImplementedError(f"Unsupported rep_loss: {self.rep_loss}")

    def report(self, carry, data):
        if not self.config.report:
            return carry, {}

        carry, obs, prevact, _, _ = self._apply_replay_context(carry, data)
        actions = self._current_actions(data)
        (enc_carry, dyn_carry, dec_carry) = carry
        B, T = obs["is_first"].shape
        RB = min(6, B)
        metrics = {}

        _, (new_carry, entries, outs, mets) = self.loss(
            carry, obs, prevact, actions, training=False)
        del entries
        metrics.update(mets)

        if self.config.report_gradnorms:
            for key in self.scales:
                try:
                    lossfn = lambda data, carry: self.loss(
                        carry, obs, prevact, actions, training=False)[1][2]["losses"][key].mean()
                    grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
                    metrics[f"gradnorm/{key}"] = optax.global_norm(grad)
                except KeyError:
                    print(f"Skipping gradnorm summary for missing loss: {key}")

        if self.rep_loss != "dreamer":
            carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
            return carry, metrics

        firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
        secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
        dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
        dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
        dyn_carry, _, obsfeat = self.dyn.observe(
            dyn_carry, firsthalf(outs["tokens"]), firsthalf(prevact),
            firsthalf(obs["is_first"]), training=False)
        _, imgfeat, _ = self.dyn.imagine(
            dyn_carry, secondhalf(prevact), length=T - T // 2, training=False)
        dec_carry, _, obsrecons = self.dec(
            dec_carry, obsfeat, firsthalf(obs["is_first"]), training=False)
        dec_carry, _, imgrecons = self.dec(
            dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs["is_first"])),
            training=False)

        for key in self.dec.imgkeys:
            assert obs[key].dtype == jnp.uint8
            true = obs[key][:RB]
            pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
            pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
            error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
            video = jnp.concatenate([true, pred, error], 2)

            video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
            mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
            border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
            border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
            video = jnp.where(mask, video, border[None, :, None, None, :])
            video = jnp.concatenate([video, 0 * video[:, :10]], 1)

            B, T, H, W, C = video.shape
            grid = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
            metrics[f"openloop/{key}"] = grid

        carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
        return carry, metrics

    def _current_actions(self, data):
        if not self.config.replay_context:
            return {k: data[k] for k in self.act_space}
        K = self.config.replay_context
        return {k: data[k][:, K:] for k in self.act_space}

    def _reset_optimizer_state_on_create(self, optimizers=None):
        if not nj.creating():
            return
        if optimizers is None:
            optimizers = [self.main_opt]
            if self.task_opt is not None:
                optimizers.append(self.task_opt)
        for optimizer in optimizers:
            for name in ("state", "adaptive"):
                state = optimizer._submodules.get(name)
                if state is not None:
                    state.treedef = None


def _encoder_output_dim(enc, obs_space):
    output_dim = getattr(enc, "output_dim", 0)
    if output_dim:
        return int(output_dim)
    total = 0
    if enc.veckeys:
        total += int(enc.units)
    if enc.imgkeys:
        key = sorted(enc.imgkeys)[0]
        h, w = obs_space[key].shape[:2]
        layers = len(enc.depths)
        if enc.outer:
            factor = 2 ** max(layers - 1, 0)
        else:
            factor = 2 ** layers
        h = int(h) // factor
        w = int(w) // factor
        total += h * w * int(enc.depths[-1])
    if total <= 0:
        raise ValueError(f"Could not infer encoder output dimension from {obs_space}.")
    return total


Agent = DreamerAgent
