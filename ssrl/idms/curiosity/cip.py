"""Controllable Information Production for JAX Plan2Explore.

Approximate finite-horizon open-loop information production from the RSSM
Jacobian. The reward has no auxiliary prediction loss.
"""

import jax
import jax.numpy as jnp
import ninjax as nj

from .._runtime import f32, sg
from .common import RunningMeanNormalizer


class CIP(nj.Module):

    state_source: str = "deter"   # one of: feat/full, deter, stoch
    probes: int = 8              # probe subspace dimension
    scale: float = 1.0
    eps: float = 1e-6
    positive: bool = True        # sum only expanding probe directions
    normalize_rewards: bool = False
    seed: int = 0
    stop_reward_grad: bool = True
    compute_dtype: str = "bfloat16"

    def __init__(self):
        self._state_source = str(self.state_source).lower()
        if self._state_source == "full":
            self._state_source = "feat"
        if self._state_source not in {"feat", "deter", "stoch"}:
            raise ValueError(
                "CIP state_source must be one of {feat, full, deter, stoch}, "
                f"got {self.state_source!r}."
            )
        self._probes = int(self.probes)
        if self._probes <= 0:
            raise ValueError("CIP probes must be positive.")
        self._scale = float(self.scale)
        self._eps = float(self.eps)
        self._positive = bool(self.positive)
        self._seed = int(self.seed)
        self._stop_reward_grad = bool(self.stop_reward_grad)
        self._compute_dtype = jnp.dtype(self.compute_dtype)
        if not jnp.issubdtype(self._compute_dtype, jnp.floating):
            raise ValueError("CIP compute_dtype must be a floating-point dtype.")
        self.reward_rms = (
            RunningMeanNormalizer(eps=self._eps, name="reward_rms")
            if bool(self.normalize_rewards)
            else None
        )

    def loss(self, *args, **kwargs):
        del args, kwargs
        zero = jnp.zeros((), jnp.float32)
        return zero, {"cip_loss": zero}

    def compute_reward(
        self,
        dyn,
        feat,
        action,
        update_reward_norm: bool = True,
        sequence_layout: str = "batch_major",
        **kwargs,
    ):
        """Return per-step CIP rewards with shape [B, T, 1].

        Args:
          dyn: Dreamer RSSM module whose _core() and _prior() methods define
            a smooth latent transition without sampling.
          feat: RSSM feature tree containing at least `deter` and `stoch`, with
            either [B, T, ...] or [B, ...] leading dimensions.
          action: Flattened action tensor with matching leading dimensions.
        """
        del kwargs
        if action is None:
            raise ValueError("CIP requires actions because the RSSM transition is action-conditioned.")
        action = f32(action)
        squeezed = action.ndim == 2
        if squeezed:
            feat = jax.tree.map(lambda x: x[:, None], feat)
            action = action[:, None]

        if sequence_layout == "batch_major":
            feat_tm = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), feat)
            action_tm = jnp.swapaxes(action, 0, 1)
        elif sequence_layout == "time_major":
            feat_tm, action_tm = feat, action
        else:
            raise ValueError(f"Unsupported CIP sequence_layout={sequence_layout!r}.")

        z0 = self._state_from_feat(jax.tree.map(lambda x: x[0, 0], feat_tm))
        q0 = self._initial_basis(z0.shape[-1], z0.dtype)

        # vmap over batch; scan over time so adjoint probes track local
        # expansion along each imagined action sequence.
        rewards_tm = jax.vmap(
            lambda feat_seq, act_seq: self._sequence_reward(dyn, feat_seq, act_seq, q0),
            in_axes=(1, 1),
            out_axes=1,
        )(feat_tm, action_tm)
        rewards = jnp.swapaxes(rewards_tm, 0, 1) if sequence_layout == "batch_major" else rewards_tm
        rewards = rewards[..., None] * self._scale
        rewards = jnp.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
        if self.reward_rms is not None:
            rewards = self.reward_rms.normalize(rewards, update=update_reward_norm)
        if self._stop_reward_grad:
            rewards = sg(rewards)
        return rewards[:, 0] if squeezed else rewards

    def _sequence_reward(self, dyn, feat_seq, act_seq, q0):
        def step(q, inputs):
            feat_t, act_t = inputs
            q_next, reward_t = self._tangent_step(dyn, feat_t, act_t, q)
            return q_next, reward_t

        _, rewards = jax.lax.scan(step, q0, (feat_seq, act_seq))
        return rewards

    def _tangent_step(self, dyn, feat_t, act_t, q):
        state = self._state_from_feat(feat_t)

        def transition(z):
            return self._next_state(dyn, z, act_t, feat_t)

        _, pullback = jax.vjp(transition, state)

        def vjp_col(v):
            return pullback(v)[0]

        tangent = jax.vmap(vjp_col, in_axes=1, out_axes=1)(q)
        tangent = jnp.nan_to_num(tangent, nan=0.0, posinf=0.0, neginf=0.0)
        q_next, r = jnp.linalg.qr(tangent, mode="reduced")
        diag = jnp.abs(jnp.diag(r))
        logs = jnp.log(jnp.maximum(diag, self._eps))
        if self._positive:
            logs = jnp.maximum(logs, 0.0)
        reward = logs.sum()
        return q_next, reward

    def _initial_basis(self, dim, dtype):
        k = min(self._probes, int(dim))
        key = jax.random.PRNGKey(self._seed)
        mat = jax.random.normal(key, (int(dim), k), dtype=jnp.float32)
        q, _ = jnp.linalg.qr(mat, mode="reduced")
        return q.astype(dtype)

    def _state_from_feat(self, feat):
        deter = f32(feat["deter"])
        stoch = f32(feat["stoch"])
        if self._state_source == "deter":
            return deter
        if self._state_source == "stoch":
            return stoch.reshape((-1,))
        return jnp.concatenate([deter, stoch.reshape((-1,))], axis=-1)

    def _state_from_parts(self, deter, stoch):
        if self._state_source == "deter":
            return deter
        if self._state_source == "stoch":
            return stoch.reshape((-1,))
        return jnp.concatenate([deter, stoch.reshape((-1,))], axis=-1)

    def _state_to_parts(self, state, template):
        deter0 = f32(template["deter"])
        stoch0 = f32(template["stoch"])
        if self._state_source == "deter":
            return state, stoch0
        if self._state_source == "stoch":
            return deter0, state.reshape(stoch0.shape)
        deter_dim = deter0.shape[-1]
        deter = state[:deter_dim]
        stoch = state[deter_dim:].reshape(stoch0.shape)
        return deter, stoch

    def _next_state(self, dyn, state, action, template):
        deter, stoch = self._state_to_parts(state, template)
        deter_in, stoch_in, action_in = jax.tree.map(
            lambda x: x.astype(self._compute_dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            (deter[None], stoch[None], action[None]),
        )
        next_deter = dyn._core(deter_in, stoch_in, action_in)[0]
        if self._state_source == "deter":
            return f32(next_deter)
        next_logit = dyn._prior(next_deter[None])[0]
        next_stoch = self._prior_probs(dyn, next_logit)
        return self._state_from_parts(f32(next_deter), next_stoch)

    def _prior_probs(self, dyn, logit):
        prob = jax.nn.softmax(f32(logit), axis=-1)
        unimix = float(getattr(dyn, "unimix", 0.0) or 0.0)
        if unimix:
            uniform = jnp.ones_like(prob) / prob.shape[-1]
            prob = (1.0 - unimix) * prob + unimix * uniform
        return prob
