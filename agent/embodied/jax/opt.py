import math
from typing import ClassVar, NamedTuple

import jax
import jax.numpy as jnp
import ninjax as nj
import optax

from . import internal, nets

f32 = jnp.float32
i32 = jnp.int32
sg = jax.lax.stop_gradient


class Optimizer(nj.Module):

  summary_depth: int = 2
  log_grads: bool = True

  def __init__(self, modules, opt, adaptive=None):
    modules = modules if isinstance(modules, (list, tuple)) else (modules,)
    self.modules = modules
    self.opt = opt
    self.adaptive = adaptive
    self.step = nj.Variable(jnp.array, 0, i32, name='step')
    self.scaling = (nets.COMPUTE_DTYPE == jnp.float16)
    if self.scaling:
      self.opt = optax.apply_if_finite(self.opt, max_consecutive_errors=1000)
      self.grad_scale = nj.Variable(jnp.array, 1e4, f32, name='grad_scale')
      self.good_steps = nj.Variable(jnp.array, 0, i32, name='good_steps')

  def __call__(self, lossfn, *args, has_aux=False, **kwargs):
    metrics = {}

    def lossfn2(*args, **kwargs):
      outs = lossfn(*args, **kwargs)
      loss, aux = outs if has_aux else (outs, None)
      assert loss.dtype == f32, (self.name, loss.dtype)
      assert loss.shape == (), (self.name, loss.shape)
      if self.scaling:
        loss *= sg(self.grad_scale.read())
      return loss, aux

    loss, params, grads, aux = nj.grad(
        lossfn2, self.modules, has_aux=True)(*args, **kwargs)
    if self.scaling:
      loss *= 1 / self.grad_scale.read()

    counts = {k: math.prod(v.shape) for k, v in params.items()}
    if nj.creating():
      print(self._summarize_params(counts, self.summary_depth))

    axes = internal.get_data_axes()
    if axes:
      grads = jax.tree.map(lambda x: jax.lax.pmean(x, axes), grads)

    if self.scaling:
      invscale = 1 / self.grad_scale.read()
      grads = jax.tree.map(lambda x: x * invscale, grads)

    state = self.sub('state', nj.Tree, self.opt.init, params)
    updates, new_state = self.opt.update(grads, state.read(), params)
    adaptive_metrics = {}
    if self.adaptive is not None:
      adaptive_advance = True
      if self.scaling:
        adaptive_advance = jnp.stack([
            jnp.isfinite(x).all() for x in jax.tree.leaves(grads)]).all()
      adaptive = self.sub('adaptive', nj.Tree, self.adaptive.init, params)
      updates, new_adaptive, adaptive_metrics = self.adaptive.update(
          updates, grads, adaptive.read(), advance=adaptive_advance)
      adaptive.write(new_adaptive)
    nj.context().update(optax.apply_updates(params, updates))
    state.write(new_state)
    grad_norm = (
        optax.global_norm(grads) if self.scaling or self.log_grads
        else None)
    if self.scaling:
      self._update_scale(grads, jnp.isfinite(grad_norm))
      grad_norm = jnp.where(jnp.isfinite(grad_norm), grad_norm, jnp.nan)
      self.step.write(self.step.read() + i32(jnp.isfinite(grad_norm)))
      metrics['grad_scale'] = self.grad_scale.read()
      metrics['grad_overflow'] = f32(~jnp.isfinite(grad_norm))
    else:
      self.step.write(self.step.read() + 1)
    metrics['loss'] = loss.mean()
    metrics['updates'] = self.step.read()
    if grad_norm is not None:
      metrics['grad_norm'] = grad_norm
    if self.log_grads:
      metrics['grad_rms'] = nets.rms(grads)
      metrics['update_rms'] = nets.rms(updates)
      metrics['param_rms'] = nets.rms([x.values for x in self.modules])
    metrics['param_count'] = jnp.array(list(counts.values()), f32).sum()
    metrics.update(adaptive_metrics)
    metrics = {f'{self.name}/{k}': v for k, v in metrics.items()}
    return (metrics, aux) if has_aux else metrics

  def _update_scale(self, grads, finite):
    keep = (finite & (self.good_steps.read() < 1000))
    incr = (finite & (self.good_steps.read() >= 1000))
    decr = ~finite
    self.good_steps.write(i32(keep) * (self.good_steps.read() + 1))
    self.grad_scale.write(jnp.clip(
        f32(keep) * self.grad_scale.read() +
        f32(incr) * self.grad_scale.read() * 2 +
        f32(decr) * self.grad_scale.read() / 2, 1e-4, 1e5))
    return finite

  def _summarize_params(self, counts, depth):
    lines = []
    pfxs = []
    for key in counts:
      parts = key.split('/')
      pfxs += ['/'.join(parts[: i + 1]) for i in range(min(len(parts), depth))]
    subcounts = {
        prefix: sum(v for k, v in counts.items() if k.startswith(prefix))
        for prefix in set(pfxs)}
    lines = [f'Optimizer {self.name} has {sum(counts.values()):,} params:']
    for prefix, count in sorted(subcounts.items(), key=lambda x: -x[1]):
      lines.append(f'{count:>14,} {prefix}')
    return '\n'.join(lines)


class AdaptiveState(NamedTuple):
  step: jax.Array
  log_scales: dict
  traces: dict


class NormalizedIDBD:
  """Bounded delayed hypergradient scaling for batched nonlinear critics.

  This is a normalized IDBD approximation, not literal linear IDBD or
  SwiftTD. It uses the normalized current raw gradient to evaluate whether an
  exponential trace of normalized previous *actual optimizer updates* was
  helpful on the next batch. The EMA trace is used at its accumulated
  magnitude rather than being normalized a second time. The resulting
  positive multiplier is applied to the final Optax update, so it composes
  with AGC, RMS normalization, momentum, and learning-rate schedules instead
  of being canceled by them.

  ``feature`` granularity shares one rate across each input row of a linear
  kernel and falls back to one rate per tensor for all other parameters.
  """

  _CRITICS: ClassVar[dict[str, tuple[str, ...]]] = {
      'value': ('value',),
      'expl_value': ('expl_value',),
      'both': ('value', 'expl_value'),
  }
  _PREFIXES: ClassVar[dict[str, str]] = {
      'value': 'val/',
      'expl_value': 'expl_val/',
  }

  def __init__(
      self,
      critics='value',
      scope='output',
      granularity='feature',
      meta_lr=1e-3,
      trace_decay=0.9,
      init_scale=1.0,
      min_scale=0.25,
      max_scale=4.0,
      warmup=1000,
      eps=1e-8,
  ):
    critics = str(critics).lower()
    scope = str(scope).lower()
    granularity = str(granularity).lower()
    if critics not in self._CRITICS:
      raise ValueError(
          'adaptive critic selection must be value, expl_value, or both; '
          f'got {critics!r}')
    if scope not in ('output', 'critic'):
      raise ValueError(
          "adaptive critic scope must be 'output' or 'critic'; "
          f'got {scope!r}')
    if granularity not in ('feature', 'tensor'):
      raise ValueError(
          "adaptive critic granularity must be 'feature' or 'tensor'; "
          f'got {granularity!r}')
    meta_lr = float(meta_lr)
    trace_decay = float(trace_decay)
    init_scale = float(init_scale)
    min_scale = float(min_scale)
    max_scale = float(max_scale)
    warmup = int(warmup)
    eps = float(eps)
    if meta_lr < 0:
      raise ValueError(f'adaptive critic meta_lr must be nonnegative: {meta_lr}')
    if not 0 <= trace_decay < 1:
      raise ValueError(
          'adaptive critic trace_decay must be in [0, 1): '
          f'{trace_decay}')
    if not 0 < min_scale <= init_scale <= max_scale:
      raise ValueError(
          'adaptive critic scales must satisfy '
          f'0 < min <= init <= max, got '
          f'{min_scale}, {init_scale}, {max_scale}')
    if warmup < 0:
      raise ValueError(f'adaptive critic warmup must be nonnegative: {warmup}')
    if eps <= 0:
      raise ValueError(f'adaptive critic eps must be positive: {eps}')
    self.critics = self._CRITICS[critics]
    self.scope = scope
    self.granularity = granularity
    self.meta_lr = meta_lr
    self.trace_decay = trace_decay
    self.init_scale = init_scale
    self.min_scale = min_scale
    self.max_scale = max_scale
    self.warmup = warmup
    self.eps = eps
    self._log_min = math.log(min_scale)
    self._log_max = math.log(max_scale)

  def init(self, params):
    selected = {k: v for k, v in params.items() if self._selected(k)}
    if not selected:
      names = ', '.join(self.critics)
      raise ValueError(
          f'adaptive critic selected no parameters for {names} '
          f'with scope {self.scope!r}')
    initial = f32(math.log(self.init_scale))
    log_scales = {
        key: jnp.full(self._stat_shape(key, value), initial, f32)
        for key, value in selected.items()
    }
    traces = {
        key: jnp.zeros_like(value, dtype=f32)
        for key, value in selected.items()
    }
    return AdaptiveState(jnp.zeros((), i32), log_scales, traces)

  def update(self, updates, grads, state, advance=True):
    updates = dict(updates)
    log_scales = {}
    traces = {}
    stats = {}
    advance = jnp.asarray(advance, bool)
    adapt = advance & (state.step >= self.warmup)

    for key, old_log_scale in state.log_scales.items():
      axes = self._reduction_axes(key, updates[key])
      grad_direction, grad_valid = self._normalize(grads[key], axes)
      trace, trace_rms, trace_valid = self._rms(
          state.traces[key], axes)
      # Each instantaneous update is normalized before entering the EMA. Use
      # the EMA directly here so inconsistent directions attenuate the score
      # through the trace magnitude instead of normalizing that evidence away.
      score = -self._mean(grad_direction * trace, axes)
      valid_score = advance & grad_valid & trace_valid & jnp.isfinite(score)
      score = jnp.where(valid_score, jnp.clip(score, -1.0, 1.0), 0.0)

      proposed_log_scale = old_log_scale + self.meta_lr * score
      candidate = jnp.clip(
          proposed_log_scale, self._log_min, self._log_max)
      scale_clipped = (
          adapt & valid_score & (candidate != proposed_log_scale))
      new_log_scale = jnp.where(adapt & valid_score, candidate, old_log_scale)
      old_scale = jnp.exp(old_log_scale)
      scale = jnp.exp(new_log_scale)
      scale_delta = scale - old_scale
      scaled_update = updates[key] * scale.astype(updates[key].dtype)
      update_direction, _ = self._normalize(scaled_update, axes)
      update_finite = self._all(
          jnp.isfinite(f32(scaled_update)), axes)
      candidate_trace = (
          self.trace_decay * state.traces[key] +
          (1 - self.trace_decay) * update_direction)
      # Finite zero updates decay old evidence instead of keeping an
      # arbitrarily stale direction. Nonfinite updates freeze the trace.
      new_trace = jnp.where(
          advance & update_finite, candidate_trace, state.traces[key])

      updates[key] = scaled_update
      log_scales[key] = new_log_scale
      traces[key] = new_trace
      critic = self._critic_for_key(key)
      values = stats.setdefault(critic, {
          'scale': [], 'score': [], 'trace_rms': [], 'scale_delta': [],
          'scale_clipped': [], 'valid': [], 'lower': [], 'upper': [],
          'param_count': 0,
      })
      values['scale'].append(scale.reshape(-1))
      values['score'].append(score.reshape(-1))
      values['trace_rms'].append(trace_rms.reshape(-1))
      values['scale_delta'].append(scale_delta.reshape(-1))
      values['scale_clipped'].append(scale_clipped.reshape(-1))
      values['valid'].append(valid_score.reshape(-1))
      values['lower'].append(
          (new_log_scale <= self._log_min + 1e-6).reshape(-1))
      values['upper'].append(
          (new_log_scale >= self._log_max - 1e-6).reshape(-1))
      values['param_count'] += math.prod(updates[key].shape)

    metrics = {}
    for critic, values in stats.items():
      scales = jnp.concatenate(values['scale'])
      scores = jnp.concatenate(values['score'])
      trace_rms = jnp.concatenate(values['trace_rms'])
      scale_delta = jnp.concatenate(values['scale_delta'])
      scale_clipped = jnp.concatenate(
          values['scale_clipped']).astype(f32)
      valid = jnp.concatenate(values['valid']).astype(f32)
      lower = jnp.concatenate(values['lower']).astype(f32)
      upper = jnp.concatenate(values['upper']).astype(f32)
      valid_count = jnp.maximum(valid.sum(), 1.0)
      root = f'adaptive/{critic}'
      metrics[f'{root}/scale_mean'] = scales.mean()
      metrics[f'{root}/scale_std'] = scales.std()
      metrics[f'{root}/scale_min'] = scales.min()
      metrics[f'{root}/scale_max'] = scales.max()
      metrics[f'{root}/trace_rms'] = trace_rms.mean()
      metrics[f'{root}/meta_score'] = (scores * valid).sum() / valid_count
      metrics[f'{root}/abs_meta_score'] = (
          jnp.abs(scores) * valid).sum() / valid_count
      # Alias for consumers of the adaptive-scale metric.
      metrics[f'{root}/meta_score_abs'] = metrics[
          f'{root}/abs_meta_score']
      metrics[f'{root}/scale_delta'] = scale_delta.mean()
      metrics[f'{root}/scale_clip_fraction'] = scale_clipped.mean()
      metrics[f'{root}/positive_score_fraction'] = (
          (scores > 0).astype(f32) * valid).sum() / valid_count
      metrics[f'{root}/negative_score_fraction'] = (
          (scores < 0).astype(f32) * valid).sum() / valid_count
      metrics[f'{root}/valid_fraction'] = valid.mean()
      metrics[f'{root}/lower_bound_fraction'] = lower.mean()
      metrics[f'{root}/upper_bound_fraction'] = upper.mean()
      metrics[f'{root}/bound_fraction'] = jnp.maximum(lower, upper).mean()
      metrics[f'{root}/active'] = f32(adapt)
      metrics[f'{root}/rate_count'] = f32(scales.size)
      metrics[f'{root}/param_count'] = f32(values['param_count'])

    step = jnp.where(
        advance, optax.safe_int32_increment(state.step), state.step)
    return updates, AdaptiveState(step, log_scales, traces), metrics

  def _selected(self, key):
    critic = self._critic_for_key(key)
    if critic not in self.critics:
      return False
    prefix = self._PREFIXES[critic]
    return self.scope == 'critic' or key.startswith(f'{prefix}head/')

  def _critic_for_key(self, key):
    for critic, prefix in self._PREFIXES.items():
      if key.startswith(prefix):
        return critic
    return None

  def _reduction_axes(self, key, value):
    if (
        self.granularity == 'feature' and
        key.endswith('/kernel') and
        value.ndim >= 2
    ):
      return tuple(range(1, value.ndim))
    return tuple(range(value.ndim))

  def _stat_shape(self, key, value):
    axes = self._reduction_axes(key, value)
    return tuple(1 if index in axes else size
                 for index, size in enumerate(value.shape))

  def _mean(self, value, axes):
    return jnp.mean(value, axes, keepdims=True) if axes else value

  def _max(self, value, axes):
    return jnp.max(value, axes, keepdims=True) if axes else value

  def _all(self, value, axes):
    return jnp.all(value, axes, keepdims=True) if axes else value

  def _rms(self, value, axes):
    value = f32(value)
    finite = self._all(jnp.isfinite(value), axes)
    safe = jnp.where(jnp.isfinite(value), value, 0.0)
    magnitude = self._max(jnp.abs(safe), axes)
    relative = safe / jnp.maximum(magnitude, self.eps)
    rms = magnitude * jnp.sqrt(self._mean(relative * relative, axes))
    valid = finite & jnp.isfinite(rms) & (rms > self.eps)
    return safe, rms, valid

  def _normalize(self, value, axes):
    safe, rms, valid = self._rms(value, axes)
    direction = jnp.where(valid, safe / jnp.maximum(rms, self.eps), 0.0)
    return direction, valid


def adaptive_critic(enabled=False, **kwargs):
  """Builds the optional critic scaler without changing the disabled path."""
  return NormalizedIDBD(**kwargs) if enabled else None


def clip_by_agc(clip=0.3, pmin=1e-3):

  def init_fn(params):
    return ()

  def update_fn(updates, state, params=None):
    def fn(param, update):
      unorm = jnp.linalg.norm(update.flatten(), 2)
      pnorm = jnp.linalg.norm(param.flatten(), 2)
      upper = clip * jnp.maximum(pmin, pnorm)
      return update * (1 / jnp.maximum(1.0, unorm / upper))
    updates = jax.tree.map(fn, params, updates) if clip else updates
    return updates, ()

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_rms(beta=0.999, eps=1e-8):

  def init_fn(params):
    nu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), i32)
    return (step, nu)

  def update_fn(updates, state, params=None):
    step, nu = state
    step = optax.safe_int32_increment(step)
    nu = jax.tree.map(
        lambda v, u: beta * v + (1 - beta) * (u * u), nu, updates)
    nu_hat = optax.bias_correction(nu, beta, step)
    updates = jax.tree.map(
        lambda u, v: u / (jnp.sqrt(v) + eps), updates, nu_hat)
    return updates, (step, nu)

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_momentum(beta=0.9, nesterov=False):

  def init_fn(params):
    mu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), i32)
    return (step, mu)

  def update_fn(updates, state, params=None):
    step, mu = state
    step = optax.safe_int32_increment(step)
    mu = optax.update_moment(updates, mu, beta, 1)
    if nesterov:
      mu_nesterov = optax.update_moment(updates, mu, beta, 1)
      mu_hat = optax.bias_correction(mu_nesterov, beta, step)
    else:
      mu_hat = optax.bias_correction(mu, beta, step)
    return mu_hat, (step, mu)

  return optax.GradientTransformation(init_fn, update_fn)
