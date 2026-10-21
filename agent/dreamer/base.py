"""DreamerV3 base agent used by the Dreamer variants."""

import re

import chex
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import rssm

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


def sequence_indices(batch_size, length, count, strategy, seed=None):
  """Selects time indices while keeping each replay sequence represented."""
  batch_size, length, count = map(int, (batch_size, length, count))
  if not 1 <= count <= length:
    raise ValueError((count, length))
  strategy = str(strategy).lower()
  if strategy == 'last':
    indices = jnp.arange(length - count, length, dtype=i32)
    return jnp.broadcast_to(indices[None], (batch_size, count))
  if strategy not in ('stratified', 'random_contiguous'):
    raise ValueError(
        "sequence strategy must be 'last', 'stratified', or "
        f"'random_contiguous', got {strategy!r}")
  if seed is None:
    raise ValueError(f'{strategy} sequence selection requires a random seed.')
  if strategy == 'random_contiguous':
    # When possible, sample one aligned block. Every timestep then has the
    # same inclusion probability while the selected subsequence stays
    # contiguous for lambda-return and replay-return losses.
    if length % count == 0:
      block = jax.random.randint(
          seed, (batch_size, 1), 0, length // count, dtype=i32)
      start = block * count
    else:
      start = jax.random.randint(
          seed, (batch_size, 1), 0, length - count + 1, dtype=i32)
    return start + jnp.arange(count, dtype=i32)[None]
  # Integer stratum edges cover [0, length) exactly. Since count <= length,
  # every stratum is non-empty and the sampled indices are unique.
  slots = jnp.arange(count, dtype=i32)
  lower = slots * length // count
  upper = (slots + 1) * length // count
  width = upper - lower
  uniform = jax.random.uniform(seed, (batch_size, count), dtype=f32)
  offset = jnp.floor(uniform * width[None]).astype(i32)
  return lower[None] + offset


def stratified_sequence_weights(length, count):
  """Horvitz-Thompson weights for one sample from each integer stratum."""
  length, count = int(length), int(count)
  if not 1 <= count <= length:
    raise ValueError((count, length))
  slots = jnp.arange(count, dtype=i32)
  lower = slots * length // count
  upper = (slots + 1) * length // count
  return (upper - lower).astype(f32)


def gather_sequence(xs, indices):
  """Gathers a different set of time indices for each batch element."""
  indices = jnp.asarray(indices, i32)
  assert indices.ndim == 2, indices.shape
  batch = jnp.arange(indices.shape[0], dtype=i32)[:, None]

  def gather(x):
    assert x.shape[0] == indices.shape[0], (x.shape, indices.shape)
    return x[batch, indices]

  return jax.tree.map(gather, xs)


def scatter_sequence(xs, indices, length, weights=None):
  """Scatters sampled losses back to time, preserving their sample mean."""
  indices = jnp.asarray(indices, i32)
  assert indices.ndim == 2, indices.shape
  batch_size, count = indices.shape
  assert 1 <= count <= length, (count, length)
  batch = jnp.arange(batch_size, dtype=i32)[:, None]
  if weights is None:
    weights = jnp.full((count,), length / count, f32)
  else:
    weights = jnp.asarray(weights, f32)
    assert weights.shape == (count,), (weights.shape, count)

  def scatter(x):
    assert x.shape[:2] == (batch_size, count), (x.shape, indices.shape)
    out = jnp.zeros((batch_size, length, *x.shape[2:]), x.dtype)
    shape = (1, count, *((1,) * (x.ndim - 2)))
    return out.at[batch, indices].set(x * weights.reshape(shape))

  return jax.tree.map(scatter, xs)


def policy_action(outputs, mode):
  """Samples the learned policy unless greedy evaluation is requested."""
  if isinstance(mode, str) and (
      mode == 'eval_greedy' or mode.startswith('eval_greedy:')):
    return jax.tree.map(lambda output: output.pred(), outputs)
  return sample(outputs)


def getcfg(config, key, default=None):
  if config is None:
    return default
  if isinstance(config, dict):
    return config.get(key, default)
  return getattr(config, key, default)


def _safe_task_label(label):
  return re.sub(r'[^0-9A-Za-z_]+', '_', str(label)).strip('_') or 'eval'


def _task_head_specs(value):
  if not value:
    return []
  if isinstance(value, str):
    value = [value]
  if isinstance(value, dict):
    if getcfg(value, 'task', None):
      value = [value]
    else:
      value = [{'label': label, 'task': task} for label, task in value.items()]
  result = []
  for spec in value:
    if spec is None:
      continue
    if isinstance(spec, str):
      text = spec.strip()
      if text.lower() in ('', 'none', 'null', 'false', 'off', '0'):
        continue
      if '=' in text:
        label, task = text.split('=', 1)
      elif ':' in text:
        label, task = text.split(':', 1)
      else:
        label = task = text
      result.append({'label': label.strip(), 'task': task.strip()})
    else:
      result.append(spec)
  return result


class Agent(embodied.jax.Agent):

  banner = [
      r"---  ___                           __   ______ ---",
      r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
      r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
      r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
  ]

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config

    exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    self.enc = rssm.ENCODERS[config.enc.typ](
        enc_space, **config.enc[config.enc.typ], name='enc')
    self._encoder_modules = (
        [self.enc] if getattr(self.enc, 'trainable', True) else [])
    self.dyn = {
        'rssm': rssm.RSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn')
    self.dec = {
        'simple': rssm.Decoder,
    }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec')

    self.feat2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')

    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='pol')

    self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
    self.slowval = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
        source=self.val, **config.slowvalue)

    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    self.task_heads = self._make_task_heads(scalar, act_space, outs)
    self.train_base_task_behavior = bool(
        getcfg(config, 'train_base_task_behavior', True))

    base_behavior = (
        [self.rew, self.pol, self.val]
        if self.train_base_task_behavior else [])
    self.modules = [
        self.dyn, *self._encoder_modules, self.dec, self.con, *base_behavior]
    self.task_modules = list(self.task_heads['modules'])
    # The shared optimizer uses the ``opt`` checkpoint namespace.
    self.main_opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt),
        adaptive=embodied.jax.opt.adaptive_critic(
            **(getcfg(config, 'adaptive_critic', {}) or {})), summary_depth=1,
        log_grads=getcfg(config, 'log_grads', True),
        name='opt')
    self.opt = self.main_opt
    self.task_opt = (
        embodied.jax.Optimizer(
            self.task_modules, self._make_opt(**config.opt), summary_depth=1,
            log_grads=getcfg(config, 'log_grads', True), name='task_opt')
        if self.task_modules else None)

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    if not self.train_base_task_behavior:
      for key in ('rew', 'policy', 'value', 'repval'):
        scales.pop(key, None)
    self.scales = scales
    self.task_scales = self._make_task_scales()

  @property
  def policy_keys(self):
    task_pols = '|'.join(
        re.escape(head.name) for head in self.task_heads['pols'].values())
    task_pols = f'|{task_pols}' if task_pols else ''
    return f'^(enc|dyn|pol{task_pols})/'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    for key in self._task_label_keys():
      spaces[key] = elements.Space(np.float32, ())
    if self.config.replay_context:
      spaces.update(elements.tree.flatdict(dict(
          enc=self.enc.entry_space,
          dyn=self.dyn.entry_space,
          dec=self.dec.entry_space)))
    return spaces

  def init_policy(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space))

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  def policy(self, carry, obs, mode='train'):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw)
    dec_entry = {}
    if dec_carry:
      dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)
    act = self._policy_action(feat, mode)
    out = {}
    if getattr(self.config, 'policy_checks', False):
      out['finite'] = elements.tree.flatdict(jax.tree.map(
          lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
          dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    carry = (enc_carry, dyn_carry, dec_carry, act)
    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return carry, act, out

  def train(self, carry, data):
    carry, obs, prevact, stepid, task_labels = self._apply_replay_context(
        carry, data)
    metrics, (carry, entries, outs, mets) = self.main_opt(
        self.loss, carry, obs, prevact, training=True, has_aux=True)
    metrics.update(mets)
    if self.train_base_task_behavior:
      self.slowval.update()
    if self.task_opt is not None:
      task_metrics, task_mets = self.task_opt(
          self.task_loss, outs['repfeat'], entries[1], carry[1], obs,
          prevact, task_labels, training=True, has_aux=True)
      metrics.update(task_metrics)
      metrics.update(task_mets)
      for slowval in self.task_heads['slowvals'].values():
        slowval.update()
    outs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, outs, metrics

  def loss(self, carry, obs, prevact, training):
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    # World model
    enc_carry, enc_entries, tokens = self.enc(
        enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)
    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(
        dec_carry, repfeat, reset, training)
    inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
    if self.train_base_task_behavior:
      losses['rew'] = self.rew(inp, 2).loss(obs['reward'])
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      losses[key] = recon.loss(sg(target))

    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    if self.train_base_task_behavior:
      # Imagination
      K = self._imag_start_count(T)
      H = self.config.imag_length
      imag_indices = self._sequence_selection(
          B, T, K, getcfg(self.config, 'imag_start_strategy', 'last'))
      starts, first = self._imag_start_data(
          repfeat, dyn_entries, dyn_carry, K, imag_indices)
      policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
      _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
      imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
      lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
      lastact = jax.tree.map(lambda x: x[:, None], lastact)
      imgact = concat([imgprevact, lastact], 1)
      assert all(
          x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
      assert all(
          x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
      inp = self.feat2tensor(imgfeat)
      los, _, mets = imag_loss(
          imgact,
          self.rew(inp, 2).pred(),
          self.con(inp, 2).prob(1),
          self.pol(inp, 2),
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.retnorm, self.valnorm, self.advnorm,
          update=training,
          contdisc=self.config.contdisc,
          horizon=self.config.horizon,
          **self.config.imag_loss)
      losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
      metrics.update(mets)

      # Replay
      if self.config.repval_loss:
        last, term, rew = [
            obs[k] for k in ('is_last', 'is_terminal', 'reward')]
        los, _, mets = self._replay_value_loss(
            repfeat, last, term, rew, self.val, self.slowval, self.valnorm,
            K, training, repval_grad=self.config.repval_grad)
        losses.update(los)
        metrics.update(prefix(mets, 'reploss'))

    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, outs, metrics)

  def report(self, carry, data):
    if not self.config.report:
      return carry, {}

    carry, obs, prevact, _, _ = self._apply_replay_context(carry, data)
    (enc_carry, dyn_carry, dec_carry) = carry
    B, T = obs['is_first'].shape
    RB = min(6, B)
    metrics = {}

    # Train metrics
    _, (new_carry, entries, outs, mets) = self.loss(
        carry, obs, prevact, training=False)
    mets.update(mets)

    # Gradient norms
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              carry, obs, prevact, training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
    dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact),
        firsthalf(obs['is_first']), training=False)
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, secondhalf(prevact), length=T - T // 2, training=False)
    dec_carry, _, obsrecons = self.dec(
        dec_carry, obsfeat, firsthalf(obs['is_first']), training=False)
    dec_carry, _, imgrecons = self.dec(
        dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])),
        training=False)

    # Video predictions
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
      metrics[f'openloop/{key}'] = grid

    carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, metrics

  def _make_task_heads(self, scalar, act_space, outs):
    specs = _task_head_specs(getcfg(self.config, 'eval_task_heads', []))
    result = {
        'specs': [],
        'rews': {},
        'pols': {},
        'vals': {},
        'slowvals': {},
        'retnorms': {},
        'valnorms': {},
        'advnorms': {},
        'modules': [],
    }
    for spec in specs:
      task_value = getcfg(spec, 'task', None)
      if not task_value:
        raise ValueError(f'Invalid eval task head spec: {spec!r}')
      task = str(task_value)
      label = _safe_task_label(getcfg(spec, 'label', task))
      if label in result['rews']:
        raise ValueError(f'Duplicate eval task head label: {label!r}')
      rew = embodied.jax.MLPHead(
          scalar, **self.config.rewhead, name=f'task_rew_{label}')
      pol = embodied.jax.MLPHead(
          act_space, outs, **self.config.policy, name=f'task_pol_{label}')
      val = embodied.jax.MLPHead(
          scalar, **self.config.value, name=f'task_val_{label}')
      slowval = embodied.jax.SlowModel(
          embodied.jax.MLPHead(
              scalar, **self.config.value, name=f'task_slowval_{label}'),
          source=val, **self.config.slowvalue)
      result['specs'].append({'label': label, 'task': task})
      result['rews'][label] = rew
      result['pols'][label] = pol
      result['vals'][label] = val
      result['slowvals'][label] = slowval
      result['retnorms'][label] = embodied.jax.Normalize(
          **self.config.retnorm, name=f'task_retnorm_{label}')
      result['valnorms'][label] = embodied.jax.Normalize(
          **self.config.valnorm, name=f'task_valnorm_{label}')
      result['advnorms'][label] = embodied.jax.Normalize(
          **self.config.advnorm, name=f'task_advnorm_{label}')
      result['modules'].extend([rew, pol, val])
    return result

  def _make_task_scales(self):
    scales = {}
    configured = getcfg(self.config, 'task_head_loss_scales', {}) or {}
    defaults = {'rew': 1.0, 'policy': 1.0, 'value': 1.0, 'repval': 0.3}
    for spec in self.task_heads['specs']:
      label = spec['label']
      scales[f'task/{label}/rew'] = float(
          getcfg(configured, 'rew', defaults['rew']))
      scales[f'task/{label}/policy'] = float(
          getcfg(configured, 'policy', defaults['policy']))
      scales[f'task/{label}/value'] = float(
          getcfg(configured, 'value', defaults['value']))
      if self.config.repval_loss:
        scales[f'task/{label}/repval'] = float(
            getcfg(configured, 'repval', defaults['repval']))
    return scales

  def _add_task_scales(self, scales):
    """Compatibility helper for callers that need a combined scale table."""
    scales.update(self._make_task_scales())

  def _mode_task_label(self, mode):
    if isinstance(mode, str):
      for prefix_ in ('eval:', 'eval_greedy:'):
        if mode.startswith(prefix_):
          return _safe_task_label(mode.split(':', 1)[1])
    return None

  def _policy_head_for_mode(self, mode):
    label = self._mode_task_label(mode)
    if label:
      if label in self.task_heads['pols']:
        return self.task_heads['pols'][label]
      raise KeyError(
          f'No evaluation task head for {label!r}; available heads: '
          f'{sorted(self.task_heads["pols"])}')
    return self.pol

  def _policy_action(self, feat, mode):
    policy_head = self._policy_head_for_mode(mode)
    policy = policy_head(self.feat2tensor(feat), bdims=1)
    return policy_action(policy, mode)

  def _task_label_key(self, spec):
    task = str(spec['task'])
    if task.startswith('robokitchen_'):
      return f'label/reward/{task.removeprefix("robokitchen_")}'
    if (task.startswith('dmc_walker_') and
        getcfg(self.config, 'walker_task_reward_labels', False)):
      return f'label/reward/{task.removeprefix("dmc_")}'
    return None

  def _task_label_keys(self):
    return tuple(
        key for spec in self.task_heads['specs']
        if (key := self._task_label_key(spec)) is not None)

  def _sequence_selection(self, batch_size, length, count, strategy):
    """Returns sampled indices, or None when every time step is selected."""
    count = min(int(count), int(length))
    if count == length:
      return None
    strategy = str(strategy).lower()
    seed = nj.seed() if strategy != 'last' else None
    return sequence_indices(batch_size, length, count, strategy, seed)

  def _select_sequence(self, xs, indices):
    return xs if indices is None else gather_sequence(xs, indices)

  def _imag_start_count(self, length):
    """Resolves and validates the runtime imagination-start count."""
    length = int(length)
    count = min(int(getcfg(self.config, 'imag_last', 0)) or length, length)
    minimum = min(2, length) if getcfg(self.config, 'repval_loss', True) else 1
    if count < minimum:
      raise ValueError(
          'imag_last must select at least '
          f'{minimum} replay positions when repval_loss is enabled; got '
          f'{count} for sequence length {length}.')
    return count

  def _imag_start_data(
      self, repfeat, dyn_entries, dyn_carry, count, indices):
    selected_entries = self._select_sequence(dyn_entries, indices)
    selected_feat = self._select_sequence(repfeat, indices)
    batch_size = jax.tree.leaves(repfeat)[0].shape[0]
    starts = self.dyn.starts(selected_entries, dyn_carry, count)
    first = jax.tree.map(
        lambda x: x.reshape(
            (batch_size * count, 1, *x.shape[2:])), selected_feat)
    return starts, first

  def _replay_value_loss(
      self, repfeat, last, term, rew, value_head, slowvalue_head, valnorm,
      count, training, repval_grad=False):
    """Fits sampled values to TD(lambda) targets from the full replay span."""
    batch_size, length = last.shape
    if length < 2:
      zero = jnp.zeros((batch_size, 1), f32)
      return {'repval': zero}, {'ret': zero[:, :0]}, {}

    value_count = min(int(count), length - 1)
    value_indices = self._sequence_selection(
        batch_size, length - 1, value_count, 'stratified')

    repl_config = dict(self.config.repl_loss)
    slowtar = bool(repl_config.pop('slowtar', True))
    target_head = slowvalue_head if slowtar else value_head
    voffset, vscale = valnorm.stats()
    target_inp = self.feat2tensor(sg(repfeat))
    bootstrap = sg(target_head(target_inp, 2).pred() * vscale + voffset)

    online_feat = sg(repfeat, skip=repval_grad)
    online_feat = jax.tree.map(lambda x: x[:, :-1], online_feat)
    online_feat = self._select_sequence(online_feat, value_indices)
    online_inp = self.feat2tensor(online_feat)
    return repl_loss(
        last, term, rew, bootstrap,
        value_head(online_inp, 2),
        slowvalue_head(online_inp, 2),
        valnorm,
        indices=value_indices,
        update=training,
        horizon=self.config.horizon,
        **repl_config)

  def _add_task_head_losses(
      self, losses, metrics, repfeat, dyn_entries, dyn_carry, obs, prevact,
      training, imag_indices=None, task_labels=None):
    if not self.task_heads['specs']:
      return

    B, T = obs['is_first'].shape
    K = self._imag_start_count(T)
    H = self.config.imag_length
    if imag_indices is None and K < T:
      imag_indices = self._sequence_selection(
          B, T, K, getcfg(self.config, 'imag_start_strategy', 'last'))
    repfeat = jax.tree.map(sg, repfeat)
    dyn_entries = jax.tree.map(sg, dyn_entries)
    dyn_carry = jax.tree.map(sg, dyn_carry)
    starts, first = self._imag_start_data(
        repfeat, dyn_entries, dyn_carry, K, imag_indices)
    starts = jax.tree.map(sg, starts)
    first = jax.tree.map(sg, first)

    for spec in self.task_heads['specs']:
      label = spec['label']
      rew_head = self.task_heads['rews'][label]
      pol_head = self.task_heads['pols'][label]
      val_head = self.task_heads['vals'][label]
      slowval = self.task_heads['slowvals'][label]
      retnorm = self.task_heads['retnorms'][label]
      valnorm = self.task_heads['valnorms'][label]
      advnorm = self.task_heads['advnorms'][label]

      target = self._task_reward(
          spec['task'], obs, prevact, task_labels, label=label)
      target = jnp.where(obs['is_first'], jnp.zeros_like(target), target)
      obs_inp = sg(self.feat2tensor(repfeat))
      losses[f'task/{label}/rew'] = rew_head(obs_inp, 2).loss(target)
      metrics[f'task/{label}/target_rew'] = target.mean()

      policyfn = lambda feat: sample(pol_head(self.feat2tensor(feat), 1))
      _, imgfeat, imgprevact = self.dyn.imagine(
          starts, policyfn, H, training)
      imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
      lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
      lastact = jax.tree.map(lambda x: x[:, None], lastact)
      imgact = concat([imgprevact, lastact], 1)
      assert all(
          x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
      assert all(
          x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
      inp = self.feat2tensor(imgfeat)
      los, _, mets = imag_loss(
          imgact,
          rew_head(inp, 2).pred(),
          self.con(inp, 2).prob(1),
          pol_head(inp, 2),
          val_head(inp, 2),
          slowval(inp, 2),
          retnorm, valnorm, advnorm,
          update=training,
          contdisc=self.config.contdisc,
          horizon=self.config.horizon,
          **self.config.imag_loss)
      losses.update({
          f'task/{label}/{k}': v.mean(1).reshape((B, K))
          for k, v in los.items()})
      metrics.update(prefix(mets, f'task/{label}/imag'))

      if self.config.repval_loss:
        last, term = [obs[k] for k in ('is_last', 'is_terminal')]
        los, _, mets = self._replay_value_loss(
            repfeat, last, term, target, val_head, slowval, valnorm,
            K, training)
        losses.update({f'task/{label}/{k}': v for k, v in los.items()})
        metrics.update(prefix(mets, f'task/{label}/reploss'))

  def task_loss(
      self, repfeat, dyn_entries, dyn_carry, obs, prevact, task_labels,
      training):
    """Updates only detached downstream task heads from a replay posterior."""
    if not self.task_heads['specs']:
      raise RuntimeError('task_loss called without configured task heads.')
    losses = {}
    metrics = {}
    self._add_task_head_losses(
        losses, metrics, repfeat, dyn_entries, dyn_carry, obs, prevact,
        training, task_labels=task_labels)
    assert set(losses) == set(self.task_scales), (
        sorted(losses), sorted(self.task_scales))
    metrics.update({f'loss/{key}': value.mean() for key, value in losses.items()})
    loss = sum(
        value.mean() * self.task_scales[key]
        for key, value in losses.items())
    return loss, metrics

  def _task_reward(
      self, task, obs, prevact, task_labels=None, *, label=None):
    task = str(task)
    if task.startswith('robokitchen_'):
      del label
      key = f'label/reward/{task.removeprefix("robokitchen_")}'
      if not task_labels or key not in task_labels:
        available = sorted(task_labels or {})
        raise KeyError(
            f'Missing learner-only reward label {key!r} for task {task!r}; '
            f'available labels: {available}')
      return f32(task_labels[key])
    if task.startswith('dmc_walker_'):
      if getcfg(self.config, 'walker_task_reward_labels', False):
        key = f'label/reward/{task.removeprefix("dmc_")}'
        if not task_labels or key not in task_labels:
          raise KeyError(f'Missing learner-only reward label {key!r}')
        return f32(task_labels[key])
      return self._walker_reward(task.rsplit('_', 1)[-1], obs)
    if task.startswith('dmc_humanoid_'):
      return self._humanoid_reward(task.rsplit('_', 1)[-1], obs, prevact)
    raise NotImplementedError(f'Unsupported eval task reward: {task}')

  def _walker_reward(self, task, obs):
    height = self._scalar_obs(self._obs_value(obs, 'height'))
    orientations = self._obs_value(obs, 'orientations')
    velocity = self._obs_value(obs, 'velocity')
    standing = self._tolerance(height, bounds=(1.2, jnp.inf), margin=0.6)
    upright = (1 + orientations[..., 0]) / 2
    stand_reward = (3 * standing + upright) / 4
    if task == 'stand':
      return stand_reward
    speed = {'walk': 1.0, 'run': 8.0}[task]
    move = self._tolerance(
        velocity[..., 0], bounds=(speed, jnp.inf), margin=speed / 2,
        sigmoid='linear', value_at_margin=0.5)
    return stand_reward * (5 * move + 1) / 6

  def _humanoid_reward(self, task, obs, prevact):
    head_height = self._scalar_obs(self._obs_value(obs, 'head_height'))
    torso_vertical = self._obs_value(obs, 'torso_vertical')
    com_velocity = self._obs_value(obs, 'com_velocity')
    standing = self._tolerance(head_height, bounds=(1.4, jnp.inf), margin=0.35)
    upright = self._tolerance(
        torso_vertical[..., 2], bounds=(0.9, jnp.inf), margin=1.9,
        sigmoid='linear', value_at_margin=0.0)
    stand_reward = standing * upright
    action = nn.DictConcat(self.act_space, 1)(prevact)
    small_control = self._tolerance(
        action, margin=1.0, sigmoid='quadratic', value_at_margin=0.0).mean(-1)
    small_control = (4 + small_control) / 5
    if task == 'stand':
      dont_move = self._tolerance(com_velocity[..., :2], margin=2.0).mean(-1)
      return small_control * stand_reward * dont_move
    speed = {'walk': 1.0, 'run': 10.0}[task]
    com_speed = jnp.linalg.norm(com_velocity[..., :2], axis=-1)
    move = self._tolerance(
        com_speed, bounds=(speed, jnp.inf), margin=speed,
        sigmoid='linear', value_at_margin=0.0)
    move = (5 * move + 1) / 6
    return small_control * stand_reward * move

  def _obs_value(self, obs, name):
    if name in obs:
      return f32(obs[name])
    matches = [key for key in obs if key.endswith('/' + name)]
    if len(matches) == 1:
      return f32(obs[matches[0]])
    if not matches:
      raise KeyError(
          f'Missing observation {name!r} required for eval task heads. '
          f'Available keys: {sorted(obs)}')
    raise KeyError(
        f'Ambiguous observation {name!r} required for eval task heads: '
        f'{matches}')

  def _scalar_obs(self, value):
    return value[..., 0] if value.ndim > 2 and value.shape[-1] == 1 else value

  def _tolerance(
      self, x, bounds=(0.0, 0.0), margin=0.0, sigmoid='gaussian',
      value_at_margin=0.1):
    x = f32(x)
    lower, upper = [f32(jnp.asarray(v)) for v in bounds]
    in_bounds = (lower <= x) & (x <= upper)
    if margin == 0:
      return f32(in_bounds)
    margin = f32(jnp.asarray(margin))
    d = jnp.where(x < lower, lower - x, x - upper) / margin
    value = self._sigmoid(d, sigmoid, value_at_margin)
    return jnp.where(in_bounds, jnp.ones_like(value), value)

  def _sigmoid(self, x, sigmoid, value_at_1):
    value_at_1 = f32(jnp.asarray(value_at_1))
    if sigmoid == 'gaussian':
      scale = jnp.sqrt(-2 * jnp.log(value_at_1))
      return jnp.exp(-0.5 * (x * scale) ** 2)
    if sigmoid == 'linear':
      scaled = x * (1 - value_at_1)
      return jnp.where(jnp.abs(scaled) < 1, 1 - scaled, 0.0)
    if sigmoid == 'quadratic':
      scaled = x * jnp.sqrt(1 - value_at_1)
      return jnp.where(jnp.abs(scaled) < 1, 1 - scaled ** 2, 0.0)
    raise NotImplementedError(sigmoid)

  def _apply_replay_context(self, carry, data):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    carry = (enc_carry, dyn_carry, dec_carry)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    task_labels = {key: data[key] for key in self._task_label_keys()}
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      return carry, obs, prevact, stepid, task_labels

    K = self.config.replay_context
    nested = elements.tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    rep_carry = (
        self.enc.truncate(lhs(entries[0]), enc_carry),
        self.dyn.truncate(lhs(entries[1]), dyn_carry),
        self.dec.truncate(lhs(entries[2]), dec_carry))
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
    rep_task_labels = {key: rhs(data[key]) for key in self._task_label_keys()}
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_stepid = rhs(stepid)

    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, prevact, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(stepid)),
        (rep_carry, rep_obs, rep_prevact, rep_stepid))
    return carry, obs, prevact, stepid, rep_task_labels

  def _make_opt(
      self,
      lr: float = 4e-5,
      agc: float = 0.3,
      eps: float = 1e-20,
      beta1: float = 0.9,
      beta2: float = 0.999,
      momentum: bool = True,
      nesterov: bool = False,
      wd: float = 0.0,
      wdregex: str = r'/kernel$',
      schedule: str = 'const',
      warmup: int = 1000,
      anneal: int = 0,
  ):
    chain = []
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
    if wd:
      assert not wdregex[0].isnumeric(), wdregex
      pattern = re.compile(wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmask))
    assert anneal > 0 or schedule == 'const'
    if schedule == 'const':
      sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
      sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
      sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
      raise NotImplementedError(schedule)
    if warmup:
      ramp = optax.linear_schedule(0.0, lr, warmup)
      sched = optax.join_schedules([ramp, sched], [warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)


def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
  losses = {}
  metrics = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

  roffset, rscale = retnorm(ret, update)
  adv = (ret - tarval[:, :-1]) / rscale
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale
  logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * sum(ents.values()))
  losses['policy'] = policy_loss

  voffset, vscale = valnorm(ret, update)
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  losses['value'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  ret_normed = (ret - roffset) / rscale
  metrics['adv'] = adv.mean()
  metrics['adv_std'] = adv.std()
  metrics['adv_mag'] = jnp.abs(adv).mean()
  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['val'] = val.mean()
  metrics['tar'] = tar_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
  for k in act:
    metrics[f'ent/{k}'] = ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)

  outs = {}
  outs['ret'] = ret
  return losses, outs, metrics


def repl_loss(
    last, term, rew, bootstrap,
    value, slowvalue, valnorm,
    indices=None,
    update=True,
    slowreg=1.0,
    horizon=333,
    lam=0.95,
):
  losses = {}
  if last.shape[1] < 2:
    zero = jnp.zeros((last.shape[0], 1), jnp.float32)
    return {'repval': zero}, {'ret': zero[:, :0]}, {}

  disc = 1 - 1 / horizon
  bootstrap = sg(bootstrap)
  ret = lambda_return(
      last, term, rew, bootstrap, bootstrap, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  target_length = ret.shape[1]
  if indices is None:
    value_count = target_length
    indices = jnp.broadcast_to(
        jnp.arange(target_length, dtype=i32)[None],
        (last.shape[0], target_length))
  else:
    indices = jnp.asarray(indices, i32)
    value_count = indices.shape[1]
  target = gather_sequence(ret_normed, indices)
  weight = gather_sequence(f32(~last[:, :-1]), indices)
  sampled = weight * (
      value.loss(sg(target)) +
      slowreg * value.loss(sg(slowvalue.pred())))
  losses['repval'] = scatter_sequence(
      sampled, indices, target_length,
      weights=stratified_sequence_weights(target_length, value_count))

  outs = {}
  outs['ret'] = ret
  metrics = {'value_count': jnp.asarray(value_count, f32)}

  return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  live = (1 - f32(term))[:, 1:] * disc
  cont = (1 - f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)
