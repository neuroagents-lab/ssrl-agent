import contextlib
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import tempfile
import threading
import time

import chex
import elements
import embodied
import jax
import jax.experimental.multihost_utils
import jax.numpy as jnp
import ninjax as nj
import numpy as np
P = jax.sharding.PartitionSpec

from agent.adaptation_protocol import STATELESS_COMPONENTS, remap_source_parameters

from . import internal
from . import nets
from . import transform


def _parameter_digest(value):
  value = np.asarray(value)
  digest = hashlib.sha256()
  digest.update(value.dtype.str.encode('ascii'))
  digest.update(repr(tuple(value.shape)).encode('ascii'))
  digest.update(np.ascontiguousarray(value).view(np.uint8))
  return digest.hexdigest()


def _write_transfer_report(path, report):
  path = pathlib.Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  if path.exists() or path.is_symlink():
    if path.is_symlink() or not path.is_file():
      raise FileExistsError(
          f'Refusing unsafe strict transfer report path: {path}')
    with path.open('r', encoding='utf-8') as stream:
      existing = json.load(stream)
    if existing != report:
      raise FileExistsError(
          f'Refusing to replace differing strict transfer report: {path}')
    return False
  descriptor, temporary_name = tempfile.mkstemp(
      prefix=f'.{path.name}.tmp.', dir=path.parent)
  temporary = pathlib.Path(temporary_name)
  try:
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
      json.dump(report, stream, sort_keys=True, indent=2)
      stream.write('\n')
      stream.flush()
      os.fsync(stream.fileno())
    os.chmod(temporary, 0o440)
    os.replace(temporary, path)
    directory = os.open(
        path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
      os.fsync(directory)
    finally:
      os.close(directory)
  finally:
    temporary.unlink(missing_ok=True)
  return True


def _discard_arrays(tree, description):
  if tree is None:
    return
  try:
    jax.tree.map(lambda x: x.delete(), tree)
  except Exception as exc:
    print(f'Could not release {description} arrays after transfer: {exc}')


def _validate_transfer_parameters(source_params, target_params, components):
  """Returns the complete selected key set and content hashes."""
  components = tuple(components)
  selected = lambda key: key.split('/', 1)[0] in components
  source_keys = {key for key in source_params if selected(key)}
  target_keys = {key for key in target_params if selected(key)}
  if source_keys != target_keys:
    missing = sorted(target_keys - source_keys)
    unexpected = sorted(source_keys - target_keys)
    raise ValueError(
        'Strict transfer selected-key coverage mismatch: '
        f'missing={missing}, unexpected={unexpected}')
  if not source_keys:
    raise ValueError(
        f'Strict transfer selected no parameters for {components!r}.')
  missing_components = set(components) - {key.split('/', 1)[0] for key in source_keys}
  if missing_components - STATELESS_COMPONENTS:
    raise ValueError(f'Strict transfer missing components: {sorted(missing_components)}')
  hashes = {}
  for key in sorted(source_keys):
    source_value = np.asarray(source_params[key])
    target_value = target_params[key]
    if tuple(source_value.shape) != tuple(target_value.shape):
      raise ValueError(
          f'Strict transfer shape mismatch for {key!r}: '
          f'{source_value.shape} != {target_value.shape}')
    if np.dtype(source_value.dtype) != np.dtype(target_value.dtype):
      raise ValueError(
          f'Strict transfer dtype mismatch for {key!r}: '
          f'{source_value.dtype} != {target_value.dtype}')
    hashes[key] = _parameter_digest(source_value)
  return source_keys, target_keys, hashes


@dataclasses.dataclass
class Options:

  policy_devices: tuple = (0,)
  train_devices: tuple = (0,)
  policy_mesh: str = '-1,1,1'
  train_mesh: str = '-1,1,1'
  profiler: bool = True
  expect_devices: int = 0
  use_shardmap: bool = False
  enable_policy: bool = True
  donate_policy_params: bool = False
  ckpt_chunksize: int = -1
  precompile: bool = True


class Agent(embodied.Agent):

  def __new__(subcls, obs_space, act_space, config):
    keys = Options.__dataclass_fields__
    options = {k: v for k, v in config.jax.items() if k in keys}
    setup = {k: v for k, v in config.jax.items() if k not in keys}
    jaxcfg = Options(**options)
    internal.setup(**setup)
    model = super().__new__(subcls)
    model.__init__(obs_space, act_space, config)
    outer = super().__new__(Agent)
    outer.__init__(model, obs_space, act_space, config, jaxcfg)
    return outer

  def __init__(self, model, obs_space, act_space, config, jaxcfg):
    assert not any(k.startswith(('log/', 'label/')) for k in obs_space)
    assert 'reset' not in act_space

    self.model = model
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config
    self.jaxcfg = jaxcfg
    self.logdir = elements.Path(config.logdir)

    ext_space = self.model.ext_space  # Extra inputs to train and report.
    elements.print('Observations', color='cyan')
    [elements.print(f'  {k:<16} {v}') for k, v in obs_space.items()]
    elements.print('Actions', color='cyan')
    [elements.print(f'  {k:<16} {v}') for k, v in act_space.items()]
    elements.print('Extras', color='cyan')
    [elements.print(f'  {k:<16} {v}') for k, v in ext_space.items()]
    self.spaces = dict(**obs_space, **act_space, **ext_space)
    assert not (obs_space.keys() & ext_space.keys()), (obs_space, ext_space)
    assert not (act_space.keys() & ext_space.keys()), (act_space, ext_space)

    available = jax.devices()
    elements.print(f'JAX devices ({jax.device_count()}):', available)
    if self.jaxcfg.expect_devices > 0:
      if len(available) != self.jaxcfg.expect_devices:
        print('ALERT: Wrong number of devices')
        while True:
          time.sleep(1)
    assert len(available) == jax.process_count() * jax.local_device_count()
    flatten = lambda x: x.reshape(-1).tolist()
    devices = np.array(available).reshape(
        jax.process_count(), jax.local_device_count())
    self.policy_devices = flatten(devices[:, self.jaxcfg.policy_devices])
    self.train_devices = flatten(devices[:, self.jaxcfg.train_devices])
    print('Policy devices:', ', '.join([str(x) for x in self.policy_devices]))
    print('Train devices: ', ', '.join([str(x) for x in self.train_devices]))

    # d = DP, f = FSDP, t = TP
    self.policy_mesh = internal.mesh(
        self.policy_devices, self.jaxcfg.policy_mesh, ('d', 'f', 't'))
    self.policy_sharded = jax.sharding.NamedSharding(
        self.policy_mesh, P(('d', 'f')))
    self.policy_mirrored = jax.sharding.NamedSharding(self.policy_mesh, P())
    self.train_mesh = internal.mesh(
        self.train_devices, self.jaxcfg.train_mesh, ('d', 'f', 't'))
    self.train_sharded = jax.sharding.NamedSharding(
        self.train_mesh, P(('d', 'f')))
    self.train_mirrored = jax.sharding.NamedSharding(self.train_mesh, P())
    if self.train_mesh.shape['t'] > len(self.jaxcfg.train_devices) or (
        self.policy_mesh.shape['t'] > len(self.jaxcfg.policy_devices)):
          raise NotImplementedError('Inter-node TP is not supported!')
    if self.jaxcfg.use_shardmap:
      assert self.train_mesh.shape['d'] == self.train_mesh.size
      assert self.policy_mesh.shape['d'] == self.policy_mesh.size

    self.partition_rules = getattr(
        self.model, 'partition_rules', ([('.*', P())], []))
    elements.print('Initializing parameters...', color='yellow')
    with self.train_mesh:
      self.params, self.train_params_sharding = self._init_params()
    elements.print('Done initializing!', color='yellow')
    pattern = re.compile(self.model.policy_keys)
    self.policy_keys = [k for k in self.params.keys() if pattern.search(k)]
    assert self.policy_keys, (list(self.params.keys()), self.model.policy_keys)

    self.policy_params_sharding = {
        k: jax.sharding.NamedSharding(self.policy_mesh, v.spec)
        for k, v in self.train_params_sharding.items()
        if k in self.policy_keys}

    shared_kwargs = {'use_shardmap': jaxcfg.use_shardmap}
    tm, ts = self.train_mirrored, self.train_sharded
    pm, ps = self.policy_mirrored, self.policy_sharded
    tp, pp = self.train_params_sharding, self.policy_params_sharding
    _, ar = self.partition_rules
    self._init_train = transform.apply(
        nj.pure(self.model.init_train), self.train_mesh,
        (tp, tm), (ts,), ar, single_output=True, static_argnums=(2,),
        **shared_kwargs)
    self._init_report = transform.apply(
        nj.pure(self.model.init_report), self.train_mesh,
        (tp, tm), (ts,), ar, single_output=True, static_argnums=(2,),
        **shared_kwargs)
    self._init_policy = transform.apply(
        nj.pure(self.model.init_policy), self.policy_mesh,
        (pp, pm), (ps,), ar, single_output=True, static_argnums=(2,),
        **shared_kwargs)
    allo_sharding = {k: v for k, v in tp.items() if k in self.policy_keys}
    dona_sharding = {k: v for k, v in tp.items() if k not in self.policy_keys}
    train_fn = nj.pure(self.model.train)
    self.algorithm2_adaptation = bool(
        getattr(self.model, 'algorithm2_adaptation', False))
    self.donate_policy_params = (
        bool(self.jaxcfg.donate_policy_params) or
        not self.jaxcfg.enable_policy)
    if self.donate_policy_params:
      train_in_shardings = (tp, {}, tm, ts, ts)
    else:
      train_in_shardings = (
          dona_sharding, allo_sharding, tm, ts, ts)
    self._train = transform.apply(
        train_fn, self.train_mesh,
        train_in_shardings, (tp, ts, ts, tm), ar,
        return_params=True, donate_params=True, first_outnums=(3,),
        **shared_kwargs)
    self._train_adaptation = None
    if self.algorithm2_adaptation:
      self._train_adaptation = transform.apply(
          nj.pure(self.model.train_adaptation), self.train_mesh,
          train_in_shardings, (tp, ts, ts, tm), ar,
          return_params=True, donate_params=True, first_outnums=(3,),
          **shared_kwargs)
    self._report = transform.apply(
        nj.pure(self.model.report), self.train_mesh,
        (tp, tm, ts, ts), (ts, tm), ar,
        first_outnums=(1,), **shared_kwargs)
    self._policy = transform.apply(
        nj.pure(self.model.policy), self.policy_mesh,
        (pp, pm, ps, ps), (ps, ps, ps), ar,
        static_argnums=(4,), **shared_kwargs)

    self.policy_lock = threading.Lock()
    self.train_lock = threading.Lock()
    self.n_updates = elements.Counter()
    self.n_batches = elements.Counter()
    self.n_actions = elements.Counter()

    self.pending_outs = None
    self.pending_mets = None
    self.pending_sync = None
    self._train_burst = False
    self._stream_sanitize_warnings = set()

    if self.jaxcfg.enable_policy:
      policy_params = {
          k: self.params[k].copy() for k in self.policy_keys}
      self.policy_params = internal.move(
        policy_params, self.policy_params_sharding)

    self._split = jax.jit(
        lambda xs: jax.tree.map(lambda x: list(x), xs),
        internal.local_sharding(self.policy_sharded),
        internal.local_sharding(self.policy_mirrored))
    self._stack = jax.jit(
        lambda xs: jax.tree.map(
            jnp.stack, xs, is_leaf=lambda x: isinstance(x, list)),
        internal.local_sharding(self.policy_mirrored),
        internal.local_sharding(self.policy_sharded))

    self._ckpt_groups = internal.grouped_ckpt_fns(
        self.params, self.jaxcfg.ckpt_chunksize)
    if self.jaxcfg.precompile:
      elements.print('Compiling train and report...', color='yellow')
      with self.train_mesh:
        self._compile_train()
        print('Train cost analysis:')
        print(self._format_jit_stats(self._train))
        self._compile_report()
        print('Report cost analysis:')
        print(self._format_jit_stats(self._report))
      elements.print('Done compiling!', color='yellow')

  def init_policy(self, batch_size):
    if not self.jaxcfg.enable_policy:
      raise Exception('Policy not available when enable_policy=False')
    batch_size = batch_size * jax.process_count()
    if self.jaxcfg.use_shardmap:
      batch_size = batch_size // self.policy_mesh.size
    return self._split(internal.to_local(self._init_policy(
        self.policy_params, self._seeds(0, self.policy_mirrored), batch_size)))

  def init_policy_device(self, batch_size):
    if not self.jaxcfg.enable_policy:
      raise Exception('Policy not available when enable_policy=False')
    batch_size = batch_size * jax.process_count()
    if self.jaxcfg.use_shardmap:
      batch_size = batch_size // self.policy_mesh.size
    with self.policy_lock:
      return self._init_policy(
          self.policy_params, self._seeds(0, self.policy_mirrored), batch_size)

  def reserve_policy_seed_device(self, amount=1):
    with self.n_actions.lock:
      counter = self.n_actions.value
      self.n_actions.value += int(amount)
    return self.policy_seed_device(counter)

  def policy_seed_device(self, counter):
    """Returns a policy seed without advancing the collection counter."""
    return self._seeds(int(counter), self.policy_mirrored)

  def sync_policy_if_needed(self):
    if not self.jaxcfg.enable_policy:
      return
    with self.policy_lock:
      if self.pending_sync:
        old = self.policy_params
        self.policy_params = self.pending_sync
        jax.tree.map(lambda x: x.delete(), old)
        self.pending_sync = None

  def begin_train_burst(self):
    if not self.jaxcfg.enable_policy:
      return
    with self.train_lock:
      assert not self._train_burst
      self._train_burst = True

  @elements.timer.section('jaxagent_sync_latest')
  def end_train_burst(self):
    if not self.jaxcfg.enable_policy:
      return
    with self.train_lock:
      assert self._train_burst
      try:
        self._queue_policy_snapshot()
      finally:
        self._train_burst = False

  def abort_train_burst(self):
    if not self.jaxcfg.enable_policy:
      return
    with self.train_lock:
      self._train_burst = False

  def _queue_policy_snapshot(self):
    policy_params = {
        k: self.params[k].copy() for k in self.policy_keys}
    snapshot = internal.move(
        policy_params, self.policy_params_sharding)
    with self.policy_lock:
      if self.pending_sync:
        jax.tree.map(lambda x: x.delete(), self.pending_sync)
      self.pending_sync = snapshot

  def policy_device(self, carry, obs, seed, mode='train', params=None):
    if not self.jaxcfg.enable_policy:
      raise Exception('Policy not available when enable_policy=False')
    params = self.policy_params if params is None else params
    return self._policy(params, seed, carry, obs, mode)

  def init_train(self, batch_size):
    batch_size = batch_size * jax.process_count()
    if self.jaxcfg.use_shardmap:
      batch_size  = batch_size // self.train_mesh.size
    return self._init_train(
        self.params, self._seeds(0, self.train_mirrored), batch_size)

  def init_report(self, batch_size):
    batch_size = batch_size * jax.process_count()
    if self.jaxcfg.use_shardmap:
      batch_size  = batch_size // self.train_mesh.size
    return self._init_report(
        self.params, self._seeds(0, self.train_mirrored), batch_size)

  @elements.timer.section('jaxagent_policy')
  def policy(self, carry, obs, mode='train', seed_counter=None):
    if not self.jaxcfg.enable_policy:
      raise Exception('Policy not available when enable_policy=False')
    assert not any(k.startswith(('log/', 'label/')) for k in obs), obs.keys()
    assert sorted(obs.keys()) == sorted(self.obs_space.keys()), (
        sorted(obs.keys()), sorted(self.obs_space.keys()))
    for key, space in self.obs_space.items():
      if np.issubdtype(obs[key].dtype, np.floating):
        assert np.isfinite(obs[key]).all(), (obs[key], key, space)

    with self.policy_lock:
      obs = internal.device_put(obs, self.policy_sharded)
      if seed_counter is None:
        with self.n_actions.lock:
          counter = self.n_actions.value
          self.n_actions.value += 1
      else:
        counter = int(seed_counter)
      seed = self.policy_seed_device(counter)
      carry = internal.to_global(self._stack(carry), self.policy_sharded)

    with self.policy_lock:
      carry, acts, outs = self._policy(
          self.policy_params, seed, carry, obs, mode)

    self.sync_policy_if_needed()

    acts, outs = self._take_outs(internal.fetch_async((acts, outs)))
    carry = self._split(internal.to_local(carry))

    finite = outs.pop('finite', {})
    for key, fin in finite.items():
      assert all(x.all() for x in jax.tree.leaves(fin)), str(finite)
    for key, space in self.act_space.items():
      if space.discrete:
        assert (acts[key] >= 0).all(), (acts[key], key, space)
      else:
        assert np.isfinite(acts[key]).all(), (acts[key], key, space)

    return carry, acts, outs

  @elements.timer.section('jaxagent_train')
  def train(self, carry, data):
    seed = data.pop('seed')
    assert sorted(data.keys()) == sorted(self.spaces.keys()), (
        sorted(data.keys()), sorted(self.spaces.keys()))
    with self.train_lock:
      with elements.timer.section('jit_train'):
        with jax.profiler.StepTraceAnnotation(
            'train', step_num=int(self.n_updates)):
          if self.donate_policy_params:
            self.params, carry, outs, mets = self._train(
                self.params, {}, seed, carry, data)
          else:
            allo = {
                k: v for k, v in self.params.items()
                if k in self.policy_keys}
            dona = {
                k: v for k, v in self.params.items()
                if k not in self.policy_keys}
            self.params, carry, outs, mets = self._train(
                dona, allo, seed, carry, data)
      if self.jaxcfg.enable_policy and not self._train_burst:
        if self.donate_policy_params:
          self._queue_policy_snapshot()
        else:
          with self.policy_lock:
            if not self.pending_sync:
              self.pending_sync = internal.move(
                  {k: allo[k] for k in self.policy_keys},
                  self.policy_params_sharding)
            else:
              jax.tree.map(lambda x: x.delete(), allo)
    self.n_updates.increment()

    return_outs = {}
    if self.pending_outs:
      return_outs = self._take_outs(self.pending_outs)
    self.pending_outs = internal.fetch_async(outs)

    return_mets = {}
    if self.pending_mets:
      return_mets = self._take_outs(self.pending_mets)
    self.pending_mets = internal.fetch_async(mets)

    if self.jaxcfg.profiler:
      outdir, copyto = self.logdir, None
      if str(outdir).startswith(('gs://', '/gcs/', '/cns/')):
        copyto = outdir
        outdir = elements.Path('/tmp/profiler')
        outdir.mkdir()
      if self.n_updates == 100:
        elements.print(f'Start JAX profiler: {str(outdir)}', color='yellow')
        jax.profiler.start_trace(str(outdir))
      if self.n_updates == 120:
        elements.print('Stop JAX profiler', color='yellow')
        jax.profiler.stop_trace()
        if copyto:
          for subdir in elements.Path(outdir).glob('*'):
            subdir.copy(copyto, recursive=True)
          print(f'Copied profiler result {outdir} to {copyto}')

    return carry, return_outs, return_mets

  @elements.timer.section('jaxagent_train_adaptation')
  def train_adaptation(self, carry, data):
    """Runs one offline Algorithm 2 reward and imagined-behavior update."""
    if self._train_adaptation is None:
      raise RuntimeError('Algorithm 2 adaptation is not enabled for this agent.')
    seed = data.pop('seed')
    assert sorted(data.keys()) == sorted(self.spaces.keys()), (
        sorted(data.keys()), sorted(self.spaces.keys()))
    with self.train_lock:
      with elements.timer.section('jit_train_adaptation'):
        with jax.profiler.StepTraceAnnotation(
            'train_adaptation', step_num=int(self.n_updates)):
          if self.donate_policy_params:
            self.params, carry, outs, mets = self._train_adaptation(
                self.params, {}, seed, carry, data)
          else:
            allo = {
                k: v for k, v in self.params.items()
                if k in self.policy_keys}
            dona = {
                k: v for k, v in self.params.items()
                if k not in self.policy_keys}
            self.params, carry, outs, mets = self._train_adaptation(
                dona, allo, seed, carry, data)
      if self.jaxcfg.enable_policy and not self._train_burst:
        if self.donate_policy_params:
          self._queue_policy_snapshot()
        else:
          with self.policy_lock:
            if not self.pending_sync:
              self.pending_sync = internal.move(
                  {k: allo[k] for k in self.policy_keys},
                  self.policy_params_sharding)
            else:
              jax.tree.map(lambda x: x.delete(), allo)
    self.n_updates.increment()

    return_outs = {}
    if self.pending_outs:
      return_outs = self._take_outs(self.pending_outs)
    self.pending_outs = internal.fetch_async(outs)

    return_mets = {}
    if self.pending_mets:
      return_mets = self._take_outs(self.pending_mets)
    self.pending_mets = internal.fetch_async(mets)
    return carry, return_outs, return_mets

  @elements.timer.section('jaxagent_report')
  def report(self, carry, data):
    seed = data.pop('seed')
    assert sorted(data.keys()) == sorted(self.spaces.keys()), (
        sorted(data.keys()), sorted(self.spaces.keys()))
    with self.train_lock:
      carry, mets = self._report(self.params, seed, carry, data)
      mets = self._take_outs(internal.fetch_async(mets))
    mets['params/summary'] = self._summary()
    return carry, mets

  def stream(self, st):
    return embodied.streams.Prefetch(st, self.prepare_stream_batch)

  def prepare_stream_batch(self, data, seed_counter=None):
    """Applies the stream transform synchronously for bounded eval streams."""
    data = self._sanitize_stream_data(data)
    data = internal.device_put(data, self.train_sharded)
    if seed_counter is None:
      with self.n_batches.lock:
        counter = self.n_batches.value
        self.n_batches.value += 1
    else:
      counter = int(seed_counter)
    seed = self._seeds(counter, self.train_mirrored)
    return {**data, 'seed': seed}

  def _sanitize_stream_data(self, data):
    data = dict(data)
    # Replay includes labels for all tasks. Remove labels unused by this model
    # before schema validation; other unexpected keys remain errors.
    data = {
        key: value for key, value in data.items()
        if not key.startswith('label/') or key in self.spaces}
    # Only the replay prefix is needed to reconstruct recurrent state.
    # Observations and actions retain their full T + C sequence.
    context = int(self.config.replay_context)
    if context:
      for key, value in list(data.items()):
        if self._is_replay_context_key(key):
          assert value.ndim >= 2 and value.shape[1] >= context, (
              key, value.shape, context)
          data[key] = value[:, :context]
    for key, value in list(data.items()):
      if not np.issubdtype(value.dtype, np.floating):
        continue
      finite = np.isfinite(value)
      if finite.all():
        continue
      if self._is_sanitizable_stream_key(key):
        if key not in self._stream_sanitize_warnings:
          self._stream_sanitize_warnings.add(key)
          print(
              'Warning: replacing non-finite replay values before '
              f'JAX device transfer: {self._nonfinite_summary(key, value, finite)}')
        data[key] = np.nan_to_num(
            value, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
      else:
        raise ValueError(
            'Non-finite value in replay batch: '
            f'{self._nonfinite_summary(key, value, finite)}')
    if context:
      for key, value in list(data.items()):
        if self._is_replay_context_key(key):
          dtype = np.dtype(self.spaces[key].dtype)
          expected_ndim = 2 + len(self.spaces[key].shape)
          if key == 'dyn/stoch' and value.ndim == expected_ndim + 1:
            value = np.argmax(value, -1).astype(dtype)
            data[key] = value
          if value.dtype != dtype:
            data[key] = self._cast_replay_context(key, value, dtype)
      key = 'dyn/deter'
      if key in data and np.issubdtype(data[key].dtype, np.unsignedinteger):
        decoded = data[key].view(np.dtype(nets.COMPUTE_DTYPE))
        finite = np.isfinite(decoded)
        if not finite.all():
          if key not in self._stream_sanitize_warnings:
            self._stream_sanitize_warnings.add(key)
            print(
                'Warning: replacing non-finite encoded replay values before '
                f'JAX device transfer: {self._nonfinite_summary(key, decoded, finite)}')
          data[key] = np.where(finite, data[key], 0).astype(data[key].dtype)
    return data

  def _cast_replay_context(self, key, value, dtype):
    """Normalizes legacy context arrays to the current compact wire dtype."""
    value = np.asarray(value)
    dtype = np.dtype(dtype)
    if key != 'dyn/deter' or not np.issubdtype(dtype, np.unsignedinteger):
      return value.astype(dtype)

    compute_dtype = np.dtype(nets.COMPUTE_DTYPE)
    if compute_dtype.itemsize != dtype.itemsize:
      raise TypeError((compute_dtype, dtype))
    if np.issubdtype(value.dtype, np.floating):
      decoded = value.astype(compute_dtype)
    elif np.issubdtype(value.dtype, np.unsignedinteger):
      if value.dtype.itemsize == 2:
        decoded = value.view(np.dtype(jnp.bfloat16))
      elif value.dtype.itemsize == 4:
        decoded = value.view(np.float32)
      else:
        raise TypeError(value.dtype)
      decoded = decoded.astype(compute_dtype)
    else:
      raise TypeError(value.dtype)
    return decoded.view(dtype)

  def _is_replay_context_key(self, key):
    return key.startswith(('enc/', 'dyn/', 'dec/'))

  def _is_sanitizable_stream_key(self, key):
    if key == 'reward' or self._is_replay_context_key(key):
      return True
    space = self.obs_space.get(key)
    return space is not None and np.issubdtype(space.dtype, np.floating)

  def _nonfinite_summary(self, key, value, finite):
    bad = ~finite
    count = int(bad.sum())
    indices = np.argwhere(bad)
    first = tuple(int(x) for x in indices[0]) if len(indices) else None
    good = value[finite]
    if good.size:
      minmax = f'min={float(good.min()):.6g}, max={float(good.max()):.6g}'
    else:
      minmax = 'no finite values'
    return (
        f'key={key!r}, shape={value.shape}, dtype={value.dtype}, '
        f'nonfinite={count}, first_index={first}, {minmax}')

  @elements.timer.section('jaxagent_save')
  def save(self):
    with self.train_lock:
      params = {}
      for keys, gather_fn, _ in self._ckpt_groups:
        group = {k: self.params[k] for k in keys}
        params.update(jax.device_get(gather_fn(group)))
    assert params
    counters = {
        'updates': int(self.n_updates),
        'batches': int(self.n_batches),
        'actions': int(self.n_actions),
    }
    data = {'params': params, 'counters': counters}
    return data

  @elements.timer.section('jaxagent_load')
  def load(self, data, regex=None, strict_transfer=None):
    params = data['params']
    assert params

    if strict_transfer is not None:
      if regex:
        raise ValueError('regex and strict_transfer are mutually exclusive.')
      return self._load_strict_transfer(data, strict_transfer)

    with contextlib.ExitStack() as stack:
      stack.enter_context(self.train_lock)
      stack.enter_context(self.policy_lock)

      with self.n_updates.lock:
        self.n_updates.value = int(data['counters']['updates'])
      with self.n_batches.lock:
        # Restoring the update counter repeats prefetched, untrained batches.
        self.n_batches.value = int(data['counters']['updates'])
      with self.n_actions.lock:
        self.n_actions.value = int(data['counters']['actions'])

      if regex:
        params = {k: v for k, v in params.items() if re.match(regex, k)}
        keys = params.keys()
        jax.tree.map(lambda x: x.delete(), [self.params[k] for k in keys])
        params = internal.ckpt_fn({k: self.params[k] for k in keys})[1](
            internal.device_put(params, self.train_mirrored))
        print('Loaded pretrained checkpoint with keys:', list(params.keys()))
        self.params.update(params)
      else:
        chex.assert_trees_all_equal_shapes(self.params, params)
        jax.tree.map(lambda x: x.delete(), self.params)

        loaded = {}
        for keys, _, shard_fn in self._ckpt_groups:
          group = {k: params[k] for k in keys}
          group = shard_fn(internal.device_put(group, self.train_mirrored))
          loaded.update(group)
        self.params = loaded

      self._train_burst = False
      if self.jaxcfg.enable_policy:
        if self.pending_sync:
          jax.tree.map(lambda x: x.delete(), self.pending_sync)
        self.pending_sync = None
        jax.tree.map(lambda x: x.delete(), self.policy_params)
        policy_params = {
            k: self.params[k].copy() for k in self.policy_keys}
        self.policy_params = internal.move(
            policy_params, self.policy_params_sharding)

  def _load_strict_transfer(self, data, options):
    """Atomically import certified model/task state, keeping optimizers fresh."""
    options = dict(options)
    components = tuple(str(x) for x in options.pop('components'))
    backend = str(options.pop('backend', 'unknown'))
    report_path = options.pop('report_path', None)
    source = options.pop('source', {})
    mapping = options.pop('source_components', {name: name for name in components})
    if options:
      raise ValueError(
          f'Unknown strict transfer options: {sorted(options)}')
    if not components or len(set(components)) != len(components):
      raise ValueError(
          f'Strict transfer components must be unique and non-empty: '
          f'{components!r}')
    if any(not re.fullmatch(r'[A-Za-z0-9_]+', x) for x in components):
      raise ValueError(f'Unsafe strict transfer component: {components!r}')
    if any(int(counter) != 0 for counter in (
        self.n_updates, self.n_batches, self.n_actions)):
      raise RuntimeError(
          'Strict transfer requires a fresh agent with zero counters.')

    source_params = data.get('params')
    if not isinstance(source_params, dict) or not source_params:
      raise ValueError('Strict transfer source has no parameter mapping.')
    if (set(mapping) != set(components) or len(set(mapping.values())) != len(mapping)
        or any(not isinstance(x, str) or not re.fullmatch(r'[A-Za-z0-9_]+', x)
               for x in mapping.values())):
      raise ValueError('Invalid strict transfer source component mapping.')
    original_source_params = source_params
    source_params = remap_source_parameters(source_params, mapping)
    source_keys, target_keys, hashes = _validate_transfer_parameters(
        source_params, self.params, components)

    # Device placement is also staged before touching the live parameter tree.
    # This makes missing devices, invalid arrays, or sharding failures atomic.
    selected_target = {key: self.params[key] for key in sorted(target_keys)}
    _, shard_fn = internal.ckpt_fn(selected_target)
    staged = shard_fn(internal.device_put(
        {key: source_params[key] for key in sorted(source_keys)},
        self.train_mirrored))
    if set(staged) != target_keys:
      raise RuntimeError(
          'Strict transfer sharding changed selected-key coverage: '
          f'{sorted(staged)} != {sorted(target_keys)}')

    loaded_source_keys = {
        mapping[key.split('/', 1)[0]] + '/' + key.split('/', 1)[1]
        for key in source_keys}
    excluded = sorted(set(original_source_params) - loaded_source_keys)
    target_excluded = sorted(set(self.params) - target_keys)
    aggregate = hashlib.sha256()
    for key in sorted(hashes):
      aggregate.update(key.encode('utf-8'))
      aggregate.update(bytes.fromhex(hashes[key]))
    report = {
        'schema_version': 1,
        'mode': ('strict_pretrained_task_transfer' if 'pol' in components
                 else 'strict_representation_transfer'),
        'source_components': mapping,
        'loaded_source_keys': sorted(loaded_source_keys),
        'backend': backend,
        'components': list(components),
        'loaded_keys': sorted(source_keys),
        'excluded_keys': excluded,
        'excluded_source_keys': excluded,
        'excluded_target_keys': target_excluded,
        'parameter_sha256': hashes,
        'loaded_sha256': aggregate.hexdigest(),
        'source': dict(source),
        'counters_restored': False,
        'fresh_counters': {'updates': 0, 'batches': 0, 'actions': 0},
    }

    staged_policy = None
    old = None
    old_policy = None
    old_pending = None
    try:
      with contextlib.ExitStack() as stack:
        stack.enter_context(self.train_lock)
        stack.enter_context(self.policy_lock)
        if any(int(counter) != 0 for counter in (
            self.n_updates, self.n_batches, self.n_actions)):
          raise RuntimeError(
              'Agent counters changed while strict transfer was staged.')

        # Build the complete post-transfer policy snapshot before changing the
        # live training or policy trees. Device movement and report I/O are the
        # remaining fallible operations and must therefore precede the swap.
        if self.jaxcfg.enable_policy:
          policy_params = {
              key: (staged[key] if key in staged else self.params[key]).copy()
              for key in self.policy_keys}
          staged_policy = internal.move(
              policy_params, self.policy_params_sharding)
        if report_path:
          _write_transfer_report(report_path, report)

        old = {key: self.params[key] for key in sorted(target_keys)}
        old_policy = self.policy_params if self.jaxcfg.enable_policy else None
        old_pending = self.pending_sync
        self.params.update(staged)
        if self.jaxcfg.enable_policy:
          self.policy_params = staged_policy
          staged_policy = None
        self.pending_sync = None
        self._train_burst = False
    except Exception:
      _discard_arrays(staged, 'staged representation')
      _discard_arrays(staged_policy, 'staged policy')
      raise

    # Cleanup errors must not invalidate the committed parameter swap.
    _discard_arrays(old, 'replaced representation')
    _discard_arrays(old_policy, 'replaced policy')
    _discard_arrays(old_pending, 'pending policy synchronization')
    print(
        'Strict transfer loaded components:',
        ', '.join(components))
    return report

  def _take_outs(self, outs):
    outs = jax.tree.map(lambda x: x.__array__(), outs)
    outs = jax.tree.map(
        lambda x: np.float32(x) if x.dtype == jnp.bfloat16 else x, outs)
    return outs

  def _seeds(self, counter, sharding):
    rng = np.random.default_rng(seed=[self.config.seed, int(counter)])
    seeds = rng.integers(0, np.iinfo(np.uint32).max, (2,), np.uint32)
    return internal.device_put(seeds, sharding)

  def _init_params(self):
    B = min(self.config.batch_size, len(self.jaxcfg.train_devices))
    GB = B * jax.process_count()
    T = self.config.batch_length
    C = self.config.replay_context
    tm, ts = self.train_mirrored, self.train_sharded
    us = self.jaxcfg.use_shardmap

    with jax._src.config.explicit_device_get_scope():
      seed = jax.device_put(np.array([self.config.seed, 0], np.uint32), tm)
    data = internal.device_put(self._stream_zeros(B, T, C), ts)
    pr, ar = self.partition_rules

    params, params_sharding = transform.init(
        self.model.init_train, self.train_mesh,
        ({}, self.train_mirrored),
        param_partition_rules=pr,
        act_partition_rules=ar,
        static_argnums=(2,),
        dummy_inputs=({}, seed, GB),
        print_partition=(len(pr) >= 2),
    )
    carry = transform.apply(
        nj.pure(self.model.init_train), self.train_mesh,
        (params_sharding, tm), (ts,), single_output=True,
        static_argnums=(2,), use_shardmap=us)(
            params, seed, GB // self.train_mesh.size if us else GB)
    params, params_sharding = transform.init(
        self.model.train, self.train_mesh,
        (params_sharding, tm, ts, ts),
        param_partition_rules=pr,
        act_partition_rules=ar,
        dummy_inputs=(params, seed, carry, data),
        print_partition=(len(pr) >= 2),
    )
    if bool(getattr(self.model, 'algorithm2_adaptation', False)):
      params, params_sharding = transform.init(
          self.model.train_adaptation, self.train_mesh,
          (params_sharding, tm, ts, ts),
          param_partition_rules=pr,
          act_partition_rules=ar,
          dummy_inputs=(params, seed, carry, data),
          print_partition=(len(pr) >= 2),
      )
    return params, params_sharding

  def _compile_train(self):
    B = self.config.batch_size
    T = self.config.batch_length
    C = self.config.replay_context
    data = self._stream_zeros(B, T, C)
    data = internal.device_put(data, self.train_sharded)
    seed = self._seeds(0, self.train_mirrored)
    carry = self.init_train(B)
    if self.donate_policy_params:
      train_args = (self.params, {}, seed, carry, data)
    else:
      allo = {k: v for k, v in self.params.items() if k in self.policy_keys}
      dona = {k: v for k, v in self.params.items() if k not in self.policy_keys}
      train_args = (dona, allo, seed, carry, data)
    self._train = self._train.lower(*train_args).compile()

  def _compile_report(self):
    B = self.config.batch_size
    T = self.config.report_length
    C = self.config.replay_context
    data = self._stream_zeros(B, T, C)
    data = internal.device_put(data, self.train_sharded)
    seed = self._seeds(0, self.train_mirrored)
    carry = self.init_report(B)
    self._report = self._report.lower(self.params, seed, carry, data).compile()

  def _summary(self):
    lines = []
    for k, v in self.params.items():
      lines.append(f'{k:<40} {v.dtype} {v.size} {v.shape}')
    return '\n'.join(lines)

  def _zeros(self, spaces, batch_shape):
    data = {k: np.zeros(v.shape, v.dtype) for k, v in spaces.items()}
    for dim in reversed(batch_shape):
      data = {k: np.repeat(v[None], dim, axis=0) for k, v in data.items()}
    return data

  def _stream_zeros(self, batch_size, length, context):
    data = self._zeros(self.spaces, (batch_size, length + context))
    if context:
      for key, value in list(data.items()):
        if self._is_replay_context_key(key):
          data[key] = value[:, :context]
    return data

  def _format_jit_stats(self, compiled):
    try:
      cost = compiled.cost_analysis()
      mem = compiled.memory_analysis()
      lines = []
      lines.append(f"FLOPS:            {cost[0]['flops']:.1e}")
      lines.append(f"Memory (temp):    {mem.temp_size_in_bytes:.1e}")
      lines.append(f"Memory (inputs):  {mem.argument_size_in_bytes:.1e}")
      lines.append(f"Memory (outputs): {mem.output_size_in_bytes:.1e}")
      alias_size = getattr(mem, 'alias_size_in_bytes', None)
      if alias_size is not None:
        lines.append(f"Memory (aliases): {alias_size:.1e}")
      lines.append(f"Memory (code):    {mem.generated_code_size_in_bytes:.1e}")
      return ''.join(f'  {line}\n' for line in lines)
    except (TypeError, AttributeError, KeyError):
      return 'No available'

def init(fun, **jit_kwargs):
  if not getattr(fun, '_is_pure', False):
    fun = nj.pure(fun)
  def wrapper(*args, **kwargs):
    state, out = fun(*args, create=True, modify=True, ignore=True, **kwargs)
    del out
    return state, ()
  return wrapper
