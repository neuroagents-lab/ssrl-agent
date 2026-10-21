import os
import stat
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

import elements
import numpy as np

from . import chunk as chunklib
from . import limiters
from . import selectors


_INVENTORY_FORMAT = 'embodied-replay-chunks-v1'


def _chunk_paths(directory):
  paths = []
  for pattern in chunklib.CHUNK_GLOBS:
    paths.extend(directory.glob(pattern))
  return paths


class Replay:

  def __init__(
      self, length, capacity=None, directory=None, chunksize=1024,
      online=False, selector=None, save_wait=False, name='unnamed', seed=0,
      replay_context_dtype=None):

    self.length = length
    self.capacity = capacity
    self.chunksize = chunksize
    self.name = name
    self.replay_context_dtype = (
        str(replay_context_dtype) if replay_context_dtype else None)
    if self.replay_context_dtype not in (None, 'bfloat16', 'float32'):
      raise ValueError(
          'replay_context_dtype must be bfloat16, float32, or None; got '
          f'{self.replay_context_dtype!r}')

    self.sampler = selector or selectors.Uniform(seed)

    self.chunks = {}
    self.refs = {}
    self.refs_lock = threading.RLock()

    self.items = {}
    self.fifo = deque()
    self.itemid = 0

    self.current = {}
    self.streams = defaultdict(deque)
    self.rwlock = elements.RWLock()

    self.online = online
    if online:
      self.lengths = defaultdict(int)
      self.queue = deque()

    self.saved = set()
    self.save_promises = {}
    self.save_filenames = {}
    self.integrity_cache = {}
    self.disk_lock = threading.RLock()
    self.checkpoint_lock = threading.RLock()
    self._pending_inventory = None
    self._committed_inventory = None
    if directory:
      self.directory = elements.Path(directory)
      self.directory.mkdir()
      self.workers = ThreadPoolExecutor(16, 'replay_saver')
    else:
      self.directory = None
    self.save_wait = save_wait

    self.metrics = {'samples': 0, 'inserts': 0, 'updates': 0}

  def __len__(self):
    return len(self.items)

  def stats(self):
    ratio = lambda x, y: x / y if y else np.nan
    m = self.metrics
    chunk_nbytes = sum(x.nbytes for x in list(self.chunks.values()))
    stats = {
        'items': len(self.items),
        'chunks': len(self.chunks),
        'streams': len(self.streams),
        'ram_gb': chunk_nbytes / (1024 ** 3),
        'inserts': m['inserts'],
        'samples': m['samples'],
        'updates': m['updates'],
        'replay_ratio': ratio(self.length * m['samples'], m['inserts']),
    }
    for key in self.metrics:
      self.metrics[key] = 0
    return stats

  @elements.timer.section('replay_add')
  def add(self, step, worker=0):
    step = {k: v for k, v in step.items() if not k.startswith('log/')}
    with self.rwlock.reading:
      step = {k: np.asarray(v) for k, v in step.items()}
      self._add_sequence({k: v[None] for k, v in step.items()}, worker)

  @elements.timer.section('replay_add_batch')
  def add_batch(self, steps, workers=None):
    steps = {k: v for k, v in steps.items() if not k.startswith('log/')}
    if not steps:
      return
    steps = {k: np.asarray(v) for k, v in steps.items()}
    shape = next(iter(steps.values())).shape[:2]
    assert len(shape) == 2, shape
    assert all(v.shape[:2] == shape for v in steps.values()), (
        shape, {k: v.shape for k, v in steps.items()})
    length, num_workers = shape
    workers = list(range(num_workers) if workers is None else workers)
    assert len(workers) == num_workers
    with self.rwlock.reading:
      for worker_index, worker in enumerate(workers):
        seq = {k: v[:, worker_index] for k, v in steps.items()}
        self._add_sequence(seq, worker)

  def _ensure_worker(self, worker):
    if worker not in self.current:
      chunk = chunklib.Chunk(self.chunksize)
      with self.refs_lock:
        self.refs[chunk.uuid] = 1
      self.chunks[chunk.uuid] = chunk
      self.current[worker] = (chunk.uuid, 0)

  def _add_sequence(self, sequence, worker):
    length = len(next(iter(sequence.values())))
    offset = 0
    while offset < length:
      self._ensure_worker(worker)
      chunkid, index = self.current[worker]
      stream = self.streams[worker]
      chunk = self.chunks[chunkid]
      assert chunk.length == index, (chunk.length, index)
      amount = min(length - offset, chunk.size - index)
      part = {k: v[offset: offset + amount] for k, v in sequence.items()}
      part['stepid'] = np.stack([
          np.frombuffer(bytes(chunkid) + (index + i).to_bytes(4, 'big'), np.uint8)
          for i in range(amount)
      ])
      start = chunk.append_many(part)
      assert start == index, (start, index)
      assert chunk.length == index + amount, (chunk.length, index + amount)
      with self.refs_lock:
        self.refs[chunkid] += amount

      for i in range(amount):
        stream.append((chunkid, index + i))
        if len(stream) >= self.length:
          # Increment is not thread safe thus inaccurate but faster than locking.
          self.metrics['inserts'] += 1
          inserted_chunkid, inserted_index = stream.popleft()
          self._insert(inserted_chunkid, inserted_index)

          if self.online and self.lengths[worker] % self.length == 0:
            self.queue.append((inserted_chunkid, inserted_index))

        if self.online:
          self.lengths[worker] += 1

      index += amount
      if index < chunk.size:
        self.current[worker] = (chunkid, index)
      else:
        self._complete(chunk, worker)
      assert len(self.streams) == len(self.current)
      offset += amount

  @elements.timer.section('replay_sample')
  def sample(self, batch, mode='train'):
    message = f'Replay buffer {self.name} is empty'
    limiters.wait(lambda: len(self.sampler), message)
    seqs, is_online = zip(*[self._sample(mode) for _ in range(batch)])
    data = self._assemble_batch(seqs, 0, self.length)
    data = self._annotate_batch(data, is_online, True)
    return data

  @elements.timer.section('replay_update')
  def update(self, data):
    stepid = data.pop('stepid')
    priority = data.pop('priority', None)
    assert stepid.ndim == 3, stepid.shape
    self.metrics['updates'] += int(np.prod(stepid.shape[:-1]))
    # Serialize replay updates with asynchronous checkpoint writes so saved
    # arrays cannot change during serialization.
    with self.rwlock.reading:
      if priority is not None:
        assert priority.ndim == 2, priority.shape
        self.sampler.prioritize(
            stepid.reshape((-1, stepid.shape[-1])),
            priority.flatten())
      if data:
        for i, item_stepid in enumerate(stepid):
          item_stepid = item_stepid[0].tobytes()
          chunkid = elements.UUID(item_stepid[:-4])
          index = int.from_bytes(item_stepid[-4:], 'big')
          values = {k: v[i] for k, v in data.items()}
          try:
            self._setseq(chunkid, index, values)
          except KeyError:
            pass

  def _sample(self, mode):
    assert mode in ('train', 'report', 'eval'), mode
    if mode == 'train':
      # Increment is not thread safe thus inaccurate but faster than locking.
      self.metrics['samples'] += 1
    while True:
      try:
        if self.online and self.queue and mode == 'train':
          chunkid, index = self.queue.popleft()
          is_online = True
        else:
          with elements.timer.section('sample'):
            itemid = self.sampler()
          chunkid, index = self.items[itemid]
          is_online = False
        seq = self._getseq(chunkid, index, concat=False)
        return seq, is_online
      except KeyError:
        continue

  def _insert(self, chunkid, index):
    while self.capacity and len(self.items) >= self.capacity:
      self._remove()
    itemid = self.itemid
    self.itemid += 1
    self.items[itemid] = (chunkid, index)
    stepids = self._getseq(chunkid, index, ['stepid'])['stepid']
    self.sampler[itemid] = stepids
    self.fifo.append(itemid)

  def _remove(self):
    itemid = self.fifo.popleft()
    del self.sampler[itemid]
    chunkid, index = self.items.pop(itemid)
    with self.refs_lock:
      self.refs[chunkid] -= 1
      if self.refs[chunkid] < 1:
        del self.refs[chunkid]
        chunk = self.chunks.pop(chunkid)
        if chunk.succ in self.refs:
          self.refs[chunk.succ] -= 1

  def _getseq(self, chunkid, index, keys=None, concat=True):
    chunk = self.chunks[chunkid]
    available = chunk.length - index
    if available >= self.length:
      with elements.timer.section('get_slice'):
        seq = chunk.slice(index, self.length)
        if not concat:
          seq = {k: [v] for k, v in seq.items()}
        return seq
    else:
      with elements.timer.section('get_compose'):
        parts = [chunk.slice(index, available)]
        remaining = self.length - available
        while remaining > 0:
          chunk = self.chunks[chunk.succ]
          used = min(remaining, chunk.length)
          parts.append(chunk.slice(0, used))
          remaining -= used
        seq = {k: [p[k] for p in parts] for k in keys or parts[0].keys()}
        if concat:
          seq = {k: np.concatenate(v, 0) for k, v in seq.items()}
        return seq

  def _setseq(self, chunkid, index, values):
    length = len(next(iter(values.values())))
    chunk = self.chunks[chunkid]
    available = chunk.length - index
    if available >= length:
      with elements.timer.section('set_slice'):
        return chunk.update(index, length, values)
    else:
      with elements.timer.section('set_compose'):
        part = {k: v[:available] for k, v in values.items()}
        values = {k: v[available:] for k, v in values.items()}
        chunk.update(index, available, part)
        remaining = length - available
        while remaining > 0:
          chunk = self.chunks[chunk.succ]
          used = min(remaining, chunk.length)
          part = {k: v[:used] for k, v in values.items()}
          values = {k: v[used:] for k, v in values.items()}
          chunk.update(0, used, part)
          remaining -= used

  @elements.timer.section('assemble_batch')
  def _assemble_batch(self, seqs, start, stop):
    shape = (len(seqs), stop - start)
    data = {
        key: np.empty((*shape, *parts[0].shape[1:]), parts[0].dtype)
        for key, parts in seqs[0].items()}
    for n, seq in enumerate(seqs):
      st, dt = 0, 0  # Source and destination time index.
      for p in range(len(seq['stepid'])):
        partlen = len(seq['stepid'][p])
        if start < st + partlen:
          part_start = max(0, start - st)
          part_stop = min(stop - st, partlen)
          num = part_stop - part_start
          for k in data.keys():
            data[k][n, dt: dt + num] = seq[k][p][part_start: part_stop]
          dt += num
        st += partlen
        if st >= stop:
          break
    return data

  @elements.timer.section('annotate_batch')
  def _annotate_batch(self, data, is_online, is_first):
    data = data.copy()
    if 'is_first' in data:
      if is_first:
        data['is_first'] = data['is_first'].copy()
        data['is_first'][:, 0] = True
      if 'is_last' in data:
        # Make sure that abandoned episodes have is_last set.
        next_is_first = np.roll(data['is_first'], shift=-1, axis=1)
        next_is_first[:, -1] = False
        data['is_last'] = data['is_last'] | next_is_first
    return data

  @elements.timer.section('replay_save')
  def save(self):
    if not self.directory:
      return None
    with self.disk_lock:
      with self.rwlock.writing:
        for worker, (chunkid, _) in self.current.items():
          chunk = self.chunks[chunkid]
          if chunk.length > 0:
            self._complete(chunk, worker)
        for chunk in self.chunks.values():
          if chunk.length > 0 and chunk.uuid not in self.saved:
            self.saved.add(chunk.uuid)
            self.save_filenames[chunk.uuid] = chunk.filename
            try:
              promise = self.workers.submit(chunk.save, self.directory)
            except Exception:
              self.saved.discard(chunk.uuid)
              self.save_filenames.pop(chunk.uuid, None)
              raise
            self.save_promises[chunk.uuid] = promise

        # Every referenced replay file must be durable before publishing the
        # checkpoint completion marker, regardless of the save_wait option.
        wrote_chunks = False
        failures = []
        for chunk in self.chunks.values():
          promise = self.save_promises.get(chunk.uuid)
          if promise is not None:
            try:
              size, digest, signature = promise.result()
            except Exception as exc:
              # Reset failed saves so subsequent checkpoints can retry the chunk.
              self.save_promises.pop(chunk.uuid, None)
              self.save_filenames.pop(chunk.uuid, None)
              self.saved.discard(chunk.uuid)
              failures.append(exc)
              continue
            self.integrity_cache[chunk.filename] = {
                'size': size, 'sha256': digest, 'signature': signature}
            self.save_promises.pop(chunk.uuid, None)
            wrote_chunks = True
        # Flush directory entries even when retrying after all writers finished.
        chunklib.fsync_directory(self.directory)
        if failures:
          raise failures[0]

        entries = []
        chunks = sorted(
            (x for x in self.chunks.values() if x.length > 0),
            key=lambda x: x.filename)
        for chunk in chunks:
          entries.append(self._inventory_entry(chunk.filename))
        inventory = {'format': _INVENTORY_FORMAT, 'chunks': entries}
        with self.checkpoint_lock:
          self._pending_inventory = self._copy_inventory(inventory)
        return inventory

  @elements.timer.section('replay_load')
  def load(self, data=None, directory=None, amount=None):
    directory = directory or self.directory
    amount = amount or self.capacity or np.inf
    if not directory:
      return
    with self.disk_lock:
      return self._load(data, directory, amount)

  @elements.timer.section('replay_transfer_load')
  def load_transfer(
      self, data, directory, schema, amount=None,
      drop_keys=('skill', 'skill_start'), clean_uncommitted=True,
      reward_key=None):
    """Authenticates and normalizes a parent replay into a fresh child.

    The parent files are only read. Loaded chunks retain their original UUIDs
    and step IDs, but own filtered arrays and are marked unsaved so the first
    child checkpoint writes a complete, independent replay generation. An
    optional Walker reward label replaces reward after source authentication.
    """
    if self.items or self.chunks or self.current:
      raise RuntimeError('Replay transfer requires a fresh empty child replay.')
    if not self.directory:
      raise RuntimeError('Replay transfer requires a directory-backed child.')
    source_directory = os.path.realpath(os.fspath(directory))
    child_directory = os.path.realpath(os.fspath(self.directory))
    if os.path.samefile(source_directory, child_directory):
      raise RuntimeError(
          'Replay transfer parent and child directories must be distinct: '
          f'{source_directory}')
    cleanup = {'removed': 0, 'bytes': 0, 'errors': 0}
    if clean_uncommitted:
      # Remove unpublished child files before reusing the parent chunk IDs.
      cleanup = self.checkpoint_gc([])
      if cleanup.get('errors') or cleanup.get('cancelled'):
        raise RuntimeError(
            f'Could not clear uncommitted child replay files: {cleanup}')
    amount = amount or self.capacity or np.inf
    with self.disk_lock:
      result = self._load_transfer_inventory(
          data, directory, schema, amount, tuple(drop_keys), reward_key)
    result['uncommitted_files_removed'] = int(cleanup.get('removed', 0))
    return result

  def _load(self, data, directory, amount):
    if data is not None:
      return self._load_inventory(data, directory, amount)

    # Checkpoints without a replay inventory load all chunks in the directory.
    with self.checkpoint_lock:
      self._pending_inventory = None
      self._committed_inventory = None
    revsorted = lambda x: list(reversed(sorted(list(x))))
    directory = elements.Path(directory)
    names_loaded = revsorted(x.filename for x in list(self.chunks.values()))
    names_ondisk = revsorted(x.name for x in _chunk_paths(directory))
    names_ondisk = [x for x in names_ondisk if x not in names_loaded]
    if not names_ondisk:
      return

    numitems = self._numitems(names_loaded + names_ondisk)
    uuids = [
        elements.UUID(chunklib.filename_stem(x).split('-')[1])
        for x in names_ondisk]
    total = 0
    numchunks = 0
    for uuid in uuids:
      numchunks += 1
      total += numitems[uuid]
      if total >= amount:
        break

    def load(filename):
      chunk = chunklib.Chunk.load(filename, error='none')
      if chunk:
        self._compact_loaded_replay_context(chunk)
      return chunk
    filenames = [directory / x for x in names_ondisk[:numchunks]]

    with ThreadPoolExecutor(16, 'replay_loader') as pool:
      chunks = [x for x in pool.map(load, filenames) if x]

    numitems = self._numitems(chunks)

    with self.rwlock.writing:
      self.saved.update([chunk.uuid for chunk in chunks])
      self.save_filenames.update({
          chunk.uuid: chunk.filename for chunk in chunks})
      with self.refs_lock:
        for chunk in chunks:
          self.chunks[chunk.uuid] = chunk
          self.refs[chunk.uuid] = 0
        for chunk in reversed(chunks):
          amount = numitems[chunk.uuid]
          self.refs[chunk.uuid] += amount
          if chunk.succ in self.refs:
            self.refs[chunk.succ] += 1
          for index in range(amount):
            self._insert(chunk.uuid, index)

  def _inventory_entry(self, name):
    filename = self.directory / name
    signature = chunklib.regular_file_stat(filename)
    cached = self.integrity_cache.get(name)
    if cached and cached['signature'] == signature:
      size, digest = cached['size'], cached['sha256']
    else:
      size, digest, signature = chunklib.file_integrity(filename)
      if cached and (size, digest) != (cached['size'], cached['sha256']):
        raise RuntimeError(
            f'Replay chunk changed after being saved: {filename}')
      self.integrity_cache[name] = {
          'size': size, 'sha256': digest, 'signature': signature}
    return {'filename': name, 'size': size, 'sha256': digest}

  def _load_inventory(self, data, directory, amount):
    entries = self._validate_inventory(data)
    directory = elements.Path(directory)

    # Load and authenticate every bound file before changing in-memory state.
    # This also makes a missing old chunk fail even when amount would otherwise
    # trim it from memory.
    def load(entry):
      filename = directory / entry['filename']
      result = chunklib.Chunk.load(
          filename, error='raise', expected_size=entry['size'],
          expected_sha256=entry['sha256'], return_integrity=True)
      chunk, integrity = result
      self._compact_loaded_replay_context(chunk)
      return chunk, integrity

    with ThreadPoolExecutor(16, 'replay_loader') as pool:
      loaded = list(pool.map(load, entries))
    chunks = [chunk for chunk, _ in loaded]

    by_name = {chunk.filename: chunk for chunk in chunks}
    if set(by_name) != {x['filename'] for x in entries}:
      raise RuntimeError('Replay inventory filenames do not match chunk data.')
    if len({chunk.uuid for chunk in chunks}) != len(chunks):
      raise RuntimeError('Replay inventory contains duplicate chunk UUIDs.')

    integrity = {
        chunk.filename: values for (chunk, values) in loaded}
    for entry in entries:
      size, digest, signature = integrity[entry['filename']]
      if (size, digest) != (entry['size'], entry['sha256']):
        raise RuntimeError(
            f'Replay chunk integrity changed for {entry["filename"]}.')
      self.integrity_cache[entry['filename']] = {
          'size': size, 'sha256': digest, 'signature': signature}

    chunks = list(reversed(sorted(chunks)))
    if chunks:
      numitems = self._numitems(chunks)
      total = 0
      numchunks = 0
      for chunk in chunks:
        numchunks += 1
        total += numitems[chunk.uuid]
        if total >= amount:
          break
      chunks = chunks[:numchunks]
      numitems = self._numitems(chunks)
    else:
      numitems = {}

    with self.rwlock.writing:
      self._clear_for_bound_load()
      self.saved.update(chunk.uuid for chunk in chunks)
      self.save_filenames.update({
          chunk.uuid: chunk.filename for chunk in chunks})
      with self.refs_lock:
        for chunk in chunks:
          self.chunks[chunk.uuid] = chunk
          self.refs[chunk.uuid] = 0
        for chunk in reversed(chunks):
          count = numitems[chunk.uuid]
          self.refs[chunk.uuid] += count
          if chunk.succ in self.refs:
            self.refs[chunk.succ] += 1
          for index in range(count):
            self._insert(chunk.uuid, index)
    inventory = {'format': _INVENTORY_FORMAT, 'chunks': entries}
    with self.checkpoint_lock:
      self._pending_inventory = self._copy_inventory(inventory)

  def _load_transfer_inventory(
      self, data, directory, schema, amount, drop_keys, reward_key=None):
    entries = self._validate_inventory(data)
    directory = elements.Path(directory)
    schema = dict(schema)
    expected = set(schema)
    if 'stepid' in expected:
      raise ValueError('Replay transfer schema must not contain stepid.')
    expected.add('stepid')
    drop_keys = set(drop_keys)
    if reward_key is not None:
      from agent.walker_replay import (
          WALKER_REWARD_KEYS, normalize_walker_replay_arrays)
      if reward_key not in WALKER_REWARD_KEYS:
        raise ValueError(f'Unsupported Walker target reward label: {reward_key!r}')
      drop_keys.update(WALKER_REWARD_KEYS)
    if expected & drop_keys:
      raise ValueError(
          f'Replay transfer cannot drop target fields: '
          f'{sorted(expected & drop_keys)}')

    def load(entry):
      filename = directory / entry['filename']
      chunk, integrity = chunklib.Chunk.load(
          filename, error='raise', expected_size=entry['size'],
          expected_sha256=entry['sha256'], return_integrity=True)
      self._compact_loaded_replay_context(chunk)
      source_data = chunk.data
      if reward_key is not None:
        source_data = normalize_walker_replay_arrays(source_data, reward_key)
      source_keys = set(source_data)
      unexpected = source_keys - expected
      if unexpected - drop_keys:
        raise ValueError(
            f'Parent replay chunk {entry["filename"]} has unsupported '
            f'fields: {sorted(unexpected - drop_keys)}')
      missing = expected - source_keys
      if missing:
        raise ValueError(
            f'Parent replay chunk {entry["filename"]} is missing target '
            f'fields: {sorted(missing)}')
      normalized = {
          key: np.array(value, copy=True, order='C')
          for key, value in source_data.items() if key in expected}
      for key, space in schema.items():
        value = normalized[key]
        target_shape = tuple(getattr(space, 'shape', ()))
        target_dtype = np.dtype(getattr(space, 'dtype', value.dtype))
        if tuple(value.shape[1:]) != target_shape:
          raise ValueError(
              f'Parent replay shape mismatch for {key!r}: '
              f'{value.shape[1:]} != {target_shape}')
        if value.dtype != target_dtype:
          raise ValueError(
              f'Parent replay dtype mismatch for {key!r}: '
              f'{value.dtype} != {target_dtype}')
      stepids = normalized['stepid']
      if stepids.dtype != np.uint8 or stepids.shape != (chunk.length, 20):
        raise ValueError(
            f'Invalid replay stepid wire format in {entry["filename"]}: '
            f'{stepids.shape} {stepids.dtype}')
      expected_stepids = np.stack([
          np.frombuffer(
              bytes(chunk.uuid) + index.to_bytes(4, 'big'), np.uint8)
          for index in range(chunk.length)])
      if not np.array_equal(stepids, expected_stepids):
        raise ValueError(
            f'Parent replay step identity mismatch in {entry["filename"]}.')
      chunk.data = normalized
      chunk.saved = False
      return chunk, integrity

    # Authenticate and normalize the complete checkpoint binding before the
    # first mutation of child replay state.
    with ThreadPoolExecutor(16, 'replay_transfer_loader') as pool:
      loaded = list(pool.map(load, entries))
    chunks = [chunk for chunk, _ in loaded]
    by_name = {chunk.filename: chunk for chunk in chunks}
    if set(by_name) != {entry['filename'] for entry in entries}:
      raise RuntimeError('Replay transfer filenames do not match inventory.')
    if len({chunk.uuid for chunk in chunks}) != len(chunks):
      raise RuntimeError('Replay transfer contains duplicate chunk UUIDs.')

    chunks = list(reversed(sorted(chunks)))
    if chunks:
      numitems = self._numitems(chunks)
      total = 0
      count = 0
      for chunk in chunks:
        count += 1
        total += numitems[chunk.uuid]
        if total >= amount:
          break
      chunks = chunks[:count]
      numitems = self._numitems(chunks)
    else:
      numitems = {}

    with self.rwlock.writing:
      self._clear_for_bound_load()
      with self.refs_lock:
        for chunk in chunks:
          self.chunks[chunk.uuid] = chunk
          self.refs[chunk.uuid] = 0
        for chunk in reversed(chunks):
          count = numitems[chunk.uuid]
          self.refs[chunk.uuid] += count
          if chunk.succ in self.refs:
            self.refs[chunk.succ] += 1
          for index in range(count):
            self._insert(chunk.uuid, index)
    with self.checkpoint_lock:
      self._pending_inventory = None
      self._committed_inventory = None
    return {
        'chunks': len(chunks),
        'items': len(self.items),
        'schema': sorted(expected),
        'dropped_fields': sorted(drop_keys),
    }

  def _compact_loaded_replay_context(self, chunk):
    """Normalizes legacy Dreamer context arrays without rewriting disk data."""
    if not self.replay_context_dtype or not chunk.data:
      return
    deter = chunk.data.get('dyn/deter')
    if deter is not None:
      if self.replay_context_dtype == 'bfloat16':
        import ml_dtypes
        float_dtype, uint_dtype = np.dtype(ml_dtypes.bfloat16), np.dtype(np.uint16)
      else:
        float_dtype, uint_dtype = np.dtype(np.float32), np.dtype(np.uint32)
      if deter.dtype != uint_dtype:
        if np.issubdtype(deter.dtype, np.floating):
          decoded = deter.astype(float_dtype)
        elif deter.dtype == np.dtype(np.uint16):
          import ml_dtypes
          decoded = deter.view(np.dtype(ml_dtypes.bfloat16)).astype(float_dtype)
        elif deter.dtype == np.dtype(np.uint32):
          decoded = deter.view(np.float32).astype(float_dtype)
        else:
          raise TypeError(
              f'Unsupported dyn/deter replay dtype: {deter.dtype}')
        chunk.data['dyn/deter'] = decoded.view(uint_dtype)
    stoch = chunk.data.get('dyn/stoch')
    if stoch is not None:
      if stoch.ndim >= 3:
        if np.issubdtype(stoch.dtype, np.floating):
          stoch = np.nan_to_num(stoch, nan=0.0, posinf=1.0, neginf=0.0)
        dtype = np.uint8 if stoch.shape[-1] <= 256 else np.uint16
        chunk.data['dyn/stoch'] = np.argmax(stoch, -1).astype(dtype)
      elif stoch.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        chunk.data['dyn/stoch'] = stoch.astype(np.uint16)

  @property
  def committed_inventory(self):
    with self.checkpoint_lock:
      return self._copy_inventory(self._committed_inventory)

  @property
  def last_committed_inventory(self):
    return self.committed_inventory

  def checkpoint_committed(self, protected_inventories=None):
    """Record that the surrounding checkpoint transaction has committed."""
    with self.checkpoint_lock:
      if self._pending_inventory is not None:
        self._committed_inventory = self._pending_inventory
        self._pending_inventory = None
      committed = self._copy_inventory(self._committed_inventory)
    if protected_inventories is not None:
      self.checkpoint_gc(protected_inventories)
    return committed

  def checkpoint_gc(self, protected_inventories, *, cancelled=None):
    """Remove orphan chunks while retaining every checkpoint-bound chunk."""
    cancelled = cancelled or (lambda: False)

    def cancelled_result():
      return {'removed': 0, 'bytes': 0, 'errors': 0, 'cancelled': True}

    if not self.directory or protected_inventories is None:
      return {'removed': 0, 'bytes': 0, 'errors': 0}
    if cancelled():
      return cancelled_result()

    protected_entries = {}

    def protect_inventory(inventory):
      for entry in self._validate_inventory(inventory):
        name = entry['filename']
        integrity = (entry['size'], entry['sha256'])
        previous = protected_entries.get(name)
        if previous is not None and previous != integrity:
          raise ValueError(
              f'Conflicting checkpoint integrity for replay chunk {name}.')
        protected_entries[name] = integrity

    for inventory in protected_inventories:
      if cancelled():
        return cancelled_result()
      # Defer collection while a retained checkpoint has no bound inventory.
      if inventory is None:
        return {'removed': 0, 'bytes': 0, 'errors': 0}
      protect_inventory(inventory)

    with self.disk_lock:
      with self.rwlock.writing:
        if cancelled():
          return cancelled_result()
        with self.checkpoint_lock:
          inventories = (
              self._committed_inventory, self._pending_inventory)
          for inventory in inventories:
            if inventory is not None:
              protect_inventory(inventory)

        # Chunks can become live after the checkpoint inventory was captured.
        # In-flight saves are protected even if their chunk was concurrently
        # evicted from the in-memory replay.
        protected = set(protected_entries)
        protected.update(
            chunk.filename for chunk in self.chunks.values()
            if chunk.length > 0)
        protected.update(
            self.save_filenames[chunkid]
            for chunkid in self.save_promises
            if chunkid in self.save_filenames)

        try:
          directory_info = os.lstat(os.fspath(self.directory))
        except OSError as exc:
          print(f'Could not inspect replay directory for GC: {exc}')
          return {'removed': 0, 'bytes': 0, 'errors': 1}
        if (
            stat.S_ISLNK(directory_info.st_mode) or
            not stat.S_ISDIR(directory_info.st_mode)):
          print(f'Refusing replay GC for unsafe directory: {self.directory}')
          return {'removed': 0, 'bytes': 0, 'errors': 1}

        for name, (expected_size, expected_digest) in protected_entries.items():
          if cancelled():
            return cancelled_result()
          filename = self.directory / name
          try:
            signature = chunklib.regular_file_stat(filename)
          except RuntimeError as exc:
            print(f'Refusing replay GC with damaged protected chunk: {exc}')
            return {'removed': 0, 'bytes': 0, 'errors': 1}
          cached = self.integrity_cache.get(name)
          if cached and cached['signature'] == signature:
            actual_size = cached['size']
            actual_digest = cached['sha256']
          else:
            try:
              actual_size, actual_digest, signature = (
                  chunklib.file_integrity(filename))
            except RuntimeError as exc:
              print(f'Refusing replay GC with damaged protected chunk: {exc}')
              return {'removed': 0, 'bytes': 0, 'errors': 1}
            self.integrity_cache[name] = {
                'size': actual_size,
                'sha256': actual_digest,
                'signature': signature,
            }
          if (
              actual_size != expected_size or
              actual_digest != expected_digest):
            print(
                f'Refusing replay GC with integrity-mismatched protected chunk: '
                f'{filename}')
            return {'removed': 0, 'bytes': 0, 'errors': 1}

        candidates = []
        try:
          entries = os.scandir(os.fspath(self.directory))
        except OSError as exc:
          print(f'Could not scan replay directory for GC: {exc}')
          return {'removed': 0, 'bytes': 0, 'errors': 1}
        with entries:
          for entry in entries:
            if cancelled():
              return cancelled_result()
            name = entry.name
            if name in protected or not self._is_chunk_filename(name):
              continue
            filename = self.directory / name
            try:
              signature = chunklib.regular_file_stat(filename)
            except RuntimeError as exc:
              # A canonical-looking symlink or special file makes the entire
              # pass fail closed rather than authorizing a partial deletion.
              print(f'Refusing replay GC with unsafe chunk entry: {exc}')
              return {'removed': 0, 'bytes': 0, 'errors': 1}
            candidates.append((name, filename, signature))

        # Recheck every candidate before the first unlink so path replacement
        # cannot silently broaden the set authorized by the scan above.
        try:
          changed = False
          for _, filename, signature in candidates:
            if cancelled():
              return cancelled_result()
            if chunklib.regular_file_stat(filename) != signature:
              changed = True
              break
          if changed:
            print('Refusing replay GC because a chunk changed during scan.')
            return {'removed': 0, 'bytes': 0, 'errors': 1}
        except RuntimeError as exc:
          print(f'Refusing replay GC because a chunk changed: {exc}')
          return {'removed': 0, 'bytes': 0, 'errors': 1}

        removed = []
        errors = 0
        was_cancelled = False
        for name, filename, signature in candidates:
          if cancelled():
            was_cancelled = True
            break
          try:
            os.unlink(os.fspath(filename))
          except FileNotFoundError:
            continue
          except OSError as exc:
            errors += 1
            print(f'Could not remove orphan replay chunk {filename}: {exc}')
          else:
            removed.append((name, signature[2]))
        if removed:
          try:
            chunklib.fsync_directory(self.directory)
          except OSError as exc:
            errors += 1
            print(f'Could not fsync replay directory after GC: {exc}')

        removed_names = {name for name, _ in removed}
        for name in removed_names:
          self.integrity_cache.pop(name, None)
        for chunkid, name in list(self.save_filenames.items()):
          if (
              name in removed_names and chunkid not in self.chunks and
              chunkid not in self.save_promises):
            self.save_filenames.pop(chunkid, None)
            self.saved.discard(chunkid)
        return {
            'removed': len(removed),
            'bytes': sum(size for _, size in removed),
            'errors': errors,
            'cancelled': was_cancelled,
        }

  def _copy_inventory(self, inventory):
    if inventory is None:
      return None
    return {
        'format': inventory['format'],
        'chunks': [entry.copy() for entry in inventory['chunks']],
    }

  def _is_chunk_filename(self, name):
    if not isinstance(name, str) or os.path.basename(name) != name:
      return False
    try:
      parts = chunklib.filename_stem(name).split('-')
    except ValueError:
      return False
    if len(parts) != 4 or not parts[0]:
      return False
    alphabet = set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')

    def valid_uuid(value):
      return (
          (len(value) == 22 and set(value) <= alphabet) or
          (value.isdigit() and len(value) <= 22 and str(int(value)) == value))

    if not valid_uuid(parts[1]) or not valid_uuid(parts[2]):
      return False
    if not parts[3].isdigit() or len(parts[3]) > 20:
      return False
    length = int(parts[3])
    return str(length) == parts[3] and length > 0

  def _validate_inventory(self, data):
    if not isinstance(data, dict) or data.get('format') != _INVENTORY_FORMAT:
      raise ValueError('Unsupported or malformed replay checkpoint inventory.')
    if set(data) != {'format', 'chunks'} or not isinstance(data['chunks'], list):
      raise ValueError('Malformed replay checkpoint inventory.')
    entries = []
    names = set()
    for entry in data['chunks']:
      if not isinstance(entry, dict) or set(entry) != {
          'filename', 'size', 'sha256'}:
        raise ValueError('Malformed replay checkpoint chunk entry.')
      name = entry['filename']
      size = entry['size']
      digest = entry['sha256']
      if (
          not isinstance(name, str) or not name or
          os.path.basename(name) != name or name in ('.', '..') or
          '/' in name or '\\' in name or
          chunklib.filename_suffix(name) is None):
        raise ValueError(f'Unsafe replay checkpoint filename: {name!r}')
      if name in names:
        raise ValueError(f'Duplicate replay checkpoint filename: {name}')
      if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError(f'Invalid replay checkpoint size for {name}.')
      if (
          not isinstance(digest, str) or len(digest) != 64 or
          any(x not in '0123456789abcdef' for x in digest)):
        raise ValueError(f'Invalid replay checkpoint SHA256 for {name}.')
      names.add(name)
      entries.append(entry.copy())
    if [x['filename'] for x in entries] != sorted(names):
      raise ValueError('Replay checkpoint inventory is not canonically sorted.')
    return entries

  def _clear_for_bound_load(self):
    self._clear_sampler(self.sampler)
    self.chunks.clear()
    with self.refs_lock:
      self.refs.clear()
    self.items.clear()
    self.fifo.clear()
    self.itemid = 0
    self.current.clear()
    self.streams.clear()
    self.saved.clear()
    self.save_filenames.clear()
    if self.online:
      self.lengths.clear()
      self.queue.clear()

  def _clear_sampler(self, sampler):
    if isinstance(sampler, selectors.Mixture):
      for child in sampler.selectors:
        self._clear_sampler(child)
    elif isinstance(sampler, selectors.Fifo):
      sampler.queue.clear()
    elif isinstance(sampler, selectors.Uniform):
      sampler.indices.clear()
      sampler.keys.clear()
    elif isinstance(sampler, selectors.Recency):
      sampler.step = 0
      sampler.steps.clear()
      sampler.items.clear()
    elif isinstance(sampler, selectors.Prioritized):
      sampler.prios.clear()
      sampler.stepitems.clear()
      sampler.items.clear()
      sampler.tree.root = selectors.SampleTreeNode()
      sampler.tree.last = None
      sampler.tree.entries.clear()
    elif self.items:
      for itemid in reversed(self.items):
        del sampler[itemid]

  @elements.timer.section('complete_chunk')
  def _complete(self, chunk, worker):
    # Checkpoint saves complete every worker's active chunk. Those chunks are
    # often much shorter than chunksize, so retaining their original backing
    # arrays would multiply replay RAM by chunksize / checkpoint_interval.
    chunk.compact()
    succ = chunklib.Chunk(self.chunksize)
    with self.refs_lock:
      self.refs[chunk.uuid] -= 1
      self.refs[succ.uuid] = 2
    self.chunks[succ.uuid] = succ
    self.current[worker] = (succ.uuid, 0)
    chunk.succ = succ.uuid
    return succ

  def _numitems(self, chunks):
    chunks = [x.filename if hasattr(x, 'filename') else x for x in chunks]
    if not chunks:
      return 0
    chunks = list(reversed(sorted([
        chunklib.filename_stem(x) for x in chunks])))
    times, uuids, succs, lengths = zip(*[x.split('-') for x in chunks])
    uuids = [elements.UUID(x) for x in uuids]
    succs = [elements.UUID(x) for x in succs]
    lengths = {k: int(v) for k, v in zip(uuids, lengths)}
    future = {}
    for uuid, succ in zip(uuids, succs):
      future[uuid] = lengths[uuid] + future.get(succ, 0)
    numitems = {}
    for uuid, succ in zip(uuids, succs):
      numitems[uuid] = lengths[uuid] + 1 - self.length + future.get(succ, 0)
    numitems = {k: np.clip(v, 0, lengths[k]) for k, v in numitems.items()}
    return numitems

  def _notempty(self, reason=False):
    if reason:
      return (True, 'ok') if len(self.sampler) else (False, 'empty buffer')
    else:
      return bool(len(self.sampler))
