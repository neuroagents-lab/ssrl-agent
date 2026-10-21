import hashlib
import io
import os
import stat
import sys
import traceback
import zipfile

import elements
import numpy as np
import zstandard as zstd


_READ_SIZE = 1024 * 1024
LEGACY_SUFFIX = '.npz'
ZSTD_SUFFIX = '.npz.zst'
CHUNK_SUFFIXES = (ZSTD_SUFFIX, LEGACY_SUFFIX)
CHUNK_GLOBS = tuple(f'*{suffix}' for suffix in CHUNK_SUFFIXES)
_ZSTD_LEVEL = 3
# Bound compressed frame and window sizes before allocating decompression
# buffers, while allowing chunks with large observations.
_MAX_DECOMPRESSED_BYTES = 1024 * 1024 * 1024
_MAX_COMPRESSED_BYTES = _MAX_DECOMPRESSED_BYTES + 64 * 1024 * 1024


def filename_suffix(filename):
  name = os.path.basename(os.fspath(filename))
  return next((x for x in CHUNK_SUFFIXES if name.endswith(x)), None)


def filename_stem(filename):
  name = os.path.basename(os.fspath(filename))
  suffix = filename_suffix(name)
  if suffix is None:
    raise ValueError(f'Invalid replay chunk filename: {name}')
  return name[:-len(suffix)]


def regular_file_stat(filename):
  """Return a cache signature while rejecting missing and unsafe files."""
  filename = os.fspath(filename)
  try:
    info = os.lstat(filename)
  except OSError as exc:
    raise RuntimeError(f'Replay chunk is missing: {filename}') from exc
  if stat.S_ISLNK(info.st_mode):
    raise RuntimeError(f'Replay chunk must not be a symlink: {filename}')
  if not stat.S_ISREG(info.st_mode):
    raise RuntimeError(f'Replay chunk is not a regular file: {filename}')
  return _stat_signature(info)


def file_integrity(filename):
  """Hash a regular file through a no-follow descriptor."""
  filename = os.fspath(filename)
  regular_file_stat(filename)
  flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
  try:
    fd = os.open(filename, flags)
  except OSError as exc:
    raise RuntimeError(f'Could not safely open replay chunk: {filename}') from exc
  try:
    with os.fdopen(fd, 'rb') as stream:
      before = os.fstat(stream.fileno())
      if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f'Replay chunk is not a regular file: {filename}')
      digest = hashlib.sha256()
      while True:
        block = stream.read(_READ_SIZE)
        if not block:
          break
        digest.update(block)
      after = os.fstat(stream.fileno())
      if _stat_signature(before) != _stat_signature(after):
        raise RuntimeError(f'Replay chunk changed while hashing: {filename}')
      return before.st_size, digest.hexdigest(), _stat_signature(before)
  except Exception:
    # fdopen() owns and closes the descriptor after it succeeds.
    raise


def fsync_directory(directory):
  flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
  fd = os.open(os.fspath(directory), flags)
  try:
    os.fsync(fd)
  finally:
    os.close(fd)


def _stat_signature(info):
  return (
      info.st_dev, info.st_ino, info.st_size,
      info.st_mtime_ns, info.st_ctime_ns)


class Chunk:

  __slots__ = (
      'time', 'uuid', 'succ', 'length', 'size', 'data', 'saved', 'suffix')

  def __init__(self, size=1024):
    self.time = elements.timestamp(millis=True)
    self.uuid = elements.UUID()
    self.succ = elements.UUID(0)
    self.length = 0
    self.size = size
    self.data = None
    self.saved = False
    self.suffix = ZSTD_SUFFIX

  def __repr__(self):
    return f'Chunk({self.filename})'

  def __lt__(self, other):
    return self.time < other.time

  @property
  def filename(self):
    succ = self.succ.uuid if isinstance(self.succ, type(self)) else self.succ
    return f'{self.time}-{str(self.uuid)}-{str(succ)}-{self.length}{self.suffix}'

  @property
  def nbytes(self):
    if not self.data:
      return 0
    return sum(x.nbytes for x in self.data.values())

  def compact(self):
    """Release unused capacity once a chunk will no longer be appended to."""
    assert 0 <= self.length <= self.size, (self.length, self.size)
    if not self.data or self.length == self.size:
      return
    # A slice would retain the original full-sized allocation through its base
    # pointer. Completed chunks therefore need owned copies of their used rows.
    self.data = {
        key: np.array(value[:self.length], copy=True, order='C')
        for key, value in self.data.items()
    }
    self.size = self.length

  def append(self, step):
    assert self.length < self.size
    if not self.data:
      example = step
      self.data = {
          k: np.empty((self.size, *v.shape), v.dtype)
          for k, v in example.items()}
    for key, value in step.items():
      self.data[key][self.length] = value
    self.length += 1

  def append_many(self, steps):
    length = len(next(iter(steps.values())))
    assert self.length + length <= self.size
    if not self.data:
      example = steps
      self.data = {
          k: np.empty((self.size, *v.shape[1:]), v.dtype)
          for k, v in example.items()}
    start = self.length
    stop = start + length
    for key, value in steps.items():
      self.data[key][start: stop] = value
    self.length = stop
    return start

  def update(self, index, length, mapping):
    assert 0 <= index <= self.length, (index, self.length)
    assert 0 <= index + length <= self.length, (index, length, self.length)
    for key, value in mapping.items():
      self.data[key][index: index + length] = value

  def slice(self, index, length):
    assert 0 <= index and index + length <= self.length
    return {k: v[index: index + length] for k, v in self.data.items()}

  @elements.timer.section('chunk_save')
  def save(self, directory, log=False):
    assert not self.saved
    filename = elements.Path(directory) / self.filename
    data = {k: v[:self.length] for k, v in self.data.items()}
    with io.BytesIO() as stream:
      # The NPZ container remains useful for dtype and shape metadata, but its
      # members are deliberately stored rather than DEFLATE-compressed. Zstd
      # then compresses the complete container in one frame, which is faster and
      # allows a single bounded decompression check on restore.
      np.savez(stream, **data)
      payload = zstd.ZstdCompressor(level=_ZSTD_LEVEL).compress(
          stream.getbuffer())
    digest = hashlib.sha256(payload).hexdigest()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, 'O_NOFOLLOW', 0)
    try:
      fd = os.open(os.fspath(filename), flags, 0o660)
    except OSError as exc:
      raise RuntimeError(
          f'Refusing to replace existing replay chunk: {filename}') from exc
    try:
      with os.fdopen(fd, 'wb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
      signature = regular_file_stat(filename)
      if signature[2] != len(payload):
        raise RuntimeError(f'Replay chunk size changed while saving: {filename}')
      self.saved = True
      log and print(f'Saved chunk: {filename.name}')
      return len(payload), digest, signature
    except BaseException:
      self.saved = False
      try:
        os.unlink(os.fspath(filename))
      except OSError:
        pass
      raise

  @classmethod
  def load(
      cls, filename, error='raise', expected_size=None, expected_sha256=None,
      return_integrity=False):
    assert error in ('raise', 'none')
    try:
      filename = elements.Path(filename)
      suffix = filename_suffix(filename.name)
      parts = filename_stem(filename.name).split('-')
      if len(parts) != 4 or suffix is None:
        raise ValueError(f'Invalid replay chunk filename: {filename.name}')
      time, uuid, succ, length = parts
      length = int(length)
      if length < 1:
        raise ValueError(f'Invalid replay chunk length: {filename.name}')

      flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
      regular_file_stat(filename)
      try:
        fd = os.open(os.fspath(filename), flags)
      except OSError as exc:
        raise RuntimeError(
            f'Could not safely open replay chunk: {filename}') from exc
      with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        actual = None
        if not stat.S_ISREG(before.st_mode):
          raise RuntimeError(f'Replay chunk is not regular: {filename}')
        if expected_size is not None and before.st_size != expected_size:
          raise RuntimeError(
              f'Replay chunk size mismatch for {filename.name}: '
              f'expected {expected_size}, got {before.st_size}')
        if expected_sha256 is not None:
          digest = hashlib.sha256()
          while True:
            block = stream.read(_READ_SIZE)
            if not block:
              break
            digest.update(block)
          actual = digest.hexdigest()
          if actual != expected_sha256:
            raise RuntimeError(
                f'Replay chunk SHA256 mismatch for {filename.name}: '
                f'expected {expected_sha256}, got {actual}')
          stream.seek(0)
        if suffix == ZSTD_SUFFIX:
          if before.st_size > _MAX_COMPRESSED_BYTES:
            raise ValueError(
                f'Compressed replay chunk exceeds safety limit: '
                f'{filename.name}: {before.st_size} bytes')
          payload = stream.read()
          params = zstd.get_frame_parameters(payload)
          declared = params.content_size
          if declared in (zstd.CONTENTSIZE_UNKNOWN, zstd.CONTENTSIZE_ERROR):
            raise ValueError(
                f'Replay chunk has no bounded Zstandard content size: '
                f'{filename.name}')
          if declared > _MAX_DECOMPRESSED_BYTES:
            raise ValueError(
                f'Replay chunk exceeds decompression safety limit: '
                f'{filename.name}: {declared} bytes')
          if params.window_size > _MAX_DECOMPRESSED_BYTES:
            raise ValueError(
                f'Replay chunk exceeds Zstandard window safety limit: '
                f'{filename.name}: {params.window_size} bytes')
          max_window_kib = max(1024, _MAX_DECOMPRESSED_BYTES // 1024)
          decoded = zstd.ZstdDecompressor(
              max_window_size=max_window_kib).decompress(
                  payload,
                  max_output_size=_MAX_DECOMPRESSED_BYTES,
                  allow_extra_data=False)
          if len(decoded) != declared:
            raise ValueError(
                f'Replay chunk decompressed size mismatch: {filename.name}: '
                f'expected {declared}, got {len(decoded)}')
          archive_stream = io.BytesIO(decoded)
          with zipfile.ZipFile(archive_stream) as container:
            members = container.infolist()
            if not members or any(
                member.compress_type != zipfile.ZIP_STORED
                for member in members):
              raise ValueError(
                  f'Zstandard replay chunk contains compressed NPZ members: '
                  f'{filename.name}')
          archive_stream.seek(0)
        else:
          archive_stream = stream
        with np.load(archive_stream, allow_pickle=False) as archive:
          data = {k: archive[k] for k in archive.keys()}
        after = os.fstat(stream.fileno())
        if _stat_signature(before) != _stat_signature(after):
          raise RuntimeError(f'Replay chunk changed while loading: {filename}')
        signature = _stat_signature(after)
      if regular_file_stat(filename) != signature:
        raise RuntimeError(f'Replay chunk path changed while loading: {filename}')
      if not data or any(
          value.ndim < 1 or len(value) != length for value in data.values()):
        shapes = {key: value.shape for key, value in data.items()}
        raise ValueError(
            f'Replay chunk payload length mismatch: {filename.name}: {shapes}')
    except Exception:
      tb = ''.join(traceback.format_exception(sys.exception()))
      print(f'Error loading chunk {filename}:\n{tb}')
      if error == 'raise':
        raise
      else:
        return None
    chunk = cls(length)
    chunk.time = time
    chunk.uuid = elements.UUID(uuid)
    chunk.succ = elements.UUID(succ)
    chunk.length = length
    chunk.data = data
    chunk.saved = True
    chunk.suffix = suffix
    if return_integrity:
      return chunk, (before.st_size, actual, signature)
    return chunk
