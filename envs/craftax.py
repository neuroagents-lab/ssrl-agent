import contextlib
import importlib
import os
import pathlib
import re
import tempfile

import gymnasium as gym
import numpy as np


_TASKS = {
    "full": "Craftax-Pixels-v1",
    "full_pixel": "Craftax-Pixels-v1",
    "full_pixels": "Craftax-Pixels-v1",
    "pixel": "Craftax-Pixels-v1",
    "pixels": "Craftax-Pixels-v1",
    "full_symbolic": "Craftax-Symbolic-v1",
    "full_symbol": "Craftax-Symbolic-v1",
    "full_symbols": "Craftax-Symbolic-v1",
    "full_vector": "Craftax-Symbolic-v1",
    "symbolic": "Craftax-Symbolic-v1",
    "symbol": "Craftax-Symbolic-v1",
    "symbols": "Craftax-Symbolic-v1",
    "vector": "Craftax-Symbolic-v1",
    "classic": "Craftax-Classic-Pixels-v1",
    "classic_pixel": "Craftax-Classic-Pixels-v1",
    "classic_pixels": "Craftax-Classic-Pixels-v1",
    "classic_symbolic": "Craftax-Classic-Symbolic-v1",
    "classic_symbol": "Craftax-Classic-Symbolic-v1",
    "classic_symbols": "Craftax-Classic-Symbolic-v1",
    "classic_vector": "Craftax-Classic-Symbolic-v1",
}


class Craftax(gym.Env):
    metadata = {}

    def __init__(
        self,
        task="pixels",
        size=(64, 64),
        seed=0,
        observation_mode=None,
        logs=True,
        render=False,
        include_image=False,
    ):
        try:
            import jax
            from craftax.craftax_env import make_craftax_env_from_name
        except ImportError as exc:
            raise RuntimeError(
                "Craftax environments require `craftax==1.6.1` and JAX. "
                "Install the runtime dependencies from requirements.txt."
            ) from exc

        self._jax = jax
        self._env_name = _env_name(task, observation_mode)
        self._symbolic = "Symbolic" in self._env_name
        self._include_image = bool(include_image)
        self._has_score = self._env_name.startswith("Craftax-Classic-")
        self._render_size = _as_size(size)
        self._size = (
            self._render_size
            if not self._symbolic or self._include_image
            else None
        )
        self._logs = bool(logs)
        self._render_video = bool(render)

        _configure_texture_cache(self._env_name)
        self._env = make_craftax_env_from_name(self._env_name, auto_reset=False)
        self._params = self._env.default_params
        self._rng = self._new_key(int(seed))
        self._state = None
        self._done = True
        self._last_image = None
        self._reset = jax.jit(self._env.reset)
        self._step = jax.jit(self._env.step)
        self._achievement_names = _achievement_names(self._env_name)
        self._render_env = None
        self._render_state = None
        if self._symbolic and (self._include_image or self._render_video):
            pixel_env_name = self._env_name.replace("Symbolic", "Pixels")
            _configure_texture_cache(pixel_env_name)
            self._render_env = make_craftax_env_from_name(
                pixel_env_name, auto_reset=False
            )
            render_state = getattr(self._render_env, "get_obs", None)
            if not callable(render_state):
                raise RuntimeError(
                    f"{pixel_env_name} does not expose get_obs() for video rendering."
                )
            self._render_state = jax.jit(render_state)

        obs_space = self._env.observation_space(self._params)
        obs_shape = tuple(int(x) for x in obs_space.shape)
        if self._symbolic:
            spaces = {
                "vector": gym.spaces.Box(
                    0.0, 1.0, obs_shape, dtype=np.float32
                )
            }
            if self._include_image:
                spaces["image"] = gym.spaces.Box(
                    0, 255, self._size + (3,), dtype=np.uint8
                )
            elif self._render_video:
                spaces["log/image"] = gym.spaces.Box(
                    0, 255, self._video_size + (3,), dtype=np.uint8
                )
        else:
            image_shape = self._size + (3,)
            spaces = {
                "image": gym.spaces.Box(0, 255, image_shape, dtype=np.uint8)
            }
        if self._logs:
            if self._has_score:
                spaces["log/score"] = gym.spaces.Box(
                    -np.inf, np.inf, shape=(), dtype=np.float32
                )
            spaces.update(
                {
                    f"log/achievement_{name}": gym.spaces.Box(
                        -np.inf, np.inf, shape=(), dtype=np.float32
                    )
                    for name in self._achievement_names
                }
            )
        self.observation_space = gym.spaces.Dict(spaces)

        action_space = self._env.action_space(self._params)
        action_count = getattr(action_space, "n", None)
        if action_count is None:
            action_count = getattr(self._env, "num_actions", None)
        if action_count is None:
            raise RuntimeError(f"Could not infer action count for {self._env_name}.")
        self.action_space = gym.spaces.Discrete(int(action_count))

    def reset(self):
        self._rng, key = self._split_key()
        obs, self._state = self._call_jax(self._reset, key, self._params)
        self._done = False
        return self._obs(obs, {}, is_first=True)

    def step(self, action):
        if self._done:
            return self.reset(), np.float32(0.0), False, {}

        self._rng, key = self._split_key()
        obs, self._state, reward, done, info = self._call_jax(
            self._step, key, self._state, int(np.asarray(action).item()), self._params
        )
        info = self._info_to_numpy(info)
        reward = np.float32(_scalar(self._device_get(reward)))
        self._done = bool(_scalar(self._device_get(done)))
        discount = float(_scalar(info.get("discount", 1.0)))
        terminal = self._done and discount == 0.0
        return (
            self._obs(obs, info, is_last=self._done, is_terminal=terminal),
            reward,
            self._done,
            info,
        )

    def render(self):
        return self._last_image

    def close(self):
        pass

    def _obs(self, raw_obs, info, is_first=False, is_last=False, is_terminal=False):
        raw_obs = self._device_get(raw_obs)
        if self._symbolic:
            obs = {"vector": np.asarray(raw_obs, dtype=np.float32)}
            if self._render_state is not None:
                image = self._device_get(
                    self._call_jax(self._render_state, self._state)
                )
                image = _image_to_uint8(image)
                image = _resize_nearest(
                    image,
                    self._size if self._include_image else self._video_size,
                )
                self._last_image = image
                if self._include_image:
                    obs["image"] = image
                else:
                    obs["log/image"] = image
        else:
            image = _image_to_uint8(raw_obs)
            image = _resize_nearest(image, self._size)
            self._last_image = image
            obs = {"image": image}
        obs.update(
            is_first=bool(is_first),
            is_last=bool(is_last),
            is_terminal=bool(is_terminal),
        )
        if self._logs:
            obs.update(self._log_obs(info))
        return obs

    @property
    def _video_size(self):
        return self._render_size

    def _log_obs(self, info):
        logs = {}
        if self._has_score:
            logs["log/score"] = np.float32(_scalar(info.get("score", 0.0)))
        for name in self._achievement_names:
            value = info.get(f"Achievements/{name}", 0.0)
            logs[f"log/achievement_{name}"] = np.float32(_scalar(value))
        return logs

    def _info_to_numpy(self, info):
        return {
            str(key): np.asarray(self._device_get(value)).astype(np.float32)
            for key, value in dict(info).items()
        }

    def _new_key(self, seed):
        with _jax_transfer_scope(self._jax, "explicit_device_put_scope"):
            return self._jax.random.PRNGKey(seed)

    def _split_key(self):
        with _jax_transfer_scope(self._jax, "explicit_device_put_scope"):
            keys = self._jax.random.split(self._rng, 2)
            return keys[0], keys[1]

    def _call_jax(self, fn, *args):
        with _jax_transfer_scope(self._jax, "explicit_device_get_scope"):
            with _jax_transfer_scope(self._jax, "explicit_device_put_scope"):
                return fn(*args)

    def _device_get(self, value):
        with _jax_transfer_scope(self._jax, "explicit_device_get_scope"):
            return self._jax.device_get(value)


def _env_name(task, observation_mode=None):
    task = str(task)
    if task in {
        "Craftax-Pixels-v1",
        "Craftax-Symbolic-v1",
        "Craftax-Pixels-AutoReset-v1",
        "Craftax-Symbolic-AutoReset-v1",
        "Craftax-Classic-Pixels-v1",
        "Craftax-Classic-Symbolic-v1",
        "Craftax-Classic-Pixels-AutoReset-v1",
        "Craftax-Classic-Symbolic-AutoReset-v1",
    }:
        return task.replace("-AutoReset-v1", "-v1")

    key = _normalize(task)
    if key in {"full", "classic"} and observation_mode is not None:
        mode = _normalize(observation_mode)
        symbolic = mode in {"symbolic", "symbol", "symbols", "vector"}
        key = f"{key}_symbolic" if symbolic else f"{key}_pixels"
    if key not in _TASKS:
        raise NotImplementedError(
            f"Unknown Craftax task {task!r}. Use craftax_pixels or "
            "craftax_symbolic for the full game, or craftax_classic_pixels "
            "or craftax_classic_symbolic for Craftax Classic."
        )
    return _TASKS[key]


def _normalize(value):
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip()).strip("_").lower()


def _achievement_names(env_name):
    try:
        if str(env_name).startswith("Craftax-Classic-"):
            from craftax.craftax_classic.constants import Achievement
        else:
            from craftax.craftax.constants import Achievement
    except ImportError:
        return []
    return [achievement.name.lower() for achievement in Achievement]


def _configure_texture_cache(env_name):
    if "Pixels" not in str(env_name):
        return
    module_name = (
        "craftax.craftax_classic.constants"
        if str(env_name).startswith("Craftax-Classic-")
        else "craftax.craftax.constants"
    )
    try:
        constants = importlib.import_module(module_name)
    except ImportError:
        return

    configured = os.environ.get("CRAFTAX_CACHE_DIR")
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if configured:
        candidates = [pathlib.Path(configured).expanduser()]
    elif xdg_cache:
        candidates = [pathlib.Path(xdg_cache).expanduser() / "craftax"]
    else:
        candidates = [pathlib.Path.home() / ".cache" / "craftax"]
    candidates.append(pathlib.Path(tempfile.gettempdir()) / "craftax")

    filename = (
        "classic_texture_cache.pbz2"
        if str(env_name).startswith("Craftax-Classic-")
        else "texture_cache.pbz2"
    )
    for directory in candidates:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        constants.TEXTURE_CACHE_FILE = str(directory / filename)
        return


def _as_size(value):
    return tuple(int(x) for x in value)


def _scalar(value):
    return np.asarray(value).reshape(()).item()


def _image_to_uint8(value):
    image = np.asarray(value)
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim >= 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.ndim >= 3 and image.shape[-1] > 3:
        image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        finite = image[np.isfinite(image)]
        max_value = float(finite.max()) if finite.size else 0.0
        if max_value <= 1.0:
            image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _resize_nearest(image, size):
    height, width = tuple(int(x) for x in size)
    if image.shape[:2] == (height, width):
        return image
    yidx = np.minimum(
        (np.arange(height) * image.shape[0] // height).astype(np.int64),
        image.shape[0] - 1,
    )
    xidx = np.minimum(
        (np.arange(width) * image.shape[1] // width).astype(np.int64),
        image.shape[1] - 1,
    )
    return image[yidx][:, xidx]


def _jax_transfer_scope(jax, name):
    config = getattr(getattr(jax, "_src", None), "config", None)
    scope = getattr(config, name, None)
    if scope is None:
        return contextlib.nullcontext()
    return scope()
