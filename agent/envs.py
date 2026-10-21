from __future__ import annotations

import contextlib
import functools
import importlib
import importlib.util
import pathlib
import re
import sys
import types
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np

from . import dinov2 as dinov2_utils


_PLAYGROUND_COUNTER_KEYS = (
    "log/termination/environment",
    "log/termination/time_limit",
    "log/termination/nonfinite",
    "log/reset/nonfinite",
    "log/replacement/nonfinite_observation",
    "log/replacement/nonfinite_reward",
    "log/replacement/nonfinite_metric",
)


_DMC_PLAYGROUND_ALIASES = {
    "acrobot_swingup": "AcrobotSwingup",
    "acrobot_swingup_sparse": "AcrobotSwingupSparse",
    "ball_in_cup_catch": "BallInCup",
    "cartpole_balance": "CartpoleBalance",
    "cartpole_balance_sparse": "CartpoleBalanceSparse",
    "cartpole_swingup": "CartpoleSwingup",
    "cartpole_swingup_sparse": "CartpoleSwingupSparse",
    "cheetah_run": "CheetahRun",
    "finger_spin": "FingerSpin",
    "finger_turn_easy": "FingerTurnEasy",
    "finger_turn_hard": "FingerTurnHard",
    "fish_swim": "FishSwim",
    "hopper_hop": "HopperHop",
    "hopper_stand": "HopperStand",
    "humanoid_stand": "HumanoidStand",
    "humanoid_walk": "HumanoidWalk",
    "humanoid_run": "HumanoidRun",
    "pendulum_swingup": "PendulumSwingup",
    "point_mass_easy": "PointMass",
    "point_mass": "PointMass",
    "reacher_easy": "ReacherEasy",
    "reacher_hard": "ReacherHard",
    "swimmer_swimmer6": "SwimmerSwimmer6",
    "walker_run": "WalkerRun",
    "walker_stand": "WalkerStand",
    "walker_walk": "WalkerWalk",
}

LOCOMOTION_PLAYGROUND_ENVS = (
    "ApolloJoystickFlatTerrain",
    "BarkourJoystick",
    "BerkeleyHumanoidJoystickFlatTerrain",
    "BerkeleyHumanoidJoystickRoughTerrain",
    "G1JoystickFlatTerrain",
    "G1JoystickRoughTerrain",
    "Go1JoystickFlatTerrain",
    "Go1JoystickRoughTerrain",
    "Go1Getup",
    "Go1Handstand",
    "Go1Footstand",
    "H1InplaceGaitTracking",
    "H1JoystickGaitTracking",
    "Op3Joystick",
    "SpotFlatTerrainJoystick",
    "SpotGetup",
    "SpotJoystickGaitTracking",
    "T1JoystickFlatTerrain",
    "T1JoystickRoughTerrain",
)

MANIPULATION_PLAYGROUND_ENVS = (
    "AlohaHandOver",
    "AlohaSinglePegInsertion",
    "PandaPickCube",
    "PandaPickCubeOrientation",
    "PandaPickCubeCartesian",
    "PandaOpenCabinet",
    "PandaRobotiqPushCube",
    "LeapCubeReorient",
    "LeapCubeRotateZAxis",
    "AeroCubeRotateZAxis",
)


def _camel_to_snake(name):
    text = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return text.lower()


def _playground_aliases(canonical_names, extra_aliases=None):
    aliases = {name: name for name in canonical_names}
    aliases.update({_camel_to_snake(name): name for name in canonical_names})
    if extra_aliases:
        aliases.update(extra_aliases)
    return aliases


DMC_PLAYGROUND_TASKS = _playground_aliases(
    tuple(dict.fromkeys(_DMC_PLAYGROUND_ALIASES.values())),
    _DMC_PLAYGROUND_ALIASES,
)

PLAYGROUND_TASKS_BY_SUITE = {
    "dmc": DMC_PLAYGROUND_TASKS,
    "locomotion": _playground_aliases(LOCOMOTION_PLAYGROUND_ENVS),
    "manipulation": _playground_aliases(MANIPULATION_PLAYGROUND_ENVS),
}

PLAYGROUND_SUITE_MODULES = {
    "dmc": "dm_control_suite",
    "locomotion": "locomotion",
    "manipulation": "manipulation",
}

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_THIS_FILE = pathlib.Path(__file__).resolve()


def _ensure_external_envs_package():
    package = sys.modules.get("envs")
    if package is not None:
        package_file = getattr(package, "__file__", None)
        package_path = pathlib.Path(package_file).resolve() if package_file else None
        if package_path != _THIS_FILE and hasattr(package, "__path__"):
            return package
        sys.modules.pop("envs", None)

    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(_REPO_ROOT))
        return importlib.import_module("envs")
    finally:
        sys.path[:] = old_path


def _import_env_module(name):
    _ensure_external_envs_package()
    return importlib.import_module(f"envs.{name}")


def _get(config, key, default=None):
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def task_env_value(env_config, key, full_task, default=None):
    """Resolve a scalar environment option, optionally from a per-task map."""
    task_values = _get(env_config, f"{key}_by_task", None)
    if task_values is None:
        return _get(env_config, key, default)
    if not isinstance(task_values, Mapping):
        raise TypeError(f"envs.{key}_by_task must be a mapping.")
    full_task = str(full_task)
    task_name = full_task.split("_", 1)[1] if "_" in full_task else full_task
    for candidate in (full_task, task_name, "default"):
        if candidate in task_values:
            return task_values[candidate]
    choices = ", ".join(str(candidate) for candidate in task_values)
    raise ValueError(
        f"envs.{key}_by_task has no entry for {full_task!r}; "
        f"available entries: {choices or '<none>'}."
    )


def environment_metadata(full_task):
    """Returns stable benchmark metadata for raw evaluation records."""
    suite = str(full_task).split("_", 1)[0]
    if suite != "robokitchen":
        return {}
    robokitchen = _import_env_module("robokitchen")
    return {
        "task_registry_hash": str(robokitchen.TASK_REGISTRY_HASH),
        "port_version": str(robokitchen.PORT_VERSION),
    }


def _as_playground_observation_keys(value):
    if value is None:
        return None
    if isinstance(value, str):
        keys = (value,)
    else:
        try:
            keys = tuple(str(key) for key in value)
        except TypeError as exc:
            raise TypeError(
                "envs.observation_keys must be a sequence of observation names."
            ) from exc
    if not keys or any(not key for key in keys):
        raise ValueError("envs.observation_keys must contain nonempty names.")
    if len(set(keys)) != len(keys):
        raise ValueError("envs.observation_keys must not contain duplicates.")
    return keys


def _filter_playground_observations(observations, observation_keys):
    if observation_keys is None:
        return observations
    allowed = set(observation_keys)
    filtered = {
        name: value
        for name, value in observations.items()
        if name.split("/", 1)[0] in allowed
    }
    if not filtered:
        available = ", ".join(str(name) for name in observations)
        requested = ", ".join(observation_keys)
        raise ValueError(
            "None of envs.observation_keys matched the Playground observation: "
            f"requested {requested or '<none>'}; "
            f"available {available or '<none>'}."
        )
    return filtered


def _is_eval_mode(mode):
    return str(mode).lower() == "eval"


def _configured_eval_seed_bank(env_config, full_task):
    bank = _get(env_config, "eval_seed_bank", None)
    if bank is None:
        return None
    if isinstance(bank, Mapping):
        _, task_name = full_task.split("_", 1)
        for key in (full_task, task_name, "default"):
            if key in bank:
                bank = bank[key]
                break
        else:
            choices = ", ".join(str(key) for key in bank)
            raise ValueError(
                f"envs.eval_seed_bank has no entry for {full_task!r}; "
                f"available entries: {choices or '<none>'}."
            )
    if isinstance(bank, (str, bytes)):
        raise TypeError("envs.eval_seed_bank must be a sequence of integer seeds.")
    try:
        seeds = tuple(int(seed) for seed in bank)
    except TypeError as exc:
        raise TypeError(
            "envs.eval_seed_bank must be a sequence of integer seeds."
        ) from exc
    if not seeds:
        raise ValueError("envs.eval_seed_bank must not be empty when configured.")
    if any(seed < 0 for seed in seeds):
        raise ValueError("envs.eval_seed_bank seeds must be nonnegative integers.")
    return seeds


def _evaluation_seeds(env_config, fallback_seed, full_task, count, *, start=0):
    count = int(count)
    start = int(start)
    if count < 0 or start < 0:
        raise ValueError(f"seed count and start must be nonnegative, got {count}, {start}")
    bank = _configured_eval_seed_bank(env_config, full_task)
    if bank is not None:
        stop = start + count
        if stop > len(bank):
            raise ValueError(
                f"envs.eval_seed_bank for {full_task!r} contains {len(bank)} "
                f"seeds but indices through {stop - 1} were requested."
            )
        return bank[start:stop]
    base = int(_get(env_config, "seed", fallback_seed))
    offset = int(_get(env_config, "eval_seed_offset", 1_000_000))
    first = base + offset + start
    if first < 0:
        raise ValueError(
            f"Evaluation seed must be nonnegative, got base={base}, offset={offset}."
        )
    return tuple(first + index for index in range(count))


def _env_seed(env_config, fallback_seed, full_task, mode, index):
    index = int(index)
    if _is_eval_mode(mode):
        return _evaluation_seeds(
            env_config, fallback_seed, full_task, 1, start=index
        )[0]
    return int(_get(env_config, "seed", fallback_seed)) + index


def _as_size(value, default=(64, 64)):
    value = value if value is not None else default
    return tuple(int(x) for x in value)


class TimeLimit:
    def __init__(self, env, duration):
        self.env = env
        self.duration = int(duration)
        self.step_count = None

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, *args, **kwargs):
        self.step_count = 0
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        assert self.step_count is not None, "Must reset environment before stepping."
        result = self.env.step(action)
        obs, reward, done, info = _unpack_step(result)
        self.step_count += 1
        if self.duration and self.step_count >= self.duration:
            environment_done = bool(done)
            done = True
            obs["is_last"] = True
            info = dict(info)
            info.setdefault("discount", np.array(1.0, dtype=np.float32))
            if not environment_done:
                info["is_terminal"] = False
            self.step_count = None
        return obs, reward, done, info

    def close(self):
        close = getattr(self.env, "close", None)
        if close is not None:
            close()


class GymToEmbodied:
    def __init__(
        self,
        env,
        elements,
        embodied,
        *,
        obs_key="image",
        act_key="action",
    ):
        del embodied
        self._env = env
        self._elements = elements
        self._obs_key = obs_key
        self._act_key = act_key
        self._obs_dict = hasattr(env.observation_space, "spaces")
        self._act_dict = hasattr(env.action_space, "spaces")
        self._done = True
        self._info = {}

    @property
    def env(self):
        return self._env

    @functools.cached_property
    def obs_space(self):
        if self._obs_dict:
            spaces = self._flatten(self._env.observation_space.spaces)
        else:
            spaces = {self._obs_key: self._env.observation_space}
        spaces = {key: self._convert_space(space) for key, space in spaces.items()}
        base = {
            "reward": self._elements.Space(np.float32),
            "is_first": self._elements.Space(bool),
            "is_last": self._elements.Space(bool),
            "is_terminal": self._elements.Space(bool),
        }
        base.update({
            name: self._elements.Space(np.float32)
            for name in _PLAYGROUND_COUNTER_KEYS
        })
        return {**spaces, **base}

    @functools.cached_property
    def act_space(self):
        if self._act_dict:
            spaces = self._flatten(self._env.action_space.spaces)
        else:
            spaces = {self._act_key: self._env.action_space}
        spaces = {key: self._convert_space(space) for key, space in spaces.items()}
        spaces["reset"] = self._elements.Space(bool)
        return spaces

    def step(self, action):
        if bool(action["reset"]) or self._done:
            self._done = False
            obs = _unpack_reset(self._env.reset())
            return self._obs(obs, 0.0, is_first=True)

        raw_action = self._unflatten(action) if self._act_dict else action[self._act_key]
        raw_action = self._convert_action(raw_action, self._env.action_space)
        obs, reward, self._done, self._info = _unpack_step(self._env.step(raw_action))
        terminal = bool(self._info.get("is_terminal", obs.get("is_terminal", self._done)))
        return self._obs(obs, reward, is_last=bool(self._done), is_terminal=terminal)

    def close(self):
        close = getattr(self._env, "close", None)
        if close is not None:
            close()

    def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        if not self._obs_dict:
            obs = {self._obs_key: obs}
        obs = dict(self._flatten(obs))
        obs = {key: np.asarray(value) for key, value in obs.items()}
        is_first = bool(obs.get("is_first", is_first))
        is_last = bool(obs.get("is_last", is_last))
        is_terminal = bool(obs.get("is_terminal", is_terminal))

        observation_finite = True
        metric_finite = True
        for key, value in obs.items():
            if not np.issubdtype(value.dtype, np.inexact):
                continue
            finite = bool(np.isfinite(value).all())
            if key.startswith("log/"):
                metric_finite &= finite
            elif key not in {"reward", "is_first", "is_last", "is_terminal"}:
                observation_finite &= finite
            if not finite:
                obs[key] = np.nan_to_num(
                    value, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        reward_array = np.asarray(reward)
        reward_finite = bool(np.isfinite(reward_array).all())
        safe_reward = np.nan_to_num(
            reward_array, copy=True, nan=0.0, posinf=0.0, neginf=0.0
        ).reshape(-1)[0]
        invalid_state = not (observation_finite and reward_finite)
        if invalid_state:
            is_last = True
            is_terminal = True
            self._done = True
        elif is_last:
            self._done = True

        counters = {
            "log/termination/environment": (
                is_last and is_terminal and not invalid_state
            ),
            "log/termination/time_limit": (
                is_last and not is_terminal and not invalid_state
            ),
            "log/termination/nonfinite": is_last and invalid_state,
            "log/reset/nonfinite": is_last and invalid_state,
            "log/replacement/nonfinite_observation": not observation_finite,
            "log/replacement/nonfinite_reward": not reward_finite,
            "log/replacement/nonfinite_metric": not metric_finite,
        }
        obs.update(
            reward=np.float32(safe_reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
        )
        obs.update({key: np.float32(value) for key, value in counters.items()})
        return obs

    def _flatten(self, nest, prefix=None):
        result = {}
        items = nest.items() if isinstance(nest, Mapping) else {}
        for key, value in items:
            key = f"{prefix}/{key}" if prefix else str(key)
            if hasattr(value, "spaces"):
                value = value.spaces
            if isinstance(value, Mapping):
                result.update(self._flatten(value, key))
            else:
                result[key] = value
        return result

    def _unflatten(self, flat):
        result = {}
        for key, value in flat.items():
            if key == "reset":
                continue
            parts = key.split("/")
            node = result
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return result

    def _convert_space(self, space):
        if hasattr(space, "n"):
            return self._elements.Space(np.int32, (), 0, int(space.n))
        if hasattr(space, "nvec"):
            return self._elements.Space(np.int32, tuple(space.nvec.shape), 0, space.nvec)
        return self._elements.Space(space.dtype, space.shape, space.low, space.high)

    def _convert_action(self, action, space):
        if hasattr(space, "n"):
            return int(np.asarray(action).item())
        if hasattr(space, "spaces") and isinstance(action, Mapping):
            return {
                key: self._convert_action(action[key], subspace)
                for key, subspace in space.spaces.items()
            }
        return np.asarray(action, dtype=getattr(space, "dtype", None))


class _CraftaxBatchState(NamedTuple):
    env_state: object
    obs: object
    reward: object
    done: object
    info: object
    rng: object


class BatchedCraftaxEnv:
    """Device-resident vectorized adapter for symbolic Craftax environments."""

    def __init__(
        self,
        task_name,
        elements,
        *,
        num_envs,
        seed=0,
        time_limit=1000,
        action_repeat=1,
        observation_mode=None,
        logs=True,
        optimistic_reset_ratio=16,
        reset_seed_bank=None,
        size=(64, 64),
        render=False,
    ):
        try:
            import jax
            import jax.numpy as jnp
            from craftax.craftax_env import make_craftax_env_from_name
        except ImportError as exc:
            raise RuntimeError(
                "Batched Craftax environments require `craftax==1.6.1` and JAX."
            ) from exc

        craftax_module = _import_env_module("craftax")
        env_name = craftax_module._env_name(task_name, observation_mode)
        if "Symbolic" not in env_name:
            raise NotImplementedError(
                "Native device batching currently supports symbolic Craftax only; "
                f"got {env_name}."
            )
        if int(action_repeat) != 1:
            raise NotImplementedError(
                "Native batched Craftax currently requires action_repeat=1."
            )

        self._elements = elements
        self._jax = jax
        self._jnp = jnp
        self._env_name = env_name
        self._has_score = env_name.startswith("Craftax-Classic-")
        self._env = make_craftax_env_from_name(env_name, auto_reset=False)
        self._params = self._env.default_params
        self._num_envs = int(num_envs)
        if self._num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self._time_limit = int(time_limit)
        self._action_repeat = 1
        self._logs = bool(logs)
        self._render_size = _as_size(size)
        self._render_video = bool(render)
        self._achievement_names = craftax_module._achievement_names(env_name)
        self._optimistic_reset_ratio = max(1, int(optimistic_reset_ratio))
        self._reset_candidates = max(
            1,
            (self._num_envs + self._optimistic_reset_ratio - 1)
            // self._optimistic_reset_ratio,
        )
        with _jax_transfer_scope(jax, "explicit_device_put_scope"):
            self._rng = jax.random.PRNGKey(int(seed))
            if reset_seed_bank is None:
                self._fixed_reset_keys = None
            else:
                reset_seed_bank = tuple(int(value) for value in reset_seed_bank)
                if len(reset_seed_bank) != self._num_envs:
                    raise ValueError(
                        "reset_seed_bank must contain one seed per environment; "
                        f"got {len(reset_seed_bank)} for {self._num_envs} environments."
                    )
                self._fixed_reset_keys = jnp.stack([
                    jax.random.PRNGKey(value) for value in reset_seed_bank
                ])

        self._reset_batched = jax.vmap(self._env.reset, in_axes=(0, None))
        self._step_batched = jax.vmap(
            self._env.step, in_axes=(0, 0, 0, None))
        self._reset = jax.jit(
            lambda keys: self._reset_batched(keys, self._params))
        self._render_env = None
        self._render_state = None
        if self._render_video:
            pixel_env_name = env_name.replace("Symbolic", "Pixels")
            craftax_module._configure_texture_cache(pixel_env_name)
            self._render_env = make_craftax_env_from_name(
                pixel_env_name, auto_reset=False
            )
            render_state = getattr(self._render_env, "get_obs", None)
            if not callable(render_state):
                raise RuntimeError(
                    f"{pixel_env_name} does not expose get_obs() for video rendering."
                )
            self._render_state = render_state

        obs_space = self._env.observation_space(self._params)
        self._obs_shape = tuple(int(value) for value in obs_space.shape)
        action_space = self._env.action_space(self._params)
        action_count = getattr(action_space, "n", None)
        if action_count is None:
            action_count = getattr(self._env, "num_actions", None)
        if action_count is None:
            raise RuntimeError(f"Could not infer action count for {env_name}.")
        self._action_count = int(action_count)

    def __len__(self):
        return self._num_envs

    @functools.cached_property
    def obs_space(self):
        spaces = {
            "vector": self._elements.Space(
                np.float32, self._obs_shape, 0.0, 1.0),
            "reward": self._elements.Space(np.float32),
            "is_first": self._elements.Space(bool),
            "is_last": self._elements.Space(bool),
            "is_terminal": self._elements.Space(bool),
        }
        if self._logs:
            if self._has_score:
                spaces["log/score"] = self._elements.Space(np.float32)
            spaces.update({
                f"log/achievement_{name}": self._elements.Space(np.float32)
                for name in self._achievement_names
            })
        if self._render_video:
            spaces["log/image"] = self._elements.Space(
                np.uint8, self._render_size + (3,), 0, 255
            )
        return spaces

    @functools.cached_property
    def act_space(self):
        return {
            "action": self._elements.Space(
                np.int32, (), 0, self._action_count),
            "reset": self._elements.Space(bool),
        }

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def time_limit(self):
        return self._time_limit

    @property
    def action_repeat(self):
        return self._action_repeat

    @property
    def terminal_on_time_limit(self):
        # Craftax includes its built-in horizon in is_terminal() and returns a
        # zero discount there, so match the host adapter at coincident limits.
        return True

    def close(self):
        pass

    def rng_device(self):
        return self._rng

    def fixed_reset_keys_device(self):
        """Return immutable per-lane evaluation reset keys, if configured."""
        return self._fixed_reset_keys

    def split_keys_device(self, rng):
        with _jax_transfer_scope(self._jax, "explicit_device_put_scope"):
            keys = self._jax.random.split(rng, self._num_envs + 1)
            return keys[0], keys[1:]

    def reset_device(self, keys):
        reset_keys, next_rng = self._split_env_keys(keys)
        obs, env_state = self._call_jax(self._reset, reset_keys)
        return _CraftaxBatchState(
            env_state=env_state,
            obs=self._jnp.asarray(obs, dtype=self._jnp.float32),
            reward=self._jnp.zeros((self._num_envs,), self._jnp.float32),
            done=self._jnp.zeros((self._num_envs,), dtype=bool),
            info=self._empty_info(self._num_envs),
            rng=next_rng,
        )

    def step_device(self, state, action):
        step_keys, next_rng = self._split_env_keys(state.rng)
        obs, env_state, reward, done, info = self._step_batched(
            step_keys,
            state.env_state,
            self._jnp.asarray(action, dtype=self._jnp.int32),
            self._params,
        )
        return _CraftaxBatchState(
            env_state=env_state,
            obs=self._jnp.asarray(obs, dtype=self._jnp.float32),
            reward=self._jnp.asarray(reward, dtype=self._jnp.float32),
            done=self._jnp.asarray(done, dtype=bool),
            info={
                key: self._jnp.asarray(value, dtype=self._jnp.float32)
                for key, value in info.items()
            },
            rng=next_rng,
        )

    def first_state_device(self, state):
        reset_keys, next_rng = self._split_env_keys(state.rng)
        candidate_keys = reset_keys[:self._reset_candidates]
        candidate_obs, candidate_state = self._reset_batched(
            candidate_keys, self._params)
        indices = self._jnp.arange(self._num_envs) % self._reset_candidates
        obs = self._jax.tree.map(lambda value: value[indices], candidate_obs)
        env_state = self._jax.tree.map(
            lambda value: value[indices], candidate_state)
        return _CraftaxBatchState(
            env_state=env_state,
            obs=self._jnp.asarray(obs, dtype=self._jnp.float32),
            reward=self._jnp.zeros((self._num_envs,), self._jnp.float32),
            done=self._jnp.zeros((self._num_envs,), dtype=bool),
            info=self._empty_info(self._num_envs),
            rng=next_rng,
        )

    def select_reset_device(self, reset, reset_state, stepped_state):
        def select(reset_value, step_value):
            mask = reset.reshape(
                (self._num_envs,) + (1,) * (step_value.ndim - 1))
            return self._jnp.where(mask, reset_value, step_value)

        return self._jax.tree.map(select, reset_state, stepped_state)

    def obs_from_state_device(
        self, state, reward, is_first, is_last, is_terminal
    ):
        obs = {
            "vector": self._jnp.asarray(state.obs, dtype=self._jnp.float32),
            "reward": self._jnp.asarray(reward, dtype=self._jnp.float32),
            "is_first": self._jnp.asarray(is_first, dtype=bool),
            "is_last": self._jnp.asarray(is_last, dtype=bool),
            "is_terminal": self._jnp.asarray(is_terminal, dtype=bool),
        }
        return obs

    def diagnostics_from_state_device(
        self, state, is_last=None, is_terminal=None
    ):
        del is_last, is_terminal
        diagnostics = {}
        if self._logs:
            if self._has_score:
                diagnostics["log/score"] = state.info.get(
                    "score",
                    self._jnp.zeros(
                        (self._num_envs,), dtype=self._jnp.float32),
                )
            for name in self._achievement_names:
                diagnostics[f"log/achievement_{name}"] = state.info[
                    f"Achievements/{name}"]
        if self._render_state is not None:
            diagnostics["log/image"] = self._render_frames_device(state)
        return diagnostics

    def _render_frames_device(self, state):
        first_state = self._jax.tree.map(
            lambda value: value[0], state.env_state
        )
        image = self._jnp.asarray(
            self._render_state(first_state), dtype=self._jnp.float32
        )
        image = self._jnp.clip(image * 255.0, 0.0, 255.0).astype(
            self._jnp.uint8
        )
        height, width = self._render_size
        if tuple(image.shape[:2]) != (height, width):
            yidx = self._jnp.minimum(
                self._jnp.arange(height) * image.shape[0] // height,
                image.shape[0] - 1,
            )
            xidx = self._jnp.minimum(
                self._jnp.arange(width) * image.shape[1] // width,
                image.shape[1] - 1,
            )
            image = image[yidx][:, xidx]
        frames = self._jnp.zeros(
            (self._num_envs, height, width, 3), dtype=self._jnp.uint8
        )
        return frames.at[0].set(image)

    def finite_state_device(self, state):
        vector = self._jnp.asarray(state.obs, dtype=self._jnp.float32)
        return (
            self._jnp.isfinite(vector.reshape((self._num_envs, -1))).all(-1)
            & self._jnp.isfinite(state.reward)
        )

    def device_get_scope(self):
        return _jax_transfer_scope(self._jax, "explicit_device_get_scope")

    def device_put_scope(self):
        return _jax_transfer_scope(self._jax, "explicit_device_put_scope")

    def _split_env_keys(self, keys):
        pairs = self._jax.vmap(self._jax.random.split)(keys)
        return pairs[:, 0], pairs[:, 1]

    def _empty_info(self, count):
        info = {
            "discount": self._jnp.ones((count,), dtype=self._jnp.float32)
        }
        if self._env_name.startswith("Craftax-Classic-"):
            info["score"] = self._jnp.zeros(
                (count,), dtype=self._jnp.float32)
        info.update({
            f"Achievements/{name}": self._jnp.zeros(
                (count,), dtype=self._jnp.float32)
            for name in self._achievement_names
        })
        return info

    def _call_jax(self, fn, *args):
        with _jax_transfer_scope(self._jax, "explicit_device_get_scope"):
            with _jax_transfer_scope(self._jax, "explicit_device_put_scope"):
                return fn(*args)


class MujocoPlaygroundEnv:
    def __init__(
        self,
        suite,
        task_name,
        elements,
        *,
        seed=0,
        time_limit=1000,
        action_repeat=1,
        include_image=False,
        size=(64, 64),
        observation_keys=None,
        fixed_reset=False,
    ):
        try:
            import jax
            import jax.numpy as jnp

            playground_suite = _load_playground_suite(suite)
        except ImportError as exc:
            raise RuntimeError(
                f"envs.backend=mjx for {suite}_{task_name} requires MuJoCo "
                "Playground. Install it in the JAX environment with "
                "`pip install playground`."
            ) from exc

        playground_task = _resolve_playground_task(suite, task_name, playground_suite)
        if playground_task is None:
            raise NotImplementedError(
                f"Task '{suite}_{task_name}' is not available in the installed "
                f"MuJoCo Playground {suite} registry. Supported local aliases "
                f"are: {', '.join(_supported_playground_task_names(suite))}."
            )

        self._suite = suite
        self._task_name = task_name
        self._playground_task = playground_task
        self._elements = elements
        self._jax = jax
        self._jnp = jnp
        self._include_image = bool(include_image)
        self._size = _as_size(size)
        self._observation_keys = _as_playground_observation_keys(
            observation_keys
        )
        env_config = _playground_config(
            playground_suite,
            playground_task,
            time_limit=time_limit,
            include_image=self._include_image,
            size=self._size,
            render_batch_size=1,
        )
        with _jax_transfer_scope(jax, "explicit_device_put_scope"):
            try:
                self._env = playground_suite.load(playground_task, config=env_config)
            except NotImplementedError as exc:
                if self._include_image:
                    raise NotImplementedError(
                        "Cannot use MuJoCo Playground vision for "
                        f"{suite}_{task_name}: {exc}"
                    ) from exc
                raise
        with _jax_transfer_scope(jax, "explicit_device_put_scope"):
            self._metric_names = _playground_scalar_metric_names(jax, self._env)
        _validate_playground_vision_env(self._env, suite, task_name, self._include_image)
        with _jax_transfer_scope(jax, "explicit_device_put_scope"):
            self._rng = jax.random.PRNGKey(int(seed))
            self._fixed_reset_key = (
                jax.random.PRNGKey(int(seed)) if fixed_reset else None
            )
        self._state = None
        self._done = True
        self._time_limit = int(time_limit)
        self._step_count = 0
        self._action_repeat = max(1, int(action_repeat))
        self._reset = jax.jit(self._env.reset)
        self._step = jax.jit(self._repeat_step)

    @functools.cached_property
    def obs_space(self):
        if self._include_image:
            spaces = {
                "image": self._elements.Space(
                    np.uint8, self._size + (3,), 0, 255
                )
            }
        else:
            with _jax_transfer_scope(self._jax, "explicit_device_put_scope"):
                observation_size = self._env.observation_size
            spaces = _filter_playground_observations(
                self._obs_spaces(observation_size),
                getattr(self, "_observation_keys", None),
            )
        base = {
            "reward": self._elements.Space(np.float32),
            "is_first": self._elements.Space(bool),
            "is_last": self._elements.Space(bool),
            "is_terminal": self._elements.Space(bool),
        }
        base.update({
            f"log/env/{name}": self._elements.Space(np.float32)
            for name in self._metric_names
        })
        base.update({
            name: self._elements.Space(np.float32)
            for name in _PLAYGROUND_COUNTER_KEYS
        })
        return {**spaces, **base}

    @functools.cached_property
    def act_space(self):
        return {
            "action": self._elements.Space(
                np.float32,
                (int(self._env.action_size),),
                -1.0,
                1.0,
            ),
            "reset": self._elements.Space(bool),
        }

    def step(self, action):
        if bool(action["reset"]) or self._done or self._state is None:
            if self._fixed_reset_key is None:
                with _jax_transfer_scope(self._jax, "explicit_device_put_scope"):
                    keys = self._jax.random.split(self._rng, 2)
                    self._rng, key = keys[0], keys[1]
            else:
                key = self._fixed_reset_key
            self._state = self._call_jax(self._reset, key)
            self._done = False
            self._step_count = 0
            obs = self._obs(0.0, is_first=True)
            self._done = bool(obs["is_last"])
            return obs

        act = np.asarray(action["action"], dtype=np.float32)
        act = np.clip(act, -1.0, 1.0)
        with _jax_transfer_scope(self._jax, "explicit_device_put_scope"):
            act = self._jax.device_put(act)
        self._state = self._call_jax(self._step, self._state, act)
        with _jax_transfer_scope(self._jax, "explicit_device_get_scope"):
            reward = float(np.asarray(self._jax.device_get(self._state.reward)))
            env_done = bool(np.asarray(self._jax.device_get(self._state.done)))
        self._step_count += self._action_repeat
        time_done = self._time_limit > 0 and self._step_count >= self._time_limit
        is_last = env_done or time_done
        is_terminal = env_done and not time_done
        obs = self._obs(
            reward, is_last=is_last, is_terminal=is_terminal)
        self._done = bool(obs["is_last"])
        return obs

    def close(self):
        pass

    def _repeat_step(self, state, action):
        if self._action_repeat == 1:
            return self._env.step(state, action)

        def body(carry, _):
            next_state = self._env.step(carry, action)
            return next_state, next_state.reward

        state, rewards = self._jax.lax.scan(
            body, state, (), length=self._action_repeat)
        return state.replace(reward=self._jnp.sum(rewards, axis=0))

    def _obs(
        self, reward, is_first=False, is_last=False, is_terminal=False
    ):
        with _jax_transfer_scope(self._jax, "explicit_device_get_scope"):
            raw_obs, raw_metrics, raw_reward, raw_qpos, raw_qvel = (
                self._jax.device_get((
                    self._state.obs,
                    self._state.metrics,
                    self._state.reward,
                    self._state.data.qpos,
                    self._state.data.qvel,
                ))
            )
        observation_finite = all(
            not np.issubdtype(np.asarray(value).dtype, np.inexact)
            or np.isfinite(value).all()
            for value in _flatten_playground_obs(raw_obs).values()
        )
        reward_finite = bool(np.isfinite(raw_reward).all())
        kinematics_finite = bool(
            np.isfinite(raw_qpos).all() and np.isfinite(raw_qvel).all())
        metric_finite = all(
            np.isfinite(np.asarray(raw_metrics[name])).all()
            for name in self._metric_names
        )
        invalid_state = not (
            observation_finite and reward_finite and kinematics_finite)
        if invalid_state:
            is_last = True
            is_terminal = True
        if self._include_image:
            obs = {"image": _image_to_uint8_np(_extract_playground_image(raw_obs))}
        else:
            obs = _filter_playground_observations(
                self._flatten_obs(raw_obs),
                getattr(self, "_observation_keys", None),
            )
        obs.update(
            reward=np.float32(
                np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)),
            is_first=bool(is_first),
            is_last=bool(is_last),
            is_terminal=bool(is_terminal),
        )
        obs.update(_playground_metric_logs_np(raw_metrics, self._metric_names))
        counters = {
            "log/termination/environment": (
                is_last and is_terminal and not invalid_state
            ),
            "log/termination/time_limit": (
                is_last and not is_terminal and not invalid_state
            ),
            "log/termination/nonfinite": is_last and invalid_state,
            "log/reset/nonfinite": is_last and invalid_state,
            "log/replacement/nonfinite_observation": not observation_finite,
            "log/replacement/nonfinite_reward": not reward_finite,
            "log/replacement/nonfinite_metric": not metric_finite,
        }
        obs.update({key: np.float32(value) for key, value in counters.items()})
        return obs

    def _call_jax(self, fn, *args):
        # MJX reset/step compilation can materialize closed-over model arrays.
        # Keep those transfers explicit so Embodied's global transfer guard stays on.
        with _jax_transfer_scope(self._jax, "explicit_device_get_scope"):
            with _jax_transfer_scope(self._jax, "explicit_device_put_scope"):
                return fn(*args)

    def _obs_spaces(self, size, prefix=None):
        if isinstance(size, Mapping):
            result = {}
            for key, value in size.items():
                name = f"{prefix}/{key}" if prefix else str(key)
                result.update(self._obs_spaces(value, name))
            return result
        shape = _shape_tuple(size)
        key = prefix or "vector"
        return {key: self._elements.Space(np.float32, shape, -np.inf, np.inf)}

    def _flatten_obs(self, obs, prefix=None):
        if isinstance(obs, Mapping):
            result = {}
            for key, value in obs.items():
                name = f"{prefix}/{key}" if prefix else str(key)
                result.update(self._flatten_obs(value, name))
            return result
        key = prefix or "vector"
        value = np.asarray(obs, dtype=np.float32)
        value = np.nan_to_num(
            value, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        return {key: value}


class BatchedMujocoPlaygroundEnv:
    def __init__(
        self,
        suite,
        task_name,
        elements,
        *,
        num_envs,
        seed=0,
        time_limit=1000,
        action_repeat=1,
        include_image=False,
        size=(64, 64),
        observation_keys=None,
        reset_seed_bank=None,
    ):
        try:
            import jax
            import jax.numpy as jnp

            playground_suite = _load_playground_suite(suite)
        except ImportError as exc:
            raise RuntimeError(
                f"envs.backend=mjx for {suite}_{task_name} requires MuJoCo "
                "Playground. Install it in the JAX environment with "
                "`pip install playground`."
            ) from exc

        playground_task = _resolve_playground_task(suite, task_name, playground_suite)
        if playground_task is None:
            raise NotImplementedError(
                f"Task '{suite}_{task_name}' is not available in the installed "
                f"MuJoCo Playground {suite} registry. Supported local aliases "
                f"are: {', '.join(_supported_playground_task_names(suite))}."
            )

        self._suite = suite
        self._task_name = task_name
        self._playground_task = playground_task
        self._num_envs = int(num_envs)
        self._elements = elements
        self._jax = jax
        self._jnp = jnp
        self._include_image = bool(include_image)
        self._size = _as_size(size)
        self._observation_keys = _as_playground_observation_keys(
            observation_keys
        )
        env_config = _playground_config(
            playground_suite,
            playground_task,
            time_limit=time_limit,
            include_image=self._include_image,
            size=self._size,
            render_batch_size=self._num_envs,
        )
        with _jax_transfer_scope(jax, "explicit_device_put_scope"):
            try:
                self._env = playground_suite.load(playground_task, config=env_config)
            except NotImplementedError as exc:
                if self._include_image:
                    raise NotImplementedError(
                        "Cannot use MuJoCo Playground vision for "
                        f"{suite}_{task_name}: {exc}"
                    ) from exc
                raise
            self._metric_names = _playground_scalar_metric_names(jax, self._env)
            _validate_playground_vision_env(self._env, suite, task_name, self._include_image)
            if self._include_image:
                wrapper_module = _load_playground_wrapper()
                madrona_wrapper = getattr(wrapper_module, "MadronaWrapper", None)
                if madrona_wrapper is not None:
                    self._env = madrona_wrapper(self._env, self._num_envs)
            self._rng = jax.random.PRNGKey(int(seed))
            if reset_seed_bank is None:
                self._fixed_reset_keys = None
            else:
                reset_seed_bank = tuple(int(value) for value in reset_seed_bank)
                if len(reset_seed_bank) != self._num_envs:
                    raise ValueError(
                        "reset_seed_bank must contain one seed per environment; "
                        f"got {len(reset_seed_bank)} for {self._num_envs} environments."
                    )
                self._fixed_reset_keys = jnp.stack([
                    jax.random.PRNGKey(value) for value in reset_seed_bank
                ])
        self._state = None
        self._time_limit = int(time_limit)
        self._action_repeat = max(1, int(action_repeat))
        if self._include_image and hasattr(self._env, "num_worlds"):
            self._reset_batched = self._env.reset
            self._step_batched_once = self._env.step
        else:
            self._reset_batched = jax.vmap(self._env.reset)
            self._step_batched_once = jax.vmap(self._env.step)
        self._step_batched = self._repeat_step_batched
        self._reset = jax.jit(self._reset_batched)
        self._step = jax.jit(self._step_batched)
        self._transition_metadata = jax.jit(self._transition_metadata_device)
        self._diagnostics = jax.jit(self.diagnostics_from_state_device)

    def __len__(self):
        return self._num_envs

    @functools.cached_property
    def obs_space(self):
        if self._include_image:
            spaces = {
                "image": self._elements.Space(
                    np.uint8, self._size + (3,), 0, 255
                )
            }
        else:
            with _jax_transfer_scope(self._jax, "explicit_device_put_scope"):
                observation_size = self._env.observation_size
            spaces = _filter_playground_observations(
                self._obs_spaces(observation_size),
                getattr(self, "_observation_keys", None),
            )
        base = {
            "reward": self._elements.Space(np.float32),
            "is_first": self._elements.Space(bool),
            "is_last": self._elements.Space(bool),
            "is_terminal": self._elements.Space(bool),
        }
        base.update({
            f"log/env/{name}": self._elements.Space(np.float32)
            for name in self._metric_names
        })
        base.update({
            name: self._elements.Space(np.float32)
            for name in _PLAYGROUND_COUNTER_KEYS
        })
        return {**spaces, **base}

    @functools.cached_property
    def act_space(self):
        return {
            "action": self._elements.Space(
                np.float32,
                (int(self._env.action_size),),
                -1.0,
                1.0,
            ),
            "reset": self._elements.Space(bool),
        }

    def step(self, action):
        reset = np.asarray(action["reset"], dtype=bool)
        if reset.shape == ():
            reset = np.full((self._num_envs,), bool(reset), dtype=bool)
        reset = reset.reshape(self._num_envs)

        fixed_reset_keys = getattr(self, "_fixed_reset_keys", None)
        if fixed_reset_keys is None:
            self._rng, keys = self._split_keys()
        else:
            # Evaluation lanes use common random numbers across checkpoints.
            # Do not advance the construction RNG or replace this held-out bank.
            keys = fixed_reset_keys
        reset_state = self._call_jax(self._reset, keys)
        if self._state is None:
            self._state = reset_state
            with _jax_transfer_scope(self._jax, "explicit_device_put_scope"):
                self._step_count = self._jnp.zeros(
                    (self._num_envs,), dtype=self._jnp.int32)
                reward = self._jnp.zeros(
                    (self._num_envs,), dtype=self._jnp.float32)
                is_first = self._jnp.ones((self._num_envs,), dtype=bool)
                is_last = self._jnp.zeros((self._num_envs,), dtype=bool)
                is_terminal = self._jnp.zeros((self._num_envs,), dtype=bool)
            return self._obs(
                reward, is_first, is_last, is_terminal)

        act = np.asarray(action["action"], dtype=np.float32)
        act = np.clip(act, -1.0, 1.0)
        with _jax_transfer_scope(self._jax, "explicit_device_put_scope"):
            act = self._jax.device_put(act)
            reset_device = self._jax.device_put(reset)
        stepped = self._call_jax(self._step, self._state, act)
        (
            reward_device,
            is_last_device,
            is_terminal_device,
            self._step_count,
        ) = self._call_jax(
            self._transition_metadata,
            stepped,
            reset_device,
            self._step_count,
        )
        self._state = self._select_reset(reset_device, reset_state, stepped)
        return self._obs(
            reward_device, reset_device, is_last_device, is_terminal_device)

    def close(self):
        pass

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def time_limit(self):
        return self._time_limit

    @property
    def action_repeat(self):
        return self._action_repeat

    def rng_device(self):
        return self._rng

    def fixed_reset_keys_device(self):
        """Return immutable per-lane evaluation reset keys, if configured."""
        return getattr(self, "_fixed_reset_keys", None)

    def split_keys_device(self, rng):
        with _jax_transfer_scope(self._jax, "explicit_device_put_scope"):
            keys = self._jax.random.split(rng, self._num_envs + 1)
            return keys[0], keys[1:]

    def reset_device(self, keys):
        state = self._call_jax(self._reset, keys)
        return self._with_first_state_device(state)

    def reset_from_rng_device(self, rng):
        """Return a freshly sampled batched reset state and advanced RNG."""
        fixed_reset_keys = getattr(self, "_fixed_reset_keys", None)
        if fixed_reset_keys is not None:
            return rng, self.reset_device(fixed_reset_keys)
        rng, keys = self.split_keys_device(rng)
        return rng, self.reset_device(keys)

    def step_device(self, state, action):
        action = self._jnp.clip(action, -1.0, 1.0)
        return self._step_batched(state, action)

    def _repeat_step_batched(self, state, action):
        if self._action_repeat == 1:
            return self._step_batched_once(state, action)

        def body(carry, _):
            next_state = self._step_batched_once(carry, action)
            return next_state, next_state.reward

        state, rewards = self._jax.lax.scan(
            body, state, (), length=self._action_repeat)
        return state.replace(reward=self._jnp.sum(rewards, axis=0))

    def first_state_device(self, state):
        return state.replace(
            data=state.info["first_data"],
            obs=state.info["first_obs"],
            reward=self._jnp.zeros_like(state.reward),
            done=self._jnp.zeros_like(state.done),
        )

    def select_reset_device(self, reset, reset_state, stepped_state):
        return self._select_reset(reset, reset_state, stepped_state)

    def obs_from_state_device(self, state, reward, is_first, is_last, is_terminal):
        if self._include_image:
            obs = {
                "image": _image_to_uint8_jnp(
                    _extract_playground_image(state.obs), self._jnp
                )
            }
        else:
            obs = _filter_playground_observations(
                self._flatten_obs_device(state.obs),
                getattr(self, "_observation_keys", None),
            )
        obs.update(
            reward=self._jnp.asarray(reward, dtype=self._jnp.float32),
            is_first=self._jnp.asarray(is_first, dtype=bool),
            is_last=self._jnp.asarray(is_last, dtype=bool),
            is_terminal=self._jnp.asarray(is_terminal, dtype=bool),
        )
        return obs

    def finite_state_device(self, state):
        finite = self._finite_observation_device(state)
        finite &= self._jnp.isfinite(state.reward).reshape((self._num_envs,))
        finite &= self._finite_kinematics_device(state)
        return finite

    def _finite_kinematics_device(self, state):
        finite = self._jnp.ones((self._num_envs,), dtype=bool)
        for name in ("qpos", "qvel"):
            value = self._jnp.asarray(getattr(state.data, name))
            finite &= self._jnp.isfinite(
                value.reshape((self._num_envs, -1))
            ).all(-1)
        return finite

    def diagnostics_from_state_device(
        self, state, is_last, is_terminal
    ):
        """Return diagnostics kept outside policy observations and replay."""
        logs, metrics_finite = self._metric_logs_device(state.metrics)
        obs_finite = self._finite_observation_device(state)
        reward_finite = self._jnp.isfinite(state.reward).reshape((self._num_envs,))
        state_finite = (
            obs_finite & reward_finite & self._finite_kinematics_device(state)
        )
        is_last = self._jnp.asarray(is_last, dtype=bool).reshape((self._num_envs,))
        is_terminal = self._jnp.asarray(
            is_terminal, dtype=bool
        ).reshape((self._num_envs,))
        env_done = self._jnp.asarray(
            state.done, dtype=bool
        ).reshape((self._num_envs,))
        counters = {
            "log/termination/environment": (
                is_last & is_terminal & env_done & state_finite
            ),
            "log/termination/time_limit": is_last & ~is_terminal & state_finite,
            "log/termination/nonfinite": is_last & ~state_finite,
            "log/reset/nonfinite": is_last & ~state_finite,
            "log/replacement/nonfinite_observation": ~obs_finite,
            "log/replacement/nonfinite_reward": ~reward_finite,
            "log/replacement/nonfinite_metric": ~metrics_finite,
        }
        logs.update({
            key: value.astype(self._jnp.float32)
            for key, value in counters.items()
        })
        return logs

    def _transition_metadata_device(self, stepped, reset, step_count):
        """Compute host-batched episode flags with device-driver semantics."""
        reset = self._jnp.asarray(reset, dtype=bool).reshape((self._num_envs,))
        env_done = self._jnp.asarray(
            stepped.done, dtype=bool).reshape((self._num_envs,))
        invalid_state = ~self.finite_state_device(stepped)
        stepped_count = self._jnp.asarray(
            step_count, dtype=self._jnp.int32) + self._action_repeat
        if self._time_limit > 0:
            time_done = stepped_count >= self._time_limit
        else:
            time_done = self._jnp.zeros_like(env_done, dtype=bool)
        active_done = env_done | time_done | invalid_state
        terminal = (env_done & ~time_done) | invalid_state
        reward = self._jnp.where(
            reset, self._jnp.zeros_like(stepped.reward), stepped.reward)
        reward = self._jnp.nan_to_num(
            reward, nan=0.0, posinf=0.0, neginf=0.0)
        is_last = self._jnp.where(
            reset, self._jnp.zeros_like(active_done), active_done)
        is_terminal = self._jnp.where(
            reset, self._jnp.zeros_like(terminal), terminal)
        next_step_count = self._jnp.where(
            reset, self._jnp.zeros_like(stepped_count), stepped_count)
        return reward, is_last, is_terminal, next_step_count


    def _finite_observation_device(self, state):
        finite = self._jnp.ones((self._num_envs,), dtype=bool)
        for value in self._jax.tree.leaves(state.obs):
            value = self._jnp.asarray(value)
            if self._jnp.issubdtype(value.dtype, self._jnp.floating):
                finite &= self._jnp.isfinite(
                    value.reshape((self._num_envs, -1))).all(-1)
        return finite

    def _metric_logs_device(self, metrics):
        finite = self._jnp.ones((self._num_envs,), dtype=bool)
        logs = {}
        for name in self._metric_names:
            value = self._jnp.asarray(
                metrics[name], dtype=self._jnp.float32
            ).reshape((self._num_envs, -1))[:, 0]
            finite &= self._jnp.isfinite(value)
            logs[f"log/env/{name}"] = self._jnp.nan_to_num(
                value, nan=0.0, posinf=0.0, neginf=0.0
            )
        return logs, finite

    def device_get(self, value):
        with _jax_transfer_scope(self._jax, "explicit_device_get_scope"):
            return self._jax.device_get(value)

    def device_get_scope(self):
        return _jax_transfer_scope(self._jax, "explicit_device_get_scope")

    def device_put_scope(self):
        return _jax_transfer_scope(self._jax, "explicit_device_put_scope")

    def _with_first_state_device(self, state):
        return state.replace(
            info={
                **state.info,
                "first_data": state.data,
                "first_obs": state.obs,
            }
        )

    def _split_keys(self):
        with _jax_transfer_scope(self._jax, "explicit_device_put_scope"):
            keys = self._jax.random.split(self._rng, self._num_envs + 1)
            rng, env_keys = keys[0], keys[1:]
        return rng, env_keys

    def _select_reset(self, reset, reset_state, stepped_state):
        def select(reset_value, step_value):
            mask = reset.reshape((self._num_envs,) + (1,) * (step_value.ndim - 1))
            return self._jnp.where(mask, reset_value, step_value)

        return self._jax.tree.map(select, reset_state, stepped_state)

    def _obs(self, reward, is_first, is_last, is_terminal):
        with _jax_transfer_scope(self._jax, "explicit_device_put_scope"):
            reward = self._jnp.asarray(reward, dtype=self._jnp.float32)
            is_first = self._jnp.asarray(is_first, dtype=bool)
            is_last = self._jnp.asarray(is_last, dtype=bool)
            is_terminal = self._jnp.asarray(is_terminal, dtype=bool)
        diagnostics = self._call_jax(
            self._diagnostics, self._state, is_last, is_terminal)
        with _jax_transfer_scope(self._jax, "explicit_device_get_scope"):
            raw_obs, reward, is_first, is_last, is_terminal, diagnostics = (
                self._jax.device_get((
                    self._state.obs,
                    reward,
                    is_first,
                    is_last,
                    is_terminal,
                    diagnostics,
                ))
            )
        if self._include_image:
            obs = {"image": _image_to_uint8_np(_extract_playground_image(raw_obs))}
        else:
            obs = _filter_playground_observations(
                self._flatten_obs(raw_obs),
                getattr(self, "_observation_keys", None),
            )
        obs.update(
            reward=np.asarray(reward, dtype=np.float32),
            is_first=np.asarray(is_first, dtype=bool),
            is_last=np.asarray(is_last, dtype=bool),
            is_terminal=np.asarray(is_terminal, dtype=bool),
        )
        obs.update({
            key: np.asarray(value, dtype=np.float32)
            for key, value in diagnostics.items()
        })
        return obs

    def _call_jax(self, fn, *args):
        with _jax_transfer_scope(self._jax, "explicit_device_get_scope"):
            with _jax_transfer_scope(self._jax, "explicit_device_put_scope"):
                return fn(*args)

    def _obs_spaces(self, size, prefix=None):
        if isinstance(size, Mapping):
            result = {}
            for key, value in size.items():
                name = f"{prefix}/{key}" if prefix else str(key)
                result.update(self._obs_spaces(value, name))
            return result
        shape = _shape_tuple(size)
        key = prefix or "vector"
        return {key: self._elements.Space(np.float32, shape, -np.inf, np.inf)}

    def _flatten_obs(self, obs, prefix=None):
        if isinstance(obs, Mapping):
            result = {}
            for key, value in obs.items():
                name = f"{prefix}/{key}" if prefix else str(key)
                result.update(self._flatten_obs(value, name))
            return result
        key = prefix or "vector"
        value = np.asarray(obs, dtype=np.float32)
        value = np.nan_to_num(value, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        return {key: value}

    def _flatten_obs_device(self, obs, prefix=None):
        if isinstance(obs, Mapping):
            result = {}
            for key, value in obs.items():
                name = f"{prefix}/{key}" if prefix else str(key)
                result.update(self._flatten_obs_device(value, name))
            return result
        key = prefix or "vector"
        value = self._jnp.asarray(obs, dtype=self._jnp.float32)
        value = self._jnp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        return {key: value}


class MujocoPlaygroundDMC(MujocoPlaygroundEnv):
    def __init__(self, task_name, elements, **kwargs):
        super().__init__("dmc", task_name, elements, **kwargs)


class BatchedMujocoPlaygroundDMC(BatchedMujocoPlaygroundEnv):
    def __init__(self, task_name, elements, **kwargs):
        super().__init__("dmc", task_name, elements, **kwargs)


def make_env(config, elements, embodied, index, *, mode="train", task=None):
    env_config = config.envs
    full_task = str(task or env_config.task)
    suite, task_name = full_task.split("_", 1)
    seed = _env_seed(env_config, config.seed, full_task, mode, index)

    if suite == "dmc":
        env = _make_dmc_env(
            config,
            env_config,
            elements,
            task_name,
            seed,
            fixed_reset=_is_eval_mode(mode),
        )

    elif suite in PLAYGROUND_TASKS_BY_SUITE:
        env = _make_playground_env(
            config,
            env_config,
            elements,
            suite,
            task_name,
            seed,
            fixed_reset=_is_eval_mode(mode),
        )

    elif suite == "dmlab":
        dmlab = _import_env_module("dmlab")

        env = dmlab.DeepMindLabyrinth(
            task_name,
            mode=mode,
            action_repeat=int(_get(env_config, "action_repeat", 4)),
            render_size=_as_size(_get(env_config, "size", (64, 64))),
            action_set=_get(env_config, "dmlab_action_set", "popart"),
            seed=seed,
            runfiles_path=_get(env_config, "dmlab_runfiles_path", None),
        )
    elif suite == "atari":
        atari = _import_env_module("atari")

        env = atari.Atari(
            task_name,
            action_repeat=int(_get(env_config, "action_repeat", 4)),
            size=_as_size(_get(env_config, "size", (64, 64))),
            gray=bool(_get(env_config, "gray", False)),
            noops=int(_get(env_config, "noops", 30)),
            lives=_get(env_config, "lives", "unused"),
            sticky=bool(_get(env_config, "sticky", False)),
            actions=_get(env_config, "actions", "needed"),
            length=int(_get(env_config, "time_limit", 108000)),
            pooling=int(_get(env_config, "pooling", 2)),
            aggregate=_get(env_config, "aggregate", "max"),
            resize=_get(env_config, "resize", "pillow"),
            autostart=bool(_get(env_config, "autostart", False)),
            clip_reward=bool(_get(env_config, "clip_reward", False)),
            seed=seed,
        )
    elif suite == "crafter":
        crafter = _import_env_module("crafter")

        env = crafter.Crafter(task_name, _as_size(_get(env_config, "size", (64, 64))), seed=seed)
        env = TimeLimit(env, int(_get(env_config, "time_limit", 1000)))
    elif suite == "craftax":
        craftax = _import_env_module("craftax")
        trainer = _get(config, "trainer", {})

        env = craftax.Craftax(
            task_name,
            size=_as_size(_get(env_config, "size", (64, 64))),
            seed=seed,
            observation_mode=_get(env_config, "observation_mode", None),
            include_image=bool(_get(env_config, "include_image", False)),
            logs=bool(_get(env_config, "logs", True)),
            render=bool(
                _get(
                    trainer,
                    "video_policy_log",
                    _get(env_config, "video_policy_log", False),
                )
            ),
        )
        env = TimeLimit(env, int(_get(env_config, "time_limit", 1000)))
    elif suite == "memorymaze":
        MemoryMaze = _import_env_module("memorymaze").MemoryMaze

        env = MemoryMaze(task_name, seed=seed)
        env = TimeLimit(env, int(_get(env_config, "time_limit", 4000)))
    elif suite == "metaworld":
        metaworld = _import_env_module("metaworld")

        env = metaworld.MetaWorld(
            task_name,
            int(_get(env_config, "action_repeat", 2)),
            _as_size(_get(env_config, "size", (64, 64))),
            _get(env_config, "camera", "corner2"),
            seed,
        )
        env = TimeLimit(env, int(_get(env_config, "time_limit", 500)))
    elif suite == "robokitchen":
        robokitchen = _import_env_module("robokitchen")

        env = robokitchen.RoboKitchen(
            task_name,
            action_repeat=int(_get(env_config, "action_repeat", 2)),
            size=_as_size(_get(env_config, "size", (64, 64))),
            seed=seed,
        )
        env = TimeLimit(env, int(_get(env_config, "time_limit", 75)))
    else:
        raise NotImplementedError(f"JAX env suite '{suite}' is not implemented.")

    is_embodied_env = hasattr(type(env), "obs_space") and hasattr(type(env), "act_space")
    if not is_embodied_env:
        env = GymToEmbodied(env, elements, embodied)
    if dinov2_utils.encoder_enabled(_get(config, "model", None)):
        env = dinov2_utils.DINOv2FeatureWrapper(
            env,
            elements,
            dinov2_utils.config_from_model(config.model),
        )
    return wrap_env(env, embodied)


def make_batched_env(
    config, elements, num_envs, *, mode="train", task=None, seed_start=0
):
    env_config = config.envs
    full_task = str(task or env_config.task)
    suite, task_name = full_task.split("_", 1)
    include_image = bool(_get(env_config, "include_image", True))
    if _is_eval_mode(mode):
        reset_seed_bank = _evaluation_seeds(
            env_config,
            config.seed,
            full_task,
            num_envs,
            start=seed_start,
        )
        seed = reset_seed_bank[0]
    else:
        reset_seed_bank = None
        seed = int(_get(env_config, "seed", config.seed))
    if suite == "craftax":
        if include_image:
            raise NotImplementedError(
                "Native device batching currently supports symbolic Craftax only."
            )
        return BatchedCraftaxEnv(
            task_name,
            elements,
            num_envs=num_envs,
            seed=seed,
            time_limit=int(_get(env_config, "time_limit", 1000)),
            action_repeat=int(_get(env_config, "action_repeat", 1)),
            observation_mode=_get(env_config, "observation_mode", None),
            logs=bool(_get(env_config, "logs", True)),
            optimistic_reset_ratio=int(
                _get(env_config, "optimistic_reset_ratio", 16)),
            reset_seed_bank=reset_seed_bank,
            size=_as_size(_get(env_config, "size", (64, 64))),
            render=bool(
                _get(
                    _get(config, "trainer", {}),
                    "video_policy_log",
                    _get(env_config, "video_policy_log", False),
                )
            ),
        )
    if suite not in PLAYGROUND_TASKS_BY_SUITE:
        supported = ", ".join(sorted((*PLAYGROUND_TASKS_BY_SUITE, "craftax")))
        raise NotImplementedError(
            f"Batched JAX envs are not implemented for {suite!r}. "
            f"Supported suites: {supported}."
        )

    backend = str(_get(env_config, "backend", "auto")).lower()
    if backend not in {"auto", "mjx", "mujoco_playground", "playground"}:
        raise ValueError(
            f"Batched env requested for non-MJX MuJoCo Playground backend {backend!r}."
        )
    env = BatchedMujocoPlaygroundEnv(
        suite,
        task_name,
        elements,
        num_envs=num_envs,
        seed=seed,
        time_limit=int(task_env_value(env_config, "time_limit", full_task, 1000)),
        action_repeat=int(
            task_env_value(env_config, "action_repeat", full_task, 1)
        ),
        include_image=include_image,
        size=_as_size(_get(env_config, "size", (64, 64))),
        observation_keys=_get(env_config, "observation_keys", None),
        reset_seed_bank=reset_seed_bank,
    )
    if dinov2_utils.encoder_enabled(_get(config, "model", None)):
        env = dinov2_utils.DINOv2FeatureWrapper(
            env,
            elements,
            dinov2_utils.config_from_model(config.model),
        )
    return env


def _make_dmc_env(
    config, env_config, elements, task_name, seed, fixed_reset=False
):
    backend = str(_get(env_config, "backend", "auto")).lower()
    include_image = bool(_get(env_config, "include_image", True))
    explicit_mjx = backend in {"mjx", "mujoco_playground", "playground"}
    can_auto_mjx = backend == "auto" and not include_image
    can_use_mjx = explicit_mjx or (can_auto_mjx and is_playground_task("dmc", task_name))
    if can_use_mjx:
        return _make_playground_env(
            config,
            env_config,
            elements,
            "dmc",
            task_name,
            seed=seed,
            fixed_reset=fixed_reset,
        )
    if backend in {"mjx", "mujoco_playground", "playground"}:
        raise NotImplementedError(
            f"Cannot use envs.backend={backend!r} for dmc_{task_name}: task is unsupported. "
            f"Supported MuJoCo Playground DMC tasks are: "
            f"{', '.join(_supported_playground_task_names('dmc'))}."
        )
    if backend not in {"auto", "dm_control", "dmc"}:
        raise ValueError(f"Unknown DMC backend {backend!r}.")

    if backend == "auto" and not include_image and not is_playground_task("dmc", task_name):
        print(
            f"Warning: dmc_{task_name} is not available in MuJoCo Playground; "
            "falling back to dm_control."
        )

    dmc = _import_env_module("dmc")

    return dmc.DeepMindControl(
        task_name,
        action_repeat=int(_get(env_config, "action_repeat", 1)),
        size=_as_size(_get(env_config, "size", (64, 64))),
        seed=seed,
        include_image=include_image,
        fixed_reset=bool(fixed_reset),
        manipulation_observation_mode=_get(
            env_config, "manipulation_observation_mode", "features"
        ),
        walker_task_reward_labels=(
            str(_get(env_config, "protocol", "standard"))
            == "concurrent_detached_heads"
            and task_name.startswith("walker_")
        ),
        include_proprio=bool(_get(env_config, "include_proprio", True)),
    )


def _make_playground_env(
    config,
    env_config,
    elements,
    suite,
    task_name,
    seed,
    fixed_reset=False,
):
    backend = str(_get(env_config, "backend", "auto")).lower()
    if backend not in {"auto", "mjx", "mujoco_playground", "playground"}:
        raise ValueError(
            f"Unknown MuJoCo Playground backend {backend!r} for {suite}_{task_name}."
        )
    full_task = f"{suite}_{task_name}"
    return MujocoPlaygroundEnv(
        suite,
        task_name,
        elements,
        seed=seed,
        time_limit=int(task_env_value(env_config, "time_limit", full_task, 1000)),
        action_repeat=int(
            task_env_value(env_config, "action_repeat", full_task, 1)
        ),
        include_image=bool(_get(env_config, "include_image", True)),
        size=_as_size(_get(env_config, "size", (64, 64))),
        observation_keys=_get(env_config, "observation_keys", None),
        fixed_reset=bool(fixed_reset),
    )


def is_playground_task(suite, task_name):
    return task_name in PLAYGROUND_TASKS_BY_SUITE.get(suite, {})


def _supported_playground_task_names(suite):
    aliases = PLAYGROUND_TASKS_BY_SUITE.get(suite, {})
    return sorted(name for name in aliases if name == name.lower())


def _resolve_playground_task(suite, task_name, playground_suite=None):
    aliases = PLAYGROUND_TASKS_BY_SUITE.get(suite)
    if aliases is None:
        return None
    candidate = aliases.get(task_name, task_name)
    if playground_suite is None:
        return candidate if candidate in aliases.values() else None
    if not hasattr(playground_suite, "ALL_ENVS"):
        return candidate if candidate in aliases.values() else None
    all_envs = set(getattr(playground_suite, "ALL_ENVS", ()))
    if candidate in all_envs:
        return candidate
    return None


def _load_playground_suite(suite):
    module_name = PLAYGROUND_SUITE_MODULES.get(suite)
    if module_name is None:
        raise ImportError(f"unknown MuJoCo Playground suite {suite!r}")

    existing = sys.modules.get(f"mujoco_playground._src.{module_name}")
    if existing is not None:
        return existing

    spec = importlib.util.find_spec("mujoco_playground")
    if spec is None or spec.submodule_search_locations is None:
        raise ImportError("mujoco_playground is not installed")

    package_path = next(iter(spec.submodule_search_locations))
    source_path = f"{package_path}/_src"

    package = sys.modules.get("mujoco_playground")
    if package is None:
        package = types.ModuleType("mujoco_playground")
        package.__path__ = [package_path]
        package.__package__ = "mujoco_playground"
        package.__spec__ = spec
        sys.modules["mujoco_playground"] = package

    source_package = sys.modules.get("mujoco_playground._src")
    if source_package is None:
        source_package = types.ModuleType("mujoco_playground._src")
        source_package.__path__ = [source_path]
        source_package.__package__ = "mujoco_playground._src"
        sys.modules["mujoco_playground._src"] = source_package

    return importlib.import_module(f"mujoco_playground._src.{module_name}")


def _load_playground_dm_control_suite():
    """Load Playground's DMC suite without top-level import side effects."""
    return _load_playground_suite("dmc")


def _load_playground_wrapper():
    """Load Playground wrappers after the package stub has been initialized."""
    _load_playground_suite("dmc")
    return importlib.import_module("mujoco_playground._src.wrapper")


def _playground_config(
    playground_suite,
    playground_task,
    *,
    time_limit,
    include_image,
    size,
    render_batch_size,
):
    config = playground_suite.get_default_config(playground_task)
    config.episode_length = int(time_limit)
    if include_image:
        config.vision = True
        vision_config = getattr(config, "vision_config", None)
        if vision_config is not None:
            _set_if_present(vision_config, "render_batch_size", int(render_batch_size))
            _set_if_present(vision_config, "nworld", int(render_batch_size))
            _set_if_present(vision_config, "render_height", int(size[0]))
            _set_if_present(vision_config, "render_width", int(size[1]))
            _set_if_present(vision_config, "cam_res", (int(size[0]), int(size[1])))
            _set_if_present(vision_config, "history", 3)
    return config


def _playground_scalar_metric_names(jax, env):
    """Discover scalar state.metrics keys without adding them to policy inputs."""
    key = jax.random.PRNGKey(0)
    state = None
    eval_shape = getattr(jax, "eval_shape", None)
    if eval_shape is not None:
        try:
            state = eval_shape(env.reset, key)
        except Exception:
            state = None
    if state is None:
        try:
            # Functional Playground resets do not mutate the environment. This
            # fallback runs only once during construction if abstract tracing is
            # unsupported by the installed Playground version.
            state = env.reset(key)
        except Exception as exc:
            print(
                "Warning: could not discover MuJoCo Playground state.metrics; "
                f"environment diagnostics will be unavailable ({exc})."
            )
            return ()
    metrics = getattr(state, "metrics", {})
    names = []
    for name, value in metrics.items():
        shape = tuple(int(dim) for dim in getattr(value, "shape", ()))
        if int(np.prod(shape, dtype=np.int64)) == 1:
            names.append(name)
    return tuple(sorted(names, key=str))


def _playground_metric_logs_np(metrics, names, batch_size=None):
    logs = {}
    for name in names:
        value = np.asarray(metrics[name], dtype=np.float32)
        if batch_size is None:
            value = np.float32(value.reshape(-1)[0])
        else:
            value = value.reshape((int(batch_size), -1))[:, 0]
        logs[f"log/env/{name}"] = np.nan_to_num(
            value, copy=True, nan=0.0, posinf=0.0, neginf=0.0
        )
    return logs


def _set_if_present(config, key, value):
    if hasattr(config, key):
        setattr(config, key, value)


def _validate_playground_vision_env(env, suite, task_name, include_image):
    if not include_image:
        return
    if hasattr(env, "renderer") or hasattr(env, "_rc_pytree"):
        return
    raise RuntimeError(
        f"MuJoCo Playground loaded {suite}_{task_name} with vision=True but did not "
        "create a vision renderer. Install a Playground build with vision support "
        "for this task and its renderer dependency."
    )


def _flatten_playground_obs(obs, prefix=None):
    if isinstance(obs, Mapping):
        result = {}
        for key, value in obs.items():
            name = f"{prefix}/{key}" if prefix else str(key)
            result.update(_flatten_playground_obs(value, name))
        return result
    return {prefix or "vector": obs}


def _extract_playground_image(obs):
    flat = _flatten_playground_obs(obs)
    for key in ("image", "pixels/view_0", "pixels/view0", "pixels"):
        if key in flat:
            return flat[key]
    for value in flat.values():
        shape = getattr(value, "shape", ())
        if len(shape) >= 3:
            return value
    raise KeyError(
        "MuJoCo Playground vision observation did not contain an image-like key."
    )


def _image_to_uint8_np(value):
    image = np.asarray(value)
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim >= 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.ndim >= 3 and image.shape[-1] > 3:
        image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        finite = image[np.isfinite(image)]
        if finite.size:
            min_value = float(finite.min())
            max_value = float(finite.max())
        else:
            min_value = 0.0
            max_value = 0.0
        if min_value < 0.0:
            image = (image + 0.5) * 255.0
        elif max_value <= 1.0:
            image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _image_to_uint8_jnp(value, jnp):
    image = jnp.asarray(value)
    if len(image.shape) == 2:
        image = image[..., None]
    if len(image.shape) >= 3 and image.shape[-1] == 1:
        image = jnp.repeat(image, 3, axis=-1)
    elif len(image.shape) >= 3 and image.shape[-1] > 3:
        image = image[..., :3]
    if np.issubdtype(np.dtype(image.dtype), np.floating):
        min_value = jnp.nanmin(image)
        max_value = jnp.nanmax(image)
        image = jnp.where(
            min_value < 0.0,
            (image + 0.5) * 255.0,
            jnp.where(max_value <= 1.0, image * 255.0, image),
        )
    return jnp.clip(image, 0, 255).astype(jnp.uint8)


def _jax_transfer_scope(jax, name):
    config = getattr(getattr(jax, "_src", None), "config", None)
    scope = getattr(config, name, None)
    if scope is None:
        return contextlib.nullcontext()
    return scope()


def wrap_env(env, embodied):
    for name, space in list(env.act_space.items()):
        if name != "reset" and not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.UnifyDtypes(env)
    env = embodied.wrappers.CheckSpaces(env)
    for name, space in list(env.act_space.items()):
        if name != "reset" and not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env


def _unpack_reset(result):
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
        return result[0]
    return result


def _unpack_step(result):
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, bool(terminated or truncated), info
    obs, reward, done, info = result
    return obs, reward, bool(done), info


def _shape_tuple(size):
    if isinstance(size, int):
        return (int(size),)
    if isinstance(size, tuple):
        return tuple(int(x) for x in size)
    if isinstance(size, list):
        return tuple(int(x) for x in size)
    return tuple(int(x) for x in np.asarray(size).reshape(-1).tolist())
