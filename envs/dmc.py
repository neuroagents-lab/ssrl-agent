import copy

import gymnasium as gym
import numpy as np


def _load_manipulation_vision(manipulation, task, seed, size):
    """Build a manipulation task with its native fixed camera observable."""
    from dm_control import composer
    from dm_control.manipulation.shared import registry

    task_name = f"{task}_vision"
    task_instance = registry.get_constructor(task_name)()
    camera_observables = task_instance.task_observables
    if "front_close" not in camera_observables:
        available = ", ".join(camera_observables) or "<none>"
        raise ValueError(
            f"DMC manipulation task {task_name!r} does not expose the expected "
            f"fixed 'front_close' camera; available task observables: {available}."
        )
    camera_observables["front_close"].configure(
        height=int(size[0]), width=int(size[1])
    )

    get_timeout = getattr(manipulation, "_get_timeout", lambda: True)
    time_limit = (
        float(getattr(manipulation, "_TIME_LIMIT", 10.0))
        if get_timeout()
        else float("inf")
    )
    env = composer.Environment(
        task_instance, time_limit=time_limit, random_state=seed
    )
    image_keys = [
        key
        for key, spec in env.observation_spec().items()
        if np.issubdtype(spec.dtype, np.uint8)
        and len(spec.shape) in (3, 4)
        and spec.shape[-1] == 3
    ]
    if image_keys != ["front_close"]:
        raise ValueError(
            f"DMC manipulation task {task_name!r} returned unexpected RGB "
            f"observables: {image_keys}."
        )
    return env, image_keys[0]


def _serialize_rng_state(rng):
    if rng is None:
        return None
    if hasattr(rng, "get_state"):
        return {
            "kind": "random_state",
            "state": copy.deepcopy(rng.get_state()),
        }
    bit_generator = getattr(rng, "bit_generator", None)
    if bit_generator is not None and hasattr(bit_generator, "state"):
        return {
            "kind": "bit_generator",
            "state": copy.deepcopy(bit_generator.state),
        }
    return None


def _restore_rng_state(rng, state):
    if rng is None or not state:
        return
    kind = state.get("kind")
    if kind == "random_state" and hasattr(rng, "set_state"):
        rng.set_state(state["state"])
    elif kind == "bit_generator":
        bit_generator = getattr(rng, "bit_generator", None)
        if bit_generator is not None and hasattr(bit_generator, "state"):
            bit_generator.state = copy.deepcopy(state["state"])


class DeepMindControl(gym.Env):
    metadata = {}

    def __init__(
        self,
        name,
        action_repeat=1,
        size=(64, 64),
        camera=None,
        seed=0,
        include_image=True,
        fixed_reset=False,
        manipulation_observation_mode="features",
        walker_task_reward_labels=False,
        include_proprio=True,
    ):
        if name.endswith("_subtle"):
            is_subtle = True
        else:
            is_subtle = False

        if name.startswith("manip_"):
            domain = "manip"
            task = name.removeprefix("manip_")
        elif "sparse" in name or "finger_turn" in name:
            _name, difficulty = name.rsplit("_", 1)
            domain, task = _name.rsplit("_", 1)
            task = task + "_" + difficulty
        else:
            domain, task = name.rsplit("_", 1)

        self._native_image_key = None
        manipulation_observation_mode = str(
            manipulation_observation_mode or "features"
        ).lower()
        if manipulation_observation_mode not in {"features", "vision"}:
            raise ValueError(
                "manipulation_observation_mode must be 'features' or 'vision', "
                f"got {manipulation_observation_mode!r}."
            )
        if is_subtle:
            from envs import dmc_subtle

            func = getattr(dmc_subtle, name)
            self._env = func(random=seed)
        elif domain == "manip":
            from dm_control import manipulation

            if manipulation_observation_mode == "vision":
                if not include_image:
                    raise ValueError(
                        "Native DMC manipulation vision requires include_image=True."
                    )
                self._env, self._native_image_key = _load_manipulation_vision(
                    manipulation, task, seed, size
                )
            else:
                self._env = manipulation.load(f"{task}_features", seed=seed)
        elif isinstance(domain, str):
            from dm_control import suite

            self._env = suite.load(
                domain,
                task,
                task_kwargs={"random": seed},
            )
        else:
            assert task is None
            self._env = domain()

        self._walker_reward_tasks = {}
        if walker_task_reward_labels:
            if domain != "walker" or task not in {"stand", "walk", "run"}:
                raise ValueError("Walker reward labels require walker stand/walk/run.")
            from dm_control.suite import walker

            # Native rewards use subtree center-of-mass velocity, which cannot
            # be replaced by a coordinate from the generalized velocity obs.
            self._walker_reward_tasks = {
                name: walker.PlanarWalker(move_speed=speed)
                for name, speed in (("stand", 0), ("walk", 1), ("run", 8))
            }

        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(quadruped=2, fish=3).get(domain, 0)
        self._camera = camera
        self._include_image = bool(include_image)
        self._include_proprio = bool(include_proprio)
        self._fixed_reset_rng_state = None
        if fixed_reset:
            task_random = getattr(
                getattr(self._env, "task", None), "random", None
            )
            self._fixed_reset_rng_state = _serialize_rng_state(task_random)
            if self._fixed_reset_rng_state is None:
                print(
                    "Warning: fixed DMC resets were requested but the task RNG "
                    "does not expose serializable state."
                )
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            if key == self._native_image_key:
                shape = value.shape
                if len(shape) == 4 and shape[0] == 1:
                    shape = shape[1:]
                spaces["image"] = gym.spaces.Box(
                    0, 255, shape, dtype=np.uint8
                )
                continue
            if not self._include_proprio:
                continue
            key = key.replace("/", "_")
            if len(value.shape) == 0:
                shape = (1,)
            else:
                shape = value.shape
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        if self._include_image and self._native_image_key is None:
            spaces["image"] = gym.spaces.Box(
                0, 255, self._size + (3,), dtype=np.uint8
            )
        for task in self._walker_reward_tasks:
            spaces[f"label/reward/walker_{task}"] = gym.spaces.Box(
                -np.inf, np.inf, (), dtype=np.float32
            )
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        labels = dict.fromkeys(self._walker_reward_tasks, 0.0)
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            for task, reward_task in self._walker_reward_tasks.items():
                labels[task] += reward_task.get_reward(self._env.physics)
            if time_step.last():
                break
        obs = self._convert_observation(time_step.observation)
        obs.update({
            f"label/reward/walker_{task}": np.float32(value)
            for task, value in labels.items()
        })
        # There is no terminal state in DMC
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        obs["is_last"] = time_step.last()
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self, **kwargs):
        if self._fixed_reset_rng_state is not None:
            task_random = getattr(
                getattr(self._env, "task", None), "random", None
            )
            _restore_rng_state(task_random, self._fixed_reset_rng_state)
        time_step = self._env.reset()
        obs = self._convert_observation(time_step.observation)
        obs.update({
            f"label/reward/walker_{task}": np.float32(0)
            for task in self._walker_reward_tasks
        })
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        obs["is_last"] = time_step.last()
        return obs

    def _convert_observation(self, observation):
        obs = {}
        for key, value in observation.items():
            if key == self._native_image_key:
                image = np.asarray(value, dtype=np.uint8)
                if image.ndim == 4 and image.shape[0] == 1:
                    image = image[0]
                obs["image"] = image
                continue
            if not self._include_proprio:
                continue
            output_key = key.replace("/", "_")
            obs[output_key] = [value] if len(value.shape) == 0 else value
        if self._include_image and self._native_image_key is None:
            obs["image"] = self.render()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)

    def get_state(self):
        state = {
            "physics": np.array(self._env.physics.get_state(), copy=True),
        }
        task_random = getattr(getattr(self._env, "task", None), "random", None)
        task_random_state = _serialize_rng_state(task_random)
        if task_random_state is not None:
            state["task_random_state"] = task_random_state
        for attr in ("_step_count", "_reset_next_step"):
            if hasattr(self._env, attr):
                state[attr] = copy.deepcopy(getattr(self._env, attr))
        return state

    def set_state(self, state):
        if not state:
            return
        physics_state = state.get("physics")
        if physics_state is not None:
            self._env.physics.set_state(np.asarray(physics_state))
            self._env.physics.forward()
        task_random = getattr(getattr(self._env, "task", None), "random", None)
        _restore_rng_state(task_random, state.get("task_random_state"))
        for attr in ("_step_count", "_reset_next_step"):
            if attr in state and hasattr(self._env, attr):
                setattr(self._env, attr, copy.deepcopy(state[attr]))
