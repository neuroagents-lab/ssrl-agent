import os
import sys

import gymnasium as gym
import numpy as np


_conda_lib = os.path.join(sys.prefix, "lib")
if os.path.isdir(_conda_lib):
    _ld_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if _conda_lib not in _ld_paths:
        os.environ["LD_LIBRARY_PATH"] = (
            _conda_lib
            if not os.environ.get("LD_LIBRARY_PATH")
            else f"{_conda_lib}:{os.environ['LD_LIBRARY_PATH']}"
        )

import deepmind_lab


class DeepMindLabyrinth(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    IMPALA_ACTION_SET = (
        (0, 0, 0, 1, 0, 0, 0),  # Forward
        (0, 0, 0, -1, 0, 0, 0),  # Backward
        (0, 0, -1, 0, 0, 0, 0),  # Strafe Left
        (0, 0, 1, 0, 0, 0, 0),  # Strafe Right
        (-20, 0, 0, 0, 0, 0, 0),  # Look Left
        (20, 0, 0, 0, 0, 0, 0),  # Look Right
        (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
        (20, 0, 0, 1, 0, 0, 0),  # Look Right + Forward
        (0, 0, 0, 0, 1, 0, 0),  # Fire
    )

    POPART_ACTION_SET = (
        (0, 0, 0, 1, 0, 0, 0),  # FW
        (0, 0, 0, -1, 0, 0, 0),  # BW
        (0, 0, -1, 0, 0, 0, 0),  # Strafe Left
        (0, 0, 1, 0, 0, 0, 0),  # Strafe Right
        (-10, 0, 0, 0, 0, 0, 0),  # Small LL
        (10, 0, 0, 0, 0, 0, 0),  # Small LR
        (-60, 0, 0, 0, 0, 0, 0),  # Large LL
        (60, 0, 0, 0, 0, 0, 0),  # Large LR
        (0, 10, 0, 0, 0, 0, 0),  # Look Down
        (0, -10, 0, 0, 0, 0, 0),  # Look Up
        (-10, 0, 0, 1, 0, 0, 0),  # FW + Small LL
        (10, 0, 0, 1, 0, 0, 0),  # FW + Small LR
        (-60, 0, 0, 1, 0, 0, 0),  # FW + Large LL
        (60, 0, 0, 1, 0, 0, 0),  # FW + Large LR
        (0, 0, 0, 0, 1, 0, 0),  # Fire
    )

    ACTION_SET_DEFAULT = POPART_ACTION_SET

    ACTION_SET_MEDIUM = (
        (0, 0, 0, 1, 0, 0, 0),  # Forward
        (0, 0, 0, -1, 0, 0, 0),  # Backward
        (0, 0, -1, 0, 0, 0, 0),  # Strafe Left
        (0, 0, 1, 0, 0, 0, 0),  # Strafe Right
        (-20, 0, 0, 0, 0, 0, 0),  # Look Left
        (20, 0, 0, 0, 0, 0, 0),  # Look Right
        (0, 0, 0, 0, 0, 0, 0),  # Idle.
    )

    ACTION_SET_SMALL = (
        (0, 0, 0, 1, 0, 0, 0),  # Forward
        (-20, 0, 0, 0, 0, 0, 0),  # Look Left
        (20, 0, 0, 0, 0, 0, 0),  # Look Right
    )

    def __init__(
        self,
        level,
        mode,
        action_repeat=4,
        render_size=(64, 64),
        action_set="popart",
        level_cache=None,
        seed=None,
        runfiles_path=None,
    ):
        if mode not in ("train", "eval", "test"):
            raise NotImplementedError(mode)
        if mode == "train" and level.endswith("_test"):
            level = level.replace("_test", "_train")
        if runfiles_path:
            print("Setting DMLab runfiles path:", runfiles_path)
            deepmind_lab.set_runfiles_path(runfiles_path)
        self._config = {}
        self._config["width"] = render_size[0]
        self._config["height"] = render_size[1]
        self._config["logLevel"] = "WARN"
        if mode in ("eval", "test"):
            self._config["allowHoldOutLevels"] = "true"
            self._config["mixerSeed"] = 0x600D5EED
        self._action_repeat = action_repeat
        self._random = np.random.RandomState(seed)
        self._env = deepmind_lab.Lab(
            level="contributed/dmlab30/" + level,
            observations=["RGB_INTERLEAVED"],
            config={k: str(v) for k, v in self._config.items()},
            level_cache=level_cache,
        )
        if isinstance(action_set, str):
            action_sets = {
                "default": self.ACTION_SET_DEFAULT,
                "popart": self.POPART_ACTION_SET,
                "impala": self.IMPALA_ACTION_SET,
                "medium": self.ACTION_SET_MEDIUM,
                "small": self.ACTION_SET_SMALL,
            }
            key = action_set.lower()
            if key not in action_sets:
                raise ValueError(
                    f"Unknown DMLab action set '{action_set}'. "
                    f"Expected one of: {sorted(action_sets.keys())}"
                )
            self._action_set = action_sets[key]
        else:
            self._action_set = tuple(action_set)
        self._last_image = None
        self._done = True

    @property
    def observation_space(self):
        shape = (self._config["height"], self._config["width"], 3)
        space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        return gym.spaces.Dict({"image": space})

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._action_set))

    def reset(self, *, seed=None, options=None):
        del options
        if seed is not None:
            self._random = np.random.RandomState(seed)
        self._done = False
        self._env.reset(seed=self._random.randint(0, 2**31 - 1))
        obs = self._get_obs()
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        return obs

    def step(self, action):
        raw_action = np.array(self._action_set[action], np.intc)
        reward = self._env.step(raw_action, num_steps=self._action_repeat)
        self._done = not self._env.is_running()
        obs = self._get_obs()
        done = self._done
        obs["is_first"] = False
        obs["is_last"] = done
        obs["is_terminal"] = done
        info = {"discount": np.array(0.0 if done else 1.0, dtype=np.float32)}
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        del args  # Unused
        del kwargs  # Unused
        return self._last_image

    def close(self):
        self._env.close()

    def _get_obs(self):
        if self._done:
            if self._last_image is None:
                image = np.zeros(
                    (self._config["height"], self._config["width"], 3),
                    dtype=np.uint8,
                )
            else:
                image = np.zeros_like(self._last_image)
        else:
            image = self._env.observations()["RGB_INTERLEAVED"]
        self._last_image = image
        return {"image": image}
