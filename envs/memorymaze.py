import gym as old_gym
import gymnasium as gym
import numpy as np


class MemoryMaze(gym.Env):
    def __init__(self, task, size=(64, 64), seed=0):
        # 9x9, 11x11, 13x13 and 15x15 are available
        self._env = old_gym.make(f"memory_maze:MemoryMaze-{task}-v0", seed=seed)
        self._obs_is_dict = hasattr(self._env.observation_space, "spaces")
        self._size = size

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def observation_space(self):
        img_shape = self._size + (3,)
        return gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, img_shape, np.uint8),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        })

    @property
    def action_space(self):
        return gym.spaces.Discrete(self._env.action_space.n)

    def step(self, action):
        image, reward, done, info = self._env.step(action)
        obs = {"image": image, "is_first": False, "is_last": done, "is_terminal": info.get("is_terminal", False)}
        return obs, reward, done, info

    def reset(self):
        image = self._env.reset()
        return {"image": image, "is_first": True, "is_last": False, "is_terminal": False}
