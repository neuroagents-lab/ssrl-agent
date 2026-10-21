import gymnasium as gym
import numpy as np

# Crafter requires the NumPy bool alias removed in NumPy 1.24.
np.bool = np.bool_


class Crafter(gym.Env):
    metadata = {}

    def __init__(self, task, size=(64, 64), seed=0):
        assert task in ("reward", "noreward")
        import crafter

        self._env = crafter.Env(size=size, reward=(task == "reward"), seed=seed)
        self._achievements = crafter.constants.achievements.copy()
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._env.observation_space.shape, dtype=np.uint8),
        }
        spaces.update({
            f"log/{k}": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32)
            for k in self._achievements
        })
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return gym.spaces.Discrete(self._env.action_space.n)

    def step(self, action):
        image, reward, done, info = self._env.step(action)
        reward = np.float32(reward)
        logs = {
            f"log/{k}": float(info["achievements"][k]) if info else float(0.0)
            for k in self._achievements
        }
        obs = {
            "image": image,
            "is_first": False,
            "is_last": done,
            "is_terminal": info["discount"] == 0,
            **logs,
        }
        return obs, reward, done, info

    def render(self):
        return self._env.render()

    def reset(self):
        image = self._env.reset()
        logs = {f"log/{k}": float(0.0) for k in self._achievements}
        return {"image": image, "is_first": True, "is_last": False, "is_terminal": False, **logs}
