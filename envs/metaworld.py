import gymnasium as gym
import numpy as np


class MetaWorld(gym.Env):
    def __init__(
        self,
        name,
        action_repeat=1,
        size=(64, 64),
        camera=None,
        seed=0,
    ):
        import metaworld

        self._camera = camera
        mt1 = metaworld.MT1(name + "-v3", seed=seed)
        env = mt1.train_classes[name + "-v3"](render_mode="rgb_array", camera_name=self._camera)
        env.set_task(mt1.train_tasks[0])

        if self._camera == "corner2":
            env.model.cam_pos[2] = [0.75, 0.075, 0.7]

        self._env = env
        self._env.mujoco_renderer.width = size[1]
        self._env.mujoco_renderer.height = size[0]
        self._env._freeze_rand_vec = False
        self._size = size
        self._action_repeat = action_repeat
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "state": self._env.observation_space,
            "log/success": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        }
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        # MetaWorld actions are bounded by [-1, 1].
        return gym.spaces.Box(
            self._env.action_space.low,
            self._env.action_space.high,
            dtype=np.float32,
        )

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0.0
        success = 0.0
        for _ in range(self._action_repeat):
            state, rew, terminated, truncated, info = self._env.step(action)
            success += float(info["success"])
            reward += rew
            if terminated or truncated:
                break
        # Normalize success to a binary value.
        success = bool(min(success, 1.0))
        # MetaWorld continues after success in training mode.
        is_last = terminated or truncated
        obs = {
            "is_first": False,
            "is_last": is_last,
            # Success permits value bootstrapping.
            "is_terminal": terminated,
            "image": self.render(),
            "state": state,
            "log/success": success,
        }
        return obs, reward, is_last, {}

    def reset(self, **kwargs):
        state, _ = self._env.reset()
        return {
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": self.render(),
            "state": state,
            "log/success": False,
        }

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        if self._camera == "corner2":
            return np.flip(self._env.render(), axis=0)

        return self._env.render()
