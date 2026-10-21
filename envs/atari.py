import collections
import os

import ale_py
import ale_py.roms as roms
import gymnasium as gym
import numpy as np
from PIL import Image


class Atari(gym.Env):
    LOCK = None
    metadata = {}
    ACTION_MEANING = (
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE",
    )
    WEIGHTS = np.array([0.299, 0.587, 1 - (0.299 + 0.587)])

    def __init__(
        self,
        name,
        action_repeat=4,
        size=(84, 84),
        gray=True,
        noops=0,
        lives="unused",
        sticky=True,
        actions="all",
        length=108000,
        pooling=2,
        aggregate="max",
        resize="pillow",
        autostart=False,
        clip_reward=False,
        seed=None,
    ):
        assert size[0] == size[1]
        assert lives in ("unused", "discount", "reset"), lives
        assert actions in ("all", "needed"), actions
        assert resize in ("opencv", "pillow"), resize
        assert aggregate in ("max", "mean"), aggregate
        assert pooling >= 1, pooling
        assert action_repeat >= 1, action_repeat

        if self.LOCK is None:
            import multiprocessing as mp

            mp = mp.get_context("spawn")
            self.LOCK = mp.Lock()

        self._resize_fn = resize
        if self._resize_fn == "opencv":
            import cv2

            self._cv2 = cv2

        if name == "james_bond":
            name = "jamesbond"

        self._repeat = action_repeat
        self._size = size
        self._gray = gray
        self._noops = noops
        self._lives = lives
        self._length = length
        self._pooling = pooling
        self._aggregate = aggregate
        self._autostart = autostart
        self._clip_reward = clip_reward
        self._rng = np.random.default_rng(seed)

        with self.LOCK:
            self.ale = ale_py.ALEInterface()
            self.ale.setLoggerMode(ale_py.LoggerMode.Error)
            self.ale.setInt(b"random_seed", self._rng.integers(0, 2**31))
            path = os.environ.get("ALE_ROM_PATH")
            if path:
                self.ale.loadROM(os.path.join(path, f"{name}.bin"))
            else:
                self.ale.loadROM(roms.get_rom_path(name))

        self.ale.setFloat("repeat_action_probability", 0.25 if sticky else 0.0)
        self.actionset = {
            "all": self.ale.getLegalActionSet,
            "needed": self.ale.getMinimalActionSet,
        }[actions]()

        H, W = self.ale.getScreenDims()
        self._buffers = collections.deque(
            [np.zeros((H, W, 3), np.uint8) for _ in range(self._pooling)], maxlen=self._pooling
        )

        self._last_lives = None
        self._done = True
        self._step = 0
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        img_shape = self._size + ((1,) if self._gray else (3,))
        return gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, img_shape, np.uint8),
            "is_first": gym.spaces.Box(0, 1, (), bool),
            "is_last": gym.spaces.Box(0, 1, (), bool),
            "is_terminal": gym.spaces.Box(0, 1, (), bool),
        })

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self.actionset))

    def step(self, action):
        total_reward = 0.0
        dead = False

        for repeat in range(self._repeat):
            reward = self.ale.act(self.actionset[action])
            self._step += 1
            total_reward += reward

            if repeat >= self._repeat - self._pooling:
                self._render()

            if self.ale.game_over():
                dead = True
                break

            current_lives = self.ale.lives()
            if self._lives != "unused" and current_lives < self._last_lives:
                dead = True
                break
            self._last_lives = current_lives

        self._done = self.ale.game_over() or (self._length and self._step >= self._length)

        return self._obs(
            total_reward,
            is_last=self._done or (dead and self._lives == "reset"),
            is_terminal=dead or self.ale.game_over(),
        )

    def reset(self):
        with self.LOCK:
            self.ale.reset_game()

        if self._noops:
            for _ in range(self._rng.integers(self._noops + 1)):
                self.ale.act(self.ACTION_MEANING.index("NOOP"))
                if self.ale.game_over():
                    with self.LOCK:
                        self.ale.reset_game()

        if self._autostart and self.ACTION_MEANING.index("FIRE") in self.actionset:
            self.ale.act(self.ACTION_MEANING.index("FIRE"))
            if self.ale.game_over():
                with self.LOCK:
                    self.ale.reset_game()

        self._last_lives = self.ale.lives()
        self._render()
        # Fill the buffer with the first frame
        for _ in range(self._pooling - 1):
            self._buffers.appendleft(self._buffers[0].copy())

        self._done = False
        self._step = 0
        obs, _, _, _ = self._obs(0.0, is_first=True)
        return obs

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        if self._clip_reward:
            reward = np.sign(reward)
        if self._aggregate == "max":
            image = np.amax(self._buffers, 0)
        elif self._aggregate == "mean":
            image = np.mean(self._buffers, 0).astype(np.uint8)

        if image.shape[:2] != self._size:
            if self._resize_fn == "opencv":
                image = self._cv2.resize(image, self._size, interpolation=self._cv2.INTER_AREA)
            if self._resize_fn == "pillow":
                image = Image.fromarray(image)
                image = image.resize(self._size, Image.BILINEAR)
                image = np.array(image)

        if self._gray:
            image = (image * self.WEIGHTS).sum(-1).astype(image.dtype)
            image = image[:, :, None]

        obs = {
            "image": image,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }
        return obs, reward, is_last, {}

    def _render(self):
        # Efficiently render by reusing buffer memory
        self._buffers.appendleft(self._buffers.pop())
        self.ale.getScreenRGB(self._buffers[0])

    def close(self):
        pass
