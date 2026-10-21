"""Real environment contracts; run with ``pytest -m integration``."""

import numpy as np
import pytest


pytestmark = pytest.mark.integration


@pytest.mark.parametrize("repeat", (1, 3))
def test_walker_labels_match_native_task_rewards(repeat):
    from dm_control import suite
    from envs.dmc import DeepMindControl

    environment = DeepMindControl(
        "walker_stand", seed=3, include_image=False, action_repeat=repeat,
        walker_task_reward_labels=True,
    )
    targets = {name: suite.load("walker", name, task_kwargs={"random": 3})
               for name in ("stand", "walk", "run")}
    try:
        obs = environment.reset()
        for target in targets.values():
            target.reset()
        for name in targets:
            assert obs[f"label/reward/walker_{name}"] == 0
        for speed in (0, 1, 2, 8):
            for native in (environment._env, *targets.values()):
                with native.physics.reset_context():
                    native.physics.named.data.qvel["rootx"] = speed
            action = np.full(environment.action_space.shape, 0.1, np.float32)
            obs, reward, _, _ = environment.step(action)
            for name, target in targets.items():
                expected = sum(target.step(action).reward for _ in range(repeat))
                assert obs[f"label/reward/walker_{name}"] == pytest.approx(expected)
            assert obs["label/reward/walker_stand"] == pytest.approx(reward)
            assert "image" not in obs
    finally:
        environment.close()
        for target in targets.values():
            target.close()


def test_craftax_classic_pixel_transition(tmp_path, monkeypatch):
    from envs.craftax import Craftax

    monkeypatch.setenv("CRAFTAX_CACHE_DIR", str(tmp_path / "textures"))
    environment = Craftax("classic_pixels", size=(64, 64), seed=0)
    try:
        first = environment.reset()
        obs, reward, done, _ = environment.step(0)
        for observation in (first, obs):
            assert observation["image"].shape == (64, 64, 3)
            assert observation["image"].dtype == np.uint8
        assert first["is_first"]
        assert not obs["is_first"]
        assert environment.action_space.n == 17
        assert np.isfinite(reward)
        assert isinstance(done, bool)
    finally:
        environment.close()
