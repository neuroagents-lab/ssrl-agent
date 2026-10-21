"""Compute budgets and replay transfer preserve the experiment protocol."""

import numpy as np
import pytest

from agent.adaptation_protocol import remap_source_parameters, source_components
from agent.train_budget import resolve_start_budget
from agent.walker_replay import WALKER_REWARD_KEYS, normalize_walker_replay_arrays


@pytest.mark.parametrize("ratio,count", [(32, 64), (64, 64), (256, 16), (512, 8), (1024, 4)])
def test_imagination_budget_scales_with_training_ratio(ratio, count):
    assert resolve_start_budget(
        target_ratio=64, train_ratio=ratio, batch_length=64,
    ) == (count, "stratified")


def test_contiguous_budget_divides_replay_length():
    assert resolve_start_budget(
        target_ratio=64, train_ratio=512, batch_length=63,
        automatic_strategy="random_contiguous", align_automatic_divisor=True,
    ) == (9, "random_contiguous")


def test_manual_budget_is_exact_and_rejects_invalid_counts():
    assert resolve_start_budget(manual_count=7, batch_length=64) == (7, "last")
    with pytest.raises(ValueError):
        resolve_start_budget(manual_count=-1)
    with pytest.raises(ValueError):
        resolve_start_budget(manual_count=1, minimum_count=2, batch_length=64)


@pytest.fixture
def walker_replay():
    return {
        "reward": np.array([0, 0.9, 0.8], np.float32),
        "is_first": np.array([True, False, False]),
        "vector": np.zeros((3, 24), np.float32),
        **{key: np.array([0, 0.1 * index, 0.2 * index], np.float32)
           for index, key in enumerate(WALKER_REWARD_KEYS, start=1)},
    }


@pytest.mark.parametrize("target", WALKER_REWARD_KEYS)
def test_replay_transfer_relabels_target_without_mutating_parent(walker_replay, target):
    original = {key: value.copy() for key, value in walker_replay.items()}
    child = normalize_walker_replay_arrays(walker_replay, target)
    assert set(child) == {"reward", "is_first", "vector"}
    np.testing.assert_array_equal(child["reward"], original[target])
    child["reward"][1] = 42
    for key, value in original.items():
        np.testing.assert_array_equal(walker_replay[key], value)


@pytest.mark.parametrize("value", (
    np.array([0, np.nan, 1], np.float32),
    np.array([0, np.inf, 1], np.float32),
    np.array([1, 0, 0], np.float32),
    np.zeros(3, np.int32), np.zeros((3, 1), np.float32),
))
def test_replay_transfer_rejects_invalid_labels(walker_replay, value):
    walker_replay[WALKER_REWARD_KEYS[0]] = value
    with pytest.raises(ValueError):
        normalize_walker_replay_arrays(walker_replay, WALKER_REWARD_KEYS[0])


@pytest.mark.parametrize("backend", ("dreamerv3", "ne_dreamer", "r2dreamer"))
def test_fresh_head_transfer_excludes_parent_behavior(backend):
    params = {"enc/kernel": 1, "dyn/kernel": 2, "rew/kernel": 3, "pol/kernel": 4}
    child = remap_source_parameters(params, source_components(backend))
    assert child == {"enc/kernel": 1, "dyn/kernel": 2}
    assert params["pol/kernel"] == 4


@pytest.mark.parametrize("task", ("stand", "walk", "run"))
def test_pretrained_transfer_selects_matching_task_head(task):
    mapping = source_components(
        "dreamerv3", "pretrained_heads", task=f"dmc_walker_{task}",
        env_config="dmc_walker_adaptation", parent_algorithm="p2e",
        parent_task="dmc_walker_stand",
    )
    params = {"pol/kernel": -1, **{
        f"task_pol_walker_{name}/kernel": index
        for index, name in enumerate(("stand", "walk", "run"))
    }}
    child = remap_source_parameters(params, mapping)
    assert child == {"pol/kernel": ("stand", "walk", "run").index(task)}


def test_bootstrapped_returns_stop_at_episode_boundaries():
    import jax.numpy as jnp
    from dreamer.base import lambda_return

    terminal = jnp.array([[False, False, True, False, False]])
    bootstrap = jnp.zeros((1, 5))
    reward = jnp.array([[0.0, 1.0, 2.0, 100.0, 200.0]])
    reference_reward = reward.at[:, 3:].set(0)
    actual = lambda_return(terminal, terminal, reward, bootstrap, bootstrap, 0.5, 1.0)
    reference = lambda_return(terminal, terminal, reference_reward, bootstrap, bootstrap, 0.5, 1.0)
    np.testing.assert_allclose(actual[:, :2], reference[:, :2])
