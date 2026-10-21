"""Shared, non-mutating reward relabeling for Walker adaptation replay."""

from collections.abc import Mapping

import numpy as np


WALKER_REWARD_KEYS = tuple(
    f"label/reward/walker_{task}" for task in ("stand", "walk", "run")
)


def normalize_walker_replay_arrays(arrays: Mapping, reward_key: str) -> dict:
    """Replace reward with an authenticated native target label and drop labels.

    Callers authenticate source bytes before invoking this function and validate
    the remaining fields against the child schema afterward. The returned dict
    owns its reward array; other arrays are unchanged. Skill filtering remains
    the responsibility of the ordinary replay-transfer normalization.
    """
    if reward_key not in WALKER_REWARD_KEYS:
        raise ValueError(f"Unsupported Walker target reward label: {reward_key!r}")
    for key in ("reward", "is_first", reward_key):
        if key not in arrays:
            raise ValueError(f"Walker parent replay is missing {key!r}.")
    reward = np.asarray(arrays["reward"])
    target = np.asarray(arrays[reward_key])
    first = np.asarray(arrays["is_first"])
    if reward.dtype != np.dtype(np.float32) or reward.ndim != 1:
        raise ValueError("Walker parent reward must be a scalar float32 sequence.")
    if target.dtype != np.dtype(np.float32):
        raise ValueError(f"Walker target reward label {reward_key!r} must be float32.")
    if target.shape != reward.shape:
        raise ValueError(
            f"Walker target reward label {reward_key!r} shape mismatch: "
            f"{target.shape} != {reward.shape}."
        )
    if first.dtype != np.dtype(bool) or first.shape != reward.shape:
        raise ValueError("Walker is_first must be a boolean scalar sequence matching reward.")
    if not np.all(np.isfinite(target)):
        raise ValueError(f"Walker target reward label {reward_key!r} must be finite.")
    if np.any(target[first] != 0):
        raise ValueError(f"Walker target reward label {reward_key!r} must be zero on reset.")
    normalized = {key: value for key, value in arrays.items() if key not in WALKER_REWARD_KEYS}
    normalized["reward"] = np.array(target, copy=True, order="C")
    return normalized
