"""Compute-budget helpers for ratio-scaled Dreamer objectives."""

from __future__ import annotations

import math


def resolve_start_budget(
    *,
    manual_count: int = 0,
    target_ratio: float = 0.0,
    train_ratio: float | None = None,
    batch_length: int | None = None,
    strategy: str | None = None,
    automatic_strategy: str = "stratified",
    minimum_count: int = 1,
    align_automatic_divisor: bool = False,
) -> tuple[int, str]:
    """Resolve samples per sequence for a target samples-per-env-step rate.

    Dreamer performs ``train_ratio / (batch_size * batch_length)`` learner
    updates per environment step. Each update starts ``batch_size * count``
    trajectories, so ``count = batch_length * target_ratio / train_ratio``
    keeps behavior learning compute stable as the world-model ratio changes.
    A nonzero manual count overrides automatic budgeting. Contiguous budgets
    can be rounded upward to a sequence-length divisor so
    aligned block sampling gives every time step equal inclusion probability.
    """

    manual_count = int(manual_count or 0)
    target_ratio = float(target_ratio or 0.0)
    minimum_count = int(minimum_count)
    if manual_count < 0:
        raise ValueError(f"manual_count must be nonnegative, got {manual_count}")
    if target_ratio < 0:
        raise ValueError(f"target_ratio must be nonnegative, got {target_ratio}")
    if minimum_count < 1:
        raise ValueError(
            f"minimum_count must be positive, got {minimum_count}"
        )

    automatic = manual_count == 0 and target_ratio > 0
    if strategy is None:
        strategy = automatic_strategy if automatic else "last"
    strategy = str(strategy).lower()
    if strategy not in {"last", "stratified", "random_contiguous"}:
        raise ValueError(
            "start strategy must be 'last', 'stratified', or "
            "'random_contiguous', "
            f"got {strategy!r}"
        )

    effective_minimum = (
        min(minimum_count, int(batch_length))
        if batch_length is not None and int(batch_length) > 0
        else minimum_count
    )
    if manual_count:
        if manual_count < effective_minimum:
            raise ValueError(
                "manual_count must be at least "
                f"{effective_minimum}, got {manual_count}"
            )
        count = manual_count
    elif not automatic or train_ratio is None or batch_length is None:
        # Zero selects every time step.
        count = 0
    else:
        train_ratio = float(train_ratio)
        batch_length = int(batch_length)
        if batch_length <= 0:
            raise ValueError(f"batch_length must be positive, got {batch_length}")
        if train_ratio <= 0:
            # Evaluation-only configurations do not require a sample budget.
            count = 0
        else:
            count = math.ceil(batch_length * target_ratio / train_ratio)
            count = max(effective_minimum, count)
            count = min(batch_length, count)
            if (
                align_automatic_divisor
                and strategy == "random_contiguous"
                and batch_length % count
            ):
                count = next(
                    candidate
                    for candidate in range(count + 1, batch_length + 1)
                    if batch_length % candidate == 0
                )

    if batch_length is not None and count:
        count = min(int(batch_length), count)
    return count, strategy
