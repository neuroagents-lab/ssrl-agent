"""Checkpoint completion and preemption preserve resumable training state."""

from types import SimpleNamespace

import elements

from agent.run import (
    _DurableCheckpoint,
    _save_final_checkpoint_if_complete,
    _save_preempt_checkpoint_if_needed,
)


def test_preemption_commits_state_before_notifying_storage(tmp_path):
    root = tmp_path / "checkpoint.pkl"
    checkpoint = _DurableCheckpoint(root, keep=2)
    checkpoint.step = elements.Counter(7)
    handler = SimpleNamespace(requested=False)
    committed = []

    def on_commit():
        generation = root / (root / "latest").read_text().strip()
        assert (generation / "done").is_file()
        committed.append(generation)

    assert not _save_preempt_checkpoint_if_needed(handler, checkpoint, on_commit)
    assert not committed
    handler.requested = True
    assert _save_preempt_checkpoint_if_needed(handler, checkpoint, on_commit)
    assert len(committed) == 1

    restored = _DurableCheckpoint(root, keep=2)
    restored.step = elements.Counter(0)
    restored.load_or_save()
    assert int(restored.step) == 7


def test_final_checkpoint_requires_completed_budget(tmp_path):
    root = tmp_path / "checkpoint.pkl"
    checkpoint = _DurableCheckpoint(root, keep=2)
    checkpoint.step = elements.Counter(9)
    handler = SimpleNamespace(requested=False)
    assert not _save_final_checkpoint_if_complete(handler, checkpoint, checkpoint.step, 10)
    checkpoint.step.increment()
    committed = []
    assert _save_final_checkpoint_if_complete(
        handler, checkpoint, checkpoint.step, 10, lambda: committed.append(True),
    )
    assert committed == [True]
    restored = _DurableCheckpoint(root, keep=2)
    restored.step = elements.Counter(0)
    restored.load_or_save()
    assert int(restored.step) == 10
    handler.requested = True
    assert not _save_final_checkpoint_if_complete(handler, checkpoint, checkpoint.step, 10)
