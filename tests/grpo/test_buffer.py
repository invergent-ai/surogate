from __future__ import annotations

import pytest
from datasets import Dataset

from surogate.core.config.grpo_orch_config import GRPOBufferConfig
from surogate.grpo.orchestrator.buffer import Buffer
from surogate.utils.dict import DictDefault


def _dataset() -> Dataset:
    # verifiers >= 0.2: EnvGroup routes via info["env_id"] instead of a `task` column.
    return Dataset.from_list(
        [
            {"example_id": 0, "info": {"env_id": "env_a"}, "prompt": [{"role": "user", "content": "a"}]},
            {"example_id": 1, "info": {"env_id": "env_a"}, "prompt": [{"role": "user", "content": "b"}]},
        ]
    )


def _rollout(example_id: int, reward: float) -> dict:
    return {
        "example_id": example_id,
        "info": {"env_id": "env_a"},
        "reward": reward,
        "trajectory": [{"role": "assistant", "content": "{}"}],
        "error": None,
    }


def _buffer(**config_overrides) -> Buffer:
    config = GRPOBufferConfig(
        DictDefault(
            {
                "easy_threshold": 1.0,
                "hard_threshold": 0.0,
                "online_difficulty_filtering": True,
                **config_overrides,
            }
        )
    )
    return Buffer(_dataset(), ["env_a"], config)


def test_online_difficulty_filtering_evicts_saturated_examples():
    buffer = _buffer()

    buffer.update([_rollout(0, 0.0), _rollout(0, 0.0)])
    buffer.update([_rollout(1, 1.0), _rollout(1, 1.0)])

    assert sum(len(examples) for examples in buffer.example_buffer.values()) == 0
    assert len(buffer.hard_examples) == 1
    assert len(buffer.easy_examples) == 1
    assert buffer.rollout_buffer == []


def test_sampling_empty_normal_pool_still_raises_without_recycling():
    buffer = _buffer()
    buffer.update([_rollout(0, 0.0), _rollout(0, 0.0)])
    buffer.update([_rollout(1, 1.0), _rollout(1, 1.0)])

    with pytest.raises(ValueError, match="No environments left with examples"):
        buffer.sample_examples(n=1)


def test_sampling_recycles_easy_and_hard_examples_when_normal_pool_is_empty():
    buffer = _buffer(recycle_easy_fraction=1.0, recycle_hard_fraction=1.0)
    buffer.update([_rollout(0, 0.0), _rollout(0, 0.0)])
    buffer.update([_rollout(1, 1.0), _rollout(1, 1.0)])

    sampled = buffer.sample_examples(n=1)

    assert sampled[0]["example_id"] in {0, 1}
    assert sum(len(examples) for examples in buffer.example_buffer.values()) == 2
    metrics = buffer.get_metrics()
    assert metrics["recycled_examples/easy"] == 1
    assert metrics["recycled_examples/hard"] == 1


def test_one_use_sampling_never_returns_a_task_twice_and_persists_consumption(
    tmp_path,
):
    buffer = _buffer(sample_without_replacement=True)

    sampled = buffer.sample_examples(n=2)

    assert {row["example_id"] for row in sampled} == {0, 1}
    assert len(buffer.consumed_examples) == 2
    with pytest.raises(ValueError, match="No environments left with examples"):
        buffer.sample_examples(n=1)

    checkpoint = tmp_path / "buffer"
    buffer.save(checkpoint)
    restored = _buffer(sample_without_replacement=True)
    restored.load(checkpoint)
    assert {row["example_id"] for row in restored.consumed_examples} == {0, 1}
    with pytest.raises(ValueError, match="No environments left with examples"):
        restored.sample_examples(n=1)


def test_wal_persists_completed_rollouts_across_restart(tmp_path):
    """A mid-step orchestrator bounce must not lose delivered groups: update()
    appends to the WAL, and a fresh buffer replays it after checkpoint load."""
    spool = tmp_path / "live_spool"
    ckpt = tmp_path / "ckpt"

    first = _buffer()
    first.attach_wal(spool)
    first.update([_rollout(0, 1.0), _rollout(0, 0.0)])
    first.save(ckpt)  # checkpoint boundary: WAL truncated, ckpt owns rollouts
    assert not (spool / "rollout_wal.jsonl").exists()
    first.update([_rollout(1, 1.0), _rollout(1, 0.0)])  # post-ckpt delta
    # crash here: `first` dies with 2 undelivered rollouts in memory + WAL

    resumed = _buffer()
    resumed.attach_wal(spool)
    resumed.load(ckpt)
    n_ckpt = len(resumed.rollout_buffer)
    restored = resumed.replay_wal()
    assert restored == 2
    assert len(resumed.rollout_buffer) == n_ckpt + 2


def test_wal_replay_is_idempotent(tmp_path):
    """Double replay (or replay overlapping checkpoint contents) must not
    duplicate rollouts — dedupe is by full-record hash."""
    spool = tmp_path / "live_spool"
    buffer = _buffer()
    buffer.attach_wal(spool)
    buffer.update([_rollout(0, 1.0), _rollout(0, 0.0)])
    n = len(buffer.rollout_buffer)
    assert buffer.replay_wal() == 0  # already in memory
    assert buffer.replay_wal() == 0
    assert len(buffer.rollout_buffer) == n


def test_wal_unarmed_is_a_noop(tmp_path):
    """Without attach_wal (e.g. the val buffer) nothing is written or replayed."""
    buffer = _buffer()
    buffer.update([_rollout(0, 1.0), _rollout(0, 0.0)])
    assert buffer.replay_wal() == 0
    assert not list(tmp_path.iterdir())
