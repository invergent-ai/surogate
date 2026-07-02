from __future__ import annotations

import pytest
from datasets import Dataset

from surogate.core.config.grpo_orch_config import GRPOBufferConfig
from surogate.grpo.orchestrator.buffer import Buffer
from surogate.utils.dict import DictDefault


def _dataset() -> Dataset:
    return Dataset.from_list(
        [
            {"example_id": 0, "task": "env_a", "prompt": [{"role": "user", "content": "a"}]},
            {"example_id": 1, "task": "env_a", "prompt": [{"role": "user", "content": "b"}]},
        ]
    )


def _rollout(example_id: int, reward: float) -> dict:
    return {
        "example_id": example_id,
        "task": "env_a",
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
