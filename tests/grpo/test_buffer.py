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


def test_sampling_empty_normal_pool_recycles_via_starvation_guard():
    """Pre-guard behavior raised here; the per-env starvation guard now returns
    a weighted env's easy/hard examples to normal instead of leaving the env
    silently unschedulable (recycle fractions unset does not disable it)."""
    buffer = _buffer()
    buffer.update([_rollout(0, 0.0), _rollout(0, 0.0)])
    buffer.update([_rollout(1, 1.0), _rollout(1, 1.0)])

    sampled = buffer.sample_examples(n=1)

    assert sampled[0]["example_id"] in {0, 1}
    assert sum(len(examples) for examples in buffer.example_buffer.values()) == 2
    assert not buffer.easy_examples and not buffer.hard_examples


def test_sampling_raises_only_when_every_pool_is_truly_empty():
    buffer = _buffer()
    buffer.example_buffer = {"env_a": {}}
    buffer.easy_examples = []
    buffer.hard_examples = []

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


def _two_env_buffer(**config_overrides) -> Buffer:
    dataset = Dataset.from_list(
        [
            {"example_id": 0, "task": "env_full", "prompt": [{"role": "user", "content": "a"}]},
            {"example_id": 1, "task": "env_full", "prompt": [{"role": "user", "content": "b"}]},
            {"example_id": 2, "task": "env_full", "prompt": [{"role": "user", "content": "c"}]},
            {"example_id": 3, "task": "env_starved", "prompt": [{"role": "user", "content": "d"}]},
        ]
    )
    config = GRPOBufferConfig(
        DictDefault(
            {
                "easy_threshold": 1.0,
                "hard_threshold": 0.5,
                "online_difficulty_filtering": True,
                "normal_pool_min_examples": 0,
                "recycle_easy_fraction": 0.15,
                "recycle_hard_fraction": 0.15,
                **config_overrides,
            }
        )
    )
    return Buffer(dataset, ["env_full", "env_starved"], config)


def test_starved_env_recycles_its_own_pool_even_when_global_floor_never_trips():
    """An env whose EVERY example was classified hard must not silently drop out
    of the deal: the global normal-pool floor cannot trip while other envs stay
    full, so the per-env guard returns the starved env's examples to normal."""
    buffer = _two_env_buffer()

    starved_rollout = {
        "example_id": 3,
        "task": "env_starved",
        "reward": 0.0,
        "trajectory": [{"role": "assistant", "content": "{}"}],
        "error": None,
    }
    buffer.update([starved_rollout, starved_rollout])

    # env_starved fully drained to hard; env_full keeps the global count high.
    assert len(buffer.example_buffer["env_starved"]) == 0
    assert len(buffer.example_buffer["env_full"]) == 3
    assert len(buffer.hard_examples) == 1

    buffer.sample_examples(n=1)

    assert len(buffer.example_buffer["env_starved"]) == 1, (
        "per-env starvation guard must return the starved env's examples to normal"
    )
    assert len(buffer.hard_examples) == 0
    assert len(buffer.example_buffer["env_full"]) == 3
