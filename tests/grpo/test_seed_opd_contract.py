from __future__ import annotations

import msgspec
import numpy as np
import pytest

from surogate.grpo.batch import prepare_sample
from surogate.grpo.config import GRPOLossConfig
from surogate.grpo.data import microbatch_to_numpy
from surogate.grpo.transport.types import TrainingSample


def _sample(*, invalid_hindsight_mask: bool = False, invalid_replay_mask: bool = False) -> TrainingSample:
    return TrainingSample(
        prompt_ids=[10, 11],
        prompt_mask=[False, False],
        completion_ids=[12, 13, 14],
        completion_mask=[True, False, True],
        completion_logprobs=[-0.3, 0.0, -0.7],
        completion_temperatures=[1.0, 1.0, 1.0],
        advantage=0.0,
        reward=0.0,
        opd_reference_logprobs=[0.0, 0.0, -0.3, 0.0, -0.7],
        hindsight_logprobs=[0.0, 0.0, -0.1, 0.0, -0.2],
        hindsight_mask=[invalid_hindsight_mask, False, True, False, True],
        replay_mask=[invalid_replay_mask, False, False, False, True],
    )


def test_hindsight_contract_survives_transport_and_batching() -> None:
    encoded = msgspec.msgpack.encode(_sample())
    restored = msgspec.msgpack.decode(encoded, type=TrainingSample)

    micro_batch = prepare_sample(restored, seq_len=16)
    arrays = microbatch_to_numpy(micro_batch)

    np.testing.assert_array_equal(
        arrays["hindsight_mask"],
        np.array([[False, False, True, False, True]]),
    )
    np.testing.assert_allclose(
        arrays["opd_reference_logprobs"],
        np.array([[0.0, 0.0, -0.3, 0.0, -0.7]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        arrays["hindsight_logprobs"],
        np.array([[0.0, 0.0, -0.1, 0.0, -0.2]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        arrays["replay_mask"],
        np.array([[False, False, False, False, True]]),
    )


def test_hindsight_contract_rejects_prompt_or_environment_tokens() -> None:
    with pytest.raises(ValueError, match="only trainable completion tokens"):
        prepare_sample(_sample(invalid_hindsight_mask=True), seq_len=16)


@pytest.mark.parametrize(
    "missing_field",
    ["opd_reference_logprobs", "hindsight_logprobs", "hindsight_mask"],
)
def test_hindsight_contract_rejects_partial_matched_scores(
    missing_field: str,
) -> None:
    sample = _sample()
    setattr(sample, missing_field, None)
    with pytest.raises(ValueError, match="must be provided together"):
        prepare_sample(sample, seq_len=16)


def test_replay_contract_rejects_prompt_or_environment_tokens() -> None:
    with pytest.raises(ValueError, match="replay_mask"):
        prepare_sample(_sample(invalid_replay_mask=True), seq_len=16)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"opd_tau": -0.1}, "opd_tau"),
        ({"opd_beta": 0.0}, "opd_beta"),
        ({"replay_tau": -0.1}, "replay_tau"),
    ],
)
def test_seed_opd_config_rejects_invalid_values(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GRPOLossConfig(**kwargs)
