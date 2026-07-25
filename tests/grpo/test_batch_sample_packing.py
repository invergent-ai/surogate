import pytest

from surogate.grpo.batch import prepare_batch, prepare_sample
from surogate.grpo.transport.types import TrainingSample


def _sample(base: int, *, length: int, advantage: float) -> TrainingSample:
    prompt_len = 2
    completion_len = length - prompt_len
    return TrainingSample(
        prompt_ids=list(range(base, base + prompt_len)),
        prompt_mask=[False] * prompt_len,
        completion_ids=list(range(base + prompt_len, base + length)),
        completion_mask=[True] * completion_len,
        completion_logprobs=[-0.1] * completion_len,
        completion_temperatures=[1.0] * completion_len,
        advantage=advantage,
        opd_reference_logprobs=[0.0] * length,
        hindsight_logprobs=[0.0] * length,
        hindsight_mask=[False] * (length - 1) + [True],
        replay_mask=[False] * (length - 1) + [True],
        replay_weights=[1.0] * (length - 1) + [2.0],
    )


def test_unpacked_batch_keeps_one_sample_per_row_and_exact_training_mass() -> None:
    samples = [
        _sample(10, length=3, advantage=0.1),
        _sample(20, length=4, advantage=0.2),
        _sample(30, length=5, advantage=0.3),
    ]
    expected = {
        tuple(sample.prompt_ids + sample.completion_ids): sample
        for sample in samples
    }

    grid = prepare_batch(
        samples,
        seq_len=12,
        num_train_workers=2,
        idxs=[0, 1, 0],
        num_loras=2,
        sample_packing=False,
    )

    assert [len(rank_rows) for rank_rows in grid] == [2, 2]
    rows = [row for rank_rows in grid for row in rank_rows]
    live_rows = [row for row in rows if any(row.loss_mask)]
    padding_rows = [row for row in rows if not any(row.loss_mask)]
    assert len(live_rows) == len(samples)
    assert len(padding_rows) == 1

    for row in live_rows:
        token_key = tuple(row.input_ids)
        assert token_key in expected
        sample = expected[token_key]
        expected_mask = sample.prompt_mask + sample.completion_mask
        assert row.position_ids == list(range(len(token_key)))
        assert row.loss_mask == expected_mask
        assert row.hindsight_mask == sample.hindsight_mask
        assert row.replay_mask == sample.replay_mask
        assert row.replay_weights == sample.replay_weights
        assert sum(row.lora_num_tokens) == len(token_key)

    assert sum(len(row.input_ids) for row in live_rows) == sum(
        len(sample.prompt_ids) + len(sample.completion_ids) for sample in samples
    )
    assert sum(sum(row.loss_mask) for row in live_rows) == sum(
        sum(sample.prompt_mask) + sum(sample.completion_mask) for sample in samples
    )
    assert all(not any(row.hindsight_mask) for row in padding_rows)
    assert all(not any(row.replay_mask) for row in padding_rows)
    assert all(all(weight == 1.0 for weight in row.replay_weights) for row in padding_rows)


def test_optional_packed_batch_still_combines_compatible_samples() -> None:
    samples = [
        _sample(10, length=4, advantage=0.1),
        _sample(20, length=4, advantage=0.2),
    ]

    grid = prepare_batch(
        samples,
        seq_len=8,
        num_train_workers=1,
        idxs=[0, 0],
        num_loras=1,
        sample_packing=True,
    )

    assert len(grid) == 1
    assert len(grid[0]) == 1
    row = grid[0][0]
    assert row.input_ids == [10, 11, 12, 13, 20, 21, 22, 23]
    assert row.position_ids == [0, 1, 2, 3, 0, 1, 2, 3]
    assert row.loss_mask == [False, False, True, True] * 2
    assert row.lora_num_tokens == [8]


def test_advantage_mask_limits_outcome_credit_without_narrowing_loss_mask() -> None:
    sample = _sample(10, length=5, advantage=0.3)
    sample.advantage_mask = [False, False, True, False, False]

    row = prepare_sample(sample, seq_len=5)

    assert row.loss_mask == [False, False, True, True, True]
    assert row.advantages == [0.0, 0.0, 0.3, 0.0, 0.0]
    assert row.replay_mask == [False, False, False, False, True]


@pytest.mark.parametrize(
    ("mask", "match"),
    [
        ([False], "align with the full sample"),
        ([False] * 5, "select at least one trainable token"),
        ([True, False, False, False, False], "trainable token"),
        ([False, False, False, False, True], "overlap replay"),
    ],
)
def test_rejects_invalid_advantage_masks(
    mask: list[bool],
    match: str,
) -> None:
    sample = _sample(10, length=5, advantage=0.3)
    sample.advantage_mask = mask

    with pytest.raises(ValueError, match=match):
        prepare_sample(sample, seq_len=5)


def test_advantage_mask_remains_aligned_on_generic_truncation() -> None:
    sample = _sample(10, length=5, advantage=0.3)
    sample.advantage_mask = [False, False, True, False, False]

    row = prepare_sample(sample, seq_len=4)

    assert row.input_ids == [10, 11, 12, 13]
    assert row.loss_mask == [False, False, True, True]
    assert row.advantages == [0.0, 0.0, 0.3, 0.0]
