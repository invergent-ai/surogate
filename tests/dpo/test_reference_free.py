"""Reference-free DPO synthetic-margin construction."""

import numpy as np

from surogate.dpo.trainer import _reference_free_logprobs


def _batch():
    return {
        "loss_mask": np.array(
            [
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
            ],
            dtype=np.uint8,
        ),
        "pair_chosen": np.array([0], dtype=np.int32),
    }


def test_target_margin_is_distributed_in_sum_mode():
    refs = _reference_free_logprobs([_batch()], B=2, T=6, beta=0.5, target_margin=1.0, length_norm=False)

    assert refs.shape == (2, 6)
    # Two chosen tokens share target_margin / beta = 2.0.
    assert refs[0].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    assert refs[1].tolist() == [0.0] * 6


def test_target_margin_is_per_token_in_length_normalized_mode():
    refs = _reference_free_logprobs([_batch()], B=2, T=6, beta=0.5, target_margin=1.0, length_norm=True)

    # The native kernel averages reference log-probs in length-normalized mode.
    assert refs[0].tolist() == [0.0, 2.0, 0.0, 2.0, 0.0, 0.0]


def test_reference_free_zero_margin_is_all_zero():
    refs = _reference_free_logprobs([_batch()], B=2, T=6, beta=0.3, target_margin=0.0, length_norm=False)

    assert not refs.any()
