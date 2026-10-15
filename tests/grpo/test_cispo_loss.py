"""CISPO as a GRPOLossConfig variant.

CISPO keeps every completion token's gradient (no probability-difference mask), has no KL term, and weights each
token by the stop-gradient importance ratio truncated at `ratio_clip`. The native seed already computes
A * min(ratio, ratio_clip) for every kept token, so the variant is a parameter setting; these tests pin it down on
the Python reference loss that `test_native_parity.py` compares with the CUDA kernel.
"""

import numpy as np
import pytest

from surogate.grpo.config import GRPOLossConfig
from surogate.grpo.loss import compute_grpo_per_token_grads


def test_cispo_variant_disables_the_mask_and_the_kl_term():
    config = GRPOLossConfig(variant="cispo", ratio_clip=1.28)
    assert config.ipo_mask_low >= 1.0 and config.ipo_mask_high >= 1.0
    assert config.kl_tau == 0.0
    assert config.ratio_clip == pytest.approx(1.28)


def test_default_variant_is_unchanged():
    config = GRPOLossConfig()
    assert config.variant == "ipo"
    assert config.ipo_mask_low == pytest.approx(0.2) and config.ipo_mask_high == pytest.approx(0.2)
    assert config.kl_tau == pytest.approx(1e-3)


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        GRPOLossConfig(variant="ppo")


def test_cispo_grads_are_advantage_times_truncated_ratio_on_every_completion_token():
    # Token 2 has a probability difference of ~0.45, which the default IPO mask (0.2) would drop.
    trainer_logprobs = np.array([-3.0, -1.0, np.log(0.9), -2.0, -0.1], dtype=np.float32)
    inference_logprobs = np.array([-3.0, -1.1, np.log(0.45), -1.5, -0.1], dtype=np.float32)
    advantages = np.array([0.0, 0.8, 0.8, -0.5, -0.5], dtype=np.float32)
    loss_mask = np.array([False, True, True, True, True])
    sample_ranges = [(0, 5)]
    clip = 1.28
    config = GRPOLossConfig(variant="cispo", ratio_clip=clip)

    grads, _ = compute_grpo_per_token_grads(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=config,
        sample_ranges=sample_ranges,
        teacher_logprobs=None,
    )

    ratio = np.exp(trainer_logprobs - inference_logprobs)
    expected_magnitude = np.where(loss_mask, np.abs(advantages * np.minimum(ratio, clip)), 0.0)
    np.testing.assert_allclose(np.abs(grads), expected_magnitude, rtol=1e-5, atol=1e-7)
    assert grads[2] != 0.0, "CISPO must keep the token the IPO mask would drop"
    assert grads[0] == 0.0, "prompt tokens carry no gradient"

    ipo = GRPOLossConfig(kl_tau=0.0, ratio_clip=clip)
    ipo_grads, _ = compute_grpo_per_token_grads(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=ipo,
        sample_ranges=sample_ranges,
        teacher_logprobs=None,
    )
    assert ipo_grads[2] == 0.0, "the default IPO mask drops the high-difference token"
    # Same sign convention as the IPO path on tokens both keep.
    np.testing.assert_allclose(grads[[1, 4]], ipo_grads[[1, 4]], rtol=1e-6)
