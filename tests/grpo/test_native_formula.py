import numpy as np
import pytest

from surogate.grpo.config import GRPOLossConfig
from surogate.grpo.loss import (
    compute_grpo_per_token_grads,
    compute_native_grpo_metrics_reference,
    compute_native_shifted_grpo_dloss_reference,
)


def _expected_shifted_dloss(
    trainer_logprobs: np.ndarray,
    inference_logprobs: np.ndarray,
    advantages: np.ndarray,
    loss_mask: np.ndarray,
    loss_config: GRPOLossConfig,
    sample_ranges: list[tuple[int, int]],
    teacher_logprobs: np.ndarray | None,
    loss_scale: float,
) -> np.ndarray:
    grads, _ = compute_grpo_per_token_grads(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=loss_config,
        sample_ranges=sample_ranges,
        teacher_logprobs=teacher_logprobs,
    )
    shifted = np.zeros_like(grads)
    for start, end in sample_ranges:
        if end - start > 1:
            shifted[start : end - 1] = grads[start + 1 : end]
    return shifted / loss_scale


def test_native_shifted_formula_matches_existing_grpo_loss_without_teacher():
    trainer_logprobs = np.array([-7.0, -1.2, -0.8, -3.1, -2.0, -0.4, -5.0], dtype=np.float32)
    inference_logprobs = np.array([-7.0, -1.4, -0.7, -2.8, -2.1, -0.5, -4.6], dtype=np.float32)
    advantages = np.array([0.0, 1.5, -0.2, 0.7, 0.0, 2.0, -1.0], dtype=np.float32)
    loss_mask = np.array([False, True, True, True, False, True, True])
    sample_ranges = [(0, 4), (4, 7)]
    config = GRPOLossConfig(ipo_mask_low=0.15, ipo_mask_high=0.15, adv_tau=0.9, teacher_tau=0.0, kl_tau=0.03)

    actual = compute_native_shifted_grpo_dloss_reference(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=config,
        sample_ranges=sample_ranges,
        teacher_logprobs=None,
        loss_scale=3.0,
    )

    expected = _expected_shifted_dloss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=config,
        sample_ranges=sample_ranges,
        teacher_logprobs=None,
        loss_scale=3.0,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_native_shifted_formula_matches_existing_grpo_loss_with_teacher():
    trainer_logprobs = np.array([-6.0, -1.0, -2.2, -0.2, -3.0, -1.1], dtype=np.float32)
    inference_logprobs = np.array([-6.0, -1.2, -2.0, -0.25, -2.7, -1.4], dtype=np.float32)
    teacher_logprobs = np.array([-6.0, -1.5, -1.7, -0.3, -2.9, -1.2], dtype=np.float32)
    advantages = np.array([0.0, 0.4, 1.2, -0.7, 2.5, -1.0], dtype=np.float32)
    loss_mask = np.array([False, True, True, True, True, False])
    sample_ranges = [(0, 3), (3, 6)]
    config = GRPOLossConfig(ipo_mask_low=0.25, ipo_mask_high=0.1, adv_tau=1.1, teacher_tau=0.4, kl_tau=0.02)

    actual = compute_native_shifted_grpo_dloss_reference(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=config,
        sample_ranges=sample_ranges,
        teacher_logprobs=teacher_logprobs,
        loss_scale=2.0,
    )

    expected = _expected_shifted_dloss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=config,
        sample_ranges=sample_ranges,
        teacher_logprobs=teacher_logprobs,
        loss_scale=2.0,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_native_metrics_reference_matches_existing_grpo_metrics():
    trainer_logprobs = np.array([-7.0, -1.2, -0.8, -3.1, -2.0, -0.4, -5.0], dtype=np.float32)
    inference_logprobs = np.array([-7.0, -1.4, -0.7, -2.8, -2.1, -0.5, -4.6], dtype=np.float32)
    teacher_logprobs = np.array([-7.0, -1.0, -1.1, -3.0, -2.3, -0.9, -4.9], dtype=np.float32)
    advantages = np.array([0.0, 1.5, -0.2, 0.7, 0.0, 2.0, -1.0], dtype=np.float32)
    loss_mask = np.array([False, True, True, True, False, True, True])
    sample_ranges = [(0, 4), (4, 7)]
    config = GRPOLossConfig(ipo_mask_low=0.15, ipo_mask_high=0.15, adv_tau=0.9, teacher_tau=0.3, kl_tau=0.03)

    _, expected = compute_grpo_per_token_grads(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=config,
        sample_ranges=sample_ranges,
        teacher_logprobs=teacher_logprobs,
    )

    actual = compute_native_grpo_metrics_reference(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=config,
        sample_ranges=sample_ranges,
        teacher_logprobs=teacher_logprobs,
    )

    for key, expected_value in expected.items():
        assert key in actual
        assert actual[key] == expected_value


def test_seed_opd_supplies_dense_credit_when_group_rewards_tie():
    trainer_logprobs = np.array([-8.0, -2.0, -1.0, -3.0], dtype=np.float32)
    inference_logprobs = trainer_logprobs.copy()
    hindsight_logprobs = np.array([-8.0, -1.0, -1.0, -4.0], dtype=np.float32)
    advantages = np.zeros(4, dtype=np.float32)
    loss_mask = np.array([False, True, True, True])
    hindsight_mask = np.array([False, True, False, True])
    config = GRPOLossConfig(adv_tau=1.0, kl_tau=0.0, opd_tau=0.4, opd_beta=2.0)

    grads, metrics = compute_grpo_per_token_grads(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=config,
        sample_ranges=[(0, 4)],
        hindsight_logprobs=hindsight_logprobs,
        hindsight_mask=hindsight_mask,
    )

    supported_gate = 1.0 / (1.0 + np.exp(-2.0))
    unsupported_gate = 1.0 / (1.0 + np.exp(2.0))
    np.testing.assert_allclose(
        grads,
        [0.0, 0.4 * supported_gate, 0.0, 0.4 * unsupported_gate],
        rtol=2e-7,
    )
    assert metrics["opd_tokens"] == 2
    assert metrics["opd_gate"] == pytest.approx(np.mean([supported_gate, unsupported_gate]))
    assert metrics["opd_shift"] == pytest.approx(0.0)
    assert metrics["policy_loss"] > 0.0


def test_seed_opd_is_inert_when_disabled():
    trainer_logprobs = np.array([-8.0, -2.0, -1.0], dtype=np.float32)
    grads, metrics = compute_grpo_per_token_grads(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=trainer_logprobs.copy(),
        advantages=np.zeros(3, dtype=np.float32),
        loss_mask=np.array([False, True, True]),
        loss_config=GRPOLossConfig(kl_tau=0.0, opd_tau=0.0, opd_beta=2.0),
        sample_ranges=[(0, 3)],
        hindsight_logprobs=np.array([-8.0, -1.0, -2.0], dtype=np.float32),
        hindsight_mask=np.array([False, True, True]),
    )

    np.testing.assert_array_equal(grads, np.zeros(3, dtype=np.float32))
    assert metrics["opd_tokens"] == 2
    assert metrics["opd_loss"] == 0.0


def test_seed_opd_gate_uses_matched_rollout_branches_not_native_drift():
    trainer_logprobs = np.array([-8.0, -2.5, -0.5], dtype=np.float32)
    inference_logprobs = np.array([-8.0, -1.0, -2.0], dtype=np.float32)
    hindsight_logprobs = np.array([-8.0, -0.5, -2.5], dtype=np.float32)
    loss_mask = np.array([False, True, True])
    config = GRPOLossConfig(adv_tau=0.0, kl_tau=0.0, opd_tau=0.4, opd_beta=2.0)

    grads, metrics = compute_grpo_per_token_grads(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        advantages=np.zeros(3, dtype=np.float32),
        loss_mask=loss_mask,
        loss_config=config,
        sample_ranges=[(0, 3)],
        hindsight_logprobs=hindsight_logprobs,
        hindsight_mask=loss_mask,
    )

    supported_gate = 1.0 / (1.0 + np.exp(-1.0))
    unsupported_gate = 1.0 / (1.0 + np.exp(1.0))
    np.testing.assert_allclose(
        grads,
        [0.0, 0.4 * supported_gate, 0.4 * unsupported_gate],
        rtol=2e-7,
    )
    assert metrics["opd_gate"] == pytest.approx(
        np.mean([supported_gate, unsupported_gate])
    )
