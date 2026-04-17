"""GRPO per-token gradient computation"""

import logging

import numpy as np

from surogate.grpo.config import GRPOLossConfig

_logger = logging.getLogger(__name__)


def _safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """Mean of values over a boolean mask; returns 0 when mask is empty."""
    denom = max(mask.sum(), 1)
    return float(values[mask].sum() / denom)


def _compute_sample_grads(
    trainer_logprobs: np.ndarray,
    inference_logprobs: np.ndarray,
    advantages: np.ndarray,
    loss_mask: np.ndarray,
    loss_config: GRPOLossConfig,
    teacher_logprobs: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute IPO per-token gradient multipliers for a single sample.

    Implements INTELLECT Policy Optimization (IPO) loss from prime-rl:
    - DPPO-Binary TV masking based on probability difference
    - Squared KL regularization

    Args:
        trainer_logprobs: [S] log-probs from current policy for this sample
        inference_logprobs: [S] log-probs from inference policy for this sample
        advantages: [S] per-token advantages for this sample
        loss_mask: [S] bool mask (True = completion token, False = prompt)
        loss_config: GRPO loss hyperparameters
        teacher_logprobs: [S] optional teacher log-probs for this sample

    Returns:
        (per_token_grads [S], metrics dict)
    """
    log_importance_ratio = trainer_logprobs - inference_logprobs
    importance_ratio = np.exp(log_importance_ratio)

    # IPO masking based on probability difference (DPPO-Binary TV)
    trainer_probs = np.exp(trainer_logprobs)
    inference_probs = np.exp(inference_logprobs)
    probs_diff = trainer_probs - inference_probs
    ipo_mask_high = probs_diff > loss_config.ipo_mask_high
    ipo_mask_low = probs_diff < -loss_config.ipo_mask_low
    is_masked = ipo_mask_high | ipo_mask_low
    keep_mask = loss_mask & ~is_masked

    # KL mismatch (for metrics)
    mismatch_kl = importance_ratio - log_importance_ratio - 1.0

    # Advantages with optional teacher KL
    scaled_advantages = loss_config.adv_tau * advantages
    if teacher_logprobs is not None:
        teacher_kl = teacher_logprobs - trainer_logprobs
        scaled_advantages = scaled_advantages + loss_config.teacher_tau * teacher_kl

    # Per-token gradient seeding for surogate's CE backward kernel.
    #
    # IPO loss = sum_t[-keep_mask_t * adv_t * ratio_t + kl_tau * loss_mask_t * log_ratio_t^2]
    #
    # Sign convention: surogate's CE backward computes dlogit = dloss * (softmax - one_hot).
    # We need dloss = -d(loss)/d(trainer_logprob_t), giving:
    #   dloss = keep_mask_t * adv_t * ratio_t - 2 * kl_tau * loss_mask_t * log_ratio_t
    per_token_grads = np.zeros_like(trainer_logprobs)
    per_token_grads[keep_mask] = scaled_advantages[keep_mask] * importance_ratio[keep_mask]
    per_token_grads[loss_mask] -= 2.0 * loss_config.kl_tau * log_importance_ratio[loss_mask]

    # Policy loss metric
    loss_mask_count = max(loss_mask.sum(), 1)
    pg_loss = np.zeros_like(trainer_logprobs)
    pg_loss[keep_mask] = scaled_advantages[keep_mask] * importance_ratio[keep_mask]
    kl_loss = np.zeros_like(trainer_logprobs)
    kl_loss[loss_mask] = loss_config.kl_tau * log_importance_ratio[loss_mask] ** 2
    policy_loss = float((-pg_loss + kl_loss).sum() / loss_mask_count)

    metrics = {
        "policy_loss": policy_loss,
        "mismatch_kl": _safe_mean(mismatch_kl, loss_mask),
        "masked_mismatch_kl": _safe_mean(mismatch_kl, loss_mask & is_masked),
        "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),
        "is_masked": float(is_masked[loss_mask].mean()) if loss_mask.any() else 0.0,
        "is_masked_low": float(ipo_mask_low[loss_mask].mean()) if loss_mask.any() else 0.0,
        "is_masked_high": float(ipo_mask_high[loss_mask].mean()) if loss_mask.any() else 0.0,
        "keep_tokens": int(keep_mask.sum()),
        "total_tokens": int(loss_mask.sum()),
    }

    if teacher_logprobs is not None:
        metrics["teacher_kl"] = _safe_mean(teacher_logprobs - trainer_logprobs, loss_mask)

    return per_token_grads, metrics


def compute_grpo_per_token_grads(
    trainer_logprobs: np.ndarray,
    inference_logprobs: np.ndarray,
    advantages: np.ndarray,
    loss_mask: np.ndarray,
    loss_config: GRPOLossConfig,
    sample_ranges: list[tuple[int, int]],
    teacher_logprobs: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute GRPO per-token gradient multipliers for a packed batch.

    Processes each sample individually (matching prime-rl's per-sample loss_fn),
    then assembles results back into the packed layout.

    The returned per_token_grads are NOT normalized by loss_scale. The C++ engine's
    caller is expected to divide by the global loss_scale (sum of loss_mask
    across all micro-batches), matching prime-rl's loss = total_loss / loss_scale.

    Args:
        trainer_logprobs: [T] log-probs from current policy (compute_logprobs output)
        inference_logprobs: [T] log-probs from inference policy (MicroBatch)
        advantages: [T] per-token advantages from orchestrator
        loss_mask: [T] bool mask (True = completion token, False = prompt)
        loss_config: GRPO loss hyperparameters
        sample_ranges: list of (start, end) tuples for each sample in the packed sequence
        teacher_logprobs: [T] optional teacher log-probs for KL distillation

    Returns:
        (per_token_grads [T], aggregated metrics dict)
    """
    T = len(trainer_logprobs)
    per_token_grads = np.zeros(T, dtype=np.float32)

    # Aggregate metrics across samples
    agg_metrics: dict[str, float] = {
        "policy_loss": 0.0,
        "mismatch_kl": 0.0,
        "masked_mismatch_kl": 0.0,
        "unmasked_mismatch_kl": 0.0,
        "is_masked": 0.0,
        "is_masked_low": 0.0,
        "is_masked_high": 0.0,
        "keep_tokens": 0,
        "total_tokens": 0,
    }
    n_samples = 0

    for s_start, s_end in sample_ranges:
        s_loss_mask = loss_mask[s_start:s_end]
        if s_loss_mask.sum() == 0:
            continue

        s_teacher = teacher_logprobs[s_start:s_end] if teacher_logprobs is not None else None

        s_grads, s_metrics = _compute_sample_grads(
            trainer_logprobs=trainer_logprobs[s_start:s_end],
            inference_logprobs=inference_logprobs[s_start:s_end],
            advantages=advantages[s_start:s_end],
            loss_mask=s_loss_mask,
            loss_config=loss_config,
            teacher_logprobs=s_teacher,
        )

        per_token_grads[s_start:s_end] = s_grads
        n_samples += 1

        # Accumulate metrics
        for key in (
            "policy_loss",
            "mismatch_kl",
            "masked_mismatch_kl",
            "unmasked_mismatch_kl",
            "is_masked",
            "is_masked_low",
            "is_masked_high",
        ):
            agg_metrics[key] += s_metrics[key]
        agg_metrics["keep_tokens"] += s_metrics["keep_tokens"]
        agg_metrics["total_tokens"] += s_metrics["total_tokens"]

        if s_teacher is not None and "teacher_kl" in s_metrics:
            agg_metrics.setdefault("teacher_kl", 0.0)
            agg_metrics["teacher_kl"] += s_metrics["teacher_kl"]

    # Average float metrics over samples
    if n_samples > 0:
        for key in (
            "policy_loss",
            "mismatch_kl",
            "masked_mismatch_kl",
            "unmasked_mismatch_kl",
            "is_masked",
            "is_masked_low",
            "is_masked_high",
        ):
            agg_metrics[key] /= n_samples
        if "teacher_kl" in agg_metrics:
            agg_metrics["teacher_kl"] /= n_samples

    return per_token_grads, agg_metrics
