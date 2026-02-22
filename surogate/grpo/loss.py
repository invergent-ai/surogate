"""GRPO per-token gradient computation"""

import numpy as np

from surogate.grpo.config import GRPOLossConfig


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
    """Compute GRPO per-token gradient multipliers for a single sample.

    This matches prime-rl's default_loss_fn: all sequence-level quantities
    (geo_seq_ratio, seq_importance_ratio, geo_mask, seq_mask) are computed
    over this sample only.

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
    token_importance_ratio = np.exp(log_importance_ratio)

    # Geometric mean ratio (exp of mean log ratio over loss_mask) — per sample
    loss_mask_count = max(loss_mask.sum(), 1)
    geo_mean_log = log_importance_ratio[loss_mask].sum() / loss_mask_count
    geo_seq_ratio = np.exp(geo_mean_log)

    # KL mismatch (for metrics)
    token_mismatch_kl = token_importance_ratio - log_importance_ratio - 1.0

    # Sequence-level importance ratio (clamped) — per sample
    seq_log_ratio = np.clip(log_importance_ratio[loss_mask].sum(), a_min=None, a_max=10.0)
    seq_importance_ratio = np.clip(np.exp(seq_log_ratio), a_min=None, a_max=loss_config.sequence_clip_high)

    # Token-level masks
    token_mask_low = token_importance_ratio < loss_config.token_mask_low
    token_mask_high = token_importance_ratio > loss_config.token_mask_high

    # Geometric masks — per sample, broadcast to all tokens
    geo_mask_low = np.full_like(loss_mask, geo_seq_ratio < loss_config.geo_mask_low)
    geo_mask_high = np.full_like(loss_mask, geo_seq_ratio > loss_config.geo_mask_high)

    # Sequence-level masks — per sample (min/max token ratio triggers full sample mask)
    masked_ratios = np.where(loss_mask, token_importance_ratio, np.inf)
    seq_min_ratio = masked_ratios.min()
    masked_ratios_high = np.where(loss_mask, token_importance_ratio, -np.inf)
    seq_max_ratio = masked_ratios_high.max()
    seq_mask_low = np.full_like(loss_mask, seq_min_ratio < loss_config.sequence_mask_low)
    seq_mask_high = np.full_like(loss_mask, seq_max_ratio > loss_config.sequence_mask_high)

    is_masked = (
        token_mask_low | token_mask_high
        | geo_mask_low | geo_mask_high
        | seq_mask_low | seq_mask_high
    )
    keep_mask = loss_mask & ~is_masked

    # Choose importance ratio type
    if loss_config.ratio_type == "sequence":
        importance_ratio = np.full_like(trainer_logprobs, seq_importance_ratio)
    else:
        importance_ratio = token_importance_ratio

    # Compute coefficient (detached in prime-rl, so treated as constant)
    scaled_advantages = loss_config.adv_tau * advantages
    if teacher_logprobs is not None:
        teacher_kl = teacher_logprobs - trainer_logprobs
        scaled_advantages = scaled_advantages + loss_config.teacher_tau * teacher_kl

    coeff = importance_ratio * (scaled_advantages - loss_config.kl_tau * log_importance_ratio)

    # Per-token gradient seeding for surogate's CE backward kernel.
    #
    # Sign convention: surogate's CE backward computes dlogit = dloss * (softmax - one_hot),
    # while prime-rl's FusedOutputLinear backward computes dlogit = g * (one_hot - softmax).
    # Since (softmax - one_hot) = -(one_hot - softmax), we need dloss = -g = +coeff.
    per_token_grads = np.zeros_like(trainer_logprobs)
    per_token_grads[keep_mask] = coeff[keep_mask]

    # For sequence ratio type, normalize by this sample's loss_mask count
    if loss_config.ratio_type == "sequence":
        per_token_grads = per_token_grads / loss_mask_count

    # Policy loss: mean of -coeff over unmasked tokens
    if keep_mask.any():
        policy_loss = float(-coeff[keep_mask].mean())
    else:
        policy_loss = 0.0

    metrics = {
        "policy_loss": policy_loss,
        "mismatch_kl": _safe_mean(token_mismatch_kl, loss_mask),
        "unmasked_mismatch_kl": _safe_mean(token_mismatch_kl, keep_mask),
        "is_masked_frac": float(is_masked[loss_mask].mean()) if loss_mask.any() else 0.0,
        "geo_seq_ratio": float(geo_seq_ratio),
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
        "unmasked_mismatch_kl": 0.0,
        "is_masked_frac": 0.0,
        "geo_seq_ratio": 0.0,
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
        for key in ("policy_loss", "mismatch_kl", "unmasked_mismatch_kl",
                     "is_masked_frac", "geo_seq_ratio"):
            agg_metrics[key] += s_metrics[key]
        agg_metrics["keep_tokens"] += s_metrics["keep_tokens"]
        agg_metrics["total_tokens"] += s_metrics["total_tokens"]

        if s_teacher is not None and "teacher_kl" in s_metrics:
            agg_metrics.setdefault("teacher_kl", 0.0)
            agg_metrics["teacher_kl"] += s_metrics["teacher_kl"]

    # Average float metrics over samples
    if n_samples > 0:
        for key in ("policy_loss", "mismatch_kl", "unmasked_mismatch_kl",
                     "is_masked_frac", "geo_seq_ratio"):
            agg_metrics[key] /= n_samples
        if "teacher_kl" in agg_metrics:
            agg_metrics["teacher_kl"] /= n_samples

    return per_token_grads, agg_metrics
