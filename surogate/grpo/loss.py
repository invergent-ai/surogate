"""GRPO per-token gradient computation"""

import logging

import numpy as np

from surogate.grpo.config import GRPOLossConfig

_logger = logging.getLogger(__name__)


def _safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """Mean of values over a boolean mask; returns 0 when mask is empty."""
    denom = max(mask.sum(), 1)
    return float(values[mask].sum() / denom)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for detached SEED confidence gates."""
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _compute_sample_grads(
    trainer_logprobs: np.ndarray,
    inference_logprobs: np.ndarray,
    advantages: np.ndarray,
    loss_mask: np.ndarray,
    loss_config: GRPOLossConfig,
    teacher_logprobs: np.ndarray | None = None,
    opd_reference_logprobs: np.ndarray | None = None,
    hindsight_logprobs: np.ndarray | None = None,
    hindsight_mask: np.ndarray | None = None,
    replay_mask: np.ndarray | None = None,
    replay_weights: np.ndarray | None = None,
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
        opd_reference_logprobs: [S] ordinary-branch log-probs from the matched deterministic scorer
        hindsight_logprobs: [S] skill-conditioned log-probs aligned to sampled tokens
        hindsight_mask: [S] bool mask selecting conductor tokens with valid hindsight scores
        replay_mask: [S] bool mask selecting train-split parent-action rehearsal tokens
        replay_weights: [S] optional positive CE multipliers on replay tokens

    Returns:
        (per_token_grads [S], metrics dict)
    """
    replay_token_mask = np.zeros_like(loss_mask, dtype=bool)
    if replay_mask is not None:
        replay_token_mask = loss_mask & replay_mask.astype(bool)
    policy_token_mask = loss_mask & ~replay_token_mask
    replay_token_weights = np.ones_like(trainer_logprobs, dtype=np.float32)
    if replay_weights is not None:
        replay_token_weights = np.asarray(replay_weights, dtype=np.float32)
        if replay_token_weights.shape != trainer_logprobs.shape:
            raise ValueError("replay_weights must match trainer_logprobs shape")
        if not np.isfinite(replay_token_weights).all():
            raise ValueError("replay_weights must be finite")
        if np.any(replay_token_weights[replay_token_mask] <= 0.0):
            raise ValueError("replay_weights must be positive on replay tokens")
        if np.any(replay_token_weights[~replay_token_mask] != 1.0):
            raise ValueError("replay_weights may differ from 1 only on replay tokens")

    log_importance_ratio = trainer_logprobs - inference_logprobs
    importance_ratio = np.exp(log_importance_ratio)
    # Gradient-side ratio cap (see GRPOLossConfig.ratio_clip). Metrics below
    # (mismatch_kl) keep the UNCAPPED ratio: monitoring must see real drift.
    capped_ratio = np.minimum(importance_ratio, loss_config.ratio_clip)

    # IPO masking based on probability difference (DPPO-Binary TV)
    trainer_probs = np.exp(trainer_logprobs)
    inference_probs = np.exp(inference_logprobs)
    probs_diff = trainer_probs - inference_probs
    ipo_mask_high = probs_diff > loss_config.ipo_mask_high
    ipo_mask_low = probs_diff < -loss_config.ipo_mask_low
    is_masked = ipo_mask_high | ipo_mask_low
    keep_mask = policy_token_mask & ~is_masked

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
    per_token_grads[keep_mask] = scaled_advantages[keep_mask] * capped_ratio[keep_mask]
    per_token_grads[policy_token_mask] -= 2.0 * loss_config.kl_tau * log_importance_ratio[policy_token_mask]

    opd_fields = (opd_reference_logprobs, hindsight_logprobs, hindsight_mask)
    if any(value is not None for value in opd_fields) and any(value is None for value in opd_fields):
        raise ValueError("opd_reference_logprobs, hindsight_logprobs, and hindsight_mask must be provided together")
    opd_mask = np.zeros_like(loss_mask, dtype=bool)
    opd_shift = np.zeros_like(trainer_logprobs)
    opd_gate = np.zeros_like(trainer_logprobs)
    if opd_reference_logprobs is not None and hindsight_logprobs is not None and hindsight_mask is not None:
        opd_mask = policy_token_mask & hindsight_mask.astype(bool)
        # Both detached branches must come from one deterministic scorer loaded
        # with the same parent adapter. Rollout logprobs remain independent and
        # are used only for the on-policy importance ratio above.
        opd_shift = hindsight_logprobs - opd_reference_logprobs
        opd_gate = _sigmoid(loss_config.opd_beta * opd_shift)
        # The gate and privileged branch are detached. Positive CE seeds increase
        # ordinary-context likelihood only on conductor tokens selected by opd_mask.
        per_token_grads[opd_mask] += loss_config.opd_tau * opd_gate[opd_mask]

    if replay_mask is not None:
        # Replay is explicit behavior rehearsal, independent of outcome
        # advantages, behavior-policy importance ratios, KL, and privileged
        # hindsight. It keeps known-good parent actions on the ordinary prompt
        # surface even when replay scores are stale or deliberately absent.
        per_token_grads[replay_token_mask] += loss_config.replay_tau * replay_token_weights[replay_token_mask]

    # Policy loss metric
    loss_mask_count = max(loss_mask.sum(), 1)
    pg_loss = np.zeros_like(trainer_logprobs)
    pg_loss[keep_mask] = scaled_advantages[keep_mask] * capped_ratio[keep_mask]
    kl_loss = np.zeros_like(trainer_logprobs)
    kl_loss[policy_token_mask] = loss_config.kl_tau * log_importance_ratio[policy_token_mask] ** 2
    opd_loss = np.zeros_like(trainer_logprobs)
    opd_loss[opd_mask] = -loss_config.opd_tau * opd_gate[opd_mask] * trainer_logprobs[opd_mask]
    replay_loss = np.zeros_like(trainer_logprobs)
    replay_loss[replay_token_mask] = (
        -loss_config.replay_tau * replay_token_weights[replay_token_mask] * trainer_logprobs[replay_token_mask]
    )
    policy_loss = float((-pg_loss + kl_loss + opd_loss + replay_loss).sum() / loss_mask_count)

    metrics = {
        "policy_loss": policy_loss,
        "mismatch_kl": _safe_mean(mismatch_kl, policy_token_mask),
        "masked_mismatch_kl": _safe_mean(mismatch_kl, policy_token_mask & is_masked),
        "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),
        "is_masked": (float(is_masked[policy_token_mask].mean()) if policy_token_mask.any() else 0.0),
        "is_masked_low": (float(ipo_mask_low[policy_token_mask].mean()) if policy_token_mask.any() else 0.0),
        "is_masked_high": (float(ipo_mask_high[policy_token_mask].mean()) if policy_token_mask.any() else 0.0),
        "keep_tokens": int(keep_mask.sum()),
        "total_tokens": int(loss_mask.sum()),
        "ratio_clipped": (
            float((importance_ratio[keep_mask]
                   > loss_config.ratio_clip).mean())
            if keep_mask.any() else 0.0),
        "opd_loss": _safe_mean(opd_loss, opd_mask),
        "opd_gate": _safe_mean(opd_gate, opd_mask),
        "opd_shift": _safe_mean(opd_shift, opd_mask),
        "opd_tokens": int(opd_mask.sum()),
        "replay_loss": _safe_mean(replay_loss, replay_token_mask),
        "replay_tokens": int(replay_token_mask.sum()),
        "replay_weight_sum": float(replay_token_weights[replay_token_mask].sum()),
    }

    if teacher_logprobs is not None:
        metrics["teacher_kl"] = _safe_mean(teacher_logprobs - trainer_logprobs, policy_token_mask)

    return per_token_grads, metrics


def compute_grpo_per_token_grads(
    trainer_logprobs: np.ndarray,
    inference_logprobs: np.ndarray,
    advantages: np.ndarray,
    loss_mask: np.ndarray,
    loss_config: GRPOLossConfig,
    sample_ranges: list[tuple[int, int]],
    teacher_logprobs: np.ndarray | None = None,
    opd_reference_logprobs: np.ndarray | None = None,
    hindsight_logprobs: np.ndarray | None = None,
    hindsight_mask: np.ndarray | None = None,
    replay_mask: np.ndarray | None = None,
    replay_weights: np.ndarray | None = None,
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
        opd_reference_logprobs: [T] optional ordinary matched-branch log-probs
        hindsight_logprobs: [T] optional skill-conditioned log-probs
        hindsight_mask: [T] optional mask for valid conductor-token OPD targets
        replay_mask: [T] optional mask for train-split parent-action rehearsal
        replay_weights: [T] optional positive CE multipliers on replay tokens

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
        "ratio_clipped": 0.0,
        "keep_tokens": 0,
        "total_tokens": 0,
        "opd_loss": 0.0,
        "opd_gate": 0.0,
        "opd_shift": 0.0,
        "opd_tokens": 0,
        "replay_loss": 0.0,
        "replay_tokens": 0,
        "replay_weight_sum": 0.0,
        "policy_sample_count": 0,
    }
    n_samples = 0

    for s_start, s_end in sample_ranges:
        s_loss_mask = loss_mask[s_start:s_end]
        if s_loss_mask.sum() == 0:
            continue

        s_teacher = teacher_logprobs[s_start:s_end] if teacher_logprobs is not None else None
        s_opd_reference = opd_reference_logprobs[s_start:s_end] if opd_reference_logprobs is not None else None
        s_hindsight = hindsight_logprobs[s_start:s_end] if hindsight_logprobs is not None else None
        s_hindsight_mask = hindsight_mask[s_start:s_end] if hindsight_mask is not None else None
        s_replay_mask = replay_mask[s_start:s_end] if replay_mask is not None else None
        s_replay_weights = replay_weights[s_start:s_end] if replay_weights is not None else None

        s_grads, s_metrics = _compute_sample_grads(
            trainer_logprobs=trainer_logprobs[s_start:s_end],
            inference_logprobs=inference_logprobs[s_start:s_end],
            advantages=advantages[s_start:s_end],
            loss_mask=s_loss_mask,
            loss_config=loss_config,
            teacher_logprobs=s_teacher,
            opd_reference_logprobs=s_opd_reference,
            hindsight_logprobs=s_hindsight,
            hindsight_mask=s_hindsight_mask,
            replay_mask=s_replay_mask,
            replay_weights=s_replay_weights,
        )

        per_token_grads[s_start:s_end] = s_grads
        n_samples += 1

        # Accumulate metrics
        agg_metrics["policy_loss"] += s_metrics["policy_loss"]
        policy_tokens = s_metrics["total_tokens"] - s_metrics["replay_tokens"]
        if policy_tokens > 0:
            agg_metrics["policy_sample_count"] += 1
            for key in (
                "mismatch_kl",
                "masked_mismatch_kl",
                "unmasked_mismatch_kl",
                "is_masked",
                "is_masked_low",
                "is_masked_high",
                "ratio_clipped",
            ):
                agg_metrics[key] += s_metrics[key]
        for key in ("opd_loss", "opd_gate", "opd_shift"):
            agg_metrics[key] += s_metrics[key] * s_metrics["opd_tokens"]
        agg_metrics["replay_loss"] += s_metrics["replay_loss"] * s_metrics["replay_tokens"]
        agg_metrics["keep_tokens"] += s_metrics["keep_tokens"]
        agg_metrics["total_tokens"] += s_metrics["total_tokens"]
        agg_metrics["opd_tokens"] += s_metrics["opd_tokens"]
        agg_metrics["replay_tokens"] += s_metrics["replay_tokens"]
        agg_metrics["replay_weight_sum"] += s_metrics["replay_weight_sum"]

        if policy_tokens > 0 and s_teacher is not None and "teacher_kl" in s_metrics:
            agg_metrics.setdefault("teacher_kl", 0.0)
            agg_metrics["teacher_kl"] += s_metrics["teacher_kl"]

    # Policy loss includes both policy and CE-only replay rows, but
    # behavior-likelihood metrics must never be diluted by replay-only rows.
    if n_samples > 0:
        agg_metrics["policy_loss"] /= n_samples
    policy_samples = agg_metrics["policy_sample_count"]
    if policy_samples > 0:
        for key in (
            "mismatch_kl",
            "masked_mismatch_kl",
            "unmasked_mismatch_kl",
            "is_masked",
            "is_masked_low",
            "is_masked_high",
            "ratio_clipped",
        ):
            agg_metrics[key] /= policy_samples
        if "teacher_kl" in agg_metrics:
            agg_metrics["teacher_kl"] /= policy_samples
    if n_samples > 0:
        if agg_metrics["opd_tokens"] > 0:
            for key in ("opd_loss", "opd_gate", "opd_shift"):
                agg_metrics[key] /= agg_metrics["opd_tokens"]
        if agg_metrics["replay_tokens"] > 0:
            agg_metrics["replay_loss"] /= agg_metrics["replay_tokens"]

    return per_token_grads, agg_metrics


def compute_native_shifted_grpo_dloss_reference(
    trainer_logprobs: np.ndarray,
    inference_logprobs: np.ndarray,
    advantages: np.ndarray,
    loss_mask: np.ndarray,
    loss_config: GRPOLossConfig,
    sample_ranges: list[tuple[int, int]],
    teacher_logprobs: np.ndarray | None = None,
    opd_reference_logprobs: np.ndarray | None = None,
    hindsight_logprobs: np.ndarray | None = None,
    hindsight_mask: np.ndarray | None = None,
    replay_mask: np.ndarray | None = None,
    replay_weights: np.ndarray | None = None,
    loss_scale: float = 1.0,
) -> np.ndarray:
    """Reference for the native CUDA GRPO dloss layout.

    Native GRPO consumes trainer logprobs in logical token layout, then writes
    dloss in the shifted target layout expected by the LM-head backward:
    gradient for logical token ``t`` is written at target slot ``t - 1`` within
    the same packed sample.
    """
    per_token_grads, _ = compute_grpo_per_token_grads(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=loss_config,
        sample_ranges=sample_ranges,
        teacher_logprobs=teacher_logprobs,
        opd_reference_logprobs=opd_reference_logprobs,
        hindsight_logprobs=hindsight_logprobs,
        hindsight_mask=hindsight_mask,
        replay_mask=replay_mask,
        replay_weights=replay_weights,
    )
    shifted = np.zeros_like(per_token_grads)
    for start, end in sample_ranges:
        if end - start > 1:
            shifted[start : end - 1] = per_token_grads[start + 1 : end]
    if loss_scale != 1.0:
        shifted = shifted / float(loss_scale)
    return shifted


def compute_native_grpo_metrics_reference(
    trainer_logprobs: np.ndarray,
    inference_logprobs: np.ndarray,
    advantages: np.ndarray,
    loss_mask: np.ndarray,
    loss_config: GRPOLossConfig,
    sample_ranges: list[tuple[int, int]],
    teacher_logprobs: np.ndarray | None = None,
    opd_reference_logprobs: np.ndarray | None = None,
    hindsight_logprobs: np.ndarray | None = None,
    hindsight_mask: np.ndarray | None = None,
    replay_mask: np.ndarray | None = None,
    replay_weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Reference metrics for the native GRPO CUDA accumulator."""
    _, metrics = compute_grpo_per_token_grads(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=loss_config,
        sample_ranges=sample_ranges,
        teacher_logprobs=teacher_logprobs,
        opd_reference_logprobs=opd_reference_logprobs,
        hindsight_logprobs=hindsight_logprobs,
        hindsight_mask=hindsight_mask,
        replay_mask=replay_mask,
        replay_weights=replay_weights,
    )
    return metrics
