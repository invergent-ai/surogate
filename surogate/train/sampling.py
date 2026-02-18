# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# Token sampling utilities for GRPO online generation.
#
# Implements temperature, top-k, and top-p (nucleus) sampling with numerically
# stable log-probability computation.  The returned log_prob is the
# log-probability of the sampled token under the *filtered* distribution
# (after temperature / top-k / top-p), which is the old_log_prob needed for
# GRPO's importance-sampling ratio.

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax.

    Handles -inf logits correctly: exp(-inf) = 0 contributes nothing to the
    partition function, so -inf tokens get log_prob = -inf.
    """
    logits = logits.astype(np.float64)
    # max over finite values only, so -inf doesn't poison the reduction
    finite = logits[np.isfinite(logits)]
    if finite.size == 0:
        raise ValueError("All logits are -inf; cannot sample")
    max_logit = float(np.max(finite))
    # exp(-inf - max) = 0 automatically
    log_sum_exp = max_logit + np.log(np.sum(np.exp(logits - max_logit)))
    return logits - log_sum_exp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = -1,
    top_p: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, float]:
    """Sample a single token from a logit vector and return its log-probability.

    Args:
        logits:      Raw unnormalized logits from the model, shape [vocab_size].
        temperature: Sampling temperature.  0.0 → greedy argmax.
        top_k:       Restrict sampling to the top-k tokens.  -1 disables.
        top_p:       Nucleus (top-p) sampling threshold in (0, 1].  1.0 disables.
        rng:         NumPy random generator; created fresh if None.

    Returns:
        (token_id, log_prob):
            token_id  — sampled token index.
            log_prob  — log P(token | filtered distribution).  Used as
                        old_log_prob in GRPO importance-sampling ratio.
    """
    if rng is None:
        rng = np.random.default_rng()

    logits = np.array(logits, dtype=np.float32)
    vocab_size = len(logits)

    # --- Greedy decoding (temperature = 0) ----------------------------------
    if temperature == 0.0:
        token = int(np.argmax(logits))
        return token, float(_log_softmax(logits)[token])

    # --- Temperature scaling ------------------------------------------------
    if temperature != 1.0:
        logits = logits / temperature

    # --- Top-k filtering ----------------------------------------------------
    if 0 < top_k < vocab_size:
        # np.partition: O(N) partial sort — threshold is the k-th largest value
        kth_value = float(np.partition(logits, -top_k)[-top_k])
        logits = np.where(logits >= kth_value, logits, -np.inf)

    # --- Top-p (nucleus) filtering ------------------------------------------
    if top_p < 1.0:
        log_probs = _log_softmax(logits)
        # Sort descending by log-prob (equivalently, by probability)
        sorted_indices = np.argsort(log_probs)[::-1]
        cum_probs = np.cumsum(np.exp(log_probs[sorted_indices]))

        # Build removal mask: tokens whose cumulative probability exceeds top_p.
        # Shift right by one so that the *first* token crossing the threshold is
        # kept (ensures at least one token survives even when a single token has
        # probability > top_p).
        to_remove = cum_probs > top_p
        to_remove[1:] = to_remove[:-1].copy()
        to_remove[0] = False

        logits[sorted_indices[to_remove]] = -np.inf

    # --- Sample -------------------------------------------------------------
    log_probs = _log_softmax(logits)
    probs = np.exp(log_probs).astype(np.float64)
    # Clip tiny negatives from floating-point error and renormalize exactly
    probs = np.maximum(probs, 0.0)
    probs /= probs.sum()

    token = int(rng.choice(vocab_size, p=probs))
    return token, float(log_probs[token])


def sample_greedy(logits: np.ndarray) -> Tuple[int, float]:
    """Greedy argmax sampling; convenience wrapper around sample_token."""
    return sample_token(logits, temperature=0.0)
