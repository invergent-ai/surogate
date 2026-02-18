"""Unit tests for surogate.train.sampling.

Pure Python / NumPy — no GPU required.  Tests cover greedy decoding,
temperature scaling, top-k and top-p filtering, numerical stability, and
the log-prob contract required by GRPO's importance-sampling ratio.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from surogate.train.sampling import sample_greedy, sample_token


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_logits(n: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32)


def log_softmax_ref(logits: np.ndarray) -> np.ndarray:
    """Reference log-softmax in float64."""
    x = logits.astype(np.float64)
    x -= x.max()
    return x - np.log(np.sum(np.exp(x)))


# ---------------------------------------------------------------------------
# Greedy / temperature=0
# ---------------------------------------------------------------------------

class TestGreedy:
    def test_returns_argmax(self):
        logits = np.array([1.0, 3.0, 2.0, 0.5], dtype=np.float32)
        token, lp = sample_greedy(logits)
        assert token == 1

    def test_log_prob_of_argmax(self):
        logits = np.array([1.0, 3.0, 2.0, 0.5], dtype=np.float32)
        token, lp = sample_greedy(logits)
        ref_lp = log_softmax_ref(logits)[token]
        assert math.isclose(lp, float(ref_lp), rel_tol=1e-5), \
            f"log_prob {lp:.6f} vs ref {float(ref_lp):.6f}"

    def test_temperature_zero_same_as_greedy(self):
        logits = make_logits(32, seed=7)
        t1, lp1 = sample_greedy(logits)
        t2, lp2 = sample_token(logits, temperature=0.0)
        assert t1 == t2
        assert math.isclose(lp1, lp2, rel_tol=1e-6)

    def test_log_prob_nonpositive(self):
        logits = make_logits(128, seed=1)
        _, lp = sample_greedy(logits)
        assert lp <= 0.0, f"log_prob should be ≤ 0, got {lp}"

    def test_uniform_logits_any_token(self):
        """With identical logits every token has the same probability."""
        logits = np.zeros(8, dtype=np.float32)
        token, lp = sample_greedy(logits)
        assert 0 <= token < 8
        assert math.isclose(lp, -math.log(8), rel_tol=1e-5)


# ---------------------------------------------------------------------------
# Temperature sampling
# ---------------------------------------------------------------------------

class TestTemperature:
    def test_valid_token_range(self):
        logits = make_logits(100)
        rng = np.random.default_rng(0)
        for _ in range(20):
            token, _ = sample_token(logits, temperature=1.0, rng=rng)
            assert 0 <= token < 100

    def test_log_prob_nonpositive(self):
        logits = make_logits(50, seed=3)
        rng = np.random.default_rng(0)
        for _ in range(20):
            _, lp = sample_token(logits, temperature=0.5, rng=rng)
            assert lp <= 1e-7, f"log_prob should be ≤ 0, got {lp:.6f}"

    def test_log_prob_is_under_filtered_distribution(self):
        """log_prob should equal log P(token) under the temperature-scaled distribution."""
        logits = make_logits(16, seed=42)
        rng = np.random.default_rng(99)
        temperature = 0.7
        token, lp = sample_token(logits, temperature=temperature, rng=rng)
        scaled = (logits / temperature).astype(np.float64)
        ref_lp = float(log_softmax_ref(scaled)[token])
        assert math.isclose(lp, ref_lp, rel_tol=1e-4), \
            f"log_prob {lp:.6f} vs ref {ref_lp:.6f}"

    def test_deterministic_with_fixed_rng(self):
        logits = make_logits(64)
        t1, lp1 = sample_token(logits, temperature=1.0, rng=np.random.default_rng(5))
        t2, lp2 = sample_token(logits, temperature=1.0, rng=np.random.default_rng(5))
        assert t1 == t2
        assert lp1 == lp2


# ---------------------------------------------------------------------------
# Top-k filtering
# ---------------------------------------------------------------------------

class TestTopK:
    def test_only_top_k_tokens_sampled(self):
        """With top_k=3 only the 3 highest logits may be chosen."""
        logits = np.arange(10, dtype=np.float32)  # token 9 has highest logit
        top3 = {7, 8, 9}
        rng = np.random.default_rng(0)
        seen = set()
        for _ in range(200):
            token, _ = sample_token(logits, temperature=1.0, top_k=3, rng=rng)
            seen.add(token)
        assert seen.issubset(top3), f"unexpected tokens: {seen - top3}"

    def test_log_prob_under_top_k_distribution(self):
        logits = make_logits(32, seed=11)
        k = 5
        rng = np.random.default_rng(0)
        token, lp = sample_token(logits, temperature=1.0, top_k=k, rng=rng)
        # Reconstruct filtered logits
        filtered = logits.copy().astype(np.float64)
        threshold = float(np.partition(logits, -k)[-k])
        filtered[filtered < threshold] = -np.inf
        ref_lp = float(log_softmax_ref(filtered)[token])
        assert math.isclose(lp, ref_lp, rel_tol=1e-4), \
            f"log_prob {lp:.6f} vs ref {ref_lp:.6f}"

    def test_top_k_disabled_when_minus_one(self):
        """top_k=-1 should not restrict anything."""
        logits = make_logits(8, seed=2)
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        t1, lp1 = sample_token(logits, temperature=1.0, top_k=-1, rng=rng1)
        t2, lp2 = sample_token(logits, temperature=1.0, rng=rng2)
        assert t1 == t2 and lp1 == lp2

    def test_top_k_one_always_returns_argmax(self):
        logits = make_logits(16, seed=5)
        best = int(np.argmax(logits))
        rng = np.random.default_rng(0)
        for _ in range(10):
            token, _ = sample_token(logits, temperature=1.0, top_k=1, rng=rng)
            assert token == best


# ---------------------------------------------------------------------------
# Top-p (nucleus) filtering
# ---------------------------------------------------------------------------

class TestTopP:
    def test_top_p_one_no_effect(self):
        """top_p=1.0 is the identity."""
        logits = make_logits(8, seed=9)
        rng1 = np.random.default_rng(3)
        rng2 = np.random.default_rng(3)
        t1, lp1 = sample_token(logits, temperature=1.0, top_p=1.0, rng=rng1)
        t2, lp2 = sample_token(logits, temperature=1.0, rng=rng2)
        assert t1 == t2 and lp1 == lp2

    def test_top_p_keeps_at_least_one_token(self):
        """Even if one token has prob > top_p, it is always kept."""
        logits = np.array([10.0, -100.0, -100.0, -100.0], dtype=np.float32)
        rng = np.random.default_rng(0)
        for _ in range(10):
            token, lp = sample_token(logits, temperature=1.0, top_p=0.1, rng=rng)
            assert token == 0
            assert lp <= 1e-7

    def test_top_p_restricts_tail(self):
        """With top_p=0.6, low-probability tail tokens should never appear."""
        rng_gen = np.random.default_rng(0)
        # First token gets ~90% of mass
        logits = np.array([5.0, -1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float32)
        seen = set()
        rng = np.random.default_rng(42)
        for _ in range(500):
            token, _ = sample_token(logits, temperature=1.0, top_p=0.9, rng=rng)
            seen.add(token)
        # Tokens with tiny probability should not appear
        assert 5 not in seen, "token 5 (lowest prob) should be filtered by top_p=0.9"

    def test_log_prob_nonpositive_with_top_p(self):
        logits = make_logits(64, seed=0)
        rng = np.random.default_rng(0)
        for _ in range(20):
            _, lp = sample_token(logits, temperature=1.0, top_p=0.9, rng=rng)
            assert lp <= 1e-7, f"log_prob should be ≤ 0, got {lp:.6f}"


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------

class TestNumericalStability:
    def test_large_positive_logits(self):
        """Logits with large values should not overflow."""
        logits = np.array([1e4, 0.0, 0.0, 0.0], dtype=np.float32)
        token, lp = sample_greedy(logits)
        assert token == 0
        assert math.isfinite(lp)
        assert lp <= 0.0

    def test_large_negative_logits(self):
        """Logits with large negative values (one large positive)."""
        logits = np.full(100, -1e4, dtype=np.float32)
        logits[37] = 0.0
        token, lp = sample_greedy(logits)
        assert token == 37
        assert math.isfinite(lp)

    def test_all_same_logits(self):
        """Uniform logits: each token has prob 1/N."""
        N = 20
        logits = np.zeros(N, dtype=np.float32)
        rng = np.random.default_rng(0)
        token, lp = sample_token(logits, temperature=1.0, rng=rng)
        assert 0 <= token < N
        expected = -math.log(N)
        assert math.isclose(lp, expected, rel_tol=1e-5), \
            f"expected {expected:.6f}, got {lp:.6f}"
