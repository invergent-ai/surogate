"""Tests for GradientTracker."""

import logging
import math

import pytest

from surogate.train.gradient_tracker import GradientTracker


@pytest.fixture
def logger():
    return logging.getLogger("test_gradient_tracker")


class TestGradientTracker:
    def test_initial_state(self, logger):
        gt = GradientTracker(logger)
        assert gt.mean == 0.0
        assert gt.max == 0.0
        assert gt.trend == 0.0

    def test_stats_after_warmup(self, logger):
        gt = GradientTracker(logger, window=20, warmup=5)
        for i in range(5):
            gt.step(1.0, step=i)
        assert gt.mean == pytest.approx(1.0)
        assert gt.max == pytest.approx(1.0)

    def test_stats_not_computed_before_warmup(self, logger):
        gt = GradientTracker(logger, window=20, warmup=10)
        for i in range(9):
            gt.step(2.0, step=i)
        # Still below warmup — stats should remain at defaults
        assert gt.mean == 0.0
        assert gt.max == 0.0
        assert gt.trend == 0.0

    def test_mean_and_max(self, logger):
        gt = GradientTracker(logger, window=10, warmup=3)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for i, v in enumerate(values):
            gt.step(v, step=i)
        assert gt.mean == pytest.approx(3.0)
        assert gt.max == pytest.approx(5.0)

    def test_positive_trend_detected(self, logger):
        """Monotonically increasing norms should produce a positive trend."""
        gt = GradientTracker(logger, window=10, warmup=5)
        for i in range(10):
            gt.step(1.0 + i * 0.5, step=i)  # 1.0, 1.5, 2.0, ...
        assert gt.trend > 0

    def test_flat_trend_near_zero(self, logger):
        gt = GradientTracker(logger, window=10, warmup=5)
        for i in range(10):
            gt.step(2.0, step=i)
        assert gt.trend == pytest.approx(0.0, abs=1e-6)

    def test_negative_trend(self, logger):
        gt = GradientTracker(logger, window=10, warmup=5)
        for i in range(10):
            gt.step(5.0 - i * 0.3, step=i)
        assert gt.trend < 0

    def test_non_finite_ignored(self, logger):
        gt = GradientTracker(logger, window=10, warmup=3)
        for i in range(5):
            gt.step(1.0, step=i)
        gt.step(float('nan'), step=5)
        gt.step(float('inf'), step=6)
        # Should still have 5 entries, not 7
        assert len(gt.history) == 5
        assert gt.mean == pytest.approx(1.0)

    def test_rolling_window_evicts_old(self, logger):
        gt = GradientTracker(logger, window=5, warmup=3)
        # Fill with 1.0
        for i in range(5):
            gt.step(1.0, step=i)
        # Now push 5 values of 3.0
        for i in range(5):
            gt.step(3.0, step=5 + i)
        assert gt.mean == pytest.approx(3.0)
        assert gt.max == pytest.approx(3.0)

    def test_warning_emitted_on_sustained_trend(self, logger, caplog):
        """A strong upward trend maintained for patience steps triggers a warning."""
        gt = GradientTracker(
            logger, window=10, warmup=5,
            trend_threshold=0.1, patience=3, cooldown=0,
        )
        # Feed strongly increasing norms to trigger warning
        with caplog.at_level(logging.WARNING):
            for i in range(20):
                gt.step(1.0 + i * 2.0, step=i)
        assert "Gradient acceleration" in caplog.text

    def test_no_warning_for_stable_norms(self, logger, caplog):
        gt = GradientTracker(
            logger, window=10, warmup=5,
            trend_threshold=0.5, patience=3, cooldown=0,
        )
        with caplog.at_level(logging.WARNING):
            for i in range(20):
                gt.step(2.0, step=i)
        assert "Gradient acceleration" not in caplog.text

    def test_cooldown_suppresses_repeated_warnings(self, logger, caplog):
        gt = GradientTracker(
            logger, window=10, warmup=5,
            trend_threshold=0.1, patience=2, cooldown=50,
        )
        with caplog.at_level(logging.WARNING):
            for i in range(30):
                gt.step(1.0 + i * 2.0, step=i)

        warnings = [r for r in caplog.records if "Gradient acceleration" in r.message]
        # Should get at most 1 warning because cooldown=50 > total steps=30
        assert len(warnings) == 1

    def test_trend_count_resets_on_stable(self, logger):
        gt = GradientTracker(
            logger, window=10, warmup=5,
            trend_threshold=0.1, patience=5, cooldown=0,
        )
        # Build up trend count with increasing norms
        for i in range(10):
            gt.step(1.0 + i * 1.0, step=i)
        saved_count = gt.trend_count
        assert saved_count > 0

        # Now flatten — trend count should reset
        for i in range(10):
            gt.step(5.0, step=10 + i)
        assert gt.trend_count == 0
