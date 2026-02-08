"""Tests for LRSchedule temporary override and LossGuard escalation."""

import logging
import math

import pytest

from surogate.train.lr_schedule import LRSchedule
from surogate.train.loss_guard import LossGuard


# ---------------------------------------------------------------------------
# LRSchedule override tests
# ---------------------------------------------------------------------------

def _make_schedule(**kwargs):
    defaults = dict(
        base_lr=1e-3, max_steps=1000, warmup_steps=100,
        cooldown_steps=100, final_lr=1e-5, schedule_type="cosine",
    )
    defaults.update(kwargs)
    return LRSchedule(**defaults)


class TestLRScheduleOverride:
    def test_no_override_returns_scheduled(self):
        sched = _make_schedule()
        assert not sched.has_override
        assert sched.get_lr(200) == sched._scheduled_lr(200)

    def test_override_returns_overridden_lr_at_start(self):
        sched = _make_schedule()
        sched.override_lr(lr=1e-4, step=200, grace_period=50)
        assert sched.has_override
        # At step 200 (elapsed=0), should return the override LR exactly
        assert sched.get_lr(200) == pytest.approx(1e-4)

    def test_override_blends_linearly(self):
        sched = _make_schedule()
        sched.override_lr(lr=1e-4, step=200, grace_period=100)
        override_lr = 1e-4
        scheduled_lr = sched._scheduled_lr(250)
        # At step 250, elapsed=50, t=0.5 → midpoint
        expected = override_lr + (scheduled_lr - override_lr) * 0.5
        assert sched.get_lr(250) == pytest.approx(expected, rel=1e-6)

    def test_override_expires_after_grace(self):
        sched = _make_schedule()
        sched.override_lr(lr=1e-4, step=200, grace_period=50)
        # At step 250 (elapsed==grace), override expires
        lr = sched.get_lr(250)
        assert lr == pytest.approx(sched._scheduled_lr(250))
        assert not sched.has_override

    def test_override_fully_expired_stays_expired(self):
        sched = _make_schedule()
        sched.override_lr(lr=1e-4, step=200, grace_period=50)
        # Well past expiry
        sched.get_lr(300)
        assert not sched.has_override
        assert sched.get_lr(400) == pytest.approx(sched._scheduled_lr(400))

    def test_override_does_not_affect_base_lr(self):
        sched = _make_schedule(base_lr=1e-3)
        sched.override_lr(lr=1e-4, step=200, grace_period=50)
        assert sched.base_lr == 1e-3

    def test_reduce_lr_is_permanent(self):
        sched = _make_schedule(base_lr=1e-3, final_lr=1e-5)
        sched.reduce_lr(0.5)
        assert sched.base_lr == pytest.approx(5e-4)
        assert sched.final_lr == pytest.approx(5e-6)

    def test_override_with_grace_1(self):
        sched = _make_schedule()
        sched.override_lr(lr=1e-4, step=200, grace_period=1)
        # At start step, elapsed=0, t=0 → override LR
        assert sched.get_lr(200) == pytest.approx(1e-4)
        # Next step, elapsed=1==grace → expires
        assert sched.get_lr(201) == pytest.approx(sched._scheduled_lr(201))

    def test_override_zero_grace_clamps_to_1(self):
        sched = _make_schedule()
        sched.override_lr(lr=1e-4, step=200, grace_period=0)
        assert sched._override_grace == 1


# ---------------------------------------------------------------------------
# LossGuard escalation tests
# ---------------------------------------------------------------------------

class _FakeLogger:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def info(self, msg, *args):
        self.infos.append(msg % args if args else msg)


def _make_guard(sched=None, **kwargs):
    if sched is None:
        sched = _make_schedule()
    logger = _FakeLogger()
    defaults = dict(
        window=20, warmup=10, cooldown=5, max_reductions=3,
        max_overrides=2, grace_period=50,
        loss_std_mult=3.0, loss_abs_min=0.5,
        grad_relative=10.0, grad_absolute=100.0, lr_factor=0.5,
    )
    defaults.update(kwargs)
    return LossGuard(sched, logger, **defaults), sched, logger


class TestLossGuardEscalation:
    def _fill_history(self, guard, steps, loss=2.0, grad=1.0):
        """Feed normal values to fill the warmup window."""
        for s in range(steps):
            guard.step(loss, grad, s)

    def test_normal_values_no_trigger(self):
        guard, sched, logger = _make_guard()
        self._fill_history(guard, 20)
        assert not sched.has_override
        assert sched.base_lr == 1e-3
        assert len(logger.warnings) == 0

    def test_loss_spike_triggers_temporary_override(self):
        guard, sched, logger = _make_guard()
        self._fill_history(guard, 15, loss=2.0, grad=1.0)
        # Inject a spike
        guard.step(20.0, 1.0, 15)
        assert sched.has_override
        assert sched.base_lr == 1e-3  # base_lr untouched
        assert "override" in logger.warnings[-1].lower()
        assert "temporary" in logger.warnings[-1].lower()

    def test_grad_explosion_triggers_temporary_override(self):
        guard, sched, logger = _make_guard()
        self._fill_history(guard, 15, loss=2.0, grad=1.0)
        # Inject gradient explosion (10x mean)
        guard.step(2.0, 50.0, 15)
        assert sched.has_override
        assert sched.base_lr == 1e-3

    def test_escalates_to_permanent_after_max_overrides(self):
        guard, sched, logger = _make_guard(max_overrides=2, cooldown=1)
        self._fill_history(guard, 15, loss=2.0, grad=1.0)

        # First spike → temporary override
        guard.step(100.0, 1.0, 15)
        assert sched.has_override
        assert guard.num_overrides == 1
        assert guard.num_reductions == 0

        # Refill with normal values to stabilise the window, then spike again
        sched._override_lr = None
        for s in range(16, 25):
            guard.step(2.0, 1.0, s)

        # Second spike → temporary override
        guard.step(100.0, 1.0, 25)
        assert guard.num_overrides == 2
        assert guard.num_reductions == 0

        # Refill again
        sched._override_lr = None
        for s in range(26, 35):
            guard.step(2.0, 1.0, s)

        # Third spike → permanent reduction (max_overrides=2 exhausted)
        guard.step(100.0, 1.0, 35)
        assert guard.num_reductions == 1
        assert sched.base_lr == pytest.approx(5e-4)
        assert "permanent" in logger.warnings[-1].lower()
        # Override counter resets after permanent reduction
        assert guard.num_overrides == 0

    def test_non_finite_triggers_reduction(self):
        guard, sched, logger = _make_guard()
        guard.step(float('nan'), 1.0, 0)
        # Non-finite should still trigger (override or permanent)
        assert len(logger.warnings) == 1

    def test_cooldown_prevents_rapid_triggers(self):
        guard, sched, logger = _make_guard(cooldown=10)
        self._fill_history(guard, 15, loss=2.0, grad=1.0)
        guard.step(20.0, 1.0, 15)  # triggers
        assert len(logger.warnings) == 1
        guard.step(20.0, 1.0, 16)  # within cooldown, no trigger
        assert len(logger.warnings) == 1

    def test_max_reductions_respected(self):
        guard, sched, logger = _make_guard(
            max_overrides=0, max_reductions=2, cooldown=1
        )
        self._fill_history(guard, 15, loss=2.0, grad=1.0)

        # First permanent reduction
        guard.step(100.0, 1.0, 15)
        assert guard.num_reductions == 1

        # Refill history with normal values to get a clean window
        for s in range(16, 30):
            guard.step(2.0, 1.0, s)

        # Second permanent reduction
        guard.step(100.0, 1.0, 30)
        assert guard.num_reductions == 2

        # Refill again
        for s in range(31, 45):
            guard.step(2.0, 1.0, s)

        # Should not trigger any more
        guard.step(100.0, 1.0, 45)
        assert guard.num_reductions == 2

    def test_override_factor_applied_correctly(self):
        sched = _make_schedule(base_lr=1e-3)
        guard, sched, logger = _make_guard(sched=sched, lr_factor=0.25)
        self._fill_history(guard, 15, loss=2.0, grad=1.0)
        guard.step(20.0, 1.0, 15)
        # Override LR should be scheduled_lr * 0.25
        expected = sched._scheduled_lr(15) * 0.25
        assert sched._override_lr == pytest.approx(expected)
