"""Tests for TrainingAdvisor cross-component intelligence."""

import logging

import pytest

from surogate.train.gradient_tracker import GradientTracker
from surogate.train.lr_schedule import LRSchedule
from surogate.train.metrics import MoEMetrics, StepMetrics
from surogate.train.moe_monitor import MoEMonitor
from surogate.train.phase_detector import PhaseDetector, TrainingPhase
from surogate.train.plateau_detector import PlateauDetector
from surogate.train.training_advisor import TrainingAdvisor


# ---------------------------------------------------------------------------
# Helpers — minimal stubs that expose the state the advisor reads
# ---------------------------------------------------------------------------

class FakeLossGuard:
    """Minimal stub exposing the fields TrainingAdvisor reads."""

    def __init__(self):
        self.num_reductions = 0
        self.num_overrides = 0
        self.last_trigger_step = -1000


def _make_schedule(base_lr=1e-3, max_steps=1000):
    return LRSchedule(
        base_lr=base_lr, max_steps=max_steps, warmup_steps=100,
        cooldown_steps=0, final_lr=1e-5, schedule_type="cosine",
    )


def _make_advisor(
    logger,
    phase_detector=None,
    gradient_tracker=None,
    plateau_detector=None,
    loss_guard=None,
    moe_monitor=None,
    lr_schedule=None,
    max_steps=1000,
    warmup_steps=100,
    cooldown=0,
):
    return TrainingAdvisor(
        logger=logger,
        phase_detector=phase_detector or PhaseDetector(logger),
        gradient_tracker=gradient_tracker or GradientTracker(logger),
        plateau_detector=plateau_detector or PlateauDetector(logger),
        loss_guard=loss_guard,
        moe_monitor=moe_monitor or MoEMonitor(logger),
        lr_schedule=lr_schedule or _make_schedule(),
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        cooldown=cooldown,
    )


def _make_metrics(step=0, loss=2.0, grad_norm=1.0, lr=1e-3, phase="converging", moe=None):
    return StepMetrics(
        step=step, loss=loss, grad_norm=grad_norm, lr=lr,
        phase=phase, moe=moe,
    )


@pytest.fixture
def logger():
    return logging.getLogger("test_training_advisor")


# ---------------------------------------------------------------------------
# Rule 1: Plateau + flat gradients + high LR
# ---------------------------------------------------------------------------

class TestPlateauHighLR:
    def test_warns_on_sustained_plateau_with_flat_grads(self, logger, caplog):
        phase_det = PhaseDetector(logger)
        grad_tracker = GradientTracker(logger, window=20, warmup=5)
        sched = _make_schedule(base_lr=1e-3, max_steps=1000)

        advisor = _make_advisor(
            logger, phase_detector=phase_det, gradient_tracker=grad_tracker,
            lr_schedule=sched, cooldown=0,
        )

        # Force phase to PLATEAU
        phase_det.current_phase = TrainingPhase.PLATEAU

        # Feed flat gradient norms so trend ≈ 0
        for i in range(60):
            grad_tracker.step(1.0, step=i)
            metrics = _make_metrics(step=i, lr=1e-3, phase="plateau")
            with caplog.at_level(logging.WARNING):
                advisor.step(metrics, i)

        assert "[Advisor] Plateau with flat gradients" in caplog.text

    def test_no_warn_when_lr_already_low(self, logger, caplog):
        phase_det = PhaseDetector(logger)
        grad_tracker = GradientTracker(logger, window=20, warmup=5)
        sched = _make_schedule(base_lr=1e-3, max_steps=1000)

        advisor = _make_advisor(
            logger, phase_detector=phase_det, gradient_tracker=grad_tracker,
            lr_schedule=sched, cooldown=0,
        )

        phase_det.current_phase = TrainingPhase.PLATEAU

        for i in range(60):
            grad_tracker.step(1.0, step=i)
            # LR is very low compared to scheduled
            metrics = _make_metrics(step=i, lr=1e-6, phase="plateau")
            with caplog.at_level(logging.WARNING):
                advisor.step(metrics, i)

        assert "Plateau with flat gradients" not in caplog.text

    def test_counter_resets_when_not_plateau(self, logger, caplog):
        phase_det = PhaseDetector(logger)
        grad_tracker = GradientTracker(logger, window=20, warmup=5)

        advisor = _make_advisor(
            logger, phase_detector=phase_det, gradient_tracker=grad_tracker,
            cooldown=0,
        )

        # 40 steps of plateau
        phase_det.current_phase = TrainingPhase.PLATEAU
        for i in range(40):
            grad_tracker.step(1.0, step=i)
            advisor.step(_make_metrics(step=i), i)

        # Switch to converging — counter should reset
        phase_det.current_phase = TrainingPhase.CONVERGING
        advisor.step(_make_metrics(step=40), 40)
        assert advisor._plateau_grad_flat_count == 0

        # Back to plateau — needs another 50 steps before warning
        phase_det.current_phase = TrainingPhase.PLATEAU
        for i in range(41, 80):
            grad_tracker.step(1.0, step=i)
            with caplog.at_level(logging.WARNING):
                advisor.step(_make_metrics(step=i), i)

        assert "Plateau with flat gradients" not in caplog.text


# ---------------------------------------------------------------------------
# Rule 2: Diverging + gradient trend up + MoE imbalance
# ---------------------------------------------------------------------------

class TestDivergingMoECollapse:
    def test_warns_on_diverging_with_moe_issues(self, logger, caplog):
        phase_det = PhaseDetector(logger)
        grad_tracker = GradientTracker(logger, window=10, warmup=5)
        moe_mon = MoEMonitor(logger, warmup=3, cooldown=0)

        advisor = _make_advisor(
            logger, phase_detector=phase_det, gradient_tracker=grad_tracker,
            moe_monitor=moe_mon, cooldown=0,
        )

        phase_det.current_phase = TrainingPhase.DIVERGING

        # Feed steeply increasing gradient norms (strong positive trend)
        # Need relative_trend = trend/mean > 0.1
        for i in range(15):
            grad_tracker.step(0.1 + i * 1.0, step=i)

        # Feed bad MoE metrics so diagnostics are unhealthy
        for i in range(10):
            moe_mon.step(MoEMetrics(
                aux_loss=0.1, z_loss=0.01,
                load_imbalance=12.0, expert_utilization=0.3,
            ), step=i)

        moe_metrics = MoEMetrics(
            aux_loss=0.1, z_loss=0.01,
            load_imbalance=12.0, expert_utilization=0.3,
        )
        metrics = _make_metrics(step=15, phase="diverging", moe=moe_metrics)

        with caplog.at_level(logging.WARNING):
            advisor.step(metrics, 15)

        assert "[Advisor] MoE routing collapse" in caplog.text
        assert "router_aux_loss_coef" in caplog.text

    def test_no_warn_without_moe(self, logger, caplog):
        phase_det = PhaseDetector(logger)
        grad_tracker = GradientTracker(logger, window=10, warmup=5)

        advisor = _make_advisor(
            logger, phase_detector=phase_det, gradient_tracker=grad_tracker,
            cooldown=0,
        )

        phase_det.current_phase = TrainingPhase.DIVERGING
        for i in range(15):
            grad_tracker.step(1.0 + i * 0.5, step=i)

        # No MoE metrics
        metrics = _make_metrics(step=15, phase="diverging", moe=None)
        with caplog.at_level(logging.WARNING):
            advisor.step(metrics, 15)

        assert "MoE routing collapse" not in caplog.text


# ---------------------------------------------------------------------------
# Rule 3: Converging + gradient shrinking + loss stalling
# ---------------------------------------------------------------------------

class TestGradientVanishing:
    def test_warns_on_vanishing_gradients(self, logger, caplog):
        phase_det = PhaseDetector(logger)
        grad_tracker = GradientTracker(logger, window=10, warmup=5)
        plateau_det = PlateauDetector(logger, warmup=5, window=20)

        advisor = _make_advisor(
            logger, phase_detector=phase_det, gradient_tracker=grad_tracker,
            plateau_detector=plateau_det, cooldown=0,
        )

        phase_det.current_phase = TrainingPhase.CONVERGING

        # Feed steadily decreasing gradient norms AND call advisor.step()
        # to build up _converging_grad_shrink_count (needs 30+ consecutive steps).
        # Need relative_trend < -0.02 sustained. Use linear decrease that
        # stays well above zero for the entire 50-step window.
        with caplog.at_level(logging.WARNING):
            for i in range(50):
                norm = 10.0 - i * 0.15  # 10.0 down to 2.65 at step 49
                grad_tracker.step(norm, step=i)
                if i >= 10:
                    plateau_det.plateau_count = 3
                metrics = _make_metrics(step=i, phase="converging")
                advisor.step(metrics, i)

        assert "[Advisor] Gradient vanishing" in caplog.text

    def test_no_warn_if_loss_still_improving(self, logger, caplog):
        phase_det = PhaseDetector(logger)
        grad_tracker = GradientTracker(logger, window=10, warmup=5)
        plateau_det = PlateauDetector(logger)

        advisor = _make_advisor(
            logger, phase_detector=phase_det, gradient_tracker=grad_tracker,
            plateau_detector=plateau_det, cooldown=0,
        )

        phase_det.current_phase = TrainingPhase.CONVERGING
        # Plateau count stays 0 — loss is still improving
        plateau_det.plateau_count = 0

        with caplog.at_level(logging.WARNING):
            for i in range(50):
                norm = 10.0 - i * 0.15
                grad_tracker.step(norm, step=i)
                metrics = _make_metrics(step=i, phase="converging")
                advisor.step(metrics, i)

        assert "Gradient vanishing" not in caplog.text


# ---------------------------------------------------------------------------
# Rule 4: Loss spike + MoE utilization drop
# ---------------------------------------------------------------------------

class TestSpikeMoECorrelation:
    def test_warns_when_spike_and_low_utilization(self, logger, caplog):
        loss_guard = FakeLossGuard()
        loss_guard.last_trigger_step = 95  # triggered recently
        moe_mon = MoEMonitor(logger, warmup=3, utilization_warn=0.8, cooldown=0)

        advisor = _make_advisor(
            logger, loss_guard=loss_guard, moe_monitor=moe_mon, cooldown=0,
        )

        moe_metrics = MoEMetrics(
            aux_loss=0.05, z_loss=0.01,
            load_imbalance=5.0, expert_utilization=0.4,
        )
        metrics = _make_metrics(step=100, moe=moe_metrics)

        with caplog.at_level(logging.WARNING):
            advisor.step(metrics, 100)

        assert "[Advisor] Loss spike correlates with MoE" in caplog.text

    def test_no_warn_when_spike_old(self, logger, caplog):
        loss_guard = FakeLossGuard()
        loss_guard.last_trigger_step = 50  # too long ago

        advisor = _make_advisor(
            logger, loss_guard=loss_guard, cooldown=0,
        )

        moe_metrics = MoEMetrics(
            aux_loss=0.05, z_loss=0.01,
            load_imbalance=5.0, expert_utilization=0.4,
        )
        metrics = _make_metrics(step=100, moe=moe_metrics)

        with caplog.at_level(logging.WARNING):
            advisor.step(metrics, 100)

        assert "Loss spike correlates" not in caplog.text

    def test_no_warn_without_loss_guard(self, logger, caplog):
        advisor = _make_advisor(logger, loss_guard=None, cooldown=0)

        moe_metrics = MoEMetrics(
            aux_loss=0.05, z_loss=0.01,
            load_imbalance=5.0, expert_utilization=0.4,
        )
        metrics = _make_metrics(step=100, moe=moe_metrics)

        with caplog.at_level(logging.WARNING):
            advisor.step(metrics, 100)

        assert "Loss spike correlates" not in caplog.text


# ---------------------------------------------------------------------------
# Rule 5: Multiple LR reductions with no improvement
# ---------------------------------------------------------------------------

class TestLRReductionsIneffective:
    def test_warns_after_multiple_reductions_no_converging(self, logger, caplog):
        phase_det = PhaseDetector(logger)
        loss_guard = FakeLossGuard()
        loss_guard.num_reductions = 3

        advisor = _make_advisor(
            logger, phase_detector=phase_det, loss_guard=loss_guard, cooldown=0,
        )

        phase_det.current_phase = TrainingPhase.UNSTABLE

        metrics = _make_metrics(step=500, phase="unstable")
        with caplog.at_level(logging.WARNING):
            advisor.step(metrics, 500)

        assert "[Advisor] LR reductions not helping" in caplog.text
        assert "data quality" in caplog.text

    def test_no_warn_if_converging(self, logger, caplog):
        phase_det = PhaseDetector(logger)
        loss_guard = FakeLossGuard()
        loss_guard.num_reductions = 3

        advisor = _make_advisor(
            logger, phase_detector=phase_det, loss_guard=loss_guard, cooldown=0,
        )

        phase_det.current_phase = TrainingPhase.CONVERGING

        metrics = _make_metrics(step=500, phase="converging")
        with caplog.at_level(logging.WARNING):
            advisor.step(metrics, 500)

        assert "LR reductions not helping" not in caplog.text

    def test_no_warn_with_single_reduction(self, logger, caplog):
        phase_det = PhaseDetector(logger)
        loss_guard = FakeLossGuard()
        loss_guard.num_reductions = 1

        advisor = _make_advisor(
            logger, phase_detector=phase_det, loss_guard=loss_guard, cooldown=0,
        )

        phase_det.current_phase = TrainingPhase.DIVERGING

        metrics = _make_metrics(step=500, phase="diverging")
        with caplog.at_level(logging.WARNING):
            advisor.step(metrics, 500)

        assert "LR reductions not helping" not in caplog.text


# ---------------------------------------------------------------------------
# Rule 6: Warmup too short
# ---------------------------------------------------------------------------

class TestWarmupTooShort:
    def test_warns_when_loss_still_dropping_at_warmup_end(self, logger, caplog):
        plateau_det = PlateauDetector(logger, warmup=5, window=20)

        advisor = _make_advisor(
            logger, plateau_detector=plateau_det,
            warmup_steps=100, cooldown=0,
        )

        # Simulate loss history with a steep downward trend at warmup boundary
        # Loss dropping from 5.0 to 3.0 over 10 steps = 40% drop
        for i in range(10):
            plateau_det.history.append(5.0 - i * 0.2)

        grad_tracker = advisor.gradient_tracker
        for i in range(10):
            grad_tracker.step(1.0, step=i)

        metrics = _make_metrics(step=100)
        with caplog.at_level(logging.INFO):
            advisor.step(metrics, 100)

        assert "[Advisor] Warmup may be too short" in caplog.text

    def test_no_warn_when_loss_flat(self, logger, caplog):
        plateau_det = PlateauDetector(logger, warmup=5, window=20)

        advisor = _make_advisor(
            logger, plateau_detector=plateau_det,
            warmup_steps=100, cooldown=0,
        )

        # Flat loss at warmup boundary
        for i in range(10):
            plateau_det.history.append(2.0)

        grad_tracker = advisor.gradient_tracker
        for i in range(10):
            grad_tracker.step(1.0, step=i)

        metrics = _make_metrics(step=100)
        with caplog.at_level(logging.INFO):
            advisor.step(metrics, 100)

        assert "Warmup may be too short" not in caplog.text

    def test_one_shot_only(self, logger, caplog):
        plateau_det = PlateauDetector(logger, warmup=5, window=20)

        advisor = _make_advisor(
            logger, plateau_detector=plateau_det,
            warmup_steps=100, cooldown=0,
        )

        for i in range(10):
            plateau_det.history.append(5.0 - i * 0.2)

        grad_tracker = advisor.gradient_tracker
        for i in range(10):
            grad_tracker.step(1.0, step=i)

        with caplog.at_level(logging.INFO):
            advisor.step(_make_metrics(step=100), 100)
            caplog.clear()
            # Second call at step 105 — should NOT fire again
            advisor.step(_make_metrics(step=105), 105)

        assert "Warmup may be too short" not in caplog.text


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------

class TestAdvisorCooldown:
    def test_cooldown_suppresses_repeated_warnings(self, logger, caplog):
        phase_det = PhaseDetector(logger)
        loss_guard = FakeLossGuard()
        loss_guard.num_reductions = 3

        advisor = _make_advisor(
            logger, phase_detector=phase_det, loss_guard=loss_guard,
            cooldown=50,
        )

        phase_det.current_phase = TrainingPhase.UNSTABLE

        with caplog.at_level(logging.WARNING):
            for i in range(20):
                advisor.step(_make_metrics(step=i, phase="unstable"), i)

        warnings = [r for r in caplog.records if "LR reductions not helping" in r.message]
        assert len(warnings) == 1

    def test_warning_repeats_after_cooldown(self, logger, caplog):
        phase_det = PhaseDetector(logger)
        loss_guard = FakeLossGuard()
        loss_guard.num_reductions = 3

        advisor = _make_advisor(
            logger, phase_detector=phase_det, loss_guard=loss_guard,
            cooldown=10,
        )

        phase_det.current_phase = TrainingPhase.UNSTABLE

        with caplog.at_level(logging.WARNING):
            advisor.step(_make_metrics(step=0, phase="unstable"), 0)
            advisor.step(_make_metrics(step=5, phase="unstable"), 5)   # within cooldown
            advisor.step(_make_metrics(step=10, phase="unstable"), 10)  # after cooldown

        warnings = [r for r in caplog.records if "LR reductions not helping" in r.message]
        assert len(warnings) == 2
