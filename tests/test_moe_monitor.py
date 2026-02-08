"""Tests for MoEMonitor."""

import logging

import pytest

from surogate.train.metrics import MoEMetrics
from surogate.train.moe_monitor import MoEMonitor, RoutingDiagnostics


@pytest.fixture
def logger():
    return logging.getLogger("test_moe_monitor")


def _make_moe(
    aux_loss=0.01, z_loss=0.001, load_imbalance=1.2, expert_utilization=0.95
):
    return MoEMetrics(
        aux_loss=aux_loss,
        z_loss=z_loss,
        load_imbalance=load_imbalance,
        expert_utilization=expert_utilization,
    )


class TestMoEMonitorStep:
    def test_none_is_noop(self, logger):
        mon = MoEMonitor(logger, warmup=3)
        mon.step(None, step=0)
        assert len(mon._aux_losses) == 0

    def test_collects_history(self, logger):
        mon = MoEMonitor(logger, warmup=3, window=10)
        for i in range(5):
            mon.step(_make_moe(), step=i)
        assert len(mon._aux_losses) == 5
        assert len(mon._imbalances) == 5

    def test_window_evicts_old(self, logger):
        mon = MoEMonitor(logger, warmup=3, window=5)
        for i in range(10):
            mon.step(_make_moe(aux_loss=float(i)), step=i)
        assert len(mon._aux_losses) == 5
        assert list(mon._aux_losses) == [5.0, 6.0, 7.0, 8.0, 9.0]


class TestImbalanceWarnings:
    def test_severe_imbalance_warns(self, logger, caplog):
        mon = MoEMonitor(logger, warmup=3, imbalance_severe=10.0, cooldown=0)
        # Fill warmup with normal values
        for i in range(3):
            mon.step(_make_moe(), step=i)
        # Now spike imbalance
        with caplog.at_level(logging.WARNING):
            mon.step(_make_moe(load_imbalance=15.0), step=3)
        assert "Severe routing imbalance" in caplog.text

    def test_moderate_imbalance_warns(self, logger, caplog):
        mon = MoEMonitor(
            logger, warmup=3, imbalance_warn=3.0, imbalance_severe=10.0, cooldown=0
        )
        for i in range(3):
            mon.step(_make_moe(), step=i)
        with caplog.at_level(logging.WARNING):
            mon.step(_make_moe(load_imbalance=5.0), step=3)
        assert "Routing imbalance" in caplog.text
        assert "Severe" not in caplog.text

    def test_normal_imbalance_no_warning(self, logger, caplog):
        mon = MoEMonitor(logger, warmup=3, imbalance_warn=3.0, cooldown=0)
        for i in range(5):
            with caplog.at_level(logging.WARNING):
                mon.step(_make_moe(load_imbalance=1.5), step=i)
        assert "imbalance" not in caplog.text.lower()


class TestUtilizationWarnings:
    def test_critical_utilization_warns(self, logger, caplog):
        mon = MoEMonitor(
            logger, warmup=3, utilization_critical=0.5, cooldown=0
        )
        for i in range(3):
            mon.step(_make_moe(), step=i)
        with caplog.at_level(logging.WARNING):
            mon.step(_make_moe(expert_utilization=0.3), step=3)
        assert "Expert collapse" in caplog.text

    def test_low_utilization_warns(self, logger, caplog):
        mon = MoEMonitor(
            logger, warmup=3, utilization_warn=0.8, utilization_critical=0.5, cooldown=0
        )
        for i in range(3):
            mon.step(_make_moe(), step=i)
        with caplog.at_level(logging.WARNING):
            mon.step(_make_moe(expert_utilization=0.7), step=3)
        assert "Low expert utilization" in caplog.text

    def test_healthy_utilization_no_warning(self, logger, caplog):
        mon = MoEMonitor(logger, warmup=3, cooldown=0)
        for i in range(5):
            with caplog.at_level(logging.WARNING):
                mon.step(_make_moe(expert_utilization=0.95), step=i)
        assert "utilization" not in caplog.text.lower()


class TestAuxLossSpikeWarning:
    def test_spike_detected(self, logger, caplog):
        mon = MoEMonitor(
            logger, warmup=5, aux_loss_spike_sigma=2.0, cooldown=0
        )
        # Stable aux_loss
        for i in range(10):
            mon.step(_make_moe(aux_loss=0.01), step=i)
        # Big spike
        with caplog.at_level(logging.WARNING):
            mon.step(_make_moe(aux_loss=1.0), step=10)
        assert "Aux-loss spike" in caplog.text

    def test_no_spike_for_normal_variation(self, logger, caplog):
        mon = MoEMonitor(logger, warmup=5, aux_loss_spike_sigma=3.0, cooldown=0)
        for i in range(10):
            mon.step(_make_moe(aux_loss=0.01 + i * 0.001), step=i)
        with caplog.at_level(logging.WARNING):
            mon.step(_make_moe(aux_loss=0.02), step=10)
        assert "Aux-loss spike" not in caplog.text


class TestCooldown:
    def test_cooldown_suppresses_repeated_warnings(self, logger, caplog):
        mon = MoEMonitor(
            logger, warmup=3, imbalance_severe=5.0, cooldown=50
        )
        for i in range(3):
            mon.step(_make_moe(), step=i)
        with caplog.at_level(logging.WARNING):
            for i in range(3, 20):
                mon.step(_make_moe(load_imbalance=15.0), step=i)
        warnings = [r for r in caplog.records if "Severe routing" in r.message]
        assert len(warnings) == 1

    def test_warning_repeats_after_cooldown(self, logger, caplog):
        mon = MoEMonitor(
            logger, warmup=3, imbalance_severe=5.0, cooldown=10
        )
        for i in range(3):
            mon.step(_make_moe(), step=i)
        with caplog.at_level(logging.WARNING):
            # First warning at step 3
            mon.step(_make_moe(load_imbalance=15.0), step=3)
            # Within cooldown — no second warning
            mon.step(_make_moe(load_imbalance=15.0), step=5)
            # After cooldown — second warning
            mon.step(_make_moe(load_imbalance=15.0), step=13)
        warnings = [r for r in caplog.records if "Severe routing" in r.message]
        assert len(warnings) == 2


class TestRoutingDiagnostics:
    def test_healthy_defaults(self, logger):
        mon = MoEMonitor(logger, warmup=5)
        diag = mon.get_routing_diagnostics()
        assert diag.healthy is True
        assert diag.recommendations == []

    def test_healthy_with_good_metrics(self, logger):
        mon = MoEMonitor(logger, warmup=5)
        for i in range(10):
            mon.step(_make_moe(), step=i)
        diag = mon.get_routing_diagnostics()
        assert diag.healthy is True
        assert diag.avg_aux_loss == pytest.approx(0.01)
        assert diag.avg_load_imbalance == pytest.approx(1.2)
        assert diag.avg_expert_utilization == pytest.approx(0.95)
        assert diag.balance_score == pytest.approx(1.0 / 1.2, rel=1e-3)
        assert diag.utilization_score == pytest.approx(0.95)
        assert diag.recommendations == []

    def test_severe_imbalance_recommendation(self, logger):
        mon = MoEMonitor(logger, warmup=3, cooldown=0)
        for i in range(10):
            mon.step(_make_moe(load_imbalance=8.0), step=i)
        diag = mon.get_routing_diagnostics()
        assert diag.healthy is False
        assert diag.balance_score < 0.3
        assert any("Severe" in r for r in diag.recommendations)

    def test_moderate_imbalance_recommendation(self, logger):
        mon = MoEMonitor(logger, warmup=3, cooldown=0)
        for i in range(10):
            mon.step(_make_moe(load_imbalance=2.0), step=i)
        diag = mon.get_routing_diagnostics()
        assert diag.healthy is False
        assert diag.balance_score < 0.7
        assert any("router_aux_loss_coef" in r for r in diag.recommendations)

    def test_low_utilization_recommendation(self, logger):
        mon = MoEMonitor(logger, warmup=3, cooldown=0)
        for i in range(10):
            mon.step(_make_moe(expert_utilization=0.6), step=i)
        diag = mon.get_routing_diagnostics()
        assert diag.healthy is False
        assert any("underused" in r for r in diag.recommendations)

    def test_expert_collapse_recommendation(self, logger):
        mon = MoEMonitor(logger, warmup=3, cooldown=0)
        for i in range(10):
            mon.step(_make_moe(expert_utilization=0.3), step=i)
        diag = mon.get_routing_diagnostics()
        assert diag.healthy is False
        assert any("collapse" in r.lower() for r in diag.recommendations)

    def test_aux_loss_trend_recommendation(self, logger):
        mon = MoEMonitor(logger, warmup=3, cooldown=0)
        # First half: low aux_loss; second half: much higher
        for i in range(5):
            mon.step(_make_moe(aux_loss=0.01), step=i)
        for i in range(5, 10):
            mon.step(_make_moe(aux_loss=0.10), step=i)
        diag = mon.get_routing_diagnostics()
        assert diag.aux_loss_trend > 0
        assert any("router_z_loss_coef" in r for r in diag.recommendations)

    def test_not_active_before_warmup(self, logger):
        mon = MoEMonitor(logger, warmup=10)
        for i in range(5):
            mon.step(_make_moe(load_imbalance=100.0), step=i)
        diag = mon.get_routing_diagnostics()
        # Before warmup, diagnostics return healthy defaults
        assert diag.healthy is True
        assert diag.recommendations == []
