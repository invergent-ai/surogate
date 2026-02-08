"""Tests for StepMetrics and MoEMetrics."""

import pytest

from surogate.train.metrics import MoEMetrics, StepMetrics


class TestMoEMetrics:
    def test_from_dict_valid(self):
        d = {
            "valid": True,
            "aux_loss": 0.1,
            "z_loss": 0.2,
            "load_imbalance": 1.5,
            "expert_utilization": 0.8,
        }
        m = MoEMetrics.from_dict(d)
        assert m is not None
        assert m.aux_loss == 0.1
        assert m.z_loss == 0.2
        assert m.load_imbalance == 1.5
        assert m.expert_utilization == 0.8

    def test_from_dict_invalid(self):
        assert MoEMetrics.from_dict({"valid": False}) is None

    def test_from_dict_empty(self):
        assert MoEMetrics.from_dict({}) is None

    def test_from_dict_missing_keys(self):
        m = MoEMetrics.from_dict({"valid": True})
        assert m is not None
        assert m.aux_loss == 0.0


class TestStepMetrics:
    def test_tokens_per_second(self):
        m = StepMetrics(tokens=10_000, elapsed_ms=500)
        assert m.tokens_per_second == pytest.approx(20_000.0)

    def test_tokens_per_second_zero_time(self):
        m = StepMetrics(tokens=10_000, elapsed_ms=0)
        assert m.tokens_per_second == 0.0

    def test_to_dict_without_moe(self):
        m = StepMetrics(step=42, epoch=1.5, loss=2.3, grad_norm=0.5,
                        lr=1e-4, tokens=8192, elapsed_ms=100, phase="converging")
        d = m.to_dict()
        assert d["step"] == 42
        assert d["epoch"] == 1.5
        assert d["loss"] == 2.3
        assert d["grad_norm"] == 0.5
        assert d["lr"] == 1e-4
        assert d["tokens"] == 8192
        assert d["elapsed_ms"] == 100
        assert d["phase"] == "converging"
        assert d["lr_overridden"] is False
        assert "tokens_per_second" in d
        assert "moe_aux_loss" not in d

    def test_to_dict_with_moe(self):
        moe = MoEMetrics(aux_loss=0.1, z_loss=0.2, load_imbalance=1.5, expert_utilization=0.8)
        m = StepMetrics(step=1, moe=moe)
        d = m.to_dict()
        assert d["moe_aux_loss"] == 0.1
        assert d["moe_z_loss"] == 0.2
        assert d["moe_load_imbalance"] == 1.5
        assert d["moe_expert_utilization"] == 0.8

    def test_lr_overridden_flag(self):
        m = StepMetrics(lr_overridden=True)
        assert m.to_dict()["lr_overridden"] is True

    def test_defaults(self):
        m = StepMetrics()
        assert m.step == 0
        assert m.loss == 0.0
        assert m.moe is None
        assert m.phase == ""
