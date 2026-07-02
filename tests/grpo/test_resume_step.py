from types import SimpleNamespace

import pytest

from surogate.grpo.runs import MultiRunManager, _start_step_override
from surogate.grpo.trainer import _latest_checkpoint_step, _weights_path_for_start_step


def test_run_manager_uses_start_step_override_without_ckpt(monkeypatch, tmp_path):
    monkeypatch.setenv("SUROGATE_GRPO_START_STEP", "25")

    manager = MultiRunManager(tmp_path, max_runs=1)
    manager._create_run_data("run_default", 0, SimpleNamespace(ckpt=None))

    assert manager.progress[0].step == 25


def test_run_manager_ckpt_resume_step_takes_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("SUROGATE_GRPO_START_STEP", "25")

    config = SimpleNamespace(ckpt=SimpleNamespace(resume_step=7))
    manager = MultiRunManager(tmp_path, max_runs=1)
    manager._create_run_data("run_default", 0, config)

    assert manager.progress[0].step == 7


def test_start_step_override_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("SUROGATE_GRPO_START_STEP", "not-an-int")

    with pytest.raises(ValueError, match="Invalid SUROGATE_GRPO_START_STEP"):
        _start_step_override()


def test_latest_checkpoint_step_respects_resume_flag(monkeypatch, tmp_path):
    def fail_if_called(_path):
        raise AssertionError("find_latest_checkpoint should not be called")

    monkeypatch.setattr("surogate.grpo.trainer._surogate.find_latest_checkpoint", fail_if_called)

    config = SimpleNamespace(resume_from_checkpoint=False, checkpoint_dir=str(tmp_path))

    assert _latest_checkpoint_step(config) == 0


def test_latest_checkpoint_step_reads_latest_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr("surogate.grpo.trainer._surogate.find_latest_checkpoint", lambda _path: 25)

    config = SimpleNamespace(resume_from_checkpoint=True, checkpoint_dir=str(tmp_path))

    assert _latest_checkpoint_step(config) == 25


def test_weights_path_for_start_step_uses_checkpoint_for_full_model(tmp_path):
    checkpoint_dir = tmp_path / "step_00000025"
    checkpoint_dir.mkdir()
    checkpoint_weights = checkpoint_dir / "model.safetensors"
    checkpoint_weights.write_text("weights")

    config = SimpleNamespace(checkpoint_dir=str(tmp_path), lora=False)

    assert _weights_path_for_start_step(config, "base.safetensors", 25) == str(checkpoint_weights)


def test_weights_path_for_start_step_keeps_base_for_lora(tmp_path):
    checkpoint_dir = tmp_path / "step_00000025"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_text("weights")

    config = SimpleNamespace(checkpoint_dir=str(tmp_path), lora=True)

    assert _weights_path_for_start_step(config, "base.safetensors", 25) == "base.safetensors"
