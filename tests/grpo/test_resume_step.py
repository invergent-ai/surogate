from types import SimpleNamespace

import pytest

from surogate.grpo.runs import MultiRunManager, _start_step_override
from surogate.grpo.trainer import (
    _latest_checkpoint_step,
    _load_initial_trainable_adapter,
    _set_initial_adapter,
    _weights_path_for_start_step,
)


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


def test_set_initial_adapter_seeds_fresh_lora_run(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    config = SimpleNamespace(adapter_path=str(adapter), lora=True)

    class Trainer:
        path = None

        def set_adapter_path(self, path):
            self.path = path

    trainer = Trainer()

    selected = _set_initial_adapter(config, trainer, start_step=0)

    assert selected == str(adapter.resolve())
    assert trainer.path == selected


def test_set_initial_adapter_does_not_override_resumed_checkpoint(tmp_path):
    config = SimpleNamespace(adapter_path=str(tmp_path / "missing"), lora=True)

    class Trainer:
        def set_adapter_path(self, path):
            raise AssertionError(path)

    assert _set_initial_adapter(config, Trainer(), start_step=1) is None


def test_trainable_initial_adapter_loads_parent_after_base_import(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        '{"r": 32, "lora_alpha": 32, "target_modules": ["q_proj"]}'
    )
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"weights")
    config = SimpleNamespace(
        adapter_path=str(adapter),
        adapter_init_mode="trainable",
        lora=True,
        lora_rank=32,
        lora_alpha=32,
        lora_target_modules=["q_proj"],
    )

    class Trainer:
        imported = None

        def set_adapter_path(self, path):
            raise AssertionError(path)

        def import_adapter(self, path):
            self.imported = path

    trainer = Trainer()
    selected = _set_initial_adapter(config, trainer, start_step=0)

    assert selected == str(adapter.resolve())
    assert trainer.imported is None
    _load_initial_trainable_adapter(config, trainer, selected)
    assert trainer.imported == str(weights.resolve())


def test_trainable_initial_adapter_rejects_shape_mismatch(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        '{"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]}'
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    config = SimpleNamespace(
        adapter_path=str(adapter),
        adapter_init_mode="trainable",
        lora=True,
        lora_rank=32,
        lora_alpha=32,
        lora_target_modules=["q_proj"],
    )

    class Trainer:
        def import_adapter(self, path):
            raise AssertionError(path)

    with pytest.raises(ValueError, match="rank"):
        _set_initial_adapter(config, Trainer(), start_step=0)
