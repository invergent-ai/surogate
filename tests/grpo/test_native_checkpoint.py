"""A restart only consumes a complete, matching pair of policy and rollout state."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from surogate.core.config.grpo_orch_config import GRPOCheckpointConfig
from surogate.grpo.native_checkpoint import NativeCheckpointCoordinator
from surogate.grpo.orchestrator.ckpt import Progress


class Buffer:
    def __init__(self, values=()):
        self.values = list(values)

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "values.json").write_text(json.dumps(self.values))

    def load(self, path):
        self.values = json.loads((path / "values.json").read_text())


class Trainer:
    def save_checkpoint(self, directory, step):
        path = Path(directory) / f"step_{step:08d}"
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_model.safetensors").write_bytes(b"adapter and optimizer fixture")
        (path / "checkpoint.json").write_text(json.dumps(dict(run=dict(step=step))))


def configs(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"llama"}')
    train = SimpleNamespace(model="fixture", model_dir=tmp_path, output_dir=tmp_path / "out",
                            checkpoint_dir=tmp_path / "checkpoints", save_steps=1, max_steps=4,
                            resume_from_checkpoint=True, lora_rank=8, lora_alpha=16,
                            lora_dtype="bf16", lora_target_modules=["all"], sequence_len=256)
    orch = SimpleNamespace(output_dir=train.output_dir / "run_default", ckpt=None)
    return train, orch


def save_pair(coordinator, step, values=(3, 7)):
    coordinator.save_training(Trainer(), step)
    coordinator.save_orchestrator(Progress(step=step, total_tokens=123), Buffer(values), next_group_id=17,
                                  depth_state=dict(_cap=5))


@pytest.mark.parametrize("trainer_first", [False, True])
def test_checkpoint_is_published_only_after_both_components_finish(tmp_path, trainer_first):
    train, orch = configs(tmp_path)
    coordinator = NativeCheckpointCoordinator(train, orch)
    actions = [lambda: coordinator.save_training(Trainer(), 1),
               lambda: coordinator.save_orchestrator(Progress(step=1), Buffer([5]), next_group_id=9, depth_state={})]
    if not trainer_first:
        actions.reverse()
    actions[0]()
    assert not (coordinator.root / "step_1").exists()
    actions[1]()
    restarted = NativeCheckpointCoordinator(train, orch)
    assert restarted.resume_step == 1
    assert restarted.trainer_resume == (str(coordinator.root / "step_1/trainer"), 0)


def test_progress_buffer_rng_and_scheduler_state_are_restored_together(tmp_path):
    train, orch = configs(tmp_path)
    coordinator = NativeCheckpointCoordinator(train, orch)
    random.seed(73)
    np.random.seed(91)
    torch.manual_seed(113)
    save_pair(coordinator, 1)
    expected = (random.random(), np.random.random(), torch.rand(3))
    random.seed(2)
    np.random.seed(2)
    torch.manual_seed(2)
    restored = NativeCheckpointCoordinator(train, orch)
    progress, buffer = Progress(), Buffer([99])
    runtime = restored.load_orchestrator(progress, buffer)
    assert progress.step == 1 and progress.total_tokens == 123
    assert buffer.values == [3, 7]
    assert runtime["next_group_id"] == 17 and runtime["depth_state"] == dict(_cap=5)
    assert random.random() == expected[0] and np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)


def test_partial_or_truncated_newer_checkpoint_is_not_selected(tmp_path):
    train, orch = configs(tmp_path)
    coordinator = NativeCheckpointCoordinator(train, orch)
    save_pair(coordinator, 1)
    coordinator.save_training(Trainer(), 2)
    assert NativeCheckpointCoordinator(train, orch).resume_step == 1
    coordinator.save_orchestrator(Progress(step=2), Buffer(), next_group_id=18, depth_state={})
    (coordinator.root / "step_2/runtime.pt").write_bytes(b"truncated")
    assert NativeCheckpointCoordinator(train, orch).resume_step == 1
    orch.ckpt = GRPOCheckpointConfig({"resume_step": 2})
    with pytest.raises(FileNotFoundError, match="complete native checkpoint"):
        NativeCheckpointCoordinator(train, orch)


def test_explicit_rollback_preserves_stale_work_and_prevents_replaying_it(tmp_path):
    train, orch = configs(tmp_path)
    coordinator = NativeCheckpointCoordinator(train, orch)
    save_pair(coordinator, 1)
    save_pair(coordinator, 2)
    paths = [Path(orch.output_dir) / name for name in ("broadcasts", "rollouts", "control", "live_spool")]
    paths.append(Path(train.output_dir) / "rollouts")
    for path in paths:
        path.mkdir(parents=True)
        (path / "stale").write_text("saved work")
    orch.ckpt = GRPOCheckpointConfig({"resume_step": 1})
    resumed = NativeCheckpointCoordinator(train, orch)
    resumed.prepare_resume()
    assert all(not path.exists() for path in paths)
    assert len(list(tmp_path.rglob("stale"))) == len(paths)
    assert not (coordinator.root / "step_2").exists()
    assert list(coordinator.root.glob("recovery/*/step_2/manifest.json"))
    assert (coordinator.root / "step_1/manifest.json").exists()
    assert orch.ckpt.resume_step == 1


def test_resume_rejects_mismatched_policy_or_partial_state_loading(tmp_path):
    train, orch = configs(tmp_path)
    save_pair(NativeCheckpointCoordinator(train, orch), 1)
    train.lora_alpha = 32
    with pytest.raises(ValueError, match="configuration does not match"):
        NativeCheckpointCoordinator(train, orch)
    train.lora_alpha = 16
    train.optimizer = "adamw"
    with pytest.raises(ValueError, match="configuration does not match"):
        NativeCheckpointCoordinator(train, orch)
    train.optimizer = "adamw_8bit"
    orch.ckpt = GRPOCheckpointConfig({"skip_buffer": True})
    with pytest.raises(ValueError, match="progress and buffer together"):
        NativeCheckpointCoordinator(train, orch)
    orch.ckpt = None
    train.resume_from_checkpoint = False
    with pytest.raises(ValueError, match="fresh output directory"):
        NativeCheckpointCoordinator(train, orch)


def test_checkpoint_cadence_and_retention_keep_complete_pairs(tmp_path):
    train, orch = configs(tmp_path)
    train.save_steps = 2
    train.max_steps = 5
    orch.ckpt = GRPOCheckpointConfig({"keep_last": 1})
    coordinator = NativeCheckpointCoordinator(train, orch)
    assert [step for step in range(1, 6) if coordinator.should_save(step)] == [2, 4, 5]
    for step in (2, 4, 5):
        save_pair(coordinator, step)
    assert [p.name for p in coordinator.root.glob("step_*")] == ["step_5"]


def test_failed_buffer_save_never_publishes_a_resumable_checkpoint(tmp_path):
    train, orch = configs(tmp_path)
    coordinator = NativeCheckpointCoordinator(train, orch)
    coordinator.save_training(Trainer(), 1)
    buffer = Buffer()

    def fail(path):
        raise OSError("disk write failed")

    buffer.save = fail
    with pytest.raises(OSError, match="disk write failed"):
        coordinator.save_orchestrator(Progress(step=1), buffer, next_group_id=1, depth_state={})
    assert not list(coordinator.root.glob("step_*"))
