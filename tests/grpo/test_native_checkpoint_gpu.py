"""Native checkpoint continuation preserves adapter and AdamW history."""

import gc
import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from tests.grpo.test_batched_decode_gpu import make_trainer

pytestmark = [pytest.mark.gpu, pytest.mark.slow,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.mark.parametrize("case", ["llama", "qwen3_5_moe"])
@pytest.mark.parametrize("optimizer", ["adamw", "adamw_8bit"])
def test_adamw_resume_matches_uninterrupted_updates(tmp_path, case, optimizer):
    from surogate import _surogate as ext

    model = tmp_path / "model"
    model.mkdir()
    trainer = make_trainer(model, case, sequence=128)
    rng = np.random.default_rng(11)
    inputs = rng.integers(3, 97, size=(2, 128), dtype=np.int32)
    targets = np.roll(inputs, -1, axis=1)
    gradients = rng.normal(0, 0.01, size=inputs.shape).astype(np.float32)
    options = ext.OptimizerConfig(optimizer=optimizer, learning_rate=1e-3)

    def update(owner, step):
        owner.step_with_custom_loss(inputs, targets, gradients)
        result = owner.update_with_config(options, step)
        assert np.isfinite(result["norm"]) and result["norm"] > 0

    update(trainer, 1)
    trainer.save_checkpoint(str(tmp_path / "saved"), 0)
    checkpoint = tmp_path / "saved/step_00000000"
    assert json.loads((checkpoint / "lora_optimizer.json").read_text())["optimizer_type"] == optimizer
    moments = load_file(str(checkpoint / "lora_optimizer.safetensors"))
    assert all(t.abs().max() > 0 for t in moments.values())
    update(trainer, 2)
    update(trainer, 3)
    expected = {name: torch.from_dlpack(t).clone().cpu() for name, t in trainer.get_lora_weights(0).items()}
    trainer.save_checkpoint(str(tmp_path / "expected"), 2)
    del trainer
    gc.collect()
    resumed = make_trainer(model, case, sequence=128)
    resumed.load_checkpoint(str(tmp_path / "saved"), 0)
    resumed.save_checkpoint(str(tmp_path / "restored"), 0)
    for filename in ("adapter_model.safetensors", "lora_optimizer.safetensors"):
        saved = load_file(str(checkpoint / filename))
        restored = load_file(str(tmp_path / "restored/step_00000000" / filename))
        assert saved.keys() == restored.keys()
        assert all(torch.equal(saved[name], restored[name]) for name in saved)
    update(resumed, 2)
    update(resumed, 3)
    for name, tensor in resumed.get_lora_weights(0).items():
        torch.testing.assert_close(torch.from_dlpack(tensor).cpu(), expected[name], atol=2e-5, rtol=0.005)
    resumed.save_checkpoint(str(tmp_path / "resumed"), 2)
    reference = load_file(str(tmp_path / "expected/step_00000002/lora_optimizer.safetensors"))
    actual = load_file(str(tmp_path / "resumed/step_00000002/lora_optimizer.safetensors"))
    for name in reference:
        torch.testing.assert_close(actual[name], reference[name], atol=2e-7, rtol=0.005)
    del resumed
    gc.collect()
