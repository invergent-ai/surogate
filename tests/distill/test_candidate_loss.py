"""End-to-end candidate-only LoRA objective against the same engine's logits."""

import gc

import numpy as np
import pytest

from surogate import _surogate as sg
from surogate.dsl.ir_builder import build_dsl_ir_for_model
from surogate.utils.hf import get_model_weights_path
from tests.distill import test_kd_gradient as helper

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_candidate_lora_loss_accumulation_and_invalid_sets(tmp_path, monkeypatch):
    assert "candidate_only" in sg.SurogateTrainer.step_with_kd.__doc__, "Rebuild the native module"
    snapshot = helper.resolve_model_path()
    if snapshot is None:
        pytest.skip("Needs cached Qwen3")
    monkeypatch.setattr(helper, "MINI_MODEL_DIR", tmp_path / "model")
    model = helper.prepare_mini_model(snapshot)
    cfg = sg.PretrainedConfig.from_pretrained(str(model), "bf16")
    opts = sg.RuntimeOptions(
        use_cuda_graphs=False,
        offload_master=False,
        offload_grads=False,
        offload_optimizer=False,
        offload_residual=False,
        shard_gradients=True,
    )
    opts.dsl_ir_json = build_dsl_ir_for_model(str(model))
    trainer = sg.SurogateTrainer(
        ngpu=1,
        config=cfg,
        options=opts,
        batch_size=1,
        seq_len=64,
        grad_accum=2,
        memcpy_all_gather=True,
        memcpy_send_recv=True,
        lora_config=sg.LoRAAdapterConfig(rank=4, alpha=8, dropout=0, dtype="fp32", target_modules=["all"]),
        qlora_config=None,
    )
    trainer.import_weights(get_model_weights_path(str(model)))
    inputs = np.arange(64, dtype=np.int32)[None, :] + 100
    targets = np.full_like(inputs, -100)
    targets[0, 20] = 101
    ids = np.full((1, 64, 4), -1, np.int32)
    ids[0, 20, :3] = [100, 101, 102]
    unused = np.zeros_like(ids, dtype=np.float32)
    kwargs = dict(top_k=4, temperature=1.0, kd_weight=1.0, ce_weight=0.0, candidate_only=True)
    for bad in [[100, 100, 101, -1], [100, 102, -1, -1], [101, -1, -1, -1], [100, 101, -2, -1]]:
        invalid = ids.copy()
        invalid[0, 20] = bad
        with pytest.raises(ValueError, match="candidate_only"):
            trainer.step_with_kd(inputs, targets, invalid, unused, **kwargs)
    logits = np.asarray(trainer.next_token_logits(inputs, np.array([20], np.int32)))[0]
    allowed = logits[[100, 101, 102]].astype(np.float64)
    expected = np.logaddexp.reduce(allowed) - allowed[1]
    for _ in range(2):
        trainer.step_with_kd(inputs, targets, ids, unused, **kwargs)
    result = trainer.update_with_config(helper.opt_config(1e-4), 1)
    metric = trainer.get_kd_loss()
    assert np.isfinite(result["norm"]) and result["norm"] > 0
    assert metric == pytest.approx(expected, abs=0.03)
    assert result["loss"] == pytest.approx(metric, abs=0.005)
    del trainer
    gc.collect()
