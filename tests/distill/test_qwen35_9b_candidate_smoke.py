"""Two-update exact-checkpoint 9B compatibility smoke; no model export."""

import gc
from pathlib import Path

import numpy as np
import pytest

from surogate.dsl.ir_builder import build_dsl_ir_for_model
from surogate.kernels.jit_compile import compile_jit_kernels
from surogate.utils.hf import get_model_weights_path
from tests.distill import test_candidate_proper_validation as helper
from tests.distill.test_kd_gradient import opt_config, torch

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

MODEL = Path.home() / (
    ".cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/"
    "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)


def make_offloaded_trainer():
    """Explicit CPU base-master streaming; all computation/optimizer stays GPU."""
    sg = helper.sg
    cfg = sg.PretrainedConfig.from_pretrained(str(MODEL), "bf16")
    opts = sg.RuntimeOptions(recipe="fp8_hybrid", use_cuda_graphs=False,
                             offload_master=True, offload_grads=False,
                             offload_optimizer=False, offload_residual=False,
                             shard_gradients=True, cpu_training=False)
    opts.dsl_ir_json = build_dsl_ir_for_model(str(MODEL))
    opts.jit_kernel_manifests = compile_jit_kernels(opts.dsl_ir_json)
    trainer = sg.SurogateTrainer(
        ngpu=2, config=cfg, options=opts, batch_size=1, seq_len=64, grad_accum=2,
        memcpy_all_gather=True, memcpy_send_recv=True,
        lora_config=sg.LoRAAdapterConfig(rank=32, alpha=32, dropout=0, dtype="fp32", target_modules=["all"]),
        qlora_config=None,
    )
    trainer.import_weights(get_model_weights_path(str(MODEL)))
    return trainer


@pytest.mark.parametrize("objective", ["cross_entropy", "brier"])
def test_qwen35_9b_two_gpu_fp8_lora_compatibility(objective):
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two visible GPUs")
    if not (MODEL / "config.json").is_file():
        pytest.skip("Requires exact cached Qwen3.5-9B revision")
    trainer = make_offloaded_trainer()
    evidence = {
        "objective": objective,
        "model": str(MODEL),
        "recipe": "fp8_hybrid",
        "runtime_options": {"offload_master": True, "offload_grads": False,
                            "offload_optimizer": False, "offload_residual": False,
                            "cpu_training": False, "use_cuda_graphs": False,
                            "shard_gradients": True},
        "ngpu": 2,
        "batch_per_rank": 1,
        "sequence_length": 64,
        "gradient_accumulation_steps": 2,
        "lora_rank": 32,
        "lora_alpha": 32,
        "lora_dtype": "fp32",
        "optimizer": {
            "type": "adamw", "learning_rate": 1e-4, "weight_decay": 0,
            "grad_clip": 1, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8,
        },
        "input": {"tokens": "arange(64)+100 on both ranks", "target_position": 20,
                  "gold_token": 101, "candidate_slots": [102, -1, 100, -1, 101]},
        "candidate_loss": {"temperature": 1, "kd_weight": 1, "ce_weight": 0,
                           "candidate_only": True, "sidecar_logprobs_ignored": True},
        "global_valid_decisions_per_update": 4,
        "saved_adapter": False,
        "updates": [],
    }
    try:
        inputs, targets, ids, unused = helper.batch(2)
        targets[:, 20] = 101
        ids[:, 20] = [102, -1, 100, -1, 101]
        for update in range(2):
            for _ in range(2):
                trainer.step_with_kd(inputs, targets, ids, unused, **helper.kwargs(objective))
            gradients = helper.raw_gradients(trainer, 0)
            finite = all(torch.isfinite(value).all().item() for value in gradients.values())
            # Gradients are already averaged across two ranks. Correct global
            # normalization multiplies by world_size / global_valid_count.
            expected_norm = helper.norm(gradients) * 2 / 4
            tensors = len(gradients)
            params = sum(value.numel() for value in gradients.values())
            del gradients
            result = trainer.update_with_config(opt_config(1e-4), update)
            finite_weights = all(
                torch.isfinite(torch.from_dlpack(value)).all().item()
                for value in trainer.get_lora_weights(0).values()
            )
            evidence["updates"].append({
                "step": update + 1, "loss": result["loss"], "norm": result["norm"],
                "expected_global_mean_norm": expected_norm, "adapter_tensors": tensors,
                "adapter_parameters": params, "all_gradients_finite": finite,
                "all_adapter_working_weights_finite": finite_weights,
            })
            helper.record(f"qwen35-9b-{objective}", evidence)
            assert finite and finite_weights
            assert np.isfinite(result["loss"]) and np.isfinite(result["norm"])
            assert result["loss"] >= 0 and result["norm"] > 0
            if objective == "brier":
                assert result["loss"] <= 2
            assert result["norm"] == pytest.approx(expected_norm, rel=3e-4, abs=1e-6)
            assert tensors == 256 and params > 42_467_328
    finally:
        del trainer
        gc.collect()
