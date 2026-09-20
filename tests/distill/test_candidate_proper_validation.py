"""Asymmetric reduction and full4B FP8 LoRA validation; no saved adapter/model."""

import gc
import json
import os
from pathlib import Path

import numpy as np
import pytest

from surogate import _surogate as sg
from surogate.dsl.ir_builder import build_dsl_ir_for_model
from surogate.utils.hf import get_model_weights_path
from tests.distill import test_kd_gradient as helper

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def record(name, value):
    """Optional compact scalar evidence; never write training/evaluation JSONL."""
    if target := os.environ.get("CANDIDATE_VALIDATION_EVIDENCE"):
        path = Path(target)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    print(name, json.dumps(value), flush=True)


def make_trainer(model, *, ngpu=1, batch=1, accum=2, rank=16):
    cfg = sg.PretrainedConfig.from_pretrained(str(model), "bf16")
    opts = sg.RuntimeOptions(recipe="fp8_hybrid", use_cuda_graphs=False, offload_master=False,
                             offload_grads=False, offload_optimizer=False,
                             offload_residual=False, shard_gradients=True)
    opts.dsl_ir_json = build_dsl_ir_for_model(str(model))
    from surogate.kernels.jit_compile import compile_jit_kernels
    opts.jit_kernel_manifests = compile_jit_kernels(opts.dsl_ir_json)
    trainer = sg.SurogateTrainer(
        ngpu=ngpu, config=cfg, options=opts, batch_size=batch, seq_len=64, grad_accum=accum,
        memcpy_all_gather=True, memcpy_send_recv=True,
        lora_config=sg.LoRAAdapterConfig(rank=rank, alpha=rank, dropout=0, dtype="fp32", target_modules=["all"]),
        qlora_config=None,
    )
    trainer.import_weights(get_model_weights_path(str(model)))
    return trainer


def raw_gradients(trainer, rank):
    return {name: helper.torch.from_dlpack(value).double().cpu().clone()
            for name, value in trainer.get_lora_gradients(rank).items()}


def norm(grads):
    return np.sqrt(sum(value.square().sum().item() for value in grads.values()))


def batch(rows):
    inputs = np.tile(np.arange(64, dtype=np.int32)[None, :] + 100, (rows, 1))
    targets = np.full_like(inputs, -100)
    ids = np.full((rows, 64, 5), -1, np.int32)
    return inputs, targets, ids, np.zeros_like(ids, dtype=np.float32)


def kwargs(objective):
    result = dict(top_k=5, temperature=1., kd_weight=1., ce_weight=0., candidate_only=True)
    if objective != "cross_entropy":
        result["candidate_objective"] = objective
    return result


@pytest.mark.parametrize("objective", ["cross_entropy", "brier", "rps"])
def test_asymmetric_two_gpu_global_decision_mean(tmp_path, monkeypatch, objective):
    if helper.torch.cuda.device_count() < 2:
        pytest.skip("Requires two visible GPUs")
    snapshot = helper.resolve_model_path()
    if snapshot is None:
        pytest.skip("Needs cached Qwen3")
    monkeypatch.setattr(helper, "MINI_MODEL_DIR", tmp_path / "model")
    model = helper.prepare_mini_model(snapshot)
    trainer = make_trainer(model, ngpu=2, batch=2)
    try:
        inputs, targets, ids, unused = batch(4)
        # Every forward input is identical on every rank. Rank0 has1 decision,
        # rank1 has2; the final accumulation microstep has no valid targets.
        for row in [0, 2, 3]:
            targets[row, 20] = 101
            ids[row, 20] = [102, -1, 100, -1, 101]
        trainer.step_with_kd(inputs, targets, ids, unused, **kwargs(objective))
        # Before the final collective, ValidTokenCount is still local. The
        # accessor nevertheless divides it by world_size, so undo that factor.
        single_decision_loss = trainer.get_kd_loss() / 2
        before = [raw_gradients(trainer, rank) for rank in range(2)]
        assert before[0].keys() == before[1].keys()
        global_sum = {name: before[0][name] + before[1][name] for name in before[0]}
        expected_mean_norm = norm(global_sum) / 3
        trainer.step_with_kd(inputs, np.full_like(targets, -100), ids, unused, **kwargs(objective))
        reduced = [raw_gradients(trainer, rank) for rank in range(2)]
        rank_disagreement = max((reduced[0][name] - reduced[1][name]).abs().max().item() for name in reduced[0])
        weight_views = {name: helper.torch.from_dlpack(value)
                        for name, value in trainer.get_lora_weights(0).items()}
        weight_dtypes = {name: value.dtype for name, value in weight_views.items()}
        weights_before = {name: value.double().cpu().clone() for name, value in weight_views.items()}
        del weight_views
        # A deliberately large epsilon makes the first AdamW update sensitive
        # to gradient magnitude; this tests the scale consumed by the optimizer,
        # not just its reported norm. No clipping or weight decay in this check.
        rate, epsilon = 1e-3, 0.1
        optimizer = sg.OptimizerConfig(optimizer="adamw", learning_rate=rate, weight_decay=0.,
                                       grad_clip=0., adamw_beta1=0.9, adamw_beta2=0.999,
                                       adamw_epsilon=epsilon)
        result = trainer.update_with_config(optimizer, 0)
        weights_after = {name: helper.torch.from_dlpack(value).double().cpu().clone()
                         for name, value in trainer.get_lora_weights(0).items()}
        assert weights_before.keys() == weights_after.keys() == global_sum.keys()
        update_error = 0.
        updates_match = True
        quantized_update_error = 0.
        for name in global_sum:
            gradient = global_sum[name] / 3
            expected_weight = weights_before[name] - rate * gradient / (gradient.abs() + epsilon)
            update_error = max(update_error, (weights_after[name] - expected_weight).abs().max().item())
            # get_lora_weights exposes BF16 working weights, not FP32 masters.
            # Compare the actual working update within its dtype's rounding
            # envelope; an erroneous world-size factor is far outside this.
            tolerance = 0.004 if weight_dtypes[name] == helper.torch.bfloat16 else 2e-5
            updates_match = updates_match and helper.torch.allclose(
                weights_after[name], expected_weight, rtol=tolerance, atol=1e-8)
            expected_work = expected_weight.to(weight_dtypes[name]).double()
            quantized_update_error = max(quantized_update_error,
                                         (weights_after[name] - expected_work).abs().max().item())
        evidence = {"objective": objective, "recipe": "fp8_hybrid", "rank_valid_counts": [1, 2],
                    "global_valid_decisions": 3, "masked_final_microstep": True,
                    "raw_sum_norm": norm(global_sum), "expected_global_mean_norm": expected_mean_norm,
                    "actual_optimizer_norm": result["norm"], "norm_ratio": result["norm"] / expected_mean_norm,
                    "reduced_rank0_norm": norm(reduced[0]), "rank_gradient_disagreement": rank_disagreement,
                    "expected_forward_loss": single_decision_loss, "actual_forward_loss": result["loss"],
                    "adamw_epsilon": epsilon, "adamw_learning_rate": rate, "grad_clip": 0.,
                    "optimizer_update_max_abs_error": update_error,
                    "optimizer_work_dtypes": sorted({str(x) for x in weight_dtypes.values()}),
                    "optimizer_updates_match_dtype_rounding": updates_match,
                    "optimizer_quantized_update_max_abs_error": quantized_update_error}
        record(f"asymmetric-{objective}", evidence)
        assert rank_disagreement < 1e-6
        assert updates_match
        assert result["loss"] == pytest.approx(single_decision_loss, rel=2e-4, abs=2e-5)
        assert result["norm"] == pytest.approx(expected_mean_norm, rel=3e-4, abs=1e-6)
    finally:
        del trainer
        gc.collect()


@pytest.mark.parametrize("objective", ["brier", "rps"])
def test_full_qwen35_4b_tiny_fp8_lora_updates(objective):
    model = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    if not (model / "config.json").is_file():
        pytest.skip("Requires the pinned original Qwen3.5-4B cache")
    trainer = make_trainer(model, rank=32)
    evidence = {"objective": objective, "model": str(model), "recipe": "fp8_hybrid", "lora_rank": 32, "lora_alpha": 32, "updates": []}
    try:
        inputs, targets, ids, unused = batch(1)
        targets[0, 20] = 101
        ids[0, 20] = [102, -1, 100, -1, 101]
        for update in range(2):
            losses = []
            for micro in range(2):
                trainer.step_with_kd(inputs, targets, ids, unused, **kwargs(objective))
                losses.append(trainer.get_kd_loss() * (micro + 1))
            gradients = raw_gradients(trainer, 0)
            expected_norm = norm(gradients) / 2
            result = trainer.update_with_config(helper.opt_config(1e-4), update)
            evidence["updates"].append({"step": update + 1, "loss": result["loss"],
                                         "expected_loss": float(np.mean(losses)),
                                         "norm": result["norm"], "expected_norm": expected_norm,
                                         "adapter_tensors": len(gradients),
                                         "all_gradients_finite": all(helper.torch.isfinite(v).all().item() for v in gradients.values())})
            record(f"full4b-{objective}", evidence)
            assert evidence["updates"][-1]["all_gradients_finite"]
            assert np.isfinite(result["loss"]) and np.isfinite(result["norm"]) and result["norm"] > 0
            assert 0 <= result["loss"] <= (2 if objective == "brier" else 1)
            assert result["loss"] == pytest.approx(np.mean(losses), abs=2e-5)
            assert result["norm"] == pytest.approx(expected_norm, rel=3e-4, abs=1e-6)
    finally:
        del trainer
        gc.collect()
