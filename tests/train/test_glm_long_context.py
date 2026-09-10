"""Tiled dense/routed/shared MLPs retain clamps, adapters and accumulated gradients."""

import gc
import json

import numpy as np
import pytest
import torch

from examples.sft.glm.create_dummy import create_dummy

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(scope="module", params=["glm", "qwen3"])
def checkpoint(tmp_path_factory, request):
    from safetensors.torch import load_file, save_file

    root = tmp_path_factory.mktemp(f"{request.param}-long-context")
    if request.param == "qwen3":
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(83)
        model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=320,
                hidden_size=128,
                intermediate_size=1024,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=64,
                max_position_embeddings=512,
            )
        ).bfloat16()
        model.save_pretrained(root)
        return root
    create_dummy(root, index_topk=8, max_sequence_length=512)
    config = json.loads((root / "config.json").read_text())
    # Exercise the clamp even with random, small initial projections.
    config["text_config"]["swiglu_limit"] = 0.03
    (root / "config.json").write_text(json.dumps(config))
    weights = load_file(root / "model.safetensors")
    # Stabilize discrete expert selection when tiled GEMMs round differently.
    for i, name in enumerate(sorted(n for n in weights if n.endswith("e_score_correction_bias"))):
        weights[name] = torch.linspace(-0.3, 0.3, 4).roll(i)
    save_file(weights, root / "model.safetensors", metadata={"format": "pt"})
    return root


def run_training(checkpoint, tiled, lora, recompute, graphs):
    from surogate import _surogate as ext
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.kernels.jit_compile import compile_jit_kernels

    options = ext.RuntimeOptions(
        recompute="true" if recompute else "false",
        use_cuda_graphs=graphs,
        master_dtype="bf16",
        offload_master=False,
        offload_grads=False,
        offload_optimizer=False,
        long_context=tiled,
        doc_masking=True,
    )
    options.glm_rollout_parity = json.loads((checkpoint / "config.json").read_text())["model_type"] == "glm5_next"
    options.dsl_ir_json = build_dsl_ir_for_model(str(checkpoint))
    options.jit_kernel_manifests = compile_jit_kernels(options.dsl_ir_json)
    adapter = ext.LoRAAdapterConfig(rank=8, alpha=13, dropout=0, dtype="bf16", target_modules=["all"]) if lora else None
    trainer = ext.SurogateTrainer(
        ngpu=1,
        config=ext.PretrainedConfig.from_pretrained(str(checkpoint), "bf16"),
        options=options,
        batch_size=2,
        seq_len=304,
        grad_accum=2,
        lora_config=adapter,
    )
    trainer.import_weights(str(checkpoint / "model.safetensors"))
    if lora:
        # Nonzero B detects omitted adapter contributions in forward/recompute.
        for name, value in trainer.get_lora_weights(0).items():
            if ".lora_B." in name:
                torch.from_dlpack(value).fill_(0.003)
        torch.cuda.synchronize()
    # Both original tokens (608) and routed rows (1216) leave a partial tile.
    ids = np.random.default_rng(83).integers(3, 259, (2, 304), dtype=np.int32)
    targets = np.roll(ids, -1, axis=1).copy()
    positions = np.stack([np.arange(304), np.concatenate([np.arange(n) for n in (3, 127, 174)])]).astype(np.int32)
    targets[:, -1] = -100
    targets[1, [2, 129]] = -100
    targets[:, 20:40] = -100  # Prompt/tool-result positions receive no loss.
    scales = (targets != -100).astype(np.float32)
    scales /= scales.sum()
    before = trainer.compute_logprobs(ids, targets, position_ids=positions).copy()
    trainer.step_with_custom_loss(ids, targets, scales, position_ids=positions)
    first = trainer.get_lora_gradients(0) if lora else trainer.get_gradients(0)
    first = {name: torch.from_dlpack(t).float().cpu() for name, t in first.items()}
    trainer.step_with_custom_loss(ids, targets, scales, position_ids=positions)
    gradients = trainer.get_lora_gradients(0) if lora else trainer.get_gradients(0)
    gradients = {name: torch.from_dlpack(t).float().cpu() for name, t in gradients.items()}
    for name, grad in gradients.items():
        if (
            name.endswith((".mlp_up_weight", ".mlp_down_weight"))
            or ".mlp." in name
            or "experts_" in name
            or "shared_expert_" in name
        ):
            target = 2 * first[name]
            error = (grad - target).square().mean().sqrt()
            assert error <= 0.01 * target.square().mean().sqrt() + 2e-7, (name, float(error))
    layout = trainer.get_debug_tensor_layout()
    arenas = trainer.get_debug_arena_summary()
    result = trainer.update_with_config(ext.OptimizerConfig(learning_rate=1e-3), 1)
    assert np.isfinite(result["norm"]) and result["norm"] > 0
    after = trainer.compute_logprobs(ids, targets, position_ids=positions).copy()
    assert np.max(np.abs(after - before)) > 1e-5
    assert np.isfinite(after).all()
    # Compare forward execution with identical updated weights: BF16 gradient
    # reduction order can move hard DSA top-k across a selection boundary.
    updated = checkpoint / f"updated-{lora}-{recompute}-{graphs}"
    if not tiled:
        if lora:
            trainer.export_adapter(str(updated), str(checkpoint))
        else:
            trainer.export_model(str(updated))
    else:
        if lora:
            trainer.import_adapter(str(updated / "adapter_model.safetensors"))
        else:
            trainer.import_weights(str(updated / "model.safetensors"))
        after = trainer.compute_logprobs(ids, targets, position_ids=positions).copy()
    del trainer
    gc.collect()
    return before, gradients, after, layout, arenas


@pytest.mark.parametrize(
    "lora,recompute,graphs", [(False, False, False), (False, True, True), (True, False, True), (True, True, False)]
)
def test_tiled_mlp_matches_full_execution(checkpoint, lora, recompute, graphs):
    expected = run_training(checkpoint, False, lora, recompute, graphs)
    actual = run_training(checkpoint, True, lora, recompute, graphs)
    glm = json.loads((checkpoint / "config.json").read_text())["model_type"] == "glm5_next"
    # GLM rollout parity fixes GEMM reduction order; ordinary BF16 cuBLAS may
    # choose a different reduction when the token dimension changes.
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-5 if glm else 0.01, rtol=0)
    assert actual[1].keys() == expected[1].keys()
    errors = []
    for name, grad in actual[1].items():
        target = expected[1][name]
        rms = (grad - target).square().mean().sqrt()
        # Per-head decay/HC gradients sum cancelling terms through BF16
        # boundaries. Use the absolute allowance from the HF gradient check;
        # matrix gradients retain the tighter bound.
        scalar_reduction = name.endswith(
            (".kda_A_log", ".hc_attn_scale", ".hc_ffn_scale", ".hc_attn_base", ".hc_ffn_base")
        )
        atol, rtol = (5e-5, 0.1) if scalar_reduction else (2e-6, 0.03)
        if not torch.isfinite(grad).all() or rms > rtol * target.square().mean().sqrt() + atol:
            errors.append((name, float(rms), float(target.square().mean().sqrt())))
    assert not errors, errors
    np.testing.assert_allclose(actual[2], expected[2], atol=1e-5 if glm else 0.01, rtol=0)
    intermediates = [
        entry
        for entry in actual[3]
        if entry["graph"] == "forward" and ".mlp_up" in entry["name"] and "weight" not in entry["name"]
    ]
    assert intermediates and all(entry["bytes"] == 0 for entry in intermediates), intermediates
    if checkpoint.name.startswith("glm-"):
        experts = [
            entry
            for entry in actual[3]
            if entry["graph"] == "forward"
            and entry["name"].endswith(
                (
                    ".expert_gate_up",
                    ".expert_act",
                    ".shared_expert_gate_out",
                    ".shared_expert_up_out",
                    ".shared_expert_gate_act",
                )
            )
        ]
        assert experts and all(entry["bytes"] == 0 for entry in experts), experts
    assert actual[4]["arena_fwd_stack_bytes"] <= expected[4]["arena_fwd_stack_bytes"]
    if not recompute:
        assert actual[4]["arena_save_for_bwd_bytes"] < expected[4]["arena_save_for_bwd_bytes"]


def test_tiling_rejects_nonzero_adapter_dropout(checkpoint, capfd):
    from surogate import _surogate as ext
    from surogate.dsl.ir_builder import build_dsl_ir_for_model

    options = ext.RuntimeOptions(long_context=True, master_dtype="bf16", use_cuda_graphs=False)
    options.dsl_ir_json = build_dsl_ir_for_model(str(checkpoint))
    adapter = ext.LoRAAdapterConfig(rank=8, alpha=16, dropout=0.1, dtype="bf16", target_modules=["all"])
    trainer = None
    with pytest.raises(RuntimeError) as error:
        trainer = ext.SurogateTrainer(
            ngpu=1,
            config=ext.PretrainedConfig.from_pretrained(str(checkpoint), "bf16"),
            options=options,
            batch_size=1,
            seq_len=128,
            grad_accum=1,
            lora_config=adapter,
        )
        trainer.init_weights()  # Join asynchronous worker initialization.
    del trainer
    assert "requires lora_dropout: 0" in str(error.value) + capfd.readouterr().err
