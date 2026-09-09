"""Native GLM kernels and a real training step on a 2.34M-parameter checkpoint."""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.sft.glm.create_dummy import checkpoint_weights, create_dummy

pytestmark = pytest.mark.gpu


def run_kernel(kind, inputs, expected, **options):
    from surogate import _surogate as ext

    outputs = [torch.empty_like(t).contiguous() for t in expected]
    stream = torch.cuda.current_stream().cuda_stream
    ext._glm5_kernel(kind, False, inputs, outputs, options, stream)
    for actual, target in zip(outputs, expected):
        torch.testing.assert_close(actual, target, atol=3e-5, rtol=3e-4)
    upstream = [torch.randn_like(t).contiguous() for t in expected]
    differentiable = [t for t in inputs if t.requires_grad]
    reference = torch.autograd.grad(expected, differentiable, grad_outputs=upstream)
    gradients = [torch.empty_like(t, dtype=torch.float32).contiguous() for t in differentiable]
    checkpoints = None
    if kind == 3:
        b, t, h, d = inputs[0].shape
        checkpoints = torch.empty((b, (t + 15) // 16, h, d, d), device="cuda")
    ext._glm5_kernel(kind, True, upstream + inputs, gradients, options, stream, checkpoints)
    for actual, target in zip(gradients, reference):
        torch.testing.assert_close(actual, target.float(), atol=2e-4, rtol=2e-3)


@pytest.mark.parametrize("streams,iters", [(2, 1), (4, 20)])
def test_mhc_mix_matches_transformers_and_autograd(streams, iters):
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperConnection

    torch.manual_seed(17)
    c = 32
    module = Glm5NextTextHyperConnection(
        Glm5NextTextConfig(hidden_size=c, hc_mult=streams, hc_sinkhorn_iters=iters)
    ).cuda()
    with torch.no_grad():
        module.fn.normal_(std=0.2)
        module.base.normal_(std=0.2)
        module.scale.copy_(torch.tensor([0.7, 1.2, 0.4], device="cuda"))
    x = torch.randn(2, 3, streams * c, device="cuda", requires_grad=True)
    post, comb, collapsed = module(x.view(2, 3, streams, c))
    run_kernel(
        0,
        [x, module.fn, module.base, module.scale],
        [collapsed, post.reshape(6, streams), comb.reshape(6, streams, streams)],
        streams=streams,
        sinkhorn_iters=iters,
    )


def test_mhc_combine_gradients():
    torch.manual_seed(19)
    b, t, h, c = 2, 3, 4, 32
    x = torch.randn(b, t, h * c, device="cuda", requires_grad=True)
    y = torch.randn(b, t, c, device="cuda", requires_grad=True)
    post = torch.randn(b * t, h, device="cuda", requires_grad=True)
    comb = torch.randn(b * t, h, h, device="cuda", requires_grad=True)
    result = post.view(b, t, h, 1) * y.unsqueeze(2) + torch.matmul(
        comb.view(b, t, h, h).transpose(-1, -2), x.view(b, t, h, c)
    )
    run_kernel(1, [x, y, post, comb], [result.reshape(b, t, h * c)])


def test_decay_gradients_include_per_channel_bias_and_per_head_rate():
    torch.manual_seed(23)
    x = torch.randn(2, 17, 2, 32, device="cuda", requires_grad=True)
    a = torch.randn(2, device="cuda", requires_grad=True)
    bias = torch.randn(64, device="cuda", requires_grad=True)
    result = -5 * torch.sigmoid(a.exp().view(1, 1, 2, 1) * (x + bias.view(1, 1, 2, 32)))
    run_kernel(2, [x, a, bias], [result])


@pytest.mark.parametrize("length,dim", [(1, 8), (17, 32), (65, 128)])
def test_kda_matches_transformers_across_checkpoint_boundaries(length, dim):
    from transformers.models.glm5_next.modeling_glm5_next import recurrent_kimi_delta_attention

    torch.manual_seed(29)
    q, k, v = [torch.randn(2, length, 2, dim, device="cuda", requires_grad=True) for _ in range(3)]
    g = (-5 * torch.rand_like(q)).requires_grad_()
    beta = torch.rand(2, length, 2, device="cuda", requires_grad=True)
    # Unwrap the optional external fused kernel to use Transformers' independent
    # PyTorch recurrence as the reference for both values and derivatives.
    result, _ = inspect.unwrap(recurrent_kimi_delta_attention)(
        q, k, v, g, beta, None, False, use_qk_l2norm_in_kernel=True
    )
    run_kernel(3, [q, k, v, g, beta], [result])


def test_packed_kda_resets_state_and_gradients():
    from transformers.models.glm5_next.modeling_glm5_next import recurrent_kimi_delta_attention

    torch.manual_seed(30)
    lengths = [3, 14, 1, 19]
    q, k, v = [torch.randn(1, sum(lengths), 2, 32, device="cuda", requires_grad=True) for _ in range(3)]
    g = (-5 * torch.rand_like(q)).requires_grad_()
    beta = torch.rand(1, sum(lengths), 2, device="cuda", requires_grad=True)
    pos = torch.cat([torch.arange(n, device="cuda", dtype=torch.int32) for n in lengths])[None]
    results, start = [], 0
    for length in lengths:
        args = [x[:, start : start + length] for x in (q, k, v, g, beta)]
        result, _ = inspect.unwrap(recurrent_kimi_delta_attention)(*args, None, False, use_qk_l2norm_in_kernel=True)
        results.append(result)
        start += length
    run_kernel(3, [q, k, v, g, beta, pos], [torch.cat(results, dim=1)])


def test_packed_convolution_resets_history_and_gradients():
    torch.manual_seed(33)
    lengths = [1, 3, 17]
    x = torch.randn(2, sum(lengths), 32, device="cuda", requires_grad=True)
    w = torch.randn(32, 1, 4, device="cuda", requires_grad=True)
    pos = torch.cat([torch.arange(n, device="cuda", dtype=torch.int32) for n in lengths])[None].repeat(2, 1)
    chunks = x.split(lengths, dim=1)
    expected = torch.cat(
        [
            torch.nn.functional.silu(
                torch.nn.functional.conv1d(c.transpose(1, 2), w, padding=3, groups=32)[:, :, : c.shape[1]]
            ).transpose(1, 2)
            for c in chunks
        ],
        dim=1,
    )
    run_kernel(5, [x, w, pos], [expected])


@pytest.mark.parametrize("packed", [False, True])
def test_clamp_backward_at_and_beyond_limits(packed):
    x = torch.tensor(
        [[-11.0, -10.0, 0.0, 10.0, 11.0, -11.0, -10.0, 0.0, 10.0, 11.0]], device="cuda", requires_grad=True
    )
    if packed:
        up, gate = x.chunk(2, dim=-1)
        target = torch.cat((up.clamp(-10, 10), gate.clamp(max=10)), dim=-1)
    else:
        target = x.clamp(-10, 10)
    run_kernel(4, [x], [target], min=-10.0, max=10.0, fused_gate_up=packed)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    from safetensors.torch import load_file, save_file

    path = tmp_path_factory.mktemp("glm5")
    create_dummy(path)
    weights = load_file(path / "model.safetensors")
    # Gradient parity needs a stable top-k branch: a one-ULP KDA difference
    # can otherwise exchange tied experts and change an entire token's MLP.
    # Nonzero, rotated correction biases also exercise frozen router buffers.
    for i, name in enumerate(sorted(n for n in weights if n.endswith("e_score_correction_bias"))):
        weights[name] = torch.linspace(-0.3, 0.3, 4).roll(i)
    save_file(weights, path / "model.safetensors", metadata={"format": "pt"})
    return path


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("index_topk", 16, "sequence_len <= index_topk"),
        ("n_group", 2, "n_group=topk_group=1"),
        ("norm_topk_prob", False, "norm_topk_prob=true"),
        ("index_kpool_always_select_tail", False, "index_kpool_always_select_tail=true"),
    ],
)
def test_unsupported_attention_and_routing_are_rejected(checkpoint, tmp_path, capfd, field, value, message):
    from surogate import _surogate as ext
    from surogate.dsl.ir_builder import build_dsl_ir_for_model

    config = json.loads((checkpoint / "config.json").read_text())
    config["text_config"][field] = value
    (tmp_path / "config.json").write_text(json.dumps(config))
    options = ext.RuntimeOptions(use_cuda_graphs=False, master_dtype="bf16")
    options.dsl_ir_json = build_dsl_ir_for_model(str(tmp_path))
    trainer = None
    with pytest.raises(RuntimeError) as error:
        trainer = ext.SurogateTrainer(
            ngpu=1,
            config=ext.PretrainedConfig.from_pretrained(str(tmp_path), "bf16"),
            options=options,
            batch_size=1,
            seq_len=32,
            grad_accum=1,
        )
        # Worker initialization is asynchronous; the first work item joins it.
        trainer.init_weights()
    del trainer
    assert message in str(error.value) + capfd.readouterr().err


@pytest.mark.parametrize(
    "lora,graphs,packed,grad_accum,doc_masking",
    [
        (False, False, False, 1, True),
        (True, False, False, 1, True),
        (False, True, True, 1, True),
        (True, True, True, 1, True),
        (False, True, True, 2, True),
        (True, True, True, 2, True),
        (False, False, False, 1, False),
    ],
    ids=["full", "lora", "full-packed", "lora-packed", "full-accum", "lora-accum", "reference-fallback"],
)
def test_dummy_checkpoint_forward_backward_and_update(
    checkpoint, tmp_path, monkeypatch, lora, graphs, packed, grad_accum, doc_masking
):
    from transformers import Glm5NextForConditionalGeneration
    from transformers.models.glm5_next import modeling_glm5_next as glm

    from surogate import _surogate as ext
    from surogate.dsl.ir_builder import build_dsl_ir_for_model

    # Transformers otherwise silently selects an installed FLA package. Keep
    # this reference independent of both the vendored and external kernels.
    monkeypatch.setattr(glm, "chunk_kimi_delta_attention", inspect.unwrap(glm.chunk_kimi_delta_attention))

    torch.manual_seed(31)
    options = ext.RuntimeOptions(
        recompute="true" if graphs else "false",
        use_cuda_graphs=graphs,
        master_dtype="bf16",
        offload_master=False,
        offload_grads=False,
        offload_optimizer=False,
        doc_masking=doc_masking,
    )
    options.dsl_ir_json = build_dsl_ir_for_model(str(checkpoint))
    from surogate.kernels.jit_compile import compile_jit_kernels

    options.jit_kernel_manifests = compile_jit_kernels(options.dsl_ir_json) if doc_masking else {}
    adapter = (
        ext.LoRAAdapterConfig(rank=8, alpha=16, dropout=0.0, dtype="bf16", target_modules=["all"]) if lora else None
    )
    trainer = ext.SurogateTrainer(
        ngpu=1,
        config=ext.PretrainedConfig.from_pretrained(str(checkpoint), "bf16"),
        options=options,
        batch_size=1,
        seq_len=32,
        grad_accum=grad_accum,
        lora_config=adapter,
    )
    trainer.import_weights(str(checkpoint / "model.safetensors"))
    ids = np.random.default_rng(31).integers(3, 259, size=(1, 32), dtype=np.int32)
    targets = np.roll(ids, -1, axis=1).copy()
    targets[:, -1] = -100
    lengths = [13, 19] if packed else [32]
    positions = np.concatenate([np.arange(n, dtype=np.int32) for n in lengths])[None]
    for end in np.cumsum(lengths):
        targets[:, end - 1] = -100
    reference = (
        Glm5NextForConditionalGeneration.from_pretrained(checkpoint, dtype=torch.bfloat16, attn_implementation="eager")
        .cuda()
        .eval()
    )
    with torch.enable_grad():
        chunks, start = [], 0
        for length in lengths:
            chunks.append(
                reference(
                    torch.from_numpy(ids[:, start : start + length]).long().cuda(), use_cache=False
                ).logits.float()
            )
            start += length
        logits = torch.cat(chunks, dim=1)
        expected = (
            logits.log_softmax(-1).gather(-1, torch.from_numpy(targets.clip(0)).long().cuda().unsqueeze(-1)).squeeze(-1)
        )
    before = trainer.compute_logprobs(ids, targets, position_ids=positions).copy()
    mask = targets != -100
    np.testing.assert_allclose(before[mask], expected.detach().cpu().numpy()[mask], atol=0.035, rtol=0)
    trainer.step_with_custom_loss(ids, targets, mask.astype(np.float32) / mask.sum(), position_ids=positions)
    gradients = trainer.get_lora_gradients(0) if lora else trainer.get_gradients(0)
    gradients = {name: torch.from_dlpack(t).clone() for name, t in gradients.items()}
    assert gradients
    assert all(torch.isfinite(t).all() for t in gradients.values())
    # The native seed multiplies per-token negative log likelihood.
    (-expected[torch.from_numpy(mask).cuda()].mean()).backward()
    grad_state = {name: p.grad for name, p in reference.named_parameters() if p.grad is not None}
    hf_grads = dict(checkpoint_weights(SimpleNamespace(state_dict=lambda: grad_state)))
    if lora:
        weights = {name: torch.from_dlpack(t) for name, t in trainer.get_lora_weights(0).items()}
        errors = []
        for name, value in gradients.items():
            if ".lora_A." in name:
                # B starts at zero, so dA must also be zero on the first step.
                assert torch.count_nonzero(value) == 0, name
                continue
            source = (
                name.removeprefix("base_model.model.")
                .replace("model.layers.", "model.language_model.layers.")
                .replace(".lora_B.weight", ".weight")
            )
            layer = int(source.split(".layers.")[1].split(".")[0])
            if layer % 4 == 3 and ".self_attn." in source:
                source = source.replace(".q_proj.", ".q_b_proj.").replace(".k_proj.", ".kv_b_proj.")
            if ".experts." in source:
                dw = torch.stack([hf_grads[source.replace(".experts.", f".experts.{i}.")] for i in range(4)])
            else:
                dw = hf_grads[source]
            a = weights[name.replace(".lora_B.", ".lora_A.")]
            # At B=0, dB = (alpha/rank) * dW @ A^T, independently of the
            # native adapter implementation. Account for BF16 GEMM rounding.
            target = 2 * (dw.float() @ a.float().transpose(-1, -2))
            error = (value.float() - target).square().mean().sqrt()
            scale = target.square().mean().sqrt()
            if error >= 0.1 * scale + 2e-7:
                errors.append((name, float(error), float(scale)))
        assert not errors, json.dumps(errors, indent=2)
    if not lora:
        assert not any(name.endswith("e_score_correction_bias") for name in gradients)
        mapping = json.loads(options.dsl_ir_json)["modules"][0]["hf_mapping"]
        errors = []
        for name, value in gradients.items():
            source = mapping[name]
            if isinstance(source, str):
                target = hf_grads[source]
            elif source["type"] == "fuse":
                target = torch.cat([hf_grads[s] for s in source["sources"]])
            else:
                pattern = source["pattern"]
                experts = []
                for i in range(4):
                    key = pattern.format(expert=i)
                    if source.get("fuse_gate_up"):
                        experts.append(torch.cat((hf_grads[key.replace("gate_proj", "up_proj")], hf_grads[key])))
                    else:
                        experts.append(hf_grads[key])
                target = torch.stack(experts)
            actual = value.float()
            target = target.float()
            assert actual.shape == target.shape, name
            error = (actual - target).square().mean().sqrt()
            scale = target.square().mean().sqrt()
            # Small HC and per-head KDA decay gradients sum cancelling terms;
            # BF16 boundaries amplify their relative error near zero. The
            # standalone FP32 kernel test checks these derivatives tightly.
            atol = (
                5e-5
                if name.endswith((".hc_attn_scale", ".hc_ffn_scale", ".hc_attn_base", ".hc_ffn_base", ".kda_A_log"))
                else 2e-7
            )
            if error >= 0.1 * scale + atol:
                errors.append(
                    (name, float(error), float(scale), float((actual * target).sum() / target.square().sum()))
                )
        assert not errors, json.dumps(errors, indent=2)
    del reference
    if grad_accum == 2:
        # A repeated microbatch must add to every gradient, including the
        # expert weights and FP32 gates, before the optimizer consumes it.
        trainer.step_with_custom_loss(ids, targets, mask.astype(np.float32) / mask.sum(), position_ids=positions)
        accumulated = trainer.get_lora_gradients(0) if lora else trainer.get_gradients(0)
        accumulation_errors = []
        for name, value in accumulated.items():
            actual = torch.from_dlpack(value).float()
            target = 2 * gradients[name].float()
            error = (actual - target).square().mean().sqrt()
            scale = target.square().mean().sqrt()
            # HC controls reduce cancelling contributions through BF16
            # activations; atomic reduction order can move near-zero scalars.
            # Keep the matrix/decay accumulation check at the tighter bound.
            atol = 1e-5 if name.endswith((".hc_attn_scale", ".hc_ffn_scale", ".hc_attn_base", ".hc_ffn_base")) else 2e-7
            if error >= 0.01 * scale + atol:
                accumulation_errors.append((name, float(error), float(scale)))
        assert not accumulation_errors, json.dumps(accumulation_errors, indent=2)
    update = trainer.update_with_config(ext.OptimizerConfig(learning_rate=1e-3), 1)
    assert np.isfinite(update["norm"]) and update["norm"] > 0
    after = trainer.compute_logprobs(ids, targets, position_ids=positions).copy()
    assert np.max(np.abs(after - before)) > 1e-4
    if lora:
        # GRPO's reference policy must remain the frozen base after updates.
        base = trainer.compute_logprobs(ids, targets, use_lora=False, position_ids=positions)
        np.testing.assert_allclose(base, before, atol=1e-5, rtol=0)
        adapter_path = tmp_path / "adapter"
        trainer.export_adapter(str(adapter_path), str(checkpoint))
        trainer.import_adapter(str(adapter_path / "adapter_model.safetensors"))
        np.testing.assert_allclose(
            trainer.compute_logprobs(ids, targets, position_ids=positions), after, atol=1e-5, rtol=0
        )
    else:
        model_path = tmp_path / "trained"
        trainer.export_model(str(model_path))
        trainer.import_weights(str(model_path / "model.safetensors"))
        np.testing.assert_allclose(
            trainer.compute_logprobs(ids, targets, position_ids=positions), after, atol=1e-5, rtol=0
        )
