"""Exercise Spark's complete backward graph against independent PyTorch autograd.

Uses small synthetic safetensors, so the test needs a GPU but no downloads.
The sequence crosses the local attention window and the hidden width differs
from the query width, as it does in the published 4B checkpoint.
"""

import json

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def checkpoint(tmp_path, torch):
    from safetensors.torch import save_file

    cfg = dict(
        architectures=["Spark2_5ForCausalLM"],
        model_type="spark2_5",
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=256,
        vocab_size=256,
        num_hidden_layers=4,
        max_position_embeddings=128,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        sliding_window=16,
        rms_norm_eps=1e-6,
        hidden_act="gelu",
        attention_bias=False,
        mlp_bias=False,
        headwise_attn_output_gate=True,
        gate_attn_act_mode="sigmoid",
        tie_word_embeddings=False,
        rope_parameters={
            "full_attention": {"rope_theta": 5000000, "partial_rotary_factor": 0.25},
            "sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0},
        },
    )
    torch.manual_seed(671)
    shapes = {"model.embedding.weight": (256, 128), "lm_head.weight": (256, 128), "model.norm.weight": (128,)}
    mapping = {"final_norm": "model.norm.weight", "embedding": "model.embedding.weight", "lm_head": "lm_head.weight"}
    for layer in range(4):
        for name, hf, shape in (
            ("ln1_weight", "input_layernorm.weight", (128,)),
            ("ln2_weight", "post_attention_layernorm.weight", (128,)),
            ("qkv_weight", "self_attn.q_k_v_proj.weight", (512, 128)),
            ("attn_gate_weight", "self_attn.g_proj.weight", (4, 128)),
            ("out_weight", "self_attn.out_proj.weight", (128, 256)),
            ("mlp_gate_weight", "mlp.gate_proj.weight", (256, 128)),
            ("mlp_up_weight", "mlp.up_proj.weight", (256, 128)),
            ("mlp_down_weight", "mlp.down_proj.weight", (128, 256)),
        ):
            hf = f"model.layers.{layer}.{hf}"
            shapes[hf] = shape
            mapping[f"blocks[{layer}].{name}"] = hf
    weights = {
        name: (torch.ones(shape) if len(shape) == 1 else torch.randn(shape) * 0.035).bfloat16()
        for name, shape in shapes.items()
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    save_file(weights, str(tmp_path / "model.safetensors"))
    return cfg, weights, mapping


def reference(torch, cfg, weights, inputs, targets):
    f = torch.nn.functional
    w = {k: v.cuda().requires_grad_() for k, v in weights.items()}
    x = f.embedding(torch.as_tensor(inputs, device="cuda", dtype=torch.long), w["model.embedding.weight"]).float()
    seq = inputs.shape[1]
    pos = torch.arange(seq, device="cuda")

    def norm(x, weight):
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + cfg["rms_norm_eps"]) * weight.float()).bfloat16()

    def rope(x, theta, width):
        angles = pos.float()[:, None] * (theta ** (-torch.arange(0, width, 2, device="cuda").float() / width))
        cos, sin = angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]
        a, b = x[..., :width].float().chunk(2, dim=-1)
        return torch.cat(((a * cos - b * sin).bfloat16(), (a * sin + b * cos).bfloat16(), x[..., width:]), -1)

    for layer, kind in enumerate(cfg["layer_types"]):
        p = f"model.layers.{layer}."
        h = norm(x, w[p + "input_layernorm.weight"])
        qkv = f.linear(h, w[p + "self_attn.q_k_v_proj.weight"])
        q, k, v = qkv.split((256, 128, 128), dim=-1)
        q, k, v = q.reshape(1, seq, 4, 64), k.reshape(1, seq, 2, 64), v.reshape(1, seq, 2, 64)
        local = kind == "sliding_attention"
        q, k = [rope(t, 10000 if local else 5000000, 64 if local else 16) for t in (q, k)]
        k, v = k.repeat_interleave(2, dim=2), v.repeat_interleave(2, dim=2)
        scores = torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-1, -2)).float() / 8
        allowed = pos[:, None] >= pos[None, :]
        if local:
            allowed &= pos[:, None] - pos[None, :] < cfg["sliding_window"]
        probs = scores.masked_fill(~allowed, float("-inf")).softmax(-1).bfloat16()
        att = torch.matmul(probs, v.transpose(1, 2)).transpose(1, 2)
        gate = f.linear(h, w[p + "self_attn.g_proj.weight"]).float().sigmoid().bfloat16()
        att = (att * gate[..., None]).reshape(1, seq, 256)
        x = x + f.linear(att, w[p + "self_attn.out_proj.weight"]).float()
        h = norm(x, w[p + "post_attention_layernorm.weight"])
        h = f.gelu(f.linear(h, w[p + "mlp.gate_proj.weight"]), approximate="none") * f.linear(
            h, w[p + "mlp.up_proj.weight"]
        )
        x = x + f.linear(h, w[p + "mlp.down_proj.weight"]).float()
    logits = f.linear(norm(x, w["model.norm.weight"]), w["lm_head.weight"])
    loss = f.cross_entropy(
        logits.float().flatten(0, 1),
        torch.as_tensor(targets, device="cuda", dtype=torch.long).flatten(),
        reduction="sum",
    )
    loss.backward()
    return loss.item(), {k: v.grad.float().cpu().numpy() for k, v in w.items()}


@pytest.mark.parametrize("recompute", [False, True])
def test_spark_backward_and_optimizer(tmp_path, monkeypatch, recompute):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    ext = pytest.importorskip("surogate._surogate")
    from surogate.dsl.ir_builder import build_dsl_ir_for_model

    cfg, weights, mapping = checkpoint(tmp_path, torch)
    inputs = np.random.default_rng(614).integers(0, cfg["vocab_size"], (1, 32), dtype=np.int32)
    targets = np.roll(inputs, -1, axis=1).copy()
    targets[:, -1] = -100
    expected_loss, gradients = reference(torch, cfg, weights, inputs, targets)
    monkeypatch.setenv("SUROGATE_MIN_STACK_MB", "128")
    options = ext.RuntimeOptions(
        recompute=str(recompute).lower(),
        use_cuda_graphs=recompute,
        offload_residual=False,
        offload_master=False,
        offload_grads=False,
        offload_optimizer=False,
        shard_gradients=True,
        use_zero_copy=False,
    )
    options.dsl_ir_json = build_dsl_ir_for_model(str(tmp_path))
    trainer = ext.SurogateTrainer(
        ngpu=1,
        config=ext.PretrainedConfig.from_pretrained(str(tmp_path), "bf16"),
        options=options,
        batch_size=1,
        seq_len=32,
        grad_accum=1,
        memcpy_all_gather=True,
        memcpy_send_recv=True,
    )
    trainer.import_weights(str(tmp_path / "model.safetensors"))
    trainer.step(inputs, targets)
    actual = trainer.get_gradients(0)
    for name, hf in mapping.items():
        result = torch.utils.dlpack.from_dlpack(actual[name]).float().cpu().numpy()
        expected = gradients[hf]
        relative = np.linalg.norm(result - expected) / max(np.linalg.norm(expected), 1e-8)
        assert relative < 0.06, (name, relative)
    first = trainer.update_with_config(ext.OptimizerConfig(learning_rate=1e-4), 1)
    assert first["loss"] == pytest.approx(expected_loss / 31, rel=0.003)
    assert np.isfinite(first["norm"]) and first["norm"] > 0
    trainer.step(inputs, targets)
    second = trainer.update_with_config(ext.OptimizerConfig(learning_rate=1e-4), 2)
    assert np.isfinite(second["norm"]) and 0 < second["loss"] < first["loss"]
    exported = tmp_path / "exported"
    trainer.export_model(str(exported))
    saved = json.loads((exported / "config.json").read_text())
    for key in ("rope_parameters", "hidden_act", "headwise_attn_output_gate", "gate_attn_act_mode", "layer_types"):
        assert saved[key] == cfg[key]
    assert build_dsl_ir_for_model(str(exported))


@pytest.mark.parametrize("targets", [["all"], ["q_k_v_proj", "out_proj", "gate_proj", "up_proj", "down_proj"]])
def test_spark_lora_export_reload_and_merge(tmp_path, monkeypatch, targets):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    ext = pytest.importorskip("surogate._surogate")
    from safetensors.torch import load_file

    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.utils.adapter_merge import merge_adapter

    base = tmp_path / "base"
    base.mkdir()
    cfg, weights, _ = checkpoint(base, torch)
    inputs = np.random.default_rng(514).integers(0, cfg["vocab_size"], (1, 32), dtype=np.int32)
    labels = np.roll(inputs, -1, axis=1).copy()
    labels[:, -1] = -100
    monkeypatch.setenv("SUROGATE_MIN_STACK_MB", "128")
    options = ext.RuntimeOptions(
        recompute="true",
        use_cuda_graphs=True,
        offload_residual=False,
        offload_master=False,
        offload_grads=False,
        offload_optimizer=False,
        shard_gradients=True,
        use_zero_copy=False,
    )
    options.dsl_ir_json = build_dsl_ir_for_model(str(base))
    trainer = ext.SurogateTrainer(
        ngpu=1,
        config=ext.PretrainedConfig.from_pretrained(str(base), "bf16"),
        options=options,
        batch_size=1,
        seq_len=32,
        grad_accum=1,
        memcpy_all_gather=True,
        memcpy_send_recv=True,
        lora_config=ext.LoRAAdapterConfig(rank=8, alpha=16, target_modules=targets),
    )
    trainer.import_weights(str(base / "model.safetensors"))
    losses = []
    for step in range(1, 4):
        trainer.step(inputs, labels)
        result = trainer.update_with_config(ext.OptimizerConfig(learning_rate=1e-3), step)
        assert np.isfinite(result["norm"]) and result["norm"] > 0
        losses.append(result["loss"])
    assert losses[-1] < losses[0]
    adapter = tmp_path / "adapter"
    trainer.export_adapter(str(adapter), str(base))
    params = load_file(str(adapter / "adapter_model.safetensors"))
    metadata = json.loads((adapter / "adapter_config.json").read_text())
    assert set(metadata["target_modules"]) == {"q_k_v_proj", "out_proj", "gate_proj", "up_proj", "down_proj"}
    assert len(params) == 4 * 5 * 2
    for name, tensor in params.items():
        assert torch.isfinite(tensor).all()
        assert tensor.count_nonzero() > 0, name
        base_name = name.removeprefix("base_model.model.").split(".lora_")[0] + ".weight"
        assert base_name in weights
        expected = (8, weights[base_name].shape[1]) if ".lora_A." in name else (weights[base_name].shape[0], 8)
        assert tuple(tensor.shape) == expected
    trainer.import_adapter(str(adapter / "adapter_model.safetensors"))
    reloaded = tmp_path / "reloaded"
    trainer.export_adapter(str(reloaded), str(base))
    for name, tensor in load_file(str(reloaded / "adapter_model.safetensors")).items():
        assert torch.equal(tensor, params[name])
    merged = tmp_path / "merged"
    merge_adapter(str(base), str(adapter), str(merged))
    result = load_file(str(merged / "model.safetensors"))
    assert result.keys() == weights.keys()
    for name, original in weights.items():
        prefix = "base_model.model." + name.removesuffix(".weight")
        if prefix + ".lora_A.weight" not in params:
            assert torch.equal(result[name], original)
            continue
        delta = params[prefix + ".lora_B.weight"].float() @ params[prefix + ".lora_A.weight"].float()
        assert torch.equal(result[name], (original.float() + 2 * delta).bfloat16())
