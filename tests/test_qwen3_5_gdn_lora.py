"""LoRA on the Qwen3.5-family linear-attention (GatedDeltaNet) mixer projections.

Before this change `lora_target_modules: ["all"]` gave a Qwen3.5 model adapters on q/k/v/o (full-attention
layers only) and on the MLPs, but none on the linear-attention layers' in_proj_qkv, in_proj_z, in_proj_a,
in_proj_b and out_proj -- three quarters of the layers carried no mixer adapter at all (Qwen3.8-27B: 39.8M
instead of 58.4M rank-8 parameters).

CPU: the DSL declares one LoRA target per HF projection tensor (lin_qkv, lin_z, lin_a, lin_b, lin_out).
GPU, on two models -- a 4-layer truncation of Qwen3.5-0.8B (Hv = Hk) and a random 4-layer fixture with the
Qwen3.8-27B value/key head ratio (Hv = 3 Hk, in_proj_qkv wider than every dense projection) -- with recompute on:
  * the exported adapter carries `<layer>.linear_attn.<proj>.lora_{A,B}.weight` with the HF shapes, and its
    adapter_config.json lists the modules;
  * with a nonzero adapter, every token's log-prob equals the one of the base weights merged with W + B@A
    (a second trainer, no LoRA) to well within the adapter's own effect, and the LoRA gradients of the new
    projections agree with an fp32 HF/PyTorch reference (adapters as forward hooks) as closely as the
    existing q/k/v/o/MLP ones do;
  * an adapter without the linear-attention modules (the pre-change `all`) is refused by name when the trainer
    allocates them, and loads with the explicit old target list; optimizer state of another geometry is
    refused on resume (AdamW 8-bit, the default).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from surogate.dsl.py_compiler import compile_model_for_hf

LINEAR = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
DSL_NAMES = {"in_proj_qkv_weight": "lin_qkv", "in_proj_z_weight": "lin_z", "in_proj_a_weight": "lin_a",
             "in_proj_b_weight": "lin_b", "out_weight": "lin_out"}


def _mini_text_config():
    return {
        "hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 16, "intermediate_size": 128, "vocab_size": 256, "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6, "attention_bias": False, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 16,
        "linear_value_head_dim": 16, "linear_num_key_heads": 2, "linear_num_value_heads": 6,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"], "full_attention_interval": 4,
        "rope_parameters": {"partial_rotary_factor": 0.25, "mrope_section": [1, 1, 0], "rope_theta": 10000.0},
    }


def _params(ir: dict) -> dict:
    mods = ir.get("modules") or []
    for m in mods:
        fwd = m.get("forward") or {}
        if fwd.get("params"):
            return fwd["params"]
    raise AssertionError("no forward params in the IR")


class TestGdnLoraTargetsDsl:
    def test_linear_mixer_declares_one_target_per_projection(self):
        cfg = {"architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text", **_mini_text_config()}
        raw = compile_model_for_hf("Qwen3_5ForCausalLM", cfg)
        ir = json.loads(raw) if isinstance(raw, str) else raw
        assert ir.get("success"), ir.get("errors")
        params = _params(ir)
        c, conv_dim, value_dim, hv = 64, 2 * 2 * 16 + 6 * 16, 6 * 16, 6
        sizes = {"lin_qkv": conv_dim, "lin_z": value_dim, "lin_a": hv, "lin_b": hv, "lin_out": c}
        seen = {}
        for name, info in params.items():
            for target in info.get("lora_targets") or []:
                if target["name"].startswith("lin_"):
                    seen.setdefault(target["name"], []).append((name, target))
        assert sorted(seen) == sorted(sizes), seen.keys()
        for target_name, entries in seen.items():
            assert len(entries) == 3, (target_name, len(entries))  # the three linear-attention layers
            for name, target in entries:
                assert target.get("size") == sizes[target_name], (name, target)
                assert int(target.get("offset", 0)) == 0
        # convolution, decay and norm parameters are not LoRA targets
        for name, info in params.items():
            if any(key in name for key in ("conv_weight", "A_log", "dt_bias", "norm_weight")) and ".mixer" in name:
                assert not info.get("lora_targets"), name


# ---------------------------------------------------------------- GPU


torch = None
OLD_ALL = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _gpu_setup():
    global torch
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    try:
        import surogate._surogate  # noqa: F401
    except ImportError:
        pytest.skip("surogate._surogate C++ extension not built")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


def _hv3_fixture(snapshot, out):
    """Random Qwen3.5 fixture with Hv = 3 Hk (the 27B's ratio): in_proj_qkv (1,280 rows) is wider than the
    hidden size (256) and the MLP (512), which the LoRA scratch slice must still hold."""
    import transformers

    if (out / "config.json").exists():
        return out
    cfg = json.loads((snapshot / "config.json").read_text())
    t = cfg["text_config"]
    t.update(hidden_size=256, intermediate_size=512, num_hidden_layers=4, num_attention_heads=2,
             num_key_value_heads=1, head_dim=128, vocab_size=1024, linear_num_key_heads=2, linear_num_value_heads=6,
             linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4, tie_word_embeddings=False,
             layer_types=["linear_attention"] * 3 + ["full_attention"])
    cfg.update(vocab_size=1024, tie_word_embeddings=False)
    cfg["vision_config"].update(depth=1, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=256,
                                num_position_embeddings=256)
    torch.manual_seed(353)
    model = transformers.Qwen3_5ForConditionalGeneration(transformers.AutoConfig.for_model(**cfg)).to(torch.bfloat16)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1 and not name.endswith(("A_log", "dt_bias")):
                p.copy_(0.1 * torch.randn_like(p, dtype=torch.float32).to(p.dtype))
    model.save_pretrained(out, safe_serialization=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if (snapshot / f).exists():
            (out / f).write_bytes((snapshot / f).read_bytes())
    return out


@pytest.fixture(scope="module", params=["qwen35_0.8b_mini", "hv3"])
def model_case(request, tmp_path_factory):
    _gpu_setup()
    from tests import test_onboarding_qwen3_5 as onboarding

    snapshot = onboarding.resolve_model_path()
    if snapshot is None:
        pytest.skip("Qwen3.5-0.8B not available (set QWEN3_5_MODEL_PATH)")
    if request.param == "hv3":
        return request.param, _hv3_fixture(snapshot, tmp_path_factory.mktemp("qwen35_hv3")), 1000
    return request.param, onboarding.prepare_mini_model(snapshot), 8000


def _trainer(model_dir, targets, seq_len, recompute="true"):
    from surogate import _surogate as sg
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.kernels.jit_compile import compile_jit_kernels
    from surogate.utils.hf import get_model_weights_path

    opts = sg.RuntimeOptions(recipe="bf16", use_cuda_graphs=False, offload_master=False, offload_grads=False,
                             offload_optimizer=False, offload_residual=False, shard_gradients=True)
    if recompute:
        opts.recompute = recompute
    opts.dsl_ir_json = build_dsl_ir_for_model(str(model_dir))
    manifests = compile_jit_kernels(opts.dsl_ir_json)
    if manifests:
        opts.jit_kernel_manifests = manifests
    lora_config = (sg.LoRAAdapterConfig(rank=8, alpha=8, dropout=0, dtype="bf16", target_modules=targets)
                   if targets else None)
    tr = sg.SurogateTrainer(ngpu=1, config=sg.PretrainedConfig.from_pretrained(str(model_dir), "bf16"),
                            options=opts, batch_size=1, seq_len=seq_len, grad_accum=1, memcpy_all_gather=True,
                            memcpy_send_recv=True, lora_config=lora_config, qlora_config=None)
    tr.import_weights(get_model_weights_path(str(model_dir)))
    return tr


def _tokens(seq, vocab_hi, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.integers(10, vocab_hi, size=(1, seq), dtype=np.int32)
    y = np.concatenate([x[:, 1:], np.full((1, 1), -100, np.int32)], axis=1).astype(np.int32)
    return x, y


def _hf_name(adapter_module: str) -> str:
    # base_model.model.model.layers.N.<rest>  ->  model.language_model.layers.N.<rest>
    rest = adapter_module[len("base_model.model."):]
    return "model.language_model." + rest[len("model."):] if rest.startswith("model.layers.") else rest


def _random_adapter(init, seed=20260925):
    g = torch.Generator().manual_seed(seed)
    return {k: (torch.randn(v.shape, generator=g) * (0.05 if "lora_A" in k else 0.02)).to(v.dtype)
            for k, v in init.items()}


@pytest.mark.gpu
@pytest.mark.slow
def test_gdn_lora_export_merge_parity_and_gradients(model_case):
    from safetensors.torch import load_file, save_file

    name, model_dir, vocab_hi = model_case
    seq = 64
    x, y = _tokens(seq, vocab_hi)
    valid = y[0] != -100
    text = json.loads((model_dir / "config.json").read_text())["text_config"]
    c = text["hidden_size"]
    key_dim = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    value_dim = text["linear_num_value_heads"] * text["linear_value_head_dim"]
    expect = {"in_proj_qkv": (c, 2 * key_dim + value_dim), "in_proj_z": (c, value_dim),
              "in_proj_a": (c, text["linear_num_value_heads"]), "in_proj_b": (c, text["linear_num_value_heads"]),
              "out_proj": (value_dim, c)}
    tr = _trainer(model_dir, ["all"], seq)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tr.export_adapter(str(tmp / "init"), str(model_dir))
        init = load_file(str(tmp / "init" / "adapter_model.safetensors"))
        cfg = json.loads((tmp / "init" / "adapter_config.json").read_text())
        # 1. export: every linear-attention layer carries the five HF modules with the HF shapes
        for layer in range(3):
            for proj, (fan_in, fan_out) in expect.items():
                a = init[f"base_model.model.model.layers.{layer}.linear_attn.{proj}.lora_A.weight"]
                b = init[f"base_model.model.model.layers.{layer}.linear_attn.{proj}.lora_B.weight"]
                assert tuple(a.shape) == (8, fan_in) and tuple(b.shape) == (fan_out, 8), (proj, a.shape, b.shape)
        assert not any(".linear_attn." in k and ".layers.3." in k for k in init)  # the full-attention layer
        assert any(".layers.3.self_attn.q_proj." in k for k in init)
        assert set(LINEAR) <= set(cfg["target_modules"]), cfg["target_modules"]

        # 2. a nonzero adapter: per-token log-probs, then one step's gradients
        live = _random_adapter(init)
        save_file(live, str(tmp / "live.safetensors"))
        tr.import_adapter(str(tmp / "live.safetensors"))
        base_lp = tr.compute_logprobs(x, y, use_lora=False)[0][valid].astype(np.float64)
        lora_lp = tr.compute_logprobs(x, y, use_lora=True)[0][valid].astype(np.float64)
        tr.step(x, y)
        grads = {k: torch.from_dlpack(v).float().cpu() for k, v in tr.get_lora_gradients(0).items()}
        del tr
        torch.cuda.empty_cache()

        # 3. per token: LoRA forward == merged weights (W + B@A in fp32, then bf16) in a trainer without LoRA
        merged_dir = tmp / "merged"
        merged_dir.mkdir()
        for f in model_dir.iterdir():
            if f.is_file() and not f.name.endswith(".safetensors"):
                (merged_dir / f.name).write_bytes(f.read_bytes())
        modules = sorted({k[: k.index(".lora_")] for k in live})
        for shard in model_dir.glob("*.safetensors"):
            tensors = load_file(str(shard))
            for module in modules:
                key = _hf_name(module) + ".weight"
                if key in tensors:
                    w = tensors[key]
                    delta = live[module + ".lora_B.weight"].float() @ live[module + ".lora_A.weight"].float()
                    tensors[key] = (w.float() + delta).to(w.dtype)
            save_file(tensors, str(merged_dir / shard.name), metadata={"format": "pt"})
        merged = _trainer(merged_dir, None, seq)
        merged_lp = merged.compute_logprobs(x, y)[0][valid].astype(np.float64)
        del merged
        torch.cuda.empty_cache()
        effect = np.abs(lora_lp - base_lp).mean()
        diff = np.abs(lora_lp - merged_lp)
        assert effect > 0.05, effect  # the adapter moves the model
        assert diff.mean() <= 0.1 * effect and diff.max() <= 0.25, (name, float(diff.mean()), float(diff.max()),
                                                                    float(effect))

        # 4. gradients against an fp32 HF reference with the adapters as forward hooks
        from transformers import AutoModelForImageTextToText

        # The mini checkpoint shrinks the (unused) vision tower, hence ignore_mismatched_sizes.
        hf = AutoModelForImageTextToText.from_pretrained(str(model_dir), dtype=torch.float32,
                                                         ignore_mismatched_sizes=True).cuda().eval()
        named = dict(hf.named_modules())
        params, hooks = {}, []
        for module in modules:
            target = named[_hf_name(module)]
            a = live[module + ".lora_A.weight"].float().cuda().requires_grad_(True)
            b = live[module + ".lora_B.weight"].float().cuda().requires_grad_(True)
            params[module] = (a, b)
            hooks.append(target.register_forward_hook(lambda m, inp, out, a=a, b=b: out + (inp[0] @ a.t()) @ b.t()))
        logits = hf(input_ids=torch.from_numpy(x).long().cuda()).logits.float()
        labels = torch.from_numpy(y).long().cuda()
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)
        loss.backward()
        for h in hooks:
            h.remove()
        del hf
        torch.cuda.empty_cache()
        # get_lora_gradients returns the step's raw sums (the optimizer normalises by the supervised-token
        # count); the reference is the mean loss's gradient.
        n_valid = int(valid.sum())

        def rel(module):
            ours = torch.cat([grads[module + ".lora_A.weight"].flatten(), grads[module + ".lora_B.weight"].flatten()])
            ours = ours / n_valid
            ref = torch.cat([params[module][0].grad.flatten().cpu(), params[module][1].grad.flatten().cpu()])
            return float((ours - ref).norm() / ref.norm().clamp_min(1e-12)), float(
                torch.nn.functional.cosine_similarity(ours, ref, dim=0))

        existing = [m for m in modules if ".linear_attn." not in m]
        new = [m for m in modules if ".linear_attn." in m]
        assert len(new) == 15, new
        floor = max(rel(m)[0] for m in existing)
        report = {m: rel(m) for m in new}
        worst = max(v[0] for v in report.values())
        print(json.dumps({"model": name, "existing_worst_relL2": floor, "linear_attn_worst_relL2": worst,
                          "linear_attn_min_cosine": min(cos for _, cos in report.values()),
                          "per_token_lora_vs_merged": {"mean": float(diff.mean()), "max": float(diff.max())},
                          "per_token_adapter_effect_mean": float(effect)}, indent=1))
        assert floor <= 0.2, floor  # the existing adapters against the same reference (calibration)
        assert all(cos > 0.98 for _, cos in report.values()), report
        assert worst <= max(0.1, 1.5 * floor), (worst, floor, report)


@pytest.mark.gpu
@pytest.mark.slow
def test_adapter_without_linear_attention_is_refused(model_case):
    """The pre-change `all` adapter (q/k/v/o/MLP only) must not import silently into an `all` trainer, and
    optimizer state of another geometry must not be restored into it; the explicit old list loads it."""
    from safetensors.torch import load_file, save_file
    from surogate import _surogate as sg

    name, model_dir, vocab_hi = model_case
    if name != "qwen35_0.8b_mini":
        pytest.skip("one model is enough")
    seq = 64
    x, y = _tokens(seq, vocab_hi)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        old = _trainer(model_dir, OLD_ALL, seq)
        old.export_adapter(str(tmp / "old"), str(model_dir))
        old_weights = load_file(str(tmp / "old" / "adapter_model.safetensors"))
        assert not any(".linear_attn." in k for k in old_weights)
        # An old-style run's checkpoint with AdamW 8-bit state (the default optimizer).
        old.step(x, y)
        old.update_with_config(sg.OptimizerConfig(optimizer="adamw_8bit", learning_rate=1e-4), 0)
        old.save_checkpoint(str(tmp / "ckpt"), 1)
        # Round trip with the explicit old list: loads.
        old.import_adapter(str(tmp / "old" / "adapter_model.safetensors"))
        del old
        torch.cuda.empty_cache()

        # A refused import leaves the trainer defunct (any worker exception does), so one trainer per case.
        def refused(action, match):
            trainer = _trainer(model_dir, ["all"], seq)
            try:
                with pytest.raises(RuntimeError, match=match):
                    action(trainer)
            finally:
                del trainer
                torch.cuda.empty_cache()

        refused(lambda t: t.import_adapter(str(tmp / "old" / "adapter_model.safetensors")),
                r"lacks 30 of the \d+ LoRA tensors.*in_proj_qkv")
        refused(lambda t: t.load_checkpoint(str(tmp / "ckpt"), 1), "lacks 30 of the")
        # Optimizer state of another geometry, behind an adapter that does match: refused before any update.
        new = _trainer(model_dir, ["all"], seq)
        new.export_adapter(str(tmp / "new"), str(model_dir))
        del new
        torch.cuda.empty_cache()
        ckpt_adapters = list((tmp / "ckpt").rglob("adapter_model.safetensors"))
        assert ckpt_adapters, list((tmp / "ckpt").rglob("*"))
        for adapter_file in ckpt_adapters:
            adapter_file.write_bytes((tmp / "new" / "adapter_model.safetensors").read_bytes())
        refused(lambda t: t.load_checkpoint(str(tmp / "ckpt"), 1), "checkpoint geometry does not match the adapter")
