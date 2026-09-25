"""LoRA on the Qwen3.5-family linear-attention (GatedDeltaNet) mixer projections.

Before this change `lora_target_modules: ["all"]` gave a Qwen3.5 model adapters on q/k/v/o (full-attention
layers only) and on the MLPs, but none on the linear-attention layers' in_proj_qkv, in_proj_z, in_proj_a,
in_proj_b and out_proj -- three quarters of the layers carried no mixer adapter at all (Qwen3.8-27B: 39.8M
instead of 58.4M rank-8 parameters).

CPU: the DSL declares one LoRA target per HF projection tensor (lin_qkv, lin_z, lin_a, lin_b, lin_out).
GPU (a 4-layer truncation of Qwen3.5-0.8B, three linear-attention layers and one full-attention layer):
  * the exported adapter carries `<layer>.linear_attn.<proj>.lora_{A,B}.weight` with the HF shapes, and its
    adapter_config.json lists the modules;
  * with a nonzero adapter, the trainer's loss equals the loss of the base weights merged with W + B@A
    (a second trainer, no LoRA), and the LoRA gradients of the new projections agree with an fp32
    HF/PyTorch reference (adapters as forward hooks) as closely as the existing q/k/v/o/MLP ones do.
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


@pytest.fixture(scope="module")
def mini_model():
    _gpu_setup()
    from tests import test_onboarding_qwen3_5 as onboarding

    snapshot = onboarding.resolve_model_path()
    if snapshot is None:
        pytest.skip("Qwen3.5-0.8B not available (set QWEN3_5_MODEL_PATH)")
    return onboarding.prepare_mini_model(snapshot)


def _trainer(model_dir, lora, seq_len):
    from surogate import _surogate as sg
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.kernels.jit_compile import compile_jit_kernels

    opts = sg.RuntimeOptions(recipe="bf16", use_cuda_graphs=False, offload_master=False, offload_grads=False,
                             offload_optimizer=False, offload_residual=False, shard_gradients=True)
    opts.dsl_ir_json = build_dsl_ir_for_model(str(model_dir))
    manifests = compile_jit_kernels(opts.dsl_ir_json)
    if manifests:
        opts.jit_kernel_manifests = manifests
    lora_config = sg.LoRAAdapterConfig(rank=8, alpha=8, dropout=0, dtype="bf16", target_modules=["all"]) if lora else None
    tr = sg.SurogateTrainer(ngpu=1, config=sg.PretrainedConfig.from_pretrained(str(model_dir), "bf16"),
                            options=opts, batch_size=1, seq_len=seq_len, grad_accum=1, memcpy_all_gather=True,
                            memcpy_send_recv=True, lora_config=lora_config, qlora_config=None)
    from surogate.utils.hf import get_model_weights_path

    tr.import_weights(get_model_weights_path(str(model_dir)))
    return tr


def _hf_name(adapter_module: str) -> str:
    # base_model.model.model.layers.N.<rest>  ->  model.language_model.layers.N.<rest>
    rest = adapter_module[len("base_model.model."):]
    return "model.language_model." + rest[len("model."):] if rest.startswith("model.layers.") else rest


@pytest.mark.gpu
@pytest.mark.slow
def test_gdn_lora_export_forward_and_gradients(mini_model):
    from safetensors.torch import load_file, save_file

    seq = 64
    rng = np.random.default_rng(7)
    x = rng.integers(10, 8000, size=(1, seq), dtype=np.int32)
    y = np.concatenate([x[:, 1:], np.full((1, 1), -100, np.int32)], axis=1).astype(np.int32)
    tr = _trainer(mini_model, lora=True, seq_len=seq)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tr.export_adapter(str(tmp / "init"), str(mini_model))
        init = load_file(str(tmp / "init" / "adapter_model.safetensors"))
        cfg = json.loads((tmp / "init" / "adapter_config.json").read_text())
        # 1. export: every linear-attention layer carries the five HF modules with the HF shapes
        c = 1024
        for layer in range(3):
            for proj in LINEAR:
                a = init[f"base_model.model.model.layers.{layer}.linear_attn.{proj}.lora_A.weight"]
                b = init[f"base_model.model.model.layers.{layer}.linear_attn.{proj}.lora_B.weight"]
                assert a.shape[0] == 8 and b.shape[1] == 8
                assert (a.shape[1] == c) == (proj != "out_proj") and (b.shape[0] == c) == (proj == "out_proj")
        assert not any(".linear_attn." in k and ".layers.3." in k for k in init)  # the full-attention layer
        assert any(".layers.3.self_attn.q_proj." in k for k in init)
        assert set(LINEAR) <= set(cfg["target_modules"]), cfg["target_modules"]

        # 2. a nonzero adapter
        g = torch.Generator().manual_seed(20260925)
        live = {k: (torch.randn(v.shape, generator=g) * (0.05 if "lora_A" in k else 0.02)).to(v.dtype)
                for k, v in init.items()}
        save_file(live, str(tmp / "live.safetensors"))
        tr.import_adapter(str(tmp / "live.safetensors"))
        tr.step(x, y)
        grads = {k: torch.from_dlpack(v).float().cpu() for k, v in tr.get_lora_gradients(0).items()}
        lora_loss = float(tr.validate(x, y))
        del tr
        torch.cuda.empty_cache()

        # 3. forward == merged weights (W + B@A, fp32 then bf16) in a trainer without LoRA
        merged_dir = tmp / "merged"
        merged_dir.mkdir()
        for f in mini_model.iterdir():
            if f.is_file() and not f.name.endswith(".safetensors"):
                (merged_dir / f.name).write_bytes(f.read_bytes())
        modules = sorted({k[: k.index(".lora_")] for k in live})
        for shard in mini_model.glob("*.safetensors"):
            tensors = load_file(str(shard))
            for module in modules:
                name = _hf_name(module) + ".weight"
                if name in tensors:
                    w = tensors[name]
                    delta = live[module + ".lora_B.weight"].float() @ live[module + ".lora_A.weight"].float()
                    tensors[name] = (w.float() + delta).to(w.dtype)
            save_file(tensors, str(merged_dir / shard.name), metadata={"format": "pt"})
        base = _trainer(merged_dir, lora=False, seq_len=seq)
        merged_loss = float(base.validate(x, y))
        del base
        torch.cuda.empty_cache()
        assert abs(lora_loss - merged_loss) <= 2e-2 * max(1.0, abs(merged_loss)), (lora_loss, merged_loss)

        # 4. gradients against an fp32 HF reference with the adapters as forward hooks
        from transformers import AutoModelForImageTextToText

        # The mini checkpoint shrinks the (unused) vision tower, hence ignore_mismatched_sizes.
        hf = AutoModelForImageTextToText.from_pretrained(str(mini_model), dtype=torch.float32,
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
        assert abs(float(loss.detach()) - merged_loss) <= 5e-2 * max(1.0, abs(float(loss.detach()))), (
            float(loss.detach()), merged_loss)

        # get_lora_gradients returns the step's raw sums (the optimizer normalises by the supervised-token
        # count); the reference is the mean loss's gradient.
        n_valid = int((y != -100).sum())

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
        print(json.dumps({"existing_worst_relL2": floor, "linear_attn_worst_relL2": worst,
                          "linear_attn_min_cosine": min(c for _, c in report.values()), "lora_loss": lora_loss,
                          "merged_loss": merged_loss, "hf_fp32_loss": float(loss.detach())}, indent=1))
        assert floor <= 0.2, floor  # the existing adapters against the same reference (calibration)
        assert all(cos > 0.98 for _, cos in report.values()), report
        assert worst <= max(0.1, 1.5 * floor), (worst, floor, report)
