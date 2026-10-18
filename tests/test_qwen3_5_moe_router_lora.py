"""MoE router LoRA (`train_router`) and export_model's `layer_types`, on a tiny random Qwen3.5-MoE.

Three defects (2026-10-18):
  * The router matmul computes its logits in FP32 and returned before the generic LoRA epilogue, so the
    forward never applied the router LoRA while the backward trained it: log-probs with a nonzero router
    adapter were bit-identical to the base model's, the merged model (W + s*B@A) differed from the trained one,
    and every LoRA gradient upstream of a nonzero router adapter was off (2-4x HF's own bf16 noise against an
    fp32 reference).
  * LoRA random_init skipped the router adapter: lora_A kept its allocation's contents (zeros), and with
    A = B = 0 both of its gradients vanish, so a train_router run never moved the router.
  * save_pretrained_config wrote `layer_types` from a sliding/full coding: export_model of a Qwen3.5/3.6 hybrid
    model came back with every layer "full_attention" (HF then builds attention layers where the checkpoint has
    GatedDeltaNet weights), and a VLM-layout config gained a wrong top-level list next to text_config's.

GPU, no checkpoint needed (a 4-layer, 8-expert random fixture: 3 GatedDeltaNet layers + 1 full-attention layer):
  1. a fresh adapter's router lora_A is initialised like every other target (B = 0), and one step gives the
     router lora_B a gradient;
  2. a router-only adapter moves the per-token log-probs, and they match a LoRA-free trainer on the checkpoint
     with the router adapter merged (W + (alpha/r) B@A) to well within the adapter's effect;
  3. LoRA gradients (router included) agree with an fp32 PyTorch reference (the HF modeling code with the
     adapter as PEFT defines it, the router's top-k softmax with the LoRA added to its logits) as closely as
     HF's own bf16 run does;
  4. export_model keeps `layer_types` for a text config (HF re-loads the export with the source's logits) and
     for a VLM-layout config (text_config kept, no wrong top-level list).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

RANK, ALPHA = 8, 16  # scale 2: the scaling must reach the router LoRA too
SCALE = ALPHA / RANK
SEQ = 256
TEXT = dict(hidden_size=128, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=128,
            vocab_size=512, max_position_embeddings=4096, rms_norm_eps=1e-6, tie_word_embeddings=False,
            linear_num_key_heads=2, linear_key_head_dim=32, linear_num_value_heads=4, linear_value_head_dim=64,
            linear_conv_kernel_dim=4, eos_token_id=1, attention_bias=False,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            num_experts=8, num_experts_per_tok=2, moe_intermediate_size=64, shared_expert_intermediate_size=64,
            rope_parameters=dict(rope_type="default", rope_theta=1e7, partial_rotary_factor=.5,
                                 mrope_section=[11, 11, 10], mrope_interleaved=True))

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
def fixture_dir(tmp_path_factory):
    """Random qwen3_5_moe_text checkpoint (bf16 on disk, fused experts as the real checkpoint stores them)."""
    _gpu_setup()
    import transformers
    from safetensors.torch import save_file

    out = tmp_path_factory.mktemp("qwen35moe_router")
    cfg = dict(TEXT, architectures=["Qwen3_5MoeForCausalLM"])
    torch.manual_seed(353)
    model = transformers.Qwen3_5MoeForCausalLM(transformers.AutoConfig.for_model("qwen3_5_moe_text", **cfg)).float()
    hv = TEXT["linear_num_value_heads"]
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("A_log"):  # slow decays (HF's init forgets within a few tokens)
                p.copy_(torch.log(torch.linspace(0.02, 1.0, hv)))
            elif name.endswith("dt_bias"):
                p.copy_(torch.linspace(-4.0, 0.0, hv))
            elif p.dim() == 1:
                p.copy_(0.1 * torch.randn_like(p))
            elif name.endswith("mlp.gate.weight"):  # HF leaves the router at zeros
                p.copy_(0.1 * torch.randn_like(p))
    save_file({k: v.detach().to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()},
              str(out / "model.safetensors"), metadata={"format": "pt"})
    model.config.save_pretrained(out)
    written = json.loads((out / "config.json").read_text())
    written["architectures"] = cfg["architectures"]
    (out / "config.json").write_text(json.dumps(written, indent=1))
    return out


def _trainer(model_dir, lora=True, init_only=False):
    from surogate import _surogate as sg
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.kernels.jit_compile import compile_jit_kernels
    from surogate.utils.hf import get_model_weights_path

    os.environ.setdefault("SUROGATE_ROPE_MAX_SEQ", str(SEQ))
    opts = sg.RuntimeOptions(recipe="bf16", use_cuda_graphs=False, offload_master=False, offload_grads=False,
                             offload_optimizer=False, offload_residual=False, shard_gradients=True)
    opts.recompute = "true"
    opts.router_aux_loss_coef = 0.0  # the reference has no aux/z loss
    opts.router_z_loss_coef = 0.0
    opts.dsl_ir_json = build_dsl_ir_for_model(str(model_dir))
    manifests = compile_jit_kernels(opts.dsl_ir_json)
    if manifests:
        opts.jit_kernel_manifests = manifests
    lora_config = (sg.LoRAAdapterConfig(rank=RANK, alpha=ALPHA, dropout=0, dtype="bf16", target_modules=["all"],
                                        train_router=True) if lora else None)
    tr = sg.SurogateTrainer(ngpu=1, config=sg.PretrainedConfig.from_pretrained(str(model_dir), "bf16"),
                            options=opts, batch_size=1, seq_len=SEQ, grad_accum=1, memcpy_all_gather=True,
                            memcpy_send_recv=True, lora_config=lora_config, qlora_config=None)
    if init_only:
        tr.init_weights()
    else:
        tr.import_weights(get_model_weights_path(str(model_dir)))
    return tr


def _tokens(seed=11):
    rng = np.random.default_rng(seed)
    x = rng.integers(3, 500, size=(1, SEQ), dtype=np.int32)
    y = np.concatenate([x[:, 1:], np.full((1, 1), -100, np.int32)], axis=1).astype(np.int32)
    return x, y


def _router_keys(adapter):
    return sorted(k for k in adapter if ".mlp.gate.lora_" in k)


def _router_only(init, seed=20261018):
    """The router LoRA large (A, B ~ 0.1 randn); every other adapter's B = 0, so only the router moves."""
    g = torch.Generator().manual_seed(seed)
    out = {k: (v if "lora_A" in k else torch.zeros_like(v)) for k, v in init.items()}
    for k in _router_keys(init):
        out[k] = (torch.randn(init[k].shape, generator=g) * 0.1).to(init[k].dtype)
    return out


def _all_random(init, seed=20261019):
    g = torch.Generator().manual_seed(seed)
    return {k: (torch.randn(v.shape, generator=g) * (0.05 if "lora_A" in k else 0.02)).to(v.dtype)
            for k, v in sorted(init.items())}


def _init_adapter(tr, model_dir, tmp):
    from safetensors.torch import load_file

    tr.export_adapter(str(tmp / "init"), str(model_dir))
    return load_file(str(tmp / "init" / "adapter_model.safetensors"))


# ---------------------------------------------------------------- reference


def _reference(model_dir, adapter, x, y, dtype):
    """Mean-CE gradients of every LoRA tensor, HF Qwen3.5-MoE modeling code with the adapter applied as PEFT
    defines it: y = W x + (alpha / r) B (A x)."""
    import torch.nn.functional as F
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype,
                                                              attn_implementation="eager").cuda().eval()
    named = dict(model.named_modules())

    def router_forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        a, b = self._lora
        logits = F.linear(hidden_states, self.weight) + SCALE * ((hidden_states @ a.t()) @ b.t())
        probs = torch.softmax(logits.float(), dim=-1)  # Qwen3_5MoeTopKRouter: softmax, top-k, renormalise
        top, idx = torch.topk(probs, self.top_k, dim=-1)
        top = top / top.sum(dim=-1, keepdim=True)
        return logits, top.to(logits.dtype), idx

    def experts_forward(self, hidden_states, top_k_index, top_k_weights):
        lo = self._lora
        final = torch.zeros_like(hidden_states)
        for e in range(self.num_experts):
            pos, tok = torch.where((top_k_index == e).t())
            if tok.numel() == 0:
                continue
            s = hidden_states[tok]
            gate, up = F.linear(s, self.gate_up_proj[e]).chunk(2, dim=-1)
            a, b = lo[(e, "gate_proj")]
            gate = gate + SCALE * ((s @ a.t()) @ b.t())
            a, b = lo[(e, "up_proj")]
            up = up + SCALE * ((s @ a.t()) @ b.t())
            h = self.act_fn(gate) * up
            a, b = lo[(e, "down_proj")]
            o = F.linear(h, self.down_proj[e]) + SCALE * ((h @ a.t()) @ b.t())
            final.index_add_(0, tok, (o * top_k_weights[tok, pos, None]).to(final.dtype))
        return final

    params, experts = {}, {}
    for k in sorted(adapter):
        if ".lora_A." not in k:
            continue
        module = k[: k.index(".lora_A.")]
        a = adapter[k].to(dtype).cuda().requires_grad_(True)
        b = adapter[module + ".lora_B.weight"].to(dtype).cuda().requires_grad_(True)
        params[module] = (a, b)
        hf = module[len("base_model.model."):].replace(".shared_experts.", ".shared_expert.")
        m = re.match(r"(.*\.mlp\.experts)\.(\d+)\.(\w+)$", hf)
        if m:
            experts.setdefault(m[1], {})[(int(m[2]), m[3])] = (a, b)
        elif hf.endswith(".mlp.gate"):
            named[hf]._lora = (a, b)
            named[hf].forward = router_forward.__get__(named[hf])
        else:
            named[hf].register_forward_hook(lambda mod, inp, o, a=a, b=b: o + SCALE * ((inp[0] @ a.t()) @ b.t()))
    for path, lo in experts.items():
        named[path]._lora = lo
        named[path].forward = experts_forward.__get__(named[path])
    logits = model(input_ids=torch.from_numpy(x).long().cuda(), use_cache=False).logits.float()
    labels = torch.from_numpy(y).long().cuda()
    F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100).backward()
    grads = {}
    for module, (a, b) in params.items():
        grads[module + ".lora_A.weight"] = a.grad.double().cpu()
        grads[module + ".lora_B.weight"] = b.grad.double().cpu()
    del model
    torch.cuda.empty_cache()
    # per-expert reference tensors -> the trainer's grouped [E, ...] gradients
    per, out = {}, {}
    for k, v in grads.items():
        m = re.match(r"(.*\.mlp\.experts)\.(\d+)\.(.*)$", k)
        if m:
            per.setdefault(f"{m[1]}.{m[3]}", {})[int(m[2])] = v
        else:
            out[k] = v
    for k, by_e in per.items():
        out[k] = torch.stack([by_e[e] for e in sorted(by_e)])
    return out


def _group(name):
    if ".mlp.experts." in name:
        return "experts"
    if ".mlp.gate." in name:
        return "router"
    if "shared_expert" in name:
        return "shared"
    if ".linear_attn." in name:
        return "gdn"
    return "attn"


# ---------------------------------------------------------------- tests


@pytest.mark.gpu
def test_router_lora_init_is_trainable(fixture_dir, tmp_path):
    """Initialised like every other target, A ~ N(0, 1/(3 C)) and B = 0, so lora_B has a gradient at step one."""
    x, y = _tokens()
    tr = _trainer(fixture_dir)
    init = _init_adapter(tr, fixture_dir, tmp_path)
    keys = _router_keys(init)
    assert len(keys) == 2 * TEXT["num_hidden_layers"], keys
    std = 1.0 / np.sqrt(3.0 * TEXT["hidden_size"])
    for k in keys:
        t = init[k].float()
        if ".lora_A." in k:
            assert 0.7 * std < float(t.std()) < 1.3 * std, (k, float(t.std()), std)
        else:
            assert float(t.abs().max()) == 0.0, k
    tr.step(x, y)
    grads = {k: torch.from_dlpack(v).float().cpu() for k, v in tr.get_lora_gradients(0).items()}
    for k in keys:
        if ".lora_B." in k:
            assert float(grads[k].norm()) > 0.0, f"{k}: no gradient (router LoRA cannot train)"
    del tr
    torch.cuda.empty_cache()


@pytest.mark.gpu
def test_router_lora_forward_matches_merged_router(fixture_dir, tmp_path):
    """A router-only adapter moves the per-token log-probs exactly as merging it into the router weight does."""
    from safetensors.torch import load_file, save_file

    x, y = _tokens()
    valid = y[0] != -100
    tr = _trainer(fixture_dir)
    init = _init_adapter(tr, fixture_dir, tmp_path)
    keys = _router_keys(init)
    adapter = _router_only(init)
    save_file(adapter, str(tmp_path / "router.safetensors"))
    tr.import_adapter(str(tmp_path / "router.safetensors"))
    lp_lora = tr.compute_logprobs(x, y, use_lora=True)[0][valid].astype(np.float64)
    lp_base = tr.compute_logprobs(x, y, use_lora=False)[0][valid].astype(np.float64)
    del tr
    torch.cuda.empty_cache()

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_bytes((fixture_dir / "config.json").read_bytes())
    weights = load_file(str(fixture_dir / "model.safetensors"))
    for k in keys:
        if ".lora_A." in k:
            module = k[: k.index(".lora_A.")]
            key = module[len("base_model.model."):] + ".weight"
            delta = SCALE * (adapter[module + ".lora_B.weight"].float() @ adapter[k].float())
            weights[key] = (weights[key].float() + delta).to(weights[key].dtype)
    save_file(weights, str(merged / "model.safetensors"), metadata={"format": "pt"})
    plain = _trainer(merged, lora=False)
    lp_merged = plain.compute_logprobs(x, y)[0][valid].astype(np.float64)
    del plain
    torch.cuda.empty_cache()

    effect = np.abs(lp_merged - lp_base).mean()
    diff = np.abs(lp_lora - lp_merged)
    print(json.dumps({"router_lora_effect": float(effect), "lora_vs_merged_mean": float(diff.mean()),
                      "lora_vs_merged_max": float(diff.max()),
                      "lora_vs_base_mean": float(np.abs(lp_lora - lp_base).mean())}))
    assert effect > 0.01, effect  # the adapter moves the model
    assert diff.mean() <= 0.1 * effect, (float(diff.mean()), float(effect))
    assert diff.max() <= 0.1, float(diff.max())


@pytest.mark.gpu
@pytest.mark.parametrize("variant", ["all_random", "router_only"])
def test_router_lora_gradients_match_reference(fixture_dir, tmp_path, variant):
    from safetensors.torch import save_file

    x, y = _tokens()
    n_valid = int((y != -100).sum())
    tr = _trainer(fixture_dir)
    init = _init_adapter(tr, fixture_dir, tmp_path)
    adapter = _all_random(init) if variant == "all_random" else _router_only(init)
    save_file(adapter, str(tmp_path / "live.safetensors"))
    tr.import_adapter(str(tmp_path / "live.safetensors"))
    tr.step(x, y)
    # get_lora_gradients returns the step's raw sums; the reference is the mean loss's gradient
    ours = {k: torch.from_dlpack(v).double().cpu() / n_valid for k, v in tr.get_lora_gradients(0).items()}
    del tr
    torch.cuda.empty_cache()
    ref = _reference(fixture_dir, adapter, x, y, torch.float32)
    hf_bf16 = _reference(fixture_dir, adapter, x, y, torch.bfloat16)  # HF's own bf16 noise floor
    assert set(ours) == set(ref), sorted(set(ours) ^ set(ref))[:4]

    groups = {}
    for name in sorted(ours):
        groups.setdefault(_group(name), []).append(name)
    assert groups.get("router"), "no router LoRA gradients"
    report = {}
    for g, names in groups.items():
        v = torch.cat([ours[n].flatten() for n in names])
        r = torch.cat([ref[n].flatten() for n in names])
        b = torch.cat([hf_bf16[n].flatten() for n in names])
        report[g] = (float((v - r).norm() / r.norm()), float((b - r).norm() / r.norm()),
                     float(torch.nn.functional.cosine_similarity(v, r, dim=0)))
    print(json.dumps({"variant": variant, "relL2_trainer_vs_fp32": {g: v[0] for g, v in report.items()},
                      "relL2_hf_bf16_vs_fp32": {g: v[1] for g, v in report.items()}}, indent=1))
    for g, (ours_rel, floor, cos) in report.items():
        assert ours_rel <= 1.5 * floor + 0.02, (g, ours_rel, floor, report)
        assert cos > 0.98, (g, cos, report)


@pytest.mark.gpu
def test_export_model_keeps_layer_types_text(fixture_dir, tmp_path):
    import transformers

    tr = _trainer(fixture_dir, lora=False)
    tr.export_model(str(tmp_path / "export"))
    del tr
    torch.cuda.empty_cache()
    source = json.loads((fixture_dir / "config.json").read_text())
    exported = json.loads((tmp_path / "export" / "config.json").read_text())
    assert exported["layer_types"] == source["layer_types"], exported.get("layer_types")
    x = torch.from_numpy(_tokens()[0][:, :64]).long().cuda()
    logits = {}
    for tag, d in (("source", fixture_dir), ("export", tmp_path / "export")):
        model = transformers.AutoModelForCausalLM.from_pretrained(str(d), dtype=torch.float32).cuda().eval()
        assert sum(hasattr(layer, "linear_attn") for layer in model.model.layers) == 3, tag
        with torch.no_grad():
            logits[tag] = model(input_ids=x).logits.float()
        del model
    assert float((logits["source"] - logits["export"]).abs().max()) < 1e-4


@pytest.mark.gpu
def test_export_model_keeps_layer_types_vlm_layout(tmp_path):
    """The real Surogate3.7 / Qwen3.6 layout: Qwen3_5MoeForConditionalGeneration, layer_types in text_config."""
    _gpu_setup()
    import transformers

    text = {k: v for k, v in TEXT.items() if k not in ("eos_token_id",)}
    text["full_attention_interval"] = 4
    vision = dict(depth=1, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=128,
                  num_position_embeddings=256)
    cfg = transformers.AutoConfig.for_model("qwen3_5_moe", text_config=text, vision_config=vision).to_dict()
    cfg.update(architectures=["Qwen3_5MoeForConditionalGeneration"], image_token_id=508, video_token_id=509,
               vision_start_token_id=510, vision_end_token_id=511)
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text(json.dumps(cfg, indent=1))
    tr = _trainer(src, lora=False, init_only=True)
    tr.export_model(str(tmp_path / "export"))
    del tr
    torch.cuda.empty_cache()
    exported = json.loads((tmp_path / "export" / "config.json").read_text())
    assert exported["text_config"]["layer_types"] == TEXT["layer_types"], exported["text_config"].get("layer_types")
    assert exported.get("layer_types") in (None, TEXT["layer_types"]), exported.get("layer_types")
    assert exported["architectures"] == ["Qwen3_5MoeForConditionalGeneration"]
