"""Chunked-sequence training (`sequence_chunks`) against the unchunked step, on a tiny random Qwen3.5-MoE
(3 Gated-DeltaNet layers + 1 full-attention layer, 8 routed experts top-2 + a shared expert; GDN decay rates of
the 35B-A3B's, so the recurrent state carries across chunks).

1. With the router aux/z losses off, the chunked step's loss and LoRA gradients equal the unchunked step's to
   within bf16 noise: per parameter group (attention, GDN, routed experts, shared expert) the relative L2
   difference stays at the level two bf16 implementations of the same step differ by (measured 2026-10-18 on
   an RTX 5090: <= 0.07 at 4 chunks; HF's own bf16 vs fp32 gradients of such fixtures differ by 0.03-0.08,
   routing flips of near-tie tokens included). The same sequence cut into independent pieces -- every carry
   (GDN state and convolution tail, attention KV prefix, the gradients back through them) missing -- differs
   by 0.5-1.1, so the agreement is not vacuous.
2. The router z-loss is one token mean per micro-step, chunked or not. Each chunk's backward used to add its
   own chunk mean, so `sequence_chunks: N` weighed the router aux/z losses N times as much as the unchunked
   step (here, 8 chunks and a large z coefficient: relative L2 3.7-5.6 against the unchunked gradient before
   the fix, 0.02-0.06 after).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

SEQ = 512
RANK = 8
E, TOP_K, C, M = 8, 2, 128, 64
HK, KD, HV, VD, CONV_K = 2, 32, 4, 64, 4
HQ, HKV, HD = 2, 1, 128
VOCAB = 512
LAYERS = ["linear_attention"] * 3 + ["full_attention"]
# Per-layer GDN decay parameters picked from Surogate3.7-35B-A3B's (slow and fast heads).
A_LOG = [[-3.31, -3.97, -2.5, -3.06], [-0.67, -1.98, 0.54, -1.39], [2.88, 0.21, 0.70, 3.69]]
DT_BIAS = [[-4.03, -1.95, 15.56, -4.25], [0.01, -5.03, -0.09, -3.47], [-6.06, -1.94, -1.13, -3.84]]


def _fixture(root: Path) -> Path:
    """A random checkpoint in the HF layout (fused experts.gate_up_proj / down_proj), written directly."""
    import torch
    from safetensors.torch import save_file

    out = root / "tiny_qwen3_5_moe"
    if (out / "config.json").exists():
        return out
    out.mkdir(parents=True)
    g = torch.Generator().manual_seed(353)

    def w(*shape, std=0.02):
        return (torch.randn(*shape, generator=g) * std).to(torch.bfloat16)

    t = {"model.embed_tokens.weight": w(VOCAB, C), "model.norm.weight": w(C, std=0.1), "lm_head.weight": w(VOCAB, C)}
    conv_dim = 2 * HK * KD + HV * VD
    for i, kind in enumerate(LAYERS):
        p = f"model.layers.{i}."
        t[p + "input_layernorm.weight"] = w(C, std=0.1)
        t[p + "post_attention_layernorm.weight"] = w(C, std=0.1)
        if kind == "linear_attention":
            t[p + "linear_attn.A_log"] = torch.tensor(A_LOG[i], dtype=torch.bfloat16)
            t[p + "linear_attn.dt_bias"] = torch.tensor(DT_BIAS[i], dtype=torch.bfloat16)
            t[p + "linear_attn.conv1d.weight"] = w(conv_dim, 1, CONV_K)
            t[p + "linear_attn.norm.weight"] = w(VD, std=0.1)
            t[p + "linear_attn.in_proj_qkv.weight"] = w(conv_dim, C)
            t[p + "linear_attn.in_proj_z.weight"] = w(HV * VD, C)
            t[p + "linear_attn.in_proj_a.weight"] = w(HV, C)
            t[p + "linear_attn.in_proj_b.weight"] = w(HV, C)
            t[p + "linear_attn.out_proj.weight"] = w(C, HV * VD)
        else:
            t[p + "self_attn.q_proj.weight"] = w(2 * HQ * HD, C)  # query and output gate
            t[p + "self_attn.k_proj.weight"] = w(HKV * HD, C)
            t[p + "self_attn.v_proj.weight"] = w(HKV * HD, C)
            t[p + "self_attn.o_proj.weight"] = w(C, HQ * HD)
            t[p + "self_attn.q_norm.weight"] = w(HD, std=0.1)
            t[p + "self_attn.k_norm.weight"] = w(HD, std=0.1)
        t[p + "mlp.gate.weight"] = w(E, C, std=0.1)
        t[p + "mlp.experts.gate_up_proj"] = w(E, 2 * M, C)
        t[p + "mlp.experts.down_proj"] = w(E, C, M)
        t[p + "mlp.shared_expert.gate_proj.weight"] = w(M, C)
        t[p + "mlp.shared_expert.up_proj.weight"] = w(M, C)
        t[p + "mlp.shared_expert.down_proj.weight"] = w(C, M)
        t[p + "mlp.shared_expert_gate.weight"] = w(1, C)
    save_file({k: v.contiguous() for k, v in t.items()}, str(out / "model.safetensors"), metadata={"format": "pt"})
    cfg = dict(architectures=["Qwen3_5MoeForCausalLM"], model_type="qwen3_5_moe_text", hidden_size=C,
               num_hidden_layers=len(LAYERS), num_attention_heads=HQ, num_key_value_heads=HKV, head_dim=HD,
               vocab_size=VOCAB, max_position_embeddings=4096, rms_norm_eps=1e-6, tie_word_embeddings=False,
               attention_bias=False, linear_num_key_heads=HK, linear_key_head_dim=KD, linear_num_value_heads=HV,
               linear_value_head_dim=VD, linear_conv_kernel_dim=CONV_K, layer_types=LAYERS, eos_token_id=1,
               num_experts=E, num_experts_per_tok=TOP_K, moe_intermediate_size=M, shared_expert_intermediate_size=M,
               rope_parameters=dict(rope_type="default", rope_theta=1e7, partial_rotary_factor=0.5,
                                    mrope_section=[11, 11, 10], mrope_interleaved=True))
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    return out


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    torch = pytest.importorskip("torch")
    sg = pytest.importorskip("surogate._surogate", reason="needs the built extension")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    # The chunk graphs size their RoPE tables from the chunk; positions reach the full sequence.
    os.environ["SUROGATE_ROPE_MAX_SEQ"] = str(SEQ)
    root = tmp_path_factory.mktemp("seq_chunks")
    model_dir = _fixture(root)
    rng = np.random.default_rng(11)
    x = rng.integers(3, VOCAB - 12, size=(1, SEQ), dtype=np.int32)
    y = np.concatenate([x[:, 1:], np.full((1, 1), -100, np.int32)], axis=1).astype(np.int32)

    # A nonzero adapter on every LoRA target (the trainer's own init has B = 0).
    from safetensors.torch import load_file, save_file

    tr = _trainer(sg, model_dir, chunks=1)
    tr.export_adapter(str(root / "init"), str(model_dir))
    del tr
    init = load_file(str(root / "init" / "adapter_model.safetensors"))
    g = torch.Generator().manual_seed(20261018)
    live = {k: (torch.randn(v.shape, generator=g) * (0.05 if "lora_A" in k else 0.02)).to(v.dtype)
            for k, v in sorted(init.items())}
    save_file(live, str(root / "live.safetensors"))
    return sg, model_dir, root / "live.safetensors", x, y


def _trainer(sg, model_dir, *, chunks, accum=1, seq_len=None, aux=0.0, z=0.0):
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.kernels.jit_compile import compile_jit_kernels
    from surogate.utils.hf import get_model_weights_path

    opts = sg.RuntimeOptions(recipe="bf16", use_cuda_graphs=False, offload_master=False, offload_grads=False,
                             offload_optimizer=False, offload_residual=False, shard_gradients=True)
    opts.recompute = "true"
    if chunks > 1:
        opts.sequence_chunks = chunks
    opts.router_aux_loss_coef = aux
    opts.router_z_loss_coef = z
    opts.dsl_ir_json = build_dsl_ir_for_model(str(model_dir))
    manifests = compile_jit_kernels(opts.dsl_ir_json)
    if manifests:
        opts.jit_kernel_manifests = manifests
    lora = sg.LoRAAdapterConfig(rank=RANK, alpha=RANK, dropout=0, dtype="bf16", target_modules=["all"],
                                train_router=False)
    tr = sg.SurogateTrainer(ngpu=1, config=sg.PretrainedConfig.from_pretrained(str(model_dir), "bf16"),
                            options=opts, batch_size=1, seq_len=seq_len or SEQ // chunks, grad_accum=accum,
                            memcpy_all_gather=True, memcpy_send_recv=True, lora_config=lora, qlora_config=None)
    tr.import_weights(get_model_weights_path(str(model_dir)))
    return tr


def _run(setup, *, chunks=1, independent=1, aux=0.0, z=0.0):
    """One step's loss and raw LoRA gradients. independent=N: the sequence as N separate sequences (global
    positions kept, no carry of any kind), accumulated over N micro-steps."""
    import torch

    sg, model_dir, live, x, y = setup
    t = SEQ // independent
    tr = _trainer(sg, model_dir, chunks=chunks, accum=independent, seq_len=t if independent > 1 else None,
                  aux=aux, z=z)
    tr.import_adapter(str(live))
    if independent > 1:
        pos = np.arange(SEQ, dtype=np.int32).reshape(1, SEQ)
        for c in range(independent):
            sl = slice(c * t, (c + 1) * t)
            tr.step(np.ascontiguousarray(x[:, sl]), np.ascontiguousarray(y[:, sl]), np.ascontiguousarray(pos[:, sl]))
    else:
        tr.step(x, y)
    grads = {k: torch.from_dlpack(v).double().cpu().numpy().copy() for k, v in tr.get_lora_gradients(0).items()}
    flush = sg.OptimizerConfig(optimizer="adamw_8bit", learning_rate=0.0, weight_decay=0.0, grad_clip=0.0)
    loss = float(tr.update_with_config(flush, 1)["loss"])
    del tr
    return loss, grads


def _group(name):
    for key, group in ((".mlp.experts.", "experts"), (".linear_attn.", "gdn"), (".self_attn.", "attention"),
                       ("shared_expert", "shared")):
        if key in name:
            return group
    return "other"


def _rel_by_group(a, b):
    out = {}
    for group in sorted({_group(n) for n in b}):
        names = sorted(n for n in b if _group(n) == group)
        va = np.concatenate([a[n].ravel() for n in names])
        vb = np.concatenate([b[n].ravel() for n in names])
        out[group] = float(np.linalg.norm(va - vb) / np.linalg.norm(vb))
    return out


@pytest.mark.gpu
def test_chunked_step_matches_the_unchunked_step(setup):
    loss1, g1 = _run(setup, chunks=1)
    loss4, g4 = _run(setup, chunks=4)
    _, cut = _run(setup, independent=4)
    chunked = _rel_by_group(g4, g1)
    no_carry = _rel_by_group(cut, g1)
    print(json.dumps({"loss": [loss1, loss4], "chunked_relL2": chunked, "independent_pieces_relL2": no_carry}))
    assert set(chunked) == {"attention", "gdn", "experts", "shared"}, chunked
    assert loss4 == pytest.approx(loss1, rel=1e-4)
    for group, err in chunked.items():
        assert err < 0.12, (group, chunked)
        # the carries matter in this fixture: without them the gradient is another one
        assert no_carry[group] > max(0.25, 4 * err), (group, chunked, no_carry)


@pytest.mark.gpu
def test_router_z_loss_weight_does_not_grow_with_the_chunk_count(setup):
    # A large z coefficient, so the router regularizer is a sizeable part of every LoRA gradient upstream of a
    # router. With the old per-chunk mean, 8 chunks carried 8 times the unchunked z-loss.
    z = 10.0
    _, g1 = _run(setup, chunks=1, z=z)
    _, g8 = _run(setup, chunks=8, z=z)
    _, g1_plain = _run(setup, chunks=1)
    effect = _rel_by_group(g1, g1_plain)
    chunked = _rel_by_group(g8, g1)
    print(json.dumps({"z_loss_share_unchunked": effect, "chunked_vs_unchunked_relL2": chunked}))
    assert max(effect.values()) > 0.3, effect  # the z-loss gradient is not lost in bf16 noise here
    for group, err in chunked.items():
        assert err < 0.12, (group, chunked)
