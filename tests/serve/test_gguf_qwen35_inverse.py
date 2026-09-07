# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# Round-trip validation of the qwen35 GGUF inverse transforms
# (surogate/serve/gguf/qwen35.py) against a verbatim port of llama.cpp's
# FORWARD export transforms (study/llama.cpp/conversion/qwen.py:
# Qwen3NextModel.modify_tensors + _LinearAttentionVReorderBase). Forward is
# reimplemented here from the studied source so a bug in the inverse cannot
# hide by construction. CPU-only, synthetic geometry.

import pytest

torch = pytest.importorskip("torch")

from surogate.serve.gguf.qwen35 import (
    GdnGeometry,
    hf_name_for,
    invert_tensor,
)

# Small but non-degenerate geometry: 2 K heads, 6 V heads (r=3), distinct dims.
G = GdnGeometry(num_k_heads=2, num_v_heads=6, head_k_dim=4, head_v_dim=8)


# --- llama.cpp forward transforms, ported verbatim --------------------------

def fwd_reorder_v_heads(tensor, dim, num_k_heads, num_v_per_k, head_dim):
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1:]
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).contiguous().reshape(*shape)


def fwd_qkv(t):
    q_dim = G.head_k_dim * G.num_k_heads
    q, k, v = t[:q_dim], t[q_dim:2 * q_dim], t[2 * q_dim:]
    v = fwd_reorder_v_heads(v, 0, G.num_k_heads, G.num_v_per_k, G.head_v_dim)
    return torch.cat([q, k, v], dim=0)


def fwd_conv1d(t):
    data = t.squeeze()
    qk = G.head_k_dim * G.num_k_heads * 2
    v = fwd_reorder_v_heads(data[qk:], 0, G.num_k_heads, G.num_v_per_k, G.head_v_dim)
    return torch.cat([data[:qk], v], dim=0)


def fwd_1d(t):
    return fwd_reorder_v_heads(t.unsqueeze(-1), 0, G.num_k_heads, G.num_v_per_k, 1).squeeze(-1)


# --- round-trip cases -------------------------------------------------------

HIDDEN = 16
rng = torch.Generator().manual_seed(3)


def rand(*shape):
    return torch.randn(*shape, generator=rng, dtype=torch.float32)


def test_qkv_v_segment_round_trip():
    rows = G.head_k_dim * G.num_k_heads * 2 + G.head_v_dim * G.num_v_heads
    hf = rand(rows, HIDDEN)
    back = invert_tensor("model.layers.0.linear_attn.in_proj_qkv.weight", fwd_qkv(hf), G)
    torch.testing.assert_close(back, hf, rtol=0, atol=0)


def test_z_rows_round_trip():
    hf = rand(G.head_v_dim * G.num_v_heads, HIDDEN)
    fwd = fwd_reorder_v_heads(hf, 0, G.num_k_heads, G.num_v_per_k, G.head_v_dim)
    back = invert_tensor("model.layers.0.linear_attn.in_proj_z.weight", fwd, G)
    torch.testing.assert_close(back, hf, rtol=0, atol=0)


def test_ab_rows_round_trip():
    hf = rand(G.num_v_heads, HIDDEN)
    fwd = fwd_reorder_v_heads(hf, 0, G.num_k_heads, G.num_v_per_k, 1)
    for which in ("in_proj_a", "in_proj_b"):
        back = invert_tensor(f"model.layers.0.linear_attn.{which}.weight", fwd, G)
        torch.testing.assert_close(back, hf, rtol=0, atol=0)


def test_out_proj_cols_round_trip():
    hf = rand(HIDDEN, G.head_v_dim * G.num_v_heads)
    fwd = fwd_reorder_v_heads(hf, 1, G.num_k_heads, G.num_v_per_k, G.head_v_dim)
    back = invert_tensor("model.layers.0.linear_attn.out_proj.weight", fwd, G)
    torch.testing.assert_close(back, hf, rtol=0, atol=0)


def test_a_log_numeric_and_order_round_trip():
    hf = -torch.rand(G.num_v_heads, generator=rng) - 0.5  # A_log is negative-ish; any real works
    fwd = fwd_1d(-torch.exp(hf))  # llama.cpp: -exp(A_log), then tiled order
    back = invert_tensor("model.layers.0.linear_attn.A_log", fwd, G)
    torch.testing.assert_close(back, hf, rtol=1e-6, atol=1e-6)


def test_dt_bias_round_trip():
    hf = rand(G.num_v_heads)
    back = invert_tensor("model.layers.0.linear_attn.dt_bias", fwd_1d(hf), G)
    torch.testing.assert_close(back, hf, rtol=0, atol=0)


def test_conv1d_squeeze_and_channels_round_trip():
    channels = G.head_k_dim * G.num_k_heads * 2 + G.head_v_dim * G.num_v_heads
    hf = rand(channels, 1, 4)  # HF layout [C, 1, K]
    back = invert_tensor("model.layers.0.linear_attn.conv1d.weight", fwd_conv1d(hf), G)
    torch.testing.assert_close(back, hf, rtol=0, atol=0)


def test_norm_plus_one_round_trip():
    hf = rand(HIDDEN)
    for name in (
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
    ):
        back = invert_tensor(name, hf + 1.0, G)
        # (γ+1)−1 is inherently lossy at the fp32 ulp level near 1.0 — that
        # rounding exists in the GGUF itself (llama.cpp stores γ+1 in F32) and
        # is far below the BF16 precision of the original checkpoint. Bound it
        # at 2 fp32 ulps rather than pretending bit-exactness is achievable.
        torch.testing.assert_close(back, hf, rtol=0, atol=2.4e-7)
    # linear_attn.norm is stored RAW — must NOT be shifted.
    back = invert_tensor("model.layers.0.linear_attn.norm.weight", hf, G)
    torch.testing.assert_close(back, hf, rtol=0, atol=0)


def test_no_reorder_when_heads_match():
    sym = GdnGeometry(num_k_heads=4, num_v_heads=4, head_k_dim=4, head_v_dim=8)
    hf = rand(sym.head_v_dim * sym.num_v_heads, HIDDEN)
    back = invert_tensor("model.layers.0.linear_attn.in_proj_z.weight", hf, sym)
    torch.testing.assert_close(back, hf, rtol=0, atol=0)


# --- name mapping (mtp-aware) ----------------------------------------------

def test_hf_names_main_layers():
    assert hf_name_for("token_embd.weight", 3) == "model.embed_tokens.weight"
    assert hf_name_for("blk.0.attn_qkv.weight", 3) == "model.layers.0.linear_attn.in_proj_qkv.weight"
    assert hf_name_for("blk.0.attn_gate.weight", 3) == "model.layers.0.linear_attn.in_proj_z.weight"
    assert hf_name_for("blk.1.ssm_a", 3) == "model.layers.1.linear_attn.A_log"
    assert hf_name_for("blk.1.ssm_dt.bias", 3) == "model.layers.1.linear_attn.dt_bias"
    assert hf_name_for("blk.2.attn_q.weight", 3) == "model.layers.2.self_attn.q_proj.weight"
    assert hf_name_for("output_norm.weight", 3) == "model.norm.weight"


def test_hf_names_mtp_block():
    # llama.cpp: mtp.fc -> blk.{n_main}.nextn.eh_proj, mtp.layers.0.* -> blk.{n_main}.*
    assert hf_name_for("blk.3.nextn.eh_proj.weight", 3) == "mtp.fc.weight"
    assert hf_name_for("blk.3.nextn.enorm.weight", 3) == "mtp.pre_fc_norm_embedding.weight"
    assert hf_name_for("blk.3.nextn.hnorm.weight", 3) == "mtp.pre_fc_norm_hidden.weight"
    assert hf_name_for("blk.3.nextn.shared_head_norm.weight", 3) == "mtp.norm.weight"
    assert hf_name_for("blk.3.attn_q.weight", 3) == "mtp.layers.0.self_attn.q_proj.weight"
    assert hf_name_for("blk.3.ffn_down.weight", 3) == "mtp.layers.0.mlp.down_proj.weight"
