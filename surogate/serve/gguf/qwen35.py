# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# Inverse of llama.cpp's qwen35-family GGUF export transforms.
#
# llama.cpp's converter (study/llama.cpp/conversion/qwen.py) does NOT store
# HF-layout tensors for the Qwen3.5/3.6 family. On export it applies, in
# Qwen3NextModel/_LinearAttentionVReorderBase/_QwenMtpMixin:
#   1. RMSNorm folding:  every *norm.weight except linear_attn.norm gets +1;
#   2. A_log:            stored as -exp(A_log);
#   3. dt_bias:          renamed dt_proj.bias (gguf: ssm_dt.bias);
#   4. conv1d:           squeezed from [C, 1, K] to [C, K];
#   5. V-head reorder:   when linear_num_key_heads != linear_num_value_heads,
#      V-indexed rows/cols/channels are permuted from HF grouped order
#      [G0v0..G0v{r-1}, G1v0..] to ggml tiled order [G0v0, G1v0, ..] in:
#      in_proj_qkv (V row segment), in_proj_z (rows), in_proj_a/b (rows),
#      A_log & dt_bias (elements), conv1d (V channel span), out_proj (COLUMNS);
#   6. MTP remap:        HF mtp.* tensors become extra layers past block_count
#      ("nextn": mtp.fc->eh_proj, pre_fc_norm_embedding->enorm,
#      pre_fc_norm_hidden->hnorm, norm->shared_head.norm,
#      mtp.layers.j.* -> model.layers.{n_main+j}.*), with the count in KV
#      {arch}.nextn_predict_layers.
#
# A GGUF-sourced conversion that skips these inversions produces silently
# damaged weights (every norm off by +1, GDN state mis-ordered). This module
# undoes all six exactly. Permutations are built by running llama.cpp's own
# index arithmetic forward on an index vector and inverting with argsort, so
# a direction mistake is structurally impossible; the round-trip unit test
# (tests/serve/test_gguf_qwen35_inverse.py) asserts identity against a port
# of the forward code.

from __future__ import annotations

import re
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GdnGeometry:
    num_k_heads: int      # KV: {arch}.ssm.group_count      (linear_num_key_heads)
    num_v_heads: int      # KV: {arch}.ssm.time_step_rank   (linear_num_value_heads)
    head_k_dim: int       # KV: {arch}.ssm.state_size       (linear_key_head_dim)
    head_v_dim: int       # inner_size / num_v_heads        (linear_value_head_dim)

    @property
    def num_v_per_k(self) -> int:
        return self.num_v_heads // self.num_k_heads

    @property
    def reordered(self) -> bool:
        return self.num_k_heads != self.num_v_heads


def _forward_perm(g: GdnGeometry, head_dim: int) -> torch.Tensor:
    """llama.cpp's grouped->tiled index permutation, verbatim arithmetic."""
    idx = torch.arange(g.num_v_heads * head_dim, dtype=torch.long)
    idx = idx.reshape(g.num_k_heads, g.num_v_per_k, head_dim)
    idx = idx.permute(1, 0, 2).contiguous().reshape(-1)
    return idx


def _inverse_perm(g: GdnGeometry, head_dim: int) -> torch.Tensor:
    return torch.argsort(_forward_perm(g, head_dim))


def invert_v_rows(t: torch.Tensor, g: GdnGeometry, head_dim: int) -> torch.Tensor:
    """Undo a tiled-order row permutation (dim 0)."""
    if not g.reordered:
        return t
    return t.index_select(0, _inverse_perm(g, head_dim))


def invert_v_cols(t: torch.Tensor, g: GdnGeometry, head_dim: int) -> torch.Tensor:
    """Undo a tiled-order column permutation (dim 1) — out_proj."""
    if not g.reordered:
        return t
    return t.index_select(1, _inverse_perm(g, head_dim))


def invert_qkv_v_segment(t: torch.Tensor, g: GdnGeometry) -> torch.Tensor:
    """in_proj_qkv rows are [q | k | v]; only the v segment was reordered."""
    if not g.reordered:
        return t
    qk = g.head_k_dim * g.num_k_heads * 2
    return torch.cat([t[:qk], invert_v_rows(t[qk:], g, g.head_v_dim)], dim=0)


def invert_conv1d(t: torch.Tensor, g: GdnGeometry) -> torch.Tensor:
    """conv1d: unsqueeze back to [C, 1, K]; V channel span was reordered."""
    if t.ndim == 2:
        t = t.unsqueeze(1)  # inverse of forward squeeze()
    if not g.reordered:
        return t
    qk_channels = g.head_k_dim * g.num_k_heads * 2
    qk, v = t[:qk_channels], t[qk_channels:]
    v = v.squeeze(1)
    v = invert_v_rows(v, g, g.head_v_dim)
    return torch.cat([qk, v.unsqueeze(1)], dim=0)


_BLK = re.compile(r"^blk\.(\d+)\.(.+)$")

# gguf suffix -> HF suffix under model.layers.N., for qwen35 GDN + attention +
# MLP + norms. Generic-map preferences mispick several of these (ssm_a ->
# A_log, attn_gate -> self_attn.gate_proj), hence the explicit family table.
_QWEN35_BLOCK_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    # GDN (linear attention) layers
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    # MTP block ("nextn") specials — mapped further by _hf_name_for_mtp.
    "nextn.eh_proj.weight": "__mtp__fc.weight",
    "nextn.enorm.weight": "__mtp__pre_fc_norm_embedding.weight",
    "nextn.hnorm.weight": "__mtp__pre_fc_norm_hidden.weight",
    "nextn.shared_head_norm.weight": "__mtp__norm.weight",
    "nextn.embed_tokens.weight": "__mtp__embed_tokens.weight",
    "nextn.shared_head_head.weight": "__mtp__shared_head.head.weight",
}

_TOP_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}


def hf_name_for(gguf_name: str, n_main_layers: int) -> str | None:
    """qwen35 gguf tensor name -> HF checkpoint name (mtp-aware)."""
    if gguf_name in _TOP_MAP:
        return _TOP_MAP[gguf_name]
    m = _BLK.match(gguf_name)
    if m is None:
        return None
    bid, rest = int(m.group(1)), m.group(2)
    mapped = _QWEN35_BLOCK_MAP.get(rest)
    if bid >= n_main_layers:
        # MTP block: HF names live under mtp.*, not model.layers.*.
        j = bid - n_main_layers
        if mapped is not None and mapped.startswith("__mtp__"):
            return "mtp." + mapped[len("__mtp__"):]
        if mapped is not None:
            return f"mtp.layers.{j}.{mapped}"
        return None
    if mapped is None or mapped.startswith("__mtp__"):
        return None
    return f"model.layers.{bid}.{mapped}"


def _is_plus_one_norm(hf_name: str) -> bool:
    # Forward: name.endswith("norm.weight") and not linear_attn.norm — but the
    # check runs on the REMAPPED name (after mtp.* -> nextn renames), so
    # mtp.pre_fc_norm_embedding/hidden (remapped 'enorm'/'hnorm') were folded
    # too even though their HF names do not end in norm.weight.
    if hf_name in (
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    ):
        return True
    return hf_name.endswith("norm.weight") and not hf_name.endswith("linear_attn.norm.weight")


def inverse_is_row_identity(hf_name: str, g: GdnGeometry) -> bool:
    """True when invert_tensor() is an exact no-op for this tensor.

    Drives the Q8_0 repack (design/serve-engine-plan.md, PATCHES.md #14): a
    quantized tensor may skip the dequant bridge only if its bridge-side
    inverse transform is the identity. A_log/dt_bias/conv are value or shape
    transforms regardless of geometry (F32 in real GGUFs, listed for honesty);
    the V reorders and the out_proj column reorder no-op only for symmetric
    (non-reordered) GDN geometries like Qwen3.5-0.8B.
    """
    if _is_plus_one_norm(hf_name):
        return False
    if hf_name.endswith(
        ("linear_attn.A_log", "linear_attn.dt_bias", "linear_attn.conv1d.weight")
    ):
        return False
    if not g.reordered:
        return True
    return not hf_name.endswith(
        (
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.out_proj.weight",
        )
    )


def inverse_row_permutation(hf_name: str, g: GdnGeometry, rows: int) -> torch.Tensor | None:
    """The inverse as a row map, when it is one, so the rows can be gathered from the file.

    `invert_tensor` is three different things depending on the tensor: a value transform for
    A_log and the plus-one norms, a shape change for conv1d, and for the projections a pure
    permutation of rows. Only the last is expressible as a gather, and only that one lets a
    quantised weight be read from the GGUF instead of dequantised and rebuilt. Returns None for
    everything else, including when the geometry is symmetric and the inverse is the identity --
    the caller already has a cheaper answer for that.
    """
    if not g.reordered:
        return None
    if hf_name.endswith("linear_attn.in_proj_qkv.weight"):
        # [q | k | v]; only the V segment moved.
        qk = g.head_k_dim * g.num_k_heads * 2
        if rows <= qk:
            return None
        index = torch.arange(rows, dtype=torch.long)
        index[qk:] = qk + _inverse_perm(g, g.head_v_dim)
        return index
    if hf_name.endswith("linear_attn.in_proj_z.weight"):
        return _inverse_perm(g, g.head_v_dim)
    if hf_name.endswith(("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight")):
        return _inverse_perm(g, 1)
    return None


def inverse_column_group_map(hf_name: str, g: GdnGeometry, group: int) -> torch.Tensor | None:
    """The inverse as a map over column groups, when it is a column permutation.

    Only `out_proj` has one: llama.cpp reorders the V heads it reads, which moves whole columns
    rather than rows. Runs describe rows, so this cannot be a gather — but the heads are
    `head_v_dim` wide and that divides the quantisation group, so the permutation moves whole
    groups and the loader can carry it as one small map. Returns None when there is nothing to
    do, or when the heads do not divide the group and the permutation is not expressible.
    """
    if not g.reordered or not hf_name.endswith("linear_attn.out_proj.weight"):
        return None
    if g.head_v_dim % group:
        return None
    columns = _inverse_perm(g, g.head_v_dim)
    per_head = g.head_v_dim // group
    return columns.reshape(-1, group)[:, 0] // group if per_head else None


def invert_tensor(hf_name: str, t: torch.Tensor, g: GdnGeometry) -> torch.Tensor:
    """Undo every llama.cpp numeric/layout transform for one HF-named tensor."""
    t = t.float() if t.dtype not in (torch.float32, torch.float64) else t

    if hf_name.endswith("linear_attn.A_log"):
        # forward stored -exp(A_log), tiled element order
        t = invert_v_rows(t.reshape(-1, 1), g, 1).reshape(t.shape)
        return torch.log(-t)
    if hf_name.endswith("linear_attn.dt_bias"):
        return invert_v_rows(t.reshape(-1, 1), g, 1).reshape(t.shape)
    if hf_name.endswith("linear_attn.in_proj_qkv.weight"):
        return invert_qkv_v_segment(t, g)
    if hf_name.endswith("linear_attn.in_proj_z.weight"):
        return invert_v_rows(t, g, g.head_v_dim)
    if hf_name.endswith(("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight")):
        return invert_v_rows(t, g, 1)
    if hf_name.endswith("linear_attn.conv1d.weight"):
        return invert_conv1d(t, g)
    if hf_name.endswith("linear_attn.out_proj.weight"):
        return invert_v_cols(t, g, g.head_v_dim)
    if _is_plus_one_norm(hf_name):
        return t - 1.0
    return t
