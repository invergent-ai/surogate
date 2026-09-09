# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Vendored from flash-linear-attention v0.5.2; see README.md and LICENSE.

import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_vector_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    # Surogate: skip unused entries in the capture-stable chunk grid.
    if IS_VARLEN and tl.load(chunk_indices + tl.program_id(1) * 2) < 0:
        return
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    o_s = i_s * BS + tl.arange(0, BS)
    m_s = (o_t[:, None] < T) & (o_s[None, :] < S)
    if HEAD_FIRST:
        p_s = s + (bos * H + i_h*T)*S + o_t[:, None] * S + o_s[None, :]
        p_o = o + (bos * H + i_h*T)*S + o_t[:, None] * S + o_s[None, :]
    else:
        p_s = s + (bos * H + i_h) * S + o_t[:, None] * (H*S) + o_s[None, :]
        p_o = o + (bos * H + i_h) * S + o_t[:, None] * (H*S) + o_s[None, :]
    # [BT, BS]
    b_s = tl.load(p_s, mask=m_s, other=0.0).to(tl.float32)
    if REVERSE:
        b_o = tl.cumsum(b_s, axis=0, reverse=True)
    else:
        b_o = tl.cumsum(b_s, axis=0)
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_s)
