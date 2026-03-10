# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# Chunk gated delta rule Triton kernels — mirrors FLA's algorithm
# (flash-linear-attention) specialized for training on SM89+.
#
# Layout: [B, T, H, K] for q/k, [B, T, H, V] for v, [B, T, H] for g/beta.
# Chunk size BT=64 throughout. Scalar gating (no per-key gk).
#
# Forward pipeline:
#   g_cum  = chunk_local_cumsum(g)
#   A      = beta * K @ K^T * exp(g_diff)         [causal, strict lower tri]
#   A_inv  = (I + A)^{-1}                          [TMA-accelerated on SM90+]
#   w, u   = A_inv @ (k*beta*exp(g)), A_inv @ (v*beta)
#   h, v_new, ht = state_recurrence(k, w, u, g_cum, h0)
#   o      = scale * (q @ h + causal_attn @ v_new)
#
# Backward pipeline:
#   dv     = causal_dv(q, k, g, do, scale)
#   dh, dh0, dv = bwd_state_recurrence(q, k, w, g, do, dv, dht, scale)
#   dq, dk, dw, dg = bwd_output(q, k, v_new, w, g, h, dh, do, dv, scale)
#   dk2, dv, db, dg2 = bwd_wy(k, v, beta, g, A_inv, dw, du)
#   dk += dk2; dg += dg2
#   dg = reverse_cumsum(dg)
#
# Each kernel is compiled AOT via compile_gated_delta_rule(), which runs
# autotuning on the target GPU and produces cubin + JSON manifests for the
# C++ JitKernel loader.

from __future__ import annotations

import logging
from pathlib import Path

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TMA / Blackwell detection (checked at import time = compile time)
# ---------------------------------------------------------------------------

_SM = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
_IS_TMA_SUPPORTED = _SM[0] >= 9
_IS_BLACKWELL = _SM[0] >= 10

if hasattr(tl, 'make_tensor_descriptor'):
    _make_td = tl.make_tensor_descriptor
elif hasattr(tl, '_experimental_make_tensor_descriptor'):
    _make_td = tl._experimental_make_tensor_descriptor
else:
    _make_td = None

# Blackwell safe_dot workaround (triton-lang/triton#8695)
if _IS_BLACKWELL:
    @triton.jit
    def safe_dot(a, b):
        return tl.inline_asm_elementwise(
            asm="mov.f32 $0, $1;",
            constraints="=r,r",
            args=[tl.dot(a, b)],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
else:
    @triton.jit
    def safe_dot(a, b):
        return tl.dot(a, b)


# =========================================================================
# Kernel 0: L2 normalization (per-head, last dim)
# =========================================================================

@triton.jit(do_not_specialize=['T'])
def l2norm_fwd_kernel(
    x, y, rstd, T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    EPS: tl.constexpr,
):
    """L2 normalize each (b, t, h) vector of D elements."""
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    o_bt = i_t * BT + tl.arange(0, BT)
    m_bt = o_bt < T

    # Compute squared norm
    acc = tl.zeros([BT], dtype=tl.float32)
    for i_d in range(tl.cdiv(D, 64)):
        p_x = tl.make_block_ptr(x + (bos * H + i_h) * D, (T, D), (H * D, 1),
                                (i_t * BT, i_d * 64), (BT, 64), (1, 0))
        b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
        acc += tl.sum(b_x * b_x, axis=1)

    # rstd = 1/max(||x||, eps)
    b_rstd = 1.0 / tl.sqrt(tl.maximum(acc, EPS))
    p_rstd = tl.make_block_ptr(rstd + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))

    # Normalize and store
    for i_d in range(tl.cdiv(D, 64)):
        p_x = tl.make_block_ptr(x + (bos * H + i_h) * D, (T, D), (H * D, 1),
                                (i_t * BT, i_d * 64), (BT, 64), (1, 0))
        p_y = tl.make_block_ptr(y + (bos * H + i_h) * D, (T, D), (H * D, 1),
                                (i_t * BT, i_d * 64), (BT, 64), (1, 0))
        b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
        b_y = (b_x * b_rstd[:, None]).to(p_y.dtype.element_ty)
        tl.store(p_y, b_y, boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def l2norm_bwd_kernel(
    x_norm, rstd, dout, dx, T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
):
    """Backward through L2 normalization: dx = rstd * (dout - x_norm * dot(dout, x_norm))."""
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    # Load rstd
    p_rstd = tl.make_block_ptr(rstd + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_rstd = tl.load(p_rstd, boundary_check=(0,)).to(tl.float32)

    # Compute dot(dout, x_norm) per token
    dot_sum = tl.zeros([BT], dtype=tl.float32)
    for i_d in range(tl.cdiv(D, 64)):
        p_xn = tl.make_block_ptr(x_norm + (bos * H + i_h) * D, (T, D), (H * D, 1),
                                 (i_t * BT, i_d * 64), (BT, 64), (1, 0))
        p_do = tl.make_block_ptr(dout + (bos * H + i_h) * D, (T, D), (H * D, 1),
                                 (i_t * BT, i_d * 64), (BT, 64), (1, 0))
        b_xn = tl.load(p_xn, boundary_check=(0, 1)).to(tl.float32)
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
        dot_sum += tl.sum(b_xn * b_do, axis=1)

    # dx = rstd * (dout - x_norm * dot_sum)
    for i_d in range(tl.cdiv(D, 64)):
        p_xn = tl.make_block_ptr(x_norm + (bos * H + i_h) * D, (T, D), (H * D, 1),
                                 (i_t * BT, i_d * 64), (BT, 64), (1, 0))
        p_do = tl.make_block_ptr(dout + (bos * H + i_h) * D, (T, D), (H * D, 1),
                                 (i_t * BT, i_d * 64), (BT, 64), (1, 0))
        p_dx = tl.make_block_ptr(dx + (bos * H + i_h) * D, (T, D), (H * D, 1),
                                 (i_t * BT, i_d * 64), (BT, 64), (1, 0))
        b_xn = tl.load(p_xn, boundary_check=(0, 1)).to(tl.float32)
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
        b_dx = b_rstd[:, None] * (b_do - b_xn * dot_sum[:, None])
        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))


# =========================================================================
# Kernel 1: chunk_local_cumsum
# =========================================================================

@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_kernel(
    s, o, T,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))

    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


# =========================================================================
# Kernel 2: chunk_scaled_dot_kkt_fwd
# =========================================================================

@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel(
    k, g, beta, A, T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_b = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1),
                                (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, tl.trans(b_k))

    p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_A *= tl.exp(b_g[:, None] - b_g[None, :])
    b_A *= b_b[:, None]

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (BT * H, 1),
                            (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


# =========================================================================
# Kernel 3: solve_tril (merge 16x16 → 64x64 inverse) — TMA support
# =========================================================================

@triton.jit(do_not_specialize=['T'])
def solve_tril_64x64_kernel(
    A, Ai, T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    # Load 4 diagonal blocks from A
    if not USE_TMA:
        p_A_11 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
        p_A_22 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
        p_A_33 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
        p_A_44 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
        b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_33 = tl.load(p_A_33, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_44 = tl.load(p_A_44, boundary_check=(0, 1)).to(tl.float32)
    else:
        desc = _make_td(A, [T, BT], [H * BT, 1], [16, 16])
        desc_o = _make_td(Ai, [T, BT], [H * BT, 1], [16, 16])
        b_Ai_11 = desc.load([i_t * BT + 0, 0]).to(tl.float32)
        b_Ai_22 = desc.load([i_t * BT + 16, 16]).to(tl.float32)
        b_Ai_33 = desc.load([i_t * BT + 32, 32]).to(tl.float32)
        b_Ai_44 = desc.load([i_t * BT + 48, 48]).to(tl.float32)

    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)

    # Scalar solve for each 16x16 diagonal block
    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H * BT + o_i)
        b_a_11 = tl.where(o_i < i, b_a_11, 0.)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)
        b_a_22 = tl.where(o_i < i - 16, b_a_22, 0.)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a_33 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 32)
        b_a_33 = tl.where(o_i < i - 32, b_a_33, 0.)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a_44 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 48)
        b_a_44 = tl.where(o_i < i - 48, b_a_44, 0.)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)

    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    # Load 6 off-diagonal blocks
    if not USE_TMA:
        p_A_21 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
        p_A_31 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
        p_A_32 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
        p_A_41 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
        p_A_42 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
        p_A_43 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
        b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
        b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
        b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
        b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
        b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
        b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_A_21 = desc.load([i_t * BT + 16, 0]).to(tl.float32)
        b_A_31 = desc.load([i_t * BT + 32, 0]).to(tl.float32)
        b_A_32 = desc.load([i_t * BT + 32, 16]).to(tl.float32)
        b_A_41 = desc.load([i_t * BT + 48, 0]).to(tl.float32)
        b_A_42 = desc.load([i_t * BT + 48, 16]).to(tl.float32)
        b_A_43 = desc.load([i_t * BT + 48, 32]).to(tl.float32)

    # Compose off-diagonal inverse blocks
    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION),
                       b_Ai_11, input_precision=DOT_PRECISION)
    b_Ai_32 = -tl.dot(tl.dot(b_Ai_33, b_A_32, input_precision=DOT_PRECISION),
                       b_Ai_22, input_precision=DOT_PRECISION)
    b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43, input_precision=DOT_PRECISION),
                       b_Ai_33, input_precision=DOT_PRECISION)

    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION) +
        tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION) +
        tl.dot(b_A_43, b_Ai_32, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION) +
        tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION) +
        tl.dot(b_A_43, b_Ai_31, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )

    # Store all 10 blocks of the inverse
    if not USE_TMA:
        p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
        p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
        p_Ai_33 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
        p_Ai_44 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
        p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
        p_Ai_31 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
        p_Ai_32 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
        p_Ai_41 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
        p_Ai_42 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
        p_Ai_43 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
        tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_33, b_Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_44, b_Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_31, b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_32, b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_41, b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_42, b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_43, b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    else:
        desc_o.store([i_t * BT + 0, 0], b_Ai_11.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 16, 16], b_Ai_22.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 32, 32], b_Ai_33.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 48], b_Ai_44.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 16, 0], b_Ai_21.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 32, 0], b_Ai_31.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 32, 16], b_Ai_32.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 0], b_Ai_41.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 16], b_Ai_42.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 32], b_Ai_43.to(desc_o.dtype, fp_downcast_rounding="rtne"))


# =========================================================================
# Kernel 4: recompute_w_u_fwd  (WY representation forward)
# =========================================================================

@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kernel(
    k, v, beta, w, u, A, g, T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    p_b = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1),
                            (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    # u = A_inv @ (v * beta)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1),
                                (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + (bos * H + i_h) * V, (T, V), (H * V, 1),
                                (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    # w = A_inv @ (k * beta * exp(g))
    p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1),
                                (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_w = tl.make_block_ptr(w + (bos * H + i_h) * K, (T, K), (H * K, 1),
                                (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_b[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


# =========================================================================
# Kernel 5: chunk_fwd_h  (state recurrence, multi-block K up to 256)
# =========================================================================

@triton.jit(do_not_specialize=['T'])
def chunk_fwd_h_kernel(
    k, v, w, v_new, g, h, h0, ht, T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    bos = i_n * T
    NT = tl.cdiv(T, BT)
    boh = i_n * NT

    # State accumulators [64, BV]
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # Offset pointers
    h += (boh * H + i_h).to(tl.int64) * K * V
    v += (bos * H + i_h).to(tl.int64) * V
    k += (bos * H + i_h).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    v_new += (bos * H + i_h).to(tl.int64) * V
    h0 = h0 + i_nh * K * V
    ht = ht + i_nh * K * V

    # Load initial state
    p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
    b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
    if K > 64:
        p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
        b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
    if K > 128:
        p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
        b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
    if K > 192:
        p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
        b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        # Store h checkpoint
        p_h1 = tl.make_block_ptr(h + i_t * H * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(h + i_t * H * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(h + i_t * H * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(h + i_t * H * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        # v_new = v - w @ h_prev
        p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))

        p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        # Store v_new
        p_vn = tl.make_block_ptr(v_new, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_vn, b_v.to(p_vn.dtype.element_ty), boundary_check=(0, 1))

        # Gate decay
        last_idx = min((i_t + 1) * BT, T) - 1
        m_t = (i_t * BT + tl.arange(0, BT)) < T
        b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_v = b_v * tl.where(m_t, tl.exp(b_g_last - b_g), 0)[:, None]
        b_g_last = tl.exp(b_g_last)
        b_h1 *= b_g_last
        if K > 64:
            b_h2 *= b_g_last
        if K > 128:
            b_h3 *= b_g_last
        if K > 192:
            b_h4 *= b_g_last

        # State update: h += k^T @ v_new
        b_v = b_v.to(k.dtype.element_ty)
        p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_v)

    # Store final state
    p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
    tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
    if K > 64:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
    if K > 128:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
    if K > 192:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


# =========================================================================
# Kernel 6: chunk_fwd_o  (output computation)
# =========================================================================

@triton.jit(do_not_specialize=['T'])
def chunk_fwd_o_kernel(
    q, k, v, h, g, o, scale, T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    NT = tl.cdiv(T, BT)
    i_tg = i_b * NT + i_t
    bos = i_b * T

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h)
        b_A += tl.dot(b_q, b_k)

    # Gate modulation
    g += bos * H + i_h
    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_o = b_o * tl.exp(b_g)[:, None]
    b_A = b_A * tl.exp(b_g[:, None] - b_g[None, :])

    # Causal mask (lower triangular including diagonal)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


# =========================================================================
# Kernel 7: chunk_bwd_dv_local  (local dv computation)
# =========================================================================

@triton.jit(do_not_specialize=['T'])
def chunk_bwd_dv_local_kernel(
    q, k, g, do, dv, scale, T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    do += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V

    g += bos * H + i_h
    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))

    # Compute transposed causal attention: A^T = (K @ Q^T) * scale * exp(g)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_q = tl.make_block_ptr(q, (K, T), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_A += tl.dot(b_k, b_q) * scale

    b_A *= tl.exp(b_g[None, :] - b_g[:, None])

    # Upper triangular mask (transposed causal)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0).to(do.dtype.element_ty)

    # dv = A^T @ do
    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


# =========================================================================
# Kernel 8: chunk_bwd_dhu  (backward state recurrence)
# =========================================================================

@triton.jit(do_not_specialize=['T'])
def chunk_bwd_dhu_kernel(
    q, k, w, g, dht, dh0, do, dh, dv, dv2, scale, T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    bos = i_n * T
    NT = tl.cdiv(T, BT)
    boh = i_n * NT

    # Gradient accumulators [64, BV]
    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    # Offset pointers
    q += (bos * H + i_h).to(tl.int64) * K
    k += (bos * H + i_h).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    do += (bos * H + i_h).to(tl.int64) * V
    dv += (bos * H + i_h).to(tl.int64) * V
    dv2 += (bos * H + i_h).to(tl.int64) * V
    dh += (boh * H + i_h).to(tl.int64) * K * V
    dh0 = dh0 + i_nh * K * V
    dht = dht + i_nh * K * V

    # Load final state gradient
    p_dht1 = tl.make_block_ptr(dht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
    b_dh1 += tl.load(p_dht1, boundary_check=(0, 1))
    if K > 64:
        p_dht2 = tl.make_block_ptr(dht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
        b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
    if K > 128:
        p_dht3 = tl.make_block_ptr(dht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
        b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
    if K > 192:
        p_dht4 = tl.make_block_ptr(dht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
        b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))

    for i_t in range(NT - 1, -1, -1):
        # Store dh checkpoint
        p_dh1 = tl.make_block_ptr(dh + i_t * H * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_dh2 = tl.make_block_ptr(dh + i_t * H * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_dh3 = tl.make_block_ptr(dh + i_t * H * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_dh4 = tl.make_block_ptr(dh + i_t * H * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        bg_last = tl.load(g + (bos + last_idx) * H + i_h)
        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        bg_last_exp = tl.exp(bg_last)
        b_g_exp = tl.exp(b_g)

        # Update dv: dv += k @ dh (inter-chunk contribution)
        p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv2 = tl.make_block_ptr(dv2, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))

        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))
        if K > 64:
            p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))
        if K > 128:
            p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))
        if K > 192:
            p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

        m_t = (i_t * BT + tl.arange(0, BT)) < T
        b_dv *= tl.where(m_t, tl.exp(bg_last - b_g), 0)[:, None]
        b_dv += tl.load(p_dv, boundary_check=(0, 1))
        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

        # Update dh: dh = dh * exp(g_last) + q^T @ do * scale - w^T @ dv
        p_w = tl.make_block_ptr(w, (K, T), (1, H * K), (0, i_t * BT), (64, BT), (0, 1))
        p_q = tl.make_block_ptr(q, (K, T), (1, H * K), (0, i_t * BT), (64, BT), (0, 1))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * b_g_exp[None, :]
        b_dh1 *= bg_last_exp
        b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

        if K > 64:
            p_q = tl.make_block_ptr(q, (K, T), (1, H * K), (64, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, H * K), (64, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_q = b_q * b_g_exp[None, :]
            b_dh2 *= bg_last_exp
            b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 128:
            p_q = tl.make_block_ptr(q, (K, T), (1, H * K), (128, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, H * K), (128, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_q = b_q * b_g_exp[None, :]
            b_dh3 *= bg_last_exp
            b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 192:
            p_q = tl.make_block_ptr(q, (K, T), (1, H * K), (192, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, H * K), (192, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_q = b_q * b_g_exp[None, :]
            b_dh4 *= bg_last_exp
            b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

    # Store dh0
    p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
    tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
    if K > 64:
        p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh0, b_dh2.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
    if K > 128:
        p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh0, b_dh3.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
    if K > 192:
        p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh0, b_dh4.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


# =========================================================================
# Kernel 9: chunk_bwd_dqkwg  (backward for dq, dk, dw, dg)
# =========================================================================

@triton.jit(do_not_specialize=['T'])
def chunk_bwd_dqkwg_kernel(
    q, k, v, g, h, do, dh, dq, dk, dw, dv, dg,
    scale, B, T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    NT = tl.cdiv(T, BT)
    i_tg = i_b * NT + i_t
    bos = i_b * T
    all = B * T

    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V
    dh += (i_tg * H + i_h).to(tl.int64) * K * V
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dw += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V

    # dg is [NK, B, T, H] — offset to this K-block's slice
    dg += i_k * all * H

    b_dg_last = tl.zeros([1], dtype=tl.float32)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh_v = tl.load(p_dh, boundary_check=(0, 1))
        b_dg_last += tl.sum(b_h * b_dh_v)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        b_dk += tl.dot(b_v, b_dh_v.to(b_v.dtype))
        # dw = -sum_v(dv @ h^T) per K-block
        p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_dv_v = tl.load(p_dv, boundary_check=(0, 1))
        b_dw += tl.dot(b_dv_v.to(b_v.dtype), b_h.to(b_v.dtype))

    p_dw = tl.make_block_ptr(dw, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dw, (-b_dw).to(p_dw.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()

    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))

    p_dq = tl.make_block_ptr(dq, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)

    b_dg = tl.zeros([BT], dtype=tl.float32)
    g += bos * H + i_h
    dg += bos * H + i_h
    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * H)
    b_dg_last *= tl.exp(b_g_last)

    b_dq = b_dq * tl.exp(b_g)[:, None] * scale
    b_dg += tl.sum(b_dq * b_q, axis=1)

    b_dk = b_dk * tl.where(m_t, tl.exp(-b_g + b_g_last), 0)[:, None]
    b_dg -= tl.sum(b_k * b_dk, axis=1)
    b_dg_last += tl.sum(b_dk * b_k)

    b_ds = tl.where(m_A, b_ds * tl.exp(b_g[:, None] - b_g[None, :]), 0) * scale
    b_ds2 = b_ds * tl.dot(b_q, tl.trans(b_k))
    b_dg += tl.sum(b_ds2, axis=1)
    b_dg -= tl.sum(b_ds2, axis=0)

    b_ds = b_ds.to(b_k.dtype)
    b_dq += tl.dot(b_ds, b_k)
    b_dk += tl.dot(tl.trans(b_ds), b_q)

    p_dg = tl.make_block_ptr(dg, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last)
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


# =========================================================================
# Kernel 10: prepare_wy_repr_bwd  (backward through WY representation)
# =========================================================================

@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_kernel(
    k, v, beta, g, A, dw, du, dk, dv, db, dg, T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    p_b = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_db = tl.make_block_ptr(db + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # A loaded transposed
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (BT, T), (1, H * BT),
                            (0, i_t * BT), (BT, BT), (0, 1))

    b_b = tl.load(p_b, boundary_check=(0,))
    b_db = tl.zeros([BT], dtype=tl.float32)
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)

    p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_g_exp = tl.exp(b_g)
    b_dg = tl.zeros([BT], dtype=tl.float32)

    # Phase 1: accumulate dA from dw and du, compute dk (first pass)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1),
                                (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + (bos * H + i_h) * K, (T, K), (H * K, 1),
                                 (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dw = tl.make_block_ptr(dw + (bos * H + i_h) * K, (T, K), (H * K, 1),
                                 (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kbg = b_k * (b_b * b_g_exp)[:, None]
        b_dw = tl.load(p_dw, boundary_check=(0, 1))

        b_dA += tl.dot(b_dw, tl.trans(b_kbg).to(b_dw.dtype))
        b_dkbg = tl.dot(b_A, b_dw)
        b_dk_val = b_dkbg * (b_g_exp * b_b)[:, None]
        b_db += tl.sum(b_dkbg * b_k * b_g_exp[:, None], 1)
        b_dg += tl.sum(b_dkbg * b_kbg, 1)
        # Load existing dk (from bwd_dqkwg) and add our contribution
        b_dk_val += tl.load(p_dk, boundary_check=(0, 1))
        tl.store(p_dk, b_dk_val.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1),
                                (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V, (T, V), (H * V, 1),
                                 (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_du = tl.make_block_ptr(du + (bos * H + i_h) * V, (T, V), (H * V, 1),
                                 (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_du = tl.load(p_du, boundary_check=(0, 1))
        b_dA += tl.dot(b_du, tl.trans(b_vb))
        b_dvb = tl.dot(b_A, b_du)
        b_dv_val = b_dvb * b_b[:, None]
        b_db += tl.sum(b_dvb * b_v, 1)
        tl.store(p_dv, b_dv_val.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    # Phase 2: gradient through the inverse
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA, 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))

    b_dA *= tl.exp(b_g[:, None] - b_g[None, :])

    b_A_recon = tl.zeros([BT, BT], dtype=tl.float32)
    b_dA = tl.where(m_A, -b_dA, 0).to(k.dtype.element_ty)

    tl.debug_barrier()

    # Phase 3: gradient through kkt
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1),
                                (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + (bos * H + i_h) * K, (T, K), (H * K, 1),
                                 (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kt = tl.trans(b_k)
        b_ktb = b_kt * b_b[None, :]

        b_A_recon += tl.dot(b_k, b_kt)
        b_dkb = tl.dot(b_dA, b_k)
        b_db += tl.sum(b_dkb * b_k, 1)
        b_dk_val = b_dkb * b_b[:, None] + tl.trans(tl.dot(b_ktb.to(b_dA.dtype), b_dA))
        b_dk_val += tl.load(p_dk, boundary_check=(0, 1))
        tl.store(p_dk, b_dk_val.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))

    b_A_recon *= b_b[:, None]
    b_AdA = b_dA * b_A_recon
    p_dg = tl.make_block_ptr(dg + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_dg += tl.sum(b_AdA, axis=1) - tl.sum(b_AdA, axis=0)
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


# =========================================================================
# Compilation
# =========================================================================

def compile_gated_delta_rule(
    H: int,
    K: int,
    V: int,
    output_dir: str = ".",
    sm: int | None = None,
    bench_B: int = 2,
    bench_T: int = 2048,
) -> dict[str, str]:
    """Compile all gated delta rule sub-kernels with AOT autotuning.

    Args:
        H: Number of attention heads.
        K: Key dimension per head.
        V: Value dimension per head.
        output_dir: Directory for cubin + JSON manifests.
        sm: Target SM version (auto-detected if None).
        bench_B: Batch size for autotuning benchmarks.
        bench_T: Sequence length for autotuning benchmarks.

    Returns:
        Dict mapping kernel names to manifest file paths.
    """
    from surogate.kernels.compiler import autotune_triton_kernel

    BT = 64
    NT = bench_T // BT
    B = bench_B
    T = bench_T
    # Disable TMA for AOT compilation: the solve_tril TMA path requires Triton's
    # global scratch allocator which isn't available when launching via cuLaunchKernel.
    use_tma = False

    manifests = {}

    def _compile(name, fn, signature, constants, configs, bench_args, grid):
        logger.info("Autotuning %s (H=%d K=%d V=%d)...", name, H, K, V)
        path = autotune_triton_kernel(
            fn=fn, signature=signature, constants=constants,
            configs=configs, bench_args=bench_args, grid=grid,
            output_dir=output_dir, kernel_name=name, sm=sm,
        )
        manifests[name] = path

    # --- 0a. L2 norm forward (for Q) ---
    q_in = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    q_out = torch.empty_like(q_in)
    q_rstd = torch.empty(B, T, H, dtype=torch.float32, device="cuda")
    _compile("gdr_l2norm_fwd_q", l2norm_fwd_kernel,
             signature={"x": "*bf16", "y": "*bf16", "rstd": "*fp32", "T": "i32"},
             constants={"H": H, "D": K, "BT": BT, "EPS": 1e-12},
             configs=[{"num_warps": w} for w in [1, 2, 4, 8]],
             bench_args=(q_in, q_out, q_rstd, T), grid=(NT, B * H))

    # --- 0b. L2 norm backward (for Q) ---
    dq = torch.randn_like(q_in)
    _compile("gdr_l2norm_bwd_q", l2norm_bwd_kernel,
             signature={"x_norm": "*bf16", "rstd": "*fp32", "dout": "*bf16", "dx": "*bf16", "T": "i32"},
             constants={"H": H, "D": K, "BT": BT},
             configs=[{"num_warps": w} for w in [1, 2, 4, 8]],
             bench_args=(q_out, q_rstd, dq, torch.empty_like(dq), T), grid=(NT, B * H))

    # --- 0c. L2 norm forward (for V-dim, if V != K) ---
    if V != K:
        v_in = torch.randn(B, T, H, V, dtype=torch.bfloat16, device="cuda")
        v_out = torch.empty_like(v_in)
        v_rstd = torch.empty(B, T, H, dtype=torch.float32, device="cuda")
        _compile("gdr_l2norm_fwd_v", l2norm_fwd_kernel,
                 signature={"x": "*bf16", "y": "*bf16", "rstd": "*fp32", "T": "i32"},
                 constants={"H": H, "D": V, "BT": BT, "EPS": 1e-12},
                 configs=[{"num_warps": w} for w in [1, 2, 4, 8]],
                 bench_args=(v_in, v_out, v_rstd, T), grid=(NT, B * H))
        _compile("gdr_l2norm_bwd_v", l2norm_bwd_kernel,
                 signature={"x_norm": "*bf16", "rstd": "*fp32", "dout": "*bf16", "dx": "*bf16", "T": "i32"},
                 constants={"H": H, "D": V, "BT": BT},
                 configs=[{"num_warps": w} for w in [1, 2, 4, 8]],
                 bench_args=(v_out, v_rstd, torch.randn_like(v_in), torch.empty_like(v_in), T),
                 grid=(NT, B * H))

    # --- 1. Cumsum forward ---
    # g_input is FP32 at runtime (output of log_sigmoid)
    s = torch.randn(B, T, H, dtype=torch.float32, device="cuda")
    o = torch.empty(B, T, H, dtype=torch.float32, device="cuda")
    _compile("gdr_cumsum_fwd", chunk_local_cumsum_kernel,
             signature={"s": "*fp32", "o": "*fp32", "T": "i32"},
             constants={"H": H, "BT": BT, "REVERSE": 0},
             configs=[{"num_warps": w} for w in [1, 2, 4, 8]],
             bench_args=(s, o, T), grid=(NT, B * H))

    # --- 2. Cumsum reverse (for backward dg) ---
    _compile("gdr_cumsum_rev", chunk_local_cumsum_kernel,
             signature={"s": "*fp32", "o": "*fp32", "T": "i32"},
             constants={"H": H, "BT": BT, "REVERSE": 1},
             configs=[{"num_warps": w} for w in [1, 2, 4, 8]],
             bench_args=(o, torch.empty_like(o), T), grid=(NT, B * H))

    # --- 3. KKT forward ---
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(B, T, H, dtype=torch.float32, device="cuda")
    beta = torch.randn(B, T, H, dtype=torch.bfloat16, device="cuda")
    A_fp32 = torch.empty(B, T, H, BT, dtype=torch.float32, device="cuda")
    _compile("gdr_kkt_fwd", chunk_scaled_dot_kkt_fwd_kernel,
             signature={"k": "*bf16", "g": "*fp32", "beta": "*bf16", "A": "*fp32", "T": "i32"},
             constants={"H": H, "K": K, "BT": BT},
             configs=[{"BK": bk, "num_warps": w, "num_stages": ns}
                      for bk in [32, 64, 128] if bk <= max(K, 32)
                      for w in [2, 4, 8] for ns in [2, 3, 4]],
             bench_args=(k, g, beta, A_fp32, T), grid=(NT, B * H))

    # --- 4. Solve tril ---
    Ai = torch.zeros(B, T, H, BT, dtype=torch.bfloat16, device="cuda")
    dot_precisions = ["ieee"]
    if use_tma:
        dot_precisions.append("tf32")
    _compile("gdr_solve_tril", solve_tril_64x64_kernel,
             signature={"A": "*fp32", "Ai": "*bf16", "T": "i32"},
             constants={"H": H, "BT": BT, "USE_TMA": int(use_tma)},
             configs=[{"DOT_PRECISION": dp, "num_warps": w, "num_stages": ns}
                      for dp in dot_precisions
                      for w in [2, 4, 8] for ns in [2, 3, 4, 5]],
             bench_args=(A_fp32, Ai, T), grid=(NT, B * H))

    # --- 5. WY forward ---
    v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device="cuda")
    w_t = torch.empty(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    u = torch.empty(B, T, H, V, dtype=torch.bfloat16, device="cuda")
    A_bf16 = Ai  # solve_tril output
    _compile("gdr_wy_fwd", recompute_w_u_fwd_kernel,
             signature={"k": "*bf16", "v": "*bf16", "beta": "*bf16",
                        "w": "*bf16", "u": "*bf16", "A": "*bf16", "g": "*fp32", "T": "i32"},
             constants={"H": H, "K": K, "V": V, "BT": BT, "BK": 64, "BV": 64},
             configs=[{"num_warps": w, "num_stages": ns}
                      for w in [2, 4, 8] for ns in [2, 3, 4]],
             bench_args=(k, v, beta, w_t, u, A_bf16, g, T), grid=(NT, B * H))

    # --- 6. Forward h (state recurrence) ---
    h = torch.empty(B, NT, H, K, V, dtype=torch.bfloat16, device="cuda")
    h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device="cuda")
    ht = torch.empty(B, H, K, V, dtype=torch.float32, device="cuda")
    v_new = torch.empty(B, T, H, V, dtype=torch.bfloat16, device="cuda")
    bv_configs = [bv for bv in [32, 64] if bv <= max(V, 32)]
    min_bv = min(bv_configs)
    _compile("gdr_fwd_h", chunk_fwd_h_kernel,
             signature={"k": "*bf16", "v": "*bf16", "w": "*bf16", "v_new": "*bf16",
                        "g": "*fp32", "h": "*bf16", "h0": "*bf16", "ht": "*fp32", "T": "i32"},
             constants={"H": H, "K": K, "V": V, "BT": BT},
             configs=[{"BV": bv, "num_warps": w, "num_stages": ns}
                      for bv in bv_configs for w in [2, 4] for ns in [2, 3, 4]],
             bench_args=(k, u, w_t, v_new, g, h, h0, ht, T),
             grid=(triton.cdiv(V, min_bv), B * H))

    # --- 7. Forward o (output) ---
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    o_out = torch.empty(B, T, H, V, dtype=torch.bfloat16, device="cuda")
    scale = K ** -0.5
    fwd_o_configs = [
        {"BK": 128, "BV": 128, "num_warps": 8, "num_stages": 3},
        {"BK": 64, "BV": 64, "num_warps": 4, "num_stages": 3},
        {"BK": 32, "BV": 32, "num_warps": 2, "num_stages": 3},
    ]
    fwd_o_configs = [c for c in fwd_o_configs if c["BK"] <= max(K, 32) and c["BV"] <= max(V, 32)]
    min_bv_o = min(c["BV"] for c in fwd_o_configs)
    _compile("gdr_fwd_o", chunk_fwd_o_kernel,
             signature={"q": "*bf16", "k": "*bf16", "v": "*bf16", "h": "*bf16",
                        "g": "*fp32", "o": "*bf16", "scale": "fp32", "T": "i32"},
             constants={"H": H, "K": K, "V": V, "BT": BT},
             configs=fwd_o_configs,
             bench_args=(q, k, v_new, h, g, o_out, scale, T),
             grid=(triton.cdiv(V, min_bv_o), NT, B * H))

    # --- 8. Backward dv_local ---
    do_t = torch.randn(B, T, H, V, dtype=torch.bfloat16, device="cuda")
    dv_out = torch.empty(B, T, H, V, dtype=torch.bfloat16, device="cuda")
    CONST_TILING = 64
    BK_bwd = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV_bwd = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    _compile("gdr_bwd_dv_local", chunk_bwd_dv_local_kernel,
             signature={"q": "*bf16", "k": "*bf16", "g": "*fp32",
                        "do": "*bf16", "dv": "*bf16", "scale": "fp32", "T": "i32"},
             constants={"H": H, "K": K, "V": V, "BT": BT, "BK": BK_bwd, "BV": BV_bwd},
             configs=[{"num_warps": w, "num_stages": ns}
                      for w in [2, 4, 8] for ns in [2, 3, 4]],
             bench_args=(q, k, g, do_t, dv_out, scale, T), grid=(NT, B * H))

    # --- 9. Backward dhu (state recurrence) ---
    dh = torch.empty(B, NT, H, K, V, dtype=torch.bfloat16, device="cuda")
    dh0 = torch.empty(B, H, K, V, dtype=torch.float32, device="cuda")
    dht = torch.randn(B, H, K, V, dtype=torch.float32, device="cuda")
    dv2 = torch.empty(B, T, H, V, dtype=torch.bfloat16, device="cuda")
    _compile("gdr_bwd_dhu", chunk_bwd_dhu_kernel,
             signature={"q": "*bf16", "k": "*bf16", "w": "*bf16", "g": "*fp32",
                        "dht": "*fp32", "dh0": "*fp32", "do": "*bf16",
                        "dh": "*bf16", "dv": "*bf16", "dv2": "*bf16",
                        "scale": "fp32", "T": "i32"},
             constants={"H": H, "K": K, "V": V, "BT": BT},
             configs=[{"BV": bv, "num_warps": w, "num_stages": ns}
                      for bv in bv_configs for w in [2, 4] for ns in [2, 3, 4]],
             bench_args=(q, k, w_t, g, dht, dh0, do_t, dh, dv_out, dv2, scale, T),
             grid=(triton.cdiv(V, min_bv), B * H))

    # --- 10. Backward dqkwg ---
    NK = triton.cdiv(K, BK_bwd)
    dq = torch.empty(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    dk = torch.empty(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    dw = torch.empty(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    dg_nk = torch.empty(NK, B, T, H, dtype=torch.float32, device="cuda")
    _compile("gdr_bwd_dqkwg", chunk_bwd_dqkwg_kernel,
             signature={"q": "*bf16", "k": "*bf16", "v": "*bf16", "g": "*fp32",
                        "h": "*bf16", "do": "*bf16", "dh": "*bf16",
                        "dq": "*bf16", "dk": "*bf16", "dw": "*bf16",
                        "dv": "*bf16", "dg": "*fp32",
                        "scale": "fp32", "B": "i32", "T": "i32"},
             constants={"H": H, "K": K, "V": V, "BT": BT, "BK": BK_bwd, "BV": BV_bwd},
             configs=[{"num_warps": w, "num_stages": ns}
                      for w in [2, 4, 8] for ns in [2, 3, 4]],
             bench_args=(q, k, v_new, g, h, do_t, dh, dq, dk, dw, dv2, dg_nk, scale, B, T),
             grid=(NK, NT, B * H))

    # --- 11. Backward WY repr ---
    db = torch.empty(B, T, H, dtype=torch.bfloat16, device="cuda")
    dg_wy = torch.empty(B, T, H, dtype=torch.float32, device="cuda")
    _compile("gdr_bwd_wy", prepare_wy_repr_bwd_kernel,
             signature={"k": "*bf16", "v": "*bf16", "beta": "*bf16", "g": "*fp32",
                        "A": "*bf16", "dw": "*bf16", "du": "*bf16",
                        "dk": "*bf16", "dv": "*bf16", "db": "*bf16", "dg": "*fp32", "T": "i32"},
             constants={"H": H, "K": K, "V": V, "BT": BT, "BK": BK_bwd, "BV": BV_bwd},
             configs=[{"num_warps": w, "num_stages": ns}
                      for w in [2, 4] for ns in [2, 3, 4]],
             bench_args=(k, v, beta, g, A_bf16, dw, dv2, dk, dv_out, db, dg_wy, T),
             grid=(NT, B * H))

    logger.info("Compiled %d gated delta rule kernels to %s", len(manifests), output_dir)
    return manifests
