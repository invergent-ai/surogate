#!/usr/bin/env python3
"""Generate golden data for DSL compiled ops.

Usage examples:
  .venv/bin/python tests/ops/generate_goldens.py --list
  .venv/bin/python tests/ops/generate_goldens.py --op matmul_swiglu
  .venv/bin/python tests/ops/generate_goldens.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

try:
    import torch
except Exception:
    torch = None


@dataclass
class GoldenCase:
    op: str
    case: str
    payload: Dict


def _dtype_name(arr: np.ndarray) -> str:
    dt = np.dtype(arr.dtype)
    if dt == np.float64:
        return "fp64"
    if dt == np.float32:
        return "fp32"
    if dt == np.int64:
        return "int64"
    if dt == np.int32:
        return "int32"
    return str(dt)


def _tensor_payload(arr: np.ndarray, dtype: str | None = None) -> Dict:
    arr = np.asarray(arr)
    return {
        "shape": list(arr.shape),
        "dtype": dtype or _dtype_name(arr),
        "data": arr.reshape(-1).tolist(),
    }


def _write_case(out_dir: str, case: GoldenCase) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{case.op}_{case.case}.json"
    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(case.payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _maybe_check_torch(
    A: np.ndarray,
    W: np.ndarray,
    DOUT: np.ndarray,
    out: np.ndarray,
    d_up: np.ndarray,
    dA: np.ndarray,
    dW: np.ndarray,
) -> None:
    if torch is None:
        return
    # Verify against autograd in float64
    A_t = torch.tensor(A, dtype=torch.float64, requires_grad=True)
    W_t = torch.tensor(W, dtype=torch.float64, requires_grad=True)
    up_t = A_t @ W_t
    B, T, N = DOUT.shape[0], DOUT.shape[1], d_up.shape[-1]
    D = DOUT.shape[2]
    up_3d = up_t.reshape(B, T, N)
    x1 = up_3d[..., :D]
    x2 = up_3d[..., D:]
    out_t = x1 * x2 * torch.sigmoid(x2)
    DOUT_t = torch.tensor(DOUT, dtype=torch.float64)
    dA_t, dW_t, dUp_t = torch.autograd.grad(out_t, (A_t, W_t, up_t), grad_outputs=DOUT_t)
    np.testing.assert_allclose(out, out_t.detach().numpy(), rtol=1e-12, atol=1e-12)
    d_up_t = dUp_t.detach().numpy().reshape(B, T, N)
    np.testing.assert_allclose(d_up, d_up_t, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dA, dA_t.detach().numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dW, dW_t.detach().numpy(), rtol=1e-12, atol=1e-12)


def _precompute_freqs_cis(dim: int, end: int, theta: float) -> np.ndarray:
    """Match kernels::precompute_freqs_cis (interleaved cos/sin)."""
    freqs = np.zeros((end, dim), dtype=np.float64)
    for i in range(dim // 2):
        inv_freq = 1.0 / (theta ** ((2 * i) / dim))
        for t in range(end):
            angle = t * inv_freq
            freqs[t, 2 * i + 0] = np.cos(angle)
            freqs[t, 2 * i + 1] = np.sin(angle)
    return freqs


def _precompute_freqs_cis_complex(dim: int, end: int, theta: float) -> np.ndarray:
    """Compute RoPE frequencies in 3D complex format [T, D//2, 2] for DSL."""
    freqs = np.zeros((end, dim // 2, 2), dtype=np.float64)
    for i in range(dim // 2):
        inv_freq = 1.0 / (theta ** ((2 * i) / dim))
        for t in range(end):
            angle = t * inv_freq
            freqs[t, i, 0] = np.cos(angle)  # real part
            freqs[t, i, 1] = np.sin(angle)  # imaginary part
    return freqs


def _rope_forward_cpu(qkv: np.ndarray, freqs: np.ndarray, position_ids: np.ndarray | None,
                      Hq: int, Hkv: int, head_dim: int, rotary_dim: int) -> np.ndarray:
    B, T, _ = qkv.shape
    N = Hq + 2 * Hkv
    HD = head_dim
    if rotary_dim == 0:
        rotary_dim = head_dim
    rotary_half = rotary_dim // 2
    hd2 = HD // 2

    out = qkv.copy()
    qkv_view = out.reshape(B, T, N, HD)
    for b in range(B):
        for t in range(T):
            t_pos = int(position_ids[b, t]) if position_ids is not None else t
            freqt = freqs[t_pos]
            for h in range(N):
                qkv_type = 0 if h < Hq else (1 if h < Hq + Hkv else 2)
                if qkv_type == 2:
                    continue  # V: copy-through
                for d in range(hd2):
                    real = qkv_view[b, t, h, d]
                    imag = qkv_view[b, t, h, d + hd2]
                    if d < rotary_half:
                        c = freqt[2 * d + 0]
                        s = freqt[2 * d + 1]
                        qkv_view[b, t, h, d] = real * c - imag * s
                        qkv_view[b, t, h, d + hd2] = real * s + imag * c
                    else:
                        qkv_view[b, t, h, d] = real
                        qkv_view[b, t, h, d + hd2] = imag
    return out


def _rope_backward_cpu(dout: np.ndarray, freqs: np.ndarray, position_ids: np.ndarray | None,
                       Hq: int, Hkv: int, head_dim: int, rotary_dim: int) -> np.ndarray:
    B, T, _ = dout.shape
    N = Hq + 2 * Hkv
    HD = head_dim
    if rotary_dim == 0:
        rotary_dim = head_dim
    rotary_half = rotary_dim // 2
    hd2 = HD // 2

    dinp = dout.copy()
    view = dinp.reshape(B, T, N, HD)
    for b in range(B):
        for t in range(T):
            t_pos = int(position_ids[b, t]) if position_ids is not None else t
            freqt = freqs[t_pos]
            for h in range(N):
                qkv_type = 0 if h < Hq else (1 if h < Hq + Hkv else 2)
                if qkv_type == 2:
                    continue
                for d in range(hd2):
                    real = view[b, t, h, d]
                    imag = view[b, t, h, d + hd2]
                    if d < rotary_half:
                        c = freqt[2 * d + 0]
                        s = -freqt[2 * d + 1]  # inverse rotation
                        view[b, t, h, d] = real * c - imag * s
                        view[b, t, h, d + hd2] = real * s + imag * c
                    else:
                        view[b, t, h, d] = real
                        view[b, t, h, d + hd2] = imag
    return dinp


def _qkv_head_rmsnorm_forward(qkv: np.ndarray, weight: np.ndarray, eps: float,
                              B: int, T: int, H: int, HS: int, channel_offset: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply head-wise RMSNorm to a slice of qkv and return updated qkv + rstd."""
    qkv_out = qkv.copy()
    BT = B * T
    qkv_view = qkv_out.reshape(BT, -1)
    rstd = np.zeros((BT, H), dtype=np.float64)
    for token in range(BT):
        for h in range(H):
            start = channel_offset + h * HS
            end = start + HS
            x = qkv_view[token, start:end]
            mean_sq = np.mean(x * x)
            s = 1.0 / np.sqrt(mean_sq + eps)
            rstd[token, h] = s
            qkv_view[token, start:end] = (x * s) * weight
    return qkv_out, rstd.reshape(B, T, H)


def _qkv_head_rmsnorm_backward_dx(d_qkv: np.ndarray, qkv_out: np.ndarray, weight: np.ndarray, rstd: np.ndarray,
                                  B: int, T: int, H: int, HS: int, channel_offset: int) -> np.ndarray:
    """In-place RMSNorm backward on a qkv slice (matches qkv_head_rmsnorm_backward_dx)."""
    qkv_view = qkv_out.reshape(B * T, -1)
    d_view = d_qkv.reshape(B * T, -1)
    rstd_view = rstd.reshape(B * T, H)

    for token in range(B * T):
        for h in range(H):
            start = channel_offset + h * HS
            end = start + HS
            out = qkv_view[token, start:end]
            dy = d_view[token, start:end]
            s = rstd_view[token, h]
            w = weight
            wdy = w * dy
            # dnorm_norm_mean = mean(dy * out)
            dnorm_norm_mean = np.mean(dy * out)
            x_hat = out / w
            dx = (wdy - x_hat * dnorm_norm_mean) * s
            d_view[token, start:end] = dx
    return d_qkv


def _rmsnorm_backward(dout: np.ndarray, inp: np.ndarray, weight: np.ndarray, rstd: np.ndarray) -> np.ndarray:
    """RMSNorm backward for input gradient (matches rmsnorm_backward_kernel10 formula)."""
    B, T, C = inp.shape
    dx = np.zeros_like(inp)
    for token in range(B * T):
        x = inp.reshape(B * T, C)[token]
        dy = dout.reshape(B * T, C)[token]
        s = rstd.reshape(B * T)[token]
        w = weight
        wdy = w * dy
        mean_wdy_x = np.mean(wdy * x)
        dnorm_norm_mean = mean_wdy_x * s
        x_hat = x * s
        dx_flat = (wdy - x_hat * dnorm_norm_mean) * s
        dx.reshape(B * T, C)[token] = dx_flat
    return dx


def _rmsnorm_dweight(dout: np.ndarray, inp: np.ndarray, rstd: np.ndarray) -> np.ndarray:
    """RMSNorm dweight = sum(dout * x_hat)."""
    B, T, C = inp.shape
    x_hat = inp * rstd.reshape(B, T, 1)
    return np.sum(dout * x_hat, axis=(0, 1))


def _logsumexp(x: np.ndarray) -> float:
    m = np.max(x)
    return float(m + np.log(np.sum(np.exp(x - m))))


def _softmax(x: np.ndarray) -> np.ndarray:
    m = np.max(x)
    e = np.exp(x - m)
    return e / np.sum(e)


def _flash_attention_ref(qkv: np.ndarray, B: int, T: int, Hq: int, Hkv: int, HS: int, causal: bool) -> tuple[np.ndarray, np.ndarray]:
    N = Hq + 2 * Hkv
    qkv4 = qkv.reshape(B, T, N, HS)
    q = qkv4[:, :, :Hq, :]
    k = qkv4[:, :, Hq:Hq + Hkv, :]
    v = qkv4[:, :, Hq + Hkv:, :]

    group = Hq // Hkv
    scale = 1.0 / np.sqrt(HS)
    out = np.zeros((B, T, Hq, HS), dtype=np.float64)
    lse = np.zeros((B, Hq, T), dtype=np.float64)

    for b in range(B):
        for hq in range(Hq):
            hk = hq // group
            for t in range(T):
                t_end = t + 1 if causal else T
                k_slice = k[b, :t_end, hk, :]
                v_slice = v[b, :t_end, hk, :]
                scores = (k_slice @ q[b, t, hq]).astype(np.float64) * scale
                lse_val = _logsumexp(scores)
                lse[b, hq, t] = lse_val
                weights = np.exp(scores - lse_val)
                out[b, t, hq, :] = weights @ v_slice

    return out, lse


def _flash_attention_backward_ref(qkv: np.ndarray, d_out: np.ndarray, B: int, T: int, Hq: int, Hkv: int,
                                  HS: int, causal: bool) -> np.ndarray:
    N = Hq + 2 * Hkv
    qkv4 = qkv.reshape(B, T, N, HS)
    q = qkv4[:, :, :Hq, :]
    k = qkv4[:, :, Hq:Hq + Hkv, :]
    v = qkv4[:, :, Hq + Hkv:, :]

    dq = np.zeros_like(q)
    dk = np.zeros_like(k)
    dv = np.zeros_like(v)

    group = Hq // Hkv
    scale = 1.0 / np.sqrt(HS)

    for b in range(B):
        for hq in range(Hq):
            hk = hq // group
            for t in range(T):
                t_end = t + 1 if causal else T
                q_vec = q[b, t, hq]
                k_slice = k[b, :t_end, hk, :]
                v_slice = v[b, :t_end, hk, :]

                scores = (k_slice @ q_vec) * scale
                p = _softmax(scores)

                d_out_vec = d_out[b, t, hq]
                dP = v_slice @ d_out_vec
                sum_p_dP = float(np.dot(p, dP))
                dS = p * (dP - sum_p_dP)

                dq[b, t, hq] += (dS[:, None] * k_slice).sum(axis=0) * scale
                dk[b, :t_end, hk] += dS[:, None] * q_vec[None, :] * scale
                dv[b, :t_end, hk] += p[:, None] * d_out_vec[None, :]

    d_qkv = np.zeros_like(qkv4)
    d_qkv[:, :, :Hq, :] = dq
    d_qkv[:, :, Hq:Hq + Hkv, :] = dk
    d_qkv[:, :, Hq + Hkv:, :] = dv
    return d_qkv.reshape(B, T, N * HS)


# -----------------------------------------------------------------------------
# Generators
# -----------------------------------------------------------------------------

def gen_matmul_swiglu() -> List[GoldenCase]:
    # swiglu_forward kernel requires (B*T*D) to be a multiple of 1024 for FP32.
    B, T, K, D = 2, 2, 3, 256
    M, N = B * T, 2 * D

    A = (np.arange(M * K, dtype=np.float64).reshape(M, K) * 0.1) - 0.2  # (M,K)
    W = ((np.arange(K * N, dtype=np.float64).reshape(K, N) % 17) - 8.0) * 0.01  # (K,N)

    # Forward
    up = A @ W  # (M,N)
    up_3d = up.reshape(B, T, N)
    x1 = up_3d[..., :D]
    x2 = up_3d[..., D:]
    sig = 1.0 / (1.0 + np.exp(-x2))
    out = x1 * x2 * sig  # (B,T,D)

    # Backward: pick a fixed upstream gradient
    DOUT = (np.arange(B * T * D, dtype=np.float64).reshape(B, T, D) * 0.01) - 0.5
    dx1 = DOUT * x2 * sig
    dx2 = DOUT * x1 * sig * (1.0 + x2 * (1.0 - sig))
    d_up = np.concatenate([dx1, dx2], axis=-1)  # (B,T,N)
    d_up_flat = d_up.reshape(M, N)

    dA = d_up_flat @ W.T  # (M,K)
    dW = A.T @ d_up_flat  # (K,N)

    # Optional torch cross-check
    _maybe_check_torch(A, W, DOUT, out, d_up, dA, dW)

    payload = {
        "op": "matmul_swiglu",
        "case": "small_case_1",
        "attrs": {"transpose": "NN"},
        "inputs": {
            "a": _tensor_payload(A),
            "b": _tensor_payload(W),
        },
        "outputs": {
            "out": _tensor_payload(out),
            "up_out": _tensor_payload(up),
        },
        "grads": {
            "d_out": _tensor_payload(DOUT),
            "d_up": _tensor_payload(d_up),
            "d_a": _tensor_payload(dA),
            "d_b": _tensor_payload(dW),
        },
        "meta": {
            "B": B,
            "T": T,
            "K": K,
            "D": D,
            "note": "A is (B*T,K) and W is (K,2D). out is swiglu(A@W).",
        },
    }

    return [GoldenCase(op="matmul_swiglu", case="small_case_1", payload=payload)]


def gen_embedding() -> List[GoldenCase]:
    # Small deterministic embedding lookup
    token_ids = np.array([[0, 1, 3], [2, 1, 0]], dtype=np.int32)
    vocab_size = 4
    hidden = 4
    weight = np.array(
        [[0.1, -0.2, 0.3, 0.4],
         [0.0, 0.5, -0.5, -0.25],
         [1.0, -1.0, 2.0, 0.75],
         [-0.75, 0.25, 0.5, -1.25]],
        dtype=np.float64,
    )
    out = weight[token_ids]

    payload = {
        "op": "embedding",
        "case": "small_case_1",
        "attrs": {},
        "inputs": {
            "token_ids": _tensor_payload(token_ids, dtype="int32"),
            "weight": _tensor_payload(weight),
        },
        "outputs": {
            "out": _tensor_payload(out),
        },
        "meta": {
            "vocab_size": vocab_size,
            "hidden": hidden,
            "note": "Embedding lookup: out[b,t] = weight[token_ids[b,t]].",
        },
    }
    return [GoldenCase(op="embedding", case="small_case_1", payload=payload)]


def gen_fused_residual_rmsnorm() -> List[GoldenCase]:
    B, T, C = 1, 2, 4
    residual_in = np.array(
        [[[-0.5, 1.0, 0.25, -1.5],
          [0.75, -0.25, 2.0, 0.5]]],
        dtype=np.float64,
    )
    inp = np.array(
        [[[1.5, -0.5, 0.75, 0.25],
          [-1.0, 0.5, -0.25, 1.0]]],
        dtype=np.float64,
    )
    weight = np.array([1.0, 0.5, 1.5, -0.25], dtype=np.float64)
    eps = 1e-5

    residual_out = residual_in + inp
    flat = residual_out.reshape(B * T, C)
    mean_sq = np.mean(flat * flat, axis=1)
    rstd = (1.0 / np.sqrt(mean_sq + eps)).reshape(B, T)
    y = residual_out * rstd[:, :, None] * weight[None, None, :]

    payload = {
        "op": "fused_residual_rmsnorm",
        "case": "small_case_1",
        "attrs": {"eps": eps},
        "inputs": {
            "residual_in": _tensor_payload(residual_in),
            "input": _tensor_payload(inp),
            "weight": _tensor_payload(weight),
        },
        "outputs": {
            "residual_out": _tensor_payload(residual_out),
            "y": _tensor_payload(y),
            "rstd": _tensor_payload(rstd),
        },
        "meta": {
            "B": B,
            "T": T,
            "C": C,
            "note": "residual_out = residual_in + input; y = residual_out * rstd * weight.",
        },
    }
    return [GoldenCase(op="fused_residual_rmsnorm", case="small_case_1", payload=payload)]


def gen_rope() -> List[GoldenCase]:
    B, T = 1, 2
    Hq, Hkv = 2, 1
    HD = 4
    rotary_dim = HD
    N = Hq + 2 * Hkv
    qkv_channels = N * HD

    qkv = np.arange(B * T * qkv_channels, dtype=np.float64).reshape(B, T, qkv_channels) * 0.1 - 0.5
    freqs = _precompute_freqs_cis(rotary_dim, T, 10000.0)
    pos_ids = np.tile(np.arange(T, dtype=np.int32), (B, 1))

    out = _rope_forward_cpu(qkv, freqs, pos_ids, Hq, Hkv, HD, rotary_dim)

    d_out = np.linspace(-1.0, 1.0, B * T * qkv_channels, dtype=np.float64).reshape(B, T, qkv_channels)
    d_inp = _rope_backward_cpu(d_out, freqs, pos_ids, Hq, Hkv, HD, rotary_dim)

    payload = {
        "op": "rope",
        "case": "small_case_1",
        "attrs": {"rotary_dim": rotary_dim},
        "inputs": {
            "qkv": _tensor_payload(qkv),
            "freqs": _tensor_payload(freqs),
            "position_ids": _tensor_payload(pos_ids, dtype="int32"),
        },
        "outputs": {
            "out": _tensor_payload(out),
        },
        "grads": {
            "d_out": _tensor_payload(d_out),
            "d_inp": _tensor_payload(d_inp),
        },
        "meta": {
            "B": B,
            "T": T,
            "Hq": Hq,
            "Hkv": Hkv,
            "head_dim": HD,
            "note": "RoPE applied to Q/K heads; V heads passthrough.",
        },
    }
    return [GoldenCase(op="rope", case="small_case_1", payload=payload)]


def gen_qkv_qk_norm_rope() -> List[GoldenCase]:
    B, T = 1, 2
    Hq, Hkv = 2, 1
    HS = 64
    qkv_channels = (Hq + 2 * Hkv) * HS
    eps = 1e-5
    rotary_dim = HS

    qkv = np.linspace(-0.8, 0.8, B * T * qkv_channels, dtype=np.float64).reshape(B, T, qkv_channels)

    q_weight = np.linspace(0.5, 1.5, HS, dtype=np.float64)
    k_weight = np.linspace(-1.0, 1.0, HS, dtype=np.float64)
    freqs = _precompute_freqs_cis(rotary_dim, T, 10000.0)
    pos_ids = np.tile(np.arange(T, dtype=np.int32), (B, 1))

    qkv_q, q_rstd = _qkv_head_rmsnorm_forward(qkv, q_weight, eps, B, T, Hq, HS, 0)
    q_rows = Hq * HS
    qkv_qk, k_rstd = _qkv_head_rmsnorm_forward(qkv_q, k_weight, eps, B, T, Hkv, HS, q_rows)
    qkv_out = _rope_forward_cpu(qkv_qk, freqs, pos_ids, Hq, Hkv, HS, rotary_dim)

    payload = {
        "op": "qkv_qk_norm_rope",
        "case": "small_case_1",
        "attrs": {"eps": eps, "rotary_dim": rotary_dim},
        "inputs": {
            "qkv": _tensor_payload(qkv),
            "q_norm": _tensor_payload(q_weight),
            "k_norm": _tensor_payload(k_weight),
            "freqs": _tensor_payload(freqs),
            "position_ids": _tensor_payload(pos_ids, dtype="int32"),
        },
        "outputs": {
            "qkv_out": _tensor_payload(qkv_out),
            "q_rstd": _tensor_payload(q_rstd),
            "k_rstd": _tensor_payload(k_rstd),
        },
        "meta": {
            "B": B,
            "T": T,
            "Hq": Hq,
            "Hkv": Hkv,
            "head_dim": HS,
            "note": "Applies Q/K RMSNorm (per-head) then RoPE on Q/K.",
        },
    }
    return [GoldenCase(op="qkv_qk_norm_rope", case="small_case_1", payload=payload)]


def gen_flash_attention() -> List[GoldenCase]:
    B, T = 1, 3
    Hq, Hkv = 2, 1
    HS = 64
    causal = True
    qkv_channels = (Hq + 2 * Hkv) * HS

    qkv = np.linspace(-0.4, 0.4, B * T * qkv_channels, dtype=np.float64).reshape(B, T, qkv_channels)
    out, lse = _flash_attention_ref(qkv, B, T, Hq, Hkv, HS, causal)

    out_flat = out.reshape(B, T, Hq * HS)

    payload = {
        "op": "flash_attention",
        "case": "small_case_1",
        "attrs": {"causal": causal},
        "inputs": {
            "qkv": _tensor_payload(qkv),
        },
        "outputs": {
            "out": _tensor_payload(out_flat),
            "lse": _tensor_payload(lse),
        },
        "meta": {
            "B": B,
            "T": T,
            "Hq": Hq,
            "Hkv": Hkv,
            "head_dim": HS,
            "note": "Reference uses causal scaled dot-product attention with GQA.",
        },
    }
    return [GoldenCase(op="flash_attention", case="small_case_1", payload=payload)]


def gen_cross_entropy_loss() -> List[GoldenCase]:
    logits = np.array(
        [[1.0, -0.5, 0.25, 2.0, -1.0, 0.5, -0.75, 1.25],
         [0.5, 1.5, -0.25, -1.0, 0.75, -0.5, 1.0, -1.25],
         [-0.5, 0.25, 1.0, -0.75, 1.5, -1.0, 0.5, 0.25]],
        dtype=np.float64,
    )
    targets = np.array([3, 1, 6], dtype=np.int32)

    losses = []
    for i in range(logits.shape[0]):
        lse = _logsumexp(logits[i])
        losses.append(lse - logits[i, targets[i]])
    loss = np.array(losses, dtype=np.float64)

    payload = {
        "op": "cross_entropy_loss",
        "case": "small_case_1",
        "attrs": {"compute_accuracy": False},
        "inputs": {
            "logits": _tensor_payload(logits),
            "targets": _tensor_payload(targets, dtype="int32"),
        },
        "outputs": {
            "loss": _tensor_payload(loss),
        },
        "meta": {
            "note": "Loss per token: logsumexp(logits) - logits[target].",
        },
    }
    return [GoldenCase(op="cross_entropy_loss", case="small_case_1", payload=payload)]


def gen_fused_lm_head_loss() -> List[GoldenCase]:
    xF = np.array(
        [[1.0, -0.5, 0.25],
         [0.5, 1.5, -1.0],
         [-0.75, 0.25, 1.25]],
        dtype=np.float64,
    )
    weight = np.array(
        [[0.5, -1.0, 0.25],
         [-0.5, 1.5, 0.75],
         [1.0, 0.0, -0.5],
         [-1.25, 0.5, 1.0]],
        dtype=np.float64,
    )
    targets = np.array([2, 1, 3], dtype=np.int32)

    logits = xF @ weight.T
    losses = []
    for i in range(logits.shape[0]):
        lse = _logsumexp(logits[i])
        losses.append(lse - logits[i, targets[i]])
    loss = np.array(losses, dtype=np.float64)

    payload = {
        "op": "fused_lm_head_loss",
        "case": "small_case_1",
        "attrs": {"compute_accuracy": False},
        "inputs": {
            "xF_flat": _tensor_payload(xF),
            "weight": _tensor_payload(weight),
            "targets": _tensor_payload(targets, dtype="int32"),
        },
        "outputs": {
            "loss": _tensor_payload(loss),
        },
        "meta": {
            "note": "Logits = xF @ weight.T; loss = logsumexp(logits) - logits[target].",
        },
    }
    return [GoldenCase(op="fused_lm_head_loss", case="small_case_1", payload=payload)]


def gen_flash_attention_backward() -> List[GoldenCase]:
    B, T = 1, 3
    Hq, Hkv = 2, 1
    HS = 64
    causal = True
    qkv_channels = (Hq + 2 * Hkv) * HS

    qkv = np.linspace(-0.4, 0.4, B * T * qkv_channels, dtype=np.float64).reshape(B, T, qkv_channels)
    out, lse = _flash_attention_ref(qkv, B, T, Hq, Hkv, HS, causal)

    d_out = np.linspace(-0.2, 0.2, B * T * Hq * HS, dtype=np.float64).reshape(B, T, Hq, HS)
    d_qkv = _flash_attention_backward_ref(qkv, d_out, B, T, Hq, Hkv, HS, causal)

    out_flat = out.reshape(B, T, Hq * HS)

    payload = {
        "op": "flash_attention_backward",
        "case": "small_case_1",
        "attrs": {"causal": causal},
        "inputs": {
            "d_out": _tensor_payload(d_out),
            "out": _tensor_payload(out_flat),
            "lse": _tensor_payload(lse),
            "qkv": _tensor_payload(qkv),
        },
        "outputs": {
            "d_qkv": _tensor_payload(d_qkv),
        },
        "meta": {
            "B": B,
            "T": T,
            "Hq": Hq,
            "Hkv": Hkv,
            "head_dim": HS,
            "note": "Backward uses naive softmax grad for causal attention with GQA.",
        },
    }
    return [GoldenCase(op="flash_attention_backward", case="small_case_1", payload=payload)]


def gen_cross_entropy_backward() -> List[GoldenCase]:
    logits = np.array(
        [[1.0, -0.5, 0.25, 2.0, -1.0, 0.5, -0.75, 1.25],
         [0.5, 1.5, -0.25, -1.0, 0.75, -0.5, 1.0, -1.25],
         [-0.5, 0.25, 1.0, -0.75, 1.5, -1.0, 0.5, 0.25]],
        dtype=np.float64,
    )
    targets = np.array([3, 1, 6], dtype=np.int32)
    BT = logits.shape[0]
    d_loss = np.ones((BT,), dtype=np.float64) / BT

    d_logits = np.zeros_like(logits)
    for i in range(BT):
        p = _softmax(logits[i])
        d_logits[i] = p * d_loss[i]
        d_logits[i, targets[i]] -= d_loss[i]

    payload = {
        "op": "cross_entropy_backward",
        "case": "small_case_1",
        "attrs": {},
        "inputs": {
            "d_loss": _tensor_payload(d_loss),
            "logits": _tensor_payload(logits),
            "targets": _tensor_payload(targets, dtype="int32"),
        },
        "outputs": {
            "d_logits": _tensor_payload(d_logits),
        },
        "meta": {
            "note": "d_logits = (softmax - one_hot) * d_loss.",
        },
    }
    return [GoldenCase(op="cross_entropy_backward", case="small_case_1", payload=payload)]


def gen_fused_lm_head_loss_backward() -> List[GoldenCase]:
    xF = np.array(
        [[1.0, -0.5, 0.25],
         [0.5, 1.5, -1.0],
         [-0.75, 0.25, 1.25]],
        dtype=np.float64,
    )
    weight = np.array(
        [[0.5, -1.0, 0.25],
         [-0.5, 1.5, 0.75],
         [1.0, 0.0, -0.5],
         [-1.25, 0.5, 1.0]],
        dtype=np.float64,
    )
    targets = np.array([2, 1, 3], dtype=np.int32)
    BT = xF.shape[0]
    d_loss = np.ones((BT,), dtype=np.float64) / BT

    logits = xF @ weight.T
    d_logits = np.zeros_like(logits)
    for i in range(BT):
        p = _softmax(logits[i])
        d_logits[i] = p * d_loss[i]
        d_logits[i, targets[i]] -= d_loss[i]

    d_xF = d_logits @ weight
    d_weight = d_logits.T @ xF

    payload = {
        "op": "fused_lm_head_loss_backward",
        "case": "small_case_1",
        "attrs": {},
        "inputs": {
            "d_loss": _tensor_payload(d_loss),
            "xF_flat": _tensor_payload(xF),
            "weight": _tensor_payload(weight),
            "targets": _tensor_payload(targets, dtype="int32"),
        },
        "outputs": {
            "d_xF": _tensor_payload(d_xF),
            "d_weight": _tensor_payload(d_weight),
        },
        "meta": {
            "note": "Backward of logits=xF@W^T and CE loss; uses d_loss scale per token.",
        },
    }
    return [GoldenCase(op="fused_lm_head_loss_backward", case="small_case_1", payload=payload)]


def gen_rope_backward() -> List[GoldenCase]:
    B, T = 1, 2
    Hq, Hkv = 2, 1
    HD = 4
    rotary_dim = HD
    N = Hq + 2 * Hkv
    qkv_channels = N * HD

    freqs = _precompute_freqs_cis(rotary_dim, T, 10000.0)
    pos_ids = np.tile(np.arange(T, dtype=np.int32), (B, 1))

    d_out = np.linspace(-0.5, 0.5, B * T * qkv_channels, dtype=np.float64).reshape(B, T, qkv_channels)
    d_qkv = _rope_backward_cpu(d_out, freqs, pos_ids, Hq, Hkv, HD, rotary_dim)

    payload = {
        "op": "rope_backward",
        "case": "small_case_1",
        "attrs": {"rotary_dim": rotary_dim},
        "inputs": {
            "d_out": _tensor_payload(d_out),
            "freqs": _tensor_payload(freqs),
            "position_ids": _tensor_payload(pos_ids, dtype="int32"),
        },
        "outputs": {
            "d_qkv": _tensor_payload(d_qkv),
        },
        "meta": {
            "B": B,
            "T": T,
            "Hq": Hq,
            "Hkv": Hkv,
            "head_dim": HD,
            "note": "Inverse rotation applied to Q/K gradients; V passthrough.",
        },
    }
    return [GoldenCase(op="rope_backward", case="small_case_1", payload=payload)]


def gen_qkv_qk_norm_rope_backward() -> List[GoldenCase]:
    B, T = 1, 2
    Hq, Hkv = 2, 1
    HS = 64
    qkv_channels = (Hq + 2 * Hkv) * HS
    eps = 1e-5
    rotary_dim = HS

    qkv = np.linspace(-0.8, 0.8, B * T * qkv_channels, dtype=np.float64).reshape(B, T, qkv_channels)

    q_weight = np.linspace(0.5, 1.5, HS, dtype=np.float64)
    k_weight = np.linspace(-1.0, 1.0, HS, dtype=np.float64)
    freqs = _precompute_freqs_cis(rotary_dim, T, 10000.0)
    pos_ids = np.tile(np.arange(T, dtype=np.int32), (B, 1))

    qkv_q, q_rstd = _qkv_head_rmsnorm_forward(qkv, q_weight, eps, B, T, Hq, HS, 0)
    q_rows = Hq * HS
    qkv_qk, k_rstd = _qkv_head_rmsnorm_forward(qkv_q, k_weight, eps, B, T, Hkv, HS, q_rows)
    qkv_out = _rope_forward_cpu(qkv_qk, freqs, pos_ids, Hq, Hkv, HS, rotary_dim)

    d_out = np.linspace(-0.4, 0.4, B * T * qkv_channels, dtype=np.float64).reshape(B, T, qkv_channels)

    d_qkv = _rope_backward_cpu(d_out, freqs, pos_ids, Hq, Hkv, HS, rotary_dim)
    qkv_pre = _rope_backward_cpu(qkv_out, freqs, pos_ids, Hq, Hkv, HS, rotary_dim)

    d_qkv = _qkv_head_rmsnorm_backward_dx(d_qkv, qkv_pre, q_weight, q_rstd, B, T, Hq, HS, 0)
    d_qkv = _qkv_head_rmsnorm_backward_dx(d_qkv, qkv_pre, k_weight, k_rstd, B, T, Hkv, HS, q_rows)

    payload = {
        "op": "qkv_qk_norm_rope_backward",
        "case": "small_case_1",
        "attrs": {"eps": eps, "rotary_dim": rotary_dim},
        "inputs": {
            "d_out": _tensor_payload(d_out),
            "qkv": _tensor_payload(qkv_out),
            "q_norm": _tensor_payload(q_weight),
            "k_norm": _tensor_payload(k_weight),
            "q_rstd": _tensor_payload(q_rstd),
            "k_rstd": _tensor_payload(k_rstd),
            "freqs": _tensor_payload(freqs),
            "position_ids": _tensor_payload(pos_ids, dtype="int32"),
        },
        "outputs": {
            "d_qkv": _tensor_payload(d_qkv),
        },
        "meta": {
            "B": B,
            "T": T,
            "Hq": Hq,
            "Hkv": Hkv,
            "head_dim": HS,
            "note": "Baseline: undo RoPE, then RMSNorm backward on Q/K.",
        },
    }
    return [GoldenCase(op="qkv_qk_norm_rope_backward", case="small_case_1", payload=payload)]


def gen_fused_residual_rmsnorm_backward() -> List[GoldenCase]:
    B, T, C = 1, 2, 4
    residual_in = np.array(
        [[[-0.5, 1.0, 0.25, -1.5],
          [0.75, -0.25, 2.0, 0.5]]],
        dtype=np.float64,
    )
    inp = np.array(
        [[[1.5, -0.5, 0.75, 0.25],
          [-1.0, 0.5, -0.25, 1.0]]],
        dtype=np.float64,
    )
    weight = np.array([1.0, 0.5, 1.5, -0.25], dtype=np.float64)
    eps = 1e-5

    residual_out = residual_in + inp
    flat = residual_out.reshape(B * T, C)
    mean_sq = np.mean(flat * flat, axis=1)
    rstd = (1.0 / np.sqrt(mean_sq + eps)).reshape(B, T)

    d_y = np.array(
        [[[0.25, -0.5, 1.0, -1.5],
          [0.75, 0.5, -0.25, 1.25]]],
        dtype=np.float64,
    )
    d_residual_next = np.array(
        [[[0.1, -0.2, 0.3, -0.4],
          [-0.5, 0.6, -0.7, 0.8]]],
        dtype=np.float64,
    )

    dx = _rmsnorm_backward(d_y, residual_out, weight, rstd)
    d_input = d_residual_next + dx
    d_residual = d_input.copy()
    d_weight = _rmsnorm_dweight(d_y, residual_out, rstd)

    payload = {
        "op": "fused_residual_rmsnorm_backward",
        "case": "small_case_1",
        "attrs": {"eps": eps},
        "inputs": {
            "d_y": _tensor_payload(d_y),
            "d_residual_next": _tensor_payload(d_residual_next),
            "residual_out": _tensor_payload(residual_out),
            "weight": _tensor_payload(weight),
            "rstd": _tensor_payload(rstd),
        },
        "outputs": {
            "d_residual": _tensor_payload(d_residual),
            "d_input": _tensor_payload(d_input),
            "d_weight": _tensor_payload(d_weight),
        },
        "meta": {
            "B": B,
            "T": T,
            "C": C,
            "note": "d_input = d_residual_next + rmsnorm_backward(d_y).",
        },
    }
    return [GoldenCase(op="fused_residual_rmsnorm_backward", case="small_case_1", payload=payload)]


def gen_add_backward() -> List[GoldenCase]:
    d_out = np.array([[1.0, -0.5, 2.0],
                      [0.25, 3.0, -1.5]], dtype=np.float64)
    d_a = d_out.copy()
    d_b = d_out.copy()

    payload = {
        "op": "add_backward",
        "case": "small_case_1",
        "attrs": {},
        "inputs": {
            "d_out": _tensor_payload(d_out),
        },
        "outputs": {
            "d_a": _tensor_payload(d_a),
            "d_b": _tensor_payload(d_b),
        },
        "meta": {
            "note": "Add backward passes gradient to both inputs unchanged.",
        },
    }
    return [GoldenCase(op="add_backward", case="small_case_1", payload=payload)]


def gen_view_backward() -> List[GoldenCase]:
    d_out = np.array(
        [[1.0, -1.0, 2.0, -2.0],
         [0.5, 0.25, -0.5, 1.5],
         [3.0, -3.0, 4.0, -4.0]],
        dtype=np.float64,
    )
    d_inp = d_out.reshape(2, 3, 2)

    payload = {
        "op": "view_backward",
        "case": "small_case_1",
        "attrs": {"shape": [2, 3, 2]},
        "inputs": {
            "d_out": _tensor_payload(d_out),
        },
        "outputs": {
            "d_inp": _tensor_payload(d_inp),
        },
        "meta": {
            "note": "View backward reshapes gradient to input shape.",
        },
    }
    return [GoldenCase(op="view_backward", case="small_case_1", payload=payload)]


def gen_bias_add_backward() -> List[GoldenCase]:
    d_out = np.array(
        [[[1.0, -0.5, 2.0, 0.75],
          [0.25, 3.0, -1.5, -2.0]]],
        dtype=np.float64,
    )
    d_x = d_out.copy()
    d_bias = d_out.sum(axis=(0, 1))

    payload = {
        "op": "bias_add_backward",
        "case": "small_case_1",
        "attrs": {},
        "inputs": {
            "d_out": _tensor_payload(d_out),
        },
        "outputs": {
            "d_x": _tensor_payload(d_x),
            "d_bias": _tensor_payload(d_bias),
        },
        "meta": {
            "note": "d_bias is sum over B,T; d_x passthrough.",
        },
    }
    return [GoldenCase(op="bias_add_backward", case="small_case_1", payload=payload)]


def gen_matmul_backward() -> List[GoldenCase]:
    A = np.array([[1.0, 2.0, -1.0],
                  [0.5, -1.0, 3.0]], dtype=np.float64)
    B = np.array([[1.0, 0.0, 2.0, -1.0],
                  [0.0, 1.0, -1.0, 0.5],
                  [1.0, 1.0, 0.0, 2.0]], dtype=np.float64)
    d_out = np.array([[1.0, -0.5, 2.0, -1.0],
                      [0.25, 3.0, -1.5, 0.5]], dtype=np.float64)
    dA = d_out @ B.T
    dB = A.T @ d_out

    payload = {
        "op": "matmul_backward",
        "case": "small_case_1",
        "attrs": {"transpose": "NN"},
        "inputs": {
            "d_out": _tensor_payload(d_out),
            "a": _tensor_payload(A),
            "b": _tensor_payload(B),
        },
        "outputs": {
            "d_a": _tensor_payload(dA),
            "d_b": _tensor_payload(dB),
        },
        "meta": {
            "note": "dA = d_out @ B^T, dB = A^T @ d_out.",
        },
    }
    return [GoldenCase(op="matmul_backward", case="small_case_1", payload=payload)]


def gen_swiglu_backward() -> List[GoldenCase]:
    # swiglu_backward kernel requires (B*T*C) to be a multiple of 1024 for FP32.
    B, T, C = 2, 2, 256
    inp = (np.arange(B * T * 2 * C, dtype=np.float64).reshape(B, T, 2 * C) * 0.01) - 0.5
    up = inp[..., :C]
    gate = inp[..., C:]
    sig = 1.0 / (1.0 + np.exp(-gate))

    d_out = (np.arange(B * T * C, dtype=np.float64).reshape(B, T, C) * 0.01) - 0.5
    dx1 = d_out * gate * sig
    dx2 = d_out * up * sig * (1.0 + gate * (1.0 - sig))
    d_inp = np.concatenate([dx1, dx2], axis=-1)

    payload = {
        "op": "swiglu_backward",
        "case": "small_case_1",
        "attrs": {},
        "inputs": {
            "d_out": _tensor_payload(d_out),
            "mlp_up": _tensor_payload(inp),
        },
        "outputs": {
            "d_inp": _tensor_payload(d_inp),
        },
        "meta": {
            "note": "SwiGLU backward for input (pre-swiglu).",
        },
    }
    return [GoldenCase(op="swiglu_backward", case="small_case_1", payload=payload)]


def gen_matmul_swiglu_backward() -> List[GoldenCase]:
    # swiglu_backward kernel requires (B*T*D) to be a multiple of 1024 for FP32.
    B, T, K, D = 2, 2, 3, 256
    M, N = B * T, 2 * D

    ln2 = (np.arange(M * K, dtype=np.float64).reshape(M, K) * 0.1) - 0.2  # (M,K)
    weight = ((np.arange(K * N, dtype=np.float64).reshape(K, N) % 17) - 8.0) * 0.01  # (K,N)

    mlp_up = (ln2 @ weight).reshape(B, T, N)
    up = mlp_up[..., :D]
    gate = mlp_up[..., D:]
    sig = 1.0 / (1.0 + np.exp(-gate))

    d_out = (np.arange(B * T * D, dtype=np.float64).reshape(B, T, D) * 0.01) - 0.5
    dx1 = d_out * gate * sig
    dx2 = d_out * up * sig * (1.0 + gate * (1.0 - sig))
    d_mlp_up = np.concatenate([dx1, dx2], axis=-1).reshape(M, N)

    d_inp = d_mlp_up @ weight.T
    d_weight = ln2.T @ d_mlp_up

    payload = {
        "op": "matmul_swiglu_backward",
        "case": "small_case_1",
        "attrs": {"transpose": "NN"},
        "inputs": {
            "d_out": _tensor_payload(d_out),
            "ln2": _tensor_payload(ln2),
            "weight": _tensor_payload(weight),
            "mlp_up": _tensor_payload(mlp_up),
        },
        "outputs": {
            "d_inp": _tensor_payload(d_inp),
            "d_weight": _tensor_payload(d_weight),
        },
        "meta": {
            "B": B,
            "T": T,
            "K": K,
            "D": D,
            "note": "Backward of fused matmul+swiglu (NN transpose).",
        },
    }
    return [GoldenCase(op="matmul_swiglu_backward", case="small_case_1", payload=payload)]


def gen_embedding_backward() -> List[GoldenCase]:
    token_ids = np.array([[0, 1, 3], [2, 1, 0]], dtype=np.int32)
    d_out = np.array(
        [[[0.1, -0.2, 0.3, 0.4],
          [0.0, 0.5, -0.5, 0.25],
          [1.0, -1.0, 2.0, -0.75]],
         [[-0.75, 0.25, 0.5, -1.25],
          [0.2, -0.1, 0.4, 0.6],
          [0.05, 0.15, -0.25, -0.35]]],
        dtype=np.float64,
    )
    vocab_size = 4
    hidden = d_out.shape[2]
    d_embedding = np.zeros((vocab_size, hidden), dtype=np.float64)
    for b in range(token_ids.shape[0]):
        for t in range(token_ids.shape[1]):
            idx = token_ids[b, t]
            d_embedding[idx] += d_out[b, t]

    payload = {
        "op": "embedding_backward",
        "case": "small_case_1",
        "attrs": {},
        "inputs": {
            "d_out": _tensor_payload(d_out),
            "token_ids": _tensor_payload(token_ids, dtype="int32"),
        },
        "outputs": {
            "d_embedding": _tensor_payload(d_embedding),
        },
        "meta": {
            "vocab_size": vocab_size,
            "hidden": hidden,
            "note": "d_embedding accumulates d_out by token id.",
        },
    }
    return [GoldenCase(op="embedding_backward", case="small_case_1", payload=payload)]


def gen_zeros_backward() -> List[GoldenCase]:
    payload = {
        "op": "zeros_backward",
        "case": "small_case_1",
        "attrs": {},
        "inputs": {},
        "outputs": {},
        "meta": {
            "note": "Zeros backward is a no-op.",
        },
    }
    return [GoldenCase(op="zeros_backward", case="small_case_1", payload=payload)]


def gen_add() -> List[GoldenCase]:
    A = np.array([[1.5, -2.0, 0.25],
                  [3.0, 0.0, -4.5]], dtype=np.float64)
    B = np.array([[-1.0, 2.0, 1.75],
                  [0.5, -3.0, 2.25]], dtype=np.float64)
    out = A + B

    d_out = np.array([[1.0, -0.5, 2.0],
                      [0.25, 3.0, -1.5]], dtype=np.float64)
    d_a = d_out.copy()
    d_b = d_out.copy()

    if torch is not None:
        A_t = torch.tensor(A, dtype=torch.float64, requires_grad=True)
        B_t = torch.tensor(B, dtype=torch.float64, requires_grad=True)
        out_t = A_t + B_t
        d_out_t = torch.tensor(d_out, dtype=torch.float64)
        dA_t, dB_t = torch.autograd.grad(out_t, (A_t, B_t), grad_outputs=d_out_t)
        np.testing.assert_allclose(out, out_t.detach().numpy(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(d_a, dA_t.detach().numpy(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(d_b, dB_t.detach().numpy(), rtol=1e-12, atol=1e-12)

    payload = {
        "op": "add",
        "case": "small_case_1",
        "attrs": {},
        "inputs": {
            "a": _tensor_payload(A),
            "b": _tensor_payload(B),
        },
        "outputs": {
            "out": _tensor_payload(out),
        },
        "grads": {
            "d_out": _tensor_payload(d_out),
            "d_a": _tensor_payload(d_a),
            "d_b": _tensor_payload(d_b),
        },
        "meta": {
            "note": "Elementwise add; grads pass through unchanged.",
        },
    }
    return [GoldenCase(op="add", case="small_case_1", payload=payload)]


def gen_view() -> List[GoldenCase]:
    x = np.arange(12, dtype=np.float64).reshape(2, 3, 2)
    out = x.reshape(3, 4)
    d_out = np.array(
        [[1.0, -1.0, 2.0, -2.0],
         [0.5, 0.25, -0.5, 1.5],
         [3.0, -3.0, 4.0, -4.0]],
        dtype=np.float64,
    )
    d_inp = d_out.reshape(x.shape)

    payload = {
        "op": "view",
        "case": "small_case_1",
        "attrs": {"shape": [3, 4]},
        "inputs": {
            "x": _tensor_payload(x),
        },
        "outputs": {
            "out": _tensor_payload(out),
        },
        "grads": {
            "d_out": _tensor_payload(d_out),
            "d_inp": _tensor_payload(d_inp),
        },
        "meta": {
            "note": "View is a reshape with identical data ordering.",
        },
    }
    return [GoldenCase(op="view", case="small_case_1", payload=payload)]


def gen_zeros() -> List[GoldenCase]:
    out = np.zeros((2, 3), dtype=np.float64)
    payload = {
        "op": "zeros",
        "case": "small_case_1",
        "attrs": {},
        "inputs": {},
        "outputs": {
            "out": _tensor_payload(out),
        },
        "meta": {
            "note": "Zeros op has no inputs; output is all zeros.",
        },
    }
    return [GoldenCase(op="zeros", case="small_case_1", payload=payload)]


def gen_bias_add() -> List[GoldenCase]:
    # x shape (B,T,C), bias shape (C)
    x = np.array(
        [[[1.0, -2.0, 0.5, 1.25],
          [0.25, 3.0, -1.5, -0.75]]],
        dtype=np.float64,
    )
    bias = np.array([0.5, -1.0, 2.0, -0.25], dtype=np.float64)
    out = x + bias

    d_out = np.array(
        [[[1.0, -0.5, 2.0, 0.75],
          [0.25, 3.0, -1.5, -2.0]]],
        dtype=np.float64,
    )
    d_x = d_out.copy()
    d_bias = d_out.sum(axis=(0, 1))

    if torch is not None:
        x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        b_t = torch.tensor(bias, dtype=torch.float64, requires_grad=True)
        out_t = x_t + b_t
        d_out_t = torch.tensor(d_out, dtype=torch.float64)
        d_x_t, d_b_t = torch.autograd.grad(out_t, (x_t, b_t), grad_outputs=d_out_t)
        np.testing.assert_allclose(out, out_t.detach().numpy(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(d_x, d_x_t.detach().numpy(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(d_bias, d_b_t.detach().numpy(), rtol=1e-12, atol=1e-12)

    payload = {
        "op": "bias_add",
        "case": "small_case_1",
        "attrs": {},
        "inputs": {
            "x": _tensor_payload(x),
            "bias": _tensor_payload(bias),
        },
        "outputs": {
            "out": _tensor_payload(out),
        },
        "grads": {
            "d_out": _tensor_payload(d_out),
            "d_x": _tensor_payload(d_x),
            "d_bias": _tensor_payload(d_bias),
        },
        "meta": {
            "note": "Bias added along last dimension; d_bias sums over B,T.",
        },
    }
    return [GoldenCase(op="bias_add", case="small_case_1", payload=payload)]


def gen_matmul() -> List[GoldenCase]:
    A = np.array([[1.0, 2.0, -1.0],
                  [0.5, -1.0, 3.0]], dtype=np.float64)  # (M,K)
    B = np.array([[1.0, 0.0, 2.0, -1.0],
                  [0.0, 1.0, -1.0, 0.5],
                  [1.0, 1.0, 0.0, 2.0]], dtype=np.float64)  # (K,N)
    out = A @ B

    d_out = np.array([[1.0, -0.5, 2.0, -1.0],
                      [0.25, 3.0, -1.5, 0.5]], dtype=np.float64)
    dA = d_out @ B.T
    dB = A.T @ d_out

    if torch is not None:
        A_t = torch.tensor(A, dtype=torch.float64, requires_grad=True)
        B_t = torch.tensor(B, dtype=torch.float64, requires_grad=True)
        out_t = A_t @ B_t
        d_out_t = torch.tensor(d_out, dtype=torch.float64)
        dA_t, dB_t = torch.autograd.grad(out_t, (A_t, B_t), grad_outputs=d_out_t)
        np.testing.assert_allclose(out, out_t.detach().numpy(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(dA, dA_t.detach().numpy(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(dB, dB_t.detach().numpy(), rtol=1e-12, atol=1e-12)

    payload = {
        "op": "matmul",
        "case": "small_case_1",
        "attrs": {"transpose": "NN"},
        "inputs": {
            "a": _tensor_payload(A),
            "b": _tensor_payload(B),
        },
        "outputs": {
            "out": _tensor_payload(out),
        },
        "grads": {
            "d_out": _tensor_payload(d_out),
            "d_a": _tensor_payload(dA),
            "d_b": _tensor_payload(dB),
        },
        "meta": {
            "note": "Standard matmul: out = A @ B.",
        },
    }
    return [GoldenCase(op="matmul", case="small_case_1", payload=payload)]


def gen_matmul_bias() -> List[GoldenCase]:
    A = np.array([[1.0, 2.0, -1.0],
                  [0.5, -1.0, 3.0]], dtype=np.float64)
    B = np.array([[1.0, 0.0, 2.0, -1.0],
                  [0.0, 1.0, -1.0, 0.5],
                  [1.0, 1.0, 0.0, 2.0]], dtype=np.float64)
    bias = np.array([0.5, -1.0, 2.0, -0.25], dtype=np.float64)
    out = A @ B + bias

    d_out = np.array([[1.0, -0.5, 2.0, -1.0],
                      [0.25, 3.0, -1.5, 0.5]], dtype=np.float64)
    dA = d_out @ B.T
    dB = A.T @ d_out
    d_bias = d_out.sum(axis=0)

    if torch is not None:
        A_t = torch.tensor(A, dtype=torch.float64, requires_grad=True)
        B_t = torch.tensor(B, dtype=torch.float64, requires_grad=True)
        b_t = torch.tensor(bias, dtype=torch.float64, requires_grad=True)
        out_t = A_t @ B_t + b_t
        d_out_t = torch.tensor(d_out, dtype=torch.float64)
        dA_t, dB_t, dBias_t = torch.autograd.grad(out_t, (A_t, B_t, b_t), grad_outputs=d_out_t)
        np.testing.assert_allclose(out, out_t.detach().numpy(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(dA, dA_t.detach().numpy(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(dB, dB_t.detach().numpy(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(d_bias, dBias_t.detach().numpy(), rtol=1e-12, atol=1e-12)

    payload = {
        "op": "matmul_bias",
        "case": "small_case_1",
        "attrs": {"transpose": "NN"},
        "inputs": {
            "a": _tensor_payload(A),
            "b": _tensor_payload(B),
            "bias": _tensor_payload(bias),
        },
        "outputs": {
            "out": _tensor_payload(out),
        },
        "grads": {
            "d_out": _tensor_payload(d_out),
            "d_a": _tensor_payload(dA),
            "d_b": _tensor_payload(dB),
            "d_bias": _tensor_payload(d_bias),
        },
        "meta": {
            "note": "Matmul with bias added on last dimension.",
        },
    }
    return [GoldenCase(op="matmul_bias", case="small_case_1", payload=payload)]


def gen_swiglu() -> List[GoldenCase]:
    # swiglu_forward kernel requires (B*T*C) to be a multiple of 1024 for FP32.
    B, T, C = 2, 2, 256
    inp = (np.arange(B * T * 2 * C, dtype=np.float64).reshape(B, T, 2 * C) * 0.01) - 0.5
    up = inp[..., :C]
    gate = inp[..., C:]
    sig = 1.0 / (1.0 + np.exp(-gate))
    out = up * gate * sig

    d_out = (np.arange(B * T * C, dtype=np.float64).reshape(B, T, C) * 0.01) - 0.5
    dx1 = d_out * gate * sig
    dx2 = d_out * up * sig * (1.0 + gate * (1.0 - sig))
    d_inp = np.concatenate([dx1, dx2], axis=-1)

    if torch is not None:
        inp_t = torch.tensor(inp, dtype=torch.float64, requires_grad=True)
        up_t = inp_t[..., :C]
        gate_t = inp_t[..., C:]
        out_t = up_t * gate_t * torch.sigmoid(gate_t)
        d_out_t = torch.tensor(d_out, dtype=torch.float64)
        (d_inp_t,) = torch.autograd.grad(out_t, (inp_t,), grad_outputs=d_out_t)
        np.testing.assert_allclose(out, out_t.detach().numpy(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(d_inp, d_inp_t.detach().numpy(), rtol=1e-12, atol=1e-12)

    payload = {
        "op": "swiglu",
        "case": "small_case_1",
        "attrs": {},
        "inputs": {
            "inp": _tensor_payload(inp),
        },
        "outputs": {
            "out": _tensor_payload(out),
        },
        "grads": {
            "d_out": _tensor_payload(d_out),
            "d_inp": _tensor_payload(d_inp),
        },
        "meta": {
            "note": "SwiGLU: out = up * gate * sigmoid(gate).",
        },
    }
    return [GoldenCase(op="swiglu", case="small_case_1", payload=payload)]


def gen_swiglu_mlp() -> List[GoldenCase]:
    """Generate SwiGLU MLP golden test (composed: matmul -> swiglu -> matmul)."""
    if torch is None:
        raise RuntimeError("PyTorch required for module golden generation")

    B, T, C, M = 2, 4, 128, 256

    # Initialize weights
    np.random.seed(42)
    up_weight = np.random.randn(2 * M, C).astype(np.float64) * 0.02
    down_weight = np.random.randn(C, M).astype(np.float64) * 0.02
    x = np.random.randn(B, T, C).astype(np.float64) * 0.5

    # Forward pass with PyTorch
    x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    up_w_t = torch.tensor(up_weight, dtype=torch.float64, requires_grad=True)
    down_w_t = torch.tensor(down_weight, dtype=torch.float64, requires_grad=True)

    # MLP forward: x -> matmul(up) -> swiglu -> matmul(down)
    x_flat = x_t.reshape(B * T, C)
    up_out = x_flat @ up_w_t.T  # (B*T, 2*M)
    up_3d = up_out.reshape(B, T, 2 * M)

    # SwiGLU
    gate = up_3d[..., :M]
    up = up_3d[..., M:]
    swiglu_out = gate * up * torch.sigmoid(up)  # (B, T, M)

    # Down projection
    swiglu_flat = swiglu_out.reshape(B * T, M)
    out_flat = swiglu_flat @ down_w_t.T
    out = out_flat.reshape(B, T, C)

    # Backward pass
    d_out = np.random.randn(B, T, C).astype(np.float64) * 0.1
    d_out_t = torch.tensor(d_out, dtype=torch.float64)
    d_x, d_up_w, d_down_w = torch.autograd.grad(
        out, (x_t, up_w_t, down_w_t),
        grad_outputs=d_out_t
    )

    # Save intermediate activations for verification
    up_3d_np = up_3d.detach().numpy()
    swiglu_out_np = swiglu_out.detach().numpy()

    payload = {
        "op": "swiglu_mlp",
        "case": "small_case_1",
        "attrs": {},
        "inputs": {
            "x": _tensor_payload(x),
            "up_weight": _tensor_payload(up_weight),
            "down_weight": _tensor_payload(down_weight),
        },
        "outputs": {
            "out": _tensor_payload(out.detach().numpy()),
            "up": _tensor_payload(up_3d_np),  # intermediate
            "swiglu": _tensor_payload(swiglu_out_np),  # intermediate
        },
        "grads": {
            "d_out": _tensor_payload(d_out),
            "d_x": _tensor_payload(d_x.numpy()),
            "d_up_weight": _tensor_payload(d_up_w.numpy()),
            "d_down_weight": _tensor_payload(d_down_w.numpy()),
        },
        "meta": {
            "B": B,
            "T": T,
            "C": C,
            "M": M,
            "note": "SwiGLU MLP: x -> up_proj -> swiglu -> down_proj",
        },
    }
    return [GoldenCase(op="swiglu_mlp", case="small_case_1", payload=payload)]


def gen_gqa_attention() -> List[GoldenCase]:
    """Generate GQA Attention golden test (composed: qkv_proj -> rope -> flash_attn -> out_proj)."""
    if torch is None:
        raise RuntimeError("PyTorch required for module golden generation")

    B, T = 2, 8
    C = 256  # d_model
    Hq, Hkv = 4, 2  # num_query_heads, num_kv_heads
    HD = 64  # head_dim
    QKV = (Hq + 2 * Hkv) * HD

    # Initialize weights
    np.random.seed(43)
    qkv_weight = np.random.randn(QKV, C).astype(np.float64) * 0.02
    out_weight = np.random.randn(C, Hq * HD).astype(np.float64) * 0.02
    x = np.random.randn(B, T, C).astype(np.float64) * 0.3
    freqs = _precompute_freqs_cis_complex(HD, T, 10000.0)  # 3D format [T, D//2, 2]
    pos_ids = np.arange(T, dtype=np.int32)  # 1D format [T]

    # Forward with PyTorch
    x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    qkv_w_t = torch.tensor(qkv_weight, dtype=torch.float64, requires_grad=True)
    out_w_t = torch.tensor(out_weight, dtype=torch.float64, requires_grad=True)
    freqs_t = torch.tensor(freqs, dtype=torch.float64)

    # QKV projection
    x_flat = x_t.reshape(B * T, C)
    qkv_flat = x_flat @ qkv_w_t.T
    qkv_packed = qkv_flat.reshape(B, T, Hq + 2 * Hkv, HD)

    # Apply RoPE
    qkv_rope = _torch_rope(qkv_packed, freqs_t, pos_ids, Hq, Hkv, HD)

    # Flash Attention (PyTorch reference for autograd)
    attn_out, lse = _torch_flash_attention_ref(qkv_rope, B, T, Hq, Hkv, HD, causal=True)

    # Output projection
    attn_flat = attn_out.reshape(B * T, Hq * HD)
    out_flat = attn_flat @ out_w_t.T
    out = out_flat.reshape(B, T, C)

    # Backward pass
    d_out = np.random.randn(B, T, C).astype(np.float64) * 0.1
    d_out_t = torch.tensor(d_out, dtype=torch.float64)
    d_x, d_qkv_w, d_out_w = torch.autograd.grad(
        out, (x_t, qkv_w_t, out_w_t),
        grad_outputs=d_out_t
    )

    # Save intermediates
    qkv_packed_np = qkv_packed.detach().numpy()
    qkv_rope_np = qkv_rope.detach().numpy()
    attn_out_np = attn_out.detach().numpy()
    lse_np = lse.detach().numpy()

    payload = {
        "op": "gqa_attention",
        "case": "small_case_1",
        "attrs": {"causal": True},
        "inputs": {
            "x": _tensor_payload(x),
            "qkv_weight": _tensor_payload(qkv_weight),
            "out_weight": _tensor_payload(out_weight),
            "rope_freqs": _tensor_payload(freqs),
            "position_ids": _tensor_payload(pos_ids, dtype="int32"),
        },
        "outputs": {
            "out": _tensor_payload(out.detach().numpy()),
            "qkv": _tensor_payload(qkv_packed_np),  # intermediate
            "qkv_rope": _tensor_payload(qkv_rope_np.reshape(B, T, QKV)),  # intermediate
            "attn_out": _tensor_payload(attn_out_np.reshape(B, T, Hq * HD)),  # intermediate
            "lse": _tensor_payload(lse_np),  # intermediate
        },
        "grads": {
            "d_out": _tensor_payload(d_out),
            "d_x": _tensor_payload(d_x.numpy()),
            "d_qkv_weight": _tensor_payload(d_qkv_w.numpy()),
            "d_out_weight": _tensor_payload(d_out_w.numpy()),
        },
        "meta": {
            "B": B,
            "T": T,
            "C": C,
            "Hq": Hq,
            "Hkv": Hkv,
            "head_dim": HD,
            "note": "GQA Attention: x -> qkv_proj -> rope -> flash_attn -> out_proj",
        },
    }
    return [GoldenCase(op="gqa_attention", case="small_case_1", payload=payload)]


def _torch_flash_attention_ref(qkv: torch.Tensor, B: int, T: int, Hq: int, Hkv: int, HS: int, causal: bool):
    """PyTorch flash attention reference (supports autograd)."""
    N = Hq + 2 * Hkv
    qkv4 = qkv.reshape(B, T, N, HS)
    q = qkv4[:, :, :Hq, :]
    k = qkv4[:, :, Hq:Hq + Hkv, :]
    v = qkv4[:, :, Hq + Hkv:, :]

    group = Hq // Hkv
    scale = 1.0 / np.sqrt(HS)

    # Reshape for batched ops: (B, Hq, T, HS)
    q = q.permute(0, 2, 1, 3)  # (B, Hq, T, HS)
    k = k.permute(0, 2, 1, 3)  # (B, Hkv, T, HS)
    v = v.permute(0, 2, 1, 3)  # (B, Hkv, T, HS)

    # Repeat KV for GQA
    k = k.repeat_interleave(group, dim=1)  # (B, Hq, T, HS)
    v = v.repeat_interleave(group, dim=1)  # (B, Hq, T, HS)

    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, Hq, T, T)

    # Apply causal mask
    if causal:
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=scores.device), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))

    # Compute LSE (logsumexp) for each query position
    lse = torch.logsumexp(scores, dim=-1)  # (B, Hq, T)

    # Softmax
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)  # (B, Hq, T, T)

    # Apply attention
    out = torch.matmul(attn_weights, v)  # (B, Hq, T, HS)

    # Reshape back
    out = out.permute(0, 2, 1, 3).reshape(B, T, Hq, HS)  # (B, T, Hq, HS)

    return out, lse


def _torch_rope(qkv: torch.Tensor, freqs: torch.Tensor, position_ids: np.ndarray,
                Hq: int, Hkv: int, head_dim: int) -> torch.Tensor:
    """Apply RoPE using PyTorch (for golden generation).

    Args:
        qkv: [B, T, N, HD] where N = Hq + 2*Hkv
        freqs: [T, HD//2, 2] - complex format (cos, sin)
        position_ids: [T] - 1D position indices
    """
    B, T, N, HD = qkv.shape
    rotary_half = HD // 2

    out = qkv.clone()
    for b in range(B):
        for t in range(T):
            t_pos = int(position_ids[t])  # 1D position_ids
            cos_vals = freqs[t_pos, :, 0]  # [HD//2] cos values
            sin_vals = freqs[t_pos, :, 1]  # [HD//2] sin values

            for h in range(N):
                # Q and K get RoPE, V passes through
                if h >= Hq and h < Hq + Hkv:
                    qkv_type = 1  # K
                elif h < Hq:
                    qkv_type = 0  # Q
                else:
                    qkv_type = 2  # V
                    continue

                # Clone to avoid in-place modification issue - real/imag are views,
                # so writing to out[..., :rotary_half] modifies what real points to
                real = out[b, t, h, :rotary_half].clone()
                imag = out[b, t, h, rotary_half:2*rotary_half].clone()

                out[b, t, h, :rotary_half] = real * cos_vals - imag * sin_vals
                out[b, t, h, rotary_half:2*rotary_half] = real * sin_vals + imag * cos_vals

    return out


def gen_embedding_module() -> List[GoldenCase]:
    """Generate Embedding module golden test (token lookup)."""
    if torch is None:
        raise RuntimeError("PyTorch required for module golden generation")

    B, T = 2, 8
    V = 256  # vocab size
    C = 128  # embedding dim

    # Initialize weights
    np.random.seed(50)
    embed_weight = np.random.randn(V, C).astype(np.float64) * 0.02
    token_ids = np.random.randint(0, V, size=(B, T), dtype=np.int32)

    # Forward with PyTorch
    embed_w_t = torch.tensor(embed_weight, dtype=torch.float64, requires_grad=True)
    token_ids_t = torch.tensor(token_ids, dtype=torch.long)

    # Embedding lookup
    out = torch.nn.functional.embedding(token_ids_t, embed_w_t)

    # Backward pass
    d_out = np.random.randn(B, T, C).astype(np.float64) * 0.1
    d_out_t = torch.tensor(d_out, dtype=torch.float64)
    d_embed_w, = torch.autograd.grad(out, (embed_w_t,), grad_outputs=d_out_t)

    payload = {
        "op": "embedding_module",
        "case": "small_case_1",
        "attrs": {},
        "inputs": {
            "token_ids": _tensor_payload(token_ids, dtype="int32"),
            "embedding_weight": _tensor_payload(embed_weight),
        },
        "outputs": {
            "out": _tensor_payload(out.detach().numpy()),
        },
        "grads": {
            "d_out": _tensor_payload(d_out),
            "d_embedding_weight": _tensor_payload(d_embed_w.numpy()),
        },
        "meta": {
            "B": B,
            "T": T,
            "V": V,
            "C": C,
            "note": "Embedding module: token_ids -> embedding lookup -> out",
        },
    }
    return [GoldenCase(op="embedding_module", case="small_case_1", payload=payload)]


def gen_rmsnorm_module() -> List[GoldenCase]:
    """Generate RMSNorm module golden test (fused residual + RMS normalization)."""
    if torch is None:
        raise RuntimeError("PyTorch required for module golden generation")

    B, T = 2, 8
    C = 128  # hidden size
    eps = 1e-5

    # Initialize
    np.random.seed(51)
    residual = np.random.randn(B, T, C).astype(np.float64) * 0.3
    x = np.random.randn(B, T, C).astype(np.float64) * 0.3
    weight = np.random.randn(C).astype(np.float64) * 0.1 + 1.0  # gamma near 1

    # Forward with PyTorch
    residual_t = torch.tensor(residual, dtype=torch.float64, requires_grad=True)
    x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    weight_t = torch.tensor(weight, dtype=torch.float64, requires_grad=True)

    # Fused residual + RMSNorm
    residual_out = residual_t + x_t
    rms = torch.sqrt(torch.mean(residual_out ** 2, dim=-1, keepdim=True) + eps)
    rstd = 1.0 / rms
    y = residual_out * rstd * weight_t

    # Backward pass
    d_y = np.random.randn(B, T, C).astype(np.float64) * 0.1
    d_y_t = torch.tensor(d_y, dtype=torch.float64)
    d_residual, d_x, d_weight = torch.autograd.grad(
        y, (residual_t, x_t, weight_t), grad_outputs=d_y_t
    )

    payload = {
        "op": "rmsnorm_module",
        "case": "small_case_1",
        "attrs": {"eps": eps},
        "inputs": {
            "residual": _tensor_payload(residual),
            "x": _tensor_payload(x),
            "weight": _tensor_payload(weight),
        },
        "outputs": {
            "residual_out": _tensor_payload(residual_out.detach().numpy()),
            "y": _tensor_payload(y.detach().numpy()),
            "rstd": _tensor_payload(rstd.detach().numpy().squeeze(-1)),  # [B, T]
        },
        "grads": {
            "d_y": _tensor_payload(d_y),
            "d_residual": _tensor_payload(d_residual.numpy()),
            "d_x": _tensor_payload(d_x.numpy()),
            "d_weight": _tensor_payload(d_weight.numpy()),
        },
        "meta": {
            "B": B,
            "T": T,
            "C": C,
            "eps": eps,
            "note": "RMSNorm module: residual + x -> fused_residual_rmsnorm -> y",
        },
    }
    return [GoldenCase(op="rmsnorm_module", case="small_case_1", payload=payload)]


def gen_transformer_block() -> List[GoldenCase]:
    """Generate TransformerBlock golden test (attention + MLP with residuals)."""
    if torch is None:
        raise RuntimeError("PyTorch required for module golden generation")

    B, T = 2, 4
    C = 128  # hidden size
    M = 256  # MLP intermediate size (2x hidden for gate+up fused)
    Hq, Hkv = 4, 2  # num heads
    HD = C // Hq  # head dim = 32
    eps = 1e-5

    QKV = (Hq + 2 * Hkv) * HD

    # Initialize weights
    np.random.seed(52)
    x = np.random.randn(B, T, C).astype(np.float64) * 0.3
    ln1_weight = np.random.randn(C).astype(np.float64) * 0.1 + 1.0
    qkv_weight = np.random.randn(QKV, C).astype(np.float64) * 0.02
    out_weight = np.random.randn(C, Hq * HD).astype(np.float64) * 0.02
    ln2_weight = np.random.randn(C).astype(np.float64) * 0.1 + 1.0
    up_weight = np.random.randn(M, C).astype(np.float64) * 0.02
    down_weight = np.random.randn(C, M // 2).astype(np.float64) * 0.02

    freqs = _precompute_freqs_cis_complex(HD, T, 10000.0)
    pos_ids = np.arange(T, dtype=np.int32)

    # Forward with PyTorch
    x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    ln1_w_t = torch.tensor(ln1_weight, dtype=torch.float64, requires_grad=True)
    qkv_w_t = torch.tensor(qkv_weight, dtype=torch.float64, requires_grad=True)
    out_w_t = torch.tensor(out_weight, dtype=torch.float64, requires_grad=True)
    ln2_w_t = torch.tensor(ln2_weight, dtype=torch.float64, requires_grad=True)
    up_w_t = torch.tensor(up_weight, dtype=torch.float64, requires_grad=True)
    down_w_t = torch.tensor(down_weight, dtype=torch.float64, requires_grad=True)
    freqs_t = torch.tensor(freqs, dtype=torch.float64)

    # Pre-attention RMSNorm
    rms1 = torch.sqrt(torch.mean(x_t ** 2, dim=-1, keepdim=True) + eps)
    ln1_out = x_t / rms1 * ln1_w_t

    # QKV projection
    ln1_flat = ln1_out.reshape(B * T, C)
    qkv_flat = ln1_flat @ qkv_w_t.T
    qkv = qkv_flat.reshape(B, T, Hq + 2 * Hkv, HD)

    # RoPE
    qkv_rope = _torch_rope(qkv, freqs_t, pos_ids, Hq, Hkv, HD)

    # Attention
    attn_out, lse = _torch_flash_attention_ref(qkv_rope, B, T, Hq, Hkv, HD, causal=True)

    # Output projection + residual
    attn_flat = attn_out.reshape(B * T, Hq * HD)
    proj_out = attn_flat @ out_w_t.T
    attn_residual = x_t + proj_out.reshape(B, T, C)

    # Pre-MLP RMSNorm
    rms2 = torch.sqrt(torch.mean(attn_residual ** 2, dim=-1, keepdim=True) + eps)
    ln2_out = attn_residual / rms2 * ln2_w_t

    # MLP with SwiGLU
    ln2_flat = ln2_out.reshape(B * T, C)
    up_out = ln2_flat @ up_w_t.T  # [B*T, M]
    gate = up_out[:, :M // 2]
    up = up_out[:, M // 2:]
    swiglu_out = torch.nn.functional.silu(gate) * up  # [B*T, M//2]
    down_out = swiglu_out @ down_w_t.T  # [B*T, C]

    # Final residual
    out = attn_residual + down_out.reshape(B, T, C)

    # Backward pass
    d_out = np.random.randn(B, T, C).astype(np.float64) * 0.1
    d_out_t = torch.tensor(d_out, dtype=torch.float64)

    grads = torch.autograd.grad(
        out, (x_t, ln1_w_t, qkv_w_t, out_w_t, ln2_w_t, up_w_t, down_w_t),
        grad_outputs=d_out_t
    )
    d_x, d_ln1_w, d_qkv_w, d_out_w, d_ln2_w, d_up_w, d_down_w = grads

    payload = {
        "op": "transformer_block",
        "case": "small_case_1",
        "attrs": {"eps": eps, "causal": True},
        "inputs": {
            "x": _tensor_payload(x),
            "ln1_weight": _tensor_payload(ln1_weight),
            "qkv_weight": _tensor_payload(qkv_weight),
            "out_weight": _tensor_payload(out_weight),
            "ln2_weight": _tensor_payload(ln2_weight),
            "up_weight": _tensor_payload(up_weight),
            "down_weight": _tensor_payload(down_weight),
            "rope_freqs": _tensor_payload(freqs),
            "position_ids": _tensor_payload(pos_ids, dtype="int32"),
        },
        "outputs": {
            "out": _tensor_payload(out.detach().numpy()),
            "ln1_out": _tensor_payload(ln1_out.detach().numpy()),
            "attn_out": _tensor_payload(attn_out.detach().numpy().reshape(B, T, Hq * HD)),
            "attn_residual": _tensor_payload(attn_residual.detach().numpy()),
            "ln2_out": _tensor_payload(ln2_out.detach().numpy()),
        },
        "grads": {
            "d_out": _tensor_payload(d_out),
            "d_x": _tensor_payload(d_x.numpy()),
            "d_ln1_weight": _tensor_payload(d_ln1_w.numpy()),
            "d_qkv_weight": _tensor_payload(d_qkv_w.numpy()),
            "d_out_weight": _tensor_payload(d_out_w.numpy()),
            "d_ln2_weight": _tensor_payload(d_ln2_w.numpy()),
            "d_up_weight": _tensor_payload(d_up_w.numpy()),
            "d_down_weight": _tensor_payload(d_down_w.numpy()),
        },
        "meta": {
            "B": B,
            "T": T,
            "C": C,
            "M": M,
            "Hq": Hq,
            "Hkv": Hkv,
            "head_dim": HD,
            "eps": eps,
            "note": "TransformerBlock: x -> ln1 -> attn -> residual -> ln2 -> mlp -> residual -> out",
        },
    }
    return [GoldenCase(op="transformer_block", case="small_case_1", payload=payload)]


def gen_lm_head_module() -> List[GoldenCase]:
    """Generate LMHead module golden test (final norm + projection + loss)."""
    if torch is None:
        raise RuntimeError("PyTorch required for module golden generation")

    B, T = 2, 8
    C = 128  # hidden size
    V = 256  # vocab size
    eps = 1e-5

    # Initialize
    np.random.seed(53)
    x = np.random.randn(B, T, C).astype(np.float64) * 0.3
    ln_weight = np.random.randn(C).astype(np.float64) * 0.1 + 1.0
    lm_head_weight = np.random.randn(V, C).astype(np.float64) * 0.02
    targets = np.random.randint(0, V, size=(B, T), dtype=np.int32)

    # Forward with PyTorch
    x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    ln_w_t = torch.tensor(ln_weight, dtype=torch.float64, requires_grad=True)
    lm_w_t = torch.tensor(lm_head_weight, dtype=torch.float64, requires_grad=True)
    targets_t = torch.tensor(targets, dtype=torch.long)

    # Final RMSNorm
    rms = torch.sqrt(torch.mean(x_t ** 2, dim=-1, keepdim=True) + eps)
    ln_out = x_t / rms * ln_w_t

    # LM Head projection
    ln_flat = ln_out.reshape(B * T, C)
    logits = ln_flat @ lm_w_t.T  # [B*T, V]
    logits_3d = logits.reshape(B, T, V)

    # Cross-entropy loss
    loss = torch.nn.functional.cross_entropy(
        logits_3d.reshape(-1, V), targets_t.reshape(-1), reduction='mean'
    )

    # Backward pass
    d_loss = torch.ones_like(loss)
    d_x, d_ln_w, d_lm_w = torch.autograd.grad(
        loss, (x_t, ln_w_t, lm_w_t), grad_outputs=d_loss
    )

    payload = {
        "op": "lm_head_module",
        "case": "small_case_1",
        "attrs": {"eps": eps},
        "inputs": {
            "x": _tensor_payload(x),
            "ln_weight": _tensor_payload(ln_weight),
            "lm_head_weight": _tensor_payload(lm_head_weight),
            "targets": _tensor_payload(targets, dtype="int32"),
        },
        "outputs": {
            "ln_out": _tensor_payload(ln_out.detach().numpy()),
            "logits": _tensor_payload(logits_3d.detach().numpy()),
            "loss": _tensor_payload(np.array([loss.item()])),
        },
        "grads": {
            "d_x": _tensor_payload(d_x.numpy()),
            "d_ln_weight": _tensor_payload(d_ln_w.numpy()),
            "d_lm_head_weight": _tensor_payload(d_lm_w.numpy()),
        },
        "meta": {
            "B": B,
            "T": T,
            "C": C,
            "V": V,
            "eps": eps,
            "note": "LMHead module: x -> rmsnorm -> lm_head -> cross_entropy -> loss",
        },
    }
    return [GoldenCase(op="lm_head_module", case="small_case_1", payload=payload)]


def gen_llama_block() -> List[GoldenCase]:
    """Generate LlamaBlock golden test.

    Matches the structure from surogate/dsl/blocks/llama.py:
    - fused_residual_rmsnorm (pre-attention)
    - QKV projection (no bias)
    - RoPE (no QK-Norm)
    - FlashAttention (causal)
    - Output projection
    - fused_residual_rmsnorm (pre-MLP)
    - MLP with SwiGLU
    - Returns (out, residual_out)
    """
    if torch is None:
        raise RuntimeError("PyTorch required for block golden generation")

    B, T = 2, 4
    Hq, Hkv = 4, 2  # num_query_heads, num_kv_heads
    HD = 64  # head_size (must be 64 or 128 for CUDA attention kernel)
    C = Hq * HD  # d_model (hidden size) = 256
    M = 512  # d_ff (intermediate size for MLP), gate+up is 2*M
    eps = 1e-6
    max_seq = 64

    QKV = (Hq + 2 * Hkv) * HD
    AttnDim = Hq * HD
    MUp = 2 * M  # gate + up fused

    # Initialize weights
    np.random.seed(60)
    x = np.random.randn(B, T, C).astype(np.float64) * 0.3
    residual = np.random.randn(B, T, C).astype(np.float64) * 0.3
    ln1_weight = np.random.randn(C).astype(np.float64) * 0.1 + 1.0
    qkv_weight = np.random.randn(QKV, C).astype(np.float64) * 0.02
    out_weight = np.random.randn(C, AttnDim).astype(np.float64) * 0.02
    ln2_weight = np.random.randn(C).astype(np.float64) * 0.1 + 1.0
    mlp_up_weight = np.random.randn(MUp, C).astype(np.float64) * 0.02
    mlp_down_weight = np.random.randn(C, M).astype(np.float64) * 0.02

    freqs = _precompute_freqs_cis_complex(HD, max_seq, 10000.0)
    pos_ids = np.arange(T, dtype=np.int32)

    # Forward with PyTorch
    x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    residual_t = torch.tensor(residual, dtype=torch.float64, requires_grad=True)
    ln1_w_t = torch.tensor(ln1_weight, dtype=torch.float64, requires_grad=True)
    qkv_w_t = torch.tensor(qkv_weight, dtype=torch.float64, requires_grad=True)
    out_w_t = torch.tensor(out_weight, dtype=torch.float64, requires_grad=True)
    ln2_w_t = torch.tensor(ln2_weight, dtype=torch.float64, requires_grad=True)
    mlp_up_w_t = torch.tensor(mlp_up_weight, dtype=torch.float64, requires_grad=True)
    mlp_down_w_t = torch.tensor(mlp_down_weight, dtype=torch.float64, requires_grad=True)
    freqs_t = torch.tensor(freqs, dtype=torch.float64)

    # === Pre-attention fused_residual_rmsnorm ===
    # res_ffn = residual + x; ln1 = rmsnorm(res_ffn)
    res_ffn = residual_t + x_t
    rms1 = torch.sqrt(torch.mean(res_ffn ** 2, dim=-1, keepdim=True) + eps)
    ln1_out = res_ffn / rms1 * ln1_w_t

    # === QKV projection (no bias) ===
    ln1_flat = ln1_out.reshape(B * T, C)
    qkv_flat = ln1_flat @ qkv_w_t.T
    qkv = qkv_flat.reshape(B, T, Hq + 2 * Hkv, HD)

    # === RoPE (no QK-Norm) ===
    qkv_rope = _torch_rope(qkv, freqs_t, pos_ids, Hq, Hkv, HD)

    # === FlashAttention (causal) ===
    attn_out, lse = _torch_flash_attention_ref(qkv_rope, B, T, Hq, Hkv, HD, causal=True)

    # === Output projection ===
    attn_flat = attn_out.reshape(B * T, AttnDim)
    att_out_flat = attn_flat @ out_w_t.T
    att_out = att_out_flat.reshape(B, T, C)

    # === Pre-MLP fused_residual_rmsnorm ===
    # res_att = res_ffn + att_out; ln2 = rmsnorm(res_att)
    res_att = res_ffn + att_out
    rms2 = torch.sqrt(torch.mean(res_att ** 2, dim=-1, keepdim=True) + eps)
    ln2_out = res_att / rms2 * ln2_w_t

    # === MLP with SwiGLU ===
    ln2_flat = ln2_out.reshape(B * T, C)
    mlp_up_flat = ln2_flat @ mlp_up_w_t.T  # [B*T, MUp]
    mlp_up = mlp_up_flat.reshape(B, T, MUp)
    # SwiGLU: CUDA kernel expects [up, gate] order, so first half is up, second half is gate
    up = mlp_up[:, :, :M]
    gate = mlp_up[:, :, M:]
    swiglu_out = up * torch.nn.functional.silu(gate)  # [B, T, M]
    mlp_act_flat = swiglu_out.reshape(B * T, M)
    out_flat = mlp_act_flat @ mlp_down_w_t.T  # [B*T, C]
    out = out_flat.reshape(B, T, C)

    # Final outputs: (out, res_att) per LlamaBlock.forward signature
    # out is the MLP output, res_att is the residual for next layer

    # Backward pass
    d_out = np.random.randn(B, T, C).astype(np.float64) * 0.1
    d_res_att = np.random.randn(B, T, C).astype(np.float64) * 0.1
    d_out_t = torch.tensor(d_out, dtype=torch.float64)
    d_res_att_t = torch.tensor(d_res_att, dtype=torch.float64)

    # Combined loss for backward
    loss = (out * d_out_t).sum() + (res_att * d_res_att_t).sum()
    loss.backward()

    payload = {
        "op": "llama_block",
        "case": "small_case_1",
        "attrs": {"eps": eps, "causal": True},
        "inputs": {
            "x": _tensor_payload(x),
            "residual": _tensor_payload(residual),
            "ln1_weight": _tensor_payload(ln1_weight),
            "qkv_weight": _tensor_payload(qkv_weight),
            "out_weight": _tensor_payload(out_weight),
            "ln2_weight": _tensor_payload(ln2_weight),
            "mlp_up_weight": _tensor_payload(mlp_up_weight),
            "mlp_down_weight": _tensor_payload(mlp_down_weight),
            "rope_freqs": _tensor_payload(freqs.astype(np.float32), dtype="fp32"),
            "position_ids": _tensor_payload(pos_ids, dtype="int32"),
        },
        "outputs": {
            "out": _tensor_payload(out.detach().numpy()),
            "residual_out": _tensor_payload(res_att.detach().numpy()),
            # Intermediate outputs for debugging
            "res_ffn": _tensor_payload(res_ffn.detach().numpy()),
            "ln1": _tensor_payload(ln1_out.detach().numpy()),
            "qkv": _tensor_payload(qkv.detach().numpy()),
            "qkv_rope": _tensor_payload(qkv_rope.detach().numpy()),
            "att": _tensor_payload(attn_out.detach().numpy().reshape(B, T, AttnDim)),
            "att_out": _tensor_payload(att_out.detach().numpy()),
            "ln2": _tensor_payload(ln2_out.detach().numpy()),
            "mlp_up": _tensor_payload(mlp_up.detach().numpy()),
            "swiglu": _tensor_payload(swiglu_out.detach().numpy()),
        },
        "grads": {
            "d_out": _tensor_payload(d_out),
            "d_residual_out": _tensor_payload(d_res_att),
            "d_x": _tensor_payload(x_t.grad.numpy()),
            "d_residual": _tensor_payload(residual_t.grad.numpy()),
            "d_ln1_weight": _tensor_payload(ln1_w_t.grad.numpy()),
            "d_qkv_weight": _tensor_payload(qkv_w_t.grad.numpy()),
            "d_out_weight": _tensor_payload(out_w_t.grad.numpy()),
            "d_ln2_weight": _tensor_payload(ln2_w_t.grad.numpy()),
            "d_mlp_up_weight": _tensor_payload(mlp_up_w_t.grad.numpy()),
            "d_mlp_down_weight": _tensor_payload(mlp_down_w_t.grad.numpy()),
        },
        "meta": {
            "B": B,
            "T": T,
            "C": C,
            "M": M,
            "Hq": Hq,
            "Hkv": Hkv,
            "head_dim": HD,
            "max_seq": max_seq,
            "eps": eps,
            "note": "LlamaBlock: fused_residual_rmsnorm -> QKV -> RoPE -> FlashAttn -> fused_residual_rmsnorm -> SwiGLU MLP",
        },
    }
    return [GoldenCase(op="llama_block", case="small_case_1", payload=payload)]


def gen_qwen3_block() -> List[GoldenCase]:
    """Generate Qwen3Block golden test.

    Matches the structure from surogate/dsl/blocks/qwen3.py:
    - fused_residual_rmsnorm (pre-attention)
    - QKV projection (optional bias, we test without)
    - QK-Norm + RoPE (fused) - key difference from LlamaBlock
    - FlashAttention (causal)
    - Output projection
    - fused_residual_rmsnorm (pre-MLP)
    - MLP with SwiGLU
    - Returns (out, residual_out)
    """
    if torch is None:
        raise RuntimeError("PyTorch required for block golden generation")

    B, T = 2, 4
    Hq, Hkv = 4, 2  # num_query_heads, num_kv_heads
    HD = 64  # head_size (must be 64 or 128 for CUDA attention kernel)
    C = Hq * HD  # d_model (hidden size) = 256
    M = 512  # d_ff (intermediate size for MLP)
    eps = 1e-6
    max_seq = 64
    use_qk_norm = True

    QKV = (Hq + 2 * Hkv) * HD
    AttnDim = Hq * HD
    MUp = 2 * M  # gate + up fused

    # Initialize weights
    np.random.seed(61)
    x = np.random.randn(B, T, C).astype(np.float64) * 0.3
    residual = np.random.randn(B, T, C).astype(np.float64) * 0.3
    ln1_weight = np.random.randn(C).astype(np.float64) * 0.1 + 1.0
    qkv_weight = np.random.randn(QKV, C).astype(np.float64) * 0.02
    out_weight = np.random.randn(C, AttnDim).astype(np.float64) * 0.02
    ln2_weight = np.random.randn(C).astype(np.float64) * 0.1 + 1.0
    mlp_up_weight = np.random.randn(MUp, C).astype(np.float64) * 0.02
    mlp_down_weight = np.random.randn(C, M).astype(np.float64) * 0.02
    # QK-Norm weights
    q_norm_weight = np.random.randn(HD).astype(np.float64) * 0.1 + 1.0
    k_norm_weight = np.random.randn(HD).astype(np.float64) * 0.1 + 1.0

    freqs = _precompute_freqs_cis_complex(HD, max_seq, 10000.0)
    pos_ids = np.arange(T, dtype=np.int32)

    # Forward with PyTorch
    x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    residual_t = torch.tensor(residual, dtype=torch.float64, requires_grad=True)
    ln1_w_t = torch.tensor(ln1_weight, dtype=torch.float64, requires_grad=True)
    qkv_w_t = torch.tensor(qkv_weight, dtype=torch.float64, requires_grad=True)
    out_w_t = torch.tensor(out_weight, dtype=torch.float64, requires_grad=True)
    ln2_w_t = torch.tensor(ln2_weight, dtype=torch.float64, requires_grad=True)
    mlp_up_w_t = torch.tensor(mlp_up_weight, dtype=torch.float64, requires_grad=True)
    mlp_down_w_t = torch.tensor(mlp_down_weight, dtype=torch.float64, requires_grad=True)
    q_norm_w_t = torch.tensor(q_norm_weight, dtype=torch.float64, requires_grad=True)
    k_norm_w_t = torch.tensor(k_norm_weight, dtype=torch.float64, requires_grad=True)
    freqs_t = torch.tensor(freqs, dtype=torch.float64)

    # === Pre-attention fused_residual_rmsnorm ===
    res_ffn = residual_t + x_t
    rms1 = torch.sqrt(torch.mean(res_ffn ** 2, dim=-1, keepdim=True) + eps)
    ln1_out = res_ffn / rms1 * ln1_w_t

    # === QKV projection (no bias) ===
    ln1_flat = ln1_out.reshape(B * T, C)
    qkv_flat = ln1_flat @ qkv_w_t.T
    qkv = qkv_flat.reshape(B, T, Hq + 2 * Hkv, HD)

    # === QK-Norm + RoPE (fused) ===
    # Split QKV
    q = qkv[:, :, :Hq, :]  # [B, T, Hq, HD]
    k = qkv[:, :, Hq:Hq+Hkv, :]  # [B, T, Hkv, HD]
    v = qkv[:, :, Hq+Hkv:, :]  # [B, T, Hkv, HD]

    # Apply RMSNorm to Q and K per-head
    q_rms = torch.sqrt(torch.mean(q ** 2, dim=-1, keepdim=True) + eps)
    q_normed = q / q_rms * q_norm_w_t
    k_rms = torch.sqrt(torch.mean(k ** 2, dim=-1, keepdim=True) + eps)
    k_normed = k / k_rms * k_norm_w_t

    # Reassemble QKV for RoPE
    qkv_normed = torch.cat([q_normed, k_normed, v], dim=2)

    # Apply RoPE
    qkv_rope = _torch_rope(qkv_normed, freqs_t, pos_ids, Hq, Hkv, HD)

    # === FlashAttention (causal) ===
    attn_out, lse = _torch_flash_attention_ref(qkv_rope, B, T, Hq, Hkv, HD, causal=True)

    # === Output projection ===
    attn_flat = attn_out.reshape(B * T, AttnDim)
    att_out_flat = attn_flat @ out_w_t.T
    att_out = att_out_flat.reshape(B, T, C)

    # === Pre-MLP fused_residual_rmsnorm ===
    res_att = res_ffn + att_out
    rms2 = torch.sqrt(torch.mean(res_att ** 2, dim=-1, keepdim=True) + eps)
    ln2_out = res_att / rms2 * ln2_w_t

    # === MLP with SwiGLU ===
    ln2_flat = ln2_out.reshape(B * T, C)
    mlp_up_flat = ln2_flat @ mlp_up_w_t.T
    mlp_up = mlp_up_flat.reshape(B, T, MUp)
    # SwiGLU: CUDA kernel expects [up, gate] order, so first half is up, second half is gate
    up = mlp_up[:, :, :M]
    gate = mlp_up[:, :, M:]
    swiglu_out = up * torch.nn.functional.silu(gate)  # [B, T, M]
    mlp_act_flat = swiglu_out.reshape(B * T, M)
    out_flat = mlp_act_flat @ mlp_down_w_t.T
    out = out_flat.reshape(B, T, C)

    # Backward pass
    d_out = np.random.randn(B, T, C).astype(np.float64) * 0.1
    d_res_att = np.random.randn(B, T, C).astype(np.float64) * 0.1
    d_out_t = torch.tensor(d_out, dtype=torch.float64)
    d_res_att_t = torch.tensor(d_res_att, dtype=torch.float64)

    loss = (out * d_out_t).sum() + (res_att * d_res_att_t).sum()
    loss.backward()

    payload = {
        "op": "qwen3_block",
        "case": "small_case_1",
        "attrs": {"eps": eps, "causal": True, "use_qk_norm": use_qk_norm},
        "inputs": {
            "x": _tensor_payload(x),
            "residual": _tensor_payload(residual),
            "ln1_weight": _tensor_payload(ln1_weight),
            "qkv_weight": _tensor_payload(qkv_weight),
            "q_norm_weight": _tensor_payload(q_norm_weight),
            "k_norm_weight": _tensor_payload(k_norm_weight),
            "out_weight": _tensor_payload(out_weight),
            "ln2_weight": _tensor_payload(ln2_weight),
            "mlp_up_weight": _tensor_payload(mlp_up_weight),
            "mlp_down_weight": _tensor_payload(mlp_down_weight),
            "rope_freqs": _tensor_payload(freqs.astype(np.float32), dtype="fp32"),
            "position_ids": _tensor_payload(pos_ids, dtype="int32"),
        },
        "outputs": {
            "out": _tensor_payload(out.detach().numpy()),
            "residual_out": _tensor_payload(res_att.detach().numpy()),
            # Intermediate outputs for debugging
            "res_ffn": _tensor_payload(res_ffn.detach().numpy()),
            "ln1": _tensor_payload(ln1_out.detach().numpy()),
            "qkv": _tensor_payload(qkv.detach().numpy()),
            "qkv_rope": _tensor_payload(qkv_rope.detach().numpy()),
            "att": _tensor_payload(attn_out.detach().numpy().reshape(B, T, AttnDim)),
            "att_out": _tensor_payload(att_out.detach().numpy()),
            "ln2": _tensor_payload(ln2_out.detach().numpy()),
            "mlp_up": _tensor_payload(mlp_up.detach().numpy()),
            "swiglu": _tensor_payload(swiglu_out.detach().numpy()),
        },
        "grads": {
            "d_out": _tensor_payload(d_out),
            "d_residual_out": _tensor_payload(d_res_att),
            "d_x": _tensor_payload(x_t.grad.numpy()),
            "d_residual": _tensor_payload(residual_t.grad.numpy()),
            "d_ln1_weight": _tensor_payload(ln1_w_t.grad.numpy()),
            "d_qkv_weight": _tensor_payload(qkv_w_t.grad.numpy()),
            "d_q_norm_weight": _tensor_payload(q_norm_w_t.grad.numpy()),
            "d_k_norm_weight": _tensor_payload(k_norm_w_t.grad.numpy()),
            "d_out_weight": _tensor_payload(out_w_t.grad.numpy()),
            "d_ln2_weight": _tensor_payload(ln2_w_t.grad.numpy()),
            "d_mlp_up_weight": _tensor_payload(mlp_up_w_t.grad.numpy()),
            "d_mlp_down_weight": _tensor_payload(mlp_down_w_t.grad.numpy()),
        },
        "meta": {
            "B": B,
            "T": T,
            "C": C,
            "M": M,
            "Hq": Hq,
            "Hkv": Hkv,
            "head_dim": HD,
            "max_seq": max_seq,
            "eps": eps,
            "use_qk_norm": use_qk_norm,
            "note": "Qwen3Block: fused_residual_rmsnorm -> QKV -> QK-Norm+RoPE -> FlashAttn -> fused_residual_rmsnorm -> SwiGLU MLP",
        },
    }
    return [GoldenCase(op="qwen3_block", case="small_case_1", payload=payload)]


def gen_qwen3_model() -> List[GoldenCase]:
    """Generate Qwen3Model golden test (1-layer model).

    Matches the structure from surogate/dsl/models/qwen3.py:
    - Embedding lookup
    - Zero initialization for residual
    - Single Qwen3Block layer (simplified: no StackedBlocks call)
    - Final fused_residual_rmsnorm
    - Fused LM head + cross-entropy loss
    """
    if torch is None:
        raise RuntimeError("PyTorch required for model golden generation")

    # Model dimensions (small for testing)
    B, T = 2, 4
    V = 128  # vocab_size
    C = 256  # d_model (hidden size)
    Hq, Hkv = 4, 2  # num_query_heads, num_kv_heads
    HD = 64  # head_size (must be 64 or 128 for CUDA attention kernel)
    M = 512  # d_ff (intermediate size)
    eps = 1e-6
    max_seq = 64
    n_layers = 1  # Single layer for simplicity
    use_qk_norm = True

    QKV = (Hq + 2 * Hkv) * HD
    AttnDim = Hq * HD
    MUp = 2 * M

    # Initialize weights
    np.random.seed(70)

    # Model-level weights
    embedding_weight = np.random.randn(V, C).astype(np.float64) * 0.02
    final_norm_weight = np.random.randn(C).astype(np.float64) * 0.1 + 1.0
    lm_head_weight = np.random.randn(V, C).astype(np.float64) * 0.02

    # Block weights (layer 0)
    ln1_weight = np.random.randn(C).astype(np.float64) * 0.1 + 1.0
    qkv_weight = np.random.randn(QKV, C).astype(np.float64) * 0.02
    out_weight = np.random.randn(C, AttnDim).astype(np.float64) * 0.02
    q_norm_weight = np.random.randn(HD).astype(np.float64) * 0.1 + 1.0
    k_norm_weight = np.random.randn(HD).astype(np.float64) * 0.1 + 1.0
    ln2_weight = np.random.randn(C).astype(np.float64) * 0.1 + 1.0
    mlp_up_weight = np.random.randn(MUp, C).astype(np.float64) * 0.02
    mlp_down_weight = np.random.randn(C, M).astype(np.float64) * 0.02

    freqs = _precompute_freqs_cis_complex(HD, max_seq, 10000.0)

    # Inputs
    token_ids = np.random.randint(0, V, size=(B, T)).astype(np.int32)
    position_ids = np.arange(T, dtype=np.int32)
    targets = np.random.randint(0, V, size=(B, T)).astype(np.int32)

    # Forward pass with PyTorch
    embedding_w_t = torch.tensor(embedding_weight, dtype=torch.float64, requires_grad=True)
    final_norm_w_t = torch.tensor(final_norm_weight, dtype=torch.float64, requires_grad=True)
    lm_head_w_t = torch.tensor(lm_head_weight, dtype=torch.float64, requires_grad=True)

    ln1_w_t = torch.tensor(ln1_weight, dtype=torch.float64, requires_grad=True)
    qkv_w_t = torch.tensor(qkv_weight, dtype=torch.float64, requires_grad=True)
    out_w_t = torch.tensor(out_weight, dtype=torch.float64, requires_grad=True)
    q_norm_w_t = torch.tensor(q_norm_weight, dtype=torch.float64, requires_grad=True)
    k_norm_w_t = torch.tensor(k_norm_weight, dtype=torch.float64, requires_grad=True)
    ln2_w_t = torch.tensor(ln2_weight, dtype=torch.float64, requires_grad=True)
    mlp_up_w_t = torch.tensor(mlp_up_weight, dtype=torch.float64, requires_grad=True)
    mlp_down_w_t = torch.tensor(mlp_down_weight, dtype=torch.float64, requires_grad=True)
    freqs_t = torch.tensor(freqs, dtype=torch.float64)

    token_ids_t = torch.tensor(token_ids, dtype=torch.long)
    targets_t = torch.tensor(targets, dtype=torch.long)

    # === 1. Embedding lookup ===
    x0 = embedding_w_t[token_ids_t]  # [B, T, C]

    # === 2. Zero residual ===
    residual0 = torch.zeros(B, T, C, dtype=torch.float64)

    # === 3. Qwen3Block forward (single layer) ===
    # Pre-attention norm
    res_ffn = residual0 + x0
    rms1 = torch.sqrt(torch.mean(res_ffn ** 2, dim=-1, keepdim=True) + eps)
    ln1_out = res_ffn / rms1 * ln1_w_t

    # QKV projection
    ln1_flat = ln1_out.reshape(B * T, C)
    qkv_flat = ln1_flat @ qkv_w_t.T
    qkv = qkv_flat.reshape(B, T, Hq + 2 * Hkv, HD)

    # QK-Norm + RoPE
    q = qkv[:, :, :Hq, :]
    k = qkv[:, :, Hq:Hq+Hkv, :]
    v = qkv[:, :, Hq+Hkv:, :]

    q_rms = torch.sqrt(torch.mean(q ** 2, dim=-1, keepdim=True) + eps)
    q_normed = q / q_rms * q_norm_w_t
    k_rms = torch.sqrt(torch.mean(k ** 2, dim=-1, keepdim=True) + eps)
    k_normed = k / k_rms * k_norm_w_t

    qkv_normed = torch.cat([q_normed, k_normed, v], dim=2)
    qkv_rope = _torch_rope(qkv_normed, freqs_t, position_ids, Hq, Hkv, HD)

    # FlashAttention
    attn_out, _ = _torch_flash_attention_ref(qkv_rope, B, T, Hq, Hkv, HD, causal=True)

    # Output projection
    attn_flat = attn_out.reshape(B * T, AttnDim)
    att_out_flat = attn_flat @ out_w_t.T
    att_out = att_out_flat.reshape(B, T, C)

    # Pre-MLP norm
    res_att = res_ffn + att_out
    rms2 = torch.sqrt(torch.mean(res_att ** 2, dim=-1, keepdim=True) + eps)
    ln2_out = res_att / rms2 * ln2_w_t

    # MLP with SwiGLU
    ln2_flat = ln2_out.reshape(B * T, C)
    mlp_up_flat = ln2_flat @ mlp_up_w_t.T
    mlp_up = mlp_up_flat.reshape(B, T, MUp)
    # SwiGLU: CUDA kernel expects [up, gate] order
    up = mlp_up[:, :, :M]
    gate = mlp_up[:, :, M:]
    swiglu_out = up * torch.nn.functional.silu(gate)
    mlp_act_flat = swiglu_out.reshape(B * T, M)
    block_out_flat = mlp_act_flat @ mlp_down_w_t.T
    block_out = block_out_flat.reshape(B, T, C)

    # Block outputs
    x1 = block_out
    residual1 = res_att

    # === 4. Final fused_residual_rmsnorm ===
    residual_final = residual1 + x1
    rms_final = torch.sqrt(torch.mean(residual_final ** 2, dim=-1, keepdim=True) + eps)
    xF = residual_final / rms_final * final_norm_w_t

    # === 5. LM head + cross-entropy loss ===
    xF_flat = xF.reshape(B * T, C)
    logits = xF_flat @ lm_head_w_t.T  # [B*T, V]

    # Cross-entropy loss (per-token, no reduction for golden)
    log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    targets_flat = targets_t.reshape(-1)
    # Gather the log probabilities for the target tokens
    loss_per_token = -log_softmax[torch.arange(B * T), targets_flat]
    loss = loss_per_token.mean()

    # Backward pass
    loss.backward()

    payload = {
        "op": "qwen3_model",
        "case": "small_1layer",
        "attrs": {
            "eps": eps,
            "causal": True,
            "use_qk_norm": use_qk_norm,
            "n_layers": n_layers,
        },
        "inputs": {
            # Model weights
            "embedding": _tensor_payload(embedding_weight),
            "final_norm": _tensor_payload(final_norm_weight),
            "lm_head": _tensor_payload(lm_head_weight),
            # Block 0 weights
            "block0_ln1_weight": _tensor_payload(ln1_weight),
            "block0_qkv_weight": _tensor_payload(qkv_weight),
            "block0_out_weight": _tensor_payload(out_weight),
            "block0_q_norm_weight": _tensor_payload(q_norm_weight),
            "block0_k_norm_weight": _tensor_payload(k_norm_weight),
            "block0_ln2_weight": _tensor_payload(ln2_weight),
            "block0_mlp_up_weight": _tensor_payload(mlp_up_weight),
            "block0_mlp_down_weight": _tensor_payload(mlp_down_weight),
            "rope_freqs": _tensor_payload(freqs.astype(np.float32), dtype="fp32"),
            # Inputs
            "token_ids": _tensor_payload(token_ids, dtype="int32"),
            "position_ids": _tensor_payload(position_ids, dtype="int32"),
            "targets": _tensor_payload(targets, dtype="int32"),
        },
        "outputs": {
            # Intermediates
            "x0": _tensor_payload(x0.detach().numpy()),
            "block_out": _tensor_payload(block_out.detach().numpy()),
            "xF": _tensor_payload(xF.detach().numpy()),
            "logits": _tensor_payload(logits.detach().numpy()),
            "loss": _tensor_payload(np.array([loss.item()])),
        },
        "grads": {
            "d_embedding": _tensor_payload(embedding_w_t.grad.numpy()),
            "d_final_norm": _tensor_payload(final_norm_w_t.grad.numpy()),
            "d_lm_head": _tensor_payload(lm_head_w_t.grad.numpy()),
            "d_block0_ln1_weight": _tensor_payload(ln1_w_t.grad.numpy()),
            "d_block0_qkv_weight": _tensor_payload(qkv_w_t.grad.numpy()),
            "d_block0_out_weight": _tensor_payload(out_w_t.grad.numpy()),
            "d_block0_q_norm_weight": _tensor_payload(q_norm_w_t.grad.numpy()),
            "d_block0_k_norm_weight": _tensor_payload(k_norm_w_t.grad.numpy()),
            "d_block0_ln2_weight": _tensor_payload(ln2_w_t.grad.numpy()),
            "d_block0_mlp_up_weight": _tensor_payload(mlp_up_w_t.grad.numpy()),
            "d_block0_mlp_down_weight": _tensor_payload(mlp_down_w_t.grad.numpy()),
        },
        "meta": {
            "B": B,
            "T": T,
            "V": V,
            "C": C,
            "M": M,
            "Hq": Hq,
            "Hkv": Hkv,
            "head_dim": HD,
            "max_seq": max_seq,
            "n_layers": n_layers,
            "eps": eps,
            "use_qk_norm": use_qk_norm,
            "note": "Qwen3Model (1-layer): embedding -> Qwen3Block -> final_norm -> lm_head_loss",
        },
    }
    return [GoldenCase(op="qwen3_model", case="small_1layer", payload=payload)]


# Registry of all ops. Add generators as they are implemented.
OP_GENERATORS: Dict[str, Callable[[], List[GoldenCase]]] = {
    # Forward ops (primitives)
    "embedding": gen_embedding,
    "zeros": gen_zeros,
    "fused_residual_rmsnorm": gen_fused_residual_rmsnorm,
    "view": gen_view,
    "add": gen_add,
    "matmul": gen_matmul,
    "matmul_bias": gen_matmul_bias,
    "bias_add": gen_bias_add,
    "swiglu": gen_swiglu,
    "matmul_swiglu": gen_matmul_swiglu,
    "qkv_qk_norm_rope": gen_qkv_qk_norm_rope,
    "rope": gen_rope,
    "flash_attention": gen_flash_attention,
    "cross_entropy_loss": gen_cross_entropy_loss,
    "fused_lm_head_loss": gen_fused_lm_head_loss,
    # Backward ops (primitives)
    "view_backward": gen_view_backward,
    "add_backward": gen_add_backward,
    "matmul_backward": gen_matmul_backward,
    "bias_add_backward": gen_bias_add_backward,
    "swiglu_backward": gen_swiglu_backward,
    "matmul_swiglu_backward": gen_matmul_swiglu_backward,
    "rope_backward": gen_rope_backward,
    "qkv_qk_norm_rope_backward": gen_qkv_qk_norm_rope_backward,
    "flash_attention_backward": gen_flash_attention_backward,
    "zeros_backward": gen_zeros_backward,
    "fused_residual_rmsnorm_backward": gen_fused_residual_rmsnorm_backward,
    "embedding_backward": gen_embedding_backward,
    "cross_entropy_backward": gen_cross_entropy_backward,
    "fused_lm_head_loss_backward": gen_fused_lm_head_loss_backward,
    # Composed modules
    "swiglu_mlp": gen_swiglu_mlp,
    "gqa_attention": gen_gqa_attention,
    "embedding_module": gen_embedding_module,
    "rmsnorm_module": gen_rmsnorm_module,
    "transformer_block": gen_transformer_block,
    "lm_head_module": gen_lm_head_module,
    # DSL Blocks (full transformer blocks matching Python DSL definitions)
    "llama_block": gen_llama_block,
    "qwen3_block": gen_qwen3_block,
    # DSL Models (full models matching Python DSL definitions)
    "qwen3_model": gen_qwen3_model,
}


def _list_ops() -> None:
    print("Available ops:")
    for name in sorted(OP_GENERATORS.keys()):
        status = "ok" if OP_GENERATORS[name] else "TODO"
        print(f"  {name:30s} {status}")


def _resolve_ops(args_ops: List[str], run_all: bool) -> List[str]:
    if run_all:
        return sorted(OP_GENERATORS.keys())
    if not args_ops:
        raise SystemExit("No ops specified. Use --op or --all.")
    missing = [op for op in args_ops if op not in OP_GENERATORS]
    if missing:
        raise SystemExit(f"Unknown ops: {', '.join(missing)}")
    return args_ops


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate golden data for DSL ops")
    parser.add_argument("--op", action="append", default=[], help="Op name (can repeat)")
    parser.add_argument("--all", action="store_true", help="Generate all ops (implemented only)")
    parser.add_argument("--list", action="store_true", help="List ops and implementation status")
    parser.add_argument("--out-dir", default="tests/ops/goldens", help="Output directory")
    args = parser.parse_args()

    if args.list:
        _list_ops()
        return 0

    ops = _resolve_ops(args.op, args.all)
    wrote = 0
    skipped = []
    for op in ops:
        gen = OP_GENERATORS.get(op)
        if gen is None:
            skipped.append(op)
            continue
        cases = gen()
        for case in cases:
            path = _write_case(args.out_dir, case)
            print(f"wrote {path}")
            wrote += 1

    if skipped:
        print("\nSkipped (no generator yet):")
        for op in skipped:
            print(f"  {op}")

    if wrote == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
