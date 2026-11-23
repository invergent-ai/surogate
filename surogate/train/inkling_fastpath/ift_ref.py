# VENDORED UNCHANGED from the autograd copy of the Inkling-2-Mini reference, inkling_ref_ad.py (sha256 cc78cb8b9bb51998..., 2026-11-21):
# the semantic op definitions (the CPU reference + the autograd STE changes) the LEVEL-0 backward recomputes.
"""AUTOGRAD COPY (2026-11-21) of ref/inkling_ref.py (sha256 16d8dba0...). Forward arithmetic unchanged;
changes, all inert unless used (checked bit for bit against the original):
  (L1) rb() / r8() are straight-through autograd functions (identity backward in fp32); same forward values.
       ROUND["on"] = False turns both into identity (smooth fp32 function: finite-difference gradient checks only).
  (L2) InklingRef.kv_store_f32: keep the KV cache in fp32 storage (values already e4m3 / bf16 exact), so autograd
       never casts gradients to float8.
  (L3) InklingRef.forward(..., ckpt=True): one-chunk forward with activation checkpointing per visit
       (torch.utils.checkpoint, REENTRANT; each visit's VisitState is re-created fresh inside the checkpoint).
  (L4) ROOT = three levels up from this file.
  (L5) expert GEMMs with grad: _WeightMM (same forward mm; the backward rebuilds the frozen fp32 expert weight
       instead of saving it).

Inkling-2-Mini (inkling2_loop) CPU reference.

Pure PyTorch, CPU, batch 1. THE semantic definition of every Inkling op for the engine and the trainer (OPS.md
section 3 names). Written by reading the vLLM implementation in vllm/vllm-openai:inkling-2-mini-x86_64-7cb5dae
(ref/src/vllm/models/inkling2/nvidia/**), the only implementation that runs this checkpoint, following the sm_120
code paths (the RTX 5090 runs: `ops/attention_sm90.py`, `ops/moe_sm120.py`, `ops/conv_mix.py` sm_120 branch).

Conventions
- Tensors are fp32 holding bf16-representable values wherever vLLM stores bf16 (`rb()` = round to bf16, keep fp32).
  Golden vectors store those as real bf16.
- Every rounding point of the reference is reproduced (bf16 stores, e4m3 / e2m1 casts, fp32 elementwise order where
  vLLM fixes it). NOT reproduced: the GPU's accumulation order inside GEMMs / MMA, the PTX approximations
  (rsqrt.approx, ex2.approx, tanh.approx, rcp.approx), and the flash kernel's bf16 rounding of P before P@V. Here
  softmax is exact fp32 (or fp64 in EXACT mode).
- EXACT mode (`set_exact(True)`): GEMMs run in fixed 32-row blocks (fp32, bitwise independent of the batch shape),
  long reductions and the attention core accumulate in fp64 and round once to fp32. Prefill / chunked prefill /
  decode then agree up to ulp effects of torch's vectorised-vs-scalar elementwise paths: used by the cache
  self-check. Default (plain fp32) is faster.
- Visit numbering (OPS.md 1): v < 6 -> (layer v, pass 0); 6 <= v < 42 -> (6 + (v-6) % 12, (v-6) // 12);
  v >= 42 -> (v - 24, pass 0). Visit 48 = final-norm output.

Checked facts: NVFP4 outer_scale_denominator = 2688 = 6 * 448 and max|w| == amax (layer 0
expert 0); w13 rows interleave (gate j = row 2j, up j = row 2j + 1) as `ops/activation.py` `_situ` reads the bf16
gate_up pairs (sm_120 keeps the checkpoint row order: no shuffle); low nibble = even element.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

import torch
from safetensors import safe_open

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # (L4)
MODEL_DIR = os.path.join(ROOT, "model")
# Prebuilt exact bf16 expert values (raw files, mmapped read-only, page cache shared by all processes on a host);
# built by `python ref/inkling_ref.py build-expert-cache`. Optional: without it the experts are dequantised at load.
EXPERT_CACHE_DIR = os.environ.get("INKLING_EXPERT_CACHE_DIR", os.path.join(ROOT, "cache", "experts_bf16"))
FP32_MAX = 3.4028234663852886e38
D = 2304
LOOP_STREAMS = ("attn", "k", "v", "mlp")

# ----------------------------------------------------------------------------------------------------------------
# precision mode
# ----------------------------------------------------------------------------------------------------------------
_EXACT = False


def set_exact(flag: bool) -> None:
    """fp64 accumulation for matmuls / reductions (batch-shape invariant), see module doc."""
    global _EXACT
    _EXACT = bool(flag)


def is_exact() -> bool:
    return _EXACT


# vLLM-artifact emulation (OFF by default; the reference semantics are the ideal fp32-accumulation ones). Measured on
# the vLLM image's own kernels (capture_compare.py, splitk_search*.py; RTX 5090, prefill chunk M = 256):
#  - cuBLAS bf16 GEMMs with small N use split-K with bf16-ROUNDED partials summed in fp32: the separate K / V
#    projections (N 1024, K 2304: 5 parts of 512) of global layers and of looped local visits, and the dynamic-conv
#    tap projection (N 32: 8 parts of 320). The q-side GEMM, the fused local qkvr, wo_ud: plain fp32 accumulation.
#  - the prefill attention kernel rounds P = exp(s - running max) to bf16 per 64-key block before P @ V.
#  M <= 16 rows use vLLM's own gemv (fp32, no split). 17 <= M < 256: not measured (assumed as M = 256).
VLLM_EMU = {"on": False, "kv_split": 512, "dyn_split": 320, "ctrl_split": 0, "p_bf16_block": 64, "gemv_max_m": 16}


def set_vllm_emulation(on: bool, **kw) -> None:
    VLLM_EMU["on"] = bool(on)
    VLLM_EMU.update(kw)


def mm_splitk(x: torch.Tensor, w: torch.Tensor, chunk: int) -> torch.Tensor:
    """cuBLAS reduced-precision split-K: sum over K-chunks of bf16-rounded fp32 partials, summed in fp32."""
    k = x.shape[1]
    out = None
    for a in range(0, k, chunk):
        part = rb(mm(x[:, a : a + chunk], w[:, a : a + chunk]))
        out = part if out is None else out + part
    return out


ROUND = {"on": True}  # (L1) False: rb / r8 are identities (smooth fp32 function, FD gradient checks only)


class _RoundSTE(torch.autograd.Function):
    """(L1) forward: x.to(dt).to(fp32) exactly as the original; backward: identity (straight-through, no rounding of
    the gradient)."""

    @staticmethod
    def forward(ctx, x, dt):
        ctx.in_dtype = x.dtype
        return x.to(dt).to(torch.float32)

    @staticmethod
    def backward(ctx, g):
        return g.to(ctx.in_dtype), None


def rb(x: torch.Tensor) -> torch.Tensor:
    """Round to bf16, return fp32 (a bf16 store of the reference)."""
    if not ROUND["on"]:
        return x.to(torch.float32)
    if x.requires_grad:
        return _RoundSTE.apply(x, torch.bfloat16)
    return x.to(torch.bfloat16).to(torch.float32)


def r8(x: torch.Tensor) -> torch.Tensor:
    """Round to FP8 e4m3fn (RNE, saturating like cvt.rn.satfinite), return fp32."""
    if not ROUND["on"]:
        return x.to(torch.float32)
    if x.requires_grad:
        return _RoundSTE.apply(x, torch.float8_e4m3fn)
    return x.to(torch.float8_e4m3fn).to(torch.float32)


_ROW_BLOCK = 32


def _blocked(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """fp32 x [M, K] @ w.T in fixed 32-row blocks: each row's result is independent of M (the CPU sgemm kernel
    sees one shape), so prefill / chunked prefill / decode agree bitwise on GEMM outputs."""
    m = x.shape[0]
    mp = -(-m // _ROW_BLOCK) * _ROW_BLOCK
    xp = torch.zeros(mp, x.shape[1])
    xp[:m] = x
    wt = w.float().T
    out = torch.empty(mp, w.shape[0])
    for i in range(0, mp, _ROW_BLOCK):
        out[i : i + _ROW_BLOCK] = xp[i : i + _ROW_BLOCK] @ wt
    return out[:m]


def mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """fp32 `x @ w.T` (x [M, K], w [N, K]). EXACT mode: batch-shape invariant (fixed-row-block fp32 GEMM)."""
    if _EXACT:
        return _blocked(x.float(), w)
    return x.float() @ w.float().T


def bmm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """fp32 batched `x @ w.transpose(-1, -2)` (x [S, T, K], w [S, N, K])."""
    if _EXACT:
        return torch.stack([_blocked(x[i].float(), w[i]) for i in range(x.shape[0])])
    return x.float() @ w.float().transpose(-1, -2)


def rsum(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Reduction sum (fp64 in EXACT mode, rounded to fp32)."""
    if _EXACT:
        return x.double().sum(dim, keepdim=keepdim).float()
    return x.sum(dim, keepdim=keepdim)


def _tf(fn, x: torch.Tensor) -> torch.Tensor:
    """Elementwise transcendental; EXACT mode evaluates it in fp64 and rounds once to fp32, so torch's vectorised vs
    scalar-tail code paths (which differ by an ulp in fp32 and depend on the tensor's size) cannot differ."""
    if _EXACT and x.dtype == torch.float32:
        return fn(x.double()).float()
    return fn(x)


def t_rsqrt(x): return _tf(torch.rsqrt, x)
def t_sqrt(x): return _tf(torch.sqrt, x)
def t_exp(x): return _tf(torch.exp, x)
def t_log(x): return _tf(torch.log, x)
def t_log1p(x): return _tf(torch.log1p, x)
def t_tanh(x): return _tf(torch.tanh, x)
def t_sigmoid(x): return _tf(torch.sigmoid, x)


def fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """fp32 fused multiply-add (exact product, one rounding; via fp64)."""
    return (a.double() * b.double() + c.double()).float()


# ----------------------------------------------------------------------------------------------------------------
# small numeric helpers
# ----------------------------------------------------------------------------------------------------------------
_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_LUT = torch.cat([_E2M1_VALUES, -_E2M1_VALUES])  # code with sign in bit 3


def e2m1_rne(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Code (uint8, sign in bit 3) and fp32 value of `x` rounded to e2m1, nearest-even, saturating
    (`cvt.rn.satfinite.e2m1x2.f32`; thresholds as `ops/nvfp4.py` `_e2m1`)."""
    a = x.abs()
    code = torch.where(
        a <= 0.25, 0,
        torch.where(a < 0.75, 1,
        torch.where(a <= 1.25, 2,
        torch.where(a < 1.75, 3,
        torch.where(a <= 2.5, 4,
        torch.where(a < 3.5, 5,
        torch.where(a <= 5.0, 6, 7)))))))
    neg = torch.signbit(x)
    value = _E2M1_VALUES[code]
    value = torch.where(neg, -value, value)
    code = (code | (neg.to(torch.int64) << 3)).to(torch.uint8)
    return code, value


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K/2] (low nibble = even element) -> fp32 values [..., K]."""
    lo = _E2M1_LUT[(packed & 15).long()]
    hi = _E2M1_LUT[(packed >> 4).long()]
    return torch.stack((lo, hi), -1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """uint8 codes [..., K] -> packed [..., K/2], low nibble = even element."""
    c = codes.reshape(*codes.shape[:-1], -1, 2)
    return (c[..., 0] | (c[..., 1] << 4)).to(torch.uint8)


def e8m0_scale(exps: torch.Tensor) -> torch.Tensor:
    """uint8 e8m0 exponents -> fp32 2^(e-127); e == 0 -> 2^-127 (`ops/moe_sm120.py` `_dequant_e4m3`)."""
    e = exps.to(torch.int32)
    bits = torch.where(e == 0, torch.full_like(e, 0x00400000), e << 23)
    return bits.view(torch.float32)


# ----------------------------------------------------------------------------------------------------------------
# weight dequantisation (checkpoint formats)
# ----------------------------------------------------------------------------------------------------------------
def dequant_nvfp4_unit(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """NVFP4 weight [N, K/2] u8 + e4m3 block scales [N, K/16] -> fp32 [N, K] = e2m1 * scale (WITHOUT the per-expert
    global amax / outer_scale_denominator, which vLLM applies after the GEMM). Exact in bf16."""
    vals = unpack_e2m1(packed)
    s = scales.to(torch.float32).repeat_interleave(16, dim=-1)
    return vals * s


def nvfp4_global(amax: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Per-expert global weight scale amax / outer_scale_denominator (fp32); the full dequantised weight is
    dequant_nvfp4_unit(...) * nvfp4_global(...)."""
    return amax.float() / denominator.float()


def dequant_mxfp8(values: torch.Tensor, exps: torch.Tensor) -> torch.Tensor:
    """MXFP8 weight e4m3 [N, K] + e8m0 [N, K/32] -> fp32 [N, K] (exact in bf16)."""
    s = e8m0_scale(exps).repeat_interleave(32, dim=-1)
    return values.to(torch.float32) * s


# ----------------------------------------------------------------------------------------------------------------
# activation quantisation (per token row)
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class QuantNVFP4:
    codes: torch.Tensor  # uint8 [R, K/2] packed e2m1 (low nibble = even element)
    scales: torch.Tensor  # float8_e4m3fn [R, K/16]
    amax: torch.Tensor  # fp32 [R]
    values: torch.Tensor  # fp32 [R, K] = e2m1 * block scale (exact bf16), without the outer scale
    outer: torch.Tensor  # fp32 [R] = amax * (1/1536)


def _pairwise16(e: torch.Tensor) -> torch.Tensor:
    """Sum over the last dim (16) as a tree of adjacent pairs, fp32 (`ops/nvfp4.py` `_block_error`)."""
    assert e.shape[-1] == 16
    while e.shape[-1] > 1:
        e = e.reshape(*e.shape[:-1], -1, 2)
        e = e[..., 0] + e[..., 1]
    return e[..., 0]


def inkling_quant_nvfp4_4o6(x: torch.Tensor) -> QuantNVFP4:
    """Per-row NVFP4 with the "four-over-six" block rule (`ops/nvfp4.py` `_quantize_row` / `_quantize_block`,
    `ops/moe.py` `_quantize_x`, `ops/activation.py` SiTU quantisation).

    Row: amax = max|x|, enc = min(1536 / amax, FP32_MAX), dec = 1 / enc, outer = amax * (1 / 1536).
    16-block with max |x| bmax: s6 = e4m3(min(bmax * (enc * (1/6)), 448)), s4 = e4m3(min(bmax * (enc * 0.25), 448)),
    inv = min(1 / (s * dec), FP32_MAX); candidate values e2m1_rne(x * inv); per-element error
    |fma(v * s, outer, -x)| (FUSED, single rounding: the compiled kernel's form); block error = pairwise sum of adjacent
    pairs; the 4-candidate wins only if its error is strictly smaller.
    """
    x = x.float()
    r, k = x.shape
    assert k % 16 == 0
    amax = x.abs().amax(-1)
    enc = torch.clamp(torch.tensor(1536.0, dtype=torch.float32) / amax, max=FP32_MAX)
    dec = torch.tensor(1.0, dtype=torch.float32) / enc
    outer = amax * torch.tensor(1.0 / 1536.0, dtype=torch.float32)
    xb = x.reshape(r, k // 16, 16)
    bmax = xb.abs().amax(-1)
    e6 = enc * torch.tensor(1.0 / 6.0, dtype=torch.float32)
    e4 = enc * torch.tensor(0.25, dtype=torch.float32)
    s6 = r8(torch.clamp(bmax * e6[:, None], max=448.0))
    s4 = r8(torch.clamp(bmax * e4[:, None], max=448.0))
    inv6 = torch.clamp(1.0 / (s6 * dec[:, None]), max=FP32_MAX)
    inv4 = torch.clamp(1.0 / (s4 * dec[:, None]), max=FP32_MAX)
    c6, v6 = e2m1_rne(xb * inv6[..., None])
    c4, v4 = e2m1_rne(xb * inv4[..., None])
    # block error |fma(v * s, outer, -x)| (the compiled kernel FUSES the multiply-subtract: measured on the
    # capture of vLLM's kernels, 0 of 36,864 blocks differ with the fused form, 3 with the unfused one), pairwise sum
    o = outer[:, None, None]
    err6 = _pairwise16((v6 * s6[..., None]).double().mul(o.double()).sub(xb.double()).float().abs())
    err4 = _pairwise16((v4 * s4[..., None]).double().mul(o.double()).sub(xb.double()).float().abs())
    choose4 = err4 < err6
    codes = torch.where(choose4[..., None], c4, c6).reshape(r, k)
    vals = torch.where(choose4[..., None], v4, v6)
    scales = torch.where(choose4, s4, s6)
    values = (vals * scales[..., None]).reshape(r, k)
    return QuantNVFP4(pack_e2m1(codes), scales.to(torch.float8_e4m3fn), amax, values, outer)


@dataclass
class QuantMXFP8:
    values_e4m3: torch.Tensor  # float8_e4m3fn [R, K]
    exps: torch.Tensor  # uint8 e8m0 [R, K/32]
    values: torch.Tensor  # fp32 dequantised [R, K] (exact bf16)


def inkling_quant_mxfp8(x: torch.Tensor) -> QuantMXFP8:
    """Per-32-block MXFP8 (`ops/mxfp8.py` `_mxfp8_block`): raw = amax * (1/448); e8m0 exponent rounds raw UP to a
    power of two (bit trick); raw <= 2^-127 -> exponent 0 (scale 2^-127); values clamp(x * (1/scale), +-448) -> e4m3."""
    x = x.float()
    r, k = x.shape
    assert k % 32 == 0
    g = x.reshape(r, k // 32, 32)
    amax = g.abs().amax(-1)
    raw = amax * torch.tensor(1.0 / 448.0, dtype=torch.float32)
    bits = raw.view(torch.int32).to(torch.int64)
    exponent = ((bits + 0x007FFFFF) & 0x7F800000) >> 23
    min_scale = torch.tensor(2.0**-127, dtype=torch.float32)
    exponent = torch.where(raw <= min_scale, torch.zeros_like(exponent), torch.clamp(exponent, max=254))
    exponent = torch.where(torch.isnan(raw), torch.full_like(exponent, 255), exponent)
    scale = torch.maximum((exponent.to(torch.int32) << 23).view(torch.float32), min_scale)
    inverse = torch.where(exponent == 0, torch.tensor(2.0**127, dtype=torch.float32), 1.0 / scale)
    v = torch.clamp(g * inverse[..., None], -448.0, 448.0)
    q = v.to(torch.float8_e4m3fn)
    exps = exponent.to(torch.uint8)
    values = (q.to(torch.float32) * e8m0_scale(exps)[..., None]).reshape(r, k)
    return QuantMXFP8(q.reshape(r, k), exps, values)


def quant_fp8_per_token(x: torch.Tensor) -> torch.Tensor:
    """W8A8 option for the FP8 attention path (DESIGN 10.1, to be measured): per-token e4m3 with scale amax/448.
    Returns the dequantised fp32 values."""
    amax = x.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    s = amax / 448.0
    return r8(x / s) * s


def fp8_quant_weight(w: torch.Tensor, scheme: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP8 e4m3 weight quantisation candidates for the attention projections (DESIGN 10.1).

    scheme: 'tensor' (one scale), 'channel' (per output row), 'block1x128' (per output row x 128 input columns),
    'block128x128'. Scale = amax / 448 of the group (fp32; 1 for an all-zero group). Returns (e4m3 values [N, K],
    scales [N / rows, K / cols], dequantised fp32 [N, K])."""
    w = w.float()
    n, k = w.shape
    rows, cols = {"tensor": (n, k), "channel": (1, k), "block1x128": (1, 128), "block128x128": (128, 128)}[scheme]
    npad = -(-n // rows) * rows
    kpad = -(-k // cols) * cols
    wp = torch.nn.functional.pad(w, (0, kpad - k, 0, npad - n))
    amax = wp.reshape(npad // rows, rows, kpad // cols, cols).abs().amax((1, 3))
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    s_full = scale.repeat_interleave(rows, 0).repeat_interleave(cols, 1)[:n, :k]
    q = (w / s_full).to(torch.float8_e4m3fn)
    return q, scale, q.float() * s_full


# ----------------------------------------------------------------------------------------------------------------
# ops (OPS.md section 3 names)
# ----------------------------------------------------------------------------------------------------------------
def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float = 1e-6) -> torch.Tensor:
    """`ops/norm.py` `_rms_norm_kernel`: rstd = rsqrt(sum(x^2) / N + eps); y = (x * rstd) * w (fp32). Caller rounds."""
    x = x.float()
    n = x.shape[-1]
    rstd = t_rsqrt(rsum(x * x, -1, keepdim=True) / n + eps)
    y = x * rstd
    if weight is not None:
        y = y * weight.float()
    return y


def inkling_embed(ids: torch.Tensor, embed: torch.Tensor, norm_weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Embedding row (bf16) -> RMSNorm with `embed_norm` -> bf16 [T, 2304] (`Inkling2Model.embed_input_ids`)."""
    return rb(rms_norm(embed[ids].float(), norm_weight, eps))


def inkling_attn_res(
    stack: list[torch.Tensor] | torch.Tensor,
    last: torch.Tensor,
    addend: torch.Tensor | None,
    queries: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float = 1e-6,
    eps: float = 1e-6,
    normalize_values: bool = True,
) -> torch.Tensor:
    """Block attention residual mix (`ops/attn_res.py`).

    Sources x_n = [*stack (bf16), last (+ addend, fp32 sum)]; per pre-scaled fp32 query q:
    rstd_n = rsqrt(sum(x_n^2) / D + eps); logit_n = (q . x_n) * rstd_n; p = softmax_n;
    mix = sum_n (p_n * rstd_n) * x_n (normalize_values) ; with norm_weight: RMSNorm(mix) * w -> bf16 [Q, T, D],
    else the fp32 mixes [Q, T, D]."""
    srcs = [s.float() for s in stack]
    lst = last.float()
    if addend is not None:
        lst = lst + addend.float()
    srcs.append(lst)
    x = torch.stack(srcs)  # [N, T, D]
    d = x.shape[-1]
    rstd = t_rsqrt(rsum(x * x, -1) / d + eps)  # [N, T]
    q = queries.float().reshape(-1, d)
    if _EXACT:
        dots = torch.einsum("qd,ntd->qnt", q.double(), x.double()).float()
    else:
        dots = torch.einsum("qd,ntd->qnt", q, x)
    logits = dots * rstd[None]
    probs = t_exp(logits - logits.amax(1, keepdim=True))  # [Q, N, T]
    inv_sum = 1.0 / rsum(probs, 1)  # [Q, T]
    p = probs * rstd[None] if normalize_values else probs
    acc = torch.zeros((q.shape[0],) + x.shape[1:], dtype=torch.float32)
    for n in range(x.shape[0]):
        acc = fma(p[:, n, :, None], x[n][None], acc)
    mix = acc * inv_sum[..., None]
    if norm_weight is None:
        return mix
    w = norm_weight.float().reshape(-1, d)
    out_rstd = t_rsqrt(rsum(mix * mix, -1, keepdim=True) / d + norm_eps)
    return rb(mix * out_rstd * w[:, None, :])


def inkling_loop_blend(
    dest: torch.Tensor,
    message: torch.Tensor,
    down: torch.Tensor,
    gain_forget: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Loop message controller + RMSNorm (`loop.py` `Inkling2LoopDecoderLayer.blend`, `ops/loop.py`).

    dest fp32 [S|1, T, D] (broadcast over streams), message fp32 [S, T, D], down bf16 [S, r, 2D],
    gain_forget bf16 [S, 2D, r] (rows: gain then forget), norm_weight [S|1, D] -> bf16 [S, T, D]:
    pair = [bf16(dest * rsqrt(mean(dest^2) + 1e-6)), bf16(msg * rsqrt(mean(msg^2) + 1e-6))];
    h = bf16(pair @ down^T); gf = fp32(h @ gain_forget^T); x = (1 - tanh(forget)) * dest + tanh(gain) * msg;
    y = bf16(x * rsqrt(mean(x^2) + eps) * w)."""
    s, t, d = message.shape
    dest = dest.float().expand(s, t, d)
    msg = message.float()
    rd = t_rsqrt(rsum(dest * dest, -1, keepdim=True) / d + 1e-6)
    rm = t_rsqrt(rsum(msg * msg, -1, keepdim=True) / d + 1e-6)
    pair = torch.cat((rb(dest * rd), rb(msg * rm)), -1)
    if VLLM_EMU["on"] and VLLM_EMU["ctrl_split"] and t > VLLM_EMU["gemv_max_m"]:
        hidden = rb(torch.stack([mm_splitk(pair[i], down[i].float(), VLLM_EMU["ctrl_split"]) for i in range(s)]))
    else:
        hidden = rb(bmm(pair, down.float()))
    gf = bmm(hidden, gain_forget.float())
    gain = t_tanh(gf[..., :d])
    forget = t_tanh(gf[..., d:])
    x = (1.0 - forget) * dest + gain * msg
    rstd = t_rsqrt(rsum(x * x, -1, keepdim=True) / d + norm_eps)
    w = norm_weight.float().reshape(-1, 1, d).expand(s, 1, d)
    return rb((x * rstd) * w)


def inkling_short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor | None,
    addend: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Static causal depthwise conv, kernel 4 (`ops/sconv.py` `_sconv_kernel` / `_tap_sum`).

    x [T, C] (rounded to bf16 = the conv-state dtype), weight [C, 4] (oldest tap first), state [3, C] = the
    previous 3 inputs (zeros before position 0). y = (sum_k w[c, k] * x[t-3+k] (fma, k = 0..3) + x[t]) * 5^-0.5,
    rounded to bf16. With addend: out = bf16((residual + addend) + y_bf16) in fp32. Returns (out, new_state)."""
    t, c = x.shape
    xs = rb(x)
    if state is None:
        state = torch.zeros(3, c)
    full = torch.cat((state.float(), xs), 0)
    w = rb(weight.float())
    acc = torch.zeros(t, c)
    for k in range(4):
        acc = fma(full[k : k + t], w[:, k][None], acc)
    scale = torch.tensor(5**-0.5, dtype=torch.float32)
    y = rb((acc + xs) * scale)
    if addend is not None:
        a = addend.float()
        if residual is not None:
            a = residual.float() + a
        y = rb(a + y)
    return y, full[-3:].clone()


def inkling_dynamic_conv(
    x: torch.Tensor,
    static_w: torch.Tensor,
    dyn: torch.Tensor,
    state: torch.Tensor | None,
    head_dim: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic causal conv of the K / V streams of GLOBAL layers (`_tap_sum` with dyn_ptr).

    x [T, C] (-> bf16), static_w [4, C] (lag 0 = current token), dyn [T, H*4] = bf16(weight_proj(x_k or x_v)),
    tap(t, c, lag) = static_w[lag, c] + dyn[t, (c // 128) * 4 + lag];
    y = (x[t]*tap0 then fma lags 1..3, + x[t]) * 9^-0.5 -> bf16. Returns (y, new_state)."""
    t, c = x.shape
    xs = rb(x)
    if state is None:
        state = torch.zeros(3, c)
    full = torch.cat((state.float(), xs), 0)  # full[3 + t - lag] = x[t - lag]
    heads = c // head_dim
    dyn4 = dyn.float().reshape(t, heads, 4)
    acc = None
    for lag in range(4):
        taps = static_w.float()[lag].reshape(1, heads, head_dim) + dyn4[:, :, lag : lag + 1]
        taps = taps.reshape(t, c)
        xv = full[3 - lag : 3 - lag + t]
        acc = xv * taps if lag == 0 else fma(xv, taps, acc)
    scale = torch.tensor(9**-0.5, dtype=torch.float32)
    y = rb((acc + xs) * scale)
    return y, full[-3:].clone()


def l2_clip(x: torch.Tensor, scale: torch.Tensor | None, threshold: float, eps: float = 1e-6) -> torch.Tensor:
    """Per-head L2 clip over the last dim (`ops/norm.py` `_l2_clip`, `_kv_conv_tile` K branch):
    x = x * scale (fp32); y = x * min(threshold / (||x|| + eps), 1). fp32 out (caller rounds)."""
    x = x.float()
    if scale is not None:
        x = x * scale.float()
    norm = t_sqrt(rsum(x * x, -1, keepdim=True))
    return x * torch.clamp(threshold / (norm + eps), max=1.0)


def inkling_log_tau(positions: torch.Tensor, n_floor: int = 8192, alpha: float = 0.1) -> torch.Tensor:
    """`ops/attention.py` `compute_log_scaling_tau`: fp32 1 + alpha * log(max((pos + 1) * (1/floor), 1))."""
    ratio = torch.clamp((positions + 1).to(torch.float32) * torch.tensor(1.0 / n_floor, dtype=torch.float32), min=1.0)
    return 1.0 + torch.tensor(alpha, dtype=torch.float32) * t_log(ratio)


def kv_store(x: torch.Tensor, kv_fp8: bool) -> torch.Tensor:
    """Value written to the KV cache from the fp32 clipped / normalised K or V: unit-scale e4m3 (FP8 KV) or bf16."""
    return r8(x) if kv_fp8 else rb(x)


@dataclass
class PrologueOut:
    q: torch.Tensor  # bf16 values [T, 24, 128]
    r: torch.Tensor  # bf16 values [T, 24, 16]: clipped (and, global, tau-scaled) r; rel = bf16(r @ rel_proj)
    rel: torch.Tensor  # bf16 values [T, 24, 1024]: bias at distance d = pos_q - pos_k (dense, unsheared)
    k: torch.Tensor  # KV-cache values [T, 8, 128] (e4m3 or bf16 values in fp32)
    v: torch.Tensor
    k_state: torch.Tensor
    v_state: torch.Tensor


def inkling_attention_prologue(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    r_scale: torch.Tensor,
    tau: torch.Tensor | None,
    rel_proj: torch.Tensor,
    k_conv: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    v_conv: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    k_state: torch.Tensor | None,
    v_state: torch.Tensor | None,
    kv_fp8: bool,
    qk_clip: float = 22.627416997969522,
    rel_clip: float = 8.0,
    eps: float = 1e-6,
) -> PrologueOut:
    """Attention input processing (`ops/sconv.py` `attention_prologue`, `ops/norm.py` `qr_clip_rel_tokens`).

    q [T, 24, 128], r [T, 24, 16], k / v [T, 1024] (bf16 projections). Conv: static weight [1024, 4], or dynamic
    (static_w [4, 1024], dyn [T, 32]). Q: bf16(l2clip(q * q_scale, 22.63)), global: bf16(bf16(.) * tau).
    R: same with r_scale, 8.0. rel[t, h, d] = bf16(sum_j r[t,h,j] * rel_proj[j, d]) (fp32 accumulate).
    K: bf16(conv) -> l2clip(., 22.63) -> KV store. V: bf16(conv) -> * rsqrt(mean(v^2) + eps) -> KV store."""
    qc = l2_clip(q, q_scale, qk_clip, eps)
    rc = l2_clip(r, r_scale, rel_clip, eps)
    if tau is not None:
        qc = rb(qc) * tau[:, None, None]
        rc = rb(rc) * tau[:, None, None]
    q_out = rb(qc)
    r_out = rb(rc)
    rel = rb(torch.einsum("thj,jd->thd", r_out.double() if _EXACT else r_out, rel_proj.double() if _EXACT else rel_proj.float()).float())
    if isinstance(k_conv, tuple):
        kc, k_state = inkling_dynamic_conv(k, k_conv[0], k_conv[1], k_state)
        vc, v_state = inkling_dynamic_conv(v, v_conv[0], v_conv[1], v_state)
    else:
        kc, k_state = inkling_short_conv(k, k_conv, k_state)
        vc, v_state = inkling_short_conv(v, v_conv, v_state)
    t = k.shape[0]
    kh = kc.reshape(t, -1, 128)
    vh = vc.reshape(t, -1, 128)
    k_out = kv_store(l2_clip(kh, None, qk_clip, eps), kv_fp8)
    v_out = kv_store(vh * t_rsqrt(rsum(vh * vh, -1, keepdim=True) / 128 + eps), kv_fp8)
    return PrologueOut(q_out, r_out, rel, k_out, v_out, k_state, v_state)


def inkling_rel_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel: torch.Tensor,
    pos_q: torch.Tensor,
    pos_k: torch.Tensor,
    window: int,
    scale: float = 128**-0.5,
    rel_extent: int = 1024,
    q_block: int = 128,
    p_bf16_block: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Relative-bias GQA attention (`ops/attention_sm90.py` `_rel_attention_kernel`, the sm_120 path).

    q [Tq, 24, 128] bf16 values, k / v [Nk, 8, 128] (KV values), rel [Tq, 24, 1024], query head h reads KV head
    h // 3. score = (q . k) * 128^-0.5 + bias, bias = rel[t, h, pos_q - pos_k] for 0 <= d < 1024, 0 for d >= 1024
    (global); masked: d < 0, and d >= window for local layers (window 1024 = keys within 1023 back + self).
    Exact softmax (fp32; fp64 in EXACT mode). Returns out fp32 [Tq, 24, 128] and natural-log lse fp32 [Tq, 24]."""
    tq, hq, dh = q.shape
    nk, hk, _ = k.shape
    g = hq // hk
    dt = torch.float64 if _EXACT else torch.float32
    kk = k.to(dt).permute(1, 0, 2)  # [Hk, Nk, D]
    vv = v.to(dt).permute(1, 0, 2)
    out = torch.empty(tq, hq, dh)
    lse = torch.empty(tq, hq)
    for b0 in range(0, tq, q_block):
        b1 = min(tq, b0 + q_block)
        pq = pos_q[b0:b1]
        lo_pos = int(pq.min())
        hi_pos = int(pq.max())
        sel = pos_k <= hi_pos
        if window > 0:
            sel &= pos_k > lo_pos - window
        idx = torch.nonzero(sel).flatten()
        k0, k1 = int(idx.min()), int(idx.max()) + 1
        pk = pos_k[k0:k1]
        qq = q[b0:b1].to(dt).reshape(b1 - b0, hk, g, dh).permute(1, 2, 0, 3)  # [Hk, G, B, D]
        s = torch.einsum("hgbd,hkd->hgbk", qq, kk[:, k0:k1]) * scale
        dist = pq[:, None] - pk[None, :]  # [B, K]
        gi = dist.clamp(0, rel_extent - 1)
        rr = rel[b0:b1].to(dt).permute(1, 0, 2)  # [Hq, B, E]
        bias = torch.gather(rr, 2, gi[None].expand(hq, -1, -1))  # [Hq, B, K]
        bias = torch.where((dist >= 0) & (dist < rel_extent), bias, torch.zeros((), dtype=dt))
        s = s + bias.reshape(hk, g, b1 - b0, -1)
        vis = dist >= 0
        if window > 0:
            vis &= dist < window
        s = s.masked_fill(~vis, float("-inf"))
        m = s.amax(-1, keepdim=True)
        p = t_exp(s - m)
        l = p.sum(-1, keepdim=True)
        if p_bf16_block:
            # vLLM's flash loop (vLLM emulation only): P = exp(s - running max) rounded to bf16 per key block of
            # p_bf16_block keys (absolute key index blocks), rescaled to the final max in fp32; l stays unrounded.
            nb = p_bf16_block
            a0 = (k0 // nb) * nb
            pad_l = k0 - a0
            kk_n = s.shape[-1] + pad_l
            pad_r = -kk_n % nb
            sp = torch.nn.functional.pad(s, (pad_l, pad_r), value=float("-inf"))
            bm = sp.reshape(*sp.shape[:-1], -1, nb).amax(-1)
            run = torch.cummax(bm, -1).values
            run = run.repeat_interleave(nb, -1)[..., pad_l : pad_l + s.shape[-1]]
            run = torch.where(torch.isfinite(run), run, m.expand_as(run))
            pr = t_exp(s - run).float().to(torch.bfloat16).to(dt) * t_exp(run - m)
            o = torch.einsum("hgbk,hkd->hgbd", pr, vv[:, k0:k1]) / l
        else:
            o = torch.einsum("hgbk,hkd->hgbd", p, vv[:, k0:k1]) / l
        ls = (m + t_log(l))[..., 0]
        out[b0:b1] = o.permute(2, 0, 1, 3).reshape(b1 - b0, hq, dh).float()
        lse[b0:b1] = ls.permute(2, 0, 1).reshape(b1 - b0, hq).float()
    return out, lse


def inkling_gate_sink(out: torch.Tensor, lse: torch.Tensor, gate: torch.Tensor, sink: torch.Tensor) -> torch.Tensor:
    """`ops/attention.py` `_gate_sink_kernel`: bf16((out * sigmoid(gate)) * sigmoid(lse - sink)) -> [T, 24*128].
    (The sink is a per-token per-head extra logit with zero value: sigmoid(lse - sink) = its softmax complement.)"""
    t, h, d = out.shape
    o = out.float() * t_sigmoid(gate.float().reshape(t, h, d))
    o = o * t_sigmoid(lse.float() - sink.float())[..., None]
    return rb(o.reshape(t, h * d))


def inkling_attn_linear(x: torch.Tensor, weight: torch.Tensor, act_fp8: bool = False) -> torch.Tensor:
    """bf16 projection with fp32 accumulation -> bf16 (`ops/linear.py` `linear`; cuBLAS / gemv). `weight` is the fp32
    weight as served: the bf16 checkpoint weight (oracle path) or the FP8-dequantised weight (fp8_quant_weight).
    act_fp8: per-token e4m3 activations (W8A8 candidate)."""
    xx = quant_fp8_per_token(x.float()) if act_fp8 else x
    return rb(mm(xx, weight))


def inkling_route(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, topk: int = 8, route_scale: float = 8.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Router (`ops/moe.py` `_select_experts`, `_selection_keys`, `_routing_weights`).

    logits = fp32(x_bf16 @ W_bf16^T) (summed over 1024-wide K groups in order); key = sigmoid(logit) + bias;
    top-8 by key descending, ties -> lower expert id; weights = exp(logsig(l_i) - lse_8(logsig)) * 8.0 with the
    8-sum in the order ((x0+x2)+(x1+x3))+((x4+x6)+(x5+x7)). Returns ids [T, 8] (key order), weights fp32 [T, 8],
    logits fp32 [T, 128]."""
    x = x.float()
    if _EXACT:
        logits = mm(x, weight)
    else:
        k = x.shape[1]
        logits = None
        for g0 in range(0, k, 1024):
            part = mm(x[:, g0 : g0 + 1024], weight[:, g0 : g0 + 1024])
            logits = part if logits is None else logits + part
    sig = 1.0 / (1.0 + t_exp(-logits))
    sel = sig + bias.float()[None]
    order = torch.sort(sel, dim=-1, descending=True, stable=True).indices[:, :topk]
    return order, route_weights(logits, order, route_scale), logits


def _router_logits(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    k = x.shape[1]
    logits = None
    for g0 in range(0, k, 1024):
        part = mm(x[:, g0 : g0 + 1024], weight[:, g0 : g0 + 1024])
        logits = part if logits is None else logits + part
    return logits


def route_weights(logits: torch.Tensor, order: torch.Tensor, route_scale: float = 8.0) -> torch.Tensor:
    """Routing weights of the chosen experts `order` [T, k] (in slot order) from the router logits (fp32): normalised
    sigmoids via log-sigmoid / logsumexp, x route_scale; 8-sum order ((x0+x2)+(x1+x3))+((x4+x6)+(x5+x7)). Used by
    routing REPLAY too (given ids, own router -> weights)."""
    topk = order.shape[1]
    lsel = torch.gather(logits, 1, order)
    lp = torch.clamp(lsel, max=0.0) - t_log1p(t_exp(-lsel.abs()))
    mx = lp.amax(-1, keepdim=True)
    sh = t_exp(lp - mx)
    if topk == 8:
        total = ((sh[:, 0] + sh[:, 2]) + (sh[:, 1] + sh[:, 3])) + ((sh[:, 4] + sh[:, 6]) + (sh[:, 5] + sh[:, 7]))
    else:
        total = sh.sum(-1)
    lse = mx[:, 0] + t_log(total)
    return t_exp(lp - lse[:, None]) * torch.tensor(route_scale, dtype=torch.float32)


def inkling_softcap_swiglu(gate_up: torch.Tensor, gamma: torch.Tensor, gate_cap: float = 4.0, up_cap: float = 25.0) -> torch.Tensor:
    """`ops/activation.py` `_situ`: gate_up bf16 [R, 2*I] with INTERLEAVED columns (2j = gate j, 2j+1 = up j);
    sig = sigmoid(g) (raw gate); g' = 4 tanh(g * 0.25); u' = 25 tanh(u / 25);
    y = bf16(((g' * sig) * u') * gamma) [R, I] (gamma = routing weight, applied before the down projection)."""
    gu = gate_up.float()
    g = gu[:, 0::2]
    u = gu[:, 1::2]
    sig = t_sigmoid(g)
    gc = gate_cap * t_tanh(g * (1.0 / gate_cap)) if gate_cap == 4.0 else gate_cap * t_tanh(g / gate_cap)
    uc = up_cap * t_tanh(u / up_cap)
    return rb(((gc * sig) * uc) * gamma.float()[:, None])


_LUT2 = torch.stack((_E2M1_LUT[torch.arange(256) & 15], _E2M1_LUT[torch.arange(256) >> 4]), -1)  # byte -> (lo, hi)


@dataclass
class MoEWeights:
    fmt: str  # 'nvfp4' | 'mxfp8'
    w13: torch.Tensor | None  # [E, 2I, H] exact values: NVFP4 e2m1*scale (no global), MXFP8 dequantised; bf16 storage
    w2: torch.Tensor | None  # [E, H, I]
    g13: torch.Tensor | None  # NVFP4 per-expert amax (fp32 [E])
    g2: torch.Tensor | None
    den13: torch.Tensor | None  # NVFP4 outer_scale_denominator (fp32 [1])
    den2: torch.Tensor | None
    gate_w: torch.Tensor  # router bf16 values [E, H] (fp32)
    gate_b: torch.Tensor  # fp32 [E]
    src: dict | None = None  # expert_mode "stream": name -> (weight slice, scale slice) of the checkpoint (mmap)

    def mat(self, name: str, e: int) -> torch.Tensor:
        """fp32 exact values of expert e's `w13` / `w2` (without the NVFP4 global)."""
        if self.src is None:
            return getattr(self, name)[e].float()
        ws, ss = self.src[name]
        q, sc = ws[e], ss[e]
        if self.fmt == "nvfp4":
            return _LUT2[q.long()].reshape(q.shape[0], -1) * sc.float().repeat_interleave(16, -1)
        return q.float() * e8m0_scale(sc).repeat_interleave(32, -1)


def make_lora(num_experts: int = 128, hidden: int = D, inter: int = 1152, rank: int = 16, alpha: float = 32.0,
              seed: int = 0, std: float = 0.02) -> dict:
    """Random expert LoRA (fixed seed; bf16-representable values) for the MoE LoRA vectors."""
    g = torch.Generator().manual_seed(seed)
    f = lambda *sh: rb(torch.randn(*sh, generator=g) * std)
    return {"w13": (f(num_experts, rank, hidden), f(num_experts, 2 * inter, rank)),
            "w2": (f(num_experts, rank, inter), f(num_experts, hidden, rank)), "scale": alpha / rank, "rank": rank,
            "seed": seed}


def _row_scale(amax: torch.Tensor, w_amax: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """NVFP4 GEMM output scale (`ops/moe.py` `_row_scales`): (x_amax * w_amax) * rcp(1536 * denominator)."""
    rcp = 1.0 / (torch.tensor(1536.0, dtype=torch.float32) * den.float())
    return (amax.float() * w_amax.float()) * rcp


class _WeightMM(torch.autograd.Function):
    """(L5) mm(a, W) for a frozen expert weight W = getter(): the backward rebuilds W instead of keeping the fp32 copy
    alive (128 experts x 48 visits of fp32 copies would be ~190 GB); same forward op."""

    @staticmethod
    def forward(ctx, a, getter):
        ctx.getter = getter
        return mm(a, getter())

    @staticmethod
    def backward(ctx, g):
        return g @ ctx.getter().float(), None


def _wmm(a: torch.Tensor, getter, name: str) -> torch.Tensor:
    if torch.is_grad_enabled() and a.requires_grad:
        return _WeightMM.apply(a, lambda: getter(name))
    return mm(a, getter(name))


def inkling_moe(
    x: torch.Tensor,
    w: MoEWeights,
    act_quant: bool = True,
    routing: tuple[torch.Tensor, torch.Tensor] | None = None,
    record: dict | None = None,
    lora: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Routed experts (`moe.py` `Inkling2Experts.forward`, sm_120 GEMMs `ops/moe_sm120.py`).

    act_quant=True (the reference): NVFP4 layers W4A4: x row quantised (four-over-six), gate_up =
    bf16(sum(a * w) * rowscale) with rowscale = (amax_x * amax_w[e]) * rcp(1536 * den); SiTU (gamma inside);
    re-quantised; down likewise. MXFP8 layers W8A8: MXFP8 activations, bf16(dequant(a) . dequant(w)).
    act_quant=False (W4A16 / W8A16, for comparisons): bf16 activations, weights dequantised in fp32 incl. the global.
    Combine: fp32 sum over the 8 slots in key order, rounded once to bf16. Returns (out [T, H], ids [T, 8], gammas).

    lora (our engine's expert LoRA; vLLM has none on NVFP4 / MXFP8 experts): {"w13": (A [E, r, H], B [E, 2I, r]),
    "w2": (A [E, r, I], B [E, H, r]), "scale": alpha / r}; the delta scale * ((a @ A_e^T) @ B_e^T) is computed in
    fp32 from the UNQUANTISED bf16 GEMM input (x for w13, the SiTU output y for w2; no intermediate rounding) and added
    to the fp32 GEMM result (after the NVFP4 row scale) before its single bf16 rounding. B's rows for w13 follow the
    interleaved (gate, up) row order of w13."""
    x = x.float()
    t, h = x.shape
    if routing is None:
        ids, gam, _ = inkling_route(x, w.gate_w, w.gate_b)
    elif routing[1] is None:  # routing replay: given expert ids (slot order), weights from this forward's router
        ids = routing[0].long()
        gam = route_weights(mm(x, w.gate_w) if _EXACT else _router_logits(x, w.gate_w), ids)
    else:
        ids, gam = routing
    topk = ids.shape[1]
    nv = w.fmt == "nvfp4"
    if act_quant:
        xq = inkling_quant_nvfp4_4o6(x) if nv else inkling_quant_mxfp8(x)
        xv = xq.values
    else:
        xq = None
        xv = x
    down = torch.zeros(t, topk, h)
    flat = ids.reshape(-1)
    acc_dt = torch.float32
    for e in torch.unique(flat).tolist():
        sel = torch.nonzero(flat == e).flatten()
        tok = sel // topk
        slot = sel % topk
        def _wt(name, e=e):  # (L5) the expert weight as the original builds it (fp32, NVFP4 global for W4A16)
            m_ = w.mat(name, e)
            if nv and not act_quant:
                m_ = m_ * nvfp4_global(w.g13[e] if name == "w13" else w.g2[e],
                                       w.den13 if name == "w13" else w.den2).to(acc_dt)
            return m_

        a = xv[tok]
        gu = _wmm(a, _wt, "w13")
        if nv and act_quant:
            gu = gu * _row_scale(xq.amax[tok], w.g13[e], w.den13)[:, None]
        if lora is not None:
            la, lb = lora["w13"]
            gu = gu + lora["scale"] * mm(mm(x[tok], la[e]), lb[e])
        gu = rb(gu)
        y = inkling_softcap_swiglu(gu, gam[tok, slot])
        if act_quant:
            yq = inkling_quant_nvfp4_4o6(y) if nv else inkling_quant_mxfp8(y)
            yv = yq.values
        else:
            yv = y
        dn = _wmm(yv, _wt, "w2")
        if nv and act_quant:
            dn = dn * _row_scale(yq.amax, w.g2[e], w.den2)[:, None]
        if lora is not None:
            la, lb = lora["w2"]
            dn = dn + lora["scale"] * mm(mm(y, la[e]), lb[e])
        down[tok, slot] = rb(dn)
        if record is not None and e == record.get("expert", -1):
            record.update(dict(tok=tok, slot=slot, a=a, gate_up=gu, act=y, act_q=yv, down=rb(dn)))
    acc = torch.zeros(t, h)
    for s in range(topk):
        acc = acc + down[:, s]
    return rb(acc), ids, gam


def inkling_lm_head(h: torch.Tensor, unembed: torch.Tensor, softcap: float = 30.0, vocab: int = 200058) -> torch.Tensor:
    """bf16 hidden x bf16 unembed with fp32 output (head_dtype float32), softcap 30 tanh(x / 30), token domain
    200,058 (`Inkling2LogitsProcessor`, `configs.py` head_dtype)."""
    logits = mm(h, unembed[:vocab])
    return t_tanh(logits / softcap) * softcap


# ----------------------------------------------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------------------------------------------
CORE = range(6, 18)
NUM_PASSES = 3
GLOBAL_LAYERS = (5, 11, 17, 23)
AUX_VISITS = (5, 23, 41)


def visits() -> list[tuple[int, int]]:
    """(layer, pass) of visits 0..47 (OPS.md 1)."""
    v = [(i, 0) for i in range(CORE.start)]
    v += [(i, p) for p in range(NUM_PASSES) for i in CORE]
    v += [(i, 0) for i in range(CORE.stop, 24)]
    return v


VISITS = visits()


def visit_of(layer: int, pass_idx: int) -> int:
    if layer < CORE.start:
        return layer
    if layer >= CORE.stop:
        return layer + (NUM_PASSES - 1) * len(CORE)
    return layer + pass_idx * len(CORE)


@dataclass
class VisitState:
    k: list = field(default_factory=list)  # chunks of [t, 8, 128] KV values (bf16 / float8 storage)
    v: list = field(default_factory=list)
    conv: dict = field(default_factory=dict)  # stream -> [3, C] last inputs (bf16 values)

    def keys(self) -> tuple[torch.Tensor, torch.Tensor]:
        k = torch.cat(self.k).float()
        v = torch.cat(self.v).float()
        if len(self.k) > 1:  # keep one chunk
            self.k = [k.to(self.k[0].dtype)]
            self.v = [v.to(self.v[0].dtype)]
        return k, v


class LayerWeights:
    pass


class InklingRef:
    """The looped target model with per-visit KV caches and conv states (one sequence).

    flags: kv_fp8 (unit-scale e4m3 KV, the served config; False = bf16 KV), act_quant (vLLM's per-token NVFP4 /
    MXFP8 activation quantisation in the experts; False = bf16 activations), attn_weights 'bf16' (faithful oracle) or
    'fp8:<scheme>' (scheme in tensor / channel / block1x128 / block128x128), attn_act_fp8 (W8A8 for those),
    exact (fp64 accumulation), expert_cache_dtype (bf16 = exact values, 49 GB; float32 = 98 GB, faster).

    forward(ids) appends ids to the sequence (prefill, chunked prefill, decode, verify are all forward calls);
    snapshot() / restore() copy the whole state (for verify-rollback tests)."""

    def __init__(
        self,
        model_dir: str = MODEL_DIR,
        kv_fp8: bool = True,
        act_quant: bool = True,
        attn_weights: str = "bf16",
        attn_act_fp8: bool = False,
        exact: bool = False,
        threads: int = 32,
        expert_cache_dtype: torch.dtype = torch.bfloat16,
        w13_interleaved: bool = True,
        nibble_swap: bool = False,
        use_expert_cache: bool = True,
        expert_mode: str = "cache",
        verbose: bool = True,
    ) -> None:
        torch.set_num_threads(threads)
        set_exact(exact)
        self.kv_fp8 = kv_fp8
        self.act_quant = act_quant
        self.attn_weights = attn_weights
        self.attn_act_fp8 = attn_act_fp8
        self.expert_cache_dtype = expert_cache_dtype
        self.w13_interleaved = w13_interleaved
        self.nibble_swap = nibble_swap  # layout hypothesis test only (default: low nibble = even element)
        self.use_expert_cache = use_expert_cache
        # "cache": exact bf16 expert values dequantised once (49 GB, fastest); "stream": dequantised per use from the
        # checkpoint's mmap (page cache, ~5 GB of RAM held; ~2 s more per fully-routed visit)
        self.expert_mode = os.environ.get("INKLING_EXPERT_MODE", expert_mode)
        self.verbose = verbose
        self.kv_store_f32 = False  # (L2)
        cfg = json.load(open(os.path.join(model_dir, "config.json")))["text_config"]
        self.cfg = cfg
        self.eps = cfg["rms_norm_eps"]
        self.local = set(cfg["local_layer_ids"])
        self.f = safe_open(os.path.join(model_dir, "model-00001-of-00001.safetensors"), "pt")
        t0 = time.time()
        g = self._get
        self.embed = g("model.llm.embed.weight")  # bf16 [201024, 2304]
        self.embed_norm = g("model.llm.embed_norm.weight").float()
        self.final_norm = g("model.llm.norm.weight").float()
        sc = D**-0.5
        self.attn_queries = [g(f"model.llm.attn_res.attn_queries.{i}").float().reshape(-1, D) * sc for i in range(23)]
        self.mlp_queries = [g(f"model.llm.attn_res.mlp_queries.{i}").float().reshape(-1, D) * sc for i in range(24)]
        self.output_query = g("model.llm.attn_res.output_query").float().reshape(1, D) * sc
        cq = g("model.llm.middle_cycle_cross_pass_queries").float().reshape(-1, 2, D)
        kq = g("model.llm.middle_cycle_cross_pass_kv_queries").float().reshape(-1, 2, D)
        self.message_queries = torch.stack((cq[:, 0], kq[:, 0], kq[:, 1], cq[:, 1]), 1) * sc  # [12, 4, D]
        self.layers = [self._load_layer(i) for i in range(24)]
        self._unembed = None
        self.reset()
        self._log(f"loaded small weights in {time.time() - t0:.1f}s")

    # ---------------------------------------------------------------- loading
    def _log(self, msg: str) -> None:
        if self.verbose:
            from datetime import datetime
            from zoneinfo import ZoneInfo

            now = datetime.now(ZoneInfo("Europe/Bucharest")).strftime("%H:%M:%S")
            print(f"[inkling_ref {now}] {msg}", flush=True)

    def _get(self, name: str) -> torch.Tensor:
        return self.f.get_tensor(name)

    def _attn_w(self, w: torch.Tensor) -> torch.Tensor:
        w = w.float()
        if self.attn_weights == "bf16":
            return w
        kind, scheme = self.attn_weights.split(":")
        assert kind == "fp8"
        return fp8_quant_weight(w, scheme)[2]

    def _load_layer(self, i: int) -> LayerWeights:
        g = self._get
        p = f"model.llm.layers.{i}."
        L = LayerWeights()
        L.idx = i
        L.is_local = i in self.local
        L.wq = self._attn_w(g(p + "attn.wq_du.weight"))
        L.wr = self._attn_w(g(p + "attn.wr_du.weight"))
        L.wg = self._attn_w(g(p + "attn.gate_attn.weight"))
        L.ws = self._attn_w(g(p + "attn.attn_sink_proj.weight"))
        L.wk = self._attn_w(g(p + "attn.wk_dv.weight"))
        L.wv = self._attn_w(g(p + "attn.wv_dv.weight"))
        L.wo = self._attn_w(g(p + "attn.wo_ud.weight"))
        L.q_scale = g(p + "attn.q_l2_clip.scale").float()
        L.r_scale = g(p + "attn.rel_l2_clip.scale").float()
        L.rel_proj = g(p + "attn.rel_logits_proj.proj").float()
        if L.is_local:
            L.k_conv = g(p + "attn.k_sconv.weight").float().reshape(1024, 4)
            L.v_conv = g(p + "attn.v_sconv.weight").float().reshape(1024, 4)
            L.attn_norm = g(p + "attn_norm.weight").float().reshape(1, D)
        else:
            L.k_static = g(p + "attn.k_sconv.static_w").float()
            L.v_static = g(p + "attn.v_sconv.static_w").float()
            L.k_wproj = g(p + "attn.k_sconv.weight_proj").float()
            L.v_wproj = g(p + "attn.v_sconv.weight_proj").float()
            L.attn_norm = torch.stack(
                [g(p + f"{s}_input_norm.weight").float() for s in ("q", "k", "v")]
            )
        L.mlp_norm = g(p + "mlp_norm.weight").float().reshape(1, D)
        L.attn_sconv = g(p + "attn_sconv.weight").float().reshape(D, 4)
        L.mlp_sconv = g(p + "mlp_sconv.weight").float().reshape(D, 4)
        L.gate_w = g(p + "mlp.gate.weight").float()
        L.gate_b = g(p + "mlp.gate.bias").float()
        if i in CORE:
            L.ctrl_down = torch.stack([g(p + f"controllers.{s}.down_weight") for s in LOOP_STREAMS]).float()
            L.ctrl_gf = torch.stack(
                [torch.cat((g(p + f"controllers.{s}.gain_weight"), g(p + f"controllers.{s}.forget_weight")))
                 for s in LOOP_STREAMS]
            ).float()
        L.moe = None
        return L

    def moe_weights(self, i: int) -> MoEWeights:
        """Experts of layer i, dequantised once to exact values (bf16 cache by default)."""
        L = self.layers[i]
        if L.moe is not None:
            return L.moe
        t0 = time.time()
        g = self._get
        p = f"model.llm.layers.{i}.mlp.experts."
        sl = self.f.get_slice
        nv = self.f.get_slice(p + "w13_weight").get_dtype() == "U8"
        dt = self.expert_cache_dtype
        mats = {}
        if self.expert_mode == "stream":
            assert self.w13_interleaved and not self.nibble_swap
            L.moe = MoEWeights(
                fmt="nvfp4" if nv else "mxfp8", w13=None, w2=None,
                g13=g(p + "w13_weight_amax").float() if nv else None,
                g2=g(p + "w2_weight_amax").float() if nv else None,
                den13=g(p + "w13_weight_outer_scale_denominator").float() if nv else None,
                den2=g(p + "w2_weight_outer_scale_denominator").float() if nv else None,
                gate_w=L.gate_w, gate_b=L.gate_b,
                src={n_: (sl(p + f"{n_}_weight"), sl(p + f"{n_}_weight_scale")) for n_ in ("w13", "w2")})
            return L.moe
        cache = self._mmap_experts(i) if self.use_expert_cache else None
        for name in ("w13", "w2") if cache is None else ():
            wsl = sl(p + f"{name}_weight")
            ssl = sl(p + f"{name}_weight_scale")
            n, kk = wsl.get_shape()[1], wsl.get_shape()[2] * (2 if nv else 1)
            out = torch.empty(128, n, kk, dtype=dt)
            for e0 in range(0, 128, 8):
                wq = wsl[e0 : e0 + 8]
                if nv and self.nibble_swap:
                    wq = ((wq & 15) << 4) | (wq >> 4)
                ws = ssl[e0 : e0 + 8]
                deq = dequant_nvfp4_unit(wq, ws) if nv else dequant_mxfp8(wq, ws)
                out[e0 : e0 + 8] = deq.to(dt)
                if e0 == 0 and dt == torch.bfloat16:
                    assert torch.equal(deq.to(dt).float(), deq), "expert dequant not exact in bf16"
            if name == "w13" and not self.w13_interleaved:
                # alternative layout hypothesis [gate rows | up rows] -> interleave for the SiTU reader
                ii = out.shape[1] // 2
                out = torch.stack((out[:, :ii], out[:, ii:]), 2).reshape(out.shape)
            mats[name] = out
        if cache is not None:
            mats = cache
        L.moe = MoEWeights(
            fmt="nvfp4" if nv else "mxfp8",
            w13=mats["w13"],
            w2=mats["w2"],
            g13=g(p + "w13_weight_amax").float() if nv else None,
            g2=g(p + "w2_weight_amax").float() if nv else None,
            den13=g(p + "w13_weight_outer_scale_denominator").float() if nv else None,
            den2=g(p + "w2_weight_outer_scale_denominator").float() if nv else None,
            gate_w=L.gate_w,
            gate_b=L.gate_b,
        )
        self._log(f"layer {i} experts ({L.moe.fmt}) dequantised in {time.time() - t0:.1f}s")
        return L.moe

    def _mmap_experts(self, i: int) -> dict | None:
        """The layer's experts from EXPERT_CACHE_DIR (bf16, mmapped) when present and the layout flags are default."""
        if self.nibble_swap or not self.w13_interleaved or self.expert_cache_dtype != torch.bfloat16:
            return None
        meta_f = os.path.join(EXPERT_CACHE_DIR, "meta.json")
        if not os.path.exists(meta_f):
            return None
        meta = json.load(open(meta_f))
        out = {}
        for name in ("w13", "w2"):
            shape = meta[f"layer{i}"][name]
            f = os.path.join(EXPERT_CACHE_DIR, f"layer{i}_{name}.bf16")
            n = math.prod(shape)
            out[name] = torch.from_file(f, shared=False, size=n, dtype=torch.bfloat16).reshape(shape)
        return out

    @property
    def unembed(self) -> torch.Tensor:
        if self._unembed is None:
            self._unembed = self._get("model.llm.unembed.weight")[:200058].float()
        return self._unembed

    # ---------------------------------------------------------------- state
    def reset(self) -> None:
        self.pos = 0
        self.state = [VisitState() for _ in range(48)]

    def snapshot(self) -> tuple:
        import copy

        return self.pos, copy.deepcopy(self.state)

    def restore(self, snap: tuple) -> None:
        import copy

        self.pos, st = snap
        self.state = copy.deepcopy(st)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        ids: torch.Tensor,
        logits: str | None = "all",
        return_aux: bool = False,
        pos_offset: int = 0,
        record_visits: tuple[int, ...] = (),
        routing_override: dict | None = None,
        ckpt: bool = False,
    ) -> dict:
        """Append `ids` [T] and run them. Returns dict(logits fp32 [T or 1, 200058], hidden bf16 [T, D],
        aux [T, 4, D] (stack[nb] at the start of visits 5, 23, 41 + final output) if return_aux,
        routing {visit: ids [T, 8]}, rec {visit: {op: tensors}} for record_visits).
        routing_override {visit: ids [T, 8]} REPLAYS routing (given ids, weights from this forward's own router) or
        {visit: (ids, weights)} forces both.
        pos_offset shifts the absolute positions (tests of tau only)."""
        ids = ids.long().reshape(-1)
        t = ids.shape[0]
        if ckpt:  # (L3) one chunk only: every visit's state is used once and re-created inside the checkpoint
            assert self.pos == 0 and not record_visits
        positions = torch.arange(self.pos, self.pos + t) + pos_offset
        self._positions = positions
        self._record_visits = set(record_visits)
        self._rec = {}
        self._routing = {}
        self._routing_override = routing_override or {}
        x = inkling_embed(ids, self.embed, self.embed_norm, self.eps)
        tau = inkling_log_tau(positions)
        stack: list[torch.Tensor] = [x]
        nb = 0
        inputs = bank = prelude_blocks = prelude_partial = prelude_dests = None
        aux = []
        for visit, (i, p) in enumerate(VISITS):
            looped = p > 0
            if i == CORE.start and not looped:
                prelude_blocks = nb
                prelude_partial = stack[nb].clone()
            elif i == CORE.start:
                assert nb == prelude_blocks + 2
                bank = [stack[prelude_blocks + 1].clone(), stack[prelude_blocks + 2].clone()]
                nb = prelude_blocks
                stack[nb] = prelude_partial.clone()
                if prelude_dests is None:
                    prelude_dests = inkling_attn_res(
                        stack[:nb], prelude_partial, None, self.attn_queries[i - 1], None
                    )
                inputs = prelude_dests
            if visit in AUX_VISITS:
                aux.append(stack[nb].clone())
            if visit + 1 == len(VISITS):
                next_query, next_norm = self.output_query, self.final_norm.reshape(1, D)
            else:
                ni, np_ = VISITS[visit + 1]
                next_query = self.attn_queries[ni - 1]
                next_norm = None if np_ else self.layers[ni].attn_norm
            run = self._run_layer_ckpt if ckpt else self._run_layer  # (L3)
            nb, inputs = run(
                visit, i, p, stack, nb, inputs, bank if looped else None, next_query, next_norm, positions, tau
            )
        out = inputs[0]
        self.pos += t
        res = {"hidden": out, "routing": self._routing}
        if logits == "all":
            res["logits"] = inkling_lm_head(out, self.unembed)
        elif logits == "last":
            res["logits"] = inkling_lm_head(out[-1:], self.unembed)
        if return_aux:
            res["aux"] = torch.stack([*aux, out], 1)
        if record_visits:
            res["rec"] = self._rec
        return res

    def _record(self, visit: int, op: str, **tensors) -> None:
        if visit in self._record_visits:
            self._rec.setdefault(visit, {})[op] = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in tensors.items()}

    def _run_layer_ckpt(self, visit, i, p, stack, nb, inputs, bank, next_query, next_norm, positions, tau):
        """(L3) _run_layer under a non-reentrant activation checkpoint: a pure function of (stack, inputs, bank);
        the visit's KV / conv state is re-created fresh at every (re)computation (one-chunk forwards only)."""
        import torch.utils.checkpoint as tuc

        n = len(stack)
        nb_out = nb + 1 if i % 6 == 0 else nb

        dev = torch.get_default_device()  # the backward recompute runs on autograd's device thread, which does not
                                          # inherit torch.set_default_device: restore it there

        def fn(_dummy, *ts):
            with torch.device(dev):
                st_list = list(ts[:n])
                bk = None if bank is None else [ts[n + 1], ts[n + 2]]
                self.state[visit] = VisitState()
                nb2, nxt = self._run_layer(visit, i, p, st_list, nb, ts[n], bk, next_query, next_norm, positions, tau)
                assert nb2 == nb_out
                return st_list[nb2], nxt

        extra = () if bank is None else (bank[0], bank[1])
        # REENTRANT checkpoint: forward without a graph, backward = recompute + local backward of this visit only
        # (deterministic memory; the non-reentrant variant kept recomputed tensors of every visit alive until the end
        # of the backward pass). The dummy input requiring grad makes visit 0 (no input requiring grad) reach its
        # LoRA parameters too.
        dummy = torch.zeros((), requires_grad=True)
        new_partial, nxt = tuc.checkpoint(fn, dummy, *stack, inputs, *extra, use_reentrant=True,
                                          preserve_rng_state=False)
        if nb_out == len(stack):
            stack.append(new_partial)
        else:
            stack[nb_out] = new_partial
        return nb_out, nxt

    def _run_layer(self, visit, i, p, stack, nb, inputs, bank, next_query, next_norm, positions, tau):
        """`loop.py` `Inkling2LoopModel._run_layer` for visit (layer i, pass p)."""
        L = self.layers[i]
        st = self.state[visit]
        partial = stack[nb]
        messages = None
        if bank is not None:
            messages = inkling_attn_res([bank[0]], bank[1], None, self.message_queries[i - CORE.start], None)
            dests = inputs
            norm_w = L.attn_norm
            if L.is_local:
                dests = dests.expand(3, -1, -1)
                norm_w = norm_w.expand(3, -1)
            attn_inputs = inkling_loop_blend(dests, messages[:3], L.ctrl_down[0:3], L.ctrl_gf[0:3], norm_w, self.eps)
            self._record(visit, "inkling_attn_res_messages", stack=bank[0][None], last=bank[1],
                         queries=self.message_queries[i - CORE.start], out=messages)
            self._record(visit, "inkling_loop_blend_attn", dest=inputs, message=messages[:3],
                         down=L.ctrl_down[0:3], gain_forget=L.ctrl_gf[0:3], norm_weight=L.attn_norm, out=attn_inputs)
        elif inputs is None:
            assert nb == 0
            attn_inputs = torch.stack([rb(rms_norm(partial, L.attn_norm[j], self.eps)) for j in range(L.attn_norm.shape[0])])
        else:
            attn_inputs = inputs
        attn_partial = self._attention(visit, L, st, attn_inputs, positions, None if L.is_local else tau)

        first = i % 6 == 0
        st_in = st.conv.get("attn")
        attn_out, st.conv["attn"] = inkling_short_conv(attn_partial, L.attn_sconv, st_in)
        self._record(visit, "inkling_short_conv_attn", x=attn_partial, weight=L.attn_sconv, out=attn_out,
                     state_in=torch.zeros(3, D) if st_in is None else st_in, state_out=st.conv["attn"])
        mlp_norm = None if messages is not None else L.mlp_norm
        if first:
            mlp_in = inkling_attn_res(stack[: nb + 1], attn_out, None, self.mlp_queries[i], mlp_norm, self.eps)
            self._record(visit, "inkling_attn_res_mlp", stack=torch.stack(stack[: nb + 1]), last=attn_out,
                         queries=self.mlp_queries[i], norm_weight=mlp_norm, out=mlp_in)
        else:
            mlp_in = inkling_attn_res(stack[:nb], partial, attn_out, self.mlp_queries[i], mlp_norm, self.eps)
            self._record(visit, "inkling_attn_res_mlp", stack=torch.stack(stack[:nb]), last=partial, addend=attn_out,
                         queries=self.mlp_queries[i], norm_weight=mlp_norm, out=mlp_in)
        if messages is not None:
            blended = inkling_loop_blend(mlp_in, messages[3:], L.ctrl_down[3:4], L.ctrl_gf[3:4], L.mlp_norm, self.eps)
            self._record(visit, "inkling_loop_blend_mlp", dest=mlp_in, message=messages[3:], down=L.ctrl_down[3:4],
                         gain_forget=L.ctrl_gf[3:4], norm_weight=L.mlp_norm, out=blended)
            mlp_in = blended

        residual = None if first else partial
        if first:
            nb += 1
        mw = self.moe_weights(i)
        ro = self._routing_override.get(visit)
        if ro is not None:
            if isinstance(ro, torch.Tensor):  # replay: ids [T, 8] of THIS forward's tokens
                ro = (ro, None)
            moe_out, rids, gam = inkling_moe(mlp_in[0], mw, self.act_quant, routing=ro)
        else:
            moe_out, rids, gam = inkling_moe(mlp_in[0], mw, self.act_quant)
        self._routing[visit] = rids
        if getattr(self, "debug_stats", None) is not None:
            rms = lambda z: float(z.float().pow(2).mean().sqrt())
            self.debug_stats.append(dict(visit=visit, layer=i, p=p, attn_in=rms(attn_inputs if False else mlp_in),
                                         attn_out=rms(attn_out), moe_out=rms(moe_out), partial=rms(partial)))
        self._record(visit, "inkling_moe", x=mlp_in[0], ids=rids, weights=gam, out=moe_out, fmt=mw.fmt)
        st_in = st.conv.get("mlp")
        new_partial, st.conv["mlp"] = inkling_short_conv(moe_out, L.mlp_sconv, st_in, addend=attn_out, residual=residual)
        self._record(visit, "inkling_short_conv_mlp", x=moe_out, weight=L.mlp_sconv, addend=attn_out,
                     state_in=torch.zeros(3, D) if st_in is None else st_in, state_out=st.conv["mlp"],
                     residual=residual if residual is not None else torch.zeros(0), out=new_partial)
        if nb == len(stack):
            stack.append(new_partial)
        else:
            stack[nb] = new_partial
        nxt = inkling_attn_res(stack[:nb], stack[nb], None, next_query, next_norm, self.eps)
        self._record(visit, "inkling_attn_res_next", stack=torch.stack(stack[:nb]), last=stack[nb], queries=next_query,
                     norm_weight=next_norm if next_norm is not None else torch.zeros(0), out=nxt)
        return nb, nxt

    def _attention(self, visit, L, st, inputs, positions, tau):
        """`attention.py` `Inkling2Attention.forward` for one visit (its own KV and conv state)."""
        if L.is_local and inputs.shape[0] == 1:
            xq = xk = xv = inputs[0]
        else:
            xq, xk, xv = inputs[0], inputs[1], inputs[2]
        t = xq.shape[0]
        a8 = self.attn_act_fp8
        q = inkling_attn_linear(xq, L.wq, a8).reshape(t, 24, 128)
        r = inkling_attn_linear(xq, L.wr, a8).reshape(t, 24, 16)
        gate = inkling_attn_linear(xq, L.wg, a8)
        sink = inkling_attn_linear(xq, L.ws, a8)
        emu = VLLM_EMU["on"] and t > VLLM_EMU["gemv_max_m"]
        separate = not (L.is_local and inputs.shape[0] == 1)
        if emu and separate and VLLM_EMU["kv_split"]:
            k = rb(mm_splitk(xk, L.wk, VLLM_EMU["kv_split"]))
            v = rb(mm_splitk(xv, L.wv, VLLM_EMU["kv_split"]))
        else:
            k = inkling_attn_linear(xk, L.wk, a8)
            v = inkling_attn_linear(xv, L.wv, a8)
        if L.is_local:
            kc, vc = L.k_conv, L.v_conv
        elif emu and VLLM_EMU["dyn_split"]:
            kc = (L.k_static, rb(mm_splitk(xk, L.k_wproj, VLLM_EMU["dyn_split"])))
            vc = (L.v_static, rb(mm_splitk(xv, L.v_wproj, VLLM_EMU["dyn_split"])))
        else:
            kc = (L.k_static, rb(mm(xk, L.k_wproj)))
            vc = (L.v_static, rb(mm(xv, L.v_wproj)))
        k_state_in = torch.zeros(3, 1024) if st.conv.get("k") is None else st.conv["k"]
        v_state_in = torch.zeros(3, 1024) if st.conv.get("v") is None else st.conv["v"]
        pro = inkling_attention_prologue(
            q, r, k, v, L.q_scale, L.r_scale, tau, L.rel_proj, kc, vc,
            st.conv.get("k"), st.conv.get("v"), self.kv_fp8,
        )
        st.conv["k"], st.conv["v"] = pro.k_state, pro.v_state
        kv_dtype = torch.float8_e4m3fn if self.kv_fp8 else torch.bfloat16
        if self.kv_store_f32:  # (L2) same values (pro.k / pro.v are e4m3- / bf16-exact), fp32 storage
            kv_dtype = torch.float32
        st.k.append(pro.k.to(kv_dtype))
        st.v.append(pro.v.to(kv_dtype))
        kall, vall = st.keys()
        npos = kall.shape[0]
        pos_k = torch.arange(npos) + (positions[0] - (npos - t))
        window = 0 if not L.is_local else 1024
        pblk = VLLM_EMU["p_bf16_block"] if VLLM_EMU["on"] and t > 21 else 0
        out, lse = inkling_rel_attention(pro.q, kall, vall, pro.rel, positions, pos_k, window, p_bf16_block=pblk)
        o = inkling_gate_sink(out, lse, gate, sink)
        y = inkling_attn_linear(o, L.wo, a8)
        if visit in self._record_visits:
            self._record(visit, "inkling_attn_linear_qkvr", x_q=xq, x_k=xk, x_v=xv, q=q.reshape(t, -1), r=r.reshape(t, -1),
                         gate=gate, sink=sink, k=k, v=v)
            self._record(visit, "inkling_attention_prologue", q=q, r=r, k=k, v=v, q_scale=L.q_scale, r_scale=L.r_scale,
                         tau=tau if tau is not None else torch.zeros(0), rel_proj=L.rel_proj,
                         k_dyn=kc[1] if isinstance(kc, tuple) else torch.zeros(0),
                         v_dyn=vc[1] if isinstance(vc, tuple) else torch.zeros(0),
                         q_out=pro.q, r_out=pro.r, rel=pro.rel, k_out=pro.k, v_out=pro.v, positions=positions,
                         k_state_in=k_state_in, v_state_in=v_state_in, k_state_out=pro.k_state,
                         v_state_out=pro.v_state, is_local=L.is_local)
            self._record(visit, "inkling_rel_attention", q=pro.q, k=kall, v=vall, rel=pro.rel, pos_q=positions,
                         pos_k=pos_k, window=window, out=out, lse=lse)
            self._record(visit, "inkling_gate_sink", out=out, lse=lse, gate=gate, sink=sink, result=o)
            self._record(visit, "inkling_attn_linear_wo", x=o, out=y)
        return y


def build_expert_cache(out_dir: str = EXPERT_CACHE_DIR, threads: int = 6) -> None:
    """Write every layer's exact bf16 expert values as raw files + meta.json (49 GB; run once on a CPU box)."""
    os.makedirs(out_dir, exist_ok=True)
    m = InklingRef(threads=threads, use_expert_cache=False)
    meta = {"source_sha256_prefix": "65e60f51", "layout": "w13 rows interleaved (gate 2j, up 2j+1); NVFP4 values "
            "e2m1*e4m3 (no global), MXFP8 values e4m3*2^(e-127)", "dtype": "bfloat16"}
    for i in range(24):
        mw = m.moe_weights(i)
        meta[f"layer{i}"] = {"fmt": mw.fmt}
        for name in ("w13", "w2"):
            t = getattr(mw, name).contiguous()
            f = os.path.join(out_dir, f"layer{i}_{name}.bf16")
            t.view(torch.int16).numpy().tofile(f + ".tmp")
            os.replace(f + ".tmp", f)
            meta[f"layer{i}"][name] = list(t.shape)
        m.layers[i].moe = None
        json.dump(meta, open(os.path.join(out_dir, "meta.json.tmp"), "w"), indent=1)
        os.replace(os.path.join(out_dir, "meta.json.tmp"), os.path.join(out_dir, "meta.json.partial"))
        time.sleep(2)
    os.replace(os.path.join(out_dir, "meta.json.partial"), os.path.join(out_dir, "meta.json"))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build-expert-cache":
        build_expert_cache(threads=int(os.environ.get("INKLING_THREADS", "6")))
