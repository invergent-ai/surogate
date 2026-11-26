"""GPU op tests of the Inkling-2 trainer kernels (csrc/src/kernels/inkling) against the fp64 references.

Rules: the tolerances of tests/inkling_ref/tolerance.py:
C.1 bf16 / e4m3 outputs, C.2 attention vs the fp32 "pt" emulation, C.3 fp32 paths, C.5 determinism (bit-exact reruns).
Production dims: Hq 24, Hkv 8, head 128, rel extent / window 1024, C 2304, k/v channels 1024; T = 4,096 rows packed
as 3 documents (2,500 / 1,000 / 596) + a key-prefix (chunk) case for the attention parts.

Kernels are called through the flat C entry points of csrc/src/kernels/inkling/inkling_capi.cpp, loaded with ctypes
from libsinfer_inkling_bwd.so (env INKLING_LIB). References run
on the CPU in fp64 (attention per KV-head group to bound memory).

Run (one card): CUDA_VISIBLE_DEVICES=0 INKLING_LIB=/path/libsinfer_inkling_bwd.so python tests/test_inkling2_kernels_gpu.py [--only name,...]
Exit code 0 = all PASS. A JSON report goes to $INKLING_REPORT (default ./inkling2_kernels_gpu_report.json).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time

import torch

REF_DIR = os.environ.get("INKLING_G2_REF", os.path.join(os.path.dirname(os.path.abspath(__file__)), "inkling_ref"))
sys.path.insert(0, REF_DIR)
import inkling_kernels_ref as R  # noqa: E402
import tolerance as TOL  # noqa: E402

LIB = os.environ.get("INKLING_LIB", "libsinfer_inkling_bwd.so")
F64, F32, BF16 = torch.float64, torch.float32, torch.bfloat16
# INKLING_DRY=1: CPU plumbing check of this script (no kernels: every call is a no-op, the checks then FAIL);
# INKLING_SMALL=1: small shapes (documents 250 / 100 / 60, prefix case 128 queries over 1,500 keys).
DRY = os.environ.get("INKLING_DRY") == "1"
SMALL = os.environ.get("INKLING_SMALL") == "1"
DEV = torch.device("cpu" if DRY else "cuda")
HQ, HKV, HD, EXT, WIN, C_MODEL, C_KV = 24, 8, 128, 1024, 1024, 2304, 1024
DOCS = (250, 100, 60) if SMALL else (2500, 1000, 596)
T = sum(DOCS)
PREFIX_Q, PREFIX_K = (128, 1500) if SMALL else (2048, 4096)
torch.set_num_threads(int(os.environ.get("INKLING_CPU_THREADS", "8")))
# The loop-blend test does the op wrapper's GEMMs with torch: make them fp32-accumulating like the wrapper's cuBLASLt
# calls (torch's default lets cuBLAS reduce bf16 GEMMs in reduced precision: 40 % of h off by >= 1 ulp, M3 run 3).
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False

_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(LIB, mode=ctypes.RTLD_GLOBAL)
        _lib.inkling_capi_last_error.restype = ctypes.c_char_p
        _lib.inkling_capi_attention_bwd_workspace_bytes.restype = ctypes.c_long
    return _lib


_KEEP: list = []


def P(t):
    """Device pointer of a tensor (None -> NULL). The tensor is kept alive until the next call() returns: `P(bf(x))`
    must not let the caching allocator hand x's block to the next argument's temporary."""
    if t is not None:
        _KEEP.append(t)
    return ctypes.c_void_p(t.data_ptr() if t is not None else 0)


def call(name, *args):
    if DRY:
        _KEEP.clear()
        return
    f = getattr(lib(), "inkling_capi_" + name)
    conv = []
    for a in args:
        if isinstance(a, bool):
            conv.append(ctypes.c_int(int(a)))
        elif isinstance(a, int):
            conv.append(ctypes.c_long(a) if abs(a) >= 2 ** 31 else ctypes.c_int(a))
        elif isinstance(a, float):
            conv.append(ctypes.c_float(a))
        else:
            conv.append(a)
    rc = f(*conv, ctypes.c_void_p(0))
    torch.cuda.synchronize()
    _KEEP.clear()
    if rc != 0:
        raise RuntimeError(f"{name}: {lib().inkling_capi_last_error().decode()}")


def L(x):
    return ctypes.c_long(x)


def bf(x):
    return x.to(BF16).to(DEV).contiguous()


def f32(x):
    return x.to(F32).to(DEV).contiguous()


def i32(x):
    return x.to(torch.int32).to(DEV).contiguous()


def cpu64(x):
    return x.detach().to("cpu").to(F64)


def positions(lengths=DOCS, offsets=None):
    offsets = offsets or [0] * len(lengths)
    return torch.cat([torch.arange(o, o + n) for n, o in zip(lengths, offsets)])


def doc_ids(lengths=DOCS):
    return torch.cat([torch.full((n,), i) for i, n in enumerate(lengths)])


RESULTS: list[dict] = []


def ulp_bf16(r):
    _, e = torch.frexp(r.double().abs().clamp(min=2.0 ** -126))
    return torch.ldexp(torch.ones_like(r.double()), e - 8)


def rule_two_roundings(x, ref64, inner_ref64, inner_factor, max_flip_frac=0.005):
    """C.1 for outputs with TWO bf16 roundings, out = bf16(f(bf16(inner))): a 1-ulp flip of the inner rounding (fp32 vs
    fp64 inner value) moves the result by inner_factor * ulp(inner), which can exceed 1 ulp of the result (tau > 1;
    cancellation in (residual + addend) + conv). Bound per element: ulp(ref) + inner_factor * ulp(inner); flips <= 0.5 %."""
    err = (x.double() - ref64.double()).abs()
    bound = ulp_bf16(ref64) + inner_factor * ulp_bf16(inner_ref64)
    over = int((err > bound * (1 + 1e-9)).sum())
    flips = float((err > 0).double().mean())
    return over == 0 and flips <= max_flip_frac, dict(over_bound=over, flip_frac=flips,
                                                     max_err_over_bound=float((err / bound).max()))


def record(test, check, ok_rep):
    ok, rep = ok_rep
    RESULTS.append(dict(test=test, check=check, ok=bool(ok), **{k: v for k, v in rep.items()}))
    print(f"  {'PASS' if ok else 'FAIL'} {test}/{check} {json.dumps(rep)}", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# 1. log tau
# ---------------------------------------------------------------------------------------------------------------
def test_log_tau():
    pos = torch.cat([positions(), torch.arange(8000, 140000, 97)])
    tau = torch.empty(pos.numel(), dtype=F32, device=DEV)
    call("log_tau", P(tau), P(i32(pos)), pos.numel(), 8192.0, 0.1)
    ref = R.log_tau(pos)
    err = float(((cpu64(tau) - ref).abs() / ref.abs()).max())
    record("log_tau", "fp32_rel_2e-7", (err <= 2e-7, dict(max_rel=err)))


# ---------------------------------------------------------------------------------------------------------------
# 2. qkv norm (q clip + tau, r clip, k clip e4m3, v rms e4m3) fwd + bwd
# ---------------------------------------------------------------------------------------------------------------
def test_qkv_norm():
    torch.manual_seed(1)
    pos = positions()
    tau64 = R.log_tau(pos + 9000)  # global-visit tau > 1
    cases = [
        ("q_clip_tau", HQ, HD, 22.627416997969522, R.CLIP, True, tau64, "bf16", 3.0),
        ("r_clip", HQ, 16, 8.0, R.CLIP, True, None, "bf16", 3.0),
        ("k_clip_e4m3", HKV, HD, 22.627416997969522, R.CLIP, False, None, "e4m3", 2.5),
        ("v_rms_e4m3", HKV, HD, 0.0, R.RMS, False, None, "e4m3", 1.0),
    ]
    for name, H, d, th, mode, has_scale, tau, store, amp in cases:
        x = (torch.randn(T, H, d, dtype=F64) * amp).to(BF16).to(F64)
        scale = (1.0 + 0.2 * torch.randn(H, d, dtype=F64)).to(BF16).to(F64) if has_scale else None
        y = torch.empty(T, H, d, dtype=BF16, device=DEV)
        call("qkv_norm_fwd", P(y), P(bf(x)), mode, P(bf(scale)) if has_scale else P(None),
             P(f32(tau)) if tau is not None else P(None), float(th), 1e-6, store == "e4m3", T, H, d)
        tau_ref = tau.to(F32).to(F64) if tau is not None else None  # the kernel gets fp32 tau
        ref = R.qkv_norm_fwd(x, scale, tau_ref, th=th, mode=mode, store=store)
        if store == "e4m3":
            record("qkv_norm_fwd", name, TOL.rule_e4m3_output(cpu64(y), ref))
        elif tau is not None:  # bf16(bf16(y) * tau): two roundings
            inner = R.qkv_norm_fwd(x, scale, None, th=th, mode=mode)
            record("qkv_norm_fwd", name, rule_two_roundings(cpu64(y), ref, inner, inner_factor=tau_ref[:, None, None]))
        else:
            record("qkv_norm_fwd", name, TOL.rule_bf16_output(cpu64(y), ref))
        dy = torch.randn(T, H, d, dtype=F64)
        dx = torch.empty(T, H, d, dtype=F32, device=DEV)
        call("qkv_norm_bwd", P(dx), P(f32(dy)), True, P(bf(x)), mode, P(bf(scale)) if has_scale else P(None),
             P(f32(tau)) if tau is not None else P(None), float(th), 1e-6, T, H, d)
        ref_dx = R.qkv_norm_bwd(dy.to(F32).to(F64), x, scale, tau_ref, th=th, mode=mode)
        record("qkv_norm_bwd", name, TOL.rule_fp32_path(cpu64(dx), ref_dx))


# ---------------------------------------------------------------------------------------------------------------
# 3 / 4. convolutions (docs from positions, carry, addend / residual), fwd + bwd
# ---------------------------------------------------------------------------------------------------------------
def test_short_conv():
    torch.manual_seed(2)
    for name, C, fused, carry_on in (("kv_carry", C_KV, False, True), ("partial_out", C_MODEL, True, False)):
        offs = [37, 0, 0] if carry_on else None  # doc 0 continues from the previous chunk
        pos = positions(DOCS, offs)
        x = torch.randn(T, C, dtype=F64).to(BF16).to(F64)
        w = (torch.randn(C, 4, dtype=F64) * 0.5).to(BF16).to(F64)
        carry = torch.randn(3, C, dtype=F64).to(BF16).to(F64) if carry_on else None
        addend = torch.randn(T, C, dtype=F64).to(BF16).to(F64) if fused else None
        resid = torch.randn(T, C, dtype=F64).to(BF16).to(F64) if fused else None
        out = torch.empty(T, C, dtype=BF16, device=DEV)
        call("short_conv_fwd", P(out), P(bf(x)), P(bf(w)), P(bf(addend)) if fused else P(None),
             P(bf(resid)) if fused else P(None), P(i32(pos)), P(bf(carry)) if carry_on else P(None),
             5 ** -0.5, T, C)
        ref, _ = R.short_conv_fwd(x, w, pos, carry, addend, resid)
        if fused:  # bf16((residual + addend) + bf16(conv)): two roundings
            inner, _ = R.short_conv_fwd(x, w, pos, carry)
            record("short_conv_fwd", name, rule_two_roundings(cpu64(out), ref, inner, inner_factor=1.0))
        else:
            record("short_conv_fwd", name, TOL.rule_bf16_output(cpu64(out), ref))
        dy = torch.randn(T, C, dtype=F64).to(F32).to(F64)
        dcn = torch.randn(3, C, dtype=F64).to(F32).to(F64)
        dx = torch.empty(T, C, dtype=F32, device=DEV)
        dc = torch.empty(3, C, dtype=F32, device=DEV)
        call("short_conv_bwd", P(dx), P(dc), P(f32(dy)), True, P(bf(w)), P(f32(dcn)), P(i32(pos)),
             P(bf(carry)) if carry_on else P(None), 5 ** -0.5, T, C)
        rdx, rdc, _, _ = R.short_conv_bwd(dy, w, pos, dcn)
        record("short_conv_bwd", name + "/dx", TOL.rule_fp32_path(cpu64(dx), rdx))
        record("short_conv_bwd", name + "/d_carry", TOL.rule_fp32_path(cpu64(dc), rdc))


def test_dynamic_conv():
    torch.manual_seed(3)
    C, hd = C_KV, HD
    pos = positions(DOCS, [5, 0, 0])
    x = torch.randn(T, C, dtype=F64).to(BF16).to(F64)
    sw = (torch.randn(4, C, dtype=F64) * 0.5).to(BF16).to(F64)
    dyn = (torch.randn(T, (C // hd) * 4, dtype=F64) * 0.3).to(BF16).to(F64)
    carry = torch.randn(3, C, dtype=F64).to(BF16).to(F64)
    out = torch.empty(T, C, dtype=BF16, device=DEV)
    call("dynamic_conv_fwd", P(out), P(bf(x)), P(bf(sw)), P(bf(dyn)), P(i32(pos)), P(bf(carry)), 9 ** -0.5, T, C, hd)
    ref, _ = R.dynamic_conv_fwd(x, sw, dyn, pos, carry)
    record("dynamic_conv_fwd", "carry", TOL.rule_bf16_output(cpu64(out), ref))
    dy = torch.randn(T, C, dtype=F64).to(F32).to(F64)
    dcn = torch.randn(3, C, dtype=F64).to(F32).to(F64)
    dx = torch.empty(T, C, dtype=F32, device=DEV)
    ddyn = torch.empty(T, (C // hd) * 4, dtype=F32, device=DEV)
    dc = torch.empty(3, C, dtype=F32, device=DEV)
    call("dynamic_conv_bwd", P(dx), P(ddyn), P(dc), P(f32(dy)), True, P(bf(x)), P(bf(sw)), P(bf(dyn)), P(f32(dcn)),
         P(i32(pos)), P(bf(carry)), 9 ** -0.5, T, C, hd)
    rdx, rddyn, rdc = R.dynamic_conv_bwd(dy, x, sw, dyn, pos, carry, dcn)
    record("dynamic_conv_bwd", "dx", TOL.rule_fp32_path(cpu64(dx), rdx))
    record("dynamic_conv_bwd", "d_dyn", TOL.rule_fp32_path(cpu64(ddyn), rddyn))
    record("dynamic_conv_bwd", "d_carry", TOL.rule_fp32_path(cpu64(dc), rdc))


# ---------------------------------------------------------------------------------------------------------------
# 5-7. attention: near (+ far) + merge, union backward
# ---------------------------------------------------------------------------------------------------------------
def _clip_rows(x, th):
    n = x.norm(dim=-1, keepdim=True)
    return x * torch.clamp(th / (n + 1e-6), max=1.0)


def _attn_inputs(tq, nk, seed):
    torch.manual_seed(seed)
    q = _clip_rows(torch.randn(tq, HQ, HD, dtype=F64) * 2.0, 22.627).to(BF16).to(F64)
    k = _clip_rows(torch.randn(nk, HKV, HD, dtype=F64) * 2.0, 22.627)
    k = k.to(F32).to(torch.float8_e4m3fn).to(F64)  # e4m3 KV values
    v = (torch.randn(nk, HKV, HD, dtype=F64)).to(F32).to(torch.float8_e4m3fn).to(F64)
    rel = (torch.randn(tq, HQ, EXT, dtype=F64) * 1.5).to(BF16).to(F64)
    sink = (torch.randn(tq, HQ, dtype=F64) * 2.0).to(BF16).to(F64)
    gate = torch.randn(tq, HQ * HD, dtype=F64).to(BF16).to(F64)
    d_res = torch.randn(tq, HQ * HD, dtype=F64).to(BF16).to(F64)
    return q, k, v, rel, sink, gate, d_res


def _ref_attention(q, k, v, rel, sink, gate, d_res, geo, is_global, dtype, emulate):
    """fp64 (emulate False) or the fp32 'pt' emulation (P / dS bf16 operands) per KV-head group."""
    outs = {n: [] for n in ("res", "out_s", "lse_s", "dq", "dk", "dv", "d_rel", "d_sink", "d_gate")}
    g = HQ // HKV
    for h in range(HKV):
        qs = slice(h * g, (h + 1) * g)
        cs = slice(h * g * HD, (h + 1) * g * HD)
        a = [t.to(dtype) for t in (q[:, qs], k[:, h : h + 1], v[:, h : h + 1], rel[:, qs], sink[:, qs], gate[:, cs])]
        res, saved = R.attention_full_fwd(*a, geo, is_global, rounding=True, p_bf16=emulate)
        dq, dk, dv, d_rel, d_sink, d_gate = R.attention_full_bwd(d_res[:, cs].to(dtype), saved, *a, geo, is_global,
                                                                  rounding=True, ds_bf16=emulate)
        for n, t in (("res", res), ("out_s", saved["out_s"]), ("lse_s", saved["lse_s"]), ("dq", dq), ("dk", dk),
                     ("dv", dv), ("d_rel", d_rel), ("d_sink", d_sink), ("d_gate", d_gate)):
            outs[n].append(t)
    cat = {"res": 1, "out_s": 1, "lse_s": 0, "dq": 1, "dk": 1, "dv": 1, "d_rel": 1, "d_sink": 1, "d_gate": 1}
    return {n: torch.cat(v, cat[n]).to(F64) for n, v in outs.items()}


def _kernel_attention(q, k, v, rel, sink, gate, d_res, cu_q, cu_k, num_segs, max_q, max_k, is_global, runs=1):
    tq, nk = q.shape[0], k.shape[0]
    qd, kd, vd, reld = bf(q), bf(k), bf(v), bf(rel)
    sinkd, gated, dresd = bf(sink), bf(gate), bf(d_res)
    cuq, cuk = i32(torch.tensor(cu_q)), i32(torch.tensor(cu_k))
    o = {}
    out_n = torch.empty(tq, HQ, HD, dtype=F32, device=DEV)
    lse_n = torch.empty(HQ, tq, dtype=F32, device=DEV)
    call("attention_part_fwd", 0, P(qd), P(kd), P(vd), P(cuq), P(cuk), P(None), num_segs, max_q, max_k, tq, HQ, HKV,
         HD ** -0.5, WIN, P(reld), EXT, P(out_n), P(lse_n))
    out_f = lse_f = None
    if is_global:
        out_f = torch.empty(tq, HQ, HD, dtype=F32, device=DEV)
        lse_f = torch.empty(HQ, tq, dtype=F32, device=DEV)
        seq = torch.empty(num_segs, dtype=torch.int32, device=DEV)
        call("attention_part_fwd", 1, P(qd), P(kd), P(vd), P(cuq), P(cuk), P(seq), num_segs, max_q, max_k, tq, HQ, HKV,
             HD ** -0.5, WIN, P(None), EXT, P(out_f), P(lse_f))
    res = torch.empty(tq, HQ * HD, dtype=BF16, device=DEV)
    out_s = torch.empty(tq, HQ, HD, dtype=F32, device=DEV)
    lse_s = torch.empty(HQ, tq, dtype=F32, device=DEV)
    call("attn_merge_fwd", P(res), P(out_s), P(lse_s), P(out_n), P(lse_n), P(out_f), P(lse_f), P(sinkd), P(gated), tq,
         HQ)
    o.update(res=res, out_s=out_s, lse_s=lse_s)
    dout = torch.empty(tq, HQ, HD, dtype=BF16, device=DEV)
    delta = torch.empty(HQ, tq, dtype=F32, device=DEV)
    d_sink = torch.empty(tq, HQ, dtype=F32, device=DEV)
    d_gate = torch.empty(tq, HQ * HD, dtype=F32, device=DEV)
    call("attn_merge_bwd_union", P(dout), P(delta), P(d_sink), P(d_gate), P(dresd), False, P(out_s), P(lse_s),
         P(sinkd), P(gated), tq, HQ)
    ws_bytes = 1 if DRY else lib().inkling_capi_attention_bwd_workspace_bytes(tq, max_q, num_segs, HQ)
    ws = torch.empty(ws_bytes, dtype=torch.uint8, device=DEV)
    grads = []
    for _ in range(runs):
        dq = torch.zeros(tq, HQ, HD, dtype=F32, device=DEV)
        dk = torch.zeros(nk, HKV, HD, dtype=F32, device=DEV)
        dv = torch.zeros(nk, HKV, HD, dtype=F32, device=DEV)
        d_rel = torch.empty(tq, HQ, EXT, dtype=F32, device=DEV)
        call("attention_part_bwd", 0, P(qd), P(kd), P(vd), P(cuq), P(cuk), num_segs, max_q, max_k, tq, nk, HQ, HKV,
             HD ** -0.5, WIN, P(reld), EXT, P(dout), P(lse_s), P(delta), P(dq), P(dk), P(dv), P(d_rel), P(ws))
        if is_global:
            call("attention_part_bwd", 1, P(qd), P(kd), P(vd), P(cuq), P(cuk), num_segs, max_q, max_k, tq, nk, HQ, HKV,
                 HD ** -0.5, WIN, P(None), EXT, P(dout), P(lse_s), P(delta), P(dq), P(dk), P(dv), P(None), P(ws))
        grads.append(dict(dq=dq, dk=dk, dv=dv, d_rel=d_rel))
    o.update(grads[0], d_sink=d_sink, d_gate=d_gate)
    return o, grads


def _check_attention(test, o, ref, pt):
    for n in ("res", "out_s", "lse_s"):
        x = cpu64(o[n]).reshape(ref[n].shape)
        r = ref[n]
        fin = torch.isfinite(r)
        record(test, n, TOL.rule_vs_pt(x[fin], pt[n].reshape(r.shape)[fin], r[fin], factor=2.0))
    for n in ("dq", "dk", "dv", "d_rel", "d_sink", "d_gate"):
        x = cpu64(o[n]).reshape(ref[n].shape)
        record(test, n, TOL.rule_vs_pt(x, pt[n].reshape(ref[n].shape), ref[n], factor=3.0))
    out_of_band = (ref["d_rel"] == 0) & (cpu64(o["d_rel"]).reshape(ref["d_rel"].shape) != 0)
    record(test, "d_rel_zero_outside_band", (int(out_of_band.sum()) == 0, dict(nonzero=int(out_of_band.sum()))))


def test_attention_whole_rows():
    for is_global in (True, False):
        q, k, v, rel, sink, gate, d_res = _attn_inputs(T, T, 10 + is_global)
        pos, doc = positions(), doc_ids()
        geo = R.AttnGeom(pos, doc, pos, doc, window=WIN, extent=EXT)
        cu = [0] + list(torch.tensor(DOCS).cumsum(0).tolist())
        o, grads = _kernel_attention(q, k, v, rel, sink, gate, d_res, cu, cu, len(DOCS), max(DOCS), max(DOCS),
                                     is_global, runs=2)
        name = f"attention_rows_{'global' if is_global else 'local'}"
        ref = _ref_attention(q, k, v, rel, sink, gate, d_res, geo, is_global, F64, False)
        pt = _ref_attention(q, k, v, rel, sink, gate, d_res, geo, is_global, F32, True)
        _check_attention(name, o, ref, pt)
        for n in ("dq", "dk", "dv", "d_rel"):
            record(name, "determinism_" + n, TOL.rule_bit_exact(grads[0][n], grads[1][n]))


def test_attention_key_prefix():
    """A chunk: 2,048 queries at positions 2,048 .. 4,095 of one document, keys 0 .. 4,095 (bottom-right aligned)."""
    tq, nk = PREFIX_Q, PREFIX_K
    q, k, v, rel, sink, gate, d_res = _attn_inputs(tq, nk, 20)
    pos_q, pos_k = torch.arange(nk - tq, nk), torch.arange(nk)
    geo = R.AttnGeom(pos_q, torch.zeros(tq, dtype=torch.long), pos_k, torch.zeros(nk, dtype=torch.long), WIN, EXT)
    o, _ = _kernel_attention(q, k, v, rel, sink, gate, d_res, [0, tq], [0, nk], 1, tq, nk, True)
    ref = _ref_attention(q, k, v, rel, sink, gate, d_res, geo, True, F64, False)
    pt = _ref_attention(q, k, v, rel, sink, gate, d_res, geo, True, F32, True)
    _check_attention("attention_key_prefix_global", o, ref, pt)


def test_attention_window_chunk():
    """Windowed visit, chunked (Q4): the keys are [the previous chunk's last W-1 rows | this chunk], packed documents:
    doc A started 3,000 tokens before this chunk (truncated to the tail), doc B starts inside the chunk. Segment key
    offsets as the wrapper's near_cu_from_window builds them; reference = the same queries over ALL their keys."""
    tq = PREFIX_Q
    tail = WIN - 1
    la = tq // 2 + 7  # doc A rows in this chunk; doc B the rest
    lb = tq - la
    a_before = 3000 if not SMALL else 1100  # doc A tokens before this chunk (> tail: truncated)
    q, k, v, rel, sink, gate, d_res = _attn_inputs(tq, tail + tq, 30)
    # key rows: tail (doc A, positions a_before - tail .. a_before - 1), doc A in chunk, doc B in chunk
    pos_k = torch.cat([torch.arange(a_before - tail, a_before), torch.arange(a_before, a_before + la), torch.arange(lb)])
    doc_k = torch.cat([torch.zeros(tail + la, dtype=torch.long), torch.ones(lb, dtype=torch.long)])
    pos_q = torch.cat([torch.arange(a_before, a_before + la), torch.arange(lb)])
    doc_q = torch.cat([torch.zeros(la, dtype=torch.long), torch.ones(lb, dtype=torch.long)])
    geo = R.AttnGeom(pos_q, doc_q, pos_k, doc_k, WIN, EXT)
    cu_q = [0, la, tq]
    cu_k = [0, tail + la, tail + tq]  # segment 0 = tail + doc A rows; segment 1 = doc B rows
    o, grads = _kernel_attention(q, k, v, rel, sink, gate, d_res, cu_q, cu_k, 2, max(la, lb), tail + la, False, runs=2)
    ref = _ref_attention(q, k, v, rel, sink, gate, d_res, geo, False, F64, False)
    pt = _ref_attention(q, k, v, rel, sink, gate, d_res, geo, False, F32, True)
    _check_attention("attention_window_chunk_local", o, ref, pt)
    for n in ("dq", "dk", "dv", "d_rel"):
        record("attention_window_chunk_local", "determinism_" + n, TOL.rule_bit_exact(grads[0][n], grads[1][n]))


# ---------------------------------------------------------------------------------------------------------------
# Backward-only entry points of the engine-op IR (kernels/inkling/visit_backward.h)
# ---------------------------------------------------------------------------------------------------------------
def rule_fp32_with_flips(x, ref64, rel=4e-6, max_frac=0.005, flip_rel=2.0 ** -7):
    """C.3 where the backward RECOMPUTES a bf16 forward value (conv output, rel bias): elements past 4e-6 max|ref|
    are allowed on <= 0.5 % (1-ulp flips of the recomputed bf16 value), and never past 2^-7 max|ref|."""
    r = ref64.double()
    err = (x.double() - r).abs()
    m = float(r.abs().max())
    frac = float((err > rel * m).double().mean())
    return frac <= max_frac and float(err.max()) <= flip_rel * m, dict(err=float(err.max()), max_ref=m, frac_over=frac)


def test_visit_attention_bwd():
    """attention_visit_bwd from the engine's (o, l) == the union backward (whole rows, 3 docs, global visit)."""
    torch.manual_seed(40)
    q, k, v, _, sink, gate, d_res = _attn_inputs(T, T, 41)
    rq = _clip_rows(torch.randn(T, HQ, 16, dtype=F64) * 2.0, 8.0).to(BF16).to(F64)
    proj = (torch.randn(16, EXT, dtype=F64) * 0.2).to(BF16).to(F64)
    rel = R.rb(torch.einsum("thk,ke->the", rq, proj))
    pos, doc = positions(), doc_ids()
    geo = R.AttnGeom(pos, doc, pos, doc, window=WIN, extent=EXT)
    g = HQ // HKV
    o_l = []
    for h in range(HKV):
        qs = slice(h * g, (h + 1) * g)
        on, ln = R.attn_part_fwd("near", q[:, qs], k[:, h:h + 1], v[:, h:h + 1], rel[:, qs], geo)
        of, lf = R.attn_part_fwd("far", q[:, qs], k[:, h:h + 1], v[:, h:h + 1], rel[:, qs], geo)
        l = torch.logaddexp(ln, lf)
        o = on * torch.exp(ln - l).T[..., None] + of * torch.where(torch.isfinite(lf), torch.exp(lf - l),
                                                                   torch.zeros_like(lf)).T[..., None]
        o_l.append((o, l))
    o64 = torch.cat([x[0] for x in o_l], 1)
    l64 = torch.cat([x[1] for x in o_l], 0)
    qside = torch.cat([torch.zeros(T, HQ * HD, dtype=F64), torch.zeros(T, HQ * 16, dtype=F64), gate, sink], 1)
    qsd = bf(qside)
    ld = qside.shape[1]
    cu = [0] + list(torch.tensor(DOCS).cumsum(0).tolist())
    dq = torch.empty(T, HQ, HD, dtype=F32, device=DEV)
    dk = torch.zeros(T, HKV, HD, dtype=F32, device=DEV)
    dv = torch.zeros(T, HKV, HD, dtype=F32, device=DEV)
    d_r = torch.empty(T, HQ, 16, dtype=F32, device=DEV)
    d_sink = torch.empty(T, HQ, dtype=F32, device=DEV)
    d_gate = torch.empty(T, HQ * HD, dtype=F32, device=DEV)
    nb = 1 if DRY else lib().inkling_capi_attention_visit_bwd_workspace_bytes(T, max(DOCS), len(DOCS), HQ, EXT)
    ws = torch.empty(nb, dtype=torch.uint8, device=DEV)
    gate_view = qsd[:, HQ * HD + HQ * 16:]
    sink_view = qsd[:, 2 * HQ * HD + HQ * 16:]
    call("attention_visit_bwd", True, P(bf(q)), P(bf(k)), P(bf(v)), P(i32(torch.tensor(cu))), P(i32(torch.tensor(cu))),
         len(DOCS), max(DOCS), max(DOCS), T, T, HQ, HKV, HD ** -0.5, WIN, EXT, P(bf(rq)), P(bf(proj)), P(f32(o64)),
         P(f32(l64)), ctypes.c_void_p(0 if DRY else gate_view.data_ptr()), L(ld),
         ctypes.c_void_p(0 if DRY else sink_view.data_ptr()), L(ld), P(bf(d_res)), False, P(dq), P(dk), P(dv), P(d_r),
         P(d_sink), P(d_gate), P(ws))
    ref = _ref_attention(q, k, v, rel, sink, gate, d_res, geo, True, F64, False)
    pt = _ref_attention(q, k, v, rel, sink, gate, d_res, geo, True, F32, True)
    o = dict(dq=dq, dk=dk, dv=dv, d_sink=d_sink, d_gate=d_gate)
    for n in ("dq", "dk", "dv", "d_sink", "d_gate"):
        record("visit_attention_bwd", n, TOL.rule_vs_pt(cpu64(o[n]).reshape(ref[n].shape), pt[n].reshape(ref[n].shape),
                                                         ref[n], factor=3.0))
    rdr = torch.einsum("the,ke->thk", ref["d_rel"], proj)
    pdr = torch.einsum("the,ke->thk", pt["d_rel"], proj)
    record("visit_attention_bwd", "d_r", TOL.rule_vs_pt(cpu64(d_r), pdr, rdr, factor=3.0))


def test_prologue_bwd():
    torch.manual_seed(42)
    for is_global in (False, True):
        name = "global" if is_global else "local"
        CK = HKV * HD
        pos = positions(DOCS, [9, 0, 0])
        tau = R.log_tau(pos + 9000).to(F32).to(F64) if is_global else None
        q_raw = (torch.randn(T, HQ, HD, dtype=F64) * 3).to(BF16).to(F64)
        r_raw = (torch.randn(T, HQ, 16, dtype=F64) * 3).to(BF16).to(F64)
        qside = torch.cat([q_raw.reshape(T, -1), r_raw.reshape(T, -1), torch.zeros(T, HQ * HD + HQ, dtype=F64)], 1)
        q_scale = (1 + 0.2 * torch.randn(HQ, HD, dtype=F64)).to(BF16).to(F64)
        r_scale = (1 + 0.2 * torch.randn(HQ, 16, dtype=F64)).to(BF16).to(F64)
        k_raw = (torch.randn(T, CK, dtype=F64) * 2).to(BF16).to(F64)
        v_raw = (torch.randn(T, CK, dtype=F64) * 2).to(BF16).to(F64)
        if is_global:
            kw = (torch.randn(4, CK, dtype=F64) * 0.5).to(BF16).to(F64)
            vw = (torch.randn(4, CK, dtype=F64) * 0.5).to(BF16).to(F64)
            kd = (torch.randn(T, HKV * 4, dtype=F64) * 0.3).to(BF16).to(F64)
            vd = (torch.randn(T, HKV * 4, dtype=F64) * 0.3).to(BF16).to(F64)
        else:
            kw = (torch.randn(CK, 4, dtype=F64) * 0.5).to(BF16).to(F64)
            vw = (torch.randn(CK, 4, dtype=F64) * 0.5).to(BF16).to(F64)
            kd = vd = None
        kc0 = torch.randn(3, CK, dtype=F64).to(BF16).to(F64)
        vc0 = torch.randn(3, CK, dtype=F64).to(BF16).to(F64)
        g = {n: torch.randn(*sh, dtype=F64).to(F32).to(F64) for n, sh in
             (("dq", (T, HQ, HD)), ("d_r", (T, HQ, 16)), ("dk", (T, HKV, HD)), ("dv", (T, HKV, HD)),
              ("d_gate", (T, HQ * HD)), ("d_sink", (T, HQ)), ("dkn", (3, CK)), ("dvn", (3, CK)))}
        keep = [i32(pos), bf(qside), bf(q_scale), bf(r_scale), f32(tau) if is_global else None, bf(k_raw), bf(v_raw),
                bf(kw), bf(vw), bf(kd) if is_global else None, bf(vd) if is_global else None, bf(kc0), bf(vc0),
                f32(g["dkn"]), f32(g["dvn"]), None, None]
        gk = [f32(g[n]) for n in ("dq", "d_r", "dk", "dv", "d_gate", "d_sink")]
        ld = qside.shape[1]
        outs = [torch.empty(T, ld, dtype=F32, device=DEV), torch.empty(T, CK, dtype=F32, device=DEV),
                torch.empty(T, CK, dtype=F32, device=DEV), torch.empty(T, HKV * 4, dtype=F32, device=DEV),
                torch.empty(T, HKV * 4, dtype=F32, device=DEV), torch.empty(3, CK, dtype=F32, device=DEV),
                torch.empty(3, CK, dtype=F32, device=DEV)]
        nb = 1 if DRY else lib().inkling_capi_attention_prologue_bwd_workspace_bytes(T, HQ, HKV, 16)
        ws = torch.empty(nb, dtype=torch.uint8, device=DEV)
        arr = lambda ts: (ctypes.c_void_p * len(ts))(*[ctypes.c_void_p(t.data_ptr() if t is not None else 0) for t in ts])
        call("attention_prologue_bwd", T, HQ, HKV, 16, is_global, L(ld), L(ld), 22.627416997969522, 8.0,
             22.627416997969522, 1e-6, arr(keep), arr(gk), arr(outs + [ws]))
        # references (fp64), the forward recomputed with the engine's bf16 rounding of the conv outputs
        rdq = R.qkv_norm_bwd(g["dq"], q_raw, q_scale, tau, th=22.627416997969522)
        rdr = R.qkv_norm_bwd(g["d_r"], r_raw, r_scale, tau, th=8.0)
        res = {}
        for nm, raw, w_, dyn, c0, dk_, dn in (("k", k_raw, kw, kd, kc0, g["dk"], g["dkn"]),
                                              ("v", v_raw, vw, vd, vc0, g["dv"], g["dvn"])):
            if is_global:
                conv, _ = R.dynamic_conv_fwd(raw, w_, dyn, pos, c0)
            else:
                conv, _ = R.short_conv_fwd(raw, w_, pos, c0)
            mode = R.CLIP if nm == "k" else R.RMS
            dconv = R.qkv_norm_bwd(dk_, conv.reshape(T, HKV, HD), mode=mode).reshape(T, CK)
            if is_global:
                dx, ddyn, dc = R.dynamic_conv_bwd(dconv, raw, w_, dyn, pos, c0, dn, scale=9 ** -0.5)
            else:
                dx, dc, _, _ = R.short_conv_bwd(dconv, w_, pos, dn)
                ddyn = None
            res[nm] = (dx, ddyn, dc)
        dqs = cpu64(outs[0])
        record("prologue_bwd", name + "/d_q_raw", rule_fp32_with_flips(dqs[:, :HQ * HD], rdq.reshape(T, -1)))
        record("prologue_bwd", name + "/d_r_raw", rule_fp32_with_flips(dqs[:, HQ * HD:HQ * HD + HQ * 16], rdr.reshape(T, -1)))
        record("prologue_bwd", name + "/d_gate_copy", TOL.rule_bit_exact(f32(dqs[:, HQ * HD + HQ * 16:2 * HQ * HD + HQ * 16]),
                                                                         f32(g["d_gate"])))
        record("prologue_bwd", name + "/d_sink_copy", TOL.rule_bit_exact(f32(dqs[:, 2 * HQ * HD + HQ * 16:]), f32(g["d_sink"])))
        for i, nm in enumerate(("k", "v")):
            dx, ddyn, dc = res[nm]
            record("prologue_bwd", f"{name}/d_{nm}_raw", rule_fp32_with_flips(cpu64(outs[1 + i]), dx))
            record("prologue_bwd", f"{name}/d_{nm}_carry", rule_fp32_with_flips(cpu64(outs[5 + i]), dc))
            if is_global:
                record("prologue_bwd", f"{name}/d_{nm}_dyn", rule_fp32_with_flips(cpu64(outs[3 + i]), ddyn))


def test_kv_plane_gather():
    torch.manual_seed(43)
    pages, rows, pos0 = 12, 600, 70
    codes = torch.randint(0, 256, (pages * HKV * 64 * 128,), dtype=torch.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0  # no NaN codes
    table = torch.randperm(pages).to(torch.int32)
    dst = torch.empty(rows, HKV, HD, dtype=BF16, device=DEV)
    call("kv_plane_gather_bf16", P(dst), P(codes.to(DEV)), P(table.to(DEV)), L(pos0), L(rows), HKV)
    vals = codes.view(torch.float8_e4m3fn).to(F64)
    p = torch.arange(pos0, pos0 + rows)
    page = table.to(torch.long)[p >> 6]
    h = torch.arange(HKV)
    d = torch.arange(HD)
    off = (128 * 64 * (h[None, :, None] + HKV * page[:, None, None]) + 128 * (p & 63)[:, None, None] + d[None, None, :])
    record("kv_plane_gather", "exact", TOL.rule_bit_exact(cpu64(dst), vals[off]))


# ---------------------------------------------------------------------------------------------------------------
# 8. AttnRes
# ---------------------------------------------------------------------------------------------------------------
def test_attn_res():
    torch.manual_seed(4)
    D = C_MODEL
    for name, N, Q, norm, fp32_last, addend in (("Q3_norm", 5, 3, True, False, False),
                                                ("Q1_addend_fp32", 4, 1, False, True, True)):
        srcs = [torch.randn(T, D, dtype=F64).to(BF16).to(F64) for _ in range(N)]
        if fp32_last:
            srcs[-1] = torch.randn(T, D, dtype=F64).to(F32).to(F64)
        add = torch.randn(T, D, dtype=F64).to(BF16).to(F64) if addend else None
        qraw = (torch.randn(Q, D, dtype=F64) * 0.05).to(F32).to(F64)
        qs = D ** -0.5
        nw = (1 + 0.1 * torch.randn(Q, D, dtype=F64)).to(BF16).to(F64) if norm else None
        dev_src = [f32(s) if (fp32_last and i == N - 1) else bf(s) for i, s in enumerate(srcs)]
        arr = (ctypes.c_void_p * 8)(*[ctypes.c_void_p(s.data_ptr()) for s in dev_src])
        fl = (ctypes.c_int * 8)(*[int(fp32_last and i == N - 1) for i in range(N)])
        out = torch.empty(Q, T, D, dtype=BF16 if norm else F32, device=DEV)
        probs = torch.empty(Q, N, T, dtype=F32, device=DEV)
        rstd = torch.empty(N, T, dtype=F32, device=DEV)
        rstd_o = torch.empty(Q, T, dtype=F32, device=DEV)
        addd = bf(add) if addend else None
        qd = f32(qraw)
        call("attn_res_fwd", P(out), P(probs), P(rstd), P(rstd_o) if norm else P(None), arr, fl, N,
             P(addd) if addend else P(None), False, P(qd), float(qs), P(bf(nw)) if norm else P(None), 1e-6, Q, T, D)
        rsrc = list(srcs)
        if addend:
            rsrc[-1] = rsrc[-1] + add
        q32 = (qraw.to(F32) * torch.tensor(qs, dtype=F32)).to(F64)
        ref, saved = R.attn_res_fwd(rsrc, q32, nw)
        if norm:
            record("attn_res_fwd", name, TOL.rule_bf16_output(cpu64(out), ref))
        else:
            record("attn_res_fwd", name, TOL.rule_fp32_path(cpu64(out), ref))
        dy = torch.randn(Q, T, D, dtype=F64).to(F32).to(F64)
        dsrc = [torch.zeros(T, D, dtype=F32, device=DEV) for _ in range(N)]
        darr = (ctypes.c_void_p * 8)(*[ctypes.c_void_p(t.data_ptr()) for t in dsrc])
        dadd = torch.zeros(T, D, dtype=F32, device=DEV) if addend else None
        call("attn_res_bwd", darr, P(dadd) if addend else P(None), P(f32(dy)), True, P(probs), P(rstd),
             P(rstd_o) if norm else P(None), arr, fl, N, P(addd) if addend else P(None), False, P(qd), float(qs),
             P(bf(nw)) if norm else P(None), 1e-6, Q, T, D)
        rdx = R.attn_res_bwd(dy, rsrc, q32, nw, saved)
        for i in range(N):
            record("attn_res_bwd", f"{name}/d_src{i}", TOL.rule_fp32_path(cpu64(dsrc[i]), rdx[i]))
        if addend:
            record("attn_res_bwd", f"{name}/d_addend", TOL.rule_fp32_path(cpu64(dadd), rdx[N - 1]))


# ---------------------------------------------------------------------------------------------------------------
# 9. loop blend (elementwise kernels + the wrapper's GEMMs done with torch here, same operand types)
# ---------------------------------------------------------------------------------------------------------------
def test_loop_blend():
    torch.manual_seed(5)
    D, Rk = C_MODEL, 64
    for name, S, bcast in (("S3_broadcast", 3, True), ("S1", 1, False)):
        dest = torch.randn(1 if bcast else S, T, D, dtype=F64).to(F32).to(F64)
        msg = torch.randn(S, T, D, dtype=F64).to(F32).to(F64)
        down = (torch.randn(S, Rk, 2 * D, dtype=F64) * 0.02).to(BF16).to(F64)
        gfw = (torch.randn(S, 2 * D, Rk, dtype=F64) * 0.1).to(BF16).to(F64)
        nw = (1 + 0.1 * torch.randn(1, D, dtype=F64)).to(BF16).to(F64)
        dd, md = f32(dest), f32(msg)
        stride = 0 if bcast else T * D
        pair = torch.empty(S, T, 2 * D, dtype=BF16, device=DEV)
        call("loop_blend_pair_fwd", P(pair), P(dd), L(stride), P(md), S, T, D)
        h = torch.stack([pair[s] @ bf(down[s]).T for s in range(S)])  # bf16 out, fp32 accumulate
        gf = torch.stack([h[s].float() @ f32(gfw[s]).T for s in range(S)]).contiguous()
        out = torch.empty(S, T, D, dtype=BF16, device=DEV)
        rx = torch.empty(S, T, dtype=F32, device=DEV)
        call("loop_blend_out_fwd", P(out), P(rx), P(dd), L(stride), P(md), P(gf), P(bf(nw)), L(0), 1e-6, S, T, D)
        ref, saved = R.loop_blend_fwd(dest, msg, down, gfw, nw)
        # The kernels' forward is judged at the GEMMs' own h: a 1-ulp bf16 flip of h (a 4,608-term fp32 dot product vs
        # fp64) moves gf, then tanh, rx and every output of the row (M3 diag 10:44: out 1 ulp at the kernel's gf,
        # 23,205 ulps against the fp64 h). The GEMM's h is checked on its own (bf16 output rule).
        # the GEMM (cuBLAS in the op wrapper, torch here) against fp64 of the SAME bf16 pair (pair flips of 1 ulp are
        # the pair kernel's, checked through the outputs; a 4,608-term sum from a flipped pair is not a GEMM error)
        h_ref = R.rb(torch.stack([cpu64(pair[i]) @ down[i].T for i in range(S)]))
        # informational (not g2 code: the wrapper's GEMM is the trainer's cuBLASLt): near-zero sums of 4,608 terms are
        # cancellation residues beyond the rule's floor (run 5: 0.05 % flips, max 12 ulp on tiny h)
        ok_h, rep_h = TOL.rule_bf16_output(cpu64(h), h_ref, max_flip_frac=0.02)
        print(f"  INFO loop_blend_fwd/{name}/h_gemm {rep_h}", flush=True)
        g64 = cpu64(gf)
        dexp = dest.expand(S, T, D)
        x_k = (1 - torch.tanh(g64[..., D:])) * dexp + torch.tanh(g64[..., :D]) * msg
        ref_k = R.rb(x_k * torch.rsqrt((x_k * x_k).mean(-1, keepdim=True) + 1e-6) * nw.reshape(1, 1, D))
        record("loop_blend_fwd", name, TOL.rule_bf16_output(cpu64(out), ref_k, max_flip_frac=0.02))
        dy = torch.randn(S, T, D, dtype=F64).to(F32).to(F64)
        dds = torch.empty(S, T, D, dtype=F32, device=DEV)
        dmsg = torch.empty(S, T, D, dtype=F32, device=DEV)
        dgf = torch.empty(S, T, 2 * D, dtype=F32, device=DEV)
        call("loop_blend_out_bwd", P(dds), P(dmsg), P(dgf), P(f32(dy)), True, P(dd), L(stride), P(md), P(gf), P(rx),
             P(bf(nw)), L(0), S, T, D)
        dh = torch.stack([dgf[s] @ f32(gfw[s]) for s in range(S)])
        dpair = torch.stack([dh[s] @ f32(down[s]) for s in range(S)]).contiguous()
        call("loop_blend_pair_bwd", P(dds), P(dmsg), P(dpair), P(dd), L(stride), P(md), S, T, D)
        ddest = torch.empty_like(dd)
        call("loop_blend_reduce_dest", P(ddest), P(dds), bcast, False, S, T, D)
        # Reference backward at the kernel's own forward point (its gf: the bf16 rounding of h by the GEMM is a
        # forward detail, C.1 above): C.3, a pure fp32 path.
        g64 = cpu64(gf)
        dst = dest.expand(S, T, D)
        gain, forget = torch.tanh(g64[..., :D]), torch.tanh(g64[..., D:])
        x = (1.0 - forget) * dst + gain * msg
        sk = dict(x=x, rx=torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6), gain=gain, forget=forget,
                  rd=torch.rsqrt((dst * dst).mean(-1, keepdim=True) + 1e-6),
                  rm=torch.rsqrt((msg * msg).mean(-1, keepdim=True) + 1e-6))
        rdd, rdm = R.loop_blend_bwd(dy, dest, msg, down, gfw, nw, sk, bcast)
        record("loop_blend_bwd", name + "/d_dest", TOL.rule_fp32_path(cpu64(ddest), rdd))
        record("loop_blend_bwd", name + "/d_msg", TOL.rule_fp32_path(cpu64(dmsg), rdm))


# ---------------------------------------------------------------------------------------------------------------
# 10 / 11. routed softcapped SwiGLU, MoE combine
# ---------------------------------------------------------------------------------------------------------------
def _routing(t, k, e=64, seed=6):
    torch.manual_seed(seed)
    experts = torch.stack([torch.randperm(e)[:k] for _ in range(t)])  # [t, k]
    order = torch.argsort(experts.reshape(-1), stable=True)  # expert-sorted rows
    scatter = torch.empty(t * k, dtype=torch.long)
    scatter[order] = torch.arange(t * k)
    return scatter


def test_swiglu_and_combine():
    torch.manual_seed(7)
    k, I, D, t = 8, 1152, C_MODEL, 1024
    scatter = _routing(t, k)
    gu = (torch.randn(t * k, 2 * I, dtype=F64) * 3).to(BF16).to(F64)
    w = torch.rand(t, k, dtype=F64).to(F32).to(F64)
    y = torch.empty(t * k, I, dtype=BF16, device=DEV)
    sd = i32(scatter)
    call("softcap_swiglu_fwd", P(y), P(bf(gu)), P(f32(w)), P(sd), t * k, I, 4.0, 25.0)
    ref = R.swiglu_routed_fwd(gu, w, scatter)
    record("softcap_swiglu_fwd", "routed", TOL.rule_bf16_output(cpu64(y), ref, max_flip_frac=0.15))
    dy = torch.randn(t * k, I, dtype=F64).to(BF16).to(F64)
    dgu = torch.empty(t * k, 2 * I, dtype=BF16, device=DEV)
    dw = torch.empty(t, k, dtype=F32, device=DEV)
    call("softcap_swiglu_bwd", P(dgu), P(dw), P(bf(dy)), False, P(bf(gu)), P(f32(w)), P(sd), t * k, I, 4.0, 25.0)
    rdgu, rdw = R.swiglu_routed_bwd(dy, gu, w, scatter)
    # d_gate_up is STORED bf16 (the grouped GEMM backward's operand): compare with the rounded fp64 value
    record("softcap_swiglu_bwd", "d_gate_up", TOL.rule_bf16_output(cpu64(dgu), R.rb(rdgu), max_flip_frac=0.15))
    record("softcap_swiglu_bwd", "d_weights", TOL.rule_fp32_path(cpu64(dw), rdw, rel=1e-5))
    rows = torch.randn(t * k, D, dtype=F64).to(BF16).to(F64)
    out = torch.empty(t, D, dtype=BF16, device=DEV)
    call("moe_combine_fwd", P(out), P(bf(rows)), P(sd), t, k, D)
    record("moe_combine_fwd", "unit", TOL.rule_bf16_output(cpu64(out), R.moe_combine_fwd(rows, scatter, k)))
    dout = torch.randn(t, D, dtype=F64).to(F32).to(F64)
    drows = torch.empty(t * k, D, dtype=BF16, device=DEV)
    call("moe_combine_bwd", P(drows), P(f32(dout)), True, P(sd), t, k, D)
    record("moe_combine_bwd", "unit",
           TOL.rule_bf16_output(cpu64(drows), R.rb(R.moe_combine_bwd(dout, scatter, k, t * k))))


TESTS = [test_log_tau, test_qkv_norm, test_short_conv, test_dynamic_conv, test_attn_res, test_loop_blend,
         test_swiglu_and_combine, test_attention_whole_rows, test_attention_key_prefix, test_attention_window_chunk,
         test_visit_attention_bwd, test_prologue_bwd, test_kv_plane_gather]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))
    print(f"lib {LIB}\ndevice {'cpu (dry run)' if DRY else torch.cuda.get_device_name()}  T={T}", flush=True)
    for fn in TESTS:
        if only and fn.__name__ not in only:
            continue
        t0 = time.time()
        print(f"[{fn.__name__}]", flush=True)
        try:
            fn()
        except Exception as ex:  # a crash is a FAIL of that test, the others still run
            RESULTS.append(dict(test=fn.__name__, check="exception", ok=False, error=repr(ex)))
            print(f"  FAIL {fn.__name__} exception {ex!r}", flush=True)
        print(f"  ({time.time() - t0:.1f} s)", flush=True)
    n_ok = sum(r["ok"] for r in RESULTS)
    print(f"SUMMARY {n_ok}/{len(RESULTS)} checks PASS", flush=True)
    with open(os.environ.get("INKLING_REPORT", "inkling2_kernels_gpu_report.json"), "w") as f:
        json.dump(RESULTS, f, indent=1)
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
