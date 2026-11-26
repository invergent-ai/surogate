"""The engine-op trainer's CUDA backward kernels (csrc/src/kernels/inkling, library libsinfer_inkling_bwd.so), loaded with
ctypes. Each function here replaces one torch recompute + autograd section of the visit backward and is checked against it
(tests/lever_check.py arms, tests/test_inkling2_moe_bwd_gpu.py). Library path: env IFT_BWD_LIB.

moe_groups: the routed experts of one visit, group-major (ift_long.MoEGM's row loop): the cotangents of the MLP input, the
routing weights and the layer's LoRA tensors.
"""
from __future__ import annotations

import ctypes
import os

import torch

F32, BF16 = torch.float32, torch.bfloat16
EG = 16  # experts decoded and handled per call (ift_backward.EG)

_LIB = None
_WS = {}


def lib():
    global _LIB
    if _LIB is None:
        path = os.environ.get("IFT_BWD_LIB")
        if not path:
            raise RuntimeError("IFT_BWD_LIB is not set (path of libsinfer_inkling_bwd.so)")
        _LIB = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        _LIB.inkling_capi_last_error.restype = ctypes.c_char_p
        _LIB.inkling_capi_moe_group_bwd_workspace_bytes.restype = ctypes.c_long
    return _LIB


def _workspace(key, nbytes):
    """A cached device byte buffer per key (a device, or (name, device))."""
    t = _WS.get(key)
    if t is None or t.numel() < nbytes:
        dev = key[1] if isinstance(key, tuple) else key
        _WS[key] = t = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    return t


def _ptr(t):
    return None if t is None else t.data_ptr()


def _vp(t):
    """A tensor's device pointer as a ctypes argument (a bare int would be passed as a C int)."""
    return None if t is None else ctypes.c_void_p(t.data_ptr())


def moe_groups(w, x16, g, tok, gam_r, starts, cnt, lora, scale, dx, dgam_r, lgr, max_rows):
    """w: ift_backward.EngineExperts of the layer. x16 bf16 [T, D] (the MLP input), g fp32 [T, D] (cotangent of the MoE
    output), tok int64 [R] / gam_r fp32 [R] (rows sorted by expert), starts / cnt: host lists per expert. lora: the layer's
    (A13, B13, A2, B2) fp32 tensors or None. Adds into dx fp32 [T, D]; writes dgam_r [R]; adds into lgr[j] (None = that
    tensor's gradient is not needed)."""
    L = lib()
    dev = x16.device
    T, D = x16.shape
    n_exp = len(cnt)
    I = D // 2
    rank = lora[0].shape[1] if lora is not None else 0
    x16 = x16.contiguous()
    g = g.contiguous()
    tok32 = tok.to(torch.int32).contiguous()
    gam_r = gam_r.contiguous()
    cap = max_rows if max_rows > 0 else 8192
    wsb = L.inkling_capi_moe_group_bwd_workspace_bytes(cap, D, I, rank)
    ws = _workspace(dev, wsb)
    if lora is not None and any(not t.is_contiguous() for t in lora):
        raise ValueError("moe_groups: the LoRA tensors must be contiguous")
    torch.cuda.current_stream(dev).synchronize()  # the kernels run on the default stream
    for g0 in range(0, n_exp, EG):
        g1 = min(n_exp, g0 + EG)
        r0, r1 = starts[g0], starts[g1 - 1] + cnt[g1 - 1]
        if r1 == r0:
            continue
        ng = g1 - g0
        u13, f13 = w.eng.decode_experts(w.layer, "w13", g0, ng)
        u2, f2 = w.eng.decode_experts(w.layer, "w2", g0, ng)
        if w.fmt == "nvfp4":
            f13h, f2h = f13.tolist(), f2.tolist()
        else:
            f13h, f2h = [1.0] * ng, [1.0] * ng
        torch.cuda.current_stream(dev).synchronize()
        seg = (ctypes.c_int * (ng + 1))(*([starts[e] for e in range(g0, g1)] + [r1]))
        lo = [None] * 4 if lora is None else [t[g0:g1] for t in lora]
        gr = [None] * 4 if lora is None else [None if t is None else t[g0:g1] for t in lgr]
        ints = (ctypes.c_int * 6)(T, D, I, ng, rank, cap)
        ptrs = (ctypes.c_void_p * 17)(_ptr(x16), _ptr(g), _ptr(tok32), _ptr(gam_r), _ptr(u13), _ptr(u2), *[_ptr(t) for t in lo],
                                       _ptr(dx), _ptr(dgam_r), *[_ptr(t) for t in gr], _ptr(ws))
        rc = L.inkling_capi_moe_group_bwd(ints, ptrs, seg, (ctypes.c_float * ng)(*f13h), (ctypes.c_float * ng)(*f2h),
                                          ctypes.c_float(float(scale) if scale is not None else 0.0), ctypes.c_long(wsb), None)
        if rc != 0:
            raise RuntimeError("inkling_capi_moe_group_bwd: " + L.inkling_capi_last_error().decode())
        torch.cuda.synchronize(dev)  # the call has finished before the decoded units go back to the allocator
        del u13, u2


# ---------------------------------------------------------------------------------------------------------------
# attention with the relative bias: the flash kernels (near part: distance < 1024 + bias; far part on global visits)
# ---------------------------------------------------------------------------------------------------------------
ATTN_Q_ROWS = int(os.environ.get("IFT_ATTN_CUDA_ROWS", "4096"))  # query rows per call (bounds rel / d_rel: 147 KB per row)
_HQ, _HKV, _HD, _EXT, _WIN = 24, 8, 128, 1024, 1024
_SCALE = 128 ** -0.5
_ATT_READY = False


def _att_lib():
    global _ATT_READY
    L = lib()
    if not _ATT_READY:
        L.inkling_capi_attention_bwd_workspace_bytes.restype = ctypes.c_long
        _ATT_READY = True
    return L


def _check(L, rc, what):
    if rc != 0:
        raise RuntimeError(what + ": " + L.inkling_capi_last_error().decode())


def _cu(n, dev):
    return torch.tensor([0, n], dtype=torch.int32, device=dev)


def _off(t, rows):
    """Device pointer of row `rows` of a contiguous tensor [T, ...]."""
    return ctypes.c_void_p(t.data_ptr() + rows * t[0].numel() * t.element_size())


class RelAttentionCuda(torch.autograd.Function):
    """ift_long.RelAttentionR on the kernel library: (out [T, 24, 128], lse [T, 24]) of one softmax over the visible keys
    with the relative bias rel = bf16(r . rel_proj) at distances < 1024; local visits (window 1024) see only those keys.
    Backward: dq, dk, dv, d r (rel_proj frozen), straight through the bias rounding. The cotangent of `out` enters the
    kernels rounded to bf16 (tensor-core flash backward), as the experts' unit GEMMs do."""

    @staticmethod
    def forward(ctx, q, k, v, r_out, rel_proj, window, passes=1):
        L = _att_lib()
        ctx.passes = passes
        dev = q.device
        T = q.shape[0]
        is_global = window == 0
        q16, k16, v16 = q.detach().to(BF16).contiguous(), k.detach().to(BF16).contiguous(), v.detach().to(BF16).contiguous()
        r16 = r_out.detach().to(BF16).contiguous()
        out = torch.empty(T, _HQ, _HD, dtype=F32, device=dev)
        lse = torch.empty(T, _HQ, dtype=F32, device=dev)
        seq = torch.zeros(1, dtype=torch.int32, device=dev)
        relp = rel_proj.float()
        torch.cuda.current_stream(dev).synchronize()
        for b0 in range(0, T, ATTN_Q_ROWS):
            b1 = min(T, b0 + ATTN_Q_ROWS)
            n = b1 - b0
            k0 = max(0, b0 - (_WIN - 1))
            rel = torch.einsum("thj,jd->thd", r16[b0:b1].float(), relp).to(BF16).contiguous()
            o_n = torch.empty(n, _HQ, _HD, dtype=F32, device=dev)
            l_n = torch.empty(_HQ, n, dtype=F32, device=dev)
            cq, ck = _cu(n, dev), _cu(b1 - k0, dev)
            _check(L, L.inkling_capi_attention_part_fwd(0, _off(q16, b0), _off(k16, k0), _off(v16, k0), _vp(cq), _vp(ck), None, 1, n,
                                                        b1 - k0, n, _HQ, _HKV, ctypes.c_float(_SCALE), _WIN, _vp(rel), _EXT,
                                                        _vp(o_n), _vp(l_n), None), "attention near fwd")
            if is_global and b1 > _WIN:
                o_f = torch.empty(n, _HQ, _HD, dtype=F32, device=dev)
                l_f = torch.empty(_HQ, n, dtype=F32, device=dev)
                ckf = _cu(b1, dev)
                _check(L, L.inkling_capi_attention_part_fwd(1, _off(q16, b0), _vp(k16), _vp(v16), _vp(cq), _vp(ckf), _vp(seq), 1, n,
                                                            b1, n, _HQ, _HKV, ctypes.c_float(_SCALE), _WIN, None, _EXT, _vp(o_f),
                                                            _vp(l_f), None), "attention far fwd")
                torch.cuda.synchronize(dev)
                ln, lf = l_n.T, l_f.T                                    # [n, Hq]
                lt = torch.logaddexp(ln, lf)
                wf = torch.exp(lf - lt)[..., None]
                out[b0:b1] = o_n * torch.exp(ln - lt)[..., None] + torch.where(wf > 0, o_f * wf, torch.zeros((), device=dev))
                lse[b0:b1] = lt
                del o_f, l_f
            else:
                torch.cuda.synchronize(dev)
                out[b0:b1] = o_n
                lse[b0:b1] = l_n.T
            del rel, o_n, l_n
        ctx.save_for_backward(q16, k16, v16, r16, out, lse)
        ctx.rel_proj, ctx.is_global = relp, is_global
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        L = _att_lib()
        q16, k16, v16, r16, out, lse = ctx.saved_tensors
        dev = q16.device
        T = q16.shape[0]
        relp = ctx.rel_proj
        if dout is None:
            dout = torch.zeros_like(out)
        # the kernels take the cotangent in bf16: one pass with its rounding, or two passes with its bf16 high and low parts
        # (the backward is linear in (dout, delta), so the passes add up to the fp32 cotangent's result within 2^-17)
        hi = dout.to(BF16).contiguous()
        parts = [hi]
        if ctx.passes == 2:
            parts.append((dout - hi.float()).to(BF16).contiguous())
        deltas = [(p.float() * out).sum(-1) for p in parts]            # [T, Hq] each
        if dlse is not None:
            deltas[0] = deltas[0] - dlse
        dq = torch.zeros(T, _HQ, _HD, dtype=F32, device=dev)
        dk = torch.zeros(T, _HKV, _HD, dtype=F32, device=dev)
        dv = torch.zeros(T, _HKV, _HD, dtype=F32, device=dev)
        dr = torch.zeros(T, _HQ, r16.shape[2], dtype=F32, device=dev)
        torch.cuda.current_stream(dev).synchronize()
        for b0 in range(0, T, ATTN_Q_ROWS):
            b1 = min(T, b0 + ATTN_Q_ROWS)
            n = b1 - b0
            k0 = max(0, b0 - (_WIN - 1))
            rel = torch.einsum("thj,jd->thd", r16[b0:b1].float(), relp).to(BF16).contiguous()
            lse_c = lse[b0:b1].T.contiguous()
            d_rel = torch.empty(n, _HQ, _EXT, dtype=F32, device=dev)
            wsb = L.inkling_capi_attention_bwd_workspace_bytes(n, n, 1, _HQ)
            ws = _workspace(("attn", dev), wsb)
            cq, ck = _cu(n, dev), _cu(b1 - k0, dev)
            for d16, delta in zip(parts, deltas):
                del_c = delta[b0:b1].T.contiguous()
                _check(L, L.inkling_capi_attention_part_bwd(0, _off(q16, b0), _off(k16, k0), _off(v16, k0), _vp(cq), _vp(ck), 1, n,
                                                            b1 - k0, n, b1 - k0, _HQ, _HKV, ctypes.c_float(_SCALE), _WIN, _vp(rel),
                                                            _EXT, _off(d16, b0), _vp(lse_c), _vp(del_c), _off(dq, b0), _off(dk, k0),
                                                            _off(dv, k0), _vp(d_rel), _vp(ws), None), "attention near bwd")
                if ctx.is_global and b1 > _WIN:
                    ckf = _cu(b1, dev)
                    _check(L, L.inkling_capi_attention_part_bwd(1, _off(q16, b0), _vp(k16), _vp(v16), _vp(cq), _vp(ckf), 1, n, b1, n,
                                                                b1, _HQ, _HKV, ctypes.c_float(_SCALE), _WIN, None, _EXT,
                                                                _off(d16, b0), _vp(lse_c), _vp(del_c), _off(dq, b0), _vp(dk),
                                                                _vp(dv), None, _vp(ws), None), "attention far bwd")
                torch.cuda.synchronize(dev)
                dr[b0:b1] += torch.einsum("thd,jd->thj", d_rel, relp)
                del del_c
            del rel, d_rel, lse_c
        return dq, dk, dv, dr, None, None, None
