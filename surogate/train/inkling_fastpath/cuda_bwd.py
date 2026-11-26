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


def _tick(name):
    """ift_backward's profiling section timer (IFT_PROFILE=1), imported late: this module has no import-time dependency on it."""
    import ift_backward as B

    return B._tick(name)


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
        with _tick("bwd.moe.cuda.decode"):
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
        with _tick("bwd.moe.cuda.kernel"):
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
        if passes == 3:  # diagnostic: the recompute's (out, lse) from the torch attention (fp32 probabilities), the backward from the kernels
            import ift_long as LG

            class _Ctx:
                def save_for_backward(self, *a):
                    pass

            out, lse = LG.RelAttentionR.forward(_Ctx(), q16.float(), k16.float(), v16.float(), r16.float(), rel_proj, window,
                                                LG._q_block(T, window))
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


# ---------------------------------------------------------------------------------------------------------------
# hand-written backward of the token-chunked stages (ift_long.TokenChunked calls fn.cuda_vjp per chunk instead of
# recomputing fn under autograd): block AttnRes and the loop blend on the kernel library, the small controller GEMMs in torch
# ---------------------------------------------------------------------------------------------------------------
_EPS = 1e-6


def _res_fwd(srcs, addend, query, norm_w):
    """Block AttnRes of fp32 sources [n, D] (the last one + addend), fp32 pre-scaled queries [Q, D]; with norm_w (bf16 [Q, D])
    the output is bf16 [Q, n, D], else the fp32 mixes. Returns the saved state for _res_bwd."""
    L = lib()
    dev = srcs[0].device
    N, (n, D), Q = len(srcs), srcs[0].shape, query.shape[0]
    srcs = [s.contiguous() for s in srcs]
    q = query.float().contiguous()
    nw = norm_w.contiguous() if norm_w is not None else None
    ad = addend.contiguous() if addend is not None else None
    out = torch.empty(Q, n, D, dtype=BF16 if nw is not None else F32, device=dev)
    probs = torch.empty(Q, N, n, dtype=F32, device=dev)
    rstd = torch.empty(N, n, dtype=F32, device=dev)
    rstd_out = torch.empty(Q, n, dtype=F32, device=dev) if nw is not None else None
    sp = (ctypes.c_void_p * N)(*[s.data_ptr() for s in srcs])
    sf = (ctypes.c_int * N)(*([1] * N))
    _check(L, L.inkling_capi_attn_res_fwd(_vp(out), _vp(probs), _vp(rstd), _vp(rstd_out), sp, sf, N, _vp(ad), 1, _vp(q),
                                          ctypes.c_float(1.0), _vp(nw), ctypes.c_float(_EPS), Q, n, D, None), "attn_res_fwd")
    return {"srcs": srcs, "ad": ad, "q": q, "nw": nw, "out": out, "probs": probs, "rstd": rstd, "rstd_out": rstd_out,
            "sp": sp, "sf": sf}


def _res_bwd(sv, d_out):
    """d_out fp32 [Q, n, D] -> (the sources' gradients [n, D] each, the addend's gradient or None)."""
    L = lib()
    srcs = sv["srcs"]
    dev = srcs[0].device
    N, (n, D), Q = len(srcs), srcs[0].shape, sv["q"].shape[0]
    d_out = d_out.contiguous()
    d_src = [torch.zeros(n, D, dtype=F32, device=dev) for _ in range(N)]
    d_add = torch.zeros(n, D, dtype=F32, device=dev) if sv["ad"] is not None else None
    dp = (ctypes.c_void_p * N)(*[t.data_ptr() for t in d_src])
    _check(L, L.inkling_capi_attn_res_bwd(dp, _vp(d_add), _vp(d_out), 1, _vp(sv["probs"]), _vp(sv["rstd"]), _vp(sv["rstd_out"]),
                                          sv["sp"], sv["sf"], N, _vp(sv["ad"]), 1, _vp(sv["q"]), ctypes.c_float(1.0),
                                          _vp(sv["nw"]), ctypes.c_float(_EPS), Q, n, D, None), "attn_res_bwd")
    return d_src, d_add


def _blend_fwd(dest, msg, down, gf_w, norm_w):
    """Loop blend of S streams (dest fp32 [S, n, D], msg fp32 [S, n, D], down bf16 [S, r, 2D], gf_w bf16 [S, 2D, r],
    norm_w bf16 [S, D]): the saved state with out bf16 [S, n, D]."""
    L = lib()
    dev = dest.device
    S, n, D = msg.shape
    dest, msg = dest.contiguous(), msg.contiguous()
    nw = norm_w.contiguous()
    pair = torch.empty(S, n, 2 * D, dtype=BF16, device=dev)
    _check(L, L.inkling_capi_loop_blend_pair_fwd(_vp(pair), _vp(dest), ctypes.c_long(n * D), _vp(msg), S, n, D, None), "blend pair fwd")
    downf, gff = down.float(), gf_w.float()
    hidden = torch.bmm(pair.float(), downf.transpose(1, 2)).to(BF16).float()          # [S, n, r], one bf16 rounding
    gf = torch.bmm(hidden, gff.transpose(1, 2)).contiguous()                           # [S, n, 2D] fp32
    out = torch.empty(S, n, D, dtype=BF16, device=dev)
    rstd = torch.empty(S, n, dtype=F32, device=dev)
    _check(L, L.inkling_capi_loop_blend_out_fwd(_vp(out), _vp(rstd), _vp(dest), ctypes.c_long(n * D), _vp(msg), _vp(gf), _vp(nw),
                                                ctypes.c_long(D), ctypes.c_float(_EPS), S, n, D, None), "blend out fwd")
    return {"dest": dest, "msg": msg, "nw": nw, "downf": downf, "gff": gff, "gf": gf, "rstd": rstd, "out": out}


def _blend_bwd(sv, d_out):
    """The cotangents (d dest [S, n, D], d msg [S, n, D]) of the blend output's cotangent d_out fp32 [S, n, D]."""
    L = lib()
    dest, msg = sv["dest"], sv["msg"]
    dev = dest.device
    S, n, D = msg.shape
    d_out = d_out.contiguous()
    d_dest = torch.empty(S, n, D, dtype=F32, device=dev)
    d_msg = torch.empty(S, n, D, dtype=F32, device=dev)
    d_gf = torch.empty(S, n, 2 * D, dtype=F32, device=dev)
    _check(L, L.inkling_capi_loop_blend_out_bwd(_vp(d_dest), _vp(d_msg), _vp(d_gf), _vp(d_out), 1, _vp(dest), ctypes.c_long(n * D),
                                                _vp(msg), _vp(sv["gf"]), _vp(sv["rstd"]), _vp(sv["nw"]), ctypes.c_long(D), S, n, D,
                                                None), "blend out bwd")
    d_hidden = torch.bmm(d_gf, sv["gff"])                                              # [S, n, r] (straight through the rounding)
    d_pair = torch.bmm(d_hidden, sv["downf"]).contiguous()                             # [S, n, 2D]
    _check(L, L.inkling_capi_loop_blend_pair_bwd(_vp(d_dest), _vp(d_msg), _vp(d_pair), _vp(dest), ctypes.c_long(n * D), _vp(msg),
                                                 S, n, D, None), "blend pair bwd")
    return d_dest, d_msg


def _blend_vjp(dest, msg, down, gf_w, norm_w, d_out):
    return _blend_bwd(_blend_fwd(dest, msg, down, gf_w, norm_w), d_out)


def mlp_in_vjp(L, n_stack, first, has_bank, ins, gouts):
    """The backward of the visit's MLP-input stage (t8_visit / ift_long f_min) for one token chunk. ins: fp32 chunks of
    (stack..., partial, attn_out[, bank0, bank1]); gouts[0] fp32 [n, D] = the cotangent of the MLP input. Returns one
    gradient per input, in order. L: the layer's ift_backward.EngineWeights entry."""
    g = gouts[0]
    st = list(ins[:n_stack])
    p, ao = ins[n_stack], ins[n_stack + 1]
    if first:
        mix = _res_fwd(st + [p, ao], None, L.q_mlp, None if has_bank else L.mlp_norm)
    else:
        mix = _res_fwd(st + [p], ao, L.q_mlp, None if has_bank else L.mlp_norm)
    d_bank = []
    if has_bank:
        b0, b1 = ins[n_stack + 2], ins[n_stack + 3]
        msg = _res_fwd([b0, b1], None, L.q_msg[3:4], None)
        d_dest, d_msg = _blend_vjp(mix["out"], msg["out"], L.ctrl_down[3:4], L.ctrl_gf[3:4], L.mlp_norm, g[None])
        d_src, d_add = _res_bwd(mix, d_dest)
        d_bank, _ = _res_bwd(msg, d_msg)
    else:
        d_src, d_add = _res_bwd(mix, g[None])
    if first:
        return d_src[:n_stack] + [d_src[n_stack], d_src[n_stack + 1]] + d_bank
    return d_src[:n_stack] + [d_src[n_stack], d_add] + d_bank


def _mm16(x16, w16):
    """x [n, K] @ w [M, K]^T, bf16 operands, fp32 accumulation and output."""
    try:
        return torch.mm(x16, w16.T, out_dtype=F32)
    except (TypeError, RuntimeError):
        return x16.float() @ w16.float().T


def _mm16_t(g, w16):
    """g fp32 [n, M] @ w [M, K] with the bf16 weight on the tensor cores: g as its bf16 high + low parts (error ~2^-17)."""
    hi = g.to(BF16)
    lo = (g - hi.float()).to(BF16)
    try:
        return torch.mm(hi, w16, out_dtype=F32) + torch.mm(lo, w16, out_dtype=F32)
    except (TypeError, RuntimeError):
        return g @ w16.float()


def _clip_bwd(gy, x16, scale, tau, threshold, H, d):
    """Backward of y = bf16(bf16(l2_clip(x * scale, threshold)) [* tau]) per head: gy fp32 [n, H, d], x16 bf16 [n, H * d]."""
    L = lib()
    n = x16.shape[0]
    gy, x16 = gy.contiguous(), x16.contiguous()
    dx = torch.empty(n, H * d, dtype=F32, device=x16.device)
    _check(L, L.inkling_capi_qkv_norm_bwd(_vp(dx), _vp(gy), 1, _vp(x16), 0, _vp(scale.contiguous()), _vp(tau),
                                          ctypes.c_float(threshold), ctypes.c_float(_EPS), n, H, d, None), "qkv_norm_bwd")
    return dx


def attn_in_proj_vjp(L, v0, looped, n_stack, has_tau, ins, gouts):
    """The backward of the visit's attention-input + projection stage (t8_visit / ift_long f_ain_proj, no attention adapter)
    for one token chunk. ins: fp32 chunks of (stack..., partial[, bank0, bank1][, tau]); gouts: the cotangents of
    (q, r, gate, sink, k, v[, k_dyn, v_dyn]) (None = no cotangent). Returns one gradient per input (None for tau)."""
    st, p = list(ins[:n_stack]), ins[n_stack]
    tau = ins[-1].contiguous() if has_tau else None
    n, D = p.shape
    dev = p.device
    Q = L.attn_norm.shape[0]
    QS = 3 if (looped or not L.is_local) else 1
    zero = lambda j, w: torch.zeros(n, w, dtype=F32, device=dev) if gouts[j] is None else gouts[j].reshape(n, w)  # noqa: E731
    # ---- forward recompute: the attention inputs (bf16 values) and the raw q / r projections
    mix = msg = blend = None
    if v0:
        rstd = torch.rsqrt((p * p).sum(-1, keepdim=True) / D + _EPS)
        ain = [((p * rstd) * L.attn_norm[j].float()).to(BF16) for j in range(Q)]
    elif not looped:
        mix = _res_fwd(st + [p], None, L.q_attn, L.attn_norm)
        ain = [mix["out"][j] for j in range(Q)]
    else:
        mix = _res_fwd(st + [p], None, L.q_attn, None)
        msg = _res_fwd([ins[n_stack + 1], ins[n_stack + 2]], None, L.q_msg[0:3], None)
        dest = mix["out"].expand(3, -1, -1) if L.is_local else mix["out"]
        nw = L.attn_norm.expand(3, -1) if L.is_local else L.attn_norm
        blend = _blend_fwd(dest, msg["out"], L.ctrl_down[0:3], L.ctrl_gf[0:3], nw)
        ain = [blend["out"][j] for j in range(3)]
    xq = ain[0]
    q_raw = _mm16(xq, L.wq).to(BF16)
    r_raw = _mm16(xq, L.wr).to(BF16)
    # ---- backward: clips (+ tau), projections
    d_q = _clip_bwd(zero(0, 3072).reshape(n, 24, 128), q_raw, L.q_scale, tau, 22.627416997969522, 24, 128)
    d_r = _clip_bwd(zero(1, 384).reshape(n, 24, 16), r_raw, L.r_scale, tau, 8.0, 24, 16)
    d_xq = _mm16_t(d_q, L.wq) + _mm16_t(d_r, L.wr)
    if gouts[2] is not None:
        d_xq = d_xq + _mm16_t(gouts[2].reshape(n, 3072), L.wg)
    if gouts[3] is not None:
        d_xq = d_xq + _mm16_t(gouts[3].reshape(n, 24), L.ws)
    d_xk = _mm16_t(zero(4, 1024), L.wk)
    d_xv = _mm16_t(zero(5, 1024), L.wv)
    if not L.is_local:
        if gouts[6] is not None:
            d_xk = d_xk + _mm16_t(gouts[6].reshape(n, 32), L.k_wproj)
        if gouts[7] is not None:
            d_xv = d_xv + _mm16_t(gouts[7].reshape(n, 32), L.v_wproj)
    d_ain = [d_xq + d_xk + d_xv] if QS == 1 else [d_xq, d_xk, d_xv]
    # ---- backward: the attention inputs
    d_bank = []
    if v0:
        d_p = torch.zeros(n, D, dtype=F32, device=dev)
        for j in range(Q):
            a = d_ain[j] * L.attn_norm[j].float()
            d_p += rstd * a - p * ((rstd * rstd * rstd / D) * (a * p).sum(-1, keepdim=True))
        d_src = [None] * n_stack + [d_p]
    elif not looped:
        d_src, _ = _res_bwd(mix, torch.stack(d_ain))
    else:
        d_dest, d_msg = _blend_bwd(blend, torch.stack(d_ain))
        d_src, _ = _res_bwd(mix, d_dest.sum(0, keepdim=True) if L.is_local else d_dest)
        d_bank, _ = _res_bwd(msg, d_msg)
    return list(d_src[:n_stack]) + [d_src[n_stack]] + list(d_bank) + ([None] if has_tau else [])


def gate_sink_wo_vjp(L, ins, gouts):
    """The backward of the visit's gate / sink + output-projection stage (t8_visit / ift_long f_gw, no attention adapter) for
    one token chunk: y = bf16(bf16((o * sigmoid(gate)) * sigmoid(lse - sink)) @ wo^T). ins: fp32 chunks of (o [n, 24, 128],
    lse [n, 24], gate [n, 3072], sink [n, 24]); gouts[0] fp32 [n, D]. Elementwise parts in fp32 torch (exact formulas), the
    projection on the tensor cores with the cotangent as bf16 high + low parts."""
    o, lse, gate, sink = ins
    n = o.shape[0]
    G = _mm16_t(gouts[0], L.wo).reshape(n, _HQ, _HD)                 # cotangent of the gated output
    a = torch.sigmoid(gate.reshape(n, _HQ, _HD))
    b = torch.sigmoid(lse - sink)                                     # [n, 24]
    d_o = G * a * b[..., None]
    Go = G * o
    d_gate = (Go * b[..., None] * (a * (1.0 - a))).reshape(n, _HQ * _HD)
    d_l = (Go * a).sum(-1) * (b * (1.0 - b))
    return [d_o, d_l, d_gate, -d_l]
