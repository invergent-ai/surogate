"""S3: the engine-op trainer's per-visit recompute (ift_long.visit_torch_long, copied unchanged apart from the lines marked T8) with the S10 attention
LoRA of the engine (targets/inkling2/impl/lora_attention.h, ops::inkling_lora_delta): on the six qkvr slices (q, r, gate, sink from x_q;
k from x_k; v from x_v) after the base projection, and on wo_ud from the gated attention output, each as out = bf16(float(bf16(base)) + delta),
delta = fp32 (x . a) . b with a = bf16(bf16(A) x alpha / r) (the store's fold; exact for alpha / r = 0.5) and b = bf16(B). Gradients for A and B
straight through the bf16 roundings, as everywhere in this backward. Without attention adapters it is visit_torch_long bit for bit (same ops).
"""
from __future__ import annotations

import torch

import ift_backward as B
import ift_ref as RA
import ift_long as LG
from ift_long import MoEGM, RelAttentionR, tc  # noqa: F401

D = B.D
EPS = B.EPS
BF16, F32 = torch.bfloat16, torch.float32
QKVR_SLICES = ("q", "r", "g", "s", "k", "v")          # kQkvrLoraSlices order: wq_du, wr_du, gate_attn, attn_sink_proj, wk_dv, wv_dv
ATTN_SLICES = QKVR_SLICES + ("o",)                     # + wo_ud
# rows and input width of each slice (lora_attention.h; wo_ud: [2304, 3072])
SLICE_DIMS = {"q": (3072, 2304), "r": (384, 2304), "g": (3072, 2304), "s": (24, 2304), "k": (1024, 2304), "v": (1024, 2304), "o": (2304, 3072)}


def qkvr_raw(L, xq, xk, xv, lw, att_scale):
    """The six qkvr outputs before the clips (bf16 values): base projection, then the S10 delta of each slice from its own input stream."""
    def proj(x, w, s):
        y = RA.inkling_attn_linear(x, w)
        return y if lw is None else lora_add(y, x, lw[s][0], lw[s][1], att_scale)

    return proj(xq, L.wq, "q"), proj(xq, L.wr, "r"), proj(xq, L.wg, "g"), proj(xq, L.ws, "s"), proj(xk, L.wk, "k"), proj(xv, L.wv, "v")


def wo_raw(L, gated, lo, att_scale):
    """wo_ud's output (bf16 values) from the gated attention output, + the S10 delta of slice o when lo = (A_o, B_o)."""
    y = RA.rb(RA.mm(gated, L.wo))
    return y if lo is None else lora_add(y, gated, lo[0], lo[1], att_scale)


def lora_dict(keys, leaves, scale, attn_scale):
    """The per-visit `lora` argument of visit_torch_t8 from the layer's keys and tensors (the ONE place the attention keys map to slices)."""
    lora = dict(zip(keys, leaves)) | {"scale": scale}
    if attn_scale is not None:
        lora["attn"] = {s: (lora[f"{s}A"], lora[f"{s}B"]) for s in ATTN_SLICES}
        lora["attn_scale"] = attn_scale
    return lora


def lora_add(y, x, A, Bm, scale):
    """bf16(y + delta), delta = (x . (A x scale)^T) . B^T in fp32 (A, B: the work values = bf16(master); scale a power of two, so A x scale is the
    store's bf16 fold exactly)."""
    low = RA.mm(x.float(), A * scale)                  # RA.mm(x, w) = x . w^T: [t, R]
    return RA.rb(y.float() + RA.mm(low, Bm))           # [t, n]


def visit_torch_t8(W: B.EngineWeights, vi, stack, partial, bank, ids, lora, chunk=None, act_vals=None,
                     act_sinks=None):
    """Same function as B.visit_torch (new partial [T, D] fp32), built from token-chunked stages. act_vals: host
    values of (stack..., partial, bank...) when the passed tensors are zero placeholders (graph nodes only)."""
    L = W.layers[vi.layer]
    T = partial.shape[0]
    n = len(stack)
    has_bank = bank is not None
    Q = L.attn_norm.shape[0]
    pos = torch.arange(T, device=partial.device)
    tau = None if L.is_local else RA.inkling_log_tau(pos)
    first, v0 = vi.first, vi.v == 0
    act = list(stack) + [partial] + (list(bank) if has_bank else [])
    lt = [lora[k] for k in B.LORA_KEYS] if lora is not None else []
    scale = lora["scale"] if lora is not None else None
    # T8: S10 attention LoRA (lora["attn"] = {slice: (A [R, k], B [n, R])} + lora["attn_scale"]); None = no attention adapter
    att = lora.get("attn") if lora is not None else None
    att_scale = lora.get("attn_scale") if att is not None else None
    ap = [t for s in QKVR_SLICES for t in att[s]] if att is not None else []
    op = list(att["o"]) if att is not None else []

    def messages_of(bk):
        return RA.inkling_attn_res([bk[0].float()], bk[1].float(), None, L.q_msg, None)

    # 1. attention inputs (per token) -> [T, Q, D]
    def f_ain(*a):
        st = [x.float() for x in a[:n]]
        p = a[n].float()
        if v0:
            ain = torch.stack([RA.rb(RA.rms_norm(p, L.attn_norm[j], EPS)) for j in range(Q)])
        elif not vi.looped:
            ain = RA.inkling_attn_res(st, p, None, L.q_attn, L.attn_norm, EPS)
        else:
            dests = RA.inkling_attn_res(st, p, None, L.q_attn, None)
            msgs = messages_of(a[n + 1:n + 3])
            norm_w = L.attn_norm
            if L.is_local:
                dests = dests.expand(3, -1, -1)
                norm_w = norm_w.expand(3, -1)
            ain = RA.inkling_loop_blend(dests, msgs[:3], L.ctrl_down[0:3], L.ctrl_gf[0:3], norm_w, EPS)
        return ain.permute(1, 0, 2)

    # 1 + 2 fused in ONE token-chunked stage: the [T, Q, D] attention inputs (and their gradient) never exist at full T
    QS = 3 if (vi.looped or not L.is_local) else 1

    def f_proj(ain, *rest):
        tau_ = rest[:len(rest) - len(ap)]
        lw = dict(zip(QKVR_SLICES, zip(rest[len(tau_)::2], rest[len(tau_) + 1::2]))) if ap else None
        xq = ain[:, 0]
        xk = ain[:, 1] if QS == 3 else xq
        xv = ain[:, 2] if QS == 3 else xq
        t = xq.shape[0]
        q, r, gate, sink, k, v = qkvr_raw(L, xq, xk, xv, lw, att_scale)
        q, r = q.reshape(t, 24, 128), r.reshape(t, 24, 16)
        qc = RA.l2_clip(q, L.q_scale, 22.627416997969522, EPS)
        rc = RA.l2_clip(r, L.r_scale, 8.0, EPS)
        if tau_:
            tt = tau_[0]
            qc = RA.rb(qc) * tt[:, None, None]
            rc = RA.rb(rc) * tt[:, None, None]
        outs = [RA.rb(qc), RA.rb(rc), gate, sink, k, v]
        if not L.is_local:
            outs += [RA.rb(RA.mm(xk, L.k_wproj)), RA.rb(RA.mm(xv, L.v_wproj))]
        return tuple(outs)

    def f_ain_proj(*a):
        ain = f_ain(*a[:len(act)])
        return f_proj(ain, *a[len(act):])

    def f_ain_proj_vjp(ins, gouts, params=None):  # the stage's hand-written backward on the kernel library (STAGE_CUDA)
        import cuda_bwd as CB

        return CB.attn_in_proj_vjp(L, v0, vi.looped, n, tau is not None, ins, gouts, params, att_scale)

    f_ain_proj.cuda_vjp, f_ain_proj.cuda_vjp_params = f_ain_proj_vjp, True

    with B._tick("fwd.ain_proj"):
        tok = act + ([tau] if tau is not None else [])
        tvals = (list(act_vals) + [None] * (len(tok) - len(act))) if act_vals is not None else None
        tsinks = (list(act_sinks) + [None] * (len(tok) - len(act))) if act_sinks is not None else None
        pr = tc(f_ain_proj, tok, [0] * len(tok), [0] * (8 if not L.is_local else 6), params=ap,
                store_bf16=[True] * len(act) + [False] * (len(tok) - len(act)), chunk=chunk, vals=tvals,
                sinks=tsinks)
    q_out, r_out, gate, sink, k_raw, v_raw = pr[:6]
    # 3. K / V causal convs + clip / norm (full T)
    with B._tick("fwd.kv"):
        if L.is_local:
            kc, _ = RA.inkling_short_conv(k_raw, L.k_conv, None)
            vc, _ = RA.inkling_short_conv(v_raw, L.v_conv, None)
        else:
            kc, _ = RA.inkling_dynamic_conv(k_raw, L.k_static, pr[6], None)
            vc, _ = RA.inkling_dynamic_conv(v_raw, L.v_static, pr[7], None)
        kh = kc.reshape(T, -1, 128)
        vh = vc.reshape(T, -1, 128)
        K = RA.kv_store(RA.l2_clip(kh, None, 22.627416997969522, EPS), True)
        V = RA.kv_store(vh * RA.t_rsqrt(RA.rsum(vh * vh, -1, keepdim=True) / 128 + EPS), True)
        del kc, vc, kh, vh, k_raw, v_raw, pr
    # 4. attention (flash, bias from r per block)
    with B._tick("fwd.attention"):
        win = 1024 if L.is_local else 0
        out, lse = LG.rel_attention(q_out, K.float(), V.float(), r_out, L.rel_proj, win, LG._q_block(T, win))
        del K, V, q_out, r_out
    # 5. gate / sink + wo_ud (per token)

    def f_gw(o, l, gt, sk, *lo):
        return wo_raw(L, RA.inkling_gate_sink(o, l, gt, sk), lo if lo else None, att_scale)

    def f_gw_vjp(ins, gouts, params=None):  # the stage's hand-written backward (STAGE_CUDA)
        import cuda_bwd as CB

        return CB.gate_sink_wo_vjp(L, ins, gouts, params, att_scale)

    f_gw.cuda_vjp, f_gw.cuda_vjp_params = f_gw_vjp, True

    with B._tick("fwd.wo"):
        attn_partial = tc(f_gw, [out, lse, gate, sink], [0, 0, 0, 0], [0], params=op, store_bf16=[False, False, True, True],
                          chunk=chunk)
    # 6. attention-output conv (full T)
    del out, lse, gate, sink
    with B._tick("fwd.wo_conv"):
        attn_out, _ = RA.inkling_short_conv(attn_partial, L.attn_sconv, None)
    del attn_partial
    # 7. MLP input + MoE (per token)

    def f_min(*a):
        st = [x.float() for x in a[:n]]
        p = a[n].float()
        ao = a[n + 1].float()
        msgs = messages_of(a[n + 2:n + 4]) if has_bank else None
        mlp_norm = None if msgs is not None else L.mlp_norm
        if first:
            mi = RA.inkling_attn_res(st + [p], ao, None, L.q_mlp, mlp_norm, EPS)
        else:
            mi = RA.inkling_attn_res(st, p, ao, L.q_mlp, mlp_norm, EPS)
        if msgs is not None:
            mi = RA.inkling_loop_blend(mi, msgs[3:], L.ctrl_down[3:4], L.ctrl_gf[3:4], L.mlp_norm, EPS)
        return mi[0]

    def f_min_vjp(ins, g):  # the stage's hand-written backward on the kernel library (STAGE_CUDA)
        import cuda_bwd as CB

        return CB.mlp_in_vjp(L, n, first, has_bank, ins, g)

    f_min.cuda_vjp = f_min_vjp

    def f_moe(*a):
        rid = a[-5] if scale is not None else a[-1]
        lo = dict(zip(B.LORA_KEYS, a[-4:])) | {"scale": scale} if scale is not None else None
        return B.moe_vec(f_min(*a[:len(a) - (5 if scale is not None else 1)]), L.experts, rid, lo)

    with B._tick("fwd.moe"):
        tok = list(stack) + [partial, attn_out] + (list(bank) if has_bank else [])
        mvals = msinks = None
        if act_vals is not None:
            mvals = list(act_vals[:n + 1]) + [None] + list(act_vals[n + 1:])
        if act_sinks is not None:
            msinks = list(act_sinks[:n + 1]) + [None] + list(act_sinks[n + 1:])
        sb = [True] * (n + 1) + [True] + ([True, True] if has_bank else [])
        if LG.MOE_GM:
            mlp_in = tc(f_min, tok, [0] * len(tok), [0], store_bf16=sb, chunk=chunk, vals=mvals, sinks=msinks)
            moe_out = MoEGM.apply(L.experts, ids, scale, mlp_in, *lt)
            del mlp_in
        else:
            moe_out = tc(f_moe, tok + [ids], [0] * (len(tok) + 1), [0], params=lt, store_bf16=sb + [False],
                         chunk=chunk, vals=None if mvals is None else mvals + [None],
                         sinks=None if msinks is None else msinks + [None])
    # 8. MLP-output conv (full T)
    with B._tick("fwd.out_conv"):
        res_ = None
        if not first:  # the partial's value (+ its graph node: a zero placeholder when the value is on the host;
            # with sinks the caller adds the output gradient to the partial's sink itself: d residual = d out, STE)
            if act_vals is None:
                res_ = partial.float()
            elif act_sinks is not None:
                res_ = act_vals[n].to(partial.device).float()
            else:
                res_ = partial + act_vals[n].to(partial.device).float()
        new_partial, _ = RA.inkling_short_conv(moe_out, L.mlp_sconv, None, addend=attn_out, residual=res_)
    return new_partial


