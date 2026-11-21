"""ENGINE-INKLING lane T: the LEVEL-0 backward for LONG rows on one card (lanes/T/NOTES.md, long-row path).

Same gradient as ift_backward (S2: lane L's W4A16 / STE math per visit, evaluated at the ENGINE's visit inputs,
routing replayed from the engine), restructured so memory no longer grows like (visits x tokens x graph):
- the engine forward keeps every visit output in PINNED HOST memory (bf16, exact), not on the card; the autograd graph
  over the 48 visits carries zero-byte placeholders (the values live on the host and in the engine's head output);
- a visit's recompute is staged: per-token stages (AttnRes / blend inputs, projections + q / r clips, gate / sink +
  wo_ud, MLP input + MoE) run as TokenChunked functions: forward in token chunks without a graph, backward = recompute
  + vjp chunk by chunk (IFT_CHUNK tokens, default 1,024); the sequence-coupled pieces (K / V causal convs, the two
  output convs) run with full-T autograd (small graphs); attention is flash-style (RelAttentionR: the relative bias is
  formed per query block from r and rel_proj, never as a dense [T, 24, 1,024] tensor);
- the final mix is TokenChunked too.
Validation: the same gradient as ift_backward on short rows (tests/long_check.py), lane L's rig via D1 as before.
"""

from __future__ import annotations

import os

import torch

import ift_backward as B
import ift_ref as RA
from ift_engine import InklingEngine, Segment
import ift_forward as F

D = B.D
BF16, F32 = torch.bfloat16, torch.float32
CHUNK = int(os.environ.get("IFT_CHUNK", "1024"))
# TokenChunked token inputs of rows at least this long are saved in pinned host memory (bf16) instead of on the card
OFFLOAD_MIN = int(os.environ.get("IFT_OFFLOAD_MIN", "8192"))
# the 48-visit backward as one node with a manual reverse loop (LongModel); 0 = per-visit autograd nodes (EngineVisitH)
MANUAL = os.environ.get("IFT_MANUAL", "1") == "1"
EPS = B.EPS


# ---------------------------------------------------------------------------------------------------------------
# generic token-chunked recompute
# ---------------------------------------------------------------------------------------------------------------
class TokenChunked(torch.autograd.Function):
    """outs = fn(*tok_inputs, *params) for a function that is independent per token. tdims_in / tdims_out: the token
    dimension of each token input / output. Forward: chunks without a graph (inputs saved, bf16 where `store_bf16`);
    backward: per chunk, recompute with a graph and take the vjp; token-input gradients land in their chunk's rows,
    parameter gradients are summed over chunks."""

    @staticmethod
    def forward(ctx, fn, tdims_in, tdims_out, n_tok, chunk, store_bf16, vals, sinks, *inputs):
        # with sinks, the LAST input is a zero-size dummy leaf (tc() makes it): it ties this node into the graph and
        # the caller lists it among autograd.grad's inputs, so the backward runs and fills the sinks
        nd = 1 if sinks is not None else 0
        ctx.nd = nd
        inputs = inputs[:len(inputs) - nd]
        T = inputs[0].shape[tdims_in[0]]
        parts = None
        dev = inputs[0].device
        vals = vals or (None,) * n_tok

        def src(i, c0, n):  # a chunk of token input i: from its host value when given, else from the input
            if vals[i] is not None:
                return vals[i].narrow(tdims_in[i], c0, n).to(dev, non_blocking=True).float()
            return inputs[i].narrow(tdims_in[i], c0, n)

        with torch.no_grad():
            for c0 in range(0, T, chunk):
                n = min(chunk, T - c0)
                args = [src(i, c0, n) for i in range(n_tok)] + list(inputs[n_tok:])
                o = fn(*args)
                o = o if isinstance(o, tuple) else (o,)
                if parts is None:
                    parts = [[] for _ in o]
                for j, t in enumerate(o):
                    parts[j].append(t)
        outs = tuple(torch.cat(p, tdims_out[j]) if len(p) > 1 else p[0] for j, p in enumerate(parts))
        ctx.fn, ctx.tdi, ctx.tdo, ctx.n_tok, ctx.chunk = fn, tdims_in, tdims_out, n_tok, chunk
        ctx.in_dtypes = [x.dtype for x in inputs]
        ctx.vals = vals
        ctx.sinks = sinks or (None,) * n_tok
        keep = [x if i >= n_tok else (None if vals[i] is not None else (x.detach().to(BF16) if store_bf16[i] else x))
                for i, x in enumerate(inputs)]
        ctx.offload = T >= OFFLOAD_MIN
        ctx.dev = dev
        ctx.host = [None] * n_tok
        if ctx.offload:  # token inputs to pinned host (non-blocking), params stay
            for i, x in enumerate(keep[:n_tok]):
                if x is None:
                    continue
                h = torch.empty(x.shape, dtype=x.dtype, device="cpu", pin_memory=True)
                h.copy_(x, non_blocking=True)  # stream-ordered: x's memory is reused only after the copy
                ctx.host[i] = h
                keep[i] = None
        ctx.n_dev = [i for i in range(n_tok) if keep[i] is not None]
        ctx.save_for_backward(*[keep[i] for i in ctx.n_dev], *keep[n_tok:])
        return outs if len(outs) > 1 else outs[0]

    @staticmethod
    def backward(ctx, *gouts):
        n_tok, chunk, tdi, tdo = ctx.n_tok, ctx.chunk, ctx.tdi, ctx.tdo
        st = ctx.saved_tensors
        dev_saved = dict(zip(ctx.n_dev, st[:len(ctx.n_dev)]))
        params_saved = st[len(ctx.n_dev):]
        dev = ctx.dev
        toks = []
        for i in range(n_tok):  # value sources: device copy, pinned host copy, or the caller's host values
            if i in dev_saved:
                x = dev_saved[i]
            elif ctx.host[i] is not None:
                x = ctx.host[i]
            else:
                x = ctx.vals[i]
            toks.append(x)
        ctx.host = None
        need = ctx.needs_input_grad[8:]
        T = toks[0].shape[tdi[0]]

        def chunk_of(i, c0, n):
            x = toks[i].narrow(tdi[i], c0, n)
            if x.device != dev:
                x = x.to(dev, non_blocking=True)
            return x.float() if (x.dtype == BF16 and ctx.in_dtypes[i] == F32) else x
        params = [p.detach().requires_grad_(bool(need[n_tok + i]) and p.is_floating_point())
                  for i, p in enumerate(params_saved)]
        sinks = ctx.sinks
        gin = [torch.zeros(toks[i].shape, dtype=F32, device=dev)
               if (need[i] and sinks[i] is None and toks[i].is_floating_point()) else None for i in range(n_tok)]
        pgr = [None] * len(params)
        for c0 in range(0, T, chunk):
            n = min(chunk, T - c0)
            with torch.enable_grad(), torch.device(dev):
                ins = [chunk_of(i, c0, n).detach().requires_grad_(gin[i] is not None or sinks[i] is not None)
                       for i in range(n_tok)]
                o = ctx.fn(*ins, *params)
                o = o if isinstance(o, tuple) else (o,)
                outs, grads_o = [], []
                for j, oj in enumerate(o):
                    if gouts[j] is not None and oj.requires_grad:
                        outs.append(oj)
                        grads_o.append(gouts[j].narrow(tdo[j], c0, n))
                want = [t for t in ins + params if t.requires_grad]
                if not outs or not want:
                    continue
                gr = torch.autograd.grad(outs, want, grads_o, allow_unused=True)
            it = iter(gr)
            for i, t in enumerate(ins):
                if t.requires_grad:
                    g = next(it)
                    if g is None:
                        continue
                    if sinks[i] is not None:  # straight into the caller's accumulator (card or host)
                        sinks[i].narrow(tdi[i], c0, n).add_(g.to(sinks[i].device))
                    else:
                        gin[i].narrow(tdi[i], c0, n).copy_(g)
            for i, p in enumerate(params):
                if p.requires_grad:
                    g = next(it)
                    if g is not None:
                        pgr[i] = g if pgr[i] is None else pgr[i] + g
        res = [None if g is None else g.to(ctx.in_dtypes[i]) for i, g in enumerate(gin)]
        res += [None if g is None else g.to(ctx.in_dtypes[n_tok + i]) for i, g in enumerate(pgr)]
        return (None, None, None, None, None, None, None, None, *res, *([None] * ctx.nd))


def tc(fn, tok, tdims_in, tdims_out, params=(), store_bf16=None, chunk=None, vals=None, sinks=None):
    """vals: per token input, None or a (host) tensor holding its VALUES; the input itself is then only a graph node
    (a zero placeholder). sinks: per token input, None or an fp32 accumulator (card or host) that receives the input's
    gradient chunk by chunk (the input then gets None from autograd)."""
    store = store_bf16 if store_bf16 is not None else [False] * len(tok)
    extra = ()
    if sinks is not None:
        if any(x is not None for x in sinks):
            d = torch.zeros(0, dtype=F32, device=tok[0].device, requires_grad=True)
            SINK_DUMMIES.append(d)
            extra = (d,)
        else:
            sinks = None
    return TokenChunked.apply(fn, tuple(tdims_in), tuple(tdims_out), len(tok), chunk or CHUNK, tuple(store),
                              tuple(vals) if vals is not None else None, tuple(sinks) if sinks is not None else None,
                              *tok, *params, *extra)


# the dummy leaves of the sink-carrying TokenChunked nodes built since the last clear (see tc)
SINK_DUMMIES = []


# ---------------------------------------------------------------------------------------------------------------
# flash attention with the relative bias formed from r per query block
# ---------------------------------------------------------------------------------------------------------------
def _rel_block(r_out, rel_proj, b0, b1):
    """rel [nq, 24, 1024] = bf16(r_out[b0:b1] . rel_proj) (lane L's prologue: rb(einsum('thj,jd->thd')))."""
    return RA.rb(torch.einsum("thj,jd->thd", r_out[b0:b1], rel_proj.float()))


class RelAttentionR(torch.autograd.Function):
    """B.RelAttention with rel = bf16(r_out . rel_proj) formed per query block (STE through the bf16 rounding):
    gradients for q, k, v and r_out; rel_proj is frozen."""

    @staticmethod
    def forward(ctx, q, k, v, r_out, rel_proj, window, q_block=128):
        T, hq, dh = q.shape
        scale = 128 ** -0.5
        out = torch.empty(T, hq, dh, dtype=F32, device=q.device)
        lse = torch.empty(T, hq, dtype=F32, device=q.device)
        vv = v.permute(1, 0, 2)
        hk = k.shape[1]
        g = hq // hk
        for b0 in range(0, T, q_block):
            b1 = min(T, b0 + q_block)
            k0 = max(0, b0 - window + 1) if window > 0 else 0
            rel = _rel_block(r_out, rel_proj, b0, b1)
            sc, _, _ = B._rel_scores(q, k, _Shift(rel, b0), b0, b1, k0, window, scale)
            m = sc.amax(-1, keepdim=True)
            p = RA.t_exp(sc - m)
            l = p.sum(-1, keepdim=True)
            o = torch.einsum("hgbk,hkd->hgbd", p, vv[:, k0:b1]) / l
            out[b0:b1] = o.permute(2, 0, 1, 3).reshape(b1 - b0, hq, dh)
            lse[b0:b1] = (m + RA.t_log(l))[..., 0].permute(2, 0, 1).reshape(b1 - b0, hq)
        ctx.save_for_backward(q.to(BF16), k.to(BF16), v.to(BF16), r_out.to(BF16), rel_proj, out, lse)
        ctx.window, ctx.q_block, ctx.scale = window, q_block, scale
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, r_out, rel_proj, out, lse = ctx.saved_tensors
        q, k, v, r_out = q.float(), k.float(), v.float(), r_out.float()
        window, qb, scale = ctx.window, ctx.q_block, ctx.scale
        T, hq, dh = q.shape
        hk = k.shape[1]
        g = hq // hk
        if dout is None:
            dout = torch.zeros_like(out)
        if dlse is None:
            dlse = torch.zeros_like(lse)
        dq = torch.zeros(q.shape, dtype=F32, device=q.device)
        dk = torch.zeros(k.shape, dtype=F32, device=k.device)
        dv = torch.zeros(v.shape, dtype=F32, device=v.device)
        dr = torch.zeros(r_out.shape, dtype=F32, device=r_out.device)
        vv = v.permute(1, 0, 2)
        kk = k.permute(1, 0, 2)
        projf = rel_proj.float()
        for b0 in range(0, T, qb):
            b1 = min(T, b0 + qb)
            nq = b1 - b0
            k0 = max(0, b0 - window + 1) if window > 0 else 0
            rel = _rel_block(r_out, rel_proj, b0, b1)
            sc, gi, valid = B._rel_scores(q, k, _Shift(rel, b0), b0, b1, k0, window, scale)
            lb = lse[b0:b1].reshape(nq, hk, g).permute(1, 2, 0)[..., None]
            p = RA.t_exp(sc - lb)
            do = dout[b0:b1].reshape(nq, hk, g, dh).permute(1, 2, 0, 3)
            ob = out[b0:b1].reshape(nq, hk, g, dh).permute(1, 2, 0, 3)
            dlb = dlse[b0:b1].reshape(nq, hk, g).permute(1, 2, 0)[..., None]
            dp = torch.einsum("hgbd,hkd->hgbk", do, vv[:, k0:b1])
            dterm = (do * ob).sum(-1, keepdim=True)
            ds = p * (dp - dterm + dlb)
            qq = q[b0:b1].reshape(nq, hk, g, dh).permute(1, 2, 0, 3)
            dq[b0:b1] += (torch.einsum("hgbk,hkd->hgbd", ds, kk[:, k0:b1]) * scale).permute(2, 0, 1, 3).reshape(nq, hq, dh)
            dk[k0:b1] += (torch.einsum("hgbk,hgbd->hkd", ds, qq) * scale).permute(1, 0, 2)
            dv[k0:b1] += torch.einsum("hgbk,hgbd->hkd", p, do).permute(1, 0, 2)
            dsr = ds.reshape(hq, nq, -1) * valid[None]
            drel = torch.zeros(hq, nq, 1024, dtype=F32, device=q.device)
            drel.scatter_add_(2, gi[None].expand(hq, -1, -1), dsr)
            dr[b0:b1] += torch.einsum("hbd,jd->bhj", drel, projf)
        return dq, dk, dv, dr, None, None, None


class _Shift:
    """rel rows of one query block addressed with absolute row indices [b0, b1) (B._rel_scores slices rel[b0:b1])."""

    def __init__(self, rel, b0):
        self.rel, self.b0 = rel, b0

    def __getitem__(self, sl):
        return self.rel[sl.start - self.b0:sl.stop - self.b0]


# ---------------------------------------------------------------------------------------------------------------
# a visit's recompute, staged
# ---------------------------------------------------------------------------------------------------------------
def visit_torch_long(W: B.EngineWeights, vi, stack, partial, bank, ids, lora, chunk=None, act_vals=None,
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

    def f_proj(ain, *tau_):
        xq = ain[:, 0]
        xk = ain[:, 1] if QS == 3 else xq
        xv = ain[:, 2] if QS == 3 else xq
        t = xq.shape[0]
        q = RA.inkling_attn_linear(xq, L.wq).reshape(t, 24, 128)
        r = RA.inkling_attn_linear(xq, L.wr).reshape(t, 24, 16)
        gate = RA.inkling_attn_linear(xq, L.wg)
        sink = RA.inkling_attn_linear(xq, L.ws)
        k = RA.inkling_attn_linear(xk, L.wk)
        v = RA.inkling_attn_linear(xv, L.wv)
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

    with B._tick("fwd.ain_proj"):
        tok = act + ([tau] if tau is not None else [])
        tvals = (list(act_vals) + [None] * (len(tok) - len(act))) if act_vals is not None else None
        tsinks = (list(act_sinks) + [None] * (len(tok) - len(act))) if act_sinks is not None else None
        pr = tc(f_ain_proj, tok, [0] * len(tok), [0] * (8 if not L.is_local else 6),
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
        out, lse = RelAttentionR.apply(q_out, K.float(), V.float(), r_out, L.rel_proj, 1024 if L.is_local else 0)
        del K, V, q_out, r_out
    # 5. gate / sink + wo_ud (per token)

    def f_gw(o, l, gt, sk):
        return RA.rb(RA.mm(RA.inkling_gate_sink(o, l, gt, sk), L.wo))

    with B._tick("fwd.wo"):
        attn_partial = tc(f_gw, [out, lse, gate, sink], [0, 0, 0, 0], [0], store_bf16=[False, False, True, True],
                          chunk=chunk)
    # 6. attention-output conv (full T)
    del out, lse, gate, sink
    with B._tick("fwd.wo_conv"):
        attn_out, _ = RA.inkling_short_conv(attn_partial, L.attn_sconv, None)
    del attn_partial
    # 7. MLP input + MoE (per token)

    def f_moe(*a):
        st = [x.float() for x in a[:n]]
        p = a[n].float()
        ao = a[n + 1].float()
        j = n + 2
        msgs = None
        if has_bank:
            msgs = messages_of(a[j:j + 2])
            j += 2
        rid = a[j]
        lo = dict(zip(B.LORA_KEYS, a[j + 1:j + 5])) | {"scale": scale} if scale is not None else None
        mlp_norm = None if msgs is not None else L.mlp_norm
        if first:
            mi = RA.inkling_attn_res(st + [p], ao, None, L.q_mlp, mlp_norm, EPS)
        else:
            mi = RA.inkling_attn_res(st, p, ao, L.q_mlp, mlp_norm, EPS)
        if msgs is not None:
            mi = RA.inkling_loop_blend(mi, msgs[3:], L.ctrl_down[3:4], L.ctrl_gf[3:4], L.mlp_norm, EPS)
        return B.moe_vec(mi[0], L.experts, rid, lo)

    with B._tick("fwd.moe"):
        tok = list(stack) + [partial, attn_out] + (list(bank) if has_bank else []) + [ids]
        mvals = msinks = None
        if act_vals is not None:
            mvals = list(act_vals[:n + 1]) + [None] + list(act_vals[n + 1:]) + [None]
        if act_sinks is not None:
            msinks = list(act_sinks[:n + 1]) + [None] + list(act_sinks[n + 1:]) + [None]
        moe_out = tc(f_moe, tok, [0] * len(tok), [0], params=lt,
                     store_bf16=[True] * (n + 1) + [True] + ([True, True] if has_bank else []) + [False], chunk=chunk,
                     vals=mvals, sinks=msinks)
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


# ---------------------------------------------------------------------------------------------------------------
# the engine forward with host saves, and the autograd graph over placeholders
# ---------------------------------------------------------------------------------------------------------------
_PINNED = {}


def _pinned(name, shape, dtype):
    n = 1
    for s in shape:
        n *= s
    buf = _PINNED.get(name)
    if buf is None or buf.numel() < n or buf.dtype != dtype:
        buf = torch.empty(max(n, 1), dtype=dtype, device="cpu", pin_memory=True)
        _PINNED[name] = buf
    return buf[:n].view(shape)


@torch.no_grad()
def forward_host(eng: InklingEngine, ids: torch.Tensor, adapters_per_layer=None):
    """F.forward with every visit output (and x0) copied to pinned host memory (bf16) instead of kept on the card."""
    T = ids.numel()
    x = torch.zeros(T, eng.c["residual"], dtype=BF16, device=eng.dev)
    ws = eng.workspace(T, adapters_per_layer is not None)
    seg = Segment(T, eng.dev)
    eng.embed(ids, x, ws)
    host = {"x0": _pinned("x0", (T, D), BF16)}
    host["x0"].copy_(F.slot(x, 0), non_blocking=True)
    routing = torch.empty(eng.c["layers"], T, 8, dtype=torch.int32, device=eng.dev)
    outs = []
    for v in range(eng.c["layers"]):
        seg.conv.zero_()
        ad = adapters_per_layer[eng.visits[v].layer] if adapters_per_layer is not None else None
        eng.run_visit(v, x, T, seg, ws, routing[v], None, ad)
        h = _pinned(f"v{v}", (T, D), BF16)
        h.copy_(F.slot(x, eng.visits[v].next_partial), non_blocking=True)
        outs.append(h)
    hidden = torch.empty(T, D, dtype=BF16, device=eng.dev)
    eng.final_hidden(x, T, hidden, ws)
    lp = F.logprobs(eng, hidden, ids)
    torch.cuda.current_stream(eng.device).synchronize()
    del x, seg, ws
    eng._ws = None  # T x ~90 KB of engine workspace: not needed by the backward
    return {"x0": host["x0"], "outs": outs, "routing": routing, "hidden": hidden, "lp": lp}


class EngineVisitH(torch.autograd.Function):
    """One visit in the graph: value = a zero-byte placeholder (the engine's output lives on the host); backward =
    visit_torch_long recomputed from the host copies of the visit's inputs."""

    @staticmethod
    def forward(ctx, W, vi, ids, n_stack, has_bank, lora_scale, host_ins, T, *tensors):
        ctx.W, ctx.vi, ctx.ids, ctx.n_stack, ctx.has_bank, ctx.lora_scale = W, vi, ids, n_stack, has_bank, lora_scale
        ctx.host_ins = host_ins
        n_act = len(host_ins)
        ctx.save_for_backward(*tensors[n_act:])  # the LoRA tensors
        dev = ids.device
        return torch.zeros(1, dtype=F32, device=dev).expand(T, D)

    @staticmethod
    def backward(ctx, g):
        dev = g.device
        need = ctx.needs_input_grad[8:]
        n_act = len(ctx.host_ins)
        with torch.enable_grad(), torch.device(dev):
            T = ctx.host_ins[0].shape[0]
            ins = [torch.zeros(1, dtype=F32, device=dev).expand(T, D).requires_grad_(bool(need[i]))
                   for i in range(n_act)]
            lt = [t.detach().requires_grad_(bool(need[n_act + i])) for i, t in enumerate(ctx.saved_tensors)]
            n = ctx.n_stack
            stack, partial = ins[:n], ins[n]
            bank = (ins[n + 1], ins[n + 2]) if ctx.has_bank else None
            lora = dict(zip(B.LORA_KEYS, lt)) | {"scale": ctx.lora_scale} if ctx.lora_scale is not None else None
            out = visit_torch_long(ctx.W, ctx.vi, stack, partial, bank, ctx.ids, lora, act_vals=ctx.host_ins)
            want = [t for t in ins + lt if t.requires_grad]
            with B._tick("bwd.autograd"):
                grads = torch.autograd.grad(out, want, g.contiguous(), allow_unused=True)
        it = iter(grads)
        res = [next(it) if t.requires_grad else None for t in ins + lt]
        return (None,) * 8 + tuple(res)


class FinalMixH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, engine_hidden, host_ins, *srcs):
        ctx.W, ctx.host_ins = W, host_ins
        return engine_hidden.float()

    @staticmethod
    def backward(ctx, g):
        W = ctx.W
        dev = g.device
        need = ctx.needs_input_grad[3:]

        def f_mix(a, b, c, d, e):
            return RA.inkling_attn_res([a.float(), b.float(), c.float(), d.float()], e.float(), None, W.q_out,
                                       W.final_norm.reshape(1, D), EPS)[0]

        with torch.enable_grad(), torch.device(dev):
            T = ctx.host_ins[0].shape[0]
            ins = [torch.zeros(1, dtype=F32, device=dev).expand(T, D).requires_grad_(bool(need[i])) for i in range(5)]
            hid = tc(f_mix, ins, [0] * 5, [0], store_bf16=[True] * 5, vals=ctx.host_ins)
            want = [t for t in ins if t.requires_grad]
            grads = torch.autograd.grad(hid, want, g.contiguous(), allow_unused=True)
        it = iter(grads)
        return (None, None, None) + tuple(next(it) if t.requires_grad else None for t in ins)


class TrainForwardLong:
    """build(ids, lora, banks) -> per-target log-probs [T - 1] (the engine's values) whose backward is the long-row
    LEVEL-0 recompute."""

    def __init__(self, eng: InklingEngine, W: B.EngineWeights):
        self.eng, self.W = eng, W

    def build(self, ids, lora=None, banks=None):
        if MANUAL:
            return build_manual(self, ids, lora, banks)
        res = forward_host(self.eng, ids, None if banks is None else [L["adapters"] for L in banks.layers])
        return self.graph(ids, res, lora), res

    def graph(self, ids, vals, lora=None):
        eng, W = self.eng, self.W
        T = ids.numel()
        dev = eng.dev
        host = {0: vals["x0"]}
        node = {0: torch.zeros(1, dtype=F32, device=dev).expand(T, D)}  # x0: no gradient (frozen embedding)
        for v, vi in enumerate(eng.visits):
            slots = list(vi.stack) + [vi.partial] + (list(vi.bank) if vi.looped else [])
            lt = [lora[vi.layer][k] for k in B.LORA_KEYS] if lora is not None else []
            scale = lora[vi.layer]["scale"] if lora is not None else None
            out = EngineVisitH.apply(W, vi, vals["routing"][v], len(vi.stack), vi.looped, scale,
                                     [host[s] for s in slots], T, *[node[s] for s in slots], *lt)
            node[vi.next_partial] = out
            host[vi.next_partial] = vals["outs"][v]
        hid = FinalMixH.apply(W, vals["hidden"], [host[s] for s in (0, 1, 2, 3, 6)],
                              *[node[s] for s in (0, 1, 2, 3, 6)])
        return B.HeadLogprob.apply(W, vals["lp"], hid[:-1], ids[1:].contiguous())


# ---------------------------------------------------------------------------------------------------------------
# the whole 48-visit backward as ONE autograd node with a manual reverse loop (no per-visit autograd accumulators:
# the gradient of a visit's output is kept on the card only when the next visit back produced it, else on the host)
# ---------------------------------------------------------------------------------------------------------------
def _head_grad(W, hidden, targets, g):
    """d/d hidden of sum_t g[t] log softmax(30 tanh(W h_t / 30))[targets[t]] (B.HeadLogprob's backward)."""
    class _Ctx:
        pass

    c = _Ctx()
    c.W = W
    c.saved_tensors = (hidden, targets)
    return B.HeadLogprob.backward(c, g)[2]


class LongModel(torch.autograd.Function):
    """lp (the engine's per-target log-probs) as a function of the LoRA tensors; backward = the 48 visits in reverse,
    each recomputed by visit_torch_long from host values, with gradient accumulators keyed by the producing visit."""

    @staticmethod
    def forward(ctx, tf, ids, vals, lora_scale, *lora_tensors):
        ctx.tf, ctx.ids, ctx.vals, ctx.scale = tf, ids, vals, lora_scale
        ctx.save_for_backward(*lora_tensors)
        return vals["lp"].clone()

    @staticmethod
    def backward(ctx, g_lp):
        tf, ids, vals = ctx.tf, ctx.ids, ctx.vals
        eng, W = tf.eng, tf.W
        dev = g_lp.device
        lt_all = ctx.saved_tensors
        nl = len(B.LORA_KEYS)
        lgrads = [None] * len(lt_all)
        T = ids.numel()
        visits = eng.visits
        # producer of every (visit, input slot): the last visit before it that wrote the slot (-1 = x0)
        last = {0: -1}
        prod = []
        for v, vi in enumerate(visits):
            slots = list(vi.stack) + [vi.partial] + (list(vi.bank) if vi.looped else [])
            prod.append([last[s] for s in slots])
            last[vi.next_partial] = v
        host_vals = {-1: vals["x0"]}
        for v in range(len(visits)):
            host_vals[v] = vals["outs"][v]
        acc_dev, acc_host = {}, {}

        def sink_for(p, next_v):
            if p < 0:
                return None
            if p == next_v:
                if p not in acc_dev:
                    acc_dev[p] = torch.zeros(T, D, dtype=F32, device=dev)
                return acc_dev[p]
            if p not in acc_host:
                acc_host[p] = torch.zeros(T, D, dtype=F32, device="cpu", pin_memory=True)
            return acc_host[p]

        # head + final mix
        with B._tick("bwd.head"):
            hid = vals["hidden"][:-1].float()
            g_hid = torch.zeros(T, D, dtype=F32, device=dev)
            g_hid[:-1] = _head_grad(W, hid, ids[1:].contiguous(), g_lp)
            del hid
        fin_slots = (0, 1, 2, 3, 6)
        fin_prod = [last[s] for s in fin_slots]
        def f_mix(a, b, c, d, e):
            return RA.inkling_attn_res([a.float(), b.float(), c.float(), d.float()], e.float(), None, W.q_out,
                                       W.final_norm.reshape(1, D), EPS)[0]

        with torch.enable_grad(), torch.device(dev):
            ph = [torch.zeros(1, dtype=F32, device=dev).expand(T, D) for _ in fin_prod]
            SINK_DUMMIES.clear()
            hm = tc(f_mix, ph, [0] * 5, [0], store_bf16=[True] * 5,
                    vals=[host_vals[p] for p in fin_prod], sinks=[sink_for(p, len(visits) - 1) for p in fin_prod])
            torch.autograd.grad(hm, list(SINK_DUMMIES), g_hid, allow_unused=True)
            SINK_DUMMIES.clear()
        del g_hid, hm, ph
        # visits, last to first
        for v in range(len(visits) - 1, -1, -1):
            vi = visits[v]
            g_out = acc_dev.pop(v, None)
            if v in acc_host:
                h = acc_host.pop(v).to(dev, non_blocking=True)
                g_out = h if g_out is None else g_out.add_(h)
            if g_out is None:
                continue
            ps = prod[v]
            lt = list(lt_all[vi.layer * nl:(vi.layer + 1) * nl]) if ctx.scale is not None else []
            n = len(vi.stack)
            sinks = [sink_for(p, v - 1) for p in ps]
            with torch.enable_grad(), torch.device(dev):
                ins = [torch.zeros(1, dtype=F32, device=dev).expand(T, D) for _ in ps]
                leaves = [t.detach().requires_grad_() for t in lt]
                lora = dict(zip(B.LORA_KEYS, leaves)) | {"scale": ctx.scale} if ctx.scale is not None else None
                SINK_DUMMIES.clear()
                out = visit_torch_long(W, vi, ins[:n], ins[n], (ins[n + 1], ins[n + 2]) if vi.looped else None,
                                       vals["routing"][v], lora, act_vals=[host_vals[p] for p in ps], act_sinks=sinks)
                with B._tick("bwd.autograd"):
                    gr = torch.autograd.grad(out, leaves + list(SINK_DUMMIES), g_out, allow_unused=True)
                gr = gr[:len(leaves)]
                SINK_DUMMIES.clear()
            if not vi.first and sinks[n] is not None:  # the MLP-output conv's residual: d partial += d out (STE)
                sinks[n].add_(g_out.to(sinks[n].device))
            del out, g_out
            for j, gj in enumerate(gr):
                if gj is None:
                    continue
                k = vi.layer * nl + j
                lgrads[k] = gj if lgrads[k] is None else lgrads[k].add_(gj)
        return (None, None, None, None, *lgrads)


def build_manual(tf: "TrainForwardLong", ids, lora=None, banks=None):
    """The engine forward + ONE autograd node (LongModel) over the LoRA tensors: lp [T - 1] with grad."""
    res = forward_host(tf.eng, ids, None if banks is None else [L["adapters"] for L in banks.layers])
    lt = [lo[k] for lo in lora for k in B.LORA_KEYS] if lora is not None else []
    scale = lora[0]["scale"] if lora is not None else None
    return LongModel.apply(tf, ids, res, scale, *lt), res
