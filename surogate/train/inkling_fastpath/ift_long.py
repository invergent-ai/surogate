"""The LEVEL-0 backward for LONG rows on one card (the long-row path).

Same gradient as ift_backward (S2: the reference's W4A16 / STE math per visit, evaluated at the ENGINE's visit inputs,
routing replayed from the engine), restructured so memory no longer grows like (visits x tokens x graph):
- the engine forward keeps every visit output in PINNED HOST memory (bf16, exact), not on the card; the autograd graph
  over the 48 visits carries zero-byte placeholders (the values live on the host and in the engine's head output);
- a visit's recompute is staged: per-token stages (AttnRes / blend inputs, projections + q / r clips, gate / sink +
  wo_ud, MLP input + MoE) run as TokenChunked functions: forward in token chunks without a graph, backward = recompute
  + vjp chunk by chunk (IFT_CHUNK tokens, default 1,024); the sequence-coupled pieces (K / V causal convs, the two
  output convs) run with full-T autograd (small graphs); attention is flash-style (RelAttentionR: the relative bias is
  formed per query block from r and rel_proj, never as a dense [T, 24, 1,024] tensor);
- the final mix is TokenChunked too.
Validation: the same gradient as ift_backward on short rows (tests/long_check.py), the learnability rig via D1 as before.
"""

from __future__ import annotations

import os

import numpy as np
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
        tick = B._tick("bwd.tc." + getattr(ctx.fn, "__name__", "fn"))
        tick.__enter__()
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
                        sink_add(sinks[i].narrow(tdi[i], c0, n), g)
                    else:
                        gin[i].narrow(tdi[i], c0, n).copy_(g)
            for i, p in enumerate(params):
                if p.requires_grad:
                    g = next(it)
                    if g is not None:
                        pgr[i] = g if pgr[i] is None else pgr[i] + g
        tick.__exit__(None, None, None)
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


def sink_add(dst, g):
    """dst += g for a gradient accumulator on the card or in PINNED host memory. A host accumulator is added to ON THE
    CARD (pinned H2D, add, pinned D2H, all non-blocking on the stream): no host sync and no pageable copy."""
    if dst.device == g.device:
        dst.add_(g)
        return
    tmp = dst.to(g.device, non_blocking=True)
    tmp.add_(g)
    dst.copy_(tmp, non_blocking=True)


# ---------------------------------------------------------------------------------------------------------------
# flash attention with the relative bias formed from r per query block
# ---------------------------------------------------------------------------------------------------------------
def _rel_block(r_out, rel_proj, b0, b1):
    """rel [nq, 24, 1024] = bf16(r_out[b0:b1] . rel_proj) (the reference's prologue: rb(einsum('thj,jd->thd')))."""
    return RA.rb(torch.einsum("thj,jd->thd", r_out[b0:b1], rel_proj.float()))


# The attention PATTERN's gradient (through the softmax scores into q, k and the relative bias r). True: the exact backward. False: the
# pattern is a constant of the backward (d scores = 0); only the value path keeps its gradient (dv = p^T dout), dq = dk = dr = 0.
# backward_row sets it per visit (pattern_grad_from); nothing else writes it.
PATTERN_GRAD = {"on": True}


class RelAttentionR(torch.autograd.Function):
    """B.RelAttention with rel = bf16(r_out . rel_proj) formed per query block (STE through the bf16 rounding):
    gradients for q, k, v and r_out; rel_proj is frozen. With PATTERN_GRAD off: the value gradient only."""

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
            sc, _, _ = _scores(q, k, _Shift(rel, b0), b0, b1, k0, window, scale)
            m = sc.amax(-1, keepdim=True)
            p = RA.t_exp(sc - m)
            l = p.sum(-1, keepdim=True)
            if ATTN_TC:
                nq = b1 - b0
                o = _bmm_s(p.reshape(hk, g * nq, -1), vv[:, k0:b1].to(BF16)).reshape(hk, g, nq, dh) / l
            else:
                o = torch.einsum("hgbk,hkd->hgbd", p, vv[:, k0:b1]) / l
            out[b0:b1] = o.permute(2, 0, 1, 3).reshape(b1 - b0, hq, dh)
            lse[b0:b1] = (m + RA.t_log(l))[..., 0].permute(2, 0, 1).reshape(b1 - b0, hq)
        ctx.save_for_backward(q.to(BF16), k.to(BF16), v.to(BF16), r_out.to(BF16), rel_proj, out, lse)
        ctx.window, ctx.q_block, ctx.scale = window, q_block, scale
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        tick = B._tick("bwd.attention")
        tick.__enter__()
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
            sc, gi, valid = _scores(q, k, _Shift(rel, b0), b0, b1, k0, window, scale)
            lb = lse[b0:b1].reshape(nq, hk, g).permute(1, 2, 0)[..., None]
            p = RA.t_exp(sc - lb)
            do = dout[b0:b1].reshape(nq, hk, g, dh).permute(1, 2, 0, 3)
            ob = out[b0:b1].reshape(nq, hk, g, dh).permute(1, 2, 0, 3)
            dlb = dlse[b0:b1].reshape(nq, hk, g).permute(1, 2, 0)[..., None]
            qq = q[b0:b1].reshape(nq, hk, g, dh).permute(1, 2, 0, 3)
            if not PATTERN_GRAD["on"]:  # the (normalised) pattern p is a constant: out = p . v carries gradient to v only
                if ATTN_TC:                # the exact branch's own dv contraction, so dv is the same number in both modes
                    dvb = _bmm_ss(p.reshape(hk, g * nq, -1).transpose(1, 2), do.reshape(hk, g * nq, dh))
                else:
                    dvb = torch.einsum("hgbk,hgbd->hkd", p, do)
                dv[k0:b1] += dvb.permute(1, 0, 2)
                continue
            if ATTN_TC:
                # the same four contractions as below, [Hk, G x nq, .] bmms on bf16 tensor cores (fp32 accumulate):
                # K, V, q hold exact bf16 values; fp32 factors (do, ds, p) are split hi + lo
                v16, k16 = vv[:, k0:b1].to(BF16), kk[:, k0:b1].to(BF16)
                do2 = do.reshape(hk, g * nq, dh)
                dp = _bmm_s(do2, v16.transpose(1, 2)).reshape(hk, g, nq, -1)
                dterm = (do * ob).sum(-1, keepdim=True)
                ds = p * (dp - dterm + dlb)
                ds2 = ds.reshape(hk, g * nq, -1)
                dqb = _bmm_s(ds2, k16).reshape(hk, g, nq, dh)
                dkb = _bmm_s(ds2.transpose(1, 2), qq.reshape(hk, g * nq, dh).to(BF16))
                dvb = _bmm_ss(p.reshape(hk, g * nq, -1).transpose(1, 2), do2)
            else:
                dp = torch.einsum("hgbd,hkd->hgbk", do, vv[:, k0:b1])
                dterm = (do * ob).sum(-1, keepdim=True)
                ds = p * (dp - dterm + dlb)
                dqb = torch.einsum("hgbk,hkd->hgbd", ds, kk[:, k0:b1])
                dkb = torch.einsum("hgbk,hgbd->hkd", ds, qq)
                dvb = torch.einsum("hgbk,hgbd->hkd", p, do)
            dq[b0:b1] += (dqb * scale).permute(2, 0, 1, 3).reshape(nq, hq, dh)
            dk[k0:b1] += (dkb * scale).permute(1, 0, 2)
            dv[k0:b1] += dvb.permute(1, 0, 2)
            dsr = ds.reshape(hq, nq, -1) * valid[None]
            drel = torch.zeros(hq, nq, 1024, dtype=F32, device=q.device)
            drel.scatter_add_(2, gi[None].expand(hq, -1, -1), dsr)
            dr[b0:b1] += torch.einsum("hbd,jd->bhj", drel, projf)
        tick.__exit__(None, None, None)
        return dq, dk, dv, dr, None, None, None


class _Shift:
    """rel rows of one query block addressed with absolute row indices [b0, b1) (B._rel_scores slices rel[b0:b1])."""

    def __init__(self, rel, b0):
        self.rel, self.b0 = rel, b0

    def __getitem__(self, sl):
        return self.rel[sl.start - self.b0:sl.stop - self.b0]


# ---------------------------------------------------------------------------------------------------------------
# LEVEL-1 lever 3b (ATTN_TC, default OFF until its card check): the attention's contractions as bf16 tensor-core bmms
# with fp32 accumulation / output. q, K, V hold exact bf16 values; fp32 factors are split hi + lo (relative error
# ~2^-16). ATTN_TC is read at call time (tests / lever_check switch it per arm).
# ---------------------------------------------------------------------------------------------------------------
ATTN_TC = os.environ.get("IFT_ATTN_TC", "0") == "1"
# ATTN_CUDA (default OFF until its card check): the relative attention of the recompute and its backward run in the CUDA kernel library
# (cuda_bwd.RelAttentionCuda: the flash kernels with the relative bias) instead of RelAttentionR's query-block loop.
ATTN_CUDA = int(os.environ.get("IFT_ATTN_CUDA", "0"))  # 1: the cotangent of the output enters rounded to bf16; 2: as its bf16 high + low parts (two passes)


def rel_attention(q, k, v, r_out, rel_proj, window, q_block):
    """The visit's relative attention: RelAttentionR, or the kernel library's with ATTN_CUDA (exact pattern gradient only)."""
    if ATTN_CUDA:
        if not PATTERN_GRAD["on"]:
            raise ValueError("ATTN_CUDA has no constant-pattern backward (pattern_grad_from needs IFT_ATTN_CUDA=0)")
        import cuda_bwd as CB

        return CB.RelAttentionCuda.apply(q, k, v, r_out, rel_proj, window, ATTN_CUDA)
    return RelAttentionR.apply(q, k, v, r_out, rel_proj, window, q_block)


def _tc_ok(dev):
    if dev not in B._OUT_DTYPE:
        B._OUT_DTYPE[dev] = B._probe_out_dtype(dev)
    return B._OUT_DTYPE[dev]


def _bmm16(a16, b16):
    """a [h, n, m] @ b [h, m, d], bf16 operands, fp32 accumulate / output (fp32 fallback where out_dtype is missing)."""
    if _tc_ok(a16.device):
        return torch.bmm(a16, b16, out_dtype=F32)
    return torch.bmm(a16.float(), b16.float())


def _split16(x):
    hi = x.to(BF16)
    return hi, (x - hi.float()).to(BF16)


def _bmm_s(a, b16):
    """fp32 a @ bf16 b: a split hi + lo."""
    hi, lo = _split16(a)
    return _bmm16(hi, b16) + _bmm16(lo, b16)


def _bmm_ss(a, b):
    """fp32 a @ fp32 b: both split hi + lo."""
    ah, al = _split16(a)
    bh, bl = _split16(b)
    return _bmm16(ah, bh) + _bmm16(ah, bl) + _bmm16(al, bh) + _bmm16(al, bl)


def _rel_scores_tc(q, k, rel, b0, b1, k0, window, scale):
    """B._rel_scores with the q . K contraction on tensor cores (both operands exact bf16 values)."""
    nq = b1 - b0
    hq, dh = q.shape[1], q.shape[2]
    hk = k.shape[1]
    g = hq // hk
    qq = q[b0:b1].reshape(nq, hk, g, dh).permute(1, 2, 0, 3).reshape(hk, g * nq, dh)  # [Hk, G x nq, D]
    kk = k[k0:b1].permute(1, 0, 2)  # [Hk, K, D]
    s = _bmm16(qq.to(BF16), kk.to(BF16).transpose(1, 2)).reshape(hk, g, nq, -1) * scale
    dist = torch.arange(b0, b1, device=q.device)[:, None] - torch.arange(k0, b1, device=q.device)[None, :]
    gi = dist.clamp(0, 1023)
    rr = rel[b0:b1].permute(1, 0, 2)  # [Hq, nq, E]
    bias = torch.gather(rr, 2, gi[None].expand(hq, -1, -1))
    in_band = (dist >= 0) & (dist < 1024)
    bias = torch.where(in_band, bias, torch.zeros((), dtype=bias.dtype, device=bias.device))
    s = s + bias.reshape(hk, g, nq, -1)
    vis = dist >= 0
    if window > 0:
        vis &= dist < window
    s = s.masked_fill(~vis, float("-inf"))
    return s, gi, in_band & vis


def _scores(q, k, rel, b0, b1, k0, window, scale):
    return (_rel_scores_tc if ATTN_TC else B._rel_scores)(q, k, rel, b0, b1, k0, window, scale)



# ---------------------------------------------------------------------------------------------------------------
# LEVEL-1 lever 1: the MoE backward group-major (same S2 math as rig.moe_vec; each 16-expert group's units decoded ONCE
# per visit per projection; rows of the whole sequence per group, in row blocks of MOE_ROWS)
# ---------------------------------------------------------------------------------------------------------------
MOE_ROWS = int(os.environ.get("IFT_MOE_ROWS", "8192"))  # lev3-1: 32k rows per block OOMed a 32k row
# LEVEL-1 lever 3: query-block rows of the flash attention adapted to the key span (fewer, larger blocks; the per-block
# score tensors [24, rows, keys] fp32 stay under ATTN_BLOCK_BYTES); 0 = the fixed 128 rows
ATTN_ADAPT = os.environ.get("IFT_ATTN_ADAPT", "1") == "1"
ATTN_BLOCK_BYTES = int(os.environ.get("IFT_ATTN_BLOCK_MB", "256")) << 20


def _q_block(T, window, hq=24):
    if not ATTN_ADAPT:
        return 128
    best = 128
    for qb in (2048, 1024, 512, 256):
        keys = min(T, qb + window - 1) if window > 0 else T
        if hq * qb * keys * 4 <= ATTN_BLOCK_BYTES:
            best = qb
            break
    return best
MOE_GM = os.environ.get("IFT_MOE_GM", "1") == "1"
# MOE_CUDA (default OFF until its card check; needs MOE_GM): the group loop of MoEGM's backward runs in the CUDA kernel library
# (cuda_bwd.moe_groups, csrc/src/kernels/inkling/moe_backward.cu) instead of the torch recompute + autograd per row block.
MOE_CUDA = os.environ.get("IFT_MOE_CUDA", "0") == "1"


class UnitBMM(torch.autograd.Function):
    """a [g, cap, K] @ unit[g]^T (bf16 operands, fp32 accumulate / output) with a given decoded unit tensor (no copy
    saved); backward da = g @ unit with g rounded to bf16 for the tensor cores (as rig.ExpertBMM)."""

    @staticmethod
    def forward(ctx, a, unit):
        ctx.unit = unit
        return B._bmm_f32(a, unit)

    @staticmethod
    def backward(ctx, g):
        if B._OUT_DTYPE.get(g.device):
            return torch.bmm(g.to(BF16), ctx.unit, out_dtype=F32), None
        return torch.bmm(g.float(), ctx.unit.float()), None


def _moe_rows(xr, gr, eg, pg, cap, ng, u13, f13, u2, f2, lo, scale):
    """down rows (fp32, bf16 values) of rows xr [n, H] routed to experts eg [n] (0..ng-1 within the group) at padded
    positions pg [n] (< cap; host int cap: no device sync here), gammas gr [n]: rig.moe_vec's per-group arithmetic
    (W4A16: unit GEMM x factor, + LoRA from the fp32 tensors, one rounding; softcapped SwiGLU with the gamma; down
    likewise). Padding / unpadding by flat row indices (index_copy / index_select: cheap backwards)."""
    flat = eg * cap + pg

    def pad(rows):
        z = torch.zeros(ng * cap, rows.shape[1], dtype=rows.dtype, device=rows.device)
        return z.index_copy(0, flat, rows).view(ng, cap, rows.shape[1])

    def unpad(p3):
        return p3.reshape(ng * cap, p3.shape[2]).index_select(0, flat)

    fe = f13[eg][:, None]
    xp = pad(xr)
    gu = unpad(UnitBMM.apply(xp, u13)) * fe
    if lo is not None:
        la = torch.bmm(xp, lo["A13"].transpose(1, 2))
        gu = gu + scale * unpad(torch.bmm(la, lo["B13"].transpose(1, 2)))
    gu = RA.rb(gu)
    y = RA.inkling_softcap_swiglu(gu, gr)
    yp = pad(y)
    dn = unpad(UnitBMM.apply(yp, u2)) * f2[eg][:, None]
    if lo is not None:
        la = torch.bmm(yp, lo["A2"].transpose(1, 2))
        dn = dn + scale * unpad(torch.bmm(la, lo["B2"].transpose(1, 2)))
    return RA.rb(dn)


# ---------------------------------------------------------------------------------------------------------------
# LEVEL-1 lever 1b (MOE_FAST, default OFF until its card check): the same S2 vjp as _moe_rows, without the id-major
# padding and without autograd over the GEMMs. Real routing (a MoE bucket census, P3' rows, 48 windows of
# 8k): the padded bmm of a 16-expert group carries 3.96x the real rows (one expert holds most of them); with the group's
# experts sorted by row count and cut optimally into <= 3 batches it carries 1.34x. The down projection's forward
# value is never computed: its output is rounded straight-through and only its cotangent is used.
# ---------------------------------------------------------------------------------------------------------------
MOE_FAST = os.environ.get("IFT_MOE_FAST", "0") == "1"
MOE_BATCHES = int(os.environ.get("IFT_MOE_BATCHES", "3"))
# MOE_DET (default OFF until its card check): lev-all-1 found the path before MoEGM repeats bit for bit (A/A rel err 0.0) and
# every MoEGM arm does not (0.36-0.92 % at 1k). A token routed to two experts of one block repeats in index_add_'s index, and
# the card adds repeats in arbitrary order in fp32; the backward's bf16 roundings then amplify the ulp differences. With
# MOE_DET the rows' dX are summed in fp64 (order-independent for these few terms) and rounded once to fp32 per visit.
MOE_DET = os.environ.get("IFT_MOE_DET", "0") == "1"
# lev-all-1: MOE_FAST costs +1.0 s at 1k (3.2 vs 2.2 s; three batches of python ops + unit copies per group, for ~1k rows) and
# gains 4.0 s at 8k. A group below MOE_SPLIT_MIN rows runs as ONE batch in expert-id order, without copying the units.
MOE_SPLIT_MIN = int(os.environ.get("IFT_MOE_SPLIT_MIN", "2048"))
# MOE_PAD_ROWS (0 = off, default OFF until its card check): at most this many padded rows (batch items x longest item) in one
# MOE_FAST batch. MOE_ROWS caps an expert's ITEM, so a batch could hold 16 x 4,096 = 65,536 real rows (tools/moe_batch_mem.py on real
# 32k routing: 576 MB per row tensor, vs 36 MB on the old block path); lev-all-3 / -4: MOE_FAST out of memory at 32k.
MOE_PAD_ROWS = int(os.environ.get("IFT_MOE_PAD_ROWS", "0"))


def _h2d(t, dev):
    """A small host tensor on `dev` through pinned memory (no pageable copy, no host sync); plain .to off CUDA."""
    if torch.device(dev).type == "cuda":
        return t.pin_memory().to(dev, non_blocking=True)
    return t.to(dev)


def _moe_plan(cnt, max_rows, nb):
    """Batches of one group: lists of items (rows, expert within the group, first row within the group's sorted rows),
    items longest first. An expert above max_rows is cut into several items. The sorted items are partitioned into at
    most nb contiguous batches minimising sum(batch items x its longest item), then batches above EG items are cut."""
    if sum(cnt) == 0:
        return []
    if (sum(cnt) < MOE_SPLIT_MIN and max(cnt) <= max_rows and len(cnt) <= B.EG
            and (not MOE_PAD_ROWS or len(cnt) * max(cnt) <= MOE_PAD_ROWS)):  # one batch, id order (rows may be 0)
        off, one = 0, []
        for e, c in enumerate(cnt):
            one.append((c, e, off))
            off += c
        return [one]
    items = []
    off = 0
    for e, c in enumerate(cnt):
        for s in range(0, c, max_rows):
            items.append((min(max_rows, c - s), e, off + s))
        off += c
    n = len(items)
    items.sort(key=lambda t: (-t[0], t[1], t[2]))
    inf = 1 << 62
    f = [[inf] * (n + 1) for _ in range(nb + 1)]
    cut = [[0] * (n + 1) for _ in range(nb + 1)]
    f[0][0] = 0
    for b in range(1, nb + 1):
        fp, fb, cb = f[b - 1], f[b], cut[b]
        for j in range(b, n + 1):
            for i in range(b - 1, j):
                if fp[i] < inf:
                    v = fp[i] + (j - i) * items[i][0]
                    if v < fb[j]:
                        fb[j], cb[j] = v, i
    best = min(range(1, nb + 1), key=lambda b: f[b][n])
    parts = []
    j = n
    for b in range(best, 0, -1):
        i = cut[b][j]
        parts.append(items[i:j])
        j = i
    parts.reverse()
    out = []
    for bt in parts:  # items longest first: a batch's padded rows = its size x its first item
        k = 0
        while k < len(bt):
            room = B.EG if not MOE_PAD_ROWS else max(1, min(B.EG, MOE_PAD_ROWS // bt[k][0]))
            out.append(bt[k:k + room])
            k += room
    return out


def _unit_bwd(g3, unit):
    """g [b, cap, N] @ unit [b, N, K]: UnitBMM's backward (g rounded to bf16 for the tensor cores, as rig.ExpertBMM)."""
    if B._OUT_DTYPE.get(g3.device):
        return torch.bmm(g3.to(BF16), unit, out_dtype=F32)
    return torch.bmm(g3.float(), unit.float())


@torch.no_grad()
def _moe_fast_batch(items, r0, x, g, tok, gam_r, u13, f13, u2, f2, lora, scale, g0, g1, dx, dgam_r, lgr):
    """One batch of a group: adds its rows' dX to dx, writes their d(gamma) into dgam_r, adds the LoRA gradients."""
    dev = x.device
    nb = len(items)
    lens = np.fromiter((it[0] for it in items), dtype=np.int64, count=nb)
    cap = int(lens.max())
    es = np.fromiter((it[1] for it in items), dtype=np.int64, count=nb)
    whole = nb == g1 - g0 and bool((es == np.arange(nb)).all())  # the group in id order: the units are used as they are
    firsts = np.fromiter((it[2] for it in items), dtype=np.int64, count=nb)
    kk = np.repeat(np.arange(nb), lens)
    pos = np.arange(int(lens.sum())) - np.repeat(np.cumsum(lens) - lens, lens)
    rel = _h2d(torch.from_numpy(firsts[kk] + pos + r0), dev)  # rows of this batch in the visit's sorted rows
    flat = _h2d(torch.from_numpy(kk * cap + pos), dev)  # their positions in the padded [nb, cap] layout
    kt = _h2d(torch.from_numpy(kk), dev)
    eix = _h2d(torch.from_numpy(es), dev)

    def pad(rows):
        z = torch.zeros(nb * cap, rows.shape[1], dtype=rows.dtype, device=dev)
        return z.index_copy(0, flat, rows).view(nb, cap, rows.shape[1])

    def unpad(p3):
        return p3.reshape(nb * cap, p3.shape[2]).index_select(0, flat)

    tb = tok[rel]
    U13, U2 = (u13, u2) if whole else (u13.index_select(0, eix), u2.index_select(0, eix))
    fe13, fe2 = f13[eix][kt][:, None], f2[eix][kt][:, None]
    xp = pad(x[tb])
    gu = unpad(B._bmm_f32(xp, U13)) * fe13
    A13 = B13 = A2 = B2 = la13 = None
    if lora is not None:
        A13, B13, A2, B2 = [t[g0:g1] if whole else t[g0:g1].index_select(0, eix) for t in lora]
        la13 = torch.bmm(xp, A13.transpose(1, 2))
        gu = gu + scale * unpad(torch.bmm(la13, B13.transpose(1, 2)))
    gu = RA.rb(gu)
    with torch.enable_grad():
        gul = gu.detach().requires_grad_()
        gml = gam_r[rel].detach().requires_grad_()
        y = RA.inkling_softcap_swiglu(gul, gml)
    del gu
    # the down projection's vjp at y
    c = g[tb]
    dyp = _unit_bwd(pad(c * fe2), U2)
    if lora is not None:
        yp = pad(y.detach())
        pc = pad(scale * c)
        dla2 = torch.bmm(pc, B2)
        dB2 = torch.bmm(pc.transpose(1, 2), torch.bmm(yp, A2.transpose(1, 2)))
        dA2 = torch.bmm(dla2.transpose(1, 2), yp)
        dyp = dyp + torch.bmm(dla2, A2)
        del yp, pc, dla2
    del c
    dy = unpad(dyp)
    del dyp
    with torch.enable_grad():
        dgu, dgm = torch.autograd.grad(y, (gul, gml), dy)
    del y, gul, gml, dy
    # the gate / up projection's vjp at x
    dxp = _unit_bwd(pad(dgu * fe13), U13)
    if lora is not None:
        pc = pad(scale * dgu)
        dla13 = torch.bmm(pc, B13)
        dB13 = torch.bmm(pc.transpose(1, 2), la13)
        dA13 = torch.bmm(dla13.transpose(1, 2), xp)
        dxp = dxp + torch.bmm(dla13, A13)
        repeats = len(set(es.tolist())) < nb  # an expert cut into items: add its parts in a fixed order, not by atomics
        for j, t in enumerate((dA13, dB13, dA2, dB2)):  # B.LORA_KEYS order
            if lgr[j] is None:
                continue
            if repeats and MOE_DET:
                for k, e in enumerate(es.tolist()):
                    lgr[j][g0 + e] += t[k]
            else:
                lgr[j][g0:g1].index_add_(0, eix, t)
    dx.index_add_(0, tb, unpad(dxp).to(dx.dtype))
    dgam_r[rel] = dgm


class MoEGM(torch.autograd.Function):
    """moe_out (placeholder: no consumer reads its value -- the MLP-output conv's backward is linear) as a function of
    mlp_in x [T, H] and the layer's LoRA tensors; backward group-major over the whole sequence."""

    @staticmethod
    def forward(ctx, experts, ids, scale, x, *lora_tensors):
        ctx.experts, ctx.ids, ctx.scale = experts, ids, scale
        ctx.offload = x.shape[0] >= OFFLOAD_MIN
        xb = x.detach().to(BF16)
        if ctx.offload:
            h = torch.empty(xb.shape, dtype=BF16, device="cpu", pin_memory=True)
            h.copy_(xb, non_blocking=True)
            ctx.xh = h
            ctx.save_for_backward(*lora_tensors)
        else:
            ctx.xh = None
            ctx.save_for_backward(xb, *lora_tensors)
        ctx.dev = x.device
        return torch.zeros(1, dtype=F32, device=x.device).expand(x.shape[0], x.shape[1])

    @staticmethod
    def backward(ctx, g):
        tick = B._tick("bwd.moe_gm")
        tick.__enter__()
        w = ctx.experts
        st = ctx.saved_tensors
        if ctx.xh is not None:
            x16 = ctx.xh.to(ctx.dev, non_blocking=True)
            lt = st
        else:
            x16 = st[0]
            lt = st[1:]
        x = x16.float()
        dev = ctx.dev
        g = g.contiguous()
        T, H = x.shape
        ids = ctx.ids.long()
        topk = ids.shape[1]
        need = ctx.needs_input_grad[4:]
        lora = lt if (ctx.scale is not None and len(lt) == 4) else None
        with torch.no_grad():
            gam = RA.route_weights(RA._router_logits(x, w.gate_w), ids)
        flat = ids.reshape(-1)
        order = torch.argsort(flat, stable=True)
        es = flat[order]
        tok = order // topk
        slot = order % topk
        counts = torch.bincount(flat, minlength=B.E)
        starts_d = torch.cumsum(counts, 0) - counts
        pos_all = torch.arange(flat.numel(), device=dev) - starts_d[es]  # position within its expert
        starts = starts_d.tolist()  # the visit's one host sync: every group's rows and padding are known from here
        cnt = counts.tolist()
        gam_r = gam[tok, slot]
        dx = torch.zeros(T, H, dtype=torch.float64 if (MOE_DET and not MOE_CUDA) else F32, device=dev)
        dgam_r = torch.zeros_like(gam_r)
        lgr = [torch.zeros_like(t) if (lora is not None and nd) else None for t, nd in zip(lt, need)] if lora else []
        if MOE_CUDA:
            import cuda_bwd as CB

            with B._tick("bwd.moe.cuda"):
                CB.moe_groups(w, x16, g, tok, gam_r, starts, cnt, lora, ctx.scale, dx, dgam_r, lgr, MOE_ROWS)
        for g0 in ([] if MOE_CUDA else range(0, B.E, B.EG)):
            g1 = min(B.E, g0 + B.EG)
            r0, r1 = starts[g0], starts[g1 - 1] + cnt[g1 - 1]
            if r1 == r0:
                continue
            with B._tick("bwd.moe.decode"):
                u13, f13 = w.eng.decode_experts(w.layer, "w13", g0, g1 - g0)
                u2, f2 = w.eng.decode_experts(w.layer, "w2", g0, g1 - g0)
            if w.fmt != "nvfp4":
                f13 = torch.ones_like(f13)
                f2 = torch.ones_like(f2)
            if MOE_FAST:
                with B._tick("bwd.moe.fast"):
                    for items in _moe_plan(cnt[g0:g1], MOE_ROWS, MOE_BATCHES):
                        _moe_fast_batch(items, r0, x, g, tok, gam_r, u13, f13, u2, f2, lora, ctx.scale, g0, g1, dx,
                                        dgam_r, lgr)
                del u13, u2
                continue
            for b0 in range(r0, r1, MOE_ROWS):
                b1 = min(r1, b0 + MOE_ROWS)
                tb = tok[b0:b1]
                eb = es[b0:b1] - g0
                # padded positions of this row block (host arithmetic: rows of expert e in [b0, b1))
                lo_e = [max(starts[e], b0) for e in range(g0, g1)]
                hi_e = [min(starts[e] + cnt[e], b1) for e in range(g0, g1)]
                cap = max(1, max(h - l for l, h in zip(lo_e, hi_e)))
                off = _h2d(torch.tensor([lo_e[i] - starts[g0 + i] for i in range(g1 - g0)]), dev)
                pb = pos_all[b0:b1] - off[eb]
                with torch.enable_grad():
                    xr = x[tb].detach().requires_grad_()
                    grr = gam_r[b0:b1].detach().requires_grad_()
                    lo = None
                    leaves = []
                    if lora is not None:
                        leaves = [t[g0:g1].detach().requires_grad_() for t in lora]
                        lo = dict(zip(B.LORA_KEYS, leaves))
                    with B._tick("bwd.moe.rows_fwd"):
                        dn = _moe_rows(xr, grr, eb, pb, cap, g1 - g0, u13, f13, u2, f2, lo, ctx.scale)
                    with B._tick("bwd.moe.rows_bwd"):
                        grads = torch.autograd.grad(dn, [xr, grr] + leaves, g[tb])
                dx.index_add_(0, tb, grads[0].to(dx.dtype))
                dgam_r[b0:b1] = grads[1]
                for j, gj in enumerate(grads[2:]):
                    if lgr[j] is not None:
                        lgr[j][g0:g1] += gj
            del u13, u2
        if dx.dtype != F32:
            dx = dx.float()
        # the routing weights' path back into x (router logits from x, ids fixed)
        dgam = torch.zeros(T, topk, dtype=F32, device=dev)
        dgam[tok, slot] = dgam_r
        with torch.enable_grad(), B._tick("bwd.moe.router"):
            xl = x.detach().requires_grad_()
            gm = RA.route_weights(RA._router_logits(xl, w.gate_w), ids)
            dx += torch.autograd.grad(gm, xl, dgam)[0]
        tick.__exit__(None, None, None)
        return (None, None, None, dx, *lgr)


# ---------------------------------------------------------------------------------------------------------------
# LEVEL-1 lever 2: tensor-core GEMMs in the recompute
#   FAST_GEMM: x (bf16 values) @ W (bf16 weight) as a bf16 MMA with fp32 accumulation / output (the same products as
#     the fp32 GEMM of those values; another summation order), dX = g @ W as TWO bf16 MMAs on g's hi / lo bf16 parts
#     (g = hi + lo, relative error ~2^-16): replaces the fp32 GEMMs with per-call fp32 copies of the weights.
#   TF32: the remaining fp32 GEMMs (attention, LoRA, controllers) on TF32 tensor cores.
# ---------------------------------------------------------------------------------------------------------------
FAST_GEMM = os.environ.get("IFT_FAST_GEMM", "1") == "1"
TF32 = os.environ.get("IFT_TF32", "0") == "1"  # lev2-1: TF32 drifts the gradient (cos 0.99978, rel err 2 %)
_RA_MM = RA.mm
_RA_BMM = RA.bmm


def _split_bf16(g):
    hi = g.to(BF16)
    return hi, (g - hi.float()).to(BF16)


class MM16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.w = w
        return B._mm_f32(x.to(BF16), w)

    @staticmethod
    def backward(ctx, g):
        hi, lo = _split_bf16(g.contiguous())
        wt = ctx.w.T  # [K, N] view: g [M, N] @ W [N, K]
        return B._mm_f32(hi, wt) + B._mm_f32(lo, wt), None


class BMM16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.w = w
        return torch.bmm(x.to(BF16), w.transpose(1, 2), out_dtype=F32)

    @staticmethod
    def backward(ctx, g):
        hi, lo = _split_bf16(g.contiguous())
        return torch.bmm(hi, ctx.w, out_dtype=F32) + torch.bmm(lo, ctx.w, out_dtype=F32), None


def _mm_fast(x, w):
    if w.dtype == BF16 and w.dim() == 2 and torch.is_grad_enabled():
        return MM16.apply(x.float(), w)
    if w.dtype == BF16 and w.dim() == 2:
        return B._mm_f32(x.to(BF16), w)
    return _RA_MM(x, w)


def _bmm_fast(x, w):
    if w.dtype == BF16 and w.dim() == 3:
        return BMM16.apply(x.float(), w) if torch.is_grad_enabled() else torch.bmm(
            x.to(BF16), w.transpose(1, 2), out_dtype=F32)
    return _RA_BMM(x, w)


class _gemm_mode:
    """Inside the recompute: FAST_GEMM patches the vendored reference's mm / bmm; TF32 enables TF32 matmuls."""

    def __enter__(self):
        self.tf32 = torch.backends.cuda.matmul.allow_tf32
        if FAST_GEMM:
            RA.mm, RA.bmm = _mm_fast, _bmm_fast
        torch.backends.cuda.matmul.allow_tf32 = bool(TF32)

    def __exit__(self, *a):
        RA.mm, RA.bmm = _RA_MM, _RA_BMM
        torch.backends.cuda.matmul.allow_tf32 = self.tf32

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
        win = 1024 if L.is_local else 0
        out, lse = rel_attention(q_out, K.float(), V.float(), r_out, L.rel_proj, win, _q_block(T, win))
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
        if MOE_GM:
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
        with _gemm_mode():
            return LongModel._backward(ctx, g_lp)

    @staticmethod
    def _backward(ctx, g_lp):
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
                sink_add(sinks[n], g_out)
            del out, g_out
            for j, gj in enumerate(gr):
                if gj is None:
                    continue
                k = vi.layer * nl + j
                lgrads[k] = gj if lgrads[k] is None else lgrads[k].add_(gj)
        return (None, None, None, None, *lgrads)


def build_manual(tf: "TrainForwardLong", ids, lora=None, banks=None):
    """The engine forward + ONE autograd node (LongModel) over the LoRA tensors: lp [T - 1] with grad."""
    with B._tick("fwd.engine"):
        res = forward_host(tf.eng, ids, None if banks is None else [L["adapters"] for L in banks.layers])
    lt = [lo[k] for lo in lora for k in B.LORA_KEYS] if lora is not None else []
    scale = lora[0]["scale"] if lora is not None else None
    return LongModel.apply(tf, ids, res, scale, *lt), res
