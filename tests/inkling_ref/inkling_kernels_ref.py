"""fp64 references (forward + backward) of the Inkling-2-Mini trainer kernels (csrc/src/kernels/inkling).

One function pair per op (the numbering of csrc/src/kernels/inkling/inkling.h):
  1 log_tau          2 qkv_norm (q / r scaled L2 clip [+ tau], k L2 clip, v RMS; optional e4m3 KV store)
  3 short_conv       4 dynamic_conv          5 attention near        6 attention far      7 merge + sink + gate
  8 attn_res         9 loop_blend           10 softcap_swiglu
The backward functions are the formulas the CUDA kernels implement (hand-written, NOT autograd); a gradcheck
checks every one with torch.autograd.gradcheck (finite differences in fp64) through the `*Fn` autograd wrappers below.

Semantics: the reference forward of the model (surogate/train/inkling_fastpath/ift_ref.py) and the engine's rounding
points. Every function computes in the dtype of its inputs
(fp64 for the reference, fp32 for the "pt" emulation of the tolerance rule) and applies the engine's storage
roundings (bf16 / e4m3) only when `rounding=True`. The backward passes treat every rounding as identity
(straight-through), which is what the kernels do; gradcheck therefore runs with `rounding=False`.

Conventions (match the kernels): token-major; positions = position within the document (int64 here); a chunk's
documents are told apart by `positions` alone for the convolutions (tap of lag L valid iff positions[t] >= L) and by
explicit document ids for attention (the kernels get the equivalent per-segment cu_q / cu_k geometry).
Attention lse tensors are [H, T] (FlashAttention's unpadded layout); attention outputs [T, H, d].
"""

from __future__ import annotations

import math

import torch

F64 = torch.float64
CLIP, RMS = 0, 1


def rb(x: torch.Tensor) -> torch.Tensor:
    """Round to bf16, keep the dtype (a bf16 store)."""
    return x.to(torch.bfloat16).to(x.dtype)


def r8(x: torch.Tensor) -> torch.Tensor:
    """Round an fp32 value to e4m3fn (RNE; the engine's `cvt.rn.satfinite` from fp32), keep the dtype."""
    return x.float().to(torch.float8_e4m3fn).to(x.dtype)


# ---------------------------------------------------------------------------------------------------------------
# 1. log tau (no gradient)
# ---------------------------------------------------------------------------------------------------------------
def log_tau(positions: torch.Tensor, n_floor: float = 8192.0, alpha: float = 0.1, dtype=F64) -> torch.Tensor:
    """tau = 1 + alpha * ln(max((pos + 1) / n_floor, 1)) (`ops/attention.py` compute_log_scaling_tau)."""
    ratio = torch.clamp((positions.to(dtype) + 1) / n_floor, min=1.0)
    return 1.0 + alpha * torch.log(ratio)


# ---------------------------------------------------------------------------------------------------------------
# 2. qkv_norm: per-head scaled L2 clip (q, r, k) or unweighted RMS (v); tau on global q / r; e4m3 KV store option
# ---------------------------------------------------------------------------------------------------------------
def qkv_norm_fwd(x, scale=None, tau=None, th=22.627416997969522, eps=1e-6, mode=CLIP, store="bf16", rounding=True):
    """x [T, H, d] (bf16 values). xs = x * scale; CLIP: y = xs * min(th / (||xs|| + eps), 1); RMS: y = xs *
    rsqrt(mean(xs^2) + eps). Rounding (engine): q / r: bf16(y), global: bf16(bf16(y) * tau); k / v: stored to the KV
    cache straight from fp32: e4m3 (FP8 KV, store='e4m3') or bf16."""
    xs = x * scale if scale is not None else x
    if mode == CLIP:
        n = torch.linalg.vector_norm(xs, dim=-1, keepdim=True)
        y = xs * torch.clamp(th / (n + eps), max=1.0)
    else:
        y = xs * torch.rsqrt((xs * xs).mean(-1, keepdim=True) + eps)
    if tau is not None:
        assert store == "bf16"
        y = (rb(y) if rounding else y) * tau[:, None, None]
    if rounding:
        y = r8(y) if store == "e4m3" else rb(y)
    return y


def qkv_norm_bwd(dy, x, scale=None, tau=None, th=22.627416997969522, eps=1e-6, mode=CLIP):
    """dx [T, H, d] from dy (fp32 in the kernels: dq / dk / dv come fp32 out of the attention backward).
    CLIP, clipped row (th / (n + eps) < 1): dxs = f * (g - (xs . g) xs / (n (n + eps))), f = th / (n + eps);
    unclipped: dxs = g. RMS: dxs = r g - r^3 / d (xs . g) xs. g = dy * tau; dx = dxs * scale. Recomputed from x."""
    g = dy * tau[:, None, None] if tau is not None else dy
    xs = x * scale if scale is not None else x
    d = x.shape[-1]
    dot = (xs * g).sum(-1, keepdim=True)
    if mode == CLIP:
        n = torch.linalg.vector_norm(xs, dim=-1, keepdim=True)
        c = th / (n + eps)
        clipped = c < 1.0
        ns = torch.where(clipped, n, torch.ones_like(n))  # keeps the unselected branch finite
        dxs = torch.where(clipped, c * (g - dot * xs / (ns * (ns + eps))), g)
    else:
        r = torch.rsqrt((xs * xs).mean(-1, keepdim=True) + eps)
        dxs = r * g - (r ** 3 / d) * dot * xs
    return dxs * scale if scale is not None else dxs


# ---------------------------------------------------------------------------------------------------------------
# 3 / 4. short convolutions with document-aware taps and chunk carries
# ---------------------------------------------------------------------------------------------------------------
def _full(x, carry):
    """[3 + T, C]: the previous chunk's last 3 inputs (oldest first; zeros if none), then x."""
    c = torch.zeros(3, x.shape[1], dtype=x.dtype) if carry is None else carry.to(x.dtype)
    return torch.cat((c, x), 0)


def _tap(full, positions, lag):
    """Input `lag` steps back for every token, zero where the document starts later (positions[t] < lag)."""
    t = full.shape[0] - 3
    m = (positions >= lag).to(full.dtype)[:, None]
    return full[3 - lag : 3 - lag + t] * m


def short_conv_fwd(x, w, positions, carry=None, addend=None, residual=None, scale=5 ** -0.5, rounding=True):
    """Static causal depthwise conv, kernel 4 (`ops/sconv.py` `_tap_sum` static): x [T, C] bf16 values, w [C, 4]
    (tap 3 = current token). conv = (sum_k w[c,k] x[t-3+k] + x[t]) * 5^-0.5 -> bf16; with addend:
    out = bf16((residual + addend) + conv_bf16) (fp32 sums; the visit's partial output). Returns (out, new_carry)."""
    full = _full(x, carry)
    acc = torch.zeros_like(x)
    for k in range(4):
        acc = acc + w[:, k][None] * _tap(full, positions, 3 - k)
    conv = (acc + x) * scale
    if rounding:
        conv = rb(conv)
    out = conv
    if addend is not None:
        a = addend if residual is None else residual + addend
        out = a + conv
        if rounding:
            out = rb(out)
    return out, full[-3:].clone()


def short_conv_bwd(dout, w, positions, d_carry_next=None, scale=5 ** -0.5, has_addend=False, has_residual=False):
    """dx [T, C] = scale * dout[t] + sum_lag scale * w[c, 3-lag] * dout[t+lag] (valid taps only)
    + d_carry_next on the last 3 rows (gradient of the NEXT chunk w.r.t. this chunk's last inputs);
    d_carry [3, C] = gradient w.r.t. this chunk's carry (to the previous chunk). d_addend = d_residual = dout
    (straight-through the roundings). Requires T >= 3 (chunks are 2,048 tokens)."""
    t = dout.shape[0]
    assert t >= 3
    d_full = torch.zeros(t + 3, dout.shape[1], dtype=dout.dtype)
    for k in range(4):
        lag = 3 - k
        m = (positions >= lag).to(dout.dtype)[:, None]
        d_full[3 - lag : 3 - lag + t] += scale * w[:, k][None] * dout * m
    dx = d_full[3:] + scale * dout
    if d_carry_next is not None:
        dx[-3:] += d_carry_next
    return dx, d_full[:3].clone(), (dout if has_addend else None), (dout if has_residual else None)


def dynamic_conv_fwd(x, static_w, dyn, positions, carry=None, scale=9 ** -0.5, head_dim=128, rounding=True):
    """Dynamic conv of global visits' K / V (`_tap_sum` with dyn_ptr): x [T, C], static_w [4, C] (lag 0 = current),
    dyn [T, H * 4] = bf16(weight_proj(x_k or x_v)); tap(t,c,lag) = static_w[lag,c] + dyn[t, (c // hd) * 4 + lag];
    y = (sum_lag tap * x[t-lag] + x[t]) * 9^-0.5 -> bf16. Returns (y, new_carry)."""
    t, c = x.shape
    h = c // head_dim
    full = _full(x, carry)
    dyn4 = dyn.reshape(t, h, 4)
    acc = torch.zeros_like(x)
    for lag in range(4):
        tap = (static_w[lag].reshape(1, h, head_dim) + dyn4[:, :, lag : lag + 1]).reshape(t, c)
        acc = acc + tap * _tap(full, positions, lag)
    y = (acc + x) * scale
    return (rb(y) if rounding else y), full[-3:].clone()


def dynamic_conv_bwd(dy, x, static_w, dyn, positions, carry=None, d_carry_next=None, scale=9 ** -0.5, head_dim=128):
    """dx [T, C], d_dyn [T, H * 4] (d_dyn[t,h,lag] = scale * sum_{c in head h} x[t-lag, c] dy[t, c]), d_carry [3, C].
    static_w and weight_proj are frozen: no gradient. Requires T >= 3."""
    t, c = x.shape
    assert t >= 3
    h = c // head_dim
    full = _full(x, carry)
    dyn4 = dyn.reshape(t, h, 4)
    d_full = torch.zeros(t + 3, c, dtype=dy.dtype)
    d_dyn = torch.zeros(t, h, 4, dtype=dy.dtype)
    for lag in range(4):
        m = (positions >= lag).to(dy.dtype)[:, None]
        tap = (static_w[lag].reshape(1, h, head_dim) + dyn4[:, :, lag : lag + 1]).reshape(t, c)
        d_full[3 - lag : 3 - lag + t] += scale * tap * dy * m
        d_dyn[:, :, lag] = scale * (_tap(full, positions, lag) * dy).reshape(t, h, head_dim).sum(-1)
    dx = d_full[3:] + scale * dy
    if d_carry_next is not None:
        dx[-3:] += d_carry_next
    return dx, d_dyn.reshape(t, h * 4), d_full[:3].clone()


# ---------------------------------------------------------------------------------------------------------------
# 5 / 6 / 7. attention: NEAR (distance < W, rel bias), FAR (distance >= W, global visits), merge + sink + gate
# ---------------------------------------------------------------------------------------------------------------
class AttnGeom:
    """Query rows (positions, document ids) and key rows (positions, document ids) of one attention call. The
    kernels get the same information as per-segment cu_q / cu_k (bottom-right aligned, the kvprefix convention)."""

    def __init__(self, pos_q, doc_q, pos_k, doc_k, window=1024, extent=1024):
        self.pos_q, self.doc_q, self.pos_k, self.doc_k = pos_q, doc_q, pos_k, doc_k
        self.window, self.extent = window, extent
        self.dist = pos_q[:, None] - pos_k[None, :]  # [Tq, Nk]
        self.same = doc_q[:, None] == doc_k[None, :]

    def mask(self, part: str) -> torch.Tensor:
        if part == "near":
            return self.same & (self.dist >= 0) & (self.dist < self.window)
        assert part == "far"
        return self.same & (self.dist >= self.window)


def _scores(q, k, rel, geo: AttnGeom, scale):
    """[Hq, Tq, Nk] logits = scale * q.k + bias; bias = rel[t, h, d] for 0 <= d < extent, else 0."""
    hq, hk = q.shape[1], k.shape[1]
    kk = k.repeat_interleave(hq // hk, dim=1)
    s = torch.einsum("thd,jhd->htj", q, kk) * scale
    d = geo.dist
    in_e = (d >= 0) & (d < geo.extent)
    idx = d.clamp(0, geo.extent - 1)
    bias = torch.gather(rel.permute(1, 0, 2), 2, idx[None].expand(hq, -1, -1))  # [Hq, Tq, Nk]
    return s + torch.where(in_e[None], bias, torch.zeros((), dtype=s.dtype))


def attn_part_fwd(part, q, k, v, rel, geo: AttnGeom, scale=128 ** -0.5, p_bf16=False):
    """NEAR or FAR part: out [Tq, Hq, d] normalised over the part's keys, lse [Hq, Tq] (natural log); a row without
    keys: out 0, lse -inf. p_bf16 emulates the kernels' bf16 rounding of P before the P V product (the "pt" rule)."""
    hq, hk = q.shape[1], k.shape[1]
    vv = v.repeat_interleave(hq // hk, dim=1)
    m = geo.mask(part)[None]
    s = _scores(q, k, rel, geo, scale).masked_fill(~m, float("-inf"))
    mx = s.amax(-1, keepdim=True)
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    p = torch.exp(s - mx)
    l = p.sum(-1, keepdim=True)
    pv = rb(p) if p_bf16 else p
    o = torch.einsum("htj,jhd->thd", pv, vv) / torch.where(l > 0, l, torch.ones_like(l)).permute(1, 0, 2)
    lse = torch.where(l[..., 0] > 0, mx[..., 0] + torch.log(l[..., 0]), torch.full_like(l[..., 0], float("-inf")))
    return o, lse


def _sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def merge_fwd(o_n, l_n, o_f, l_f, sink, gate, rounding=True):
    """(o, l) = LSE merge of near and far (far None on local visits); lse_s = logaddexp(l, sink);
    out_s = o * exp(l - lse_s) (saved; = the output normalised over keys + the zero-value sink key);
    result = bf16((o * sigmoid(gate)) * sigmoid(l - sink)) [T, Hq * d] -- the engine's order (`_gate_sink_kernel`).
    sink [T, Hq], gate [T, Hq * d]; l / lse [Hq, T]. Returns (result, out_s, lse_s)."""
    t, hq, d = o_n.shape
    if o_f is None:
        o, l = o_n, l_n
    else:
        l = torch.logaddexp(l_n, l_f)
        o = o_n * torch.exp(l_n - l).T[..., None] + o_f * torch.exp(l_f - l).T[..., None]
    sk = sink.T  # [Hq, T]
    lse_s = torch.logaddexp(l, sk)
    out_s = o * torch.exp(l - lse_s).T[..., None]
    res = (o * _sigmoid(gate.reshape(t, hq, d))) * _sigmoid(l - sk).T[..., None]
    res = res.reshape(t, hq * d)
    return (rb(res) if rounding else res), out_s, lse_s


def merge_bwd(d_result, out_s, lse_s, sink, gate, rounding=True):
    """From d_result [T, Hq*d]: dout_s = bf16(d_result * sigmoid(gate)) (the MMA operand of both attention
    backwards), delta = rowsum(dout_s * out_s) [Hq, T], d_sink = -exp(sink - lse_s) * delta [T, Hq],
    d_gate = d_result * out_s * sg (1 - sg) [T, Hq*d] (fp32)."""
    t, hq, d = out_s.shape
    sg = _sigmoid(gate.reshape(t, hq, d))
    dr = d_result.reshape(t, hq, d)
    dout = dr * sg
    if rounding:
        dout = rb(dout)
    delta = (dout * out_s).sum(-1).T  # [Hq, T]
    d_sink = -(torch.exp(sink.T - lse_s) * delta).T
    d_gate = (dr * out_s * sg * (1 - sg)).reshape(t, hq * d)
    return dout, delta, d_sink, d_gate


def attn_part_bwd(part, dout, lse_s, delta, q, k, v, rel, geo: AttnGeom, scale=128 ** -0.5, ds_bf16=False):
    """Union-softmax backward of one part with the FINAL (lse_s, delta): P_j = exp(s_j - lse_s), dP_j = dout . v_j,
    dS_j = P_j (dP_j - delta); dq = scale dS K, dk = scale dS^T Q and dv = P^T dout summed over the 3 q heads of a kv
    head, d_rel[t, h, d] = dS at the pair (t, t - d) for d < extent (one element per pair, no reduction).
    Returns (dq [Tq,Hq,d], dk [Nk,Hkv,d], dv [Nk,Hkv,d], d_rel [Tq,Hq,E])."""
    tq, hq, dh = q.shape
    nk, hk = k.shape[0], k.shape[1]
    g = hq // hk
    kk = k.repeat_interleave(g, dim=1)
    vv = v.repeat_interleave(g, dim=1)
    m = geo.mask(part)[None]
    s = _scores(q, k, rel, geo, scale)
    p = torch.where(m, torch.exp(s - lse_s[..., None]), torch.zeros((), dtype=s.dtype))
    dp = torch.einsum("thd,jhd->htj", dout, vv)
    ds = p * (dp - delta[..., None])
    dsm = rb(ds) if ds_bf16 else ds
    pm = rb(p) if ds_bf16 else p
    dq = torch.einsum("htj,jhd->thd", dsm, kk) * scale
    dk = (torch.einsum("htj,thd->jhd", dsm, q) * scale).reshape(nk, hk, g, dh).sum(2)
    dv = torch.einsum("htj,thd->jhd", pm, dout).reshape(nk, hk, g, dh).sum(2)
    d = geo.dist
    sel = m[0] & (d < geo.extent)
    d_rel = torch.zeros(tq, hq, geo.extent, dtype=ds.dtype)
    ti, ji = torch.nonzero(sel, as_tuple=True)
    d_rel[ti, :, d[ti, ji]] = ds[:, ti, ji].T
    return dq, dk, dv, d_rel


def attention_full_fwd(q, k, v, rel, sink, gate, geo: AttnGeom, is_global, scale=128 ** -0.5, rounding=True,
                       p_bf16=False):
    """near (+ far) + merge: the trainer's composition. Returns (result, saved dict). p_bf16: the "pt" emulation."""
    o_n, l_n = attn_part_fwd("near", q, k, v, rel, geo, scale, p_bf16)
    o_f = l_f = None
    if is_global:
        o_f, l_f = attn_part_fwd("far", q, k, v, rel, geo, scale, p_bf16)
    res, out_s, lse_s = merge_fwd(o_n, l_n, o_f, l_f, sink, gate, rounding)
    return res, dict(out_s=out_s, lse_s=lse_s)


def attention_full_bwd(d_result, saved, q, k, v, rel, sink, gate, geo: AttnGeom, is_global, scale=128 ** -0.5,
                       rounding=True, ds_bf16=False):
    """merge bwd -> near bwd (+ far bwd) in the union form. ds_bf16: the "pt" emulation of the MMA operands."""
    dout, delta, d_sink, d_gate = merge_bwd(d_result, saved["out_s"], saved["lse_s"], sink, gate, rounding)
    dq, dk, dv, d_rel = attn_part_bwd("near", dout, saved["lse_s"], delta, q, k, v, rel, geo, scale, ds_bf16)
    if is_global:
        dq2, dk2, dv2, d_rel2 = attn_part_bwd("far", dout, saved["lse_s"], delta, q, k, v, rel, geo, scale,
                                              ds_bf16)
        dq, dk, dv, d_rel = dq + dq2, dk + dk2, dv + dv2, d_rel + d_rel2
    return dq, dk, dv, d_rel, d_sink, d_gate


def attention_bruteforce(q, k, v, rel, sink, gate, geo: AttnGeom, is_global, scale=128 ** -0.5):
    """One softmax over all visible keys + the zero-value sink logit, times sigmoid(gate): the engine's attention
    (`_rel_attention_kernel` + `_gate_sink_kernel`) written as SDPA with sinks (transformers eager form)."""
    t, hq, dh = q.shape
    vv = v.repeat_interleave(hq // k.shape[1], dim=1)
    vis = geo.mask("near") | (geo.mask("far") if is_global else torch.zeros_like(geo.same))
    s = _scores(q, k, rel, geo, scale).masked_fill(~vis[None], float("-inf"))
    s = torch.cat((s, sink.T[..., None]), -1)
    p = torch.softmax(s, -1)[..., :-1]
    o = torch.einsum("htj,jhd->thd", p, vv)
    return (o * _sigmoid(gate.reshape(t, hq, dh))).reshape(t, hq * dh)


# ---------------------------------------------------------------------------------------------------------------
# 8. block AttnRes mix
# ---------------------------------------------------------------------------------------------------------------
def attn_res_fwd(srcs, queries, norm_w=None, eps=1e-6, norm_eps=1e-6, rounding=True):
    """srcs: list of N [T, D] sources ([*stack, last (+ addend, summed by the caller in fp32)]); queries [Q, D]
    (pre-scaled by D^-0.5). r_n = rsqrt(mean(x_n^2) + eps); logit = (q . x_n) r_n; p = softmax_n;
    mix_q = sum_n p r_n x_n; with norm_w [Q, D]: out = bf16(mix * rsqrt(mean(mix^2) + norm_eps) * w), else the fp32
    mixes. Returns (out [Q, T, D], saved = (p [Q, N, T], r [N, T], ro [Q, T] or None))."""
    x = torch.stack(srcs)
    d = x.shape[-1]
    r = torch.rsqrt((x * x).mean(-1) + eps)
    a = torch.einsum("qd,ntd->qnt", queries, x)
    p = torch.softmax(a * r[None], dim=1)
    mix = torch.einsum("qnt,ntd->qtd", p * r[None], x)
    ro = None
    out = mix
    if norm_w is not None:
        ro = torch.rsqrt((mix * mix).mean(-1) + norm_eps)
        out = mix * ro[..., None] * norm_w[:, None, :]
        if rounding:
            out = rb(out)
    return out, (p, r, ro)


def attn_res_bwd(d_out, srcs, queries, norm_w, saved):
    """d_src [N, T, D] (the kernels ACCUMULATE these into the fp32 gradient slots of the block states).
    dmix (norm: ro g - ro^3/D (mix . g) mix, g = d_out * w); u = dmix . x_n; dp = r u; dlogit = p (dp - sum p dp);
    dr = sum_q (p u + dlogit (q . x_n)); dx_n = sum_q (p r dmix + dlogit r q) - dr r^3 / D x_n."""
    p, r, ro = saved
    x = torch.stack(srcs)
    d = x.shape[-1]
    mix = torch.einsum("qnt,ntd->qtd", p * r[None], x)
    if norm_w is not None:
        g = d_out * norm_w[:, None, :]
        dmix = ro[..., None] * g - (ro ** 3 / d)[..., None] * (mix * g).sum(-1, keepdim=True) * mix
    else:
        dmix = d_out
    u = torch.einsum("qtd,ntd->qnt", dmix, x)
    dp = r[None] * u
    dlogit = p * (dp - (p * dp).sum(1, keepdim=True))
    a = torch.einsum("qd,ntd->qnt", queries, x)
    dr = (p * u + dlogit * a).sum(0)
    dx = (torch.einsum("qnt,qtd->ntd", p * r[None], dmix) + torch.einsum("qnt,qd->ntd", dlogit * r[None], queries)
          - (dr * r ** 3 / d)[..., None] * x)
    return dx


# ---------------------------------------------------------------------------------------------------------------
# 9. loop message controller (pair RMS -> rank-64 down -> gain / forget -> blend -> RMSNorm)
# ---------------------------------------------------------------------------------------------------------------
def loop_blend_fwd(dest, msg, down, gf_w, norm_w, eps=1e-6, rounding=True):
    """dest [S|1, T, D], msg [S, T, D] (fp32), down [S, R, 2D], gf_w [S, 2D, R] (rows: gain then forget),
    norm_w [S|1, D]. pair = bf16([dest rd, msg rm]); h = bf16(pair @ down^T); gf = h @ gf_w^T (fp32);
    x = (1 - tanh(forget)) dest + tanh(gain) msg; out = bf16(x rsqrt(mean(x^2) + eps) w).
    Returns (out [S, T, D], saved dict)."""
    s, t, d = msg.shape
    dest = dest.expand(s, t, d)
    w = norm_w.reshape(-1, 1, d).expand(s, 1, d)
    rd = torch.rsqrt((dest * dest).mean(-1, keepdim=True) + 1e-6)
    rm = torch.rsqrt((msg * msg).mean(-1, keepdim=True) + 1e-6)
    pair = torch.cat((dest * rd, msg * rm), -1)
    if rounding:
        pair = rb(pair)
    h = pair @ down.transpose(1, 2)
    if rounding:
        h = rb(h)
    gf = h @ gf_w.transpose(1, 2)
    gain, forget = torch.tanh(gf[..., :d]), torch.tanh(gf[..., d:])
    x = (1.0 - forget) * dest + gain * msg
    rx = torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)
    out = x * rx * w
    return (rb(out) if rounding else out), dict(rd=rd, rm=rm, h=h, gain=gain, forget=forget, x=x, rx=rx)


def loop_blend_bwd(d_out, dest, msg, down, gf_w, norm_w, saved, dest_broadcast: bool):
    """d_dest [S|1, T, D] (summed over streams when dest broadcasts), d_msg [S, T, D]; weights frozen."""
    s, t, d = msg.shape
    dst = dest.expand(s, t, d)
    w = norm_w.reshape(-1, 1, d).expand(s, 1, d)
    x, rx, gain, forget, rd, rm = (saved[k] for k in ("x", "rx", "gain", "forget", "rd", "rm"))
    gw = d_out * w
    dx = rx * gw - (rx ** 3 / d) * (x * gw).sum(-1, keepdim=True) * x
    d_dest = (1.0 - forget) * dx
    d_msg = gain * dx
    d_gf = torch.cat((dx * msg * (1 - gain * gain), -dx * dst * (1 - forget * forget)), -1)
    dh = d_gf @ gf_w
    d_pair = dh @ down
    dpd, dpm = d_pair[..., :d], d_pair[..., d:]
    d_dest = d_dest + rd * dpd - (rd ** 3 / d) * (dst * dpd).sum(-1, keepdim=True) * dst
    d_msg = d_msg + rm * dpm - (rm ** 3 / d) * (msg * dpm).sum(-1, keepdim=True) * msg
    if dest_broadcast:
        d_dest = d_dest.sum(0, keepdim=True)
    return d_dest, d_msg


# ---------------------------------------------------------------------------------------------------------------
# 10. softcapped SwiGLU, routing weight before down
# ---------------------------------------------------------------------------------------------------------------
def softcap_swiglu_fwd(gate_up, gamma=None, gate_cap=4.0, up_cap=25.0, rounding=True):
    """gate_up [R, 2I] INTERLEAVED (col 2j gate j, 2j+1 up j); y = bf16(((4 tanh(g/4) sigmoid(g)) 25 tanh(u/25)) gamma)."""
    g, u = gate_up[:, 0::2], gate_up[:, 1::2]
    a = gate_cap * torch.tanh(g / gate_cap)
    y = (a * _sigmoid(g)) * (up_cap * torch.tanh(u / up_cap))
    if gamma is not None:
        y = y * gamma[:, None]
    return rb(y) if rounding else y


def softcap_swiglu_bwd(dy, gate_up, gamma=None, gate_cap=4.0, up_cap=25.0):
    """d_gate_up [R, 2I] interleaved (bf16 in the kernel), d_gamma [R] (fp32, to moe_topk's backward)."""
    g, u = gate_up[:, 0::2], gate_up[:, 1::2]
    tg, tu = torch.tanh(g / gate_cap), torch.tanh(u / up_cap)
    a, sg, b = gate_cap * tg, _sigmoid(g), up_cap * tu
    gm = gamma[:, None] if gamma is not None else torch.ones((), dtype=dy.dtype)
    dg = dy * gm * b * ((1 - tg * tg) * sg + a * sg * (1 - sg))
    du = dy * gm * a * sg * (1 - tu * tu)
    d = torch.stack((dg, du), -1).reshape(gate_up.shape)
    d_gamma = (dy * a * sg * b).sum(-1) if gamma is not None else None
    return d, d_gamma


# ---------------------------------------------------------------------------------------------------------------
# autograd wrappers: forward without rounding, backward = the hand-written functions above (gradcheck targets)
# ---------------------------------------------------------------------------------------------------------------
class QkvNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, tau, th, mode):
        ctx.save_for_backward(x)
        ctx.args = (scale, tau, th, mode)
        return qkv_norm_fwd(x, scale, tau, th, 1e-6, mode, rounding=False)

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        scale, tau, th, mode = ctx.args
        return qkv_norm_bwd(dy, x, scale, tau, th, 1e-6, mode), None, None, None, None


class ShortConvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, carry, addend, residual, w, positions):
        ctx.save_for_backward(w, positions)
        out, _ = short_conv_fwd(x, w, positions, carry, addend, residual, rounding=False)
        return out

    @staticmethod
    def backward(ctx, dout):
        w, positions = ctx.saved_tensors
        dx, dc, da, dr = short_conv_bwd(dout, w, positions, None, has_addend=True, has_residual=True)
        return dx, dc, da, dr, None, None


class DynamicConvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dyn, carry, static_w, positions, head_dim):
        ctx.save_for_backward(x, dyn, carry, static_w, positions)
        ctx.head_dim = head_dim
        y, _ = dynamic_conv_fwd(x, static_w, dyn, positions, carry, head_dim=head_dim, rounding=False)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, dyn, carry, static_w, positions = ctx.saved_tensors
        dx, d_dyn, dc = dynamic_conv_bwd(dy, x, static_w, dyn, positions, carry, head_dim=ctx.head_dim)
        return dx, d_dyn, dc, None, None, None


class AttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, rel, sink, gate, geo, is_global, scale):
        res, saved = attention_full_fwd(q, k, v, rel, sink, gate, geo, is_global, scale, rounding=False)
        ctx.save_for_backward(q, k, v, rel, sink, gate, saved["out_s"], saved["lse_s"])
        ctx.meta = (geo, is_global, scale)
        return res

    @staticmethod
    def backward(ctx, d_res):
        q, k, v, rel, sink, gate, out_s, lse_s = ctx.saved_tensors
        geo, is_global, scale = ctx.meta
        g = attention_full_bwd(d_res, dict(out_s=out_s, lse_s=lse_s), q, k, v, rel, sink, gate, geo, is_global,
                               scale, rounding=False)
        return (*g, None, None, None)


class AttnResFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, queries, norm_w, *srcs):
        out, saved = attn_res_fwd(list(srcs), queries, norm_w, rounding=False)
        ctx.save_for_backward(queries, *srcs)
        ctx.norm_w, ctx.saved = norm_w, saved
        return out

    @staticmethod
    def backward(ctx, d_out):
        queries, *srcs = ctx.saved_tensors
        dx = attn_res_bwd(d_out, srcs, queries, ctx.norm_w, ctx.saved)
        return (None, None, *dx.unbind(0))


class LoopBlendFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dest, msg, down, gf_w, norm_w):
        out, saved = loop_blend_fwd(dest, msg, down, gf_w, norm_w, rounding=False)
        ctx.save_for_backward(dest, msg, down, gf_w, norm_w)
        ctx.saved = saved
        return out

    @staticmethod
    def backward(ctx, d_out):
        dest, msg, down, gf_w, norm_w = ctx.saved_tensors
        dd, dm = loop_blend_bwd(d_out, dest, msg, down, gf_w, norm_w, ctx.saved, dest.shape[0] == 1 < msg.shape[0])
        return dd, dm, None, None, None


class SwigluFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_up, gamma):
        ctx.save_for_backward(gate_up, gamma)
        return softcap_swiglu_fwd(gate_up, gamma, rounding=False)

    @staticmethod
    def backward(ctx, dy):
        gate_up, gamma = ctx.saved_tensors
        d, dg = softcap_swiglu_bwd(dy, gate_up, gamma)
        return d, dg


# ---------------------------------------------------------------------------------------------------------------
# 5-7 bis. the PER-PART (standard chain rule) backward: what g1's 3-op graph (35ea4be4 blocks/inkling2.py:208-224)
# needs with standard autodiff. The same kernel serves both forms: it takes (dout operand, lse, delta):
#   union form:    dout = dout_s (shared), lse = lse_s, delta = rowsum(dout_s * out_s)
#   per-part form: dout = d_out_p,          lse = lse_p, delta = rowsum(d_out_p * out_p) - d_lse_p
# ---------------------------------------------------------------------------------------------------------------
def merge_bwd_std(d_result, o_n, l_n, o_f, l_f, sink, gate):
    """Standard backward of merge_fwd (no rounding): returns d_o_n, d_l_n, d_o_f, d_l_f, d_sink, d_gate.
    w_p = exp(l_p - l), a = sigmoid(l - sink); d_o = dres sg a; d_a = sum dres o sg; d_l_tot = d_a a (1 - a)
    - sum_p (d_o . o_p) w_p; d_l_p = w_p ((d_o . o_p) + d_l_tot)."""
    t, hq, d = o_n.shape
    dr = d_result.reshape(t, hq, d)
    sg = _sigmoid(gate.reshape(t, hq, d))
    if o_f is None:
        l, o = l_n, o_n
        parts = [(o_n, l_n)]
    else:
        l = torch.logaddexp(l_n, l_f)
        o = o_n * torch.exp(l_n - l).T[..., None] + o_f * torch.exp(l_f - l).T[..., None]
        parts = [(o_n, l_n), (o_f, l_f)]
    a = _sigmoid(l - sink.T).T  # [T, Hq]
    d_o = dr * sg * a[..., None]
    d_a = (dr * o * sg).sum(-1)
    d_gate = (dr * o * a[..., None] * sg * (1 - sg)).reshape(t, hq * d)
    d_x = d_a * a * (1 - a)  # gradient of (l - sink)
    d_sink = -d_x
    d_l_tot = d_x.clone()
    dw = []
    for o_p, l_p in parts:
        w_p = torch.exp(l_p - l).T
        g = (d_o * o_p).sum(-1)
        dw.append((w_p, g))
        d_l_tot = d_l_tot - g * w_p
    outs = []
    for (o_p, l_p), (w_p, g) in zip(parts, dw):
        outs += [d_o * w_p[..., None], (w_p * (g + d_l_tot)).T]
    if o_f is None:
        outs += [None, None]
    return (*outs, d_sink, d_gate)


def attn_part_bwd_std(part, d_out, d_lse, out_p, lse_p, q, k, v, rel, geo: AttnGeom, scale=128 ** -0.5):
    """FA backward of one part with its own (out_p, lse_p) and incoming (d_out, d_lse): the union kernel with
    delta = rowsum(d_out * out_p) - d_lse."""
    delta = (d_out * out_p).sum(-1).T - d_lse
    return attn_part_bwd(part, d_out, lse_p, delta, q, k, v, rel, geo, scale)


class AttnPartFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, rel, part, geo, scale):
        o, l = attn_part_fwd(part, q, k, v, rel, geo, scale)
        ctx.save_for_backward(q, k, v, rel, o, l)
        ctx.meta = (part, geo, scale)
        return o, l

    @staticmethod
    def backward(ctx, d_o, d_l):
        q, k, v, rel, o, l = ctx.saved_tensors
        part, geo, scale = ctx.meta
        dq, dk, dv, d_rel = attn_part_bwd_std(part, d_o, d_l, o, l, q, k, v, rel, geo, scale)
        return dq, dk, dv, d_rel, None, None, None


class MergeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, o_n, l_n, o_f, l_f, sink, gate):
        res, _, _ = merge_fwd(o_n, l_n, o_f, l_f, sink, gate, rounding=False)
        ctx.save_for_backward(o_n, l_n, o_f, l_f, sink, gate)
        return res

    @staticmethod
    def backward(ctx, d_res):
        o_n, l_n, o_f, l_f, sink, gate = ctx.saved_tensors
        return merge_bwd_std(d_res, o_n, l_n, o_f, l_f, sink, gate)


# ---------------------------------------------------------------------------------------------------------------
# 11. MoE combine (unit weights; g1 B.4) and the SiTU's gamma lookup through the scatter indices
# ---------------------------------------------------------------------------------------------------------------
def moe_combine_fwd(rows, scatter, top_k, rounding=True):
    """out[t] = bf16(sum_{s=0..k-1} rows[scatter[t*k + s]]) -- fp32 sum in slot (routing key) order, one rounding
    (`inkling_moe`). rows [R, D] in expert-sorted order; scatter [T*k] = row of assignment (t, s)."""
    t = scatter.shape[0] // top_k
    idx = scatter.reshape(t, top_k)
    acc = torch.zeros(t, rows.shape[1], dtype=rows.dtype)
    for s in range(top_k):
        acc = acc + rows[idx[:, s]]
    return rb(acc) if rounding else acc


def moe_combine_bwd(d_out, scatter, top_k, num_rows):
    """d_rows[scatter[t*k + s]] = d_out[t] (bf16 in the kernel: the grouped down-GEMM backward's input)."""
    d_rows = torch.zeros(num_rows, d_out.shape[1], dtype=d_out.dtype)
    d_rows[scatter] = d_out.repeat_interleave(top_k, dim=0)
    return d_rows


def swiglu_routed_fwd(gate_up, weights, scatter, gate_cap=4.0, up_cap=25.0, rounding=True):
    """g1 B.4 form: gamma of expert-sorted row scatter[a] = weights.flatten()[a] (a = t*k + s)."""
    gamma = torch.empty(gate_up.shape[0], dtype=gate_up.dtype)
    gamma[scatter] = weights.reshape(-1)
    return softcap_swiglu_fwd(gate_up, gamma, gate_cap, up_cap, rounding)


def swiglu_routed_bwd(dy, gate_up, weights, scatter, gate_cap=4.0, up_cap=25.0):
    """d_gate_up [R, 2I], d_weights [T, k] (d_weights[a] = d_gamma[scatter[a]], to moe_topk's backward)."""
    gamma = torch.empty(gate_up.shape[0], dtype=gate_up.dtype)
    gamma[scatter] = weights.reshape(-1)
    d, d_gamma = softcap_swiglu_bwd(dy, gate_up, gamma, gate_cap, up_cap)
    return d, d_gamma[scatter].reshape(weights.shape)


class MoeCombineFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, scatter, top_k):
        ctx.save_for_backward(scatter)
        ctx.meta = (top_k, rows.shape[0])
        return moe_combine_fwd(rows, scatter, top_k, rounding=False)

    @staticmethod
    def backward(ctx, d_out):
        (scatter,) = ctx.saved_tensors
        top_k, n = ctx.meta
        return moe_combine_bwd(d_out, scatter, top_k, n), None, None


class SwigluRoutedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_up, weights, scatter):
        ctx.save_for_backward(gate_up, weights, scatter)
        return swiglu_routed_fwd(gate_up, weights, scatter, rounding=False)

    @staticmethod
    def backward(ctx, dy):
        gate_up, weights, scatter = ctx.saved_tensors
        d, dw = swiglu_routed_bwd(dy, gate_up, weights, scatter)
        return d, dw, None
