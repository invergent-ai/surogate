"""The LEVEL-0 backward of the fast route.

Forward = the engine's (ift_forward.forward: bitwise the engine's prefill, routing included). Backward, visits last to
first: visit v is RECOMPUTED in torch from the engine's own visit inputs (the AttnRes stack slots it read, saved as the
outputs of earlier visits) with the validated reference math (ift_ref = inkling_ref_ad.py, vendored; math unchanged):
W4A16 / W8A16 straight-through the activation and KV quantisers (rb / r8 are STE), routing REPLAYED from the engine's
own ids, the expert LoRA from the fp32 masters; its vjp gives the gradient of the visit inputs and of the layer's LoRA.
The weights are the ENGINE's (views of the loaded artifact, no copy); expert units come from the engine's exact decode.

Autograd graph (built after the engine forward, `TrainForward.build`):
  x0 (leaf, no grad) -> EngineVisit(v) for v = 0..47 (inputs: the slots it reads + its layer's LoRA tensors; output: the
  engine's visit output) -> FinalMix (engine hidden; backward: torch attn_res + final norm) -> HeadLogprob (engine
  log-probs; backward: the torch log-softmax gradient of the softcapped fp32 head, vocabulary-chunked).
"""

from __future__ import annotations

import math
import struct
from types import SimpleNamespace

import torch

import ift_ref as RA
from ift_engine import InklingEngine, LoraBank, MoeAdapters, device_view
import ift_forward as F

D, I, E, TOPK = 2304, 1152, 128, 8
RANK, ALPHA = 16, 32.0
EPS = 1e-6
VOCAB = 200058
GLOBAL_LAYERS = (5, 11, 17, 23)
BF16, F32 = torch.bfloat16, torch.float32
import os as _os
import time as _time

PROFILE = {"on": _os.environ.get("IFT_PROFILE", "0") == "1", "t": {}}

# The vendored reference's fma runs through fp64 (an exactly rounded fused op on the CPU reference); its autograd keeps
# fp64 copies of both operands per source (2 x 8 B x Q x T x D per AttnRes source: ~3 GB per visit at 3k tokens).
# The recompute only feeds the gradient, so an fp32 addcmul (one rounding off at most) is used instead.
RA.fma = lambda a, b, c: torch.addcmul(c.float(), a.float(), b.float())


class _tick:
    """IFT_PROFILE=1: accumulated wall time per named section (synchronising; profiling runs only)."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if PROFILE["on"]:
            torch.cuda.synchronize()
            self.t0 = _time.time()

    def __exit__(self, *a):
        if PROFILE["on"]:
            torch.cuda.synchronize()
            PROFILE["t"][self.name] = PROFILE["t"].get(self.name, 0.0) + _time.time() - self.t0


# ---------------------------------------------------------------------------------------------------------------
# the engine's weights in the reference's conventions (views; its functions upcast on use)
# ---------------------------------------------------------------------------------------------------------------
class EngineExperts:
    """rig.PackedExperts over the engine's banks: dequant(name, e0, e1) -> exact bf16 units [g, 2304, K] (the engine's
    decode); g13 / g2 = per-expert amax, den13 / den2 = outer_scale_denominator (NVFP4; factor = amax / den)."""

    def __init__(self, eng: InklingEngine, layer: int, gate_w: torch.Tensor, gate_b: torch.Tensor):
        self.eng, self.layer = eng, layer
        d13 = eng.tensor(f"L{layer}.w13.codes")
        self.fmt = "mxfp8" if d13["extra"] == 1 else "nvfp4"
        self.gate_w, self.gate_b = gate_w, gate_b
        self.g13 = self.g2 = self.den13 = self.den2 = None
        if self.fmt == "nvfp4":
            for name in ("w13", "w2"):
                da = eng.tensor(f"L{layer}.{name}.amax")
                amax = device_view(da["ptr"], (E,), F32, eng.device)
                den = struct.unpack("<f", struct.pack("<i", da["den_bits"]))[0]
                setattr(self, "g" + name[1:], amax)
                setattr(self, "den" + name[1:], torch.tensor([den], dtype=F32, device=eng.dev))

    def dequant(self, name: str, e0: int, e1: int) -> torch.Tensor:
        unit, _ = self.eng.decode_experts(self.layer, name, e0, e1 - e0)
        return unit


class EngineWeights:
    def __init__(self, eng: InklingEngine):
        if eng.c.get("fp8_attention") == 1:
            raise NotImplementedError("LEVEL-0 backward: FP8 attention weights not wired yet (BF16 is the default)")
        self.eng = eng
        dev = eng.device

        def v(name, shape, dtype=BF16):
            t = eng.tensor(name)
            n = math.prod(shape)
            if t["numel"] != n:
                raise RuntimeError(f"{name}: engine numel {t['numel']} != {n} for {shape} (ne {t['ne']})")
            return device_view(t["ptr"], shape, dtype, dev)

        self.layers = []
        for i in range(24):
            L = SimpleNamespace(idx=i, is_local=i not in GLOBAL_LAYERS, core=6 <= i < 18)
            Q = 1 if L.is_local else 3
            L.attn_norm = v(f"L{i}.attn_input_norm", (Q, D))
            qkvr = v(f"L{i}.qkvr", (8600, D))
            L.wq, L.wr, L.wg, L.ws = qkvr[0:3072], qkvr[3072:3456], qkvr[3456:6528], qkvr[6528:6552]
            L.wk, L.wv = qkvr[6552:7576], qkvr[7576:8600]
            L.wo = v(f"L{i}.output", (D, 3072))
            L.q_scale = v(f"L{i}.q_scale", (24, 128))
            L.r_scale = v(f"L{i}.r_scale", (24, 16))
            L.rel_proj = v(f"L{i}.rel_proj", (16, 1024))
            if L.is_local:
                L.k_conv = v(f"L{i}.k_conv", (1024, 4))
                L.v_conv = v(f"L{i}.v_conv", (1024, 4))
            else:
                L.k_static = v(f"L{i}.k_conv_static", (4, 1024))
                L.v_static = v(f"L{i}.v_conv_static", (4, 1024))
                L.k_wproj = v(f"L{i}.k_conv_proj", (32, D))
                L.v_wproj = v(f"L{i}.v_conv_proj", (32, D))
            L.attn_sconv = v(f"L{i}.attn_out_conv", (D, 4))
            L.mlp_sconv = v(f"L{i}.mlp_out_conv", (D, 4))
            L.mlp_norm = v(f"L{i}.moe_input_norm", (1, D))
            L.gate_w = v(f"L{i}.router", (E, D))
            L.gate_b = v(f"L{i}.router_bias", (E,), F32)
            if L.core:
                L.ctrl_down = v(f"L{i}.loop_down", (4, 64, 2 * D))
                L.ctrl_gf = v(f"L{i}.loop_gain_forget", (4, 2 * D, 64))
                L.q_msg = v(f"L{i}.query.message", (4, D), F32)
            L.q_attn = v(f"L{i}.query.attn", (Q, D), F32) if i > 0 else None
            L.q_mlp = v(f"L{i}.query.mlp", (1, D), F32)
            L.experts = EngineExperts(eng, i, L.gate_w, L.gate_b)
            self.layers.append(L)
        self.q_out = v("query.output", (1, D), F32)
        self.final_norm = v("final_norm", (D,))
        self.head = v("output_head", (eng.c["output_rows"], D))[:VOCAB]


# ---------------------------------------------------------------------------------------------------------------
# MoE recompute: the reference's rig.moe_vec (vectorised over expert groups) with W4A16 and replayed routing
# ---------------------------------------------------------------------------------------------------------------
EG = 16


def _probe_out_dtype(dev) -> bool:
    try:
        a = torch.ones(1, 32, 32, dtype=BF16, device=dev)
        y = torch.bmm(a, a, out_dtype=F32)
        return y.dtype == F32 and float(y[0, 0, 0]) == 32.0
    except Exception:
        return False


_OUT_DTYPE = {}


def _bmm_f32(a, w):
    dev = a.device
    if dev not in _OUT_DTYPE:
        _OUT_DTYPE[dev] = _probe_out_dtype(dev)
    if _OUT_DTYPE[dev]:
        return torch.bmm(a.to(BF16), w.transpose(1, 2), out_dtype=F32)
    return torch.bmm(a.float(), w.float().transpose(1, 2))


class ExpertBMM(torch.autograd.Function):
    """y = a @ W^T for a group of frozen experts: W decoded in the forward and AGAIN in the backward (never saved)."""

    @staticmethod
    def forward(ctx, a, w, name, g0, g1):
        ctx.w, ctx.name, ctx.g0, ctx.g1 = w, name, g0, g1
        return _bmm_f32(a, w.dequant(name, g0, g1))

    @staticmethod
    def backward(ctx, g):
        W = ctx.w.dequant(ctx.name, ctx.g0, ctx.g1)
        if _OUT_DTYPE.get(g.device):
            ga = torch.bmm(g.to(BF16), W, out_dtype=F32)
        else:
            ga = torch.bmm(g.float(), W.float())
        return ga, None, None, None, None


def bmm32(a, w, name, g0, g1):
    if torch.is_grad_enabled() and a.requires_grad:
        return ExpertBMM.apply(a, w, name, g0, g1)
    return _bmm_f32(a, w.dequant(name, g0, g1))


# The gradient through the router's gate weights (the normalised sigmoids of the replayed experts) into the MLP input. True: the exact
# backward. False: the gate weights are constants of the backward. backward_row sets it per visit (gate_grad_from); nothing else writes it.
GATE_GRAD = {"on": True}


def moe_vec(x, w, ids, lora=None):
    """rig.moe_vec with act_quant=False and routing=(ids, None): gammas from this recompute's router (gradient flows
    into x through them), W4A16 GEMMs (NVFP4 global applied to the fp32 output), LoRA from the fp32 masters
    ({"A13" [E,r,H], "B13" [E,2I,r], "A2" [E,r,I], "B2" [E,H,r], "scale"}), one bf16 rounding per GEMM output."""
    x = x.float()
    t, h = x.shape
    ids = ids.long()
    lg = RA._router_logits(x, w.gate_w)
    gam = RA.route_weights(lg if GATE_GRAD["on"] else lg.detach(), ids)
    topk = ids.shape[1]
    nv = w.fmt == "nvfp4"
    flat = ids.reshape(-1)
    n = flat.shape[0]
    order = torch.argsort(flat, stable=True)
    es = flat[order]
    tok = order // topk
    slot = order % topk
    counts = torch.bincount(flat, minlength=E)
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.arange(n) - starts[es]
    cnt = counts.tolist()
    st = starts.tolist()
    gam_r = gam[tok, slot]
    dn_parts = []
    for g0 in range(0, E, EG):
        g1 = min(E, g0 + EG)
        r0, r1 = st[g0], st[g1 - 1] + cnt[g1 - 1]
        if r1 == r0:
            continue
        cap = max(cnt[g0:g1])
        eg = es[r0:r1] - g0
        pg = pos[r0:r1]
        tg = tok[r0:r1]
        ng = g1 - g0

        def pad(rows, width):
            z = torch.zeros(ng, cap, width, dtype=rows.dtype)
            return z.index_put((eg, pg), rows)

        gu = bmm32(pad(x[tg], h), w, "w13", g0, g1)[eg, pg]
        if nv:
            gu = gu * RA.nvfp4_global(w.g13[g0:g1][eg], w.den13)[:, None]
        if lora is not None:
            la = torch.bmm(pad(x[tg], h), lora["A13"][g0:g1].transpose(1, 2))
            gu = gu + lora["scale"] * torch.bmm(la, lora["B13"][g0:g1].transpose(1, 2))[eg, pg]
        gu = RA.rb(gu)
        y = RA.inkling_softcap_swiglu(gu, gam_r[r0:r1])
        dn = bmm32(pad(y, y.shape[1]), w, "w2", g0, g1)[eg, pg]
        if nv:
            dn = dn * RA.nvfp4_global(w.g2[g0:g1][eg], w.den2)[:, None]
        if lora is not None:
            la = torch.bmm(pad(y, y.shape[1]), lora["A2"][g0:g1].transpose(1, 2))
            dn = dn + lora["scale"] * torch.bmm(la, lora["B2"][g0:g1].transpose(1, 2))[eg, pg]
        dn_parts.append(RA.rb(dn))
    dn_sorted = torch.cat(dn_parts)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(n)
    down = dn_sorted[inv].reshape(t, topk, h)
    acc = torch.zeros(t, h)
    for s in range(topk):
        acc = acc + down[:, s]
    return RA.rb(acc)


# ---------------------------------------------------------------------------------------------------------------
# one visit in torch, cut exactly as the engine's forward.cpp::run_visit (inputs at the start, partial at the end)
# ---------------------------------------------------------------------------------------------------------------
def _rel_scores(q, k, rel, b0, b1, k0, window, scale):
    """Scores of query rows [b0, b1) against keys [k0, b1) as the reference's inkling_rel_attention forms them:
    [Hk, G, nq, K] fp32 (masked -inf), plus the distance index and the bias-validity mask for d_rel."""
    nq = b1 - b0
    hq, dh = q.shape[1], q.shape[2]
    hk = k.shape[1]
    g = hq // hk
    qq = q[b0:b1].reshape(nq, hk, g, dh).permute(1, 2, 0, 3)  # [Hk, G, nq, D]
    kk = k[k0:b1].permute(1, 0, 2)  # [Hk, K, D]
    s = torch.einsum("hgbd,hkd->hgbk", qq, kk) * scale
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


class RelAttention(torch.autograd.Function):
    """the reference's inkling_rel_attention (one segment, positions 0..T-1; exact fp32 softmax) with a flash-style backward:
    the forward keeps (q, k, v, rel, out, lse) only, the backward recomputes each query block's probabilities from lse.
    Gradients for q, k, v and rel (the dense relative logits) incl. the lse path (the sink gate reads lse)."""

    @staticmethod
    def forward(ctx, q, k, v, rel, window, q_block=128):
        T, hq, dh = q.shape
        hk = k.shape[1]
        g = hq // hk
        scale = 128 ** -0.5
        out = torch.empty(T, hq, dh, dtype=F32, device=q.device)
        lse = torch.empty(T, hq, dtype=F32, device=q.device)
        vv = v.permute(1, 0, 2)
        for b0 in range(0, T, q_block):
            b1 = min(T, b0 + q_block)
            k0 = max(0, b0 - window + 1) if window > 0 else 0
            sc, _, _ = _rel_scores(q, k, rel, b0, b1, k0, window, scale)
            m = sc.amax(-1, keepdim=True)
            p = RA.t_exp(sc - m)
            l = p.sum(-1, keepdim=True)
            o = torch.einsum("hgbk,hkd->hgbd", p, vv[:, k0:b1]) / l
            out[b0:b1] = o.permute(2, 0, 1, 3).reshape(b1 - b0, hq, dh)
            lse[b0:b1] = (m + RA.t_log(l))[..., 0].permute(2, 0, 1).reshape(b1 - b0, hq)
        ctx.save_for_backward(q, k, v, rel, out, lse)
        ctx.window, ctx.q_block, ctx.scale = window, q_block, scale
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, rel, out, lse = ctx.saved_tensors
        window, qb, scale = ctx.window, ctx.q_block, ctx.scale
        T, hq, dh = q.shape
        hk = k.shape[1]
        g = hq // hk
        if dout is None:
            dout = torch.zeros_like(out)
        if dlse is None:
            dlse = torch.zeros_like(lse)
        dq = torch.zeros_like(q)
        dk = torch.zeros(k.shape, dtype=F32, device=k.device)
        dv = torch.zeros(v.shape, dtype=F32, device=v.device)
        drel = torch.zeros(rel.shape, dtype=F32, device=rel.device)
        vv = v.permute(1, 0, 2)
        kk = k.permute(1, 0, 2)
        for b0 in range(0, T, qb):
            b1 = min(T, b0 + qb)
            nq = b1 - b0
            k0 = max(0, b0 - window + 1) if window > 0 else 0
            sc, gi, valid = _rel_scores(q, k, rel, b0, b1, k0, window, scale)
            lb = lse[b0:b1].reshape(nq, hk, g).permute(1, 2, 0)[..., None]  # [Hk, G, nq, 1]
            p = RA.t_exp(sc - lb)
            do = dout[b0:b1].reshape(nq, hk, g, dh).permute(1, 2, 0, 3)  # [Hk, G, nq, D]
            ob = out[b0:b1].reshape(nq, hk, g, dh).permute(1, 2, 0, 3)
            dlb = dlse[b0:b1].reshape(nq, hk, g).permute(1, 2, 0)[..., None]
            dp = torch.einsum("hgbd,hkd->hgbk", do, vv[:, k0:b1])
            dterm = (do * ob).sum(-1, keepdim=True)
            ds = p * (dp - dterm + dlb)  # [Hk, G, nq, K]
            qq = q[b0:b1].reshape(nq, hk, g, dh).permute(1, 2, 0, 3)
            dq[b0:b1] += (torch.einsum("hgbk,hkd->hgbd", ds, kk[:, k0:b1]) * scale).permute(2, 0, 1, 3).reshape(nq, hq, dh)
            dk[k0:b1] += (torch.einsum("hgbk,hgbd->hkd", ds, qq) * scale).permute(1, 0, 2)
            dv[k0:b1] += torch.einsum("hgbk,hgbd->hkd", p, do).permute(1, 0, 2)
            dsr = (ds.reshape(hq, nq, -1) * valid[None]).float()
            drel_blk = torch.zeros(hq, nq, rel.shape[2], dtype=F32, device=rel.device)
            drel_blk.scatter_add_(2, gi[None].expand(hq, -1, -1), dsr)
            drel[b0:b1] += drel_blk.permute(1, 0, 2)
        return dq, dk, dv, drel, None, None


def attention(L, attn_in, pos, tau):
    """the reference's InklingRef._attention for one fresh visit (one chunk, no carried state): [T, 3072] gated output."""
    if L.is_local and attn_in.shape[0] == 1:
        xq = xk = xv = attn_in[0]
    else:
        xq, xk, xv = attn_in[0], attn_in[1], attn_in[2]
    t = xq.shape[0]
    q = RA.inkling_attn_linear(xq, L.wq).reshape(t, 24, 128)
    r = RA.inkling_attn_linear(xq, L.wr).reshape(t, 24, 16)
    gate = RA.inkling_attn_linear(xq, L.wg)
    sink = RA.inkling_attn_linear(xq, L.ws)
    k = RA.inkling_attn_linear(xk, L.wk)
    v = RA.inkling_attn_linear(xv, L.wv)
    if L.is_local:
        kc, vc = L.k_conv, L.v_conv
    else:
        kc = (L.k_static, RA.rb(RA.mm(xk, L.k_wproj)))
        vc = (L.v_static, RA.rb(RA.mm(xv, L.v_wproj)))
    pro = RA.inkling_attention_prologue(q, r, k, v, L.q_scale, L.r_scale, tau, L.rel_proj, kc, vc, None, None, True)
    window = 1024 if L.is_local else 0
    out, lse = RelAttention.apply(pro.q, pro.k.float(), pro.v.float(), pro.rel.float(), window)
    return RA.inkling_gate_sink(out, lse, gate, sink)


def visit_torch(W: EngineWeights, vi, stack, partial, bank, ids, lora):
    """New partial [T, D] (fp32, bf16 values) of visit vi from its inputs (bf16 / fp32 tensors [T, D])."""
    L = W.layers[vi.layer]
    T = partial.shape[0]
    pos = torch.arange(T)
    tau = None if L.is_local else RA.inkling_log_tau(pos)
    stack = [s.float() for s in stack]
    partial = partial.float()
    messages = None
    Q = L.attn_norm.shape[0]
    with _tick("fwd.attn_in"):
        if vi.v == 0:
            attn_in = torch.stack([RA.rb(RA.rms_norm(partial, L.attn_norm[j], EPS)) for j in range(Q)])
        elif not vi.looped:
            attn_in = RA.inkling_attn_res(stack, partial, None, L.q_attn, L.attn_norm, EPS)
        else:
            dests = RA.inkling_attn_res(stack, partial, None, L.q_attn, None)
            messages = RA.inkling_attn_res([bank[0].float()], bank[1].float(), None, L.q_msg, None)
            norm_w = L.attn_norm
            if L.is_local:
                dests = dests.expand(3, -1, -1)
                norm_w = norm_w.expand(3, -1)
            attn_in = RA.inkling_loop_blend(dests, messages[:3], L.ctrl_down[0:3], L.ctrl_gf[0:3], norm_w, EPS)
    with _tick("fwd.attention"):
        gated = attention(L, attn_in, pos, tau)
    with _tick("fwd.wo_conv"):
        attn_partial = RA.rb(RA.mm(gated, L.wo))
        attn_out, _ = RA.inkling_short_conv(attn_partial, L.attn_sconv, None)
    with _tick("fwd.mlp_in"):
        mlp_norm = None if messages is not None else L.mlp_norm
        if vi.first:
            mlp_in = RA.inkling_attn_res(stack + [partial], attn_out, None, L.q_mlp, mlp_norm, EPS)
        else:
            mlp_in = RA.inkling_attn_res(stack, partial, attn_out, L.q_mlp, mlp_norm, EPS)
        if messages is not None:
            mlp_in = RA.inkling_loop_blend(mlp_in, messages[3:], L.ctrl_down[3:4], L.ctrl_gf[3:4], L.mlp_norm, EPS)
    with _tick("fwd.moe"):
        moe_out = moe_vec(mlp_in[0], L.experts, ids, lora)
    with _tick("fwd.out_conv"):
        new_partial, _ = RA.inkling_short_conv(moe_out, L.mlp_sconv, None, addend=attn_out,
                                               residual=None if vi.first else partial)
    return new_partial


# ---------------------------------------------------------------------------------------------------------------
# LoRA: fp32 masters (rig.make_lora layout) and the engine's bank table (the store's fold, lora_store.cpp set_slot)
# ---------------------------------------------------------------------------------------------------------------
def make_lora(device, seed=0, b_std=0.0, layers=24):
    """PEFT-style init in rig.make_lora's order and generator (CPU, fixed): identical masters to the learnability rig."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for _ in range(layers):
        A13 = (torch.rand(E, RANK, D, generator=g, device="cpu") * 2 - 1) / math.sqrt(D)
        A2 = (torch.rand(E, RANK, I, generator=g, device="cpu") * 2 - 1) / math.sqrt(I)
        B13 = torch.randn(E, 2 * I, RANK, generator=g, device="cpu") * b_std
        B2 = torch.randn(E, D, RANK, generator=g, device="cpu") * b_std
        out.append({k: torch.nn.Parameter(t.to(device)) for k, t in
                    (("A13", A13), ("B13", B13), ("A2", A2), ("B2", B2))} | {"scale": ALPHA / RANK})
    return out


LORA_KEYS = ("A13", "B13", "A2", "B2")


class EngineBanks:
    """The engine's expert-LoRA bank table of every physical layer for ONE slot (slot 0), built from the masters with
    the store's fold: a = bf16(bf16(A) x alpha / r) (set_slot), b = bf16(B); the fused gate_up adapter split as the
    engine's MoE binding does (gate = B13 rows 2j, up = rows 2j + 1, A shared). Table order [0] router, [1] bias (empty),
    [2 + 3e + {0, 1, 2}] = gate / up / down. Rebuild after every optimizer step (refresh)."""

    def __init__(self, lora, device):
        self.dev = device
        self.slot = torch.zeros(4, dtype=torch.int32, device=device)  # uniform slot 0 (16-byte aligned cell)
        self.layers = []
        for lo in lora:
            st = {"A13": torch.empty(E, RANK, D, dtype=BF16, device=device),
                  "Bg": torch.empty(E, I, RANK, dtype=BF16, device=device),
                  "Bu": torch.empty(E, I, RANK, dtype=BF16, device=device),
                  "A2": torch.empty(E, RANK, I, dtype=BF16, device=device),
                  "B2": torch.empty(E, D, RANK, dtype=BF16, device=device)}
            table = (LoraBank * (2 + 3 * E))()
            for e in range(E):
                for p, (a, b, n, k) in enumerate(((st["A13"], st["Bg"], I, D), (st["A13"], st["Bu"], I, D),
                                                  (st["A2"], st["B2"], D, I))):
                    bank = table[2 + 3 * e + p]
                    bank.a = a[e].data_ptr()
                    bank.b = b[e].data_ptr()
                    bank.a_stride = RANK * k
                    bank.b_stride = n * RANK
                    bank.rank, bank.n, bank.k = RANK, n, k
            raw = bytes(table)
            dev_table = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
            ad = MoeAdapters()
            ad.table = dev_table.data_ptr()
            ad.token_slot = None
            ad.uniform_slot = self.slot.data_ptr()
            ad.max_rank = RANK
            self.layers.append({"store": st, "table": dev_table, "adapters": ad})
        self.refresh(lora)

    @torch.no_grad()
    def refresh(self, lora):
        for lo, L in zip(lora, self.layers):
            s = lo["scale"]
            st = L["store"]
            st["A13"].copy_((lo["A13"].detach().to(BF16).float() * s).to(BF16))
            st["A2"].copy_((lo["A2"].detach().to(BF16).float() * s).to(BF16))
            b13 = lo["B13"].detach().to(BF16)
            st["Bg"].copy_(b13[:, 0::2])
            st["Bu"].copy_(b13[:, 1::2])
            st["B2"].copy_(lo["B2"].detach().to(BF16))


# ---------------------------------------------------------------------------------------------------------------
# autograd: engine forward values, torch backward
# ---------------------------------------------------------------------------------------------------------------
class EngineVisit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, vi, ids, n_stack, has_bank, lora_scale, engine_out, *tensors):
        # tensors = stack[0..n_stack-1], partial, [bank0, bank1], A13, B13, A2, B2
        ctx.W, ctx.vi, ctx.ids, ctx.n_stack, ctx.has_bank, ctx.lora_scale = W, vi, ids, n_stack, has_bank, lora_scale
        ctx.in_dtypes = [t.dtype for t in tensors]
        n_act = len(tensors) - (4 if lora_scale is not None else 0)
        # activations: bf16 copies (their values are bf16: exact, half the bytes); LoRA tensors: as they are
        ctx.save_for_backward(*[t.detach().to(BF16) if i < n_act else t for i, t in enumerate(tensors)])
        return engine_out.float()

    @staticmethod
    def backward(ctx, g):
        saved = ctx.saved_tensors
        dev = g.device
        with torch.enable_grad(), torch.device(dev):
            need = ctx.needs_input_grad[7:]
            ins = [(t.detach().float() if t.dtype == BF16 else t.detach()).requires_grad_(bool(nd))
                   for t, nd in zip(saved, need)]
            n = ctx.n_stack
            stack, partial = ins[:n], ins[n]
            bank = (ins[n + 1], ins[n + 2]) if ctx.has_bank else None
            lt = ins[-4:]
            lora = None
            if ctx.lora_scale is not None:
                lora = dict(zip(LORA_KEYS, lt)) | {"scale": ctx.lora_scale}
            out = visit_torch(ctx.W, ctx.vi, stack, partial, bank, ctx.ids, lora)
            want = [t for t in ins if t.requires_grad]
            with _tick("bwd.autograd"):
                grads = torch.autograd.grad(out, want, g, allow_unused=True)
        it = iter(grads)
        res = []
        for t, dt in zip(ins, ctx.in_dtypes):
            gr = next(it) if t.requires_grad else None
            res.append(None if gr is None else gr.to(dt))
        return (None, None, None, None, None, None, None, *res)


class FinalMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, engine_hidden, *srcs):  # srcs = slots 0, 1, 2, 3, then the coda partial (slot 6)
        ctx.W = W
        ctx.save_for_backward(*srcs)
        return engine_hidden.float()

    @staticmethod
    def backward(ctx, g):
        W = ctx.W
        with torch.enable_grad(), torch.device(g.device):
            ins = [t.detach().float().requires_grad_() for t in ctx.saved_tensors]
            h = RA.inkling_attn_res(ins[:4], ins[4], None, W.q_out, W.final_norm.reshape(1, D), EPS)[0]
            grads = torch.autograd.grad(h, ins, g)
        return (None, None, *grads)


def _mm_f32(a_bf16, w_bf16):
    """a [n, K] @ w [m, K]^T, bf16 operands, fp32 accumulate and output (torch out_dtype; fp32 upcast fallback)."""
    try:
        return torch.mm(a_bf16, w_bf16.T, out_dtype=F32)
    except (TypeError, RuntimeError):
        return a_bf16.float() @ w_bf16.float().T


class HeadLogprob(torch.autograd.Function):
    """log p(target | hidden) per row: value = the engine's; gradient = d/dh of log_softmax(30 tanh(W h / 30))[tgt]
    in fp32 over the 200,058 real rows (vocabulary-chunked; nothing of size [N, V] is kept)."""

    @staticmethod
    def forward(ctx, W, engine_lp, hidden, targets):
        ctx.W = W
        ctx.save_for_backward(hidden, targets)
        return engine_lp.clone()

    @staticmethod
    def backward(ctx, g):
        hidden, targets = ctx.saved_tensors
        head = ctx.W.head  # bf16 [200058, 2304]
        V = head.shape[0]
        dh = torch.zeros(hidden.shape, dtype=F32, device=hidden.device)
        rows, vchunk = 256, 32768
        for a in range(0, hidden.shape[0], rows):
            hb = hidden[a:a + rows].to(BF16)  # bf16 values: exact
            ga = g[a:a + rows]
            raw = torch.cat([_mm_f32(hb, head[v0:v0 + vchunk]) for v0 in range(0, V, vchunk)], 1)
            th = torch.tanh(raw / 30.0)
            del raw
            p = torch.softmax(th * 30.0, -1)
            dz = -p * ga[:, None]
            del p
            dz.scatter_add_(1, targets[a:a + rows].long()[:, None], ga[:, None])
            draw = dz * (1.0 - th * th)
            del dz, th
            for v0 in range(0, V, vchunk):
                dh[a:a + rows] += draw[:, v0:v0 + vchunk] @ head[v0:v0 + vchunk].float()
            del draw
        return None, None, dh, None


class TrainForward:
    """One sequence: the engine forward (bitwise the engine's) + the autograd graph over its saved visit outputs.
    build(...) returns per-target log-probs [T - 1] whose backward runs the LEVEL-0 recompute."""

    def __init__(self, eng: InklingEngine, W: EngineWeights):
        self.eng, self.W = eng, W

    def build(self, ids: torch.Tensor, lora=None, banks: EngineBanks | None = None):
        eng = self.eng
        res = F.forward(eng, ids, save=True, adapters_per_layer=(None if banks is None else
                                                                 [L["adapters"] for L in banks.layers]))
        res["lp"] = F.logprobs(eng, res["hidden"], ids)
        lp = self.graph(ids, res, lora)
        return lp, res

    def graph(self, ids: torch.Tensor, vals: dict, lora=None):
        """The autograd graph over given forward values {x0, outs[48], routing [48, T, 8], hidden, lp [T - 1]} (the
        engine's, or any other chain's: tests feed the torch recompute's own chain to check the glue)."""
        eng, W = self.eng, self.W
        slots = {0: vals["x0"]}
        for v, vi in enumerate(eng.visits):
            stack = [slots[s] for s in vi.stack]
            partial = slots[vi.partial]
            bank = [slots[vi.bank[0]], slots[vi.bank[1]]] if vi.looped else []
            lt = [lora[vi.layer][k] for k in LORA_KEYS] if lora is not None else []
            scale = lora[vi.layer]["scale"] if lora is not None else None
            out = EngineVisit.apply(W, vi, vals["routing"][v], len(stack), vi.looped, scale, vals["outs"][v],
                                    *stack, partial, *bank, *lt)
            slots[vi.next_partial] = out
        vals["outs"] = None  # the fp32 visit outputs above hold the values
        hid = FinalMix.apply(W, vals["hidden"], slots[0], slots[1], slots[2], slots[3], slots[6])
        tgt = ids[1:].contiguous()
        return HeadLogprob.apply(W, vals["lp"], hid[:-1], tgt)


@torch.no_grad()
def torch_chain(eng: InklingEngine, W: EngineWeights, ids: torch.Tensor, x0: torch.Tensor, routing: torch.Tensor,
                lora=None) -> dict:
    """The torch recompute run as a whole forward (W4A16, the given routing): {x0, outs, routing, hidden, lp}. Test
    only: TrainForward.graph over these values must give the exact gradient of this chain (the glue check)."""
    with torch.device(eng.dev):
        slots = {0: x0}
        outs = []
        for v, vi in enumerate(eng.visits):
            bank = (slots[vi.bank[0]], slots[vi.bank[1]]) if vi.looped else None
            o = visit_torch(W, vi, [slots[s] for s in vi.stack], slots[vi.partial], bank, routing[v],
                            None if lora is None else lora[vi.layer])
            outs.append(o)
            slots[vi.next_partial] = o
        hid = RA.inkling_attn_res([slots[0], slots[1], slots[2], slots[3]], slots[6], None, W.q_out,
                                  W.final_norm.reshape(1, D), EPS)[0]
        tgt = ids[1:].long()
        lps = []
        headf = W.head.float()
        for a in range(0, hid.shape[0] - 1, 256):
            b = min(hid.shape[0] - 1, a + 256)
            z = torch.tanh((hid[a:b] @ headf.T) / 30.0) * 30.0
            lps.append(z.gather(1, tgt[a:b, None])[:, 0] - torch.logsumexp(z, -1))
        del headf
    return {"x0": x0, "outs": outs, "routing": routing, "hidden": hid, "lp": torch.cat(lps)}
