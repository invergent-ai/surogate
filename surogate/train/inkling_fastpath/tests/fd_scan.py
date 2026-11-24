"""Micro-scan of the looped backward against the forward on ONE real plan row (one card per part, in-process worker; nothing trained).

The function: the trainer's fp32 recompute chained as a forward (t8_diag.chain_S), routing frozen at the start point's, every rounding off
(RA.ROUND off): a smooth fp32 function of the adapter tensors. S = sum of log pi over the row's loss tokens.
Gradients at the chain's own states:
  G_P  backward_row with the production storage (bf16 copies of token inputs inside TokenChunked and of q / k / v / r in RelAttentionR),
  G_X  the same with those copies kept in fp32 (exact_storage: the backward recomputes at exactly the values the forward used),
both with the exact fp32 head gradient. Steps go along G_X; every prediction is <G_X, the step actually applied> (work values are bf16).

LINES, written before any data (2026-11-24):
 Part X  X1 S at the start point three times and once through set_one with a zero step: floor = the largest difference.
         X2 G_P against G_X (cos, norm ratio) per tensor and per visit. "rough at one bf16 ulp": cos < 0.9 at 12.B13 or 3.B13 while
            21.B13 >= 0.99.
         X3 S(theta0 + k h u), u = G_X / |G_X|, h = a predicted 0.005 nats, k = -4 .. 4 for 12.B13 and 3.B13, k = -2 .. 2 for the control
            21.B13. Least-squares line of S against the prediction: slope (the follow) and R^2.
            The control must give slope in [0.9, 1.1] and R^2 >= 0.98, or X3 is not read.
            L-exact: R^2 >= 0.98 and slope in [0.8, 1.25]: the backward is the exact gradient of the chain at its point.
            L-sum:   R^2 >= 0.98 and slope < 0.5: a smooth function with a wrong gradient (the summing across visits, or a path).
            L-rough: R^2 < 0.9, or residuals above 5 x the X1 floor with the slope outside [0.8, 1.25]: not differentiable at this
                     resolution; no exact backward can be followed.
            Anything else: cannot tell.
         X4 per visit: share of |g_out|^2 on the top 1 / 10 / 100 / 1,000 tokens, their positions, their output RMS. Recorded.
 Part Y  central and one-sided differences at predicted 0.002, 0.031, 0.5, 0.0005 nats for 12.B13, an injection into the output of visit 12
         along g_out, visit 6, 3.B13, visit 30; visit 42 at 0.002 and 0.5 as the control. Gives the size of the linear range if X3 is exact.
 Part Z  prefixes of the row (512, 1,024, 2,048, 4,096, then the full row, then 8,192), S = the LAST 128 positions' next-token log-probs
         (context tokens, not the row's loss tokens). Per length: G_X, the chain's follow at predicted 0.03 and 0.5 nats and the ENGINE's
         follow (its own routing, its own activation quantisation) at predicted 0.5 and 4 nats, for 12.B13, 3.B13, 21.B13.
         Chain follow of 12.B13 >= 0.7 at 1,024 tokens with < 0.3 at the full length: length drives the chain's failure; < 0.3 at 512: it
         does not. Engine follow of 12.B13 >= 0.5 at 512 - 1,024 tokens: short rows are followed by the engine; < 0.3: they are not.
Every result is written at once; a cut at the deadline costs only the tail."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import t8_diag as TD  # noqa: E402  (sets the proven stack's environment before the trainer modules are imported)
import torch  # noqa: E402

B, F, LG, RA, TL, TV = TD.B, TD.F, TD.LG, TD.RA, TD.TL, TD.TV
log = TD.log
NAMES = ("12.B13", "3.B13", "21.B13", "12.qB", "3.qB", "21.qB")


def exact_storage():
    """Test-side patch: TokenChunked and RelAttentionR keep fp32 copies, so the backward recomputes at the forward's own values. Returns undo."""
    orig_tc = LG.tc

    def tc_exact(fn, tok, tdims_in, tdims_out, params=(), store_bf16=None, chunk=None, vals=None, sinks=None):
        return orig_tc(fn, tok, tdims_in, tdims_out, params=params, store_bf16=None, chunk=chunk, vals=vals, sinks=sinks)

    LG.tc = tc_exact
    TV.tc = tc_exact
    undo_att = TD._patch_source(LG, "forward", [("ctx.save_for_backward(q.to(BF16), k.to(BF16), v.to(BF16), r_out.to(BF16), rel_proj, out, lse)",
                                                  "ctx.save_for_backward(q, k, v, r_out, rel_proj, out, lse)")], cls="RelAttentionR")

    def undo():
        LG.tc = orig_tc
        TV.tc = orig_tc
        undo_att()

    return undo


def head_exact(W_, hidden, targets, gg):
    """The exact fp32 head gradient (the same arithmetic as the chain's head)."""
    h = hidden.detach().float()
    hf = W_.head.float()
    out = torch.zeros_like(h)
    with torch.enable_grad():
        for c0 in range(0, h.shape[0], 128):
            if not bool((gg[c0:c0 + 128] != 0).any()):
                continue
            hh = h[c0:c0 + 128].clone().requires_grad_(True)
            z = torch.tanh((hh @ hf.T) / 30.0) * 30.0
            val = ((z.gather(1, targets[c0:c0 + 128].long()[:, None])[:, 0] - torch.logsumexp(z, -1)) * gg[c0:c0 + 128]).sum()
            out[c0:c0 + 128] = torch.autograd.grad(val, hh)[0]
    del hf
    return out


def cmp(x, y):
    x, y = x.double().reshape(-1), y.double().reshape(-1)
    return {"cos": float((x * y).sum() / (x.norm() * y.norm()).clamp(min=1e-300)), "norm_ratio": float(x.norm() / y.norm().clamp(min=1e-300)),
            "rel_diff": float((x - y).norm() / y.norm().clamp(min=1e-300))}


def concentration(go, out_host, lo):
    e = go.double().pow(2).sum(-1)
    tot = float(e.sum())
    sv, si = e.sort(descending=True)
    d = {"norm": tot ** 0.5, "top_pos": [int(i) for i in si[:8].tolist()], "share_from_first_loss_pos": float(e[lo:].sum() / tot) if tot else None}
    for k in (1, 10, 100, 1000):
        d[f"top{k}"] = float(sv[:k].sum() / tot) if tot else None
    ed = go.double().pow(2).sum(0)
    d["dim_top10"] = float(ed.sort(descending=True).values[:10].sum() / tot) if tot else None
    if out_host is not None:
        r = out_host.double().pow(2).mean(-1).sqrt()
        d["rms_median"], d["rms_min"], d["rms_max"] = float(r.median()), float(r.min()), float(r.max())
        d["rms_top"] = [float(r[int(i)]) for i in si[:8].tolist()]
    return d


class Ctx:
    """The rig at the start point, the row, the engine's routing and x0."""

    def __init__(self, a, keep_engine=False):
        RA.ROUND["on"] = False
        self.rig = TD.Rig(a)
        rig = self.rig
        self.row, _ = TD.load_row(a.plan)
        self.dev = rig.w.dev
        self.eng, self.W = rig.w.eng, rig.w.W
        lp, ctx = rig.w._forward(self.row)
        self.ids, self.qb = ctx["ids"], ctx["qb"]
        self.routing = ctx["res"]["routing"].clone()
        self.x0 = ctx["res"]["host"][-1].to(self.dev).float().clone()   # the normed embedding the engine stored (no second forward: at 57k
        #                                                                  tokens a forward that keeps all 48 outputs does not fit beside it)
        self.engine_res = ctx["res"] if keep_engine else None      # the engine's stored states (host views; valid until the next _forward)
        if not keep_engine:
            rig.w._release(ctx)
        del ctx, lp
        self.eng._ws = None
        self.mask = torch.as_tensor(self.row.loss_mask.copy(), device=self.dev)
        self.scale, self.attn_scale = B.ALPHA / B.RANK, rig.cfg.attn_alpha / rig.cfg.attn_rank
        keys = rig.keys
        self.tt = {(int(nm.split(".")[0]), keys.index(nm.split(".")[1])): nm for nm in NAMES}
        self.T = int(self.ids.numel())

    def view(self, Tp=None, last=128):
        """(ids, x0, routing, mask, g, qb) of the full row, or of its first Tp tokens with the last `last` positions as the loss."""
        if Tp is None or Tp >= self.T and last is None:
            return self.ids, self.x0, self.routing, self.mask, self.mask[1:].float(), self.qb
        Tp = min(int(Tp), self.T)
        m = torch.zeros(Tp, dtype=torch.bool, device=self.dev)
        m[Tp - last:] = True
        return self.ids[:Tp].contiguous(), self.x0[:Tp].contiguous(), self.routing[:, :Tp].contiguous(), m, m[1:].float(), TL.qb_for(Tp)

    def chain(self, vw, inject=None, keep_outs=None, keep_hidden=None):
        ids, x0, routing, mask = vw[:4]
        return TD.chain_S(self.rig, ids, x0, routing, mask, inject=inject, keep_outs=keep_outs, keep_hidden=keep_hidden)[0]

    def engine_S(self, vw):
        ids, mask = vw[0], vw[3]
        with torch.no_grad():
            res = F.forward(self.eng, ids, save=False, adapters_per_layer=self.rig.w._adapters())
            lpx = F.logprobs(self.eng, res["hidden"], ids)
        s = float(lpx.double()[mask[1:]].sum())
        del res, lpx
        self.eng._ws = None
        return s

    def backward(self, vw, outs, hid, keep_g=(), stats=None):
        ids, x0, routing, mask, g, qb = vw
        host = {-1: x0}
        for v, o in enumerate(outs):
            host[v] = o
        vals = {"host": host, "routing": routing, "hidden": hid, "lp": None}
        Gs, gv, norms = {}, {}, {}
        lo = int(torch.nonzero(mask)[0, 0])

        def ong(v, layer, j, gj):
            nm = self.tt.get((layer, j))
            if nm is not None:
                c = gj.detach().float().cpu()
                Gs[nm] = c if nm not in Gs else Gs[nm].add_(c)

        def onv(v, go):
            norms[v] = float(go.double().norm())
            if keep_g is None or v in keep_g:
                gv[v] = go.detach().float().cpu()
            if stats is not None:
                stats[str(v)] = concentration(go, outs[v], lo)

        rig = self.rig
        TL.backward_row(self.eng, self.W, ids, vals, rig.w.lora, self.scale, rig.w.v_first, g, rig.w.buffers, ong, head_grad=head_exact,
                        q_block=qb, train_idx=rig.w.train_idx, attn_scale=self.attn_scale, on_visit_grad=onv)
        return Gs, gv, norms

    def set_step(self, nm, G, step):
        """theta0 + step on tensor nm (None: back to theta0). Returns <G, the applied step> and the applied step's norm."""
        L, k = nm.split(".")
        self.rig.set_one(int(L), k, step)
        if step is None:
            return 0.0, 0.0
        applied = self.rig.work[int(L)][k].detach().double().cpu() - TD.bf16(self.rig.theta0[int(L)][k]).double()
        return float((G.double() * applied).sum()), float(applied.norm())

    def unit_step(self, G, pred):
        """The step along G with a predicted change `pred` (before the bf16 rounding of the work values)."""
        g = G.double()
        return (g / g.norm() * (pred / float(g.norm()))).float()


def fit_line(xs, ys):
    x, y = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    A = np.stack([x, np.ones_like(x)], 1)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ sol
    ss = float(((y - y.mean()) ** 2).sum())
    return {"slope": float(sol[0]), "intercept": float(sol[1]), "r2": 1.0 - float((res ** 2).sum()) / ss if ss > 0 else None,
            "resid_max": float(np.abs(res).max()), "resid_rms": float(np.sqrt((res ** 2).mean()))}


def part_X(a, C, rep, save, deadline):
    vw = C.view()
    outs, hid = [], []
    S0 = C.chain(vw, keep_outs=outs, keep_hidden=hid)
    rep["S0"] = S0
    t0 = time.time()
    G_P, g_P, n_P = C.backward(vw, outs, hid[0], keep_g=None)
    rep["backward_s"] = time.time() - t0
    undo = exact_storage()
    stats = {}
    try:
        G_X, g_X, n_X = C.backward(vw, outs, hid[0], keep_g=None, stats=stats)
    finally:
        undo()
    rep["X2"] = {"tensors": {nm: cmp(G_P[nm], G_X[nm]) for nm in G_X}, "per_visit": {str(v): cmp(g_P[v], g_X[v]) for v in sorted(g_X) if v in g_P},
                 "norms_GX": {nm: float(t.double().norm()) for nm, t in G_X.items()}, "norms_GP": {nm: float(t.double().norm()) for nm, t in G_P.items()},
                 "gout_norm_GX": {str(v): n_X[v] for v in sorted(n_X)},
                 "patch_effective": any(float((G_P[nm] - G_X[nm]).abs().max()) > 0 for nm in G_X)}
    rep["X4"] = stats
    log("X2 G_P vs G_X: " + ", ".join(f"{nm} cos {x['cos']:.4f} ratio {x['norm_ratio']:.3f}" for nm, x in rep["X2"]["tensors"].items())
        + f"; patch effective {rep['X2']['patch_effective']}")
    log("X4 top-10 token share of |g_out|^2 per visit: " + " ".join(f"{v}:{stats[v]['top10']:.2f}" for v in sorted(stats, key=int)))
    save()
    del g_P, g_X, G_P
    gc.collect()
    # X1
    reps = [S0, C.chain(vw), C.chain(vw)]
    C.set_step("12.B13", G_X["12.B13"], torch.zeros_like(G_X["12.B13"]))
    reps.append(C.chain(vw))
    C.set_step("12.B13", None, None)
    rep["X1"] = {"S": reps, "floor": max(reps) - min(reps)}
    log(f"X1 S0 repeats {reps}, floor {rep['X1']['floor']:.3g}")
    save()
    # X3
    rep["X3"] = {}
    H = 0.005
    for nm, ks in (("21.B13", (-2, -1, 1, 2)), ("12.B13", (1, -1, 2, -2, 3, -3, 4, -4)), ("3.B13", (1, -1, 2, -2, 3, -3, 4, -4))):
        G = G_X[nm]
        pts = [{"k": 0, "pred": 0.0, "S": S0}]
        r = {"points": pts}
        rep["X3"][nm] = r
        for k in ks:
            if time.time() > deadline:
                rep["dropped"].append(["X3", nm, k])
                continue
            pred, an = C.set_step(nm, G, C.unit_step(G, k * H))
            s = C.chain(vw)
            pts.append({"k": k, "pred": pred, "S": s, "applied_norm": an})
            if len(pts) >= 3:
                r["fit"] = fit_line([p["pred"] for p in pts], [p["S"] - S0 for p in pts])
            save()
        C.set_step(nm, None, None)
        if "fit" in r:
            log(f"X3 {nm}: slope {r['fit']['slope']:.4f}, R2 {r['fit']['r2']:.4f}, resid max {r['fit']['resid_max']:.3g}; "
                + " ".join(f"{p['k']}:{p['S'] - S0:+.5f}" for p in sorted(pts, key=lambda p: p["k"])))
        save()


def part_Y(a, C, rep, save, deadline):
    vw = C.view()
    outs, hid = [], []
    S0 = C.chain(vw, keep_outs=outs, keep_hidden=hid)
    outs2 = []
    S0b = C.chain(vw, keep_outs=outs2)
    rep["S0"], rep["S0_repeat"] = S0, S0b
    rep["outs_bitwise_repeat"] = all(bool(torch.equal(x, y)) for x, y in zip(outs, outs2))
    del outs2
    undo = exact_storage()
    try:
        G_X, g_X, n_X = C.backward(vw, outs, hid[0], keep_g=(42, 30, 12, 6))
    finally:
        undo()
    rep["norms_GX"] = {nm: float(t.double().norm()) for nm, t in G_X.items()}
    rep["gout_norm"] = {str(v): n_X[v] for v in sorted(n_X)}
    rep["ladder"] = []
    log(f"Y: S0 {S0:.6f} / {S0b:.6f}, outs bitwise on repeat {rep['outs_bitwise_repeat']}")
    save()
    dev = C.dev

    def param(nm, p):
        G = G_X[nm]
        st = C.unit_step(G, p)
        pred, _ = C.set_step(nm, G, st)
        sp = C.chain(vw)
        predm, _ = C.set_step(nm, G, -st)
        sm = C.chain(vw)
        C.set_step(nm, None, None)
        return {"target": nm, "p": p, "pred+": pred, "pred-": predm, "S+": sp, "S-": sm}

    def inj(v, p):
        go = g_X[v].to(dev)
        base = outs[v].to(dev)
        gn = float(go.double().norm())
        d = (go.double() / gn * (p / gn)).float()
        ap = (base + d) - base
        am = (base - d) - base
        pp, pm = float((go.double() * ap.double()).sum()), float((go.double() * am.double()).sum())
        rr = {"target": f"v{v}", "p": p, "pred+": pp, "pred-": pm, "applied_over_intended": float(ap.double().norm() / d.double().norm())}
        del base, go
        rr["S+"] = C.chain(vw, inject=(v, d))
        rr["S-"] = C.chain(vw, inject=(v, -d))
        del d, ap, am
        return rr

    tasks = []
    for p in (0.002, 0.031, 0.5, 0.0005):
        if p in (0.002, 0.5):
            tasks.append(("i", 42, p))
        tasks += [("p", "12.B13", p), ("i", 12, p), ("i", 6, p), ("p", "3.B13", p), ("i", 30, p)]
    for kind, tgt, p in tasks:
        if time.time() > deadline:
            rep["dropped"].append([kind, tgt, p])
            continue
        r = param(tgt, p) if kind == "p" else inj(tgt, p)
        den = r["pred+"] - r["pred-"]
        r["f_central"] = (r["S+"] - r["S-"]) / den if den else None
        r["f_plus"] = (r["S+"] - S0) / r["pred+"] if r["pred+"] else None
        r["f_minus"] = (r["S-"] - S0) / r["pred-"] if r["pred-"] else None
        r["even"] = (r["S+"] + r["S-"] - 2 * S0) / 2
        rep["ladder"].append(r)
        log(f"Y {r['target']} p {p:g}: central {r['f_central']}, + {r['f_plus']}, - {r['f_minus']}, even {r['even']:+.3g}, "
            f"pred {r['pred+']:+.3g}/{r['pred-']:+.3g}")
        save()


def part_Z(a, C, rep, save, deadline):
    rep["lengths"] = []
    for Tp in (512, 1024, 2048, 4096, C.T, 8192):
        if time.time() > deadline - (60 if Tp <= 4096 else 330):
            rep["dropped"].append(["Z", Tp])
            continue
        vw = C.view(Tp, last=128)
        outs, hid = [], []
        S0 = C.chain(vw, keep_outs=outs, keep_hidden=hid)
        e0 = C.engine_S(vw)
        undo = exact_storage()
        try:
            G_X, _, n_X = C.backward(vw, outs, hid[0], keep_g=())
        finally:
            undo()
        del outs, hid
        r = {"T": int(vw[0].numel()), "S0_chain": S0, "S0_engine": e0, "norms_GX": {nm: float(t.double().norm()) for nm, t in G_X.items()},
             "gout_norm": {str(v): n_X[v] for v in sorted(n_X)}, "chain": [], "engine": []}
        rep["lengths"].append(r)
        save()
        for nm in ("12.B13", "3.B13", "21.B13"):
            G = G_X[nm]
            for mode, p in (("chain", 0.03), ("chain", 0.5), ("engine", 0.5), ("engine", 4.0)):
                if time.time() > deadline:
                    rep["dropped"].append(["Z", Tp, nm, mode, p])
                    continue
                st = C.unit_step(G, p)
                fn = (lambda: C.chain(vw)) if mode == "chain" else (lambda: C.engine_S(vw))
                base = S0 if mode == "chain" else e0
                pp, _ = C.set_step(nm, G, st)
                sp = fn()
                pm, _ = C.set_step(nm, G, -st)
                sm = fn()
                C.set_step(nm, None, None)
                x = {"tensor": nm, "p": p, "pred+": pp, "pred-": pm, "S+": sp, "S-": sm, "f_central": (sp - sm) / (pp - pm),
                     "f_plus": (sp - base) / pp, "f_minus": (sm - base) / pm, "even": (sp + sm - 2 * base) / 2}
                r[mode].append(x)
                log(f"Z T {r['T']} {mode} {nm} p {p:g}: central {x['f_central']:.4f}, + {x['f_plus']:.4f}, - {x['f_minus']:.4f}, even {x['even']:+.3g}")
                save()
        del G_X
        gc.collect()
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("X", "Y", "Z"))
    ap.add_argument("--plan", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, default=840.0)
    a = ap.parse_args()
    deadline = time.time() + a.budget_s
    rep = {"row": TD.ROW, "part": a.part, "dropped": []}

    def save():
        json.dump(rep, open(a.out + ".tmp", "w"), indent=1)
        os.replace(a.out + ".tmp", a.out)

    C = Ctx(a)
    rep["T"], rep["loss_tokens"] = C.T, int(C.mask.sum())
    try:
        {"X": part_X, "Y": part_Y, "Z": part_Z}[a.part](a, C, rep, save, deadline)
    except Exception:
        import traceback

        rep["error"] = traceback.format_exc()[-3000:]
        log("ERROR " + rep["error"][-600:])
        save()
        C.rig.w.close()
        return 1
    rep["seconds"] = a.budget_s - (deadline - time.time())
    save()
    C.rig.w.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
