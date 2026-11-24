"""Does the serving engine follow a full-depth backward whose attention pattern is held constant? One real plan row per process, one card.

Variants of the row backward (t8_lean.backward_row; every one runs all 48 visits, so every physical layer gets a gradient), at the start
point, with the production settings on the engine's stored states:
  H30   pattern_grad_from = 30: visits 0-29 hold their attention pattern constant (no gradient through the softmax scores into q, k and the
        relative bias; the value path keeps its gradient); the last pass and the visits after the loop (30-47) keep the exact backward.
  H36   the same with the cut at visit 36.
  Q     pattern_grad_from = 48: every visit holds its pattern constant.
  QD    Q, and the router's gate weights are constants of the backward at every visit (gate_grad_from = 48).
  HD30  H30, and the gate weights are constants at visits 0-29.
  full  the exact backward (reference).
Measured per row and variant:
  cos     between the variant's gradient on the engine's stored states (rounding straight-through, production head) and on the fp32
          chain's own states (rounding off, exact head), and the norm ratio, for B13 (experts), vB and oB (attention value / output
          adapters) and qB of layers 3, 9, 12, 15, 21.
  follow  in the ENGINE (its own routing, activation quantisation, key / value store): central difference of S = sum of log pi over the
          row's loss tokens along the variant's gradient taken on the engine's states,
          f = [S(theta0 + step) - S(theta0 - step)] / (<G, step+> - <G, step->), the step sized for a predicted change p,
          for B13 of layers 12, 3, 9, 15, 21 one tensor at a time, then for "vo" = vB and oB of layers 3, 9, 12, 15 moved together, and
          for the hybrids "qkr" = qB, kB, rB of layers 9, 12, 15 moved together (the adapters only the kept visits train).
  scatter the engine's S under four steps predicted at 0.01 nats (rms of S - S0): what any step, however small, moves S by.
  p       = max(4 nats, 6 x scatter); recorded with S0.
LINES, written before any data (2026-11-24; the coordinator's): a variant MEETS THE LINES on a row when, for the experts' tensor of every
layer read before the loop (3) and in it (9, 12, 15), cos >= 0.9 and the engine follow is in [0.7, 1.3]. The control (21.B13, after the
loop) must be followed or the row is not read. A reading whose predicted change is below 4 x scatter is not readable and counts for
nothing. "vo" and "qkr" are reported with the same band and do not enter the rule. Selection (fixed before data): the hybrid if it meets
the lines on all rows; else QD; else Q.
Every result is written at once; the deadline cuts the tail."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import fd_scan as FS  # noqa: E402
import torch  # noqa: E402

TD = FS.TD
RA, TL = FS.RA, FS.TL
log = TD.log
LAYERS = (3, 9, 12, 15, 21)
KEYS = ("B13", "vB", "oB", "qB", "kB", "rB")
N_V = 48
VARIANTS = {"H30": {"pattern_grad_from": 30}, "H36": {"pattern_grad_from": 36}, "Q": {"pattern_grad_from": N_V},
            "QD": {"pattern_grad_from": N_V, "gate_grad_from": N_V}, "HD30": {"pattern_grad_from": 30, "gate_grad_from": 30},
            "full": {}, "cut0": {"bank_grad_from": N_V}}


def run(C, tt, vals, head, round_on, kw):
    """{tensor: gradient on the host (summed over the layer's visits)}, seconds, number of visits that received a gradient."""
    out, seen = {}, []

    def ong(v, layer, j, gj):
        nm = tt.get((layer, j))
        if nm is not None:
            c = gj.detach().float().cpu()
            out[nm] = c if nm not in out else out[nm].add_(c)

    RA.ROUND["on"] = round_on
    torch.cuda.synchronize()
    t0 = time.time()
    try:
        TL.backward_row(C.eng, C.W, C.ids, vals, C.rig.w.lora, C.scale, C.rig.w.v_first, C.mask[1:].float(), C.rig.w.buffers, ong,
                        head_grad=head, q_block=C.qb, train_idx=C.rig.w.train_idx, attn_scale=C.attn_scale,
                        on_visit_grad=lambda v, go: seen.append(v), **kw)
    finally:
        RA.ROUND["on"] = False
    torch.cuda.synchronize()
    return out, time.time() - t0, len(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--variants", default="H30,QD,H36,Q,HD30")
    ap.add_argument("--no-cos", action="store_true", help="skip the chain-state backwards")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, default=720.0)
    a = ap.parse_args()
    deadline = time.time() + a.budget_s
    TD.ROW = a.row
    order = [v for v in a.variants.split(",") if v]
    rep = {"row": a.row, "variants": order, "dropped": [], "follow": [], "times": {}, "norms": {}, "cos": {}}

    def save():
        json.dump(rep, open(a.out + ".tmp", "w"), indent=1)
        os.replace(a.out + ".tmp", a.out)

    def left():
        return deadline - time.time()

    t0 = time.time()
    C = FS.Ctx(a, keep_engine=True)
    rep["T"], rep["loss_tokens"] = C.T, int(C.mask.sum())
    rep["times"]["setup_s"] = time.time() - t0
    keys = C.rig.keys
    tt = {(L, keys.index(k)): f"{L}.{k}" for L in LAYERS for k in KEYS}
    vw = C.view()
    assert len(C.eng.visits) == N_V
    GE, chain = {}, {}
    try:
        tf = time.time()
        e0 = C.engine_S(vw)
        tfwd = time.time() - tf
        e0b = C.engine_S(vw)
        rep.update({"S0_engine": e0, "S0_engine_repeat": e0b})
        rep["times"]["engine_forward_s"] = tfwd
        log(f"row {a.row} T {C.T} loss tokens {rep['loss_tokens']}: S0 engine {e0:.4f} (repeat {e0b:.4f}), engine forward {tfwd:.1f} s")
        save()

        def follow(var, label, parts, pp):
            """parts: [(tensor, gradient)] moved together along the variant's gradient, predicted change pp."""
            parts = [(nm, G) for nm, G in parts if G is not None and float(G.double().norm()) > 0.0]
            if not parts:
                rep["dropped"].append([var, label, "no gradient"])
                return
            if left() < 2.5 * tfwd + 5:
                rep["dropped"].append([var, label, pp, "deadline"])
                return
            n2 = sum(float(G.double().pow(2).sum()) for _, G in parts)
            S, pred = {}, {}
            for sg in (1.0, -1.0):
                pred[sg] = 0.0
                for nm, G in parts:
                    pr, _ = C.set_step(nm, G, (G.double() * (sg * pp / n2)).float())
                    pred[sg] += pr
                S[sg] = C.engine_S(vw)
            for nm, _ in parts:
                C.set_step(nm, None, None)
            r = {"variant": var, "what": label, "p": pp, "pred+": pred[1.0], "pred-": pred[-1.0], "S+": S[1.0], "S-": S[-1.0],
                 "f": (S[1.0] - S[-1.0]) / (pred[1.0] - pred[-1.0]), "f_plus": (S[1.0] - e0) / pred[1.0], "f_minus": (S[-1.0] - e0) / pred[-1.0],
                 "even": (S[1.0] + S[-1.0] - 2 * e0) / 2, "norm": n2 ** 0.5, "readable": abs(pred[1.0]) >= 4 * rep["scatter"]}
            r["real_change_per_unit_step"] = r["f"] * r["norm"]
            rep["follow"].append(r)
            log(f"follow {var} {label} p {pp:.2f}: f {r['f']:.3f} (+ {r['f_plus']:.3f}, - {r['f_minus']:.3f}), |G| {r['norm']:.3f}, "
                f"f x |G| {r['real_change_per_unit_step']:.3f}, readable {r['readable']}")
            save()

        def cos_for(var):
            if a.no_cos or var in rep["cos"]:
                return
            tE = rep["times"].get(f"{var}_s", 0.0)
            if not chain:
                if left() < 1.6 * tE + 30:
                    rep["dropped"].append([var, "cos", "deadline"])
                    return
                try:
                    outs, hid = [], []
                    tc0 = time.time()
                    rep["S0_chain"] = C.chain(vw, keep_outs=outs, keep_hidden=hid)
                    rep["times"]["chain_forward_s"] = time.time() - tc0
                    host = {-1: C.x0}
                    for v, o in enumerate(outs):
                        host[v] = o
                    chain["vals"] = {"host": host, "routing": C.routing, "hidden": hid[0], "lp": None}
                except torch.OutOfMemoryError as e:
                    rep["chain_error"] = str(e)[:400]
                    a.no_cos = True
                    torch.cuda.empty_cache()
                    log("chain forward out of card memory: cos not measured on this row")
                    save()
                    return
            if left() < 1.15 * tE + 10:
                rep["dropped"].append([var, "cos", "deadline"])
                return
            GS, _, _ = run(C, tt, chain["vals"], FS.head_exact, False, VARIANTS[var])
            rep["cos"][var] = {nm: FS.cmp(GE[var][nm], GS[nm]) for nm in GE[var] if nm in GS and float(GE[var][nm].double().norm()) > 0}
            log(f"cos {var} engine-state vs chain-state: " + " ".join(f"{nm} {x['cos']:.3f}" for nm, x in rep["cos"][var].items()
                                                                       if nm.endswith("B13") or nm.endswith("vB") or nm.endswith("oB")))
            del GS
            save()

        for i, var in enumerate(order):
            tE_est = max([rep["times"][f"{v}_s"] for v in order if f"{v}_s" in rep["times"]] or [0.0])
            if left() < 1.1 * tE_est + 6 * tfwd:
                rep["dropped"].append([var, "deadline"])
                continue
            G, sE, nvis = run(C, tt, C.engine_res, None, True, VARIANTS[var])
            GE[var] = G
            rep["times"][f"{var}_s"] = sE
            rep["times"][f"{var}_visits"] = nvis
            rep["norms"][var] = {nm: float(g.double().norm()) for nm, g in G.items()}
            log(f"{var}: backward {sE:.0f} s, {nvis} visits; |G| " + " ".join(f"{nm} {rep['norms'][var][nm]:.3f}" for nm in sorted(G) if nm.endswith("B13")))
            save()
            if "scatter" not in rep:
                devs = []
                for nm in ("3.B13", "12.B13"):
                    for sg in (1.0, -1.0):
                        C.set_step(nm, G[nm], C.unit_step(G[nm], sg * 0.01))
                        devs.append(C.engine_S(vw) - e0)
                    C.set_step(nm, None, None)
                rep["tiny_step_devs"] = devs
                rep["scatter"] = (sum(d * d for d in devs) / len(devs)) ** 0.5
                rep["p"] = max(4.0, 6.0 * rep["scatter"])
                log(f"tiny-step deviations {[round(d, 3) for d in devs]}; scatter {rep['scatter']:.3f}; p {rep['p']:.2f}")
                save()
            p = rep["p"]
            for L in (12, 3, 9, 15, 21):
                follow(var, f"{L}.B13", [(f"{L}.B13", G.get(f"{L}.B13"))], p)
            follow(var, "vo", [(f"{L}.{k}", G.get(f"{L}.{k}")) for L in (3, 9, 12, 15) for k in ("vB", "oB")], p)
            if var.startswith("H") or var == "full":
                follow(var, "qkr", [(f"{L}.{k}", G.get(f"{L}.{k}")) for L in (9, 12, 15) for k in ("qB", "kB", "rB")], p)
            if i >= 1:                     # the chain's states after the first two variants' follows: those first
                for v2 in order[:i + 1]:
                    if v2 in GE:
                        cos_for(v2)
        for v2 in order:
            if v2 in GE:
                cos_for(v2)
    except Exception:
        import traceback

        rep["error"] = traceback.format_exc()[-3000:]
        log("ERROR " + rep["error"][-800:])
        save()
        C.rig.w.close()
        return 1
    rep["seconds"] = a.budget_s - left()
    save()
    C.rig.w.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
