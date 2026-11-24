"""Does the serving engine follow a full-depth backward whose attention pattern is held constant? One real plan row per process, one card.

Variants of the row backward (t8_lean.backward_row; every one runs all 48 visits, so every physical layer gets a gradient), at the start
point, with the production settings on the engine's stored states:
  H30   pattern_grad_from = 30: visits 0-29 hold their attention pattern constant (no gradient through the softmax scores into q, k and the
        relative bias; the value path keeps its gradient); the last pass and the visits after the loop (30-47) keep the exact backward.
  H36   the same with the cut at visit 36.
  Q     pattern_grad_from = 48: every visit holds its pattern constant.
  QD    Q, and the router's gate weights are constants of the backward at every visit (gate_grad_from = 48).
  HD30  H30, and the gate weights are constants at visits 0-29.
  H42 / HD42 / HD36  the same forms with the cut at visit 42 (only the visits after the loop keep the exact backward) or 36. Added after
        the first data of H30, before any data of theirs; same lines.
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
  scatter the engine's S under six steps predicted at 0.01 nats: the mean of S - S0 (a shift every step causes: the row's tokens were
          sampled from the start policy, so any change of the function lowers their summed log-probability; a central difference cancels
          it) and its standard deviation about that mean = the scatter.
  p       = min(max(4 nats, 6 x scatter), 0.4 |S0|); recorded with S0. S <= 0, so a step predicted near |S0| cannot be followed upward: a row
          where 6 x scatter exceeds the cap cannot be read (p_valid False), and a reading predicted above 0.81 |S0| is not readable.
          (The first version took the scatter as an rms about zero, which counts the shift as noise, and had no cap; both were changed
          after the first data and are recorded as such with the results.)
  repeat  a gating reading in [0.5, 0.7) or (1.3, 1.5] is read once more at twice the step; only a second reading outside [0.7, 1.3]
          counts as outside (set before any data was read).
LINES, written before any data (2026-11-24): a variant MEETS THE LINES on a row when, for the experts' tensor of every
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
import numpy as np  # noqa: E402
import torch  # noqa: E402

TD = FS.TD
RA, TL = FS.RA, FS.TL
log = TD.log
LAYERS = (3, 9, 12, 15, 21)
KEYS = ("B13", "vB", "oB", "qB", "kB", "rB")
N_V = 48
VARIANTS = {"H30": {"pattern_grad_from": 30}, "H36": {"pattern_grad_from": 36}, "Q": {"pattern_grad_from": N_V},
            "QD": {"pattern_grad_from": N_V, "gate_grad_from": N_V}, "HD30": {"pattern_grad_from": 30, "gate_grad_from": 30},
            "H42": {"pattern_grad_from": 42}, "HD42": {"pattern_grad_from": 42, "gate_grad_from": 42},
            "HD36": {"pattern_grad_from": 36, "gate_grad_from": 36},
            "full": {}, "cut0": {"bank_grad_from": N_V}}


def run(C, tt, vals, head, round_on, kw, seed_scale=None):
    """{tensor: gradient on the host (summed over the layer's visits)}, seconds, number of visits that received a gradient.
    seed_scale: the seed on every loss token's log-prob (default 1; the trainer's PPO seed at advantage 1 is -1 / N)."""
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
        g_seed = C.mask[1:].float() if seed_scale is None else C.mask[1:].float() * torch.tensor(seed_scale, dtype=torch.float32, device=C.dev)
        TL.backward_row(C.eng, C.W, C.ids, vals, C.rig.w.lora, C.scale, C.rig.w.v_first, g_seed, C.rig.w.buffers, ong,
                        head_grad=head, q_block=C.qb, train_idx=C.rig.w.train_idx, attn_scale=C.attn_scale,
                        on_visit_grad=lambda v, go: seen.append(v), **kw)
    finally:
        RA.ROUND["on"] = False
    torch.cuda.synchronize()
    return out, time.time() - t0, len(seen)


def gate_verdict(rep, var, labels=("12.B13", "3.B13", "21.B13")):
    """The finite-difference gate of a backward form on one real row. PASS =
    (1) follow: every label's last reading (the repeat at twice the step when there was one) is readable, predicts at least 3 nats and is
        inside [0.7, 1.3]: one tensor inside the loop, one before it, the control after it;
    (2) cos: the form's gradient on the engine's stored states against the recompute chain's own states is >= 0.9 for the two gating tensors;
    (3) when the trainer path was run: the step's gradient through the trainer's configuration equals the direct row backward's
        (cos >= 0.99999, norm within 0.1 %) on the tensors compared."""
    out, ok = {}, True
    for lab in labels:
        rs = [r for r in rep["follow"] if r["variant"] == var and r["what"] == lab]
        if not rs:
            out[lab] = "missing"
        elif not rs[-1]["readable"] or abs(rs[-1]["pred+"]) < min(3.0, 0.39 * abs(rep["S0_engine"])):
            out[lab] = f"not readable ({rs[-1]['f']:.3f})"
        else:
            out[lab] = f"{'in band' if 0.7 <= rs[-1]['f'] <= 1.3 else 'outside'} ({rs[-1]['f']:.3f})"
        ok = ok and out[lab].startswith("in band")
    cos = rep.get("cos", {}).get(var, {})
    for lab in labels[:2]:
        c = cos.get(lab)
        out[f"cos {lab}"] = "missing" if c is None else f"{'ok' if c['cos'] >= 0.9 else 'low'} ({c['cos']:.3f})"
        ok = ok and c is not None and c["cos"] >= 0.9
    tp = rep.get("trainer_path_vs_direct")
    if tp is not None:
        good = all(x["cos"] >= 0.99999 and abs(x["norm_ratio"] - 1.0) <= 1e-3 for x in tp.values())
        out["trainer path"] = f"{'equal' if good else 'DIFFERS'} (min cos {min(x['cos'] for x in tp.values()):.7f})"
        ok = ok and good
    return {"variant": var, "readings": out, "pass": bool(ok)}


def trainer_path_check(C, var, G, keys):
    """The same row's gradient through the trainer's own path (configuration -> worker -> row backward), against the direct call's."""
    import t8_step as TS
    from surogate.inkling_rl.step import StepConfig

    cfg, rig = C.rig.cfg, C.rig
    old_opts = {k: getattr(cfg, k) for k in ("pattern_grad_from", "gate_grad_from", "bank_grad_from")}
    RA.ROUND["on"] = True          # the trainer's setting (this test's context switches it off for the chain; the direct gradient ran with it on)
    try:
        for k_, v_ in VARIANTS[var].items():
            setattr(cfg, k_, v_)
        TS.validate_config(cfg)
        row = C.row
        row.advantage = 1.0
        row.notes["old_lp"] = rig.h.call("logprobs_row", row)
        out = TS.compute_step([rig.h], rig.layout, [rig.flat], rig.shapes, cfg, [row], 0, StepConfig(max_policy_lag=None), seeds="ppo")
        N = float(out.metrics["total_tokens"])
        nk = len(keys)
        res = {}
        if C.seed_check is not None:
            # the direct call with the trainer's own seed (-1 / N on every loss token) against the step's raw gradient: the backward of the
            # experts' GEMMs rounds its cotangent to bf16, so the gradient is linear in the seed only up to that rounding; with the SAME
            # seed the two paths should agree to fp32 accumulation, far below the 1e-3 seen with seeds 1 and -1 / N
            tt_ = {(int(nm.split(".")[0]), keys.index(nm.split(".")[1])): nm for nm in G}
            Gd, _, _ = run(C, tt_, C.engine_res, None, True, VARIANTS[var], seed_scale=float(np.float32(-1.0 / N)))
            RA.ROUND["on"] = True
            for nm in sorted(G):
                L, k = nm.split(".")
                raw = out.grads[int(L) * nk + keys.index(k)].detach().float().cpu()
                d = Gd.get(nm)
                C.seed_check[nm] = {"bitwise_equal": bool(d is not None and torch.equal(raw, d)),
                                    "rel_diff": None if d is None or float(raw.double().norm()) == 0 else float((raw.double() - d.double()).norm() / raw.double().norm()),
                                    "both_zero": bool(d is not None and float(raw.double().norm()) == 0 and float(d.double().norm()) == 0)}
        for nm in sorted(G):
            L, k = nm.split(".")
            gt = (-N * out.grads[int(L) * nk + keys.index(k)].double()).float().cpu()
            if float(G[nm].double().norm()) > 0:
                res[nm] = FS.cmp(gt, G[nm])
            else:
                res[nm] = {"cos": 1.0 if float(gt.double().norm()) == 0 else 0.0, "norm_ratio": 1.0 if float(gt.double().norm()) == 0 else float("inf"),
                           "rel_diff": float(gt.double().norm())}
        return res
    finally:
        RA.ROUND["on"] = False
        for k_, v_ in old_opts.items():
            setattr(cfg, k_, v_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--variants", default="H30,QD,H36,Q,HD30")
    ap.add_argument("--no-cos", action="store_true", help="skip the chain-state backwards")
    ap.add_argument("--seed-check", action="store_true", help="with --gate: also the direct call with the trainer's own seed, against the step's raw gradient")
    ap.add_argument("--gate", action="store_true", help="the trainer's finite-difference gate: verdict for the FIRST variant from the engine "
                    "follow of one tensor inside the loop (12.B13), one before it (3.B13) and the control (21.B13); exit 3 on a fail")
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

        def follow(var, label, parts, pp, rep_of=None):
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
                 "even": (S[1.0] + S[-1.0] - 2 * e0) / 2, "norm": n2 ** 0.5,
                 "readable": bool(abs(pred[1.0]) >= 4 * rep["scatter"] and abs(pred[1.0]) <= 0.81 * abs(e0) and rep.get("p_valid", True))}
            r["real_change_per_unit_step"] = r["f"] * r["norm"]
            r["repeat_of_p"] = rep_of
            rep["follow"].append(r)
            log(f"follow {var} {label} p {pp:.2f}: f {r['f']:.3f} (+ {r['f_plus']:.3f}, - {r['f_minus']:.3f}), |G| {r['norm']:.3f}, "
                f"f x |G| {r['real_change_per_unit_step']:.3f}, readable {r['readable']}" + (" (repeat at twice the step)" if rep_of else ""))
            save()
            if rep_of is None and r["readable"] and (0.5 <= r["f"] < 0.7 or 1.3 < r["f"] <= 1.5):
                follow(var, label, parts, 2 * pp, rep_of=pp)

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
                for nm in ("3.B13", "12.B13", "9.B13"):
                    for sg in (1.0, -1.0):
                        C.set_step(nm, G[nm], C.unit_step(G[nm], sg * 0.01))
                        devs.append(C.engine_S(vw) - e0)
                    C.set_step(nm, None, None)
                mean = sum(devs) / len(devs)
                rep["tiny_step_devs"] = devs
                rep["tiny_step_shift"] = mean
                rep["scatter_rms_about_zero"] = (sum(d * d for d in devs) / len(devs)) ** 0.5
                rep["scatter"] = (sum((d - mean) ** 2 for d in devs) / (len(devs) - 1)) ** 0.5
                # p: 6 x the scatter, at least 4 nats where the row allows it, and never above 0.4 |S0| (S <= 0: a step predicted near |S0|
                # cannot be followed upward, the + side saturates). A row where 6 x scatter exceeds 0.4 |S0| cannot be read: p_valid False.
                cap = 0.4 * abs(e0)
                rep["p"] = min(max(4.0, 6.0 * rep["scatter"]), cap)
                rep["p_valid"] = bool(6.0 * rep["scatter"] <= cap)
                log(f"tiny-step deviations {[round(d, 3) for d in devs]}; shift {mean:.3f}; scatter (sd) {rep['scatter']:.3f}; p {rep['p']:.2f}; "
                    f"0.4 |S0| {cap:.2f}; p valid {rep['p_valid']}")
                save()
            p = rep["p"]
            for L in (12, 3, 9, 15, 21):
                follow(var, f"{L}.B13", [(f"{L}.B13", G.get(f"{L}.B13"))], p)
            follow(var, "vo", [(f"{L}.{k}", G.get(f"{L}.{k}")) for L in (3, 9, 12, 15) for k in ("vB", "oB")], p)
            if var.startswith("H") or var == "full":
                follow(var, "qkr", [(f"{L}.{k}", G.get(f"{L}.{k}")) for L in (9, 12, 15) for k in ("qB", "kB", "rB")], p)
                follow(var, "qkr21", [(f"21.{k}", G.get(f"21.{k}")) for k in ("qB", "kB", "rB")], p)
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
    rc = 0
    if a.gate:
        if order[0] in GE and left() > 60:
            C.seed_check = {} if a.seed_check else None
            try:
                rep["trainer_path_vs_direct"] = trainer_path_check(C, order[0], GE[order[0]], keys)
                if C.seed_check is not None:
                    rep["trainer_path_vs_direct_same_seed"] = C.seed_check
                    rd = [x["rel_diff"] for x in C.seed_check.values() if x["rel_diff"] is not None]
                    log(f"same seed: bitwise equal on {sum(x['bitwise_equal'] for x in C.seed_check.values())} of {len(C.seed_check)} tensors; "
                        f"largest relative difference {max(rd) if rd else 0.0:.3g}")
                log("trainer path vs direct: " + " ".join(f"{nm} {x['cos']:.7f}/{x['norm_ratio']:.5f}" for nm, x in rep["trainer_path_vs_direct"].items()
                                                         if nm.endswith("B13") or nm.endswith("vB")))
            except Exception:
                import traceback

                rep["trainer_path_error"] = traceback.format_exc()[-2000:]
                rep["trainer_path_vs_direct"] = {"error": {"cos": 0.0, "norm_ratio": 0.0, "rel_diff": 0.0}}
                log("trainer path ERROR " + rep["trainer_path_error"][-500:])
        rep["gate"] = gate_verdict(rep, order[0])
        log(f"FINITE-DIFFERENCE GATE ({order[0]}): {'PASS' if rep['gate']['pass'] else 'FAIL'} {rep['gate']['readings']}")
        rc = 0 if rep["gate"]["pass"] else 3
    save()
    C.rig.w.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
