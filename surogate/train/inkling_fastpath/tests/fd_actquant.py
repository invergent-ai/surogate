"""How much of the gap between the serving forward and the trainer's recompute is the experts' ACTIVATION quantisation? One real row, one card.

The serving engine quantises the input of both expert GEMMs per token (NVFP4 in the 4-bit layers, MXFP8 in the 8-bit ones); the trainer's
recompute (ift_backward.moe_vec) feeds them unquantised values. Here a test-side copy of moe_vec quantises both inputs with the reference
quantisers and passes the gradient straight through ("A4"); the original is "A16".
 1 STATES   per visit, from the ENGINE's stored inputs (one visit, not chained): relative L2 distance of the recompute's output to the engine's
            stored output, A16 and A4.
 2 GRADIENT the pattern + gate constant backward (QD) with A16 and with A4 on the engine's states: cos and norm ratio per tensor, seconds.
 3 FOLLOW   the engine's follow (central difference, step rule of fd_variants) along each of the two gradients, B13 of layers 12, 3, 9, 15, 21.
LINES, written before any data: "the activation quantisation is the main part of the state distance" = the A4 distance is below half the
A16 distance at 36 of the 40 visits of the 4-bit layers (visits of layers 0-19). "It matters for the gradient" = cos(A4, A16) < 0.98 at a
gating tensor (12.B13, 3.B13, 9.B13, 15.B13); >= 0.98 at all four = modelling it would change the update by less than the follow can
read on one row. The follow of the two is reported side by side; with a per-reading noise of about 0.1 a difference below 0.2 is not read."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import fd_variants as FV  # noqa: E402
import torch  # noqa: E402

FS = FV.FS
TD = FS.TD
B, RA, TL, TV = FS.B, FS.RA, FS.TL, FS.TV
log = TD.log
QD = FV.VARIANTS["QD"]


def quant_ste(x, nv: bool):
    """The engine's per-row activation quantiser (values dequantised), gradient straight through."""
    with torch.no_grad():
        if nv:
            q = RA.inkling_quant_nvfp4_4o6(x)
            deq = q.values * q.outer[:, None]
        else:
            deq = RA.inkling_quant_mxfp8(x).values
    return x + (deq - x).detach()


def moe_vec_a4(x, w, ids, lora=None):
    """ift_backward.moe_vec with the two GEMM inputs quantised as the engine does (the LoRA deltas read the unquantised values, as there)."""
    x = x.float()
    t, h = x.shape
    ids = ids.long()
    lg = RA._router_logits(x, w.gate_w)
    gam = RA.route_weights(lg if B.GATE_GRAD["on"] else lg.detach(), ids)
    topk = ids.shape[1]
    nv = w.fmt == "nvfp4"
    xq = quant_ste(x, nv)
    flat = ids.reshape(-1)
    n = flat.shape[0]
    order = torch.argsort(flat, stable=True)
    es = flat[order]
    tok = order // topk
    slot = order % topk
    counts = torch.bincount(flat, minlength=B.E)
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.arange(n) - starts[es]
    cnt = counts.tolist()
    st = starts.tolist()
    gam_r = gam[tok, slot]
    dn_parts = []
    for g0 in range(0, B.E, B.EG):
        g1 = min(B.E, g0 + B.EG)
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

        gu = B.bmm32(pad(xq[tg], h), w, "w13", g0, g1)[eg, pg]
        if nv:
            gu = gu * RA.nvfp4_global(w.g13[g0:g1][eg], w.den13)[:, None]
        if lora is not None:
            la = torch.bmm(pad(x[tg], h), lora["A13"][g0:g1].transpose(1, 2))
            gu = gu + lora["scale"] * torch.bmm(la, lora["B13"][g0:g1].transpose(1, 2))[eg, pg]
        gu = RA.rb(gu)
        y = RA.inkling_softcap_swiglu(gu, gam_r[r0:r1])
        yq = quant_ste(y, nv)
        dn = B.bmm32(pad(yq, y.shape[1]), w, "w2", g0, g1)[eg, pg]
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


class a4:
    """Within the block the recompute's experts quantise their inputs."""

    def __enter__(self):
        self.old = B.moe_vec
        B.moe_vec = moe_vec_a4

    def __exit__(self, *e):
        B.moe_vec = self.old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, default=900.0)
    a, _ = ap.parse_known_args()
    deadline = time.time() + a.budget_s
    TD.ROW = a.row
    rep = {"row": a.row, "dropped": [], "follow": []}

    def save():
        json.dump(rep, open(a.out + ".tmp", "w"), indent=1)
        os.replace(a.out + ".tmp", a.out)

    C = FS.Ctx(a, keep_engine=True)
    dev, eng, W = C.dev, C.eng, C.W
    rep["T"], rep["loss_tokens"] = C.T, int(C.mask.sum())
    RA._E2M1_VALUES = RA._E2M1_VALUES.to(dev)                  # the reference quantiser's table, on the card
    keys = C.rig.keys
    tt = {(L, keys.index(k)): f"{L}.{k}" for L in FV.LAYERS for k in ("B13", "vB", "oB")}
    vw = C.view()
    try:
        # 1 states
        prod, _ = TL.producers(eng.visits)
        host = C.engine_res["host"]
        lora = []
        for d in C.rig.work:
            ks = [k for k in d if k != "scale"]
            lora.append(TV.lora_dict(ks, [d[k].to(dev) for k in ks], C.scale, C.attn_scale))
        RA.ROUND["on"] = True
        rep["states"] = []
        for v, vi in enumerate(eng.visits):
            n = len(vi.stack)
            r = {"v": v, "layer": int(vi.layer)}
            e = host[v].to(dev).float()
            for name in ("a16", "a4"):
                with torch.no_grad(), torch.device(dev):
                    ins = [host[p].to(dev).float() for p in prod[v]]
                    bank = (ins[n + 1], ins[n + 2]) if vi.looped else None
                    if name == "a4":
                        with a4():
                            o = TV.visit_torch_t8(W, vi, ins[:n], ins[n], bank, C.routing[v], lora[vi.layer])
                    else:
                        o = TV.visit_torch_t8(W, vi, ins[:n], ins[n], bank, C.routing[v], lora[vi.layer])
                    r[name] = float((o.float() - e).double().norm() / e.double().norm())
                    r[name + "_bitwise_share"] = float((o.to(torch.bfloat16) == e.to(torch.bfloat16)).float().mean())
                del ins, o
            rep["states"].append(r)
            del e
        RA.ROUND["on"] = False
        four = [r for r in rep["states"] if r["layer"] < 20]
        rep["states_line"] = {"visits_4bit": len(four), "a4_below_half_a16": sum(1 for r in four if r["a4"] < 0.5 * r["a16"])}
        log("states (visit: A16 / A4): " + " ".join(f"{r['v']}:{r['a16']:.4f}/{r['a4']:.4f}" for r in rep["states"]))
        log(f"A4 below half of A16 at {rep['states_line']['a4_below_half_a16']} of {rep['states_line']['visits_4bit']} visits of the 4-bit layers")
        save()
        # 2 gradients
        G16, s16, _ = FV.run(C, tt, C.engine_res, None, True, QD)
        with a4():
            G4, s4, _ = FV.run(C, tt, C.engine_res, None, True, QD)
        rep["backward_s"] = {"a16": s16, "a4": s4}
        rep["a4_vs_a16"] = {nm: FS.cmp(G4[nm], G16[nm]) for nm in sorted(G16) if nm in G4}
        log(f"backward A16 {s16:.0f} s, A4 {s4:.0f} s; cos A4 vs A16: " + " ".join(f"{nm} {x['cos']:.4f}/{x['norm_ratio']:.3f}" for nm, x in rep["a4_vs_a16"].items()))
        save()
        # 3 follow
        e0 = C.engine_S(vw)
        devs = []
        for nm in ("3.B13", "12.B13", "9.B13"):
            for sg in (1.0, -1.0):
                C.set_step(nm, G16[nm], C.unit_step(G16[nm], sg * 0.01))
                devs.append(C.engine_S(vw) - e0)
            C.set_step(nm, None, None)
        mean = sum(devs) / len(devs)
        sd = (sum((d - mean) ** 2 for d in devs) / (len(devs) - 1)) ** 0.5
        p = min(max(4.0, 6.0 * sd), 0.4 * abs(e0))
        rep.update({"S0_engine": e0, "scatter": sd, "shift": mean, "p": p, "p_valid": bool(6.0 * sd <= 0.4 * abs(e0))})
        log(f"S0 {e0:.3f}, scatter {sd:.3f}, shift {mean:.3f}, p {p:.2f}, valid {rep['p_valid']}")
        for L in (12, 3, 9, 15, 21):
            nm = f"{L}.B13"
            for name, G in (("a16", G16[nm]), ("a4", G4[nm])):
                if time.time() > deadline:
                    rep["dropped"].append([nm, name])
                    continue
                st = C.unit_step(G, p)
                p1, _ = C.set_step(nm, G, st)
                sp = C.engine_S(vw)
                p2, _ = C.set_step(nm, G, -st)
                sm = C.engine_S(vw)
                C.set_step(nm, None, None)
                r = {"tensor": nm, "recompute": name, "p": p, "f": (sp - sm) / (p1 - p2), "norm": float(G.double().norm())}
                rep["follow"].append(r)
                log(f"follow {nm} {name}: f {r['f']:.3f}, |G| {r['norm']:.3f}")
                save()
    except Exception:
        import traceback

        rep["error"] = traceback.format_exc()[-3000:]
        log("ERROR " + rep["error"][-900:])
        save()
        C.rig.w.close()
        return 1
    save()
    C.rig.w.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
