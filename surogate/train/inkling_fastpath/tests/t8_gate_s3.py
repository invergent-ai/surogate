"""Lane T8, S3 card gate (PLAN-SKYRL.md): attention LoRA r64 / alpha 32 (scale 0.5) on the seven S10 slices. PASS LINES, written before any run (also in
lanes/T8/NOTES.md; REVISED 12:3x after the Opus review, still before any run). Nothing is trained; no adapter is kept. Adapter: attention A = the
trainer's kaiming init, B = N(0, 0.2) (seed 13; large, so that each delta is far above one bf16 ulp of the base); experts at the zero adapter.
Row: the first 4,096 tokens of gate_rows.npz row 1.
  G13 forward wiring, every one of the 48 visits, on the SAME visit inputs: (a) run with no attention adapter, (b) qkvr adapter only, (c) qkvr + wo_ud.
      qkvr(b) == bf16(qkvr(a) + (x . a^T) . b^T) per slice from that slice's own input stream (x_q for q / r / gate / sink, x_k for k, x_v for v);
      attn_partial(c) == bf16(attn_partial(b) + (gated(b) . a_o^T) . b_o^T). Line: >= 99.9 % of elements bitwise equal and every element within one bf16
      ulp (the engine sums in fp32 fma chains, torch in another order); the deltas are non-zero; and the T8 shim's base forward (no adapter) gives
      the ROLLOUT's recorded log-probs bitwise at the loss tokens (the copy changed nothing else; review M2).
      AMENDED after run s3-1 (13:0x), pending the coordinator's approval, both reported against BOTH lines: (i) the base check runs on the WHOLE
      16,009-token gate row 0 (its loss tokens start at 14,404; s3-1 compared a 4,096-token prefix that holds no loss token, i.e. against the
      export's zeros: invalid by construction); (ii) "every element within one bf16 ulp" is replaced by "every element within one bf16 ulp OR within
      1e-5 x the slice's largest |delta|": where base + delta nearly cancels, the result's ulp is far below the fp32 rounding of a 64-term delta
      (s3-1: >= 99.98 % of elements bitwise in every slice, 243 of 288 slices had some element beyond one ulp).
      APPROVED by the coordinator 13:1x with conditions, fixed here BEFORE the rerun: per slice the worst element's distance in ulps and the share of
      elements beyond one ulp are reported; the cancellation explanation is checked on the elements beyond one ulp: it HOLDS iff, in every slice
      that has such elements, the median of |want| / max(|base|, |delta|) over them is <= 0.25 (|base + delta| small against its parts); if it does not
      hold, G13 FAILS whatever the amended line says.
  G15 the TRAINER's own pieces against the engine (review M1), visits 0, 5, 18, 30, 41, 47: t8_visit.qkvr_raw / wo_raw on the engine's own visit inputs
      (attn_in tap of run (a); gated tap of run (b)), with the slices taken from the trainer's WORK values through t8_visit.lora_dict (the one key-to-slice
      map the backward uses) and the trainer's scale: (torch with adapter - torch without) vs (engine (b) - engine (a)) per qkvr slice and
      (torch wo with - without) vs (engine (c) - engine (b)): every slice cos >= 0.9999, rel err <= 1 %.
  G14 gradients of the attention keys through the lean K = 48 backward: finite, non-zero for every slice of every layer; A/A bitwise on one card;
      TokenChunked in chunks of 1,024 vs the same recompute in one chunk for visits 47, 41 (global), 30 (looped core), 5: cos >= 0.9999, rel <= 1 %
      (not 1e-5: a different chunk width may pick other GEMM kernels and flip bf16 roundings; review M3).
  G14c (diagnostic, no line): the engine's change of the masked log-prob sum along -grad of layer 23's attention B (L2 1e-3) against the first-order prediction.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import t8_gate as GT  # noqa: E402

BF = torch.bfloat16
D = 2304


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ulp_stats(got, want, base, delta):
    """per slice: worst distance in ulps of `want`, share of elements beyond one ulp, and the cancellation ratio median |want| / max(|base|, |delta|)
    over those elements (None when there are none)."""
    w = want.to(BF)
    wf = w.float()
    up = (torch.nextafter(w, torch.full_like(w, float("inf"))).float() - wf).abs()
    dn = (wf - torch.nextafter(w, torch.full_like(w, float("-inf"))).float()).abs()
    ulp = torch.maximum(up, dn).clamp(min=2 ** -133)
    d = (got - want).abs() / ulp
    beyond = d > 1.0001
    share = float(beyond.float().mean())
    ratio = None
    if bool(beyond.any()):
        den = torch.maximum(base.abs(), delta.abs()).clamp(min=1e-30)
        ratio = float((want.abs() / den)[beyond].median())
    return {"worst_ulps": float(d.max()), "share_beyond_one_ulp": share, "cancel_ratio_median": ratio}


def ulp_cmp(got, want, delta=None):
    """got / want bf16-valued fp32: fraction bitwise equal, and whether every element is within one bf16 ulp of `want`
    (with `delta`: one ulp OR 1e-5 x max |delta| of the slice; the amended line)."""
    eq = float((got == want).float().mean())
    w = want.to(BF)
    wf = w.float()
    up = (wf + (torch.nextafter(w, torch.full_like(w, float("inf"))).float() - wf).abs())
    dn = (wf - torch.nextafter(w, torch.full_like(w, float("-inf"))).float()).abs()
    ulp = torch.maximum((up - wf).abs(), dn).clamp(min=2 ** -133)
    err = (got - want).abs()
    if delta is None:
        return eq, bool((err <= ulp * 1.0001).all())
    tol = torch.maximum(ulp * 1.0001, torch.full_like(ulp, 1e-5 * float(delta.abs().max())))
    return eq, bool((err <= tol).all())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import ift_backward as B
    import ift_forward as F
    import ift_long as LG
    import t8_engine as TE
    import t8_lean as TL
    import t8_step as TS
    import t8_worker as TW
    from ift_engine import Segment, VisitTaps

    TS.verify_stack("A0")
    rows = GT.gate_rows(a.rows)
    ids_full = rows[1].ids[:4096]
    mask = rows[1].loss_mask[:4096].copy()
    if not mask.any():
        mask[2048:] = True
    eng = TE.T8Engine(a.bin, a.artifact, 0)
    dev = eng.dev
    W = B.EngineWeights(eng)
    shapes = TW.param_shapes(64)
    flat = torch.zeros(sum(int(np.prod(v)) for d in shapes for v in d.values()), dtype=torch.float32)
    lora = TW.work_views(flat, shapes)
    init = B.make_lora("cpu", seed=0, b_std=0.0)
    gA, gB = torch.Generator().manual_seed(1000), torch.Generator().manual_seed(13)
    for d, lo in zip(lora, init):
        for k in TS.LORA_KEYS:
            d[k].copy_(lo[k].detach().to(BF).float())
        for s in TS.ATTN_SLICES:
            A = d[f"{s}A"]
            A.copy_(((torch.rand(A.shape, generator=gA) * 2 - 1) / A.shape[1] ** 0.5).to(BF).float())
            d[f"{s}B"].copy_((torch.randn(d[f"{s}B"].shape, generator=gB) * 0.2).to(BF).float())
    banks = B.EngineBanks([{k: v.to(dev) if torch.is_tensor(v) else v for k, v in d.items()} for d in lora], dev)
    attn = TW.AttnBanks([{k: v.to(dev) if torch.is_tensor(v) else v for k, v in d.items()} for d in lora], dev, 64, 32.0)
    T = ids_full.shape[0]
    ids = torch.as_tensor(ids_full, dtype=torch.int32, device=dev)
    rep = {"G13": [], "T": T}
    # ---------------------------------------------------------------- G13: three runs of each visit on the same inputs
    x = torch.zeros(T, eng.c["residual"], dtype=BF, device=dev)
    eng.attn_layers = None
    ws = eng.workspace(T, True)
    seg = Segment(T, dev)
    eng.embed(ids, x, ws)
    qkvr_only = []
    for ad in attn.adapters:
        q = TE.AttnAdapters()
        q.qkvr_table, q.wo_table, q.uniform_slot, q.rank = ad.qkvr_table, None, ad.uniform_slot, ad.rank
        qkvr_only.append(q)
    sl = {"q": (0, 3072), "r": (3072, 384), "g": (3456, 3072), "s": (6528, 24), "k": (6552, 1024), "v": (7576, 1024)}
    ok = True
    ok2 = True
    cancel_ok = True
    import t8_visit as TV
    keys = TS.param_keys(64)
    rep["G15"] = []
    ok15 = True
    for v in range(48):
        vi = eng.visits[v]
        Q = 3 if (vi.global_ or vi.looped) else 1
        outs = {}
        for name, layers in (("a", None), ("b", qkvr_only), ("c", attn.adapters)):
            xx = x.clone()
            seg.conv.zero_()
            taps, bufs = F.make_taps(eng, v, T, ("attn_in", "qkvr", "gated", "attn_partial"))
            eng.attn_layers = layers
            eng.run_visit(v, xx, T, seg, ws, None, taps, None)
            torch.cuda.synchronize()
            outs[name] = ({k: t.float() for k, t in bufs.items()}, xx)
        eng.attn_layers = None
        x = outs["a"][1]                                     # carry the base forward to the next visit
        ta, tb, tc_ = outs["a"][0], outs["b"][0], outs["c"][0]
        st = attn.layers[vi.layer]["store"]
        ain = ta["attn_in"]                                  # [Q, T, D]
        row = {"v": v}
        for s, (r0, n) in sl.items():
            stream = (0 if s in "qrgs" else (1 if s == "k" else 2)) if Q == 3 else 0
            xin = ain[stream]
            af, bf = st[s][0].float(), st[s][1].float()
            want = ((ta["qkvr"][:, r0:r0 + n] + (xin @ af.T) @ bf.T).to(BF)).float()
            eq, within = ulp_cmp(tb["qkvr"][:, r0:r0 + n], want)
            dl = (xin @ af.T) @ bf.T
            _, within2 = ulp_cmp(tb["qkvr"][:, r0:r0 + n], want, dl)
            us = ulp_stats(tb["qkvr"][:, r0:r0 + n], want, ta["qkvr"][:, r0:r0 + n], dl)
            if us["cancel_ratio_median"] is not None and us["cancel_ratio_median"] > 0.25:
                cancel_ok = False
            nz = float((tb["qkvr"][:, r0:r0 + n] != ta["qkvr"][:, r0:r0 + n]).float().mean())
            row[s] = (eq, within, nz, within2, us)
            ok = ok and eq >= 0.999 and within and nz > 0
            ok2 = ok2 and eq >= 0.999 and within2 and nz > 0
        af, bf = st["o"][0].float(), st["o"][1].float()
        want = ((tb["attn_partial"] + (tb["gated"] @ af.T) @ bf.T).to(BF)).float()
        eq, within = ulp_cmp(tc_["attn_partial"], want)
        dl = (tb["gated"] @ af.T) @ bf.T
        _, within2 = ulp_cmp(tc_["attn_partial"], want, dl)
        us = ulp_stats(tc_["attn_partial"], want, tb["attn_partial"], dl)
        if us["cancel_ratio_median"] is not None and us["cancel_ratio_median"] > 0.25:
            cancel_ok = False
        nz = float((tc_["attn_partial"] != tb["attn_partial"]).float().mean())
        row["o"] = (eq, within, nz, within2, us)
        ok = ok and eq >= 0.999 and within and nz > 0
        ok2 = ok2 and eq >= 0.999 and within2 and nz > 0
        rep["G13"].append(row)
        if v in (0, 5, 18, 30, 41, 47):                      # G15: the trainer's own pieces (work values, key map, scale, streams)
            Lw = W.layers[vi.layer]
            ld = TV.lora_dict(keys, [lora[vi.layer][k].to(dev) for k in keys], B.ALPHA / B.RANK, 0.5)
            xq_, xk_, xv_ = ain[0], (ain[1] if Q == 3 else ain[0]), (ain[2] if Q == 3 else ain[0])
            with torch.no_grad():
                t0_ = TV.qkvr_raw(Lw, xq_, xk_, xv_, None, None)
                t1_ = TV.qkvr_raw(Lw, xq_, xk_, xv_, ld["attn"], 0.5)
                w0_ = TV.wo_raw(Lw, tb["gated"], None, None)
                w1_ = TV.wo_raw(Lw, tb["gated"], ld["attn"]["o"], 0.5)
            r15 = {"v": v}
            for (s, (r0, n)), a0, a1 in zip(sl.items(), t0_, t1_):
                c = GT.compare({0: (a1 - a0).float().reshape(T, -1).cpu()}, {0: (tb["qkvr"][:, r0:r0 + n] - ta["qkvr"][:, r0:r0 + n]).cpu()})
                r15[s] = (c["worst_cos"], c["worst_rel"])
                ok15 = ok15 and c["pass"]
            c = GT.compare({0: (w1_ - w0_).float().cpu()}, {0: (tc_["attn_partial"] - tb["attn_partial"]).cpu()})
            r15["o"] = (c["worst_cos"], c["worst_rel"])
            ok15 = ok15 and c["pass"]
            rep["G15"].append(r15)
            log(f"G15 visit {v}: {r15}")
        if v in (0, 5, 18, 41, 47):
            log(f"G13 visit {v}: {row}")
    del x, seg, ws
    eng._ws = None
    # (a) == the T shim's base forward: compare the base log-probs of the T8 engine (no adapters) with lane T's engine
    r0_ = rows[0]                                            # (i) the whole row 0: its loss tokens start at 14,404
    ids0 = torch.as_tensor(np.ascontiguousarray(r0_.ids), dtype=torch.int32, device=dev)
    with torch.no_grad():
        r8 = F.forward(eng, ids0, save=False)
        lp8 = F.logprobs(eng, r8["hidden"], ids0).cpu()
    del r8, ids0
    eng._ws = None
    torch.cuda.empty_cache()
    m1 = r0_.loss_mask[1:]
    want_lp = np.asarray(r0_.engine_lp[1:], dtype=np.float32)
    got_lp = lp8.numpy().astype(np.float32)
    base_bad = int((got_lp[m1].view(np.int32) != want_lp[m1].view(np.int32)).sum())
    rep["G13_base_vs_rollout"] = {"loss_tokens": int(m1.sum()), "mismatches": base_bad}
    log(f"G13 base forward of the T8 shim vs the rollout: {base_bad} mismatches of {int(m1.sum())}")
    rep["G13_pass_as_registered"] = bool(ok and base_bad == 0 and int(m1.sum()) > 0)
    rep["G13_cancellation_explanation_holds"] = bool(cancel_ok)
    rep["G13_pass"] = bool(ok2 and cancel_ok and base_bad == 0 and int(m1.sum()) > 0)   # the amended line (approved 13:1x) + the cancellation check
    log(f"G13: registered line {rep['G13_pass_as_registered']}; amended line + cancellation check {rep['G13_pass']} (cancellation holds: {cancel_ok}); base {base_bad} of {int(m1.sum())}")
    rep["G15_pass"] = bool(ok15 and len(rep["G15"]) == 6)
    # ---------------------------------------------------------------- G14: gradients of the attention keys
    eng.attn_layers = attn.adapters
    host = [{k: v for k, v in d.items()} for d in lora]
    mask_t = torch.as_tensor(mask[1:].astype(np.float32), device=dev)
    grads = {}
    ALL = tuple(range(len(TS.param_keys(64))))             # every key trained (review B1: the default trains the 4 expert keys only)

    def run_bw():
        out = {}
        buf = TL.RowBuffers(eng.visits, 0, T)
        vals = TL.forward_keep(eng, ids, [L["adapters"] for L in banks.layers], buf)

        def on(vis, layer, j, gj):
            k = (layer, j)
            c = gj.float().cpu()
            out[k] = c if k not in out else out[k].add_(c)

        TL.backward_row(eng, W, ids, vals, host, B.ALPHA / B.RANK, 0, mask_t, buf, on, q_block=128, attn_scale=0.5, train_idx=ALL)
        buf.close()
        return out

    g1 = run_bw()
    g2 = run_bw()
    keys = TS.param_keys(64)
    aa = all(torch.equal(g1[k], g2[k]) for k in g1) and set(g1) == set(g2)
    att_idx = [keys.index(k) for k in TS.ATTN_KEYS]
    nonzero = all((L, j) in g1 and float(g1[(L, j)].abs().max()) > 0 and bool(torch.isfinite(g1[(L, j)]).all()) for L in range(24) for j in att_idx
                  if not keys[j].endswith("A") or True)
    rep["G14"] = {"aa_bitwise": bool(aa), "attn_grads_nonzero_finite": bool(nonzero)}
    # chunked == one chunk, per visit (the recompute of one visit with its host inputs)
    chunk_cmp = {}
    real = LG.CHUNK
    for v in (47, 41, 30, 5):
        per = {}
        for ch in (1024, T):
            LG.CHUNK = ch
            o = {}
            buf = TL.RowBuffers(eng.visits, v, T)
            vals = TL.forward_keep(eng, ids, [L["adapters"] for L in banks.layers], buf)

            def on(vis, layer, j, gj, o=o, v=v):
                if vis == v:
                    o[(layer, j)] = gj.float().cpu()

            TL.backward_row(eng, W, ids, vals, host, B.ALPHA / B.RANK, v, mask_t, buf, on, q_block=128, attn_scale=0.5, train_idx=ALL)
            buf.close()
            per[ch] = {k: t for k, t in o.items() if k[1] in att_idx}
        c = GT.compare(per[1024], per[T])
        chunk_cmp[v] = c
        log(f"G14 visit {v}: chunk 1024 vs one chunk: {c}")
    LG.CHUNK = real
    rep["G14"]["chunked_vs_one"] = {str(k): v for k, v in chunk_cmp.items()}
    rep["G14_pass"] = bool(aa and nonzero and all(c["pass"] for c in chunk_cmp.values()))
    # ---------------------------------------------------------------- G14c: the engine along -grad of layer 23's attention B (diagnostic)
    try:
        dirs = {k: -g1[(23, keys.index(k))] for k in TS.ATTN_KEYS if k.endswith("B")}
        nrm = sum(float(t.double().pow(2).sum()) for t in dirs.values()) ** 0.5
        eps = 1e-3 / nrm
        pred = -eps * sum(float(g1[(23, keys.index(k))].double().pow(2).sum()) for k in dirs)
        with torch.no_grad():
            def lp_sum():
                r = F.forward(eng, ids, save=False, adapters_per_layer=[L["adapters"] for L in banks.layers])
                return float((F.logprobs(eng, r["hidden"], ids) * mask_t).double().sum())
            base = lp_sum()
            for k, t in dirs.items():
                host[23][k] += (t * eps)
            attn.refresh([{kk: vv.to(dev) if torch.is_tensor(vv) else vv for kk, vv in d.items()} for d in host])
            moved = lp_sum()
        rep["G14c"] = {"predicted": pred, "engine": moved - base, "ratio": (moved - base) / pred if pred else None}
        log(f"G14c: {rep['G14c']}")
    except Exception as e:  # diagnostic only
        rep["G14c"] = {"error": str(e)[:300]}
    rep["base_lp_sum_t8"] = float(lp8.double().sum())
    rep["pass"] = bool(rep["G13_pass"] and rep["G14_pass"] and rep["G15_pass"])
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    eng.close()
    print(f"T8 GATE S3: {'PASS' if rep['pass'] else 'FAIL'}", flush=True)
    return 0 if rep["pass"] else 4


if __name__ == "__main__":
    sys.exit(main())
