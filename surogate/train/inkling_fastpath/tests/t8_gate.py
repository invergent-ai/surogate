"""the U1 card gate. PASS LINES, written before any run (also in the design notes):
Phase "lean" (one card; rows of 16k and 28k; plain head):
  G1  the lean K = 48 backward == the stock path (ift_long.LongModel) on every LoRA tensor.
  G2  the truncated backward at K = 30 / 18 / 6 == the sum of the last K visits' contributions of the full lean backward.
Phase "trainer" (two cards; four real rows of 16k / 28k / 47k / 57k; K = 18, centred head, seeds "weights", B keys):
  G3  each row's gradient on card A == on card B.
  G4  the two-worker step's gradient == the sum of the rows' gradients; its segment sums add up to it.
  G5  the hard gates hold on real rows: replay 0 mismatches, support gate, zero-adapter check.
  G6  host peak per worker <= 20 GiB with the 57k row; no row out of memory at query block 8.
Phase "g7" (one card; rows of 16k / 28k / 47k; centred head, seeds "weights", B keys):
  G7  the step's gradient at K = 48 and K = 18 == the FROZEN reference gradient worker's gradient on the same rows (an external reference script on a frozen snapshot, fp32 A): per tensor,
      per segment (A, B, C, D1, D0; at K = 18: A, B) and for the per-layer totals. Line as G1-G3. IF G7 FAILS THE STEP IS NOT CLEARED FOR THE LOOP. No other statistic is read.
  G7c = the same line and comparison on the 57k row (--g7-rows 3), run inside the 8-card gate.
  G7b (diagnostic, no line): the same lean path with the reference gradient worker's fp32 A at K = 18 on the 16k and 28k rows against the reference: is A's bf16 rounding the only difference?
Line for G1-G3 and G7: every tensor cos >= 0.9999 and relative error <= 1 % (G4: 1e-5), all finite. Whether a comparison is BITWISE is reported beside it; bitwise is expected for G1-G3.
No adapter is saved; nothing is trained.
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
GIB = 2 ** 30
COS, REL = 0.9999, 0.01


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cmp_t(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    nb = float(b.norm())
    na = float(a.norm())
    return (float((a * b).sum()) / (na * nb) if na > 0 and nb > 0 else (1.0 if na == nb else 0.0),
            float((a - b).norm()) / nb if nb > 0 else (0.0 if na == 0 else float("inf")))


def compare(xs: dict, ys: dict, rel_line=REL) -> dict:
    """Over the union of keys; a key missing on one side counts as a zero tensor there (it must then be zero on the other). An empty or all-zero comparison does NOT pass."""
    worst_cos, worst_rel, n, bit, nonzero, shapes = 1.0, 0.0, 0, True, 0, True
    finite = True
    for k in sorted(set(xs) | set(ys), key=str):
        a, b = xs.get(k), ys.get(k)
        a = torch.zeros_like(b) if a is None else a
        b = torch.zeros_like(a) if b is None else b
        n += 1
        if a.shape != b.shape:
            shapes, bit = False, False
            continue
        finite = finite and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
        bit = bit and torch.equal(a, b)
        if float(b.double().norm()) == 0 and float(a.double().norm()) == 0:
            continue
        nonzero += 1
        c, r = cmp_t(a, b)
        worst_cos, worst_rel = min(worst_cos, c), max(worst_rel, r)
    return {"tensors": n, "nonzero": nonzero, "worst_cos": worst_cos, "worst_rel": worst_rel, "bitwise": bool(bit), "finite": bool(finite), "shapes_equal": bool(shapes),
            "pass": bool(finite and shapes and nonzero > 0 and worst_cos >= COS and worst_rel <= rel_line)}


def gate_rows(path):
    from surogate.inkling_rl.records import Row

    z = np.load(path)
    rows = []
    for n in range(sum(1 for k in z.files if k.startswith("ids_"))):
        ids, mask, lp = z[f"ids_{n}"].astype(np.int64), z[f"mask_{n}"].astype(bool), z[f"lp_{n}"].astype(np.float64)
        notes = {"weight": 1.0}
        if f"fin_{n}" in z.files:
            notes["final_mask"] = z[f"fin_{n}"].astype(bool)
        rows.append(Row(ids=ids, loss_mask=mask, engine_lp=lp, advantage=0.0, versions=None, teacher_lp=None, routed=None, notes=notes))
    return rows


# ---------------------------------------------------------------------------------------------------------------------
def phase_lean(a, rows):
    import ift_backward as B
    import ift_engine as E
    import ift_long as LG
    import t8_lean as TL
    import t8_step as TS

    rep_stack = TS.verify_stack("A0")
    eng = E.InklingEngine(a.bin, a.artifact, 0)
    W = B.EngineWeights(eng)
    lora = B.make_lora(eng.dev, seed=0, b_std=0.0)
    with torch.no_grad():                                     # the kit's work values: bf16(master)
        for lo in lora:
            for k in B.LORA_KEYS:
                lo[k].copy_(lo[k].to(torch.bfloat16).float())
    banks = B.EngineBanks(lora, eng.dev)
    adapters = [L["adapters"] for L in banks.layers]
    tf = LG.TrainForwardLong(eng, W)
    params = [lo[k] for lo in lora for k in B.LORA_KEYS]
    host = [{k: lo[k].detach().cpu() for k in B.LORA_KEYS} for lo in lora]
    scale = lora[0]["scale"]
    rep = {"rows": [], "stack": rep_stack}
    for r in rows[:2]:
        T = r.T
        ids = torch.as_tensor(np.ascontiguousarray(r.ids), dtype=torch.int32, device=eng.dev)
        g = torch.as_tensor(r.loss_mask[1:].astype(np.float32), device=eng.dev)
        out = {"T": T}
        # stock
        t0 = time.time()
        res = LG.forward_host(eng, ids, adapters)
        lp_stock = res["lp"].clone()
        lp = LG.LongModel.apply(tf, ids, res, scale, *params)
        (g * lp).sum().backward()
        torch.cuda.synchronize()
        out["stock_s"] = time.time() - t0
        stock = {}
        for i, p in enumerate(params):
            if p.grad is not None:
                stock[(i // 4, i % 4)] = p.grad.detach().cpu()
                p.grad = None
        del lp, res
        LG._PINNED.clear()
        torch.cuda.empty_cache()
        torch._C._host_emptyCache()
        # lean, K = 48, with the partial sums of the last K visits
        buf = TL.RowBuffers(eng.visits, 0, T)
        t0 = time.time()
        v = TL.forward_keep(eng, ids, adapters, buf)
        out["lp_bitwise_vs_stock"] = bool(torch.equal(v["lp"], lp_stock))
        full, last = {}, {K: {} for K in (30, 18, 6)}

        def on48(vis, layer, j, gj):
            c = gj.float().cpu()
            for K, d in last.items():
                if vis >= 48 - K:
                    d[(layer, j)] = c.clone() if (layer, j) not in d else d[(layer, j)].add_(c)
            full[(layer, j)] = c if (layer, j) not in full else full[(layer, j)].add_(c)

        TL.backward_row(eng, W, ids, v, host, scale, 0, g, buf, on48, q_block=128)
        out["lean48_s"] = time.time() - t0
        out["G1"] = compare(full, stock)
        del v
        buf.close()
        torch.cuda.empty_cache()
        log(f"T {T}: G1 {out['G1']}; stock {out['stock_s']:.1f} s, lean {out['lean48_s']:.1f} s ({T / out['lean48_s']:.0f} tok/s)")
        # truncated
        out["G2"] = {}
        for K in (30, 18, 6):
            buf = TL.RowBuffers(eng.visits, 48 - K, T)
            t0 = time.time()
            v = TL.forward_keep(eng, ids, adapters, buf)
            got = {}

            def onk(vis, layer, j, gj):
                c = gj.float().cpu()
                got[(layer, j)] = c if (layer, j) not in got else got[(layer, j)].add_(c)

            TL.backward_row(eng, W, ids, v, host, scale, 48 - K, g, buf, onk, q_block=128)
            sec = time.time() - t0
            c = compare(got, last[K])
            c.update({"seconds": sec, "tok_per_s": T / sec, "arena_gib": buf.bytes / GIB})
            out["G2"][str(K)] = c
            del v
            buf.close()
            torch.cuda.empty_cache()
            log(f"T {T}: G2 K={K} {c}")
        rep["rows"].append(out)
        del ids, g, stock, full, last
        torch.cuda.empty_cache()
    rep["pass"] = all(o["G1"]["pass"] and o["lp_bitwise_vs_stock"] and all(x["pass"] for x in o["G2"].values()) for o in rep["rows"])
    eng.close()
    return rep


# ---------------------------------------------------------------------------------------------------------------------
def b_grads(out):
    import t8_step as TS

    d = {}
    for i, g in enumerate(out.grads):
        if TS.LORA_KEYS[i % 4] in ("B13", "B2"):
            d[(i // 4, i % 4)] = g
    return d


def phase_trainer(a, rows):
    from surogate.inkling_rl.step import StepConfig
    import t8_step as TS

    cards = [int(x) for x in a.cards.split(",")]
    base = dict(bin_dir=a.bin, artifact=a.artifact, K=18, head="centred", max_row_tokens=58010, segment_keys=("B13", "B2"), train_keys=("B13", "B2"),
                run_dir=a.run_dir)
    scfg = StepConfig(max_policy_lag=None)
    rep = {}
    per = {}
    for c in cards:
        tr = TS.T8Trainer(TS.T8Config(cards=(c,), **base))
        try:
            if c == cards[0]:
                rep["zero_adapter"] = tr.check_zero_adapter(rows[:1])
            per[c] = []
            for i, r in enumerate(rows):
                o = tr.compute([r], 0, scfg, seeds="weights")
                assert o.sense == "ascent"
                per[c].append({k: v.clone() for k, v in b_grads(o).items()})
                log(f"card {c} row {i} (T {r.T}): {o.rows[0]['seconds']:.1f} s = {r.T / o.rows[0]['seconds']:.0f} tok/s, block {o.rows[0]['q_block']}, retries {o.rows[0]['oom_retries']}; worker {o.metrics['workers'][0]}")
                rep.setdefault(f"one_card_{c}", []).append({"T": r.T, "seconds": o.rows[0]["seconds"], "q_block": o.rows[0]["q_block"], "oom_retries": o.rows[0]["oom_retries"],
                                                            "replay_checked": o.rows[0]["replay_checked"], "decision_tokens": o.rows[0]["decision_tokens"], "worker": o.metrics["workers"][0]})
        finally:
            tr.close()
    rep["G3"] = [compare(x, y) for x, y in zip(per[cards[0]], per[cards[1]])]
    log(f"G3 {rep['G3']}")
    tr = TS.T8Trainer(TS.T8Config(cards=tuple(cards), **base))
    try:
        o = tr.compute(list(rows), 0, scfg, seeds="weights")
    finally:
        tr.close()
    want = {}
    for d in per[cards[0]]:
        for k, v in d.items():
            want[k] = v.double().clone() if k not in want else want[k].add_(v.double())
    got = b_grads(o)
    rep["G4"] = compare(got, {k: v.float() for k, v in want.items()}, rel_line=1e-5)
    segsum = {}
    for (part, seg, L, key), t in o.segments.items():
        k = (L, TS.LORA_KEYS.index(key))
        segsum[k] = t.double().clone() if k not in segsum else segsum[k].add_(t.double())
    rep["G4_segments"] = compare({k: v.float() for k, v in segsum.items()}, got, rel_line=1e-5)
    rep["step"] = {k: o.metrics[k] for k in ("row_tokens", "wall_s", "tok_per_s_per_card", "engine_mismatch_kl", "total_tokens", "workers")}
    rep["step_rows"] = [{k: r[k] for k in ("i", "T", "worker", "seconds", "q_block", "oom_retries")} for r in o.rows]
    log(f"G4 {rep['G4']}; segments {rep['G4_segments']}; step {rep['step']}")
    peak = max(w["VmHWM_gib"] for x in cards for w in [e["worker"] for e in rep[f"one_card_{x}"]]) if cards else 0
    peak = max(peak, max(w["VmHWM_gib"] for w in o.metrics["workers"]))
    rep["host_peak_per_worker_gib"] = peak
    rep["G5"] = bool(rep.get("zero_adapter")) and all(e["replay_checked"] for x in cards for e in rep[f"one_card_{x}"])
    rep["G6"] = bool(peak <= 20.0)
    rep["pass"] = all(x["pass"] for x in rep["G3"]) and rep["G4"]["pass"] and rep["G4_segments"]["pass"] and rep["G5"] and rep["G6"]
    return rep


G7_LAYERS = {"A": range(18, 24), "B": range(6, 18), "C": range(6, 18), "D1": range(6, 18), "D0": range(0, 6)}


def g7_compare(mine: dict, ref: dict, segments) -> dict:
    """Per segment and for the per-(layer, key) totals over `segments`. Keys: (segment, layer, key). Vacuous passes are refused: exactly the expected keys on both sides, equal shapes,
    every tensor non-zero on both sides."""
    expected = {(sg, L, k) for sg in segments for L in G7_LAYERS[sg] for k in ("B13", "B2")}
    r = {k: v for k, v in ref.items() if k[0] in segments}
    problems = []
    if set(mine) != expected:
        problems.append(f"step keys differ from the expected {len(expected)}: missing {len(expected - set(mine))}, extra {len(set(mine) - expected)}")
    if set(r) != expected:
        problems.append(f"reference keys differ from the expected {len(expected)}: missing {len(expected - set(r))}, extra {len(set(r) - expected)}")
    for k in sorted(expected & set(mine) & set(r), key=str):
        if mine[k].shape != r[k].shape:
            problems.append(f"{k}: shapes {tuple(mine[k].shape)} / {tuple(r[k].shape)}")
        elif float(mine[k].double().norm()) == 0 or float(r[k].double().norm()) == 0:
            problems.append(f"{k}: a zero tensor")
    out = {"segments": {}, "problems": problems[:12], "expected_tensors": len(expected)}
    for sg in segments:
        out["segments"][sg] = compare({k: v for k, v in mine.items() if k[0] == sg}, {k: v for k, v in r.items() if k[0] == sg})

    def total(d):
        t = {}
        for (sg, L, key), v in d.items():
            if sg in segments:
                t[(L, key)] = v.double().clone() if (L, key) not in t else t[(L, key)].add_(v.double())
        return {k: v.float() for k, v in t.items()}

    out["total"] = compare(total(mine), total(r))
    out["pass"] = bool(not problems and out["total"]["pass"] and all(x["pass"] and x["tensors"] == sum(1 for e in expected if e[0] == sg) for sg, x in out["segments"].items()))
    return out


def phase_g7(a, rows):
    from types import SimpleNamespace

    from surogate.inkling_rl.step import StepConfig
    import ift_backward as B
    import ift_engine as E
    import t8_center as RC
    import t8_lean as TL
    import t8_step as TS

    card = int(a.cards)
    which = [int(x) for x in a.g7_rows.split(",")]
    scfg = StepConfig(max_policy_lag=None)
    base = dict(bin_dir=a.bin, artifact=a.artifact, cards=(card,), head="centred", max_row_tokens=58010, segment_keys=("B13", "B2"), train_keys=("B13", "B2"), run_dir=a.run_dir)

    def wait_for(name):
        """the reference may run beside this phase, on another card: wait for its file; its failure marker or 40 min without it ends the phase"""
        path = os.path.join(a.ref_dir, name)
        for _ in range(2400):
            if os.path.exists(path):
                return path
            if os.path.exists(os.path.join(a.ref_dir, "ref_failed")):
                raise SystemExit("the reference run failed")
            time.sleep(1)
        raise SystemExit(f"no {name} after 40 min")

    ref_of = lambda n: torch.load(wait_for(f"ref_{n}.pt"), map_location="cpu", weights_only=False)
    rep = {"K": {}, "rows": which}
    for K, segs in ((48, ("A", "B", "C", "D1", "D0")), (18, ("A", "B"))):
        tr = TS.T8Trainer(TS.T8Config(K=K, **base))
        try:
            rep["K"][str(K)] = []
            for n in which:
                o = tr.compute([rows[n]], 0, scfg, seeds="weights")
                mine = {(sg, L, key): t for (part, sg, L, key), t in o.segments.items()}
                c = g7_compare(mine, ref_of(n), segs)
                c.update({"T": rows[n].T, "seconds": o.rows[0]["seconds"], "tok_per_s": rows[n].T / o.rows[0]["seconds"], "q_block": o.rows[0]["q_block"], "oom_retries": o.rows[0]["oom_retries"],
                          "worker": o.metrics["workers"][0]})
                rep["K"][str(K)].append(c)
                log(f"G7 K={K} T {rows[n].T}: pass {c['pass']}; total 1-cos {1 - c['total']['worst_cos']:.2e}, rel {c['total']['worst_rel']:.4f}, bitwise {c['total']['bitwise']}; {c['tok_per_s']:.0f} tok/s")
                del o, mine
        finally:
            tr.close()
    rep["G7"] = all(c["pass"] for v in rep["K"].values() for c in v)
    rep["ref_meta"] = json.load(open(wait_for("ref_meta.json")))
    if not {0, 1} <= set(which):                              # the G7b diagnostic reads rows 0 and 1
        rep["pass"] = rep["G7"]
        return rep
    # G7b: the reference gradient worker's fp32 A through the lean path (diagnostic)
    eng = E.InklingEngine(a.bin, a.artifact, 0)
    W = B.EngineWeights(eng)
    lora = B.make_lora(eng.dev, seed=0, b_std=0.0)
    banks = B.EngineBanks(lora, eng.dev)
    adapters = [L["adapters"] for L in banks.layers]
    host = [{k: lo[k].detach().cpu() for k in B.LORA_KEYS} for lo in lora]
    scale = lora[0]["scale"]
    for lo in lora:
        for k in B.LORA_KEYS:
            lo[k].data = lo[k].data.cpu()
    torch.cuda.empty_cache()
    rep["G7b"] = []
    for n in range(2):
        r = rows[n]
        ids = torch.as_tensor(np.ascontiguousarray(r.ids), dtype=torch.int32, device=eng.dev)
        buf = TL.RowBuffers(eng.visits, 30, r.T)
        v = TL.forward_keep(eng, ids, adapters, buf)
        g = torch.as_tensor(r.loss_mask[1:].astype(np.float32), device=eng.dev)
        pos = torch.nonzero(g > 0)[:, 0]
        sup = RC.logits_and_support(SimpleNamespace(eng=eng), v["hidden"], pos, ids[1:][pos])
        got = {}

        def on(vis, layer, j, gj):
            k = (TL.segment_of(vis), layer, B.LORA_KEYS[j])
            c = gj.float().cpu()
            got[k] = c if k not in got else got[k].add_(c)

        TL.backward_row(eng, W, ids, v, host, scale, 30, g, buf, on, head_grad=lambda W_, h, t, gg: RC.head_grad_centered(W_, h, gg, sup),
                        q_block=TL.qb_for(r.T), train_idx=(1, 3))
        c = g7_compare(got, ref_of(n), ("A", "B"))
        c["T"] = r.T
        rep["G7b"].append(c)
        log(f"G7b T {r.T}: total 1-cos {1 - c['total']['worst_cos']:.2e}, rel {c['total']['worst_rel']:.2e}, bitwise {c['total']['bitwise']}")
        del v, sup, ids, g, got
        buf.close()
        torch.cuda.empty_cache()
    eng.close()
    rep["pass"] = rep["G7"]
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("lean", "trainer", "g7"))
    ap.add_argument("--ref-dir", default="")
    ap.add_argument("--g7-rows", default="0,1,2")
    ap.add_argument("--bin", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--cards", default="0")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = gate_rows(a.rows)
    rep = {"lean": phase_lean, "trainer": phase_trainer, "g7": phase_g7}[a.phase](a, rows)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    print(f"T8 GATE {a.phase}: {'PASS' if rep['pass'] else 'FAIL'}", flush=True)
    return 0 if rep["pass"] else 4


if __name__ == "__main__":
    sys.exit(main())
