"""Lane T8, the U1 card gate. PASS LINES, written before any run (also in lanes/T8/NOTES.md):
Phase "lean" (one card; rows of 16k and 28k; plain head):
  G1  the lean K = 48 backward == the stock path (ift_long.LongModel) on every LoRA tensor.
  G2  the truncated backward at K = 30 / 18 / 6 == the sum of the last K visits' contributions of the full lean backward.
Phase "trainer" (two cards; four real rows of 16k / 28k / 47k / 57k; K = 18, centred head, seeds "weights", B keys):
  G3  each row's gradient on card A == on card B.
  G4  the two-worker step's gradient == the sum of the rows' gradients; its segment sums add up to it.
  G5  the hard gates hold on real rows: replay 0 mismatches, support gate, zero-adapter check.
  G6  host peak per worker <= 20 GiB with the 57k row; no row out of memory at query block 8.
Line for G1-G4: every tensor cos >= 0.9999 and relative error <= 1 % (G4: 1e-5), all finite. Whether a comparison is BITWISE is reported beside it; bitwise is expected for G1-G3.
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
    """Over the union of keys; a key missing on one side counts as a zero tensor there (it must then be zero on the other)."""
    worst_cos, worst_rel, n, bit = 1.0, 0.0, 0, True
    finite = True
    for k in sorted(set(xs) | set(ys), key=str):
        a, b = xs.get(k), ys.get(k)
        a = torch.zeros_like(b) if a is None else a
        b = torch.zeros_like(a) if b is None else b
        finite = finite and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
        bit = bit and torch.equal(a, b)
        if float(b.double().norm()) == 0 and float(a.double().norm()) == 0:
            n += 1
            continue
        c, r = cmp_t(a, b)
        worst_cos, worst_rel = min(worst_cos, c), max(worst_rel, r)
        n += 1
    return {"tensors": n, "worst_cos": worst_cos, "worst_rel": worst_rel, "bitwise": bool(bit), "finite": bool(finite),
            "pass": bool(finite and worst_cos >= COS and worst_rel <= rel_line)}


def gate_rows(path):
    from surogate.inkling_rl.records import Row

    z = np.load(path)
    rows = []
    for n in range(sum(1 for k in z.files if k.startswith("ids_"))):
        ids, mask, lp = z[f"ids_{n}"].astype(np.int64), z[f"mask_{n}"].astype(bool), z[f"lp_{n}"].astype(np.float64)
        rows.append(Row(ids=ids, loss_mask=mask, engine_lp=lp, advantage=0.0, versions=None, teacher_lp=None, routed=None, notes={"weight": 1.0}))
    return rows


# ---------------------------------------------------------------------------------------------------------------------
def phase_lean(a, rows):
    import ift_backward as B
    import ift_engine as E
    import ift_long as LG
    import t8_lean as TL

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
    rep = {"rows": []}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("lean", "trainer"))
    ap.add_argument("--bin", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--cards", default="0")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = gate_rows(a.rows)
    rep = (phase_lean if a.phase == "lean" else phase_trainer)(a, rows)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    print(f"T8 GATE {a.phase}: {'PASS' if rep['pass'] else 'FAIL'}", flush=True)
    return 0 if rep["pass"] else 4


if __name__ == "__main__":
    sys.exit(main())
