"""Lane T long-row path check (one card):
1. EQUIVALENCE on a short row (o00[:1024], LoRA rig seed 7 b_std 1e-3): ift_long's gradient vs ift_backward's (the
   path validated against lane L's rig), every LoRA tensor of every layer: cosine and relative error; losses equal.
2. MEMORY / SPEED per row length with ift_long (stops at the first OOM): forward, backward, tok/s, peak.
Usage: python long_check.py --bin B --artifact A --prompts DIR --lengths 4096,8192,16384,32768 --out report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ift_engine import InklingEngine  # noqa: E402
import ift_backward as B  # noqa: E402
import ift_long as LG  # noqa: E402


def log(msg):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    print(f"[T {datetime.now(ZoneInfo('Europe/Bucharest')).strftime('%H:%M:%S')}] {msg}", flush=True)


def grads_of(params):
    return [p.grad.detach().clone() if p.grad is not None else None for p in params]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--lengths", default="4096,8192,16384,32768")
    ap.add_argument("--equiv-tokens", type=int, default=1024)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    eng = InklingEngine(a.bin, a.artifact, 0)
    W = B.EngineWeights(eng)
    lora = B.make_lora(eng.dev, seed=7, b_std=1e-3)
    banks = B.EngineBanks(lora, eng.dev)
    params = [lo[k] for lo in lora for k in B.LORA_KEYS]
    pool = []
    for n in ("o25", "o00"):
        pool += json.load(open(os.path.join(a.prompts, n + ".json")))["ids"]
    while len(pool) < 140000:
        pool = pool + pool
    rep = {}
    # 1. equivalence
    ids = torch.tensor(pool[:a.equiv_tokens], dtype=torch.int32, device=eng.dev)
    for p in params:
        p.grad = None
    lp0, _ = B.TrainForward(eng, W).build(ids, lora, banks)
    l0 = -lp0.mean()
    l0.backward()
    g_old = grads_of(params)
    loss_old = float(l0.detach())
    del lp0, l0
    for p in params:
        p.grad = None
    torch.cuda.reset_peak_memory_stats()
    lp1, _ = LG.TrainForwardLong(eng, W).build(ids, lora, banks)
    l1 = -lp1.mean()
    l1.backward()
    g_new = grads_of(params)
    cos, rel = [], []
    for x, y in zip(g_new, g_old):
        xd, yd = x.flatten().double(), y.flatten().double()
        cos.append(float(xd @ yd / (xd.norm() * yd.norm()).clamp_min(1e-300)))
        rel.append(float((xd - yd).norm() / yd.norm().clamp_min(1e-300)))
    rep["equivalence"] = {"T": a.equiv_tokens, "loss_old": loss_old, "loss_new": float(l1.detach()),
                          "cos_min": min(cos), "rel_err_max": max(rel), "rel_err_median": sorted(rel)[len(rel) // 2],
                          "peak_new_gib": torch.cuda.max_memory_allocated() / 2**30}
    log("equivalence " + json.dumps(rep["equivalence"]))
    del lp1, l1, g_old, g_new
    # 2. memory / speed
    rep["rows"] = []
    tf = LG.TrainForwardLong(eng, W)
    for T in [int(x) for x in a.lengths.split(",")]:
        for p in params:
            p.grad = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ids = torch.tensor(pool[:T], dtype=torch.int32, device=eng.dev)
        row = {"T": T}
        try:
            t0 = time.time()
            lp, res = tf.build(ids, lora, banks)
            torch.cuda.synchronize()
            row["forward_s"] = time.time() - t0
            row["peak_after_forward_gib"] = torch.cuda.max_memory_allocated() / 2**30
            loss = -lp.mean()
            t1 = time.time()
            loss.backward()
            torch.cuda.synchronize()
            row["backward_s"] = time.time() - t1
            row["loss"] = float(loss.detach())
            row["finite"] = all(bool(torch.isfinite(p.grad).all()) for p in params if p.grad is not None)
            row["tok_per_s"] = T / (row["forward_s"] + row["backward_s"])
            del lp, res, loss
        except torch.OutOfMemoryError as e:
            row["oom"] = str(e).split(". ")[0][:200]
        row["peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2**30
        rep["rows"].append(row)
        log(json.dumps(row))
        if "oom" in row:
            break
    json.dump(rep, open(a.out, "w"), indent=1)
    eng.close()


if __name__ == "__main__":
    main()
