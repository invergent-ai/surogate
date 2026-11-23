"""Lane T: memory / time probe of one training micro-step (engine forward + LEVEL-0 backward) per row length on one
card. Rows: real prompt ids (o25 4,963 tokens, then o00 appended, repeated) cut to T. LoRA r16 on all 24 layers (rig
init seed 7, b_std 1e-3). Reports peak torch allocation / reservation and the card's free memory; stops at the first
OOM. Usage: python mem_probe.py --bin B --artifact A --prompts DIR --lengths 2048,4096,6144,8192 --out report.json
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


def log(msg):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    print(f"[T {datetime.now(ZoneInfo('Europe/Bucharest')).strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--lengths", default="2048,4096,6144,8192")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    eng = InklingEngine(a.bin, a.artifact, 0)
    W = B.EngineWeights(eng)
    lora = B.make_lora(eng.dev, seed=7, b_std=1e-3)
    banks = B.EngineBanks(lora, eng.dev)
    tf = B.TrainForward(eng, W)
    params = [lo[k] for lo in lora for k in B.LORA_KEYS]
    pool = []
    for n in ("o25", "o00"):
        pool += json.load(open(os.path.join(a.prompts, n + ".json")))["ids"]
    while len(pool) < 16384:
        pool = pool + pool
    base_free = torch.cuda.mem_get_info()[0]
    log(f"static: torch alloc {torch.cuda.memory_allocated() / 2**30:.2f} GiB, card free {base_free / 2**30:.2f} GiB")
    rep = {"static_free_gib": base_free / 2**30, "rows": []}
    for T in [int(x) for x in a.lengths.split(",")]:
        for p in params:
            p.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ids = torch.tensor(pool[:T], dtype=torch.int32, device=eng.dev)
        row = {"T": T}
        try:
            t0 = time.time()
            lp, res = tf.build(ids, lora, banks)
            torch.cuda.synchronize()
            row["forward_s"] = time.time() - t0
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
            row["oom"] = str(e).split(".")[0]
        row["peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2**30
        row["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        rep["rows"].append(row)
        log(json.dumps(row))
        torch.cuda.empty_cache()
        if "oom" in row:
            break
    json.dump(rep, open(a.out, "w"), indent=1)
    eng.close()


if __name__ == "__main__":
    main()
