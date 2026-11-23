"""Lane T: torch.profiler table of one training micro-step (engine forward + long-row LEVEL-0 backward) at T tokens:
the top ops by CUDA time and by CPU time (where the LEVEL-1 time goes below the section timers).
Usage: python prof_ops.py --bin B --artifact A --prompts DIR --tokens 8192 --out table.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ift_engine import InklingEngine  # noqa: E402
import ift_backward as B  # noqa: E402
import ift_long as LG  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    eng = InklingEngine(a.bin, a.artifact, 0)
    W = B.EngineWeights(eng)
    lora = B.make_lora(eng.dev, seed=7, b_std=1e-3)
    banks = B.EngineBanks(lora, eng.dev)
    tf = LG.TrainForwardLong(eng, W)
    pool = []
    for n in ("o25", "o00"):
        pool += json.load(open(os.path.join(a.prompts, n + ".json")))["ids"]
    while len(pool) < a.tokens:
        pool += pool
    ids = torch.tensor(pool[:a.tokens], dtype=torch.int32, device=eng.dev)
    # warm-up step (allocator, cuBLAS handles), then the profiled one
    lp, _ = tf.build(ids, lora, banks)
    (-lp.mean()).backward()
    for lo in lora:
        for k in B.LORA_KEYS:
            lo[k].grad = None
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        lp, _ = tf.build(ids, lora, banks)
        (-lp.mean()).backward()
        torch.cuda.synchronize()
    ka = prof.key_averages()
    with open(a.out, "w") as f:
        f.write(ka.table(sort_by="cuda_time_total", row_limit=40))
        f.write("\n\n")
        f.write(ka.table(sort_by="cpu_time_total", row_limit=40))
    print(ka.table(sort_by="cuda_time_total", row_limit=25), flush=True)
    eng.close()


if __name__ == "__main__":
    main()
