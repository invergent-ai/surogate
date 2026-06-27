"""$5 calibration: measure ACTUAL per-call cost + tokens for the 3-pool at the screen's sampling,
then project the 300-item (n=8) screen cost precisely. Reports both OpenRouter-reported cost_usd and
a token x price derivation (in case a provider reports cost=0)."""
from __future__ import annotations

import asyncio
import json
import os
import random
from collections import defaultdict

import numpy as np

ITEMS = int(os.getenv("ITEMS", "12"))
N = int(os.getenv("N", "2"))
PRICE_OUT = {"glm": 4.0, "gemini": 12.0, "opus": 25.0}
PRICE_IN = {"glm": 0.95, "gemini": 1.5, "opus": 5.0}  # approx input $/Mtok


def main():
    from director.config import PoolConfig, default_frontier_pool
    from director.shared.providers import build_pool
    from director.shared.types import Sampling

    specs = default_frontier_pool()
    wids = [w.worker_id for w in specs]
    pool = build_pool(PoolConfig(budget_usd=None, max_concurrency=16, timeout_s=240, max_retries=2), specs)
    samp = Sampling(temperature=0.7, max_tokens=8192, reasoning_effort="high")  # matches the screen

    items = [json.loads(l) for l in open("manifests/fugu_clean_v1/probe.jsonl") if l.strip()]
    by = defaultdict(list)
    for it in items:
        by[it["domain"]].append(it)
    rng = random.Random(0)
    sel = []
    for d in ["math", "code", "science", "general"]:
        pd = by[d][:]; rng.shuffle(pd); sel += pd[:max(1, ITEMS // 4)]
    sel = sel[:ITEMS]
    print(f"calibrating on {len(sel)} items x {len(wids)} workers x n={N} = {len(sel)*len(wids)*N} calls", flush=True)

    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    rec = defaultdict(list)  # wid -> (cost_usd, ptok, ctok)

    async def run():
        for it in sel:
            msgs = ([{"role": "system", "content": it["system"]}] if it.get("system") else []) \
                + [{"role": "user", "content": it["prompt"]}]
            for w in wids:
                try:
                    comps = await pool.sample(w, msgs, N, samp)
                    for c in comps:
                        rec[w].append((c.cost_usd, c.prompt_tokens, c.completion_tokens))
                except Exception as e:
                    print(f"  ! {w}: {type(e).__name__}: {str(e)[:80]}", flush=True)
    loop.run_until_complete(run())

    print("\nper-worker ACTUAL (avg per call):", flush=True)
    item_cost_reported, item_cost_tok = 0.0, 0.0
    spent = 0.0
    for w in wids:
        a = np.array(rec[w])
        if len(a) == 0:
            print(f"  {w}: NO DATA"); continue
        cost, pt, ct = a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()
        tok_cost = pt * PRICE_IN[w] / 1e6 + ct * PRICE_OUT[w] / 1e6
        print(f"  {w:8} reported ${cost:.4f}/call | token-derived ${tok_cost:.4f}/call  (in {pt:.0f}, out {ct:.0f})  [n={len(a)}]", flush=True)
        item_cost_reported += cost
        item_cost_tok += tok_cost
        spent += a[:, 0].sum()

    print(f"\nper-item (1 call each of {len(wids)} workers):  reported ${item_cost_reported:.4f}  | token-derived ${item_cost_tok:.4f}", flush=True)
    for label, ic in [("reported", item_cost_reported), ("token-derived", item_cost_tok)]:
        print(f"  PROJECTED 300-item screen (n=8): ${ic*8*300:.0f}   [{label}]", flush=True)
    print(f"\n(this calibration actually spent ~${spent:.2f})", flush=True)


if __name__ == "__main__":
    main()
