"""Measure whether a routable pool EXISTS, before building anything.

Builds a denoised (n=N) performance matrix E[domain, model] over candidate models spanning many
training LINEAGES (different labs + reasoning/direct), then tests the two things that decide whether
accuracy routing is viable on our tasks:
  1. RANK-FLIP: does the best model CHANGE across domains? (one dominant model => routing is pointless)
  2. RER vs the n=N NOISE NULL: is the per-item oracle headroom real, or just sampling noise?

This is Trinity's A.6 pool-selection logic (maximize relative error reduction / complementarity),
applied as a go/no-go gate. Reuses probe.jsonl items (prompt+solution+grader) — single-step, no
agentic rollouts. Resumable (appends per item to OUT).

Env: N, PER_DOMAIN, OUT, MAX_CONC.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
from collections import defaultdict

import numpy as np

# Pool comes from the locked config (default_frontier_pool): the 4 frontier labs.
N = int(os.getenv("N", "8"))  # critique: n>=8 so noisy means don't become near-hard labels; enables cross-fit
PER_DOMAIN = int(os.getenv("PER_DOMAIN", "30"))
MAX_CONC = int(os.getenv("MAX_CONC", "56"))
DOMAINS = ["math", "code", "science", "general"]
PROBE = "manifests/fugu_clean_v1/probe.jsonl"
OUT = os.getenv("OUT", "manifests/fugu_clean_v1/pool_matrix.jsonl")


def _crossfit(items):
    """Cross-fitted per-item oracle headroom (removes winner's-curse): pick the apparent-best worker
    on rep-half A, score it on held-out half B; swap; average. Compare to best-single. Bootstrap CI
    over items. ``items`` = list of (L, n) per-rep 0/1 arrays. Returns (headroom, lo, hi, best, oracle_cf)."""
    items = [np.asarray(it, dtype=float) for it in items]

    def stats(sel):
        am = np.array([it.mean(1) for it in sel])      # (I, L) per-worker mean accuracy
        best = float(am.mean(0).max())                 # best single worker
        vals = []
        for it in sel:                                 # (L, n)
            n = it.shape[1]
            h = n // 2
            A, B = it[:, :h].mean(1), it[:, h:].mean(1)
            vals.append((B[int(A.argmax())] + A[int(B.argmax())]) / 2)  # cross-fitted oracle reward
        return best, float(np.mean(vals))

    best, oracle_cf = stats(items)
    rng = np.random.default_rng(0)
    I = len(items)
    boot = []
    for _ in range(1000):
        idx = rng.integers(0, I, I)
        b, o = stats([items[k] for k in idx])
        boot.append(o - b)
    lo, hi = (float(x) for x in np.quantile(boot, [0.025, 0.975]))
    return oracle_cf - best, lo, hi, best, oracle_cf


def analyze():
    recs = [json.loads(l) for l in open(OUT) if l.strip()] if os.path.exists(OUT) else []
    if not recs:
        print("no records yet")
        return
    wids = recs[0]["worker_ids"]
    by_dom_rw = defaultdict(list)  # domain -> list of (L,n) per-rep arrays
    for r in recs:
        by_dom_rw[r["domain"]].append(r.get("rewards") or [[v] for v in r["r_bar"]])
    all_rw = [r.get("rewards") or [[v] for v in r["r_bar"]] for r in recs]
    print(f"\n===== POOL MATRIX ({len(recs)} items, n={N}, {len(wids)} models) =====")
    print("per-domain mean accuracy:")
    print("domain     " + "".join(f"{w[:11]:>12}" for w in wids))
    flip = set()
    for d in DOMAINS:
        if not by_dom_rw.get(d):
            continue
        means = np.array([[np.mean(c) for c in it] for it in by_dom_rw[d]]).mean(0)
        bi = int(means.argmax())
        flip.add(wids[bi])
        print(f"{d:10}" + "".join(f"{m:>12.3f}" for m in means) + f"   <-best: {wids[bi]}")
    print(f"\nRANK-FLIP: best model per domain = {sorted(flip)}  -> "
          + ("FLIPS (routable structure exists)" if len(flip) > 1 else "NO FLIP (one dominator -> not routable)"))

    print("\nCROSS-FITTED oracle headroom (winner's-curse removed; pool viable iff 95% CI lower bound > 0):")
    hr, lo, hi, best, ocf = _crossfit(all_rw)
    print(f"  [overall ] best-single={best:.3f} oracle_cf={ocf:.3f} headroom={hr:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}] -> "
          + ("VIABLE" if lo > 0 else "not viable"))
    for d in DOMAINS:
        if by_dom_rw.get(d):
            hr, lo, hi, best, ocf = _crossfit(by_dom_rw[d])
            print(f"  [{d:8}] best-single={best:.3f} oracle_cf={ocf:.3f} headroom={hr:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}] -> "
                  + ("VIABLE" if lo > 0 else "not viable"))


async def collect():
    from director.config import PoolConfig, default_frontier_pool
    from director.shared.providers import build_pool
    from director.shared.types import Sampling
    from director.shared.verifiers import get_grader

    specs = default_frontier_pool()  # single source of truth: the locked 4-frontier pool
    pool = build_pool(PoolConfig(budget_usd=None, max_concurrency=MAX_CONC, timeout_s=240, max_retries=2), specs)
    wids = [w.worker_id for w in specs]
    sampling = Sampling(temperature=0.7, max_tokens=8192, reasoning_effort="high")

    items = [json.loads(l) for l in open(PROBE) if l.strip()]
    by_dom = defaultdict(list)
    for it in items:
        by_dom[it["domain"]].append(it)
    rng = random.Random(0)
    sel = []
    for d in DOMAINS:
        pd = by_dom.get(d, [])[:]
        rng.shuffle(pd)
        sel += pd[:PER_DOMAIN]
    print(f"selected {len(sel)} items: " + str({d: min(len(by_dom.get(d, [])), PER_DOMAIN) for d in DOMAINS}), flush=True)

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            if l.strip():
                done.add(json.loads(l)["task_id"])
    todo = [it for it in sel if it["task_id"] not in done]
    print(f"{len(done)} cached, {len(todo)} to do (n={N}, {len(wids)} models, {len(todo)*len(wids)*N} completions)", flush=True)
    if not todo:
        return
    fh = open(OUT, "a")
    sem = asyncio.Semaphore(MAX_CONC)
    lock = asyncio.Lock()
    fail = defaultdict(int)

    async def grade_model(msgs, grader, sol, wid):
        try:
            async with sem:
                comps = await pool.sample(wid, msgs, N, sampling)
            rs = await asyncio.to_thread(lambda: [float(grader(c.text, sol)) for c in comps])
            return rs  # per-rep outcomes (for cross-fitting), not just the mean
        except Exception as e:
            fail[wid] += 1
            if fail[wid] <= 3:
                print(f"  ! {wid}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            return [0.0] * N

    async def one(it):
        grader = get_grader(it["grader"])
        msgs = ([{"role": "system", "content": it["system"]}] if it.get("system") else []) \
            + [{"role": "user", "content": it["prompt"]}]
        rewards = list(await asyncio.gather(*[grade_model(msgs, grader, it["solution"], w) for w in wids]))
        rec = {"task_id": it["task_id"], "domain": it["domain"], "worker_ids": wids,
               "rewards": rewards, "r_bar": [sum(rs) / max(len(rs), 1) for rs in rewards]}
        async with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

    for i in range(0, len(todo), 16):
        await asyncio.gather(*[one(it) for it in todo[i:i + 16]])
        print(f"  {min(i+16, len(todo))}/{len(todo)} items  (fails so far: {dict(fail)})", flush=True)
    fh.close()


if __name__ == "__main__":
    import sys
    if "--analyze" in sys.argv:
        analyze()
    else:
        asyncio.run(collect())
        analyze()
