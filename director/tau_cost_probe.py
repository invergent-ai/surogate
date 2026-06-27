"""Agentic cost+quality probe: does GLM's $4/Mtok edge materialize on MULTI-TURN tasks (long output),
or is Opus still terse/cheap as on single-step? Runs each of GLM/Gemini/Opus SOLO on a few tau tasks
through OUR pool (so cost_usd is captured), measuring per-task cost (budget delta) + success. SEQUENTIAL
because the budget accumulator is global.
"""
from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

TASKS = int(os.getenv("TASKS", "3"))  # per env (retail + airline)


def main():
    from director.config import DirectorConfig, FeaturizerConfig, PoolConfig, default_frontier_pool
    from director.agentic.runners import load_tau_tasks, run_tau
    from director.fugu.run import _sampling, build_router
    from director.shared.providers import build_pool

    cfg = DirectorConfig(workers=default_frontier_pool(), featurizer=FeaturizerConfig(context_window=4096))
    wids = cfg.worker_ids
    slugs = {w.worker_id: w.model for w in cfg.workers}
    pool = build_pool(PoolConfig(budget_usd=None, max_concurrency=8, timeout_s=300, max_retries=2), cfg.workers)
    router = build_router(cfg)
    ctx = SimpleNamespace(router=router, pool=pool, sampling=_sampling(cfg), worker_slugs=slugs,
                          tau_user_model="openrouter/openai/gpt-5-mini")

    tasks = [("retail", it) for it in load_tau_tasks("retail", TASKS)] \
        + [("airline", it) for it in load_tau_tasks("airline", TASKS)]
    print(f"probe: {len(tasks)} tasks x {len(wids)} workers (SEQUENTIAL) = {len(tasks)*len(wids)} episodes", flush=True)

    res = defaultdict(list)
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)

    async def run():
        for s, it in tasks:
            for w in wids:
                before = pool.budget.spent_usd
                try:
                    r, _ = await run_tau(ctx, it["payload"], {w})
                except Exception as e:
                    print(f"  ! {w} {it['item_id']}: {type(e).__name__}: {str(e)[:80]}", flush=True)
                    continue
                cost = pool.budget.spent_usd - before
                res[w].append((cost, float(r)))
                print(f"  {w:8} {it['item_id']:16} cost=${cost:.4f}  reward={r}", flush=True)
    loop.run_until_complete(run())

    print("\n=== PER-WORKER (avg over tau tasks) ===", flush=True)
    print(f"  {'worker':8} {'avg $/task':>11} {'success':>8}   n")
    for w in wids:
        a = np.array(res[w]) if res[w] else np.zeros((0, 2))
        if len(a):
            print(f"  {w:8} {a[:,0].mean():>11.4f} {a[:,1].mean():>8.2f}   {len(a)}", flush=True)
    print(f"\ntotal spent: ${pool.budget.spent_usd:.2f}", flush=True)
    print("read: compare $/task ranking here (multi-turn) to single-step ($/call: opus 0.010, glm 0.010, "
          "gemini 0.048). If Opus stays cheapest, GLM cost product fails on agentic too; if Opus balloons, "
          "GLM's $4 edge returns.", flush=True)


if __name__ == "__main__":
    main()
