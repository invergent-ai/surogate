"""Stage-2 agentic bank: SOLO-worker rollouts on the non-eval agentic TRAINING sources (SWE-smith +
tau retail/airline) -> per-(task, worker) terminal-reward cells. This establishes the agentic headroom
(oracle vs best-single) that Stage-2 CMA-ES then tries to capture with live per-step routing.

Eval benchmarks (SWE-bench Verified, Terminal-Bench, tau^3 banking) are deliberately NOT here — they
stay held out for the final eval. Resumable: skips (item_id, worker) cells already in agentic_bank.jsonl.

Env: MANIFEST_DIR, SM (swesmith tasks), TAU (tasks per tau env), PARALLEL.
"""
from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from director.agentic.runners import load_swesmith_tasks, load_tau_tasks, run_swesmith, run_tau
from director.config import DirectorConfig, FeaturizerConfig, PoolConfig, default_frontier_pool
from director.fugu.run import _sampling, build_router
from director.shared.providers import build_pool

MANIFEST_DIR = os.getenv("MANIFEST_DIR", "manifests/fugu_clean_v1")
SM = int(os.getenv("SM", "20"))
TAU = int(os.getenv("TAU", "15"))
PARALLEL = int(os.getenv("PARALLEL", "8"))
BANK = os.path.join(MANIFEST_DIR, "agentic_bank.jsonl")
RUNNER = {"swesmith": run_swesmith, "tau_retail": run_tau, "tau_airline": run_tau}


def load_tasks():
    """(source, item) pairs across the agentic training sources, same loaders/seed CMA-ES will reuse.
    SM/TAU = 0 skips that source (so a tau-only smoke doesn't stream all of SWE-smith)."""
    tasks = []
    if SM:
        tasks += [("swesmith", it) for it in load_swesmith_tasks(SM)]
    if TAU:
        tasks += [("tau_retail", it) for it in load_tau_tasks("retail", TAU)]
        tasks += [("tau_airline", it) for it in load_tau_tasks("airline", TAU)]
    return tasks


def _done():
    seen = set()
    if os.path.exists(BANK):
        for l in open(BANK):
            if l.strip():
                c = json.loads(l)
                seen.add((c["item_id"], c["worker"]))
    return seen


def report():
    cfg = DirectorConfig(workers=default_frontier_pool())
    wids = cfg.worker_ids
    items = defaultdict(dict)
    for l in open(BANK):
        if l.strip():
            c = json.loads(l)
            items[(c["domain"], c["item_id"])][c["worker"]] = c["reward"]
    complete = {k: v for k, v in items.items() if all(w in v for w in wids)}
    if not complete:
        print("no complete agentic items yet", flush=True)
        return
    R = np.array([[v[w] for w in wids] for v in complete.values()], dtype=float)
    best = float(R.mean(0).max())
    oracle = float(R.max(1).mean())
    disc = int(sum(1 for row in R if 0 < row.sum() < len(wids)))  # some pass, some fail
    by_dom = defaultdict(int)
    for (dom, _), _ in complete.items():
        by_dom[dom] += 1
    lines = [
        "", "## Stage-2 agentic bank (solo rollouts)",
        f"- complete items: {len(complete)} ({dict(by_dom)}) | discriminative: {disc} "
        f"({disc/len(complete):.0%})",
        f"- agentic oracle: {oracle:.3f} vs best-single: {best:.3f} "
        f"(**headroom {oracle-best:+.3f}** — what CMA-ES chases)",
        "- per-worker mean reward: " + "  ".join(f"{w}={R[:, j].mean():.2f}" for j, w in enumerate(wids)),
    ]
    with open(os.path.join(MANIFEST_DIR, "data_report.md"), "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    cfg = DirectorConfig(workers=default_frontier_pool(), featurizer=FeaturizerConfig(context_window=4096))
    pool = build_pool(PoolConfig(budget_usd=None, max_concurrency=48, timeout_s=120, max_retries=2), cfg.workers)
    wids = cfg.worker_ids
    slugs = {w.worker_id: w.model for w in cfg.workers}
    # solo rollouts force allowed={w}, so the (untrained) router selection is overridden — fine here.
    ctx = SimpleNamespace(router=build_router(cfg), pool=pool, sampling=_sampling(cfg), worker_slugs=slugs,
                          tau_user_model="openrouter/openai/gpt-5-mini")

    tasks = load_tasks()
    done = _done()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    sem = asyncio.Semaphore(PARALLEL)
    wlock = asyncio.Lock()
    fh = open(BANK, "a")
    stats = {"done": 0, "cost": 0.0, "total": 0}  # live progress + running cost (no budget cap by choice)

    async def cell(src, item, wid):
        ok, err = True, ""
        async with sem:
            try:
                reward, cost = await RUNNER[src](ctx, item["payload"], {wid})
            except Exception as e:
                reward, cost, ok, err = 0.0, 0.0, False, type(e).__name__
        async with wlock:
            if ok:  # failed cells are NOT banked -> retried on the next (resumable) run
                fh.write(json.dumps({"domain": src, "item_id": item["item_id"], "worker": wid,
                                     "reward": float(reward), "cost": float(cost)}) + "\n")
                fh.flush()
            stats["done"] += 1
            stats["cost"] += cost
            tag = f"r={reward:.0f} ${cost:.3f}" if ok else f"FAIL:{err}"
            print(f"  [{stats['done']}/{stats['total']}] {src}/{item['item_id'][:24]}/{wid} {tag} "
                  f"| running ${stats['cost']:.2f}", flush=True)

    async def go():
        jobs = [cell(src, item, w) for (src, item) in tasks for w in wids
                if (item["item_id"], w) not in done]
        stats["total"] = len(jobs)
        print(f"agentic bank: {len(tasks)} tasks x {len(wids)} workers -> {len(jobs)} cells to run "
              f"({len(done)} cached) | live progress + running cost below", flush=True)
        await asyncio.gather(*jobs)

    loop.run_until_complete(go())
    fh.close()
    print(f"\nbank done: {stats['done']} cells, total cost ${stats['cost']:.2f}", flush=True)
    report()


if __name__ == "__main__":
    main()
