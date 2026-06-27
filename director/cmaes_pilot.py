"""Stage-2 sep-CMA-ES pilot: warm-start from the SFT router, evolve it on LIVE routed agentic rollouts
(the candidate routes per-step, allowed=None) maximizing mean terminal reward — this is where Fugu's
real lift comes from (SFT is only the warm-start). Reports CMA-ES routed reward vs the SFT-only router
vs best-single (after-agentic gates).

Reuses build_agentic_bank's loaders/seed so the eval tasks match the bank (best-single is comparable).
Run build_agentic_bank.py (headroom) + train_eval_pilot.py (sft_router.pt) first.

Env: MANIFEST_DIR, CMAES_GENERATIONS, CMAES_EVAL_TASKS, CMAES_POPSIZE, CMAES_SIGMA0, PARALLEL, SM, TAU.
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
from director.fugu.cmaes import evolve
from director.fugu.model import save_router
from director.fugu.run import _sampling, build_router
from director.shared.providers import build_pool

MANIFEST_DIR = os.getenv("MANIFEST_DIR", "manifests/fugu_clean_v1")
SFT_CKPT = os.path.join(MANIFEST_DIR, "sft_router.pt")
BANK = os.path.join(MANIFEST_DIR, "agentic_bank.jsonl")
GENERATIONS = int(os.getenv("CMAES_GENERATIONS", "3"))
EVAL_TASKS = int(os.getenv("CMAES_EVAL_TASKS", "6"))
POPSIZE = int(os.getenv("CMAES_POPSIZE", "0")) or None
SIGMA0 = float(os.getenv("CMAES_SIGMA0", "0.03"))  # Fugu's SVF sigma0
PARALLEL = int(os.getenv("PARALLEL", "6"))
SM = int(os.getenv("SM", "20"))   # must match build_agentic_bank's loaders/seed
TAU = int(os.getenv("TAU", "15"))
RUNNER = {"swesmith": run_swesmith, "tau_retail": run_tau, "tau_airline": run_tau}


def load_tasks():
    tasks = []
    if SM:
        tasks += [("swesmith", it) for it in load_swesmith_tasks(SM)]
    if TAU:
        tasks += [("tau_retail", it) for it in load_tau_tasks("retail", TAU)]
        tasks += [("tau_airline", it) for it in load_tau_tasks("airline", TAU)]
    return tasks


def bank_best_oracle(item_ids, wids):
    """best-single + oracle over the bank cells for these eval items (comparable to routed reward)."""
    items = defaultdict(dict)
    if os.path.exists(BANK):
        for l in open(BANK):
            if l.strip():
                c = json.loads(l)
                items[c["item_id"]][c["worker"]] = c["reward"]
    rows = [[items[i][w] for w in wids] for i in item_ids if all(w in items.get(i, {}) for w in wids)]
    if not rows:
        return None, None
    R = np.array(rows, dtype=float)
    return float(R.mean(0).max()), float(R.max(1).mean())


def main():
    if not os.path.exists(SFT_CKPT):
        print(f"need SFT checkpoint {SFT_CKPT} — run train_eval_pilot.py first", flush=True)
        return
    cfg = DirectorConfig(workers=default_frontier_pool(), featurizer=FeaturizerConfig(context_window=4096))
    pool = build_pool(PoolConfig(budget_usd=None, max_concurrency=48, timeout_s=120, max_retries=2), cfg.workers)
    wids = cfg.worker_ids
    slugs = {w.worker_id: w.model for w in cfg.workers}
    router = build_router(cfg, ckpt=SFT_CKPT)  # warm-start from SFT (evolve uses this vector as x0)
    ctx = SimpleNamespace(router=router, pool=pool, sampling=_sampling(cfg), worker_slugs=slugs,
                          tau_user_model="openrouter/openai/gpt-5-mini")

    # eval tasks: round-robin a small slice across the sources so it isn't all one source.
    all_eval = load_tasks()
    nsrc = max(1, len({s for s, _ in all_eval}))
    cap = max(1, -(-EVAL_TASKS // nsrc))  # ceil per source so we actually reach EVAL_TASKS
    per_src, eval_tasks = defaultdict(int), []
    for src, item in all_eval:
        if per_src[src] < cap:
            eval_tasks.append((src, item))
            per_src[src] += 1
    eval_tasks = eval_tasks[:EVAL_TASKS]
    best, oracle = bank_best_oracle([it["item_id"] for _, it in eval_tasks], wids)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    sem = asyncio.Semaphore(PARALLEL)

    async def routed_reward():
        async def one(src, item):
            async with sem:
                try:
                    r, _ = await RUNNER[src](ctx, item["payload"], None)  # allowed=None -> route LIVE
                    return float(r)
                except Exception as e:
                    print(f"  ! {item['item_id']}: {type(e).__name__}", flush=True)
                    return 0.0
        rs = await asyncio.gather(*[one(s, it) for s, it in eval_tasks])
        return sum(rs) / max(len(rs), 1)

    def eval_fn():
        return loop.run_until_complete(routed_reward())  # persistent loop -> pool stays bound

    print(f"[cmaes] eval tasks={len(eval_tasks)} ({dict(per_src)}) | "
          f"bank best-single={best} oracle={oracle}", flush=True)
    sft_routed = eval_fn()  # SFT-only baseline (router == SFT vector before evolving)
    print(f"[cmaes] SFT-only routed reward={sft_routed:.3f}", flush=True)
    print(f"[cmaes] Stage 2: {GENERATIONS} gens x popsize={POPSIZE or 'auto'} (sigma0={SIGMA0})", flush=True)
    res = evolve(router, eval_fn, generations=GENERATIONS, sigma0=SIGMA0, popsize=POPSIZE,
                 checkpoint_dir=os.path.join(MANIFEST_DIR, "cmaes_ckpt"), resume=True, verbose=True)
    save_router(router, os.path.join(MANIFEST_DIR, "cmaes_router.pt"), worker_ids=wids)
    cmaes_routed = res.best_fitness  # best in-sample routed reward (router now holds best_x)

    gates = {
        "CMA-ES > SFT-only": cmaes_routed > sft_routed,
        "CMA-ES >= best-single": best is not None and cmaes_routed >= best,
    }
    lines = [
        "", "## After-agentic (Stage-2 CMA-ES)",
        f"- eval tasks={len(eval_tasks)} | best-single={best} | oracle={oracle}",
        f"- SFT-only routed={sft_routed:.3f} → CMA-ES routed={cmaes_routed:.3f} "
        f"(lift {cmaes_routed - sft_routed:+.3f}) over {res.generations_run} gens",
        "", "### after-agentic gates",
        *[f"- [{'PASS' if ok else 'FAIL'}] {name}" for name, ok in gates.items()],
        "_(in-sample lift on the eval tasks — a positive signal that CMA-ES captures agentic headroom;"
        " the held-out eval harness is the final arbiter.)_",
    ]
    with open(os.path.join(MANIFEST_DIR, "data_report.md"), "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"\nAFTER-AGENTIC GATES: {'ALL PASS' if all(gates.values()) else 'SOME FAILED'}", flush=True)
    loop.close()


if __name__ == "__main__":
    main()
