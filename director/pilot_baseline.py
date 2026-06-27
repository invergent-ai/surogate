"""Agentic headroom baseline: run each open worker SOLO on N SWE-Bench Verified tasks.
Decisive cheap cut — do different workers solve different tasks (complementary => routing
headroom)? Mirrors the single-step headroom analysis, on real multi-turn agentic tasks.

Budget-capped; rollouts are cheap (open pool), time is dominated by Docker.
Usage: .venv/bin/python pilot_baseline.py
"""
from __future__ import annotations
import asyncio, time
import numpy as np
from director.config import DirectorConfig, PoolConfig, default_frontier_pool
from director.fugu.run import build_router, _sampling
from director.shared.providers import build_pool
from director.agentic.swebench_env import load_swebench, build_swebench_factories
from director.agentic.rollout import agentic_rollout

N_TASKS = 8
MAX_TURNS = 30
MAX_PARALLEL = 8
cfg = DirectorConfig(workers=default_frontier_pool())
wids = cfg.worker_ids
pool = build_pool(PoolConfig(budget_usd=100.0), cfg.workers)
router = build_router(cfg)  # only used for featurize; allowed= forces the solo worker
samp = _sampling(cfg)

insts = load_swebench(limit=N_TASKS, shuffle=True, seed=0)
tasks = [i["instance_id"] for i in insts]
facs = build_swebench_factories(insts, step_timeout=120.0)
print(f"pilot: {len(tasks)} SWE-Bench Verified tasks x {len(wids)} workers, max_turns={MAX_TURNS}")

sem = asyncio.Semaphore(MAX_PARALLEL)
R = np.zeros((len(tasks), len(wids)))


async def one(ti, wj):
    w = wids[wj]
    async with sem:
        env = facs[ti]()
        t0 = time.time()
        try:
            res = await agentic_rollout(router, pool, env, max_turns=MAX_TURNS, sampling=samp, allowed={w})
            R[ti, wj] = res.reward
            print(f"  [{tasks[ti][:28]:28}] {w:9} reward={res.reward} turns={res.turns} "
                  f"sub={res.submitted} ${res.cost_usd:.3f} {time.time()-t0:.0f}s spent=${pool.budget.spent_usd:.2f}", flush=True)
        except Exception as e:
            print(f"  [{tasks[ti][:28]:28}] {w:9} ERROR {type(e).__name__}: {str(e)[:70]}", flush=True)
        finally:
            env.close()


async def main():
    await asyncio.gather(*[one(ti, wj) for ti in range(len(tasks)) for wj in range(len(wids))])
    per = R.mean(0); bi = int(per.argmax())
    oracle = R.max(1).mean(); best = per[bi]
    print("\n=== AGENTIC HEADROOM (SWE-Bench Verified) ===")
    print("per-worker resolve:", {wids[j]: round(per[j], 3) for j in range(len(wids))})
    print(f"best worker = {wids[bi]} ({best:.3f})")
    print(f"oracle (best-per-task) = {oracle:.3f}")
    print(f"headroom = {oracle - best:+.3f}   RER = {(oracle-best)/(1-best+1e-9):.3f}")
    solved_by = {tasks[ti]: [wids[j] for j in range(len(wids)) if R[ti, j] > 0] for ti in range(len(tasks))}
    print("solved-by (complementarity):")
    for t, ws in solved_by.items():
        if ws:
            print(f"  {t[:34]:34} {ws}")
    print(f"\ntotal spent: ${pool.budget.spent_usd:.2f}")


asyncio.run(main())
