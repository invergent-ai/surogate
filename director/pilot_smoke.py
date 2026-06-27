"""1-task SWE-Bench smoke: confirm a real agentic rollout + grading works end-to-end
with an open worker, before scaling to the headroom baseline. Capped at $5."""
from __future__ import annotations
import asyncio, time
import torch
from director.config import DirectorConfig, PoolConfig, default_frontier_pool
from director.fugu.run import build_router, _sampling
from director.shared.providers import build_pool
from director.agentic.swebench_env import load_swebench, build_swebench_factories
from director.agentic.rollout import agentic_rollout

cfg = DirectorConfig(workers=default_frontier_pool())
WORKER = "kimi"  # code specialist in the open pool
pool = build_pool(PoolConfig(budget_usd=5.0), cfg.workers)
insts = load_swebench(limit=1, shuffle=True, seed=0)
print(f"task: {insts[0]['instance_id']}")
fac = build_swebench_factories(insts, step_timeout=120.0)[0]

router = build_router(cfg)
with torch.no_grad():  # force the router to always pick WORKER
    router.head.weight.zero_()
    router.head.weight[cfg.worker_ids.index(WORKER), :] = 10.0

async def main():
    env = fac()
    t0 = time.time()
    res = await agentic_rollout(router, pool, env, max_turns=15, sampling=_sampling(cfg))
    env.close()
    print(f"\nworker={WORKER} reward={res.reward} turns={res.turns} submitted={res.submitted} "
          f"cost=${res.cost_usd:.4f} wall={time.time()-t0:.0f}s")
    print(f"pool spent=${pool.budget.spent_usd:.4f}")
    print("worker seq:", res.worker_sequence[:8], "...")

asyncio.run(main())
