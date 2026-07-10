"""Grok-4.5 pool-candidate probe: full-strength solos on fshard-26 (+ trend60 optional).

Answers the two pool questions on OUR tasks (benchmark-win != our-harness win):
  1. capability: solo rate vs the existing premium rows (gemini-pro 0.846, opus 0.731,
     gpt 0.615, glm 0.154 on fshard);
  2. complementarity: per-task overlap — does grok solve tasks the current pool misses
     (raises the oracle) or a subset of what gemini-pro solves (dominated)?

Run: ULTRA_ALLOW_YUNWU=1 PYTHONPATH=ultra .venv/bin/python scratchpad/probe_grok_solo.py
"""
import asyncio, json, sys
from pathlib import Path

sys.path.insert(0, "ultra")
from dotenv import load_dotenv
load_dotenv()

from ultra.workers import Sampling
from ultra.workers.factory import build_pool
from ultra.config import PoolConfig, WorkerSpec
from ultra.grading import get_grader
from ultra.schemas import TaskSpec

D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
MANIFEST = f"{D}/heldout_fshard_taskspecs.jsonl"
OUT = Path("scratchpad/grok_solo_rows.jsonl")
SAMP = Sampling(temperature=0.2, top_p=1.0, max_tokens=32768, reasoning_effort="high")

tasks = [TaskSpec.model_validate(json.loads(l)) for l in open(MANIFEST)]
done = set()
if OUT.exists():
    done = {json.loads(l)["task_id"] for l in open(OUT)}

pool = build_pool(
    [WorkerSpec(worker_id="grok", model="grok")],
    PoolConfig(split_provider_routing=True, cache_dir=".ultra_cache/eval_fullstrength",
               max_concurrency=6, timeout_s=1800.0, max_retries=3, budget_usd=25.0),
)


async def one(task):
    if task.task_id in done:
        return
    try:
        c = await pool.call("grok", [dict(m) for m in task.input.messages], SAMP)
        score = await asyncio.wait_for(asyncio.to_thread(
            get_grader(task.grader.type), c.text or "", task.grader.expected_answer), timeout=90.0)
        row = {"task_id": task.task_id, "arm": "solo__grok", "cap": task.capability,
               "score": float(score >= task.grader.success_threshold),
               "status": "ok", "finish": c.finish_reason,
               "completion_tokens": c.completion_tokens, "cost": c.cost_usd}
    except Exception as e:
        row = {"task_id": task.task_id, "arm": "solo__grok", "cap": task.capability,
               "score": 0.0, "status": f"error:{type(e).__name__}"}
    with OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"{row['task_id']}: {row['score']} ({row['status']})", flush=True)


async def main():
    await asyncio.gather(*[one(t) for t in tasks])
    rows = [json.loads(l) for l in open(OUT)]
    rate = sum(r["score"] for r in rows) / len(rows)
    cost = sum(r.get("cost") or 0 for r in rows)
    toks = sum(r.get("completion_tokens") or 0 for r in rows)
    print(f"\nGROK-4.5 fshard solo: {rate:.3f} (n={len(rows)}) | ${cost:.2f} | {toks} completion toks")
    # overlap vs existing premium rows
    prem = {}
    for line in open("scratchpad/fshard_pro_rows.jsonl"):
        r = json.loads(line)
        prem.setdefault(r["task_id"], {})[r["arm"]] = r["score"]
    W = ["solo__st_opus", "solo__st_gemini", "solo__st_gpt", "solo__st_glm"]
    grok = {r["task_id"]: r["score"] for r in rows}
    only_grok = [t for t in grok if grok[t] == 1 and all(prem.get(t, {}).get(w, 0) == 0 for w in W)]
    grok_missed = [t for t in grok if grok[t] == 0 and any(prem.get(t, {}).get(w, 0) == 1 for w in W)]
    old_oracle = sum(1 for t in grok if any(prem.get(t, {}).get(w, 0) == 1 for w in W)) / len(grok)
    new_oracle = sum(1 for t in grok if grok[t] == 1 or any(prem.get(t, {}).get(w, 0) == 1 for w in W)) / len(grok)
    print(f"grok-only solves (raise the oracle): {only_grok}")
    print(f"grok misses that others solve: {len(grok_missed)}")
    print(f"pool oracle: {old_oracle:.3f} -> {new_oracle:.3f} with grok added")


asyncio.run(main())
