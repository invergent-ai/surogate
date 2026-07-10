"""Pool-candidate solo probe (generalized from the grok probe): any worker, any manifest, any sampling.

Answers the two pool questions on OUR tasks (benchmark-win != our-harness win):
  1. capability: solo rate vs the existing premium rows (gemini-pro 0.846, opus 0.731,
     gpt 0.615, glm 0.154 on fshard);
  2. complementarity: per-task overlap — does grok solve tasks the current pool misses
     (raises the oracle) or a subset of what gemini-pro solves (dominated)?

Run: ... probe_worker_solo.py --worker gpt-terra --manifest heldout_trend60_taskspecs.jsonl --handicap
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
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--worker", required=True)          # logical model, e.g. gpt-terra / gpt-sol / grok
ap.add_argument("--manifest", default="heldout_fshard_taskspecs.jsonl")
ap.add_argument("--handicap", action="store_true")  # 4096/minimal (training protocol) vs 16384..32768/high
ap.add_argument("--conc", type=int, default=3)
ap.add_argument("--budget", type=float, default=25.0)
args = ap.parse_args()
MANIFEST = f"{D}/{args.manifest}"
OUT = Path(f"scratchpad/{args.worker}_solo_{'hc' if args.handicap else 'fs'}_{Path(args.manifest).stem[:12]}.jsonl")
SAMP = (Sampling(temperature=0.2, top_p=1.0, max_tokens=4096, reasoning_effort="minimal") if args.handicap
        else Sampling(temperature=0.2, top_p=1.0, max_tokens=32768, reasoning_effort="high"))

tasks = [TaskSpec.model_validate(json.loads(l)) for l in open(MANIFEST)]
done = set()
if OUT.exists():
    done = {json.loads(l)["task_id"] for l in open(OUT)}

pool = build_pool(
    [WorkerSpec(worker_id=args.worker, model=args.worker)],
    PoolConfig(split_provider_routing=True, cache_dir=".ultra_cache/eval_fullstrength",
               max_concurrency=args.conc, timeout_s=1800.0, max_retries=3, budget_usd=args.budget),
)


async def one(task):
    if task.task_id in done:
        return
    try:
        c = await pool.call(args.worker, [dict(m) for m in task.input.messages], SAMP)
        score = await asyncio.wait_for(asyncio.to_thread(
            get_grader(task.grader.type), c.text or "", task.grader.expected_answer), timeout=90.0)
        row = {"task_id": task.task_id, "arm": f"solo__{args.worker}", "cap": task.capability,
               "score": float(score >= task.grader.success_threshold),
               "status": "ok", "finish": c.finish_reason,
               "completion_tokens": c.completion_tokens, "cost": c.cost_usd}
    except Exception as e:
        row = {"task_id": task.task_id, "arm": f"solo__{args.worker}", "cap": task.capability,
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
    print(f"\n{args.worker} solo: {rate:.3f} (n={len(rows)}) | ${cost:.2f} | {toks} completion toks")
    # overlap vs the CURRENT-pool fshard rows (pro rows are banned/deprecated)
    import os as _os
    ref_path = "scratchpad/fshard_rows.jsonl"
    if "fshard" not in args.manifest or not _os.path.exists(ref_path):
        return  # overlap analysis only defined for the fshard manifest
    prem = {}
    for line in open(ref_path):
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
