"""Build a paid frontier/open agentic matrix for Ultra pool selection.

This is intentionally separate from build_agentic_bank.py. The existing bank is
the six-open-worker historical evidence; this script measures the current Ultra
candidate pool on the same tau-bench retail/airline task ids when possible.

Example:
  OPENROUTER_API_KEY=... director/.venv/bin/python build_ultra_frontier_agentic_bank.py \
    --existing-bank manifests/fugu_clean_v1/agentic_bank.jsonl \
    --out manifests/fugu_clean_v1/agentic_frontier_bank.jsonl \
    --tasks-per-domain 2 --budget 20 --concurrency 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import combinations
from pathlib import Path

from director.agentic.taubench_env import TauBenchEnv
from director.agentic.toolcall import TOOL_SYSTEM
from director.agentic.toolenv import RESPOND, ToolAction
from director.config import PoolConfig, WorkerSpec
from director.shared.providers import build_pool
from director.shared.types import Sampling

WORKER_SPECS = [
    WorkerSpec(worker_id="opus", model="anthropic/claude-opus-4.8"),
    WorkerSpec(worker_id="gemini", model="google/gemini-3.1-pro-preview"),
    WorkerSpec(worker_id="gpt", model="openai/gpt-5.5"),
    WorkerSpec(worker_id="glm", model="z-ai/glm-5.2", provider_sort=None),
    WorkerSpec(worker_id="flash", model="deepseek/deepseek-v4-flash"),
    WorkerSpec(worker_id="mimo", model="xiaomi/mimo-v2.5-pro"),
    WorkerSpec(worker_id="kimi-code", model="moonshotai/kimi-k2.7-code"),
    WorkerSpec(worker_id="minimax", model="minimax/minimax-m3"),
    WorkerSpec(worker_id="deepseek-pro", model="deepseek/deepseek-v4-pro"),
]
PROPOSED_POOL = ["opus", "gemini", "gpt", "glm", "flash", "mimo"]
DOMAINS = ["tau_retail", "tau_airline"]


def default_manifest_dir() -> Path:
    return Path(__file__).resolve().parent / "manifests" / "fugu_clean_v1"


def read_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    return (json.loads(line) for line in path.read_text().splitlines() if line.strip())


def parse_tau_item(domain: str, item_id: str) -> tuple[str, int]:
    if domain == "tau_retail":
        prefix = "tau-retail-"
        env_name = "retail"
    elif domain == "tau_airline":
        prefix = "tau-airline-"
        env_name = "airline"
    else:
        raise ValueError(f"unsupported tau domain {domain!r}")
    if not item_id.startswith(prefix):
        raise ValueError(f"{item_id!r} does not match {domain!r}")
    return env_name, int(item_id.removeprefix(prefix))


def select_tasks(
    existing_bank: Path,
    *,
    tasks_per_domain: int,
    seed: int,
    difficulty: str,
    min_existing_workers: int,
) -> list[dict]:
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in read_jsonl(existing_bank):
        domain = row.get("domain")
        if domain in DOMAINS:
            grouped[(domain, row["item_id"])][row["worker"]] = float(row["reward"])

    by_domain: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (domain, item_id), rewards in grouped.items():
        if len(rewards) < min_existing_workers:
            continue
        successes = sum(v >= 1.0 for v in rewards.values())
        if difficulty == "discriminative" and not (0 < successes < len(rewards)):
            continue
        if difficulty == "hard" and not (0 < successes <= 2):
            continue
        if difficulty == "unsolved" and successes != 0:
            continue
        by_domain[domain].append((item_id, successes))

    rng = random.Random(seed)
    selected = []
    for domain in DOMAINS:
        items = by_domain[domain]
        rng.shuffle(items)
        for item_id, successes in items[:tasks_per_domain]:
            env_name, idx = parse_tau_item(domain, item_id)
            selected.append(
                {
                    "domain": domain,
                    "item_id": item_id,
                    "env_name": env_name,
                    "task_index": idx,
                    "existing_open_successes": successes,
                }
            )
    rng.shuffle(selected)
    return selected


def load_done(path: Path) -> set[tuple[str, str]]:
    done = set()
    for row in read_jsonl(path):
        done.add((row["item_id"], row["worker"]))
    return done


async def run_tau_solo(
    pool,
    worker_id: str,
    task: dict,
    *,
    sampling: Sampling,
    tau_user_model: str,
    max_turns: int,
) -> tuple[float, float]:
    env = TauBenchEnv(
        task["env_name"],
        task["task_index"],
        user_model=tau_user_model,
        user_provider="openrouter",
    )
    try:
        user_msg, tools = await asyncio.to_thread(env.reset)
        messages = [{"role": "system", "content": TOOL_SYSTEM}, {"role": "user", "content": user_msg}]
        cost = 0.0
        for _ in range(max_turns):
            resp = await pool.call_tools(worker_id, messages, tools, sampling)
            cost += resp.cost_usd
            if resp.tool_calls:
                tc = resp.tool_calls[0]
                messages.append(
                    {
                        "role": "assistant",
                        "content": resp.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                            }
                        ],
                    }
                )
                step = await asyncio.to_thread(env.step, ToolAction(name=tc.name, arguments=tc.arguments))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": step.observation})
            else:
                messages.append({"role": "assistant", "content": resp.content or ""})
                step = await asyncio.to_thread(
                    env.step,
                    ToolAction(name=RESPOND, arguments={"content": resp.content or ""}),
                )
                messages.append({"role": "user", "content": step.observation})
            if step.done:
                break
        return float(await asyncio.to_thread(env.reward)), cost
    finally:
        await asyncio.to_thread(env.close)


def summarize(path: Path, workers: list[str]) -> dict:
    rows = list(read_jsonl(path))
    by_worker: dict[str, list[dict]] = defaultdict(list)
    by_task: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_worker[row["worker"]].append(row)
        by_task[row["item_id"]][row["worker"]] = float(row.get("reward", 0.0)) if row.get("valid", True) else 0.0

    worker_summary = {}
    for worker, wrs in sorted(by_worker.items()):
        valid = [r for r in wrs if r.get("valid", True)]
        worker_summary[worker] = {
            "n": len(wrs),
            "valid": len(valid),
            "success_rate": (sum(r["reward"] >= 1.0 for r in valid) / len(valid)) if valid else None,
            "cost_usd": sum(float(r.get("cost", 0.0)) for r in wrs),
            "errors": dict(Counter(r.get("error") for r in wrs if not r.get("valid", True))),
        }

    task_ids = sorted(by_task)

    def score(subset: tuple[str, ...]) -> float | None:
        if not task_ids:
            return None
        return sum(any(by_task[tid].get(w, 0.0) >= 1.0 for w in subset) for tid in task_ids) / len(task_ids)

    best_by_size = {}
    for k in range(1, min(6, len(workers)) + 1):
        best_score = -1.0
        tied = []
        for subset in combinations(workers, k):
            s = score(subset)
            if s is None:
                continue
            if s > best_score:
                best_score = s
                tied = [list(subset)]
            elif s == best_score:
                tied.append(list(subset))
        best_by_size[str(k)] = {
            "score": best_score if best_score >= 0 else None,
            "subsets": tied[:10],
            "n_tied": len(tied),
        }

    proposed_score = score(tuple(PROPOSED_POOL))
    loo = {}
    for worker in PROPOSED_POOL:
        kept = tuple(w for w in PROPOSED_POOL if w != worker)
        without = score(kept)
        loo[worker] = {
            "score_without": without,
            "delta_kept": None if proposed_score is None or without is None else proposed_score - without,
        }

    return {
        "n_rows": len(rows),
        "n_tasks": len(task_ids),
        "total_cost_usd": sum(float(r.get("cost", 0.0)) for r in rows),
        "workers": worker_summary,
        "task_oracle": score(tuple(workers)),
        "best_by_size": best_by_size,
        "proposed_pool": PROPOSED_POOL,
        "proposed_score": proposed_score,
        "proposed_leave_one_out": loo,
    }


async def run(args: argparse.Namespace) -> dict:
    if not os.environ.get("OPENROUTER_API_KEY") and not args.dry_run:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    workers = [w for w in WORKER_SPECS if w.worker_id in set(args.workers.split(","))]
    worker_ids = [w.worker_id for w in workers]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tasks = select_tasks(
        Path(args.existing_bank),
        tasks_per_domain=args.tasks_per_domain,
        seed=args.seed,
        difficulty=args.difficulty,
        min_existing_workers=args.min_existing_workers,
    )
    done = load_done(out) if args.resume else set()
    jobs = [(task, worker_id) for task in tasks for worker_id in worker_ids if (task["item_id"], worker_id) not in done]
    existing_spend = sum(float(row.get("cost", 0.0)) for row in read_jsonl(out)) if args.resume else 0.0
    remaining_budget = max(0.0, args.budget - existing_spend)
    plan = {
        "tasks": len(tasks),
        "domains": dict(Counter(t["domain"] for t in tasks)),
        "workers": {w.worker_id: w.model for w in workers},
        "jobs": len(jobs),
        "existing_spend_usd": existing_spend,
        "remaining_budget_usd": remaining_budget,
        "difficulty": args.difficulty,
        "tasks_per_domain": args.tasks_per_domain,
    }
    plan_path = out.with_suffix(".plan.json")
    plan_path.write_text(json.dumps({"plan": plan, "tasks": tasks}, indent=2))
    if args.dry_run:
        return {"plan": plan, "summary": None, "plan_path": str(plan_path)}

    pool = build_pool(
        PoolConfig(
            budget_usd=remaining_budget,
            max_concurrency=args.concurrency,
            timeout_s=args.timeout,
            max_retries=args.max_retries,
            cache_dir=args.cache_dir,
        ),
        workers,
    )
    sampling = Sampling(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
        reasoning_effort=args.reasoning,
    )
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    started = time.time()

    async def one(task: dict, worker_id: str) -> None:
        async with sem:
            valid = True
            error = None
            try:
                reward, cost = await run_tau_solo(
                    pool,
                    worker_id,
                    task,
                    sampling=sampling,
                    tau_user_model=args.tau_user_model,
                    max_turns=args.max_turns,
                )
            except Exception as exc:  # record model/provider/tool failures; resume should not loop forever
                reward, cost = 0.0, 0.0
                valid = False
                error = type(exc).__name__
        row = {
            "domain": task["domain"],
            "item_id": task["item_id"],
            "task_index": task["task_index"],
            "existing_open_successes": task["existing_open_successes"],
            "worker": worker_id,
            "reward": float(reward),
            "cost": float(cost),
            "valid": valid,
            "error": error,
            "elapsed_s": time.time() - started,
        }
        async with lock:
            with out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            tag = f"r={reward:.0f} ${cost:.3f}" if valid else f"FAIL:{error}"
            print(f"[{out.name}] {task['domain']}/{task['item_id']}/{worker_id} {tag}", flush=True)

    await asyncio.gather(*(one(task, worker_id) for task, worker_id in jobs))
    summary = summarize(out, worker_ids)
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    return {"plan": plan, "summary": summary, "plan_path": str(plan_path), "summary_path": str(summary_path)}


def parse_args() -> argparse.Namespace:
    manifest_dir = default_manifest_dir()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-bank", default=str(manifest_dir / "agentic_bank.jsonl"))
    parser.add_argument("--out", default=str(manifest_dir / "agentic_frontier_bank.jsonl"))
    parser.add_argument("--workers", default=",".join(w.worker_id for w in WORKER_SPECS))
    parser.add_argument("--tasks-per-domain", type=int, default=4)
    parser.add_argument("--difficulty", choices=["all", "discriminative", "hard", "unsolved"], default="discriminative")
    parser.add_argument("--min-existing-workers", type=int, default=6)
    parser.add_argument("--budget", type=float, default=50.0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--reasoning", default="high")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau-user-model", default="openrouter/openai/gpt-5-mini")
    parser.add_argument("--cache-dir", default="./.director_cache/ultra_agentic_frontier")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(asyncio.run(run(parse_args())), indent=2))


if __name__ == "__main__":
    main()
