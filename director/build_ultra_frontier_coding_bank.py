"""Build a paid frontier/open agentic-coding matrix for Ultra pool selection.

The coding stratum runs SWE-smith repo-repair tasks through Ultra's OpenCode
container harness. It records both direct solo workers and a preregistered set
of heterogeneous workflow arms. Subset scoring gives a pool credit for a solved
task only when the successful arm uses workers inside that pool.

Run with director's venv so SWE-smith/Docker deps are present:
  PYTHONPATH=../ultra YUNWU_API_KEY=... director/.venv/bin/python \
    build_ultra_frontier_coding_bank.py --tasks 2 --budget 40 --concurrency 1

When the active provider does not report cost telemetry, for example Yunwu, the
``cost`` fields are reported as zero/unknown and external spend monitoring is
authoritative. The ``--budget`` cap only applies to provider-reported cost.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ULTRA_ROOT = ROOT / "ultra"
if str(ULTRA_ROOT) not in sys.path:
    sys.path.insert(0, str(ULTRA_ROOT))

from director.agentic.runners import load_swesmith_tasks  # noqa: E402
from ultra.agentic_scaffolds import ag_direct, builder_debugger, debate_synth, ladder, self_repair  # noqa: E402
from ultra.harness.opencode import run_agentic_workflow  # noqa: E402
from ultra.providers import active as active_provider, provider as provider_cfg, slug as provider_slug  # noqa: E402

WORKER_MODEL_KEYS = {"kimi-code": "kimi"}
DEFAULT_WORKER_ORDER = ["opus", "gemini", "gpt", "glm", "flash", "mimo", "kimi-code", "minimax", "deepseek-pro"]
PROPOSED_POOL = ["opus", "gemini", "gpt", "glm", "flash", "mimo"]


def worker_slug(worker: str) -> str:
    return provider_slug(WORKER_MODEL_KEYS.get(worker, worker))


def default_manifest_dir() -> Path:
    return Path(__file__).resolve().parent / "manifests" / "fugu_clean_v1"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_arms(index: dict[str, int], stages: set[str] | None = None) -> list[dict]:
    arms = []
    for worker in DEFAULT_WORKER_ORDER:
        if worker in index:
            arms.append(
                {
                    "name": f"direct__{worker}",
                    "stage": "direct",
                    "workers": [worker],
                    "build": lambda worker=worker: ag_direct(index[worker]),
                }
            )
    candidate_specs = [
        ("self_repair__glm", "self", ["glm"], lambda: self_repair(index["glm"])),
        ("self_repair__opus", "self", ["opus"], lambda: self_repair(index["opus"])),
        ("builder__glm__debug__opus", "mixed", ["glm", "opus"], lambda: builder_debugger(index["glm"], index["opus"])),
        ("builder__flash__debug__opus", "mixed", ["flash", "opus"], lambda: builder_debugger(index["flash"], index["opus"])),
        ("builder__kimi_code__debug__opus", "mixed", ["kimi-code", "opus"], lambda: builder_debugger(index["kimi-code"], index["opus"])),
        ("builder__glm__debug__gpt", "mixed", ["glm", "gpt"], lambda: builder_debugger(index["glm"], index["gpt"])),
        ("ladder__glm__gemini__opus", "mixed", ["glm", "gemini", "opus"], lambda: ladder(index["glm"], index["gemini"], index["opus"])),
        (
            "debate__glm__flash__synth__opus",
            "mixed",
            ["glm", "flash", "opus"],
            lambda: debate_synth(index["glm"], index["flash"], index["opus"]),
        ),
        (
            "debate__opus__gpt__synth__gemini",
            "mixed",
            ["opus", "gpt", "gemini"],
            lambda: debate_synth(index["opus"], index["gpt"], index["gemini"]),
        ),
    ]
    for name, stage, workers, build in candidate_specs:
        if all(w in index for w in workers):
            arms.append({"name": name, "stage": stage, "workers": workers, "build": build})
    if stages is not None:
        arms = [arm for arm in arms if arm["stage"] in stages]
    return arms


def load_done(path: Path) -> set[tuple[str, str]]:
    return {(row["task_id"], row["arm"]) for row in read_jsonl(path)}


def summarize(path: Path, worker_order: list[str]) -> dict:
    rows = read_jsonl(path)
    by_arm: dict[str, list[dict]] = defaultdict(list)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
        by_task[row["task_id"]].append(row)

    arms = {}
    for arm, arm_rows in sorted(by_arm.items()):
        valid = [r for r in arm_rows if r.get("valid", True)]
        arms[arm] = {
            "n": len(arm_rows),
            "valid": len(valid),
            "success_rate": (sum(r["reward"] >= 1.0 for r in valid) / len(valid)) if valid else None,
            "reported_cost_usd": sum(float(r.get("cost", 0.0)) for r in arm_rows),
            "workers": arm_rows[0].get("workers", []) if arm_rows else [],
            "errors": dict(Counter(r.get("error") for r in arm_rows if not r.get("valid", True))),
        }

    task_ids = sorted(by_task)

    def score(pool: tuple[str, ...]) -> float | None:
        if not task_ids:
            return None
        allowed = set(pool)
        solved = 0
        for task_id in task_ids:
            for row in by_task[task_id]:
                if set(row.get("workers", [])) <= allowed and row.get("valid", True) and row.get("reward", 0.0) >= 1.0:
                    solved += 1
                    break
        return solved / len(task_ids)

    best_by_size = {}
    for k in range(1, min(6, len(worker_order)) + 1):
        best = -1.0
        tied = []
        for subset in combinations(worker_order, k):
            s = score(subset)
            if s is None:
                continue
            if s > best:
                best = s
                tied = [list(subset)]
            elif s == best:
                tied.append(list(subset))
        best_by_size[str(k)] = {"score": best if best >= 0 else None, "subsets": tied[:10], "n_tied": len(tied)}

    proposed_score = score(tuple(PROPOSED_POOL))
    loo = {}
    for worker in PROPOSED_POOL:
        without = score(tuple(w for w in PROPOSED_POOL if w != worker))
        loo[worker] = {
            "score_without": without,
            "delta_kept": None if proposed_score is None or without is None else proposed_score - without,
        }

    return {
        "n_rows": len(rows),
        "n_tasks": len(task_ids),
        "total_reported_cost_usd": sum(float(r.get("cost", 0.0)) for r in rows),
        "cost_note": "Provider-reported cost only; external ledger is authoritative when provider emits no cost.",
        "arms": arms,
        "task_oracle": score(tuple(worker_order)),
        "best_by_size": best_by_size,
        "proposed_pool": PROPOSED_POOL,
        "proposed_score": proposed_score,
        "proposed_leave_one_out": loo,
    }


async def run(args: argparse.Namespace) -> dict:
    provider = provider_cfg()
    key_env = os.environ.get("ULTRA_OC_KEY_ENV", provider["key_env"])
    if not os.environ.get(key_env) and not args.dry_run:
        raise SystemExit(f"{key_env} is not set")

    worker_order = [w for w in args.workers.split(",") if w]
    index = {worker: i for i, worker in enumerate(worker_order)}
    slugs = [worker_slug(w) for w in worker_order]
    stages = set(args.stages.split(",")) if args.stages else None
    arms = build_arms(index, stages=stages)
    if args.arms:
        allowed = set(args.arms.split(","))
        arms = [arm for arm in arms if arm["name"] in allowed]

    loaded = load_swesmith_tasks(args.tasks)
    tasks = [{"task_id": row["item_id"], "payload": row["payload"]} for row in loaded]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out) if args.resume else set()
    jobs = [(task, arm) for task in tasks for arm in arms if (task["task_id"], arm["name"]) not in done]
    existing_reported_cost = sum(float(row.get("cost", 0.0)) for row in read_jsonl(out)) if args.resume else 0.0
    plan = {
        "tasks": len(tasks),
        "arms": [arm["name"] for arm in arms],
        "jobs": len(jobs),
        "provider": active_provider(),
        "key_env": key_env,
        "workers": {w: worker_slug(w) for w in worker_order},
        "existing_reported_cost_usd": existing_reported_cost,
        "reported_cost_budget_usd": args.budget,
        "cost_note": "Provider-reported cost only; external ledger is authoritative when provider emits no cost.",
        "stages": sorted(stages) if stages else None,
    }
    plan_path = out.with_suffix(".plan.json")
    plan_path.write_text(json.dumps({"plan": plan, "tasks": [t["task_id"] for t in tasks]}, indent=2))
    if args.dry_run:
        return {"plan": plan, "summary": None, "plan_path": str(plan_path)}

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    started = time.time()
    reported_cost = existing_reported_cost

    async def one(task: dict, arm: dict) -> None:
        nonlocal reported_cost
        async with sem:
            if args.budget >= 0 and reported_cost >= args.budget:
                return
            try:
                result = await run_agentic_workflow(
                    task["payload"],
                    arm["build"](),
                    slugs,
                    os.environ[key_env],
                )
                valid = bool(result.get("valid", True))
                reward = float(result.get("reward", 0.0))
                cost = float(result.get("cost", 0.0))
                error = result.get("error")
                steps = result.get("steps", [])
            except Exception as exc:
                valid, reward, cost, error, steps = False, 0.0, 0.0, type(exc).__name__, []
        row = {
            "task_id": task["task_id"],
            "arm": arm["name"],
            "stage": arm["stage"],
            "workers": arm["workers"],
            "reward": reward,
            "cost": cost,
            "valid": valid,
            "error": error,
            "steps": steps,
            "elapsed_s": time.time() - started,
        }
        async with lock:
            reported_cost += cost
            with out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            tag = f"r={reward:.0f} reported_cost=${cost:.3f}" if valid else f"FAIL:{error}"
            print(
                f"[{out.name}] {task['task_id']}/{arm['name']} {tag} | "
                f"reported_total=${reported_cost:.2f}",
                flush=True,
            )

    await asyncio.gather(*(one(task, arm) for task, arm in jobs))
    summary = summarize(out, worker_order)
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    return {"plan": plan, "summary": summary, "plan_path": str(plan_path), "summary_path": str(summary_path)}


def parse_args() -> argparse.Namespace:
    manifest_dir = default_manifest_dir()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(manifest_dir / "agentic_coding_frontier_bank.jsonl"))
    parser.add_argument("--workers", default=",".join(DEFAULT_WORKER_ORDER))
    parser.add_argument("--tasks", type=int, default=2)
    parser.add_argument("--stages", default="direct,self,mixed")
    parser.add_argument("--arms", default=None)
    parser.add_argument("--budget", type=float, default=50.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(asyncio.run(run(parse_args())), indent=2))


if __name__ == "__main__":
    main()
