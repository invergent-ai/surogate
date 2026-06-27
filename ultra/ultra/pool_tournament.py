"""Bounded paid tournament for selecting the Ultra worker pool.

This is intentionally small and preregistered. It runs a fixed set of workflow arms
against a balanced validation sample, records every rollout, and stops before a spend
cap. The point is not to exhaustively search 9^3 workflows; it is to prove or falsify
the current six-worker hypothesis with the least OpenRouter spend.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Callable

from .config import PoolConfig, WorkerSpec
from .executor import execute_workflow
from .scaffolds import debate_synthesize, direct, plan_execute, solve_critique_revise
from .schemas import TaskSpec, Workflow
from .sources.existing_bank import ExistingBankAdapter
from .workers import Sampling
from .workers.budget import BudgetExceeded
from .workers.factory import build_pool


MODEL_SLUGS = {
    "flash": "deepseek/deepseek-v4-flash",
    "deepseek-pro": "deepseek/deepseek-v4-pro",
    "glm": "z-ai/glm-5.2",
    "kimi-code": "moonshotai/kimi-k2.7-code",
    "mimo": "xiaomi/mimo-v2.5-pro",
    "minimax": "minimax/minimax-m3",
    "opus": "anthropic/claude-opus-4.8",
    "gemini": "google/gemini-3.1-pro-preview",
    "gpt": "openai/gpt-5.5",
}

# OpenRouter /models catalog prices observed from https://openrouter.ai/api/v1/models.
# Values are USD per token in the catalog response; reports render them per million tokens.
MODEL_PRICING = {
    "flash": {"prompt": 0.00000009, "completion": 0.00000018},
    "deepseek-pro": {"prompt": 0.000000435, "completion": 0.00000087},
    "glm": {"prompt": 0.00000095, "completion": 0.000003},
    "kimi-code": {"prompt": 0.00000074, "completion": 0.0000035},
    "mimo": {"prompt": 0.000000435, "completion": 0.00000087},
    "minimax": {"prompt": 0.0000003, "completion": 0.0000012},
    "opus": {"prompt": 0.000005, "completion": 0.000025},
    "gemini": {"prompt": 0.000002, "completion": 0.000012},
    "gpt": {"prompt": 0.000005, "completion": 0.00003},
}

DEFAULT_WORKERS = ["opus", "gemini", "gpt", "glm", "flash", "mimo", "kimi-code", "minimax", "deepseek-pro"]
PROPOSED_POOL = ["opus", "gemini", "gpt", "glm", "flash", "mimo"]


@dataclass(frozen=True)
class Arm:
    name: str
    worker_names: tuple[str, ...]
    build: Callable[[dict[str, int]], Workflow]
    domains: tuple[str, ...] | None = None
    stage: str = "main"

    def applies(self, task: TaskSpec) -> bool:
        return self.domains is None or (task.metadata.domain or "") in self.domains


@dataclass(frozen=True)
class RolloutSummary:
    arm: str
    task_id: str
    domain: str | None
    success: bool | None
    reward: float | None
    cost_usd: float
    valid: bool
    failure_class: str | None


def _idx(name: str, index: dict[str, int]) -> int:
    return index[name]


def preregistered_arms() -> list[Arm]:
    """Fixed arms from ``pool_selection_report.md``.

    Main arms run on every sampled direct/code task. Challenger arms are domain-filtered
    so they only spend on the capabilities they could plausibly rescue.
    """

    arms: list[Arm] = []
    for worker in ["opus", "gemini", "gpt", "glm", "flash", "mimo", "kimi-code", "minimax", "deepseek-pro"]:
        arms.append(
            Arm(
                name=f"direct__{worker}",
                worker_names=(worker,),
                build=lambda ix, w=worker: direct(_idx(w, ix)),
                stage="single",
            )
        )
    for worker in ["opus", "gemini", "gpt", "glm", "flash"]:
        arms.append(
            Arm(
                name=f"scr__{worker}",
                worker_names=(worker,),
                build=lambda ix, w=worker: solve_critique_revise(_idx(w, ix)),
                stage="same_worker",
            )
        )

    arms.extend(
        [
            Arm(
                name="plan__gemini__solve__opus",
                worker_names=("gemini", "opus"),
                build=lambda ix: plan_execute(_idx("gemini", ix), _idx("opus", ix)),
                stage="mixed",
            ),
            Arm(
                name="debate__opus__gpt__synth__gemini",
                worker_names=("opus", "gpt", "gemini"),
                build=lambda ix: debate_synthesize(_idx("opus", ix), _idx("gpt", ix), _idx("gemini", ix)),
                stage="mixed",
            ),
            Arm(
                name="debate__flash__glm__synth__opus",
                worker_names=("flash", "glm", "opus"),
                build=lambda ix: debate_synthesize(_idx("flash", ix), _idx("glm", ix), _idx("opus", ix)),
                stage="mixed",
            ),
            Arm(
                name="solve__opus__critic__gpt__revise__opus",
                worker_names=("opus", "gpt"),
                build=lambda ix: Workflow(
                    steps=[
                        direct(_idx("opus", ix)).steps[0],
                        solve_critique_revise(_idx("gpt", ix)).steps[1].model_copy(update={"access": [0]}),
                        solve_critique_revise(_idx("opus", ix)).steps[2],
                    ]
                ),
                stage="mixed",
            ),
            Arm(
                name="solve__glm__critic__mimo__revise__glm",
                worker_names=("glm", "mimo"),
                build=lambda ix: Workflow(
                    steps=[
                        direct(_idx("glm", ix)).steps[0],
                        solve_critique_revise(_idx("mimo", ix)).steps[1].model_copy(update={"access": [0]}),
                        solve_critique_revise(_idx("glm", ix)).steps[2],
                    ]
                ),
                stage="mixed",
            ),
            Arm(
                name="scr__kimi_code__code_only",
                worker_names=("kimi-code",),
                build=lambda ix: solve_critique_revise(_idx("kimi-code", ix)),
                domains=("code",),
                stage="challenger",
            ),
            Arm(
                name="scr__deepseek_pro__reasoning_only",
                worker_names=("deepseek-pro",),
                build=lambda ix: solve_critique_revise(_idx("deepseek-pro", ix)),
                domains=("math", "science"),
                stage="challenger",
            ),
            Arm(
                name="solve__glm__critic__minimax__revise__glm",
                worker_names=("glm", "minimax"),
                build=lambda ix: Workflow(
                    steps=[
                        direct(_idx("glm", ix)).steps[0],
                        solve_critique_revise(_idx("minimax", ix)).steps[1].model_copy(update={"access": [0]}),
                        solve_critique_revise(_idx("glm", ix)).steps[2],
                    ]
                ),
                domains=("science", "general"),
                stage="challenger",
            ),
        ]
    )
    return arms


def build_workers(names: list[str]) -> list[WorkerSpec]:
    return [
        WorkerSpec(
            worker_id=name,
            model=MODEL_SLUGS[name],
            # GLM had known provider-routing issues in the existing project; use default
            # routing for it, price routing for the rest.
            provider_sort=None if name == "glm" else "price",
        )
        for name in names
    ]


def sample_balanced_tasks(
    manifest_path: Path,
    *,
    split: str,
    tasks_per_domain: int,
    seed: int,
    verdict: str | None,
    open_success_min: int | None = None,
    open_success_max: int | None = None,
) -> list[TaskSpec]:
    adapter = ExistingBankAdapter(manifest_path)
    open_success = _open_success_counts(manifest_path)
    by_domain: dict[str, list[TaskSpec]] = defaultdict(list)
    for task in adapter.materialize_all(split=split, verdict=verdict):
        raw_task_id = task.task_id.removeprefix("existing_bank__")
        n_success = open_success.get(raw_task_id)
        if open_success_min is not None and (n_success is None or n_success < open_success_min):
            continue
        if open_success_max is not None and (n_success is None or n_success > open_success_max):
            continue
        domain = task.metadata.domain or "unknown"
        if domain in {"math", "code", "science", "general"}:
            by_domain[domain].append(task)
    rnd = random.Random(seed)
    out: list[TaskSpec] = []
    for domain in ["math", "code", "science", "general"]:
        tasks = by_domain[domain]
        rnd.shuffle(tasks)
        out.extend(tasks[:tasks_per_domain])
    rnd.shuffle(out)
    return out


def _open_success_counts(manifest_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with manifest_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rewards = row.get("rewards")
            if isinstance(rewards, list):
                counts[str(row["task_id"])] = sum(float(x) > 0 for x in rewards)
    return counts


def estimate_rollout_count(tasks: list[TaskSpec], arms: list[Arm]) -> tuple[int, int]:
    rollouts = 0
    worker_calls = 0
    for task in tasks:
        for arm in arms:
            if arm.applies(task):
                rollouts += 1
                worker_calls += len(arm.build({name: i for i, name in enumerate(DEFAULT_WORKERS)}).steps)
    return rollouts, worker_calls


def _arm_index(arms: list[Arm] | None = None) -> dict[str, Arm]:
    return {arm.name: arm for arm in (arms or preregistered_arms())}


def _subset_success(
    records: list[RolloutSummary],
    subset: tuple[str, ...] | list[str],
    arm_by_name: dict[str, Arm],
) -> float:
    """Oracle over completed arms whose workers are contained in ``subset``.

    This answers: if the Conductor is allowed to use only this worker subset and can
    choose among the fixed tournament arms, what fraction of completed tasks has at
    least one successful eligible arm? Missing eligible records count as failure; the
    full run should therefore be preferred over partial summaries for final decisions.
    """

    allowed = set(subset)
    by_task: dict[str, list[RolloutSummary]] = defaultdict(list)
    for rec in records:
        if rec.valid and rec.success is not None:
            arm = arm_by_name.get(rec.arm)
            if arm is not None and set(arm.worker_names).issubset(allowed):
                by_task[rec.task_id].append(rec)
    all_tasks = sorted({rec.task_id for rec in records})
    if not all_tasks:
        return 0.0
    scores = []
    for task_id in all_tasks:
        rows = by_task.get(task_id, [])
        scores.append(max((float(r.success) for r in rows), default=0.0))
    return mean(scores)


def select_subsets_from_records(
    records: list[RolloutSummary],
    *,
    arms: list[Arm] | None = None,
    max_size: int = 6,
) -> dict:
    arm_by_name = _arm_index(arms)
    out: dict = {
        "n_tasks": len({rec.task_id for rec in records}),
        "max_size": max_size,
        "best_by_size": {},
        "proposed_pool": PROPOSED_POOL,
    }
    for size in range(1, max_size + 1):
        best = -1.0
        winners: list[tuple[str, ...]] = []
        for subset in itertools.combinations(DEFAULT_WORKERS, size):
            score = _subset_success(records, subset, arm_by_name)
            if score > best + 1e-12:
                best = score
                winners = [subset]
            elif abs(score - best) <= 1e-12:
                winners.append(subset)
        out["best_by_size"][str(size)] = {
            "score": best,
            "subsets": [list(w) for w in winners[:10]],
            "n_tied": len(winners),
        }

    proposed_score = _subset_success(records, PROPOSED_POOL, arm_by_name)
    out["proposed_score"] = proposed_score
    out["proposed_leave_one_out"] = {}
    for worker in PROPOSED_POOL:
        subset = [w for w in PROPOSED_POOL if w != worker]
        without = _subset_success(records, subset, arm_by_name)
        out["proposed_leave_one_out"][worker] = {
            "score_without": without,
            "delta_kept": proposed_score - without,
        }
    return out


def summarize_records(records: list[RolloutSummary]) -> dict:
    by_arm: dict[str, list[RolloutSummary]] = defaultdict(list)
    for rec in records:
        by_arm[rec.arm].append(rec)
    arms = {}
    for arm, rows in sorted(by_arm.items()):
        valid = [r for r in rows if r.valid and r.success is not None]
        arms[arm] = {
            "n": len(rows),
            "valid": len(valid),
            "success_rate": mean(float(r.success) for r in valid) if valid else None,
            "cost_usd": sum(r.cost_usd for r in rows),
            "domains": dict(Counter(r.domain for r in rows)),
        }

    # Per-task oracle over completed arms. This is descriptive only; selection should use
    # held-out paired comparisons after enough tasks finish.
    by_task: dict[str, list[RolloutSummary]] = defaultdict(list)
    for rec in records:
        by_task[rec.task_id].append(rec)
    task_oracle = []
    for rows in by_task.values():
        valid = [r for r in rows if r.valid and r.success is not None]
        if valid:
            task_oracle.append(max(float(r.success) for r in valid))
    return {
        "n_rollouts": len(records),
        "total_cost_usd": sum(r.cost_usd for r in records),
        "arms": arms,
        "task_oracle": mean(task_oracle) if task_oracle else None,
        "subset_selection": select_subsets_from_records(records),
    }


def load_rollout_summaries(path: Path) -> list[RolloutSummary]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(RolloutSummary(**json.loads(line)))
    return records


def analyze_rollout_file(path: Path) -> dict:
    return summarize_records(load_rollout_summaries(path))


async def run_tournament(args) -> dict:
    workers = build_workers(DEFAULT_WORKERS)
    index = {w.worker_id: i for i, w in enumerate(workers)}
    open_success_min = getattr(args, "open_success_min", None)
    open_success_max = getattr(args, "open_success_max", None)
    tasks = sample_balanced_tasks(
        Path(args.manifest_path),
        split=args.split,
        tasks_per_domain=args.tasks_per_domain,
        seed=args.seed,
        verdict=None if args.all_difficulties else "discriminative",
        open_success_min=open_success_min,
        open_success_max=open_success_max,
    )
    arms = preregistered_arms()
    if args.stages:
        allowed = set(args.stages.split(","))
        arms = [arm for arm in arms if arm.stage in allowed]
    if args.arms:
        allowed_arms = set(args.arms.split(","))
        arms = [arm for arm in arms if arm.name in allowed_arms]

    rollouts, worker_calls = estimate_rollout_count(tasks, arms)
    plan = {
        "tasks": len(tasks),
        "domains": dict(Counter(t.metadata.domain for t in tasks)),
        "arms": [arm.name for arm in arms],
        "rollouts": rollouts,
        "worker_calls": worker_calls,
        "workers": {w.worker_id: w.model for w in workers},
        "budget_usd": args.budget,
        "spend_stop_usd": args.budget * args.stop_ratio,
        "open_success_min": open_success_min,
        "open_success_max": open_success_max,
    }
    if args.dry_run:
        if args.out_dir:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "pool_tournament_plan.json").write_text(json.dumps(plan, indent=2))
        return {"plan": plan, "summary": None}

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set; export it in your shell before live runs")

    records: list[RolloutSummary] = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "pool_tournament_rollouts.jsonl"
    summary_path = out_dir / "pool_tournament_summary.json"
    plan_path = out_dir / "pool_tournament_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))

    if args.resume and records_path.exists():
        records = load_rollout_summaries(records_path)
    seen = {(r.arm, r.task_id) for r in records}

    spend_stop = args.budget * args.stop_ratio
    prior_spend = sum(r.cost_usd for r in records)
    remaining_budget = max(0.0, spend_stop - prior_spend)
    if remaining_budget <= 0.0:
        summary = summarize_records(records)
        summary["stopped"] = "spend stop already reached from resumed records"
        summary["plan"] = plan
        summary_path.write_text(json.dumps(summary, indent=2))
        return {"plan": plan, "summary": summary}

    pool = build_pool(
        workers,
        PoolConfig(
            max_concurrency=args.concurrency,
            budget_usd=remaining_budget,
            cache_dir=args.cache_dir,
            timeout_s=args.timeout,
            max_retries=args.max_retries,
        ),
    )
    sampling = Sampling(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
        reasoning_effort=args.reasoning,
    )

    jobs = [
        (task, arm)
        for task in tasks
        for arm in arms
        if arm.applies(task) and (arm.name, task.task_id) not in seen
    ]
    queue: asyncio.Queue[tuple[TaskSpec, Arm]] = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)

    write_lock = asyncio.Lock()
    stop_event = asyncio.Event()
    stop_reason: str | None = None

    async def append_record(row: RolloutSummary, f) -> None:
        async with write_lock:
            records.append(row)
            f.write(json.dumps(asdict(row)) + "\n")
            f.flush()

    async def run_one(task: TaskSpec, arm: Arm) -> RolloutSummary:
        rollout_id = f"poolsel-{args.seed}-{arm.name}-{task.task_id}"
        try:
            rec = await execute_workflow(
                task,
                arm.build(index),
                pool,
                sampling,
                rollout_id,
                worker_ids=[w.worker_id for w in workers],
            )
            cost = sum(step.cost_usd for step in rec.execution.steps)
            return RolloutSummary(
                arm=arm.name,
                task_id=task.task_id,
                domain=task.metadata.domain,
                success=bool(rec.grade.success) if rec.grade is not None else None,
                reward=rec.reward,
                cost_usd=cost,
                valid=rec.valid_for_training and rec.failure_class is None,
                failure_class=rec.failure_class,
            )
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 - provider/harness failures are heterogeneous
            return RolloutSummary(
                arm=arm.name,
                task_id=task.task_id,
                domain=task.metadata.domain,
                success=None,
                reward=None,
                cost_usd=0.0,
                valid=False,
                failure_class=f"{type(exc).__name__}: {exc}",
            )

    async def worker_loop(f) -> None:
        nonlocal stop_reason
        while not stop_event.is_set():
            try:
                task, arm = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                row = await run_one(task, arm)
                await append_record(row, f)
                if prior_spend + pool.budget.spent_usd >= spend_stop:
                    stop_reason = "spend stop reached"
                    stop_event.set()
            except BudgetExceeded as exc:
                stop_reason = str(exc)
                stop_event.set()
            finally:
                queue.task_done()

    with records_path.open("a") as f:
        workers_n = max(1, args.concurrency)
        await asyncio.gather(*(worker_loop(f) for _ in range(workers_n)))

    if stop_reason:
        summary = summarize_records(records)
        summary["stopped"] = stop_reason
        summary["plan"] = plan
        summary_path.write_text(json.dumps(summary, indent=2))
        return {"plan": plan, "summary": summary}

    summary = summarize_records(records)
    summary["stopped"] = None
    summary["plan"] = plan
    summary_path.write_text(json.dumps(summary, indent=2))
    return {"plan": plan, "summary": summary}


def default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[2] / "director" / "manifests" / "fugu_clean_v1" / "manifest.jsonl"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the bounded Ultra pool-selection tournament")
    parser.add_argument("--manifest-path", default=str(default_manifest_path()))
    parser.add_argument("--split", default="online_validation")
    parser.add_argument("--tasks-per-domain", type=int, default=10)
    parser.add_argument("--all-difficulties", action="store_true")
    parser.add_argument("--open-success-min", type=int, default=None)
    parser.add_argument("--open-success-max", type=int, default=None)
    parser.add_argument("--stages", default="single,same_worker,mixed,challenger")
    parser.add_argument("--arms", default=None, help="comma-separated arm names to run")
    parser.add_argument("--budget", type=float, default=200.0)
    parser.add_argument("--stop-ratio", type=float, default=0.8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--reasoning", default="high")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default="./.ultra_cache/completions")
    parser.add_argument("--out-dir", default="./pool_tournament")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    result = asyncio.run(run_tournament(args))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
