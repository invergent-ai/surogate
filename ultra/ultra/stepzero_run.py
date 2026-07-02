"""Live step-zero entrypoint: build a real pool, sample bank tasks, run the headroom test.

Drives ``ultra.stepzero.run_stepzero`` (cross-fitted Δ_fixed + null-corrected oracle) against
an OpenRouter-backed WorkerPool over the runnable (direct_qa / code_exec) bank tasks.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from .config import PoolConfig, WorkerSpec
from .providers import load_dotenv, required_key_envs
from .scaffolds import SCAFFOLDS
from .sources.existing_bank import ExistingBankAdapter
from .stepzero import HeadroomReport, run_stepzero
from .workers import Sampling
from .workers.factory import build_pool


def parse_workers(spec: str) -> list[WorkerSpec]:
    """Parse ``id=model_slug,id2=slug2`` into WorkerSpecs (worker index 0 = anchor)."""
    workers: list[WorkerSpec] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"worker {part!r} must be 'id=model_slug'")
        wid, slug = part.split("=", 1)
        workers.append(WorkerSpec(worker_id=wid.strip(), model=slug.strip()))
    if not workers:
        raise ValueError("no workers parsed from --workers")
    return workers


def sample_bank_tasks(
    n: int,
    split: str = "grpo_train",
    harness: str = "direct_qa",
    discriminative: bool = True,
    bank_path: str | None = None,
    seed: int = 0,
):
    adapter = ExistingBankAdapter(bank_path) if bank_path else ExistingBankAdapter()
    tasks = list(
        adapter.materialize_all(
            split=split,
            harness=harness,
            verdict="discriminative" if discriminative else None,
            limit=n,
            shuffle=True,
            seed=seed,
        )
    )
    return adapter, tasks


def _verdict(report: HeadroomReport) -> str:
    lo, hi = report.delta_fixed_ci
    if lo > 0:
        gate = "clears the ~3pt gate" if lo >= 0.03 else "below the ~3pt gate — borderline"
        return (
            f"GO: cross-fitted Δ_fixed = {report.delta_fixed_cv:+.3f} "
            f"(95% CI [{lo:+.3f}, {hi:+.3f}]) is above 0 and {gate}."
        )
    if report.oracle_signal >= 0.05:
        return (
            f"MAYBE: no fixed scaffold wins (Δ_fixed CI [{lo:+.3f}, {hi:+.3f}] includes 0), "
            f"but null-corrected oracle signal = {report.oracle_signal:+.3f} ≥ 0.05 → real "
            f"per-task headroom exists. Train a query-selector and require it to capture ≥0.02."
        )
    return (
        f"NO-GO: Δ_fixed CI [{lo:+.3f}, {hi:+.3f}] includes/below 0 AND null-corrected oracle "
        f"signal {report.oracle_signal:+.3f} < 0.05 — same regime as the Fugu null. Don't start "
        f"GRPO; check scaffold prompts / pool / task difficulty first."
    )


def format_report(report: HeadroomReport, worker_ids: list[str], harness: str) -> str:
    lines = [
        f"\n=== step-zero headroom ({report.n_tasks} tasks × {report.n_reps} reps, harness={harness}) ===",
        f"best single worker : {report.best_single:.3f}",
        "  per worker:",
    ]
    for idx, acc in sorted(report.single_acc.items()):
        wid = worker_ids[idx] if idx < len(worker_ids) else str(idx)
        lines.append(f"    [{idx}] {wid:<20} {acc:.3f}")
    lines.append("scaffolds (in-sample mean acc):")
    for name, acc in sorted(report.scaffold_acc.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {name:<28} {acc:.3f}")
    lo, hi = report.delta_fixed_ci
    lines += [
        "cross-fitted Δ_fixed (best scaffold − best single, dev-select / held-out-score):",
        f"    Δ_fixed = {report.delta_fixed_cv:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]",
        "workflow oracle:",
        f"    observed   {report.oracle_obs:.3f}",
        f"    perm-null  {report.oracle_null:.3f}   (no task↔scaffold interaction)",
        f"    signal     {report.oracle_signal:+.3f}   (observed − null)",
        f"spend              : ${report.total_cost_usd:.4f}",
        f"\nverdict: {_verdict(report)}",
    ]
    return "\n".join(lines)


async def run_cli(args) -> None:
    workers = parse_workers(args.workers)
    load_dotenv()
    missing = [key_env for key_env in required_key_envs([worker.model for worker in workers]) if not os.environ.get(key_env)]
    if missing:
        raise SystemExit(f"missing provider key env vars: {', '.join(missing)}")
    pool = build_pool(
        workers,
        PoolConfig(max_concurrency=args.concurrency, budget_usd=args.budget),
    )
    _adapter, tasks = sample_bank_tasks(
        args.n_tasks,
        split=args.split,
        harness=args.harness,
        discriminative=not args.all_difficulties,
        bank_path=args.bank_path,
        seed=args.seed,
    )
    if not tasks:
        raise SystemExit(
            f"no tasks matched split={args.split} harness={args.harness} "
            f"discriminative={not args.all_difficulties}"
        )

    scaffold_calls = sum(len(build().steps) for build in SCAFFOLDS.values())
    est = len(tasks) * args.reps * (len(workers) + scaffold_calls)
    print(
        f"step-zero: {len(tasks)} tasks × {args.reps} reps × ({len(workers)} workers + "
        f"{len(SCAFFOLDS)} scaffolds) ≈ {est} worker calls upper bound (cache hits free)",
        flush=True,
    )

    sampling = Sampling(
        temperature=args.temperature, max_tokens=args.max_tokens, reasoning_effort=args.reasoning
    )
    assignment: dict[str, tuple] = {}
    for part in (args.worker_assignment or "").split(";"):
        part = part.strip()
        if part:
            name, idxs = part.split("=", 1)
            assignment[name.strip()] = tuple(int(x) for x in idxs.split(","))
    if assignment:
        print(f"  worker-assignment: {assignment}", flush=True)

    report = await run_stepzero(
        tasks, pool, sampling, worker_assignment=assignment,
        n_reps=args.reps, n_folds=args.folds, seed=args.seed
    )
    print(format_report(report, pool.worker_ids, args.harness), flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(asdict(report), indent=2))
        print(f"\nwrote report -> {args.out}", flush=True)
