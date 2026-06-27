"""Agentic-coding step-zero — workflow headroom on SWE-smith via OpenCode (opencode_repo).

Mirrors ``stepzero.py`` but every (arm, task, rep) is a CONTAINER rollout
(``run_agentic_workflow``), not a chat call — so it reuses that module's cross-fitted Δ_fixed
and permutation-null oracle math verbatim. This is the rigorous re-test of the n=20 ladder
finding (single ~0.40 vs ladder ~0.65): does the agentic workflow headroom survive cross-fit +
the null, and is any of it learnable (per-task complementarity), unlike math/MC?

Run from director/.venv with ultra on PYTHONPATH (needs docker + swesmith + the ``oc`` binary):
  NTASKS=10 REPS=2 CONC=4 OUT=stepzero_agentic.json \\
    PYTHONPATH=<repo>/ultra OPENROUTER_API_KEY=... director/.venv/bin/python -m ultra.stepzero_agentic
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from statistics import mean, stdev

from .agentic_scaffolds import AGENTIC_SCAFFOLDS, ag_direct
from .harness.opencode import run_agentic_workflow
from .stepzero import HeadroomReport, _crossfit_delta_fixed, _oracle, _oracle_null

DEFAULT_SLUGS = "z-ai/glm-5.2,google/gemini-3.1-pro-preview,anthropic/claude-opus-4.8"


async def _arm_probs_agentic(instances, build_workflow, slugs, key, n_reps, sem, tag):
    """Per-task success probability over ``n_reps`` independent container rollouts."""

    async def one(inst):
        async with sem:  # containers are heavy — cap concurrency
            return await run_agentic_workflow(inst, build_workflow(), slugs, key)

    coros = [one(inst) for inst in instances for _ in range(n_reps)]
    results = await asyncio.gather(*coros, return_exceptions=True)

    probs, idx, failed = [], 0, 0
    for _i in range(len(instances)):
        succ = ok = 0
        for _r in range(n_reps):
            res = results[idx]
            idx += 1
            if isinstance(res, BaseException) or not res.get("valid"):
                failed += 1  # container/infra failure — skip so one bad rollout can't kill the run
                continue
            ok += 1
            succ += int(res["reward"] >= 1.0)
        probs.append(succ / ok if ok else 0.0)
    done = sum(1 for p in probs if p > 0)
    print(f"  [{tag}] solved>0 on {done}/{len(instances)} tasks"
          + (f"  ({failed} reps skipped)" if failed else ""), flush=True)
    return probs


async def run_stepzero_agentic(
    instances, slugs, key, n_reps=2, n_folds=5, null_perms=200, seed=0, concurrency=4
) -> HeadroomReport:
    sem = asyncio.Semaphore(concurrency)
    n = len(instances)

    # Per-worker direct (single-worker baselines).
    direct_p: dict[int, list[float]] = {}
    for w in range(len(slugs)):
        print(f"[single-{w} {slugs[w]}] {n} tasks × {n_reps} reps ...", flush=True)
        direct_p[w] = await _arm_probs_agentic(instances, lambda w=w: ag_direct(w), slugs, key, n_reps, sem, f"single-{w}")
    single_acc = {w: mean(p) for w, p in direct_p.items()}
    best_single = max(single_acc.values()) if single_acc else 0.0

    # Container-native scaffolds (heterogeneous by default).
    scaffold_p: dict[str, list[float]] = {}
    for name, build in AGENTIC_SCAFFOLDS.items():
        print(f"[{name}] {n} tasks × {n_reps} reps ...", flush=True)
        scaffold_p[name] = await _arm_probs_agentic(instances, build, slugs, key, n_reps, sem, name)
    scaffold_acc = {s: mean(p) for s, p in scaffold_p.items()}
    best_scaffold = max(scaffold_acc.items(), key=lambda kv: kv[1]) if scaffold_acc else ("", 0.0)

    # Reuse stepzero.py's bias-removed Δ_fixed + null-corrected oracle.
    d = _crossfit_delta_fixed(direct_p, scaffold_p, n, n_folds, seed) if (n and direct_p and scaffold_p) else [0.0]
    delta_cv = mean(d)
    se = stdev(d) / (len(d) ** 0.5) if len(d) > 1 else 0.0
    ci = (delta_cv - 1.96 * se, delta_cv + 1.96 * se)
    oracle_obs = _oracle(scaffold_p, n) if (n and scaffold_p) else 0.0
    oracle_null = _oracle_null(scaffold_p, n, null_perms, seed) if (n and scaffold_p) else 0.0

    return HeadroomReport(
        n_tasks=n, n_reps=n_reps, best_single=best_single, single_acc=single_acc,
        scaffold_acc=scaffold_acc, best_scaffold=best_scaffold, delta_fixed_cv=delta_cv,
        delta_fixed_ci=ci, oracle_obs=oracle_obs, oracle_null=oracle_null,
        oracle_signal=oracle_obs - oracle_null, total_cost_usd=0.0,
    )


def main() -> None:
    from director.agentic.runners import load_swesmith_tasks  # lazy: heavy agentic dep

    key = os.environ["OPENROUTER_API_KEY"]
    n_tasks = int(os.getenv("NTASKS", "10"))
    reps = int(os.getenv("REPS", "2"))
    conc = int(os.getenv("CONC", "4"))
    slugs = os.getenv("SLUGS", DEFAULT_SLUGS).split(",")

    tasks = load_swesmith_tasks(n_tasks)
    instances = [t["payload"] for t in tasks]
    print(f"agentic step-zero: {len(instances)} SWE-smith tasks × {reps} reps | "
          f"workers={slugs} | scaffolds={list(AGENTIC_SCAFFOLDS)} | conc={conc}", flush=True)

    report = asyncio.run(run_stepzero_agentic(instances, slugs, key, n_reps=reps, concurrency=conc))

    print(f"\n=== agentic step-zero (opencode_repo) — {report.n_tasks} tasks × {report.n_reps} reps ===", flush=True)
    print(f"best single worker : {report.best_single:.3f}", flush=True)
    for w, a in report.single_acc.items():
        print(f"    [{w}] {slugs[w]:42} {a:.3f}", flush=True)
    print("scaffolds (in-sample mean acc):", flush=True)
    for s, a in sorted(report.scaffold_acc.items(), key=lambda kv: -kv[1]):
        print(f"    {s:20} {a:.3f}", flush=True)
    print(f"cross-fitted Δ_fixed = {report.delta_fixed_cv:+.3f}  "
          f"95% CI [{report.delta_fixed_ci[0]:+.3f}, {report.delta_fixed_ci[1]:+.3f}]", flush=True)
    print(f"workflow oracle    : obs {report.oracle_obs:.3f}  null {report.oracle_null:.3f}  "
          f"signal {report.oracle_signal:+.3f}", flush=True)

    out = os.getenv("OUT")
    if out:
        with open(out, "w") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"\nwrote report -> {out}", flush=True)


if __name__ == "__main__":
    main()
