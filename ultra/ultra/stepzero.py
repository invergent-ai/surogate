"""Step-zero workflow-headroom test (ultra-intro §3) — the go/no-go gate before GRPO.

WHY THIS IS NOT THE FUGU MISTAKE AGAIN
--------------------------------------
Fugu's single-step result showed the *selection* oracle (best WORKER per task) sat below
the noise floor — the pool has no routing complementarity. Step-zero measures a DIFFERENT
axis: the *workflow* oracle (best SCAFFOLD per task), whose headroom comes from
orchestration (decompose / critique→revise / debate→synthesize), not from picking which
model answers. A correlated, one-dominant pool can still have workflow headroom.

But the n=1 statistical trap is identical, so this runner guards against it the way the
Fugu post-mortem taught:
  * ``n_reps`` draws per (arm, task) with DISTINCT seeds (distinct cache keys → distinct
    samples) → estimate per-task success *probabilities*, not single coin-flips.
  * ``delta_fixed`` is CROSS-FITTED: pick the best scaffold on dev folds, score it on the
    held-out fold. This removes the in-sample max-selection (winner's-curse) bias that a
    naive ``max_s mean_s − max_w mean_w`` carries (~1-2 pts at N=100 — a real fraction of
    the ~3 pt gate). Reported with a paired 95% CI.
  * the workflow oracle is compared to a PERMUTATION NULL (shuffle the task↔scaffold
    alignment). ``oracle_signal = observed − null`` is the *real* per-task headroom; if it
    doesn't clear the null, the oracle is noise + marginals (the Fugu verdict).

Read ``delta_fixed_cv`` (cross-fitted, CI) as the primary verdict; ``oracle_obs`` raw is an
inflated upper bound until null-corrected.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, replace
from statistics import mean, stdev

from .executor import execute_workflow
from .scaffolds import SCAFFOLDS, direct
from .schemas import TaskSpec
from .workers import Sampling, WorkerPool


@dataclass
class HeadroomReport:
    n_tasks: int
    n_reps: int
    best_single: float
    single_acc: dict[int, float]
    scaffold_acc: dict[str, float]  # in-sample means (display only)
    best_scaffold: tuple[str, float]
    delta_fixed_cv: float  # cross-fitted best-scaffold − best-single (bias removed)
    delta_fixed_ci: tuple[float, float]  # paired 95% CI
    oracle_obs: float
    oracle_null: float  # permutation null (no task↔scaffold interaction)
    oracle_signal: float  # oracle_obs − oracle_null (real per-task headroom)
    total_cost_usd: float = 0.0


def _with_seed(s: Sampling, k: int) -> Sampling:
    return replace(s, seed=(s.seed or 0) + k)


async def _arm_probs(tasks, build_workflow, pool, sampling, n_reps, tag):
    """Per-task success probability over ``n_reps`` distinct-seed draws (+ total cost)."""
    coros = [
        execute_workflow(task, build_workflow(), pool, _with_seed(sampling, r), f"{tag}-{k}-{r}")
        for k, task in enumerate(tasks)
        for r in range(n_reps)
    ]
    recs = await asyncio.gather(*coros, return_exceptions=True)
    probs: list[float] = []
    cost = 0.0
    idx = 0
    failed = 0
    for _k in range(len(tasks)):
        succ = ok = 0
        for _r in range(n_reps):
            rec = recs[idx]
            idx += 1
            if isinstance(rec, BaseException):
                failed += 1  # infra/API failure on this rep — skip so one bad call can't kill the run
                continue
            ok += 1
            succ += int(bool(rec.grade and rec.grade.success))
            cost += sum(st.cost_usd for st in rec.execution.steps)
        probs.append(succ / ok if ok else 0.0)
    if failed:
        print(f"  [{tag}] skipped {failed} failed reps (infra/API)", flush=True)
    return probs, cost


def _mean_at(v: list[float], idx: list[int]) -> float:
    return mean(v[i] for i in idx) if idx else 0.0


def _kfold(n: int, k: int, seed: int) -> list[list[int]]:
    order = list(range(n))
    random.Random(seed).shuffle(order)
    k = max(1, min(k, n))
    return [order[f::k] for f in range(k)]  # k disjoint folds covering every task


def _crossfit_delta_fixed(direct_p, scaffold_p, n_tasks, n_folds, seed) -> list[float]:
    """Per-task paired diff d_i = p[best-scaffold-on-dev] − p[best-worker-on-dev], scored
    only on the held-out fold (removes in-sample max-selection bias)."""
    d = [0.0] * n_tasks
    for fold in _kfold(n_tasks, n_folds, seed):
        held = set(fold)
        train = [i for i in range(n_tasks) if i not in held]
        s_star = max(scaffold_p, key=lambda s: _mean_at(scaffold_p[s], train))
        w_star = max(direct_p, key=lambda w: _mean_at(direct_p[w], train))
        for i in fold:
            d[i] = scaffold_p[s_star][i] - direct_p[w_star][i]
    return d


def _oracle(scaffold_p, n_tasks) -> float:
    return mean(max(scaffold_p[s][i] for s in scaffold_p) for i in range(n_tasks))


def _oracle_null(scaffold_p, n_tasks, perms, seed) -> float:
    """Oracle under no task↔scaffold interaction: independently permute each scaffold's
    per-task vector (keeps marginals, destroys alignment), average the resulting oracle."""
    rnd = random.Random(seed)
    out = []
    for _ in range(perms):
        cols = {s: rnd.sample(v, len(v)) for s, v in scaffold_p.items()}
        out.append(mean(max(cols[s][i] for s in cols) for i in range(n_tasks)))
    return mean(out) if out else 0.0


async def run_stepzero(
    tasks: list[TaskSpec],
    pool: WorkerPool,
    sampling: Sampling | None = None,
    scaffolds: dict | None = None,
    worker_assignment: dict[str, tuple] | None = None,
    n_reps: int = 3,
    n_folds: int = 5,
    null_perms: int = 200,
    seed: int = 0,
) -> HeadroomReport:
    sampling = sampling or Sampling()
    scaffolds = scaffolds or SCAFFOLDS
    worker_assignment = worker_assignment or {}
    n = len(tasks)
    total_cost = 0.0

    # Per-worker direct (scaffold A) probabilities → best single worker.
    direct_p: dict[int, list[float]] = {}
    for w in range(len(pool.worker_ids)):
        p, c = await _arm_probs(tasks, lambda w=w: direct(w), pool, sampling, n_reps, f"single-{w}")
        direct_p[w] = p
        total_cost += c
    single_acc = {w: mean(p) for w, p in direct_p.items()} if direct_p else {}
    best_single = max(single_acc.values()) if single_acc else 0.0

    # Each scaffold under its (optional) worker assignment.
    scaffold_p: dict[str, list[float]] = {}
    for name, build in scaffolds.items():
        args = worker_assignment.get(name, ())
        p, c = await _arm_probs(tasks, lambda b=build, a=args: b(*a), pool, sampling, n_reps, name)
        scaffold_p[name] = p
        total_cost += c
    scaffold_acc = {s: mean(p) for s, p in scaffold_p.items()}
    best_scaffold = max(scaffold_acc.items(), key=lambda kv: kv[1]) if scaffold_acc else ("", 0.0)

    # Cross-fitted Δ_fixed + paired CI (the primary, bias-removed verdict).
    if n and direct_p and scaffold_p:
        d = _crossfit_delta_fixed(direct_p, scaffold_p, n, n_folds, seed)
    else:
        d = [0.0]
    delta_cv = mean(d)
    se = stdev(d) / (len(d) ** 0.5) if len(d) > 1 else 0.0
    ci = (delta_cv - 1.96 * se, delta_cv + 1.96 * se)

    # Null-corrected workflow oracle.
    oracle_obs = _oracle(scaffold_p, n) if n and scaffold_p else 0.0
    oracle_null = _oracle_null(scaffold_p, n, null_perms, seed) if n and scaffold_p else 0.0

    return HeadroomReport(
        n_tasks=n,
        n_reps=n_reps,
        best_single=best_single,
        single_acc=single_acc,
        scaffold_acc=scaffold_acc,
        best_scaffold=best_scaffold,
        delta_fixed_cv=delta_cv,
        delta_fixed_ci=ci,
        oracle_obs=oracle_obs,
        oracle_null=oracle_null,
        oracle_signal=oracle_obs - oracle_null,
        total_cost_usd=total_cost,
    )
