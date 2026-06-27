"""Curation pipeline: turn a large raw candidate pool into a balanced, split,
disagreement-rich training manifest.

    probe (n=1, cheap) -> classify -> balance -> split -> manifest

The probe pass is the cost filter: sample each worker once per candidate, grade, and
keep only items where workers *disagree* (the routing signal). Probing is concurrent,
budget-aware, and fully resumable (records append to probe.jsonl; re-running skips
already-probed ids and reuses the completion cache).
"""

from __future__ import annotations

import asyncio
import random
from collections import defaultdict
from itertools import combinations

import numpy as np

from ..data.manifest import (
    CuratedItem,
    ManifestMeta,
    ProbeRecord,
    append_probe,
    classify,
    manifest_stats,
    probe_stats,
    probed_ids,
    read_probes,
    write_manifest,
    write_meta,
)
from ..shared.budget import BudgetExceeded
from .tasks import Dataset, Task
from .types import Sampling
from .verifiers import get_grader

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


async def _probe_one(pool, task: Task, sampling: Sampling) -> ProbeRecord:
    grader = get_grader(task.grader)
    worker_ids = pool.worker_ids
    msgs = task.messages()

    async def one(wid: str) -> float:
        # Resilient per-worker: a call that errors/times out (bounded by the pool's wait_for) scores
        # 0 instead of killing the whole task's gather -> the probe never hangs on a stuck worker.
        # Grade off the event loop (sympy/code-exec can be slow) so one task can't stall the loop.
        try:
            comp = await pool.call(wid, msgs, sampling)
        except BudgetExceeded:
            raise
        except Exception:
            return 0.0
        try:
            return float(await asyncio.to_thread(grader, comp.text, task.solution))
        except Exception:
            return 0.0

    rewards = list(await asyncio.gather(*[one(w) for w in worker_ids]))
    verdict, winners = classify(rewards, worker_ids)
    md = task.metadata
    return ProbeRecord(
        task_id=task.task_id,
        domain=md.get("domain", "unknown"),
        source=md.get("source", "unknown"),
        prompt=task.prompt,
        solution=task.solution,
        grader=task.grader,
        system=task.system,
        rewards=rewards,
        winners=winners,
        verdict=verdict,
        subdomain=md.get("subdomain"),
        dataset=md.get("dataset"),
    )


async def probe(
    pool,
    candidates: Dataset,
    manifest_dir: str,
    *,
    sampling: Sampling | None = None,
    max_in_flight: int = 64,
) -> list[ProbeRecord]:
    """Probe every not-yet-probed candidate once per worker; append records.

    Resumable: skips ids already present in probe.jsonl. Stops cleanly on
    BudgetExceeded with all completed records persisted.
    """
    sampling = sampling or Sampling()
    done = probed_ids(manifest_dir)
    todo = [t for t in candidates if t.task_id not in done]
    if not todo:
        return read_probes(manifest_dir)

    sem = asyncio.Semaphore(max_in_flight)
    bar = tqdm(total=len(todo), desc="probing") if tqdm is not None else None
    lock = asyncio.Lock()
    stopped = {"budget": False}

    async def run(task: Task) -> None:
        if stopped["budget"]:
            return
        async with sem:
            try:
                rec = await _probe_one(pool, task, sampling)
            except BudgetExceeded:
                stopped["budget"] = True
                return
        async with lock:  # serialize appends
            append_probe(manifest_dir, rec)
        if bar is not None:
            bar.update(1)

    await asyncio.gather(*[run(t) for t in todo])
    if bar is not None:
        bar.close()
    if stopped["budget"]:
        print("[curate] budget cap hit during probe; partial probe saved (resumable).")
    return read_probes(manifest_dir)


# ---------------------------------------------------------------------------
# dataset-agent subset selection by Relative Error Reduction (Trinity A.6)
# ---------------------------------------------------------------------------


def accuracy_matrix(probes: list[ProbeRecord], worker_ids: list[str]) -> tuple[list[str], np.ndarray]:
    """E[d, j] = mean reward (accuracy) of worker j on dataset d — the E(D,M) matrix."""
    by_ds: dict[str, list[list[float]]] = defaultdict(list)
    for p in probes:
        by_ds[p.dataset or p.domain].append(p.rewards)
    datasets = sorted(by_ds)
    E = np.array([np.mean(by_ds[d], axis=0) for d in datasets], dtype=float)
    return datasets, E


def rer(E: np.ndarray) -> float:
    """Relative Error Reduction over a (D, W) accuracy matrix (Trinity A.6, error space):
    (oracle − best_single) / (1 − best_single). oracle = mean per-dataset best-in-pool;
    best_single = best single agent averaged over datasets. Rewards complementarity."""
    if E.size == 0:
        return 0.0
    oracle = float(E.max(axis=1).mean())
    best = float(E.mean(axis=0).max())
    return 0.0 if best >= 1.0 else (oracle - best) / (1.0 - best)


def select_agents(
    probes: list[ProbeRecord], worker_ids: list[str], *, max_agents: int | None = None, top_pct: float = 0.05
) -> tuple[list[str], float]:
    """Pick the agent subset that maximizes RER across all datasets (Trinity A.6).

    Filters to agents on the top-(top_pct) accuracy frontier, then exhaustively enumerates
    subsets, maximizing RER and breaking ties toward FEWER agents (parsimony — a worker
    that doesn't raise RER is dominated and dropped, as in the paper's 3-of-7 result)."""
    datasets, E = accuracy_matrix(probes, worker_ids)
    thr = float(np.quantile(E, 1.0 - top_pct))
    frontier = sorted({j for d in range(E.shape[0]) for j in range(E.shape[1]) if E[d, j] >= thr})
    if not frontier:
        frontier = list(range(E.shape[1]))
    cap = min(max_agents or len(frontier), len(frontier))
    best_key, best_agents, best_rer = None, [worker_ids[frontier[0]]], 0.0
    for k in range(1, cap + 1):
        for agents in combinations(frontier, k):
            r = rer(E[:, list(agents)])
            key = (round(r, 6), -len(agents))  # max RER, then fewest agents
            if best_key is None or key > best_key:
                best_key, best_agents, best_rer = key, [worker_ids[j] for j in agents], r
    return best_agents, best_rer


# ---------------------------------------------------------------------------
# balance + split
# ---------------------------------------------------------------------------


def balance(
    discriminative: list[ProbeRecord],
    per_domain_target: dict[str, int] | int,
    *,
    seed: int = 0,
) -> list[ProbeRecord]:
    """Select up to a per-domain quota, flattening the sole-winner distribution so the
    router can't collapse to 'always pick the globally-strongest worker'.

    Sole-winner items are drawn round-robin across winners; multi-winner items
    (ambiguous, unbiased) backfill any remaining quota.
    """
    rng = random.Random(seed)
    by_domain: dict[str, list[ProbeRecord]] = defaultdict(list)
    for r in discriminative:
        by_domain[r.domain].append(r)

    selected: list[ProbeRecord] = []
    for domain, items in by_domain.items():
        target = per_domain_target if isinstance(per_domain_target, int) else per_domain_target.get(domain, 0)
        if target <= 0:
            continue
        sole: dict[str, list[ProbeRecord]] = defaultdict(list)
        multi: list[ProbeRecord] = []
        for it in items:
            (sole[it.winners[0]] if len(it.winners) == 1 else multi).append(it)
        for lst in sole.values():
            rng.shuffle(lst)
        rng.shuffle(multi)

        chosen: list[ProbeRecord] = []
        winners = list(sole)
        rng.shuffle(winners)
        cursor = {w: 0 for w in winners}
        progressing = True
        while len(chosen) < target and progressing:
            progressing = False
            for w in winners:
                if cursor[w] < len(sole[w]):
                    chosen.append(sole[w][cursor[w]])
                    cursor[w] += 1
                    progressing = True
                    if len(chosen) >= target:
                        break
        mi = 0
        while len(chosen) < target and mi < len(multi):
            chosen.append(multi[mi])
            mi += 1
        selected.extend(chosen)
    return selected


def split(
    items: list[ProbeRecord], *, train_ratio: float = 0.85, seed: int = 0
) -> list[CuratedItem]:
    """Per-domain stratified train/held-out split."""
    rng = random.Random(seed)
    by_domain: dict[str, list[ProbeRecord]] = defaultdict(list)
    for r in items:
        by_domain[r.domain].append(r)
    out: list[CuratedItem] = []
    for _domain, group in by_domain.items():
        rng.shuffle(group)
        cut = int(round(len(group) * train_ratio))
        for i, r in enumerate(group):
            out.append(CuratedItem(**vars(r), split=("train" if i < cut else "test")))
    return out


def curate(
    manifest_dir: str,
    *,
    per_domain_target: dict[str, int] | int,
    worker_ids: list[str],
    sources: list[str],
    train_ratio: float = 0.85,
    seed: int = 0,
    note: str = "",
) -> list[CuratedItem]:
    """Build the manifest from existing probe records: classify (already stored) ->
    balance -> split -> write manifest + meta. Returns the curated items."""
    probes = read_probes(manifest_dir)
    print(probe_stats(probes))
    disc = [p for p in probes if p.verdict == "discriminative"]
    balanced = balance(disc, per_domain_target, seed=seed)
    items = split(balanced, train_ratio=train_ratio, seed=seed)
    write_manifest(manifest_dir, items)
    write_meta(
        manifest_dir,
        ManifestMeta(worker_ids=worker_ids, sources=sources, seed=seed, note=note),
    )
    print(manifest_stats(items))
    return items


def train_dataset(items: list[CuratedItem]) -> Dataset:
    return Dataset([it.to_task() for it in items if it.split == "train"], name="curated-train")


def test_items(items: list[CuratedItem]) -> list[CuratedItem]:
    return [it for it in items if it.split == "test"]
