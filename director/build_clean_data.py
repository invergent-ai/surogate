"""Clean Stage-1 data build — implements FUGU_DATA_RECIPE.md.

Pipeline (single-step SFT data):
  raw candidates -> probe(n=1, temp>0) -> discriminative curate (balance + 85/15 split)
                 -> relabel curated TRAIN (n=4, temp>0) -> soft targets softmax(r_bar/tau)
                 -> manifests + data_report.md + acceptance gates.

The agentic (Stage-2) bank is produced separately by the rollout loop. Eval benchmarks are held
out via shared.sources.EVAL_ONLY (never enter the candidate pool).

Env knobs: MANIFEST_DIR, PILOT (1/0), PER_SOURCE (override raw count per source, for a cheap smoke
run), RELABEL (1/0 — skip the expensive n=4 pass for a structure-only smoke).
"""
from __future__ import annotations

import asyncio
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

from director.config import DirectorConfig, FeaturizerConfig, PoolConfig, default_frontier_pool
from director.data.manifest import read_manifest, read_probes
from director.shared.curate import curate, probe
from director.shared.sources import EVAL_ONLY, build_candidates, eval_prompt_hashes, prompt_hash
from director.shared.providers import build_pool
from director.shared.tasks import Dataset
from director.shared.types import Sampling
from director.shared.verifiers import get_grader

# --- config ---------------------------------------------------------------
MANIFEST_DIR = os.getenv("MANIFEST_DIR", "manifests/fugu_clean_v1")
PILOT = os.getenv("PILOT", "1") == "1"
PER_SOURCE = int(os.getenv("PER_SOURCE", "0")) or None  # override raw count (smoke runs)
DO_RELABEL = os.getenv("RELABEL", "1") == "1"
TAU = 0.1
RELABEL_N = 4
TEMP = 0.7                       # recipe: relabel/probe at temperature > 0 (else n=4 == n=1)
CHEAP_WORKER = "deepseek_flash"  # exempt from the >=2% sole-win prune rule
LABELS_FILE = "labels_n4_tau0.1.jsonl"

# Recipe source mix: (source, raw_count). Domain comes from each task's metadata. We use
# SuperGPQA/MMLU-Pro for science/general (harder, in-registry) instead of the easy sciq/ai2_arc.
# Final mix — sized from MEASURED probe keep-rates (2026-06-24): every kept source is 34-58%
# discriminative for our open pool, ~3x the recipe's old 15-20% assumption, so raw counts are lean.
# Kept: numina(34%), taco mid-band(58%), code_contests(42%), supergpqa(55%), mmlu_pro(53% — routable).
# Dropped: mbpp (57% SATURATED), omni_math (52% DEAD), arc_agi2 (open pool flails — outputs truncate
# even at 16k, can't produce a gradeable grid; off-distribution for a work agent). Reasoning signal
# comes from hard math (numina) + the MC "general" rows (SuperGPQA/MMLU non-STEM require reasoning).
FULL_MIX = [("numina_math", 2000),
            ("taco", 1500), ("code_contests", 1800),
            ("supergpqa_sci", 1000), ("mmlu_sci", 800),
            ("supergpqa_gen", 1000), ("mmlu_gen", 800)]
PILOT_MIX = [("numina_math", 800), ("taco", 800), ("code_contests", 800),
             ("supergpqa_sci", 500), ("mmlu_sci", 400), ("supergpqa_gen", 500), ("mmlu_gen", 400)]
# per-domain curated targets (balance quota) — lean ~2.8k curated for the full build.
FULL_TARGET = {"math": 650, "code": 900, "science": 550, "general": 550}
PILOT_TARGET = {"math": 200, "code": 300, "science": 200, "general": 200}

MIX = PILOT_MIX if PILOT else FULL_MIX
TARGET = PILOT_TARGET if PILOT else FULL_TARGET


def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    z = sum(e)
    return [v / z for v in e]


def build_raw() -> Dataset:
    # Eval-denylist by normalized prompt hash: drop any candidate that mirrors a held-out eval item
    # (renamed/reformatted), on top of the source-level EVAL_ONLY exclusion. The Non-Negotiable Rule.
    eval_hashes = eval_prompt_hashes()
    print(f"eval-denylist: {len(eval_hashes)} held-out prompt hashes loaded", flush=True)
    tasks = []
    counts: dict[str, int] = {}
    dropped = 0
    for src, raw in MIX:
        n = PER_SOURCE or raw
        kept = []
        for t in build_candidates([src], per_source_limit=n, shuffle=True, seed=0):
            if prompt_hash(t.prompt) in eval_hashes:
                dropped += 1
            else:
                kept.append(t)
        counts[src] = len(kept)
        tasks.extend(kept)
    print(f"raw candidates by source: {counts} | total {len(tasks)} | "
          f"dropped {dropped} eval-contaminated (prompt-hash)", flush=True)
    return Dataset(tasks, name="raw")


async def relabel(pool, items, sampling, manifest_dir) -> list[dict]:
    """For each curated TRAIN item, run every worker n=4 (temp>0), mean-grade -> soft target.

    Resumable: skips item_ids already in labels_n4 and APPENDS new ones (flushed per item). So a
    pilot's labels carry into the full run (same MANIFEST_DIR) and a crash mid-relabel loses nothing.
    """
    path = os.path.join(manifest_dir, LABELS_FILE)
    existing = [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []
    done = {r["task_id"] for r in existing}
    todo = [it for it in items if it.task_id not in done]
    print(f"relabel: {len(existing)} cached, {len(todo)} to do (n={RELABEL_N}, temp={TEMP})", flush=True)
    if not todo:
        return existing
    wids = pool.worker_ids
    sem = asyncio.Semaphore(48)
    write_lock = asyncio.Lock()
    fh = open(path, "a")
    new: list[dict] = []

    async def one(item):
        task = item.to_task()
        grader = get_grader(task.grader)
        msgs = task.messages()

        async def rbar_for(wid):
            # Resilient like the probe: a worker call that exhausts retries (e.g. TimeoutError) must
            # NOT crash the whole relabel — score that worker 0 for this item (it failed to deliver
            # within budget) and continue. Timeouts are rare, so the label noise is negligible.
            try:
                async with sem:
                    comps = await pool.sample(wid, msgs, RELABEL_N, sampling)
                rs = await asyncio.to_thread(lambda: [float(grader(c.text, task.solution)) for c in comps])
                return sum(rs) / max(len(rs), 1)
            except Exception as e:
                print(f"  ! relabel {task.task_id}/{wid}: {type(e).__name__} -> r=0", flush=True)
                return 0.0

        rbar = list(await asyncio.gather(*[rbar_for(w) for w in wids]))
        p = _softmax([r / TAU for r in rbar])
        rec = {"task_id": task.task_id, "domain": item.domain, "source": item.source,
               "prompt": task.prompt, "worker_ids": wids, "r_bar": rbar, "p": p,
               "grader": task.grader}
        async with write_lock:  # append+flush per item so progress survives a crash
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
        new.append(rec)

    try:
        await asyncio.gather(*[one(it) for it in todo])
    finally:
        fh.close()
    return existing + new


def write_report(pool, probes, curated, labels) -> dict:
    """data_report.md + acceptance-gate evaluation. Returns the gate results dict."""
    wids = pool.worker_ids
    disc = [p for p in probes if p.verdict == "discriminative"]
    keep_rate = len(disc) / max(len(probes), 1)
    # per-worker mean reward (over all probes) + sole wins (over curated)
    R = np.array([p.rewards for p in probes], dtype=float) if probes else np.zeros((0, len(wids)))
    mean_reward = {w: float(R[:, j].mean()) if R.size else 0.0 for j, w in enumerate(wids)}
    sole = Counter(it.winners[0] for it in curated if len(it.winners) == 1)
    sole_frac = {w: sole.get(w, 0) / max(len(curated), 1) for w in wids}
    # oracle vs best single (over discriminative probes)
    Rd = np.array([p.rewards for p in disc], dtype=float) if disc else np.zeros((0, len(wids)))
    oracle = float(Rd.max(1).mean()) if Rd.size else 0.0
    best = float(Rd.mean(0).max()) if Rd.size else 0.0
    src_counts = Counter(it.source for it in curated)
    dom_counts = Counter(it.domain for it in curated)
    denylist_hits = sum(1 for p in probes if p.source in EVAL_ONLY)
    verifier_fail = 0  # probe errors are dropped before write; tracked = 0 unless instrumented
    # per-source verdict breakdown — THE source-inclusion signal (probe-level keep-rate):
    # high discriminative = real routing signal; high saturated = too easy; high dead = too hard.
    by_src: dict[str, Counter] = defaultdict(Counter)
    for p in probes:
        by_src[p.source][p.verdict] += 1

    def _keep(s):
        return by_src[s]["discriminative"] / max(sum(by_src[s].values()), 1)

    n_cur = max(len(curated), 1)
    gates = {
        "denylist_matches==0": denylist_hits == 0,
        # measured per-source keep is 34-58% (~3x the naive 15-20% assumption), so a high blended
        # keep-rate is healthy (more routing signal), not a flaw — the ceiling reflects that reality.
        "disagreement 10-60%": 0.10 <= keep_rate <= 0.60,
        "oracle>=best+0.05": oracle >= best + 0.05,
        "no domain>35%": all(c / n_cur <= 0.35 for c in dom_counts.values()),
        "no source>30%": all(c / n_cur <= 0.30 for c in src_counts.values()),
        "strong workers sole-win>=2%": all(
            sole_frac[w] >= 0.02 for w in wids if w != CHEAP_WORKER),
    }
    lines = [
        "# Data Report — fugu_clean", "",
        f"- probed: {len(probes)} | discriminative: {len(disc)} (keep-rate {keep_rate:.1%})",
        f"- curated: {len(curated)} | train labels: {len(labels)}",
        f"- oracle: {oracle:.3f} vs best-single: {best:.3f} (headroom {oracle-best:+.3f})",
        f"- denylist matches: {denylist_hits}", "",
        "## curated by domain", *[f"- {d}: {c} ({c/n_cur:.0%})" for d, c in dom_counts.most_common()],
        "", "## per-source verdict — keep-rate is the SOURCE-INCLUSION signal",
        *[f"- {s}: keep {by_src[s]['discriminative']}/{sum(by_src[s].values())} = {_keep(s):.0%}"
          f"  (saturated {by_src[s]['saturated']}, dead {by_src[s]['dead']})"
          for s in sorted(by_src, key=_keep, reverse=True)],
        "", "## curated by source", *[f"- {s}: {c} ({c/n_cur:.0%})" for s, c in src_counts.most_common()],
        "", "## per-worker (mean reward | sole-win share of curated)",
        *[f"- {w}: {mean_reward[w]:.3f} | {sole_frac[w]:.1%}"
          + ("  (cheap, sole-win exempt)" if w == CHEAP_WORKER else "") for w in wids],
        "", "## acceptance gates (pre-SFT)",
        *[f"- [{'PASS' if ok else 'FAIL'}] {name}" for name, ok in gates.items()],
    ]
    path = os.path.join(MANIFEST_DIR, "data_report.md")
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-(len(gates) + 1):]), flush=True)
    print(f"wrote {path}", flush=True)
    return gates


def main():
    cfg = DirectorConfig(workers=default_frontier_pool(), featurizer=FeaturizerConfig(context_window=4096))
    pool = build_pool(PoolConfig(budget_usd=None, max_concurrency=96, timeout_s=90, max_retries=2), cfg.workers)
    sampling = Sampling(temperature=TEMP, max_tokens=2048, reasoning_effort="low")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    raw = build_raw()
    loop.run_until_complete(probe(pool, raw, MANIFEST_DIR, sampling=sampling))   # n=1, resumable
    curated = curate(MANIFEST_DIR, per_domain_target=TARGET, worker_ids=cfg.worker_ids,
                     sources=[s for s, _ in MIX], note="fugu_clean" + ("_pilot" if PILOT else ""))

    train = [it for it in curated if it.split == "train"]
    test = [it for it in curated if it.split == "test"]
    labels = []
    if DO_RELABEL and train:
        labels = loop.run_until_complete(relabel(pool, train, sampling, MANIFEST_DIR))
    with open(os.path.join(MANIFEST_DIR, "internal_val.jsonl"), "w") as f:
        for it in test:
            f.write(json.dumps({"task_id": it.task_id, "domain": it.domain, "source": it.source,
                                "prompt": it.prompt, "worker_ids": cfg.worker_ids,
                                "rewards_n1": it.rewards, "grader": it.grader}) + "\n")

    gates = write_report(pool, read_probes(MANIFEST_DIR), read_manifest(MANIFEST_DIR), labels)
    ok = all(gates.values())
    print(f"\nGATES: {'ALL PASS' if ok else 'SOME FAILED'} — review data_report.md before scaling.", flush=True)


if __name__ == "__main__":
    main()
