"""Fugu training-framework loop: incremental, multi-domain, resumable.

Each STEP sweeps ALL domains (one batch each) and banks per-item worker reward vectors.
Every EVAL_EVERY steps it (re)trains the router on the whole bank, evaluates held-out
routing lift (per-domain + overall), checkpoints the router, and logs the metric — so you
watch routing improve as data grows, and can stop/resume anytime.

Domains:
  single-step (code/math/reasoning/general): each worker sampled n=4, graded -> reward vector.
  agentic (SWE-Bench): each worker solo via mini-swe-agent -> resolved {0,1}.
Both reduce to (prompt -> reward-vector-over-workers), so one router trains across all domains.

Bank: production_bank.jsonl (resumable). Metrics: fugu_metrics.jsonl. Router: fugu_router.pt.
Caps: pool budget (single-step) + MSWEA_GLOBAL_COST_LIMIT (agentic).
Env knobs: STEPS, EVAL_EVERY, SINGLE_BATCH, AGENTIC_BATCH.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch

from director.agentic.runners import (
    load_swe_pro_tasks, load_swe_tasks, load_swesmith_tasks, load_tau_tasks, load_terminal_tasks,
    run_swe, run_swe_pro, run_swesmith, run_tau, run_terminal,
)
from director.config import DirectorConfig, FeaturizerConfig, PoolConfig, default_frontier_pool
from director.fugu.inference import select_worker
from director.fugu.labels import SoftLabel
from director.fugu.model import save_router
from director.fugu.run import _sampling, build_router
from director.fugu.sft import train_sft
from director.shared.providers import build_pool
from director.shared.sources import build_candidates
from director.shared.transcript import raw_query
from director.shared.types import Sampling
from director.shared.verifiers import get_grader

STEPS = int(os.getenv("STEPS", "6"))
EVAL_EVERY = int(os.getenv("EVAL_EVERY", "2"))
SINGLE_BATCH = int(os.getenv("SINGLE_BATCH", "10"))
AGENTIC_BATCH = int(os.getenv("AGENTIC_BATCH", "4"))
AGENTIC_PARALLEL = int(os.getenv("AGENTIC_PARALLEL", "24"))  # GLOBAL cap on concurrent agentic rollouts
SWE_DATASET = "princeton-nlp/SWE-bench_Verified"  # HELD-OUT eval only (never trained on)
# Production Stage-1 data — matches Fugu §3.1.2 (coding, mathematics, reasoning, language
# understanding, agentic), all VERIFIABLE + executable, with Fugu's reported benchmarks HELD OUT
# for honest eval (see EVAL_ONLY in shared/sources.py). Single-step `src` is a LIST: each domain is
# fed by several LARGE non-benchmark sources for scale + difficulty range — the routing signal lives
# in mid-difficulty tasks where workers diverge, so we want breadth, not a tiny saturated eval set.
DOMAINS = [
    # --- single-step (Stage-1 SFT) ---
    ("code",      "single", ["taco", "code_contests", "mbpp"]),  # hard stdin/stdout + breadth; HumanEval held out
    ("math",      "single", ["numina_math", "omni_math"]),       # ~1M range + olympiad; MATH-500/AIME held out
    ("science",   "single", ["mmlu_sci", "supergpqa_sci"]),      # GPQA-Diamond held out
    ("general",   "single", ["mmlu_gen", "supergpqa_gen"]),      # language understanding (Fugu §3.1.2)
    ("reasoning", "single", ["arc_agi2"]),                       # abstract reasoning
    # --- agentic: TRAINING sources only (Fugu's eval benchmarks held out) ---
    ("swesmith",    "agentic", "swesmith"),     # SWE-smith: agentic-coding TRAINING (replaces SWE-Verified)
    ("tau_retail",  "agentic", "tau_retail"),   # tau-bench (Fugu evals tau-3 BANKING, which we hold out)
    ("tau_airline", "agentic", "tau_airline"),
    # HELD-OUT EVAL (never trained): SWE-bench Verified/Pro, Terminal-Bench, tau-3 banking, GPQA,
    # HumanEval, MATH-500, AIME (+ LiveCodeBench/SciCode/HLE once loaders exist).
]
BANK = "production_bank.jsonl"
METRICS = "fugu_metrics.jsonl"
ROUTER_CKPT = "fugu_router.pt"
TAU = 0.1
W_COST = float(os.getenv("W_COST", "0.3"))  # cost-performance weight in the soft target

cfg = DirectorConfig(workers=default_frontier_pool(), featurizer=FeaturizerConfig(context_window=4096))
wids = cfg.worker_ids
slugs = {w.worker_id: w.model for w in cfg.workers}
pool = build_pool(PoolConfig(budget_usd=None, max_concurrency=48, timeout_s=90, max_retries=2), cfg.workers)
WORKER_TIMEOUT = 240  # per-worker cap for single-step. Some workers (e.g. mimo) ignore the low-reasoning
# hint and ramble to the token cap, taking >120s; give them room to finish (fewer spurious timeout-zeros).
# On timeout we bank reward=0 (cell completes, no perpetual re-run).
gen_router = build_router(cfg)  # used only to force solo workers in agentic gen (allowed=)
samp = _sampling(cfg)          # max reasoning, 32k (agentic path uses this / its own high setting)
# Single-step labels are only the SFT WARM-START (Stage 1) — dense, cheap signal to teach the
# head+SVF task-type features. They are NOT where routing lift comes from (that's the agentic
# stage), so we generate them at LOW reasoning with a tight token cap: ~5-10x faster/cheaper,
# and it bounds the max-reasoning tail latency that made single-step crawl.
samp_single = replace(samp, reasoning_effort="low", max_tokens=2048)  # warm-start answers are short;
# a tight cap also bounds workers (e.g. glm-5.2) that ignore the low-effort hint and ramble to the cap
_lock = threading.Lock()

# Agentic generation context + per-source runners. Generation always forces a SOLO worker
# (allowed={w}), so the router selection is overridden -> the (untrained) gen_router is fine.
# pool_tracked=True means the runner's cost is already in pool.budget (don't double-count).
ctx = SimpleNamespace(
    router=gen_router, pool=pool, sampling=samp, worker_slugs=slugs,
    tau_user_model="openrouter/openai/gpt-5-mini",   # tau-bench user simulator (via OpenRouter)
    config_path=None, ckpt_path=None,                # terminal: solo-forced, so router is irrelevant
    terminal_dataset="terminal-bench-core", terminal_version="0.1.1", work_dir="./.director_tb",
    terminal_dataset_path=None,
)
# Pre-resolve the terminal-bench dataset to its LOCAL path once (single-threaded, downloads if
# needed). Parallel Harnesses then use this read-only local path instead of each re-resolving the
# registry — which under concurrency races the cache and produces malformed ".." compose paths.
try:
    from terminal_bench.dataset.dataset import Dataset as _TBDataset
    ctx.terminal_dataset_path = str(_TBDataset(name=ctx.terminal_dataset, version=ctx.terminal_version)._path.absolute())
    print(f"terminal dataset local path: {ctx.terminal_dataset_path}", flush=True)
    # terminal-bench builds a malformed runtime compose path: CACHE/terminal-bench-core/../<task>,
    # which resolves to CACHE/<task> — a dir that doesn't exist -> `docker compose up` fails (flood).
    # We can't change its internal path computation, so make that resolved path VALID: symlink
    # CACHE/<task> -> the real .../0.1.1/<task> for every task. Then its buggy `..` path resolves.
    _tb_cache = os.path.dirname(os.path.dirname(ctx.terminal_dataset_path))  # = ~/.cache/terminal-bench
    _linked = 0
    for _t in os.listdir(ctx.terminal_dataset_path):
        _src = os.path.join(ctx.terminal_dataset_path, _t)
        _lnk = os.path.join(_tb_cache, _t)
        if os.path.isdir(_src) and not os.path.lexists(_lnk):
            try:
                os.symlink(_src, _lnk)
                _linked += 1
            except OSError:
                pass
    print(f"terminal: symlinked {_linked} task dirs into {_tb_cache} "
          f"(workaround for terminal-bench '..' compose-path bug)", flush=True)
except Exception as _e:
    print(f"terminal dataset path resolve failed ({type(_e).__name__}: {_e}) — falling back to registry", flush=True)
AGENTIC_RUNNERS = {            # source -> (runner, cost_already_in_pool_budget)
    "swebench": (run_swe, False),
    "swesmith": (run_swesmith, False),       # SWE-smith: agentic-coding TRAINING corpus (Python)
    "swebench_pro": (run_swe_pro, True),
    "terminal": (run_terminal, False),
    "tau_retail": (run_tau, True),
    "tau_airline": (run_tau, True),
}


def _start_network_janitor(interval: float = 45.0) -> None:
    """Each terminal trial spins up a Docker compose network; a trial that fails mid-setup LEAKS
    it, and Docker caps user networks at ~31 (default address pool) -> once leaked networks fill
    the pool, every `compose up` fails to create a network and ALL terminal trials cascade-fail.
    `docker network prune` removes ONLY unused networks (active trials' networks are untouched), so
    running it periodically reclaims leaks continuously and keeps the pool from ever filling."""
    import subprocess as _sp

    def _loop():
        while True:
            time.sleep(interval)
            try:
                _sp.run(["docker", "network", "prune", "-f"], capture_output=True, timeout=30)
            except Exception:
                pass

    threading.Thread(target=_loop, daemon=True, name="net-janitor").start()
    print(f"network janitor started (prune unused every {interval:.0f}s)", flush=True)


_start_network_janitor()


def load_bank() -> list[dict]:
    out = []
    if os.path.exists(BANK):
        for line in open(BANK):
            if line.strip():
                out.append(json.loads(line))
    return out


def append_bank(rec: dict) -> None:
    with _lock:
        with open(BANK, "a") as f:
            f.write(json.dumps(rec) + "\n")


def migrate_bank() -> None:
    """Migrate legacy per-item records ({rewards:{w:r}, costs:{w:c}}) to per-(item,worker) CELL
    records ({worker, reward, cost}). Idempotent: cell records already carry 'worker'. Run once
    at startup so the resumable, per-cell loop reads a uniform format without losing banked data."""
    if not os.path.exists(BANK):
        return
    recs = load_bank()
    if not recs or all("worker" in r for r in recs):
        return  # already in cell format (or empty)
    cells = []
    for r in recs:
        if "worker" in r:
            cells.append(r)
            continue
        for w, reward in r.get("rewards", {}).items():
            cells.append({"domain": r["domain"], "kind": r.get("kind", "single"),
                          "item_id": r["item_id"], "prompt": r.get("prompt", ""), "worker": w,
                          "reward": reward, "cost": r.get("costs", {}).get(w, 0.0)})
    with open(BANK, "w") as f:
        for c in cells:
            f.write(json.dumps(c) + "\n")
    print(f"migrated bank: {len(recs)} per-item -> {len(cells)} per-(item,worker) cells", flush=True)


def softmax(xs):
    m = max(xs)
    e = [math.exp((x - m) / TAU) for x in xs]
    z = sum(e)
    return [v / z for v in e]


# ---- generation -----------------------------------------------------------
_agentic_cost = [0.0]


def _spent() -> float:
    return pool.budget.spent_usd + _agentic_cost[0]


async def gen_single(step: int, name: str, items, seen_cells: set) -> int:
    """Per-(item,worker) resumable generation: produce ONLY the missing cells. Each cell is
    banked independently the moment it lands, so a slow/hung worker (bounded by WORKER_TIMEOUT)
    just leaves a gap that gets retried on a later step — no whole-item loss, no re-running the
    cells that already succeeded, never blocks or crashes the loop."""
    jobs = [(it, w) for it in items for w in wids if (name, it.task_id, w) not in seen_cells]
    if not jobs:
        return 0
    total = len(jobs)
    done = [0]
    fail = [0]
    t0 = time.time()

    async def cell(it, w):
        grader = get_grader(it.grader)
        msgs = it.messages()
        try:
            # NB: do NOT wrap in asyncio.wait_for here — pool.sample includes waiting for a
            # pool-gate slot, so a short timeout would fire on QUEUE wait (not the call) and
            # mass-fail cells under high concurrency. The provider already bounds each call
            # (call_timeout + client timeout + retries).
            comps = await pool.sample(w, msgs, 4, samp_single)
            rs = await asyncio.to_thread(  # grade off the event loop (sympy/subprocess can be slow)
                lambda: [float(grader(c.text, it.solution)) for c in comps])
            reward = sum(rs) / len(rs)
            cost = sum(c.cost_usd for c in comps) / max(len(comps), 1)
            append_bank({"domain": name, "kind": "single", "item_id": it.task_id,
                         "prompt": it.prompt, "worker": w, "reward": reward, "cost": cost})
            seen_cells.add((name, it.task_id, w))
            done[0] += 1
            print(f"      [s{step}/{name}] cell {done[0]}/{total} {it.task_id[:16]}/{w} ok "
                  f"| spent ${_spent():.4f} | {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            # errored/timed out -> NOT a valid grade. Do NOT record it; leave the cell missing so
            # it re-runs on a later step. A fake reward=0 here is indistinguishable from a genuine
            # graded 0 and would pollute the routing signal.
            print(f"      [s{step}/{name}] cell {it.task_id[:16]}/{w} SKIPPED ({type(e).__name__}) "
                  f"after {time.time()-t0:.0f}s — errored/timed out, not recorded", flush=True)
            fail[0] += 1

    await asyncio.gather(*[cell(it, w) for it, w in jobs])
    return total


async def gen_agentic(step: int, name: str, source: str, tasks, seen_cells: set, sem) -> int:
    """Per-(task,worker) resumable agentic rollouts across ANY agentic source (swe / swe_pro /
    terminal / tau). Runs ONLY the missing cells, each via the source's runner with a forced solo
    worker, and banks each cell the moment it's graded (interrupt-safe). A rollout that errors
    leaves its cell missing (retried next step). Concurrency is bounded by a semaphore; blocking
    harness/grading ops run off the event loop inside each runner."""
    runner, pool_tracked = AGENTIC_RUNNERS[source]
    by_id = {t["item_id"]: t for t in tasks}
    jobs = [(tid, w) for tid in by_id for w in wids if (name, tid, w) not in seen_cells]
    if not jobs:
        return 0
    total = len(jobs)
    done = [0]
    fail = [0]
    t0 = time.time()

    async def cell(tid, w):
        async with sem:
            try:
                reward, cost = await runner(ctx, by_id[tid]["payload"], {w})
            except Exception as e:
                # errored/timed out (rollout error, harness failure, terminal test/agent timeout)
                # -> NOT a valid grade. Do NOT record it; leave the cell missing so it re-runs on a
                # later step. A fake 0 is indistinguishable from a genuine graded 0.
                print(f"    [s{step}/{name}] agentic cell {tid[:22]}/{w} SKIPPED "
                      f"({type(e).__name__}: {str(e)[:80]}) — errored/timed out, not recorded", flush=True)
                fail[0] += 1
                return
        if not pool_tracked:
            _agentic_cost[0] += cost
        append_bank({"domain": name, "kind": "agentic", "item_id": tid,
                     "prompt": by_id[tid]["prompt"], "worker": w, "reward": reward, "cost": cost})
        seen_cells.add((name, tid, w))
        done[0] += 1
        print(f"    [s{step}/{name}] agentic cell {done[0]}/{total} ({w} {tid[:22]}) reward={reward:.0f} "
              f"| spent ${_spent():.4f} | {time.time()-t0:.0f}s", flush=True)

    await asyncio.gather(*[cell(tid, w) for tid, w in jobs])
    return total


# ---- train + eval ---------------------------------------------------------
def _soft_cost(rvec, cvec):
    """Pure-reward soft target (faithful Fugu Stage-1): softmax(r̄/τ). Cost is intentionally NOT
    used here — the cost-tiebreak was biasing routing toward cheap-but-failing workers and hurting
    resolve. Cost shaping is deferred to Stage-2 (sep-CMA-ES on terminal reward). ``cvec`` is kept
    only for call-site signature compatibility."""
    return softmax(rvec)


def train_and_eval(step: int) -> None:
    # reconstruct (item -> reward/cost vectors) from per-(item,worker) cells; keep COMPLETE rows
    items: dict = {}
    for c in load_bank():
        if "worker" not in c:
            continue
        d = items.setdefault((c["domain"], c["item_id"]),
                             {"prompt": c.get("prompt", ""), "rewards": {}, "costs": {}})
        d["rewards"][c["worker"]] = c["reward"]
        d["costs"][c["worker"]] = c["cost"]
    recs = [{"domain": k[0], "item_id": k[1], "prompt": v["prompt"],
             "rewards": v["rewards"], "costs": v["costs"]}
            for k, v in items.items() if all(w in v["rewards"] for w in wids)]
    by_dom = defaultdict(list)
    for r in recs:
        by_dom[r["domain"]].append(r)
    train, test = [], []
    for dom, items in by_dom.items():
        items = sorted(items, key=lambda r: r["item_id"])
        cut = max(1, int(0.8 * len(items)))
        train += items[:cut]
        test += items[cut:]
    if len(train) < 8 or len(test) < 4:
        print(f"  [eval] step {step}: only {len(recs)} complete items — skip until more data", flush=True)
        return
    labels = [SoftLabel(r["item_id"], r["prompt"], wids,
                        [r["rewards"][w] for w in wids],
                        _soft_cost([r["rewards"][w] for w in wids], [r["costs"][w] for w in wids]),
                        "g") for r in train]
    print(f"  [eval] step {step}: training router (SFT) on {len(labels)} labels / {len(recs)} "
          f"complete items, {len(test)} held-out...", flush=True)
    router = build_router(cfg)
    train_sft(router, labels, epochs=120, lr=0.02, batch_size=8, log_every=20)  # prints [sft] epoch kl
    save_router(router, ROUTER_CKPT, worker_ids=wids)
    print(f"  [eval] step {step}: SFT done -> evaluating held-out routing ({len(test)} items)...", flush=True)

    idx = {w: j for j, w in enumerate(wids)}
    def ev(rows):
        R = np.array([[r["rewards"][w] for w in wids] for r in rows], dtype=float)
        C = np.array([[r["costs"][w] for w in wids] for r in rows], dtype=float)
        routed = [idx[select_worker(router, raw_query(r["prompt"]))] for r in rows]
        ar = np.arange(len(rows))
        orch_res = R[ar, routed].mean(); orch_cost = C[ar, routed].mean()
        bj = int(R.mean(0).argmax())
        return (round(orch_res, 3), round(orch_cost, 5), round(R[:, bj].mean(), 3),
                round(C[:, bj].mean(), 5), wids[bj], round(R.max(1).mean(), 3))

    o_r, o_c, b_r, b_c, b_w, orc = ev(test)
    save = (1 - o_c / b_c) if b_c > 0 else 0.0
    rec = {"step": step, "n_train": len(train), "n_test": len(test),
           "overall": {"orch_resolve": o_r, "orch_cost": o_c, "best_worker": b_w,
                       "best_resolve": b_r, "best_cost": b_c, "oracle": orc,
                       "resolve_lift": round(o_r - b_r, 3), "cost_savings": round(save, 3)}}
    per = {}
    for dom in by_dom:
        drows = [r for r in test if r["domain"] == dom]
        if drows:
            dr = ev(drows)
            per[dom] = {"n": len(drows), "orch_resolve": dr[0], "best_resolve": dr[2],
                        "resolve_lift": round(dr[0] - dr[2], 3),
                        "cost_savings": round((1 - dr[1] / dr[3]) if dr[3] > 0 else 0.0, 3)}
    rec["per_domain"] = per
    with open(METRICS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  [eval] step {step}: train={len(train)} test={len(test)} | "
          f"resolve_lift={o_r-b_r:+.3f} (orch {o_r:.3f} vs best {b_r:.3f}, oracle {orc:.3f}) | "
          f"cost_savings={save:+.0%} (orch ${o_c:.4f} vs best ${b_c:.4f})", flush=True)
    for dom, m in per.items():
        print(f"      {dom:10} n={m['n']:3} resolve_lift={m['resolve_lift']:+.3f} "
              f"cost_savings={m['cost_savings']:+.0%}", flush=True)


# ---- the loop -------------------------------------------------------------
migrate_bank()  # bring any legacy per-item records into the per-(item,worker) cell format
seen_cells = {(c["domain"], c["item_id"], c["worker"]) for c in load_bank() if "worker" in c}


def n_complete() -> int:
    by_item: dict = {}
    for d, i, w in seen_cells:
        by_item.setdefault((d, i), set()).add(w)
    return sum(1 for ws in by_item.values() if all(w in ws for w in wids))


print(f"start: bank has {len(seen_cells)} cells / {n_complete()} complete items, "
      f"{STEPS} steps, eval every {EVAL_EVERY}", flush=True)
def _build_single_pool(srcs):
    # build_candidates concatenates sources block-by-block; shuffle the merged pool so the growing
    # per-step window (single_pools[name][:N]) samples ALL sources, not just the first one.
    import random as _random
    pool = list(build_candidates(srcs, per_source_limit=(STEPS + 2) * SINGLE_BATCH * 3, seed=0))
    _random.Random(0).shuffle(pool)
    return pool

single_pools = {name: _build_single_pool(src if isinstance(src, list) else [src])
                for name, kind, src in DOMAINS if kind == "single"}
AGENTIC_LIMIT = (STEPS + 2) * AGENTIC_BATCH * 3


def _load_agentic(src: str):
    if src == "swesmith":
        return load_swesmith_tasks(AGENTIC_LIMIT)
    if src == "swebench":
        return load_swe_tasks(SWE_DATASET, AGENTIC_LIMIT)
    if src == "swebench_pro":
        return load_swe_pro_tasks(AGENTIC_LIMIT)
    if src == "terminal":
        return load_terminal_tasks("terminal-bench-core", AGENTIC_LIMIT)
    if src == "tau_retail":
        return load_tau_tasks("retail", AGENTIC_LIMIT)
    if src == "tau_airline":
        return load_tau_tasks("airline", AGENTIC_LIMIT)
    raise ValueError(f"unknown agentic source {src}")


# Load each agentic pool, but degrade gracefully: a source that fails to load (missing dataset,
# harness, etc.) is skipped with a warning rather than crashing the whole loop.
agentic_pools = {}
for _name, _kind, _src in DOMAINS:
    if _kind != "agentic":
        continue
    try:
        agentic_pools[_name] = _load_agentic(_src)
        print(f"agentic pool {_name} ({_src}): {len(agentic_pools[_name])} tasks", flush=True)
    except Exception as _e:
        agentic_pools[_name] = []
        print(f"agentic pool {_name} ({_src}) FAILED to load: {type(_e).__name__}: {_e} — skipping", flush=True)

# One persistent event loop for ALL gen_single calls. The pool's asyncio.Semaphore binds to
# the first loop that uses it, so a fresh asyncio.run() per domain crashes with "bound to a
# different event loop" on the second domain. Reuse a single loop instead.
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

async def run_step(step: int, agentic_sem) -> None:
    """All domains for a step run CONCURRENTLY on the one event loop (single-step + every
    agentic source interleaved). Agentic rollouts share ONE semaphore so total concurrent
    docker rollouts stay capped; single-step + worker calls share the pool's rate gate."""
    async def run_domain(name, kind, src):
        # growing target window: coverage grows by BATCH/step while earlier items' missing
        # cells (e.g. a flaky worker) get retried — bounded to STEPS attempts total.
        if kind == "single":
            target = single_pools[name][: SINGLE_BATCH * (step + 1)]
            n = await gen_single(step, name, target, seen_cells)
            print(f"  {name}: +{n} single cells  (spent ${pool.budget.spent_usd:.4f})", flush=True)
        else:
            target = agentic_pools.get(name, [])[: AGENTIC_BATCH * (step + 1)]
            n = await gen_agentic(step, name, src, target, seen_cells, agentic_sem)
            print(f"  {name}: +{n} {name} agentic cells", flush=True)

    await asyncio.gather(*[run_domain(name, kind, src) for name, kind, src in DOMAINS])


for step in range(STEPS):
    t0 = time.time()
    print(f"=== STEP {step+1}/{STEPS} | bank {len(seen_cells)} cells / {n_complete()} complete items "
          f"| spent ${_spent():.4f} ===", flush=True)
    loop.run_until_complete(run_step(step, asyncio.Semaphore(AGENTIC_PARALLEL)))
    print(f"  step {step} done in {time.time()-t0:.0f}s", flush=True)
    if (step + 1) % EVAL_EVERY == 0:
        train_and_eval(step)
# ---- Stage 2: sep-CMA-ES on end-to-end terminal reward (the evolutionary step) ----------
# Off by default (CMAES_GENERATIONS=0). When enabled, it runs AFTER the gen+SFT loop: warm-start
# from the SFT checkpoint, then evolve the router on LIVE routed agentic rollouts (the candidate
# routes per-step, allowed=None) maximizing mean terminal reward. This is where Fugu's real lift
# comes from (SFT is only the warm-start). Expensive: popsize x generations x eval-tasks rollouts.
CMAES_GENERATIONS = int(os.getenv("CMAES_GENERATIONS", "0"))
CMAES_EVAL_TASKS = int(os.getenv("CMAES_EVAL_TASKS", "6"))
CMAES_POPSIZE = int(os.getenv("CMAES_POPSIZE", "0")) or None
CMAES_SIGMA0 = float(os.getenv("CMAES_SIGMA0", "0.03"))  # Fugu's SVF sigma0
# Sources whose runner routes via ctx.router (the CMA-ES candidate) qualify. select_worker now
# globally locks the GPU forward, so mini-swe-agent's threaded featurize is race-free alongside the
# event-loop runners -> swebench (mini-swe-agent) is included (mandatory: it's Fugu's SWE harness).
# terminal is still excluded: its DirectorAgent loads its OWN router, not the in-process candidate.
CMAES_SOURCES = {"swebench", "swebench_pro", "tau_retail", "tau_airline"}


def run_cmaes() -> None:
    from director.fugu.cmaes import evolve

    if not os.path.exists(ROUTER_CKPT):
        print("[cmaes] no SFT checkpoint — running one train_and_eval first", flush=True)
        train_and_eval(STEPS - 1)
    if not os.path.exists(ROUTER_CKPT):
        print("[cmaes] still no checkpoint (not enough complete items) — skipping CMA-ES", flush=True)
        return
    cmaes_router = build_router(cfg, ckpt=ROUTER_CKPT)                 # warm-start from SFT
    cmaes_ctx = SimpleNamespace(**vars(ctx))
    cmaes_ctx.router = cmaes_router
    per = max(1, CMAES_EVAL_TASKS // max(len([1 for _, k, s in DOMAINS if k == "agentic" and s in CMAES_SOURCES]), 1))
    eval_tasks = [(s, t["payload"]) for name, k, s in DOMAINS if k == "agentic" and s in CMAES_SOURCES
                  for t in agentic_pools.get(name, [])[:per]][:CMAES_EVAL_TASKS]
    if not eval_tasks:
        print("[cmaes] no eval tasks available — skipping CMA-ES", flush=True)
        return

    async def _routed_eval() -> float:
        sem = asyncio.Semaphore(AGENTIC_PARALLEL)

        async def one(src, payload):
            runner, _ = AGENTIC_RUNNERS[src]
            async with sem:
                try:
                    reward, _c = await runner(cmaes_ctx, payload, None)  # allowed=None -> route LIVE
                    return float(reward)
                except Exception:
                    return 0.0

        rs = await asyncio.gather(*[one(s, p) for s, p in eval_tasks])
        return sum(rs) / max(len(rs), 1)

    def eval_fn() -> float:
        return loop.run_until_complete(_routed_eval())  # reuse the persistent loop (pool stays bound)

    print(f"[cmaes] Stage 2: {CMAES_GENERATIONS} gens x popsize={CMAES_POPSIZE or 'auto'} "
          f"over {len(eval_tasks)} live routed eval-tasks (sigma0={CMAES_SIGMA0})", flush=True)
    res = evolve(cmaes_router, eval_fn, generations=CMAES_GENERATIONS, sigma0=CMAES_SIGMA0,
                 popsize=CMAES_POPSIZE, checkpoint_dir="cmaes_ckpt", resume=True, verbose=True)
    save_router(cmaes_router, "fugu_router_cmaes.pt", worker_ids=wids)
    print(f"[cmaes] done: best terminal reward={res.best_fitness:.3f} over {res.generations_run} gens "
          f"-> saved fugu_router_cmaes.pt", flush=True)


if CMAES_GENERATIONS > 0:
    run_cmaes()

loop.close()
print("loop finished.", flush=True)
