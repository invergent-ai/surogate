"""Agentic headroom baseline via the REAL harness (mini-swe-agent): each open worker solo
on N SWE-Bench Verified tasks. Decisive cut — do different workers solve different tasks
(complementary => routing headroom)? Per-worker rollouts threaded; grading batched (1 harness
call per worker). Global $ cap via MSWEA_GLOBAL_COST_LIMIT.
"""
from __future__ import annotations
import json, os, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

RESULTS = "swebench_results.jsonl"  # append-only bank of (task_id, worker) -> rollout outcome
_wlock = threading.Lock()


def _load_results() -> dict:
    recs = {}
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            line = line.strip()
            if line:
                r = json.loads(line)
                recs[(r["task_id"], r["worker"])] = r
    return recs


def _append_result(rec: dict) -> None:
    with _wlock:
        with open(RESULTS, "a") as f:
            f.write(json.dumps(rec) + "\n")
from director.config import DirectorConfig, FeaturizerConfig, default_frontier_pool
from director.fugu.run import build_router
from director.agentic.swebench_env import load_swebench
from director.agentic.swebench_mini import run_instance, grade_batch

N_TASKS = 50
PARALLEL = 24              # 64 cores / 328GB free => embarrassingly parallel rollouts
COST_LIMIT = 0.4           # per-task cost cap (real solves land well under this)
STEP_BACKSTOP = 80         # bound steps too, in case a provider reports cost=0
DATASET = "princeton-nlp/SWE-bench_Verified"
# routing only needs goal + recent state; cap context so the 0.6B forward stays small
cfg = DirectorConfig(workers=default_frontier_pool(), featurizer=FeaturizerConfig(context_window=2048))
wids = cfg.worker_ids
slugs = {w.worker_id: w.model for w in cfg.workers}
router = build_router(cfg)
insts = load_swebench(dataset=DATASET, limit=N_TASKS, shuffle=True, seed=0)
iids = [i["instance_id"] for i in insts]
print(f"REAL-harness pilot: {N_TASKS} SWE-Bench Verified x {len(wids)} workers (solo), cost_limit=${COST_LIMIT}/task")

recs = _load_results()
print(f"loaded {len(recs)} prior banked rollouts; "
      f"reusing those, running only new (task,worker) pairs")
# TASK-MAJOR (interleave workers within each task) so complete (task x all-workers) rows
# accumulate first -> we can stop anytime and still have a trainable matrix. NOT worker-major
# (which would finish one worker's whole column before starting the next).
jobs = [(wj, ti) for ti in range(N_TASKS) for wj in range(len(wids))
        if (iids[ti], wids[wj]) not in recs]
print(f"new rollouts to run: {len(jobs)} (of {N_TASKS*len(wids)} total), task-major order")
t0 = time.time()


def do(wj, ti):
    r = run_instance(router, insts[ti], slugs, allowed={wids[wj]}, cost_limit=COST_LIMIT,
                     step_limit=STEP_BACKSTOP, do_grade=False)
    return wj, ti, r

# Phase 1: run only the NEW rollouts, banking each patch immediately (crash-safe + reusable)
with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
    futs = [ex.submit(do, wj, ti) for wj, ti in jobs]
    done = 0
    for fut in as_completed(futs):
        wj, ti, r = fut.result()
        rec = {"task_id": iids[ti], "worker": wids[wj], "dataset": DATASET, "patch": r["patch"],
               "exit_status": r["exit_status"], "cost": r["cost"], "steps": len(r["worker_sequence"]),
               "resolved": None}
        recs[(iids[ti], wids[wj])] = rec
        _append_result(rec)
        done += 1
        print(f"  [{done:3}/{len(jobs)}] {wids[wj]:9} {iids[ti][:28]:28} steps={rec['steps']:3} "
              f"patch={len(r['patch']):5} {r['exit_status']} (elapsed {time.time()-t0:.0f}s)", flush=True)

# Phase 2: grade any ungraded patches (cheap; reuses banked patches), per worker
for w in wids:
    ungraded = {iids[ti]: recs[(iids[ti], w)]["patch"] for ti in range(N_TASKS)
                if (iids[ti], w) in recs and recs[(iids[ti], w)].get("resolved") is None}
    if ungraded:
        resolved = grade_batch(ungraded, DATASET, run_id=f"pilot_{w}", max_workers=8)
        for iid in ungraded:
            recs[(iid, w)]["resolved"] = 1.0 if iid in resolved else 0.0
# rewrite the bank with grades filled (deduped, resolved persisted)
with open(RESULTS, "w") as f:
    for rec in recs.values():
        f.write(json.dumps(rec) + "\n")

R = np.zeros((N_TASKS, len(wids)))
for ti in range(N_TASKS):
    for wj, w in enumerate(wids):
        rec = recs.get((iids[ti], w))
        R[ti, wj] = float(rec["resolved"]) if rec and rec.get("resolved") is not None else 0.0
for wj, w in enumerate(wids):
    print(f"  {w}: resolved {int(R[:, wj].sum())}/{N_TASKS}", flush=True)

per = R.mean(0); bi = int(per.argmax())
oracle = R.max(1).mean(); best = per[bi]
print("\n=== AGENTIC HEADROOM (SWE-Bench Verified, real mini-swe-agent harness) ===")
print("per-worker resolve:", {wids[j]: round(per[j], 3) for j in range(len(wids))})
print(f"best worker = {wids[bi]} ({best:.3f}) | oracle (best-per-task) = {oracle:.3f}")
print(f"headroom = {oracle - best:+.3f}   RER = {(oracle-best)/(1-best+1e-9):.3f}")
print("solved-by (complementarity):")
for ti, iid in enumerate(iids):
    ws = [wids[j] for j in range(len(wids)) if R[ti, j] > 0]
    print(f"  {iid[:36]:36} {ws}")
np.save("pilot_swebench_R.npy", R)
print("\nsaved matrix to pilot_swebench_R.npy")
