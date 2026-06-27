"""Decisive validation, offline from a probed manifest (no extra API calls).

Answers the two make-or-break questions:
  Q1 (headroom):  is oracle-over-pool > best single worker?  (overall + per domain)
  Q2 (routing):   does a trained router beat the best single worker OUT-OF-SAMPLE?

Uses the probe's per-worker reward vectors as ground truth: trains SFT on the train
split's soft targets and evaluates routing on the held-out split — all from the manifest.

Usage: .venv/bin/python validate_run.py [manifest_dir] [config.yaml]
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict

import numpy as np

from director.data.manifest import read_manifest, read_meta, read_probes
from director.fugu.inference import select_worker
from director.fugu.labels import SoftLabel
from director.fugu.run import build_router, load_config
from director.fugu.sft import train_sft
from director.shared.transcript import raw_query

MDIR = sys.argv[1] if len(sys.argv) > 1 else "manifests/validate"
CFG = sys.argv[2] if len(sys.argv) > 2 else "validate.yaml"


def _softmax(xs, tau=0.1):
    m = max(xs)
    e = [math.exp((x - m) / tau) for x in xs]
    z = sum(e)
    return [v / z for v in e]


def headroom(items, worker_ids, label):
    if not items:
        print(f"{label}: (none)")
        return
    R = np.array([it.rewards for it in items], dtype=float)
    per_worker = R.mean(0)
    bi = int(per_worker.argmax())
    best, best_w = per_worker[bi], worker_ids[bi]
    oracle = (R.max(1) > 0).mean()
    rer = 0.0 if best >= 1.0 else (oracle - best) / (1.0 - best)  # error-space (Trinity A.6)
    print(f"{label:>12}: n={len(items):4d}  oracle={oracle:.3f}  "
          f"best={best_w}({best:.3f})  headroom={oracle - best:+.3f}  RER={rer:.3f}")


def main():
    cfg = load_config(CFG)
    meta = read_meta(MDIR)
    worker_ids = meta.worker_ids
    probes = read_probes(MDIR)        # ALL probed items (incl. saturated/dead) for headroom
    items = read_manifest(MDIR)       # discriminative + balanced + split for train/eval
    train = [it for it in items if it.split == "train"]
    test = [it for it in items if it.split == "test"]
    print(f"probed={len(probes)} (full dist) | manifest={len(items)} discriminative "
          f"({len(train)} train / {len(test)} test), workers={worker_ids}\n")

    # --- Q1: headroom (oracle vs best worker) on the FULL probed distribution ---
    print("=== HEADROOM on FULL distribution (oracle vs best single worker) ===")
    headroom(probes, worker_ids, "ALL")
    bydom = defaultdict(list)
    for it in probes:
        bydom[it.domain].append(it)
    for d, its in sorted(bydom.items()):
        headroom(its, worker_ids, d)
    print("\n=== HEADROOM on DISCRIMINATIVE items (where routing matters) ===")
    headroom(items, worker_ids, "DISC")

    # --- per-worker marginal value: does each worker earn its place? (full dist) ---
    print("\n=== PER-WORKER MARGINAL VALUE (keep/prune guide, full dist) ===")
    R = np.array([it.rewards for it in probes], dtype=float)
    correct = R >= 1.0
    base_oracle = correct.any(1).mean()
    for j, w in enumerate(worker_ids):
        solo = R[:, j].mean()
        sole = int((correct[:, j] & (correct.sum(1) == 1)).sum())  # only this worker solved
        loo = correct[:, [k for k in range(len(worker_ids)) if k != j]].any(1).mean()
        drop = base_oracle - loo  # ceiling lost if this worker removed
        verdict = "PRUNE?" if (sole == 0 and drop < 0.005) else "keep"
        print(f"  {w:>8}: solo={solo:.3f}  sole_wins={sole:3d}  leave-one-out_oracle_drop={drop:+.3f}  -> {verdict}")

    # --- train SFT on train-split soft targets (from probe reward vectors) ---
    print("\n=== TRAIN (SFT on train split) ===")
    labels = [
        SoftLabel(task_id=it.task_id, prompt=it.prompt, worker_ids=worker_ids,
                  r_bar=it.rewards, p=_softmax(it.rewards), grader=it.grader)
        for it in train
    ]
    router = build_router(cfg)
    print(router.summary())
    stats = train_sft(router, labels, epochs=120, lr=0.02, batch_size=4, log_every=40)
    print(f"SFT final KL: {stats.final_loss:.4f}")

    # --- Q2: held-out routing vs best worker (from probe rewards, no API) ---
    print("\n=== HELD-OUT ROUTING (does the router beat the best worker out-of-sample?) ===")
    idx_of = {w: j for j, w in enumerate(worker_ids)}
    R = np.array([it.rewards for it in test], dtype=float)
    routed = [select_worker(router, raw_query(it.prompt)) for it in test]
    orch = np.array([test[i].rewards[idx_of[routed[i]]] for i in range(len(test))])
    best = R.mean(0).max()
    best_w = worker_ids[int(R.mean(0).argmax())]
    oracle = (R.max(1) > 0).mean()
    from collections import Counter
    dist = Counter(routed)
    print(f"held-out n={len(test)}")
    print(f"  orchestrator : {orch.mean():.3f}")
    print(f"  best worker  : {best:.3f} ({best_w})")
    print(f"  oracle       : {oracle:.3f}")
    print(f"  LIFT vs best : {orch.mean() - best:+.3f}   (captured {(orch.mean()-best)/(oracle-best+1e-9):.0%} of headroom)")
    print(f"  routing dist : {dict(dist)}")


if __name__ == "__main__":
    main()
