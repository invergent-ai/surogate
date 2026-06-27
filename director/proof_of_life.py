"""Architecture-sanity gate (revised-recipe Step 1): a DELIBERATELY complementary pool where the
router MUST beat best-single. Validates hidden extraction, worker ordering, head dims, SVF grads, KL
direction, argmax, eval — end to end. If this FAILS, there is an implementation bug. If it PASSES,
our single-step pool rejection is a genuine pool/data result, not a code defect.

Construction: 4 synthetic experts, one per domain; reward = 1 iff the item's domain matches the
expert, else 0. So best-single = largest domain fraction (~0.25), oracle = 1.0 (huge headroom). The
0.6B's penultimate features are ~100% domain-separable (Trinity §4.6), so a correct impl should
capture ~all of it. Real prompts (features), synthetic rewards (guaranteed complementarity). GPU-only.
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict

import numpy as np

MANIFEST = "manifests/fugu_clean_v1"
DOMAINS = ["math", "code", "science", "general"]
TAU = 0.1


def soft(r):
    z = np.array(r) / TAU
    z -= z.max()
    e = np.exp(z)
    return (e / e.sum()).tolist()


def main():
    from director.config import DirectorConfig, FeaturizerConfig, WorkerSpec
    from director.fugu.inference import select_worker
    from director.fugu.labels import SoftLabel
    from director.fugu.run import build_router
    from director.fugu.sft import train_sft
    from director.shared.transcript import raw_query

    cfg = DirectorConfig(workers=[WorkerSpec(worker_id=d, model=f"synthetic/{d}") for d in DOMAINS],
                         featurizer=FeaturizerConfig(context_window=4096))
    wids = cfg.worker_ids
    idx = {w: j for j, w in enumerate(wids)}

    rows = [json.loads(l) for l in open(os.path.join(MANIFEST, "labels_n4_tau0.1.jsonl")) if l.strip()]
    rows = [r for r in rows if r["domain"] in DOMAINS]
    random.Random(0).shuffle(rows)

    def Rrow(dom):
        return [1.0 if w == dom else 0.0 for w in wids]

    nval = max(60, int(len(rows) * 0.2))
    val, tr = rows[:nval], rows[nval:]
    labels = [SoftLabel(task_id=r["task_id"], prompt=r["prompt"], worker_ids=wids,
                        r_bar=Rrow(r["domain"]), p=soft(Rrow(r["domain"])), grader="syn") for r in tr]
    R_val = np.array([Rrow(r["domain"]) for r in val])
    best, oracle = float(R_val.mean(0).max()), float(R_val.max(1).mean())
    print(f"[POL] experts={DOMAINS} train={len(tr)} val={len(val)} | "
          f"best-single={best:.3f} oracle={oracle:.3f} (headroom {oracle-best:+.3f})", flush=True)

    router = build_router(cfg)
    texts = [raw_query(r["prompt"]) for r in val]

    def routed():
        picks = [idx[select_worker(router, t)] for t in texts]
        return float(R_val[np.arange(len(val)), picks].mean()), picks

    r0, _ = routed()
    print(f"[POL] before SFT: routed={r0:.3f}", flush=True)
    train_sft(router, labels, epochs=int(os.getenv("EPOCHS", "150")), lr=float(os.getenv("LR", "1e-3")),
              batch_size=64, micro_batch=16, log_every=50)
    orch, picks = routed()
    pick_w = [wids[p] for p in picks]
    cap = (orch - best) / (oracle - best) if oracle > best else 0.0
    print(f"[POL] after SFT: routed={orch:.3f} capture={cap:.0%} dist={dict(Counter(pick_w))}", flush=True)
    bydom = defaultdict(lambda: [0, 0])
    for r, p in zip(val, pick_w):
        bydom[r["domain"]][0] += (p == r["domain"])
        bydom[r["domain"]][1] += 1
    for d in DOMAINS:
        if bydom[d][1]:
            print(f"    {d:8}: {bydom[d][0]}/{bydom[d][1]} routed to its expert", flush=True)
    ok = cap >= 0.90
    print(f"\nPROOF-OF-LIFE: {'PASS' if ok else 'FAIL'} (capture {cap:.0%}; expect ~100% — "
          f"0.6B features are ~100% domain-separable per Trinity §4.6)", flush=True)


if __name__ == "__main__":
    main()
