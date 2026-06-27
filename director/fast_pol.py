"""FAST proof-of-life: head-only on CACHED features. Precompute the frozen backbone's penultimate
features once, then train the bias-free linear head on the cached vectors (instant). Deliberately
complementary synthetic pool (one expert per domain) => router MUST beat best-single. Validates hidden
extraction, worker ordering, head dims, KL direction, argmax, eval. (SVF grads tested separately.)
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

MANIFEST = "manifests/fugu_clean_v1"
DOMAINS = ["math", "code", "science", "general"]
TAU = 0.1


def main():
    from director.config import DirectorConfig, FeaturizerConfig, WorkerSpec
    from director.fugu.run import build_router
    from director.shared.transcript import raw_query

    cfg = DirectorConfig(workers=[WorkerSpec(worker_id=d, model=f"synthetic/{d}") for d in DOMAINS],
                         featurizer=FeaturizerConfig(context_window=2048))
    router = build_router(cfg)
    feat = router.featurizer
    dev = next(router.parameters()).device
    d = feat.d

    rows = [json.loads(l) for l in open(f"{MANIFEST}/labels_n4_tau0.1.jsonl") if l.strip()]
    rows = [r for r in rows if r["domain"] in DOMAINS]
    random.Random(0).shuffle(rows)
    rows = rows[:1500]
    nval = 350
    val, tr = rows[:nval], rows[nval:]
    dom2j = {dm: j for j, dm in enumerate(DOMAINS)}

    @torch.no_grad()
    def feats(rs):
        H = []
        for i in range(0, len(rs), 32):
            H.append(feat.features([raw_query(r["prompt"]) for r in rs[i:i + 32]]).float())
        return torch.cat(H)

    print(f"[POL] caching features: train={len(tr)} val={len(val)} ...", flush=True)
    Htr, Hval = feats(tr), feats(val)
    print("[POL] features cached", flush=True)

    yval = np.array([dom2j[r["domain"]] for r in val])
    P = torch.stack([F.softmax(torch.tensor([1.0 if DOMAINS[j] == r["domain"] else 0.0
                                             for j in range(4)], device=dev) / TAU, dim=0) for r in tr])
    # best-single / oracle on val (reward=1 iff routed worker == item's domain expert)
    frac = Counter(r["domain"] for r in val)
    best = max(frac.values()) / len(val)
    oracle = 1.0

    head = torch.nn.Linear(d, 4, bias=False).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    for ep in range(400):
        opt.zero_grad()
        loss = F.kl_div(F.log_softmax(head(Htr), -1), P, reduction="batchmean")
        loss.backward()
        opt.step()
    picks = head(Hval).argmax(-1).cpu().numpy()
    acc = float((picks == yval).mean())  # == routed reward in this synthetic setup
    cap = (acc - best) / (oracle - best)
    dist = Counter(DOMAINS[p] for p in picks)
    print(f"[POL] best-single={best:.3f} oracle=1.000 | routed(head-only)={acc:.3f} capture={cap:.0%} dist={dict(dist)}", flush=True)
    bydom = defaultdict(lambda: [0, 0])
    for r, p in zip(val, picks):
        bydom[r["domain"]][0] += (DOMAINS[p] == r["domain"])
        bydom[r["domain"]][1] += 1
    for dm in DOMAINS:
        if bydom[dm][1]:
            print(f"    {dm:8}: {bydom[dm][0]}/{bydom[dm][1]} routed to its expert", flush=True)
    print(f"\nPROOF-OF-LIFE: {'PASS' if cap >= 0.90 else 'FAIL'} (capture {cap:.0%})", flush=True)


if __name__ == "__main__":
    main()
