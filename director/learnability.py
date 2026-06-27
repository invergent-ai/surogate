"""Learnability check for the escalation router (GLM default -> escalate to frontier when needed).

Trains a COST-AWARE router (head-only, on cached frozen features) on the screen data and measures, via
4-fold CV, the HELD-OUT cost-quality frontier as we sweep the cost weight lambda. Compares to the
reference points: always-GLM ($4), always-best-single (Opus $25), and the perfect-escalation ceiling
(cheapest-adequate). The question: does a real router capture a meaningful share of the $8-vs-$25 gap
at matched quality on UNSEEN queries?
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

MANIFEST = "manifests/fugu_clean_v1"
PRICE = {"opus": 25.0, "gemini": 12.0, "gpt": 30.0, "glm": 4.0}
TAU = 0.1


def main():
    from director.config import DirectorConfig, FeaturizerConfig, default_frontier_pool
    from director.fugu.run import build_router
    from director.shared.transcript import raw_query

    rew = {}
    for l in open(f"{MANIFEST}/pool_matrix_frontier.jsonl"):
        if l.strip():
            d = json.loads(l); rew[d["task_id"]] = d
    prompts = {}
    for l in open(f"{MANIFEST}/probe.jsonl"):
        if l.strip():
            d = json.loads(l); prompts[d["task_id"]] = d["prompt"]
    ids = [t for t in rew if t in prompts]
    wids = rew[ids[0]]["worker_ids"]
    price = np.array([PRICE[w] for w in wids])
    R = np.array([rew[t]["r_bar"] for t in ids])  # (I, 4) mean reward
    texts = [raw_query(prompts[t]) for t in ids]
    I = len(ids)
    glm_j = wids.index("glm")
    print(f"items={I} workers={wids} prices={[PRICE[w] for w in wids]}", flush=True)

    cfg = DirectorConfig(workers=default_frontier_pool(), featurizer=FeaturizerConfig(context_window=2048))
    router = build_router(cfg); feat = router.featurizer; dev = next(router.parameters()).device; d = feat.d

    @torch.no_grad()
    def feats(ix):
        H = []
        for i in range(0, len(ix), 32):
            H.append(feat.features([texts[j] for j in ix[i:i + 32]]).float())
        return torch.cat(H)
    print("caching features...", flush=True)
    Hall = feats(list(range(I)))

    # references
    best_q = float(R.mean(0).max()); best_w = wids[int(R.mean(0).argmax())]
    glm_q = float(R[:, glm_j].mean()); oracle_q = float(R.max(1).mean())
    def cheapest_adeq(Rm):
        q, c = [], []
        for i in range(len(Rm)):
            a = np.where(Rm[i] >= Rm[i].max() - 1e-9)[0]; w = a[np.argmin(price[a])]
            q.append(Rm[i, w]); c.append(price[w])
        return float(np.mean(q)), float(np.mean(c))
    ca_q, ca_c = cheapest_adeq(R)
    print(f"\nREFERENCES:")
    print(f"  always-GLM:        q={glm_q:.3f}  ${4}")
    print(f"  best-single({best_w}): q={best_q:.3f}  ${PRICE[best_w]:g}")
    print(f"  perfect escalation: q={ca_q:.3f}  ${ca_c:.1f}   (per-item oracle q={oracle_q:.3f})", flush=True)

    def soft(Rm, lam):
        ra = Rm - lam * (price / price.max())[None, :]
        z = ra / TAU; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

    def cv(lam, folds=4, epochs=400):
        idx = list(range(I)); random.Random(0).shuffle(idx); fsz = I // folds
        qs, cs, picks_all = [], [], []
        for f in range(folds):
            val = idx[f * fsz:(f + 1) * fsz] if f < folds - 1 else idx[f * fsz:]
            tr = [i for i in idx if i not in set(val)]
            P = torch.tensor(soft(R[tr], lam), dtype=torch.float32, device=dev)
            head = torch.nn.Linear(d, 4, bias=False).to(dev)
            opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-3)
            Htr = Hall[tr]
            for _ in range(epochs):
                opt.zero_grad(); loss = F.kl_div(F.log_softmax(head(Htr), -1), P, reduction="batchmean")
                loss.backward(); opt.step()
            with torch.no_grad():
                pk = head(Hall[val]).argmax(-1).cpu().numpy()
            for vi, p in zip(val, pk):
                qs.append(R[vi, p]); cs.append(price[p]); picks_all.append(wids[p])
        return float(np.mean(qs)), float(np.mean(cs)), Counter(picks_all)

    print(f"\nLEARNED ROUTER (4-fold CV, held-out) — cost-quality frontier:")
    print(f"  {'lambda':>7} {'held-out q':>11} {'avg $':>7}   routing dist")
    for lam in [0.0, 0.03, 0.06, 0.1, 0.15, 0.25, 0.4, 0.7]:
        q, c, dist = cv(lam)
        print(f"  {lam:>7.2f} {q:>11.3f} {c:>6.1f}   {dict(dist)}", flush=True)
    print(f"\nread: a point that beats best-single ($={PRICE[best_w]:g}, q={best_q:.3f}) means SAME-OR-BETTER "
          f"quality at LOWER cost -> escalation is learnable. Ceiling is q={ca_q:.3f} @ ${ca_c:.1f}.")


if __name__ == "__main__":
    main()
