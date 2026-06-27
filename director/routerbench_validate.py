"""FREE validation of the Fugu router on RouterBench's real 11-model reward matrix.

Tests the core bet — can our frozen Qwen3-0.6B penultimate features + a bias-free head
learn to route? — at $0 (rewards are precomputed in RouterBench, no API calls).

Pipeline: load RouterBench (per-prompt × per-model correctness) → precompute OUR faithful
features once (raw role:content, EOS-appended, penultimate) → train the routing head by
soft-KL to softmax(r/τ) → measure held-out routing lift vs best single worker, oracle, RER.

Head-only (SVF frozen) for speed; per Trinity's ablation SVF adds only ~marginal lift, so
this is a faithful read on the make-or-break learnability. Usage: .venv/bin/python routerbench_validate.py
"""

from __future__ import annotations

import math
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from director.config import DirectorConfig, FeaturizerConfig, WorkerSpec
from director.fugu.run import build_router
from director.shared.transcript import raw_query

TAU = 0.1
FEAT_CACHE = "routerbench_feats.pt"
MAX_ITEMS = 12000  # stratified subsample for a strong-but-fast signal
torch.manual_seed(0)
np.random.seed(0)

import pandas as pd

p = hf_hub_download("withmartian/routerbench", "routerbench_0shot.pkl", repo_type="dataset")
df = pd.read_pickle(p)
# No dominant model: drop gpt-4, use evenly-matched weaker models so dominance shifts by
# domain (a code specialist + general models) — tests whether routing wins on a COMPLEMENTARY pool.
MODELS = [
    "gpt-3.5-turbo-1106",
    "claude-v2",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
    "meta/code-llama-instruct-34b-chat",
]
short = [m.split("/")[-1] for m in MODELS]

df = df.dropna(subset=MODELS)
R = df[MODELS].to_numpy(dtype=float)
R = np.clip(R, 0.0, 1.0)  # MT-Bench-style continuous scores clipped into [0,1]
prompts = [pp[0] if isinstance(pp, (list, tuple)) else str(pp) for pp in df["prompt"].tolist()]
evals = df["eval_name"].tolist()

# stratified subsample by eval_name
by_eval = defaultdict(list)
for i, e in enumerate(evals):
    by_eval[e].append(i)
rng = np.random.default_rng(0)
keep: list[int] = []
per = max(1, MAX_ITEMS // max(len(by_eval), 1))
for e, idxs in by_eval.items():
    idxs = list(idxs)
    rng.shuffle(idxs)
    keep.extend(idxs[:per])
rng.shuffle(keep)
keep = keep[:MAX_ITEMS]
prompts = [prompts[i] for i in keep]
evals = [evals[i] for i in keep]
R = R[keep]
N = len(prompts)
print(f"RouterBench: {N} prompts × {len(MODELS)} models, {len(by_eval)} eval domains")
per_worker = R.mean(0)
print("per-worker overall accuracy:")
for j, m in enumerate(short):
    print(f"  {m:34} {per_worker[j]:.3f}")
print(f"oracle(mean row-max)={R.max(1).mean():.3f}  best={short[int(per_worker.argmax())]}({per_worker.max():.3f})")

cfg = DirectorConfig(
    workers=[WorkerSpec(worker_id=s, model=m) for s, m in zip(short, MODELS)],
    featurizer=FeaturizerConfig(context_window=2048),  # routing needs the question, not full long context
)
router = build_router(cfg)
feat = router.featurizer

# precompute features once (cached to disk)
texts = [raw_query(t) for t in prompts]
if os.path.exists(FEAT_CACHE):
    blob = torch.load(FEAT_CACHE)
    H = blob["H"]
    assert H.shape[0] == N, "cache size mismatch; delete routerbench_feats.pt"
    print(f"loaded cached features {tuple(H.shape)}")
else:
    print("precomputing features (one-time)...")
    feats = []
    with torch.no_grad():
        B = 8
        for i in range(0, N, B):
            feats.append(feat.features(texts[i : i + B]).float().cpu())
            if (i // B) % 20 == 0:
                print(f"  {i}/{N}", flush=True)
    H = torch.cat(feats)
    torch.save({"H": H}, FEAT_CACHE)
    print(f"features {tuple(H.shape)} cached")

# split
idx = np.arange(N)
rng.shuffle(idx)
cut = int(0.85 * N)
tr, te = idx[:cut], idx[cut:]
Hn = H.numpy()
mu = Hn[tr].mean(0); sd = Hn[tr].std(0) + 1e-6      # STANDARDIZE (frozen feats have large scale)
Hs = (Hn - mu) / sd
Rt = R[tr]; Rte = R[te]
best = Rte.mean(0).max(); best_w = short[int(Rte.mean(0).argmax())]
oracle = Rte.max(1).mean()


def report(tag, routed):
    orch = Rte[np.arange(len(te)), routed].mean()
    rer = (orch - best) / (1 - best + 1e-9)
    dist = Counter(short[j] for j in routed)
    print(f"\n=== {tag} (held-out n={len(te)}, $0) ===")
    print(f"  orchestrator={orch:.3f}  best={best:.3f}({best_w})  oracle={oracle:.3f}  "
          f"LIFT={orch-best:+.3f} ({(orch-best)/(oracle-best+1e-9):.0%} of headroom)  RER={rer:.3f}")
    print(f"  routing dist: {dict(sorted(dist.items(), key=lambda x:-x[1]))}")


# (1) clean feature-routability probe: torch multinomial logistic regression -> per-prompt
# best worker (Trinity Fig 12). Pure torch, no sklearn dependency.
import torch.nn as nn
dev = next(router.parameters()).device
L = len(short)
dfeat = Hs.shape[1]
ytr = Rt.argmax(1)
Xtr = torch.tensor(Hs[tr], dtype=torch.float32, device=dev)
Xte = torch.tensor(Hs[te], dtype=torch.float32, device=dev)
ytr_t = torch.tensor(ytr, dtype=torch.long, device=dev)
probe = nn.Linear(dfeat, L).to(dev)
popt = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-4)
npr = len(tr); bs = 256
for epoch in range(200):
    perm = torch.randperm(npr, device=dev)
    for s in range(0, npr, bs):
        b = perm[s : s + bs]
        popt.zero_grad()
        loss = F.cross_entropy(probe(Xtr[b]), ytr_t[b])
        loss.backward(); popt.step()
with torch.no_grad():
    pred = probe(Xte).argmax(1).cpu().numpy()
acc = (pred == Rte.argmax(1)).mean()
print(f"\nlinear probe: best-worker classification acc={acc:.3f} (chance={1/L:.3f})")
report("LINEAR PROBE ROUTING (full dist)", pred)

# Fugu's value lives on DISCRIMINATIVE items (workers disagree AND it's solvable).
disc = (Rte.max(1) > Rte.min(1)) & (Rte.max(1) > 0)
nd = int(disc.sum())
od = Rte[disc][np.arange(nd), pred[disc]].mean()
bd = Rte[disc].mean(0).max(); bdw = short[int(Rte[disc].mean(0).argmax())]
ord_ = Rte[disc].max(1).mean()
print(f"\n=== LINEAR PROBE ROUTING (DISCRIMINATIVE subset n={nd}/{len(te)}, $0) ===")
print(f"  orchestrator={od:.3f}  best={bd:.3f}({bdw})  oracle={ord_:.3f}  "
      f"LIFT={od-bd:+.3f} ({(od-bd)/(ord_-bd+1e-9):.0%} of headroom)  RER={(od-bd)/(1-bd+1e-9):.3f}")

# (2) our faithful soft-KL head (standardized feats, minibatch, sane lr) -- confirms training works
Ht = torch.tensor(Hs[tr], dtype=torch.float32, device=dev)
P = np.exp(R / TAU); P = P / P.sum(1, keepdims=True)
Pt = torch.tensor(P[tr], dtype=torch.float32, device=dev)
head = router.head.to(dev)
for p_ in router.featurizer.parameters():
    p_.requires_grad_(False)
opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
ntr = len(tr); bs = 256
for epoch in range(120):
    perm = torch.randperm(ntr, device=dev)
    tot = 0.0
    for s in range(0, ntr, bs):
        b = perm[s : s + bs]
        opt.zero_grad()
        loss = F.kl_div(F.log_softmax(head(Ht[b]), dim=-1), Pt[b], reduction="batchmean")
        loss.backward(); opt.step(); tot += float(loss)
    if epoch % 40 == 0 or epoch == 119:
        print(f"[kl-head] epoch {epoch:4d} kl={tot/max(1,ntr//bs):.4f}")
with torch.no_grad():
    routed = head(torch.tensor(Hs[te], dtype=torch.float32, device=dev)).argmax(1).cpu().numpy()
report("SOFT-KL HEAD ROUTING", routed)
