"""CMA-ES stage validation (cache-only, zero API calls).

Builds n=4 soft labels from cache, runs the SFT warm-start, then refines with
sep-CMA-ES against the ACTUAL routing objective (argmax worker -> cached n=4 reward),
not SFT's KL-to-soft-target proxy. Reports held-out lift before vs after CMA-ES.

CAVEAT: 11,264 trainable params vs ~73 train items => CMA-ES can overfit train reward.
This validates the mechanism (does the loop lift train fitness?) and reads held-out
honestly; the real CMA-ES payoff only shows at production data scale.

Usage: .venv/bin/python validate_cmaes.py
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import torch

import os

from director.data.manifest import read_manifest
from director.fugu.cmaes import evolve
from director.fugu.inference import select_worker
from director.fugu.labels import SoftLabel, load_labels, save_labels
from director.fugu.run import build_router, load_config
from director.fugu.sft import train_sft
from director.shared.cache import CompletionCache, completion_key
from director.shared.transcript import raw_query
from director.shared.types import Sampling
from director.shared.verifiers import get_grader

N = 4
GENERATIONS = 30
LABELS_PATH = "labels_n4.jsonl"
cfg = load_config("validate.yaml")
wids = cfg.worker_ids
models = [w.model for w in cfg.workers]
cache = CompletionCache(cfg.pool.cache_dir)
items = read_manifest("manifests/validate")


def cached_rbar(task):
    grader = get_grader(task.grader)
    msgs = task.messages()
    rbar = []
    for m in models:
        rewards = []
        for s in range(N):
            comp = cache.get(completion_key(m, msgs, Sampling(temperature=0.7, top_p=1.0, max_tokens=16384, seed=s)))
            if comp is None:
                return None
            rewards.append(float(grader(comp.text, task.solution)))
        rbar.append(sum(rewards) / len(rewards))
    return rbar


def softmax(xs, tau=0.1):
    m = max(xs)
    e = [math.exp((x - m) / tau) for x in xs]
    z = sum(e)
    return [v / z for v in e]


# Reuse cached n=4 labels across runs: re-grading 1.5k samples (code_exec subprocesses
# + sympy) costs ~5 min, but the graded r_bar is deterministic, so persist it once.
split_of = {it.task_id: it.split for it in items}
if os.path.exists(LABELS_PATH):
    labels = load_labels(LABELS_PATH)
    print(f"loaded {len(labels)} cached labels from {LABELS_PATH}")
else:
    labels = []
    for it in items:
        rb = cached_rbar(it.to_task())
        if rb is None:
            continue
        labels.append(SoftLabel(it.task_id, it.prompt, wids, rb, softmax(rb), it.grader))
    save_labels(labels, LABELS_PATH)
    print(f"built + saved {len(labels)} labels to {LABELS_PATH}")
train = [lab for lab in labels if split_of.get(lab.task_id) == "train"]
test = [lab for lab in labels if split_of.get(lab.task_id) == "test"]
print(f"train n={len(train)}  held-out n={len(test)}  (cache-only)\n")

idx = {w: j for j, w in enumerate(wids)}
testR = np.array([lab.r_bar for lab in test], dtype=float)
trainR = np.array([lab.r_bar for lab in train], dtype=float)
best = testR.mean(0).max()
best_w = wids[int(testR.mean(0).argmax())]
oracle = testR.max(1).mean()


def heldout_lift(router, tag):
    routed = [select_worker(router, raw_query(lab.prompt)) for lab in test]
    orch = np.array([test[i].r_bar[idx[routed[i]]] for i in range(len(test))]).mean()
    pct = (orch - best) / (oracle - best + 1e-9)
    print(f"  [{tag}] held-out orch={orch:.3f}  best={best:.3f}({best_w})  oracle={oracle:.3f}  "
          f"LIFT={orch-best:+.3f} ({pct:.0%} of headroom)  dist={dict(Counter(routed))}")
    return orch


router = build_router(cfg)
print(router.summary())
# Attention is O(batch * seq^2): a batch padded to one 4096-tok item OOMs at large
# counts, so 8 is the safe fixed size (8 * 4096^2 fits). Bigger gains come from the
# length-aware batching in the CMA-ES eval below.
stats = train_sft(router, train, epochs=120, lr=0.02, batch_size=8, log_every=120)
print(f"SFT final KL: {stats.final_loss:.4f}")
sft_orch = heldout_lift(router, "SFT")

# ---- CMA-ES on the actual routing objective (cached train reward) ----
train_texts = [raw_query(lab.prompt) for lab in train]

# Length-aware batching: sort by token length and pack each batch to a fixed
# (count * max_len) budget, so short prompts batch wide and long ones stay tiny --
# fast without the O(batch*seq^2) OOM that a fixed count hits on long items.
_tok = router.featurizer.tokenizer
_lens = [len(_tok(t, add_special_tokens=True)["input_ids"]) for t in train_texts]
_BUDGET = 12000  # batch_size * max_len cap (2*4096 worst case fits comfortably)
_order = sorted(range(len(train_texts)), key=lambda i: _lens[i])
_batches, _cur = [], []
for i in _order:
    _cur.append(i)
    if len(_cur) * _lens[i] >= _BUDGET:  # i is the longest (sorted asc) so far
        _batches.append(_cur)
        _cur = []
if _cur:
    _batches.append(_cur)
print(f"CMA-ES eval: {len(train_texts)} items in {len(_batches)} length-aware batches")


def train_fitness() -> float:
    """Mean cached reward of the worker the current router picks, over the train set."""
    router.eval()
    routed = np.empty(len(train), dtype=int)
    with torch.no_grad():
        for batch in _batches:
            lg = router.logits([train_texts[i] for i in batch]).argmax(1).cpu().numpy()
            for k, i in enumerate(batch):
                routed[i] = lg[k]
    return float(trainR[np.arange(len(train)), routed].mean())


sft_train_fit = train_fitness()
print(f"\nCMA-ES warm-start train fitness (SFT): {sft_train_fit:.3f}  (train oracle={trainR.max(1).mean():.3f})")
res = evolve(router, train_fitness, generations=GENERATIONS, sigma0=0.03, seed=0, verbose=True)
print(f"CMA-ES best train fitness: {res.best_fitness:.3f}  (gens run={res.generations_run})")
cma_orch = heldout_lift(router, "CMA-ES")

print("\n=== SFT vs CMA-ES (held-out) ===")
print(f"  SFT    : orch={sft_orch:.3f}  lift={sft_orch-best:+.3f}")
print(f"  CMA-ES : orch={cma_orch:.3f}  lift={cma_orch-best:+.3f}")
print(f"  delta from CMA-ES refinement: {cma_orch-sft_orch:+.3f}")
