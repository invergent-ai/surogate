"""n=4 verdict built from CACHE ONLY (zero API calls).

The live n=4 labeler kept hanging on the last few items' calls. All completed
worker samples are on disk, so this reconstructs the n=4 soft targets straight from
the cache: for each manifest item it grades the 4 cached samples per worker, skips any
item with a missing sample, then runs the same headroom + SFT + held-out routing eval.

Usage: .venv/bin/python validate_cached.py
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from director.data.manifest import read_manifest
from director.fugu.inference import select_worker
from director.fugu.labels import SoftLabel
from director.fugu.run import build_router, load_config
from director.fugu.sft import train_sft
from director.shared.cache import CompletionCache, completion_key
from director.shared.transcript import raw_query
from director.shared.types import Sampling
from director.shared.verifiers import get_grader

N = 4
cfg = load_config("validate.yaml")
wids = cfg.worker_ids
models = [w.model for w in cfg.workers]
print(f"workers ({len(wids)}): {wids}")

cache = CompletionCache(cfg.pool.cache_dir)
items = read_manifest("manifests/validate")


def cached_rbar(task):
    """Mean graded pass-rate per worker from cached samples; None if any sample missing."""
    grader = get_grader(task.grader)
    msgs = task.messages()
    rbar = []
    for m in models:
        rewards = []
        for s in range(N):
            key = completion_key(m, msgs, Sampling(temperature=0.7, top_p=1.0, max_tokens=16384, seed=s))
            comp = cache.get(key)
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


labels, split_of, skipped = [], {}, 0
for it in items:
    rbar = cached_rbar(it.to_task())
    if rbar is None:
        skipped += 1
        continue
    labels.append(SoftLabel(it.task_id, it.prompt, wids, rbar, softmax(rbar), it.grader))
    split_of[it.task_id] = it.split
print(f"reconstructed {len(labels)} items from cache  (skipped {skipped} with a missing sample)\n")

# headroom on graded n=4 rewards
R = np.array([lab.r_bar for lab in labels], dtype=float)
per = R.mean(0)
bi = int(per.argmax())
print("=== HEADROOM (n=4 graded, discriminative set) ===")
print(f"  oracle(mean max)={R.max(1).mean():.3f}  best={wids[bi]}({per[bi]:.3f})  "
      f"headroom={R.max(1).mean()-per[bi]:+.3f}")
print("  per-worker mean pass-rate:", {w: round(per[j], 3) for j, w in enumerate(wids)})

# train SFT on train split, eval held-out
train = [lab for lab in labels if split_of.get(lab.task_id) == "train"]
test = [lab for lab in labels if split_of.get(lab.task_id) == "test"]
router = build_router(cfg)
print(f"\n{router.summary()}")
stats = train_sft(router, train, epochs=120, lr=0.02, batch_size=4, log_every=40)
print(f"SFT final KL: {stats.final_loss:.4f}")

idx = {w: j for j, w in enumerate(wids)}
testR = np.array([lab.r_bar for lab in test], dtype=float)
routed = [select_worker(router, raw_query(lab.prompt)) for lab in test]
orch = np.array([test[i].r_bar[idx[routed[i]]] for i in range(len(test))])
best = testR.mean(0).max()
best_w = wids[int(testR.mean(0).argmax())]
oracle = testR.max(1).mean()
print("\n=== HELD-OUT ROUTING (n=4 de-noised, cache-only) ===")
print(f"  train n={len(train)}  held-out n={len(test)}")
print(f"  orchestrator : {orch.mean():.3f}")
print(f"  best worker  : {best:.3f} ({best_w})")
print(f"  oracle       : {oracle:.3f}")
print(f"  LIFT vs best : {orch.mean()-best:+.3f}  (captured {(orch.mean()-best)/(oracle-best+1e-9):.0%} of headroom)")
print(f"  routing dist : {dict(Counter(routed))}")
