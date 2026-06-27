"""n=4 de-noised re-test (5-worker pool, kimi removed).

The n=1 probe gives noisy binary rewards. This re-labels the discriminative items at
n=4 (graded pass-rates 0/.25/.5/.75/1), which de-noises the soft targets AND surfaces
reliability differences. Then it re-trains SFT and re-measures out-of-sample routing lift.

Seed-0 samples are cached from the probe → only seeds 1-3 are new spend.
Usage: .venv/bin/python validate_n4.py
"""

from __future__ import annotations

import asyncio

import numpy as np

from director.data.manifest import read_manifest
from director.fugu.inference import select_worker
from director.fugu.labels import generate_soft_targets
from director.fugu.run import build_router, load_config
from director.fugu.sft import train_sft
from director.shared.providers import build_pool
from director.shared.tasks import Dataset
from director.shared.transcript import raw_query
from director.shared.types import Sampling

cfg = load_config("validate.yaml")
wids = cfg.worker_ids
print(f"workers ({len(wids)}): {wids}")

items = read_manifest("manifests/validate")
split_of = {it.task_id: it.split for it in items}
ds = Dataset([it.to_task() for it in items], name="revalidate")

pool = build_pool(cfg.pool, cfg.workers)
sampling = Sampling(temperature=0.7, max_tokens=16384, seed=0)
labels = asyncio.run(generate_soft_targets(pool, ds, n_samples=4, tau=0.1, sampling=sampling))
print(f"re-labeled {len(labels)} items at n=4  |  spent ${pool.budget.spent_usd:.2f}\n")

# headroom on graded n=4 rewards (5 workers)
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
from collections import Counter
print("\n=== HELD-OUT ROUTING (n=4 de-noised) ===")
print(f"  held-out n={len(test)}")
print(f"  orchestrator : {orch.mean():.3f}")
print(f"  best worker  : {best:.3f} ({best_w})")
print(f"  oracle       : {oracle:.3f}")
print(f"  LIFT vs best : {orch.mean()-best:+.3f}  (captured {(orch.mean()-best)/(oracle-best+1e-9):.0%} of headroom)")
print(f"  routing dist : {dict(Counter(routed))}")
