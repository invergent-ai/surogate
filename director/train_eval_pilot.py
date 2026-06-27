"""Train the Stage-1 SFT warm-start router on a curated manifest, then evaluate internal-val routing
lift. Produces the recipe's AFTER-SFT acceptance gates + the learning curve (lift vs #curated) so we
can size the full run from where lift plateaus.

Reads  MANIFEST_DIR/labels_n4_tau0.1.jsonl (soft targets) + internal_val.jsonl (held-out n=1 rewards).
Appends an "After-SFT" section to MANIFEST_DIR/data_report.md and saves sft_router.pt.
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter

import numpy as np

MANIFEST_DIR = os.getenv("MANIFEST_DIR", "manifests/fugu_clean_v1")
CHEAP_WORKER = "deepseek_flash"
FAST = os.getenv("CURVE", "1") == "0"  # CURVE=0 -> single training, no learning curve
CURVE_FRACS = (1.0,) if FAST else (0.25, 0.5, 0.75, 1.0)
EPOCHS = int(os.getenv("EPOCHS", "40" if FAST else "120"))  # override for a full-convergence single run
LR = float(os.getenv("LR", "1e-6"))        # Trinity: lr 1e-6 (our old 0.02 collapsed the head onto 1 worker)
BATCH = int(os.getenv("BATCH", "64"))      # Trinity: effective batch 64 (via grad-accum micro-batches)
MICRO = int(os.getenv("MICRO", "8"))       # memory-safe micro-batch (batch 64 at ctx 4096 OOMs otherwise)
LOG_EVERY = int(os.getenv("LOG_EVERY", "1"))  # per-epoch KL + ETA so training is observable
TRAIN_FRAC = float(os.getenv("TRAIN_FRAC", "1.0"))  # subset train labels (fast lr sweeps); val unchanged


def _load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def main():
    from director.config import DirectorConfig, FeaturizerConfig, default_frontier_pool
    from director.fugu.inference import select_worker
    from director.fugu.labels import SoftLabel
    from director.fugu.model import save_router
    from director.fugu.run import build_router
    from director.fugu.sft import train_sft
    from director.shared.transcript import raw_query

    cfg = DirectorConfig(workers=default_frontier_pool(), featurizer=FeaturizerConfig(context_window=4096))
    wids = cfg.worker_ids
    idx = {w: j for j, w in enumerate(wids)}

    labels_raw = _load_jsonl(os.path.join(MANIFEST_DIR, "labels_n4_tau0.1.jsonl"))
    val = _load_jsonl(os.path.join(MANIFEST_DIR, "internal_val.jsonl"))
    if not labels_raw or not val:
        print(f"need labels_n4_tau0.1.jsonl + internal_val.jsonl in {MANIFEST_DIR}", flush=True)
        return
    labels = [SoftLabel(task_id=r["task_id"], prompt=r["prompt"], worker_ids=r["worker_ids"],
                        r_bar=r["r_bar"], p=r["p"], grader=r["grader"]) for r in labels_raw]
    R = np.array([[v["rewards_n1"][j] for j in range(len(wids))] for v in val], dtype=float)
    ar = np.arange(len(val))
    best = float(R.mean(0).max())          # best single worker on val
    oracle = float(R.max(1).mean())        # per-item best-in-pool
    print(f"labels={len(labels)} val={len(val)} | best-single={best:.3f} oracle={oracle:.3f} "
          f"(headroom {oracle-best:+.3f})", flush=True)

    def eval_router(router):
        routed = [idx[select_worker(router, raw_query(v["prompt"]))] for v in val]
        orch = float(R[ar, routed].mean())
        cap = (orch - best) / (oracle - best) if oracle > best else 0.0
        return orch, cap, routed

    # learning curve: train on increasing fractions. Build the backbone ONCE and reset only the
    # trainable head+SVF between fractions (snapshot the init vector, reload it) — avoids reloading
    # the 0.6B per fraction. CURVE_FRACS ends at 1.0, so the router is left trained on the full set.
    shuffled = labels[:]
    random.Random(0).shuffle(shuffled)
    if TRAIN_FRAC < 1.0:
        shuffled = shuffled[:max(8, int(len(shuffled) * TRAIN_FRAC))]
        print(f"TRAIN_FRAC={TRAIN_FRAC}: training on {len(shuffled)} labels", flush=True)
    router = build_router(cfg)
    init_vec = router.trainable_vector().detach().clone()
    curve, final_routed = [], None
    for frac in CURVE_FRACS:
        k = max(8, int(round(len(shuffled) * frac)))
        router.load_vector(init_vec)  # reset to init — no backbone reload
        train_sft(router, shuffled[:k], epochs=EPOCHS, lr=LR, batch_size=BATCH, micro_batch=MICRO,
                  log_every=LOG_EVERY)
        orch, cap, routed = eval_router(router)
        curve.append((k, orch, cap))
        print(f"  [{frac:.0%}] k={k:4} orch={orch:.3f} lift={orch-best:+.3f} capture={cap:.0%}", flush=True)
        final_routed = routed

    save_router(router, os.path.join(MANIFEST_DIR, "sft_router.pt"), worker_ids=wids)
    orch, cap, _ = curve[-1][1], curve[-1][2], None
    dist = Counter(wids[i] for i in final_routed)
    top_share = max(dist.values()) / max(len(final_routed), 1)
    cheap_share = dist.get(CHEAP_WORKER, 0) / max(len(final_routed), 1)

    gates = {
        "router lift > 0 (orch > best)": orch > best,
        "captures >= 25% of headroom": cap >= 0.25,
        "no collapse (top worker < 80%)": top_share < 0.80,
    }
    lines = [
        "", "## After-SFT (internal validation)",
        f"- n_val={len(val)} | orch={orch:.3f} vs best-single={best:.3f} (lift {orch-best:+.3f}) | "
        f"oracle={oracle:.3f} | headroom captured **{cap:.0%}**",
        "- routing distribution: " + "  ".join(f"{w}={dist.get(w,0)}" for w in wids)
        + f"  (cheap {CHEAP_WORKER}={cheap_share:.0%})",
        "- learning curve (k → lift / capture): "
        + "  ".join(f"{k}:{o-best:+.2f}/{c:.0%}" for k, o, c in curve),
        "", "### after-SFT gates",
        *[f"- [{'PASS' if ok else 'FAIL'}] {name}" for name, ok in gates.items()],
    ]
    with open(os.path.join(MANIFEST_DIR, "data_report.md"), "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"\nAFTER-SFT GATES: {'ALL PASS' if all(gates.values()) else 'SOME FAILED'} "
          f"— learning curve shows where lift plateaus (size the full run there).", flush=True)


if __name__ == "__main__":
    main()
