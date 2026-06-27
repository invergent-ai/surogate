"""One-off eval on the CURRENT bank (decoupled from the loop's slow step gating). GPU 1."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json, math
from collections import defaultdict
import numpy as np
from director.config import DirectorConfig, FeaturizerConfig, default_frontier_pool
from director.fugu.run import build_router
from director.fugu.labels import SoftLabel
from director.fugu.sft import train_sft
from director.fugu.inference import select_worker
from director.shared.transcript import raw_query

TAU = 0.1
def softmax(xs):
    m = max(xs); e = [math.exp((x - m) / TAU) for x in xs]; z = sum(e); return [v / z for v in e]

cfg = DirectorConfig(workers=default_frontier_pool(), featurizer=FeaturizerConfig(context_window=8192))
wids = cfg.worker_ids
items = {}
for l in open("production_bank.jsonl"):
    if not l.strip(): continue
    c = json.loads(l)
    if "worker" not in c: continue
    d = items.setdefault((c["domain"], c["item_id"]), {"prompt": c.get("prompt", ""), "rewards": {}, "costs": {}})
    d["rewards"][c["worker"]] = c["reward"]; d["costs"][c["worker"]] = c["cost"]
recs = [{"domain": k[0], "item_id": k[1], "prompt": v["prompt"], "rewards": v["rewards"]}
        for k, v in items.items() if all(w in v["rewards"] for w in wids)]
by_dom = defaultdict(list)
for r in recs: by_dom[r["domain"]].append(r)
print(f"complete items: {len(recs)} | per-domain:", {d: len(v) for d, v in by_dom.items()}, flush=True)
train, test = [], []
for dom, its in by_dom.items():
    its = sorted(its, key=lambda r: r["item_id"]); cut = max(1, int(0.8 * len(its))); train += its[:cut]; test += its[cut:]
labels = [SoftLabel(r["item_id"], r["prompt"], wids, [r["rewards"][w] for w in wids],
                    softmax([r["rewards"][w] for w in wids]), "g") for r in train]
router = build_router(cfg)
train_sft(router, labels, epochs=120, lr=0.02, batch_size=8, log_every=0)
idx = {w: j for j, w in enumerate(wids)}
def ev(rows):
    R = np.array([[r["rewards"][w] for w in wids] for r in rows], dtype=float)
    routed = [idx[select_worker(router, raw_query(r["prompt"]))] for r in rows]
    ar = np.arange(len(rows)); bj = int(R.mean(0).argmax())
    return R[ar, routed].mean(), R[:, bj].mean(), wids[bj], R.max(1).mean()
o, b, bw, orc = ev(test)
print(f"\nOVERALL (train={len(train)} test={len(test)}): resolve_lift={o-b:+.3f} "
      f"(orch {o:.3f} vs best {b:.3f} [{bw}], oracle {orc:.3f}, headroom {orc-b:+.3f})", flush=True)
for dom in sorted(by_dom):
    dr = [r for r in test if r["domain"] == dom]
    if dr:
        oo, bb, _, oc = ev(dr)
        print(f"  {dom:12} n={len(dr):3} lift={oo-bb:+.3f} (orch {oo:.3f} vs best {bb:.3f}, oracle {oc:.3f})", flush=True)
