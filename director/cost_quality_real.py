"""Re-run cost-quality with REAL per-call costs (from calibration) instead of flat $/Mtok.
Calibration (single-step, high reasoning) measured avg $/call: opus $0.0100 (out 330 tok, terse),
glm $0.0103 (out 2614), gemini $0.0477 (out 3933, verbose). So per-CALL cost ranks gemini >> glm ~ opus,
which inverts the sticker $/Mtok ladder. Recompute cheapest-adequate routing with these real costs."""
from __future__ import annotations

import json
import os
import numpy as np

OUT = "manifests/fugu_clean_v1/pool_matrix_frontier.jsonl"
CALL = {"opus": 0.0100, "glm": 0.0103, "gemini": 0.0477}  # measured $/call (calibration)
MTOK = {"opus": 25.0, "glm": 4.0, "gemini": 12.0}          # old sticker assumption


def main():
    recs = [json.loads(l) for l in open(OUT) if l.strip()]
    wids = recs[0]["worker_ids"]  # screen used 4-pool incl gpt; keep only the locked 3
    keep = [w for w in wids if w in CALL]
    ji = [wids.index(w) for w in keep]
    R = [np.asarray(r["rewards"], dtype=float)[ji] for r in recs]  # per item (3, n)
    P = np.stack([r.mean(1) for r in R])  # (I,3) success rate
    I = len(R); n = R[0].shape[1]
    call = np.array([CALL[w] for w in keep])
    print(f"=== COST-QUALITY with REAL per-call costs (items={I}, n={n}, pool={keep}) ===")
    print("per-worker:  " + "  ".join(f"{w}: q={P[:,j].mean():.3f} ${CALL[w]:.4f}/call" for j, w in enumerate(keep)))

    bi = int(P.mean(0).argmax()); best_w = keep[bi]; best_q = P.mean(0).max()
    print(f"best-single: {best_w} q={best_q:.3f} @ ${CALL[best_w]:.4f}/call   | per-item oracle q={P.max(1).mean():.3f}")
    for w in keep:
        j = keep.index(w)
        print(f"  always-{w}: q={P[:,j].mean():.3f} @ ${CALL[w]:.4f}/call")

    # cheapest-adequate by per-CALL cost, cross-fitted (split reps, decide on A eval on B, swap)
    h = n // 2
    def ca(decide, evalh, costs):
        q, c = [], []
        for i in range(I):
            a = np.where(decide[i] >= decide[i].max() - 1e-9)[0]
            w = a[np.argmin(costs[a])]
            q.append(evalh[i, w]); c.append(costs[w])
        return float(np.mean(q)), float(np.mean(c))
    A = np.stack([r[:, :h].mean(1) for r in R]); B = np.stack([r[:, h:].mean(1) for r in R])
    for label, costs in [("REAL per-call", call), ("OLD $/Mtok (flat)", np.array([MTOK[w] for w in keep]))]:
        q1, c1 = ca(A, B, costs); q2, c2 = ca(B, A, costs)
        unit = "/call" if "call" in label else "/Mtok"
        print(f"\ncheapest-adequate [{label}]: q={(q1+q2)/2:.3f}  cost={ (c1+c2)/2:.4f}{unit}")

    # per-domain best (quality) for context
    dom = {}
    for r in recs:
        dom.setdefault(r["domain"], []).append(np.asarray(r["rewards"], dtype=float)[ji].mean(1))
    print("\nper-domain best worker (quality):")
    for d, rows in dom.items():
        m = np.array(rows).mean(0); print(f"  {d:8} best={keep[int(m.argmax())]}  " + " ".join(f"{w}:{m[j]:.2f}" for j, w in enumerate(keep)))


if __name__ == "__main__":
    main()
