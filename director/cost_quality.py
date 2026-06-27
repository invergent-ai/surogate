"""Cost-quality / escalation analysis of the pool screen, on the GLM-anchored frontier pool.

The product is cost-efficiency: route to cheap GLM by default, escalate to a frontier model only when
needed -> best-of-frontier quality per query at a fraction of the average price. This reads the per-rep
outcomes from the screen and answers:
  1. Per-worker quality vs price.
  2. GLM-suffices rate: on what fraction does GLM match the per-item oracle? (cross-fitted)
  3. Cheapest-adequate routing (the product's upper bound): if the router escalated PERFECTLY, what
     quality and average price? vs always-best-single. (cross-fitted to remove winner's curse)
Cost is a RELATIVE proxy = output $/Mtok (assumes comparable output lengths; we don't have per-call
tokens). Prices are user-provided ($/Mtok out).
"""
from __future__ import annotations

import json
import os

import numpy as np

OUT = os.getenv("OUT", "manifests/fugu_clean_v1/pool_matrix_frontier.jsonl")
PRICE = {"opus": 25.0, "gpt": 30.0, "gemini": 12.0, "glm": 4.0}  # $/Mtok out (user-provided)


def main():
    recs = [json.loads(l) for l in open(OUT) if l.strip()]
    wids = recs[0]["worker_ids"]
    price = np.array([PRICE[w] for w in wids])
    # rewards: list over items of (L, n) arrays
    R = [np.asarray(r.get("rewards") or [[v] for v in r["r_bar"]], dtype=float) for r in recs]
    I, L, n = len(R), len(wids), R[0].shape[1]
    P = np.stack([r.mean(1) for r in R])  # (I, L) per-item per-worker success rate
    print(f"=== COST-QUALITY (items={I}, n={n}, workers={wids}) ===")
    print("per-worker:  " + "  ".join(f"{w}: q={P[:,j].mean():.3f} ${PRICE[w]:g}" for j, w in enumerate(wids)))

    bi = int(P.mean(0).argmax())
    best_q, best_w, best_price = P.mean(0).max(), wids[bi], PRICE[wids[bi]]
    oracle_q = P.max(1).mean()
    print(f"best-single: {best_w} q={best_q:.3f} @ ${best_price:g}   |   per-item oracle q={oracle_q:.3f}")

    glm = wids.index("glm") if "glm" in wids else None
    if glm is not None:
        # GLM-suffices: GLM reaches the per-item oracle (cross-fitted: decide on half, no quality bias here
        # since "reaches oracle" is about GLM vs others; use full P for the rate, it's a descriptive stat)
        suffices = (P[:, glm] >= P.max(1) - 1e-9).mean()
        glm_solo_q = P[:, glm].mean()
        print(f"GLM: solo q={glm_solo_q:.3f} @ $4  | GLM-suffices (matches per-item oracle) on {suffices:.0%} of items")

    # Cheapest-adequate routing, CROSS-FITTED: split reps A/B; on A pick the cheapest worker that reaches
    # the A-oracle; score it on B. Swap. Avoids winner's-curse inflation of both quality and the savings.
    h = n // 2
    def cheapest_adequate(decide, evalh):  # decide,(I,L) rates ; evalh,(I,L) rates
        q, c = [], []
        for i in range(I):
            adeq = np.where(decide[i] >= decide[i].max() - 1e-9)[0]  # workers tied for best on decide-half
            w = adeq[np.argmin(price[adeq])]                          # cheapest among them
            q.append(evalh[i, w]); c.append(price[w])
        return float(np.mean(q)), float(np.mean(c))
    A = np.stack([r[:, :h].mean(1) for r in R]); B = np.stack([r[:, h:].mean(1) for r in R])
    qAB, cAB = cheapest_adequate(A, B); qBA, cBA = cheapest_adequate(B, A)
    ca_q, ca_c = (qAB + qBA) / 2, (cAB + cBA) / 2
    # bootstrap CI over items
    rng = np.random.default_rng(0); bq, bc = [], []
    for _ in range(1000):
        idx = rng.integers(0, I, I)
        Rs = [R[k] for k in idx]
        As = np.stack([r[:, :h].mean(1) for r in Rs]); Bs = np.stack([r[:, h:].mean(1) for r in Rs])
        q1, c1 = cheapest_adequate(As, Bs); q2, c2 = cheapest_adequate(Bs, As)
        bq.append((q1 + q2) / 2); bc.append((c1 + c2) / 2)
    qlo, qhi = np.quantile(bq, [.025, .975]); clo, chi = np.quantile(bc, [.025, .975])

    print(f"\ncheapest-adequate routing (cross-fitted, PERFECT escalation = product ceiling):")
    print(f"  quality={ca_q:.3f}  95%CI[{qlo:.3f},{qhi:.3f}]   (vs best-single {best_q:.3f}, oracle {oracle_q:.3f})")
    print(f"  avg price=${ca_c:.1f}/Mtok  95%CI[${clo:.1f},${chi:.1f}]   (vs always-{best_w} ${best_price:g})")
    if best_price > 0:
        print(f"  => {best_price/ca_c:.1f}x cheaper than always-best-single, at quality {ca_q:.3f} vs {best_q:.3f}")
    print("\nNOTE: this is the PERFECT-escalation ceiling. A real router only captures part of it; the gap")
    print("between cheapest-adequate quality and best-single tells you how much escalation accuracy matters.")


if __name__ == "__main__":
    main()
