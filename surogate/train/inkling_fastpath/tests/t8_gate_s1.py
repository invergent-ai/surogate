"""Lane T8, S1 + S2 card gate (PLAN-SKYRL.md). PASS LINES, written before any run (also in lanes/T8/NOTES.md). No adapter is kept; nothing is trained beyond
two test optimizer steps on 4 rows whose result is discarded.
Expert adapters at SkyRL's shape (r16 / alpha 32 per expert, A AND B trained), A = the kaiming-uniform seed-0 masters, B = N(0, 1e-3) (seed 11) so that A has
a gradient. Real rows of 16,009 / 28,002 / 47,014 / 57,431 tokens (gate_rows.npz), full backward K = 48, plain head.
  G10 (1 card, 16k + 28k rows) the lean K = 48 gradient of ALL FOUR expert keys == the stock path (ift_long.LongModel) at the same non-zero adapter,
      every tensor cos >= 0.9999, rel err <= 1 %, finite (bitwise expected, reported); A13 / A2 gradients non-zero.
  G12 (2 cards, 4 rows, two groups of two, GRPO advantages of synthetic scores) one SkyRL update `ppo_batch` with 2 mini-batches, lr 2e-5:
      the stale gate (mini-batch 1 forward == the recomputed old log-probs, bitwise) and the replay gate on the old log-probs hold; mini-batch 2 runs at
      moved weights (some ratio != 1); all gradients finite; A and B masters both changed; host peak per worker reported (all four keys, K = 48, the 57k row).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import t8_gate as GT  # noqa: E402

GIB = 2 ** 30


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def nonzero_masters():
    import ift_backward as B

    lo = B.make_lora("cpu", seed=0, b_std=0.0)
    g = torch.Generator().manual_seed(11)
    out = []
    for d in lo:
        out.append({"A13": d["A13"].detach().clone(), "A2": d["A2"].detach().clone(),
                    "B13": torch.randn(d["B13"].shape, generator=g) * 1e-3, "B2": torch.randn(d["B2"].shape, generator=g) * 1e-3})
    return out


def phase_g10(a, rows):
    import ift_backward as B
    import ift_engine as E
    import ift_long as LG
    import t8_lean as TL
    import t8_step as TS

    stack = TS.verify_stack("A0")
    eng = E.InklingEngine(a.bin, a.artifact, 0)
    W = B.EngineWeights(eng)
    lora = B.make_lora(eng.dev, seed=0, b_std=0.0)
    M = nonzero_masters()
    with torch.no_grad():                                     # the work values the kit hands the engine: bf16(master)
        for lo, m in zip(lora, M):
            for k in B.LORA_KEYS:
                lo[k].copy_(m[k].to(torch.bfloat16).float().to(lo[k].device))
    banks = B.EngineBanks(lora, eng.dev)
    adapters = [L["adapters"] for L in banks.layers]
    tf = LG.TrainForwardLong(eng, W)
    params = [lo[k] for lo in lora for k in B.LORA_KEYS]
    host = [{k: lo[k].detach().cpu() for k in B.LORA_KEYS} for lo in lora]
    rep = {"rows": [], "stack": stack}
    for r in rows[:2]:
        ids = torch.as_tensor(np.ascontiguousarray(r.ids), dtype=torch.int32, device=eng.dev)
        g = torch.as_tensor(r.loss_mask[1:].astype(np.float32), device=eng.dev)
        res = LG.forward_host(eng, ids, adapters)
        lp = LG.LongModel.apply(tf, ids, res, lora[0]["scale"], *params)
        (g * lp).sum().backward()
        stock = {}
        for i, p in enumerate(params):
            if p.grad is not None:
                stock[(i // 4, i % 4)] = p.grad.detach().cpu()
                p.grad = None
        del lp, res
        LG._PINNED.clear()
        torch.cuda.empty_cache()
        torch._C._host_emptyCache()
        buf = TL.RowBuffers(eng.visits, 0, r.T)
        t0 = time.time()
        v = TL.forward_keep(eng, ids, adapters, buf)
        got = {}

        def on(vis, layer, j, gj):
            c = gj.float().cpu()
            got[(layer, j)] = c if (layer, j) not in got else got[(layer, j)].add_(c)

        TL.backward_row(eng, W, ids, v, host, lora[0]["scale"], 0, g, buf, on, q_block=TL.qb_for(r.T))
        sec = time.time() - t0
        c = GT.compare(got, stock)
        a_nonzero = all(float(got[(L, j)].abs().max()) > 0 for L in range(24) for j in (0, 2) if (L, j) in got) and all((L, j) in got for L in range(24) for j in range(4))
        c.update({"T": r.T, "a_grads_nonzero": bool(a_nonzero), "tok_per_s": r.T / sec})
        rep["rows"].append(c)
        log(f"G10 T {r.T}: {c}")
        del v, got, stock, ids, g
        buf.close()
        torch.cuda.empty_cache()
    rep["pass"] = all(x["pass"] and x["a_grads_nonzero"] for x in rep["rows"])
    eng.close()
    return rep


def phase_g12(a, rows):
    from surogate.inkling_rl.step import StepConfig
    import t8_skyrl as KS
    import t8_step as TS

    cards = tuple(int(x) for x in a.cards.split(","))
    tr = TS.T8Trainer(TS.T8Config(bin_dir=a.bin, artifact=a.artifact, cards=cards, K=48, head="plain", max_row_tokens=58010, segment_keys=(),
                                  train_keys=("A13", "B13", "A2", "B2"), run_dir=a.run_dir))
    rep = {}
    try:
        tr.set_masters(nonzero_masters())
        before = [m.detach().clone() for m in tr.opt.masters]
        for k, r in enumerate(rows):
            r.group_idx = k // 2
        adv = KS.grpo_advantages([1.0, 0.0, 0.0, 1.0], [r.group_idx for r in rows])
        for r, x in zip(rows, adv):
            r.advantage = float(x)
        # the rows carry the rollout's log-probs at the ZERO adapter; this gate's adapter is not zero, so the rollout log-probs are replaced by the
        # trainer's own at this adapter (the replay gate then proves old log-probs == the forward that the stale gate reads)
        lp0 = tr.logprobs(rows)
        for r, x in zip(rows, lp0):
            e = np.zeros(r.T, dtype=np.float64)
            e[1:] = x.astype(np.float64)
            r.engine_lp = e * r.loss_mask
        res = TS.ppo_batch(tr, rows, 0, 2e-5, 2, StepConfig(max_policy_lag=None))
        rep["minibatches"] = []
        for o, n in res:
            fin = all(bool(torch.isfinite(g).all()) for g in o.grads)
            rep["minibatches"].append({"grad_norm": n, "finite": fin, "rows": [x["T"] for x in o.rows], "clip_ratio": o.metrics.get("ppo_clip_ratio"),
                                       "ratio_max": o.metrics.get("ppo_ratio_max"), "abs_diff_mean": o.metrics.get("rollout_train_logprobs_abs_diff_mean"),
                                       "workers": o.metrics["workers"], "tok_per_s_per_card": o.metrics["tok_per_s_per_card"]})
            log(f"G12 mini-batch: {rep['minibatches'][-1]}")
        after = tr.opt.masters
        changed = {k: any(float((after[L * 4 + j] - before[L * 4 + j]).abs().max()) > 0 for L in range(24)) for j, k in enumerate(TS.LORA_KEYS)}
        rep["masters_changed"] = changed
        rep["host_peak_per_worker_gib"] = max(w["VmHWM_gib"] for m in rep["minibatches"] for w in m["workers"])
        mb2 = rep["minibatches"][1]
        rep["pass"] = bool(all(m["finite"] for m in rep["minibatches"]) and all(changed.values()) and mb2["ratio_max"] is not None and mb2["ratio_max"] != 1.0)
    finally:
        tr.close()
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("g10", "g12"))
    ap.add_argument("--bin", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--cards", default="0")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = GT.gate_rows(a.rows)
    rep = (phase_g10 if a.phase == "g10" else phase_g12)(a, rows)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    print(f"T8 GATE S1 {a.phase}: {'PASS' if rep['pass'] else 'FAIL'}", flush=True)
    return 0 if rep["pass"] else 4


if __name__ == "__main__":
    sys.exit(main())
