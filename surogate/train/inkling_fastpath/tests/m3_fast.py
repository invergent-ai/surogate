"""Lane T M3, phase A (the fast route): LEVEL-0 backward on one real sequence.

1. FORWARD AGREEMENT (mapping check): the torch recompute of a visit (lane L's math, W4A16, engine routing) from the
   engine's own visit inputs vs the engine's visit output (W4A4): relative L2 error per checked visit (a weight-mapping
   slip shows as O(1); act-quant noise is small).
2. GRADIENT: expert LoRA r16 a32 on all 24 layers, rig.make_lora(seed 7, b_std 1e-3) masters, engine banks folded from
   them, loss = mean NLL of the sequence's targets through the ENGINE's forward; backward = LEVEL 0. Finite check; saves
   the gradient slices lane L's R3 checks (layers 0 / 5 / 13 / 20 / 23, experts 0..15, A13 / B13 / A2 / B2), the
   per-layer norms, the engine's routing and the ids for phase B (m3_rig.py: lane L's rig on the same routing).
3. OVERFIT: 10 AdamW steps (lr 1e-3, fp32 masters + moments on the host) on the same sequence; the engine's loss must
   fall (banks refreshed from the masters before every forward).
Usage: python m3_fast.py --bin B --artifact A --prompt o00.json --tokens 1024 --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ift_engine import InklingEngine  # noqa: E402
import ift_backward as B  # noqa: E402

SLICE_LAYERS = (0, 5, 13, 20, 23)


def log(msg):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    print(f"[T {datetime.now(ZoneInfo('Europe/Bucharest')).strftime('%H:%M:%S')}] {msg}", flush=True)


class HostAdamW:
    """fp32 AdamW with masters and moments on the host (lane L's rig.CpuAdamW)."""

    def __init__(self, params, **kw):
        self.dev = list(params)
        self.host = [p.detach().to("cpu", copy=True).requires_grad_(True) for p in self.dev]
        self.opt = torch.optim.AdamW(self.host, **kw)

    def step(self):
        for pd, ph in zip(self.dev, self.host):
            ph.grad = None if pd.grad is None else pd.grad.detach().to("cpu")
        self.opt.step()
        with torch.no_grad():
            for pd, ph in zip(self.dev, self.host):
                pd.copy_(ph.detach().to(pd.device, non_blocking=True))

    def zero_grad(self):
        for pd in self.dev:
            pd.grad = None
        for ph in self.host:
            ph.grad = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rep = {}
    t0 = time.time()
    eng = InklingEngine(a.bin, a.artifact, 0)
    W = B.EngineWeights(eng)
    log(f"loaded {time.time() - t0:.1f}s; free {torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB")
    rec = json.load(open(a.prompt))
    ids = torch.tensor(rec["ids"][:a.tokens], dtype=torch.int32, device=eng.dev)
    T = ids.numel()
    lora = B.make_lora(eng.dev, seed=7, b_std=1e-3)
    params = [lo[k] for lo in lora for k in B.LORA_KEYS]
    banks = B.EngineBanks(lora, eng.dev)
    tf = B.TrainForward(eng, W)
    log(f"LoRA + banks; free {torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB")

    # 1 + 2: one forward / backward
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    lp, res = tf.build(ids, lora, banks)
    torch.cuda.synchronize()
    t_fwd = time.time() - t1
    agree = {}
    loss = -lp.mean()
    t2 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    t_bwd = time.time() - t2
    finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
    norms = [{k: float(lo[k].grad.norm()) for k in B.LORA_KEYS} for lo in lora]
    slices = {i: {k: lora[i][k].grad[:16].detach().cpu() for k in B.LORA_KEYS} for i in SLICE_LAYERS}
    rep["grad"] = {"T": T, "loss": float(loss), "finite": finite, "forward_s": t_fwd, "backward_s": t_bwd,
                   "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "norm_total": math.sqrt(sum(v * v for n in norms for v in n.values()))}
    log(f"grad: loss {float(loss):.5f} finite {finite} fwd {t_fwd:.2f}s bwd {t_bwd:.2f}s peak "
        f"{rep['grad']['peak_gib']:.2f} GiB")
    torch.save({"ids": ids.cpu(), "routing": res["routing"].cpu(), "loss": float(loss), "norms": norms,
                "slices": slices}, os.path.join(a.out, "m3_fast.pt"))
    del lp, res, loss

    # 1. forward agreement (separate pass: engine forward with the same banks, then torch visits from its inputs)
    with torch.no_grad():
        import ift_forward as F
        res = F.forward(eng, ids, save=True, adapters_per_layer=[L["adapters"] for L in banks.layers])
        slots = {0: res["x0"]}
        for v, vi in enumerate(eng.visits):
            if v in (0, 5, 6, 11, 18, 23, 29, 41, 42, 47):
                with torch.device(eng.dev):
                    stack = [slots[s] for s in vi.stack]
                    bank = (slots[vi.bank[0]], slots[vi.bank[1]]) if vi.looped else None
                    mine = B.visit_torch(W, vi, stack, slots[vi.partial], bank, res["routing"][v], lora[vi.layer])
                ref = res["outs"][v].float()
                rel = float((mine - ref).norm() / ref.norm())
                agree[v] = {"rel_l2": rel, "max_abs": float((mine - ref).abs().max())}
                log(f"forward agreement visit {v} (layer {vi.layer} pass {vi.pass_}): rel L2 {rel:.3e}")
            slots[vi.next_partial] = res["outs"][v]
        del res
    rep["forward_agreement"] = agree

    # 3. overfit
    opt = HostAdamW(params, lr=a.lr, weight_decay=0.0)
    losses = []
    for s in range(a.steps + 1):
        banks.refresh(lora)
        opt.zero_grad()
        t1 = time.time()
        lp, _ = tf.build(ids, lora, banks)
        loss = -lp.mean()
        if s < a.steps:
            loss.backward()
            opt.step()
        losses.append(float(loss))
        log(f"overfit step {s}: loss {float(loss):.5f} ({time.time() - t1:.1f}s)")
        del lp, loss
    rep["overfit"] = {"lr": a.lr, "losses": losses, "falls": losses[-1] < losses[0] - 0.05}
    rep["checks"] = {"M3_finite": finite, "M3_overfit_falls": rep["overfit"]["falls"],
                     "forward_agreement_max_rel": max(x["rel_l2"] for x in agree.values())}
    json.dump(rep, open(os.path.join(a.out, "m3_fast.json"), "w"), indent=1)
    log("CHECKS " + json.dumps(rep["checks"]))
    eng.close()


if __name__ == "__main__":
    main()
