"""LEARNABILITY 2a-lite on lane T's engine-op trainer (agent-EN 18:4x; lines pre-registered by lane L in LEARNABILITY.md,
last section; config agreed with agent-EN-L 18:4x):

  data      lanes/L/rig/data_2a.pt: "train" 256 / "heldout" 64 windows {"ids" [<= 1024], "tgt_pos" [k]}; position p
            predicts ids[p + 1]
  model     expert LoRA r16 a32 on w13 / w2 of all 24 layers, A = PEFT init (rig.make_lora order, seed 0), B = 0
  train     fp32 AdamW (0.9, 0.999, 1e-8, wd 0) on host masters, batch 32 windows (run_2a's shuffle), token-mean NLL
            over the batch's target positions; forward = the ENGINE's (bank fold of the current tensors), backward =
            lane T's long-row LEVEL-0 path (S2)
  evaluate  steps 0 / 5 / ... / 25: the engine's W4A4 free-routing forward + head log-prob with the adapter (ONE
            evaluator throughout), per-window NLL sums / counts of all 256 + 64 windows
  output    run_2a's JSON format (lanes/L/rig/analyze_2a.py) + the step-1 gradient of batch 1 (per-tensor norms per
            layer, 16-expert slices of layers 0 / 5 / 13 / 20 / 23) for lane L's cosine vs the rig
Usage: python l2a_lite.py --bin B --artifact A --data data_2a.pt --lr 5e-5 --out DIR [--steps 25] [--eval-every 5]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ift_engine import InklingEngine  # noqa: E402
import ift_backward as B  # noqa: E402
import ift_forward as F  # noqa: E402
import ift_long as LG  # noqa: E402

SLICE_LAYERS = (0, 5, 13, 20, 23)


def log(msg):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    print(f"[T {datetime.now(ZoneInfo('Europe/Bucharest')).strftime('%H:%M:%S')}] {msg}", flush=True)


class HostAdamW:
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


@torch.no_grad()
def evaluate(eng, banks, data):
    ad = [L["adapters"] for L in banks.layers]
    out = {}
    for split in ("train", "heldout"):
        sums, cnts = [], []
        for s in data[split]:
            ids = s["ids"].to(torch.int32).to(eng.dev).contiguous()
            res = F.forward(eng, ids, save=False, adapters_per_layer=ad)
            lp = F.logprobs(eng, res["hidden"], ids).double()
            p = s["tgt_pos"].long().to(eng.dev)
            sums.append(float(-lp[p].sum()))
            cnts.append(int(p.numel()))
        out[split] = {"nll_sum": sums, "count": cnts, "nll": sum(sums) / sum(cnts)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = torch.load(a.data, weights_only=False)
    eng = InklingEngine(a.bin, a.artifact, 0)
    W = B.EngineWeights(eng)
    lora = B.make_lora(eng.dev, seed=a.seed, b_std=0.0)
    params = [lo[k] for lo in lora for k in B.LORA_KEYS]
    banks = B.EngineBanks(lora, eng.dev)
    tf = LG.TrainForwardLong(eng, W)
    opt = HostAdamW(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    out_f = os.path.join(a.out, f"run_2a_lr{a.lr:g}.json")
    rec = {"args": vars(a) | {"trainer": "lane T engine-op (forward = engine, LEVEL-0 S2 backward)",
                              "chunk": LG.CHUNK, "moe_gm": LG.MOE_GM, "fast_gemm": LG.FAST_GEMM, "tf32": LG.TF32},
           "evals": {}, "steps": []}
    rng = random.Random(a.seed)
    perm = []

    def save():
        json.dump(rec, open(out_f + ".tmp", "w"))
        os.replace(out_f + ".tmp", out_f)

    t0 = time.time()
    rec["evals"]["0"] = evaluate(eng, banks, data)
    log(f"step 0 eval: train {rec['evals']['0']['train']['nll']:.4f} heldout {rec['evals']['0']['heldout']['nll']:.4f} "
        f"({time.time() - t0:.0f}s)")
    save()
    for step in range(1, a.steps + 1):
        t0 = time.time()
        if len(perm) < a.batch:
            idx = list(range(len(data["train"])))
            rng.shuffle(idx)
            perm += idx
        bidx, perm = perm[:a.batch], perm[a.batch:]
        seqs = [data["train"][i] for i in bidx]
        n_tot = sum(len(s["tgt_pos"]) for s in seqs)
        loss_sum = 0.0
        tokens = 0
        for s in seqs:
            ids = s["ids"].to(torch.int32).to(eng.dev).contiguous()
            lp, _ = tf.build(ids, lora, banks)
            p = s["tgt_pos"].long().to(eng.dev)
            loss = -lp[p].sum() / n_tot
            loss.backward()
            loss_sum += float(loss.detach())
            tokens += ids.numel()
            del lp, loss
        if step == 1:  # the step-1 gradient of batch 1, for lane L's cosine vs the rig
            torch.save({"bidx": bidx, "norms": [{k: float(lo[k].grad.norm()) for k in B.LORA_KEYS} for lo in lora],
                        "slices": {i: {k: lora[i][k].grad[:16].detach().cpu() for k in B.LORA_KEYS}
                                   for i in SLICE_LAYERS}}, os.path.join(a.out, "step1_grad.pt"))
        gnorm = float(torch.sqrt(sum((p.grad.double() ** 2).sum() for p in params if p.grad is not None)))
        opt.step()
        opt.zero_grad()
        banks.refresh(lora)
        dt = time.time() - t0
        rec["steps"].append({"step": step, "trainer_loss": loss_sum, "grad_norm": gnorm, "seconds": dt,
                             "tokens": tokens, "tok_per_s": tokens / dt})
        log(f"step {step}: trainer loss {loss_sum:.4f} grad norm {gnorm:.3e} ({dt:.0f}s, {tokens / dt:.0f} tok/s)")
        if step % a.eval_every == 0 or step == a.steps:
            t1 = time.time()
            rec["evals"][str(step)] = evaluate(eng, banks, data)
            log(f"step {step} eval: train {rec['evals'][str(step)]['train']['nll']:.4f} heldout "
                f"{rec['evals'][str(step)]['heldout']['nll']:.4f} ({time.time() - t1:.0f}s)")
        save()
    rec["done"] = True
    save()
    eng.close()


if __name__ == "__main__":
    main()
