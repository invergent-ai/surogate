"""Lane T M3, phase B: lane L's validated rig (lanes/L/rig/rig.py, W4A16 trainer mode) computes the LoRA gradient of
the same sequence with the same masters (rig.make_lora seed 7, b_std 1e-3) and the ENGINE's routing replayed (from
phase A's m3_fast.pt); compared with the fast route's LEVEL-0 gradient.
PASS iff every slice tensor's cosine >= 0.99 (agent-EN's M3 line), the gradients are finite and the losses agree
within 2 % (the fast route's loss is the engine's W4A4 one, the rig's is W4A16).
Usage: RIG_DEVICE=cuda python m3_rig.py <rig dir> <model dir> <m3_fast.pt> <out.json>
"""

import json
import os
import sys
import time

rig_dir, model_dir, fast_pt, out = sys.argv[1:5]
sys.path.insert(0, rig_dir)
import torch  # noqa: E402

import rig  # noqa: E402  (sets the default device)

t0 = time.time()
m = rig.RigModel(model_dir)
fast = torch.load(fast_pt, weights_only=False)
ids = fast["ids"].long().cuda()
T = ids.numel()
lora = rig.make_lora(seed=7, b_std=1e-3)
m.lora = lora
rs = fast["routing"].long().cuda()
t1 = time.time()
h, _ = m.hidden(ids[None], "train", routing={v: rs[v] for v in range(48)})
lpt = m.target_logprobs(h[:-1], ids[1:], grad=True)
loss = -lpt.mean()
loss.backward()
t_rig = time.time() - t1
cos = {}
for i, sl in fast["slices"].items():
    for k, g in sl.items():
        a = lora[i][k].grad[:16].flatten().double().cpu()
        b = g.flatten().double()
        cos[f"L{i}.{k}"] = float(a @ b / (a.norm() * b.norm()).clamp_min(1e-300))
nrel = []
for i, L in enumerate(lora):
    for k in ("A13", "B13", "A2", "B2"):
        r = float(L[k].grad.norm())
        f = fast["norms"][i][k]
        nrel.append(abs(r - f) / max(r, 1e-30))
nrel_t = torch.tensor(nrel)
finite = all(bool(torch.isfinite(L[k].grad).all()) for L in lora for k in ("A13", "B13", "A2", "B2"))
res = {"T": T, "loss_rig_w4a16": float(loss), "loss_fast_engine": fast["loss"], "cos_min": min(cos.values()),
       "cos": cos, "norm_rel_err_max": float(nrel_t.max()), "norm_rel_err_median": float(nrel_t.median()),
       "rig_finite": finite, "rig_seconds": t_rig, "load_s": t1 - t0}
ok = res["cos_min"] >= 0.99 and finite and abs(res["loss_rig_w4a16"] - fast["loss"]) / fast["loss"] <= 0.02
res["M3_grad_pass"] = ok
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "cos"}), flush=True)
print("M3 GRAD PASS" if ok else "M3 GRAD FAIL", flush=True)
