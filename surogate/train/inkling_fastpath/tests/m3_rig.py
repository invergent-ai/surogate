"""M3, phase B: the validated learnability rig (rig.py, W4A16 trainer mode) computes the LoRA gradient of
the same sequence with the same masters (rig.make_lora seed 7, b_std 1e-3) and the ENGINE's routing replayed (from
phase A's m3_fast.pt); compared with the fast route's LEVEL-0 gradient.
PASS iff every slice tensor's cosine >= 0.99 (the M3 pass line), the gradients are finite and the losses agree
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
def cosines(slices):
    out_ = {}
    for i, sl in slices.items():
        for k, g in sl.items():
            a = lora[i][k].grad[:16].flatten().double().cpu()
            b = g.flatten().double()
            out_[f"L{i}.{k}"] = float(a @ b / (a.norm() * b.norm()).clamp_min(1e-300))
    return out_


cos = cosines(fast["slices"])
cos_d1 = cosines(fast["slices_d1"]) if "slices_d1" in fast else {}
nrel = []
for i, L in enumerate(lora):
    for k in ("A13", "B13", "A2", "B2"):
        r = float(L[k].grad.norm())
        f = fast["norms"][i][k]
        nrel.append(abs(r - f) / max(r, 1e-30))
nrel_t = torch.tensor(nrel)
finite = all(bool(torch.isfinite(L[k].grad).all()) for L in lora for k in ("A13", "B13", "A2", "B2"))
res = {"T": T, "loss_rig_w4a16": float(loss.detach()), "loss_fast_engine": fast["loss"],
       "loss_fast_torch_chain": fast.get("loss_d1"), "cos_min": min(cos.values()), "cos": cos,
       "cos_d1_min": min(cos_d1.values()) if cos_d1 else None, "cos_d1": cos_d1, "norm_rel_err_max": float(nrel_t.max()), "norm_rel_err_median": float(nrel_t.median()),
       "rig_finite": finite, "rig_seconds": t_rig, "load_s": t1 - t0}
ok = res["cos_min"] >= 0.99 and finite and abs(res["loss_rig_w4a16"] - fast["loss"]) / fast["loss"] <= 0.02
res["glue_d1_pass"] = bool(cos_d1) and res["cos_d1_min"] >= 0.99
res["M3_grad_pass"] = ok
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("cos", "cos_d1")}), flush=True)
print("cos engine-LEVEL0 vs rig: " + " ".join(f"{k}:{v:.3f}" for k, v in cos.items()), flush=True)
print("cos chain-LEVEL0 vs rig: " + " ".join(f"{k}:{v:.3f}" for k, v in cos_d1.items()), flush=True)
print("M3 GRAD PASS" if ok else "M3 GRAD FAIL", flush=True)
