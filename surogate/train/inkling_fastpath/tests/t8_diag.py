"""Diagnostic of the micro-step's follow failure on ONE real plan row (one card, in-process worker; nothing trained, nothing kept as a model).
PASS LINES, written before any data (2026-11-24, with the coordinator's additions):
  Row: plan row 3 (11,031 tokens, 176 loss tokens). S = sum over the loss tokens of log pi. Start point theta0 = the micro-step's (A = the
  trainer's init, B = 0). Six tensors: qB and B13 of physical layer 3 (before the loop), 12 (looped, 3 passes summed), 21 (after, the control).
  G_X = -N x the PPO gradient at advantage 1 (= grad S), the micro-step's path; variants: A production; B chunk 512 + halved query block;
  C the recompute's bf16 STE rounding off; D the router's gate weights detached in the recompute (no gradient through routing weights).
  Directions per tensor: "s" = sign(G_A) x e, "g" = G_A / |G_A| x e sqrt(n) (same L2 norm); e on the ladder 2.4e-4 x {1/4, 1, 4, 16, 64, 256}.
  follow f = [S(theta0 + step) - S(theta0 - step)] / (2 <G_A, step>), measured with
    free   = the engine's own forward (bitwise the engine; its own top-k routing at each point; flips counted per visit),
    frozen = the trainer's fp32 recompute run as a forward (visit_torch_t8 chained) with theta0's routing, STE rounding on,
    smooth = the same with the rounding off (the function the backward differentiates when the rounding is treated as identity).
  T-a "G right on this row": free f(s) and f(g) in [0.7, 1.3] for all six tensors at the micro-step's e (x1) and the next rung.
      "row-level mismatch, depth-dependent": layer 21 in [0.7, 1.3] while layer 12 or 3 < 0.3.  "cannot tell": free f unstable (two
      consecutive rungs differ by > 20 %) at every rung for that tensor.
  T-b "accumulation noise": |G_A - G_B| >= |G_A| (1 - f) for a layer-12 / layer-3 tensor.  "deterministic, from stored values / STE":
      cos(G_A, G_B) >= 0.999 and G_C's free follow >= 2x G_A's.  "deterministic, elsewhere": A, B, C agree and f stays low.
  Erasure (coordinator): free f rises by >= 3x between the smallest and the largest stable rung AND smooth f in [0.7, 1.3].
  Routing (coordinator): frozen f in [0.7, 1.3] while free f < 0.3 at every rung.
Phases: grad (one variant per process: env T8D_VARIANT = A | B | C | D) and fd (one process, a deadline; tasks in priority order, results written
after every task, so a stop at the deadline keeps what was measured)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import t8_step as TS  # noqa: E402

os.environ.update({**TS.STACKS["A0"], **TS.WORKER_ENV})      # before ift_long is imported: the stack the gates proved
import torch  # noqa: E402

import ift_backward as B  # noqa: E402
import ift_forward as F  # noqa: E402
import ift_long as LG  # noqa: E402
import ift_ref as RA  # noqa: E402
import t8_lean as TL  # noqa: E402
import t8_micro as MI  # noqa: E402
import t8_visit as TV  # noqa: E402
import t8_worker as TW  # noqa: E402
from surogate.inkling_rl import adapter as AD  # noqa: E402
from surogate.inkling_rl.step import StepConfig  # noqa: E402

ROW = 3
TENSORS = (("12", "qB"), ("12", "B13"), ("3", "qB"), ("3", "B13"), ("21", "qB"), ("21", "B13"))
BASE_E = 2.4e-4
LADDER = (0.25, 1.0, 4.0, 16.0, 64.0, 256.0)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bf16(t: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(AD.round_bf16(t.detach().cpu().numpy().astype(np.float32)).reshape(tuple(t.shape)))


class Rig:
    """The micro-step's worker in this process (CUDA_VISIBLE_DEVICES = the card), at theta0."""

    def __init__(self, a):
        self.cfg = TS.T8Config(bin_dir=a.bin, artifact=a.artifact, cards=(0,), K=48, head="plain", attn_rank=64, attn_alpha=32.0,
                               train_keys=TS.param_keys(64), segment_keys=(), run_dir=a.run_dir, max_row_tokens=58010)
        cfg = self.cfg
        self.shapes = TW.param_shapes(64)
        self.keys = tuple(self.shapes[0])
        n_all = sum(int(np.prod(s[k])) for s in self.shapes for k in self.keys)
        self.work_flat = torch.zeros(n_all, dtype=torch.float32)
        self.work = TW.work_views(self.work_flat, self.shapes)
        # theta0 exactly as T8Trainer builds it (masters), then work = bf16(master) (MasterAdamW.sync_work)
        init = B.make_lora("cpu", seed=cfg.lora_seed, b_std=0.0)
        self.theta0 = []
        g = torch.Generator().manual_seed(int(cfg.lora_seed) + 1000)
        for lo in init:
            d = {k: lo[k].detach().clone().float() for k in TS.LORA_KEYS}
            for s in TS.ATTN_SLICES:
                n, k = TW.SLICE_DIMS[s]
                d[f"{s}A"] = (torch.rand((64, k), generator=g) * 2 - 1) / float(k) ** 0.5
                d[f"{s}B"] = torch.zeros((n, 64))
            self.theta0.append(d)
        self.set_state(self.theta0)
        layers = TL.trained_layers([TS.SimpleVisit(v) for v in range(48)], 0)
        self.layout = TS.ParamLayout(self.shapes, layers, cfg.train_keys)
        self.flat = torch.zeros(self.layout.size, dtype=torch.float32)
        self.w = TW.T8Worker(cfg, self.work_flat, self.flat, self.shapes, layers, card=0)
        self.h = TS.InlineHandle(self.w)

    def set_state(self, state):
        with torch.no_grad():
            for d, s in zip(self.work, state):
                for k in self.keys:
                    d[k].copy_(bf16(s[k]))
        if hasattr(self, "w"):
            self.w.refresh()

    def set_one(self, L: int, key: str, step):
        """theta0 everywhere except tensor (L, key) = theta0 + step (None: theta0); work = bf16(master) of that one tensor."""
        with torch.no_grad():
            v = self.theta0[L][key] if step is None else self.theta0[L][key] + step
            self.work[L][key].copy_(bf16(v))
        self.w.refresh()


def load_row(plan):
    rows, info = MI.load_plan(plan)
    r = rows[ROW]
    return r, info


# ------------------------------------------------------------------------------------------------------------------------------- grad
def phase_grad(a):
    var = os.environ.get("T8D_VARIANT", "A")
    if var == "B":
        LG.CHUNK = 512
        q0 = TL.qb_for
        TL.qb_for = lambda T: max(8, q0(T) // 2)
    elif var == "C":
        RA.ROUND["on"] = False
    elif var == "D":
        rw = RA.route_weights
        RA.route_weights = lambda logits, ids, *x, **k: rw(logits.detach(), ids, *x, **k)
    rig = Rig(a)
    row, _ = load_row(a.plan)
    row.advantage = 1.0
    t0 = time.time()
    old = rig.h.call("logprobs_row", row)
    row.notes["old_lp"] = old
    out = TS.compute_step([rig.h], rig.layout, [rig.flat], rig.shapes, rig.cfg, [row], 0, StepConfig(max_policy_lag=None), seeds="ppo")
    N = float(out.metrics["total_tokens"])
    nk = len(rig.keys)
    G = {f"{L}.{k}": (-N * out.grads[L * nk + j].double()).float() for L in range(len(rig.shapes)) for j, k in enumerate(rig.keys)}
    norms = {name: float(t.double().norm()) for name, t in G.items()}
    keep = {f"{L}.{k}": G[f"{L}.{k}"] for L, k in TENSORS}
    torch.save({"G": keep, "norms": norms, "N": N, "variant": var, "S0": float(np.asarray(old, np.float64)[row.loss_mask[1:]].sum()),
                "replay_checked_rows": out.metrics["replay_checked_rows"], "stale_checked_rows": out.metrics["stale_checked_rows"],
                "seconds": time.time() - t0}, os.path.join(a.run_dir, f"grad_{var}.pt"))
    log(f"grad {var}: N {N:.0f}, replay {out.metrics['replay_checked_rows']}, stale {out.metrics['stale_checked_rows']}, "
        f"|G| per tensor {[round(norms[f'{L}.{k}'], 4) for L, k in TENSORS]}, {time.time() - t0:.0f} s")
    rig.w.close()
    return 0


# ------------------------------------------------------------------------------------------------------------------------------- fd
def chain_S(rig, ids, x0, routing, mask_t, keep_outs=None, cmp_outs=None):
    """S of the trainer's recompute run as a forward with the given routing (fp32; RA.ROUND as set). keep_outs: list to append each visit's output
    (cpu fp32). cmp_outs: per visit fp32 outputs at theta0 (cpu) -> per visit (share changed, share of changes below half an ulp)."""
    eng, W = rig.w.eng, rig.w.W
    dev = eng.dev
    attn_scale = rig.cfg.attn_alpha / rig.cfg.attn_rank
    lora = []
    for d in rig.work:
        keys = [k for k in d if k != "scale"]
        lora.append(TV.lora_dict(keys, [d[k].to(dev) for k in keys], B.ALPHA / B.RANK, attn_scale))
    stats = []
    with torch.no_grad(), torch.device(dev):
        slots = {0: x0.float()}
        for v, vi in enumerate(eng.visits):
            bank = (slots[vi.bank[0]], slots[vi.bank[1]]) if vi.looped else None
            o = TV.visit_torch_t8(W, vi, [slots[s] for s in vi.stack], slots[vi.partial], bank, routing[v], lora[vi.layer])
            slots[vi.next_partial] = o
            if keep_outs is not None:
                keep_outs.append(o.float().cpu())
            if cmp_outs is not None:
                ref = cmp_outs[v].to(dev)
                dlt = (o.float() - ref).abs()
                st = ref.to(torch.bfloat16).float()
                half = torch.ldexp(torch.ones_like(st), torch.frexp(st)[1] - 9)   # x = m 2^e, m in [0.5, 1): bf16 ulp 2^(e - 8), half of it
                ch = dlt > 0
                stats.append((float(ch.float().mean()), float(((dlt < half) & ch).float().sum() / ch.float().sum().clamp(min=1))))
                del ref, dlt, st, half, ch
        hid = RA.inkling_attn_res([slots[0], slots[1], slots[2], slots[3]], slots[6], None, W.q_out, W.final_norm.reshape(1, B.D), B.EPS)[0]
        tgt = ids[1:].long()
        headf = W.head.float()
        S = 0.0
        pos = torch.nonzero(mask_t[1:])[:, 0]
        for a0 in range(0, int(pos.numel()), 128):
            p = pos[a0:a0 + 128]
            z = torch.tanh((hid[p] @ headf.T) / 30.0) * 30.0
            S += float((z.gather(1, tgt[p][:, None])[:, 0] - torch.logsumexp(z, -1)).double().sum())
        del headf, slots, hid
    return S, stats


def engine_S(rig, ids, mask_t, base=None):
    """S, routing, outs of the engine's forward at the current banks; with base: flips per visit and changed stored values per visit."""
    res = F.forward(rig.w.eng, ids, save=True, adapters_per_layer=rig.w._adapters())
    lp = F.logprobs(rig.w.eng, res["hidden"], ids)
    m = mask_t[1:]
    S = float(lp.double()[m].sum())
    out = {"S": S, "lp_mean": float(lp.double()[m].mean())}
    if base is not None:
        r0, r1 = base["routing"].sort(-1)[0], res["routing"].sort(-1)[0]
        fl = (r0 != r1).any(-1)                               # [48, T] a token's expert set changed at that visit
        out["flips_per_visit"] = [int(x) for x in fl.sum(1).tolist()]
        anyf = fl.any(0)
        first = int(torch.nonzero(anyf)[0, 0]) if bool(anyf.any()) else ids.numel()
        lpos = torch.nonzero(m)[:, 0] + 1
        out["loss_tokens_after_first_flip"] = float((lpos >= first).float().mean())
        out["stored_changed_per_visit"] = [float((o0.view(torch.int16) != o1.view(torch.int16)).float().mean())
                                           for o0, o1 in zip(base["outs"], res["outs"])]
    keep = {"routing": res["routing"], "outs": res["outs"]}
    del res, lp
    rig.w.eng._ws = None
    return out, keep


def phase_fd(a):
    deadline = time.time() + a.budget_s
    rig = Rig(a)
    row, _ = load_row(a.plan)
    ids = rig.w._ids(row)
    mask_t = torch.as_tensor(row.loss_mask.copy(), device=rig.w.dev)
    G = {}
    for var in ("A", "B", "C", "D"):
        p = os.path.join(a.run_dir, f"grad_{var}.pt")
        if os.path.exists(p):
            G[var] = torch.load(p, map_location="cpu", weights_only=False)
    GA = G["A"]["G"]
    rep = {"row": ROW, "T": row.T, "loss_tokens": int(row.loss_mask.sum()), "variants": {}, "fd": [], "dropped": []}
    # T-b: the variants against A
    for var, d in G.items():
        rep["variants"][var] = {}
        for name, ga in GA.items():
            gx = d["G"][name].double()
            ga = ga.double()
            dis = torch.sign(gx) != torch.sign(ga)
            rep["variants"][var][name] = {"norm": float(gx.norm()), "cos": float((gx * ga).sum() / (gx.norm() * ga.norm()).clamp(min=1e-300)),
                                          "rel_err": float((gx - ga).norm() / ga.norm().clamp(min=1e-300)),
                                          "l1_share_sign_disagree": float(ga.abs()[dis].sum() / ga.abs().sum().clamp(min=1e-300)),
                                          "l2_share_sign_disagree": float((ga[dis] ** 2).sum() / (ga ** 2).sum().clamp(min=1e-300))}
    rep["norms_A"] = G["A"]["norms"]

    def save():
        json.dump(rep, open(a.out, "w"), indent=1)

    save()
    # theta0 reference values
    with torch.no_grad():
        e0, base = engine_S(rig, ids, mask_t)
        x0 = F.forward(rig.w.eng, ids, save=True, adapters_per_layer=rig.w._adapters())["x0"]
    rep["S0_engine"] = e0["S"]
    routing0 = base["routing"]
    RA.ROUND["on"] = True
    t0 = time.time()
    S0_frozen, _ = chain_S(rig, ids, x0, routing0, mask_t)
    rep["chain_seconds"] = time.time() - t0
    RA.ROUND["on"] = False
    smooth_outs = []
    S0_smooth, _ = chain_S(rig, ids, x0, routing0, mask_t, keep_outs=smooth_outs)
    RA.ROUND["on"] = True
    rep["S0_frozen"], rep["S0_smooth"] = S0_frozen, S0_smooth
    log(f"theta0: S engine {e0['S']:.6f}, frozen chain {S0_frozen:.6f}, smooth {S0_smooth:.6f}; chain forward {rep['chain_seconds']:.0f} s")
    save()
    # tasks in priority order: (tensor, direction, rung, modes)
    tasks = []
    for mult in (1.0, 16.0, 256.0, 4.0, 64.0, 0.25):              # the ladder's spine first, then the rungs between
        for L, k in TENSORS:
            for dname in ("s", "g"):
                modes = ["free"]
                if L in ("12", "3") or dname == "s":
                    modes.append("frozen")
                if L == "12":
                    modes.append("smooth")
                tasks.append((L, k, dname, mult, modes))
    for L, k, dname, mult, modes in tasks:
        if time.time() > deadline:
            rep["dropped"].append([L, k, dname, mult, modes])
            continue
        g = GA[f"{L}.{k}"].double()
        n = g.numel()
        e = BASE_E * mult
        step = (torch.sign(g) * e) if dname == "s" else (g / g.norm() * e * n ** 0.5)
        pred = float((g * step).sum())
        res = {"tensor": f"{L}.{k}", "dir": dname, "mult": mult, "e": e, "pred": pred}
        for sign in (1, -1):
            rig.set_one(int(L), k, (sign * step).float())
            with torch.no_grad():
                eo, _ = engine_S(rig, ids, mask_t, base=base)
            res[f"free{'+' if sign > 0 else '-'}"] = eo
            if "frozen" in modes:
                RA.ROUND["on"] = True
                res[f"frozen{'+' if sign > 0 else '-'}"] = chain_S(rig, ids, x0, routing0, mask_t)[0]
            if "smooth" in modes:
                RA.ROUND["on"] = False
                s, st = chain_S(rig, ids, x0, routing0, mask_t, cmp_outs=smooth_outs if sign > 0 else None)
                RA.ROUND["on"] = True
                res[f"smooth{'+' if sign > 0 else '-'}"] = s
                if st:
                    res["smooth_changed_share_and_below_half_ulp_per_visit"] = st
        rig.set_one(int(L), k, None)
        res["f_free"] = (res["free+"]["S"] - res["free-"]["S"]) / (2 * pred) if pred else None
        res["curv_free"] = (res["free+"]["S"] + res["free-"]["S"] - 2 * e0["S"]) / 2
        res["drift_per_token_free"] = [res["free+"]["lp_mean"] - e0["lp_mean"], res["free-"]["lp_mean"] - e0["lp_mean"]]
        if "frozen" in modes:
            res["f_frozen"] = (res["frozen+"] - res["frozen-"]) / (2 * pred) if pred else None
        if "smooth" in modes:
            res["f_smooth"] = (res["smooth+"] - res["smooth-"]) / (2 * pred) if pred else None
        rep["fd"].append(res)
        log(f"{L}.{k} {dname} x{mult:g}: pred {pred:.4g}, f free {res['f_free']:.4f}, frozen {res.get('f_frozen')}, smooth {res.get('f_smooth')}, "
            f"flips {sum(res['free+']['flips_per_visit'])}")
        save()
    rep["seconds"] = a.budget_s - (deadline - time.time())
    save()
    rig.w.close()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("grad", "fd"))
    ap.add_argument("--plan", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--budget-s", type=float, default=1500.0)
    a = ap.parse_args()
    return phase_grad(a) if a.phase == "grad" else phase_fd(a)


if __name__ == "__main__":
    sys.exit(main())
