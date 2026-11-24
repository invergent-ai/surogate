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
def chain_S(rig, ids, x0, routing, mask_t, keep_outs=None, cmp_outs=None, inject=None, keep_hidden=None):
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
            if inject is not None and inject[0] == v:         # diag-2: a perturbation of this visit's output
                o = o.float() + inject[1]
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
        if keep_hidden is not None:
            keep_hidden.append(hid.detach().float().clone())
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


def phase_inject(a):
    """diag-2: per-visit Jacobian check with the rounding OFF in both the backward and the chained forward (the smooth function; routing frozen at
    theta0's). g_out(v) = the backward's d S / d (output of visit v); the chain's S with that output perturbed by +- eps u; follow
    f(v) = [S(+) - S(-)] / (2 eps <g_out(v), u>), eps = 0.5 / <g_out(v), u> (a predicted change of 0.5 nats). A correct backward gives f(v) in
    [0.9, 1.1] at every v; the first v (going back from 47; pairs across every pass boundary) where f leaves [0.9, 1.1] locates the fault between
    that visit and the previous one checked. If f(v) stays in [0.9, 1.1] at every injected visit, the fault is in the vjp onto the LoRA tensors,
    not in the chain through the residual stream.
    Order (written before any data): u = g_out/|g_out| at visits 47, 42, 41, 36, 30, 29, 24, 18, 17, 12, 6, 5, 3, 45; then u = sign(g_out)/sqrt(n)
    at 47, 36, 24, 12, 3; last, central differences of the smooth chain along the smooth gradient G_C for 12.B13 (then 3.B13) at x1/4 and x1/16
    of diag-1's e: follow near diag-1's (0.26 / 0.07) = the gradient is wrong for the chain; above 0.7 = the micro-step's size is outside the
    linear range. The chain freezes the expert SET only (gate weights recomputed from the stepped state, as the backward differentiates them)
    and applies a looped layer's tensor at all its visits. Every task's result is saved at once; a cut at the deadline costs only the tail."""
    deadline = time.time() + a.budget_s
    RA.ROUND["on"] = False
    rig = Rig(a)
    row, _ = load_row(a.plan)
    dev = rig.w.dev
    mask_t = torch.as_tensor(row.loss_mask.copy(), device=dev)
    lp, ctx = rig.w._forward(row)
    g = torch.as_tensor(row.loss_mask[1:].astype(np.float32), device=dev)
    G_ORDER = (47, 42, 41, 36, 30, 29, 24, 18, 17, 12, 6, 5, 3, 45)
    S_ORDER = (47, 36, 24, 12, 3)
    norms, gouts, pgrad = {}, {}, {}

    def onv(v, g_out):
        norms[v] = float(g_out.double().norm())
        if v in G_ORDER:
            gouts[v] = g_out.detach().float().cpu()            # on the host; one at a time back to the card

    def ong(v, layer, j, gj):
        if j == 1 and layer in (12, 3):
            c = gj.detach().float().cpu()
            pgrad[layer] = c if layer not in pgrad else pgrad[layer].add_(c)

    t0 = time.time()
    TL.backward_row(rig.w.eng, rig.w.W, ctx["ids"], ctx["res"], rig.w.lora, B.ALPHA / B.RANK, rig.w.v_first, g, rig.w.buffers,
                    ong, q_block=ctx["qb"], train_idx=rig.w.train_idx, attn_scale=0.5, on_visit_grad=onv)
    rep = {"row": ROW, "round": False, "gout_norm_per_visit": {str(k): norms[k] for k in sorted(norms)}, "backward_s": time.time() - t0,
           "inject": [], "small_steps": [], "dropped": []}
    routing0 = ctx["res"]["routing"].clone()
    ids = ctx["ids"]
    rig.w._release(ctx)
    with torch.no_grad():
        x0 = F.forward(rig.w.eng, ids, save=True, adapters_per_layer=rig.w._adapters())["x0"]
    S0, _ = chain_S(rig, ids, x0, routing0, mask_t)
    rep["S0_smooth"] = S0
    log(f"backward {rep['backward_s']:.0f} s; |g_out| per visit {[round(norms[k], 3) for k in sorted(norms)]}")
    json.dump(rep, open(a.out, "w"), indent=1)

    def inject(v, dname):
        go = gouts[v].to(dev)
        u = go / go.norm() if dname == "g" else torch.sign(go) / float(go.numel()) ** 0.5
        ip = float((go.double() * u.double()).sum())
        del go
        eps = 0.5 / ip
        sp = chain_S(rig, ids, x0, routing0, mask_t, inject=(v, eps * u))[0]
        sm = chain_S(rig, ids, x0, routing0, mask_t, inject=(v, -eps * u))[0]
        f = (sp - sm) / (2 * eps * ip)
        rep["inject"].append({"visit": v, "dir": dname, "eps": eps, "ip": ip, "pred": eps * ip, "S+": sp, "S-": sm, "f": f,
                              "curv": (sp + sm - 2 * S0) / 2})
        log(f"visit {v} {dname}: f {f:.4f}, curvature {(sp + sm - 2 * S0) / 2:.4g}")

    def small(L, mult):
        gC = pgrad[L].double()
        step = (gC / gC.norm() * BASE_E * mult * gC.numel() ** 0.5).float()
        pred = float((gC * step.double()).sum())
        rig.set_one(L, "B13", step)
        sp = chain_S(rig, ids, x0, routing0, mask_t)[0]
        rig.set_one(L, "B13", -step)
        sm = chain_S(rig, ids, x0, routing0, mask_t)[0]
        rig.set_one(L, "B13", None)
        f = (sp - sm) / (2 * pred)
        rep["small_steps"].append({"tensor": f"{L}.B13", "mult": mult, "pred": pred, "S+": sp, "S-": sm, "f_smooth": f})
        log(f"smooth chain {L}.B13 along G_C x{mult:g}: pred {pred:.4g}, f {f:.4f}")

    tasks = [("inject", v, "g") for v in G_ORDER] + [("inject", v, "s") for v in S_ORDER] + \
            [("small", 12, 0.25), ("small", 12, 0.0625), ("small", 3, 0.25), ("small", 3, 0.0625)]
    for t in tasks:
        if time.time() > deadline:
            rep["dropped"].append(list(t))
            continue
        inject(t[1], t[2]) if t[0] == "inject" else small(t[1], t[2])
        json.dump(rep, open(a.out, "w"), indent=1)
    json.dump(rep, open(a.out, "w"), indent=1)
    rig.w.close()
    return 0


def phase_local(a):
    """diag-3: per-visit LOCAL Jacobian check on plan row 3, rounding OFF, routing frozen at theta0's, at the smooth chain's own slot values (forward
    and backward share the point). For visit v: cotangent c = the backward's g_out(v); the vjp into every input slot (stack..., partial, bank...)
    computed EXACTLY as backward_row does it (placeholders + act_vals + act_sinks, autograd.grad over SINK_DUMMIES, then the manual residual add
    when not `first`), and into the layer's qB and B13; then for each slot i: u = vjp_i / |vjp_i|, eps = 0.5 / |vjp_i|, the single visit
    recomputed with visit_torch_t8 on real tensors with slot i at +- eps u: f_local = <c, o(+) - o(-)> / (2 eps |vjp_i|). Same for qB / B13.
    PASS LINE (written before any data): a correct visit gives f_local in [0.98, 1.02] for every slot and tensor. The first (visit, slot) outside
    it names the operation path; all inside means the fault is in how the per-visit gradients are accumulated across visits (then the chain under
    plain autograd against backward_row is the next check). The gating vjp runs as backward_row runs it: inside LG._gemm_mode() and with
    LG._q_block fixed to the row's query block.
    Added (agent-RX, before any data): (b) the same vjp on the plain path (no gemm mode, default query block): cos < 0.999 or norm ratio outside
    [0.99, 1.01] at any slot / tensor = the fast path differs. (c) the gating vjp at the ENGINE's saved values (backward_row's own act_vals),
    same cotangent, against the vjp at the chain's values: cos and norm ratio per slot, recorded with no line. Also per visit: RMS of each
    input slot's value and of the output."""
    deadline = time.time() + a.budget_s
    RA.ROUND["on"] = False
    rig = Rig(a)
    row, _ = load_row(a.plan)
    dev = rig.w.dev
    eng, W = rig.w.eng, rig.w.W
    T = row.T
    mask_t = torch.as_tensor(row.loss_mask.copy(), device=dev)
    lp, ctx = rig.w._forward(row)
    g = torch.as_tensor(row.loss_mask[1:].astype(np.float32), device=dev)
    gouts = {}
    TL.backward_row(eng, W, ctx["ids"], ctx["res"], rig.w.lora, B.ALPHA / B.RANK, rig.w.v_first, g, rig.w.buffers, lambda *x: None,
                    q_block=ctx["qb"], train_idx=rig.w.train_idx, attn_scale=0.5,
                    on_visit_grad=lambda v, go: gouts.__setitem__(v, go.detach().float().cpu()))
    routing0 = ctx["res"]["routing"].clone()
    ids = ctx["ids"]
    host_vals = ctx["res"]["host"]                            # the engine's saved values backward_row uses: kept alive until the end
    qb = ctx["qb"]
    with torch.no_grad():
        x0 = F.forward(eng, ids, save=True, adapters_per_layer=rig.w._adapters())["x0"].float()
    outs = []
    S0, _ = chain_S(rig, ids, x0, routing0, mask_t, keep_outs=outs)
    prod, _ = TL.producers(eng.visits)
    scale, attn_scale = B.ALPHA / B.RANK, rig.cfg.attn_alpha / rig.cfg.attn_rank
    rep = {"row": ROW, "round": False, "S0_smooth": S0, "local": [], "dropped": [],
           "visits": [{"v": vi.v, "layer": vi.layer, "pass": vi.pass_, "first": vi.first, "global": vi.global_, "looped": vi.looped,
                       "producers": prod[vi.v]} for vi in eng.visits]}
    json.dump(rep, open(a.out, "w"), indent=1)
    log(f"setup done: S0 smooth {S0:.6f}")

    def val(p):
        return x0 if p == -1 else outs[p].to(dev)

    def one(v):
        vi = eng.visits[v]
        ps = prod[v]
        n = len(vi.stack)
        c = gouts[v].to(dev)
        vals = [val(p) for p in ps]
        lo = rig.w.lora[vi.layer]
        keys = [k for k in lo if k != "scale"]
        lt = [lo[k].to(dev) for k in keys]

        def vjp(act, fast):
            """backward_row's per-visit vjp at the values `act` (device fp32 or the engine's host bf16): sinks per slot + qB / B13 grads."""
            sinks = [torch.zeros(T, B.D, dtype=torch.float32, device=dev) for _ in ps]
            saved_qb = LG._q_block
            try:
                if fast:
                    LG._q_block = lambda T_, window, hq=24: qb
                ctxm = LG._gemm_mode() if fast else __import__("contextlib").nullcontext()
                with ctxm, torch.enable_grad(), torch.device(dev):
                    ins = [torch.zeros(1, dtype=torch.float32, device=dev).expand(T, B.D) for _ in ps]
                    leaves = [t.detach().requires_grad_(k in ("qB", "B13")) for k, t in zip(keys, lt)]
                    want = [t for t in leaves if t.requires_grad]
                    lora = TV.lora_dict(keys, leaves, scale, attn_scale)
                    LG.SINK_DUMMIES.clear()
                    out = TV.visit_torch_t8(W, vi, ins[:n], ins[n], (ins[n + 1], ins[n + 2]) if vi.looped else None, routing0[v], lora,
                                            act_vals=act, act_sinks=sinks)
                    got = torch.autograd.grad(out, want + list(LG.SINK_DUMMIES), c, allow_unused=True)
                    LG.SINK_DUMMIES.clear()
                pk = [k for k, t in zip(keys, leaves) if t.requires_grad]
                pg = {k: (x.detach() if x is not None else None) for k, x in zip(pk, got[:len(want)])}
            finally:
                LG._q_block = saved_qb
                LG.SINK_DUMMIES.clear()
            if not vi.first:
                LG.sink_add(sinks[n], c)                      # the MLP-output conv's residual, as backward_row
            return sinks, pg

        def cmp(xs, ys):
            out_ = []
            for x_, y_ in zip(xs, ys):
                if x_ is None or y_ is None:
                    out_.append(None)
                    continue
                x_, y_ = x_.double(), y_.double()
                ny = float(y_.norm())
                out_.append({"cos": float((x_ * y_).sum() / (x_.norm() * y_.norm()).clamp(min=1e-300)),
                             "norm_ratio": float(x_.norm()) / ny if ny else None})
            return out_

        sinks, pgr = vjp(vals, True)                          # the gating vjp: backward_row's path (gemm mode, fixed query block)
        base_lora = TV.lora_dict(keys, lt, scale, attn_scale)

        def fwd(vs, lo_):
            with torch.no_grad(), torch.device(dev):
                return TV.visit_torch_t8(W, vi, vs[:n], vs[n], (vs[n + 1], vs[n + 2]) if vi.looped else None, routing0[v], lo_).float()

        res = {"visit": v, "layer": vi.layer, "pass": vi.pass_, "first": vi.first, "looped": vi.looped, "slots": [], "lora": {}}
        res["rms_in"] = [float(x.double().pow(2).mean().sqrt()) for x in vals]
        res["rms_out"] = float(outs[v].double().pow(2).mean().sqrt())
        try:                                                  # (b) the plain path
            sb, pb = vjp(vals, False)
            res["b_plain_vs_fast"] = {"slots": cmp(sb, sinks), "lora": dict(zip(("qB", "B13"), cmp([pb.get("qB"), pb.get("B13")],
                                                                                                   [pgr.get("qB"), pgr.get("B13")])))}
            del sb, pb
        except Exception as e:
            res["b_plain_vs_fast"] = {"error": f"{type(e).__name__}: {e}"[:500]}
        try:                                                  # (c) at the engine's saved values
            sc, pc = vjp([host_vals[p] for p in ps], True)
            res["c_engine_vs_chain"] = {"slots": cmp(sc, sinks), "lora": dict(zip(("qB", "B13"), cmp([pc.get("qB"), pc.get("B13")],
                                                                                                    [pgr.get("qB"), pgr.get("B13")])))}
            del sc, pc
        except Exception as e:
            res["c_engine_vs_chain"] = {"error": f"{type(e).__name__}: {e}"[:500]}
        torch.cuda.empty_cache()
        names = [f"stack{j}" for j in range(n)] + ["partial"] + (["bank0", "bank1"] if vi.looped else [])
        for i, (nm, p) in enumerate(zip(names, ps)):
            nv = float(sinks[i].double().norm())
            if nv == 0.0:
                res["slots"].append({"slot": nm, "producer": p, "vjp_norm": 0.0, "f": None})
                continue
            u = sinks[i] / nv
            eps = 0.5 / nv
            vp, vm = list(vals), list(vals)
            vp[i], vm[i] = vals[i] + eps * u, vals[i] - eps * u
            d = float((c.double() * (fwd(vp, base_lora).double() - fwd(vm, base_lora).double())).sum())
            res["slots"].append({"slot": nm, "producer": p, "vjp_norm": nv, "f": d / (2 * eps * nv)})
        for k in ("qB", "B13"):
            gk = pgr.get(k)
            if gk is None or float(gk.norm()) == 0.0:
                res["lora"][k] = None
                continue
            nk = float(gk.double().norm())
            u = gk / nk
            eps = 0.5 / nk
            j = keys.index(k)
            lp_, lm_ = list(lt), list(lt)
            lp_[j], lm_[j] = lt[j] + eps * u, lt[j] - eps * u
            d = float((c.double() * (fwd(vals, TV.lora_dict(keys, lp_, scale, attn_scale)).double()
                                     - fwd(vals, TV.lora_dict(keys, lm_, scale, attn_scale)).double())).sum())
            res["lora"][k] = {"grad_norm": nk, "f": d / (2 * eps * nk)}
        del sinks, vals, c
        torch.cuda.empty_cache()
        return res

    order = [37, 38, 39, 40, 41, 31, 32, 33, 34, 35, 30, 36, 43, 25, 26, 27, 28, 29, 19, 20, 21, 22, 23, 13, 14, 15, 16, 17,
             7, 8, 9, 10, 11, 6, 12, 18, 24, 1, 2, 3, 4, 5]
    order += [v for v in range(48) if v not in order]
    for v in order:
        if time.time() > deadline:
            rep["dropped"].append(v)
            continue
        t0 = time.time()
        r = one(v)
        rep["local"].append(r)
        bad = [s["slot"] for s in r["slots"] if s["f"] is not None and not 0.98 <= s["f"] <= 1.02] + \
              [k for k, x in r["lora"].items() if x is not None and not 0.98 <= x["f"] <= 1.02]
        log(f"visit {v} (layer {r['layer']}, pass {r['pass']}): " + ", ".join(f"{s['slot']} {s['f']:.4f}" if s["f"] is not None else f"{s['slot']} -"
                                                                         for s in r["slots"])
            + "; " + ", ".join(f"{k} {x['f']:.4f}" for k, x in r["lora"].items() if x is not None)
            + (f"  OUTSIDE: {bad}" if bad else "") + f"  ({time.time() - t0:.1f} s)")
        json.dump(rep, open(a.out, "w"), indent=1)
    rig.w._release(ctx)
    rig.w.close()
    return 0


def phase_lin(a):
    """diag-4: the linearisation point. Plan row 3, routing frozen at theta0's, theta0. Lines written before any data:
    1 STATES: per visit, relative L2 distance of the engine's stored output to the smooth chain's (rounding off) and to the frozen chain's
      (rounding on) output. Recorded, no line.
    2 BACKWARD AT THE CHAIN'S OWN STATES: backward_row with vals = the smooth chain's outputs (fp32), its final hidden and theta0's routing,
      rounding off -> G_S (six tensors) and g_S(v); against G_E / g_E(v) = backward_row at the engine's states, rounding off, same session:
      cos and norm ratio per tensor and per visit. Recorded, no line.
    3 FOLLOW OF G_S ON THE SMOOTH CHAIN: central differences along G_S at x1 (e = 2.4e-4, direction G_S / |G_S| x e sqrt(n)) for 12.B13,
      3.B13, 21.B13, then 12.qB. LINE: f in [0.9, 1.1] at layers 12, 3 and 21 = the backward is right when linearised on the chain's own
      trajectory (the micro-step's failure is the linearisation point); f < 0.9 at layer 12 or 3 while layer 21 passes = the accumulation
      across visits is wrong after all.
    4 FOLLOW OF G_S IN THE ENGINE (routing free): central differences along G_S at x1 and x4 for 12.B13 and 3.B13. LINE: at x4 f in
      [0.7, 1.3] = a remedy exists (linearise on the fp32 chain); f < 0.5 = it does not transfer to the engine. The engine line is read at
      x4; where the prediction at x4 is below 3 nats it is read at x16 instead (the engine's routing scatter is 0.6-0.8 nats per side).
    5 (budget allowing) 2-3 on the FROZEN chain's states (rounding on): G_F, its follow on the frozen chain for the same three B13 tensors.
    The head's backward reads the hidden as bf16 (B.HeadLogprob), as in production; the smooth chain's hidden is rounded there."""
    deadline = time.time() + a.budget_s
    rig = Rig(a)
    row, _ = load_row(a.plan)
    dev = rig.w.dev
    eng, W = rig.w.eng, rig.w.W
    mask_t = torch.as_tensor(row.loss_mask.copy(), device=dev)
    g = torch.as_tensor(row.loss_mask[1:].astype(np.float32), device=dev)
    J_QB, J_B13 = rig.keys.index("qB"), rig.keys.index("B13")
    TT = {(12, J_B13): "12.B13", (3, J_B13): "3.B13", (21, J_B13): "21.B13", (12, J_QB): "12.qB", (3, J_QB): "3.qB", (21, J_QB): "21.qB"}
    rep = {"row": ROW, "dropped": []}

    def save():
        json.dump(rep, open(a.out, "w"), indent=1)

    def run_backward(vals):
        Gs, gv = {}, {}

        def ong(v, layer, j, gj):
            nm = TT.get((layer, j))
            if nm is not None:
                c = gj.detach().float().cpu()
                Gs[nm] = c if nm not in Gs else Gs[nm].add_(c)

        TL.backward_row(eng, W, ids, vals, rig.w.lora, B.ALPHA / B.RANK, rig.w.v_first, g, rig.w.buffers, ong, q_block=qb,
                        train_idx=rig.w.train_idx, attn_scale=0.5, on_visit_grad=lambda v, go: gv.__setitem__(v, go.detach().float().cpu()))
        return Gs, gv

    def cmp(x, y):
        x, y = x.double(), y.double()
        return {"cos": float((x * y).sum() / (x.norm() * y.norm()).clamp(min=1e-300)), "norm_ratio": float(x.norm() / y.norm().clamp(min=1e-300))}

    # the engine's states and G_E (rounding off, as diag-2 / diag-3)
    RA.ROUND["on"] = False
    lp, ctx = rig.w._forward(row)
    ids, qb = ctx["ids"], ctx["qb"]
    routing0 = ctx["res"]["routing"].clone()
    host = ctx["res"]["host"]
    t0 = time.time()
    G_E, g_E = run_backward(ctx["res"])
    rep["backward_engine_s"] = time.time() - t0
    with torch.no_grad():
        x0 = F.forward(eng, ids, save=True, adapters_per_layer=rig.w._adapters())["x0"].float()
    # the smooth chain's and the frozen chain's states
    outs_s, hid_s = [], []
    S_s, _ = chain_S(rig, ids, x0, routing0, mask_t, keep_outs=outs_s, keep_hidden=hid_s)
    RA.ROUND["on"] = True
    outs_f, hid_f = [], []
    S_f, _ = chain_S(rig, ids, x0, routing0, mask_t, keep_outs=outs_f, keep_hidden=hid_f)
    RA.ROUND["on"] = False
    rep["S"] = {"smooth": S_s, "frozen": S_f}
    # 1 STATES
    rep["states"] = []
    for v in range(len(eng.visits)):
        if v not in host:
            rep["states"].append(None)
            continue
        e = host[v].to(dev).double()
        rep["states"].append({"v": v, "engine_vs_smooth": float((e - outs_s[v].to(dev).double()).norm() / e.norm()),
                              "engine_vs_frozen": float((e - outs_f[v].to(dev).double()).norm() / e.norm())})
        del e
    log("states: " + " ".join(f"{x['v']}:{x['engine_vs_smooth']:.3f}/{x['engine_vs_frozen']:.3f}" for x in rep["states"] if x))
    save()
    rig.w._release(ctx)
    del ctx, host

    def chain_vals(outs, hid):
        h = {-1: x0}
        for v, o in enumerate(outs):
            h[v] = o                                          # host fp32; TokenChunked moves each chunk to the card
        return {"host": h, "routing": routing0, "hidden": hid[0], "lp": None}

    # 2 BACKWARD AT THE SMOOTH CHAIN'S STATES
    t0 = time.time()
    G_S, g_S = run_backward(chain_vals(outs_s, hid_s))
    rep["backward_chain_s"] = time.time() - t0
    rep["G_S_vs_G_E"] = {nm: cmp(G_S[nm], G_E[nm]) for nm in G_S}
    rep["gS_vs_gE_per_visit"] = {str(v): cmp(g_S[v], g_E[v]) for v in sorted(g_S) if v in g_E}
    rep["norms"] = {"G_S": {nm: float(t.norm()) for nm, t in G_S.items()}, "G_E": {nm: float(t.norm()) for nm, t in G_E.items()},
                    "g_S": {str(v): float(t.norm()) for v, t in sorted(g_S.items())}}
    log("G_S vs G_E: " + ", ".join(f"{nm} cos {x['cos']:.3f} ratio {x['norm_ratio']:.3f}" for nm, x in rep["G_S_vs_G_E"].items()))
    save()

    def fd_chain(G, nm, round_on):
        L, k = nm.split(".")
        gC = G[nm].double()
        step = (gC / gC.norm() * BASE_E * gC.numel() ** 0.5).float()
        pred = float((gC * step.double()).sum())
        RA.ROUND["on"] = round_on
        rig.set_one(int(L), k, step)
        sp = chain_S(rig, ids, x0, routing0, mask_t)[0]
        rig.set_one(int(L), k, -step)
        sm = chain_S(rig, ids, x0, routing0, mask_t)[0]
        rig.set_one(int(L), k, None)
        RA.ROUND["on"] = False
        return {"tensor": nm, "pred": pred, "S+": sp, "S-": sm, "f": (sp - sm) / (2 * pred)}

    def fd_engine(G, nm, mult):
        L, k = nm.split(".")
        gC = G[nm].double()
        step = (gC / gC.norm() * BASE_E * mult * gC.numel() ** 0.5).float()
        pred = float((gC * step.double()).sum())
        r = {}
        for sg in (1, -1):
            rig.set_one(int(L), k, sg * step)
            with torch.no_grad():
                res = F.forward(eng, ids, save=False, adapters_per_layer=rig.w._adapters())
                lpx = F.logprobs(eng, res["hidden"], ids)
            r[sg] = float(lpx.double()[mask_t[1:]].sum())
            del res, lpx
            eng._ws = None
        rig.set_one(int(L), k, None)
        return {"tensor": nm, "mult": mult, "pred": pred, "S+": r[1], "S-": r[-1], "f": (r[1] - r[-1]) / (2 * pred)}

    # 3 and 4
    rep["follow_GS_smooth"], rep["follow_GS_engine"] = [], []
    tasks = [("c", "12.B13"), ("c", "3.B13"), ("c", "21.B13"), ("e", "12.B13", 1.0), ("e", "12.B13", 4.0), ("e", "3.B13", 1.0),
             ("e", "3.B13", 4.0), ("e", "3.B13", 16.0), ("c", "12.qB")]
    for t in tasks:
        if time.time() > deadline:
            rep["dropped"].append(list(t))
            continue
        if t[0] == "c":
            r = fd_chain(G_S, t[1], False)
            rep["follow_GS_smooth"].append(r)
            log(f"smooth chain along G_S {t[1]}: pred {r['pred']:.4g}, f {r['f']:.4f}")
        else:
            r = fd_engine(G_S, t[1], t[2])
            rep["follow_GS_engine"].append(r)
            log(f"engine along G_S {t[1]} x{t[2]:g}: pred {r['pred']:.4g}, f {r['f']:.4f}")
        save()
    # 5 the frozen chain's states (rounding on)
    if time.time() < deadline - 150:
        RA.ROUND["on"] = True
        G_F, g_F = run_backward(chain_vals(outs_f, hid_f))
        RA.ROUND["on"] = False
        rep["G_F_vs_G_E"] = {nm: cmp(G_F[nm], G_E[nm]) for nm in G_F}
        rep["follow_GF_frozen"] = []
        save()
        for nm in ("12.B13", "3.B13", "21.B13"):
            if time.time() > deadline:
                rep["dropped"].append(["frozen", nm])
                continue
            r = fd_chain(G_F, nm, True)
            rep["follow_GF_frozen"].append(r)
            log(f"frozen chain along G_F {nm}: pred {r['pred']:.4g}, f {r['f']:.4f}")
            save()
    else:
        rep["dropped"].append("part 5")
    save()
    rig.w.close()
    return 0


def _patch_source(mod, func_src_name: str, edits: list, cls=None):
    """Test-side patch: re-define a function of `mod` (or a staticmethod of `mod.cls`) from its source with text edits. Returns an undo."""
    import inspect
    import textwrap

    obj = getattr(getattr(mod, cls), func_src_name) if cls else getattr(mod, func_src_name)
    src = textwrap.dedent(inspect.getsource(obj))
    src = "\n".join(ln for ln in src.splitlines() if ln.strip() != "@staticmethod") + "\n"
    for a, b in edits:
        if src.count(a) < 1:
            raise RuntimeError(f"patch anchor not found in {func_src_name}: {a!r}")
        src = src.replace(a, b)
    ns = {}
    exec(compile(src, f"<patched {func_src_name}>", "exec"), mod.__dict__, ns)
    new = ns[func_src_name]
    if cls:
        old = getattr(mod, cls).__dict__[func_src_name]
        setattr(getattr(mod, cls), func_src_name, staticmethod(new))
        return lambda: setattr(getattr(mod, cls), func_src_name, old)
    old = getattr(mod, func_src_name)
    setattr(mod, func_src_name, new)
    return lambda: setattr(mod, func_src_name, old)


def variant_patches(name: str) -> list:
    """Q: no gradient through the attention scores (ds = 0; the value path stays). D: router gate weights held constant. M: the source-mix
    probabilities of inkling_attn_res and the tanh gates of inkling_loop_blend held constant (the normalised values keep their gradient)."""
    undo = []
    if "Q" in name:
        undo.append(_patch_source(LG, "backward", [("ds = p * (dp - dterm + dlb)", "ds = torch.zeros_like(p)")], cls="RelAttentionR"))
    if "D" in name:
        rw = RA.route_weights
        RA.route_weights = lambda logits, ids, *x, **k: rw(logits.detach(), ids, *x, **k)
        undo.append(lambda: setattr(RA, "route_weights", rw))
    if "M" in name:
        undo.append(_patch_source(RA, "inkling_attn_res", [("probs = t_exp(logits - logits.amax(1, keepdim=True))",
                                                            "probs = t_exp(logits - logits.amax(1, keepdim=True)).detach()")]))
        undo.append(_patch_source(RA, "inkling_loop_blend", [("gain = t_tanh(gf[..., :d])", "gain = t_tanh(gf[..., :d]).detach()"),
                                                             ("forget = t_tanh(gf[..., d:])", "forget = t_tanh(gf[..., d:]).detach()")]))
    return undo


def phase_sum(a):
    """diag-5 (agent-RX's design; lines written before any data). Plan row 3, routing frozen at theta0's, theta0.
    A THE SUMMING, no finite differences: on the first --prefix tokens (512; 2,048 ran out of card memory next to the engine) of row 3 (S = sum of log pi over every position; rounding ON, so every
      stored state is a bf16 value and TokenChunked's bf16 storage is exact in both), the frozen chain under plain autograd (real tensors,
      act_vals None, dS/d(output of visit v) by retain_grad, the six tensors as leaves) against backward_row fed exactly the chain's states
      (its outputs, x0, the chain's final hidden, theta0's routing), with an exact fp32 head gradient in both. LINE: cos >= 0.999 and norm
      ratio in [0.99, 1.01] at every visit and tensor = the summing across visits is right (the backward is the exact gradient at its point);
      below at some visit = the summing is wrong, the latest such visit names where. If A fails, B and C are skipped.
    B IS THERE A PART OF THE GRADIENT THAT HOLDS? Variants (test-side patches): Q (attention score cotangent zero), D (router gate weights
      constant), QD, M (attn_res source-mix probabilities and loop-blend tanh gates constant). Each: backward_row on the engine's states and on
      the smooth chain's states (rounding off, as diag-4 item 2); cos and norm ratio of the two for 12.B13, 3.B13, 21.B13 and g(v) per visit.
      LINE: cos > 0.8 at layers 12 and 3 = a candidate; otherwise discarded without finite differences. For candidates: follow in the ENGINE
      along the variant's gradient taken on the engine's states: 12.B13 and 3.B13 at x4 (x16 where the x4 prediction is under 3 nats),
      21.B13 at x1. LINE: [0.7, 1.3] at layers 12 and 3 = a usable gradient exists; < 0.5 = it does not.
    B first (agent-RX, before data): TRUNCATION K = 18 / 24 / 30 (backward_row with v_first 30 / 24 / 18), on the engine's and the chain's
      states: cos and norm ratio for 12.B13, 12.qB, 21.B13 and per visit; the norm of 12.B13; always the engine follow of 12.B13 along the
      variant's gradient on the engine's states (x4, x16 where the x4 prediction is under 3 nats). LINE: engine follow of 12.B13 in
      [0.7, 1.3] = a backward truncated at that depth gives a gradient the engine follows at a looped layer; < 0.5 = it does not. 21.B13 at
      x1 once as control. Patch variants: a candidate PER LAYER (12.B13 cos > 0.8 -> follow 12.B13; 3.B13 cos > 0.8 -> follow 3.B13).
      An error in A is recorded and B / C go on (flagged "A did not run"); only a measured fail of A's line stops the session. A runs in its
      own process (--parts A, diag5A.json), B / C in a second one (--parts BC) unless A wrote a measured fail (exit 3). Truncation records
      also the share of the full gradient's norm kept (12.B13, 12.qB) and the engine follow of 12.qB (same rung rule).
      Order: A; chains and G_S / G_F; K18, K24, K30; Q; M; D; QD; C (the deadline cuts from the tail).
    C trend only, last: the smooth chain along G_S (full gradient, chain states), 12.B13 at x1/4, x1/16, x1/64, S+ and S- printed.
    Recorded once: relative L2 distance smooth vs frozen chain outputs per visit; cos of G_F (frozen chain states, rounding on) against G_S."""
    deadline = time.time() + a.budget_s
    rig = Rig(a)
    row, _ = load_row(a.plan)
    dev = rig.w.dev
    eng, W = rig.w.eng, rig.w.W
    mask_t = torch.as_tensor(row.loss_mask.copy(), device=dev)
    g_full = torch.as_tensor(row.loss_mask[1:].astype(np.float32), device=dev)
    J_QB, J_B13 = rig.keys.index("qB"), rig.keys.index("B13")
    TT = {(12, J_B13): "12.B13", (3, J_B13): "3.B13", (21, J_B13): "21.B13", (12, J_QB): "12.qB", (3, J_QB): "3.qB", (21, J_QB): "21.qB"}
    scale, attn_scale = B.ALPHA / B.RANK, rig.cfg.attn_alpha / rig.cfg.attn_rank
    rep = {"row": ROW, "dropped": []}

    def save():
        json.dump(rep, open(a.out, "w"), indent=1)

    def cmp(x, y):
        x, y = x.double().reshape(-1), y.double().reshape(-1)
        return {"cos": float((x * y).sum() / (x.norm() * y.norm()).clamp(min=1e-300)), "norm_ratio": float(x.norm() / y.norm().clamp(min=1e-300))}

    def run_backward(vals, ids_, g_, qb_, head_grad=None, v_first=None):
        Gs, gv = {}, {}

        def ong(v, layer, j, gj):
            nm = TT.get((layer, j))
            if nm is not None:
                c = gj.detach().float().cpu()
                Gs[nm] = c if nm not in Gs else Gs[nm].add_(c)

        TL.backward_row(eng, W, ids_, vals, rig.w.lora, scale, rig.w.v_first if v_first is None else v_first, g_, rig.w.buffers, ong,
                        head_grad=head_grad, q_block=qb_,
                        train_idx=rig.w.train_idx, attn_scale=attn_scale,
                        on_visit_grad=lambda v, go: gv.__setitem__(v, go.detach().float().cpu()))
        return Gs, gv

    # the engine's states (kept alive for B), x0, routing
    RA.ROUND["on"] = False
    lp, ctx = rig.w._forward(row)
    ids, qb = ctx["ids"], ctx["qb"]
    routing0 = ctx["res"]["routing"].clone()
    with torch.no_grad():
        x0 = F.forward(eng, ids, save=True, adapters_per_layer=rig.w._adapters())["x0"].float()

    # ---------------------------------------------------------------- A (inside a function: its locals die on return)
    def part_A():
        resA = {}
        Tp = int(a.prefix)
        ids_p = ids[:Tp].contiguous()
        rout_p = routing0[:, :Tp].contiguous()
        x0_p = x0[:Tp].clone().requires_grad_(True)              # a leaf, so every visit output carries a graph (its gradient is not used)
        g_p = torch.ones(Tp - 1, dtype=torch.float32, device=dev)
        RA.ROUND["on"] = True
        t0 = time.time()
        try:
            with torch.device(dev):
                leaves = {}
                lora = []
                for L, d in enumerate(rig.work):
                    keys = [k for k in d if k != "scale"]
                    ts = []
                    for j, k in enumerate(keys):
                        t = d[k].to(dev)
                        if (L, j) in TT:
                            t = t.detach().requires_grad_(True)
                            leaves[TT[(L, j)]] = t
                        ts.append(t)
                    lora.append(TV.lora_dict(keys, ts, scale, attn_scale))
                slots = {0: x0_p}
                outs_p = []
                for v, vi in enumerate(eng.visits):
                    bank = (slots[vi.bank[0]], slots[vi.bank[1]]) if vi.looped else None
                    o = TV.visit_torch_t8(W, vi, [slots[s] for s in vi.stack], slots[vi.partial], bank, rout_p[v], lora[vi.layer])
                    o.retain_grad()
                    outs_p.append(o)
                    slots[vi.next_partial] = o
                hid = RA.inkling_attn_res([slots[0], slots[1], slots[2], slots[3]], slots[6], None, W.q_out, W.final_norm.reshape(1, B.D), B.EPS)[0]
                headf = W.head.float()
                tgt = ids_p[1:].long()
                parts = []
                for c0 in range(0, Tp - 1, 128):
                    c1 = min(Tp - 1, c0 + 128)
                    z = torch.tanh((hid[c0:c1] @ headf.T) / 30.0) * 30.0
                    parts.append((z.gather(1, tgt[c0:c1, None])[:, 0] - torch.logsumexp(z, -1)).sum())
                total = torch.stack(parts).sum()
                total.backward()                                  # one backward through the whole chain
                S = float(total)
                del headf, parts, total
            g_auto = {v: o.grad.detach().float().cpu() for v, o in enumerate(outs_p) if o.grad is not None}
            G_auto = {nm: t.grad.detach().float().cpu() for nm, t in leaves.items() if t.grad is not None}
            host_p = {-1: x0_p.detach()}
            for v, o in enumerate(outs_p):
                host_p[v] = o.detach().float()
            hid_d = hid.detach().float()
            del outs_p, slots, leaves, lora, o

            def head_exact(W_, hidden, targets, gg):
                """the exact fp32 head gradient (the same arithmetic as the chain's head above)"""
                h = hidden.detach().float()
                hf = W_.head.float()
                out = torch.zeros_like(h)
                with torch.enable_grad():
                    for c0 in range(0, h.shape[0], 128):
                        hh = h[c0:c0 + 128].clone().requires_grad_(True)
                        z = torch.tanh((hh @ hf.T) / 30.0) * 30.0
                        val = ((z.gather(1, targets[c0:c0 + 128].long()[:, None])[:, 0] - torch.logsumexp(z, -1)) * gg[c0:c0 + 128]).sum()
                        out[c0:c0 + 128] = torch.autograd.grad(val, hh)[0]
                del hf
                return out

            G_row, g_row = run_backward({"host": host_p, "routing": rout_p, "hidden": hid_d, "lp": None}, ids_p, g_p, TL.qb_for(Tp),
                                        head_grad=head_exact)
            resA = {"T": Tp, "S": S, "seconds": time.time() - t0,
                        "per_visit": {str(v): cmp(g_row[v], g_auto[v]) for v in sorted(g_row) if v in g_auto},
                        "tensors": {nm: cmp(G_row[nm], G_auto[nm]) for nm in G_auto if nm in G_row}}
            bad = [v for v, x in resA["per_visit"].items() if x["cos"] < 0.999 or not 0.99 <= x["norm_ratio"] <= 1.01]
            badt = [nm for nm, x in resA["tensors"].items() if x["cos"] < 0.999 or not 0.99 <= x["norm_ratio"] <= 1.01]
            resA["pass"] = not bad and not badt
            resA["outside"] = {"visits": bad, "tensors": badt}
            log(f"A: {'PASS' if rep['A']['pass'] else 'FAIL'}; outside visits {bad}, tensors {badt}; per visit cos min "
                f"{min(x['cos'] for x in rep['A']['per_visit'].values()):.6f}; {rep['A']['seconds']:.0f} s")
            del host_p, hid_d, g_auto, G_auto
        except Exception as e:
            import traceback
            resA = {"error": traceback.format_exc()[-2000:], "pass": None, "flag": "A did not run"}
            log(f"A ERROR (A did not run; B and C go on): {type(e).__name__}: {e}")
        return resA

    if "A" in a.parts:
        rep["A"] = part_A()
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    else:
        rep["A"] = {"pass": None, "flag": "A runs in its own process (diag5A.json)"}
    RA.ROUND["on"] = False
    torch.cuda.empty_cache()
    save()
    if a.parts == "A":
        rig.w._release(ctx)
        rig.w.close()
        return 3 if rep["A"].get("pass") is False else 0
    if rep["A"].get("pass") is False:
        rep["dropped"].append("B and C (A's line failed)")
        save()
        rig.w._release(ctx)
        rig.w.close()
        return 0

    # ---------------------------------------------------------------- the chains' states (full row)
    outs_s, hid_s, outs_f, hid_f = [], [], [], []
    try:
        chain_S(rig, ids, x0, routing0, mask_t, keep_outs=outs_s, keep_hidden=hid_s)
        RA.ROUND["on"] = True
        chain_S(rig, ids, x0, routing0, mask_t, keep_outs=outs_f, keep_hidden=hid_f)
        RA.ROUND["on"] = False
        rep["smooth_vs_frozen_states"] = [float((outs_s[v].double() - outs_f[v].double()).norm() / outs_f[v].double().norm())
                                          for v in range(len(outs_s))]
    except Exception as e:
        rep["chains_error"] = f"{type(e).__name__}: {e}"[:1500]
        log(f"chains ERROR: {rep['chains_error'][:200]}")
        save()
        rig.w._release(ctx)
        rig.w.close()
        return 4

    def chain_vals(outs, hid):
        h = {-1: x0}
        for v, o in enumerate(outs):
            h[v] = o
        return {"host": h, "routing": routing0, "hidden": hid[0], "lp": None}

    G_full_E, _ = run_backward(ctx["res"], ids, g_full, qb)
    rep["norms_full_engine"] = {nm: float(t.norm()) for nm, t in G_full_E.items()}
    G_S, g_S = run_backward(chain_vals(outs_s, hid_s), ids, g_full, qb)
    RA.ROUND["on"] = True
    G_F, _ = run_backward(chain_vals(outs_f, hid_f), ids, g_full, qb)
    RA.ROUND["on"] = False
    rep["G_F_vs_G_S"] = {nm: cmp(G_F[nm], G_S[nm]) for nm in G_S}
    log("G_F vs G_S: " + ", ".join(f"{nm} {x['cos']:.3f}" for nm, x in rep["G_F_vs_G_S"].items()))
    save()

    def engine_follow(Gt, nm, mult):
        L, k = nm.split(".")
        gC = Gt.double()
        m_ = mult
        if nm != "21.B13" and float(gC.norm()) * BASE_E * m_ * gC.numel() ** 0.5 < 3.0:
            m_ = 16.0
        step = (gC / gC.norm() * BASE_E * m_ * gC.numel() ** 0.5).float()
        pred = float((gC * step.double()).sum())
        sv = {}
        for sg in (1, -1):
            rig.set_one(int(L), k, sg * step)
            with torch.no_grad():
                res = F.forward(eng, ids, save=False, adapters_per_layer=rig.w._adapters())
                lpx = F.logprobs(eng, res["hidden"], ids)
            sv[sg] = float(lpx.double()[mask_t[1:]].sum())
            del res, lpx
            eng._ws = None
        rig.set_one(int(L), k, None)
        return {"tensor": nm, "mult": m_, "pred": pred, "S+": sv[1], "S-": sv[-1], "f": (sv[1] - sv[-1]) / (2 * pred)}

    # ---------------------------------------------------------------- B, truncation first
    rep["B"] = {}
    for K in (18, 24, 30):
        if time.time() > deadline - 60:
            rep["dropped"].append(f"B K{K}")
            continue
        vf = 48 - K
        try:
            Ge, ge = run_backward(ctx["res"], ids, g_full, qb, v_first=vf)
            Gc, gc = run_backward(chain_vals(outs_s, hid_s), ids, g_full, qb, v_first=vf)
            r = {"v_first": vf, "engine_vs_chain": {nm: cmp(Ge[nm], Gc[nm]) for nm in ("12.B13", "12.qB", "21.B13") if nm in Ge and nm in Gc},
                 "per_visit": {str(v): cmp(ge[v], gc[v]) for v in sorted(ge) if v in gc},
                 "norm_12B13_engine": float(Ge["12.B13"].norm()) if "12.B13" in Ge else None,
                 "share_of_full_norm": {nm: float(Ge[nm].norm()) / rep["norms_full_engine"][nm] for nm in ("12.B13", "12.qB") if nm in Ge}}
            r["engine_follow"] = [engine_follow(Ge["12.B13"], "12.B13", 4.0), engine_follow(Ge["12.qB"], "12.qB", 4.0)]
            if K == 18:
                r["engine_follow"].append(engine_follow(Ge["21.B13"], "21.B13", 1.0))
            rep["B"][f"K{K}"] = r
            log(f"B K{K}: cos engine vs chain 12.B13 {r['engine_vs_chain'].get('12.B13', {}).get('cos')}, |12.B13| {r['norm_12B13_engine']}; "
                + "; ".join(f"engine follow {x['tensor']} x{x['mult']:g} pred {x['pred']:.4g} f {x['f']:.4f}" for x in r["engine_follow"]))
        except Exception as e:
            import traceback
            rep["B"][f"K{K}"] = {"error": traceback.format_exc()[-1500:]}
            log(f"B K{K} ERROR: {type(e).__name__}: {e}")
        save()
    for var in ("Q", "M", "D", "QD"):
        if time.time() > deadline - 120:
            rep["dropped"].append(f"B {var}")
            continue
        undo = []
        try:
            undo = variant_patches(var)
            Ge, ge = run_backward(ctx["res"], ids, g_full, qb)
            Gc, gc = run_backward(chain_vals(outs_s, hid_s), ids, g_full, qb)
        except Exception as e:
            import traceback
            rep["B"][var] = {"error": traceback.format_exc()[-1500:]}
            log(f"B {var} ERROR: {type(e).__name__}: {e}")
            continue
        finally:
            for u in reversed(undo):
                u()
        r = {"engine_vs_chain": {nm: cmp(Ge[nm], Gc[nm]) for nm in ("12.B13", "3.B13", "21.B13", "12.qB", "3.qB", "21.qB") if nm in Ge},
             "per_visit": {str(v): cmp(ge[v], gc[v]) for v in sorted(ge) if v in gc},
             "norms_engine": {nm: float(t.norm()) for nm, t in Ge.items()}}
        cands = [nm for nm in ("12.B13", "3.B13") if r["engine_vs_chain"][nm]["cos"] > 0.8]
        r["candidate_layers"] = cands
        log(f"B {var}: cos engine vs chain 12.B13 {r['engine_vs_chain']['12.B13']['cos']:.3f}, 3.B13 {r['engine_vs_chain']['3.B13']['cos']:.3f}, "
            f"21.B13 {r['engine_vs_chain']['21.B13']['cos']:.3f}; candidate layers {cands}")
        r["engine_follow"] = []
        for nm in cands:
            if time.time() > deadline:
                rep["dropped"].append(f"B {var} follow {nm}")
                continue
            x = engine_follow(Ge[nm], nm, 4.0)
            r["engine_follow"].append(x)
            log(f"B {var} engine follow {nm} x{x['mult']:g}: pred {x['pred']:.4g}, f {x['f']:.4f}")
        rep["B"][var] = r
        save()

    # ---------------------------------------------------------------- C
    rep["C"] = []
    for mult in (0.25, 0.0625, 0.015625):
        if time.time() > deadline:
            rep["dropped"].append(f"C x{mult:g}")
            continue
        gC = G_S["12.B13"].double()
        step = (gC / gC.norm() * BASE_E * mult * gC.numel() ** 0.5).float()
        pred = float((gC * step.double()).sum())
        rig.set_one(12, "B13", step)
        sp = chain_S(rig, ids, x0, routing0, mask_t)[0]
        rig.set_one(12, "B13", -step)
        sm = chain_S(rig, ids, x0, routing0, mask_t)[0]
        rig.set_one(12, "B13", None)
        rep["C"].append({"mult": mult, "pred": pred, "S+": sp, "S-": sm, "f": (sp - sm) / (2 * pred)})
        log(f"C 12.B13 along G_S x{mult:g}: pred {pred:.4g}, S+ {sp:.6f}, S- {sm:.6f}, f {(sp - sm) / (2 * pred):.4f}")
        save()
    save()
    rig.w._release(ctx)
    rig.w.close()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("grad", "fd", "inject", "local", "lin", "sum"))
    ap.add_argument("--plan", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--budget-s", type=float, default=1500.0)
    ap.add_argument("--parts", default="ABC", help="phase sum: A (its own process) or BC")
    ap.add_argument("--prefix", type=int, default=512, help="phase sum part A: prefix tokens")
    a = ap.parse_args()
    return {"grad": phase_grad, "fd": phase_fd, "inject": phase_inject, "local": phase_local, "lin": phase_lin, "sum": phase_sum}[a.phase](a)


if __name__ == "__main__":
    sys.exit(main())
