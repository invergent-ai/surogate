"""Update rules for the step session: what is added to the masters for one update, given the update's gradient and the optimizer's own step.

The optimizer (the kit's fp32 AdamW on the masters, global clip first) always runs; its step d_A and its moments are the reference. A rule
keeps d_A's size and changes its direction:
  A  the optimizer's step itself (today's rule).
  P  gradient-proportional:            d = -k g,  k such that |d|_2 = |d_A|_2 over all tensors together.
  T  proportional inside each tensor:  d[t] = -g[t] |d_A[t]|_2 / |g[t]|_2 for every tensor t with |g[t]|_2 > 0, zero elsewhere
     (the optimizer's norms across tensors, the gradient's direction inside each).
  E  Adam's direction at a larger eps: d = -k u,  u = m_hat / (sqrt(v_hat) + eps) from the optimizer's moments AFTER its step (at the first
     step u = g / (|g| + eps)),  k such that |d|_2 = |d_A|_2.
g is the gradient the optimizer saw: the update's gradient times the clip coefficient. Every norm and product is taken in float64; the result is
rounded once to fp32 when it is added to the masters. A rule's size s multiplies d (s = 1: the rule itself).
The optimizer's moments depend on the gradients only, so they are the same under every rule: a checkpoint's optimizer state stays valid when its
masters are rebuilt under another rule (rebuild_checkpoint).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

RULES = ("A", "P", "T", "E")
EXPERT_KEYS = ("A13", "B13", "A2", "B2")


def parse_state(spec: str):
    """"P:8" -> ("P", 8.0, None); "E:1:1e-6" -> ("E", 1.0, 1e-6). The size defaults to 1, E's eps to 1e-6."""
    parts = spec.split(":")
    rule = parts[0]
    if rule not in RULES or len(parts) > 3:
        raise ValueError(f"rule state {spec!r}: <rule>[:<size>[:<eps>]] with rule one of {RULES}")
    size = float(parts[1]) if len(parts) > 1 and parts[1] else 1.0
    eps = float(parts[2]) if len(parts) > 2 else (1e-6 if rule == "E" else None)
    if not size > 0 or (rule == "E" and not eps > 0) or (rule != "E" and len(parts) > 2):
        raise ValueError(f"rule state {spec!r}")
    return rule, size, eps


def _norm(xs) -> float:
    return float(np.sqrt(sum(float((x.double() * x.double()).sum()) for x in xs)))


def adam_direction(m: list, v: list, step: float, betas=(0.9, 0.999), eps: float = 1e-8) -> list:
    """u = m_hat / (sqrt(v_hat) + eps), float64, from the optimizer's moments after `step` steps (torch.optim.AdamW's arithmetic)."""
    b1, b2 = 1.0 - betas[0] ** step, 1.0 - betas[1] ** step
    return [(mi.double() / b1) / ((vi.double() / b2).sqrt() + eps) for mi, vi in zip(m, v)]


def rule_delta(rule: str, grads: list, coef: float, d_adam: list, size: float = 1.0, eps: float | None = None, moments=None) -> list:
    """The rule's delta (float64 tensors, one per master). grads: the update's gradient before the clip; coef: the clip coefficient;
    d_adam: the optimizer's own step (masters after - before); moments = (m, v, step, betas) for rule E."""
    if rule not in RULES:
        raise ValueError(f"rule {rule!r}")
    dA = [d.double() for d in d_adam]
    if rule == "A":
        out = dA
    elif rule == "P":
        g = [x.double() * coef for x in grads]
        ng = _norm(g)
        k = _norm(dA) / ng if ng > 0 else 0.0
        out = [-k * x for x in g]
    elif rule == "T":
        out = []
        for x, d in zip(grads, dA):
            gx = x.double() * coef
            ng = float(gx.norm())
            out.append(-gx * (float(d.norm()) / ng) if ng > 0 else torch.zeros_like(gx))
    else:
        if moments is None or eps is None:
            raise ValueError("rule E needs the optimizer's moments and an eps")
        m, v, step, betas = moments
        u = adam_direction(m, v, step, betas, eps)
        nu = _norm(u)
        k = _norm(dA) / nu if nu > 0 else 0.0
        out = [-k * x for x in u]
    return [x * float(size) for x in out] if size != 1.0 else out


def add_delta(base: list, delta: list) -> list:
    """fp32(float64(base) + delta), one rounding; an element whose delta is exactly zero keeps the base's bits (a start adapter may hold
    negative zeros, and -0.0 + 0.0 would turn them into +0.0: the same value, another file)."""
    return [torch.where(d == 0, b, (b.double() + d).float()) for b, d in zip(base, delta)]


def table_row(name: str, delta: list, d_adam: list, nm: list, blocks: dict) -> dict:
    """What a reader needs to check a rule before any card is used: the delta's L2 norm overall, per layer block and kind (attention / experts),
    the number of non-zero entries, and its cosine with the optimizer's step."""
    dA = [d.double() for d in d_adam]
    dot = sum(float((x * y).sum()) for x, y in zip(delta, dA))
    nd, na = _norm(delta), _norm(dA)
    by = {}
    for x, n in zip(delta, nm):
        L, key = n.split(".", 1)
        blk = next(b for b, rng in blocks.items() if int(L) in rng)
        kind = "experts" if key in EXPERT_KEYS else "attention"
        for k in (blk, kind, f"{blk}.{kind}"):
            by[k] = by.get(k, 0.0) + float((x * x).sum())
    return {"state": name, "l2": nd, "l2_adam_step": na, "cos_with_adam_step": dot / (nd * na) if nd > 0 and na > 0 else 0.0,
            "nonzero_entries": int(sum(int((x != 0).sum()) for x in delta)), "entries": int(sum(x.numel() for x in delta)),
            "nonzero_tensors": int(sum(1 for x in delta if bool((x != 0).any()))),
            "l2_by_block": {k: float(np.sqrt(v)) for k, v in sorted(by.items())}}


def per_tensor_norms(delta: list, nm: list) -> dict:
    return {n: float(x.norm()) for x, n in zip(delta, nm)}


def load_first_update(ckpt_dir: str, grad_path: str, sha256, state_sha):
    """A checkpoint of the FIRST update from the zero adapter, with its gradient file. -> dict with theta0, theta1, d_adam, grads, coef, moments,
    names, meta. theta0 is rebuilt from the checkpoint: every A as it is (at the first update from B = 0 no A has a gradient: checked on the
    optimizer's first moment, which must be exactly zero for every A), every B zero, an unmoved B element copied with its bits (the start
    adapter's zeros may be negative zeros, so theta0 cannot be checked by its hash; its VALUES are exact). The optimizer state must be the
    first step's, the gradient's hash the one grad.json records, and the first moment (1 - beta1) x the clipped gradient."""
    meta = json.load(open(os.path.join(ckpt_dir, "meta.json")))
    for name in ("masters.pt", "optimizer.npz"):
        if sha256(os.path.join(ckpt_dir, name)) != meta["sha256"][name]:
            raise ValueError(f"{ckpt_dir}: {name} does not match the hash in meta.json")
    saved = torch.load(os.path.join(ckpt_dir, "masters.pt"), map_location="cpu", weights_only=True)
    nm = list(saved)
    theta1 = [saved[n].float() for n in nm]
    is_b = [n.split(".", 1)[1].endswith("B") or n.split(".", 1)[1] in ("B13", "B2") for n in nm]
    theta0 = [torch.where(t == 0, t, torch.zeros_like(t)) if b else t.clone() for t, b in zip(theta1, is_b)]
    gj = json.load(open(os.path.splitext(grad_path)[0] + ".json"))
    if sha256(grad_path) != gj["sha256"] or int(gj["update"]) != int(meta["update"]):
        raise ValueError("the gradient file is not the one its record names, or it is another update's")
    gs = torch.load(grad_path, map_location="cpu", weights_only=True)
    if list(gs) != nm:
        raise ValueError("the gradient file's tensors are not the checkpoint's")
    grads = [gs[n].float() for n in nm]
    coef = float(gj["clip_coefficient"])
    with np.load(os.path.join(ckpt_dir, "optimizer.npz"), allow_pickle=False) as z:
        m = [torch.from_numpy(np.array(z[f"{i}.exp_avg"])) for i in range(len(nm))]
        v = [torch.from_numpy(np.array(z[f"{i}.exp_avg_sq"])) for i in range(len(nm))]
        steps = {float(z[f"{i}.step"][0]) for i in range(len(nm))}
    if steps != {1.0}:
        raise ValueError(f"the optimizer state is not the first step's (steps {sorted(steps)})")
    if any(bool(mi.ne(0).any()) for mi, b in zip(m, is_b) if not b):
        raise ValueError("an A tensor has a gradient in this checkpoint: it is not the first update from the zero adapter")
    betas = tuple(float(b) for b in meta["optimizer"]["betas"])
    worst = 0.0                                               # the moments must be this gradient's: m = (1 - beta1) coef g
    for g, mi in zip(grads, m):
        want = g.double() * coef * (1.0 - betas[0])
        den = float(want.abs().max())
        if den > 0:
            worst = max(worst, float((mi.double() - want).abs().max()) / den)
    if worst > 1e-5:
        raise ValueError(f"the optimizer's first moment is not (1 - beta1) x the clipped gradient (worst relative difference {worst:.2e})")
    return {"meta": meta, "names": nm, "theta0": theta0, "theta1": theta1, "d_adam": [a - b for a, b in zip(theta1, theta0)], "grads": grads,
            "coef": coef, "moments": (m, v, 1.0, betas), "grad_sha256": gj["sha256"], "moment_check_worst_rel": worst}
