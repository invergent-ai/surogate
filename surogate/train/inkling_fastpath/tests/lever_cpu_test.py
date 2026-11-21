"""Lane T: CPU equivalence tests of the LEVEL-1 levers that are switched by a flag (no engine, no card).

  plan     _moe_plan covers every row exactly once, batch limits hold
  moe      MoEGM's vjp with MOE_FAST=1 vs the group-major autograd path (MOE_FAST=0) on random experts with a skewed
           routing: dX and all four LoRA gradients, at MOE_ROWS large (one block) and small (experts cut into items)
  attn     RelAttentionR with ATTN_TC=1 vs the einsum path (forward out / lse, backward dq / dk / dv / dr), local window
           and global, fixed and adaptive query blocks

On CPU out_dtype is normally unavailable, so the tensor-core calls run their fp32 fallback on the SAME hi / lo split: this
proves the shapes, indices and the vjp algebra. It does NOT prove the speed or the bf16 MMA arithmetic of the card; the
card check (tests/lever_check.py) does that.
Usage: python lever_cpu_test.py [--threads 8]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ift_backward as B  # noqa: E402
import ift_long as LG  # noqa: E402

D, I, E, TOPK, RANK = 2304, 1152, 128, 8, 16
BF16, F32 = torch.bfloat16, torch.float32


def rel(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300)), float(a @ b / (a.norm() * b.norm()).clamp_min(1e-300))


class FakeEng:
    def __init__(self, nexp, gen):
        self.u13 = (torch.randn(nexp, D, D, generator=gen) / 48).to(BF16)
        self.u2 = (torch.randn(nexp, D, I, generator=gen) / 48).to(BF16)
        self.f13 = torch.rand(nexp, generator=gen) + 0.5
        self.f2 = torch.rand(nexp, generator=gen) + 0.5

    def decode_experts(self, layer, which, first, count):
        u, f = (self.u13, self.f13) if which == "w13" else (self.u2, self.f2)
        return u[first:first + count], f[first:first + count]


class FakeW:
    def __init__(self, eng, gen):
        self.eng, self.layer, self.fmt = eng, 0, "nvfp4"
        self.gate_w = (torch.randn(E, D, generator=gen) / 48).to(BF16)


def skewed_ids(T, nexp, gen):
    p = 1.0 / torch.arange(1, nexp + 1, dtype=torch.float32) ** 1.2
    p = p[torch.randperm(nexp, generator=gen)]
    return torch.multinomial(p.expand(T, -1), TOPK, replacement=False, generator=gen).to(torch.int32)


def moe_grads(w, ids, x0, g, lora0, fast, rows, det=False):
    LG.MOE_FAST, LG.MOE_ROWS, LG.MOE_DET = fast, rows, det
    x = x0.clone().requires_grad_()
    lt = [t.clone().requires_grad_() for t in lora0]
    out = LG.MoEGM.apply(w, ids, 2.0, x, *lt)
    return torch.autograd.grad(out, [x] + lt, g)


def test_plan():
    gen = torch.Generator().manual_seed(1)
    for trial in range(300):
        n = 16
        cnt = (torch.rand(n, generator=gen) ** 6 * 5000).long().tolist()
        if trial % 7 == 0:
            cnt[3] = 0
        max_rows = [64, 256, 8192][trial % 3]
        nb = 1 + trial % 4
        plan = LG._moe_plan(cnt, max_rows, nb)
        seen = []
        for bt in plan:
            assert len(bt) <= B.EG
            assert all(bt[0][0] >= it[0] for it in bt), "first item of a batch is its longest"
            for ln, e, first in bt:
                assert 0 < ln <= max_rows and 0 <= e < n
                seen += list(range(first, first + ln))
        assert sorted(seen) == list(range(sum(cnt))), "every row exactly once"
        # each row's item belongs to its expert's row range
        st = [sum(cnt[:e]) for e in range(n)]
        for bt in plan:
            for ln, e, first in bt:
                assert st[e] <= first and first + ln <= st[e] + cnt[e]
    return {"trials": 300, "pass": True}


def test_moe():
    gen = torch.Generator().manual_seed(2)
    nexp = 64  # the first 4 groups: rows only there, so the fake experts stay ~2 GB
    eng = FakeEng(nexp, gen)
    w = FakeW(eng, gen)
    T = 640
    ids = skewed_ids(T, nexp, gen)
    x = (torch.randn(T, D, generator=gen)).to(BF16).float()
    g = torch.randn(T, D, generator=gen)
    lora = [torch.randn(E, RANK, D, generator=gen) / 48, torch.randn(E, 2 * I, RANK, generator=gen) * 1e-3,
            torch.randn(E, RANK, I, generator=gen) / 34, torch.randn(E, D, RANK, generator=gen) * 1e-3]
    res = {}
    names = ("dx", "A13", "B13", "A2", "B2")
    for rows in (8192, 128):
        ref = moe_grads(w, ids, x, g, lora, False, rows)
        new = moe_grads(w, ids, x, g, lora, True, rows)
        res[f"rows_{rows}"] = {n: {"rel": rel(a, b)[0], "cos": rel(a, b)[1]} for n, a, b in zip(names, new, ref)}
        for fast in (False, True):  # MOE_DET on both MoE paths
            new = moe_grads(w, ids, x, g, lora, fast, rows, det=True)
            res[f"rows_{rows}_det_fast{int(fast)}"] = {n: {"rel": rel(a, b)[0], "cos": rel(a, b)[1]}
                                                       for n, a, b in zip(names, new, ref)}
    LG.MOE_FAST, LG.MOE_ROWS, LG.MOE_DET = False, 8192, False
    worst = max(v["rel"] for r in res.values() for v in r.values())
    res["worst_rel"] = worst
    res["pass"] = worst < 2e-4
    return res


def attn_run(q0, k0, v0, r0, proj, win, qb, tc, do, dl):
    LG.ATTN_TC = tc
    q, k, v, r = [t.clone().requires_grad_() for t in (q0, k0, v0, r0)]
    out, lse = LG.RelAttentionR.apply(q, k, v, r, proj, win, qb)
    gr = torch.autograd.grad([out, lse], [q, k, v, r], [do, dl])
    return out.detach(), lse.detach(), gr


def test_attn():
    gen = torch.Generator().manual_seed(3)
    T = 400
    q = (torch.randn(T, 24, 128, generator=gen) * 0.6).to(BF16).float()
    k = (torch.randn(T, 8, 128, generator=gen) * 0.8).to(torch.float8_e4m3fn).float()
    v = (torch.randn(T, 8, 128, generator=gen) * 0.8).to(torch.float8_e4m3fn).float()
    r = (torch.randn(T, 24, 16, generator=gen) * 0.3).to(BF16).float()
    proj = (torch.randn(16, 1024, generator=gen) * 0.05).to(BF16)
    do = torch.randn(T, 24, 128, generator=gen)
    dl = torch.randn(T, 24, generator=gen)
    res = {}
    worst = 0.0
    for win, qb in ((64, 128), (64, 256), (0, 128), (0, 256)):
        o0, l0, g0 = attn_run(q, k, v, r, proj, win, qb, False, do, dl)
        o1, l1, g1 = attn_run(q, k, v, r, proj, win, qb, True, do, dl)
        row = {"out": rel(o1, o0)[0], "lse": rel(l1, l0)[0]}
        for n, a, b in zip(("dq", "dk", "dv", "dr"), g1, g0):
            row[n] = rel(a, b)[0]
        res[f"win{win}_qb{qb}"] = row
        worst = max(worst, max(row.values()))
    LG.ATTN_TC = False
    res["worst_rel"] = worst
    res["pass"] = worst < 5e-4
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", default="plan,attn,moe")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rep = {"out_dtype_on_cpu": B._probe_out_dtype(torch.device("cpu"))}
    B._OUT_DTYPE[torch.device("cpu")] = rep["out_dtype_on_cpu"]
    for name, fn in (("plan", test_plan), ("attn", test_attn), ("moe", test_moe)):
        if name in a.only.split(","):
            rep[name] = fn()
            print(name, json.dumps(rep[name]), flush=True)
    ok = all(rep[n]["pass"] for n in ("plan", "attn", "moe") if n in rep)
    print("ALL PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
