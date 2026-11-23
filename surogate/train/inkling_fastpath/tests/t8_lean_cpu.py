"""Lane T8, S4 (a) + (b) `lean_host` on CPU: t8_step.accumulate_row, the function the real worker uses.
 1. lean_host on == off, BITWISE, on Inkling's real visit -> layer table (looped layers visited 3 times), at K = 48 and K = 18;
 2. default path: an out-of-memory row is retried with the query block halved and gives the same sums as a clean run;
 3. lean_host: out of memory BEFORE the first addition -> retried (nothing added yet); AFTER it -> RuntimeError, no retry;
 4. end to end through compute_step / ppo_batch: the step raises, writes STOP, returns no gradient, and ppo_batch puts the masters and the
    optimizer state back as they were (never continues with a partial sum)."""
import copy
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from surogate.inkling_rl.step import StepConfig  # noqa: E402
import t8_lean as TL  # noqa: E402
import t8_ppo_cpu as P  # noqa: E402
import t8_step as TS  # noqa: E402

VL = [TS.SimpleVisit(v).layer for v in range(48)]


def fake_worker(lean, v_first, shapes, keys):
    layout = TS.ParamLayout(shapes, sorted(set(VL[v_first:])), keys)
    return SimpleNamespace(cfg=SimpleNamespace(lean_host=lean), layout=layout, flat=torch.zeros(layout.size), seg_keys=set(), v_first=v_first,
                           _segment_name=TL.segment_of)


def make_run(seed, v_first, shapes, oom_at=None, oom_once=True):
    state = {"fired": False}

    def run(on_grad, qb):
        g = torch.Generator().manual_seed(seed)
        for v in range(47, v_first - 1, -1):             # the backward's reverse order
            L = VL[v]
            for j, k in enumerate(TS.LORA_KEYS):
                on_grad(v, L, j, torch.randn(shapes[L][k], generator=g) * (1 + v))
            if oom_at is not None and v == oom_at and not (oom_once and state["fired"]):
                state["fired"] = True
                raise torch.OutOfMemoryError("fake")
    return run


def into_flat(w, pg):  # what process_row does with the default path's sums
    for (L, j), t in pg.items():
        w.layout.view(w.flat, L, j).add_(t)


def main():
    shapes = [{k: (3, 5) for k in TS.LORA_KEYS} for _ in range(24)]
    for K in (48, 18):
        v0 = 48 - K
        for seed in range(3):
            a = fake_worker(False, v0, shapes, TS.LORA_KEYS)
            b = fake_worker(True, v0, shapes, TS.LORA_KEYS)
            for row in range(2):                           # two rows into the same worker sum
                pg, _, _, _ = TS.accumulate_row(a, make_run(seed * 10 + row, v0, shapes), 1000, "main", 128, VL, lambda: None)
                into_flat(a, pg)
                pg2, _, _, _ = TS.accumulate_row(b, make_run(seed * 10 + row, v0, shapes), 1000, "main", 128, VL, lambda: None)
                assert pg2 == {}, "lean_host left sums behind"
            assert torch.equal(a.flat, b.flat), f"K {K}: lean_host is not bitwise"
    print("lean_host == default, bitwise (K = 48 and 18, looped layers visited 3 times): PASS")
    # retries
    a = fake_worker(False, 0, shapes, TS.LORA_KEYS)
    ref = fake_worker(False, 0, shapes, TS.LORA_KEYS)
    pg, _, r, qb = TS.accumulate_row(a, make_run(5, 0, shapes, oom_at=30), 1000, "main", 128, VL, lambda: None)
    pg_ref, _, _, _ = TS.accumulate_row(ref, make_run(5, 0, shapes), 1000, "main", 128, VL, lambda: None)
    assert r == 1 and qb == 64 and all(torch.equal(pg[k], pg_ref[k]) for k in pg_ref)
    print("default path: out-of-memory row retried at block 64, same sums: PASS")
    b = fake_worker(True, 0, shapes, TS.LORA_KEYS)
    # OOM at the first visit, before any layer's last visit (visit 47 is layer 23's only and last visit: its add happens AT visit 47, before the raise)
    try:
        TS.accumulate_row(b, make_run(6, 0, shapes, oom_at=47), 1000, "main", 128, VL, lambda: None)
        raise AssertionError("no error after a partial add")
    except RuntimeError as e:
        assert "not retried" in str(e)
    c = fake_worker(True, 30, shapes, TS.LORA_KEYS)
    # K = 18 from visit 30: OOM raised by a run that fails before calling on_grad at all -> retried
    calls = {"n": 0}

    def run_fail_first(on_grad, qb):
        calls["n"] += 1
        if calls["n"] == 1:
            raise torch.OutOfMemoryError("fake, before any gradient")
        make_run(7, 30, shapes)(on_grad, qb)
    pg, _, r, qb = TS.accumulate_row(c, run_fail_first, 1000, "main", 128, VL, lambda: None)
    assert r == 1 and qb == 64 and pg == {}
    print("lean_host: OOM after the first add -> RuntimeError, no retry; before it -> retried: PASS")

    # end to end: a worker that runs out of memory after its first add, inside ppo_batch
    class OOMW(P.FakeW):
        def _backward_part(self, ctx, row, g_lp, sup, part):
            first = [True]

            def run(on_grad, qb):
                if first[0]:
                    first[0] = False
                    for L in (2, 1):                       # layers visited once in the fake model: their adds happen, then the OOM
                        on_grad(L, L, 1, torch.ones(self.be.lora[L]["B13"].shape))
                    raise torch.OutOfMemoryError("fake")
            vl = [0, 1, 1, 1, 2]
            return TS.accumulate_row(self, run, row.T, part, 128, vl, lambda: None)

    tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    cfg = TS.T8Config(head="plain", max_row_tokens=10 ** 6, segment_keys=(), run_dir=tmp, visits=5, top_k=2, K=5, lean_host=True)
    tr = P.FakeTrainer(cfg, 2)
    tr.workers = [OOMW(cfg, f, tr.layout) for f in tr.flats]
    for w in tr.workers:
        w.v_first = 0
    tr.be = tr.workers[0].be
    tr.handles = [TS.InlineHandle(w) for w in tr.workers]
    tr.work = tr.be.parameters()
    tr.opt = P.MasterAdamW(tr.work, lr=P.LR, weight_decay=0.0, max_grad_norm=1.0)
    rows = P.make(3)
    ok = P.FakeTrainer(TS.T8Config(head="plain", max_row_tokens=10 ** 6, segment_keys=(), run_dir=tmp, visits=5, top_k=2, K=5), 2)
    TS.ppo_batch(ok, copy.deepcopy(rows), 3, P.LR, 1, StepConfig(max_policy_lag=None))   # a step that works, so the optimizer has state
    m0 = [m.detach().clone() for m in tr.opt.masters]
    try:
        TS.ppo_batch(tr, copy.deepcopy(rows), 3, P.LR, 2, StepConfig(max_policy_lag=None))
        raise AssertionError("the step went on")
    except RuntimeError as e:
        assert "not retried" in str(e), str(e)
    assert all(torch.equal(a, b) for a, b in zip(tr.opt.masters, m0)) and not tr.opt.opt.state, "masters / optimizer not restored"
    assert os.path.exists(os.path.join(tmp, "STOP")), "no STOP file"
    print("end to end: the step fails loudly, STOP written, masters and optimizer restored: PASS")
    # per-attempt sketches (accumulate_row + WorkerCore._sketch_row)
    keys = TS.param_keys(64)
    sh2 = [{k: (3, 5) for k in keys} for _ in range(24)]
    w = fake_worker(True, 0, sh2, keys)
    w.sk_visits, w.sk_stride, w._row_sketch = {45, 20}, 4, None

    def run_all(on_grad, qb, seed=9):
        g = torch.Generator().manual_seed(seed)
        rec = {}
        for v in range(47, -1, -1):
            L = VL[v]
            for j, k in enumerate(keys):
                x = torch.randn(sh2[L][k], generator=g)
                rec[(v, L, j)] = x.clone()
                on_grad(v, L, j, x)
        return rec
    recs = {}
    TS.accumulate_row(w, lambda og, qb: recs.update(run_all(og, qb)), 1000, "main", 128, VL, lambda: None)
    rs = w._row_sketch
    assert torch.equal(rs[("visit", 45, "qA")], recs[(45, VL[45], keys.index("qA"))]) and ("visit", 44, "qA") not in rs
    assert ("visit", 20, "A13") not in rs, "expert keys are sampled per layer, not per visit"
    L = VL[20]
    tot = sum(recs[(v, L2, 1)] for (v, L2, j) in recs if L2 == L and j == 1)
    assert torch.allclose(rs[("layer", L, "B13")], tot.reshape(-1)[::4], rtol=0, atol=1e-6)
    w2 = SimpleNamespace(sketch={}, sketch_meta={})
    row = SimpleNamespace(trial_idx=7, group_idx=1, advantage=-0.5, notes={})
    TS.WorkerCore._sketch_row(w2, row, {k: v.clone() for k, v in rs.items()}, "ppo", 200.0)
    assert torch.allclose(w2.sketch[7][("visit", 45, "qA")], rs[("visit", 45, "qA")] / (0.5 / 200.0))
    row0 = SimpleNamespace(trial_idx=8, group_idx=1, advantage=0.0, notes={})
    TS.WorkerCore._sketch_row(w2, row0, rs, "ppo", 200.0)
    assert w2.sketch_meta[8]["rows_skipped_zero_factor"] == 1 and 8 not in w2.sketch
    print("sketches: named visits' attention keys, expert samples of layer totals, divided by -A / N, A = 0 counted: PASS")
    print("T8 LEAN CPU TEST: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
