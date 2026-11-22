"""Lane T8, U1 CPU test (pass line in lanes/T8/PLAN.md U1): the row-parallel step on lane RL's fake Inkling against the kit's own run_step.
 1. one worker: gradients BITWISE equal to run_step, metrics equal; 2. two / three workers: metrics equal, gradients within 1e-5, A/A bitwise;
 3. refusals (long row, off-policy row) as run_step; 4. gates: a changed log-prob, changed routing and a non-finite gradient each raise and write STOP;
 5. seeds="weights": gradient of sum w log pi, sense "ascent"; 6. the visit tables (kept outputs, accumulators) and the host buffers' layout.
"""
import copy
import math
import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from surogate.inkling_rl.backend_fake import FakeInkling  # noqa: E402
from surogate.inkling_rl.records import Row, replay_rows  # noqa: E402
from surogate.inkling_rl.step import StepConfig, run_step  # noqa: E402
import t8_lean as TL  # noqa: E402
import t8_step as TS  # noqa: E402

KEYS = TS.LORA_KEYS


class FakeWorker(TS.WorkerCore):
    def __init__(self, cfg, flat, layout):
        self.be = FakeInkling(seed=0, rank=4, alpha=8)
        self.setup_sums(layout, flat, cfg)
        self.break_grad = False

    def _forward(self, row):
        rp, _ = replay_rows(row, self.be.visits, self.be.K)
        lp, cap = self.be.forward(torch.as_tensor(row.ids), replay=rp, lora=self.be.lora)
        return lp[1:].detach().numpy().astype(np.float32), {"lp": lp, "cap": cap}

    def _check_routing(self, ctx, rp):
        cap = ctx["cap"].numpy()
        return int(((cap != rp.ids).any(-1).any(-1) & rp.covered).sum())

    def _support(self, ctx, row):
        raise AssertionError("the fake has no centred head")

    def _backward_part(self, ctx, row, g_lp, sup, part):
        for p in self.be.parameters():
            p.grad = None
        (g_lp * ctx["lp"][1:]).sum().backward(retain_graph=True)
        pg, ps = {}, {}
        for L, d in enumerate(self.be.lora):
            for j, k in enumerate(KEYS):
                g = d[k].grad.detach().clone()
                if self.break_grad:
                    g[0] = float("nan")
                pg[(L, j)] = g
                if j in self.seg_keys:
                    ps[(part, "A", L, k)] = g.clone()
        return pg, ps, 0, 128


def make_rows(n, seed, version=3):
    be = FakeInkling(seed=0, rank=4, alpha=8)
    g = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        T = int(g.integers(12, 40))
        ids = g.integers(0, be.V, T).astype(np.int64)
        with torch.no_grad():
            lp, cap = be.forward(torch.as_tensor(ids), lora=be.lora)
        mask = np.zeros(T, dtype=bool)
        mask[int(g.integers(3, T // 2 + 3)):] = True
        v = np.full(T, -1, dtype=np.int32)
        v[mask] = version
        rows.append(Row(ids=ids, loss_mask=mask, engine_lp=lp.double().numpy() * mask, advantage=float(g.normal()), versions=v, teacher_lp=None,
                        routed=cap.numpy().astype(np.int16), trial_idx=i, group_idx=i % 3, notes={"weight": float(g.normal())}))
    return rows


def reference(rows, version, cfg):
    be = FakeInkling(seed=0, rank=4, alpha=8)
    res = run_step(be, copy.deepcopy(rows), version, cfg, log=lambda *_: None)
    return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in be.parameters()], res.metrics


def trainer_parts(n_workers, cfg):
    be = FakeInkling(seed=0, rank=4, alpha=8)
    shapes = [{k: tuple(d[k].shape) for k in KEYS} for d in be.lora]
    layout = TS.ParamLayout(shapes, list(range(len(shapes))), cfg.train_keys)
    flats = [torch.zeros(layout.size, dtype=torch.float32) for _ in range(n_workers)]
    workers = [FakeWorker(cfg, f, layout) for f in flats]
    return [TS.InlineHandle(w) for w in workers], layout, flats, shapes, workers


def rel(a, b):
    return float((a.double() - b.double()).norm() / max(float(b.double().norm()), 1e-30))


def main():
    version = 3
    rows = make_rows(14, 1, version)
    # a worker takes its rows longest first; the reference gets them in that same order, so that "bitwise" compares the same fp32 summation
    rows = [rows[i] for i in sorted(range(len(rows)), key=lambda j: (-rows[j].T, j))]
    scfg = StepConfig(max_policy_lag=1)
    g_ref, m_ref = reference(rows, version, scfg)
    tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    cfg = TS.T8Config(head="plain", max_row_tokens=10 ** 6, segment_keys=("B13", "B2"), run_dir=tmp, visits=5, top_k=2)
    runs = {}
    for n in (1, 2, 3):
        h, lay, fl, sh, _ = trainer_parts(n, cfg)
        runs[n] = TS.compute_step(h, lay, fl, sh, cfg, copy.deepcopy(rows), version, scfg, "gspo")
        assert runs[n].sense == "descent"
        for k in ("engine_mismatch_kl", "engine_log_ratio", "policy_loss", "mismatch_kl", "is_masked", "total_tokens", "expected_loss_scale", "r3_rows",
                  "r3_replayed_tokens", "offpolicy_dropped_samples", "refused_long_rows", "engine_behavior_w_tokens"):
            a, b = runs[n].metrics.get(k), m_ref.get(k)
            assert (a is None and b is None) or math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12), (n, k, a, b)
        for k in ("engine_behavior_w_mean", "engine_behavior_w_min", "engine_behavior_w_max", "engine_behavior_w_capped"):
            assert math.isclose(float(runs[n].metrics[k]), float(m_ref[k]), rel_tol=1e-6, abs_tol=1e-9), (n, k)
        worst = max(rel(a, b) for a, b in zip(runs[n].grads, g_ref))
        print(f"workers {n}: largest relative difference to run_step {worst:.2e}")
        if n == 1:
            assert all(torch.equal(a, b) for a, b in zip(runs[n].grads, g_ref)), "one worker must equal run_step bitwise"
        assert worst <= 1e-5
        h, lay, fl, sh, _ = trainer_parts(n, cfg)
        again = TS.compute_step(h, lay, fl, sh, cfg, copy.deepcopy(rows), version, scfg, "gspo")
        assert all(torch.equal(a, b) for a, b in zip(again.grads, runs[n].grads)), "A/A"
        assert set(again.segments) == set(runs[n].segments) and all(torch.equal(again.segments[k], runs[n].segments[k]) for k in again.segments)
    # the segment sums are the gradient of their keys (the fake reports one segment)
    for L in range(3):
        for j, k in ((1, "B13"), (3, "B2")):
            assert rel(runs[2].segments[("main", "A", L, k)], runs[2].grads[L * 4 + j]) < 1e-5
    print("step == run_step: PASS")

    # refusals, in run_step's way
    r2 = copy.deepcopy(rows)
    r2[-1].versions[r2[-1].loss_mask] = version + 5              # off-policy (the shortest row; the longest is the refused one)
    long_T = max(r.T for r in r2)
    assert r2[-1].T < long_T
    cfg2 = TS.T8Config(head="plain", max_row_tokens=long_T - 1, run_dir=tmp, visits=5, top_k=2)
    g2, m2 = reference(r2, version, StepConfig(max_policy_lag=1, max_row_tokens=long_T - 1))
    h, lay, fl, sh, _ = trainer_parts(2, cfg2)
    o2 = TS.compute_step(h, lay, fl, sh, cfg2, copy.deepcopy(r2), version, StepConfig(max_policy_lag=1), "gspo")
    for k in ("offpolicy_dropped_samples", "refused_long_rows", "total_tokens", "expected_loss_scale"):
        assert float(o2.metrics[k]) == float(m2[k]), k
    assert m2["refused_long_rows"] >= 1 and m2["offpolicy_dropped_samples"] == 1
    assert max(rel(a, b) for a, b in zip(o2.grads, g2)) <= 1e-5
    print("refusals: PASS")

    # gates
    def gate(mutate, want, n=2, wbreak=False):
        stop = os.path.join(tmp, "STOP")
        if os.path.exists(stop):
            os.remove(stop)
        r = copy.deepcopy(rows)
        mutate(r)
        h, lay, fl, sh, ws = trainer_parts(n, cfg)
        for w in ws:
            w.break_grad = wbreak
        try:
            TS.compute_step(h, lay, fl, sh, cfg, r, version, scfg, "gspo")
        except TS.T8GateError as e:
            assert want in str(e), str(e)
            assert os.path.exists(stop), "no STOP file"
            return
        raise AssertionError(f"gate {want} did not fire")

    def bad_lp(r):
        t = int(np.flatnonzero(r[5].loss_mask)[0])
        r[5].engine_lp[t] = np.nextafter(np.float32(r[5].engine_lp[t]), np.float32(0.0))

    def bad_route(r):
        x = r[4].routed.copy()
        x[1, 0, 0] = (x[1, 0, 0] + 1) % 8
        r[4].routed = x

    gate(bad_lp, "REPLAY GATE")
    gate(bad_route, "ROUTING GATE")
    gate(lambda r: None, "FINITE GATE", wbreak=True)
    print("gates: PASS")

    # seeds = "weights": the gradient of sum_row weight x sum_t log pi over the loss tokens
    h, lay, fl, sh, _ = trainer_parts(3, cfg)
    ow = TS.compute_step(h, lay, fl, sh, cfg, copy.deepcopy(rows), version, scfg, "weights")
    assert ow.sense == "ascent"
    be = FakeInkling(seed=0, rank=4, alpha=8)
    tot = 0.0
    for r in rows:
        rp, _ = replay_rows(r, be.visits, be.K)
        lp, _ = be.forward(torch.as_tensor(r.ids), replay=rp, lora=be.lora)
        tot = tot + float(r.notes["weight"]) * (torch.as_tensor(r.loss_mask.astype(np.float32)) * lp).sum()
    tot.backward()
    assert max(rel(a, p.grad) for a, p in zip(ow.grads, be.parameters())) <= 1e-5
    print("weights: PASS")

    # the visit tables of the real model
    class V:
        pass

    def visits():
        out = []
        for layer, p in [(i, 0) for i in range(6)] + [(i, q) for q in range(3) for i in range(6, 18)] + [(i, 0) for i in range(18, 24)]:
            v = V()
            v.layer, v.looped, v.first = layer, 6 <= layer < 18 and p >= 1, layer % 6 == 0
            if layer < 6:
                v.stack, v.partial, v.next_partial = ((), 0, 1) if layer == 0 else ((0,), 1, 1)
                v.bank = ()
            elif layer < 18:
                b = 2 if p % 2 == 0 else 4
                if layer < 12:
                    v.stack, v.partial, v.next_partial = ((0,), 1, b) if layer == 6 else ((0, 1), b, b)
                else:
                    v.stack, v.partial, v.next_partial = ((0, 1), b, b + 1) if layer == 12 else ((0, 1, b), b + 1, b + 1)
                q0 = 2 if (p - 1) % 2 == 0 else 4
                v.bank = (q0, q0 + 1) if p >= 1 else ()
            else:
                v.stack, v.partial, v.next_partial = ((0, 1, 2), 3, 6) if layer == 18 else ((0, 1, 2, 3), 6, 6)
                v.bank = ()
            out.append(v)
        return out

    vs = visits()
    assert [len(TL.keep_outputs(vs, 48 - K)) + 1 for K in (6, 18, 30, 48)] == [10, 22, 34, 49]
    assert [TL.acc_peak(vs, 48 - K) for K in (6, 18, 30, 48)] == [0, 2, 3, 4]
    assert TL.trained_layers(vs, 30) == list(range(6, 24)) and TL.trained_layers(vs, 42) == list(range(18, 24))
    assert [TS.SimpleVisit(v).layer for v in range(48)] == [x.layer for x in vs]
    assert [TL.segment_of(v) for v in (0, 5, 6, 17, 18, 29, 30, 41, 42, 47)] == ["D0", "D0", "D1", "D1", "C", "C", "B", "B", "A", "A"]
    assert [TL.qb_for(t) for t in (32768, 32769, 50000, 50001, 100001, 200001)] == [128, 64, 64, 32, 16, 8]
    b = TL.RowBuffers(vs, 30, 100)
    o = b.outputs(100)
    assert len(o) == 22
    for x in o.values():
        x.fill_(0.0)
    for j, (v, x) in enumerate(sorted(o.items())):
        x[:] = float(j + 1)
    k1, a1 = b.acc(100)
    k2, a2 = b.acc(60)
    assert k1 != k2 and float(a1.abs().sum()) == 0 and a2.shape == (60, TL.D)
    a1.fill_(7.0)
    a2.fill_(9.0)
    for j, (v, x) in enumerate(sorted(o.items())):
        assert bool((x.float() == j + 1).all()), f"output {v} was overwritten"
    assert bool((a1[60:] == 7.0).all())
    b.release(k1)
    k3, a3 = b.acc(100)
    assert k3 == k1 and float(a3.abs().sum()) == 0, "a reused accumulator is zero"
    try:
        b.outputs(101)
        raise AssertionError("a row longer than the buffers must be refused")
    except ValueError:
        pass
    b.close()
    assert sorted(sum(TS.assign_rows({i: r.T for i, r in enumerate(rows)}, 3), [])) == list(range(len(rows)))
    assert TS.assign_rows({0: 5, 1: 5, 2: 9}, 2) == [[2], [0, 1]]
    print("tables and buffers: PASS")
    print("T8 STEP CPU TEST: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
