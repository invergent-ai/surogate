"""CPU checks of the step session (tests/t8_stepsession.py, tests/t8_stepsession_reads.py) on the fake model: the trainer side, the read side
against a fake serving engine, and a stub judge, as three parties that meet only through the run directory. Pass line:
(1) the plan loader accepts the agreed form and refuses: no "updates", a group in no update, a group twice, an order other than the rows',
    a round-trip group outside the plan, a row of another adapter version;
(2) a complete session of K = 2 updates (every hand-over, round trip, read and GO) leaves masters and optimizer state BITWISE equal to
    t8_step.ppo_batch(n_minibatches=2) on a new trainer; the controls have per tensor the accumulated update's norm and are zero where the
    update is zero; READS_DONE lists every read file with its hash; update 1 reads ratio 1 / clip 0 and update 2 a ratio that moved;
(3) resume: (A) a new trainer restored from update 1's checkpoint and continued with update 2 gives update 2's checkpoint (masters, optimizer
    state) and exported adapter BITWISE as the uninterrupted session; the restored state is read again and needs its own GO; (B) a next plan
    whose first update is number 3, resumed from update 2's checkpoint, ends BITWISE where one trainer running both plans in a row ends;
    a wrong old-log-prob file, a changed checkpoint file and a plan that does not follow the checkpoint are refused;
(4) verdicts: STOP and INCONCLUSIVE end the session with exit 20, nothing more applied, v labelled "not kept", last_adapter_above_line = n - 1;
    no verdict (timeout) and a verdict with a wrong hash end it with v labelled "not judged"; check_verdict refuses another session, another
    update, a file READS_DONE lists that the verdict does not cover, and a file changed after READS_DONE;
(5) round trip: an engine that serves the previous adapter's state for the next one (a leak), or one changed value, gives "differ", no
    READS_DONE, exit 21 on both sides and nothing more applied; a read side that dies gives exit 6;
(6) a STOP that arrives while the next gradient is being computed interrupts it (the gradient is not finished, nothing is applied);
(7) rows whose seeds are all zero: the step's gradient with their backward skipped is BITWISE the gradient with it run; they are counted;
    their attempts are in the sketch records as skipped.
"""
import copy
import json
import os
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from surogate.inkling_rl import adapter as AD  # noqa: E402
from surogate.inkling_rl.backend_fake import FakeInkling  # noqa: E402
from surogate.inkling_rl.step import StepConfig  # noqa: E402
import t8_micro as MI  # noqa: E402
import t8_micro_cpu as MC  # noqa: E402
import t8_ppo_cpu as P  # noqa: E402
import t8_step as TS  # noqa: E402
import t8_stepsession as SS  # noqa: E402
import t8_stepsession_reads as RD  # noqa: E402

KEYS = TS.LORA_KEYS
BLOCKS = {"before": range(0, 1), "inside": range(1, 2), "after": range(2, 3)}
CAP = 10 ** 6


def cfg_for(tmp, **kw):
    return TS.T8Config(head="plain", max_row_tokens=CAP, segment_keys=(), run_dir=tmp, visits=5, top_k=2, K=5, **kw)


def base_rows(n_groups=4):
    rows = P.make(3)[: 3 * n_groups]
    scores = ([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0] * n_groups)[: len(rows)]   # every group has both outcomes
    for k, (r, a) in enumerate(zip(rows, P.KS.grpo_advantages(scores, [r.group_idx for r in rows]))):
        r.trial_idx, r.advantage = k, float(a)
    return rows


def write_step_plan(d, rows, updates, extra=None, mutate=None, panel_like=False):
    MC.write_plan(d, rows, mutate)
    if not panel_like:
        meta = {"updates": updates}
        meta.update(extra or {})
        json.dump(meta, open(os.path.join(d, "rows.json"), "w"))
    return d


def load(d):
    return SS.load_step_plan(d, cap=CAP)


def fake_export(n_layers):
    def fn(state, dst, tag):
        layers = [{k: state[L * len(KEYS) + j].detach().numpy() for j, k in enumerate(KEYS)} for L in range(n_layers)]
        sha = AD.export_adapter(layers, dst, 4, 8, "BF16", metadata={"step": tag})
        torch.save([t.detach().clone() for t in state], os.path.join(dst, "state.pt"))
        return sha, {"r": 4, "lora_alpha": 8}
    return fn


class FakeEngines:
    """The serving engines of the test: one fake model per URL, one adapter slot. An adapter's work values are bf16(master), as the trainer's.
    leak: after an unload the NEXT loaded adapter is served with the previous one's values (a state carried over).
    flip: one log-prob of the named adapter changed by one ulp. die_at: raise in the n-th score call."""

    def __init__(self, leak=False, flip=None, die_at=None):
        self.models, self.slot, self.prev, self.lock = {}, {}, {}, threading.Lock()
        self.leak, self.flip, self.die_at, self.calls = leak, flip, die_at, 0
        self.base = [p.detach().clone() for p in FakeInkling(seed=0, rank=4, alpha=8).parameters()]

    def _model(self, url):
        with self.lock:
            if url not in self.models:
                self.models[url] = FakeInkling(seed=0, rank=4, alpha=8)
            return self.models[url]

    def load(self, url, name, path):
        st = torch.load(os.path.join(path, "state.pt"), weights_only=True)
        vals = [torch.from_numpy(AD.round_bf16(t.numpy()).reshape(tuple(t.shape))) for t in st]
        if self.leak and url in self.prev:
            vals = self.prev[url]
        self.slot[url] = (name, vals)

    def unload(self, url, name):
        assert self.slot[url][0] == name
        self.prev[url] = self.slot.pop(url)[1]

    def score(self, url, name, ids):
        with self.lock:
            self.calls += 1
            if self.die_at is not None and self.calls >= self.die_at:
                raise RuntimeError("the fake engine died")
        be = self._model(url)
        vals = self.base if name is None else self.slot[url][1]
        assert name is None or self.slot[url][0] == name
        with torch.no_grad():
            for p, v in zip(be.parameters(), vals):
                p.copy_(v)
            lp, _ = be.forward(torch.as_tensor(np.asarray(ids, dtype=np.int64)), replay=None, lora=be.lora)
        out = lp.numpy().astype(np.float32).copy()
        out[0] = 0.0
        if self.flip is not None and name is not None and name.endswith("_" + self.flip):
            out[-1] = np.nextafter(out[-1], np.float32(-np.inf))
        return out


class Parties:
    """The read side and the stub judge as threads beside a session that runs in the main thread."""

    def __init__(self, run, plan_dir, panels, transfer, engines, outcomes, first=1, urls=("e0", "e1")):
        self.run, self.th, self.reader_rc = run, [], None
        os.makedirs(run, exist_ok=True)
        open(os.path.join(run, "session_id.txt"), "w").write("cpu-test-" + os.path.basename(run))
        self.reader = RD.Reader(engines, list(urls), run, plan_dir, panels, transfer, engine_sha="fake", poll_s=0.02)
        self.outcomes, self.first = outcomes, first

    def _read(self):
        try:
            rc = self.reader.serve()
        except Exception as e:
            rc = RD.EXIT_READS if not isinstance(e, ValueError) else RD.EXIT_INPUT
        self.reader_rc = rc
        open(os.path.join(self.run, "reader.exit"), "w").write(str(rc))

    def _judge(self):
        SS.phase_stub_judge(SimpleNamespace(run_dir=self.run, outcomes=",".join(self.outcomes), first_update=self.first, deadline_min=2.0))

    def __enter__(self):
        self.th = [threading.Thread(target=self._read, daemon=True), threading.Thread(target=self._judge, daemon=True)]
        for t in self.th:
            t.start()
        return self

    def __exit__(self, *a):
        if not any(os.path.exists(os.path.join(self.run, f)) for f in ("TRAINER_DONE.json", "STOPPED.json")):
            open(os.path.join(self.run, "trainer.exit"), "w").write("x")
        for t in self.th:
            t.join(60)


def session(tmp, name, plan_dir, outcomes, tr=None, engines=None, first=1, transfer="", panels=None, workers=2, **opts):
    """One whole session on the fake. -> (exit code, run dir, trainer, session, parties)."""
    run = os.path.join(tmp, name)
    plan = load(plan_dir)
    tr = tr or P.FakeTrainer(cfg_for(tmp), workers)
    aborted = []
    s = SS.Session(tr, plan, run, os.path.join(run, "work"), "cpu-test-" + name, fake_export(len(tr.shapes)), BLOCKS,
                   verdict_timeout_s=opts.pop("verdict_timeout_s", 30.0), reads_timeout_s=opts.pop("reads_timeout_s", 30.0), poll_s=0.02,
                   abort_fn=lambda: aborted.append(1), close_fn=lambda: None, **opts)
    SS.install_signals(s)
    with Parties(run, plan_dir, panels or {}, transfer, engines or FakeEngines(), outcomes, first) as pt:
        rc = s.run_all()
    s.aborted = aborted
    return rc, run, tr, s, pt


def opt_state(tr):
    return {k: np.array(v) for k, v in tr.opt.state_arrays().items()}


def same_state(a, b):
    return sorted(a) == sorted(b) and all(a[k].tobytes() == b[k].tobytes() for k in a)


def main():
    tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    rows = base_rows()
    groups = [0, 1, 2, 3]
    plan_dir = write_step_plan(os.path.join(tmp, "plan"), rows, [[0, 1], [2, 3]])
    panel_rows = P.C.make_rows(5, 11)
    panel_dir = write_step_plan(os.path.join(tmp, "panel_a"), panel_rows, None, panel_like=True)
    transfer_rows = P.C.make_rows(4, 12)
    transfer_dir = write_step_plan(os.path.join(tmp, "transfer"), transfer_rows, None, panel_like=True)
    panels = {"a": panel_dir}

    # (1) the plan loader
    plan = load(plan_dir)
    assert plan.updates == [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]] and plan.rt_groups == [0, 2] and plan.first == 1
    assert plan.info["update_loss_tokens"] == [sum(int(rows[i].loss_mask.sum()) for i in u) for u in plan.updates]
    bad = {"no updates": (None, None, None), "a group in no update": ([[0, 1], [2]], None, None), "a group twice": ([[0, 1], [1, 2, 3]], None, None),
           "another order": ([[1, 0], [2, 3]], None, None), "an empty update": ([[0, 1, 2, 3], []], None, None),
           "round-trip group outside": ([[0, 1], [2, 3]], {"round_trip_group": [0, 9]}, None),
           "round-trip list too short": ([[0, 1], [2, 3]], {"round_trip_group": [0]}, None),
           "another version": ([[0, 1], [2, 3]], {"start_version": 2},
                               lambda z: z.__setitem__("version", np.array([2] * 11 + [1], dtype=np.int32))),
           "first update 0": ([[0, 1], [2, 3]], {"first_update_number": 0}, None)}
    for what, (ups, extra, mut) in bad.items():
        d = os.path.join(tmp, "bad_" + what.replace(" ", "_"))
        write_step_plan(d, rows, ups, extra, mut)
        if ups is None:
            json.dump({"schema": "x"}, open(os.path.join(d, "rows.json"), "w"))
        try:
            load(d)
            raise AssertionError(f"the plan loader accepted: {what}")
        except SS.PlanError:
            pass
    d = write_step_plan(os.path.join(tmp, "ok_version"), rows, [[0, 1], [2, 3]], {"start_version": 2, "round_trip_group": [1, 3]},
                        lambda z: z.__setitem__("version", np.full(12, 2, dtype=np.int32)))
    assert load(d).start_version == 2 and load(d).rt_groups == [1, 3]
    print("(1) plan loader: PASS")

    # (2) a complete session == ppo_batch, bitwise
    ref = P.FakeTrainer(cfg_for(tmp), 2)
    ref_rows = load(plan_dir).rows
    theta0 = [m.detach().clone() for m in ref.opt.masters]
    res = TS.ppo_batch(ref, ref_rows, 0, MI.LR, 2, StepConfig(max_policy_lag=None))
    assert TS.split_minibatches(ref_rows, 2) == plan.updates
    rc, run, tr, s, pt = session(tmp, "full", plan_dir, ["GO", "GO"], panels=panels, transfer=transfer_dir)
    assert rc == 0 and pt.reader_rc == 0, (rc, pt.reader_rc, s.sess["exit"])
    assert all(torch.equal(a, b) for a, b in zip(tr.opt.masters, ref.opt.masters)), "masters differ from ppo_batch"
    assert same_state(opt_state(tr), opt_state(ref)), "optimizer state differs from ppo_batch"
    assert os.path.exists(os.path.join(run, "TRAINER_DONE.json")) and not os.path.exists(os.path.join(run, "STOPPED.json"))
    nm = MI.names(len(tr.shapes), KEYS)
    for n in (1, 2):
        ud = os.path.join(run, "updates", str(n))
        st = json.load(open(os.path.join(ud, "step.json")))
        o, _ = res[n - 1]
        assert st["N"] == o.metrics["total_tokens"] and abs(st["policy_loss"] - o.metrics["policy_loss"]) == 0.0
        assert st["clip_share"] == o.metrics["ppo_clip_ratio"] and st["ratio_max"] == o.metrics["ppo_ratio_max"]
        assert st["grad_norm"] == res[n - 1][1] and st["grad_norm"] > 0.0
        if n == 1:
            assert st["ratio_max"] == 1.0 and st["clip_share"] == 0.0 and st["A_delta_max_abs"] == 0.0
            assert all(a["log_ratio_abs_max"] == 0.0 for a in st["attempts"].values())
        else:
            assert any(a["log_ratio_abs_max"] > 0.0 for a in st["attempts"].values()) and st["A_delta_max_abs"] > 0.0
        assert len(st["attempts"]) == 6 and all(set(a) >= {"advantage", "clip_share", "log_ratio_abs_mean", "sketch"} for a in st["attempts"].values())
        ad = json.load(open(os.path.join(ud, "adapters.json")))
        assert set(ad) == {"v", "R", "R2"} and len({ad[k]["sha256"] for k in ad}) == 3
        v = torch.load(os.path.join(run, "work", "adapters", f"u{n}_v", "state.pt"), weights_only=True)
        ck = torch.load(os.path.join(ud, "ckpt", "masters.pt"), weights_only=True)
        assert all(torch.equal(ck[k], x) for k, x in zip(nm, v))
        for name in ("R", "R2"):
            R = torch.load(os.path.join(run, "work", "adapters", f"u{n}_{name}", "state.pt"), weights_only=True)
            for t0, x, r, k in zip(theta0, v, R, nm):
                du, dr = float((x.double() - t0.double()).norm()), float((r.double() - t0.double()).norm())
                assert (du == 0.0 and torch.equal(r, t0)) or abs(dr - du) <= 1e-5 * du + 1e-12, (n, name, k, du, dr)
        rd = json.load(open(os.path.join(ud, "READS_DONE")))
        want = {f"updates/{n}/reads_dmg/a/{x}.{e}" for x in ("v", "R", "R2", "base2") for e in ("npz", "json")}
        want |= {"base/reads_dmg/a/base.npz", "base/reads_dmg/a/base.json", f"updates/{n}/adapters.json", f"updates/{n}/roundtrip/result.json"}
        want |= {f"updates/{n}/roundtrip/{side}_{x}.npz" for side in ("trainer", "engine") for x in ("v", "R", "R2")}
        want |= {f"updates/{n}/roundtrip/engine_base2.npz"}
        assert set(rd["files"]) == want, sorted(set(rd["files"]) ^ want)
        assert all(MI.sha256(os.path.join(run, k)) == h for k, h in rd["files"].items())
        assert rd["adapters"] == {k: ad[k]["sha256"] for k in ad}
        rr = json.load(open(os.path.join(ud, "roundtrip", "result.json")))
        assert rr["result"] == "equal" and rr["checks"]["v_vs_base_positions_differing"] > 0
        assert all(c["differ"] == 0 and c["loss_tokens"] > 0 for k, c in rr["checks"].items() if isinstance(c, dict))
        b0, b2 = np.load(os.path.join(run, "base", "reads_dmg", "a", "base.npz")), np.load(os.path.join(ud, "reads_dmg", "a", "base2.npz"))
        vz = np.load(os.path.join(ud, "reads_dmg", "a", "v.npz"))
        assert all(b0[k].tobytes() == b2[k].tobytes() for k in b0.files) and any(b0[k].tobytes() != vz[k].tobytes() for k in b0.files if k != "ids_crc32")
        js = json.load(open(os.path.join(ud, "reads_dmg", "a", "v.json")))
        assert set(js) == {"adapter_sha256", "head", "engine_sha", "source", "session_id", "panel_rows_sha256"} and js["adapter_sha256"] == ad["v"]["sha256"]
    ident = json.load(open(os.path.join(run, "base", "transfer", "identity.json")))
    assert ident["base_read_vs_recorded"]["differ"] == 0 and ident["base_read_vs_recorded"]["loss_tokens"] > 0
    rd2 = RD.Reader(FakeEngines(), ["e0", "e1", "e2"], run, plan_dir, {}, transfer_dir, engine_sha="fake")
    assert rd2.transfer_reads(2) == 0
    assert SS.phase_finish(SimpleNamespace(plan=plan_dir, run_dir=run, transfer=transfer_dir)) == 0 and os.path.exists(os.path.join(run, "STEP_DONE"))
    assert SS.MI.sha256 and json.load(open(os.path.join(run, "STEP_DONE")))["updates"] == [1, 2]
    print("(2) complete session == ppo_batch bitwise; controls, reads, READS_DONE, STEP_DONE: PASS")

    # (3A) resume inside the step
    ck1, ck2 = os.path.join(run, "updates", "1", "ckpt"), os.path.join(run, "updates", "2", "ckpt")
    rcA, runA, trA, sA, ptA = session(tmp, "resumeA", plan_dir, ["GO", "GO"], panels=panels, resume=ck1, old_lp=os.path.join(run, "old_lp.npz"))
    assert rcA == 0, sA.sess["exit"]
    mA, m2 = json.load(open(os.path.join(runA, "updates", "2", "ckpt", "meta.json"))), json.load(open(os.path.join(ck2, "meta.json")))
    assert mA["masters_sha256"] == m2["masters_sha256"] and mA["sha256"] == m2["sha256"] and mA["adapter_sha256"] == m2["adapter_sha256"]
    assert all(torch.equal(a, b) for a, b in zip(trA.opt.masters, ref.opt.masters)) and same_state(opt_state(trA), opt_state(ref))
    r1 = json.load(open(os.path.join(runA, "updates", "1", "TRAINER_READY.json")))
    assert r1["resumed_state"] and not os.path.isdir(os.path.join(runA, "updates", "1", "ckpt"))
    assert os.path.exists(os.path.join(runA, "updates", "1", "READS_DONE")) and os.path.exists(os.path.join(runA, "updates", "1", "VERDICT.json"))
    assert json.load(open(os.path.join(runA, "updates", "1", "adapters.json")))["v"]["sha256"] == json.load(open(os.path.join(ck1, "meta.json")))["adapter_sha256"]
    # a STOP on the restored state: nothing is applied
    rcS, runS, trS, sS, _ = session(tmp, "resumeA_stop", plan_dir, ["STOP"], panels=panels, resume=ck1, old_lp=os.path.join(run, "old_lp.npz"))
    stp = json.load(open(os.path.join(runS, "STOPPED.json")))
    assert rcS == 20 and stp["at_update"] == 1 and stp["last_adapter_above_line"] == 0 and stp["v_1"] == "not kept"
    assert not os.path.isdir(os.path.join(runS, "updates", "2", "ckpt"))
    # refusals
    other = os.path.join(tmp, "other_old.npz")
    z = dict(np.load(os.path.join(run, "old_lp.npz")))
    z["old_0"] = z["old_0"] + np.float32(1e-3)
    np.savez(other, **z)
    for what, kw in (("another old-lp file", {"resume": ck1, "old_lp": other}), ("no old-lp file", {"resume": ck1}),
                     ("the last update's checkpoint", {"resume": ck2, "old_lp": os.path.join(run, "old_lp.npz")})):
        rcx, _, _, sx, _ = session(tmp, "refuse_" + what.replace(" ", "_").replace("'", ""), plan_dir, [], **kw)
        assert rcx == SS.EXIT_INPUT, (what, rcx, sx.sess["exit"])
    import shutil
    bad_ck = os.path.join(tmp, "bad_ck")
    shutil.copytree(ck1, bad_ck)
    with open(os.path.join(bad_ck, "optimizer.npz"), "r+b") as f:
        f.seek(-1, 2)
        b = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([b[0] ^ 1]))
    rcx, _, _, sx, _ = session(tmp, "refuse_changed_ckpt", plan_dir, [], resume=bad_ck, old_lp=os.path.join(run, "old_lp.npz"))
    assert rcx == SS.EXIT_INPUT, sx.sess["exit"]
    rcx, _, _, sx, _ = session(tmp, "refuse_first3_no_resume", write_step_plan(os.path.join(tmp, "plan_first3"), rows, [[0, 1], [2, 3]],
                                                                           {"first_update_number": 3}), [])
    assert rcx == SS.EXIT_INPUT, sx.sess["exit"]
    print("(3A) resume inside the step == uninterrupted, bitwise; restored state re-read and judged; refusals: PASS")

    # (3B) the next plan from the last checkpoint == one trainer running both plans
    helper = P.FakeTrainer(cfg_for(tmp), 1)
    SS.load_ckpt(helper, ck2, nm)
    rows2 = P.C.make_rows(9, 21)
    for k, r in enumerate(rows2):
        r.trial_idx, r.group_idx = 100 + k, 10 + k // 3
    sc = np.random.default_rng(3).choice([0.0, 1.0], size=len(rows2))
    for r, a_ in zip(rows2, P.KS.grpo_advantages(sc, [r.group_idx for r in rows2])):
        r.advantage = float(a_)
    for r, x in zip(rows2, helper.logprobs(rows2)):           # the next step's rows as sampled under the adapter of update 2
        r.engine_lp = np.concatenate([[0.0], x]).astype(np.float64) * r.loss_mask
    plan2_dir = write_step_plan(os.path.join(tmp, "plan2"), rows2, [[10], [11, 12]], {"first_update_number": 3})
    ref2_rows = load(plan2_dir).rows
    assert TS.split_minibatches(ref2_rows, 2) == load(plan2_dir).updates
    TS.ppo_batch(ref, ref2_rows, 0, MI.LR, 2, StepConfig(max_policy_lag=None))      # `ref` already ran plan 1: the uninterrupted trainer
    rcB, runB, trB, sB, _ = session(tmp, "resumeB", plan2_dir, ["GO", "GO", "GO"], first=2, panels=panels, resume=ck2)
    assert rcB == 0, sB.sess["exit"]
    assert all(torch.equal(a, b) for a, b in zip(trB.opt.masters, ref.opt.masters)) and same_state(opt_state(trB), opt_state(ref))
    assert sorted(os.listdir(os.path.join(runB, "updates"))) == ["2", "3", "4"]
    st3 = json.load(open(os.path.join(runB, "updates", "3", "step.json")))
    assert st3["ratio_max"] == 1.0 and st3["clip_share"] == 0.0 and st3["stale_checked_rows"] == 3
    Rb = torch.load(os.path.join(runB, "work", "adapters", "u4_R", "state.pt"), weights_only=True)
    for t0, x, r in zip(theta0, trB.opt.masters, Rb):         # the control follows the update accumulated since the RUN's start
        du, dr = float((x.double() - t0.double()).norm()), float((r.double() - t0.double()).norm())
        assert abs(dr - du) <= 1e-5 * du + 1e-12
    rcx, _, _, sx, _ = session(tmp, "refuse_plan2_from_ck1", plan2_dir, [], resume=ck1)
    assert rcx == SS.EXIT_INPUT, sx.sess["exit"]
    print("(3B) next plan from the checkpoint == one trainer running both, bitwise: PASS")

    # (4) verdicts
    for oc, label, code in (("STOP", "not kept", 20), ("INCONCLUSIVE", "not kept", 20), ("NONE", "not judged", 20), ("BADHASH", "not judged", 20)):
        rc4, run4, tr4, s4, _ = session(tmp, "v1_" + oc, plan_dir, [oc, "GO"], panels=panels, verdict_timeout_s=1.0 if oc == "NONE" else 30.0)
        stp = json.load(open(os.path.join(run4, "STOPPED.json")))
        assert rc4 == code and stp["at_update"] == 1 and stp["v_1"] == label and stp["last_adapter_above_line"] == 0 and stp["resume_from"] is None, stp
        assert not os.path.isdir(os.path.join(run4, "updates", "2", "ckpt")) and not os.path.exists(os.path.join(run4, "TRAINER_DONE.json"))
        assert os.path.isdir(os.path.join(run4, "updates", "1", "ckpt")) and os.path.exists(os.path.join(run4, "updates", "1", "READS_DONE"))
        m1 = json.load(open(os.path.join(run4, "updates", "1", "ckpt", "meta.json")))
        assert m1["masters_sha256"] == json.load(open(os.path.join(ck1, "meta.json")))["masters_sha256"]
    rc4, run4, tr4, s4, _ = session(tmp, "v2_STOP", plan_dir, ["GO", "STOP"], panels=panels)
    stp = json.load(open(os.path.join(run4, "STOPPED.json")))
    assert rc4 == 20 and stp["at_update"] == 2 and stp["v_2"] == "not kept" and stp["last_adapter_above_line"] == 1 and stp["resume_from"] == "updates/1/ckpt"
    rc4, run4, tr4, s4, _ = session(tmp, "v2_NONE", plan_dir, ["GO", "NONE"], panels=panels, verdict_timeout_s=1.0)
    stp = json.load(open(os.path.join(run4, "STOPPED.json")))
    assert rc4 == 20 and stp["at_update"] == 2 and stp["v_2"] == "not judged" and "READS_DONE" in stp["reason"]
    # check_verdict on the complete run's files
    sid = "cpu-test-full"
    assert SS.check_verdict(run, 1, sid) == ("GO", None)
    assert SS.check_verdict(run, 1, "another")[0] is None and SS.check_verdict(run, 3, sid)[0] is None
    vp = os.path.join(run, "updates", "1", "VERDICT.json")
    good = json.load(open(vp))
    for what, mut in (("another update", lambda v: v.__setitem__("update", 2)), ("another session", lambda v: v.__setitem__("session_id", "x")),
                      ("an outcome not in the list", lambda v: v.__setitem__("outcome", "PASS")),
                      ("a file not covered", lambda v: v["judged"].pop("updates/1/reads_dmg/a/R2.npz")),
                      ("base not covered", lambda v: v["judged"].pop("base/reads_dmg/a/base.npz"))):
        v = copy.deepcopy(good)
        mut(v)
        json.dump(v, open(vp, "w"))
        assert SS.check_verdict(run, 1, sid)[0] is None, what
    json.dump(good, open(vp, "w"))
    assert SS.check_verdict(run, 1, sid) == ("GO", None)
    fp = os.path.join(run, "updates", "1", "reads_dmg", "a", "v.json")
    keep = open(fp).read()
    open(fp, "w").write(keep + " ")
    assert SS.check_verdict(run, 1, sid)[0] is None, "a file changed after READS_DONE"
    open(fp, "w").write(keep)
    print("(4) verdicts: STOP, INCONCLUSIVE, timeout, wrong hash, and check_verdict's refusals: PASS")

    # (5) round trip and a dead read side
    for what, eng, code in (("leak", FakeEngines(leak=True), 21), ("one value in R", FakeEngines(flip="R"), 21), ("one value in v", FakeEngines(flip="v"), 21)):
        rc5, run5, tr5, s5, pt5 = session(tmp, "rt_" + what.replace(" ", "_"), plan_dir, ["GO", "GO"], engines=eng, panels=panels)
        rr = json.load(open(os.path.join(run5, "updates", "1", "roundtrip", "result.json")))
        assert rc5 == code and pt5.reader_rc == 21 and rr["result"] == "differ", (what, rc5, pt5.reader_rc)
        assert not os.path.exists(os.path.join(run5, "updates", "1", "READS_DONE")) and not os.path.isdir(os.path.join(run5, "updates", "2", "ckpt"))
        assert json.load(open(os.path.join(run5, "STOPPED.json")))["v_1"] == "not judged"
    rc5, run5, tr5, s5, pt5 = session(tmp, "reader_dies", plan_dir, ["GO", "GO"], engines=FakeEngines(die_at=8), panels=panels)
    assert rc5 == SS.EXIT_READS and pt5.reader_rc == RD.EXIT_READS and not os.path.isdir(os.path.join(run5, "updates", "2", "ckpt")), (rc5, pt5.reader_rc)
    print("(5) round trip: a leaked adapter state and a single changed value give 'differ' and exit 21; a dead read side exit 6: PASS")

    # (6) a STOP while the next gradient runs interrupts it
    class SlowW(P.FakeW):
        def process_row(self, i, row, job):
            if job.get("minibatch_index", 0) >= 1:
                time.sleep(1.0)
            return super().process_row(i, row, job)

    class SlowTrainer(P.FakeTrainer):
        def __init__(self, cfg, n):
            super().__init__(cfg, n)
            self.workers = [SlowW(cfg, f, self.layout) for f in self.flats]
            self.be = self.workers[0].be
            self.handles = [TS.InlineHandle(w) for w in self.workers]
            self.work = self.be.parameters()
            self.opt = P.MasterAdamW(self.work, lr=MI.LR, weight_decay=0.0, max_grad_norm=1.0)
            self._sync()

    t0 = time.time()
    rc6, run6, tr6, s6, _ = session(tmp, "early", plan_dir, ["STOP"], tr=SlowTrainer(cfg_for(tmp), 1), panels=panels)
    el = time.time() - t0
    whats = [p["what"] for p in s6.sess["phases"]]
    assert rc6 == 20 and any("gradient interrupted" in w for w in whats) and s6.aborted, (rc6, whats)
    assert el < 5.0, f"the 6 s gradient was not interrupted ({el:.1f} s)"
    assert json.load(open(os.path.join(run6, "STOPPED.json")))["v_1"] == "not kept" and not os.path.isdir(os.path.join(run6, "updates", "2", "ckpt"))
    rc6, run6, tr6, s6, _ = session(tmp, "no_early", plan_dir, ["STOP"], tr=SlowTrainer(cfg_for(tmp), 2), panels=panels, early_stop=False)
    assert rc6 == 20 and not any("gradient interrupted" in p["what"] for p in s6.sess["phases"]) and os.path.exists(os.path.join(run6, "updates", "2", "step.json"))
    assert not os.path.isdir(os.path.join(run6, "updates", "2", "ckpt"))
    print(f"(6) a STOP during the next gradient interrupts it ({el:.1f} s instead of 6): PASS")

    # (7) rows whose seeds are all zero
    zrows = load(plan_dir).rows
    for r in zrows:
        if r.group_idx == 1:
            r.advantage = 0.0
    outs = {}
    for skip in (True, False):
        TS.SKIP_ZERO_SEED_PARTS = skip
        sk = os.path.join(tmp, f"sk_{skip}")
        trz = P.FakeTrainer(cfg_for(tmp, sketch_expert_stride=1, sketch_dir=sk), 2)
        rz = copy.deepcopy(zrows)
        for r, x in zip(rz, trz.logprobs(rz)):
            r.notes["old_lp"] = x
        outs[skip] = trz.compute(rz, 0, StepConfig(max_policy_lag=None), seeds="ppo", minibatch_index=0)
    TS.SKIP_ZERO_SEED_PARTS = True
    a_, b_ = outs[True], outs[False]
    assert all(x.numpy().tobytes() == y.numpy().tobytes() for x, y in zip(a_.grads, b_.grads)), "the skip changed the gradient"
    assert a_.metrics["zero_seed_rows"] == 3 and b_.metrics["zero_seed_rows"] == 0
    assert a_.metrics["total_tokens"] == b_.metrics["total_tokens"] and a_.metrics["replay_checked_rows"] == len(zrows) == a_.metrics["stale_checked_rows"]
    assert sum(1 for r in a_.rows if r["passes"] == 0) == 3 and all(r["passes"] == 1 for r in b_.rows)
    recs = [torch.load(f, weights_only=False) for f in a_.metrics["sketch_files"]]
    assert sorted(d["trial"] for d in recs) == [3, 4, 5] and all(d["rows_skipped_zero_factor"] == 1 and d["rows"] == 0 for d in recs)
    z0 = base_rows()
    for r in z0:
        if r.group_idx == 0:
            r.advantage = 0.0
    zplan = write_step_plan(os.path.join(tmp, "plan_zero_update"), z0, [[0], [1, 2, 3]])
    rc7, run7, _, s7, _ = session(tmp, "zero_update", zplan, ["GO", "GO"], panels=panels)
    assert rc7 == SS.EXIT_GATE and "exactly zero" in s7.sess["exit"]["reason"], s7.sess["exit"]
    print("(7) zero-seed rows: gradient bitwise with and without their backward; counted; in the sketch records; an update that moves nothing is a gate: PASS")
    print("T8 STEP SESSION CPU TEST: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
