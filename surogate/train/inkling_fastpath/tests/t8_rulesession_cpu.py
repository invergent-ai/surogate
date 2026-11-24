"""CPU checks of the rule test (tests/t8_rulesession.py, tests/t8_rulesession_reads.py) on the fake model. Pass line:
(1) the state list parser and the round-trip row choice (two longest + four shortest of a group);
(2) build, from a real first update's checkpoint and gradient file of a step session on the fake: rules.json's checks pass (P and E at the
    optimizer step's overall norm, T at its per-tensor norms, A<s> = theta0 + s x step, A at size 1 = the checkpoint's masters and adapter
    file); a control has its rule state's per-tensor norms and another direction; first-order numbers scale with the size;
(3) the session (trainer side + read side on fake engines): every state is read on exactly the sets its entry names, every round trip
    (first and last read) is "equal", the base re-read at the end equals the first read, RULES_DONE lists every file with its hash, an
    "own" read names its plan rows;
(4) a state served with the previous state's values (a leak) stops the read side with exit 21 and no READS_DONE for that state;
    rules.json built from another gradient is refused."""
import json
import os
import sys
import tempfile
import threading

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from surogate.inkling_rl import adapter as AD  # noqa: E402
import t8_micro as MI  # noqa: E402
import t8_ppo_cpu as P  # noqa: E402
import t8_rules as RU  # noqa: E402
import t8_rulesession as RS  # noqa: E402
import t8_rulesession_reads as RR  # noqa: E402
import t8_stepsession as SS  # noqa: E402
import t8_stepsession_cpu as C  # noqa: E402

STATES = "A8=a+own,P8=a+own,T8=a,E8=a,RA8=a+own,R2A8=a,RP8=a,A4=a,P12=a"


def fake_import(adapter_dir):
    """The exported file's values in the fake trainer's parameter order (the kit's reader of its own adapter form)."""
    layers, _ = AD.import_adapter(adapter_dir, physical_layers=3)
    return [torch.from_numpy(np.ascontiguousarray(d[k], dtype=np.float32)) for d in layers for k in C.KEYS]


def run_session(tmp, name, plan_dir, build_dir, ck, grad, panel_dir, engines):
    run = os.path.join(tmp, name)
    os.makedirs(run)
    open(os.path.join(run, "session_id.txt"), "w").write("rules-cpu-" + name)
    tab = SS.read_json(os.path.join(build_dir, "rules.json"))
    SS.write_json(os.path.join(run, "rules.json"), tab)
    reader = RR.StateReader(engines, ["e0", "e1"], run, plan_dir, {"a": panel_dir}, [0, 1], 0, ["a", "own"], engine_sha="fake", poll_s=0.02)
    out = {}

    def read():
        try:
            out["rc"] = reader.serve_states([r["name"] for r in tab["states"]])
        except Exception as e:
            out["rc"], out["err"] = RR.RD.EXIT_READS, repr(e)
        open(os.path.join(run, "reader.exit"), "w").write(str(out["rc"]))

    th = threading.Thread(target=read, daemon=True)
    th.start()
    tr = P.FakeTrainer(C.cfg_for(tmp), 1)
    m_before = [m.detach().clone() for m in tr.opt.masters]
    try:
        trc = RS.run_states(tr, C.load(plan_dir), tab, run, os.path.join(build_dir, "adapters"), 0, "rules-cpu-" + name, fake_import,
                            log=lambda *_: None)
        assert all(torch.equal(a, b) for a, b in zip(tr.opt.masters, m_before)), "the trainer's masters were not put back"
    except SS.SessionStop as e:
        trc = e.code
    SS.write_json(os.path.join(run, "trainer_end.json"), {"exit": trc})
    th.join(120)
    return trc, out, run, tab


def main():
    tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    # (1)
    st = RS.parse_states("A8=p1+p2+own,R2A8=p1,RP8=p1,E12=p1")
    assert [(s["name"], s["kind"], s["rule"], s["size"], s["draw"]) for s in st] == [("A8", "rule", "A", 8.0, None), ("R2A8", "control", "A", 8.0, "R2"),
                                                                                     ("RP8", "control", "P", 8.0, "R"), ("E12", "rule", "E", 12.0, None)]
    assert st[0]["sets"] == ["p1", "p2", "own"]
    for bad in ("X8=a", "A8", "A8=a,A8=b", "RR8=a", ""):
        try:
            RS.parse_states(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    T = {n: 100 + 7 * n for n in range(10)}
    assert RR.pick_round_trip_rows(T) == [0, 1, 2, 3, 4, 5, 8, 9] and RR.pick_round_trip_rows({3: 5, 4: 9}) == [3, 4]
    assert RR.pick_round_trip_rows({n: 50 - n for n in range(12)}) == [0, 1, 6, 7, 8, 9, 10, 11]
    print("(1) state list and round-trip row choice: PASS")

    # a first update on the fake, through the step session itself
    rows = C.base_rows()
    plan_dir = C.write_step_plan(os.path.join(tmp, "plan"), rows, [[0, 1], [2, 3]])
    panel_dir = C.write_step_plan(os.path.join(tmp, "panel_a"), P.C.make_rows(5, 11), None, panel_like=True)
    rc, run1, tr1, s1, _ = C.session(tmp, "update1", plan_dir, ["GO"], panels={"a": panel_dir}, last_update=1)
    assert rc == 0, s1.sess["exit"]
    ck, grad = os.path.join(run1, "updates", "1", "ckpt"), os.path.join(run1, "updates", "1", "grad.pt")
    nm = MI.names(len(tr1.shapes), C.KEYS)
    exp = lambda st_, dst, tag: C.fake_export(3)(st_, dst, tag)[0]  # noqa: E731

    # (2)
    bd = os.path.join(tmp, "build")
    tab = RS.build(ck, grad, RS.parse_states(STATES), bd, exp, C.BLOCKS, log=lambda *_: None, plan_dir=plan_dir, rt_group=0,
                   import_fn=fake_import, keep_masters=True)
    assert set(tab["checks"]["adapter_file_is_bf16_of_masters"]) == set(by_ := [r["name"] for r in tab["states"]]) and all(tab["checks"]["adapter_file_is_bf16_of_masters"].values())
    for r in tab["states"]:                                   # the kept masters are the state: their hash is rules.json's
        kept = torch.load(os.path.join(bd, r["masters_file"]), weights_only=True)
        assert SS.state_sha([kept[k] for k in nm], nm) == r["masters_sha256"] and MI.sha256(os.path.join(bd, r["masters_file"])) == r["masters_file_sha256"]
    # a file that is not its state's (another state's adapter under its name) fails the build's own check
    swap = lambda d: fake_import(os.path.join(os.path.dirname(d), "A8")) if os.path.basename(d) == "P8" else fake_import(d)  # noqa: E731
    tab_bad = RS.build(ck, grad, RS.parse_states("A8=a,P8=a"), os.path.join(tmp, "build_swapped"), exp, C.BLOCKS, log=lambda *_: None, import_fn=swap)
    assert tab_bad["checks"]["adapter_file_is_bf16_of_masters"] == {"A8": True, "P8": False} and not tab_bad["checks_pass"]
    assert not RS.build(ck, grad, RS.parse_states("A8=a"), os.path.join(tmp, "build_noimport"), exp, C.BLOCKS, log=lambda *_: None)["checks_pass"]
    assert tab["round_trip_rows"] == [0, 1, 2] and tab["round_trip_group"] == 0
    by = {r["name"]: r for r in tab["states"]}
    assert tab["checks_pass"] and tab["checks"]["A_at_size_1"]["adapter_equal_checkpoint"] and tab["checks"]["A_at_size_1"]["masters_equal_checkpoint"], tab["checks"]
    assert set(tab["checks"]["overall_norm_vs_adam_step"]) == {"A8", "P8", "E8", "A4", "P12"} and set(tab["checks"]["per_tensor_norm_vs_adam_step"]) == {"A8", "T8", "A4"}
    na = tab["adam_step_l2"]
    assert all(abs(by[k]["l2"] - s * na) <= 1e-6 * s * na for k, s in (("A8", 8), ("P8", 8), ("E8", 8), ("A4", 4), ("P12", 12))) and abs(by["A8"]["cos_with_adam_step"] - 1) <= 1e-9
    assert 0 < by["P8"]["cos_with_adam_step"] < 1 and abs(by["P12"]["first_order"]["at_size"] - 12 * by["P12"]["first_order"]["size_1"]) <= 1e-9 * abs(by["P12"]["first_order"]["at_size"])
    assert abs(by["P8"]["first_order"]["size_1"] - by["P12"]["first_order"]["size_1"]) <= 1e-9 * abs(by["P8"]["first_order"]["size_1"]) and by["P8"]["first_order"]["at_size"] > 0
    assert set(tab["checks"]["control_per_tensor_norm_vs_its_state"]) == {"RA8", "R2A8", "RP8"} and all(tab["checks"]["control_zero_where_its_state_is_zero"].values())
    for ctl, of in (("RA8", "A8"), ("R2A8", "A8"), ("RP8", "P8")):
        assert by[ctl]["kind"] == "control" and by[ctl]["of"] == of and abs(by[ctl]["cos_with_adam_step"]) < 0.5
        for k in nm:                                          # the rule state's per-tensor norms (to fp32 rounding), zero where it is zero
            a_, b_ = by[of]["per_tensor_l2"][k], by[ctl]["per_tensor_l2"][k]
            assert (a_ == 0.0 and b_ == 0.0) or abs(a_ - b_) <= 1e-3 * a_, (ctl, k, a_, b_)
    assert by["RA8"]["masters_sha256"] != by["R2A8"]["masters_sha256"] and len({r["adapter_sha256"] for r in tab["states"]}) == len(tab["states"])
    assert all(os.path.exists(os.path.join(bd, "adapters", r["name"], "adapter_model.safetensors")) for r in tab["states"])
    print("(2) build: rules.json's checks, controls with their rule state's per-tensor norms, first-order numbers: PASS")

    # (3)
    trc, out, run, tab = run_session(tmp, "rules", plan_dir, bd, ck, grad, panel_dir, C.FakeEngines())
    assert trc == 0 and out.get("rc") == 0, (trc, out)
    done = json.load(open(os.path.join(run, "RULES_DONE")))
    assert done["states"] == [r["name"] for r in tab["states"]] and all(MI.sha256(os.path.join(run, k)) == h for k, h in done["files"].items())
    plan = C.load(plan_dir)
    own_rows = [i for i, r in enumerate(plan.rows) if r.group_idx in (0, 1)]
    for r in tab["states"]:
        sd = os.path.join(run, "states", r["name"])
        assert sorted(os.listdir(os.path.join(sd, "reads"))) == sorted(r["sets"]), (r["name"], os.listdir(os.path.join(sd, "reads")))
        rr = json.load(open(os.path.join(sd, "roundtrip", "result.json")))
        assert rr["result"] == "equal" and all(c["differ"] == 0 and c["loss_tokens"] > 0 for c in rr["checks"].values()) and rr["state_vs_base_positions_differing"] > 0
        rdone = json.load(open(os.path.join(sd, "READS_DONE")))
        want = {f"states/{r['name']}/reads/{s}/read.{e}" for s in r["sets"] for e in ("npz", "json")}
        want |= {f"states/{r['name']}/roundtrip/{f}" for f in ("trainer.npz", "engine.npz", "engine_last.npz", "result.json")} | {f"states/{r['name']}/adapter.json"}
        assert set(rdone["files"]) == want and rdone["adapter_sha256"] == r["adapter_sha256"], sorted(set(rdone["files"]) ^ want)
        js = json.load(open(os.path.join(sd, "reads", "a", "read.json")))
        assert js["adapter_sha256"] == r["adapter_sha256"] and js["session_id"] == "rules-cpu-rules"
        if "own" in r["sets"]:
            jo = json.load(open(os.path.join(sd, "reads", "own", "read.json")))
            zo = np.load(os.path.join(sd, "reads", "own", "read.npz"))
            assert jo["rows"] == own_rows and sorted(int(k[3:]) for k in zo.files if k.startswith("lp_")) == own_rows
    for s in ("a", "own"):
        a_, b_ = np.load(os.path.join(run, "base", s, "base.npz")), np.load(os.path.join(run, "base", s, "base2.npz"))
        assert sorted(a_.files) == sorted(b_.files) and all(a_[k].tobytes() == b_[k].tobytes() for k in a_.files)
    rj = json.load(open(os.path.join(run, "base", "reread.json")))
    idj = json.load(open(os.path.join(run, "base", "roundtrip", "identity.json")))
    assert rj["result"] == "equal" and idj["result"] == "equal" and idj["differ_from_recorded"] == 0 and idj["round_trip_rows"] == tab["round_trip_rows"]
    assert all(abs(by[c_]["first_order"]["size_1"]) < abs(by["A8"]["first_order"]["size_1"]) for c_ in ("RA8", "R2A8", "RP8"))
    assert {"rules.json", "base/reread.json", "base/a/base.npz", "base/own/base2.npz", "states/A8/READS_DONE"} <= set(done["files"])
    a8 = np.load(os.path.join(run, "states", "A8", "reads", "a", "read.npz"))
    p8 = np.load(os.path.join(run, "states", "P8", "reads", "a", "read.npz"))
    assert any(a8[k].tobytes() != p8[k].tobytes() for k in a8.files if k.startswith("lp_")), "two rules read the same"
    print("(3) the session: every state on its sets, round trips equal (first and last), base re-read equal, RULES_DONE with hashes: PASS")

    # (4)
    trc, out, run4, _ = run_session(tmp, "rules_leak", plan_dir, bd, ck, grad, panel_dir, C.FakeEngines(leak=True))
    assert out.get("rc") == RR.RD.EXIT_ROUNDTRIP and trc in (0, SS.EXIT_ROUNDTRIP), (trc, out)
    assert os.path.exists(os.path.join(run4, "states", "A8", "READS_DONE")) and not os.path.exists(os.path.join(run4, "states", "P8", "READS_DONE"))
    assert json.load(open(os.path.join(run4, "states", "P8", "roundtrip", "result.json")))["result"] == "differ" and not os.path.exists(os.path.join(run4, "RULES_DONE"))
    ck_meta, g_json = json.load(open(os.path.join(ck, "meta.json"))), json.load(open(os.path.splitext(grad)[0] + ".json"))
    RS.check_build(tab, ck_meta, g_json)
    RS.check_build(tab, None, None)
    for what, bad_tab in (("another gradient", dict(tab, grad_sha256="0" * 64)), ("failed checks", dict(tab, checks_pass=False)),
                          ("another checkpoint", dict(tab, checkpoint=dict(tab["checkpoint"], masters_sha256="0" * 64)))):
        try:
            RS.check_build(bad_tab, ck_meta, g_json)
            raise AssertionError(f"a rules.json with {what} was accepted")
        except SS.PlanError:
            pass
    # the trainer side takes a state from its adapter FILE: the values it runs are the file's (bf16), whatever host built the masters
    a8 = fake_import(os.path.join(bd, "adapters", "A8"))
    fu2 = RU.load_first_update(ck, grad, MI.sha256, SS.state_sha)
    want8 = RU.add_delta(fu2["theta0"], RU.rule_delta("A", fu2["grads"], fu2["coef"], fu2["d_adam"], 8.0))
    assert all(torch.equal(x, torch.from_numpy(AD.round_bf16(w.numpy()).reshape(tuple(w.shape)))) for x, w in zip(a8, want8)), "the file is not bf16(masters)"
    rd_ok = RR.StateReader(C.FakeEngines(), ["e0", "e1"], os.path.join(tmp, "rules"), plan_dir, {"a": panel_dir}, [0, 1], 0, ["a"], engine_sha="fake")
    RR.check_sets(tab, rd_ok)
    for what, kw in (("a set the states do not name", {"sets": {"a": panel_dir, "b": panel_dir}, "own": [0, 1]}), ("no own set", {"sets": {"a": panel_dir}, "own": []})):
        rd_bad = RR.StateReader(C.FakeEngines(), ["e0", "e1"], os.path.join(tmp, "rules"), plan_dir, kw["sets"], kw["own"], 0, ["a"], engine_sha="fake")
        try:
            RR.check_sets(tab, rd_bad)
            raise AssertionError(what)
        except ValueError:
            pass
    try:                                                      # more engines than round-trip rows: refused when the read side is built
        RR.StateReader(C.FakeEngines(), [f"e{i}" for i in range(4)], os.path.join(tmp, "rules"), plan_dir, {"a": panel_dir}, [0, 1], 0, ["a"], engine_sha="fake")
        raise AssertionError("four engines for three round-trip rows were accepted")
    except ValueError:
        pass
    tr = P.FakeTrainer(C.cfg_for(tmp), 1)
    t2 = dict(tab, states=[dict(tab["states"][0], adapter_sha256="0" * 64)])
    os.makedirs(os.path.join(tmp, "rules_badsha"))
    try:
        RS.run_states(tr, plan, t2, os.path.join(tmp, "rules_badsha"), os.path.join(bd, "adapters"), 0, "x", fake_import, log=lambda *_: None)
        raise AssertionError("a state whose adapter file is not rules.json's was handed over")
    except SS.PlanError:
        pass
    assert not os.path.exists(os.path.join(tmp, "rules_badsha", "states", "A8", "TRAINER_READY.json"))
    rd_dl = RR.StateReader(C.FakeEngines(), ["e0", "e1"], os.path.join(tmp, "rules_deadline"), plan_dir, {"a": panel_dir}, [0, 1], 0, ["a"], engine_sha="fake")
    os.makedirs(os.path.join(tmp, "rules_deadline"))
    open(os.path.join(tmp, "rules_deadline", "session_id.txt"), "w").write("x")
    assert rd_dl.serve_states(["A8", "P8"], deadline_s=1e-9) == RR.RD.EXIT_READS, "the reads' time limit did not stop the read side"
    assert json.load(open(os.path.join(tmp, "rules_deadline", "READS_STOPPED.json")))["not_read"] == ["A8", "P8"]
    print("(4) a leaked state stops the read side (exit 21, nothing listed for it); a state that does not match rules.json is not handed over; "
          "rules.json from another gradient or checkpoint, set names that do not match, too few round-trip rows are refused: PASS")
    print("T8 RULE SESSION CPU TEST: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
