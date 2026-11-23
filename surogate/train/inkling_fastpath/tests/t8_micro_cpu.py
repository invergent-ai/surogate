"""CPU checks of the micro-step driver (tests/t8_micro.py) on the fake model and on synthetic files. Pass line:
(1) load_plan accepts the agreed form and refuses each broken variant (crc32, mask[0], no loss token, cap, split group, attempt in two groups);
(2) G = -N x the first PPO mini-batch's gradient == the autograd gradient of sum_rows A sum_t log pi at the start weights, rel <= 1e-5;
(3) the first-order prediction <G, k d> matches the fake's actual change of sum A sum_t log pi along k d within 1 % at k = 0.3 and 1 (larger k
    flip the fake's top-k routing, a jump the first order does not hold);
(4) the adapter states: U_k = theta0 + k d exactly; R12 per tensor of norm ||12 d||; a block state differs from theta0 only in its block and equals
    the whole state there; R12_<block> uses R12's draw;
(5) the compare phase on synthetic reads passes on matching files and fails on a one-ulp change in Z (M1), U12 (M5) or base (M2), and on a missing
    round-2 read (M6); parts merge, and a row read twice is refused."""
import copy
import json
import os
import sys
import tempfile
import zlib

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from surogate.inkling_rl.backend_fake import FakeInkling  # noqa: E402
from surogate.inkling_rl.step import StepConfig  # noqa: E402
import t8_micro as MI  # noqa: E402
import t8_ppo_cpu as P  # noqa: E402
import t8_step as TS  # noqa: E402

KEYS = TS.LORA_KEYS


def write_plan(d, rows, mutate=None):
    os.makedirs(d, exist_ok=True)
    arr = {}
    for n, r in enumerate(rows):
        arr[f"ids_{n}"] = r.ids.astype(np.int32)
        arr[f"mask_{n}"] = r.loss_mask.copy()
        arr[f"lp_{n}"] = (r.engine_lp * r.loss_mask).astype(np.float32)
    arr["advantage"] = np.array([r.advantage for r in rows], dtype=np.float64)
    arr["trial_idx"] = np.array([r.trial_idx for r in rows], dtype=np.int32)
    arr["group_idx"] = np.array([r.group_idx for r in rows], dtype=np.int32)
    arr["T"] = np.array([r.T for r in rows], dtype=np.int32)
    arr["ids_crc32"] = np.array([zlib.crc32(r.ids.astype("<i4").tobytes()) & 0xFFFFFFFF for r in rows], dtype=np.uint32)
    if mutate:
        mutate(arr)
    np.savez(os.path.join(d, "rows.npz"), **arr)
    json.dump({"files": {"rows.npz": MI.sha256(os.path.join(d, "rows.npz"))}}, open(os.path.join(d, "manifest.json"), "w"))


def plan_rows():
    rows = P.make(3)
    for k, r in enumerate(rows):
        r.trial_idx = k
    return rows


def F_of(be, rows):
    tot = 0.0
    for r in rows:
        lp, _ = be.forward(torch.as_tensor(r.ids), lora=be.lora)
        tot = tot + float(r.advantage) * (lp * torch.as_tensor(r.loss_mask.astype(np.float32))).sum()
    return tot


def main():
    tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    # (1) the plan file
    rows = plan_rows()
    good = os.path.join(tmp, "plan")
    write_plan(good, rows)
    got, info = MI.load_plan(good, cap=10 ** 6)
    assert len(got) == len(rows) and info["loss_tokens"] == sum(int(r.loss_mask.sum()) for r in rows)
    assert all(np.array_equal(a.ids, b.ids) and np.array_equal(a.loss_mask, b.loss_mask) for a, b in zip(got, rows))
    bads = {
        "crc32": lambda z: z.__setitem__("ids_crc32", z["ids_crc32"] ^ np.uint32(1)),
        "mask0": lambda z: z["mask_0"].__setitem__(0, True),
        "no loss": lambda z: z["mask_1"].__setitem__(slice(None), False),
        "split group": lambda z: z["group_idx"].__setitem__(0, 3),
        "attempt in two groups": lambda z: z["trial_idx"].__setitem__(3, 0),   # row 3 is in group 1, attempt 0 in group 0
        "positive lp": lambda z: z["lp_2"].__setitem__(np.flatnonzero(z["mask_2"])[0], 0.5),
    }
    for what, mut in bads.items():
        d = os.path.join(tmp, "bad_" + what.replace(" ", "_"))
        write_plan(d, rows, mut)
        try:
            MI.load_plan(d, cap=10 ** 6)
            raise AssertionError(f"load_plan accepted: {what}")
        except ValueError:
            pass
    try:
        MI.load_plan(good, cap=max(r.T for r in rows) - 1)
        raise AssertionError("load_plan accepted a row over the cap")
    except ValueError:
        pass
    d = os.path.join(tmp, "bad_sha")
    write_plan(d, rows)
    json.dump({"files": {"rows.npz": "0" * 64}}, open(os.path.join(d, "manifest.json"), "w"))
    try:
        MI.load_plan(d, cap=10 ** 6)
        raise AssertionError("load_plan accepted a sha mismatch")
    except ValueError:
        pass
    print("(1) plan file checks: PASS")

    # (2) G == the advantage-weighted gradient at the start weights; (3) first-order prediction vs the actual change
    cfg = TS.T8Config(head="plain", max_row_tokens=10 ** 6, segment_keys=(), run_dir=tmp, visits=5, top_k=2, K=5)
    tr = P.FakeTrainer(cfg, 2)
    # a non-zero B so that every tensor has a gradient (at B = 0 the A gradients vanish)
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for m in tr.opt.masters:
            m.add_(torch.randn(m.shape, generator=g) * 0.02)
        tr.opt.sync_work()
    tr._sync()
    rr = copy.deepcopy(rows)
    lps = tr.logprobs(rr)
    for r, x in zip(rr, lps):                                 # the rows as sampled at these weights (the replay gate's premise)
        r.engine_lp = np.concatenate([[0.0], x]).astype(np.float64) * r.loss_mask
        r.routed = None                                       # the capture was taken at other weights; the micro-step's rows carry none
    ref = FakeInkling(seed=0, rank=4, alpha=8)
    with torch.no_grad():
        for p, w in zip(ref.parameters(), tr.work):
            p.copy_(w)
    theta0 = [m.detach().clone() for m in tr.opt.masters]
    res = TS.ppo_batch(tr, rr, 3, MI.LR, 1, StepConfig(max_policy_lag=None))
    out, _ = res[0]
    N = float(out.metrics["total_tokens"])
    assert N == sum(int(r.loss_mask.sum()) for r in rr)
    for p in ref.parameters():
        p.grad = None
    F_of(ref, rr).backward()
    G = [-N * x.double() for x in out.grads]
    worst = max(P.rel(a, p.grad) for a, p in zip(G, ref.parameters()))
    print(f"(2) G vs autograd of sum A sum_t log pi: worst rel {worst:.2e}")
    assert worst <= 1e-5
    delta = [m.detach() - t for m, t in zip(tr.opt.masters, theta0)]
    n_layers = len(ref.lora)
    blocks = {"before": range(0, 1), "inside": range(1, max(2, n_layers - 1)), "after": range(max(2, n_layers - 1), n_layers)}
    fo = MI.first_order(out.grads, N, delta, KEYS, blocks)
    assert abs(sum(fo["by_block"][b] for b in blocks) - fo["by_block"]["all"]) <= 1e-12 * max(1.0, abs(fo["by_block"]["all"]))
    assert abs(fo["attention"] + fo["experts"] - fo["by_block"]["all"]) <= 1e-9 * max(1.0, abs(fo["by_block"]["all"]))
    errs = []
    with torch.no_grad():
        base = F_of(ref, rr)
        for k in (0.3, 1.0):
            moved = FakeInkling(seed=0, rank=4, alpha=8)
            for p, t, dd in zip(moved.parameters(), theta0, delta):
                p.copy_(t + k * dd)
            # base at theta0 exactly (the masters, not their bf16 work copies, so that the change is k d)
            at0 = FakeInkling(seed=0, rank=4, alpha=8)
            for p, t in zip(at0.parameters(), theta0):
                p.copy_(t)
            dF = float(F_of(moved, rr)) - float(F_of(at0, rr))
            pred = k * fo["by_block"]["all"]
            errs.append(abs(dF - pred) / abs(pred))
            print(f"(3) k {k:g}: actual {dF:.6e}, predicted {pred:.6e}, rel err {errs[-1]:.2e}")
    assert max(errs) <= 0.01, errs
    del base

    # (4) the adapter states
    nm = MI.names(n_layers, KEYS)
    st = MI.states(theta0, delta, nm, blocks, ("Z", "U12", "U4", "R12", "U12_inside", "R12_inside", "R12_after"))
    assert all(torch.equal(a, b) for a, b in zip(st["Z"], theta0))
    assert all(torch.equal(a, t + 12.0 * d) for a, t, d in zip(st["U12"], theta0, delta))
    assert all(torch.equal(a, t + 4.0 * d) for a, t, d in zip(st["U4"], theta0, delta))
    for a, t, d in zip(st["R12"], theta0, delta):
        r = (a - t).double()
        want = 12.0 * float(d.double().norm())
        assert abs(float(r.norm()) - want) <= 1e-4 * max(want, 1e-30) + 1e-12, (float(r.norm()), want)
    for w, full in (("U12_inside", "U12"), ("R12_inside", "R12"), ("R12_after", "R12")):
        blk = w.split("_")[1]
        for name, a, f, t in zip(nm, st[w], st[full], theta0):
            assert torch.equal(a, f if MI.block_of(name, blocks) == blk else t), (w, name)
    print("(4) adapter states: PASS")

    # (5) the compare phase on synthetic files
    run = os.path.join(tmp, "run")
    rd = os.path.join(run, "reads")
    os.makedirs(rd)
    prow = {r.notes["plan_row"]: r for r in got}
    old = {n: (r.engine_lp.astype(np.float32)) for n, r in prow.items()}
    pos = {f"pos_{n}": np.nonzero(r.loss_mask)[0].astype(np.int32) for n, r in prow.items()}
    olds = {f"old_{n}": old[n][r.loss_mask] for n, r in prow.items()}
    np.savez(os.path.join(run, "rows_used.npz"), **pos, **olds, N=np.array([1.0]))
    for f in ("first_order.json", "adapters.json", "gates.json"):
        json.dump({}, open(os.path.join(run, f), "w"))
    os.makedirs(os.path.join(run, "masters"))
    os.makedirs(os.path.join(run, "roundtrip"))
    for f in ("masters/theta0.pt", "masters/theta1.pt"):
        torch.save({}, os.path.join(run, f))
    rt = [n for n, r in prow.items() if r.group_idx == 0]
    rep0 = {"M2_trainer": True, "M3": True, "M4": True, "M6_trainer": True, "round_trip_rows": rt, "zero_adapter": {"checks": 6, "all_bitwise": True},
            "plan": info, "A_delta_max_abs": 0.0,
            "step": {"N": float(info["loss_tokens"]), "rows_used": len(prow), "rows_refused": 0, "finite": True, "ratio_max": 1.0, "clip_ratio": 0.0,
                     "stale_checked_rows": len(prow), "replay_checked_rows": len(prow), "policy_loss": -0.035, "metrics": {}}}
    open(os.path.join(run, "session_id.txt"), "w").write("micro-test-session\n")
    open(os.path.join(run, "engine_flags.txt"), "w").write("--served-model-name inkling --prefix-cache-host-gb 2\n")
    open(os.path.join(run, "rounds.txt"), "w").write("r1 ran\nr2 ran\nr3 skipped_for_budget\n")
    rng = np.random.default_rng(3)
    moved = {n: (MI.lp_at_mask(old[n] - rng.random(len(old[n])).astype(np.float32) * 1e-2, prow[n].loss_mask)) for n in prow}

    def write(name, val, ns, parts=2):
        ns = sorted(ns)
        for k in range(parts):
            np.savez(os.path.join(rd, f"{name}.part{k}.npz"), **{f"lp_{n}": val[n] for n in ns[k::parts]})

    def scenario(tag, edit=None, drop=None, partial=None):
        for f in os.listdir(rd):
            os.remove(os.path.join(rd, f))
        V = {"base": {n: MI.lp_at_mask(old[n], prow[n].loss_mask) for n in prow}, "U12": moved}
        write("base", V["base"], prow)
        write("Z", V["base"], prow)
        write("U12", moved, prow)
        for w in ("R12", "U4") + MI.ROUND2:
            if w != drop:
                write(w, moved, [n for n in prow if not (w == partial and n == rt[0])])
        for w in MI.RT_ADAPTERS:
            np.savez(os.path.join(run, "roundtrip", f"trainer_{w}.npz"), **{f"lp_{n}": moved[n] for n in rt})
        if edit:
            name, n = edit
            fname = os.path.join(run, "roundtrip", f"{name}.npz") if name.startswith("trainer_") else \
                os.path.join(rd, [f for f in os.listdir(rd) if f.startswith(name + ".part") and f"lp_{n}" in np.load(os.path.join(rd, f)).files][0])
            z = dict(np.load(fname))
            t = int(np.flatnonzero(prow[n].loss_mask)[0])
            z[f"lp_{n}"] = z[f"lp_{n}"].copy()
            z[f"lp_{n}"][t] = np.nextafter(z[f"lp_{n}"][t], np.float32(0))
            np.savez(fname, **z)
        json.dump(rep0, open(os.path.join(run, "micro.json"), "w"))
        a = type("A", (), {"plan": good, "run_dir": run, "out": os.path.join(run, "micro.json"), "engine_sha_file": ""})()
        orig = MI.load_plan
        MI.load_plan = lambda p, cap=MI.CAP: orig(p, cap=10 ** 6)
        try:
            rc = MI.phase_compare(a)
        finally:
            MI.load_plan = orig
        rep = json.load(open(os.path.join(run, "micro.json")))
        return rc, rep

    rc, rep = scenario("match")
    assert rc == 0 and rep["pass"], rep
    assert rep["session_id"] == "micro-test-session" and rep["rounds"] == {"r1": "ran", "r2": "ran", "r3": "skipped_for_budget"}
    assert rep["engine_cmdline"].endswith("--prefix-cache-host-gb 2") and json.load(open(os.path.join(rd, "base.json")))["session_id"] == rep["session_id"]
    print("MICRO_JSON_SAMPLE " + json.dumps(rep, default=float))
    print("GATES_JSON_SAMPLE " + json.dumps(json.load(open(os.path.join(run, "gates.json"))), default=float))
    for w in MI.ROUND1:                                       # the follow read's form: every plan row, 0 off the mask, a JSON with head plain
        z = np.load(os.path.join(rd, w + ".npz"))
        assert sorted(int(k[3:]) for k in z.files) == sorted(prow) and all(not z[f"lp_{n}"][~prow[n].loss_mask].any() for n in prow)
        assert json.load(open(os.path.join(rd, w + ".json")))["head"] == "plain"
    assert not os.path.exists(os.path.join(rd, "trainer_U12.npz"))
    for tag, edit, flag in (("Z", ("Z", rt[0]), "M1"), ("U12", ("U12", rt[-1]), "M5"), ("base", ("base", max(prow)), "M2"),
                            ("trainer", ("trainer_U12", rt[0]), "M5"), ("R12", ("R12", rt[0]), "M5"), ("trainerU4", ("trainer_U4", rt[-1]), "M5")):
        rc, rep = scenario(tag, edit=edit)
        assert rc != 0 and not rep[flag], (tag, rep)
    rc, rep = scenario("drop", drop="R12")
    assert rc != 0 and not rep["M6"], rep
    rc, rep = scenario("drop2", drop="U12_after")               # round 2 is reported, not a line
    assert rc == 0 and rep["pass"] and not rep["reads_complete"]["U12_after"], rep
    rc, rep = scenario("partial", partial="U12_inside")           # a read on some rows only: parts kept, no reads/<X>.npz, recorded
    assert rc == 0 and rep["pass"] and rep["reads_incomplete"].get("U12_inside") == len(prow) - 1, rep
    assert not os.path.exists(os.path.join(rd, "U12_inside.npz")) and not os.path.exists(os.path.join(rd, "U12_inside.json"))
    g_ = json.load(open(os.path.join(run, "gates.json")))
    assert g_["replay"]["mismatch_tokens"] == 0 and g_["replay"]["rows_checked"] == len(prow) and g_["zero_adapter"]["bitwise"] is True
    # a row read twice
    for f in os.listdir(rd):
        os.remove(os.path.join(rd, f))
    write("base", {n: MI.lp_at_mask(old[n], prow[n].loss_mask) for n in prow}, prow)
    for f in [f for f in os.listdir(rd) if not f.endswith(".npz") or ".part" not in f]:
        os.remove(os.path.join(rd, f))
    np.savez(os.path.join(rd, "base.part9.npz"), **{f"lp_{rt[0]}": MI.lp_at_mask(old[rt[0]], prow[rt[0]].loss_mask)})
    a = type("A", (), {"plan": good, "run_dir": run, "out": os.path.join(run, "micro.json"), "engine_sha_file": ""})()
    orig = MI.load_plan
    MI.load_plan = lambda p, cap=MI.CAP: orig(p, cap=10 ** 6)
    try:
        MI.phase_compare(a)
        raise AssertionError("a row read twice passed")
    except ValueError:
        pass
    finally:
        MI.load_plan = orig
    print("(5) compare phase: PASS")
    print("T8 MICRO CPU TEST: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
