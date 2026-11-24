"""Rule test: forward-only reads of adapter states built from ONE update's gradient under several update rules (t8_rules), each state at a
size, with random controls of the same per-tensor norms. Nothing is trained and nothing is kept as a model.

A state's name says what it is:  <rule><size>  (A8 = theta0 + 8 x the optimizer's step; P, T, E as in t8_rules), or a control
R<rule><size> / R2<rule><size> = theta0 + a fixed-seed random direction with that state's per-tensor norms (first / second seed).

build (CPU, no card): from a first update's checkpoint and gradient file, every state's masters -> its adapter directory and rules.json:
  per state {name, kind, rule, size, eps, of, seed, masters_sha256, adapter_sha256, l2, l2_by_block, per_tensor_l2, nonzero_entries,
  cos_with_adam_step, first_order {at_size, size_1} = N x <-grad, d>, sets}; "round_trip_rows" (with a plan given); "checks": P and E have the optimizer step's overall norm
  and T its per-tensor norms (largest relative difference), the score adapters (q, k, r) are untouched in every state, A<s> is
  theta0 + s x (theta1 - theta0), A at size 1 is the checkpoint's masters and exports to the checkpoint's adapter hash.
states (trainer side, one worker): the trainer's start adapter must equal the checkpoint's by value and replay the round-trip rows' records
  bitwise; then per state, in the list's order: the masters rebuilt (their hash must be rules.json's), the shipped adapter's hash checked,
  the trainer's log-probs of the round-trip rows -> states/<name>/roundtrip/trainer.npz, and the hand-over states/<name>/TRAINER_READY.json.
The read side (tests/t8_rulesession_reads.py) reads every state on the row sets its entry names. Exit codes as the step session's
(0; 21 a round trip or a base re-read not bitwise; 4 a trainer gate; 5 inputs; 6 the read side failed).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import t8_micro as MI  # noqa: E402
import t8_rules as RU  # noqa: E402
import t8_stepsession as SS  # noqa: E402
from t8_rulesession_reads import pick_round_trip_rows  # noqa: E402

NAME = re.compile(r"^(R2|R|)([APTE])(\d+(?:\.\d+)?)$")
SEEDS = {"R": MI.R_SEED, "R2": MI.R_SEED_B}


def parse_states(spec: str) -> list:
    """"A8=t0x+census+own+transfer,RA8=t0x+census,..." -> [{"name", "kind", "rule", "size", "draw", "sets"}], in the given order."""
    out, seen = [], set()
    for item in [x for x in spec.split(",") if x]:
        name, _, sets = item.partition("=")
        m = NAME.match(name)
        if not m or name in seen or not sets:
            raise ValueError(f"state {item!r}: <name>=<set>+<set>..., name [R|R2]<A|P|T|E><size>, each name once")
        seen.add(name)
        out.append({"name": name, "kind": "control" if m.group(1) else "rule", "draw": m.group(1) or None, "rule": m.group(2),
                    "size": float(m.group(3)), "sets": [s for s in sets.split("+") if s]})
    if not out:
        raise ValueError("no states")
    return out


def state_masters(st: dict, fu: dict, eps: float) -> tuple:
    """-> (masters, delta as float64 list): theta0 + size x the rule's delta, or for a control theta0 + a random direction with the per-tensor
    norms of that rule state's fp32 update (the step session's control construction)."""
    d = RU.rule_delta(st["rule"], fu["grads"], fu["coef"], fu["d_adam"], st["size"], eps if st["rule"] == "E" else None, fu["moments"])
    masters = RU.add_delta(fu["theta0"], d)
    if st["kind"] == "control":
        masters = SS.control_state(fu["theta0"], masters, fu["names"], SEEDS[st["draw"]])
        masters = [torch.where(c == t0, t0, c) for c, t0 in zip(masters, fu["theta0"])]     # an untouched element keeps theta0's bits
        d = [c.double() - t0.double() for c, t0 in zip(masters, fu["theta0"])]
    return masters, d


def round_trip_rows_of(plan_dir: str, rt_group: int) -> list:
    """The session's round-trip rows, from the plan's row lengths alone: the two longest and the four shortest rows of the group."""
    with np.load(os.path.join(plan_dir, "rows.npz"), allow_pickle=False) as z:
        T, g = np.array(z["T"]), np.array(z["group_idx"])
    grp = {int(n): int(T[n]) for n in range(len(T)) if int(g[n]) == int(rt_group)}
    if not grp:
        raise ValueError(f"round-trip group {rt_group} is not in the plan")
    return pick_round_trip_rows(grp)


def build(ckpt_dir: str, grad_path: str, states: list, out_dir: str, export_fn, blocks: dict, eps: float = 1e-6, log=MI.log,
          plan_dir: str = "", rt_group: int = 0) -> dict:
    """Card-free: every state's adapter directory under <out>/adapters/<name>/ and <out>/rules.json."""
    fu = RU.load_first_update(ckpt_dir, grad_path, MI.sha256, SS.state_sha)
    rt_rows = round_trip_rows_of(plan_dir, rt_group) if plan_dir else None
    nm, meta = fu["names"], fu["meta"]
    gj = json.load(open(os.path.splitext(grad_path)[0] + ".json"))
    N = float(gj["N"])
    G = [-g.double() for g in fu["grads"]]                    # the gradient of sum A sum_t log pi over N (ascent sense), before the clip
    dA = [d.double() for d in fu["d_adam"]]
    score = [i for i, n in enumerate(nm) if n.split(".", 1)[1] in ("qA", "qB", "kA", "kB", "rA", "rB")]
    os.makedirs(os.path.join(out_dir, "adapters"), exist_ok=True)
    rows, checks = [], {"overall_norm_vs_adam_step": {}, "per_tensor_norm_vs_adam_step": {}, "score_adapters_untouched": {}, "A_is_theta0_plus_size_x_step": {}}
    for st in states:
        masters, d = state_masters(st, fu, eps)
        row = RU.table_row(st["name"], d, fu["d_adam"], nm, blocks)
        row.update(name=st["name"], kind=st["kind"], rule=st["rule"], size=st["size"], eps=eps if st["rule"] == "E" else None, sets=st["sets"],
                   of=None if st["kind"] == "rule" else f"{st['rule']}{st['size']:g}", seed=SEEDS.get(st["draw"]),
                   per_tensor_l2=RU.per_tensor_norms(d, nm), masters_sha256=SS.state_sha(masters, nm))
        fo = N * sum(float((g * x).sum()) for g, x in zip(G, d))
        row["first_order"] = {"at_size": fo, "size_1": fo / st["size"], "N": N, "what": "N x <-grad, d>"}
        sha = export_fn(masters, os.path.join(out_dir, "adapters", st["name"]), f"rule-test-{st['name']}")
        row["adapter_sha256"] = sha[0] if isinstance(sha, tuple) else sha
        checks["score_adapters_untouched"][st["name"]] = all(torch.equal(masters[i], fu["theta0"][i]) for i in score)
        s, na = st["size"], RU._norm(dA)
        if st["kind"] == "rule":
            if st["rule"] in ("P", "E", "A"):
                checks["overall_norm_vs_adam_step"][st["name"]] = abs(row["l2"] / s - na) / na if na > 0 else 0.0
            if st["rule"] in ("T", "A"):
                checks["per_tensor_norm_vs_adam_step"][st["name"]] = max(
                    (abs(float(x.norm()) / s - float(a.norm())) / float(a.norm()) for x, a in zip(d, dA) if float(a.norm()) > 0), default=0.0)
            if st["rule"] == "A":
                want = RU.add_delta(fu["theta0"], [a * s for a in dA])
                checks["A_is_theta0_plus_size_x_step"][st["name"]] = all(torch.equal(m, w) for m, w in zip(masters, want))
        rows.append(row)
        log(f"{st['name']}: l2 {row['l2']:.6e}, cos with the optimizer's step {row['cos_with_adam_step']:+.4f}, non-zero {row['nonzero_entries']}, "
            f"first order {fo:+.4f}, adapter {row['adapter_sha256'][:16]}")
        del masters, d
    a1 = RU.add_delta(fu["theta0"], dA)                       # A at size 1 must be the checkpoint itself, to the bit and to the file
    sha1 = export_fn(a1, os.path.join(out_dir, "adapters", "_A1_check"), f"step-session-u{int(meta['update'])}-v")
    sha1 = sha1[0] if isinstance(sha1, tuple) else sha1
    checks["A_at_size_1"] = {"masters_equal_checkpoint": SS.state_sha(a1, nm) == meta["masters_sha256"],
                             "adapter_sha256": sha1, "checkpoint_adapter_sha256": meta.get("adapter_sha256"),
                             "adapter_equal_checkpoint": sha1 == meta.get("adapter_sha256")}
    ok = (all(v <= 1e-6 for v in checks["overall_norm_vs_adam_step"].values()) and all(v <= 1e-6 for v in checks["per_tensor_norm_vs_adam_step"].values())
          and all(checks["score_adapters_untouched"].values()) and all(checks["A_is_theta0_plus_size_x_step"].values())
          and checks["A_at_size_1"]["masters_equal_checkpoint"] and checks["A_at_size_1"]["adapter_equal_checkpoint"])
    tab = {"checkpoint": {"update": meta["update"], "masters_sha256": meta["masters_sha256"], "adapter_sha256": meta.get("adapter_sha256"),
                          "session_id": meta.get("session_id")},
           "grad_sha256": fu["grad_sha256"], "N": N, "clip_coefficient": fu["coef"], "lr": meta.get("lr"), "rule_E_eps": eps,
           "moment_check_worst_rel": fu["moment_check_worst_rel"], "adam_step_l2": RU._norm(dA), "grad_l2": RU._norm(fu["grads"]),
           "control_seeds": SEEDS, "round_trip_group": int(rt_group) if rt_rows is not None else None, "round_trip_rows": rt_rows,
           "states": rows, "checks": checks, "checks_pass": bool(ok)}
    SS.write_json(os.path.join(out_dir, "rules.json"), tab)
    log(f"rules.json: {len(rows)} states; checks {'PASS' if ok else 'FAIL'}: {json.dumps({k: v for k, v in checks.items() if k != 'score_adapters_untouched'}, default=float)[:600]}")
    return tab


def run_states(tr, plan, fu: dict, tab: dict, run_dir: str, adapters_dir: str, rt_group: int, session_id: str, eps: float, log=MI.log) -> int:
    """The trainer side of the card session (one worker): per state the trainer's log-probs of the round-trip rows and the hand-over."""
    nm = fu["names"]
    os.makedirs(os.path.join(run_dir, "states"), exist_ok=True)
    grp = [r for r in plan.rows if r.group_idx == rt_group]
    pick = set(pick_round_trip_rows({int(r.notes["plan_row"]): r.T for r in grp}))
    rt = [r for r in grp if int(r.notes["plan_row"]) in pick]
    if tab.get("round_trip_rows") is not None and (sorted(pick) != sorted(tab["round_trip_rows"]) or tab.get("round_trip_group") != rt_group):
        raise SS.PlanError("rules.json names other round-trip rows than this plan and group give")
    start = [m.detach().clone() for m in tr.opt.masters]
    if not all(torch.equal(a, b) for a, b in zip(start, fu["theta0"])):      # by value: the zeros' sign bits are the trainer's own
        raise SS.GateFailure("the trainer's start adapter is not the checkpoint's theta0")
    lz = tr.logprobs(rt)
    differ = sum(MI.bits_equal(np.concatenate([[0.0], x]), r.engine_lp, r.loss_mask)[1] for r, x in zip(rt, lz))
    SS.write_json(os.path.join(run_dir, "trainer_start.json"), {"session_id": session_id, "round_trip_group": rt_group,
                                                                "round_trip_rows": sorted(pick), "replay_differ": differ,
                                                                "loss_tokens": int(sum(int(r.loss_mask.sum()) for r in rt))})
    if differ:
        raise SS.GateFailure(f"REPLAY GATE: {differ} loss tokens of the round-trip rows differ from the records at the start adapter")
    by = {s["name"]: s for s in parse_states(",".join(f"{r['name']}={'+'.join(r['sets'])}" for r in tab["states"]))}
    for row in tab["states"]:
        name, t0 = row["name"], time.time()
        masters, _ = state_masters(by[name], fu, eps)
        if SS.state_sha(masters, nm) != row["masters_sha256"]:
            raise SS.GateFailure(f"state {name}: the masters rebuilt here do not hash to rules.json's")
        ad = os.path.join(adapters_dir, name)
        if MI.sha256(os.path.join(ad, "adapter_model.safetensors")) != row["adapter_sha256"]:
            raise SS.PlanError(f"state {name}: the shipped adapter is not the file rules.json names")
        SS.put_masters(tr, masters)
        del masters
        lu = tr.logprobs(rt)
        sd = os.path.join(run_dir, "states", name, "roundtrip")
        os.makedirs(sd, exist_ok=True)
        p = os.path.join(sd, "trainer.npz")
        with open(p + ".tmp", "wb") as f:
            np.savez(f, **{f"lp_{r.notes['plan_row']}": MI.lp_at_mask(np.concatenate([[0.0], x]), r.loss_mask) for r, x in zip(rt, lu)})
        os.replace(p + ".tmp", p)
        SS.write_json(os.path.join(run_dir, "states", name, "TRAINER_READY.json"),
                      {"state": name, "session_id": session_id, "adapter": {"path": ad, "sha256": row["adapter_sha256"], "lora_name": f"rule_{name}"},
                       "sets": row["sets"], "round_trip_rows": sorted(pick), "files": {os.path.relpath(p, run_dir): MI.sha256(p)}})
        log(f"state {name}: handed to the read side ({time.time() - t0:.0f} s)")
        for f in ("reader.exit",):                            # a read side that has ended cannot read the remaining states
            fp = os.path.join(run_dir, f)
            if os.path.exists(fp) and open(fp).read().strip() != "0":
                code = open(fp).read().strip()
                raise SS.SessionStop(SS.EXIT_ROUNDTRIP if code == "21" else SS.EXIT_READS, 0, f"the read side ended (exit {code})", "n/a")
    SS.put_masters(tr, start)
    SS.write_json(os.path.join(run_dir, "STATES_DONE.json"), {"session_id": session_id, "states": [r["name"] for r in tab["states"]], "time": time.time()})
    return 0


def phase_build(a) -> int:
    import t8_skyrl as KS

    try:
        tab = build(a.resume, a.grad, parse_states(a.states), a.out, SS.real_export, KS.BLOCKS, a.rule_eps, plan_dir=a.plan, rt_group=a.rt_group)
    except (ValueError, OSError, KeyError) as e:
        print(f"REFUSED: {type(e).__name__}: {e}", flush=True)
        return SS.EXIT_INPUT
    return 0 if tab["checks_pass"] else SS.EXIT_GATE


def phase_states(a) -> int:
    import t8_step as TS

    def _term(signum, frame):
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, _term)
    try:
        plan = SS.load_step_plan(a.plan)
        tab = SS.read_json(os.path.join(a.build, "rules.json"))
        if not tab.get("checks_pass"):
            raise SS.PlanError("rules.json's checks did not pass")
        fu = RU.load_first_update(a.resume, a.grad, MI.sha256, SS.state_sha)
        if fu["grad_sha256"] != tab["grad_sha256"] or fu["meta"]["masters_sha256"] != tab["checkpoint"]["masters_sha256"]:
            raise SS.PlanError("rules.json was built from another checkpoint or gradient")
    except (ValueError, OSError, KeyError) as e:
        print(f"REFUSED: {type(e).__name__}: {e}", flush=True)
        return SS.EXIT_INPUT
    sid = open(os.path.join(a.run_dir, "session_id.txt")).read().strip()
    card = int(a.cards.split(",")[0])
    tr = TS.T8Trainer(MI.trainer_cfg(a, (card,)))
    rc, why = 1, ""
    try:
        rc = run_states(tr, plan, fu, tab, a.run_dir, os.path.join(a.build, "adapters"), a.rt_group, sid, float(tab["rule_E_eps"]))
    except SS.GateFailure as e:
        rc, why = SS.EXIT_GATE, str(e)
    except SS.PlanError as e:
        rc, why = SS.EXIT_INPUT, str(e)
    except SS.SessionStop as e:
        rc, why = e.code, e.reason
    finally:
        tr.close()
    SS.write_json(os.path.join(a.run_dir, "trainer_end.json"), {"exit": rc, "reason": why, "time": time.time()})
    print(f"RULE TEST, TRAINER SIDE: exit {rc} {why}", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("build", "states"))
    ap.add_argument("--resume", required=True, help="the first update's checkpoint directory")
    ap.add_argument("--grad", required=True, help="its grad.pt (grad.json beside it)")
    ap.add_argument("--states", default="", help="build: <name>=<set>+<set>,... in the order they will be read")
    ap.add_argument("--rule-eps", type=float, default=1e-6, help="build: rule E's eps")
    ap.add_argument("--out", default="", help="build: the directory for rules.json and adapters/")
    ap.add_argument("--build", default="", help="states: the build directory (rules.json, adapters/)")
    ap.add_argument("--plan", default="")
    ap.add_argument("--rt-group", type=int, default=0)
    ap.add_argument("--bin", default="")
    ap.add_argument("--artifact", default="")
    ap.add_argument("--cards", default="0")
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--pattern-grad-from", type=int, default=0)
    ap.add_argument("--gate-grad-from", type=int, default=0)
    a = ap.parse_args()
    a.bank_grad_from = 0
    return {"build": phase_build, "states": phase_states}[a.phase](a)


if __name__ == "__main__":
    sys.exit(main())
