"""Step session, trainer side: the optimizer updates of one PPO step taken ONE AT A TIME, each followed by a checkpoint, the export of the
accumulated adapter and of two same-norm random controls, the trainer half of a bitwise round trip, and a hand-over to the read side
(tests/t8_stepsession_reads.py, its own process on serving engines beside the trainer). The next update is applied only after an external
verdict file says GO for the previous one. The arithmetic of the updates is t8_step.ppo_batch's, opened up: the old log-probs of ALL the step's
rows at the start weights, then per update compute(seeds="ppo", minibatch_index) and apply.

Inputs (read-only): a plan directory in the micro-step driver's form (t8_micro.load_plan) whose rows.json also carries
  "updates": [[group ids of update 1], [group ids of update 2], ...]   every group in exactly one update, in row order (refused otherwise);
  "round_trip_group": [one group id per update]                        optional; default: the first group of each update;
  "first_update_number": n                                             optional, default 1: the number of the plan's first update in the run;
  "start_version": s and a per-row vector "version" in rows.npz         optional; a row of another version than s is REFUSED (no lagged path).
Per update <n> the session writes under <run>/updates/<n>/:
  step.json, first_order.json, sketches/, ckpt/ (masters.pt, optimizer.npz, meta.json), adapters.json (v, R, R2), roundtrip/trainer_<name>.npz,
  TRAINER_READY.json (the hand-over; written last, atomically).
The read side adds roundtrip/result.json, reads_dmg/<panel>/{v,R,R2,base2}.npz and READS_DONE; the loop side adds VERDICT.json.
  v  = the accumulated adapter after update n;  R, R2 = theta0 + random_like(v - theta0), two seeds (per tensor the accumulated update's norm,
  exactly zero where the update is zero).
Handshake: before update n + 1 is APPLIED (its gradient is computed meanwhile) the session needs, for update n: the round trip "equal" and a
VERDICT.json {"update": n, "session_id", "outcome": "GO", "judged": {path: sha256}} whose judged hashes equal the session's own hash of every
file READS_DONE lists. STOP / INCONCLUSIVE: nothing is applied, v_n is labelled "not kept". Malformed verdict or timeout: nothing is applied,
v_n is labelled "not judged". Either way STOPPED.json names last_adapter_above_line = n - 1 and the session ends (exit 20). After the last
update the trainer is closed first, then its verdict awaited; TRAINER_DONE.json is written on GO.
Resume (--resume <ckpt dir>): masters and optimizer state restored (hashes checked, theta0 and the re-exported adapter compared with the
checkpoint's record). A checkpoint of update u continues (A) the same plan at update u + 1 (needs that session's old log-probs file) or (B) a
plan whose first update is u + 1 (old log-probs recomputed at the restored weights). The restored state is exported, round-tripped and read
again under the new session, and needs its own GO before the next apply.
Exit codes: 0 complete; 20 stopped by a verdict or its timeout; 21 round trip not bitwise; 4 a trainer gate; 5 inputs; 6 the read side failed.
Phases: check (the plan), trainer (the session), finish (STEP_DONE), stub_judge (TEST ONLY: writes verdicts from a fixed list).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
import threading
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import t8_micro as MI  # noqa: E402

EXIT_OK, EXIT_GATE, EXIT_INPUT, EXIT_READS, EXIT_STOPPED, EXIT_ROUNDTRIP = 0, 4, 5, 6, 20, 21
CONTROLS = (("R", MI.R_SEED), ("R2", MI.R_SEED_B))            # the control's name and the seed of its draw
ADAPTER_NAMES = ("v",) + tuple(n for n, _ in CONTROLS)
OUTCOMES = ("GO", "STOP", "INCONCLUSIVE")


class PlanError(ValueError):
    """The plan is not in the agreed form (exit 5)."""


class SessionStop(Exception):
    """The session ends here without another apply. code: 20 verdict / timeout, 21 round trip, 6 the read side."""

    def __init__(self, code: int, update: int, reason: str, label: str, outcome: str | None = None):
        super().__init__(reason)
        self.code, self.update, self.reason, self.label, self.outcome = code, update, reason, label, outcome


class GateFailure(Exception):
    """A trainer-side gate of the session (exit 4)."""


class EarlyStop(BaseException):
    """Raised in the main thread while a gradient is being computed: the previous update can no longer get a GO."""


def write_json(path: str, obj, **kw) -> None:
    """Atomic: a reader never sees half a file."""
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=float, **kw)
    os.replace(path + ".tmp", path)


def read_json(path: str):
    with open(path) as f:
        return json.load(f)


def state_sha(state: list, nm: list) -> str:
    """sha256 over the tensors' names and fp32 bytes, in the trainer's order."""
    h = hashlib.sha256()
    for t, n in zip(state, nm):
        h.update(n.encode("utf-8"))
        h.update(np.ascontiguousarray(t.detach().cpu().numpy().astype("<f4")).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------------------- the plan
class StepPlan:
    def __init__(self, rows, info, updates, rt_groups, first, start_version, group_ids):
        self.rows, self.info, self.updates, self.rt_groups = rows, info, updates, rt_groups
        self.first, self.start_version, self.group_ids = first, start_version, group_ids

    def number(self, j: int) -> int:
        return self.first + j

    def rt_rows(self, j: int) -> list:
        g = self.rt_groups[j]
        return [r for r in self.rows if r.group_idx == g]


def load_step_plan(plan_dir: str, cap: int = MI.CAP) -> StepPlan:
    try:
        rows, info = MI.load_plan(plan_dir, cap=cap)
    except ValueError as e:
        raise PlanError(str(e)) from e
    meta = read_json(os.path.join(plan_dir, "rows.json"))
    if not isinstance(meta, dict) or not isinstance(meta.get("updates"), list) or not meta["updates"]:
        raise PlanError('rows.json has no "updates" list')
    ups = meta["updates"]
    if any(not isinstance(u, list) or not u for u in ups):
        raise PlanError('"updates": every update is a non-empty list of group ids')
    seq = []
    for r in rows:                                            # the groups in row order (load_plan has checked that each is contiguous)
        if not seq or seq[-1] != r.group_idx:
            seq.append(r.group_idx)
    flat = [int(g) for u in ups for g in u]
    if flat != seq:
        raise PlanError(f'"updates" must list every group exactly once, in the row order: updates {flat}, rows {seq}')
    updates = [[i for i, r in enumerate(rows) if r.group_idx in {int(g) for g in u}] for u in ups]
    rt = meta.get("round_trip_group")
    if rt is None:
        rt = [int(u[0]) for u in ups]
    if not isinstance(rt, list) or len(rt) != len(ups) or any(int(g) not in seq for g in rt):
        raise PlanError('"round_trip_group": one group id of the plan per update')
    first = int(meta.get("first_update_number", 1))
    if first < 1:
        raise PlanError('"first_update_number" must be >= 1')
    sv = int(meta.get("start_version", 0))
    z = np.load(os.path.join(plan_dir, "rows.npz"), allow_pickle=False)
    if "version" in z.files:
        v = np.asarray(z["version"])
        if v.shape != (len(rows),) or bool((v != sv).any()):
            raise PlanError(f'rows of another adapter version than start_version {sv}: the lagged-row path is not built; refused')
    info = dict(info, updates=[len(u) for u in updates], update_groups=[[int(g) for g in u] for u in ups], first_update_number=first,
                start_version=sv, round_trip_group=[int(g) for g in rt],
                update_loss_tokens=[int(sum(int(rows[i].loss_mask.sum()) for i in u)) for u in updates],
                update_row_tokens=[int(sum(rows[i].T for i in u)) for u in updates])
    return StepPlan(rows, info, updates, [int(g) for g in rt], first, sv, seq)


# ------------------------------------------------------------------------------------------------------------------------------- states
def masters_list(tr) -> list:
    return [m.detach().clone() for m in tr.opt.masters]


def put_masters(tr, state: list) -> None:
    """The masters replaced in place (the optimizer's state, keyed by these tensors, is untouched), work values and workers refreshed."""
    with torch.no_grad():
        for m, s in zip(tr.opt.masters, state):
            m.copy_(s)
    tr.opt.sync_work()
    tr._refresh()


def control_state(theta0: list, cur: list, nm: list, seed: int) -> list:
    """theta0 + random_like(cur - theta0): per tensor a fixed-seed Gaussian direction scaled to the accumulated update's norm (float64 sum, rounded
    once to fp32); exactly theta0 where the update is zero. One tensor at a time (t8_micro.random_like draws each tensor from its name alone)."""
    out = []
    for t0, c, n in zip(theta0, cur, nm):
        r = MI.random_like([c - t0], [n], 1.0, seed)[0]
        out.append((t0.double() + r).float())
    return out


def save_ckpt(tr, d: str, nm: list, meta: dict) -> dict:
    os.makedirs(d, exist_ok=True)
    mp, op = os.path.join(d, "masters.pt"), os.path.join(d, "optimizer.npz")
    torch.save({n: m.detach().clone() for n, m in zip(nm, tr.opt.masters)}, mp + ".tmp")
    os.replace(mp + ".tmp", mp)
    arrs = tr.opt.state_arrays()
    with open(op + ".tmp", "wb") as f:
        np.savez(f, **arrs)
    os.replace(op + ".tmp", op)
    g = tr.opt.opt.param_groups[0]
    meta = dict(meta, sha256={"masters.pt": MI.sha256(mp), "optimizer.npz": MI.sha256(op)}, masters_sha256=state_sha(tr.opt.masters, nm),
                optimizer={"betas": [float(b) for b in g["betas"]], "eps": float(g["eps"]), "weight_decay": float(g["weight_decay"]),
                           "max_grad_norm": float(tr.opt.max_grad_norm), "state_tensors": len(arrs) // 3})
    write_json(os.path.join(d, "meta.json"), meta)
    return meta


def load_ckpt(tr, d: str, nm: list) -> dict:
    """Masters and optimizer state from a checkpoint directory, every recorded hash checked. -> its meta."""
    meta = read_json(os.path.join(d, "meta.json"))
    mp, op = os.path.join(d, "masters.pt"), os.path.join(d, "optimizer.npz")
    for name, path in (("masters.pt", mp), ("optimizer.npz", op)):
        if MI.sha256(path) != meta["sha256"][name]:
            raise PlanError(f"checkpoint {d}: {name} does not match the hash in meta.json")
    saved = torch.load(mp, map_location="cpu", weights_only=True)
    if list(saved) != list(nm) or any(tuple(saved[n].shape) != tuple(m.shape) for n, m in zip(nm, tr.opt.masters)):
        raise PlanError(f"checkpoint {d}: tensor names or shapes differ from this trainer's")
    g = tr.opt.opt.param_groups[0]
    o = meta["optimizer"]
    if [float(b) for b in g["betas"]] != o["betas"] or float(g["eps"]) != o["eps"] or float(g["weight_decay"]) != o["weight_decay"] \
            or float(tr.opt.max_grad_norm) != o["max_grad_norm"]:
        raise PlanError(f"checkpoint {d}: optimizer settings differ from this trainer's")
    put_masters(tr, [saved[n] for n in nm])
    with np.load(op, allow_pickle=False) as z:
        arrs = {k: z[k] for k in z.files}
    if len(arrs) != 3 * len(nm):
        raise PlanError(f"checkpoint {d}: optimizer state for {len(arrs) // 3} of {len(nm)} tensors")
    tr.opt.load_state_arrays(arrs)
    if state_sha(tr.opt.masters, nm) != meta["masters_sha256"]:
        raise PlanError(f"checkpoint {d}: the restored masters do not hash to the recorded value")
    return meta


# ------------------------------------------------------------------------------------------------------------------------------- verdicts
def check_verdict(run_dir: str, n: int, session_id: str):
    """The loop side's verdict of update n. -> (outcome, None) when well formed, (None, problem) otherwise. A verdict is accepted only if it
    names this update and session, and its judged hashes equal this process's own hash of every file READS_DONE lists."""
    ud = os.path.join(run_dir, "updates", str(n))
    try:
        v = read_json(os.path.join(ud, "VERDICT.json"))
    except (OSError, ValueError) as e:
        return None, f"VERDICT.json unreadable: {e}"
    try:
        rd = read_json(os.path.join(ud, "READS_DONE"))
    except (OSError, ValueError) as e:
        return None, f"a verdict without a readable READS_DONE: {e}"
    if not isinstance(v, dict) or not isinstance(rd, dict):
        return None, "VERDICT.json or READS_DONE is not an object"
    if rd.get("update") != n or rd.get("session_id") != session_id:
        return None, "READS_DONE names another update or session"
    if v.get("update") != n or v.get("session_id") != session_id:
        return None, f"the verdict names update {v.get('update')!r}, session {v.get('session_id')!r}"
    if v.get("outcome") not in OUTCOMES:
        return None, f"outcome {v.get('outcome')!r}"
    judged, files = v.get("judged"), rd.get("files")
    if not isinstance(judged, dict) or not isinstance(files, dict) or not files:
        return None, "no judged hashes, or READS_DONE lists no file"
    for rel, h in sorted(files.items()):
        p = os.path.join(run_dir, rel)
        if not os.path.isfile(p):
            return None, f"{rel}: listed in READS_DONE, not on disk"
        own = MI.sha256(p)
        if own != h:
            return None, f"{rel}: changed after READS_DONE"
        if judged.get(rel) != own:
            return None, f"{rel}: not judged, or judged with another hash"
    return v["outcome"], None


class Watch:
    """While a gradient is computed: if the previous update's verdict turns out not to be a GO (or its round trip differs, or the read side
    died), interrupt the main thread, so that no card time goes into a gradient that will not be applied."""

    def __init__(self, session, n):
        self.s, self.n, self.ev, self.t = session, n, threading.Event(), None

    def __enter__(self):
        if self.n is not None and self.s.o["early_stop"] and threading.current_thread() is threading.main_thread():
            self.s.armed = True
            self.t = threading.Thread(target=self.run, daemon=True)
            self.t.start()
        return self

    def run(self):
        main = threading.main_thread().ident
        while not self.ev.wait(self.s.o["poll_s"]):
            try:
                bad = self.s.bad_news(self.n)
            except Exception:                                 # a file caught in mid-write: look again
                bad = None
            if bad and self.s.armed:
                signal.pthread_kill(main, signal.SIGUSR1)
                return

    def __exit__(self, *a):
        self.s.armed = False
        self.ev.set()
        if self.t is not None:
            self.t.join(10)


class MemLow:
    """The lowest MemAvailable (GB) seen while the session runs."""

    def __init__(self):
        self.low, self.ev = 1e9, threading.Event()
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()

    def run(self):
        while not self.ev.wait(2.0):
            try:
                for line in open("/proc/meminfo"):
                    if line.startswith("MemAvailable"):
                        self.low = min(self.low, int(line.split()[1]) / 1048576)
            except OSError:
                pass

    def take(self) -> float:
        """The low since the last call."""
        low, self.low = self.low, 1e9
        return low


def self_hwm_gib() -> float:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM"):
                return int(line.split()[1]) * 1024 / 2 ** 30
    except OSError:
        pass
    return 0.0


# ------------------------------------------------------------------------------------------------------------------------------- the session
class Session:
    """tr: a trainer with opt (masters, sync_work, state_arrays, load_state_arrays), _refresh, logprobs, compute, apply, shapes.
    export_fn(state, dst_dir, tag) -> (sha256, record): the adapter file of a parameter state."""

    def __init__(self, tr, plan: StepPlan, run_dir: str, work_dir: str, session_id: str, export_fn, blocks: dict, version=None, **opts):
        self.tr, self.plan, self.run, self.work, self.sid, self.export_fn, self.blocks = tr, plan, run_dir, work_dir, session_id, export_fn, blocks
        self.o = {"lr": MI.LR, "verdict_timeout_s": 1200.0, "reads_timeout_s": 2700.0, "poll_s": 2.0, "early_stop": True,
                  "backward_options": {}, "resume": "", "old_lp": "", "plan_dir": "", "cards": [], "close_fn": None, "abort_fn": None,
                  "zero_check_fn": None}
        unknown = set(opts) - set(self.o)
        if unknown:
            raise ValueError(f"unknown options {sorted(unknown)}")
        self.o.update(opts)
        self.version = plan.start_version if version is None else version
        self.keys = tuple(tr.shapes[0])
        self.nm = MI.names(len(tr.shapes), self.keys)
        self.cur, self.pending, self.armed, self.j0 = None, None, False, 0
        self.handoff, self.reads_seen = {}, {}
        self.mem = MemLow()
        self.sess = {"session_id": session_id, "time_start": time.time(), "plan": plan.info, "lr": self.o["lr"],
                     "backward_options": self.o["backward_options"], "cards": list(self.o["cards"]), "control_seeds": dict(CONTROLS),
                     "verdict_timeout_s": self.o["verdict_timeout_s"], "reads_timeout_s": self.o["reads_timeout_s"], "updates": {},
                     "phases": [], "exit": None}
        os.makedirs(os.path.join(run_dir, "updates"), exist_ok=True)

    # ------------------------------------------------------------------ small things
    def ud(self, n: int) -> str:
        return os.path.join(self.run, "updates", str(n))

    def phase(self, what: str, **kw) -> None:
        self.sess["phases"].append({"t": round(time.time() - self.sess["time_start"], 1), "what": what, **kw})
        self.sess["coordinator_VmHWM_gib"] = self_hwm_gib()
        write_json(os.path.join(self.run, "session.json"), self.sess)
        MI.log(what + ("" if not kw else " " + json.dumps(kw, default=float)))

    def close_trainer(self) -> None:
        if self.tr is not None and self.o["close_fn"]:
            self.o["close_fn"]()
        self.tr = None

    def abort_trainer(self) -> None:
        if self.tr is not None and (self.o["abort_fn"] or self.o["close_fn"]):
            (self.o["abort_fn"] or self.o["close_fn"])()
        self.tr = None

    # ------------------------------------------------------------------ start
    def begin(self) -> None:
        tr, plan, rows = self.tr, self.plan, self.plan.rows
        self.theta0 = masters_list(tr)                        # a new trainer holds the run's start adapter (a fixed function of its seed)
        self.sess["theta0_sha256"] = state_sha(self.theta0, self.nm)
        b_names = [i for i, n in enumerate(self.nm) if n.split(".", 1)[1].endswith("B") or n.split(".", 1)[1] in ("B13", "B2")]
        if any(bool(self.theta0[i].ne(0).any()) for i in b_names):
            raise GateFailure("a new trainer's start adapter has B != 0: the controls and the resume need the zero adapter as theta0")
        resumed, old_file = None, ""
        if self.o["resume"]:
            meta = load_ckpt(tr, self.o["resume"], self.nm)
            if meta.get("theta0_sha256") != self.sess["theta0_sha256"]:
                raise PlanError("the checkpoint's theta0 is not this trainer's start adapter")
            u = int(meta["update"])
            if u == plan.first - 1:                           # (B) the next step's plan: old log-probs at the restored weights
                self.j0 = 0
            elif plan.first <= u < plan.first + len(plan.updates) - 1:   # (A) the same plan, later update: the step's own old log-probs
                self.j0 = u - plan.first + 1
                old_file = self.o["old_lp"]
                if not old_file:
                    raise PlanError("resuming inside a step needs --old-lp (the earlier session's old log-probs at the step's start weights)")
                if meta.get("plan_rows_sha256") != plan.info["rows_sha256"] or MI.sha256(old_file) != meta.get("old_lp_sha256"):
                    raise PlanError("the checkpoint was written for another plan or other old log-probs")
            else:
                raise PlanError(f"checkpoint of update {u}: this plan holds updates {plan.first}..{plan.first + len(plan.updates) - 1}")
            self.cur = masters_list(tr)
            resumed = {"update": u, "meta": meta, "dir": self.o["resume"]}
            self.sess["resumed_from"] = {"dir": self.o["resume"], "update": u, "session_id": meta.get("session_id"),
                                         "kind": "same plan" if old_file else "next plan"}
        elif plan.first != 1:
            raise PlanError(f"a plan whose first update is number {plan.first} needs --resume with the checkpoint of update {plan.first - 1}")
        elif self.o["zero_check_fn"]:
            self.sess["zero_adapter"] = self.o["zero_check_fn"](min(rows, key=lambda r: r.T))
        self.zero_start = resumed is None
        # the old log-probs of ALL the step's rows at the step's start weights (as ppo_batch), and the replay gate on every row before any gradient
        op = os.path.join(self.run, "old_lp.npz")
        if old_file:
            with np.load(old_file, allow_pickle=False) as z:
                olds = [np.asarray(z[f"old_{r.notes['plan_row']}"], dtype=np.float32) for r in rows]
            if os.path.abspath(old_file) != os.path.abspath(op):
                shutil.copyfile(old_file, op)
        else:
            t0 = time.time()
            olds = tr.logprobs(rows)
            self.sess["old_lp_wall_s"] = time.time() - t0
            with open(op + ".tmp", "wb") as f:
                np.savez(f, **{f"old_{r.notes['plan_row']}": np.asarray(x, dtype=np.float32) for r, x in zip(rows, olds)})
            os.replace(op + ".tmp", op)
        self.old_sha = MI.sha256(op)
        eq = ne = 0
        for r, x in zip(rows, olds):
            if np.asarray(x).shape != (r.T - 1,):
                raise GateFailure(f"plan row {r.notes['plan_row']}: {np.asarray(x).shape} old log-probs for T {r.T}")
            r.notes["old_lp"] = np.asarray(x, dtype=np.float32)
            e, d = MI.bits_equal(np.concatenate([[0.0], r.notes["old_lp"]]), r.engine_lp, r.loss_mask)
            eq, ne = eq + e, ne + d
        self.sess["replay"] = {"rows": len(rows), "equal": eq, "differ": ne}
        self.phase("start", replay=self.sess["replay"], resumed=bool(resumed))
        if ne or eq != plan.info["loss_tokens"]:
            raise GateFailure(f"REPLAY GATE: {ne} of {eq + ne} loss tokens differ between the trainer's old log-probs and the rows' records")
        if resumed:                                           # the restored state under THIS session: export, round trip, reads, its own verdict
            self.post_state(resumed["update"], max(self.j0 - 1, 0), want_sha=resumed["meta"].get("adapter_sha256"))
            self.pending = resumed["update"]

    # ------------------------------------------------------------------ one update
    def update(self, j: int) -> None:
        from surogate.inkling_rl.step import StepConfig

        tr, plan = self.tr, self.plan
        n = plan.number(j)
        sub = [plan.rows[i] for i in plan.updates[j]]
        self.phase(f"update {n}: gradient", rows=len(sub))
        t0 = time.time()
        try:
            with Watch(self, self.pending):
                out = tr.compute(sub, self.version, StepConfig(max_policy_lag=None), seeds="ppo", minibatch_index=j)
        except EarlyStop:
            self.abort_trainer()
            self.phase(f"update {n}: gradient interrupted, update {self.pending} cannot get a GO")
            self.await_verdict(self.pending, time.time(), last=False)
            raise SessionStop(EXIT_READS, self.pending, "the gradient was interrupted but the previous update shows no stop", "not judged")
        wall = time.time() - t0
        m = out.metrics
        N = float(m["total_tokens"])
        want_N = float(sum(int(r.loss_mask.sum()) for r in sub))
        step = {"update": n, "index_in_plan": j, "N": N, "rows": len(sub), "rows_used": len(out.rows), "rows_refused": len(sub) - len(out.rows),
                "finite": all(bool(torch.isfinite(g).all()) for g in out.grads), "replay_checked_rows": m["replay_checked_rows"],
                "replay_skipped_rows": m["replay_skipped_rows"], "stale_checked_rows": m["stale_checked_rows"],
                "ratio_max": m.get("ppo_ratio_max"), "clip_share": m.get("ppo_clip_ratio"), "policy_loss": m.get("policy_loss"),
                "tok_per_s_per_card": m["tok_per_s_per_card"], "row_tokens": m["row_tokens"], "gradient_wall_s": wall,
                "zero_seed_rows": m.get("zero_seed_rows"), "backward_options": self.o["backward_options"],
                "workers": [{k: x.get(k) for k in ("card", "VmHWM_gib", "card_peak_torch_gib")} for x in m.get("workers", [])],
                "metrics": {k: v for k, v in m.items() if isinstance(v, (int, float))}}
        bad = []
        if not step["finite"]:
            bad.append("a non-finite gradient")
        if step["rows_refused"] or step["replay_skipped_rows"] or step["replay_checked_rows"] != len(sub):
            bad.append(f"rows refused {step['rows_refused']}, replay skipped {step['replay_skipped_rows']}, checked {step['replay_checked_rows']} of {len(sub)}")
        if N != want_N:
            bad.append(f"N {N:.0f} != the update's loss tokens {want_N:.0f}")
        if j == 0 and not (step["stale_checked_rows"] == len(sub) and step["ratio_max"] == 1.0 and step["clip_share"] == 0.0):
            bad.append(f"first update of the step: stale checked {step['stale_checked_rows']} of {len(sub)}, ratio max {step['ratio_max']}, "
                       f"clip share {step['clip_share']} (must be all rows, 1.0, 0.0)")
        ud = self.ud(n)
        os.makedirs(ud, exist_ok=True)
        if bad:
            write_json(os.path.join(ud, "step.json"), dict(step, gate_failures=bad))
            raise GateFailure(f"update {n}: " + "; ".join(bad))
        step["attempts"] = self.take_sketches(n, sub, out)
        if self.pending is not None:                          # nothing is applied before the previous update has its GO
            self.await_verdict(self.pending, time.time(), last=False)
        prev = self.cur if self.cur is not None else self.theta0
        first_ever = self.cur is None
        step["grad_norm"] = tr.apply(out, self.o["lr"])
        self.cur = masters_list(tr)
        delta = [c - p for c, p in zip(self.cur, prev)]
        fo = MI.first_order(out.grads, N, delta, self.keys, self.blocks)
        a_keys = [i for i, x in enumerate(self.nm) if x.split(".", 1)[1].endswith("A") or x.split(".", 1)[1] in ("A13", "A2")]
        step["A_delta_max_abs"] = max(float(delta[i].abs().max()) for i in a_keys)
        step["first_order"] = fo
        write_json(os.path.join(ud, "first_order.json"),
                   {**fo["by_block"], "N": N, "rows_used": len(out.rows), "rows_refused": 0, "update": n, "G": "-N x this update's gradient",
                    "d": "masters after - masters before this update", "attention": fo["attention"], "experts": fo["experts"],
                    "note": "exactly the advantage-weighted first-order change only where the ratio is 1 (the first update of a step)",
                    "backward_options": self.o["backward_options"]})
        del out, delta, prev
        if first_ever and self.zero_start and step["A_delta_max_abs"] != 0.0:
            write_json(os.path.join(ud, "step.json"), dict(step, gate_failures=["A moved at the first update from B = 0"]))
            raise GateFailure(f"update {n}: A moved by {step['A_delta_max_abs']} at the first update from B = 0")
        step["min_mem_available_gb"] = self.mem.take()
        self.sess["updates"][str(n)] = {k: step[k] for k in ("N", "rows", "policy_loss", "clip_share", "ratio_max", "grad_norm", "gradient_wall_s",
                                                              "tok_per_s_per_card", "min_mem_available_gb")}
        write_json(os.path.join(ud, "step.json"), step)
        self.phase(f"update {n}: applied", grad_norm=step["grad_norm"], clip_share=step["clip_share"], first_order=fo["by_block"]["all"])
        self.post_state(n, j, step=step)
        self.pending = n

    def take_sketches(self, n: int, sub: list, out) -> dict:
        """The update's sketch files moved under updates/<n>/sketches, and per attempt what the signal read needs to know about them."""
        sd = os.path.join(self.ud(n), "sketches")
        files = list(out.metrics.get("sketch_files", []))
        got = {}
        if files:
            os.makedirs(sd, exist_ok=True)
        for f in files:
            d = torch.load(f, map_location="cpu", weights_only=False)
            g = got.setdefault(int(d["trial"]), {"rows": 0, "tensors": 0})
            g["rows"] += int(d["rows"])
            g["tensors"] += len(d["tensors"])
            del d
            shutil.move(f, os.path.join(sd, os.path.basename(f)))
        att = {}
        for res in out.rows:
            r = sub[res["i"]]
            a = att.setdefault(int(r.trial_idx), {"advantage": float(r.advantage), "group": int(r.group_idx), "rows": 0, "loss_tokens": 0,
                                                  "clipped_tokens": 0.0, "lr_sum": 0.0, "log_ratio_abs_max": 0.0, "backward_rows": 0})
            a["rows"] += 1
            a["loss_tokens"] += int(res["loss_tokens"])
            a["clipped_tokens"] += float(res.get("ppo", {}).get("clipped_tokens", 0.0))
            s, mx = res.get("log_ratio_abs", (0.0, 0.0))
            a["lr_sum"] += float(s)
            a["log_ratio_abs_max"] = max(a["log_ratio_abs_max"], float(mx))
            a["backward_rows"] += int(bool(res.get("passes")))
        missing = []
        for t, a in att.items():
            a["clip_share"] = a["clipped_tokens"] / max(a["loss_tokens"], 1)
            a["log_ratio_abs_mean"] = a.pop("lr_sum") / max(a["loss_tokens"], 1)
            g = got.get(t, {"rows": 0, "tensors": 0})
            if a["advantage"] == 0.0:
                a["sketch"] = "absent: advantage 0"
            elif g["rows"] > 0 and g["tensors"] > 0:
                a["sketch"] = "written"
            elif a["clip_share"] == 1.0:
                a["sketch"] = "absent: every loss token clipped"
            else:
                a["sketch"] = "absent"
                missing.append(t)
        if files and missing:                                 # only a session that asked for sketches gates on them
            raise GateFailure(f"update {n}: no sketch for attempts {sorted(missing)[:8]} with a non-zero advantage")
        return {str(t): att[t] for t in sorted(att)}

    # ------------------------------------------------------------------ after an apply (or a resume): checkpoint, adapters, round trip, hand-over
    def post_state(self, n: int, j: int, step: dict | None = None, want_sha: str | None = None) -> None:
        tr, plan = self.tr, self.plan
        ud = self.ud(n)
        os.makedirs(os.path.join(ud, "roundtrip"), exist_ok=True)
        rt = plan.rt_rows(j)
        recs, files = {}, {}

        def trainer_lp(name):
            lu = tr.logprobs(rt)
            p = os.path.join(ud, "roundtrip", f"trainer_{name}.npz")
            with open(p + ".tmp", "wb") as f:
                np.savez(f, **{f"lp_{r.notes['plan_row']}": MI.lp_at_mask(np.concatenate([[0.0], x]), r.loss_mask) for r, x in zip(rt, lu)})
            os.replace(p + ".tmp", p)
            files[os.path.relpath(p, self.run)] = MI.sha256(p)
            return lu

        t0 = time.time()
        sha, rec = self.export_fn(self.cur, os.path.join(self.work, "adapters", f"u{n}_v"), f"step-session-u{n}-v")
        recs["v"] = dict(rec, sha256=sha, path=os.path.join(self.work, "adapters", f"u{n}_v"), lora_name=f"u{n}_v")
        if want_sha is not None and sha != want_sha:
            raise PlanError(f"the restored state exports to {sha[:16]}, the checkpoint recorded {want_sha[:16]}")
        if step is not None:
            save_ckpt(tr, os.path.join(ud, "ckpt"), self.nm,
                      {"update": n, "lr": self.o["lr"], "adapter_sha256": sha, "theta0_sha256": self.sess["theta0_sha256"],
                       "plan_rows_sha256": plan.info["rows_sha256"], "old_lp_sha256": self.old_sha, "session_id": self.sid,
                       "first_update_number": plan.first, "backward_options": self.o["backward_options"]})
        lv = trainer_lp("v")                                  # the masters are v
        for name, seed in CONTROLS:
            st = control_state(self.theta0, self.cur, self.nm, seed)
            sha_c, rec = self.export_fn(st, os.path.join(self.work, "adapters", f"u{n}_{name}"), f"step-session-u{n}-{name}")
            recs[name] = dict(rec, sha256=sha_c, path=os.path.join(self.work, "adapters", f"u{n}_{name}"), lora_name=f"u{n}_{name}", seed=seed)
            put_masters(tr, st)
            del st
            trainer_lp(name)
        put_masters(tr, self.cur)                             # back to v: the masters bit for bit, and the workers' forward bit for bit
        if not all(torch.equal(m, c) for m, c in zip(tr.opt.masters, self.cur)):
            raise GateFailure(f"update {n}: the masters were not restored after the controls' round trip")
        again = tr.logprobs(rt[:1])[0]
        if MI.bits_equal(np.concatenate([[0.0], again]), np.concatenate([[0.0], lv[0]]), rt[0].loss_mask)[1]:
            raise GateFailure(f"update {n}: the workers' forward at v differs after the controls' round trip")
        if all(torch.equal(c, t0) for c, t0 in zip(self.cur, self.theta0)):
            raise GateFailure(f"update {n}: the accumulated update is exactly zero (v is the start adapter; its controls would be too)")
        write_json(os.path.join(ud, "adapters.json"), {k: {x: y for x, y in v.items() if x != "path"} for k, v in recs.items()})
        files[os.path.relpath(os.path.join(ud, "adapters.json"), self.run)] = MI.sha256(os.path.join(ud, "adapters.json"))
        write_json(os.path.join(ud, "TRAINER_READY.json"),
                   {"update": n, "session_id": self.sid, "adapters": {k: {"path": v["path"], "sha256": v["sha256"], "lora_name": v["lora_name"]}
                                                                       for k, v in recs.items()},
                    "round_trip_rows": [int(r.notes["plan_row"]) for r in rt], "round_trip_group": plan.rt_groups[j], "files": files,
                    "resumed_state": step is None})
        self.handoff[n] = time.time()
        self.phase(f"update {n}: handed to the read side", adapters={k: v["sha256"][:16] for k, v in recs.items()},
                   round_trip_rows=len(rt), seconds=round(time.time() - t0, 1))

    # ------------------------------------------------------------------ the wait
    def reader_state(self, n: int):
        """-> None, or (code, reason) when the read side can no longer deliver update n."""
        ud = self.ud(n)
        res = os.path.join(ud, "roundtrip", "result.json")
        if os.path.exists(res):
            r = read_json(res)
            if r.get("result") != "equal":
                return EXIT_ROUNDTRIP, f"round trip of update {n}: {r.get('result')!r}"
        ident = os.path.join(self.run, "base", "roundtrip", "identity.json")
        if os.path.exists(ident) and read_json(ident).get("result") != "equal":
            return EXIT_ROUNDTRIP, "the engines' base read of the round-trip rows differs from the rows' recorded log-probs"
        ex = os.path.join(self.run, "reader.exit")
        if os.path.exists(ex) and not os.path.exists(os.path.join(ud, "READS_DONE")):
            code = open(ex).read().strip()
            return EXIT_READS, f"the read side ended (exit {code}) without the reads of update {n}"
        return None

    def bad_news(self, n: int):
        """What the watcher looks for while the next gradient runs: anything that already rules out a GO for update n."""
        rs = self.reader_state(n)
        if rs:
            return rs[1]
        if os.path.exists(os.path.join(self.ud(n), "VERDICT.json")):
            outcome, problem = check_verdict(self.run, n, self.sid)
            if problem or outcome != "GO":
                return problem or outcome
        return None

    def await_verdict(self, n: int, t_ready: float, last: bool) -> None:
        """Returns on a well-formed GO for update n; raises SessionStop otherwise. Clock: verdict_timeout_s after the next gradient is ready
        (after the last update: after READS_DONE was first seen); the reads themselves must be done reads_timeout_s after the hand-over."""
        ud = self.ud(n)
        deadline = None if last else t_ready + self.o["verdict_timeout_s"]
        reads_deadline = self.handoff.get(n, t_ready) + self.o["reads_timeout_s"]
        self.phase(f"update {n}: waiting for its verdict")
        while True:
            rs = self.reader_state(n)
            if rs:
                raise SessionStop(rs[0], n, rs[1], "not judged")
            if os.path.exists(os.path.join(ud, "VERDICT.json")):
                outcome, problem = check_verdict(self.run, n, self.sid)
                if problem:
                    raise SessionStop(EXIT_STOPPED, n, f"malformed verdict: {problem}", "not judged")
                if outcome == "GO":
                    res = os.path.join(ud, "roundtrip", "result.json")
                    if not os.path.exists(res) or read_json(res).get("result") != "equal":
                        raise SessionStop(EXIT_ROUNDTRIP, n, f"a GO for update {n} without a round trip that reads 'equal'", "not judged")
                    self.sess["updates"].setdefault(str(n), {})["verdict"] = "GO"
                    self.phase(f"update {n}: GO")
                    return
                raise SessionStop(EXIT_STOPPED, n, f"verdict {outcome}", "not kept", outcome)
            now = time.time()
            done = os.path.exists(os.path.join(ud, "READS_DONE"))
            if done and n not in self.reads_seen:
                self.reads_seen[n] = now
                if last:
                    deadline = now + self.o["verdict_timeout_s"]
            if not done and now > reads_deadline:
                raise SessionStop(EXIT_READS, n, f"the reads of update {n} were not done {self.o['reads_timeout_s']:.0f} s after the hand-over", "not judged")
            if deadline is not None and now > deadline:
                raise SessionStop(EXIT_STOPPED, n, f"no verdict {self.o['verdict_timeout_s']:.0f} s after "
                                  + ("READS_DONE" if last else "the next gradient was ready"), "not judged")
            time.sleep(self.o["poll_s"])

    # ------------------------------------------------------------------ ends
    def finish(self) -> None:
        self.close_trainer()                                  # the cards are free while the last reads and their verdict are awaited
        if self.pending is not None:
            self.await_verdict(self.pending, time.time(), last=True)
        last = self.plan.number(len(self.plan.updates) - 1)
        self.sess["exit"] = {"code": EXIT_OK, "reason": "every update applied and judged GO"}
        self.sess["min_mem_available_gb"] = min([self.mem.take()] + [u.get("min_mem_available_gb", 1e9) for u in self.sess["updates"].values()])
        write_json(os.path.join(self.run, "TRAINER_DONE.json"), {"session_id": self.sid, "last_update": last, "time": time.time()})
        self.phase("trainer done")

    def stopped(self, e: SessionStop) -> None:
        n = e.update
        ck = os.path.join(self.ud(n - 1), "ckpt")
        info = {"at_update": n, "reason": e.reason, "outcome": e.outcome, "exit_code": e.code, "session_id": self.sid,
                f"v_{n}": e.label, "last_adapter_above_line": n - 1,
                "last_adapter_note": "0 = the base model" if n - 1 == 0 else f"the accumulated adapter after update {n - 1}",
                "resume_from": os.path.relpath(ck, self.run) if os.path.isdir(ck) else None,
                "resume_note": None if os.path.isdir(ck) or n - 1 == 0 else "the checkpoint of that update is in the session this one resumed from",
                "time": time.time()}
        write_json(os.path.join(self.run, "STOPPED.json"), info)
        self.sess["exit"] = {"code": e.code, "reason": e.reason, "at_update": n}
        self.phase("stopped", **{k: info[k] for k in ("at_update", "reason", "last_adapter_above_line")})

    def failed(self, code: int, why: str) -> None:
        self.sess["exit"] = {"code": code, "reason": why}
        self.phase("failed", code=code, why=why[:500])

    def run_all(self) -> int:
        """The whole session. The trainer is closed (or its workers ended) on every way out."""
        try:
            self.begin()
            for j in range(self.j0, len(self.plan.updates)):
                self.update(j)
            self.finish()
            return EXIT_OK
        except SessionStop as e:
            self.abort_trainer()
            self.stopped(e)
            return e.code
        except GateFailure as e:
            self.abort_trainer()
            self.failed(EXIT_GATE, str(e))
            return EXIT_GATE
        except PlanError as e:
            self.abort_trainer()
            self.failed(EXIT_INPUT, str(e))
            return EXIT_INPUT
        except BaseException as e:
            self.abort_trainer()
            gate = type(e).__name__ == "T8GateError"
            self.failed(EXIT_GATE if gate else 1, f"{type(e).__name__}: {e}")
            if gate:
                return EXIT_GATE
            raise
        finally:
            self.mem.ev.set()


def install_signals(session: Session) -> None:
    def usr1(signum, frame):
        if session.armed:
            raise EarlyStop()

    def term(signum, frame):
        raise SystemExit(143)

    signal.signal(signal.SIGUSR1, usr1)
    signal.signal(signal.SIGTERM, term)


# ------------------------------------------------------------------------------------------------------------------------------- phases
def phase_check(a) -> int:
    try:
        plan = load_step_plan(a.plan)
    except PlanError as e:
        print(f"PLAN REFUSED: {e}", flush=True)
        return EXIT_INPUT
    print(json.dumps(plan.info, indent=1))
    return 0


def phase_trainer(a) -> int:
    import t8_adapter as TA
    import t8_skyrl as KS
    import t8_step as TS

    try:
        plan = load_step_plan(a.plan)
    except PlanError as e:
        print(f"PLAN REFUSED: {e}", flush=True)
        return EXIT_INPUT
    cards = tuple(int(x) for x in a.cards.split(","))
    sid_file = os.path.join(a.run_dir, "session_id.txt")
    sid = open(sid_file).read().strip() if os.path.exists(sid_file) else f"step-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    if a.deadline_min > 0:
        def _late(signum, frame):
            raise TimeoutError(f"the trainer phase passed its {a.deadline_min} min deadline")
        signal.signal(signal.SIGALRM, _late)
        signal.alarm(int(a.deadline_min * 60))
    tr = TS.T8Trainer(MI.trainer_cfg(a, cards, sketch_dir=os.path.join(a.run_dir, "sketches_tmp")))
    n_layers, keys, rank = len(tr.shapes), tr.keys, tr.B.RANK
    attn_rank, attn_alpha = int(tr.cfg.attn_rank), float(tr.cfg.attn_alpha)

    def export(state, dst, tag):
        sha = TA.export_combined(MI.layers_of(state, n_layers, keys), dst, attn_rank, attn_alpha, rank, "BF16", metadata={"step": tag})
        ac = read_json(os.path.join(dst, "adapter_config.json"))
        ex = [int(v) for v in (ac.get("rank_pattern") or {}).values()]
        return sha, {"r": ac["r"], "lora_alpha": ac["lora_alpha"], "rank_pattern": ac.get("rank_pattern"), "alpha_pattern": ac.get("alpha_pattern"),
                     "scale_attention": float(ac["lora_alpha"]) / float(ac["r"]), "scale_experts": [float(ac["lora_alpha"]) / e for e in ex]}

    def abort():
        for h in list(tr.handles):                            # workers may be in the middle of a row: end them, do not wait for it
            p = getattr(h, "proc", None)
            if p is not None and p.is_alive():
                p.terminate()
        for h in list(tr.handles):
            p = getattr(h, "proc", None)
            if p is not None:
                p.join(20)
        tr.handles = []

    def zero_check(row):
        zc = tr.check_zero_adapter([row])
        return {"checks": len(zc), "all_bitwise": all(x.get("log_probs_equal") and x.get("routing_equal") for x in zc)}

    s = Session(tr, plan, a.run_dir, a.work, sid, export, KS.BLOCKS, lr=a.lr, verdict_timeout_s=a.verdict_timeout_min * 60.0,
                reads_timeout_s=a.reads_timeout_min * 60.0, early_stop=not a.no_early_stop, backward_options=MI.backward_options(a),
                resume=a.resume, old_lp=a.old_lp, plan_dir=a.plan, cards=list(cards), close_fn=tr.close, abort_fn=abort, zero_check_fn=zero_check)
    install_signals(s)
    rc = s.run_all()
    signal.alarm(0)
    print(f"STEP SESSION, TRAINER SIDE: exit {rc}: {s.sess['exit']}", flush=True)
    return rc


def phase_finish(a) -> int:
    """STEP_DONE, written last on a complete step: every update has its GO, and the transfer reads (when a transfer plan was given) are there."""
    plan = load_step_plan(a.plan)
    sess = read_json(os.path.join(a.run_dir, "session.json"))
    done = read_json(os.path.join(a.run_dir, "TRAINER_DONE.json"))
    ups = []
    for j in range(len(plan.updates)):
        n = plan.number(j)
        if os.path.isdir(os.path.join(a.run_dir, "updates", str(n), "ckpt")):
            outcome, problem = check_verdict(a.run_dir, n, sess["session_id"])
            if problem or outcome != "GO":
                print(f"update {n}: {problem or outcome}: no STEP_DONE", flush=True)
                return EXIT_STOPPED
            ups.append(n)
    if not ups or ups[-1] != done["last_update"]:
        print("the last update has no checkpoint in this run: no STEP_DONE", flush=True)
        return EXIT_GATE
    tfiles = {}
    if a.transfer:
        for name in ADAPTER_NAMES:
            for ext in (".npz", ".json"):
                p = os.path.join(a.run_dir, "transfer", "reads", name + ext)
                if not os.path.isfile(p):
                    print(f"transfer read {name}{ext} is missing: no STEP_DONE", flush=True)
                    return EXIT_READS
                tfiles[os.path.relpath(p, a.run_dir)] = MI.sha256(p)
    write_json(os.path.join(a.run_dir, "STEP_DONE"), {"session_id": sess["session_id"], "updates": ups, "transfer_files": tfiles, "time": time.time()})
    print(f"STEP_DONE: updates {ups}, transfer reads {len(tfiles) // 2}", flush=True)
    return 0


def phase_stub_judge(a) -> int:
    """TEST ONLY (smokes and CPU tests): stands where the loop side's judge runs. For each update in order it waits for READS_DONE and writes
    VERDICT.json with the next outcome of --outcomes; "NONE" writes nothing (the session's timeout), "BADHASH" a GO with one wrong hash."""
    outcomes = [x for x in a.outcomes.split(",") if x]
    sid = read_json(os.path.join(a.run_dir, "session.json"))["session_id"] if os.path.exists(os.path.join(a.run_dir, "session.json")) else None
    n = a.first_update
    t_end = time.time() + a.deadline_min * 60 if a.deadline_min > 0 else None
    for oc in outcomes:
        ud = os.path.join(a.run_dir, "updates", str(n))
        while not os.path.exists(os.path.join(ud, "READS_DONE")):
            if any(os.path.exists(os.path.join(a.run_dir, f)) for f in ("STOPPED.json", "trainer.exit")) or (t_end and time.time() > t_end):
                print(f"stub judge: the session ended before the reads of update {n}", flush=True)
                return 0
            time.sleep(1.0)
        rd = read_json(os.path.join(ud, "READS_DONE"))
        sid = sid or rd["session_id"]
        if oc == "NONE":
            print(f"stub judge: update {n}: no verdict written (timeout test)", flush=True)
        else:
            judged = {rel: MI.sha256(os.path.join(a.run_dir, rel)) for rel in rd["files"]}
            if oc == "BADHASH":
                k = sorted(judged)[0]
                judged[k] = "0" * 64
            write_json(os.path.join(ud, "VERDICT.json"), {"update": n, "session_id": rd["session_id"], "outcome": "GO" if oc == "BADHASH" else oc,
                                                           "judged": judged, "identity": "STUB JUDGE (test only): no damage number was computed",
                                                           "D": {}, "judge_sha256": None})
            print(f"stub judge: update {n}: wrote {oc}", flush=True)
        n += 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("check", "trainer", "finish", "stub_judge"))
    ap.add_argument("--plan", default="")
    ap.add_argument("--bin", default="")
    ap.add_argument("--artifact", default="")
    ap.add_argument("--cards", default="0", help="the trainer's cards (one worker each)")
    ap.add_argument("--work", default="", help="the run's scratch directory (adapters; not copied back)")
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--lr", type=float, default=MI.LR)
    ap.add_argument("--verdict-timeout-min", type=float, default=20.0)
    ap.add_argument("--reads-timeout-min", type=float, default=45.0)
    ap.add_argument("--resume", default="", help="a checkpoint directory (updates/<n>/ckpt of an earlier session)")
    ap.add_argument("--old-lp", default="", help="resume inside a step: that session's old_lp.npz")
    ap.add_argument("--transfer", default="", help="finish: the transfer plan directory, when its reads are part of the step")
    ap.add_argument("--no-early-stop", action="store_true", help="do not interrupt a running gradient when the previous update is stopped")
    ap.add_argument("--deadline-min", type=float, default=0.0)
    ap.add_argument("--outcomes", default="", help="stub_judge: comma list, one per update: GO, STOP, INCONCLUSIVE, NONE, BADHASH")
    ap.add_argument("--first-update", type=int, default=1, help="stub_judge: the number of the first update it judges")
    ap.add_argument("--pattern-grad-from", type=int, default=0, help="visits below it hold their attention pattern constant in the backward")
    ap.add_argument("--gate-grad-from", type=int, default=0, help="visits below it hold their router gate weights constant in the backward")
    ap.add_argument("--bank-grad-from", type=int, default=0, help="not available in this session (see t8_micro)")
    a = ap.parse_args()
    if a.bank_grad_from:
        ap.error("--bank-grad-from cannot run in this session: its trainer runs with lean_host (the sketches)")
    return {"check": phase_check, "trainer": phase_trainer, "finish": phase_finish, "stub_judge": phase_stub_judge}[a.phase](a)


if __name__ == "__main__":
    sys.exit(main())
