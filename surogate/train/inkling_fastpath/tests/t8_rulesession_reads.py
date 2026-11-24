"""Rule test, read side (tests/t8_rulesession.py holds what a state is): every adapter state read forward-only on the row sets its entry in
rules.json names, on serving engines beside the trainer's one worker. Sets: any panel or plan directory given as <name>=<dir>, and "own" =
the plan's rows of the given groups.

start:  without an adapter: the round-trip rows -> base/roundtrip/base_g<group>.npz (must equal the rows' records bitwise: exit 21 if not),
        then every set -> base/<set>/base.npz (+ .json).
per state, in rules.json's order, when states/<name>/TRAINER_READY.json appears, on EVERY engine:
        load the state; the round-trip rows; its sets; the round-trip rows again in the opposite order; unload.
        round trip, bitwise at every loss token: the first and the last read == the trainer's log-probs of that state (exit 21 if not)
        -> states/<name>/roundtrip/{engine.npz, engine_last.npz, result.json}, states/<name>/reads/<set>/read.npz (+ read.json),
           states/<name>/adapter.json, states/<name>/READS_DONE (the state's files with their sha256, written last).
end:    without an adapter again: the round-trip rows and every set named for the re-read -> base/<set>/base2.npz; every re-read must
        equal the first read bitwise (base/reread.json; exit 21 if not). RULES_DONE lists every file of the run with its sha256.
The row-to-engine split of a set is the same for every read of it. The round-trip rows are a fixed choice from one plan group: its two longest
and its six shortest rows (pick_round_trip_rows).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t8_stepsession_reads as RD  # noqa: E402


def pick_round_trip_rows(T: dict) -> list:
    """From {plan row: length} of one group: the two longest and the six shortest rows (all of them if the group has eight or fewer).
    Eight rows: every read engine of an eight-card session (seven engines) must hold at least one."""
    by = sorted(T, key=lambda n: (T[n], n))
    return sorted(by) if len(by) <= 8 else sorted(by[:6] + by[-2:])


class StateReader(RD.Reader):
    def __init__(self, client, urls, run_dir, plan_dir, sets: dict, own_groups, rt_group: int, reread: list, engine_sha="", poll_s=2.0):
        super().__init__(client, urls, run_dir, plan_dir, {}, "", engine_sha, poll_s)
        bad = [n for n in sets if n in ("rt", "rt2", "plan", "own", "base") or not n or "/" in n]
        if bad:
            raise ValueError(f"set names {bad} are not allowed")
        self.sets = {name: RD.RowSet(name, d) for name, d in sets.items()}
        if own_groups:
            gs = {int(g) for g in own_groups}
            self.sets["own"] = self.plan.subset("own", [n for n in range(self.plan.R) if int(self.plan.group[n]) in gs])
        grp = {n: self.plan.T[n] for n in range(self.plan.R) if int(self.plan.group[n]) == int(rt_group)}
        if not grp:
            raise ValueError(f"round-trip group {rt_group} is not in the plan")
        self.rt_group, self.rt = int(rt_group), self.plan.subset("rt", pick_round_trip_rows(grp))
        if len(self.rt.rows) < len(self.urls):
            raise ValueError(f"{len(self.rt.rows)} round-trip rows for {len(self.urls)} engines (every engine must read at least one)")
        self.reread = [s for s in reread if s]
        if any(s not in self.sets for s in self.reread):
            raise ValueError(f"re-read sets {self.reread}: not among the sets {sorted(self.sets)}")
        self.files = {}

    def write_set(self, d, name, rs, got, adapter_sha):
        self.write_read(d, name, rs, got, adapter_sha, self.files)
        if rs.tag == "own":                                   # a subset of the plan: its reads name the plan rows they hold
            jp = os.path.join(d, name + ".json")
            meta = json.load(open(jp))
            meta["rows"] = rs.rows
            RD.write_json(jp, meta)
            self.files[self.rel(jp)] = RD.sha256(jp)

    def base_pass(self, name: str, sets: list) -> int:
        """The round-trip rows and the given sets without an adapter. name "base": the first read; "base2": the re-read, compared with it."""
        t0 = time.time()
        res = RD.run_pass(self.c, self.urls, [(name, None, None)], [self.rt] + [self.sets[s] for s in sets])
        got = res[("rt", name)]
        rd = os.path.join(self.run, "base", "roundtrip")
        os.makedirs(rd, exist_ok=True)
        if name == "base":
            p = os.path.join(rd, f"base_g{self.rt_group}.npz")
            RD.save_npz(p, **{f"lp_{r}": got[r] for r in self.rt.rows})
            self.files[self.rel(p)] = RD.sha256(p)
            self.rt_base = got
            differ = sum(RD.bits_differ(got[r], self.rt.recorded(r), self.rt.mask(r)) for r in self.rt.rows)
            tokens = int(sum(int(self.rt.mask(r).sum()) for r in self.rt.rows))
            ip = os.path.join(rd, "identity.json")
            RD.write_json(ip, {"result": "equal" if differ == 0 and tokens > 0 else "differ", "round_trip_group": self.rt_group,
                               "round_trip_rows": self.rt.rows, "loss_tokens": tokens, "differ_from_recorded": differ})
            self.files[self.rel(ip)] = RD.sha256(ip)
            for s in sets:
                self.write_set(os.path.join(self.run, "base", s), "base", self.sets[s], res[(s, "base")], None)
            RD.log(f"base reads of the round-trip rows ({differ} of {tokens} loss tokens differ from the records) and of {sets}: {time.time() - t0:.0f} s")
            return RD.EXIT_ROUNDTRIP if differ or not tokens else 0
        rep = {"round_trip_rows": {"rows": len(self.rt.rows), "differ": sum(RD.bits_differ(got[r], self.rt_base[r], self.rt.mask(r)) for r in self.rt.rows)}}
        for s in sets:
            rs = self.sets[s]
            self.write_set(os.path.join(self.run, "base", s), "base2", rs, res[(s, "base2")], None)
            first = np.load(os.path.join(self.run, "base", s, "base.npz"))
            rep[s] = {"rows": len(rs.rows), "loss_tokens": int(sum(int(rs.mask(n).sum()) for n in rs.rows)),
                      "differ": sum(RD.bits_differ(res[(s, "base2")][n], first[f"lp_{n}"], rs.mask(n)) for n in rs.rows)}
        rep["result"] = "equal" if all(v["differ"] == 0 for v in rep.values() if isinstance(v, dict)) else "differ"
        jp = os.path.join(self.run, "base", "reread.json")
        RD.write_json(jp, rep)
        self.files[self.rel(jp)] = RD.sha256(jp)
        RD.log(f"base re-read at the end: {rep}; {time.time() - t0:.0f} s")
        return 0 if rep["result"] == "equal" else RD.EXIT_ROUNDTRIP

    def state(self, name: str) -> int:
        sd = os.path.join(self.run, "states", name)
        tr = json.load(open(os.path.join(sd, "TRAINER_READY.json")))
        sid = self.session_id()
        if tr.get("state") != name or tr.get("session_id") != sid or sorted(tr["round_trip_rows"]) != self.rt.rows:
            raise ValueError(f"state {name}: the hand-over names another state, session or round-trip rows")
        ad = tr["adapter"]
        if RD.sha256(os.path.join(ad["path"], "adapter_model.safetensors")) != ad["sha256"]:
            raise ValueError(f"state {name}: the adapter on disk is not the file handed over")
        files = {}
        for rel, h in tr["files"].items():
            if RD.sha256(os.path.join(self.run, rel)) != h:
                raise ValueError(f"state {name}: {rel} changed after the hand-over")
            files[rel] = h
        sets = [self.sets[s] for s in tr["sets"]]
        t0 = time.time()
        res = RD.run_pass(self.c, self.urls, [(name, ad["lora_name"], ad["path"])], [self.rt] + sets + [self.rt.subset("rt2", self.rt.rows, reverse=True)])
        rd = os.path.join(sd, "roundtrip")
        tz = np.load(os.path.join(rd, "trainer.npz"))
        if sorted(int(k[3:]) for k in tz.files) != self.rt.rows:
            raise ValueError(f"state {name}: trainer.npz does not hold exactly the round-trip rows")
        tokens = int(sum(int(self.rt.mask(r).sum()) for r in self.rt.rows))
        first = sum(RD.bits_differ(res[("rt", name)][r], tz[f"lp_{r}"], self.rt.mask(r)) for r in self.rt.rows)
        last = sum(RD.bits_differ(res[("rt2", name)][r], tz[f"lp_{r}"], self.rt.mask(r)) for r in self.rt.rows)
        moved = sum(RD.bits_differ(res[("rt", name)][r], self.rt_base[r], self.rt.mask(r)) for r in self.rt.rows)
        RD.save_npz(os.path.join(rd, "engine.npz"), **{f"lp_{r}": res[("rt", name)][r] for r in self.rt.rows})
        RD.save_npz(os.path.join(rd, "engine_last.npz"), **{f"lp_{r}": res[("rt2", name)][r] for r in self.rt.rows})
        result = "equal" if first == 0 and last == 0 and tokens > 0 else "differ"
        RD.write_json(os.path.join(rd, "result.json"),
                      {"result": result, "state": name, "session_id": sid, "adapter_sha256": ad["sha256"],
                       "checks": {"trainer_vs_engine": {"rows": len(self.rt.rows), "loss_tokens": tokens, "differ": first},
                                  "trainer_vs_engine_last": {"rows": len(self.rt.rows), "loss_tokens": tokens, "differ": last}},
                       "state_vs_base_positions_differing": moved,
                       "round_trip_rows_per_engine": {str(e): self.rt.share(e, len(self.urls)) for e in range(len(self.urls))}})
        RD.log(f"state {name}: round trip {result} (first {first}, last {last} of {tokens} loss tokens differ; differs from base at {moved})")
        if result != "equal":
            return RD.EXIT_ROUNDTRIP
        for f in sorted(os.listdir(rd)):
            files[self.rel(os.path.join(rd, f))] = RD.sha256(os.path.join(rd, f))
        keep, self.files = self.files, files                  # this state's files only
        for rs in sets:
            self.write_set(os.path.join(sd, "reads", rs.tag), "read", rs, res[(rs.tag, name)], ad["sha256"])
        self.files = keep
        ap = os.path.join(sd, "adapter.json")
        RD.write_json(ap, {"sha256": ad["sha256"], "lora_name": ad["lora_name"], "sets": tr["sets"]})
        files[self.rel(ap)] = RD.sha256(ap)
        RD.write_json(os.path.join(sd, "reads_stats.json"), {"wall_s": round(time.time() - t0, 1), "per_engine": res["stats"]})
        RD.write_json(os.path.join(sd, "READS_DONE"), {"state": name, "session_id": sid, "adapter_sha256": ad["sha256"], "files": files})
        self.files.update(files)
        self.files[self.rel(os.path.join(sd, "READS_DONE"))] = RD.sha256(os.path.join(sd, "READS_DONE"))
        RD.log(f"state {name}: READS_DONE on {tr['sets']}, {len(files)} files, {time.time() - t0:.0f} s")
        return 0

    def serve_states(self, names: list, deadline_s: float = 0.0) -> int:
        t_start = time.time()
        rc = self.base_pass("base", sorted(self.sets))
        if rc:
            return rc
        for name in names:
            if deadline_s and time.time() - t_start > deadline_s:      # a read side that is this late is stopped before its next state
                left = names[names.index(name):]
                RD.log(f"the reads passed their limit of {deadline_s / 60:.0f} min before state {name}; not read: {left}")
                RD.write_json(os.path.join(self.run, "READS_STOPPED.json"), {"reason": "the reads' time limit", "limit_min": deadline_s / 60,
                                                                           "read": names[:names.index(name)], "not_read": left})
                return RD.EXIT_READS
            ready = os.path.join(self.run, "states", name, "TRAINER_READY.json")
            while not os.path.exists(ready):
                end = os.path.join(self.run, "trainer_end.json")
                if os.path.exists(end) or os.path.exists(os.path.join(self.run, "trainer.exit")):
                    RD.log(f"the trainer side ended before state {name} was handed over")
                    return RD.EXIT_READS
                time.sleep(self.poll)
            rc = self.state(name)
            if rc:
                return rc
        rc = self.base_pass("base2", self.reread)
        if rc:
            return rc
        for f in ("rules.json", "trainer_start.json"):
            p = os.path.join(self.run, f)
            if os.path.exists(p):
                self.files[f] = RD.sha256(p)
        RD.write_json(os.path.join(self.run, "RULES_DONE"), {"session_id": self.session_id(), "states": names, "files": self.files})
        RD.log(f"RULES_DONE: {len(names)} states, {len(self.files)} files")
        return 0


def check_sets(rules: dict, rd: "StateReader") -> None:
    """Every set a state names must be one the read side holds, every set it holds must be named by a state, and the round-trip rows must be
    rules.json's: a mismatch found at the last state would cost the whole run."""
    need = {s for r in rules["states"] for s in r["sets"]}
    if need != set(rd.sets):
        raise ValueError(f"rules.json names the sets {sorted(need)}; the read side was given {sorted(rd.sets)}")
    if rules.get("round_trip_rows") is not None and (sorted(rules["round_trip_rows"]) != rd.rt.rows or rules.get("round_trip_group") != rd.rt_group):
        raise ValueError("rules.json names other round-trip rows than this plan and group give")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True)
    ap.add_argument("--model", default="inkling")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--set", action="append", default=[], help="name=directory (repeat)")
    ap.add_argument("--own-groups", default="", help="comma list of plan groups: the set 'own'")
    ap.add_argument("--rt-group", type=int, default=0)
    ap.add_argument("--reread", default="", help="comma list of sets read again without an adapter at the end")
    ap.add_argument("--rules", required=True, help="rules.json (the states and their order)")
    ap.add_argument("--deadline-min", type=float, default=0.0, help="no further state is started after this many minutes (0 = no limit)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--engine-sha-file", default="")
    a = ap.parse_args()
    rc = RD.EXIT_READS
    try:
        names = [r["name"] for r in json.load(open(a.rules))["states"]]
        eng = open(a.engine_sha_file).read().strip() if a.engine_sha_file else ""
        rules = json.load(open(a.rules))
        rd = StateReader(RD.HttpClient(a.model), [u for u in a.urls.split(",") if u], a.run_dir, a.plan, dict(x.split("=", 1) for x in a.set),
                         [g for g in a.own_groups.split(",") if g], a.rt_group, a.reread.split(","), eng)
        check_sets(rules, rd)                                 # before a single token is read
        rc = rd.serve_states(names, a.deadline_min * 60.0)
    except (ValueError, KeyError, OSError) as e:
        RD.log(f"READ SIDE: inputs: {type(e).__name__}: {e}")
        rc = RD.EXIT_INPUT
    except Exception as e:
        RD.log(f"READ SIDE FAILED: {type(e).__name__}: {e}")
        rc = RD.EXIT_READS
    with open(os.path.join(a.run_dir, "reader.exit.tmp"), "w") as f:
        f.write(str(rc))
    os.replace(os.path.join(a.run_dir, "reader.exit.tmp"), os.path.join(a.run_dir, "reader.exit"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
