"""Step session, read side: its own process, on serving engines that run beside the trainer. It never touches the trainer; the two sides
meet only through files of the run directory (tests/t8_stepsession.py holds the layout and the handshake).

serve (while the trainer runs), on the read engines:
  start:  base reads (no adapter) of every panel -> base/reads_dmg/<panel>/base.npz (+ .json); of every update's round-trip rows ->
          base/roundtrip/base_g<group>.npz (when the plan's rows were sampled by the base model, i.e. its first update is number 1, this read
          must equal the rows' recorded log-probs bitwise, else exit 21); then of the transfer plan -> base/transfer/base.npz.
  per update n, when updates/<n>/TRAINER_READY.json appears: on EVERY engine, in this order,
      for X in v, R, R2:  load X; its share of the round-trip rows; its share of every panel; the round-trip rows AGAIN; unload X
      base again:         its share of the round-trip rows; its share of every panel (-> base2); the round-trip rows again
    ROUND TRIP (bitwise, at every loss token of the round-trip rows): engine(X) == the trainer's log-probs at X for v, R and R2, and the base
    re-read == the base read of the same rows at the start; BOTH reads of the rows under each adapter. The last requests under one adapter and
    the first under the next are then the same prompts on the same engine: a state carried over from one adapter into the next request
    (slot or prefix cache) would show here, whatever the panel reads in between did to the cache. -> roundtrip/engine_<X>.npz (first read), engine_<X>_last.npz (last read), engine_base2.npz, result.json ("result": "equal" |
    "differ"). On "differ" nothing more is written for the update and the process ends with exit 21.
    Panels -> reads_dmg/<panel>/{v,R,R2,base2}.npz: lp_<row> float32 [T] = the value at the panel's mask, 0 elsewhere, + ids_crc32 [R] of the ids
    sent; <X>.json {"adapter_sha256", "head", "engine_sha", "source", "session_id", "panel_rows_sha256"}. A read that is not complete is not written.
    READS_DONE is written last, atomically: {"update", "session_id", "adapters": {v, R, R2: sha256}, "files": {relative path: sha256}} over
    every read file of the update, its adapters.json and round-trip files, and the base reads.
  The row-to-engine split is a fixed function of the row lengths and the number of engines: the same for base, v, R, R2 and base2 at every update.
  It ends when the trainer side has finished (TRAINER_DONE.json), stopped (STOPPED.json) or exited (trainer.exit) and nothing is left to read.
transfer (after the last GO, on all engines): the transfer plan under the last update's v, R and R2 -> transfer/reads/<X>.npz (+ .json).
Exit codes: 0; 21 round trip not bitwise; 6 an engine or a read failed; 5 inputs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t8_micro_reads as MR  # noqa: E402

NAMES = ("v", "R", "R2")
EXIT_INPUT, EXIT_READS, EXIT_ROUNDTRIP = 5, 6, 21


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def save_npz(path, **arrs):
    with open(path + ".tmp", "wb") as f:
        np.savez(f, **arrs)
    os.replace(path + ".tmp", path)


class HttpClient:
    """The serving engine's API: one adapter slot, adapters loaded and unloaded at run time; prompt log-probs of given token ids."""

    def __init__(self, base_model="inkling"):
        self.base_model = base_model

    def load(self, url, name, path):
        r = MR.post(url, "/v1/load_lora_adapter", {"lora_name": name, "lora_path": path})
        if not (isinstance(r, str) and "added successfully" in r):
            raise RuntimeError(f"load of {name} answered {str(r)[:200]!r}")

    def unload(self, url, name):
        r = MR.post(url, "/v1/unload_lora_adapter", {"lora_name": name})
        if not (isinstance(r, str) and "removed successfully" in r):
            raise RuntimeError(f"unload of {name} answered {str(r)[:200]!r}")

    def score(self, url, name, ids):
        """float32 [T]: position t = log p(ids[t] | ids[:t]) (position 0 = 0). name None = the base model."""
        return MR.score(url, self.base_model if name is None else name, ids)


class RowSet:
    """Rows to read: a plan or panel directory in the micro-step driver's on-disk form, checked against its manifest. Every array a read needs
    (ids, masks, recorded log-probs) is loaded into memory HERE, by the thread that builds the set: the engine threads never touch the .npz
    file (a zip archive read from several threads at once is not safe)."""

    def __init__(self, tag: str, d: str, only=None):
        self.tag, self.dir = tag, d
        man = json.load(open(os.path.join(d, "manifest.json")))
        want = {os.path.basename(k): v for k, v in (man.get("files") or man).items() if isinstance(v, str) and len(v) == 64}
        path = os.path.join(d, "rows.npz")
        self.rows_sha = sha256(path)
        if want.get("rows.npz") != self.rows_sha:
            raise ValueError(f"{tag}: rows.npz does not match its manifest sha256")
        with np.load(path, allow_pickle=False) as z:
            self.R = sum(1 for k in z.files if k.startswith("ids_") and k[4:].isdigit())
            self._ids = {n: np.array(z[f"ids_{n}"]) for n in range(self.R)}
            self._mask = {n: np.array(z[f"mask_{n}"]).astype(bool) for n in range(self.R)}
            self._lp = {n: np.array(z[f"lp_{n}"]).astype(np.float32) for n in range(self.R) if f"lp_{n}" in z.files}
            self.T = {n: int(z["T"][n]) for n in range(self.R)}
            self.group = np.array(z["group_idx"]) if "group_idx" in z.files else None
            recorded_crc = np.array(z["ids_crc32"]).astype(np.uint32) if "ids_crc32" in z.files else None
        if any(len(self._ids[n]) != self.T[n] or len(self._mask[n]) != self.T[n] for n in range(self.R)):
            raise ValueError(f"{tag}: a row's ids or mask do not have its length T")
        self.crc = np.array([zlib.crc32(self._ids[n].astype("<i4").tobytes()) & 0xFFFFFFFF for n in range(self.R)], dtype=np.uint32)
        if recorded_crc is not None and not np.array_equal(self.crc, recorded_crc):
            raise ValueError(f"{tag}: crc32 of the ids differs from the recorded ids_crc32")
        self.rows = sorted(self.T) if only is None else sorted(int(n) for n in only)
        if any(n not in self.T for n in self.rows):
            raise ValueError(f"{tag}: rows {self.rows} are not all in the file")

    def subset(self, tag: str, only, reverse: bool = False) -> "RowSet":
        s = RowSet.__new__(RowSet)
        s.__dict__.update(self.__dict__)
        s.tag, s.rows, s.reverse = tag, sorted(int(n) for n in only), reverse
        if any(n not in self.T for n in s.rows):
            raise ValueError(f"{tag}: rows {s.rows} are not all in the file")
        return s

    def share(self, e: int, engines: int) -> list:
        """Engine e's rows: longest first to the engine with the fewest tokens (t8_micro_reads.split), on this set's rows only.
        A reversed set gives the same rows in the opposite order (shortest first)."""
        mine = MR.split({n: self.T[n] for n in self.rows}, engines)[e]
        return mine[::-1] if getattr(self, "reverse", False) else mine

    def ids(self, n):
        return self._ids[n]

    def mask(self, n):
        return self._mask[n]

    def recorded(self, n):
        return self._lp[n]


def run_pass(client, urls: list, adapters: list, rowsets: list) -> dict:
    """One thread per engine; each runs, adapter after adapter (name, lora name, path; path None = the base model), its share of every row set
    in the order given. -> {(row set tag, adapter name): {row: float32 [T], the value at the set's mask and 0 elsewhere}}.
    Any failure on any engine fails the pass (RuntimeError); an adapter is unloaded on every way out."""
    res = {(rs.tag, name): {} for rs in rowsets for name, _, _ in adapters}
    errors, lock, stats = [], threading.Lock(), []

    def one(e, url):
        try:
            for name, lora, path in adapters:
                t0, tok = time.time(), 0
                if path is not None:
                    client.load(url, lora, path)
                try:
                    for rs in rowsets:
                        for n in rs.share(e, len(urls)):
                            with lock:
                                if errors:
                                    return
                            lp = np.asarray(client.score(url, None if path is None else lora, rs.ids(n)), dtype=np.float32)
                            mask = rs.mask(n)
                            if lp.shape != mask.shape:
                                raise RuntimeError(f"{rs.tag} row {n}: {lp.shape} log-probs for T {mask.shape}")
                            out = np.zeros(len(mask), dtype=np.float32)
                            out[mask] = lp[mask]
                            if not np.isfinite(out).all():
                                raise RuntimeError(f"{rs.tag} row {n} under {name}: a non-finite log-prob")
                            with lock:
                                res[(rs.tag, name)][n] = out
                            tok += rs.T[n]
                finally:
                    if path is not None:
                        client.unload(url, lora)
                with lock:
                    stats.append({"engine": e, "adapter": name, "tokens": tok, "seconds": round(time.time() - t0, 1)})
        except BaseException as ex:                           # SystemExit from the reader's own checks included
            with lock:
                errors.append(f"engine {e} ({url}): {type(ex).__name__}: {ex}")

    th = [threading.Thread(target=one, args=(e, u), daemon=True) for e, u in enumerate(urls)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if errors:
        raise RuntimeError("; ".join(errors)[:2000])
    for rs in rowsets:
        for name, _, _ in adapters:
            if sorted(res[(rs.tag, name)]) != rs.rows:
                raise RuntimeError(f"{rs.tag} under {name}: {len(res[(rs.tag, name)])} of {len(rs.rows)} rows read")
    res["stats"] = stats
    return res


def bits_differ(a, b, mask) -> int:
    a32, b32 = np.asarray(a, dtype=np.float32)[mask], np.asarray(b, dtype=np.float32)[mask]
    return int((a32.view(np.int32) != b32.view(np.int32)).sum())


class Reader:
    def __init__(self, client, urls, run_dir, plan_dir, panels: dict, transfer_dir="", engine_sha="", poll_s=2.0):
        self.c, self.urls, self.run, self.poll = client, list(urls), run_dir, poll_s
        self.plan = RowSet("plan", plan_dir)
        bad = [name for name in panels if name in ("rt", "rt2", "plan", "transfer", "base") or not name or "/" in name]
        if bad:                                               # these names are row sets of the read side itself
            raise ValueError(f"panel names {bad} are not allowed")
        self.panels = {name: RowSet(name, d) for name, d in panels.items()}
        self.transfer = RowSet("transfer", transfer_dir) if transfer_dir else None
        self.engine_sha = engine_sha
        self.base_files = {}
        self.sid = None
        meta = json.load(open(os.path.join(plan_dir, "rows.json")))
        ups = meta["updates"]
        self.rt_groups = [int(g) for g in (meta.get("round_trip_group") or [u[0] for u in ups])]
        self.sampled_by_base = int(meta.get("first_update_number", 1)) == 1   # else: by the adapter the session resumes from
        gi = self.plan.group
        self.rt_sets = {g: self.plan.subset("rt", [n for n in range(self.plan.R) if int(gi[n]) == g]) for g in sorted(set(self.rt_groups))}
        self.rt_base = {}

    def rel(self, p):
        return os.path.relpath(p, self.run)

    def session_id(self):
        if self.sid is None:
            f = os.path.join(self.run, "session_id.txt")
            if os.path.exists(f):
                self.sid = open(f).read().strip()
        return self.sid

    def write_read(self, d, name, rs: RowSet, got: dict, adapter_sha, files: dict | None = None):
        """<d>/<name>.npz + .json for a COMPLETE read of the row set."""
        os.makedirs(d, exist_ok=True)
        for n in rs.rows:
            x = got[n]
            if x.dtype != np.float32 or x.shape != (rs.T[n],) or bool(x[~rs.mask(n)].any()):
                raise RuntimeError(f"{rs.tag}/{name}: row {n} is not float32 [T] with zeros off the mask")
        p = os.path.join(d, name + ".npz")
        save_npz(p, ids_crc32=rs.crc[rs.rows] if len(rs.rows) != rs.R else rs.crc, **{f"lp_{n}": got[n] for n in rs.rows})
        write_json(os.path.join(d, name + ".json"), {"adapter_sha256": adapter_sha, "head": "plain", "engine_sha": self.engine_sha, "source": "engine",
                                                    "session_id": self.session_id(), "panel_rows_sha256": rs.rows_sha})
        if files is not None:
            for q in (p, os.path.join(d, name + ".json")):
                files[self.rel(q)] = sha256(q)

    # ------------------------------------------------------------------ start
    def base_reads(self):
        sets = list(self.panels.values())
        if sets:
            t0 = time.time()
            res = run_pass(self.c, self.urls, [("base", None, None)], sets)
            for rs in sets:
                self.write_read(os.path.join(self.run, "base", "reads_dmg", rs.tag), "base", rs, res[(rs.tag, "base")], None, self.base_files)
            log(f"base reads of the panels {[rs.tag for rs in sets]}: {time.time() - t0:.0f} s")
        ident = {}
        for g, rt in self.rt_sets.items():                    # each update's round-trip rows, with the split their later reads use
            if len(rt.rows) < len(self.urls):
                raise ValueError(f"round-trip group {g}: {len(rt.rows)} rows for {len(self.urls)} engines (every engine must read at least one)")
            got = run_pass(self.c, self.urls, [("base", None, None)], [rt])[("rt", "base")]
            self.rt_base[g] = got
            p = os.path.join(self.run, "base", "roundtrip", f"base_g{g}.npz")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            save_npz(p, **{f"lp_{r}": got[r] for r in rt.rows})
            self.base_files[self.rel(p)] = sha256(p)
            ident[str(g)] = {"rows": len(rt.rows), "loss_tokens": int(sum(int(rt.mask(r).sum()) for r in rt.rows)),
                             "differ_from_recorded": sum(bits_differ(got[r], rt.recorded(r), rt.mask(r)) for r in rt.rows) if self.sampled_by_base else None}
        if ident:
            bad = self.sampled_by_base and any(v["differ_from_recorded"] or not v["loss_tokens"] for v in ident.values())
            write_json(os.path.join(self.run, "base", "roundtrip", "identity.json"),
                       {"result": "differ" if bad else "equal", "sampled_by_base": self.sampled_by_base, "groups": ident,
                        "check": "the engines' base read of the round-trip rows == the rows' recorded log-probs, bitwise"
                                 if self.sampled_by_base else "not applicable: the rows were sampled under an adapter"})
            log(f"base read of the round-trip rows: {ident}")
            if bad:
                return EXIT_ROUNDTRIP
        if self.transfer is not None:
            t0 = time.time()
            res = run_pass(self.c, self.urls, [("base", None, None)], [self.transfer])
            got = res[("transfer", "base")]
            self.write_read(os.path.join(self.run, "base", "transfer"), "base", self.transfer, got, None)
            differ = sum(bits_differ(got[n], self.transfer.recorded(n), self.transfer.mask(n)) for n in self.transfer.rows)
            tokens = sum(int(self.transfer.mask(n).sum()) for n in self.transfer.rows)
            write_json(os.path.join(self.run, "base", "transfer", "identity.json"),
                       {"base_read_vs_recorded": {"loss_tokens": tokens, "differ": differ}, "note": "recorded, not gated here"})
            log(f"base read of the transfer plan: {len(self.transfer.rows)} rows, {differ} of {tokens} loss tokens differ from the records, {time.time() - t0:.0f} s")
        return 0

    # ------------------------------------------------------------------ one update
    def update(self, n: int) -> int:
        ud = os.path.join(self.run, "updates", str(n))
        tr = json.load(open(os.path.join(ud, "TRAINER_READY.json")))
        sid = self.session_id()
        if tr.get("update") != n or tr.get("session_id") != sid:
            raise ValueError(f"update {n}: TRAINER_READY names update {tr.get('update')!r}, session {tr.get('session_id')!r}")
        files = dict(self.base_files)
        for rel, h in tr["files"].items():                    # the trainer's round-trip log-probs and adapters.json, as handed over
            if sha256(os.path.join(self.run, rel)) != h:
                raise ValueError(f"update {n}: {rel} changed after the hand-over")
            files[rel] = h
        ads = tr["adapters"]
        for name in NAMES:
            st = os.path.join(ads[name]["path"], "adapter_model.safetensors")
            if sha256(st) != ads[name]["sha256"]:
                raise ValueError(f"update {n}: adapter {name} on disk is not the exported file")
        g = int(tr["round_trip_group"])
        if g not in self.rt_sets or self.rt_sets[g].rows != sorted(int(x) for x in tr["round_trip_rows"]):
            raise ValueError(f"update {n}: the round-trip rows handed over are not group {g}'s rows of the plan")
        rt, base0 = self.rt_sets[g], self.rt_base[g]
        adapters = [(name, ads[name]["lora_name"], ads[name]["path"]) for name in NAMES] + [("base2", None, None)]
        t0 = time.time()
        # the last read runs the engine's round-trip rows in the opposite order: the last request under one adapter and the first under
        # the next are then the SAME row (the engine's longest)
        res = run_pass(self.c, self.urls, adapters, [rt] + list(self.panels.values()) + [rt.subset("rt2", rt.rows, reverse=True)])
        # the round trip
        rd = os.path.join(ud, "roundtrip")
        checks, differ_total = {}, 0
        for name in NAMES:
            tz = np.load(os.path.join(rd, f"trainer_{name}.npz"))
            if sorted(int(k[3:]) for k in tz.files) != rt.rows:
                raise ValueError(f"update {n}: trainer_{name}.npz does not hold exactly the round-trip rows")
            d = sum(bits_differ(res[("rt", name)][r], tz[f"lp_{r}"], rt.mask(r)) for r in rt.rows)
            checks[f"trainer_vs_engine_{name}"] = {"rows": len(rt.rows), "loss_tokens": int(sum(int(rt.mask(r).sum()) for r in rt.rows)), "differ": d}
            differ_total += d
            save_npz(os.path.join(rd, f"engine_{name}.npz"), **{f"lp_{r}": res[("rt", name)][r] for r in rt.rows})
        d = sum(bits_differ(res[("rt", "base2")][r], base0[r], rt.mask(r)) for r in rt.rows)
        checks["engine_base_reread_vs_start_base"] = {"rows": len(rt.rows), "loss_tokens": checks["trainer_vs_engine_v"]["loss_tokens"], "differ": d}
        differ_total += d
        save_npz(os.path.join(rd, "engine_base2.npz"), **{f"lp_{r}": res[("rt", "base2")][r] for r in rt.rows})
        second = {}                                           # the rows' LAST read under each adapter (after the panels): a file of its own
        for name in NAMES:
            tz = np.load(os.path.join(rd, f"trainer_{name}.npz"))
            d = sum(bits_differ(res[("rt2", name)][r], tz[f"lp_{r}"], rt.mask(r)) for r in rt.rows)
            second[name] = {"rows": len(rt.rows), "loss_tokens": checks[f"trainer_vs_engine_{name}"]["loss_tokens"], "differ": d}
            differ_total += d
            save_npz(os.path.join(rd, f"engine_{name}_last.npz"), **{f"lp_{r}": res[("rt2", name)][r] for r in rt.rows})
        d = sum(bits_differ(res[("rt2", "base2")][r], base0[r], rt.mask(r)) for r in rt.rows)
        base_second = {"rows": len(rt.rows), "loss_tokens": checks["trainer_vs_engine_v"]["loss_tokens"], "differ": d}
        differ_total += d
        moved = sum(bits_differ(res[("rt", "v")][r], res[("rt", "base2")][r], rt.mask(r)) for r in rt.rows)
        per_engine = {str(e): rt.share(e, len(self.urls)) for e in range(len(self.urls))}
        result = "equal" if differ_total == 0 and checks["trainer_vs_engine_v"]["loss_tokens"] > 0 else "differ"
        write_json(os.path.join(rd, "result.json"), {"result": result, "update": n, "session_id": sid, "checks": checks,
                                                    "v_vs_base_positions_differing": moved,   # reported: an adapter that moves nothing
                                                    "second_read": second, "base_second_read": base_second,
                                                    "round_trip_rows_per_engine": per_engine,
                                                    "order": "on each engine: v, R, R2, no adapter; under each: round-trip rows, panels, round-trip rows",
                                                    "adapters": {k: ads[k]["sha256"] for k in NAMES}})
        log(f"update {n}: round trip {result}: {checks}; second read {second}; v differs from base at {moved} positions")
        if result != "equal":
            return EXIT_ROUNDTRIP
        for f in sorted(os.listdir(rd)):
            files[self.rel(os.path.join(rd, f))] = sha256(os.path.join(rd, f))
        for rs in self.panels.values():
            d_ = os.path.join(ud, "reads_dmg", rs.tag)
            for name in NAMES:
                self.write_read(d_, name, rs, res[(rs.tag, name)], ads[name]["sha256"], files)
            self.write_read(d_, "base2", rs, res[(rs.tag, "base2")], None, files)
        write_json(os.path.join(ud, "reads_stats.json"), {"wall_s": round(time.time() - t0, 1), "per_engine": res["stats"]})
        write_json(os.path.join(ud, "READS_DONE"), {"update": n, "session_id": sid, "adapters": {k: ads[k]["sha256"] for k in NAMES}, "files": files})
        log(f"update {n}: READS_DONE, {len(files)} files, {time.time() - t0:.0f} s")
        return 0

    def serve(self) -> int:
        rc = self.base_reads()
        if rc:
            return rc
        done = set()
        while True:
            root = os.path.join(self.run, "updates")
            ready = sorted(int(x) for x in (os.listdir(root) if os.path.isdir(root) else [])
                           if x.isdigit() and int(x) not in done and os.path.exists(os.path.join(root, x, "TRAINER_READY.json")))
            if ready:
                rc = self.update(ready[0])
                if rc:
                    return rc
                done.add(ready[0])
                continue
            if any(os.path.exists(os.path.join(self.run, f)) for f in ("TRAINER_DONE.json", "STOPPED.json", "trainer.exit")):
                return 0
            time.sleep(self.poll)

    # ------------------------------------------------------------------ transfer
    def transfer_reads(self, n: int) -> int:
        if self.transfer is None:
            raise ValueError("no transfer plan")
        ads = json.load(open(os.path.join(self.run, "updates", str(n), "TRAINER_READY.json")))["adapters"]
        for name in NAMES:
            if sha256(os.path.join(ads[name]["path"], "adapter_model.safetensors")) != ads[name]["sha256"]:
                raise ValueError(f"transfer: adapter {name} on disk is not the exported file")
        t0 = time.time()
        res = run_pass(self.c, self.urls, [(name, ads[name]["lora_name"], ads[name]["path"]) for name in NAMES], [self.transfer])
        d = os.path.join(self.run, "transfer", "reads")
        for name in NAMES:
            self.write_read(d, name, self.transfer, res[("transfer", name)], ads[name]["sha256"])
        write_json(os.path.join(d, "stats.json"), {"update": n, "wall_s": round(time.time() - t0, 1), "per_engine": res["stats"]})
        log(f"transfer reads under update {n}'s v, R, R2: {len(self.transfer.rows)} rows, {time.time() - t0:.0f} s on {len(self.urls)} engines")
        return 0


def probe(client, urls, panels: dict, adapters: dict, run_dir: str) -> int:
    """Adapter-swap probe on panel rows. Engine 0: load X; read each panel's shortest and longest row; unload X; load Y (the swap a session
    makes); read the longest and the shortest row, twice. Engine 1, which never saw X: load Y; read the same rows. PASS line (exit 0): every
    read under Y on engine 0, at EVERY position of the row, equals engine 1's bit for bit, and X differs from Y somewhere on every row (else
    the comparison would be empty). Exit 21 otherwise. -> <run dir>/probe.json"""
    if len(urls) != 2 or set(adapters) != {"X", "Y"}:
        raise ValueError("the probe takes two engines and the adapters X and Y")
    e0, e1 = urls
    rows = []
    for name, d in panels.items():
        rs = RowSet(name, d)
        by = sorted(rs.rows, key=lambda n: (rs.T[n], n))
        rows += [(name, by[0], rs), (name, by[-1], rs)]

    def read(url, lora):
        return {(nm, n): np.asarray(client.score(url, lora, rs.ids(n)), dtype=np.float32) for nm, n, rs in order}

    order = rows                                              # shortest, longest per panel
    client.load(e0, "probe_X", adapters["X"])
    x0 = read(e0, "probe_X")
    client.unload(e0, "probe_X")
    client.load(e0, "probe_Y", adapters["Y"])
    order = rows[::-1]                                        # the row asked last under X is asked first under Y
    y0a = read(e0, "probe_Y")
    y0b = read(e0, "probe_Y")
    client.unload(e0, "probe_Y")
    client.load(e1, "probe_Y", adapters["Y"])                 # the fresh engine
    y1 = read(e1, "probe_Y")
    client.unload(e1, "probe_Y")
    rep, ok = {"rows": [], "engines": urls}, True
    for nm, n, rs in rows:
        k = (nm, n)
        bits = lambda a, b: int((a.view(np.int32) != b.view(np.int32)).sum())  # noqa: E731
        r = {"panel": nm, "row": int(n), "T": rs.T[n], "swapped_first_vs_fresh": bits(y0a[k], y1[k]), "swapped_second_vs_fresh": bits(y0b[k], y1[k]),
             "X_vs_Y_positions_differing": bits(x0[k], y1[k])}
        ok = ok and r["swapped_first_vs_fresh"] == 0 and r["swapped_second_vs_fresh"] == 0 and r["X_vs_Y_positions_differing"] > 0
        rep["rows"].append(r)
        log(f"probe {nm} row {n} (T {rs.T[n]}): after the swap vs a fresh engine: {r['swapped_first_vs_fresh']} / {r['swapped_second_vs_fresh']} "
            f"positions differ; X vs Y: {r['X_vs_Y_positions_differing']}")
    rep["result"] = "equal" if ok else "differ"
    write_json(os.path.join(run_dir, "probe.json"), rep)
    log(f"ADAPTER-SWAP PROBE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else EXIT_ROUNDTRIP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("serve", "transfer", "probe"))
    ap.add_argument("--probe-adapters", default="", help="probe: X=<adapter dir>,Y=<adapter dir>")
    ap.add_argument("--urls", required=True, help="comma list of engine URLs")
    ap.add_argument("--model", default="inkling")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--panel", action="append", default=[], help="name=directory (repeat)")
    ap.add_argument("--transfer", default="")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--engine-sha-file", default="")
    ap.add_argument("--update", type=int, default=0, help="transfer: the update whose adapters are read")
    a = ap.parse_args()
    if a.phase == "probe":
        try:
            return probe(HttpClient(a.model), [u for u in a.urls.split(",") if u], dict(x.split("=", 1) for x in a.panel),
                         dict(x.split("=", 1) for x in a.probe_adapters.split(",")), a.run_dir)
        except Exception as e:
            log(f"PROBE FAILED TO RUN: {type(e).__name__}: {e}")
            return EXIT_READS
    try:
        panels = dict(x.split("=", 1) for x in a.panel)
        eng = open(a.engine_sha_file).read().strip() if a.engine_sha_file else ""
        rd = Reader(HttpClient(a.model), [u for u in a.urls.split(",") if u], a.run_dir, a.plan, panels, a.transfer, eng)
    except (OSError, ValueError, KeyError) as e:
        log(f"READ SIDE REFUSED: {e}")
        return EXIT_INPUT
    try:
        rc = rd.serve() if a.phase == "serve" else rd.transfer_reads(a.update)
    except ValueError as e:
        log(f"READ SIDE: inputs: {e}")
        rc = EXIT_INPUT
    except Exception as e:
        log(f"READ SIDE FAILED: {type(e).__name__}: {e}")
        rc = EXIT_READS
    if a.phase == "serve":                                    # the trainer side looks at this file
        with open(os.path.join(a.run_dir, "reader.exit.tmp"), "w") as f:
            f.write(str(rc))
        os.replace(os.path.join(a.run_dir, "reader.exit.tmp"), os.path.join(a.run_dir, "reader.exit"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
