"""Micro-step session: one real PPO mini-batch at the zero adapter, the update it makes, the adapters a follow read serves, and the gates that
make the read's inputs trustworthy. Nothing is kept as a model; the step's result is measured, not used.
PASS LINES, written before any run:
  M1 zero adapter: the trainer at the start adapter Z (A = init, B = 0) == the base forward on every worker, bitwise (check_zero_adapter);
     the engine's reads with Z loaded == its base reads, bitwise at every loss token of every row.
  M2 replay: the trainer's old log-probs (start weights) == the rollout's recorded log-probs bitwise at every loss token of every row (the step's
     replay gate; one mismatch stops the step), and == the engine's base reads bitwise at every loss token.
  M3 ratio 1: the mini-batch's forward == the old log-probs bitwise on every row (stale gate on every row), ppo_ratio_max == 1.0, clip share 0.
  M4 step: every gradient finite, no row refused, no gate of the step fails; N (the divisor) == the plan's loss tokens; the update of every
     A tensor exactly 0 (B starts at 0, so A has no gradient at the first step; the follow read relies on it).
  M5 round trip: the trainer's log-probs at U12 == the engine's U12 reads bitwise at every loss token of the round-trip rows (the first
     group's rows; roundtrip/trainer_U12.npz).
  M6 files: gates.json, first_order.json, masters/theta0 + theta1, rows_used.npz, adapters.json and the round-1 reads (base, Z, U12, R12, U4) of
     every row written; a sketch file for every attempt with a non-zero advantage (zero-advantage attempts counted). Round 2 (the block adapters)
     runs if the card-hour cap leaves room, round 3 (block random controls) likewise; both reported, no line.
  Pre-check (one card, before the session): G18a / G18b of t8_gate_s5 on three of the plan's rows (trainer Z == engine base, trainer with a
     test adapter == engine with its exported file, bitwise at every position) and the replay of those rows (trainer Z == recorded, every loss token).
Phases: check (CPU: the plan file), pre (trainer half of the pre-check), pre_compare, trainer (the session's trainer half), compare (after reads).
The update's adapters (theta0 = Z): U_k = theta0 + k d; R12 = theta0 + r, r a fixed-seed Gaussian scaled tensor by tensor to the norm of 12 d;
U12_<block> / R12_<block> = the same restricted to one layer block (before 0-5, inside 6-17, after 18-23).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zlib

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

CAP = 58010
LR = 2e-5
R_SEED = 20261124                                            # the follow read's seed for R12's direction
ROUND1 = ("base", "Z", "U12", "R12", "U4")
ROUND2 = ("U12_before", "U12_inside", "U12_after")
ROUND3 = ("R12_before", "R12_inside", "R12_after")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------------------- the plan
def load_plan(plan_dir: str, cap: int = CAP):
    """The planner's rows.npz as records.Row objects, every check of the agreed form; -> (rows, info)."""
    from surogate.inkling_rl.records import Row

    path = os.path.join(plan_dir, "rows.npz")
    man = json.load(open(os.path.join(plan_dir, "manifest.json")))
    want = {os.path.basename(k): v for k, v in (man.get("files") or man).items() if isinstance(v, str) and len(v) == 64}
    if "rows.npz" in want and want["rows.npz"] != sha256(path):
        raise ValueError("rows.npz does not match its manifest sha256")
    z = np.load(path, allow_pickle=False)
    R = sum(1 for k in z.files if k.startswith("ids_") and k[4:].isdigit())   # not ids_crc32
    adv, trial, group, Tv, crc = (z[k] for k in ("advantage", "trial_idx", "group_idx", "T", "ids_crc32"))
    if not all(len(x) == R for x in (adv, trial, group, Tv, crc)):
        raise ValueError("per-row vectors and row count differ")
    rows = []
    for n in range(R):
        ids = z[f"ids_{n}"].astype(np.int32)
        mask = z[f"mask_{n}"].astype(bool)
        lp = z[f"lp_{n}"].astype(np.float32)
        if not (len(ids) == len(mask) == len(lp) == int(Tv[n])):
            raise ValueError(f"row {n}: lengths differ")
        if zlib.crc32(ids.astype("<i4").tobytes()) & 0xFFFFFFFF != int(crc[n]):
            raise ValueError(f"row {n}: crc32 of ids differs")
        if mask[0] or not mask.any() or len(ids) > cap:
            raise ValueError(f"row {n}: mask[0] set, no loss token, or T {len(ids)} > {cap}")
        if not np.isfinite(lp[mask]).all() or (lp[mask] > 0).any():
            raise ValueError(f"row {n}: recorded log-probs not finite or > 0 at a loss token")
        notes = {"weight": 1.0, "plan_row": n}
        if f"fin_{n}" in z.files:
            fin = z[f"fin_{n}"].astype(bool)
            if (fin & ~mask).any():
                raise ValueError(f"row {n}: final-reply tokens outside the loss mask")
            notes["final_mask"] = fin
        r = Row(ids=ids.astype(np.int64), loss_mask=mask, engine_lp=lp.astype(np.float64), advantage=float(adv[n]), versions=None,
                teacher_lp=None, routed=None, notes=notes)
        r.trial_idx, r.group_idx = int(trial[n]), int(group[n])
        rows.append(r)
    seen = []
    for r in rows:                                            # each group's rows contiguous, and an attempt in one group
        if not seen or seen[-1] != r.group_idx:
            if r.group_idx in seen:
                raise ValueError(f"group {r.group_idx} is not contiguous")
            seen.append(r.group_idx)
    tg = {}
    for r in rows:
        if tg.setdefault(r.trial_idx, (r.group_idx, r.advantage)) != (r.group_idx, r.advantage):
            raise ValueError(f"attempt {r.trial_idx}: rows with different group or advantage")
    info = {"rows": R, "row_tokens": int(sum(r.T for r in rows)), "loss_tokens": int(sum(int(r.loss_mask.sum()) for r in rows)),
            "attempts": len(tg), "groups": len(seen), "zero_advantage_attempts": sum(1 for g, a in tg.values() if a == 0.0),
            "longest": int(max(r.T for r in rows)), "rows_sha256": sha256(path)}
    return rows, info


# ------------------------------------------------------------------------------------------------------------------------------- the update
def names(n_layers: int, keys: tuple) -> list:
    return [f"{L}.{k}" for L in range(n_layers) for k in keys]


def block_of(name: str, blocks: dict) -> str:
    L = int(name.split(".", 1)[0])
    return next(b for b, rng in blocks.items() if L in rng)


def random_like(delta: list, nm: list, scale: float, seed: int = R_SEED) -> list:
    """R's direction, reproducible from the tensor names alone: per tensor a Gaussian z from numpy default_rng([seed, crc32(name UTF-8)]) with
    the tensor's shape, r = z / ||z|| x ||scale d|| (float64); r = 0 where d is exactly 0 (the follow read's random_like)."""
    out = []
    for d, n in zip(delta, nm):
        dl = d.detach().double().numpy() * float(scale)
        nrm = float(np.linalg.norm(dl))
        if nrm == 0.0:
            out.append(torch.zeros(d.shape, dtype=torch.float64))
            continue
        z = np.random.default_rng([int(seed), zlib.crc32(n.encode("utf-8")) & 0xFFFFFFFF]).standard_normal(dl.shape)
        out.append(torch.from_numpy(z / np.linalg.norm(z) * nrm))
    return out


def states(theta0: list, delta: list, nm: list, blocks: dict, which: tuple) -> dict:
    """The named parameter states (lists of fp32 tensors in the trainer's order)."""
    out = {}
    rnd = None
    for w in which:
        base, _, blk = w.partition("_")
        if base == "Z":
            out[w] = [t.clone() for t in theta0]
            continue
        if base.startswith("U"):
            k = float(base[1:])
            step = [d * k for d in delta]
        elif base.startswith("R"):
            rnd = rnd if rnd is not None else random_like(delta, nm, float(base[1:]))
            out[w] = [(t0.double() + (s if not blk or block_of(n, blocks) == blk else torch.zeros_like(s))).float()
                      for t0, s, n in zip(theta0, rnd, nm)]
            continue
        else:
            raise ValueError(w)
        out[w] = [t0 + (s if not blk or block_of(n, blocks) == blk else torch.zeros_like(s)) for t0, s, n in zip(theta0, step, nm)]
    return out


def first_order(grads: list, N: float, delta: list, keys: tuple, blocks: dict) -> dict:
    """<G, d> per block and in all, G = -N x the first mini-batch's PPO gradient (ratio 1: the advantage-weighted gradient of sum A sum_t log pi),
    and the same split into attention and expert tensors."""
    import t8_skyrl as KS

    G = [g.double() * (-float(N)) for g in grads]
    res = {"by_block": KS.first_order_by_block(G, delta, keys, blocks)}
    nk = len(keys)
    for kind, pick in (("attention", lambda k: k not in ("A13", "B13", "A2", "B2")), ("experts", lambda k: k in ("A13", "B13", "A2", "B2"))):
        res[kind] = float(sum(float((g * d.double()).sum()) for i, (g, d) in enumerate(zip(G, delta)) if pick(keys[i % nk])))
    res["grad_norm_G"] = float(torch.sqrt(sum((g * g).sum() for g in G)))
    res["delta_norm"] = float(torch.sqrt(sum((d.double() * d.double()).sum() for d in delta)))
    return res


def layers_of(flat_list: list, n_layers: int, keys: tuple) -> list:
    nk = len(keys)
    return [{k: flat_list[L * nk + j].detach().cpu().numpy() for j, k in enumerate(keys)} for L in range(n_layers)]


# ------------------------------------------------------------------------------------------------------------------------------- reads
def lp_at_mask(lp_full: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """[T] float32 with the values at the loss tokens and 0 elsewhere (lp_full: [T], position t = log p(ids[t] | ids[:t]))."""
    out = np.zeros(len(mask), dtype=np.float32)
    out[mask] = np.asarray(lp_full, dtype=np.float32)[mask]
    return out


def bits_equal(a: np.ndarray, b: np.ndarray, mask: np.ndarray):
    """(equal, differ) at the mask's positions, float32 bit patterns."""
    a32, b32 = np.asarray(a, dtype=np.float32)[mask], np.asarray(b, dtype=np.float32)[mask]
    same = a32.view(np.int32) == b32.view(np.int32)
    return int(same.sum()), int((~same).sum())


def cmp_reads(x: dict, y: dict, masks: dict):
    """x, y: {plan row: [T] float32}; compared on the rows both hold. -> (rows, equal, differ)."""
    common = sorted(set(x) & set(y))
    eq = ne = 0
    for n in common:
        e, d = bits_equal(x[n], y[n], masks[n])
        eq, ne = eq + e, ne + d
    return len(common), eq, ne


def load_reads(path: str) -> dict:
    z = np.load(path)
    return {int(k[3:]): z[k] for k in z.files if k.startswith("lp_")}


# ------------------------------------------------------------------------------------------------------------------------------- phases
def trainer_cfg(a, cards, sketch_dir=""):
    import t8_step as TS

    kw = {}
    if sketch_dir:
        kw = {"sketch_visits": (3, 12, 36, 45), "sketch_expert_stride": 64, "sketch_dir": sketch_dir}
    return TS.T8Config(bin_dir=a.bin, artifact=a.artifact, cards=cards, K=48, head="plain", attn_rank=64, attn_alpha=32.0,
                       train_keys=TS.param_keys(64), segment_keys=(), run_dir=a.run_dir, lean_host=bool(sketch_dir), max_row_tokens=CAP, **kw)


def pick_pre_rows(rows: list, k: int = 3) -> list:
    return sorted(range(len(rows)), key=lambda i: (rows[i].T, i))[:k]


def phase_pre(a):
    import t8_step as TS

    rows, info = load_plan(a.plan)
    pick = pick_pre_rows(rows)
    sub = [rows[i] for i in pick]
    tr = TS.T8Trainer(trainer_cfg(a, (int(a.cards.split(",")[0]),)))
    try:
        lz = tr.logprobs(sub)
        M = tr.masters()
        g = torch.Generator().manual_seed(21)
        for d in M:
            for key in d:
                if key.endswith("B") or key in ("B13", "B2"):
                    d[key] = torch.randn(d[key].shape, generator=g) * 0.02
        tr.set_masters(M)
        la = tr.logprobs(sub)
        sha = tr.export_adapter(os.path.join(a.work, "adapters", "pretest"), "BF16", meta={"step": "t8-micro-pre"})
    finally:
        tr.close()
    eq = ne = 0
    for r, x in zip(sub, lz):
        e, d = bits_equal(np.concatenate([[0.0], x]), r.engine_lp, r.loss_mask)
        eq, ne = eq + e, ne + d
    json.dump([{"ids": r.ids.tolist()} for r in sub], open(os.path.join(a.run_dir, "rows_ids.json"), "w"))
    np.savez(os.path.join(a.run_dir, "trainer_zero.npz"), **{f"lp_{i}": x for i, x in enumerate(lz)})
    np.savez(os.path.join(a.run_dir, "trainer_adapter.npz"), **{f"lp_{i}": x for i, x in enumerate(la)})
    rep = {"plan": info, "pre_rows": pick, "pre_rows_T": [r.T for r in sub], "adapter_sha256": sha,
           "replay": {"equal": eq, "differ": ne, "pass": bool(ne == 0 and eq > 0)}}
    json.dump(rep, open(a.out, "w"), indent=1)
    log(f"pre trainer half: rows {rep['pre_rows_T']}, replay {rep['replay']}")
    return 0 if rep["replay"]["pass"] else 4


def phase_pre_compare(a):
    import t8_gate_s5 as S5

    rep = json.load(open(a.out))
    d = a.run_dir
    e0, n0 = S5.cmp(os.path.join(d, "trainer_zero.npz"), os.path.join(d, "engine_base.json"))
    e1, n1 = S5.cmp(os.path.join(d, "trainer_adapter.npz"), os.path.join(d, "engine_adapter.json"))
    za, zb = np.load(os.path.join(d, "trainer_zero.npz")), np.load(os.path.join(d, "trainer_adapter.npz"))
    moved = int(sum(int((za[k] != zb[k]).sum()) for k in za.files))
    rep["G18a"] = {"equal": e0, "differ": n0, "pass": bool(n0 == 0 and e0 > 0)}
    rep["G18b"] = {"equal": e1, "differ": n1, "adapter_changed_positions": moved, "pass": bool(n1 == 0 and e1 > 0 and moved > 0)}
    rep["pass"] = bool(rep["G18a"]["pass"] and rep["G18b"]["pass"] and rep["replay"]["pass"])
    json.dump(rep, open(a.out, "w"), indent=1)
    print(f"replay {rep['replay']}\nG18a {rep['G18a']}\nG18b {rep['G18b']}\nT8 MICRO PRE-CHECK: {'PASS' if rep['pass'] else 'FAIL'}", flush=True)
    return 0 if rep["pass"] else 4


def phase_trainer(a):
    import t8_adapter as TA
    import t8_gate8 as G8
    import t8_skyrl as KS
    import t8_step as TS
    from surogate.inkling_rl.step import StepConfig

    rows, info = load_plan(a.plan)
    cards = tuple(int(x) for x in a.cards.split(","))
    sk = os.path.join(a.run_dir, "sketches")
    rep = {"plan": info, "cards": list(cards), "lr": LR, "time_start": time.time()}
    tr = TS.T8Trainer(trainer_cfg(a, cards, sketch_dir=sk))
    keys, n_layers = tr.keys, len(tr.shapes)
    nm = names(n_layers, keys)
    try:
        with G8.MemWatch() as mw:
            shortest = min(rows, key=lambda r: r.T)
            zc = tr.check_zero_adapter([shortest])            # M1 (trainer half), every worker
            theta0 = [m.detach().clone() for m in tr.opt.masters]
            t0 = time.time()
            res = TS.ppo_batch(tr, rows, 0, LR, 1, StepConfig(max_policy_lag=None))
            rep["step_wall_s"] = time.time() - t0
            out, gnorm = res[0]
            theta1 = [m.detach().clone() for m in tr.opt.masters]
            delta = [t1 - t0_ for t0_, t1 in zip(theta0, theta1)]
            N = float(out.metrics["total_tokens"])
            m = out.metrics
            fo = first_order(out.grads, N, delta, keys, KS.BLOCKS)
            rep["first_order"] = fo
            rep["step"] = {"N": N, "rows_used": len(out.rows), "rows_refused": len(rows) - len(out.rows), "grad_norm": gnorm,
                           "finite": all(bool(torch.isfinite(g).all()) for g in out.grads),
                           "replay_checked_rows": m["replay_checked_rows"], "replay_skipped_rows": m["replay_skipped_rows"],
                           "replay_checked_final_tokens": m.get("replay_checked_final_tokens"), "stale_checked_rows": m["stale_checked_rows"],
                           "ratio_max": m.get("ppo_ratio_max"), "clip_ratio": m.get("ppo_clip_ratio"), "tok_per_s_per_card": m["tok_per_s_per_card"],
                           "row_tokens": m["row_tokens"], "wall_s": m["wall_s"],
                           "refused": {k: m.get(k) for k in ("refused_long_rows", "r3_refused_rows", "r3_lagged_uncovered_refused_rows")},
                           "workers": [{"card": x["card"], "VmHWM_gib": x["VmHWM_gib"], "card_peak_torch_gib": x["card_peak_torch_gib"]}
                                       for x in m["workers"]]}
            rep["zero_adapter_checks"] = len(zc)
            # the files of the follow read
            fo_json = {**fo["by_block"], "N": N, "rows_used": len(out.rows), "rows_refused": len(rows) - len(out.rows), "k": 1,
                       "G": "-N x grads of mini-batch 0", "d": "theta1 - theta0", "attention": fo["attention"], "experts": fo["experts"]}
            json.dump(fo_json, open(os.path.join(a.run_dir, "first_order.json"), "w"), indent=1)
            os.makedirs(os.path.join(a.run_dir, "masters"), exist_ok=True)
            torch.save(dict(zip(nm, theta0)), os.path.join(a.run_dir, "masters", "theta0.pt"))
            torch.save(dict(zip(nm, theta1)), os.path.join(a.run_dir, "masters", "theta1.pt"))
            a_keys = [i for i, n in enumerate(nm) if n.split(".", 1)[1] in ("A13", "A2") or n.split(".", 1)[1].endswith("A")]
            rep["A_delta_max_abs"] = max(float(delta[i].abs().max()) for i in a_keys)  # B0 = 0: A has no gradient, d_A == 0 exactly
            json.dump({"replay": {"rows": len(rows), "rows_checked": int(m["replay_checked_rows"]), "mismatch_tokens": 0},
                       "ratio_first_update": {"loss_tokens": int(N), "tokens_not_exactly_1": 0 if m["stale_checked_rows"] == len(rows) else None},
                       "zero_adapter": {"bitwise": True, "workers_rows_checked": len(zc)},
                       "finite": all(bool(torch.isfinite(g).all()) for g in out.grads), "rows_refused": len(rows) - len(out.rows),
                       "A_delta_exactly_zero": rep["A_delta_max_abs"] == 0.0},
                      open(os.path.join(a.run_dir, "gates.json"), "w"), indent=1)  # the step raises on any replay / stale / zero-adapter mismatch
            pos, old = {}, {}
            for r in rows:
                n = r.notes["plan_row"]
                p = np.nonzero(r.loss_mask)[0].astype(np.int32)
                pos[f"pos_{n}"] = p
                old[f"old_{n}"] = np.asarray(r.notes["old_lp"], dtype=np.float32)[p - 1]
            np.savez(os.path.join(a.run_dir, "rows_used.npz"), **pos, **old, N=np.array([N]))
            # M5, trainer half: log-probs at U12 on the round-trip rows
            u12 = states(theta0, delta, nm, KS.BLOCKS, ("U12",))["U12"]
            rt = [r for r in rows if r.group_idx in sorted({x.group_idx for x in rows})[:a.rt_groups]]
            tr.set_masters([{k: u12[L * len(keys) + j] for j, k in enumerate(keys)} for L in range(n_layers)])
            lu = tr.logprobs(rt)
            del u12
            os.makedirs(os.path.join(a.run_dir, "roundtrip"), exist_ok=True)   # a subset of rows: kept out of reads/ (reads/ holds every row)
            np.savez(os.path.join(a.run_dir, "roundtrip", "trainer_U12.npz"),
                     **{f"lp_{r.notes['plan_row']}": lp_at_mask(np.concatenate([[0.0], x]), r.loss_mask) for r, x in zip(rt, lu)})
            rep["round_trip_rows"] = [r.notes["plan_row"] for r in rt]
            rank = tr.B.RANK
            tr.close()                                        # the workers' host memory is free before the exports
            # the adapters, one state at a time, exported from the parameter states directly (no worker state is involved)
            rep["adapters"] = {}
            for w in ROUND1[1:] + ROUND2 + ROUND3:
                st = states(theta0, delta, nm, KS.BLOCKS, (w,))[w]
                rep["adapters"][w] = TA.export_combined(layers_of(st, n_layers, keys), os.path.join(a.work, "adapters", w), 64, 32.0, rank,
                                                        "BF16", metadata={"step": f"t8-micro-{w}"})
                del st
            json.dump(rep["adapters"], open(os.path.join(a.run_dir, "adapters.json"), "w"), indent=1)
        rep["min_mem_available_gb"] = mw.low
    finally:
        tr.close()
    files = out.metrics.get("sketch_files", [])
    sketched, zero_counted = set(), 0
    for f in files:
        d_ = torch.load(f, map_location="cpu", weights_only=False)
        if d_["rows"] > 0 and d_["tensors"]:
            sketched.add(d_["trial"])
        zero_counted += d_["rows_skipped_zero_factor"]
    want = {r.trial_idx for r in rows if r.advantage != 0.0}
    rep["sketches"] = {"files": len(files), "sketched_attempts": sorted(sketched), "missing": sorted(want - sketched),
                       "zero_advantage_rows_counted": zero_counted, "zero_advantage_rows": sum(1 for r in rows if r.advantage == 0.0),
                       "bytes": sum(os.path.getsize(f) for f in files)}
    s = rep["step"]
    rep["M3"] = bool(s["stale_checked_rows"] == len(rows) and s["ratio_max"] == 1.0 and s["clip_ratio"] == 0.0)
    rep["M4"] = bool(s["finite"] and s["rows_refused"] == 0 and s["replay_skipped_rows"] == 0 and s["N"] == info["loss_tokens"]
                     and rep["A_delta_max_abs"] == 0.0)
    rep["M2_trainer"] = bool(s["replay_checked_rows"] == len(rows))  # the step raises on any replay mismatch
    rep["M6_trainer"] = bool(not rep["sketches"]["missing"] and zero_counted == rep["sketches"]["zero_advantage_rows"])
    rep["host_peak_sum_gib"] = sum(x["VmHWM_gib"] for x in s["workers"])
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    ok = rep["M2_trainer"] and rep["M3"] and rep["M4"] and rep["M6_trainer"]
    log(f"trainer half: N {s['N']:.0f}, first order {fo['by_block']}, M2t {rep['M2_trainer']} M3 {rep['M3']} M4 {rep['M4']} M6t {rep['M6_trainer']}")
    return 0 if ok else 4


def phase_compare(a):
    rows, info = load_plan(a.plan)
    masks = {r.notes["plan_row"]: r.loss_mask for r in rows}
    rd = os.path.join(a.run_dir, "reads")
    rep = json.load(open(a.out))
    R = {}
    fl = sorted(f for f in os.listdir(rd) if f.endswith(".npz"))
    merged = {f[:-4] for f in fl if ".part" not in f}
    for f in fl:                                              # <name>.npz, or (if absent) the engine parts <name>.part<k>.npz merged
        name = f[:-4].split(".part")[0]
        if ".part" not in f or name not in merged:
            part = load_reads(os.path.join(rd, f))
            if set(part) & set(R.get(name, {})):
                raise ValueError(f"{f}: a row read twice")
            R.setdefault(name, {}).update(part)
    ad = json.load(open(os.path.join(a.run_dir, "adapters.json")))
    eng = open(a.engine_sha_file).read().strip() if a.engine_sha_file else ""
    for name, v in R.items():                                 # one file per adapter for the follow read, and its JSON
        if not os.path.exists(os.path.join(rd, name + ".npz")):
            np.savez(os.path.join(rd, name + ".npz"), **{f"lp_{n}": x for n, x in sorted(v.items())})
        json.dump({"adapter_sha256": ad.get(name), "head": "plain", "engine_sha": eng, "source": "engine"},
                  open(os.path.join(rd, name + ".json"), "w"), indent=1)
    R["trainer_U12"] = load_reads(os.path.join(a.run_dir, "roundtrip", "trainer_U12.npz"))
    rep["reads"] = {k: len(v) for k, v in R.items()}
    u = np.load(os.path.join(a.run_dir, "rows_used.npz"))
    old = {}
    for r in rows:
        n = r.notes["plan_row"]
        x = np.zeros(r.T, dtype=np.float32)
        x[u[f"pos_{n}"]] = u[f"old_{n}"]
        old[n] = x
    rec = {r.notes["plan_row"]: r.engine_lp.astype(np.float32) for r in rows}
    full = len(rows)
    out = {}
    out["M1_engine_Z_vs_base"] = cmp_reads(R["Z"], R["base"], masks)
    out["M2_old_vs_recorded"] = cmp_reads(old, rec, masks)
    out["M2_old_vs_engine_base"] = cmp_reads(old, R["base"], masks)
    out["M5_trainer_vs_engine_U12"] = cmp_reads(R["trainer_U12"], R["U12"], masks)
    rep["compare"] = {k: {"rows": v[0], "equal": v[1], "differ": v[2]} for k, v in out.items()}
    rep["M1"] = bool(out["M1_engine_Z_vs_base"][2] == 0 and out["M1_engine_Z_vs_base"][1] > 0)
    rep["M2"] = bool(rep.get("M2_trainer") and all(out[k][2] == 0 and out[k][0] == full for k in ("M2_old_vs_recorded", "M2_old_vs_engine_base")))
    rep["M5"] = bool(out["M5_trainer_vs_engine_U12"][2] == 0 and out["M5_trainer_vs_engine_U12"][0] == len(rep["round_trip_rows"]))
    need = ("first_order.json", "gates.json", "masters/theta0.pt", "masters/theta1.pt", "rows_used.npz", "adapters.json")
    rep["M6"] = bool(rep.get("M6_trainer") and all(os.path.exists(os.path.join(a.run_dir, f)) for f in need)
                     and all(len(R.get(w, {})) == full for w in ROUND1))
    rep["reads_complete"] = {w: len(R.get(w, {})) == full for w in ROUND1 + ROUND2 + ROUND3}
    rep["pass"] = bool(rep["M1"] and rep["M2"] and rep.get("M3") and rep.get("M4") and rep["M5"] and rep["M6"])
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    for k, v in rep["compare"].items():
        print(f"{k}: {v}")
    print(f"M1 {rep['M1']} M2 {rep['M2']} M3 {rep.get('M3')} M4 {rep.get('M4')} M5 {rep['M5']} M6 {rep['M6']}")
    print(f"T8 MICRO SESSION: {'PASS' if rep['pass'] else 'FAIL'}", flush=True)
    return 0 if rep["pass"] else 4


def phase_check(a):
    rows, info = load_plan(a.plan)
    print(json.dumps(info, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("check", "pre", "pre_compare", "trainer", "compare"))
    ap.add_argument("--plan", default="")
    ap.add_argument("--bin", default="")
    ap.add_argument("--artifact", default="")
    ap.add_argument("--cards", default="0")
    ap.add_argument("--work", default="", help="the run's scratch directory (adapters; not copied back)")
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--rt-groups", type=int, default=1, help="round trip on the rows of the first n groups")
    ap.add_argument("--engine-sha-file", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    return {"check": phase_check, "pre": phase_pre, "pre_compare": phase_pre_compare, "trainer": phase_trainer, "compare": phase_compare}[a.phase](a)


if __name__ == "__main__":
    sys.exit(main())
