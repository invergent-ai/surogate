#!/usr/bin/env python3
"""The trainer side of a multi-step RL run on real task rollouts: one long-lived process on a GPU box that takes one step plan at a time, applies ONE
optimizer step to it and hands back the next adapter. The rollouts, the plan builder and the loop that chains them live outside this file.

Trainer: T8Trainer (stack C1 by default), expert LoRA r16, B only (train_keys B13, B2; segment_keys ()), K 48, plain head, row cap 58,010; the backward
is exact or, with --pattern-grad-from 48 --gate-grad-from 48, holds every visit's attention pattern and router gates constant ("QD").
One optimizer step per plan: tr.ppo_batch(rows, v, lr, 1) (SkyRL's dual-clip PPO; the replay gate demands the trainer's forward at adapter v ==
the rollout's decode-time log-probs, bit for bit, on every loss token of every row).

  rl_step_server.py serve --work W --cards 0,1,2,3,4,5 --bin BIN --artifact ART --panel DIR [--lr 1e-4] [--start-version V --resume-ckpt DIR]
                          [--hold-workers 0|1] [--damage-every 5] [--idle-min 240]
  rl_step_server.py smoke --work W --cards ... --bin BIN --artifact ART --panel DIR --plan DIR [--stage-a-groups 1] [--budget-min 70]

serve: requests are directories under W/inbox, each made complete by the file READY (written last by the caller):
  step-<v>/plan/ + READY {"version": v, "adapter_sha256": <sha256 of the adapter file the rollout engines held>, "damage": bool}
      -> one update on the plan's rows at adapter v (must be this trainer's current version, and the sha must be the one it exported for v;
         version 0 = the zero adapter, any sha), adapter v+1 exported to W/outbox/step-<v>/adapter/, checkpoint W/ckpt/v<v+1>/ (masters.pt,
         optimizer.npz, meta.json), step.json; DONE written last. A failure writes FAILED.json and ends the process (exit 4 gate, 5 input, 1 other).
  gate-<name>/plan/ + READY {"version": v, "adapter_sha256": ...}   forward only: the replay gate on the plan's rows at the current adapter; no update
  control/READY {"seed": n}   the same-norm random-direction control of the current adapter (per tensor theta0 + a Gaussian direction of the
      accumulated update's norm; exactly theta0 where the update is zero), exported to W/outbox/control/adapter/; the trainer's state is not touched
  STOP   the process ends (exit 0)
A plan is a step-plan directory as t8_stepsession.load_step_plan reads it (rows with token ids, loss masks, recorded log-probs and advantages).
Damage panel (--panel: a plan-form directory of base rows from tasks NOT in the run): teacher-forced nats per token = mean over its loss tokens
of (log-prob at theta0 - log-prob at the current adapter); read at the start (must equal the recorded engine log-probs bit for bit) and after an
update when the new version is a multiple of --damage-every or the request asks for it.
--hold-workers 0 (default): the workers are closed after each request and started again for the next one from the checkpoint just written (hash
checked): no card is held while rollouts run elsewhere. 1: the same T8Trainer object lives across requests.

Measured with this server (8 x RTX 5090: 6 trainer cards here, rollouts through an agent harness on 4 cards of a second box): 18 steps on 8 agentic
office tasks, the same 8 every step, 8 fresh attempts each (64 attempts, about 8 M row tokens a step), lr 1e-4, QD backward, training rollouts
sampled untruncated (temperature 1, top_p 1, top_k 0). Pass rate on those tasks: 67.2 % over steps 0-2, 67.0 % over steps 3-7, 76.5 % over steps
8-12, 79.6 % over steps 13-17 (+12.4 pp against steps 0-2, 95 % interval 5.1 to 19.7); a final batch under the last adapter 84.4 % against 64.1 %
under its same-norm random-direction control (+20.3 pp, interval 7.8 to 32.8); 7 of 8 tasks higher late than early; damage panel 0.0037, 0.0036,
0.0038, 0.0043 nats per token after updates 5, 10, 15, 18. Every update: replay-checked rows = rows (243-308), about 1,000-1,460 tok/s per card,
about 24 minutes; 70 card-hours in all. One run, the trained tasks themselves (fresh attempts): not a transfer result. On long rows use the QD
backward: with the exact one the engine did not follow the gradient in an earlier check (0.04 against 0.96).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

LABEL = "rl-step-server"
EXIT_OK, EXIT_FAIL, EXIT_GATE, EXIT_INPUT = 0, 1, 4, 5
CAP = 58010


def now():
    return time.strftime("%H:%M:%S", time.gmtime(time.time() + 7200))


def log(m):
    print(f"[{now()}] {m}", flush=True)


def jdump(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def mem_available_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576.0
    return -1.0


def sums_file(d):
    """SHA256SUMS over every regular file under d (relative names), written last but one"""
    out = []
    for root, _, files in os.walk(d):
        for f in sorted(files):
            if f in ("SHA256SUMS", "DONE", "FAILED.json"):
                continue
            p = os.path.join(root, f)
            out.append(f"{sha256(p)}  {os.path.relpath(p, d)}")
    with open(os.path.join(d, "SHA256SUMS.tmp"), "w") as f:
        f.write("\n".join(sorted(out)) + "\n")
    os.replace(os.path.join(d, "SHA256SUMS.tmp"), os.path.join(d, "SHA256SUMS"))


class Core:
    """The trainer and what the run reads from it. `mods` = (TS, SS, MI) so that the CPU test can pass fakes."""

    def __init__(self, a, mods):
        self.a, (self.TS, self.SS, self.MI) = a, mods
        self.cards = tuple(int(x) for x in a.cards.split(","))
        self.tr = None
        self.theta0 = None
        self.nm = None
        self.panel = None
        self.panel_lp0 = None
        self.version = 0
        self.adapter_sha = {}            # version -> sha256 of the adapter file this trainer exported for it
        self.worker_starts = []

    # ------------------------------------------------------------------ workers
    def cfg(self):
        TS = self.TS
        c = TS.T8Config(bin_dir=self.a.bin, artifact=self.a.artifact, cards=self.cards, K=self.a.K, head="plain", max_row_tokens=CAP, segment_keys=(),
                        train_keys=("B13", "B2"), run_dir=os.path.join(self.a.work, "trainer"), pattern_grad_from=self.a.pattern_grad_from,
                        gate_grad_from=self.a.gate_grad_from, bank_grad_from=self.a.bank_grad_from)
        if c.stack != self.a.stack:
            raise SystemExit(f"the trainer's stack is {c.stack}, asked for {self.a.stack} (T8_STACK)")
        return c

    def open(self, ckpt=None):
        """workers up at theta0; then, with ckpt, the masters and the optimizer state of that checkpoint (hashes checked by load_ckpt)"""
        if self.tr is not None:
            return
        t0 = time.time()
        os.makedirs(os.path.join(self.a.work, "trainer"), exist_ok=True)
        cfg = self.cfg()
        self.tr = self.TS.T8Trainer(cfg)
        self.nm = self.MI.names(len(self.tr.shapes), tuple(self.tr.shapes[0]))
        th = self.SS.masters_list(self.tr)
        sha0 = self.SS.state_sha(th, self.nm)
        if self.theta0 is None:
            self.theta0, self.theta0_sha = th, sha0
        elif sha0 != self.theta0_sha:
            raise RuntimeError("a new trainer's initial masters differ from the first one's (theta0 is not reproducible)")
        meta = None
        if ckpt:
            meta = self.SS.load_ckpt(self.tr, ckpt, self.nm)
            if int(meta["update"]) != self.version:
                raise RuntimeError(f"checkpoint {ckpt} is of update {meta['update']}, the trainer is at version {self.version}")
        self.worker_starts.append({"seconds": time.time() - t0, "ckpt": ckpt, "version": self.version, "stack": cfg.stack})
        log(f"workers up on cards {self.cards} in {time.time() - t0:.0f} s (stack {cfg.stack}, K {cfg.K}, train keys {cfg.train_keys}, "
            f"device_sum {cfg.device_sum}, lean_host {cfg.lean_host}); {'checkpoint ' + ckpt if ckpt else 'theta0'}; MemAvailable {mem_available_gb():.0f} GB")
        return meta

    def close(self):
        if self.tr is not None:
            self.tr.close()
            self.tr = None
            time.sleep(float(os.environ.get("RLS_CLOSE_SLEEP", "3")))   # the cards' memory comes back before the next start

    # ------------------------------------------------------------------ reads
    def load_panel(self):
        rows, info = self.MI.load_plan(self.a.panel, cap=CAP)
        self.panel = rows
        self.panel_info = info
        log(f"damage panel: {info['rows']} rows, {info['row_tokens']} row tokens, {info['loss_tokens']} loss tokens ({self.a.panel})")

    def panel_read(self):
        t0 = time.time()
        lp = self.tr.logprobs(self.panel)
        return [np.asarray(x, np.float32) for x in lp], time.time() - t0

    def panel_base(self):
        """at theta0: the trainer's forward must be the panel rows' recorded engine log-probs, bit for bit (they were sampled with the zero adapter)"""
        lp, sec = self.panel_read()
        bad = sum(int((x[r.loss_mask[1:]] != r.engine_lp[1:].astype(np.float32)[r.loss_mask[1:]]).sum()) for x, r in zip(lp, self.panel))
        tok = sum(r.T for r in self.panel)
        self.panel_lp0 = lp
        rec = {"rows": len(self.panel), "row_tokens": tok, "loss_tokens": self.panel_info["loss_tokens"], "loss_tokens_differ_from_engine": bad, "seconds": sec,
               "forward_tok_per_s_per_card": tok / max(sec, 1e-9) / len(self.cards)}
        log(f"panel at theta0: {bad} loss tokens differ from the engine's records; {sec:.0f} s, forward {rec['forward_tok_per_s_per_card']:.0f} tok/s per card")
        if bad:
            raise self.TS.T8GateError(f"PANEL GATE: {bad} loss tokens of the damage panel differ from the engine's records at theta0")
        return rec

    def damage(self):
        lp, sec = self.panel_read()
        num = sum(float((x0.astype(np.float64)[r.loss_mask[1:]] - x1.astype(np.float64)[r.loss_mask[1:]]).sum()) for x0, x1, r in zip(self.panel_lp0, lp, self.panel))
        den = sum(int(r.loss_mask[1:].sum()) for r in self.panel)
        absd = sum(float(np.abs(x0.astype(np.float64)[r.loss_mask[1:]] - x1.astype(np.float64)[r.loss_mask[1:]]).sum()) for x0, x1, r in zip(self.panel_lp0, lp, self.panel))
        rec = {"version": self.version, "nats_per_token": num / max(den, 1), "mean_abs_shift": absd / max(den, 1), "loss_tokens": den, "seconds": sec, "line": 0.010}
        log(f"damage panel at version {self.version}: {rec['nats_per_token']:+.6f} nats per token over {den} tokens (reference line 0.010); mean |shift| {rec['mean_abs_shift']:.6f}; {sec:.0f} s")
        return rec

    # ------------------------------------------------------------------ the update
    def update(self, rows, lr, out_dir, want_damage=False, zero_check=False):
        """one optimizer step on `rows` at the current version; -> the step record. The adapter of the new version is in out_dir/adapter."""
        TS, SS = self.TS, self.SS
        v = self.version
        rec = {"version": v, "rows": len(rows), "row_tokens": int(sum(r.T for r in rows)), "loss_tokens": int(sum(int(r.loss_mask.sum()) for r in rows)),
               "groups": len({r.group_idx for r in rows}), "attempts": len({r.trial_idx for r in rows}),
               "zero_advantage_rows": sum(1 for r in rows if r.advantage == 0.0), "lr": lr, "mem_available_gb_before": mem_available_gb(),
               "backward_options": {"pattern_grad_from": self.a.pattern_grad_from, "gate_grad_from": self.a.gate_grad_from, "bank_grad_from": self.a.bank_grad_from}}
        if zero_check:
            t = time.time()
            rec["zero_adapter_checks"] = len(self.tr.check_zero_adapter(rows[:2]))
            rec["zero_check_s"] = time.time() - t
        t1 = time.time()
        outs = self.tr.ppo_batch(rows, v, lr, 1)
        rec["ppo_batch_s"] = time.time() - t1
        o, norm = outs[0]
        m = o.metrics
        live = sum(1 for r in rows if r.loss_mask.any())
        rec.update(grad_norm=float(norm), replay_checked_rows=int(m["replay_checked_rows"]), replay_skipped_rows=int(m["replay_skipped_rows"]), live_rows=live,
                   stale_checked_rows=int(m.get("stale_checked_rows", -1)), zero_seed_rows=int(m.get("zero_seed_rows", -1)),
                   refused_long_rows=float(m.get("refused_long_rows", 0)), ppo_clip_ratio=m.get("ppo_clip_ratio"), ppo_ratio_max=m.get("ppo_ratio_max"),
                   policy_loss=m.get("policy_loss"), compute_wall_s=float(m["wall_s"]), compute_tok_per_s_per_card=float(m["tok_per_s_per_card"]),
                   old_logprob_forward_s=rec["ppo_batch_s"] - float(m["wall_s"]), oom_retries=int(sum(r.get("oom_retries", 0) for r in o.rows)),
                   host_peak_gib=[w.get("VmHWM_gib") for w in m.get("workers", [])], rollout_train_abs_diff_max=m.get("rollout_train_logprobs_abs_diff_max"),
                   step_tok_per_s_per_card=rec["row_tokens"] / max(rec["ppo_batch_s"], 1e-9) / len(self.cards), cards=len(self.cards))
        if rec["replay_checked_rows"] != live or rec["replay_skipped_rows"]:
            raise TS.T8GateError(f"REPLAY GATE COVERAGE: {rec['replay_checked_rows']} rows checked, {live} rows with loss tokens, {rec['replay_skipped_rows']} skipped")
        self.version = v + 1
        t2 = time.time()
        ad = os.path.join(out_dir, "adapter")
        self.tr.export_adapter(ad, dtype="BF16", meta={"label": LABEL, "version": str(self.version)})
        s = sha256(os.path.join(ad, "adapter_model.safetensors"))
        self.adapter_sha[self.version] = s
        rec.update(adapter_version=self.version, adapter_sha256=s, export_s=time.time() - t2)
        t3 = time.time()
        ck = os.path.join(self.a.work, "ckpt", f"v{self.version}")
        meta = SS.save_ckpt(self.tr, ck, self.nm, {"update": self.version, "label": LABEL, "adapter_sha256": s, "theta0_sha256": self.theta0_sha, "lr": lr})
        rec.update(ckpt=ck, ckpt_masters_sha256=meta["masters_sha256"], ckpt_s=time.time() - t3)
        cur = SS.masters_list(self.tr)
        d2 = sum(float((c.double() - t.double()).pow(2).sum()) for c, t in zip(cur, self.theta0))
        rec["update_norm_from_theta0"] = d2 ** 0.5
        rec["tensors_moved"] = sum(1 for c, t in zip(cur, self.theta0) if bool((c != t).any()))
        if want_damage:
            rec["damage"] = self.damage()
        rec["mem_available_gb_after"] = mem_available_gb()
        log(f"update at version {v}: rows {rec['rows']} ({rec['row_tokens']} tokens, {rec['loss_tokens']} loss tokens), replay-checked {rec['replay_checked_rows']} of {live}, "
            f"grad norm {rec['grad_norm']:.6f}, clip share {rec['ppo_clip_ratio']}, compute {rec['compute_wall_s']:.0f} s at {rec['compute_tok_per_s_per_card']:.0f} tok/s per card, "
            f"old log-probs {rec['old_logprob_forward_s']:.0f} s, whole step {rec['step_tok_per_s_per_card']:.0f} tok/s per card, host peak {rec['host_peak_gib']}, adapter v{self.version} {s[:16]}")
        return rec

    def gate_only(self, rows):
        t0 = time.time()
        lp = self.tr.logprobs(rows)
        bad_rows, bad_tok, tok = 0, 0, 0
        for x, r in zip(lp, rows):
            x = np.asarray(x, np.float32)
            b = int((x[r.loss_mask[1:]] != r.engine_lp[1:].astype(np.float32)[r.loss_mask[1:]]).sum())
            bad_tok += b
            bad_rows += int(b > 0)
            tok += int(r.loss_mask.sum())
        rec = {"version": self.version, "rows": len(rows), "row_tokens": int(sum(r.T for r in rows)), "loss_tokens": tok, "rows_with_a_difference": bad_rows,
               "loss_tokens_differ": bad_tok, "seconds": time.time() - t0, "replay_gate": "PASS" if bad_tok == 0 else "FAIL"}
        log(f"gate-only read at version {self.version}: {rec['replay_gate']}: {bad_tok} of {tok} loss tokens differ on {bad_rows} of {len(rows)} rows; {rec['seconds']:.0f} s")
        return rec

    def control(self, out_dir, seed):
        """the control: per tensor theta0 + a random direction of the accumulated update's norm; exported without touching the trainer"""
        import torch
        cur = self.SS.masters_list(self.tr)
        g = torch.Generator().manual_seed(int(seed))
        ctrl, moved = [], 0
        for t0, c in zip(self.theta0, cur):
            delta = c.double() - t0.double()
            n = float(delta.norm())
            if n == 0.0:
                ctrl.append(t0.clone())
            else:
                z = torch.randn(delta.shape, generator=g, dtype=torch.float64)
                ctrl.append((t0.double() + z * (n / float(z.norm()))).float())
                moved += 1
        keys = self.tr.keys
        layers, i = [], 0
        for _ in self.tr.shapes:
            layers.append({k: ctrl[i + j].numpy() for j, k in enumerate(keys)})
            i += len(keys)
        from surogate.inkling_rl import adapter as AD
        ad = os.path.join(out_dir, "adapter")
        AD.export_adapter(layers, ad, self.tr.B.RANK, int(self.tr.B.ALPHA), "BF16", metadata={"label": LABEL, "control": "true", "of_version": str(self.version)})
        s = sha256(os.path.join(ad, "adapter_model.safetensors"))
        n_c = sum(float((c.double() - t.double()).pow(2).sum()) for c, t in zip(ctrl, self.theta0)) ** 0.5
        n_v = sum(float((c.double() - t.double()).pow(2).sum()) for c, t in zip(cur, self.theta0)) ** 0.5
        rec = {"of_version": self.version, "seed": int(seed), "tensors_moved": moved, "control_norm": n_c, "update_norm": n_v, "adapter_sha256": s}
        log(f"control of version {self.version}: {moved} tensors, norm {n_c:.6f} (update {n_v:.6f}), adapter {s[:16]}")
        return rec


def load_rows(SS, plan_dir, update=0):
    plan = SS.load_step_plan(plan_dir)
    return plan, [plan.rows[i] for i in plan.updates[update]]


def fail(out_dir, e, code):
    os.makedirs(out_dir, exist_ok=True)
    jdump(os.path.join(out_dir, "FAILED.json"), {"error": f"{type(e).__name__}: {e}"[:4000], "exit": code, "time": now()})
    log(f"FAILED ({code}): {type(e).__name__}: {str(e)[:600]}")


def serve(a, mods):
    TS, SS, MI = mods
    W = a.work
    inbox, outbox = os.path.join(W, "inbox"), os.path.join(W, "outbox")
    for d in (inbox, outbox, os.path.join(W, "ckpt")):
        os.makedirs(d, exist_ok=True)
    t_proc = time.time()
    core = Core(a, mods)
    core.version = int(a.start_version)
    ckpt = a.resume_ckpt or None
    if bool(ckpt) != (core.version > 0):
        raise SystemExit("--start-version > 0 needs --resume-ckpt, and a checkpoint needs its version")
    events = os.path.join(W, "events.jsonl")

    def event(kind, **kw):
        with open(events, "a") as f:
            f.write(json.dumps({"time": now(), "kind": kind, **kw}, default=float) + "\n")

    try:
        core.load_panel()
        core.open()                                           # theta0 first: the panel's base read needs it
        base = core.panel_base()
        jdump(os.path.join(outbox, "panel_base.json"), base)
        if ckpt:
            meta = SS.load_ckpt(core.tr, ckpt, core.nm)
            if int(meta["update"]) != core.version:
                raise SS.PlanError(f"checkpoint {ckpt} is of update {meta['update']}, --start-version is {core.version}")
            if meta.get("theta0_sha256") not in (None, core.theta0_sha):
                raise SS.PlanError("the checkpoint's theta0 is not this trainer's initial state")
            core.adapter_sha[core.version] = meta.get("adapter_sha256")
            d = core.damage()
            jdump(os.path.join(outbox, f"resume-v{core.version}.json"), {"ckpt": ckpt, "meta": {k: v for k, v in meta.items() if k != "sha256"}, "damage": d})
        event("start", version=core.version, ckpt=ckpt, panel_base=base, theta0_sha256=core.theta0_sha, hold_workers=a.hold_workers)
        jdump(os.path.join(outbox, "READY.json"), {"version": core.version, "theta0_sha256": core.theta0_sha, "time": now(), "cards": list(core.cards),
                                                    "startup_s": time.time() - t_proc, "panel_base": base, "hold_workers": a.hold_workers})
        if not a.hold_workers:
            core.close()
    except TS.T8GateError as e:
        fail(outbox, e, EXIT_GATE)
        core.close()
        return EXIT_GATE
    except (SS.PlanError, ValueError, OSError) as e:
        fail(outbox, e, EXIT_INPUT)
        core.close()
        return EXIT_INPUT
    t_idle = time.time()
    done = set()
    while True:
        if os.path.exists(os.path.join(inbox, "STOP")):
            log("STOP file: ending")
            event("stop")
            core.close()
            return EXIT_OK
        req = None
        for name in sorted(os.listdir(inbox)):
            if name in done or not os.path.isfile(os.path.join(inbox, name, "READY")):
                continue
            if os.path.exists(os.path.join(outbox, name, "DONE")):
                done.add(name)
                continue
            req = name
            break
        if req is None:
            if time.time() - t_idle > a.idle_min * 60:
                log(f"no request for {a.idle_min} min: ending")
                event("idle_end")
                core.close()
                return EXIT_OK
            time.sleep(float(os.environ.get("RLS_IDLE_SLEEP", "5")))
            continue
        rdir, odir = os.path.join(inbox, req), os.path.join(outbox, req)
        if os.path.isdir(odir):
            shutil.rmtree(odir)
        os.makedirs(odir)
        t0 = time.time()
        code = EXIT_OK
        try:
            ready = json.load(open(os.path.join(rdir, "READY")))
            cur_ckpt = os.path.join(W, "ckpt", f"v{core.version}") if core.version > 0 else None
            if core.version > 0 and a.resume_ckpt and core.version == int(a.start_version) and not os.path.isdir(cur_ckpt):
                cur_ckpt = a.resume_ckpt
            core.open(cur_ckpt)
            rec = {"request": req, "ready": ready, "time_start": now()}
            if req.startswith("step-") or req.startswith("gate-"):
                if int(ready["version"]) != core.version:
                    raise SS.PlanError(f"{req}: the request is for version {ready['version']}, the trainer is at {core.version}")
                want = core.adapter_sha.get(core.version)
                if core.version > 0 and ready.get("adapter_sha256") != want:
                    raise SS.PlanError(f"{req}: the rollout engines held adapter {str(ready.get('adapter_sha256'))[:16]}, this trainer's version {core.version} is {str(want)[:16]}")
                plan, rows = load_rows(SS, os.path.join(rdir, "plan"))
                if len(plan.updates) != 1:
                    raise SS.PlanError(f"{req}: the plan has {len(plan.updates)} updates; one optimizer step per plan")
                rec["plan"] = {k: plan.info[k] for k in ("rows", "row_tokens", "loss_tokens", "attempts", "groups", "zero_advantage_attempts", "longest", "rows_sha256", "rows_json_sha256")}
                if req.startswith("step-"):
                    rec["step"] = core.update(rows, a.lr, odir, want_damage=bool(ready.get("damage")) or ((core.version + 1) % a.damage_every == 0),
                                              zero_check=(core.version == 0))
                else:
                    rec["gate"] = core.gate_only(rows)
            elif req.startswith("control"):
                rec["control"] = core.control(odir, int(ready.get("seed", 20261127 + 7)))
            else:
                raise SS.PlanError(f"unknown request {req}")
            rec.update(seconds=time.time() - t0, time_end=now(), worker_starts=core.worker_starts[-1:], version_after=core.version)
            jdump(os.path.join(odir, "step.json"), rec)
            sums_file(odir)
            with open(os.path.join(odir, "DONE.tmp"), "w") as f:
                f.write(now() + "\n")
            os.replace(os.path.join(odir, "DONE.tmp"), os.path.join(odir, "DONE"))
            event("done", request=req, seconds=rec["seconds"], version_after=core.version)
            done.add(req)
            # the orchestrator has copied everything older before it sent this request: keep the newest checkpoint and adapter only
            for name in os.listdir(os.path.join(W, "ckpt")):
                if name.startswith("v") and name[1:].isdigit() and int(name[1:]) < core.version:
                    shutil.rmtree(os.path.join(W, "ckpt", name), ignore_errors=True)
            for name in os.listdir(outbox):
                if name.startswith("step-") and name[5:].isdigit() and int(name[5:]) < core.version - 1:
                    shutil.rmtree(os.path.join(outbox, name, "adapter"), ignore_errors=True)
            if os.path.isdir(os.path.join(rdir, "plan")):
                shutil.rmtree(os.path.join(rdir, "plan"), ignore_errors=True)   # the plan stays on the shared disk
        except TS.T8GateError as e:
            fail(odir, e, EXIT_GATE)
            code = EXIT_GATE
        except (SS.PlanError, KeyError, ValueError, OSError) as e:
            fail(odir, e, EXIT_INPUT)
            code = EXIT_INPUT
        except Exception as e:                                 # a worker failure, out of memory: ppo_batch has put the masters back
            fail(odir, e, EXIT_FAIL)
            code = EXIT_FAIL
        if code != EXIT_OK:
            event("failed", request=req, exit=code)
            core.close()
            return code
        if not a.hold_workers:
            core.close()
        t_idle = time.time()


def smoke(a, mods):
    """Trainer-side smoke on real rows of an existing plan. Stage A: one optimizer step on the plan's first --stage-a-groups groups, export,
    checkpoint, damage read, workers closed, a new trainer from the checkpoint (state, optimizer arrays and exported adapter compared bit for bit).
    Stage B (only if the rates of stage A put it inside --budget-min): a new trainer at theta0, one optimizer step on ALL rows of the plan's first
    update. Nothing of it is a model: the adapters and checkpoints stay in the run directory and go with it."""
    TS, SS, MI = mods
    W = a.work
    t_start = time.time()
    os.makedirs(os.path.join(W, "out"), exist_ok=True)
    res = {"label": "rl-step-server-smoke", "args": vars(a), "time_start": now()}
    core = Core(a, mods)
    try:
        core.load_panel()
        plan, rows_all = load_rows(SS, a.plan)
        gseq = []
        for r in rows_all:
            if r.group_idx not in gseq:
                gseq.append(r.group_idx)
        # stage A's groups: the first --stage-a-groups groups with a non-zero advantage (a group of zeros would run no backward)
        mixed = [g for g in gseq if any(r.advantage != 0.0 for r in rows_all if r.group_idx == g)]
        ga = mixed[:a.stage_a_groups]
        rows_a = [r for r in rows_all if r.group_idx in ga]
        res["plan"] = {"dir": a.plan, "rows_update_1": len(rows_all), "row_tokens_update_1": int(sum(r.T for r in rows_all)), "groups": gseq, "mixed_groups": mixed,
                       "stage_a_groups": ga, "longest_row": int(max(r.T for r in rows_all))}
        core.open()
        res["panel_base"] = core.panel_base()
        A = core.update(rows_a, a.lr, os.path.join(W, "out", "a"), want_damage=True, zero_check=True)
        res["stage_a"] = A
        st0 = core.tr.opt.state_arrays()
        m0 = SS.state_sha(core.tr.opt.masters, core.nm)
        core.close()
        t = time.time()
        core.open(os.path.join(W, "ckpt", "v1"))
        st1 = core.tr.opt.state_arrays()
        ad2 = os.path.join(W, "out", "a-reload", "adapter")
        core.tr.export_adapter(ad2, dtype="BF16", meta={"label": LABEL, "version": "1"})
        d_re = core.damage()
        res["reload"] = {"masters_equal": SS.state_sha(core.tr.opt.masters, core.nm) == m0,
                         "optimizer_arrays_equal": sorted(st0) == sorted(st1) and all(np.array_equal(st0[k], st1[k]) for k in st0),
                         "optimizer_arrays": len(st0), "adapter_equal": sha256(os.path.join(ad2, "adapter_model.safetensors")) == A["adapter_sha256"],
                         "damage_equal": d_re["nats_per_token"] == A["damage"]["nats_per_token"], "damage_after_reload": d_re, "seconds": time.time() - t,
                         "worker_starts": core.worker_starts}
        log("reload from the checkpoint: " + json.dumps({k: v for k, v in res["reload"].items() if k.endswith("equal")}))
        core.close()
        jdump(os.path.join(W, "out", "smoke.json"), res)
        ok = all(res["reload"][k] for k in ("masters_equal", "optimizer_arrays_equal", "adapter_equal"))
        if not ok:
            raise RuntimeError("the reload from the checkpoint is not bit for bit")
        # stage B inside the budget?
        tok_b = res["plan"]["row_tokens_update_1"]
        back_b = int(sum(r.T for r in rows_all if r.advantage != 0.0))
        fwd_rate = res["panel_base"]["forward_tok_per_s_per_card"] * len(core.cards)
        cmp_rate = A["compute_tok_per_s_per_card"] * len(core.cards)
        old_rate = A["row_tokens"] / max(A["old_logprob_forward_s"], 1e-9)
        pred = tok_b / max(old_rate, 1e-9) + back_b / max(cmp_rate, 1e-9) + (tok_b - back_b) / max(fwd_rate, 1e-9) + 240
        left = a.budget_min * 60 - (time.time() - t_start)
        res["stage_b_prediction"] = {"row_tokens": tok_b, "tokens_with_a_backward": back_b, "predicted_s": pred, "budget_left_s": left}
        log(f"stage B prediction: {pred:.0f} s for {tok_b} row tokens ({back_b} with a backward); budget left {left:.0f} s")
        if pred > left:
            res["stage_b"] = {"skipped": "predicted time over the budget left"}
        else:
            a.work = os.path.join(W, "b")                     # stage B's own checkpoint directory (a fresh trainer at theta0, version 0)
            os.makedirs(a.work, exist_ok=True)
            core = Core(a, mods)
            core.load_panel()
            core.open()
            res["panel_base_b"] = core.panel_base()
            _, rows_b = load_rows(SS, a.plan)                 # fresh Row objects (stage A wrote old_lp into its rows' notes)
            res["stage_b"] = core.update(rows_b, a.lr, os.path.join(W, "out", "b"), want_damage=True, zero_check=False)
            core.close()
        res.update(time_end=now(), seconds=time.time() - t_start)
        jdump(os.path.join(W, "out", "smoke.json"), res)
        log("SMOKE_DONE " + json.dumps({"stage_a_tok_s_card": A["compute_tok_per_s_per_card"], "stage_b": {k: res["stage_b"].get(k) for k in
                                         ("rows", "row_tokens", "replay_checked_rows", "compute_tok_per_s_per_card", "step_tok_per_s_per_card", "ppo_batch_s", "host_peak_gib", "grad_norm", "skipped")}}, default=float))
        return EXIT_OK
    except TS.T8GateError as e:
        res["error"] = f"{type(e).__name__}: {e}"
        jdump(os.path.join(W, "out", "smoke.json"), res)
        fail(os.path.join(W, "out"), e, EXIT_GATE)
        return EXIT_GATE
    finally:
        core.close()


def gradprobe(a, mods):
    """Forward + backward only, nothing applied: the PPO gradient of the plan's first --stage-a-groups mixed groups at theta0, once with the exact
    backward (0, 0, 0) and once with the surrogate QD (48, 48, 0), on the same rows with the rows' own records as old log-probs (the replay and
    stale gates run in both). Reports per backward: norm in all and per layer block, tok/s, host peak; between them: cosine in all and per block,
    the share of coordinates with the same sign (Adam's first step is a sign step), the norm ratio. -> <work>/out/gradprobe.json"""
    import copy
    import torch
    from surogate.inkling_rl.step import StepConfig
    TS, SS, MI = mods
    W = a.work
    os.makedirs(os.path.join(W, "out"), exist_ok=True)
    blocks = {"before": range(0, 6), "inside": range(6, 18), "after": range(18, 24)}
    res, grads, nm = {"label": "rl-step-server-gradprobe", "args": vars(a), "time_start": now()}, {}, None
    for name, opt in (("exact", (0, 0, 0)), ("qd", (48, 48, 0))):
        b = copy.copy(a)
        b.pattern_grad_from, b.gate_grad_from, b.bank_grad_from = opt
        b.work = os.path.join(W, name)
        core = Core(b, mods)
        try:
            core.open()
            nm = core.nm
            _, rows_all = load_rows(SS, a.plan)
            gseq = []
            for r in rows_all:
                if r.group_idx not in gseq:
                    gseq.append(r.group_idx)
            mixed = [g for g in gseq if any(r.advantage != 0.0 for r in rows_all if r.group_idx == g)][:a.stage_a_groups]
            rows = [r for r in rows_all if r.group_idx in mixed]
            for r in rows:
                r.notes["old_lp"] = np.asarray(r.engine_lp[1:], dtype=np.float32)
            t0 = time.time()
            out = core.tr.compute(rows, 0, StepConfig(max_policy_lag=None), seeds="ppo", minibatch_index=0)
            m = out.metrics
            g = [x.detach().float().clone() for x in out.grads]
            per = {bk: sum(float(t.double().pow(2).sum()) for t, n in zip(g, nm) if int(n.split(".")[0]) in rng) ** 0.5 for bk, rng in blocks.items()}
            res[name] = {"backward_options": opt, "groups": mixed, "rows": len(rows), "row_tokens": int(sum(r.T for r in rows)), "wall_s": time.time() - t0,
                         "tok_per_s_per_card": m["tok_per_s_per_card"], "replay_checked_rows": m["replay_checked_rows"], "stale_checked_rows": m.get("stale_checked_rows"),
                         "norm": sum(float(t.double().pow(2).sum()) for t in g) ** 0.5, "norm_by_block": per, "host_peak_gib": [w.get("VmHWM_gib") for w in m.get("workers", [])],
                         "finite": bool(all(bool(torch.isfinite(t).all()) for t in g))}
            grads[name] = g
            log(f"gradprobe {name} {opt}: rows {len(rows)}, norm {res[name]['norm']:.6g}, by block {per}, {m['tok_per_s_per_card']:.0f} tok/s per card")
        finally:
            core.close()
    e, q = grads["exact"], grads["qd"]

    def cmp(sel):
        dot = sum(float((x.double() * y.double()).sum()) for x, y, n in zip(e, q, nm) if sel(n))
        ne = sum(float(x.double().pow(2).sum()) for x, n in zip(e, nm) if sel(n)) ** 0.5
        nq = sum(float(y.double().pow(2).sum()) for y, n in zip(q, nm) if sel(n)) ** 0.5
        both = sum(int(((x != 0) & (y != 0)).sum()) for x, y, n in zip(e, q, nm) if sel(n))
        same = sum(int(((x != 0) & (y != 0) & (torch.sign(x) == torch.sign(y))).sum()) for x, y, n in zip(e, q, nm) if sel(n))
        return {"cos": dot / max(ne * nq, 1e-300), "norm_exact": ne, "norm_qd": nq, "norm_ratio": ne / max(nq, 1e-300), "coordinates_nonzero_in_both": both,
                "same_sign_share": same / max(both, 1)}
    res["compare"] = {"all": cmp(lambda n: True), **{bk: cmp(lambda n, rng=rng: int(n.split(".")[0]) in rng) for bk, rng in blocks.items()}}
    res["time_end"] = now()
    jdump(os.path.join(W, "out", "gradprobe.json"), res)
    log("GRADPROBE_DONE " + json.dumps(res["compare"], default=float))
    return EXIT_OK


def real_mods():
    here = os.environ.get("RLS_PY")
    if here:
        sys.path.insert(0, os.path.join(here, "tests"))
        sys.path.insert(0, here)
    import t8_step as TS
    import t8_micro as MI
    import t8_stepsession as SS
    return TS, SS, MI


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("serve", "smoke", "gradprobe"))
    # the backward's constants (t8_step.T8Config; all 0 = the exact backward; 48 48 0 = the surrogate "QD": attention
    # pattern and router gate weights held constant at every visit)
    ap.add_argument("--pattern-grad-from", type=int, default=0); ap.add_argument("--gate-grad-from", type=int, default=0); ap.add_argument("--bank-grad-from", type=int, default=0)
    ap.add_argument("--work", required=True); ap.add_argument("--cards", required=True); ap.add_argument("--bin", required=True); ap.add_argument("--artifact", required=True)
    ap.add_argument("--panel", required=True); ap.add_argument("--plan", default=""); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--stack", default="C1"); ap.add_argument("--start-version", type=int, default=0); ap.add_argument("--resume-ckpt", default="")
    ap.add_argument("--hold-workers", type=int, default=0); ap.add_argument("--damage-every", type=int, default=5); ap.add_argument("--idle-min", type=float, default=240.0)
    ap.add_argument("--stage-a-groups", type=int, default=1); ap.add_argument("--budget-min", type=float, default=70.0)
    return ap


def main(argv=None, mods=None):
    a = parser().parse_args(argv)
    mods = mods or real_mods()
    return serve(a, mods) if a.mode == "serve" else (smoke(a, mods) if a.mode == "smoke" else gradprobe(a, mods))


if __name__ == "__main__":
    sys.exit(main())
