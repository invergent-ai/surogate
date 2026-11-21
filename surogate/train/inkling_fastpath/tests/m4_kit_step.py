"""Lane T M4: one kit GSPO step from ENGINE rollouts on the engine-op trainer (backend_engine.EngineBackend) with
agent-EN-RL's glue (surogate.inkling_rl: records.Row, step.run_step = the deployed loss sha 041f2c0c, shim.MasterAdamW,
adapter.export_adapter / the engine's fold).

  prep    <ad_dir>                       the step-0 adapter v0 = the trainer's own masters (seed 0, B = 0), BF16
  step    <bin> <artifact> <rollouts.json> <ad_dir> <out_dir>
          rows from the engine's responses (prompt_token_ids + token_ids, log-probs, routed experts, rule 20: ids only)
          -> step-start score (= the engine's forward) -> engine_mismatch_kl + per-position bit equality vs the engine's
          sampled log-probs + routing equality vs the engine's capture -> GSPO seeds -> LEVEL-0 backward -> fp32 AdamW
          on host masters (lr 5e-5, clip 1.0) -> export v1 (BF16) -> the trainer's scores under v1 (for compare)
  compare <out_dir> <engine_v1_scores.json>  the engine (fresh load of v1) vs the trainer under v1, bit for bit
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def log(msg):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    print(f"[T {datetime.now(ZoneInfo('Europe/Bucharest')).strftime('%H:%M:%S')}] {msg}", flush=True)


def masters_numpy(seed=0):
    import torch
    import ift_backward as B

    lo = B.make_lora("cpu", seed=seed, b_std=0.0)
    return [{k: d[k].detach().numpy().copy() for k in B.LORA_KEYS} for d in lo]


def prep(ad_dir):
    from surogate.inkling_rl import adapter as AD

    sha = AD.export_adapter(masters_numpy(0), os.path.join(ad_dir, "v0"), 16, 32, "BF16", metadata={"step": "0"})
    log(f"v0 exported sha {sha[:16]}")


def rows_from(rollouts):
    from surogate.inkling_rl.records import Row

    rows, meta = [], []
    rewards = []
    for r in rollouts:
        if "response" not in r:
            continue
        resp = r["response"]
        c = resp["choices"][0]
        P = list(resp["prompt_token_ids"])
        C = list(c["token_ids"])
        lps = [float(x["logprob"]) for x in c["logprobs"]["content"]]
        if len(lps) != len(C):
            log(f"row skipped: {len(lps)} log-probs for {len(C)} tokens")
            continue
        T = len(P) + len(C)
        ids = np.asarray(P + C, dtype=np.int64)
        mask = np.zeros(T, dtype=bool)
        mask[len(P):] = True
        elp = np.zeros(T, dtype=np.float64)
        elp[len(P):] = lps
        versions = np.full(T, -1, dtype=np.int32)
        versions[len(P):] = 0
        routed = np.full((T, 48, 8), -1, dtype=np.int16)
        re_ = c.get("routed_experts")
        if re_:
            n = int(re_["shape"][0])
            s = int(re_.get("start", 0))
            arr = np.frombuffer(base64.b64decode(re_["data"]), dtype="<i2").reshape(n, 48, 8)
            routed[s:s + n] = arr
        content = (c.get("message") or {}).get("content") or ""
        reward = 1.0 if r["job"]["answer"] in content.replace(",", "") else 0.0
        rewards.append(reward)
        rows.append(Row(ids=ids, loss_mask=mask, engine_lp=elp, advantage=0.0, versions=versions, teacher_lp=None,
                        routed=routed, trial_idx=len(rows), group_idx=r["job"]["group"]))
        meta.append({"group": r["job"]["group"], "P": len(P), "C": len(C), "finish": c.get("finish_reason"),
                     "reward": reward, "lora_versions": c.get("lora_versions")})
    # GSPO group advantages: reward - group mean; an all-equal batch gets a synthetic +-1 (plumbing only, flagged)
    groups = {}
    for i, m in enumerate(meta):
        groups.setdefault(m["group"], []).append(i)
    synthetic = False
    for g, idx in groups.items():
        mu = float(np.mean([meta[i]["reward"] for i in idx]))
        for i in idx:
            rows[i].advantage = meta[i]["reward"] - mu
    if all(abs(r.advantage) < 1e-12 for r in rows):
        synthetic = True
        for i, r in enumerate(rows):
            r.advantage = 1.0 if i % 2 == 0 else -1.0
    return rows, meta, synthetic


def step(bin_dir, artifact, rollouts_path, ad_dir, out_dir):
    import torch
    from surogate.inkling_rl import adapter as AD
    from surogate.inkling_rl.shim import MasterAdamW, masters_layers
    from surogate.inkling_rl.step import StepConfig, run_step
    from surogate.inkling_rl.records import replay_rows
    import backend_engine

    os.makedirs(out_dir, exist_ok=True)
    rows, meta, synthetic = rows_from(json.load(open(rollouts_path)))
    log(f"{len(rows)} rows; tokens P+C {[m['P'] + m['C'] for m in meta]}; rewards {[m['reward'] for m in meta]}; "
        f"synthetic advantages {synthetic}")
    t0 = time.time()
    be = backend_engine.EngineBackend({"inkling_rl": {"engine": {"bin_dir": bin_dir, "artifact": artifact, "device": 0,
                                                                "lora_seed": 0}}, "lora_rank": 16, "lora_alpha": 32})
    log(f"backend up {time.time() - t0:.1f}s, engine_sha {be.engine_sha[:16]}")
    opt = MasterAdamW(be.parameters(), lr=5e-5, weight_decay=0.0, max_grad_norm=1.0)
    rep = {"rows": meta, "synthetic_advantages": synthetic, "engine_sha": be.engine_sha}
    # per-position check of the step-start score vs the engine's sampled log-probs (before the step mutates anything)
    eq = ne = 0
    zero_eng_nonzero_mine = 0
    worst = 0.0
    per_row = []
    for r in rows:
        rp, _ = replay_rows(r)
        lp = be.score(r, rp)
        m = r.loss_mask
        a = lp[m].astype(np.float32)
        b = r.engine_lp[m].astype(np.float32)
        same = a.view(np.int32) == b.view(np.int32)
        eq += int(same.sum())
        ne += int((~same).sum())
        zero_eng_nonzero_mine += int(((b == 0.0) & (a != 0.0)).sum())
        if (~same).any():
            worst = max(worst, float(np.abs(a - b)[~same].max()))
        per_row.append({"equal": int(same.sum()), "differ": int((~same).sum()),
                        "first_differ": int(np.flatnonzero(m)[np.flatnonzero(~same)[0]]) if (~same).any() else None})
    rep["step_start_vs_engine"] = {"equal": eq, "differ": ne, "worst_abs": worst,
                                   "engine_zero_trainer_nonzero": zero_eng_nonzero_mine, "per_row": per_row}
    rep["routing_check"] = dict(be.r3_check)
    log(f"step-start score vs the engine's sampled log-probs: equal {eq} / {eq + ne} (worst {worst:.3g}; engine 0.0 vs "
        f"trainer != 0 on {zero_eng_nonzero_mine}); routing {be.r3_check}")
    # the kit step
    be.r3_check = {"rows": 0, "covered_positions": 0, "mismatched_positions": 0}
    t1 = time.time()
    res = run_step(be, rows, 0, StepConfig(), log=log)
    t_step = time.time() - t1
    gn = opt.step()
    rep["metrics"] = res.metrics
    rep["grad_norm"] = gn
    rep["step_seconds"] = t_step
    rep["loss_tokens"] = res.loss_tokens
    log(f"run_step {t_step:.1f}s, loss tokens {res.loss_tokens}, engine_mismatch_kl "
        f"{res.metrics.get('engine_mismatch_kl')}, grad_norm {gn:.4g}; metrics {json.dumps(res.metrics)}")
    sha1 = AD.export_adapter(masters_layers(be, opt), os.path.join(ad_dir, "v1"), 16, 32, "BF16", metadata={"step": "1"})
    rep["v1_sha"] = sha1
    # the trainer's scores under v1 (work params = bf16(masters) after the step)
    v1 = []
    for r in rows:
        v1.append(be.score(r, None).tolist())
    json.dump([{"ids": r.ids.tolist()} for r in rows], open(os.path.join(out_dir, "rows_ids.json"), "w"))
    json.dump(v1, open(os.path.join(out_dir, "trainer_v1.json"), "w"))
    json.dump(rep, open(os.path.join(out_dir, "m4_step.json"), "w"), indent=1, default=float)
    log(f"v1 exported sha {sha1[:16]}; trainer v1 scores saved")


def compare(out_dir, engine_scores):
    mine = json.load(open(os.path.join(out_dir, "trainer_v1.json")))
    eng = json.load(open(engine_scores))
    eq = ne = 0
    worst = 0.0
    for a, b in zip(mine, eng):
        a = np.asarray(a[1:], dtype=np.float32)
        b = np.asarray(b["lp"][1:], dtype=np.float32)
        same = a.view(np.int32) == b.view(np.int32)
        eq += int(same.sum())
        ne += int((~same).sum())
        if (~same).any():
            worst = max(worst, float(np.abs(a - b)[~same].max()))
    rep = json.load(open(os.path.join(out_dir, "m4_step.json")))
    rep["v1_engine_vs_trainer"] = {"equal": eq, "differ": ne, "worst_abs": worst}
    s0 = rep["step_start_vs_engine"]
    rep["checks"] = {
        "M4_step_start_bitwise": s0["differ"] == 0 and s0["equal"] > 0,
        "M4_engine_mismatch_kl_zero": rep["metrics"].get("engine_mismatch_kl") == 0.0,
        "M4_routing_equal": rep["routing_check"]["mismatched_positions"] == 0,
        "M4_update_nonzero": rep["grad_norm"] > 0,
        "M4_v1_engine_eq_trainer": ne == 0 and eq > 0,
    }
    json.dump(rep, open(os.path.join(out_dir, "m4_report.json"), "w"), indent=1, default=float)
    log(f"v1 engine vs trainer: equal {eq} / {eq + ne} (worst {worst:.3g})")
    log("CHECKS " + json.dumps(rep["checks"]))
    print("M4 PASS" if all(rep["checks"].values()) else "M4 FAIL", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep(sys.argv[2])
    elif cmd == "step":
        step(*sys.argv[2:7])
    else:
        compare(sys.argv[2], sys.argv[3])
