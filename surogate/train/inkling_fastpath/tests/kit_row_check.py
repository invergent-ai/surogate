"""Lane T: the trainer on REAL kit rows in the kit's memory conditions (agent-EN 21:1x; PASS line in lanes/T/NOTES.md):
EngineBackend + MasterAdamW (fp32 masters and moments on the host) + agent-EN-RL's run_step with host gradient accumulators
(inkling/RL 6ec1442c), rows in the order given (longest last, so it runs on top of everything earlier rows left behind).

Reports per row: T, loss tokens, seconds and tok/s of its update (engine forward + backward + host accumulate), peak torch memory; and
for the step: the scoring pass, the optimizer step, engine_mismatch_kl, finite gradients. An out-of-memory error is a RESULT (the row is
named), not a crash. No adapter is saved: this run only measures memory and speed, and its update is discarded (LANE-RULES 22).
Usage: python kit_row_check.py --bin B --artifact A --rollouts rollouts.bin --T 16150,28542,32510 --out report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def log(msg):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    print(f"[T {datetime.now(ZoneInfo('Europe/Bucharest')).strftime('%H:%M:%S')}] {msg}", flush=True)


def run_rows(be, rows, opt, cfg, run_step, logf=log):
    """One kit step over `rows`; returns the report. be.logprobs_grad is wrapped only to time the rows."""
    starts, peaks = [], {}
    orig = be.logprobs_grad

    def timed(row, replay):
        starts.append((time.time(), row.T, int(row.loss_mask.sum())))
        return orig(row, replay)

    def lg(msg):
        m = re.match(r"update row (\d+) / \d+: .*peak torch memory ([0-9.]+) GiB", msg)
        if m:
            peaks[int(m.group(1)) - 1] = float(m.group(2))
        logf(msg)

    be.logprobs_grad = timed
    rep = {"rows": [], "oom": None}
    t0 = time.time()
    res = None
    try:
        res = run_step(be, rows, 0, cfg, log=lg)
    except torch.OutOfMemoryError as e:
        last = starts[-1] if starts else (None, None, None)
        rep["oom"] = {"row_T": last[1], "rows_done": max(len(starts) - 1, 0), "error": str(e).split(". ")[0][:200]}
        logf(f"OUT OF MEMORY on the row of T = {last[1]}")
    finally:
        be.logprobs_grad = orig
    t_end = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        t_end = time.time()
    done = len(starts) - (1 if rep["oom"] else 0)
    for k in range(done):
        t1 = starts[k + 1][0] if k + 1 < len(starts) else t_end
        dt = t1 - starts[k][0]
        rep["rows"].append({"T": starts[k][1], "loss_tokens": starts[k][2], "update_s": dt, "update_tok_per_s": starts[k][1] / dt,
                            "peak_torch_gib": peaks.get(k)})
    rep["score_pass_s"] = (starts[0][0] - t0) if starts else None
    rep["step_s"] = t_end - t0
    if res is not None:
        params = list(be.parameters())
        rep["finite"] = all(bool(torch.isfinite(p.grad).all()) for p in params if p.grad is not None)
        rep["engine_mismatch_kl"] = res.metrics.get("engine_mismatch_kl")
        rep["loss_tokens"] = res.loss_tokens
        rep["refused_long_rows"] = res.metrics.get("refused_long_rows")
        rep["offpolicy_dropped_samples"] = res.metrics.get("offpolicy_dropped_samples")
        t2 = time.time()
        rep["grad_norm"] = float(opt.step())
        rep["optimizer_s"] = time.time() - t2
    rep["all_rows_done"] = res is not None and len(rep["rows"]) == len(rows)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--T", default="16150,28542,32510", help="the rows, by their exact length, in this order")
    ap.add_argument("--max-row-tokens", type=int, default=32768)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from surogate.inkling_rl.records import load_rows
    from surogate.inkling_rl.shim import MasterAdamW
    from surogate.inkling_rl.step import StepConfig, run_step
    import backend_engine
    import ift_long as LG

    _, all_rows = load_rows(a.rollouts)
    rows = []
    for t in (int(x) for x in a.T.split(",")):
        hit = [r for r in all_rows if r.T == t]
        if not hit:
            raise SystemExit(f"no row of T = {t}")
        rows.append(hit[0])
    be = backend_engine.EngineBackend({"inkling_rl": {"engine": {"bin_dir": a.bin, "artifact": a.artifact, "device": 0,
                                                                "lora_seed": 0}}, "lora_rank": 16, "lora_alpha": 32})
    opt = MasterAdamW(be.parameters(), lr=5e-5, weight_decay=0.0, max_grad_norm=1.0)
    flags = {k: getattr(LG, k) for k in ("MOE_GM", "MOE_DET", "FAST_GEMM", "ATTN_ADAPT", "CHUNK", "MOE_FAST", "ATTN_TC", "MOE_ROWS",
                                         "MOE_PAD_ROWS")}
    log("flags " + json.dumps(flags))
    rep = run_rows(be, rows, opt, StepConfig(max_row_tokens=a.max_row_tokens), run_step)
    rep["flags"] = flags
    rep["engine_sha"] = be.engine_sha
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    for r in rep["rows"]:
        log(f"row T={r['T']}: update {r['update_s']:.1f} s = {r['update_tok_per_s']:.0f} tok/s, peak torch {r['peak_torch_gib']} GiB")
    log(f"KIT ROWS {'DONE' if rep['all_rows_done'] else 'NOT DONE'}; oom {rep['oom']}; mismatch {rep.get('engine_mismatch_kl')}; "
        f"finite {rep.get('finite')}; step {rep['step_s']:.1f} s")
    be.eng.close()
    return 0 if rep["all_rows_done"] else 4


if __name__ == "__main__":
    sys.exit(main())
