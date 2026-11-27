"""Replay of one applied RL update's GRADIENT with the stack named in T8_STACK: the step session's own inputs (plan, the session's old log-probs), the
session's trainer settings (t8_micro.trainer_cfg, no sketches), compute(seeds="ppo", minibatch_index=0) on the update's rows; the result is compared
with the gradient the session stored (updates/<n>/grad.pt). Nothing is applied or exported."""
import argparse, json, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True); ap.add_argument("--old-lp", required=True); ap.add_argument("--grad", required=True)
    ap.add_argument("--cards", required=True); ap.add_argument("--run-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--bin", required=True); ap.add_argument("--artifact", required=True)
    ap.add_argument("--pattern-grad-from", type=int, default=48); ap.add_argument("--gate-grad-from", type=int, default=48); ap.add_argument("--bank-grad-from", type=int, default=0)
    a = ap.parse_args()
    from surogate.inkling_rl.step import StepConfig
    import t8_step as TS
    import t8_micro as MI
    import t8_stepsession as SS
    plan = SS.load_step_plan(a.plan)
    rows = plan.rows
    with np.load(a.old_lp, allow_pickle=False) as z:
        for r in rows:
            r.notes["old_lp"] = np.asarray(z[f"old_{r.notes['plan_row']}"], dtype=np.float32)
    sub = [rows[i] for i in plan.updates[0]]
    cards = tuple(int(x) for x in a.cards.split(","))
    cfg = MI.trainer_cfg(a, cards)
    print("stack", cfg.stack, "rows", len(sub), "tokens", sum(r.T for r in sub), "backward", MI.backward_options(a), "device_sum", cfg.device_sum, flush=True)
    tr = TS.T8Trainer(cfg)
    try:
        t0 = time.time()
        out = tr.compute(sub, plan.start_version, StepConfig(max_policy_lag=None), seeds="ppo", minibatch_index=0)
        wall = time.time() - t0
    finally:
        tr.close()
    ref = list(torch.load(a.grad).values())
    assert len(ref) == len(out.grads), (len(ref), len(out.grads))
    per = []
    for i, (x, y) in enumerate(zip(ref, out.grads)):
        x, y = x.double().flatten(), y.double().flatten()
        nx = float(x.norm())
        per.append({"tensor": i, "norm": nx, "new_norm": float(y.norm()), "cos": float(torch.dot(x, y) / (x.norm() * y.norm()).clamp_min(1e-300)) if nx > 0 else None,
                    "rel": float((y - x).norm() / x.norm()) if nx > 0 else None, "bit_equal": bool(torch.equal(x, y))})
    live = [o for o in per if o["norm"] > 0]
    gn = lambda g: float(torch.sqrt(sum(t.double().pow(2).sum() for t in g)))
    m = out.metrics
    rep = {"stack": cfg.stack, "rows": len(sub), "wall_s": wall, "tok_per_s_per_card": m["tok_per_s_per_card"], "row_tokens": m["row_tokens"],
           "replay_checked_rows": m["replay_checked_rows"], "stale_checked_rows": m.get("stale_checked_rows"), "zero_seed_rows": m.get("zero_seed_rows"),
           "oom_retries": sum(r["oom_retries"] for r in out.rows), "host_peak_gib": [w["VmHWM_gib"] for w in m["workers"]],
           "grad_norm_stored": gn(ref), "grad_norm_new": gn(out.grads), "n": len(per), "n_live": len(live), "n_bit_equal": sum(o["bit_equal"] for o in per),
           "dead_but_nonzero_now": sum(1 for o in per if o["norm"] == 0 and o["new_norm"] > 0),
           "cos_min": min(o["cos"] for o in live), "rel_max": max(o["rel"] for o in live), "rel_median": sorted(o["rel"] for o in live)[len(live) // 2], "per_tensor": per}
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    print("DONE", json.dumps({k: v for k, v in rep.items() if k != "per_tensor"}, default=float), flush=True)


if __name__ == "__main__":
    main()
