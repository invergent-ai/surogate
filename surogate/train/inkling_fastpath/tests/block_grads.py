"""The gradient of each named update (block of task groups) of a step plan, EVERY block at the plan's start weights (the zero adapter), with the step
session's trainer settings (t8_micro.trainer_cfg: K 48, plain head, rank-64 attention adapters + expert A and B, the backward options given) on the
stack named in T8_STACK, per-attempt sketches kept on request. Nothing is applied or exported. It answers whether a batch of tasks carries a
direction other tasks share: split the tasks by seed into blocks, compute each block's gradient here, then compare them with block_cos.py.
Per block: <out>/grad-b<j>.pt (the step session's grad.pt form), <out>/block-b<j>.json, <out>/sketches-b<j>/. The old log-probs are the rows' own
records (on-policy rows of the start weights): the replay and stale gates then demand the trainer's forward to equal them bit for bit on every
loss token.
  T8_STACK=C1 block_grads.py --plan PLAN_DIR --blocks 0,1 --cards 0,1,2,3,4,5 --bin BIN --artifact MODEL.sinfer --run-dir DIR --out OUT_DIR"""
import argparse, json, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True); ap.add_argument("--blocks", required=True); ap.add_argument("--cards", required=True)
    ap.add_argument("--run-dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--bin", required=True); ap.add_argument("--artifact", required=True)
    ap.add_argument("--sketches", type=int, default=1); ap.add_argument("--pattern-grad-from", type=int, default=48); ap.add_argument("--gate-grad-from", type=int, default=48); ap.add_argument("--bank-grad-from", type=int, default=0)
    a = ap.parse_args()
    from surogate.inkling_rl.step import StepConfig
    import t8_step as TS
    import t8_micro as MI
    import t8_stepsession as SS
    plan = SS.load_step_plan(a.plan)
    blocks = [int(x) for x in a.blocks.split(",")]
    cards = tuple(int(x) for x in a.cards.split(","))
    os.makedirs(a.out, exist_ok=True)
    for r in plan.rows:
        r.notes["old_lp"] = np.asarray(r.engine_lp[1:], dtype=np.float32)
    for j in blocks:
        sub = [plan.rows[i] for i in plan.updates[j]]
        sk = os.path.join(a.out, f"sketches-b{j}")
        cfg = MI.trainer_cfg(a, cards, sketch_dir=sk) if a.sketches else MI.trainer_cfg(a, cards)
        print(f"block {j}: stack {cfg.stack}, rows {len(sub)}, tokens {sum(r.T for r in sub)}, groups {sorted({int(r.group_idx) for r in sub})}, "
              f"backward {MI.backward_options(a)}, lean_host {cfg.lean_host}", flush=True)
        tr = TS.T8Trainer(cfg)
        try:
            t0 = time.time()
            out = tr.compute(sub, plan.start_version, StepConfig(max_policy_lag=None), seeds="ppo", minibatch_index=0)
            wall = time.time() - t0
        finally:
            tr.close()
        names = MI.names(len(tr.shapes), tuple(tr.shapes[0]))   # the step session's names of grad.pt
        assert len(names) == len(out.grads), (len(names), len(out.grads))
        torch.save({n: g.detach() for n, g in zip(names, out.grads)}, os.path.join(a.out, f"grad-b{j}.pt.tmp"))
        os.replace(os.path.join(a.out, f"grad-b{j}.pt.tmp"), os.path.join(a.out, f"grad-b{j}.pt"))
        m = out.metrics
        rep = {"block": j, "stack": cfg.stack, "rows": len(sub), "groups": sorted({int(r.group_idx) for r in sub}), "wall_s": wall,
               "tok_per_s_per_card": m["tok_per_s_per_card"], "row_tokens": m["row_tokens"], "N": float(m.get("total_tokens", 0)),
               "replay_checked_rows": m["replay_checked_rows"], "replay_skipped_rows": m["replay_skipped_rows"], "stale_checked_rows": m.get("stale_checked_rows"),
               "zero_seed_rows": m.get("zero_seed_rows"), "oom_retries": sum(r["oom_retries"] for r in out.rows),
               "host_peak_gib": [w["VmHWM_gib"] for w in m["workers"]], "finite": bool(all(bool(torch.isfinite(g).all()) for g in out.grads)),
               "grad_norm": float(torch.sqrt(sum(g.double().pow(2).sum() for g in out.grads))), "ppo_ratio_max": m.get("ppo_ratio_max"),
               "sketch_files": len(m.get("sketch_files", [])), "backward_options": MI.backward_options(a)}
        json.dump(rep, open(os.path.join(a.out, f"block-b{j}.json"), "w"), indent=1, default=float)
        print("BLOCK_DONE", json.dumps(rep, default=float), flush=True)
        del out, tr


if __name__ == "__main__":
    main()
