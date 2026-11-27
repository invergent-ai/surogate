"""One gradient compute on real long rows with the stack named in T8_STACK, to compare two stacks (run it once per stack, then compare the saved
gradients per tensor: cosine and relative error). K = 48, plain head, seeds "weights"; expert B keys, plus the attention adapters with --attn-rank,
plus the RL step's shape with --qd (expert A and B, every visit's attention pattern and router gates constant in the backward). --perturb puts seeded
noise into the B tensors so that the A gradients are live. Rows: an .npz with ids_<i>, mask_<i>, lp_<i> and group_<name> index lists.
  T8_STACK=C1 two_stack_grad.py --rows rows.npz --group speed --bin BIN --artifact MODEL.sinfer --cards 0,1 --run-dir DIR --out out.json"""
import argparse, json, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True); ap.add_argument("--group", required=True); ap.add_argument("--take", type=int, default=0)
    ap.add_argument("--attn-rank", type=int, default=0); ap.add_argument("--perturb", type=float, default=0.0); ap.add_argument("--qd", type=int, default=0); ap.add_argument("--bin", required=True); ap.add_argument("--artifact", required=True); ap.add_argument("--cards", required=True); ap.add_argument("--run-dir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from surogate.inkling_rl.step import StepConfig
    from surogate.inkling_rl.records import Row
    import t8_step as TS
    z = np.load(a.rows)
    idx = [int(x) for x in z["group_" + a.group]]
    if a.take:
        idx = sorted(idx, key=lambda i: -z[f"ids_{i}"].shape[0])[:a.take]
    rows = [Row(ids=z[f"ids_{i}"].astype(np.int64), loss_mask=z[f"mask_{i}"].astype(bool), engine_lp=z[f"lp_{i}"].astype(np.float64), advantage=0.0, versions=None,
                teacher_lp=None, routed=None, notes={"weight": 1.0}) for i in idx]
    cards = tuple(int(x) for x in a.cards.split(","))
    keys = ("B13", "B2") + (tuple(TS.ATTN_KEYS) if a.attn_rank else ())
    if a.qd:                                                  # the RL step's shape: expert A and B + attention, pattern and gates constant in every visit
        keys = tuple(TS.LORA_KEYS) + tuple(TS.ATTN_KEYS)
    cfg = TS.T8Config(bin_dir=a.bin, artifact=a.artifact, cards=cards, K=48, head="plain", max_row_tokens=58010, segment_keys=(),
                      train_keys=keys, attn_rank=a.attn_rank, allow_lagged=True, run_dir=a.run_dir,
                      pattern_grad_from=48 if a.qd else 0, gate_grad_from=48 if a.qd else 0)
    print("stack", cfg.stack, "attn_rank", cfg.attn_rank, "train keys", len(keys), "rows", [int(r.ids.shape[0]) for r in rows], flush=True)
    tr = TS.T8Trainer(cfg)
    try:
        if a.perturb:                                         # a non-zero adapter, so that the A-key gradients are live: seeded noise in every attention B
            g = torch.Generator().manual_seed(1234)
            m = tr.masters()
            for d in m:
                for k in (tuple(TS.ATTN_KEYS) + (("B13", "B2") if a.qd else ())):
                    if k.endswith("B") or k in ("B13", "B2"):
                        d[k] = torch.randn(d[k].shape, generator=g) * a.perturb
            tr.set_masters([{k: np.array(v, np.float32) for k, v in d.items()} for d in m])
            for r in rows:                                    # sampled by another adapter: the replay gate does not apply
                r.versions = np.full(r.ids.shape[0], 5, dtype=np.int64)
        t0 = time.time()
        o = tr.compute(rows, 0, StepConfig(max_policy_lag=None), seeds="weights")
        wall = time.time() - t0
    finally:
        tr.close()
    rep = {"stack": cfg.stack, "rows": [int(r.ids.shape[0]) for r in rows], "wall_s": wall,
           "tok_per_s_per_card": o.metrics["tok_per_s_per_card"], "row_tokens": o.metrics["row_tokens"],
           "host_peak_gib": [w["VmHWM_gib"] for w in o.metrics["workers"]], "oom_retries": sum(r["oom_retries"] for r in o.rows),
           "workers": o.metrics["workers"], "finite": bool(all(bool(torch.isfinite(g).all()) for g in o.grads)), "n_grads": len(o.grads)}
    torch.save([g.cpu() for g in o.grads], a.out + ".grads.pt")
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    print("DONE", json.dumps({k: rep[k] for k in ("stack", "wall_s", "tok_per_s_per_card", "oom_retries", "finite", "host_peak_gib")}, default=float), flush=True)


if __name__ == "__main__":
    main()
