"""Inspect candidate hard-reasoning datasets: fields, verifiable-answer presence, difficulty, size."""
import os, sys, json
os.environ.pop("HF_HUB_OFFLINE", None)
from datasets import load_dataset, get_dataset_config_names

DSIDS = [
    "Menlo/Maze-Reasoning-very-hard-v0.1",
    "reasoning-degeneration-dev/sdc-responses-hard-v1",
    "TongZheng1999/Reasoning-Gym-Hard",
]
for dsid in DSIDS:
    print(f"\n===== {dsid} =====", flush=True)
    try:
        cfgs = get_dataset_config_names(dsid)
        print("configs:", cfgs, flush=True)
    except Exception as e:
        cfgs = [None]; print("configs: (default)", str(e)[:80], flush=True)
    try:
        cfg = cfgs[0] if cfgs and cfgs[0] else None
        # try common splits
        split = None
        for s in ("train", "test", "validation"):
            try:
                ds = load_dataset(dsid, cfg, split=s, streaming=True) if cfg else load_dataset(dsid, split=s, streaming=True)
                split = s; break
            except Exception:
                continue
        if split is None:
            print("  no train/test/validation split found", flush=True); continue
        r = next(iter(ds))
        print(f"split={split} FIELDS:", list(r.keys()), flush=True)
        for k, v in r.items():
            print(f"  {k}: {repr(v)[:150]}", flush=True)
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}", flush=True)
print("\nDONE", flush=True)
