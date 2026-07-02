"""Locate ARC-AGI-2 on HF + inspect schema; show grid_exact grader's expected format."""
import os, sys, inspect
os.environ.pop("HF_HUB_OFFLINE", None)
sys.path.insert(0, "ultra")
from ultra.grading import verifiers as V

print("=== grid_exact grader ===", flush=True)
print(inspect.getsource(V.grid_exact)[:900], flush=True)

from datasets import load_dataset, get_dataset_config_names
for dsid in ["arc-agi-community/arc-agi-2", "arcprize/arc_agi_2", "arc-agi/ARC-AGI-2", "dataartist/arc-agi-2", "MohamedAshraf701/ARC-AGI-2"]:
    print(f"\n=== {dsid} ===", flush=True)
    try:
        cfgs = get_dataset_config_names(dsid)
        print("configs:", cfgs, flush=True)
        cfg = cfgs[0] if cfgs else None
        for s in ("train", "test", "training", "evaluation", "validation"):
            try:
                r = next(iter(load_dataset(dsid, cfg, split=s, streaming=True) if cfg else load_dataset(dsid, split=s, streaming=True)))
                print(f"split={s} FIELDS: {list(r.keys())}", flush=True)
                for k, v in list(r.items())[:8]:
                    print(f"  {k}: {repr(v)[:160]}", flush=True)
                break
            except Exception:
                continue
    except Exception as e:
        print("FAIL:", str(e)[:120], flush=True)
print("DONE", flush=True)
