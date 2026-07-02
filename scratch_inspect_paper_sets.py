"""Locate + inspect the paper's missing datasets: MMLU-Pro (MC) and RLPR (real-world reasoning).
Check fields, verifiable-answer, difficulty, size."""
import os, sys
os.environ.pop("HF_HUB_OFFLINE", None)
from datasets import load_dataset, get_dataset_config_names

CANDS = [
    ("MMLU-Pro", "TIGER-Lab/MMLU-Pro", None),
    ("RLPR (openbmb)", "openbmb/RLPR-Train-Dataset", None),
    ("RLPR (alt)", "vwxyzjn/rlpr", None),
]
for name, dsid, cfg in CANDS:
    print(f"\n===== {name} ({dsid}) =====", flush=True)
    try:
        cfgs = get_dataset_config_names(dsid)
        print("configs:", cfgs, flush=True)
        cfg = cfg or (cfgs[0] if cfgs else None)
    except Exception as e:
        print("configs FAIL:", str(e)[:120], flush=True); continue
    got = False
    for s in ("train", "test", "validation"):
        try:
            ds = load_dataset(dsid, cfg, split=s, streaming=True) if cfg else load_dataset(dsid, split=s, streaming=True)
            r = next(iter(ds)); got = True
            print(f"split={s} FIELDS:", list(r.keys()), flush=True)
            for k, v in list(r.items())[:12]:
                print(f"  {k}: {repr(v)[:130]}", flush=True)
            break
        except Exception:
            continue
    if not got:
        print("  no readable split", flush=True)
print("\nDONE", flush=True)
