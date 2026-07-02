"""Locate BBEH (BIG-Bench Extra Hard) on HF + inspect schema (question/target, verifiable)."""
import os, sys
os.environ.pop("HF_HUB_OFFLINE", None)
from datasets import load_dataset, get_dataset_config_names
for dsid in ["BBEH/bbeh", "google/bbeh", "lukaemon/bbeh", "hkust-nlp/BBEH", "MiniLLM/BBEH", "SaylorTwift/bbeh", "bbeh"]:
    print(f"\n=== {dsid} ===", flush=True)
    try:
        cfgs = get_dataset_config_names(dsid)
        print("configs:", cfgs[:8], "..." if len(cfgs) > 8 else "", flush=True)
        cfg = cfgs[0] if cfgs else None
        for s in ("train", "test"):
            try:
                r = next(iter(load_dataset(dsid, cfg, split=s, streaming=True) if cfg else load_dataset(dsid, split=s, streaming=True)))
                print(f"split={s} FIELDS: {list(r.keys())}", flush=True)
                for k, v in list(r.items())[:8]:
                    print(f"  {k}: {repr(v)[:160]}", flush=True)
                break
            except Exception:
                continue
    except Exception as e:
        print("FAIL:", str(e)[:100], flush=True)
print("DONE", flush=True)
