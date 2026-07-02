"""Inspect CodeContests + TACO-verified schemas (streaming, online) to build hard-code taskspecs
with code_exec_stdio grading. Need: problem text, stdin/stdout tests, difficulty, and a GOLD
solution (to validate the grader before trusting rewards)."""
import os, sys, json
os.environ.pop("HF_HUB_OFFLINE", None)
from datasets import load_dataset

def show(name, dsid, split, cfg=None):
    print(f"\n===== {name} ({dsid} / {split}{'/'+cfg if cfg else ''}) =====", flush=True)
    try:
        ds = load_dataset(dsid, cfg, split=split, streaming=True) if cfg else load_dataset(dsid, split=split, streaming=True)
        it = iter(ds)
        r = next(it)
        print("FIELDS:", list(r.keys()), flush=True)
        for k, v in r.items():
            s = repr(v)
            print(f"  {k}: {s[:160]}", flush=True)
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {str(e)[:300]}", flush=True)

show("CodeContests", "deepmind/code_contests", "train")
show("TACO-verified", "likaixin/TACO-verified", "train")
print("\nDONE", flush=True)
