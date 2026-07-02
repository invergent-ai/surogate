"""Prepare a HARDER code train set in case LCB-V1 is too easy (GPT-5.5-minimal dominates).
TACO-verified HARD/VERY_HARD: competitive-programming hard, train-allowed, local. stdin->stdout,
code_exec_stdio grader (same as LCB). No worker calls here -- pure dataset processing."""
import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, "ultra")
from ultra.sources.direct import OmniMathAdapter  # noqa (ensure package import path)
from ultra.sources.code import TACOAdapter

D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
out = f"{D}/code_hard_taco_taskspecs.jsonl"
ad = TACOAdapter(difficulties=("HARD", "VERY_HARD"), max_tests=8)
n = 0
with open(out, "w") as fo:
    try:
        for s in ad.materialize_all(limit=80, shuffle=True, seed=5):
            fo.write(s.model_dump_json() + "\n"); n += 1
            if n >= 80:
                break
    except Exception as e:
        print(f"TACO load FAIL: {type(e).__name__}: {str(e)[:200]}", flush=True)
print(f"TACO-HARD materialized: {n} -> {out}", flush=True)
print("DONE", flush=True)
