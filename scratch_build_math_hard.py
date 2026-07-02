"""Build a HARD, VERIFIABLE reasoning set from Omni-MATH (olympiad, exact-answer, math_equal grader).
Verifiable math (not MC) is the honest reasoning test -- exact answers, no guessing confound.
Validate: keep only tasks whose GOLD answer is parseable/gradeable by the (pre-audited) math grader.
No worker calls -> no cost. Whether to TRAIN on it is a separate headroom decision."""
import os, sys, json
os.environ.pop("HF_HUB_OFFLINE", None)
sys.path.insert(0, "ultra")
from datasets import load_dataset
from ultra.sources.hf import make_taskspec
from ultra.grading.verifiers import math_equal

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 120
OUT = "director/manifests/fugu_clean_v1/grpo_pilot_train/reasoning_hard_omnimath_taskspecs.jsonl"
SYS = "Solve the problem. Put the final answer in \\boxed{}."
ds = load_dataset("KbsdJames/Omni-MATH", split="test", streaming=True)
fo = open(OUT, "w")
kept = scanned = badgold = 0
for r in ds:
    if kept >= TARGET:
        break
    q = r.get("problem"); a = r.get("answer")
    if not q or a is None or not str(a).strip():
        continue
    scanned += 1
    gold = str(a).strip()
    # validate: the grader can parse+match the gold answer against itself
    try:
        if math_equal(f"The answer is \\boxed{{{gold}}}.", gold) < 1.0:
            badgold += 1; continue
    except Exception:
        badgold += 1; continue
    spec = make_taskspec(
        task_id=f"omnimath__{scanned}", capability="math", source_name="omni_math", source_version="v1",
        policy="train_allowed", harness="direct_qa", grader_type="math_equal", expected_answer=gold,
        prompt=str(q), system=SYS, group_id="omni_math", domain="math",
        tags=["math", "omni_math", "olympiad", "hard"], url_or_ref="KbsdJames/Omni-MATH")
    fo.write(spec.model_dump_json() + "\n"); fo.flush(); kept += 1
    if kept % 30 == 0:
        print(f"  kept {kept} (scanned {scanned}, ungradeable-gold dropped {badgold})", flush=True)
fo.close()
print(f"DONE: kept {kept} HARD Omni-MATH tasks | scanned {scanned} | gold-ungradeable drop {badgold} ({badgold/max(scanned,1):.0%}) -> {OUT}", flush=True)
