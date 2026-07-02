"""Build a HARD, DIVERSE, VERIFIABLE reasoning set from Reasoning-Gym-Hard (exact ground_truth,
many task types). Grade by math_equal (numbers + normalized-string exact match). Keep only tasks
whose gold ground_truth is gradeable by the (pre-audited) grader. No worker calls -> no cost."""
import os, sys, json
os.environ.pop("HF_HUB_OFFLINE", None)
sys.path.insert(0, "ultra")
from collections import Counter
from datasets import load_dataset
from ultra.sources.hf import make_taskspec
from ultra.grading.verifiers import math_equal

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 120
OUT = "director/manifests/fugu_clean_v1/grpo_pilot_train/reasoning_hard_rgym_taskspecs.jsonl"
SYS = "Solve the reasoning problem. Put the final answer in \\boxed{}."
ds = load_dataset("TongZheng1999/Reasoning-Gym-Hard", split="train", streaming=True)
fo = open(OUT, "w")
kept = scanned = badgold = 0
bytype = Counter()
for r in ds:
    if kept >= TARGET:
        break
    q = r.get("question"); gt = r.get("ground_truth"); tn = r.get("task_name") or "rgym"
    if not q or gt is None or not str(gt).strip():
        continue
    scanned += 1
    gold = str(gt).strip()
    try:
        if math_equal(f"The answer is \\boxed{{{gold}}}.", gold) < 1.0:
            badgold += 1; continue
    except Exception:
        badgold += 1; continue
    spec = make_taskspec(
        task_id=f"rgym__{tn}__{scanned}", capability="reasoning", source_name="reasoning_gym", source_version="v1",
        policy="train_allowed", harness="direct_qa", grader_type="math_equal", expected_answer=gold,
        prompt=str(q), system=SYS, group_id=f"rgym_{tn}", domain="reasoning",
        tags=["reasoning", "reasoning_gym", "hard", str(tn)], url_or_ref="TongZheng1999/Reasoning-Gym-Hard")
    fo.write(spec.model_dump_json() + "\n"); fo.flush(); kept += 1; bytype[tn] += 1
    if kept % 30 == 0:
        print(f"  kept {kept} (scanned {scanned}, ungradeable dropped {badgold})", flush=True)
fo.close()
print(f"DONE: kept {kept} rgym tasks | scanned {scanned} | ungradeable-gold drop {badgold} ({badgold/max(scanned,1):.0%})", flush=True)
print("task-type spread:", dict(bytype.most_common(15)), flush=True)
print("->", OUT, flush=True)
