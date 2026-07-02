"""Build BBEH (BIG-Bench Extra Hard) -- 23 hard reasoning task types, exact-answer targets.
math_equal grader (numbers + normalized-string exact). Round-robin across task types for diversity;
gold-validate each target is gradeable. No worker calls -> no cost."""
import os, sys, json
os.environ.pop("HF_HUB_OFFLINE", None)
sys.path.insert(0, "ultra")
from collections import Counter
from datasets import load_dataset
from ultra.sources.hf import make_taskspec
from ultra.grading.verifiers import math_equal

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 120
PERTASK = 8
OUT = "director/manifests/fugu_clean_v1/grpo_pilot_train/reasoning_bbeh_taskspecs.jsonl"
SYS = "Solve the reasoning problem. Put the final answer in \\boxed{}."
ds = load_dataset("BBEH/bbeh", split="train")  # full, non-streaming for round-robin
fo = open(OUT, "w"); kept = badgold = 0; percat = Counter(); cats = Counter()
rows = list(ds)
import random; random.Random(9).shuffle(rows)
for r in rows:
    if kept >= TARGET:
        break
    task = r.get("task", "?"); q = r.get("input"); tgt = r.get("target")
    if not q or tgt is None or not str(tgt).strip() or len(str(tgt)) > 80:
        continue
    if percat[task] >= PERTASK:
        continue
    gold = str(tgt).strip()
    try:
        if math_equal(f"The answer is \\boxed{{{gold}}}.", gold) < 1.0:
            badgold += 1; continue
    except Exception:
        badgold += 1; continue
    spec = make_taskspec(task_id=f"bbeh__{task.replace(' ','_')}__{kept}", capability="reasoning",
        source_name="bbeh", source_version="v1", policy="train_allowed", harness="direct_qa",
        grader_type="math_equal", expected_answer=gold, prompt=str(q), system=SYS, group_id=f"bbeh_{task}",
        domain="reasoning", subdomain=task, tags=["reasoning", "bbeh", "hard", str(task)], url_or_ref="BBEH/bbeh")
    fo.write(spec.model_dump_json() + "\n"); kept += 1; percat[task] += 1; cats[task] += 1
fo.close()
print(f"DONE: kept {kept} BBEH tasks | ungradeable-gold {badgold} -> {OUT}", flush=True)
print("task-type spread:", dict(cats.most_common(25)), flush=True)
