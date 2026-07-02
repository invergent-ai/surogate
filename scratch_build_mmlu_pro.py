"""Build the paper's MMLU domain as MMLU-Pro (harder, 10-choice), graded by mc_letter (clean:
the answer IS a letter -> no false-negative hazard). Paper trains on MMLU, so MC is IN for training
(the guessing confound is a measurement artifact, not a training problem). No worker calls -> no cost."""
import os, sys, json
os.environ.pop("HF_HUB_OFFLINE", None)
sys.path.insert(0, "ultra")
from collections import Counter
from ultra.sources.direct import MMLUProAdapter
from ultra.grading.verifiers import mc_letter

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 120
OUT = "director/manifests/fugu_clean_v1/grpo_pilot_train/reasoning_mmlu_pro_taskspecs.jsonl"
ad = MMLUProAdapter()
fo = open(OUT, "w")
kept = badgold = 0
bycat = Counter()
for s in ad.materialize_all(limit=TARGET * 3, shuffle=True, seed=7):
    if kept >= TARGET:
        break
    ea = s.grader.expected_answer
    # validate: gold letter is gradeable by mc_letter
    try:
        if mc_letter(f"The answer is \\boxed{{{ea}}}.", ea) < 1.0:
            badgold += 1; continue
    except Exception:
        badgold += 1; continue
    fo.write(s.model_dump_json() + "\n"); fo.flush(); kept += 1
    bycat[(s.metadata.subdomain or "?")] += 1
fo.close()
print(f"DONE: kept {kept} MMLU-Pro tasks | gold-ungradeable {badgold} -> {OUT}", flush=True)
print("category spread:", dict(bycat.most_common(12)), flush=True)
