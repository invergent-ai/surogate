"""Build the paper's RLPR domain from openbmb/RLPR-Train-Dataset (WebInstruct-verified).
It ships rule-style ground_truth short answers -> grade with math_equal (no judge needed).
Extract the question from the RLPR chat template; gold-validate the answer is parseable. No worker calls."""
import os, sys, json, re
os.environ.pop("HF_HUB_OFFLINE", None)
sys.path.insert(0, "ultra")
from collections import Counter
from datasets import load_dataset
from ultra.sources.hf import make_taskspec
from ultra.grading.verifiers import math_equal

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 120
OUT = "director/manifests/fugu_clean_v1/grpo_pilot_train/reasoning_rlpr_taskspecs.jsonl"
SYS = "Solve the problem. Put the final answer in \\boxed{}."

def extract_q(prompt):
    # prompt is a list of chat msgs; grab user content, strip the RLPR wrapper
    if isinstance(prompt, list):
        txt = " ".join(m.get("content", "") for m in prompt if isinstance(m, dict))
    else:
        txt = str(prompt)
    m = re.search(r"User:\s*(.*?)\s*Assistant:", txt, re.S)
    q = m.group(1).strip() if m else txt
    # drop any leading "think first" instruction remnants
    return q

ds = load_dataset("openbmb/RLPR-Train-Dataset", split="train", streaming=True)
fo = open(OUT, "w")
kept = scanned = badgold = 0; ab = Counter()
for r in ds:
    if kept >= TARGET:
        break
    rm = r.get("reward_model") or {}
    gt = rm.get("ground_truth")
    q = extract_q(r.get("prompt"))
    if not gt or not q or len(str(gt).strip()) > 60:  # short verifiable answers only
        continue
    scanned += 1
    gold = str(gt).strip()
    try:
        if math_equal(f"The answer is \\boxed{{{gold}}}.", gold) < 1.0:
            badgold += 1; continue
    except Exception:
        badgold += 1; continue
    spec = make_taskspec(task_id=f"rlpr__{r.get('__index_level_0__', scanned)}", capability="reasoning",
        source_name="rlpr", source_version="v1", policy="train_allowed", harness="direct_qa",
        grader_type="math_equal", expected_answer=gold, prompt=q, system=SYS, group_id="rlpr",
        domain="reasoning", subdomain=str(r.get("ability") or ""), tags=["reasoning", "rlpr", "webinstruct", str(r.get("ability") or "")],
        url_or_ref="openbmb/RLPR-Train-Dataset")
    fo.write(spec.model_dump_json() + "\n"); kept += 1; ab[str(r.get("ability") or "?")] += 1
fo.close()
print(f"DONE: kept {kept} RLPR tasks | scanned {scanned} | ungradeable {badgold} -> {OUT}", flush=True)
print("ability spread:", dict(ab.most_common(12)), flush=True)
