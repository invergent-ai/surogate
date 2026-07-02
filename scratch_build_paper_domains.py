"""Fix MMLU-Pro (diverse categories, non-streaming shuffle) + build RLPR (detect answer format).
No worker calls."""
import os, sys, json
os.environ.pop("HF_HUB_OFFLINE", None)
sys.path.insert(0, "ultra")
from collections import Counter
from datasets import load_dataset
from ultra.sources.hf import make_taskspec
from ultra.grading.verifiers import mc_letter

D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
_MC = "Answer the question. Put the final answer letter in \\boxed{}."

# ---- MMLU-Pro: diverse ----
print("== MMLU-Pro (diverse) ==", flush=True)
ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")  # non-streaming full
ds = ds.shuffle(seed=7)
fo = open(f"{D}/reasoning_mmlu_pro_taskspecs.jsonl", "w")
kept = 0; cats = Counter(); percat = Counter()
for r in ds:
    if kept >= 120:
        break
    cat = r.get("category", "?")
    if percat[cat] >= 12:   # round-robin cap per category for diversity
        continue
    letter = str(r.get("answer", "")).strip()[:1].upper()
    if not letter:
        continue
    opts = r.get("options", [])
    body = r["question"] + "\n\n" + "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(opts))
    spec = make_taskspec(task_id=f"mmlupro__{r.get('question_id', kept)}", capability="reasoning",
        source_name="mmlu_pro", source_version="v1", policy="train_allowed", harness="direct_qa",
        grader_type="mc_letter", expected_answer=letter, prompt=body, system=_MC, group_id="mmlu_pro",
        domain="knowledge", subdomain=cat, tags=["mc", "mmlu_pro", "hard", str(cat)], url_or_ref="TIGER-Lab/MMLU-Pro")
    fo.write(spec.model_dump_json() + "\n"); kept += 1; percat[cat] += 1; cats[cat] += 1
fo.close()
print(f"  kept {kept} | categories: {dict(cats)}", flush=True)

# ---- RLPR: detect format ----
print("\n== RLPR (format detection) ==", flush=True)
for dsid in ["openbmb/RLPR-Train-Dataset", "vwxyzjn/rlpr"]:
    try:
        r = next(iter(load_dataset(dsid, split="train", streaming=True)))
        print(f"  {dsid} FIELDS: {list(r.keys())}", flush=True)
        for k, v in list(r.items())[:10]:
            print(f"    {k}: {repr(v)[:140]}", flush=True)
        break
    except Exception as e:
        print(f"  {dsid} FAIL: {str(e)[:120]}", flush=True)
print("DONE", flush=True)
