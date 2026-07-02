"""Materialize the PAPER's hard single-turn TRAIN datasets into one taskspecs file.
Math: NuminaMath + Omni-Math | Knowledge: MMLU-Pro (via HF adapters, parquet).
Code: LiveCodeBench V1 read DIRECTLY from the locally-downloaded lite test.jsonl
      (medium+hard, stdin-type public tests -> code_exec_stdio). V6 reserved for eval.
"""
import json, sys
sys.path.insert(0, "ultra")
from collections import Counter
from ultra.sources.hf import make_taskspec
from ultra.sources.code import _STDIO_SYS
from ultra.policy import SOURCE_POLICY
from ultra.sources.direct import NuminaMathAdapter, MMLUProAdapter, OmniMathAdapter

D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
LITE = "/var/lib/mesh/flavius/huggingface/hub/datasets--livecodebench--code_generation_lite/snapshots/0fe84c3912ea0c4d4a78037083943e8f0c4dd505/test.jsonl"
out = []

# --- math + knowledge via HF adapters (parquet, datasets 4.6.1 OK, HF-cached -> fast) ---
for tag, ad, lim in [("math", NuminaMathAdapter(), 150),
                     ("knowledge", MMLUProAdapter(), 150),
                     ("math_hard", OmniMathAdapter(), 80)]:
    n = 0
    try:
        for s in ad.materialize_all(limit=lim, shuffle=True, seed=7):
            out.append(s); n += 1
    except Exception as e:
        print(f"  {tag:10} {ad.source_name:14} FAIL: {str(e)[:90]}", flush=True); continue
    print(f"  {tag:10} {ad.source_name:14} -> {n}", flush=True)

# --- LiveCodeBench V1 from local lite file: medium+hard, stdin public tests ---
nl = 0; diffc = Counter()
with open(LITE) as f:
    for i, line in enumerate(f):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("difficulty") not in ("medium", "hard"):
            continue
        try:
            pub = json.loads(r["public_test_cases"]) if isinstance(r["public_test_cases"], str) else r["public_test_cases"]
        except Exception:
            continue
        tests = [{"input": c.get("input", ""), "output": c.get("output", "")}
                 for c in pub if c.get("testtype") == "stdin"]
        if not tests or not r.get("question_content"):
            continue
        out.append(make_taskspec(
            task_id=f"livecodebench__{r.get('question_id', i)}",
            capability="unit_code", source_name="livecodebench", source_version="v1",
            policy=SOURCE_POLICY["livecodebench_old"], harness="code_exec", grader_type="code_exec_stdio",
            expected_answer={"tests": tests, "timeout": 10}, prompt=r["question_content"], system=_STDIO_SYS,
            group_id="livecodebench", domain="code", tags=["code", "lcb", r["difficulty"]],
            url_or_ref="livecodebench/code_generation_lite"))
        nl += 1; diffc[r["difficulty"]] += 1
print(f"  code       livecodebench  -> {nl} {dict(diffc)} (stdin, medium+hard)", flush=True)

path = f"{D}/paper_train_taskspecs.jsonl"
with open(path, "w") as fo:
    for s in out:
        fo.write(s.model_dump_json() + "\n")
print(f"\nTOTAL {len(out)} paper-dataset train tasks -> {path}", flush=True)
print("harness:", dict(Counter(s.environment.harness for s in out)), flush=True)
print("capability:", dict(Counter(s.capability for s in out)), flush=True)
print("DONE", flush=True)
