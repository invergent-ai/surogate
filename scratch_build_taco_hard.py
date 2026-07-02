"""Build a HARD, diverse code-gen training set from TACO-verified (HARD/VERY_HARD, stdin/stdout),
graded by code_exec_stdio like LCB. CRITICAL: validate each task with its GOLD Python-3 solution
through the SAME grader -- keep ONLY tasks whose gold passes (guarantees tests+grader are correct,
so GRPO rewards are trustworthy). No worker calls -> no cost."""
import os, sys, json
os.environ.pop("HF_HUB_OFFLINE", None)
sys.path.insert(0, "ultra")
from datasets import load_dataset
from ultra.sources.hf import make_taskspec
from ultra.grading.verifiers import code_exec_stdio

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 120
CAP_TESTS = 12
OUT = "director/manifests/fugu_clean_v1/grpo_pilot_train/code_hard_taco_taskspecs.jsonl"
SYS = ("You are an expert competitive programmer. Read input from stdin and write output to stdout. "
       "Provide your complete solution in a single ```python code block.")
ds = load_dataset("likaixin/TACO-verified", split="train", streaming=True)
fo = open(OUT, "w")
scanned = kept = gold_fail = 0
for r in ds:
    if kept >= TARGET:
        break
    if r.get("difficulty") not in ("HARD", "VERY_HARD"):
        continue
    if (r.get("starter_code") or "").strip():
        continue  # function-style needs a different harness; keep pure stdin/stdout
    try:
        io = json.loads(r["input_output"])
        ins, outs = io.get("inputs", []), io.get("outputs", [])
    except Exception:
        continue
    if len(ins) < 2 or len(ins) != len(outs):
        continue
    tests = [{"input": str(i), "output": str(o)} for i, o in zip(ins, outs)][:CAP_TESTS]
    ea = {"tests": tests, "timeout": 10}
    scanned += 1
    # validate: does a GOLD solution pass the grader?
    sols = r.get("solutions") or []
    if isinstance(sols, str):
        try: sols = json.loads(sols)
        except Exception: sols = [sols]
    ok = False
    for gold in sols[:3]:
        try:
            if code_exec_stdio(f"```python\n{gold}\n```", ea) >= 1.0:
                ok = True; break
        except Exception:
            continue
    if not ok:
        gold_fail += 1
        continue
    spec = make_taskspec(
        task_id=f"taco__{r['id']}", capability="unit_code", source_name="taco", source_version="v1",
        policy="train_allowed", harness="direct_qa", grader_type="code_exec_stdio", expected_answer=ea,
        prompt=str(r["question"]), system=SYS, group_id="taco", domain="code",
        tags=["code", "taco", str(r.get("difficulty")).lower()], url_or_ref="likaixin/TACO-verified")
    fo.write(spec.model_dump_json() + "\n"); fo.flush(); kept += 1
    if kept % 20 == 0:
        print(f"  kept {kept} (scanned {scanned}, gold-fail dropped {gold_fail})", flush=True)
fo.close()
print(f"DONE: kept {kept} HARD TACO tasks | scanned {scanned} | gold-validation drop {gold_fail} ({gold_fail/max(scanned,1):.0%}) -> {OUT}", flush=True)
