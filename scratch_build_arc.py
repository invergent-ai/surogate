"""Build ARC-AGI-2 taskspecs (grid_exact grader). Infer-rule-from-examples -> apply -> exact grid.
Cleanly verifiable + decomposition-friendly. Gold-validate: the gold grid must grade against itself.
Filter to token-manageable grids. No worker calls -> no cost."""
import os, sys, json
os.environ.pop("HF_HUB_OFFLINE", None)
sys.path.insert(0, "ultra")
from datasets import load_dataset
from ultra.sources.hf import make_taskspec
from ultra.grading.verifiers import grid_exact

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUT = "director/manifests/fugu_clean_v1/grpo_pilot_train/reasoning_arc_agi2_taskspecs.jsonl"
SYS = ("You are solving an ARC-AGI puzzle. From the input->output examples, infer the single "
       "transformation rule, then apply it to the test input. Output ONLY the resulting grid as a "
       "JSON 2D array of integers, e.g. [[0,1],[1,0]].")
def gstr(g): return "\n".join(" ".join(str(x) for x in row) for row in g)
def big(g): return len(g) > 30 or (g and len(g[0]) > 30)  # ARC max is 30x30; keep full range incl large/hard

ds = list(load_dataset("arc-agi-community/arc-agi-2", split="train"))  # full 1000, non-streaming
import random; random.Random(3).shuffle(ds)
# prefer LARGER puzzles (the hard ones) — sort by test-input area, descending, then take
def area(r):
    q = r.get("question") or [{}]
    g = q[0].get("input") or [[0]]
    return len(g) * (len(g[0]) if g else 1)
ds.sort(key=area, reverse=True)
fo = open(OUT, "w"); kept = scanned = badgold = toobig = 0
for i, r in enumerate(ds):
    if kept >= TARGET:
        break
    fs = r.get("fewshots") or []; q = r.get("question") or []
    if not fs or not q:
        continue
    test_in = q[0]["input"]; gold = q[0]["output"]
    if big(test_in) or big(gold) or any(big(e["input"]) or big(e["output"]) for e in fs):
        toobig += 1; continue
    scanned += 1
    if grid_exact(json.dumps(gold), gold) < 1.0:   # gold must grade against itself
        badgold += 1; continue
    ex = "\n\n".join(f"Example {j+1} input:\n{gstr(e['input'])}\nExample {j+1} output:\n{gstr(e['output'])}"
                     for j, e in enumerate(fs))
    prompt = f"{ex}\n\nTest input:\n{gstr(test_in)}\n\nGive the test output grid as a JSON 2D array."
    spec = make_taskspec(task_id=f"arc2__{i}", capability="reasoning", source_name="arc_agi_2",
        source_version="v1", policy="train_allowed", harness="direct_qa", grader_type="grid_exact",
        expected_answer=gold, prompt=prompt, system=SYS, group_id="arc_agi_2", domain="reasoning",
        tags=["reasoning", "arc_agi_2", "hard", "grid"], url_or_ref="arc-agi-community/arc-agi-2")
    fo.write(spec.model_dump_json() + "\n"); kept += 1
fo.close()
print(f"DONE: kept {kept} ARC-AGI-2 tasks | scanned {scanned} | gold-fail {badgold} | too-big skipped {toobig} -> {OUT}", flush=True)
