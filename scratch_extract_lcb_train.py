"""Extract the already-processed LCB-V1 code tasks (public+private capped tests) from the
paper_train mix into a clean LCB-ONLY train manifest. The mix dilutes the GRPO gradient with
~0-headroom math/MMLU (see HEADROOM MAP); training on code-only concentrates the signal."""
import json, sys, os
D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
src = f"{D}/paper_train_taskspecs.jsonl"
out = f"{D}/lcb_train_taskspecs.jsonl"
n = 0; testcounts = []
with open(out, "w") as fo:
    for line in open(src):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("capability") != "unit_code":
            continue
        tests = (r.get("grader", {}).get("expected_answer") or {}).get("tests")
        if not tests:
            continue
        testcounts.append(len(tests))
        fo.write(line if line.endswith("\n") else line + "\n"); n += 1
sz = os.path.getsize(out) / 1e6
print(f"LCB-V1 train: {n} tasks -> {out} ({sz:.1f} MB)", flush=True)
if testcounts:
    print(f"tests/task: min={min(testcounts)} max={max(testcounts)} avg={sum(testcounts)/len(testcounts):.1f}", flush=True)
# sanity: show a couple task_ids
for line in open(out):
    print("  sample:", json.loads(line)["task_id"], flush=True); break
print("DONE", flush=True)
