"""Fix LiveCodeBench grading: official LCB grades on public + ~59 hidden tests; ours used
public only (overstates accuracy). Decode the private tests and grade on public+private
(stdin-type), capped at CAP for grading speed within the eval's wait_for timeout.
Surgically updates ONLY the livecodebench taskspecs in train + held-out (math/knowledge untouched).
"""
import json, base64, zlib, pickle, sys
sys.path.insert(0, "ultra")
D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
LITE = "/var/lib/mesh/flavius/huggingface/hub/datasets--livecodebench--code_generation_lite/snapshots/0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
CAP = 12

def decode_private(priv):
    if not priv:
        return []
    try:
        return json.loads(priv)
    except Exception:
        pass
    try:
        raw = pickle.loads(zlib.decompress(base64.b64decode(priv.encode("utf-8"))))
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return []

def stdin_tests(row):
    try:
        pub = json.loads(row["public_test_cases"]) if isinstance(row["public_test_cases"], str) else row["public_test_cases"]
    except Exception:
        pub = []
    priv = decode_private(row.get("private_test_cases"))
    out = []
    for c in list(pub) + list(priv):
        if c.get("testtype") == "stdin":
            out.append({"input": c.get("input", ""), "output": c.get("output", "")})
        if len(out) >= CAP:
            break
    return out

# build qid -> tests from BOTH V1 (test.jsonl) and V6 (test6.jsonl)
qid_tests = {}
for fn in ["test.jsonl", "test6.jsonl"]:
    for line in open(f"{LITE}/{fn}"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        qid = str(r.get("question_id"))
        qid_tests[qid] = stdin_tests(r)

# update the LCB taskspecs in both files
for fn in ["paper_train_taskspecs.jsonl", "heldout_eval_taskspecs.jsonl"]:
    rows = [json.loads(l) for l in open(f"{D}/{fn}")]
    updated = 0; before_avg = []; after_avg = []
    for r in rows:
        if r["capability"] != "unit_code":
            continue
        qid = r["task_id"].split("__", 1)[1]  # livecodebench__{qid} / lcbv6__{qid}
        if qid not in qid_tests or not qid_tests[qid]:
            continue
        before_avg.append(len(r["grader"]["expected_answer"]["tests"]))
        r["grader"]["expected_answer"]["tests"] = qid_tests[qid]
        after_avg.append(len(qid_tests[qid]))
        updated += 1
    with open(f"{D}/{fn}", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ba = sum(before_avg)/len(before_avg) if before_avg else 0
    aa = sum(after_avg)/len(after_avg) if after_avg else 0
    print(f"{fn}: updated {updated} LCB tasks | tests/task avg {ba:.1f} -> {aa:.1f}", flush=True)
print("DONE", flush=True)
