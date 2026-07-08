"""Build heldout_fshard_taskspecs.jsonl — the FULL-STRENGTH-CONTESTED code set.

trend60 is near-saturated at full strength (best solo 0.933, oracle 0.933, fs_verdict60
2026-07-08) → no arithmetic room to WIN. This set = every UNUSED LCB-V6 difficulty=hard
problem (benchmark's own label — no worker-failure selection, so the bar is unbiased),
capped at 40, disjoint from: training mix, trend60, and the SEALED confirmation set
(ids excluded WITHOUT reading its contents beyond task ids).

Same decode/spec path as scratch_build_eval_extension.py (CAP=12 tests, MIN_TESTS=8,
identical grader/timeout).

Run:  PYTHONPATH=ultra .venv/bin/python scratchpad/build_fshard_manifest.py
"""
import base64, hashlib, json, pickle, random, sys, zlib

sys.path.insert(0, "ultra")
from ultra.sources.hf import make_taskspec

D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
LITE = ("/var/lib/mesh/flavius/huggingface/hub/datasets--livecodebench--code_generation_lite/"
        "snapshots/0fe84c3912ea0c4d4a78037083943e8f0c4dd505")
CODE_SYS = ("Write a complete Python program that reads from stdin and writes to stdout. "
            "Return only the program in a code block.")
CAP = 12
MIN_TESTS = 8
N_MAX = 40
OUT = f"{D}/heldout_fshard_taskspecs.jsonl"


def h(txt):
    return hashlib.sha256(" ".join(str(txt).split()).encode()).hexdigest()


trend60 = [json.loads(l) for l in open(f"{D}/heldout_trend60_taskspecs.jsonl")]
conf = [json.loads(l) for l in open(f"{D}/heldout_confirmation_taskspecs.jsonl")]
train = [json.loads(l) for l in open(f"{D}/hard_mix_all_taskspecs.jsonl")]
used_lcb = ({r["task_id"].split("__", 1)[1] for r in trend60 if r["task_id"].startswith("lcbv6")}
            | {r["task_id"].split("__", 1)[1] for r in conf if r["task_id"].startswith("lcbv6")})
seen_hashes = ({h(r["input"]["messages"][-1]["content"]) for r in train}
               | {h(r["input"]["messages"][-1]["content"]) for r in trend60}
               | {h(r["input"]["messages"][-1]["content"]) for r in conf})
code_timeout = next(r["grader"]["expected_answer"].get("timeout") for r in trend60
                    if r["capability"] == "unit_code")
print(f"exclusions: {len(used_lcb)} lcb qids, {len(seen_hashes)} content hashes | timeout={code_timeout}")


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
    out = []
    for c in list(pub) + list(decode_private(row.get("private_test_cases"))):
        if c.get("testtype") == "stdin":
            out.append({"input": c.get("input", ""), "output": c.get("output", "")})
        if len(out) >= CAP:
            break
    return out


eligible = []
for line in open(f"{LITE}/test6.jsonl"):
    row = json.loads(line)
    qid = row.get("question_id")
    if not qid or row.get("difficulty") != "hard":
        continue
    if qid in used_lcb or h(row.get("question_content", "")) in seen_hashes:
        continue
    tests = stdin_tests(row)
    if len(tests) >= MIN_TESTS:
        eligible.append((qid, row["question_content"], tests))
print(f"eligible unused HARD problems with >= {MIN_TESTS} stdin tests: {len(eligible)}")
random.Random(20260708).shuffle(eligible)
picked = eligible[:N_MAX]

specs = []
for qid, content, tests in picked:
    specs.append(make_taskspec(
        task_id=f"lcbv6__{qid}", capability="unit_code", source_name="livecodebench_v6",
        source_version="v6", policy="train_allowed", harness="code_exec",
        grader_type="code_exec_stdio", expected_answer={"tests": tests, "timeout": code_timeout},
        prompt=str(content), system=CODE_SYS, group_id="lcbv6", domain="code",
        tags=["code", "lcb", "v6", "fshard"], url_or_ref="livecodebench/code_generation_lite::v6"))

ids = [s.task_id for s in specs]
assert len(ids) == len(set(ids))
assert not ({i.split("__", 1)[1] for i in ids} & used_lcb), "overlap with trend60/confirmation!"
with open(OUT, "w") as f:
    for s in specs:
        f.write(s.model_dump_json() + "\n")
print(f"wrote {len(specs)} fshard taskspecs to {OUT}")
