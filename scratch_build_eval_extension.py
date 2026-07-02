"""Build the two eval manifests that fix the n=30 noise problem (zero worker calls, zero cost):

  1. heldout_trend60_taskspecs.jsonl  -- DECISION set: the existing 30 (continuity) + 15 new LCB-V6
     + 15 new AIME_2024 (the unused complement). Halves the SE (+-0.08 -> +-0.06) for all decisions.
  2. heldout_confirmation_taskspecs.jsonl -- CONFIRMATION set (15 fresh LCB-V6 + 15 fresh Omni-MATH):
     NEVER evaluated during the run. Used exactly once, on the final best-checkpoint, to defeat
     max-selection bias (selecting best-of-16 checkpoints on the same set you report inflates the
     result -- the headroom-metrics-must-crossfit lesson).

Disjointness enforced by assertion across: training manifest, original 30, extension, confirmation.
"""
import base64, hashlib, json, pickle, random, sys, zlib

sys.path.insert(0, "ultra")
from ultra.grading.verifiers import math_equal
from ultra.schemas import TaskSpec
from ultra.sources.hf import make_taskspec

D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
LITE = "/var/lib/mesh/flavius/huggingface/hub/datasets--livecodebench--code_generation_lite/snapshots/0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
CODE_SYS = "Write a complete Python program that reads from stdin and writes to stdout. Return only the program in a code block."
AIME_SYS = "Solve the problem. Put the final integer answer in \\boxed{}."
OMNI_SYS = "Solve the problem. Put the final answer in \\boxed{}."
CAP = 12
MIN_TESTS = 8

def h(txt):
    return hashlib.sha256(" ".join(str(txt).split()).encode()).hexdigest()

heldout30 = [json.loads(l) for l in open(f"{D}/heldout_trend_taskspecs.jsonl")]
train = [json.loads(l) for l in open(f"{D}/hard_mix_all_taskspecs.jsonl")]
used_lcb = {r["task_id"].split("__", 1)[1] for r in heldout30 if r["task_id"].startswith("lcbv6")}
used_aime = {r["task_id"].split("__", 1)[1] for r in heldout30 if r["task_id"].startswith("aime")}
seen_hashes = {h(r["input"]["messages"][-1]["content"]) for r in train} | {
    h(r["input"]["messages"][-1]["content"]) for r in heldout30}
code_timeout = next(r["grader"]["expected_answer"].get("timeout") for r in heldout30
                    if r["capability"] == "unit_code")
print(f"exclusions: {len(used_lcb)} lcb qids, {len(used_aime)} aime ids, {len(seen_hashes)} content hashes"
      f" | code timeout={code_timeout}")

# ---- LCB-V6 pool (same decode path as scratch_fix_lcb_tests) ----
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
    if not qid or qid in used_lcb or h(row.get("question_content", "")) in seen_hashes:
        continue
    tests = stdin_tests(row)
    if len(tests) >= MIN_TESTS:
        eligible.append((qid, row["question_content"], tests))
print(f"LCB-V6 eligible (>= {MIN_TESTS} stdin tests, unused): {len(eligible)}")
assert len(eligible) >= 30, "not enough fresh V6 problems"
random.Random(7).shuffle(eligible)
lcb_ext, lcb_conf = eligible[:15], eligible[15:30]

def code_spec(qid, content, tests, split_tag):
    return make_taskspec(
        task_id=f"lcbv6__{qid}", capability="unit_code", source_name="livecodebench_v6",
        source_version="v6", policy="train_allowed", harness="code_exec",
        grader_type="code_exec_stdio", expected_answer={"tests": tests, "timeout": code_timeout},
        prompt=str(content), system=CODE_SYS, group_id="lcbv6", domain="code",
        tags=["code", "lcb", "v6", split_tag], url_or_ref="livecodebench/code_generation_lite::v6")

# ---- AIME complement ----
from datasets import load_dataset
aime = load_dataset("Maxwell-Jia/AIME_2024", split="train")
aime_new = [r for r in aime if str(r["ID"]) not in used_aime]
print(f"AIME complement: {len(aime_new)} (expect 15)")
assert len(aime_new) == 15

def aime_spec(r):
    return make_taskspec(
        task_id=f"aime_old__{r['ID']}", capability="math", source_name="aime_old", source_version="v1",
        policy="train_allowed", harness="direct_qa", grader_type="math_equal",
        expected_answer=str(r["Answer"]).strip(), prompt=str(r["Problem"]), system=AIME_SYS,
        group_id="aime", domain="math", tags=["math", "competition"],
        url_or_ref="Maxwell-Jia/AIME_2024")

# ---- Omni-MATH confirmation slice (skip training-250 by content hash) ----
omni = load_dataset("KbsdJames/Omni-MATH", split="test", streaming=True)
omni_conf = []
for r in omni:
    if len(omni_conf) >= 15:
        break
    q, a = r.get("problem"), r.get("answer")
    if not q or a is None or not str(a).strip() or h(q) in seen_hashes:
        continue
    gold = str(a).strip()
    try:
        if math_equal(f"The answer is \\boxed{{{gold}}}.", gold) < 1.0:
            continue
    except Exception:
        continue
    omni_conf.append(make_taskspec(
        task_id=f"omniconf__{len(omni_conf) + 1}", capability="math", source_name="omni_math",
        source_version="v1", policy="train_allowed", harness="direct_qa", grader_type="math_equal",
        expected_answer=gold, prompt=str(q), system=OMNI_SYS, group_id="omni_math_conf",
        domain="math", tags=["math", "omni_math", "olympiad", "confirmation"],
        url_or_ref="KbsdJames/Omni-MATH"))
print(f"Omni-MATH confirmation: {len(omni_conf)}")
assert len(omni_conf) == 15

# ---- write + verify ----
ext_specs = [code_spec(*t, "eval_ext") for t in lcb_ext] + [aime_spec(r) for r in aime_new]
conf_specs = [code_spec(*t, "confirmation") for t in lcb_conf] + omni_conf

with open(f"{D}/heldout_trend60_taskspecs.jsonl", "w") as f:
    for r in heldout30:
        f.write(json.dumps(r) + "\n")
    for s in ext_specs:
        f.write(s.model_dump_json() + "\n")
with open(f"{D}/heldout_confirmation_taskspecs.jsonl", "w") as f:
    for s in conf_specs:
        f.write(s.model_dump_json() + "\n")

all_ids = set()
for path in (f"{D}/hard_mix_all_taskspecs.jsonl", f"{D}/heldout_trend60_taskspecs.jsonl",
             f"{D}/heldout_confirmation_taskspecs.jsonl"):
    for l in open(path):
        r = json.loads(l)
        TaskSpec.model_validate(r)
        assert r["task_id"] not in all_ids, f"DUPLICATE across manifests: {r['task_id']} in {path}"
        all_ids.add(r["task_id"])
n60 = sum(1 for _ in open(f"{D}/heldout_trend60_taskspecs.jsonl"))
nc = sum(1 for _ in open(f"{D}/heldout_confirmation_taskspecs.jsonl"))
print(f"DONE: trend60={n60} rows, confirmation={nc} rows, {len(all_ids)} unique ids across all manifests")
