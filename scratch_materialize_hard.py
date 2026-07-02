"""Materialize a HARD, accurately-gradeable eval set where gemini-flash should FAIL,
exposing real headroom: HLE multiple-choice (text-only) + GPQA-Diamond + AIME.
All use exact graders (mc_letter / math_equal). Writes incrementally so a finalization
crash can't corrupt the output."""
import json, sys
sys.path.insert(0, "ultra")
from ultra.sources.hf import hf_rows, make_taskspec
from ultra.policy import SOURCE_POLICY

D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
out_path = f"{D}/hard_eval_taskspecs.jsonl"
fo = open(out_path, "w")
n_total = 0

# --- HLE multiple-choice, text-only (mc_letter -- exact grading) ---
_MC_SYS = "Answer the question. Put the final answer letter in \\boxed{}."
n = 0
for i, r in enumerate(hf_rows("cais/hle", "test", streaming=True, limit=400)):
    if r.get("answer_type") != "multipleChoice":
        continue
    if str(r.get("image") or "").strip() not in ("", "None"):
        continue
    q = r.get("question"); a = r.get("answer")
    if not q or not a:
        continue
    spec = make_taskspec(
        task_id=f"hle_mc__{r.get('id', i)}", capability="science_knowledge",
        source_name="hle", source_version="v1", policy=SOURCE_POLICY.get("hle", "final_eval_only"),
        harness="direct_qa", grader_type="mc_letter", expected_answer=str(a).strip()[:1].upper(),
        prompt=str(q), system=_MC_SYS, group_id="hle", domain="expert",
        tags=["hle", "hard", str(r.get("category") or "")], url_or_ref="cais/hle")
    fo.write(spec.model_dump_json() + "\n"); fo.flush(); n += 1
    if n >= 50:
        break
print(f"  HLE-MC (text-only): {n}", flush=True); n_total += n

# --- GPQA-Diamond (mc_letter) + AIME (math_equal) via adapters ---
from ultra.sources.direct import GPQAStyleAdapter, AIMEAdapter
for tag, ad, lim in [("gpqa_diamond", GPQAStyleAdapter(), 40), ("aime", AIMEAdapter(), 30)]:
    n = 0
    try:
        for s in ad.materialize_all(limit=lim, shuffle=True, seed=5):
            fo.write(s.model_dump_json() + "\n"); fo.flush(); n += 1
    except Exception as e:
        print(f"  {tag}: FAIL {str(e)[:70]}", flush=True); continue
    print(f"  {tag}: {n}", flush=True); n_total += n

fo.close()
print(f"TOTAL hard tasks: {n_total} -> {out_path}", flush=True)
print("DONE", flush=True)
