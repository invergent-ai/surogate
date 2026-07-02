"""MANDATORY pre-flight gate: verify EVERY MISSION.md condition before a live GRPO launch.
Hard requirement (user, 2026-07-01): never launch GRPO without this passing.
Checks GRPO recipe params, worker pool, source data, duration, provider routing against
orch_lcb.yaml / train_lcb.yaml / pilot_config / providers.py / the task manifest.
Exit 0 = LAUNCH ALLOWED (all pass). Exit 1 = BLOCKED."""
import json, sys, re
sys.path.insert(0, "ultra")
import yaml
D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
orch = yaml.safe_load(open(f"{D}/orch_lcb.yaml"))
train = yaml.safe_load(open(f"{D}/train_lcb.yaml"))
pilot = json.load(open(f"{D}/pilot_config_singleturn.json"))
from ultra.providers import MODELS, FORCE_PROVIDER, DISALLOWED_MODEL_PROVIDERS
envargs = orch["env"][0]["args"]
manifest = envargs["task_manifest_path"]
rows = [json.loads(l) for l in open(manifest)]

checks = []  # (name, ok, detail)
def chk(name, ok, detail): checks.append((name, bool(ok), detail))

# --- GRPO RECIPE (MISSION "NON-NEGOTIABLE GRPO RECIPE") ---
chk("temperature == 1.0", orch["sampling"]["temperature"] == 1.0, orch["sampling"]["temperature"])
chk("rollouts_per_example >= 16", orch["rollouts_per_example"] >= 16, orch["rollouts_per_example"])
chk("learning_rate == 1e-6", float(train["learning_rate"]) == 1e-6, train["learning_rate"])
chk("KL penalty == 0", float(train["loss"]["kl_tau"]) == 0.0, train["loss"]["kl_tau"])
chk("worker_max_tokens == 4096", envargs["worker_max_tokens"] == 4096, envargs["worker_max_tokens"])
chk("worker_temperature == 0.2", envargs["worker_temperature"] == 0.2, envargs["worker_temperature"])
chk("worker_reasoning_effort == minimal", envargs["worker_reasoning_effort"] == "minimal", envargs["worker_reasoning_effort"])
chk("batch_size % rpe == 0", orch["batch_size"] % orch["rollouts_per_example"] == 0, f"{orch['batch_size']}%{orch['rollouts_per_example']}")
chk("max_concurrent >= rpe", orch["max_concurrent"] >= orch["rollouts_per_example"], orch["max_concurrent"])
# base = workflow-SFT, NOT repair-SFT / solo (FORBIDDEN)
base = str(train["model"])
chk("base == workflow-SFT (not repair/solo)", "workflow_sft" in base and "repair" not in base, base.split("/")[-1])
# NOT replay reward (live execution required)
chk("NOT replay (live execution)", not envargs.get("replay_seed_manifest"), envargs.get("replay_seed_manifest"))

# --- WORKER POOL (MISSION line 155: Opus-4.8 / GPT-5.5 / Gemini-3.5-Flash / GLM-5.2) ---
POOL_EXPECT = {"opus": "claude-opus-4-8", "gpt": "gpt-5.5", "gemini": "gemini-3.5-flash", "glm": "glm-5.2"}
names = pilot.get("worker_pool_names", [])
chk("pool == [st_opus, st_gemini, st_gpt, st_glm]", set(names) == {"st_opus","st_gemini","st_gpt","st_glm"}, names)
for logical, expect in POOL_EXPECT.items():
    slug = MODELS.get(logical, {})
    got = slug.get("yunwu") or slug.get("openrouter", "")
    chk(f"pool model {logical} == {expect} (NOT gemini-3.1-pro)", expect in " ".join(slug.values()), slug)

# --- SOURCE DATA (MISSION HEADROOM MAP: verifiable CODE, difficulty-filtered) ---
caps = {r["capability"] for r in rows}; grads = {r["grader"]["type"] for r in rows}; srcs = {r["source"]["name"] for r in rows}
# code core + user-directed VERIFIABLE reasoning (exact-answer) -- see MISSION HEADROOM MAP 2026-07-01 UPDATE
CODE_SOURCES = {"livecodebench", "taco", "code_contests"}
REASONING_SOURCES = {"omni_math", "reasoning_gym", "rlpr"}  # MMLU-Pro dropped: saturated (>90%) for 2026 pool
VERIFIABLE_GRADERS = {"code_exec_stdio", "math_equal"}  # exact/test-based; NOT mc_letter (guessing/saturated MC)
ALLOWED_CAPS = {"unit_code", "math", "reasoning"}
diffs = {t for r in rows for t in r["metadata"]["tags"] if t in ("easy","medium","hard","very_hard","olympiad")}
n_code = sum(1 for r in rows if r["source"]["name"] in CODE_SOURCES)
chk("all capabilities allowed (code/math/reasoning)", caps <= ALLOWED_CAPS, caps)
chk("graders VERIFIABLE (code_exec_stdio/math_equal, NO mc_letter)", grads <= VERIFIABLE_GRADERS, grads)
chk("NO multiple-choice (mc_letter) tasks", "mc_letter" not in grads, grads)
chk("sources are code+verifiable-reasoning", srcs <= (CODE_SOURCES | REASONING_SOURCES) and srcs, srcs)
chk("data is HARD (no easy/medium)", diffs and not (diffs & {"easy","medium"}), diffs)
chk("code is a strong share (>= 33%, headroom core)", n_code >= len(rows) * 0.33, f"{n_code}/{len(rows)} code")
chk("online_difficulty_filtering == true", orch["buffer"]["online_difficulty_filtering"] is True, orch["buffer"].get("online_difficulty_filtering"))
chk("task count > 0", len(rows) > 0, len(rows))

# --- DURATION (MISSION: MIN 200 steps; checkpoint every 10) ---
chk("orch max_steps >= 200", orch["max_steps"] >= 200, orch["max_steps"])
chk("train max_steps >= 200", train["max_steps"] >= 200, train["max_steps"])
chk("train max_steps == orch max_steps", train["max_steps"] == orch["max_steps"], f"{train['max_steps']} vs {orch['max_steps']}")
chk("save_steps <= 10 (outage-resilient)", train["save_steps"] <= 10, train["save_steps"])

# --- PROVIDER ROUTING (MISSION Operational Invariants) ---
chk("FORCE_PROVIDER gemini -> openrouter", FORCE_PROVIDER.get("gemini") == "openrouter", FORCE_PROVIDER.get("gemini"))
chk("gpt DISALLOWED on openrouter", "openrouter" in DISALLOWED_MODEL_PROVIDERS.get("gpt", set()), DISALLOWED_MODEL_PROVIDERS.get("gpt"))
chk("allow_yunwu_live == true", envargs.get("allow_yunwu_live") is True, envargs.get("allow_yunwu_live"))

# --- report ---
print("="*72)
print("PRE-FLIGHT CHECK vs MISSION.md  (LCB GRPO run)")
print("="*72)
fails = 0
for name, ok, detail in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}   ({detail})")
    fails += (not ok)
print("="*72)
if fails:
    print(f"GATE: BLOCKED -- {fails} condition(s) FAIL. Do NOT launch.")
    sys.exit(1)
print(f"GATE: LAUNCH ALLOWED -- all {len(checks)} MISSION conditions satisfied.")
print("Note: min-200-step DURATION is set; verify/refine emergence + best-checkpoint eval are POST-run gates (Fig-3 dynamics).")
sys.exit(0)
