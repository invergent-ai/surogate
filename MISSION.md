# Mission: Fugu-Ultra

Objective: create a highly performant Fugu-Ultra model that outperforms any individual model or individual model+scaffold worker.

## HEADROOM MAP — benchmark-confirmed (2026-06-30): TRAIN ON VERIFIABLE CODE, NOT REASONING

**This is the load-bearing strategic fact. Read it before choosing any task, probe, or training run.**

The official Fugu-Ultra benchmark (the TRAINED conductor vs FULL-STRENGTH workers Opus-4.8 / Gemini-3.1-Pro / GPT-5.5) pins down exactly where the conductor beats the best single worker. Δ = Fugu-Ultra − best individual worker:

| Tier | Benchmark | Fugu-Ultra | best worker | Δ | nature |
|---|---|---:|---:|---:|---|
| **1 — REAL HEADROOM (all verifiable code)** | LiveCodeBench | 93.2 | 88.5 | **+4.7** | single-turn code-gen, test-graded — **FEASIBLE to train** |
| | SWE Bench Pro | 73.7 | 69.2 | **+4.5** | agentic — eval-only |
| | TerminalBench 2.1 | 82.1 | 78.2 | **+3.9** | agentic — eval-only |
| | LiveCodeBench Pro | 90.8 | 88.4 | **+2.4** | code-gen — feasible |
| **2 — ~ZERO HEADROOM (all single-turn reasoning)** | HLE | 50.0 | 49.8 | +0.2 | reasoning, no cheap verifier |
| | GPQA-D | 95.5 | 94.3 | +1.2 | reasoning (MC) |
| | SciCode | 58.7 | 58.9 | **−0.2** | reasoning |
| **3 — conductor LOSES** | MRCRv2 | 93.6 | 94.8 | −1.2 | long-context |
| | Long Context | 73.3 | 74.3 | −1.0 | long-context |

**MECHANISM** — why this is true, not just an observation: the conductor's only edge is DECOMPOSE→VERIFY→REFINE, which adds value ONLY when (a) the task is decomposable and (b) there is a cheap EXTERNAL verifier to catch a draft's errors. **Code has unit tests; math/knowledge reasoning does not.** Decomposing one limited worker with another equally-limited worker flips nothing without an external check. This is a VERIFIABILITY axis, NOT a hard-vs-easy axis — HLE is the single *hardest* benchmark here and the single *worst* place to look (+0.2 even for the fully-trained model).

**WHERE TO LOOK** (the feasible × headroom intersection): **LiveCodeBench / LiveCodeBench-Pro** — single-turn code generation, hidden-test graded. It has the headroom (+4.7) AND is feasible to train (4096-tok single-call workers, no Docker/agent-loops → no call explosion). This is the proven beat-the-workers training core. The decomposition skill learned here TRANSFERS to the agentic eval-only targets (SWE Bench Pro / TerminalBench). Escalate difficulty toward harder **CODE** (LCB-Pro, BigCodeBench-hard, SWE-smith code-gen) — never toward harder reasoning.

**WHAT NOT TO DO**:
1. Do NOT probe or train on single-turn **REASONING** (MATH500, MMLU, HLE, GPQA-Diamond, AIME, FrontierMath). The benchmark proves ~0 conductor headroom there; it is the FLAT part of the curve, not a difficulty problem. CONFIRMED empirically 2026-06-30: a full session built a "harder" HLE-MC(50)+GPQA-D(40)+AIME(30) set and re-measured — headroom **+0.04** (within MC-guessing noise), exactly as this table predicts. The old "if too easy, escalate to HLE/FrontierMath/GPQA" guidance ([[frontier-pool-makes-paper-tasks-easy]], DATASETS note below) was HALF WRONG — only its LCB-Pro (code) half had headroom.
2. Do NOT TRAIN on live-agentic SWE-Bench / TerminalBench (containers, 20-turn workers). Call explosion → infeasible (recorded: 0 rollouts / 11 min, $100s wasted). These are EVAL-ONLY; the skill transfers from LCB code-gen training.
3. Do NOT gate on ROUTING headroom (oracle-of-N-workers vs best-single). The conductor DECOMPOSES, it does not route — per-task routing is NULL (cross-fit p=0.90, [[swesmith-per-task-routing-null]]). The real gate = within-group reward VARIANCE from decomposition on verifiable code (some workflows pass tests, some don't), difficulty-filtered to the learnable band.
4. Beware MC GUESSING inflation: four guessers on a 4-choice MC have oracle 0.68 by luck. Never read a multiple-choice oracle as headroom.

**UPDATE (user-directed 2026-07-01) — verifiable reasoning ADDED to the mix as an experiment**: hard EXACT-ANSWER reasoning (NOT multiple-choice) — Omni-MATH olympiad (120) + Reasoning-Gym-Hard (30) — is added to the training mix alongside the code core (TACO+LCB, 181), a deliberate exception to WHAT-NOT-TO-DO #1. Rationale: (a) the paper trained a MATH+code mix (Fig 3 bottom = a physics decomposition); (b) my "reasoning≈0 headroom" was measured with ROUTING oracle only — never the DECOMPOSITION test that flipped code from +0.00 routing to +0.05; (c) exact-answer math avoids the MC-guessing confound. Code-dominant (55%); `online_difficulty_filtering` evicts no-variance reasoning tasks; held-out eval (LCB-V6 + reasoning) is the verdict. This does NOT reverse the verifiability conclusion — MC reasoning (GPQA/HLE-MC) stays excluded.

## NON-NEGOTIABLE OBJECTIVE (hard-enforced 2026-06-30)

The Fugu-Ultra model MUST be a **task-specific DECOMPOSITION conductor**, faithful to the Conductor paper. No deviation is accepted.

REQUIRED (what "done" means):
- The conductor READS a task and GENERATES a **task-specific decomposition** — real subtasks for *this* problem (with dependencies/access), NOT generic role labels, NOT a choice among fixed templates.
- Subtasks are dispatched to workers; their results COMPOSE into the final solution.
- Trained via **RL (GRPO)** on **LIVE execution**; reward = did the composed solution solve a task the best single worker FAILS whole.
- The conductor must DISCOVER good decompositions through reward — generalizing to held-out tasks.

FORBIDDEN (do NOT pursue; these do not count as the model):
- Pure routing (pick one model per task) — proven NULL (per-task routability cross-fit p=0.90).
- Fixed coordination templates (draft->debug, ladder) — shallow, not task-specific.
- SFT / behavioral-cloning shortcuts — imitation, not decomposition discovery.
- Replay over a fixed arm-set — structurally cannot represent novel decompositions.

WHY (evidence this is the only lever): a crude 2-3 step decomposition already recovered **9 of 16 tasks Opus fails whole** (Opus-solo 9/25 -> decomposition 19/25). Routing and templates underwhelmed exactly because they are not decomposition. Task-specific, RL-learned decomposition is the sole mechanism that beats a frontier model.

## NON-NEGOTIABLE GRPO RECIPE (hard-enforced 2026-06-30)

GRPO params MUST match the Conductor paper (A.1). Every prior collapse came from deviating. No deviation is accepted.

| param | REQUIRED (paper) | rationale |
|---|---|---|
| temperature | **1.0** | exploration -> within-group reward variance |
| rollouts_per_example | **>=16 (paper: 64)** | THE variance source; rpe 2-4 = grad->0 collapse (every failed run) |
| learning_rate | **1e-6** | paper; lr 1e-4 (100x) caused instability |
| KL penalty | **NONE (kl_tau = 0)** | Conductor A.1: "set the reference model KL divergence penalty to 0"; Fugu 3.2.3: "without any KL divergence penalty". The beta*D_KL in eq6/eq1 has beta=0. (I wrongly added a KL anchor from the formula -- reverted.) |
| max_workflow_steps | **<=5 (avg ~3)** | paper: "agentic workflows of up to 5 steps"; Conductor learns avg 3. A 2-step gate FORCED solos = DEGENERATE. |
| TASK TYPE (core) | **single-turn CODE generation** (LiveCodeBench-style, test-graded) | Conductor A.1 trained on MATH/MMLU/RLPR/LCB-V1 — but only the CODE half carries conductor headroom (see HEADROOM MAP); math/MMLU/reasoning are ~0-headroom filler. FAST + feasible (200 iters, 2 H100s). |
| worker call (single-turn) | **single 4096-tok call, temp 0.2, MINIMAL reasoning** | Conductor A.1 exact training setting. No containers/apt-get/multi-turn loops -> cheap rollouts. |
| worker turns (multi-turn EXT) | **UNLIMITED** | Fugu 3.2.3 extension ONLY ("any agent permitted unlimited interaction"). Expensive (call explosion); layer on AFTER the single-turn core works. |
| advantage | **A_i = (r_i - mean)/std** (eq 7) | grouped Monte-Carlo over G rollouts/question |
| reward | **0 / 0.5 / 1** (format / correctness) | r=0 unparseable; r=1 final output matches solution; r=0.5 well-formatted but wrong |
| few-shot prompt | **REQUIRED (coldstart)** | Conductor uses pretrained LM + few-shot examples in system prompt as the coldstart (B.7: removing few-shot drops LCB 64->55). No separate distillation-SFT needed. |

ANTI-DEGENERACY (the 3 things that ARE the value -- none may be stripped):
1. **Multi-step <=5** workflows (plan->code->verify, build->debug, debate). NOT solo. The repair-SFT base (solos) is the MOST degenerate base -- FORBIDDEN.
2. **Targeted per-worker subtasks** + topologies (best-of-N, chain, tree). These EMERGE FROM GRPO -- the paper: "we observe the emergence of problem decompositions and prompt-engineered subtasks... alongside communication topologies." NOT pre-distilled, NOT generic "Solve the task" (Table 9: stripping subtasks drops LCB 64.3->58.6).
3. **Complementary FRONTIER pool** (Opus/Gemini/GPT/GLM) + **data diversity**. NOT the tight 3-terminal pool (not complementary -> random route = 0.5, zero generalization).

REAL EXECUTION REQUIRED: the reward MUST come from LIVE worker execution of the multi-step+subtask workflow (for single-turn = a real single worker call per step, graded vs ground truth). Replay / `routed-arm in successful_arms` is BLIND to subtasks+multi-step -> can only learn pure routing -> degenerate.

PROPER BASE = pretrained LM + **few-shot Conductor prompt** (the paper's coldstart). We use the **workflow-SFT base** (`output/fugu_ultra_workflow_sft_qwen3_8b`) as a stronger coldstart -- it reliably EMITS multi-step (<=5) workflows under the few-shot prompt (verified: 5-step plan->implement->critique->revise). Decomposition + targeted subtasks then EMERGE from GRPO (no separate distillation-SFT). The repair-SFT (solos) is FORBIDDEN -- solos give GRPO nothing to grow. FORBIDDEN params: rpe<16, temp<1.0, lr>1e-6, **any KL (kl_tau>0)**, max_workflow_steps<=2, the repair-SFT/solo base, replay reward -- the exact causes of every prior collapse + the degenerate conductor.

Fugu-Ultra = Conductor + long-horizon function-calling + multi-agent workflows via adaptive agent memory (the report's extensions over the base Conductor).

### SYSTEM PROMPT (Conductor prompt -- ICLR Fig 13 + B.2/B.7)
- Output = three Python lists after CoT: `model id`, `subtasks`, `access list` (same length). Our env uses an info-equivalent JSON `{"steps":[{worker_id,subtask,access,budget}]}` -- same content.
- The prompt instructs: solve INDIRECTLY by querying numbered models; up to 5 steps; each step = (worker id, natural-language subtask, access list of prior steps; "all" or []); a subtask may solve-from-scratch / refine / plan / verify.
- **Few-shot examples are REQUIRED** (the coldstart) -- removing them drops LCB 64->55 (B.7). We rewrote `_system_prompt` paper-faithful: decomposition guidance + 2 few-shot examples (plan->implement; draft->verify->refine), removed the "short" bias. VERIFIED: workflow-SFT base now emits 5-step decompositions.
- Paper nuances to adopt: (a) workers presented as ORDINAL numbers ("Model 0,1,...") to avoid brand bias; (b) few-shot examples drawn from OOD tasks generalize BEST (B.2 -- prevents reward-hackable repetition). Our examples are generic (not from the train set) -- good.

### DATASETS — FINAL CALIBRATED MIX (measured at the handicap, 2026-07-01)

The paper's dataset LIST (MATH500 / MMLU / RLPR / LCB-V1) is **superseded by measurement**. Every candidate source was probed with the 4 workers **at the actual training handicap (4096 tokens / minimal reasoning)** — NOT the full-strength benchmark. A source earns a slot only if the handicapped workers land in the **learnable band** (not saturated, not all-fail). The load-bearing variable is the 2026 pool's difficulty at the handicap, not the paper's dataset names.

**KEPT — the 461-task mix** (`grpo_pilot_train/hard_mix_all_taskspecs.jsonl`), all gold-validated (each source's gold solutions pass its grader ≥98%), all exact/test-graded, **no multiple-choice**:

| source | tasks | grader | handicap probe (workers) | role |
|---|---:|---|---|---|
| **Omni-MATH** (olympiad) | 250 | math_equal | opus 0.27–0.40, others ~0 | headroom-rich (in-band, workers fail) |
| **Reasoning-Gym-Hard** | 30 | math_equal | opus 0.40 / oracle 0.60 | headroom-rich (+0.20 routing) |
| **TACO-hard** (codeforces) | 120 | code_exec_stdio | opus 1.0, glm 0.2 (band 9/10) | code: gradient + eval-relevance |
| **LiveCodeBench-hard V1** | 61 | code_exec_stdio | gemini/gpt ~1.0 (band 8/10) | code: gradient + eval-relevance |

Code-share 39%; the headroom core is the math/reasoning half (that's where handicapped workers actually fail). Held-out EVAL (never train): **LiveCodeBench-V6** (code) + a disjoint hard-math set (Omni-MATH held-out / AIME).

**DROPPED — with measured reasons (all at 4096/minimal):**
- **MMLU-Pro** — SATURATED: every worker ~1.0 even at the handicap (MC is answerable in 4096 tok), learnable-band 1/10 → zero gradient. (">90%" is the full-strength number; it holds at the handicap too.)
- **RLPR** (WebInstruct-verified) — ALL-FAIL: oracle 0.10; and a **decomposition** probe (plan→exec + draft→verify→refine) also scored 0.00, recovering 0/15 — the physics/chem is beyond the handicapped workers even decomposed.
- **BBEH** (BIG-Bench Extra Hard) — opus/gemini 0.90, near-saturated → thin headroom.
- **SciCode** — un-validatable grading (scipy harness, per-step functions, `general_solution` often None) + the **one benchmark the conductor LOSES** (−0.2). Grading-risk, skip.
- **ARC-AGI-3** — interactive/agentic (no static dataset), multi-turn → not single-turn-trainable.
- **Candidate, not currently in the mix:** ARC-AGI-2 (`arc-agi-community/arc-agi-2`, large 30×30 puzzles, `grid_exact`) — opus 0.60, learnable/moderate; keep-able if we want a 3rd reasoning source.

**KEY FINDING (valid):** at the 4096/minimal handicap the 2026 pool is strong enough that the only headroom-rich sources (best-worker ≤0.4) are hard **olympiad math + puzzle reasoning**; code is Opus-dominant (thin handicap-headroom, kept for eval-relevance). This partially inverts the full-strength HEADROOM MAP (code +4.7 was measured with full-strength workers; training uses handicapped ones).

**EPISTEMIC CAVEAT (user-established 2026-07-01):** the single-worker saturation table above is VALID (it's about data difficulty). But **hand-crafted decomposition probes prove NOTHING** about the trainable decomposition space — they only show Fig-3's *starting point* (untrained conductor ≈ workers). Only GRPO training tests whether the conductor beats the best worker; do not use fixed-workflow probes to argue feasibility for/against.

### FULL TRAINING PARAMS (ICLR A.1 -- exact)
- Base: Qwen2.5-7B (we use the stronger workflow-SFT coldstart of Qwen3-8B). max completion 1024.
- **200 GRPO iterations**, **4 questions/iter x 64 rollouts/question = batch 256**, temperature **1.0**.
- AdamW, lr **1e-6**, cosine schedule, warmup ratio 0.03, b1=0.9 b2=0.999 eps=0.2.
- **NO KL** (ref sync disabled, KL penalty = 0). Reward 0/0.5/1. Grouped advantage A=(r-mean)/std.
- Workers (single-turn): **4096 max tokens, temp 0.2, MINIMAL reasoning** (single call per step). Compute: 2x H100 80GB.
- Recursion + adaptive-pool are LATER finetune extensions (half-batch recursion instantiation; k-subset sampling) -- not the base run.

### EXPECTED TRAINING DYNAMICS (Figure 3) -- MIN 200 STEPS, don't stop early

Run for a **MINIMUM of 200 GRPO steps**. The Conductor does NOT beat the workers until near the end -- stopping at 150 catches it right at the crossing point, not with a banked win. Figure 3 (paper pool: Gemini-2.5-Pro 75.5 / GPT-5 72 / Sonnet-4 68.8):

- **Starts BELOW every worker** (~59% acc / low reward). This is EXPECTED, not failure -- do not kill an early run that looks worse than the workers.
- **Non-monotonic climb**: crosses the weakest worker ~iter 30, briefly passes the mid worker ~iter 48 then DIPS back (iter 50-75, 73->69.5), recovers. **Do not kill on a mid-run dip.**
- **Crosses the BEST worker only ~iter 150-160**, keeps climbing to ~78 by iter 200 -- the margin is banked in the last ~50 steps.
- **Decomposition strategy EMERGES late**: early workflows are sound subtasks but NO verification (just plan->solve); near convergence the Conductor learns planners + targeted instructions + **verify/refine** (`<idea>`/`<answer>` tags: "verify the strategy, find gaps" -> "refine then implement"). Reward climbing without verify/refine in the emitted workflows = not yet converged.

**Our caveat (stronger 2026 pool)**: our best worker (GPT-5.5 ~0.90 on LCB) sits far above the Conductor's ~0.66 start -- a ~24-pt climb vs the paper's ~16. So **200 is the FLOOR, not a guarantee**; if reward is still climbing at 200, extend (resume from the latest checkpoint). Track the per-step reward trajectory + watch for verify/refine to appear in emitted workflows before declaring convergence. Checkpoint every 10 steps (outage-resilient + trajectory eval); eval the BEST checkpoint on held-out, not blindly the last.

Last updated: 2026-07-01 (added Figure-3 expected dynamics + MIN-200-step floor; code-gen (LCB) core per HEADROOM MAP; KL=0).

## EXTERNAL BENCHMARK — SWE-Together (eval-only transfer target, recorded 2026-07-01)

`study/SWE-Together` (arXiv 2606.29957) — 109 **multi-turn interactive** agentic coding tasks. Each replays a real user–agent session with a reactive **user simulator** (progressive reveal: `instruction.md` turn 0, then follow-ups / course-corrections) in a Dockerized repo pinned to a base commit + `tests/` gate + reference patch + frozen judge rubric. Harnesses: opencode / claude-code / codex / mini-swe-agent. Axes: **Correctness** (agentic judge; pass@1 / pass²) + **User Correction** (#correction + 0.2·nudge).

Leaderboard (opencode harness, k=2, sorted pass@1):

| model | pass@1 | pass² | judge↑ | corr↓ | tok | min |
|---|--:|--:|--:|--:|--:|--:|
| *Oracle (ref ceiling)* | *~78%* | — | *0.904* | — | — | — |
| claude-opus-4.8 | 63% | 52% | 0.801 | 1.38 | 74.0k | 23.3 |
| gpt-5.5 | 58% | 48% | 0.763 | 1.59 | 29.9k | 10.7 |
| claude-opus-4.6 | 58% | 46% | 0.755 | 1.59 | 42.0k | 23.2 |
| glm-5.2 | 55% | 42% | 0.735 | 1.53 | 41.7k | 24.5 |
| glm-5.1 | 52% | 34% | 0.729 | 1.54 | 41.6k | 38.8 |
| deepseek-v4-pro | 48% | 29% | 0.679 | 1.76 | 49.8k | 21.0 |
| minimax-2.7 | 39% | 24% | 0.630 | 2.17 | 43.4k | 36.2 |

**Verdict: eval-only transfer target, NOT a training source.** Multi-turn interactive agentic = the infeasible-to-train category ([[fugu-ultra-single-turn-rl]]); tasks carry `reference_patch` + tests but need a repo-sandbox grader and would land all-fail at the 4096/minimal handicap. The aggregate board is **Opus-4.8-dominant on every quality axis** (single-call pattern; no per-task routing signal is recoverable from aggregates) — but it confirms our pool ordering (opus-4.8 > gpt-5.5 > glm-5.2; 3/4 of the training pool present, no Gemini). Novel axis: **User Correction** (multi-turn steerability), unused by the current objective. Post-training, run the conductor through the 109 as a real-world multi-turn transfer eval (needs a conductor→agent shim).

## Global Milestone Checklist

Final objective:
- [ ] Train a Fugu-Ultra Conductor/model that outperforms the strongest individual model+scaffold worker.
- [ ] Evaluate against individual models, scaffolded workers, fixed workflows, best-of-N, self-reflection baselines, and final held-out targets.

## Current Status & Next Steps

**LIVE: paper-exact group-64 GRPO — resumed 2026-07-02 from `step_00000040` (the 0.867 parity peak), now running from step 41.**

What changed vs the plateaued group-16 run (config audit vs Conductor App. A.1 — full findings in the Run Log, 2026-07-02):
- **batch 256 = 4 questions × 64 rollouts** (was 4 × 16 — off-recipe)
- **advantage `(r−mean)/std`** (was mean-only — ~2-4× under-scaled gradients)
- **#2 redundancy penalty kept** (deliberate deviation; targets over-decomposition our no-CoT conductor won't self-fix)
- unchanged: temp 1.0, lr 1e-6, no KL, ≤5 steps, `online_difficulty_filtering: true`

Run facts:
- ~700–900 worker calls/step (4× old), ~40–70 min/step
- checkpoints every 10 → `output/fugu_ultra_lcb/step_NNNNNNNN`
- live held-out eval WITHOUT stopping training (conc-3 vs the `default` adapter) → `output/fugu_ultra_lcb/heldout_trend.log`
- group-16 branch (steps 41–126) archived → `output/fugu_ultra_lcb_group16_archive/`
- stop+resume: `scratch_resume_lcb.sh` (self-healing broadcast + stale-rollout prune)

### The bar (held-out, n=30; rows ≤120 = superseded group-16 branch; group-64 rows append below)

| checkpoint | conductor | code / math | best worker | oracle | wf steps | gap |
|---|--:|--:|--:|--:|--:|--:|
| base (untrained) | 0.733 | 0.667 / 0.800 | GPT-5.5 0.867 | 0.967 | 3.72 | −0.133 |
| step 10 | 0.600 | — | GPT-5.5 0.867 | 0.967 | 3.93 | −0.267 |
| step 20 (live) | 0.800 | 0.800 / 0.800 | GPT-5.5 0.867 | 0.967 | 3.80 | −0.067 |
| step 30 (live) | 0.833 | 0.733 / 0.933 | GPT-5.5 0.867 | 0.967 | 3.79 | −0.033 |
| step 40 (live) | 0.867 | 0.867 / 0.867 | GPT-5.5 0.867 | 0.967 | 3.59 | +0.000 |
| step 60 (live) | 0.800 | 0.667 / 0.933 | GPT-5.5 0.867 | 0.967 | 3.83 | −0.067 |
| step 80 (frozen) | 0.800 | 0.733 / 0.867 | GPT-5.5 0.867 | 0.967 | 3.73 | −0.067 |
| step 100 (live) | 0.700 | 0.533 / 0.867 | GPT-5.5 0.867 | 0.967 | 3.76 | −0.167 |
| step 120 (live) | 0.800 | 0.733 / 0.867 | GPT-5.5 0.867 | 0.967 | 3.93 | −0.067 |

Group-16 summary: early dip (s10) → **parity at iter 40 (0.867)** → 0.70–0.80 oscillation through 120; the win (>0.867) never came. Oracle 0.967 ≈ 0.10 headroom uncaptured. Interventions/decisions along the way: Run Log.

Eval rules (standing):
- **Eval workers run FULL-STRENGTH** — the 4096/minimal handicap is a training-only gradient device.
- **Judge the trajectory, not single points** (n=30 SE ≈ ±0.08; n=60 ≈ ±0.06).
- Under the fixed recipe the **training curve should now climb too** (paper Fig-3: 57→78); a flat train curve is no longer excusable noise.

### Next steps
1. **Health gate** (first fresh trainer step): grad_norm ~2-4× above the old ~0.001 (std scaling), clean kl/masked% — else abort and diagnose.
2. **Eval every 10 steps**, live conc-3, no stop: `EVAL_MANIFEST=heldout_trend60_taskspecs.jsonl ... scratch_eval_live_throttled.py --label stepN_g64 --conc 3`.
3. **Success** = training grade_success climbing AND held-out >0.867. **Still flat by ~step 70–80** → next lever: reasoning preamble (CoT-before-workflow, needs prompt/SFT work), then the learnability sampler — one change at a time.
4. **Watch**: `metrics/ultra_redundancy_penalty` declining; verify/refine-loop prevalence (#2's false-positive class); code-vs-math split.
5. **Endgame**: select the best checkpoint on the 60-set; confirm ONCE on the sealed `heldout_confirmation_taskspecs.jsonl` — never evaluate it earlier (max-selection bias).
6. **Final verdict**: best checkpoint vs the FULL baseline set (single workers, best-of-N, self-reflection, fixed workflows), then transfer evals (SWE-Bench Pro / TerminalBench / SWE-Together — need harness shims).

### Reference (paths & sets)
- **Base**: `output/fugu_ultra_workflow_sft_qwen3_8b/` (workflow-SFT warm start; repair-SFT / solo bases FORBIDDEN — they give GRPO nothing to grow).
- **Data**: 461-task mix ≈ 55% hard math/reasoning + 39% code — authoritative table + rationale in DATASETS. Agentic benches are EVAL-ONLY.
- **LCB**: `/var/lib/mesh/flavius/huggingface/hub/datasets--livecodebench--code_generation_lite/` (`test.jsonl`=V1 train, `test6.jsonl`=V6 held-out; graded on public + decoded-private tests).
- **Decision sets**: `heldout_trend60_taskspecs.jsonl` (n=60 = original 30 + 15 LCB-V6 + 15 AIME) for every decision; `heldout_confirmation_taskspecs.jsonl` (n=30, SEALED) for the final verdict only.

## Operational Invariants (do not relitigate)

- **Provider routing**: GPT / Opus / Gemini → **Yunwu**; open/specialist (GLM, Kimi, MiMo) → **OpenRouter**. **GPT must NEVER route through OpenRouter.** Live Yunwu calls fail closed unless `ULTRA_ALLOW_YUNWU=1`. Gemini must go through OpenRouter for single-turn (Yunwu ignores `max_tokens` → 13–19k-token reasoning blowups); enforced by `FORCE_PROVIDER={"gemini":"openrouter"}` in `ultra/ultra/providers.py`.
- **Live-worker safety gate**: live mode requires a reviewed safety manifest matching the run's lanes / workers / providers / budget.
- **Harness families available** (`ultra/ultra/harness/`): OpenCode, Codex, Claude-Code, direct QA, code_exec, tool-dialogue, terminal/Harbor, long-context. The single-turn training core uses `single_call` (one `pool.call` + grade) — no containers.
- **Graders (audited, trustworthy)**: math → HuggingFace `math_verify` ($-wrapped gold, gold-first) + normalizer fallback; code → LiveCodeBench public + decoded-private tests; MC → `mc_letter`.
- **Eval baselines** (the objective's comparison set): best individual model+scaffold worker; best commercial worker; best open worker; best fixed workflow; best-of-N single-worker; single-worker self-reflection; prompt-only vs SFT vs GRPO conductor. Always name exact model+scaffold+settings.
- **Frozen manifests**: `director/manifests/fugu_clean_v1/frozen_manifests/freeze_report.json` (online / pool / final / deep_swe eval, hashed). DeepSWE is target/final eval ONLY — never train.

## Condensed Lessons (durable conclusions; the day-by-day handover log was removed — recoverable from git, and the facts live in memory)

- **Verifiability, not difficulty, is the headroom axis** — the load-bearing correction. HEADROOM MAP; [[fugu-ultra-headroom-is-agentic]].
- **Per-task routing within a domain is NULL** (cross-fit p=0.90) → the conductor must DECOMPOSE, not route. [[swesmith-per-task-routing-null]]
- **Coordination/decomposition is the beat-the-best-worker lever** — GLM-draft→Opus-debug 44% vs Opus-solo 36% on SWE (oracle 0.68→0.76). [[coordination-beats-opus-swe]]
- **Live-agentic GRPO is infeasible to train** (Docker + 20-turn workers = call explosion; 0 rollouts / 11 min). Train single-turn test-graded code; agentic is eval-only. [[fugu-ultra-single-turn-rl]]
- **Replay-over-fixed-arms does NOT learn** — free-form conductor outputs miss the arm set → uniform 0.5 reward → zero gradient. Decomposition must be live-graded, not replay-matched. [[replay-grpo-coverage-blocker]]
- **GRPO learns only from within-group reward VARIANCE** — solve-all and fail-all both give zero gradient; difficulty-filter to the learnable band. [[frontier-pool-makes-paper-tasks-easy]]
- **Every prior collapse came from deviating from the recipe** — rpe<16, temp<1.0, lr>1e-6, any KL, a ≤2-step gate, or a solo/repair-SFT base. The recipe table above is non-negotiable.
- **Ultra-track TRAINING pool = Opus-4.8 / GPT-5.5 / Gemini-3.5-Flash / GLM-5.2** (user-confirmed 2026-07-01; per [[pool-complementarity-map]]). Do NOT swap in Gemini-3.1-Pro — that is the PRODUCT-track/benchmark worker; [[keep-gemini-31-pro-not-flash]] is product-track guidance and does NOT govern this training pool. (Flash "dominating" the reasoning probes was a symptom of reasoning having no headroom, not grounds to drop it from the CODE pool.)

## Run Log (chronological — history lives here, not in Current Status)

### 2026-07-01 — group-16 run (steps 0–126): loop, resume crash, early findings
200-step GRPO on the 461-task mix at `batch_size 64 = 4 × 16` (later found to be off-recipe — see 2026-07-02 branch cut). Loop: training continuous (checkpoint every 10), **live held-out eval without stopping** (`scratch_eval_live_throttled.py`, conc-3 against the moving `default` adapter; validated zero disruption) — replaced the pause→eval→resume loop and its ~30-min restart penalty (empty generation pipeline + cold worker cache). Resume-crash fixed: the orch loads step-N weights from `run_default/broadcasts/step_N`, which broadcast cleanup prunes — `scratch_resume_lcb.sh` got a self-healing broadcast rebuild from the checkpoint. Proven end-to-end: conductor emits valid multi-step workflows, `direct_qa` + `code_exec` harnesses grade, GRPO trains cleanly (clean groups, non-zero advantage, no collapse). Refuted two worries: held-out climbed while training reward stayed flat/noisy ~0.58–0.75, and the model is trainable (not a Qwen3-non-thinking ceiling). Steps 20–60 evals were LIVE (moving adapter); base + step-10 frozen-ckpt, same config, comparable. Reward-trajectory chart of this branch: `output/fugu_ultra_lcb/reward_trajectory.png` (mean 0.648, slope ≈ +0.02/run — flat). The `--think` probe is DEAD for this base (workflow-SFT baked in non-thinking; `enable_thinking=True` is a no-op → empty `<think></think>`; a real thinking test needs raw Qwen3-8B).

### 2026-07-01 — Intervention #2 @ step 80: redundancy penalty (reward-side)
Persistent defect in rollouts (steps 14/35/41/63): **N×-near-identical-subtask → aggregate** ensembles — penalized by GRPO but not ground out (weak gradient). Fix in `fugu_ultra_pilot.py`: `_redundancy_penalty(workflow)` — token-Jaccard ≥ 0.75 over stopword-stripped subtasks, −0.1 per redundant step capped at −0.3, preserving reward ordering (correct-redundant ≥ 0.7 > valid-wrong-clean 0.5 > valid-wrong-redundant ≥ 0.2 > unparseable 0.0). Validated on real rollouts: winners/diverse 0.00, 4×-identical loser 0.30, reworded pair 0.10. Observable as `metrics/ultra_redundancy_penalty` (zero-weight rubric metric). Step-80 frozen eval (0.800) = pre-intervention baseline. Post-#2 status through iter 122: firing on 11–17% of workflows, rate softly declining (0.035–0.045 @ s92–95 → 0.003–0.022 @ s113–120), fine-grained gradient visible (s122 reward dist `{1.0:19, 0.8:2, 0.7:1, 0.5:37, 0.4:1, 0.3:2, 0.0:2}`). Training mechanics clean throughout (grad 0.0001–0.003, kl 0.5–0.6, std 0.26–0.30). Known false-positive class: verify/refine loops repeating iteration wording verbatim (≤0.3 capped) — add an iteration-aware exemption if verify-loop prevalence declines.

### 2026-07-02 — Step-120 decision: #2 kept
s120 (n=30) = 0.800, recovered from s100's 0.700 → cleared the ≥0.80 keep-threshold → s100 judged noise (code-only dip, 0.533 on n=15, ~1.2 SE; no damaging mechanism in rollouts — parse ~97%, wf-steps unchanged). The planned n=60 confirmation on the ~s125 policy was stopped mid-flight by the branch cut below and never completed. (The old rollback path — resume `step_00000100` sans penalty — is moot: that checkpoint now lives in the group-16 archive.)

### 2026-07-02 — BRANCH CUT: restart from step 40 on the paper-exact recipe
Trigger: paper re-read + full config audit against Conductor App. A.1 after the 60–120 parity plateau. Findings — the group-16 run was NOT the paper's recipe:
1. **Group size 16 → 64** (`batch_size` 64→256 = 4 × 64, paper-exact). Rare good decompositions need ~64 samples/group to appear and get reinforced; 16 starves the emergence Fig-3 depends on. Prime suspect for the plateau.
2. **Advantage std normalization was MISSING**: the recipe says `A=(r−mean)/std` but the `surogate` default is mean-only — silently ~2-4× smaller gradients than lr 1e-6 was tuned for (consistent with the run-long tiny grad_norms). Fixed: new `std_normalized_advantage` in `surogate/grpo/orchestrator/advantage.py`, wired via `advantage: {type: custom}` in `orch_lcb.yaml`, unit-tested through the exact yaml→config→orch path.
3. **#2 kept** (deliberate paper-deviation, user-ratified): it targets a real defect the paper's conductor self-corrects via CoT-before-workflow — ours emits workflows with no reasoning preamble (the known remaining gap; next lever if group-64+std stalls).
Protocol fixes en route: `max_concurrent` must be ≥ rollouts_per_example (→64, orch crash caught it); `max_inflight_rollouts` 96→384 (1.5× batch); **resume off-by-one fixed in `scratch_resume_lcb.sh`** — on resume at checkpoint N the trainer instantly re-consumed the stale `rollouts/step_N` (double-applying one ~zero-size update) then ignored the orch's regenerated step-N batch entirely (one full generation step wasted; also explains "first step after restart is slow/expensive") — the script now prunes `rollouts/step_≥N` at launch. Restart executed as: archive step>40 artifacts → resume from `step_00000040` (the 0.867 peak; under-powered-gradient hypothesis says its weights are good, just under-trained) → orch relaunched at step 41 against the trainer's step-41 broadcast. Known remaining deviations (assessed, accepted): IPO-mask loss vs PPO-clip (equivalent trust region; one optimizer step per orch step confirmed in `trainer.py`), constant lr vs cosine (≈9e-7 vs 1e-6 in the relevant window; warmup moot on resume), LoRA r16 vs presumed full-FT, 4-worker pool + hard-mix data + difficulty filtering (deliberate track decisions), AdamW β/eps standard (0.9/0.999/1e-8; paper's "eps 0.2" is a typo'd clip-ε), weight_decay 0.01 (paper unstated).
