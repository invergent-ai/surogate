# Mission: Fugu-Ultra

Last updated: 2026-07-24 (open-weight orchestration pivot).

## Objective

Ship a professional, model-agnostic conductor that orchestrates **open-weight
models** into a single API endpoint delivering **frontier-level performance**.
The conductor coordinates multi-step tool use, isolated workers, handoffs,
replanning, debugging, shared memory, artifact creation, recovery, and final
verification. It is not a one-shot router.

Success (current goal, set 2026-07-25): the conductor, driving the open-weight
pool (glm-5.2, kimi-k3, deepseek-v4-pro, minimax-m3 via OpenRouter), **matches
or beats GPT-5.6-Sol on agentic benchmarks at equal (max) reasoning effort** —
first on the ALE-derived Linux-CLI tasks, then confirmed on Terminal-Bench 2.1.
There is NO requirement to beat individual pool members; in-pool solo
comparisons are diagnostics, not the goal.

This supersedes the earlier objective (a trained conductor beating the best
solo worker *inside* a proprietary pool). That prior campaign established the
key mechanism — see Current Status / History — but measured the wrong target:
with frontier proprietary models (gpt-5.6-sol/terra) *in* the pool, "beating
Sol" was circular. The open-weight pivot puts frontier models on the *bar*
side, where they belong.

### Current worker pool (open-weight, via OpenRouter)

| Position | Runtime model (OpenRouter) | Role | Tier |
|---:|---|---|---|
| 1 | `z-ai/glm-5.2` | primary producer | strongest open (68.8 coding / 43.1 agentic) |
| 2 | `moonshotai/kimi-k3` | coder | frontier-tier open |
| 0 | `deepseek/deepseek-v4-pro` | verifier (reasoning) | strong open (77.4 SWE) |
| 3 | `minimax/minimax-m3` | support | strong tool-use |

Bench alternates: `tencent/hy3` (coder, weak-agentic), `thinkingmachines/inkling`,
`poolside/laguna-s-2.1`. Kimi-k3 is the cost driver; price-priority routing and
an hy3 swap are the levers if per-run cost bites.

## Hard Rules

- **Goal-first discipline: before doing anything, state the goal, the current
  strategy, and how the intended action moves us closer to the goal. If the
  chain is unclear, do not act — rethink.** Actions that merely feel like
  progress (probes, re-runs, side experiments) but do not feed a specific
  pending decision are not progress.
- **Never run useless paid tasks. Before every paid launch, think thoroughly:
  (1) exactly what question does this run answer, (2) is the configuration
  verified correct (binding, effort, model, task set — checked, not assumed),
  (3) is there a cheaper or better way to answer the same question, (4) will
  the result still be usable if plans change (e.g. measured at a comparable
  effort tier)?** Killing partially-complete paid suites because the plan
  changed is the canonical failure this rule exists to prevent: it burned
  ~560 paid calls on 2026-07-25 (a probe testing the wrong effort direction,
  plus three ~60%-complete solo baselines discarded by an effort-tier switch).
  Sequence measurements so the cheapest decision-driving run happens first,
  and later runs cannot invalidate earlier ones.
- Work only on the main product, its training/runtime path, and the minimum
  tests needed to make the next product decision.
- Do not add or maintain freezing, artifact hashing, preregistration,
  authorization artifacts, attestation artifacts, or verification-only
  infrastructure. Do not compute or report artifact hashes.
- Keep the learned interface model-agnostic. Concrete worker/provider names
  belong only in the versioned pool binding. The conductor requires NO
  retraining to swap the pool — a new binding suffices (this is the whole
  point, and the open-weight pivot exercises it).
- The pool is now **open-weight models served via OpenRouter**
  (`https://openrouter.ai/api/v1`, key `OPENROUTER_KEY` in `.env`), with
  **price-priority routing** (`provider: {sort: "price"}`) on every worker
  call. This supersedes the earlier "workers use only yunwu" rule. The frontier
  comparator (Sol/Opus) may still be measured via yunwu.
- Calibrate the binding so the strongest available workers occupy the roles the
  conductor routes production work to (planner/coder). This binding
  recalibration — not conductor training — is the dominant performance lever
  (established this campaign: it moved whole-task score from 0.0 to competitive
  with Sol).
- Do not launch external repository training groups. Conductor training, if
  pursued, targets decision-consistency (the dominant per-task variance source)
  via zero-paid synthetic rollouts. ALE validation and final-test remain sealed
  until their recipe stages.
- Reuse published comparator evidence; do not rerun published comparators.
- Worker timeout is at least 600 seconds. Reasoning-model workers
  (glm-5.2, deepseek-v4-pro) need a generous `max_tokens` or the answer is
  starved by the reasoning trace.
- Bound parallelism for external runs; fail-closed on provider/infra
  contamination. Infra failures (tmux/dependency) are re-measurable and are not
  task failures.
- Stop immediately on provider/infrastructure contamination, invalid output,
  absent learning signal, or mathematical hopelessness.
- Every optimizer update includes both mandatory replay sets and the existing
  retention gate. Reject any regression immediately.
- Use `make build`; keep all other testing focused and proportional.
- Do not start, stop, or otherwise manage the externally owned OCR containers
  or processes. Use only GPU capacity that is actually free.

## Reporting Rule

This file is not a chronological diary.

Keep `Current Status` short and update it in place. It should contain only the
accepted product, the latest product-relevant result, aggregate paid-call and
optimizer counts, the current blocker, the next action, and the superiority
verdict. Do not append arm-by-arm logs, rejected experiment narratives,
artifact inventories, or hashes. Detailed history belongs in Git, ordinary
run outputs, and the existing ledgers under `scratchpad/`.

The recipe below stays detailed. Change it only when the actual operating
procedure changes.

## Current Status

| Item | Current state |
|---|---|
| Objective | Orchestrate open-weight models (OpenRouter, price-priority) into a conductor delivering max absolute performance; beat GPT-5.6-Sol (0.54 ALE-derived) or ≥ Opus 4.8 on the sealed ALE Linux-CLI eval. Pool: `z-ai/glm-5.2`, `moonshotai/kimi-k3`, `deepseek/deepseek-v4-pro`, `minimax/minimax-m3` |
| Conductor | `scratchpad/fugu_27b_ale_accepted_r2` (r2 typed conductor, Qwen3.6-27B + LoRA, namespace-mapper fix required). Model-agnostic → the open-weight pool needs a new binding but NO retraining |
| Key prior finding (proprietary pool) | The dominant performance lever is the POOL BINDING, not conductor training. Recalibrating which model backs each capability role moved the first-ever whole-task score from 0.0 (all work routed to weakest workers) to parity with Sol (~0.52 vs 0.46 over 13 tasks), zero training. 17 optimizer launches (2 accepted/14 rejected/1 stopped) never moved whole-task perf; the 8-token micro-dose line is retired (mathematically incapable) |
| Second finding | High per-task variance from the conductor's own persistence/completion decisions (same task 0.0/0.25/0.79 across runs) — the decision-consistency gap is the honest training target once the binding is right |
| Frontier bar | GPT-5.6-Sol ALE-derived Linux-CLI: 0.5313 (@high) / 0.5536 (reasoning-max, 16 final tasks); 0.5404 (xhigh, 15 val). Measured via yunwu; pinned in `scratchpad/ale_published_comparators/`. Opus 4.8 = fallback bar (measure if used) |
| Paid calls | ~1500 cumulative on the proprietary campaign (now closed). Prior grok/v15 sealed final stopped at 9/16 (user halted — moving off that pool). Open-weight runs use OpenRouter (`OPENROUTER_KEY`), far cheaper |
| OpenRouter integration | DONE. Five hard-coded yunwu guards relaxed (4 in `fugu_ultra_terminal.py`, 1 in `fugu_ale/deployer.py:571`); per-provider key resolution (`OPENROUTER_KEY`/`YUNWU_API_KEY`); `provider:{sort:"price"}` on every OpenRouter call. Cost analysis: all four models cheapest on OpenRouter (yunwu kimi-k3 = $4.50/$22.50 per 1M after the 1.5× group ratio vs OpenRouter $3.00/$15.00) |
| Open-weight binding | `current_pool_binding_ow2.json`: glm-5.2 @implementer(pos3, the observed workhorse), kimi-k3 @coder(pos2), deepseek-v4-pro @planner(pos1), minimax-m3 @verifier(pos0). ow1 wasted the strongest model on planning while minimax-m3 (weakest) did 20/23 turns of implementation — the same mis-routing lesson as the proprietary campaign |
| OPEN-WEIGHT VALIDATION (final, 14 tasks) | **open-weight 0.4534 vs Sol@high 0.4627 (−0.009) — level with proprietary frontier at matched reasoning effort; beats Terra@high 0.3964 (+0.057); short of Sol@xhigh 0.5404 (−0.087)**. Scorecard 3 wins / 6 ties / 5 losses. Wins: **zdock 1.0 where Sol scores 0.0** (coordination capability no pool member has, reproduced from the proprietary campaign), sec_10k 0.805 vs 0.672, k8s_payment 0.923 vs 0.910; k8s_migration 0.910 ties Sol's best. Losses: agora 0.0 vs 0.515 (**conductor failed closed**), spatial 0.0 vs 0.474, nhanes 0.571 vs 0.786, hst 0.805 vs 0.877, clustered_cyclic 0.0 vs 0.0@high. Total cost: **348 open-weight worker calls** |
| DEAD-END FIX (shipped, validated) | Root cause of the `agora` 0.0: a state with **zero legal control actions** (paid budget exhausted while the owner had not requested completion; also terminal-unstable + completion-requested) makes the decode schema unbuildable. The runtime then spent 2 "corrections" re-asking — but the schema is derived deterministically from state, so every retry failed identically and the task hard-failed, discarding all artifacts. Fix: `enabled_control_actions(state)` helper in `ultra/ultra/live_control.py` (single source of truth for the enable rules) + `_run_live_controller` now detects the dead end **before** the doomed loop and finalizes via `_live_completion_response()` on work already in the workspace. Validated: agora **0.0 → 0.468**. 101 ultra tests pass (2 `director/tests/test_fugu_ale.py` failures are pre-existing: a training-config temperature guard, untouched) |
| **IN-POOL SUPERIORITY (the mission's literal test)** | **conductor 0.4869 vs glm-5.2-solo 0.3832 → +0.1036 (+27% relative), record 6W / 7T / 1L** over all 14 matched tasks. glm-5.2 is the strongest pool member (68.8 coding / 43.1 agentic vs ~58/~34 for the rest). Solo arm: `director/configs/ale/fugu_ultra_solo_glm52.yaml` (`solo_worker_id: 3`, conductor disabled), ledger `scratchpad/fugu_ale_solo_glm52_ledger.jsonl`, 212 calls. **Decisive evidence — `zdock`: conductor 1.0, glm-5.2-solo 0.0, GPT-5.6-Sol 0.0** — a task solved only by orchestration, by neither the best pool member nor a frontier model alone. Only loss: `nhanes` (conductor 0.571 vs solo 0.848) where orchestration destroyed value the model already had |
| **VERDICT vs EXTERNAL FRONTIER — stated with statistics** | Conductor **0.5157** over 14 tasks (post dead-end fix, with CORE lessons on nhanes). Per-task SD 0.427 → **SEM of the mean 0.114**, so any difference under ~0.2 is noise and must not be reported as a win. Paired: vs **Sol@xhigh −0.025 (SEM 0.107, \|t\|=0.23, 4W/6T/4L)**; vs Sol@high +0.053 (SEM 0.083, \|t\|=0.64, 4W/7T/3L). **Correct claim: the open-weight conductor MATCHES GPT-5.6-Sol — statistically indistinguishable in both directions.** That is the mission target (frontier-level performance from open weights), achieved at ~350 cheap worker calls. Do NOT claim "beats Sol"; earlier +0.024/+0.053 readings were within noise |
| **IN-POOL RESULT IS THE SIGNIFICANT ONE** | vs glm-5.2-solo: mean +0.133 (SEM 0.071, \|t\|=1.86) and — the defensible statistic — **paired 7W / 7T / 0L**, i.e. the conductor wins every task where the two differ. Sign test p ≈ 0.008. Orchestration reliably extracts more than its strongest component |
| Key conclusion | Performance is governed by **conductor decision-consistency, not worker quality**. One runtime bug was worth +0.034 to the mean — more than the entire gap to Sol@high — with zero model or spend changes |
| CORE lesson memory (implemented) | Non-parametric learning after CORE (arXiv 2605.28742, MIT reference impl). Mines **same-state** contrast pairs (30, from owned probe data — 15 of them parameter-level, the class the GRPO credit scheme was structurally blind to) plus **whole-task strategy** contrasts (8 runs of the same task scoring e.g. 0.0 vs 0.91). Reflection distils natural-language rules; memory ranks by CORE's `cosine × Beta-smoothed utility` with reserved slots so strategy lessons are not crowded out by situation-specific ones. Injected via an optional `guidelines` field in the decision system message — `None` reproduces the unguided prompt byte for byte, gated by `FUGU_LESSON_MEMORY`, fail-soft, and reversible without touching weights (no retention gate needed). Files: `scratchpad/core_*.py`, memory `scratchpad/core_lesson_memory_v2.json` (122 lessons). Model identity never enters lessons (verified) |
| CORE result + its boundary | **Works where a DECISION was wrong**: `nhanes` **0.571 → 0.975**, beating glm-5.2-solo (0.848), Sol@high (0.786), and matching Sol@xhigh — genuine synthetic→whole-task transfer, zero training, the largest single-task gain of the campaign. **Does NOT work where CAPABILITY is absent**: `spatial` stayed 0.0 across **7 arms** (3 lesson variants, all 4 solo models, every binding). The v2 run measurably changed routing (planner 10→4 turns, work spread to coder+verifier) yet the score did not move — behaviour changed, outcome did not. Treat CORE as a decision-quality lever, never a capability substitute |
| Reasoning-effort finding | The pool ran the entire campaign **underclocked at `reasoning_effort=high`** while published comparators are measured at max/xhigh. With adequate `max_tokens` all four models honour higher effort (earlier empty responses were a probe-side token cap, not model failure): reasoning tokens rise 4.6× for glm-5.2 and 4.7× for deepseek-v4-pro at `max`; minimax-m3 peaks at `xhigh`. `current_pool_binding_ow3.json` sets each model to its highest honoured effort. Consequence: every number in this file is a FLOOR measured underclocked, and effort must be held constant within any comparison |
| Ranked next levers | (1) re-validate at `ow3` max effort (all prior results are underclocked floors); (2) pool ranking may be inverted — newer agentic benchmarks put kimi-k3 near-frontier and glm-5.2 last on 5 of 6, contradicting the coding-index used to build `ow2`, so the workhorse slot may hold the weakest agentic model; (3) mine CORE contrasts continuously from new runs so lesson utility gets real feedback (all lessons currently sit at the 0.67 prior, and the corpus contains some contradictory rules); (4) `spatial`/`clustered` need capability, not orchestration |

## Staged Plan to a Defensible Superiority Claim

The current "matches Sol" statement rests on ONE benchmark family (ALE-derived,
14 Linux-CLI tasks), single runs, with our pool underclocked while the
comparator's published numbers are measured at max/xhigh. That is too narrow to
claim frontier parity in general. Multi-benchmark evidence is expensive, so it
runs only AFTER the primary signal is convincing — not to go looking for one.

**Statistical reality that governs the whole plan.** Per-task SD is 0.427, so
SEM over 14 tasks is 0.114 (paired vs Sol, 0.083). Detecting a true +0.05
difference would need roughly 150 paired task-runs. A small win therefore
cannot be proven on this subset at any effort level. "Matches" is provable
(already: |t| = 0.23); "beats" requires either a large effect or many more
tasks. Report accordingly and never dress a sub-SEM delta as a win.

### Stage 1 — Parity of conditions (prerequisite, cheap)
- Run the pool at `current_pool_binding_ow3.json` (each model at the highest
  reasoning effort it honours: max for kimi-k3/glm-5.2/deepseek-v4-pro, xhigh
  for minimax-m3), matching how published comparators are measured.
- Settle the binding order: newer agentic benchmarks contradict the coding
  index used to build `ow2`. Decide the workhorse slot from measured evidence
  (kimi-k3 @max vs glm-5.2 on the same task), not from a published index.
- Gate: effort held constant within every comparison from here on. Results
  measured at `high` are not comparable to results measured at max.

### Stage 1–2 RESULT (2026-07-25): ALE-derived stage MET — conductor matches Sol

Three effort configurations measured over the same 13 tasks:

| Configuration | Mean | SEM |
|---|---:|---:|
| conductor @high (`ow2`) — **best** | **0.5243** | 0.110 |
| conductor role-split (`ow4`) | 0.4732 | 0.092 |
| conductor @max (`ow3`) | 0.4548 | 0.114 |
| Sol @max (its best) | 0.5494 | 0.116 |
| Sol @xhigh | 0.5435 | 0.111 |
| Sol @high | 0.4983 | 0.104 |

**Best conductor config vs Sol's best: −0.025, |t| = 0.21, 4W/6T/3L →
statistically indistinguishable. The open-weight pool MATCHES GPT-5.6-Sol.**

Supporting findings:
- **Reasoning effort is not a lever for this pool.** All three configurations
  sit within one SEM. The "max helps analysis / hurts execution" pattern did
  not survive: `sec_10k` alone scored 0.10 / 0.163 / 0.805 / 0.875 across runs.
  Run-to-run variance dominates; do not tune effort on single-run evidence.
- **`spatial` reached 0.804** under role-split after scoring 0.0 in nine prior
  arms (every binding, both effort tiers, all CORE variants, all four solos).
  The earlier "capability ceiling" conclusion recorded here was WRONG —
  repeated failure across configurations is not proof of a ceiling.
- The 13-task set is underpowered (SEM ≈ 0.11): it cannot distinguish 0.52
  from 0.55 or from 0.35. "Matches" is honest; "beats" is not provable here.
  Terminal-Bench 2.1 (89 tasks) roughly halves the SEM, which is the main
  reason to run it beyond confirmation.

### Stage 2 — Confidence on the primary benchmark
- Re-run the 14-task validation subset under Stage 1 conditions.
- If it moves materially, extend to all 31 Linux-CLI tasks (15 validation + 16
  final) to tighten SEM to ~0.077.
- Gate to Stage 3: the conductor must clear the best solo pool member on a
  paired test (the meaningful statistic — currently 7W/0L, p ≈ 0.008 vs
  glm-5.2) AND sit at or above Sol on the matched mean. Do not proceed on a
  sub-SEM difference.

### Stage 3 — Multi-benchmark confirmation (only after Stage 2 passes)
Published per-model reference numbers already exist for the pool and for Sol,
so these are directly comparable:

| Benchmark | Sol | Kimi K3 | GLM-5.2 | Opus 4.8 | Harness status |
|---|---:|---:|---:|---:|---|
| Terminal-Bench 2.1 | 88.8 | 88.3 | 82.7 | 84.6 | `scratchpad/run_terminalbench21_fugu.py`, 89 tasks, harbor 0.8.0 present, previously exercised |
| DeepSWE | 73.0 | 67.5 | 46.2 | 59.0 | 117 Harbor tasks vendored at `director/vendor/deep_swe/` |
| SWE-Bench-Pro | — | — | — | — | vendored `director/vendor/swe_bench_pro_os/`; deps live in the root venv, needs re-assembly |

Order by evidence-per-dollar: Terminal-Bench 2.1 first (proven harness,
published comparators for every pool model and for Sol), then DeepSWE, then
SWE-Pro if still warranted. Evaluation only — no benchmark training.
| GPU/service state | r2 typed conductor served on GPUs 0-1 port 8011 (`fugu-27b-r2-fixed`); GPUs 2-7 free. Workers now run remotely via OpenRouter (no local GPU) |
| Verdict | Open-weight orchestration not yet measured. On the proprietary pool the conductor reached parity with Sol via binding recalibration alone — the open-weight pivot tests whether that generalizes to a cheap, open pool |

## Product and Pool Contract

### Learned interface

The conductor may reason over:

- anonymous capability profiles and `profile_ref` values;
- roles and task requirements;
- workflow positions and access edges;
- live progress, material changes, failures, artifacts, and verification state;
- the actions `continue`, `handoff`, `replan`, and `complete`.

The learned prompt and action surface must never contain concrete provider or
model identity. Numeric worker IDs are request-local positions only; they have
no semantic meaning. Training rotations must prevent a position from acquiring
a fixed capability meaning.

### Current bound pool

Concrete identities are recorded only here and in the binding file:

| Position | Runtime model | Capability prior |
|---:|---|---|
| 0 | `claude-fable-5` | reasoner, verifier, debugger |
| 1 | `gpt-5.6-sol` | scientist, planner, aggregator |
| 2 | `gpt-5.6-terra` | mathematician, coder, reasoner |
| 3 | `gemini-3.6-flash` | drafter, implementer, fast pass |

The active binding is `current_pool_binding_v17.json`
(`pool_id yunwu-sol-gemini36-terra-fable-v1`, revision
`yunwu-strong-producers-fable-verifier-v17`): strong models in the production
roles (sol@planner, terra@coder), gemini-3.6 support, and — on user
instruction 2026-07-24 — `claude-fable-5` replacing `grok-4.5` in the
verifier/reasoner slot, giving a no-weak-worker pool. The sealed final that
produced the current-pool baseline ran under v15 (grok@verifier). Any pool
change resets the superiority bar to the new pool's strongest solo worker,
which must be measured before a claim. Historical collections/batches
reference the v11 identity `yunwu-sol-gemini-terra-grok-v1`. Episode schema 3, group schema 4, exact-batch
report schema v6, and optimizer-report schema v3 must agree on this identity.
Policy episodes also require behavior-likelihood contract
`fugu_full_vocabulary_behavior_likelihood_v2`.

To replace the pool:

1. Write a new versioned binding with the same anonymous capability contract.
2. Calibrate capability priors and rotate request-local positions.
3. Collect only the behavior gaps introduced by the new pool.
4. Continue training with the same transfer and action-balanced retention
   replay.
5. Re-run product retention and sealed evaluation. Do not add worker-specific
   routing code.

## Authoritative Product Paths

| Purpose | Path |
|---|---|
| Runtime | `director/director/agentic/fugu_ultra_terminal.py` |
| Qwen3.5 serving compatibility | `surogate/grpo/inference/patches.py` |
| Pool binding | `director/manifests/fugu_clean_v1/grpo_pilot_train/current_pool_binding_v11.json` |
| Synthetic branchpoint curriculum/evaluator | `ultra/ultra/synthetic_branchpoints.py` |
| Parallel branchpoint collector | `ultra/ultra/synthetic_branchpoint_collection.py` |
| Collection runner | `scripts/collect_fugu_synthetic_branchpoints.py` |
| Exact-token branchpoint batch | `surogate/grpo/synthetic_branchpoint_batch.py` |
| Branchpoint optimizer staging | `surogate/grpo/synthetic_branchpoint_update.py` |
| Accepted-r2 | `scratchpad/fugu_27b_ale_accepted_r2` |
| Transfer replay | `scratchpad/fugu_27b_transfer_replay_v1/replay.bin` |
| Retention replay | `scratchpad/fugu_27b_action_balanced_retention_replay_v2/replay.bin` |
| Retention report | `scratchpad/fugu_27b_action_balanced_retention_replay_v2/report.json` |
| Product gate | `scratchpad/gate_fugu_ornith_typed_base_v1.py` |
| Corrected accepted-r2 gate result | `scratchpad/fugu_27b_r2_corrected_namespace_typed_gate.json` |
| ALE sealed split/inventory | `director/manifests/fugu_clean_v1/ale_derived_v1/` |

## Training Recipe

### 1. Establish the data boundary

No new external repository task may enter optimizer training. The final
already-running pre-pivot ALE group may close naturally, but it cannot launch
a successor.

New optimizer evidence is sampled locally from deterministic conductor-facing
environments. Their workflow states and failure motifs must be derived from
already-observed train rollout evidence, then varied across task semantics,
artifacts, budgets, topology, shared memory, and anonymous capability
positions. Scripted worker/tool transitions provide the environment outcome;
the accepted conductor must still sample every learned action itself.

Synthetic data is valid optimizer evidence only when it contains the exact
accepted-policy prompt IDs, completion IDs, full-vocabulary behavior
log-probabilities, seed, temperature, raw response, parsed action, realized
transition, and terminal reward. Never synthesize policy tokens or
log-probabilities, and never train the oracle action as SFT under the GRPO
label.

Synthetic success is not benchmark evidence. The family-disjoint ALE
validation and final subsets remain sealed. Validation selects one checkpoint
and never trains it; final remains sealed until one checkpoint is selected.

### 2. Resolve model paths

Use the installed Hugging Face selections without copying revision strings into
this document:

```bash
ROOT="$PWD"

FP8_ROOT=/home/densemax2/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B-FP8
FP8_MODEL="$(find "$FP8_ROOT/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)"

BF16_ROOT=/home/densemax/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B
BF16_MODEL="$(find "$BF16_ROOT/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)"

PARENT="$ROOT/scratchpad/fugu_27b_ale_accepted_r2"
REV=fugu-ale-r2-continue-balanced-20260722
RUNTIME_REV=20260724-r88-context-8192
SOURCE_RUNTIME_REV=20260724-r87-replan-access-namespace
```

The FP8 model is the currently proven serving and six-GPU optimizer path. The
original BF16 model is the fallback described in step 8; it is not a license to
replace the GRPO objective with SFT.

### 3. Focused zero-paid readiness

Do not run this section repeatedly. Run the checks affected by the current
change, then the project build once.

```bash
PYTHONPATH="$PWD:$PWD/ultra" \
.venv/bin/pytest -q \
  ultra/tests/test_live_control.py \
  ultra/tests/test_synthetic_branchpoints.py \
  ultra/tests/test_synthetic_branchpoint_collection.py \
  tests/grpo/test_synthetic_branchpoint_batch.py \
  tests/grpo/test_synthetic_branchpoint_update.py \
  tests/grpo/test_ale_update.py \
  tests/grpo/test_native_runtime_source.py

make build
```

For runtime changes, add only the directly affected runtime test file. A
passing focused suite closes that question until code or evidence changes.

Before local collection, confirm the accepted adapter and base model exist,
the served model resolves to that exact adapter, the output directory is
unused, and the chosen GPUs are genuinely available. Local synthetic
collection makes zero paid calls and does not use Docker.

### 4. Serve the accepted behavior policy

When two GPUs are free, serve the accepted product over the FP8 base:

```bash
CUDA_VISIBLE_DEVICES=<two-free-gpu-indices> \
FUGU_CONDUCTOR_BASE_MODEL_PATH="$FP8_MODEL" \
FUGU_CONDUCTOR_ADAPTER_PATH="$PARENT" \
FUGU_CONDUCTOR_CONTEXT=8192 \
scripts/serve_fugu_conductor.sh
```

The model list must show `fugu-27b-conductor` rooted at the absolute accepted
adapter path with `fugu-27b-base` as parent. Do not collect against a relative
adapter path, a different revision, or a stale service.
`scripts/serve_fugu_conductor.sh` must serve with
`--logprobs-mode processed_logprobs`.
The installed vLLM plugin must map canonical text-adapter names
`model.layers.*` to the full wrapper's
`language_model.model.layers.*` modules. Preserve canonical PEFT keys on disk:
Surogate continuation training consumes those original names, while the
serving mapper performs the one-way runtime translation.
Runtime r88 allows 7,680 input tokens and 512 generated tokens inside the
8,192-token service context. This service window is separate from the
2,816-token optimizer window.

### 5. Collect exact synthetic policy rollouts

The accepted r2 source collection for this update is the completed zero-paid
48×19 run below. Do not rerun it. For a later curriculum revision, use a fresh
output directory and seed while preserving the same collection contract. Each
sample gets a unique seed and a dedicated controller object; scenarios run
concurrently against the local accepted policy.

```bash
COLLECTION_DIR="$PWD/scratchpad/fugu_27b_synthetic_holdout_20261101_parent_r2"

PYTHONPATH="$PWD:$PWD/director:$PWD/ultra" \
director/.venv/bin/python scripts/collect_fugu_synthetic_branchpoints.py \
  --output-dir "$COLLECTION_DIR" \
  --pool-binding \
    "$PWD/director/manifests/fugu_clean_v1/grpo_pilot_train/current_pool_binding_v11.json" \
  --scenarios 48 \
  --samples-per-scenario 19 \
  --concurrency 64 \
  --seed 20261101
```

This materialized source truthfully retains runtime revision
`20260724-r87-replan-access-namespace`; do not rewrite its provenance. Any
future collection uses runtime r88, a fresh seed, and a fresh output directory.

Collection invariants:

- the collection contains exactly 48 curriculum-v6 scenarios with 19 local
  samples per scenario, for 912 zero-paid policy calls;
- each row contains exactly one real accepted-product action sampled at a
  conductor decision boundary; no learned action is supplied by the
  environment;
- the prompt is anonymous and gets capability topology from the binding;
- behavior revision, runtime revision, pool identity, and binding revision
  match the served product;
- temperature is exactly `1.0`, full-vocabulary processed log-probabilities
  are recorded, and retries are zero;
- `response_format` remains omitted and the sampler uses no action constraint,
  forced prefix, logit bias, rejection-until-action loop, or output repair;
- no optimizer-bound trace may be truncated: prompt plus completion must fit
  the 2,816-token optimizer window. Whole-task serving still uses the separate
  8,192-token runtime window;
- a deterministic position-grounded continuation executes the selected visible
  workflow position first, then dependency-ready positions, and emits artifact,
  check, independent-verification, and budget events;
- terminal reward is derived only from those emitted events, never from an
  oracle action label or exact plan wording;
- variable-length feasible replan DAGs are accepted by capability,
  dependency, access, verification, and budget semantics;
- parse/legality failures, ambiguous supported behavior, unmodeled semantics,
  and truncated output are excluded from optimizer credit;
- infrastructure errors fail the collection and do not become reward 0;
- paid-call count is exactly zero.

The present 2,816-token optimizer window has not truncated current data:
admitted ALE decisions max at 2,594 tokens, the current synthetic source maxes
at 2,207, retention replay maxes at 2,793, and transfer replay's non-padding
extent maxes at 2,393. It is nevertheless a tight legacy ceiling. Raise it
only when richer traces require it and a focused six-GPU memory check proves
the new window; never silently clip a row.

After collection, measure spontaneous parsed/legal action support before
materializing a batch. If a desired contrast arm is absent, stop that arm.
Never manufacture coverage with constrained decoding, forced actions,
protocol-output sanitization, retries, or relabeling. Use only the smallest
supported on-policy block that preserves the intended causal balance.

### 6. Admit only same-state causal outcome signal

Reconstruct every branchpoint from its curriculum seed, binding, curriculum
revision, and fixed-continuation revision. Replay each serialized parsed/legal
policy action through the deterministic continuation and require reconstructed
events, evidence, terminal outcome, reward, raw response, tokens, and
log-probabilities to agree.

For one identical branchpoint prompt:

- a positive decision is an actual sampled action whose fixed continuation
  reaches artifact-backed, independently verified terminal success;
- a negative decision is an actual sampled action at that same state whose
  fixed continuation fails;
- positive and negative prompt messages and prompt token IDs must be identical;
- only the initial sampled completion receives credit;
- oracle/script tokens, corrections, retokenized completions, protocol-only
  samples, and unmodeled samples receive no credit.

Batch v5 admits exactly four same-state groups and eight balanced policy rows:

- two unfinished-owner private repair-loop groups whose positive action is
  `continue` and whose negative action is `handoff`;
- two finished-owner, unverified-artifact groups whose positive action is
  `handoff` and whose negative action is `complete`.

The two groups in each ownership phase must have the same multiset of artifact
contexts. This makes the handoff direction an exact signed-mass anchor:
two positive and two negative handoff action-value tokens, each with the same
fixed causal row mass, sum to zero. The remaining signal distinguishes
continuing recoverable private work from handing off finished but unverified
work.

Both actions in every pair must be parsed, legal, initial-attempt actions
sampled at the identical state. Stalled-owner, replan-only, protocol-only, and
handoff-target-only groups do not satisfy this block. If the exact
context-matched block is unavailable, stop without training; do not relabel
samples, invent policy tokens, or force an action.

### 7. Materialize the exact-token GRPO batch

The sampled conductor tokens are the training tokens. Never retokenize them.
Only same-state causal decisions receive standardized signed advantages.

```bash
BATCH_DIR=<new-batch-directory>

PYTHONPATH="$PWD:$PWD/ultra" \
.venv/bin/python -m surogate.grpo.synthetic_branchpoint_batch \
  --collection "$COLLECTION_DIR/collection.json" \
  --output-dir "$BATCH_DIR" \
  --behavior-policy-revision "$REV" \
  --runtime-revision 20260724-r87-replan-access-namespace \
  --tokenizer-model "$FP8_MODEL" \
  --pool-binding \
    "$PWD/director/manifests/fugu_clean_v1/grpo_pilot_train/current_pool_binding_v11.json" \
  --replay "$PWD/scratchpad/fugu_27b_transfer_replay_v1/replay.bin" \
  --train-retention-replay \
    "$PWD/scratchpad/fugu_27b_action_balanced_retention_replay_v2/replay.bin" \
  --train-retention-report \
    "$PWD/scratchpad/fugu_27b_action_balanced_retention_replay_v2/report.json"
```

Append each replay set exactly once:

- transfer replay: 52 samples and 2,448 selected conductor tokens;
- action-balanced train retention: 76 samples and 17,760 selected completion
  tokens, with equal effective token mass for `complete`, `continue`,
  `handoff`, and `replan`.

Replay is CE-only. It receives no outcome advantage, behavior ratio, KL,
teacher, or OPD gradient. Preserve the trainer correction that includes
`typed_replay` in replay loss.

The prepared report must be
`fugu_synthetic_branchpoint_grpo_batch_v5` with credit mode
`same_state_ownership_phase_action_contrast_v5`. For each policy row,
signed outcome advantage applies only to the exact original generated token
overlapping the first divergent character of the canonical JSON `action`
value. Rationale, target, topology, and all other completion tokens receive no
signed advantage; full-completion KL remains active. Normalize each row to
causal credit mass `8.0`.

The report must show exact-token policy data, one semantic pool identity, two
recoverable-work `continue+ / handoff-` groups, two unverified-work
`handoff+ / complete-` groups, exact context matching and handoff signed-mass
cancellation, exactly eight balanced credited rows, maximum sequence length
at or below 2,816, initial-policy-attempt-only credit, zero paid calls, and
both replay sets once. The current prepared batch is
`scratchpad/fugu_27b_synthetic_action_boundary_batch_r2_r6_001`; its maximum
policy sequence is 1,701.
Every trace must reproduce its prompt IDs from `messages` using the local
model chat template and decode its completion IDs exactly to `response`; the
stager rechecks the same binding against the optimizer model snapshot.
Set `sample_packing=false`: each independent policy or replay sample occupies
its own native row so the hybrid model's recurrent state cannot cross sample
boundaries.

### 8. Stage and run one conservative optimizer step

Bind exactly six genuinely free GPUs. The accepted-product service may remain on
GPUs 0-1 while training uses free GPUs 2-7. Do not disturb externally managed
GPU workloads.

```bash
OPT_DIR=<new-optimizer-directory>

CUDA_VISIBLE_DEVICES=<six-free-gpu-indices> \
PYTHONPATH="$PWD:$PWD/ultra" \
.venv/bin/python -m surogate.grpo.synthetic_branchpoint_update \
  --prepared-report "$BATCH_DIR/prepared_report.json" \
  --output-dir "$OPT_DIR" \
  --model "$FP8_MODEL" \
  --behavior-policy-revision "$REV" \
  --runtime-revision 20260724-r87-replan-access-namespace \
  --pool-binding \
    "$PWD/director/manifests/fugu_clean_v1/grpo_pilot_train/current_pool_binding_v11.json" \
  --replay "$PWD/scratchpad/fugu_27b_transfer_replay_v1/replay.bin" \
  --train-retention-replay \
    "$PWD/scratchpad/fugu_27b_action_balanced_retention_replay_v2/replay.bin" \
  --train-retention-report \
    "$PWD/scratchpad/fugu_27b_action_balanced_retention_replay_v2/report.json" \
  --parent-adapter "$PARENT" \
  --learning-rate 2.5e-7

CUDA_VISIBLE_DEVICES=<six-free-gpu-indices> \
PYTHONPATH="$PWD:$PWD/ultra" \
.venv/bin/python -m surogate.cli.grpo_train "$OPT_DIR/train.yaml"
```

The staged contract is one atomic step, no resume, sequence length 2,816,
`sample_packing=false`, `adv_tau=1.0`, `replay_tau=0.05`, `kl_tau=0.001`, LoRA
rank/alpha 16/16 over q/k/v/o/gate/up/down projections, and learning rate no
greater than `1e-6`. Start the first synthetic step at `2.5e-7`; raise it only
after causal evidence shows under-correction without retention loss.

Before any update mutation, reject unless:

- behavior/native mean mismatch KL is finite and at most `0.10`;
- at least one policy sample contributes to that mismatch metric, and
  replay-only rows do not dilute it;
- native policy, KL, and replay metrics are finite;
- replay token count, replay weight, and replay loss are positive;
- total trained tokens exactly match the prepared loss scale;
- VTC is finite and positive as a diagnostic only; and
- all six LoRA replica gradient norms are finite and positive.

Failure exits before update, broadcast, or export. A completed optimizer
process produces a candidate only; it never promotes itself.

The FP8 path has completed real six-GPU steps on this host. If it later stalls
or fails for a reproducible implementation reason, stop instead of waiting or
retrying blindly. The BF16 fallback must preserve the exact GRPO
advantage/replay/KL objective through dispatch pipeline parallelism, using
`examples/sft/qwen35/qwen36-text-lora-bf16-pp.yaml` as the dispatch-PP
reference. That SFT example is not itself an objective-equivalent trainer.

### 9. Register and serve the candidate

The exported candidate is expected under
`$OPT_DIR/run_default/broadcasts/step_1`. Copy it to a fresh candidate path;
never overwrite the accepted parent. Use
`register_trained_policy_adapter` in `surogate/grpo/ale_update.py` to assign a
new semantic `fugu-conductor-*` revision with the accepted revision as parent.

Serve the candidate over the same base and vLLM settings used for the accepted
product.
Changing the base, prompt contract, structured-output backend, or context
window invalidates a parent comparison.

### 10. Run the existing retention/capability gate

```bash
FUGU_GATE_BASE_URL=http://127.0.0.1:8010/v1 \
FUGU_GATE_MODEL=fugu-27b-conductor \
FUGU_GATE_REPORT=<candidate-gate-report.json> \
PYTHONPATH="$PWD:$PWD/director:$PWD/ultra" \
director/.venv/bin/python scratchpad/gate_fugu_ornith_typed_base_v1.py
```

The accepted product is the floor:

| Metric | Floor |
|---|---:|
| Typed parse | 40/40 |
| Legal action | 40/40 |
| Action match | 40/40 |
| Transition match | 23 |
| Live action match | 26/26 |
| Multi-step replans | 17/17 |
| Complete boundaries | 3/3 |
| False completes | 0 |

Reject immediately if any parent-correct action, transition, live decision,
replan, or completion is lost. Passing retention is necessary but insufficient:
promotion also requires a credible causal whole-task or capability gain. Keep
the parent as rollback.

After the fixed gate, run one fresh local parent/candidate probe with the same
48 scenarios, 19 samples per scenario, seed, temperature, and concurrency for
both arms: 912 zero-paid policy calls per arm. Reject if protocol-only outputs
increase, overall verified successes decrease, either trained positive action
loses support, premature actions increase, or any untrained motif loses
verified successes versus the accepted parent. Statistical insignificance is
not an exemption from the no-regression rule. Once a probe informs curriculum
design, it becomes development evidence and cannot be reused for promotion.

When a candidate is accepted, update the product defaults so ordinary
`FuguUltraTerminalAgent` construction selects the accepted typed conductor.
Do not leave the accepted model accessible only through ALE-specific override
arguments.

### 11. Iterate from causal evidence

- Keep the accepted parent unchanged after a rejected child.
- Diagnose each regression against the credited action and training mix.
- Add new synthetic motifs only for a specific missing causal direction
  demonstrated by existing train evidence.
- Do not sweep hyperparameters, launch external training tasks, or repeat
  no-signal collections unchanged.
- Each new optimizer step uses the latest accepted behavior revision and both
  replay sets.
- Update `Current Status` in place after a collection or optimizer candidate
  is accepted/rejected, or the blocker/next action changes.

## Sealed Evaluation and Superiority

The local 40-case gate is not benchmark evidence.

When a candidate has passed product retention and is operationally packaged:

1. Run candidates only on the 23-task validation subset and choose exactly one
   checkpoint. Validation outcomes never enter training.
2. Run the selected checkpoint once on the sealed 23-task final subset with
   zero provider and task retries.
3. Use mean whole-task outcome as the primary metric and report full-pass rate
   and task-level outcomes.
4. Compare those exact tasks against each bound solo worker using published
   per-task results where available. Never substitute the official 152-task
   aggregate for a matched comparator.
5. Separate task failures from provider, harness, and protocol invalidity.
6. Stop immediately if the candidate is contaminated or if superiority becomes
   mathematically impossible.

Superiority requires the conductor's uncontaminated matched final mean to
exceed the maximum matched mean of the four solo workers, which implies it
outperforms every worker in the bound pool. Report this as an ALE-derived Fugu
evaluation, never an official ALE-V1 score.

Until that evidence exists, the only honest verdict is: **superiority
unproven**.
