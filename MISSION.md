# Mission: Fugu-Ultra

Last updated: 2026-07-26 (topology-training pivot: make collaboration pay).

## Objective

Ship a professional, model-agnostic conductor that orchestrates **open-weight
models** into a single API endpoint delivering **frontier-level performance**.
The conductor coordinates multi-step tool use, isolated workers, handoffs,
replanning, debugging, shared memory, artifact creation, recovery, and final
verification. It is not a one-shot router.

Success (current goal, set 2026-07-26): a **trained router whose multi-step
workflows BOOST accuracy over solo workers on every task type**. "Collaboration
doesn't pay on this task type" is not an acceptable conclusion — it is the
problem the training must solve (user directive 2026-07-26). Target audience is
**general office/personal-use agents, not coding-first** — but the assistant
needs a strong terminal/agentic score too, since most office tasks touch the
terminal. Performance bar: **Opus-4.8 / GPT-5.6-Terra level** (Sol parity is
explicitly NOT the bar for this pool). Production cost matters: the router must
show real benchmark improvements at open-weight prices to give people a reason
to use it.

The product is the model-agnostic Fugu-style router. Beating individual pool
members is now IN scope — that lift over the best solo worker is precisely the
value a trained conductor adds (Fugu's own GPQA result: 95.5 via trees vs best
worker 94.3).

**Where the gap actually is (measured 2026-07-25).**

*Terminal-Bench 2.1, full product measurement (2026-07-26)*:
**product (r2 + static guidance) 61.0% (50/82, SEM 5.4pp) vs unguided baseline
60.7% (51/84) vs Sol 88.8.** The guidance produced 13 recoveries and 12
regressions (sign test p = 1.0) — pure churn, no aggregate effect; timeout
failures identical (16 vs 17). The earlier A/B that showed "4/8 recoveries"
was one-sided by construction (it re-ran only failed tasks and could not
observe regressions on passing ones); do not repeat that design.

**Where the ~28-point gap to Sol actually lives:** not in conductor decisions
— prompting cannot move the aggregate — and not primarily in pool capability
(kimi-k3 scores 88.3 on this benchmark in its NATIVE scaffold). Solo kimi
routed through OUR harness also fails most hard tasks. The gap is dominated by
the **worker-level terminal scaffold**: our batch-command worker loop versus
the mature native agent loops (Terminus 2 / Claude Code class). The
highest-value work is scaffold quality — the worker interaction loop (command
batching, polling, recovery, context management) — not more conductor
prompting or posture training.

*The ALE-derived "match" does NOT support an equivalence claim.* The paired
difference vs Sol@max is −0.023 with a **95% CI of [−0.237, +0.191]** — the
data is equally compatible with the conductor being 24 points worse or 19
points better. |t| = 0.21 is a failure to detect, not evidence of parity: with
n = 14 and per-task SD 0.41 the test could only ever resolve differences larger
than ~0.21. Resolving a 5-point difference would need ~257 task-runs (≈20
repeats per task). The comparison is also cross-harness (our staged ALE runs vs
the official leaderboard's Sol scores) and single-run per task on a benchmark
where the same task has scored 0.0/0.25/0.68/0.91 across repeats. Do not cite
the ALE result as "matches Sol"; cite it as "no gap detectable at this power".

This supersedes the earlier objective (a trained conductor beating the best
solo worker *inside* a proprietary pool). That prior campaign established the
key mechanism — see Current Status / History — but measured the wrong target:
with frontier proprietary models (gpt-5.6-sol/terra) *in* the pool, "beating
Sol" was circular. The open-weight pivot puts frontier models on the *bar*
side, where they belong.

### Current worker pool (Pool C, general-use, via OpenRouter)

Active binding: `current_pool_binding_general.json`
(`pool-c-general-use-20260726`).

| Slot | Runtime model (OpenRouter) | Role prior | Why |
|---:|---|---|---|
| 0 | `deepseek/deepseek-v4-pro` | aggregator/verifier/knowledge/tool_user | SimpleQA 84.4, MMLU 94.8, BFCL 82.7 |
| 1 | `minimax/minimax-m3` | reasoner/scientist/long_context | GPQA 92.7, cheap |
| 2 | `z-ai/glm-5.2` | debugger/analyst/hard_reasoner | strong, much cheaper than inkling |
| 3 | `thinkingmachines/inkling` | drafter/implementer/instruction_follower | IFEval 79.8, MMLU-Pro 88.7 |

Planned (coding phase, confirmed 2026-07-26): add `moonshotai/kimi-k3` as a
**coder** slot — a binding data edit; the conductor is never trained on model
names, only on capability profiles, so absorbing kimi (or whatever beats it
next) requires no retraining. Bench alternates: `tencent/hy3`,
`poolside/laguna-s-2.1`, `xiaomi/mimo-v2.5-pro`.

## Hard Rules

- **Update MISSION.md with EVERY result or failure obtained along the process,
  at the moment it lands — not at session end (user rule, 2026-07-26). Results
  go into Current Status / the relevant results section; failures go into
  Tried and Failed with the transferable lesson.**
- **NEVER classify prompts/tasks by words, strings, or substring matching
  (user HARD RULE, 2026-07-27). Prompts can be in any language. This is an
  orchestrator, not a word-based router: all task understanding goes through
  the conductor model. The keyword competence router was REMOVED ENTIRELY
  (module deleted; light-path selection is now conductor-driven with a
  per-task cache; string-marker failure detection also removed). No benchmark
  number was ever measured with it — every run had `--competence-table ""`.**
- **xhigh/max reasoning effort HARMS benchmarks — tested twice (ALE: max
  0.4548 vs high 0.5243; MMLU-Pro same-70: declared 82.9 vs default 85.7).
  Run workers at high/default; binding efforts capped at `high` (user rule,
  2026-07-27).**
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
- **HARD RULE — never pay to measure a solo model's benchmark number that is
  publicly available. Use published numbers for solo pool members and
  comparators (benchmarklist.com, provider cards, leaderboards). Paid runs are
  ONLY for numbers that do not exist publicly: our own orchestrated system, or
  a solo model under a specific non-standard harness we must control for.
  Measuring glm/kimi/deepseek/minimax solo on a standard benchmark to
  "confirm" a published figure is banned — it burned money on 2026-07-26
  (kimi-solo TB2.1, a differentiation probe) before this rule.**
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

Keep `Tried and Failed` current too: one row per abandoned approach with the
outcome and the transferable lesson. It exists so no one re-runs a dead line —
several entries there cost real money to learn. Update both sections whenever
a result lands, not at the end of a session.

The recipe below stays detailed. Change it only when the actual operating
procedure changes.

## Current Status

Last updated: 2026-07-26 (topology-training pivot).

| Item | State |
|---|---|
| Objective | Train the router so multi-step workflows BEAT solo workers on every task type; bar = Opus-4.8 / Terra level; general office use, coding phase next |
| Conductor | **REVISED UNDERSTANDING 2026-07-26: the "trained conductor" r2 is a ~zero adapter — production conductor behavior has always been base Qwen3.6-27B + typed schema-constrained decoding.** The r4 topology line is the FIRST training that will actually alter conductor behavior at serving. Serving contract going forward: merged weights only (vLLM LoRA no-ops on this stack) |
| Pool | Pool C (`current_pool_binding_general.json`): deepseek-v4-pro, minimax-m3, glm-5.2, inkling; kimi-k3 coder slot planned |
| Router benchmarks (measured) | GPQA-Diamond **92.5%** (n=40; clears Opus-4.8 92.0), IFEval **79.5%** (n=39) — each within 0.3pp of the domain-best member via a DIFFERENT specialist; ~$1.23/M vs Opus ~$11.25/M |
| **Topology finding (2026-07-26)** | **tree3 (3 diverse models → aggregator) is the universal winner.** GPQA n=24: single 87.5 / chain 91.7 / vote3 91.7 / **tree3 95.8** (+8.3pp over solo, net +2; diversity beats self-consistency +4.1pp). MMLU-Pro n=28 stratified: single 85.7 / chain 85.2 / vote3 89.3 / tree3 89.3. Chain (r2's current shape) never wins |
| r4 dataset | `fugu_topology_sft_v1.jsonl`: 489 rows, all validated 4-step diverse trees; 111-task gate holdout; zero leakage; zero truncation risk |
| r4 training | v1 gate FAIL led to two structural discoveries (see Tried and Failed): **r2 is a ~zero adapter** and **vLLM LoRA serving no-ops on this stack**. v3 (topology-only, 40 steps, loss→0.07) FIXED it |
| **r4 SHAPE GATE: PASS 40/40 (100%)** | 2026-07-26 19:10 UTC, served merged model (`output/fugu_27b_r4_merged_bf16`, original-layout shard patch, TP4). Every holdout task (4 gpqa + 36 mmlu_pro) → diverse tree (3 distinct independent leaves + aggregator); base model baseline: 0/40 trees. **First 27B training to reach serving.** Serving note: text-only `save_pretrained` config breaks vLLM's qwen3_5 route — patch tensors inside ORIGINAL snapshot layout instead |
| **r4 EXECUTION GATE: PASS** | 2026-07-26 19:45 UTC, 40 holdout tasks, r4-planned trees executed end-to-end (wave-scheduled worker loops) vs paired domain-best solo: **r4 87.2% (34/39) vs solo 82.5% (33/40), +4.7pp; paired 2 recovered / 1 broken (net +1)**; 39/39 scored plans were 4-step trees; 1 execution failure excluded. Small-n: net +1 alone is not significance — but direction matches the topology arms (net +3 over 52 tasks). **The trained router's multi-step workflows beat the solo worker on held-out tasks — the mandate's first end-to-end proof** |
| **r4 router GPQA (EvalScope, same 40 questions/seed as the 92.5% run)** | **36/39 = 92.3%** (user-stopped at 39/40; the 40th spun on a hung price-sorted provider — infra, not model). Matches pool best (minimax published 92.7), clears Opus-4.8 (92.0), identical to the pre-training router's 92.5. Reading: GPQA n=40 is a CEILING regime (~3 questions of headroom) — parity is expected and the tree lift (95.8 on the n=24 arm) cannot be resolved at this n. IFEval: SKIPPED (user, 2026-07-26) |
| **r4 router MMLU-Pro (stratified 5×14 = 70, pre-fix)** | 57/70 = **81.4%** — below published solo bars. Diagnosis (all free/local): role assignment CORRECT (20/20 plans = the measured-best specialist set), format intact (68/70) — the gap was SERVING-PATH DILUTION + two defects: (1) reasoning-only responses (`content: null`) read as empty, (2) an empty aggregator output discarding three good leaf answers. Also found: leaves received 7.9k chars of workflow wrapping around a question the winning arm sent bare |
| **MMLU-Pro failure diagnosis (leaf capture, 2026-07-26)** | The 10 stable failures split: **5 all-leaves-wrong** (capability ceiling), **3 aggregation-miss**, **1 majority-override**, 1 flickers. Full lever sweep then ran (see Tried and Failed): instructions, aggregator models (incl. kimi-k3), mechanical majority, leaf compositions — ALL 0/4 on the recoverable class, controls 8/8 throughout. **VERDICT: 9 of 10 stable failures are unfixable by any available orchestration mechanism with this pool — the persuasive-wrong-answer ceiling. Realistic router max on this 70-question sample ≈ 60-61/70 (85.7-87.1%); beating the published solo best (88.7 → needs 63/70) requires pool capability that does not exist in the current binding.** Goal "beat solo on MMLU-Pro without damaging accuracy": not reachable via orchestration/training on this sample; remaining options are a pool upgrade (a model whose knowledge de-correlates from the pool's shared misconceptions) or accepting measured parity |
| **r4 router MMLU-Pro (same 70, POST-FIX)** | **60/70 = 85.7% (+4.3pp on identical questions)** — recovered categories (business/health/history +1 each) are exactly where the empty-answer and format defects lived. Standing vs published solo: inkling 88.7 / deepseek 87.5 / glm 86.7 — at n=70 (SEM ≈ 4.2pp) the router is statistically INDISTINGUISHABLE from the solo bars; neither "beats" nor "trails" is claimable at this n. Law persistently weak (2/5 both runs; coherent-but-wrong legal reasoning — no law strength in the pool). Fixes shipped with tests (441 green): bare-request leaves, format reminder last in aggregator context, minimal system prompt, reasoning-field fallback, aggregator-empty fallback. Attempt truncation REVERTED to off after user challenge (the winning arm passed full attempts; the 1200-char cap was an unmeasured invention) |
| r4 pipeline (built, offline-verified) | measurement harness `topology_policy_data.py` (per-task records, failure-separated, stratified) → SFT builder `build_conductor_topology_sft.py` (bare prompts, type-level winners, measurement-gated collaboration, ~600-text augmentation, gate holdout) → trainer config `fugu_r4_topology_sft/` (r3's proven settings, v3-mixed to retain live control) → gate `gate_r4_topology.py` (shape vs r2, then execution ≥ solo) |
| Heavy orchestrator | independent workflow steps run CONCURRENTLY (wave scheduling by access lists); tree latency ≈ one leaf + aggregator. **Wave timeout added 2026-07-26 (default 480s): a stalled leaf is DROPPED and aggregation proceeds with the attempts that returned** — without it, one slow provider gated the whole tree past EvalScope's client timeout and 2/40 GPQA questions spun in retry churn (each retry re-executing a full tree). 6 offline tests |
| GPU/service state | ALL 8 GPUs on r4 v3 training (~100 min from 17:35 UTC); :8011 down until merge-and-serve (TP4, GPUs 0-3); router :8022 up but needs restart for the wave-scheduler change and the merged-conductor URL |
| Coding measurement | **STOPPED at 1/20 by decision (user, 2026-07-27) — see "Coding measurement: stopped" below.** Pipeline proven; published kimi numbers now cover the external comparison; n=20 could not resolve the remaining question at ~$5-6/task |
| r5 base decision | RUNNING (now the ONLY paid work): 27B vs 8B on 60 shared holdout questions, pre-registered rule, conclusive by construction (see R5 section). Early: both parse 100% of sampled plans; 27B emits 4-step trees (4.0 worker calls/rollout), 8B emits 2-3 step chains (2.8) — 8B collects ~6× faster and ~30% cheaper per group, but showed an all-1.0 (zero-gradient) group early, which is the failure mode the informative-fraction rule exists to catch. **Fixed path after: winner → GRPO → gates → final benchmarks → ship** |
| Discarded verdict | A monitor bug (`if grep … \| head -1` always exits 0) fired the decision function on EMPTY files and printed "R5 BASE = 27B". **That output was not a result** — zero groups. Guard: never let a completion check run the decision path without asserting row counts |
| Plan (user-confirmed 2026-07-27) | Product = FUGU-ULTRA ONLY (no fast variant). Current line: coding baseline + r5 base decision → r5 GRPO → gates → final table |
| Verdict | Ultra RUNTIME complete and live-proven end-to-end (MC benchmarks + coding pipeline). Trained-orchestration LIFT beyond coldstart = r5's job; baselines being banked now |

### Historical context (scaffold finding, TB2.1 campaign)

The worker interaction loop, NOT the conductor and NOT the pool, was the ~22pt
TB2.1 gap: glm-5.2 alone under Terminus 2 scores ~83 (published 82.7) vs our
old 4-model stack's 61.0. Guidance injection was churn (13 up/12 down, p=1.0).
ALE-derived: conductor 0.5157 vs Sol@max 0.5494, 95% CI [−0.237, +0.191] —
"no gap detectable at this power", never "matches Sol". Paid spend to date:
~1500 calls proprietary campaign + ~350 open-weight ALE + ~$40 TB2.1/EvalScope
+ ~$15 benchmark/topology measurement 2026-07-26.

## Tried and Failed (do not repeat)

| Attempt | Outcome | Lesson |
|---|---|---|
| Micro-dose GRPO on the conductor (17 optimizer launches) | 2 accepted / 14 rejected / 1 stopped; never moved whole-task performance | Retired as mathematically incapable at that dose |
| CORE lesson injection (122 lessons, cosine × Beta utility retrieval) | `nhanes` 0.571→0.975 in isolation, but **no aggregate effect** on TB2.1 (61.0 vs 60.7) | Single-task wins on a high-variance benchmark are not evidence; measure the full set |
| One-sided A/B (re-run only *failed* tasks with guidance) | Showed "4/8 recoveries", implied a large win | **Invalid design** — cannot observe regressions on passing tasks. Full-set re-run showed the win was zero |
| r3 SFT v1 (posture distillation) | Gate FAIL: short-horizon posture 12%→0% | Targets mined from `raw_plan` are POST-translation (`worker_id`); the model emits `profile_ref`. Training on untranslatable targets moves nothing |
| r3 SFT v3 (format fixed, 84 examples) | Gate FAIL again: posture still 0% at temp 0 | ROOT CAUSE FOUND 2026-07-26 (via r4): `adapter_init_mode: merge` — see r4 v1 row below. The chat-template suspicion was wrong (renderings verified byte-identical) |
| r4 topology SFT v1 (mixed with v3 dataset) | Shape gate FAIL 0/40. Diagnosis chain (all offline, no paid calls): temp-0 probe on verbatim training row → template renderings byte-compared (identical — hypothesis killed) → adapter composition (no change) → label masking check (rows trained, 298 positions) → weight norms → **manual delta application to the BF16 base: target NLL 0.970→0.046, so training DID learn the mapping** | THREE root causes stacked: **(1) r2, the "accepted trained conductor", is a ~ZERO adapter** (max\|B\| 3e-4; its LoRA product is nothing; all "r2 behavior" is the base model + schema-constrained decoding — consistent with micro-dose GRPO history). **(2) vLLM LoRA serving on this hybrid stack silently applies adapters as NO-OPS** (verified with wrapper-name AND canonical-name keys; masked for months by r2≈0). Never serve behavior-bearing adapters as LoRA here — MERGE into weights and serve the merged model. **(3) The v3 dataset mix fought the topology target**: near-identical control prompts with r2-style continuations out-competed the fixed tree reason at the first branch token (exposure bias: NLL 0.046 under teacher forcing, yet greedy takes the v3-style 1-step path). Fix: train topology rows ALONE — the "retain live control" rationale for mixing died with finding (1) |
| v2 `adapter_init_mode: trainable` attempt | Crashed at startup: native `SurogateTrainer` has no `import_adapter` | Commit 85424bbd added only the Python side of trainable-init; the C++ binding was never implemented. Check the native binding exists before relying on a new trainer mode |
| Sampling collector without constrained decoding | 0 accepted samples in 25 min | Production constrains decoding to the capability JSON schema; free-form sampling yields prose the contract rejects |
| Docker default address pool under parallel pier runs (2026-07-27) | Every `compose up` failed with "all predefined address pools have been fully subnetted"; tasks returned reward=None in SECONDS and looked like model failures | Each pier trial creates a bridge network; the default pool holds ~31 and interrupted runs LEAK them (29 orphaned). Fix: per-run scoped `docker network prune --filter name=<job>` in the driver + concurrency 6 (host had 64 cores free — the network pool, not CPU, was the limit). **A "result" that arrives implausibly fast is an infra failure until proven otherwise** |
| Uncited benchmark figures in this file | The "kimi DeepSWE 67.5" drove a whole measurement's framing; the real independent number is 68.5 pass@1 / 82.0 pass@2 | Every comparator MUST carry its source. An uncited number is a hypothesis, not a bar |
| Paid solo baseline for MMLU-Pro (2026-07-26) | ~59 calls spent re-measuring inkling, whose MMLU-Pro score is PUBLISHED (88.7); user stopped it | **HARD-RULE VIOLATION.** The "different sample so published numbers aren't comparable" argument is exactly the rationalization the rule forbids — comparing OUR measured system against PUBLISHED solo numbers IS the accepted method here. Never pay for a solo number that exists publicly, however good the methodological excuse sounds |
| MMLU-Pro aggregation lever sweep (2026-07-26, arm tests over identical attempts with stable-correct controls) | ALL levers dead on the 4 "recoverable" failures: 3 aggregation instructions → identical outputs; 3 aggregator models incl. kimi-k3 → 0/4; mechanical majority → 0/4 AND drops a control that judge-by-merit rescues; 3 leaf compositions (strongest-trio, 4-leaf) → 0/4 with the gold answer PRESENT in the attempts each time. Controls 8/8 throughout | **Judge-based aggregation converges on the most PERSUASIVE answer; for these questions the persuasive answer is wrong across every judge, instruction, and composition.** Minority-correct recovery through merit review is capped by pool-wide correlated conviction — an orchestration-mechanism ceiling, not a prompt or binding bug. Also: "recoverable" classifications are unstable under leaf resampling (the correct minority appears stochastically) |
| Uniform `max` reasoning effort | 0.4548 vs 0.5243 at `high` (13 ALE tasks) | Effort tier is not a lever for this pool; all configs within 1 SEM. Higher effort costs 3-4× wall-clock for nothing |
| Binding-declared efforts on MMLU-Pro (2026-07-26, same 70 questions) | 82.9% at declared (xhigh/max/high) vs 85.7% at provider default — a 2-question swing, inside noise | REPLICATES the effort-is-not-a-lever finding on an MCQ family. The heavy path now honours the binding's efforts (contract correctness, wired + tested), but do not expect or chase score changes from effort tuning |
| Role-split effort (analysis=max, execution=high) | 0.4732 — no better | Same conclusion; the "max helps analysis" pattern did not survive contact with the full set |
| EvalScope calibration attempts 1-3 | 20+ trials with no verdict, scores unusable | Three infra faults: 120s per-command timeout (`timeout_multiplier: 2.0`), **Docker disk at 93%** (keep >400GB free), over-parallel builds (batch 8-10, not 20). **Always check the no-verdict count before reading a score** |
| Distilling Sakana's Fugu API decisions | Impossible | The API returns only a final `message` item — no reasoning summaries, no plan, no worker attribution. Only `usage.orchestration_*_tokens` is exposed (a trivial query cost 1,260 orchestration input tokens) |
| Chain topology (worker → different-model verifier) as the collaboration shape | Worst collaboration arm on both benchmarks measured (MMLU-Pro 85.2 vs solo 85.7; adds nothing paired) — and it is what r2 currently emits | Review-style chains don't pay; INDEPENDENT attempts + aggregation do (+3.6pp MMLU-Pro). Train topology selection; don't assume review helps |
| Concluding "collaboration doesn't pay on this task type" from 3 hand-written topologies | Framing rejected (user, 2026-07-26) | Fixed hand-picked plans don't bound what a trained conductor can find. Always include a self-consistency vote arm as control to separate ensembling from model diversity |
| Sequential topology diagnostic without per-task records | Killed at 6/24 after ~20 min; output was aggregate prints only | Measurement scripts must persist per-task JSONL (paired stats + training reuse need it) and run arms concurrently; score failures separately, never as wrong |

## Product API surface (from Sakana's shipped `configs/files/fugu.json`)

Sakana exposes **reasoning effort as a parameter on the orchestrator itself**,
not on individual workers: `fugu` and `fugu-ultra-v1.0` accept `high`/`xhigh`;
`fugu-ultra-v1.1` adds `max`. That is the clean contract form of the dial this
campaign found empirically — the conductor needs a LEAN posture on
time-limited work (single builder, minimal supervision; validated on
Terminal-Bench) and a DEEP posture on long-horizon work (multi-step,
role-distributed, verified; validated on ALE-derived tasks).

Our endpoint should therefore take `reasoning_effort` from the caller and map
it to orchestration depth (high = lean, xhigh = builder + verification pass,
max = deep multi-position work), falling back to the task's observed wall-clock
budget when the caller does not specify. Worker-level effort stays in the pool
binding and is a separate concern.

Other fields worth matching: `context_window` 1,000,000;
`truncation_policy {mode: tokens, limit: 10000}`; `supports_parallel_tool_calls`;
`input_modalities [text, image]`. Their `base_instructions` (959 chars) is a
pure safety preamble (do not kill your own runtime; never force-kill by raw
PID) — orchestration strategy is NOT in the prompt, it is in the weights, which
is the same conclusion the distillation run here reached.

## ORCHESTRATION CEILING FOR THIS POOL (published data, 2026-07-26)

The objective is overall improvement over the pool: the endpoint should beat
the BEST single member averaged across a diverse suite, by routing each task to
the right specialist (Fugu Table 1 result). The ceiling of that is the ORACLE
router (best model per task). Computed from published benchmarklist.com numbers
— NO paid measurement:

Specialist split (real differentiation):
- kimi-k3: agentic coding (TB2.1 88.3), reasoning (GPQA 93.5, HLE 56), SciCode
- deepseek-v4-pro: competitive coding (LiveCodeBench 92.5 vs glm 69.5), factual
  knowledge (SimpleQA 84.4 vs glm 38.1, MMLU 94.8), tool use (BFCL 82.7)
- glm-5.2: SWE-bench (82.8, slight)
- minimax-m3: rarely the outright winner

**Sobering ceiling:** on the 5 benchmarks all four report, kimi-k3 wins 4/5,
so oracle-router beats kimi-solo by only **+1.2 points** — and a real router
captures a fraction of that, likely within noise. kimi-k3 is a near-dominant
generalist, which caps the orchestration margin the way a frontier-diverse pool
(Fugu's Opus/GPT/Gemini) does not.

**Consequence for the product — the DURABLE thesis (user, 2026-07-26):**

The product is the **model-agnostic Fugu-style router**, NOT any model behind a
loop. Do not ship "kimi-k3 behind a loop": that is a snapshot, obsolete the
moment deepseek-5 / glm-6 ship. The router's value is not the +1.2 it extracts
from THIS pool today — it is that it extracts the max from ANY pool, routes by
anonymous capability role, and absorbs every new model via a one-line binding
swap with NO retraining. Pools only improve; the router compounds that for free.

Therefore:
1. The router must contain NO model name. It routes to capability ROLES
   (implementer/debugger/planner/verifier); the versioned binding resolves
   role -> model. "Swap in the new best model" = edit the binding, nothing else.
2. The small ceiling today is a property of a kimi-dominant pool, not of the
   architecture. Report it honestly, but it does NOT argue for hardcoding kimi —
   it argues for keeping the binding recalibratable as models arrive.
3. The build target is the Fugu-style conductor-driven router: the trained,
   model-agnostic conductor (r2) plans capability-role workflows; each role runs
   an isolated worker loop; binding maps roles to current models. This is the
   one asset that stays valuable as the model landscape churns.

## MEASURED RESULTS — fugu-open router vs PUBLISHED solo numbers (2026-07-26)

Our system measured; every solo/comparator number is PUBLISHED (no paid solo runs).

| Benchmark | fugu-open (measured) | Pool best (published) | Opus-4.8 bar | Other members |
|---|---:|---:|---:|---|
| GPQA-Diamond | **92.5%** (n=40, SEM 4.2pp) | minimax-m3 92.7 | 92.0 | deepseek 89.4, inkling 87.9, glm 85.6 |
| IFEval | **79.5%** (n=39, SEM 6.5pp) | inkling 79.8 | — | deepseek 76.5, glm 73.3 |

| MMLU-Pro | **VOID — not comparable** | inkling 88.7 | — | deepseek 87.5, glm 86.7 |

MMLU-Pro measured 97.5% but is DISCARDED: `--limit 40` took the first 40 rows,
all from the *math* subset, while the published figure averages 14 categories.
A valid run must sample across categories (per-subset limits or a much larger
n). Recording it as a result would have been a fabricated win — the giveaway
was that it exceeded every published solo number by ~9 points.

### TOPOLOGY MEASUREMENT (2026-07-26) — which workflow shape pays, per type

Four arms on the SAME questions, same pool, per-task records
(`scratchpad/topology_policy_*.jsonl`); failures excluded, never scored wrong.
`single` = the domain-best pool member (the baseline the product must beat);
`vote3` = 3 samples of one model → aggregate (ensembling control);
`tree3` = 3 DIFFERENT models independently → aggregator (Fugu's GPQA winner).

| Arm | MMLU-Pro (n=28, stratified 14 cats) | GPQA-Diamond (n=24) |
|---|---:|---:|
| single (domain best) | 85.7% | 87.5% |
| chain (r2's current shape) | 85.2% | 91.7% (net +1) |
| vote3 | **89.3%** (net +1) | 91.7% (net +0) |
| **tree3** | **89.3%** (net +1) | **95.8%** (net **+2**, 2 recovered / 0 broken) |

**tree3 is the universal winner (user decision 2026-07-26):** strictly best on
GPQA — where model DIVERSITY beats self-consistency by +4.1pp (95.8 vs 91.7),
the direct measurement of routing's value — and tied with vote3 on MMLU-Pro at
one extra call. One uniform target also removes the 453:36 class-imbalance risk
from training. n is small by design: these arms pick TRAINING TARGETS;
significance comes from the retrained router's full benchmark run. Chain — the
shape r2 actually emits — never wins anywhere.

**r4 dataset built and validated (2026-07-26):**
`fugu_topology_sft_v1.jsonl` — 489 rows (36 gpqa + 453 mmlu_pro, all
14 categories), every target a 4-step diverse tree (3 independent leaves,
distinct profiles, aggregator with access [0,1,2]); 489/489 parse under the
live contract; 0/111 holdout leakage; max row 3,794 tokens < 4,096 (zero
truncation); 111-task gate holdout reserved. Serving-side fix applied with it:
`TrainedConductor` now pins `chat_template_kwargs {enable_thinking: false}` so
serving matches the `qwen3_nothinking` training template — the r3 v3 killer.

**Two for two on the valid benchmarks, and on a DIFFERENT specialist each time** (minimax for reasoning,
inkling for instruction-following), each within 0.3pp of that member's published
score. This is the routing thesis validated end-to-end: one endpoint delivers
each domain's best model without the caller knowing which to pick.

**Reading:** the router CLEARS the Opus-4.8 target (+0.5) and lands within noise
of the pool's strongest member, i.e. routing correctly sends reasoning tasks to
minimax AND the published number reproduces through our conductor -> worker-loop
serving path. Against the members it did not route to, the user gains +2.9 to
+6.9 without needing to know which model suits the question. Effective cost
~$1.23/M vs Opus-4.8 ~$11.25/M.

**Two bugs that only benchmarking could surface (both fixed):**
1. Isolation orphaned tool results — masking a foreign assistant turn dropped
   its `tool_calls`, leaving a `tool` message unpaired and violating the OpenAI
   contract. Tool calls/results are ENVIRONMENT (shared); only reasoning is
   isolated.
2. The conductor's subtask paraphrase replaced the original request, destroying
   answer-format requirements: GPQA scored **7.5% (below random)**. The original
   request now passes through verbatim alongside the subtask; a single-letter
   question now correctly returns `B`.

This is why router numbers must be MEASURED and cannot be derived from published
solo numbers: plans are sampled, may span multiple workers, and the serving path
itself can silently destroy or add performance.

## IMPLEMENTATION COMPLETE (2026-07-26) — Fugu-heavy stack

The product is the model-agnostic conductor emitting PLANS (not a deterministic
per-task classifier). Plans are sampled and may involve one or many workers, so
outcomes CANNOT be derived from published solo numbers — a multi-worker plan can
exceed every individual worker (Fugu's own result: GPQA 95.5 vs best worker 94.3).
That is why measurement of OUR system is the only legitimate paid run.

Shipped components (all model-agnostic; swapping a model = binding edit):

| File | Role |
|---|---|
| `ultra/trained_conductor.py` | conductor client (:8011) samples a typed capability workflow -> steps. (2026-07-26: the model behind it was base+schema all along — r2's adapter is ~zero; r4-merged replaces it) |
| `ultra/fugu_heavy.py` | executes the plan: one ISOLATED worker loop per step, access lists, shared env |
| `ultra/worker_loop.py` | per-worker function-calling loop + per-agent call attribution |
| `ultra/router_endpoint.py` | OpenAI-compatible surface; heavy path + light per-request path |
| `ultra/router_server.py` | `/v1/chat/completions`, `/v1/models` (`--heavy`) |
| `competence_table.json` | published model x domain competence (binding-calibration prior) |
| `current_pool_binding_general.json` | Pool C: deepseek-v4-pro, minimax-m3, glm-5.2, inkling |

Fugu §3.2.2 invariants — all verified by test:
- any agent may call functions at any time (each worker owns a FULL tool loop)
- every call attributed to its emitting agent, results routed back to that loop
- intra-workflow isolation: private transcripts; prior work only via access list
- persistent shared memory: tool calls/results are SHARED (environment), only
  assistant reasoning is isolated (fixed a real bug: masking a foreign assistant
  turn used to orphan its tool result and violate the OpenAI contract)

Verified behaviour: the conductor plans ADAPTIVELY — "convert CSVs and verify"
-> 2 steps (coder -> verifier, access=(0,)); "who is the CEO of Siemens" -> 1
step. End-to-end query through the endpoint returns a correct answer.

r3 training line: PARKED and its objective is now OBSOLETE (taught
single-builder posture; failed its gate twice — target-format bug, then
undertrained + chat-template mismatch). Its successor is the r4 topology line
below, which fixes r3's core defect: r4 trains on MEASURED multi-worker
outcomes, not synthetic targets.

## FUGU TECHNICAL REPORT FULL READ (2026-07-27, all 31 pages)

Sakana ships TWO orchestrators, and the split maps onto our two paths:

**Fugu (fast) = decision-only selection, NOT a text-generating router.** A
lightweight prediction head (linear/low-rank/sparse/block-diagonal) attached
after the backbone's final hidden layer outputs L logits (one per worker) from
the hidden state at an early token position — no autoregressive decoding at
all, which is why its latency matches a direct model call. Backbone adapted
via singular-value fine-tuning only. NO role assignment (they dropped
Trinity's roles: selection-only "keeps the orchestration decision simple").
Trained in two stages: (1) SFT on single-step verifiable tasks with SOFT
target distributions — run every worker n times per question, average
rewards, softmax(r̄/τ), KL loss (richer than argmax labels); (2) **sep-CMA-ES
evolutionary optimization directly maximizing terminal reward on END-TO-END
multi-turn trajectories** from Claude Code/Codex/OpenCode-class environments
— more stable than SFT on multi-turn data, handles sparse/noisy success
signals without ranking labels.

**Fugu-Ultra = the Conductor (GRPO 0/0.5/1, no KL, ≤5 steps) + function-
calling extensions** (workflow state tracking; intra-workflow isolation via
access lists; persistent shared memory for tool calls across workflows; pool
Gemini-3.1-Pro/Opus-4.8/GPT-5.5 with unlimited env interaction). Training
data: public data + expert-designed end-to-end environments.

**Result structure that matters to us:** Fugu-fast ≈ Ultra on MC/knowledge
benchmarks (GPQA both 95.5; LCB 92.9 vs 93.2; TB 80.2 vs 82.1) and fast even
WINS some (SciCode, τ³, LCR). Ultra's decisive edge is LONG-HORIZON agentic:
SWE Bench Pro 73.7 vs 59.0 (+14.7) and HLE 50.0 vs 47.2. **Per-input/per-turn
selection captures nearly all orchestration value on short tasks — heavy
workflows earn their cost on long-horizon software engineering.** This matches
every measurement we made (MMLU/GPQA parity regime; execution-gate lift on
mixed holdout; coding as the open frontier).

**§4.4 names our measured MMLU-Pro ceiling exactly:** "dynamic adaptation of
an aggregator role is precisely the kind of adaptation unavailable to
existing multi-agent systems, which necessitate a fixed model to ALWAYS act
as a final synthesizer... bottlenecked by that rigidity, and typically
struggle to surpass the performance of the aggregator for tasks outside the
aggregator's expertise." Ultra varies the aggregator per question (Gemini
aggregates trivia, GPT aggregates math; HMM3 example: Gemini-as-aggregator
synthesized a fully correct answer from two partially-correct leaves). Our
fixed-deepseek aggregator is that named bottleneck; per-question aggregator
selection is a REQUIRED r5 behavior, not an option.

**Validated strategies (§4.4):** build-and-debug with per-step alternation
(GPT builds, Opus deployed at critical debugging moments — Fugu-fast does
this PER TURN in TB); bringing in a specialist (GPT-as-math re-deriving a
differential cryptanalysis constant); debate/tree aggregation in knowledge
domains. Open-ended generalization: AutoResearch (beats all three frontier
solo), kana reading order, CAD, Rubik's (300/300 with shortest solutions),
blindfold chess, online trading.

**Eval configs (App. A):** EvalScope 1.8.1 defaults; mini-swe-agent and
Terminus 2 as deliberately MINIMAL harnesses; max turns effectively
uncapped (SWE Pro 1000, TB 500); τ³ pass@4; LCB patched stdin buffer;
baselines provider-reported wherever available (our published-numbers rule
is their methodology too).

**PRODUCT DECISION (user, 2026-07-27): we build FUGU-ULTRA ONLY — no
Fugu-fast, no selection head, no per-turn routing layer.** Ultra is
self-contained: the conductor decides at WORKFLOW level; each step assigns
one worker who owns its whole function-calling loop; workers are called
directly by the runtime. The light router (:8023) and the interim
plan-step[0] selection are RETIRED. Remaining consequences: (1) r5 training
must include per-question aggregator selection (the §4.4 rigidity = our
measured MMLU-Pro ceiling); (2) heavy orchestration's proving ground is
DeepSWE/TB long-horizon work — the confirmed coding-phase plan, with the
driver's agent runs executed by conductor-assigned workers directly.

## CONDUCTOR PAPER RE-READ (arXiv 2512.04388v5, full read 2026-07-27)

The original Sakana Conductor paper, read completely. What it establishes and
how our current state maps onto it:

**The winning artifact is an RL-trained conductor with natural-language
freedom — NOT a topology selector.** The paper's baselines (MASRouter,
RouterDC, MoA, Smoothie) lose precisely because they "select models and/or
human-designed coordination topologies from pre-specified options"; the
Conductor wins through "complete specification freedom... natural language as
its output medium." **Our r4 — one fixed tree shape with fixed subtask
strings — is architecturally in the BASELINE class, not the Conductor class.**
Its gates passed, but it is a bootstrap, not the product's end state.

**Training recipe (paper-exact):** Qwen2.5-7B, 200 GRPO iterations, batch 256
(4 questions × 64 rollouts), temp 1.0, lr 1e-6 cosine (NO KL), max completion
1024; rewards 0 unparseable / 0.5 well-formed-wrong / 1 correct; workers
CONSTRAINED during training (4096 tokens, temp 0.2, minimal reasoning) for
cheap rollouts. Our 8B stage2 reproduced exactly this (now fully explained:
its chains ARE paper-class behavior for its task mix). 2×H100 sufficed.

**Task adaptivity is the paper's answer to our MMLU-Pro finding:** the trained
Conductor learns MC/factual tasks need 1–2 steps ("targeted information
retrieval", sometimes a 1-shot single model), while LiveCodeBench earns 3–5
step plans with planners→implementer→verifier. Its MMLU gain comes from
targeted routing + light verify, not heavy trees. Our measured "collaboration
barely pays on MMLU-Pro" MATCHES the paper's learned equilibrium — and our
uniform 4-step trees on MC tasks are what the paper's conductor learned NOT
to do. Also B.7: removing subtask generation costs little on MMLU (93.14→
92.75) but a lot on LCB (64.29→58.62) — bespoke subtask prompt-engineering is
where coding gains live (7B vs 3B ablation: subtask QUALITY is the scaling
axis).

**Other directly applicable findings:** (1) OOD few-shots beat in-distribution
few-shots for conditioning (prevents strategy exploitation); (2) recursion =
+gains as tunable test-time compute (20-iteration finetune, half-batch
recursion, 0.25 discount on round 0); (3) weak open models add value in
specific ROLES (format checker rescuing GPT-5 on BCB); (4) open-models-only
finetune beat Claude Sonnet 4 by ~10% constrained — weak pools have MORE
conductor headroom, good for us; (5) GPT-5 medium > high effort on BCB in the
paper too — third independent confirmation of the user's effort rule; (6)
frontier models as untrained conductors beat their constituents but LOSE to
the trained 7B — training the conductor is the moat.

**Consequence for r5:** end-to-end GRPO on measured outcomes with
natural-language freedom (paper recipe adapted to our stack): sample plans at
temp 1.0 from the typed contract, execute through the heavy orchestrator with
CONSTRAINED workers, reward 0/0.5/1, group advantage, no KL. We own every
piece: proven GRPO trainer (8B ran it), executor, verifiable rewards (MC +
DeepSWE unit tests), serving via merged weights. r4 stays as the SFT
coldstart (the paper itself coldstarts via few-shot conditioning).

## R5 — FUGU-ULTRA GRPO (the real training; design locked 2026-07-27)

Goal: the conductor LEARNS orchestration end-to-end (paper-faithful), not a
fixed SFT topology. r4 = coldstart only.

| Component | Design | Status |
|---|---|---|
| Policy | 27B merged r4 as init; LoRA r16 on the BF16 base; serve candidates MERGED (never LoRA) | r4 exists |
| Sampling | temp 1.0, typed capability contract, constrained decoding (production parity); G=8-16 rollouts/question (paper used 64 with cheap workers — start smaller, scale on signal) | contract live |
| Executor | fugu_heavy heavy path per rollout; workers CONSTRAINED for training (max_tokens ~4096, temp 0.2, effort high) — rollouts must be cheap | built |
| Reward | 0 unparseable / 0.5 well-formed wrong / 1 correct; graders: MC letter match + LCB-class unit tests. NO judge models | MC proven; LCB grader via esvenv |
| Advantage | group-normalized (r−mean)/std; no KL; clip per GRPO | trainer exists (8B proved it) |
| Data | MC (GPQA-class + stratified MMLU-Pro trainsplit) + LiveCodeBench-class single-call coding; ~500-1000 questions; OOD few-shot conditioning in the conductor prompt | partial |
| Must-learn behaviors | task-adaptive step count (1-2 for MC, 3-5 for code — paper Fig 8); PER-QUESTION AGGREGATOR SELECTION (§4.4 rigidity = our measured ceiling); free-form subtask prompt engineering | — |
| Gates | contract validity ≥ r4; holdout shape/execution ≥ r4; NO-REGRESSION on the 70-question MMLU set (56 stable-correct floor) + GPQA 40; B-norm check on every candidate adapter | gates exist |
| Serving lesson | every candidate merged into original-layout shards before ANY gate (vLLM LoRA no-op) | script exists |

**Build status 2026-07-27:** rollout collector SHIPPED
(`ultra/grpo_rollout_collector.py`, reward ladder 0/0.5/1, infra-failures
excluded not zeroed, informative-group fraction as the viability stat; 4
tests, suite 452). `TrainedConductor.sample_plan` added (raw completion +
token logprobs for the exact-token contract). Worker client takes
rollout-only max_tokens/temperature. **Stage2 rollout mine complete** (steps
168-199): 111 questions with per-question reward-variance priors (103
informative — question-curation prior), **1,936 parseable positive-advantage
completions** = the paper-prescribed few-shot library (real learned
workflows). Old rollouts are NOT optimizer data (off-policy, old contract,
old pool — exact-token rule); their value = the 8B weights + these artifacts.

### R5 BASE DECISION — FINAL (2026-07-27): 27B, on the full registered data

**R5 BASE = 27B** — decided by the pre-registered rule's third clause
(pre-committed default) on the COMPLETE 60-group data for both arms
(`r5_base_decision.py --decide`, no early stop, no substitutions):

| full 60 groups | informative fraction | mean group reward |
|---|---:|---:|
| 27B (r4 SFT) | 0.283 | 0.918 |
| 8B (stage2 GRPO) | 0.267 | 0.900 |
| gap | +0.017 (< 0.15) | +0.018 (< 0.08) |

Neither gap reaches its threshold, so the default fires: **27B** (paper scale
finding, our typed contract, serving stack in place). The two policies are
functionally equivalent as GRPO starting points on this question set.
Diversity prerequisite also verified: 27B emits 4.43 unique plans per group
of 8, 8B emits 8.00 — no group was all-identical, so gradients reflect plan
differences, not just worker noise.

*Process note:* an earlier stop at 39/60 was self-audited as resting on the
wrong reference (paired-39 instead of the registered full-60 basis); the
remaining 21 groups were collected (~$12) and the verdict above is from the
registered data only. The interim paired-39 numbers (0.308/0.905 vs
0.282/0.888) pointed the same way but are superseded.

**Challenged and CONFIRMED (2026-07-27):** production-cost concern ("8B is
within noise, 27B adds serving overhead") examined against the full data.
Conceded: paired per-question diff +0.018, 95% CI [−0.002, +0.041] — the
decision-set scores ARE a tie; 27B serving is TP4/4 GPUs vs 1. Decision
stands on three measured asymmetries: (1) RL headroom — 8B stage2 is
post-GRPO (200 steps banked) and only TIES the SFT-only 27B; (2) topology —
from all 956 final plans: 27B = 3.93 steps, 2.92 independent leaves,
multi-input aggregation in 476/477 (100%) = measured-best tree; 8B = 2.69
steps, exactly 1.00 leaves = the chain we measured WORST (trees +3.6pp
MMLU-Pro; chain below solo). On the 27 not-both-saturated questions 27B
better 15 vs 8 (+0.039). The 8B's ~30% cheaper collection = 2.7 vs 3.9
worker calls — it is cheaper by orchestrating less with the losing topology,
not by being smaller; (3) per-task conductor cost is one ~305-token
completion (max 2048) on own GPUs vs ~4 paid worker calls × ≤4096 tokens —
footprint is fixed infra, not per-task cost. Hedge if footprint ever binds:
distill trained-27B → 8B via SFT on its plans (local, no paid rollouts);
the reverse (RL-ing the 8B out of its chain habit) risks the whole r5
budget. **User confirmed: 27B is the r5 base.**

**THE FINDING THAT MATTERS MORE — the questions are far too easy:**
reward mix 27B {1.0: 255, 0.5: 55, 0.0: 2} / 8B {1.0: 385, 0.5: 94, 0.0: 1};
**~62-63% of groups are ALL-1.0, i.e. zero gradient.** At G=8, roughly
two-thirds of rollout spend on this question distribution buys nothing.
Consequence for r5: question curation is now a BLOCKER, not a nicety —
train on questions with measured reward variance (the stage2 mine already
provides 103 with per-question variance priors), or filter online (drop a
question once a group returns uniform). Expect a ~2.5-3x effective cost
reduction per unit of learning signal from this fix alone.

**BLOCKER RESOLVED — curation design quantified from paid-for data
(2026-07-27, no new spend):** `scratchpad/r5_gradient_yield.py` +
`scratchpad/r5_curation_design.py` (formulas verified against brute force in
`scratchpad/test_r5_gradient_yield.py`, 14 tests). Method: nonparametric EM
over per-question latent difficulty p from the decision-run reward vectors,
then closed-form gradient magnitude Σ|A_i| = 2√(k(G−k)) projected to unpaid
group sizes. Findings:

1. **The pool, not the policy, is the problem — curate once, reuse for either
   base.** Paired on all 60 shared questions: 3-bucket agreement
   (solved/trainable/dead) 45/60 = 0.750, Cohen's κ = 0.506; 85% of questions
   solved by one policy are solved by the other. Fitted always-solved mass
   ≈ 0.70 for BOTH arms; never-solved ≈ 0.07-0.10. No group size rescues
   those ~80%: raising G only mines the ~20% band.
2. **Raising G is NOT a fix on the uncurated pool** (paper G=64 ≈ our G=8 in
   efficiency): informative fraction rises (0.28→0.59 at G=64) but gradient
   per rollout only 0.24→0.33 — the extra rollouts land on saturated
   questions.
3. **Curation ~doubles learning signal per dollar at equal spend.** Same
   51,200-rollout budget ($≈3.6k at $0.07/rollout): uncurated G=64 (paper
   recipe) = 1.00x baseline; probe-then-train m=3 with G=64 → 1.84-2.03x,
   G=128 → 2.1-2.3x. Probe = 3 rollouts/candidate at temp 1.0, keep on any
   reward variance (~16-19 probe rollouts per retained question — measured
   outcome-based filtering, no text/keyword classification anywhere).
4. **Caveat honestly held:** total gradient magnitude is not gradient
   diversity — curation at fixed budget trains ~600-1500 distinct questions
   vs 800 uncurated at G=64; the m=3 probe biases retention toward
   mid-difficulty (retained-pool grad/rollout 0.82 vs 0.35 uncurated).
   Dr. GRPO accounting (no /std) agrees on every ranking above.

**R5 curation decision:** probe m=3 pre-pass on candidate questions (SFT
holdout + stage2 mine's 103 variance-prior questions first, they are already
half-probed), train on survivors at G=16-32 (steps × 4 questions of the
recipe unchanged; G chosen at launch from the live probe keep-rate), drop a
training question permanently once it returns two consecutive uniform groups
(online attrition, same outcome-based rule). This replaces "curation is a
blocker" with a costed mechanism.

**SHIPPED as product component (2026-07-27):**
`ultra/ultra/question_curation.py` — `ProbeFilter` (journaled/resumable
probe pass; informative probe groups returned as reusable training groups so
probe spend is recycled; infra-incomplete probes re-probed, never classified
from an outage) + `AttritionTracker` (consecutive-uniform drop, streak reset
on informative, outages observe nothing, serializable state for resume).
14 tests in `ultra/tests/test_question_curation.py`, all outcome-based —
no text/keyword classification anywhere.

**R5 DATA MIX DECISION (2026-07-27, user-confirmed): REUSE the 8B 200-step
run's hard mix as the r5 candidate pool.** Audited on disk: 782 unique
questions across `director/manifests/fugu_clean_v1/grpo_pilot_train/
hard_mix_*_taskspecs.jsonl` — math 250 (omni_math), reasoning 312
(reasoning_gym + rlpr), unit_code 220 (taco + LCB) — plus measured variance
priors on 111 of them from the stage2 mine
(`scratchpad/stage2_mined/question_difficulty.jsonl`: median
uninformative_share 0.00, 103/111 mostly-informative, 8 dead). Grounds:
(1) difficulty-selected mix — the profile the blocker analysis calls for,
unlike the 70%-saturated SFT-holdout pool; (2) zero benchmark contamination
by construction (omni_math/taco/LCB/reasoning_gym disjoint from MMLU-Pro/
GPQA/ALE) — the Conductor paper's cross-domain generalization is the
transfer mechanism, r4 SFT already anchors topology; (3) pool-intrinsic
difficulty (κ 0.506) means the stage2-era priors largely transfer to the
27B system. Conditions: priors are ORDERING ONLY — every candidate is
re-priced by the m=3 probe under the CURRENT system (27B r4 + current
binding; worst case 782 × 3 ≈ 2.3k rollouts ≈ $160, less in practice since
probing stops at sufficient retention and informative probe groups are
recycled as training groups). Probe order: the 103 prior-informative first,
then remaining math+reasoning (562 candidates); **unit_code (220) deferred
behind the LCB grader integration** (open item) — code rollouts are the most
expensive, they benefit most from probing before buying. Expected retention
at 40-50% on a hard mix: ~250-350 questions — sufficient for 200 steps × 4
questions/step with cross-step reuse.

**R5 SERVING-PROMPT A/B/C (2026-07-27, user-directed, pre-registered,
stage 1 running):** does Conductor-paper planning guidance in the 27B system
message help? Wired via the sanctioned `guidelines` slot
(`TrainedConductor.guidelines`, default None = r4 SFT serving byte-identical
— verified). Arms: A control (None) / B strategy-guidance-only (no format
text — the typed contract is grammar-enforced) / C guidance + 2 paper
examples rewritten into the typed contract. Stage 1 ZERO-PAID (local 8011):
20 prior-ordered mix questions × G=8 plan samples per arm; kill rule: parse
< 0.98 or mean independent leaves < 2 (chain collapse). Stage 2 (paid,
survivors only, ~$11/arm): paired informative-fraction/mean-reward vs A on
the same questions; adopt non-control ONLY on gap ≥ 0.10 informative or
≥ 0.05 mean reward; ties → A. Whatever wins is FROZEN into both GRPO
collection and tokenize_prompt (exact-token contract).
Script: `scratchpad/prompt_ab_stage1.py`.
**STAGE 1 RESULT (2026-07-27): both guidance arms SURVIVE — and both fix
the parse failures outright.** 160 plans/arm on the first 20 prior-ordered
mix questions: A control parse 0.894 / steps 3.86 / leaves 2.85 / uniq
7.25; B guidance-only parse **1.000** / 4.05 / 2.96 / 7.35; C
guidance+fewshot parse **1.000** / 3.98 / 2.56 / 8.00. All arms aggregate
in 100% of parsed plans. Reading: the guidance stabilizes generation
against the temp-1.0 early-EOS/control-char failures (real product
failures — production plan() degrades to single-worker on parse failure);
C pulls independent leaves down (its 2-leaf example showing through), B
preserves the r4 tree shape best. STAGE 2 (paid) executes arms on the same
questions with mix-native graders; pre-registered adoption rule:
non-control needs informative-fraction gap ≥ +0.10 OR paired mean-reward
gap ≥ +0.05 vs A; both qualify → larger mean-reward gap; ties → A. Script:
`scratchpad/prompt_ab_stage2.py` (--decide --freeze). Its first execution
used a truncating worker config and was discarded — see WORKER-TRUNCATION
entry below; re-run design pending decision there.
Corrected en route: the mix needs
NO new grader work — all three taskspec grader types (math_equal 560,
code_exec_stdio 402, rlpr_lenient 300) are implemented in
`ultra/ultra/grading/verifiers.py`; the earlier "LCB grader integration"
open item was already satisfied. Also shipped: `ultra/ultra/grpo_campaign.py`
— the r5 campaign driver (prior-ordered candidate loading byte-keyed to the
stage2 priors, probe phase, keep-rate → G rule, attrition-aware step loop,
exact-token batches, trainer injected; 10 tests in
`ultra/tests/test_grpo_campaign.py`; refuses to start on a saturated pool,
stops early when every question attrites).

**EXACT-TOKEN DEFECT FOUND AND FIXED BEFORE IT COST ANYTHING (2026-07-27):**
re-encoding a sampled completion does NOT reproduce the policy's tokens —
measured 0/480 clean round-trips on the decision run's r4 plans (grammar-
constrained decoding samples non-canonical splits; vLLM logprob token
strings are display-form and unconvertible (152/313 in one plan); served
sequences end in <|im_end|> which stripped text lacks). Training batches
built by re-tokenization would have violated the exact-token contract for
every single sample. Fix shipped: vLLM `return_token_ids` (verified live on
:8011) threaded through the whole chain — `TrainedConductor.sample_plan`
(meta.token_ids/prompt_token_ids) → `RolloutCollector` passthrough →
`build_conductor_batch` prefers served ids (tokenizers demoted to legacy-
record fallback). Logprobs align 1:1 with token_ids incl. EOS. 38 tests
green across the r5 component suite. ALSO measured en route: 27B parse rate
on MIX questions at temp 1.0 is ~0.95 (not ~0.99 as on holdout) — failure
modes: early-EOS truncation mid-string and raw control chars inside JSON
strings; parser stays STRICT, these earn reward 0 (format condition) and are
exactly the behavior GRPO trains away; the prompt-A/B stage-1 kill rule was
re-anchored RELATIVE to control (before any B/C sample existed).

**R5 serving-refresh RESOLVED (2026-07-27, from code not assumption):** the
surogate GRPO loss is importance-corrected — `surogate/grpo/loss.py:75`
computes `trainer_logprobs − inference_logprobs` per token, applies the
ratio in the IPO objective, and reports `mismatch_kl`; `TrainingSample.
completion_logprobs` flows into `inference_logprobs` via `surogate/grpo/
batch.py:20`. Therefore **periodic merged-weight reloads are safe by
design**: staleness between reloads is corrected by the ratio and the IPO
mask, and monitored by mismatch_kl (reload cadence tuned to keep it small).
Per-step TP4 reloads are NOT required. Corollary handled: samples whose
behavior logprobs fall back to zeros read as inference_prob=1.0 → the IPO
mask silently discards them (paid rollouts buying nothing) — now surfaced
as `rollouts_logprob_fallback` in BatchStats; with return_token_ids the
alignment is structural and this must stay ~0.

**ADVERSARIAL PRE-FLIGHT REVIEW COMPLETE (2026-07-27, 29 agents,
refute-by-default): 13 confirmed findings, 7 LAUNCH-GATING. No paid GRPO
rollout until 1-7 land.** Ranked verdict (full detail in the workflow
journal wf_4051af2f-ae3):
1. GRADER DISPATCH ABSENT (launch-blocking): campaign/probe drop
   Candidate.grader_type; one injected grader misgrades a 3-grader pool
   (math_equal stringifies code dict-golds → uniform 0.5 → entire 221-code
   slice probe-dropped after paying for it; str golds can't disambiguate
   math vs rlpr). Fix: thread grader_type end-to-end, per-question
   get_grader in the collector.
2. NO SERVING-REFRESH MECHANISM: checkpoint_every is dead code; as shipped
   all 200 steps would sample frozen r4. Fix: periodic merged reload of
   :8011 every checkpoint_every steps (reuse trainer merge path), block
   collection during reload, dry-test once before spend.
3. STEP-NUMBERING/TRAINER DEADLOCK: campaign stamps 1-based sparse steps;
   filesystem transport needs contiguous 0-based; trainer would block
   forever on step_0 while the campaign keeps buying rollouts. Fix:
   sent-batch counter stamped only on sent batches + trainer ack/health
   check that halts spend.
4. NO MID-RUN RESUME: crash at step 150 of G=32 re-buys ~$1,350. Fix:
   persist {step, tracker state, rng} per step; restore on start;
   train_groups.jsonl already journals bought groups.
5. RLPR_LENIENT SUBSTRING FALLBACK (upgraded to launch-blocking): gold
   "2.7" matches "12.7"; wrong answers score 1.0 on up to 300/801
   questions, flipping informative groups to uniform-1.0 → attrition
   permanently drops trainable questions. Fix: exact normalized match;
   numeric tolerance only when both sides parse as floats; extract
   final-answer line first.
6. NO r5 TRAIN.YAML AND lr 1e-6 REPEATS A PAID MISTAKE: stage2 paid 64
   steps to learn LoRA needs ~10x (train_stage2.yaml documents the mid-run
   correction to 1e-5); cosine-to-zero on LoRA r16 at 1e-6 = negligible
   motion for thousands of dollars. Fix: write train.yaml pinning lr 1e-5,
   constant (or cosine w/ floor ≥0.1), kl_tau 0, save_steps 10; diff vs
   defaults as pre-spend checklist.
7. UNCLAMPED IMPORTANCE RATIO under periodic reload: loss.py:75-76 has no
   ratio clip; rare-token ratios ~1e3-1e5 pass the |Δp|<0.2 IPO mask and
   dominate the gradient — the MISSION reload-safety rationale assumed a
   clip that does not exist. Fix: per-token ratio clamp (≤e²) in loss.py +
   CUDA path; pre-register reload cadence + abort thresholds on
   mismatch_kl/is_masked; zero-paid dry batch to measure grammar-
   constrained baseline mismatch first.
Cost items: (8) probe recycling promised-but-discarded (~$55-85) — fold
probe groups into first batches; (9) probe early-stop claimed but absent —
add retention-target stop; (10) pool is 801 uniques not 782, 39/40 repair
tasks near-dupe code tasks — dedup by problem identity + re-register audit;
(11) 8 giant LCB golds (≤110MB, 992MB file) — journal gold by reference,
drop hard_mix_all from MIX_FILES (contributes zero), exclude/cap the 8.
Advisory: (12) attrition counts 2-of-16-scored outage groups as uniform —
require majority scored; (13) temperature unenforced across
sampler/batch/trainer — assert at campaign construction.
REFUTED (for the record): code_exec_stdio sandboxing as launch-blocking
(production precedent at full scale), zero-fill logprob poisoning (already
countered by rollouts_logprob_fallback), infra-failure G-flip arithmetic
(~4x overstated), retention-optimism (spend is gated by measured keep-rate
at launch, not the estimate).
NOTE: the running prompt-A/B stage 2 is UNAFFECTED (its script does
per-candidate grader dispatch; all 20 questions are math_equal).

**ALL 13 REVIEW ITEMS LANDED (2026-07-27, same day):**
1. Grader dispatch: `RolloutCollector.collect_question(question, gold,
   grader_type)` resolves per question via the grading registry (injected
   grader = single-pool fallback); threaded through ProbeFilter/GrpoCampaign
   as (question, gold, grader_type) triples; groups record grader_type.
   Definitive test: same output, gold 2.7 → rlpr 1.0 / math 0.5.
2. Serving refresh: `ultra/ultra/serving_reload.py` MergedReloadController —
   merge newest trainer checkpoint (surogate adapter_merge) → restart :8011
   matched by `--port` (never touches :8012) → block until /v1/models
   healthy (raise on timeout = halt spend) → prune to keep_merged=2
   (~55GB/snapshot). Wired to GrpoCampaign.on_checkpoint every
   checkpoint_every SENT batches; campaign blocks during reload. 6 tests.
   Dry-test against a manufactured checkpoint = pre-spend checklist item.
3. Step contract: batch.step = contiguous 0-based SENT-batch counter
   (skipped steps consume no number); TrainerStep contract requires the
   adapter to raise when the trainer stops consuming.
4. Resume: campaign_state.json (next_step, sent counter, attrition state,
   RNG state) atomically saved per step; run() restores and continues —
   test proves steps [3,4] after a 2-step run, zero probe re-spend.
5. rlpr_lenient rebuilt: final-answer span (last boxed / last 'answer'
   marker / last line) + exact normalized match + 2% tolerance only when
   both sides parse as float; substring fallback DELETED. 10 exploit-pinned
   tests ('2.7' vs '12.7', intermediate values, '2' vs '2x+1').
6. `train_r5.yaml` written (27B base, lr 1e-5 constant per the stage2 paid
   correction, kl_tau 0, ratio_clip e², save_steps 10 = reload cadence,
   merge_adapter true) with the 4-item pre-spend checklist inline.
7. ratio_clip e² threaded through the FULL stack: GRPOLossConfig →
   loss.py gradient+pg_loss (metrics keep uncapped ratio; new
   `ratio_clipped` metric) → GrpoNativeLossConfig → CUDA kernel
   (fminf in dloss/policy_sum only) → kernels.h → dsl_model_execution →
   py_train → binding (nb::arg default e²) → trainer.py passes
   config.loss.ratio_clip. Native rebuilt, binding verified live; formula
   test pins ratio 1e5 → grad e², mismatch_kl still >1e3.
8-13: probe recycling (informative probe groups = batch 0 on fresh start),
   probe retention_target early-stop, repair-vs-code identity dedup +
   hard_mix_all dropped from MIX_FILES + golds >1MB excluded + journal rows
   strip gold, attrition ignores outage-majority groups, temperature assert
   at campaign start.
Also fixed en route: GRPOLossConfig was missing opd_tau/opd_beta/replay_tau
(loss.py + 13 tests referenced them; pre-existing gap — those tests were
failing at HEAD). r5-relevant suites all green: 69 ultra r5-component tests
+ 15 native-formula tests. (Pre-existing, unrelated: tests/grpo collection
errors from missing `ultra.ale_training`; ultra synthetic-collection path
assumptions — identical at HEAD with our changes stashed.)

**LAUNCH ASSEMBLY SHIPPED (2026-07-27): `ultra/ultra/r5_launch.py`** —
wires every tested component into the paid loop: TrainedConductor (frozen
guidelines) → RolloutCollector (registry dispatch; the injected fallback
grader RAISES on the mixed pool) → GrpoCampaign → FileSystemTrainingBatch
Sender + `TrainerHealth` stall detector (lag > 25 batches AND no trainer-
metrics progress for 30 min → raise → campaign halts spend) →
MergedReloadController on_checkpoint. Tokenizer fallbacks also RAISE (a
rollout without served token_ids must never be silently re-tokenized in
the paid path). Modes: `--dry-batch` (checklist item 1: 2 real :8011 plans,
fabricated reward split, batch 0 → trainer mismatch_kl must be ~0),
`--probe-only`, `--run` (REFUSES until
`grpo_pilot_train/r5_serving_guidelines.json` exists — written only by
`prompt_ab_stage2.py --decide --freeze`). 5 TrainerHealth tests; r5 suite
now 64 + 15.
**GO SEQUENCE:** stage-2 verdict → --decide --freeze → start trainer
(train_r5.yaml) → --dry-batch PASS → reload dry-test PASS → --probe-only
(keep-rate sanity vs the ~0.2-0.5 expectation) → --run.
(Reload dry-test must NOT run while any collection is using :8011 — it
restarts the server.)

**GO EXECUTION LOG (2026-07-27, user GO):**
* Serving prompt FROZEN: arm A (guidelines=None, SFT-matched) →
  `r5_serving_guidelines.json`; the corrected prompt A/B remains available
  as GEPA trigger 1.
* Worker regime FINAL: default reasoning effort, max_tokens UNSET
  (production-proven; even 16384 truncated at default effort — verified),
  temp 0.2 (group-variance concentration). Re-priced: probe ~$200-400,
  campaign ~$2-4k — the cost of deployment-faithful rewards.
* GPU replan (user): vLLM TP2 on GPUs 6-7 (:8011, gpu-mem-util 0.95,
  enforce-eager; fits at 31.4GB/card), trainer on GPUs 0-5
  (train_r5.yaml: gpus 6 + sequence/lmhead chunks 8 + offload_residual +
  cpu_training, memory kit from the proven r4 SFT config; template
  qwen3_nothinking pinned — serving/training template identity). Our idle
  8B server (:8012) stopped to free 4-5.
* RELOAD DEFECT FOUND LIVE AND FIXED during the TP4→TP2 switch: a dying
  server keeps answering /v1/models for seconds, so the controller's
  health check passed against the OLD process while the new one was still
  loading shards — at a real checkpoint this resumes collection on a stale
  policy. Fix: _stop_server now polls pgrep until the old process is gone
  (120s, then SIGKILL) before spawning; pinned by test. 7 reload tests
  green. TP2 verified GENUINELY healthy after the fix (startup complete,
  workers resident, model answering).
* TRAINER BRING-UP (2026-07-27, iterative — each step zero-paid):
  (1) sequence_len 8192 OOM'd the save_for_bwd arena (10.6GB/GPU at
  1024-tok chunks) → measured real need (worst prompt 4,024 tok +
  2,048 completion cap = 6,072) → sequence_len 6144 / chunks 12 =
  512-tok chunks, the r4-proven geometry. (2) Import then crashed with
  async illegal access at ~tensor 130-175 across FOUR runs — recipe
  bf16 and fp8-hybrid both, world size 6 and 4 both — eliminating
  recipe and world-size hypotheses (and an MoE-partitioning guess the
  user corrected: the 27B is dense hybrid, NOT MoE). Remaining variable
  common to all crashes: the import SOURCE was the hand-patched
  fugu_27b_r4_merged_bf16 serving dir — built for vLLM, never before
  used as a trainer import. Fix: import the ORIGINAL base snapshot +
  r4 v3 adapter with adapter_init_mode: merge (the exact r4-SFT-proven
  pattern; weight-identical to the merged dir up to rounding — and the
  dry-batch gate verifies that identity empirically against the served
  merged weights). gpus: 4 on 0-3 (13.5GB/GPU), GPUs 4-5 spare.
* IMPORT-CRASH ISOLATION (2026-07-27, ongoing — 7 crashed runs, each
  elimination zero-paid): EXONERATED: recipe (bf16+fp8), world size (6+4),
  import source (merged-dir + base-snapshot), locked-memory limits, vLLM
  co-residency, and sequence-CHUNKING (the GRPOTrainer raw-seq_len vs
  trainer_seq_len discrepancy at surogate/grpo/trainer.py:103 is REAL and
  worth an engine fix, but unchunked 6144 crashed identically at 293/787
  — chunking is not the root cause). FACTS that remain: SFT path with the
  SAME binary imports at 5 GB/s and trains (but at graph seq 512 — small-
  geometry probe); GRPO path imports at 0.9 GB/s (5.5x slower) and dies
  at a MOVING point (tensors 134→293 across runs, always surfacing as a
  sticky async illegal access at the import thread's next CUDA call) —
  signature of a RACE between import streaming and concurrent
  GRPO-trainer-path activity, not a layout bug. Discriminator running:
  SFT at the EXACT GRPO geometry (6144 unchunked, gpus 4) — crash there
  = geometry/arena problem; clean there = GRPO-path race confirmed →
  compute-sanitizer memcheck to name the faulting op.
* IMPORT CRASH **SOLVED** (2026-07-27, 9 diagnostic runs, all zero-paid):
  root cause = **the fp32-master import path** — GRPOTrainConfig silently
  defaults master_dtype/gradient_dtype to fp32 (rationale: GRPO's tiny
  gradients die in bf16 rounding) while SFT uses bf16 masters; the
  fp32-master streaming import is broken on the 27B hybrid (async illegal
  access at a moving point, 2x slower from tensor 0). With
  master/gradient_dtype bf16 the SAME run imports 787/787 at 4.95 GB/s
  and reaches packer-ready. Chunking discrepancy and settle-race theories
  were tested and withdrawn en route; final elimination matrix: recipe,
  world size, import source, locked-mem, vLLM co-residency, chunking,
  construction race — all exonerated; dtype confirmed by discriminator.
  DECISION: r5 runs bf16 masters (the r4-SFT-proven precision) with a
  hard safety net — **B-norm on the step-10 checkpoint must show a live
  adapter** (the tiny-gradient risk is exactly what the adapter reality
  audit detects), plus the existing grad_norm/mismatch monitoring and
  lr revert rule.
* SECOND REAL BUG FOUND EN ROUTE AND FIXED: GRPOTrainer never called the
  adapter-init protocol (configure_initial_adapter / set_adapter_path
  before import + import_initial_trainable_adapter after) — adapter_path
  was SILENTLY IGNORED; the r5 run would have GRPO-trained the RAW BASE
  instead of base+r4. Fixed in surogate/grpo/trainer.py mirroring the SFT
  wrapper.
* ENGINE-FIX LIST (for the user, non-blocking now): fp32-master import
  path on the hybrid stack (also 5-10x SLOWER everywhere — 0.49 GB/s on
  the plain 8B vs 4.8 bf16); in-stream adapter merge under GRPO config
  (crashes import even with bf16 masters); GRPOTrainer passes raw
  sequence_len where the SFT wrapper passes trainer_seq_len (= seq/chunks)
  — chunked GRPO has never actually run; GRPO fresh_run/resume
  adapter-init interaction.

**BF16-MASTER GRPO SIGNAL TEST — PASS (2026-07-27, user-challenged,
measured):** controlled A/B on the 8B, 20 byte-identical synthetic batches
(deterministic tokens/logprobs/rewards; informative groups), two trainers
differing ONLY in master/gradient dtype. Result: mean|B| after 20 steps
bf16 3.906e-05 vs fp32 3.905e-05 — **ratio 1.000**; loss and grad_norm
trajectories track step-for-step (step19: -0.2910/0.0147 vs
-0.2918/0.0146). AdamW's variance normalization makes update size ~lr
regardless of raw gradient magnitude, so the "GRPO gradients are 10-100x
smaller" concern does not translate into lost updates on this stack.
Honest caveat: measured at 20 steps from B=0; a late-run bf16
accumulation stall (update/|B| < ~2^-8 as |B| grows) is theoretically
possible around a few hundred steps — covered by the existing B-norm
gates at 50/100/150/200 and grad/no-motion monitoring. Test lessons
recorded: fabricated behavior logprobs must sit near the trainer's real
token probabilities or the IPO |dp| mask silently zeroes every token
(first two attempts trained NOTHING at masked=100% — always check
keep_tokens>0 before believing a training metric); the trainer requires
run_default/control/orch.yaml before consuming transport batches.
**DECISION STANDS: r5 trains bf16 masters on the 27B.** 27B trainer
un-held; dry-batch gate next.
* CHUNKED GRPO IMPLEMENTED IN THE ENGINE (2026-07-27, user GO "implement
  it and continue"): step_grpo_native_chunked mirrors the SFT step_chunked
  two-phase loop (KV sweep forward_no_save → reverse re-forward + windowed
  GRPO dloss + backward); kernel gains a window_start param (global-coords
  sample intersection, chunk-local loss/dloss buffers, first-slice-only
  sample counting); GRPO scratch sized B*T*SequenceChunks; DSL split into
  grpo_native_upload_full + step_grpo_native_window (unchunked = window 0,
  behavior-identical).
  VALIDATION LADDER (all zero-paid, 8B rig, identical batches):
  (1) 20-step trajectory compare — confounded by chaotic drift, rejected
  as a method; (2) 1-STEP gradient-direction test — A matrices cosine
  1.0000 (init determinism control ✓, B=0 ⇒ dA=0), B matrices cosine 0.38
  attn / 0.43 mlp ⇒ upstream of attention; (3) boundary-free (both samples
  inside chunk 0) STILL 0.51 ⇒ not boundary math; (4) SINGLE-SAMPLE:
  cosine 0.94, kl within 1.8% ⇒ benign chunked-kernel numerics. THE BUG:
  **packed-document isolation does not exist in the chunked TRAINING
  attention path** — ChunkPackMeta/kvprefix doc metadata is honored by
  execute_eval (validate_chunked) but not by the training forward; packed
  samples attend across sample boundaries (dense doc-masking is skipped in
  chunk mode BY DESIGN). Proper fix = doc-aware chunked attention backward
  kernels (engine-fix list).
  SHIPPED SOLUTION — sidesteps the missing kernel entirely on validated
  paths: `single_sample_bins` (GRPOTrainConfig knob, plumbed
  batch.py → packer → trainer): one sample per micro-batch = no doc
  boundaries = nothing to isolate; + native all-padding TAIL-CHUNK SKIP
  (a 1.4k-token sample costs ceil(len/512)=3 chunk passes, not 12) +
  pad_to_multiple_of: 512. Cost: <512 pad tokens per sample (~15% at the
  measured 1.4k median), no packing efficiency loss that matters at GRPO
  batch sizes.
* **DRY-BATCH GATE PASS (2026-07-27 14:03):** truly-chunked 27B trainer
  (6144/12 = 512-tok graph, gpus 6, bf16 masters, single-sample bins)
  imported, consumed the staged 2-plan dry batch, stepped.
  **unmasked_mismatch_kl = 0.0049** — near-exact trainer↔serving logprob
  agreement on the 93.7% kept tokens, validating in one number: the
  exact-token chain (template identity, served ids, logprobs), the
  fp8-hybrid-trainer vs bf16-serving numerics, and merged-dir weight
  identity. Mismatch concentrates on the 6.3% IPO-masked tokens
  (masked_kl 14.9 — the rare/non-canonical grammar-constrained tokens the
  mask exists to exclude; is_masked_low 5.5%, high 0.7%). 568 exact
  tokens, 551 kept, grad flowed. Config-promotion trap noted in the yaml
  (sequence_chunks silently reset to 1 by an earlier copy — reproduced
  the unchunked OOM until caught). Dry-batch-trained state wiped before
  the real run. Reload restart-half rehearsed live (TP4→TP2 + stop-wait
  fix); merge-half validates at the first real checkpoint with bounded
  risk (≤10 batches).
  **ALL LAUNCH GATES GREEN → probe (--probe-only, retention_target 350)
  then campaign (--run) per the standing user GO.**
* Probe throughput fix (2026-07-27): ProbeFilter probed candidates
  SERIALLY (~5-8 min each on the hard tier ≈ days for the pool) — added
  windowed concurrency (probe_concurrency=6, sliding window preserving
  prior-order approximately; retention-target stop drains in-flight paid
  work into the journal; journal-resume semantics unchanged; 29 tests).
  Probe relaunched with journal resume — zero rollouts re-bought.
* **R5 RESHAPED TO THE PAPER'S MIXTURE (2026-07-27, user-decided):** the
  single-turn verifiable campaign alone = the stage2 recipe at 27B scale
  (same hard mix, same 200-step shape — and stage2 only TIED the SFT-27B),
  while the Fugu paper attributes expertise-discernment gains to
  end-to-end task BREADTH (tool-usage, multi-turn, dialog, planning).
  DECISION: ONE campaign, curated verifiable lanes INTERLEAVED with
  fugu-ultra-pilot environment lanes (repo_tool / repair / multi-turn),
  paper-style. Vehicle: the stage2-PROVEN grpo-orch (multi-env,
  env_ratios draw weighting, max_turns, token client, own resume/pacing)
  — NOT the campaign driver, whose unique pieces port: probe-retained
  list → curated lanes' task manifests; MergedReloadController → sidecar
  on trainer checkpoints (replacing the orch's vLLM-LoRA broadcast, which
  the 27B cannot use); worker regime → env args. Scoping workflow
  (wf_8ef7952d-305) mapping orch config schema, env lanes/rewards, token
  client fidelity, and broadcast→reload wiring into a concrete
  integration plan with per-lane costs. Probe v3 continues in parallel —
  its output feeds the mixture either way.
* **INTERLEAVED-CAMPAIGN BUILD COMPLETE (2026-07-27, from plan
  wf_8ef7952d-305) — all code changes landed, tests green:**
  (1) `ultra/ultra/typed_contract.py` + env wiring: the pilot env gains
  `conductor_contract: "typed_control"` — prompts via TrainedConductor's
  OWN _messages/_state (byte-identity pinned by test), parses via
  parse_capability_control_action → Workflow schema; parse failure routes
  to the existing invalid_workflow_trainable branch (reward 0, trainable).
  Without this every typed plan scored 0 in the env (launch-gating catch).
  (2) `ultra/ultra/probe_manifest_export.py`: probe journal → r5_{lane}
  _taskspecs.jsonl + pilot_config_singleturn_r5.json with the worker pool
  REBUILT from current_pool_binding_general (the stage2-era config routed
  st_* workers via yunwu/commercial — wrong pool for r5) and openrouter-
  only provider policy. (3) serving_reload fixes: checkpoint.json
  completeness gate (mid-save race), same-checkpoint reload skip, dual
  served aliases (name + model path — the orch requests by path);
  (4) `serving_reload_sidecar.py`: reload-on-new-checkpoint watcher
  replacing the orch's broadcast consumer (broadcast itself stays ON —
  orch pacing blocks on its STABLE markers; lora_adapter unset → 404
  no-op). (5) orch_r5.yaml written: 3 curated lanes (16k/0.2/default
  effort) + tool_dialogue + repo_terminal (8k/high/long budget),
  env_ratios [.22 .39 .22 .085 .085] (repair dropped — dedup audit),
  TITO off, temp 1.0 / max_tokens 1024 sampling, contiguous pacing via
  stage2 machinery. (6) live_safety_r5_singleturn.json +
  live_safety_r5_envlanes.json DRAFTED with approved:false — commercial
  workers (gpt-5.5 terminal agent, opus reviewer) EXCLUDED from env
  lanes; **USER REVIEW + approved:true flip REQUIRED before any spend.**
  Test totals: typed-contract 3, exporter 3, reload+sidecar 11.
  REMAINING BEFORE LAUNCH: probe completes → export manifests → set
  rollouts_per_example from keep-rate → user-approved safety manifests →
  zero-paid dry gates (typed parse rate ≥~70% unconstrained, mismatch_kl
  ≈0, packer-drop + sample-split rates) → launch order per plan
  (sidecar-serve → orch → trainer → watcher).
* **USER CATCH (2026-07-28): the env lanes' pilot_config.json is a STALE
  pool** — it listed workers (xiaomi/mimo-v2.5-pro, moonshotai/
  kimi-k2.7-code, gpt-5.5) that are NOT in our current binding. Fixed:
  `pilot_config_r5_envlanes.json` generated with the worker pool rebuilt
  from current_pool_binding_general using the binding's own role priors —
  tool_dialogue: td_deepseek (tool_user) / td_glm / td_minimax;
  repo_open_repo_terminal: term_kimi (kimi-k3, terminal_operator) /
  term_glm / term_inkling — agentic backends (tool_dialog/terminal,
  80 turns) preserved from the lane design, `conductor_contract:
  typed_control` set. Safety manifest worker lists and orch_r5.yaml
  updated to match; the exporter now also stamps conductor_contract into
  the generated singleturn config (was missing — would have re-opened the
  typed-vs-three-list scoring hole).
* **SAFETY MANIFESTS APPROVED (2026-07-28, user-directed review):**
  reviewed against the REAL validator, not by eye — which caught two
  defects before approval: (1) the manifests' custom version strings
  would have been rejected (validator pins the exact constant
  fugu_ultra_live_worker_safety_v1); (2) `thinkingmachines/inkling` AND
  `moonshotai/kimi-k3` were UNREGISTERED in the providers MODELS table —
  unregistered models fall to DEFAULT_PROVIDER=yunwu, which the manifests
  forbid → both registered as openrouter open-weight entries. All five
  pool models now route openrouter; all three lanes VALIDATED end-to-end
  (single_turn/short, tool_dialogue/long, repo_open_repo_terminal/long)
  with approved:true. Human gate cleared.
* **OFFICE-TASK MIXTURE EXPANSION (2026-07-28, user-decided):** the
  product is a router for PERSONAL/OFFICE agentic tasks — the mixture
  must be office-shaped, not STEM-shaped. User-proposed datasets, triaged
  by reward verifiability + integration cost: tau-bench retail/airline
  (already our tool_dialogue lane — expand tasks), tau2 telecom (quick
  harness generalization), BIRD text-to-SQL (execution-accuracy reward;
  DB grader — medium), DABstep data analysis (factoid-verified; medium),
  CRMArena (env-verified, heaviest — later), FinanceBench (judge-
  dependent → numeric/exact subset only or defer; judge rewards are
  noisy/hackable). CONTAMINATION DISCIPLINE: anything we may report as a
  benchmark (tau especially) enters training by TRAIN SPLIT only; test
  splits sealed for the final table. SEQUENCING (user GO): launch r5 as
  staged on probe completion; build office lanes in parallel in priority
  order tau-telecom → BIRD → DABstep → CRMArena and BOUNCE them into the
  running campaign at checkpoint boundaries (stage2 precedent: staged
  mix rebalance at a bounce), ratios shifting office-ward as lanes land;
  anything not ready by campaign end becomes the r6 mixture.
* **OFFICE LANE 1 BUILT: tau2-telecom SOLO harness + sealed-split
  manifest (2026-07-28, offline-validated):** the "tau-telecom" dataset is
  NOT in the installed `tau_bench` package (retail/airline only) — it is
  Sierra's separate `tau2-bench` (installed from GitHub, additive deps
  only; repo vendored at director/vendor/tau2_bench@1d244f5 for data
  files; PyPI's "tau2" is an unrelated physics package). Structure is
  ideal for us: `telecom` = the published 114-task benchmark (SEALED,
  eval-only), `telecom_full` = 2285 generated tickets, strict superset →
  TRAIN POOL = 2171. Built `ultra.harness.tau2_solo` (registered
  "tau2_solo"): drives tau2's Orchestrator in SOLO mode — worker resolves
  the ticket operating agent tools + the customer's device tools, stops
  via the `done` tool; NO user-simulator LLM, so env + reward are fully
  deterministic (tau2's own env-assertion evaluation) and there is no
  per-turn user-sim spend. Worker injected via a queue-agent
  (timestamp-at-dequeue matters: trajectory is timestamp-sorted and
  misordering breaks evaluation replay — found via failing gold replay).
  CONTROLS: gold-action replay through the real run_step → reward 1.0;
  do-nothing → 0.0; prose-instead-of-tools → agent_error, graded 0.0
  (trainable). `ultra.tau2_manifest` exports the train manifest with a
  HARD sealed-leak refusal: tau2_telecom_taskspecs.jsonl (2171 rows,
  TaskSpec-validated) + tau2_telecom_sealed_eval_ids.json written to
  grpo_pilot_train/. 4 tests green. ALSO CONFIRMED: existing
  tool_dialogue lane's 60 tau retail tasks are all retail TRAIN split
  (500 train/115 test; airline is test-only → stays sealed) — no
  contamination; retail lane can expand to 500 later.
* **OFFICE LANE 1 BOUNCE-IN ARTIFACTS READY (2026-07-28) — ONE USER
  ACTION OUTSTANDING:** `ultra.tau2_lane_config` (3 tests) emits both
  files a checkpoint-boundary bounce needs: `pilot_config_r5_office.json`
  (env-lane config + `office_telecom` lane: 240-task deterministic
  sha1-ordered draw from the 2171 train tasks, worker mask = the proven
  tool_dialogue tool-callers td_deepseek/td_glm/td_minimax, group_size 4,
  existing lanes byte-unchanged) and `live_safety_r5_office.json`, written
  **approved:false — the user must flip it**; a test asserts the
  unapproved manifest is REJECTED by validate_live_worker_safety and
  accepted once flipped. `orch_r5_bounce_office.yaml` = orch_r5 + the
  office env lane with office-ward ratios (code 0.39->0.30, math/rlpr
  0.22->0.19, office 0.15; sum 1.0, lane/ratio alignment asserted).
  ZERO-PAID DRY GATE PASSED: offline load of the office lane builds 240
  rows, routes harness tau2_solo, and the TICKET TEXT reaches the
  conductor prompt — the first draft passed only an opaque task id, so
  the conductor had nothing to route on (the solo worker gets the same
  ticket in its tau2 system prompt, so this leaks nothing). Turn cap set
  from data, not guesswork: gold actions run 1-11 (median 6, p99 10), so
  20 turns instead of 30 — ~33% off the per-rollout ceiling.
* **PROBE MID-RUN ANALYSIS — RATIOS WILL NEED REBALANCING (2026-07-28,
  measured at 216/718 probed):** per-lane keep rates differ sharply —
  math_equal 72/165 (44%), code_exec_stdio 15/32 (47%), rlpr_lenient 6/26
  (23%). Projected onto the full pool (math 280, rlpr 300, code 138) the
  final retained set is ≈123 math / 69 rlpr / 65 code ≈ 257 total, i.e.
  math ~48% of retained but code only ~25%. orch_r5.yaml currently gives
  CODE THE LARGEST SHARE (0.39) and math 0.22 — inverted relative to the
  data we will actually have, so code's ~65 tasks would be redrawn every
  ~10 steps while math's ~123 sit idle ~35 steps apart. ACTION: recompute
  env_ratios from the FINAL retained composition at export time (step 3 of
  the runbook), not from the pre-probe plan. WHY QUESTIONS ARE REJECTED
  (uniform-group reward values): math 72 uniform at 0.5 vs 21 at 1.0, rlpr
  18 at 0.5 vs 2 at 1.0 — rejections are dominated by CONSISTENTLY
  UNSOLVED (0.5 = valid workflow, wrong answer), not by too-easy; code is
  the opposite (13 uniform at 1.0 = too easy vs 4 at 0.5). The hard mix
  was built against frontier workers and is genuinely hard for the
  open-weight pool, exactly as the curation-as-filter design assumed.
  THROUGHPUT AFTER THE c32 RAISE: 206→216 completes in ~2 minutes
  (~300/hour vs ~10-15/hour at c6) — remaining ~500 candidates land in
  hours, not days.
* **OFFICE LANES APPROVED BY USER (2026-07-28): "of course I want them in
  training, this is why i gave them to you".** live_safety_r5_office.json
  flipped to approved:true, covering office_telecom (240 tau2-telecom
  tasks) and the expanded tool_dialogue lane (500 tau retail train tasks).
  Approval is now a PARAMETER of the exporters (`--approved` on both
  tau2_lane_config and bird_manifest, default False) rather than a hand
  edit, so regenerating a manifest preserves the decision instead of
  silently reverting it — with a test asserting exactly that. The BIRD
  lane's manifest will be generated approved when its data lands.
  BOUNCE PREFLIGHT NOW: 19 pass / 9 fail; all three office/env lanes pass
  safety (approved, task counts within caps, workers in allowlist), and
  every remaining failure is a probe-produced single-turn manifest that
  needs the paused probe to finish.
* **BIRD GOES THROUGH THE PROBE (2026-07-28, user question: "shouldn't we
  wait for the dataset download to include items in the probe?" — yes, and
  my earlier blanket statement that office lanes cannot be probed was too
  broad).** The distinction is lane SHAPE, not "office vs not": telecom
  and retail are agentic env lanes that run through the pilot env's
  harness path, which the probe does not drive. BIRD is single-turn with a
  REGISTRY grader (`sql_exec`), i.e. exactly the shape `load_mix_candidates`
  and the probe already handle — question from input.messages, gold from
  grader.expected_answer, grader by name. THE ARITHMETIC FAVOURS PROBING:
  curating a question costs 3 rollouts; an uncurated dud costs a full
  training group (~16 rollouts) before the online filter drops it, and a
  uniform-reward group contributes exactly zero gradient. At a ~45%
  keep-rate, probing 400 BIRD candidates (1200 rollouts) avoids ~220 duds
  × 16 = ~3500 wasted training rollouts. WIRED: `bird_probe_candidates_
  taskspecs.jsonl` added to MIX_FILES, `write_probe_sample()` emits a
  deterministic sha1-ordered 400-question subset of the ~9.4k train split
  (probing all of it would cost more than the campaign), with a test that
  the campaign loader actually reads it back as sql_exec candidates.
  DONE (05:5x): train split exported — 9428 questions, 69 databases, no
  missing DBs, no task-id collisions (train rows carry neither question_id
  nor difficulty, unlike dev, so the id falls back to row position and the
  exporter now rejects collisions rather than silently merging questions).
  TWO UPSTREAM DEFECTS FOUND AND FILTERED BEFORE THEY COST ANYTHING:
  (a) ~5% of train gold SQL does not execute (38/40 on a random sample) —
  those questions would score every attempt 0 and masquerade as
  impossibly hard; (b) 2 of 69 databases have schemas so large the prompt
  alone runs ~9k tokens (works_cycles 37.7k chars / 474 questions,
  mondial_geo 14.2k / 293), far over the conductor's ~5120-token budget.
  `write_probe_sample` now walks the sha1 order skipping both and reports
  the counts: the 400-question sample cost 53 oversized + 13 broken-gold
  skips and spans 63 databases. Probe pool is now 1118 (math 280, rlpr
  300, code 138, sql 400), restarted with BIRD included.
  RETENTION TARGET RAISED 350 -> 600: the probe STOPS as soon as it has
  kept `retention_target` questions, and BIRD is last in MIX_FILES order,
  so a 350 cap would have halted partway through the SQL questions —
  ~295 kept from math/rlpr/code plus only ~55 BIRD, gutting the office
  lane we added BIRD for. 600 exceeds the pool's ~470 achievable, i.e. it
  means "probe everything". Extra cost ≈285 candidates × 3 rollouts
  (~$20-35) for a properly sized office lane; restart cost only the ~32
  in-flight questions, which the journal re-probes.
* **USER CORRECTION (2026-07-28): "you left out some datasets i gave you:
  financial qa, CRMArena, DABstep" — RIGHT, all three dropped too fast;
  re-examined, ALL THREE ARE RECOVERABLE and are being built:**
  (1) FINANCIAL QA — BUILT AND IN THE PROBE. FinanceBench's open release
  (150 rows) stays an eval probe, but the CATEGORY is served by FinQA
  (6,251 train) + TAT-QA (13,215 train), both with real train splits;
  dev/test sealed. Verifiable subset: FinQA numeric exe_ans (6,115) +
  TAT-QA arithmetic/count (5,845) = 11,960 tasks exported. New
  `finance_numeric` grader: last-number extraction, 5e-3 relative
  tolerance, percent normalization driven ONLY by numeric magnitude and
  TAT-QA's explicit scale metadata — no question-wording classification
  (hard rule). 400-question sha1 sample in the probe pool (206 finqa /
  194 tatqa); pool now 1,518; retention target 1200 (= probe everything);
  singleturn safety cap 600->800; orch lane fugu_r5_finance (seed 5108,
  same 16k/0.2/default regime); bounce config regenerated (8 lanes).
  (2) CRMArena — DATA SECURED, NO SALESFORCE ORG NEEDED: the repo
  publishes the full org data as LOCAL SQLITE (local_data/*.db — 8MB v1 +
  34MB b2b + 57MB b2c, all downloaded to director/vendor/crmarena) and
  we already hold the 1,170 tasks w/ answers. Build = query-tool harness
  over the sqlite + exact/fuzzy grader. NOTE: no train split exists —
  training on it burns CRMArena-as-published for reporting; a stratified
  internal holdout will be sealed instead. Bounce-in lane, after DABstep.
  (3) DABstep — GOLDS ARE RECONSTRUCTABLE: answers are stripped from the
  450-task set, but the public leaderboard publishes every submission's
  per-task answer WITH a correctness bit (data/task_scores/*.jsonl, 2,120
  files downloading). Any task some submission solved ⇒ its gold is
  recoverable by majority-of-correct. Same reporting consequence as
  CRMArena (train-on-it = don't report it; the 10-answer dev split stays
  a smoke eval). Agentic lane over the public context files. 589 tests
  green; probe at 758/1518 (sql questions keeping ~25-30% so far).
* **SERVING/VIEW RECIPE (2026-07-28, user-directed):** the conductor is
  served at --max-model-len 32768 (bf16 KV cache, 78,643 tokens; the
  user-approved --kv-cache-dtype fp8 fallback exists in serving_reload if
  KV pressure ever bites; sidecar default is 32768). The conductor's TASK
  VIEW — in the probe exactly as in training — is the first 12k chars of
  the payload (bounded by the 6144-token trainer sequence, the measured
  hardware fit); workers always receive the full payload.
* **RECIPE — TWO LAUNCH-CRITICAL FACTS (2026-07-28):**
  (1) **Typed-contract parsing takes the FIRST JSON value of the
  completion** (`json.JSONDecoder.raw_decode`): the exact-token path
  delivers the plan WITH its end-of-turn framing token, which strict
  `json.loads` rejects as "Extra data". Measured on a real batch: 0% ->
  89.3% parse; the residual ~10% are genuine model formatting errors
  (extra key in a step, truncated JSON) and are the intended reward-0
  signal. Dry gates missed this because the plain chat API strips the
  framing token.
  (2) **ORPHANED ENV SERVERS MUST BE KILLED BEFORE EVERY RELAUNCH.**
  Killing the orchestrator does NOT kill its spawned `spawn_main` env
  servers — they survive, reparent to init, and keep holding their ZMQ
  ports, so the next orchestrator CONNECTS TO STALE SERVERS RUNNING OLD
  CODE. This masked fix (1) across four consecutive relaunches (offline
  scored 0.4, live scored 0.0 on the same completion). Relaunch sequence:
  `kill <orch>` -> `pkill -9 -f spawn_main` -> verify zero
  `spawn_main`/ZMQ listeners -> launch. A permanent SCORE-REJECT
  diagnostic now logs lane + exception + raw head/tail from the env's
  parse-failure branch; it identified this in one shot after several
  wrong theories.
  VERIFIED CLEAN: after the purge, only 7 rejects (6 extra-key, 1
  malformed JSON) vs previously 100% of rollouts.
* **R5 CAMPAIGN LAUNCHED (2026-07-28):** full preflight 52/52 READY →
  all four components live: vLLM serving (32k, both aliases, GPUs 6-7),
  orch on orch_r5_full.yaml (10 lanes, retention-proportional ratios,
  G=32, price-sorted workers), trainer on train_r5.yaml (GPUs 0-5, bf16
  masters, chunked 6144/12, ratio_clip e^2), merged-reload sidecar
  watching for checkpoints. CADENCE (user 2026-07-28): trainer
  save_steps=5 — checkpoint + merged serving reload + BUDGET REVIEW every
  5 steps (monitor sums recorded worker cost_usd; OpenRouter dashboard
  authoritative). WATCH DURING STEPS 1-5: packer-drop rate, sample-split
  rate, mismatch_kl. Hard review still at step 50 (~$2k envelope for 100
  steps at pre-price-sort rates; cheapest-first routing should land
  below it). Preflight fix that landed during
  launch (recipe): the tasks_in_manifest check now requires a NON-EMPTY
  manifest∩lane-ids intersection (shared lane id lists across six
  single-turn envs and oversized manifests like telecom's 2,171-row file
  are by-design; only a disjoint manifest is fatal).
* **PROBE COMPLETE + FINAL MIX DECIDED (2026-07-28, user-decided):**
  probe stopped by user at 1,895/1,968 (96%) — 440 RETAINED (23%
  overall). Per lane: math 108 (39% keep), code 66 (48%), rlpr 61 (20%),
  bird 50 (13% — rejects skew TOO EASY, 214 vs 133), finance 70 (19% —
  rejects skew too easy, 241 vs 53), dabstep 85 (20% — rejects skew
  UNSOLVED, 299 vs 32). Office-shaped = 205/440 (47%) of the curated
  pool. Exported with G=32: r5_{math,code,rlpr,bird,finance,dabstep}_
  taskspecs.jsonl + pilot_config_singleturn_r5.json (typed_control,
  ow_* price-sorted pool, group 32).
  MIX DECISIONS (user): (1) LAUNCH WITH THE FULL 10-LANE MIXTURE FROM
  STEP 0 — the two-phase bounce plan is retired since every office lane
  is ready pre-launch; agentic share 32% from the start. (2) Ratios =
  retention-proportional with the 10% floor: math .1668 / code .102 /
  rlpr .0943 / bird .0773 / finance .1082 / dabstep .1314 (single-turn
  0.68) + agentic tool_dialogue .08 / repo_terminal .07 / telecom .09 /
  crm .08 (0.32). (3) G=32, 8 questions/step at batch 256 (decision
  rule's output at 23% keep; yield curve nearly flat vs G=16).
* **CAMPAIGN WORKER ROUTING = OPENROUTER CHEAPEST-FIRST (2026-07-28,
  user-decided):** all campaign worker pools (ow_* single-turn, td_*
  tool/telecom/crm, term_* terminal) carry provider_sort "price" —
  OpenRouter routes each call to the cheapest provider serving the model.
  Accepted cost: occasional bad-provider timeouts, which the trainability
  machinery excludes from loss AND advantage baseline (a lost rollout, not
  a false negative), bounded by per-call timeouts + retries. This
  supersedes the 2026-07-26 price-sort ban FOR THE CAMPAIGN ONLY — evals
  keep default ordering (the ban's origin was a stalled eval). The probe
  ran on default ordering (same weights, different serving hosts) — the
  seam is provider selection only, not model or regime.
* **SEQUENCING UPDATE (2026-07-28, user-decided): launch WAITS on a
  joint mix-balancing decision after the probe.** On probe completion the
  deliverable is the balancing picture — per-lane retained counts and
  keep-rates, the retention-proportional ratio suggestion (10% floor),
  agentic share at launch vs after the office bounce, and the G
  recommendation — for the user to set the final mix; export/preflight/
  launch follow that decision.
* **PROBE REGIME (2026-07-28, final): concurrency 64, interleaved across
  types** (`--probe-interleave` round-robin; wall-clock is provider-bound,
  so concurrency trades time not spend; interleaving overlaps DABstep's
  local script-grading with network-bound types and surfaces every type's
  keep-rate early). Conductor client timeout 600s; sampler failures are
  recorded as infra errors exactly like executor failures (reward None →
  incomplete group → re-probed on resume), so no transient serving error
  can end a probe run. MEASURED: SQL keep-rate ~11% (uniform rejects: 153
  all-solved vs 97 all-failed — BIRD skews EASY for this pool); the ratio
  floor guarantees the small lane ~8% of the single-turn share.
* **ALL THREE RECOVERED DATASETS BUILT (2026-07-28) — every dataset the
  user named is now in the mixture:**
  DABSTEP (450/450 tasks recovered): every task has ≥27 correct public
  submissions (median 418; median top-answer agreement 0.89), so golds =
  the SET of accepted variants with ≥2 votes. Lane shape: single-turn
  code-exec — the worker writes ONE python script, `dabstep_exec` runs it
  isolated with CWD = the context dir (23MB payments.csv read by the
  script, never inlined) and matches its last stdout line (numeric-
  tolerant / casefold). Probe-curated like all single-turn lanes. Prompt
  puts the QUESTION FIRST — the 22k-char manual would otherwise push it
  past the conductor's 12k-char view (caught before export). Only 3 tasks
  have 'Not Applicable' golds, so no NA-gaming surface. 5 tests.
  CRMARENA (offline, no Salesforce): new `crm_query` harness — read-only
  row-capped run_sql tool over the published sqlite (16 tables), plain
  answer ends the loop, exact-match grade (None golds require literal
  "None"); transcript CONTINUES across workflow steps (the tau2 lesson,
  tested). fuzzy_match/knowledge_qa (130) EXCLUDED — judge metric. 1,040
  exact tasks → 832 train + 208 SEALED stratified holdout (26/type,
  final_eval_only). 6 tests.
  WIRED: probe pool now 1,968 (math 280 / rlpr 300 / code 138 / sql 400 /
  finance 400 / dabstep 450); probe restarted losslessly, running.
  orch_r5 = 8 lanes; bounce = 10 lanes via NEW `ultra.r5_bounce_config`
  (the hand-copied bounce went stale once already; now regenerated with
  lane/ratio consistency checks). office pilot config + safety carry
  office_crm (832 cap, td_* workers) under the user's blanket training
  approval. 600 tests green; bounce preflight: all office_crm checks
  PASS, remaining failures = the awaited probe exports.
  REPORTING LEDGER (recorded once, applies from here): trained-on and
  therefore NOT reportable-as-published: DABstep, CRMArena (internal
  holdout instead), tau retail (train split; airline test sealed), tau2
  telecom (114 sealed), BIRD (dev sealed), FinQA/TAT-QA (dev/test
  sealed). FinanceBench 150 stays a clean eval probe.
* **TRAIN HARNESS + CONFIG RE-REVIEW COMPLETE (2026-07-28,
  user-requested):** full pass over train_r5.yaml, both orch configs,
  safety manifests, harnesses, launch assembly. Recipe facts from the
  pass: single_turn safety cap = 800 (sized to the grown probe pool);
  export() emits ratio suggestions for BOTH orch configs (0.17 and 0.32
  e2e shares); bird lane stays max_turns 2 — measured over all 400
  candidates (median 1608 / max 4175 tokens), 0 overflow on turn 1 and
  1.8% only-if-repair-turn, bounded and visible in the packer-drop gate;
  multi-step workflow steps CONTINUE a harness's environment state
  (regression-tested for tau2_solo and crm_query). Verified: trainer and
  launcher agree on all paths, sequence_chunks 12, template
  qwen3_nothinking matches serving, ratio_clip e^2, save_steps 10 =
  reload cadence.
  RECIPE: every probe-curated grader maps to an export lane
  (probe_manifest_export LANE_BY_GRADER / MANIFEST_BY_GRADER — unmapped
  graders and journal/manifest mismatches fail loudly, never silently);
  orch_r5.yaml carries a bird single-turn lane (seed 5107, same
  16k/0.2/default-effort regime); orch_r5_bounce_office.yaml is always
  REGENERATED from orch_r5.yaml (ultra.r5_bounce_config), never
  hand-maintained.
* **TELECOM/RETAIL STILL ARE NOT PROBED — AND DO NOT NEED TO BE
  (2026-07-28, decided on data):** the probe reads only the four hard_mix_* banks (718
  single-turn questions); office_telecom and tau retail enter training
  unprobed and are filtered instead by the orch's ONLINE difficulty
  filtering (easy_threshold 1.0 / hard_threshold 0.5, 15% recycle) — the
  same variance test applied live. That mechanism is proven on this stack:
  stage2 step-200 buffers hold 105 easy + 134 hard filtered examples.
  Draw representativeness verified by gold-action count: the 240-task
  telecom draw tracks the 2171-task pool within ~2pp at every difficulty
  level (e.g. 6 actions 21.2% vs 20.1%). DECISION: no paid spot-check, no
  re-draw — the live filter handles residual duds (a useless group costs
  ~$1.60 and shows up in the filter rate within the first steps).
* **PROBE PAUSED BY USER — CREDITS RUNNING LOW (2026-07-28, 05:11):**
  user asked to stop until they top up; probe killed cleanly at 239/718
  complete, 100 retained (41.8%): math 79/181 (44%), code 15/32 (47%),
  rlpr 6/26 (23%). No work lost — resume is `--probe-only
  --probe-concurrency 32`, which re-reads the journal and re-probes only
  incomplete rows. NOTE the probing order has been math-heavy so far, so
  the 79/15/6 split reflects sampling order, not final composition; the
  projected end state remains ≈123 math / 65 code / 69 rlpr.
* **RATIO RECIPE (2026-07-28):** env_ratios are computed at launch by
  `env_ratios_from_retention` — the single-turn share splits by retained
  counts (equal draw pressure per task) with a 10% floor so a thin lane
  never vanishes; ratios sum to exactly 1.0 with the e2e share.
  `export()` returns the suggestion for both orch configs.
* **PACKER-DROP GATE PRE-CLEARED OFFLINE (2026-07-28, zero-paid):** this
  was listed as runtime-observable only, but it can be computed. Tokenized
  every lane's REAL conductor prompt with the 27B's own tokenizer:
  math n=72 median 1125 / max 2780, code n=15 max 2149, rlpr n=6 max 1104,
  office_telecom n=240 max 935, tau retail n=500 max 827 — zero prompts
  over the 5120 budget (6144 minus the 1024 plan cap). For the 2-turn
  single-turn lanes the feedback turn is bounded by construction: the
  revise instruction caps the executed outcome at 1500 chars = 245 tokens,
  so the worst case is 2780 + 1024 + 245 + 1024 = 5073 tokens, leaving
  1071 headroom under sequence_len 6144. Packer drops should be ~0; if
  step 1-5 shows otherwise, the cause is elsewhere and worth halting for.
* **R5 LAUNCH RUNBOOK (current as of 2026-07-28 — supersedes the older
  "remaining before launch" list above):**
  1. Probe runs to POOL EXHAUSTION, not to target: the mix holds 718
     candidates and keep-rate is ~43%, so max retained ≈312 and the
     `--retention-target 350` default can never trigger. Remaining ≈517
     candidates at the c32 rate.
  2. `python -m ultra.probe_manifest_export` → r5_{math,code,rlpr}_taskspecs
     + pilot_config_singleturn_r5.json.
  3. Set `rollouts_per_example` in orch_r5.yaml from
     `choose_group_size(keep-rate)` (43% keep ⇒ G=16).
  4. `python -m ultra.r5_preflight` must print READY TO LAUNCH. It is the
     gate, not a formality: today it says DO NOT LAUNCH.
  5. USER flips `approved:true` in the safety manifests that apply.
  6. Launch: sidecar-serve (owns :8011, serves BOTH model names) → orch →
     trainer → sidecar watcher.
  7. Watch steps 1-5 for the two runtime-only gates: packer-drop rate and
     sample-split rate. Hard budget review at step 50.
  OFFICE BOUNCE (separate, at a checkpoint boundary): flip approval in
  live_safety_r5_office.json → stop orch → resume from checkpoint with
  orch_r5_bounce_office.yaml (adds office_telecom 240 tasks, grows tau
  retail 60→500, ratios shift office-ward). BIRD lane bounces the same
  way once its manifest and lane artifacts are exported.
* **OFFICE LANE 1 LIVE SMOKE PASSED (2026-07-28, ~$0.10 spend):** the one
  path offline tests cannot cover is a real model driving tau2's tools, so
  one task was run live through the actual harness with a one-worker pool
  (glm-5.2, 12-turn cap, outside the pilot lane — the unapproved safety
  manifest correctly blocks live LANE runs). RESULT: solved in 8 turns,
  tau2 programmatic reward 1.0, termination "completed", $0.0955. Real
  tool-calling, real env state, real reward — the lane is de-risked before
  any campaign spend. COST SHAPE (matters for the bounce): 81.6k prompt
  tokens vs 267 completion tokens — cost is dominated by re-sending tau2's
  ~6k-token domain policy plus the growing transcript every turn, NOT by
  worker thinking. At ~$0.10/rollout the office lane at ratio 0.15 runs
  ≈$3-4/step; prompt caching is the obvious lever if that needs trimming.
  OFFLINE INTEGRATION ALSO PROVEN (zero spend): a typed single-step plan
  fed to the pilot runtime's own `score()` for the office_telecom lane
  drove executor → tau2_solo harness → tau2 env → reward mapping and
  returned 0.5 — the "valid workflow, task unsolved" trainable signal
  (fake worker emits prose, not tool calls). Plan parse, worker-id
  resolution, harness routing by `environment.harness`, and reward
  mapping are all confirmed on the real config, not mocked.
* **OFFICE LANE 2 BUILT: BIRD text-to-SQL execution grader + manifest
  (2026-07-28):** new `sql_exec` grader (registered; 10 tests) implements
  BIRD's official execution accuracy — candidate and gold both run in
  ISOLATED subprocesses (`grading/sql_exec_runner.py`, read-only URI
  connection, 2GB address-space cap, parent wall-clock kill) and are
  compared by a sha1 digest of their sorted row SETS, with BIRD's type
  semantics left unnormalized so our numbers stay comparable to published
  ones. A runaway recursive CTE times out to 0.0 instead of hanging the
  training loop; writes are refused and the benchmark DB is verified
  unchanged. REWARD-HACK CLOSED: a bare `SELECT 1` matched a real dev task
  whose answer is 1 — the grader now requires the candidate to have
  actually read a table, detected via sqlite's AUTHORIZER callback (exact,
  not parsed from query text). VALIDATED ON REAL DATA: 60 random BIRD dev
  tasks, gold replay 60/60 = 1.0, `SELECT 1` false positives 0/60, ~0.10s
  per grade. `ultra.bird_manifest` (4 tests) emits single-turn direct_qa
  rows carrying question + BIRD evidence hint + the DB's own DDL (read
  from the sqlite file, so the schema shown is exactly what the grader
  executes against; max DDL 7.3k chars ≈ 2k tokens), and REFUSES to build
  a training manifest from the sealed dev split. Train export pending the
  8.9GB train.zip download (~1.5GB in).
* **OFFICE TRIAGE UPDATE — DABstep IS NOT TRAINABLE (2026-07-28,
  measured):** pulled the real dataset: DABstep's 450-task main set ships
  with `answer` STRIPPED (held-out leaderboard); only the 10-task dev
  split has answers. There is no verifiable reward to train on, so it is
  removed from the lane plan — it can still serve later as a leaderboard
  submission. CRMArena also downgraded in practice: its 1170 tasks (975
  answered, exact/fuzzy match) require a provisioned SALESFORCE ORG to
  query, so it stays the heaviest option and is deferred, not next.
  FinanceBench also DROPPED on measurement, not judgement about judges:
  the open release is 150 rows TOTAL, of which only 52 have cleanly
  numeric answers — negligible as a GRPO lane at any reward quality (it
  may still serve as a small eval probe later).
  REPLACEMENT — the cheapest office win was already in hand: tau retail
  expanded from the 60-task pilot slice to the FULL 500-task TRAIN split
  (`tau_retail_train_full_taskspecs.jsonl`, materialized via the existing
  high-action generator; strict superset of the 60, all train, airline
  test-only and untouched). Bundled into the same bounce: the pilot
  config's tool_dialogue lane and its safety cap both grow to 500
  (`expand_lane`, tested), and the bounce orch yaml points that lane at
  the full manifest. BOUNCE DRY-LOADED OFFLINE: tool_dialogue 500 rows,
  office_telecom 240 rows.
* **PROBE THROUGHPUT (2026-07-28, measured):** probe wall-clock is
  provider-side worker latency, not GPU (vLLM idle at concurrency 16) —
  each question is 3 independent 16k-token worker calls, so
  `--probe-concurrency` trades wall-clock without changing total spend
  (~10-15 completes/hour at c6, near-linear scaling above).
* **LAUNCH PREFLIGHT BUILT — `ultra.r5_preflight` (2026-07-28, 9 tests):**
  one command that answers "is it safe to spend money on this launch?"
  against the files and processes actually referenced by an orch config:
  lane/ratio alignment and sum, group size divides batch, every lane's
  pilot config + task manifest + safety manifest exist and are non-empty,
  each LIVE lane's safety manifest is version-correct AND APPROVED AND
  covers the lane's workers within its example cap with yunwu off, every
  lane task id actually present in the manifest that lane reads, trainer
  and orch sequence_len agreement, and the endpoint answering to the EXACT
  model name the orch will send. Missing files are REPORTED, never raised.
  Current verdict on orch_r5.yaml: 12 pass / 9 fail — DO NOT LAUNCH, all
  failures being the probe-produced single-turn manifests. On the office
  bounce config: 17 pass / 11 fail, correctly flagging both office lanes
  as NOT APPROVED. Confirmed (not assumed) that the sidecar starts vLLM
  with BOTH `--served-model-name` values, so the orch's path-style model
  name resolves once the sidecar owns :8011; the current hand-started
  server serves only the short alias and is replaced at launch.
* **DRY GATE: UNCONSTRAINED TYPED PARSE RATE = 0.958 (2026-07-28, PASS,
  gate ≥0.70, zero-paid):** 12 mix questions × 4 samples at temp 1.0 from
  :8011 with NO response_format (mirroring the orch's unconstrained
  sampling): 46/48 parse via parse_typed_workflow; 37/46 plans are 4-step
  trees (topology preserved without the grammar); 2 failures = truncated
  JSON at the 1024-token plan cap → reward-0 format signal by design.
  Risk #2 of the integration plan (train/serve decode delta) is bounded:
  the policy barely needs the grammar. Combined with the earlier dry-batch
  mismatch_kl 0.0049, the remaining launch gates are runtime-observable
  only (packer-drop rate, sample-split rate — watch during steps 1-5).
* **PROBE PAUSED — OPENROUTER CREDITS EXHAUSTED (2026-07-28):** at ~200
  candidates probed / 57 retained (28.5% keep under 16k), every worker
  call began returning HTTP 402 (insufficient credits); ~300 subsequent
  journal rows are empty/incomplete groups. The outage protection worked:
  incomplete groups were skipped WITHOUT verdicts and re-probe
  automatically on resume — no wrong classifications entered the retained
  set. Probe stopped cleanly. RESUME = top up credits → relaunch
  `r5_launch --probe-only` (journal-resume keeps the 200 valid probes,
  re-probes the 402-era rows). Then: export manifests → dry gates →
  user-approved safety manifests → launch.
* **WORKER-REGIME FINAL v2 (2026-07-27, user-decided after the effort/cap
  matrix): DEFAULT effort + max_tokens 16384 + temp 0.2 — "curation as
  filter".** User caught the cost signature first (most worker calls 40k+
  tokens under uncapped-default) and correctly challenged transplanting the
  paper's 4k cap: the paper's pool (frontier Opus/GPT/Gemini) was naturally
  concise; ours couples competence to thinking length. MEASURED (36-call
  matrix, 3 informative hard-tier questions × 4 pool models × 3 regimes):
  the effort knob cannot make this pool concise — models fill ANY budget
  (low+6k: 9/12 truncated at 5.9k avg; medium+12k: 10/12 at 12k; default+
  24k: 7/12 at 21.8k; solo-call solves 0/12 everywhere — solve rates only
  meaningful at plan level). RESOLUTION: cap at 16k and let CURATION make
  it coherent — the probe retains only questions whose reward variance
  exists within the budget; needs-40k-thinking questions truncate →
  uniform-0.5 → correctly rejected as untrainable under this regime. The
  probe thus filters difficulty AND affordability simultaneously.
  Deployment/gates keep full-strength workers (transfer verified there).
  Re-priced: probe ≈ $150-200, campaign ≈ $1-1.5k. The 75 uncapped-probed
  candidates archived (probe.jsonl.uncapped-regime.bak); full re-probe
  under 16k for regime purity.

**WORKER-TRUNCATION CONFIG DEFECT FOUND AND FIXED (2026-07-27, user-
caught):** most OpenRouter requests failed with length > 4096 — the
collection executor applied the paper's 4096 cap WITHOUT the paper's
minimal-reasoning constraint, so workers at provider-default effort burned
the budget on reasoning and truncated before answering; rewards measured
truncation, not ability. The worker_loop docstring documents this exact
failure mode from 2026-07-26; the collection paths reintroduced it. USER
DECISION: default reasoning stays (consistent with "high effort harms
benchmarks"); the cap is the variable. Verified live: even 16384 truncates
on olympiad-tier questions (minimax-m3: 12,540 reasoning tokens, no
answer) — default-effort reasoning is effectively unbounded on that tier;
production already serves with max_tokens UNSET for this reason. Fix
applied to `prompt_ab_stage2.py` and `r5_launch.build_campaign` (this
would otherwise have poisoned ALL r5 training rewards). The stage-2 run
under the truncating regime was stopped and its data discarded (archived
offline); the prompt A/B re-run design is an open decision: (a) corrected
regime + difficulty-stratified questions (~$60-120), or (b) skip and
freeze arm A — the SFT-matched prompt, which is also the registered
no-margin outcome (~$0, recommended). The r5 base decision (27B) is NOT
reopened: both arms shared the regime on easy holdout questions (mean
0.90+, truncation rare there), paired comparison, decided by pre-committed
default. Durable observations that survive the discarded run: the
GO-sequence order (freeze prompt BEFORE probe) removes the format-variance
axis from probing so retention measures correctness variance; expect
substantial never-solved mass at the hard end of the prior-ordered mix.

**GEPA optimize_anything — decision framework (2026-07-27, user-confirmed):
NOT on the r5 critical path; two explicit triggers.** Assessment: our
plumbing already matches GEPA's API (RolloutCollector = evaluator; journaled
rewards/parse failures/plan shapes = actionable side information), but
GEPA's sweet spot is cheap low-noise evaluators (its examples: 100-350
metric calls on code/kernels), while ours is paid and noisy — a credible
prompt score costs ~$3-6 (10 q × 4-8 rollouts) and the effects at stake are
±0.03-0.05 mean reward, i.e. noise-level at that spend; a 50-candidate run
= $150-300 of poorly-resolved scores. Live stage-2 evidence says prompt
text is a WEAK correctness lever anyway (guidance saturates format at parse
1.000; paired correctness lift ≈ 0 so far), and the serving prompt must be
frozen BEFORE the probe pass (exact-token + probe-ordering), so any search
phase delays the critical path to optimize the weak variable.
TRIGGER 1 (pre-r5, only if stage 2 falsifies the above): a non-control arm
clears the pre-registered +0.05 margin → prompt content demonstrably moves
correctness → run ONE bounded GEPA pass over the guidelines slot before
freezing: seed = winning arm, evaluator = collector on ~10 probe questions,
ASI = failure transcripts, HARD CAP ~$150.
TRIGGER 2 (post-r5, no freeze constraint): eval-time prompt tuning against
the MMLU-Pro ceiling failures (9/10 persuasive-wrong-answer convergence) —
failure transcripts as ASI, reflection proposes counter-instructions.
This is the natural first real GEPA use.
Outside these triggers: no prompt-search spend.

**OOD FEW-SHOT ASSEMBLY SHIPPED (2026-07-27):**
`ultra/ultra/fewshot_assembly.py` — mined high-advantage plans → typed-
contract few-shot block for the guidelines slot. Selection is outcome+
structure only (advantage rank, one exemplar per topology shape — no
text/keyword classification); rendering is model-anonymous (profile_a...,
access "all" expanded to positions), bounded, deterministic, never
truncates mid-JSON. On the real mine: 4 examples / 2,948 chars. 6 tests.
ADOPTION GATED: enters the serving prompt only if the stage-2 verdict says
few-shot content helps (arm C evidence); otherwise it stays a built,
unused capability.

**R5 BASE DECISION protocol (pre-registered, decisive by construction):**
27B (typed SFT) vs 8B stage2 (RL-trained, native three-list contract via the
legacy parser) on the SAME 60 SFT-holdout questions (neither model trained on
them) × G=8 at temp 1.0, executed on OUR pool with constrained workers.
Rule (locked before data): informative-fraction gap ≥15pp decides; else
mean-group-reward gap ≥0.08 decides; else **27B by pre-committed default**
(paper scale finding + typed contract + serving stack). Script:
`scratchpad/r5_base_decision.py` (--decide prints the verdict).
**FIXED PATH AFTER: winner → GRPO (this recipe) → gates → final benchmark
table → ship. No further forks.**

Remaining build items: LCB grader integration; exact-token batch adapter for
the surogate trainer. DeepSWE/TB stay EVALUATION-ONLY (rollout cost);
recursion is a post-r5 finetune (paper: 20 iterations, half-batch recursion,
0.25 discount).

## R4 CONDUCTOR TRAINING — expert iteration on measured topology winners

The mandate (user, 2026-07-26): the router must make multi-step workflows beat
solo workers on any task type. Method: **expert iteration**, not online GRPO —
r2 emits near-identical chains per task type, so sampled plan groups would have
~zero reward variance (zero GRPO advantage, paid rollouts teaching nothing). A
forced topology menu guarantees contrast; the measured winner per TASK TYPE
becomes the SFT target.

Pipeline (rebuilt 2026-07-26 after the v1 postmortem, in `scratchpad/`):

| Stage | Tool | Design rules bought by failures |
|---|---|---|
| Measure arms | `topology_policy_data.py` | per-task JSONL; failures excluded not wrong; stratified round-robin loading; vote3 control arm; no max_tokens |
| Build dataset | `build_conductor_topology_sft.py` | BARE prompts; TYPE-level winners; collaboration only where measured > solo; augmentation + every-5th holdout; targets round-trip under the live contract |
| Train (v3, running) | `fugu_r4_topology_sft/train_r4_topology_sft.yaml` | **topology rows ONLY** — the v1 mix with `fugu_typed_sft_mixed_v3` out-competed the tree target at the first branch token, and its "retain live control" rationale died with the r2≈0 finding. 40 steps, fp8-hybrid, 8 GPUs |
| Serve | `fugu_r4_topology_sft/merge_and_serve_v3.sh` | **merge the delta into BF16 weights, serve merged (TP4)** — vLLM LoRA serving no-ops on this hybrid stack. Merge script hard-fails unless all 256 deltas apply. Adapter keys are wrapper-style (`language_model.layers.*`); the BF16 text class wants `model.layers.*` |
| Gate | `gate_r4_topology.py` | shape check on 40 holdout tasks: ≥80% diverse trees. Baseline (base model = r2) already recorded at **0% trees** (singles + chains) in `gate_shape.log`. Then execution gate ≥ solo |

The trained layer stays model-agnostic: targets carry anonymous profile_refs
derived from capability tags. The conductor learns "science reasoning → diverse
tree; instruction tasks → whatever measured best; coding → build-and-debug with
the coder profile" — and the binding decides which MODEL backs each profile.
Tree execution is production-viable: the heavy orchestrator wave-schedules
independent steps concurrently (tree latency ≈ 1 leaf + aggregator, verified by
`ultra/tests/test_fugu_heavy_topology.py`).

Coding phase (next after r4 gates): kimi-k3 enters the binding as coder slot
(data edit, no retraining); build-and-debug topology measured on unit-test
verifiable coding tasks (reward = tests pass, no judge needed); coding becomes
the third contrast type in the same SFT pipeline. Serves the terminal-bench
goal too.

### Coding-phase recon (2026-07-26, 5-agent survey — verified against files)

**Primary task source: DeepSWE** (`director/vendor/deep_swe/`, 113 Harbor-v1.1
tasks: 35 TS / 34 Go / 34 Py / 5 JS / 5 Rust). Reward is binary + fractional
from HELD-OUT tests in a separate pristine verifier container (reward 1 iff all
fail-to-pass tests pass and no pass-to-pass regresses); deterministic subsets
(`--sample-seed`) give identical tasks across arms for paired stats; per-task
`reward.json` = the per-task record requirement; `solution/solve.sh` enables a
ZERO-TOKEN oracle preflight of the whole loop before any paid call. Blockers,
each with a verified fix:
1. harbor 0.8.0 (installed) never runs `pre_artifacts.sh` → model.patch never
   captured → reward 0 by construction. Fix: `datacurve-pier` 0.3.0 (PyPI,
   fresh venv — conflicts with harbor) or a 5-line post-agent hook.
2. Task images lack tmux/asciinema and run `--network none`, so terminus-2
   can't run inside as-is. Fix: pre-bake derived images (apt at build time).
3. pier's squid egress proxy only allows ports 80/443 → router :8022 denied.
   Fix: front the router on host port 80, or patch pier's Safe_ports.
4. ECR anonymous pulls rate-limit (~840MB/image, ~20GB for 25 tasks): prefetch
   with backoff before the run.
5. NEVER relax `allow_internet` — upstream repos contain the real fixes; the
   images gc future git history precisely to prevent leakage.

**Fallback (same-day, cheapest): EvalScope live_code_bench** — programmatic
local test execution, api_base flag proven against router :8022; weaker
topology fidelity (single request; topology lives behind the endpoint).
swe_bench_pro (731 tasks) is viable but strictly dominated by DeepSWE;
terminal_bench is the wrong domain for a coding verdict (it feeds the
agentic_terminal domain instead).

**Kimi-k3 binding facts:** exact id `moonshotai/kimi-k3` already used in past
ow bindings (slot shape: role_prior [mathematician,coder,reasoner]); the
pool-binding path needs NO code change for a coder slot. `ultra/providers.py`
maps logical "kimi" → kimi-k2.7-code (stale; only matters for the OpenCode
path). No k3 pricing recorded in-repo; cost control stays provider price-sort.

**Infra note:** the only evalscope install lived in the SESSION /tmp scratchpad
(GC risk for the whole benchmark phase) — a durable copy now lives at repo
`scratchpad/esvenv` (evalscope 1.9.1; add `[sandbox]` before mbpp-class
benchmarks, `[terminal_bench]` before TB runs).

**Conductor generalization observed (2026-07-27, free probe):** given the NEW
5-slot binding (kimi coder added), r4 — trained only on 4-slot diverse trees —
emitted a NOVEL 5-step plan for a DeepSWE task: two independent analysis
leaves (glm, minimax) → inkling implements (access 0,1) → kimi builds (access
0,1) → deepseek verifies/aggregates (access 2,3). It incorporated the unseen
coder profile unprompted and shaped a build-flavored topology. Light-path
per-turn selection v1 takes the plan's first role (glm here) — closing the
"coding turn → coder profile" mapping properly is the r5 training item
(coding as the third contrast type).

### Coding measurement: STOPPED after 1/20 (2026-07-27) — decision record

**Verified external comparator (Together AI, independent, 113 tasks × 4
trials, kimi at MAX effort): kimi-k3 DeepSWE pass@1 68.5%, pass@2 82.0%,
pass@4 89.4%, $4.65/rollout** (Fable 5: 69.9 / 80.2 / 88.5 at $13.41).
This SUPERSEDES the uncited "67.5" that sat in the Stage-3 table since
2026-07-25 (that row also said "117 tasks" vs the vendored 113 — treat any
uncited figure in this file as suspect until sourced).

Three findings that ended the run:
1. **The honest bar is pass@2 (82.0), not pass@1 (68.5).** Our
   build→verify→fix spends TWO kimi attempts; kimi alone with two attempts
   already resolves 82%. Beating 68.5 would prove nothing. Any future coding
   claim must clear ~82 or show recovery a blind retry cannot produce.
2. **Cost measured, not estimated: pier reported $1.52 and $2.00 per AGENT
   RUN** (3 runs/task) → **~$5-6/task, ~$100-120 for 20** — on ~12M input
   tokens for 10 trials. That buys ~100 GRPO steps instead.
3. **n=20 cannot resolve it.** SEM ≈ 11pp vs an external aggregate; only a
   large paired effect would register. Paying $100 for a probable
   "suggestive but not conclusive" is the displacement activity to avoid.

Kept: the harness (oracle-verified, smoke-passed, live-graded), the
per-task paired design (the driver's BUILD run IS kimi-solo on the same
task — no historical kimi DeepSWE number ever existed here; it is generated
in-run, and it is NOT free: the build phase is paid, it simply requires no
ADDITIONAL run), and one graded record (abs-stepped-slices: solo build 0/6
f2p + 1 p2p regression, partial 0.417). Re-open only with the full 113 ×
multiple trials, after r5 — never as a 20-task teaser.

**CODING SMOKE PASS (2026-07-27, ~$2):** full Fugu-Ultra build-and-debug
pipeline executed live on a DeepSWE task — conductor plan → 3 sequential
agent runs (build 8.8KB patch → verify with REVIEW.md → fix) → held-out
verifier graded: **reward 0 but partial 0.91 (18/20 fail-to-pass passed,
3/3 pass-to-pass intact)** with builder=inkling (selection bug: step-order
beat tag-priority; fixed for the measurement — coder/kimi builds). Two
integration lessons burned in cheaply: mini-swe model ids need pier's
`openrouter/` prefix (its native chat-completions class; `openai/` selects
the Responses API, which the provider rejects), and per-run models flow via
the driver's label→model map. Fractional f2p/p2p rewards confirmed live —
the continuous metric for paired stats.

**Coding phase bootstrap COMPLETE (2026-07-27, zero paid calls):**
pier 0.3.0 in `scratchpad/piervenv`; 3 task images prefetched; **oracle
preflight PASS — reward 1.0 on 3/3 tasks in 50s** (whole loop verified:
agent container → pre_artifacts patch capture → separate verifier → held-out
tests → reward.json). Binding now 5 slots (kimi-k3 coder,
revision pool-c-general-use-plus-coder-20260727); competence table has a
coding domain (kimi 88.3 / glm 82.8 / deepseek 77.4 published); ALL slot
efforts capped at `high` (xhigh/max harms benchmarks — tested twice, user
rule). OPEN DESIGN DECISION before paid measurement: coding tasks are
MULTI-TURN agentic — topology applies at TASK level (builder run → verifier
run → builder fix, sequential agent runs), NOT per-turn trees (4× calls per
agent turn would explode cost and mismatch the scaffold finding). Options:
(a) per-turn routing (router routes each mini-swe turn to the coder slot) —
cheap, measures endpoint-neutrality vs published kimi 67.5; (b) task-level
build-and-debug driver around pier (the paper-validated coding topology) —
the real orchestration claim, needs a small driver. |

## PRODUCT ARCHITECTURE (decided 2026-07-26)

The scaffold finding forces the product shape. The worker interaction loop, not
the conductor or the pool, was the ~22-point gap. So the product does NOT
rebuild a loop — it adopts a proven open-source one and adds only what
orchestration uniquely provides.

**Shape:** a mature agent loop (Terminus 2, already open-source in `harbor` and
already a dependency) drives the terminal and calls a single OpenAI-compatible
endpoint. That endpoint (`ultra/router_endpoint.py` + `ultra/router_server.py`,
served as `fugu-open`) routes each turn to a pool worker and forwards it. This
is Sakana's own deployment shape (`fugu.json` is an agent-CLI model config).

**Built and proven this session:**
- Router endpoint + HTTP server; end-to-end pipeline verified live
  (Terminus 2 → router :8022 → OpenRouter → real solutions, 16k tokens).
- Fugu §3.2.2 memory model: agent trajectories ISOLATED per worker (foreign
  assistant turns masked), environment observations SHARED. Prevents
  orchestration collapse on model switch. (Access-list *selective* sharing is
  the not-yet-built middle ground; needs no training, only matters once the
  conductor switches models.)
- Multi-worker `select_worker`: the report's build-and-debug pattern as a
  per-turn policy (strong builder default; different model on a failure
  signal).

**Key design realisation:** the trained r2 conductor is a WORKFLOW PLANNER;
this architecture needs a PER-TURN ROUTER. r2's planning training does not drop
into per-turn selection. Two paths: (1) a per-turn routing policy [current],
(2) conductor-as-planner emitting a role sequence the endpoint executes.

**The bar orchestration must clear:** the pool's STRONGEST solo under the same
loop, not glm's 82.4%. On TB2.1 that is kimi-k3 (published 88.3). If kimi-k3
solo under Terminus 2 lands near 88, the goal ("as close to Sol 88.8 as the
pool allows") is essentially met by the simplest product — one strong open
model behind a proven loop — and multi-worker routing must beat ~88 to justify
existing. Measuring kimi-solo is therefore the decisive next number.

Solo baselines under Terminus 2: glm-5.2 **82.4%** (34 tasks, paired 10W/1L vs
our old stack, p=0.012); kimi-k3 in progress.

## SCAFFOLD FINDING (2026-07-26) — the gap is the worker loop, not the models

Controlled measurement: **glm-5.2, same model, same provider (OpenRouter),
same benchmark (TB2.1), only the agent scaffold changed.**

| Configuration | Score |
|---|---:|
| glm-5.2 alone + Terminus 2 (EvalScope 1.9.1, Sakana's eval stack) | **83% @ 18 trials** (published 82.7) |
| Our full 4-model conductor stack | 61.0% (89 tasks) |
| Our conductor + static guidance | 61.0% (82 tasks) |

One open-weight model in a mature agent loop beats our entire orchestration by
~22 points. The pool was never the limitation; **our worker interaction loop
loses roughly a fifth of the models' capability.** Every conductor-side lever
tried this campaign (guidance injection, posture distillation, effort tiers,
binding recalibration) was optimising a layer sitting on top of that loss.

**Product consequence:** adopt a Terminus-2-class worker loop and place the
conductor *behind the model interface* as a per-step router — the deployment
shape Sakana's own `fugu.json` implies (`shell_type: shell_command`,
`apply_patch_tool_type`, `truncation_policy`) and the one their published
numbers were measured in. Orchestration must then beat the solo-model baseline
measured under the SAME scaffold to justify its existence.

**Environment lesson (cost this campaign ~$25 in voided runs):** three
infrastructure faults silently invalidated earlier calibration attempts —
Terminus-2's 120s per-command timeout (use `timeout_multiplier: 2.0`), Docker
disk at 93% (image builds fail; keep >400GB free), and over-parallel
environment builds (batch 8-10, not 20). Always check the no-verdict count
before reading a benchmark score.

### Adapter reality audit (2026-07-26, B-norm test)

A LoRA's effect is B×A and B starts at zero, so mean|B| tells whether an
adapter contains real training. Fleet audit: **`fugu_ultra_stage2` (the
200-step GRPO adapter) is REAL** — mean|B| grows 0.037→0.050→0.054 over steps
145→190→final, genuine monotonic GRPO accumulation. **8B pilot lanes REAL**
(0.11–0.14: pilot/stable_lanes/after_workflow_sft); `filtered_recycle` small
but real (0.013–0.022). Dense Qwen3-8B serves LoRA through vanilla vLLM, so
the 8B campaign's training was also APPLIED at serving — its measured effects
were real model changes. **~Zero no-ops: `after_parent_repair_sft` lanes
(0.0002–0.0005)** — conclusions from those lanes were base-vs-base — and 27B
r2 (0.0002). r4 v1 SFT reference: 0.39. Run this test on every future adapter
before gating it.

Consequence (user-surfaced, confirmed in this file's git history at d8f954cc):
**the 8B `fugu_ultra_stage2` conductor is the only genuinely TRAINED
orchestration asset besides r4.** Its documented wins were real trained
behavior — task-conditional routing from anonymous ordinals, same-task
conversions (3/64→19/64), 84% diagnose→derive→verify→implement repair plans.

**8B stage2 PROBED 2026-07-26 (40 holdout tasks, temp 0, local GPUs only):
it emits CHAINS, never trees — 40/40 linear plans (30×3-step, 10×2-step),
0 parse failures, 0 trees. Shape match vs the measured winner: 0/40.**
Method was rigorous: the prompt was recovered from the actual training
rollouts (`output/fugu_ultra_stage2/rollouts/step_199/rank_0.bin`, msgpack
MicroBatch) and verified byte-identical to the rebuilt
`ultra/conductor_prompt.py` prompt on 3 training examples; the base Qwen3-8B
weights had to be re-fetched at the exact training revision `b968826d`. The
un-adapted base emits different text, so the LoRA IS applied — the 200-step
GRPO simply never moved topology off "chain".

Reading: the 8B's trained strength is *chain quality* (its
plan→verify→answer decomposition is coherent and parses 100%), NOT topology
diversity — GRPO on that pool/task-mix rewarded better chains, not trees. On
the evidence, r4 (trees, 40/40 shape, execution 87.2 vs 82.5 solo) is the
better conductor for the current product, and the 8B is not a drop-in
replacement. Its remaining value is as a comparison arm and as proof that
outcome-driven GRPO trains real orchestration behavior at the right LR.

## Conductor training status — POSTSCRIPT 2026-07-26

Every observation in the r3 section below is now explained by the three
findings in Tried and Failed: r2 is a ~zero adapter, vLLM LoRA serving no-ops
on this stack, and SFT targets mixed with near-identical competing prompts lose
the argmax race at branch tokens. The r3 "posture did not transfer" mystery was
never about templates or undertraining — the served model simply never
contained ANY adapter's behavior. Kept for historical record.

## Conductor training status (r3 line, 2026-07-25)

Goal of r3: bake the validated build-and-debug posture into the weights so the
product does not depend on the 380-token guidance block. Two full iterations
ran on the v1.4.0 trainer (bf16 27B + r2 merged, fresh LoRA r16, seq 4096 in 8
chunks, 8 GPUs):

- v1 failed its gate for a data reason: targets were mined from `raw_plan`,
  which production logs POST-translation (`worker_id`/`access`), while the
  model emits the capability contract (`profile_ref`/`access_positions`).
  Training on untranslatable targets moved nothing.
- v3 fixed the format end-to-end (teacher-guided sampling under production
  constrained decoding; 84 short-horizon examples, 54 single-step; 284
  long-horizon reward-1.0 GRPO winners re-serialized). Loss fell 45% both
  runs, long-horizon shape reached 100%, but the single-builder posture did
  NOT transfer to temp-0 serving (0% vs r2's 12-25%) — even on tasks in the
  training set. Prime suspect: chat-template mismatch between the trainer
  (`qwen3_nothinking`) and vLLM serving (model default template). PARKED, not
  abandoned: the fix is template alignment, then re-gate.

The PRODUCT remains r2 + default-on static guidance
(`conductor_guidance_v1.md`), whose components the A/B validated (4/8 failure
recoveries; beat solo kimi-k3 on write-compressor and qemu-alpine-ssh). Gate
instrument is now trustworthy (r2 baseline: 100% contract validity, 12-25%
short single-builder, 88% long multi-step).

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

### Historical bound pool (proprietary campaign, superseded 2026-07-26)

The ACTIVE binding is Pool C, `current_pool_binding_general.json` (see the
worker-pool table near the top). The proprietary binding below is retained for
provenance of the closed campaign's collections and batches only:

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
