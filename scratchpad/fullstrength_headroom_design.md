# Full-strength verdict — PRE-REGISTERED PROTOCOL (2026-07-08)

## REV3 (2026-07-08, user correction — SUPERSEDES the arm design below)

**The subject under test is FUGU-ULTRA (the decomposition conductor), not generic
composition-vs-routing science.** User: "our product is NOT a cost-escalation router!!!
our product is fugu ultra!!!" ([[fugu-ultra-is-the-product]]). Zenith is a source of
borrowable mechanisms FOR Fugu-Ultra, not a competitor to benchmark.

Final design (implemented in `scratchpad/eval_fullstrength_verdict.py`):
- Substrate: the held-out trend60 set (the endgame full-strength comparison MISSION
  reserved), workers UNCONSTRAINED: max_tokens 16384 + reasoning_effort high (the Fugu
  report §4.1.1 setting); temp 0.2/top_p 1.0 unchanged so ONLY effort+cap move.
- Arms: solo__{opus,gpt,gemini,glm} (the bar) · solo2__W (oracle-retry control: binary
  "incorrect, try again", retries only on failures) · fu1 (Fugu-Ultra one-shot, step-145
  adapter via live vLLM :8007, conductor gen unchanged temp 1.0/1024/no-think) ·
  fu2 (Fugu-Ultra + Zenith-borrowed feedback loop: fu1's outcome handed back via the
  multiturn env's REVISE_INSTRUCTION → new plan → execute; turn-1 ≡ fu1 sample).
- Fairness: fu2 and solo2 receive the SAME binary incorrect signal → the loop-vs-loop
  comparison (fu2 vs best solo2) is the honest one; fu1 vs best solo is the base verdict.
- Verdict reads: fu1 − best_solo (does trained Fugu-Ultra beat the bar unhandicapped?);
  fu2 − best_solo2 (does the feedback loop add conductor-specific value beyond retry?).
  If fu2 wins its pairing → the loop ships as part of Fugu-Ultra AND the next training
  run is the 2-turn env (train the conductor to exploit feedback — Stage-2 with evidence).
- Accounting: every pool.call metered (tokens/model, truncation, reported $), hard
  budget stop, resumable rows in scratchpad/fs_verdict_rows.jsonl.

The SWE-smith matrix (below) and SWE-Together tier remain as TRANSFER follow-ups for
Fugu-Ultra (via the conductor→agent shim), not the primary verdict.

---

# [superseded original draft follows]

The single decisive test from the 2026-07-08 pivot (MISSION.md). Everything downstream
gates on this result. Pre-registered BEFORE any run so the analysis cannot chase the data.

## Question

At FULL worker strength (the paper's actual setting — the one regime never tested here),
does a multi-worker composition beat the best solo worker on contested agentic coding,
out-of-sample?

## GO gate (pre-registered, from MISSION.md)

Cross-fitted composition advantage **Δ ≥ +0.08** over cross-fitted best solo.
- Δ < +0.08 → NO: ship the single-call cost-escalation router; agentic training is dead.
- Δ ≥ +0.08 → YES: a real case for agentic training exists (full-strength workers +
  tight cost guard by design).
Secondary (reported, not gating): McNemar exact p on winning-comp vs best-solo pairs;
permutation null for Δ; conductor-arm transfer score.

## Task set (n=50)

- 25 = the derisk-study set (SWE-smith stream, shuffle seed=0, first 25) — keeps the
  paired history; ALL 25 kept (the 6 handicap-all-fail may become contested at full
  strength; dropping them would bias toward the old regime's difficulty map).
- 25 fresh = next instances from the SAME stream (seed=0, positions 26+), filtered to:
  (a) repo ∈ the 12 study-proven repos (local Docker images, proven test-suite cost —
      the gpxpy lesson: slow-suite repos burn the whole step budget),
  (b) instance_id ∉ the study 25.
- Manifest frozen to scratchpad/fullstrength_tasks.jsonl before any worker call.

## Arms (9; fixed before the run — REVISED after reading study/zenith, 2026-07-08)

REVISION RATIONALE (the Zenith report, `study/zenith` + Technical_Report.pdf, user-supplied):
- Zenith's ablation isolates WHAT makes composition win at full strength: repeated
  GAP-FINDING against the original requirement (RALPH = strongest simple baseline,
  3/8 wins) and INDEPENDENT VERIFICATION + adaptive orchestration (Zenith: best mean
  rank 1.38 at 43% of RALPH's cost, 5/8 wins). Fixed upfront work lists (Plan-RALPH)
  rank WORST of the multi-pass methods — and our draft compositions (builder→debug,
  ladder) are exactly that class. Testing only fixed templates could kill the agentic
  case while the real full-strength headroom (gap-finding/verification loops) went
  unmeasured.
- External prior FOR full-strength orchestration headroom: GPT-5.5+Zenith ranks #1 on
  FrontierSWE, beating the same GPT-5.5 under its native Codex harness (avg rank 2.06
  vs 5.53) and beating Opus-4.8+Claude-Code. Composition-over-scaffold is real there.
- CAVEAT (horizon mismatch, recorded now so the verdict is honest): Zenith's wins are
  on LONG-HORIZON tasks (hours-days; premature completion is the failure mode). Our
  SWE-smith tasks are minutes-scale single-bug repairs. If multi-pass arms show no
  lift here, that says "no headroom on short contested bugs" — it does NOT refute the
  long-horizon result; the long-horizon regime would then be the remaining untested
  territory (and a different, more expensive test).

Solos (the bar — "best individual model+scaffold worker"):
1. `direct__opus`   — Opus-4.8, OpenCode agent
2. `direct__gpt`    — GPT-5.5, OpenCode agent (Yunwu; adaptive reasoning = full effort)
3. `direct__gemini` — Gemini-3.5-Flash, OpenCode agent (ultra-track pool "gemini")
4. `direct__glm`    — GLM-5.2, OpenCode agent
Same-model multi-pass (test-time-scaling class — the retry-confound control AND the
Zenith/RALPH mechanism, same model so heterogeneity is isolated):
5. `self_repair__opus` — Opus builds, Opus debugs its own workspace (2-pass).
6. `ralph3__opus` — 3 sequential Opus sessions in one workspace; passes 2-3 use an
   explicit RALPH-style gap-finding prompt ("compare current repo state against the
   problem statement, find the most important remaining gap, close it, run the tests").
   NEW workflow (added to agentic_scaffolds pattern), the paper's mechanism at task scale.
Heterogeneous compositions (the 2 pre-registered derisk-study winners — chosen on OLD
data, so the 25 fresh tasks judge them honestly):
7. `builder__glm__debug__opus` — GLM drafts → Opus debugs (the 0.44 arm)
8. `ladder__glm__gemini__opus` — escalation ladder (the other 0.44 arm)
Conductor (transfer eval, reported separately — NOT part of the composition-vs-solo gate):
9. `conductor__s145` — live vLLM :8007 `default` adapter (= step-145 policy) generates a
   plan per task from the paper prompt; plan → Workflow → executed with the same
   full-strength OpenCode workers. Answers "does text-learned routing transfer?" —
   the Stage-2c ladder's rung 2, folded in for free.

VERDICT SEMANTICS (pre-registered, three-way — sharper than the original binary gate):
- Heterogeneous comps (7-8) clear +0.08 over best solo AND over best same-model
  multi-pass (5-6) → heterogeneity headroom is real → agentic conductor training case.
- Same-model multi-pass clears +0.08 but heterogeneous adds nothing over it → the
  full-strength headroom is test-time scaling + verification, NOT multi-model routing →
  ship the router; bolt a verify/retry loop onto the product (cheap, no RL needed);
  no multi-model agentic training case. (This is the +2/25 retry confound, finally
  instrumented properly.)
- Nothing clears +0.08 → ship the single-call cost-escalation router. Done.
Phase B (conditional, user GO): if any arm lands within ±0.03 of the gate, or the
heterogeneity question stays open, run actual Zenith (hybrid orchestrator) on a 15-20
task subset as the composition ceiling — integration cost (node/ACP inside SWE-smith
containers) is only paid if Phase A says it matters.

All arms run through `run_agentic_workflow` (ultra/ultra/harness/opencode.py) on
SWE-smith containers with the official grader — the exact derisk-study path.
Worker model settings: OpenCode/provider defaults, NO handicap (no max-token cap, no
effort=minimal). GPT-5.5 reasons adaptively (documented Yunwu behavior) — that IS its
full strength. Per-step wall cap ULTRA_OC_TIMEOUT=900s (vs study 600) so full-effort
runs aren't truncated by the old budget; wall-clock, not tokens, is the enforced cap
(the streaming-budgets lesson).

Deliberately NOT included (scope control): claude-code-scaffold Opus solo (premium
scaffold changes two variables at once; if Δ lands near the gate ±0.03 we run it as a
follow-up on the same manifest before declaring), debate/synth arms (lost the derisk
study), flash/kimi/mimo workers (not in the ultra-track pool).

## Cross-fit protocol (the +2/25 lesson, mechanized)

For each of 1000 random stratified-by-repo 50/50 splits (seed=20260708):
- On the TRAIN half: pick best solo arm s* = argmax mean reward among arms 1-4;
  pick best composition c* = argmax among arms 5-8; pick best same-model multi-pass
  m* = argmax among arms 5-6; pick best heterogeneous h* = argmax among arms 7-8.
- On the TEST half: Δ_comp = mean(c*) − mean(s*); Δ_hetero = mean(h*) − mean(m*).
- Report means over splits; CI from the split distribution + a task-level bootstrap.
- Gate reads: Δ_comp against +0.08 (the MISSION gate); Δ_hetero decides WHICH verdict
  branch fires (heterogeneity vs test-time-scaling).
Also reported: full 8×50 matrix, per-arm rates, per-task oracle (labeled AS oracle,
never as headroom), and the handicap-vs-full contrast on the shared 25 tasks.

## Spend guard (the Yunwu-$0 blind spot, closed)

- Token metering: parse OpenCode --format json events for token usage per step;
  price via an explicit per-model table (estimates, labeled as such) so Yunwu-billed
  arms produce a non-zero ESTIMATED cost line. Reported + estimated cost both logged.
- Hard caps: --budget on (reported + estimated) total, checked before each run;
  per-run step cap = len(workflow.steps), per-step 900s wall.
- Execution order: cheap arms first per task (glm/gemini → gpt → opus-involving; the
  opus-heavy multi-pass arms last), so a budget abort still leaves an analyzable
  partial matrix.
- Resume: (task_id, arm) done-set from the output JSONL (same as the study script).
- SMOKE FIRST: 3 tasks × all 9 arms (~27 runs) → measure real per-arm cost/wall →
  re-project the full ~450-run matrix → USER GO before the rest. HONESTY NOTE: the
  MISSION's "$40-60" was estimated on GLM/flash medians with Yunwu costs invisible; a
  full-strength opus-heavy matrix plausibly runs $100-250. The smoke decides; trim
  levers if over: tasks 50→40, drop ralph3 to the opus-solo-fail subset, or drop one
  heterogeneous arm.

## Outputs

- scratchpad/fullstrength_matrix.jsonl — one row per (task, arm): reward, cost,
  est_cost, tokens, wall_s, steps, error.
- scratchpad/fullstrength_verdict.json — cross-fitted Δ, gate verdict, all secondaries.

## Tier 2 — SWE-Together substrate (added 2026-07-08 after studying study/SWE-Together)

The SWE-smith matrix above (Tier 1) has a known horizon risk: minutes-scale single-bug
repairs are the regime where BOTH our probes found nothing to orchestrate, and Zenith's
composition wins live on longer-horizon work. `study/SWE-Together` is the corrective
substrate, already vendored and already flagged in MISSION as the transfer target:

- 109 multi-turn interactive tasks (real user sessions + reactive Gemini user-sim),
  Dockerized (`ghcr.io/togetherbench/*`, local Docker supported), frozen per-task judge
  rubrics → scores comparable across cohorts.
- FULL STRENGTH IS NATIVE: reasoning_effort=high, premium scaffolds (opencode /
  claude-code / codex / mini-swe-agent via Harbor), 1h agent timeouts.
- CONTESTED AT FULL STRENGTH IS PROVEN: leaderboard pass@1 0.39–0.63, oracle ~78 —
  no saturation risk (the Tier-1 abort condition cannot fire here).
- 20–90 min expert-time tasks = exactly the Zenith horizon.
- Plugging a composition/conductor = a custom `UserEnabled*` agent wrapper
  (src/user_agent/agents/ pattern, Harbor import_path) — the "conductor→agent shim"
  MISSION already listed for the endgame. Solos are turn-key today.
- Local state: fresh checkout — no .env, no pulled images, no trials. Needs GEMINI key
  (user sim, every run), ANTHROPIC-compatible judge key (Yunwu base-URL pattern exists
  in their code), provider keys, GHCR pulls.
- Cost scale: opus ≈ 74k tok / 23 min per task-trial → ~$2-5/task-trial estimated;
  a 25-task × 4-solos + 2-comps tier ≈ $150-400. NOT inside the $40-60 MISSION budget.

SEQUENCING (recommended): run Tier 1 first (cheap, existing infra); its three-way
verdict picks WHICH composition class (if any) earns the Tier-2 spend. Tier-2 solos
double as the endgame transfer baselines regardless of the Tier-1 verdict, so that
spend is never wasted — but it is a separate user GO.

## Abort conditions (pre-registered)

- Estimated+reported spend hits budget → stop, analyze partial (resume possible).
- >20% container/harness errors in smoke → fix harness before spending more.
- Opus solo ≥0.9 on smoke (full strength saturates the band) → stop and re-curate
  harder tasks before the matrix; a saturated band cannot show Δ either way.
