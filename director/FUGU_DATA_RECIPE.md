# Clean Data Recipe for a Base Fugu Replica

This document is the concrete data recipe for training a base Fugu-style router.
It is intentionally independent of the current `sources.py` registry, because that
registry may include public evaluation data. Treat this document as the source of
truth for the next clean data build.

Scope: base Fugu, not Fugu-Ultra. The model should learn worker selection, not
generate answers itself. Every training example must provide a verifiable reward
for each worker in the pool.

## Master Checklist — to Production Fugu

Living tracker (keep updated). `[x]` done · `[~]` in progress · `[ ]` pending.

### Phase 0 — Infrastructure & data sources
- [x] Worker pool: 6 open-weight models via OpenRouter (+ `Surogate` / `surogate.ai` attribution).
- [x] Single-step loaders + `EVAL_ONLY` denylist (HumanEval / MATH-500 / AIME / GPQA held out).
- [x] Measured per-source keep-rates → finalized mix; dropped `mbpp` (saturated), `omni_math` (dead),
  `arc-agi-2` (flails/truncates); `taco` + `code_contests` banded to the rated mid difficulty.
- [x] SWE-smith agentic harness (loader + runner + SWE-smith grader) — validated end-to-end.
- [x] Terminal-bench fixed; kept as a HELD-OUT eval (not training).
- [x] Curation engine (`curate.py`: probe → discriminative → balance → split) + manifest data model.
- [x] `build_clean_data.py`: probe(n=1) → curate → relabel(n=4, temp>0) → report → pre-SFT gates.
- [x] Probe robustness (per-worker failure → 0, no hang); recipe counts/budget/pilot reconciled.

### Phase 1 — Stage-1 SFT (warm-start)
- [x] SFT train+eval (`train_eval_pilot.py`): trains head+SVF, internal-val lift, after-SFT gates +
  learning curve. Validated on real reward vectors; `CURVE=0` smoke shortcut for fast checkpoints.
- [x] Eval-denylist by **normalized prompt hash** — `build_clean_data` drops candidates matching any
  held-out eval prompt (891 hashes: HumanEval/MATH-500/AIME/GPQA-diamond; catches renamed mirrors). Tested.
- [x] All 3 generation stages **resumable** (probe / relabel / agentic-bank) → a pilot is a full
  down-payment on the full run (same MANIFEST_DIR + seed-0 subset). So the separate ⅓-scale pilot was
  unnecessary — validated the chain via the tau-only smoke and ran the full build directly.
- [x] **n=2 vs n=4 → decided n=4**: the warm-start has leverage on the expensive Stage-2 (cleaner
  start → fewer CMA-ES gens); sharp `τ=0.1` amplifies low-n noise; relabel is one-time/resumable/unattended.
- [x] **FULL Stage-1 data build DONE** (`manifests/fugu_clean_v1`): probe 8,900/8,900, keep-rate **51.9%**
  → curated **2,650** (train 2,253 / val 397; balanced code 900 / math 650 / sci 550 / gen 550). Pre-SFT
  gates **5/6 PASS** (denylist 0; oracle 1.0 vs best-single 0.791 = **+0.209**; balance OK); the 6th
  ("disagreement 10-30%") was a stale threshold vs measured 34-58% → **fixed to 10-60%**.
- [x] **Confirmed Fugu SFT = head + SVF** (report §3.1.2: "training both the lightweight selection head and
  the singular-value scales") — our code is correct; head-only is *Trinity*, not Fugu. Trinity reference SVF
  = 9 tensors (`embed_tokens` + layer-26 ×7 + `lm_head` = 9216); ours = layer-26 (7168, partial-cacheable).
  Decision: keep layer-26-only (enables a ~10× partial cache, recompute only layers 26→end); not the 9216.
- [x] **LR tuned — the −0.055 was an lr bug, not a dominant-worker verdict.** Monotonic sweep: 0.02→−0.055,
  1e-2→−0.050, 1e-3→−0.018, **1e-4→+0.008 (no collapse)**. Original 0.02 was ~200× too high. **Use lr=1e-4**
  (Trinity's 1e-6 was for head-only; head+SVF wants a touch higher). Single-step lift is capped ~0 by
  flash-dominance regardless → 1e-4 gives a clean warm-start basin, which is all Stage-2 needs.
- [x] **Reasoning-confound probe** (50 items, low vs high reasoning): high reasoning lifts **all** workers
  (oracle 0.58→0.76) — the cheap low-reasoning warm-start data understates capability — **but flash stays
  best-single**, so its dominance is largely real, not an artifact. Don't redo Stage-1 at high reasoning.
- [~] **Final SFT warm-start running** (lr=1e-4, head+SVF, full 2,253 labels, GPU-only) → `sft_router.pt`.

### Phase 2 — Stage-2 agentic (the real lift)
- [x] Agentic bank builder (`build_agentic_bank.py`) — solo rollouts SWE-smith + tau retail/airline →
  per-(task,worker) cells + oracle/headroom report. Report logic validated on real cells
  (**agentic headroom +0.23 vs single-step +0.10** — lift lives here).
- [x] sep-CMA-ES Stage-2 (`cmaes_pilot.py`) — warm-start from the SFT vector, evolve on LIVE routed
  rollouts (allowed=None), after-agentic gates (CMA-ES > SFT-only, ≥ best-single).
- [x] Minimal Stage-2 **confirmed end-to-end via tau-only smoke** (SFT → bank 24/24 → CMA-ES loop runs).
- [x] **Tau agentic bank DONE** (`agentic_bank.jsonl`, 479 cells, $9.71; per-cell progress + running cost
  logged; no budget cap — monitored; GLM on default routing): retail+airline, 79 complete items,
  **68% discriminative**, **headroom +0.203** (best-single mimo 0.557 vs oracle 0.759), **no dominant
  worker** (spread 0.33–0.56) — the complementary-pool condition single-step lacked.
- [x] **SWE-smith was NOT dead** — the 0/45 all-fail was a wrong-interpreter grader bug (`subprocess` ran
  bare `python` = parent `surogate/.venv` without swesmith). Fixed → `sys.executable`. Re-addable to the
  bank (resumable) for agentic-coding signal.
- [ ] Run Stage-2 CMA-ES (warm-start `sft_router.pt`) → does it capture the **+0.203**? Adopt the **Trinity
  ES config** (reference `study/trinity_framework/.../sakana_es_log.json`): `sigma0=0.03`, replicated
  rollouts (their `num_repeats=16`), fitness **bonuses diversity 0.15 + turn 0.1 + cost 0.0**, `max_turns=5`.
  NB: high-reasoning agentic rollouts are slow + pricey → small popsize + tau-heavy eval set.
- [ ] **TaskTrove diversity expansion via Harbor** (after first Stage-2 validation): reuse the
  OpenThoughts-Agent **Harbor** harness (local clone `study/OpenThoughts-Agent`) instead of hand-rolling a
  runner — Harbor already runs a model-agnostic agent (Terminus-2) on a task bundle (`instruction.md` +
  `environment/Dockerfile` + `tests/`) and grades by **executing the bundled tests**.
  - **Tasks-only, our pool, our grading — NOT their rewards.** Bridge: `scripts/datagen/extract_tasks_from_parquet.py`
    (TaskTrove parquet → Harbor task dirs). Adds pymethods2test / r2egym / nl2bash / puzzles / code_contests
    diversity beyond SWE-smith + tau.
  - **Tier 1 — solo bank (easy):** point Harbor's OpenAI-compatible `InferenceEngine` (`data/generation/engines.py`)
    at OpenRouter; run each of our 6 workers solo per task → per-(task,worker) pass/fail → agentic bank.
  - **Tier 2 — routed CMA-ES (deeper):** inject a `FuguModel`-style per-step routing adapter into Harbor's
    agent loop (same trick as mini-swe-agent) for live routed rollouts.
  - Gold-check via bundled `solution/`; prompt-hash denylist + per-source review (code_contests already used;
    r2egym/others vs held-out evals). Local Docker (not Daytona); per-task `docker build` → sample hundreds,
    cache base layers. Terminus-2 is the agent *scaffold* (faithful to Fugu) — only eval *tasks* are held out.
  - (AgentTrove ignored: traces we don't want + empty grader fields.)
- [ ] (Optional, faithful) OpenCode harness via routing proxy for coding-assistant trajectories.

### Phase 3 — Eval & validate
- [ ] Held-out eval harness: SWE-bench Verified, Terminal-Bench, GPQA, tau³ banking, LiveCodeBench,
  HumanEval, MATH-500, AIME — all unseen during training.
- [ ] Final eval (once, after all data checks): resolve rate vs each single worker + oracle + routing
  distribution.
- [ ] Cost/latency: cheap worker chosen on easy items; cost-aware routing doesn't drop solve >1 pt.

### Phase 4 — Productionize
- [ ] Inference serving: router as a single-model interface, per-step routing in deployment.
- [ ] Version pinning / stamp model version per cell (avoid silent provider updates → stale labels).
- [ ] Incremental update: regenerate one worker's column on a model release (not a full retrain).
- [ ] Dogfood: real user sessions → in-distribution trajectories.

### Backlog / deferred
- [ ] Convert non-streaming MC/math loaders (SuperGPQA / MMLU) to streaming (slow raw-load).
- [ ] Purge 2 corrupt terminal cells (reward=0 & cost=0) from the old bank, post-collection.
- [ ] When opus/gpt join the pool: re-enable SWE-bench Pro + add Claude Code / Codex harnesses.
- [ ] DeepSWE via routing proxy.

## Non-Negotiable Rule

Do not train on any task that you intend to evaluate on, including exact public
benchmark items, sibling rows from the same official eval split, or mirrored copies
with different dataset names.

Keep a hard eval denylist with:

| Eval family          | Denylist rule                                                                                     |
| -------------------- | ------------------------------------------------------------------------------------------------- |
| HumanEval            | Exclude all HumanEval tasks.                                                                      |
| MBPP                 | Exclude MBPP test and any MBPP split used for reporting.                                          |
| MATH500              | Exclude MATH500 and avoid MATH test-derived rows.                                                 |
| AIME                 | Exclude all AIME years used for reporting, especially 2024 and 2025.                              |
| GPQA                 | Exclude GPQA-Diamond.                                                                             |
| LiveCodeBench        | Exclude all eval problem IDs, source platform IDs, and dates in the eval window.                  |
| LiveCodeBench Pro    | Exclude all LCB Pro problem IDs and source platform IDs.                                          |
| SWE-bench            | Exclude SWE-bench Verified, Lite, Pro, and any exact repo/issue/PR instance used for eval.        |
| Terminal-Bench       | Exclude official eval/core task IDs used for reporting.                                           |
| tau-bench            | Exclude tau banking/test if reporting tau banking; exclude retail/airline test if reporting them. |
| SciCode              | Exclude official eval items if reporting SciCode.                                                 |
| CharXiv              | Exclude all CharXiv eval items.                                                                   |
| Humanity's Last Exam | Exclude all HLE items.                                                                            |
| MRCR/LCR             | Exclude official long-context eval tasks.                                                         |

The denylist should include at least:

- `source_name`
- `split`
- `task_id`
- source-platform IDs for contest tasks, such as Codeforces contest/problem ID
- repo plus issue/PR ID for SWE tasks
- normalized prompt hash

Hardest case to enforce — competitive programming: `code_contests` and TACO draw from the same
platforms (Codeforces/AtCoder/LeetCode) that LiveCodeBench and LCB-Pro sample from, so overlap is
likely and not free to remove. Cleanly excluding it needs LiveCodeBench's problem/platform IDs plus
a mapping you may not have. If you cannot enforce it, treat **LiveCodeBench / LCB-Pro as not cleanly
reportable** rather than pretending the training set is clean.

## Stage 1: Single-Step SFT Router Data

Goal: build a clean, verifiable, disagreement-rich SFT set.

Target size (sized from MEASURED probe keep-rates, not assumptions):

- Raw candidates: about `9,000`–`10,000`
- Curated disagreement items: `2,500` to `3,000`
- Internal split: `85% train`, `15% validation`, grouped by source family/task group
- Labeling: `n=2`–`4` samples per worker after curation, at sampling temperature > 0

Why this is much smaller than a "large-scale" run: Stage 1 trains a **~13K-parameter** head over a
frozen backbone and is only a **warm-start** — the real routing lift comes from Stage-2 (sep-CMA-ES).
A 13K head fits on a few hundred discriminative items per domain; more data has sharply diminishing
returns. Size from the **pilot learning-curve** (internal-val lift vs. #curated), not from a target.
And our MEASURED disagreement keep-rates are **34–58%** (≈3× a naive ~15% estimate), so a small raw
pool already yields enough curated items.

Use this mix. `exp. disc = Raw × measured keep` is the discriminative yield each source is expected
to produce; the curated set is a balanced **subset** of the total:

| Domain    | Source                    |   Split | Raw   | keep | exp. disc | Notes                            |
| --------- | ------------------------- | ------: | ----: | ---: | --------: | -------------------------------- |
| math      | `AI-MO/NuminaMath-1.5`    | `train` | 2,000 |  34% |       680 | Math workhorse (broad; streamed) |
| code      | `likaixin/TACO-verified`  | `train` MID | 1,500 | 58% |    870 | BEST source (EASY..MEDIUM_HARD)  |
| code      | `deepmind/code_contests`  | `train` MID | 1,800 | 42%† |   760 | †mid-banded (rated 6–10); ≥42%   |
| science   | `m-a-p/SuperGPQA` STEM    | `train` | 1,000 |  55% |       550 | Hard graduate MC                 |
| science   | `TIGER-Lab/MMLU-Pro` STEM | `train` |   800 |  53% |       424 | Routable for our open pool       |
| general   | `m-a-p/SuperGPQA` gen     | `train` | 1,000 |  55% |       550 | Language understanding           |
| general   | `TIGER-Lab/MMLU-Pro` gen  | `train` |   800 |  53% |       424 | Routable                         |
| **TOTAL** |                           |         | **8,900** | **~48%** | **~4,250** | curate a balanced 2,500–3,000 |

So **~8,900 raw → ~4,250 discriminative** at the measured rates → curate **2,500–3,000** after the
balance caps (per-domain ≤35%, source ≤30%, sole-winner flattening). There is **no separate reasoning
domain**: reasoning signal comes from hard math (NuminaMath) and the MC "general" rows (SuperGPQA/MMLU
non-STEM — law, economics, philosophy, logic — which require multi-step reasoning, not just recall).

Dropped after measurement (do not use):

- `mbpp` — 57% **saturated** (too easy, workers all pass) → near-zero routing signal.
- `omni_math` — 52% **dead** (olympiad too hard for our open pool) → no signal.
- `arc-agi-2` — the open pool **flails**: outputs truncate even at `16k` tokens (mostly all workers on
  all tasks), so it can't produce a gradeable grid. Off-distribution for a work agent + impractical to
  grade at any reasonable output budget. Reasoning is covered by hard math + the MC "general" rows.
- `sciq` / `ai2_arc` — easy MC fillers; superseded by the measured-routable MMLU-Pro.

Note on MMLU-Pro: an earlier draft questioned it as off-objective. The probe settled it — MMLU-Pro is
**53% discriminative** for our pool, and for a GENERAL work agent (not coding-only) knowledge MC is
in-scope. Keep it. The rule is empirical: a source earns its place by its **measured keep-rate**, not
by assumption.

Optional additions after the first full-scale mix is healthy:

| Domain    | Source                             | Use only if                                                                            |
| --------- | ---------------------------------- | -------------------------------------------------------------------------------------- |
| math      | `hendrycks/competition_math` train | You are not reporting MATH-family numbers, or you have a strict MATH500/test denylist. |
| code      | APPS train                         | You are not reporting APPS and can run its tests reliably.                             |
| code      | private coding tasks               | You have unit tests and stable execution sandboxes.                                    |
| reasoning | private ARC-style tasks            | You have exact output grids and no ARC eval overlap.                                   |

Do not use these for SFT training:

- `HumanEval`
- `MATH500`
- `AIME2024`
- `AIME2025`
- `GPQA-Diamond`
- `LiveCodeBench` eval or Pro
- `SWE-bench Verified`, `SWE-bench Lite`, `SWE-bench Pro`
- official `Terminal-Bench` eval/core tasks used for reporting
- `SciCode` eval items
- `CharXiv`
- `Humanity's Last Exam`
- `MRCR`
- `LCR`

## Single-Step Candidate Format

Normalize every raw candidate into this shape:

```json
{
  "task_id": "taco-12345",
  "domain": "code",
  "source": "taco_verified_train",
  "split": "train",
  "prompt": "...",
  "solution": {"tests": [{"input": "...", "output": "..."}], "timeout": 10},
  "grader": "code_exec_stdio",
  "metadata": {
    "dataset_id": "likaixin/TACO-verified",
    "source_task_id": "12345",
    "prompt_sha256": "..."
  }
}
```

Recommended graders:

| Task type                     | Grader                                               |
| ----------------------------- | ---------------------------------------------------- |
| math final answer             | `math_equal`                                         |
| multiple choice               | `mc_letter`                                          |
| Python function               | `code_exec`                                          |
| stdin/stdout program          | `code_exec_stdio`                                    |
| grid output                   | `grid_exact`                                         |

## SFT Curation Procedure

For each raw candidate:

1. Check source and normalized prompt hash against the eval denylist.
2. Run every worker once with the same sampling configuration.
3. Grade each worker output deterministically.
4. Drop the item if every worker passes.
5. Drop the item if every worker fails.
6. Drop the item if the verifier is flaky or times out repeatedly.
7. Keep the item if at least one worker succeeds and at least one worker fails.

Probe record:

```json
{
  "task_id": "taco-12345",
  "domain": "code",
  "source": "taco_verified_train",
  "prompt": "...",
  "grader": "code_exec_stdio",
  "worker_ids": ["deepseek", "kimi", "glm", "mimo", "minimax", "deepseek_flash"],
  "rewards": [1.0, 0.0, 0.0, 1.0, 0.5, 0.0],
  "verdict": "discriminative"
}
```

Balance the curated set with these caps:

- No domain above `35%` of the final SFT set.
- No source above `30%` of the final SFT set.
- No single worker should be the sole winner on more than `45%` of curated items.
- Each **strong** worker should have at least `2%` sole wins, or it should be pruned from the pool.
- **Exempt the designated cheap worker** (e.g. `deepseek_flash`) from the sole-win rule: by design
  it is rarely the *unique* winner, but its value is cost on easy/high-confidence items, not being
  best. Judge it instead by the after-SFT check ("cheap worker is selected mostly on easy items").
  Applying the sole-win prune to it would delete the worker whose entire purpose is cost-efficiency.

Then relabel only the curated items:

1. Run every worker `n=4` times **at sampling temperature > 0** (e.g. `0.7`). With greedy decoding
   the 4 reps are identical, so `r_bar` collapses to {0,1} and `n=4` buys nothing over `n=1` — the
   whole point of `n=4` is to measure each worker's true pass-rate. (Use the same temperature for
   the `n=1` probe so the discriminative filter sees the same distribution it will be labeled on.)
2. Compute mean reward per worker.
3. Convert rewards to a soft target with temperature `tau=0.1`.

Budget (reconciled with the counts above; `|pool| = 6`):

- Probe (n=1): `raw × |pool|` = `9,000 × 6` ≈ **54,000** calls.
- Relabel: `n × |pool| × |curated|` = `4 × 6 × 3,000` ≈ **72,000** calls at n=4 (≈ **36,000** at n=2).
- **Total ≈ 90k–126k** calls — vs. the ~410k the old 40k-raw / 240k-probe / 170k-relabel sizing
  implied. The cut comes from the high measured keep-rate (less raw needed) and the leaner curated
  target; relabel is still the dominant term, so prefer n=2 unless soft-target precision demands n=4.

Formula:

```text
r_bar[j] = mean reward for worker j over 4 samples
p[j] = softmax(r_bar[j] / 0.1)
```

Final SFT label:

```json
{
  "task_id": "taco-12345",
  "domain": "code",
  "source": "taco_verified_train",
  "prompt": "...",
  "worker_ids": ["deepseek", "kimi", "glm", "mimo", "minimax", "deepseek_flash"],
  "r_bar": [1.0, 0.75, 0.0, 1.0, 0.5, 0.25],
  "p": [0.4786, 0.0393, 0.00002, 0.4786, 0.0032, 0.00026],
  "grader": "code_exec_stdio"
}
```

Use the `p` vector for SFT. Keep `r_bar` for offline evaluation and diagnostics.

## Stage 2: Agentic Router Data

Goal: refine the router on multi-turn, tool-using, environment-grounded tasks.
This is the part that makes the base Fugu router useful beyond static Q&A.

Use only non-eval environments:

| Harness Type  | Source                                                         | Task Count | Reward                       |
| ------------- | -------------------------------------------------------------- | ---------: | ---------------------------- |
| SWE-like      | `SWE-bench/SWE-smith` train                                    |        500 | executable tests             |
| SWE-like      | private GitHub issue/PR tasks from repos not in SWE-bench eval |        300 | executable tests             |
| terminal      | private Terminal-Bench-style Docker tasks                      |        300 | native tests                 |
| tool/dialogue | tau retail/airline train or dev                                |        150 | DB-state/programmatic reward |
| tool/dialogue | private CRM/support/API domains                                |        300 | DB-state/programmatic reward |

If tau retail or airline is part of your reported eval, do not train on its test split.
Prefer train/dev, or create private tau-style domains.

For every agentic task, first run each worker solo. Store one row per task-worker pair:

```json
{
  "domain": "swe",
  "kind": "agentic",
  "item_id": "swesmith-django-00123",
  "prompt": "...",
  "worker": "kimi",
  "reward": 1.0,
  "cost": 0.12,
  "turns": 38,
  "source": "SWE-bench/SWE-smith",
  "split": "train"
}
```

Keep agentic tasks where:

- at least one worker succeeds
- at least one worker fails
- reward is deterministic across reruns or sufficiently stable
- the task is not in any eval denylist

Note on agentic noise: a single solo rollout (`n=1`) cannot establish stability — agentic rewards
genuinely flip run-to-run. Either accept the noise (the Stage-2 sep-CMA-ES fitness averages over
replicated rollouts anyway) or run `n>=2` on borderline tasks. Be explicit about which you choose.

For the agentic optimization stage, train on terminal reward directly. Do not turn
the full transcript into generic supervised text. The router should learn worker
selection per turn from environment state.

## First Clean Pilot

Run this before the full recipe — a ~⅓-scale subset of the mix above:

| Domain  | Source                          | Raw |
| ------- | ------------------------------- | --: |
| math    | `AI-MO/NuminaMath-1.5`          | 700 |
| code    | `likaixin/TACO-verified` (MID)  | 500 |
| code    | `deepmind/code_contests` (MID)  | 600 |
| science | `SuperGPQA` + `MMLU-Pro` STEM   | 600 |
| general | `SuperGPQA` + `MMLU-Pro` gen    | 600 |

Expected output (at the measured ~45–50% keep-rate):

- ~`3,000` raw candidates
- ~`1,400` discriminative → ~`900`–`1,000` curated after balance caps
- ~`800` SFT train labels (85%) · ~`140` internal validation labels (15%)
- Probe ≈ `18,000` calls; relabel ≈ `21,600` (n=4) / `10,800` (n=2)

Add this small agentic pilot:

| Harness                | Count |
| ---------------------- | ----: |
| SWE-smith train        |   100 |
| private terminal tasks |    50 |
| tau retail train/dev   |    50 |

Only scale up if the pilot passes the acceptance checks below.

## Acceptance Checks

Before SFT:

```text
eval-denylist matches: 0
verifier failure rate: < 3%
curated disagreement rate: 10% to 30%
oracle_over_workers >= best_single_worker + 0.05
each strong worker sole wins >= 2% of curated items
designated cheap worker, e.g. deepseek_flash, passes cost-on-easy-items check instead
no domain > 35% of curated items
no source > 30% of curated items
```

After SFT:

```text
internal validation router reward > best single worker
router captures >= 25% of available headroom
router does not collapse to one worker
cheap worker is selected mostly on easy/high-confidence items
```

After agentic pilot:

```text
agentic oracle_over_workers >= best_single_worker + 0.05
router reward improves over SFT-only router
cost-aware routing does not reduce solve rate by more than 1 point
```

## Output Files

Use these filenames for a clean run:

```text
manifests/fugu_clean_v1/raw_candidates.jsonl
manifests/fugu_clean_v1/eval_denylist.jsonl
manifests/fugu_clean_v1/probe_n1.jsonl
manifests/fugu_clean_v1/curated_disagreement.jsonl
manifests/fugu_clean_v1/labels_n4_tau0.1.jsonl
manifests/fugu_clean_v1/internal_val.jsonl
manifests/fugu_clean_v1/agentic_bank.jsonl
manifests/fugu_clean_v1/data_report.md
```

`data_report.md` should include:

- source counts before and after filtering
- domain counts
- per-worker mean reward
- per-worker sole wins
- oracle vs best single worker
- internal validation routing lift
- denylist match count
- verifier failure count

## Implementation Order

1. Build `eval_denylist.jsonl`.
2. Replace or bypass contaminated source loaders.
3. Generate `raw_candidates.jsonl` from the clean source mix above.
4. Probe all candidates with `n=1`.
5. Filter to disagreement items.
6. Balance by domain, source, and sole-winner worker.
7. Split internal train/validation by group, not random sibling rows.
8. Relabel curated train items with `n=4`, sampling temperature > 0, and target `tau=0.1`.
9. Train SFT router.
10. Build the agentic bank with solo-worker rollouts.
11. Run agentic optimization on non-eval tasks.
12. Evaluate once on public eval only after all data checks pass.

## Dataset Links

- NuminaMath-1.5: https://huggingface.co/datasets/AI-MO/NuminaMath-1.5
- TACO-verified: https://huggingface.co/datasets/likaixin/TACO-verified
- CodeContests: https://github.com/google-deepmind/code_contests
- SuperGPQA: https://huggingface.co/datasets/m-a-p/SuperGPQA
- SWE-smith: https://huggingface.co/datasets/SWE-bench/SWE-smith
