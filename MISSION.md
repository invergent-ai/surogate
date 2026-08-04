# Mission: Fugu-Ultra — Pool-D GRPO Campaign (reproduction guide)

Full program history, incident narratives, and superseded results live in
`MISSION_ARCHIVE.md` (and git history). This file holds ONLY what is needed to
reproduce, monitor, and evaluate the training run, plus the standing rules.

## Objective

Ship a model-agnostic conductor that orchestrates **open-weight models** into a
single API endpoint at frontier-level performance (bar: Opus-4.8 / GPT-5.6-Terra
class; audience: office/personal agents with strong terminal capability).
Current goal: train the Fugu-Ultra conductor with a **200-step GRPO run on a
fully-local worker pool at $0/rollout** (8x RTX 5090). The local pool is a
TRAINING pool optimized for balance/selection-headroom; production models are
bound later via a short binding recal + finetune.

## Hard Rules (standing, user-set)

- Update MISSION.md with every result/failure THE MOMENT it lands. Scope:
  progress + recipe only — bug fixes enter as the surviving recipe/config
  fact, never as failure narratives.
- NEVER classify prompts/tasks by words/strings/substrings (any language).
  All task understanding goes through the conductor model.
- Workers run at `high`/default reasoning effort — xhigh/max measured WORSE
  twice.
- Goal-first discipline before any action; never launch paid work without
  verifying config/sequencing. Published solo numbers are never re-measured
  with paid calls.
- Kill processes by exact PID only — no age/pattern kill heuristics.
- Parallelize every fan-out from t=0.
- Simple answers; decide, don't flag.

## Current Status (2026-08-03)

**Definitive from-0 run in progress** (launched 2026-08-02 19:03): 200 steps
from the r4 merged base under the first fully-correct settings. Step ~35/200.
Engine on main 3c20f4e9 since ckpt 25 (FP8 resident masters, −14% s/ktok).
Repo lane LIVE since batch ~35 (Harbor env fix, launch step 4; before that it
produced zero training signal — quarantined silently). Evals every 10 steps,
floors perfect throughout (40/40 parse, 0 false completes, all seven evals);
composites vs parent 0.685: s30 0.734, s40 0.738, s50 0.706, s60 0.712,
**s70 0.7456 (best)**. s70 lanes: single 0.85 + tool 0.87 (highs), telecom
0.70 (REBOUND — the s60 0.53 was the noise floor; patience protocol 2-for-2
after repo), repo 0.425 (ties best, now with trained behavior; arms series
0.30 parent / 0.425 / 0.3625 / 0.25 / 0.4125 / 0.425), crm 0.61 (slow drift
0.74→0.65→0.61, still +0.11 over parent — current watch lane, same
protocol). Shape: 4.11 steps/plan, 3.78 edges (denser trees), diverse 11 /
dup 6, group_std 0.2313, zero-var 0, adaptivity corr 0.313 / delta 0.762.
Engine: vtc=0 ROOT CAUSE SOLVED (2026-08-04, issue #74): forward_for_grpo
zeroes loss buffers per INVOCATION; chunked GRPO invokes per CHUNK with
phase B in REVERSE order → step-end vtc = last micro's CHUNK-0 count. A
>1536-token-prompt sample packed last ⇒ chunk 0 all-masked ⇒ vtc=0 (verified:
forensic last micros [0,125]/[0,167] vs healthy single-chunk 136==logged).
Incidence ≈ once/50-100 steps; guard #72 fired at steps 73/82/85 (zero
policy damage). MITIGATED in-campaign (2026-08-04): trainer.py packing-order
guard swaps the max-chunk0-valid micro into the last slot (gradient-neutral;
restart at ckpt 85, upgraded at ckpt 90). ALSO: the non-zero form distorted
the grad-norm instrument all campaign — corr(grad_norm, 1/vtc)=0.83 over 76
steps; the historical 'spikes' (25/34/39/62/86) were denominator artifacts,
and the 0.05 clip SUPPRESSED real updates on low-vtc steps. With the guard,
norms are honest and the 0.05 rule stands on real footing. Engine fix
(step-accumulated counter) = post-campaign upstream work (issue #74).
Step-40 shape battery best-of-campaign: dup-multistep 9→4, diverse 13,
zero-var 0, adaptivity corr 0.18, single 0.797 + tool 0.84 highs (AQN
triggers armed, group_std 0.245 healthy). Docker/Harbor infra hardened
2026-08-03 (launch steps 4/6/7): repo lane dead before then; network pool
expanded to ~270 for 64-rollout group bursts. Prior eras archived under
`output/fugu_ultra_pool_d/cap3_era/` (steps 0-71, max_workflow_steps=3 bug)
and `rewind_era/` (seeded steps 2-17, settings churn) — do not train on them.

## Serving Stack (Pool D)

Launch: `scripts/serve_pool_d.sh`; watchdog: `scripts/pool_watchdog.sh`
(liveness = engine PROGRESS via `vllm:generation_tokens_total`, not latency;
TCP-refused = dead → immediate restart; probe timeout 150s x3 strikes).

| Role | Model (served name) | GPU | Port |
|---|---|---|---|
| knowledge/tool anchor | `google/gemma-4-26b-a4b-it` (unsloth NVFP4) | 0 | 8015 |
| reasoner | `qwen/qwen3.6-35b-a3b` (nvidia NVFP4) | 1 | 8016 |
| builder | `deepreinforce/ornith-1.0-35b` (NVFP4) | 3 | 8018 |
| leaf generalist | `zai-org/glm-4.7-flash` (NVFP4, **no fp8 KV** — shared-mem limit) | 5 | 8020 |
| conductor (policy) | `fugu_27b_r4_merged_bf16` + LoRA `fugu-pool-d-policy` | 6-7 (TP2) | 8011 |
| trainer | — | 2 | — |
| spare / replica | — | 4 | — |

Worker vLLM settings: max-model-len 32768, max-num-seqs 96, fp8 KV (except
GLM), prefix caching, `--language-model-only`, enable_thinking=false
server-side on thinking models.

Conductor MUST be served by **`surogate grpo-infer infer_pool_d.yaml`** (stock
vLLM lacks the `/update_weights` + `/load_lora_adapter` admin routes the
broadcast reload needs). `--served-model-name` carries BOTH ids (merged path +
adapter name); removing either causes a silent 404 loop. Key infer settings
(`director/manifests/fugu_clean_v1/grpo_pilot_train/infer_pool_d.yaml`):
TP2, max_model_len 8192, **max_num_seqs 16** (the knob that makes 27B+LoRA
fit — NOT gpu_memory_utilization), fp8 KV (+28% busy throughput), enable_lora,
max_lora_rank 16.

Health gate before any launch: `ultra/pool_health.py --orch <orch.yaml>`
(verifies the conductor serves the orch config's exact `model.name`).

## Data Mix

Ten lanes, `buffer.env_ratios` (sums 1.0), from the completed-probe export
(2026-07-30, 382 retained single-turn tasks):

| Lane | Ratio | Manifest (director/manifests/fugu_clean_v1/grpo_pilot_train/) |
|---|---|---|
| math | 0.105 | pool_d_math_taskspecs.jsonl |
| code | 0.1407 | pool_d_code_taskspecs.jsonl |
| rlpr | 0.1299 | pool_d_rlpr_taskspecs.jsonl |
| bird (SQL) | 0.0961 | pool_d_bird_taskspecs.jsonl |
| finance | 0.1068 | pool_d_finance_taskspecs.jsonl |
| dabstep | 0.1015 | pool_d_dabstep_taskspecs.jsonl |
| tool_dialogue | 0.08 | tau_retail_train_full_taskspecs.jsonl |
| repo_terminal | 0.07 | (env default manifest, envlanes config) |
| office_telecom | 0.09 | tau2_telecom_taskspecs.jsonl (sealed 114-task benchmark EXCLUDED) |
| office_crm | 0.08 | crmarena_train_taskspecs.jsonl (sealed 20% holdout in separate file) |

Single-turn six run `max_turns: 2` (plan + one feedback retry); the four
agentic lanes run the env default `max_turns: 1`. Buffer: easy_threshold 1.0,
hard_threshold 0.5, recycle_easy/hard 0.15, normal_pool_min_examples 16,
online_difficulty_filtering true.

**Held-out / sealed sets** (never train, never mid-run-probe the sealed ones):
`heldout_eval_taskspecs.jsonl` (120: 60 math + 60 unit_code — probe set),
`tau2_telecom_sealed_eval_ids.json`, `crmarena_sealed_holdout_taskspecs.jsonl`,
`heldout_confirmation(_v2)/trend60/fshard` (promotion evidence).

## Trainer Recipe (`train_pool_d.yaml` — authoritative; diff against
GRPOTrainConfig defaults before any launch)

| Setting | Value | Why (surviving fact) |
|---|---|---|
| model | `output/fugu_27b_r4_merged_bf16` | import the merged serving dir directly (787/787 tensors; in-stream adapter merge under GRPO config crashes) |
| recipe | fp8-hybrid | bf16 import faults on this hybrid stack at 6 GPUs; fp8-hybrid is the r4-proven path. Consequence: trainer-vs-serving numerics band kl≈0.24 at identical weights — that is BASELINE, not drift |
| cpu_training | true | 5090s cannot hold the 27B resident; streams ~27 TB/step (why steps cost 20-60 min) |
| gpus / CUDA_VISIBLE_DEVICES | 1 / `2` | multi-GPU import crashes (world-size-dependent path); GPU 2 is the trainer's |
| sequence_len / sequence_chunks | 6144 / 4 | worst prompt 4024 + 2048 cap; chunk COUNT sets streaming cost |
| lmhead_chunks | 12 | memory only (drop both chunk knobs to 1 on a large resident GPU) |
| single_sample_bins | true | chunked attention has no packed-doc isolation |
| learning_rate | **1e-5 constant** | 2e-5 produced escalating grad spikes (0.043→0.164) at BOTH lr eras' spike batches; identical batches at 1e-5: 9x smaller norms. 10x-LoRA rule sets the scale |
| max_grad_norm | **0.05** | at 1.0 the clip never engaged; occasional strong-gradient batches (topology re-learning) are bounded at the guard threshold; average case 0.005-0.03 untouched. This clip capped the vtc=0 corruption event 37x |
| loss | adv_tau 1.0, teacher_tau 0, **kl_tau 0**, ipo_mask 0.2/0.2, **ratio_clip 7.389056 (e²)** | Conductor recipe: no KL; ratio_clip mandatory under periodic merged reloads (unclamped rare-token ratios 1e3-1e5 pass the IPO mask) |
| lora | r16, alpha 32, all 7 proj modules | |
| master/gradient dtype | bf16 / (GRPO default fp32) | LoRA precision comes from lora_dtype fp32 |
| max_async_level | 1 | |
| save_steps / checkpoint_dir | 5 / == output_dir | find_latest_checkpoint alignment. NOTE: the scanner ignores step 0 — a seed staged as step_00000000 silently starts a FRESH adapter; stage seeds as step_00000001 |
| template | qwen3_nothinking | serving samples with enable_thinking=false; training must render the SAME template or exact-token prompt ids mismatch |
| max_steps | 200 | must equal batches actually sent |
| env | `SUROGATE_DISPATCH_PREFETCH_BLOCKS=3 SUROGATE_RESIDENT_LAYERS=20` | engine >= 3c20f4e9: first 20 layers' FP8 masters resident on device (-6.0% s/mb measured on this exact replay; bit-inert; OOM at 24 on 32 GB). VERIFY the `[resident] promoted N FP8 block masters` log line — the mechanism fails SILENTLY without it |

**vtc==0 guard** (in `surogate/grpo/trainer.py`, upstream PR #72): if
ValidTokenCount is 0 with micro-batches processed, the optimizer runs at
lr=0/wd=0 to flush the (corrupt, ~1e5x-scaled) accumulation without moving the
policy. vtc=0 is a first-class alarm.

## Orchestrator Recipe (`orch_pool_d.yaml`)

- **Group config (paper A.1, verified realized in rollouts.bin):**
  `batch_size: 256`, `rollouts_per_example: 64` → exactly 4 questions x G=64
  per step; multi-turn lanes append turn-2 rows beyond 256.
- **Advantage:** `surogate.grpo.orchestrator.advantage.std_normalized_advantage`
  (eps 1e-4) — (r-mean)/std per group, Conductor eq. 2.
- **Conductor sampling:** temperature 1.0, max_tokens 1536,
  `extra_body: chat_template_kwargs: {enable_thinking: false}` (omit this and
  the conductor burns the whole budget thinking, parse 0).
- **Workers:** temperature 0.2, max_tokens 16384 single-turn / 8192 agentic,
  reasoning effort high, budgets: short (single-turn) / long (agentic).
- **max_off_policy_steps: 4** (default 8 let long rollouts age 8 policies →
  30-54% IPO-masked batches; 4 discards instead. An orch restart resets the
  accumulated staleness level to 0 — cheap reset, proven).
- **max_workflow_steps: 5 in ALL THREE pilot configs AND safety files**
  (singleturn/office/envlanes). THE critical fact: an inherited 3 hard-zeroed
  every 4-5-step plan (`invalid_workflow_trainable`) while the contract prompt
  says "AT MOST 5" — it silently trained the conductor out of trees. Verify 5
  before any run.
- Reward = faithful Ultra reward − `_redundancy_penalty` (0.1 per
  near-duplicate subtask at token-Jaccard ≥0.75, cap 0.3): deliberately taxes
  copy-paste ensembles; diverse decompositions are penalty-free.
- ckpt: `{interval: 1, keep_last: 15, resume_step: -1, wait_for_weights_timeout: 5400}`.
  The weight-wait is SILENT (83 min observed, healthy) — a resumed orch waiting
  on `broadcasts/step_<resume>/STABLE` is not hung; check that file, not logs.
- client: conductor `http://localhost:8011/v1`, timeout 1200.
- `trainable_metric: ultra_valid_for_training`; `dump_metrics: false`.

## Launch Procedure (from-0)

1. Serve pool (`serve_pool_d.sh`), start watchdog, run
   `pool_health.py --orch orch_pool_d.yaml`.
2. Cold start the serving policy with the **zero-B adapter**
   (`output/fugu_ultra_pool_d/eval_frozen/zero_adapter` — all lora_B zeroed,
   mean|B|=0.0 = base-identical): load as `fugu-pool-d-policy` via
   `/v1/load_lora_adapter`. Never launch with the adapter name unregistered
   (404 loop) or stale (wrong policy for batch 0).
3. Trainer: `CUDA_VISIBLE_DEVICES=2 setsid nohup env PYTHONPATH=$REPO:$REPO/ultra
   SUROGATE_DISPATCH_PREFETCH_BLOCKS=3 SUROGATE_RESIDENT_LAYERS=20
   .venv/bin/python -m surogate.cli.grpo_train
   director/manifests/fugu_clean_v1/grpo_pilot_train/train_pool_d.yaml >> output/fugu_ultra_pool_d/trainer.log 2>&1 &`
   (engine ≥ 3c20f4e9; VERIFY the `[resident] promoted N FP8 block masters` log line.)
4. Orch: same pattern, `-m surogate.cli.grpo_orch ... orch_pool_d.yaml
   >> output/fugu_ultra_pool_d/orch.log`, with the repo-lane env REQUIRED:
   `ULTRA_HARBOR_BIN=/home/densemax/.harbor-venv/bin/harbor OPENAI_API_KEY=EMPTY`.
   The terminal_sandbox harness shells out to the Harbor CLI (isolated venv:
   `python3 -m venv ~/.harbor-venv && ~/.harbor-venv/bin/pip install harbor`;
   0.20.0 verified — flags + `verifier_result.rewards` schema match, accepts
   agent `terminus-2`); litellm inside Harbor needs OPENAI_API_KEY even for
   local vLLM. WITHOUT these every repo rollout quarantines SILENTLY
   (`valid_for_training=False`, no log line) and the lane trains nothing.
5. Verify: trainer step 0 must show kl ≈ 0.24 (the fp8-vs-bf16 baseline band);
   materially higher = policy mismatch, STOP.
6. Lane liveness gate (launch + after any venv rebuild): for EACH lane run one
   episode end-to-end and require a graded `valid_for_training=True` record
   with non-empty artifacts. Uniform 0.0 with sub-second episodes = infra-dead
   lane, not hard tasks (repo probe check: fresh Harbor job dirs must appear
   under `.ultra_harbor_runs/`).
7. Harbor compose GC: run `scripts/harbor_sweeper.sh` (setsid nohup, logs to
   `output/fugu_ultra_pool_d/harbor_sweeper.log`). Harbor leaves each
   rollout's compose project (container + docker network) running; the
   default ~31-network pool exhausts after ~30 rollouts and ALL repo episodes
   then fail fast with "no Harbor verifier rewards found" — same instant-zero
   signature as a missing CLI. `HarborHarness.close()` handles the harbor
   ≥0.20 `<trial>__env` project naming, per-step teardown in `run()` bounds
   64-rollout groups, and /etc/docker/daemon.json widens the address pool to
   ~270 (`default-address-pools` incl. 10.201.0.0/16 size 24; daemon restart
   required). The sweeper also GCs per-trial compose-BUILD images
   (`inferredbugs-*__<trial>__env-main`, one 500MB tag per rollout-step) and
   LRU-evicts unused multi-GB swe-bench BASE images below 80GB free (re-pull
   on demand). Health: networks < 70 during a repo group, disk ≥ 80GB.

**Seeded restart variant:** copy a native checkpoint to
`checkpoint_dir/step_00000001` (adapter + optimizer moments restore;
batches 0-1 sacrificial). `adapter_init_mode: trainable` is NOT supported by
the GRPO native trainer.

**Ops rules:** SIGTERM first, by exact PID; orch shutdown hangs — with the
conductor at 0 running requests, force-kill after ~15 min (checkpoint interval
1 makes it safe). Never suppress stderr on state-mutating commands; verify
cache purges by listing. Purge `.ultra_cache/*workflow_records*` after any
scoring-rule change (records carry valid_for_training=true and replay stale
rewards; worker completions cache is safe to keep).

## Evaluation (every 10th trainer step, automatic)

`PYTHONPATH=ultra .venv/bin/python -m ultra.conductor_checkpoint_eval --step N`
— freezes `broadcasts/step_N` (md5-verified) to `eval_frozen/stepN`, registers
it as a DISTINCT vLLM id (never eval the live policy name; unloads stale
same-name ids first), then reports into `eval_frozen/shape_ledger.jsonl`:

- 40-case typed gate (`scratchpad/gate_fugu_ornith_typed_base_v1.py`) —
  floors only; aggregate counts saturated at step 0 (40/40 parse/legal).
  ALWAYS diff per-case raw generations before calling "no change".
- Topology shape: steps/plan, independent-fraction, access-edges, size
  adaptivity (corr + small/large delta).
- Plan diversity: diverse-multistep vs duplicate-heavy (redundancy-penalty
  escape watch).
- `reward_diversity()`: within-group reward std (exploration health;
  AQN/QeRL noise stays OFF unless group_std <0.15 sustained, zero-var groups
  climbing, or reward slope negative — implementation exists end-to-end,
  sigma_start must be 1e-2 not the 5e-2 default).
- `anchored_reward()`: per-task repeats (composition-drift-proof reward trend).
- **Real-task probe** (`ultra/heldout_probe.py`): 68 tasks/arm — heldout
  math/code 30 + telecom 10 + crm 10 + tool 10 + repo 8, campaign-verbatim
  lane settings, fixed seed 20260802, conductor t=0. `--lanes` runs a subset
  (paired via the fixed seed). Paired vs the parent baseline
  `eval_frozen/probe_parent_r4base.json` + `probe_parent_repo_harbor.json`:
  **parent composite 0.685** (single 0.77 / telecom 0.81 / crm 0.50 /
  tool 0.80 / repo **0.30**). The repo row REQUIRES the Harbor env (launch
  step 4); pre-2026-08-03 repo zeros in any artifact are infra-dead values,
  not measurements (probe runs with empty artifacts = the signature).
  Absolute numbers are NOT leaderboard-comparable (training pool).
  Step-30 corrected composite **0.734** (repo 0.425, 2/8 solved,
  `probe_step30_repo_harbor.json`) — ≥ parent on 8/8 paired repo tasks with
  zero repo training signal (transfer from the other lanes).
  Step-40 ledger repo row (0.0625) is POISONED — docker network exhaustion
  (launch step 7), not model behavior; the valid number is
  `probe_step40_repo_harbor.json`: repo 0.3625 (1/8 solved), corrected
  composite **0.738** (best yet; step-30 0.734, parent 0.685). Repo arms
  trajectory 0.300 → 0.425 → 0.3625 = noise band at n=8 (±0.5 single-task
  swings); first checkpoint with real trained-repo exposure is step 50.

## Paper Fidelity (Fugu technical report, full read 2026-08-03)

Verified matches to §3.2 / §4.4 (keep these): <=5-step workflows; GRPO with
**no KL penalty**; MC group advantage (r-mean)/std; access-list-only visibility
between workers — the report's stated reason is preventing **orchestration
collapse**, where the first agent to touch the environment steers all later
agents into redundant contributions (our `access_positions` is exactly this
mechanism, and it is why isolated no-access positions matter); persistent
inter-workflow shared memory alongside intra-workflow isolation (our
`shared_memory` state field); learned topologies averaging ~3 steps.

Deliberate deviations (recorded, do not "fix" without a decision):
1. **Reward shape.** Paper: 0 unparseable / 0.5 well-formed-but-wrong / 1
   correct. Ours: graded Ultra reward minus `_redundancy_penalty`. Ours gives
   denser signal but taxes duplicate-subtask ensembles the paper never
   penalizes.
2. **Worker identity.** Paper routes integer `model_id` over 3 NAMED frontier
   models and demonstrably learns brand-level priors (GPT for math, Opus for
   debugging, Gemini for niche recall; Fig. 5 routing distributions). Ours uses
   anonymous permutation-invariant `profile_ref` capability tags — same
   information, no name dependence, per the no-keyword hard rule.
3. **Conductor turns.** Paper: workflow state tracked across a whole multi-turn
   conversation with unlimited worker-environment interaction. Ours: `max_turns`
   1 (agentic lanes) / 2 (curated single-turn), worker tool loops private and
   preserved via `continue`.
4. **Two-tier product.** Paper ships FUGU (one worker per input, latency/cost
   tier — still beats frontier on TB2.1 80.2 vs 74.6 and LCB 92.9 vs 87.8) plus
   FUGU-ULTRA (multi-agent composition for hardest tasks). Our conductor is the
   ULTRA analogue; a cheap single-worker tier is a separate product decision,
   and is the answer to per-query cost in production.
5. Paper allows the orchestrator itself as a worker agent; we do not.

## Promotion Protocol (unchanged)

B-norm audit every adapter (mean|B| growth = real training) before gating.
Typed-contract retention floors vs the accepted product, then the fresh
parent/candidate whole-task probe (48 scenarios x 19 samples, identical
seed/temp/concurrency, 912 zero-paid calls/arm; no-regression required;
probes used for curriculum become development evidence). Sealed evaluation:
23-task validation subset selects ONE checkpoint; single run on the sealed
final subset, zero retries; matched per-task comparison vs each bound solo
worker; report as ALE-derived, never official ALE-V1. Until then:
superiority unproven.

## Engine / Infra Facts

- Engine: repo main ≥ c6a099f9 (2.1x chunked-GRPO streaming; measured 1.38x
  wall in production + clip/prefetch facts above). PRs upstreamed from this
  campaign: #70 (resume, stall visibility, max_num_seqs/kv_cache_dtype),
  #71 (chunked GRPO engine, ratio_clip, adapter init, buffer recycling),
  #72 (vtc==0 guard).
- Rebuild: `make build` copies .so into the live package — only with the
  trainer STOPPED. Makefile resolves CUDA_HOME through version symlinks.
- Published trainer images: pin by digest (`@sha256:...`) — Modal (and any
  cache) serves stale mutable tags. 1.4.6+ carries the full recipe (ratio_clip
  + chunked); with it, NO engine mount/overlay is needed (the overlay itself
  caused the import hang). Modal H200 path works (1233s/step measured) but is
  RETIRED on cost (~$400/campaign vs $0 local).
- qwen3.6 local serving: use `output/qwen36_ct_patched` (RedHatAI
  compressed-tensors blobs + collapsed in_proj regex) — the modelopt NVFP4
  variant is unstable.
