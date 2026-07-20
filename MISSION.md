# Mission: Fugu-Ultra

Last updated: 2026-07-20 11:56 UTC.

## Product Objective

Ship a trained conductor for long-horizon agentic work that outperforms every
individual worker in its bound pool. It must coordinate multi-step tool use,
subagents, artifact creation, debugging, handoffs, recovery, and independent
final verification. One-shot routing to a nominally strongest model is not the
product.

Success requires verified superiority on TerminalBench 2.1 and credible
SWE-Pro-class evidence. Sol's reported TerminalBench 2.1 score is 88%, so strict
superiority requires at least 79 passes on the 89-task suite. Published
comparator results are reused; comparator models are not rerun.

No superiority claim is currently authorized.

## Model-Agnostic Hard Rule

The runtime and learned decision surface must remain replaceable-pool
architecture:

- The conductor reasons over anonymous capability profiles, roles, workflow
  topology, access edges, and live agentic state.
- Concrete provider model identities exist only in a versioned pool binding and
  provenance. They never appear in learned prompts or actions.
- There is no globally preferred, default, or fallback worker.
- Worker IDs have no meaning without the capability profiles supplied for that
  request.
- Training data must permute capability profiles across worker IDs so a fixed
  slot cannot acquire semantic meaning.
- Replacing any or all workers requires a new binding, capability calibration,
  and possibly bounded continued/fine-tuning. It must not require orchestrator
  code changes or runtime special cases.
- Checkpoints remain pool-specific and hash-bound. A checkpoint/pool mismatch
  fails closed.

This rule applies to initial planning, live control, function-call routing,
replacement workflows, aggregation, debugging, and recovery.

## Current Pool

The current product round uses this fixed pool:

| Anonymous worker | Bound runtime model | Capability prior |
|---:|---|---|
| 0 | `gpt-5.6-sol` | reasoner, verifier, debugger |
| 1 | `gemini-3.5-flash` | scientist, planner, aggregator |
| 2 | `gpt-5.6-terra` | mathematician, coder, reasoner |
| 3 | `grok-4.5` | drafter, implementer, fast pass |

Binding:
`director/manifests/fugu_clean_v1/grpo_pilot_train/current_pool_binding_v11.json`

- Binding SHA-256:
  `4c0bd95fa62189e3d3bf31a807bc600ec98c74d39444ce760d737f4b3ea5568a`
- Pool fingerprint:
  `6ace6d4fdc84b56f8991af8a2cbc42d125f26f8bd682c3119a657eb56c1925c0`

## Current Product

The current trained product is reproducible but not proven superior.

| Component | Authoritative artifact | SHA-256 |
|---|---|---|
| Product runtime | `director/director/agentic/fugu_ultra_terminal.py` | `8324eeeff127752e0ba31293e7d2fa72e0168afdaa1136e541bb443580058978` |
| Runtime revision | `20260720-r57-unified-conductor-candidate` | source above |
| Initial planner config | `output/fugu_ultra_planner_composite_v11_s20/adapter_config.json` | `57095afdb2d1c8f2c8a435afa6b8856f57a3bced5107e85b393fe4100a7575b3` |
| Initial planner weights | `output/fugu_ultra_planner_composite_v11_s20/adapter_model.safetensors` | `f794ac11c862293ac37fd8ab7c3ba56e2c8ccc39e380ec8efb0f110b50c92fe0` |
| Live-control config | `output/fugu_ultra_live_control_grpo_v16/final_adapter/adapter_config.json` | `a06c3b34e0076b6cf26bd36bd24e01e0010b3bb5432155a2747a473cd30ae366` |
| Live-control weights | `output/fugu_ultra_live_control_grpo_v16/final_adapter/adapter_model.safetensors` | `0a987412de57f3afe5d35be2a9ce4c3b03d4ffcd676d2605941ee71f000448b3` |

The runtime supports `continue`, `handoff`, `replan`, and `complete` at live
terminal boundaries. It preserves function-call ownership and each worker's
private tool loop. Within a workflow, a worker sees another worker's trajectory
only through the conductor's access list. Verified state from completed prior
workflows is available as persistent shared memory. Runtime r57 adds an opt-in
unified full-action controller that owns both initial/replacement topology and
live actions. The accepted product still uses the planner-v11/live-v16 split;
the unified path is a training candidate and is not promoted by source changes.

The accepted checkpoint's current defects are:

- Initial plans are executable but insufficiently adaptive; a nine-task local
  probe produced 9/9 executable identity-free workflows but used the same
  `[2,1,0]` topology on 8/9 tasks.
- Ordinary generative SFT and isolated selector attempts did not change the
  relevant routing behavior and were rejected before paid evaluation.
- Earlier V18/V19 GRPO could train only compact live decisions while topology
  remained frozen, so it could not train the whole conductor. r57 removes that
  training-architecture limitation without changing the accepted checkpoint.
- The accepted live-control adapter was trained on the compact action protocol;
  it does not yet produce valid unified full-action outputs. Stage-1 full-action
  SFT is required before unified OPD training.
- No later rejected checkpoint was composed into the product artifacts above.

## Hard Operating Rules

- Every external paid worker request must use `https://yunwu.ai/v1`.
- Worker request timeout is 600 seconds.
- Each arm has one global ceiling of 120 paid worker calls.
- Provider retries and task retries are zero.
- Run one task attempt at a time. Do not launch broad campaigns.
- Stop immediately on infrastructure invalidity, protocol invalidity, a closed
  verdict, or loss of a credible learning signal.
- Do not retry a closed task or campaign without a genuinely new preregistered
  version and a causal reason.
- Never treat infrastructure failures as task or conductor negatives.
- Protect benchmark-owned tracked tests from agent modification.
- Do not start the grand evaluation until the candidate passes local,
  whole-task holdout, identity, and anti-forgetting gates.
- Update this document after preregistration, every completed arm, every
  admission decision, every training gate, and every stop verdict.

## Current Evidence

### Benchmark verdict

The last partial TerminalBench campaign, r33, was rejected after 19 passes and
9 failures in 28 scored tasks. It does not support a superiority claim and must
not be resumed. Full comparator reruns are prohibited.

### Unified conductor and dense-credit readiness

The zero-cost r57/SEED preflight is complete:

- One policy can now emit the initial topology, replacement topology,
  `continue`, `handoff`, and `complete` actions inside the same verifier-scored
  episode. Compact historical rollouts and unified trajectories have distinct,
  fail-closed protocol attestations.
- The trainer now carries aligned `hindsight_logprobs` and an explicit
  `hindsight_mask`. The native CUDA loss implements confidence-gated on-policy
  distillation independently from outcome GRPO and the existing teacher signal.
- A tied-reward synthetic group produced nonzero dense credit only on the two
  selected conductor tokens. Prompt and environment tokens remained zero; a
  skill-supported action received more credit than an unsupported action.
- The hindsight contract accepts anonymous capability/role guidance tied to
  observed decision IDs and rejects concrete model/provider identities, fixed
  worker slots, oracle data, hidden tests, reference patches, and unobserved
  evidence.
- The native extension rebuilt successfully. Focused verification is 147/147
  runtime/control tests plus 48/48 trainer, audit, rescoring, and
  hindsight-contract tests.
- Readiness report:
  `scratchpad/fugu_seed_opd_v1/readiness_report.json`
  (`b48c87e395c436f6491ee2fb017c1167435386cc77200b289102aa52d2c79237`).

Verdict: `LOSS_FORMULA_AND_CONTRACT_READY_RESCORE_BACKEND_INVALID`. The native
loss, transport, masks, and fail-closed skill contract remain verified. The
later repeatability audit invalidated the vLLM prompt-logprob backend for dense
credit, so no OPD optimizer result is accepted until deterministic matched-
branch scores replace it. No training job is running.

### SEED Stage-1 corpus gate

The first fail-closed analyzer-corpus inventory is complete. It reviewed only
explicitly admitted current-pool evidence rather than globbing historical route
logs:

- Five authoritative completed trajectories were reviewed. Four are
  TerminalBench-derived and remain excluded from training.
- One clean, non-benchmark, train-only success is eligible for analysis. It was
  sanitized into four capability-profile permutations with no provider/model
  identity or fixed worker slot in the learned surface.
- There are zero eligible failures, zero eligible causal train pairs, and only
  one distinct task. The minimum pilot gate is 12 independent trajectories on
  six tasks, including at least three successes, three failures, and six causal
  pairs.
- The historical validators cannot be replayed byte-for-byte because their
  frozen manifests hashed a mutable runtime path that changed at r57. The new
  audit therefore verifies the immutable conversion rows and the complete
  result/route/trajectory hash chain directly and records the runtime-source
  drift instead of hiding it.

Verdict: `INSUFFICIENT_ADMITTED_TRAJECTORIES_NO_TRAINING`. Training authorization
is false. No external calls, paid calls, optimizer steps, or checkpoint were
created.

Artifacts:

- Inventory spec: `scratchpad/fugu_seed_stage1_v1/inventory_spec.json`
  (`547ef8e80365d7dc8698a3d162085748848f30c1a774f0e1536756dd0ce91f7a`).
- Identity-free candidates:
  `scratchpad/fugu_seed_stage1_v1/analyzer_candidates.jsonl`
  (`1493bee69b635a93e4639d38b8ee4c000aaf43aa58562b3a2ad7397ac90492e4`).
- Inventory report: `scratchpad/fugu_seed_stage1_v1/inventory_report.json`
  (`54bc252e28f8dbf110e05369a2f33cef355dd7d83df654925cf109bb907f96af`).

### Skill-conditioned rescoring

The token-alignment scorer contract is implemented:

- It reconstructs the ordinary chat-template tokens and fails closed on any
  mismatch before adding privileged context.
- It appends the validated training-only skill to the final user message, then
  submits the original sampled conductor token IDs for prefill scoring. It does
  not generate or substitute a counterfactual action.
- The real local smoke reconstructed all 843 ordinary prompt tokens exactly,
  produced a 910-token skill-augmented prompt, and re-scored the same 32 sampled
  conductor tokens. All 32 action tokens and zero environment tokens were
  selected by the hindsight mask.
- The extractor now selects the submitted token ID explicitly when a response
  also contains top alternatives, and fails closed if that token is absent.
  Interleaving preserves exact alignment over later environment observations
  and multiple conductor decisions.

The original numerical readiness verdict is revoked. Ten identical parent
requests on each of three frozen rows produced mean-token log-probability ranges
of `0.04625`, `0.01518`, and `0.00500` on the vLLM endpoint. These are much
larger than the OPD shifts used by v3-v6. Disabling CUDA graphs and prefix
caching and switching to Triton attention did not make the endpoint repeatable.
Therefore vLLM prompt logprobs are invalid for dense credit and sub-percent
rollback measurement on this host.

A direct Transformers/PEFT scorer using deterministic algorithms, math SDPA,
the same loaded model, and adapter switching reproduced four parent probes
exactly, including the longest 7,859-token case. Long contexts use chunked KV
prefill without changing target tokens. The scorer is now integrated into the
OPD data path. Ordinary rollout probabilities remain separate from a new
`opd_reference_logprobs` channel; the Python loss and native CUDA loss compute
the skill shift only from deterministic ordinary and hindsight branches. A
partial reference/hindsight/mask triple fails closed, and the retired endpoint
scorer aborts before it can create training data.

The clean v7 train-only audit then used one fresh contract-valid `continue`
action sampled locally from the accepted parent. The same loaded parent adapter
scored the ordinary 846-token prompt and the skill-augmented 922-token prompt
twice each with exact repeats and the same seven submitted action tokens. The
token shift was nonconstant, ranging from `0.000000` to `0.040357` with mean
`0.006877`; the resulting mean confidence gate was `0.508572`. The learned
surface remained identity-free, the parent SHA was unchanged, and no v3-v6
score value was reused. `make build` completed and all 55 GRPO tests pass.

Verdict: `DETERMINISTIC_MATCHED_BRANCH_BATCH_AUDIT_PASS`. One replay-anchored
local optimizer preflight is authorized. Product promotion and the held-out
evaluation remain unauthorized. No external or paid calls were made and no
optimizer step was taken.

Matched batch:
`scratchpad/fugu_seed_opd_matched_v7/matched_batch.bin`
(`98c0e3d3cff63f65f31caf74eb99a01b75607296554e10b2492d23e93a103009`).
Audit report:
`scratchpad/fugu_seed_opd_matched_v7/prepared_report.json`
(`a28f07fc63b0d8add408acdd2b6a87d9fe7269f2a729838e8ca57dea511ce06b`).

The authorized replay-anchored v7 optimizer input is also prepared. It combines
the two matched OPD examples with 30 train-only replay examples across all nine
training tasks. Replay action-token mass is 72 `complete`, 72 `continue`, 68
`handoff`, and 72 `replan`; all replay references reproduced exactly twice
under the direct parent scorer. The current scorer also reproduced the stored
ordinary and hindsight OPD branches exactly. Validation labels were excluded.
The batch used 64 local direct-model score calls, zero external calls, zero
paid calls, and zero optimizer steps.

Preparation report:
`scratchpad/fugu_seed_opd_optimizer_v7/prepared_report.json`
(`a93794530c2acaab6bdc14ef02407a1d0fcec49980ba578c80360dfa6d262a9b`).
Training batch:
`output/fugu_seed_opd_optimizer_preflight_v7/run_default/rollouts/step_0/rollouts.bin`
(`e6dd800b0d73836ac93cd1f1b5c2f57da4d463f36a50f8f63db80c84469bb3d3`).

Historical vLLM smoke (numerically invalid):
`scratchpad/fugu_seed_opd_v1/rescore_local_report.json`.
Repeatability audit:
`scratchpad/fugu_seed_opd_optimizer_v6/vllm_rescore_repeatability.json`
(`9f8d7ce05e1618a4f4042878cdeb40925457e3cdd8e691fece45eb781040016c`).

### Native optimizer and anti-forgetting gate

The zero-cost v3 preflight exercised one real native optimizer step from the
accepted live-control parent:

- Commit `d08d50bf` adds exact hindsight transport/rescoring, matched-rollout
  OPD gating, and trainable parent-adapter initialization. The exported child is
  a standalone adapter over the original base rather than an incomplete delta
  over a merged parent.
- The tied group had two reward-0/advantage-0 trajectories and 14 selected
  action tokens. Native mean gate `0.50857258` and shift `0.0068776407` matched
  the precomputed references within `1e-6`; gradient norm was `0.00098309`.
- The parent SHA remained unchanged. All 504 child tensors preserved the parent
  schema and changed finitely; the relative update L2 was only `1.34e-4`.
- Paired replay compared parent and child on all 178 task-isolated v16
  validation decisions. Both remained 178/178 contract-valid, but six action
  signatures changed: one improved and five regressed. Frozen-label matches
  fell from 82 to 78 and false completions increased from one to two.

Verdict: native dense-credit optimization is technically operational, but the
disposable child fails the anti-forgetting gate and is permanently rejected.
No accepted checkpoint changed. The operation used zero external or paid
calls. Future training must mix parent replay/regularization into every update
and retain a post-update rollback gate.

Artifacts:

- Prepared batch: `scratchpad/fugu_seed_opd_optimizer_v3/prepared_report.json`
  (`50afb2e24774e5dfcaa0afa5c28ded4087bde5a28a0ccf7cb68670b58c513e5b`).
- Native optimizer audit: `scratchpad/fugu_seed_opd_optimizer_v3/audit_report.json`
  (`3b69e63a6b77bdef2ff41fded386d0eb414ced8feea1e9809b966dc0bea8c191`).
- Paired replay: `scratchpad/fugu_seed_opd_optimizer_v3/replay_report.json`
  (`5dd8887c5120767e703dd62649397814dee10e2f447605896f4659abee7d0bd7`).

Commit `d4f5279d` adds a first-class replay anchor through the Python transport,
batcher, reference loss, native CUDA loss, and metrics. The zero-cost v5
preflight then tested one replay-anchored update from the same accepted parent:

- The batch contained the exact tied reward-0 OPD group (14 selected action
  tokens) plus 16 train-split replay decisions (168 tokens), four examples per
  action, six tasks, and four anonymous profile permutations. Validation labels
  were not used for training.
- Native gate `0.50955904` and shift `0.0076760696` matched the prepared
  references within `1e-6`. All 14 OPD and 168 replay tokens were consumed;
  gradient norm was finite and nonzero. The parent hash stayed unchanged and
  all 504 child tensors changed finitely.
- All 178 generated holdout decisions remained contract-valid with no new false
  completion. Among 175 parent-stable rows, however, the child introduced one
  regression and one improvement. Three parent rows were decode-unstable.
- The contemporaneous vLLM fixed-token likelihood comparison is invalidated by
  the later repeatability audit and must not be used as evidence.

Verdict: `REPLAY_ANCHORED_V5_REJECTED_ANTI_FORGETTING_GATE`. The replay anchor
substantially reduced forgetting but did not meet the preregistered no-regression
gate. The child is permanently rejected and no accepted checkpoint changed.
The experiment used one optimizer step, local inference only, and zero
external or paid calls. The structural defect is that equal replay-example
counts were not equal replay gradient mass: the handoff targets contributed 68
tokens while complete and continue contributed only 32 each.

Artifacts:

- Prepared batch: `scratchpad/fugu_seed_opd_optimizer_v5/prepared_report.json`
  (`47e4b9b690514924ac3624ed2c60d9d64a814c54ae22f5da3fbb3a629ab6c483`).
- Native audit: `scratchpad/fugu_seed_opd_optimizer_v5/audit_report.json`
  (`5a83cdb6db4a0f61a06f27188f74e536235d53bba7c98bf3507b1290eed005c3`).
- Generation replay: `scratchpad/fugu_seed_opd_optimizer_v5/replay_report.json`
  (`b2d1bc1369025382fad2403ede1994e8bb15153372b0274eed8bad7edaca72e0`).
- Fixed-token likelihood: `scratchpad/fugu_seed_opd_optimizer_v5/likelihood_report.json`
  (`1e27dcf95228d4ba9972d53a58cc9e0cae7929106d4bbc5de9acb868171815ad`).
- Final decision: `scratchpad/fugu_seed_opd_optimizer_v5/decision_report.json`
  (`96dbf2a8075d3aa8e647fc12f0788c3987ff043e0b494acaeb7d8dbfe6c42815`).

The causally isolated v6 preflight reused the exact v5 OPD samples and changed
only train-side replay composition. It used 30 replay examples across all nine
training tasks, with action-token mass of 72 complete, 72 continue, 68 handoff,
and 72 replan tokens:

- Native optimization consumed exactly 14 OPD and 284 replay tokens. The parent
  remained unchanged, all 504 child tensors changed finitely, and the stopped
  pre-control launch performed zero optimizer steps.
- Generation remained 178/178 contract-valid. Across 175 parent-stable rows,
  v6 introduced zero regressions and corrected one parent error; false
  completion remained one.
- The repeatable direct-model audit nevertheless found overall target
  likelihood down `0.000100` per token, handoff down `0.000010`, and replan down
  `0.003735`. Complete and continue improved.
- More fundamentally, v6 reused the same nondeterministic vLLM ordinary and
  hindsight OPD scores as v5. The native step faithfully optimized its inputs,
  but those inputs do not establish valid dense credit.

Verdict: `V6_REJECTED_INVALID_DENSE_CREDIT_AND_ANTI_FORGETTING_REGRESSION`.
The candidate is permanently rejected; no accepted checkpoint changed and no
paid calls were made.

Artifacts:

- Prepared batch: `scratchpad/fugu_seed_opd_optimizer_v6/prepared_report.json`
  (`5cd10b4d3548181abbfac26fe81cede4743a1142da32d5b41a318983f3d2550e`).
- Native audit: `scratchpad/fugu_seed_opd_optimizer_v6/audit_report.json`
  (`2cb98e6849a3987e82d6209812791c62000a18598af22d80e29f8eadfc6edb56`).
- Generation replay: `scratchpad/fugu_seed_opd_optimizer_v6/replay_report.json`
  (`bf5923b22560f7d87632a74a66aba6bcacc02484befb10bc374c5a9e45aea820`).
- Direct likelihood: `scratchpad/fugu_seed_opd_optimizer_v6/direct_likelihood_report.json`
  (`9c3fafb6106fe861db4bb976a67c6df8f304cc6196110fabed1e71bd1e4fa356`).
- Final decision: `scratchpad/fugu_seed_opd_optimizer_v6/decision_report.json`
  (`dc7c37f752e83623eae595a0b4d73cd0efdc2dfd5eab8b3a3608a371caed6f21`).

The v7 preflight is the first optimizer run whose OPD ordinary and hindsight
branches both came from the integrated deterministic direct scorer. It used the
matched seven-token `continue` group plus the same action-token-normalized
30-example train replay:

- One native optimizer step consumed 14 OPD and 284 replay tokens. All 504 child
  tensors changed finitely, the gradient norm was nonzero, and the accepted
  parent hash remained unchanged.
- Direct train-side target likelihood improved overall and for every action.
- A failed eight-way generation launch was classified as infrastructure-invalid
  after the local server died; it was not counted against the conductor. The
  preregistered serial replacement completed 178/178 contract-valid decisions
  with zero regressions across 175 stable parent rows.
- The final exact-repeat direct holdout improved overall likelihood by
  `0.000398` per token. `complete`, `continue`, and `handoff` improved, but
  `replan` regressed by `0.003700` per token across its 18 frozen cases.

Verdict: `V7_REJECTED_REPLAN_ANTI_FORGETTING_REGRESSION`. The corrected
matched-reference OPD pipeline is operational, but this one-sample update is
not a product checkpoint. The child is permanently rejected, the accepted
planner/live-controller artifacts remain unchanged, and no external or paid
calls were made.

Artifacts:

- Native audit: `scratchpad/fugu_seed_opd_optimizer_v7/audit_report.json`
  (`fbac7c20503bc0df42ec4f76ea4f89a853835e302d5306586c0f18936bcad3c7`).
- Train score gate: `scratchpad/fugu_seed_opd_optimizer_v7/train_score_report.json`
  (`285f9671dddfb1bf6800ca2d388bb38c915ccf242da0ba9de38bc086da845acb`).
- Serial generation gate: `scratchpad/fugu_seed_opd_optimizer_v7/generation_report.json`
  (`d2454d316ffe007971d3635d53e177e030d1fe3d255c97ee415018e1cb54910f`).
- Direct holdout likelihood:
  `scratchpad/fugu_seed_opd_optimizer_v7/direct_likelihood_report.json`
  (`ced16f2ff5f3a225e7e5a074b44df2fc30e9f546ec6d365cef4dec57f0621112`).

### Verified causal coordination

One same-task, same-runtime, same-pool causal pair is independently admitted:

- Task: `configure-git-webserver`, permanently excluded from future evaluation.
- Solo arm: anonymous builder, 8 Yunwu calls, clean reward 0; final deployed
  state was removed and the verifier received HTTP 404.
- Coordinated arm: identical builder plus one anonymous independent final-state
  auditor, 7 Yunwu calls, reward 1.
- Both arms used runtime r56, the same pool, fresh environments, zero retries,
  and had no provider, protocol, harness, or integrity error.
- The only intervention was the added auditor position with access to the
  builder.

Artifacts:

- Campaign: `scratchpad/fugu_causal_coordination_v1/campaign_frozen.json`
  (`4d9636973723956c5f515bf99f1f214ac4adfab9f559625f119a9e8c15e68ed7`)
- Pair report: `scratchpad/fugu_causal_coordination_v1/pair_report.json`
  (`6d6f97fa9c6a1448d45453bd394bcd910f14e3ea1f1d98b6270140a6e13eb07d`)
- Independent admission:
  `scratchpad/fugu_causal_coordination_v1/admission_v1.json`
  (`9b6236f4f22d96074f5c1c2473bff5a04337e4e0203156b7caf21ab60852fb9d`)

This proves one observed coordination lift. It does not estimate expected lift
and does not authorize training.

## Next Operation

1. Keep product promotion and grand evaluation stopped. The v3, v5, v6, and v7
   children are rejected and must not be used.
2. Gate every hindsight annotation against observed conductor decision IDs and
   reject task solutions, concrete model/provider identities, fixed worker
   slots, and unsupported causal claims.
3. Preserve the deterministic direct scorer as the only authorized OPD score
   source. Do not reuse v3-v6 score values or the endpoint prompt-logprob path.
4. Do not make a v8 replay-tuning attempt from the same single `continue` OPD
   group. v7 proves that this data is too narrow to preserve all four actions,
   even when the transport, native step, and replay are correct.
5. Collect the missing clean train-side current-pool trajectories one task at a
   time, targeting observed decision points and causal coordination evidence
   across all four actions. Use Yunwu only, a 600-second worker timeout,
   120-call global ceiling, zero retries, and stop immediately on invalidity or
   absent learning value. TerminalBench, holdouts, oracle runs, and the rejected
   historical outcome corpus remain excluded.
6. Re-run the frozen Stage-1 gate after every admitted task. At threshold, train
   full-action Stage-1 SFT with a replay-majority mixture, then continue with
   replay-anchored OPD. Reject either stage on action validity, replay,
   profile-equivariance, completion, or causal-lift regression.
7. Promote nothing until whole-task train/holdout evidence shows coordination
   lift while preserving anti-forgetting, model-agnostic construction, and pool
   replacement behavior.
8. Run the smallest held-out Yunwu product gate only after those conditions;
   do not restart TerminalBench r33 or rerun published comparators.

## Grand Evaluation

The grand evaluation is the final product gate, not a research exercise.

1. Use official published comparator scores; do not rerun other models.
2. Start with the smallest preregistered held-out Yunwu promotion slice.
3. Classify task failures separately from provider, harness, and protocol
   invalidity.
4. Stop as soon as superiority is mathematically impossible or the candidate
   is invalid.
5. Run remaining TerminalBench 2.1 tasks only while a credible path to at least
   79/89 passes remains.
6. Require consistent SWE-Pro-class agentic evidence before a product claim.

## Reproduction

Verify the accepted trained adapters and current runtime without model calls:

```bash
sha256sum \
  director/director/agentic/fugu_ultra_terminal.py \
  output/fugu_ultra_planner_composite_v11_s20/adapter_config.json \
  output/fugu_ultra_planner_composite_v11_s20/adapter_model.safetensors \
  output/fugu_ultra_live_control_grpo_v16/final_adapter/adapter_config.json \
  output/fugu_ultra_live_control_grpo_v16/final_adapter/adapter_model.safetensors
```

The hashes must match the Current Product table.

Rebuild and verify the unified conductor and dense-credit path without model
calls:

```bash
make build PARALLEL_JOBS=8

PYTHONPATH="$PWD:$PWD/director:$PWD/ultra" \
  director/.venv/bin/pytest -q \
  ultra/tests/test_live_control.py \
  director/tests/test_fugu_ultra_terminal.py \
  director/tests/test_fugu_live_agentic_grpo.py \
  director/tests/test_fugu_decision_correction.py

PYTHONPATH="$PWD:$PWD/director:$PWD/ultra" \
  .venv/bin/pytest -q \
  tests/grpo/test_native_formula.py \
  tests/grpo/test_inference_config.py \
  tests/grpo/test_native_runtime_source.py \
  tests/grpo/test_seed_opd_contract.py \
  tests/grpo/test_seed_hindsight_rescore.py \
  tests/grpo/test_live_agentic_v18_audit.py \
  tests/grpo/test_live_agentic_v19_operation.py \
  tests/grpo/test_resume_step.py \
  ultra/tests/test_seed_hindsight.py

PYTHONPATH="$PWD:$PWD/director:$PWD/ultra" \
  .venv/bin/python scratchpad/audit_fugu_seed_opd_readiness.py

PYTHONPATH="$PWD:$PWD/director:$PWD/ultra" \
  director/.venv/bin/pytest -q \
  director/tests/test_fugu_seed_stage1_corpus.py

PYTHONPATH="$PWD:$PWD/director:$PWD/ultra" \
  director/.venv/bin/python scratchpad/audit_fugu_seed_stage1_inventory.py

PYTHONPATH="$PWD:$PWD/director:$PWD/ultra" \
  .venv/bin/python scratchpad/audit_fugu_seed_rescore_local.py

PYTHONPATH="$PWD" \
  .venv/bin/python scratchpad/audit_fugu_seed_opd_optimizer_v3.py
```

Expected results: native build succeeds; 147/147 runtime/control tests pass;
48/48 trainer/audit/hindsight/rescorer tests pass; readiness status is
`LOSS_FORMULA_AND_CONTRACT_READY_RESCORE_BACKEND_INVALID`. The Stage-1 corpus tests pass
2/2 and its audit returns `INSUFFICIENT_ADMITTED_TRAJECTORIES_NO_TRAINING` with
one eligible independent trajectory, four identity-free profile permutations,
and zero calls or optimizer steps. The local rescore audit returns
`LOCAL_EXACT_RESCORE_READY` with exact ordinary-prompt reconstruction, identical
sampled action token IDs, zero masked environment tokens, and zero paid calls or
optimizer steps. The v3 native audit returns
`NATIVE_OPD_OPTIMIZER_READY_DISPOSABLE_CANDIDATE_NOT_PROMOTABLE`; its paired
replay report must remain rejected and proves why replay anchoring is the next
required training change.


## Closed Work

The following verdicts are final unless a genuinely new preregistered design
changes the causal question:

- TerminalBench r33: rejected; do not resume.
- `video-processing` recovery v4: clean real reward-0 task/product failure; do
  not retry.
- Caffe recovery v3: infrastructure-invalid; do not retry.
- Capability-routing SFT, primary-worker selector, topology selectors/editors,
  and fixed-length route critic: rejected local gates; not integrated.
- Local Ornith qualification: one solo win and two ties, zero coordination
  wins; it did not provide evidence for a superior conductor.
- V18/V19 live-agentic GRPO attempts: no promotable update; zero optimizer
  steps. They are historical diagnostics, not current work.
- Planner-role V2: one real local optimizer step, decisively rejected by the
  exhaustive parent comparison; not integrated and not eligible for paid work.
- SEED OPD v3: one disposable local optimizer step passed native correctness but
  failed anti-forgetting replay; the child is rejected and not integrated.
- SEED OPD v5: replay-anchored native optimization passed, but one stable
  generation regression failed the anti-forgetting gate; its vLLM likelihood
  audit is invalidated. The child is rejected and not integrated.
- SEED OPD v6: action-token-normalized replay removed stable generation
  regressions, but repeatable direct scoring found overall and replan likelihood
  regression. Its OPD score source was also nondeterministic. The child is
  rejected and not integrated.
- SEED OPD v7: deterministic matched-reference OPD and normalized replay
  preserved generated actions and improved overall direct likelihood, but
  regressed frozen `replan` likelihood. The child is rejected and not
  integrated; do not tune another child from the same one-sample OPD group.
- Historical outcome-corpus V2: identity-free and leak-free, but its frozen
  task-acquisition and workflow-ranker gates failed; it is not training data.
- Adaptive causal `swesmith-03234` v3: solo arm was invalid after 25 Yunwu
  calls; the conditional coordinated arm is permanently closed.

Detailed historical evidence remains in the frozen manifests and ledgers under
`scratchpad/`. It is intentionally not duplicated here.
