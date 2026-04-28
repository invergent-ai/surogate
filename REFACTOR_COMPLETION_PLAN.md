# Refactor Completion Plan

This document tracks the remaining implementation work needed to move from the completed tracked phases in `REFACTOR.md` to the fully realized runtime architecture. It intentionally excludes the open roadmap-question bucket from `REFACTOR.md`.

## Status

Tracked phases 0-5 are complete for the scoped implementation. The remaining work is mostly replacing compatibility fallbacks with authoritative runtime paths, then validating the end state.

## Track 1 - Remove Native FP8 MoE WGrad

- [x] Removed the native FP8 CUTLASS MoE wgrad implementation after the real LoRA acceptance workload showed it was slower end-to-end than the BF16 wgrad path.
- [x] Deleted `csrc/src/kernels/moe/moe_fp8_wgrad_cutlass.*` and removed it from the CMake source list.
- [x] Removed the public `moe_grouped_gemm_weight_grad_fp8` kernel API and the focused FP8 wgrad unit test.
- [x] Removed recipe and executor routing that quantized BF16 activations/gradients into FP8 only to fall back or lose end-to-end time.
- [x] Kept the BF16 MoE wgrad path as the authoritative implementation.
- [x] Kept the MoE FP8 expert-weight cache as a separate dgrad optimization gate under `SUROGATE_FP8_MOE_WEIGHT_CACHE`.

Acceptance:

- [x] Codebase has no callable FP8 CUTLASS MoE wgrad path.
- [x] LoRA MoE adapter gradients route directly to BF16 grouped wgrad.
- [ ] Re-run 5-step Qwen3.5 MoE acceptance once the local `.venv` has PyTorch restored.

## Track 2 - Promote Hook Dispatch From Opt-In To Runtime Path

- [x] Move CPU-stream prefetch from imperative gather/wait paths into registered `before_consume` schema hooks.
- [x] Move streaming grad offload after all-reduce into registered `after_all_reduce` hooks.
- [x] Move EP all-to-all observation and follow-up actions into registered `after_all_to_all` hooks.
- [x] Move LoRA after-produce actions for dense, router, shared expert, and grouped expert projections into registered schema hooks.
- [x] Keep the current imperative paths as fallback until hook parity is green under CI.
- [x] Flip default dispatch after parity; keep only compatibility fallbacks that still have a non-schema or opt-out caller.

Completed subphase:

- Grouped MoE LoRA after-produce dispatch now uses the compiled structural activation slot and schema id for `expert_gate_up`, `expert_up`, and `expert_down`; legacy slot names remain only as fallback for graphs without structural metadata.
- MoE grouped GEMM compile attrs now carry `layer_idx`, `hook_schema_id`, `allow_quant`, and forward activation schema slot metadata, matching the dense/router/shared expert hook path.
- Nemotron-style separate expert-up grouped GEMM now emits the structural `expert_up` activation name so schema hooks can target it.
- Schema hook dispatch is now enabled by default across the executor, gradient store, and LoRA gradient manager. `SUROGATE_ENABLE_SCHEMA_HOOK_DISPATCH=0` remains as an emergency opt-out while compatibility fallbacks stay in place for non-schema graphs.

Acceptance:

- [x] `SUROGATE_ENABLE_SCHEMA_HOOK_DISPATCH=1` and default dispatch use the same runtime path; real-model 5-step parity remains part of the model acceptance queue.
- [x] Hook registry reports full target coverage for CPU streaming, DP all-reduce, EP all-to-all, reduce-scatter, and LoRA after-produce paths.
- [x] No runtime path depends on ad hoc hook-slot names when a structural `(schema_id, slot)` target exists.

## Track 3 - Make Schema-Driven Allocation Authoritative

- [x] Replace enum-driven `BufferPlan` allocation decisions with schema-derived per-layer/per-slot allocation plans.
- [x] Use schema shape, lifetime, residency, distribution, and save-for-backward metadata as the primary allocator inputs.
- [x] Keep the current TensorSlot/legacy enum plan as a parity comparator during rollout.
- [x] Remove or narrow legacy safety slacks after schema allocation parity is validated.
- [x] Preserve capture safety for CUDA graph paths and arena fallback diagnostics.

Completed subphase:

- `BufferPlan` now assigns each block-schema slot an explicit allocation decision: lifetime, residency, full/local bytes, and whether that decision is authoritative.
- Runtime debug and regression artifacts expose authoritative schema arena totals for frame, save-for-backward, persistent activation, host-stream activation, and total activation arena bytes.
- Phase arena summaries carry the same schema-authoritative totals alongside the compiled-region arena sizes, making the compiled layout the rollout parity comparator.
- Coverage reports now mark schema allocation readiness separately from schema presence, storage declaration, hook readiness, and descriptor readiness.
- Schema-authoritative phase arenas now expose compiled-vs-schema safety/extra bytes and size frame/save-for-backward arenas with the schema plan while retaining compiled-layout coverage as the capture-safe floor.
- Non-MoE authoritative schema plans use a narrower upfront stack heuristic; MoE keeps the conservative slack until its op-internal temps are fully stack-bound.

Acceptance:

- [x] `SUROGATE_BLOCK_SCHEMA_PLAN_ASSERT=1` is green across the real-model queue.
- [x] Schema allocation bytes are authoritative in runtime debug and regression artifacts.
- [x] Peak CUDA memory does not regress; Gemma4 and MoE acceptance rows show the expected allocation savings.
- [x] Save-for-backward and frame-local arena summaries match schema lifetime declarations.

Validation evidence:

- 2026-04-28: `SUROGATE_BLOCK_SCHEMA_PLAN_ASSERT=1` 5-step real-model queue passed in `regression_baselines/current/schema_assert_20260428_r4`.
- Matrix coverage: `qwen3__fp8__single_gpu__gpu__dense`, `qwen3_5__fp8__single_gpu__gpu__dense`, `gemma4__fp8__single_gpu__gpu__dense`, `gpt_oss__fp8__single_gpu__gpu__moe_grouped`, and `qwen3_6_moe__fp8__2gpu_dp_ep__gpu__moe_grouped` report `coverage=1.0`, `eligible=5`, `passed=5`.
- Direct queue add-on: `examples/sft/qwen3moe/qwen3moe-nvfp4.yaml` passed 5 steps with artifact `regression_baselines/current/schema_assert_20260428_r4/artifacts/qwen3moe-nvfp4-direct.json`.
- Peak CUDA memory guardrail: every r4 acceptance artifact records `cuda_peak_memory_bytes=519569408`; locked baseline files currently carry empty metrics, so this r4 queue is the first complete recorded memory baseline for these rows.
- Gemma4 allocation savings: `schema_allocation_authoritative=1`, `schema_allocation_unresolved_slots=0`, and `schema_activation_shape_savings_bytes=687865856`; the authoritative activation arena is `792723456` bytes versus `1908408320` bytes of baseline max activation shape.
- MoE allocation savings: EP MoE activation arenas remain conservative because dynamic activation slots are still unresolved, but expert-parallel parameter locality savings are recorded: Qwen3.6 MoE reports `schema_expert_parallel_param_shape_savings_bytes=32212254720`, and Qwen3MoE NVFP4 direct reports `43486543872`. GPT-OSS single-GPU MoE has no EP-locality savings expected.

## Track 4 - Remove Descriptor Capability Fallbacks - COMPLETE

- [x] Audited quantized recipe ops for explicit `OpCapabilities`, `MatmulCapabilities`, `MoECapabilities`, storage compatibility, communication profile, and grouped semantics.
- [x] Converted recipe capability fallback logging into descriptor-deny diagnostics for supported quantized recipes.
- [x] Removed legacy allow-paths now that dense, Mamba out projection, and grouped MoE ops are annotated.
- [x] Kept explicit unsupported statuses for combinations that are intentionally unavailable.

Acceptance:

- [x] `SUROGATE_RECIPE_CAPABILITY_LOG=1` reports no legacy allow-paths for the real-model acceptance queue.
- [x] North-star coverage rows derive FP8/FP4 support from descriptor results, not fallback assumptions.
- [x] Unsupported FP4/MoE combinations fail or skip with explicit descriptor reasons.

## Track 5 - Activate Fusion Registry Rewrites

- [x] Add a non-mutating graph rewrite preview that lists exact fusion candidates and replacement descriptors.
- [x] Implement the first descriptor-driven rewrite for a low-risk dense fusion family.
- [x] Keep MoE and Mamba candidates visible in preview with explicit unsupported-runtime reasons until fused descriptors and parity tests exist.
- [x] Remove rollout rollback flags; supported fusion rewrites are the default runtime path.

Completed subphase:

- Compiled graphs now keep a deterministic `fusion_rewrite_preview` with rule name, replacement op, candidate span, source op ids/names, applied state, and rollout reason.
- Forward `matmul -> bias_add` is rewritten to the existing `matmul_bias` descriptor when the matmul output has exactly one consumer and operand arity matches the fused kernel contract.
- Fusion rewrites are production-enabled by default for supported rules; there is no global or per-rule rollback env for production rewrites.
- Removed the rollout-era per-rule opt-in path; unsupported rules are preview-only until their production parity evidence exists.
- Production descriptor rewrites now cover every registered rule that has a real fused runtime descriptor: `matmul_bias`, `matmul_swiglu`, `qkv_qknorm_rope` (`qkv_qk_norm + rope -> qkv_qk_norm_rope`), `residual_rmsnorm`, `lmhead_loss`, and `moe_routing_topk_softmax`.
- `moe_routing_topk_softmax` forward rewrites `moe_softmax -> moe_topk` to `moe_topk(softmax=True)` for both selected-normalized and full-softmax selected-weight routing, and backward rewrites `moe_topk_backward -> moe_softmax_backward` to `moe_topk_backward(softmax=True)`.
- Rules without a fused runtime descriptor remain preview-only and are not required runtime paths: `qkv_qknorm`, `mamba_gated_rmsnorm`, and `moe_permute_quantize`.
- MoE and Mamba fusion candidates report explicit unsupported-runtime reasons instead of silent scaffold behavior.
- Regression artifacts now include `fusion_preview` alongside descriptor, schema, buffer-plan, and arena summaries.

Acceptance:

- [x] Fusion preview is deterministic and appears in regression artifacts.
- [x] At least one dense fusion rewrite is enabled by default with numerical parity and no perf regression.
- [x] MoE fusion candidates are explicitly unsupported as runtime rewrites until fused descriptors and parity tests exist.

Validation evidence:

- 2026-04-28: `cmake --build csrc/build --target unit-tests -j 16` passed.
- 2026-04-28: `./csrc/build/unit-tests "[fusion_rule]"` passed, including compile tests for all production descriptor rewrites.
- 2026-04-28: `.venv/bin/pytest -q tests/test_regression_artifact_writer.py tests/test_regression_baseline_runner.py --no-gpu` passed.
- 2026-04-28: `make wheel-dev` passed and refreshed the `.venv` extension.
- 2026-04-28: 5-step `qwen3_5__fp8__single_gpu__gpu__dense` regression passed in `regression_baselines/current/fusion_rewrites_20260428`; the artifact includes `fusion_preview` and no candidates for this already-fused model graph.
- 2026-04-28: 5-step `qwen3_5_moe__fp8__single_gpu__gpu__moe_grouped` regression passed in `regression_baselines/current/moe_routing_fusion_saved_logits_20260428`; artifact shows all 80 MoE routing candidates applied (`40` forward `moe_routing_topk_softmax`, `40` backward `moe_routing_topk_softmax_backward`).

## Track 6 - Remove Remaining Name-Only Gradient Fallback - COMPLETE

- [x] Replaced executor split-planner use of `base_param_from_grad_heuristic` with an exact precomputed parameter-gradient output map derived from `DslGradStore::param_names()`.
- [x] Kept the old `d_<base>` inference only as a private graph-compiler debug cross-check under tensor-kind diagnostics.
- [x] Deleted the public executor heuristic utility once no runtime caller remained.

Acceptance:

- [x] Runtime gradient routing never infers parameter gradients from `d_<base>` names alone.
- [x] Tensor-kind debug checks still catch accumulator and activation-gradient misclassification.
- [x] Existing CUDA graph split behavior is unchanged by using exact grad-store names for the split-planner tail.

## Track 7 - Final Validation And Cleanup

- [ ] Run required local gates: `make wheel-dev`, `make test-unit`, `make test-integration`, and no-GPU regression tests excluding `tests/test_distributed.py`.
- [ ] Run real-model 5-step acceptance queue:
  - [x] `examples/sft/qwen3/qwen3-lora-fp8.yaml`
  - [x] `examples/sft/qwen35/qwen35-text-lora-fp8.yaml`
  - [x] `examples/sft/gemma4/gemma4-e2b-lora-fp8.yaml`
  - [x] `examples/sft/gpt-oss/gptoss-lora-mxfp4.yaml`
  - [x] `examples/sft/qwen36moe/qwen36moe-lora-fp8.yaml`
  - [x] `examples/sft/qwen3moe/qwen3moe-nvfp4.yaml`
- [ ] Run selected 2-GPU DP and EP smoke cases with the new authoritative paths enabled.
- [ ] Refresh regression artifacts and locked baselines only after the new paths are stable.
- [ ] Update `REFACTOR.md` to mark the completion tracks closed and point to this document for final evidence.

Acceptance:

- [ ] No required path depends on scaffold-only, inert-only, or diagnostic-only implementation for the refactor goals; preview-only fusion families without runtime descriptors are explicitly non-required unsupported paths.
- [ ] All remaining fallback paths are either explicit unsupported-mode fallbacks or documented operational safety fallbacks.
- [ ] `git status --short` is clean after final commits.

Order:

Track 6 - remove name-only grad fallback.
Track 4 - remove descriptor capability fallbacks.
Track 1 - remove FP8 MoE wgrad.
Track 2 - promote hook dispatch.
Track 3 - schema-driven allocation authoritative.
Track 5 - activate fusion rewrites.
Track 7 - final validation.
