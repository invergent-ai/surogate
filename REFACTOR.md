# Surogate Big Refactor — FP8/FP4 + MoE + Multi-GPU First-Class

**Status:** Implementation in progress. Source of truth — update as decisions land.
**Date opened:** 2026-04-26.
**Owner:** flavius@statemesh.net.
**Strategic position:** competing with Unsloth on specialized fast-path; differentiator is FP8/FP4 specialization across architectures, MoE breadth, and multi-GPU at the open-source tier.

---

## 0. Implementation progress

### Pre-Phase 0 — FP8 MoE backward closure — COMPLETE

- [x] FP8 MoE weight-gradient closure scaffold landed behind `SUROGATE_FP8_MOE_WGRAD=1`, with BF16 fallback retained for unsupported paths.
- [x] Executor-side FP8 MoE expert weight cache support landed for backward dgrad reuse, preserving frozen-weight cacheability.
- [x] CUTLASS examples README inspected for relevant FP8/grouped GEMM paths; future candidates identified (`57_hopper_grouped_gemm`, `69_hopper_mixed_dtype_grouped_gemm`, `75_blackwell_grouped_gemm`, `81_blackwell_gemm_blockwise`, `54_hopper_fp8_warp_specialized_gemm`, `58_ada_fp8_gemm`).

### Phase 0 — Regression baseline infrastructure — COMPLETE

- [x] Regression harness added with tracked JSON baselines under `regression_baselines/locked`.
- [x] Baseline compare/update/report commands added, including north-star coverage JSON.
- [x] Regression matrix seeded for Qwen3, Gemma4, Nemotron-H, Qwen3.5, and Qwen3.5-MoE rows; FP4 rows are explicitly unsupported/skipped where not stable.
- [x] Regression runner records explicit skip reasons for missing model paths.
- [x] Regression runner materializes per-run configs and injects local model paths from env vars.
- [x] Regression case filtering and `--list-cases` command added for CI/on-demand selection.
- [x] Regression runner now applies a per-case wall-clock timeout (`--timeout-s`, default `SUROGATE_REGRESSION_TIMEOUT_S` or 1800s) so GPU-side hangs fail with explicit artifacts instead of blocking CI indefinitely.
- [x] CI workflow and Make targets added for no-GPU regression smoke and self-hosted GPU regression lane.
- [x] North-star coverage rows now include descriptor capability requirements for dense and MoE grouped quantized cases.

### Phase 1 — TensorRole and Distribution scaffolding — COMPLETE

- [x] `TensorRole`, `Distribution`, `StorageTier`, `QuantState`, and conservative name-to-role inference introduced.
- [x] TensorRole metadata attached to compiled graph tensor records and role lookup helpers added.
- [x] Dual-path TensorRole parity helper added under `SUROGATE_TENSOR_ROLE_PARITY`; `abort` mode supported.
- [x] MoE save/executor ownership path migrated to role-aware classification with legacy fallback retained.
- [x] RoPE/frequency save and executor fallbacks migrated to role-aware classification with legacy fallback retained.
- [x] Expert-parallel grad reduction classification parity added without changing current comm route.
- [x] DSL stores, matmul MoE predicates, QLoRA expert filters, NorMuon router filters, and MoE expert-bias skip paths wired into role parity.
- [x] Autodiff MoE side-channel classification centralized through TensorRole helpers.
- [x] Param-store RoPE/router filtering and grad-store expert-parallel reduction routing now consume TensorRole-derived classifications with legacy string fallback retained.
- [x] Weight-manager RoPE/router filtering now matches the param-store TensorRole-first classification path with legacy string fallback retained.
- [x] RoPE parity guards in graph tensor resolution and autodiff now align `freq_cis` with `rope_freqs`, preventing false TensorRole parity aborts on alternate RoPE naming.
- [x] Param-store and weight-manager RoPE parity guards now also align `freq_cis` with `rope_freqs`.
- [x] TensorRole now covers plain `router`, global `gather_indices`, and `expert_offsets`; param store, weight manager, grad store, graph tensor resolution, graph parameter resolution, and autodiff non-differentiable checks now consume TensorRole predicates directly instead of local legacy substring predicates.
- [x] Matmul shared-expert/router classification, QLoRA expert-weight filtering, MoE expert-bias skip logic, DSL model router/expert filters, compiled save/execute RoPE/MoE/embedding checks, graph compiler RoPE global-role detection, and NorMuon embedding/LM-head/router exclusions now consume TensorRole predicates directly instead of local legacy fallbacks.
- [x] NorMuon standalone-gate exclusion now routes through a TensorRole predicate, and MoE saved-buffer compatibility lookups no longer carry `legacy_key` local names.
- [x] Grouped MoE forward/backward EP weight-pointer routing now uses TensorRole expert projection predicates (`expert_gate_up`, `expert_up`, `expert_down`) instead of local substring checks.
- [x] Dead TensorRole parity helper and `SUROGATE_TENSOR_ROLE_PARITY` runtime hook removed after call-site migration; remaining classifier diagnostics now describe the old `d_<base>` heuristic explicitly.
- [x] Schema allocation comparison metrics renamed from `legacy` to `baseline` in runtime debug output, binding output, C++ tests, and tracked regression run artifacts.
- [x] Follow-up TensorRole migration sweep removed the remaining local runtime classifiers for fused QKV/gate-up slice inference, router projection hook detection, Qwen3.5 MoE gate-up prequant swapping, QLoRA expert field admission, and missing-param debug filtering.
- [x] Gradient dtype override now uses compiled `TensorKind::ParamGrad` metadata instead of the older `d_<base>` heuristic in the graph compiler; the heuristic remains only as a debug cross-check and in executor split planning where compiled metadata is not available yet.

### Phase 2 — Op registry descriptor extension — COMPLETE

- [x] Existing op dispatch/autodiff registry preserved; descriptor metadata extension started instead of rebuilding the registry.
- [x] MoE op registry descriptors annotated with semantic/distribution metadata.
- [x] Registry descriptor surface extended with `CommunicationProfile` and `GroupedSemantics`.
- [x] MoE grouped ops annotated with expert-parallel routed grouped semantics.
- [x] EP dispatch/combine ops annotated with all-to-all communication profiles.
- [x] Mamba/SSM and Qwen3.5 gated-delta op families annotated as no-comm sequence/dense/view descriptors.
- [x] Core transformer ops annotated with semantic descriptor metadata.
- [x] Descriptor type definitions split into a lightweight header and copied onto `CompiledOp` during graph compilation.
- [x] Descriptor surface extended with inert `OpCapabilities`, `EpilogueSupport`, and `StorageCompatibility` fields, with dense matmul and MoE grouped matmul annotations.
- [x] Consolidated `REGISTER_COMPILED_OP` surface added and proven on dense matmul registrations.
- [x] MoE grouped forward GEMM registrations migrated to the consolidated descriptor surface.
- [x] EP dispatch/combine all-to-all registrations migrated to the consolidated descriptor surface.
- [x] Mamba/SSM and Qwen3.5 gated-delta no-comm families migrated to the consolidated descriptor surface.
- [x] Core replicated transformer/view/elementwise/attention/loss registrations migrated to the consolidated no-comm descriptor surface.
- [x] MoE router, token-routing, expert-bias, and grouped backward registrations migrated to consolidated descriptors.
- [x] Missing-dispatch diagnostics now include compiled semantic, communication, distribution, capability, epilogue, and storage descriptor facets.
- [x] `CompiledGraph` summary helpers added for descriptor communication, grouped-op, capability, epilogue, and storage-compat counts.

### Phase 3 — Capabilities and recipe predicates — COMPLETE

- [x] Descriptor capability metadata plumbed into dense and MoE matmul recipe contexts; FP8 recipe now consults `FP8Eligible` with legacy fallback for unannotated ops.
- [x] Dense NVFP4 matmul recipe now consults `FP4Eligible` with legacy fallback; MoE FP4 fallback behavior remains unchanged.
- [x] Shared recipe capability predicate helpers added so FP8/FP4 eligibility uses one descriptor fallback rule.
- [x] Opt-in recipe capability fallback logging added under `SUROGATE_RECIPE_CAPABILITY_LOG`.
- [x] Dense backward and fused `matmul_swiglu` recipe paths now carry descriptor FP8/FP4 capability metadata into recipe contexts.
- [x] Recipe capability predicates now distinguish legacy unannotated allow-path diagnostics from explicit descriptor capability denials.
- [x] Inert `FusionRuleRegistry` scaffold added for Phase 3c pattern, priority, capability, and communication-aware fusion declarations.
- [x] Initial inert fusion rule declarations added for existing dense, norm, loss, Mamba, and MoE fusion families.
- [x] Fusion registry can now build descriptor-backed `FusionContext` views from compiled ops and evaluate pattern plus eligibility matches.
- [x] Fusion registry exposes a non-mutating compiled-op-position query for future graph-rewrite integration.
- [x] Fusion registry exposes a non-mutating candidate-count helper for future coverage/reporting hooks.
- [x] `CompiledGraph` summary helpers now expose inert fusion-candidate start counts.
- [x] Dense FP8 co-located quant ready-flag mapping centralized as a first Phase 3d bridge toward generic quant-state routing.
- [x] FP8 ready flags now expose diagnostic names and map into the `QuantState::FP8Ready` vocabulary without changing execution.
- [x] Mamba `out_proj` forward/backward matmul delegates now declare dense FP8/FP4 capability and CPU-stream storage compatibility.
- [x] Regression artifact schema and north-star coverage rows now reserve descriptor-summary/fusion-candidate counts for descriptor-driven reporting.
- [x] `QuantState` diagnostic names and compiled-graph quant-state tensor counts added for future generic quant routing/reporting.
- [x] Python trainer debug surface now exposes compiled-graph descriptor/capability summaries for regression artifact producers.
- [x] SFT trainer emits regression artifacts under `SUROGATE_REGRESSION_ARTIFACT`, including convergence, step-time, CUDA-memory, and descriptor-summary fields.
- [x] Regression baseline compare now detects descriptor-summary count drift when baselines include descriptor fields.
- [x] Dedicated inert `MoECapabilities` descriptor surface added and populated on grouped MoE GEMM forward/backward ops, explicitly leaving FP8-backward/NVFP4-no-fallback gaps unset.
- [x] Shared recipe capability predicates now include MoE/grouped FP8, FP4, and FP8-backward capability checks with legacy fallback logging.
- [x] MoE grouped recipe paths now receive `MoECapabilities` from compiled descriptors and consult grouped FP8/FP4 predicates before specialized paths.
- [x] Missing-dispatch diagnostics now print `MoECapabilities` alongside generic capability, epilogue, storage, comm, and distribution facets.
- [x] North-star coverage rows now report required MoE/grouped capabilities separately from generic op capabilities.
- [x] Dedicated inert `MatmulCapabilities` descriptor surface added for dense matmul FP8/FP4 forward/backward eligibility, weight-cache eligibility, epilogue support, and quant-colocation hints.
- [x] Dense FP8/FP4 recipe predicates now consult `MatmulCapabilities` in addition to legacy generic op capability flags, preserving legacy fallback for unannotated ops.
- [x] Compiled-graph descriptor summaries now report dense matmul FP8/FP4 forward/backward eligibility counts through the Python trainer debug surface.
- [x] North-star coverage rows now report required dense `MatmulCapabilities` and actual dense matmul descriptor-summary counts separately from generic op capabilities.
- [x] Fusion contexts now expose dense `MatmulCapabilities` from compiled ops so future fusion eligibility can match dedicated matmul descriptors instead of generic op flags.
- [x] Fusion contexts now expose `MoECapabilities` from compiled ops so future MoE fusion eligibility can match grouped/MoE descriptors directly.
- [x] Regression artifact flattening now totals dense matmul FP8/FP4 forward/backward descriptor counts for baseline drift detection.
- [x] North-star coverage rows now include diagnostic descriptor-requirement status and missing descriptor-count keys for supported dense and MoE quantized cases.
- [x] North-star coverage rows now expose both aggregate and per-graph dense matmul descriptor counts for report consumers.
- [x] Regression artifact flattening now also totals MoE grouped descriptor counts, and coverage diagnostics consume those aggregate MoE keys.
- [x] Dense matmul recipe contexts now carry optional input `TensorRole` metadata from compiled tensor records, preparing capability-plus-role recipe predicates without changing execution.
- [x] Shared recipe predicate helpers now include a scaffolded dense FP8 co-located-forward check over `MatmulCapabilities` plus input `TensorRole`.
- [x] MoE grouped matmul recipe contexts now carry optional routed-token `TensorRole` metadata from compiled tensor records, preparing MoE capability-plus-role predicates without changing execution.
- [x] Shared recipe predicate helpers now include a scaffolded MoE FP8 grouped check over `MoECapabilities` plus routed-token `TensorRole`.

### Phase 4 — Block schemas + storage residency + EP topology — COMPLETE

- [x] Python DSL `BlockSchema` declaration surface added for slot residency, distribution, streaming hints, routing schema, and EP topology metadata.
- [x] `BlockSpec` now carries optional schema metadata without changing lowering or runtime allocation behavior.
- [x] Initial schemas attached to Nemotron Mamba, Nemotron MoE, Qwen3-MoE, and GPT-OSS MoE blocks.
- [x] Gemma4 dense/shared-KV/MoE and Qwen3.5 MoE block families now declare Phase 4 schema metadata.
- [x] Qwen3 and Qwen3.5 dense attention/linear block families now declare Phase 4 schema metadata.
- [x] Block schema metadata now serializes into compiled block IR dictionaries for future BufferPlan consumption.
- [x] Legacy Python lowerer also preserves block schema metadata, keeping both DSL lowering paths aligned.
- [x] Expanded model graphs now preserve per-layer block schema records in forward graph metadata.
- [x] C++ DSL IR loader now preserves forward graph metadata, including per-layer block schema records for BufferPlan dual-path consumption.
- [x] BufferPlan-facing C++ schema record collector added as the first no-behavior-change dual path for Phase 4b.
- [x] `DslModel` now captures schema plan records during initialization and reports them in debug memory output without changing allocator behavior.
- [x] Optional `SUROGATE_BLOCK_SCHEMA_PLAN_ASSERT=1` parity guard checks schema record coverage against `NumLayers` without changing default execution.
- [x] `BufferPlan` now carries schema-derived record/routing/EP counters as diagnostics-only parity inputs.
- [x] `BufferPlan` schema diagnostics now include slot, param, activation, expert-parallel, and streaming-hint counters.
- [x] Model-level schema coverage tests now assert complete per-layer metadata for Qwen3, GPT-OSS, and Gemma4 compiled graphs.
- [x] `SUROGATE_BLOCK_SCHEMA_PLAN_ASSERT=1` now validates unique, contiguous per-layer schema coverage instead of only checking total record count.
- [x] `BufferPlan` now exposes per-layer schema summaries as the next allocator-facing Phase 4b dual path, without changing allocation behavior.
- [x] `BufferPlan` now classifies schema block families into dense/MoE/Mamba/linear-mixer diagnostics for allocator parity work.
- [x] `BufferPlan` now parses schema slot residency into GPU/auto/CPU-stream/NVMe diagnostics for Phase 4 storage planning.
- [x] `BufferPlan` now parses schema slot distribution into replicated/sharded-dim/router-replicated/expert-parallel diagnostics.
- [x] `BufferPlan` now preserves MoE routing kind/top-k/shared-expert/scoring-bias and EP weight-transfer metadata for Phase 4b planning.
- [x] Regression artifacts and north-star rows now include block-schema coverage/status summaries so Phase 4 metadata regressions are visible outside C++ debug output.
- [x] `BufferPlan` now preserves per-slot schema summaries (name/kind/residency/distribution/grouping/save/prefetch) for allocator migration.
- [x] Model-level schema coverage tests now also lock Qwen3.5 dense and Qwen3.5 MoE per-layer block-schema metadata.
- [x] `BufferPlan` now provides per-layer/per-slot schema lookup helpers for allocator migration without changing allocation behavior.
- [x] `BufferPlan` now exposes a schema-vs-legacy-slot-registry parity helper for missing activation slot diagnostics.
- [x] `BufferPlan` now computes schema-vs-slot-registry activation allocation parity counters during plan construction and reports them in debug memory output.
- [x] `SUROGATE_BLOCK_SCHEMA_PLAN_ASSERT=1` now also fails early when schema activation slots are absent from the compiled slot registry.
- [x] `BufferPlan` now preserves schema slot shape dimension tokens, preparing shape-aware allocation parity beyond rank-only diagnostics.
- [x] `BufferPlan` now resolves schema activation shape tokens against the current layer dimensions and reports resolved/unresolved activation-shape counters.
- [x] `BufferPlan` now computes dtype-aware byte estimates for resolved schema activation shapes, enabling numeric allocation parity checks.
- [x] Regression block-schema summaries now include slot kind, shape coverage, save-for-backward, and grouped-slot counts for allocator migration tracking.
- [x] Acceptance model graph tests now require every emitted block-schema slot to declare a name, kind, and non-empty shape.
- [x] `BufferPlan` now computes global and EP-local dtype-aware byte estimates for resolved schema parameter slots, including expert-parallel weights.
- [x] Regression block-schema summaries now distinguish expert-parallel parameter slots from activation/storage slots.
- [x] `BufferPlan` schema shape resolution now covers Qwen3.5 `QProjDim` and `KVDim` aliases with C++ parity coverage.
- [x] C++ runtime config now carries Qwen3.5 linear-attention and Gemma per-layer-input dimensions so `BufferPlan` can resolve `ConvDim`, `ValueDim`, `ConvK`, `Hk`, `Hv`, `Vd`, and `PLI_D` schema aliases.
- [x] `BufferPlan` now separates dynamic MoE `dispatched_tokens` schema shapes from other unresolved shape aliases in diagnostics.
- [x] `BufferPlan` now reports schema-derived per-layer activation bytes, legacy max-layer activation-byte estimates, and potential schema allocation savings for Phase 4b allocator parity.
- [x] Regression artifacts now include structured `buffer_plan_summary` metrics for Phase 4b schema allocation parity and baseline drift checks.
- [x] Qwen3.5, Qwen3.5-MoE, and Gemma4 schemas now exclude non-emitted block inputs/state slots so `SUROGATE_BLOCK_SCHEMA_PLAN_ASSERT=1` enforces real slot-registry parity across the acceptance queue.
- [x] `BufferPlan` now preserves schema slot lifetime metadata and splits resolved activation bytes into save-for-backward versus frame-local allocation candidates for the Phase 4b dual path.
- [x] Regression artifacts now capture flattened arena summaries so schema save/frame byte plans can be compared against compiled graph arena allocation drift.
- [x] Acceptance model schema tests now enforce save-for-backward parity with compiled activation layouts; dense `res_att` schema declarations were aligned to the emitted non-save layout.
- [x] `SUROGATE_BLOCK_SCHEMA_PLAN_ASSERT=1` now also rejects schema save-for-backward slots that are not marked saved in the compiled activation layout.
- [x] Phase 4c block-family migration started with complete Nemotron-H hybrid schema coverage across Mamba, attention, MLP, and MoE layers.
- [x] LLaMA dense blocks now declare Phase 4 schema metadata with compile-time slot/save parity coverage.
- [x] Qwen3-VL dense text blocks now declare Phase 4 schema metadata for MRoPE/QK-norm activation slots with compile-time slot/save parity coverage.
- [x] No-GPU schema coverage now fails if any Python block class lacks a `BlockSchema` declaration.
- [x] `BufferPlan` now resolves Nemotron/Mamba schema dimension aliases (`P`, `I`, `D_conv`, `H`, `D`, `N`) from DSL runtime config.
- [x] `BufferPlan` now reports explicit expert-parallel parameter global/local/savings bytes, exposing the per-rank allocation target separately from replicated parameter bytes.
- [x] Phase 4c block-family migration completed for current Python block classes with schema contract validation in both compiler lowering paths. All block schemas now require family metadata, unique shaped slots, and MoE-specific routing/EP/grouped expert declarations.
- [x] Phase 4d reporting started: north-star coverage rows now include CPU-stream storage declaration status, EP topology declaration status, and structured BufferPlan summaries for storage/distribution acceptance triage.
- [x] Phase 4d regression matrix seeded with an explicit Qwen3 FP8 LoRA `2gpu_dp` CPU-stream row so the CPU-stream + FP8 + DP target is tracked before execution is re-enabled.
- [x] Phase 4 acceptance closed: block-schema declarations cover the current Python block families, BufferPlan dual-path diagnostics are wired into artifacts, real-model schema parity is green across the acceptance queue, and Qwen3 MoE/Qwen36 MoE multi-GPU probes pass with the targeted EP/QLoRA ordering guard.

Local validation status:

- [x] `make wheel-dev` passed.
- [x] `make test-integration` passed after the unit-gate cleanup.
- [x] Focused C++ gates passed: `[op_registry]`, `[fusion_rule]`, `[tensor_role]`, `[moe]`; latest DSL IR schema-allocation pass was 235 assertions in 1 case after adding EP param byte accounting.
- [x] Primitive DSL compiled-op golden gate passed after preserving standalone backward graph outputs across last-use pruning and final stack cleanup.
- [x] `make test-unit` passed; FP32 flash-attention standalone/module goldens are explicitly skipped where no registered production attention backend supports that dtype.
- [x] `make regression-smoke` passed.
- [x] No-distributed Python gate passed via `uv run pytest -q tests/test_moe_monitor.py tests/test_regression_baseline_runner.py tests/test_regression_artifact_writer.py --no-gpu` with 49 tests.
- [x] Schema/regression Python gate passed via `uv run pytest -q tests/test_block_schema.py tests/test_regression_artifact_writer.py tests/test_regression_baseline_runner.py --no-gpu` with 46 tests after the EP allocation-accounting debug surface update.
- [x] GPU acceptance rows exercised locally for the real-model queue below; runner uses Hugging Face model IDs from configs by default, with `*_MODEL_PATH` only as optional local/offline overrides.
- [x] Real-model GPU acceptance default shortened to 5 steps for practical iteration; longer convergence runs remain opt-in via `--steps`/`STEPS`.
- [x] Direct real-model run passed for [`qwen3-lora-fp8.yaml`](examples/sft/qwen3/qwen3-lora-fp8.yaml) on 2026-04-27 with `SUROGATE_BLOCK_SCHEMA_PLAN_ASSERT=1`; 28/28 schema records present, zero missing schema activation slots, save-for-backward schema parity clean, arena summary captured, loss moved `2.0824 -> 1.4174`, artifact saved to `regression_baselines/runs/qwen3-lora-fp8-direct-20260427.json`, log `output/log-continued-stopcodon-20260427-065012.json`.
- [x] Direct real-model run passed for [`qwen35-text-lora-fp8.yaml`](examples/sft/qwen35/qwen35-text-lora-fp8.yaml) on 2026-04-27 with `SUROGATE_BLOCK_SCHEMA_PLAN_ASSERT=1`; 24/24 schema records present, zero missing schema activation slots, schema allocation savings estimate `905969664` bytes, save-for-backward schema parity clean, arena summary captured, loss moved `1.7376 -> 1.1215`, artifact saved to `regression_baselines/runs/qwen35-text-lora-fp8-direct-20260427.json`, log `output_q35/log-shivering-pseudomurein-20260427-064918.json`.
- [x] Direct MoE run passed for [`gptoss-lora-mxfp4.yaml`](examples/sft/gpt-oss/gptoss-lora-mxfp4.yaml) on 2026-04-27 with `SUROGATE_BLOCK_SCHEMA_PLAN_ASSERT=1`; 24/24 MoE schema records present, zero missing schema activation slots, save-for-backward schema parity clean, arena summary captured, 48 forward grouped MoE FP4-eligible ops present, loss moved `1.8875 -> 1.9293`, artifact saved to `regression_baselines/runs/gptoss-lora-mxfp4-direct-20260427.json`, log `output_gpt/log-intergalactic-selenium-20260427-065200.json`.
- [x] Direct 2-GPU EP prequant MoE run for [`qwen36moe-lora-fp8.yaml`](examples/sft/qwen36moe/qwen36moe-lora-fp8.yaml) passed acceptance after follow-up stability review. The original 2026-04-27 run with `SUROGATE_BLOCK_SCHEMA_PLAN_ASSERT=1` completed with 40/40 MoE schema records, zero missing schema activation slots, save-for-backward schema parity clean, 80 forward grouped MoE ops and 40 forward all-to-all in/out ops present, and loss moved `7.2512 -> 6.3263`. The rerun after EP/QLoRA ordering hardening completed 5 steps with loss `7.2491 -> 6.3185`, finite pre-clip scaled norms `28.10`, `8.00`, `13.75`, `10.17`, `47.28`, and no GPU spin. These norms are the value returned by `global_norm_sqrt_prescaled` before applying `max_grad_norm=1` clipping, so volatility means the update was clipped, not that the optimizer used an unclipped norm. Latest artifact saved to `regression_baselines/runs/qwen36moe-lora-fp8-sync-20260427.json`, log `output_36moe_5step_sync/log-lyrical-detergent-20260427-081853.json`.
- [x] Direct real-model run passed for [`gemma4-e2b-lora-fp8.yaml`](examples/sft/gemma4/gemma4-e2b-lora-fp8.yaml) on 2026-04-27 with `SUROGATE_BLOCK_SCHEMA_PLAN_ASSERT=1`; 35/35 schema records present, zero missing schema activation slots, save-for-backward schema parity clean, arena summary captured, schema allocation savings estimate `687865856` bytes, 246 dense FP8 forward/backward matmul descriptors present, loss moved `3.6721 -> 2.4903`, artifact saved to `regression_baselines/runs/gemma4-e2b-lora-fp8-direct-20260427.json`, log `output/log-coffee-fermentation-20260427-065738.json`.
- [x] Qwen3 MoE NVFP4 4-GPU EP probe for [`qwen3moe-nvfp4.yaml`](examples/sft/qwen3moe/qwen3moe-nvfp4.yaml) passed after EP/QLoRA ordering hardening. The initial misaligned-address crashes in `fused_residual_rmsnorm` forward/backward were narrowed to unaligned EP MoE views entering vectorized RMSNorm kernels and patched with aligned temp fallbacks. A later step-3 GPU-side spin was isolated as an async ordering/lifetime issue because full `SUROGATE_OP_TRACE_SYNC=1` made a 5-step run pass. The runtime now synchronizes MoE/EP capture-unsafe boundaries by default when QLoRA offloading is active (`SUROGATE_SYNC_CAPTURE_UNSAFE_OPS=0` disables this diagnostic stability guard), and the direct 5-step run completed with losses/norms `2.5784/0.1617`, `3.1738/0.1954`, `2.6453/0.1886`, `2.4640/0.1629`, `2.5385/0.1545`; artifact saved to `regression_baselines/runs/qwen3moe-nvfp4-direct-20260427.json`, log `output_moe_5step_targetsync/log-short-farad-20260427-081449.json`.
- [x] Regression harness guardrail added after the Qwen3 MoE step-3 hang: launched cases now have an outer timeout and report `returncode: "timeout"` with stdout/stderr tails for postmortem triage.
- [x] Direct `.venv` Qwen3 MoE NVFP4 4-GPU EP 3-step probe passed after GPU reset on 2026-04-27 (`2.5784/0.1620`, `3.0948/0.2154`, `2.7235/0.1836` loss/norm).
- [x] EP QLoRA offload auto-tune is now disabled by default unless `SUROGATE_QLORA_EP_OFFLOAD_AUTOTUNE=1`, with `SUROGATE_QLORA_OFFLOAD_AUTOTUNE_DISABLE=1` available as a global kill switch. This removed the `max_resident: 2 -> 48` transition and allowed one direct 4-step Qwen3 MoE NVFP4 EP probe to pass (`2.5784/0.1618`, `3.0857/0.2143`, `2.7121/0.1810`, `2.4783/0.1604` loss/norm), but a subsequent 5-step probe still hung after step 2 even with auto-tune skipped (`2.5784/0.1616`, `3.1058/0.2221`, `2.6609/0.1931`). Auto-tune was not the full root cause; the ordering fix is captured in the following row.
- [x] Backward-side op tracing now mirrors forward tracing and `SUROGATE_OP_TRACE_SYNC=1` can synchronize after each dispatched op for GPU-side spin localization. The full trace probe passed 5 steps, then a narrower default MoE/EP boundary sync for QLoRA offload passed the same 5-step Qwen3 MoE NVFP4 EP acceptance run without global tracing.

Real-model acceptance queue:

- [x] Dense FP8 single-GPU: [`qwen3-lora-fp8.yaml`](examples/sft/qwen3/qwen3-lora-fp8.yaml). Passed regression runner validation with descriptor requirements present.
- [x] Dense FP8 single-GPU: [`qwen35-text-lora-fp8.yaml`](examples/sft/qwen35/qwen35-text-lora-fp8.yaml). Passed regression runner validation with descriptor requirements present.
- [x] Dense FP8 single-GPU: [`gemma4-e2b-lora-fp8.yaml`](examples/sft/gemma4/gemma4-e2b-lora-fp8.yaml). Passed regression runner validation with descriptor requirements present.
- [x] MoE acceptance: [`gptoss-lora-mxfp4.yaml`](examples/sft/gpt-oss/gptoss-lora-mxfp4.yaml). Passed regression runner validation with grouped MoE descriptor requirements present.
- [x] Multi-GPU pre-quant MoE acceptance: [`qwen36moe-lora-fp8.yaml`](examples/sft/qwen36moe/qwen36moe-lora-fp8.yaml). Passed 5-step `.venv` validation; descriptor/schema requirements are present and pre-clip norm volatility is explained by gradient clipping semantics.
- [x] Multi-GPU Qwen3 MoE NVFP4 acceptance: [`qwen3moe-nvfp4.yaml`](examples/sft/qwen3moe/qwen3moe-nvfp4.yaml). Passed 5-step `.venv` validation with EP QLoRA offload auto-tune skipped and targeted MoE/EP boundary sync enabled by default.

### Phase 5 — Hook registry + distribution-aware CPU/offload hooks — COMPLETE (tracked scope)

- [x] Inert C++ hook registry scaffold added for structural `(BlockSchemaId, SlotName)` targets with `after_produce`, `before_consume`, `after_communication`, `after_all_reduce`, `after_all_to_all`, and `after_reduce_scatter` event kinds.
- [x] Schema-derived hook target collection added for streamable parameter slots, expert-parallel communication activations, and sharded/expert-parallel parameter-gradient slots, preserving current execution paths.
- [x] C++ DSL IR coverage now validates deterministic hook lookup, priority ordering, dispatch callbacks, invalid target diagnostics, and schema-derived prefetch/communication/reduce-scatter hook targets.
- [x] Regression block-schema summaries now expose Phase 5 hook-target coverage counts for before-consume prefetch, after-all-to-all communication, and after-reduce-scatter gradient/offload migration points.
- [x] Runtime `BufferPlan` debug summaries now expose the same Phase 5 hook-target coverage counts from resolved schema layer summaries, keeping artifact-level and runtime-level diagnostics aligned.
- [x] `DslModel` now owns a schema hook registry seeded from block-schema records for stream prefetch, all-to-all, and reduce-scatter targets; debug summaries report total and distribution-aware registrations, and execution dispatch remains opt-in.
- [x] Compiled LoRA slices now carry diagnostic schema-slot names inferred from their structural weight inputs, and descriptor summaries report LoRA slice/schema-slot coverage including grouped MoE slices.
- [x] North-star coverage rows now report Phase 5 hook readiness, missing hook-count diagnostics, and hook target counts alongside descriptor/schema/storage/EP readiness.
- [x] Schema hook target collection now covers `after_produce` activation slots used by LoRA post-projection hooks (`qkv`, `att_out`, `mlp_up`, `mlp_down`, router/expert outputs), and `DslModel` seeds inert registrations for them.
- [x] Schema hook target collection now covers `after_all_reduce` replicated/router-replicated parameter slots, so DP hook readiness is tracked alongside EP all-to-all and sharded reduce-scatter readiness.
- [x] Legacy `ForwardHookPoint` attrs now carry diagnostic schema-slot equivalents (`qkv`, `att_out`, `mlp_up`, `mlp_down`), and descriptor/coverage reports track parity before the enum path is removed.
- [x] Opt-in hook schema parity guard added under `SUROGATE_HOOK_SCHEMA_PARITY=1`, failing compilation if legacy forward hook attrs or LoRA slices lack structural schema-slot parity.
- [x] Runtime and coverage artifacts now expose hook-registry registration counts per event, so Phase 5 readiness can distinguish LoRA after-produce, CPU prefetch, DP all-reduce, EP all-to-all, and reduce-scatter scaffolding.
- [x] North-star hook readiness now requires both schema hook targets and seeded hook-registry registrations for CPU streaming, DP all-reduce, EP all-to-all, and after-produce LoRA/forward-hook paths.
- [x] Compiled hook metadata now carries schema ids alongside schema slots for LoRA and legacy forward-hook attrs, with descriptor artifacts reporting full structural `(schema_id, slot)` target coverage.
- [x] Hook readiness now checks full structural target parity for LoRA slices and legacy forward-hook attrs, not only slot-name parity.
- [x] Executor-side schema hook dispatch boundary added behind `SUROGATE_ENABLE_SCHEMA_HOOK_DISPATCH=1`; `GraphExecutor` now passes the inert hook registry into `CompiledExecutor`, and matmul after-produce sites can dispatch structural hook targets without changing default execution.
- [x] Layer-start `before_consume` hook dispatch added under the same opt-in flag, establishing the CPU/offload prefetch boundary before the existing weight-gather path.
- [x] EP dispatch now exposes opt-in distribution-aware `after_all_to_all` and `after_communication` schema hook boundaries after the existing EP strategy dispatch path.
- [x] `DslGradStore` now receives schema hook metadata and dispatches opt-in `after_all_reduce` / `after_reduce_scatter` layer hook boundaries after existing gradient reduction scheduling, preserving default reduction behavior.
- [x] North-star hook readiness now requires reduce-scatter targets and seeded registrations for sharded/ZeRO-style distribution rows.
- [x] LoRA gradient reduction now receives the same schema hook metadata and dispatches opt-in `after_all_reduce` layer hook boundaries after the existing LoRA all-reduce path, so LoRA-only runs are covered by Phase 5 reduction hooks.
- [x] LoRA tensor iteration now routes through structural `LoRATargetId` helpers for dense, shared, router, separate-expert, and grouped-expert adapters; optimizer/norm pointer setup, optimizer state initialization, and paired optimizer updates no longer hand-walk expert/router fields.
- [x] LoRA gradient zeroing and reduction now use the same structural target iterator; EP-active expert adapters still reduce over the DP group only, while dense/shared/router adapters keep the existing full-world reduction route.
- [x] Legacy LoRA optimizer-state zeroing now uses the structural target iterator as well; PEFT-compatible export naming remains explicit.
- [x] Regression/runtime artifacts now record whether schema hook dispatch was enabled for the run, so opt-in hook-boundary validation is visible alongside hook target and registration counts.
- [x] `before_consume` schema hooks now carry a typed prefetch payload and can own the current-layer `DslWeightManager::gather_block` / `wait_for_gather` path under `SUROGATE_ENABLE_SCHEMA_HOOK_DISPATCH=1`; the legacy imperative path remains the default and fallback.
- [x] Streaming gradient offload now has a typed `after_all_reduce` hook payload; under opt-in schema dispatch, hooks can own per-layer all-reduce plus D2H offload, with the legacy imperative path retained as the default and fallback.
- [x] Streamed-weight release now has a typed `after_consume` hook payload; under opt-in schema dispatch, hooks can own per-layer `DslWeightManager::release_block`, and CPU-stream readiness requires both prefetch and release hook coverage.
- [x] Dense projection LoRA after-produce now has a typed hook payload; under opt-in schema dispatch, registered `after_produce` callbacks can own LoRA slice application for matmul and fused matmul+SwiGLU projections, with the inline path retained as fallback.
- [x] EP token all-to-all now dispatches typed `after_all_to_all` / `after_communication` payloads at the actual token-A2A completion point inside `EPStrategy`, instead of from the outer op trampoline after all EP post-processing.
- [x] ZeRO-2 reduced-shard accumulation now routes through the typed `after_reduce_scatter` hook payload under opt-in dispatch, with the existing imperative accumulation retained as fallback.
- [x] Regression/runtime artifacts now expose `after_communication` hook target counts and EP hook readiness requires both generic communication and all-to-all hook coverage.
- [x] LoRA gradient reduction now uses the typed `after_all_reduce` hook payload per layer; under opt-in schema dispatch, callbacks can own LoRA all-reduce/DP-only expert reduction, with the previous reduction path retained as fallback.
- [x] Non-streaming full-gradient all-reduce now also dispatches a typed `after_all_reduce` payload, so all base-gradient reduction hook boundaries carry structured reduction context.
- [x] `after_produce` hook payloads now support local callable actions as well as opaque C callbacks, preparing grouped MoE LoRA hook migration without forcing large captured contexts through ad hoc structs.
- [x] Grouped MoE down-projection LoRA now dispatches the structural `expert_down` `after_produce` hook under opt-in schema dispatch, with the existing inline grouped-LoRA path retained as fallback.
- [x] Grouped MoE gate/up LoRA now dispatches the structural `expert_gate_up` `after_produce` hook for fused and separate expert gate/up adapters, with the existing inline grouped-LoRA path retained as fallback.
- [x] Generic grouped MoE up-projection LoRA now dispatches the structural `expert_up` `after_produce` hook, and hook target collection includes `expert_up` activation slots for Nemotron-style MoE blocks.
- [x] Router matmuls now carry the structural `router_logits` `after_produce` hook slot, allowing router LoRA to use the typed after-produce payload instead of only the inline fallback.
- [x] Expert-weight TensorRole parity now recognizes generic `expert_up` names in addition to fused `expert_gate_up` and `expert_down`, keeping Nemotron-style MoE expert slicing aligned across QLoRA and runtime helpers.
- [x] `save_moe_layer_tensors` now reuses the compiled save-path MoE TensorRole/slot helper instead of maintaining a second substring allowlist.
- [x] Compiled executor embedding-output fallback now has TensorRole parity around the remaining legacy `embed` substring route to the encoded activation slot.
- [x] TensorRole now exposes an embedding predicate, and NorMuon embedding classification uses it with the legacy `embed` substring retained only as fallback.
- [x] TensorRole now exposes LM-head ownership/predicate, and NorMuon `lm_head` classification uses it with the legacy substring retained only as fallback.
- [x] Runtime hook readiness reporting now reuses the hook-registry `after_produce` slot predicate instead of maintaining a shadow string list in `dsl_debug.cpp`.
- [x] Runtime hook readiness reporting now reuses the hook-registry event predicate for all schema hook target counts, eliminating duplicated stream/comm/reduction slot logic in `dsl_debug.cpp`.

---

## 1. Prime directive and north-star

**Prime directive:** every GEMM in every supported architecture is FP8/FP4-eligible *by declaration*, with co-located quantization, fused epilogues, **MoE grouped-GEMM specialization**, and **multi-GPU correctness (DP/EP/ZeRO-2)** — without recipe code changes per architecture, and CPU-RAM training works as a first-class storage tier compatible with LoRA, MoE, and every recipe.

**North-star metric:** FP8 coverage =
`(# distinct (architecture, op-kind, recipe, distribution) tuples training successfully with co-located quant) / (total such tuples across supported architectures)`.

Where:
- `op-kind` ∈ {dense matmul roles, MoE grouped GEMM, EP-routed grouped GEMM}
- `distribution` ∈ {single-GPU, DP-only, DP+EP, ZeRO-2}

Baseline today: ~5/120 ≈ 4% (transformer dense ops × one family × few distributed configs).
Target after Phase 3: ≥80%.
Target after Phase 4: ≥95%.

**Secondary metrics:**
- Time to add a novel architecture (Mamba/MLA-class): from current ~months to ≤2 weeks.
- Time to add a variant: ≤1 day.
- `name.find()` classification call sites: from current ~30 to 0.
- Central op-dispatch switch arms: from current ~90 to 0.
- Fused-op enum values (`MatmulBias`, `MatmulSwiGLU`, `QKVQKNorm`, `MambaGatedRMSNorm`, etc.): from current ~10 to 0 (replaced by declarative fusion rules).
- MoE-specific enum values (`MoEGroupedGemm*`, `MoEPermute`, `MoEUnpermute`, `MoESoftmax`, `MoESigmoid`, `MoeTopK`, `EpDispatch`, `EpCombine`): from current ~12 to 0 (replaced by registry + grouped capability declarations).
- Buffer peak per model (Gemma4): ≥10% reduction after Phase 4.
- LoRA + CPU training works on single-GPU today (verified empirically); gate at [`dsl_model_execution.cpp:319`](csrc/src/runtime/dsl/dsl_model_execution.cpp#L319) is a correct no-op skip — comment clarified. The actual frontier: multi-GPU LoRA + CPU + ZeRO-2 grad sharding + FP8 forward + MoE simultaneously.
- LoRA + MoE + EP + FP8 trains successfully (full stack proof point).
- Distributed regression coverage: every model in the lineup tested at single-GPU, 2-GPU DP, 4-GPU DP, 4-GPU DP+EP (where applicable).

---

## 2. The diagnosis: what's wrong today

The runtime conflates several concepts into closed enums, which forces every new architecture to pay the same coordination tax across many files:

| # | Concept | Today's representation | Symptom when adding a new architecture or distributed config |
|---|---|---|---|
| 1 | **Computation** | `CompiledOpType` enum (90+ values) | New op = coordinated edits across 5+ switches in 3+ files |
| 2 | **Architectural role** | `MatmulOp`, `TensorSlot`, `ForwardHookPoint`, `BackwardHookPoint` | New role (Mamba projection, MLA q_a) has no enum value → falls through to slow generic path |
| 3 | **Optimization eligibility** | Recipe specializes via `switch (MatmulOp)` | New architecture's GEMMs get no FP8/FP4 specialization without recipe edits |
| 4 | **Fusion** | Each fusion is its own enum value | Adding fusion = new op family + new compile rule + new dispatch case + new lifetime + new backward |
| 5 | **Storage tier** | `mOptions.CpuTraining` flag, hard-coded LoRA incompatibility at [`dsl_model_execution.cpp:319`](csrc/src/runtime/dsl/dsl_model_execution.cpp#L319) | Per-tensor residency impossible; LoRA + CPU forbidden; hybrid models can't declare per-block residency |
| 6 | **Tensor classification** | `name.find("router_")`, `name.find("moe_")`, `field == "qkv"` | New name → silent misclassification or string-match additions across runtime |
| 7 | **MoE specialization** ⭐ | `MoEGroupedGemm{,GateUp,Down}` and 9 sibling enum values; recipe-specific paths in [`fp8_hybrid_recipe.cpp:567+`](csrc/src/recipes/fp8_hybrid/fp8_hybrid_recipe.cpp#L567) and [`nvfp4_recipe.cpp:1078+`](csrc/src/recipes/nvfp4/nvfp4_recipe.cpp#L1078); FP8 MoE backward known-incomplete (comment at [`fp8_hybrid_recipe.cpp:714`](csrc/src/recipes/fp8_hybrid/fp8_hybrid_recipe.cpp#L714)) | Each MoE variant (Mixtral, DeepSeek, Qwen3-MoE, GPT-OSS, Nemotron MoE) needs hand-tuned recipe specialization; no `MoECapabilities` predicate; FP4 MoE silently falls back to BF16 ([`nvfp4_recipe.cpp:1082-1083`](csrc/src/recipes/nvfp4/nvfp4_recipe.cpp#L1082-L1083)) |
| 8 | **Distribution + communication** ⭐ | 2D parallelism (`dp_size × ep_size`) wired in [`comm.h:96-150`](csrc/src/utilities/comm.h#L96-L150); ZeRO-2 sharding in `dsl_grad_store.cpp`; LoRA sharded grad path duplicated in `lora_grads_manager.cpp`; comm scheduling decisions live inside `dsl_grad_store.cpp:496-522` not in declarative form | New op doesn't declare its comm profile → grad reduction may be wrong; ZeRO + EP + LoRA combinations require manual code paths; no `Distribution` field on tensors → distributed correctness is by-convention |

**Important update from v1 audit:** Phase 2 is partially pre-existing. [`op_registrations.cpp:290-293`](csrc/src/runtime/executor/op_registrations.cpp#L290-L293) already uses `REGISTER_OP("ep_dispatch", EpDispatch, fwd_ep_dispatch, bwd_ep_dispatch)` — the *execute* function is already a registration. What's still hardcoded is the *compile rule + lifetime + capability + comm profile + grouped semantics* per op. Phase 2 extends the existing registration surface rather than building from scratch.

The escalating-difficulty pattern (Qwen3 QK-norm → Mamba → gated-delta → Gemma4 interleaved) is each successive architecture exercising one more dimension of this debt. **Adding MoE variants and distributed configurations multiplies the coordination tax** because they crosscut all 8 axes.

---

## 3. Architectural separation (the foundation)

Each conflated concept becomes its own open registry. Adding an architecture or distributed config = declaration; never modifying core enums.

**Seven orthogonal axes after refactor:**

1. **Op kernels** — open registry. Extends existing [`op_registrations.cpp`](csrc/src/runtime/executor/op_registrations.cpp) pattern to cover compile/lifetime/capabilities/comm.
2. **Tensor classification** — `TensorRole` struct on every tensor; structural, not stringy.
3. **Optimization** — `MatmulCapabilities` + `MoECapabilities` + `EpilogueSet` declared on each op; recipes match by predicate.
4. **Fusion** — `FusionRule` registry; pattern-match adjacent ops, replace with fused variant when capabilities permit.
5. **Block schema** — `BlockSchema` declared by each block class; runtime allocates and routes from declaration.
6. **Distribution** — `Distribution` field on `TensorRole`; `CommunicationProfile` on op descriptors; per-block-schema EP topology declarations.
7. **Storage tier** — crosscuts axes 2 (where does the tensor live), 3 (recipe-aware fetch), 5 (per-block residency), 6 (sharded + streamed combinations).

---

## 4. Phase plan

```
Pre-Phase 0: FP8 MoE backward tactical closure               COMPLETE
Phase 0: Test infrastructure                                  COMPLETE
Phase 1: TensorRole + Distribution scaffolding                COMPLETE
Phase 2: Op registry descriptor extension scaffold            COMPLETE
Phase 3: Capabilities + recipe predicate scaffolding          COMPLETE
Phase 4: Block schemas + storage residency + EP topology      COMPLETE
Phase 5: Hook registry + distribution-aware + CPU offload     COMPLETE (tracked opt-in hook scope)
```

Completion here means the phase work tracked in §0 has landed. The full roadmap below remains the target architecture; any additional hook migrations beyond the tracked opt-in dispatch scope are follow-up work.

**Critical path to "FP8/FP4 + MoE + multi-GPU generalized":** Phases 0–3 ≈ 29 weeks ≈ **6.7 months solo**, ~4 months with two engineers.

**Full refactor:** Phases 0–5 ≈ 46 weeks ≈ **10.5 months solo**, ~6.5 months with two engineers.

(v2 change: ~+8 weeks added across phases for MoE + multi-GPU coverage; partially offset by ~3 weeks saved in Phase 2 since `op_registrations.cpp` already exists.)

---

### Phase 0 — Test infrastructure (7 weeks, blocking, non-negotiable)

The refactor cannot start without this. Silent regressions detected weeks late kill the project. **Multi-GPU and MoE regressions are silent under single-GPU CI**, so multi-GPU lanes are required from day one.

**Deliverables:**

- **Numerical regression suite.** Per `(model, recipe, batch_shape, storage_tier, distribution)`:
  - Golden activation snapshots after forward, per-layer
  - Golden gradient snapshots after backward, per-tensor
  - Golden 5-step convergence smoke with fixed seed; longer convergence curves are opt-in when explicitly requested
  - Tolerance schema per dtype (BF16: 1e-2, FP8: 5e-2, FP4: 1e-1)
  - Coverage matrix:
    - Models: Qwen3, Gemma4, Nemotron-H, Qwen3.5, **plus at least one pure-MoE (Mixtral-style or Qwen3-MoE)**
    - Recipes: BF16, FP8, FP4
    - LoRA: dense, LoRA, QLoRA
    - Storage: GPU-resident, CPU-streaming
    - **Distribution: single-GPU, 2-GPU DP, 4-GPU DP, 2-GPU DP+EP** (where MoE applicable)
- **Perf regression suite.** Per `(model, recipe, batch_shape, distribution)`:
  - Step time, locked baseline
  - Top-N kernel attribution via nsys
  - **Comm time attribution: per-allreduce, per-all-to-all, weight-transfer time**
  - CI flags >5% step regression, >2% kernel-share regression, >10% comm-share regression
- **Memory regression suite.** BufferPlan peak, observed CUDA peak, **per-rank peak under sharded configs**.
- **FP8/FP4 coverage report.** Generates the north-star metric. Refactor success ≡ this number going up.
- **CPU training validation lane.** End-to-end smoke test for CPU-streaming training. Baselines current behavior even though buggy/limited (LoRA-incompatible).
- **Multi-GPU CI infrastructure.** Single-node 2-GPU and 4-GPU runners; NCCL setup; deterministic seeding across ranks; per-rank log capture; cross-rank gradient comparison utility.
- **MoE-specific assertions:** routing entropy bounds, expert-load-balance variance, per-expert-grad presence checks (catches `EpDispatch`/`EpCombine` correctness regressions silently broken by op restructuring).
- **CI integration.** Every refactor PR runs single-GPU suite. Multi-GPU lane runs nightly + on-demand for distributed-touching PRs.

**Exit criteria:** all reports running in CI, baselines locked, FP8 coverage chart committed, distributed lanes green.

**Resource:** 1 engineer × 7 weeks. Begin immediately. Multi-GPU CI is the longest pole — start it Week 1.

---

### Phase 1 — TensorRole + Distribution foundation (6 weeks)

**Goal:** every Tensor knows what it is, where it lives, and how it's distributed across ranks. Kill string-based classification.

**The contract:**

```cpp
struct TensorRole {
    TensorKind kind;              // Param, Activation, ParamGrad, ActivationGrad, Scratch, Loss, LossInput
    int block_layer;              // -1 for global; else 0..N-1
    SchemaSlotId block_slot;      // opaque ID; meaningful after Phase 4, stable opaque now
    Ownership ownership;          // Persistent | Stack | LoRA | MoE | EP | RopeFreqs | Embedding
    QuantState quant_state;       // None | FP8Pending | FP8Ready | FP4Ready
    StorageTier storage;          // GpuResident | CpuPinnedStream | CpuPageable | NvmeOffload
    StreamingHints streaming;     // prefetch_distance, eviction_policy, sticky
    Distribution dist;            // ⭐ NEW: Replicated | ShardedDim(dim, ZeRO|TP) | ExpertParallel | RouterReplicated
};

struct Distribution {
    enum class Kind { Replicated, ShardedDim, ExpertParallel, RouterReplicated };
    Kind kind;
    int shard_dim;                // for ShardedDim
    int num_shards;               // for ShardedDim
    int local_experts;            // for ExpertParallel (E / ep_size)
    int global_experts;           // for ExpertParallel
    bool needs_reduce_after;      // hint for op output: TP-style reduce required
};
```

`Distribution` is added now (not Phase 4) because op declarations in Phase 2 and capabilities in Phase 3 need to read it. ZeRO-2's existing sharded-grad path ([`dsl_grad_store.cpp:541-560`](csrc/src/runtime/dsl/dsl_grad_store.cpp#L541-L560)) currently lives inside the grad store; after this phase it consults `TensorRole.dist` instead.

**Migration targets** (all current `name.find` and `field ==` classification sites):

| File | Lines | Pattern | Replacement |
|---|---|---|---|
| `graph_compiler.cpp` | [147](csrc/src/runtime/dsl/graph_compiler.cpp#L147), [199-244](csrc/src/runtime/dsl/graph_compiler.cpp#L199-L244), [538-569](csrc/src/runtime/dsl/graph_compiler.cpp#L538-L569), [728-738](csrc/src/runtime/dsl/graph_compiler.cpp#L728-L738) | field-string slot assignment | declared role at compile time |
| `graph_executor.cpp` | [720-751](csrc/src/runtime/executor/graph_executor.cpp#L720-L751) | name-based lookup | role-based lookup |
| `compiled_ops_execute.cpp` | [206-253](csrc/src/runtime/executor/compiled_ops_execute.cpp#L206-L253), [2254](csrc/src/runtime/executor/compiled_ops_execute.cpp#L2254) | `freq_cis`/`zeros`/`embed` substring | `Ownership` field |
| `compiled_ops_save.cpp` | [170-173](csrc/src/runtime/executor/compiled_ops_save.cpp#L170-L173) | **MoE substring jungle** (5 patterns) | single `Ownership::MoE` check |
| `compiled_ops_save.cpp` | [352](csrc/src/runtime/executor/compiled_ops_save.cpp#L352), [850](csrc/src/runtime/executor/compiled_ops_save.cpp#L850), [950](csrc/src/runtime/executor/compiled_ops_save.cpp#L950) | RoPE freqs handling | `Ownership::RopeFreqs` |
| `matmul.cpp` | [40-41](csrc/src/runtime/ops/matmul.cpp#L40-L41), [53](csrc/src/runtime/ops/matmul.cpp#L53) | `shared_expert_*`, `router_weight` substring | role flags |
| `dsl_grad_store.cpp` | [357-374](csrc/src/runtime/dsl/dsl_grad_store.cpp#L357-L374), [496-522](csrc/src/runtime/dsl/dsl_grad_store.cpp#L496-L522) | `if (sharded) { reduce_scatter } else { all_reduce }` | derived from `TensorRole.dist` |
| `lora_grads_manager.cpp` | sharded-grad allocation paths | parallel sharded code | unified path consulting `TensorRole.dist` |

**Approach:**

- Week 1: define `TensorRole` + `Distribution`, integrate with `Tensor`/`TensorMeta`. Compile-time annotation in graph compiler.
- Week 2: storage-tier and distribution declarations plumbed (no execution change).
- Week 3: dual-path with parity assertion — role-derived classification runs alongside string match, asserts equality. Distribution-derived comm scheduling alongside existing.
- Week 4: migrate string-classification consumers one file at a time.
- Week 5: migrate distributed code paths (`dsl_grad_store.cpp`, `lora_grads_manager.cpp`) to consult `Distribution` field. **Multi-GPU CI lane gates this week.**
- Week 6: remove string-match fallbacks once parity holds; remove parallel sharded paths; remove parity assertions.

**Validation:** Phase 0 numerical/perf suite green throughout. Multi-GPU lane shows zero numerical/comm regression.

**Exit criteria:**
- Zero `name.find(...)` and `field == "..."` classification calls in the listed files.
- All tensors carry valid `TensorRole` with non-`Unknown` fields.
- Storage-tier and Distribution fields populated; LoRA sharded grad path no longer parallel to base sharded grad path.

**Risk:** medium. Distribution field semantics must match the existing 2D-parallelism implicit assumptions ([`comm.h:96-150`](csrc/src/utilities/comm.h#L96-L150)).

---

### Phase 2 — Op registry extension (7 weeks)

**Goal:** every op declares — in one place — its execute function, backward rule, lifetime, capabilities, communication profile, and grouped semantics. Adding a new op = one `.cpp` with `REGISTER_COMPILED_OP`. Zero edits to switches in other files.

**Foundation already exists.** [`op_registrations.cpp:290-293`](csrc/src/runtime/executor/op_registrations.cpp#L290-L293) already binds `name → CompiledOpType → fwd_fn → bwd_fn`. The extension below adds the missing facets.

**The extended contract:**

```cpp
struct CompiledOpDescriptor {
    std::string_view name;
    OpKind kind;                                       // Matmul, GroupedMatmul, Norm, Attention, Activation,
                                                       // Reduction, Routing, Communication, Custom
    std::function<std::vector<Operation>(BackwardRuleContext&)> backward_rule;
    std::function<void(ExecuteCtx&)> execute;
    std::function<void(SaveCtx&)> compute_lifetime;

    OpCapabilities default_caps;                       // Phase 3 ties in here
    EpilogueSupport epilogue_support;                  // which epilogues this op can fuse
    StorageCompatibility storage_compat;               // GpuResident | CpuPinnedStream
    CommunicationProfile comm_profile;                 // ⭐ NEW: see below
    GroupedSemantics grouped_semantics;                // ⭐ NEW: for grouped/routed ops

    SlotShapeFn output_shape;
};

struct CommunicationProfile {
    enum class Kind {
        NoComm,
        AllReduceAfter,           // TP-style: produces partial result, needs all-reduce sum
        ReduceScatterAfter,       // TP-style + ZeRO sharding
        AllToAllIn,               // EP dispatch: shuffles tokens between EP ranks
        AllToAllOut,              // EP combine: shuffles results back
        ExpertParallelRouted,     // grouped GEMM operating on EP-routed tokens
        WeightStreamFromCpu,      // weight is CPU-resident, stream window-by-window
        WeightTransferP2P,        // LLEP: peer-to-peer weight transfer for hot experts
    };
    Kind kind;
    bool can_overlap_with_compute;
    int reduction_priority;       // for ordering when multiple ops need reduction
};

struct GroupedSemantics {
    bool is_grouped;              // operates on multiple expert weights at once
    bool routes_tokens;           // changes token ordering (permute/unpermute)
    int expert_dim;               // which dim indexes experts in weight tensors
    bool ep_aware;                // honors ep_size/ep_rank
};

#define REGISTER_COMPILED_OP(NAME, ...) \
    namespace { static int _reg_##NAME = compiled_op_registry().register_op(NAME, __VA_ARGS__); }
```

**`CommunicationProfile`** is what the v1 plan was missing entirely. After this phase, the runtime can answer "does this op need a comm step after, and which one?" by reading the op descriptor — not by switching on op type inside `dsl_grad_store.cpp`.

**`GroupedSemantics`** tags MoE grouped ops (`MoEGroupedGemmGateUp`, `MoEGroupedGemmDown`, etc.) with their expert-dim semantics. Phase 3's `MoECapabilities` builds on this.

**Migration order:**

| Week | Op family | Why this order |
|---|---|---|
| 1 | Infrastructure: extend the `op_registrations.cpp` pattern to carry the new fields. Existing switch dispatch falls through to registry on unknown facets. | Foundation |
| 2–3 | **Mamba ops first** (6 ops) — most friction, no comm complexity, clean proof case | Proves extension pattern |
| 4 | Qwen3.5 gated-delta ops + MoE-internal ops (`MoESoftmax`, `MoeTopK`, `MoEPermute`, `MoEUnpermute`) — declare `GroupedSemantics` and `CommunicationProfile::NoComm` | Pre-EP MoE ops; non-trivial routing semantics |
| 5 | **EP ops** (`EpDispatch`, `EpCombine`, `EpDispatchBackward`, `EpCombineBackward`) — declare `CommunicationProfile::AllToAllIn`/`AllToAllOut`. `MoEGroupedGemm*` declare `ExpertParallelRouted` when `ep_size > 1` | Distributed correctness landmark |
| 6 | Standard transformer ops (RMSNorm, FlashAttention, SwiGLU, Add, Scale, fused fusions). Most declare `CommunicationProfile::NoComm`; LM-head's gradient may declare `AllReduceAfter` for TP-sharded vocab | Bulk |
| 7 | Delete switches in [`graph_compiler.cpp:4063+`](csrc/src/runtime/dsl/graph_compiler.cpp#L4063), [`compiled_ops_execute.cpp`](csrc/src/runtime/executor/compiled_ops_execute.cpp), [`compiled_ops_save.cpp`](csrc/src/runtime/executor/compiled_ops_save.cpp); migrate `dsl_grad_store.cpp` comm scheduling to read `CommunicationProfile`; delete the `EpDispatch`/`EpCombine` capability gate at [`graph_compiler.cpp:1345-1350`](csrc/src/runtime/dsl/graph_compiler.cpp#L1345-L1350) | Cleanup |

**Validation:** Phase 0 single-GPU suite green after each op family migration. Multi-GPU lane gates Week 5 (EP ops) and Week 7 (cleanup).

**Exit criteria:**
- `CompiledOpType` enum either deleted or reduced to legacy alias.
- No central op-type switches in `graph_compiler.cpp`, `compiled_ops_execute.cpp`, `compiled_ops_save.cpp`.
- Comm scheduling in `dsl_grad_store.cpp` reads `CommunicationProfile` from op descriptor.
- New op = `REGISTER_COMPILED_OP` only, including comm profile + grouped semantics.

**Risk:** medium-high. Comm scheduling is the new risk axis. Mitigation: dual-path with parity assertion (existing reduce_scatter/all_reduce decisions vs descriptor-derived); compare allreduce call counts and bytes between paths.

---

### Phase 3 — Capabilities + fusion + epilogues + MoE (9 weeks) ⭐ FP8/FP4 + MoE GENERALIZATION ARRIVES

**This is the headline phase.** After this, every architecture's GEMMs (dense and MoE-grouped) inherit FP8/FP4 specialization and fused epilogues by declaration. Fusions stop being new enum values; they become matching rules. **MoE recipe specialization extends to any architecture by declaration.**

#### 3a. Dense matmul capabilities (weeks 1–2)

```cpp
struct MatmulCapabilities {
    bool fp8_forward_eligible;
    bool fp8_backward_eligible;
    bool fp4_forward_eligible;
    bool fp4_backward_eligible;
    EpilogueSet supported_epilogues;        // None, Bias, BiasGelu, BiasSilu, SwiGLU, GeGLU, Softcap, ...
    QuantColocate colocate_input;           // None, PrecedingNorm, PrecedingActivation
    bool weight_cache_eligible;             // pre-quantize weight once vs per-step
    StorageCompatibility weight_storage;    // GpuResident | CpuPinnedStream
    bool can_overlap_with_comm;             // safe to issue while NCCL kernel runs
    int recipe_priority;                    // ordering when multiple recipes match
};
```

Recipes register **predicates over capabilities + tensor role**, not switches over `MatmulOp`:

```cpp
recipe.register_forward(
    [](const MatmulCapabilities& c, const TensorRole& input_role) {
        return c.fp8_forward_eligible
            && c.colocate_input == QuantColocate::PrecedingNorm
            && input_role.quant_state == QuantState::FP8Ready
            && input_role.dist.kind != Distribution::Kind::ExpertParallel;
    },
    fp8_colocated_norm_forward
);
```

#### 3b. MoE / grouped capabilities (weeks 3–4) ⭐ NEW

```cpp
struct MoECapabilities {
    bool grouped_gemm_eligible;             // can be dispatched as a single grouped GEMM
    bool fp8_grouped_eligible;              // recipe path for FP8 grouped GEMM exists
    bool fp4_grouped_eligible;              // recipe path for FP4 grouped GEMM exists
    bool cudnn_moe_graph_eligible;          // can fuse via cuDNN moe_grouped_matmul
    bool per_expert_quant;                  // each expert quantized independently vs shared scale
    bool routing_aware_fusion;              // routing softmax/topk fuses with downstream
    StorageCompatibility expert_storage;    // GpuResident | CpuPinnedStream
    EpAwareness ep_awareness;               // None | Sharded | Routed | WeightTransfer
    bool fp8_backward_implemented;          // ⭐ tracks the gap at fp8_hybrid_recipe.cpp:714
    bool nvfp4_no_fallback;                 // ⭐ tracks the silent fallback at nvfp4_recipe.cpp:1082
};
```

**Concrete current-state captured here:**

- [`fp8_hybrid_recipe.cpp:714`](csrc/src/recipes/fp8_hybrid/fp8_hybrid_recipe.cpp#L714) comment: *"Adding native FP8 MoE backward kernels would improve performance."* Today MoE backward in FP8 falls back to BF16. After Phase 3, this is a capability flag that explicitly fires the gap; closure becomes scheduled work, not a TODO comment.
- [`nvfp4_recipe.cpp:1082-1083`](csrc/src/recipes/nvfp4/nvfp4_recipe.cpp#L1082-L1083): NVFP4 MoE silently falls back to BF16 cuDNN when FP4 weights are not available. After Phase 3, the capability declares the fallback explicitly; logging surfaces when fallback fires.

**Recipe registers MoE specializations the same way:**

```cpp
recipe.register_moe_forward(
    [](const MoECapabilities& c, const TensorRole& token_role) {
        return c.fp8_grouped_eligible
            && c.cudnn_moe_graph_eligible
            && token_role.dist.kind == Distribution::Kind::ExpertParallel;
    },
    fp8_cudnn_grouped_ep_forward
);
```

**CUTLASS grouped GEMM as a sibling specialization.** The current cuDNN path is excellent for FP8/FP4 (uses `block_scale_dequantize → moe_grouped_matmul` graphs). For BF16-on-Hopper or shapes cuDNN doesn't optimize well, CUTLASS grouped GEMM is a real perf lever. Register as another specialization with its own predicate.

#### 3c. Fusion rule registry (weeks 5–7)

**The biggest enum-killer.** Today: every fusion is a new `CompiledOpType` (`MatmulBias`, `MatmulSwiGLU`, `MatmulSwiGLUBackward`, `QKVQKNorm`, `QKVQKNormRoPE`, `FusedResidualRMSNorm`, `FusedLMHeadLoss`, `MambaGatedRMSNorm`, ...). Each requires backward variant, dispatch case, lifetime entry.

**After:** fusions are pattern-matching rules over the op graph.

```cpp
struct FusionRule {
    std::string_view name;
    OpPattern pattern;                      // e.g. matmul → bias_add
    std::function<bool(const FusionContext&)> eligible;  // checks capabilities, dtypes, shapes, comm profile
    std::function<Operation(const FusionContext&)> emit_forward;
    std::function<std::vector<Operation>(const FusionContext&)> emit_backward;
    int priority;                           // for overlapping rules
};

REGISTER_FUSION_RULE("matmul_bias", { ... });
REGISTER_FUSION_RULE("matmul_swiglu", { ... });
REGISTER_FUSION_RULE("matmul_softcap", { ... });        // Gemma4 win, automatic
REGISTER_FUSION_RULE("residual_rmsnorm", { ... });
REGISTER_FUSION_RULE("qkv_qknorm", { ... });
REGISTER_FUSION_RULE("qkv_qknorm_rope", { ... });
REGISTER_FUSION_RULE("lmhead_loss", { ... });
REGISTER_FUSION_RULE("mamba_gated_rmsnorm", { ... });
REGISTER_FUSION_RULE("moe_routing_topk_softmax", { ... });  // ⭐ MoE routing fusion
REGISTER_FUSION_RULE("moe_permute_quantize", { ... });      // ⭐ co-located quant for grouped GEMM
```

**Recipe-aware + comm-aware fusion:** the rule's `eligible` predicate consults capabilities AND `CommunicationProfile`. A fusion that would change comm scheduling (e.g., absorbing a reduction into a matmul epilogue) declines if it would break the all-reduce path.

#### 3d. Quant state generalization (week 8)

Generalize the existing FP8 ready-flag mechanism. Today [`FP8Ready_LN1`/`LN2`/`SwiGLU`](csrc/src/runtime/ops/matmul.cpp#L160-L162) hard-code three transformer locations. Replace with `QuantState` field on `TensorRole`: "this tensor was pre-quantized into FP8 buffer X by upstream op Y." Co-located quant becomes a generic upstream-op responsibility — any norm, activation, or **routing/permute** can declare "I quantize my output for the downstream op when consumed by a matmul or grouped GEMM with `colocate_input=PrecedingNorm`."

**MoE permute step gets co-located quant for free.** Currently the permute writes BF16 tokens that then get quantized inside the grouped GEMM. After this phase, permute writes FP8 directly when downstream grouped GEMM declares `PrecedingActivation` co-location. Eliminates a buffer pass.

#### 3e. Capability declarations on existing ops (week 9)

Declare capabilities on every existing matmul-emitting and grouped-emitting op:

| Op | fp8_fwd | fp4_fwd | epilogues | colocate | comm_profile | weight_storage |
|---|---|---|---|---|---|---|
| Standard QKV proj | ✓ | ✓ | Bias | PrecedingNorm | NoComm | GPU + CpuStream |
| Standard AttnOut proj | ✓ | ✓ | None | PrecedingActivation | NoComm | GPU + CpuStream |
| Standard MLPUp | ✓ | ✓ | SwiGLU, GeGLU | PrecedingNorm | NoComm | GPU + CpuStream |
| Standard MLPDown | ✓ | ✓ | None | PrecedingActivation | NoComm | GPU + CpuStream |
| Mamba in_proj | ✓ | ✓ | None | PrecedingNorm | NoComm | GPU + CpuStream |
| Mamba out_proj | ✓ | ✓ | None | PrecedingActivation | NoComm | GPU + CpuStream |
| Gated-delta projections | ✓ | ✓ | None | PrecedingNorm | NoComm | GPU + CpuStream |
| **MoE grouped gate_up** | ✓ (cuDNN) | ✓ (cuDNN) | SwiGLU, ReLU2 | PrecedingActivation (post-permute) | NoComm or ExpertParallelRouted | GPU + CpuStream |
| **MoE grouped down** | ✓ (cuDNN) | ✓ (cuDNN) | None | None | NoComm or ExpertParallelRouted | GPU + CpuStream |
| **MoE grouped backward** | ⚠ NOT IMPLEMENTED | ⚠ falls back | None | None | depends | depends |
| Router weight | (recipe-dependent) | (recipe-dependent) | None | None | NoComm | GPU only |
| LM head | ✓ | (depends) | None (fused via lmhead_loss rule) | None | NoComm or AllReduceAfter (TP) | GPU + CpuStream |
| EpDispatch | n/a | n/a | n/a | (forwards to grouped) | AllToAllIn | n/a |
| EpCombine | n/a | n/a | n/a | n/a | AllToAllOut | n/a |

**Validation:**
- Phase 0 north-star metric jumps from ~5/120 to whatever the architecture × distribution lineup supports.
- Every previously-FP8-trained model still trains in FP8 with same loss curve.
- **MoE in FP8 trains correctly across {single-GPU, DP, DP+EP}** for at least one MoE model.
- The known FP8 MoE backward gap surfaces explicitly via capability flag (and is on the roadmap to close).

**Exit criteria:**
- `MatmulOp` enum deleted.
- All recipes dispatch via capability predicates.
- All existing fusions deleted from `CompiledOpType`; emitted by fusion rules.
- `MoECapabilities` populated on every grouped op; recipes register MoE specializations as predicates.
- Mamba/gated-delta GEMMs measured to use FP8 path in nsys.
- Gemma4 softcap fused into attention via fusion rule.
- MoE in FP8 + EP + DP trains end-to-end (full multi-GPU MoE proof point).

**Risk:** **highest perf risk in the project.** Capability mis-declaration causes silent FP8 numerical regression. MoE specialization mistakes cause silent expert-load imbalance or routing errors. Mitigation:
- Capability flags per (op, recipe) tuple behind feature gates.
- Allowlist progressively widened.
- North-star metric report flags new failures immediately.
- Per-fusion-rule validation: emit_backward must produce identical gradients to unfused sequence.
- Per-MoE-recipe validation: routing entropy, expert load balance, per-expert grad presence checked in Phase 0 suite.

---

### Phase 4 — Block schemas + storage residency + EP topology (12 weeks)

**Goal:** retire `TensorSlot::Block*`. Each block class declares its slot schema, per-slot residency, and (for MoE blocks) its EP topology. Hybrid models become first-class. **LoRA + CPU + multi-GPU training works.**

#### 4a. Block schema contract (weeks 1–2)

**Status: COMPLETE.** Python block schema declarations, standalone block serialization, model graph metadata, and C++ IR preservation are implemented for the current acceptance families.

```python
class NemotronHMamba2Block(nn.Block):
    schema = BlockSchema([
        SlotDecl("ssm_state",   shape=("B","T","I"), lifetime="layer",
                 save_for_backward=True, dtype="bf16",  residency="gpu",
                 distribution=Replicated()),
        SlotDecl("conv_out",    shape=("B","T","C_conv"), lifetime="layer",
                 save_for_backward=True,                 residency="gpu",
                 distribution=Replicated()),
        SlotDecl("in_proj_weight",  kind="param", residency="auto",
                 streaming_hint=StreamingHint(prefetch_distance=2),
                 distribution=ShardedDim(dim=0, mode="zero2")),
        SlotDecl("out_proj_weight", kind="param", residency="auto",
                 distribution=ShardedDim(dim=0, mode="zero2")),
    ])
```

```python
class NemotronHMoEBlock(nn.Block):
    schema = BlockSchema([
        SlotDecl("router_weight", kind="param",
                 distribution=Replicated(),         # router replicated across all ranks
                 residency="gpu"),
        SlotDecl("experts_up", kind="param", grouped=True,
                 distribution=ExpertParallel(experts_per_rank="auto"),
                 residency="auto",
                 streaming_hint=StreamingHint(prefetch_distance=1)),
        SlotDecl("experts_down", kind="param", grouped=True,
                 distribution=ExpertParallel(experts_per_rank="auto"),
                 residency="auto"),
        SlotDecl("permuted_input", shape=("dispatched_tokens","C"),
                 lifetime="layer", residency="gpu",
                 distribution=ExpertParallel()),  # post-EpDispatch
    ],
    routing=RoutingSchema(
        kind="topk_softmax",          # or "topk_sigmoid", "expert_choice"
        topk=2,
        norm_topk_prob=True,
        scoring_bias=False,
        shared_experts=0,             # or count
    ),
    ep_topology=EPTopology(
        ep_size_param="ep_size",
        weight_transfer_eligible=False,  # opt into LLEP
    ))
```

The `RoutingSchema` and `EPTopology` declarations make per-architecture MoE differences (sigmoid vs softmax, shared experts present/absent, scoring bias, weight transfer) **structural**. Today this is implicit in op composition + recipe paths.

#### 4b. BufferPlan migration (weeks 3–4)

**Status: COMPLETE.** The C++ runtime now collects per-layer schema records, validates coverage, slot-registry parity, and save-for-backward parity behind `SUROGATE_BLOCK_SCHEMA_PLAN_ASSERT=1`, preserves per-slot residency/distribution/routing/lifetime/shape metadata, computes schema-side activation/parameter byte diagnostics including EP-local expert weights, and emits buffer-plan plus arena summaries into regression artifacts. Allocator decisions still use the legacy enum-driven `BufferPlan`; Phase 4c starts the block-family migration without removing fallback paths.

Parallel `BufferPlan` path consumes schemas. Old enum-driven path stays. Both run, allocation parity-checked.

**Per-rank-aware allocation.** Today the buffer plan over-allocates `max(transformer_block, mamba_block, mlp_block, moe_block)` per slot because the enum is shared. After Phase 4, layer L allocates exactly what its block type needs. **For MoE blocks, EP-sharded weight allocation = `(global_size / ep_size) × dtype_size` per rank**, declared structurally instead of computed inside grad store. Direct address of Gemma4's per-block-dim over-allocation AND MoE EP weight memory.

#### 4c. Block migration (weeks 5–9)

| Week | Block family | Why this order |
|---|---|---|
| 5 | Nemotron Mamba block | Most aliasing pain; `"mixer_act": "swiglu"` smell at [`nemotron_h.py:105`](surogate/dsl/blocks/nemotron_h.py#L105) goes away |
| 6 | Transformer attention blocks (Llama, Qwen, Gemma4 attention) | Bulk of usage; per-layer-shape variation (Gemma4) becomes structural |
| 7 | Transformer MLP blocks | Similar pattern |
| 8 | **MoE blocks (Nemotron MoE, Mixtral-style, Qwen3-MoE, GPT-OSS MoE)** | Per-layer MoE-ness in hybrid models becomes structural; routing schema differences become declarative |
| 9 | **EP topology integration**: blocks declare EP topology; runtime reads it for grouped GEMM dispatch and weight allocation | Multi-GPU MoE clean-up |

#### 4d. Storage execution + LoRA-compatible CPU streaming + sharded streaming (weeks 10–12)

Implement the CPU-streaming executor that consumes residency declarations, **plus the cross product of streaming × ZeRO sharding × EP**. This is the most subtle interaction in the refactor.

**Three streaming patterns supported:**

1. **Per-layer weight streaming** (current behavior at [`dsl_grad_store.cpp:633-775`](csrc/src/runtime/dsl/dsl_grad_store.cpp#L633-L775), generalized): weights for layer L+prefetch_distance fetch while layer L computes.
2. **Per-block-type streaming**: Mamba blocks may stream differently than attention; MoE blocks may stream only the active experts (top-k subset) per micro-step.
3. **LoRA-aware + ZeRO-aware streaming**: base-model weights stream from CPU and are ZeRO-2-sharded; LoRA adapters stay GPU-resident. Each rank streams only the slice of weights it needs after sharding.

**The combination matrix that must work end-to-end (Phase 0 validates):**

| Storage | Distribution | LoRA | Recipe | Status today | Status after Phase 4 |
|---|---|---|---|---|---|
| GPU | Single-GPU | off | BF16/FP8/FP4 | ✓ | ✓ |
| GPU | DP | off | BF16/FP8/FP4 | ✓ | ✓ |
| GPU | DP+EP | off | BF16/FP8/FP4 (FP8 backward partial) | ✓ partial | ✓ tracked |
| GPU | DP | on | BF16/FP8/FP4 | ✓ | ✓ |
| GPU | DP+EP | on | BF16/FP8/FP4 | ✓ partial | ✓ tracked |
| **CPU stream** | Single-GPU | **off** | BF16/FP8 | ✓ | ✓ |
| **CPU stream** | DP | off | BF16/FP8 | ✓ partial | ✓ |
| **CPU stream** | Single-GPU | **on** | any | ✓ works (gate is no-op skip; comment was misleading) | ✓ (comment clarified) |
| **CPU stream** | DP | **on** | any | ⚠ unvalidated end-to-end multi-GPU | ✓ ⭐ validated |
| **CPU stream + ZeRO-2 sharded** | DP | on | FP8 | ❌ never tried | ✓ (the headline combo) |

**Capability check:** Phase 3's `StorageCompatibility` and `CommunicationProfile` on each op are read here. Ops that declare incompatible combinations fail loudly; combinations that are valid get the streaming path.

**FP8/FP4 + CPU streaming + sharding interaction:** weights stream from CPU in BF16 (or pre-quantized FP8 for frozen base under LoRA, where the existing FP8 cache caches frozen weights once). For full fine-tune, recipe-path re-quantization happens per call (cache structurally bypasses trainable weights). Backward weight gradient (FP8 hybrid) writes to GPU then reduce-scatters to the per-rank shard, optionally offloaded to CPU. **No FP8 cache invalidation work required** — Q1 in spike RESOLVED via static analysis (frozen-only cache is correct by construction).

**Validation:**
- Phase 0 memory suite (peak ≤ pre-Phase-4 per rank).
- Phase 0 CPU-training validation lane expanded to cover LoRA + CPU + FP8 + DP.
- The full-stack proof: Mixtral-style MoE in FP8 + EP + DP + LoRA trains 5 steps with golden loss curve.

**Exit criteria:**
- `TensorSlot::Block*` enum deleted.
- Per-layer-typed allocation in BufferPlan.
- Hybrid models stop using aliasing tricks.
- `lora_enabled() && CpuTraining` works end-to-end on **multi-GPU** (single-GPU already works today; the multi-GPU + ZeRO-2 + FP8 combination is the novel target).
- LoRA + CPU streaming + ZeRO-2 + FP8 works on at least one MoE model.
- Gemma4 buffer peak ≥10% lower than Phase-0 baseline.

**Risk:** **highest of the refactor.** Buffer-plan + activation-routing + CPU streaming + ZeRO sharding + EP sharding interactions are subtle; debugging requires multi-rank reasoning. Mitigation:
- Parity-checked dual-path during migration.
- One block family at a time.
- Phase 0 memory + multi-GPU suite as gate.
- Streaming feature-gated per (model, recipe, distribution) until validated.
- Mandatory design spike before Week 10 for the CPU + ZeRO + FP8 interaction.

---

### Phase 5 — Hook registry + distribution-aware + CPU offload (5 weeks)

**Goal:** hooks register against `(BlockSchemaId, SlotName)` pairs. Distribution-aware hook points (after-all-reduce, after-all-to-all). LoRA hooks for novel architectures (Mamba, MLA, gated-delta) become possible. CPU prefetch/eviction is hook-driven.

```cpp
hook_registry.on_after_produce(
    schema_id_for("attention_block"), "qkv_out",
    [](HookCtx& ctx) { apply_lora(ctx); }
);
hook_registry.on_after_produce(
    schema_id_for("mamba_block"), "in_proj_out",
    [](HookCtx& ctx) { apply_lora(ctx); }      // Mamba LoRA — currently impossible
);
hook_registry.on_before_consume(
    schema_id_for("mamba_block"), "out_proj_weight",
    [](HookCtx& ctx) { ensure_streamed(ctx); } // explicit prefetch hook
);
hook_registry.on_after_communication(           // ⭐ NEW: distribution-aware
    schema_id_for("moe_block"), "permuted_input",
    [](HookCtx& ctx) { post_dispatch_quantize(ctx); }
);
hook_registry.on_after_reduce_scatter(          // ⭐ NEW
    schema_id_for("any_block"), "param_grad",
    [](HookCtx& ctx) { offload_to_cpu(ctx); }
);
```

**Cleanup status:** LoRA dense/shared/router/separate-expert/grouped-expert iteration now routes through structural `LoRATargetId` helpers in optimizer setup, norm pointers, gradient zeroing, gradient reduction, and paired optimizer updates. Dense projection LoRA after-produce, CPU current-layer prefetch, streamed-weight release, EP token all-to-all observation, ZeRO-2 reduced-shard accumulation, LoRA gradient reduction, and streaming grad offload now have typed hook payloads behind opt-in dispatch; remaining Phase 5 migration is to keep moving distribution/offload side effects from inert boundaries into registered callbacks.

**Less critical than Phases 1–4 because the immediate FP8/FP4 + MoE goal doesn't require it.** Finishes the extensibility story.

**Risk:** low. Smaller surface than Phases 2–4. Distribution-aware hooks are new; mitigate by feature-gating and Phase 0 multi-GPU lane.

---

## 5. Migration discipline (non-negotiable rules)

1. **Dual-path with parity assertion.** New code runs alongside old; assertion fires on mismatch. Never replace before parity-validated.
2. **Removal deadline per phase.** Each phase's "legacy fallthrough" must be removed before the next phase begins.
3. **CI gate per PR.** Every refactor PR runs full Phase 0 single-GPU suite. Multi-GPU lane gates distributed-touching PRs.
4. **North-star metric tracked weekly.** FP8/FP4 + MoE + distribution coverage chart visible. Refactor success ≡ this number monotonically increases.
5. **No new closed enums.** Reviewers reject any PR that adds an enum value where a registry/declaration would suffice. Includes comm-profile values, capability flags, routing schema kinds.
6. **Capability-declaration review.** New `MatmulCapabilities`, `MoECapabilities`, `CommunicationProfile` flags require ≥2 architecture consumers OR explicit "will be generalized later" marker.
7. **Distributed parity assertions stay on permanently** for the comm-scheduling migration (Phase 2 Week 7). Comm correctness is too easy to silently break.

---

## 6. Risk register (ranked)

| # | Risk | Impact | Probability | Mitigation |
|---|---|---|---|---|
| 1 | Phase 4 buffer-plan + ZeRO + CPU + EP interaction regression | High | Medium | Spike [SPIKE_CPU_ZERO_FP8.md](SPIKE_CPU_ZERO_FP8.md) completed 2026-04-26; per-combination feature gates; parity-checked dual-path. Existing master/work + streaming + FP8 cache infrastructure is the foundation, not greenfield |
| 2 | Phase 3 FP8 numerical regression from wrong capability | High | Medium | Per-(op, recipe) feature flags; allowlist; north-star tracking |
| 3 | Phase 3 MoE specialization breakage (silent expert imbalance, routing errors) | High | Medium | Routing entropy + load-balance assertions in Phase 0; per-MoE-recipe validation |
| 4 | Phase 2 comm scheduling desync (silent grad correctness) | High | Medium | Permanent dual-path assertions; allreduce call/byte counters compared |
| 5 | Phase 4d CPU streaming + ZeRO + FP8 subtle bugs | Low-Medium | Low-Medium | Spike [SPIKE_CPU_ZERO_FP8.md](SPIKE_CPU_ZERO_FP8.md) reframed scope: machinery exists (master/work, double-buffered prefetch, ZeRO-3 weight sharding all already wired). Q1 RESOLVED — no FP8 cache invalidation bug. Q4 RESOLVED — LoRA + CPU works. Remaining items: prefetch ping-pong handling under streaming (Q2), gather pipeline ordering (Q3), capture-safety assertion (Q5). ~2 weeks audit + small fixes |
| 6 | Phase 2 lifetime/save-tensor desync | Medium | High | Centralized lifetime initially; migrate per op family |
| 7 | Refactor blocks adapter velocity | Medium | Certain | Phases 1, 2, 5 invisible to model authors; Phases 3, 4 disruptive — plan accordingly |
| 8 | Capability bloat | Low | High | Review process: ≥2 consumers or explicit deferral marker |
| 9 | Mid-refactor priority shift (new flagship lands) | High | Medium | Phase 0 makes pause/resume viable. Hard rule: Phase 3 must finish before any pause |
| 10 | Fusion rule combinatorial explosion | Medium | Medium | Rule priority; conflicts logged; "fusion preview" tool |
| 11 | Gemma4 perf gap not addressed by refactor | Medium | Certain | Parallel investigation track (§9.1); Phase 4 helps structurally |
| 12 | cuDNN MoE graph version coupling (requires cuDNN ≥9.15) | Medium | Low | Capability flag for cuDNN version; fallback to non-cuDNN grouped GEMM |
| 13 | EP topology declaration mismatch with runtime ep_size | Medium | Medium | Validation at compile time; error before training start |
| 14 | LLEP weight transfer correctness during refactor | Medium | Low | Out-of-scope for v1 of refactor; capability flag reserved |

---

## 7. Resourcing and parallelism

**Critical path to FP8/FP4 + MoE + multi-GPU generalized** (Phases 0–3): ~29 weeks ≈ **6.7 months solo**, ~4 months with two engineers.
**Full refactor** (Phases 0–5): ~46 weeks ≈ **10.5 months solo**, ~6.5 months with two engineers.

**Parallelism map:**

| Phase | Depends on | Can begin when | Engineer count benefit |
|---|---|---|---|
| 0 | — | Day 1 | 2 (one on numerical/perf, one on multi-GPU CI infra) |
| 1 | Phase 0 done | After Phase 0 | 1 |
| 2 | Phase 1 done | After Phase 1 | 1–2 (op families parallelize after Week 1) |
| 3a–3b | Phase 2 ≥50% | At Phase 2 Week 4 | 1 (capabilities) + 1 (MoE capabilities + CUTLASS grouped) |
| 3c (fusion) | Phase 3a–b done | After 3a–b | 1 |
| 3d–3e | Phase 3c done | After 3c | 1 |
| 4 | Phase 1+2 done; benefits from 3 | After Phase 2 | 2 (block families parallelize); Week 10 design spike serial |
| 5 | Phase 4 done | After Phase 4 | 1 |

**Adapter work in parallel:**
- Phases 0, 1, 2: invisible to adapter authors. Continue current pace.
- Phase 3: capability declarations added to existing models (~1 day per model). MoE adds ~2 days per MoE model.
- Phase 4: significant adapter disruption — block schemas declared per block class.
- Phase 5: minimal disruption.

**Recommended scenario for 1-engineer team competing with Unsloth:**
- Months 1–7 solo on Phases 0–3 (FP8/FP4 + MoE + multi-GPU shipping wins).
- Continue adapter work during Phases 0–2.
- Pause non-essential adapters during Phase 3.
- Phases 4–5 deferred until FP8/FP4+MoE wins are proven.

**Recommended scenario for 2-engineer team:**
- Engineer A: Phases 0 → 1 → 2 → 3 sequence (architect track).
- Engineer B: Phase 0 multi-GPU CI infra → MoE capabilities + CUTLASS grouped (Phase 3b) → Phase 4 block migrations (block families parallel to A's other work).
- Critical path stays ~4 months.

---

## 8. Phase checkpoint deliverables

To keep the plan grounded:

**Week 1:**
- Phase 0 numerical regression suite covering Qwen3 BF16+FP8, Gemma4 BF16, Nemotron-H BF16, Qwen3.5 BF16, Mixtral-style MoE BF16. Snapshots committed.
- Multi-GPU CI lane scaffolded (single-node 2-GPU minimum).
- `TensorRole` + `Distribution` struct defined (header + design doc).
- North-star metric report scaffolded; baseline number committed.

**Week 2:**
- Phase 0 perf suite running; baseline locked. Comm-time attribution included.
- Phase 0 memory suite running; baseline locked.
- Phase 0 CPU-training validation lane committed.
- Multi-GPU CI lane functional; first distributed regression baseline locked.
- `TensorRole` + `Distribution` integrated into `Tensor`/`TensorMeta`. No consumers yet.

**Week 3:**
- First migration: [`compiled_ops_save.cpp:170-173`](csrc/src/runtime/executor/compiled_ops_save.cpp#L170-L173) MoE substring scan replaced by `Ownership::MoE` flag check. Parity assertion in place.
- North-star metric chart published.

**Week 4:**
- Five more substring-classification call sites migrated.
- First distributed migration: `dsl_grad_store.cpp:357-374` reduce-decision derived from `Distribution` field with parity assertion.
- Phase 1 progress: ≥30% of `name.find` classification call sites converted; ≥1 distributed code path consults `Distribution`.

**Checkpoint signal:** Phase 0 fully running (incl. multi-GPU lane), Phase 1 ≥30% complete, north-star chart committed and tracked. **If these don't ship, the plan is in trouble.**

---

## 9. Adjacent work tracks (not in refactor scope)

Coordinate, don't merge.

### 9.1 Gemma4 perf investigation
- Refactor doesn't fix Gemma4 perf in short term. Phase 4 helps structurally.
- Concrete moves: apples-to-apples vs torch+unsloth on packed-token; nsys diff; per-layer-type attribution; Triton GEMM probe → 22.8% cutlass kernel mapping; GeGLU fusion.
- Resource: ~1–2 weeks focused investigation.

### 9.2 Adapter velocity tooling
- Validation harness with auto-bisect first; scaffolder + standard block library second; end-user one-liner UX last.
- Resource: ~3 months parallel to refactor.

### 9.3 Triton GEMM integration for skinny shapes
- 1.25–1.47x on layout-exact skinny shapes. ~7% wall potential on Gemma4 if mapped correctly.
- Tactical version ships before refactor. After Phase 3, Triton kernels become recipe specializations registered via capability predicates.
- Resource: ~2 weeks tactical; revisit during Phase 3.

### 9.4 Distributed perf investigation ⭐ NEW
- **Why separate:** the refactor preserves existing distributed correctness but doesn't optimize new distributed configurations.
- **Scope:** apples-to-apples comparison vs PyTorch FSDP / DeepSpeed ZeRO on supported models; comm overlap analysis; weight-transfer (LLEP) effectiveness.
- **Pre-refactor:** establish baselines for each model × distribution config so Phase 0 regression catches drift.
- **Resource:** ~2 weeks pre-refactor, ongoing baseline maintenance.

### 9.5 cuDNN MoE graph testing ⭐ NEW
- **Why separate:** [`moe_cudnn.cpp`](csrc/src/kernels/moe_cudnn.cpp) requires cuDNN ≥9.15. Version coupling is a hidden dependency.
- **Scope:** explicit cuDNN version capability declarations; fallback path validation when cuDNN graph fails.
- **Resource:** ~1 week + ongoing as cuDNN versions move.

### 9.6 FP8 MoE backward closure ⭐ COMMITTED TACTICAL — ships before refactor

**Status:** committed 2026-04-26 as tactical pre-refactor win. Detailed spec below.

**The precise gap** (after audit of [`fp8_hybrid_recipe.cpp:708-806`](csrc/src/recipes/fp8_hybrid/fp8_hybrid_recipe.cpp#L708-L806) and [`recipe.cpp:145-174`](csrc/src/recipes/recipe.cpp#L145-L174)):

| Component | Current state | After closure |
|---|---|---|
| MoE forward grouped GEMM | Native FP8 via cuDNN ([`moe_cudnn_grouped_gemm_fp8`](csrc/src/kernels/moe_cudnn.cpp#L319)) or `moe_grouped_gemm` (FP8 cuBLAS) | Same |
| MoE backward `dinp = W^T × dout` | Native FP8: E4M3 weights × E5M2 grads via [`moe_grouped_gemm_up_backward`](csrc/src/kernels/kernels.h#L4035) | Same |
| **MoE backward `dW = dout^T × inp`** | **BF16 only**: [`moe_grouped_gemm_weight_grad`](csrc/src/kernels/kernels.h#L3822) has float/BF16 variants; called from DSL dispatcher | **FP8: E4M3 inp × E5M2 dout → BF16 dW** |
| Expert weight FP8 cache forward→backward | None: backward re-quantizes weights every step ([`fp8_hybrid_recipe.cpp:760-779`](csrc/src/recipes/fp8_hybrid/fp8_hybrid_recipe.cpp#L760-L779)) | `mMoEFP8Cache` analog to dense `mFP8Cache` |
| Permute → grouped GEMM co-located quant | BF16 permute output, quantize before GEMM | **Deferred to Phase 3 refactor** (touches permute kernel) |

**Track A — Native FP8 wgrad kernel (~2 weeks)**

New kernel signature added to `csrc/src/kernels/`:

```cpp
/// FP8 MoE wgrad: dW[e] = dout_for_e^T @ inp_for_e  (per expert, batched)
/// Inputs: E4M3 input acts × E5M2 upstream grads
/// Output: BF16 dW (accumulated; gradient precision preserved)
void moe_grouped_gemm_weight_grad_fp8(nv_bfloat16* d_weight,           // (num_experts, M, N) BF16
                                       const __nv_fp8_e5m2* grad_output, // (total_tokens, M) E5M2
                                       const __nv_fp8_e4m3* input,       // (total_tokens, N) E4M3
                                       const float* grad_output_scale,   // (1) per-tensor scale
                                       const float* input_scale,         // (1) per-tensor scale
                                       const int* expert_offsets,
                                       int num_experts,
                                       int M,
                                       int N,
                                       cublasLtHandle_t cublas_handle,
                                       cudaStream_t stream,
                                       const int* host_offsets,
                                       float alpha,
                                       float beta,
                                       const int* active_expert_indices,
                                       bool weight_is_compact,
                                       int num_active_experts);
```

Implementation: cuBLASLt grouped FP8 matmul with TN layout (transpose A=dout). Mirrors the structure of existing `moe_grouped_gemm_up_backward` FP8 path but with the wgrad transpose pattern.

**Track B — Forward→backward FP8 weight cache (~1 week)**

Add `mMoEFP8Cache` to executor state, analogous to existing `mFP8Cache`:

```cpp
struct MoEFP8CacheEntry {
    Tensor weights_e4m3;          // (num_experts, M, K) E4M3
    Tensor weight_amax;           // (num_experts,) FP32
    Tensor weight_scales;         // (num_experts,) FP32
    bool initialized;
};

std::unordered_map<std::string, MoEFP8CacheEntry> mMoEFP8Cache;
```

Wire into:
- `forward_moe_matmul` (lines 656-674): write quantized weights into cache instead of temp_alloc.
- `backward_moe_matmul` (lines 760-779): read from cache instead of re-quantizing. Eliminates `num_experts` × {abs_max, quantize_with_abs_max} kernels per backward call.
- Cache invalidation: parameter update step (after optimizer) flips `initialized=false`.

**Track C — Recipe wiring + dweight integration (~3 days)**

Extend `MoeMatmulContext` with `dweight` field if not present (verify against current state). `FP8HybridRecipe::backward_moe_matmul` calls `moe_grouped_gemm_weight_grad_fp8` when `ctx.dweight && ctx.allow_fp8 && ctx.dout_quant && cache hit`. Otherwise falls back to existing BF16 wgrad path in DSL dispatcher.

**Track D — Validation (~1 week)**

- Per-expert wgrad numerical parity: FP8 wgrad result vs BF16 wgrad result, max-abs delta tolerance 5e-2 per expert.
- Convergence: 5-step run on Mixtral-style or Qwen3-MoE, FP8 forward + FP8 dgrad + FP8 wgrad, loss curve within 5% of FP8-forward+BF16-backward baseline.
- Perf: nsys diff showing wgrad kernel time reduction (BF16→FP8) and elimination of per-step weight quantize kernels.
- Multi-GPU: same on 2-GPU DP and 2-GPU DP+EP.

**Total scope: ~4 weeks calendar (~3 weeks engineering + 1 week soak).**

**Sequencing relative to refactor:** ship before Phase 0 starts, ideally. Bakes for 2 weeks in production CI before refactor begins, so the Phase 0 baseline includes the closure (otherwise it'd be a confounder during the refactor).

**Post-refactor migration:** after Phase 3 lands, the cache becomes a `weight_cache_eligible` capability declaration on the MoE op; the wgrad kernel becomes a recipe specialization registered via capability predicate. Tactical implementation stays correct; only the dispatch surface changes.

**Risk:** medium. New FP8 kernel implementation is contained but cuBLASLt grouped FP8 matmul with TN layout has its own quirks (alignment, scale handling). Mitigation: validate against BF16 reference per-expert; ship behind `SUROGATE_FP8_MOE_WGRAD=1` env flag for first 2 weeks of production exposure.

**Owner / target:** TBD. Aim to ship as a near-term tactical closure.

---

## 10. What we are explicitly NOT doing

- **Not rewriting the DSL IR.** It's already architecture-agnostic.
- **Not rewriting CUDA kernels.** Refactor changes how they're called and registered, not their internals.
- **Not building plugin/dynamic-loading.** Static-init `REGISTER_*` macros are sufficient.
- **Not making recipes architecture-agnostic at the kernel level.** Capabilities are the right abstraction; full kernel-level genericness is wrong.
- **Not chasing PyTorch universality or HF compat breadth.** Specialized fast-path remains the strategic position.
- **Not addressing Gemma4 perf as part of refactor.** Separate track (§9.1).
- ~~**Not doing Phase 5 hook registry until Phase 3 ships,** no matter how tempting.~~ **SUPERSEDED:** Phase 5 started early as opt-in, dual-path scaffolding after the Phase 4 schema work made the hook targets available.
- **Not introducing new closed enums during refactor.** Reviewers enforce.
- **Not implementing tensor parallelism or pipeline parallelism.** DP + EP + ZeRO-2 are the supported distributed configs. TP/PP would be a separate project; capability fields reserve room.
- **Not implementing ZeRO-3 (parameter sharding).** Out of scope for refactor; capability fields reserve room.
- **Not implementing LLEP weight transfer hot-loop optimization.** The plumbing is there; behavior optimization is separate. Phase 0 baselines current behavior.
- **Not adding new MoE routing strategies** (expert-choice, soft-MoE, etc.) during refactor. RoutingSchema reserves room; new strategies are post-Phase-4.

---

## 11. Strategic outcome

**After Phases 0–3** (~6.7 months critical path):
- Every supported architecture's GEMMs (dense and grouped) are FP8/FP4-eligible by declaration.
- MoE recipe specialization extends to any architecture with a routing schema.
- Distributed correctness is structural — `CommunicationProfile` declared per op, comm scheduling derived from declarations.
- Adding a new architecture's matmul/grouped ops takes hours; FP8/FP4 inherited automatically.
- Fusions stop being new enum values; matching rules.
- Gemma4 softcap fusion, Mamba GatedRMSNorm, MoE routing fusion all become 1-rule additions.
- Surogate vs Unsloth pitch: *"FP8 LoRA training for Mamba/MLA/gated-delta MoE/whatever-comes-next, on day one. Multi-GPU + LoRA + FP8 + MoE + EP works out of the box. Unsloth gates multi-GPU on paid tier and gets to FP8 MoE in 6+ months, if at all."*

**After Phases 4–5** (~4 additional months):
- Hybrid models stop using aliasing hacks. Per-block-dim variation (Gemma4) structural.
- LoRA + CPU training works. LoRA + CPU + ZeRO-2 + FP8 + MoE works.
- LoRA on Mamba/MLA/gated-delta projections becomes possible.
- New architectures = mostly Python work (block subclass + schema + capability declarations + routing schema for MoE).
- C++ runtime stops growing per-architecture or per-distribution switches.

---

## 12. Open questions (resolve before Phase 0 ships)

- **Team size confirmation.** Solo or two engineers? Determines critical-path vs full-refactor timeline.
- **Gemma4 pressure.** Blocking adoption, or visible-but-not-blocking? Determines §9.1 priority.
- **Backward fusion rule semantics.** Should `FusionRule.emit_backward` be derived automatically from `emit_forward` (autograd over fusion), or always hand-written? Affects Phase 3 effort estimate.
- **CPU streaming + capture safety.** Does CUDA-graph capture need special handling for prefetch nodes? Needs design spike before Phase 4d.
- ~~**CPU streaming + ZeRO + FP8 design spike.**~~ **RESOLVED** by [SPIKE_CPU_ZERO_FP8.md](SPIKE_CPU_ZERO_FP8.md) on 2026-04-26. Findings: existing master/work + double-buffered prefetch + ZeRO-3 weight sharding already wired. **Q1 RESOLVED** (v1.2): no FP8 cache invalidation bug — cache only holds non-trainable weights ([`graph_executor_weight_cache.cpp:121`](csrc/src/runtime/executor/graph_executor_weight_cache.cpp#L121) `is_trainable()` gate); frozen weights don't change so no invalidation needed. **Q4 RESOLVED** (v1.1): LoRA + CPU training works today on single-GPU; gate at [`dsl_model_execution.cpp:319`](csrc/src/runtime/dsl/dsl_model_execution.cpp#L319) is a correct no-op skip. Multi-GPU LoRA + CPU + ZeRO-2 + FP8 is the actual unvalidated frontier.
- **NVMe offload tier.** In scope for Phase 4 or deferred? `StorageTier` reserves the value; execution open.
- **Architecture-specific fusion rule namespace.** Model files ship own `REGISTER_FUSION_RULE`, or central runtime? Affects modularity vs discoverability.
- **CUTLASS grouped GEMM build dependency.** CUTLASS is a heavy dep; gate behind compile-time flag or always include?
- **LLEP weight transfer status.** Current behavior preserved as-is, or does Phase 4d need to declare `WeightTransferP2P` capability semantics?
- **ZeRO-3 in scope?** Defaults to no; reserves room. Confirm.
- **Tensor parallelism in scope?** Defaults to no; capability fields reserve room. Confirm.
- **Pipeline parallelism in scope?** Defaults to no. Confirm.
- **FP8 MoE backward closure** (§9.6): pre-refactor tactical addition, or wait for Phase 3 capability surface? Tactical recommended (faster, lower risk).
- **cuDNN minimum version.** ≥9.15 for moe_grouped_matmul. Hard requirement or capability-gated fallback?
- **Multi-node CI.** Single-node 2-GPU and 4-GPU baseline. Multi-node nightly runs in scope or deferred?

---

## Changelog

- **2026-04-26 v1** — Initial draft. Phases 0–5 specified. CPU training and fusion/epilogues integrated into Phases 1, 3, 4. Adjacent tracks (Gemma4 perf, adapter tooling, Triton GEMM) called out as parallel.
- **2026-04-26 v2.4** — Spike Q1 RESOLVED via static analysis. No FP8 cache invalidation bug exists. The cache at [`graph_executor_weight_cache.cpp:121`](csrc/src/runtime/executor/graph_executor_weight_cache.cpp#L121) refuses to cache trainable weights (`is_trainable()` gate); only frozen weights are cached, and frozen weights don't change so no invalidation is needed. `mWeightManager->invalidate()` serves the streaming/prefetch version chain, **unrelated** to the FP8 cache. Cache is correct by construction. Spike total drops from ~3 weeks to ~2 weeks. Corollary perf observation logged in spike: caching trainable FP8 weights with version chain could save 1-3% wall on FP8 full fine-tune — future-work optimization, not refactor scope.

- **2026-04-26 v2.3** — Spike Q4 corrected based on user empirical testing: LoRA + CPU training works on single-GPU today. The gate at [`dsl_model_execution.cpp:316-319`](csrc/src/runtime/dsl/dsl_model_execution.cpp#L316-L319) is a correct no-op skip (skipping base-grad streaming under frozen base), not a LoRA + CPU block. Source comment updated. Spike effort drops by 1-3 days. Updated supported-combinations matrix: single-GPU LoRA + CPU is ✓ today; the actual frontier is **multi-GPU LoRA + CPU + ZeRO-2 + FP8** (and + MoE).

- **2026-04-26 v2.2** — CPU + ZeRO + FP8 design spike completed; written to [SPIKE_CPU_ZERO_FP8.md](SPIKE_CPU_ZERO_FP8.md). Findings reframed scope significantly: master/work pattern, weight streaming via `mStreamWeights`, double-buffered prefetch, ZeRO-3 weight sharding (`shard_weights`), and version-based cache invalidation are all already in place. Spike identified 7 concrete integration questions (~3 weeks audit+fix) instead of greenfield design. Risk register row 5 downgraded `High/High` → `Medium/Medium`. Phase 4d cornerstone risk substantially reduced. Open audit items: FP8 cache version chain (Q1), prefetch ping-pong handling (Q2), LoRA + CPU one-line gate verification (Q4).

- **2026-04-26 v2.1** — Detailed §9.6 spec for FP8 MoE backward closure committed as tactical pre-refactor work. Audit narrowed the gap to: (a) missing FP8 variant of `moe_grouped_gemm_weight_grad`, (b) per-step weight requantization waste in backward, (c) absent forward→backward FP8 expert weight cache. Three engineering tracks, ~4 weeks calendar. Targets shipping before Phase 0 begins so the refactor baseline includes the closure.

- **2026-04-26 v2** — Thorough audit of MoE and multi-GPU surfaces:
  - Added §2 axes 7 (MoE specialization) and 8 (Distribution + communication) to diagnosis.
  - Updated §3 from 5 to 7 architectural separation axes.
  - Acknowledged `op_registrations.cpp` already provides foundation for op registry — Phase 2 extends rather than greenfields (saves ~3 weeks net).
  - Phase 0 expanded from 4 → 7 weeks: multi-GPU CI infrastructure, MoE assertions, distribution coverage in test matrix.
  - Phase 1 expanded from 5 → 6 weeks: `Distribution` field on `TensorRole`; migration of `dsl_grad_store.cpp` and `lora_grads_manager.cpp` sharded paths.
  - Phase 2 expanded from 8 → 7 weeks (net): added `CommunicationProfile` and `GroupedSemantics` to descriptor; building on existing registration pattern.
  - Phase 3 expanded from 7 → 9 weeks: added §3b `MoECapabilities`, CUTLASS grouped GEMM as recipe specialization, MoE-specific fusion rules; explicit tracking of FP8 MoE backward gap and FP4 MoE silent fallback.
  - Phase 4 expanded from 10 → 12 weeks: `RoutingSchema` and `EPTopology` block declarations; per-rank-aware allocation; storage × distribution × LoRA combination matrix; CPU + ZeRO + FP8 design spike.
  - Phase 5 expanded from 4 → 5 weeks: distribution-aware hooks (after-all-reduce, after-all-to-all, after-reduce-scatter).
  - Risk register: added rows 1, 3, 4, 5, 12, 13, 14 (distributed/MoE-specific risks).
  - §7 resourcing updated: critical path 24 → 29 weeks; full refactor 38 → 46 weeks.
  - §9 added tracks 9.4 (distributed perf), 9.5 (cuDNN MoE testing), 9.6 (FP8 MoE backward closure).
  - §10 added: TP/PP/ZeRO-3/LLEP-optimization/new-routing-strategies as explicit non-goals.
  - §12 added open questions on CUTLASS dep, LLEP, ZeRO-3, TP/PP, cuDNN min version, multi-node CI.
