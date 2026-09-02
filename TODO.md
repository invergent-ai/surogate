# TODO — serve-engine branch

Status legend: **DONE** (commit) · **IN PROGRESS** · **TODO** · **DEFERRED** (recorded, not scheduled)

Working order is top to bottom within a group; groups are done serially, each
built, tested, verified on GPU 1 and committed before the next.

## Group 1 — W8 `linear_add` plan and launchers

- [x] **DONE** (this pass) k % 256 scale-row alignment refused in the plan by name, and per launch in both MMA launchers; a registered shape whose table has an MMA band fails to compile if misaligned.
- [x] **DONE** (this pass) One registered-shape list (`w8_linear_add_registered_shapes`) feeds the plan, the wrapper's W8 gate and the conformance test; the wrapper no longer restates it (gemma3 `{640, *}` was admitted by one and refused by the other).
- [x] **DONE** (this pass) TinyLlama `{2048, 5632}` exact-T bake over T=2..32, measured 2.8–4.2x over the runtime tile (PATCHES.md #90); exact-T tables keyed on (rows, k) in one table; bench `--k` accepts any registered k.
- [ ] **TODO** `w8_linear_swiglu_gemm_mma.cu` launches the same MMA kernel with no alignment guard; its plan has no per-table rule; the wrapper/plan two-list drift is likely repeated in `wrapper/linear_swiglu.cpp` and `wrapper/attn_input_proj.cpp`.
- [ ] **TODO** bench `--rows` so gemma3's 640-row shapes can be measured for a bake of their own.

## Group 2 — frontend

- [x] **DONE** (Group 2 commit) `validate_tokenizer_config` no longer fails open: string, list-with-`default`, or absent; every other form refused by name; `sinfer_family_tokenizer_config_test` covers the forms without tokenizer resources.
- [x] **DONE** (Group 2 commit) The SentencePiece delegate now renders the template the frontend serves (artifact jinja, or the `--chat-template` override, which it silently ignored before).
- [ ] **TODO** (Group 4) converter `_tokenizer_config_with_template` should refuse an explicit `null` and `_chat_template()`'s message should say "is not a string" for the list form.
- [ ] **DEFERRED** `test_frontend.cpp` asserts `resources("{{ messages }}")` throws, but `resolve` stopped throwing for unknown templates in 3bc719fd; the test skips without tokenizer resources so nobody sees it.

## Group 3 — generator / DSL

- [ ] **TODO** `from_dsl` reads only `sliding_window_pattern`; a `layer_types`-only config emits period 0 = every layer windowed. Emit the resolved per-layer schedule as data; refuse `window > 0` with no schedule; drop the hardcoded Gemma phase from the emitter.
- [ ] **TODO** `attention_scale` derived as `head_dim ** -0.5`, ignoring `query_pre_attn_scalar` (Gemma3-27B: scalar 168 vs head dim 128).
- [ ] **TODO** `TargetSpec.validate()` enforce "all three window fields or none"; hybrid emitter emits the window block or raises, never silently drops it (gemma4, laguna declare a window).
- [ ] **TODO** Embedding-scale parity: DSL `gemma3.py` applies fp32 `sqrt(d)`, serving rounds to bf16 as HF does and as `gemma4.py` does; settle against HF and align the DSL + comments.
- [ ] **TODO** Regenerate `targets/gemma3/impl/config.h` from the emitter; add gemma3 to `check_roundtrip` if its naming allows.

## Group 4 — Gemma 3 converter

- [ ] **TODO** `_OPTIONAL_CONFIG` is dead: all four keys are also in `_REQUIRED_CONFIG` and the required check runs first, so an export without `use_bidirectional_attention` is refused. Make the tables disjoint.
- [ ] **TODO** `_ENGINE_CONSTANTS` demands `_sliding_window_pattern`; accept `layer_types`, derive the schedule as the DSL does, refuse a checkpoint that disagrees with the header's schedule (contract set by Group 3).
- [ ] **TODO** Tied lm head stored and bound twice (~170 MB artifact, ~168 MB device). Alias `text/output_head` onto `text/token_embedding` via `LogicalAliasSpec` end to end (inventory, converter, bindings); reconvert and re-verify HF match.

## Group 5 — CUDA-graph budget accounting

- [ ] **TODO** `graph_bytes_` is a `cudaMemGetInfo` free-before minus free-after, so two engines starting on one GPU attribute each other's weight loads to graph preparation and abort. Make the measurement or the decision process-robust; keep refusing to serve without headroom.

## Group 6 — windowed attention performance

- [ ] **TODO** Prefill iterates key blocks from 0 and decode splits over `[0, last_pos+1)`, masking outside the window; Gemma's 15 windowed layers do O(context) work for a 512-key answer. Trim the key range (prefill `n_block_min`, decode re-based splits) keeping the decode split count capture-stable; measure TTFT and decode tok/s at ~8k tokens.
- [ ] **TODO** `Gqa256_4q1` `DecodeSplitScale = 4` drives 16-key splits against a 32-key tile (2x staged bytes and MMA); clamp the policy to the tile or register scale 2, by measurement.
- [ ] **TODO** Add `{"gemma3_270m", 4, 1}` to `test_gqa_attention` `kGeometries` so the MQA shape gets numerical conformance.

## Group 7 — Gemma sandwich norms

- [ ] **TODO** `rmsnorm` + `residual_add` at both sandwich sites cost 72 extra graph nodes and 36 planes per round. Add a generic accumulating `rmsnorm_add` (`out += bf16(rmsnorm(x) * gain)`, bit-identical to the two-step form), use it at both sites, shrink the workspace accounting; measure decode tok/s and TTFT.
- [ ] **TODO** (with the above) gemma3's `attention_output_workspace_bytes` still reserves `linear_add` capacity for a path the leaf never runs; delete it when the accounting is rewritten.

## Recorded, deferred

- [ ] **DEFERRED** Per-layer KV capacity for windowed layers (Gemma: 15 of 18 layers need 512 tokens of KV, not the full context; ~5.6x KV reduction). Structural.
- [ ] **DEFERRED** Converter/registry/package boilerplate duplicated per target (driver loop, `LoadPlan`/`LoadedModel` pimpl, `LoadedX`/`XInstance`, debug probe, unrunnable-leaf stubs). Hoist into `family/`.
- [ ] **DEFERRED** `_float_literal` emits 7 significant digits; a C++ float needs 9 to round-trip. Changing it regenerates every target header.
- [ ] **DEFERRED** Seven real-artifact serve tests skip without `SINFER_*_WEIGHTS`; they are the end-to-end tests of `advance_prefill`.
- [ ] **DEFERRED** MTP/DFlash attention sites pass window 0 (documented as deliberate in `mtp_impl.h`); a windowed model with a draft head must decide its window there.

## Standing goal — design/MODELS.md

- [x] Row 1 Qwen3ForCausalLM — SERVES
- [x] Row 3 Gemma3ForCausalLM — SERVES
- [x] Row 7 LlamaForCausalLM — SERVES
- [ ] Rows 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14 — no serving target yet
