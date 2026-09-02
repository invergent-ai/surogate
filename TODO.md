# TODO — serve-engine branch

Status legend: **DONE** (commit) · **IN PROGRESS** · **TODO** · **DEFERRED** (recorded, not scheduled)

Each group is built, tested, verified against the HF reference and committed
before the next. Run the suite with `CUDA_VISIBLE_DEVICES=<free gpu> ctest
--test-dir csrc/build-serve -j8` (53 s, 104 tests) rather than a serial loop.

## Group 1 — W8 `linear_add` plan and launchers — **DONE** (`8beb4e69`)

- [x] **DONE** (this pass) k % 256 scale-row alignment refused in the plan by name, and per launch in both MMA launchers; a registered shape whose table has an MMA band fails to compile if misaligned.
- [x] **DONE** (this pass) One registered-shape list (`w8_linear_add_registered_shapes`) feeds the plan, the wrapper's W8 gate and the conformance test; the wrapper no longer restates it (gemma3 `{640, *}` was admitted by one and refused by the other).
- [x] **DONE** (this pass) TinyLlama `{2048, 5632}` exact-T bake over T=2..32, measured 2.8–4.2x over the runtime tile (PATCHES.md #90); exact-T tables keyed on (rows, k) in one table; bench `--k` accepts any registered k.
- [x] **DONE** (`28406e6e`) `w8_linear_swiglu_gemm_mma.cu` refuses a misaligned k like its two siblings. Every swiglu k admitted today is a multiple of 256, so nothing served changes.
- [ ] **TODO** The swiglu wrapper still restates the plan's shape list (six hand-written shape predicates); same two-list drift `linear_add` had. `wrapper/attn_input_proj.cpp` may too.
- [ ] **TODO** bench `--rows` so gemma3's 640-row shapes can be measured for a bake of their own.

## Group 2 — frontend — **DONE** (`52abbe9a`)

- [x] **DONE** (Group 2 commit) `validate_tokenizer_config` no longer fails open: string, list-with-`default`, or absent; every other form refused by name; `sinfer_family_tokenizer_config_test` covers the forms without tokenizer resources.
- [x] **DONE** (Group 2 commit) The SentencePiece delegate now renders the template the frontend serves (artifact jinja, or the `--chat-template` override, which it silently ignored before).
- [ ] **TODO** (Group 4) converter `_tokenizer_config_with_template` should refuse an explicit `null` and `_chat_template()`'s message should say "is not a string" for the list form.
- [ ] **DEFERRED** `test_frontend.cpp` asserts `resources("{{ messages }}")` throws, but `resolve` stopped throwing for unknown templates in 3bc719fd; the test skips without tokenizer resources so nobody sees it.

## Group 3 — generator / DSL — **DONE** (`0132d93f`)

- [x] **DONE** Schedule emitted as a per-layer `kWindowedAttention` array, not a period; the phase convention comes from the declaration, not the emitter; a window with no resolvable schedule raises. (Root cause was `sliding_window_pattern: None` read as 0 = all windowed.)
- [x] **DONE** `attention_scale` prefers `query_pre_attn_scalar` (270M unchanged; a 27B-shaped spec now emits 0.0771517 instead of 0.0883883).
- [x] **DONE** `validate()` enforces all-or-none plus per-layer coverage; the hybrid emitter emits the window block instead of dropping it.
- [x] **DONE** Embedding scale settled against HF source (`embed_scale.to(weight.dtype)`): the serve side was right, the DSL now rounds to bf16 like its Gemma 4 sibling. Serving numerics unchanged; training moves 0.19% at layer 0 toward the reference.
- [x] **DONE** `config.h` regenerated and covered by `check_roundtrip` (3/3).

## Group 4 — Gemma 3 converter — **DONE** (`ea84f7e7`)

- [x] **DONE** Config tables disjoint; an export without `use_bidirectional_attention` converts again.
- [x] **DONE** Either schedule spelling accepted, resolved through the DSL's own function, and a disagreeing checkpoint refused naming every differing layer. (Fixed a shadowing bug: a present-but-null plain key hid the underscored one.)
- [x] **DONE** Tied head aliased end to end. **Measured**: artifact 497,748,992 → 319,491,072 bytes; device weights 441.71 → 271.71 MiB (170 MiB each). Aliased artifact is HF-exact; an older artifact still loads.
- [ ] **TODO** `convert.py` still writes the fp32 embedding scale into its summary and its comment, one rounding behind the header (cosmetic; the engine reads its own constant).

## Group 5 — CUDA-graph budget accounting — **DONE** (`d6ec11f7`)

- [x] **DONE** (Group 5 commit) Graph preparation is measured per process via NVML (`core/device_footprint.*`, dlopened so the CUDA stub library is never linked); the two derived-plane registries accumulate the sizes they allocate instead of a device-wide delta, so both sides of the subtraction are this process's own. The refusal fires only on an attributed figure; unattributed it is reported. Two engines now start together on one GPU and both answer.

## Group 6 — windowed attention performance — **PARTLY DONE** (`4595ff1c`)

- [x] **DONE** Prefill starts at the window's first key block in both kernels instead of scanning from key 0. **Measured** at 8k, prefix reuse off, medians of 3: prefill **28,412 → 30,657 tok/s (+7.9%)**, HF-exact preserved. (The first attempt measured wall time, which is dominated by tokenization — see below.)
- [x] **DONE** `{"gemma3_270m", 4, 1}` added to `kGeometries` — the first numerical conformance the multi-query shape has had, and what makes the trim a tested path.
- [x] **DONE** (`4e1fccb8`) Decode splits the window's key range, not all of history. **Measured** at 8k context, interleaved: **375.8 → 414.8 tok/s (+10.3%)**. Against 434 tok/s at 8 tokens of context, the old path gave up 13.4% by 8k and the new one gives up 4.5% — the rest is the 3 global layers. HF-exact, deterministic, ladder matches 10/10.
- [x] **DONE** The gqa test now poisons the workspace. A split past the active count returns without writing, so zeros in the partial buffers hid a reducer/kernel split-count disagreement; with the poison, reverting the reducer fails exactly the decode window cases (without it, it passed).
- [ ] **TODO** `Gqa256_4q1` `DecodeSplitScale`: apply the tile floor only to the default tiers, never to the measured INT8 special cases (`24/scale`, `32/scale` bands are deliberately sub-tile).

## Group 7 — Gemma sandwich norms — **DONE** (`913dc6e3`)

- [x] **DONE** Generic `ops::rmsnorm_add` (`out += bf16(rmsnorm(x) * gain)`), used at both sandwich sites. **Measured**: largest captured decode graph 353 → 317 nodes (exactly 36 = one `residual_add` per site × 18 layers, 10.2% of dispatch); single-user decode 439.2 → 445.1 tok/s (+1.3%, interleaved A/B/A/B, 8 samples each). Bit-identical to the pair it replaces, pinned by a composition test over 7 shapes; removing the pre-add rounding fails 3 of them.
- [x] **DONE** The dead `linear_add` reservation in `attention_output_workspace_bytes` is gone, and each leaf drops one hidden-wide plane.
- Note: the review's original "72 nodes / 36 planes" was 2× too high — only `residual_add` is removed per site, since a row-wide reduction cannot fuse into the GEMM. Measurement confirms 36.
- Graph node counts are now reportable behind `SUROGATE_SERVE_GRAPH_NODES`; nothing else exposed what a captured round costs to dispatch.

## Performance found along the way

- [x] **DONE** (`9ce7c3b6`) SentencePiece encoding was O(n²) — it rescanned every adjacent pair per merge and built a key string per pair. 4,000 tokens took 7.5 s, **425× slower than the reference tokenizer**, and an 8k prompt ~29 s against 0.26 s of prefill, so a long request was almost entirely tokenization and back-to-back ones expired in the queue. Now a merge heap over a linked list of input spans: **54× at 4,000 tokens**, linear growth, token-for-token identical output.
- [x] **DONE** (`5b578d6f`) Test oracles spawned a thread set per call — over a million threads in the widest sparse-MoE cases, with the profile almost entirely `clone3`/`allocate_stack`. One shared pool (`ops/parallel_rows.h`): **sparse-MoE 62 s → 10 s**. Also, `ctest` reported a skip (exit 77) as a failure because only op-tests set `SKIP_RETURN_CODE`.
- [x] **DONE** Suite: **352 s serial → 40–60 s at `ctest -j16`**, 104 tests, 100% passing.
- [x] **DONE** (`59df8f5a`) `projection_oracle` re-decoded a whole 5,120-wide weight row once **per token**, through a per-element accessor that revalidates metadata and switches on quantization type every element. Decoding each sampled row once is the same arithmetic in the same order: **NVFP4 21.2 → 2.2 s, FP8 10.0 → 1.9 s**; attention input projection 39 → 11 s, GDN (shares the oracle) 18 → 8 s.
- [x] **DONE** Oracle pools capped at 8 threads (`SUROGATE_TEST_ORACLE_THREADS` overrides). One machine-sized pool per process oversubscribed under `ctest -j` — 8 processes × 64 threads on 64 cores took the suite 53 → 132 s and failed a CPU-bound test. **Suite is now 40–60 s at `-j16`, 104 tests, 100% passing**, against 352 s serially.
- [ ] **TODO** `sinfer_gqa_attention_test` is the floor at ~32 s, and it is GPU work rather than oracle arithmetic.
- [ ] **TODO** `linear_test_common.cpp` and `linear_swiglu_test_common.cpp` still spawn per call; they run once per case rather than per token so the churn is bounded, but the shared helper (including a range form for per-thread scratch) now exists.

## Found while measuring

- [ ] **TODO** (investigated, not changed) EmbeddingGemma's `embedding_scale`. The reference casts the scalar to the **weight** dtype, so the right value depends on the dtype of the run being matched: bf16 → 27.75, fp32 → 27.712812921102035, differing 0.134% on the residual stream. The generation path rounds because its target is bf16; this encoder is checked against an **fp32** reference (cosine 0.999855), which points the other way. The reasoning is recorded beside the constant; settling it needs that reference re-run, which needs sentence-transformers (absent here).
- [x] **DONE** (`28406e6e`) `bindings.h` no longer describes the schedule as a period.
- [x] **DONE** (`28406e6e`) Test targets get the CUDA include path. The elastic-KV work made a target package header reach `cuda_runtime.h` via the KV pool, breaking any test linking `sinfer_engine`.

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
