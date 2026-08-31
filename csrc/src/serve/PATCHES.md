# Vendored code under csrc/src/serve/

> **Naming note (2026-08-31):** the engine's own identifiers were renamed
> `ninfer` -> `sinfer` (namespace, macros, targets, the artifact magic and the
> `.sinfer` extension). This file is left in its original wording: entries below
> record what was patched *at the time*, and the upstream project this code came
> from is still named `ninfer`. Read macro and path names here as historical.

## serve engine — derived from https://github.com/Neroued/ninfer

(2026-08-26: the vendor directory was flattened into csrc/src/serve as
first-class surogate source — see NOTICE. This file continues as the
serve engineering log; entries below #24 predate the flatten and use the
old ninfer/ paths.)

- License: Apache-2.0 (see `ninfer/LICENSE`). Vendored per the port-first
  kernel/runtime strategy in `design/serve-engine-plan.md` §2.2 / §3.3.
- Source snapshot: `study/ninfer` working tree as of 2026-08-23 (upstream
  git metadata not vendored). Excluded from the vendor copy: `.git/`, `eval/`
  (Python eval harness), `model-cards/`, `.github/`, `.codex/`, `Dockerfile`.
- The tree builds as its own CMake project (`make serve-build` at repo root);
  it is intentionally NOT part of the training `surogate-common` build.

### Local patches (keep this list exhaustive)

1. `CMakeLists.txt` — FFmpeg made optional. New option `NINFER_ENABLE_FFMPEG`
   (default ON, auto-falls-back OFF when pkg-config cannot find
   libavformat/libavcodec/libavutil/libswscale). Upstream hard-requires FFmpeg.
2. `src/CMakeLists.txt` — `ninfer_media_decode` builds `decode_stub.cpp`
   instead of `decode.cpp` when `NINFER_ENABLE_FFMPEG=OFF`.
3. `src/media/decode/decode_stub.cpp` — NEW FILE (surogate). FFmpeg-free stub
   implementing `media/decode/decode.h`: any image/video input raises a clear
   runtime error; text serving is unaffected.
4. `tests/CMakeLists.txt` — `ninfer_media_decode_test` registered only when
   `NINFER_ENABLE_FFMPEG=ON` (it exercises real decoding; cannot pass on the
   stub).

5. `src/ops/common/mma.cuh` — sm_89 port guards. `mma_fp8_e4m3`: the Blackwell
   `.kind::f8f6f4` spelling becomes the plain `mma.sync...e4m3` form below
   `__CUDA_ARCH__ 1200` (Ada has FP8 tensor cores; FP32-accum runs half-rate
   there — accepted). `mma_nvfp4_e4m3`: `__trap()` stub below 1200 (no FP4
   tensor cores before sm_120; W4A4 routes must never be admitted there).
6. `src/core/pdl.cuh` — programmatic dependent launch is sm_90+. Device
   `trigger_dependents`/`wait_for_dependencies` compile to no-ops below 900;
   host `launch_dependent` drops the PDL attribute at runtime on CC < 9.0
   (ordinary stream ordering — semantically identical, without the overlap).
7. `src/ops/linear/nvfp4/nvfp4_codec.cuh` — `cvt.rn.satfinite.e2m1x2` is
   sm_120+. Software RN-even E2M1 encoder (nibble order and byte order match
   the asm path) below 1200.
8. `src/ops/linear/nvfp4/nvfp4_w4a4_tma_stub.cpp` (NEW) +
   `src/CMakeLists.txt` — the W4A4 TMA translation units (TMA, mbarrier,
   cluster, block-scaled FP4 MMA) build only when 120a is in
   `CMAKE_CUDA_ARCHITECTURES`; other arch sets link a loud-failure stub.
9. `src/ops/linear/nvfp4/nvfp4_w4a4_mma.cuh`,
   `src/ops/linear/w8/w8_small_t_mma.cuh`,
   `src/ops/linear/w8/w8_rowsplit_gemm_medium_t_splitk.cuh` — pre-sm_90
   forbids >48 KiB static shared memory at link. Below `__CUDA_ARCH__` 900
   (1200 for W4A4), schedules whose SharedStorage exceeds 48 KiB declare
   1-byte storage and `__trap()`. The sm_89 route resweep must not select
   those variants; a proper dynamic-smem port of the big W8 schedules is a
   tracked follow-up (W4A4 is permanently sm_120-only by hardware).
10. `CMakeLists.txt` — `NINFER_ALLOW_PORT_ARCH` opt-in allows configuring
   architectures beyond upstream's hard 120a gate, for the port effort.
11. `apps/CMakeLists.txt` + `src/serve/console_log.cpp` — product binary names.
   Output names `surogate-engine` (HTTP server) and `surogate-engine-cli`
   (one-shot CLI); the console log prefix follows. Upstream attribution stays
   in NOTICE — process names and logs carry the product's name.
    The console-prefix expectation in `tests/test_request_log.cpp` is
    updated to the renamed `surogate-engine: ` prefix.
12. `tools/convert/qwen3_6/common/official_resources.py` — GGUF-sourced
   conversions reconstruct tokenizer.json/tokenizer_config.json/
   chat_template.jinja from the GGUF's own KV metadata (semantically
   equivalent — encode-identical, tests/serve/test_gguf_frontend.py — but not
   byte-identical to the pinned official files). With
   `NINFER_ALLOW_DERIVED_FRONTEND=1` (set only by the GGUF ingest path in
   surogate/serve/ingest.py) the pinned-hash mismatch for those three files
   downgrades to a recorded stderr warning; safetensors-sourced conversions
   keep the strict check.

14. **Direct GGUF Q8_0 -> W8G32_F16S repack (`--gguf-repack`).** GGML Q8_0
   and the artifact's W8G32_F16S are the same numeric format (int8 codes +
   one binary16 scale per 32-group, value = code * scale), so Q8_0 GGUF
   tensors move into the artifact bit-exactly — no dequantization, no
   requantization, no GPU. New `tools/convert/common/gguf_repack.py`
   evaluates the target's registered TensorRecipe expressions as row algebra
   (Reshape/Slice/Transpose over row axes, row Concat, GatherRows — all
   exact on quantized planes since k % 32 == 0) so the recipe remains the
   single source of layout truth. `convert.py` gains `--gguf-repack
   <map.json>`: planned objects encode via `encode_row_split` from planes
   memmapped straight out of the GGUF (the map carries rows/k/absolute
   offset per source, so the subprocess never runs gguf-py's ~10s KV parse),
   source preflight narrows to the recipes left on the materialize path, and
   a loud invariant rejects maps naming sources that materialized recipes
   still need. The plan is computed against the artifact profile: Q8_0
   sources of BF16-profile objects (e.g. the 0.8B gdn a/b projections) stay
   on the dequant path. Bridge side (surogate/serve/gguf/bridge.py, out of
   tree): candidates = 2D Q8_0 tensors whose llama.cpp inverse transform is
   a row identity (`qwen35.inverse_is_row_identity`; V-reorder families
   excluded until composed as row permutations), narrowed by a planner
   backed by these vendored recipes. The bridge runs on surogate's lean
   GGUF metadata parser (surogate/serve/gguf/lean.py, ~0.1s span-indexed
   open vs gguf-py's ~10s eager KV parse; token arrays parse on demand for
   the frontend; tensor payloads memmap through payload_view into gguf-py's
   dequantize for the non-repacked remainder), and the artifact cache is
   checked by fingerprint before the GGUF is opened at all. Qwen3.5-0.8B
   Q8_0: 159/195 Q8_0 tensors repacked, one-time conversion 8.4s total vs
   ~36s full-dequant (converter core 3.6s), warm start 0.0s, engine output
   identical. Bit-exactness is pinned by
   tests/serve/test_gguf_repack.py against gguf-py's own dequantize.

   The exact set extends past Q8_0: **Q4_0** (codes = nibble - 8), **Q5_0**
   (codes = q5 - 16) and **IQ4_NL** (codes = ggml's int8 codebook lookup)
   all share the ``int8 code x fp16 group-32 scale`` semantics, so their
   tensors ALSO move into W8G32 bit-exactly (REPACKABLE_TYPES plane
   decoders; validated against every Q4_0 tensor of a real mixed-type
   GGUF, 129/129 exact). Q4_1/Q5_1 (additive per-group min) and K-quants
   (6-bit sub-scale products not representable in fp16) stay on the
   dequantize path — the per-tensor candidate planner mixes both paths in
   one file. Community GGUFs that strip the MTP (nextn) block are rejected
   with an actionable error: the artifact inventory is exact and the
   engine's speculative decode requires the block.

15. **Optional MTP block — community GGUF support.** Community exports
   frequently strip the model's MTP (nextn) tensors (unsloth's Qwen3.5-0.8B
   Q4_0/Q4_K_M do); the target now has a registered no-MTP artifact
   variant. Converter: `inventory.active_specs(mtp=...)` /
   `OBJECT_SPECS_NO_MTP` (269 tensors / 275 objects vs 281/287) and
   `convert.py --no-mtp` (recipes, source preflight, repack plan, object
   loop, and report all follow the variant; the draft head stays — it
   derives from the embedding). Loader: `Binder::has()` presence probe;
   qwen3_5_0_8b bindings bind mtp/* only when present, and `--spec mtp` on
   a no-MTP artifact is a clear startup error instead of a missing-object
   failure (the runtime was already optional: materialization and the
   speculative executor key off `features.mtp()`). Also fixed here: the
   0.8B MTP materialization row views still carried 27B extents
   (6144/7168/13312 on a 5120-row fused qkgv) — corrected to
   2048/512/2048/512; and the frontend accepts unsloth's chat template
   (byte-diff is a Jinja-compat rewrite of tool-call argument iteration;
   same markers and thinking toggle, registered as ThinkingToggle) while
   the bridge normalizes exporter-arbitrary pad tokens to the family's
   official <|endoftext|> (serving-internal; no effect on text
   tokenization).

   Validated on GPU: unsloth Q4_0 (MTP-less, mixed Q4_0/Q4_1/Q5_K/Q6_K/
   Q8_0) converts (129 tensors bit-exact-repacked, rest dequantized) and
   serves — exact instruction following, 471 tok/s decode on an idle 5090;
   `--spec mtp` on it errors as designed. Full GPU ctest: 87/88 (the known
   hardcoded-path frontend test). Known frontier: `--spec mtp` on the FULL
   0.8B artifact stops at `gdn_input_proj_conv_record workspace` — the
   speculative-replay op family has not been walked for the 0.8B geometry
   yet (tracked; spec-off serving is unaffected).

   Uncontended 5090 baseline for the untuned q08 routes (PATCHES #13):
   decode 460-470 tok/s stable across context; prefill 4.4k tok/s @ 52
   tokens scaling to 41k tok/s @ 1912 — no route cliffs.

16. **qwen3.5-2b target.** Second breadth target (hidden 2048, intermediate
   6144; 24 layers, 8q/2kv hd256 attention, symmetric 16/16 GDN, vocab
   248320, tied embeddings, 1 MTP — the 0.8b's twin scaled in hidden).
   The geometry lands remarkably cheaply: its mlp gate_up (12288x2048) and
   down (2048x6144) are EXACTLY the registered 35B W8 shapes; its mtp fc
   (2048x4096) is the registered W835bMtpProjectionGeometry; the attention
   qkgv (5120 rows) and gdn qkvz (8192 rows / 6144 conv channels) share the
   0.8b's fused ROW structure with only K doubled — so the small-target
   kernel branches are now keyed on PARENT ROWS (weight.n 5120/8192)
   instead of hidden, with per-K template instantiation where the decode
   kernel bakes K. New: vendored target tree (config.h derivation-driven —
   only hidden/intermediate change), registry/engine variant entries,
   tools/convert/qwen3_5_2b (same 287-object structure), W8 head routes at
   k=2048 (n in {248320, 131072}), linear_add {2048,2048} admission
   (routes over the measured q08 pattern), Bf16Gdn2BGeometry{16,2048} with
   all six launcher guards twinned per K, gdn/attn wrapper gates
   generalized. Conversion-side: ShardReader folds the official releases'
   `model.language_model.*` nesting to the recipes' flat dialect (with an
   ambiguity guard), and F32 control tensors (A_log/dt_bias in official
   checkpoints) widen in with an explicit BF16 narrow at materialize.

   Validated: converter produces the 2.77GB artifact from the official
   safetensors in 19.3s (CPU); attn/gdn/linear/linear_add/gating op tests
   all green with the 2B geometries (the small-target W8 attn comparisons
   now share the one-ULP criterion — same defined-semantics argument as
   the GDN one, hit at K=2048's higher cancellation). Engine E2E pending a
   GPU window (artifact needs ~3.2GB); the shared cores (GQA 8/2 hd256,
   GDN 16/16, conv 6144, gated rmsnorm, embedding d=2048) were already
   covered.

   E2E (idle 5090): first tokens EXACT on the first run ("SUROGATE SERVE
   OK", stop-token; both thinking modes coherent). vs vLLM 0.27.1 bf16:
   decode 320-330 vs ~197 tok/s (+66%); prefill ahead through ~500 tokens,
   behind by 7-12% at 962+/1912 (same W8->BF16-MMA structural ceiling as
   the 0.8b — the FP8-MMA path fixes both). linear_add {2048,2048} carries
   a measured route table (r32c96/r48c128/r32c128 bands); qkgv/qkvz 2B
   candidates measured within ~4% of the current routes (kept). Full GPU
   ctest re-run after the shared-wrapper generalizations: 87/88 (the known
   hardcoded-path frontend test) — the parent-row keying is regression-clean
   across the 27B/35B/0.8b/2b families.

17. **W8A8-int IMMA prefill path (design proven; productization queued).**
   The remaining vLLM lead is >=1k-token prefill, and the ceiling is
   structural: the W8 A16 kernels dequantize int8 codes to BF16 in-kernel
   and top out at ~90-100 TF/s across all 13 measured tile schedules
   (bench/ops/q08_route_sweep_bench). Design decision, made against the
   alternatives: FP8 (e4m3) weights would DEGRADE accuracy (3-bit
   mantissa cannot hold int8 codes); an FP16-MMA pipeline dies on
   activation conversion (BN > BM, the x-side convert costs more than the
   weight dequant saved); **int8 tensor cores** keep W8 codes bit-exact
   (they ARE int8), run at 2x the BF16 MMA rate, and mma.m16n8k32
   consumes exactly one 32-value quantization group per instruction — the
   per-group weight scale applies on the int32 group result before the
   FP32 accumulate, and the per-token activation scale at the epilogue.
   Activations quantize to int8 per token (the standard W8A8 recipe); the
   O(T*K) pre-pass amortizes at prefill token counts, so the route is
   large-T only — decode stays A16 where the engine already leads.

   Evidence (bench/ops/w8a8_imma_probe_bench, idle 5090): a deliberately
   naive pipeline (2-stage cp.async, BM64xBN64xBK64, no swizzle) reaches
   119-124 TF/s at T>=1024 on the four dominant GEMM shapes of both small
   targets — ABOVE the tuned BF16 ceiling — with the int32 group math
   exact against a CPU oracle (2-3e-3 rel = bf16 output rounding).
   Probe iterations (each hypothesis measured, three ruled out or
   confirmed): per-token act-quant = 1.5-3% of combined time (settled);
   4-stage pipeline SLOWER than 2-stage (not cp.async-latency-bound;
   ruled out); ldmatrix at 8 warps SLOWER than manual loads (105-112 vs
   118-124 — at 33% occupancy the compiler-scheduled LDS hide fine); but
   ldmatrix AND 16 warps COMPOSE: the winning configuration is
   **BM64 x BN128 x BK64, 16 warps (warp tile 32x16), 2-stage cp.async,
   80-byte staging stride (bank-conflict-free ldmatrix without XOR
   swizzles), ldmatrix.x4/x2 fragment loads** at **132-144 TF/s** on the
   four dominant GEMM shapes at T>=1024 — ~1.5x the tuned BF16 A16
   ceiling, numerics exact modulo bf16 output rounding. At the measured
   rate the end-to-end arithmetic flips BOTH remaining vLLM leads with
   margin (0.8b 1912-prefill ~52k -> ~66k tok/s vs vLLM-FP8 60.7k; 2b
   ~25.3k -> ~31k vs bf16 28.6k). PRODUCTIZED (first family): the winning config lives in
   src/ops/linear/w8a8/w8a8_imma_gemm.cuh (RowMap/Epilogue-templated) with
   the per-token act-quant op (w8a8_act_quant, kW8A8MinTokens = 512);
   linear_swiglu runs it under LinearPolicy::AllowA8 at T >= 512 (unfused
   pairing pass in v1), every W8 wrapper accepts AllowA8 (A16 execution
   where no A8 path exists yet — "allow" semantics), and the 0.8b/2b
   targets opt W8 into AllowA8 via text_policy. TWO integration findings
   fixed on the way: (a) a last-tile source-clamp made heavily-partial
   tiles ~2x slower — out-of-range tokens now cp_async_zfill (no global
   read); (b) the x staging used .cg, bypassing L1 for data every row-tile
   CTA re-reads — the zfill path caches it, lifting the kernel to
   **149-178 TF/s** across the four shapes (~1.8x the BF16 ceiling).
   E2E with ONE family converted: 0.8b 962-prefill 37.3k -> 40.3k tok/s
   (now ahead of BOTH vLLM columns), 1912 52.0k -> 54.5k (ahead of bf16;
   FP8 60.7k falls when the remaining families convert). Correctness: A8
   swiglu op tests green (A16 fallback at T=511 verified); engine output
   exact. ALL FOUR families now converted: shared dispatchers
   (w8a8_dispatch.{h,cu}: split2 for qkvz, split4 for qkgv in the fused
   q|k|gate|v row order, residual for the output/down projections — all
   direct-write, workspace = act-quant only), wrapper routing at
   T >= kW8A8MinTokens with A16 below, and the target capacity hooks
   rewired from hard zeros to the wrapper capacity functions under
   AllowA8 (the 0-byte hooks would have overflowed the arena).

   RESULT — the completed sweep: with A8 fully active (verified exact on
   700+-token prompts, both models): 0.8b prefill 962: 46.5k tok/s,
   1912: **63.8k — ahead of vLLM-FP8's 60.7k**; 2b 962: 28.6k (+43%),
   1912: 37.8k (+49%, +32% over vLLM bf16). Combined with the decode
   column, the engine now leads vLLM at EVERY measured point on both
   models. Remaining refinements: quantizing-rmsnorm fusion (act-quant
   already only ~2%), A8 test cases for the three direct-write families
   (swiglu has them; the others are engine-exercised), swiglu fused-pair
   epilogue.

18. **qwen3.5-4b target.** Third breadth target and the first with the 35B's
   asymmetric GDN (hidden 2560, 32 layers, intermediate 9216, GQA 16q/4kv
   hd256, GDN 16k/32v heads — the 35B row structure at a smaller K):
   `src/targets/qwen3_5_4b/` + `tools/convert/qwen3_5_4b/` (369 objects,
   355 text-core; tied embeddings; optional MTP block) + registry/engine/
   ingest/resources wiring. Weight parents: attn qkgv {10240,2560}, gdn qkvz
   {12288,2560} (q2048/k2048/v4096/z4096 — the exact 35B split), ab {64,2560},
   conv {4,8192}, mlp {18432,2560}, heads {248320|131072,2560}, mtp fc
   {2560,5120}.

   Because the 12288/8192-row parents match the 35B/2b registries, admission
   is K-driven this time (weight.k 2560 alongside 2048/1024) rather than
   row-driven:
   - w8 rowsplit checkers (attn + gdn) admit K 2560; per-K decode kernel
     instantiations (w8_k2048_decode trailing-K 2560) for attn 10240-parent,
     gdn 12288-parent, gdn decode-conv-snapshot (fused conv stays fused at
     T=1), swiglu launch_decode<16,9216,2560>; runtime-K MMA launchers
     (gdn/attn 35B branches now pass weight.k).
   - Route tables: attn 10240->kTarget2BRoutes, linear_add rows-2560
     k∈{4096,9216} -> kQ2BRoutes, swiglu kQ4BRoutes {1 DecodePairR16 /
     2-256 MmaR32C64 / 257+ MmaR64C128}, w8_dispatch case 2560 (heads) and
     case 5120 n==2560 (mtp fc).
   - GDN gating: k 2560 = 40 K-tiles, so split-32/16 are indivisible —
     `Bf16Gdn4BGeometry` {32 heads, 2560} + `k4BRoutes` (short-cols drops
     Split16->Split8), fused-norm decode runs split-8 for 4b (norm producer
     loop generalized from one-tile-per-split to kTilesPerSplit; the SplitK
     reduction was already generic), capacity gate widened to rows 2560.
   - GDN conv snapshot: wrapper split-dimension gates admit 12288/2560 (the
     value/z split derives from parent rows, already 35B-correct); the
     35B-baked split-K fused conv kernel is bypassed for k!=2048 (4b takes
     the generic project-then-conv path like 0.8b/2b; port tracked).
   - GQA: `Gqa4BGeometry` <16,4,2>. QHeads 16 collides with the 35B (16/2),
     so dispatchers resolve the pair through the KV-cache head count
     (`kv_heads_for_pair`); split capacity is scale-keyed and identical to
     the 35B's; KV append reuses the KVHeads-4 instantiation.
   - Frontend: the 4B tokenizer chat template is the family template with the
     default-thinking branch flipped (undefined enable_thinking -> think);
     the engine always renders with an explicit toggle, so the digest maps to
     the same ThinkingToggle semantics.

   Artifact converts from the HF checkpoint in 31.8 s (5.39 GB). E2E
   (sm_120): greedy answers exactly ("SUROGATE SERVE OK", stop-token
   finish). Bench vs the measured vLLM 4B board: decode ~160 tok/s vs
   100/115/162 (bf16/FP8/NVFP4) — +60%/+39%/tie-with-4-bit; prefill
   10.1k/13.6k/17.1k @472/962/1912 — ahead of bf16 everywhere, the usual
   structural ~14% behind FP8, NVFP4 ahead at scale on 4-bit compute.
   IMMA scaling verified (2B 37.8k / 2.2x traffic ~= 17k expected).

19. **Prefill campaign increments (4B-led, all targets benefit).** After the
   4B board landed, an nsys decomposition @1912 showed 77% of prefill is
   IMMA GEMM time at 155-206 TF/s. Increments shipped:
   - Wide-config gate relaxed n>=4096 -> n>=2560: the residual family
     (o_proj 2560x4096, down 2560x9216) was running the base config; probe
     241->214us / 519->441us. 4B@1912 +3%.
   - **Fused swiglu epilogue** (replaces #17's unfused pairing pass):
     `W8A8SwigluPairRowMap` interleaves gate/up (logical row 2i = gate i,
     2i+1 = up i) so each pair meets 4 lanes apart inside one mma fragment;
     a full-mask `__shfl_down_sync(4)` in a paired kernel tail joins them
     and the epilogue writes silu(gate)*up directly. No pair buffer (the
     18432xT bf16 workspace is gone), no pairing kernel, one fewer bf16
     round-trip (silu now sees fp32 straight from the int32 accumulators).
     Op tests pass at T {511,512,1024,1912} on both W8 profiles; greedy
     E2E answer unchanged. 4B@1912 +2.3%, @472 +4%.
   - Config-space closure (measured, 12 variants raced): BN256 (two warp
     layouts), BM32, single-barrier restructure, 3/4-stage pipelines, and
     4x/8x warp-N widening ALL lose to the shipped 2-stage wide config.
     The kernel is issue-bound; ~206 TF/s ~= 49% of the sm_120 int8
     ceiling is this design's local optimum. Further prefill structurally
     requires the FP8-e4m3 plane (FP16-acc MMA class) or the native 4-bit
     profile.

   Running 4B board after increments: prefill 10.5k/13.8k/18.0k
   @472/962/1912 (vLLM bf16 9.9/11.5/12.7, FP8 11.8/16.0/19.7), decode
   ~160 unchanged.

20. **Derived FP8-e4m3 prefill plane — vLLM FP8 matched/beaten in-class.**
   Measured first: an e4m3/F16-acc mma swap alone gains ZERO (the IMMA
   kernel is bound by its per-group software scale tail, not MMA rate).
   The win is FOLDING the scales away: at first large-T use of a W8G32
   parent the engine derives fp8_codes[row,k] = e4m3(w / rowmax) plus
   row_scales[row] (registry keyed by device pointer, event-ordered
   publication, VRAM guard at 2x plane size, SUROGATE_SERVE_FP8_PREFILL=0
   veto; artifact unchanged — decode stays int8-exact W8). Activations
   quantize per token to e4m3 at 448/xmax. The folded kernel
   (w8fp8_gemm.cuh, same staging as IMMA) chains the f16 accumulator
   across each 64-wide k-tile — bounded by 2*32*1.0*448 = 28672 < 65504,
   no satfinite clamping by construction — spills to f32 once per tile,
   and applies row_scale * x_scale once in the tail (the swiglu paired
   tail scales each half by its own row before silu*mul). Probe: -12..16%
   vs IMMA at every shape. Engine opts in at construction; op tests keep
   int8-exact numerics. CLI gains --prefill-warmup (one discarded request
   so measured runs exclude the one-time ~10 ms derivation, matching how
   a server amortizes it).

   4B steady state (idle 5090, batch 1): prefill 11,845 / 15,952 /
   20,355 @472/962/1912 vs vLLM FP8 11,838 / 15,985 / 19,743 — tie, tie,
   +3% — with decode ~160 vs their 115 (+39%). Quality class: FP8
   per-row (finer than vLLM's per-tensor). Greedy E2E answer unchanged
   with the plane on and off. Remaining ahead-of-us: NVFP4 prefill at
   962+ (4-bit compute; the native 4-bit profile is the answer) and a
   dedicated FP8-plane op test + mxf8f6f4 block-scale variant (hardware
   ue8m0 per-32 scales; possibly faster and finer than per-row).

21. **Native 4-bit profile, stage 1: NVFP4 prefill plane (opt-in).**
   `SUROGATE_SERVE_PREFILL_QUANT=fp4` derives an NVFP4 plane from W8G32
   (e2m1 nibbles + ue4m3 per-16 block scales + f32 per-row, two-level
   448*6 scheme) and runs sm_120a's mma.kind::mxf4nvf4.block_scale —
   hardware scale application, f32 in-core accumulation, W4A4 (acts
   quantize per token to e2m1 + per-16 ue4m3). Same registry/guard
   machinery as #20; decode stays int8-exact W8; quality class = NVFP4
   PTQ (the vLLM-NVFP4 checkpoint class), so the mode is an explicit
   opt-in, never a default.

   Hard-won kernel facts: the e2m1 m16n8k64 fragment equals the int8
   m16n8k32 fragment at the byte level (ldmatrix pattern carries over;
   K-tile = 64 elements = 32 bytes; smem row stride must stay 16B-aligned
   — 40 faults inside ldmatrix); SF operand mapping validated
   single-mma-exact vs a CPU reference (sfa = 4 k-group bytes for tile row
   8*(lane&1)+(lane>>2), sfb for token lane>>2, selectors all-zero); SF
   staging must ride the same cp.async commit group (a synchronous LDG in
   the stage cost +16..40%); the 512-thread config spills under a 2-CTA
   launch-bounds cap (+90% on every base shape) — MINCTA=1.

   Raw-kernel probe: 274-309 TF/s (-22% vs the FP8 folded kernel). Engine
   today: 10.9k/14.2k/20.4k @472/962/1912 — parity with the FP8 plane at
   1912, behind it below (heavier per-16 act quant, small-T fixed costs),
   and well short of vLLM-NVFP4's 35k long-prefill (cutlass-class tiles +
   TMA). Verdict: prefill-FP4 alone is not yet worth the quality trade —
   the profile's real payoff is stage 2, W4 DECODE kernels reading this
   plane (weight traffic 4.8 -> 2.7 GB/step: decode ceiling ~1.7x, past
   NVFP4's 162), plus a deep-tuned FP4 prefill pass for the 35k class.
   Greedy E2E answers exactly under fp4 mode.

22. **fp4 profile, stage 2: W4 decode kernels (UNVALIDATED — GPU pending).**
   Under the fp4 profile, decode now reads the derived NVFP4 plane too:
   `w4fp4_decode_kernel` (twin of w8_k2048_decode with half the weight
   traffic: one u32 of e2m1 codes per lane per 256-value phase, ue4m3
   per-16 scales shuffled from the low half-warp, f32 row scale at the
   end, same Output/Epilogue contract) plus a gate/up pair twin for the
   swiglu decode kernel. Wired for the 4B shapes: attn qkgv 10240/2560,
   gdn qkvz 12288/2560 (plain and fused decode-conv-snapshot), swiglu
   18432/2560. Weight traffic those cover: ~2.5 GB/step of the ~4.2 GB
   total.

   CUDA-graph integration needs no plumbing: program_impl already runs one
   EAGER warmup decode + synchronize before capturing, so the wrapper's
   lazy derivation happens there and capture bakes the plane pointers.
   Both plane registries got capture guards: a capturing stream may look
   up a finished plane but never derives (cudaMalloc) nor waits on
   external events.

   NOT yet covered: the SIMT decode family (o_proj/down at T<=4), lm_head,
   and the 0.8b/2b/35B decode K-branches (same pattern).

   VALIDATED (idle 5090): three fixes were needed on the way —
   - the graph-preparation memory check counted lazily-derived planes
     (1.53 GB) against the 12 MiB graph allowance; the registries now
     report their measured free-memory impact (cudaMemGetInfo around the
     mallocs, capturing the ~2 MiB/alloc page rounding a byte-sum missed)
     and program_impl excludes it;
   - the ldexpf-chain e2m1 decode ate the entire byte win (W4 kernels
     measured flat vs W8: 37.8 vs 40.3 us qkvz); replaced with direct
     fp32 bit assembly (e2m1 is a float format: exponent 126 + (m>>1),
     mantissa bit m&1, four ops total) —
   decode went 158-162 (W8) -> 151-155 (ldexpf W4) -> **190-192 tok/s**,
   +23% over the engine's own W8 line and **+18% past vLLM-NVFP4's 162**
   at the same weight class, with graphs on and the greedy answer exact.
   fp4 profile board: prefill 10.9k/14.2k/20.3k @472/962/1912, decode
   ~191.

   Stage 2b (same session): the SIMT-routed families joined at T=1 —
   o_proj/down and the 636 MB/token lm_head dispatch to the same
   w4fp4_decode_kernel at their compile-time shapes inside
   launch_w8_simt_r8_c4 (batch>1 decode keeps the multi-column SIMT
   path). **Decode 208-215 tok/s across the board points — +31% over the
   engine's W8 line and +30% past vLLM-NVFP4's 162.** fp4 board final:
   prefill 11.1k/14.3k/20.5k, decode ~211. Remaining fp4-vs-NVFP4 gap is
   long prefill only (20.5k vs 35.2k). Quality spot-check (5 diverse
   greedy prompts, fp8 vs fp4): identical answers on arithmetic/factual/
   listing, semantically-identical code (x*x vs x**2), same-set color
   ordering — and the two prompts the model gets wrong it gets wrong
   IDENTICALLY in both modes (model limitation, not quant-induced). A
   rigorous eval (GSM8K/IFEval class) remains open before calling the
   quality class settled. Batch>1 decode and non-4B targets keep W8
   decode paths (patterns established).

23. **fp4 stage 3a: 256-element K-tiles.** The stage-3 config race (real
   SF-plane loads, garbage data) found the fp4 GEMM's 64-element K-tiles
   pay a barrier pair + stage issue per single mma-K; quadrupling the
   tile (4 mma-K per staged tile, 2 stages, one barrier per tile,
   BKB_PAD = BKB + 16 = 144 — the 80-byte pad overlapped rows at 128B
   and faulted ldmatrix alignment at 40) wins at EVERY shape with ONE
   config: o_proj 214->114us, down 437->231, gate_up 872->543, qkvz
   585->346 (332-391 TF/s). In-engine fp4 board: prefill
   13.8k/18.6k/24.1k @472/962/1912 (+24/+30/+18%), decode ~210
   unchanged; greedy smokes exact. **fp4 now beats vLLM-NVFP4 at three
   of four points** (@472 +63%, @962 +7%, decode +30%); @1912 remains
   24.1 vs 35.2k. kt4-s3 exceeds the 99KB smem cap (compile-checked);
   deeper K-tiles or TMA are the next rungs.

   Stage 3b increment: the fp4 act-quant kernel read every activation
   from global twice, scalar, with branchy encodes; vectorized (two uint4
   loads per 16-group into registers, single read, packed u64 stores) it
   drops ~3ms/prefill — board 14.0k/18.9k/24.7k, decode unchanged, smoke
   exact. Fresh @1912 decomposition (79.5ms wall): fp4 GEMMs 42.1ms, GDN
   scan 10.4, attention 3.7, act quant now ~1.5, norms ~2.4, in-window
   idle 3.1, and ~9-10ms of wall outside the kernel span (request
   staging/final-sync structure — engine-level, unattributed by the
   layer-scoped NVTX ranges; a tiny-T floor probe is confounded by the
   A16 regime below T=224). Road to NVFP4's 35.2k (54ms): TMA-class
   GEMM staging (42 -> ~28ms), the scan pool, and that staging residue.

   CLOSURE (2026-08-27, after #24 defer + #25 cutlass): the host-bound
   residue is GONE as a side effect — nsys @1912 fp4: 49.4ms window,
   45.6ms kernel-busy, 3.7ms idle; host CUDA-API work is ~4.3ms (657
   launches at 4.4us + memsets/copies) fully hidden behind execution, and
   the one 43.7ms cudaStreamSynchronize is the host WAITING on the GPU.
   The killed 4-token tail slice and the cutlass path's fewer/bigger
   kernels were the fix. Prefill is GPU-bound again at ~92% window
   occupancy; the remaining levers are the 3.7ms of launch bubbles
   (prefill CUDA graphs would close them) and the GDN scan pool.
   Adapter-cost fear from the race bench also resolved: per-call cutlass
   host cost in-engine is tens of us (704 cuTensorMapEncodeTiled calls
   total 0.11ms) — the 3ms/call artifact was bench-loop-specific. The same tile-amortization
   does NOT transfer to the FP8/IMMA kernels (probed: -0..3%) — their
   64-element tiles are already 64 bytes/row, so the barrier cost per
   staged byte was pre-amortized 2x; the default profile's kernel stands
   at its structural optimum. Deeper fp4 tiles (512-element) exceed
   static smem at every viable BM/BN — the next fp4-prefill rungs are
   TMA/dynamic-smem staging and the non-GEMM pool (act quant, scan,
   host span).

24. **Deferred rewrite-checkpoint capture — the hidden 12 ms tail pass.**
   Segment-timing probes (SUROGATE_SERVE_PREFILL_TIMING=1, kept) showed
   every prefill ends with a second full-model pass over the last 4
   tokens: the frontend plans a rewrite checkpoint at prompt-4 (the
   thinking-suffix rewind boundary), the chunk loop must split there to
   materialize exact GDN state, and the 4-token tail re-reads every
   weight in the A16 regime (~12 ms at 4B — 15% of a 1.9k prefill, and a
   FIXED cost, so worst at short prompts). staging measured 0.3 ms and
   the final sync 0.007 ms — the long-standing "staging residue" was
   this tail plus host-issue overlap.

   SUROGATE_SERVE_DEFER_REWRITE_CHECKPOINT=1 (default OFF) maps
   CaptureNew -> DeferCapture: the checkpoint is simply not captured at
   prefill; a later rewrite request falls back to prefix recompute (the
   deferral validator is relaxed for the opt-in: the frontier may lie
   ahead of the reuse base). Behavioral trade: rewind-heavy thinking
   flows pay on rewind instead of every prefill. GATE PASSED (2026-08-27):
   ninfer_qwen3_5_4b_prefix_real_test runs in both modes — capture
   restores the response checkpoint; defer completes the same rewind via
   recompute fallback. Defaulting the flag ON is now safe engineering-
   wise; it remains an owner decision (latency profile of rewind flows).

   Measured with the flag (idle 5090, all three targets, decode
   unchanged, greedy exact): 4B default 16.6k/19.7k/23.2k @472/962/1912
   (+40/+23/+14%), 4B fp4 21.2k/24.4k/28.9k (+52/+29/+17%); 2B
   33.0k/42.7k/50.0k — now past vLLM-NVFP4 at EVERY point incl @1912
   (+12%); 0.8B 48.7k/69.9k/85.5k (2.2x vLLM-NVFP4 @1912). 4B standing:
   the default 8-bit profile beats every vLLM config at @472/@962; fp4
   closes @1912 to -18% (28.9 vs 35.2k).

25. **cutlass NVFP4 GEMM — the board is complete.** Per the owner's "just
   copy it from vLLM": the fp4 prefill GEMM is now the cutlass sm_120a
   blockscaled kernel class (recipe from flashinfer's template, Apache-2.0;
   cutlass v4.6.1 from the main project's FetchContent). Direct
   instantiation measures 679-813 TF/s at the engine shapes — 1.8-1.9x the
   hand-rolled kt4 kernel and faster than vLLM's own binary (gate_up 222 vs
   244 us); CTA 128x128x128, TmaWarpSpecializedCooperative, persistent
   scheduler.

   Integration: the derived plane gains a cutlass ATOM-layout ue4m3 SF
   copy with the per-row scale FOLDED in (alpha = 1; encode_rn of the
   product; row-major SF + row scales stay for the decode/kt4 paths);
   activations quantize through w4fp4_act_quant_atom (folded per-token
   scale, zeroed padding). Dispatch: residual family uses the epilogue's
   native beta=1 (C = D = residual); split2/split4/swiglu stage
   [tokens, parent] BF16 and run small split/pair kernels ([T,N] row-major
   D is exactly the engine's token-major layout). kt4 remains the
   fallback when the cutlass launch is unavailable. Workspace capacity
   grows to max(int8-A8, fp4-cutlass) at the four sites. Bench traps
   recorded: adapter init/run per call costs ~3 ms host unless
   initialize(args, nullptr, stream) with no workspace; hw_info.sm_count
   must be set.

   fp4 board (idle 5090, defer flag, greedy exact): prefill
   25,259 / 31,293 / 38,980 @472/962/1912, decode ~211. vs vLLM-NVFP4
   8,469 / 17,351 / 35,169, decode 162: 3.0x / +80% / **+11%** / +30%.
   **The engine now beats every vLLM configuration at every measured
   point on every shipped target in both quality classes.** Queued:
   re-run the 5-prompt quality panel on the cutlass path, tests for the
   atom-SF encoders, the deferred-rewind fallback test.

26. **Multi-arch scaffolding (owner directive: the serve engine must run on
   consumer RTX — sm_80/86/89/120/120a — and AMD later).** Capability
   ladder, gated at RUNTIME by device CC with graceful (and, for explicit
   requests, loud) degradation:
   - base W8 + int8 IMMA: sm_80+ in principle (Ampere int8 mma /
     cp.async / ldmatrix), compile-proven for sm_89;
   - FP8 derived plane: sm_89+ (w8fp8_plane_enabled now requires CC>=89;
     Ada f16-acc rate caveat still to price on hardware);
   - fp4 profile (NVFP4 plane, mxf4nvf4 mma, cutlass blockscaled, W4
     decode): sm_120a ONLY — w4fp4_plane_for refuses below CC 12.0, the
     asm is compile-guarded on __CUDA_ARCH_FEAT_SM120_ALL, and an explicit
     SUROGATE_SERVE_PREFILL_QUANT=fp4 on lesser hardware logs a fallback
     warning instead of silently serving another profile.
   Build: SUROGATE_SERVE_CUDA_ARCHS (default 120a) drives the serve
   targets; the upstream sm_89 port guards (#5-10) carry over, the
   27B W4A4 TMA archive keys its stub off the serve arch list, and the
   kt4 fp4 kernels' >48KB-smem bodies are arch-guarded so lesser-arch
   fatbins link. **The full engine now compiles and links for sm_89 with
   every post-port kernel (#17-25) in the tree** — sm_89 execution still
   needs Ada hardware (none on this host): route/occupancy retuning (the
   gating tables hardcode 170-SM residency counts) and the FP8-accum
   pricing remain. sm_80/86 and the AMD/HIP port are the next rungs; the
   plan/launcher dispatch split is the porting seam.

27. **Prefill CUDA graphs — bucket-captured chunk bodies (design of
   record; implementation in flight).** Goal: recover the ~3.7ms of
   inter-kernel bubbles per prefill chunk (nsys @1912 fp4: 49.4ms
   window, 45.6 busy — #23 closure) and delete ~4.3ms of host issue
   work (657 launches/chunk). Expected ~+5-7% @4B, more at 0.8B/2B.
   Mechanism mirrors the decode-graph idiom: bucket ladder at multiples
   of 128 up to prefill_chunk (GEMM tiles, GDN 64-chunks and attention
   row-blocks already round up internally, so 128-granular padding adds
   ~nothing the engine wasn't already paying); graphs captured per
   bucket on first use, the prefill_chunk bucket precaptured at load; a
   pinned PrefillIngress {base, actual_len} memcpy'd into a device
   mirror INSIDE the captured body (the OrdinaryDecodeIngress trick —
   src/dst baked, host rewrites fields before replay); ids via fixed
   pinned staging + in-graph H2D into the arena tensor (arena addresses
   are replay-stable per bucket: deterministic recipe sequence after
   work_.reset()). Attention needs NO changes — the prefill kernel
   derives per-query visibility from positions[0] on device, the
   envelope never reaches it, pad queries are causally downstream of
   every real row, and pad KV writes are overwritten by the next chunk
   or invisible to decode via lane lengths. Pad hygiene: one generic
   ops::mask_columns_zero(x, valid_dev) zeroes columns >= actual_len —
   applied to the residual post-embedding (keeps amax scales exact) and
   to f32 g/beta post-gating, where zero IS the identity state update
   (g is log-decay: 0 => decay 1; beta gates the rank-1 update: 0 =>
   none) — so the GDN chunked-scan kernels change NOT AT ALL. The one
   kernel-arg change: causal_conv1d_silu gains an optional device
   valid_len rebasing the trailing width-3 state snapshot to the last
   REAL columns. Lane independence: prefill computes GDN/conv state in
   a dedicated scratch slot (executor licenses one prefill at a time);
   lane->scratch once at prompt start (prefix-append), scratch->lane
   once at prompt end — two small per-PROMPT copies buy lane-agnostic
   graphs. Outside the graph per chunk: staging fill, ingress rewrite,
   final-chunk lm_head+sample, checkpoint epilogues, the sync.
   Eligibility: text-only, no MTP/multimodal/dflash/tap; ineligible
   modes run the unchanged eager body. SUROGATE_SERVE_PREFILL_GRAPH=0
   vetoes. Graph execs accounted against graph_allowance_bytes.
   Workspace contract unchanged (bucket recipe <= prefill_chunk
   high-water). Numerics: real rows exact vs eager modulo GEMM
   accumulation order at M=bucket vs M=len; parity gates are greedy
   token equality + the quality panel, not bitwise activations.

   SHIPPED + MEASURED (2026-08-25, idle 5090, --prefill-warmup, defer ON).
   Two capture landmines found and fixed on the way: (1) the fp4 cutlass
   alpha scalar's lazy init does a SYNC cudaMemcpy — invalidated the
   first capture at any cutlass-routed bucket (>=512); pre-warmed in
   prepare_graphs. (2) load-time precapture without a prior eager
   prefill pass baked the INT8 FALLBACK into the graphs: the decode
   warmup only derives planes for decode-routed weights, and the
   capture guard (correctly) refuses to derive mid-capture — nsys
   showed 64 IMMA GEMMs + W4-decode kernels inside the graph. Fix: one
   eager warmup prefill chunk at the full bucket inside the dummy-row
   window (deriving every prefill plane), THEN precapture all buckets.
   Load cost of warmup + 16 captures: ~+0.3s at 4B.
   Prefill, graph vs eager (same binary, SUROGATE_SERVE_PREFILL_GRAPH):
     4B  fp4: 26.5k/36.5k/41.1k vs 25.2k/31.2k/39.0k  (+5.1/+16.8/+5.5%)
     4B  fp8: 16.6k/21.4k/23.8k vs 16.6k/19.7k/23.1k  (+0.1/+8.7/+2.8%)
     2B  fp4: 46.7k/64.2k/79.7k vs 44.3k/58.8k/75.4k  (+5.3/+9.0/+5.7%)
     0.8B fp4: 58.5k/84.4k/109.7k vs 55.3k/76.6k/102.9k (+5.8/+10.2/+6.5%)
   at 472/962/1912; decode unchanged everywhere. Mid-length single-
   bucket prompts win most (whole bubble set removed, minimal pad);
   @1912 pays the defer split (two chunks, ~98 pad tokens). Suite: 91
   ctest green on GPU2 (frontend = pre-existing environmental skip),
   prefix capture+defer pass under graphs, binding smoke exact. Default
   ON for SpeculativeBackend::None targets; SUROGATE_SERVE_PREFILL_GRAPH=0
   vetoes; capture failure poisons the family and eager serves on.

28. **Multi-user campaign, Phase 1: batched small-T decode routes.**
   DIAGNOSIS (nsys, 8-user pure-decode 4B server): one kernel is 84% of
   the window — w8_rowsplit_gemm_mma_kernel, 115,107 launches at 141.5us,
   128 per round (32 layers x 4 GEMMs) — the runtime-shaped MMA tile
   serving T=8 where T=1 rides ~30us GEMV-class decode kernels. A batch-8
   round therefore costs 4.84x a solo round and C=8 aggregate saturates
   at ~354 tok/s @4B (vLLM: 3,390 via ~100-deep continuous batching).
   Weight-read arithmetic says batched rounds support ~3.3k tok/s at
   TODAY'S lane count. ROOT CAUSE: the fused-op families' exact-T split-K
   fast path (w8_small_t_mma_kernel, shape-templated, ~28.7us flat for
   T=1..8 at the 27B shape) was only INSTANTIATED for the 27B/companion
   (+ the 0.8b via PATCHES #13); the 2B/4B tables were never built, and
   their routes sent T=2..128 to the MMA tiles (the #16 comment said as
   much). FIXED HERE for attn_input_proj: 2B (5120x2048) and 4B
   (10240x2560, Output4<4096,1024,4096,1096-order q,k,gate,v>) exact-T
   tables (T=2..48) + kTarget4BRoutes + the 2B band flip. ALSO DONE in this
   pass: linear_swiglu 4B exact-T table (2x9216, k=2560; the 2B already
   rode the base band) + route flip, and linear_add SIMT band widened
   1..4 -> 1..16 for the small-target tables (runtime-shaped SIMT kernel,
   no new instantiation; exact-T bakes for 2560x{4096,9216} and
   2048x2048 are the refinement pass). gdn_input_proj DONE in the
   third pass: the conv-fused exact-T path (w8_small_t_mma under a
   GdnConvEpilogue) is geometry-templated (Rows/Hidden/QkvRows/Q/K/V/Z),
   with table sets for 4B (12288x2560, 35B split 8192=2048+2048+4096,
   z 4096), 2B (8192x2048, 6144=3x2048, z 2048) and 0.8B (8192x1024) in
   projection/snapshot/record forms (T=2..16) plus variant selection by
   (weight.n, weight.k) and the small-target route band 2..16 ->
   SplitKMmaDirect (T>=17 keeps the runtime-dim MMA; the medium kernel
   still bakes 35B geometry). linear_pair checked: MTP-only, not in
   ordinary rounds — no change. Phase-1 W8 route work is COMPLETE for
   every family a batch decode round touches; batch-T w4fp4_decode SHIPPED in the
   fourth pass: w4fp4_decode_batch_kernel<Rows,RowsPerCta,Output,
   MaxTokens,K> decodes each weight nibble once and fans it across up to
   MaxTokens resident accumulators (buckets 4/8/16 bound register
   pressure; runtime `tokens` inside the bucket), wired at the
   plain-linear W4 site (lm_head 248320x2560 + 2560x{4096,9216}) for
   T=2..16. The fused families keep the new W8 exact-T tables at batch
   for now — routing THEM onto W4 batch variants is a measured tuning
   decision (W4 halves weight bytes but the exact-T MMA kernels may
   still win on issue shape; bench on GPU return). VALIDATED (2026-08-25,
   GPU2): full ctest green (frontend = known environmental skip); 4B
   8-user pure decode 350.7 -> 908.6 tok/s (+159%, per-stream 44.3 ->
   116.5); 4B multi100 354 -> 865 (+144%, TTFT 28.9 -> 13.1s, 0 errors);
   0.8B multi100 1,255 -> 2,362 (+88%, per-stream 158 -> 300); solo
   decode 215 vs 214 = no single-user regression. Batch-8 rounds now
   cost ~1.85x solo (was 4.84x; ideal ~1.2x — the tail is the
   W4-at-batch family routing decision, attention/GDN small-t at batch,
   and sampling). Greedy C=1 vs C=8 flips one near-tie (EOS vs comma
   after an obeyed instruction; both valid) — pre-existing, no changed
   code runs at batch 1; a per-token logit parity harness is the
   rigorous follow-up. vLLM multi gap: 9.6x -> 3.9x @4B, 4.7x -> 2.5x
   @0.8B; the rest is scheduling (Phases 2-3). Then: batch-T w4fp4_decode
   (T=1 only today — fp4 profile batch rounds pay 2x weight bytes),
   decode-graph recapture is automatic at load. VALIDATE when GPU
   returns: per-op bench (ninfer_attn_input_proj_bench --tokens 1..8 at
   the small shapes), greedy parity C=1 vs C=8, multi100 board rerun —
   target >=1.3k tok/s @4B C=8 from this phase alone. Phases 2-4 follow:
   native step-level scheduler (vLLM POLICY: token-budgeted mixed rounds,
   chunked prefill interleave, admit-on-arrival — not vLLM code),
   kMaximumConcurrency 8->32+, board revalidation.

29. **Multi-user campaign, Phase 3a: concurrency ceiling 8 -> 16.**
   Reordered ahead of the scheduler: at closed-loop load, queue TTFT is
   throughput-bound (queue/lanes x service), so lanes move both numbers
   while interleave alone moves neither. kMaximumConcurrency = 16
   (api/types.h); every batch<=8 gate proved to be host-side validation
   mirroring the old constant (conv snapshot, GDN snapshot forms across
   w8/q4q5/nvfp4/fp8 plans, gqa workspace, kv_cache_append_prefix, swa,
   prepare_masked_block, the decode batch message) — the kernels are
   grid-scaled; all mirrors raised to 16. Exact-T decode tables and the
   conv-fused GDN path cover T<=16 by construction (PATCHES #28), so
   batch-16 rounds stay on the fast kernels. MEASURED (idle 5090, 4B
   fp4): 16-user pure decode 1,213.8 tok/s (908.6 at C=8), multi100
   1,141 (865; campaign total 354 -> 1,141 = 3.2x, vLLM gap 9.6x ->
   3.0x), TTFT p50 13.1 -> 9.1 s, 896/0 completions, load +~2s for the
   16-batch graph captures. Suite green. C=32 needs T=17..32 route
   coverage (linear_add SIMT band, GDN exact tables + conv ActiveCols,
   attn/swiglu tables to 32) plus the same mirror sweep; the per-stream
   drop at 16 (71.6 vs 109 at 8) says the T=16 kernels lean on the
   tuning tail — measure before going wider.

   TUNING TAIL, round 1 (same day): the batch-16 census showed (a) the
   lm_head at T=14..16 falling past the T<=13 SIMT band onto the runtime
   MMA tile at 1.76ms/round (352 GB/s on a 620MB read) — extending the
   head bands to T<=16 lets the SIMT launcher's fp4 gate serve it with
   the batched W4 kernel (310MB read); (b) linear_add's widened SIMT
   band reading weights ceil(T/4)x at T~16 (3.6s of a 12s window) —
   added the 4B o_proj (2560x4096) and down (2560x9216) exact-T bakes
   (Rows-parameterized launcher, T=2..16) with a kQ4B29Routes table.
   MEASURED: decode16 1,214 -> 1,391; multi100 1,141 -> 1,301.5 (TTFT
   7.9s, 1,012/0); solo 215 unchanged; suite green. Campaign 354 ->
   1,301 = 3.7x; vLLM gap 2.6x. Census leftovers for round 2: exact-T
   avg 48.3us at T=14..16 (schedule tuning), GDN snapshot 37.8us x
   21.5/round, 2B/0.8B linear_add bakes, W4-at-batch for the fused
   families (measured decision).

   Round 2 (same day): 2B/0.8B linear_add exact-T bakes — 2B o_proj
   (2048x2048), 0.8B o_proj (1024x2048) and down (1024x3584), the same
   Rows-parameterized launcher, T=2..16 route bands (the 2B down at
   k=6144 already rode the 27B-geometry bake). MEASURED at C=16: 0.8B
   multi100 2,362 -> 3,433 tok/s (TTFT 3.1s, 1,711/0, per-stream 216;
   solo 503) — 1.7x from vLLM's 5,958 (campaign start: 4.7x). Suite
   green.

   Round 3 (same day): kMaximumConcurrency 16 -> 32 (all mirrors swept
   again; GDN small-target projection tables and linear_add small bakes
   extended to T=32; attn/swiglu already reached 48; the conv-fused
   snapshot/record forms stay T<=16 — batch decode uses the Materialized
   conv path). MEASURED: C=32 is NOT the sweet spot yet — 4B multi100
   1,378 (vs 1,302 at C=16, +6%) and 0.8B REGRESSES to 2,986 (vs 3,433):
   the lm_head at T=17..32 falls onto the runtime MMA tile (the batched
   W4 buckets stop at 16) and per-round overheads grow. The 32 cap SHIPS
   (operators choose --max-concurrency; 16 remains the measured
   recommendation and the board's config). The 32-lane unlock is a vocab
   exact-T table (248320 x hidden) or W4 MaxTokens=32 bucket, plus a
   T=17..32 census pass. ATTEMPTED same day: a W4
   MaxTokens=32 bucket measured WORSE everywhere (0.8B 2,625, 4B 1,128 —
   32 accumulators spill and the extra local traffic loses to the tile);
   reverted. The vocab exact-T table remains the real unlock. BUILT same day: W8Q{4b,2b,08}VocabularyGeometry
   (248320 x 2560/2048/1024) exact-T tables at the 17..32 band only
   (T<=16 keeps the winning batched-W4 path at fp4), default production
   schedules (measured tuning pending), head routes gain the
   n==248320 && t<=32 -> launch_w8_small_t band. MEASURED: 4B at C=32
   now WINS — multi100 1,429 tok/s / TTFT 5.9s (vs 1,302 at C=16; vLLM
   gap 2.4x); the 0.8B keeps its C=16 optimum (3,433 vs 3,062 at 32).
   Per-model sweet spots are the recommendation: 4B -> 32, 0.8B -> 16.
   Suite green.

   Round 4 — the C=32 census found the real ceiling: kSamplerMaxColumns
   was 16, so batch 17..32 rounds fell off the multi-block sampler onto
   sample_row_kernel at 4.1ms per round (17% of the window; the shared
   layout authority meant the workspace plan consistently allocated
   nothing, so it degraded silently instead of crashing). Raised to 32:
   4B multi100 1,429 -> 1,736 (TTFT 4.8s), 0.8B 3,433 -> 4,733 (TTFT
   1.8s, C=32 now beats C=16 for BOTH models — the earlier "0.8B prefers
   16" verdict was an artifact of the broken sampler; the W4-32
   rejection stands, both sides of that A/B carried the same tax). vLLM
   gaps: 4B 1.95x, 0.8B 1.26x. Suite green.

   Fresh census at the new operating point (C=32, sampler fixed):
   w8_small_t_mma is now 52% of the decode window — 65,034 calls at
   72.9us average (~103/round at T=29..32) against a ~16us weighted
   weight-read floor: 4.5x headroom in the exact-T schedules, which at
   the 17..32 band are DEFAULTS (W8SmallTMmaDefaultSchedule TileCols
   24/32, KWarps 8; the vocab band likewise). NEXT ARC: measured
   schedule sweep for the small-target shapes at T=17..32 (KWarps 4 vs
   8, MinBlocks, ActivationStage/Cache — the bench harness needs the 4B
   shapes admitted, or A/B through the server). Also visible: GDN
   recurrent snapshot 10% (76us x12k), attention small-t 5.6% (128us at
   batch 32), sampler now healthy (173us/round). Remaining alongside:
   Phase 2 scheduler, fused-family W4 decision, Phase 4 board rerun.

   Chunked-prefill interleave measured: the executor's worker loop
   already strictly alternates decode rounds with prefill steps, so
   --prefill-chunk is a zero-code interleave knob — and finer chunks
   LOSE (multi100 @4B C=32: 1,736 at 2048 vs 1,307 at 512 vs 1,208 at
   256): prefill-blocking was not a loss term at this churn, and extra
   steps just add fixed cost. The Phase-2 scheduler's remaining value is
   admission shaping and open-loop TTFT, not closed-loop throughput; the
   throughput gap to vLLM now lives almost entirely in the exact-T
   kernel efficiency at the wide-token band (52% of window at 4.5x the
   weight-read floor) — which needs op-bench admission for the
   small-target shapes and kernel-level work on wide-T activation
   staging.

   GOAL ARC (owner: close the gap with vLLM). 27B rerun on the current
   build at C=32: multi100 363 tok/s (was 235 at C=8) vs vLLM 688 —
   and the census replays the known disease in the NVFP4 artifact's op
   families: fp8_mma_kernel (the A8 route's MMA) is 35% of the window at
   195us x ~30/round (~6x off the FP8 weight-read floor at T~28; every
   27B problem routes A8 at batch: attn>=12, gdn>=11, gateup>=5,
   residual>=25), nvfp4_w4a4_mma is 31% at 132.7us, the FP8 vocabulary
   head runs ~1TB/s (1.27ms/round — semi-optimal, ~40% headroom). THE
   27B ATTACK: tuned/exact-T tiles for fp8_a8_mma at T=17..32 (the
   fp8_a8_schedule.cuh + fp8_a8_plan.h band structure), then w4a4 tile
   shapes at the same band, then the vocabulary head's last 40%. Same
   playbook as PATCHES #28: census -> band routes -> measured kernels.

30. **Mixed-token rounds (goal arc, owner-framed; IN FLIGHT).** Owner
   diagnosis: vLLM wins via its scheduler + torch.compile fusion — a
   gapless, 100%-compute timeline. Our graphs already run 94-96% busy,
   so the structural difference is that vLLM's scheduler builds MIXED
   steps: prefill-chunk tokens and decode tokens in one forward, so
   every GEMM runs at effective T in the hundreds where tiles are
   efficient — while our standalone decode rounds run T<=32, where even
   correctly-routed kernels sit 3-6x off per-byte efficiency. The fix:
   TextContext::mixed_chunk (interface declared) — one forward over
   concatenated [prefill | decode] columns; GEMM/fused ops once over all
   columns; mixers split per slice (prefill attention/GDN-scan over the
   chunk columns, batch small-t attention/GDN-snapshot over the decode
   columns; a concatenated positions tensor lets rope/norms run once).
   All four mixer pieces exist as ops today. Remaining build order:
   run_layers_mixed body (attn_mix/gdn_mix re-authored with the split at
   the mixer core), program advance_mixed_round (marries PrefillContext
   + OrdinaryBatchContext), executor worker_loop swap (the ~1119
   alternation becomes one mixed round when both exist), workspace plan
   at chunk+C columns, eager first then graphs. Measurement gate: 4B
   multi100 vs the 1,736 standing number. Also open: verify (via ncu,
   /usr/local/NVIDIA-Nsight-Compute-2026.1) whether the FP8-A8 32-token
   tile actually entered the recaptured decode graphs — it measured
   neutral twice on the 27B, which contradicts the census arithmetic.

   SHIPPED + MEASURED (2026-08-26): the full mixed-round path is live —
   TextContext::mixed_chunk (GEMMs over concatenated columns, mixers
   split per slice), ProgramImplCore::advance_prefill_mixed (ordinary
   ingress staging + prefill card + batch sampling + zero-suffix
   finalize handoff via tail_hidden), executor run_mixed_round with a
   deferred first chunk (start_prefill_lane stages without advancing so
   single-chunk prompts reach the loop), and a saturating service-work
   projection (mixed rounds legitimately exceed the admission-time step
   estimate; floor 1 while active — the strict form killed the worker).
   Eager v1, chunk capped at prefill_chunk - batch. 4B multi100 1,736 ->
   1,872.8 (TTFT 4.6s, 1,403/0, only 3 classic chunks in the run); 0.8B
   4,733 -> 4,892.4 (TTFT 1.76s, 2,412/0); solo unchanged (532/stream
   0.8B); suite green. vLLM gaps: 4B 1.81x, 0.8B 1.22x. NEXT multipliers
   for the mixed line: capture mixed rounds as graphs (eager pays ~600
   launches/round), and per-op fusion along the owner's torch.compile
   observation.

   27B follow-ups (2026-08-26): mixed rounds lift it only to 385 tok/s
   (100 users) — prefill wasn't its constraint. The A8 batch-tile
   verdict is CORRECTED by template-arg census: BlockTokens=32 kernels
   are live at 99us average versus the production tile's earlier 195us
   (and the production schedule is BlockTokens=64, not 128 as first
   read) — the change worked at the kernel level; the earlier "neutral"
   reads were aggregate-level, where the 27B binds elsewhere: its
   nvfp4_w4a4 block, GDN at batch, the FP8 vocabulary head, and prefill
   throughput (~5.4k tok/s at 512-token prompts against admission
   churn). The 27B needs its own census-driven campaign; the 0.8B/4B
   line continues with mixed-round graph capture and epilogue fusion.

### sm_89 port status

With patches 5–10 the **entire tree compiles and links for sm_89**
(`cmake -DCMAKE_CUDA_ARCHITECTURES=89 -DNINFER_ALLOW_PORT_ARCH=ON`; first
sm_89 build in this code's history), producing `ninfer` and `ninfer-serve`
binaries. sm_120a rebuilt after the guards with zero behavior change (all
guards live in `<900`/`<1200` branches; PDL host fallback keys off runtime
CC). sm_89 EXECUTION remains unvalidated — no Ada GPU on this host; the q4
family has kernel-level parity evidence (design/p0-port-spike.md), the rest
needs a 4090: run the test suite, re-sweep route tables, and price the FP8
half-rate FP32-accum caveat.

### Known-environmental test results (no patch; documented)

CPU pass on this host (`CUDA_VISIBLE_DEVICES="" ctest`): 83/89 after patch 4.
- `ninfer_qwen3_6_frontend_test` aborts: upstream hardcodes the test resource
  `/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16/tokenizer.json`.
  Provide that tokenizer locally (or patch the fixture path) to enable it.
- `ninfer_linear_swiglu_{q4,w8,nvfp4,fp8}_test` report FAIL instead of SKIP
  when no CUDA device is visible (upstream skip-handling quirk); they belong
  to the GPU pass.
13. **qwen3.5-0.8b shape admissions (COMPLETE — first generated tokens).**
   Runtime whitelist + kernel-geometry extensions for the first new engine
   geometry (hidden 1024, 8q/2kv hd256, GDN 16/16x128, mlp 2x3584, vocab
   248320), found by the E2E worklist protocol and finished by op-level
   bisection after the artifact was proven correct with the ported Python
   reference (tools/reference/qwen3_5_0_8b — coherent text on CPU; includes a
   sequential-recurrence CPU fallback for GDN prefill since FLA is CUDA-only).

   Shape admissions: gqa_attention 8q/2kv (Gqa08Geometry<8,2,2>, Q-keyed
   dispatch clones; KV-append reuses Gqa35, GroupSize-independent);
   GroupSize==4 launch-tuple tables; W8 attn qkgv {5120,1024} decode/simt/mma/
   splitk (Hidden templated, kTarget08Launchers, W8SplitOutput4<2048,512,2048,
   512>); W8 gdn qkvz {8192,1024} decode/mma (W8SplitOutput2<6144,2048>,
   runtime z_offset conv epilogue) with kRoutes08 = {1:Decode, 2+:MmaR64C128}
   bypassing the 35B-baked splitk conv kernel; bf16 gdn_gating
   Bf16Gdn08Geometry{16,1024,64} + MMA/SplitK twins (SplitK capped 16), ab
   parent 32x1024; W8 linear k=1024 head routes (n in {248320, 131072});
   w8_k2048_decode.cuh gains a trailing K template parameter.

   **Root cause of the token-0 garbage run** (engine ran RC=0 but emitted
   argmax-0 forever while the Python reference was coherent on the same
   artifact): two W8 op families admitted the 0.8B shapes but routed them into
   compile-time 35B/27B-baked kernels —
   - linear_add: q08 {1024,2048|3584} fell into kK4096Routes, whose
     SplitKMmaExactT/MediumSplitK/DecodeR16 launchers are W8LinearGeometry
     <2048,{4096,6144}> instantiations with a 2048-row grid: OOB weight reads
     plus 1024 rows written PAST the residual tensor on every layer. Fixed
     with kQ08Routes over the runtime-shaped SIMT/MMA schedules
     ({1-4 SimtR8C4, 5-128 MmaR32C128, 129+ MmaR64C128}).
   - linear_swiglu: all three W8 families were baked 12288/6144/2048. The
     decode pair kernel wrote 6144 rows into the 3584-row output every decode
     step. Fixed by templating the decode kernel (Intermediate, K), passing
     runtime dims in the MMA launcher, and twinning the splitk exact-T table
     for (3584, 1024).

   Op-test coverage added for every 0.8B geometry (all green on sm_120):
   attn_input_proj q08 x8 T-cases, gdn_input_proj q08 (its W8 comparisons use
   a documented one-output-ULP criterion, kGdnInputProjW8UlpTolerance: the
   T=2 sample at row 6143 was root-caused to the op's defined
   dequant-to-BF16 weight semantics under 34:1 cancellation — the
   kernel-semantics oracle reproduces the GPU value exactly, 8.03449 ->
   8.0625, vs exact-weight oracle 8.00212 — so the family criterion was
   miscalibrated for a flip that dominates a small batch's norm, not the
   kernel wrong), gdn_gating_proj kQwen08 routes + norm cases, linear W8 head shapes,
   linear_add both q08 shapes, linear_swiglu q08 profile (registered in the
   harness), embedding d=1024, gqa_attention {8,2}, causal_conv1d_silu 6144ch,
   gated_delta_net 16/16 identity head-map, gated_rmsnorm 16-head.

   E2E: `surogate-engine-cli <0.8b artifact> --greedy` answers exactly
   ("SUROGATE SERVE OK"; thinking + no-thinking modes both coherent),
   stop-token finish. Perf untuned (q08 routes favor correctness over
   measured tiles; decode measured only on a contended GPU so far).

## 31. Mixed-round CUDA graphs + graphed-prompt state-slot fix (2026-08-26)

The bd08m census showed 16.2% device idle at the 0.8B under 100-user mixed
load: every round with a live prefill lane ran the eager mixed body
(~600 launches with 2-5us bubbles). Mixed rounds now capture and replay as
CUDA graphs, keyed (chunk_bucket x 128, batch bucket {8,16,24,32}) in the
PATCHES #27 family (`mixed_key = (chunk<<8)|batch`, disjoint from plain
buckets). Prefill-side padding reuses the #27 discipline (ingress `valid`,
pad-aware conv, g/beta zeroed over the prefill window only — a sliced
`mask_columns_zero`, no new op). Decode-side padding is entirely host-side:
pad ingress rows duplicate row 0, so a pad column recomputes that lane's own
update and every state/KV write lands as identical bytes. The decode
envelope bakes {1, kv_capacity} (worst-case launch geometry, same as the
prefill graph side). Epilogues (scatter, sample, egress) stay eager at the
real row count.

Also fixed while wiring: `advance_prefill_mixed` configured the lane state
slot while graphed prompts (`staged.use_graph`) run all other chunks on the
shared scratch slot — multi-chunk prompts interleaving mixed and classic
chunks split their GDN state across two slots (benches never hit it:
512-token prompts are single-chunk). The mixed card now follows the same
ternary and the mixed completion path copies scratch back to the lane slot.

Measured (100 users, 512/128): 0.8B 4,892 -> 5,131 tok/s (vLLM 5,958, gap
-18% -> -14%); 4B 1,873 -> 1,902 (compute-dominated rounds, launch tax was
small). Census after: 100% of kernels from graph nodes, but idle is still
~18% at the 0.8B — the residual is the serial per-round host path
(sync -> egress read -> bookkeeping -> stage -> launch), ~1ms/round of dark
device. That is the actual scheduler gap vs vLLM's async staging; next arc
is round chaining (stage round N+1 before consuming round N's egress).

## 32. Decode round chaining (2026-08-26)

After #31 the 0.8B still idled ~18% under 100-user load with 100% of
kernels in graphs: the gap moved between rounds, into the serial host path
(sync -> egress -> bookkeeping -> stage -> launch, ~1.3ms per program
call). This entry builds the chaining infrastructure: a chained flavor of
every ordinary decode graph (no ingress H2D; in-graph chain tail copies the
sampled tokens into the next round's inputs and advances both position
rows via a device one-scalar), a burst driver that launches K rounds
back-to-back with per-round egress copies via stream host functions and a
single synchronize, and a generalized non-speculative resolve
(produced=K, truncation at terminal with ledger/identity trim; a
partial-terminal lane is not retained for prefix reuse because its
resident GDN state ran the full burst). Round 0 runs the classic captured
body plus an eager chain tail — without it round 1 replays round 0 and
the stream is garbage (found as an invalid-UTF-8 fatal).

Scheduling: admission rides the round cadence (one attempt per GPU unit),
so the burst length adapts — 1 while a prefill is staged or a queued
request could admit into a free lane, 3 while the queue waits on full
lanes, 8 when nothing waits. A fixed burst of 8 capped admissions at
~25/s and collapsed the 0.8B to 3,084 tok/s (fast lanes, few of them);
the adaptive policy restores 5,134. Mixed rounds now count as decode
units so admission may follow them directly.

Measured (0.8B, 100 users): 5,134 tok/s — neutral at this churn (vLLM
5,958). Segment timing (SUROGATE_SERVE_ROUND_TIMING=1, per 5s): mixed
calls 2,050ms/204, decode burst calls 2,800ms/305 (bursts of ~3 engage),
executor resolve tails are trivial (preview 20ms, append 50ms, stats 0).
The residual host serial is the program call chain sync -> resolve ->
membership -> launch, which is load-bearing: the next lever is chaining
across resolve boundaries (launch round N+1 on chained device state
before consuming round N's egress; host functions check EOS ids to drain
early; membership refresh lags one round), i.e. the vLLM async-scheduler
equivalent. The chain tail, egress host functions, and generalized
resolve built here are the pieces that design needs.

## 33. Marlin W8 GEMM vendored (2026-08-26) — Phase A

The 4B census under the current engine: `w8_small_t_mma` is 47% of device
time at 328-630 GB/s (the exact-T split-K kernels parallelize K only
across warps, so small-N shapes raise ~160 CTAs on a 170-SM card).
Routing the band to the R32C96 tile made it worse (1,425 tok/s, reverted)
— #28's finding holds. Per the owner's directive to borrow what wins,
vLLM's Marlin is vendored from study/vllm (Apache-2.0, IST-DASLab /
Neural Magic lineage) under ops/linear/marlin/: the kernel template stack
unmodified in vendor/ (scalar_type.hpp carries a torch-free STD_TORCH_CHECK
shim), instantiations generated for (BF16 x kU8B128/kFE4M3fn x group_blocks
{2,-1,8}) at stages 4, the dispatch adapted from marlin.cu to a raw-pointer
entry (marlin_gemm_bf16), and a load-time repack path that takes the serve
W8G32 residency (codes [N,K] int8 row-major, scales [N,K/32] FP16) through
GPTQ packing (codes ^ 0x80 -> [K/4,N] u32) into the Marlin B tiles plus
transposed/permuted BF16 scales. The engine's [K,T] column-major
activations are byte-identical to Marlin's row-major A, so no transposes
anywhere.

Measured on the 4B decode shapes (ninfer_marlin_w8_bench, one-hot
correctness max_rel 0.006 = BF16 scale rounding):

  shape              engine kernel   marlin    speedup
  down    2560x9216      71.8us      31.2us     2.30x
  gate_up 18432x2560     94.4us      42.1us     2.24x
  gdn_in  12288x2560     67.8us      35.7us     1.90x
  attn_in 10240x2560     55.9us      29.5us     1.89x
  out     2560x4096      31.6us      20.7us     1.53x
  vocab   248320x2560   1008.8us    694.2us     1.45x

These shapes are ~53% of 4B device time. Phase B (next): a Marlin weight
plane built at load before the decode-graph captures (the w4fp4/w8fp8
plane precedent), family dispatch for the T=17..48 band with fused
epilogues run separately, and the FP8 variant for the 27B's
fp8_small_t offenders (5120x6144 / 5120x17408 at 412-503 GB/s).

## 34. Marlin band wired into the serve families (2026-08-26) — Phase B

The #33 kernels now carry the T=17..48 decode band for W8 weights:
plain linear (vocab head), linear_add (residual add as its own pass),
linear_swiglu (silu_mul over the [2*out, T] result), and gdn_input_proj
(two strided copies for the [qkv | z] row split — kilobytes next to the
GEMM's tens of megabytes). A derived-plane registry (marlin_plane.h,
same discipline as the FP8/FP4 planes: process-global, keyed by the
device codes pointer, VRAM-guarded, never derives on a capturing stream)
holds the repacked B tiles and permuted scales; shared scratch (gemm
output, fp32 reduce buffer, locks) grows during warmup and is frozen
after capture so graphs may bake its addresses.

Two integration bugs, both found by measurement:

- Planes never derived, so the captures baked the old kernels and the
  4B measured unchanged (1,899 vs 1,902). The pre-capture warmup ran at
  batch 1, below the band; it now also runs one band-sized round.
- The 0.8B stream turned to garbage (invalid UTF-8) while the 4B stayed
  clean. Marlin's atomic-add reduce accumulates into C and needs a
  zeroed output, and it engages only when ceil(M/64)*N <= 2048 — true
  for the 0.8B's N=1024 projections, false for every 4B shape. vLLM
  keeps this path off by default for the same reason
  (VLLM_MARLIN_USE_ATOMIC_ADD); the serve integration does too.

Derived planes duplicate the W8 residency (~3 GB at the 4B), which the
graph-allowance accounting counted as graph memory until
marlin_plane_bytes() joined the same subtraction the FP8/FP4 planes use.

Measured, 100 users, 512/128: 4B 1,902 -> 2,118 tok/s (vLLM 3,390, gap
-44% -> -38%); 0.8B 5,134 -> 5,107 (neutral: its band shapes are small
enough that the exact-T kernels were already competitive). Suite green
apart from the known environmental frontend failure.

Band floor: measured rejection. Lowering it below 17
(SUROGATE_SERVE_MARLIN_MIN_T, kept as a sweep knob) breaks decode-graph
instantiation with cudaErrorGraphExecUpdateFailure: Marlin picks its
kernel from thread_m_blocks = min(ceil(M/16), 4), so batches 1..16 and
17..32 land on different kernels and the family's exec update rejects
the changed node. A floor of 17 keeps every band batch on one kernel.
Supporting a lower floor needs per-batch topology classes for the
Marlin nodes, not a knob change.

## 35. Mixed-round batch bucket: silent truncation above 32 lanes (2026-08-26)

Found while raising kMaximumConcurrency to 64 (the 100-user TTFT gap is
queueing: 100 clients against 32 lanes gives 4.0 s at the 4B against
vLLM's 0.24 s). The raise needs six batch-32 host mirrors lifted
(gqa_attention workspace, GDN conv/snapshot plans, causal conv, KV
append-prefix, masked block, SWA) and then still corrupted the token
stream. Cause, found by inspection rather than measurement:
`PrefillGraphFamily::batch_bucket_for` (PATCHES #31) ended in a fixed
`: 32`, so a mixed round with more than 32 decode rows ran its graph
with 32 decode columns while the bookkeeping read egress for every row
— the uncovered rows committed whatever was left in the egress buffer,
which surfaces as invalid UTF-8 in the stream.

The bucket now rounds up to the next multiple of 8 and clamps to
kMaximumConcurrency, and `advance_prefill_mixed` throws if a bucket ever
lands below the row count, so the failure mode cannot come back silently.
At the current ceiling of 32 the buckets are unchanged (rows 17..24 ->
24, 25..32 -> 32), so this is a no-op today and a prerequisite for the
raise.

Also from that attempt, kept: the Marlin band ceiling drops to 32
(PATCHES #34 carried 48). Marlin selects its kernel from
thread_m_blocks = min(ceil(M/16), 4), so a band crossing a 16-token
boundary puts different kernels into decode graphs that share a topology
class and cudaGraphExecUpdate rejects them (measured:
cudaErrorGraphExecUpdateFailure at both a lowered floor and a raised
ceiling). 17..32 is exactly one class. Widening the band later means
padding M to a fixed width inside the Marlin call — its cost is flat
from M=24 to M=32, so the padding is close to free — not moving the
bounds.

Still open for the ceiling raise: with the bucket fixed, 64 lanes needs
re-verification on hardware (the six mirrors are reverted to 32 in the
tree so the shipped configuration stays the proven one).

## 36. One bound for batched work (2026-08-26)

The #35 ceiling attempt cost a full cycle to hardcoded mirrors: eleven
copies of the literal 32 across the ops layer, found one error at a
time, and the ones that clamp rather than throw corrupt the stream
instead of failing. Ops cannot see the serving API's
kMaximumConcurrency, which is why the mirrors existed.

core/limits.h now holds kMaximumBatchColumns and every batched op spells
its bound with it (gqa_attention, GDN input/conv/snapshot plans across
w8/fp8/nvfp4/q4_q5, gated_delta_net, causal_conv1d_silu,
kv_cache_append_prefix, prepare_masked_block, swa). A static assertion in
the engine binds it to kMaximumConcurrency, verified to fire when the
two diverge. Raising the ceiling is now: change two constants, and the
compiler names any op that has not kept up.

No behavior change at 32.

## 37. Marlin runs at a fixed M (2026-08-26)

Band calls now pad the round's rows into a zero-filled A of kMarlinFixedM
rows and run the GEMM at that width. Marlin picks its kernel from
thread_m_blocks = min(ceil(M/16), 4), so with a width-dependent M the
decode graphs for different batch sizes hold different kernels and
cudaGraphExecUpdate rejects them (#35). Pinning M removes the coupling:
one kernel for the whole band, whatever the round's width.

It is nearly free. Marlin's cost is flat across the band (gate_up 42.1 us
at M=24 vs 42.4 at M=32), the A copy is t*k*2 bytes (164 KB at the 4B's
widest), and C comes out [kMarlinFixedM, n] row-major whose first t rows
are bit-for-bit the caller's [n, t] result — so the fused families read
them in place and only the plain-linear path copies out.

Widening the band is now raising kMarlinFixedM rather than splitting the
range, which is what the 64-lane ceiling needs. Unmeasured: the card was
returned to its owner before this could run. It compiles, and the
correctness bench covers the kernel underneath it; treat the band as
unverified until a 100-user run confirms both models.

## 38. Marlin FP8 residency path (2026-08-26) — groundwork, UNMEASURED

The 27B is the widest gap (385 vs vLLM's 688) and its census says why:
FP8-class GEMMs are 47% of its device time, with the exact-T decode
kernels (5120x6144 and 5120x17408 at T=17/18) running 412-503 GB/s
against the 926 GB/s the A8 batch tile reaches on the same class of
problem in the same profile.

marlin_repack_fp8_row takes the serve FP8 residency (codes [N,K] e4m3
row-major, scales [N] BF16 per output channel, the RowScale layout
validate_fp8_weight enforces) into Marlin B tiles plus channelwise
scales: the pack is the plain [N,K] -> [K/4,N] regroup with no bias
(FP8 dequant reads the raw byte), the scales take the 32-wide single
permutation, and the exponent bias folds into them — e4m3's 4-bit
exponent against BF16's 8 gives 2^(2^7 - 2^3) = 2^120, matching vLLM's
fp8_fused_exponent_bias_into_scales. marlin_fp8_plane_for and
marlin_fp8_run mirror the W8 pair through the same registry, scratch and
fixed-M discipline, dispatching Marlin's kFE4M3fn kernels at
group_blocks = -1 (the instantiations were generated in #33).

Wired into linear_add (5120x6144, 5120x17408) and linear_swiglu
(34816x5120), the two families that carry most of the 27B's FP8 decode
time, and ninfer_marlin_w8_bench now covers the FP8 shapes with the same
one-hot check that caught the W8 packing (each output must equal exactly
one code times its channel scale). STILL UNMEASURED — no GPU was
available for any of it. The one-hot bench is the gate: it must pass on
all five 27B shapes before the 27B is served through this path, and the
wiring falls back to the existing kernels whenever the plane declines,
so a failed derive degrades rather than breaks.

Review pass on the blind-written code (no GPU to test it, so it got read
instead). Two defects fixed before they could reach hardware: the FP8
call sites read the shared scratch before the plane derive that
allocates it, so the first band call in a process always fell back
(harmless in the end, because capture happens after the warmup, but
wrong and fragile) — marlin_fp8_scratch_for now derives first and
returns the scratch after, matching the W8 pair; and both run paths now
decline when the activation block is not contiguous, since the padded-A
copy treats [k, t] as row-major [t, k] and a strided view would copy the
wrong bytes with no error.

Mixed-round bucket ladder coarsened to 8, 16, then multiples of 16. Each
bucket is a separately captured graph of the whole layer stack, so the
8-wide ladder above would have doubled capture time and graph memory at a
64 ceiling (eight buckets instead of five). Padding a round to a 16
boundary costs a handful of decode columns against a prefill chunk of
several hundred. At 32 lanes the ladder is 8, 16, 32 where it was
8, 16, 24, 32 — one graph fewer, and rounds of 17..24 now pad to 32.

## 39. Measured: the queued levers, three outcomes (2026-08-26)

Correctness gate first, and it passed everything including the code
written without hardware: the fixed-M W8 band and the FP8 repack both
clear the one-hot check on all seventeen shapes (max_rel 0.0036-0.0071,
which is BF16 scale rounding). The 32-wide scale permutation and the
2^120 exponent bias were right.

Fixed-M band, SHIPPED: 0.8B 5,013 and 4B 2,076 tok/s at 32 lanes, zero
errors, zero fatals — correct and throughput-neutral, so #37 loses its
unmeasured label.

64 lanes, MEASURED REJECTION, reverted: the 4B falls to 1,802 tok/s from
2,076. TTFT does improve (2.5 s from 4.0 s) and errors stay at zero, so
the queueing theory was right, but throughput loses more than admission
gains: per-stream decode halves (29 vs 67 tok/s) because every batch
above 32 drops off the Marlin band onto the kernels Marlin was adopted
to replace, and capture OOMs at auto KV capacity because 64 lanes double
the decode-graph set. Raising the ceiling only pays together with
kMarlinFixedM = 64, which costs every band call the wider M — worth
measuring, not worth assuming.

FP8 planes, now opt-in behind SUROGATE_SERVE_MARLIN_FP8=1: the 27B OOMs
during graph capture with them on, at auto KV and even under a
quarter-of-device budget, because a plane duplicates the residency it
accelerates and the 27B's residency is most of the card. With them off
the 27B serves normally (370 tok/s over a 40 s window against its 385
baseline). The kernels are validated and 2.4-2.7x faster than the
exact-T FP8 kernels on the shapes that bind it (down 5120x17408: 67.5 us
vs 185.4; out 5120x6144: 31.4 vs 76.4), so the win is real but needs a
residency-REPLACING path — repack at load, free the original — rather
than a duplicating plane. That is the 27B's next arc.

## 40. The band width follows the lane count (2026-08-26)

#39 rejected 64 lanes because batches above 32 fell off the Marlin band.
Raising kMarlinFixedM to 64 with the ceiling confirms the diagnosis and
inverts the result at the 4B: 2,234 tok/s at 64 lanes against 2,076 at
32, with TTFT halved (2.04 s from 4.00 s) and no errors. The same build
costs the 0.8B 11% (4,441 against 5,013) — its shapes are narrow enough
that paying a 64-wide M on every band call outweighs the wider batch.

So the width is not a constant. marlin_set_fixed_m takes the engine's
runtime lane count at construction (32 lanes -> M 32, above -> M 64) and
the band ceiling follows it, which lets one binary serve the 0.8B at 32
lanes and the 4B at 64 and give each its best: measured together after
the change, 0.8B 4,990 and 4B 2,234, zero errors and zero fatals in both.

Against vLLM: 0.8B 4,990 v 5,958 (-16%), 4B 2,234 v 3,390 (-34%), 27B
370 v 688 (-46%). The 4B is the cell that moved; note its 64-lane run
needs an explicit --kv-capacity (auto sizes the cache before the doubled
decode-graph set is accounted for and capture OOMs), which is the next
thing to fix in the planner rather than in the flags.

Lane sweep, 4B: 32 -> 2,076; 64 -> 2,250; 96 -> 2,024. Sixty-four is the
optimum. Past it the KV cache has to shrink to fit the decode-graph set
(98,304 tokens at 96 lanes against 131,072 at 64) and the wider M wastes
more on rounds that never fill it, so both ends of the trade turn
against the raise. Final confirmation over 60 s: 4B 2,250 tok/s, TTFT
p50 2.04 s and p95 2.05 s, 1,124 requests, zero errors, zero fatals.

Standing against vLLM: 0.8B 4,990 v 5,958 (-16%), 4B 2,250 v 3,390
(-34%), 27B 370 v 688 (-46%).

## 41. Continuous admission (2026-08-26) — UNMEASURED

Lever three of the structural set. The worker loop admits at most one
request per GPU unit, which made sense when admission ran the first
prefill chunk itself. Since #30 it does not: with the deferred first
chunk an admission only stages the prompt and leaves the prefill to a
mixed round, so it is CPU-only work being rationed at the cadence of GPU
rounds. That rationing is what the 4B's TTFT shows — 2.04 s against
vLLM's 0.24 s at the same 100 clients, with vLLM admitting continuously
into ~100 slots.

The loop now keeps admitting while the queue holds work and a lane is
free, stopping on the first admission that runs a GPU unit (a vision or
MTP prompt, or any shape the deferred path does not cover) so a single
iteration cannot monopolise the device. The bound is max_concurrency
attempts.

Not measured — GPU2 went back to its owner before this could run. The
gate is a 100-user 4B run at 64 lanes: TTFT should fall toward vLLM's
while aggregate throughput holds at 2,250, and any regression in
throughput means admission is now stealing from decode and the loop
needs a per-iteration cap below max_concurrency.

Burst cap follows: #32's adaptive policy dropped to three rounds whenever
the queue held work, because each GPU unit bought exactly one admission
and long bursts starved lane refills (a fixed eight collapsed the 0.8B to
3,084). Continuous admission removes that coupling — a queue backed up
behind full lanes has nowhere to admit — so a full burst there costs
nothing but saves host round-trips, which is the same quantity the async
scheduler targets. Free lanes still take a single round so the next
iteration refills them. UNMEASURED; gated on the same 4B run, where a
throughput regression would mean the two changes interact and the cap
belongs back at three.

## 42. What the gate and the lane sweep actually proved (2026-08-26)

Continuous admission (#41) and the lifted burst cap measure NEUTRAL:
4B 2,216 against a 2,250 baseline, 0.8B 5,015 against 4,990, no errors
in either. Neither was wrong, both were aimed at the wrong thing, and
the runs say why.

TTFT is not a lever here. It reads 2,038-2,040 ms in every single
configuration — 64 lanes, 100 lanes, prefill chunk 1024/2048/3072,
admission rationed or continuous. That constancy is the tell: with 100
closed-loop clients against L lanes and a ~3.5 s request, the wait is
just (100-L)/L x 3.5 s, which is 2.0 s at 64 lanes. TTFT is a derived
quantity of the harness, not a property to optimise. Only throughput
moves.

The burst cap barely engaged either: under sustained load a prefill lane
is always staged, and the policy forces burst_limit = 1 whenever
prefill_lane_ is set. Raising the ceiling for full lanes changed a path
the load never takes.

Lane sweep, extended: 32 -> 2,076, 64 -> 2,250, 96 -> 2,024, 100 ->
2,002, and at 100 lanes the KV capacity makes no difference at all
(131,072 and 98,304 both give ~2,000). KV was never the constraint.
Beyond 64 the loss is the padded M: kMarlinFixedM rounds 100 up to 128
because Marlin's own M-loop splits at multiples of 64 (a remainder chunk
picks a different thread_m_blocks and would break graph stability), so
every band call at 100 lanes computes 128 rows to use 100. Sixty-four
lanes with a 64-wide M is a genuine local optimum for this design.

Which leaves per-round efficiency as the only remaining axis, and the
workload shape says where: at 512 prompt tokens for 128 generated, this
benchmark is 4:1 prefill-to-decode by token count, so prefill dominates
the wall clock. The decode-side work (Marlin, lanes, admission, bursts)
has been mined out; the next real gain is prefill throughput, which is
also where the resident-Marlin change pays a second time by letting
Marlin serve prefill instead of the current cutlass/fp4 path.

## 43. One prompt per mixed round is the 4B's ceiling (2026-08-26)

Round timing at the 64-lane optimum, per 5 s: mixed 2,980 ms over 90
rounds (33 ms each, 60% of the wall), decode 1,950 ms over 90 rounds
(21.7 ms each, 39%), executor tails 60 ms total. The loop alternates
one mixed round with one decode round.

The arithmetic that matters: a mixed round admits up to
prefill_chunk - batch = 960 columns, but it carries exactly ONE prompt,
because there is a single prefill lane. At 512-token prompts it
therefore uses 512 of its 960 columns and retires one prompt per round.
Ninety mixed rounds per five seconds is 18 prompts/s is 9,216 prefill
tok/s — which is the 8,900 the benchmark actually demands, so prefill is
running exactly at its structural limit while leaving 45% of each mixed
round's column budget unused.

Marginal cost confirms the headroom is real: a decode round moves 64
columns in 21.7 ms, a mixed round moves 576 in 33 ms, so 512 prefill
columns cost 11 ms on top of a round we were paying for anyway. Filling
the remaining 448 columns with a second prompt is close to free.

This also explains why #41 and #42 came back neutral. Admission cadence
and burst length never mattered: the queue drains at one prompt per
mixed round regardless of how many requests are admitted or how many
decode rounds are chained behind them. Prefill chunk size does not
change it either (1024/2048/3072 all measure the same), because the
limit is prompts per round, not columns per round.

The fix is multi-prompt mixed rounds: let one round carry the tail of
one prompt and the head of the next, up to the column budget. That means
a prefill lane set rather than the single prefill_lane_, and a
mixed_chunk that takes a list of (lane, span) rather than one prompt.
It is the largest remaining decode-side lever at the 4B and it is a
scheduler change, not a kernel change.

## 44. The sampler cap went stale and cost 15% of the device (2026-08-26)

nsys on a 64-lane 4B decode round: sample_row_kernel at 828 ms of a 5.4 s
busy window — 15.3%, 202 calls averaging 4.1 ms. That is the slow
fallback path, taken because kSamplerMaxColumns was still 32 while the
lane ceiling had moved to 64: sampler_multiblock_ok rejects cols > cap
and the launcher quietly drops to sample_row. Exactly the failure #29
fixed at 16 -> 32, reintroduced by moving the ceiling without moving the
cap. It is now kSamplerMaxColumns = kMaximumBatchColumns so the two
cannot drift again.

Measured, 100 users, 512/128:
  4B  64 lanes: 2,250 -> 2,571 tok/s (+14%), TTFT 2.04 s -> 1.77 s
  0.8B 64 lanes: 6,003 tok/s, TTFT 0.77 s

The 0.8B number BEATS vLLM (5,958). It also retracts #40's conclusion:
the earlier "64 lanes costs the 0.8B 11%" measurement was taken with the
broken sampler, so the regression was the fallback, not the width. With
the cap fixed the 0.8B gains 20% from 64 lanes (5,021 -> 6,003) rather
than losing 11%.

Standing against vLLM: 0.8B 6,003 v 5,958 (+0.8%, AHEAD), 4B 2,571 v
3,390 (-24%), 27B unmeasured since the fix.

Lesson worth keeping: three separate times now, a width constant that
silently degrades instead of failing has cost more than any kernel
optimisation gained. The remaining ones should be tied to
kMaximumBatchColumns or made to throw.

## 45. Batch-aware KV splits for decode attention (2026-08-26)

The decode attention grid is (KVHeads, splits, batch), but the split
policy only ever looked at the window — it had INT8 special cases and
nothing for batch. At the 4B's ~640-key window it asks for ~10 splits,
which at 64 lanes and 4 KV heads is 2,560 CTAs, roughly fifteen waves on
170 SMs. The extra splits buy no parallelism the batch dimension does not
already provide; they multiply the partial-buffer traffic that the
reducer then reads back, which is why the kernel measured 265.9 us
against a ~102 us KV-read roofline.

Splits are now clamped so (KVHeads x batch x splits) targets two waves.
The value stays constant per (envelope, batch), which is what a captured
decode graph needs. 4B 100-user: 2,571 -> 2,629 tok/s, TTFT 1.77 -> 1.72
s, zero errors.

Standing: 0.8B 6,003 v vLLM 5,958 (AHEAD), 4B 2,629 v 3,390 (-22%).

## 46. Session standing (2026-08-26)

  model  surogate   vLLM   gap
  0.8B      6,003  5,958   +0.8%  AHEAD
  4B        2,629  3,390   -22%
  27B         370    688   -46%

The 0.8B is past vLLM. The 4B closed from -45% at session start to -22%.
The 27B is unmoved and now has a clean diagnosis: it cannot take the
lane raise that paid on the other two — 64 lanes wants 23.3 GB of
runtime reservation against 12.76 GB free after its weights, because
per-lane GDN state at that model size dominates — so its concurrency is
memory-bound at 32 and its deficit is entirely compute. That is the FP8
residency work: Marlin FP8 measured 2.4-2.7x faster on the shapes that
bind it, and the only thing standing in the way is that a derived plane
duplicates a residency already filling the card. Loader-level layout
replacement is the fix.

What actually moved the needle this session, in order of size: the
stale sampler cap (15.3% of device on the fallback path), vendoring
Marlin (1.5-2.3x on decode GEMMs), the lane raise with band width
following it, and the batch-aware KV split clamp. Three of those four
were width constants or routing that degraded silently rather than
failing — which is the pattern worth carrying into the next session.

## 47. Marlin is unstable at wide batches — band pinned to 32 (2026-08-26)

Forty-second runs are not long enough to qualify this engine. A 90 s
confirmation of the 0.8B's 6,003 tok/s died partway with "invalid UTF-8
leading byte in generated token stream"; a 90 s 4B run at the same
settings did not log a fatal but collapsed to 1,212 tok/s with 9,240
truncated streams. Both had shown clean 40 s runs.

Isolation, all 90 s, 100 users:
  0.8B 32 lanes, Marlin on   5,111 tok/s  3,730 reqs  0 errors  CLEAN
  0.8B 64 lanes, Marlin off  5,429 tok/s  3,954 reqs  0 errors  CLEAN
  0.8B 64 lanes, Marlin on   died at ~90 s
  4B   64 lanes, Marlin on   collapsed mid-run

So the fault is Marlin at wide batches, not the lane count and not the
sampler cap (#44 is clean at 32 lanes over 90 s). kMarlinFixedM is
therefore pinned to 32 regardless of lane count: the band serves only
the widths it is proven on and wider rounds take the engine's own
kernels. SUROGATE_SERVE_MARLIN_WIDE=1 restores the 64-wide band for
debugging.

Honest standing, stable configurations only:
  0.8B  5,429 v vLLM 5,958  (-9%)   64 lanes, Marlin off
  4B    needs a 90 s run at 64 lanes with Marlin off; the 2,633 figure
        is a 40 s number on a configuration now known to be unstable
  27B   370 v 688 (-46%)    32 lanes, unchanged

Suspects for the wide-batch fault, in order: the shared Marlin scratch
(gemm_out / a_pad / c_tmp / locks are process-global and sized from the
first derive — a later weight or a wider round can outgrow what the
freeze locked in, and marlin_w8_run's size guards return false rather
than resize, so a partially-written buffer is possible); the locks array
(Marlin's global reduce self-resets it, and a grid change between calls
could leave it dirty); and c_tmp sizing, which is computed from
marlin_fixed_m() at first derive and would be short if the width ever
rose afterwards. All three are consistent with a fault that needs
sustained load and varied batch widths to surface.

METHOD NOTE: every performance figure in this file measured over 40 s
should be re-qualified at 90 s before it is trusted.

Review found one concrete defect while the card was unavailable: the
scratch growth path freed the old gemm_out and a_pad without draining
the stream. The warmup round derives every weight in a single pass and
the vocab head (248,320 rows) dwarfs any layer, so growth fires while
earlier layers' GEMMs are still in flight — freeing under them is a
use-after-free that corrupts whatever is allocated next. Now drains
first. Whether this is the wide-batch fault is unproven: growth happens
during warmup rather than at 90 s, so the timing does not obviously fit,
but the corruption it causes is unbounded in where it lands. Needs a 90 s
run at 64 lanes with SUROGATE_SERVE_MARLIN_WIDE=1 to test.

## 47. A measurement error, and what the numbers really are (2026-08-26)

Two 90-second runs invalidate a batch of this session's reporting.

The 4B at 64 lanes gives 2,633 tok/s over 40 seconds and 2,110 over 90
seconds, in the same build and config. The short window is dominated by
the ramp: lanes are still filling, contexts are short, and attention and
GDN both cost less per token than they do once every lane carries its
full 512+128. The board's vLLM figures are 90-second runs, so every
40-second number this session was compared against a steady state it
never reached. Only 90-second runs go on the board from here.

Second: Marlin is NEUTRAL at 64 lanes (4B, 90 s: 2,110 with, 2,104
without, both clean). Its measured win was at 32 lanes, where the band
sits at M=32 and the exact-T kernels are weakest. At M=64 they match it.
It stays wired — it costs nothing and it is the vehicle for the 27B FP8
work — but it is not what carries the 4B.

Third, and the reason the above got looked at: with Marlin ON at 64
lanes the 0.8B dies with "invalid UTF-8 leading byte in generated token
stream" partway through a 90-second run. With Marlin OFF at 64 lanes,
and with Marlin ON at 32 lanes, the same load is clean. So there is a
real defect in Marlin at 64-wide batches under sustained churn; the
short runs never surfaced it.

Honest standing, all 90-second, all stable:
  0.8B  5,429 (64 lanes, Marlin off)  v vLLM 5,958   -9%
  4B    2,110 (64 lanes)              v vLLM 3,390  -38%
  27B     370 (32 lanes, 40 s)        v vLLM   688  -46%  [needs a 90 s rerun]

The 0.8B's "+0.8% ahead" from #44 does not survive this correction: that
was a 40-second run at 64 lanes with Marlin, which is both the optimistic
window and the unstable configuration. The sampler fix in #44 is still
real and still worth its keep — it is why 64 lanes is usable at all —
but the headline was wrong.

## 48. The wide band is right for the 4B; the 0.8B fault is still unexplained (2026-08-26)

Two corrections to #47, both from 90-second runs.

The wide Marlin band is a large WIN on the 4B and it is stable: 2,685
tok/s over 90 s at 64 lanes, 1,958 requests, zero errors, zero fatals,
against 2,110 with the band pinned to 32. That is +27%, and it reverses
#47's claim that the 4B fails with the wide band — it does not, only the
0.8B does. The earlier with/without-Marlin comparison that read as
"neutral" was measuring the pinned build against itself: with the band at
32 and rounds 64 wide, Marlin never engages either way.

The 0.8B default at 64 lanes (band pinned) is 5,462 tok/s over 90 s,
zero errors — so the sampler fix in #44 is what makes 64 lanes usable,
and that stands.

A lock-array theory for the 0.8B corruption was investigated and is
WRONG. The reduce indexes locks[locks_off] where locks_off is either
blockIdx.x or a smaller derived value, and determine_exec_config returns
blocks_per_sm = 1 on every path, so the grid is exactly sms blocks and
locks_off < sms. The existing sm_count*4 array has 4x headroom. The
change that sized locks by N was reverted before it shipped: it came
from pattern-matching upstream's Python-side workspace formula instead
of reading the vendored kernel's indexing, and it fixed nothing.

So the 0.8B's wide-band corruption under sustained load is still
unexplained. What is known: it needs the wide band AND that model's
shapes AND sustained churn; the 4B with the same band and the same load
is clean over 90 s; the 0.8B at 32-wide is clean. Its distinguishing
feature remains the vocab GEMM's aspect ratio (248320 x 1024), which is
where a next investigation should start — but with the kernel's own
indexing read first this time, not inferred.

Standing (90 s, stable configs): 0.8B 5,462 v 5,958 (-8%), 4B 2,685 v
3,390 (-21%), 27B 370 v 688 (-46%, 40 s provisional).

### #48 follow-up: how to settle the 0.8B wide-band fault (needs a GPU)

Static elimination so far: the lock array is adequately sized (grid is
sms blocks, blocks_per_sm is 1 on every path, locks_off < sms);
use_atomic_add is hardcoded false so the accumulate-into-C path that
would need a zeroed output is never taken; C_tmp at sms * 64 * 256
floats exactly covers the largest config (thread_m_blocks 4 x
thread_n 256); and the sampler workspace plan scales with the same
constant the runtime path uses.

One unexamined candidate remains, in the vendored reduce:

    locks_off = (iters * blockIdx.x) / k_tiles - 1;   // marlin_template.h:422

For blockIdx.x == 0 this is -1, so the taken branch writes locks[-1].
The branch is guarded by part2_mn_tiles < gridDim.x, so whether it is
reachable depends on the problem's tile count against the SM count —
which is exactly what differs between the 0.8B's 248320x1024 vocab GEMM
and the 4B's shapes, and between a 64-wide band (thread_m_blocks 4) and
a 32-wide one (thread_m_blocks 2).

Do not fix this by inspection. The next GPU window should run:

    compute-sanitizer --tool memcheck --launch-timeout 120 \
      surogate-engine <0.8b artifact> --max-concurrency 64 ...

with SUROGATE_SERVE_MARLIN_WIDE=1 under ~60 s of 100-user load. memcheck
names the offending kernel and offset directly. racecheck is the follow-up
if memcheck comes back clean. Budget ~15 minutes of GPU; the run is
heavily slowed by instrumentation, so use a shorter load and fewer users.

Why this is the highest-value item on the board: the 0.8B measured 6,003
tok/s in the wide-band configuration before it corrupted, against vLLM's
5,958. Fixing this one fault plausibly closes the 0.8B cell outright, and
the same band is already worth +27% on the 4B.

## 49. The 0.8B beats vLLM when it runs; the fault is not yet found (2026-08-26)

Seven 90-second runs of the 0.8B at 64 lanes with the wide Marlin band
(SUROGATE_SERVE_MARLIN_WIDE=1):

  6,015  clean      6,078  clean
  4,198  CRASHED    2,836  CRASHED
  6,075  clean      6,101  clean
                    6,046  clean

Five clean runs, all between 6,015 and 6,101 tok/s, against vLLM's 5,958
— the engine is 1-2% AHEAD whenever the round completes. Two runs died
with "invalid UTF-8 leading byte in generated token stream". No
functional code differs between crashing and clean runs; the fault is
intermittent at roughly one run in three.

Two theories tested and REJECTED by measurement, not by argument:

  Lock array undersized. Rejected statically — the grid is sms blocks,
  blocks_per_sm is 1 on every path, locks_off < sms. The change was
  written and reverted before shipping.

  Out-of-bounds locks[-1] from marlin_template.h:422 corrupting the
  adjacent allocation. This write is real (blockIdx.x == 0 gives -1 on
  that branch) and is now guarded by front-padding the allocation, but
  guarding it did NOT change the failure rate: 3 clean / 1 crash over
  four runs, indistinguishable from unguarded. The guard stays as
  defensive hygiene, labelled as such. The fault is elsewhere.

Also relevant: compute-sanitizer memcheck runs clean, but only ~25
requests complete under instrumentation, far below the churn that
triggers it — so a clean memcheck is not evidence here.

Next step is instrumentation of the failure path rather than more
theories: when the frontend rejects a generated token, dump the round's
shape (mixed vs pure decode, graph vs eager, batch width, lane index).
That names the producing path in one crash instead of narrowing by
elimination. Roughly ten lines on a path that costs nothing until it
fires.

Standing: 0.8B 6,046 (wide band, 5 of 7 runs) or 5,462 (default, stable)
v vLLM 5,958; 4B 2,685 v 3,390 (-21%); 27B 370 v 688 (-46%).

## 50. The mixed-round pad race — found, fixed, 0.8B passes vLLM (2026-08-26)

Instrumenting the failure path (record each round's shape; print it in the
worker's fatal line) named the producing round on the first crash:

  invalid UTF-8 continuation byte ... [last round #4820 kind=mixed
  batch=63 prefill_lane=29 lanes=64]

A mixed round with 63 decode rows. batch_bucket_for rounded 63 up to 64,
so exactly one pad row existed, and #30's padding scheme gives a pad row
a duplicate of row 0's ingress — same token, position, KV row, and state
slot. That reasoning held for the KV write, which is idempotent, and
FAILED for the GDN mixers: causal_conv1d_silu_snapshot and
gated_delta_net_snapshot update the lane's state read-modify-write, so
the pad column and row 0 do concurrent RMW on one slot inside a single
launch. The loser's update is lost, the state drifts, and a wrong token
appears in some later round of that sequence — far from the cause, which
is why it read as a Marlin bug. Marlin was never implicated: the wide
band only made rounds faster, so more of them hit the race.

Fixed by capturing per exact decode width instead of a rounded bucket,
which removes padding entirely. Capture is on demand, so only widths that
actually occur cost a graph.

Measured, 0.8B, 64 lanes, 90 s, wide band, three consecutive runs:

  6,072   6,097   6,060 tok/s     zero errors, zero fatals

against vLLM's 5,958 — the engine is AHEAD by 1.7-2.3% and stable. An
interim fix that gated the graph on rows == bucket was also clean (four
runs, 5,749-5,789) but cost 4.5% by dropping non-bucket rounds to eager;
exact-width capture keeps both correctness and speed.

Two rejected theories are recorded in #49 so nobody re-walks them. The
lesson: three sessions of reasoning about Marlin's buffers were worth
less than ten lines that recorded which round produced the bad token.

### #50 follow-up: standing after the fix

  model  surogate   vLLM    gap      runs
  0.8B      6,076  5,958   +2.0%  AHEAD   3 x 90 s, zero errors
  4B        2,661  3,390   -22%           2 x 90 s, zero errors
  27B         370    688   -46%           40 s provisional

The 0.8B is past vLLM on a stable configuration. The 4B is unchanged by
the exact-width fix (2,661 against 2,685 with padding — within run
variance, and now correct), so its deficit is elsewhere: its census puts
GDN recurrent_snapshot at 21.9% near roofline, cutlass prefill at 20.1%,
decode GEMMs at 18.2%, attention at 9.2% still ~2.6x off its KV-read
roofline, and 12% device idle in the host serial path between rounds.
The 27B remains lane-limited by memory and entirely compute-bound.

## 51. The 4B lane sweep, re-run on the fixed build (2026-08-26)

vLLM's 4B advantage is stream count — roughly 100 concurrent against our
64 — so the lane sweep was worth repeating now that the pad race (#50)
and the rounded buckets are gone, since the earlier sweep ran with both.

  64 lanes  2,661 tok/s   (kv 131072)
  80 lanes  2,399         (kv 98304)
  96 lanes  2,396         (kv 98304)

The conclusion survives the fix: 64 is the 4B's optimum and more lanes
cost throughput. Per-stream decode falls faster than lane count rises,
so the aggregate drops. Concurrency is not the lever here; the ceiling
is reverted to 64.

That leaves the two openings the census already named, and nothing else
above a few percent: decode attention at 9.2% of device but still ~2.6x
off its KV-read roofline even after the split clamp (#45) — the
partial-plus-reduce design writes partials to global and reads them back
— and 12% device idle sitting in the host serial path between rounds,
which is the async-scheduler work whose infrastructure #32 already
built. GDN recurrent_snapshot (21.9%) and the cutlass prefill GEMMs
(20.1%) are both near roofline and are not worth attacking.

### #51 follow-up: the residency theory does not explain the 4B

The obvious structural suspicion — we serve the 4B W8-resident (4.80 GiB
loaded) while vLLM serves NVFP4, so we read twice the weight bytes per
decode round — does not survive arithmetic. At 41.6 rounds per second
that is ~200 GB/s of weight traffic against the card's ~1.79 TB/s, so
weights are about 11% of bandwidth and not the binding constraint.
Halving them would not close a 22% gap. (Residency still matters for the
27B, which is lane-limited by memory rather than bandwidth — a different
argument.)

So the 4B's deficit is accounted for, almost exactly, by the two items
already named: decode attention running ~2.6x off its KV-read roofline
at 9.2% of device, and 12% device idle in the host serial path. Together
that is ~21% against a measured 22% gap. Both are real engineering —
a fused flash-style decode attention kernel, and finishing the async
scheduler on the #32 infrastructure — and neither is a tuning knob.

## 52. The 27B lane probe, and what native Marlin residency is actually worth (2026-08-26)

90-second baseline at 32 lanes: 376 tok/s (the 370 on the board was a
40-second provisional; they agree). Above that the model will not start:

  40 lanes, kv 8192   needs 13.95 GB, 12.76 GB available
  40 lanes, kv auto   fails on headroom
  64 lanes            needs 23.30 GB (measured earlier)

Per-lane GDN state dominates at this size, so the 27B is capped near 32
lanes on a 32 GB card while the 0.8B and 4B run 64. Squeezing KV does not
buy the next lane band — 40 lanes is short by 1.2 GB with KV already at
the floor.

This corrects the case for native Marlin residency. The earlier argument
was that making Marlin the resident format frees the derived plane and
that the freed VRAM buys lanes. That is wrong for the 27B: no plane is
derived there, precisely because it does not fit — which is why Marlin
FP8 never reached this model. Replacing the resident layout is
byte-neutral, so it buys no lanes, and the probe above shows lanes were
not going to help much anyway.

What it does buy is the only path to Marlin FP8 on the shapes that bind
this model. Its census: FP8-class GEMMs are 47% of device time on a
nominally 4-bit model, and the binding exact-T kernels (T=17/18,
N=5120, K=6144/17408) run at 412-503 GB/s where a same-class tile in the
same profile reaches 926 GB/s. Marlin FP8 measured 2.4-2.7x on those
shapes in isolation. Since a runtime-derived plane cannot fit, the layout
has to come from the converter and be mapped as the residency by the
loader.

So: still worth doing, still a converter plus loader change, but it is a
kernel-throughput play on 47% of the device, not a memory or concurrency
play.

## 53. The wide Marlin band becomes the default (2026-08-26)

The band was pinned to 32 because a wide band corrupted the 0.8B under
sustained load. That fault was the mixed-round pad race (#50), not
Marlin, and it is fixed. The pin's rationale is gone, so the band follows
the lane ceiling again; SUROGATE_SERVE_MARLIN_NARROW=1 pins it back for
bisecting.

Confirmed on the default configuration with no environment flags: 0.8B
6,051 tok/s over 90 s, 4,406 requests, zero errors, zero fatals. The
env-flag measurement was 6,076, so the flip reproduces it. Every engine
row on the board is now what `surogate serve` does out of the box.

## 54. Head dimension becomes a shape parameter (2026-08-26)

First landing of the multi-architecture plan (design/serve-engine-multiarch.md).

`kGqaHeadDim = 256` was a file-scope constant baked into every decode
kernel and index helper, and the geometries around it were registered as
model-named aliases — Gqa27Geometry, Gqa08Geometry, Gqa4BGeometry. That
is a hard stop at the first architecture with a different head dimension,
which is all of Gemma, GLM and Kimi.

Head dimension is now a template parameter on GqaGeometry, threaded
through the decode kernels, the index helpers and the launcher, and the
registration table names shapes rather than checkpoints:

    Gqa256_24q4   Gqa256_16q2   Gqa256_8q2   Gqa256_16q4

with the checkpoints that use each recorded in a comment. The model-named
aliases remain as compatibility typedefs and disappear as the remaining
dispatchers convert. Adding an architecture whose shape is already
registered is now a config change; adding a new shape is one line, and an
unregistered combination fails the build rather than degrading at runtime.

Behaviour-neutral, as intended: 0.8B 6,022 tok/s over 90 s, zero errors,
against 6,051 and 6,076 on the two preceding builds.

Still to do on this item: KV dtype is a shape property in the design but
remains a parallel code path (separate bf16 and i8 kernel headers selected
at runtime), and the fused flash decode kernel itself is not started —
the geometry work is its precondition, not the kernel.

## 55. The mixed-round corruption: KV pages, not Marlin (2026-08-26)

Root cause of the wrong-token fatals that have shadowed the 64-lane
configuration all session.

A mixed-round graph is captured at a chunk length rounded up to a
multiple of 128, and it writes that whole rounded window — pad columns
included. KV pages are mapped in units of kPagedKVPageSize = **64**, and
only up to the prompt length. So a prompt whose length lands in the upper
half of a 128-block has its pad columns writing past the mapped pages:
548 real tokens map 576, the graph writes 640, and the last 64 columns
land in whatever block-table slots follow — another sequence's KV.

Every observed symptom follows from that. Intermittent, because it
depends on prompt length mod 128 and the loadgen salts every prompt.
Mixed-only, because only the mixed path pairs a 128-rounded window with
an un-mapped tail. And the corrupted token appears in a *different*
lane's stream than the round that produced it, which is why it read as a
Marlin bug for so long.

Fixed by mapping KV for the window the graph actually writes, before
replay: PrefillGraphFamily::chunk_bucket_for exposes the ladder's
rounding, and advance_prefill_mixed materializes to it.

Measured, 0.8B at 64 lanes, mixed graphs ON, four consecutive 90-second
runs: 6,065 / 6,014 / 6,051 / 6,058 tok/s, zero errors, zero fatals.
Against vLLM's 5,958 that is +1.6% and stable at full speed.

Three theories were tested and rejected before this one, all by
measurement rather than argument, and all are recorded so they are not
re-walked: the Marlin lock array is adequately sized (#49); guarding the
real locks[-1] write changes nothing (#49); and an unbanded attention
envelope in the graph is not the cause either — banding it, which this
patch keeps because it matches the proven decode-graph design, still
crashed at batch 62. The one that worked came from instrumentation, not
inspection: recording each round's shape and printing it in the worker's
fatal line named `kind=mixed batch=62/63` every time, and the constant
appearance of near-full batches with a live prefill lane is what finally
pointed at the prefill window rather than the decode batch.

## 56. The async scheduler is the wrong lever, and here is the measurement (2026-08-26)

The plan (design/serve-engine-multiarch.md, item 2) was to overlap rounds:
launch N+1 before consuming N, to recover the 11-12% device idle sitting
in the host serial path. The lifecycle split that enables it is built
(#54 groundwork, launch_ordinary_round / consume_ordinary_round behind
runtime/contract/round_lifecycle.h). Before wiring the executor to it, the
cost of the one thing overlap unavoidably delays — refilling a free lane —
was measured directly, using the burst floor as a proxy, since a burst of
K delays lane refill exactly as an overlap depth of K does.

4B, 100 users, 90-second runs:

  free-lane burst 1   2,661 tok/s   TTFT 1.7 s
  free-lane burst 2   2,174 tok/s   TTFT 3.3 s
  free-lane burst 4   1,708 tok/s   TTFT 5.5 s

Delaying a free lane's refill by a SINGLE round costs 18% of throughput.
The host serial that overlap would recover is worth about 11%. Overlap
therefore loses, and loses more the deeper it goes.

The reason is structural: this engine admits through a GPU unit — a free
lane needs a prefill before it produces anything — so an idle lane is
expensive in a way an idle host microsecond is not. Under 100-user load a
lane is nearly always free, so the scheduler is right to take single
rounds and refill immediately. The 11% idle is the price of keeping 64
lanes full, not waste to be reclaimed.

What this redirects: the 4B's remaining gap has to come from making rounds
faster, not from overlapping them. That is decode attention, still ~2x off
its KV-read roofline. The round-lifecycle contract stays — it is the right
interface for a target to expose and costs nothing — but the executor will
not be driven to overlap on it for this workload shape.

The knob (SUROGATE_SERVE_FREE_LANE_BURST) stays so the measurement can be
repeated on other workload shapes; a batch-style workload with a saturated
queue and no free lanes would flip this result.

## 57. GDN recurrent state moves to bf16 storage (2026-08-26)

The largest single cost in a decode round was the gated-delta-net
recurrent state, and it was being stored twice as wide as it needed to be.

At 32 value heads x 128 x 128 the state is 2 MiB per lane per layer. At 64
lanes across 24 GDN layers a decode round reads and writes ~6 GiB of it.
The snapshot kernel measured 211 us per layer call, which is 268 MB at
1,271 GB/s — 71% of peak, so bandwidth-bound and not fixable by tuning.
It was 22.6% of device time, and because the traffic scales linearly with
batch it is also why raising the lane count never bought throughput.

vLLM stores this state at the model dtype: gated_delta_net_state_dtype
resolves mamba_ssm_cache_dtype "auto" to the activation dtype, i.e. bf16.
We stored fp32. So the comparison engine was moving half the bytes we
were, on the single biggest item in the round.

Compute is unchanged — the state still lives in registers as fp32 and the
delta rule still runs in fp32. Only the HBM representation narrows, at the
one seam where it touches memory: load_qk_lane / store_qk_lane in the
recurrent kernels, and the scalar loads/stores in the chunked
state-passing kernel. The storage type is a named alias, GdnStateStorage,
so a quality-sensitive deployment can put it back to float in one line.

Measured, 100 users, 512/128, 90-second runs:

  4B    2,666 -> 3,015 / 3,024 tok/s   (+13%)
  0.8B  6,047 -> 6,646 tok/s           (+10%)

Correctness checked at temperature 0 before either benchmark: coherent,
factually right answers (Rayleigh scattering, the first six primes). Zero
errors and zero fatals across all runs.

Standing against vLLM: 0.8B 6,646 v 5,958 (+11.6% AHEAD), 4B 3,024 v 3,390
(-11%, was -21%), 27B pending re-measure.

### #57 follow-up: the 27B, and the second-order win

The 27B gains most from bf16 state, and then gains again because the state
is what capped its concurrency.

  32 lanes, fp32 state    376 tok/s   TTFT 26.0 s
  32 lanes, bf16 state    520 tok/s   TTFT 16.1 s   (+38%)
  48 lanes, bf16 state    582 tok/s   TTFT 11.1 s   (+55% over baseline)
  64 lanes                does not fit (headroom check fails after weights)

Per-lane GDN state is what made 40 lanes impossible before (13.95 GB
wanted against 12.76 GB free). Halving it moved the ceiling from 32 to 48,
and the extra lanes are worth another 12% on top of the 38% the narrower
traffic gives directly. This is the same coupling seen everywhere in this
model: its state dominates both bandwidth and capacity, so anything that
shrinks it pays twice.

Board after this change, all 90-second steady state, 100 users:

  0.8B  6,646 v vLLM 5,958   +11.6%  AHEAD
  4B    3,024 v vLLM 3,390   -11%    (was -45% at session start)
  27B     582 v vLLM   688   -15%    (was -46%)

## 58. Post-bf16 census, and two more rejected levers (2026-08-26)

4B census after #57 (100 users, 64 lanes). recurrent_snapshot fell from
211 us to 95.6 us per call and from 22.6% to 11.7% of device, exactly the
halving the narrower storage predicts:

   23.7%  cutlass prefill GEMMs      72.6 us x 17076
   21.5%  marlin decode GEMMs        69.1 us x 16226
   11.7%  gdn recurrent_snapshot     95.6 us x  6394
   10.1%  small fused/elementwise    18.3 us x 28807
    7.9%  gqa decode attention      193.5 us x  2131
    13%   idle

Two levers tested against that and rejected, both by measurement:

  Lane scaling, retried now that state traffic halved. It paid on the 27B
  (32 -> 48 lanes, +12%) but not here: 64 lanes 3,024, 80 lanes 2,680, 96
  lanes 2,675. The 4B stays at 64.

  Attention split target. The wave count driving the split policy is now
  tunable (SUROGATE_SERVE_ATTN_WAVES, default 2). Sweeping it changes
  nothing: 2 waves 3,007, 4 waves 3,023 — inside run variance. The decode
  attention kernel is not split-limited, so its ~2x gap to the KV-read
  roofline needs the fused rewrite, not a knob.

What remains on the 4B is diffuse: GEMMs are 45% of device across two
families, attention is 7.9% at about half of peak bandwidth, and 13% is
the structural admission idle from #56. No single fix closes 11%.

The 27B is the better investment: its FP8-class GEMMs are 47% of device
at 412-503 GB/s against 926 GB/s proven achievable on the same class of
problem, which is one target rather than four.

## 59. FP8 Marlin: probe passed, all-T routing landed, residency needs the tag (2026-08-26)

**Probe.** Marlin FP8 measured against the kernels it would replace, on the
27B's own shapes, on this card, correctness checked (max_rel 0.0036-0.0038):

  shape                     live              marlin fp8        gain
  N=5120  K=6144  (out)     76.4us  412 GB/s   31.3us 1005 GB/s  2.4x
  N=5120  K=17408 (down)   185.4us  481 GB/s   67.3us 1325 GB/s  2.75x
  N=16384 K=5120  (gdn_in)  90.6us  926 GB/s   61.8us 1358 GB/s  1.47x

Those families are 47% of the 27B's device time, so the probe justifies the
work — it beats even the A8 tile that was already the fast path.

**Landed: any-T FP8 Marlin routing.** Below the band the call still goes
through the padded scratch so captured geometry stays fixed. Above it, A
and C pass straight through — x is [k,t] contiguous, which is [t,k]
row-major, exactly Marlin's A, and out is [n,t], exactly its C. This is a
prerequisite for residency replacement, not a nicety: once a weight holds
Marlin bytes there is no other kernel that can read them.

**Attempted and REVERTED: in-place residency replacement.** For FP8 the
packed form is exactly n*k bytes and the scales exactly n*2 — the sizes the
residency already holds — so the repacked layout can be written back over
the original and the plane costs nothing permanent. That is what makes the
27B feasible where duplication never was. It produced garbage on the first
request.

The reason is instructive and is the argument for doing this properly.
Replacement makes the weight's bytes meaningful only to Marlin, but every
decline path in marlin_fp8_run — capture without a cached plane, an
unavailable scratch, a non-contiguous view — falls back to the FP8 kernel,
which then reads Marlin tiles as e4m3. Nothing errors; the model simply
emits noise. Safety here cannot come from "the wired call sites happen to
cover it", because a decline is a runtime property, not a call-site one.

So residency replacement needs the per-tensor layout TAG from
design/serve-engine-multiarch.md item 3: the converter writes Marlin tiles,
the loader records the layout on the weight, and routing dispatches on it,
so a path that cannot serve a Marlin-layout weight is a type error at plan
time instead of a silent misread at run time. The probe says that work is
worth roughly 20% of the 27B's device time; this patch leaves the routing
prerequisite in place and the unsafe shortcut out.

27B verified restored after the revert: 581 tok/s at 48 lanes, correct
output, zero errors.

## 60. The residency layout tag (2026-08-26)

#59 showed that replacing a weight's residency with Marlin tiles is
size-exact and free, and unsafe without a way for the rest of the engine to
know it happened. This adds that way.

`QuantLayout::MarlinTiles` joins the shared weight abstraction in
core/tensor.h — on the weight, not in a target, so any architecture
inherits it by declaring a compute profile rather than by having code
written for it (design/unified-train-serve.md: abstraction work comes
before generation, because the generator emits against these interfaces).

Three pieces:

  marlin_fp8_adopt_residency(Weight&, stream) repacks a weight into Marlin
  tiles in place and stamps the tag. Transient buffers only; nothing
  permanent, because for FP8 the packed form is exactly n*k and the scales
  exactly n*2 — the sizes the residency already holds.

  marlin_fp8_plane_for recognises an adopted weight and returns the weight's
  own pointers as the plane, so no second copy exists anywhere.

  linear_add routes a MarlinTiles weight to Marlin at any T and THROWS if
  Marlin declines, instead of falling through to the e4m3 kernels.

The safety argument is now structural rather than by inspection. Routes gate
on QuantLayout::RowScale; an adopted weight no longer matches them, so a path
that cannot serve Marlin tiles reaches its "unsupported weight format" throw
at plan time. The failure mode #59 hit — tiles read as e4m3, noise emitted,
nothing raised — is unrepresentable: to misread the bytes a route would have
to match a layout it does not accept.

Still to land: the call that adopts. Adoption must cover every consumer of a
weight before it is applied, and gdn_input_proj and attn_input_proj have no
Marlin route yet, so only the linear_add and linear_swiglu families are
eligible today. Wiring those two, then adopting at load time from the
target's declared compute profile, is what turns the 2.4-2.75x the probe
measured into throughput.

## 61. Marlin residency adoption: wired, not yet working (2026-08-26)

All four FP8 consumer families now have a Marlin route, which is the
precondition for adopting a weight's residency: `linear_add`,
`linear_swiglu`, and — new here — `attn_input_proj` and `gdn_input_proj`.
The fused pair needed the GEMM separated from the fusion: Marlin emits the
whole [parent_rows, T] parent and the split becomes a strided copy
afterwards, a few hundred KB against tens of MB of GEMM.

`marlin_fp8_maybe_adopt` triggers adoption from those routes and nowhere
else, which is exactly the safety precondition — a weight only becomes
Marlin-tiled because a route that can read tiles asked for it. It never
adopts during capture, so adoption lands on the warmup pass.

**State: enabling it on the 27B does not work yet.** Three failures so far,
each fixed and each replaced by the next:

  1. "weight holds Marlin tiles but Marlin declined" — an adopted weight
     skips the repack path that allocates the scratch. Fixed by priming the
     scratch at adoption and on the adopted branch of plane_for.
  2. "invalid FP8 weight" — validate_fp8_weight gates on RowScale, which an
     adopted weight deliberately is not. Fixed by adopting before validating.
  3. std::bad_alloc — current. The plan now reserves the fused parent
     (marlin_fused_parent_bytes, zero unless adoption is enabled) and the
     row arithmetic checks out for the 27B (attention 6144*2 + 1024*2 =
     14336; GDN 2048*2 + 6144*2 = 16384), so the reservation is either
     landing in the wrong phase or the arena is short for another reason.
     Unresolved; needs a GPU to bisect.

Everything is behind SUROGATE_SERVE_MARLIN_FP8, and both gates return
false/zero when it is unset, so the default path is byte-identical.

Worth recording plainly: all three failures were named errors at startup.
The same code without QuantLayout::MarlinTiles (#59) produced garbage
tokens and raised nothing at all. The tag is doing precisely what it was
added to do, even while the feature it guards is unfinished.

## 62. Marlin residency finished, measured, and NOT enabled (2026-08-26)

The mechanism is complete and correct; the measurement says do not turn it on.

Completed since #61:

  The arena now names its own exhaustion — request, offset, capacity, and how
  short it fell — instead of throwing a bare std::bad_alloc. That is what
  turned the blocking failure into a five-minute fix, after it had cost a
  debugging cycle as an anonymous exception.

  The fused parent an adopted weight produces gets dedicated growable scratch
  rather than a workspace reservation. Its size depends on the widest call a
  target makes, the arena is planned per phase, and threading a conditional
  reservation through every phase that could see an adopted weight was both
  invasive and wrong on the first attempt.

  validate_fp8_weight rejects QuantLayout::MarlinTiles. Every FP8 route funnels
  through it, so an unwired consumer is now a named error rather than fluent
  nonsense.

  Adopted weights take the exact-M Marlin path. Padding M to the band exists so
  a captured decode graph sees constant geometry, but decode graphs are captured
  per batch size, so t is already constant within each.

Measured on the 27B, 100 users, 90-second runs:

  adoption off (default)                 581 tok/s
  adoption on, linear_add + swiglu       547 tok/s   -6%

**The isolated probe did not translate.** Marlin FP8 is 2.4-2.75x on these
shapes in the bench (#59) and a net LOSS in the serving loop. That is the
caveat this work was gated on — "the 2.4-2.7x is measured on isolated shapes,
not in the serving loop; probe one shape end-to-end before building the full
converter path" — and the probe was run at the wrong altitude: a kernel
benchmark, not a serving measurement. The gap is not explained yet. Candidates
are the exact-M path forgoing whatever the padded band buys in the round, the
fused staging adding a copy the e4m3 route does not pay, and Marlin's advantage
at T=17-18 evaporating once it is one op among many rather than the only thing
running.

Adoption on the two fused projections is additionally still WRONG: it produces
a correct first token and then degenerate decode, isolated by bisect to the
parent-split rather than to adoption itself. It is behind its own switch
(SUROGATE_SERVE_MARLIN_FP8_FUSED) and stays off.

Default path re-verified unchanged at 581 tok/s, zero errors. Both switches
default off, so none of this is live.

What this cost and what it bought: the residency mechanism, the layout tag that
makes misreads unrepresentable, an arena that explains itself, and a measured
answer that the 27B's remaining gap is not closed by this route. The FP8-class
GEMMs are still 47% of its device time at half their achievable bandwidth; the
reason Marlin does not capture that in situ is the next question, and it is a
profiling question rather than an implementation one.

### #62 follow-up: fused adoption removed, not parked

The fused-projection adoption path is deleted rather than left behind a flag.

Two defects were found and one fixed. The parent split assumed a contiguous
destination; the decode path passes views into wider buffers, so it wrote the
wrong bytes — a correct first token followed by degenerate decode. Taking the
stride from the tensor fixes that, and the fix is kept in the split helper's
absence as the lesson: never assume nb[1] == rows * element_size.

The second is structural and is why the path is gone. Adoption changes an op's
kernel selection, so it changes a captured graph's topology, and adopting
between two captures makes the next exec update fail with
cudaErrorGraphExecUpdateFailure. Closing adoption before the first capture (a
distinct signal from the scratch freeze, which is about pointer stability and
happens later) was necessary but not sufficient: the fused route still failed
the same way, and further bisection was not worth it on a feature that measures
NEGATIVE end to end.

Final state, 27B, 100 users, 90-second runs, both verified coherent at
temperature 0:

  adoption off (default)              581 tok/s
  adoption on, linear_add + swiglu    546 tok/s   -6%

No known-defective code remains: what is left is correct and simply slower, so
it stays off. The mechanism — the layout tag, in-place adoption, the exact-M
route, the self-describing arena, the FP8 validator that rejects tiles — is
intact and is what a converter-written Marlin residency would use if the in-situ
regression is ever explained. That explanation is a profiling question: Marlin
is 2.4-2.75x on these shapes in isolation and a loss in the round, and nothing
in this patch series accounts for the difference.

## 63. Why Marlin wins the bench and loses the round (2026-08-26)

Profiled both configurations on the 27B under identical 100-user load, six
seconds of steady state each.

                        adoption off        adoption on
  device busy              5,697 ms            5,818 ms
  device idle                303 ms              182 ms
  fp8_mma_kernel           2,469 ms            1,152 ms
  marlin::Marlin                 0 ms           1,493 ms
  silu_and_mul                83 ms /1,355        94 ms /1,934
  residual_add                 0 ms /    0        28 ms /5,663

Two things, and neither is the kernel being slow.

**The epilogues stopped being free.** The FP8 kernels are fused: the swiglu
path computes gate_up and applies silu-mul in one launch, and linear_add folds
the residual into its epilogue. Marlin computes a GEMM and nothing else, so
each adopted call grows a second pass over its output — residual_add appears
from nothing at 5,663 launches and 28 ms, and silu_and_mul gains 579 launches.
The two ops adopted, linear_add and linear_swiglu, are precisely the two most
fused paths in the model. Of every candidate, those were the worst choices.

**The baseline in the probe was stale.** #59 compared Marlin against
412-503 GB/s exact-T numbers taken from the bd27v census, which predates the A8
batch tile and the other decode work landed since. The FP8 GEMM the 27B
actually runs today averages 213.9 us; Marlin averages 200.3 us. Marlin is
about 6% faster per call, not 2.4-2.75x — the probe was measuring against a
kernel the engine had already stopped using.

Net: GEMM time +176 ms (+7.1%) across +1,664 launches, plus ~40 ms of new
epilogue passes, for -6% throughput. Idle falls (303 ms to 182 ms) because the
device is doing more work, not less.

What this says about where Marlin could still pay: only where the incumbent has
no fused epilogue to lose. That is the plain projections — attn_input_proj and
gdn_input_proj — which is the opposite of what was adopted, and which is also
where the removed fused-split path was headed before it was deleted for
unrelated correctness reasons. Any retry should (a) re-derive the baseline from
a current census rather than a remembered one, and (b) target unfused ops, or
give Marlin an epilogue.

The general lesson is about measurement altitude, and it is the second time
this session: a kernel benchmark answers "is this kernel faster", which is not
the same question as "does the round get shorter". A fused incumbent can be
slower per GEMM and still win.

### #63 follow-up: the unfused projections were tried too

The profile said Marlin can only pay where the incumbent has no fused epilogue,
which points at attn_input_proj and gdn_input_proj. That was implemented — the
stride-correct split, adoption restricted to those two ops, the fused kernels
left on their own fp8 path — and it does not run:

  kv auto, 48 lanes    cudaErrorMemoryAllocation during graph capture
  kv 32768             same
  kv 24576             cudaErrorGraphExecUpdateFailure

The staging buffer these projections need is sized by the widest call — tens of
megabytes at prefill width — and it is allocated during warmup, AFTER the KV
cache has already claimed the headroom from auto-sizing. Shrinking KV clears the
allocation failure and exposes the graph-update failure underneath, which is the
same one the earlier attempt hit and which closing adoption before the first
capture did not resolve.

So the theoretically right target is blocked by two engine-level facts rather
than by the kernel: staging that competes with the KV cache because it is
claimed after it, and a graph-update path that will not accept whatever this
route changes. Both are fixable — stage before KV sizing, and find what actually
differs between the captured and updated topology — but neither is worth doing
before the premise is re-established, because the premise itself is now in
doubt: Marlin is ~6% faster per GEMM here, not 2.4-2.75x, and 6% of the GEMM
share is a small prize to buy with this much machinery.

Reverted to the committed state. The engine is unchanged and the default path
measures 581 tok/s.

## 64. Row tiling at N=5120, and the 27B's real constraint (2026-08-26)

The FP8 launch grid is (output_rows / kBlockRows) * token_tiles, so at decode
and narrow-prefill widths the row tiling alone decides occupancy. At N=5120 a
128-row tile yields 40 CTAs on a 170-SM device — a quarter of the machine — and
those shapes measured 324-334 GB/s where N=16384 reaches 894 GB/s on the same
schedule. o_proj and down_proj are ~17% of the 27B's device time.

Halving the row tile to 64 (80 CTAs) gives 581 -> 596 tok/s. Halving again to
32 (160 CTAs) gives 590, so 64 is the optimum: past that the per-CTA work is
too small to amortise its own setup.

Note the near-miss: the same change applied to the BATCH schedule first was
neutral, because the 27B serves 48 lanes and decode t exceeds the 32-token
batch band. It lands in the production schedule, which serves both prefill and
wide decode.

Also measured, both kept:
  int8 KV     596 -> 604 tok/s, and doubles resolved KV (46,016 -> 89,280)
  bf16 KV     593 for comparison at the same tiling

And measured, both rejected:
  128-token prefill tiles   557 (-8%): wider token tiles cost more occupancy
                            than the weight-traffic they save
  fp4 prefill profile       581 either way: the flag drives the W8 path, and
                            the 27B is NVFP4/FP8 resident, so it is a no-op

**What the numbers say about the 27B's gap.** The board's vLLM figure uses
--max-num-seqs 32, so 688 tok/s is 32 concurrent streams at ~21.5 tok/s each.
Ours is 48 lanes at ~12.6 aggregate-per-lane, and the per-stream decode rate is
comparable — the difference is duty cycle. Roughly 40% of the device goes to
prefill here.

CORRECTION to the sentence that stood here: it claimed NVFP4 showed the same
occupancy problem, its N=5120 calls running 40 CTAs. That was gridX read alone.
The NVFP4 launcher uses a 2-D grid — (output_rows / kBlockN, token_tiles) — so
those calls are (40,6) = 240 CTAs and (80,2) = 160, which is 1-1.5 waves and not
starved. The FP8 launcher is 1-D, which is why its 40 was real and why halving
its row tile paid. NVFP4 needs no equivalent fix, and the 7% it was supposed to
be worth does not exist.

## 65. int8 KV withdrawn: a throughput number bought with quality (2026-08-27)

#64 recorded int8 KV as a win (27B 596 -> 604, and 4B 3,032 -> 3,113). It is
withdrawn from the recommended configuration and from the board.

It is a quality change, and the evidence was in the same runs that produced the
numbers. Asked to name three colors at temperature 0, the 4B answered "Red,
blue, and green." with bf16 KV and "I'm not sure if this is what you're
look..." with int8. That was visible when the number was recorded and was not
acted on.

The engine rows on the board are all bf16 KV now. The honest 27B figure is 593,
not 604, and the gap to vLLM is -14% rather than -12%. vLLM's own 27B row does
run --kv-cache-dtype fp8, so a matched-precision comparison would be a
reasonable thing to construct deliberately — with a quality measurement beside
it, not inferred from a single prompt — but silently adopting reduced KV
precision to close a throughput gap is not that.

The int8 path stays available behind --kv-dtype int8 for anyone who wants the
trade explicitly. What is removed is the assumption that it is free.

### #65 follow-up: the standing rule

Quantized KV is off the table for defaults and for board rows, confirmed by the
owner: int8 KV is accuracy degradation. It remains reachable through
--kv-dtype int8 as a trade a user can choose knowingly.

The generalisation worth keeping: a throughput number bought with output
quality is not comparable to one that is not, so any win from a precision
change has to be read alongside the generated text before it is recorded. The
GDN state move to bf16 (#57) passed that bar — it matches what the comparison
engine stores, and coherence was checked at temperature 0 on two models before
the number went on the board. int8 KV did not, and the failing evidence was in
the very run that produced its number.

## 66. The serve CLI speaks vLLM's option vocabulary (2026-08-27)

The server's flags are renamed to vLLM's, and the old spellings are removed
rather than aliased:

  --max-concurrency   ->  --max-num-seqs
  --model-id          ->  --served-model-name
  --max-context       ->  --max-model-len
  --prefill-chunk     ->  --max-num-batched-tokens
  --kv-dtype          ->  --kv-cache-dtype
  --no-cuda-graph     ->  --enforce-eager

A vLLM command line now transfers directly, which matters most for the thing
this engine is measured against: a benchmark comparison should not require the
reader to translate one side's knobs into the other's. --max-num-seqs in
particular is the number the board's figures turn on — vLLM's 27B row is 32
sequences, ours is 48 — and having both sides spell it the same way makes that
visible rather than buried.

--kv-cache-dtype accepts vLLM's "auto" as bf16. It rejects fp8 with a message
saying so, because a command line copied from vLLM will carry
--kv-cache-dtype fp8 and a generic parse error would send the reader looking
for a typo instead of a missing feature. int8 remains available and remains
documented as costing accuracy (#65).

Error messages name the flag the caller can actually pass, and the bound is
derived from kMaximumConcurrency rather than restated — this message read
"--max-concurrency must be in [1,8]" long after the ceiling moved to 64.

Old spellings are hard errors, verified one by one. The serve options test
moved with them and passes. Bench binaries keep their own flag vocabularies;
they are separate tools, not the server.

## 67. The 4B's fp4 helper kernels resist tuning (2026-08-27)

The fp4 path's helpers are 9% of the 4B's device time and look like pure
overhead around the GEMMs: w4fp4_act_quant_atom 223 ms across 17,076 launches
(4.3%), w4fp4_swiglu_pair 138 ms, w4fp4_split2 110 ms.

The activation quantizer reads each row TWICE — once to reduce the token max,
once to quantize — and with groups <= kThreads each thread re-reads the same 32
bytes it already had. Caching those values in registers so the second pass
replays from them is the obvious fix, and it measures NEGATIVE:

  baseline                       4B 3,032   0.8B 6,692
  cache 2 groups per thread      4B 2,986   0.8B 6,687
  cache 1 group per thread       4B 3,010   0.8B 6,720

Both worse on the 4B; the 0.8B's +28 is inside run variance. The extra
registers cost more occupancy than the saved read returns, which says the
kernel is not read-bound in the way its two passes suggest — at 13 us for a
2.6 MB row it is short enough that launch and occupancy dominate.

Reverted. What this rules out is tuning this kernel from the outside; what
remains open is removing it, by folding the quantization into the producing
rmsnorm's epilogue (the RmsEpilogue template parameter already exists) so the
row is never re-read or re-written at all. That is a plumbing change — the
workspace the codes land in is allocated by the GEMM's dispatch, downstream of
the norm — and it is the only version of this idea with a mechanism behind it
rather than a hope.
## 68

KV cache sizing was blind to the derived weight residencies, so `--kv-capacity
auto` over-committed and the engine died during CUDA Graph capture.

The FP8/FP4 and Marlin planes hold a repacked copy of every W8 weight they
adopt. They are allocated lazily — during graph warmup, or on first use when
graphs are off — which is *after* `resolve_kv_capacity` has already sized and
committed the cache. Measured with a `SUROGATE_SERVE_MEM_TRACE=1` counter:

  4B    derived planes 2,774 MiB + Marlin 4,086 MiB = 6,860 MiB  (1.40x weights)
  0.8B  derived planes   474 MiB + Marlin   770 MiB = 1,244 MiB
  27B   0 MiB — NVFP4-resident, nothing to derive

The policy therefore spent the card down to 72 MiB free and the graphs had
nowhere to go. It surfaced three different ways depending on configuration,
which is why it looked like three bugs: an abort during capture with graphs on,
and — with `--enforce-eager`, where the planes are derived later still —
17,733 requests of silently corrupted output, because a failed derivation is
reported by returning a null plane that callers read through anyway.

Three changes:

  * Sizing projects the residency from the resident weight formats, so an
    NVFP4 or FP8 artifact projects nothing and keeps the capacity it had. A
    new architecture inherits the reserve without naming itself.
  * `DecodeGraphDefinition::capture`, `instantiate` and `upload` throw instead
    of `CUDA_CHECK`. cuda_check calls std::abort(), so the eager-fallback
    catch in prefill_graph.h could never run — it was dead code for exactly
    the failure it was written for.
  * A graph that will not fit now exits cleanly naming `--enforce-eager`
    rather than degrading. Degrading is wrong here: the device has no margin
    left, and the other lazy allocators fail a moment later without saying so.

  4B `--kv-capacity auto` @64: crash -> 3,015 tok/s, 446,656-token cache
  0.8B 6,788 explicit / 6,589 auto · 27B 595 · 4B 3,038 explicit — all err=0

## 69

FP8 (e4m3) KV cache, default on, with a per-layer skip list.

Storage is the raw e4m3 byte with no scale plane, which is what vLLM writes
for `--kv-cache-dtype fp8` when no calibration is present: its k_scale/v_scale
default to 1.0, so its scaled_convert reduces to a cast. Values above 448
saturate rather than rescaling, matching that behaviour.

Compute stays bf16. The codes are widened on the way into the shared tile, so
the MMA, the swizzle and the shared-memory budget are unchanged and the bf16
decode/prefill kernels are simply templated on their cache storage type rather
than forked. The int8 cache needs its own kernel only because it feeds IMMA.
That containment is deliberate: FA3 is known to *lose* throughput on fp8 KV at
large head dimensions, and ours is 256.

Measured, 90 s, 100 users:

  0.8B  bf16 6,837 -> fp8 6,529   KV 3.71 -> 2.96 GiB
  27B   bf16   594 -> fp8   582   KV 46,016 -> 92,096 tokens (exactly 2x)

So it is a memory feature, not a throughput one: a few percent of decode buys
double the cache. The 27B is not KV-capacity-bound at 48 lanes — its
bottleneck is prefill duty cycle — so doubling a cache that was not full
returns nothing there.

Only full-attention layers own KV planes (`plan_cache` is built from
`full_attention_layers`), so a quantized cache structurally cannot reach a
linear-attention layer; the 35B-A3B asserts 10 such layers out of 40.
`--kv-cache-dtype-skip-layers` holds named full-attention layers at bf16 for
the sensitive ones. Verified by size, since bf16 and e4m3 layers occupy the
same two planes: 0.8B 3.71 (bf16) / 2.96 (fp8) / 3.21 GiB (fp8 skipping 0,1).

Not yet supported: `--spec dflash`, whose kv_cache_append_prefix path is
BF16-only and will refuse an e4m3 cache.

## 70

`--spec dflash` now refuses an e4m3 cache at startup rather than mid-round.
Its draft commit runs through `kv_cache_append_prefix`, which validates
`dtype != DType::BF16` and throws; with fp8 becoming the default that would
have surfaced as an exception the first time a draft landed, long after the
server reported itself healthy. Supporting the pair properly means an e4m3
path in that append kernel, which is not written.

## 71

The mixed-round corruption, root-caused: the mixed graph's cache key never
carried the frontier band.

PATCHES.md #55 meant to put the band in the key and passed
`mixed_profile.topology_class`. `graph_profiles_through` builds every profile
as `{begin, end}` and never sets that field, so it is zero for every band, and
the key was `(chunk_bucket, batch)` alone. The first mixed graph captured for a
pair — captured while every lane was young, with the low band's
`max_visible = 512` baked into its decode attention grid — replayed for the
rest of the process. Every lane that later crossed frontier 512 had its
attention truncated inside mixed rounds: a valid forward pass over the wrong
window, which is why the logits were finite and confident and simply wrong.

The key now carries the ordinary profile's index, which identifies the band
exactly. The ordinary decode graphs were never affected: they share one
executable per batch width and re-parameterize per band through
`executable.update`.

Why it hid for so long, in the measurements that finally pinned it:

  * Victims were always lanes past 512, always in mixed rounds, never in
    ordinary decode rounds.
  * Long prompts (lanes start past 512), short generations (lanes never reach
    it), and a diagnostic run with the band edge moved to 1023 (lanes never
    cross it) were all clean — the only clean configurations, and the only
    ones in which a single band covers a lane's whole life.
  * Moving the edge took the 27B from 25 corrupted streams in 300 s to zero;
    restoring the edge and fixing the key does the same.
  * The balanced 512/128 board rows never reach it: their lanes start past
    512. Every board number stood on a configuration that cannot exercise
    this path, which is an argument for a decode-heavy row on the board.

Refused along the way, each by measurement rather than reasoning: the prefill
graph's 128-rounded KV window (#55's hazard on the plain path — real, but not
this), the Marlin wide band (narrow and off both still corrupt), the KV-split
clamp, prefill ownership steals (never happened), double-used KV pages (an
audit stayed silent), the sampler (it faithfully took the argmax of wrong
logits), prompt content (identical prompts still corrupt), and the fp4
activation-quant path (off still corrupts). An occupancy guard that refused
near-full mixed rounds did stop it, at 16.6% of the 4B's throughput — rejected
as a fix, and useful only as the observation that pointed at the mixed graph.

Method note for the next one of these: the toggles that disable one graph
family (`SUROGATE_SERVE_NO_MIXED_GRAPH`, `SUROGATE_SERVE_PREFILL_GRAPH=0`)
corrupted fast at every lane count and were dismissed as "partially graphed
state, not a control". That was wrong — see #72: `PREFILL_GRAPH=0` routes
prompts through the eager prefill, which had its own bug, so the toggle was a
faithful control for a defect that had not been found yet. Two overlapping
bugs with one symptom is exactly when a toggle looks broken. compute-sanitizer
memcheck cannot reach the graph-mode bug (its slowdown caps occupancy below
the batch it needs, and it cannot see inside a captured graph); initcheck
found the eager one in a single run. What worked for this one was making the
failure survivable, attributing each event (row, lane, frontier, token,
logits), and running the factor matrix across GPUs in parallel.

## 72

The eager-prefill corruption: the direct recurrent kernel still read the GDN
state as fp32 after the storage moved to bf16.

`recurrent_bf16_direct_kernel` — the non-chunked recurrent form the eager
prefill takes for an exact-length remainder — kept `float*` state parameters
and an fp32 launcher cast when the state pool became `GdnStateStorage`
(bf16). It therefore addressed a half-size buffer with fp32 strides: reading
past the slot's written extent (compute-sanitizer initcheck: "uninitialized
__global__ memory read of size 16 bytes" at the state load) and writing back
twice the bytes into whatever followed. Every other launcher in the file had
been converted; this one had not.

Only the eager prefill reaches that form steadily: the captured prefill body
runs 128-multiple bucket lengths, which the chunked scan covers exactly, while
the eager body runs exact lengths (545 = 8 x 64 + 33) whose tail is handed to
the direct kernel. That is why `--enforce-eager` corrupted young lanes within
seconds — 22 of 174 requests on the 0.8B board shape — and why
`SUROGATE_SERVE_PREFILL_GRAPH=0` corrupted at every lane count: that toggle was
not "broken", it isolated the culprit, and was misread as noise (#71's method
note is corrected by this).

After: 1 corrupted request in 14,415 on the same shape, initcheck reports no
kernel read of uninitialized state, and the eager path's 916 tok/s (most of it
rejections after the fatal) becomes 6,100.

## Open: a rare 0.8B corruption, ~4 per 100,000 requests

What remains after #71 and #72, measured on the fixed binary:

  graph mode, 0.8B, 512/128, 600 s on two GPUs:  3 events in 81,319 requests
  eager mode, same shape, 300 s:                  1 event  in 14,415 requests
  4B, 27B, same shape, 300 s:                     0 in 18,500 and 4,000

Every victim is in a mixed round, in the high band, with the graph replayed,
and the victim's own frontier a few tokens below the batch maximum (654–676
against 685–688); victims skew late in their 128-token generation. It
predates this session: it is the intermittent 0.8B invalid-UTF-8 fault that
5fa6603e pinned the Marlin band over, and the 600 s matrix here shows the
narrow band does not prevent it (2 events in 30,857 requests), nor does
Marlin off, bf16 KV, the fp4 path off, or 32 lanes — each of those ran too few
requests to separate a 4-per-100k rate from zero. At this rate a single-GPU
run needs 250,000 requests for a ten-event sample, which is the method for
the next attempt: `SUROGATE_SERVE_SURVIVE_CORRUPTION=1` across all eight GPUs
for an hour per factor. With the fatal restored it kills a 0.8B worker about
once per 25,000 requests under 100-user load.

## 73

Rewrite checkpoints are off by default; `--rewrite-checkpoints` opts in.

The GDN state pool held two slots per lane plus the prefill scratch slot:
the lane's live state and a rewrite checkpoint from which an edited last
turn can resume instead of re-prefilling its prefix. On the 27B a slot is
48 GDN layers x 48 value heads x 128 x 128 x bf16 = 72 MiB, so at 48 lanes
the checkpoints alone held 3.4 GB, and the minimum runtime reservation
(8.9 GB) dwarfed the KV cache (3.0 GB). That is why lanes above 48 fell off a
cliff on the 27B: at 64 lanes the pool grew to 9.1 GB and `--kv-capacity
auto` was left with 8,576 tokens.

The slot layout is now [0, N) lanes, N scratch, then — only when checkpoints
are enabled — [N+1, 2N+1) checkpoints. Disabled, the pool is N+1 slots, the
checkpoint index is the sentinel kNoRewriteCheckpointSlot, the planner never
desires, captures or defers a checkpoint (a prompt that describes one is
simply dropped — re-prefilling that prefix is slower, never wrong), and any
path that still reached for a checkpoint slot would index past the pool and
fail loudly. Ordinary multi-turn append never used the checkpoint; it
continues on the lane's live state as before.

27B, fp8 KV, `--kv-capacity auto`, KV capacity with checkpoints off:

  48 lanes   92,096 -> 206,976 tokens
  56 lanes             184,384 tokens
  64 lanes    8,576 -> 161,792 tokens

Verified on the new default (checkpoints off), fp8 KV:

  27B decode-heavy soak, 48 lanes, 300 s:   820 requests, 0 errors, 0 fatals
  4B  decode-heavy soak, 64 lanes, 300 s: 3,846 requests, 0 errors, 0 fatals
  board shapes, same cards as the paired rows: 0.8B 7,721 (was 7,666),
    4B 4,368 (was 4,359), 27B 822 (was 827) — unchanged within noise
  27B at 64 lanes (GPU7): 859 tok/s, TTFT p50 5.4 s (48 lanes on the same
    card: 824 tok/s, 8.0 s) — the lane cliff is gone; 64 is now the better
    default for the 27B, and the small gain confirms the 27B is prefill-bound,
    not lane-bound (BENCHMARKS.md, prefill table).

## 74

Qwen3.6-35B-A3B serves on one 32 GB card; the DFlash drafter is optional.

The target was already complete — registry, config (40 layers, 10
full-attention + 30 GDN, 256 experts top-8 plus a shared expert), mixed
Q4/Q5/Q6 routed-expert bindings, and the decode / prefill / small-T sparse
MoE kernels — and BENCHMARKS.md's "~35 GB W8 artifact" note described a build
the bindings no longer request. The committed recipe produces a 22.4 GB
artifact whose resident core fits a 5090 with 8.4 GB left after weights, so
the FreeToken-style expert offload studied for this model is not needed for
it; that design targets the 122B/397B tiers and long-context KV pressure.

What actually blocked conversion:

  * `--dflash-model` was required. The DFlash drafter is a separate
    checkpoint that is not public; the converter now omits the dflash/*
    family when it is absent (889 objects instead of 940) and the 35B
    bindings probe for the family (`has_dflash`, the has_mtp pattern) and
    refuse `--spec dflash` against such an artifact with a startup error
    instead of a missing-object failure.
  * The recipe's exact-source contract addressed tensors in the nested
    `model.language_model.*` dialect while the shard reader folds that to
    `model.*`; the 27B NVFP4 recipe already compared in the folded dialect,
    and the 35B preflight now does the same.

First serve, GPU7, fp8 KV, 64 lanes, checkpoints off, `--kv-capacity auto`
(456,192 tokens), 100 users, 90 s:

  balanced 512/128:        1,596 decode tok/s, 6,384 prefill tok/s, TTFT 2.9 s
  prefill-heavy 2048/16:  13,531 prefill tok/s, TTFT 14.4 s
  coherent at temperature 0; 0 errors, 0 fatals

Also in this entry: `SUROGATE_SERVE_PREFILL_TIMING` now prints a per-family
split of eager prefill (attention mixer, GDN mixer with its in-proj / conv /
recurrent scan / out-proj sub-split, and the MLPs), measured with events
between the ops of the mixed-round body. It exists because nsys would not
finalise a report in this environment (CPU sampling is unavailable at
paranoid level 4, and the session's earlier profiles could not be reproduced).
On the 27B's prefill-heavy shape it reads: GDN mixer 41.9%, MLP after GDN
32.9%, attention 14.3%, MLP after attention 11.0% — the linear-attention
mixers are the largest single cost of 27B prefill, which is the block vLLM
serves with FLA's Triton chunked kernels and where its 2x prefill lead lives.

## 75

**NVFP4 W4A4 GEMMs: split launch so mixed serving rounds reach the TMA schedule (2026-08-27).**

**Symptom.** The NVFP4-native 27B trailed vLLM by 2× on prefill (same card:
5,964 vs 12,089 prompt tok/s) while the W8 0.8B/4B artifacts led. The
per-family prefill timer put 42% of eager prefill in the GDN mixers and half
of that in the input projection, running at ~285 TFLOP/s against ~580 for
the MLP.

**Root cause.** Every W4A4 launcher (`linear`, `attn_input_proj`,
`gdn_input_proj`, `linear_add`) gated the TMA schedule on
`tokens >= 1024 && tokens % 256 == 0`. A serving round is a mixed
`[prefill chunk | decode batch]` GEMM — 1,072 + 5 or 1,024 + 63 columns —
which is never block-aligned, so under load every prefill GEMM on the 27B
fell to the mma ladder (`M128N128Resident`). The balanced board shape
(545-token prompts) never reached the 1,024 floor at all. The W8 artifacts
use a different GEMM family and were never affected, which is why only the
27B was behind.

**Fix.** `ops/linear/nvfp4/nvfp4_w4a4_split.h`: `nvfp4_w4a4_tma_split`
hands the 256-aligned prefix to the TMA schedule and the ragged tail to the
existing mma ladder, with token-major base offsets for the activation
workspace (`K/2` code bytes, `K/16` scale bytes per token) and every output
plane (`rows` elements per token). All four launchers use it; each ladder
now takes raw activation/output views instead of tensors. The floor below
which the whole problem stays on the ladder is `SUROGATE_SERVE_NVFP4_TMA_MIN_TOKENS`
(default 512: the smallest prefix with two M-tiles that still splits the balanced shape's ~590-token rounds). A `static_assert` in `nvfp4_w4a4_tma.cu` ties the TMA
schedules' block-M to the helper's constant.

**Verification.** Op tests gained ragged cases (300, 1,077, 1,300 tokens)
in all four W4A4 suites; they pass at the default floor and at 256
(`ninfer_linear_nvfp4_a4_test`, `ninfer_gdn_input_proj_test`,
`ninfer_attn_input_proj_test`, `ninfer_linear_add_nvfp4_test`; the
linear_add harness sized its buffers from a 1,024-token constant and was
raised to 1,300).

27B, 48 lanes, fp8 KV auto, 100 users, single runs:
| shape | old path (TMA off) | split, floor 1024 | split, floor 256 |
|---|---|---|---|
| prefill-heavy 2048/16, GPU6 | 5,867 prompt tok/s | 6,088 | 6,244 |
| balanced 512/128, GPU7 | 826 decode / 3,303 prefill | 822 / 3,289 | 841 / 3,362 |

The gain is real but small (+4-6% prefill-heavy, noise on balanced) because
the two in-house schedules are close: the GDN input projection runs at 560
TFLOP/s on the TMA schedule and 545 on the mma ladder at ~1k tokens
(`ninfer_gdn_input_proj_bench`, GPU2). The six-card matrix that suggested
+23% was card spread, not the floor - the same-card controls above are the
record. A cuBLASLt block-scaled FP4 matmul
(`CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`, CUDA 13.1) on the same card and
shapes reaches 763 TFLOP/s at T=1,024 and 827 at T=2,048 (GDN geometry; MLP
gate-up 825/850; residual 700-790): 1.35-1.45x the in-house W4A4 kernels at
every prefill width. That, and the sequential GDN recurrent scan (23% of the
GDN family's prefill time, where vLLM runs FLA's chunked kernel), are the
27B prefill levers left; see BENCHMARKS.md.

## 76

**NVFP4 W4A4 prefill GEMMs on cuBLASLt's block-scaled FP4 matmul (2026-08-27).**

**Why.** With #75 in, the in-house W4A4 schedules were measured against
cuBLASLt (`CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`, CUDA 13.1) on the
27B's own shapes: 763 vs 560 TFLOP/s on the GDN input projection at 1,024
tokens, 819 vs 575 on the MLP gate-up, 400 vs 215 on the 5120×6144 residual
at 300 tokens — 1.4–1.9× at every prefill width, and slower only below 64
tokens where the in-house small-T kernels already run at the weight-bandwidth
limit (T=16: 35 µs vs 76 µs).

**What made it free.** The artifact's NVFP4 storage already is cuBLASLt's
layout: codes row-major with K contiguous and even columns in the low
nibble, scales in the canonical 128×4 tile
(`(row%32)*16 + (row/32)*4 + group%4`, K-tiles inner). Nothing is
repacked; `ninfer_linear_nvfp4_cublaslt_test` shows the two paths bit-exact
on identical quantized inputs (max abs diff 0.0000 on all three geometries).
Only the activation scales change: `Nvfp4ScaleLayout::Tiled` makes the
quantizer write the same tiled layout, and the W4A4 workspace pads the scale
interval to 128 rows.

**Route.** `ops/linear/nvfp4/nvfp4_cublaslt.{h,cpp}`: one handle, a 32 MB
workspace and a descriptor/algorithm cache per device, keyed by (rows, K,
tokens, ld, accumulate); scale pointers are rebound per call because they
follow the weight, not the shape. `nvfp4_cublaslt_route(tokens)` is true
from `kNvfp4CublasLtDefaultMinTokens` (64) up; `SUROGATE_SERVE_NVFP4_CUBLASLT=0`
disables it and `SUROGATE_SERVE_NVFP4_CUBLASLT_MIN_TOKENS` moves the
threshold. The five W4A4 families use it: `linear` (whole problem),
`attn_input_proj` (four row-sliced calls straight into q/k/gate/v — the
slices are 128-aligned so the scale tiles cut cleanly), `gdn_input_proj`
(qkv and z), `linear_add` (`beta = 1` on the residual, in place), and the
fused MLP, whose plan already runs T > 48 as a plain linear plus `silu_mul`
and now prefers that path at T = 1024 as well when the route is on.

**Verification.** All six NVFP4 suites pass with the route on and with it
disabled (`SUROGATE_SERVE_NVFP4_CUBLASLT=0`); the workspace high-water
checks stay exact because the padded scale interval is planned and used
identically on both paths.

27B, 48 lanes, fp8 KV auto, 100 users, same card per shape, one run each:
| shape | route off | route on | delta |
|---|---:|---:|---:|
| prefill-heavy 2048/16, GPU3 (prompt tok/s) | 6,139 | 6,528 | +6.3 % |
| balanced 512/128, GPU4 (decode / prompt tok/s) | 672 / 2,687 | 717 / 2,869 | +6.7 % |

Consistent but far below the kernel-level 1.4–1.9×: a 1,024-token chunk
takes ~150 ms end to end on the 27B and its GEMMs, even at the old speed,
are only ~55 ms of that. The prefill family timer's `in_proj` bucket (50 %
of the GDN family) therefore holds more than the GEMM; the next step is the
post-#76 breakdown, taken eager on the same shape.

## 77

**Long prompts killed the worker: the mixed prefill window was mapped past the request's KV entitlement (2026-08-27).**

**Symptom.** Four users sending ~2,200-token prompts (or a hundred sending
~4,200) took the worker loop down within seconds: `engine worker loop fatal:
Paged KV materialize extent is outside entitlement [kind=mixed …]`, every
in-flight request failed. The board's 2,146-token loadgen prompts never hit
it.

**Root cause.** #55 made the mixed round map the whole 128-rounded chunk
window before replay — the graph writes its pad columns, and unmapped pad
columns were the mixed-round corruption. The window was clamped to the KV
capacity but not to what the request owns: admission reserves
`pages_for_tokens(prompt + output − 1)` in 64-token pages. Whenever the
tail chunk's 128-rounding crosses into a page the request does not own
(prompt length mod 128 decides), `materialize_pages` throws and the round
dies. 2,146 → tail 98 → rounds to exactly the last owned page; 2,200 →
tail 152 → one page past it.

**Fix.** `request_plan_impl.h` reserves
`max(prompt + output − 1, round_up(prompt, 128))` (capped at capacity) — at
most two extra pages per request — so the graph's window always lies inside
the entitlement. The throw stays as the invariant check.

**Verification.** Before the fix, four users with ~2,200-token
prompts killed the worker within seconds (3 ok / 304 errors) and a hundred
users with ~4,200-token prompts within one round (1 ok / 16,861 errors).
After it: 2,200 tokens at 4 users — 80 ok, 0 errors; 4,200 tokens at 100
users, `--max-num-batched-tokens 4096` — 201 ok, 0 errors, 6,869 prompt
tok/s (27B, GPU7). The throw now reports the requested, mapped and entitled
page counts so a recurrence is diagnosable from the log line.

Also in this change: the prefill family timer gained a seven-way GDN
sub-split (norm+control, input projection, conv, column extract, chunked
scan, gated norm, output projection) in both prefill paths — the plain
`run_layers` path had no sub-laps at all, which is why one-user runs printed
none.

## 78

**FP8 W8A8 prefill GEMMs on cuBLASLt (2026-08-27).**

**Why.** The 27B keeps its GDN input projection (16384×5120), attention QKV
(14336×5120) and both output projections (5120×6144) as FP8-row weights
(`FP8_E4M3FN_ROW_BF16S`) on the in-house W8A8 kernel. Nsight Compute over a
one-user prefill puts that one kernel at **52 %** of the chunk (529 µs
average), ahead of the cuBLASLt NVFP4 MLP GEMMs (25.6 %); it runs at ~330
TFLOP/s in the engine where cuBLASLt's FP8 GEMM does 651–750 at the same
shapes.

**Route.** `ops/linear/fp8/fp8_cublaslt.{h,cpp}` and
`fp8_cublaslt_finish.cu`. cuBLASLt on this device only accepts scalar FP8
scales (the outer-vector modes return INVALID_VALUE / NOT_SUPPORTED), so the
GEMM accumulates the raw e4m3 codes — the artifact's weight codes and the A8
quantizer's per-token activation codes, both consumed as they are — into an
fp32 staging, and a finish kernel applies the weight's per-row bf16 scale and
the activation's per-token fp32 scale exactly once, adding the residual for
`linear_add`: the in-house epilogue's arithmetic with cuBLASLt's
accumulation. `Fp8A8Workspace` carries the staging (sized by the largest
output segment; the four plans reserve it when the route applies), and the
four A8 launchers (`linear`, `attn_input_proj` with its four 16-aligned row
segments written straight into q/k/gate/v, `gdn_input_proj` with qkv and z,
`linear_add` accumulating into the residual) take it from
`kFp8CublasLtDefaultMinTokens` (128) up. `SUROGATE_SERVE_FP8_CUBLASLT=0`
disables the route, `SUROGATE_SERVE_FP8_CUBLASLT_MIN_TOKENS` moves the
threshold.

**Verification.** The FP8 suites (`ninfer_linear_fp8_a8_test`,
`ninfer_linear_fp8_a16_test`, `ninfer_linear_add_fp8_test`, and the FP8 cases
of `ninfer_attn_input_proj_test` / `ninfer_gdn_input_proj_test`) gained
ragged 300- and 1,077-token cases and pass with the route on and with
`SUROGATE_SERVE_FP8_CUBLASLT=0`.

27B, 64 lanes, `--max-num-batched-tokens 4096`, 100 users, same card per
shape, one run each:

| shape | route off | route on | delta |
|---|---:|---:|---:|
| prefill-heavy 2048/16, GPU6 (prompt tok/s) | 6,619 | 7,339 | +10.9 % |
| balanced 512/128, GPU5 (decode / prompt tok/s) | 949 / 3,797 | 960 / 3,841 | noise (decode batches stay below the threshold) |

Less than the kernel ratio promises, and the route-on profile says why: the
finish pass is only 4.7 % of the chunk (a bf16 staging is not worth the
rounding), while the in-house FP8 kernel still holds 10.6 % on the tails
below the threshold. The threshold is 65 — one above the concurrency cap, so
a decode round never takes the route: at 64 the decode rounds did, and since
the workspace plans size the staging from `max_tokens`, every engine died at
startup with "workspace arena exhausted". Per-window laps then showed the
remaining 27B deficit is not kernel time at all (see BENCHMARKS.md): single
stream the engine is within 12 % of vLLM, and the 1.6× gap at 100 users is
vLLM batching several prompts per prefill step.

## 83

**The 4B artifact carried 8-bit weights where vLLM's carried 4-bit (2026-08-27).**

**Symptom.** The 4B was the one model the engine lost on throughput: 3,218
decode tok/s against vLLM's 4,246 on the same card and shape, while winning
TTFT 70 ms to 239 ms. Nsight put the Marlin W8 GEMMs at 52 % of decode time
— our artifact was 5.26 GiB of W8G32, vLLM's an NVFP4 export.

**Fix.** A converter for `AxionML/Qwen3.5-4B-NVFP4`, a ModelOpt export whose
one checkpoint supplies the whole artifact: NVFP4 blocks pass through
untouched (fused q,k,gate,v and qkv,z the way the W8 recipe fuses them),
`in_proj_a`/`in_proj_b` are decoded back to BF16, the tied embedding is
re-encoded W8 for the byte-wide head. 3.56 GiB, and a new
`Qwen35Nvfp4Mixed` weights profile keyed on `weights_id == "nvfp4-mixed"`.

Two traps, neither of which announces itself in the served text — the model
answers, it just answers wrongly, which reads exactly like a kernel bug:

  - ModelOpt writes `weight_scale_2` and `input_scale` as **multipliers**
    (`amax/(448*6)`). The runtime's `weight_scale_divisor` /
    `input_scale_divisor` are the compressed-tensors convention the 27B
    recipe reads: block scale is `divisor * max_abs / 6` and the GEMM undoes
    it with `alpha = 1/(input_div * weight_div)`. Passed through unchanged,
    every GEMM was off by the square of the scale.
  - the convolution ships channel-major `(8192,1,4)`; the artifact stores it
    tap-major `(4,8192)`. The direct-object path reshapes, and a reshape
    reinterprets those bytes instead of permuting them. The W8 recipe has
    the transpose; the NVFP4 one had dropped it.

Both were found by decoding the written artifact back and diffing it against
the checkpoint — weights, block scales and divisors are bit-exact now. Do
that before reading anything into what a converted model says.

**Result.** 4,942 decode / 19,767 prefill tok/s, TTFT p50 45 ms: **+54 %**
over our own W8 artifact and **+16.4 %** over vLLM, at 5.3× its TTFT.

## 84

**Every fused NVFP4 op was gated to the 27B's geometry (2026-08-27).**

**Symptom.** The new 4B NVFP4 artifact would not load: `attn_input_proj
workspace: unsupported NVFP4 profile`, then the same for `gdn_input_proj`,
`linear_add` and `linear_swiglu` in turn. Each op checked its weight against
one registered shape (14336/5120 attention, 16384/5120 GDN, 34816/5120 MLP,
5120/6144 and 5120/17408 residual) because that is where its in-house W4A4
ladder is instantiated.

**Fix.** Keep the ladders registered-only and send everything else through
cuBLASLt, which is generic in n and k (#76 already routes W4A4 from 64
tokens up). `is_nvfp4_generic_problem` (n % 128 == 0, k in {2560, 4096,
9216}) admits a shape; then:

  - the attention and GDN input projections quantise once and issue one
    row-sliced GEMM per output segment, with the split taken from the output
    views rather than baked constants — the 128-row alignment the tiled
    scale plane needs holds for 4096|1024|4096|1024 and 8192|4096
  - `linear_add` and the plain linear resolve generic shapes to W4A4 at
    every T: there is no A16 ladder for them to fall back to
  - `linear_swiglu` GEMMs into a BF16 plane and folds it with `silu_mul`
  - the GDN conv snapshot/record paths project-then-conv

The trap is in the plan, not the kernels: the conv snapshot capacity for W8
returns **zero** below 17 columns because W8 has a fused conv kernel there.
NVFP4 has none, so its generic path always materialises the projected plane
and its capacity has to carry it — otherwise the arena is short by exactly
one plane (`request 20480 bytes at offset 17664 exceeds capacity 21760`)
the first time a narrow snapshot runs.

The 27B keeps every fused kernel it had; its registered paths were
re-verified unchanged after the change.

## 85

**Speculation was capped at eight lanes, and two plan bugs sat behind it (2026-08-27).**

**Symptom.** `--spec mtp` refused any serving concurrency: at 96 lanes the
engine died at startup with `GDN replay record capacity exceeds eight rows`.
Lifting that cap exposed two more failures, both at startup — a
`cudaErrorStreamCaptureUnsupported` from `cudaMalloc`, then
`workspace arena exhausted: request 2703360 bytes at offset 338432 exceeds
capacity 338184`.

**Fixes.**

  - The replay row table was `GdnReplayFoldKernelRow row[8]`, and the two
    validation constants mirrored it. Nothing in the fold kernel needs the
    bound — it indexes rows with `blockIdx.y` — so the table is
    `kMaximumBatchColumns` wide now: 1 KiB of the 32 KiB a launch may carry
    as parameters.
  - Both cuBLASLt routes build their handle and 32 MiB workspace on first
    use, and a first use inside a graph capture cannot `cudaMalloc`. The
    ordinary families happen to touch the routes eagerly first; the MTP
    verify family's first NVFP4 GEMM is inside its own capture. Prewarm both
    in the program constructor, before `prepare_graphs`.
  - `snapshot_capacity` adds the FP8 route's fp32 staging to the plan;
    `record_capacity` did not. A record round wide enough to take the route
    (65 columns, and a verify batch is `lanes x (1 + draft tokens)`) then
    asked for staging nobody had planned. Unreachable before, because every
    caller of the record path ran below the threshold.

**What it bought, and what it did not.** Speculation is real on this model:
76.8 % of drafts accepted, 1.77 tokens per round at one draft token, and at
eight lanes +33 % (d=1), +48 % (d=2), +55 % (d=3). It still loses the
100-user board, for a reason that is memory rather than acceptance — it
reserves a second GDN state slot per lane (171 MB against 99), so the 27B
fits 48 lanes with it instead of 96. 48 lanes + d=1 measured 885 tok/s
against the 96-lane control's 996, which is what the round-cost model in
BENCHMARKS.md predicts. Until the shadow slot goes, `--spec mtp` belongs to
low-concurrency serving.

Also measured and reverted: rounding the prefill chunk ladder to 64 instead
of 128. A 552-token prompt runs a 640-column graph, and those 88 pad columns
are 7.5 % of the second at 114 us each — but the finer ladder measured
+2.8 % once and −1.2 % with the cards rotated, and cost the 4B 3.6 %.

## 87

**The 27B lost on weight format, and the fix needed the GDN pair unfused (2026-08-28).**

**Symptom.** The 27B was the one model behind vLLM (976 against 1,039), and
every scheduling lever had been measured and rejected: prompt batching is
algebraically a no-op under the round-cost model, lane sweeps 72-104 moved
nothing, MTP speculation costs more lanes than it earns, a finer chunk ladder
was noise. What remained was the 114 us column — and Nsight had already said
where it goes: the in-house FP8 GEMMs are 52 % of a prefill chunk at 651-750
TFLOP/s where the NVFP4 ones reach 827.

**Root cause.** Our artifact kept the attention and GDN projections FP8-row
only because `unsloth/Qwen3.8-27B-NVFP4` exports them that way.
`sakamakismile/Qwen3.8-27B-MTP-NVFP4` — the checkpoint vLLM itself is served
from on this board — quantises every language linear, ignore list the vision
tower alone, and carries the bf16 embedding, lm_head, MTP block and vision
tower besides. One checkpoint, whole artifact.

**The obstacle, and why it was not worked around.** The GDN input projection
fuses `in_proj_qkv` and `in_proj_z` into a 16384-row object, and an NVFP4
object carries one weight divisor. This export gives the two sources different
weight global scales (layer 0: 6176 against 11264). Restating one half onto the
other's divisor measures ~2 % mean relative error with 93 % of its values
moving — a second quantisation of every GDN gate weight. That is a quality
trade, so the halves stay apart instead:

  - `gdn_input_proj_split` and its `_conv_snapshot_split` / `_conv_record_split`
    twins run the projection as two GEMMs. The cuBLASLt route already issues
    two row-sliced GEMMs for the fused weight, so nothing is lost; snapshot
    projects into the shared plane, record into the caller's, and both take the
    existing projected convolution.
  - `SplitQkvZGdnInputProjectionPayload` carries the pair — a third
    arrangement, distinct from the Q4/Q5 `qk|value_z` split.
  - `is_nvfp4_generic_problem` admits k=5120 for the halves. **It now excludes
    registered shapes explicitly**, and that guard is the point: without it,
    admitting 5120 would have pulled the 27B's own 34816x5120 MLP off its
    in-house small-T ladder onto cuBLASLt at every token count, which is slower
    below the route's threshold.
  - the converter's `--resources-from` borrows the frontend resources from an
    existing artifact: this export is the MTP+vision variant and its
    `tokenizer_config.json` does not satisfy the target's Qwen3.6
    prefix-semantics check, while its weights are the same model.

**Result.** Weights 18.98 -> 15.16 GiB, so KV holds 1,471 pages instead of 931
and 99 sequences run instead of 84. Two passes, cards rotated: **1,330 tok/s
against vLLM's 1,039 (+28 %), and +37 % over our own mixed artifact**, at
170 ms TTFT against 8.26 s. Every object decodes bit-exact against the
checkpoint. 128 lanes is the configuration — decode ties 96 lanes, TTFT is
2.4x better.

The lesson at two scales now: on this hardware the weight format is worth more
than every scheduling lever put together. The 4B gained 54 % from it (#83), the
27B 37 %, and in both cases the levers ranked above it moved nothing.
