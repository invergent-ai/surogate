# Vendored code under csrc/src/serve/

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
