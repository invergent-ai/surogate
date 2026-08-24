# Vendored code under csrc/src/serve/

## ninfer/  — vendored from https://github.com/Neroued/ninfer

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
