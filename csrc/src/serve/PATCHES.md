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
   backed by these vendored recipes; one shared GGUFReader across
   summary/bridge. Qwen3.5-0.8B Q8_0: 159/195 Q8_0 tensors repacked,
   one-time conversion 17.7s total vs ~36s full-dequant (converter core
   3.6s), engine output identical. Bit-exactness is pinned by
   tests/serve/test_gguf_repack.py against gguf-py's own dequantize.

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
