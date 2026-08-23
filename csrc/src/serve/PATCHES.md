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
