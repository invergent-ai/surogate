# P0 Port-Validation Spike — NInfer q4 rowsplit family (E2, §2.2/§13)

**Date:** 2026-08-23. **Claim under test** (§2.2a): NInfer's kernel bodies are sm_80-class CUDA with Blackwell asm confined to `src/ops/common/mma.cuh`, so the same code covers sm_89 with mechanical guards. **Result: the claim survives, and is stronger than stated — the entire q4 file set compiles for sm_89 with ZERO source changes** (the plan budgeted for PDL/f8f6f4/fp4/carveout guards; none of those constructs instantiates in the q4 A16 family). Parity of the packed layout + T=1 GEMV + T=8 small-T MMA verified on a 5090 at the BF16 rounding floor (max rel err 3.9e-3, tol 1e-2), peak 140 MB device. **sm_89 is compile+link-validated only — this host has 8×5090 and no sm_89 GPU; no sm_89 execution happened** (risk §12.1 stays open until the 4090 run).

Workspace: `/tmp/claude-1000/-home-densemax2-work-flavius-surogate/1598de2e-5193-47a9-bb25-f5f62bf83fc9/scratchpad/port_spike/` (`pristine/` = verbatim copies, `work/` = post-fix tree, `build.sh`, `harness/main.cu`, per-TU `.err` logs under `build_*`). Nothing under `study/ninfer/` was modified; no `make` was run.

## 1. Extraction: file list and include-graph findings

27 files close the include graph for the five q4 TUs. Verbatim copies, relative paths preserved:

| From `study/ninfer/` | Files | Why |
|---|---|---|
| `src/ops/linear/q4/` | all 13 (`q4_dispatch.{h,cpp}`, `q4_launch.h`, `q4_rowsplit_{gemv,gemm_mma,gemm_simt}.{cu,cuh}`, `q4_small_t_mma.{cu,cuh}`, `q4_rowsplit_storage.cuh`) | the family under test |
| `src/ops/common/` | `math.h`, `math.cuh`, `memory.cuh`, `mma.cuh`, `warp.cuh`, `token_slices.h` | direct includes (`bf16_vector.cuh`, also in `common/`, is NOT needed by q4) |
| `src/core/` | `tensor.{h,cpp}`, `dtype.{h,cpp}`, `device.{h,cu}`, `pdl.cuh`, `arena.h` | `Tensor`/`Weight` PODs, `CUDA_CHECK`, `Tensor::slice` (used by the simt/mma launchers' token slicing), PDL template plumbing |
| `include/ninfer/ops/` | `linear.h` | `LinearPolicy` enum consumed by `q4_dispatch.h` |

Include-graph observations (all favorable to the seam claim):

1. **The graph bottoms out in PODs.** `Tensor`/`Weight` are plain structs in `core/tensor.h`; the only host `.cpp`/`.cu` needed beyond the q4 TUs are `tensor.cpp` (slice/view arithmetic), `dtype.cpp` (one switch), and `device.cu` (only for `cuda_check`; replaceable by ~10 lines). No Engine, no arena implementation, no launcher-wrapper layer required — the extraction seam the plan describes (§2.2a "POD launch contract") is real.
2. **`arena.h` is pulled header-only** via `ninfer/ops/linear.h` (for the `WorkspaceArena = DeviceArena` alias in a doc signature); `arena.cu` is NOT needed — the q4 A16 family takes no workspace.
3. **`pdl.cuh` reaches sm_89 harmlessly.** The GEMV/SIMT kernels carry `TriggerPdl/JoinPdl` template parameters defaulting to `false`; every q4 launcher instantiates the false variants, so `cudaTriggerProgrammaticLaunchCompletion`/`cudaGridDependencySynchronize` are never codegen'd (verified: no PDL symbols in any sm_89 object). The sm_90+ guard the plan budgets is needed only where a launcher passes `true` — none in q4.
4. **`mma.cuh` confines Blackwell asm exactly as claimed.** `mma_fp8_e4m3` (`kind::f8f6f4`) and `mma_nvfp4_e4m3` (`kind::mxf4nvf4`) sit beside the sm_80-class wrappers as unused inline device functions; nvcc never emits them for q4, so they compile-through on sm_89. The one-line respell/compile-out remains a REAL obligation for the FP8/NVFP4 family ports — just not exercised here.

## 2. Compile matrices (nvcc 13.1, `-std=c++20 -O3 --expt-relaxed-constexpr`)

**(a) sm_120a, pristine — MUST be clean, is clean:**

| TU | sm_120a | sm_89 |
|---|---|---|
| `q4_rowsplit_gemv.cu` | clean | clean |
| `q4_rowsplit_gemm_mma.cu` | clean | clean |
| `q4_rowsplit_gemm_simt.cu` | clean | clean |
| `q4_small_t_mma.cu` | clean | clean |
| `q4_dispatch.cpp` (g++ 13, host) | clean | clean (arch-independent) |
| support: `device.cu`, `tensor.cpp`, `dtype.cpp` | clean | clean |
| `ar` → `libninfer_q4.a` + full harness link | OK | OK |

(Only build-script friction: nvcc wants `-arch=sm_120a`, not the CMake-style `120a`.)

**(b) sm_89 error log: EMPTY.** Every `.err` file under `build_89_pristine/` is zero bytes. There were no errors to fix, hence **`diff -ru pristine/ work/` is empty** — zero MECHANICAL and zero STRUCTURAL changes. The plan's claim ("mechanical guards only") doesn't just hold; for the q4 A16 family the guard classes (PDL launch, `kind::f8f6f4` respell, `cuda_fp4.h`/`cvt.e2m1` fallback, smem carveout) never even instantiate. The claim would die on a STRUCTURAL change; instead the diff is the empty set.

**Codegen sanity (not empty-stub compiles):** sm_89 `q4_rowsplit_gemm_mma.o` contains 1428 HMMA/LDSM instructions — the same count as sm_120a; GEMV objects carry LDGSTS (`cp.async`); kernel counts identical per arch (2/27/4/26 per TU). Resource usage:

| TU | sm_89 max SHARED / REG | sm_120a max SHARED / REG |
|---|---|---|
| gemv | 4128 B / 40 | 5152 B / 43 |
| gemm_mma | 45568 B / **237** | 46592 B / 168 |
| gemm_simt | 8704 B / 128 | 9728 B / 127 |
| small_t_mma | 28928 B / 40 | 29952 B / 40 |

All static smem fits sm_89's 48 KiB default (no carveout opt-in needed anywhere in q4 — the 96 KiB case the plan flags lives in the attention prefill family, unporteded here). One **perf flag for the 4090 run**: sm_89 ptxas allocates up to 237 regs/thread on the largest MMA schedules vs 168 on sm_120a — occupancy will differ and the route tables are 5090-swept; this is exactly the §2.2 per-card resweep obligation, not a correctness issue.

## 3. Packing spec as implemented (`row-split-k128-v1`, Q4G64_F16S) — the repack-encoder contract

Replicated bit-for-bit from `tools/artifact/layouts.py` (`row_split_geometry` + `encode_row_split` + `_pack_low_nibbles`) and validated against the kernels' decode atoms (`q4_rowsplit_storage.cuh`). For logical `[N, K]`, group size 64 along K:

- `k_pad = align_up(K, 128)` (the "k128"); `groups_per_row = k_pad/64`; padding groups are zero-filled codes and scales.
- **Codes**: signed int4, `q ∈ [-8, 7]`, two's complement in the low nibble; dequant is `w = q × scale`, **no zero point** (symmetric only — the plan's zero-plane gap is confirmed: nothing in this format carries an affine offset).
- **Base plane** (offset 0, `N × groups_per_row × 32` bytes): row-major `[row][group][32 B]`. Within a group of 64 codes, **byte j = (q[2j] & 0xF) | (q[2j+1] & 0xF) << 4** — sequential K order, even K index in the low nibble, odd in the high. Kernel-side decode confirms: `q0 = ((byte & 0xF) ^ 8) − 8` (low nibble → even K), `q1 = ((byte >> 4) ^ 8) − 8`; the word-decode path (`decode_eight`) consumes the same order 8-at-a-time via the `^0x88888888` + fp16-magic (`0x6400`+u − 1032) trick.
- **High plane**: empty for 4-bit (`high_bytes_per_group = 0`); for Q5/Q6 it holds the packed upper bits at `align_up(base_bytes, 256)`.
- **Scale plane** at `scale_offset = align_up(base_bytes, 256) + align_up(high_bytes, 256)`: FP16 words, row-major `[row][group]`, 2 B per group. Scales must be the exact FP16 values the encoder rounded to (the kernel does `__half2float` and multiplies in FP32 — `q × fp16-scale` is exact in FP32, ≤15 mantissa bits).
- Plane alignment 256 B (`PLANE_ALIGNMENT`); `Weight.qdata` → base plane, `Weight.qhigh` → high plane, `Weight.scales` → scale plane; `Weight.padded_shape[1] = k_pad` (the small-T launcher matches on it).
- Encoder used in the harness: per-group `scale = fp16(amax/7)`, `q = clamp(round(w/scale), −8, 7)` — matches `tools/convert/common/quantize.py` semantics.

## 4. Parity on sm_120 (RTX 5090)

Harness (`harness/main.cu`): packs random N(0,1) weights per the spec above into one payload buffer (planes at the exact artifact offsets), uploads, builds `Tensor`/`Weight` PODs, and calls **`ninfer::ops::detail::q4_dispatch`** — so the route table in `q4_dispatch.cpp` (used as the grid/schedule spec, per `q4_launch.h`) does the selection; no Engine involved. Reference: FP64 accumulation over the FP32-exact dequantized weights × BF16-exact activations (NInfer's own oracle discipline from `linear.h`).

| Case | Route selected (by table) | max abs | max rel | verdict |
|---|---|---|---|---|
| T=1, N=4096, K=5120 (decode GEMV) | `launch_q4_gemv_r1_w8_direct` (1 row/CTA, 8 warps, static 80 groups/row) | 9.71e-1 | **3.79e-3** | PASS |
| T=8, N=131072, K=2048 (small-T MMA, draft-head geometry) | `launch_q4_draft_head_small_t` → `q4_small_t_mma_kernel` (BF16-decode + `mma_bf16`, FP32 accum, per-64-group scale in FP32) | 5.00e-1 | **3.89e-3** | PASS |

**Tolerance derivation:** the output is BF16 (8 significand bits) → half-ulp rounding alone bounds rel err at 2⁻⁸ ≈ 3.9e-3; FP32 accumulation over K=5120 adds ~√K·2⁻²⁴ ≈ 4e-6 (negligible); the small-T path also rounds decoded weights to BF16 inside the MMA, RMS-accumulating to ~2e-3. Observed maxima land exactly ON the BF16 half-ulp floor — the kernels are bit-faithful to the packed layout; any packing/order bug produces O(1) errors, so tol 1e-2 (≈2.5× the floor, the plan's ballpark) separates cleanly. Rel-err denominator floored at the row-sum RMS to keep cancellation on near-zero sums from manufacturing false failures.

**VRAM envelope:** `cudaMemGetInfo`-bracketed peak **140.0 MB** (largest resident: the 136 MB small-T payload) — under the 200 MB target and the 500 MB cap. **Execution-GPU deviation, stated plainly:** GPUs 6/7 (0% util) were re-checked at run time and had only **349 MiB truly free** (nvidia-smi "free" of 843 MiB includes 496 MiB driver-reserved) — below the 600 MiB gate, and a bare CUDA context creation OOMs on both (probe: `harness/ctxprobe.cu`). The run there was aborted per protocol; parity executed on **GPU 5** (0% util at launch, 2.85 GiB free, context OK, 2348.9 MB free post-context). GPU 5 hosts an idle vLLM EngineCore with its pool already reserved; the harness ran correctness-only kernels (no timings — the host is contended and perf is explicitly out of scope for this spike).

## 5. What this proves / does not prove

**Proves:** (i) the q4 family — the decode-critical GEMV plus all three T-regime bodies and the dispatch layer — extracts on a 27-file seam with no Engine coupling; (ii) it compiles for sm_89 **verbatim** (first sm_89 compile in this code's history, per §2.2c) with real tensor-core codegen and no PDL/f8f6f4/fp4/carveout obstruction; (iii) the `row-split-k128-v1` packing is now an executable spec — an independent encoder reproduces layouts the unmodified kernels read to the BF16 rounding floor through the real route table, on both the byte-decode (SIMT/GEMV) and BF16-mma decode paths.

**Does NOT prove:** (i) **sm_89 execution** — no Ada GPU on this host; compile+link only, the binary `harness/parity_sm89` is built and unrun; (ii) **performance anywhere** — no timings taken (contended host, out of scope), so the 89.9–91.8%-of-ceiling port target and the sm_89 occupancy question (237-reg MMA schedules) are unmeasured; (iii) anything about the other families — W8/Q5/Q6/BF16 linear, attention (where `kGqaHeadDim=256` and the 96 KiB smem carveout live), FP8 (where the `f8f6f4` respell becomes real), NVFP4 (where `cuda_fp4.h` fallbacks become real), MoE, GDN; (iv) the launcher-replacement work — the exact-shape route table (`K∈{5120,2048,1152}` × registered N) was used as-is; geometry-parameterized admission is untouched.

## 6. Remaining steps to close the P0 gate (E2)

1. **4090 build + parity run** — copy `port_spike/` to an sm_89 machine, run `parity_sm89`; then the E1 read-ceiling probe and cold-L2 median-of-20 GB/s on the T=1 GEMV vs that ceiling (establish the first-ever sm_89 number; NCU one-pass weight-read verification). Watch the 237-reg schedules.
2. **5090 perf leg on an idle GPU** — same GEMV measured against E1's locally measured 5090 ceiling; must reproduce the published 1505–1537 GB/s band within ~5% under our runtime.
3. **Attention D=128** — the biggest hidden port cost (§12.2) is untouched by this spike; the q4 result says nothing about it. Start the `kGqaHeadDim` template generalization with the D=128 bidirectional kernel as the existence proof.
4. **Route-table resweep** — one `bench/ops`-harness-derived resweep on the 4090 to price the per-card sweep obligation (losing candidates were deleted in-tree; C9b confirmed). The sm_89/sm_120a register-allocation divergence measured here says the 5090 tables will NOT transfer.
5. **FreeToken leg of E2** — AOT the NVFP4 W4A16 GEMV through `compiler.py`→JitKernel (unstarted here).
6. **First real guard work** — port the FP8 row-scale family, where the `kind::f8f6f4`→`e4m3` respell behind `__CUDA_ARCH__` actually fires; this spike shows the q4 family needed none, so the guard budget shifts entirely to FP8/NVFP4/attention.

---

## Addendum (same day): full-tree sm_89 compile inventory + GPU test pass

**Full-tree sm_89 compile** (`csrc/build-serve-sm89`, `-DCMAKE_CUDA_ARCHITECTURES=89
-DNINFER_ALLOW_PORT_ARCH=ON`, ninja `-k 0`): **262/278 TUs compile verbatim; 16 fail
in exactly 4 error classes**, all MECHANICAL or compile-out-by-physics — zero
STRUCTURAL failures. This closes the guard-list question the q4 spike left open:

| Class | ptxas error | TUs | Fix |
|---|---|---|---|
| FP8 MMA spelling | `.kind::f8f6f4 not supported on sm_89` | fp8_a8, fp8_{linear_add,linear_swiglu,attn_input_proj,gdn_input_proj} | respell to plain `mma...e4m3` behind `__CUDA_ARCH__` (sm_89 HAS FP8 tensor cores) |
| PDL | `.launch_dependents requires sm_90+` | sparse_moe_{decode,small_t}, q4_q5_gdn_input_{independent,conv_snapshot} | guard `cudaTriggerProgrammaticLaunchCompletion`/PDL launch attrs OFF for sm_89 |
| TMA/cluster | `.op_restrict/.mbarrier_init/cluster/setmaxnreg require sm_90+` | nvfp4_w4a4_tma (swiglu + linear) | compile out on sm_89 — W4A4 is 50-series-bound by hardware (no FP4 tensor cores on Ada) |
| FP4 cvt | `cvt.e2m1x2.f32 not supported on sm_89` | nvfp4_w4a4, nvfp4 A16 family members | software E2M1 decode fallback (`cuda_fp4.h`-equivalent) for the A16 decode path; W4A4 users compile out |

**GPU test pass** (RTX 5090, GPU 5, idle window, sequential): **86/89 PASS (97%)**,
including all four `linear_swiglu_{q4,w8,nvfp4,fp8}` kernels that the CPU pass
could not run. The 3 non-passes are environmental: media-decode (pre-patch stale
registration; gone after reconfigure), qwen3_6_frontend (upstream hardcoded
tokenizer path), gdn_replay_fold (`cudaMalloc` OOM — wants more than the ~2.8 GiB
free on the shared card; rerun on an empty GPU). **Zero engine defects found in
the vendored tree.**
