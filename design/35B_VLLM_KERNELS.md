# Taking vLLM's kernels: what, where, and what it costs — 2026-08-30

The owner's standing permission is to take kernels from vLLM directly rather than write CUDA
from scratch. This note says where that would help, measured, and prices the one place it
would.

## Where the gaps are, by measurement

| shape | ours | vLLM | verdict |
|---|---:|---:|---|
| 4B, 1 user | **313** / 30 ms | 249 / 60 ms | nothing to take: was a routing gap, our GEMV wins once reached |
| 4B, 100 users | **5,345** | 4,481 | nothing to take |
| 0.8B, 1 / 100 users | **673** / **11,166** | 498 / 7,009 | nothing to take |
| 27B, 1 user | 70.8 / 170 ms | 71.7 / 140 ms | parity; their dense kernel is FlashInfer's cutlass block-scaled GEMM, ours the decode GEMV at 76 % of peak |
| 27B, 100 users | **1,302** / 170 ms | 1,120 / 8 s | nothing to take |
| **35B-A3B, 100 users** | 1,984 / 0.32 s | **2,162** / 3.17 s | **the one place their kernel is the difference** |

Dense linears: we lead or tie everywhere, and the single-user deficit that looked like kernel
quality was our own unregistered-shape routing (fixed today, `fc2906fc`). The 35B's routed
experts are the only gap left, and the ceiling of our own routed-expert family is now measured
and understood (`serve-engine-backlog.md` B1: 16 of 48 warps, registers and shared memory
binding together; three hand-written variants flat today). That is the gap a borrowed kernel
should close.

## Which kernel vLLM actually uses for the 35B

Confirmed from vLLM 0.27.1's own startup log on this host, serving the same
`RedHatAI/Qwen3.6-35B-A3B-NVFP4`:

    Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend out of potential backends:
      ['FLASHINFER_TRTLLM', 'FLASHINFER_CUTEDSL', 'FLASHINFER_CUTEDSL_BATCHED',
       'FLASHINFER_CUTLASS', 'VLLM_CUTLASS', 'MARLIN', ...]
    [Autotuner]: Config cache hit for trtllm::fused_moe::gemm1 / gemm2

So the 2,162 is **FlashInfer's TRT-LLM cutlass fused MoE**, JIT-built for sm_120
(`gen_cutlass_fused_moe_sm120_module`) with autotuned tile configs cached — not vLLM's own
in-tree kernel, which vLLM ranks *below* it and only falls back to.

## The three candidates, sized

**(a) vLLM's in-tree kernel** — `study/vllm/csrc/libtorch_stable/quantization/fp4/nvfp4_blockwise_moe_kernel.cu`,
747 lines: a cutlass 3.x grouped block-scaled GEMM (128x128x128 tile, 1x1x1 cluster,
`nv_float4_t<e2m1>` A and B, `ue4m3` scale factors, per-expert pointer arrays built on device).
Builds for sm_120. Our cutlass is 4.6.1 against their pin of 4.4.2 and already carries
`sm120_blockscaled_mma_array_tma.hpp`, so no dependency bump. The glue we would rewrite in
C++ from their Python pipeline: `moe_data.cu` (336 lines — expert offsets, 128-row-padded
block-scale offsets, problem sizes), `nvfp4_experts_quant.cu` (452 — per-expert activation
quant into the blocked SF layout, and the fused SiLU-mul-requant between the two GEMMs), plus
shuffle and finalize (ours exist). About 1.5k lines to port and ~500 to integrate; **1-2 days**.
Ceiling: vLLM's *VLLM_CUTLASS* backend — by vLLM's own ordering, below the 2,162. The cheapest
option, and by construction it does not reach their number.

**(b) FlashInfer's TRT-LLM cutlass fused MoE** — what produced the 2,162. Sources it compiles
for sm_120: the whole `moe_gemm` family (3,957 lines; `moe_gemm_kernels_fp4_fp4.cu` is w4a4,
`bf16_fp4.cu` is w4a16), `moe_kernels` (routing, expand, finalize; 1,133-line header), and
`cutlass_extensions` (19,468), out of a 50.5k-line `nv_internal` tree. It ships as a torch
extension, but the inner `CutlassMoeFCRunner` is raw-pointer C++, so the realistic route is to
compile that subset into our build and call the runner from C++ — no torch. **2-3 days if the
build cooperates**; the risks are cutlass-version coupling (FlashInfer vendors its own; ours is
4.6.1), TRT-LLM runtime dependencies under `common/`, and matching its activation-quant
contract. Ceiling: parity with vLLM's kernel, i.e. ~2,160 (+9 %), with our scheduler's TTFT
advantage (0.32 s against 3.17 s) kept on top.

**(c) FlashInfer's CuTe-DSL "B12x" MoE** — `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/`,
~22k lines of Python written *for SM120/SM121* (`@supported_compute_capability([120, 121])`),
including a **w4a16** variant — FP4 weights with BF16 activations, which is our routed path's
contract exactly. Its entry points are `@cute.jit` functions over `cute.Pointer` arguments and
`@cute.kernel` device kernels: a plain-pointer ABI, compiled per shape by `cute.compile` into
cubins — the form `surogate/kernels` already loads through `cuLibraryLoadData` with a JSON
manifest (`compile_cute_kernel`, `JitKernel::load_manifest()`). This is the JIT/autotuned path
the owner asked about, already written for this card. vLLM keeps it out of auto-selection only
for an SM121 build guard (`moe_backend="flashinfer_b12x"` opts in). **Measured on vLLM itself:
see the A/B below.** Cost if it wins: export the compiled kernels for our shapes and rewrite
the host orchestration (route pack, dispatch, activation, finalize — the bulk of those 22k
lines is orchestration and tuning, not kernels) in C++; **3-5 days**.

## The A/B, and why it decided less than planned

vLLM, same checkpoint, same flags, 100 users at 512/128, this host:

| MoE backend | flags | prefill | decode | TTFT p50 |
|---|---|---:|---:|---:|
| FLASHINFER_CUTLASS (auto-selected; the board's row) | vLLM defaults (8,192 batched, 128 seqs, 0.92 util) | 8,946 | **2,162** | 3.17 s |
| FLASHINFER_CUTLASS | seqs 100, batched 512, util 0.9 | 4,369 | 1,056 | 6.65 s |
| flashinfer_b12x (CuTe DSL) | util 0.85 / batched 2,048; then 0.9 / 512 | — | **OOM at init**, three attempts | — |

**B12x could not be measured on this card through vLLM.** It dies in
`allocate_sm120_static_workspace` with ~100 MiB free: FlashInfer sizes a *per-expert worst-case*
static workspace, `[256 experts x max_num_tokens x top_k x K/2]` (`max_routed_rows =
max_num_tokens * top_k`), several GB beside 21 GB of weights on a 32 GB card, and vLLM's
`max_num_tokens` does not shrink with `--max-num-batched-tokens` far enough to fit. That is an
allocation policy, not a property of the kernels — in our engine the routed job list bounds the
same buffer at tokens x top-k rows in total (tens of MB) — so (c) stays viable *inside our
engine*, but its speed on this card is unknown and cannot be learned cheaply.

The second row is a finding of its own: **vLLM's 35B decode is configuration-sensitive by
2x** — a 512-token batch cap halves it at 100 users — so the board's pair is, correctly,
defaults against defaults, and no matched-flags comparison against our 1,984 exists.

**Decision: (b).** It is the kernel behind the measured 2,162 on this card. (c) becomes the
follow-on experiment once (b) has built the packed-weight and workspace plumbing, which both
consume the same NVFP4 blocked layout; it is also the JIT/autotuned path the owner asked about,
so it earns its measurement — in our engine, not through vLLM. (a) only if (b)'s build fights
us. In every case the weight format is the routed-NVFP4 artifact, rebuilt in
77 s (`convert.py --routed-nvfp4`; the 21.9 GB file was removed in today's cleanup — 2 % fewer
bytes than groupwise-int, per-expert second-level scales as reciprocal arrays), our route and scheduler stay, and only the expert GEMMs become
theirs. Expected outcome on the 35B at 100 users: 1,984 → 2,160-2,300.

## What not to do

- Port kernels for the dense family. We lead or tie on every dense shape, and today's
  single-user gap was routing, not kernels.
- Write another routed-expert kernel by hand. Three variants measured flat today; the ceiling
  is residency, which no restructuring inside the block moves.
- Take the in-tree kernel expecting vLLM's number. vLLM does not get its number from it.
