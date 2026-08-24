# surogate serving engine — definitive architecture and implementation plan

**Codename:** `surogate-serve`. **Product category:** NInfer / FreeToken — a fast standalone serving engine for consumer NVIDIA RTX cards.
**Target cards (definitive, from the repo owner):** RTX 4070, RTX 4090, RTX 5070, RTX 5080, RTX 5090, RTX Pro 6000 Blackwell. **Arch set: sm_89 and sm_120 only.**
**Status:** decided architecture; schedule pending the Phase-0 decision-blocking experiments (§13).
**Supersedes:** the previous draft of this file (codename Kiln), which was written under a vLLM/GRPO-replacement objective and a surogate-formats-only scope. Its verified facts, roofline arithmetic, and dequant-into-GEMV analysis are retained (and re-cited below); its motivation, phasing, and §7 GRPO-driven gates are discarded. **Native GRPO rollouts are a possible downstream benefit of this engine and are NOT a driver of its scope, phasing, or gates** (see §14 for the small set of decisions that are free now and expensive to retrofit).

Every load-bearing repo claim below was re-verified against the tree at `84b48018` (verification appendix, §A). Where the adversarial critique found a proposal's claim false, the corrected fact is used; where this plan disagrees with the critique, it says so and shows the evidence.

---

## 1. Executive summary

Build a new C++ serving runtime under `csrc/src/serve/` that never routes decode through `graph_executor.cpp`/`compiled_ops` (C1), reusing everything below and beside the executor (C5): the AOT-Triton→`JitKernel` manifest pipeline that already ships decode-shaped (M=16) specializations in production (C4), the in-tree CUDA-graph capture idiom whose hard problem — stable temp addresses across replay — is already solved by `DeviceMemoryStack` checkpoint/restore (C3, corrected), the native BPE+minja tokenizer, the DSL model definitions as the declarative source of truth for 11 families, and the safetensors/HF weight loading. The **format axis** is decided as *normalize-at-ingest into three engine-owned layouts* (affine→Marlin-tiled AQT, block-float→native-planes BSF with per-arch scale swizzle at load, codebook→LUT), cached in an arch-neutral on-disk artifact, with a slow-but-correct Tier-0 dequant path so a missing fast kernel degrades instead of refusing. The **kernel axis** is decided as *port-first* (owner directive: "the idea was to port the kernels already written for FreeToken and NInfer"; both repos Apache-2.0): NInfer's hand-CUDA kernel set — measured at 89.9–91.8% of a 5090's read ceiling, and verified sm_80-class in almost every body, so it covers sm_89 as well as sm_120 — is vendored as the decode core; FreeToken's Triton kernels ride the existing `compiler.py`→`JitKernel` AOT pipeline for the formats and the MoE-offload wing NInfer does not cover; new kernel code is written **only for the verified gap list** (§2.2). The port-validation spike (§13 E2, reframed) runs **before** the schedule freezes. Execution follows the owner's C12 hybrid, **confirmed with two corrections** (§8.1): dense and resident-MoE models get the NInfer-style startup-frozen, graph-captured fast path over paged KV (P=64, INT8-G64); larger-than-VRAM MoE gets the FreeToken-style pinned-host-bank + device-LRU offload wing — one binary, one endpoint, one load UX, with the fork limited to weight residency and the expert executor. Multi-GPU ships DP replicas and layer-split; **P2P is assumed disabled on all RTX cards including the Pro 6000 (owner directive)**, so tensor parallelism is not a committed feature on any tier — a real-write P2P probe remains as a defensive runtime check that can opportunistically unlock TP2 if a platform genuinely provides P2P, but no target, gate, or phase depends on it (§8b). Serving primitives — OpenAI + Anthropic APIs, SSE, streaming detok, stop strings, tool calls, cancellation, model-switch time — are Phase-1/2 scope, not an afterthought (C10, §9).

**Product statement:** point `surogate serve` at an HF repo, a GGUF file, or a local checkpoint — GPTQ, AWQ, GGUF K-quants, FP8, NVFP4, MXFP4, NF4, or plain BF16 — and get an OpenAI-/Anthropic-compatible endpoint that decodes at ≥70–85% of your card's memory-bandwidth roofline, with every unsupported or non-viable configuration refused with the arithmetic printed rather than served slow.

**Headline targets** (decode B=1, η=0.84 of spec BW pending E1 re-measurement; ship gate ≈70–75% of roofline, re-derived from E2's ported-kernel measurements on our cards — NInfer's published band says the ceiling is ~90%, our job is to not lose it in transit):

| Card | 8B-class W4.5 (4.45 GB streamed) | 32B-class W4.5 (18.4 GB) | MoE / offload |
|---|---|---|---|
| RTX 3090 (sm_86) | **not a target** — below the `SUROGATE_MIN_CUDA_ARCH=89` build floor; W4A16 path noted as an unfunded stretch, zero spend (§4.5) | — | — |
| RTX 4090 (24 GB, 1008 GB/s) | roofline 190 → **ship ≥140 tok/s** | roofline 46 → **≥32 tok/s @ ≥16k ctx (INT8-KV)** | gpt-oss-120b MXFP4 offload ≥15 tok/s warm |
| RTX 5090 (32 GB, 1792 GB/s) | roofline 338 → **ship ≥240 tok/s** | roofline 82 → **≥57 tok/s @ ≥32k ctx (INT8-KV)** | Qwen3-30B-A3B W4.5 resident ≥340; gpt-oss-120b offload ≥25 warm |

Capacity headline: a 4090 serves Qwen3-32B at 4.5 bits with 24k tokens of INT8 KV; a 5090 with 87k; the 12 GB tiers (4070/5070) serve 8B-class at 4–4.5 bits with ≥64k INT8-KV context and get 27B+ dense **refused with the printed byte arithmetic**, never an OOM. Prefill is parity, not victory (C6).

---

## 2. The two decisions that define the engine

### 2.1 The format axis: normalize-at-ingest into three engine layouts (decided)

**Decision.** Every source quantization convention is mapped at load — transparently, on first load, cached — into **the ported kernels' native layouts**. The tier names AQT/BSF/LUT survive throughout this document, but their concrete forms are no longer invented to suit hypothetical kernels: they are the byte-exact layouts the ported kernels already read, per NInfer's `docs/maintainer/{storage-layouts.md, tensor-formats.md}` and FreeToken's bank schemas. NInfer's layouts are **not** equivalent to the previous draft's AQT up to renaming — they carry no Marlin tile interleave, no zero plane, and no g_idx — so the mapping below states exactly what each name now means, and the deltas are funded work items, not footnotes. **This composition is now the plan's central argument: port proven, measured kernels, and repack every foreign format into the layouts those kernels were tuned for.** The kernels are the asset; the layouts are their calling convention; the format subsystem's whole job is translation into that convention. *Erratum rule for the rest of this document: wherever a section says "Marlin-tiled AQT" or "Marlin-permuted", read row-split-k128-v1 — the Marlin interleave is dropped (neither repo ships a Marlin GEMM source; FreeToken imports Marlin from vLLM wheels).*

| Layout tier | Concrete form (= ported kernels' native layout) | Absorbs | Deltas / notes |
|---|---|---|---|
| **AQT** (affine quantized) | **NInfer `row-split-k128-v1`**: rank-2 `[N,K]`, K padded to 128; base-code plane (nibble-packed for int4, byte for int8), Q5/Q6 lane-major high-bit plane, then binary16 scale plane, each at 256 B boundaries; G64 (Q4/Q5/Q6) or G32 (W8); **symmetric two's-complement codes, no zero plane, no g_idx**; row-major group traversal | GGUF Q4_0/Q5_0/Q8_0 and symmetric-flattened K-quants (Q6_K/Q3_K) **losslessly today**; BF16 via group-RTN quantize-at-load; GPTQ (both zero conventions), AWQ, HQQ, GGUF Q4_1/Q5_1 and K-quant min planes **via the zero-plane extension** | Two funded extensions: (i) **optional zero/min plane + mainloop support**, +1.5–2 wk per code-width family incl. re-qualification (P2 funds q4 + w8 — the GPTQ/AWQ carriers); until it lands, asymmetric sources serve via Tier-0; (ii) **FP32-scale-plane variant** for exact K-quant `d×sc` flattening (fp16 scale words are the layout default and are silently wrong for K-quants — the §5.2 exactness rule), riding NInfer's existing scale-access-mode schedule-template axis. g_idx act-order is handled as the K-permutation baked at repack — costs the kernels nothing |
| **BSF** (block-scaled float, native planes) | **NInfer `blockscale-k16-m128x4-v1`** for NVFP4: packed E2M1 plane; E4M3 scale plane in 512-B tiles of 128 rows × 4 sixteen-wide groups (offset formula in §5.2) — the swizzle **is** the `scale_vec::4X` MMA operand layout, so it is byte-exact for the sm_120 W4A4 path and merely decoded for W4A16; one FP32 divisor; N%128==0, K%64==0. **NInfer `row-scale-v1`** for FP8: row-major E4M3 bytes + one BF16 multiplier per row. **FreeToken's native planes** for per-tensor / 128×128-block FP8 (`weight_scale_inv [⌈N/128⌉,⌈K/128⌉]` bf16) and MXFP4 (transposed `blocks_t [K/2,N]` uint8 + `scales_t [K/32,N]` e8m0 — the gpt-oss offload bank format) | ModelOpt + compressed-tensors NVFP4 (divisor/multiplier unified at repack), compressed-tensors FP8 all three granularities, MXFP4, surogate's own fp8/nvfp4 training outputs (row-scale ≈ per-channel FP8) | NInfer has **zero MXFP4 support** (no E8M0 anywhere in the repo) — the previous draft's "one template serves NVFP4 and MXFP4" holds only for the FreeToken Triton family; a dense MXFP4 W4A16 codec variant of the NInfer GEMV (E8M0 = exponent shift, ~2–3 d) is on the gap list. FP8 per-tensor/block granularities have no NInfer kernel — they stay on the ported FreeToken Triton family |
| **LUT** (codebook epilogue) | AQT skeleton + a 16-entry smem codebook in the GEMV epilogue (unchanged — new code, the smallest family) | BnB NF4 (incl. double-quant), GGUF IQ4_NL | |

Two non-weight layout commitments ride the ports and bind the repacker equally hard: (1) **the KV pool layout is NInfer's** — P=64 page groups spanning all layers, per-plane page-major order, INT8-G64 code+scale planes sharing the page id (§8) — because the ported attention family defines it; (2) **fused-op parent tensors**: the repacker must emit gate|up as one concatenated `[2N,K]` rowsplit parent and q|k / v|z as concatenated parents with fixed row order — NInfer's fusion catalog (linear_swiglu / attn_input_proj / gdn_input_proj) reads parent tensors, not separate weights. Offload expert banks follow FreeToken's `_BANK_SCHEMAS` registry: expert-major fixed-byte rows with the GPU slot cache byte-identical per row (the in-place repack invariant its gather kernels require).

**Refused with a printed reason** (not silently degraded): GGUF i-quants (E8-lattice codebooks), EXL2 (mixed per-row bitrates), EXL3/AQLM (trellis). These are not losslessly repackable and their kernel families are the 6,493-line vendored-ggml class of cost for a small audience that llama.cpp already serves.

**The kernel-body counts that decide it.** Native-format kernels remain the measured disaster: FreeToken vendors **7,550 lines** of llama.cpp CUDA for GGUF *alone* (verified `wc -l` this session; dp4a-only — 45 `__dp4a` uses in `vecdotq.cuh`, zero tensor cores — the exact mechanical reason vLLM's GGUF path is slow, with i-quants having no batched MMQ path at all). Normalize-at-ingest kills that entire class: **only `dequantize.cuh` (~600 LOC, all types incl. i-quants) is ported, as the ingest reader, never as a serving kernel.** What changes under port-first is the other half of the old argument: NInfer's "~92 hand-qualified launchers" were previously cited as the cost of hand-CUDA — the porting inventory shows the launchers/wrappers (5.3k lines of exact-model-shape whitelists) are the disposable closed-product layer, while the **bodies behind them are self-contained templates behind a POD launch contract** (Tensor/Weight structs + `WorkspaceArena` + stream, zero cuBLAS/CUTLASS/cuDNN in the whole repo) and are the asset being ported. The qualification-cell arithmetic stands — ~40–60 perf-gated CI cells across layouts × families × arches once variant axes are admitted — but the bodies inside the cells are now **ported and already tuned**, not written; what we author is dispatch (geometry-parameterized admission replacing whitelists), per-card route-table sweeps (data), and the gap list in §2.2.

**The UX arithmetic.** Repack runs on-GPU, streamed per-layer: seconds-to-two-minutes for a 30B checkpoint (E9 measures it), peak host RAM ≈ one layer, peak VRAM ≈ one tensor. The result is cached as a **4096-aligned, 8-GiB-sharded, JSON-indexed artifact** (O_DIRECT-readable, pin-after-fill-loadable — the FTW recipe) whose **fingerprint is arch-neutral**: source-file hashes + layout version only, never compute capability. Per-arch scale swizzles are re-applied at load (1–2 s for a 20 GB model), which fixes FreeToken's cc-in-fingerprint portability trap — one artifact serves a 4090 and a 5090. `--no-cache` converts into RAM-resident banks without touching disk for users with 40 GB of GGUFs and no spare SSD. Cost vs llama.cpp's run-the-file-you-downloaded: one-time conversion latency + one disk copy (~source size, +~3% for widened scales). Payoff: tensor-core prefill everywhere the source format would have forced dp4a, one GEMV family instead of twenty, and the property below.

**The SGLang evidence, weighed as evidence.** SGLang's weight-cache daemon (lmsys, 2026-08-21) exports weights as raw CUDA IPC handles and had to **blocklist per-tensor FP8, Marlin, AWQ and GPTQ** — precisely because those formats carry source-specific, host-side (Python) metadata and per-tensor transforms that raw IPC cannot express. Our internal layouts are deliberately **self-describing**: codes, scales, zeros/mins, and permutations live in device-resident planes; the only host-side state is the artifact JSON index, which is re-derivable and layout-stable. Consequence: any tensor in any supported source format can be exported/attached as a raw device buffer with no format-specific host logic — the property SGLang lacks. This is a direct argument for normalize-at-ingest over native-format kernels (which would inherit SGLang's blocklist problem), and it is also what makes a future weight daemon (§9.6) and a future trainer→engine zero-copy weight sync (§14) cheap.

**Where prefill and decode differ on this axis:** they don't — both consume the same three layouts. What differs is who reads them: decode reads packed planes in a dequant-in-register GEMV; prefill dequantizes tiles into tensor-core fragments inside the K-loop (never materialized to DRAM) or, for AQT at large M, dequantizes N-chunks once into a scratch and calls cuBLASLt (parity path, C6).

### 2.2 The kernel axis: port-first, three sources in priority order (decided)

**Decision (owner directive).** The decode/serving kernel set is **ported, not authored**, from two Apache-2.0 codebases sitting in `study/ninfer` and `study/FreeToken`, with new code written only for verified gaps — in this priority order:

**(a) NInfer's hand-tuned kernel bodies** for every format/op it covers — the decode core. These are the only ~90%-of-roofline datapoints in evidence (1505–1537 GB/s on a 5090 = 89.9–91.8% of its measured 1674.5 GB/s read ceiling, `docs/maintainer/linear-benchmark.md`, NCU-verified zero weight replay), and the decisive inventory finding is that **despite the sm_120a-only build flag, almost every body is plain sm_80-class CUDA** (cp.async, ldmatrix, `mma.sync` bf16/f16/s8/tf32) whose entire tensor-core surface funnels through a handful of inline-asm wrappers in `src/ops/common/mma.cuh` — so porting NInfer covers **sm_89 and sm_120 both**, not just the 50-series. Per-family arch coverage: Q4/Q5/Q6/W8 rowsplit linear, BF16 linear, the fused-epilogue catalog (SwiGLU/add/QKV/GDN projections), paged GQA attention (bf16 + INT8-KV), GDN recurrent+chunked+conv1d, sparse MoE, sampler/argmax, speculative/MTP, elementwise — **all sm_89 + sm_120** (sm_80-class bodies; PDL launches in 8 files degrade to plain launches behind an `sm_90+` guard). FP8 row-scale — **both arches**, with the A8 tensor-core path needing a **one-line respell** of `kind::f8f6f4` to the plain sm_89 `e4m3` mma form behind `__CUDA_ARCH__` (sm_89 FP8 mma at fp32 accumulate runs half-rate vs fp16 accumulate — re-bench, don't assume). NVFP4 W4A16 GEMV/small-T — **both arches** (E2M1 decode via `cuda_fp4.h` casts, software/LUT fallback below sm_100). NVFP4 **W4A4 block-scaled MMA + TMA variant — sm_120 only, by physics** (Ada has no FP4 tensor cores; `kind::mxf4nvf4` + `cvt.e2m1` exist in no sm_89 spelling) — this ported body **replaces the previously tracked CUTLASS Sm120 escape hatch**: smaller, already tuned, and the repo has zero cuBLAS/CUTLASS/cuDNN dependencies. NInfer bodies enter surogate as **native `.cu` translation units under `csrc/src/serve/` compiled by nvcc — a second kernel supply chain beside AOT-Triton/JitKernel**; the manifest-constants idea maps onto their `plan.cpp` route tables, whose tuned constants must be re-swept per card (the sweeps' losing candidates were deleted — C9b's unreproducible-constants concern is confirmed in-tree, and the `bench/ops` harness ports with them, ~2–3 d). The extraction seam is the POD launch contract: free `__global__` templates over raw device pointers + `Tensor`/`Weight` structs + caller-owned `WorkspaceArena` + `cudaStream_t`, with `workspace_capacity_bytes(min,max)` dry-run queries; the wrappers' exact-shape whitelists are **replaced** by geometry-parameterized admission, never ported.

**(b) FreeToken's Triton kernels, AOT-compiled through the existing pipeline** — `surogate/kernels/compiler.py` emits cubin + JSON manifest; `csrc/src/runtime/jit/jit_kernel.{h,cpp}` loads via `cuModuleLoadData` / `cuLibraryLoadData` (required sm_120+); grids from manifest constants, never hardcoded (the GDR autotune-grid bug is institutional memory). The inventory verifies AOT feasibility kernel-by-kernel: the decode-path kernels carry **no `@triton.autotune`** — fixed constexpr tiles, static grids, counts in device `int64[1]` tensors, explicitly CUDA-graph-safe — exactly the signature shape `compiler.py` wants. Ported this way: the NVFP4 W4A16 dense family (GEMV / split-K with fused in-kernel reduce / small-M GEMM), NVFP4 fused MoE, MXFP4 split-K MoE + gpt-oss swiglu/routing, the FP8 family (128×128-block, per-tensor, mxfp8, blockscale MoE), bf16 fused MoE + `moe_align`, split-KV paged attention + extend prefill, the fused sampler, the FLA/GDN decode step, and the **offload LRU/policy kernels** (single-CTA device-side expert cache that rewrites `topk_ids`→slot ids in place). Two caveats owned up front: the FLA chunk-prefill and sampler kernels **do** carry `@triton.autotune` — their config spaces are swept and pinned offline per arch before manifests exist; and all tiles are H100-tuned — 4090/5090 re-sweeps are mandatory. Additionally, FreeToken's **plain-CUDA offload/gather/copy kernels** (`fast_index_copy` + fused multi-bank variant, KV `store.cu`, embedding `index.cu`, `cudaMemcpyBatchAsync` wrapper with CUDA-12 fallback, ~1.1k LOC) port near-verbatim as nvcc `.cu` units for the MoE wing — the highest verbatim-survival code in that repo (its tvm-ffi host glue is rewritten, 3–5 d total).

**(c) NEW code, only for the verified gaps** — the explicit list, from both inventories: (1) **the zero/min-plane extension** of row-split-k128 + the GEMV/small-T mainloops per code-width family (+1.5–2 wk each; q4+w8 funded in P2) — nothing in either repo supports asymmetric affine, so GPTQ/AWQ/HQQ/Q4_1/Q5_1/K-quant-min absorption is written work; (2) **GPTQ/AWQ/GGUF→internal-layout streamed on-GPU repack encoders** — NInfer's canonical encoders are CPU-side converter oracles in `tools/`, FreeToken only calls vLLM's `gptq_marlin_repack`; the D2D-capable repack kernels (§14.2) are written fresh, with GGUF `dequantize.cuh` ported as the ingest reader; (3) **D=128 GQA generalization** — NInfer's main attention path hardcodes `kGqaHeadDim = 256` as a file-scope constant (verified), and most launch targets are D=128; threading the head dim through every derived constexpr is the biggest hidden port cost (the D=128 bidirectional/DFlash kernel is the existence proof), folded into the attention port at 2.5–4 wk for the family; (4) **an MXFP4 (E8M0) codec variant** of the NInfer W4A16 GEMV (~2–3 d) — NInfer is NVFP4-only; (5) **a full-vocab sampler stage** (radix-select) — the ported sampler clamps top-k at 20 and ships as v1, the clamp-free stage lands behind the same interface; (6) **an INT8-G64-KV variant of the FreeToken Triton attention** (+5–8 d) *only if* it is still the serving kernel when INT8 KV lands — the NInfer attention port obviates it; (7) **the multi-CTA reshape of the single-CTA LRU ensure** (bounded follow-up) and the `flashlib` question resolved by unifying on the vendored hybrid kernel with uncapped fetch; (8) geometry parameterization passes throughout (MoE 256-experts/top-8/2048-hidden constants, W8 per-(N,K) route tables, embed-gather vocab constants). **Notably NOT a gap: sm_89 quantized GEMV coverage** — the inventory expected NInfer's bodies to be sm_120-bound and verified the opposite; the sm_89 work is mechanical guards (PDL no-op, one mma respell, fp4-decode fallback, the 96 KiB prefill smem needs sm_89's ~99 KiB opt-in carveout call), plus per-card sweeps — but sm_89 has **never been compiled** in that repo, so a 4090 build+parity run is P0's first deliverable, before any 40-series claim.

**Honest counts.** Ported: ~36.3k lines of NInfer device code across ~13 kernel families plus ~12k lines of FreeToken Triton/CUDA across ~10 families — call it **~23 of ~26 kernel families ported, 3 written** (zero-plane mainloops, repack encoders, full-vocab sampler stage) plus adaptation labor (D=128 generalization, geometry parameterization, per-card sweeps). Kernel effort re-cut in §10: **~11 nominal engineer-weeks for the NInfer side + ~9–11 for the FreeToken side** — comparable in wall-clock to the old authoring plan, but the ~90%-roofline decode numbers are now **a port target with published evidence, not a bet**. **Vendor-locked bodies, tracked per C9b:** the cuBLASLt call sites behind `MatmulContext` and the vendored FA2 varlen prefill kernels (unchanged); the CUTLASS Sm120 W4A4 item is **retired** in favor of the ported NInfer body. The AMD-portability story narrows honestly: the FreeToken Triton wing stays largely ROCm-portable; the NInfer decode core is CUDA and is priced as such.

**The measurement gate, reframed (E2, §13).** The old E2 asked "can Triton reach the roofline?" — an authoring bet. The new E2 is a **port-validation spike**: stand up the seam (PODs, `WorkspaceArena`, `mma.cuh` with the sm_89 guards), port the q4 rowsplit GEMV verbatim, and measure it under **surogate's runtime** on a 4090 and a 5090 against E1's locally measured read ceilings — it must reproduce NInfer's published band (~90% on the 5090; establish the sm_89 number, which has never existed) — and AOT one FreeToken Triton family (NVFP4 W4A16 GEMV, the repo's own E2-candidate at a self-measured ~64% of roofline) through `compiler.py` end-to-end to prove the manifest path on ported sources. E2 no longer decides Triton-vs-hand; it **validates the ports hit their published perf under our runtime and sizes the gap list** (route-table re-sweep cost, sm_89 deltas, which Triton families need a hand twin from the NInfer side).

**Why the training op graph cannot be the inference path** (C1, C2 — verified, not assumed): 119 coarse `REGISTER` entries in `op_registrations.cpp` and **no fusion pass anywhere** (every "fused" op is a hand-written monolith; the `fuse()` helper concatenates weights at load); graph compilation is per-(B,T) and reruns the buffer planner + SaveForBwd promotion + five-arena resize on every shape change — continuous batching would recompile continuously; the IR has exactly two runtime shape symbols (B, T) — page tables can only ride as opaque bindings; `DslGradStore` is constructed unconditionally (`dsl_model.cpp:1310`), so a forward-only flow through today's `DslModel` pays full training VRAM; and every MoE/EP op is declared capture-unsafe (`graph_executor.cpp:67-92`) because `moe_permute.cpp` stages `expert_offsets` D2H and **spin-polls `cudaStreamQuery`** — one host sync per MoE layer, invisible in a 7 s training micro-batch, fatal in a 2 ms decode step. Adding a fusion pass to that graph would mean writing new CUDA anyway (C2) — so the specialization happens in Triton templates instead, where the (N,K,codec,arch) axes are `constexpr`.

**What C4 already proves.** `~/.cache/surogate/kernels/` contains `matmul_tn_bf16_m16_n4096_k1536_bm16_bn128_bk128.json` — a decode-shaped M=16 GEMM with autotuned tiles, a 3-pointer signature, and `extra_null_params: 2`, sitting beside m1536/m2048/m4096 variants with *different* autotuned tiles, all compiled by Python and launched by C++ with no Python in the loop. The mechanism this plan scales up is already shipping.

**The maintenance-tax arithmetic, re-run for ports.** What made NInfer's method untenable as an authoring model — ~92 launchers / 7 formats / 1 arch / 3 models, sweep constants with deleted losing candidates — splits cleanly under the porting lens: the *launcher/wrapper* layer is the closed-product tax and is replaced (geometry-parameterized admission, plan-shaped dispatch with `workspace_capacity_bytes` queries — the parts §3.1 already ADOPTs); the *bodies* are schedule-templated C++ (`RowsPerCta`, warps, pipeline stages, scale access mode as template parameters) where new model geometry is an instantiation plus a route-table sweep, not a kernel. The per-card sweep obligation is real and budgeted (the harness ports); it is data regeneration, not body maintenance. The FreeToken Triton wing keeps the old arithmetic: templates + manifests, new geometry = autotune run. What porting does **not** reduce, same as before: ~6k lines of serve-runtime C++ (scheduler, paged KV, ServeExecutor, sampler host side, HTTP) — and it *adds* one honest item, the C++ rewrite of FreeToken's offload host machinery (~1.7k lines of Python scheduler code that ports as an executable spec, 10–15 d, §8.9).

**Where prefill and decode differ on this axis:** decode is where the engine wins and where the ported NInfer GEMV/attention families live; prefill bottoms out in the ported rowsplit/A8 MMA GEMMs, cuBLASLt, and FA2 like everyone else's, targets parity ±15%, and ships **zero authored** GEMM kernels — the dequant-in-K-loop tensor-core paths (small-T/rowsplit MMA) and the Phase-5 Sm120 W4A4 body all arrive by port.

### 2.2b Porting inventory (condensed; full verdicts in the study inventories)

**NInfer** (`study/ninfer`, Apache-2.0, ~36.3k device LOC + ~14.8k host):

| Family | Verdict | Arch coverage | Port cost |
|---|---|---|---|
| Q4/Q5/Q6 rowsplit linear (GEMV / small-T MMA / rowsplit GEMM / SIMT), ~5k LOC | **PORT** | sm_89 + sm_120 (PDL guard) | 3–5 d bodies + new dispatch; zero-plane ext. +1.5–2 wk/family |
| W8 grouped-int linear (incl. vocab/MTP routes), 2.6k LOC | **PORT** | sm_89 + sm_120 | 2–4 d + per-card route re-sweep |
| BF16 linear GEMV/small-T (skip its large-T GEMM — cuBLASLt) | **PORT-ADAPT** | sm_89 + sm_120 | 1–2 d |
| FP8 row-scale linear, A16 + A8 tensor-core, ~1.6k LOC | **PORT-ADAPT** | both; A8 needs 1-line mma respell on sm_89 | ~1 wk |
| NVFP4 W4A16 GEMV/small-T | **PORT** | both (decode fallback < sm_100) | ~1 wk; +2–3 d for an MXFP4/E8M0 variant (written) |
| NVFP4 W4A4 block-scaled MMA + TMA | **PORT** | **sm_120 only (physics)** | 1–2 wk; retires the CUTLASS Sm120 hatch |
| Fused-epilogue catalog (swiglu/add/pair/QKV/GDN projections), 13.7k LOC | **PORT-ADAPT** | as parent families | +30–50% on each parent; ~2–3 wk total |
| Paged GQA attention (split-KV decode bf16+i8, chunked prefill, fused KV append/quant) + INT8-G64 KV | **PORT-ADAPT** | sm_89 + sm_120 | 2.5–4 wk (largest item; D=256→template D, per-card split sweeps; 68.6 KB parity oracle ports too) |
| SWA/bidirectional GQA (DFlash) | **SKIP** (port with speculation) | both | 1–1.5 wk then |
| GDN linear attention (recurrent T=1..8, chunked prefill, ReplaySSM, conv1d), 4.6k LOC | **PORT** | sm_89 + sm_120 | 1.5–2 wk; no Triton GDN decode exists in-tree |
| Sparse MoE (route + device-job-list decode + prefill), 2.8k LOC | **PORT-ADAPT** | both (PDL guard) | 1.5–2 wk; geometry-parameterize 256/8/2048; offload splice point kept explicit |
| Fused sampler + argmax (~900 LOC) | **PORT-ADAPT** | both | 3–5 d; top-k≤20 ships v1, full-vocab stage written later |
| Speculative verify + MTP round kernels | **SKIP** (port with speculation phase) | both | ~1 wk then |
| Elementwise + quantized embed_gather | **PORT** | both | 2–4 d; embed_gather saves a full BF16 embedding table |
| Vision kernels | **SKIP** (multimodal out of v1 scope) | both | — |
| Seam: Tensor/Weight PODs, `WorkspaceLayoutBuilder`, `pdl.cuh`, `mma.cuh`; plan-shape dispatch | **PORT-ADAPT** (wrappers **never**) | host + guards | ~1 wk |

**FreeToken** (`study/FreeToken`, Apache-2.0):

| Family | Verdict | Arch coverage | Port cost |
|---|---|---|---|
| Offload copy/gather CUDA (`fast_index_copy`(+multi-bank), `store.cu`, `index.cu`, batch-memcpy), 1.1k LOC | **PORT** | sm_89 + sm_120 (CUDA≥13 for batch-memcpy; 12.x fallback) | 3–5 d |
| MoE offload LRU/policy Triton (device expert cache, `topk_ids`→slot rewrite) | **PORT** (AOT via compiler.py) | both | 2–3 d + 2–3 d for the flashlib-external default (unify on hybrid kernel) |
| Offload host machinery (banks, double-buffer, O_DIRECT, FTW) — Python | **REWRITE-CHEAPER** (executable spec) | host | 10–15 d + 3–5 d loaders |
| NVFP4 W4A16 dense (GEMV / split-K fused-reduce / small-M) | **PORT-ADAPT** | sm_89+ (fp8e4nv scales) | 4–6 d; the E2 Triton baseline (~64% roofline self-measured) |
| NVFP4 fused MoE decode | **PORT-ADAPT** | sm_89+ | 2–3 d |
| MXFP4 MoE (gpt-oss) split-K GEMV + prefill + swiglu/routing | **PORT-ADAPT** | both (even < sm_89) | 3–4 d |
| FP8 family (block / per-tensor / mxfp8 linear + blockscale MoE) | **PORT-ADAPT** | sm_89+ | 4–6 d |
| bf16 fused MoE + `moe_align` + decode grouped GEMM | **PORT-ADAPT** | both | 3–5 d |
| Triton paged attention (split-KV decode + extend prefill) | **PORT-ADAPT** | both (smem-aware tiles handle sm_89) | 5–8 d as-is (bf16 KV, page_size=1→P=64 adaptation); +5–8 d INT8-KV variant only if needed |
| FA/FlashInfer/trtllm attention wrappers | **SKIP** (external wheels; no source) | — | confirms C9b: vendor FA2 ourselves |
| FLA/GDN Triton (fused decode step, chunked prefill, conv1d), 3.6k LOC | **PORT-ADAPT** | both | 3–4 d decode + 5–7 d prefill (pin autotune configs) |
| Fused sampler (split-vocab, full top-k/top-p) | **PORT-ADAPT** | both | 3–4 d (pin autotune; plan math to C++) |
| Elementwise (rope gather-in-kernel, *_and_mul PDL, strided qk-norm) | **PORT** (gaps only) | both | 1–2 d each |
| GGUF vendored ggml CUDA (7,550 LOC) | **PORT-ADAPT `dequantize.cuh` only** (ingest reader) | both (dp4a) | 2–3 d; MMQ/MMVQ never serve |
| CPU-MoE executor (AVX-512) | **SKIP** (rejected wing); port doorbell mechanism only | x86 | 1–2 d for doorbells |
| DF11, DSV4/GLM/M3 sparse, pynccl | **SKIP** | — | 0 |

---

## 3. What we learned — and what we now port

*(This section was written when both repos were treated as evidence to learn from. Under the owner's port-first directive they are also source trees to vendor from — both Apache-2.0, verified at their repo roots. Verdicts below are upgraded accordingly: **PORT** means the code itself crosses over; ADOPT still means we re-implement the design.)*

### 3.1 NInfer (C++/CUDA, sm_120a-built but sm_80-class in body, 2 model families, ~144k LOC)

| Mechanism | Verdict | Reason |
|---|---|---|
| **The kernel bodies themselves** — ~36.3k device LOC: rowsplit linear all formats, fused-epilogue catalog, paged GQA + fused INT8-G64 KV codec, GDN stack, sparse MoE, sampler/argmax, elementwise | **PORT** (the headline change; §2.2/§2.2b) | Apache-2.0; 89.9–91.8% of measured read BW on a 5090; bodies are sm_80-class behind the `common/mma.cuh` wrappers, so they cover sm_89 too; self-contained `__global__` templates behind a POD launch contract, zero cuBLAS/CUTLASS/cuDNN |
| Device-side MoE decode job list (`adaptive_route_jobs` read in-kernel — no host loop, PDL-chained, graph-safe) | **PORT** | Verified in `sparse_moe_decode_kernels.cu`; it *is* the §8.1 graph-compatibility recipe. (The previous draft's "per-token MoE host loop" REJECT entry was wrong for the decode path and is retracted) |
| Portable CPU-oracle test suites (`tests/ops`, e.g. the 68.6 KB `test_gqa_attention.cpp`) | **PORT with their kernels** | Parity oracles travel with the bodies; they are the qualification harness for every port |
| Boundary loop: one owner thread, membership/state mutate only between GPU units, forced decode/prefill alternation | **ADOPT** | ~100 lines; eliminates the race class around KV/state ownership; makes graph replay safe by construction |
| Startup memory by dry-run simulation of the real allocation recipes (`WorkspaceLayoutBuilder`) | **PORT** (~150 lines of actual code, not just the pattern) | Plan and execution share one code path; drift impossible; replaces hand-maintained byte formulas |
| Affine KV capacity curve (two full candidates at M_min/M_min+1, closed-form solve vs free VRAM − headroom, affinity asserted) | **ADOPT** | Auditable `--kv auto`; constraint: MoE workspace must be sized on a fixed token-slice cap or the curve breaks |
| Page group P=64 spanning all layers + K/V(+scale) planes; block table `[pages, slots]` int32 at engine-stable address | **ADOPT** | The mechanism that makes paging CUDA-graph-safe: graph binds pool bases, page ids are data |
| INT8-G64 KV codec: FP16-rounded-scale-before-encode, scale plane shares the page id, quant fused into append, dequant into staging, **no standalone codec kernels** | **PORT** (rides the attention-family port) | Halves KV bytes; the fusion *is* the win |
| Pinned fixed-capacity SoA ingress/egress inside the graph; one host sync per round | **ADOPT** | No per-row pointers, no mid-round syncs |
| Execution envelope `[min,max]` visible keys as graph key; device-recomputed splits; CTA early-exit | **ADOPT** | Ragged context under a static grid |
| Reservation admission with completion guarantee | **ADAPT** | Right shape; needs the max_tokens cap fix (§8.5) or concurrency collapses on 12–16 GB — NInfer's own documented degeneration |
| Retained-continuation prefix cache, exact identity predicate (token ids + types + position axes + media descriptors) | **ADOPT the predicate + per-slot cache in core phases** | Cheap, correct; the radix tree stays deferred (§8.7) |
| `top_k≤20` fused sampler | **PORT as v1, extend later** (was REJECT) | The ported sampler ships with the clamp; the full-vocab radix-select stage is written behind the same interface (§2.2 gap 5) — a product-decision reversal the port makes cheap |
| Exact-B graphs; W≤16 attention when B>1; SHA-pinned chat templates; sealed `.ninfer` container with per-model converter inventories; **wrapper exact-shape whitelists (5.3k lines)** | **REJECT** | B=1-first / closed-product artifacts; padded buckets, minja templates, HF-repo-first ingest, and geometry-parameterized admission instead — wrappers are the one layer that is *never* ported |
| ReplaySSM (record raw k/v/g/β, fold accepted prefix, record+fold from ONE templated body) | **PORT with the GDN family, use deferred with speculation** (host-side, comes along nearly free) | 74–86× cheaper than state snapshots; bit-identity is a property of shared code |
| O_DIRECT + 4×64 MiB pinned-slot streaming weight upload; bind-socket-before-load; `ToolCallStreamFilter` with longest-suffix-prefix hold-back + emitted-bytes handshake | **ADOPT** | Straight into §9 |
| Hand-tuned per-(N,K,T) route tables as dispatch | **ADOPT shape, REGENERATE contents per card** | Finite validated routes + `workspace_capacity_bytes(min,max)` query yes; the tuned constants are 5090-specific (170-SM grid arithmetic) and their sweeps' losing candidates were deleted — the `bench/ops` harness ports (~2–3 d) and re-sweeps run on every target card |

### 3.2 FreeToken (Python/PyTorch/Triton, MoE-offload-centric)

| Mechanism | Verdict | Reason |
|---|---|---|
| **In-repo Triton kernel files** (~6.5k LOC quant linear/MoE + 950 attention + 2.9k FLA/GDN + 619 sampler + elementwise) | **PORT via `compiler.py` AOT → JitKernel** (§2.2b) | Decode paths carry no `@triton.autotune` — fixed constexpr tiles, static grids, device-count tensors, graph-safe by construction: exactly the AOT signature shape. H100 tiles re-swept per card; the two autotuned families (FLA chunk-prefill, sampler) get their config spaces pinned offline |
| **Offload copy/gather CUDA bodies** (`fast_index_copy`(+multi-bank), `store.cu`, `index.cu`, batch-memcpy wrapper), 1.1k LOC | **PORT (near-verbatim `.cu`)** | The highest verbatim-survival code in the repo; only the tvm-ffi host glue is rewritten (3–5 d); bypasses the Triton pipeline entirely |
| Graph-compatibility recipe: every host decision → fixed-shape device tensor; every variable-size memcpy → fixed-grid kernel reading its count from device `int64[1]` | **ADOPT — the most reusable idea in either repo** | The general fix for `moe_permute.cpp` and any data-dependent op under capture |
| Device-side flat-id LRU expert cache; in-kernel `topk_ids`→slot rewrite; fused multi-bank UVA gather (`fast_index_copy_multi`, ~30 lines) | **PORT (offload wing)** — the LRU/policy kernels AOT cleanly (single-CTA, `grid=(1,)`, per-model constexprs) | Complete graph-safe expert cache. Two knowns: the default non-hybrid `lru_ensure` lives in the **external `flashlib==0.3.0`** package (verified import), so we unify on the vendored hybrid kernel with uncapped fetch; the single-CTA ensure is reshaped multi-CTA as a bounded follow-up, not port-blocking |
| **Offload host machinery** (bank registry, double-buffer choreography, hit/miss-split prefetch, O_DIRECT + pin-after-fill loading, FTW reader) — ~1.7k lines of Python | **REWRITE-CHEAPER in C++, Python kept as executable spec** | Zero code survives (torch streams/events/numpy throughout) but every mechanism is documented with its measured rationale; 10–15 d + 3–5 d loaders — the single largest port item, and it is scheduler code, not kernels |
| GGUF vendored ggml CUDA: `dequantize.cuh` (all types incl. i-quants) | **PORT-ADAPT as the ingest/Tier-0 reader only** (2–3 d) | Perf-uncritical repack input; MMQ/MMVQ/moe-vec are never serving kernels — porting them would import the dp4a prefill ceiling §2.1 exists to escape |
| Prefill whole-layer double-buffer borrowing slots [0,2E); begin-fence/ready/release events | **ADOPT** | Minimal correct PCIe-hiding structure |
| `cudaMemcpyBatchAsync` sub-256 KiB synchronous-degradation guard; pin-after-fill (never `cudaHostRegister` a lazy mmap); `expandable_segments` check | **ADOPT** | Hard-won driver lore, free |
| `cuStreamWriteValue64/WaitValue64` doorbells for CPU work in a graph | **ADOPT mechanism, defer use case** | CPU-MoE hybrid itself is rejected (below) |
| Pure-function memory planner (`cache_budget.py`), GPU-free-testable | **ADOPT** | Directly answers the GPU-less CI requirement |
| Backend capability matrix as data; closed `is_smXX_family()` vs open predicates | **ADOPT** | `cc>=(10,0)` meaning "Blackwell" is a latent crash on sm_120 |
| FTW container (4096-aligned, 8 GiB shards, JSON index) | **ADOPT, minus the cc-in-fingerprint trap** | §2.1: arch-neutral fingerprint, swizzle at load |
| Elastic VRAM rebuild (two-phase validate/rollback, same budget fn for limits and fit-check) | **ADOPT the spec, ship post-v1** | Right design, wrong release for v1 scope |
| Probed effort/thinking dialect (render the checkpoint's own template with probe kwargs) | **ADOPT (Phase 3)** | Zero per-family code for reasoning-effort knobs |
| Daemon/supervisor process (torch-free, oom_score_adj, receipt outbox) | **ADAPT (§9.6)** | Compare with SGLang's weight-cache daemon; v1 ships the artifact cache, daemon deferred with its design constraint taken now |
| CPU-MoE hybrid (q\* split) | **REJECT for this card list** | FreeToken's own 2× rule fails on PCIe 5.0 + dual-channel DDR5 (75/52 = 1.44×); it is a 40-series+DDR5 niche; movement primitives adopted anyway |
| MoE-first budget priority; page_size=1 tables; Python scheduler plumbing; `auto→offload` default | **REJECT** | KV-first with explicit expert budget; P=64; C++ loop; resident-first admission |

### 3.3 License and attribution (new — porting obligations)

Both source repos are **Apache-2.0** (verified: `LICENSE` at both repo roots this session). Obligations and conventions, decided now so no port lands without them:

- **Vendoring convention:** ported CUDA/C++ bodies land under `csrc/src/serve/third_party/{ninfer,freetoken}/`, preserving upstream directory structure where practical; ported Triton sources under `surogate/kernels/serve/third_party/{ninfer,freetoken}/`. Every ported file keeps its upstream license header and gains a provenance header: upstream path + commit hash + a one-line summary of local modification (Apache-2.0 §4(b) requires modification notices). Heavily adapted files (geometry parameterization, D=128 generalization) stay in the third_party tree until the divergence is structural, then move with their history noted.
- **NOTICE file:** a `NOTICE` at the surogate repo root (Apache-2.0 §4(d)) carrying both projects' attributions; it ships in source and binary distributions. CI greps that every file under the third_party roots retains its header.
- **Manifest provenance:** AOT-compiled cubins from ported Triton sources record the upstream source id in their JSON manifests, so a shipped kernel binary is traceable to its origin.
- **No copyleft exposure:** Apache-2.0 imposes no reciprocal licensing; the vendored ggml-derived GGUF files (`dequantize.cuh` lineage: sgl-kernel ← vLLM ← llama.cpp, all MIT/Apache-compatible) get their chain-of-custody noted in the same headers.

---

## 4. The hardware envelope

### 4.1 The cards

Decode roofline = η × BW_spec, η = 0.84 (NInfer's 5090 calibration: 1505–1537 GB/s realized vs 1792 spec, 89.9–91.8% of a measured 1674.5 GB/s pure-read ceiling, NCU-verified zero weight replay). **η is borrowed and re-measured per card in E1 before targets are committed.**

| Card | Arch | VRAM | BW spec (GB/s) | η·BW | PCIe | Status |
|---|---|---|---|---|---|---|
| RTX 4070 | sm_89 | 12 GB | 504 | 423 | 4.0 ×16 | **First-class tier.** Aggressive weight quant + INT8 KV mandatory |
| RTX 5070 | sm_120 | 12 GB | 672 | 565 | 5.0 ×16 | First-class tier, same rules |
| RTX 5080 | sm_120 | 16 GB | 960 | 806 | 5.0 ×16 | 8–14B tier; 32B dense refused |
| RTX 4090 | sm_89 | 24 GB | 1008 | 847 | 4.0 ×16 | Primary sm_89 target |
| RTX 5090 | sm_120 | 32 GB | 1792 | 1505 | 5.0 ×16 | Flagship |
| RTX Pro 6000 Blackwell | sm_120 | 96 GB | ~1792 | ~1505 | 5.0 ×16, **P2P assumed off** (owner directive) | Top tier; 96 GB single-card capacity; pairs scale via layer-split/DP (§8b) |
| RTX 3090 / 3090 Ti | sm_86 | 24 GB | 936/1008 | — | 4.0 ×16, NVLink | **Out of scope** (§4.5) |

Consumer realities the design must respect: GDDR not HBM; no NVLink anywhere on the list; **P2P assumed disabled on every RTX card on the list, Pro 6000 included** (owner directive; measured `can_device_access_peer == False` on this host's 5090 pairs) — all GPU↔GPU traffic is planned as host-staged; consumer boards run x8/x8 splits and chipset x4 slots (4 of 8 GPUs on this very machine run x8); desktop CPUs/RAM (dual-channel DDR5 ≈ 75 GB/s).

### 4.2 Decode roofline arithmetic

`tok/s(B=1) = η·BW / (W_streamed·b_w + C·kv_bytes_per_token)`; b_w = 2 (BF16), 1 (FP8), 0.5625 (NVFP4/W4.5), 0.53125 (Q4G64). Embedding is a gather, not streamed.

| Model | Streamed | BF16 | FP8 | W4.5 (+W8 head) |
|---|---|---|---|---|
| Llama-3.1-8B | 6.98e9 body + 0.53e9 head | 15.0 GB | 7.5 GB | **4.45 GB** |
| Qwen3-32B | 31.2e9 + 0.78e9 | 64.0 GB | 32.0 GB | **18.4 GB** |
| Qwen3-30B-A3B (active) | 3.3e9 | 6.6 GB | 3.3 GB | **1.86 GB** |
| gpt-oss-20b (active, MXFP4) | ~3.6e9 | — | — | **~2.0 GB** |

Rooflines (tok/s, B=1, weights-only term): 8B W4.5 — 4070 95 / 5070 127 / 5080 181 / 4090 190 / 5090 338. 32B W4.5 — 4090 46 / 5090 82 (no fit ≤16 GB). 30B-A3B W4.5 at **η_moe = 0.59** (MoE GEMVs realize ~70% of dense efficiency — NInfer's 35B-A3B measured 271 tok/s implies ~1.4× byte inflation on scattered top-8 expert reads) — 4090 320 / 5090 568.

### 4.3 KV capacity

KV bytes/token (K+V, all layers): Llama-8B 128 KiB BF16 / 64 KiB INT8; Qwen3-32B 256 / 128; Qwen3-30B-A3B 96 / 48; 27B-class GDN-hybrid **64 / 32** (only 16 of 64 layers are full attention — architecture beats KV quantization, 4× cheaper per token than a same-size dense model).

Budget model: `total − CUDA ctx/modules (~0.5 GB) − weights − graph pool (ASSUMED 0.8 GB → E3 measures) − activations/prefill scratch (0.6–1.0 GB) − ~5% fragmentation = KV + state pool + (offload) expert slots`. Resulting context: 32B W4.5 on 4090 → 12k BF16-KV / **24k INT8-KV**; on 5090 → 43k / **87k**; 8B W4.5 on 4070 → ≥64k INT8-KV; 27B-hybrid on 5090 → ~200k+ INT8-KV. The startup planner prints this exact budget before serving.

### 4.4 Non-negotiables that fall out

1. **Sub-5-bit weights are mandatory for every ≥27B dense target on every listed card** — the largest lever, and it is a format decision, not a kernel decision. This is why quantize-at-load for BF16 repos (§5.4) is a product requirement, not a convenience.
2. **INT8-G64 KV is the default at long context** and effectively mandatory for 32B on 24 GB and everything on 12 GB. Honest caveat: no reference repo publishes an end-to-end INT8-KV quality A/B (only op-level tolerances, rel-L2 3.15e-3 vs 2.8e-3 BF16); §11.2 measures and publishes ours.
3. **Hybrid (GDN) models carry a context-independent state floor** (~144 MiB/slot at 27B-class; 2.3 GiB at 16 slots) that caps concurrency on 12–16 GB cards; the planner prices it explicitly.
4. **Dead ends, refused at load with the arithmetic printed:** 27B+ dense on 12/16 GB (weights alone exceed the budget); GLM-4.7-class MoE offload anywhere (189 GB banks; 9,461 MB cold/token → 2.8–5.5 tok/s at any consumer link; ≤12.7% residency on 32 GB); MoE offload over x8 links (~13 GB/s → single digits); BF16-expert offload of any model (quantize first — Qwen3-30B-A3B NVFP4 banks fit *resident* on 24 GB, which is the best kind of offload); TP on GeForce (§8b).
5. **sm_89 FP8 honesty:** Ada FP8 at the safe FP32 accumulator is 2× BF16, not 4× (4× requires FP16 accumulate over K=5120 reductions — not our default); sm_89 has no block-scaled MMA, so fine-grained-scale FP8 dequantizes in the K-loop there.
6. **sm_120 ≠ sm_100.** Consumer Blackwell has warp-scoped `mma.sync` `kind::mxf4nvf4`/`kind::f8f6f4` + CTA-scope TMA; it does **not** have tcgen05/TMEM. trtllm-gen cubins, FA3 (sm_90a wgmma), FA4, CUTLASS Sm100 collectives all refuse to load. Every "we'll use the Blackwell FP4 GEMM" line item in this plan means CUTLASS **Sm120** collectives or hand warp-MMA, budgeted as original work.
7. **Consumer opt-in shared memory is ~99 KiB** (vs 164/227 KiB on A100/H100); attention tiles at head_dim ≥256 must be sized for it (NInfer proves ≥96 KiB usable on sm_120a); probe `cudaDevAttrMaxSharedMemoryPerBlockOptin` at startup rather than assuming.

### 4.5 sm_86 / RTX 3090: out of scope, deliberately

`csrc/CMakeLists.txt` sets `SUROGATE_MIN_CUDA_ARCH 89` and FATAL_ERRORs below. Lifting it costs: no FP8 tensor cores, no block-scaled MMA, and Triton rejects the `fp8e4nv` type anywhere in an sm<89 kernel (even a pointer argument), forcing a uint8-view software codec through the whole AOT path. Per the owner's card list, **zero design or kernel effort is spent**. Noted stretch only: the AQT W4A16 GEMV and prefill templates contain no fp8 types and would likely compile for sm_86 unmodified if the build floor were ever lowered; no number is promised, no CI column exists.

---

## 5. The format subsystem

This is the largest single work item (C7) and `surogate/quant/` is verified greenfield (`git ls-files` → 0 files). It lives in `surogate/serve/formats/` (Python readers + converter driver) and `csrc/src/serve/artifact/` (artifact reader, O_DIRECT loader, repack kernel launchers).

### 5.1 Readers (source-format ingestion)

| Source | Detection | The gotchas the reader must own | → Layout |
|---|---|---|---|
| GGUF (single-file v3) | `*.gguf` glob (`registry.py:157`) | Header/KV/tensor-dir parse; dims reversed vs torch; padding only before tensor data; **tokenizer + chat template + config live in KV** and are extracted to sibling JSONs at conversion | AQT (legacy + K-quants), LUT (IQ4_NL), refuse (i-quants) |
| GPTQ | `quant_method=="gptq"` (`hf_config.py:96-106`) | qweight int32 `[K·bits/32, N]` K-packed; **zero−1 vs gptq_v2 true-zero branch on `checkpoint_format`**; `g_idx` act-order → row sort at repack + K-permutation baked into the tile order (activation permute fused into the GEMV prologue); 3-bit refused | AQT |
| AWQ | `quant_method=="awq"` | qweight `[K, N/8]` **N-packed** (the shape discriminator vs GPTQ), nibble interleave `[0,2,4,6,1,3,5,7]` inverted at repack; true zeros | AQT |
| HQQ | config | plain affine group-wise; zero sometimes itself quantized | AQT |
| compressed-tensors FP8 | `quant_method=="compressed-tensors"`/`fp8` | three scale granularities — per-tensor fp32 scalar, per-channel `[N]`, 128×128 block `weight_scale_inv` — **dispatched on scale SHAPE+DTYPE, never name** | BSF |
| NVFP4, both conventions | config groups `{num_bits:4, type:float, group_size:16}` vs ModelOpt names | compressed-tensors: `weight_packed`/`weight_scale`(E4M3 [N,K/16])/`weight_global_scale` **as DIVISOR**; ModelOpt: `weight`/`weight_scale`/`weight_scale_2` **as MULTIPLIER**; mixed checkpoints overload `weight_scale` as BF16 `[N,1]` per-row FP8 — again dispatch on shape+dtype; group_size 32 + float ⇒ MXFP4, must not route here | BSF |
| MXFP4 (GPT-OSS) | `_blocks`/`_scales` suffixes | reader already exists end-to-end (`mxfp4_quantizer.cpp`, `dsl_model.cpp:993`) | BSF |
| BnB NF4 | `quant_method=="bitsandbytes"` | reader exists incl. double-quant absmax reconstruction (`bnb_quantizer.h`, `dsl_qlora_pipeline.cpp:95-190`) | LUT |
| **BF16/FP16 (stock HF repo)** | default | **group-RTN quantize-at-load into AQT** (group 64/128, per-group scale (+min for asymmetric), W8 head default) whenever the planner says BF16 does not fit the card; user-selectable 4/5/8-bit; cached like any conversion | AQT |
| surogate fp8/nvfp4 training outputs | recipe metadata | same numerics as training (`csrc/src/recipes/`) | BSF |

### 5.2 Repack rules and exactness

- Affine sources repack losslessly by construction (bit-exact code round-trip). **K-quant two-level scales (`fp16 d × int6 sc`) flatten to FP32 per-sub-block scales — exact in fp32, silently wrong in fp16**; the FP32-scale variant flag exists for exactly this.
- Every format ships a **CPU reference dequant oracle** (FreeToken's `dequant.py` is the K-quant reference; NInfer's tensor-formats §12 discipline is the model): repack → dequant → compare vs source-dequant, bit-exact for affine, ≤1e-6 rel for scale-widened. CI-gated per format.
- BSF keeps codes native; the only transforms are scale-plane layout (canonical row-major in the artifact; per-arch swizzle at load: NVIDIA 128×4 interleave — closed formula `(row_tile·K_tiles + scale_tile)·512 + (row_inner%32)·16 + (row_inner/32)·4 + scale_lane` — for the sm_120 W4A4 route; Marlin-permuted for W4A16) and global-scale normalization (divisor/multiplier unified into one canonical multiplier + descriptor).
- AQT planes are 256 B-aligned and row-addressable **at tile granularity** (16-row tiles) — sufficient for MoE expert row spans, which are tile-aligned by construction; the finer claim of arbitrary row slicing (true of NInfer's non-interleaved row-split layout, not of a Marlin interleave) is *not* made.

### 5.2b Supported input formats (owner directive, 2026-08-24)

**The engine's supported inputs are safetensors (HF repos, incl. GPTQ/AWQ/FP8/NVFP4
variants) and GGUF. The `.ninfer` container is NOT a supported input format** — no
user is ever asked to download or produce one. The vendored NInfer artifact reader
survives only as an internal seam while the native safetensors/GGUF loaders are
built against it, and is retired from the product surface after that. The on-disk
cache below (§5.3) is an internal, transparent, regenerable acceleration of the
load-time repack — never an interchange format, never published, never required
(`--no-cache` serves without it).

### 5.3 The artifact cache

`~/.cache/surogate/artifacts/<fingerprint>/` — one logical byte region, every tensor start 4096-aligned and padded, shards cut at 8 GiB on aligned boundaries, JSON index `{tensors[], shards[], meta}`, plus extracted tokenizer/config for GGUF sources. Read path: O_DIRECT + `preadv` loop (short-read-safe) through 4×64 MiB pinned slots, **pin-after-fill**; fallback probe at startup (tmpfs/network-FS/WSL2 without O_DIRECT → buffered mmap + `MADV_SEQUENTIAL`). **Fingerprint = source file hashes + layout version. Never compute capability** — swizzles are load-time. Conversion runs transparently on first `serve`, or explicitly via `surogate convert`; `--no-cache` builds banks in RAM only.

### 5.4 End-to-end UX

```
surogate serve TheBloke/Qwen3-32B-GPTQ          # detect → convert-on-first-load (progress bar) → serve
surogate serve ~/models/qwen3-8b-q4_k_m.gguf     # tokenizer/config from GGUF KV → AQT → serve
surogate serve Qwen/Qwen3-8B --bits 4            # stock BF16 repo, group-RTN at load, cached
surogate convert <src> [--bits N] [--out DIR]    # explicit, scriptable
surogate serve <src> --no-cache                  # RAM-only conversion, no disk artifact
```
Every load prints: detected format → chosen layout(s) → tier (fast/generic, §8.2) → predicted decode tok/s from the roofline → KV/context budget. Refusals (i-quants, EXL2/3, AQLM, dead-end configs from §4.4) name the reason and the nearest supported alternative.

---

## 6. Model coverage

**Verified inventory (correcting the critique):** `surogate/dsl/models/` contains **11** families — gemma4, gpt_oss, laguna, lfm2, llama, nemotron_h, qwen3, qwen3_5, qwen3_5_moe, qwen3_moe, qwen3_vl (the critique's "12" is wrong; `ls` shows 11 model files + `__init__.py`). Onboarding HF-parity harnesses exist for **9 of 11** (`tests/test_onboarding_*.py`; `gemma4_unified` is a gemma4 variant). **`llama` and `qwen3_5_moe` have no harness — writing those two oracles is costed work in Phase 3, not assumed.**

**How an arbitrary HF checkpoint gets served.** `serve <path>` → `hf_config` detects `architectures[0]` + `quant_method` (or GGUF arch KV) → matched to a DSL family → format subsystem picks layouts and tier → **decode-trace mode** on the DSL emits the T=1 serving graph (`paged_attention` op with cache operands replacing `flash_attention`; `lm_head_logits` replacing `fused_lm_head_loss`; GDN state-slot op for hybrids) → ServeExecutor consumes the ServePlan → serve. Verified: only `surogate/dsl/modules/{attention,embedding,gated_delta_rule}.py` emit the three training-coupled ops, so the decode-trace change is localized in the module layer; all 11 model files, HF weight mappings, and config parsing are untouched. **The DSL family is the unit of support** — an unmatched architecture gets a refusal naming the closest family and the porting doc, never a silent generic build. An HF-config-driven generic builder is rejected: the DSL defs already encode the per-family quirks (gpt-oss sinks, gemma4 per-layer SWA + k_eq_v, laguna per-layer head dims, GDN state) that a generic builder would silently get wrong.

**Per-new-model cost, stated:** new checkpoint of a known family = 0 code. New size/variant = config + weight-map deltas, ~0.5–1 day. New dense family = DSL def + decode trace + parity harness, **~1–2 weeks** (the "3–5 days" claims in the proposals are rejected; gemma4/laguna onboarding history says otherwise). New hybrid or MoE family = +1–2 weeks. New op class = kernel work first, 2–4+ weeks.

**Launch model list** (chosen by consumer download reality — GGUF-first, then AWQ/GPTQ, then official FP8/NVFP4): Qwen3 dense 8/14/32B (GGUF Q4_K/Q5_K/Q6_K, AWQ, GPTQ, FP8, BF16-RTN); Llama-3.1-8B (same); gpt-oss-20b MXFP4 resident; gpt-oss-120b MXFP4 offload (24–32 GB); Qwen3-30B-A3B / Qwen3.5-35B-A3B quantized (resident where they fit); official NVFP4 releases on sm_120; gemma4, lfm2, nemotron_h, qwen3_5, laguna as decode-trace breadth in Phase 3; qwen3_vl text-only path (vision deferred, clean 400).

---

## 7. What surogate already has

### 7.1 REUSE AS-IS

| Asset | Path | Note |
|---|---|---|
| Native BPE tokenizer + minja Jinja templates + special tokens + per-turn masking | `csrc/src/tokenizer/` | Gap: whole-sequence decode only — streaming detok is new (§9.2). `ChatMessage` is `{role, content}` only — must be extended for tools (§9.3) |
| AOT JIT loader + manifests | `csrc/src/runtime/jit/jit_kernel.{h,cpp}` | `cuModuleLoadData`/`cuLibraryLoadData` (sm_120+), constants-driven grids, `extra_null_params` |
| Worked 11-kernel AOT precedent | `surogate/kernels/gated_delta_rule.py` + `jit/gated_delta_rule_kernels.h` | The pipeline this plan scales |
| Triton AOT compiler + content-addressed cache | `surogate/kernels/compiler.py`, `cache.py` | |
| CUDA-graph capture idiom | `graph_executor_utils.h:118-152` + `utilities/stack.h` | Capture→instantiate→checkpoint/restore→replay; verified in-tree |
| Capture debugging | `py_train.cpp:1512-1650`, `SUROGATE_DEBUG_DUMP_CAPTURED_GRAPH` | Pinned-stable-host-buffer trick + memcpy-node walker |
| Elementwise/norm/rope/swiglu kernel bodies | `csrc/src/kernels/` | Zero new elementwise kernels |
| cuBLASLt path + heuristic cache (M-keyed) + op-role tags | `csrc/src/kernels/matmul.cpp`, `matmul_context.h` | Prefill; no M==1 special case exists today (E4 checks) |
| Weight loading + HF name mapping | `dsl_weight_loader.cpp`, `surogate/dsl/hf.py` | Engine and trainer load the same names |
| DSL model definitions (11 families) | `surogate/dsl/models/` | The declarative truth (C5) |
| Quant codec math | `qlora/quantized_tensor.h`, `kernels/{block_quant,mxfp4_dequant}.cu`, `recipes/` | Ports into Triton codec device functions |
| LoRA fold-in | `qlora/adapter_merger.{h,cpp}` | v1 serving = merged adapters (§9.5) |
| NCCL bootstrap/threading/abort hardening | `utilities/comm.{h,cpp}` | §8b; NOT the per-step transaction model |
| Pinned bank memory layer + ring arena + LPT planner | `runtime/ep/{weight_transfer.cpp, ring_arena.h, lpt_planner.h}` | Offload wing memory layer; NOT the EP control plane |

### 7.2 REUSE WITH CHANGES

| Asset | Path | Change |
|---|---|---|
| Forward-only executor path | `execution_request.h` (`disable_forward_saves`), `graph_executor.cpp:1547-1603` | **Prefill only** (C1 forbids decode). Production-exercised via `compute_inference_logprobs` |
| Inference-only model construction | `dsl_model.cpp:1310` | Gate `DslGradStore` + SaveForBwd planning — **mandatory**; forward-only today still allocates training VRAM |
| Last-token logits kernels | `gather_rows_bf16` (`kernels.h:1717-1726`), `lm_head_logits_matmul` (`fused_lm_head_loss.cpp:115-180`) | **Lift the kernels, not the CompiledExecutor members** — ServeExecutor calls them directly with a host-supplied index list, replacing the device target-scan + `cudaStreamSynchronize` at `fused_lm_head_loss.cpp:803` |
| LM-head precision | `dsl/modules/embedding.py:118-129` (`quantizable=False`) | Serving overrides to W8 — at V≈128–152k the head is ~21% of an 8B BF16 decode step |
| KV append scatter | `flash_attn_scatter.cu:245` (`append_kv_to_cache`) | Block-table variant (~30-line kernel change) |
| Attention backend registry | `attention_backend.h:207-250` | Add paged backend at priority 120 gating on `page_table != nullptr`; relax/fork `kvprefix`'s **throwing** `supports()` (throws on sinks, B≠1 — serving must return false, or handle) |
| GDN op | `gated_delta_rule.cpp` (initial_state `inputs[5]`, final_state `outputs[1]`; BT=64 at `:179,:463`) | State-carry interface exists; **needs a T=1 recurrence kernel** |
| Per-layer state pools | `compiled_ops.h:831-849` (`ChunkGdnState/ChunkConvState`) | Generalize to lane-affine per-sequence slots |
| DSL trace surface | `dsl/modules/{attention,embedding,gated_delta_rule}.py` | Add decode-trace mode (§6) |
| MXFP4 + NF4 readers | `qlora/{mxfp4_quantizer.cpp, bnb_quantizer.h}`, `dsl_qlora_pipeline.cpp` | Feed the format subsystem instead of dequant-to-BF16-for-QLoRA |

### 7.3 MUST BUILD (verified absent)

Paged KV store/block table/page allocator/refcounting (zero grep hits; `backend_kvprefix.cpp` is a contiguous B=1 training chunk cache, not a starting point) · paged attention kernels (FA2 vendored **without** splitkv/paged `.cu` files; `num_splits=0` hardcoded) · **ServeExecutor** (§8.3 — the decode runner all three proposals left unnamed) · sampler (only argmax in-tree is an accuracy counter, `fused_classifier.cu:1079`) · scheduler/admission/continuous batching/cancellation · device-side MoE routing · GDN T=1 kernel + slot pool · streaming detok, stop-string hold-back, tool/reasoning parsers, constrained decoding · HTTP layer (no networking dependency in `csrc/CMakeLists.txt`) · format subsystem (§5) · GPU-free unit-test target (only `integration-tests` exists, 5 sources). **Port-first note:** "absent in surogate" no longer means "written from scratch" — the paged attention kernels, sampler, device-side MoE routing, and GDN T=1 kernel on this list are sourced by port (§7.4); what remains authored is the runtime around them.

### 7.4 PORT (new reuse source: the study repos, Apache-2.0 — verdicts from the porting inventories, condensed in §2.2b)

| Asset | Source path | Verdict |
|---|---|---|
| Q4/Q5/Q6/W8 rowsplit linear (GEMV/small-T/GEMM) | `study/ninfer/src/ops/linear/{q4,q5,q6,w8}/` | **PORT** — the measured ~90%-of-roofline decode core |
| Fused-epilogue linear catalog | `study/ninfer/src/ops/{linear_swiglu,linear_add,linear_pair,attn_input_proj,gdn_input_proj,gdn_gating_proj}/` | **PORT-ADAPT** (rides parent families) |
| FP8 row-scale linear (A16 + A8 MMA) | `study/ninfer/src/ops/linear/fp8/` | **PORT-ADAPT** (sm_89 mma respell) |
| NVFP4 W4A16 + **W4A4(+TMA, sm_120-only)** | `study/ninfer/src/ops/linear/nvfp4/` | **PORT** — W4A4 retires the CUTLASS Sm120 hatch |
| Paged GQA attention + fused INT8-G64 KV codec | `study/ninfer/src/ops/kernel/gqa_attention_*.cuh`, `paged_kv_address.cuh`, launchers | **PORT-ADAPT** (D=128 generalization; largest item) |
| GDN recurrent/chunked/ReplaySSM/conv1d | `study/ninfer/src/ops/linear_attention/gated_delta_net/`, `src/ops/kernel/causal_conv1d.cuh` | **PORT** — no Triton GDN decode exists in surogate |
| Sparse MoE (device job-list decode) | `study/ninfer/src/ops/sparse_moe/` | **PORT-ADAPT** (geometry-parameterize) |
| Fused sampler + argmax; speculative/MTP (deferred) | `study/ninfer/src/ops/kernel/{sampling,argmax,speculative_round,mtp_*}.cuh` | **PORT-ADAPT** / SKIP-until-speculation |
| Elementwise + quantized embed_gather | `study/ninfer/src/ops/kernel/` | **PORT** |
| Seam: PODs, arena, `WorkspaceLayoutBuilder`, `pdl.cuh`, `mma.cuh` | `study/ninfer/src/core/`, `src/ops/common/` | **PORT-ADAPT** (wrappers under `src/ops/wrapper/` are **never ported**) |
| Parity oracles + bench harness | `study/ninfer/tests/ops/` (e.g. `test_gqa_attention.cpp`), `bench/` | **PORT** with their kernels |
| Offload copy/gather CUDA | `study/FreeToken/python/freetoken/kernel/csrc/jit/{fast_index_copy.cuh,store.cu,index.cu,batch_memcpy.cuh}` | **PORT** (near-verbatim) |
| Device LRU expert-cache policy kernels | `study/FreeToken/python/freetoken/moe/offload_kernels.py` | **PORT** (AOT via `compiler.py`; `flashlib` default replaced by hybrid kernel) |
| NVFP4/MXFP4/FP8/bf16 Triton linear + MoE families | `study/FreeToken/python/freetoken/kernel/triton/{nvfp4_linear,nvfp4_fused_moe,mxfp4_moe,fp8_*,mxfp8_linear,fused_moe,moe_align,decode_moe}.py` | **PORT-ADAPT** (AOT; per-card tile re-sweep) |
| Triton paged attention (split-KV + extend) | `study/FreeToken/python/freetoken/kernel/triton/attention.py` | **PORT-ADAPT** (P1 bridge on bf16 KV; P=64 adaptation gated separately) |
| FLA/GDN Triton + conv1d; fused sampler; elementwise gaps | `study/FreeToken/python/freetoken/kernel/fla/`, `triton/{sampling,causal_conv1d_triton,norm,rope,activation}.py` | **PORT-ADAPT** (pin autotune configs offline) |
| GGUF dequant (ingest only) | `study/FreeToken/python/freetoken/kernel/csrc/gguf/dequantize.cuh` | **PORT-ADAPT** — never a serving kernel |
| Offload host machinery (banks/double-buffer/FTW) | `study/FreeToken/python/freetoken/moe/{offload_cache,host_banks,expert_banks}.py` | **REWRITE-CHEAPER** in C++; Python kept as executable spec |
| CPU-MoE executor, DF11, DSV4/GLM/M3 sparse, pynccl, FA/FI wrappers | various | **SKIP** (doorbell mechanism only, ~150 LOC) |

---

## 8. Architecture

### 8.1 The C12 verdict: two-path hybrid CONFIRMED, with two corrections

The owner's null hypothesis — NInfer-style fixed-memory execution for dense, FreeToken-style offload co-execution for larger-than-VRAM MoE, as one engine — **survives the evidence**, with these corrections:

1. **Device-side MoE routing is built once and serves both paths.** The training MoE path is host-synchronous by verified design (`moe_permute.cpp` spin-poll; every MoE/EP op declared capture-unsafe). Replacing it with device-resident routing (offsets computed in a device scan, grouped launches reading problem sizes from device memory, FreeToken's in-kernel `topk_ids`→slot rewrite) is what *simultaneously* makes resident-MoE decode graph-capturable **and** enables the offload wing. It is one build item, not two, and it is the hardest kernel work in the plan (Phase 4, prototyped in Phase 2 spare cycles).
2. **Offloaded-MoE decode IS graph-capturable** (FreeToken proves it in production: fixed-shape device LRU bookkeeping, fixed-grid UVA gather reading its count from device `int64[1]`). So the fork between the paths is **not** capture vs no-capture. What forks: weight residency (VRAM arena vs pinned host banks + device slot cache), the expert executor (in-place indexed GEMV vs LRU-ensure→gather→GEMV), and the memory-planner policy (dense: affine KV solve; offload: KV-first split with an explicit expert-slot budget — inverting FreeToken's MoE-first default). Prefill streaming on the offload path runs event-choreographed *outside* graphs (FreeToken's double-buffer), which is fine — prefill isn't captured on either path.

**Shared vs forked:**

| Component | Dense + resident-MoE path | Offload-MoE path |
|---|---|---|
| Scheduler/boundary loop, admission, cancellation | **shared** | shared |
| Paged KV + INT8-G64 + block tables | **shared** | shared |
| Sampler (+ grammar mask hook) | **shared** | shared |
| Serving layer, tokenizer, detok, parsers | **shared** | shared |
| Format subsystem + artifact cache | **shared** (expert banks are just tensors in AQT/BSF bank schemas) | shared |
| CUDA-graph inventory + ServeExecutor | **shared** | shared (decode captured on both) |
| Weight residency | VRAM weight arena | pinned host banks + device LRU slot cache |
| Expert executor | in-place indexed grouped GEMV | LRU-ensure kernel → fused UVA gather → same GEMV over slot cache |
| Memory planner policy | affine KV solve | KV-first budget with expert-slot term |
| Prefill expert weights | resident | whole-layer double-buffer streaming + hit-D2D split |

**Load-time admission rule** (mechanical, printed at load):
`resident_bytes = non_expert_weights + quantized_expert_banks + KV_floor(8k tokens) + state_floor + graph_allowance + fixed_overhead`. If `resident_bytes ≤ VRAM budget` → **dense/resident path**. Else if `expert_banks ≤ pinned-host budget` AND `link_tok_s(cold) ≥ floor` (E6-measured gather BW) → **offload path**, with the predicted warm/cold tok/s printed. Else → refusal with the arithmetic. "MoE that fits resident" is therefore on the dense path by construction — e.g. Qwen3-30B-A3B NVFP4 (~16 GB banks) on a 24 GB card never touches the offload machinery.

### 8.2 The tier ladder (failure behavior, from LADDER)

- **Tier-1 (fast path):** AQT/BSF/LUT layouts, the ported kernel families (NInfer hand-CUDA decode core + FreeToken AOT-Triton, §2.2), graph-captured decode. The launch list (§6) ships Tier-1.
- **Tier-0 (generic path):** for any ingestible format/family combination without a qualified fast kernel: dequant-to-BF16 (resident once if it fits) + cuBLASLt, eager launches, same parity oracles. **Hardened per the critique:** Tier-0 is admission-gated — the load banner predicts tok/s from the roofline, and if the prediction is below a floor (default 5 tok/s, configurable) the engine **refuses with a guided-convert suggestion** instead of serving a 10×-slower-than-llama.cpp experience. Tier-0 that cannot dequant-resident does not layer-stream at PCIe speed; it refuses. Tier-0 is a correctness floor and a promotion staging area, and the banner says which tier is live.
- **Promotion pipeline:** per-(family × format × arch) telemetry counters + a documented rule — parity oracle green, demand shown, autotune sweep run, roofline gate passed in CI. Most promotions are autotune + qualification (~2–5 days); a new packing quirk is the full 1–2 weeks.

### 8.3 ServeExecutor — the decode runner (the component the proposals left unnamed)

The critique's top must-fix, honored here as a first-class component. **`csrc/src/serve/exec/` — ~4–6k lines, Phase 1.**

- **Input:** a `ServePlan` per (model, bucket) emitted by the DSL decode trace — a linear op list with concrete shapes, a tensor slot table (weights by name → arena offsets; activations → workspace offsets; KV/state/page-table bindings), and a workspace recipe.
- **Own op table:** `ServeOpTable` in `csrc/src/serve/exec/serve_ops.cpp` — flat `ServeOp{kind, kernel_ref (JitKernel | builtin fn), arg binding, grid expression over manifest constants}`. **Hard rule, CI-greppable: nothing under `csrc/src/serve/` includes `op_registrations.cpp`, `compiled_ops*`, or uses `REGISTER_COMPILED_OP`.** That is how C1 is enforced structurally, not by convention. (Forge's wording violated this; corrected.)
- **Workspace planning:** the dry-run byte-counting builder (NInfer's `WorkspaceLayoutBuilder` pattern) shared with the memory planner — one code path for plan and execution.
- **Execution modes:** eager (Phase 1) and captured (Phase 2) via `trace_or_execute_cuda_graph_with_stack` + `DeviceMemoryStack` — a new caller of the existing idiom, per C3.
- Builtin ops (non-JIT): cuBLASLt matmuls via `MatmulContext`, `gather_rows_bf16` + lm-head matmul (kernels lifted, CompiledExecutor members not called), KV append, D2H/H2D SoA staging.

### 8.4 Prefill path, end-to-end (the second unnamed piece, specified)

- **Phases 1–2:** fresh-prompt prefill runs through the **existing GraphExecutor in forward-only mode** (`disable_forward_saves=true`, inference-only construction with `DslGradStore` gated) at B=1 with T-buckets {512, 1024, 2048, 4096} and `cu_seqlens` packing so multiple admitted prompts batch into one chunk. This is permitted — C1 forbids *decode* through the executor — and it buys parity-tested correctness on day one. Compile cost and VRAM per bucket are measured in E7; buckets are precompiled at startup, never at request time.
- **KV export:** the serving attention backend (priority 120, activates on `page_table != nullptr`) appends the chunk's post-RoPE K/V into the paged pool via the block-table variant of `append_kv_to_cache`, with INT8-G64 quantization fused into the append.
- **Prefill-with-paged-prefix** (chunked prefill over existing context, needed for chunk 2+ and prefix reuse): the Triton paged-attention family at BLOCK_M 128–1024, cross-shape `cu_q ≠ cu_k`, bottom-right causal. The existing `kvprefix` kernel cannot do this (contiguous cache only, throwing `supports()`).
- **Family quirks:** gpt-oss attention **sinks** are a constexpr epilogue term in the Triton paged kernels (kvprefix throws on them today); gemma4 **per-layer SWA** is a per-layer window clamp on the visible-key envelope (window already rides in `AttentionParams`); laguna per-layer head dims are per-layer constants in the ServePlan.
- **Last-token logits:** host-supplied row-index list → `gather_rows_bf16` → lm-head matmul (FP8-cache-aware). The `cudaStreamSynchronize` at `fused_lm_head_loss.cpp:803` is not on this path.
- **Phase 3:** prefill migrates onto ServeExecutor (same ServePlan machinery at chunk shapes), retiring the GraphExecutor dependency and its per-bucket compile cost; the executor path remains as a parity oracle.

### 8.5 Scheduler, continuous batching, admission

One owner thread (NInfer's boundary loop): decode-round / prefill-chunk alternation, membership and slot/state binding mutate only at boundaries, chunked prefill at 1024 tokens (multiple of 128), batched prefill (multiple prompts per chunk), cancellation as a boundary event (disconnect → row dropped next round, pages/entitlement released). Exactly one host sync per decode round (event on the previous round's egress D2H).

**Admission** is a four-axis page-rounded entitlement vector `{lanes, kv_pages, state_slots, expert_slots}` (NInfer's hardcoded 3-axis struct is a named pitfall) with a completion guarantee — **fixed for the client-default-max_tokens collapse** (critique must-fix): the *guaranteed* reservation is `prompt + min(max_tokens, reserve_cap)` pages (reserve_cap default 2048 tokens, planner-derived per tier); growth beyond the cap is best-effort incremental allocation. On pool exhaustion the engine never OOMs and never truncates mid-token: the youngest best-effort request finishes cleanly with `finish_reason: "length"` + a `capacity_exhausted` detail. Per-tier concurrency tables are published (12 GB tier: C=4 typical). No preemption/swap in v1.

### 8.6 CUDA-graph strategy

Padded batch buckets **{1, 2, 4, 8}** (default; {16, 32} opt-in for DP-less throughput serving) × context-frontier profiles placed **where kernel route selection changes** (the attention split-policy tiers), one `cudaGraphExec` per topology class with `cudaGraphExecUpdate` between profiles. All capture at startup or model load, never during serving; FreeToken's protocol verbatim (one eager forward per bucket immediately before capture; teardown drops execs AND static buffers). Everything variable is a value in one pinned SoA ingress or an int32 index into a captured-base device table. Capture uses one private page per lane repeated across the block-table row. Graph memory AND **graph capture time** are budgeted and measured (E3) — SGLang's residual-startup data (7.7 s capture for their bucket set) makes capture time a first-class startup cost; our default bucket set (~4 × ~4 profiles ≈ 16 captures) targets ≤10 s total, with lazy capture of rare buckets behind a flag.

### 8.7 Prefix caching (in the core phases, per the critique)

- **Phase 2:** per-slot **retained-continuation cache** — a finished request's lane keeps its pages + (hybrid) state, and a new request whose prompt exactly extends that continuation resumes at the frontier. Identity predicate adopted from NInfer verbatim: token ids + token types + all position axes + media descriptors, never splitting a media item. This makes multi-turn chat — the dominant consumer workload — re-prefill only the new turn, for the whole v1 life. Cost: ~400 lines on top of the lane machinery.
- **Refcount field in the page allocator from day one** (cheap now, enables fork later — §14).
- **Deferred:** the radix prefix tree with copy-on-write pages (post-v1). Hybrid models structurally cap reuse at full-state checkpoints (GDN state is not reconstructible from KV) — stated in docs, with FreeToken's semantic-anchor (tool-call-opener checkpoint) as the post-v1 enhancement.

### 8.8 Sampler

Fused device sampler inside the captured graph: temperature → penalties (repetition/presence/frequency) → top-k/top-p/min-p (full vocabulary, **no top-k clamp**) → categorical draw via **stateless counter RNG** keyed (seed, position, purpose) — bit-reproducible under graph replay. Two routes by V (cooperative single block; partial-topk + group-finalize for V≈128–152k). Per-token **logprob of the sampled token at the sampling temperature** emitted from day one (cheap; contract-level — §14). **Constrained-decoding hook designed in now** (critique must-fix): the sampler consumes an optional per-row vocab bitmask `[B, ceil(V/32)]` at a fixed device address inside the graph; the host grammar engine (vendored llguidance-class token-mask computation) fills it per step through the pinned ingress path — mask computation outside the graph, application inside, graph-safe. `response_format: json_schema`, `tool_choice: required/named` ship on this hook in Phase 3.

### 8.9 MoE and offload — the bandwidth math that bounds it

Resident decode MoE: warp-per-expert-path GEMV indexing packed expert banks in place at `expert × rows` (no gather, no permute, no sort) at T≤~32; grouped path for prefill. Offload wing: pinned host banks (bank schema registry — a format is a tuple of bank names; movement layer layout-agnostic), device flat-id LRU (multi-CTA ensure kernel — FreeToken's single-CTA `grid=(1,)` argmin is a named batch-size cliff), fused multi-bank UVA gather with device-resident count, whole-layer double-buffered prefill streaming, sub-256 KiB `cudaMemcpyBatchAsync` guard, pin-after-fill.

Cold PCIe bytes/token = `L_moe × topk × expert_bytes`:

| Model | Cold MB/tok | Banks | 4.0×16 (≈26 GB/s) | 5.0×16 (≈52) |
|---|---|---|---|---|
| Qwen3.6-35B-A3B NVFP4 | 568 | 18.2 GB | 46 | 92 — but it fits **resident** ≥24 GB |
| gpt-oss-120b MXFP4 | 1,906 | 61 GB | 13.6 | 27.3 |
| Qwen3-30B-A3B BF16 | 3,624 | 58 GB | 7 — **refused; quantize first** | 14 |
| GLM-4.7 NVFP4 | 9,461 | 189 GB | 2.8 — **refused** | 5.5 — **refused** |

**All offload tok/s gates derive from E6's *measured* gather bandwidth, gated at ≤70% of it** (critique must-fix: the spec-sheet ceilings above are upper bounds; FreeToken measured ~31 GB/s where spec said 26–31, and realized gather < link spec is the norm). Host-RAM requirement (e.g. 61 GB pinned for gpt-oss-120b) is stated at load and pinning is verified. CPU-MoE hybrid co-execution is **not built** (fails FreeToken's own 2× rule on the 50-series half of the card list); its movement primitives (stream memops, doorbells) are adopted where useful.

### 8.10 Speculative decoding

Deferred post-v1, doors held open cheaply: the paged-KV frontier design already supports provisional leads (rejected tails become unreachable; pages sized `+C·ceil((K−1)/64)`); the sampler's RNG purposes enum reserves speculative-accept/correction/bonus; hybrids get ReplaySSM (record raw k/v/g/β per verify column, fold accepted prefix, **record and fold generated from one Triton source with pinned constants** — bit-identity is a property of shared code). MTP-style linear chains only; DFlash-style bespoke drafters rejected (lost to MTP3 on 5 of 7 workloads, needs a per-checkpoint trained drafter). Acceptance is a checkpoint property (NInfer: 68–71% vs 46–49% on identical module inventories) — exit criteria will gate on end-to-end throughput, never on acceptance.

### 8.11 Memory plan

Startup-frozen: three owning allocations (weights arena — separately-addressable and layout-stable, see §14; persistent arena: KV pools + state slots + round frames; shared workspace arena) plus an enforced graph allowance. Sized by dry-run simulation of the real recipes; KV capacity solved on the affine curve; headroom measured (E3), not assumed at 1 GiB. The budget module is pure integer arithmetic over measured bytes, unit-tested with no GPU. The planner prices: weights, expert banks/slots, KV, GDN state floor (context-independent, concurrency-capping), graph pool, workspace, fragmentation margin — and prints the reconciliation vs `cudaMemGetInfo` at startup (must agree within 5%).

---

## 8b. Multi-GPU (C11)

**The interconnect reality**: **P2P is assumed `False` on every card on the target list — Pro 6000 included (owner directive)**. The measured basis agrees where we can measure: `torch.cuda.can_device_access_peer() == False` between two RTX 5090s on the same host bridge (driver 590.44.01) — and some driver versions have *lied* (reported capable, writes silently failed: vLLM #2728), which is why the only trusted signal is a real-write probe and its default expectation is failure. Every GPU↔GPU byte is planned as host-staged; a 2-rank NCCL all-reduce costs ~25–60 µs regardless of payload (8–16 KB at decode). 3090 NVLink pairs (~1.5–2 µs) are legacy-only and out of scope with sm_86.

**Per-token TP arithmetic** (2 all-reduces/layer, payload hidden×2B): comms/token = `2·L·t_ar`. 8B (L=32) host-staged-2r ≈ 2.6 ms vs ~5.3 ms single-GPU compute at Q4 → **wash**. 32B (L=64) on 2×4090 ≈ 5.1 ms comms vs 10.6 ms compute saved → fragile ~1.35×, destroyed by t_ar jitter or x8 slots. 32B on 2×5090 → wash (faster HBM makes fixed comms relatively bigger). 4-rank on consumer boards (x8/chipset lanes, t_ar 100–200 µs) → **loss**. 70B on 2×Pro 6000 host-staged (L=80): 4–9.6 ms comms vs ~13 ms saved → fragile 1.15–1.5× at best — and a 70B W4.5 (~39 GB) fits a single 96 GB card anyway, making TP2 there a latency-only play that the host-staged arithmetic does not reliably pay for. **Conclusion: TP is a net win nowhere on the target list under the P2P=False assumption.**

**Verdict table:**

| Config | Mode | Rationale |
|---|---|---|
| Model fits one GPU, N GPUs | **DP replicas** (shared-nothing, API-layer routing) | Only mode that scales throughput with zero per-token collectives; **TP refused** (wash at best) |
| Dense model needs 2 GPUs, GeForce (2×4090, 2×5090) | **Layer-split** (contiguous layer ranges, one 8–16 KB activation hop per boundary via host-staged `cudaMemcpyAsync` — no collectives needed) | Capacity scaling, per-GPU graph capture; startup prints "capacity, not latency: bs=1 tok/s ≈ single-GPU-equivalent" |
| MoE needs 2 GPUs, GeForce | Layer-split with experts local per stage | **EP refused**: only the naive all-gather a2a works without NVLink/RDMA and it loses at decode batch sizes |
| 4× GeForce | Layer-split only | TP4 host-staged AR latency ≥ compute saved on real boards |
| 2× RTX Pro 6000 | **Layer-split / DP, same as GeForce** (P2P assumed off) | 96+96 GB layer-split holds ~180 GB of model+KV that no GeForce pair can; TP2 offers no reliable win host-staged (see arithmetic above) |
| Any TP request on this card list | **Refused, arithmetic printed** (`2·L·t_ar` vs `W/(N·BW)`) — unless the real-write probe passes at startup, which unlocks TP2 opportunistically (not expected on any listed card) | Ships nothing that benchmarks worse than one GPU |

**Mechanics:** startup runs a **real-write P2P probe** (vLLM's `can_actually_p2p` pattern — subprocess IPC-handle write + compare; never trust `cudaDeviceCanAccessPeer`), reads `nvidia-smi topo` (this dev fleet itself runs 4 of 8 GPUs at x8 across NUMA nodes — uniform-x16 assumptions are wrong in the field), and prints the computed budget for every offered/refused mode. **The probe's expected outcome on every listed card is failure (owner directive: assume P2P False); it exists as a defensive check, and TP2 code activates only on a genuine pass — no shipped configuration depends on one.** If TP2 ever activates: KV is sharded by head (with the GQA caveat: kv_heads < tp ⇒ replicated KV, eroding the capacity win — printed). NCCL-inside-graph-capture is production-proven on P2P transports (vLLM, FreeToken) with the teardown invariant **graphs destroyed before comms** (`ncclCommAbort` deadlocks otherwise — documented in both codebases) enforced from day one; **NCCL-SHM-transport under capture (the only transport available with P2P off) is unproven anywhere and is validated in E8 before any collective is ever captured** — layer-split needs no collectives, so every listed card ships regardless. Reused from surogate: `NCCLCommunicator` bootstrap, single-process-per-GPU threading, abort/watchdog hardening (`comm.{h,cpp}`); **not** reused: the transaction model (host-barrier per execute cycle — throughput-shaped for per-step gradient reduces, catastrophic at 2 collectives/layer/token) and the EP dispatch control plane (host spin-waits, per-layer D2H — training-shaped, capture-incompatible).

---

## 9. Serving layer

Present from Phase 1 (C10). C++ (`csrc/src/serve/http/`, cpp-httplib vendored via FetchContent like minja); no Python in the serving path.

### 9.1 APIs
`POST /v1/chat/completions`, `/v1/completions` (OpenAI), `POST /v1/messages`, `/v1/messages/count_tokens` (Anthropic, Phase 3), `/v1/models` (list/load/unload), `/tokenize`, `/health`, `/metrics` (Prometheus text: queue depth, per-round timing, KV/expert-cache occupancy and hit rates, tier and roofline-fraction per model). One protocol-neutral typed event core (GenEvent: ReasoningDelta/ContentDelta/ToolCallStart/ArgsDelta/Done) fanned into thin per-protocol formatters — Anthropic consumes semantic events, never re-parses OpenAI SSE. Errors follow each protocol's envelope. SSE with disconnect-aware cancellation (atomic flag OR sink-writable check); generation runs inside the chunked-content provider; the GPU worker only appends deltas under a mutex.

### 9.2 Text pipeline
Streaming incremental detokenization (partial-UTF-8 / byte-level Ġ state machine per request — new, ~300 lines); stop-string matching across token boundaries with longest-suffix-prefix hold-back; lossless tool-call stream filter with the emitted-bytes handshake (NInfer's design); reasoning-channel splitting per family. Usage accounting from engine counts, never re-tokenized: `cached_tokens` from prefix reuse, `reasoning_tokens` from the output channel.

### 9.3 Chat templates and tool calls
Real Jinja via the existing minja stack — HF `chat_template` and GGUF-embedded templates (extracted at conversion) both render live; fine-tuned templates just work (NInfer's SHA-pinned compiled templates rejected). `tokenizer::ChatMessage` is extended with `tool_calls`/`tool_call_id`/`reasoning_content` + a `tools` context variable (verified gap: today it is `{role, content}` only). Streaming tool-call + reasoning parsers for the launch families (Qwen ChatML first, gpt-oss Harmony next; take FreeToken's architecture, not its 3.7k-line file). Constrained decoding (§8.8 hook): `response_format: json_schema` and `tool_choice` forcing in Phase 3. FreeToken's probed effort/thinking dialect (render the checkpoint's template with probe kwargs) adopted for reasoning-effort support with zero per-family code.

### 9.4 Startup and model-switch time — a first-class product property

For a local serving app the most frequent operation after "generate" is "switch model". Budgets, printed as a breakdown at every load:

| Stage | Cold (first ever load) | Warm (artifact cached) | Mechanism |
|---|---|---|---|
| Format conversion/repack | seconds–2 min (30B; E9 measures) | **0** | streamed per-layer on-GPU repack → artifact cache |
| Weight load → VRAM | — | 8B ≈ 2–4 s, 32B ≈ 8–15 s | O_DIRECT + 4×64 MiB pinned slots, pin-after-fill, ≥70% of NVMe seq read |
| Per-arch scale swizzle | 1–2 s | 1–2 s | load-time, keeps the artifact arch-neutral |
| Graph capture | ≤10 s target (E3 adds the time dimension) | same | small default bucket set (~16 captures); lazy capture of rare buckets |
| Tokenizer init | <1 s (native C++; measured, not assumed — SGLang's 13 s is a Python-stack number) | same | |
| **Total model switch (warm)** | — | **8B ≤ 15 s, 32B ≤ 30 s** | exit-criterion-gated in Phase 2 |

**The artifact cache is the primary startup-cost amortizer**: repack (the CPU/GPU-expensive part SGLang's analysis flags for artifact-first designs) is paid once per checkpoint per layout version, not per process start — and unlike FreeToken's FTW it is arch-portable.

### 9.5 Model management and LoRA
Load/unload/hot-swap via `/v1/models` with the maintenance-gate state machine (loading/serving/switching; late replies cannot resurrect a latched failure — FreeToken's tested design). **Serve-time LoRA position (v1): merged-only** — `AdapterMerger` folds the adapter at load; the artifact variant is fingerprinted by adapter hash. Runtime multi-adapter serving is deferred with its retrofit cost named: the ServeOp argument binding reserves an optional adapter-pointer slot now (cheap), so adding per-request adapter GEMVs later is an op-table change, not an ABI break.

### 9.6 Weight-cache daemon: assessed, deferred, designed-for

SGLang's daemon (weights loaded once, exported as CUDA IPC handles, engines attach zero-copy; 495 s → 0.63 s weight load) solves a multi-minute, multi-GPU datacenter cold start. The transferable parts for a desktop app: (i) a **resident weight daemon makes model *switch* and engine *restart/upgrade* near-instant** — worth having eventually, and FreeToken already ships the process shape (torch-free supervisor, pidfile, oom_score_adj, health proxy); (ii) it amortizes repack across restarts — but our disk artifact already does that, so the daemon's marginal win is skipping the NVMe read + VRAM upload (~2–15 s), i.e. a v2 feature, not a v1 requirement; (iii) CUDA graphs are not IPC-shareable, so capture time is paid per engine process regardless — another reason to keep the bucket set small. **Decision: v1 ships the artifact cache + fast loader; the daemon is post-v1. The design constraint is taken now:** one `cudaMalloc` weight arena per model, layout-stable per-tensor offsets recorded in the (re-derivable) artifact index, all quantization metadata in-band and device-resident — so every tensor is IPC-exportable raw, and the daemon (and the §14 trainer handoff) bolt on without touching the format subsystem. SGLang could not do this for GPTQ/AWQ/Marlin/per-tensor-FP8; we can, because of §2.1.

### 9.7 Platform position (stated and CI'd)
**v1 is Linux and WSL2.** Windows-native is deferred (WDDM breaks UVA host-pointer gather and stream memops; pin quotas differ). Every Linux-ism is probed at startup with graceful degradation, not assumed: O_DIRECT (fallback: buffered mmap + MADV_SEQUENTIAL — required for tmpfs/network FS/WSL2 anyway), pin-after-fill (fallback: staged copies), UVA gather (fallback: per-bank `cudaMemcpyAsync`, offload wing slower but alive). CI includes a WSL2 job for the load paths.

---

## 10. Phased plan

Efforts are **nominal / planned**, planned = nominal × ~1.75 (the critique's re-baseline; historical multiplier on this class of work is 1.5–2.5×). One experienced CUDA/systems engineer per line unless noted. Total to end of Phase 5: **~66 nominal / ~115 planned engineer-weeks** — parallelizable across 2–3 engineers to roughly 10–12 calendar months. Under port-first the kernel share of that total is **~20–22 nominal engineer-weeks of porting** (NInfer ~11, FreeToken ~9–11, per the §2.2b costs) — roughly what authoring would have cost in wall-clock, but with published perf numbers as the target instead of a bet; the risk moves from "can we write it" to "does it survive transit" (§12).

| # | Phase | Deliverable | Exit criteria (testable) | Nom/Plan | Depends |
|---|---|---|---|---|---|
| **P0** | **Calibrate & port-validate** (runs before schedule freeze; overlaps P1 start) | The §13 experiments: read-ceiling probe, **the port-validation spike (E2 reframed)** — NInfer seam (`core/` + `common/` with sm_89 guards) + q4 rowsplit GEMV ported verbatim, measured under surogate's runtime on 4090+5090 (**the first sm_89 compile in that code's history**), plus one FreeToken Triton family (NVFP4 W4A16) AOT'd end-to-end through `compiler.py`→JitKernel — capture cost+time, gather bench, prefill-executor cost, conversion cost | All experiments report numbers; every decode/offload gate re-derived from them; sm_89 build + parity of NInfer `common/` + q4 GEMV green on a 4090; ported q4 GEMV within ~5% of NInfer's published 5090 band; written sizing of the §2.2 gap list (zero-plane, sweeps, hand-twin candidates) | 3/5 w | — |
| **P1** | **Served generation, end to end** | `surogate serve Qwen/Qwen3-8B` (+ Llama-3.1-8B once its oracle exists): safetensors BF16 + surogate-FP8; **ServeExecutor eager decode**; paged KV (BF16, P=64); decode attention = **ported FreeToken split-KV Triton kernel** as the bridge (bf16 KV, 5–8 d, page-table adaptation) while the NInfer GQA port proceeds in P2; **ported NInfer fused sampler** (3–5 d; top-k≤20 v1, logprob-at-temperature, mask hook stubbed); **ported FreeToken `store.cu`/`index.cu`** for KV append + embedding gather (with §7.2's block-table variant); NInfer seam + elementwise ports from P0 (~1.5 wk total incl. BF16 decode GEMV); prefill via GraphExecutor forward-only + paged KV export; inference-only DslModel; OpenAI chat/completions + SSE + streaming detok + stop strings + cancellation; boundary-loop scheduler with capped-reservation admission | (a) curl streams a correct completion on a 4090; (b) greedy decode-with-cache **token-identical** to full-forward (onboarding-harness oracle) for 512 tokens; (c) B=1 BF16-8B ≥ **60%** of roofline (eager gate; ≥34 tok/s on 4090); (d) 4 concurrent streams interleave correctly, disconnect cancels within one round; (e) peak VRAM < weights + 1.5 GB (grad store proven gone); (f) planner report reconciles to `cudaMemGetInfo` within 5% | 12/20 w | P0 partial |
| **P2** | **Fast path + quant front door** | CUDA-graph decode (buckets {1,2,4,8} × ctx profiles) with sampler in-graph; **NInfer rowsplit linear family ported** (Q4/Q5/Q6/W8 GEMV/small-T/GEMM, ~2–3 wk, + fused SwiGLU/add/QKV epilogue catalog +2–3 wk — repack emits the concatenated parent tensors §2.1 requires); **NInfer paged GQA + fused INT8-G64 KV ported with the D=128 template generalization** (2.5–4 wk, retiring the P1 Triton bridge; its 68.6 KB parity oracle ports with it); route tables re-swept per target card (`bench/ops` harness port, 2–3 d); **zero-plane mainloop extension for q4+w8** (+1.5–2 wk/family — the GPTQ/AWQ carriers); INT8-G64 KV default; format subsystem v1: GGUF (Q4_0/Q8_0/Q4_K/Q5_K/Q6_K + tokenizer-in-file), GPTQ (incl. act-order), AWQ → AQT (row-split-k128 + zero plane); **BF16 group-RTN quantize-at-load**; artifact cache (arch-neutral) + `surogate convert` + `--no-cache`; per-format CPU oracles in CI (NInfer `tools/` converter oracles as reference); **per-slot retained-continuation prefix cache**; Tier-0 generic path with floor-gated admission; model-switch UX + warm-switch budget | (a) a llama.cpp-quantized Q4_K_M GGUF and a TheBloke-style act-order GPTQ repo each serve, logits match CPU dequant oracle; (b) Qwen3-32B W4.5 ≥ E2-calibrated gate (provisional: ≥32 tok/s 4090 @≥16k INT8-ctx, ≥57 5090 @≥32k); (c) one graph launch + one host sync per round (nsys-committed trace); (d) same seed → identical tokens across replays and buckets; (e) multi-turn chat re-prefills only the new turn (measured prefill-token reduction ≥60% on a 10-turn script); (f) warm model switch ≤15 s (8B) / ≤30 s (32B) incl. capture; (g) refusals print arithmetic (i-quant, 32B-on-16GB) | 14/25 w | P1 |
| **P3** | **Breadth + serving completeness** | Decode traces for all 11 families; **llama + qwen3_5_moe parity oracles written**; **GDN stack ported** (NInfer recurrent + chunked + conv1d + ReplaySSM, 1.5–2 wk, geometry asserts relaxed; FreeToken FLA decode step, 3–4 d, as the Triton alternative where head counts differ) + lane-affine state pool → qwen3_5/lfm2/nemotron_h/laguna; gpt-oss sinks + gemma4 per-layer SWA in the paged backend; prefill migrates onto ServeExecutor (paged-prefix chunked prefill); BSF formats live via ports: **NInfer FP8 row-scale incl. A8** (~1 wk, sm_89 mma respell + accumulate-rate re-bench), **NInfer NVFP4 W4A16** (~1 wk incl. sm_89 decode fallback), **FreeToken FP8 Triton files** for per-tensor/128×128-block/mxfp8 (4–6 d), **written MXFP4/E8M0 codec variant** (2–3 d), NF4-LUT written; Anthropic `/v1/messages`; tool-call + reasoning stream parsers; **constrained decoding** (json_schema/tool_choice via the mask hook); probed effort dialect | (a) all 11 families pass decode parity (hybrids: state after N tokens matches chunked forward within tolerance; bit-compat plan for T=1-vs-BT=64 executed); (b) gpt-oss-20b MXFP4 resident ≥100 tok/s on 4090 and serves on a 12 GB card with ≥16k INT8-ctx; (c) Anthropic client completes a multi-turn tool round-trip, zero leaked markup; (d) a json_schema request emits schema-valid output under the grammar mask inside the captured graph; (e) an NVIDIA NVFP4 and a DeepSeek-style block-FP8 repo serve on sm_120 via BSF, matching oracles | 14/25 w | P1 (perf bits P2) |
| **P4** | **MoE dual path** | **NInfer sparse MoE ported + geometry-parameterized** (1.5–2 wk; 256-experts/top-8/2048 constants → parameters; the device job-list decode kept verbatim — it *is* the graph-compatibility recipe, and it deletes the `moe_permute` host spin-poll for serving) → resident MoE graph-captured over AQT/BSF expert banks; offload wing: **FreeToken LRU/policy kernels AOT'd** (4–6 d incl. unifying on the hybrid kernel over external-flashlib), copy/gather CUDA already in from P1, **C++ offload-cache rewrite from the Python executable spec** (10–15 d + 3–5 d O_DIRECT/pin-after-fill loaders), **MXFP4 MoE (3–4 d) + NVFP4 fused MoE (2–3 d) Triton ports**, multi-CTA LRU reshape as follow-up, KV-first planner, load-time admission rule; kill-list refusals | (a) resident MoE decode fully inside one captured graph (zero stream syncs — capture-walker-verified); (b) Qwen3-30B-A3B W4.5 resident ≥220 tok/s (4090) / ≥340 (5090); (c) gpt-oss-120b MXFP4 on 5090 (61 GB pinned banks): warm ≥25 tok/s, cold ≥ 70% of E6-measured gather ceiling; 4090 warm ≥15; (d) GLM-class load refuses with the bandwidth arithmetic; (e) resident-vs-offload admission decision printed with bytes at load | 13/23 w | P2, P3 (gpt-oss) |
| **P5** | **sm_120 W4A4 + multi-GPU** | **NInfer NVFP4 W4A4 block-scaled MMA + TMA variant ported** (1–2 wk, sm_120-only cells by physics; the 128×4 scale swizzle from §5.2 is its MMA operand layout; replaces the previously tracked CUTLASS Sm120 escape hatch — smaller, tuned, dependency-free); DP replica routing behind one endpoint; layer-split 2–4 GPUs with per-GPU graphs; P2P write probe + topology report; TP refusal with printed arithmetic (P2P assumed False on all listed cards — TP2 exists only as probe-gated opportunistic code, not a deliverable, and carries no gate) | (a) ported NVFP4 W4A4 prefill reproduces NInfer's hand-tuned 52–59%-of-dense-FP4-peak band on 5090 (gate ≥50%); (b) 2×5090 layer-split serves a 70B-class W4.5 model neither card fits, single-stream ≤1.15× ideal-combined-compute; DP ≥1.9× at 8 concurrent; (c) 2×Pro 6000 layer-split serves a ~120B-class W4.5 model with the same ≤1.15× bound; (d) any TP request exits with printed arithmetic (unless the real-write probe genuinely passes); (e) graphs-before-comms teardown passes kill/restart soak | 10/17 w | P2 (graphs); indep. of P3/P4 |
| **P6** | **Post-v1 backlog (demand-gated)** | Elastic VRAM rebuild (FreeToken's two-phase spec, kept); radix prefix tree + CoW fork; speculative decoding (MTP + ReplaySSM); weight-cache daemon (§9.6); Windows-native; CPU-MoE hybrid **only if** 40-series+DDR4/DDR5-PCIe4 demand shows in telemetry; runtime multi-LoRA | per-item | v1 |

**P1 produces working SERVED generation end-to-end** — eager, honest 60%-roofline gate, real HTTP/SSE — with graphs and the 80%+ gates in P2 where they belong.

---

## 11. Testing and validation

1. **Correctness ladder.** (i) Per-format **CPU dequant oracles** (bit-exact affine; ≤1e-6 rel scale-widened) gate every reader/repacker. (ii) **HF reference parity**: greedy decode-with-cache token-identical (fallback criterion: top-1 ≥99.5% + logit rel-L2 <3e-3, recorded which was used) vs full-sequence forward via the onboarding harnesses — extended to llama and qwen3_5_moe in P3. (iii) **Cross-tier parity**: Tier-1 fast path vs Tier-0 dequant path on identical inputs. (iv) Hybrid state parity: T=1 GDN step vs BT=64 chunked forward on committed state after N tokens — the bit-compat requirement is a designed test (shared codec constants, pinned tiles), not an assertion.
2. **Output quality after repack/quantization — measured, never cited.** For each {AQT-4/5/8, BSF-FP8, BSF-NVFP4, MXFP4, NF4} × {BF16-KV, INT8-G64-KV}: perplexity on a fixed 100k-token held-out set vs the BF16 oracle, published per phase report. (No reference repo publishes an INT8-KV end-to-end A/B; we generate our own.)
3. **Perf regression harness.** Per-kernel GB/s vs the **locally measured** read ceiling (never spec), cold-L2, median-of-≥20, full pointwise curve with the largest adjacent-extent jump named; NCU gate on every GEMV: DRAM read within +2% of one-pass weight bytes; per-engine tok/s at B∈{1,2,4,8} × ctx∈{1k,8k,32k} per card per commit. Per-codec roofline gates in CI (Verbatim's discipline) — a combo that regresses below its gate demotes to Tier-0 and fails the build note.
4. **Determinism.** Fixed-seed graph replay reproduces tokens bit-exactly; same-binary-twice produces identical outputs (the repo's own history shows this failing — `project_moe_run_nondeterminism`); bucket-divergence measured and tracked (informational for serving; contract-relevant for §14).
5. **GPU-less CI** (new `unit-tests` target — none exists today): format readers/repack reference codecs (CPU), artifact index round-trip, budget/planner integer arithmetic, ServePlan emission (given model IR + target tuple, assert op list/constants/grid expressions), grid-expression evaluation vs synthetic manifests, scheduler/admission state machines with a mock device, codegen idempotence (regeneration is a no-op diff). Plus a WSL2 CI job for load-path probes and 12 GB-card CI for the planner (a ±0.5 GB error moves context ±12k tokens on 24 GB and OOMs the 12 GB tier).
6. **Soaks.** 1k-request mixed-length soak, zero recaptures, zero VRAM growth; kill/disconnect storms; model-switch loops (50 switches, budget held).

---

## 12. Risks and open questions (ranked)

| # | Risk | Early resolution |
|---|---|---|
| 1 | **Arch-boundedness of the NInfer bodies on sm_89.** The ~90%-roofline decode core has **never been compiled for sm_89** — the inventory verified the bodies are sm_80-class, but the guards are unproven in anger: PDL launches in 8 files, the `kind::f8f6f4` mma respell, `cvt.e2m1`/`cuda_fp4.h` fallbacks, the 96 KiB prefill smem vs sm_89's ~99 KiB opt-in carveout, FP8-mma fp32-accumulate half-rate, occupancy constants tuned to a 170-SM 5090. If sm_89 perf lands far below the 5090 band, the 4070/4090 gates slip | **P0, week 1: sm_89 compile + parity of `common/` + the q4 GEMV on a 4090** — days, not weeks, and it de-risks the entire 40-series claim before any other port proceeds; a 4090 CI build is mandatory before any 40-series number is published |
| 2 | **Launcher-coupling depth.** The bodies are clean but everything around them is model-shaped: 5.3k lines of exact-shape wrapper whitelists (replaced, not ported), `kGqaHeadDim=256` as a file-scope constant whose D=128 generalization touches every derived constexpr in the attention family (the i8 producer/consumer warp partition assumes D=256 divisibility), MoE 256/8/2048 constants, per-(N,K) route tables whose losing sweep candidates were deleted. The biggest hidden port cost is here, not in the instructions | D=128 generalization costed inside the attention port (2.5–4 wk family total) with the D=128 bidirectional kernel as the existence proof; geometry-parameterization named per family in §2.2b; the ported `bench/ops` harness regenerates route tables per card; ported parity oracles (`tests/ops`) gate every generalization step |
| 3 | **Triton-AOT feasibility for the FreeToken kernels.** Decode paths are autotune-free and AOT-clean, but the FLA chunk-prefill and sampler carry `@triton.autotune`, several hosts read device props at import, and per-device cached workspaces (split-K counters, capture buffers) must become manifest-declared workspaces — if `compiler.py` can't express any of this, those families need C++ sequencing shims or re-hosting | **P0 AOTs one family end-to-end** (NVFP4 W4A16 GEMV) and one single-CTA LRU policy kernel through `compiler.py`→JitKernel; autotuned config spaces swept and pinned offline (the existing `autotune_triton_kernel` path) before manifests exist; the workspace convention is designed in the P1 seam, not retrofitted |
| 4 | **Scope/effort under-estimate.** A from-scratch serving engine + the largest-single-work-item format subsystem; NInfer is 144k LOC for 2 models on 1 arch, and porting adds integration/qualification labor the inventories price in engineer-days that historically stretch | ×1.75 planning multiplier applied; re-baseline after P2 with real velocity; tier ladder means a slipped fast path degrades to Tier-0 instead of blocking a release |
| 5 | **The symmetric-quantization gap bites late.** Nothing in either repo supports a zero/min plane; GPTQ/AWQ — headline P2 formats — depend on the written zero-plane mainloop extension (+1.5–2 wk per code-width family) landing on schedule, or they serve via Tier-0 | Zero-plane decision forced in P2 **before** the repacker is written (the inventory's explicit sequencing); Tier-0 admission-gated fallback keeps GPTQ/AWQ *served* (slow) if the extension slips; K-quant FP32-scale variant rides the existing scale-access-mode template axis |
| 6 | **Model-family long tail** (sinks, per-layer SWA, per-layer head dims, GDN T=1 bit-compat, VL); llama + qwen3_5_moe oracles don't exist yet; per-family quirks historically overrun (gemma4, laguna in MEMORY.md) | Per-family cost stated at 1–2 wk not days; oracles costed in P3; the parity harness catches divergence immediately; families ship incrementally behind the tier banner |
| 7 | **Consumer-market UX risks**: conversion friction vs llama.cpp's zero-step UX; Windows-majority market vs Linux/WSL2 v1; 12 GB planner errors are OOMs on the flagship budget tier | Transparent first-load conversion + `--no-cache`; WSL2 declared and CI'd, Windows-native in P6; memGetInfo-bracketed budget enforcement + 12 GB CI card |
| 8 | **Admission/UX interactions**: capped reservation may end long generations with `length` under load; retained-continuation cache misses when clients rewrite history (thinking-block stripping) | Per-tier concurrency tables published; reserve_cap configurable; semantic-anchor checkpoint (P6) addresses the rewrite case; identity predicate keeps misses correct, never wrong |
| 9 | **Offload driver pathologies** on consumer boards (batch-memcpy degradation class, CUDA≥13 requirement for `cudaMemcpyBatchAsync`, x8 links, WDDM) | E6 benches the production gather on real boards; guards + CUDA-12 fallback ported from FreeToken; refusal gates on x8 |
| 10 | **Affine KV solver breaks on non-linear workspace** (MoE expert buffers) | Size MoE workspace on a fixed token-slice cap; assert affinity at finalize, fail loudly |
| 11 | **Format-convention drift** (NVFP4 divisor/multiplier, `weight_scale` shape overloading, gptq_v2, new GGUF types, split-GGUF) | Shape+dtype-based scale dispatch; per-format oracles; quarterly format-share review of top HF downloads; refusal path keeps unknown variants honest |

---

## 13. Decision-blocking experiments

All run in P0 on one 4090 + one 5090 (idle GPUs on this fleet; obey the no-builds-during-runs rule). **The schedule does not freeze until E1–E3 and E6–E7 report; E2 is no longer a go/no-go on Triton-vs-hand — it is the port-validation spike that calibrates every gate and sizes the gap list.**

| # | Question | Exact measurement | Decides |
|---|---|---|---|
| **E1** | Real pure-read ceiling per card | Port NInfer's probe: 4 GiB `uint4` grid-stride read, `csrc/src/testing/bench/hbm_probe.cu`; `CUDA_VISIBLE_DEVICES=1 ./bench_hbm` on 4090 and 5090; expect ~93% of spec | Re-baselines η and every tok/s gate |
| **E2** | **Do the PORTED kernels hit their published perf under surogate's runtime — and how big is the gap list?** (port validation, replacing the Triton-vs-hand bet) | 2-week spike: (a) port the NInfer seam (`core/` PODs + arena, `common/{mma,memory,warp,math}.cuh` with sm_89 guards) and the **q4 rowsplit GEMV verbatim**; build for sm_89 **and** sm_120 (first-ever sm_89 compile of this code); run its ported CPU-oracle parity tests; measure cold-L2 median-of-20 vs E1's ceilings, NCU one-pass verification — target: reproduce NInfer's 89.9–91.8% band on the 5090, establish the sm_89 number; (b) AOT the **FreeToken NVFP4 W4A16 GEMV family** through `compiler.py`→JitKernel end-to-end (its ~64%-of-roofline self-measurement is the comparison point) plus one single-CTA LRU policy kernel — proving manifest-workspace + device-count-tensor conventions on ported Triton; (c) one route-table re-sweep on the 4090 to price the per-card sweep obligation | Every decode gate re-derived from the ported numbers per card; the §2.2 gap list sized with measurements (sm_89 deltas, sweep cost, which FreeToken Triton families warrant an NInfer-side hand twin); go/no-go on nothing — the kernels already exist; what E2 decides is *calibration and sequencing* |
| **E3** | CUDA-graph capture cost in VRAM **and seconds** | Bracket capture of a realistic decode step (8B model) with `cudaMemGetInfo` + wall clock across buckets {1,2,4,8}×4 ctx profiles | Graph allowance (replaces the assumed 0.8 GB, moves context ±12k/±0.5 GB on 24 GB) and the ≤10 s capture budget in §9.4 (SGLang's 7.7 s datapoint says this is real) |
| **E4** | Where does cuBLASLt stand at decode/prefill shapes vs vLLM? | `matmul.cpp` heuristic path at M∈{1,2,4,8,16} and prefill shapes for 8B/32B geometries vs a vLLM baseline, same card/prompts | Prefill-parity claim; whether a pinned skinny-GEMM algo entry is needed (no M==1 case exists today) |
| **E5** | Batch-bucket logprob divergence | Score one 512-token sequence at bucket 1 vs 8, split-K fixed per (N,K, ctx-bucket) vs free | Informational for serving; sets the §14.5 note on what a later batch-invariance fix costs |
| **E6** | **Measured** PCIe gather bandwidth on real boards | FreeToken-style `bench bw`: production-shape UVA gather kernel over pinned synthetic banks, 4090 (PCIe4) + 5090 (PCIe5), x16 and (if available) x8 slot | All offload cold gates (set at ≤70% of this number, not link spec — fixes the proposals' 93%-of-theoretical gates) |
| **E7** | GraphExecutor forward-only prefill: per-bucket compile time, VRAM with `DslGradStore` gated | Instrument `compile_graphs` at T∈{512,1024,2048,4096}, B=1, on the inference-only construction branch | P1 prefill bucket set; startup-latency line; whether P3's ServeExecutor-prefill migration must accelerate |
| **E8** | NCCL-SHM under capture + P2P assumption check | (a) capture a 2-rank pynccl all-reduce on the 5090 pair, replay 1k× — does SHM proxy progression survive? (b) vLLM-style write-probe on a Pro 6000 pair when hardware is available — expected result **False** (owner directive: assume P2P off on all RTX cards); this experiment can only confirm the assumption or surprise us, it gates nothing | Whether any collective may ever be captured (layer-split doesn't need it); TP2 stays dormant unless (b) surprises |
| **E9** | Conversion cost truth | `surogate convert` prototype on a 30B Q4_K_M GGUF and a 32B GPTQ repo: wall time, peak host RAM, peak VRAM, artifact size | The §5.4 UX numbers and the P2 exit criterion; decides default chunk/stream sizes |

---

## 14. Forward compatibility with a later GRPO switch

The owner's sequencing: ship the serving engine first; migrate GRPO rollouts onto it and retire vLLM later. **That migration does not drive this plan's scope, phasing, or gates.** These five decisions are cheap now and expensive to retrofit; each is taken (or explicitly deferred with its cost):

1. **Embeddable library, not server-only — TAKEN.** The engine is `libsurogate-serve` (C++ library, NInfer's `engine.h` shape); the HTTP server is a thin app over it; a pybind11 binding (~1–2 weeks, and P1's test harness wants it anyway) gives an in-process Python caller. A future colocated rollout driver calls the library directly — no socket, no subprocess.
2. **Weight storage that can be overwritten in place — TAKEN.** One layout-stable, separately-addressable arena per model with per-tensor offsets in the re-derivable artifact index (§9.6 already requires this). The real tension is named: a trainer holds BF16/FP8 masters, so weight sync must **quantize-and-pack into the engine's internal layout on update**. The repack kernels are therefore written device-pointer-in → device-pointer-out (they already run on-GPU at ingest), and a `refresh_weights(named device ptrs)` entry point is stubbed on the library API. Cost taken now: keeping repack kernels D2D-capable (~0). What is *not* built: the sync orchestration, prefix-cache invalidation protocol, or any trainer-side plumbing. CUDA IPC is the natural zero-copy handoff for the colocated case — and works for every format we serve, because the layouts are self-describing (§2.1/§9.6); SGLang's blocklist experience is the cautionary tale this design already avoids.
3. **Per-token logprobs at the sampling temperature — TAKEN.** In the sampler's output contract from day one (§8.8); changing a sampler's contract after the fact ripples through the graph-captured egress struct. OpenAI-compatible top-k logprobs ride the same machinery.
4. **Prefix sharing / fork — ALIGNED.** The page allocator carries a refcount field from Phase 2 and the retained-continuation cache establishes the identity predicate; a G-way rollout fan-out is the same fork mechanism the radix tree (P6) needs. Hybrid models cannot fork recurrent state — a shared-checkpoint-per-group degradation, stated.
5. **Reduction-order stability — NOT built; retrofit path bounded.** Variance would come from split-K/split-KV geometry chosen per envelope/bucket. Because split policy lives in manifests as a pure function of (N, K, ctx-bucket) — never batch size or occupancy — fixing it per (N,K) later is a **manifest/table change, local, not a rewrite**. E5 records the baseline divergence so the later decision is informed.

---

## 15. Explicit non-goals

1. **Beating anyone at prefill.** Parity ±15% via cuBLASLt/CUTLASS/FA2 (C6); the only new prefill GEMMs are the dequant-in-K-loop template and the Phase-5 Sm120 W4A4 body.
2. **A general fusion or auto-scheduling compiler.** Decided on arithmetic: at B=1–8, elementwise fusion saves <0.2% of layer traffic while dequant-into-GEMV saves ~8× — templates + epilogue parameters capture the win.
3. **A persistent megakernel.** CUDA graphs already remove the launch tax; a megakernel welds format × family × arch into one unreviewable body and its smem reservation excludes attention permanently at head_dim 256 on ~99 KiB consumer smem.
4. **Tensor parallelism anywhere on this card list; expert parallelism for inference anywhere on this card list.** P2P is assumed disabled on all RTX cards including the Pro 6000 (owner directive), and host-staged TP arithmetic is a wash-to-loss everywhere. Refused with printed arithmetic; layer-split + DP instead. TP2 survives only as dormant probe-gated code for platforms that genuinely pass a real-write P2P check.
5. **sm_86 / RTX 3090.** Below the build floor; unfunded stretch note only.
6. **CPU–GPU hybrid MoE co-execution** in v1 (fails FreeToken's own 2× rule on 50-series machines); movement primitives adopted, executor not built.
7. **KV quantization below 8 bits.** No validated recipe exists in evidence; INT8-G64 ships and is measured.
8. **Preemption/swap/KV host-offload.** Capped reservation + clean `length` finish instead (§8.5).
9. **i-quants, EXL2, EXL3, AQLM.** Refused with printed reasons; their audience is llama.cpp's/exllama's.
10. **Multimodal serving in v1** (qwen3_vl serves text-only; vision returns a clean 400).
11. **Native GRPO rollout serving as a v1 feature.** §14 takes the free alignments; nothing else is built for it, and no phase gates on it.
12. **Windows-native in v1.** Linux + WSL2, declared and CI'd; Windows-native is P6.
13. **Serving what the arithmetic forbids.** Every dead-end combination in §4.4 is a refusal with numbers, not a degraded mode.

---

## Appendix A — verification record (tree `84b48018`)

| Claim | Status | Evidence |
|---|---|---|
| C2: 119 coarse REGISTER entries, no fusion pass | **TRUE** | `grep -c REGISTER op_registrations.cpp` = 119; every `fuse` hit is a hand-written monolith or load-time weight concat |
| C3 (corrected): CUDA-graph capture exists in-tree | **TRUE** | `graph_executor_utils.h:118-152` — full capture/instantiate/replay with `DeviceMemoryStack` checkpoint/restore (re-read this session); ~39 refs across 10 files; whole-step capture in `py_train.cpp:1512-1650` |
| C4: AOT decode-shaped specialization already shipping | **TRUE** | `~/.cache/surogate/kernels/` live with sm_120 cubins; `matmul_tn_bf16_m16_...json` carries `constants {M:16,...}`, 3-pointer signature, `extra_null_params: 2` |
| `surogate/quant/` is greenfield | **TRUE** | `git ls-files surogate/quant` → 0 files (re-run this session) |
| No HTTP/networking in csrc | **TRUE** | zero `httplib/asio/beast` hits in `csrc/CMakeLists.txt` (re-run this session) |
| DSL model family count | **11, not 12** — the critique's "12" is rejected | `ls surogate/dsl/models/` → 11 model files + `__init__.py` (re-run this session) |
| Onboarding parity harness coverage | **9 of 11 families** (`llama`, `qwen3_5_moe` missing; `gemma4_unified` is a gemma4 variant) | `ls tests/ | grep onboarding` (re-run this session) |
| MoE host round-trip location | `csrc/src/runtime/ops/moe_permute.cpp` (spin-polls `cudaStreamQuery`); the previously cited `moe/offload_manager.cpp` **does not exist** | prior draft's verification, upheld |
| `DslGradStore` unconditional | **TRUE** | `dsl_model.cpp:1310` |
| FA2 vendored without splitkv/paged kernels | **TRUE** | 8 fwd `.cu` instantiations in `FLASH_ATTN_SOURCES`; `num_splits=0` at `flash_attn_varlen.cpp:102,:397` |
| `kvprefix` is not a paged KV manager | **TRUE** | contiguous per-layer training chunk cache; `supports()` throws on B≠1/sinks/non-BF16 |
| Last-token logits kernels exist; compact path blocks on a stream sync | **TRUE** | `gather_rows_bf16` (`kernels.h:1717-1726`), `lm_head_logits_matmul` (`fused_lm_head_loss.cpp:115-180`), sync at `:803`; these are reachable as kernels — ServeExecutor must not call the CompiledExecutor members |
| Only 3 DSL module files emit training-coupled ops | **TRUE** | `modules/{attention,embedding,gated_delta_rule}.py`; `blocks/qwen3_5.py:242` is a docstring |
| No sampler anywhere | **TRUE** | only argmax is an accuracy counter (`fused_classifier.cu:1079`) |
| No unit-tests target | **TRUE** | one executable, `integration-tests`, 5 sources (`csrc/CMakeLists.txt:719`) |
| GeForce P2P disabled on this fleet | **MEASURED** | `can_device_access_peer == False` between 5090 pairs, driver 590.44.01; 4 of 8 GPUs at x8 |
| Build floor SM89 | **TRUE** | `csrc/CMakeLists.txt` `SUROGATE_MIN_CUDA_ARCH 89`, FATAL_ERROR below |
