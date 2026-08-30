# NVFP4 routed experts for the 35B-A3B

**Why.** The 35B is the last model on a non-NVFP4 routed artifact and the last one behind
vLLM: 1,984 tok/s decode against 2,162 on the same shape and day (92 %), with 10× better TTFT.
Every other model gained more from the weight format than from any scheduling lever — the 4B
+54 % and the 27B +37 % over our own W8/mixed artifacts — and the board's own reading is that
format beats scheduling on this hardware. The routed experts are the only part of the 35B
still at Q4/Q5/Q6-from-GGUF.

**What exists already.** `QType::NVFP4` (= 7) with `StorageLayout::BlockScaleK16M128x4V1` is
shipped and serving: the dense linear family (`ops/linear`, `linear_add`, `linear_swiglu`,
`attn_input_proj`, `gdn_input_proj`) all have NVFP4 arms, including a w4a4 TMA path, and the
27B runs an all-NVFP4 artifact. The converter has `convert_nvfp4.py` / `inventory_nvfp4.py`
for the 27B (dense tensors).

**What is missing.** The sparse-MoE path accepts `Q4G64_F16S`, `Q5G64_F16S`, `Q6G64_F16S`,
`W8G32_F16S` and nothing else, and the expert slot cache stores W8 (or the Q4G32AM host bank).
Three components plus a recipe:

## Phase 1 — the decode codec (self-contained, testable)

`ops/sparse_moe/decode/sparse_moe_decode_body.inc` templates every routed kernel over a codec
struct: `kGroupK`, plus `load_one` / `load_pair` / `load_eight` returning dequantised floats
(see `W8Codec`, `Q5Codec`). An `Nvfp4Codec` is mechanical:

- `kGroupK = 16` (NVFP4 blocks a scale over 16 values, against W8's 32).
- `load_eight`: read 4 bytes (eight e2m1 nibbles), one e4m3 block scale, multiply by the
  per-tensor global scale. The dense NVFP4 kernels already carry the decode atoms to copy.
- The D3/D4 schedules assume `kGroupK` divides the intermediate and hidden dims; 16 divides
  both (640 and 2,048/2,560) so the existing schedules hold.

Test: extend `testing/serve/ops/test_sparse_moe.cpp` with an NVFP4 profile beside the Q4/Q5/W8
ones — it already packs weights per codec and compares against an fp64 oracle, so the arm is a
`CodecProfile` entry plus a packer.

**The one design decision Phase 1 forces: where the per-tensor global scale goes.** NVFP4 is
two-level — an e4m3 scale per 16 values *and* a per-tensor divisor, which the dense path
applies as `decode_nvfp4_e4m3(scale) * inverse_weight_divisor` (`nvfp4_gemv.cuh`). The MoE
codec interface (`load_one` / `load_pair` / `load_eight`) has no slot for a scalar, and the
obvious shortcut — folding the divisor into the router weight `alpha` — **only works for the
down projection**. Gate/up feeds SwiGLU, and the nonlinearity means the scale has to be applied
before SiLU, not after. So Phase 1 must either extend the codec interface with a per-expert
divisor pointer (indexed like the codes) or pre-scale at pack time. Pre-scaling is not free
either: NVFP4's block scales are e4m3 and folding a divisor into them loses range. The
interface extension is the honest option, and it is small: the kernels already carry the
expert index needed to look the divisor up.

## Phase 2 — the prefill path

`sparse_moe_prefill_body.inc` is an MMA path that dequantises into bf16 fragments. Two options:

1. **Dequant to bf16, keep the MMA.** Lowest risk, reuses every tile/schedule; wins only the
   bytes (NVFP4 is ~4.25 bpw against W8's 8.5 for the same experts), not the math.
2. **Native fp4 MMA**, as the dense w4a4 TMA path does. Larger, and the routed case has to
   handle the grouped-GEMM job list; do it only if (1) shows the bytes were not the whole win.

Start with (1): it is the smaller change and the board's dense-model evidence says the format
(bytes) is where the gain came from.

## Phase 3 — the expert slot cache

`ops/expert_slot_cache` stores four planes (gate/up codes + scales, down codes + scales) and
gathers 16-byte units. NVFP4 needs codes + e4m3 block scales + a global scale per expert, and
the block scales are a different granularity (16 vs 32), so:

- `expert_slot_pool_bytes` / `expert_slot_weights` gain an NVFP4 arm, as the Q4 host bank did.
- `gather_kernel`'s bank descriptor already carries four planes; the global scale is a small
  per-expert side array (like `q4_bank_planes` added the mins plane).
- The Q4G32AM **host** bank is a separate thing and stays as it is: it feeds the CPU expert
  path, which is VNNI integer and has no fp4 route.

## Phase 4 — the recipe

`RedHatAI/Qwen3.6-35B-A3B-NVFP4` (already on this host, and what vLLM serves for the pair) has
NVFP4 expert tensors. The 35B recipe builds expert-major sources from the GGUF today; add an
NVFP4 source arm that reads that checkpoint's expert tensors and repacks them expert-major into
`BlockScaleK16M128x4V1`, following `qwen3_6_27b/convert_nvfp4.py`. Its `config.json` needs the
same `linear_attn` ignore-list patch the vLLM copy carries.

## Order and gates

1. Phase 1 + its unit test — proves the decode arm numerically.
2. Phase 3 — without it Phase 1 cannot be served, only tested.
3. Phase 4 — produces the artifact.
4. Measure decode at 100 users against the current 1,984 and vLLM's 2,162 before touching
   Phase 2. If the bytes alone close the gap, the prefill MMA work is optional.

**The other half of the 35B gap.** The board attributes it to per-round routed-kernel
efficiency — "at decode width our routed kernel leaves three of four warps idle" — so a
row-parallel decode-width routed kernel is an independent lever worth roughly as much
(~2,340 by the round model). NVFP4 and that kernel compose; neither subsumes the other.
