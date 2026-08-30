# NVFP4 routed experts for the 35B-A3B

**Status 2026-08-30: built, measured, and not adopted.** The path is complete — decode codec,
small-T, wide rounds as small-T slices, the format's per-expert second level, a converter arm, a
`routed-nvfp4` weights profile and a 21.9 GB artifact that serves coherent text. It is also
**6.4x slower at prompt processing and flat at decode**, so the 35B keeps its groupwise-int
artifact. The numbers and the reason are in `design/INFERENCE.md` under 2026-08-30; the short
version is one line of arithmetic this plan never did:

    NVFP4 is 4.5 bits/weight. Q4G64_F16S is 4.25.

NVFP4 spends an E4M3 scale every 16 values (0.5 bpw) where Q4G64 spends an FP16 every 64
(0.25). The 4B's +54 % and the 27B's +37 % came from **W8 (8.5 bpw) → NVFP4**, halving the
bytes. The 35B's routed gate/up was already Q4G64, so it *grew* 5.9 %; only the Q5/Q6 down
projection shrank, and the whole artifact moved 19.59 → 19.19 GiB, **2.0 %**. A 2 % byte cut
cannot move a bandwidth-bound decode, and it did not.

So the premise at the top of this plan — "every other model gained more from the weight format
than from any scheduling lever" — was true and still led to the wrong target. What those models
gained was *fewer bytes*; the 35B had already spent that gain.

**What the work is still worth.** The arm is correct at every width against the fp64 oracle and
is what a routed model arriving on W8 would want; the packed-lane generalisation and the
per-expert second level are both reusable; and the bits-per-weight table above is the part to
remember before the next "switch it to NVFP4" proposal.

**What the 35B's decode gap actually needs** is the other lever this document already names: a
row-parallel decode-width routed kernel (~2,340 tok/s by the round model). NVFP4 does not
substitute for it.

Two corrections to the plan below, both found by measuring:

* **Phase 3 (the expert slot cache) was never on this path.** The slot cache belongs to one
  target, `qwen4exp` (Flash-Next, host-offloaded); the 35B is VRAM-resident, so its routed
  weights are plain resident `Weight`s. Phase 1 was never blocked on it.
* **Phase 2 (the prefill MMA arm) is a performance item, not a correctness gate** — wide rounds
  run as small-T slices. It is also the thing that would have to be built to make this artifact
  competitive, and given the 2 % byte finding there is no reason to build it for this model.

**And one thing the plan got right and had to pay for:** the second level really is per expert.
Over layers 0 and 20, `weight_global_scale` takes 96-118 distinct values across the 256 experts
of one projection, spanning 3.3-7.0x. It is applied as a multiply on the finished dot, which is
exact and costs one multiply per expert path.

The checkpoint is 40 layers x 256 experts, `moe_intermediate` 512 over hidden 2,048, all 40
layers MoE, `compressed-tensors` / `nvfp4-pack-quantized`.

---

*The original plan follows. Its Phase 1 is shipped; Phases 3 and 4 are superseded by the notes
above and by the converter that now exists.*


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

**Where the global scale goes — settled against the checkpoint (2026-08-30).** NVFP4 is
two-level: an e4m3 scale per 16 values and a global scale. `RedHatAI/Qwen3.6-35B-A3B-NVFP4`
stores that second level **per expert and per projection** —
`layers.L.mlp.experts.E.gate_proj.weight_global_scale` — so `Weight::weight_scale_divisor`,
a single float for a whole tensor, cannot express it once 256 experts are stacked into one
routed weight.

The fix is smaller than it looks, and it is *not* an interface change to the codec. A divisor
constant over an expert factors out of the dot product:

    sum_k (code_k · blockscale_k · divisor) · x_k  =  divisor · sum_k (code_k · blockscale_k) · x_k

so the codec decodes exactly as the others do and the kernel multiplies the finished dot by
`divisor[expert]` once. That also lands the scale *before* SwiGLU on the gate/up path, which is
where it has to be. What Phase 1 adds is therefore two device arrays on `SparseMoeWeights`
(`[experts]` divisors for gate/up and for down) and one multiply per dot, not a new codec
signature.

## The scale layout: a decision Phase 1 cannot avoid (found 2026-08-30 10:40)

The dense NVFP4 path stores block scales in `QuantLayout::BlockScaleK16M128x4` — swizzled in
tiles of 128 rows by 4 — because that is what its MMA and TMA kernels want, and the test
fixture (`quantized_weight.h`) packs exactly that, refusing N not divisible by 128 and K not
divisible by 64. Every other MoE codec (Q4/Q5/Q6/W8) instead reads scales row-major, indexed
`row * groups_per_row + group`, and `Nvfp4Codec` as written follows that convention.

So the routed experts have to pick one, and the choice is not free either way:

- **Row-major scales for routed experts** (what the codec assumes). The decode kernels stay
  simple and identical in shape to the other codecs, and the expert slot cache keeps copying
  opaque planes. The converter must then write a routed-expert layout that differs from the
  dense one, and a future prefill MMA arm would have to swizzle or re-pack.
- **Reuse `BlockScaleK16M128x4`.** One layout everywhere, and Phase 2's MMA arm inherits the
  dense path's tiling for free. The decode codec must then implement the swizzle in its
  `load_eight`, and the 128-row alignment has to hold *per expert*, which for the 35B's
  `[experts × 2·intermediate]` gate/up stack means checking that each expert's row block starts
  on a 128 boundary (2 · 640 = 1,280 rows per expert, so it does).

**Recommendation: reuse the dense layout.** The alignment works out, it keeps one packing path
in the converter, and it is the only option that does not strand Phase 2. The cost is confined
to `Nvfp4Codec::load_eight`, which has to map (row, group) through the tile swizzle instead of
multiplying — a change to one function, against a converter and a second layout to maintain
forever.

Until that is done, Phase 1 is **groundwork only**: the codec compiles and the packed-loop
generalisation is proven not to disturb Q4/Q5/W8 (the oracle test passes unchanged), but no
kernel instantiates it and the fixture cannot pack a matching input, so it is not yet possible
to test the codec numerically. That test is the next step, and it needs the swizzle first.

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

## The two NVFP4 checkpoint formats (read this before writing Phase 4)

"NVFP4" names a numeric format, not a checkpoint layout, and two ecosystems write it
differently. Our entire NVFP4 path — converter validation and engine — currently understands
exactly one of them.

| | **compressed-tensors** (llm-compressor, RedHat/neuralmagic) | **ModelOpt** (NVIDIA TensorRT) |
|---|---|---|
| config | `quant_method: compressed-tensors`, `format: nvfp4-pack-quantized` | `quant_algo: NVFP4` under a ModelOpt block |
| packed codes | `…weight_packed` | `…weight` |
| block scales (e4m3, per 16) | `…weight_scale` | `…weight_scale` |
| global scale | `…weight_global_scale` | `…weight_scale_2` |
| activation scale | `…input_global_scale` | `…input_scale` |

**What we support.** `qwen3_6_27b/convert_nvfp4.py` asserts `compressed-tensors` and
`nvfp4-pack-quantized` and rejects anything else, and the engine consumes the global scale as a
*divisor* — `alpha = 1 / (input_scale_divisor · weight_scale_divisor)`
(`nvfp4_cublaslt.cpp`, `nvfp4_linear_swiglu_plan.cpp`). That matches compressed-tensors, where
the global scale is `amax`-derived and divides.

**Why it matters here.** `RedHatAI/Qwen3.6-35B-A3B-NVFP4` — the checkpoint Phase 4 reads and
the one vLLM serves for the board's pair — is compressed-tensors, so the 35B recipe can follow
the 27B path. But NVFP4 releases of the same model from other publishers are frequently
ModelOpt, and the two differ in the *direction* of the global scale as well as its name. A
ModelOpt checkpoint fed to a converter that assumes compressed-tensors would either fail a
name lookup (the good case) or silently mis-scale every expert (the bad one).

**So Phase 4 owes two things**, neither large: detect the format from `quantization_config`
rather than from tensor names, and refuse an unrecognised one with a message that says which
format was found. Supporting ModelOpt as a second arm is optional and should not be written
from memory — the multiply-versus-divide convention has to be verified against a real ModelOpt
checkpoint and a numerical round-trip before it is trusted.

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
