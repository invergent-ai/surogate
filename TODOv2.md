# TODOv2 — serving quantized checkpoints

Tracker for one endeavor: serve any quantized checkpoint generically — GGUF
K-quants first, then compressed-tensors NVFP4, FP8 and the rest — without a
per-model converter or a per-export profile. `TODO.md` remains the tracker for
pre-existing review items; nothing moves between the two.

Legend: `[x]` done · `[~]` partly done · `[ ]` open

---

## Status

**A GGUF is served where it lies.** `surogate serve model.gguf` reads the file's
own bytes: no dequantise-and-requantise anywhere on the text path, and no copy
of the weights. What lands beside the file is a small index naming the stretches
of it each object is assembled from.

`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` — 22.13 GB, 34.66 B parameters, 256 experts of
which 8 route — on one 5090, 654-token prompt, 128 generated, warm, greedy:

| | prefill tok/s | decode tok/s |
|---|---:|---:|
| **surogate, native K-quant** | **11,100** | **317** |
| llama.cpp, same file | 8,408 | 278 |
| surogate, dequantised to Q4G64 | 13,700 | 346 |

Ahead of llama.cpp on both. Our own row-split path still leads prefill, which is
the honest open gap — to ourselves, not to a competitor.

| | |
|---|---:|
| the GGUF | 22.13 GB |
| artifact beside it, before this line of work | 22.30 GB (a full copy) |
| **artifact today** | **70 MB** |
| one-time step | 18 s, 0.0 GiB of BF16 staging |

Of the 70 MB: router 42, tokenizer 10, BF16 norms and the draft head's token ids.
Nothing large is computed any more.

Also shipped in this line: the 0.8B/2B dense GGUFs, RedHatAI's
compressed-tensors NVFP4 35B from its own directory (M1), BF16 linears at any
8-aligned shape, text-only GGUF exports of a vision family, and a tied head
stored once.

---

## Roadmap

1. **[ ] Close the native prefill gap** — 11,100 against the row-split path's
   13,700 on the same model. Everything else on this list is smaller. The scale
   arithmetic is *not* the cause (measured — see *Rejected*); the remaining
   named suspects are Q6_K's scalar staging and the tile's `cp_async` shape.
2. **[ ] K6 — retire Q4G64/Q5G64/Q6G64, add `surogate quantize`.** Converters
   stop emitting the home-grown formats; `surogate quantize` writes a BF16 GGUF
   and calls `llama-quantize` (built at `study/llama.cpp-master/build/bin`).
   Gated on 1: retiring them sends every converted checkpoint down the K-quant
   path, so the gap would become a regression.
3. **[ ] N — NVFP4 ModelOpt ingest.** `weight_scale_2` is a multiplier where
   compressed-tensors' global scale is a divisor; parents split per component.
4. **[ ] F — FP8.** compressed-tensors per-channel/per-tensor is per-row with an
   FP32 scale — add `_F32S`, or accept the BF16 cast. HF fine-grained FP8 is
   block-scaled and has no runtime format at all.
5. **[ ] M2 — one directory per architecture, geometry as a template parameter.**
   `qwen3_5_{0_8b,2b,4b}` are ~1,750 lines each for seven integers; `variant.h`
   differs by 2 lines across the three. Nothing requires the split: all 52
   headers in `csrc/src/serve/api/ops/` take runtime shapes.
6. **[ ] M4 — unify weight loading with the trainer.** Serve's `recipe.py` +
   `inventory.py` per target restate what the trainer's `hf_mapping` DSL already
   declares (`fuse`, `split`, `stack_experts`); the trainer's
   `SafeTensorsReader` is the better reader (multi-shard, GDS, strided).
7. **[ ] K5c — fused K-quant GDN projection-and-convolution.** Built, measured,
   left off: costs more in kernel launches than it saves in bandwidth.
8. **[ ] Drift to fix.** `--no-cache` is unimplemented, `surogate convert` does
   not exist, and `surogate/serve/tools/README.md` still tells users to download
   artifacts from Hugging Face — a posture the owner rejected — while linking
   three files that do not exist.

---

## Format coverage

What the serve runtime can route today, per stored format.

| Format | Status |
|---|---|
| GGML Q2_K–Q6_K | **[x]** linear, embedding, MoE (decode, small-T, prefill), read from the GGUF |
| GGML Q8_0 | **[x]** linear, embedding, MoE; the fused projections read it and rearrange at load |
| BF16 | **[x]** any 8-aligned shape (cuBLASLt), registered shapes keep the hand kernels |
| W8G32_F16S | **[x]** the broadest-supported format; what the fused projections consume |
| NVFP4 (compressed-tensors) | **[x]** generic ingest (M1); TRT-LLM cutlass for routed MoE |
| NVFP4 (ModelOpt) | **[ ]** roadmap 3 |
| FP8 | **[~]** only `FP8_E4M3FN_ROW_BF16S`; per-tensor, per-channel and block unsupported |
| W4A16 / W4A16_ASYM | **[ ]** `Q4G64_F16S` is symmetric with no zero point and no actorder |
| MXFP4 / MXFP8 | **[ ]** nothing in serve; the trainer decodes MXFP4 |
| GPTQ / AWQ | **[ ]** off the roadmap by owner decision |
| `kv_cache_scheme` | **[ ]** refuse or support — do not ignore silently |

Known runtime constraints: NVFP4 needs `n % 128 == 0 && k % 64 == 0` with no
padding path; `embedding` has no NVFP4 or Q4/Q5; `linear_pair` is W8 only.

---

## How a GGUF is served

The artifact directory is an *index*, not a container. Four mechanisms, each
added when the previous one ran out; together they take the 35B from a 22.30 GB
copy to 70 MB.

- **`external` + per-object `runs`.** The directory names files it does not
  contain and, per object, the stretches it is assembled from. Runs rather than
  one offset, because the fused objects are not slices: a routed gate/up
  interleaves each expert's gate rows with its up rows, and the GGUF keeps those
  as two tensors, so it is 512 runs. The materializer orders and coalesces reads
  *within* a source, never across one — offsets only order inside a file. Two
  objects may legitimately read the same external bytes (a tied embedding and
  output head), so overlap stays an error only inside the artifact's own payload.
- **Q8_0 as a served format.** Took the embedding table, the attention output and
  the draft head's shortlist gather off the copy.
- **A load `transform`.** `q8_0-to-w8g32` rearranges the file's blocks into the
  row-split planes on the device. Q8_0 and W8G32_F16S hold the same numbers and
  differ only in arrangement, so the weights whose kernels want planes need no
  copy either — and no kernel had to change.
- **Permutation maps.** llama.cpp reorders a GDN projection's V heads. Our
  inverse was applied by the bridge, which made the recipe's row program describe
  rows the file does not hold. As a *row* map it composes into the row program (a
  permuted 128-row head is one run, so the 802 MB fused projection is 64 runs);
  as a *column* map it rides on the transform (`group_map`, one entry per 32
  columns, shared by every row), because a head is 128 columns and a block is 32,
  so the permutation moves whole blocks.

Only a **value** transform now forces the dequantise path — `A_log`'s logarithm,
a plus-one norm's subtraction. That split is the general mechanism; the table of
which tensor is which remains family knowledge (`surogate/serve/gguf/qwen35.py`).

Verified against the artifact the converter wrote for all 150 rearranged objects,
30 of them carrying a column map: **every decoded weight identical**.

---

## Decisions that govern

- **GGUF K-quants are the product** (owner, 2026-09-02). Support natively and at
  the highest performance: GGUF Q\*_K, NVFP4 compressed-tensors and ModelOpt,
  FP8, BF16. GPTQ/AWQ optional and off the roadmap. Where anything else in this
  file disagrees, this wins.
- **`.sinfer` is a transparent cache, never an interchange format** (owner,
  2026-08-24). Never published, never required. The eight-entry hardcoded
  registry in `ingest.py` is the rejected shape.
- **The checkpoint's tensor names are the authority for structure; the config is
  the authority for format.** Established while resolving the RedHatAI 35B:
  transformers represents 256 routed experts as one fused module, so the matcher
  never sees a per-expert `Linear`. Resolution runs over checkpoint-derived
  names, class taken from the meta skeleton where the name exists and from the
  fused module's constituents where it does not — resolved 30,880 = packed
  30,880, zero either way, no heuristic on tensor names.
- **Format is data, read per tensor.** Structure stays compiled in; the binder
  checks shape and "can an op route this format", not "is this the format I was
  compiled for".
- **Dequantise-and-requantise is not free.** Measured on Qwen3.5-0.8B-Q4_K_M:
  K-quant → BF16 → W8 adds 5.7e-3 relative error and *doubles* the bytes. That is
  why the native path exists.

---

## Traps

Each of these cost real time; none is inferable from the code.

- **A format label mismatch is silent when the sizes agree.** `bind_moe`
  discovered each routed tensor's stored format and kept only the handle, so
  `load_moe` passed the *profile's* expectation on. Q4_K superblocks went to the
  groupwise-int row-split codec: same byte count, different meaning, no
  exception, every expert noise, token 0 forever. **When output is degenerate but
  nothing throws, print the qtype the kernel actually received before auditing
  weights.** Hours went into verifying weights that were all correct.
- **"This code path works" can mean "it has never run."** The GDN V-head
  permutation only applies when `num_k_heads != num_v_heads`; every model
  validated before the 35B had them equal. Check the KV that gates a path.
- **Where a scale lives decides a kernel's shape.** The row-split MoE prefill
  feeds the MMA raw integer codes and scales the accumulator afterwards, because
  a 64-value group has one scale. A K-quant's 64-wide tile spans two sub-block
  scales and carries an affine min that depends on the activation sum, so it must
  dequantise inside `decode_weight`. That single difference is the whole K-quant
  prefill kernel.
- **Alignment is a per-format fact.** Q6_K's 210-byte block leaves consecutive
  blocks 2-byte aligned, so `cp_async<16>` is illegal on it and it stages in
  scalar pairs — llama.cpp reads Q6_K through 2-byte accessors for the same
  reason. Q4_K (144) and Q5_K (176) are fine.
- **Compare decoded values, not bytes, when the oracle went through BF16.** Eight
  rows of a GDN projection differ in bytes from the converter's output; their
  scale is zero and the dequantise path had normalised the codes the file
  carries. Every decoded weight is identical — that path was the lossy one.
- **Fewer bytes is not faster.** K5c removes the last lossy bytes from the GDN
  input projection and loses 6 % of decode (885 → 828 on the 0.8B), because the
  fused parent runs one tuned kernel per layer and the split runs two GEMVs plus
  an unfused convolution.
- **The `.inc` bodies are instantiated per geometry.** Guard a new kernel's
  dispatch with `if constexpr` on the geometry rather than `static_assert`, or
  geometries that will never run it fail to compile.
- **`plan()` skipped anything with a "native" source**, which silently orphaned
  Q8_0 objects the moment Q8_0 became native. Guard on 256-value superblocks.
- **Read the harness's criterion before chasing a kernel.** The A4 op oracle does
  not model activation quantisation; its allowance is the format's own headroom.

---

## Measured and rejected

Written down so they are not retried.

- **Building the index in memory to drop the file.** The one-time step is 18 s
  and 70 MB, and the bridge stages 0.0 GiB of BF16. Removing the file means
  porting the recipes, the row algebra, the name mapping and the permutation
  tables to C++ — a large change to save eighteen seconds once and a file
  smaller than the tokenizer's own vocabulary.
- **Broadcasting the K-quant sub-block scale instead of unpacking it per lane.**
  One lane computes the 6-bit scale and shuffles it to the other fifteen:
  **slower**, 9,700–10,300 against 11,100. The two `__shfl_sync` calls cost more
  in the inner loop than the unpack they remove.
- **llama.cpp's small-K mmvq schedule.** 862 → 847 here; disabled with the
  measurement beside the condition.
- **K5c's split of a mixed-format fused parent.** Above.
- **Porting MMQ for prefill (K3).** The wide route dequantises once and uses BF16
  tensor cores instead: ~200 lines against ~6,000, more accurate, and our MoE
  prefill kernel already beats llama.cpp's MMQ by 1.65×. Worth revisiting only
  where the dequantisation tile is the constraint.
- **Q8_0 codecs in the fused-projection kernels.** They cp_async 16-byte chunks
  out of separate code/scale planes, so Q8_0's 34-byte blocks mean six kernel
  rewrites. The load transform gets the same result and touches no kernel.

---

## Done

- **[x] M1 — generic compressed-tensors NVFP4.** RedHatAI's 35B-A3B converts from
  its own directory; the config is the authority, resolved with the library's own
  matcher.
- **[x] M3 — generic GGUF.** Per-tensor GGML type → runtime format, from the
  header, never a file-level "quant type".
- **[x] K0–K5 — the K-quant line.** The bar, the native read path, Q8_0 (served,
  not merely decided), the wide-batch route, the routed MoE decode *and* prefill,
  and the three `QType` switches. K5's fused projections never needed a codec:
  their weights are read from the GGUF and rearranged at load. Six tuned kernels
  not written.
- **[x] Direct GGUF loading.** 22.30 GB → 70 MB; see *How a GGUF is served*.
- **[x] BF16 at any 8-aligned shape.**

Commits, most recent first: `6d045ef7` `3becc963` `8374dfc5` `13c05055`
`996e4166` `4bc55762` `3e94fa60` `0e4a1576` `d52a5e62` `db4cc18e` `7bebf54e`
`0624d4b6` `300f9d26` `1455c3b0` `700b9a23` `9995f894` `48a25753` `1b33158b`
(2026-09-03), and `7aea807e` `7de24bab` `2432825e` `478ab1ae` `ca656302`
(2026-09-02).

Board rows: `surogate/serve/BENCHMARKS.md`.
