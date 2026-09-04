# TODOv2 — serving quantized checkpoints

Tracker for one endeavor: serve any quantized checkpoint generically — GGUF
K-quants first, then compressed-tensors NVFP4, FP8 and the rest — without a
per-model converter or a per-export profile. `TODO.md` remains the tracker for
pre-existing review items; nothing moves between the two.

Legend: `[x]` done · `[~]` partly done · `[ ]` open

---

## Status

**A GGUF is served where it lies.** `surogate serve model.gguf` reads the file's
own bytes: no dequantise-and-requantise on the text path and no copy of the
weights. What lands beside it is a small index.

**One target per architecture, serving every size of its family.**
`csrc/src/serve/targets/` holds gemma3, llama, qwen3, qwen3_5, qwen3_5_moe and
qwen4exp. Qwen3, Llama, Gemma 3 and Qwen3.5 take any size of their family; the
rest are the sizes their loaders are still written around. Qwen3.5, 3.6 and 3.8
are one architecture and share one target, which is what their own checkpoints
say: every one declares `model_type: qwen3_5`.

`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` — 22.13 GB, 34.66 B parameters, 256 experts of
which 8 route — on one 5090, single stream, ~700-token prompt, 128 generated,
warm, greedy (2026-09-03, both native rows the same session):

| | prefill tok/s | decode tok/s |
|---|---:|---:|
| **surogate, native K-quant** | **13,195** | **315** |
| llama.cpp, same file | 8,408 | 278 |
| surogate, dequantised to Q4G64 | 13,700 | 346 |

Ahead of llama.cpp on both, at the accuracy llama-perplexity measures:

| wikitext-2 test, 145 windows of 2048, second halves scored | PPL |
|---|---:|
| llama.cpp, same file | 6.2311 ± 0.040 |
| **surogate** | **6.2370 ± 0.040** |

| | |
|---|---:|
| the GGUF | 22.13 GB |
| artifact beside it, before this line of work | 22.30 GB (a full copy) |
| **artifact today** | **70 MB** |
| one-time step | 18 s |

The harness is `surogate/serve/tools/eval/perplexity.py`.

---

## Roadmap

Six of eleven are closed (1, 3, 6, 8, 9, and 5's first size). Two are decisions
waiting on `surogate quantize` as a product rather than tasks (2 and 11). The
rest are open, each with what an attempt needs written down: FP8 (4), the two
smaller NVFP4 sizes (5), the trainer/serve mapping duplication (7), and the
Q6_K down kernel (10).

1. **[x] Native K-quant prefill (2026-09-03).** 13,195 tok/s against the
   BF16-activation path's 10,299 on a ~700-token prompt, 17,300 against 14,300
   at 8k. Perplexity 6.2370 against llama.cpp's 6.2311 (± 0.040), which was the
   gate. Decode unchanged. `SUROGATE_SERVE_MOE_INT8=0` reverts.
2. **[~] K6 — retire Q4G64/Q5G64/Q6G64.** The three home-grown formats and the
   converters that produce them would leave together, roughly 140 references.
   The old argument against it is stale: with the int8 route the K-quant path
   measures 13,195 tok/s against the row-split path's 13,700, and decode is
   unaffected.
   **What the deletion strands, checked 2026-09-03.** One live path still
   produces these formats: the 27B's `Qwen36GroupwiseInt` profile, whose
   endpoints bind `Q6G64_F16S` and whose layers bind `Q4G64_F16S`. Nothing else
   selects them — the other targets import the names and use `W8G32_F16S` or
   NVFP4. So this is one target's safetensors profile, not five, and it is a
   decision rather than a task: it costs the 27B its groupwise-int route until
   `surogate quantize` (item 11) is a product.
3. **[x] GGUF coverage beyond the Qwen3.5/3.6 families (2026-09-03).** Qwen3,
   Llama and Gemma 3 serve from a GGUF at any size of the family, on the file's
   own metadata: no vendored config, nothing to register per size. TinyLlama
   matches llama.cpp word for word on the same file, and both families encode
   3,104 corpus strings identically to their official tokenizers. SentencePiece
   vocabularies reconstruct.
   A GGUF published without a chat template is refused, which is what the base
   `google.gemma-3-270m` files are.
4. **[ ] F — FP8, and the two kinds are not the same job (checked 2026-09-03).**
   - *compressed-tensors per-channel/per-tensor* is per-row with an FP32 scale.
     The engine has `FP8_E4M3FN_ROW_BF16S`, so this is either an `_F32S` variant
     or a documented BF16 cast of the row scales. Small — but **no checkpoint of
     this kind is on this machine**, so it cannot be written against anything.
   - *HF fine-grained FP8* is block-scaled, and that is what the three local FP8
     checkpoints are: `models--surogate--Qwen3.5-{0.8B,2B,4B}-FP8` declare
     `quant_method: fp8`, `weight_block_size: [128, 128]`, and carry
     `weight_scale_inv` as a 2-D F32 grid (e.g. `[48, 8]` for a `[6144, 1024]`
     projection). No runtime format holds a 2-D block scale, so this is a new
     weight format and GEMM support, not an ingest change.
5. **[~] N — NVFP4 ModelOpt ingest: the 4B serves (2026-09-03).**
   `surogate serve <Qwen3.5-4B-NVFP4>` works end to end. **Left: the 0.8B and
   the 2B have no NVFP4 recipe** — their checkpoints are on this machine and are
   refused by name until one exists.
   What such a recipe reads: `weight` `[n, k/2]` U8, `weight_scale` `[n, k/16]`
   E4M3, `weight_scale_2` a scalar F32, `input_scale` a scalar F32, per
   component. Two differences from the compressed-tensors path the 27B converter
   already handles:
   - **The global scale is a multiplier, not a divisor.** The engine binds
     `weight_scale_divisor` / `input_scale_divisor` and validates them positive
     and finite, so ingest inverts: `divisor = 1 / weight_scale_2`. Measured on
     layer 11: `weight_scale_2 = 7.30242e-05`, `input_scale = 8.97507e-03`, and
     q/k/v share both while `o_proj` has its own — so the divisor groups the
     recipe already models are the right shape, they are just per component.
   - **Parents are split per component.** ModelOpt writes `q_proj`, `k_proj`,
     `v_proj` separately where the artifact fuses them, so the converter fuses
     and must check the three share a divisor before it does.
   The engine side is ready for any size: `qwen3_5` compiles both NVFP4 profiles
   and binds NVFP4 parents with both divisors.
6. **[x] M2 — one directory per architecture (2026-09-04).**
   `csrc/src/serve/targets/` holds six directories — gemma3, llama, qwen3,
   qwen3_5, qwen3_5_moe, qwen4exp — and none names a size or a generation. A
   checkpoint states its dimensions in the index built beside it and the engine
   binds against those, so a target serves every size and generation of its
   family: Qwen3-1.7B runs on the code compiled for Qwen3-0.6B, and Qwen3.5,
   Qwen3.6 and Qwen3.8 share one target where five directories stood. The
   converter, the reference implementation and the vendored resources are keyed
   the same way. Two dimensions stay compiled in the hybrid forward interface,
   marked where they are: the attention head width and the GDN output-gate
   width.
7. **[ ] M4 — unify weight loading with the trainer, still true but smaller
   than it was (checked 2026-09-03).** Serve's `recipe.py` + `inventory.py` per
   target restate what the trainer's declarations in `surogate/dsl/models/`
   already say. Both sides describe the same nineteen architectures: the DSL has
   `qwen3.py`, `qwen3_5.py`, `llama.py`, `gemma3.py` and fifteen more, and serve
   has a recipe per converter — 4,468 lines of them.
   The remaining duplication is smaller than it was. The recipes no longer
   restate *dimensions*, because the artifact carries them and the binder
   validates against them; what they still restate is the *mapping* — which
   checkpoint tensor becomes which artifact object, and how fused objects are
   assembled. That is the part worth unifying, and the part `hf_mapping` already
   spells out.
8. **[x] K5c — fused K-quant GDN projection-and-convolution: closed, negative
   (2026-09-03).** Measured slower; not in the tree. Reopening it needs a reason
   the measurement did not have.
9. **[x] Drift to fix (2026-09-03).** A K-quant draft block binds and serves,
   proved on our own export of Qwen3.5-0.8B carrying 12 MTP objects across BF16,
   Q4_K, Q6_K and W8G32_F16S. `--no-cache` rebuilds the index instead of reusing
   one. The converter and the artifact container live in the serving path
   (`serve/convert/`, `serve/artifact/`) rather than in `serve/tools/`, whose
   README is accurate again. `surogate convert` does not exist and is not
   wanted: `surogate serve` converts.
   **One gap found and not closed (2026-09-04):** a text-only GGUF export of the
   27B vision family is refused, because its converter expects the vision
   tensors the export drops (`KeyError: model.visual.patch_embed.proj.weight` on
   unsloth's `Qwen3.8-27B-UD-Q4_K_M.gguf`). Same class of thing item 3 fixed for
   the other families, and it predates the target renames: the board's
   Qwen3.8-27B rows are all NVFP4 from safetensors, so a 27B-class GGUF has
   never been served.
10. **[ ] Q6_K down. Measured and root-caused 2026-09-03; the fix is written
   down here and not built.** The op benchmark now carries the native codecs
   (`--codec q4_k-q4_k`, `--codec q4_k-q6_k`). One 5090, 256 unique experts,
   warm, median of three:

   | tokens | q4_k-q4_k | q4_k-q6_k |
   |---|---:|---:|
   | 128 | 518 us | 999 us |
   | 512 | 655 us | 1,346 us |
   | 1024 | 764 us | 1,178 us |

   Q6_K down roughly **doubles the whole MoE body**, and reproduces in seconds
   rather than needing a 22 GB model. The kernel runs at 21.9 % of peak
   bandwidth where the Q4_K one reaches 38.9 %.
   **Why, exactly.** A Q6_K block is 210 bytes, so block `b` starts at a
   16-byte misalignment of `2b mod 16`, cycling with period eight. `cp_async`
   needs 4-, 8- or 16-byte alignment, so this codec sets `kCpAsync = false` and
   stages its 96-byte tile as **48 two-byte scalar loads** where every other
   codec issues 6 sixteen-byte async copies. That is the whole gap; the two
   `m16n8k16` MMAs the sixteen-wide scales force are the smaller half.
   **The fix that follows from it.** Within a block the chunks are 16 bytes
   apart, so a tile's misalignment `off` is constant across its chunks. Stage
   seven aligned 16-byte chunks from `src & ~15` instead of six from `src`, and
   the tile's bytes land at shared offset `off`; the shared tile stride is
   already 112 bytes for the int8 route, so it fits. The consumer then reads
   `__funnelshift_r(w[0], w[1], 8 * ((off + byte) & 3))` over two aligned words
   instead of one unaligned one — twice the shared traffic to remove seven
   eighths of the global staging. `off` varies per row, so the shift is
   per-thread and must stay branchless.
   The layer-level reading is the same effect measured the other way: 688 us
   against the row-split kernel's 337, on `routed_down` of 3 of 40 layers.
11. **[~] DEFERRED, off the critical path — `surogate quantize`, the export of a
   model we trained.** Revisit once the serving engine is complete (owner,
   2026-09-03). The thin version is in (`surogate/cli/quantize.py`) because it
   turned out to be two subprocess calls; everything a real product needs
   around it is not, and is listed below.

---

## `surogate quantize`, as investigated (2026-09-03)

A downloaded model is already a GGUF and is served where it lies. A model
trained here has none, so producing one is ours to do: `surogate sft` →
`surogate merge` → **this step** → `surogate serve model.gguf`.

**Why it cannot be our own arithmetic, or Python.** Every published GGUF was
made with llama.cpp's encoders, and they exist nowhere else: the `gguf` package
writes Q8_0 and raises `NotImplementedError` for Q4_K, Q5_K and Q6_K. Writing
our own K-quant encoder would mean matching `make_qkx2_quants` and the
per-tensor type mixture in `llama-quant.cpp` (~1,500 lines over
`ggml-quants.c`'s 5,667) closely enough that our output is not quietly worse
than the file a user could have downloaded instead.

**What unsloth does, since the question was whether to follow it.**
`study/unsloth-zoo/unsloth_zoo/llama_cpp.py` is ~3,500 lines and implements no
quantisation at all. It locates, downloads or builds llama.cpp, then runs one
command in `quantize_gguf`:

    llama-quantize [--imatrix f] [--tensor-type pat=TYPE] in.gguf out.gguf q4_k_m <threads>

The type string reaches the shell straight from the user's
`quantization_method`, validated only as a bare token. Their own presets are
not formats: `Q2_K_L` is `q2_k` plus `--tensor-type .ffn_down_exps=Q3_K`,
`--output-tensor-type Q6_K`, `--token-embedding-type Q4_K`. They build five
llama.cpp targets, of which `llama-quantize` is the one that matters, and it is
CPU-only — which is why they can ship prebuilt CPU archives for it.

**A K-quant takes two passes, not one.** llama.cpp's converter reads the
Hugging Face checkpoint and writes a BF16 GGUF, because it holds the
per-architecture tensor mapping and the tokenizer; `llama-quantize` then reads
that and applies the mixture. The converter is now a `conversion/` package,
~21,000 lines with a module per architecture, and it registers
`Qwen3_5MoeForConditionalGeneration` — the architecture of the 35B we serve and
train — alongside Qwen3, Qwen3Moe and Gemma3. `llama-quantize` builds clean in
the vendored tree with one `cmake --build build --target llama-quantize`.

**Measured end to end.** `Qwen3-0.6B`: 311 tensors to a 1.51 GB BF16 GGUF, then
Q4_K_M at 456 MiB (5.09 bits a weight) in 5.1 s of quantiser time. Qwen3.5-0.8B:
1.56 GB BF16 → 265 MB Q4_K_M. Both conversions and both quantisations are clean.

**Serving our own export fails, and the blocker is on the engine's side.** The
0.8B export loads to `tensor descriptor does not match target contract:
mtp/input_projection`. The cause is not the exporter: our conversion keeps the
MTP block (`blk.24.nextn.eh_proj.weight` and friends) and llama.cpp's `q4_k_m`
mixture quantised it to Q4_K, while every MTP binding in the target contract
demands `W8G32_F16S` or BF16 — eleven bindings across the 0.8B and 2B targets,
none of which accepts a K-quant. It has never shown up because the published
0.8B GGUFs we validated against strip the nextn block entirely, so ingest takes
the no-MTP variant and the bindings are never exercised. Two ways out, and the
first is one flag: pin the MTP tensors at quantise time
(`--tensor-type "nextn=q8_0"` and the rest of `blk.<mtp>.`, which is exactly the
mechanism behind unsloth's presets), or teach the binder to read a K-quant MTP
block. The second is the real fix and belongs to the engine, not to this item.

**What is deliberately not built, and is the actual work when this comes back:**

- **No importance matrix.** `--imatrix` is passed through, but nothing produces
  one, and the IQ types require it. Generating one means running the model over
  a calibration corpus (llama.cpp's `llama-imatrix`), which is a training-side
  job, not a two-subprocess one.
- **No mixture of our own.** We accept llama.cpp's `q4_k_m` mixture as given.
  Whether a Surogate preset should exist — the "UD" mixes are exactly this — is
  a quality question that wants perplexity evidence per candidate, which the
  gate in `tools/eval/perplexity.py` can now supply.
- **The llama.cpp dependency is a checkout, not a dependency.** It resolves to
  `study/llama.cpp-master`, which is a study tree and not shippable. A product
  version either vendors the quantiser sources into `csrc` or fetches a pinned
  release the way unsloth does.
- **No MoE-specific handling, no vision towers, no sharded output**
  (`--keep-split`), and no LoRA-adapter GGUF path (`convert_lora_to_gguf.py`
  exists upstream).
- **Untested beyond the 0.8B**, and untested on anything we trained ourselves.
- **No tensor pinning.** The MTP failure above needs it, and so would any other
  contract that wants a particular format for a particular tensor.

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
| NVFP4 (ModelOpt) | **[~]** the 4B serves; the 0.8B and 2B need a recipe (roadmap 5) |
| FP8 | **[~]** only `FP8_E4M3FN_ROW_BF16S`; per-tensor, per-channel and block unsupported |
| W4A16 / W4A16_ASYM | **[ ]** `Q4G64_F16S` is symmetric with no zero point and no actorder |
| MXFP4 / MXFP8 | **[ ]** nothing in serve; the trainer decodes MXFP4 |
| GPTQ / AWQ | **[ ]** off the roadmap by owner decision |
| `kv_cache_scheme` | **[ ]** refuse or support — do not ignore silently |

Known runtime constraints: NVFP4 needs `n % 128 == 0 && k % 64 == 0` with no
padding path; `embedding` has no NVFP4 or Q4/Q5; `linear_pair` is W8 only.

---

## What an index can hold

The artifact directory is an *index*, not a container: it names the GGUF and,
per object, the byte runs it is assembled from. Four things a future object can
declare, in the order they were needed:

- **`external` + per-object `runs`** — the file it does not contain, and the
  stretches of it each object takes. Runs rather than one offset, because a
  fused object is not a slice.
- **A GGML block format served directly**, so nothing is copied to satisfy a
  kernel that reads planes.
- **A load `transform`** — a rearrangement applied on the device at load, for a
  kernel that wants a different arrangement of the same numbers.
- **A permutation map** — a row map composes into the run program, a column map
  rides on the transform.

Only a **value** transform forces the dequantise path: `A_log`'s logarithm, a
plus-one norm's subtraction. Which tensor needs which is family knowledge
(`surogate/serve/gguf/qwen35.py`).

---

## Decisions that govern

- **GGUF K-quants are the product** (owner, 2026-09-02). Support natively and at
  the highest performance: GGUF Q\*_K, NVFP4 compressed-tensors and ModelOpt,
  FP8, BF16. GPTQ/AWQ optional and off the roadmap. Where anything else in this
  file disagrees, this wins.
- **We do not quantise what a user brings us; we do quantise what we train**
  (owner, 2026-09-03). A checkpoint someone hands the engine is served in the
  format it arrives in, and nothing in the serving path re-encodes weights. But
  a model trained here has no published GGUF, so producing one is our job:
  `surogate quantize` stays, it takes a trained checkpoint to a GGUF, and the
  quantisation arithmetic is llama.cpp's rather than ours. **It is a separate
  product and not on the critical path** (owner, 2026-09-03): the serving engine
  comes first, and the export command is revisited after. See item 11 for what
  exists and what does not.
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

- **A gap is a bug only once the measurement condition matches the reference.**
  The first perplexity probe read ~600 against llama.cpp's 6.2 and half a day
  went into kernels, norms, the head and per-layer lenses. The engine was
  right: the probe scored Wikipedia prose inside a *user turn* of a thinking
  model, which predicts `<|im_end|>` for most user-text positions, while
  llama-perplexity scores raw text and does not even parse special tokens. The
  original BF16 model on CPU (HF transformers, `causal_conv1d` and `fla`
  blocked so it falls back to torch) reproduced our per-position NLL to two
  decimals. Hence `SUROGATE_SERVE_RAW_PROMPT`.
- **`ops::sample` scores rows below `kTokenDomain` (248,077), not the head's
  248,320.** Anything that reads logits itself must take the same domain.
- **A greedy prompt under 47 tokens never reaches the MoE prefill kernels.**
  Coherent short answers prove nothing about them; score a long prompt.

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
- **A pipelined kernel's critical path is max(compute, memory), not the sum.**
  Eleven staging variants of the BF16-activation K-quant prefill kernel were
  measured against a reading that turned out to be an artefact: cutting the
  decode out made the kernel 265 µs faster and made header staging look like a
  198 µs cost, so the next several attempts chased staging. Making the header
  fully resident *with the real decode* came out at 780,788 ns against the real
  kernel's 758,000 — no faster at all. Compute was the bound the whole time and
  the staging hid beneath it; removing the decode had exposed the memory path,
  and only then did the header show. Cut a kernel apart to find its bound, but
  read every cut against the whole, never against another cut.
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
- **Folding a lane's two code bytes into one 16-bit load.** GGML's nibble layout
  puts a lane's pair in two different bytes; reading them as one `uint16` is
  exactly neutral. The compiler was already coalescing them.
- **Giving the Q5_K down kernel a third block per SM** by moving `qh` out of the
  staged header and reading it from the file. It works — 34 KB to 30 KB, two
  blocks to three — and the kernel time does not move (507,962 → 507,498 ns).
  That is the proof these kernels are not occupancy-bound; it was reverted
  because global `qh` then cost Q5_K ~7 µs.
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

- 2026-09-04 — **the engine binds on the device it was asked for**. The
  executor's worker thread never called `cudaSetDevice`, so every kernel
  launched on device 0 while the weights sat on the requested one. `--device 0`
  worked by coincidence and every other device failed in warmup.
- 2026-09-04 — **one directory per architecture, end to end**. Qwen3.5, 3.6 and
  3.8 are one architecture, so they are one target, one converter, one Python
  reference and one vendored config; the MoE sibling keeps its own. Five
  directories became two on each of those four axes. Quantisation is a
  parameter, not a directory: one object contract with a table of five export
  profiles, named as the engine names them. The 27B-class binder stopped
  spelling its size out inline (95 occurrences of 5120 in one, 84 of 2048 in
  the other).
- 2026-09-03 — **geometry as data**: the artifact declares its dimensions, the
  binder validates against them, and one target serves every size of its
  family (Qwen3-1.7B on the 0.6B's code; three Qwen3.5 directories into one,
  3,932 lines deleted). A ModelOpt NVFP4 4B serves. A K-quant draft block
  binds. `--no-cache` rebuilds. The pad-token literal is gone. The converter
  and the artifact container left `serve/tools/`. The MoE benchmark measures
  the native K-quant codecs.
- 2026-09-03 — int8 tensor-core route for K-quant routed experts (gate/up and
  down, Q4_K/Q5_K/Q6_K), on by default; `quantize_q8_1_planes`; the NLL probe
  and `ops::next_token_nll`; `tools/eval/perplexity.py`; accuracy gate passed.

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

Commits are on `serve-engine`; `git log --oneline c5d327fe..` is the line of
work this file tracks.

Board rows: `surogate/serve/BENCHMARKS.md`.
