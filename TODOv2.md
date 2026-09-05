# TODOv2 — serving quantized checkpoints

Tracker for one endeavor: serve any quantized checkpoint generically — GGUF
K-quants first, then compressed-tensors NVFP4, FP8 and the rest — without a
per-model converter or a per-export profile. `TODO.md` remains the tracker for
pre-existing review items; nothing moves between the two.

Legend: `[~]` partly done · `[ ]` open. Completed items are not kept here —
what shipped is in `git log --oneline c5d327fe..` and `design/INFERENCE.md`.

---

## Status

**A GGUF is served where it lies**, at any size of a supported family, ahead of
llama.cpp on the formats the board covers. `csrc/src/serve/targets/` holds six
architectures — gemma3, llama, qwen3, qwen3_5, qwen3_5_moe, qwen4exp — and none
names a size: the artifact declares its dimensions and the binder validates
against them.

"Where it lies" is now literal for every GGML type: the artifact beside
`Qwen3.5-0.8B-UD-Q8_K_XL` is 18 MB against the file's 1.19 GB, and nothing in
the serving path re-encodes a weight.

**The suite runs.** `make serve-check` is the command a change to `csrc/src/serve` or
`surogate/serve` has to pass: 107 C++ tests (~3 min on one GPU) and 148 Python tests
(~20 s, no GPU, no checkpoint). The Python half runs in CI on every push. Until
2026-09-05 nothing ran either, the C++ tests were excluded from `all` and built by
nothing, and one had been failing to compile against a rename for long enough that
nobody remembered it.

**What is left is not the engine.** One roadmap item remains and it is a product
decision. The backlog that matters now is architectures: the trainer declares
seventeen and the engine serves six, so `deepseek_v4`, `gemma4`, `glm5_next`,
`gpt_oss`, `laguna`, `lfm2`, `lfm2_moe`, `lfm2_vl`, `nemotron_h`, `qwen3_moe` and
`qwen3_vl` have no serve target. A new dense family costs roughly 950 lines of C++
across seven files and 1,000 of Python, and that was measured rather than guessed:
requiring a block to be identical across the existing targets, only ~86 lines are
shared, so what a family costs is its own specifics, not boilerplate waiting to be
extracted.

Board rows are `surogate/serve/BENCHMARKS.md`; the history of what was tried is
`design/INFERENCE.md`.

---

## Roadmap

One open, and it is a decision about product scope rather than blocked work.
Nothing here is waiting on someone to find time.

1. **[ ] `surogate quantize` as a product — the engine blocker is gone (2026-09-05).**
   The recorded blocker was ours: every MTP binding demanded `W8G32_F16S` or BF16, so
   an export that quantised its own nextn block was refused at
   `mtp/input_projection`. The MTP matrices bind at whatever format the artifact
   declares now, and the whole chain runs: `surogate quantize --type q4_k_m` on
   Qwen3.5-0.8B writes 335 tensors carrying the nextn block at Q4_K, the engine
   serves it, and it scores **14.9559** against llama-perplexity's 14.9713 on the
   same file — better than the *published* Q4_K_M of the same model (15.031/15.025),
   which is what a full-precision source and llama.cpp's own mixture buy.
   So nothing here is engine work any more, and the toolchain is no longer a
   checkout on somebody's disk: llama.cpp is fetched at a pinned commit and built
   into the wheel. What is left is product scope, listed below — no importance
   matrix, no mixture of our own, untested beyond the 0.8B and on anything we
   trained ourselves. **This is a decision about what to build, not a blocked task.**

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
train — alongside Qwen3, Qwen3Moe and Gemma3. Both halves are fetched at a pinned
commit and built by our own CMake (`csrc/cmake/llama_cpp_quantizer.cmake`).

**Measured end to end.** `Qwen3-0.6B`: 311 tensors to a 1.51 GB BF16 GGUF, then
Q4_K_M at 456 MiB (5.09 bits a weight) in 5.1 s of quantiser time. Qwen3.5-0.8B:
1.56 GB BF16 → 265 MB Q4_K_M. Both conversions and both quantisations are clean.

**Serving our own export worked, once the engine stopped refusing it (2026-09-05).**
It used to fail at `tensor descriptor does not match target contract:
mtp/input_projection`: our conversion keeps the MTP block
(`blk.24.nextn.eh_proj.weight` and friends) and llama.cpp's `q4_k_m` mixture
quantised it to Q4_K, while every MTP binding demanded `W8G32_F16S` or BF16. It
had never shown up because the published 0.8B GGUFs strip the nextn block, so
ingest took the no-MTP variant and those bindings were never exercised. The right
fix was the engine's and it is in: `MtpPlan` binds its five matrices at whatever
format the artifact declares, because the kernels dispatch on the weight's qtype
and demanding one format refused a file for no reason they had.
**Measured end to end (2026-09-05):** `surogate quantize --type q4_k_m` on
Qwen3.5-0.8B writes 335 tensors — 168 Q4_K, 27 Q6_K, 140 F32 — with the nextn
block at Q4_K; the engine serves it at 798 tok/s and scores **14.9559** against
llama-perplexity's 14.9713 on the same file, and against the published Q4_K_M's
15.031. A full-precision source and llama.cpp's own mixture beat the download.

**What is deliberately not built, and is the actual work when this comes back:**

- **No importance matrix.** `--imatrix` is passed through, but nothing produces
  one, and the IQ types require it. Generating one means running the model over
  a calibration corpus (llama.cpp's `llama-imatrix`), which is a training-side
  job, not a two-subprocess one.
- **No mixture of our own.** We accept llama.cpp's `q4_k_m` mixture as given.
  Whether a Surogate preset should exist — the "UD" mixes are exactly this — is
  a quality question that wants perplexity evidence per candidate, which the
  gate in `tools/eval/perplexity.py` can now supply.
- **No MoE-specific handling, no vision towers, no sharded output**
  (`--keep-split`), and no LoRA-adapter GGUF path (`convert_lora_to_gguf.py`
  exists upstream).
- **Untested beyond the 0.8B**, and untested on anything we trained ourselves.
- **No tensor pinning.** The MTP failure above needs it, and so would any other
  contract that wants a particular format for a particular tensor.

---

## Format coverage

Routed today, everywhere the runtime needs them: **every GGML weight type llama.cpp
stores** -- Q2_K–Q6_K, Q8_0, Q4_0/Q4_1/Q5_0/Q5_1, IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/
IQ3_XXS/IQ3_S/IQ4_NL/IQ4_XS, TQ1_0/TQ2_0, MXFP4, NVFP4_GGML, Q1_0/Q2_0, F16 -- on linear,
embedding and MoE decode/small-T/prefill (Q4_K/Q5_K/Q6_K on the int8 tensor-core
route, the rest on the BF16 route), read from the GGUF; **BF16** at any 8-aligned
shape, **W8G32_F16S**, and **NVFP4 compressed-tensors** (TRT-LLM cutlass for routed
MoE); **NVFP4 ModelOpt** at every published size; **FP8** per-row, per-channel and
[128,128] block; and a declared `kv_cache_scheme`, honoured where the engine's `auto`
already satisfies it and refused, naming the flag, where it does not.

What is not:

| Format | Status |
|---|---|
| FP8 per-*tensor* | one scalar scale for a whole weight; the route takes a scale grid, so this is its degenerate cell |
| W4A16 / W4A16_ASYM | `Q4G64_F16S` is symmetric with no zero point and no actorder |
| MXFP4 / MXFP8 (compressed-tensors) | nothing in serve; the GGUF MXFP4 block type is read, and the trainer decodes MXFP4 |
| GPTQ / AWQ | off the roadmap by owner decision |
| F32 (GGUF) | bridged, and correctly so: every F32 tensor in a real file is a norm, an `A_log` or a conv1d, each of which needs a value or shape transform. 0.07 % of the elements. |

Known runtime constraints: NVFP4 needs `n % 128 == 0 && k % 64 == 0` with no
padding path; `embedding` has no NVFP4 or Q4/Q5; `linear_pair` is W8 only.

---

## What an index can hold

The artifact directory is an *index*, not a container: it names the GGUF and,
per object, the byte runs it is assembled from. Five things an object can
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
- **Typed row `segments`** — one object whose rows come in more than one format,
  which is what a UD mixture makes of a fused parent (q Q5_K beside k Q4_K, or
  Q8_0 rows beside F16 ones). A component never straddles a type run, so a row
  range resolves to its segment and the fused ops project into it unchanged.

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
  comes first, and the export command is revisited after. See item 1 for what
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
- **The KV cache default is `auto` (2026-09-04): BF16 where every layer is
  attention, e4m3 where linear-attention layers carry the stack.** e4m3 halves
  the cache and costs a 3:1 GDN stack 0-0.4 % of perplexity; it costs a
  pure-attention model 1.5-2.6 %, which is the whole of the `qwen3` target's
  offset against llama.cpp's f16 cache. `--kv-cache-dtype bf16|fp8` pins either.

- **NVFP4 ModelOpt exports of every size serve (2026-09-04), and the small ones
  are worth what their checkpoints are worth.** The uniform export builds its
  recipes from the checkpoint's `config.json` and tensor-name root (the 4B under
  `model.`, the VL-style 0.8B/2B under `model.language_model.`), and the NVFP4
  generic route admits every K its activation quantiser is built for -- one list
  (`SINFER_NVFP4_FOR_EACH_ACTIVATION_K`) that the predicate and the launch switch
  both expand -- so a new width is one entry. The export writers had also been
  unreachable since the recipes moved into `exports/` (a stale import), which
  had broken the 4B's conversion too. Perplexity against llama.cpp's 4-bit
  GGUFs on the same windows: 0.8B **17.57 vs 15.15**, 2B **11.64 vs 10.29**, 4B
  **8.97 vs 8.24**. That is the checkpoints, not the engine: the 0.8B's NVFP4
  weights dequantised exactly inside the BF16 transformers model score 17.33
  with exact activations. ModelOpt's per-16 E2M1 with a calibrated static
  activation scale is simply a worse 4-bit quantisation of these models than a
  K-quant; the engine is faithful to it. Decode on one 5090: 0.8B 481, 2B 380,
  4B 354 tok/s.

- **FP8 checkpoints of both kinds serve natively (2026-09-04).** Hugging Face
  fine-grained FP8 (`weight_scale_inv` over [128,128] blocks) is
  `FP8_E4M3FN_BLK128_F32S`; compressed-tensors per-channel FP8 (`weight_scale`
  per row, dynamic per-token activations) is `FP8_E4M3FN_ROW_F32S`. One route
  serves both -- the scale grid's cell, [k per scale, rows per scale], is a
  runtime parameter of the same e4m3 tile and GEMV -- because this cuBLASLt
  admits only scalar FP8 scales on sm_120 and a K-varying scale cannot be
  applied after a GEMM. Activations are quantised per token per 128 for both.
  Export profiles `fp8-block` / `fp8-channel`, detected from
  `quantization_config`; a text-only release's missing pixel-processor configs
  are optional resources now, on both sides. Qwen3.5-0.8B: block **14.74**,
  per-channel **15.11** (`mahadev9/Qwen3.5-0.8B-fp8`, torch reference over the
  same weights 15.01), against the BF16 model's 14.60 and llama.cpp's IQ4_XS
  15.15; 2B block 10.12 vs Q4_K_M 10.29, 4B block 8.15 vs 8.24.

- **Q4G64/Q5G64/Q6G64 stay** (measured 2026-09-05). Removing them was never the
  objective; performance is, and a GGUF is already served in place rather than
  quantised at runtime — so the only question was whether anything still needs
  them. It does: at 0.8B/2B they are the **vision tower's storage and nothing
  else** (the text stack there is W8 throughout), and at 27B they are the text
  stack of a BF16 safetensors conversion. Nothing else stores the tower at any
  size. On bytes, which is what decode is bound by, Q4G64 is **0.531 per value
  against GGML Q4_K's 0.5625 and W8's 1.062** — denser than both. Reopen only if
  the tower gains a K-quant route and a 27B safetensors conversion measures worse
  than one through `surogate quantize`.
- **Dequantise-and-requantise is not free.** Measured on Qwen3.5-0.8B-Q4_K_M:
  K-quant → BF16 → W8 adds 5.7e-3 relative error and *doubles* the bytes. That is
  why the native path exists.
- **A checkpoint's `quantization_config` states more than its format, and the rest is
  read now (2026-09-05).** `kv_cache_scheme` is honoured where `auto` already resolves to
  what it asks and refused, naming the flag, where it does not; `sparsity_config` and
  `transform_config` are refused. And the declaration is cross-checked against the
  checkpoint's own tensors, because it is a claim rather than a fact: a weight is quantised
  exactly when a scale sits beside it. Two published NVFP4 exports of the same 27B declare
  `ignore` lists of 2 and 303 entries while quantising identical tensors, and one of them
  omits 27 unquantised vision projections it never mentions. **Where the two differ, the
  tensors win and the disagreement is printed.**
- **A measurement written down goes stale silently.** The `nvfp4-mixed-bf16` export table
  names six attention layers left in BF16, measured from a published file. Neither
  `nvidia/Qwen3.6-27B-NVFP4` nor `unsloth/Qwen3.6-27B-NVFP4` has any: both quantise every
  attention and MLP layer. Whatever file the table describes, it is not either of the ones
  published today, so converting one now refuses instead of building an artifact that claims
  formats its own weights do not have. The tables stay until a checkpoint proves what should
  replace them; the check is what makes that visible.
- **A format the index cannot hold is a format we quietly re-quantise**
  (2026-09-05). F16 was the last one, and the breach was invisible because the
  numbers looked fine: `Qwen3.5-0.8B-UD-Q8_K_XL` scored 14.6946 against
  llama.cpp's 14.7129 while 53 % of its elements were being re-encoded to eight
  bits. Being *below* the reference was the tell — we were not serving the same
  model. Reading the file faithfully moved us to 14.7167, which is what agreement
  looks like. **Check what the artifact stores, not only what it scores:**
  `formats` and `indexed vs stored` over the objects say in one line what a
  perplexity number can hide.

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
  Q8_0 objects the moment Q8_0 became native. The fix guarded on "is the block
  256 values", which was the *wrong question* and only ever right by accident:
  Q1_0 (128), Q2_0 (64), MXFP4 (32) and NVFP4 (64) would each have reached a
  `KeyError` in `planes()`, and F16 (32) actually did. It asks whether the W8
  plane path can decode the type now. **When a predicate stands in for a
  capability, name the capability.**
- **A new GGML type fails silently in four places, and this build has no
  `-Werror=switch`.** `block_bytes` and `type_name` fall through to `0` and
  `"?"`; `block_values` defaults to `QK_K` (256); `dequantize_superblock`'s
  `if constexpr` chain has no `else`. Python has the same shape:
  `native_block_values` defaults to 256, so a type absent from
  `NATIVE_BLOCK_VALUES` gets 256-value rows and corrupt byte runs with no
  exception. Only the `Traits`, `decode_eight` and vec-dot templates fail loud.
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
- *(Shipped, not rejected, but it belongs beside the Q5_K note above.)* **Q6_K's
  210-byte blocks staged as forty-eight two-byte loads** were the whole of the
  Q6_K down gap. Covering each span with aligned sixteen-byte `cp_async` from the
  rounded-down address and folding the row's offset into the readers (`kCover`)
  took the routed body from 1,014 / 1,362 / 1,182 us to **643 / 842 / 963** at
  128 / 512 / 1,024 tokens, outputs bit-identical, against Q4_K's 522 / 657 / 764.
- **K5c's split of a mixed-format fused parent.** Above.
- **Porting MMQ for prefill (K3).** The wide route dequantises once and uses BF16
  tensor cores instead: ~200 lines against ~6,000, more accurate, and our MoE
  prefill kernel already beats llama.cpp's MMQ by 1.65×. Worth revisiting only
  where the dequantisation tile is the constraint.
- **An int8 dense prefill linear on the routed experts' tile (2026-09-04).**
  The MoE int8 tile (`ggml_i8_tile.cuh`) run over the dense K-quant linears with
  `row_base` the row's first superblock and `act_row` the identity, a persistent
  grid of 64-row × 64-token items, activations quantised once per prefill. It
  passes the fixture suite and **loses**: 27B 2,048-token prefill 881 ms / 2,390
  tok/s against 812 / 2,600 on the BF16 wide route, 512 tokens 269 against 257
  ms. The kernels took ~262 ms per prefill where cuBLASLt plus its staging took
  ~225 on the same weights. The instruction stream is the per-32 affine
  scale-apply — per MMA (16×8×32, 8,192 ops, ~4 SM-cycles) each thread converts
  four accumulators and applies two FMAs to each, ~3 SM-cycles of FP32 issue —
  so the tile tops out near the BF16 rate whatever the tensor cores can do,
  and widening BN or amortising the unpack moves it 10–15 %, not the 2× it
  needs. That is also why llama.cpp's MMQ, the same structure, sits at 172
  TFLOP/s effective in `llama-bench`. What would change it: accumulate a whole
  superblock in int32 with the 6-bit sub-block scales applied as integer
  multiplies and the (d, dmin) pair once per 256 — which needs one activation
  scale per 256, not per 32, i.e. a numerics change against llama.cpp that the
  perplexity gate would have to judge. Not retried without that design.
- **Flash-Next's remaining offload levers (closed 2026-09-05).** All three were
  measured and none is worth building as things stand.
  *A copy-engine gather* would take the expert fetch off the SMs, which is the real
  constraint: our gather is a kernel holding SMs while it waits on PCIe. But
  `cudaMemcpyBatchAsync` needs the miss list on the host, and a host node inside a
  captured prefill graph cannot issue copies, so it is an eager-prefill path or
  nothing. The next-layer prefetch built to hide the same cost measured a loss
  (TTFT 1.37 s against 0.84) and was removed. *The CPU expert path* decodes GGML
  blocks scalar at ~1 GB/s against a 24 GB/s PCIe gather, so the auto split takes
  30 % of misses and does not pay; the bank is decoded to Q4G32AM at load now, so
  this only bites `SUROGATE_SERVE_HOST_BANK_NATIVE=1`. *MTP on an offloaded MoE* is
  worth +8 % at draft 1 (35.2 against 32.6 tok/s, 71.6 % accepted) and less at every
  longer draft, because a wider verify round touches more distinct experts and pays
  more gathers — the same constraint again. The board rows are met on defaults
  (33.6 / 85.7 / 116.4 at 1 / 16 / 64 users) and the 28k row holds at 7.07 s TTFT
  with `--max-num-batched-tokens 8192`. Design and measurements: memory
  `reference_freetoken`, `project_serve_flash_next_board_recovery`,
  `project_serve_qwen4exp_mtp`. **Reopen only with a copy-engine gather design that
  survives a captured graph.**
- **Q8_0 codecs in the fused-projection kernels.** They cp_async 16-byte chunks
  out of separate code/scale planes, so Q8_0's 34-byte blocks mean six kernel
  rewrites. The load transform gets the same result and touches no kernel.

---

Commits are on `serve-engine`; `git log --oneline c5d327fe..` is the line of
work this file tracks.

Board rows: `surogate/serve/BENCHMARKS.md`.
