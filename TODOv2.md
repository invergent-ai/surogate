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

What is left is in this file. Board rows are `surogate/serve/BENCHMARKS.md`;
the history of what was tried is `design/INFERENCE.md`.

---

## Roadmap

Seven open or partly done. Two are decisions waiting on `surogate quantize` as a
product rather than tasks (1 and 7); the rest are work.

1. **[~] Retire Q4G64/Q5G64/Q6G64.** The three home-grown formats and the
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
   `surogate quantize` (item 7) is a product.
2. **[~] FP8: the block-scaled kind serves (2026-09-04); the per-channel kind waits on
   a runtime-shaped row route.**
   - *HF fine-grained FP8* (`quant_method: fp8`, `weight_block_size: [128, 128]`,
     `weight_scale_inv` an F32 grid) is a new format, `FP8_E4M3FN_BLK128_F32S` in layout
     `block-scale-128-fp8-v1`: the checkpoint's E4M3 codes and its scale grid, untouched.
     This cuBLASLt admits only scalar FP8 scales on sm_120 (probed: `VEC128`, `BLK128x128`
     and `OUTER_VEC` all `NOT_SUPPORTED`), so the routes are the engine's own and
     runtime-shaped: a 64x64 e4m3 tensor-core tile that applies the block scale once per
     128 of K (activations quantised per token per 128, the recipe's convention), and a
     decode GEMV on exact BF16 activations. The fused ops project it by rows through the
     same branch the K-quants take, so no size registers a shape. Export profile
     `fp8-block` (`--profile` auto-detected from `quantization_config`), the uniform
     recipe's object graph at any geometry. Qwen3.5-0.8B-FP8: **14.7418** against the
     BF16 torch reference's 14.60 on the same 40 windows (+1 %, the recipe's cost) and
     llama.cpp's IQ4_XS 15.15; 833 tok/s decode on one 5090.
   - *compressed-tensors per-channel/per-tensor* (a per-row F32 scale). The engine's
     `FP8_E4M3FN_ROW_BF16S` route holds the numbers but is compile-time geometry -- the
     27B's shapes only -- so a small checkpoint of this kind would be refused before its
     scale precision mattered. A checkpoint is now obtainable
     (`RedHatAI/Qwen3-0.6B-FP8-dynamic`); the work is a runtime-shaped row-scaled route
     (or reading per-row scales into the block route with a 1-row block), then the
     `_F32S` variant or a documented BF16 cast of the scales.
3. **[~] Every GGML weight type is read where it lies (2026-09-04); what the UD mixtures
   still cost.** The 27B's refusal was never about vision: unsloth's "UD-Q4_K_M" holds 117
   IQ4_XS and 4 IQ3_S tensors, and the engine read 13 of llama.cpp's 27 storable types. The
   other fourteen shipped in 6613389f -- the eight IQ formats, TQ1_0/TQ2_0, MXFP4, ggml's
   NVFP4 (`NVFP4_GGML` here: the name was taken), Q1_0, Q2_0 -- on every route, 1,307
   fixture cases against gguf-py's dequantiser plus real tensors from the UD files, and
   every type switch in the tree now expands the one list. Perplexity, wikitext-2 test, 40
   windows of 2048, ours (eager, raw prompt) against llama-perplexity on the same windows:

   | file | ours | llama.cpp |
   |---|---:|---:|
   | Qwen3.5-0.8B-IQ4_XS (50 % IQ4_XS) | 15.094 +/- 0.225 | 15.151 +/- 0.226 |
   | Qwen3.5-0.8B-UD-Q2_K_XL (Q2_K/Q3_K + IQ3_S/IQ3_XXS/IQ2_S/IQ4_XS) | 20.209 +/- 0.305 | 20.016 +/- 0.302 |
   | Qwen3-0.6B-UD-IQ2_M (IQ2_S/IQ3_S/IQ3_XXS) | 40.128 +/- 0.702 | 42.045 +/- 0.743 |
   | Qwen3-0.6B-UD-IQ3_XXS | 29.509 +/- 0.505 | 30.250 +/- 0.522 |
   | Qwen3-0.6B-IQ4_XS (64 % IQ4_XS, 35 % Q6_K) | 18.330 +/- 0.294 | 17.866 +/- 0.286 |

   The IQ2/IQ3 rows come out ahead because the vec-dots evaluate the block scale exactly
   where llama.cpp's integer form truncates. The 0.6B IQ4_XS row is 2.6 % behind (1.6
   sigma) while the 0.8B IQ4_XS row is not. The Q4_K_M control on the same target reads
   17.675 +/- 0.28 against llama.cpp's 17.510 +/- 0.28 (+0.9 %, old types only), so the offset
   was the `qwen3` target's, not the codec's. **Found (2026-09-04): the FP8 KV cache.** A
   probe ladder against an fp32 transformers forward over the same GGUF weights put every
   attention *input* at BF16 noise (5e-3) and the attention *output* at 3e-2; recomputing the
   attention from the engine's own q/k/v in fp32 reproduced the 3e-2, and a BF16 cache took it
   to 1.5e-3. e4m3's three mantissa bits are ~2 % of noise on every K and V, which a
   pure-attention stack pays in every layer: with a BF16 cache Qwen3-0.6B-Q4_K_M reads
   **17.4315** (llama.cpp 17.5103; fp32 reference 17.4127) and IQ4_XS **17.8580** (17.8659).
   The 27B, a 3:1 GDN stack, moves only 5.1699 -> 5.1495 against 5.0166, so its gap is
   elsewhere (below). The KV default is now `auto`: BF16 for a pure-attention target, e4m3
   where linear-attention layers carry the stack -- "Decisions that govern". Left:
   **The 27B-class GGUF serves, with MTP (2026-09-04).** `unsloth/Qwen3.8-27B-GGUF`
   UD-Q4_K_M, by the end of the day: **83.3 tok/s** decode with `--spec mtp --draft-tokens 1`
   against llama.cpp's 44.8 on the same file (46.5 at noon), TTFT 0.24 s against 1.18 s,
   perplexity **5.0477 +/- 0.129** against 5.0166 +/- 0.127 (5.1699 at noon). What the UD
   mixture still cost, and what removed it: the file quantises the components of a fused
   parent to different types (q Q5_K beside k Q4_K, gate IQ4_XS beside up Q4_K, 25 of 48
   `value_z`), and llama.cpp's V-head reorder puts a *column* permutation on every GDN
   `out_proj`; both were dequantised and requantised to Q4G64/Q5G64 -- 33 % of the bytes, a
   second quantisation, and the whole of the 2.6 % gap (the 0.8B, 3 % bridged, matched). Now a
   native object may carry typed row `segments` (the fused ops already projected row ranges,
   and a component never straddles a type run: `ggml_weight_rows` resolves the range to its
   segment; `ops::weight_rows` is public and the qwen3_5 loader's row views use it), and a
   single-source object whose source has a column map keeps the file's column order and
   carries `group_map` with no transform -- the runtime permutes the activation's 32-row groups
   before the launch (`input_for`, one small kernel per GDN layer, measured free). 328 of 330
   objects, 16.97 GB, read in place; the two left are Q8_0 a/b projections of a few MB.
   The first segmented board read *35.8*: the decode profile put 70 % of the round in the
   IQ4_XS GEMV at 232 us/call (Q4_K: 19) -- a per-lane `__constant__` table read, serialised
   32-way -- which the bridge had hidden on most IQ4_XS tensors. The byte-permute lookup the
   file already used for MXFP4 fixed it; the old 46.5 had been throttled by the same kernel.
   Four things were wrong at noon, none of them the formats: the repack planner was not a fixed
   point; a GGUF's `block_count` includes the MTP block; the native path silently dropped the
   column permutation llama.cpp's V-head reorder puts on the GDN `out_proj` (48 objects, and
   the reason the model produced confident noise at 4.6M perplexity); and the 27B-class
   groupwise runtime had been incomplete since f308f747 -- the split attention loader was cut
   as unreachable, the GDN `QkPlusVz` payload had no runtime, and the W8 capacity queries
   refused shapes their routes never registered. `mtp/input_projection` was bound with
   `K = query_size` where the object is `[hidden, 2*hidden]`; the two coincide at the size this
   target compiles, so every larger model failed at its first draft round.
   **Prefill, measured (2026-09-04).** Through the server we lead: a 2,048-token prompt is
   **752 ms** TTFT against llama.cpp's 1,265 (812 at noon), and **2,804 tok/s** of prompt-eval
   against its 2,565. `llama-bench pp2048` is the compute bar at 3,190 -- it gets one
   2,048-wide batch where the server splits into 512s -- and we are 14 % under it, 149 TFLOP/s
   effective against 174. The noon nsys capture (`--cuda-graph-trace=node`, or the graph hides
   everything) said prefill is GEMM-bound with no scheduling gap: 42 % cutlass BF16 (the
   dequantise-then-GEMM route), 38 % our groupwise kernels on the re-encoded halves (gone
   with the native pass), 7.5 % `dequantize_rows` staging, ~4 % GDN and attention. Measured route rates at the dominant MLP shape: BF16 222 TFLOP/s, Q4G64
   187, fused Q4 SwiGLU 183, W8 159, against 838 of int8/fp8 tensor throughput on the card.
   **The int8 dense prefill GEMM on the routed experts' tile was built and measured a loss**
   (881 ms against 812; "Measured and rejected" below has the design and the reason: the
   per-32 scale-apply is the instruction stream, not the MMA). Re-routing the groupwise
   weights through the BF16 path is measured and *not* worth it either -- the staging pass
   moves the crossover to T ~= 1,500 for a ~5 % win. What is left on prefill is the 38 % in
   the groupwise kernels on re-encoded halves, which a native mixed-type fused parent would
   put on the BF16 route (222 against 183-187 TFLOP/s, ~5 % whole-model), and the numerics
   of the int8 tile itself, fixed here: its activation planes carried the raw Σx and now
   carry d·Σq, matching the GEMV route (real-tensor error 1.25e-2 -> 1.77e-3 relative).

4. **[ ] Unify weight loading with the trainer, still true but smaller than it
   was (checked 2026-09-03).** Serve's `recipe.py` + `inventory.py` per target
   restate what the trainer's declarations in `surogate/dsl/models/` already say.
   Both sides describe the same nineteen architectures: the DSL has `qwen3.py`,
   `qwen3_5.py`, `llama.py`, `gemma3.py` and fifteen more, and serve has a recipe
   per converter — 4,468 lines of them.
   The remaining duplication is smaller than it was. The recipes no longer
   restate *dimensions*, because the artifact carries them and the binder
   validates against them; what they still restate is the *mapping* — which
   checkpoint tensor becomes which artifact object, and how fused objects are
   assembled. That is the part worth unifying, and the part `hf_mapping` already
   spells out.
5. **[~] Flash-Next: the offload path's remaining levers (2026-09-04).** The
   board rows are met on defaults (33.6 / 85.7 / 116.4 decode at 1 / 16 / 64
   users); what is left is above them.
   - **A copy-engine gather.** Our expert gather is a kernel, so it holds SMs
     while it waits on PCIe. That is why the next-layer prefetch was built,
     measured a loss (TTFT 1.37 s against 0.84) and discarded: overlapping a
     kernel gather with the next layer's compute starves that compute rather
     than hiding the transfer. `cudaMemcpyBatchAsync` off the SMs would change
     that, but it needs the miss list on the host, and a host node inside a
     captured prefill graph cannot issue copies — so it is an eager-prefill
     path first, if at all. Design and measurements: memory
     `reference_freetoken`.
   - **The CPU expert path decodes GGML blocks scalar.** `--cpu-moe-share` works
     on a GGML-block bank, but auto measures the host at ~1 GB/s against a
     24 GB/s PCIe gather and takes 30 % of misses, which is not yet a win. The
     bank is decoded to Q4G32AM at load now, so this only bites a run that keeps
     the file's blocks (`SUROGATE_SERVE_HOST_BANK_NATIVE=1`).
   - **The 28k-prompt board row** (long-context ingestion) has not been re-measured
     since the native path landed.
6. **[~] MTP for Flash-Next serves; the acceptance is not the speedup
   (2026-09-04).** `--spec mtp` runs the NextN head end to end at 78.6 %
   acceptance — which is the evidence the graph is right — but decode moves
   30.6 → 34.1 tok/s, not the 1.3-1.7x the head is advertised at. Acceptance
   being high, the cost is the round, not the drafts: a 4-6 column verify touches
   more distinct experts than a single token and pays more PCIe gathers with
   3,172 of 5,110 experts resident. The graph and the levers are in memory
   `project_serve_qwen4exp_mtp`.
7. **[~] DEFERRED, off the critical path — `surogate quantize`, the export of a
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

Routed today, everywhere the runtime needs them: **every GGML weight type llama.cpp
stores** -- Q2_K–Q6_K, Q8_0, Q4_0/Q4_1/Q5_0/Q5_1, IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/
IQ3_XXS/IQ3_S/IQ4_NL/IQ4_XS, TQ1_0/TQ2_0, MXFP4, NVFP4_GGML, Q1_0/Q2_0 -- on linear,
embedding and MoE decode/small-T/prefill (Q4_K/Q5_K/Q6_K on the int8 tensor-core
route, the rest on the BF16 route), read from the GGUF; **BF16** at any 8-aligned
shape, **W8G32_F16S**, and **NVFP4 compressed-tensors** (TRT-LLM cutlass for routed
MoE). What is not:

| Format | Status |
|---|---|
| NVFP4 (ModelOpt) | **[~]** the 4B serves; the 0.8B and 2B need a recipe (roadmap 3) |
| FP8 | **[~]** only `FP8_E4M3FN_ROW_BF16S`; per-tensor, per-channel and block unsupported (roadmap 2) |
| W4A16 / W4A16_ASYM | **[ ]** `Q4G64_F16S` is symmetric with no zero point and no actorder |
| MXFP4 / MXFP8 | **[ ]** nothing in serve; the trainer decodes MXFP4 |
| GPTQ / AWQ | **[ ]** off the roadmap by owner decision |
| `kv_cache_scheme` | **[ ]** refuse or support — do not ignore silently |
| F16 (GGUF) | **[ ]** bridged to BF16, three mantissa bits lost; UD-Q8_K_XL is mostly F16 |

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
- **Q8_0 codecs in the fused-projection kernels.** They cp_async 16-byte chunks
  out of separate code/scale planes, so Q8_0's 34-byte blocks mean six kernel
  rewrites. The load transform gets the same result and touches no kernel.

---

Commits are on `serve-engine`; `git log --oneline c5d327fe..` is the line of
work this file tracks.

Board rows: `surogate/serve/BENCHMARKS.md`.
