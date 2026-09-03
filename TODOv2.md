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
which 8 route — on one 5090, single stream, ~700-token prompt, 128 generated, warm,
greedy (2026-09-03, both rows the same session; the row-split row is the older 654-token pass):

| | prefill tok/s | decode tok/s |
|---|---:|---:|
| **surogate, native K-quant, int8 tensor-core route** | **13,195** | **315** |
| surogate, native K-quant, BF16 activations (the path before it) | 10,299 | 316 |
| llama.cpp, same file | 8,408 | 278 |
| surogate, dequantised to Q4G64 | 13,700 | 346 |

Ahead of llama.cpp on both. The int8 route is the routed experts' prefill on
llama.cpp's own arithmetic -- int8 activations per 32 values with a block sum,
the weights' codes as the other MMA operand -- and it passed the accuracy gate
the owner set for it, measured the way llama-perplexity measures:

| wikitext-2 test, 145 windows of 2048, second halves scored | PPL |
|---|---:|
| llama.cpp, same file | 6.2311 ± 0.040 |
| surogate, BF16 activations | 6.2378 ± 0.040 |
| **surogate, int8 route** | **6.2370 ± 0.040** |

`surogate/serve/tools/eval/perplexity.py` is the harness; the engine's side is
the NLL probe (`SUROGATE_SERVE_NLL_DUMP`, raw prompts via
`SUROGATE_SERVE_RAW_PROMPT`, eager prefill). Decode is untouched by the route.

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

1. **[x] Close the native prefill gap — the int8 tensor-core route (2026-09-03).**
   The BF16-activation kernel was at its floor (eleven variants); the route
   that replaced it quantises the prefill activations to int8 per 32 with a
   block sum, unpacks the weights' codes to int8 once per K tile, and runs the
   s8 MMA. Accuracy gate: 6.2370 against llama.cpp's 6.2311 (± 0.040). Prefill
   13,195 against 10,299 tok/s single stream (~700-token prompt) and 17,300
   against 14,300 at 8k, same session, three prompts each. `SUROGATE_SERVE_MOE_INT8=0`
   keeps the BF16-activation kernels. The kernel's design is in the progress log
   (2026-09-03); what the eleven earlier variants taught is under Traps.
2. **[~] K6 — retire Q4G64/Q5G64/Q6G64, and `surogate quantize` in their place.**
   The converters stop quantising and the three home-grown formats leave the
   engine with them, roughly 140 references. What replaces them for a model we
   trained is `surogate quantize`, which is item 11 and deferred: a thin version
   exists, so the capability does not vanish with the formats, but it is not a
   product yet. Until it is, a checkpoint we trained is either served as BF16
   or exported through that command's two llama.cpp passes. The old
   argument against the deletion is stale: the K-quant path costs ~19 % of
   prefill only against the BF16-activation kernel, and with the int8 route it
   measures 13,195 tok/s against the row-split path's 13,700 from an earlier
   pass, so one same-session comparison settles it. Decode is unaffected.
   **What the deletion would strand, checked 2026-09-03.** One live path still
   produces these formats: the 27B's `Qwen36GroupwiseInt` profile, whose
   endpoints bind `Q6G64_F16S` and whose layers bind `Q4G64_F16S`. Nothing else
   selects them -- the other targets import the names and use `W8G32_F16S` or
   NVFP4. So the deletion is one target's safetensors profile, not five, and it
   is a decision rather than an open task: it costs the 27B its groupwise-int
   route until `surogate quantize` (item 11) is a product.
3. **[x] GGUF coverage beyond the Qwen3.5/3.6 families (2026-09-03).** A GGUF is servable only
   if `gguf_target_key` resolves it, and until today that was four Qwen shapes.
   Qwen3 now resolves and serves (a `Qwen3-0.6B` Q4_K_M converts, caches and
   answers), on two pieces of new machinery: `config.json` is **synthesised from
   the GGUF's own metadata** for a plain dense decoder, so a family needs no
   vendored config per size and we vendor no config whose licence is not ours;
   and `_HF_ALIAS_FIXUPS` names the handful of tensors where gguf-py's alias
   preference picks a spelling the converter does not use (Qwen3's per-head
   norms are `q_norm`, the generic map reaches them as `q_layernorm`).
   Llama and Gemma 3 followed, and needed three more things. **SentencePiece
   vocabularies** (`tokenizer.ggml.model == "llama"`) now reconstruct: a GGUF
   carries pieces and scores and no merges, so `_spm_merges` recovers the merge
   list by splitting every piece and ordering by score — it reproduces
   TinyLlama's 61,249 merges exactly and in order, and both families then encode
   3,104 corpus strings identically to their official tokenizers. **The alias
   preference is deterministic** rather than first-listed, which had put
   Gemma 3's up projection under `feed_forward`. And **llama.cpp's own export
   transforms are inverted**: Gemma folds the +1 its norm applies, Llama
   permutes Q and K for its rotary, and both are silent when missed — TinyLlama
   answered fluently and wrongly, Gemma produced multilingual noise. With them
   inverted, TinyLlama matches llama.cpp word for word on the same file.
   The deeper limit is item 6: each target is one compiled geometry, so this
   covers *those* sizes, not those families — Qwen3-8B still has nowhere to go.
   A GGUF published without a chat template is also refused, which is what the
   base `google.gemma-3-270m` files are.
   **Any size of those three families now serves** (see item 6): the gates ask
   for the architecture, the binders validate against the artifact's declared
   dimensions, and Qwen3-1.7B is served by the target compiled for the 0.6B.
   Llama and Gemma 3 took the same treatment; no second size of either was on
   this machine to prove it with.
   **The official unsloth `Qwen3-0.6B-Q4_K_M.gguf` now serves**, after two
   things that only a published export exposes. It declares `<|vision_pad|>`
   where the Qwen repository declares `<|endoftext|>`, and the frontend asserted
   the literal — a *registered-checkpoint identity* check standing on the
   generic path. A pad token says nothing about how text tokenizes: the frontend
   resolves it to an id and nothing pads with it. The literal now runs only for
   a target claiming the registered tokenizer, and the bridge stopped rewriting
   the pad token to match, which had been putting a token in the artifact that
   the file does not claim. Both directions were the same mistake.
4. **[ ] F — FP8.** compressed-tensors per-channel/per-tensor is per-row with an
   FP32 scale — add `_F32S`, or accept the BF16 cast. HF fine-grained FP8 is
   block-scaled and has no runtime format at all.
5. **[ ] N — NVFP4 ModelOpt ingest.** `weight_scale_2` is a multiplier where
   compressed-tensors' global scale is a divisor; parents split per component.
6. **[~] M2 — one directory per architecture.** `qwen3_5_{0_8b,2b,4b}` are
   ~1,750 lines each for seven integers; `variant.h` differs by 2 lines across
   the three. Nothing requires the split: all 52 headers in
   `csrc/src/serve/api/ops/` take runtime shapes, and the attention kernel is
   already dispatched at runtime from a registry of eight `GqaGeometry` shapes.
   **Geometry as data, not as a template parameter** (owner: "choose the best
   option"). Shapes alone cannot yield an RMS epsilon, a rope base or a layer
   schedule, and the converter holds all of them, so the *artifact carries the
   numbers* and the binder checks tensor shapes against them. Config is the
   authority; the container is a cache.
   Shipped so far: `family::TextGeometry`, the compiled config as a value, with
   primary dimensions as members and derived ones (`key_dim`,
   `convolution_dim`, the MTP row counts) as functions, so a declaration cannot
   leave a stale derived value behind; an optional `geometry` root member on the
   artifact directory, read by both the C++ reader and the Python container;
   `ModelConfig` turned from static constants into data, and `kCfg` into
   `TextContext::cfg_` (324 call sites); the qwen3 converter writing the block,
   with `token_domain` taken from the tokenizer rather than the padded
   `vocab_size`.
   **The runtime consumes it, and one target serves every size of its family.**
   `Qwen3-1.7B` is served by the target compiled for `Qwen3-0.6B`, from its
   GGUF, through `surogate serve`; the 0.6B is unchanged to the token. What that
   took: the planner and the sequence plan carry the geometry and the 49 layout
   reads follow it; the workspace recipe (the shape contract the layout
   simulation and the real schedule share) is parameterised by it, as are all
   twelve projection workspace sizers across the nine targets; the program core
   holds it and the scattered reads in decode, prefill, MTP, DFlash and the
   request planner read it; the model view and the binding plans size their
   layer storage when the weights are bound. The residual width is derived from
   the hidden state for a family that does not widen it -- leaving it compiled
   is a mismatch only the first embedding lookup finds.
   Two kernel tables were the real gate, and both were policy rather than
   limit. The fused ungated attention projection gained the 4096-row parent at
   hidden 2048 (same row split as the 0.6B, so only K moves), and a K-quant
   parent now splits by row range at any width, the way the gated one already
   did. The w8 linear dispatcher no longer refuses a shape it has no measured
   route for: its launchers take n, k and T at runtime, so unregistered shapes
   take the family's default bands and the measured entries stay measured.
   **Left: the three Qwen3.5 directories.** Their binders, variants and
   converters now bind against the declared geometry and the artifacts state
   it, so the mechanism is in place. The collapse itself is not done, and it is
   not just a rename: the 2B carries a vision tower the other two do not, so a
   merged target has to take that path with it. Two dimensions also stay
   compiled in the hybrid forward interface and are marked in the code -- the
   attention head width (the family's at every size) and the GDN output-gate
   width (no runtime handle; the split projection payload would have to carry
   it).
7. **[ ] M4 — unify weight loading with the trainer.** Serve's `recipe.py` +
   `inventory.py` per target restate what the trainer's `hf_mapping` DSL already
   declares (`fuse`, `split`, `stack_experts`); the trainer's
   `SafeTensorsReader` is the better reader (multi-shard, GDS, strided).
8. **[x] K5c — fused K-quant GDN projection-and-convolution: closed, negative
   (2026-09-03).** Built and measured; it costs more in kernel launches than it
   saves in bandwidth, so it was left off and the code is not in the tree. A
   result, not a task: reopening it needs a reason the measurement did not have.
9. **[x] Drift to fix (2026-09-03).** The MTP block's five matrices demanded
   `W8G32_F16S`, so a GGUF keeping its nextn tensors was refused; nothing in the
   kernels wanted that, since they dispatch on the weight's qtype. They bind
   through the same `bind_linear` the text layers use. Proved on our own
   `surogate quantize` export of Qwen3.5-0.8B, whose q4_k_m mixture puts
   `eh_proj` at Q4_K: 12 MTP objects across BF16, Q4_K, Q6_K and W8G32_F16S bind
   and serve, where that file previously failed on `mtp/input_projection`.
   `--no-cache` was in the help text and did nothing; it now skips cache reuse
   on every path, which is what you want after editing a converter, since the
   cache is keyed on the checkpoint. The tools README no longer tells users to
   download artifacts from Hugging Face and its links resolve.
   `surogate convert` still does not exist and is not wanted: `surogate serve`
   converts.
   **The converter is not a tool** (owner, 2026-09-03). `surogate serve` runs it
   on every first load, so `serve/tools/convert/` is now `serve/convert/` and
   `serve/tools/artifact/` is `serve/artifact/`; `serve/tools/` keeps the
   workflows an owner runs by hand (bench, eval, parity, probe, reference,
   smoke, generate). The README above is what is left of that drift.
10. **[ ] Q6_K down, the one tensor the int8 route did not fix.** 688 µs against
   the row-split kernel's 337, where Q4_K and Q5_K now beat theirs (498/608 and
   316/335). It is `routed_down` on 3 of 40 layers, so it costs ~1 ms of a
   35 ms round. The cause is structural: Q6_K's scales cover sixteen values, so
   its groups run two `m16n8k16` MMAs instead of one `m16n8k32`, and its
   210-byte block is only two-byte aligned, so the tile stages with scalar
   loads where the others use `cp_async`. Both are worth one attempt: a
   sixteen-wide scale table read twice, and a staged copy that realigns the
   block on the way into shared memory.
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

Commits, most recent first: `6d045ef7` `3becc963` `8374dfc5` `13c05055`
`996e4166` `4bc55762` `3e94fa60` `0e4a1576` `d52a5e62` `db4cc18e` `7bebf54e`
`0624d4b6` `300f9d26` `1455c3b0` `700b9a23` `9995f894` `48a25753` `1b33158b`
(2026-09-03), and `7aea807e` `7de24bab` `2432825e` `478ab1ae` `ca656302`
(2026-09-02).

Board rows: `surogate/serve/BENCHMARKS.md`.
