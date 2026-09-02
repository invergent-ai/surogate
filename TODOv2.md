# TODOv2 — loader and quantization redesign

Tracker for one endeavor: clean up the serving targets, support quantized
checkpoints generically (any compressed-tensors NVFP4, any GGUF), and unify
weight loading with the trainer. `TODO.md` remains the tracker for the
pre-existing review items; nothing moves between the two.

Status legend: `[ ]` open · `[~]` in progress · `[x]` done (commit) · `[?]` open question

## 0. What the code says (verified 2026-09-02)

Each claim below was checked against the tree, not inferred.

- **Per-size targets are not required by anything.** All 52 headers in
  `csrc/src/serve/api/ops/` take runtime shapes; none is templated on
  geometry. `qwen3_5_{0_8b,2b,4b}` are ~1,750 lines each for seven integers
  (`hidden`, `layers`, `intermediate`, `gdn_value_heads`, `query_heads`,
  `kv_heads`, plus vision tower). `variant.h` differs by 2 lines across the
  three, `variant.cpp` by ~37 of 512, one of which is the namespace macro.
- **Format is already data everywhere except one line.** The artifact stores
  `NumericFormat` per object (`artifact/reader.h` `TensorDescriptor`). The
  runtime `Weight` (`core/tensor.h`) carries `qtype`, `layout`, `group`,
  `scale_dtype`, `scale_ne/nb`, `weight_scale_divisor`,
  `input_scale_divisor`. `ops::linear(x, w, out, stream)` routes on
  `w.qtype`. The only hardcoding is `Binder::require_tensor`
  (`artifact/binder.cpp:50`), which asserts the caller's format and throws
  on mismatch — that is what forces one compiled-in `WeightsProfile` per
  export.
- **The profile fan-out has already started.** The 4B carries two NVFP4
  profiles (`Qwen35Nvfp4` "mirrors an export that left particular layers
  BF16", `Qwen35Nvfp4Mixed` "every linear weight NVFP4"). Its 58-line
  `bind_nvfp4_uniform_text_layers` has zero numeric literals and sixteen
  `TextConfig::` references — fully geometry-generic — and is absent from
  the 0.8B and 2B only because nobody copied it.
- **The converters pin one exact export each.** `qwen3_6_27b/convert_nvfp4.py`
  demands `format: "mixed-precision"`, exactly `config_groups.group_0`,
  `ignore: ["lm_head"]`, and a `mixed_native_manifest.json` histogram down to
  `linear/NVFP4: 379`. The real RedHatAI 35B-A3B NVFP4 on disk declares
  `format: nvfp4-pack-quantized`, `targets: ['Linear']`, and an `ignore`
  list of 343 entries including regexes (`re:^mtp.*`,
  `re:.*linear_attn\.in_proj_.*`). It would be refused on nearly every
  clause. `routed_nvfp4.py:151` already states the right principle —
  "`quantization_config` is the authority, not the tensor names" — but uses
  it only to refuse. `ingest.py:64`'s whole compressed-tensors detection is
  `"NVFP4" in json.dumps(quant)[:2000]`.
- **The spec and its matcher are installed.** `compressed-tensors 0.17.0` in
  the venv: `QuantizationConfig.from_pretrained`, `match_targets`,
  `match_named_modules`, `is_narrow_match`. Re-implementing the `ignore`
  regex semantics would repeat the safetensors mistake below.
- **Weight loading is duplicated, and the trainer's half is the larger.**
  The trainer has `csrc/src/utilities/safetensors.{h,cpp}` (881 lines:
  multi-shard `SafeTensorsReader`, GDS/cuFile with POSIX fallback, strided
  partial reads, on-read dtype cast, `SafeTensorWriter`, HF cache
  resolution) and **already loads four pre-quantized formats** — HF
  fine-grained FP8, ModelOpt NVFP4, MXFP4, BnB NF4 — through
  `csrc/src/runtime/qlora/dsl_qlora_pipeline.cpp:1222-1998`, detected from
  `config.json` in `surogate/core/model/hf_config.py:94-180`, honoring
  `ignore`/`modules_to_not_convert` with globs. It maps HF names through a
  declarative DSL (`surogate/dsl/hf.py`: `fuse`, `split`, `transform`,
  `tied_to`, `stack_experts`, compiled to IR JSON, consumed by
  `dsl::MappingSpec`), with q/k/v fusion as a strided read into destination
  views. Serve has its own `artifact/` (1,239 lines) behind a bespoke
  `.sinfer` container, a Python `ShardReader`, a hand-rolled safetensors
  header parser in `serve/lora_registry.cpp:30-117`, and a hand-written
  `recipe.py` + `inventory.py` per target that restates what the DSL
  mapping already declares.
- **What the trainer lacks:** compressed-tensors (zero hits for
  `weight_packed`, `weight_global_scale`, `nvfp4-pack-quantized`; its NVFP4
  `data_suffix` is `""`, so it looks for `.weight` and misses
  `.weight_packed`), GPTQ/AWQ (recognized, no loader), activation scales,
  any GGUF, and host-side delivery (both read paths are hardcoded H2D).
  Its consumption model is dequant-to-BF16; fused q/k/v from a prequantized
  checkpoint is dequant → concat → requantize.
- **A real test file is on disk.**
  `/home/densemax2/work/models/hf/Qwen3.6-35B-A3B-NVFP4-redhat-vllm`: 24 GB,
  124,325 tensors, 30,880 `weight_packed` + 30,880 `weight_global_scale`,
  three shards. Nothing in the tree can load it alone: the closest path,
  the 35B routed converter, accepts `nvfp4-pack-quantized` for the experts
  but needs a BF16 sibling checkpoint for everything else (not on disk) and
  pins the target.

## 1. The spec surface to support

Not a survey of files. `compressed_tensors.QuantizationConfig` is the
contract; support what it can express and every export follows.

| Field | Values in 0.17.0 |
|---|---|
| `CompressionFormat` | dense, sparse-bitmask, sparse-24-bitmask, int-quantized, float-quantized, naive-quantized, pack-quantized, marlin-24, mixed-precision, **nvfp4-pack-quantized**, mxfp4-pack-quantized, mxfp8-quantized |
| `QuantizationStrategy` | tensor, channel, group, block, token, **tensor_group**, attn_head |
| `QuantizationType` | int, float |
| `QuantizationArgs` | num_bits, type, symmetric, group_size, strategy, block_structure, dynamic, actorder, scale_dtype, zp_dtype, observer, observer_kwargs |
| `QuantizationScheme` | targets, weights, input_activations, output_activations, format |
| `QuantizationConfig` | config_groups, quant_method, kv_cache_scheme, format, quantization_status, global_compression_ratio, ignore, run_compressed |
| Preset schemes | FP8, FP8_BLOCK, FP8_DYNAMIC, INT8, MXFP4, MXFP4A16, MXFP8, MXFP8A16, NVFP4, NVFP4A16, UNQUANTIZED, W4A16, W4A16_ASYM, W4A8, W4AFP8, W8A16, W8A8 |

Resolution rule (theirs, not ours): a module's scheme is the `config_groups`
entry whose `targets` match it (class name like `Linear`, exact module name,
or `re:` regex), unless it matches `ignore`; `mixed-precision` at the top
level means per-group formats. Multiple groups are normal.

The product statement the plan commits to (`design/serve-engine-plan.md:16`):
"GPTQ, AWQ, GGUF K-quants, FP8, NVFP4, MXFP4, NF4, or plain BF16". Only
GGUF i-quants, EXL2, EXL3 and AQLM may be refused, and only "with a printed
reason" (`:44`).

What each preset needs from the runtime, against the coverage in §2a:

- [ ] NVFP4 / NVFP4A16 — fp4 pairs, fp8 per-16 block scales, fp32 global scale (compressed-tensors: global scale *divides*; ModelOpt exports invert it). Runtime: `QType::NVFP4` exists; shape-gated, see M1.
- [ ] FP8 / FP8_DYNAMIC / FP8_BLOCK — e4m3 with per-tensor / per-channel / block scales. Runtime: only `FP8_E4M3FN_ROW_BF16S` (per-row BF16 scale), six registered shapes, no generic path.
- [ ] W4A16 / W4A16_ASYM / W4A8 — int4 `pack-quantized`, group strategy, optional zero points and actorder. Runtime: `Q4G64_F16S` is symmetric, group 64; asymmetric needs a min plane (`Q4G32AM` exists CPU-only).
- [ ] W8A8 / W8A16 / INT8. Runtime: `W8G32_F16S` (group 32, fp16 scale) is the broadest-supported format; W8A8 IMMA prefill exists.
- [ ] MXFP4 / MXFP8 — e8m0 block scales. Runtime: nothing in serve; the trainer decodes MXFP4 (`kernels/mxfp4_dequant.cu:231`).
- [ ] `kv_cache_scheme` — refuse or support; do not ignore silently.

## 2. The design

Three independent axes. Today all three are collapsed into "one directory
per model size, one profile enum per export, one converter per file".

**A. Format is data, read per tensor.** Structure (which tensors exist, what
shape the geometry implies) stays compiled in. Format comes from the
checkpoint's `quantization_config`, resolved with compressed-tensors' own
matcher, and is carried per tensor. Detection lives where the trainer
already does it — `surogate/core/model/hf_config.py:94-180` maps
`quant_method` fp8 / modelopt / mxfp4 / bitsandbytes today; compressed-tensors
is added there through the library, and serve reads the same answer.
The binder checks shape and "can an op route this format", not "is this the
format I was compiled for". `WeightsProfile` disappears; the per-export
`bind_*` functions merge; `make_sequence_planner` takes formats from the
load plan (they exist by then: `plan_load` runs at `targets/registry.cpp:206`,
the planner at `:219`).

**B. Geometry is a template parameter.** One directory per architecture;
a size is an instantiation (`Qwen35<Geometry<2048, 24, 6144, 8, 2, …>>`);
route choices that depend on shape (`linear` vs `linear_pair` for a 512-row
kv block — `qwen3_5_2b/impl/variant.cpp:183`) become `if constexpr` on the
geometry. Keeps specialization, build-time shape checks, and the baked
tables. Magic numbers that are already named constants (`5120`/`10240` =
`mtp_attention_input_rows`) become expressions again.

**C. One weight loader, three concrete merges.**
1. *Structure*: a serve target's `recipe.py` + `inventory.py` is replaced by
   the trainer's `hf_mapping` for that architecture. The DSL already says
   `qkv_weight = fuse(q_proj, k_proj, v_proj, dim=0)`
   (`surogate/dsl/modules/attention.py:33-48`); serve restates the same
   fact as literal row ranges per target.
2. *Reading*: safetensors through `SafeTensorsReader` (multi-shard, GDS,
   strided reads). Its delivery is device-only, which suits a repack-at-load
   on the GPU — where the plan puts the repack anyway.
3. *Codecs*: one NVFP4 stored-layout codec. The trainer's
   `dequantize_fp4_block` + `swizzle_fp8_scales_rowmajor_to_f8_128x4` and
   serve's `swizzle_nvfp4_scales` / `nvfp4_tiled_scale_offset` both claim
   the F8_128x4 layout (`colocate.py:290-296` records vLLM's swizzle as
   bit-identical to the trainer's). Being checked; see §4c.

**Fusion under differing global scales.** A compressed-tensors export gives
every `Linear` its own `weight_global_scale`; our fused parents
(`query_key_gate_value`, `gate_up`, `query_key_value_z`) hold one
`weight_scale_divisor`. Today the converter aborts (`_same_divisor`,
`qwen3_6_27b/recipe_nvfp4.py:396-411`) or the inventory splits the parent by
hand (`gdn/query_key_value` + `gdn/z`). The trainer's stacked-expert path
instead rescales block scales to a common global (`rescale_fp8_scales`),
which PATCHES #87 measured at ~2 % mean relative error on a GDN parent —
lossy, so never the default. Rule: **resolve constituents individually; when
their scales differ, bind unfused and run the unfused ops** (the 2B already
runs `linear`×2 for k/v). Follow-up that restores the fused kernels without
the loss: a per-row-range divisor on `Weight`.

Constraint on all three: the format assertion that goes away was a loud
failure mode. Its replacement is shape assertion + routability check, not
nothing.

## 2a. Serve kernel coverage today (from the survey; file:line in the survey report)

Nine closed formats: `Q4G64_F16S`, `Q5G64_F16S`, `Q6G64_F16S`, `W8G32_F16S`,
`BF16_CTRL`, `FP32_CTRL`, `I32_CTRL`, `NVFP4`, `FP8_E4M3FN_ROW_BF16S`.
Format → storage layout is 1:1 by construction (`typed_binding.cpp:13-30`).
Adding a format means editing four enums plus `quant_geometry`,
`format_name`, `qtype_for`, `storage_layout_for`, and every `require_*`
validator in the op wrappers.

| Op | Formats | Real constraint |
|---|---|---|
| `linear` | Q4, Q5, Q6, W8, BF16, NVFP4, FP8 | NVFP4: 5 registered shapes, or "generic" with **K ∈ {2560, 4096, 5120, 9216}** because the activation quantizer is instantiated per K (`nvfp4_config.h:168-174, 206`); FP8: 6 shapes, no generic |
| `linear_add` | BF16, W8, Q5, NVFP4 (2 shapes + generic), FP8 (2 shapes) | Q4/Q6 refused |
| `linear_pair` | W8 only | `n = 1024` hardcoded, K ∈ {5120, 2048} |
| `attn_input_proj` | BF16 (1 shape), NVFP4, FP8 (1 shape), W8 (4 shapes), Q4+Q5 pair | NVFP4 infers the q/kv split from parent row count ∈ {14336, 10240, 5120} |
| `gdn_input_proj` | NVFP4, FP8 (1 shape), W8, Q4+Q5 pair | |
| `linear_swiglu` | Q4 (1 shape), W8 (5 shapes), NVFP4, FP8 (1 shape) | |
| `sparse_moe` | three codec profiles: Q4+Q5/Q6, W8+W8, NVFP4+NVFP4 | two geometries registered; NVFP4 = TRT-LLM cutlass, **sm_120a builds only**; six side arrays required |
| `expert_slot_cache` | W8 only | |
| `cpu_expert_compute` | W8G32, Q4G32AM (not a `QType`) | |
| `embedding` | BF16, Q6, W8, FP8 | **no NVFP4, no Q4/Q5** |

NVFP4 needs `n % 128 == 0 && k % 64 == 0` with no padding path
(`nvfp4_format.cpp:40`). `artifact::materialized_weight` **throws** for
NVFP4 (`typed_binding.cpp:171-175`); every target re-implements
`bind_nvfp4_weight`. Two global-scale conventions coexist: the dense 27B
path copies the checkpoint word and the engine divides; the routed 35B path
writes the reciprocal and forces the divisor to identity
(`require_identity_divisor`).

## 3. Open questions — answered by the surveys

- [x] **Fate of the `.sinfer` container: keep it as a transparent cache;
  retire the per-model inventories.** Owner directive, 2026-08-24
  (`design/serve-engine-plan.md:260-269`): "The engine's supported inputs
  are safetensors (HF repos, incl. GPTQ/AWQ/FP8/NVFP4 variants) and GGUF.
  The `.ninfer` container is NOT a supported input format … an internal,
  transparent, regenerable acceleration of the load-time repack — never an
  interchange format, never published, never required (`--no-cache` serves
  without it)." "Sealed `.ninfer` container with per-model converter
  inventories" is a **REJECT** verdict by name (`:143`), in favor of
  "HF-repo-first ingest, and geometry-parameterized admission" — the
  eight-entry hardcoded registry in `ingest.py:70-91` is the rejected shape.
  The repack itself cannot be eliminated, only relocated: the ported
  kernels' layouts are the calling convention (`:34`), and Marlin residency
  must be pre-baked because deriving it at load costs a second copy that
  does not fit at 27B (`serve-engine-multiarch.md:90-108`). Drift to fix
  along the way: `--no-cache` is unimplemented, `surogate convert` does not
  exist, and `surogate/serve/tools/README.md` still tells users to download
  artifacts from Hugging Face — the rejected posture — while linking three
  files that do not exist.
- [x] **Fusion at load time** — rule stated in §2.
- [x] **Descriptor carries no group size / closed format registry.** Confirmed
  (§2a). Decision: the descriptor grows explicit `{bits, type, group,
  scale_dtype, layout}` fields rather than one enum entry per combination
  — the closed enum is what made every new export a code change. This also
  supplies the per-tensor *layout tag* `serve-engine-multiarch.md` asks for
  (`MarlinTiles` exists only as a runtime stamp with no `StorageLayout`).
- [x] **GGUF scope.** Four types repack bit-exactly, all onto `W8G32_F16S`:
  Q8_0, Q4_0, Q5_0, IQ4_NL (`gguf_repack.py:110-115`). Everything else —
  every K-quant, Q4_1, Q5_1 — is dequantized to BF16 on disk and
  re-quantized, a documented double quantization (`bridge.py:19-20`),
  even though `Q4G64`/`Q5G64`/`Q6G64` kernels exist. Whether a K-quant can
  land losslessly on them is itself open: K-quant sub-blocks carry a
  6-bit scale *and* a 6-bit min (asymmetric) inside a 256-element
  super-block, while `Q4G64_F16S` is symmetric per 64 — so exactness needs
  either a min plane on the GPU formats (the CPU `Q4G32AM` shape) or
  accepting the lossy path with the loss printed. See M3.
- [?] **Load-time budget.** 0.8B ≈ 0.6 s, 4B ≈ 1.9 s, 27B ≈ 7.7 s, 35B ≈ 11 s
  warm (`design/ROADMAP.md:146-153`), with the 2.0 GiB/s copy called "a
  loader artifact, not a floor" (pinned bounce buffer → ~1.5–2.5 s). No test
  gates load time and the benchmark board excludes it. A repack-at-load
  design must not regress these; the cache is what keeps it from being paid
  per start.

## 4. Milestones

Each has one falsifiable acceptance check. Order is chosen so every step is
verifiable against something on disk.

### M1 — Generic compressed-tensors NVFP4 (format axis, A)

Acceptance: `Qwen3.6-35B-A3B-NVFP4-redhat-vllm` converts and serves **from
that one file** with zero code naming it, answers "Paris.", and the
per-tensor formats in the artifact equal what `compressed_tensors` resolves
from its `quantization_config`.

- [ ] Detection: compressed-tensors parsed in `hf_config.py` through the library; `ingest.py:64`'s substring test removed.
- [ ] Resolver: meta-device skeleton + checkpoint-derived module names, library matcher, `ignore` first by name (proven exact, §4a). Output is library-native `QuantizationScheme` per module; the map to storage format is a separate, small table.
- [ ] Converter: structure from the architecture's mapping, format from the resolver; `_validate_nvfp4_config`'s single-group and `_validate_source_manifest`'s histogram assertions gone.
- [ ] C++: `Binder` gains a read-format path; shape still asserted; routability check per op.
- [ ] C++: `WeightsProfile` removed from the family interface; `resolve_weights` collapses; the planner takes formats from the load plan.
- [ ] C++: one shared NVFP4 `Weight` constructor replacing the per-target `bind_nvfp4_weight` copies and the `materialized_weight` throw.
- [ ] C++: merge `bind_nvfp4_text_layers` / `bind_nvfp4_uniform_text_layers` into the structural binder; 0.8B and 2B gain NVFP4 for free — verify by loading a synthetic-format artifact for each.
- [ ] Kernels: **K = 2048 NVFP4 activation quantizer.** The 35B-A3B's attention and shared-expert Linears are K = 2048 (`attention/query_key_gate_value` 9216×2048), which is not in the generic set; generalize the per-K instantiation or add the size. n = 9216 satisfies `% 128`.
- [ ] One global-scale convention: checkpoint word stored verbatim, engine divides, direction decided by `quant_method` (compressed-tensors divides; ModelOpt inverts) — never by tensor names. Retire the reciprocal-and-identity path.
- [ ] Fusion policy (§2): constituents resolved individually; unfused binding when scales differ; test with a synthetic export that quantizes `q_proj` and ignores `k_proj`.
- [ ] `embedding` has no NVFP4 path — this file ignores `lm_head`, so not blocking; record the gap.
- [ ] MoE on this file: routed experts through the existing TRT-LLM profile (sm_120a); refuse loudly elsewhere.

### M2 — One architecture, one directory (geometry axis, B)

Acceptance: `targets/qwen3_5/` exists, `qwen3_5_{0_8b,2b,4b}/` do not, all
three artifacts still answer identically to before (same greedy output on
the same prompt, captured against the commit that precedes the change),
105/105 tests pass.

- [ ] `Geometry<…>` parameter pack; `Qwen35<Geometry>` variant; three instantiations in one `package.cpp`.
- [ ] Route choices as `if constexpr` on geometry; literals replaced by named geometry expressions.
- [ ] Registry: one alias line per size; `ActiveTarget` derives (already does for executors, `bda92164`).
- [ ] Route tables gated on literal hidden sizes (`design/serve-engine-multiarch.md:24-30` names this as the violation): a shape off the table must degrade to the generic path with the tier printed, not throw at plan time.
- [ ] Repeat for the 27B family (`qwen3_6_27b` + `qwen3_8_27b` share a target already).

### M3 — Generic GGUF

Acceptance: the Qwen3.5-0.8B in **Q8_0, Q4_0 and Q4_K_M** (all on disk)
loads through one generic path, each tensor lands in the format the table
in §4b assigns to its GGML type, the bit-exact ones are verified bit-exact
against gguf-py's dequantize, the lossy ones print their measured error,
and greedy output on a fixed prompt is compared against the
safetensors-derived 0.8B artifact.

- [ ] GGML type → runtime format table, per tensor, from the header — never a file-level "quant type".
- [ ] Decide the K-quant landing: a min plane on the GPU int formats (exact) vs dequant + requant with the loss printed. Measure both on `Qwen3.5-0.8B-Q4_K_M`.
- [ ] Architecture and hyperparameters from GGUF metadata (`bridge.py` reads `{arch}.embedding_length`, `.block_count`, head counts, ssm.* already); static `config.json` is vendored per target under `serve/resources/` today and must instead be derived.
- [ ] Tokenizer from GGUF metadata: BPE only today (`frontend.py:45-56`, `tokenizer.ggml.pre` must be in a literal table); SentencePiece GGUFs refused — connect to the tree's own SPM tokenizer (`csrc/src/tokenizer`).
- [ ] Architecture table `gguf_target_key` (`bridge.py:298-327`) is literal; replace with the same architecture registry M2 produces.
- [ ] The bridge materializes the whole model as BF16 on disk before converting (2 B/param — ~200 GB of temp for the 106 GB Flash-Next). Reader injection instead.
- [ ] Export-transform inversion (`gguf/qwen35.py`: norm `+1`, `-exp(A_log)`, conv1d squeeze, V-head tiling) is per family and hand-written; make it part of the architecture's mapping.

### M4 — Unified weight loading (C)

Acceptance: `csrc/src/serve/artifact/` no longer contains a safetensors or
quantized-layout decoder that also exists under `csrc/src/utilities/` or
`csrc/src/recipes/`; both trainer and serve link the same reader; the
container remains as the transparent cache the directive describes, and
`--no-cache` serves without it.

- [ ] Serve targets' structure comes from the DSL `hf_mapping` (§2 C.1).
- [ ] Serve reads safetensors through `SafeTensorsReader`; retire `convert/common/safetensors.py` and the hand-rolled parser in `serve/lora_registry.cpp:30-117`.
- [ ] Trainer reader fixes taken on adoption: O(n) `find_entry` (`safetensors.cpp:317-321`), `load_tensors` silently skipping mismatches both ways (`:281-289`), non-copyable reader with raw back-pointers (`safetensors.h:69,100`), narrow cast table (`cu_file_common.cpp:92-114`), unchecked `getenv("HOME")` (`:661-671`).
- [ ] Trainer's compressed-tensors gap closed by the same M1 work (`data_suffix` `weight_packed`, `weight_global_scale`, `input_global_scale`).
- [ ] One NVFP4 stored-layout codec (pending §4c).
- [ ] `--no-cache` direct path and `surogate convert` subcommand — both promised by the plan (`serve-engine-plan.md:277-284`), neither exists.
- [ ] Load-time guard: the warm numbers in §3 recorded as a test threshold before the reader changes.

### M5 — Cleanup

- [ ] Delete per-file converters superseded by M1/M3 (`convert_nvfp4.py`, `convert_nvfp4_all.py`, `routed_nvfp4.py`'s refusal-only checks).
- [ ] `TODO.md`'s parked converter-driver hoist: close as superseded.
- [ ] `PATCHES.md` entries that describe per-export workarounds: retire.
- [ ] `surogate/serve/tools/README.md`: stale NInfer text, three dead links, the rejected download-artifacts posture. Rewrite or delete.
- [ ] `ingest.py` refusal strings list different model sets at `:173` and `:463`.

## 4a. M1 findings

- **Meta-device skeleton is the mechanism for scheme resolution.**
  `AutoModelForImageTextToText.from_config` under `torch.device('meta')`
  builds `Qwen3_5MoeForConditionalGeneration` in 1.1 s: 1,119 modules,
  35.1B params, zero bytes. `match_named_modules(model, targets, ignore)`
  then runs the library's own `targets`/`ignore` semantics (class name,
  exact name, `re:` regex) with no re-implementation.
- **Resolution is exact where the skeleton and the checkpoint agree on
  structure.** 160 modules resolved (10 full-attention layers × q/k/v/o +
  40 × shared-expert gate/up/down); all 160 are packed in the file; zero
  false positives.
- **The one seam: fused experts.** transformers 5.16.1 represents the 256
  routed experts per layer as one fused module; the checkpoint stores them
  per-expert (`experts.N.{gate,up,down}_proj.weight_packed`, 40 × 256 × 3 =
  30,720). The matcher therefore never sees a per-expert `Linear`.
  Principle that follows: **the checkpoint's tensor names are the authority
  for structure; the config is the authority for format.** Resolution has
  to run over checkpoint-derived module names, with class taken from the
  skeleton where the name exists and from the fused module's constituents
  where it does not.
- **Done that way, resolution is exact on the whole file.** Module names
  derived from the three shard headers (31,233 candidates: 503 with a class
  from the skeleton, 30,720 expert leaves, 10 outside the skeleton);
  `match_targets` applied per name with the library's own semantics:
  **resolved 30,880 = packed 30,880, zero either way.** No heuristic on
  tensor names decides format; the config does.
- Design rules that fall out:
  1. Apply `ignore` first and by name — it needs no class. The 10 modules
     outside the skeleton are all `mtp.*`, caught by `re:^mtp.*`; a
     resolver that demanded a class for them would fail on a file it can in
     fact serve.
  2. `targets` needs a class only for class-name targets like `Linear`;
     take it from the skeleton, and for leaves of a fused module from the
     module it fuses.
  3. transformers is pinned `>=5.5.0` (`pyproject.toml:29`) with a
     `transformers_v5_compat` patch hook, so the fused-expert
     representation is the one to design for, not the legacy per-expert
     one.
  4. **`fused:` is not for the expert seam — it is for ours.** Its
     semantics (`compressed_tensors/utils/match.py::is_match`) are vLLM's
     `packed_modules_mapping`: `{"qkv_proj": ["q_proj","k_proj","v_proj"],
     "gate_up_proj": ["gate_proj","up_proj"]}`, so a *runtime-fused* module
     name can be matched against `targets`/`ignore` entries written for
     its constituents. Our runtime fuses exactly this way. It is how the
     resolver detects the mixed-scheme parent the fusion rule in §2 refuses.

## 4b. M3 findings (GGUF on disk)

`/home/densemax2/work/flavius/surogate/models/` holds real GGUF files (the
HF-cache entries are 1 MB stubs, never downloaded). Per-tensor type
histograms read from the headers:

| File | Size | arch | Tensor types |
|---|---|---|---|
| `Qwen3.5-0.8B-Q8_0.gguf` | 795 MB | qwen35 | Q8_0:195, F32:140 |
| `Qwen3.5-0.8B-Q4_0.gguf` | 484 MB | qwen35 | **Q4_0:129, Q8_0:36, Q5_K:18, Q4_1:3, Q6_K:1**, F32:133 |
| `Qwen3.5-0.8B-Q4_K_M.gguf` | 508 MB | qwen35 | Q4_K:98, Q5_K:36, Q8_0:36, Q6_K:17, F32:133 |
| `Qwen3.5-2B-Q4_K_M.gguf` | 1.2 GB | qwen35 | (K-quant mix) |
| `Qwen_Qwen3.5-9B-Q5_K_M.gguf` | 6.8 GB | qwen35 | Q5_K:125, Q8_0:44, Q6_K:41, F32:232 |
| `embeddinggemma-300M-Q8_0.gguf` | 319 MB | gemma-embedding | Q8_0:171, F32:145 |
| `Qwen3.8-Flash-Next-UD-Q4_K_XL-*` (4 shards) | ~106 GB | qwen4exp | Unsloth Dynamic, mixed per tensor |
| `GLM-5.3-Flash-UD-Q4_K_XL-*` (6 shards) | ~190 GB | glm5next | Unsloth Dynamic, mixed per tensor |

- **Every file mixes types, not just the Unsloth-Dynamic ones.** The
  "Q4_0" 0.8B holds five quant types. A GGUF loader keyed on a file-level
  quant type is the same mistake as a per-export NVFP4 profile; the
  per-tensor type in the header is the authority.
- **The 0.8B Q4_0 is not bit-exact today either**: its 3 Q4_1 and 19
  K-quant tensors take the dequant → requant path.
- **M3 acceptance ladder: the 0.8B in Q8_0, Q4_0, Q4_K_M.** Same model
  three ways, and a safetensors-derived 0.8B artifact already exists to
  cross-check greedy output against.

## 4c. M4 findings

- [~] NVFP4 scale swizzle: trainer (`quant_fp4.cu:1512-1529`) vs serve
  (`layouts.py:466-483`, `nvfp4_config.h:181-188`). Being compared.

## 5. Surveys feeding this document — all complete

- [x] Trainer loader capabilities — folded into §0, §2 C, M4.
- [x] Serve quant coverage — folded into §2a, M1, M3.
- [x] Container rationale and owner directives — folded into §3.

## 6. Progress log

- 2026-09-02 — Redesign scoped. Prior boilerplate hoists (`f1a869eb`,
  `3673030c`, `a0d40f05`, `89d96c64`, `bda92164`) treated the symptom; this
  removes the cause. Converter-driver hoist parked as superseded.
- 2026-09-02 — Scheme resolution proven exact on the RedHatAI 35B-A3B NVFP4
  (30,880 = 30,880) with library code only.
- 2026-09-02 — Three surveys complete; open questions in §3 answered from
  the owner's own directives. Container stays as cache; per-model
  inventories go; the trainer's DSL mapping is the structure source.
