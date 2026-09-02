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
  list of hundreds of module names plus regexes (`re:^mtp.*`,
  `re:.*linear_attn\.in_proj_.*`). It would be refused on nearly every
  clause. `routed_nvfp4.py:151` already states the right principle —
  "`quantization_config` is the authority, not the tensor names" — but uses
  it only to refuse.
- **The spec and its matcher are installed.** `compressed-tensors 0.17.0` in
  the venv: `QuantizationConfig.from_pretrained`, `match_targets`,
  `match_named_modules`, `is_narrow_match`. Re-implementing the `ignore`
  regex semantics would repeat the safetensors mistake below.
- **Weight loading is duplicated.** The trainer has
  `csrc/src/utilities/safetensors.{h,cpp}` (881 lines: `SafeTensorsReader`,
  `SafeTensorWriter`, HF hub resolution, cuFile, NCCL-aware writes) and a
  full `csrc/src/recipes/nvfp4/` tree. Serve has its own `artifact/`
  (1,239 lines) behind a bespoke `.sinfer` container fed by a Python
  converter per model.
- **A real test file is on disk.**
  `/home/densemax2/work/models/hf/Qwen3.6-35B-A3B-NVFP4-redhat-vllm`: 24 GB,
  124,325 tensors, 30,880 `weight_packed` + 30,880 `weight_global_scale`,
  three shards. Nothing in the tree can load it.

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

What each preset needs from the runtime, to be filled from survey §2:

- [ ] NVFP4 / NVFP4A16 — fp4 pairs, fp8 per-16 block scales, fp32 global scale (compressed-tensors: global scale *divides*; ModelOpt exports invert it)
- [ ] FP8 / FP8_DYNAMIC / FP8_BLOCK — e4m3 with per-tensor / per-channel / block scales
- [ ] W4A16 / W4A16_ASYM / W4A8 — int4 `pack-quantized`, group strategy, optional zero points and actorder
- [ ] W8A8 / W8A16 / INT8
- [ ] MXFP4 / MXFP8 — e8m0 block scales
- [ ] `kv_cache_scheme` — refuse or support; do not ignore silently

## 2. The design

Three independent axes. Today all three are collapsed into "one directory
per model size, one profile enum per export, one converter per file".

**A. Format is data, read per tensor.** Structure (which tensors exist, what
shape the geometry implies) stays compiled in. Format comes from the
checkpoint's `quantization_config`, resolved with compressed-tensors' own
matcher, and is carried per tensor. The binder checks shape and "can an op
route this format", not "is this the format I was compiled for".
`WeightsProfile` disappears; the per-export `bind_*` functions merge;
`make_sequence_planner` takes formats from the load plan (they exist by
then: `plan_load` runs at `targets/registry.cpp:206`, the planner at `:219`).

**B. Geometry is a template parameter.** One directory per architecture;
a size is an instantiation (`Qwen35<Geometry<2048, 24, 6144, 8, 2, …>>`);
route choices that depend on shape (`linear` vs `linear_pair` for a 512-row
kv block — `qwen3_5_2b/impl/variant.cpp:183`) become `if constexpr` on the
geometry. Keeps specialization, build-time shape checks, and the baked
tables. Magic numbers that are already named constants (`5120`/`10240` =
`mtp_attention_input_rows`) become expressions again.

**C. One weight loader.** Serve reads safetensors through the trainer's
`SafeTensorsReader`, and shares one set of quantized-layout decoders with
the trainer, rather than maintaining `artifact/` + a converter per model.

Constraint on all three: the format assertion that goes away was a loud
failure mode. Its replacement is shape assertion + routability check, not
nothing.

## 3. Open questions

- [?] **Fate of the `.sinfer` container.** Options: (i) keep it as a
  load-time cache that a generic ingest builds from any HF/GGUF source;
  (ii) load HF safetensors and GGUF directly at engine start, fusing and
  repacking in memory as vLLM/llama.cpp do, container gone. Materially
  different work. Decision needs the container's stated rationale
  (survey §3: mmap layout? fusion? startup time?) and the owner's call.
- [?] **Fusion at load time.** The runtime depends on fused tensors
  (`query_key_value`, `gate_up`, `query_key_gate_value`). Under NVFP4 each
  constituent has its own global scale — the "fused-divisor trap" — so
  fusing quantized constituents is not a concatenation. Where does fusion
  live once conversion is generic: converter, loader, or runtime?
- [?] **`TensorSpec` carries no group size** — it is encoded in the format
  enum (`Q4G64_F16S`, `W8G32_F16S`). Fine for NVFP4 (always
  `tensor_group`/16). Any preset with a different strategy or group size
  needs the container/descriptor to carry it, or an enum entry per
  combination. Decide which before extending.
- [?] **GGUF scope.** Which GGUF tensor types must execute natively vs be
  repacked vs dequantized on load (survey §2 answers what exists today).

## 4. Milestones

Each has one falsifiable acceptance check. Order is chosen so every step is
verifiable against something on disk.

### M1 — Generic compressed-tensors NVFP4 (format axis, A)

Acceptance: `Qwen3.6-35B-A3B-NVFP4-redhat-vllm` converts and serves with
**zero code naming that checkpoint**, answers "Paris.", and the per-tensor
formats in the artifact equal what `compressed_tensors` resolves from its
`quantization_config`.

- [ ] Converter: resolve module → scheme via `QuantizationConfig.from_pretrained` + the library matcher; emit per-tensor `NumericFormat` from the resolved scheme; drop `_validate_nvfp4_config`'s single-group and `_validate_source_manifest`'s histogram assertions.
- [ ] Converter: a generic "which HF name maps to which artifact object" layer that does not depend on the export's format (structure from the architecture, format from the config).
- [ ] C++: `Binder` gains a read-format path; shape still asserted; new routability check per op.
- [ ] C++: `WeightsProfile` removed from the family interface; `resolve_weights` collapses; the planner takes formats from the load plan.
- [ ] C++: merge `bind_nvfp4_text_layers` / `bind_nvfp4_uniform_text_layers` into the structural binder; 0.8B and 2B gain NVFP4 for free — verify by loading a synthetic-format artifact for each.
- [ ] Global-scale direction handled from `quant_method` (compressed-tensors divides), never from tensor names.
- [ ] Fused constituents with distinct global scales: resolved (see open question) and tested.

### M2 — One architecture, one directory (geometry axis, B)

Acceptance: `targets/qwen3_5/` exists, `qwen3_5_{0_8b,2b,4b}/` do not, all
three artifacts still answer identically to before (same greedy output on
the same prompt), 105/105 tests pass.

- [ ] `Geometry<…>` parameter pack; `Qwen35<Geometry>` variant; three instantiations in one `package.cpp`.
- [ ] Route choices as `if constexpr` on geometry; literals replaced by named geometry expressions.
- [ ] Registry: one alias line per size; `ActiveTarget` derives (already does for executors, `bda92164`).
- [ ] Repeat for the 27B family (`qwen3_6_27b` + `qwen3_8_27b` share a target already).

### M3 — Generic GGUF

Acceptance: a GGUF file not previously converted (pick one from
`study/` or the HF cache) loads through the same generic path, executes its
native quant types where kernels exist, and refuses loudly, naming the
tensor and type, where they do not.

- [ ] GGUF type → runtime format table, from survey §2's coverage matrix.
- [ ] Architecture + hyperparameters read from GGUF metadata, not from a sibling `config.json`.
- [ ] Tokenizer from GGUF metadata.

### M4 — Unified weight loading (C)

Acceptance: `csrc/src/serve/artifact/` no longer contains a safetensors or
quantized-layout decoder that also exists under `csrc/src/utilities/` or
`csrc/src/recipes/`; both trainer and serve link the same reader.

- [ ] Blocked on open question: container fate.
- [ ] Serve reads safetensors through `SafeTensorsReader`.
- [ ] One set of NVFP4 / FP8 / int-group layout decoders shared with `recipes/`.

### M5 — Cleanup

- [ ] Delete per-file converters superseded by M1/M3 (`convert_nvfp4.py`, `convert_nvfp4_all.py`, `routed_nvfp4.py`'s refusal-only checks).
- [ ] `TODO.md`'s parked converter-driver hoist: close as superseded.
- [ ] `PATCHES.md` entries that describe per-export workarounds: retire.

## 4a. M1 findings (running)

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
     module it fuses. `match_named_modules(fused=…)` ("mapping from
     suffixes of fused modules to the suffixes of their corresponding
     shards", see `compressed_tensors.utils.match.is_match`) is the
     library's own form of this and is preferred over a hand-written
     expert regex once its exact shape is confirmed.
  3. transformers is pinned `>=5.5.0` (`pyproject.toml:29`) with a
     `transformers_v5_compat` patch hook, so the fused-expert
     representation is the one to design for, not the legacy per-expert
     one.

## 5. Surveys feeding this document

- [~] Trainer loader capabilities (`SafeTensorsReader` API, pre-quantized loading, `recipes/nvfp4/` inputs, HF-name mapping, GGUF)
- [~] Serve quant coverage (QType × op matrix, NVFP4 runtime representation, GGUF support, container layouts and fusions, `ingest.py`)
- [~] Container rationale and owner directives (`design/*.md`, `PATCHES.md`, documented user flow, layout-dependent tests)

## 6. Progress log

- 2026-09-02 — Redesign scoped. Prior boilerplate hoists (`f1a869eb`,
  `3673030c`, `a0d40f05`, `89d96c64`, `bda92164`) treated the symptom; this
  removes the cause. Converter-driver hoist parked as superseded.
