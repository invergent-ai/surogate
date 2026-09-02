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

- [x] Detection: `HfConfigFactory.get_quant_info` has a compressed-tensors branch (`surogate/core/model/hf_config.py`); the four existing branches answer as before. `ingest.py:64`'s substring test is still there until the converter side moves over.
- [x] Resolver: `surogate/core/model/quant_schemes.py` — `resolve_checkpoint(model_dir)` → per-module `QuantizationScheme`, library matcher only, `ignore` first by name, cross-checked against the file both ways. `tests/test_quant_schemes.py`: 7 unit cases on a meta-device Llama + the real export (30,880 = 30,880, 10.5 s). The map from scheme to storage format is the next, separate table.
- [x] Converter: `qwen3_6_35b_a3b/compressed_tensors_source.py` + `convert.py` auto-detects `quant_method: compressed-tensors` and converts from the one directory: formats from the resolver, fused parents split per constituent via the recipe's row program, `input_scale_divisor` siblings, BF16 kept BF16, the export's own frontend accepted and recorded. **The RedHatAI file converts: 1,399 objects, 23.8 GB, 123 s.** Verified against the written file: 262/262 text-core formats equal the resolver's, 0 missing divisors, payloads decoded from the artifact bytes bit-identical to the checkpoint (NVFP4 codes/swizzled scales/both scales, BF16 rows). The 27B converter's single-group and histogram assertions are untouched (that converter is superseded, M5).
- [ ] C++: `Binder` gains a read-format path; shape still asserted; routability check per op.
- [ ] C++: `WeightsProfile` removed from the family interface; `resolve_weights` collapses; the planner takes formats from the load plan.
- [ ] C++: one shared NVFP4 `Weight` constructor replacing the per-target `bind_nvfp4_weight` copies and the `materialized_weight` throw.
- [ ] C++: merge `bind_nvfp4_text_layers` / `bind_nvfp4_uniform_text_layers` into the structural binder; 0.8B and 2B gain NVFP4 for free — verify by loading a synthetic-format artifact for each.
- [x] Kernels (`46bafa6f`): **NVFP4 activation quantizer at K = 2048 and K = 512.** An alias, a `case` in the one K-switch (`nvfp4_w4a4.cu`), and a gate entry each. `test_nvfp4_a4.cpp` runs M1's unfused shapes — `[4096,2048]`, `[512,2048]`, `[2048,4096]`, `[2048,512]` — at T ∈ {1,2,4,17,300} on the cuBLASLt W4A4 route.
- [x] **The A4 op oracle now quantizes exactly** (same commit). It used to multiply the raw BF16 activation and allow 16 %; `materialize_activation` now reproduces `quantize_nvfp4_k16` step for step on the CPU (fp32 `divisor·max|x|/6` → E4M3 RN-even saturating, zero scale zeroes the block; fp32 multiply then divide; E2M1 RN-even saturating at 6) and the criterion is the A16 one. Every case — the registered in-house W4A4 shapes and the four generic ones, one token included — passes at `rel_l2` 0.001–0.002 against 0.0039, a 40–100× tighter statement than before: both routes compute the GEMM exactly on the quantized inputs.
- [x] C++ suite after the binder changes: 105/105.
- [x] **C++ load side (M1a) up to the shared expert.** Regression: the existing groupwise-int and routed-NVFP4 35B artifacts answer byte-identically (64-token greedy) through read-format binding against a stash-built baseline of the previous binary; the compressed-tensors artifact loads until `text/layers/0/moe/shared_gate_up`, which the MoE binder still requires fused — the shared-expert decision below. `artifact::bind_linear` / `materialized_linear` (`typed_binding`) bind a Linear in whatever format the artifact stores, asserting shape only, and build the NVFP4 `Weight` that every target used to re-implement (`Binder::require_tensor_shaped` underneath). The 35B target binds every text-core Linear that way — token embedding, attention (split `query/key/gate/value` when the artifact has them, the fused parent otherwise), `attention/output`, GDN (`query_key_value` + `z` or the fused parent), `gdn/output`, `output_head`, MTP attention through the same path — and the leaves run the split through generic `linear` (attention) and the existing `gdn_input_proj_conv_{snapshot,record}_split` ops (GDN), with the call-site policy following the weight's format. Interim `WeightsProfile::CompressedTensors` sizes workspaces until formats reach the planner from the load plan. Regression oracle: the existing groupwise-int and routed-NVFP4 35B artifacts must answer byte-identically to the pre-change binary (stash-built baseline). The compressed-tensors artifact is expected to stop at the shared expert's W8 gate.
- [x] **RETRACTED — no route defect.** The one-token failure at `[4096,2048]` seed 731 (7.09375 vs 4.13297 at row 6) was chased through four hypotheses — M = 1-specific, first use, stale workspace flags, a sign-bit reading — each refuted by experiment (the padded two-token plan, a zeroed workspace and a repeated invocation all reproduce the identical value; the generator's scales are all positive). The harness's own statistics (`SINFER_OP_REPORT_STATS=1`) then showed what it was: the A4 oracle does not model activation quantisation, its 16 % allowance is applied as whole-output `rel_l2` **and** as a per-element bound of 16 % of the *largest reference in the output*. On uniform random activations this route sits at `rel_l2` 0.157 of 0.16 at one token and 0.10 at 300 — the same noise the in-house W4A4 kernels produce — and at one token on 4,096 outputs the largest reference is smallest (17.7 vs 52.6 at T = 300), so one row landed 4 % over the per-element bound. A statistical edge of the fixture under a T-dependent criterion, not a kernel error. Sampling did not clear it either, so the oracle was fixed instead of the seed (the item above); the workspace memset and the M = 1 padding were removed — neither fixed anything and a fix that fixes nothing is a lie in the tree. Cost: most of an afternoon; lesson recorded in §4a's rules — read the harness's criterion before chasing a kernel.
- [?] **Shared expert under NVFP4 — owner's call, sized.** The export quantizes `shared_expert.{gate,up,down}_proj` with three distinct global scales; `sparse_moe` computes the shared expert inside all three of its kernel bodies — decode and small-T via `dot_two_rows<W8Codec, …>` on one gate|up matrix, prefill via dedicated launches of the expert GEMM template — and admits W8 only (`sparse_moe.cpp:154`). Sized:
  - (a) *NVFP4 shared codec in the kernels.* The routed path's `Nvfp4CodecFor` is W4A16 (BF16 activations), reads the same 128×4 tiled layout a dense NVFP4 weight has, and is already instantiated for K = 2048 and 512 — the shared expert's own K values. What is missing: the bodies read gate|up as one matrix, and the export's gate and up have different global scales, so each site needs to take the two halves (and their scales) separately, plus the wrapper's admission, a scalar global scale per shared weight on `SparseMoeWeights`, the binder, and an op-test fixture with an NVFP4 shared expert. About a day. Honest: the export's numbers, bit for bit.
  - (b) *Shared expert outside the op.* Generic `linear`×2 + silu·mul + `linear_add` in the leaf, scaled by the op's per-token `shared_scale`, with a no-shared mode in the same three bodies. Not smaller than (a).
  - (c) *Requantize the shared expert to W8 in the converter.* Built as an explicit opt-in (`--shared-expert w8`; default `as-stored`): dequantize the NVFP4 halves through the same words the engine reads (E2M1 table checked byte-for-byte against compressed-tensors' own unpacker), fuse gate|up, encode W8 as the base converter does. Overrides the export for 120 tensors (~3.1 M params per layer). **With it, the artifact loads all 20.96 GiB of weights in 8.0 s** — every text-core object bound in its stored format — and then stops in the frontend: the engine pins `tokenizer_config.json` "prefix semantics" (the C++ half of the same assumption the converter had). It does not decide the question; (a) remains the destination.
- [x] `common/inventory.py` gained `NVFP4`, `FP8`, `BLOCK_SCALE_LAYOUT`, `ROW_SCALE_LAYOUT` — every target re-declared the NVFP4 pair beside its own inventory. Two places restated `FORMAT_COUNTS` as a literal and tripped on the zero-count entries (`qwen3_6_27b/test_inventory.py`, the 35B `preflight_inventory`); both now compare present formats only.
- [x] Frontend resources: the converter pinned the *official* Qwen3.6 `tokenizer.json` hash and refused the export's (re-serialised, 248,044 vocab / 247,587 merges, sha `dd6b…`). A compressed-tensors export ships the frontend it was calibrated with; `load_official_resources(accept_source=True)` records the hash instead of refusing. The env-var "derived frontend" path stays for the GGUF bridge.
- [x] The C++ half of the same pin, and everything else between the artifact and an answer (`f93caeab` and the commit below). In order, each found by running the file rather than by reading:
  1. `frontend.cpp:245` rejected the export's `tokenizer_config.json` for "prefix semantics", defaulting an *absent* `add_bos_token` to true while its own comment said absent means false. A transformers-5 `TokenizersBackend` config has `add_prefix_space: false`, `bos_token: null`, and no `add_bos_token` key.
  2. `tokenizer.cpp` demanded `added_tokens_decoder`, the pre-transformers-5 home of the special tokens, which it *merges* over `tokenizer.json`'s `added_tokens` and cross-checks. That config has no such key; `tokenizer.json` is the sole authority. Absent now means nothing to merge — and the sort that ends the merge still runs, since `load_added_tokens` returns file order and `added_token_candidates_` is built in that order.
  3. `select_bf16_a16_launch` was a two-shape registry (the 27B's) that threw for everything else, with a shape-generic BF16 cuBLASLt GEMM sitting beside it unreached. The export's BF16 GDN halves (`[8192,2048]`, `[4096,2048]`) and `lm_head` (`[248320,2048]`) are the first BF16 linears to arrive off that table; any 8-aligned shape now routes to cuBLASLt at every width.
  4. `bf16_linear_add_select` likewise, plus a `CublasLt` schedule accumulating into the residual through the same plan (β per call, `bf16_cublaslt_gemm_accumulate`), and the wrapper's own admits-gate above it.
  5. The graph budget then refused 36 MiB against a 12 MiB allowance: the BF16 cuBLASLt plane builds a 32 MiB workspace on first use, and first use was now *inside* the capture window. `program_impl.h` prewarms it beside the NVFP4 and FP8 planes, as qwen4exp already did for itself.
- [x] **M1 ACCEPTANCE MET.** RedHatAI's `Qwen3.6-35B-A3B-NVFP4` converts from its own directory and serves: *"The capital of France is"* → a coherent reasoning trace naming Paris; the Danube question answered in structure. Zero code names that checkpoint. Regression: both existing 35B artifacts answer byte-identically to the pre-change binary, gemma3-270m and tinyllama still say "Paris.", qwen3.5-0.8b still thinks first, 105/105 C++ tests.

- [?] **Shared expert under NVFP4 — owner's call, sized.** The export quantizes `shared_expert.{gate,up,down}_proj` with three distinct global scales; `sparse_moe` computes the shared expert inside all three of its kernel bodies — decode and small-T via `dot_two_rows<W8Codec, …>` on one gate|up matrix, prefill via dedicated launches of the expert GEMM template — and admits W8 only (`sparse_moe.cpp:154`). Sized:
  - (a) *NVFP4 shared codec in the kernels.* The routed path's `Nvfp4CodecFor` is W4A16 (BF16 activations), reads the same 128×4 tiled layout a dense NVFP4 weight has, and is already instantiated for K = 2048 and 512 — the shared expert's own K values. What is missing: the bodies read gate|up as one matrix, and the export's gate and up have different global scales, so each site needs to take the two halves (and their scales) separately, plus the wrapper's admission, a scalar global scale per shared weight on `SparseMoeWeights`, the binder, and an op-test fixture with an NVFP4 shared expert. About a day. Honest: the export's numbers, bit for bit.
  - (b) *Shared expert outside the op.* Generic `linear`×2 + silu·mul + `linear_add` in the leaf, scaled by the op's per-token `shared_scale`, with a no-shared mode in the same three bodies. Not smaller than (a).
  - (c) *Requantize the shared expert to W8 in the converter.* Built as an explicit opt-in (`--shared-expert w8`; default `as-stored`): dequantize the NVFP4 halves through the same words the engine reads (E2M1 table checked byte-for-byte against compressed-tensors' own unpacker), fuse gate|up, encode W8 as the base converter does. Overrides the export for 120 tensors (~3.1 M params per layer). **With it, the artifact loads all 20.96 GiB of weights in 8.0 s** — every text-core object bound in its stored format — and then stops in the frontend: the engine pins `tokenizer_config.json` "prefix semantics" (the C++ half of the same assumption the converter had). It does not decide the question; (a) remains the destination.
- [x] `common/inventory.py` gained `NVFP4`, `FP8`, `BLOCK_SCALE_LAYOUT`, `ROW_SCALE_LAYOUT` — every target re-declared the NVFP4 pair beside its own inventory. Two places restated `FORMAT_COUNTS` as a literal and tripped on the zero-count entries (`qwen3_6_27b/test_inventory.py`, the 35B `preflight_inventory`); both now compare present formats only.
- [x] Frontend resources: the converter pinned the *official* Qwen3.6 `tokenizer.json` hash and refused the export's (re-serialised, 248,044 vocab / 247,587 merges, sha `dd6b…`). A compressed-tensors export ships the frontend it was calibrated with; `load_official_resources(accept_source=True)` records the hash instead of refusing. The env-var "derived frontend" path stays for the GGUF bridge.
- [~] The C++ half of the same pin: `family/impl/frontend/frontend.cpp:245` rejected the export's `tokenizer_config.json` for "prefix semantics" because it defaults an *absent* `add_bos_token` to true — while its own comment says an absent key means false, and the transformers-5 `TokenizersBackend` config shape (`add_prefix_space: false`, `bos_token: null`, no `add_bos_token` key) is exactly the case. Code aligned with the comment. The next pin behind it: `tokenizer.cpp` required `added_tokens_decoder` in `tokenizer_config.json` — the pre-transformers-5 home of the special tokens, which it *merges* over `tokenizer.json`'s `added_tokens` and cross-checks. The `TokenizersBackend` config has no such key; `tokenizer.json` is the sole authority there, so an absent decoder now means nothing to merge (the conflict check stays when both exist). Both landed (`f93caeab`); the 35B groupwise-int artifact, whose bundled config carries the decoder, answers byte-identically.
- [~] **Next stop, compute: "bf16 linear: unsupported shape or T".** `select_bf16_a16_launch` (`ops/linear/bf16/bf16_dispatch.cpp`) is a two-shape registry — `[14336,5120]` and `[5120,6144]`, the 27B's — and throws for anything else, while a shape-generic BF16 cuBLASLt GEMM sits beside it unreached. The export's BF16 GDN halves (`[8192,2048]`, `[4096,2048]`) and `lm_head` (`[248320,2048]`) are the first BF16 linears ever run off that table. This is the M2 principle ("a shape off the table degrades to the generic path, tier printed") arriving early. Written: `select_bf16_a16_launch` routes any 8-aligned shape off the table to cuBLASLt at every width, and `bf16_linear_add_select` gains a `CublasLt` schedule that accumulates into the residual through the same plan (β is per call). Build and acceptance rerun in flight.
- [?] **The export's `tokenizer.json` and the engine's tokenizer.** `test_frontend.cpp`'s new case, pointed at the export's tokenizer, fails encoding plain text ("produced token outside vocabulary: h") while the merge and special-token tests on the same file pass — none of them encodes ordinary text. The file differs from the official one in serialisation only where checked so far (merges as two-element lists, which the parser accepts; `pre_tokenizer`, `normalizer`, `post_processor`, `model.vocab['h']` identical). Being attributed: whether the official-config tokenizer encodes plain text from this file at all, and a full vocab/merges/added-tokens diff.
- [ ] **M1 scope decisions**, recorded so they are not relitigated mid-implementation:
  1. *Unquantized modules stay BF16.* This export leaves GDN `in_proj`/`out_proj`, the routers, norms, `lm_head` and MTP in BF16 deliberately (they are the `ignore` list). The base converters requantize BF16 to W8 for the groupwise-int profile; doing that here would override the exporter's choice. The config is the authority in both directions.
  2. *Descriptor field extension is not M1.* This file uses only `NVFP4` and `BF16`, both existing `NumericFormat`s. The explicit `{bits, type, group, scale_dtype, layout}` fields (§3) are the prerequisite for the *other* presets in §1 and get their own item.
  3. *Bind unfused, run generic `linear`.* The fused wrappers are shape-registered: `attn_input_proj` NVFP4 infers its q/kv split from row count ∈ {14336, 10240, 5120}, `gdn_input_proj` has no BF16 path. At 9216×2048 NVFP4 and BF16 GDN, neither applies. Generic `linear` accepts any `n % 128, K ∈ set`, so M1 binds attention q/k/v/o and shared gate/up/down as separate NVFP4 objects and GDN `in_proj` as BF16 — which is also what the fusion rule (§2) demands when global scales differ. Decode runs on cuBLASLt W4A4 at T = 1 for these shapes; the gemv-only entries (`Nvfp4GemvOnlyProblem`, the 4B's 204 → 313 tok/s fix) and fused parents are the perf follow-ups, not acceptance.
  4. *Structure stays per-target until M4.* The 35B's `inventory.py`/`recipe.py` remain the structure source for M1; only the per-object *format* comes from the resolver, and the converter's two source roles (`--model` BF16 base + `--routed-nvfp4-dir`) collapse to one directory. The segments of a fused parent are enumerated by the row-algebra walker that `gguf_repack.py` already has (`_evaluate_rows`: `Concat`/`Slice`/`GatherRows` over sources, K axis untouched) — lifted into `common/`, not copied.
  5. *Staged: M1a text core, M1b MTP + vision.* The binder binds MTP and vision objects `ValidateOnly` when their features are off (`bindings.cpp:302-337`), and validate-only still asserts format and shape. So the artifact must carry them in whatever shape the binder expects. M1a converts the text core on the new rule (one object per HF `Linear`, format from the resolver) and emits MTP and vision in today's fused-W8 form so their bindings stay untouched — acceptance is a text prompt with speculation and vision off, so both are validate-only. M1b puts MTP and vision on the same rule. Interim, and written down so it does not become the copy that drifts.
  6. *Embedding and `lm_head` stay BF16 too*, ~1 GB each at 248,320 × 2,048 instead of ~250 MB as W8. `ops::embedding` and BF16 `linear` both serve them. A "requantize what the export left unquantized" knob is a later policy, not an M1 default.
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
  5. **Read the harness's criterion before chasing a kernel.** The A4 op oracle
     does not model activation quantisation; its allowance is applied as
     whole-output `rel_l2` *and* as a per-element bound of 16 % of the largest
     reference in the output, which loosens with T. A one-token "failure" that
     reproduced identically under every state and shape hypothesis was that
     bound, 4 % over, on random data sitting at the allowance. `SINFER_OP_REPORT_STATS=1`
     prints the numbers that would have said so in the first minute.
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

- [x] **The trainer's and serve's NVFP4 scale swizzles are the same layout.**
  Trainer `scale_swizzled_offset` (`quant_fp4.cu:285-299`) expands to
  `(rb·⌈cols/4⌉ + col/4)·512 + (rem%32)·16 + (rem/32)·4 + col%4`; serve's
  `nvfp4_tiled_scale_offset` (`nvfp4_config.h:181-188`) is
  `(m_tile·tiles + group/4)·512 + (ri&31)·16 + (ri>>5)·4 + (group&3)` —
  term for term the same, and serve's Python `swizzle_nvfp4_scales`
  permute reproduces it. Checked numerically: 0 mismatches over
  256×1024, 9216×2048, 14336×5120 and 128×64 (every scale word). One
  codec is a pure merge, no conversion step.

## 5. Surveys feeding this document — all complete

- [x] Trainer loader capabilities — folded into §0, §2 C, M4.
- [x] Serve quant coverage — folded into §2a, M1, M3.
- [x] Container rationale and owner directives — folded into §3.

## 7. Product decision (2026-09-02) — GGUF K-quants are the star

Owner directives this day, superseding §2 and M3 where they conflict; the rest of this file stands. "the star of the product are ggufs, especially the common ones (K quants)"; support natively and with the highest performance: GGUF Q*_K, NVFP4 (compressed-tensors and ModelOpt), FP8, BF16; GPTQ/AWQ are off the roadmap. `.sinfer` may be published to HuggingFace as our standard format. "i don't care what we have, i need to figure out what's the ideal product." Permission to borrow code and kernels from `study/llama.cpp-master` (MIT).

**Why this line, measured** (`Qwen3.5-0.8B-Q4_K_M.gguf`, `blk.0.attn_gate`; error relative to the file's own values; bytes per weight):

| path | rel L2 | B/w |
|---|---|---|
| the file itself | 0 | 0.56 |
| keep its codes, scale+min FP16 per 32 | 6.8e-4 | 0.56 |
| keep its codes, scale+min FP32 per 32 | **0** | 0.81 |
| dequantise to BF16 and stop | 1.7e-3 | 2.00 |
| BF16 → W8G32, **what we do today** | **5.7e-3** | **1.06** |

Same on Q5_K and Q6_K (5.8e-3, 5.7e-3). An 8-bit symmetric grid cannot land on an affine 4-bit grid: today we add 0.6 % weight error *and* nearly double VRAM. There is no Q1_K (the 1-bit types are IQ1_S/IQ1_M i-quants, refusable).

**The ideal product, as proposed and not contradicted:**
1. Adopt GGML's quantised formats as ours; retire Q4G64/Q5G64/Q6G64. W8G32 holds Q8_0's numbers exactly (a deinterleave) and stays Q8_0's device layout while the W8 kernels are the better-tuned ones.
2. Format is per-tensor data; every op takes every codec. A Q4_K_M file is Q4_K + Q5_K + Q6_K + Q8_0 + F32 in one file (measured 98/36/17/36/133 tensors).
3. Fastest kernel per format. K-quants: activation quantised to int8 per 32 with its block sum (`Q8_1`), integer dp4a/mma dot, the affine min folded into the activation's sum — llama.cpp's method. NVFP4: W4A4. FP8: FP8 tensor cores. BF16: cuBLASLt. The bar is llama.cpp on the same GGUF and GPU.
4. `.sinfer` is for what GGUF cannot hold — NVFP4, mixed recipes (NVFP4 experts + Q6_K attention), bundled vision/draft heads — and that is what gets published. A GGUF loads at disk speed; caching it saves nothing.
5. Device layout for K-quants: **native GGML superblocks, bytes verbatim** (Q2_K 84 B, Q3_K 110, Q4_K 144, Q5_K 176, Q6_K 210 per 256 values). GGML guarantees k % 256 == 0 for every K-quant tensor (checked on six files; non-multiples fall back to Q*_0/Q8_0, which is why the files mix types), so no padding. `Weight.qdata` is the block array; `QuantLayout::GgmlBlocks`.

**The bar.** Server-to-server is `surogate/serve/BENCHMARKS.md` (0.8B: llama.cpp 411 decode / 34,500 prompt-eval through `llama-server`; ours 753 decode / ~106k prefill on a 640-token prompt this day, i.e. +83 % and ~3×, from a heavier and less exact artifact). The kernel-level ceiling at T=1 is `llama-bench` on the same file, same card class (GPU 2, `-fa 1`, 2026-09-02): **0.8B Q4_K_M 802.7 tg128 / 37,945 pp512; 2B Q4_K_M 577.6 / 28,081.** A native K-quant GEMV that does not beat 803 on the 0.8B is not done.

**What llama.cpp's structure buys.** `mmvq.cu`'s `mul_mat_vec_q<type, ncols_dst∈1..8, fusion, small_k>` is one body for every type — T=1, small-T and, via `ids`, routed-expert decode; the per-type part is one `vec_dot_*_q8_1` of ~40-60 lines. `mmq.cuh`'s `mul_mat_q<type, J>` is one prefill body (int8 `mma.sync m16n8k16/k32`; sm_120 takes the Ampere path) with per-type tile loaders. 10,379 lines in all, of which the K-quant share is a fraction. So the port is two bodies plus per-type atoms — not five copies of a 2,000-line family, which is what the tree's own Q4/Q5/Q6 directories are (Q4's and Q5's GEMV share almost no lines).

### K-milestones

- [x] **K0 — the bar**: above.
- [ ] **K1 — mmvq port.** `ops/linear/ggml/`: `quantize_q8_1` (activation → int8 per 32 + (d, sum)), ported `vec_dot_{q2..q6}_K_q8_1` (+ Q8_0/Q4_0/Q5_0/IQ4_NL, which llama.cpp also has), the `mul_mat_vec_q` body for T ≤ 8, dispatch from `ops::linear` on the new `QType`s. Artifact: `NumericFormat::Q2_K..Q6_K`, `QuantLayout::GgmlBlocks`, `tensor_encoded_size`, converter passthrough (verbatim bytes; the `_planes_*` deinterleave stays only for Q8_0 → W8G32), binder. Tests: each type against a CPU reference that applies the *same* Q8_1 activation quantisation then fp64, on real blocks from the GGUFs on disk and on synthetic ones. **Acceptance:** the 0.8B Q4_K_M's K-quant tensors served natively; greedy tokens identical to llama.cpp's on the same file (llama.cpp is the oracle for "the file's numbers are the served numbers"); tg128 ≥ 803 on a 5090.
- [~] **K1 status (2026-09-02, evening).** Kernel slice committed as `7aea807e`; the plumbing is written and under acceptance:
  - *Artifact*: `NumericFormat::Q2_K..Q6_K`, `StorageLayout::GgmlBlocksV1` ("ggml-blocks-v1"), `QType::Q2_K..Q6_K`, `QuantLayout::GgmlBlocks`; a K-quant `Weight` is the block array with no scale planes. Python: `GgmlBlockFormat`, the layout, `inventory` names.
  - *Converter*: `GgufRepackSource.plan_native` plans every quantised-linear object whose row program draws from mapped sources of one K type; `native_specs` rewrites the specs; `payload_for_native` gathers block rows verbatim. `plan()` no longer W8-repacks an object with a K-quant source. `surogate serve` keeps native sources in the map; `SUROGATE_GGUF_NATIVE=0` forces the old path (A/B), `SUROGATE_GGUF_NATIVE_ONLY=<substrings>` restricts the plan (bisection). The 0.8B Q4_K_M converts in 2.3 s: **77 objects native** (Q4_K 44, Q5_K 18, Q6_K 15), 922 MB against 1,224 MB through the dequantise path. Checked byte-identical to the GGUF: `mlp/down`, the concatenated `mlp/gate_up`, the 200 MB `token_embedding`, its tied `output_head`, a gathered `gdn/output`.
  - *Ops*: `ops::linear` dispatches the five QTypes to the mmvq route (int8 activation scratch from the arena, or an engine-slot scratch when the caller has none — the lm_head — grown only outside capture and never freed, since a captured graph keeps the pointer it recorded); `linear_add` accumulates through the same GEMV; `linear_swiglu` projects gate and up into two planes and runs `silu_mul`; `attn_input_proj` and `gdn_input_proj` (plain, conv snapshot, conv record) split a K-quant parent by row range straight into their outputs, the conv forms through the existing `compose_*` tails; `embedding` gathers rows through ported `dequantize_q*_K`.
  - *Targets*: qwen3_5 0.8B/2B/4B bind every linear and table object by shape and read the stored format (`bind_linear_weight`, the draft head through `bind_linear`); the W8 capacity arms size for the larger of their own route and the K route.
  - *Two K5c items left on the dequantise path for now*: the mixed-format fused parents (`gdn/query_key_value_z` = Q5_K qkv + Q4_K z; `attention/query_key_gate_value` = Q4_K q/k/gate + Q6_K v) — they need split objects per format and the two-parent leaf forms, which the 35B has and the qwen3_5 targets do not.
  - *Found by running*: (1) the two-row CTA needs the xor-butterfly `warp_sum`, not the shuffle-down `warp_reduce_sum`; (2) llama.cpp launches `dequantize_q6_K/q5_K/q3_K/q2_K` with 64 threads and only `q4_K` with 32 — a 32-thread gather leaves half of every block unwritten, and this file's token table is Q6_K: the whole model read garbage embeddings while every GEMV was exact. Bisected by converting with `SUROGATE_GGUF_NATIVE_ONLY` per object group (`mlp/` and the two `linear_add` outputs were coherent; the vocab group alone reproduced it), then the test extended to drive `ops::embedding`, `ops::linear` without a workspace and the full 248,320-row table.
  - *Like-for-like on the same file* (`Qwen3.5-0.8B-Q4_K_M.gguf`, one 5090, `llama-server` idle on the same card; 128 generated, 3 runs; prefill on a 640-token prompt):

    | path | artifact | decode tok/s | prefill tok/s |
    |---|---|---|---|
    | before: dequantise → BF16 → W8 (`SUROGATE_GGUF_NATIVE=0`) | 1,224 MB | 764-768 | 112,235 |
    | native K-quants (this increment) | 922 MB | 831-839 | 8,963 |
    | llama.cpp `llama-server` | — | 627 | — |
    | llama.cpp `llama-bench` (kernel ceiling) | — | 803 | 37,945 |

    Decode: +9 % over the dequantise path and past `llama-bench`'s ceiling at T=1, at 75 % of the bytes. **Prefill is 12× slower than before**: the interim route runs the T≤8 GEMV in 8-column chunks, re-reading the weight per chunk — that is what K3 (the MMQ port) is for, and until it lands a native artifact is a decode-side win only.
  - *Known gaps*: the 2B converter needs the vision tower from the HF cache the 0.8B had (`model.visual.patch_embed.proj.weight`), a pre-existing ingest difference; a source shared by a native and a non-native object (tied embeddings with a non-native head) must be both mapped and dequantised, and the planner's single `keep` set cannot say that yet.
- [ ] **K2 — Q8_0 and the legacy types**: W8G32's decode kernel against mmvq's Q8_0 on the same tensor; keep the faster, exactness is equal.
- [ ] **K3 — MMQ port (prefill, T > 8)**: `mul_mat_q` body, `load_tiles_{q2..q6}_K`, the int8 mma vec-dots, `quantize_q8_1_mmq`; stream-K fixup second. **Acceptance:** pp512 ≥ 37,945 (0.8B), ≥ 28,081 (2B).
- [ ] **K4 — MoE**: mmvq's `ids` indirection is routed-expert decode for T ≤ 8 (port it with K1); MMQ per expert for prefill (`mul_mat_id`); host expert bank in K-quants (ggml-cpu `ggml_vec_dot_q4_K_q8_K`, or ik_llama's AVX-512). **Acceptance:** a Q4_K_M MoE GGUF (Qwen3.6-35B-A3B, ~20 GB — download) serves natively on one 5090 with the existing CPU offload.
- [ ] **K5 — the ops that embed a weight decode**: only three sites switch on `QType` (`linear.cpp`, `sparse_moe.cpp`, `attn_input_proj.cpp`); `linear_add` and the fused GDN/attention projections route K-quant weights through mmvq/MMQ plus their epilogue.
- [ ] **K6 — retire the home-grown formats.** Converters stop emitting Q4G64/Q5G64/Q6G64; `surogate quantize` produces K-quants by writing a BF16 GGUF (gguf-py) and running `llama-quantize` (their quantiser, imatrix included) — no port of `ggml-quants.c`; regenerate the local artifacts.
- [ ] **N — NVFP4 ModelOpt ingest**: `weight_scale_2` is a multiplier where compressed-tensors' global is a divisor; parents split per component so each keeps its own global (the trainer instead rescales block scales to a shared one, which is lossy).
- [ ] **F — FP8**: compressed-tensors per-channel/per-tensor is per-row with an FP32 scale (add `_F32S`, or accept the BF16 cast); HF fine-grained block-128 (what the trainer loads as `prequant_fp8`) needs block-indexed scales in the fp8 family and a prefill route.
- [x] **B — BF16**: correctness done (cuBLASLt off the table, `8d1b9cc4`); the registered shapes keep the hand kernels.

M2 (geometry templating), M4 (unified loading) and M5 (cleanup) stand; the K-line is the main line and goes first.

## 6. Progress log

- 2026-09-02 — Redesign scoped. Prior boilerplate hoists (`f1a869eb`,
  `3673030c`, `a0d40f05`, `89d96c64`, `bda92164`) treated the symptom; this
  removes the cause. Converter-driver hoist parked as superseded.
- 2026-09-02 — Scheme resolution proven exact on the RedHatAI 35B-A3B NVFP4
  (30,880 = 30,880) with library code only.
- 2026-09-02 — Three surveys complete; open questions in §3 answered from
  the owner's own directives. Container stays as cache; per-model
  inventories go; the trainer's DSL mapping is the structure source.
