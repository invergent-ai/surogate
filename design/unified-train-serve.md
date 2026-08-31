# One framework: what actually duplicates between train and serve

The goal is a single codebase serving both training and inference across many
architectures. This document measures where the duplication really is, because
the answer is not where it looks.

## Measured today

| | files | lines |
|---|---:|---:|
| training kernels (`csrc/src/kernels`) | 40 `.cu` | 20,559 |
| serve ops (`csrc/src/serve/ops`) | 138 `.cu` | 18,956 |
| serve targets (`csrc/src/serve/targets`) | 84 | 23,126 |
| DSL model declarations (`surogate/dsl/models`) | 10 `.py` | 3,471 |

Cost of adding one architecture:

- **Training**: one DSL declaration. `qwen3_vl` 156 lines, `lfm2` 241,
  `laguna` 363, `nemotron_h` 471, `qwen3_5` 486, `qwen3_5_moe` 534,
  `gemma4` 717.
- **Serving**: a hand-written C++ target of ~1,700 lines, plus — for anything
  that is not a Qwen variant — a sibling to the 14,648-line `qwen3_6`
  implementation, which bakes in Qwen's mixer schedule.

That asymmetry is why training covers seven architecture families and serving
covers Qwen only.

## The finding

Two whole serve targets, 1,723 lines each, **share about 82% of their lines**.
Measured file by file with difflib rather than a shell diff (an earlier
revision of this document said 75 lines and 96%; that came from diffing the
`impl` directory without `-r`, which counts only the summary lines — the
figures below are the corrected ones):

    CMakeLists.txt         8 lines,   0 changed
    impl/config.h         93 lines,  20 changed
    impl/load/bindings.cpp 605 lines, 218 changed
    impl/load/bindings.h  225 lines,  14 changed
    impl/package.cpp      143 lines,  14 changed
    impl/variant.cpp      519 lines,  35 changed
    impl/variant.h        130 lines,   8 changed
    TOTAL               1,723 lines, 309 changed  (18%)

Of the 309, about 80% carry a shape constant or a quoted tensor name, and the
rest are the namespace and include renames that follow from the target's name:

    -namespace ninfer::targets::qwen3_5_0_8b::detail {
    +namespace ninfer::targets::qwen3_5_4b::detail {
    -    out.gate_up = materialized_weight(materialized, plan.gate_up, 7168, 1024);
    +    out.gate_up = materialized_weight(materialized, plan.gate_up, 18432, 2560);

So the architecture-specific content of a serve target is a config struct and a
weight table; the other ~1,400 lines are ceremony that a generator should be
emitting. The conclusion is unchanged from the earlier revision — 82% is still
overwhelming duplication — but the number is now one that can be reproduced.

## What to unify, in order of payoff

**1. Generate serve targets from the DSL declaration.** The DSL is already the
source of truth for what a model *is*, and it already describes every shape the
serve target restates. A build-time generator that emits the target from the
same declaration removes ~1,400 of every 1,723 lines and makes "train it, then
serve it" one declaration instead of two implementations. This is the single
biggest win and it needs no kernel work.

**2. De-Qwen the shared implementation.** `targets/qwen3_6` (14,648 lines) is
not a model — it is the engine's forward pass wearing a model's name. What
makes it Qwen-specific is a small set of assumptions, chiefly the mixer
schedule (which layers are full attention and which are gated delta net) and
the fused projection layouts. Those belong in the declaration, so Gemma, GLM
and Kimi instantiate the engine rather than fork it. Until this lands, every
non-Qwen architecture pays a five-figure line count.

**3. Converge kernels only where semantics match.** Norms, RoPE, MoE routing
and quantisation are the same mathematics in both stacks and are duplicated
(serve has 12 norm files, training 3; MoE 12 and 16). Attention is *not* a
convergence candidate: training needs varlen plus backward, serving needs
paged KV plus single-token decode, and forcing one kernel to do both would cost
more than it saves. Converge the first group, leave the second deliberately
separate, and write that decision down so it is not relitigated.

## Relationship to the three throughput workstreams

`design/serve-engine-multiarch.md` specifies fused decode attention, the async
scheduler contract, and native Marlin residency. All three are stated so that a
new architecture declares a shape rather than writing a kernel, which is the
same principle as this document at a smaller scale. The first landing
(PATCHES.md #54, head dimension becoming a geometry parameter rather than a
file-scope 256 with model-named aliases) is step one of both plans at once.

The sequencing between the two efforts matters: **generation before
optimisation**. Optimising the current targets means hand-tuning code that is
about to be generated, and the tuning would have to be re-expressed in the
generator anyway. The exception is work that changes the *abstraction* — the
geometry parameterisation, the round-lifecycle contract — because the generator
will emit against those interfaces and they should be right first.

## Progress: the generator

`surogate/serve/tools/generate/` holds the first piece.

  target_spec.py     one architecture described once: shapes, head geometry,
                     linear-attention dims, tensor names. Validates what C++
                     would otherwise discover as a template error — head
                     grouping, head-dim multiples, rotary within head dim.
  emit_config.py     emits impl/config.h from a spec.
  check_roundtrip.py regenerates the committed targets and diffs.

Correctness is established by regeneration against the committed targets as
fixtures, not by review. Both qwen3_5_0_8b and qwen3_5_4b reproduce byte for
byte through the emitted section. A generator that cannot reproduce what it
replaces has not earned the right to replace it, and the check is what makes
deleting the hand-written copies safe later rather than hopeful.

It has already paid: the first run rejected the 0.8B declaration, where two
values had been written from memory (layers 28 for 24, key heads 8 for 16).
Wrong constants in a serve target do not fail to build — they produce a model
that loads and generates nonsense, which is the failure mode this session spent
hours chasing from the other direction.

Next, in order: extend emission through the derived block and the MTP section
of config.h; then bindings.h/bindings.cpp, which is where 218 of the 309
differing lines live and where the tensor-name mappings from the DSL
(`_hf_block_mappings_`) replace hand-written tables; then read the spec from
the DSL declaration itself rather than restating it in check_roundtrip.py, so
that a model trained by the DSL is servable without a second description.

### What does not work: text templating

The obvious way to generate `bindings.cpp` — take a committed target, replace
its shape values with placeholders, render for other models — is unsound, and
the check that proves it is worth keeping.

Rendering the derived template for the model it came from always succeeds, so a
single-model check would have passed it. Rendering for a second model fails on
28 lines, every one a constant that shares a value with a shape:

    const std::uint64_t low_group = 32;    32 is a group size, not value_heads
    materialized_weight(..., 2560, 1024)   2560 is another variant's branch

A shape and a constant that coincide are indistinguishable to substitution, and
the failure is silent for the source model. So `bindings.cpp` has to be emitted
from structure — a weight table whose entries carry their own rows and columns
computed from the spec — as `emit_config.py` does for `config.h`, rather than
patched from an existing file. The rejected attempt is retained under
`rejected_text_template*.py` so the next person does not spend the afternoon
rediscovering it.

### What does work: de-literalising, not generating

The templating failure pointed at a better design. `config.h` is already
generated and already names every shape a target needs, so `bindings.cpp` does
not need to be generated at all — it needs to stop restating those shapes as
literals.

`deliteralize.py` rewrites the shape literals in a target's `bindings.cpp` into
the `TextConfig` expressions that equal them, confining substitution to the
argument lists of the calls that carry weight shapes so that a group size or a
vocabulary constant sharing a value with a shape is left alone. It is
value-preserving by construction: a literal is only ever replaced by an
expression equal to it in that target's own config, so anything ambiguous stays
literal rather than being guessed at.

Result on the two Qwen variants: **309 differing lines fall to 38**, of which 12
are the target's own name in namespaces and includes. Both targets compile.

The 26 that remain are worth reading rather than papering over, because they are
places where the two targets genuinely disagree:

    0.8b: materialized_weight(..., split->query_key, 2560, TextConfig::hidden)
    4b:   materialized_weight(..., split->query_key, TextConfig::hidden, ...)

A literal 2560 in the 0.8B, whose hidden size is 1024. Either that path is dead
for this target or it carries a wrong constant; either way de-literalising made
a discrepancy visible that reading 605 lines of near-identical code would not.
Resolving those is the remaining work before `bindings.cpp` can move to the
shared implementation and stop being copied per target.

### Where de-literalising landed

Applied across the three Qwen3.5 variants, with the tool grown to handle what
the first pass got wrong:

  qwen3_5_0_8b vs qwen3_5_4b   218 differing lines -> 34 (18 beyond the name)
  qwen3_5_2b   vs the others   still ~120, its remaining literals ambiguous

Three corrections were needed along the way, each caught by comparing against a
second model rather than by review:

  The value map was a dict literal, so two config quantities sharing a value
  silently overwrote each other — the 2B's intermediate and convolution_dim are
  both 6144, its hidden and query_size both 2048. Built as pairs now, so a
  collision survives to be refused instead of being resolved at random.

  Refusing ambiguous values is correct but leaves them literal. Two kinds of
  rule resolve most of them: the call's own text (a weight named "mlp/down" has
  intermediate columns) and the weight's identity fixing what each dimension
  means (`plan.down` is hidden rows by intermediate columns), because rows and
  columns cannot be told apart by value.

  The dead `SplitAttentionProjectionPlan` branch, which only the 27B ever
  constructs, carried shape constants copied from a sibling target — a hidden
  size that was not the target's own. Unreachable code, so nothing caught it.
  It now throws in the three Qwen3.5 targets rather than materialising wrong
  extents if it ever becomes reachable.

That last one is the argument for this whole exercise in miniature: the wrong
constant sat in three targets, was invisible to review, and only surfaced when
two files that should agree were made to agree.

### De-literalising, finished pass

  qwen3_5_0_8b vs qwen3_5_4b   218 differing lines -> 30 (14 beyond the name)
  qwen3_5_2b   vs either        ~70 (54 beyond the name)

Two more rules were needed, and one of them was wrong first:

  A weight's trailing dimension is its input width, so an ambiguous value
  sitting there means hidden. Except in `row_view`, whose trailing argument is
  a row COUNT — applying the rule there rewrote a count as hidden, which is
  correct by value on the 2B (where hidden and query_size are both 2048) and
  wrong in meaning. `row_view` is excluded now.

  The MTP packed attention layout (q | k | gate | v) is expressed symbolically
  in all three targets: offsets 0, q, q+kv, 2q+kv and counts q, kv, q, kv.
  Verified against the pre-refactor literals — all twelve extents in all three
  targets reproduce exactly, so the change is value-preserving rather than
  merely plausible.

The 2B keeps more literals than its siblings because more of its quantities
coincide: hidden and query_size are both 2048, intermediate and
convolution_dim both 6144. Every remaining literal there is one the tool
refused rather than guessed, which is the intended behaviour.

### Weight shapes by name — the form that actually closes it

Value-based substitution cannot resolve a literal whose value coincides with
another quantity, and positional rules cannot tell rows from columns without
knowing which weight is being bound. The artifact names every weight, and a name
fixes both dimensions exactly: "mlp/down" is hidden rows by intermediate
columns, whatever those equal for a given model.

`weight_shapes.py` rewrites shapes by name and closes the residue:

  qwen3_5_0_8b vs qwen3_5_4b   218 differing lines -> 30 (14 beyond the name)
  qwen3_5_0_8b vs qwen3_5_2b                        -> 34 (18)
  qwen3_5_2b   vs qwen3_5_4b                        -> 36 (20)

Verified value-preserving the only way that counts: all 29 named weights in each
of the three targets were compared against their pre-refactor literals — 87
shapes, zero mismatches. A shape table keyed by name is also the form the
generator should emit from, so this pass is a step toward emission rather than a
detour around it.

Regression after the whole series, 100 users, 90 seconds, coherent at
temperature 0: 0.8B 6,692 tok/s, 4B 3,032 tok/s, both zero errors.
## Progress: the qwen4_exp DSL declaration (2026-08-31)

The declaration-first half is now real for Flash-Next:
`surogate/dsl/models/qwen4_exp.py` (+ `blocks/qwen4_exp.py`,
`modules/hyper_connection.py`) compiles the full 48-layer model — 36 GDN + 12
gated-attention layers, 512-expert top-10 MoE — in 0.1 s, and
`tests/test_qwen4_exp_dsl.py` holds the contract on CPU with no weights.

What it took beyond composition of existing parts: hyper-connections (the
model's replacement for every layer norm — four residual streams, mix/combine
around each sublayer, the final mix standing in for the output norm) are
expressed in pure DSL ops, no new kernels; the GDN output gate gained a
`gate_activation` attribute through primitive → emitter → C++ forward/backward
because Flash-Next gates with sigmoid where Qwen3.5 uses SiLU; and the router
renormalises its top-10 (`norm_topk_prob=True`), unlike the Qwen3.5-MoE
declaration it descends from. Three traps worth recording: HF stores
`hc_norm.weight` zero-centred (the GGUF exporter's `norm.weight` +1 rule
catches it, which is why GGUF-side references see a plain gamma), the block
schema rejects routing metadata unless `block_family` contains "moe", and the
checkpoint is Conditional-only (`model.language_model.*`) with a `.weight`
suffix on every hyper-connection tensor.

Validated structurally against the Hub's tensor index: the mapping covers the
checkpoint exactly in both directions — every unused tensor is in a deliberate
deferral (PLE 137, indexer 36, vision 333, MTP 31), zero unexpected. The
deferrals are captured as config (ints reach the runtime config, so the future
generator sees them) but not yet in the training graph; PLE is the one that
matters — it is load-bearing in the forward pass, so training the real
checkpoint is not numerically faithful until its three small primitives land
(n-gram row-index input, signed-sqrt gate, dilated causal conv).

Numeric parity is the next step, and it cannot go through transformers: no
released or dev version ships `qwen4_exp`, and the checkpoint carries no remote
code. The path is the one the serve side already built — the GGUF-fed CPU
references in `surogate/serve/tools/parity/qwen4exp/` against
`SUROGATE_DEBUG_DUMP_TENSORS` dumps.

## Progress: the DSL becomes the source (2026-08-31, step ①)

`from_dsl.py` reads a declaration and produces the `TargetSpec` the emitters
consume, so the shape constants a serve target states are now the values
training actually compiles rather than a transcription of them. The two inputs
are the ones training already uses — `surogate/dsl/models/*.py` for the
architecture, the checkpoint's own `config.json` for this instance's sizing —
and the handful of scalars the declaration neither derives nor interprets
(`rope_theta`) are read from the config and named in `CONFIG_PASSTHROUGH`, so
the places where the DSL is *not* yet the source are countable.

`check_roundtrip.py` no longer restates the values it checks: it builds both
specs through `from_dsl` and still reproduces `qwen3_5_0_8b` and `qwen3_5_4b`
`config.h` byte for byte, which is what retires the hand-written literals.

For targets nobody intends to generate yet there is a second, weaker-looking but
stricter check. `qwen4exp/impl/config.h` is hand-written and carries prose that
records why values are what they are; a generator forced to reproduce that prose
would relocate the duplication rather than remove it. So `check_contract.py`
parses the literal `static constexpr` values out of the committed header and
compares them against the declaration. On Flash-Next: **30 constants agree, none
disagree** — hidden, layer schedule, both head geometries, hyper-connection count
and rank, all three MoE widths, all four indexer fields, every PLE quantity
including the 1-based-to-0-based layer conversion, and the MTP count. Four
constants sit outside the contract by design (`eos_token`, the IQ4_NL table
facts, the `rope_theta` pass-through), and `ple_embed` is checked through its
inputs because the header derives it.

That result is worth more than the check: the declaration written yesterday and a
serving target written independently, tuned and benchmarked, agree on every
architectural quantity. Two constants of that parse — `DFlashConfig` reuses the
name `layers` — is also why the parser scopes to `struct TextConfig`; flattening
the file silently substituted a draft-head constant for the text stack's.

**LoRA is carried from the start.** `TargetSpec.params` holds every declared
parameter with its checkpoint path and its adapter slices, lifted from the IR's
`lora_targets` — 72 adapter-addressable parameters on the 0.8B, 96 on the 4B. The
slices already carry the fused-projection offsets (`mlp_up_weight` is
`[(up, 0, 3584), (gate, 3584, 3584)]`), which is exactly what serving LoRA needs
in order to apply an adapter trained on one logical projection to the right row
range of a fused serve tensor. Serving has no LoRA math and no weight-update path
today, so this is preparation, not capability — but it means the contract will
not have to be re-derived when GRPO needs it.

Next: bindings.h/cpp from the same contract (218 of the 309 differing lines),
then the converter inventory, then the train/serve numeric parity test, with
GLM-5.3-Flash written declaration-first as the acceptance test.

## Progress: the third description joins the check (step ②)

Step ② set out to emit `bindings.h/cpp` from the contract. Measuring first
changed the target. The two dense targets' bindings are **identical** in the
header and differ by 122 lines in the source, of which ~8 are the namespace and
the rest are one extra quantisation profile the 4B export happens to support.
The shapes are not restated there at all: they are already written as
`TextConfig::` references, so they arrived from the declaration the moment step ①
generated `config.h`. Generating `bindings.cpp` would have bought almost nothing
and would have pulled residency and profile policy — genuinely serving decisions
— into the generator.

The duplication is somewhere else. Artifact object names (`gdn/query_key_value_z`,
`attention/query_key_gate_value`) are written twice: once in the C++ binder as
`prefix + leaf`, once whole in the converter's `inventory.py`. And the converter
restates the geometry in its own constants — `LAYERS = 48`, `HIDDEN = 2560`,
`HC_COUNT = 4` — a third independent copy of what the declaration compiles. A
converter and a binder that disagree about a fused row count produce a
hundred-gigabyte artifact that fails at load; one that disagrees about a layer
index produces an artifact that loads and is quietly wrong.

So `check_contract.py` now checks both consumers against the declaration. On
Flash-Next: **30 header constants and 35 converter constants agree, none
disagree**, across 986 declared artifact objects. The converter table includes the
derived fused row counts (`ATTENTION_FUSED_ROWS`, `GDN_FUSED_ROWS`, `HC_WIDTH`,
`ROUTER_ROWS`) precisely because those are what a load-time shape error is made of.

`tests/test_serve_contract.py` turns both checkers into tests that need no GPU, no
weights and no built extension. The drift detector was verified by injecting
drift: changing `hc_low_rank` to 256 in the committed header turns the suite red
with `DISAGREE hc_low_rank: header=256 declaration=320`. A guard that has never
failed has not been shown to guard anything.

Emitting bindings is still worth doing, but as the *shared* implementation the two
dense targets already almost are, not as a per-target generator — and after the
converter inventory, which is where the object names actually live.

## Three descriptions become one

The question that reframed this work: if the DSL is the source of truth, why does
the serving policy live somewhere else? The objection to putting it there — that
formats and layouts are deployment concerns, not architecture — does not survive
contact with the code. The declaration already carries `quantizable` (71
references), `residency` (43), `offload_group` (27), `streaming_hint` (17);
`SlotDecl` has residency and streaming fields; and `MoESharedExpert` declares
`quantizable=False` with the comment *"Mirrors router weights being kept full
precision"*, which is a numeric-format decision recorded in the declaration years
before this exercise. Artifact names and serving formats are the same kind of
thing, not a new kind.

So `ServeObject` now sits in `block_schema.py` beside `SlotDecl`: an artifact
object's name, numeric format, shape, the declared parameters that compose it *in
row order*, and the name of the repacking the converter applies when the
composition is not a plain concatenation. Blocks carry their objects on
`BlockSchema.serve_objects`; the model carries the handful outside the stack.

`emit_inventory.py` expands that over the layer schedule. Against the committed
converter for Flash-Next: **986 tensors emitted, 986 committed, every name, shape
and numeric format identical.** The converter's hand-written inventory is not a
second description of the model — it is exactly what the declaration implies, and
can be deleted in favour of the emitter.

What deliberately stays hand-written is the repacking itself: untiling GDN value
heads, unfolding a norm's folded `+1`, splitting an interleaved query/gate
projection. Those are algorithms, not data. The declaration names them
(`ServeObject.transform`) so the set in play is visible from the model, and stops
there.

The component lists are the second dividend. `attention/query_key_gate_value`
records that it is built from the query, key and value projections under the
`split_interleaved_query_gate` repacking; `gdn/query_key_value_z` that it fuses
the qkv projection with the gate's z projection. That is the map an adapter needs
to land on the right rows of a fused serve tensor — the thing serving LoRA cannot
be written safely without, and the reason it was worth declaring now rather than
reconstructing later.

Both new guards were verified by breaking them: changing `hc_low_rank` in the
committed header, and changing one declared serve format, each turn the suite red
with the specific disagreement named.

## Everything the artifact contains, declared once

Covering one model and calling the pattern proven was premature: seven converters
exist, and only Flash-Next had been declared. Doing the rest surfaced two things
the single-model version had no way to show.

**Numeric width is a profile's choice, not the model's.** The 0.8B stores every
weight W8; the 35B stores routed experts Q4, their down projections Q5 and the
output head Q6. A fixed `format` per object cannot express both. `ServeFormat`
gained `quantised`, meaning *the export decides* — and the declaration pins only
what the model itself pins, which is that a norm is never quantised. That is the
same line `quantizable` already draws on parameters.

**A served artifact is more than the text stack.** It carries a speculative draft
head, an MTP head whose single layer mirrors a text block, and on the larger
targets a 27-layer vision tower and a six-layer DFlash scorer. Those are model
components, so `ServeSection` now declares them: a prefix, a repeat count read
from the declaration, and the objects beneath it. The MTP head reuses the block's
own object declarations rather than restating them, which is why the dense
family's section is twelve objects and the MoE family's is fifteen — the
difference is exactly that its one layer is a MoE layer.

Vision geometry now comes from the declaration too: `use_visual_inputs` already
received the whole `vision_config` and collapsed it to a flag; it keeps the
geometry instead.

Derived exactly, against every committed converter whose checkpoint config is on
this machine:

| target | objects | contents |
| --- | --- | --- |
| qwen3_5_0_8b | 281 | text + MTP |
| qwen3_5_2b | 281 | text + MTP |
| qwen3_5_4b | 369 | text + MTP |
| qwen4exp | 986 | text + PLE + indexer |
| qwen3_6_35b_a3b | 934 | text + MTP + vision (333) + DFlash (51) |

Every name, shape and numeric width matches. `qwen3_6_27b` and `qwen3_8_27b` are
not listed only because their checkpoint configs are not on this machine; nothing
about them is known to be undeclarable.

One boundary is real and worth stating: whether an artifact *ships* the vision
tower is an export decision, not a property of the architecture. Qwen3.5-4B has a
`vision_config` and its artifact carries no tower, while the 35B's does. So the
declaration describes the tower; the target decides whether that section is
exported.

## Vision is not optional, and the stub that said otherwise

The previous section concluded that whether an artifact ships the vision tower is
an export decision, on the evidence that three converters carried no vision
objects. That was reading a stub as a design. The dense converters contain:

```python
def _build_vision_specs(): return build_vision_specs(5120)
VISION_TENSOR_SPECS: tuple[TensorSpec, ...] = ()  # text-only target
```

— a builder that is defined and never called, with a width (5120) that matches
none of those models' hidden sizes. Unfinished work, not a boundary. Serving
carries vision on every target.

Fixing it exposed a second thing: the shared `build_vision_specs` hardcoded the
Qwen3.6 tower, 27 layers of 1152, so it would have been wrong for all three even
had it been called. The towers differ — the 0.8B is 12 layers of 768, the 2B and
4B are 24 of 1024, Flash-Next and the 35B are 27 of 1152 — and the geometry was
sitting unused in each model's `vision_config`, which the declaration collapsed to
a boolean. The builder now takes the geometry; the declaration supplies it.

Every target now carries its tower, and every converter agrees with the
declaration exactly:

| target | before | after | vision |
| --- | --- | --- | --- |
| qwen3_5_0_8b | 281 | **434** | 153 |
| qwen3_5_2b | 281 | **578** | 297 |
| qwen3_5_4b | 369 | **666** | 297 |
| qwen4exp | 986 | **1319** | 333 |
| qwen3_6_35b_a3b | 934 | 934 | 333 (already) |

Two consequences worth stating rather than discovering later. The artifact grows,
and for `qwen4exp` the engine does not yet bind what it now carries — that
target's `bindings.cpp` contains no `vision/` objects at all and its `VisionConfig`
is, in its own words, "instantiated but never enabled". The tower travels in the
artifact and is ignored until the target enables it. And the qwen4exp converter's
own W8-count invariant caught the change immediately, which is the second time an
existing check has earned its keep here.

The emitter also gained a small rule that makes this maintainable: an object whose
declared geometry resolves to zero is not in the artifact. A text-only checkpoint
carries no tower, and that now falls out of the geometry instead of requiring a
second object list.

## A converter has two halves, and the tests only checked one

Adding the vision tower to three inventories broke conversion for all three, and
the suite stayed green. A converter is an inventory (what objects the artifact
holds) *and* a recipe (where each object comes from in the checkpoint); the
inventory tests checked the first and said nothing about the second, so three
`_build_vision_recipes()` still returning `()` sailed through. `recipe.py` has
always validated its own coverage on import — nothing was importing it.

The shared `build_vision_recipes` needed the same treatment as
`build_vision_specs`: it hardcoded the Qwen3.6 tower, so it could not have
sourced the Qwen3.5 towers even had it been called. Both now take the geometry,
and `validate_recipe_coverage` is what forces them to agree.

`test_conversion_recipe_covers_its_inventory` closes the hole for all five
targets that carry a recipe, and was verified by re-stubbing one: coverage fails
with *"recipe order or coverage does not match the tensor inventory"*. Full CPU
suite: 374 passed, 1 skipped — the DPO end-to-end included, now that the fp8
LM-head nondeterminism is fixed.

## The same facts, a fourth time

Conversion still refused after the recipe fix: `convert.py` guards every run with
`preflight_inventory`, which compares hardcoded section counts —
`(6, 267, 2, 12, 0, 281, 287)` — against the inventory. The `0` is the vision
count. So the geometry of an artifact was stated in four places: the declaration,
the inventory, the recipe, and this tuple, and only the first three had been
reconciled.

Updated for all three dense targets, and `test_converter_preflight_accepts_its_own_inventory`
now runs it, so all three halves of a converter are checked together. That is the
third guard this exercise has had to add after being caught by a failure rather
than a test — which is itself the argument for deriving these things rather than
restating them.

With that, Qwen3.5-2B converts: **584 objects**, exactly the count the declaration
implies, preflight passing with the vision tower included and its 632 source
tensors resolving out of the checkpoint. The tower was in the weights all along;
only the converter had it stubbed out.

One unrelated gap found on the way: `Qwen/Qwen3.5-0.8B` does not publish a
`generation_config.json` (the hub returns 404), and the family's `RESOURCE_SPECS`
requires one, so that target cannot convert from its own checkpoint. The 9B is in
the same position. Not caused by this work, but it is why the parity run uses the
2B.

## The engine settles it: an artifact may not carry what no binder consumes

Converting the 2B and starting the engine on it produced:

```
error: artifact object was not consumed by the selected target: vision/patch_embedding
```

That refutes the caveat two sections above, which said the tower would "travel in
the artifact and be ignored". It is not ignored; it is a hard load failure. The
previous two commits therefore made four targets unservable, and only running the
engine found it — no test did, because every test compared descriptions to each
other and none of them loaded an artifact.

It also corrects the reading that started this. `qwen3_5_2b/impl/load/bindings.cpp`
says, in C++:

```cpp
// Text-only target: the qwen3_5_2b artifact carries no vision objects
if (features.vision) {
    throw std::runtime_error("qwen3.5-2b target is text-only: --vision is unsupported");
}
```

So `VISION_TENSOR_SPECS = ()  # text-only target` was not a stub contradicting a
design — it agreed with one. The dead `_build_vision_specs` beside it was the only
genuine leftover. "Serving needs vision always" is a direction the engine does not
yet implement for these targets, not a description of what it does.

The resolution keeps both halves honest. `ServeObject` and `ServeSection` carry a
`capability`, and `inventory_for` takes the set a target implements: the
declaration describes the whole model — every Qwen3.5 checkpoint has a tower, with
its own geometry — while a target exports only what its binder consumes.
`test_declaration_describes_more_than_any_target_exports` pins that distinction so
the declaration cannot quietly decay into a description of one target's export.

Capability by target, read from the binders rather than assumed: `qwen3_6_27b` and
`qwen3_6_35b_a3b` consume vision; `qwen3_5_0_8b`, `qwen3_5_2b` and `qwen3_5_4b`
reject it explicitly; `qwen4exp` has no vision binding at all. Making vision
universal means implementing it in those targets — binder, forward and the
`--vision` gate — which is C++ work this exercise has now scoped rather than done.

Verified end to end: the 2B converts to 281 objects, the engine loads it, and it
answers a real request. 380 CPU tests pass.

## Training with vision: the tower the declaration was missing

"If the model supports vision, we must be able to train and serve with vision" is
two pieces of work, and measuring them first settled what each is.

**Training** was closer than it looked. `Qwen3_5ConditionalModel.forward` has
always scattered `visual_embeds` into the token embeddings by mask, so the
*injection* path existed; what was missing was the tower that produces them. Every
primitive a ViT needs is already in the DSL — `layernorm`, `gelu`, `softmax`,
`matmul_bias`, `transpose`, `permute`, `concat` — and `flash_attention` already
takes `causal`, so `VisionTower` is pure composition, like hyper-connections: no
new kernels.

Two details in it are correctness properties that no shape check would catch, so
they are called out in the module and pinned by tests. The norms are **LayerNorm
with a bias**, not the RMSNorm the text stack uses everywhere — which is why the
checkpoint carries `norm1.bias` at all. And the attention is **bidirectional**: a
vision transformer has no causal structure, and running it causally would still
produce plausible-looking embeddings.

**Serving** is the larger half and is C++. `qwen3_6::VisionBackboneConfig` is a
`static constexpr` struct pinned to one tower — 27 layers of 1152, 4304
intermediate, 16 heads — and every target inherits it. The towers are not the
same: the 0.8B is 12 layers of 768, the 2B and 4B are 24 of 1024, Flash-Next and
the 35B are 27 of 1152. So that struct needs the same parameterisation the Python
side just got, after which the text-only targets need their binder to consume the
vision objects, their forward to run the tower, and their `--vision` gate opened.
Until then those targets stay text-only, because the engine refuses an artifact
carrying objects no binder consumes.

The declaration now describes all of it either way, which is the point: the tower's
geometry has one home, and both halves read it from there.
