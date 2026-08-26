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