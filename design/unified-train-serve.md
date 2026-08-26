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