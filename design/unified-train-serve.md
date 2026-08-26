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

Two whole serve targets, 1,723 lines each, **differ by 75 lines**:

    $ diff -r targets/qwen3_5_0_8b/impl targets/qwen3_5_4b/impl | grep -c '^[<>]'
    75
    $ diff -q targets/qwen3_5_0_8b/CMakeLists.txt targets/qwen3_5_4b/CMakeLists.txt
    (identical)

and every one of those 75 lines is a shape or a tensor name — head counts,
projection widths, embedding names. About 96% of each per-architecture target
is copy-paste. The architecture-specific content of a serve target is roughly
the size of a config struct; the rest is ceremony that a code generator should
be emitting.

## What to unify, in order of payoff

**1. Generate serve targets from the DSL declaration.** The DSL is already the
source of truth for what a model *is*, and it already describes every shape the
serve target restates. A build-time generator that emits the target from the
same declaration removes ~1,650 of every 1,723 lines and makes "train it, then
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
