# Serve engine: three workstreams, built for many architectures

Today the engine serves Qwen dense. Qwen MoE, Gemma, GLM and Kimi are
coming. The three pieces of work below are the remaining throughput gaps,
and each one is a chance to either entrench per-model special-casing or
remove it. This document fixes the abstraction for each before any code
is written.

## The rule

A new architecture should require **declaring a shape, not writing a
kernel**. Concretely: adding a model may add entries to a registry and a
config struct. It must not require new kernel instantiations named after
the model, new branches in a route table that test a hidden size, or a
new copy of a scheduling loop.

Today the engine violates this in three measurable ways, all of which
this session hit:

- `ops/kernel/gqa_attention_geometry.cuh` registers geometries as
  model-named aliases (`Gqa27Geometry`, `Gqa08Geometry`, `Gqa4BGeometry`)
  and treats head dimension as a shared constant. Gemma, GLM and Kimi do
  not share it.
- Route tables gate on hidden sizes (`problem.input_rows == 1024 ||
  problem.qkv_rows == 6144`) rather than on a declared property, so a new
  model silently falls off the fast path — which is exactly how the
  batch-decode routes were lost until PATCHES #28 found them.
- The mixed-round and chained-round scheduling lives inside
  `targets/qwen3_6/impl/runtime/program_impl.h`. Any new target either
  copies it or does without.

## 1. Fused decode attention

**Why.** Decode attention runs ~2x its KV-read roofline. The current
split-KV design writes per-split partials to global memory and reads them
back in a second kernel; at multi-user batch the partial traffic can
exceed the KV traffic it is meant to parallelise.

**What to build.** A single kernel that keeps the running softmax
(m, l, acc) in registers across KV tiles and writes the output once —
the standard flash decode structure — with the split path kept only for
the long-context, small-batch case where it genuinely wins.

**The generic part.** Geometry becomes a full shape descriptor rather
than a model alias:

    template <int HeadDim, int QHeads, int KVHeads, class KvDtype>
    struct AttentionGeometry;

`HeadDim` must be a parameter (Qwen 128/256, Gemma 256, GLM 128, and Kimi
differs again). `KvDtype` carries bf16/i8/fp8 so quantised KV is a shape
property, not a separate code path. Registration moves to one table
mapping *declared* shapes to instantiations, and a model's config names
its shape. Adding Gemma then means adding a row, and if the row names an
uninstantiated combination the build fails loudly instead of the runtime
degrading silently.

**Guardrail.** Every fallback in this path must be observable. The two
most expensive bugs this session (PATCHES #29, #44) were both silent
degradations to a slow path. A route that cannot serve a shape should
throw at plan time, never quietly pick something slower.

## 2. Async scheduler

**Why.** ~11-12% of device time is idle in the host serial path between
rounds: sync, read egress, bookkeeping, stage, launch. vLLM overlaps this;
we do not.

**What to build.** Launch round N+1 on chained device state before
consuming round N's egress. Membership refresh lags one round; EOS is
detected device-side so a stopped sequence drains without a host
round-trip. The infrastructure already exists from PATCHES #32 — chained
graph flavour, host-function egress copies, multi-token resolve — what is
missing is the lifecycle change.

**The generic part.** This is scheduling, not mathematics, and none of it
should live in a target. The round lifecycle belongs in
`runtime/contract` as an interface every target implements:

    struct RoundPlan   { RoundKind kind; std::span<const Lane> lanes; ... };
    struct RoundResult { ... };
    virtual RoundHandle launch_round(const RoundPlan&) = 0;
    virtual RoundResult consume_round(RoundHandle) = 0;

`ConcurrentExecutor` then drives launch/consume generically and the
overlap is written once. A target supplies the forward pass; it does not
re-implement pipelining. Qwen MoE in particular must not need its own
copy — its rounds differ in expert routing, not in lifecycle.

## 3. Native Marlin residency

**Why.** Marlin's kernels are 1.5-2.3x faster on decode GEMMs, but they
need Marlin's tile layout. Today that layout is *derived* at load time
beside the resident weights, which costs a second copy — affordable at
0.8B/4B, impossible at 27B, where the copy does not fit and the model
therefore never gets Marlin at all. Its FP8-class GEMMs are 47% of device
time at 412-503 GB/s against 926 GB/s achievable on the same class of
problem.

**What to build.** The converter writes Marlin tiles; the loader maps
them as the residency. One copy, no derivation. Residency is a single
arena allocation, so this cannot be retrofitted at runtime — it has to
come from the artifact.

**The generic part.** A resident *layout* is a property of a weight, not
of a model recipe. The artifact format should carry, per tensor, a layout
tag (row-major, Marlin-4bit, Marlin-fp8, ...) that the loader dispatches
on, and the converter should choose layouts from the target's declared
compute profile rather than from per-recipe code. Kimi and GLM at 4 bits
then inherit Marlin residency by declaring a profile, with no converter
changes of their own.

**Caveat to settle first.** The 2.4-2.7x is measured on isolated shapes,
not in the serving loop. Probe one shape end-to-end before building the
full converter path.

## Sequencing

The scheduler contract (2) should land first: it is pure refactor, it is
the piece every future architecture depends on, and it makes the other
two measurable in isolation. Attention (1) second, since it is the larger
single win and its geometry rework unblocks non-Qwen head dims. Residency
(3) last, gated on the end-to-end probe.

Each lands with a 90-second steady-state measurement on at least two
models. 40-second runs measure the ramp and read up to 25% high; the
board's competitor figures are 90-second, so anything shorter is not
comparable.
