# Dispatch Pipeline Parallelism (dispatch-PP)

Dispatch Pipeline Parallelism is a single-node, **model-parallel** training mode for models whose
**base weights do not fit on a single GPU**, on **PCIe-only** multi-GPU boxes (no NVLink / no
GPU-to-GPU P2P required). It complements the data-parallel paths ([Multi-GPU / ZeRO](multi-gpu.md)),
which require the base weights to fit on each GPU.

It treats the node's GPUs as a stateless **round-robin compute pool**: the model's transformer
blocks are grouped into small contiguous *stages*, each stage runs on whichever GPU is next, its
weights are streamed from pinned CPU memory on demand, and the activations/gradients are handed from
one stage to the next **through host memory** — so a model that OOMs under DDP/ZeRO can train on a
box of consumer cards.

## When to use it

Use dispatch-PP when the **base model weights alone exceed single-GPU memory**:

- ZeRO (even ZeRO-3) cannot train such a model — it reconstructs the full weights via all-gather
  during forward/backward, so each GPU must momentarily hold full weight tensors (see the
  limitations in [Multi-GPU Training](multi-gpu.md)).
- Tensor parallelism would normally be required, but it needs fast GPU-to-GPU interconnect.
- Dispatch-PP needs neither: boundaries cross GPUs **through host memory**, so it runs on
  PCIe-only / retail multi-GPU systems without NVLink or P2P.

**Validated:**

- Qwen3.6-27B (~54 GB in BF16, cannot fit on a 32 GB card) trains across 4× 32 GB RTX 5090 with
  LoRA + FP8 weight streaming.
- A Qwen3.5-35B-A3B model, from an NVFP4 (compressed-tensors) export, runs GRPO at 131k tokens per row across
  8× 32 GB RTX 5090 with stage-local LoRA on the routed and shared experts (see
  [Long sequences](#long-sequences-chunked-up-to-131k)).

## How it works

- **Round-robin stages.** Layers are split into *small* contiguous stages (4 blocks each by default,
  or an explicit plan), so a whole stage stays resident across its microbatches and fits the VRAM
  budget. Stage `s` runs on GPU `s % gpus`, and stages pipeline across the GPUs (N parallel PCIe
  links). With more stages than GPUs, each GPU interleaves the units of its stages.
- **Per-stage weight streaming.** With `offload_master`, the frozen base weights live in pinned CPU
  RAM and are streamed to the GPU **once per step**, then reused across all of the step's
  microbatches. The trainers size the resident window (`SUROGATE_DISPATCH_PREFETCH_BLOCKS`) to the
  layers each GPU runs plus two, so a GPU's stages stay resident while their units run.
- **Host-staged boundaries.** Each stage's output activations (forward) and input gradients
  (backward) are copied GPU→host→GPU to the next stage — **no NCCL / P2P on the hot path**, which is
  what makes it work without NVLink.
- **Bounded memory.** Activation memory is one stage's worth; resident base weights are one GPU's
  stages; only the small, frozen-base **LoRA adapters** stay resident and get optimized, each GPU
  holding the adapters of its own stages ([stage-local LoRA](#stage-local-lora)). Peak memory is
  independent of model depth.

## Configuration

Minimal dispatch-PP run:

```yaml
parallelism: dispatch_pp     # opt in to model-parallel mode
offload_master: true         # stream the frozen base from pinned CPU per block (required)
gpus: 4                       # the round-robin GPU pool

recipe: fp8_hybrid           # or bf16

lora: true                   # frozen base + trained adapters (the supported mode)
lora_rank: 16
lora_alpha: 32
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

### Key parameters

| Key | Meaning |
| --- | --- |
| `parallelism: dispatch_pp` | Enables the mode. Mutually exclusive with ZeRO sharding, `cpu_training`, and MoE expert parallelism. |
| `offload_master: true` | Streams the frozen base weights from pinned CPU per block. Required for models that can't sit resident; the same scheduler runs without it for models that fit. |
| `gpus` | Size of the round-robin pool. More GPUs pipeline more stages concurrently for throughput. `gpus: 1` also works (single-GPU streaming). |
| `recipe` | `bf16` or `fp8_hybrid` (see below). The online NVFP4 recipe is deferred; a pre-quantized NVFP4 checkpoint loads under `bf16` (see [Constraints](#constraints)). |
| `gradient_accumulation_steps` | Microbatch count — see [Effective batch](#effective-batch-and-throughput). |
| `SUROGATE_DISPATCH_STAGE_BLOCKS` | Env var; blocks per stage (default `4`). Larger stages mean fewer cross-stage handoffs but more resident VRAM per stage. |
| `SUROGATE_DISPATCH_STAGE_PLAN` | Env var; explicit stage sizes, comma-separated, summing to the layer count (e.g. `6,5,5,5,5,5,5,4`). Overrides `SUROGATE_DISPATCH_STAGE_BLOCKS`, for example to give the stage that also runs the lm-head fewer blocks. A plan may have more stages than GPUs: one task per GPU then interleaves the units of all its stages, taking whichever ready stage is furthest behind. |
| `SUROGATE_DISPATCH_LORA_LOCAL` | Env var; `1` (set by both trainers unless already set): [stage-local LoRA](#stage-local-lora). `0` replicates every adapter on every GPU. |
| `SUROGATE_DISPATCH_PREFETCH_BLOCKS` | Env var; the resident weight window in layers. Both trainers set it to the most layers any GPU runs plus two; the GRPO trainer keeps a value you set yourself, the SFT trainer always sets it. The window must hold all of a GPU's stages, or each stage re-streams its weights for every microbatch. |
| `sequence_chunks` | Runs each row as windows of `sequence_len / sequence_chunks` tokens through every stage. See [Long sequences](#long-sequences-chunked-up-to-131k). |

### Stage-local LoRA

With `SUROGATE_DISPATCH_LORA_LOCAL=1` (the default under dispatch-PP), each GPU allocates, trains and
checkpoints only the adapters of the stages it runs (stage `s` on GPU `s % gpus`). Replicated LoRA
(`0`) keeps every layer's adapter on every GPU, collects every stage's gradients on GPU 0, runs the
optimizer there and copies the weights back to all GPUs through the host.

- **What it saves.** Adapter weights, gradients and optimizer state for the other GPUs' layers, and the
  host round trip of every step. On Qwen3.5-35B-A3B at 131k with expert LoRA (8× RTX 5090), the peak per
  card goes from 27.3 to 23.9 GB at r8, and r16, out of memory replicated, fits at 24.5 GB. At 4k the
  r16 train step goes from 5.2 to 1.65 s.
- **What stays the same.** Log-probs and exported adapters match the replicated run bit for bit, and
  grad norms to 6e-7. The LoRA grad norm is summed across the GPUs inside the step, and every GPU
  applies its own update.
- **Files.** The exported adapter is one PEFT file: rank 0 writes the tensors that every rank
  contributes. A checkpoint holds that adapter plus the optimizer state as one file per rank, and a
  resume reproduces the next steps exactly.
- **When to use it.** Always with the SFT and GRPO trainers. Only the native API's one-step-stale mode
  (`dispatch_pp_train_step_multigpu(stale=True)`, which the trainers do not use) needs replicated LoRA.
  The stage plan is part of the adapter layout: keep `SUROGATE_DISPATCH_STAGE_PLAN` (or
  `SUROGATE_DISPATCH_STAGE_BLOCKS`) fixed for a run and its resumes. The dispatch-PP entry points refuse
  a plan whose layers would run on a GPU that does not hold their adapters.

### Long sequences (chunked, up to 131k)

`sequence_chunks: N` runs each row as `N` windows of `sequence_len / N` tokens. Every window passes
through every stage, and each stage carries the row's chunk state from window to window on its own
GPU: the attention K/V of the earlier windows, plus the Gated DeltaNet state and conv tails of hybrid
models (kept in FP32). Gradients are exact. The backward walks each row's windows last to first and
accumulates dK/dV. A row only costs the windows up to its last target; later windows are skipped in
every stage and pass.

```yaml
sequence_len: 131072
sequence_chunks: 32           # 32 windows of 4,096 tokens
per_device_train_batch_size: 1
single_sample_bins: true      # GRPO: one sample per row (chunked attention has no packed-doc isolation)
lora_dropout: 0.0
```

- **Requirements.** As for chunked training in general: `per_device_train_batch_size: 1`,
  `lora_dropout: 0`, BF16 attention. CUDA graphs are off under dispatch-PP anyway.
- **Chunk state for the backward.** Each stage needs a row's chunk state again before it runs that
  row's windows backward, but the forward of the later rows has overwritten it. A GRPO log-prob pass
  that keeps its boundaries also copies each row's chunk state to pinned host memory, and the train
  step copies it back. That budget is `SUROGATE_DISPATCH_CHUNK_SNAPSHOT_GIB` per GPU (default 4; about
  0.74 GB per 131k row for a stage with two attention layers). Rows past the budget, and steps without
  a kept log-prob pass, re-run the stage over the earlier windows instead (`SUROGATE_DISPATCH_KV_SWEEP=1`
  forces that). Both give bit-identical results.
- **Attention backward.** The kv-prefix attention backward sums each dQ element over its key blocks in
  a fixed order (ordered dQ), so chunked steps are bit-reproducible run to run.
- **Memory.** The stack also reserves the attention backward's dK/dV expansion over the whole row:
  about 2 GiB at 131k.

**Production setting on 32 GB cards** (a Qwen3.5-35B-A3B model's NVFP4 export, 8× RTX 5090, GRPO):

```yaml
model: <path to the NVFP4 checkpoint>   # pre-quantized base, see QLoRA: compressed-tensors NVFP4
recipe: bf16
parallelism: dispatch_pp
offload_master: true
gpus: 8
sequence_len: 131072
sequence_chunks: 32
single_sample_bins: true
pad_to_multiple_of: 4096
lmhead_chunks: 32
per_device_train_batch_size: 1
lora: true
lora_rank: 16
lora_alpha: 32
lora_dtype: fp32                     # see "Adapter precision" below
lora_dropout: 0.0
lora_target_modules: [gate_proj, up_proj, down_proj]
recompute: false
qlora_recompute: false
fp8_weight_cache: off
```

```bash
export SUROGATE_DISPATCH_STAGE_PLAN=6,5,5,5,5,5,5,4   # the lm-head stage keeps one attention layer
export SUROGATE_DISPATCH_PREFETCH_BLOCKS=6            # resident window = the largest stage
export SUROGATE_DISPATCH_LORA_LOCAL=1
```

| | 32k | 64k | 131k |
| --- | --- | --- | --- |
| Warm step, 2 rows (log-prob pass + train step) | 5.48 s | 11.23 s | 29.53 s (8,876 tok/s) |

The step times were measured with bf16 LoRA; fp32 LoRA steps take the same time.

- **GPU memory.** Peaks at 131k are 28.4-30.5 GB per card with bf16 LoRA and 28.8-31.1 GB with fp32 LoRA
  (the fullest cards keep about 1.5 GB free).
- **Host memory.** About 77 KB per token of a dispatch call (see [GRPO steps](#grpo-steps)), on top of
  the pinned base.
- **End to end.** A GRPO step of ten 131k rows (1.3M tokens, two dispatch calls, `teacher_from_base`
  on, fp32 LoRA) took 186 s.

Per-stage plans are the main knob. With 8 stages on 8 GPUs, the 6-block first stage and the 4-block
lm-head stage balance the critical path. A 16-stage plan (two stages per GPU) needs about 2 GB less
memory per card at 131k, but it is not faster at 32-64k.

### FP8 weight streaming (`recipe: fp8_hybrid`)

Dispatch-PP on the 27B is **transfer-bound** — each step is dominated by streaming the base weights
over PCIe. With `recipe: fp8_hybrid`, the frozen **matmul** base weights are quantized to FP8-E4M3
**once at load** and streamed as FP8, halving the bytes on the wire and feeding the FP8 GEMM
directly. Only the matmul weights are quantized; conv / norm / `A_log` and other non-matmul
parameters stay BF16, so there is no silent precision loss in those kernels.

- **~1.45× faster step** on the 27B (4× RTX 5090, seq 2048) versus BF16 streaming.
- **Loss parity** — FP8-E4M3 base quantization is near-lossless; it also halves the pinned host RAM
  needed for the base.
- Set `recipe: bf16` to disable and stream BF16 weights.

### Effective batch and throughput

A dispatch-PP step runs `M = gpus × gradient_accumulation_steps` microbatches, all accumulated into
**one** optimizer step:

```
Effective batch = gpus × per_device_train_batch_size × gradient_accumulation_steps
```

Because the per-stage weight stream is fixed per step (each stage streams once and all microbatches
reuse it), raising `gradient_accumulation_steps` amortizes the stream over more tokens and keeps the
pipeline fuller (fewer bubbles) at **no extra peak memory** — so treat it as both the effective-batch
and the throughput knob.

### GRPO steps

GRPO drives the two halves of a dispatch-PP step itself. A staged forward through the loss stage returns
the per-token log-probs, Python forms the GRPO per-token gradients from them, and the staged backward is
seeded with those gradients. They are normalized by the step's loss tokens (`loss_scale`), as on the
standard path; the optimizer applies no second token scale.
With `teacher_from_base: true` each call first runs the same pass with the LoRA delta off; its log-probs
are the rows' teacher log-probs (a KL anchor to the base weights, `loss.teacher_tau`).

The log-prob pass keeps every stage's output boundary in host memory, so the training step over the same
rows skips its own forward (`SUROGATE_DPP_REUSE_FORWARD=0` turns this off). A call holds these boundaries
until its training step ends: about 77 KB per token of its rows on the 35B-A3B at 131k (57 KB of
boundaries over 8 stages, ~20 KB of chunk states). Freed blocks stay in a pinned pool for the next call
(`SUROGATE_PINNED_POOL_GIB`, default 32). A step with more tokens than fit runs as several calls, each with
its own log-prob pass and staged backward. The gradients accumulate on the stages' GPUs and one optimizer
update follows the last call, so the step trains exactly as one call would.

| Env var | Meaning |
| --- | --- |
| `SUROGATE_DPP_CALL_TOKENS` | Tokens per call; a row counts the windows up to its last target. Unset: half of the host's available memory (`MemAvailable`) at the start of the step, at `SUROGATE_DPP_HOST_BYTES_PER_TOKEN` bytes per token (default `120000`). |

**Adapter precision.** Use `lora_dtype: fp32` (the GRPO default) for expert LoRA. The adapter's
gradients are kept in `lora_dtype` and summed over every window of every row in the step. With bf16
the running sum is rounded after each window. On a 1.04M-token step whose advantages largely cancel,
as a GRPO group's do, the bf16-summed expert gradient was 41 % off the FP32 one (cosine 0.93); the
shared-expert A matrices fared worst. BF16 masters also drop updates smaller than half a bf16 ulp, so at
GRPO learning rates the A matrices barely move. FP32 adds 0.6-0.7 GB per card at 131k on the 35B-A3B.

`broadcast_dtype: bf16` (the default) writes weight broadcasts as the FP32 masters rounded to BF16
(round to nearest even): the PEFT layout and size of a bf16 adapter (2.53 GB for r16 on the 35B-A3B's
experts), which is what a BF16 serving engine loads. Checkpoints keep the FP32 masters.
`broadcast_dtype: master` broadcasts them as stored.

## Constraints

- **LoRA only (frozen base)** is the supported training mode. Full fine-tuning runs with `M = 1`
  (cross-microbatch gradient accumulation for FFT is not wired yet).
- **Mutually exclusive** with ZeRO sharding (`zero_level > 1`, `shard_weights`, `shard_gradients`),
  `cpu_training`, and MoE expert parallelism (`ep_size > 1`).
- **CUDA graphs are auto-disabled** — per-stage weight streaming cannot be captured.
- **Recipes:** `bf16` or `fp8_hybrid`. The online NVFP4 recipe for stage streaming is deferred (the
  4-bit base degrades convergence on this architecture and the block-scaled transpose negates the byte
  savings). A **pre-quantized** NVFP4 checkpoint loads as the base under `recipe: bf16`: its experts
  stream as NVFP4 and are dequantized to BF16 on the GPU ([QLoRA](qlora.md)).
- `recompute: true` (the dispatch default) bounds activation memory by replaying each stage's forward
  during the backward pass. `recompute: false` keeps the stage re-forward's activations instead, which saves
  one block forward per backward window (131k: 44.4 -> 37.8 s per step); with a QLoRA or pre-quantized base
  it also needs `qlora_recompute: false`.
- **One-step-stale updates** (native API only) need replicated LoRA (`SUROGATE_DISPATCH_LORA_LOCAL=0`).
- **Chunked sequences** need `per_device_train_batch_size: 1` and `lora_dropout: 0`.

## Example

A complete config for Qwen3.6-27B on 4× 32 GB GPUs ships at
[`examples/sft/qwen35/qwen36-text-lora-fp8-pp.yaml`](https://github.com/invergent-ai/surogate/blob/main/examples/sft/qwen35/qwen36-text-lora-fp8-pp.yaml).

```yaml
model: Qwen/Qwen3.6-27B
gpus: 4
per_device_train_batch_size: 1
gradient_accumulation_steps: 8     # M = gpus*GA = 32 microbatches/step

sequence_len: 2048
recipe: fp8_hybrid                  # FP8 weight streaming (half the PCIe bytes)

parallelism: dispatch_pp
offload_master: true               # stream the frozen base from pinned CPU

lora: true
lora_rank: 16
lora_alpha: 32
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

---

## See also

- [Multi-GPU Training](multi-gpu.md)
- [Offloading](offloading.md)
- [Precision and recipes](precision-and-recipes.md)
- [Config reference](../reference/config.md)
- [Back to docs index](../index.md)
