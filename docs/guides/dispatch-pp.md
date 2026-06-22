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

**Validated:** Qwen3.6-27B (~54 GB in BF16, cannot fit on a 32 GB card) trains across 4× 32 GB
RTX 5090 with LoRA + FP8 weight streaming.

## How it works

- **Round-robin stages.** Layers are split into many *small* stages (~4 blocks each — more than the
  GPU count) so a whole stage stays resident across its microbatches and 2× a stage fits the VRAM
  budget. Stage `s` runs on GPU `s % gpus`, and stages pipeline across the GPUs (N parallel PCIe
  links).
- **Per-stage weight streaming.** With `offload_master`, the frozen base weights live in pinned CPU
  RAM and are streamed to the GPU one block at a time, **once per step** and reused across all of
  the step's microbatches. Resident base-weight memory is therefore ~2 blocks (a prefetch
  double-buffer), independent of model depth.
- **Host-staged boundaries.** Each stage's output activations (forward) and input gradients
  (backward) are copied GPU→host→GPU to the next stage — **no NCCL / P2P on the hot path**, which is
  what makes it work without NVLink.
- **Bounded memory.** Activation memory is one stage's worth; resident base weights are ~2 blocks;
  only the small, frozen-base **LoRA adapters** stay resident and get optimized. Peak memory is
  independent of model size and GPU count.

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
| `recipe` | `bf16` or `fp8_hybrid` (see below). NVFP4 stage streaming is deferred. |
| `gradient_accumulation_steps` | Microbatch count — see [Effective batch](#effective-batch-and-throughput). |
| `SUROGATE_DISPATCH_STAGE_BLOCKS` | Env var; blocks per stage (default `4`). Larger stages mean fewer cross-stage handoffs but more resident VRAM per stage. |

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

## Constraints

- **LoRA only (frozen base)** is the supported training mode. Full fine-tuning runs with `M = 1`
  (cross-microbatch gradient accumulation for FFT is not wired yet).
- **Mutually exclusive** with ZeRO sharding (`zero_level > 1`, `shard_weights`, `shard_gradients`),
  `cpu_training`, and MoE expert parallelism (`ep_size > 1`).
- **CUDA graphs are auto-disabled** — per-stage weight streaming cannot be captured.
- **Recipes:** `bf16` or `fp8_hybrid`. NVFP4 stage streaming is deferred (the 4-bit base degrades
  convergence on this architecture and the block-scaled transpose negates the byte savings).
- `recompute: true` (the dispatch default) bounds activation memory by replaying each stage's forward
  during the backward pass.

## Example

A complete config for Qwen3.6-27B on 4× 32 GB GPUs ships at
[`examples/sft/qwen35/qwen36-text-lora-bf16-pp.yaml`](https://github.com/invergent-ai/surogate/blob/main/examples/sft/qwen35/qwen36-text-lora-bf16-pp.yaml).

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
