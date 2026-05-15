# Offloading in Surogate

Surogate supports offloading model components to CPU or NVMe storage to optimize GPU memory usage during training and inference. Offloading allows larger models to fit within limited GPU memory by moving less frequently accessed data to slower but larger storage.

The following offloading options are available in Surogate:

| Parameter             | Description                                                                                          | Default |
| --------------------- | ---------------------------------------------------------------------------------------------------- | ------- |
| **offload_residual**  | Offloads residual activations to CPU to reduce GPU memory usage during training.                     | False   |
| **offload_optimizer** | Offloads optimizer states (e.g., momentum, variance) to CPU to save GPU memory during training.      | False   |
| **offload_master**    | Offloads master weights to CPU to save GPU memory during optimizer updates.                          | False   |
| **offload_grads**     | Offloads gradients to CPU to reduce GPU memory usage during backpropagation.                         | False   |
| **offload_quants**    | Offloads quantized weights to CPU to reduce GPU memory usage during forward/backward passes.         | False   |
| **persistent_quants** | Keeps quantized weights in GPU memory between iterations to speed up training at the cost of memory. | False   |

## Offloading Residual Activations

When `offload_residual` is enabled, Surogate offloads residual activations from GPU to pinned CPU memory during the forward pass, then prefetches them back during the backward pass. Combined with `recompute_block`, this makes activation memory **O(1) instead of O(L)** where L is the number of layers.

### Buffer Allocation

**With offloading enabled:**

- **GPU buffers:** 2 double-buffered device tensors (shared across all layers via round-robin)
- **CPU buffers:** L pinned host buffers (one per layer to store each layer's residual)

**Without offloading:**

- L device buffers (one per layer, all on GPU)

### Memory Scaling

| Mode            | GPU Memory       | CPU Memory  | Transfers/step |
| --------------- | ---------------- | ----------- | -------------- |
| **Offload ON**  | O(1) - 2 buffers | O(L) pinned | 2L (D2H + H2D) |
| **Offload OFF** | O(L) - L buffers | 0           | 0              |

## Offloading Optimizer States

When `offload_optimizer` is enabled, Surogate stores optimizer states (first moment `m` and second moment `v` for AdamW) in pinned CPU memory instead of GPU memory. This significantly reduces GPU memory usage during training, especially for large models.

### Optimizer State Structure

The optimizer maintains per-parameter states:

- **First moment (m):** Exponential moving average of gradients (momentum)
- **Second moment (v):** Exponential moving average of squared gradients (variance)
- **FP8 scales:** Additional scale tensors when using 8-bit optimizer states

### Double-Buffering Mechanism

Surogate uses double-buffering to overlap data transfers with computation:

```
Buffer 0: layer 0, 2, 4, ...  (layer_idx % 2 == 0)
Buffer 1: layer 1, 3, 5, ...  (layer_idx % 2 == 1)
```

Two temporary GPU buffers are allocated during the optimizer pass, allowing one buffer to receive data while the other is being used for computation.

### Memory Impact

| Configuration        | GPU Memory             | CPU Memory              |
| -------------------- | ---------------------- | ----------------------- |
| **Offload OFF**      | ~8 bytes/param (m + v) | 0                       |
| **Offload ON**       | 2 temporary buffers    | ~8 bytes/param (pinned) |
| **Offload ON + FP8** | 2 temporary buffers    | ~2 bytes/param + scales |

### Performance Considerations

This will slow down the optimizer step drastically (memory-bound operation), but if enough gradient accumulation steps are performed, the overall contribution of the optimizer step will be negligible.

The optimizer step becomes PCIe-bound when offloading, but this overhead is amortized across gradient accumulation steps. With `gradient_accumulation_steps=4`, the optimizer runs once per 4 forward/backward passes, reducing its relative impact.

## Offloading Master Weights

When `offload_master` is enabled, Surogate stores the master copy of model weights in pinned CPU memory instead of GPU memory. Master weights are the full-precision (typically FP32) copies used for optimizer updates, while lower-precision work weights (BF16/FP8/FP4) are used for forward/backward passes.

### What are Master Weights?

In mixed-precision training, two copies of weights exist:

- **Master weights** (`master_dtype`): Full-precision (FP32 or BF16) weights updated by the optimizer
- **Work weights** (`matmul_dtype`): Lower-precision weights (BF16, FP8, FP4) used for forward/backward passes

The optimizer updates master weights, which are then converted to work weights for the next iteration.

### Double-Buffering Mechanism

Like optimizer state offloading, master weights use double-buffering:

```
Layer 0 → Buffer 0
Layer 1 → Buffer 1
Layer 2 → Buffer 0  (reuse)
Layer 3 → Buffer 1  (reuse)
...
```

Two temporary GPU buffers are allocated during the optimizer pass. While one layer is being updated, the next layer's weights are prefetched.

### Memory Impact

| Configuration   | GPU Memory             | CPU Memory                      |
| --------------- | ---------------------- | ------------------------------- |
| **Offload OFF** | L × master_weight_size | 0                               |
| **Offload ON**  | 2 × master_weight_size | L × master_weight_size (pinned) |

### Performance Considerations

Offloading master weights adds PCIe overhead during the optimizer step:

- H2D transfer before each layer's update
- D2H transfer after each layer's update

This is worthwhile when GPU memory is constrained and gradient accumulation amortizes the optimizer overhead.

### QLoRA Mode

**Note:** `offload_master` has no effect when using QLoRA. In QLoRA mode, base model weights are frozen and stored in quantized format. Only LoRA adapter weights are trainable, and these are small enough that offloading provides no significant benefit.

## Offloading Gradients (Multi-GPU only)

When `offload_grads` is enabled, Surogate stores the persistent gradient shards in pinned CPU memory instead of GPU memory. This reduces GPU memory usage during training while maintaining gradient accumulation correctness.

**Important:** `offload_grads` requires `shard_gradients=true` (ZeRO-2). It is not supported for unsharded gradients.

### How Gradient Sharding Works (ZeRO-2)

With gradient sharding enabled:

- Each GPU computes full gradients during backward pass
- Gradients are reduce-scattered across GPUs (each rank receives its local shard of the sum)
- Each rank only stores its own shard persistently
- During optimizer step, each rank updates only its shard of the weights

### Buffer Architecture

**GPU buffers (always on device):**

- 2 double-buffered full gradient tensors for backward computation
- Used round-robin: layer L uses buffer (L % 2)

**Persistent shards (HOST or GPU depending on `offload_grads`):**

- L sharded gradient tensors (one per layer, only local shard)
- If `offload_grads=true`: stored in pinned host memory
- If `offload_grads=false`: stored on GPU

### Memory Impact

| Configuration   | GPU Memory                | CPU Memory        |
| --------------- | ------------------------- | ----------------- |
| **Offload OFF** | 2 full buffers + L shards | 0                 |
| **Offload ON**  | 2 full buffers            | L shards (pinned) |

### Performance Considerations

- GPU only holds double-buffered full gradients during backward
- Persistent shards on host memory are accessed via CUDA's host memory mapping
- Reduce-scatter communication overlaps backward computation via double-buffering
- Trade-off: reduced GPU memory vs. PCIe bandwidth for optimizer access
- On a single GPU, there's no sharding benefit

## Quantized Weights: persistent_quants and offload_quants (Multi-GPU only)

These two options work together to manage quantized weight storage in distributed training scenarios.

**Important constraints:**

- `persistent_quants` requires `shard_weights=true` (ZeRO-3)
- `offload_quants` requires `persistent_quants=true`

### What are Quantized Weights?

In weight-sharded (ZeRO-3) training, weights must be gathered across GPUs before each forward/backward pass. Quantized weights are lower-precision copies (FP8, FP4, or INT8) of the master weights that:

- Reduce all-gather communication bandwidth
- Reduce GPU memory for the gathered full weights
- Are re-quantized after each optimizer step (when master weights change)

### persistent_quants

When `persistent_quants` is enabled:

- Quantized weights are computed once and stored persistently (one copy per layer)
- Avoids re-quantizing on every forward pass (saves compute)
- After optimizer updates, quantized weights are marked **stale** via version tracking
- Stale weights are re-quantized on the next `gather_block()` call

**Without persistent_quants:** Weights are quantized on-the-fly during each gather operation.

### offload_quants

When `offload_quants` is enabled (requires `persistent_quants`):

- Persistent quantized weights are stored in **pinned host memory** instead of GPU
- Double-buffered GPU staging buffers handle H2D transfers
- Quantized weights are streamed to GPU only when needed for gather/all-gather

### Memory Impact

| Configuration                          | GPU Memory                      | CPU Memory                         |
| -------------------------------------- | ------------------------------- | ---------------------------------- |
| **Neither**                            | Quantize on-the-fly (temporary) | 0                                  |
| **persistent_quants only**             | L × quantized_weight_size       | 0                                  |
| **persistent_quants + offload_quants** | 2 staging buffers               | L × quantized_weight_size (pinned) |

### Performance Considerations

- `persistent_quants` trades memory for compute (avoids repeated quantization)
- `offload_quants` moves quantized weights to CPU, reducing GPU memory
- In PCIe multi-GPU setups, H2D transfer may overlap with all-gather communication
- Combined with `--memcpy-all-gather`, can lead to speedups

### QLoRA Mode

**Note:** `persistent_quants` and `offload_quants` have no effect when using QLoRA (FP8 or FP4).

---

## See also

- [Memory](memory.md)
- [Multi-GPU](multi-gpu.md)
- [Config reference](../reference/config.md)
- [Back to docs index](../index.mdx)
