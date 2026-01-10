# Multi-GPU Training

Surogate provides robust support for multi-GPU training, enabling efficient utilization of multiple GPUs to accelerate model training and handle larger models and datasets. The framework leverages data parallelism techniques with optimized communication patterns to distribute the workload across available GPUs.

## Training Models That Don't Fit on a Single GPU

Surogate's ZeRO implementation (stages 1-3) combined with CPU offloading enables training models that exceed single GPU memory capacity, **with important limitations**:

**✅ What ZeRO Enables:**
- Train models where the **base weights fit** on a single GPU, but **weights + optimizer states + gradients + activations** don't
- Example: Training a 7B model (14 GB weights) that requires 66 GB total memory can work on 4x 24GB GPUs with ZeRO-3
- Memory savings scale with number of GPUs: ZeRO-3 divides optimizer states, gradients, and weight storage by the number of GPUs

**❌ What ZeRO Cannot Do:**
- Train models where **base model weights alone** exceed single GPU memory
- Example: A 70B model (140 GB in BF16) cannot be trained even with ZeRO-3 on 8x 80GB GPUs
- **Reason**: During forward/backward passes, weights are reconstructed via all-gather operations, requiring each GPU to temporarily hold full weight tensors

**💡 Extending Capacity with CPU Offloading:**
Combine ZeRO with CPU offloading (`offload_optimizer`, `offload_master`, `offload_grads`, `offload_residual`) to train even larger models by using CPU RAM as overflow storage. This can extend capacity by 2-3x depending on your CPU RAM and PCIe bandwidth.

**Note:** For models exceeding these limits (e.g., 70B+ parameter models), tensor parallelism would be required, which splits model weights across GPUs during computation rather than just for storage.

## Multi-Threading vs Multiprocessing

Within a single node, there are two main approaches for handling multi-GPU support:

1. **One process per GPU (multiprocessing)**: This approach can help avoid Python's Global Interpreter Lock (GIL) and scales beyond a single node, making it suitable for distributed training across multiple machines.

2. **Multiple threads in a single process (multi-threading)**: This is Surogate's primary focus. Multi-threading exploits the shared address space, allowing direct GPU-to-GPU memory copies without resorting to IPC handles, which provides better performance for single-node multi-GPU training.

While Surogate supports both options, the framework is optimized for the multi-threaded setup and its direct communication capabilities.

## Data Parallelism with ZeRO-1

When running on multiple GPUs, Surogate always shards optimizer states (ZeRO-1) by default. This approach is strictly better than traditional Distributed Data Parallel (DDP) with replicated optimizer states, as it leads to reduced memory consumption without increasing the amount of communication.

In this setup, the model is replicated on each GPU and input data is split across the GPUs. Each GPU processes its portion of the data independently, computes gradients, and then synchronizes the gradients across all GPUs to update the model parameters.

## Kernel Fusion and Communication Optimization

To minimize memory bandwidth and improve performance, Surogate employs aggressive kernel fusion strategies:

### Fused Operations

To avoid unnecessary round trips to device memory, Surogate fuses all successive operations that are not either a global reduction or involve a matrix multiplication. Key fusion strategies include:

- **Non-linearity operators**: All non-linearity operators have an additional output parameter that returns the abs-max of their result, enabling efficient FP8 quantization without extra memory reads.
- **RMS-norm and residual addition**: These operations are handled in a joint kernel, which also returns the abs-max of the RMS-norm output.
- **Transpose and quantize**: Since FP8 GEMM on consumer GPUs only supports the TN (transpose:non-transpose) layout, Surogate uses a fused transpose+quantize kernel to handle the required transposes manually.
- **Cross-entropy loss**: The forward and backward passes of the cross-entropy loss are fused into a single kernel, avoiding the need to materialize a large per-token loss tensor.

These optimizations significantly reduce memory traffic and improve training throughput, especially when using FP8 precision.

## LM-Head and Embeddings: Replication Strategy

Due to the large vocabulary dimension in language models, both compute and communication costs for the LM-head (language modeling head) significantly exceed those of a regular transformer block. Surogate employs a specialized strategy to handle this imbalance:

### Replication Instead of Sharding

Rather than sharding the LM-head and token embeddings across workers, Surogate **replicates** them on each GPU. This design choice offers several advantages:

- **Reduced synchronization frequency**: Gradients for the LM-head only need to be synchronized at the last gradient accumulation step, rather than at every step.
- **Better communication overlap**: The LM-head is placed in a separate buffer from the double-buffered transformer blocks, enabling gradient communication to overlap with computation during the backward pass.

### Communication Overlap Strategies

During the last backward pass:

1. **LM-head gradient communication** can be overlapped with computing the gradients for the last two transformer blocks, whose weights are still available locally from the preceding forward pass.
2. **Backward matrix scheduling**: The two backward matrices of the LM-head gradient are scheduled such that the weight gradient calculation is handled first, allowing communication to overlap with the input gradient calculation.

However, the effectiveness of this optimization diminishes with increased gradient accumulation chunking, as gradient communication can only commence in the last chunk.

### Token Embeddings Limitation

For token embeddings, the next required operation after the backward pass is the global norm reduction of the gradients. Unfortunately, there is no computation available to overlap with this communication, so this latency cannot be hidden.

## Configuration Parameters

### `gpus`

Specifies the number of GPUs to use for training.

- **Default**: `1` (uses the first available GPU)
- **Special value**: `0` uses all available GPUs

Example:

```yaml
gpus: 4 # Use 4 GPUs
```

The effective batch size scales with the number of GPUs:

```
Effective batch size = per_device_batch_size × gradient_accumulation_steps × gpus
```

### `zero_level`

Controls the ZeRO (Zero Redundancy Optimizer) optimization level, which determines how optimizer states, gradients, and weights are partitioned across GPUs to reduce memory consumption.

- **Default**: `1`

| Level | Description              | What's Sharded                               |
| ----- | ------------------------ | -------------------------------------------- |
| 1     | Sharded optimizer states | Optimizer states (momentum, variance)        |
| 2     | Sharded gradients        | Optimizer states + gradients                 |
| 3     | Sharded weights          | Optimizer states + gradients + model weights |

Example:

```yaml
zero_level: 2 # Shard optimizer states and gradients
```

Higher ZeRO levels reduce per-GPU memory consumption but increase communication overhead. ZeRO-3 shards weights for storage, but full weights are reconstructed during forward/backward passes via all-gather operations (see limitations in the introduction above).

### `shard_weights`

Enables sharding of model weights across data-parallel processes.

- **Default**: `false`

When enabled:

- Model weights are partitioned across GPUs
- Each GPU only stores a fraction of the weights
- All-gather operations reconstruct full weights when needed
- Enables more effective use of CPU offloading
- Reduces per-GPU memory consumption

Example:

```yaml
shard_weights: true
```

### `shard_gradients`

Enables sharding of gradients across data-parallel processes.

- **Default**: `false`

When enabled:

- Gradients are partitioned across GPUs via reduce-scatter operations
- Each GPU only stores and updates its shard of gradients
- Enables more effective use of CPU offloading
- Reduces per-GPU memory consumption

Example:

```yaml
shard_gradients: true
```

## Advanced Options

For fine-grained control over multi-GPU communication and memory:

- `memcpy_all_gather`: Use memcpy for all-gather operations (threads backend only)
- `memcpy_send_recv`: Use memcpy for send/receive operations (threads backend only)
- `use_all_to_all_reduce`: Use all-to-all-based reduce algorithm for potentially better performance

---

## See also

- [Offloading](offloading.md)
- [Config reference](../reference/config.md)
- [Back to docs index](../index.md)
