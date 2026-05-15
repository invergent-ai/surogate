# Long Context Training

When fine-tuning at very long sequence lengths (8K, 32K, 128K+ tokens), MLP intermediate activations become the dominant source of GPU memory consumption. The `long_context` option enables **tiled MLP execution**, which chunks the MLP computation along the sequence dimension to reduce peak memory usage.

## The problem

In a standard transformer MLP, the intermediate activations have shape `[B*T, intermediate_size]`. For a 7B model with `intermediate_size = 11008` at 128K context in BF16:

- `gate_proj` output: `B*T * 11008 * 2 bytes` = 2.7 GB per layer
- `up_proj` output: 2.7 GB per layer
- SwiGLU output: 1.4 GB per layer

Even with gradient checkpointing (`recompute: true`), the backward pass must recompute these intermediates and hold them in memory while computing gradients. At long sequence lengths, this can easily exceed GPU memory.

## How tiled MLP works

MLPs have **no cross-token dependencies** -- each token is processed independently through the gate/up projection, activation function, and down projection. This means the computation can be split along the sequence (token) dimension without changing the result.

With `long_context: true`, Surogate:

1. **Detects MLP op groups** at graph compilation time by scanning for `mlp_up_weight` / `mlp_down_weight` matmul patterns
2. **Chunks forward execution**: instead of computing `[B*T, intermediate]` in one shot, processes `[chunk_size, intermediate]` at a time, freeing intermediates after each chunk
3. **Chunks backward execution**: for each chunk, recomputes the forward (up-proj + SwiGLU), then immediately runs the backward (down-proj grad, SwiGLU grad, up-proj grad), accumulating weight gradients across chunks
4. **Invokes LoRA hooks** per-chunk so adapter gradients are computed correctly

The chunk size is `min(B*T, hidden_dim)`. At short sequences where `B*T <= hidden_dim`, this results in a single chunk -- zero overhead, identical execution to the non-tiled path.

### Memory savings

Per-layer MLP memory drops from `O(B*T * intermediate_size)` to `O(chunk_size * intermediate_size)`:

| Context length | Without tiling | With tiling | Savings |
|---------------|---------------|-------------|---------|
| 2K            | 42 MB         | 42 MB       | 0       |
| 32K           | 672 MB        | 168 MB      | 504 MB  |
| 128K          | 2.7 GB        | 168 MB      | 2.5 GB  |

*(Llama-7B, intermediate_size=11008, BF16)*

### Performance

- **Short sequences** (T <= hidden_dim): single chunk, zero overhead
- **Long sequences**: ~5-10% slower step time due to per-chunk recomputation in backward. The tradeoff is worthwhile since it enables training at sequence lengths that would otherwise OOM

## Usage

Add `long_context: true` to your config:

```yaml
model: Qwen/Qwen3-0.6B
sequence_len: 32768
long_context: true

per_device_train_batch_size: 1
gradient_accumulation_steps: 16

lora: true
lora_rank: 16
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

datasets:
  - path: "emozilla/pg19-test"
    split: test
    type: auto
```

When `long_context` is enabled, CUDA graphs are automatically disabled (required because sequence lengths vary with sample packing).

## Supported models

Tiled MLP applies to **dense models** that use the standard gate+up / SwiGLU / down MLP pattern:

- Llama (all variants)
- Qwen3
- Qwen3.5 (both attention and linear recurrence blocks)
- Qwen3-VL

Models that are **not** tiled (their MLP patterns differ):

- **MoE models** (Qwen3-MoE, GPT-OSS, NemotronH-MoE): use grouped GEMM with dynamic routing
- **NemotronH MLP blocks**: use ReLU^2 activation, not SwiGLU

No configuration is needed to exclude these -- the detection is automatic based on weight naming conventions.

## Correctness

The tiled implementation is **numerically identical** to the non-tiled version. Matrix multiplication distributes over concatenation along the token dimension, so chunking produces the same result regardless of chunk count. This has been verified with bit-for-bit loss and gradient norm matching across all training steps.

## Combining with other options

`long_context` works with:

- **LoRA / QLoRA**: LoRA backward hooks are invoked per-chunk with correctly shaped tensors
- **Gradient checkpointing** (`recompute: true`): tiled MLP is used during both the forward replay and the backward pass
- **Sample packing** (`sample_packing: true`): works correctly since CUDA graphs are disabled
- **Multi-GPU** (`gpus > 1`): compatible with all ZeRO levels
- **FP8 / FP4 recipes**: compatible (the tiled path uses the same matmul dispatch)

## See also

- [Memory](memory.md)
- [Offloading](offloading.md)
- [Config reference](../reference/config.md)
