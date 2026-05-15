# Precision & Recipes

This guide helps you choose a precision setup (recipes and optional QLoRA) and explains how the dtype knobs interact.

## How to choose

- Start with **BF16** if you want maximum stability and portability.
- Use **FP8-hybrid** if you are on SM89+ (Ada/Hopper/Blackwell) and want higher throughput.
- Use **NVFP4** if you are on SM100+ (Blackwell) and want maximum compression/speed.
- Add **QLoRA** when you want to freeze base weights and fine-tune adapters with minimal VRAM.

For QLoRA details, see [QLoRA](qlora.md).

---

## Precision Recipes

Surogate provides 3 out-of-the-box precision recipes for the 3 most common numerical formats used in training:

- **BF16 (bfloat16)**: default recipe providing maximum numerical accuracy and most memory usage.
- **FP8-Hybrid (float8)**: provides a balance between numerical accuracy and memory usage by using 8-bit floating point precision.
- **FP4 (nvfp4)**: provides maximum acceleration on Blackwell GPUs by using 4-bit floating point precision, at the cost of some numerical accuracy.

### BF16

This recipe uses `bfloat16` for all GEMM operations without any quantization. It is suitable when memory and compute resources are not constrained, or when training smaller models where the savings from lower precision formats are not significant.

Use this recipe when:

- Only bfloat16 is supported on your hardware
- Memory and compute are not constrained
- You need a baseline for comparing quantized training
- Training smaller models where FP8/FP4 savings aren't significant

#### Forward/Backward Format

| Pass     | Data Type | Scaling |
| -------- | --------- | ------- |
| Forward  | bfloat16  | None    |
| Backward | bfloat16  | None    |

#### Example

```yaml
recipe: bf16
```

### FP8-Hybrid

This recipe uses FP8 with E4M3 format for the forward pass and E5M2 format for the backward pass, employing delayed scaling for improved stability.

- **E4M3** (max=448): Used for forward pass activations and weights - higher precision
- **E5M2** (max=57344): Used for backward pass gradients - larger dynamic range

Delayed scaling uses scale factors computed from the previous iteration's abs-max values, providing more stable training than just-in-time scaling. The recipe maintains an amax history window and uses the maximum value from the history to compute scale factors.

The numerical accuracy is generally comparable to bfloat16, while providing significant memory savings and speedup on supported hardware with FP8 tensor cores (SM89+: Ada Lovelace, Hopper, Blackwell).

Use this recipe when:

- Your GPU supports FP8 tensor cores (SM89+: Ada Lovelace, Hopper, Blackwell)
- You accept a minor drop in numerical accuracy for significant memory and speed benefits
- Training large models

#### Forward/Backward Format

| Pass     | Data Type | Max Value | Scaling            |
| -------- | --------- | --------- | ------------------ |
| Forward  | FP8 E4M3  | 448       | Per-tensor delayed |
| Backward | FP8 E5M2  | 57344     | Per-tensor delayed |

#### Parameters

| Parameter                 | Default | Description                                                    |
| ------------------------- | ------- | -------------------------------------------------------------- |
| `fp8_amax_history`        | 1024    | Length of amax history window for delayed scaling              |
| `skip_quant_first_layers` | 0       | Number of first layers to skip quantization (keep in bfloat16) |
| `skip_quant_last_layers`  | 0       | Number of last layers to skip quantization (keep in bfloat16)  |

#### Stability Tips

- Use `skip_quant_first_layers: 1` to keep embedding layer in BF16
- Use `skip_quant_last_layers: 2` if training is unstable (keeps lm_head layers in BF16)

#### Example

```yaml
recipe: fp8-hybrid
skip_quant_first_layers: 1
skip_quant_last_layers: 2
```

### FP4 (NVFP4)

This recipe uses NVIDIA's NVFP4 format for both forward and backward passes, employing two-level block scaling for improved stability. It uses FP8 E4M3 scales per 16 values and a global FP32 amax, along with 2D block quantization for weights, stochastic rounding for gradients, and optional Random Hadamard Transforms (RHT) to spread outliers before quantization.

It also includes the Four Over Six (4/6) technique (enabled by default), a modification to the NVFP4 quantization algorithm that evaluates two potential scale factors (max=4.0 vs max=6.0) for each block of values and selects the one with lower quantization error.

FP4 E2M1 representable values: ±{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}

Use this recipe when:

- You are training on Blackwell GPUs with FP4 support (SM100+)

#### Forward/Backward Format

| Tensor      | Data Type | Scale Format    | Block Size |
| ----------- | --------- | --------------- | ---------- |
| Activations | FP4 E2M1  | FP8 E4M3 + FP32 | 16         |
| Weights     | FP4 E2M1  | FP8 E4M3 + FP32 | 16x16 (2D) |
| Gradients   | FP4 E2M1  | FP8 E4M3 + FP32 | 16         |

#### Parameters

| Parameter                    | Default | Description                                                       |
| ---------------------------- | ------- | ----------------------------------------------------------------- |
| `fp4_backend`                | cutlass | Matmul backend: `cutlass` (default) or `cudnn`                    |
| `no_fp4_stochastic_rounding` | false   | Disable stochastic rounding for gradients                         |
| `skip_quant_first_layers`    | 0       | Skip quantization for first N layers (keep in BF16 for stability) |
| `skip_quant_last_layers`     | 0       | Skip quantization for last N layers (keep in BF16 for stability)  |

#### Backend Selection

- **cutlass** (default): Uses CUTLASS with Sm1xxBlkScaledConfig interleaved scale layout. Supports alpha fusion in epilogue for direct BF16 output.
- **cudnn**: Uses cuDNN with F8_128x4 scale swizzling layout.

Both backends implement the same quantization strategy; choose based on performance benchmarks for your workload.

#### Weight Caching (SM100+)

On Blackwell GPUs (SM100+), FP4 weight caching is **enabled by default** to eliminate per-forward weight quantization overhead.

**How it works:**

1. Weights are pre-quantized to FP4 format with CUTLASS-optimized layout during model initialization
2. The cached FP4 weights (packed data + FP8 block scales + global amax) are reused across forward passes
3. A separate transposed weight cache is maintained for the backward pass (dgrad)

**Requirements:**

- Blackwell GPU (SM100+)
- ZeRO-3/FSDP weight streaming disabled (weights must be static on device)
- Best suited for LoRA/QLoRA fine-tuning where base weights are frozen

#### Stability Tips

- Use `skip_quant_first_layers: 1` to keep embedding layer in BF16
- Use `skip_quant_last_layers: 4` if training is unstable (keeps lm_head layers in BF16)
- Random Hadamard Transforms and stochastic rounding are recommended (enabled by default)

#### Example

```yaml
recipe: nvfp4
skip_quant_first_layers: 1
skip_quant_last_layers: 4
```

---

## Mixed-Precision Training

Surogate is a versatile framework that supports mixed-precision training using a combination of numerical formats to optimize memory usage and computational speed while maintaining model accuracy.

The framework provides the following parameters to configure the precision of different components during training:

| Parameter          | Options          | Description                                                                                                                                                         |
| ------------------ | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **matmul_dtype**   | fp32, bf16, e4m3 | Data type for matrix multiplications. Defaults to model_dtype. e5m2/fp16/e2m1 not supported for forward pass. FP8 requires SM89+ (Ada/Hopper)                       |
| **gradient_dtype** | fp32, bf16, e5m2 | Data type for activation gradients and backward matmuls. Defaults to matmul_dtype. fp16/e4m3/e2m1 not supported. fp8-hybrid recipe forces e5m2                      |
| **master_dtype**   | fp32, bf16       | Master weight dtype for optimizer updates. Defaults to model_dtype. Only fp32 and bf16 are supported                                                                |
| **model_dtype**    | fp32, bf16       | Data type for non-matmul weights (RMSNorm, embeddings) and activations. Defaults to bf16. Other dtype params fall back to this. Only fp32/bf16 supported by kernels |
| **lora_dtype**     | fp32, bf16       | LoRA adapter master weights dtype for optimizer/export. Defaults to fp32. Work weights converted to model_dtype for compute. Only fp32↔bf16 conversion supported    |

### Matmul Dtype and Gradient Dtype

**Note on recipe behavior:** The `matmul_dtype` and `gradient_dtype` parameters are only respected when using the default `bf16` recipe. When using `fp8-hybrid` or `nvfp4` recipes, these parameters are overridden:

| Recipe       | Forward matmul | Backward matmul |
| ------------ | -------------- | --------------- |
| `bf16`       | matmul_dtype   | gradient_dtype  |
| `fp8-hybrid` | e4m3 (forced)  | e5m2 (forced)   |
| `nvfp4`      | e2m1 (forced)  | e2m1 (forced)   |

### Supported Matmul Dispatches

The following dtype combinations are supported for matrix multiplications:

| A (input/weight) | B (input/weight) | C (output) | Use Case                                        |
| ---------------- | ---------------- | ---------- | ----------------------------------------------- |
| fp32             | fp32             | fp32       | Full precision training                         |
| bf16             | bf16             | fp32       | Mixed precision (BF16 compute, FP32 accumulate) |
| bf16             | bf16             | bf16       | Pure BF16 training                              |
| e4m3             | e4m3             | fp32       | FP8 forward pass                                |
| e4m3             | e4m3             | bf16       | FP8 forward pass (BF16 output)                  |
| e4m3             | e5m2             | bf16       | FP8 backward pass (weight × gradient)           |

### Master Weights Dtype

The `master_dtype` parameter controls the precision of **master weights** - the authoritative copy of model weights used for:

1. **Optimizer updates**
2. **Checkpointing**
3. **Weight synchronization**

**Work weights** vs **Master weights**:

- **Work weights**: Used for forward/backward passes
- **Master weights**: Used for optimizer updates

When `master_dtype` differs from `model_dtype` or `matmul_dtype`, separate storage is allocated:

- Master weights are updated by the optimizer
- Work weights are converted from master weights before each forward pass

### Model Dtype

The `model_dtype` parameter is the fundamental dtype that controls the precision of non-matmul parameters and activations.

### LoRA Dtype

The `lora_dtype` parameter controls the precision of **LoRA adapter master weights**.

---

## See also

- [QLoRA](qlora.md)
- [Config reference](../reference/config.md)
- [Back to docs index](../index.mdx)

