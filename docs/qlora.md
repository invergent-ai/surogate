# Quantized LoRA (QLoRA)

QLoRA enables memory-efficient fine-tuning by quantizing the frozen base model weights while training LoRA adapters in higher precision. Surogate supports three QLoRA quantization formats:

| Aspect                 | FP8 QLoRA                      | FP4 QLoRA                    | NF4 QLoRA (BitsAndBytes)          |
| ---------------------- | ------------------------------ | ---------------------------- | --------------------------------- |
| **Format**             | E4M3 (fwd), E5M2 (bwd)         | E2M1 (both)                  | NF4 (4-bit normal float)          |
| **Scaling**            | Per-tensor delayed             | Two-level block (FP8 + FP32) | Per-block absmax (+ double quant) |
| **GPU Requirement**    | SM89+ (Ada, Hopper, Blackwell) | SM100+ (Blackwell only)      | Any CUDA GPU                      |
| **Memory Compression** | ~50% vs FP16                   | ~75% vs FP16                 | ~75% vs FP16                      |

## QLoRA vs Recipes

**QLoRA** determines how the frozen base model weights are stored and used during the forward pass. The base weights remain quantized and are never updated.

**Recipes** (see [recipes.md](recipes.md)) determine the precision format used for LoRA adapter computations, activations, and gradients during training.

You can combine any QLoRA format with any compatible recipe:

```
QLoRA (base weights) + Recipe (LoRA training) = Full Configuration
```

## FP8 QLoRA

FP8 QLoRA stores base model weights in FP8 format, reducing memory by ~50% compared to FP16/BF16.

### How It Works

Base weights are quantized to FP8 using two formats optimized for their use cases:

| Format   | Exponent | Mantissa | Max Value | Use Case                             |
| -------- | -------- | -------- | --------- | ------------------------------------ |
| **E4M3** | 4 bits   | 3 bits   | 448       | Forward pass (higher precision)      |
| **E5M2** | 5 bits   | 2 bits   | 57344     | Backward pass (larger dynamic range) |

**Delayed Scaling**: Scale factors are computed from the previous iteration's abs-max values (history window of 1024 by default), providing more stable training than just-in-time scaling.

### Parameters

| Parameter                 | Default | Description                          |
| ------------------------- | ------- | ------------------------------------ |
| `qlora_fp8`               | false   | Enable FP8 QLoRA                     |
| `margin`                  | 0       | Margin for scale factor computation  |
| `amax_history_len`        | 1024    | Length of amax history window        |
| `amax_compute_algo`       | MAX     | Algorithm: MAX or MOST_RECENT        |
| `reduce_amax`             | true    | Reduce amax across distributed group |
| `skip_quant_first_layers` | 0       | Skip FP4 for first N layers          |
| `skip_quant_last_layers`  | 0       | Skip FP4 for last N layers           |

### Recommended Recipe Combinations

| Recipe     | Use Case                                     |
| ---------- | -------------------------------------------- |
| **bf16**   | Maximum LoRA accuracy, any GPU (Recommended) |
| fp8-hybrid | Faster LoRA compute on SM89+ GPUs            |
| nvfp4      | Maximum speed on Blackwell (experimental)    |

### Example

```yaml
qlora_fp8: true
skip_quant_first_layers: 1
skip_quant_last_layers: 2
recipe: bf16
lora: true
lora_rank: 16
```

## FP4 QLoRA

FP4 QLoRA stores base model weights in NVIDIA's FP4 E2M1 format, reducing memory by ~75% compared to FP16/BF16. Requires Blackwell GPUs (SM100+).

### How It Works

FP4 E2M1 provides extreme compression with only 8 representable values per sign:

| Property          | Value                                     |
| ----------------- | ----------------------------------------- |
| **Exponent bits** | 2                                         |
| **Mantissa bits** | 1                                         |
| **Values**        | ±{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} |
| **Storage**       | 2 values per byte (4 bits each)           |

**Two-Level Block Scaling**:

- Level 1: FP8 E4M3 scales per block (16 values for activations, 16x16 for weights)
- Level 2: FP32 global amax baked into block scales

**Stability Techniques**:

- **Random Hadamard Transform (RHT)**: Spreads outliers before quantization
- **Stochastic Rounding**: Prevents quantization bias accumulation in gradients
- **Four-Over-Six (4/6) Adaptive Scaling**: Selects optimal scale per block ([arXiv:2512.02010](https://arxiv.org/abs/2512.02010))
- **Layer Skipping**: Keep critical layers (embedding, lm_head) in BF16

### Parameters

| Parameter                     | Default | Description                         |
| ----------------------------- | ------- | ----------------------------------- |
| `qlora_fp4`                   | false   | Enable FP4 QLoRA                    |
| `disable_rht`                 | false   | Disable Random Hadamard Transform   |
| `disable_stochastic_rounding` | false   | Disable stochastic rounding         |
| `disable_2d_quantization`     | false   | Use 1D instead of 2D weight scaling |
| `skip_quant_first_layers`     | 0       | Skip FP4 for first N layers         |
| `skip_quant_last_layers`      | 0       | Skip FP4 for last N layers          |
| `backend`                     | cutlass | Backend: cudnn or cutlass           |
| `enable_four_over_six`        | true    | Enable 4/6 adaptive scaling         |
| `four_over_six_metric`        | MSE     | Error metric: MSE, L1, or AbsMax    |

### Recommended Recipe Combinations

| Recipe     | Use Case                                       |
| ---------- | ---------------------------------------------- |
| **nvfp4**  | Maximum speed, full FP4 pipeline (Recommended) |
| bf16       | Higher LoRA accuracy, slower                   |
| fp8-hybrid | Balance of speed and accuracy                  |

### Example

```yaml
qlora_fp4: true
recipe: nvfp4
lora: true
lora_rank: 16
skip_quant_first_layers: 1
skip_quant_last_layers: 4
```

## NF4 QLoRA (BitsAndBytes)

NF4 QLoRA uses the BitsAndBytes NF4 (NormalFloat4) quantization format, providing ~75% memory reduction with broad GPU compatibility. This is the same quantization format used by the popular BitsAndBytes library.

### How It Works

NF4 is a 4-bit data type optimized for normally distributed weights:

| Property           | Value                                             |
| ------------------ | ------------------------------------------------- |
| **Bits per value** | 4                                                 |
| **Storage**        | 2 values per byte                                 |
| **Quantile-based** | 16 levels mapped to normal distribution quantiles |
| **Block size**     | Configurable (default: 64 values per block)       |

**Block-wise Quantization**:

- Weights are divided into blocks (default 64 values)
- Each block stores an FP32 absmax scale factor
- Values are quantized to 4-bit indices into a fixed NF4 lookup table

**Double Quantization** (optional):

- Absmax scales are further quantized to INT8
- Groups of 256 blocks share an FP32 scale and offset
- Reduces scale overhead from 4 bytes to ~1 byte per block

### Memory Layout

For a weight tensor with N elements using block size 64:

| Component     | Size (bytes) | With Double Quant  |
| ------------- | ------------ | ------------------ |
| NF4 data      | N / 2        | N / 2              |
| Absmax scales | (N / 64) × 4 | (N / 64) × 1       |
| Double quant  | —            | (N / 64 / 256) × 8 |

### Parameters

| Parameter                | Default | Description                             |
| ------------------------ | ------- | --------------------------------------- |
| `qlora_bnb`              | false   | Enable BitsAndBytes NF4 QLoRA           |
| `qlora_bnb_block_size`   | 64      | Block size for quantization (64 or 128) |
| `qlora_bnb_double_quant` | true    | Enable double quantization for scales   |

### GPU Compatibility

Unlike FP8 and FP4 QLoRA which require specific GPU architectures, NF4 QLoRA works on any CUDA GPU. The dequantization happens on-the-fly during forward and backward passes.

### Recommended Recipe Combinations

| Recipe     | Use Case                                         |
| ---------- | ------------------------------------------------ |
| **bf16**   | Best accuracy, broad compatibility (Recommended) |
| fp8-hybrid | Faster compute on SM89+ GPUs                     |

### Example

```yaml
model: Qwen/Qwen3-4B
lora: true
lora_rank: 16
lora_alpha: 32

qlora_bnb: true
qlora_bnb_block_size: 64
qlora_bnb_double_quant: true

recipe: bf16
```
