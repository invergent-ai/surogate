# Quantized LoRA (QLoRA)

QLoRA enables memory-efficient fine-tuning by quantizing the frozen base model weights while training LoRA adapters in higher precision. Surogate supports two modes:

- **Online QLoRA**: Weights are quantized on-the-fly at startup from a standard BF16/FP16 checkpoint.
- **Pre-quantized models**: Load a model that is already quantized (e.g., from HuggingFace Hub). Supported formats are **Fine-Grained FP8** (compatible with NVIDIA ModelOpt FP8 exports), **ModelOpt NVFP4** (two-level FP8+FP32 block scaling), and **MXFP4** (microscaling FP4, GPT-OSS models). Pre-quantized models skip the quantization step at startup and load directly into the quantized format.

Surogate supports three online QLoRA quantization formats:

| Aspect                 | FP8 QLoRA                      | FP4 QLoRA                    | NF4 QLoRA (BitsAndBytes)          |
| ---------------------- | ------------------------------ | ---------------------------- | --------------------------------- |
| **Format**             | E4M3 (fwd), E5M2 (bwd)         | E2M1 (both)                  | NF4 (4-bit normal float)          |
| **Scaling**            | Per-tensor delayed             | Two-level block (FP8 + FP32) | Per-block absmax (+ double quant) |
| **GPU Requirement**    | SM89+ (Ada, Hopper, Blackwell) | SM100+ (Blackwell only)      | Any CUDA GPU                      |
| **Memory Compression** | ~50% vs FP16                   | ~75% vs FP16                 | ~75% vs FP16                      |

## Pre-Quantized Models

When a HuggingFace model already ships with pre-quantized weights (e.g., a ModelOpt-quantized checkpoint), Surogate can load the quantized tensors directly from the safetensors files without running an online quantization pass at startup. This saves significant startup time for large models and ensures the training run uses exactly the same quantized representation as the published checkpoint.

### Supported Formats

| Format                                                                                           | HF `quant_method`  | Data dtype | Scale dtype             | Notes                                                       |
| ------------------------------------------------------------------------------------------------ | ------------------ | ---------- | ----------------------- | ----------------------------------------------------------- |
| [**Fine-Grained FP8**](https://huggingface.co/docs/transformers/en/quantization/finegrained_fp8) | `fp8`              | FP8 E4M3   | FP32 (inverse scale)    | Per-block 2D tile scaling                                   |
| [**ModelOpt NVFP4**](https://huggingface.co/docs/transformers/en/quantization/mxfp4)             | `modelopt` / NVFP4 | FP4 E2M1   | FP8 block + FP32 global | Two-level scaling (FP8 + FP32)                              |
| [**MXFP4**](https://huggingface.co/docs/transformers/en/quantization/mxfp4)                      | `mxfp4`            | FP4 E2M1   | FP8 per-block           | GPT-OSS models only; uses `_blocks`/`_scales` tensor naming |

### How It Works

Surogate inspects the model's `quantization_config` in its HuggingFace `config.json` at startup to detect the pre-quantized format automatically — no extra config keys are needed.

**HuggingFace tensor naming:**

| Format           | Data tensor     | Block scale tensor        | Global scale tensor     |
| ---------------- | --------------- | ------------------------- | ----------------------- |
| Fine-Grained FP8 | `{name}.weight` | `{name}.weight_scale_inv` | —                       |
| ModelOpt NVFP4   | `{name}.weight` | `{name}.weight_scale`     | `{name}.weight_scale_2` |
| MXFP4            | `{base}_blocks` | `{base}_scales`           | —                       |

MXFP4 uses a different base name convention — there is no `.weight` suffix; instead, the packed FP4 data is stored as `{base}_blocks` and the FP8 scales as `{base}_scales`.

For all formats, layers listed in the model's `modules_to_not_convert` (typically the embedding and LM head) are loaded at full BF16 precision.

Fused weight tensors (e.g., `gate_up_proj`) that are stored as separate components in the pre-quantized checkpoint are loaded, dequantized, fused, and re-quantized automatically.

### Constraints

- **Requires `lora: true`** — base weights are frozen; only LoRA adapters are trained.
- **Mutually exclusive with online QLoRA** — `qlora_fp8`, `qlora_fp4`, and `qlora_bnb` must all be `false`.
- **No adapter merging** — `adapter_path` is not supported because there is no BF16 intermediate.

### Example — Fine-Grained FP8

```yaml
model: nvidia/Llama-3.1-8B-Instruct-FP8
lora: true
lora_rank: 16
recipe: bf16
```

No quantization flags are needed — the format is detected from the checkpoint's `quantization_config`.

### Example — ModelOpt NVFP4

```yaml
model: nvidia/Llama-3.1-8B-Instruct-NVFP4
lora: true
lora_rank: 16
recipe: nvfp4
```

### Example — MXFP4 (GPT-OSS)

```yaml
model: nvidia/Nemotron-H-47B-Base-MXFP4
lora: true
lora_rank: 16
recipe: bf16
```

MXFP4 is currently only supported for GPT-OSS model architectures. The format is detected automatically from the checkpoint's `quantization_config`.

---

## QLoRA vs Recipes

**QLoRA** determines how the frozen base model weights are stored and used during the forward pass. The base weights remain quantized and are never updated.

**Recipes** determine the precision format used for LoRA adapter computations, activations, and gradients during training.

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
| `skip_quant_first_layers` | 0       | Skip quantization for first N layers |
| `skip_quant_last_layers`  | 0       | Skip quantization for last N layers  |

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
- **Four-Over-Six (4/6) Adaptive Scaling**: Selects optimal scale per block
- **Layer Skipping**: Keep critical layers (embedding, lm_head) in BF16

### Parameters

| Parameter                 | Default | Description                 |
| ------------------------- | ------- | --------------------------- |
| `qlora_fp4`               | false   | Enable FP4 QLoRA            |
| `skip_quant_first_layers` | 0       | Skip FP4 for first N layers |
| `skip_quant_last_layers`  | 0       | Skip FP4 for last N layers  |
| `backend`                 | cutlass | Backend: cudnn or cutlass   |

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

NF4 QLoRA uses the BitsAndBytes NF4 (NormalFloat4) quantization format, providing ~75% memory reduction with broad GPU compatibility.

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

---

## See also

- [Precision & recipes](precision-and-recipes.md)
- [Config reference](../reference/config.md)
- [Back to docs index](../index.mdx)

