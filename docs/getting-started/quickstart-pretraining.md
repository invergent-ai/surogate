# Quickstart: Pretraining (PT)

This runs a pretraining job using a YAML config.

## 1) Pick an example config

Example configs are in `examples/pt/`.

Start with:
- `examples/pt/qwen3.yaml`

## 2) Run

```bash
surogate pt examples/pt/qwen3.yaml
```

Or, via `uv`:

```bash
uv run surogate pt examples/pt/qwen3.yaml
```

## 3) Outputs

Outputs are written under the config's `output_dir`.

## 4) Recommended Hyperparameters

### Learning Rate

For pretraining from scratch, use a learning rate schedule with warmup and cosine decay:

- **Learning rate**: `1e-4` to `5e-4` (depending on model size)
  - Smaller models (< 1B): `3e-4` to `5e-4`
  - Medium models (1B-7B): `1e-4` to `3e-4`
  - Larger models (> 7B): `1e-4` to `2e-4`
- **Warmup steps**: 2000-5000 steps (or 1-5% of total steps)
- **Min learning rate**: `1e-5` (10% of peak LR)

### Batch Size

Global batch size should be large for stable pretraining:

- **Recommended global batch size**: 1M-4M tokens
  - Small models: 1M-2M tokens
  - Medium models: 2M-4M tokens
  - Large models: 2M-4M tokens

Calculate per-device batch size:
```
global_batch_size = per_device_train_batch_size × gradient_accumulation_steps × num_gpus × sequence_len
```

### Sequence Length

- **Standard**: 2048 or 4096 tokens
- **Long context**: 8192+ tokens (may require position interpolation)

### Weight Decay

- **Recommended**: `0.1`
- Applied to all parameters except biases and layer norms

### Gradient Clipping

- **Recommended**: `1.0`
- Prevents training instability from outlier gradients

### Precision

- **BF16** (default): Best accuracy, good speed on Ampere+
- **FP8-Hybrid**: Faster on Hopper+ (H100, H200), slight accuracy tradeoff
- **NVFP4**: Maximum speed on Blackwell+ (B200, B300), experimental

### ZeRO Configuration

- **1-2 GPUs**: ZeRO disabled or ZeRO-1
- **4-8 GPUs**: ZeRO-2 (shard optimizer + gradients)
- **8+ GPUs**: ZeRO-3 (shard optimizer + gradients + weights)

### Example Configuration

```yaml
model: Qwen/Qwen3-0.6B
output_dir: ./output

# Batch configuration (4M tokens global batch)
per_device_train_batch_size: 4
gradient_accumulation_steps: 16
sequence_len: 2048
gpus: 8

# Learning rate
learning_rate: 3e-4
lr_scheduler_type: cosine
warmup_steps: 2000
min_lr_ratio: 0.1

# Regularization
weight_decay: 0.1
max_grad_norm: 1.0

# Precision
recipe: bf16

# Parallelism
zero_level: 2
```

## See also

- [Training modes: Pretraining vs Full Fine-Tuning vs LoRA](training-modes.md)
- [Quickstart: SFT](quickstart-sft.md)
- [Configuration](../guides/configuration.md)
- [Back to docs index](../index.mdx)
