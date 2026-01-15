# Quickstart: Supervised Fine-Tuning (SFT)

This runs a small LoRA SFT example using a YAML config.

## 1) Pick an example config

Example configs are in `examples/sft/`.

A reasonable default starting point:
- `examples/sft/qwen3-lora-bf16.yaml`

## 2) Run

```bash
surogate sft examples/sft/qwen3-lora-bf16.yaml
```

If you use `uv` and want to guarantee you’re running inside the project environment:

```bash
uv run surogate sft examples/sft/qwen3-lora-bf16.yaml
```

## 3) Outputs

Your run outputs (checkpoints, logs, artifacts) are written under the config's `output_dir`.

## 4) Recommended Hyperparameters

### Dense Models (Standard Fine-Tuning)

**Learning Rate:**
- **LoRA**: `1e-4` to `5e-4` (typically `2e-4`)
- **Full fine-tuning**: `5e-6` to `2e-5` (typically `1e-5`)
- **Warmup ratio**: `0.03` to `0.1` (3-10% of total steps)

**Batch Size:**
- **Global batch size**: 256K-1M tokens recommended
  - LoRA: 256K-512K tokens (less memory intensive)
  - Full fine-tuning: 512K-1M tokens
- Calculate: `per_device_train_batch_size × gradient_accumulation_steps × num_gpus × sequence_len`

**LoRA Configuration:**
- **Rank**: `8`, `16`, or `32`
  - `r=8`: Fast, memory-efficient, good for simple adaptations
  - `r=16`: Balanced (recommended default)
  - `r=32`: Higher capacity for complex tasks
- **Alpha**: Match rank (e.g., `lora_alpha=16` for `lora_rank=16`)
- **Target modules**: All linear layers by default (Q, K, V, O, gate, up, down)

**Regularization:**
- **Weight decay**: `0.01` to `0.1` (typically `0.01` for SFT)
- **Gradient clipping**: `1.0`

**Precision Options:**
- **BF16 LoRA** (default): Best accuracy
- **FP8 QLoRA**: 2x faster, minimal accuracy loss (Hopper+)
- **FP4 QLoRA**: 4x faster, slight accuracy tradeoff (Blackwell+)
- **BnB QLoRA**: Memory-efficient, works on all GPUs

**Example Dense Model Config:**
```yaml
model: Qwen/Qwen3-0.6B
output_dir: ./output

# Batch configuration (512K tokens global batch)
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
sequence_len: 2048
gpus: 4

# Learning rate
learning_rate: 2e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05

# LoRA
lora: true
lora_rank: 16
lora_alpha: 16

# Regularization
weight_decay: 0.01
max_grad_norm: 1.0

# Precision
recipe: bf16

# Parallelism
zero_level: 2
```

## 5) Fine-Tuning Mixture-of-Experts (MoE) Models

MoE models require different hyperparameters than dense models, especially when training upcycled MoE models (converted from dense checkpoints).

### Recommended Settings for MoE

**Key Differences from Dense Models:**

| Parameter | Dense Models | MoE Models | Reason |
|-----------|-------------|------------|--------|
| Learning rate | `2e-4` | `1e-5` | MoE routers are sensitive to aggressive LR |
| Warmup ratio | `0.05` | `0.15` | Longer warmup for router stability |
| Gradient clipping | `1.0` | `0.5` | Prevents routing instability |
| Weight decay | `0.01` | `0.01` | Regularizes router weights |

**Learning Rate:**
- **Standard MoE**: `1e-5` to `5e-5` (start with `1e-5`)
- **Upcycled MoE**: `1e-5` (conservative, increase only if training is too slow)
- **Warmup ratio**: `0.15` (15% of total steps for router stability)

**Batch Size:**
- **Global batch size**: 256K-512K tokens (smaller than dense models)
- MoE models are more sample-efficient; smaller batches work well

**LoRA Configuration:**
- **Rank**: `16` or `32` (same as dense models)
- **Alpha**: Match rank
- **Target modules**: Apply to expert layers for maximum adaptation

**Regularization:**
- **Weight decay**: `0.01`
- **Gradient clipping**: `0.5` (tighter than dense models)
- **Auxiliary loss coefficient**: Use default router aux loss settings

**Monitoring MoE Training:**
- **Gradient norms**: Should stay below `0.4` during training
  - If norms exceed `0.5`: reduce learning rate or increase `router_aux_loss_coef` in the model's `config.json` file
  - If norms spike above `0.8`: training is likely diverging
- **Router collapse symptoms**:
  - Sudden loss increases after initial decrease
  - Gradient norm spikes
  - Eval gap turning positive and growing

**Example MoE Config:**
```yaml
model: path/to/moe_model
model_type: qwen3_moe
output_dir: ./output

# Batch configuration (256K tokens global batch)
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
sequence_len: 2048
gpus: 4

# Learning rate (conservative for MoE)
learning_rate: 1e-5
lr_scheduler_type: cosine
warmup_ratio: 0.15

# LoRA
lora: true
lora_rank: 16
lora_alpha: 16

# Regularization (tighter for MoE)
weight_decay: 0.01
max_grad_norm: 0.5

# Precision
recipe: bf16

# Parallelism
zero_level: 2
```

### Upcycled MoE Models

If you've upcycled a dense model to MoE using `scripts/upcycle_moe.py`, additional care is needed during the initial training phase.

**Critical Guidelines:**

1. **Use conservative learning rates**
   - Start with `1e-5` and increase only if training is too slow
   - Router weights are freshly initialized and require careful tuning

2. **Monitor gradient norms closely**
   - Target: Keep gradient norms below `0.4` during training
   - Warning signs:
     - Norms exceed `0.5`: Reduce learning rate or increase `router_aux_loss_coef`
     - Norms spike above `0.8`: Training is likely diverging, restart with lower LR

3. **Watch for router collapse**
   - Symptoms include:
     - Sudden loss increases after initial decrease
     - Gradient norm spikes
     - Eval gap turning positive and growing
   - Solution: Reduce learning rate and increase warmup duration

**Training Data Requirements:**

Based on the [upcycling paper](https://arxiv.org/abs/2410.07524):
- **Budget**: ~150k supervised samples (sufficient for router training)
- **Duration**: 1 epoch recommended
- **Hardware**: Single GPU with 24GB+ VRAM is sufficient
- **Focus**: Training primarily adapts the router; expert weights inherit from dense model

**Upcycled MoE Example Config:**
```yaml
model: path/to/upcycled_moe_model
model_type: qwen3_moe
output_dir: ./output

# Very conservative settings for upcycled models
learning_rate: 1e-5
warmup_ratio: 0.2           # Extra-long warmup for router
max_grad_norm: 0.3          # Very tight clipping

lora: true
lora_rank: 16
lora_alpha: 16

# Small batch, single GPU is fine
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
sequence_len: 2048
```

### Example MoE Configs

- [qwen3moe-lora-bf16.yaml](../../examples/sft/qwen3moe-lora-bf16.yaml) - Standard MoE training
- [qwen3moe-lora-qbnb.yaml](../../examples/sft/qwen3moe-lora-qbnb.yaml) - MoE with QLoRA (BitsAndBytes quantization)

## Notes

- For private Hugging Face models/datasets, pass `--hub_token`.

## See also

- [Training modes: Pretraining vs Full Fine-Tuning vs LoRA](training-modes.md)
- [Quickstart: Pretraining](quickstart-pretraining.md)
- [Configuration](../guides/configuration.md)
- [Back to docs index](../index.mdx)
