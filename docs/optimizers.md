# Optimizers in Surogate

Surogate supports two optimizers for training: **AdamW 8-bit** and **NorMuon**. Both optimizers use 8-bit quantized state storage to reduce memory consumption while maintaining training stability.

## Selecting an Optimizer

Use the `optimizer` configuration option to select which optimizer to use:

```yaml
optimizer: adamw_8bit  # Default
# or
optimizer: normuon
```

---

## AdamW 8-bit

AdamW 8-bit is the default optimizer in Surogate. It implements the AdamW algorithm with 8-bit blockwise quantization for optimizer states, following the bitsandbytes approach.

### How It Works

AdamW maintains two state tensors per parameter:
- **First moment (m)**: Exponential moving average of gradients
- **Second moment (v)**: Exponential moving average of squared gradients

These states are quantized to 8-bit using dynamic block quantization with per-block absmax scaling. This reduces optimizer state memory by approximately 4x compared to full-precision AdamW while maintaining training quality.

The update rule is:
```
m = β₁ · m + (1 - β₁) · g
v = β₂ · v + (1 - β₂) · g²
p = p - lr · m / (√v + ε) - lr · wd · p
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `learning_rate` | `2e-4` | Initial learning rate |
| `adamw_beta1` | `0.9` | Exponential decay rate for first moment |
| `adamw_beta2` | `0.999` | Exponential decay rate for second moment |
| `adamw_epsilon` | `1e-8` | Small constant for numerical stability |
| `weight_decay` | `0.1` | Weight decay coefficient (decoupled from gradient) |

### Example Configuration

```yaml
optimizer: adamw_8bit
learning_rate: 2e-4
adamw_beta1: 0.9
adamw_beta2: 0.999
adamw_epsilon: 1e-8
weight_decay: 0.1
```

### When to Use AdamW

- **General fine-tuning**: Works well for most fine-tuning tasks
- **LoRA training**: Recommended for adapter-based training
- **Stability**: More stable for small learning rates and noisy gradients
- **Compatibility**: Well-understood behavior, extensive research backing

---

## NorMuon

NorMuon (Normalized Momentum with Orthogonalization) is an advanced optimizer that applies orthogonalized momentum updates to 2D weight matrices. It is inspired by the Muon optimizer and uses the Polar Express algorithm for efficient matrix orthogonalization.

### How It Works

NorMuon uses a **hybrid approach**:
- **2D weight matrices** (attention projections, MLP weights): Updated using orthogonalized momentum
- **0D/1D parameters** (embeddings, layer norms, biases, lm_head): Updated using standard AdamW 8-bit

For 2D weights, the algorithm performs these steps each update:
1. **Momentum update**: Smooth gradients with exponential moving average
2. **Polar Express orthogonalization**: Transform the momentum matrix to its nearest orthogonal matrix using Newton-Schulz iterations
3. **Variance reduction**: Apply Adafactor-style adaptive scaling based on row/column statistics
4. **Cautious weight decay**: Apply weight decay only when the update direction aligns with the parameter sign

The orthogonalization helps maintain well-conditioned weight matrices during training, which can improve convergence for transformer models.

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `learning_rate` | `2e-4` | Learning rate for AdamW parameters (embeddings, norms, lm_head) |
| `normuon_momentum` | `0.95` | Momentum coefficient (β₁) for NorMuon 2D updates |
| `normuon_beta2` | `0.95` | EMA coefficient for variance estimation |
| `normuon_cautious_wd` | `true` | Enable cautious (sign-aware) weight decay |
| `weight_decay` | `0.1` | Weight decay coefficient |

NorMuon automatically applies a learning rate multiplier based on weight shape: `√(max(1, rows/cols))`. This boosts the effective learning rate for weights with more inputs than outputs.

### Example Configuration

```yaml
optimizer: normuon
learning_rate: 2e-4
normuon_momentum: 0.95
normuon_beta2: 0.95
normuon_cautious_wd: true
weight_decay: 0.1
```

### When to Use NorMuon

- **Full fine-tuning**: Can provide better convergence for full model training
- **Pre-training**: Designed for training transformers from scratch
- **Large models**: Benefits from orthogonalization are more pronounced in larger models
- **Multi-GPU training**: Fully supported with data-parallel training

### Cautious Weight Decay

When `normuon_cautious_wd` is enabled (default), weight decay is only applied when the update direction agrees with the parameter sign:

```
mask = sign(update) == sign(param)
p = p - lr · wd · p · mask - lr · update
```

This prevents weight decay from fighting against the gradient direction, which can improve training stability.

---

## Common Options

These options apply to both optimizers:

| Option | Default | Description |
|--------|---------|-------------|
| `max_grad_norm` | `0.0` | Maximum gradient norm for clipping (0 = disabled) |
| `gradient_accumulation_steps` | `4` | Number of micro-steps before optimizer update |
| `warmup_steps` | `0` | Linear warmup steps from 0 to learning_rate |
| `warmup_ratio` | `0.0` | Warmup as fraction of total steps |
| `lr_scheduler_type` | `linear` | Learning rate schedule: `linear`, `cosine`, or `wsd` |
| `cooldown_steps` | `0` | Linear cooldown steps at end of training |
| `final_lr_fraction` | `0.0` | Final LR as fraction of initial (for cooldown) |

---

## Memory Considerations

Both optimizers use 8-bit quantized states, consuming approximately:
- **AdamW 8-bit**: ~2 bytes per parameter (1 byte for m, 1 byte for v, plus absmax overhead)
- **NorMuon**: ~2 bytes per parameter for 2D weights, ~2 bytes per parameter for AdamW-managed parameters

Additional memory for NorMuon:
- Variance buffers: O(max(rows, cols)) per 2D weight tensor
- Polar Express workspace: Temporary buffers for matrix operations

For memory-constrained scenarios, consider:
- `offload_optimizer: true` - Store optimizer state in pinned host memory
- Increase `gradient_accumulation_steps` to reduce activation memory

---

## Performance Comparison

| Aspect | AdamW 8-bit | NorMuon |
|--------|-------------|---------|
| Memory | ~2 bytes/param | ~2 bytes/param + workspace |
| Speed | Faster | ~15% slower due to orthogonalization |
| Stability | Very stable | Stable with proper hyperparameters |
| Best for | LoRA, general fine-tuning | Full fine-tuning, pre-training |

---

## Multi-GPU Training

Both optimizers support multi-GPU training with data parallelism. Gradients are synchronized across GPUs before the optimizer step, and each GPU maintains its own shard of the optimizer state when using ZeRO optimization.

Configure multi-GPU training with:
```yaml
gpus: 4  # Number of GPUs to use
zero_level: 1  # ZeRO optimization level (1, 2, or 3)
```
