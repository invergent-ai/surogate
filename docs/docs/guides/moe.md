# Mixture-of-Experts (MoE) Models

Surogate provides full support for pre-training and fine-tuning Mixture-of-Experts (MoE) models. MoE architectures replace dense feed-forward networks (FFN) with multiple "expert" FFNs, where a learned router selects which experts process each token. This allows models to scale parameters without proportionally increasing compute.

## Supported MoE Models

Surogate natively supports:

| Model                 | Architecture | Experts      | Active (top-k) |
| --------------------- | ------------ | ------------ | -------------- |
| Qwen3-MoE-30B-A3B     | qwen3_moe    | 128          | 8              |
| Upcycled dense models | qwen3_moe    | Configurable | Configurable   |

## Creating a MoE Model

There are two paths to creating an MoE model:

1. **Upcycle a dense model** - Convert an existing dense model to MoE architecture
2. **Use a pre-trained MoE** - Start from an existing MoE model like Qwen3-MoE-30B-A3B

### Upcycling from a Dense Model

The `scripts/upcycle_moe.py` script converts a dense HuggingFace transformer into a Sparse MoE model. This is useful when you want to increase model capacity without training from scratch.

**Basic usage:**

```bash
python scripts/upcycle_moe.py \
    --model_id "Qwen/Qwen3-0.6B" \
    --num_experts 8 \
    --top_k 2 \
    --save_path ./my-moe-model
```

**Parameters:**

| Parameter          | Default  | Description                               |
| ------------------ | -------- | ----------------------------------------- |
| `--model_id`       | required | HuggingFace model ID or local path        |
| `--num_experts`    | 8        | Total number of experts per layer         |
| `--top_k`          | 2        | Number of experts activated per token     |
| `--dus_pct`        | 20       | Depth Upscaling percentage (0 to disable) |
| `--save_path`      | required | Output directory for the MoE model        |
| `--max_shard_size` | 5GB      | Maximum checkpoint shard size             |

**Recommended configurations for Qwen3-0.6B:**

- **8-2 configuration** (8 experts, top-k=2): Highly recommended. Provides good balance of capacity and efficiency.
- **Depth Upscaling 20%**: Optional but can provide small accuracy improvements.
- **Avoid high top-k**: Do not use top-k > 2 for small models. Higher values increase compute without consistent accuracy gains.

**What the script does:**

1. Loads the dense model
2. Optionally applies Depth Upscaling (DUS) to add more layers
3. Replaces each FFN block with an MoE layer containing `num_experts` copies of the original FFN
4. Initializes the router with small random weights
5. Saves the model in HuggingFace format

**Important:** Upcycled models have an untrained router. The router is initialized randomly, so the model requires fine-tuning before it can effectively route tokens to different experts.

### Pre-training from Scratch

To pre-train an MoE model from scratch, use the `surogate pt` command with an MoE model preset:

```yaml
# pt-moe-config.yaml
model: Qwen3-MoE-30B-A3B  # Or path to upcycled model
model_type: qwen3_moe

# Pre-training specific settings
sequence_len: 4096
per_device_train_batch_size: 1
gradient_accumulation_steps: 16

# Dataset for pre-training
datasets:
  - path: "HuggingFaceFW/fineweb-edu"
    type: text
    text_field: text
```

## Fine-Tuning MoE Models

Fine-tuning MoE models requires special care because of the router. There are three approaches:

### 1. Full Fine-Tuning (FFT)

Train all model weights including experts and router:

```yaml
# moe-fft.yaml
model: ./my-moe-model
model_type: qwen3_moe
output_dir: ./output-moe-fft

lora: false  # Disable LoRA for full fine-tuning

# Critical: Use lower learning rate than dense models
learning_rate: 1e-4
warmup_ratio: 0.15
max_grad_norm: 0.5
weight_decay: 0.01

# MoE models don't support CUDA graphs due to dynamic routing
use_cuda_graphs: false

per_device_train_batch_size: 1
gradient_accumulation_steps: 8
gpus: 4
```

### 2. LoRA Fine-Tuning (Experts Only)

Train LoRA adapters on expert weights while keeping the router frozen:

```yaml
# moe-lora.yaml
model: ./my-moe-model
model_type: qwen3_moe
output_dir: ./output-moe-lora
merge_adapter: true

lora: true
lora_rank: 16
lora_alpha: 32
lora_dtype: bf16

# Target expert projections (applied per-expert)
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj  # Expert gate projection
  - up_proj    # Expert up projection
  - down_proj  # Expert down projection

learning_rate: 1e-4
warmup_ratio: 0.15
max_grad_norm: 0.5
use_cuda_graphs: false
```

### 3. LoRA + Router Training (Recommended for Upcycled Models)

Train both LoRA adapters and the router gate. This is **essential for upcycled models** where the router is randomly initialized:

```yaml
# moe-lora-router.yaml
model: ./my-moe-model
model_type: qwen3_moe
output_dir: ./output-moe-lora-router
merge_adapter: true

lora: true
lora_rank: 16
lora_alpha: 32
lora_dtype: bf16

# Enable router training (critical for upcycled models)
train_router: true

# Optional: Tune MoE loss coefficients (defaults from model config)
# router_aux_loss_coef: 0.01   # Higher = more load balancing
# router_z_loss_coef: 0.001    # Regularize router logits

lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# Slightly higher learning rate works well with router training
learning_rate: 3e-5
warmup_ratio: 0.15
max_grad_norm: 1.0
weight_decay: 0.1
use_cuda_graphs: false
```

**What `train_router: true` does:**

1. Unfreezes the router gate weights (normally frozen in LoRA mode)
2. Computes gradients for the router during backward pass
3. Updates router weights alongside LoRA adapters
4. Exports trained router weights in the adapter checkpoint
5. Merges router weights when using `merge_adapter: true`

## QLoRA for MoE Models

MoE models can be fine-tuned with QLoRA to reduce memory usage. All QLoRA variants (BnB, FP8, FP4) are supported:

```yaml
# moe-qlora-bnb.yaml
model: Qwen/Qwen3-30B-A3B
model_type: qwen3_moe
output_dir: ./output-moe-qlora

lora: true
lora_rank: 16
lora_alpha: 32

# Enable BitsAndBytes NF4 quantization
qlora_bnb: true
qlora_bnb_block_size: 64
qlora_bnb_double_quant: true

# Optional: Router training works with QLoRA too
train_router: true
```

## Configuring MoE Loss Coefficients

MoE training uses two loss terms to regularize the router behavior:

- **Auxiliary Loss** (`router_aux_loss_coef`): Encourages balanced token distribution across experts. Higher values push for more uniform load balancing.
- **Z-Loss** (`router_z_loss_coef`): Regularizes router logits to prevent them from growing too large, which can cause routing instability.

Both coefficients can be configured via the training YAML:

```yaml
# MoE router loss coefficients
router_aux_loss_coef: 0.01   # Load balancing (default: from model config, typically 0.001-0.01)
router_z_loss_coef: 0.001    # Logit regularization (default: from model config, typically 0.001)
```

**When to adjust these values:**

| Scenario | Aux Loss Coef | Z-Loss Coef | Reasoning |
| -------- | ------------- | ----------- | --------- |
| Default/pre-trained models | Use model default | Use model default | Well-tuned for the architecture |
| Upcycled models | 0.01 - 0.05 | 0.001 - 0.01 | Higher aux loss helps untrained router learn balanced routing |
| Router collapse detected | Increase 2-5x | Increase 2-5x | Stronger regularization to stabilize routing |
| Over-uniform routing | Decrease 2-5x | Keep default | Allow more routing specialization |
| Large batch training | Keep default | Keep default | Usually stable |
| Small batch training | Increase 2x | Increase 2x | More regularization helps with noisy gradients |

**Note:** Setting these values in the config overrides the model's default coefficients. Omit them to use the model's pre-configured values.

## Hyperparameter Recommendations

### For Upcycled MoE Models

Upcycled models need careful tuning because the router starts untrained:

| Parameter               | Recommended Value | Notes                                    |
| ----------------------- | ----------------- | ---------------------------------------- |
| `learning_rate`         | 1e-5 to 3e-5      | Much lower than dense models (10x lower) |
| `warmup_ratio`          | 0.15-0.20         | Longer warmup helps router stability     |
| `max_grad_norm`         | 0.5-1.0           | Gradient clipping prevents instability   |
| `weight_decay`          | 0.01-0.1          | Standard values work                     |
| `train_router`          | true              | Essential for upcycled models            |
| `router_aux_loss_coef`  | 0.01-0.05         | Higher than default for faster router convergence |
| `router_z_loss_coef`    | 0.001-0.01        | Standard values work                     |

### For Pre-trained MoE Models (e.g., Qwen3-MoE-30B-A3B)

Pre-trained MoE models have a well-trained router, so standard LoRA hyperparameters work:

| Parameter       | Recommended Value | Notes                       |
| --------------- | ----------------- | --------------------------- |
| `learning_rate` | 1e-4 to 2e-4      | Standard LoRA learning rate |
| `warmup_ratio`  | 0.03-0.10         | Standard warmup             |
| `max_grad_norm` | 1.0               | Standard clipping           |
| `train_router`  | false             | Router is already trained   |

### Dataset Size Guidelines

For upcycled models, the lightweight fine-tuning stage requires:

- **Minimum:** ~50,000 samples
- **Recommended:** ~150,000 samples
- **Training duration:** 1 epoch is typically sufficient
- **Hardware:** Single GPU (RTX 4090/A100) in 1.5-8 hours depending on model size

## Monitoring MoE Training

Surogate provides dedicated MoE metrics to monitor router health and expert utilization during training. These metrics are logged automatically for all MoE models and can be viewed in the console, JSON logs, or external backends (wandb/Aim).

### MoE-Specific Metrics

| Metric               | Description                                       | Healthy Range |
| -------------------- | ------------------------------------------------- | ------------- |
| `aux_loss`           | Load balancing auxiliary loss (sum across layers) | 0.001 - 0.1   |
| `z_loss`             | Router z-loss for logit regularization            | 0.0001 - 0.01 |
| `expert_utilization` | Fraction of experts receiving tokens (0-1)        | 0.7 - 1.0     |
| `load_imbalance`     | Ratio of max to mean token counts (1.0 = perfect) | 1.0 - 2.5     |

### Enabling MoE Metrics

MoE metrics are logged automatically when training MoE models. To view them in external backends:

```yaml
report_to: wandb  # or [wandb, aim]
```

Metrics appear as:
- **wandb/Aim**: `train/moe_aux_loss`, `train/moe_z_loss`, `train/moe_load_imbalance`, `train/moe_expert_utilization` (logged with each training step)

### Interpreting the Metrics

#### Expert Utilization

Measures what fraction of experts are receiving tokens each step. Monitor via:
- **Console**: Not shown inline
- **JSON logs**: `moe_expert_utilization` field in step logs
- **wandb/Aim**: `train/moe_expert_utilization`

| Value     | Interpretation                       |
| --------- | ------------------------------------ |
| 0.9 - 1.0 | Excellent - all experts contributing |
| 0.7 - 0.9 | Good - most experts active           |
| 0.5 - 0.7 | Warning - some experts underutilized |
| < 0.5     | Critical - possible router collapse  |

**For upcycled models:** Expect low utilization (0.3-0.5) initially since the router is random. It should increase steadily during training when using `train_router: true`.

#### Load Imbalance

Measures how evenly tokens are distributed across active experts. Monitor via:
- **Console**: `imbal` field shown inline for MoE models
- **JSON logs**: `moe_load_imbalance` field in step logs
- **wandb/Aim**: `train/moe_load_imbalance`

| Value     | Interpretation                    |
| --------- | --------------------------------- |
| 1.0 - 1.5 | Excellent - near-perfect balance  |
| 1.5 - 2.5 | Good - acceptable imbalance       |
| 2.5 - 4.0 | Warning - some experts overloaded |
| > 4.0     | Critical - severe load imbalance  |

#### Auxiliary Loss

The load balancing loss that encourages uniform expert utilization. Monitor via:
- **Console**: `aux` field shown inline for MoE models
- **JSON logs**: `moe_aux_loss` field in step logs
- **wandb/Aim**: `train/moe_aux_loss`
- **Config**: Adjust strength with `router_aux_loss_coef` (see [Configuring MoE Loss Coefficients](#configuring-moe-loss-coefficients))

| Value      | Interpretation                        |
| ---------- | ------------------------------------- |
| < 0.01     | Very low - router may be too uniform  |
| 0.01 - 0.1 | Normal range                          |
| 0.1 - 1.0  | Elevated - router learning to balance |
| > 1.0      | High - significant load imbalance     |

#### Z-Loss

Regularization term that prevents router logits from becoming too large. Monitor via:
- **Console**: Not shown inline (check JSON logs or external backends)
- **JSON logs**: `moe_z_loss` field in step logs
- **wandb/Aim**: `train/moe_z_loss`
- **Config**: Adjust strength with `router_z_loss_coef` (see [Configuring MoE Loss Coefficients](#configuring-moe-loss-coefficients))

| Value        | Interpretation                          |
| ------------ | --------------------------------------- |
| < 0.001      | Normal                                  |
| 0.001 - 0.01 | Slightly elevated                       |
| > 0.01       | Warning - router logits may be unstable |
| > 0.1        | Critical - possible routing collapse    |

### Signs of Healthy Training

Monitor these indicators for healthy MoE training:

- **Loss**: Decreases steadily without sudden spikes
- **Gradient norm**: Stays below 0.4 (or 1.0 with `max_grad_norm: 1.0`)
- **Expert utilization**: Above 0.7 and stable or increasing
- **Load imbalance**: Below 2.5 and stable
- **Aux loss**: Decreasing or stable in the 0.01-0.1 range

### Signs of Router Collapse

Router collapse occurs when the router stops distributing tokens effectively:

| Symptom                  | Metric Indicator           |
| ------------------------ | -------------------------- |
| All tokens to one expert | `expert_utilization` < 0.2 |
| Severe load imbalance    | `load_imbalance` > 5.0     |
| Router instability       | `z_loss` > 0.1 or spiking  |
| Training divergence      | `aux_loss` > 2.0           |

**Recovery steps:**

1. **Reduce learning rate** by 2-5x
2. **Increase warmup ratio** to 0.15-0.20
3. **Enable gradient clipping** with `max_grad_norm: 0.5`
4. **Enable router training** with `train_router: true`
5. **Increase loss coefficients** - set `router_aux_loss_coef: 0.05` and `router_z_loss_coef: 0.01` for stronger regularization
6. **Check batch size** - very small batches can cause routing instability

### Example: Monitoring an Upcycled Model

When fine-tuning an upcycled model, you should see this progression:

**Early training (steps 0-100):**
```
expert_utilization: 0.35  # Low - router is random
load_imbalance: 4.2       # High - uneven distribution
aux_loss: 0.8             # Elevated
```

**Mid training (steps 100-500):**
```
expert_utilization: 0.65  # Improving
load_imbalance: 2.1       # Better balance
aux_loss: 0.15            # Decreasing
```

**Late training (steps 500+):**
```
expert_utilization: 0.85  # Good utilization
load_imbalance: 1.6       # Well balanced
aux_loss: 0.05            # Stable
```

### Programmatic Access to MoE Metrics

You can access MoE metrics programmatically:

```python
# Get MoE stats from trainer
moe_stats = trainer.get_moe_stats()

if moe_stats['valid']:
    print(f"Aux loss: {moe_stats['aux_loss']:.4f}")
    print(f"Z loss: {moe_stats['z_loss']:.4f}")
    print(f"Expert utilization: {moe_stats['expert_utilization']:.2%}")
    print(f"Load imbalance: {moe_stats['load_imbalance']:.2f}")
```

For more details on all available metrics, see [Training Metrics & Monitoring](metrics.md)

## Memory Considerations

MoE models have more parameters than dense models but similar active compute:

| Model              | Total Params | Active Params | VRAM (BF16) | VRAM (QLoRA BnB) |
| ------------------ | ------------ | ------------- | ----------- | ---------------- |
| Qwen3-0.6B (dense) | 0.6B         | 0.6B          | ~2GB        | ~1GB             |
| Qwen3-0.6B 8x2 MoE | ~2.4B        | ~0.6B         | ~6GB        | ~2GB             |
| Qwen3-MoE-30B-A3B  | 30B          | 3B            | ~60GB       | ~12GB            |

Tips for reducing memory:
- Use QLoRA (`qlora_bnb: true`) for large models
- Reduce `per_device_train_batch_size` and increase `gradient_accumulation_steps`
- Enable `recompute_block: true` (default)

## Limitations

1. **CUDA Graphs:** MoE models cannot use CUDA graphs due to dynamic expert routing. Always set `use_cuda_graphs: false`.

2. **ZeRO-3:** MoE expert weights can be sharded with ZeRO-3, but routing overhead increases with world size.

## Example Configurations

### Upcycled Model with Router Training

```yaml
model: ./upcycled-qwen3-moe-8x2
model_type: qwen3_moe
output_dir: ./output

per_device_train_batch_size: 2
gradient_accumulation_steps: 8
gpus: 4
use_cuda_graphs: false

sample_packing: true
sequence_len: 2048
num_epochs: 1

# Critical hyperparameters for upcycled models
learning_rate: 3e-5
warmup_ratio: 0.15
lr_scheduler_type: cosine
max_grad_norm: 1.0
weight_decay: 0.1

# LoRA + Router training
lora: true
lora_rank: 16
lora_alpha: 32
lora_dtype: bf16
train_router: true
merge_adapter: true

# Tuned for upcycled models (higher aux_loss for faster router convergence)
router_aux_loss_coef: 0.02
router_z_loss_coef: 0.001

lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

datasets:
  - path: "teknium/OpenHermes-2.5"
    samples: 80000
    type: auto
  - path: "Open-Orca/SlimOrca"
    type: auto
    samples: 40000
  - path: "meta-math/MetaMathQA"
    type: auto
    samples: 30000
```

### Large MoE Model with QLoRA

```yaml
model: Qwen/Qwen3-30B-A3B
model_type: qwen3_moe
output_dir: ./output-30b

per_device_train_batch_size: 1
gradient_accumulation_steps: 16
gpus: 4
use_cuda_graphs: false

sequence_len: 2048
num_epochs: 3

learning_rate: 2e-4
warmup_ratio: 0.03
max_grad_norm: 1.0

lora: true
lora_rank: 32
lora_alpha: 64

# QLoRA for memory efficiency
qlora_bnb: true
qlora_bnb_block_size: 64
qlora_bnb_double_quant: true

lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

datasets:
  - path: "your-dataset"
    type: auto
```

## Verifying Router Training

To confirm the router is being trained when using `train_router: true`:

1. **Check adapter checkpoint:** The exported adapter should contain router weights:
   ```python
   from safetensors.torch import load_file
   weights = load_file("output/adapter_model.safetensors")
   router_keys = [k for k in weights.keys() if "mlp.gate" in k]
   print(f"Router weights in adapter: {len(router_keys)}")
   # Should show one per layer (e.g., 28 for 28-layer model)
   ```

2. **Monitor gradient norms:** During training, router gradients contribute to the total gradient norm. You should see non-zero gradients being clipped.

3. **Compare before/after:** Load the merged model and compare router weights to the original. They should differ if the router was trained.
