# Mixture-of-Experts (MoE) Models

Surogate provides full support for pre-training and fine-tuning Mixture-of-Experts (MoE) models. MoE architectures replace dense feed-forward networks (FFN) with multiple "expert" FFNs, where a learned router selects which experts process each token. This allows models to scale parameters without proportionally increasing compute.

## Supported MoE Models

Surogate natively supports:

| Model                   | Architecture | Experts | Active (top-k) |
| ----------------------- | ------------ | ------- | -------------- |
| Qwen3-MoE-30B-A3B       | qwen3_moe    | 128     | 8              |
| GPT-OSS-20B             | gpt_oss      | 128     | 4              |
| Nemotron-H (MoE layers) | nemotron_h   | 64      | 4              |

## Pre-training from Scratch

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

| Scenario                   | Aux Loss Coef     | Z-Loss Coef       | Reasoning                                                     |
| -------------------------- | ----------------- | ----------------- | ------------------------------------------------------------- |
| Default/pre-trained models | Use model default | Use model default | Well-tuned for the architecture                               |
| Router collapse detected   | Increase 2-5x     | Increase 2-5x     | Stronger regularization to stabilize routing                  |
| Over-uniform routing       | Decrease 2-5x     | Keep default      | Allow more routing specialization                             |
| Large batch training       | Keep default      | Keep default      | Usually stable                                                |
| Small batch training       | Increase 2x       | Increase 2x       | More regularization helps with noisy gradients                |

**Note:** Setting these values in the config overrides the model's default coefficients. Omit them to use the model's pre-configured values.

## Hyperparameter Recommendations

### For Upcycled MoE Models

Upcycled models need careful tuning because the router starts untrained:

| Parameter              | Recommended Value | Notes                                             |
| ---------------------- | ----------------- | ------------------------------------------------- |
| `learning_rate`        | 1e-5 to 3e-5      | Much lower than dense models (10x lower)          |
| `warmup_ratio`         | 0.15-0.20         | Longer warmup helps router stability              |
| `max_grad_norm`        | 0.5-1.0           | Gradient clipping prevents instability            |
| `weight_decay`         | 0.01-0.1          | Standard values work                              |
| `train_router`         | true              | Essential for upcycled models                     |
| `router_aux_loss_coef` | 0.01-0.05         | Higher than default for faster router convergence |
| `router_z_loss_coef`   | 0.001-0.01        | Standard values work                              |

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

| Metric               | Description                                       | Healthy Range              |
| -------------------- | ------------------------------------------------- | -------------------------- |
| `aux_loss`           | Load balancing auxiliary loss (sum across layers) | 0.001 - 0.1                |
| `z_loss`             | Router z-loss for logit regularization            | 0.0001 - 0.01              |
| `expert_utilization` | Fraction of experts receiving tokens (0-1)        | 0.7 - 1.0                  |
| `load_imbalance`     | Ratio of max to mean expert token counts (1.0 = perfect) | Depends on architecture* |

*Load imbalance healthy range depends on the number of experts — see [Load Imbalance](#load-imbalance) for per-architecture guidance.

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

Measures how evenly tokens are distributed across active experts, computed as `max_expert_tokens / mean_expert_tokens`. Monitor via:
- **Console**: `imbal` field shown inline for MoE models
- **JSON logs**: `moe_load_imbalance` field in step logs
- **wandb/Aim**: `train/moe_load_imbalance`

**Important:** The expected load imbalance depends heavily on the number of experts and top-k. With more experts, the most popular expert will naturally receive many more tokens than the average, even with a well-trained router. A single threshold does not apply across architectures.

**Per-architecture expected ranges (pre-trained models with trained router):**

| Architecture                          | Experts | Top-k | Expected Imbalance | Warning Threshold |
| ------------------------------------- | ------- | ----- | ------------------ | ----------------- |
| Small MoE (e.g., 8x2 upcycled)       | 8       | 2     | 1.5 - 3.0         | > 5.0             |
| Nemotron-H MoE layers                 | 64      | 4     | 3.0 - 6.0         | > 10.0            |
| GPT-OSS (128 experts, top-4)          | 128     | 4     | 4.0 - 6.0         | > 10.0            |
| Qwen3-MoE-30B-A3B (128 experts, top-8) | 128   | 8     | 8.0 - 12.0        | > 20.0            |

**Why large expert counts have high imbalance:** The metric reports the ratio of the *single busiest expert* to the *average across all experts*. With 128 experts and top-8 routing, each expert receives on average `tokens * 8 / 128 = 6.25%` of tokens. A trained router will specialize — popular experts may receive 50-75% of tokens while many experts receive very few. This is normal and expected behavior, not a sign of router collapse.

**What to watch for:** Rather than absolute values, monitor for:
- **Sudden spikes** in imbalance (2x+ increase within a few steps)
- **Monotonically increasing** imbalance (router concentrating on fewer experts over time)
- **Imbalance combined with low utilization** (< 0.5) — this indicates actual router collapse

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
- **Load imbalance**: Stable and within expected range for the architecture (see [Load Imbalance](#load-imbalance))
- **Aux loss**: Decreasing or stable in the 0.01-0.1 range

### Signs of Router Collapse

Router collapse occurs when the router stops distributing tokens effectively:

| Symptom                     | Metric Indicator                                           |
| --------------------------- | ---------------------------------------------------------- |
| All tokens to few experts   | `expert_utilization` < 0.2                                 |
| Severe load imbalance       | `load_imbalance` > 2x the expected range for architecture  |
| Router instability          | `z_loss` > 0.1 or spiking                                 |
| Training divergence         | `aux_loss` > 2.0                                           |

**Recovery steps:**

1. **Reduce learning rate** by 2-5x
2. **Increase warmup ratio** to 0.15-0.20
3. **Enable gradient clipping** with `max_grad_norm: 0.5`
4. **Enable router training** with `train_router: true`
5. **Increase loss coefficients** - set `router_aux_loss_coef: 0.05` and `router_z_loss_coef: 0.01` for stronger regularization
6. **Check batch size** - very small batches can cause routing instability

### Example: Monitoring an Upcycled Model (8 experts, top-2)

When fine-tuning a small upcycled model (e.g., 8 experts, top-2), you should see this progression:

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

For large pre-trained MoE models (128 experts), load imbalance will be significantly higher throughout training — see [Load Imbalance](#load-imbalance) for expected ranges.

## Memory Considerations

MoE models have more parameters than dense models but similar active compute:

| Model              | Total Params | Active Params | VRAM (BF16) | VRAM (QLoRA BnB) |
| ------------------ | ------------ | ------------- | ----------- | ---------------- |
| Qwen3-0.6B (dense) | 0.6B         | 0.6B          | ~2GB        | ~1GB             |
| Qwen3-0.6B 8x2 MoE | ~2.4B        | ~0.6B         | ~6GB        | ~2GB             |
| Qwen3-MoE-30B-A3B  | 30B          | 3B            | ~60GB       | ~12GB            |

Tips for reducing memory:
- Use QLoRA (`qlora_bnb: true` or `qlora_fp8: true`) for large models
- Use Expert Parallelism (`ep_size: N`) to shard experts across GPUs — each GPU holds `1/N` of the experts
- Enable `qlora_offload_experts: true` to offload inactive expert weights to CPU
- Reduce `per_device_train_batch_size` and increase `gradient_accumulation_steps`
- Enable `recompute` (default)

## Expert Parallelism (EP)

By default, every GPU holds a full copy of all expert weights (data-parallel only). For large MoE models with many experts, this means each GPU must fit all expert weights in memory — even though only a few experts are active per token.

**Expert Parallelism** distributes experts across GPUs so that each GPU holds only `num_experts / ep_size` local experts. Tokens are routed to the correct GPU via all-to-all communication, experts run in parallel, and results are sent back. This reduces per-GPU memory and enables parallel expert compute across GPUs.

### Configuration

```yaml
gpus: 4
ep_size: 2                       # 2-way expert parallelism
ep_load_balance_threshold: 1.3   # LLEP activation threshold (default)
```

| Parameter                   | Type  | Default | Description                                                                                                                                          |
| --------------------------- | ----- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ep_size`                   | int   | 1       | Number of GPUs in each expert-parallel group. Must divide `gpus`. Each GPU holds `num_experts / ep_size` local experts.                              |
| `ep_load_balance_threshold` | float | 1.3     | Imbalance ratio (`max_gpu_load / mean_gpu_load`) above which dynamic load balancing (LLEP) activates. Set to `1.0` for always-active load balancing. |

**Constraints:**

- `ep_size` must evenly divide `gpus`
- `num_experts` must be evenly divisible by `ep_size`
- `ep_size = 1` (default) disables EP and uses pure data parallelism

### How EP Works

With `gpus: 4` and `ep_size: 2`:

- **EP groups**: Ranks {0, 1} and {2, 3} each form an expert-parallel group
- **DP groups**: Ranks {0, 2} and {1, 3} each form a data-parallel group
- Each GPU holds `num_experts / 2` local experts
- Dense layer gradients are all-reduced across all 4 GPUs (global)
- Expert weight gradients are all-reduced across the DP group only (2 GPUs)

### LLEP Load Balancing

MoE routing is inherently imbalanced — some experts receive more tokens than others. Without load balancing, the GPU with the most-loaded experts becomes a bottleneck.

**Least-Loaded EP (LLEP)** dynamically redistributes work when imbalance is detected:

1. At each layer, expert token counts are measured across the EP group
2. If `max_gpu_load / mean_gpu_load` exceeds `ep_load_balance_threshold`, the LPT (Longest Processing Time) scheduler activates
3. Overloaded experts are temporarily transferred to underloaded GPUs along with their weights
4. After computation, results and gradients are routed back to the native GPU

LLEP is automatic when `ep_size > 1`. The `ep_load_balance_threshold` controls sensitivity:

| Value  | Behavior                                                        |
| ------ | --------------------------------------------------------------- |
| `1.0`  | Always active — LPT scheduling runs every layer                 |
| `1.3`  | Default — activates only when significant imbalance is detected |
| `2.0+` | Rarely activates — only under extreme imbalance                 |

### EP with QLoRA

EP works with all QLoRA variants. When combined with expert offloading (`qlora_offload_experts: true`), each GPU offloads only its local expert shard, reducing both GPU and CPU memory:

```yaml
model: Qwen/Qwen3-30B-A3B
model_type: qwen3_moe
gpus: 4
ep_size: 2

lora: true
lora_rank: 16
lora_alpha: 32

qlora_fp8: true
qlora_offload_experts: true
ep_load_balance_threshold: 1.0

lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

### When to Use EP

| Scenario                                             | Recommendation                               |
| ---------------------------------------------------- | -------------------------------------------- |
| Model fits on one GPU with QLoRA                     | EP not needed (`ep_size: 1`)                 |
| Many GPUs, want faster throughput                    | Use EP to parallelize expert compute         |
| Expert weights too large for one GPU even with QLoRA | Use EP to shard experts across GPUs          |
| High routing imbalance                               | Use EP with `ep_load_balance_threshold: 1.0` |

## Limitations

1. **CUDA Graphs:** MoE models cannot use CUDA graphs due to dynamic expert routing. Always set `use_cuda_graphs: false`.

2. **ZeRO-3:** MoE expert weights can be sharded with ZeRO-3, but routing overhead increases with world size.

3. **EP checkpoint compatibility:** Checkpoints saved with EP require the same `ep_size` when resuming training.

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
