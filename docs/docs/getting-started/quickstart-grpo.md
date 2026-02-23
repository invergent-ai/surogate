# Quickstart: RL Training (GRPO)

This runs a GRPO reinforcement learning example. GRPO coordinates three components — an inference server (vLLM), an orchestrator (rollouts + rewards), and a Surogate trainer (policy gradient updates) — via a single command.

## 1) Pick example configs

Example configs are in `examples/grpo/`. GRPO uses three config files:

- **`train.yaml`** — Trainer settings (model, LoRA, precision, loss function)
- **`infer.yaml`** — vLLM inference server settings
- **`orch.yaml`** — Orchestrator settings (environment, batch size, sampling)

## 2) Run

```bash
surogate grpo --train examples/grpo/train.yaml --infer examples/grpo/infer.yaml --orch examples/grpo/orch.yaml
```

If you use `uv`:

```bash
uv run surogate grpo --train examples/grpo/train.yaml --infer examples/grpo/infer.yaml --orch examples/grpo/orch.yaml
```

This starts all three components in a single process with zero-copy GPU weight sharing (co-locate mode). No manual memory tuning needed — `gpu_memory_utilization` is computed automatically.

## 3) Outputs

Outputs (checkpoints, LoRA adapters, logs) are written under the trainer's `output_dir`.

## 4) Example Configuration

A minimal setup using the **reverse-text** environment on a single GPU:

**`train.yaml`**:

```yaml
model: Qwen/Qwen3-0.6B
output_dir: ./outputs
gpus: 1

per_device_train_batch_size: 1
sequence_len: 2048
max_steps: 40
logging_steps: 1

learning_rate: 2e-4
lr_scheduler_type: constant
max_grad_norm: 1.0
weight_decay: 0.01

recipe: fp8-hybrid

lora: true
lora_rank: 16
lora_alpha: 32

# GRPO loss
loss:
  ratio_type: token
  kl_tau: 0.0
  adv_tau: 1.0
  token_mask_low: 0.125
  token_mask_high: 8.0
  geo_mask_low: 0.1
  geo_mask_high: 10.0

# QeRL noise scheduler (optional, improves exploration)
noise_scheduler:
  enabled: true
  sigma_start: 5e-2
  sigma_end: 5e-4
  num_stages: 10
```

**`infer.yaml`**:

```yaml
model: Qwen/Qwen3-0.6B
enable_lora: true
max_lora_rank: 32
```

**`orch.yaml`**:

```yaml
model:
  name: Qwen/Qwen3-0.6B
  lora_adapter: default
  lora_rank: 16
  lora_alpha: 32

env:
  - id: reverse-text

batch_size: 128
rollouts_per_example: 16
seq_len: 2048
max_steps: 40

sampling:
  max_tokens: 128
```

## 5) Recommended Hyperparameters

### Learning Rate

RL training typically uses lower learning rates than SFT:

- **Recommended range**: `5e-7` to `5e-5` (start with `5e-6`)
- **Schedule**: `constant` or `cosine` (constant is common for RL)
- **Warmup**: 0 steps is fine for RL; use a few steps if training is unstable

### Batch Size

- **`batch_size`** (in `orch.yaml`): Number of rollouts per training step. `128`-`512` is typical.
- **`rollouts_per_example`**: Samples per prompt. `8`-`16` for diverse reward signal.
- **`per_device_train_batch_size`**: Typically `1` (packed sequences fill the batch).

### GRPO Loss

- **`ratio_type`**: `"token"` (per-token ratios, recommended) or `"sequence"` (per-sequence)
- **`kl_tau`**: KL penalty coefficient. Start with `0.0`; increase if the policy diverges too fast.
- **`adv_tau`**: Advantage scaling. Default `1.0` works well.

### Masking Thresholds

Masks filter tokens/sequences with extreme policy drift:

- **`token_mask_low`/`token_mask_high`** (default `0.125`/`8.0`): Per-token importance ratio bounds
- **`geo_mask_low`/`geo_mask_high`** (default `0.1`/`10.0`): Per-sequence geometric mean bounds
- If `masked` fraction exceeds 50% in logs, reduce learning rate or increase `kl_tau`

### QeRL Noise

QeRL adds controlled noise to inference weights for exploration:

- **`sigma_start`**: `5e-2` (initial noise level)
- **`sigma_end`**: `5e-4` (final noise level)
- **`num_stages`**: `10` (geometric decay intervals)
- Useful when rollouts produce low reward diversity early in training

### Precision

All precision options from SFT are available:

- **FP8-Hybrid** (`recipe: fp8-hybrid`): Recommended for Hopper+ GPUs
- **BF16** (`recipe: bf16`): Maximum accuracy
- **QLoRA**: Add `qlora_fp8: true`, `qlora_bnb: true`, or `qlora_fp4: true` for quantized base weights

## 6) Multi-Process Mode

If you need inference and training on separate GPUs (or separate nodes), run three commands instead:

```bash
# Terminal 1: Inference server
CUDA_VISIBLE_DEVICES=0 surogate grpo-infer infer.yaml

# Terminal 2: Orchestrator
surogate grpo-orch orch.yaml

# Terminal 3: Trainer
CUDA_VISIBLE_DEVICES=1 surogate grpo-train train.yaml
```

For single-GPU setups, co-locate mode is recommended since it shares base weights automatically.

## Notes

- GRPO requires `vllm` to be installed for the inference server.
- The `model` field must match across all three config files.
- `max_steps` in `train.yaml` and `orch.yaml` should match.

## See also

- [RL Training guide](../guides/rl-training.md) — Full architecture details, config reference, and tuning tips
- [Quickstart: SFT](quickstart-sft.md)
- [Quickstart: Pretraining](quickstart-pretraining.md)
- [Configuration](../guides/configuration.md)
- [Back to docs index](../index.mdx)
