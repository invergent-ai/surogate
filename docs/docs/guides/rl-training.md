# RL Training (GRPO)

Surogate supports reinforcement learning fine-tuning via GRPO (Group Relative Policy Optimization). The pipeline consists of three coordinated processes: a vLLM inference server, a GRPO orchestrator, and the Surogate trainer.

This gives you:

- Surogate's near-SOL training throughput (LoRA, QLoRA, FP8)
- Async RL pipeline (rollouts, reward computation, sample packing)
- vLLM for fast inference and generation

## Architecture

GRPO training uses a three-process architecture:

```
┌─────────────┐    rollouts    ┌──────────────┐    batches    ┌──────────────────┐
│   vLLM      │ ─────────────> │ Orchestrator │ ────────────> │ Surogate Trainer │
└─────────────┘   new weights  └──────────────┘               └──────────────────┘
       ^                                                             │
       └─────────────── weight broadcast (filesystem) ───────────────┘
```

1. **vLLM inference** (`surogate grpo-infer`) generates completions with log-probabilities
2. **Orchestrator** (`surogate grpo-orch`) collects rollouts, computes rewards and advantages, packs samples into training batches
3. **Surogate trainer** (`surogate grpo-train`) performs the policy gradient update and broadcasts updated weights back to vLLM

The three processes communicate via a shared filesystem directory (`output_dir`).

## Quick Start

This walkthrough uses the **reverse-text** example with a local SFT checkpoint — a lightweight task that runs on a single node with 2 GPUs (one for inference, one for training).

### 1. Create the inference config

**`examples/grpo/infer.yaml`**:

```yaml
model: "./reverse-fft"
enable_lora: false
```

Key inference options:

| Key                       | Default        | Description                                            |
| ------------------------- | -------------- | ------------------------------------------------------ |
| `model`                   | (required)     | HuggingFace model ID or local path                     |
| `host`                    | `null`         | Bind address (null = all interfaces)                   |
| `port`                    | `8000`         | Bind port                                              |
| `dtype`                   | `"auto"`       | Data type (`float16`, `bfloat16`, `auto`)              |
| `max_model_len`           | `null`         | Maximum context length                                 |
| `enforce_eager`           | `false`        | Disable CUDA graphs (useful for debugging)             |
| `trust_remote_code`       | `false`        | Allow custom HF model code                             |
| `tp`                      | `1`            | Tensor parallelism degree                              |
| `dp`                      | `1`            | Data parallelism degree                                |
| `enable_lora`             | `true`         | Enable LoRA hot-reload                                 |
| `max_lora_rank`           | `null`         | Maximum LoRA rank (auto-rounded to vLLM valid values)  |
| `max_loras`               | `8`            | Max simultaneously loaded LoRA adapters                |
| `max_cpu_loras`           | `100`          | Max LoRA adapters cached on CPU                        |
| `enable_prefix_caching`   | `null`         | Enable prefix caching (null = vLLM default)            |
| `gpu_memory_utilization`  | `0.9`          | Fraction of GPU memory for KV cache                    |
| `weight_broadcast_type`   | `"filesystem"` | How to receive weight updates (`filesystem` or `nccl`) |
| `reasoning_parser`        | `null`         | Parser for extracting reasoning content                |
| `enable_auto_tool_choice` | `false`        | Enable auto tool choice                                |
| `rope_scaling`            | `null`         | RoPE scaling configuration dict                        |

### 2. Create the orchestrator config

**`examples/grpo/orch.yaml`**:

```yaml
model:
  name: "./reverse-fft"

env:
  - id: reverse-text

batch_size: 128
rollouts_per_example: 16
seq_len: 2048
max_steps: 20

sampling:
  max_tokens: 128
```

Key orchestrator settings:

| Key                    | Default                 | Description                                                 |
| ---------------------- | ----------------------- | ----------------------------------------------------------- |
| `model.name`           | `null`                  | HuggingFace model ID or local path                          |
| `batch_size`           | `128`                   | Number of rollouts per training step                        |
| `rollouts_per_example` | `1`                     | Samples generated per prompt                                |
| `seq_len`              | `2048`                  | Maximum sequence length for packing                         |
| `max_steps`            | `null`                  | Total training steps (null = run indefinitely)              |
| `max_async_level`      | `1`                     | How many steps inference can lag behind trainer             |
| `max_off_policy_steps` | `8`                     | Max allowed policy lag for a rollout before it is discarded |
| `oversampling_factor`  | `1.0`                   | Factor by which to oversample rollout requests              |
| `output_dir`           | `"outputs/run_default"` | Directory for checkpoints, weights, rollouts, and logs      |
| `seed`                 | `42`                    | Random seed                                                 |

**Sampling** (`sampling.*`):

| Key                                         | Default | Description                                                   |
| ------------------------------------------- | ------- | ------------------------------------------------------------- |
| `sampling.max_tokens`                       | `null`  | Max tokens per generation                                     |
| `sampling.temperature`                      | `1.0`   | Sampling temperature (set this OR `temp_scheduler`, not both) |
| `sampling.temp_scheduler.type`              | —       | Temperature schedule shape: `linear` or `cosine`              |
| `sampling.temp_scheduler.start_temperature` | —       | Temperature at step 0                                         |
| `sampling.temp_scheduler.end_temperature`   | —       | Temperature at final step                                     |
| `sampling.repetition_penalty`               | `1.0`   | Repetition penalty                                            |
| `sampling.min_tokens`                       | `0`     | Minimum tokens per sequence                                   |
| `sampling.seed`                             | `null`  | Sampling seed                                                 |

**Environments** (`env[]`):

| Key             | Default    | Description                                              |
| --------------- | ---------- | -------------------------------------------------------- |
| `env[].id`      | (required) | Environment ID from the verifiers registry               |
| `env[].name`    | `null`     | Optional human-readable name                             |
| `env[].args`    | `{}`       | Environment-specific arguments                           |
| `env[].address` | `null`     | Address of external env server (null = spawn subprocess) |

**Client** (`client.*`):

| Key                            | Default                        | Description                             |
| ------------------------------ | ------------------------------ | --------------------------------------- |
| `client.base_url`              | `["http://localhost:8000/v1"]` | Inference server URL(s)                 |
| `client.timeout`               | `1200`                         | Request timeout in seconds              |
| `client.api_key_var`           | `"VLLM_API_KEY"`               | Env var name for the API key            |
| `client.skip_model_check`      | `false`                        | Skip checking `/models` endpoint        |
| `client.elastic.hostname`      | —                              | DNS hostname for elastic pool discovery |
| `client.elastic.port`          | `8000`                         | Port for elastic pool servers           |
| `client.elastic.sync_interval` | `5.0`                          | Discovery re-check interval (seconds)   |

**Buffer** (`buffer.*`):

| Key                                  | Default              | Description                                             |
| ------------------------------------ | -------------------- | ------------------------------------------------------- |
| `buffer.seed`                        | `null`               | Random seed for deterministic sampling                  |
| `buffer.online_difficulty_filtering` | `false`              | Skip rollouts with reward 0.0 or 1.0                    |
| `buffer.easy_threshold`              | `null`               | Reward threshold above which a problem is "easy"        |
| `buffer.hard_threshold`              | `null`               | Reward threshold below which a problem is "hard"        |
| `buffer.easy_fraction`               | `0.0`                | Fraction of easy problems to promote to normal on start |
| `buffer.hard_fraction`               | `0.0`                | Fraction of hard problems to promote to normal on start |
| `buffer.hash_keys`                   | `["task", "prompt"]` | Keys used for example deduplication                     |
| `buffer.skip_verification`           | `false`              | If true, disable reward scoring (rewards always 0)      |
| `buffer.env_ratios`                  | `null`               | Per-environment sampling ratios (list, must sum to >0)  |

**Advantage** (`advantage.*`):

| Key                              | Default     | Description                                       |
| -------------------------------- | ----------- | ------------------------------------------------- |
| `advantage.type`                 | `"default"` | `"default"` or `"custom"`                         |
| `advantage.length_weighted_mean` | `false`     | Weight advantage by sequence length               |
| `advantage.import_path`          | —           | (custom only) Import path to advantage function   |
| `advantage.kwargs`               | —           | (custom only) Kwargs passed to advantage function |

**Filters** (`filters[]`):

| Key                            | Default  | Description                                              |
| ------------------------------ | -------- | -------------------------------------------------------- |
| `filters[].type`               | —        | `"gibberish"` or `"repetition"`                          |
| `filters[].enforce`            | `false`  | If true, mask flagged rollouts from training loss        |
| `filters[].token_id_threshold` | `100000` | (gibberish) Min token ID to flag as gibberish candidate  |
| `filters[].logprob_offset`     | `2.0`    | (gibberish) Offset from uniform-distribution logprob     |
| `filters[].window`             | `3000`   | (repetition) Consecutive high-prob steps before flagging |
| `filters[].prob_threshold`     | `0.99`   | (repetition) Per-token probability threshold             |

**Checkpointing** (`ckpt.*`):

| Key                             | Default | Description                                    |
| ------------------------------- | ------- | ---------------------------------------------- |
| `ckpt.interval`                 | `null`  | Save checkpoint every N steps                  |
| `ckpt.resume_step`              | `null`  | Step to resume from (-1 = latest available)    |
| `ckpt.wait_for_weights_timeout` | `null`  | Seconds to wait for weight directory on resume |
| `ckpt.keep_last`                | `null`  | Keep at most N recent checkpoints              |
| `ckpt.keep_interval`            | `null`  | Permanently keep checkpoints at every N steps  |
| `ckpt.skip_progress`            | `false` | Skip restoring progress state from checkpoint  |
| `ckpt.skip_buffer`              | `false` | Skip restoring buffer state from checkpoint    |

**Online evaluation** (`eval.*`):

| Key                                     | Default    | Description                              |
| --------------------------------------- | ---------- | ---------------------------------------- |
| `eval.env[].id`                         | (required) | Eval environment ID                      |
| `eval.num_examples`                     | `-1`       | Examples per eval environment (-1 = all) |
| `eval.rollouts_per_example`             | `1`        | Rollouts per example during eval         |
| `eval.interval`                         | `100`      | Evaluate every N training steps          |
| `eval.eval_base_model`                  | `true`     | Also evaluate the unmodified base model  |
| `eval.skip_eval_on_resume`              | `true`     | Skip eval immediately after resuming     |
| `eval.cancel_inflight_rollouts_on_eval` | `false`    | Cancel in-flight rollouts before eval    |

**Reporting** (`report_to.*`):

| Key                       | Default      | Description                        |
| ------------------------- | ------------ | ---------------------------------- |
| `report_to.project`       | `"Surogate"` | W&B project name                   |
| `report_to.name`          | `null`       | W&B run name                       |
| `report_to.offline`       | `false`      | Run W&B in offline mode            |
| `report_to.samples`       | `null`       | Log prompt/response samples        |
| `report_to.distributions` | `null`       | Log reward/advantage distributions |
| `report_to.interval`      | `10`         | Logging interval in steps          |

**Transport** (`rollout_transport.*`, `weight_broadcast.*`):

| Key                        | Default        | Description                                      |
| -------------------------- | -------------- | ------------------------------------------------ |
| `rollout_transport.type`   | `"filesystem"` | `"filesystem"` or `"zmq"`                        |
| `rollout_transport.host`   | `"localhost"`  | (zmq only) ZMQ bind host                         |
| `rollout_transport.port`   | `5555`         | (zmq only) ZMQ bind port                         |
| `rollout_transport.hwm`    | `10`           | (zmq only) High-water mark (max queued messages) |
| `weight_broadcast.type`    | `"filesystem"` | `"filesystem"` or `"nccl"`                       |
| `weight_broadcast.host`    | `"localhost"`  | (nccl only) NCCL rendezvous host                 |
| `weight_broadcast.port`    | `29501`        | (nccl only) NCCL rendezvous port                 |
| `weight_broadcast.timeout` | `1200`         | (nccl only) NCCL timeout in seconds              |

### 3. Create the Surogate trainer config

**`examples/grpo/train.yaml`**:

```yaml
model: "./reverse-fft"
output_dir: ./outputs
gpus: 1

# No datasets — GRPO data comes from the transport layer
sample_packing: false
datasets: []

per_device_train_batch_size: 1
sequence_len: 2048

# max_steps must match orch.yaml max_steps
max_steps: 20
logging_steps: 1

learning_rate: 3e-6
lr_scheduler_type: constant
warmup_steps: 0
max_grad_norm: 1.0
weight_decay: 0.01
optimizer: adamw_8bit

recipe: bf16

lora: false

# GRPO loss
loss:
  ratio_type: token
  kl_tau: 0.0
  adv_tau: 1.0
  token_mask_low: 0.125
  token_mask_high: 8.0
  geo_mask_low: 0.1
  geo_mask_high: 10.0

# Prime-RL integration
transport_type: filesystem
max_async_level: 1

# Checkpointing
save_steps: 100
checkpoint_dir: ./outputs/checkpoints
```

The orchestrator's `output_dir` must be a `run_*` subdirectory of the trainer's `output_dir`. By default the orchestrator uses `outputs/run_default`, and the trainer config above sets `output_dir: ./outputs` — so the trainer discovers the run by scanning `outputs/run_*`.

### 4. Start the three processes

```bash
# Terminal 1: Start vLLM inference server (GPU 0)
CUDA_VISIBLE_DEVICES=0 surogate grpo-infer examples/grpo/infer.yaml

# Terminal 2: Start orchestrator (CPU only)
# Default output_dir = outputs/run_default (a run_* subdirectory of trainer's output_dir)
surogate grpo-orch examples/grpo/orch.yaml

# Terminal 3: Start Surogate trainer (GPU 1)
# output_dir = ./outputs (parent of orchestrator's run_default)
CUDA_VISIBLE_DEVICES=1 surogate grpo-train examples/grpo/train.yaml
```

The trainer will block at startup until the orchestrator delivers the first batch. The orchestrator blocks until inference has generated enough rollouts. Once all three are running, the pipeline flows automatically.

### Single-GPU setup

If you only have one GPU, share it between inference and training. Use `gpu_memory_utilization` to limit vLLM's memory:

```bash
# Terminal 1: Inference — limit to 50% of GPU memory
CUDA_VISIBLE_DEVICES=0 surogate grpo-infer examples/grpo/infer.yaml  # set gpu_memory_utilization: 0.5

# Terminal 2: Orchestrator
surogate grpo-orch examples/grpo/orch.yaml

# Terminal 3: Surogate trainer — shares the same GPU
CUDA_VISIBLE_DEVICES=0 surogate grpo-train examples/grpo/train.yaml
```

### Math reasoning example

For a more practical task, here's a Hendrycks MATH setup with Qwen3-4B:

**`orch.yaml`**:
```yaml
model:
  name: "Qwen/Qwen3-4B-Instruct"

env:
  - id: "primeintellect/math-env"
    name: "hendrycks-math"
    args:
      dataset_name: "PrimeIntellect/Hendrycks-Math"
      dataset_subset: "default"

batch_size: 512
rollouts_per_example: 16
seq_len: 2048
max_steps: 500

sampling:
  max_tokens: 2048
```

**`infer.yaml`**:
```yaml
model: "Qwen/Qwen3-4B-Instruct"
```

## Configuration Reference

### Trainer config (inherited from SFT)

GRPO trainer configs inherit all fields from `SFTConfig`. The most relevant ones for RL training:

| Parameter                     | Default    | Description                                                 |
| ----------------------------- | ---------- | ----------------------------------------------------------- |
| `model`                       | (required) | HuggingFace model ID or local path                          |
| `gpus`                        | 1          | Number of GPUs                                              |
| `sequence_len`                | 1024       | Maximum sequence length                                     |
| `learning_rate`               | 2e-4       | Initial learning rate                                       |
| `max_steps`                   | -1         | Training steps (-1 = run until orchestrator stops)          |
| `gradient_accumulation_steps` | 4          | Micro-batches per optimizer step                            |
| `per_device_train_batch_size` | 2          | Batch size per device (typically 1 for packed RL sequences) |
| `optimizer`                   | adamw_8bit | Optimizer type                                              |
| `recipe`                      | bf16       | Precision recipe (bf16, fp8_hybrid, nvfp4)                  |
| `lora`                        | true       | Enable LoRA adapters                                        |
| `lora_rank`                   | 16         | LoRA rank                                                   |
| `output_dir`                  | output     | Parent directory containing orchestrator `run_*` subdirs    |

All LoRA, QLoRA, precision, and multi-GPU settings from SFT are available. See the [Configuration guide](configuration.md) for the full list.

### GRPO-specific trainer fields

#### `loss` (nested)

Controls the GRPO policy gradient loss computation.

| Parameter            | Default   | Description                                                                        |
| -------------------- | --------- | ---------------------------------------------------------------------------------- |
| `ratio_type`         | `"token"` | Importance ratio granularity: `"token"` (per-token) or `"sequence"` (per-sequence) |
| `kl_tau`             | 0.0       | KL penalty coefficient. Higher values keep the policy closer to the reference      |
| `adv_tau`            | 1.0       | Advantage scaling factor                                                           |
| `teacher_tau`        | 0.0       | Teacher KL distillation coefficient (requires teacher logprobs)                    |
| `token_mask_low`     | 0.125     | Mask tokens with importance ratio below this threshold                             |
| `token_mask_high`    | 8.0       | Mask tokens with importance ratio above this threshold                             |
| `geo_mask_low`       | 0.1       | Mask entire sequence when geometric mean ratio < threshold                         |
| `geo_mask_high`      | 10.0      | Mask entire sequence when geometric mean ratio > threshold                         |
| `sequence_mask_low`  | 0.0       | Mask sequence when min token ratio < threshold                                     |
| `sequence_mask_high` | 100.0     | Mask sequence when max token ratio > threshold                                     |
| `sequence_clip_high` | 10.0      | Clip sequence-level importance ratio                                               |

#### `transport_type`

How the orchestrator delivers batches to the trainer.

| Value          | Description                                                    |
| -------------- | -------------------------------------------------------------- |
| `"filesystem"` | (default) IPC via shared filesystem. Simple and reliable       |
| `"zmq"`        | IPC via ZeroMQ sockets. Lower latency for co-located processes |

#### `max_async_level`

Controls how many weight broadcasts can be in-flight simultaneously. Default: `1`. Higher values allow the inference engine to lag behind the trainer by more steps.

## How the Training Loop Works

Each training step performs:

1. **Weight broadcast** (after step > 0): Saves LoRA adapter to `{output_dir}/broadcasts/step_{N}/` with a `STABLE` marker file. vLLM polls for this marker to hot-reload weights.

2. **Pack and receive batch**: The packer (on master) converts `TrainingBatch` from the orchestrator into packed `MicroBatch` sequences. The data loader delivers these as numpy arrays.

3. **For each micro-batch** (gradient accumulation):
    - `compute_logprobs()` computes log-probabilities under the current policy
    - `compute_grpo_per_token_grads()` computes per-token gradient multipliers using the GRPO loss formula
    - `step_with_custom_loss()` performs the forward + backward pass with GRPO gradient seeding

4. **Optimizer step**: Updates LoRA adapter weights with the configured optimizer and learning rate schedule.

### GRPO Loss Formula

The GRPO loss for each token is:

```
loss = -(coeff * trainer_logprobs)[keep_mask].sum()

where:
  log_ratio = trainer_logprobs - inference_logprobs
  importance_ratio = exp(log_ratio)
  coeff = importance_ratio * (adv_tau * advantages - kl_tau * log_ratio)
```

The coefficient is treated as a constant (detached) during backpropagation, so the per-token gradient is simply:

```
grad[t] = -coeff[t] * keep_mask[t] / loss_scale
```

Tokens are masked (excluded) when their importance ratio falls outside `[token_mask_low, token_mask_high]`, or when sequence-level ratios exceed the geometric or sequence mask thresholds.

## QLoRA for RL Training

All QLoRA formats work with GRPO. QLoRA is particularly useful for RL training since it reduces the memory footprint of the frozen base model, leaving more room for the sequence buffers and logprob computations:

```yaml
# FP8 QLoRA (SM89+: RTX 40xx, L40, H100)
lora: true
qlora_fp8: true
recipe: bf16

# NF4 QLoRA (any GPU)
lora: true
qlora_bnb: true
recipe: bf16

# FP4 QLoRA (SM100+: Blackwell)
lora: true
qlora_fp4: true
recipe: nvfp4
```

See [QLoRA guide](qlora.md) for details on each format.

## Multi-GPU RL Training

Multi-GPU training works the same as SFT. Surogate handles data parallelism internally — the trainer presents as a single process to the orchestrator:

```yaml
gpus: 4
zero_level: 1  # Default: shard optimizer states
```

Each micro-batch is replicated across all GPUs. The per-token gradient computation happens on the first GPU's logprobs, and the resulting gradients are replicated for the backward pass.

## Tuning Tips

### Learning Rate

RL training typically uses a lower learning rate than SFT (5e-7 to 5e-5). Start with `5e-6` and adjust based on the KL divergence metrics.

### Masking Thresholds

The importance ratio masks are critical for training stability:

- **Token masks** (`token_mask_low`/`token_mask_high`): Filter individual tokens with extreme policy drift. The defaults (0.125, 8.0) allow up to 8x ratio before masking.
- **Geometric masks** (`geo_mask_low`/`geo_mask_high`): Filter entire sequences based on the geometric mean of token ratios. Catches sequences where many tokens have drifted moderately.
- If you see high `is_masked_frac` in logs (>50%), your policy is drifting too fast. Reduce the learning rate or increase `kl_tau`.

### KL Penalty

Setting `kl_tau > 0` adds a KL penalty that keeps the policy close to the reference (inference) policy. This prevents reward hacking but slows learning. Start with `kl_tau: 0.0` and increase if the policy diverges.

### Gradient Accumulation

Unlike SFT where gradient accumulation increases effective batch size, in RL training it controls how many packed micro-batches are processed per optimizer step. With `gradient_accumulation_steps: 1` and packed sequences, each step processes one densely-packed sequence.

## Monitoring

The trainer logs these GRPO-specific metrics at each step:

| Metric      | Description                                             |
| ----------- | ------------------------------------------------------- |
| `kl`        | Mean KL divergence between current and inference policy |
| `masked`    | Fraction of loss-eligible tokens that were masked       |
| `tokens`    | Total loss-eligible tokens in the step                  |
| `loss`      | Training loss (from the backward pass)                  |
| `grad_norm` | Gradient norm after clipping                            |

A healthy training run shows:

- `kl` gradually increasing from near-zero (policy is improving)
- `masked` staying below 30-40% (policy isn't drifting too fast)
- `loss` trending downward
- `grad_norm` staying within the clip threshold

---

## See also

- [QLoRA](qlora.md)
- [Multi-GPU Training](multi-gpu.md)
- [Precision & Recipes](precision-and-recipes.md)
- [Config reference](../reference/config.md)
- [Back to docs index](../index.mdx)
