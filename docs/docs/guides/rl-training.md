# RL Training (GRPO)

Surogate supports reinforcement learning fine-tuning via GRPO (Group Relative Policy Optimization) in two modes:

- **Native trainer** (`surogate grpo-native`): single-process loop with built-in generation
- **vLLM pipeline** (`surogate grpo-infer` / `grpo-orch` / `grpo-train`): inference server + orchestrator + trainer

This gives you:

- Surogate's near-SOL training throughput (LoRA, QLoRA, FP8)
- Native single-process GRPO with one YAML config file
- Async RL pipeline mode (rollouts, reward computation, sample packing) via vLLM
- vLLM pipeline mode for distributed/separate-process RL training
- Training & Evaluation with [Environments](./rl-environments.md)
  
## Architecture

### Native trainer (single process, no vLLM)

The native trainer runs generation, reward scoring, and optimization in one process:

```bash
surogate grpo-native config.yaml
```

```
┌───────────────────────────────────────────────────────────┐
│  surogate grpo-native (single process)                    │
│                                                           │
│  1. Native generation via C++ engine                      │
│  2. Reward scoring (verifiers or callback)                │
│  3. Advantage computation                                 │
│  4. GRPO backward + optimizer step                        │
└───────────────────────────────────────────────────────────┘
```

Native mode uses one config file (`NativeGRPOConfig`) and does not require `vllm`.

### vLLM pipeline mode (three processes)

The original three-process architecture, useful for multi-node setups or when inference and training run on different GPUs:

```
┌─────────────┐    rollouts    ┌──────────────┐    batches    ┌──────────────────┐
│   vLLM      │ ─────────────> │ Orchestrator │ ────────────> │     Trainer      │
└─────────────┘   new weights  └──────────────┘               └──────────────────┘
       ^                                                             │
       └─────────────── weight broadcast (filesystem) ───────────────┘
```

1. **vLLM inference** (`surogate grpo-infer`) generates completions with log-probabilities
2. **Orchestrator** (`surogate grpo-orch`) collects rollouts, computes rewards and advantages, packs samples into training batches
3. **Surogate trainer** (`surogate grpo-train`) performs the policy gradient update and broadcasts updated weights back to vLLM

The three processes communicate via a shared filesystem directory (`output_dir`).

## Quick Start (native trainer)

This walkthrough uses the native reverse-text config in `examples/grpo-native/reverse-text.yaml`:

```bash
surogate grpo-native examples/grpo-native/reverse-text.yaml
```

If you use `uv`:

```bash
uv run surogate grpo-native examples/grpo-native/reverse-text.yaml
```

Native trainer highlights:

1. One config file (`generation`, `reward`, `sampling`, `loss`, `eval`)
2. No vLLM server or orchestrator process
3. Built-in adaptive generation batch fallback via `generation.max_prompts_per_batch` + `generation.adaptive_batch_on_oom`

## Quick Start (vLLM multi-process mode)

This walkthrough uses the **reverse-text** example — a lightweight task that runs on a single GPU.

### 1. Create the three config files

**`train.yaml`**:

```yaml
model: "Qwen/Qwen3-0.6B"
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
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

loss:
  ratio_type: token
  kl_tau: 0.0
  adv_tau: 1.0
  token_mask_low: 0.125
  token_mask_high: 8.0
  geo_mask_low: 0.1
  geo_mask_high: 10.0
```

**`infer.yaml`**:

```yaml
model: "Qwen/Qwen3-0.6B"
enable_lora: true
max_lora_rank: 32
```

**`orch.yaml`**:

```yaml
model:
  name: "Qwen/Qwen3-0.6B"
  lora_adapter: "default"
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

### 2. Run with three commands

```bash
# Terminal 1: Start vLLM inference server (GPU 0)
CUDA_VISIBLE_DEVICES=0 surogate grpo-infer infer.yaml

# Terminal 2: Start orchestrator (CPU only)
surogate grpo-orch orch.yaml

# Terminal 3: Start Surogate trainer (GPU 1)
CUDA_VISIBLE_DEVICES=1 surogate grpo-train train.yaml
```

The trainer blocks at startup until the orchestrator delivers the first batch. The orchestrator blocks until inference has generated enough rollouts. Once all three are running, the pipeline flows automatically.

### Single-GPU multi-process setup

If you only have one GPU, share it between inference and training. Set `gpu_memory_utilization` in `infer.yaml` to limit vLLM's memory:

```yaml
# infer.yaml
model: "Qwen/Qwen3-0.6B"
enable_lora: true
gpu_memory_utilization: 0.5
```

```bash
CUDA_VISIBLE_DEVICES=0 surogate grpo-infer infer.yaml
surogate grpo-orch orch.yaml
CUDA_VISIBLE_DEVICES=0 surogate grpo-train train.yaml
```

For single-GPU setups, use conservative `gpu_memory_utilization` in `infer.yaml` so inference and training can coexist.

## How the Training Loop Works

### Native trainer loop

Each native GRPO step performs:

1. Generate `generation.num_completions` completions per prompt
2. Score rollouts via `reward` (verifiers or callback function)
3. Compute advantages
4. Run GRPO backward pass and optimizer update

### vLLM pipeline loop

Each training step performs:

1. **Weight broadcast** (after step > 0): Saves LoRA adapter to `{output_dir}/broadcasts/step_{N}/` with a `STABLE` marker file. vLLM polls for this marker to hot-reload weights. If QeRL is enabled, noisy norm weights are saved alongside the adapter.

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

## Asynchronous Off-Policy Training (vLLM pipeline)

Surogate implements asynchronous off-policy training: inference generates rollouts from a policy that may lag behind the trainer by up to `max_async_level` steps (call this $k$). With $k=1$ and equal trainer/inference step times, neither component idles. The default is `max_async_level: 1`; increase it to `2` when weight broadcasts have higher latency (e.g., over a network).

### Step Semantics

Surogate uses a global step counter $n = 1, 2, 3, \ldots$ to tag all artifacts:

- **Trainer**: Produces policy $\pi_n$ with weights $\theta_n$ from rollouts $(x_n, y_n)$
- **Inference**: Produces rollouts $(x_n, y_n)$ from policy $\pi_{\max(0,\, n-k)}$

The off-policy gap is at most $k$ steps. Rollouts whose gap exceeds `max_off_policy_steps` are discarded by the orchestrator.

### Loss Objective

The loss is a token-level variant of the [AIPO objective](https://arxiv.org/abs/2505.24034) (introduced in Llama-RL), without the entropy and KL terms. For $N$ prompts, each with a group of $G$ rollouts:

$$
\mathcal{J}(\theta)
= \frac{1}{\sum_{j=1}^N \sum_{i=1}^G |y_i^{(j)}|}
\sum_{j=1}^N
\sum_{i=1}^G
\sum_{t=1}^{|y_i^{(j)}|}
\min\!\left(
\frac{\pi_\theta(y^{(j)}_{i,t}\mid x_j, y^{(j)}_{i,<t})}{\mu(y^{(j)}_{i,t}\mid x_j, y^{(j)}_{i,<t})},\;
\delta
\right)\hat{A}^{(j)}_{i,t}
$$

where $\mu$ is the rollout policy, $\pi_\theta$ is the current trainer policy, $\hat{A}_{i,t}$ is the token-level advantage, and $\delta$ is the importance-sampling clip ratio (`token_mask_high`). The token masking thresholds (`token_mask_low`, `token_mask_high`, `geo_mask_low`, `geo_mask_high`) guard against tokens or sequences with extreme importance ratios caused by the off-policy gap.

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

### QeRL Adaptive Quantization Noise

QeRL (Quantization-enhanced RL, [arXiv:2510.11696](https://arxiv.org/abs/2510.11696)) adds controlled Gaussian noise to the inference model's RMSNorm weights during rollout generation. This encourages exploration by making the inference policy slightly stochastic, improving reward signal diversity in early training.

#### How it works

1. At each training step, the noise scheduler computes a sigma value based on a geometric decay schedule
2. All RMSNorm weights (input_layernorm, post_attention_layernorm, etc.) are read from the base model
3. Gaussian noise N(0, sigma^2) is added to produce noisy copies
4. The noisy norm weights are applied to vLLM's model before the next rollout batch
5. The trainer always uses the clean (non-noisy) weights for gradient computation

The sigma decays geometrically from `sigma_start` to `sigma_end` over `num_stages` intervals. The first interval uses sigma=0 (no noise) to establish a baseline.

#### Configuration

Add `noise_scheduler` to `train.yaml`:

```yaml
noise_scheduler:
  enabled: true
  sigma_start: 5e-2    # Initial noise level
  sigma_end: 5e-4      # Final noise level
  num_stages: 10        # Number of decay intervals
```

| Parameter     | Default | Description                                     |
| ------------- | ------- | ----------------------------------------------- |
| `enabled`     | `false` | Enable QeRL noise injection                     |
| `sigma_start` | `5e-2`  | Initial noise standard deviation                |
| `sigma_end`   | `5e-4`  | Final noise standard deviation                  |
| `num_stages`  | `10`    | Number of geometric decay intervals             |

#### When to use QeRL

- Models that converge too quickly to a local optimum
- Tasks where reward signal diversity is low (many rollouts get the same reward)
- Pre-quantized models (NVFP4, FP8) where quantization already introduces noise — QeRL amplifies this effect in a controlled way

QeRL works with both native trainer mode and the vLLM pipeline.

### On-Policy Distillation

On-policy distillation uses a teacher model to provide dense token-level feedback alongside (or instead of) the reward signal. The student generates rollouts, and the teacher's log-probabilities guide the student to stay close to stronger behavior while still learning from rewards.

The loss coefficient for each token becomes:

```
coeff = importance_ratio * (adv_tau * advantages + teacher_tau * (teacher_logprob - trainer_logprob) - kl_tau * log_ratio)
```

The `teacher_tau * (teacher_logprob - trainer_logprob)` term is positive when the teacher assigns higher probability to a token than the student, pulling the student toward the teacher's distribution.

#### Enabling distillation

Set `teacher_tau > 0` in your GRPO loss config:

- **vLLM pipeline**: set `teacher_tau` in `train.yaml` and configure `teacher_model` in `orch.yaml`
- **Native trainer**: set `loss.teacher_tau` and `teacher_model` in the same native config file

**`train.yaml`** — add `teacher_tau` to the loss block:

```yaml
loss:
  ratio_type: token
  adv_tau: 1.0
  teacher_tau: 0.5   # blend: half reward signal, half teacher signal
  kl_tau: 0.0
```

**`orch.yaml`** — add a `teacher_model` section to the orchestrator:

```yaml
model:
  name: "Qwen/Qwen3-0.6B"
  lora_adapter: "default"
  lora_rank: 16

teacher_model:
  model:
    name: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

env:
  - id: reverse-text

batch_size: 128
rollouts_per_example: 16
seq_len: 2048
max_steps: 40
```

If the teacher inference server is already running externally, point to it with a `client` entry instead:

```yaml
teacher_model:
  model:
    name: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  client:
    base_url: ["http://teacher-server:8000/v1"]
```

#### Pure distillation (no reward verification)

For agentic tasks where verification is expensive (code execution, tool use, multi-turn), skip reward scoring entirely and learn only from the teacher signal:

**`train.yaml`**:

```yaml
loss:
  adv_tau: 0.0       # disable reward signal
  teacher_tau: 1.0   # learn only from teacher
  kl_tau: 0.0
```

**`orch.yaml`**:

```yaml
teacher_model:
  model:
    name: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

buffer:
  skip_verification: true  # skip reward scoring

env:
  - id: your-env
```

#### Monitoring

When teacher log-probabilities are present, the trainer logs an additional metric:

| Metric       | Description                                                   |
| ------------ | ------------------------------------------------------------- |
| `teacher_kl` | Mean KL divergence from teacher to student (lower = closer to teacher) |

A decreasing `teacher_kl` confirms the student is learning to match the teacher's distribution.


### Learning Rate

RL training typically uses a lower learning rate than SFT (5e-7 to 5e-5). Start with `5e-6` and adjust based on the KL divergence metrics.

### Masking Thresholds

The importance ratio masks are critical for training stability:

- **Token masks** (`token_mask_low`/`token_mask_high`): Filter individual tokens with extreme policy drift. The defaults (0.125, 8.0) allow up to 8x ratio before masking.
- **Geometric masks** (`geo_mask_low`/`geo_mask_high`): Filter entire sequences based on the geometric mean of token ratios. Catches sequences where many tokens have drifted moderately.
- If you see high `is_masked_frac` in logs (>50%), your policy is drifting too fast. Reduce the learning rate or increase `kl_tau`.

In native trainer mode, masking is controlled by `loss.ipo_mask_low` and `loss.ipo_mask_high`.

### KL Penalty

Setting `kl_tau > 0` adds a KL penalty that keeps the policy close to the reference (inference) policy. This prevents reward hacking but slows learning. Start with `kl_tau: 0.0` and increase if the policy diverges.

### Gradient Accumulation

Unlike SFT where gradient accumulation increases effective batch size, in RL training it controls how many packed micro-batches are processed per optimizer step. With `gradient_accumulation_steps: 1` and packed sequences, each step processes one densely-packed sequence.

## Monitoring

Native trainer commonly logs metrics like `reward/mean`, `train/policy_loss`, `train/mismatch_kl`, `train/grad_norm`, `sampling/temperature`, and `time/step`.

### vLLM pipeline metrics

The trainer logs these GRPO-specific metrics at each step:

| Metric       | Description                                             |
| ------------ | ------------------------------------------------------- |
| `kl`         | Mean KL divergence between current and inference policy |
| `masked`     | Fraction of loss-eligible tokens that were masked       |
| `tokens`     | Total loss-eligible tokens in the step                  |
| `loss`       | Training loss (from the backward pass)                  |
| `grad_norm`  | Gradient norm after clipping                            |
| `teacher_kl` | KL from teacher to student (only when `teacher_tau > 0`) |

A healthy training run shows:

- `kl` gradually increasing from near-zero (policy is improving)
- `masked` staying below 30-40% (policy isn't drifting too fast)
- `loss` trending downward
- `grad_norm` staying within the clip threshold

---

## Configuration Reference

### Native trainer config (`grpo-native`)

The native trainer uses a single YAML file (`NativeGRPOConfig`) and inherits all core SFT fields (model, LoRA, QLoRA, optimizer, precision, multi-GPU).

Key native fields:

| Key                      | Default | Description                                                          |
| ------------------------ | ------- | -------------------------------------------------------------------- |
| `model`                  | (required) | HuggingFace model ID or local path                               |
| `output_dir`             | `output` | Output directory for checkpoints/logs/final export                 |
| `gpus`                   | `1`     | Number of GPUs                                                       |
| `max_steps`              | `-1`    | Training steps                                                       |
| `problems_per_step`      | `8`     | Number of unique prompts sampled per optimizer step                  |
| `save_steps`             | `0`     | Save checkpoint every N steps (`0` disables periodic checkpointing)  |
| `checkpoint_dir`         | `null`  | Checkpoint root directory override                                   |
| `resume_from_checkpoint` | `null`  | Path to checkpoint directory to resume from                          |
| `doc_masking`            | `true`  | Enable document-level attention masking for packed RL micro-batches  |
| `teacher_model`          | `null`  | Optional teacher model path for distillation (`loss.teacher_tau > 0`) |

**Generation** (`generation.*`):

| Key                               | Default | Description                                                                           |
| --------------------------------- | ------- | ------------------------------------------------------------------------------------- |
| `generation.num_completions`      | `4`     | Number of completions per prompt                                                      |
| `generation.max_gen_len`          | `512`   | Max generated tokens per completion                                                   |
| `generation.top_p`                | `1.0`   | Nucleus sampling                                                                      |
| `generation.top_k`                | `0`     | Top-k cutoff (`0` disables top-k filtering)                                           |
| `generation.min_p`                | `0.0`   | Minimum-probability sampling floor                                                    |
| `generation.prefill_chunk_size`   | `256`   | Prefill chunk size (`0` disables chunked prefill)                                     |
| `generation.max_prompts_per_batch` | `0`     | Max prompts per native `generate()` call (`0` = all prompts, fastest/highest VRAM) |
| `generation.adaptive_batch_on_oom` | `true`  | Automatically retry generation with smaller prompt batches on OOM                   |

**Sampling** (`sampling.*`):

| Key                              | Default | Description                                                               |
| -------------------------------- | ------- | ------------------------------------------------------------------------- |
| `sampling.temperature`           | `1.0`   | Constant sampling temperature                                             |
| `sampling.temp_scheduler.type`   | `null`  | Temperature schedule type (`linear`, `cosine`) when scheduler is enabled |
| `sampling.temp_scheduler.start_temperature` | `null` | Temperature at step 0 when scheduler is enabled |
| `sampling.temp_scheduler.end_temperature`   | `null` | Temperature at final step when scheduler is enabled |
| `sampling.temp_scheduler.total_steps`       | `null` | Steps for schedule completion (falls back to training max steps) |

**Reward** (`reward.*`):

| Key                            | Default     | Description                                                          |
| ------------------------------ | ----------- | -------------------------------------------------------------------- |
| `reward.mode`                  | `verifiers` | Reward source: `verifiers` or `callback`                            |
| `reward.env[]`                 | `null`      | Verifiers environments (required for `mode: verifiers`)             |
| `reward.reward_fn_import_path` | `null`      | Import path to callback reward function (required for `mode: callback`) |
| `reward.dataset_path`          | `null`      | JSONL path with `prompt` field (callback mode input prompts)        |
| `reward.prompts`               | `null`      | Inline list of prompts (callback mode input prompts)                |
| `reward.multiturn`             | `false`     | Enable multi-turn verifiers rollout extraction path                 |

**Loss** (`loss.*`):

| Key                  | Default | Description                                  |
| -------------------- | ------- | -------------------------------------------- |
| `loss.kl_tau`        | `1e-3`  | KL regularization coefficient                 |
| `loss.adv_tau`       | `1.0`   | Advantage scaling                             |
| `loss.teacher_tau`   | `0.0`   | Teacher distillation scaling                  |
| `loss.ipo_mask_low`  | `0.2`   | Lower IPO mask threshold                      |
| `loss.ipo_mask_high` | `0.2`   | Upper IPO mask threshold                      |

**Evaluation** (`eval.*`):

| Key                         | Default | Description                                      |
| --------------------------- | ------- | ------------------------------------------------ |
| `eval.env[]`                | `null`  | Eval environments                                |
| `eval.num_examples`         | `100`   | Number of eval examples per environment          |
| `eval.rollouts_per_example` | `1`     | Rollouts per eval example                        |
| `eval.interval`             | `10`    | Evaluate every N training steps                  |
| `eval.temperature`          | `0.0`   | Eval temperature (`0.0` = greedy)                |
| `eval.max_gen_len`          | `null`  | Eval max generation length (falls back to train) |

### vLLM inference config

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
| `gpu_memory_utilization`  | `0.9`          | Fraction of GPU memory for KV cache |
| `weight_broadcast_type`   | `"filesystem"` | How to receive weight updates (`filesystem` or `nccl`) |
| `reasoning_parser`        | `null`         | Parser for extracting reasoning content                |
| `enable_auto_tool_choice` | `false`        | Enable auto tool choice                                |
| `rope_scaling`            | `null`         | RoPE scaling configuration dict                        |

### vLLM orchestrator config

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
| `sampling.temp_scheduler.type`              | ---     | Temperature schedule shape: `linear` or `cosine`              |
| `sampling.temp_scheduler.start_temperature` | ---     | Temperature at step 0                                         |
| `sampling.temp_scheduler.end_temperature`   | ---     | Temperature at final step                                     |
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
| `client.elastic.hostname`      | ---                            | DNS hostname for elastic pool discovery |
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
| `advantage.import_path`          | ---         | (custom only) Import path to advantage function   |
| `advantage.kwargs`               | ---         | (custom only) Kwargs passed to advantage function |

**Filters** (`filters[]`):

| Key                            | Default  | Description                                              |
| ------------------------------ | -------- | -------------------------------------------------------- |
| `filters[].type`               | ---      | `"gibberish"` or `"repetition"`                          |
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

### vLLM trainer config (inherited from SFT)

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

#### `noise_scheduler` (nested)

QeRL Adaptive Quantization Noise. See [QeRL section](#qerl-adaptive-quantization-noise) above.

| Parameter     | Default | Description                                     |
| ------------- | ------- | ----------------------------------------------- |
| `enabled`     | `false` | Enable QeRL noise injection                     |
| `sigma_start` | `5e-2`  | Initial noise standard deviation                |
| `sigma_end`   | `5e-4`  | Final noise standard deviation                  |
| `num_stages`  | `10`    | Number of geometric decay intervals             |

#### `transport_type`

How the orchestrator delivers batches to the trainer.

| Value          | Description                                                    |
| -------------- | -------------------------------------------------------------- |
| `"filesystem"` | (default) IPC via shared filesystem. Simple and reliable       |
| `"zmq"`        | IPC via ZeroMQ sockets. Lower latency for same-host processes   |

#### `max_async_level`

Controls how many weight broadcasts can be in-flight simultaneously. Default: `1`. Higher values allow the inference engine to lag behind the trainer by more steps.

## See also

- [QLoRA](qlora.md)
- [Multi-GPU Training](multi-gpu.md)
- [Precision & Recipes](precision-and-recipes.md)
- [Config reference](../reference/config.md)
- [Back to docs index](../index.mdx)
