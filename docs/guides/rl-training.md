# RL Training (GRPO)

Surogate supports reinforcement learning fine-tuning via GRPO (Group Relative Policy Optimization). The pipeline coordinates a vLLM inference server, a GRPO orchestrator, and the Surogate trainer.

This gives you:

- Surogate's near-SOL training throughput (LoRA, QLoRA, FP8)
- Async RL pipeline (rollouts, reward computation, sample packing)
- vLLM for fast inference and generation
- Two single-command runners — split-GPU (`grpo`) and co-locate (`grpo-colocate`)
- Training & Evaluation with [Environments](./rl-environments.md)

## Architecture

GRPO training supports three deployment modes:

| Mode | Command | When to use |
| --- | --- | --- |
| **Split-GPU** | `surogate grpo` | Multiple GPUs available; want vLLM and the trainer on disjoint GPU sets |
| **Co-locate** | `surogate grpo-colocate` | Single GPU, or any case where vLLM and the trainer should share GPUs (zero-copy base weights) |
| **Multi-process** | `surogate grpo-infer` + `grpo-orch` + `grpo-train` | Multi-node, or when each component must run in its own process |

### Split-GPU mode

vLLM runs in a spawned subprocess with its own `CUDA_VISIBLE_DEVICES`; the trainer runs in the parent process on the remaining GPUs. Weight broadcasts go through the filesystem (no shared GPU memory):

```
surogate grpo --train train.yaml --infer infer.yaml --orch orch.yaml \
    --vllm-gpus 0,1,2,3 --trainer-gpus 4,5,6,7
```

```
┌───────────────────────────────────────────────────────────┐
│  surogate grpo (split mode)                               │
│                                                           │
│  Parent process (CUDA_VISIBLE_DEVICES=<trainer ids>)      │
│   ├─ Trainer thread                                       │
│   │   └─ Loads its own copy of base weights from disk     │
│   └─ Orchestrator (main async event loop)                 │
│                                                           │
│  Spawned vLLM subprocess (CUDA_VISIBLE_DEVICES=<vllm ids>)│
│   └─ vLLM serves /v1/chat/completions                     │
│                                                           │
│  Communication:                                           │
│   - Rollouts:  Orchestrator → vLLM via HTTP               │
│   - Batches:   Orchestrator → Trainer via filesystem/zmq  │
│   - Weights:   Trainer → vLLM via filesystem broadcast    │
└───────────────────────────────────────────────────────────┘
```

The CLI is the source of truth for the trainer GPU count: `--trainer-gpus 4,5,6,7` automatically sets `train.gpus = 4`, and (for MoE models) `ep_size` is set to the same value. The YAML `gpus` field is ignored in this mode. The vLLM count must equal `infer.dp * infer.tp`.

### Co-locate mode

A single `surogate grpo-colocate` command starts all three components in one process with zero-copy GPU weight sharing:

```
surogate grpo-colocate --train train.yaml --infer infer.yaml --orch orch.yaml
```

```
┌───────────────────────────────────────────────────────────┐
│  surogate grpo-colocate (single process)                  │
│                                                           │
│  1. vLLM server (background thread, engine in subprocess) │
│     └─ Owns quantized base weights on GPU                 │
│     └─ Serves /v1/chat/completions                        │
│                                                           │
│  2. Trainer (background thread)                           │
│     └─ Borrows vLLM's quantized weights (zero-copy IPC)   │
│     └─ Dequantizes on-the-fly for forward/backward        │
│                                                           │
│  3. Orchestrator (main async event loop)                  │
│     └─ Sends rollout requests to vLLM via HTTP            │
│     └─ Computes rewards and advantages                    │
│     └─ Sends training batches via filesystem transport    │
│     └─ Signals LoRA weight updates to vLLM                │
└───────────────────────────────────────────────────────────┘
```

**How weight sharing works**: At startup, vLLM loads and quantizes the base model on GPU. The trainer then receives GPU pointers to those quantized tensors via CUDA IPC — no copy, no duplicate memory. Only the base weights (linear layers) are shared; small non-quantized weights (norms, embeddings) are loaded separately from disk. LoRA adapter updates are small (~10 MB) and go through the filesystem.

**Automatic memory management**: The trainer's GPU memory footprint (LoRA parameters, activations, dequantization buffers) is estimated automatically, and `gpu_memory_utilization` is computed so vLLM uses the remaining GPU memory for its KV cache. No manual tuning needed.

### Multi-process mode

The original three-process architecture, useful for multi-node setups or any case where each component must live in its own process:

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

## Quick Start (split-GPU mode)

This walkthrough uses the **reverse-text** example. Split-GPU mode needs at least two GPUs (one for vLLM, one for the trainer). For a single-GPU setup, see [Co-locate mode](#quick-start-co-locate-mode) below.

### 1. Create the three config files

**`train.yaml`**:

```yaml
model: "Qwen/Qwen3-0.6B"
output_dir: ./outputs
gpus: 1  # ignored in split mode (auto-derived from --trainer-gpus)

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

env:
  - id: reverse-text

batch_size: 128
rollouts_per_example: 16
seq_len: 2048
max_steps: 40

sampling:
  max_tokens: 128
```

### 2. Run with a single command

```bash
surogate grpo --train train.yaml --infer infer.yaml --orch orch.yaml \
    --vllm-gpus 0 --trainer-gpus 1
```

That's it. The command:

1. Spawns the vLLM inference server as a subprocess with `CUDA_VISIBLE_DEVICES=0`
2. Polls `/health` until vLLM is ready
3. Starts the Surogate trainer in a background thread, restricted to GPU 1
4. Runs the orchestrator, which coordinates rollouts and training steps
5. Shuts down vLLM and the trainer cleanly when `max_steps` is reached

`--trainer-gpus` doubles as the trainer's GPU count, so the YAML `gpus` field is optional. For MoE models, `ep_size` is also auto-set to the same count.

## Quick Start (co-locate mode)

For a single-GPU setup, run vLLM and the trainer on the same GPU with `surogate grpo-colocate`:

```bash
surogate grpo-colocate --train train.yaml --infer infer.yaml --orch orch.yaml
```

You do **not** need to set `gpu_memory_utilization` in co-locate mode — it is computed automatically based on the trainer's memory requirements. Base weights are shared zero-copy via CUDA IPC.

## Quick Start (multi-process mode)

For multi-node setups, or any case where you want each component in its own process, use the three individual commands:

```bash
# Terminal 1: Start vLLM inference server (GPU 0)
CUDA_VISIBLE_DEVICES=0 surogate grpo-infer infer.yaml

# Terminal 2: Start orchestrator (CPU only)
surogate grpo-orch orch.yaml

# Terminal 3: Start Surogate trainer (GPU 1)
CUDA_VISIBLE_DEVICES=1 surogate grpo-train train.yaml
```

The trainer blocks at startup until the orchestrator delivers the first batch. The orchestrator blocks until inference has generated enough rollouts. Once all three are running, the pipeline flows automatically.

## How the Training Loop Works

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

## Asynchronous Off-Policy Training

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

## Split-GPU Mode Details

### GPU assignment

The `--vllm-gpus` and `--trainer-gpus` CLI flags use **driver-level GPU indices** (the same values you would put in `CUDA_VISIBLE_DEVICES`). The two sets must be disjoint.

- `--vllm-gpus N0,N1,...` count must equal `infer.dp * infer.tp`.
- `--trainer-gpus M0,M1,...` count overrides `train.gpus`. The YAML field is ignored when present.
- For MoE models, `ep_size` is set to the trainer GPU count automatically. The runner re-validates that `num_experts` is divisible by `ep_size`; if not, the run fails fast with a clear error.

The CLI applies `CUDA_VISIBLE_DEVICES=<trainer ids>` to the parent process **before** any torch import, and `CUDA_VISIBLE_DEVICES=<vllm ids>` to the spawned vLLM child via its own environment. The two values are interpreted by the CUDA driver independently — the parent's mask does not propagate to the child.

### Weight broadcast

In split mode, weight broadcasts use the filesystem backend regardless of what is configured in the YAML. Each broadcast writes the LoRA adapter (~10 MB) to `{output_dir}/broadcasts/step_N/`; vLLM polls for the `STABLE` marker file and hot-reloads the adapter.

### Lifecycle

The `surogate grpo` process owns the vLLM subprocess: it terminates and (if necessary) kills the child during shutdown. If the parent is `SIGKILL`ed, the vLLM subprocess and its engine workers can be left orphaned holding GPU memory — clean up with `nvidia-smi` if that happens.

## Co-locate Mode Details

### How weight sharing works

In co-locate mode, the base model is loaded only once:

1. vLLM starts first and loads the model (quantized weights go to GPU)
2. The trainer receives GPU pointers to vLLM's quantized tensors via CUDA IPC
3. Both vLLM and the trainer read from the same GPU memory — zero copy, zero duplication
4. Only LoRA adapter updates (~10 MB) are written to disk and reloaded by vLLM

This saves roughly 50% of GPU memory for the base model. For example, a Qwen3-8B model in NF4 takes ~4.5 GB — in co-locate mode this is shared instead of duplicated.

### Automatic gpu_memory_utilization

In co-locate mode, `gpu_memory_utilization` is computed automatically by estimating the trainer's GPU memory needs:

| Component              | Estimate                                                        |
| ---------------------- | --------------------------------------------------------------- |
| LoRA parameters        | Weight + master copy + gradient + 8-bit optimizer (6 bytes/param) |
| Activations            | Working set (6 BF16 tensors/layer) + logits + residual checkpoints |
| Dequantization buffers | 3 concurrent BF16 buffers (max weight size)                     |
| Embeddings + LM head   | vocab_size * hidden_size * 2 bytes (BF16, loaded from disk)     |
| Fixed overhead         | 2.5 GB (CUDA context, cuDNN workspace, allocator fragmentation) |

The remaining GPU memory (minus a 10% safety margin) is assigned to vLLM. You can override this by setting `gpu_memory_utilization` explicitly in `infer.yaml`.

### Multi-GPU co-locate

Both the trainer and vLLM use data parallelism (`tp=1, dp=N`). Each GPU has a full model replica, and weight sharing is 1:1 per GPU:

```yaml
# train.yaml
gpus: 2

# infer.yaml
dp: 2
```

### Supported quantization formats

Co-locate weight sharing works with all quantization formats since it operates on raw GPU pointers:

- **BnB NF4** — Packed uint8 data + FP32 scales
- **FP8** (E4M3) — FP8 data + FP32 block scales
- **NVFP4** — Packed FP4 data + FP8 scales + FP32 global scale

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

QeRL works in all three deployment modes (split, co-locate, multi-process).

### On-Policy Distillation

On-policy distillation uses a teacher model to provide dense token-level feedback alongside (or instead of) the reward signal. The student generates rollouts, and the teacher's log-probabilities guide the student to stay close to stronger behavior while still learning from rewards.

The loss coefficient for each token becomes:

```
coeff = importance_ratio * (adv_tau * advantages + teacher_tau * (teacher_logprob - trainer_logprob) - kl_tau * log_ratio)
```

The `teacher_tau * (teacher_logprob - trainer_logprob)` term is positive when the teacher assigns higher probability to a token than the student, pulling the student toward the teacher's distribution.

#### Enabling distillation

Set `teacher_tau > 0` in `train.yaml` and configure the teacher in `orch.yaml`. The orchestrator computes teacher log-probabilities and delivers them alongside advantages in each micro-batch — no changes to the trainer command are needed.

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


### RULER (LLM-as-Judge Rewards)

RULER ([Relative Universal LLM-Elicited Rewards](https://art.openpipe.ai/fundamentals/ruler)) replaces or augments the env's hand-crafted rubric with an LLM judge that ranks all rollouts in a group. It works because GRPO normalizes scores within each group — only the *relative* ordering matters, so the judge doesn't need to produce calibrated absolute values.

Use it when:

- You don't have a verifiable reward function for your task (open-ended generation, agentic tasks)
- Hand-crafting a rubric would take days; you'd rather pay for judge tokens
- You want richer reward signal than a pass/fail check from the env
- You need a quick baseline before investing in a custom rubric

#### How it works

1. The orchestrator wraps each train env's rubric in a `vf.RubricGroup` that adds a group-level RULER reward function
2. When all `rollouts_per_example` rollouts of a scenario complete, the orchestrator sends the full group to the judge in a single request
3. The judge returns one score in `[0, 1]` per rollout plus a short explanation
4. Scores become per-rollout rewards; GRPO computes advantages via group-mean normalization as usual

Group judging runs concurrently with rollout generation — the scheduler dispatches each completed group as an `asyncio.Task` so judge latency is amortized over the next step's rollouts rather than blocking the pipeline. Concurrency is capped globally by `ruler.max_concurrent_judges`.

RULER requires `rollouts_per_example >= 2` (a single rollout has no peers to rank against); ART's launch report recommends **4–8 rollouts per scenario** for stable relative rankings.

#### Configuration

The judge runs as a *separate* OpenAI-compatible inference server — start it independently before launching `surogate grpo`:

**Terminal 1** — judge vLLM:

```bash
CUDA_VISIBLE_DEVICES=2 surogate grpo-infer judge.yaml
```

where `judge.yaml` is a stock inference config (`enable_lora: false`, the model you want to judge with). Any OpenAI-compatible endpoint works — local vLLM, hosted API, or a LiteLLM proxy.

**Terminal 2** — `surogate grpo` with RULER enabled in `orch.yaml`:

```yaml
verification:
  enabled: true   # required: RULER plugs into the deferred group-scoring path

batch_size: 16
rollouts_per_example: 4   # >= 2 required; 4-8 recommended

ruler:
  enabled: true
  mode: replace                  # 'replace' | 'add' | 'metric'
  judge_model: "Qwen/Qwen3-1.7B"
  judge:
    base_url: ["http://localhost:8001/v1"]
    api_key_var: RULER_JUDGE_API_KEY
    timeout: 90.0
  weight: 1.0
  max_concurrent_judges: 8
  sampling:
    temperature: 0.0
    max_completion_tokens: 1024
  swallow_exceptions: true
  debug: true
```

#### Modes

| Mode      | Behavior                                                                                  |
| --------- | ----------------------------------------------------------------------------------------- |
| `replace` | Drop the env's existing rubric. RULER is the sole reward source.                          |
| `add`     | Sum RULER's score with the env's existing rubric (each weighted). Combines both signals.  |
| `metric`  | Run RULER for observability only — `weight=0`, the env's rubric drives the reward.        |

In `add` and `metric` mode, the env must have a rubric that produces non-zero rewards (otherwise `metric` mode trains on zeros). `replace` is the right default when the env has no usable rubric.

#### Custom rubric

The built-in rubric is generic (efficiency, goal achievement, partial credit, hallucination penalty). Override per-task when needed:

```yaml
ruler:
  enabled: true
  rubric: |
    - Reward concise, well-formatted answers.
    - Penalize tool-call hallucinations and off-topic responses.
    - Award partial credit for incomplete but on-track attempts.
```

The env's system prompts give the judge implicit context — make sure each env clearly states the agent's goal.

#### Cost tracking

Set per-token rates (USD per 1M tokens) to surface a cost-in-dollars metric:

```yaml
ruler:
  cost:
    input_per_million: 0.15
    output_per_million: 0.60
```

#### Monitoring

When RULER is enabled, the orchestrator adds these metrics:

| Metric                         | Description                                                                              |
| ------------------------------ | ---------------------------------------------------------------------------------------- |
| `ruler/total_judge_calls`      | Total judge HTTP calls this step (= `batch_size / rollouts_per_example` in steady state) |
| `ruler/total_input_tokens`     | Step total of judge prompt tokens                                                        |
| `ruler/total_output_tokens`    | Step total of judge completion tokens                                                    |
| `ruler/total_judge_latency_ms` | Step total of judge compute time (sum of per-call durations)                             |
| `ruler/total_judge_cost_usd`   | Step total of judge USD cost (zero unless `ruler.cost.*` is set)                         |
| `ruler/score_mean`             | Mean RULER score across the batch                                                        |
| `ruler/score_std`              | Std-dev of RULER scores — collapses to 0 when the judge is undiscriminating              |
| `ruler/judge_failure_rate`     | Fraction of groups where the judge call raised; >0 indicates trouble                     |
| `metrics/ruler_*`              | Per-rollout averages of the same fields                                                  |

Healthy signals during a run:

- `ruler/total_judge_calls` exactly equals `batch_size / rollouts_per_example` (no retries firing)
- `ruler/judge_failure_rate == 0`
- `ruler/score_std > 0` and ideally > 0.1 (the judge is meaningfully discriminating between rollouts)
- `ruler/total_judge_latency_ms` × `max_concurrent_judges`-amortization fits within `time/step` — otherwise the judge is the bottleneck (use a smaller judge or add replicas to `ruler.judge.base_url`)
- Under `mode: replace`, `reward/mean` should equal `ruler/score_mean` exactly

#### Evaluating a RULER-trained model

RULER attaches **only to training environments**. Two paths run the env's *native, ground-truth* rubric instead — by design:

| Path                                | Rubric used                       | Why                                                                                |
| ----------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------- |
| `eval:` block in `orch.yaml`        | Eval env's own rubric             | Online evaluation during training measures task improvement, not judge fit          |
| `surogate vf-eval` (standalone CLI) | Env's own rubric                  | `vf-eval` doesn't import the orchestrator config; it loads the env directly         |

This separation is the point. If you trained against the judge, evaluating against the same judge is circular — the model just learns to look good to *that* judge. Evaluating against the env's hand-crafted verifier tells you whether the RULER training actually improved task performance.

The recommended workflow:

1. Train with RULER as the reward source (`ruler.enabled: true`, optionally `mode: replace` if no native rubric exists).
2. Evaluate the trained model with `surogate vf-eval` against the env's native rubric:

   ```bash
   surogate vf-eval markdown-table-qa \
       --model Qwen/Qwen3-0.6B \
       --api-base-url http://localhost:8007/v1 \
       --num-examples 200 --rollouts-per-example 1 \
       --save-results
   ```

3. Compare to a baseline run (untrained model, or trained without RULER) to isolate the RULER training effect.

**When the env has no usable rubric** (open-ended generation, agentic tasks), `vf-eval` can't measure ground truth. Two pragmatic options:

- Use a *different, stronger* judge model for offline evaluation than the one used during training. Hold it out from the training loop entirely. This breaks the circularity but doesn't eliminate judge bias.
- Run `ruler.mode: metric` during training so the judge logs `ruler/score_mean` for observability without driving the gradient. Useful for sanity-checking judge alignment as you iterate on the rubric prompt.

A complete worked example — `train.yaml`, `infer.yaml`, `judge.yaml`, `orch.yaml`, and a step-by-step runbook — lives in [`examples/ruler/`](https://github.com/invergent-ai/surogate/tree/main/examples/ruler).

#### Reasoning-mode judges and token budgets

Reasoning models (Qwen3 with thinking on, OpenAI o-series, DeepSeek-R1) can consume thousands of tokens chain-of-thought before they begin emitting the JSON schema. If the judge runs out of tokens mid-response, the OpenAI SDK raises `LengthFinishReasonError`. RULER catches this and retries with a doubled `max_completion_tokens` (capped at 16K), but the wasted first attempt costs latency and tokens. Two things help:

1. Set a generous budget up front: `ruler.sampling.max_completion_tokens: 4096` is a safe starting point for non-reasoning judges; bump to 8192+ if you want a reasoning judge.
2. Disable reasoning on the judge entirely. For Qwen3:

   ```yaml
   ruler:
     extra_body:
       chat_template_kwargs:
         enable_thinking: false
   ```

   For OpenAI o-series, set `ruler.sampling.reasoning_effort: minimal`.


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

### Inference config

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
| `gpu_memory_utilization`  | `0.9`          | Fraction of GPU memory for KV cache (auto in co-locate mode) |
| `weight_broadcast_type`   | `"filesystem"` | How to receive weight updates (`filesystem` or `nccl`) |
| `reasoning_parser`        | `null`         | Parser for extracting reasoning content                |
| `enable_auto_tool_choice` | `false`        | Enable auto tool choice                                |
| `rope_scaling`            | `null`         | RoPE scaling configuration dict                        |

### Orchestrator config

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

**RULER** (`ruler.*`) — see [RULER (LLM-as-Judge Rewards)](#ruler-llm-as-judge-rewards) above:

| Key                                     | Default                 | Description                                                                                           |
| --------------------------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------- |
| `ruler.enabled`                         | `false`                 | Master switch                                                                                         |
| `ruler.mode`                            | `"replace"`             | `"replace"` (RULER alone), `"add"` (sum with env rubric), `"metric"` (observability only, weight 0)   |
| `ruler.judge_model`                     | `null`                  | Judge model name (must be served by `judge.base_url`); required when `enabled: true`                  |
| `ruler.judge.base_url`                  | `null`                  | List of OpenAI-compatible judge endpoints (round-robined); required when `enabled: true`              |
| `ruler.judge.api_key_var`               | `"RULER_JUDGE_API_KEY"` | Env var holding the judge API key (use any string for unauthenticated local vLLM)                     |
| `ruler.judge.timeout`                   | `120.0`                 | Per-request timeout in seconds                                                                        |
| `ruler.judge.connect_timeout`           | `5.0`                   | Per-request connect timeout in seconds                                                                |
| `ruler.judge.max_connections`           | `256`                   | Max HTTP connections per endpoint                                                                     |
| `ruler.judge.max_keepalive_connections` | `256`                   | Max keep-alive HTTP connections per endpoint                                                          |
| `ruler.judge.max_retries`               | `4`                     | Auto-retries inside the OpenAI client                                                                 |
| `ruler.judge.headers`                   | `{}`                    | Extra HTTP headers sent with each judge request                                                       |
| `ruler.rubric`                          | (built-in default)      | Free-form rubric text passed to the judge as grading guidance                                         |
| `ruler.weight`                          | `1.0`                   | Multiplier on RULER score before combining with the reward; ignored in `metric` mode                  |
| `ruler.max_concurrent_judges`           | `32`                    | Global cap on concurrent judge HTTP calls across all envs (`null` = unbounded)                        |
| `ruler.request_timeout`                 | `null`                  | Per-judge-call timeout override (falls back to `judge.timeout`)                                       |
| `ruler.max_retries_on_parse_error`      | `2`                     | Retries when the judge returns malformed JSON                                                         |
| `ruler.swallow_exceptions`              | `true`                  | When true, judge failures drop the group instead of crashing the orchestrator                         |
| `ruler.debug`                           | `false`                 | Log per-group judge reasoning at INFO (otherwise DEBUG)                                               |
| `ruler.extra_body`                      | `{}`                    | Extra request-body fields forwarded to the judge (e.g. vLLM `guided_json` knobs)                      |
| `ruler.sampling`                        | `{}`                    | Sampling overrides for the judge call (`temperature`, `max_completion_tokens`, `reasoning_effort`)    |
| `ruler.cost.input_per_million`          | `null`                  | USD per 1M input tokens (drives `ruler/total_judge_cost_usd`)                                         |
| `ruler.cost.output_per_million`         | `null`                  | USD per 1M output tokens                                                                              |

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

### Trainer config (inherited from SFT)

GRPO trainer configs inherit all fields from `SFTConfig`. The most relevant ones for RL training:

| Parameter                     | Default    | Description                                                 |
| ----------------------------- | ---------- | ----------------------------------------------------------- |
| `model`                       | (required) | HuggingFace model ID or local path                          |
| `gpus`                        | 1          | Number of GPUs (auto-derived from `--trainer-gpus` in split mode) |
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
| `"zmq"`        | IPC via ZeroMQ sockets. Lower latency for co-located processes |

#### `max_async_level`

Controls how many weight broadcasts can be in-flight simultaneously. Default: `1`. Higher values allow the inference engine to lag behind the trainer by more steps.

## See also

- [QLoRA](qlora.md)
- [Multi-GPU Training](multi-gpu.md)
- [Precision & Recipes](precision-and-recipes.md)
- [Config reference](../reference/config.md)
- [Back to docs index](../index.mdx)
