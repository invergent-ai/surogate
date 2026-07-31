# Quickstart: RL Training (GRPO)

This runs a GRPO reinforcement learning example. GRPO coordinates three components — an inference server (vLLM), an orchestrator (rollouts + rewards), and a Surogate trainer (policy gradient updates) — via a single command.

## 1) Pick example configs

Example configs are in `examples/grpo/`. GRPO uses three config files:

- **`train.yaml`** — Trainer settings (model, LoRA, precision, loss function)
- **`infer.yaml`** — vLLM inference server settings
- **`orch.yaml`** — Orchestrator settings (environment, batch size, sampling)

## 2) Run

GRPO can run in two single-command modes depending on your GPU layout:

### Split-GPU mode — `surogate grpo` (recommended for ≥2 GPUs)

vLLM and the trainer run on disjoint GPU sets, communicating via the filesystem. You explicitly assign GPU ids to each side:

```bash
surogate grpo --train examples/grpo/train.yaml --infer examples/grpo/infer.yaml --orch examples/grpo/orch.yaml \
    --vllm-gpus 0 --trainer-gpus 1
```

The trainer's GPU count is derived from `--trainer-gpus` automatically — the YAML `gpus` field becomes optional. For MoE models, `ep_size` is also auto-set to the trainer GPU count.

### Co-locate mode — `surogate grpo-colocate` (single-GPU or shared-GPU setups)

vLLM and the trainer share the same GPUs and exchange base weights via zero-copy CUDA IPC:

```bash
surogate grpo-colocate --train examples/grpo/train.yaml --infer examples/grpo/infer.yaml --orch examples/grpo/orch.yaml
```

No manual memory tuning needed — `gpu_memory_utilization` is computed automatically.

If you use `uv`, prefix any of the above with `uv run`.

## 3) Outputs

Outputs (checkpoints, LoRA adapters, logs) are written under the trainer's `output_dir`.

### Resuming an interrupted run

The trainer resumes from its latest checkpoint automatically —
`resume_from_checkpoint` defaults to `true`, and `checkpoint_dir` is where it
looks. Re-run the same command and it picks up where it stopped:

```
Resuming from checkpoint step 5 (next batch: 6)
```

A checkpoint at step `S` means "trained through batch `S`", so the resumed run
starts at `S + 1`. That one number drives everything downstream — which batch
is packed, which batch is read from the transport, the LR schedule position,
the save cadence, and the step number the broadcast weights are published
under — so the orchestrator, which may already be waiting on the broadcast for
step `S`, is unblocked rather than desynchronized.

Set `resume_from_checkpoint: false` in `train.yaml` to force a fresh run.

## 4) Example Configuration

A minimal setup using the **reverse-text** environment. With co-locate (`grpo-colocate`) this runs on a single GPU; with split (`grpo`) it runs on two GPUs (one for vLLM, one for the trainer):

**`train.yaml`**:

```yaml
model: Qwen/Qwen3-0.6B
output_dir: ./outputs
gpus: 1  # ignored in split mode — derived from --trainer-gpus

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

# Optional; omit to use vLLM's own defaults.
# max_num_seqs: 16      # concurrency cap — see "Serving memory" below
# kv_cache_dtype: fp8   # halves KV bytes/token
```

**`orch.yaml`**:

```yaml
model:
  name: Qwen/Qwen3-0.6B
  lora_adapter: default

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

### Serving Memory

Two `infer.yaml` keys decide whether a large policy plus LoRA fits:

- **`max_num_seqs`** caps concurrent sequences and so sizes the CUDA-graph and
  activation buffers. A 27B model served TP2 with `enable_lora: true` OOMs at
  the vLLM default and fits at `16`. Lowering `gpu_memory_utilization` does
  **not** help — it shrinks the budget those buffers draw from.
- **`kv_cache_dtype: fp8`** halves KV bytes per token, raising sustainable
  concurrency on a KV-bound server. It also perturbs sampled logprobs, which
  feed GRPO's importance ratio, so check `mismatch_kl` after enabling it.

Both default to unset, in which case vLLM's own defaults apply.

## 6) Advanced: Three-Process Mode

For multi-node setups (or any case where you want each component in its own process), run three commands separately:

```bash
# Terminal 1: Inference server
CUDA_VISIBLE_DEVICES=0 surogate grpo-infer infer.yaml

# Terminal 2: Orchestrator
surogate grpo-orch orch.yaml

# Terminal 3: Trainer
CUDA_VISIBLE_DEVICES=1 surogate grpo-train train.yaml
```

For single-host runs, prefer `surogate grpo` (split GPUs) or `surogate grpo-colocate` (shared GPUs) — they manage the lifecycle of all three components for you.

## Notes

- GRPO requires `vllm` to be installed for the inference server.
- **Reading progress from a log file.** The orchestrator's rollout progress bar
  is rendered by `tqdm`, which needs a TTY — under `nohup` or any redirect it
  writes nothing useful. Alongside it, the orchestrator emits plain `[progress]`
  lines carrying the same information, so a redirected log stays readable:

  ```
  [progress] step 11 Generating rollouts (train): 128/256 (50%) | 42.7/min | elapsed 3m00s | eta 3m00s
  ```

  A step that stops completing rollouts is reported explicitly instead of
  falling silent:

  ```
  [progress] step 11 Generating rollouts (train): 128/256 — NO completions in the last 60s (inflight 64, scoring 3, elapsed 14m22s)
  ```

  That line is the signal to check the inference server: a deep queue, an
  unresponsive engine, or rollouts erroring out all look identical from a
  frozen progress bar.
- The `model` field must match across all three config files.
- `max_steps` in `train.yaml` and `orch.yaml` should match.

## See also

- [RL Training guide](../guides/rl-training.md) — Full architecture details, config reference, and tuning tips
- [Quickstart: SFT](quickstart-sft.md)
- [Quickstart: Pretraining](quickstart-pretraining.md)
- [Configuration](../guides/configuration.md)
- [Back to docs index](../index.mdx)
