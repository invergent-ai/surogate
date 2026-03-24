# Quickstart: RL Training (GRPO)

Surogate supports two GRPO workflows:

- **Native trainer** (`surogate grpo-native`): one process, one config file, built-in generation.
- **vLLM pipeline** (`surogate grpo-infer` + `surogate grpo-orch` + `surogate grpo-train`): separate inference/orchestrator/trainer processes.

## 1) Native trainer quickstart (single config)

Native GRPO is the simplest way to start RL training:

```bash
surogate grpo-native examples/grpo-native/reverse-text.yaml
```

If you use `uv`:

```bash
uv run surogate grpo-native examples/grpo-native/reverse-text.yaml
```

Example native configs:

- `examples/grpo-native/reverse-text.yaml` (verifiers environment)
- `examples/grpo-native/qwen3-lora.yaml` (callback reward function)

Minimal native config shape:

```yaml
model: Qwen/Qwen3-0.6B
output_dir: ./output-grpo-native
gpus: 1
max_steps: 20
per_device_train_batch_size: 1
sequence_len: 2048

lora: true
lora_rank: 16
lora_alpha: 32

generation:
  num_completions: 16
  max_gen_len: 128
  max_prompts_per_batch: 2
  adaptive_batch_on_oom: true

sampling:
  temperature: 1.0

reward:
  mode: verifiers
  env:
    - id: reverse-text
```

## 2) vLLM pipeline quickstart (multi-process, three configs)

Example configs are in `examples/grpo/`. The pipeline uses:

- **`train.yaml`**: trainer settings (model, LoRA, precision, GRPO loss).
- **`infer.yaml`**: vLLM inference server settings.
- **`orch.yaml`**: orchestrator settings (environment, rollout batch size, sampling).

Run:

```bash
# Terminal 1: Inference server
CUDA_VISIBLE_DEVICES=0 surogate grpo-infer examples/grpo/infer.yaml

# Terminal 2: Orchestrator
surogate grpo-orch examples/grpo/orch.yaml

# Terminal 3: Trainer
CUDA_VISIBLE_DEVICES=1 surogate grpo-train examples/grpo/train.yaml
```

If you use `uv`, run the same commands with `uv run` prefix.

## 3) Outputs

Training artifacts are written under `output_dir`:

- Checkpoints
- Logs/metrics
- Final adapter (`final_adapter/`) when `lora: true`
- Final merged model (`final_model/`) when `lora: false`

## 4) Recommended hyperparameters

### Learning rate

RL training typically uses lower learning rates than SFT:

- Recommended range: `5e-7` to `5e-5` (start around `5e-6`)
- Schedule: `constant` or `cosine`
- Warmup: `0` is common; add warmup steps if training is unstable

### Batch sizing

- Native trainer: `problems_per_step` controls unique prompts per optimizer step; `generation.num_completions` controls rollouts per prompt.
- vLLM pipeline: `orch.batch_size` controls rollouts per step; `orch.rollouts_per_example` controls rollouts per prompt.
- `per_device_train_batch_size` is commonly `1` for packed RL sequences.

### GRPO loss

- vLLM pipeline commonly uses `ratio_type`, `token_mask_*`, and `geo_mask_*`.
- Native trainer uses `loss.kl_tau`, `loss.adv_tau`, `loss.teacher_tau`, `loss.ipo_mask_low`, `loss.ipo_mask_high`.

Start with low KL regularization and increase only if the policy drifts too quickly.

### Precision

All SFT precision options are available:

- FP8-Hybrid (`recipe: fp8-hybrid`)
- BF16 (`recipe: bf16`)
- QLoRA (`qlora_fp8`, `qlora_bnb`, `qlora_fp4`)

## 5) Multi-process vLLM mode

If you want separate processes (or separate GPUs/nodes), run:

```bash
# Terminal 1: Inference server
CUDA_VISIBLE_DEVICES=0 surogate grpo-infer infer.yaml

# Terminal 2: Orchestrator
surogate grpo-orch orch.yaml

# Terminal 3: Trainer
CUDA_VISIBLE_DEVICES=1 surogate grpo-train train.yaml
```

## Notes

- `vllm` is required for `surogate grpo-infer`.
- `vllm` is **not** required for `surogate grpo-native`.
- In the vLLM pipeline, `model` must match across `train.yaml`, `infer.yaml`, and `orch.yaml`.
- In the vLLM pipeline, keep `max_steps` aligned between trainer and orchestrator configs.

## See also

- [RL Training guide](../guides/rl-training.md)
- [Quickstart: SFT](quickstart-sft.md)
- [Quickstart: Pretraining](quickstart-pretraining.md)
- [Configuration](../guides/configuration.md)
- [Back to docs index](../index.mdx)
