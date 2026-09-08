# Single-GPU GRPO

Native co-locate mode runs GRPO rollouts and LoRA training on one GPU. The base
weights are loaded once and shared, including embeddings and normalization
weights. Generation pauses during each training update, and the next batch uses
the updated adapter. Per-step adapter updates stay in GPU memory.

The first supported configuration is a **dense Qwen3 BF16 safetensors model**
with LoRA. Choose a model and sequence length that fit your GPU together with
training activations and optimizer state. Serving releases its own GPU buffers
during training; training buffers remain reserved throughout the run.

Create `train.yaml`:

```yaml
model: Qwen/Qwen3-0.6B
output_dir: ./outputs/shared-grpo
gpus: 1
per_device_train_batch_size: 1
sequence_len: 2048
max_steps: 20
learning_rate: 1e-4
lr_scheduler_type: constant
recipe: bf16
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
```

Create `infer.yaml`:

```yaml
backend: surogate
model: Qwen/Qwen3-0.6B
enable_lora: true
max_model_len: 2048
max_num_seqs: 4
port: 8007
```

Create `orch.yaml`:

```yaml
model:
  name: Qwen/Qwen3-0.6B
  lora_adapter: default
  lora_rank: 16
  lora_alpha: 32
env:
  - id: markdown-table-qa
batch_size: 4
rollouts_per_example: 4
sequence_len: 2048
max_steps: 20
use_token_client: true
sampling:
  max_tokens: 1024
  temperature: 1.0
  top_p: 1.0
output_dir: ./outputs/shared-grpo/run_default
```

Run it on your chosen GPU:

```bash
CUDA_VISIBLE_DEVICES=0 surogate grpo-colocate \
    --train train.yaml --infer infer.yaml --orch orch.yaml
```

Use the same model in all three files, matching `max_steps` in training and
orchestration, and an orchestrator output directory directly inside the training
output directory. Start each run in a fresh directory. The runner connects the
orchestrator to the local server and sets synchronous generation automatically.

Reduce `max_num_seqs` to lower serving memory use. Reduce `sequence_len`
and `max_model_len` together to lower memory use. Set `sequence_len` in both the
training and orchestrator configs. Keep enough generation tokens for the model
to finish its answer and receive a useful reward.

The run writes `shared_weights.jsonl` in the training output directory. It reports
the shared base size, policy version, and serving buffer capacity at each phase. Base
upload bytes and serving base allocation bytes should both be zero. Normal final
adapter saving and training checkpoints still work; per-step broadcast folders
contain readiness markers only.

Quantized checkpoints, QLoRA, full fine-tuning, other model families, multiple
GPUs, CPU weight offload, QeRL weight noise, and checkpoint resume are not yet
supported by native co-locate mode. Use the existing split-GPU runner or
`backend: vllm` where appropriate.
