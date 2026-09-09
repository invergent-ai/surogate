# Single-GPU GRPO

Native co-locate mode runs GRPO rollouts and LoRA training on one GPU. The base
weights are loaded once and shared, including embeddings and normalization
weights. Generation pauses during each training update, and the next batch uses
the updated adapter. Per-step adapter updates stay in GPU memory.

This mode supports the training families below, excluding **Nemotron**, using
**unquantized BF16 safetensors weights with LoRA**:

- Qwen3 and Qwen3.5, including their MoE variants and compatible Qwen3.6 checkpoints.
- Llama, MiniCPM5, and Spark-X2.5.
- Gemma 3 and Gemma 4, including E-series, unified, and MoE variants.
- LFM2/LFM2.5, LFM2-MoE, Qwen3-VL, and LFM2-VL.
- GPT-OSS and Laguna, when supplied as unquantized BF16 checkpoints.

Multimodal checkpoints use **text prompts only**. Experimental training definitions
such as DeepSeek-V4, Flash-Next, and GLM-5.3-Flash are not included.

Dense Qwen3 and Qwen3.5 use the optimized generation server. Other families generate
through the same model instance that performs training. This general path processes
requests one at a time and recomputes the prefix for every generated token, so long
prompts and completions are considerably slower. Start with short contexts and a
small rollout batch. It supports text completions, streaming, and token-based
multi-turn conversations; tool-call parsing and structured outputs are not available
on this path.

Choose a model and sequence length that fit your GPU together with training
activations and optimizer state. The entire base must fit on one GPU, including all
experts for MoE models. Training buffers remain reserved throughout the run.

Create `train.yaml`:

```yaml
model: Qwen/Qwen3.5-0.8B
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
model: Qwen/Qwen3.5-0.8B
enable_lora: true
max_model_len: 2048
max_num_seqs: 4
port: 8007
```

Create `orch.yaml`:

```yaml
model:
  name: Qwen/Qwen3.5-0.8B
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

To use another supported model, replace the model in all three files with its
checkpoint and select the LoRA targets from that model's training example.
`lora_target_modules: [all]` selects the targets supported by its training definition.
For MoE models, also set `lora_dtype: bf16`; expert LoRA training currently requires
BF16 adapters. Official quantized GPT-OSS and FP8 Laguna checkpoints must first be
exported as BF16; their quantized formats cannot be used in this mode.

Use the same model in all three files, matching `max_steps` in training and
orchestration, and an orchestrator output directory directly inside the training
output directory. Start each run in a fresh directory. The runner connects the
orchestrator to the local server and sets synchronous generation automatically.

Reduce `max_num_seqs` to lower optimized serving memory use. On the general path,
it limits admitted requests; generation still processes one request at a time.
Reduce `sequence_len` and `max_model_len` together to lower memory use. Set `sequence_len` in both the
training and orchestrator configs. Keep enough generation tokens for the model
to finish its answer and receive a useful reward.

The run writes `shared_weights.jsonl` in the training output directory. It reports
the shared base size and policy version at each phase. Base
upload bytes and serving base allocation bytes should both be zero. Normal final
adapter saving and training checkpoints still work; per-step broadcast folders
contain readiness markers only.

Quantized checkpoints, QLoRA, full fine-tuning, image/video prompts,
Nemotron, multiple GPUs, CPU weight offload, QeRL weight noise, and
checkpoint resume are not yet supported by native co-locate mode. Use the split-GPU runner for those.
