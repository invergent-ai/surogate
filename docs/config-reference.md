# Configuration Reference

This section provides a comprehensive reference for all configuration options available in Surogate. Each option is described in detail, including its purpose, default value, and possible values.

## General Settings

| Option                     | Type   | Default        | Description                                                                          |
| -------------------------- | ------ | -------------- | ------------------------------------------------------------------------------------ |
| `run_name`                 | string | auto-generated | A descriptor for the run. If not provided, a unique name is generated automatically. |
| `apply_recommended_values` | bool   | `false`        | Whether to apply recommended configuration values.                                   |
| `num_epochs`               | int    | `3`            | Total number of training epochs to perform.                                          |
| `output_dir`               | string | `"output"`     | The output directory where the model predictions and checkpoints will be written.    |
| `checkpoint_dir`           | string | `"output"`     | Directory to save checkpoints during training. If None, defaults to `output_dir`.    |
| `resume_from_checkpoint`   | bool   | `false`        | Continue from checkpoint. If enabled, uses the latest checkpoint.                    |
| `save_steps`               | int    | `50`           | Number of steps between saving checkpoints.                                          |
| `save_total_limit`         | int    | `5`            | Limit the total amount of checkpoints. Deletes older checkpoints in `output_dir`.    |

## Model Settings

| Option          | Type   | Default     | Description                                                                                                                                                                                                                                                                               |
| --------------- | ------ | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`         | string | required    | Path or HuggingFace model identifier (e.g., `"Qwen/Qwen3-0.6B"`).                                                                                                                                                                                                                         |
| `model_type`    | string | auto-detect | Type of the model group. Automatically detected from model config if not specified.                                                                                                                                                                                                       |
| `sequence_len`  | int    | model's max | Maximum sequence length for training. Defaults to model's `max_model_len`.                                                                                                                                                                                                                |
| `max_model_len` | int    | auto-detect | Maximum model length for rope scaling. Automatically detected from model config if not specified.                                                                                                                                                                                         |
| `rope_scaling`  | string | `null`      | Type of RoPE scaling. Pass a string like `"linear"`, `"dynamic"`, or `"yarn"` along with `max_model_len` to automatically configure rope_scaling. Alternatively, pass a JSON string like `'{"factor": 2.0, "type": "yarn"}'` to directly override the rope_scaling in the model's config. |
| `torch_dtype`   | string | auto-detect | PyTorch data type for model weights. Options: `"bfloat16"`, `"float16"`, `"float32"`. Automatically detected from model config if not specified.                                                                                                                                          |

## Recomputation Options

Recomputation options trade compute for memory by recomputing activations during the backward pass instead of storing them.

| Option              | Type | Default | Description                                                                                                                                                                                                                            |
| ------------------- | ---- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `recompute_swiglu`  | bool | `true`  | Recompute SwiGLU activation during backward pass. As SwiGLU is at the widest part of the model, this results in substantial memory savings at moderate compute cost.                                                                   |
| `recompute_rmsnorm` | bool | `true`  | Recompute RMSNorm activations during backward pass to save memory.                                                                                                                                                                     |
| `recompute_ffn`     | bool | `true`  | Recompute Feed-Forward Network (FFN) activations during backward pass. Implies `recompute_swiglu`.                                                                                                                                     |
| `recompute_qkv`     | bool | `true`  | Recompute QKV projections during backward pass to save memory.                                                                                                                                                                         |
| `recompute_att`     | bool | `true`  | Recompute attention block during backward pass. Implies `recompute_qkv`.                                                                                                                                                               |
| `recompute_block`   | bool | `true`  | Recompute entire Transformer block during backward pass to save memory.                                                                                                                                                                |
| `recompute_lora`    | bool | `true`  | Recompute ln1/ln2 activations during LoRA backward pass instead of storing per-layer. Only effective when LoRA is enabled. Requires and sets `recompute_block` to `true`. When used with `offload_residual`, CUDA graphs are disabled. |

### Recomputation Hierarchy

The recomputation options form a hierarchy:

- `recompute_block` → implies `recompute_att`, `recompute_ffn`, `recompute_rmsnorm`
- `recompute_att` → implies `recompute_qkv`
- `recompute_ffn` → implies `recompute_swiglu`

## Offloading Options

Offloading options move tensors to host (CPU) memory to reduce GPU memory usage at the cost of increased data transfer overhead.

| Option               | Type | Default | Description                                                                                                                                                                                                                   |
| -------------------- | ---- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `offload_residual`   | bool | `false` | Offload residuals (of the FFN block) to pinned host memory. Combined with `recompute_block`, total activation memory becomes independent of network depth.                                                                    |
| `offload_master`     | bool | `false` | Store master weights in pinned host memory.                                                                                                                                                                                   |
| `offload_quants`     | bool | `false` | Store quantized weights in pinned host memory. Requires `persistent_quants`.                                                                                                                                                  |
| `offload_optimizer`  | bool | `false` | Store optimizer state in pinned host memory. Slows down optimizer step drastically, but with enough gradient accumulation steps, the overall contribution becomes negligible.                                                 |
| `offload_grads`      | bool | `false` | Offload gradients to pinned host memory.                                                                                                                                                                                      |
| `persistent_quants`  | bool | `false` | Avoid re-quantization of weights. Increases memory, but when combined with `offload_quants`, the additional memory is placed on the host. In PCIe settings, this can lead to significant speed-ups. Requires `shard_weights`. |
| `use_zero_copy`      | bool | `false` | Use ZeroCopy memory access instead of double-buffered cudaMemcpy for offloaded optimizer states. DMA is slower on consumer cards but faster on professional cards.                                                            |
| `use_write_combined` | bool | `false` | Use write-combined memory for offloaded tensors. May improve PCIe throughput in some situations.                                                                                                                              |

## Distributed Training (ZeRO) Options

| Option                  | Type | Default | Description                                                                                                                                                                     |
| ----------------------- | ---- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `zero_level`            | int  | `1`     | ZeRO redundancy optimization level: `1` = sharded optimizer states (default), `2` = sharded gradients + optimizer states, `3` = sharded weights + gradients + optimizer states. |
| `shard_weights`         | bool | `false` | Shard model weights across data-parallel processes. Enables more effective offloading and reduces memory consumption.                                                           |
| `shard_gradients`       | bool | `false` | Shard gradients across data-parallel processes. Enables more effective offloading and reduces memory consumption.                                                               |
| `use_all_to_all_reduce` | bool | `false` | Use all-to-all-based reduce algorithm (combine with `memcpy_send_recv`).                                                                                                        |
| `memcpy_all_gather`     | bool | `false` | Use memcpy for all-gather operations (threads backend only). Generally gets better bandwidth utilization on PCIe and does not consume SM resources.                             |
| `memcpy_send_recv`      | bool | `false` | Use memcpy for send/receive operations (threads backend only).                                                                                                                  |

## Hardware Settings

| Option            | Type | Default | Description                                                                                                                       |
| ----------------- | ---- | ------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `gpus`            | int  | `1`     | Number of GPUs to use for training. Use `0` for all available GPUs.                                                               |
| `use_cuda_graphs` | bool | `true`  | Enable CUDA graphs for performance. Automatically disabled for QLoRA and when `recompute_lora` conflicts with `offload_residual`. |

## Mixed Precision & Recipe Options

| Option           | Type   | Default  | Description                                                                                                                   |
| ---------------- | ------ | -------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `recipe`         | string | `"bf16"` | Mixed precision training recipe. Options: `"bf16"` (default), `"fp8_hybrid"`, `"nvfp4"`.                                      |
| `gradient_dtype` | string | `null`   | Dtype for activation gradients / backward matmul policy. Defaults to matmul-dtype. Note: recipes may override backward dtype. |
| `master_dtype`   | string | `null`   | Master weight dtype for optimizer updates (e.g., FP32 for stable full fine-tuning). Defaults to model-dtype.                  |
| `use_fused_rope` | bool   | `false`  | Use fused RoPE kernel with on-the-fly cos/sin computation (saves memory, reduces bandwidth).                                  |

### FP8 Recipe Options

| Option             | Type | Default | Description                                                        |
| ------------------ | ---- | ------- | ------------------------------------------------------------------ |
| `fp8_amax_history` | int  | `1024`  | FP8 delayed scaling amax history length (for `fp8_hybrid` recipe). |

### FP4/NVFP4 Recipe Options

| Option        | Type   | Default     | Description                                                                  |
| ------------- | ------ | ----------- | ---------------------------------------------------------------------------- |
| `fp4_backend` | string | `"cutlass"` | FP4 matmul backend: `"cutlass"` (default) or `"cudnn"` (for `nvfp4` recipe). |

### Layer Quantization Skip Options

| Option                    | Type | Default | Description                                                                           |
| ------------------------- | ---- | ------- | ------------------------------------------------------------------------------------- |
| `skip_quant_first_layers` | int  | `0`     | Skip quantization for the first N transformer layers (embedding layers kept in BF16). |
| `skip_quant_last_layers`  | int  | `0`     | Skip quantization for the last N transformer layers (lm_head layers kept in BF16).    |

## Optimizer Settings

| Option              | Type   | Default        | Description                                                                                      |
| ------------------- | ------ | -------------- | ------------------------------------------------------------------------------------------------ |
| `optimizer`         | string | `"adamw_8bit"` | Optimizer type. Options: `"adamw_8bit"` (8-bit AdamW), `"normuon"` (NorMuon hybrid)              |
| `learning_rate`     | float  | `2e-4`         | The initial learning rate for the optimizer.                                                     |
| `lr_scheduler_type` | string | `"linear"`     | Learning rate schedule function: `"linear"`, `"cosine"`, or `"wsd"`.                             |
| `warmup_ratio`      | float  | `0.0`          | Ratio of total training steps used for linear warmup from 0 to `learning_rate`.                  |
| `warmup_steps`      | int    | `0`            | Number of steps for linear warmup. Overrides `warmup_ratio` if set.                              |
| `cooldown_steps`    | int    | `0`            | Number of steps for linear cooldown from `learning_rate` to `final_lr_fraction * learning_rate`. |
| `final_lr_fraction` | float  | `0.0`          | Final learning rate as a fraction of the initial learning rate.                                  |
| `weight_decay`      | float  | `0.1`          | Weight decay applied to all layers except bias and LayerNorm weights.                            |
| `max_grad_norm`     | float  | `1.0`          | Maximum gradient norm for gradient clipping. `0.0` disables clipping.                            |

### AdamW 8-bit Optimizer Parameters

Used when `optimizer: "adamw_8bit"` (default).

| Option          | Type  | Default | Description                                |
| --------------- | ----- | ------- | ------------------------------------------ |
| `adamw_beta1`   | float | `0.9`   | The beta1 parameter for AdamW optimizer.   |
| `adamw_beta2`   | float | `0.999` | The beta2 parameter for AdamW optimizer.   |
| `adamw_epsilon` | float | `1e-8`  | The epsilon parameter for AdamW optimizer. |

### NorMuon Optimizer Parameters

Used when `optimizer: "normuon"`. NorMuon uses a hybrid approach: AdamW for embeddings/norms/lm_head, and orthogonalized momentum for 2D weight matrices.

| Option                | Type  | Default | Description                                                                            |
| --------------------- | ----- | ------- | -------------------------------------------------------------------------------------- |
| `normuon_momentum`    | float | `0.95`  | Momentum coefficient for orthogonalized momentum updates in 2D weight matrices.        |
| `normuon_beta2`       | float | `0.95`  | Second moment coefficient for variance tracking in NorMuon optimizer.                  |
| `normuon_cautious_wd` | bool  | `true`  | Enable cautious weight decay that only applies decay when gradient and momentum align. |

## Training Loop Settings

| Option                        | Type | Default | Description                                                                                                                                              |
| ----------------------------- | ---- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `per_device_train_batch_size` | int  | `2`     | Batch size per device during training/evaluation.                                                                                                        |
| `gradient_accumulation_steps` | int  | `4`     | Number of update steps to accumulate gradients before performing backward/update pass. Effective batch size = batch_size × grad_accumulation × num_gpus. |
| `max_steps`                   | int  | `-1`    | Total number of training steps. `-1` derives from epochs and dataset size.                                                                               |
| `eval_steps`                  | int  | `100`   | Run evaluation every N optimizer steps.                                                                                                                  |

## Dataset Settings

| Option                   | Type  | Default | Description                                                                                                                                                                                                                       |
| ------------------------ | ----- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `datasets`               | list  | `null`  | List of datasets for training. Each dataset should specify `path`, `type`, and other dataset-specific options. See [Dataset Configuration Options](#dataset-configuration-options) below.                                         |
| `validation_datasets`    | list  | `null`  | List of datasets for validation during training. If not provided, uses `validation_split_ratio` to create validation split from training data. Uses same format as `datasets`.                                                    |
| `validation_split_ratio` | float | `0.1`   | Ratio of training data to use for validation if no `validation_datasets` are provided. Value between 0.0 and 1.0.                                                                                                                 |
| `train_seed`             | int   | `1234`  | Random seed for the training dataloader. Controls shuffling and sampling order.                                                                                                                                                   |
| `eval_seed`              | int   | `1234`  | Random seed for the evaluation dataloader. Controls shuffling and sampling order.                                                                                                                                                 |
| `dataloader_num_workers` | int   | auto    | Number of subprocesses to use for data loading. `0` means data will be loaded in the main process. Defaults to optimal value based on CPU count.                                                                                  |
| `sample_packing`         | bool  | `true`  | Whether to enable sample packing to fit multiple data samples into a single sequence. Packing reduces the number of samples in the dataset; adjust gradient accumulation steps and learning rate accordingly for packed datasets. |

### Dataset Configuration Options

Each dataset in the `datasets` or `validation_datasets` list is configured with the following options. Dataset type determines which additional fields are required.

#### Base Dataset Options (All Types)

| Option    | Type   | Default   | Description                                                                                                          |
| --------- | ------ | --------- | -------------------------------------------------------------------------------------------------------------------- |
| `path`    | string | required  | HuggingFace dataset repo, s3:// URL, gs:// URL, or path to local file or directory.                                  |
| `type`    | string | required  | Dataset type. Options: `"text"`, `"instruction"`, `"conversation"`, `"auto"` (auto-detect format).                   |
| `subset`  | string | `null`    | HuggingFace dataset subset/configuration name to load (e.g., `"default"` for datasets with multiple configurations). |
| `split`   | string | `"train"` | Dataset split to load. Common values: `"train"`, `"test"`, `"validation"`.                                           |
| `samples` | int    | `null`    | Limit the number of samples to use from this dataset. If not specified, uses all available samples.                  |

#### Text Dataset Options (`type: "text"`)

For pre-training or continued pre-training on raw text data.

| Option       | Type   | Default  | Description                                                           |
| ------------ | ------ | -------- | --------------------------------------------------------------------- |
| `text_field` | string | `"text"` | Name of the column in the dataset that contains the raw text content. |

**Example:**
```yaml
datasets:
  - path: "HuggingFaceFW/fineweb-edu"
    type: text
    text_field: text
    split: train
    samples: 100000
```

#### Instruction Dataset Options (`type: "instruction"`)

For instruction-following datasets with system/instruction/input/output format.

| Option                   | Type   | Default  | Description                                                                                                       |
| ------------------------ | ------ | -------- | ----------------------------------------------------------------------------------------------------------------- |
| `instruction_field`      | string | required | Name of the column containing the instruction/question.                                                           |
| `output_field`           | string | required | Name of the column containing the expected output/answer.                                                         |
| `input_field`            | string | `null`   | Name of the column containing additional input context (optional).                                                |
| `system_prompt_type`     | string | `null`   | How to provide system prompt. Options: `"field"` (from dataset column), `"fixed"` (same for all samples), `null`. |
| `system_prompt_field`    | string | `null`   | Name of the column containing system prompts (required when `system_prompt_type: "field"`).                       |
| `system_prompt`          | string | `null`   | Fixed system prompt text to use for all samples (required when `system_prompt_type: "fixed"`).                    |
| `prompt_format`          | string | `null`   | Custom prompt format template. Use `{system}`, `{instruction}`, `{input}`, `{output}` as placeholders.            |
| `prompt_format_no_input` | string | `null`   | Custom prompt format when no input field. Use `{system}`, `{instruction}`, `{output}` as placeholders.            |

**Example:**
```yaml
datasets:
  - path: "yahma/alpaca-cleaned"
    type: instruction
    instruction_field: instruction
    input_field: input
    output_field: output
    system_prompt_type: fixed
    system_prompt: "You are a helpful AI assistant."
```

#### Conversation Dataset Options (`type: "conversation"`)

For multi-turn conversational datasets in chat format.

| Option                      | Type   | Default                                       | Description                                                                      |
| --------------------------- | ------ | --------------------------------------------- | -------------------------------------------------------------------------------- |
| `messages_field`            | string | `"messages"`                                  | Name of the column containing the list of conversation messages.                 |
| `system_field`              | string | `null`                                        | Name of the column containing the system prompt for the conversation (optional). |
| `tools_field`               | string | `null`                                        | Name of the column containing tool/function definitions for function calling.    |
| `message_property_mappings` | dict   | `{"role": "role", "content": "content", ...}` | Mapping of message property names if dataset uses non-standard field names.      |

**Example:**
```yaml
datasets:
  - path: "HuggingFaceH4/ultrachat_200k"
    type: conversation
    messages_field: messages
    split: train_sft
```

## Memory Optimization Settings

| Option                     | Type | Default | Description                                                                                               |
| -------------------------- | ---- | ------- | --------------------------------------------------------------------------------------------------------- |
| `lmhead_chunks`            | int  | `1`     | Split LM-head computation into N chunks to reduce logit tensor size by factor of N.                       |
| `attn_bwd_chunks`          | int  | `1`     | Split attention backward pass into N chunks to save workspace memory.                                     |
| `init_projections_to_zero` | bool | `false` | Initialize projection weights (FFN down and attention out) to zero. Only used when training from scratch. |

## LoRA Settings

| Option                | Type   | Default          | Description                                                        |
| --------------------- | ------ | ---------------- | ------------------------------------------------------------------ |
| `lora`                | bool   | `true`           | Whether to use LoRA adapters for training.                         |
| `lora_rank`           | int    | `16`             | Rank for LoRA adapters.                                            |
| `lora_alpha`          | int    | `32`             | Alpha value for LoRA adapters.                                     |
| `lora_dropout`        | float  | `0.05`           | Dropout rate for LoRA adapters.                                    |
| `lora_dtype`          | string | `"fp32"`         | Data type for LoRA adapters: `"bf16"` or `"fp32"`.                 |
| `lora_target_modules` | list   | `["all-linear"]` | List of module names to apply LoRA adapters to.                    |
| `merge_adapter`       | bool   | `false`          | Whether to merge LoRA adapters into the base model after training. |

## QLoRA Settings

| Option                   | Type | Default | Description                                                                                                                                                      |
| ------------------------ | ---- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `qlora_fp4`              | bool | `false` | Enable NVFP4 QLoRA mode (base weights quantized to FP4 E2M1). **Requires Blackwell GPU (SM100+)**.                                                               |
| `qlora_fp8`              | bool | `false` | Enable FP8 QLoRA mode (base weights quantized to FP8 with per-block scales).                                                                                     |
| `qlora_bnb`              | bool | `false` | Enable BitsAndBytes NF4 QLoRA mode (base weights quantized to NF4 with per-block absmax). Works on any CUDA GPU.                                                 |
| `qlora_block_size`       | int  | `128`   | Block size for FP8 QLoRA quantization. Valid values: `64`, `128`, `256`.                                                                                         |
| `qlora_bnb_block_size`   | int  | `64`    | Block size for BnB NF4 QLoRA quantization. Valid values: `64`, `128`, `256`, `512`.                                                                              |
| `qlora_bnb_double_quant` | bool | `true`  | Enable double quantization for BnB (quantize absmax values to INT8 for extra memory savings).                                                                    |
| `qlora_four_over_six`    | bool | `true`  | Enable Four Over Six (4/6) adaptive block scaling for NVFP4 QLoRA quantization. Evaluates both max=4 and max=6 scaling per block and selects lower error option. |

## Chat Template Settings

Chat template settings control how conversations are formatted for training and inference.

| Option                   | Type   | Default     | Description                                                                                                                                                                                  |
| ------------------------ | ------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `use_chat_template`      | bool   | `true`      | Whether to use chat template for training.                                                                                                                                                   |
| `template`               | string | auto        | The chat template to use. Automatically detected from model if not specified. Available templates defined in CHAT_TEMPLATE_MAPPING.                                                          |
| `system`                 | string | `null`      | Override the default system prompt in the template. Use `\n` for newlines.                                                                                                                   |
| `max_length`             | int    | `null`      | Maximum length for tokenized conversations. Defaults to `sequence_len` if not specified.                                                                                                     |
| `truncation_strategy`    | string | `"delete"`  | How to handle conversations exceeding max_length. Options: `"delete"` (skip sample), `"left"` (truncate from start), `"right"` (truncate from end), `"split"` (split into multiple samples). |
| `padding_side`           | string | `"right"`   | Which side to pad sequences on. Options: `"left"`, `"right"`.                                                                                                                                |
| `padding_free`           | bool   | `false`     | Enable padding-free training for more efficient packing.                                                                                                                                     |
| `loss_scale`             | string | `"default"` | Loss scaling strategy. Options: `"default"`, or custom scaling configuration.                                                                                                                |
| `sequence_parallel_size` | int    | `1`         | Sequence parallelism size for distributed training across sequence dimension.                                                                                                                |
| `response_prefix`        | string | `null`      | Prefix to add before model responses during inference. Use `\n` for newlines.                                                                                                                |
| `max_pixels`             | int    | `null`      | Maximum number of pixels for vision models (multimodal only).                                                                                                                                |
| `norm_bbox`              | string | `null`      | Bounding box normalization strategy for vision models. Options: `"norm1000"`, `"none"`, `null`.                                                                                              |
| `agent_template`         | string | `null`      | Template for agent-style conversations (advanced usage).                                                                                                                                     |

## Logging & Reporting

| Option         | Type   | Default        | Description                                                                                 |
| -------------- | ------ |f -------------- | ------------------------------------------------------------------------------------------- |
| `report_to`    | list   | `null`         | Report results and logs to specified platforms. Options: `"wandb"`, `"aim"`.                |
| `log_file`     | string | auto-generated | Where to save the training log. Defaults to `{output_dir}/log-{run_name}-{timestamp}.json`. |
| `log_gpu_util` | int    | `100`          | Interval for logging GPU utilization.                                                       |

### WandB (Weights & Biases) Settings

| Option          | Type   | Default      | Description                                                      |
| --------------- | ------ | ------------ | ---------------------------------------------------------------- |
| `wandb_project` | string | `"Surogate"` | WandB project name for logging.                                  |
| `wandb_name`    | string | `run_name`   | WandB run name for logging. Defaults to the value of `run_name`. |

### Aim Settings

| Option           | Type   | Default      | Description                                                     |
| ---------------- | ------ | ------------ | --------------------------------------------------------------- |
| `aim_experiment` | string | `"Surogate"` | Aim experiment name for logging.                                |
| `aim_repo`       | string | `null`       | Aim repository path for logging. Uses default if not specified. |
| `aim_name`       | string | `run_name`   | Aim run name for logging. Defaults to the value of `run_name`.  |

## Debugging Options

| Option                   | Type | Default | Description                                                                             |
| ------------------------ | ---- | ------- | --------------------------------------------------------------------------------------- |
| `debug_time_breakdown`   | bool | `false` | Enable detailed training timing breakdown for debugging.                                |
| `debug_memory_breakdown` | bool | `false` | Print detailed memory breakdown after model allocation (useful for QLoRA optimization). |

## Recipe Comparison

| Recipe       | Format                      | GPU Requirement                | Use Case                             |
| ------------ | --------------------------- | ------------------------------ | ------------------------------------ |
| `bf16`       | BF16 forward/backward       | Any CUDA GPU                   | Baseline, maximum compatibility      |
| `fp8_hybrid` | FP8 E4M3 fwd / E5M2 bwd     | SM89+ (Ada, Hopper, Blackwell) | 2x throughput, minimal accuracy loss |
| `nvfp4`      | FP4 E2M1 with block scaling | SM100+ (Blackwell only)        | Maximum memory efficiency            |

## Example Configuration

```yaml
# Model
model: Qwen/Qwen3-0.6B
model_type: qwen  # auto-detected if not specified
sequence_len: 2048
max_model_len: 2048
torch_dtype: bfloat16  # auto-detected if not specified

# Output
output_dir: ./output
save_steps: 100
save_total_limit: 3

# Training
num_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2e-4
lr_scheduler_type: cosine
warmup_ratio: 0.03

# Dataset
datasets:
  # Conversation dataset (most common for fine-tuning)
  - path: "mlabonne/FineTome-100k"
    type: conversation
    messages_field: conversations
    split: train
  # Or use instruction dataset format
  # - path: "yahma/alpaca-cleaned"
  #   type: instruction
  #   instruction_field: instruction
  #   input_field: input
  #   output_field: output
validation_split_ratio: 0.1
train_seed: 1234
eval_seed: 1234
sample_packing: true
dataloader_num_workers: 4

# Chat Template
use_chat_template: true
template: qwen  # auto-detected if not specified
truncation_strategy: delete
padding_side: right

# LoRA
lora: true
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_dtype: fp32

# Memory optimization
recompute_block: true
recipe: bf16

# Hardware
gpus: 1
use_cuda_graphs: true
```
