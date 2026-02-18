# Configuration Reference

This section provides a comprehensive reference for all configuration options available in Surogate. Each option is described in detail, including its purpose, default value, and possible values.

## General Settings

| Option                     | Type   | Default        | Description                                                                            |
| -------------------------- | ------ | -------------- | -------------------------------------------------------------------------------------- |
| `run_name`                 | string | auto-generated | A descriptor for the run. If not provided, a unique name is generated automatically.   |
| `apply_recommended_values` | bool   | `false`        | Whether to apply recommended configuration values.                                     |
| `num_epochs`               | int    | `3`            | Total number of training epochs to perform.                                            |
| `output_dir`               | string | `"output"`     | The output directory where the model predictions and checkpoints will be written.      |
| `checkpoint_dir`           | string | `null`         | Directory to save checkpoints during training. If None, defaults to `output_dir`.      |
| `resume_from_checkpoint`   | bool   | `true`         | Continue from checkpoint. If enabled, uses the latest checkpoint.                      |
| `save_steps`               | int    | `50`           | Number of steps between saving checkpoints.                                            |
| `save_total_limit`         | int    | `5`            | Limit the total amount of checkpoints. Deletes older checkpoints in `output_dir`.      |
| `from_scratch`             | bool   | `false`        | Train from scratch (random initialization) instead of fine-tuning a pre-trained model. |

## Model Settings

| Option          | Type   | Default     | Description                                                                                                                                                                                                                                                                               |
| --------------- | ------ | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`         | string | required    | Path or HuggingFace model identifier (e.g., `"Qwen/Qwen3-0.6B"`).                                                                                                                                                                                                                         |
| `model_type`    | string | auto-detect | Type of the model group. Automatically detected from model config if not specified.                                                                                                                                                                                                       |
| `sequence_len`  | int    | `1024`      | Maximum sequence length for training. Samples exceeding this length are truncated.                                                                                                                                                                                                        |
| `max_model_len` | int    | `null`      | Maximum model length for rope scaling. Automatically detected from model config if not specified.                                                                                                                                                                                         |
| `rope_scaling`  | string | `null`      | Type of RoPE scaling. Pass a string like `"linear"`, `"dynamic"`, or `"yarn"` along with `max_model_len` to automatically configure rope_scaling. Alternatively, pass a JSON string like `'{"factor": 2.0, "type": "yarn"}'` to directly override the rope_scaling in the model's config. |
| `torch_dtype`   | string | auto-detect | PyTorch data type for model weights. Options: `"bfloat16"`, `"float16"`, `"float32"`. Automatically detected from model config if not specified.                                                                                                                                          |

## Recomputation

Recomputation trades compute for memory by recomputing activations during the backward pass instead of storing them.

| Option      | Type | Default | Description                                                                                                                                                                   |
| ----------- | ---- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `recompute` | bool | `true`  | Enable activation recomputation. `false` saves all activations (fastest, most memory). `true` recomputes intermediates from checkpoints (saves VRAM, small compute overhead). |

## Offloading Options

Offloading options move tensors to host (CPU) memory to reduce GPU memory usage at the cost of increased data transfer overhead.

| Option               | Type | Default | Description                                                                                                                                                                                                                   |
| -------------------- | ---- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `offload_residual`   | bool | `false` | Offload residuals (of the FFN block) to pinned host memory. Combined with `recompute`, total activation memory becomes independent of network depth.                                                                          |
| `offload_master`     | bool | `false` | Store master weights in pinned host memory.                                                                                                                                                                                   |
| `offload_quants`     | bool | `false` | Store quantized weights in pinned host memory. Requires `persistent_quants`.                                                                                                                                                  |
| `offload_optimizer`  | bool | `false` | Store optimizer state in pinned host memory. Slows down optimizer step drastically, but with enough gradient accumulation steps, the overall contribution becomes negligible.                                                 |
| `offload_grads`      | bool | `false` | Offload gradients to pinned host memory. Requires `shard_gradients=true` or `zero_level >= 2`.                                                                                                                                |
| `persistent_quants`  | bool | `false` | Avoid re-quantization of weights. Increases memory, but when combined with `offload_quants`, the additional memory is placed on the host. In PCIe settings, this can lead to significant speed-ups. Requires `shard_weights`. |
| `use_zero_copy`      | bool | `false` | Use ZeroCopy memory access instead of double-buffered cudaMemcpy for offloaded optimizer states. DMA is slower on consumer cards but faster on professional cards.                                                            |
| `use_write_combined` | bool | `false` | Use write-combined memory for offloaded tensors. May improve PCIe throughput in some situations.                                                                                                                              |

## Multi-GPU Training (ZeRO) Options

These options apply to single-node multi-GPU training. For multi-node distributed training, see [Multi-Node Distributed Training](#multi-node-distributed-training).

| Option                  | Type | Default | Description                                                                                                                                                                     |
| ----------------------- | ---- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `zero_level`            | int  | `1`     | ZeRO redundancy optimization level: `1` = sharded optimizer states (default), `2` = sharded gradients + optimizer states, `3` = sharded weights + gradients + optimizer states. |
| `shard_weights`         | bool | `false` | Shard model weights across data-parallel processes. Enables more effective offloading and reduces memory consumption.                                                           |
| `shard_gradients`       | bool | `false` | Shard gradients across data-parallel processes. Enables more effective offloading and reduces memory consumption.                                                               |
| `use_all_to_all_reduce` | bool | `false` | Use all-to-all-based reduce algorithm (combine with `memcpy_send_recv`).                                                                                                        |
| `memcpy_all_gather`     | bool | `false` | Use memcpy for all-gather operations (threads backend only). Generally gets better bandwidth utilization on PCIe and does not consume SM resources.                             |
| `memcpy_send_recv`      | bool | `false` | Use memcpy for send/receive operations (threads backend only).                                                                                                                  |

## Multi-Node Distributed Training

Configuration for training across multiple machines using Ray and NCCL. See the [Multi-Node Training Guide](guides/multi-node.md) for detailed setup instructions.

| Option                          | Type   | Default  | Description                                                                                                                                                  |
| ------------------------------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `distributed.ray_address`       | string | `"auto"` | Ray cluster address. Options: `"auto"` (connect to existing cluster), `"local"` (start local instance), `"ray://host:port"` (connect to specific head).      |
| `distributed.num_nodes`         | int    | `1`      | Total number of nodes to use for training. Set to `> 1` to enable multi-node training.                                                                       |
| `distributed.gpus_per_node`     | int    | `0`      | Number of GPUs per node. If `0`, uses the value from `gpus` config parameter.                                                                                |
| `distributed.worker_output_dir` | string | `null`   | Base directory for worker-local tokenized data. Each worker creates a `node-{rank}/` subdirectory. If `null`, uses `/tmp/surogate-{run_name}/` on each node. |

**Example configuration:**
```yaml
distributed:
  ray_address: "auto"
  num_nodes: 2
  gpus_per_node: 8
  worker_output_dir: /shared/surogate-data
```

## Hardware Settings

| Option            | Type | Default | Description                                                         |
| ----------------- | ---- | ------- | ------------------------------------------------------------------- |
| `gpus`            | int  | `1`     | Number of GPUs to use for training. Use `0` for all available GPUs. |
| `use_cuda_graphs` | bool | `true`  | Enable CUDA graphs for performance.                                 |

## Mixed Precision & Recipe Options

| Option           | Type   | Default  | Description                                                                                                                   |
| ---------------- | ------ | -------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `recipe`         | string | `"bf16"` | Mixed precision training recipe. Options: `"bf16"` (default), `"fp8_hybrid"`, `"nvfp4"`, `"nvfp4_quartet"`.                   |
| `gradient_dtype` | string | `null`   | Dtype for activation gradients / backward matmul policy. Defaults to matmul-dtype. Note: recipes may override backward dtype. |
| `master_dtype`   | string | `null`   | Master weight dtype for optimizer updates (e.g., FP32 for stable full fine-tuning). Defaults to model-dtype.                  |
| `use_fused_rope` | bool   | `false`  | Use fused RoPE kernel with on-the-fly cos/sin computation (saves memory, reduces bandwidth).                                  |

### FP8 Recipe Options

| Option             | Type | Default | Description                                                        |
| ------------------ | ---- | ------- | ------------------------------------------------------------------ |
| `fp8_amax_history` | int  | `16`    | FP8 delayed scaling amax history length (for `fp8_hybrid` recipe). |

### FP4/NVFP4 Recipe Options

| Option        | Type   | Default     | Description                                                                  |
| ------------- | ------ | ----------- | ---------------------------------------------------------------------------- |
| `fp4_backend` | string | `"cutlass"` | FP4 matmul backend: `"cutlass"` (default) or `"cudnn"` (for `nvfp4` recipe). |

### Layer Quantization Skip Options

| Option                    | Type | Default | Description                                                                                               |
| ------------------------- | ---- | ------- | --------------------------------------------------------------------------------------------------------- |
| `skip_quant_first_layers` | int  | `0`     | Skip quantization for the first N transformer decoder layers. (embedding layers are always kept in BF16). |
| `skip_quant_last_layers`  | int  | `0`     | Skip quantization for the last N transformer decoder layers (lm_head layers are always kept in BF16).     |

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
| `max_grad_norm`     | float  | `0.0`          | Maximum gradient norm for gradient clipping. `0.0` disables clipping.                            |

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
| `multimodal_on_the_fly`       | bool | `null`  | If `true`, compute visual inputs on-the-fly during training instead of using pre-tokenized files. If `null` and the model is multimodal, defaults to `true`. |

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

| Option                | Type   | Default   | Description                                                                |
| --------------------- | ------ | --------- | -------------------------------------------------------------------------- |
| `lora`                | bool   | `true`    | Whether to use LoRA adapters for training.                                 |
| `lora_rank`           | int    | `16`      | Rank for LoRA adapters.                                                    |
| `lora_alpha`          | int    | `32`      | Alpha value for LoRA adapters.                                             |
| `lora_dropout`        | float  | `0.05`    | Dropout rate for LoRA adapters.                                            |
| `lora_dtype`          | string | `"fp32"`  | Data type for LoRA adapters: `"bf16"` or `"fp32"`.                         |
| `lora_target_modules` | list   | `["all"]` | List of module names to apply LoRA adapters to.                            |
| `train_router`        | bool   | `false`   | Train MoE router gate during LoRA fine-tuning. Only applies to MoE models. |
| `adapter_path`        | string | `null`    | Path to a PEFT adapter directory to merge into base weights before training. Requires `lora: true`. Not supported with pre-quantized models. |
| `merge_adapter`       | bool   | `false`   | Whether to merge LoRA adapters into the base model after training.         |

## MoE Settings

MoE (Mixture-of-Experts) settings control router loss coefficients for load balancing during training.

| Option                 | Type  | Default | Description                                                                             |
| ---------------------- | ----- | ------- | --------------------------------------------------------------------------------------- |
| `router_aux_loss_coef` | float | `null`  | MoE auxiliary (load balancing) loss coefficient. `null` uses model config default.      |
| `router_z_loss_coef`   | float | `null`  | MoE z-loss (router logit regularization) coefficient. `null` uses model config default. |

**Understanding MoE Losses:**

- **Auxiliary Loss (`aux_loss`)**: Encourages load balancing across experts. Higher values enforce more even token distribution but may reduce model capacity. Typical range: 0.001-0.1.
- **Z-Loss (`z_loss`)**: Regularizes router logits to prevent them from growing too large, which can cause routing collapse. Typical range: 0.0001-0.01.

## QLoRA Settings

| Option                           | Type | Default | Description                                                                                                                                                                                                                                |
| -------------------------------- | ---- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `qlora_fp4`                      | bool | `false` | Enable NVFP4 QLoRA mode (base weights quantized to FP4 E2M1). **Requires Blackwell GPU (SM100+)**.                                                                                                                                         |
| `qlora_fp8`                      | bool | `false` | Enable FP8 QLoRA mode (base weights quantized to FP8 with per-block scales).                                                                                                                                                               |
| `qlora_bnb`                      | bool | `false` | Enable BitsAndBytes NF4 QLoRA mode (base weights quantized to NF4 with per-block absmax). Works on any CUDA GPU.                                                                                                                           |
| `qlora_block_size`               | int  | `128`   | Block size for FP8 QLoRA quantization. Valid values: `64`, `128`, `256`.                                                                                                                                                                   |
| `qlora_bnb_block_size`           | int  | `64`    | Block size for BnB NF4 QLoRA quantization. Valid values: `64`, `128`, `256`, `512`.                                                                                                                                                        |
| `qlora_bnb_double_quant`         | bool | `true`  | Enable double quantization for BnB (quantize absmax values to INT8 for extra memory savings).                                                                                                                                              |
| `qlora_four_over_six`            | bool | `true`  | Enable Four Over Six (4/6) adaptive block scaling for NVFP4 QLoRA quantization. Evaluates both max=4 and max=6 scaling per block and selects lower error option.                                                                           |
| `qlora_selective_expert_dequant` | bool | `false` | Enable selective expert dequantization for MoE models to reduce dequant buffer memory. When enabled, it only dequantizes the experts that are actually selected by the router for each forward pass, rather than dequantizing all experts. |
| `qlora_offload_experts`          | bool | `false` | Offload expert weights in QLoRA MoE models to host memory. Works at the layer level (loads/unloads entire layer's experts as groups).                                                                                                                                                                                |

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

| Option         | Type   | Default | Description                                                                  |
| -------------- | ------ | ------- | ---------------------------------------------------------------------------- |
| `report_to`    | list   | `null`  | Report results and logs to specified platforms. Options: `"wandb"`, `"aim"`. |
| `log_file`     | string | `null`  | Where to save the training log. If `null`, no log file is written.           |
| `log_gpu_util` | int    | `100`   | Interval for logging GPU utilization.                                        |

### WandB (Weights & Biases) Settings

| Option          | Type   | Default    | Description                                                      |
| --------------- | ------ | ---------- | ---------------------------------------------------------------- |
| `wandb_project` | string | `null`     | WandB project name for logging.                                  |
| `wandb_name`    | string | `run_name` | WandB run name for logging. Defaults to the value of `run_name`. |

### Aim Settings

| Option           | Type   | Default    | Description                                                     |
| ---------------- | ------ | ---------- | --------------------------------------------------------------- |
| `aim_experiment` | string | `null`     | Aim experiment name for logging.                                |
| `aim_repo`       | string | `null`     | Aim repository path for logging. Uses default if not specified. |
| `aim_name`       | string | `run_name` | Aim run name for logging. Defaults to the value of `run_name`.  |

## Debugging Options

| Option                   | Type | Default | Description                                                                             |
| ------------------------ | ---- | ------- | --------------------------------------------------------------------------------------- |
| `debug_time_breakdown`   | bool | `false` | Enable detailed training timing breakdown for debugging.                                |
| `debug_memory_breakdown` | bool | `false` | Print detailed memory breakdown after model allocation (useful for QLoRA optimization). |

## Training Diagnostics & Automation

These options control automatic training monitoring, early stopping, and compute-optimal adjustments. All are disabled by default and safe to enable — they only add diagnostics or automation on top of the normal training loop.

| Option              | Type | Default | Description                                                                                                                                                                                                                                                          |
| ------------------- | ---- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `auto_lr_reduction` | bool | `false` | Detect loss spikes and gradient explosions, then permanently reduce the learning rate. Monitors a rolling window of loss/grad-norm values; when an anomaly is detected (loss > mean + 3σ, or grad_norm > 10× average), the LR schedule is scaled down by 50%. Up to 5 reductions. |
| `early_stop`        | bool | `false` | Multi-criteria early stopping. Stops training when ANY of: (1) convergence score > 0.85 for 5 consecutive evals, (2) compute efficiency (loss reduction per FLOP) drops below 50% of peak, (3) training diverges for 200+ consecutive steps, (4) loss plateaus for 500+ consecutive steps. Uses the `6N` approximation for FLOPs/token. |
| `epoch_adjustment`  | bool | `false` | Automatically adjust `num_epochs` to match the Chinchilla-optimal token budget (20× model parameters). If the dataset is smaller than the budget, increases epochs; if larger, decreases them. Only applies when `max_steps` is not explicitly set. |

**Always-on diagnostics** (no config flag required):

- **Plateau detection**: Warns when training loss stops improving over a rolling window. No automatic action taken.
- **Phase detection**: Classifies training into WARMUP / CONVERGING / PLATEAU / UNSTABLE / DIVERGING phases. Phase transitions are logged and the current phase is shown in the step log output.
- **Chinchilla token budget**: Printed at training start — shows the Chinchilla-optimal token count (20 × params) alongside planned tokens, so you can gauge training sufficiency at a glance.

## Recipe Comparison

| Recipe          | Format                      | GPU Requirement                | Use Case                             |
| --------------- | --------------------------- | ------------------------------ | ------------------------------------ |
| `bf16`          | BF16 forward/backward       | Any CUDA GPU                   | Baseline, maximum compatibility      |
| `fp8_hybrid`    | FP8 E4M3 fwd / E5M2 bwd     | SM89+ (Ada, Hopper, Blackwell) | 2x throughput, minimal accuracy loss |
| `nvfp4`         | FP4 E2M1 with block scaling | SM100+ (Blackwell only)        | Maximum memory efficiency            |
| `nvfp4_quartet` | FP4 E2M1 quartet scaling    | SM100+ (Blackwell only)        | Higher accuracy FP4 training         |

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
recompute: true
recipe: bf16

# Hardware
gpus: 1
use_cuda_graphs: true
```
