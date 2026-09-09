# Training features

Run from the repository root. These are complete configs, so run
`surogate sft examples/training/<file>.yaml`. They use Qwen3-0.6B and the same
instruction dataset as the basic [BF16 LoRA example](../sft/qwen3/qwen3-lora-bf16.yaml).
Replace the dataset and step budget for your task. Each config has its own output
directory and starts a fresh run by default.

| Config | Feature | Resources |
|---|---|---|
| [full-finetune.yaml](full-finetune.yaml) | Update all parameters; FP32 master weights/gradients, AdamW, cosine schedule | One GPU; more memory than LoRA |
| [multi-gpu.yaml](multi-gpu.yaml) | Four-GPU data parallelism, ZeRO-3, FP8 persistent quants in host RAM, offloaded gradients/optimizer, communication overlap | Four FP8-capable GPUs and host RAM |
| [cpu-offload.yaml](cpu-offload.yaml) | Full fine-tuning with CPU weights/gradients/optimizer and saved activation offload | One GPU plus host RAM; GPU computes forward/backward |
| [long-context.yaml](long-context.yaml) | 8K context, tiled MLP, recomputation, activation offload, chunked LM head and attention backward | One GPU and host RAM |
| [adaptive.yaml](adaptive.yaml) | Automatic LR reduction, early stopping, epoch/token-budget adjustment, WSD schedule and local metrics | One GPU; epoch-derived step budget |
| [continue-adapter.yaml](continue-adapter.yaml) | Import an existing trainable PEFT adapter | Run the basic BF16 LoRA recipe first |
| [runtime-lora.yaml](runtime-lora.yaml) | Attention-only adapter for native runtime serving | One GPU; then use [serving examples](../serve/README.md) |

For **LoRA / online QLoRA / prequantized bases**, use the
[model recipes](../README.md#model-and-precision-recipes). BF16 is the broadest
training precision; FP8 needs compatible hardware and NVFP4 compute needs Blackwell.
Quantized base storage and compute precision are separate settings: `qlora_bnb`,
`qlora_fp8`, and `qlora_fp4` choose online base quantization; `recipe` chooses compute.
Do not enable online QLoRA flags on an already quantized checkpoint.

## Multiple GPUs and large models

One Surogate process manages the GPUs; do not wrap the examples in `torchrun`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 surogate sft examples/training/multi-gpu.yaml
surogate sft examples/sft/qwen3/qwen3-lora-bf16-ray.yaml
surogate sft examples/sft/qwen35/qwen36-text-lora-fp8-pp.yaml
```

The Ray config needs a running cluster with two nodes and four GPUs per node. Use
the same checkout/environment and reachable data/model paths on every node; see
[multi-node setup](../../docs/guides/multi-node.md). Its `ray_address: auto` connects
to the existing cluster.

The dispatch pipeline config is a separate four-GPU LoRA workflow for a 27B model.
`offload_master` streams the frozen base from pinned host RAM. It cannot combine
with ZeRO sharding, expert parallelism, `cpu_training`, or residual offload. The
scheduler chooses stages automatically; see [dispatch-PP](../../docs/guides/dispatch-pp.md).

For **MoE expert parallelism and routing**, use the Qwen3.5/3.6 MoE or Laguna recipes.
They demonstrate `ep_size`, load balancing, BF16 expert adapters, and router losses.
`ep_size` must divide both GPU and expert counts. `train_router: true` optionally
updates the router during LoRA training; use the model's supported recipe and
monitor routing metrics. Expert offload and selective dequantization are demonstrated
by the Qwen3 MoE BnB example.

For **chunked sequences**, the [Laguna recipe](../sft/laguna/laguna-s-lora-fp8.yaml)
and [Qwen3.6 MoE recipe](../sft/qwen35moe/qwen36moe-lora-fp8.yaml) set
`sequence_chunks: 8`. The chunk count must divide `sequence_len`, batch size must
be one, and LoRA dropout must be zero. Tiled MLP (`long_context`) is a separate
memory-saving technique for dense models.

## Continue an adapter, stack an adapter, or resume a run

```bash
surogate sft examples/sft/qwen3/qwen3-lora-bf16.yaml
surogate sft examples/training/continue-adapter.yaml
```

The second command imports the first run's PEFT adapter and starts a fresh optimizer.
The adapter's rank, alpha and targets must match. To **stack a new adapter**, copy
`continue-adapter.yaml`, set `adapter_init_mode: merge`, and give it a new
`output_dir`. That merges the parent into an unquantized base and trains a new
adapter on top; prequantized bases cannot use this merge initialization.

To **resume optimizer state and step count**, set `resume_from_checkpoint: true`
in the original config and rerun it with the same output directory and training
budget. This discovers the latest `step_XXXXXXXX` checkpoint. `save_steps` controls
checkpoint frequency and `save_total_limit` controls retention. A completed run
already at its step budget has no remaining steps; raise the budget to extend it.

SFT exports its final adapter directly in `output_dir` (alongside
`adapter_config.json`), and full fine-tuning exports the model there.
`merge_adapter: true` also writes merged model weights into that directory.
The [reverse-text example](../sft/reverse-text-qwen3.yaml) demonstrates automatic
merging and split prompt/completion message columns. The
[serving guide](../serve/README.md#runtime-adapters-and-merged-models) shows an
explicit merge into a separate directory, followed by GGUF quantization.

## Data, validation, and monitoring

```bash
surogate tokenize examples/datasets/mixed.yaml --debug
surogate sft examples/datasets/mixed.yaml
surogate sft examples/training/adaptive.yaml
```

The [dataset example](../datasets/README.md) includes local instruction/chat data,
column mapping, dataset mixing and held-out validation. Standard model recipes use
packing and reserve the default 10% validation split. Use `loss_scale: all` for
raw-text next-token training; see [pretraining](../pt/README.md).

Adaptive training writes local metrics to `outputs/training/adaptive/metrics.jsonl`.
Its `max_steps: -1` lets the epoch and token-budget logic choose the step count.
To enable optional services, install/configure the corresponding integration and
add these fields to a copy of the config:

```yaml
report_to: [surogate, wandb, aim]
wandb_project: surogate-examples
wandb_name: qwen3-adaptive
aim_experiment: surogate-examples
aim_repo: ./outputs/aim
```

Authenticate W&B first and install `aim` if needed. The built-in JSON training log
and `training_plot.png` do not need either service. To view local metrics in the
live dashboard:

```bash
SUROGATE_METRICS_PATH=./outputs/training/adaptive/metrics.jsonl surogate jackalope
```

For timing/memory diagnostics, add `debug_time_breakdown: true` or
`debug_memory_breakdown: true` to a copy. For a short smoke run, copy any base
config and set `max_steps: 3`, `eval_steps: 0`, `save_steps: 0`, and a fresh
`output_dir`. To measure recomputation cost on a model that fits, compare
`recompute: true` and `false`; QLoRA forces recomputation on. Dedicated benchmark
and one-step YAML duplicates are kept out of the public example library.
