# Examples library

The [repository example index](../../examples/README.md) maps training and serving
features to maintained configs, launch scripts, and API clients. Run commands from
the repository root after installing Surogate.

| Workflow | Examples |
|---|---|
| SFT, LoRA, QLoRA, precision and model families | [Model recipes](../../examples/README.md#model-and-precision-recipes) |
| Full fine-tuning, adapters, distributed training, memory and monitoring | [Training features](../../examples/training/README.md) |
| Scratch and continued pretraining | [Pretraining](../../examples/pt/README.md) |
| Local data, column mapping, mixing and validation | [Datasets](../../examples/datasets/README.md) |
| Preference optimization | [DPO](../../examples/dpo/README.md) |
| Teacher capture and tokenizer transplantation | [Distillation](../../examples/distillation/README.md) |
| Split/colocated GRPO, evaluation and checkpoints | [GRPO](../../examples/grpo/README.md) |
| Judge rewards and multi-turn on-policy distillation | [RULER](../../examples/ruler/README.md), [TurnOPD](../../examples/turnopd/README.md) |
| HTTP APIs, media, adapters, placement, caching and speculation | [Serving](../../examples/serve/README.md) |

```bash
surogate sft examples/sft/qwen3/qwen3-lora-bf16.yaml
surogate dpo examples/dpo/qwen3.yaml
bash examples/serve/launch.sh chat
```

For every supported workflow and its prerequisites, use the
[feature map](../../examples/README.md#feature-map). Benchmark fixtures are maintained
separately from user-facing examples.
