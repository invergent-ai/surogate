# Configuration

Surogate is driven by a YAML config file.

## Start from an example

- SFT examples live in `examples/sft/`
- Pretraining examples live in `examples/pt/`
- Knowledge-distillation examples live in `examples/distillation/`

Run via CLI:

```bash
surogate sft path/to/config.yaml
# or
surogate pt path/to/config.yaml
```

To distill a larger teacher into the student during SFT, add a `distillation:` block and run `surogate distill-capture` before `surogate sft` — see [Knowledge Distillation](distillation.md).

## What to edit first

For most runs you’ll edit:
- `model`
- `output_dir`
- `datasets`
- `per_device_train_batch_size`, `gradient_accumulation_steps`, `sequence_len`
- `learning_rate`
- (optional) `lora`, `lora_rank` / QLoRA options

## See also

- [Config reference](../reference/config.md)
- [Datasets](datasets.md)
- [Knowledge Distillation](distillation.md)
- [Back to docs index](../index.mdx)
