# Configuration

Surogate is driven by a YAML config file.

## Start from an example

- SFT examples live in `examples/sft/`
- Pretraining examples live in `examples/pt/`

Run via CLI:

```bash
surogate sft path/to/config.yaml
# or
surogate pt path/to/config.yaml
```

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
- [Back to docs index](../index.md)
