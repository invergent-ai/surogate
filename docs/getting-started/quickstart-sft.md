# Quickstart: Supervised Fine-Tuning (SFT)

This runs a small LoRA SFT example using a YAML config.

## 1) Pick an example config

Example configs are in `examples/sft/`.

A reasonable default starting point:
- `examples/sft/qwen3-lora-bf16.yaml`

## 2) Run

```bash
surogate sft examples/sft/qwen3-lora-bf16.yaml
```

If you use `uv` and want to guarantee you’re running inside the project environment:

```bash
uv run surogate sft examples/sft/qwen3-lora-bf16.yaml
```

## 3) Outputs

Your run outputs (checkpoints, logs, artifacts) are written under the config’s `output_dir`.

## Notes

- For private Hugging Face models/datasets, pass `--hub_token`.

## See also

- [Quickstart: Pretraining](quickstart-pretraining.md)
- [Configuration](../guides/configuration.md)
- [Back to docs index](../index.md)
