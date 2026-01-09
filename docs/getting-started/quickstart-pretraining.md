# Quickstart: Pretraining (PT)

This runs a pretraining job using a YAML config.

## 1) Pick an example config

Example configs are in `examples/pt/`.

Start with:
- `examples/pt/qwen3.yaml`

## 2) Run

```bash
surogate pt examples/pt/qwen3.yaml
```

Or, via `uv`:

```bash
uv run surogate pt examples/pt/qwen3.yaml
```

## 3) Outputs

Outputs are written under the config’s `output_dir`.

## See also

- [Quickstart: SFT](quickstart-sft.md)
- [Configuration](../guides/configuration.md)
- [Back to docs index](../index.md)
