# CLI reference

Surogate exposes a small CLI with subcommands for common workflows.

## Synopsis

```bash
surogate <command> config.yaml [--hub_token <token>]
```

If the YAML config file is missing, the CLI prints help and exits with a non-zero status.

## Commands

### `sft`

Supervised fine-tuning.

```bash
surogate sft examples/sft/qwen3-lora-bf16.yaml
```

Options:

- `--hub_token <token>`: optional, Hugging Face token for private model access

### `pt`

Pretraining.

```bash
surogate pt examples/pt/qwen3.yaml
```

Options:

- `--hub_token <token>`: optional, Hugging Face token for private model access

### `tokenize`

Tokenize datasets for training.

```bash
surogate tokenize <path/to/config.yaml>
```

Options:

- `--debug`: print tokens with labels to confirm masking/ignores
- `--hub_token <token>`: optional, Hugging Face token for private model access

## Notes

- The top-level CLI prints system diagnostics at startup (GPU, CUDA, etc.).

---

## See also

- [Config reference](config.md)
- [Back to docs index](../index.md)
