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

### `merge`

Merge a LoRA checkpoint into the base model, producing a ready-to-serve model directory.

```bash
surogate merge \
    --base-model Qwen/Qwen3.5-0.8B \
    --checkpoint-dir ./output_q35/step_00000002 \
    --output ./merged_q35
```

Options:

- `--base-model <path>`: required, path to base model directory or HuggingFace model ID
- `--checkpoint-dir <path>`: required, path to a LoRA checkpoint directory (e.g. `output/step_00000050`)
- `--output <path>`: required, output directory for the merged model

## Notes

- The top-level CLI prints system diagnostics at startup (GPU, CUDA, etc.).

---

## See also

- [Config reference](config.md)
- [Back to docs index](../index.mdx)
