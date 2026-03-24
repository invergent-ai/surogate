# CLI reference

Surogate exposes a CLI with subcommands for training, tokenization, RL workflows, and native OpenAI-compatible serving.

## Synopsis

```bash
surogate <command> [config.yaml] [--key value] [--nested.key=value] [--flag|--no-flag]
```

`config.yaml` is optional. If omitted, command defaults are used, and required fields must be provided via CLI overrides.

Unknown CLI flags are parsed as config overrides (for example, `--max-num-seqs=128` or `--max_num_seqs=128`).

## Commands

### `sft`

Supervised fine-tuning.

```bash
surogate sft examples/sft/qwen3-lora-bf16.yaml
```

Options:

- `--hub_token <token>`: optional, Hugging Face token for private model access
- `--<config_key>=<value>`: optional config override(s)

### `pt`

Pretraining.

```bash
surogate pt examples/pt/qwen3.yaml
```

Options:

- `--hub_token <token>`: optional, Hugging Face token for private model access
- `--<config_key>=<value>`: optional config override(s)

### `serve`

Native OpenAI-compatible inference serving.

```bash
surogate serve serve.yaml
```

Or fully from CLI overrides:

```bash
surogate serve --model Qwen/Qwen3-0.6B --port 8000 --api_key my-secret-key
```

Common options:

- `--model <hf_id_or_path>`: required if not set in config file
- `--host <addr>`: bind host (default `0.0.0.0`)
- `--port <int>`: server port (default `8000`)
- `--api_key <token>`: optional bearer token required for API access when set
- `--dtype <bf16|fp32>`: runtime dtype (default `bf16`)
- `--gpus <int>`: number of GPUs (default `1`)
- `--max_num_seqs <int>`: max concurrent sequences (default `64`)
- `--max_num_batched_tokens <int>`: scheduler token budget (default `2048`)
- `--max_model_len <int>`: max context length (default: inferred from model/tokenizer)
- `--gpu_memory_utilization <float>`: GPU memory budget fraction in `(0, 1]` (default `0.9`)
- `--max_gen_len <int>`: default max generated tokens (default `512`)
- `--temperature <float>`: default temperature (default `1.0`)
- `--top_k <int>`: default top-k cutoff (default `0`)
- `--top_p <float>`: default top-p (default `1.0`)
- `--min_p <float>`: default min-p (default `0.0`)
- `--repetition_penalty <float>`: default repetition penalty (default `1.0`)
- `--use_cuda_graphs <bool>`: enable CUDA graphs (default `true`)
- `--offload_experts <bool>`: enable MoE expert offloading when supported (default `false`)

Exposed endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`

If `api_key` is set, include `Authorization: Bearer <api_key>` on API requests.

### `tokenize`

Tokenize datasets for training.

```bash
surogate tokenize <path/to/config.yaml>
```

Options:

- `--debug`: print tokens with labels to confirm masking/ignores
- `--hub_token <token>`: optional, Hugging Face token for private model access
- `--<config_key>=<value>`: optional config override(s)

## Notes

- The top-level CLI prints system diagnostics at startup (GPU, CUDA, etc.).

---

## See also

- [Quickstart: Native Serving](../getting-started/quickstart-serving.md)
- [Config reference](config.md)
- [Back to docs index](../index.mdx)
