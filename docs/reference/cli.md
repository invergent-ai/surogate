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

### `distill-capture`

Capture teacher top-K logprobs for offline knowledge distillation. Writes a `.kd` sidecar next to each tokenized `train-*.bin` shard; a following `surogate sft` run with the same config trains against them. See the [Knowledge Distillation guide](../guides/distillation.md).

```bash
surogate distill-capture examples/distillation/qwen3-kd.yaml
```

Options:

- `--api-base <url>`: OpenAI-compatible base URL of a served teacher, e.g. `http://localhost:8000/v1` (overrides `distillation.teacher_api_base`). Requires a vLLM-compatible server started with `--max-logprobs >= distillation.top_k`
- `--device <device>`: device to run the local teacher model on (default `cuda:0`); ignored with a warning in API mode
- `--allow-cross-doc-attention`: allow the sdpa fallback when flash-attention-2 is unavailable; packed documents will attend across document boundaries during capture; ignored with a warning in API mode
- `--hub_token <token>`: optional, Hugging Face token for private model access

### `transplant-tokenizer`

Transplant a teacher's tokenizer onto a student model for cross-tokenizer distillation (wraps `mergekit-tokensurgeon` as an external subprocess; requires `pip install mergekit`). The output model uses the teacher's tokenizer, so the standard KD pipeline runs unchanged against it. See [Cross-tokenizer distillation](../guides/distillation.md#cross-tokenizer-distillation).

```bash
surogate transplant-tokenizer --student Qwen/Qwen3-8B --teacher deepseek-ai/DeepSeek-V3 \
    --output ./qwen3-8b-dsv3-vocab
```

Options:

- `--student <model>`: student model directory or HuggingFace ID (the model to be trained). Required unless `--restore` (where it is the KD-trained model to convert back)
- `--teacher <model>`: teacher model directory or HuggingFace ID (tokenizer donor). Required unless `--restore`
- `--output <dir>`: required, output directory for the transplanted model
- `--method <name>`: approximation method for new vocabulary rows (default `omp`, recommended)
- `--k <int>`: sparsity level / neighbor count for the approximation (default `64`)
- `--device <device>`: device for the approximation solve (e.g. `cuda`)
- `--trust-remote-code`: pass `--trust-remote-code` to `mergekit-tokensurgeon`
- `--restore <manifest>`: reverse mode — path to a `transplant_manifest.json`; transplants the distilled model (given via `--student`) back to the original student tokenizer

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
