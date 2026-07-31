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

### `dpo`

Offline Direct Preference Optimization on `{prompt, chosen, rejected}` pairs.

```bash
surogate dpo path/to/config.yaml
```

Options:

- `--hub_token <token>`: optional, Hugging Face token for private model access

### `grpo`

GRPO RL training with vLLM and the trainer on **disjoint** GPU sets, communicating
over the filesystem.

```bash
surogate grpo --train examples/grpo/train.yaml --infer examples/grpo/infer.yaml --orch examples/grpo/orch.yaml \
    --vllm-gpus 0,1 --trainer-gpus 2,3
```

Options:

- `--train <path>`: required, trainer config YAML
- `--infer <path>`: required, vLLM inference config YAML
- `--orch <path>`: required, orchestrator config YAML
- `--vllm-gpus <ids>`: required, comma-separated GPU ids for vLLM. Count must equal `infer.dp * infer.tp`
- `--trainer-gpus <ids>`: required, comma-separated GPU ids for the trainer. Count must equal `train.gpus`, which
  therefore becomes optional in the YAML
- `--judge-infer <path>`: optional, inference config for a RULER judge server. With `--judge-gpus` and
  `ruler.enabled: true` in `orch.yaml`, a second vLLM subprocess is spawned for the judge
- `--judge-gpus <ids>`: optional, GPU ids for the judge. Must be disjoint from `--vllm-gpus` and `--trainer-gpus`

### `grpo-colocate`

Same three configs, but vLLM and the trainer **share** GPUs and exchange base weights
via zero-copy CUDA IPC. `gpu_memory_utilization` is computed automatically.

```bash
surogate grpo-colocate --train examples/grpo/train.yaml --infer examples/grpo/infer.yaml --orch examples/grpo/orch.yaml
```

Options: `--train`, `--infer`, `--orch` (all required). No GPU-assignment flags — the
components share every visible GPU.

### `grpo-infer`

Runs only the inference server. Use it for multi-node setups, or any case where each
component should own its process. This is the server that exposes the weight-update and
LoRA hot-load admin routes the trainer broadcasts into — a stock `vllm serve` does not.

```bash
CUDA_VISIBLE_DEVICES=0 surogate grpo-infer infer.yaml
```

### `grpo-orch`

Runs only the orchestrator: samples rollouts against the environments, scores them, and
writes batches to the transport.

```bash
surogate grpo-orch orch.yaml
```

### `grpo-train`

Runs only the trainer: consumes batches, applies GRPO updates, and broadcasts weights.
Resumes from the latest checkpoint in `checkpoint_dir` automatically — see
[GRPO (RL) Settings](config.md#grpo-rl-settings).

```bash
CUDA_VISIBLE_DEVICES=1 surogate grpo-train train.yaml
```

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
