# Inference

Surogate serves models with the same CLI that trains them. Start a server, then connect an
OpenAI- or Anthropic-compatible client:

```bash
surogate serve Qwen/Qwen3.6-27B --port 8080
```

## Features

- **Chat and text completion**, with streaming responses, tool calls, and separate reasoning output.
- **Concurrent requests** — set `--max-num-seqs` to the number of requests to process at once.
- **Prompt caching** — reuse compatible earlier prompts to reduce processing time.
- **Speculative decoding** — speed up supported models with MTP or DFlash (`--spec`).
- **Models larger than GPU memory** — use system RAM for part of a model, or spread supported
  models across several GPUs.
- **Several models across GPUs**, with per-model placement and optional sleep mode to free memory when a model is idle.
- **Images and video** for supported vision models (`--vision`).
- **Embeddings** on GPU or CPU through `surogate serve --embed`.

See [Serving models](serving-models.md) for worked examples.

## Loading a model

The model argument can be a **Hugging Face repo id, a local safetensors directory, or a GGUF
file**:

```bash
surogate serve Qwen/Qwen3.6-27B
surogate serve ~/models/qwen3.6-27b-hf/
surogate serve ~/models/qwen3.6-27b-Q4_K_M.gguf
```

Surogate reads the model's dimensions and context limit from `config.json`, or from the GGUF
metadata. Keep the checkpoint's configuration and tokenizer files alongside local safetensors
weights. Renaming the directory or setting `--served-model-name` changes the name used to
identify the model; it does not select a different model size or change its settings.

The first start prepares the model and saves reusable files under `~/.cache/surogate/serve`.
Later starts skip that preparation, but still need time to load the model. Set
`SUROGATE_SERVE_CACHE` to use a different cache directory; `--no-cache` rebuilds an entry.
Changes to local configuration, tokenizer, or chat-template files trigger fresh preparation.

Supported GGUF weights are read from the original file, so preparation does not require a
second full copy of the model. Keep the GGUF files at their original paths while using the
cache. For a split GGUF, pass the first shard; the remaining shards are found automatically.

The model's tokenizer and chat template are included during preparation. A base model without
a chat template can be used through `/v1/completions`.

To serve a model you trained here, merge its adapter first. You can serve the merged
safetensors directory directly or create a GGUF with `surogate quantize`; see the
[CLI reference](../reference/cli.md).

## Supported quantizations

Quantization is detected from the model files. There is no separate serving flag to select the
weight format. Supported formats depend on the model family and export:

| Format | What to know |
|---|---|
| GGUF K-quants (`Q2_K` through `Q6_K`) | Supported models retain the GGUF's weight quantization |
| GGUF `Q8_0` | Eight-bit weights; preparation preserves their values |
| Other GGUF formats | Includes `Q4_0`, `Q5_0`, `Q4_1`, `Q5_1`, and supported IQ formats; availability depends on the model |
| BF16 safetensors | Prepared using the model family's conversion settings |
| NVFP4 checkpoints | Four-bit floating-point weights for supported Blackwell GPUs |
| FP8 checkpoints | Supported row- and block-scaled exports |

### Preparing model files

| Source | First-start behavior |
|---|---|
| Supported GGUF | Prepare cached metadata and read supported weights from the source files |
| Hugging Face repo | Download the checkpoint, prepare it, and cache the result |
| Local safetensors directory | Prepare and cache the model without downloading the checkpoint |
| NVFP4 export | Detect its quantization settings; some exports also need a compatible base checkpoint for missing model files |

Weight precision and prompt-cache precision are separate settings. Host offload also has its
own precision option: forcing `--host-expert-bank q4` can reduce the precision of weights
originally stored at more than four bits.

### KV cache precision

The KV cache stores information from earlier tokens so the model can continue a response
without processing the whole conversation again. Set its precision with
`--kv-cache-dtype auto|fp8|bf16|int8`.

The default, **auto**, chooses a setting for the model: BF16 for models such as Qwen3 and
Llama, and FP8 for hybrid models such as Qwen3.5/3.6/3.8. Explicit `fp8` uses half the cache
storage of BF16; explicit `bf16` keeps BF16 regardless of model family. Changing cache
precision can affect output quality. `int8` is also available for comparison.

DFlash supports BF16 and FP8 caches. Use `--kv-cache-dtype fp8` to reduce cache memory.

## Model families

The family is detected automatically. Available formats and optional features vary by model:

| Family | Notes |
|---|---|
| Qwen3 | Dense models |
| Qwen3-VL | Dense and MoE safetensors or GGUF checkpoints; text, images, and video (`--vision`). GGUFs need a matching `--mmproj` file. |
| Qwen3 MoE | Mixture-of-experts models |
| Llama | Includes TinyLlama |
| MiniCPM5 | Hugging Face safetensors and GGUF; thinking can be enabled or disabled |
| Spark-X2.5 | Hugging Face safetensors; thinking can be enabled or disabled |
| Gemma 3 | Text generation; images and sampled video frames on vision-enabled checkpoints |
| Gemma 4 | Text, images and video with dense, E-series, and mixture-of-experts models |
| LFM2 / LFM2.5 | Dense text models from Hugging Face safetensors or GGUF |
| LFM2-MoE | Hugging Face safetensors and GGUF |
| LFM2-VL / LFM2.5-VL | Text, images and sampled video frames from safetensors or paired GGUF files; use `--vision` |
| Qwen3.5/3.6/3.8 | Includes BF16 and NVFP4 exports |
| Qwen3.5/3.6 MoE | Includes 35B-A3B; optional speculative decoding |
| Qwen3.8 Flash-Next | GGUF; supports CPU offload |
| GLM-5.3-Flash | GGUF; supports CPU offload, multiple GPUs, and MTP |
| EmbeddingGemma 300M | Embeddings on GPU or CPU |

Model size is detected from the checkpoint. Its format and features must still be supported;
unsupported checkpoints are refused with an error message.

LFM2, LFM2.5, and LFM2-MoE GGUF files include their configuration, tokenizer, and chat template, so no
separate Hugging Face download is needed. They are prepared as 8-bit serving weights;
lower-bit GGUFs can therefore require more disk space and memory after preparation.

## Hardware

Supported GPU builds target NVIDIA Ada and Blackwell cards, including RTX 4070/4090,
RTX 5070/5080/5090, and RTX Pro 6000 Blackwell. The accelerated NVFP4 path requires Blackwell.
See the [CLI reference](cli.md#devices) for models that can use several GPUs.

CPU-only serving supports **embedding models** and requires AVX-512. Generative models need
an NVIDIA GPU even when some work is offloaded to the CPU.

## Next

- [OpenAI-compatible API](api.md) — endpoints, request fields, and supported client features.
- [CLI and parameters](cli.md) — options and defaults.
- [Serving models](serving-models.md) — NVFP4, GGUF, offload, and embedding examples.
