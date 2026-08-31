# Inference

Surogate serves models with the same CLI that trains them: `surogate serve` starts a
standalone C++/CUDA runtime that answers OpenAI- and Anthropic-compatible HTTP requests.

```bash
surogate serve Qwen/Qwen3.6-27B --port 8080
```

It is deliberately **not** the training executor. A training step wants a full graph with
activations retained for backward; a decode round wants one token per sequence at minimum
latency, forever. Serving is its own runtime, and decode never routes through the training
graph executor.

## What the engine does

- **CUDA-graph-captured decode** over a paged KV cache, with continuous batching across lanes.
- **Prefix reuse** — a shared prompt prefix is prefilled once and reused (`--no-prefix-reuse` disables).
- **Streaming SSE**, tool calls, reasoning/thinking separation, request cancellation.
- **Speculative decoding** with MTP or DFlash draft heads (`--spec`, `--draft-tokens`).
- **Multi-GPU** as data-parallel replicas or pipeline stages (`--devices 0,1,...`). Tensor
  parallelism is not offered: P2P is disabled on the consumer cards this engine targets.
- **MoE larger than VRAM** — a pinned host expert bank, a device LRU slot cache, and optional
  CPU expert compute that takes a measured share of the routed work (`--expert-slots`,
  `--cpu-moe-share`).
- **Vision input** (images, video) for models that carry a vision tower (`--vision`).

Embedding models take a separate, much smaller path: an encoder runs **one forward** — no KV
cache, no sampler, no CUDA graphs, no round N+1 — so `surogate serve --embed` runs it in its own
process, on either GPU or CPU. See [Serving models](serving-models.md#embedding-model-cpu-and-gpu).

## Conversion is transparent

Point `surogate serve` at a **Hugging Face repo id, a local safetensors directory, or a GGUF
file**. The first load converts the model into engine-owned layouts and caches the result under
`~/.cache/surogate/serve`; every later start reuses it and begins in seconds.

```
HF repo id     ─┐
safetensors dir ┼─► surogate serve ─► (convert once, cached) ─► OpenAI/Anthropic endpoint
GGUF file      ─┘
```

The cached form is an internal, regenerable detail — not an interchange format. Nothing is
quantized or re-laid-out at startup, so load time is a file read, and the cache carries the
model's own tokenizer and chat template, so a serving host needs no Python and no
`transformers`.

## Supported quantizations

Weight storage is a closed registry — every tensor the engine loads is in one of these:

| Format | Weight | Group | Scale | Typical use |
|---|---|---|---|---|
| `BF16` | 16-bit float | — | — | norms, small projections, embedding tables |
| `FP32` | 32-bit float | — | — | host-side reference objects |
| `I32` | 32-bit int | — | — | index resources |
| `Q4G64_F16S` | 4-bit signed (−8…7) | 64 | binary16 per group | aggressive weight compression |
| `Q5G64_F16S` | 5-bit signed (−16…15) | 64 | binary16 per group | |
| `Q6G64_F16S` | 6-bit signed (−32…31) | 64 | binary16 per group | |
| `W8G32_F16S` | 8-bit signed (−127…127) | 32 | binary16 per group | the workhorse; identical to GGUF `Q8_0` |
| `NVFP4` | E2M1 (4-bit float) | 16 | E4M3FN per group | Blackwell FP4 tensor cores |
| `FP8_E4M3FN_ROW_BF16S` | E4M3FN | per row | BF16 per row | FP8 checkpoints |

### What converts from what

| Source | Handling |
|---|---|
| GGUF `Q8_0` | **Bit-exact repack** into `W8G32_F16S` — the two are the same numeric format (int8 codes, one binary16 scale per 32 values). No dequantize, no GPU. |
| GGUF `Q4_0`, `Q5_0`, `IQ4_NL` | Bit-exact plane repack by the same path. |
| GGUF K-quants (`Q4_K`, `Q5_K`, `Q5_1`) | Dequantized, then quantized to `W8G32_F16S` (measured 0.55 % rel-L2 on Flash-Next experts). |
| HF safetensors BF16 | Encoded direct, or quantized per the family recipe. |
| NVFP4 checkpoints (vLLM / compressed-tensors) | Block scales re-encoded to the engine's swizzle; paired with a BF16 base checkpoint for the objects NVFP4 does not cover. |
| FP8 row-scaled checkpoints | Encoded as `FP8_E4M3FN_ROW_BF16S`. |

### KV cache precision

Separate from weight storage, and set at launch: `--kv-cache-dtype auto|fp8|bf16|int8`.
The default is **fp8 (e4m3)**, which halves the cache; `auto` means the same; `bf16` asks for a
full-precision cache. Only full-attention layers hold a KV cache at all, so linear-attention
(GDN) layers are never quantized. `--kv-cache-dtype-skip-layers L,...` holds named layers at
BF16. `int8` exists for experiments and is not recommended — it costs measurable accuracy.

## Model families

Recognised automatically from the checkpoint:

| Family | Sizes | Notes |
|---|---|---|
| Qwen3.5 | 0.8B, 2B, 4B | dense; NVFP4 for the 4B |
| Qwen3.6 | 27B | dense; BF16 and NVFP4 |
| Qwen3.6 MoE | 35B-A3B | routed experts; optional draft head |
| Qwen3.8 | 27B | BF16 and NVFP4 |
| Qwen3.8 Flash-Next | MoE | GGUF source; the CPU-offload tier |
| EmbeddingGemma | 300M | encoder; GPU and CPU |

An unrecognised model is refused at load with the reason printed, never served incorrectly.

## Hardware

The engine targets **sm_89 and sm_120** — RTX 4070 and 4090, RTX 5070/5080/5090, and
RTX Pro 6000 Blackwell. Tensor parallelism is not offered on any of them: P2P is disabled on
consumer cards, so multi-GPU means data-parallel replicas or pipeline stages.

The CPU path covers the **encoder (embedding) models only**. It needs AVX-512, dispatches at
runtime rather than at build time, and reaches vendor GEMM libraries through a seam:
oneDNN by default (also what OpenVINO's CPU plugin uses underneath), ZenDNN when injected at
configure time, and a portable AVX-512 microkernel that is always present.

## Next

- [OpenAI-compatible API](api.md) — endpoints, supported request fields, and what is refused.
- [CLI and parameters](cli.md) — every flag on the three serving binaries.
- [Serving models](serving-models.md) — worked NVFP4, GGUF, and embedding examples.
