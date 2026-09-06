# Inference

Surogate serves models with the same CLI that trains them: `surogate serve` starts a
standalone C++/CUDA runtime that answers OpenAI- and Anthropic-compatible HTTP requests.

```bash
surogate serve Qwen/Qwen3.6-27B --port 8080
```

## What the engine does

- **CUDA-graph-captured decode** over a paged KV cache, with continuous batching across lanes.
- **Prefix reuse** — a shared prompt prefix is prefilled once and reused (`--no-prefix-reuse` disables).
- **Elastic KV cache** — the pool is a virtual span and only the pages in use hold VRAM, so an
  idle model gives its cache back and several models on one GPU can share the room
  (`--elastic-kv-overcommit`; `--no-elastic-kv` for the static arena).
- **Streaming SSE**, tool calls, reasoning/thinking separation, request cancellation.
- **Speculative decoding** with MTP or DFlash draft heads (`--spec`, `--draft-tokens`).
- **Multi-GPU** as data-parallel replicas or pipeline stages (`--devices 0,1,...`). Tensor
  parallelism is not offered: P2P is disabled on the consumer cards this engine targets.
- **A model larger than VRAM** — weights the card has no room for live in pinned,
  device-mapped host memory and the kernels read them over PCIe: `--host-moe-layers N|all` for
  a mixture's routed experts, `--gpu-layers N` (`-ngl`) for whole layers. GLM-5.3-Flash, 200 GB,
  serves on one 32 GB card. Any target inherits this; the Flash-Next target adds a device LRU
  slot cache and CPU expert compute on top (`--expert-slots`, `--cpu-moe-share`).
- **Vision input** (images, video) for models that carry a vision tower (`--vision`).

Embedding models run it in its own process, on either GPU or CPU. See [Serving models](serving-models.md#embedding-model-cpu-and-gpu).

## A GGUF is served where it lies

Point `surogate serve` at a **Hugging Face repo id, a local safetensors directory, or a GGUF
file**.

```
HF repo id     ─┐                    ┌─ safetensors: converted once, cached
safetensors dir ┼─► surogate serve ──┤                                        ─► OpenAI/Anthropic
GGUF file      ─┘                    └─ GGUF: read in place, small index beside it
```

For a GGUF there is no copy and no requantisation. The first start writes a small index naming
the stretches of the file each tensor is assembled from, and the weights are read from the GGUF
itself on every load: a 22.13 GB `Q4_K_M` file gets a **70 MB** index, built in about 18 seconds,
and its K-quant tensors reach the GPU as the superblocks the file stores. Nothing is dequantised
on the way in, so serving a GGUF costs exactly the accuracy the file already has.

For a Hugging Face or safetensors source there is real work to do, and the result is cached under
`~/.cache/surogate/serve` so later starts are a file read. That cached form is an internal,
regenerable detail, never an interchange format. Either way the artifact carries the model's own
tokenizer and chat template, so a serving host needs no Python and no `transformers`.

A model you trained here has no GGUF yet; `surogate quantize` produces one from a merged
checkpoint (see the [CLI reference](../reference/cli.md)).

## Supported quantizations

Weight storage is a closed registry — every tensor the engine loads is in one of these:

GGML's own block formats are served as the file stores them, byte for byte:

| Format | Weight | Block | Scales | Notes |
|---|---|---|---|---|
| `Q4_K` | 4-bit | 256 | two binary16 plus 6-bit sub-scales | 144 B a superblock |
| `Q5_K` | 5-bit | 256 | same, plus a high-bit plane | 176 B |
| `Q6_K` | 6-bit | 256 | one binary16, 8-bit sub-scales | 210 B, symmetric |
| `Q2_K`, `Q3_K` | 2- and 3-bit | 256 | | accepted, rarely wanted |
| `Q8_0` | 8-bit signed | 32 | one binary16 | identical to `W8G32_F16S` |

The rest are the engine's own, for sources that are not GGUF:

| Format | Weight | Group | Scale | Typical use |
|---|---|---|---|---|
| `BF16` | 16-bit float | — | — | norms, small projections, embedding tables |
| `FP32` | 32-bit float | — | — | host-side reference objects |
| `I32` | 32-bit int | — | — | index resources |
| `W8G32_F16S` | 8-bit signed (−127…127) | 32 | binary16 per group | the workhorse; identical to GGUF `Q8_0` |
| `NVFP4` | E2M1 (4-bit float) | 16 | E4M3FN per group | Blackwell FP4 tensor cores |
| `FP8_E4M3FN_ROW_BF16S` | E4M3FN | per row | BF16 per row | FP8 checkpoints |
| `Q4G64_F16S`, `Q5G64_F16S`, `Q6G64_F16S` | 4-, 5-, 6-bit signed | 64 | binary16 per group | older home-grown formats, being retired |

### What converts from what

| Source | Handling |
|---|---|
| GGUF K-quants (`Q2_K`…`Q6_K`) | **Served natively.** The superblocks are read from the file as they are; no dequantise, no requantise, no copy. |
| GGUF `Q8_0` | **Bit-exact rearrangement** into `W8G32_F16S` — the same numbers (int8 codes, one binary16 scale per 32), gathered into code and scale planes at load. |
| GGUF `Q4_0`, `Q5_0`, `IQ4_NL` | Bit-exact plane repack by the same path. |
| HF safetensors BF16 | Encoded direct, or quantized per the family recipe. |
| NVFP4 checkpoints (vLLM / compressed-tensors) | Block scales re-encoded to the engine's swizzle; paired with a BF16 base checkpoint for the objects NVFP4 does not carry. |
| FP8 row-scaled checkpoints | Encoded as `FP8_E4M3FN_ROW_BF16S`. |

Serving a K-quant GGUF is therefore lossless with respect to the file. Measured against
llama.cpp on the same `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, wikitext-2 perplexity over 145 windows
of 2048 tokens: **6.2370 for us, 6.2311 for llama.cpp**, both ±0.040.

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
| Qwen3 | any | dense |
| Llama | any | dense |
| Gemma 3 | any | dense; sliding-window attention |
| Qwen3.5 | any | dense; NVFP4 for the 4B |
| Qwen3.6 | 27B | dense hybrid; BF16 and NVFP4 |
| Qwen3.6 MoE | 35B-A3B | routed experts; optional draft head |
| Qwen3.8 | 27B | BF16 and NVFP4 |
| Qwen3.8 Flash-Next | MoE | GGUF source; the CPU-offload tier |
| EmbeddingGemma | 300M | encoder; GPU and CPU |

"Any" means what it says. A checkpoint states its own dimensions in the index built beside
it, and the engine binds against those, so a family listed that way serves whatever size you
hand it -- Qwen3-1.7B is served by the same code as Qwen3-0.6B, with nothing to register. The
sizes named for the other families are the ones their loaders are still written around.

An unrecognised model is refused at load with the reason printed, never served incorrectly.

## Hardware

The engine targets **sm_89 and sm_120** — RTX 4070 and 4090, RTX 5070/5080/5090, and
RTX Pro 6000 Blackwell. Multi-GPU is available through data-parallel replicas or pipeline stages. Tensor parallelism is not offered on any of them because P2P/NVLink is not enabled on consumer cardds. 

The CPU path covers the **encoder (embedding) models only** and needs AVX-512 support in the CPU.

## Next

- [OpenAI-compatible API](api.md) — endpoints, supported request fields, and what is refused.
- [CLI and parameters](cli.md) — every flag on the three serving binaries.
- [Serving models](serving-models.md) — worked NVFP4, GGUF, and embedding examples.
