<div align="center">

<a href="https://surogate.ai">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-white.svg">
    <img src="assets/surogate-logo.png" alt="Surogate" width="150">
  </picture>
</a>

<h1>Surogate</h1>

<p><strong>Native C++/CUDA LLM training and serving: GGUF, NVFP4, BF16, FP8. <br>
From your first fine-tune to hundreds of concurrent requests.</strong></p>

<p>
  <a href="https://surogate.ai">Website</a> ·
  <a href="docs/index.md">Documentation</a> ·
  <a href="#speed-you-can-measure">Benchmarks</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#supported-models">Models</a> ·
  <a href="examples">Examples</a>
</p>

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![C++ / CUDA](https://img.shields.io/badge/Engines-C%2B%2B_%2F_CUDA-76B900)](csrc/src)
[![GitHub stars](https://img.shields.io/github/stars/invergent-ai/surogate?style=social)](https://github.com/invergent-ai/surogate)
[![Follow on X](https://img.shields.io/twitter/follow/surogate_ai?style=social)](https://x.com/surogate_ai)

<table>
<tr>
<td align="center"><strong>136,200 tok/s</strong><br>Training · 4× RTX 5090<br><sub>Qwen3-0.6B · FP4 LoRA</sub></td>
<td align="center"><strong>2.53× training throughput</strong><br>vs. Unsloth · 1× H100<br><sub>Qwen3-0.6B · BF16 on both</sub></td>
<td align="center"><strong>802 tok/s</strong><br>Serving · one user · 1× RTX 5090<br><sub>Qwen3.5-0.8B · native GGUF</sub></td>
<td align="center"><strong>7.0× serving throughput</strong><br>vs. llama.cpp · 8× RTX 5090<br><sub>GLM-5.3-Flash · 16 users · decode</sub></td>
</tr>
</table>

<sub>Selected results from the repository's <a href="docs/reference/benchmarks.md">training benchmarks</a> and <a href="surogate/serve/BENCHMARKS.md">serving benchmarks</a>. Workloads and comparison details below.</sub>

</div>

Surogate puts **training and serving in one toolkit**, with dedicated native engines built to push NVIDIA hardware. Pretrain a model, fine-tune with LoRA or QLoRA, optimize with GRPO or DPO, distill a teacher, and serve the result through familiar HTTP APIs.

Speed drives the design: compiled training graphs, fused CUDA kernels, low-precision tensor cores, concurrent serving, and deliberate control over every byte of GPU memory. Scale from a single workstation to multiple GPUs and training clusters, or use system RAM to work with models larger than your cards.

## Speed you can measure

### Serving: fast for one user. Fast under load.

**Native GGUF decoding at 802 tokens/s on one RTX 5090. A 200 GB MoE serving 16 users at 7× llama.cpp's aggregate decode throughput on eight cards.**

| Model / workload | RTX 5090 GPUs | Surogate decode | Baseline decode | Throughput gain |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B · 1 user | 1 | **802 tok/s** | vLLM: 346 tok/s | **2.32×** |
| Qwen3.5-0.8B · 8 users | 1 | **2,765 tok/s** | llama.cpp: 683 tok/s | **4.05×** |
| Qwen3.5-4B · 100 users | 1 | **5,345 tok/s** | vLLM: 4,481 tok/s | **1.19×** |
| GLM-5.3-Flash · 16 users | 8 | **292.9 tok/s** | llama.cpp: 41.9 tok/s | **7.0×** |

Responsiveness matters, too. At 100 users, Qwen3.5-4B delivers **40 ms median time to first token**, versus 230 ms for vLLM. At 16 users, GLM-5.3-Flash delivers **1.59 seconds**, versus 55.13 seconds for llama.cpp.

### Training: more experiments per GPU-hour

**53,900 training tokens/s on a single H100. 136,200 on four RTX 5090s.**

| Model | Hardware | Surogate | Unsloth | Throughput gain |
|---|---|---:|---:|---:|
| Qwen3-0.6B | 1× H100 80 GB | **53,900 tok/s · BF16** | 21,300 tok/s · BF16 | **2.53×** |
| Qwen3-0.6B | 1× RTX 5090 32 GB | **30,100 tok/s · BF16** | 22,100 tok/s · BF16 | **1.36×** |
| Qwen3-8B | 1× RTX 5090 32 GB | **6,900 tok/s · FP4** | 3,500 tok/s · BF16 | **1.97×** |

Need more throughput? Qwen3-0.6B FP4 LoRA scales from **36,400 tok/s on one RTX 5090 to 136,200 tok/s on four**: 3.74× aggregate throughput. Qwen3-8B reaches **27,100 tok/s** on the same four-card setup.

### Every model and GPU we have measured

Each comparison ran both engines on the same kind of GPU with the same requests. A ratio above 1× favours Surogate. The rows where the other engine leads stay in the table.

**Serving.** Decode and prefill are tokens/s summed over all users, except where a row says "per user".

| GPU | Model · weights | Load | Surogate | Other engine | Ratio |
|---|---|---|---:|---:|---:|
| RTX 5090 | Qwen3.5-0.8B · GGUF Q4_K_M | 1 user, decode | **802 tok/s** | vLLM (NVFP4): 346 | **2.32×** |
| RTX 5090 | Qwen3.5-0.8B · GGUF Q4_K_M | 8 users, decode | **2,765 tok/s** | llama.cpp: 683 | **4.05×** |
| RTX 5090 | Qwen3.5-4B · NVFP4 | 1 user, decode | **313 tok/s** | vLLM: 249 | **1.26×** |
| RTX 5090 | Qwen3.5-4B · NVFP4 | 100 users, decode | **5,345 tok/s** | vLLM: 4,481 | **1.19×** |
| RTX 5090 | Qwen3.8-27B · NVFP4 | 1 user, decode | 70.1 tok/s (107.3 with MTP on benchmark prose) | vLLM: 71.6 | 0.98× |
| RTX 5090 | Qwen3.8-27B · NVFP4 | 100 users, decode | **1,321 tok/s** | vLLM: 1,109 | **1.19×** |
| RTX 5090 | Qwen3.8-27B · NVFP4 | 100 users, 2,048-token prompts, prefill | 12,006 tok/s | vLLM: 12,402 | 0.97× |
| RTX 5090 | Qwen3.6-35B-A3B · NVFP4 | 100 users, decode | **2,607 tok/s** | vLLM: 2,162 | **1.21×** |
| RTX 5090 | Qwen3.8-Flash-Next · GGUF, experts in host RAM | 1 user, decode | **35.2 tok/s** | llama.cpp: 20.8 | **1.69×** |
| RTX 5090 | Qwen3.8-Flash-Next · GGUF, experts in host RAM | 16 users, decode | **96.7 tok/s** | llama.cpp: 56.8 | **1.70×** |
| 8× RTX 5090 | GLM-5.3-Flash · GGUF | 16 users, decode | **292.9 tok/s** | llama.cpp: 41.9 | **7.0×** |
| H100 | Qwen3-8B · FP8 | 1 user, decode | **233.4 tok/s** | vLLM: 228.8 | **1.02×** |
| H100 | Qwen3-8B · FP8 | 16 users, decode per user | **171.6 tok/s** | vLLM: 153.6 | **1.12×** |
| H100 | Qwen3-8B · FP8 | 64 users, decode per user | 90.1 tok/s | vLLM: 92.4 | 0.98× |
| H100 | Qwen3-8B · FP8 | 32 users, 2,048-token prompts, prefill | 52.5k tok/s | vLLM: 55.1k | 0.95× |
| H100 | Qwen3-8B · FP8 | same load, time to first token (p50) | 661 ms | vLLM: 325 ms | 0.49× |
| H100 | Qwen3.6-35B-A3B · FP8 | 1 user, decode | **282 tok/s** | vLLM: 251 | **1.12×** |
| H100 | Qwen3.6-35B-A3B · FP8 | 64 users, decode | **2,291 tok/s** | vLLM: 1,923 | **1.19×** |
| H100 | Qwen3.6-35B-A3B · FP8 | 32 users, 2,048-token prompts, prefill | 35.7k tok/s | vLLM: 39.5k | 0.90× |
| H100 | Qwen3.6-35B-A3B · FP8 | 32 agents, 6k shared system prompt, 8 turns | **331 turns/min** | vLLM: 269 | **1.23×** |
| H100 | Qwen3.6-27B · FP8 | 64 users, decode | 1,434 tok/s | vLLM: 1,626 | 0.88× |
| DGX Spark | Surogate3.7-35B-A3B · NVFP4 | 1 user, decode | **70.4 tok/s** | vLLM: 60.8 | **1.16×** |
| DGX Spark | Surogate3.7-35B-A3B · NVFP4 | 8 users, decode | **220.8 tok/s** | vLLM: 204.8 | **1.08×** |
| DGX Spark | Surogate3.7-35B-A3B · NVFP4 | 32 users, decode | **473.6 tok/s** | vLLM: 406.4 | **1.17×** |
| DGX Spark | Surogate3.7-35B-A3B · NVFP4 | 8 users, 2,048-token prompts, prefill | **6,042 tok/s** | vLLM: 5,069 | **1.19×** |
| DGX Spark | Surogate3.7-35B-A3B · NVFP4 | cold 108k-token prompt, time to first token | **22.2 s** | vLLM: 32.4 s | **1.46×** |
| DGX Spark | Surogate3.7-35B-A3B · NVFP4 | agents at 100k-200k context, warm-turn time to first token (p50), 1 / 8 agents | 2.2 s / **4.7 s** | vLLM: 1.8 s / 7.6 s | 0.82× / **1.62×** |
| DGX Spark | Surogate3.7-35B-A3B · NVFP4 | 16 users, real text, MTP | **444.7 tok/s** (362.8 without MTP) | not measured | — |
| DGX Spark | Qwen3-8B · FP8 | 1 user, decode | **28.8 tok/s** | vLLM: 22.4 | **1.29×** |
| DGX Spark | Qwen3-8B · FP8 | 8 users, decode | **182.4 tok/s** | vLLM: 153.6 | **1.19×** |
| DGX Spark | Qwen3-8B · FP8 | 32 users, decode | 470.4 tok/s | vLLM: 496.0 | 0.95× |
| DGX Spark | Qwen3-8B · FP8 | 8 users, 2,048-token prompts, prefill | **5,939 tok/s** | vLLM: 5,632 | **1.05×** |
| DGX Spark | Qwen3.8-Flash-Next · NVFP4 | 1 user, decode | **31.4 tok/s** (with MTP: 45.9 on real text, 67.2 on benchmark prose) | vLLM: does not fit in 121.7 GiB | — |
| DGX Spark | Qwen3.8-Flash-Next · NVFP4 | 16 users, real text, MTP | **115.7 tok/s** (97.2 without MTP) | vLLM: does not fit | — |
| DGX Spark | Qwen3.8-Flash-Next · NVFP4 | cold 32k-token prompt, prefill | **2,008 tok/s** | llama.cpp (UD-IQ4_XS GGUF): 629 | **3.2×** |
| DGX Spark | EmbeddingGemma-300M · W8 | 16 clients, one text per request | **2,442 texts/s** | vLLM (BF16): 510-562 | **4.3×** |
| DGX Spark | EmbeddingGemma-300M · W8 | 4 clients, 32 texts per request | **814 texts/s** | vLLM (BF16): 745 | **1.09×** |

Also measured, without another engine alongside: single-user decode of Qwen3-0.6B at 685 tok/s and Qwen3.5-0.8B at 563 tok/s on an H100; Qwen3-0.6B at 220 tok/s, Qwen3.5-4B NVFP4 at 73 tok/s, Gemma 3 1B Q8_0 at 138 tok/s and Qwen3.6-27B NVFP4 at 12.0 tok/s on a DGX Spark, the last at the Spark's memory-bandwidth limit.

**Training.** LoRA, tokens/s.

| GPU | Model | Surogate | Unsloth | Ratio |
|---|---|---:|---:|---:|
| H100 80 GB | Qwen3-0.6B | **53,900 · BF16** | 21,300 · BF16 | **2.53×** |
| H100 80 GB | Qwen3-8B | **11,600 · FP8** | 8,900 · BF16 | **1.30×** |
| H100 80 GB | Qwen3-14B | **7,200 · FP8** | 5,600 · BF16 | **1.29×** |
| RTX 5090 | Qwen3-0.6B | **30,100 · BF16** | 22,100 · BF16 | **1.36×** |
| RTX 5090 | Qwen3-8B | **6,900 · FP4** | 3,500 · BF16 | **1.97×** |
| 4× RTX 5090 | Qwen3-0.6B | **136,200 · FP4** | — | — |
| 4× RTX 5090 | Qwen3-8B | **27,100 · FP4** | — | — |
| DGX Spark | Qwen3-0.6B | ~7,100 · FP8 | — | — |
| DGX Spark | Qwen3-8B | 1,433 · FP8 | — | — |
| DGX Spark | Qwen3.6-35B-A3B | 674 · FP8 | — | — |

<sub>RTX 5090 serving rows: 2026-08-30 to 2026-09-07, cards at a 400 W limit, vLLM 0.27.1 and llama.cpp CUDA builds. H100 and DGX Spark rows: 2026-10-05 to 2026-10-09, vLLM 0.31. On the DGX Spark, vLLM's FP8 KV cache fails to start, so vLLM ran a BF16 cache; Surogate ran a BF16 cache for the 35B's short-prompt rows and an FP8 cache for its long-context and MTP rows and for Flash-Next. Qwen3.8-Flash-Next's NVFP4 release needs 123.5 GiB in vLLM on a 121.7 GiB Spark; a public vLLM preview with MTP is reported at 31-41 tok/s for one user there (not measured here). Surogate's Flash-Next rows run the 105 GB artifact converted from that release, and its prefill row is against llama.cpp on Unsloth's UD-IQ4_XS GGUF. Benchmark prose is the repeated 512-token text the throughput rows use, where up to 95% of MTP drafts are accepted; "real text" means distinct Wikipedia passages, where 52-54% are. Full methods: <a href="surogate/serve/BENCHMARKS.md">serving benchmarks</a>, <a href="docs/reference/benchmarks.md">training benchmarks</a>, and the pull requests behind each result (#296-#308).</sub>

## Two engines. One workflow.

### Training engine

From raw text to specialized models, with Python configuration and native C++/CUDA execution.

| Capability | What you get |
|---|---|
| **Pretraining & full fine-tuning** | Train from scratch, continue pretraining, or update the full model with SFT. |
| **LoRA & QLoRA** | Adapter training with BF16 bases or FP8, NVFP4, and BnB/NF4 quantization; supported pre-quantized checkpoints and stacked LoRA adapters. |
| **Native precision recipes** | BF16, hybrid FP8, and Blackwell NVFP4, with configurable model, gradient, and adapter precision. |
| **GRPO reinforcement learning** | Reward environments, evaluation, and policy updates with native serving; shared-weight single-GPU BF16 LoRA across supported training families except Nemotron. |
| **DPO preference training** | Learn from chosen/rejected pairs, with an inline frozen reference, optional length normalization, and differing-span masking. |
| **Knowledge distillation** | Capture teacher top-K distributions, then train a student with KL divergence and optional cross-entropy. |
| **Multi-GPU & multi-node** | Native threaded data parallelism, ZeRO sharding, communication overlap, and Ray for multi-node training. |
| **Models beyond one GPU's capacity** | Dispatch pipeline parallelism streams frozen weights for LoRA across PCIe GPUs, including systems without NVLink or GPU-to-GPU P2P. |
| **MoE training** | Expert parallelism, load balancing, routing metrics, and expert imbalance detection. |
| **Memory control** | CPU offload for weights, gradients, optimizer state, activations, and quants; recomputation and tiled MLP execution for long contexts. |
| **Multimodal fine-tuning** | Qwen3-VL and Qwen3.5 vision training examples, plus text-backbone training for supported multimodal checkpoints. |
| **Optimizers & monitoring** | AdamW 8-bit, NorMuon, Weights & Biases, loss charts, checkpoints, and resume. |
| **Adaptive training** | Phase detection, early stopping, learning-rate management, token budgeting, and dynamic epoch adjustment. |
| **Extensible architectures** | Python DSL with ahead-of-time automatic differentiation, explicit graphs, and native kernel dispatch. |

Explore [training modes](docs/getting-started/training-modes.md), [precision recipes](docs/guides/precision-and-recipes.md), [memory management](docs/guides/memory.md), [GRPO](docs/guides/rl-training.md), [DPO](docs/getting-started/quickstart-dpo.md), and [distillation](docs/guides/distillation.md).

### Serving engine

A native C++/CUDA HTTP server built for quick responses, concurrent workloads, and efficient model placement.

| Capability | What you get |
|---|---|
| **Familiar APIs** | OpenAI-compatible Chat Completions, Completions, and Responses; Anthropic-compatible Messages. See the [API guide](docs/inference/api.md) for supported fields. |
| **Streaming, reasoning & tools** | Streaming output, separate reasoning content, thinking controls, and model-specific function-call parsers. |
| **Concurrent serving** | Continuous batching, chunked prompt processing, CUDA graphs, and up to 128 active sequences per model, with a configurable pending queue. |
| **Prompt reuse** | Prefix caching and optional checkpoints for edited conversation turns. |
| **Speculative decoding** | MTP, including supported multi-GPU models, and DFlash with a compatible drafter on a single GPU. |
| **Native GGUF** | Supported K-quants, Q8_0, legacy and IQ formats; split-file loading and direct access to supported source weights. |
| **Hugging Face checkpoints** | Repo IDs or local safetensors, with automatic preparation and caching; supported BF16, FP8, and NVFP4 exports. |
| **KV memory on demand** | Elastic cache allocation, automatic capacity sizing, BF16/FP8 cache options, and shared spare cache memory across models. |
| **Runtime LoRA** | Load and unload compatible PEFT adapters without restarting; select an adapter per request. |
| **Several models on one GPU** | Named models, priorities, and optional sleep/wake to move idle models into system RAM. |
| **Large models on available hardware** | Multi-GPU layer pipelines, CPU weight offload for every serving generation model, plus GPU expert caching and CPU/GPU expert compute sharing for every supported MoE family. |
| **Vision & embeddings** | Images and video for supported vision models; EmbeddingGemma on GPU or AVX-512 CPU through a separate embeddings server. |
| **Operations** | API-key authentication, health checks, Prometheus metrics, request logs, tokenization, and cache statistics. |

Formats, LoRA, vision, speculation, and placement options depend on the model family. Multi-model hosting, sleep mode, and DFlash currently require one GPU. [Serving guide →](docs/inference/index.md) · [CLI reference →](docs/inference/cli.md) · [Deployment examples →](docs/inference/serving-models.md)

## Supported models

Training and serving have different architecture coverage. Model dimensions come from the checkpoint; the examples below are representative sizes, with precision and memory requirements depending on the configuration.

### Training

| Family | Representative models / sizes | Examples or implementation |
|---|---|---|
| **Qwen3** | 0.6B, 1.7B, 4B, 8B, 14B, 32B | [BF16, FP8, FP4 & QLoRA](examples/sft/qwen3) |
| **Qwen3 MoE** | 30B-A3B, 235B-A22B | [MoE recipes](examples/sft/qwen3moe) |
| **Qwen3-VL** | 2B, 4B, 8B, 32B | [Vision & text training](examples/sft/qwen3vl) |
| **Qwen3.5 / Qwen3.6 dense** | 3.5: 0.8B, 2B, 4B, 9B, 27B; 3.6: 27B | [Text, vision & pipeline examples](examples/sft/qwen35) |
| **Qwen3.5 / Qwen3.6 MoE** | 35B-A3B; 3.5 also 122B-A10B, 397B-A17B | [MoE recipes](examples/sft/qwen35moe) |
| **Llama 3.1 / 3.2** | 3.1: 8B, 70B, 405B; 3.2: 1B, 3B | [Llama example](examples/sft/llama) |
| **MiniCPM5** | 1B, 2B | [BF16 LoRA](examples/sft/minicpm5) |
| **Spark-X2.5** | 1.7B, 4B | [BF16 LoRA](examples/sft/spark) |
| **Gemma 4** | E2B, 12B, 26B-A4B; text backbones | [LoRA recipes](examples/sft/gemma4) |
| **Nemotron 3 / Cascade 2** | Nano 30B-A3B, Super 120B-A12B, Cascade 2 30B-A3B | [Nemotron recipes](examples/sft/nemotron3) |
| **GPT-OSS** | 20B, 120B; MXFP4 checkpoints use QLoRA | [GPT-OSS recipes](examples/sft/gpt-oss) |
| **Laguna** | Laguna-S-2.1 | [FP8 LoRA](examples/sft/laguna) |
| **LFM2 / LFM2.5** | LFM2: 350M, 700M, 1.2B, 2.6B; LFM2.5-350M example | [BF16 & FP8 LoRA](examples/sft/lfm2) |

Additional architecture definitions include [Gemma 3](surogate/dsl/models/gemma3.py), [LFM2-MoE](surogate/dsl/models/lfm2_moe.py), and [LFM2-VL](surogate/dsl/models/lfm2_vl.py). Check their implementation and validation coverage before using a new checkpoint. DeepSeek-V4, Flash-Next, and GLM-5.3-Flash training definitions still have deferred components and are not listed as ready-to-train models.

### Serving

| Family | Coverage |
|---|---|
| **Qwen3 / Qwen3 MoE** | Dense and mixture-of-experts text models. |
| **Qwen3.5 / Qwen3.6 / Qwen3.8** | Dense hybrid models, including supported BF16, FP8, and NVFP4 exports. |
| **Qwen3.5 / Qwen3.6 MoE** | Including 35B-A3B; optional speculative decoding with compatible draft weights. |
| **Qwen3.8 Flash-Next** | GGUF and NVIDIA's NVFP4 release, MTP, CPU offload, GPU expert caching, and multiple GPUs. |
| **GLM-5.3-Flash** | GGUF, CPU/GPU expert compute, multiple GPUs, and MTP. |
| **Llama** | Llama-family text models, including TinyLlama. |
| **Gemma 3 / Gemma 4** | Text generation; Gemma 4 dense, E-series, and MoE variants. |
| **LFM2 / LFM2.5** | Dense text models from safetensors or GGUF. |
| **MiniCPM5** | Safetensors and GGUF, with thinking controls. |
| **Spark-X2.5** | 1.7B and 4B safetensors, thinking controls, and `spark25` tool parsing; runtime LoRA is not yet supported. |
| **EmbeddingGemma 300M** | Embeddings on NVIDIA GPU or AVX-512 CPU. |

For per-family format support, conversion behavior, and hardware details, see the [serving model guide](docs/inference/index.md#model-families).

## Quickstart

### Install

Use **Linux x86_64 and Python 3.12** with a supported NVIDIA GPU and CUDA 13 (driver 580 or newer), or a DGX Spark (GB10, Linux aarch64). See [hardware](#hardware) for the separate training and serving GPU targets.

```bash
curl -LsSf https://github.com/invergent-ai/surogate/releases/latest/download/install.sh | bash
source .venv/bin/activate
```

The installer verifies the install and downloads the example configurations. The wheel is
also installable on its own, with no index flags — every dependency resolves from PyPI:

```bash
pip install https://github.com/invergent-ai/surogate/releases/latest/download/surogate-<version>+cu130-cp312-abi3-manylinux_2_39_x86_64.whl
```

On a DGX Spark the installer picks the `manylinux_2_39_aarch64` wheel, built for GB10 only.

There is no CUDA 12 package. The serving engine's W8 and NVFP4 kernels declare more shared
memory per block than a CUDA 12 toolkit will assemble for the RTX line, so CUDA 13.0 is the
floor for building it, and a package that serves is the product. The engine carries its own
FFmpeg, numa and ICU; it expects the system's glib and X11 client libraries, which a minimal
server image may lack (Ubuntu: `libglib2.0-0t64 libx11-6 libxext6 libxrender1`). The
installer checks.

### Serve a model

```bash
surogate serve Qwen/Qwen3.5-0.8B \
  --served-model-name surogate \
  --port 8080 --max-model-len 4096 \
  --max-num-seqs 16 --kv-capacity auto
```

The first launch prepares and caches the checkpoint. Later launches reuse that preparation. You can also pass a local safetensors directory or a supported `.gguf` file.

Send a streaming request from another terminal:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "surogate",
    "messages": [{"role": "user", "content": "Explain why low latency matters."}],
    "max_tokens": 256,
    "stream": true
  }'
```

Point existing OpenAI-compatible clients at `http://127.0.0.1:8080/v1`. To try a Blackwell NVFP4 checkpoint, use a supported export such as `nvidia/Qwen3.6-27B-NVFP4`. [More serving examples →](docs/inference/serving-models.md)

### Train an adapter

Save this as `train.yaml`:

```yaml
model: Qwen/Qwen3-0.6B
output_dir: ./output

recipe: bf16                  # fp8-hybrid for FP8; nvfp4 for Blackwell FP4
lora: true
lora_rank: 16
lora_alpha: 32
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

sequence_len: 2048
sample_packing: true
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2e-4
max_steps: 100
save_steps: 50

datasets:
  - path: mlabonne/FineTome-100k
    type: auto
```

```bash
surogate sft train.yaml
```

Choose a precision recipe for your GPU, or start with the checked-in [training examples](examples/sft). For memory-constrained runs, see [QLoRA](docs/guides/qlora.md) and [CPU offloading](docs/guides/offloading.md).

### Serve what you trained

The completed training run exports its final adapter directly to `output/`. Merge it into its base model, then start the server:

```bash
surogate merge \
  --base-model Qwen/Qwen3-0.6B \
  --checkpoint-dir output \
  --output merged

surogate serve merged --served-model-name my-finetune --port 8080
```

Use `"model": "my-finetune"` in requests. To create a quantized GGUF for serving:

```bash
surogate quantize --model merged --output merged-Q4_K_M.gguf --type q4_k_m
surogate serve merged-Q4_K_M.gguf --served-model-name my-finetune
```

Run one server at a time on the same port. Compatible adapters can also be [loaded at runtime](docs/inference/api.md#lora-adapters-at-runtime) with `--enable-lora`.

### Run GRPO on one GPU

**Load the base model once for both serving and training.** Native co-locate mode supports BF16 safetensors models with LoRA for text rollouts across the supported training families, excluding Nemotron. Dense Qwen3 and Qwen3.5 use optimized serving; other families reuse the training model for generation and recompute the prefix per token, which is slower. MoE models require `lora_dtype: bf16`, and the entire base must fit on one GPU. It generates a batch of rollouts, pauses generation for the training update, then generates the next batch with the updated adapter. Per-step adapter updates stay in GPU memory.

Create the three configuration files from the [Single-GPU GRPO guide](docs/guides/rl-colocate.md), using `backend: surogate` for inference and a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=0 surogate grpo-colocate \
  --train train.yaml --infer infer.yaml --orch orch.yaml
```

Training buffers remain reserved during generation, so choose a model and context length that fit your GPU. Quantized bases, multiple GPUs, and checkpoint resume are not yet supported in this native mode. See the [GRPO guide](docs/guides/rl-training.md) for the separate-GPU runner.

<details>
<summary><strong>Docker and source builds</strong></summary>

The container is `ghcr.io/invergent-ai/surogate:latest` (also tagged `latest-cu130` and by release version).

```bash
# From the directory containing train.yaml; output stays in ./output on the host.
docker run --gpus all --rm \
  -v "$PWD:/workspace" -w /workspace \
  ghcr.io/invergent-ai/surogate:latest-cu130 sft train.yaml
```

For development, clone the repository and install with a CUDA toolkit, NCCL development libraries, and FFmpeg/libcurl development packages for serving. See [CMake](csrc/CMakeLists.txt) for the full build configuration:

```bash
git clone https://github.com/invergent-ai/surogate.git
cd surogate
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

`make serve-build` rebuilds the serving engine in a source checkout. [Installation details →](docs/getting-started/installation.md)

</details>

## Why it is fast

- **Compiled training graphs.** A Python DSL defines the model; ahead-of-time autodiff produces its backward graph so the native runtime can plan execution and memory across both passes.
- **Kernels that do more per launch.** Fused operations, specialized matrix multiplication, low-precision tensor cores, and CUDA graphs reduce dispatch overhead and memory traffic.
- **Serving that keeps work moving.** Continuous batching, chunked prefill, prefix reuse, and speculative decoding improve throughput and latency for different request patterns.
- **Memory treated as part of the engine.** Planned buffer reuse, asynchronous offload, elastic KV caches, and expert placement make more of the available GPU and system memory useful.

Read [how training works](docs/about/how-it-works.md), the [DSL guide](docs/about/dsl.md), or the [serving benchmark analysis](surogate/serve/BENCHMARKS.md).

## Hardware

| Component | Current requirements / targets |
|---|---|
| **Platform** | Linux x86_64, and Linux aarch64 for DGX Spark (GB10); the published wheels target Python 3.12 and CUDA 13 (driver 580+). |
| **Training** | SM89+ in the current build: Ada (RTX 40 series, L4/L40), Hopper (H100/H200), and supported Blackwell targets, including GB10 (SM121, DGX Spark). |
| **FP8 / NVFP4 training** | FP8 requires SM89+; native NVFP4 requires a supported Blackwell GPU and matching build. |
| **Generative serving** | Default x86_64 builds target Ada (SM89: RTX 40 series, L4/L40), Hopper (SM90a: H100/H200) and RTX Blackwell (SM120a: RTX 50 series, RTX PRO); aarch64 builds target GB10 (SM121a: DGX Spark). NVFP4 checkpoints need SM120 or SM121. |
| **CPU embeddings** | AVX-512 CPU; generative serving still requires a GPU when using CPU offload. |
| **Multi-GPU / offload** | NCCL for distributed training; sufficient system RAM for offloaded weights and state. Dispatch-PP supports PCIe systems without NVLink. |

The training and serving engines have separate CUDA build targets. Check the [build configuration](csrc/CMakeLists.txt) when targeting a GPU beyond the default serving build.

## Explore and contribute

| Start here | Go deeper |
|---|---|
| [Training examples](examples) | [Configuration reference](docs/reference/config.md) |
| [Serving examples](docs/inference/serving-models.md) | [API reference](docs/inference/api.md) |
| [Pretraining](docs/getting-started/quickstart-pretraining.md) | [Multi-GPU](docs/guides/multi-gpu.md) · [Multi-node](docs/guides/multi-node.md) · [Dispatch-PP](docs/guides/dispatch-pp.md) |
| [GRPO](docs/getting-started/quickstart-grpo.md) · [DPO](docs/getting-started/quickstart-dpo.md) | [RL environments](docs/guides/rl-environments.md) · [Distillation](docs/guides/distillation.md) |
| [Training benchmarks](docs/reference/benchmarks.md) | [Serving benchmarks](surogate/serve/BENCHMARKS.md) · [Benchmark tools](surogate/serve/tools/bench) |

Contributions are welcome: model support, kernels, precision recipes, benchmarks, documentation, and bug fixes. Include the problem, how to reproduce or validate the change, and any GPU or architecture assumptions in your pull request. [Open an issue](https://github.com/invergent-ai/surogate/issues) or [send a PR](https://github.com/invergent-ai/surogate/pulls).

**Apache 2.0** · [License](LICENSE) · Built by [Invergent](https://github.com/invergent-ai)
