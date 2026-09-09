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

## Two engines. One workflow.

### Training engine

From raw text to specialized models, with Python configuration and native C++/CUDA execution.

| Capability | What you get |
|---|---|
| **Pretraining & full fine-tuning** | Train from scratch, continue pretraining, or update the full model with SFT. |
| **LoRA & QLoRA** | Adapter training with BF16 bases or FP8, NVFP4, and BnB/NF4 quantization; supported pre-quantized checkpoints and stacked LoRA adapters. |
| **Native precision recipes** | BF16, hybrid FP8, and Blackwell NVFP4, with configurable model, gradient, and adapter precision. |
| **GRPO reinforcement learning** | Reward environments, evaluation, and policy updates with native serving or vLLM; shared-weight single-GPU BF16 LoRA across supported training families except Nemotron. |
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
| **Qwen3.8 Flash-Next** | GGUF, CPU offload, GPU expert caching, and multiple GPUs. |
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

Use **Linux x86_64 and Python 3.12** with a supported NVIDIA GPU and CUDA 12.8, 12.9, or 13.x. See [hardware](#hardware) for the separate training and serving GPU targets.

```bash
curl -LsSf https://github.com/invergent-ai/surogate/releases/latest/download/install.sh | bash
source .venv/bin/activate
```

The installer selects a CUDA-specific wheel and downloads the example configurations.

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

Training buffers remain reserved during generation, so choose a model and context length that fit your GPU. Quantized bases, multiple GPUs, and checkpoint resume are not yet supported in this native mode. See the [GRPO guide](docs/guides/rl-training.md) for separate-GPU and vLLM options.

<details>
<summary><strong>Docker and source builds</strong></summary>

CUDA-specific containers are available as `ghcr.io/invergent-ai/surogate:latest-cu128`, `latest-cu129`, and `latest-cu130`.

```bash
# From the directory containing train.yaml; output stays in ./output on the host.
docker run --gpus all --rm \
  -v "$PWD:/workspace" -w /workspace \
  ghcr.io/invergent-ai/surogate:latest-cu129 sft train.yaml
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
| **Platform** | Linux x86_64; published wheels target Python 3.12; CUDA 12.8, 12.9, or 13.x. |
| **Training** | SM89+ in the current build: Ada (RTX 40 series, L4/L40), Hopper (H100/H200), and supported Blackwell targets. |
| **FP8 / NVFP4 training** | FP8 requires SM89+; native NVFP4 requires a supported Blackwell GPU and matching build. |
| **Generative serving** | Current default builds target **SM120a**: RTX 50 series and RTX PRO Blackwell. The SM89/Ada port compiles, with runtime validation pending. |
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
