<div align="center" style="padding: 2rem">
<p align="center">
  <a href="https://surogate.ai/#gh-dark-mode-only">
    <img
      alt="Surogate"
      width="40%"
      src="https://github.com/invergent-ai/surogate/raw/main/assets/logo-white.svg#gh-dark-mode-only"
    />
  </a>

  <a href="https://surogate.ai/#gh-light-mode-only">
    <img
      alt="Surogate"
      width="40%"
      src="https://github.com/invergent-ai/surogate/raw/main/assets/logo-black.svg#gh-light-mode-only"
    />
  </a>
</p>

<h3>⚡ FP8/FP4 Training, Fine-tuning and RL at the speed of light</h3>
<h4></h4>
<div>
<a href="https://surogate.ai">Home</a> ·
<a href="https://docs.surogate.ai">Docs</a> ·
<a href="https://github.com/invergent-ai/surogate/tree/master/examples">Examples</a> ·
<a href="https://docs.surogate.ai/reference/benchmarks">Benchmarks</a> ·
<a href="https://github.com/invergent-ai/surogates">Managed Agents</a>
</div>
<br/>

[![GitHub stars](https://img.shields.io/github/stars/invergent-ai/surogate?style=social)](https://github.com/invergent-ai/surogate)
[![GitHub issues](https://img.shields.io/github/issues/invergent-ai/surogate)](https://github.com/invergent-ai/surogate/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/invergent-ai/surogate)](https://github.com/invergent-ai/surogate/pulls)
[![Twitter Follow](https://img.shields.io/twitter/follow/surogate_ai?style=social)](https://x.com/surogate_ai)

</div>

# Surogate Trainer
Surogate Trainer is built for developers and enterprises that need fast experimentation — whether running on-premise or in the cloud.

**⚡ The Surogate trainer surpasses all existing training frameworks in performance for single-GPU, multi-GPU and GPU+CPU by a large margin.**

**✨ The native CPU offloading feature achieves superior performance and VRAM usage compared to QLoRA. You can fine-tune models at native bf16 precision, rendering QLoRA obsolete.**

### Highlights

- **🔧 Pre-training + Fine-tuning**: full fine-tuning, LoRA
- [**🔧 BF16, FP8 and NVFP4 Reinforcement Learning**](https://docs.surogate.ai/guides/rl-training): advanced GRPO training and evaluation with custom, deterministic environments
- [**🔧 RL Environments***](https://docs.surogate.ai/guides/rl-environments): predictable environments for RL training
- [**🖥️...🖥️ Native multi-GPU**](https://docs.surogate.ai/guides/multi-gpu) training with multi-threaded backend
- [**🖥️...🖥️ Native multi-Node**](https://docs.surogate.ai/guides/multi-node) DDP training with Ray
- **⚡ Native C++/CUDA engine** for near–Speed-Of-Light (SOL) throughput
- [**🔥 Python DSL**](https://docs.surogate.ai/about/dsl) with AOT auto-differentiation for adding new model architectures
- [**⚖️ Smart CPU Offloading**](https://docs.surogate.ai/guides/offloading) for weights, gradients, activations, quants
- **📜 Pre-built training recipes**:
  - [**💎 BF16**](https://docs.surogate.ai/guides/precision-and-recipes#bf16): Baseline recipe using `bfloat16` for all GEMMs, designed for maximum numerical accuracy. No quantization is applied.
  - [**🔥 FP8**](https://docs.surogate.ai/guides/precision-and-recipes#fp8-hybrid): Native `FP8` training delivering extreme performance with `E4M3` used for activations and weights and `E5M2` for gradients. Uses per-tensor delayed scaling to provide stable training.
  - [**🔥 NVFP4**](https://docs.surogate.ai/guides/precision-and-recipes#fp4-nvfp4): Native CUTLASS `FP4 E2M1` training with two-level block scaling for extreme performance and memory efficiency on Blackwell GPUs (**SM100+**: B200, B300, RTX 50xx series). Uses stochastic rounding and random Hadamard Transforms for numerical stability. **Supports NVIDIA B200, B300, RTX 5070, 5080, 5090 !!**
- [**⚡ BnB/FP8/NVFP4 QLoRA**](https://docs.surogate.ai/guides/qlora) Support for a variety of QLoRA configurations, including online quantization (FP8, NVFP4, BnB) or loading pre-quantized weights (FP8, NVFP4)
- [**👌 Optimizers**](https://docs.surogate.ai/guides/optimizers): AdamW 8bit, !! NorMuon !!
- **🖥️ Runs on all NVIDIA GPUs**: sm80, sm86, sm89, sm90, sm100, sm103, sm120, sm121
- [**🧪 Mixed-precision training**](https://docs.surogate.ai/guides/precision-and-recipes#mixed-precision-training): Mix different dtypes for GEMMs, model, gradients and LoRA recipes to create your own flavor.
- **🛡️ Designed for reliability**: deterministic configs, explicit recipes, and a clear C++ core
- [**🧬 Adaptive Training**](https://docs.surogate.ai/about/adaptive-training): built-in automated training monitoring with automatic phase detection, multi-criteria early stopping (convergence, compute-efficiency, divergence, plateau), auto LR management, MoE imbalance detection, Chinchilla token budgeting and dynamic epoch adjustment
- [**🎨 Dedicated MoE Features**](https://docs.surogate.ai/guides/moe): Expert Parallelism, Least-Loaded EP load-balancing, MoE training metrics, Imbalance detection
- **🥞 Stacked LoRA training**: Train a LoRA adapter on top of another LoRA adapter to skip offline merging into base model.

---

## 🧠 Supported Models:
We support the following models. Please create a PR if you need a specific model

| Model              | Architecture                                            | Model Sizes                   |
| ------------------ | ------------------------------------------------------- | ----------------------------- |
| Qwen3              | Qwen3ForCausalLM                                        | 0.6B, 1.7B, 4B, 8B, 14B, 35B  |
| Qwen3VL            | Qwen3VLForConditionalGeneration                         | 2B, 4B, 8B, 32B               |
| Qwen3 MoE          | Qwen3MoeForCausalLM                                     | 30B-A3B, 235B-A22B            |
| Qwen3.5            | Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration     | 0.8B, 2B 4B, 9B, 27B          |
| Qwen3.5 Moe        | Qwen3MoeForCausalLM, Qwen3_5MoeForConditionalGeneration | 35B-A3B, 122B-A10B, 397B-A17B |
| Nemotron Nano v3   | NemotronHForCausalLM                                    | 30B-A3B                       |
| Nemotron Super v3  | NemotronHForCausalLM                                    | 120B-A12B                     |
| Nemotron Cascade 2 | NemotronHForCausalLM                                    | 30B-A3B                       |
| GPT-OSS            | GptOssForCausalLM                                       | 20B, 120B                     |
| Llama 3.1          | LlamaForCausalLM                                        | 8B, 70B, 405B                     |
| Llama 3.2          | LlamaForCausalLM                                        | 1B, 3B                      |


## 🚀 Quickstart
You can interact with the Surogate High-Performance Training Engine at the framework level via the CLI.

### Run the Surogate Training Engine:

#### Option A: Run using Docker (recommended)
Surogate provides 3 docker images for various CUDA versions. Currently only the `x86-64` architecture is supported.

| CUDA   | Image                                        | Recommended NVIDIA Driver | Minimum NVIDIA Driver |
| ------ | -------------------------------------------- | ------------------------- | --------------------- |
| 12.8.1 | `ghcr.io/invergent-ai/surogate:latest-cu128` | `>= 570.124.06`           | `>= 525`              |
| 12.9.1 | `ghcr.io/invergent-ai/surogate:latest-cu129` | `>= 575.57.08`            | `>= 525`              |
| 13.1   | `ghcr.io/invergent-ai/surogate:latest-cu130` | `>= 590.48.01`            | `>= 580`              |

```bash
docker run --gpus=all -v /my/local/config.yaml:/home/surogate/config.yaml -v /my/local/output_dir:<OUTPUT_DIR_FROM_CONFIG_YAML> <IMAGE> sft config.yaml
```

#### Option B: Install via script
```bash
curl -LsSf https://surogate.ai/install.sh | sh
```

#### Option C: Build from source (dev / contributors)
You need CUDA 12.8/12.9/13.x installed on your machine and NCCL development libraries libnccl-dev for your CUDA version

```bash
# ...clone repo...
uv pip install -e .
```

---

## Quickstart (SFT)

1) Create a config (example):

```yaml
model: Qwen/Qwen3-0.6B
output_dir: ./output

# training
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
sequence_len: 2048
learning_rate: 2e-4

# LoRA / QLoRA
lora: true
lora_rank: 16
# qlora_fp8: true  # optional, hardware-dependent
# qlora_fp4: true  # Blackwell+
# qlora_bnb: true  # Any GPU, lowest

datasets:
  - path: "mlabonne/FineTome-100k"
    type: auto
```

2) Run:
```bash
surogate sft config.yaml
```

3) Outputs:
- checkpoints, logs and artifacts are written under `output_dir`

---

## Hardware / Requirements

- NVIDIA GPU + recent driver
- CUDA **12.8, 12.9, 13**, NCCL, cuDNN
- Linux x86_64

### Supported NVIDIA GPUs:
- `SM80`: A100, A30
- `SM86`: A2, A16, A10, A40, RTX3050, RTX3060, RTX 3070, RTX 3080, RTX 3090, A2000, A3000, A4000, A5000, A6000
- `SM89`: L4, L40, L40S, RTX 4050, RTX 4060, RTX 4070, RTX 4080, RTX 4090, RTX 2000 Ada, RTX 4000 SFF Ada, RTX 4000 Ada, RTX 4500 Ada, RTX 5000 Ada, RTX 6000 Ada
- `SM90`: H100, H200, GH200
- `SM100`: B200, GB200
- `SM103`: B300, GB300
- `SM120`: RTX PRO 6000/5000/4000/2500/2000 Blackwell,  RTX 5050,  RTX 5060,  RTX 5070,  RTX 5080,  RTX 5090
- `SM121`: DGX Spark

---

## Documentation / Examples

- Docs: https://docs.surogate.ai
- Examples: https://github.com/invergent-ai/surogate/tree/master/examples

---

## Contributing

We welcome contributions across the entire ecosystem! If you are submitting a PR to the core framework, please ensure you include a clear description, steps to test locally, and relevant examples.

If you’re adding kernels/recipes or touching build/tooling, please keep changes minimal and include:
- a short description of the change,
- how to reproduce/validate locally (`make test` where applicable),
- and any GPU/arch assumptions.

---

## License

Apache 2.0 — see [LICENSE](./LICENSE).
