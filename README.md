<div align="center" style="padding: 2rem">
<p align="center">
  <a href="https://surogate.ai/">
    <img alt="Surogate" width="40%"
      src="https://raw.githubusercontent.com/invergent-ai/surogate/main/assets/logo-white.svg#gh-dark-mode-only" />
    <img alt="Surogate" width="40%"
      src="https://raw.githubusercontent.com/invergent-ai/surogate/main/assets/logo-black.svg#gh-light-mode-only" />
  </a>
</p>

<h3>High-performance, mixed-precision LLM pre-training & fine-tuning <br/> (C++/CUDA core, Python wrapper, BF16, FP8, NF4, NVFP4)</h3>
<div>
<a href="https://surogate.ai">Home</a> ·
<a href="https://docs.surogate.ai">Docs</a> ·
<a href="https://github.com/invergent-ai/surogate/tree/master/examples">Examples</a> ·
<a href="https://docs.surogate.ai/reference/benchmarks">Benchmarks</a> ·
<a href="https://github.com/invergent-ai/surogate-studio">Studio</a>
</div>
<br/>
  
[![GitHub stars](https://img.shields.io/github/stars/invergent-ai/surogate?style=social)](https://github.com/invergent-ai/surogate)
[![GitHub issues](https://img.shields.io/github/issues/invergent-ai/surogate)](https://github.com/invergent-ai/surogate/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/invergent-ai/surogate)](https://github.com/invergent-ai/surogate/pulls)
[![Twitter Follow](https://img.shields.io/twitter/follow/surogate_ai?style=social)](https://x.com/surogate_ai)

</div>

## What is Surogate?

Surogate is a **production-grade LLM training framework** and **LLMOps platform** engineered to operate at practical hardware limits, delivering near–speed-of-light throughput, low-latency execution, and multi-GPU/multi-Node training scaling at scale.

By combining a native **C++/CUDA execution engine**, a low-overhead Python frontend, and a highly optimized **multi-threaded scheduler**, Surogate achieves industry-leading Speed-Of-Light (SOL) utilization on NVIDIA GPUs — **outperforming existing training toolkits by a wide margin**. 

See reproducible comparisons in [Benchmarks](https://docs.surogate.ai/reference/benchmarks).


[Surogate Studio](https://github.com/invergent-ai/surogate-studio) is the no-code companion of Suroagate for enterprise-grade LLMOps.

![alt text](assets/studio.png)

---

## ✨ Highlights
Surogate is built for developers and enterprises that need fast experimentation scalability and predictable outcomes — whether running on-premise, in private clouds, or inside turnkey systems such as the [DenseMAX Appliance](https://www.invergent.ai/densemax-appliance).

- **🔧 Pre-training + Fine-tuning**: full fine-tuning, LoRA/QLoRA
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
- **🧠 Supported models**: Qwen2.5, Qwen3, Qwen3 MoE, Llama 3+, Nemotron Nano. Models can be added easily, please create a PR if you need a specific model.
- **[Surogate Studio](https://github.com/invergent-ai/surogate-studio)**: Unified no-code platform for managing your AI infrastructure and operations: Training, Fine-Tuning, Inference and Quantization
  
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

## Install

### Option A: Install via script (recommended)
```bash
curl -LsSf https://surogate.ai/install.sh | sh
```

### Option B: Build from source (dev / contributors)
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

## Documentation / Examples

- Docs: https://docs.surogate.ai
- Examples: https://github.com/invergent-ai/surogate/tree/master/examples

---

Harness the full power of **Surogate** with **[Surogate Studio](https://github.com/invergent-ai/surogate-studio)** — a fully visual, easy-to-use platform for fine-tuning, deployment, evaluation, and production LLMOps.

---

## Contributing

PRs and issues are welcome.
If you’re adding kernels/recipes or touching build/tooling, please keep changes minimal and include:
- a short description of the change,
- how to reproduce/validate locally (`make test` where applicable),
- and any GPU/arch assumptions.

---

## License

Apache 2.0 — see [LICENSE](./LICENSE).
