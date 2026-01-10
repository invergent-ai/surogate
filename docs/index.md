# Surogate
<div align="center">
<a href="https://surogate.ai/">
<img width="120" alt="Surogate logo" src="../assets/logo.jpg" />
</a>
<h1>Surogate</h1>
<h3>High-performance, mixed-precision LLM pre-training & fine-tuning <br/> (C++/CUDA core, Python wrapper, BF16, FP8, NF4, NVFP4)</h3>
<a href="https://surogate.ai">Home</a> ·
<a href="https://github.com/invergent-ai/surogate/tree/master/examples">Examples</a> ·
<a href="./benchmarks/speed.md">Benchmarks</a>
<br/><br/>
<b>If Surogate saves you time or GPUs, consider ⭐ starring ⭐ the repo.</b>
<br/><br/>
<iframe src="https://ghbtns.com/github-btn.html?user=twbs&repo=bootstrap&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="GitHub"></iframe>
</div>

## What is Surogate?

Surogate is a **production-grade LLM training framework** engineered to operate at practical hardware limits, delivering near–speed-of-light throughput, low-latency execution, and predictable multi-GPU scaling at scale.

By combining a native **C++/CUDA execution engine**, a low-overhead Python frontend, and a highly optimized **multi-threaded scheduler**, Surogate achieves industry-leading Speed-Of-Light (SOL) utilization on NVIDIA GPUs — **outperforming existing training toolkits by a wide margin**. 


## ✨ Highlights
Surogate is built for developers and enterprises that need fast experimentation scalability and predictable outcomes — whether running on-premise, in private clouds, or inside turnkey systems such as the [DenseMAX Appliance](https://www.invergent.ai/densemax-appliance).

- **🔧 Pre-training + Fine-tuning**: full fine-tuning, LoRA/QLoRA
- [**🖥️...🖥️ Native multi-GPU**](./guides/multi-gpu.md) training with multi-threading and MPI backends
- **⚡ Native C++/CUDA engine** for near–Speed-Of-Light (SOL) throughput
- [**🗲 CUDA Kernel Fusions**](./guides/transformer.md#kernel-fusions) for maximum throughput  
- [**⚖️ Smart CPU Offloading**](./guides/offloading.md) for weights, gradients, activations, quants
- **📜 Pre-built training recipes**: 
  - [**💎 BF16**](./guides/precision-and-recipes.md#bf16): Baseline recipe using `bfloat16` for all GEMMs, designed for maximum numerical accuracy. No quantization is applied.
  - [**🔥 FP8**](./guides/precision-and-recipes.md#fp8-hybrid): Native `FP8` training delivering extreme performance with `E4M3` used for activations and weights and `E5M2` for gradients. Uses per-tensor delayed scaling to provide stable training.
  - [**🔥 NVFP4**](./guides/precision-and-recipes.md#fp4-nvfp4): Native CUTLASS `FP4 E2M1` training with two-level block scaling for extreme performance and memory efficiency on Blackwell GPUs (**SM100+**: B200, B300, RTX 50xx series). Uses stochastic rounding and random Hadamard Transforms for numerical stability. **Supports NVIDIA B200, B300, RTX 5070, 5080, 5090 !!**
- [**⚡ BnB/FP8/NVFP4 QLoRA**](./guides/qlora.md) to maximize SOL on Hopper/Blackwell GPUs
- [**👌 Optimizers**](./guides/optimizers.md): AdamW 8bit, !! NorMuon !!
- **🖥️ Runs on all NVIDIA GPUs**: sm80, sm86, sm89, sm90, sm100, sm103, sm120, sm121
- [**🧪 Mixed-precision training**](./guides/precision-and-recipes.md#mixed-precision-training): Mix different dtypes for GEMMs, model, gradients and LoRA recipes to create your own flavor.
- **🛡️ Designed for reliability**: deterministic configs, explicit recipes, and a clear C++ core
- **🧠 Supported models**: Qwen2.5, Qwen3 Dense, LLama 3.2, more to come shortly


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
  

## Quickstart
Install Surogate using the following command on a machine with CUDA 12 or 13:

```bash
curl -LsSf https://surogate.ai/install.sh | sh
```

Run your first pretraining/fine-tuning:

- [Quickstart: SFT](getting-started/quickstart-sft.md)
- [Quickstart: Pretraining](getting-started/quickstart-pretraining.md)

## Guides
- [Architecture](guides/transformer.md)
- [Configuration](guides/configuration.md)
- [Datasets](guides/datasets.md)
- [Precision & recipes](guides/precision-and-recipes.md)
- [QLoRA](guides/qlora.md)
- [Optimizers](guides/optimizers.md)
- [Memory](guides/memory.md)
- [Offloading](guides/offloading.md)
- [Multi-GPU](guides/multi-gpu.md)
- [Multi-Node](guides/multi-node.md)
- [Performance](guides/performance.md)
- [Debugging](guides/debugging.md)

## Reference

- [Config reference](reference/config.md)
- [CLI reference](reference/cli.md)
- [Python API](reference/python-api.md)

## Appendix

- [FAQ](appendix/faq.md)
- [Glossary](appendix/glossary.md)
- [Compatibility](appendix/compatibility.md)
- [Release notes](appendix/release-notes.md)
