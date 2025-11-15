<div align="center">
<a href="https://surogate.ai/">
<img width="50" alt="surogate llmops framework" src="./docs/static/img/logo-white.svg" />
</a>

<div align="center">
<h1>Surogate LLMOps Framework</h1>
</div>

<div align="center">
    <a href="https://surogate.ai">Home Page</a> |
    <a href="https://docs.surogate.ai">Documentation</a> |
    <a href="https://github.com/invergent-ai/surogate/tree/master/examples">Examples</a> 
</div>

<br/>

<div align="center">
    
[![GitHub stars](https://img.shields.io/github/stars/invergent-ai/surogate?style=social)](https://github.com/invergent-ai/surogate)
[![GitHub issues](https://img.shields.io/github/issues/invergent-ai/surogate)](https://github.com/invergent-ai/surogate/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/invergent-ai/surogate)](https://github.com/invergent-ai/surogate/pulls)
[![Twitter Follow](https://img.shields.io/twitter/follow/invergentai?style=social)](https://twitter.com/invergentai)

</div>

<div align="center">
Do you like what we're doing? Give us a star ️⬆⭐
</div>
<br/>
</div>

Surogate is an end-to-end LLMOps framework that simplifies the development, deployment, and maintenance of Large Language Models (LLMs). It offers a complete toolkit and proven workflows for data processing, model training and fine-tuning, evaluation, quantization, and deployment — enabling efficient, reliable, and scalable LLM operations.

Surogate is built for teams that need fast experimentation scalability and predictable outcomes — whether running on-premise, in private clouds, or inside turnkey systems such as the DenseMAX Appliance.


## All-in-One LLMOps Platform
Everything required to build, adapt, deploy, and monitor generative AI systems:
- Model serving with KV-cache routing, GPU sharding, and high-throughput pipelines.
- Quantization and optimization for low-latency inference (4-bit, 8-bit, GPTQ, AWQ, etc.).
- Built-in model evaluation (MMLU, ARC, GSM8k, TruthfulQA, HellaSwag, etc.) and red-teaming tools. 
- Pre-training, Continued Pre-training
- LoRA, QLoRA, and full-fine-tuning support
- Reinforcement-learning workflows such as DPO, PPO, and GRPO for model alignment.
- Synthetic data generation and reward model training.
- Model distillation for smaller, faster variants.
- Data management and preprocessing tools for text, code, and multimodal datasets.
- Experiment tracking and model versioning with build-in Data Hub
- Seamless integration with Hugging Face
- Modular architecture for easy customization and extension.


## Why Surogate?
**Why build a LLMOps framework when you can piece together open-source tools?**: because building and maintaining reliable, scalable, and efficient LLM systems is **hard** — and the current open-source landscape is highly fragmented.

While there are many excellent community-driven tools, most are still early-stage — fragmented, inconsistently documented, hard to integrate, and rarely tested under real enterprise workloads. As a result, teams trying to build their own stack end up juggling separate components for inference, fine-tuning, evaluation, dataset management, and monitoring, all of which require significant engineering effort to stitch together and maintain.

Instead of juggling 5–10 separate open-source components, **Surogate** provides everything in a single, coherent platform — built to work together, tested under real workloads, and hardened for enterprise use. This dramatically accelerates the journey from prototype to production and ensures your AI systems remain stable, reproducible, and secure.

- **Fast Time-to-Value**: Removes complexity from deployment, training, and model lifecycle management. Teams can go from “idea” to “production-ready AI application” in days, not months.
- **Open & Extensible**: 
  - Developers can **extend** the platform with custom modules.
  - Organizations can **self-host, audit, and customize** every layer.
  - The community can **contribute connectors, runtimes, adapters, and integrations** — without being locked into proprietary APIs or closed cloud services.

Surogate brings coherence, reliability, and enterprise engineering to a landscape that has long been fragmented — while remaining open, transparent, and community-driven.


## Getting Started
Coming soon: installation guides, deployment configuration, developer docs, and API references.


# Installation

```bash
uv venv --python 3.12
sh requirements/raw-deps.sh
uv pip install -r requirements/torch29.txt
uv pip install -r requirements/build.txt
uv pip install -r requirements/common.txt
uv pip install -r requirements/cuda.txt

# install sgl-kernel

uv pip install "numpy==2.2.6"
rm -rf .venv/lib/python3.12/site-packages/triton_kernels
```

## Contributing
Contributions are welcome! Please open a PR or issue, and follow the contributing guidelines (to be published).

## License
Surogate is released under the Apache 2.0 License. See [LICENSE](./LICENSE) for details.

