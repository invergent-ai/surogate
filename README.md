<div align="center">
<a href="https://surogate.ai/">
<img width="50" alt="surogate llmops framework" src="./docs/static/img/logo-white.svg" />
</a>

<div align="center">
<h1>Surogate</h1>
<h3>The Enterprise LLMOps Framework</h3>
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

Surogate is an end-to-end Enterprise LLMOps framework that simplifies the development, deployment, and maintenance of organization-specific Large Language Models (LLMs). It offers a complete toolkit and proven workflows for data processing, model training and fine-tuning, evaluation, quantization, and deployment — enabling efficient, reliable, and scalable LLM operations tailored to enterprise needs.

Surogate is built for enterprises that need fast experimentation scalability and predictable outcomes — whether running on-premise, in private clouds, or inside turnkey systems such as the DenseMAX Appliance.


## Getting Started
Coming soon: installation guides, deployment configuration, developer docs, and API references.


# Installation

```bash
uv venv --python 3.12
sh requirements/raw-deps.sh
uv pip install -r requirements/torch29.txt
uv pip install -r requirements/build.txt
uv pip install -r requirements/common.txt
MAX_JOBS=8 uv pip install -r requirements/cuda.txt
uv pip install "nvidia-nccl-cu12==2.28.3"

# build sgl-kernel
# build lmcache

uv pip install "numpy==2.2.6"
rm -rf .venv/lib/python3.12/site-packages/triton_kernels
```

## Contributing
Contributions are welcome! Please open a PR or issue, and follow the contributing guidelines (to be published).

## License
Surogate is released under the Apache 2.0 License. See [LICENSE](./LICENSE) for details.

