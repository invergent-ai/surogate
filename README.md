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
[![Twitter Follow](https://img.shields.io/twitter/follow/invergent-ai?style=social)](https://twitter.com/invergent-ai)

</div>

<div align="center">
⭐ Like what we're doing? Give us a star ⬆️
</div>
<br/>
</div>

Surogate is a comprehensive LLMOps framework designed to streamline the development, deployment, and management of Large Language
Models (LLMs). It provides all the necessary tools and best practices for data preprocessing, model training, evaluation, quantization, deployment and monitoring,
ensuring efficient and effective LLM operations.

# Integrations

- https://github.com/datajuicer/data-juicer?tab=readme-ov-file
- https://github.com/modelscope/ms-swift
- https://github.com/modelscope/evalscope -> bencharmking
- https://github.com/IST-DASLab/llmq
- https://github.com/axolotl-ai-cloud/axolotl
- https://github.com/NVIDIA/Megatron-LM
- https://github.com/skypilot-org/skypilot/
- https://github.com/aimhubio/aim
- https://github.com/sgl-project/sglang
- https://github.com/vllm-project/vllm
- https://github.com/confident-ai/deepeval ->
- https://github.com/confident-ai/deepteam
- https://www.jackyoustra.com/blog/road-to-petaflop
- 
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

# Run

```
uv run surogate
```




