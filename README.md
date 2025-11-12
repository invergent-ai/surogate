# LLMOps Framework

LLMOps is a comprehensive framework designed to streamline the development, deployment, and management of Large Language
Models (LLMs). It provides tools and best practices for data preprocessing, model training, evaluation, and monitoring,
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




