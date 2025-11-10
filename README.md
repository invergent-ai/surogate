# LLMOps Framework

LLMOps is a comprehensive framework designed to streamline the development, deployment, and management of Large Language
Models (LLMs). It provides tools and best practices for data preprocessing, model training, evaluation, and monitoring,
ensuring efficient and effective LLM operations.

# Integrations

- https://github.com/datajuicer/data-juicer?tab=readme-ov-file
- https://github.com/modelscope/ms-swift
- https://github.com/modelscope/evalscope
- https://github.com/IST-DASLab/llmq
- https://github.com/axolotl-ai-cloud/axolotl
- https://github.com/NVIDIA/Megatron-LM
- https://github.com/skypilot-org/skypilot/
- https://github.com/aimhubio/aim
- https://github.com/sgl-project/sglang
- https://github.com/vllm-project/vllm
- https://github.com/confident-ai/deepeval
- https://github.com/confident-ai/deepteam

# Installation

```bash
uv venv --python 3.12
uv pip install -r requirements-torch.txt
uv pip install -r requirements.txt
bash install-vllm.sh
bash install-sglang.sh
bash install-fa.sh
```

# Run

```
python -m surogate.cli.main
```
