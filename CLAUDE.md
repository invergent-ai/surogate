# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Surogate is an Enterprise LLMOps Framework for developing, deploying, and maintaining organization-specific Large Language Models. It provides an integrated toolkit for data processing, model training, fine-tuning, evaluation, quantization, and deployment.

**Key differentiator**: Unlike fragmented open-source tools, Surogate offers an end-to-end solution with enterprise-grade security, compliance, and on-premise deployment capabilities.

## Installation

The installation uses `uv` package manager with Python 3.12:

```bash
uv venv --python 3.12
sh requirements/raw-deps.sh
uv pip install -r requirements/torch29.txt
uv pip install -r requirements/build.txt
uv pip install -r requirements/common.txt
MAX_JOBS=8 uv pip install -r requirements/cuda.txt
uv pip install "numpy==2.2.6"
rm -rf .venv/lib/python3.12/site-packages/triton_kernels
```

**Important**:
- Installation requires CUDA-enabled GPU
- `raw-deps.sh` installs special dependencies (vLLM, SGLang, ms-swift, etc.) without dependency resolution
- The framework requires Linux platform only

## Commands

All commands use the `surogate` CLI with the following structure:

```bash
surogate <command> --config <config.yaml> [options]
```

### Core Commands

1. **Supervised Fine-Tuning (SFT)**:
   ```bash
   surogate sft --config examples/sft/qwen3-0.6b-simple.yaml
   ```

2. **Post-Training Quantization (PTQ)**:
   ```bash
   surogate ptq --config examples/ptq/awq.yaml
   ```
   Supported schemes: `fp8`, `awq`, `gptq_int4`, `gptq_int8`, `nvfp4`

3. **Model Serving**:
   ```bash
   surogate serve --config examples/serve/vllm.yaml
   ```

4. **Evaluation**:
   ```bash
   # Run evaluation
   surogate eval --config examples/eval/config.yaml

   # View results
   surogate eval --list                    # List all results
   surogate eval --view <filename>         # View specific result
   surogate eval --compare <file1> <file2> # Compare two results
   ```

### Distributed Training

Commands automatically use `torchrun` when distributed environment variables are set:

```bash
export NPROC_PER_NODE=4      # Number of GPUs per node
export NNODES=1              # Number of nodes
export MASTER_PORT=29500     # Master port
export MASTER_ADDR=localhost # Master address
export NODE_RANK=0           # Current node rank

# The CLI automatically wraps commands with torchrun
surogate sft --config config.yaml
# Becomes: python -m torch.distributed.run --nproc_per_node 4 ... sft.py --config config.yaml
```

### Testing

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_datasets.py
```

## Architecture

### High-Level Structure

The framework follows a command-based architecture where each major operation (SFT, PTQ, Serve, Eval) is implemented as a `SurogateCommand` subclass:

```
surogate/
├── cli/           # CLI entry points and argument parsing
├── config/        # Configuration schemas (Pydantic dataclasses)
├── datasets/      # Dataset loading and preprocessing
├── eval/          # Evaluation engine with metrics, benchmarks, security testing
├── loaders/       # Model and tokenizer loading
├── ptq/           # Post-training quantization using llmcompressor
├── serve/         # FastAPI-based model serving
├── sft/           # Supervised fine-tuning with LoRA/adapters
└── utils/         # Shared utilities (logging, distributed, commands)
```

All commands are launched via the `surogate` CLI, which handles argument parsing, distributed environment detection, and subprocess execution.
Some commands use `torchrun` for distributed training.
Some commands use external libraries (e.g., ms-swift, llmcompressor) for specialized functionality.

### Command Execution Flow

1. **CLI Entry** (`surogate/cli/main.py`):
   - Parses command-line arguments
   - Detects distributed environment variables
   - Launches commands as subprocesses (optionally via `torchrun`)
   - Commands run in separate processes to integrate with `torchrun`/`accelerate`

2. **Command Classes** (inherit from `SurogateCommand`):
   - `SurogateSFT` - Fine-tuning with LoRA adapters
   - `SurogatePTQ` - Quantization with llmcompressor
   - `SurogateServe` - Model serving with FastAPI
   - `SurogateEval` (in eval.py) - Comprehensive evaluation

3. **Configuration System**:
   - All configs are Pydantic dataclasses in `surogate/config/`
   - YAML files are loaded and validated at runtime
   - Example configs in `examples/` directory

### Key Architectural Patterns

**Dataset Loading**:
- Unified dataset loader in `surogate/datasets/datasets.py`
- Supports: instruction-tuning, conversation, text, and custom datasets
- Lazy loading with streaming support for large datasets
- Tokenization strategies in `surogate/datasets/tokenization.py`

**Model Loading**:
- Centralized in `surogate/loaders/loader.py`
- Handles model registry lookup, downloading, and initialization
- Supports multimodal models via ms-swift integration
- Manages quantization configs and device placement

**Fine-Tuning (SFT)**:
- Uses ms-swift for LoRA adapters and templates
- Custom trainer: `SurogateSeq2SeqTrainer` extends HuggingFace Trainer
- Supports DeepSpeed for distributed training
- Callbacks: `TrainAdapterCallback`, `EarlyStopCallback`
- Padding-free training for efficiency

**Quantization (PTQ)**:
- Uses llmcompressor for one-shot quantization
- Supports FP8, INT4, INT8, NVFP4 schemes
- Calibration datasets loaded from config
- Saves compressed models

**Serving**:
- Two backends: PyTorch (PtEngine) and vLLM/SGLang (InferEngine)
- OpenAI-compatible API via FastAPI
- Adapter switching for multi-LoRA serving
- Endpoints: `/v1/chat/completions`, `/v1/models`, `/health`, `/stats`
- LMCache integration for caching

**Evaluation**:
- Multi-faceted evaluation framework in `surogate/eval/`:
  - `benchmarks/`: Academic benchmarks (MMLU, GSM8K, HellaSwag, etc.)
  - `metrics/`: Quality metrics (G-Eval, DAG metrics, classification, etc.)
  - `security/`: Red teaming and guardrails testing
  - `backend/`: API client backends (OpenAI, Anthropic, local models)
- Supports stress testing with resource monitoring
- LLM-as-judge evaluation for quality metrics
- Results saved as JSON + Markdown reports

### Configuration Files

Configuration is YAML-based with the following structure:

**SFT Config** (`examples/sft/qwen3-0.6b-simple.yaml`):
```yaml
model: Qwen/Qwen3-0.6B
model_type: qwen3_thinking
save_path: ./output
num_train_epochs: 1
sequence_len: 2048
deepspeed: zero3_offload
datasets:
  - path: "invergent/alpaca-gpt4-data-en"
    samples: 500
    type: instruction
```

**PTQ Config** (`examples/ptq/awq.yaml`):
```yaml
model: Qwen/Qwen3-0.6B
scheme: awq  # fp8, awq, gptq_int4, gptq_int8, nvfp4
save_path: output/Qwen3-0.6B-AWQ
sequence_len: 2048
datasets:
  - path: "wikimedia/wikipedia"
    samples: 512
    type: text
```

**Serve Config** (`examples/serve/vllm.yaml`):
```yaml
model: invergent/Qwen3-0.6B-NVFP4
infer_backend: vllm
max_context: 2048
tensor_parallel: 1
cache:
  enabled: 1
  max_memory_cache_gb: 1
```

**Eval Config**: See `examples/eval/config.yaml` for comprehensive example with:
- Multiple targets (OpenRouter, local vLLM)
- Quality metrics (correctness, coherence, relevance)
- Safety metrics (toxicity, bias, harm)
- Performance metrics (latency, throughput, token speed)
- Benchmarks (MMLU, GSM8K, etc.)
- Red teaming and guardrails testing

## Development Notes

### Environment Variables

**Build-time**:
- `SUROGATE_TARGET_DEVICE`: Target device (`cuda`, `cpu`, `empty`)
- `MAX_JOBS`: Max parallel build jobs (default: 8)
- `SUROGATE_MAIN_CUDA_VERSION`: CUDA version (default: 12.9)

**Runtime (Distributed Training)**:
- `NPROC_PER_NODE`: GPUs per node
- `NNODES`: Number of nodes
- `MASTER_ADDR`: Master node address
- `MASTER_PORT`: Master port
- `NODE_RANK`: Current node rank

**Evaluation**:
- `OPENAI_API_KEY`: OpenAI API key
- `OPENROUTER_API_KEY`: OpenRouter API key
- `ANTHROPIC_API_KEY`: Anthropic API key

### Extending the Framework

**Adding a new dataset type**:
1. Add dataset class in `surogate/datasets/`
2. Register in `load_datasets()` function
3. Update `DatasetConfig` schema in `surogate/config/dataset_config.py`

**Adding a new evaluation metric**:
1. Create metric class in `surogate/eval/metrics/`
2. Register in metric factory
3. Update `EvalConfig` schema

**Adding a new quantization scheme**:
1. Add scheme to `SUPPORTED_SCHEMES` in `surogate/ptq/ptq.py`
2. Implement recipe using llmcompressor modifiers
3. Update documentation

### DeepSpeed Configuration

The framework supports DeepSpeed ZeRO optimization:
- `zero3_offload`: ZeRO-3 with CPU/NVMe offloading (in `surogate/sft/deepspeed.py`)
- Configured via `deepspeed` field in SFT config
- Automatically handles gradient accumulation and optimizer states

### Testing Strategy

- Tests located in `tests/` directory
- Currently has dataset loading tests
- Run with `pytest tests/`
- Framework uses subprocess execution, so integration tests should account for this
