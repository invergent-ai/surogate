---
id: overview
title: VLM/LLM Evaluation Overview
sidebar_label: Overview
---

# VLM/LLM Evaluation

Surogate provides a unified evaluation framework for Vision-Language Models (VLMs) and Large Language Models (LLMs). This system combines model evaluation, performance benchmarking, and safety testing in a single config-driven tool.

## Key Features

- **Multi-Model Support**: Evaluate LLMs, multimodal models, embeddings, CLIP, and rerankers
- **Flexible Providers**: OpenAI, Anthropic, Cohere, OpenRouter, local models (vLLM, transformers)
- **Comprehensive Metrics**: Quality, safety, performance, and conversation metrics
- **Standard Benchmarks**: 40+ academic benchmarks (MMLU, HellaSwag, GSM8K, HumanEval, etc.)
- **Stress Testing**: Find breaking points and measure throughput/latency
- **Config-Driven**: Define everything in YAML/JSON configuration files

## Quick Start
```bash
# Run evaluation
surogate eval --config path/to/config.yaml

# View results
surogate eval --list
surogate eval --view <run_id>
```

## Architecture
```
Config File → Target Definition → Evaluation/Benchmarks → Results
     ↓              ↓                    ↓                    ↓
  YAML/JSON    API/Local Model    Metrics/Datasets      JSON/Markdown
```

### Core Components

1. **Targets**: Model endpoints (API or local)
2. **Datasets**: Test cases in JSONL/CSV format
3. **Metrics**: Quality, safety, performance evaluators
4. **Benchmarks**: Standard academic/industry benchmarks
5. **Results**: Structured output with detailed analytics

## What You Can Evaluate

| Model Type | Examples | Use Cases |
|------------|----------|-----------|
| **LLMs** | GPT-4, Claude, Llama | Text generation, reasoning, coding |
| **Multimodal** | GPT-4V, Qwen-VL, Claude-Vision | Image understanding, VQA |
| **Embeddings** | text-embedding-3, BGE | Semantic search, similarity |
| **CLIP** | OpenAI CLIP | Image-text matching |
| **Rerankers** | Cohere rerank | Search result ranking |

## Evaluation Types

### 1. Functional Evaluation
Assess output quality using:
- **G-Eval**: LLM-as-judge for custom criteria
- **Multi-turn**: Conversation coherence and context retention
- **Safety**: Toxicity, bias, harm detection

### 2. Performance Testing
Measure speed and scalability:
- **Latency**: Response time metrics
- **Throughput**: Requests per second
- **Stress Testing**: Breaking point detection

### 3. Standard Benchmarks
Run industry-standard tests:
- Academic: MMLU, BBH, ARC
- Reasoning: GSM8K, MathQA
- Coding: HumanEval, IFEval
- Vision: MMMU, ChartQA

## Next Steps

- [Model Support](./model-support.md) - Configure your models
- [Datasets](./datasets.md) - Prepare test data
- [Functional Metrics](./functional-metrics.md) - Quality evaluation
- [Configuration Guide](../getting-started/configuration.md) - Full config reference