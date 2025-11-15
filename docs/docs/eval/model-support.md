---
id: model-support
title: Model Support
sidebar_label: Model Support
---

# Model Support

Surogate supports multiple model types and providers through a unified target configuration system.

## Supported Model Types

### 1. Large Language Models (LLMs)

Standard text generation models.
```yaml
targets:
  - name: gpt4-target
    type: llm
    provider: openai
    model: gpt-4-turbo-preview
    api_key: ${OPENAI_API_KEY}
```

### 2. Multimodal Models

Vision-language models that process images and text.
```yaml
targets:
  - name: qwen-vl
    type: multimodal
    provider: openai
    model: qwen/qwen-2-vl-7b-instruct
    base_url: https://openrouter.ai/api/v1
    api_key: ${OPENROUTER_API_KEY}
```

### 3. Embedding Models

Generate semantic embeddings for text.
```yaml
targets:
  - name: openai-embeddings
    type: embedding
    provider: openai
    model: text-embedding-3-small
    api_key: ${OPENAI_API_KEY}
```

### 4. CLIP Models

Image-text matching models.
```yaml
targets:
  - name: clip-model
    type: clip
    provider: local
    model: openai/clip-vit-base-patch32
```

### 5. Reranker Models

Re-rank search results or passages.
```yaml
targets:
  - name: cohere-reranker
    type: reranker
    provider: cohere
    model: rerank-english-v3.0
    api_key: ${COHERE_API_KEY}
```

## Supported Providers

### OpenAI (and OpenAI-compatible APIs)

Supports OpenAI, OpenRouter, Together AI, Anyscale, and custom proxies.
```yaml
- name: openrouter-claude
  type: llm
  provider: openai  # OpenRouter is OpenAI-compatible
  model: anthropic/claude-3.5-sonnet
  base_url: https://openrouter.ai/api/v1
  api_key: ${OPENROUTER_API_KEY}
```

### Anthropic

Native Claude API support.
```yaml
- name: claude-direct
  type: llm
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  api_key: ${ANTHROPIC_API_KEY}
```

### Cohere

Embeddings and rerankers.
```yaml
- name: cohere-embed
  type: embedding
  provider: cohere
  model: embed-english-v3.0
  api_key: ${COHERE_API_KEY}
```

### Local Models

Run models locally using vLLM, transformers, or sentence-transformers.

#### vLLM (High Performance)
```yaml
- name: local-vllm
  type: llm
  provider: openai  # vLLM uses OpenAI-compatible API
  model: Qwen/Qwen3-0.6B
  base_url: http://localhost:8888/v1
  # No API key needed
```

Start vLLM server:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B \
  --port 8888 \
  --trust-remote-code
```

#### Transformers (Direct Loading)
```yaml
- name: local-transformers
  type: llm
  provider: local
  model: meta-llama/Llama-3.2-1B-Instruct
  device: cuda  # or cpu
```

## Environment Variables

Use `${VAR}` syntax for sensitive credentials:
```yaml
targets:
  - name: my-model
    api_key: ${OPENAI_API_KEY}  # Reads from environment
    base_url: ${CUSTOM_BASE_URL}  # Optional override
```

Set in your shell:
```bash
export OPENAI_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-or-..."
```

## Health Checks

Targets automatically perform health checks:
- **API targets**: Verify credentials work
- **Local targets**: Confirm model is loaded

Failed health checks will be logged but won't stop evaluation.

## Resource Cleanup

Targets use context managers for automatic cleanup:
```python
with target:
    # Model stays loaded
    response = target.generate(request)
# Automatic cleanup here
```

## Next Steps

- [Datasets](./datasets.md) - Prepare test data for your models
- [Configuration](../getting-started/configuration.md) - Full config reference