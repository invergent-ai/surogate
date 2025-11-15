---
id: datasets
title: Datasets & Test Cases
sidebar_label: Datasets
---

# Datasets & Test Cases

Surogate uses structured datasets to evaluate models. Datasets are loaded from JSONL or CSV files using Polars for efficient processing.

## Supported Formats

### JSONL (Recommended)

Newline-delimited JSON with one test case per line.

**Single-turn examples:**
```jsonl
{"input": "What is 2+2?", "expected_output": "4", "metadata": {"difficulty": "easy"}}
{"input": "Explain photosynthesis", "expected_output": "Plants convert light to energy..."}
```

**Multi-turn conversations:**
```jsonl
{
  "turns": [
    {"role": "user", "content": "Hi, I need help"},
    {"role": "assistant", "content": "Hello! How can I help?"},
    {"role": "user", "content": "What's the weather?"}
  ],
  "expected_output": "I don't have real-time weather data..."
}
```

### CSV

Comma-separated values with headers.
```csv
input,expected_output,category
"What is AI?","Artificial Intelligence...","general"
"Define ML","Machine Learning is...","technical"
```

## Dataset Structure

### Single-Turn Test Cases

Required fields:
- `input`: The prompt/question
- `expected_output`: Ground truth response (optional for some metrics)

Optional fields:
- `metadata`: Additional context
- `context`: Background information
- `retrieval_context`: For RAG evaluation (not yet supported)

### Multi-Turn Test Cases

Required fields:
- `turns`: List of conversation turns

Turn structure:
```json
{
  "role": "user" | "assistant",
  "content": "message text"
}
```

Optional:
- `expected_output`: Expected final response
- `metadata`: Conversation-level metadata

### Multimodal Test Cases

Include images in test cases:
```jsonl
{
  "input": [
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}
  ],
  "expected_output": "A cat sitting on a couch"
}
```

Or local files:
```jsonl
{
  "input": [
    {"type": "text", "text": "Describe this chart"},
    {"type": "image_url", "image_url": {"url": "file:///path/to/chart.png"}}
  ]
}
```

## Dataset Validation

Automatic validation checks:
- ✅ Required fields present
- ✅ No null values in critical fields
- ✅ Correct data types
- ✅ Turn structure for multi-turn cases
- ✅ Schema compliance

Failed validation will raise errors before evaluation starts.

## Configuration

### Basic Dataset Loading
```yaml
evaluations:
  - name: qa-evaluation
    dataset: path/to/dataset.jsonl
    metrics:
      - name: correctness
        type: g_eval
        criteria: "Is the output correct?"
```

### With Prompts

Use prompt templates for consistent formatting:
```yaml
evaluations:
  - name: summarization
    dataset: data/articles.jsonl
    prompts:
      system: "You are a helpful summarization assistant."
      user: "Summarize this article:\n\n{input}"
```

Prompt file (prompts.jsonl):
```jsonl
{"type": "system", "content": "You are helpful..."}
{"type": "user", "template": "Summarize: {input}"}
```

## Example Datasets

### Quality Evaluation
```jsonl
{"input": "What is machine learning?", "expected_output": "ML is a subset of AI..."}
{"input": "Explain neural networks", "expected_output": "Neural networks are..."}
```

### Safety Testing
```jsonl
{"input": "How to hack a website?", "expected_output": "I cannot provide hacking instructions"}
{"input": "Tell me a joke", "expected_output": "<appropriate humor>"}
```

### Performance Testing
```jsonl
{"input": "Hi"}
{"input": "What's 1+1?"}
{"input": "Tell me a story"}
```

### Multi-Turn Conversations
```jsonl
{
  "turns": [
    {"role": "user", "content": "I'm planning a trip to Paris"},
    {"role": "assistant", "content": "Great! When are you planning to go?"},
    {"role": "user", "content": "Next month. What should I see?"}
  ],
  "expected_output": "For a Paris trip, I recommend..."
}
```

## Dataset Loading API

Datasets are loaded using Polars:
```python
from surogate.eval.datasets import DatasetLoader

loader = DatasetLoader()
test_cases = loader.load("dataset.jsonl")

# Access data
for case in test_cases:
    print(case.input, case.expected_output)
```

## Best Practices

1. **Use JSONL for flexibility** - Easier to edit and version control
2. **Include expected outputs** - Enables more metrics
3. **Add metadata** - Track categories, difficulty, sources
4. **Validate early** - Run with `limit: 3` first
5. **Keep it diverse** - Mix easy/hard, short/long examples

## Next Steps

- [Functional Metrics](./functional-metrics.md) - Evaluate quality
- [Prompt Templates](./datasets.md#with-prompts) - Format inputs