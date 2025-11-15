---
id: standard-benchmarks
title: Standard Benchmarks
sidebar_label: Standard Benchmarks
---

# Standard Benchmarks

Run industry-standard academic benchmarks to compare models objectively. Surogate supports 40+ benchmarks through integrated backends.

## Quick Start
```yaml
targets:
  - name: my-model
    type: llm
    provider: openai
    model: gpt-4
    api_key: ${OPENAI_API_KEY}
    
    evaluations:
      - name: benchmark-suite
        benchmarks:
          - name: mmlu
            num_fewshot: 5
            limit: 100  # Limit to 100 examples for testing
          
          - name: gsm8k
            num_fewshot: 5
            limit: 50
```

## Supported Benchmarks

### Academic Benchmarks

General knowledge and reasoning.
```yaml
benchmarks:
  # MMLU - Multitask Language Understanding
  - name: mmlu
    num_fewshot: 5
    tasks: ["STEM", "Humanities"]  # Optional: specific subjects
    limit: 100
  
  # HellaSwag - Commonsense reasoning
  - name: hellaswag
    num_fewshot: 10
    limit: 200
  
  # BIG-Bench Hard
  - name: bbh
    num_fewshot: 3
    tasks: ["Logical Reasoning", "Mathematical Reasoning"]
    limit: 50
  
  # TruthfulQA - Truthfulness evaluation
  - name: truthfulqa
    num_fewshot: 0
    limit: 100
  
  # ARC - AI2 Reasoning Challenge
  - name: arc_challenge
    num_fewshot: 25
    limit: 100
  
  - name: arc_easy
    num_fewshot: 25
    limit: 100
  
  # Winogrande - Commonsense reasoning
  - name: winogrande
    num_fewshot: 5
    limit: 100
  
  # LAMBADA - Next word prediction
  - name: lambada
    num_fewshot: 0
    limit: 100
```

### Reading Comprehension
```yaml
benchmarks:
  # SQuAD - Stanford Question Answering
  - name: squad
    num_fewshot: 0
    limit: 100
  
  # DROP - Discrete Reasoning Over Paragraphs
  - name: drop
    num_fewshot: 3
    limit: 50
  
  # BoolQ - Boolean Questions
  - name: boolq
    num_fewshot: 0
    limit: 100
```

### Reasoning Benchmarks

Math and logical reasoning.
```yaml
benchmarks:
  # GSM8K - Grade School Math
  - name: gsm8k
    num_fewshot: 5
    limit: 100
    backend_params:
      max_tokens: 512  # Math needs more tokens
  
  # MathQA
  - name: mathqa
    num_fewshot: 4
    limit: 50
  
  # LogiQA - Logical reasoning
  - name: logiqa
    num_fewshot: 0
    limit: 50
```

### Coding Benchmarks
```yaml
benchmarks:
  # HumanEval - Code generation
  - name: humaneval
    num_fewshot: 0
    limit: 50
    backend_params:
      max_tokens: 1024  # Code needs more tokens
      k: 1  # Number of samples per problem
  
  # IFEval - Instruction following
  - name: ifeval
    num_fewshot: 0
    limit: 100
```

### Specialized Benchmarks
```yaml
benchmarks:
  # BBQ - Bias Benchmark for QA
  - name: bbq
    num_fewshot: 3
    limit: 100
  
  # CMMLU - Chinese MMLU
  - name: cmmlu
    num_fewshot: 5
    tasks: ["STEM"]
    limit: 50
  
  # C-Eval - Chinese Evaluation
  - name: ceval
    num_fewshot: 5
    limit: 50
```

## Configuration Options

### Basic Options
```yaml
- name: mmlu
  num_fewshot: 5     # Number of few-shot examples
  limit: 100          # Limit number of test cases (optional)
```

### Task/Subject Filtering
```yaml
- name: mmlu
  num_fewshot: 5
  tasks:              # Specific subjects only
    - "STEM"
    - "Humanities"
    - "Social Sciences"
  limit: 200
```

### Subset Filtering
```yaml
- name: bbh
  num_fewshot: 3
  subset:
    - "logical_deduction_three_objects"
    - "causal_judgement"
  limit: 50
```

### Backend Parameters

Control generation settings:
```yaml
- name: gsm8k
  num_fewshot: 5
  backend_params:
    max_tokens: 512      # Maximum tokens to generate
    temperature: 0.0     # Deterministic generation
    top_p: 1.0
    frequency_penalty: 0
```

### Judge Models

Some benchmarks use LLM judges for evaluation:
```yaml
- name: mmlu
  num_fewshot: 5
  judge_model:
    target: gpt4-judge  # Must be defined in targets
```

**When judges are used:**
- Multiple choice questions (MMLU, HellaSwag, etc.)
- Open-ended generation requiring validation

**When judges are NOT needed:**
- Code execution (HumanEval - runs tests)
- Exact match scoring (SQuAD, DROP)
- Mathematical extraction (GSM8K - regex-based)

## Custom Datasets

Use your own benchmark data:
```yaml
- name: custom_benchmark
  dataset: path/to/custom_data.jsonl
  num_fewshot: 3
  limit: 100
```

Dataset format:
```jsonl
{"question": "What is 2+2?", "answer": "4", "category": "math"}
{"question": "Capital of France?", "answer": "Paris", "category": "geography"}
```

## Concurrent Evaluation

Speed up benchmarks with parallel execution:
```yaml
targets:
  - name: my-model
    type: llm
    provider: openai
    model: gpt-4
    api_key: ${OPENAI_API_KEY}
    
    infrastructure:
      backend: local
      workers: 8  # 8 parallel workers
    
    evaluations:
      - name: fast-benchmarks
        benchmarks:
          - name: mmlu
            num_fewshot: 5
            backend_params:
              judge_worker_num: 4  # 4 concurrent judge evaluations
```

## Complete Example
```yaml
targets:
  # Model to evaluate
  - name: vllm-qwen
    type: llm
    provider: openai
    model: Qwen/Qwen3-0.6B
    base_url: http://localhost:8888/v1
    
    infrastructure:
      backend: local
      workers: 3
    
    evaluations:
      - name: comprehensive-benchmarks
        benchmarks:
          # Academic
          - name: mmlu
            num_fewshot: 3
            tasks: ["STEM"]
            limit: 100
            backend_params:
              max_tokens: 256
              temperature: 0.0
          
          # Reasoning
          - name: gsm8k
            num_fewshot: 5
            limit: 50
            backend_params:
              max_tokens: 512
          
          - name: bbh
            num_fewshot: 3
            subset: ["Logical Reasoning"]
            limit: 30
          
          # Reading
          - name: squad
            num_fewshot: 0
            limit: 100
          
          # Coding
          - name: humaneval
            num_fewshot: 0
            limit: 50
            backend_params:
              max_tokens: 1024
              k: 1
  
  # Judge model (optional - for benchmarks that need it)
  - name: gpt4-judge
    type: llm
    provider: openai
    model: gpt-4-turbo-preview
    api_key: ${OPENAI_API_KEY}
```

## Running Benchmarks
```bash
# Run all benchmarks
surogate eval --config config.yaml

# Run specific target
surogate eval --config config.yaml --target vllm-qwen

# Test with limits first
# Edit config to set limit: 3 for each benchmark
surogate eval --config config.yaml
```

## Expected Results

### Small Models (0.5B - 3B params)

| Benchmark | Expected Score |
|-----------|----------------|
| MMLU (STEM) | 0.25-0.35 |
| HellaSwag | 0.45-0.55 |
| GSM8K | 0.10-0.20 |
| HumanEval | 0.05-0.15 |
| TruthfulQA | 0.30-0.40 |

### Medium Models (7B - 13B params)

| Benchmark | Expected Score |
|-----------|----------------|
| MMLU (STEM) | 0.45-0.60 |
| HellaSwag | 0.70-0.80 |
| GSM8K | 0.30-0.50 |
| HumanEval | 0.20-0.35 |
| TruthfulQA | 0.40-0.55 |

### Large Models (70B+ params, GPT-4, Claude)

| Benchmark | Expected Score |
|-----------|----------------|
| MMLU | 0.75-0.90 |
| HellaSwag | 0.85-0.95 |
| GSM8K | 0.80-0.95 |
| HumanEval | 0.70-0.90 |
| TruthfulQA | 0.60-0.80 |

## Performance Tips

1. **Start with limits** - Test with `limit: 3` first
2. **Use appropriate few-shot** - More isn't always better
3. **Set max_tokens** - Avoid truncation for code/math
4. **Enable parallelization** - Use multiple workers
5. **Monitor resources** - Large benchmarks use significant compute

## Benchmark Backends

Surogate automatically selects the best backend:

1. **EvalScope** (Primary) - Most benchmarks
2. **lm-eval-harness** (Fallback) - If installed
3. **DeepEval** (Fallback) - Specific metrics

No backend configuration needed - it's automatic!

## Next Steps

- [Third-Party Benchmarks](./third-party-benchmarks.md) - Specialized tests
- [Results & Reporting](./results.md) - View benchmark results