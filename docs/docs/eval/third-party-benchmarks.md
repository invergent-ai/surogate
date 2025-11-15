---
id: third-party-benchmarks
title: Third-Party Benchmarks
sidebar_label: Third-Party Benchmarks
---

# Third-Party Benchmarks

Specialized benchmarks beyond standard academic tests, including function calling, long-context, and vision evaluations.

## Available Benchmarks

All third-party benchmarks use the same configuration format as standard benchmarks. They're automatically registered and work with the same evaluation pipeline.

### Function Calling & Tool Use

#### BFCL v3 & v4

Berkeley Function Calling Leaderboard for evaluating function calling accuracy.
```yaml
benchmarks:
  - name: bfcl-v3
    num_fewshot: 0
    limit: 100
    backend_params:
      max_tokens: 1024
  
  - name: bfcl-v4
    num_fewshot: 0
    limit: 100
```

**What it tests:**
- Function signature understanding
- Argument extraction and formatting
- Multi-function scenarios
- Complex parameter types

#### Tau-Bench & Tau2-Bench

Agent evaluation for task completion (if available in EvalScope backend).
```yaml
benchmarks:
  - name: tau-bench
    num_fewshot: 0
    limit: 50
    backend_params:
      max_tokens: 2048
  
  - name: tau2-bench
    num_fewshot: 0
    limit: 30
```

**What it tests:**
- Multi-step task execution
- Tool usage patterns
- Real-world task completion

#### ToolBench

Tool-use evaluation (if available in EvalScope backend).
```yaml
benchmarks:
  - name: toolbench
    num_fewshot: 3
    limit: 100
```

**What it tests:**
- Tool selection
- API calling accuracy
- Parameter passing

### Long-Context Evaluation

#### LongBench

Comprehensive long-context benchmark suite.
```yaml
benchmarks:
  - name: longbench
    num_fewshot: 0
    limit: 50
    backend_params:
      max_tokens: 1024
```

**What it tests:**
- Single/multi-document QA
- Long text summarization
- Code understanding
- Few-shot learning with long context

**Available tasks:**
- `narrativeqa` - Story understanding
- `qasper` - Scientific paper QA
- `multifieldqa_en` - Multi-domain QA
- `hotpotqa` - Multi-hop reasoning
- `2wikimqa` - Wikipedia multi-hop
- `gov_report` - Government document summarization
- `qmsum` - Meeting summarization
- `multi_news` - News summarization
- `trec` - Passage retrieval
- `triviaqa` - Trivia questions
- `samsum` - Dialogue summarization
- `passage_count` - Counting tasks
- `passage_retrieval_en` - Passage finding
- `lcc` - Code completion
- `repobench-p` - Code repository understanding

Filter specific tasks:
```yaml
- name: longbench
  num_fewshot: 0
  tasks:
    - "narrativeqa"
    - "qasper"
    - "gov_report"
  limit: 30
```

#### LongBench-Write

Long-form text generation evaluation.
```yaml
benchmarks:
  - name: longbench-write
    num_fewshot: 0
    limit: 20
    backend_params:
      max_tokens: 4096  # Long outputs needed
```

**What it tests:**
- Long document generation
- Coherence over extended text
- Structure maintenance

#### Needle in a Haystack

Long-context retrieval test (if available).
```yaml
benchmarks:
  - name: needle-in-haystack
    num_fewshot: 0
    limit: 20
    backend_params:
      max_tokens: 512
```

**What it tests:**
- Information retrieval from long contexts
- Attention span
- Recall at different positions

### Vision Benchmarks

Multimodal benchmarks require a `multimodal` target type.

#### MMMU

Massive Multitask Multimodal Understanding.
```yaml
targets:
  - name: vision-model
    type: multimodal
    provider: openai
    model: qwen/qwen-2-vl-7b-instruct
    base_url: https://openrouter.ai/api/v1
    api_key: ${OPENROUTER_API_KEY}
    
    evaluations:
      - name: vision-benchmarks
        benchmarks:
          - name: mmmu
            num_fewshot: 0
            limit: 100
            subset:
              - "Math"
              - "Physics"
              - "Chemistry"
```

**What it tests:**
- College-level subject knowledge
- Visual reasoning
- Diagram interpretation
- Multi-step problem solving

**Available subjects:**
- STEM: Math, Physics, Chemistry, Biology, Computer Science
- Humanities: History, Literature, Philosophy
- Social Sciences: Economics, Psychology, Sociology
- Professional: Medicine, Law, Business

#### MMMU Pro

Advanced version of MMMU (if available).
```yaml
- name: mmmu-pro
  num_fewshot: 0
  limit: 50
```

#### MathVista / Math-Vista

Mathematical visual reasoning.
```yaml
- name: mathvista
  num_fewshot: 0
  limit: 50
  backend_params:
    max_tokens: 512

# Alternative name
- name: math-vista
  num_fewshot: 0
  limit: 50
```

**What it tests:**
- Math from images
- Chart/graph interpretation
- Geometric reasoning
- Visual calculation

#### ChartQA

Question answering on charts and graphs.
```yaml
- name: chartqa
  num_fewshot: 0
  limit: 100
```

**What it tests:**
- Chart comprehension
- Data extraction from visualizations
- Trend analysis

#### Other Vision Benchmarks
```yaml
# Document understanding
- name: docvqa
  num_fewshot: 0
  limit: 50

- name: infovqa
  num_fewshot: 0
  limit: 50

# Diagram understanding
- name: ai2d
  num_fewshot: 0
  limit: 50

# General vision
- name: seed-bench
  num_fewshot: 0
  limit: 100

- name: mm-bench
  num_fewshot: 0
  limit: 100

- name: mm-star
  num_fewshot: 0
  limit: 50

- name: pope
  num_fewshot: 0
  limit: 100

- name: real-world-qa
  num_fewshot: 0
  limit: 50
```

## Complete Example
```yaml
targets:
  # Text model for function calling and long-context
  - name: text-model
    type: llm
    provider: openai
    model: gpt-4-turbo-preview
    api_key: ${OPENAI_API_KEY}
    
    infrastructure:
      backend: local
      workers: 4
    
    evaluations:
      - name: specialized-benchmarks
        benchmarks:
          # Function calling
          - name: bfcl-v4
            num_fewshot: 0
            limit: 100
          
          # Long context
          - name: longbench
            num_fewshot: 0
            tasks:
              - "narrativeqa"
              - "qasper"
            limit: 30
          
          - name: longbench-write
            num_fewshot: 0
            limit: 20
            backend_params:
              max_tokens: 4096

  # Vision model
  - name: vision-model
    type: multimodal
    provider: openai
    model: qwen/qwen-2-vl-7b-instruct
    base_url: https://openrouter.ai/api/v1
    api_key: ${OPENROUTER_API_KEY}
    
    infrastructure:
      backend: local
      workers: 2
    
    evaluations:
      - name: vision-evaluation
        benchmarks:
          - name: mmmu
            num_fewshot: 0
            limit: 50
            subset:
              - "Math"
              - "Physics"
          
          - name: mathvista
            num_fewshot: 0
            limit: 30
          
          - name: chartqa
            num_fewshot: 0
            limit: 50
```

## Running Third-Party Benchmarks
```bash
# Run all benchmarks
surogate eval --config config.yaml

# Run specific target
surogate eval --config config.yaml --target vision-model

# Test with small limits first
# Edit config to set limit: 3
surogate eval --config config.yaml
```

## Backend Integration

All third-party benchmarks use the **EvalScope** backend automatically. The benchmarks are registered in `surogate/eval/benchmarks/__init__.py`:
```python
EVALSCOPE_BENCHMARKS = [
    # ... standard benchmarks ...
    
    # Third-party
    'tau_bench', 'tau2_bench', 'toolbench',
    'bfcl', 'bfcl_v3', 'bfcl_v4',
    'longbench', 'longbench_write',
    'mmmu', 'mmmu_pro', 'mathvista', 'math_vista',
    'chartqa', 'docvqa', 'infovqa', 'ai2d',
    # ... and more
]
```

**No special configuration needed** - just specify the benchmark name and it works.

## Dataset Caching

Benchmark datasets are automatically downloaded and cached:
```
~/.cache/surogate/benchmarks/
├── longbench/
├── mmmu/
├── bfcl/
└── ...
```

First run downloads datasets, subsequent runs use cache.

## Expected Performance

### Function Calling (GPT-4 level)

| Benchmark | Expected Score |
|-----------|----------------|
| BFCL-v3 | 0.75-0.85 |
| BFCL-v4 | 0.70-0.80 |

### Long-Context (32k context)

| Benchmark | Expected Score |
|-----------|----------------|
| LongBench | 0.45-0.65 |
| LongBench-Write | 0.50-0.70 |

### Vision (7B multimodal)

| Benchmark | Expected Score |
|-----------|----------------|
| MMMU (Math) | 0.35-0.50 |
| MathVista | 0.40-0.55 |
| ChartQA | 0.50-0.65 |

## Performance Tips

1. **Start small** - Use `limit: 3` for testing
2. **Vision models** - Ensure target type is `multimodal`
3. **Long context** - Use models with 32k+ context window
4. **Function calling** - Works best with GPT-4, Claude, or instruction-tuned models
5. **Monitor resources** - Vision and long-context benchmarks use more memory

## Availability Note

Some benchmarks (tau-bench, toolbench, needle-in-haystack) may not be available in all EvalScope versions. If a benchmark fails with "not found", it's not available in your EvalScope installation.

Check available benchmarks:
```python
from surogate.eval.benchmarks import EVALSCOPE_BENCHMARKS
print(EVALSCOPE_BENCHMARKS)
```

## Next Steps

- [Standard Benchmarks](./standard-benchmarks.md) - Academic benchmarks
- [Results & Reporting](./results.md) - View and compare results