---
id: functional-metrics
title: Functional Metrics
sidebar_label: Functional Metrics
---

# Functional Metrics

Functional metrics evaluate the quality, safety, and correctness of model outputs using LLM-as-judge and other evaluation techniques.

## Core Metrics

### G-Eval

LLM-as-judge metric for custom evaluation criteria.
```yaml
metrics:
  - name: correctness
    type: g_eval
    criteria: "Does the output correctly answer the input question?"
    evaluation_params:
      - actual_output
    judge_model:
      target: openrouter-gpt4
```

**Evaluation Parameters:**
- `actual_output`: The model's response (required)
- `input`: The original prompt
- `expected_output`: Ground truth answer
- `context`: Additional context

**Advanced Options:**
```yaml
- name: detailed-correctness
  type: g_eval
  criteria: "Evaluate correctness, completeness, and accuracy"
  evaluation_steps:
    - "Check if key facts are correct"
    - "Verify all parts of question are addressed"
    - "Assess accuracy of technical details"
  evaluation_params:
    - actual_output
    - expected_output
    - input
  threshold: 0.7  # Minimum passing score
  judge_model:
    target: openrouter-gpt4
```

### Conversational G-Eval

Evaluate multi-turn conversations.
```yaml
- name: conversation-quality
  type: conversational_g_eval
  criteria: "Does the assistant maintain context and provide helpful responses throughout the conversation?"
  judge_model:
    target: openrouter-gpt4
```

Automatically handles conversation turns and evaluates coherence across the dialogue.

### Multimodal G-Eval

Evaluate vision-language model outputs.
```yaml
- name: image-text-coherence
  type: multimodal_g_eval
  criteria: "Are the images and text coherent and logically aligned?"
  evaluation_params:
    - input
    - actual_output
  threshold: 0.7
  judge_model:
    target: qwen-vl
```

**With Rubric:**
```yaml
- name: visual-quality
  type: multimodal_g_eval
  criteria: "Assess overall multimodal quality"
  evaluation_params:
    - input
    - actual_output
    - expected_output
  rubric:
    - score_range: [1, 3]
      expected_outcome: "Images unclear, text inaccurate, poor coherence"
    - score_range: [4, 7]
      expected_outcome: "Images recognizable, text mostly accurate"
    - score_range: [8, 10]
      expected_outcome: "Clear images, accurate text, excellent alignment"
  threshold: 0.6
  judge_model:
    target: qwen-vl
```

### DAG Metric

Complex evaluation with decision graphs.
```yaml
- name: format-correctness-dag
  type: dag
  threshold: 0.6
  
  root_nodes:
    - "check_title"
  
  nodes:
    # Check for title
    - id: "check_title"
      type: "task"
      instructions: "Does the summary start with a clear title?"
      output_label: "has_title"
      evaluation_params:
        - "actual_output"
      children:
        - "title_decision"
    
    - id: "title_decision"
      type: "binary_judgement"
      criteria: "Does the summary have a title line?"
      evaluation_params:
        - "actual_output"
      children:
        - "no_title"
        - "has_title_continue"
    
    - id: "no_title"
      type: "verdict"
      verdict: false
      score: 0.1
    
    - id: "has_title_continue"
      type: "verdict"
      verdict: true
      children:
        - "count_sections"
    
    # Count sections
    - id: "count_sections"
      type: "task"
      instructions: "Count labeled sections in the summary"
      output_label: "section_count"
      evaluation_params:
        - "actual_output"
      children:
        - "evaluate_sections"
    
    - id: "evaluate_sections"
      type: "non_binary_judgement"
      criteria: "How many well-structured sections?"
      evaluation_params:
        - "actual_output"
      children:
        - "excellent_sections"
        - "good_sections"
        - "minimal_sections"
    
    - id: "excellent_sections"
      type: "verdict"
      verdict: "3 or more sections"
      score: 1.0
    
    - id: "good_sections"
      type: "verdict"
      verdict: "2 sections"
      score: 0.7
    
    - id: "minimal_sections"
      type: "verdict"
      verdict: "0-1 sections"
      score: 0.3
```

**Node Types:**
- `task`: Execute instructions and store output
- `binary_judgement`: Yes/no decision (2 children)
- `non_binary_judgement`: Multiple choice (3+ children)
- `verdict`: Terminal node with score

## Multi-Turn Metrics

### Conversation Coherence
```yaml
- name: conversation-coherence
  type: conversation_coherence
  window_size: 3  # Check last 3 turns
  judge_model:
    target: openrouter-gpt4
```

Evaluates logical flow and consistency across conversation turns.

### Context Retention
```yaml
- name: context-retention
  type: context_retention
  key_info_threshold: 0.7  # Must retain 70% of key info
  judge_model:
    target: openrouter-gpt4
```

Checks if the model remembers important information from earlier turns.

### Turn-Level Analysis
```yaml
- name: turn-analysis
  type: turn_analysis
  analyze_all_turns: true  # Analyze each turn individually
  judge_model:
    target: openrouter-gpt4
```

Provides detailed per-turn evaluation scores.

## Safety Metrics

### Toxicity Detection
```yaml
- name: toxicity-check
  type: toxicity
  threshold: 0.5  # Max acceptable toxicity score
  judge_model:
    target: openrouter-gpt4
```

Detects harmful, offensive, or toxic content. Returns score 0-1 (higher = more toxic).

### Bias Detection
```yaml
- name: bias-detection
  type: bias
  bias_types:
    - gender
    - race
    - religion
    - age
    - nationality
  threshold: 0.5
  judge_model:
    target: openrouter-gpt4
```

Identifies potential biases in model outputs across multiple dimensions.

### Harm Assessment
```yaml
- name: harm-assessment
  type: harm
  harm_categories:
    - violence
    - self_harm
    - illegal_activity
    - misinformation
    - hate_speech
  threshold: 0.3
  judge_model:
    target: openrouter-gpt4
```

Comprehensive safety check for various harm categories.

## Non-LLM Metrics

### Embedding Similarity
```yaml
- name: semantic-similarity
  type: embedding_similarity
  similarity_function: cosine  # cosine, euclidean, dot_product
  threshold: 0.8
```

Requires embedding target and expected embeddings in dataset.

### Classification Metrics
```yaml
- name: classification-accuracy
  type: classification
  metric_type: accuracy  # accuracy, precision, recall, f1
```

For classification tasks with labeled outputs.

## Judge Models

### Using Same Target as Judge
```yaml
judge_model:
  target: openrouter-gpt4  # Must be defined in targets
```

### Using Different Target
```yaml
# In targets section
- name: gpt4-judge
  type: llm
  provider: openai
  model: gpt-4-turbo-preview
  api_key: ${OPENAI_API_KEY}

# In metrics
metrics:
  - name: quality
    type: g_eval
    criteria: "..."
    judge_model:
      target: gpt4-judge  # Use dedicated judge model
```

## Complete Example
```yaml
targets:
  - name: test-model
    type: llm
    provider: openai
    model: gpt-4
    api_key: ${OPENAI_API_KEY}

  - name: judge-model
    type: llm
    provider: openai
    model: gpt-4-turbo-preview
    api_key: ${OPENAI_API_KEY}

    evaluations:
      - name: comprehensive-quality
        dataset: data/qa.jsonl
        
        metrics:
          # Quality
          - name: correctness
            type: g_eval
            criteria: "Is the answer factually correct?"
            evaluation_params:
              - actual_output
              - expected_output
            judge_model:
              target: judge-model
          
          - name: relevance
            type: g_eval
            criteria: "Is the answer relevant to the question?"
            evaluation_params:
              - input
              - actual_output
            threshold: 0.7
            judge_model:
              target: judge-model
          
          # Safety
          - name: toxicity
            type: toxicity
            threshold: 0.3
            judge_model:
              target: judge-model
          
          - name: bias
            type: bias
            bias_types: [gender, race]
            threshold: 0.4
            judge_model:
              target: judge-model
```

## Best Practices

1. **Choose appropriate judge models** - GPT-4 or Claude for best results
2. **Set realistic thresholds** - Test with small samples first
3. **Combine multiple metrics** - Quality + Safety + Performance
4. **Use clear criteria** - Specific, measurable evaluation standards
5. **Include expected_output** - When available, enables better evaluation

## Next Steps

- [Performance Testing](./performance-testing.md) - Speed and scalability
- [Results & Reporting](./results.md) - View evaluation results