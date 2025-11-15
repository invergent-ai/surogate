---
id: performance-testing
title: Performance Testing
sidebar_label: Performance Testing
---

# Performance Testing

Measure model speed, throughput, and scalability under load. Performance testing helps identify bottlenecks and optimize deployment.

## Performance Metrics

### Latency

Measure response time for individual requests.
```yaml
metrics:
  - name: response_latency
    type: latency
    threshold_ms: 8000  # Maximum acceptable latency in milliseconds
```

**Measured:**
- Time from request sent to response received
- Per-request timing
- Percentiles (p50, p95, p99)

**Typical Values:**
- GPT-4: 3-8 seconds
- GPT-3.5: 1-3 seconds
- Claude: 2-6 seconds
- Local vLLM: 0.05-2 seconds

### Token Generation Speed

Measure tokens generated per second.
```yaml
metrics:
  - name: token_generation_speed
    type: token_generation_speed
    min_tokens_per_sec: 15  # Minimum acceptable speed
```

**Measured:**
- Tokens generated / total time
- Output token count
- Generation efficiency

**Typical Values:**
- GPT-4: 15-25 tokens/sec
- GPT-3.5: 25-40 tokens/sec
- Claude: 20-35 tokens/sec
- Local vLLM: 50-300 tokens/sec

### Throughput

Measure requests processed per second.
```yaml
metrics:
  - name: request_throughput
    type: throughput
    min_rps: 0.3  # Minimum requests per second
```

**Measured:**
- Total requests / total time
- Concurrent request handling
- System capacity

**Typical Values:**
- API services: 0.1-1 req/sec (rate limited)
- Local vLLM: 10-200 req/sec (depends on hardware)

## Stress Testing

Find breaking points and measure performance under load.

### Fixed Load Test

Send fixed number of concurrent requests.
```yaml
targets:
  - name: vllm-model
    type: llm
    provider: openai
    model: Qwen/Qwen3-0.6B
    base_url: http://localhost:8888/v1
    
    stress_testing:
      enabled: true
      dataset: data/prompts.jsonl
      
      # Fixed load
      num_concurrent: 20  # 20 simultaneous requests
      num_requests: 200   # Total requests to send
      
      # Resource monitoring
      monitor_resources: true
      monitoring_interval: 0.5  # Sample every 0.5 seconds
      
      # Warmup
      warmup_requests: 10  # Prime the model
      
      # Failure handling
      max_failures: 20  # Stop if too many failures
```

**Results Include:**
- Throughput (req/s, tokens/s)
- Latency percentiles
- Success/failure rates
- Resource utilization (GPU, CPU, memory)

### Progressive Load Test

Gradually increase load to find breaking point.
```yaml
stress_testing:
  enabled: true
  dataset: data/prompts.jsonl
  
  # Progressive load
  progressive: true
  start_concurrent: 2   # Start with 2 concurrent
  step_concurrent: 3    # Increase by 3 each step
  step_duration_seconds: 20  # Test each level for 20s
  num_concurrent: 50    # Maximum to test
  
  # Resource monitoring
  monitor_resources: true
  monitoring_interval: 0.3
  
  warmup_requests: 5
  max_failures: 10
```

**Breaking Point Detection:**
- Monitors failure rate at each level
- Stops when failures exceed threshold
- Identifies maximum sustainable load

### Duration-Based Test

Run for fixed time period.
```yaml
stress_testing:
  enabled: true
  dataset: data/prompts.jsonl
  
  # Duration-based
  num_concurrent: 15
  duration_seconds: 60  # Run for 60 seconds
  # num_requests not specified = unlimited
  
  monitor_resources: true
  monitoring_interval: 1.0  # Every second
  warmup_requests: 10
```

**Use Cases:**
- Stability testing
- Long-running performance validation
- Production load simulation

## Resource Monitoring

Track system resources during testing.

**Requirements:**
```bash
pip install psutil nvidia-ml-py3
```

**Monitored Metrics:**
- **GPU**: Memory usage, utilization %, temperature
- **CPU**: Usage %, per-core utilization
- **Memory**: RAM usage, available memory
- **System**: Load average, swap usage

**Output:**
```
Resource Utilization:
  GPU Memory: 4.2 GB / 24 GB (17.5%)
  GPU Utilization: 85%
  CPU Usage: 45%
  System Memory: 12.3 GB / 64 GB
```

## Complete Example
```yaml
targets:
  # Performance evaluation with metrics
  - name: gpt4-performance
    type: llm
    provider: openai
    model: gpt-4-turbo-preview
    base_url: https://openrouter.ai/api/v1
    api_key: ${OPENROUTER_API_KEY}
    
    infrastructure:
      backend: local
      workers: 4
    
    evaluations:
      - name: performance-benchmark
        dataset: data/short_prompts.jsonl
        
        metrics:
          - name: latency_benchmark
            type: latency
            threshold_ms: 5000
          
          - name: throughput_benchmark
            type: throughput
            min_rps: 0.4
          
          - name: token_speed_benchmark
            type: token_generation_speed
            min_tokens_per_sec: 18

  # Stress testing on local vLLM
  - name: vllm-stress-test
    type: llm
    provider: openai
    model: Qwen/Qwen3-0.6B
    base_url: http://localhost:8888/v1
    
    infrastructure:
      backend: local
      workers: 1
    
    # Fixed load stress test
    stress_testing:
      enabled: true
      dataset: data/prompts.jsonl
      num_concurrent: 20
      num_requests: 200
      monitor_resources: true
      monitoring_interval: 0.5
      warmup_requests: 10
      max_failures: 20
    
    evaluations: []  # No regular evaluations

  # Progressive stress test
  - name: vllm-progressive
    type: llm
    provider: openai
    model: Qwen/Qwen3-0.6B
    base_url: http://localhost:8888/v1
    
    stress_testing:
      enabled: true
      dataset: data/medium_prompts.jsonl
      progressive: true
      start_concurrent: 2
      step_concurrent: 3
      step_duration_seconds: 20
      num_concurrent: 50
      monitor_resources: true
      monitoring_interval: 0.3
      warmup_requests: 5
      max_failures: 10
    
    evaluations: []
```

## Running Stress Tests

### Start vLLM Server
```bash
export VLLM_USE_MODELSCOPE=True
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B \
  --served-model-name "Qwen/Qwen3-0.6B" \
  --trust-remote-code \
  --port 8888 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.9
```

### Run Tests
```bash
# Run all stress tests
surogate eval --config config.yaml

# Run specific target only
surogate eval --config config.yaml --target vllm-stress-test
```

## Expected Performance

### vLLM (Qwen3-0.6B on single GPU)

| Metric | Value |
|--------|-------|
| Throughput | 50-200 req/s |
| Latency (p50) | 50-200ms |
| Latency (p99) | 200-500ms |
| Token Speed | 100-300 tokens/s |
| Breaking Point | 20-40 concurrent requests |

### API Services (GPT-4, Claude)

| Metric | Value |
|--------|-------|
| Throughput | 0.1-1 req/s (rate limited) |
| Latency (p50) | 3-6 seconds |
| Latency (p99) | 8-15 seconds |
| Token Speed | 15-25 tokens/s |

## Best Practices

1. **Start small** - Test with low concurrency first
2. **Use warmup** - Prime the model before measurement
3. **Monitor resources** - Track GPU/CPU/memory usage
4. **Set realistic limits** - Don't overload API services
5. **Use appropriate datasets** - Match production workload
6. **Test progressively** - Find breaking points safely

## Next Steps

- [Standard Benchmarks](./standard-benchmarks.md) - Academic evaluations
- [Results & Reporting](./results.md) - Analyze performance data