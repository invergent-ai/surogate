# Surogate vs Unsloth

* Across all model sizes, Surogate beats Unsloth  by a large margin for every listed precision format.
* Surogate FP4 is the best throughput option, delivering about +107% to +125% vs Unsloth NF4 across all model sizes
* Overall average speed improvement vs Unsloth NF4:
  * Surogate BF16: +32%
  * Surogate FP8: +77%
  * Surogate QFP8: +64%
  * Surogate FP4: +115%
  * Surogate QFP4: +72%


## GPU: 1x NVIDIA RTX 5090 32GB (tok/sec)
| Model          | Unsloth NF4 | Unsloth BF16 | Surogate BF16 | Surogate FP8 | Surogate QFP8 | Surogate FP4 | Surogate QFP4 |
|----------------|-------------|--------------|---------------|--------------|---------------|--------------|---------------|
| **Qwen3 0.6B** | 19k         | 22,9k        | 32k           | 38,3k        | 33,6k         | 41k          | 35.9k         |
| **Qwen3 1.7B** | 11,7k       | 12,1k        | 15,2k         | 20,7k        | 19,0k         | 24,2k        | 19,5k         |
| **Qwen3 4B**   | 5,6k        | 5,8k         | 7,2k          | 9,5k         | 8,8k          | 12,0k        | 9,1k          |
| **Qwen3 8B**   | 3,6k        | 3,4k         | 4,2k          | 5,9k         | 5,3k          | 8,2k         | 5,5k          | 

Configurations used:
- Surogate BF16: ./benchmarks/benchmark.sh "Qwen/Qwen3-0.6B" bf16
- Surogate FP8: ./benchmarks/benchmark.sh "Qwen/Qwen3-0.6B" fp8
- Surogate QFP8: ./benchmarks/benchmark.sh "Qwen/Qwen3-0.6B" qfp8
- Surogate FP4: ./benchmarks/benchmark.sh "Qwen/Qwen3-0.6B" fp4
- Surogate QFP4: ./benchmarks/benchmark.sh "Qwen/Qwen3-0.6B" qfp4
- 
## GPU: 1x NVIDIA H100 SXM (tok/sec)
| Model          | Unsloth NF4 | Unsloth BF16 | Surogate BF16 | Surogate FP8 | Surogate QFP8 | Surogate FP4 | Surogate QFP4 |
|----------------|-------------|--------------|---------------|--------------|---------------|--------------|---------------|
| **Qwen3 0.6B** | 21k         | 24,5k        |               |              |               |              |               |
| **Qwen3 1.7B** | 20k         | 23k          |               |              |               |              |               |
| **Qwen3 4B**   | 12,6k       | 13,1k        |               |              |               |              |               |
| **Qwen3 8B**   | 8,6k        | 9,1k         |               |              |               |              |               | 



## GPU Memory Usage (Gb)
| Model          | Unsloth NF4 | Surogate FP8 | Surogate QFP8 | Surogate FP4 | Surogate QFP4 |
|----------------|-------------|--------------|---------------|--------------|---------------|
| **Qwen3 0.6B** | 4.0         | 5.4          | 5.0           | 5.5          | 4.8           |
| **Qwen3 1.7B** | 5.0         | 8.8          | 7.4           | 9            | 6.9           |
| **Qwen3 4B**   | 7.6         | 16           | 12.0          | 16           | 10.7          |
| **Qwen3 8B**   | 11.8        | 26.7         | 20.1          | 27.3         | 17.5          |


--

## Formulas used
$$\text{Tokens/sec} = \frac{\text{Batch Size} \times \text{Grad Accum Steps} \times \text{Max Seq Length} \times \text{Num GPUs}}{\text{sec/iter}}$$

$$\text{Tokens/sec} = (\text{iter/sec}) \times \text{Batch Size} \times \text{Grad Accum Steps} \times \text{Max Seq Length} \times \text{Num GPUs}$$


Benchmark configuration:
```
dataset = 10000 samples
max_seq_length=2048
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
packing = True
lora rank = 32
lora_alpha = 32
lora_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```