# Surogate vs Unsloth


## GPU: 1x NVIDIA RTX 5090 32GB (tok/sec) (CUDA 13.1)
| Model          | Unsloth NF4 | Unsloth BF16 | Surogate BF16 | Surogate FP8 | Surogate QFP8 | Surogate FP4 | Surogate QFP4 | Surogate NF4 |
| -------------- | ----------- | ------------ | ------------- | ------------ | ------------- | ------------ | ------------- | ------------ |
| **Qwen3 0.6B** | 19,1k       | 21,7k        | 32,5k         | 40.2k        | 35,3k         | 42,2k        | 36,4k         | 30,1k        |
| **Qwen3 1.7B** | 12k         | 12,6k        | 16k           | 22,4k        | 20,0k         | 25,1k        | 20,3k         | 15,2k        |
| **Qwen3 4B**   | 6k          | 6.1k         | 7,1k          | 10,3k        | 9,1k          | 12,5k        | 9,4k          | 6.8k         |
| **Qwen3 8B**   | 3,4k        | 3,5k         | 4,2k          | 6,4k         | 5,7k          | 8,5k         | 5,9k          | 4k           |
 
 ## GPU: 1x NVIDIA RTX 5090 32GB (tok/sec) (CUDA 12.9)
| Model                  | Unsloth NF4 | Unsloth BF16 | Surogate BF16 | Surogate FP8 | Surogate QFP8 | Surogate FP4 | Surogate QFP4 | Surogate NF4 |
| ---------------------- | ----------- | ------------ | ------------- | ------------ | ------------- | ------------ | ------------- | ------------ |
| **Qwen3 0.6B**         | 19,1k       | 22,1k        | 32,0k         | 38,2k        | 33,5k         | 41,5k        | 34,5k         | 29,7k        |
| **Qwen3 1.7B**         | 12k         | 12,6k        | 16k           | 22,4k        | 20,0k         | 25,1k        | 20,3k         | 15,2k        | x |
| **Qwen3 4B**           | 6k          | 6.1k         | 7,1k          | 10,3k        | 9,1k          | 12,5k        | 9,4k          | 6.8k         | x |
| **Qwen3 8B**           | 3,4k        | 3,5k         | 3,8k          | 5,4k         | 5,7k          | 8,5k         | 5,9k          | 4k           | x |
| **Qwen/Qwen3-30B-A3B** |             | 0.016k       | 0.3k          |              |               |              |               |              |   |



* Across all model sizes, Surogate beats Unsloth by a large margin for every listed precision format.
* Surogate FP4 is the best throughput option, delivering about +107% to +128% vs Unsloth NF4 across all model sizes
* Overall average speed improvement vs Unsloth NF4:
  * Surogate BF16: +36%
  * Surogate FP8: +78%
  * Surogate QFP8: +61%
  * Surogate FP4: +116%
  * Surogate QFP4: +68%
  * Surogate NF4: +68%

## GPU: 1x NVIDIA H100 SXM (tok/sec)
| Model          | Unsloth NF4 | Unsloth BF16 | Surogate BF16 | Surogate FP8 | Surogate QFP8 | Surogate FP4 | Surogate QFP4 | Surogate NF4 |
| -------------- | ----------- | ------------ | ------------- | ------------ | ------------- | ------------ | ------------- | ------------ |
| **Qwen3 0.6B** | 21k         | 24,5k        | 62k           | 58k          | 45,6k         | -            | -             |              |
| **Qwen3 1.7B** | 20k         | 23k          | 37,5k         | 37,5k        | 30.6k         | -            | -             |              |
| **Qwen3 4B**   | 12,6k       | 13,1k        | 17,5k         | 18.7k        | 15,5k         | -            | -             |              |
| **Qwen3 8B**   | 8,6k        | 9,1k         | 11,6k         | 12,9k        | 10,2k         | -            | -             |              |
| **Qwen3 14B**  | 5,2k        | 5,6k         | 6,8k          | 8k           | 6,2k          | -            | -             |              |

* Surogate BF16 achieves up to 3x throughput on small models (0.6B) due to better memory bandwidth utilization on datacenter GPUs.
* FP8 matches BF16 throughput while enabling larger batch sizes through reduced memory footprint.
* Overall average speed improvement vs Unsloth NF4:
  * Surogate BF16: +89%
  * Surogate FP8: +90%
  * Surogate QFP8: +53%

## GPU: 1x NVIDIA H200 (tok/sec)
| Model          | Unsloth NF4 | Unsloth BF16 | Surogate BF16 | Surogate FP8 | Surogate QFP8 | Surogate FP4 | Surogate QFP4 |
| -------------- | ----------- | ------------ | ------------- | ------------ | ------------- | ------------ | ------------- |
| **Qwen3 0.6B** | 18,3k       | 21,7k        | 65,2k         | 62k          | 48k           | -            | -             |
| **Qwen3 1.7B** | 18,3k       | 21,4k        | 39k           | 40,8k        | 32,8k         | -            | -             |
| **Qwen3 4B**   | 12,1k       | 12,8k        | 18,3k         | 20,5k        | 16,5k         | -            | -             |
| **Qwen3 8B**   | 8,4k        | 9,1k         | 11,7k         | 14k          | 10,9k         | -            | -             |

* H200's higher memory bandwidth amplifies Surogate's advantage, with BF16 reaching 3.5x throughput on small models (0.6B).
* FP8 slightly outperforms BF16 on H200, suggesting better utilization of the Transformer Engine.
* Overall average speed improvement vs Unsloth NF4:
  * Surogate BF16: +115%
  * Surogate FP8: +124%
  * Surogate QFP8: +77%


## GPU: 1x NVIDIA B200 (CUDA 13, cuDNN 9.17)
| Model          | Unsloth NF4 | Unsloth BF16 | Surogate BF16 | Surogate FP8 | Surogate QFP8 | Surogate FP4 | Surogate QFP4 |
| -------------- | ----------- | ------------ | ------------- | ------------ | ------------- | ------------ | ------------- |
| **Qwen3 0.6B** | 17k         | 19,1k        | 95k           | 86k          | 51,8k         | 67k          |               |
| **Qwen3 1.7B** | 16,7k       | 20,3K        | 65k           | 62k          | 43k           | 46,9k        |               |
| **Qwen3 4B**   | 13,1k       | 14,8K        | 32k           | 33,5k        | 21,4k         | 24,5k        |               |
| **Qwen3 8B**   | 11,3k       | 12,4K        | 21,3k         | 24,2k        | 13,8k         | 18k          |               |
| **Qwen3 14B**  |             | 8,6k         | 12,9k         | 15k          | 8,1k          | 11,5k        |               |
| **Qwen3 32B**  |             | 4,2k         | 5,8k          | 6,8k         |               | 5,3k         | 4,8k          |

* If FP4 underperforms FP8 on B200, it’s usually overhead-dominated: FP4 needs a global amax (two-level scaling), and older builds computed that amax with a full-tensor `abs_max` reduction per matmul. On very fast SM100 GPUs this extra memory pass can outweigh the FP4 GEMM speedup; the fix is to reuse amax computed inside RMSNorm/SwiGLU/attention and skip the extra reduction in the FP4 quantizer.


## GPU: 1x NVIDIA B300 SXM6 AC (CUDA 13.0)
| Model          | Unsloth NF4 | Unsloth BF16 | Surogate BF16 | Surogate FP8 | Surogate QFP8 | Surogate FP4 | Surogate QFP4 |
| -------------- | ----------- | ------------ | ------------- | ------------ | ------------- | ------------ | ------------- |
| **Qwen3 0.6B** |             |              | 82,5k         | 75,5k        |               |              |               |
| **Qwen3 1.7B** |             |              | 59,3k         | 57k          |               |              |               |
| **Qwen3 4B**   |             |              | 29,9k         | 30k          |               |              |               |
| **Qwen3 8B**   |             |              | 21k           | 22,7k        |               |              |               |

WARNING: unknown device NVIDIA B300 SXM6 AC

qfp8:
[NVML WARNING] NVML_ERROR_NOT_SUPPORTED
480 tps

fp4:
  File "/root/.venv/lib/python3.12/site-packages/surogate/train/trainer.py", line 214, in run_training_loop
    self.trainer.step(in_tokens, out_tokens)
RuntimeError: CUTLASS FP4 GEMM (alpha-ptr) not compiled for this architecture. Ensure CUDA_ARCHITECTURES includes 120 or 121.
terminate called after throwing an instance of 'cuda_error'
  what():  Cuda Error in /__w/surogate/surogate/csrc/src/binding/py_train.cpp:81 (cudaSetDevice(ctx.Communicator->rank())): cudaErrorInvalidDevice: invalid device ordinal

qfp4:
very slow

compute_cap
10.3


## GPU Memory Usage (Gb)

| Model          | Unsloth NF4 | Surogate FP8 | Surogate QFP8 | Surogate FP4 | Surogate QFP4          | Surogate NF4 |
| -------------- | ----------- | ------------ | ------------- | ------------ | ---------------------- | ------------ |
| **Qwen3 0.6B** | 4.0         | 5.4          | 5.0           | 5.5          | 4.8                    |              |
| **Qwen3 1.7B** | 5.0         | 8.8          | 7.4           | 9            | 6.9                    | 6.2          |
| **Qwen3 4B**   | 7.6         | 16           | 12.0          | 16           | 10.7                   | 11           |
| **Qwen3 8B**   | 11.8        | 27.0         | 32.0          | 27.3         | 17.5                   | 15           |
| **Qwen3 14B**  | 11.8        | 43.0         | 59.0          | 42,6         | hang on 100% gpu usage |              |
| **Qwen3 32B**  |             | 84,6         |               | 84.0         | 105                    |              |

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
lora rank = 16
lora_alpha = 32
lora_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

# Surogate install
```shell
curl -sSL https://surogate.ai/install.sh | bash
source .venv/bin/activate
```

Configurations used:
- Surogate BF16: ./benchmarks/benchmark_sft.sh "Qwen/Qwen3-0.6B" bf16
- Surogate FP8: ./benchmarks/benchmark_sft.sh "Qwen/Qwen3-0.6B" fp8
- Surogate QFP8: ./benchmarks/benchmark_sft.sh "Qwen/Qwen3-0.6B" qfp8
- Surogate FP4: ./benchmarks/benchmark_sft.sh "Qwen/Qwen3-0.6B" fp4
- Surogate QFP4: ./benchmarks/benchmark_sft.sh "Qwen/Qwen3-0.6B" qfp4
- Surogate NF4: ./benchmarks/benchmark_sft.sh "Qwen/Qwen3-0.6B" qbnb


# Unsloth install
```shell
apt install -y python3-dev
uv venv --python=3.12
source .venv/bin/activate
uv pip install unsloth
```
