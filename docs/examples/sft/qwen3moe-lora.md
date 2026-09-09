# Qwen3 MoE QLoRA

```bash
surogate sft examples/sft/qwen3moe/qwen3moe-lora-qbnb.yaml
```

The [maintained config](../../../examples/sft/qwen3moe/qwen3moe-lora-qbnb.yaml) uses
Qwen3-30B-A3B, NF4 base quantization, BF16 adapters, expert offload and selective
expert dequantization. The [MoE and multi-GPU examples](../../../examples/training/README.md#multiple-gpus-and-large-models)
also cover expert parallelism, balancing, router losses and chunked sequences.
