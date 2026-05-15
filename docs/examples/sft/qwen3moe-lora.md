# Qwen 3 MoE Fine-Tuning

Surogate provides optimized support for Mixture-of-Experts (MoE) models, including efficient expert parallelism and memory-saving techniques.

## Running the example

This example uses BitsAndBytes NF4 quantization to fit a MoE model on consumer hardware.

```bash
surogate sft examples/sft/qwen3moe-lora-qbnb.yaml
```

## Configuration Highlights

```yaml
model: Qwen/Qwen3-MoE-A2.7B # Example MoE model
lora: true
qlora: true
quantization: bnb-nf4

# MoE specific (if applicable)
expert_parallelism: true 
```

Note: Surogate automatically detects MoE architectures and applies optimized kernels for expert routing and execution.
