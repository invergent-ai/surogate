# Qwen 3 QLoRA Fine-Tuning

Surogate supports various quantization backends for memory-efficient fine-tuning (QLoRA), including BitsAndBytes (NF4), Native FP8, and Native NVFP4 (for Blackwell GPUs).

## 1. Native FP8 QLoRA
Recommended for ADA (RTX 4090, L40S) and Hopper (H100) GPUs.

```bash
surogate sft examples/sft/qwen3-lora-qfp8.yaml
```

**Key Config:**
```yaml
recipe: fp8-hybrid
lora: true
qlora: true # Enables quantization of the base model
```

## 2. Blackwell NVFP4 QLoRA
Maximum efficiency on B200 and RTX 50xx series.

```bash
surogate sft examples/sft/qwen3-lora-qfp4.yaml
```

**Key Config:**
```yaml
recipe: fp4-nvfp4
lora: true
qlora: true
```

## 3. BitsAndBytes (NF4)
Compatible with almost all modern NVIDIA GPUs.

```bash
surogate sft examples/sft/qwen3-lora-qbnb.yaml
```

**Key Config:**
```yaml
recipe: bf16
lora: true
qlora: true
quantization: bnb-nf4
```
