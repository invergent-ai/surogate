# Examples Library

Surogate comes with a collection of pre-built training recipes and configurations for popular models and hardware setups. You can find these in the `examples/` directory of the repository.

## Pre-training (PT)

Pre-training examples for base models on large datasets.

- **[Qwen 3 Dense (PT)](examples/pt/qwen3.md)**: Standard pre-training configuration for Qwen 3 using FP8 Mixed Precision and the NorMuon optimizer.

## Supervised Fine-Tuning (SFT)

Fine-tuning examples for chat and instruction models.

- **[Qwen 3 LoRA (BF16)](examples/sft/qwen3-lora.md)**: Standard LoRA fine-tuning in BFloat16 precision.
- **[Qwen 3 QLoRA (FP4/FP8)](examples/sft/qwen3-qlora.md)**: Memory-efficient fine-tuning using quantization on modern GPUs.
- **[Qwen 3 MoE (QLoRA)](examples/sft/qwen3moe-lora.md)**: Fine-tuning Mixture-of-Experts models.

## Reinforcement Learning (GRPO)

GRPO examples for reward-based RL fine-tuning.

- **[Reverse-text GRPO walkthrough](examples/grpo/grpo.md)**: End-to-end SFT warmup + GRPO fine-tuning.
- Native GRPO YAMLs: `examples/grpo-native/reverse-text.yaml`, `examples/grpo-native/qwen3-lora.yaml`.

## How to use these examples

All examples are provided as YAML configuration files. You can run them using the Surogate CLI:

```bash
# PT / SFT
surogate [pt|sft] path/to/example.yaml

# GRPO (vLLM pipeline)
CUDA_VISIBLE_DEVICES=0 surogate grpo-infer infer.yaml
surogate grpo-orch orch.yaml
CUDA_VISIBLE_DEVICES=1 surogate grpo-train train.yaml

# GRPO (native single process)
surogate grpo-native path/to/native-grpo.yaml
```

For more details on configuration options, see the [Configuration Guide](../guides/configuration.md).
