# Qwen 3 Pre-training (PT)

This example demonstrates how to pre-train a Qwen 3 model using Surogate's high-performance FP8 hybrid recipe.

## Configuration Highlights

- **Model**: `Qwen/Qwen3-0.6B`
- **Precision**: `fp8-hybrid` (Native FP8 training on Hopper/Blackwell)
- **Optimizer**: `normuon` (Optimized for faster convergence)
- **Batch Size**: 8 per device with 4 GPUs (Effective batch size 32)
- **Dataset**: `HuggingFaceFW/fineweb-2`

## Running the example

```bash
surogate pt examples/pt/qwen3.yaml
```

## Config File (`examples/pt/qwen3.yaml`)

```yaml
model: Qwen/Qwen3-0.6B
output_dir: ./output

per_device_train_batch_size: 8
gradient_accumulation_steps: 1
sequence_len: 2048

recipe: fp8-hybrid
optimizer: normuon
gpus: 4

datasets:
  - path: "HuggingFaceFW/fineweb-2"
    subset: ron_Latn
    split: train
    type: text
```
