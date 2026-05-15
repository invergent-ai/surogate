# Qwen 3 LoRA Fine-Tuning (SFT)

This example shows how to perform Supervised Fine-Tuning (SFT) on a Qwen 3 model using LoRA in BF16 precision.

## Configuration Highlights

- **Model**: `Qwen/Qwen3-0.6B`
- **Technique**: LoRA (Rank 16)
- **Precision**: `bf16`
- **Dataset**: `OpenLLM-Ro/ro_gsm8k` (Math reasoning in Romanian)

## Running the example

```bash
surogate sft examples/sft/qwen3-lora-bf16.yaml
```

## Config File (`examples/sft/qwen3-lora-bf16.yaml`)

```yaml
model: Qwen/Qwen3-0.6B
output_dir: ./output

per_device_train_batch_size: 2
gradient_accumulation_steps: 4
sequence_len: 2048

recipe: bf16

lora: true
lora_rank: 16
lora_alpha: 32
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

datasets:
  - path: "OpenLLM-Ro/ro_gsm8k"
    type: auto
```
