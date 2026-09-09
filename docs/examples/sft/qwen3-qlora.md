# Qwen3 QLoRA

Choose one online quantization mode:

```bash
surogate sft examples/sft/qwen3/qwen3-lora-qbnb.yaml
surogate sft examples/sft/qwen3/qwen3-lora-qfp8.yaml
surogate sft examples/sft/qwen3/qwen3-lora-qfp4.yaml
```

The configs use `qlora_bnb`, `qlora_fp8`, or `qlora_fp4` with `lora: true`.
`recipe` controls compute precision separately from frozen base storage. NVFP4
requires Blackwell; check the [precision guide](../../guides/precision-and-recipes.md)
for GPU requirements. Already quantized FP8/NVFP4 checkpoints load directly and
must not enable online quantization flags. See the
[recipe index](../../../examples/README.md#model-and-precision-recipes).
