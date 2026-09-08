# Fine-tuning MiniCPM5 and Spark-X2.5

Use the Hugging Face safetensors checkpoints for training. Supported checkpoints include `openbmb/MiniCPM5-1B`, `openbmb/MiniCPM5-2B`, `XHToken/Spark-X2.5-1.7B`, and `XHToken/Spark-X2.5-4B`. Model dimensions are read from `config.json`, so a local checkpoint directory works too. GGUF files are for serving.

## Start a LoRA run

Choose an example, replace its dataset with your own, and run:

```bash
surogate sft examples/sft/minicpm5/minicpm5-lora-bf16.yaml
# Or:
surogate sft examples/sft/spark/spark-lora-bf16.yaml
```

To train the larger checkpoint, change only the `model` field to `openbmb/MiniCPM5-2B` or `XHToken/Spark-X2.5-4B`. Adjust the sequence length and batch size to fit your GPU memory. Both examples use BF16 and activation recomputation.

MiniCPM5 uses these LoRA targets:

```yaml
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

Spark uses different attention projection names:

```yaml
lora_target_modules: [q_k_v_proj, out_proj, gate_proj, up_proj, down_proj]
```

You can also use `lora_target_modules: [all]`. For Spark, `q_k_v_proj` trains its query, key, and value projection together. The attention gate is updated during full fine-tuning; it is not a LoRA target.

For full fine-tuning, set `lora: false` and start with `learning_rate: 1e-5`. Full fine-tuning requires more memory for gradients and optimizer state.

## Use the trained model

Checkpoints are saved under `output_dir`. To merge a saved adapter into a standalone safetensors checkpoint:

```bash
surogate merge \
  --base-model XHToken/Spark-X2.5-1.7B \
  --checkpoint-dir output/spark/step_00000100 \
  --output merged-spark
```

Use the matching MiniCPM5 model and checkpoint directory to merge a MiniCPM5 adapter. Spark adapters use the checkpoint's native projection names and can also be loaded by PEFT.

Serve the merged Spark checkpoint with:

```bash
surogate serve merged-spark --tool-call-parser spark25
```

Spark serving currently requires a merged checkpoint; loading a separate adapter at serving time is not supported.
