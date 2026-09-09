# Pretraining

These two configs use raw text (`Trelis/tiny-shakespeare`, column `Text`) and compute
loss on all tokens. Run from the repository root. The small corpus demonstrates the
workflow; it is not enough data to train a useful 0.6B model from scratch.

```bash
# Randomly initialize Qwen3's architecture and train all weights with NorMuon.
surogate pt examples/pt/qwen3.yaml
# Continue from Qwen3's pretrained weights with raw-text SFT and AdamW 8-bit.
surogate sft examples/pt/qwen3-continue.yaml
```

**`surogate pt` always sets `from_scratch: true` and initializes projections to zero.**
Use `surogate sft` for continued pretraining. Both configs use `lora: false`,
`loss_scale: all`, packing, a cosine learning-rate schedule, explicit validation,
and checkpoints. Replace `model` with a compatible checkpoint or architecture and
`datasets` with your corpus. Reduce sequence/batch size if full-model training
exceeds memory, or use the [offload and multi-GPU recipes](../training/README.md).
