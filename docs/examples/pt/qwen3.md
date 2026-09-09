# Qwen3 pretraining

[The maintained pretraining examples](../../../examples/pt/README.md) demonstrate
random initialization with NorMuon and continued pretraining with AdamW 8-bit on raw text.

```bash
surogate pt examples/pt/qwen3.yaml
surogate sft examples/pt/qwen3-continue.yaml
```

`pt` always initializes from scratch. Continued pretraining uses `sft` with
`lora: false` and `loss_scale: all`. Both examples use the tiny Shakespeare corpus
to demonstrate the workflow, with explicit validation and checkpointing.
