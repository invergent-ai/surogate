# Preference training (DPO)

[qwen3.yaml](qwen3.yaml) teaches Qwen3-0.6B to prefer corrected grammar using the
eight local [pairs](pairs.jsonl). These rows demonstrate the format and are too
small for a production fine-tune.

```bash
surogate dpo examples/dpo/qwen3.yaml
```

The reference is the frozen starting policy, evaluated inline. Only the adapter
is trained; adapter weights and gradients use FP32 to preserve the small preference
signal. Each pair occupies two rows, so `per_device_train_batch_size` must be even.
`span_mask: true` restricts the loss to differing token spans in these minimal pairs.
DPO performs its own pair packing; generic SFT packing is disabled.

To try **reference-free preference training**, copy this config, choose a different
output directory and replace its `loss` block with:

```yaml
loss:
  type: dpo
  dpo_beta: 0.1
  reference_free: true
  target_margin: 1.0
  length_norm: true
  span_mask: false
```

This skips the reference forward and optimizes a length-normalized likelihood gap.
`target_margin` requires `reference_free: true`. Length normalization is useful
when your real chosen/rejected answers differ substantially in length.

Data can come from JSONL, JSON, CSV, parquet or a Hugging Face dataset. Map custom
columns using `prompt_field`, `chosen_field`, `rejected_field`, and optionally
`enable_thinking_field`. A prompt may be text or chat messages. Replace the tiny
fixture with real preferences before extending the run.

DPO saves periodic `step_XXXXXXXX` checkpoints and a `final_adapter` directory:

```bash
surogate merge --base-model Qwen/Qwen3-0.6B \
  --checkpoint-dir ./outputs/dpo/qwen3/final_adapter --output ./outputs/dpo/merged
surogate serve ./outputs/dpo/merged
```

For more details, see [DPO](../../docs/getting-started/quickstart-dpo.md).
