# Qwen3.6-27B Vision parity tool

This tool compares the independent `.ninfer` Python reference with the source BF16 Vision tower at
matching semantic boundaries.

Compare quantized artifact Vision activations with the source BF16 tower:

```bash
python -m tools.parity.qwen3_6_27b.vision \
  --weights out/qwen3_6_27b.ninfer \
  --model-dir /path/to/Qwen3.6-27B/base-hf-bf16 \
  --messages messages.json
```

The diagnostic reports numerical differences directly; it does not materialize an activation dump
or require exact generated-token equality between independent implementations.
