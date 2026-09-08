# `qwen3_5_moe` Python reference

This diagnostic Python reference reads dimensions, expert counts, routing settings, attention
placement, and context limits from the converted artifact. The model name does not select a
size. MTP and vision are available when the checkpoint includes their weights.

The reference reads inline BF16 and groupwise integer weights with fused text projections.
Other storage layouts require the C++ serving runtime. Artifacts created before complete
checkpoint metadata was stored must be rebuilt from the original checkpoint.

An optional DFlash companion can be inspected and included in a weight-memory plan. This
reference does not execute DFlash generation.

The reference is an independent diagnostic implementation for the registered artifact. The C++
Engine target is registered separately; this Python route is not its generated-token golden and
does not define equality across different numerical paths. It does not need the original Hugging
Face checkpoint at inference time: tokenizer, generation, template, image, and video resources are
read from the artifact.

## Run

Install the target dependencies from `requirements.txt`, then run:

```bash
python3 \
  -m tools.reference.qwen3_5_moe \
  --weights out/model.sinfer \
  --prompt "请简短介绍一下你自己。" --decode 128
```

The input is exactly one of `--prompt`, `--ids`, or `--messages`. Structured messages may contain
images and videos in the normal Transformers format. Thinking is enabled by default and can be
disabled with `--no-thinking`.

MTP is disabled by default. Enable one to five draft positions with `--mtp-draft-tokens 1..5`;
`--draft-head` selects the artifact's optimized proposal head. Target verification always uses the
full output head. The CLI reports proposal acceptance, timing, memory planning, Vision work, and
peak CUDA allocation.

Important runtime controls include:

- `--gpu-memory auto|24GiB` and `--headroom 2GiB`;
- `--kv-dtype bf16|int8`;
- `--prefill-chunk N`;
- `--greedy` or sampling overrides;
- `--vision-attention-limit N`;
- `--activation-dump DIR --dump-level layer|op`.

The sparse-MoE path resolves router ids first and materializes only the row spans of experts that
receive tokens. It never expands a complete routed bank. Vision processes each image or video item
independently, retains only its merged BF16 embeddings on the host, and releases its streaming
weight store before Text preparation. Prompt chunks transfer only the required image/video rows;
multimodal MTP uses the same composed Vision embedding for shifted inputs.

Decode, MTP proposal/verification, and both output heads use the contract's FP32-dequant Small-T
projection profile. Text/MTP prefill retains the separate BF16 MMA-weight rounding boundary.
