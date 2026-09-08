# `qwen3_5` Python reference

This diagnostic Python reference reads model dimensions, attention placement, normalization,
and context limits from the converted artifact. The model name does not select a size.
MTP and vision are available when the checkpoint includes their weights.

The reference currently reads inline BF16 and groupwise integer weights. Other formats require
the C++ serving runtime. Artifacts created before complete checkpoint metadata was stored must
be rebuilt from the original checkpoint.

It does not need the original Hugging Face checkpoint at inference time. `Frontend` materializes the
tokenizer, chat template, generation defaults, and image/video processor resources embedded in the
artifact, then delegates those functions to Transformers.

## Run

Install the target dependencies from `requirements.txt`, then run:

```bash
python3 \
  -m tools.reference.qwen3_5 \
  --weights out/model.sinfer \
  --prompt "请简短介绍一下你自己。" --decode 512
```

The input is exactly one of `--prompt`, `--ids`, or `--messages`. Structured messages may contain
images and videos in the normal Transformers format. Thinking is enabled by default and can be
disabled with `--no-thinking`.

MTP is disabled by default. Enable one to five draft positions with
`--mtp-draft-tokens 1..5`; `--draft-head` selects the artifact's optimized proposal head. Target
verification always uses the full output head. The CLI reports round counts, per-position accepted
drafts, fallback steps, timing, memory planning, and peak CUDA allocation.

Important runtime controls include:

- `--gpu-memory auto|24GiB` and `--headroom 2GiB`;
- `--kv-dtype bf16|int8`;
- `--prefill-chunk N`, which controls the number of prompt tokens processed at once;
- `--greedy` or sampling overrides for temperature, top-p, top-k, and penalties;
- `--vision-attention-limit N`;
- `--activation-dump DIR --dump-level layer|op`.

Vision runs only for a checkpoint whose artifact ships a tower; `RefModel.vision_config` is
`None` where it does not.

Quantized Text matrices retain the decoded/packed/streamed residency plan and compiled low-bit
codec. Vision decodes large matrices one at a time and releases its weight store before Text weight
preparation. Multimodal MTP uses the composed Vision embedding for the shifted input, including at
prefill chunk boundaries.

The source-BF16 Vision comparison lives in [`tools/parity/qwen3_5/`](../../parity/qwen3_5/README.md).
