# `qwen3_5` Python reference

One reference for the whole architecture: an interleaved gated-delta-net / full-attention decoder
with a dense MLP, an MTP draft block, and an optional vision tower. This is the complete Text,
Vision, MTP, sampling, state, and weight-residency reference over a native `.sinfer` artifact. It
uses typed artifact bindings and remains independent from the C++ Engine implementation.

Size is data, not code. The binding reads the decoder's dimensions from the artifact's own object
shapes and its declared `geometry`, so the 0.8B, the 27B and every checkpoint between them run the
same program. It also reads which projections a checkpoint fuses: a 27B-class export stores
`attention/query_key` beside `attention/gate_value` and `gdn/query_key` beside `gdn/value_z`, a
K-quant GGUF export stores `gdn/query_key_value` beside `gdn/z`, and the other exports store the
one fused parent of each. Which objects the artifact contains decides that -- never its size.

It does not need the original Hugging Face checkpoint at inference time. `Frontend` materializes the
tokenizer, chat template, generation defaults, and image/video processor resources embedded in the
artifact, then delegates those functions to Transformers.

## Run

Install the target dependencies from `requirements.txt`, then run:

```bash
python3 \
  -m tools.reference.qwen3_5 \
  --weights out/qwen3_6_27b.sinfer \
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
- `--prefill-chunk N`, which otherwise follows the artifact's own schedule chunk;
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
