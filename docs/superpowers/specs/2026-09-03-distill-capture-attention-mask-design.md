# distill-capture: replace flash-attn varlen with a block-diagonal attention mask

## Problem

`surogate distill-capture` cannot run in any shipped image. It aborts before the
teacher is even downloaded:

```
RuntimeError: distill-capture requires flash-attention-2 for per-document attention
isolation (position_ids resets -> varlen). Install flash-attn, or pass
--allow-cross-doc-attention to accept the cross-document approximation with sdpa.
```

`_load_teacher` (`surogate/distill/capture.py:66-82`) hard-requires
flash-attention-2. `flash-attn` is not a dependency and is installed by none of
`Dockerfile.cu128/129/130`. vLLM's vendored `vllm_flash_attn` is a different
distribution and does not satisfy `transformers.utils.is_flash_attn_2_available()`.

Live-confirmed 2026-09-03 on Modal (`kd-test-1`): failed in 68 s, `$0.0151`,
`surogate sft` never invoked, zero metric rows. This blocks knowledge distillation
end to end on the platform.

## Why the obvious fix is unavailable

Adding `flash-attn` to the images is not a Dockerfile line. Dao-AILab publishes
prebuilt FA2 wheels (cp312 / linux_x86_64) for torch **2.2 through 2.10**; the
trainer images ship **torch 2.11.0+cu128**. The remaining routes all have real
costs: a source compile on every image build, a torch downgrade affecting every
training method, an indefinite upstream wait, or an engine change to accept FA3
(which `is_flash_attn_2_available()` cannot see, since FA3 ships as the separate
`flash_attn_interface` distribution).

## What flash-attn was doing, and why it matters

Tokenized shards **pack** several short documents into one `sequence_len` window
(real run: 1800 docs, mean 162 tokens, into 2048-token windows) and mark each
document start by resetting `position_ids` to 0.

The teacher must not let a packed document attend to its neighbours. If it does,
the stored distribution answers a question that will never be asked at training
time. flash-attn's varlen mode achieved this by taking document boundaries
(`cu_seq_lens`) and never computing attention across them. This is also why
`capture_shard` flattens the batch to `[1, B*S]` before the forward: that shape
plus boundaries is what triggers transformers' padding-free flash path.

## Design: let transformers build the mask

**Revised during the simplify gate.** The first implementation hand-rolled a 4D
block-diagonal mask in a new `surogate/distill/mask.py`. That module has been
deleted: `transformers >= 5.5` already does exactly this, and doing it by hand is
actively worse (see "Why not a hand-rolled mask" below).

Capture passes `position_ids` and `use_cache=False`, and transformers'
packed-sequence support (`masking_utils.find_packed_sequence_indices` ->
`packed_sequence_mask_function` -> `sdpa_mask`) derives the document boundaries
and builds the mask itself. The mask it produces is what we want. For a window
holding doc A (3 tokens) and doc B (2 tokens):

```
tokens        A1  A2  A3  B1  B2
position_ids   0   1   2   0   1      <- reset marks a new document

        A1 A2 A3 B1 B2
    A1   x  .  .  .  .
    A2   x  x  .  .  .
    A3   x  x  x  .  .
    B1   .  .  .  x  .       block diagonal: B cannot see A
    B2   .  .  .  x  x       lower triangular within each block: causal
```

Allowed iff `same_document(i, j) AND j <= i`.

This needs **no flash-attn, no torch pin and no image change**, and no mask code
of our own.

### `use_cache=False` is load-bearing

`_preprocess_mask_arguments` runs packed detection only when
`attention_mask is None and past_key_values is None`.

The trap is that reading the model source suggests omitting `use_cache` is safe:
`Qwen3Model.forward` declares `use_cache: bool | None = None` and allocates a
cache only under `if use_cache and past_key_values is None`. But the forward
carries `@merge_with_config_defaults` (`transformers/utils/generic.py:908`, with
`use_cache` first in its list), which substitutes `config.use_cache` -- normally
`True` -- before the body runs. A `DynamicCache` is then allocated, detection is
skipped, and the teacher attends across document boundaries.

Measured on transformers 5.7.0, max abs divergence from per-document gold:
`use_cache=False` **1.8e-07**, argument **omitted 0.535**, `use_cache=True`
**0.535**. An external reviewer read the signature and the `if use_cache` line
and concluded the argument was inert; it is not. Pinned by a test.

### Why not a hand-rolled mask

Passing a 4D `attention_mask` makes `_preprocess_mask_arguments` early-return it
verbatim for **both** `create_causal_mask` and `create_sliding_window_causal_mask`.
A sliding-window teacher (Gemma3, gpt-oss class) therefore loses its sliding
layers and silently gets full within-document attention. Measured on a 2-layer
Gemma3 with `layer_types=['sliding_attention','full_attention']`, against
per-document gold:

| | max abs logit error vs gold |
|---|---|
| transformers packed support | **2.76e-07** |
| hand-rolled 4D block-diagonal mask | **0.51** |

That is the same order as no isolation at all, on any sliding-window teacher, and
`tests/distill/test_capture.py` only exercises Qwen3 so it would not have caught
it. On Qwen3 the two paths are bit-identical (max diff exactly 0.0), so the
hand-rolled mask bought nothing and cost correctness elsewhere.

### Consequence: drop the flattening, which fixes bug 34 for free

The `unsqueeze(0)` flattening exists *only* to trigger the flash path. With an
explicit mask that reason is gone, so capture returns to a natural `[B, S]`
batch. `_topk_logprobs` (`capture.py:94-106`) was written to iterate one window
at a time and only degenerated into converting the whole batch to fp32 because
flattening made the batch dimension 1. Restoring `[B, S]` makes the loop do what
its docstring says. Measured cost of the current behaviour: capture peaked at
**21,569 MiB on a 24 GB L4** for a 1.7B teacher whose weights are ~3.4 GB.

## Evidence

Scored against **gold** (each document run alone, unpacked, which is by
definition what capture should store) on the top-64 id set the sidecar holds.
Four mixed documents (English, Romanian, Python source, business prose):

| packing strategy | top-64 overlap vs gold | argmax agreement |
|---|---|---|
| `fa2` varlen (today's intended path) | 98.32% | 98.72% |
| **`sdpa` + block-diagonal mask** | **98.46%** | **98.72%** |
| `sdpa`, no mask (`--allow-cross-doc-attention`) | **42.67%** | 42.95% |

Isolation with the mask is exact (max delta 0.0). Noise floor: FA2 vs plain sdpa
on a *single unpacked document*, computing identical maths, still differ by 4.08
max / 0.045 mean on logits of magnitude 33, with 98.16% top-64 overlap. So
98.46 vs 98.32 is a tie.

Cost at the real 4x2048 shape, Qwen3-1.7B on an L4:

| arm | peak | median/forward |
|---|---|---|
| `fa2`, flattened | 8967 MiB | 0.808 s |
| `sdpa` + mask, flattened | 9095 MiB | 1.397 s |
| **`sdpa` + mask, batched `[B, S]`** | **8999 MiB** | **0.952 s** |

Batched is the layout to adopt: +0.4% memory and +18% time versus FA2, and 32%
faster than masking the flattened layout.

## Decisions

1. **One attention path.** Always masked `sdpa`. No conditional FA2 fast path.
   A path that cannot be exercised in CI or in the shipped image (no torch 2.11
   wheel) is permanently untested code, which is the exact mechanism that
   produced this bug. The 18% is a one-time capture cost, not per-step training.
2. **Delete `--allow-cross-doc-attention`.** It now means "produce supervision
   that is 43% correct" and there is nothing left to fall back from.
3. **Mask memory ceiling is accepted, not guarded.** The dense mask transformers
   materialises is a **bool** `[B, 1, S, S]` (1 byte/element, confirmed against
   `create_causal_mask`): ~17 MB at the defaults (B=4, S=2048), 268 MB at
   `sequence_len` 8192, ~1073 MB at 16384 (which `sft_config` permits up to the
   model max). O(B*S^2) for structure that is O(#docs). Named here rather than hidden;
   the upgrade path is `attn_implementation="flex_attention"`, which transformers
   supports with a sparse BlockMask and which also needs no flash-attn. Add it
   only if someone hits the ceiling.
4. **Shards older than version 3 keep today's behaviour.** They carry no
   `position_ids`, so capture passes `position_ids=None` and transformers
   defaults to arange per row: one document per window, plain causal. No fake
   positions are synthesised, so `None` keeps meaning "no information".

## Non-goals

- The vLLM API capture backend (`teacher_api_base`) is untouched. It needs no
  flash-attn already and is unaffected.
- Cross-tokenizer / transplant paths are untouched.
- No image, Dockerfile or torch version change.

## Testing

`tests/distill/test_capture.py` **already implements the gold comparison** ("compares
each captured row against a reference computed by feeding EVERY DOCUMENT AS ITS
OWN SEQUENCE"). It has never run: line 25 is
`pytest.importorskip("flash_attn")`, plus `gpu` and `slow` marks. The check that
would have caught this has been dark since it was written.

1. **Un-gate it.** Remove the flash-attn skip. It still needs a GPU and weights,
   so it stays `gpu` + `slow`.
2. **Add `tests/distill/test_capture_packing.py`**, CPU, tiny random-weight
   models, no download. Three tests, guarding *our usage* rather than library
   internals: packed matches per-document gold; `use_cache=True` demonstrably
   breaks isolation; and a sliding-window teacher stays isolated, which fails if
   anyone reintroduces a hand-rolled 4D mask.

**Why gold, not isolation.** During the spike a hand-rolled mask was written
*anti-causal* (the causal term transposed). It **passed an isolation check at
delta exactly 0.0** while scoring 29.54% against gold, worse than no mask at all:
cross-document blocking was correct, only the within-document direction was
reversed. An isolation assertion cannot catch that. Delegating the mask to
transformers removes this whole bug class, but the gold comparison stays as the
assertion of record.

## Files

- `surogate/distill/capture.py` — `_load_teacher` (drop the requirement),
  `capture_shard` (pass `position_ids` + `use_cache=False`, drop flattening),
  `_topk_logprobs` (docstring), `run_capture` / `distill_capture_main` (flag
  removal).
- `surogate/cli/distill_capture.py` — remove `--allow-cross-doc-attention`.
- `tests/distill/test_capture.py` — un-gate.
- `tests/distill/test_capture_packing.py` — new CPU tests.
- `docs/guides/distillation.md` — the flash-attn requirement, the flag, and the
  troubleshooting entry all describe behaviour that no longer exists.
