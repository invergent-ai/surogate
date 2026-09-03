"""The teacher forward for packed capture, and its safety check.

Kept out of `capture.py` on purpose: that module pulls in the compiled runtime
through `SFTConfig`, so nothing importing it can be exercised on CPU. The two
things here are the ones worth testing without a GPU, and the ones whose failure
is silent.

Packed documents are isolated by transformers' own packed-sequence support, which
reads `position_ids` resets.

`use_cache=False` is load-bearing, for a reason that is easy to misread from the
source. `Qwen3Model.forward` and friends declare `use_cache: bool | None = None`
and allocate a cache only under `if use_cache and past_key_values is None`, which
looks as though omitting the argument is safe. It is not: the forward carries a
`@merge_with_config_defaults` decorator (`transformers/utils/generic.py`, and
`use_cache` is the first name in its list) that substitutes `config.use_cache`
-- normally `True` -- before the body runs. A cache is then allocated,
`_preprocess_mask_arguments` sees `past_key_values is not None`, packed detection
is skipped, and the teacher attends straight across document boundaries.
Measured: omitting the argument diverges from per-document gold by 0.535,
identical to passing `use_cache=True`.
"""

from __future__ import annotations

# A correctly isolated capture reproduces the standalone document almost exactly
# (~98-100% of top-k ids agree; the rest is bf16 kernel noise). A capture that has
# lost isolation scores ~0.49-0.64 on the scored window below.
MIN_TOPK_AGREEMENT = 0.80

# Contamination concentrates at the START of a packed document: a token deep
# inside a long document has so much of its own context that a few leaked
# neighbouring tokens barely move its distribution. Averaging over a whole
# document therefore HIDES a total leak -- measured, a 482-token document
# preceded by 30 tokens scores 0.89 with isolation completely lost, which would
# pass. Scoring only the leading positions separates cleanly (1.00 vs 0.64).
PROBE_SCORE_POSITIONS = 32

# A probe document must be long enough to score, and must have enough text in
# front of it for leakage to be visible at all.
MIN_PROBE_DOC_LEN = 8
MIN_PRECEDING_TOKENS = 8

# Fixed, deliberately independent of `distillation.top_k`, which the user may set
# as low as 1. At top_k=1 the metric degenerates to argmax agreement, where two
# bf16 flips on a short probe would abort a perfectly good capture.
PROBE_TOP_K = 64


def teacher_logits(model, input_ids, position_ids):
    """One packed forward. `use_cache=False` is what keeps isolation on."""
    return model(
        input_ids=input_ids,
        position_ids=position_ids,
        use_cache=False,
    ).logits


def _document_starts(position_ids_row) -> list[int]:
    """Indices where a packed document begins, matching the runtime's rule.

    The trainer that consumes these shards treats a document as starting wherever
    the position id does not advance by exactly one
    (`csrc/.../causal_lm_execution_profile.cpp`), which also covers positions that
    restart at 0. Index 0 always begins a document: the DataLoader serves each row
    independently, so a row start is a document start even when a document spans
    the window boundary.
    """
    starts = [0]
    prev = int(position_ids_row[0])
    for i in range(1, len(position_ids_row)):
        curr = int(position_ids_row[i])
        if curr - prev != 1:
            starts.append(i)
        prev = curr
    return starts


def _find_probe(position_ids_row, seq_len: int):
    """First document in the row usable as an isolation probe, or None.

    Must not be the row's first document (only later documents can be
    contaminated), must have enough preceding text for a leak to show, and must
    be long enough to score.
    """
    starts = _document_starts(position_ids_row)
    for i in range(1, len(starts)):
        a = starts[i]
        b = starts[i + 1] if i + 1 < len(starts) else seq_len
        if a >= MIN_PRECEDING_TOKENS and b - a >= MIN_PROBE_DOC_LEN:
            return a, b
    return None


def verify_isolation(model, batch_ids, batch_position_ids, batch_logits) -> None:
    """Prove, on real data, that the packed forward isolated its documents.

    Capture cannot see isolation in its own output. If it is lost -- a cache
    reaching the forward, a transformers change, a future edit to
    `teacher_logits` -- the teacher attends across document boundaries and capture
    writes a sidecar whose stored distributions are badly wrong. Nothing
    downstream notices: `validate_sidecar` checks the header and the tokenize
    hash, not the values, so the bad sidecar is accepted and reused until
    something forces a recapture.

    So this checks the property rather than any one of its causes: take a document
    that is not first in its row, re-run its opening tokens on their own, and
    compare top-k ids against the packed forward's rows for the same positions.

    Every row of the batch is searched for a usable probe, because a shard whose
    first row happens to hold one long document would otherwise be captured
    entirely unverified.

    Costs one forward over at most `PROBE_SCORE_POSITIONS` tokens. The flash-attn
    path this replaced failed loudly when it could not isolate; this keeps that.
    """
    import torch

    seq_len = int(batch_ids.shape[1])
    found = None
    for row in range(int(batch_ids.shape[0])):
        probe = _find_probe(batch_position_ids[row], seq_len)
        if probe is not None:
            found = (row, *probe)
            break
    if found is None:
        # No row in this batch packs a scorable document behind another, so there
        # is no cross-document conditioning here that could go wrong.
        return
    row, a, b = found

    # Score only the document's opening tokens. Logits at those positions depend
    # solely on tokens [a, end), so the standalone forward needs no more than
    # that -- which also bounds the comparison below.
    end = min(b, a + PROBE_SCORE_POSITIONS)
    with torch.inference_mode():
        alone = model(input_ids=batch_ids[row, a:end].unsqueeze(0), use_cache=False).logits[0]
    packed = batch_logits[row, a:end]

    k = min(PROBE_TOP_K, int(alone.shape[-1]))
    ids_alone = alone.topk(k, dim=-1).indices
    ids_packed = packed.topk(k, dim=-1).indices
    # [L, k, k] booleans, bounded by the caps above (<= 32 x 64 x 64) rather than
    # by the user's top_k, which may be 1024 over a whole window.
    agreement = float(
        (ids_alone.unsqueeze(-1) == ids_packed.unsqueeze(-2)).any(-1).to(torch.float32).mean()
    )

    if agreement < MIN_TOPK_AGREEMENT:
        raise RuntimeError(
            f"Packed-document isolation check failed: the opening {end - a} tokens of a "
            f"document captured inside a packed window agree with the same document "
            f"captured alone on only {agreement:.1%} of their top-{k} tokens (expected "
            f">= {MIN_TOPK_AGREEMENT:.0%}). The teacher is attending across document "
            f"boundaries, so the captured logprobs would be conditioned on unrelated "
            f"neighbouring text. Likely causes: the installed transformers no longer "
            f"supports packed-sequence masking, or a cache reached the forward "
            f"(`use_cache` must be False). Refusing to write a sidecar that looks valid "
            f"but is not."
        )
