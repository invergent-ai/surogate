"""Preference-pair tokenization: response-only masks, shifted targets, pairing."""

import numpy as np
import pytest

from surogate.dpo.data import tokenize_preference_pairs


class FakeTok:
    """Minimal whitespace tokenizer: token id = ord-sum of the word, +1/+2 specials."""

    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        ids = [(abs(hash(w)) % 5000) + 10 for w in text.split()]
        if add_special_tokens:
            ids = [1] + ids  # a BOS-like token
        return {"input_ids": ids}


class FakeChatTok(FakeTok):
    def __init__(self):
        self.rendered_modes = []

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking=None):
        assert add_generation_prompt and tokenize
        self.rendered_modes.append(enable_thinking)
        return [1, 101 if enable_thinking else 102]


class FakeMappingChatTok(FakeChatTok):
    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking=None):
        return {"input_ids": [[1, 101 if enable_thinking else 102]]}


def test_masks_and_targets_and_pairing():
    tok = FakeTok()
    rows = [{"prompt": "scrie un cuvant", "chosen": "mergeam acasa", "rejected": "mergeram acasa"}]
    b = tokenize_preference_pairs(rows, tok, max_len=32)

    assert b.n_pairs == 1
    assert b.n_seq == 2
    # Width tracks the data, not the `max_len` cap: these rows are far short
    # of 32, and allocating at the cap was the padding waste bug 26 removes.
    assert b.input_ids.shape == (2, int(b.seq_len.max()))

    for k in range(2):
        L = int(b.seq_len[k])
        # response tokens (the 2-word continuation) are scored; prompt + pad are not.
        assert b.loss_mask[k, :L].sum() == 2
        assert b.loss_mask[k, L:].sum() == 0  # padding unscored
        assert b.loss_mask[k, 0] == 0  # first (prompt/BOS) token unscored
        assert b.loss_mask[k, L - 1] == 1  # LAST response token IS scored
        # targets are input_ids shifted left by one within the real span.
        assert np.array_equal(b.targets[k, : L - 1], b.input_ids[k, 1:L])
        # position ids restart at 0 per row.
        assert b.position_ids[k, 0] == 0 and b.position_ids[k, L - 1] == L - 1

    # chosen (row 0) and rejected (row 1) differ on the changed word.
    assert not np.array_equal(b.input_ids[0], b.input_ids[1])


def test_drops_rows_that_do_not_fit():
    tok = FakeTok()
    rows = [
        {
            "prompt": "a b c d e f",
            "chosen": "x",
            "rejected": "y",
        },  # fits in max_len=4? prompt too long -> response kept via left-trunc
        {"prompt": "p", "chosen": "ok", "rejected": "no"},
    ]
    # max_len=3 forces left-truncation; response (1 token) must survive.
    b = tokenize_preference_pairs(rows, tok, max_len=3)
    assert b.n_pairs >= 1
    for k in range(b.n_seq):
        L = int(b.seq_len[k])
        assert b.loss_mask[k, :L].sum() >= 1  # at least one response token survives


def test_raises_when_response_empty():
    tok = FakeTok()
    # An empty response leaves no scored token, so the pair is dropped; with every
    # row dropped, tokenization raises rather than emit a zero-pair batch.
    with pytest.raises(ValueError, match="no preference pair"):
        tokenize_preference_pairs([{"prompt": "hello world", "chosen": "", "rejected": ""}], tok, max_len=32)


def test_chat_pair_can_request_thinking_generation_prefix():
    tok = FakeChatTok()
    rows = [
        {
            "prompt": [{"role": "user", "content": "Calculează."}],
            "chosen": "corect",
            "rejected": "greșit",
            "enable_thinking": True,
        }
    ]
    batch = tokenize_preference_pairs(rows, tok, max_len=16)

    assert batch.n_pairs == 1
    assert tok.rendered_modes == [True, True]


def test_chat_template_mapping_output_is_normalized_to_token_ids():
    tok = FakeMappingChatTok()
    rows = [
        {
            "prompt": [{"role": "user", "content": "Calculează."}],
            "chosen": "corect",
            "rejected": "greșit",
            "enable_thinking": True,
        }
    ]
    batch = tokenize_preference_pairs(rows, tok, max_len=16)

    assert batch.n_pairs == 1
    assert batch.input_ids[0, :2].tolist() == [1, 101]


def test_span_mask_scores_only_disjoint_edits():
    tok = FakeTok()
    rows = [
        {
            "prompt": "cerere",
            "chosen": "text corect între formă bună final",
            "rejected": "text greșit între formă rea final",
        }
    ]

    batch = tokenize_preference_pairs(rows, tok, max_len=32, span_mask=True)

    expected = [3, 6]
    for row in range(2):
        assert np.flatnonzero(batch.loss_mask[row]).tolist() == expected


def test_span_mask_keeps_surviving_edits_after_left_truncation():
    tok = FakeTok()
    rows = [
        {
            "prompt": "prompt foarte lung care va dispărea",
            "chosen": "unu corect trei bun",
            "rejected": "unu greșit trei rău",
        }
    ]

    batch = tokenize_preference_pairs(rows, tok, max_len=4, span_mask=True)

    for row in range(2):
        assert batch.seq_len[row] == 4
        assert np.flatnonzero(batch.loss_mask[row]).tolist() == [1, 3]


def test_span_mask_drops_pair_when_only_one_side_has_surviving_edit_tokens():
    tok = FakeTok()
    rows = [{"prompt": "p", "chosen": "shared", "rejected": "extra shared"}]

    with pytest.raises(ValueError, match="no preference pair"):
        tokenize_preference_pairs(rows, tok, max_len=8, span_mask=True)


# ── padding must not look like document boundaries ─────────────────


def test_position_ids_advance_across_padding():
    """A padded row must read as ONE document to the engine, not as one per
    pad token.

    `compute_doc_masking` starts a new document wherever a position id fails to
    advance by one, and the flash-varlen backward sizes `dq_accum` per
    document, so a zero-filled pad tail made the allocation scale with padding:
    4027 MB against a 1298 MB arena for 101 real tokens in 4096 slots.

    The fixture must contain rows of DIFFERING length. Rows are trimmed to the
    longest one, so a uniform fixture ends up with no padding at all and this
    would assert nothing.
    """
    tok = FakeTok()
    rows = [
        {"prompt": "un prompt ceva mai lung aici", "chosen": "raspuns lung", "rejected": "raspuns scurt"},
        {"prompt": "scurt", "chosen": "da", "rejected": "nu"},
    ]
    b = tokenize_preference_pairs(rows, tok, max_len=2048)

    padding = [b.width - int(L) for L in b.seq_len]
    assert max(padding) > 0, "fixture must actually pad, or this test is vacuous"

    for k in range(b.n_seq):
        # Every step is exactly one, so the engine sees a single document
        # spanning the row however much of it is padding.
        assert np.all(np.diff(b.position_ids[k]) == 1)
        # And the padding stays out of the loss, as it always did.
        L = int(b.seq_len[k])
        assert b.loss_mask[k, L:].sum() == 0
        assert np.all(b.targets[k, L:] == 0)


# ── the batch is sized to the data, not to the config ──────────────


def test_batch_width_tracks_the_data_not_the_cap():
    """A 50-token pair used to cost a 2048-token forward, three times over:
    the policy pass, the frozen reference pass, and the backward."""
    tok = FakeTok()
    rows = [{"prompt": "scrie un cuvant", "chosen": "mergeam acasa", "rejected": "mergeram acasa"}]
    b = tokenize_preference_pairs(rows, tok, max_len=2048)

    longest = int(b.seq_len.max())
    assert b.width == longest, f"width {b.width} should track the longest row {longest}"
    assert b.input_ids.shape == (b.n_seq, longest)
    assert b.position_ids.shape == (b.n_seq, longest)
    assert b.loss_mask.shape == (b.n_seq, longest)


def test_width_never_exceeds_the_cap():
    """Trimming must not turn the cap into a suggestion. Note rows longer than
    `max_len` are left-truncated to it rather than dropped, so what this checks
    is that the realised width still honours the ceiling."""
    tok = FakeTok()
    rows = [
        {"prompt": "a b c d e f g h", "chosen": "x y z", "rejected": "p q r"},
        {"prompt": "scurt", "chosen": "da", "rejected": "nu"},
    ]
    b = tokenize_preference_pairs(rows, tok, max_len=4)

    assert b.width <= 4
    assert int(b.seq_len.max()) <= 4


def test_padding_still_present_when_rows_differ_in_length():
    """Trimming is to the longest row, not per row: shorter rows keep their
    padding, and it still must not read as document boundaries."""
    tok = FakeTok()
    rows = [
        {"prompt": "un prompt ceva mai lung aici", "chosen": "raspuns lung", "rejected": "raspuns scurt"},
        {"prompt": "scurt", "chosen": "da", "rejected": "nu"},
    ]
    b = tokenize_preference_pairs(rows, tok, max_len=2048)

    lengths = {int(x) for x in b.seq_len}
    assert len(lengths) > 1, "fixture must contain rows of differing length"
    assert b.width == max(lengths)
    for k in range(b.n_seq):
        assert np.all(np.diff(b.position_ids[k]) == 1)


# ── the width must keep the divisibility the config was validated for ──


def test_width_is_rounded_up_to_the_requested_multiple():
    """`sequence_len` is validated against `lmhead_chunks` at config time; a
    trimmed width has to carry that forward itself.

    The fused lm-head splits `B * T` into `lmhead_chunks` equal nano-batches by
    truncating division and runs exactly that many, so a non-divisible width
    silently drops the remainder. In the reference forward those tokens come
    back as logprob 0.0, and because the width is the *longest* row, the dropped
    tail belongs to a row with real scored tokens: wrong margins, no error.
    """
    tok = FakeTok()
    rows = [
        {"prompt": "un prompt ceva mai lung aici", "chosen": "raspuns lung", "rejected": "raspuns scurt"},
        {"prompt": "scurt", "chosen": "da", "rejected": "nu"},
    ]
    natural = tokenize_preference_pairs(rows, tok, max_len=2048)
    aligned = tokenize_preference_pairs(rows, tok, max_len=2048, width_multiple=16)

    assert aligned.width % 16 == 0
    assert aligned.width >= natural.width
    assert aligned.width - natural.width < 16  # rounded up, not inflated


def test_alignment_never_exceeds_the_cap():
    """Rounding up must not push the width past `max_len`, whatever multiple is
    asked for."""
    tok = FakeTok()
    rows = [{"prompt": "scrie un cuvant", "chosen": "mergeam acasa", "rejected": "mergeram acasa"}]
    b = tokenize_preference_pairs(rows, tok, max_len=8, width_multiple=64)

    assert b.width <= 8


def test_padding_from_alignment_is_still_one_document():
    """The pad cells alignment adds are padding like any other, and must not
    read as document boundaries."""
    tok = FakeTok()
    rows = [{"prompt": "scrie un cuvant", "chosen": "mergeam acasa", "rejected": "mergeram acasa"}]
    b = tokenize_preference_pairs(rows, tok, max_len=2048, width_multiple=16)

    assert b.width > int(b.seq_len.max()), "alignment must have added padding here"
    for k in range(b.n_seq):
        assert np.all(np.diff(b.position_ids[k]) == 1)
