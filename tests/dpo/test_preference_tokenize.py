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
    assert b.input_ids.shape == (2, 32)

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
