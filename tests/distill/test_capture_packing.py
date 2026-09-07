"""Packed-document isolation during teacher capture: CPU, no GPU, no downloads.

`distill-capture` packs several documents into one `sequence_len` window. The
teacher must not let a packed document attend to its neighbours, or the stored
distribution is conditioned on text that will never be present at training time.

Isolation is transformers' own packed-sequence support, driven by the shard's
`position_ids` resets. These tests guard *our usage* of it, which has two sharp
edges that are easy to lose in a refactor:

1. `use_cache=False` is load-bearing. With a cache allocated,
   `_preprocess_mask_arguments` skips packed detection and the teacher attends
   straight across document boundaries.
2. Passing a hand-rolled 4D `attention_mask` instead short-circuits *both* the
   causal and the sliding-window mask builders, so a sliding-window teacher
   silently loses its sliding layers.

Gold is each document run alone, which is by definition what capture should
store.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from surogate.distill.packed import teacher_logits, verify_isolation

# Shaped so document 2 is a usable isolation probe: preceded by >=
# MIN_PRECEDING_TOKENS of other text (leakage is invisible without it) and at
# least MIN_PROBE_DOC_LEN long.
DOC_LENS = [12, 20, 16]
SEQ_LEN = sum(DOC_LENS)
OFFSETS = [sum(DOC_LENS[:i]) for i in range(len(DOC_LENS))]

POS = torch.cat([torch.arange(n) for n in DOC_LENS]).unsqueeze(0)
IDS = torch.arange(1, SEQ_LEN + 1).unsqueeze(0) % 200


def _gold(model):
    """Each document forwarded on its own."""
    with torch.inference_mode():
        return torch.cat(
            [model(input_ids=IDS[:, s : s + n]).logits[0] for s, n in zip(OFFSETS, DOC_LENS)]
        )


def _packed(model):
    """The production call. If `use_cache=False` is dropped from
    `teacher_logits`, these tests go red."""
    with torch.inference_mode():
        return teacher_logits(model, IDS, POS)[0]


@pytest.fixture(scope="module")
def qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=512, attn_implementation="sdpa",
    )
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def gemma3_sliding():
    """A teacher with sliding-window layers, where a 4D mask would break isolation."""
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    cfg = Gemma3TextConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=64, sliding_window=4,
        layer_types=["sliding_attention", "full_attention"],
        attn_implementation="sdpa",
    )
    torch.manual_seed(0)
    return Gemma3ForCausalLM(cfg).eval()


def test_packed_capture_matches_unpacked_gold(qwen3):
    """The load-bearing assertion: packed capture reproduces each document alone."""
    torch.testing.assert_close(_packed(qwen3), _gold(qwen3), atol=1e-4, rtol=1e-4)


def test_use_cache_must_be_false(qwen3):
    """Without use_cache=False the packed detection is skipped and isolation is
    silently lost. Nothing else in the forward signals this, so it is pinned."""
    with torch.inference_mode():
        leaky = qwen3(input_ids=IDS, position_ids=POS, use_cache=True).logits[0]
    assert not torch.allclose(leaky, _gold(qwen3), atol=1e-4, rtol=1e-4)


def test_verify_isolation_passes_on_an_isolated_forward(qwen3):
    verify_isolation(qwen3, IDS, POS, _packed(qwen3).unsqueeze(0))


def test_verify_isolation_refuses_a_leaking_forward(qwen3):
    """The check must catch the real failure, whatever caused it.

    A cache reaching the forward is the cause we know about; the check looks at
    the property instead, so it also catches causes we have not thought of.
    """
    with torch.inference_mode():
        leaky = qwen3(input_ids=IDS, position_ids=POS, use_cache=True).logits
    with pytest.raises(RuntimeError, match="isolation check failed"):
        verify_isolation(qwen3, IDS, POS, leaky)


def test_verify_isolation_catches_a_leak_hidden_by_an_unbalanced_pack(qwen3):
    """A short document followed by a long one hides a total leak from a naive
    average, which is why only the probe's opening tokens are scored.

    Contamination concentrates at a document's start: deep inside a long document
    the leaked neighbours are a vanishing fraction of the context. Averaging over
    the whole document therefore reports ~0.89 for a forward with isolation
    COMPLETELY lost, which would clear MIN_TOPK_AGREEMENT. This test pins both the
    refusal and the reason for it.
    """
    from surogate.distill.packed import MIN_TOPK_AGREEMENT, PROBE_SCORE_POSITIONS

    a, b = 12, 200
    pos = torch.cat([torch.arange(a), torch.arange(b)]).unsqueeze(0)
    ids = torch.arange(1, a + b + 1).unsqueeze(0) % 200

    with torch.inference_mode():
        leaky = qwen3(input_ids=ids, position_ids=pos, use_cache=True).logits
        alone = qwen3(input_ids=ids[:, a:], use_cache=False).logits[0]

    def agreement(window):
        x = alone[window].topk(64, -1).indices
        y = leaky[0, a:][window].topk(64, -1).indices
        return float((x.unsqueeze(-1) == y.unsqueeze(-2)).any(-1).float().mean())

    whole_doc = agreement(slice(None))
    leading = agreement(slice(0, PROBE_SCORE_POSITIONS))
    assert whole_doc > MIN_TOPK_AGREEMENT, (
        f"premise of this test is gone: whole-document scoring no longer hides the "
        f"leak ({whole_doc:.3f})"
    )
    assert leading < MIN_TOPK_AGREEMENT

    with pytest.raises(RuntimeError, match="isolation check failed"):
        verify_isolation(qwen3, ids, pos, leaky)


def test_verify_isolation_searches_past_an_unprobeable_first_row(qwen3):
    """A shard whose first row holds one long document must not go unverified.

    The probe is taken from whichever row of the batch first yields one, not from
    row 0 -- otherwise every remaining row of that shard is captured with no check
    at all.
    """
    row0 = torch.arange(SEQ_LEN).unsqueeze(0)  # one document, no probe available
    pos = torch.cat([row0, POS], dim=0)
    ids = torch.cat([IDS, IDS], dim=0)
    with torch.inference_mode():
        leaky = qwen3(input_ids=ids, position_ids=pos, use_cache=True).logits
    with pytest.raises(RuntimeError, match="isolation check failed"):
        verify_isolation(qwen3, ids, pos, leaky)


def test_verify_isolation_skips_rows_with_nothing_packed(qwen3):
    """A plain-arange row is one document: no cross-document conditioning to
    check, and no probe to take. Must not raise."""
    plain = torch.arange(SEQ_LEN).unsqueeze(0)
    with torch.inference_mode():
        logits = teacher_logits(qwen3, IDS, plain)
    verify_isolation(qwen3, IDS, plain, logits)


def test_document_starts_matches_the_runtime_rule():
    """Boundary is 'position did not advance by one', the rule the compiled
    trainer uses on these same shards -- not merely 'position == 0'."""
    from surogate.distill.packed import _document_starts

    assert _document_starts(torch.tensor([0, 1, 2, 0, 1])) == [0, 3]
    assert _document_starts(torch.tensor([3, 4, 5, 0, 1])) == [0, 3]  # row opens mid-doc
    assert _document_starts(torch.tensor([0, 1, 2, 5, 6])) == [0, 3]  # jump, no reset
    assert _document_starts(torch.tensor([0, 1, 2, 3, 4])) == [0]


def test_sliding_window_teacher_is_isolated(gemma3_sliding):
    """Guards against reintroducing a hand-rolled 4D attention_mask: it would
    override the sliding-window mask builder and this test would fail."""
    torch.testing.assert_close(
        _packed(gemma3_sliding), _gold(gemma3_sliding), atol=1e-4, rtol=1e-4
    )
