"""Bundled token ranks need provenance; output-head padding is not a tokenizer."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from surogate.serve.convert.common.draft_head import compute_shortlist
from surogate.serve.convert.qwen3_5 import draft_head


def fixture(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "tokenizer.json").write_text(json.dumps({
        "model": {"vocab": {"a": 0, "b": 1, "c": 3, "d": 6}},
        "added_tokens": [{"id": 7, "content": "stop", "special": True}],
    }))
    (root / "tokenizer_config.json").write_text(json.dumps({
        "added_tokens_decoder": {"7": {"special": True}},
    }))
    ranking = tmp_path / "ranking.train.counts.i64"
    counts = np.zeros(16, dtype="<i8")
    counts[[2, 6, 3, 15]] = [1000, 200, 100, 2000]
    counts.tofile(ranking)
    return root, ranking


def shortlist(root, ranking, *, require=True):
    return compute_shortlist(ranking, root, n=3, vocab=16, tokenizer_vocab_size=8,
                             require_tokenizer_match=require)


def test_unattested_ranking_uses_known_ids_and_special_tokens(tmp_path):
    root, ranking = fixture(tmp_path)
    result = shortlist(root, ranking)
    assert result.ranking is None
    assert result.selected.tolist() == [0, 1, 7]
    # Neither absent IDs inside the domain nor padded output rows may be proposed.
    assert set(result.selected) <= {0, 1, 3, 6, 7}


def test_matching_tokenizer_attestation_enables_frequency_ranks(tmp_path):
    root, ranking = fixture(tmp_path)
    manifest = {"vocab": 16, "tokenizer_sha256": hashlib.sha256((root / "tokenizer.json").read_bytes()).hexdigest()}
    (tmp_path / "ranking.train.manifest.json").write_text(json.dumps(manifest))
    result = shortlist(root, ranking)
    assert result.ranking == ranking
    assert result.selected.tolist() == [6, 3, 7]
    # The same ID range with a changed tokenizer is not compatible provenance.
    path = root / "tokenizer.json"
    path.write_text(path.read_text().replace('"a"', '"renamed"'))
    assert shortlist(root, ranking).selected.tolist() == [0, 1, 7]


def test_explicit_ranking_is_checked_against_actual_token_ids(tmp_path):
    root, ranking = fixture(tmp_path)
    assert shortlist(root, ranking, require=False).selected.tolist() == [6, 3, 7]
    counts = np.fromfile(ranking, dtype="<i8")
    counts[0] = -1
    counts.tofile(ranking)
    with pytest.raises(ValueError, match="nonnegative"):
        shortlist(root, ranking, require=False)


def test_default_ranking_is_not_assumed_compatible_with_a_target(tmp_path):
    root, _ = fixture(tmp_path)
    geometry = SimpleNamespace(vocab=16, token_domain=8, draft_vocab=3)
    result = draft_head.compute_shortlist(draft_head.DEFAULT_RANKING, root, geometry=geometry)
    assert result.ranking is None
    assert result.selected.tolist() == [0, 1, 7]


def test_frontend_edits_invalidate_the_serving_cache(tmp_path):
    from surogate.serve.ingest import source_fingerprint
    root, _ = fixture(tmp_path)
    (root / "config.json").write_text('{"hidden_size": 128}')
    before = source_fingerprint(root)
    tokenizer = root / "tokenizer.json"
    tokenizer.write_text(tokenizer.read_text().replace('"a"', '"z"'))
    assert source_fingerprint(root) != before
    before = source_fingerprint(root)
    (root / "generation_config.json").write_text('{"eos_token_id": 7}')
    assert source_fingerprint(root) != before
