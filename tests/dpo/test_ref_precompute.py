"""Reference-logprob sidecar: cache key invalidation + save/load roundtrip.

(The full precompute needs a built trainer + GPU; it is exercised by the
end-to-end smoke test in Task 7.)
"""

import numpy as np

from surogate.dpo.data import load_ref_sidecar, rows_digest, save_ref_sidecar, sidecar_hash


def test_hash_changes_with_model_maxlen_and_rows():
    base = dict(model="A", max_len=64, n_rows=10, rows_digest="d1")
    h = sidecar_hash(**base)
    assert h != sidecar_hash(**{**base, "model": "B"})
    assert h != sidecar_hash(**{**base, "max_len": 128})
    assert h != sidecar_hash(**{**base, "n_rows": 11})
    assert h != sidecar_hash(**{**base, "rows_digest": "d2"})
    assert h == sidecar_hash(**base)  # deterministic


def test_rows_digest_is_content_sensitive():
    a = [{"prompt": "p", "chosen": "c", "rejected": "r"}]
    b = [{"prompt": "p", "chosen": "c", "rejected": "DIFFERENT"}]
    assert rows_digest(a) != rows_digest(b)
    assert rows_digest(a) == rows_digest(list(a))


def test_sidecar_roundtrip_and_key_mismatch(tmp_path):
    ref = np.random.RandomState(0).randn(6, 8).astype(np.float32)
    path = str(tmp_path / "ref.npz")
    save_ref_sidecar(path, "key123", ref)

    got = load_ref_sidecar(path, "key123")
    assert got is not None and np.allclose(got, ref)

    # wrong key => None (forces recompute), never silently wrong references.
    assert load_ref_sidecar(path, "OTHER") is None
    # missing file => None.
    assert load_ref_sidecar(str(tmp_path / "nope.npz"), "key123") is None
