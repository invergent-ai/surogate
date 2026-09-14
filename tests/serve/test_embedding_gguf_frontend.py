"""Embedding GGUF tokenizer reconstruction and tokenizer-aware preparation caching."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from surogate.serve import ingest
from surogate.serve.convert.gemma_embedding.frontend import frontend_from_gguf


@pytest.fixture
def metadata():
    return {"tokenizer.ggml.model": "llama",
            "tokenizer.ggml.tokens": ["<pad>", "<eos>", "<bos>", "<unk>", "a", "b", "ab", "▁", "  ", "\n", "\t"],
            "tokenizer.ggml.scores": [-1000.] * 4 + [0., 0., 2., 0., -1000., -1000., -1000.],
            "tokenizer.ggml.token_type": [3, 3, 3, 3, 1, 1, 1, 1, 4, 4, 4],
            "tokenizer.ggml.unknown_token_id": 3, "tokenizer.ggml.bos_token_id": 2,
            "tokenizer.ggml.eos_token_id": 1, "tokenizer.ggml.padding_token_id": 0,
            "tokenizer.ggml.add_space_prefix": False, "tokenizer.ggml.add_bos_token": True,
            "tokenizer.ggml.add_eos_token": True, "gemma-embedding.context_length": 2048}


def test_reconstruction_keeps_ids_and_whitespace(metadata):
    from sentencepiece import SentencePieceProcessor

    resources = frontend_from_gguf(SimpleNamespace(kv=metadata.get))
    tokenizer = SentencePieceProcessor(model_proto=resources["frontend/tokenizer.model"])
    assert tokenizer.vocab_size() == len(metadata["tokenizer.ggml.tokens"])
    assert (tokenizer.bos_id(), tokenizer.eos_id(), tokenizer.unk_id(), tokenizer.pad_id()) == (2, 1, 3, 0)
    assert tokenizer.encode("ab") == [6]
    assert tokenizer.encode("a  b") == [4, 8, 5]
    assert tokenizer.encode("  ") == [8]
    assert tokenizer.encode(" a\n\tb ") == [7, 4, 9, 10, 5, 7]
    assert tokenizer.encode("") == []
    config = json.loads(resources["frontend/tokenizer_config.json"])
    assert config["model_max_length"] == 2048
    assert config["bos_token"] == "<bos>" and config["eos_token"] == "<eos>"


@pytest.mark.parametrize("key,value,match", [
    ("model", "gpt2", "SentencePiece"),
    ("scores", [], "matching lengths"),
    ("token_type", [], "matching lengths"),
    ("unknown_token_id", None, "unknown_token_id"),
    ("bos_token_id", 999, "bos_token_id"),
    ("eos_token_id", True, "eos_token_id"),
    ("add_eos_token", False, "add_eos_token"),
])
def test_incompatible_tokenizer_metadata_is_rejected(metadata, key, value, match):
    metadata["tokenizer.ggml." + key] = value
    with pytest.raises(ValueError, match=match):
        frontend_from_gguf(SimpleNamespace(kv=metadata.get))


def test_encoder_uses_gguf_without_sidecars_and_tracks_overrides(tmp_path, monkeypatch):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"weights fixture")
    monkeypatch.setenv("SUROGATE_SERVE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(ingest, "_gguf_architecture", lambda _: "gemma-embedding")
    commands = []

    def convert(command, **kwargs):
        commands.append(command)
        Path(command[command.index("--out") + 1]).write_bytes(b"prepared")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ingest.subprocess, "run", convert)
    prepared = ingest.ensure_encoder_weights(str(gguf))
    assert "--frontend" not in commands[-1]
    assert prepared.read_bytes() == b"prepared"
    assert ingest.ensure_encoder_weights(str(gguf)) == prepared
    assert len(commands) == 1
    external = tmp_path / "frontend"
    external.mkdir()
    for name in ingest.ENCODER_FRONTEND_FILES:
        (external / name).write_bytes(b"custom")
    custom = ingest.ensure_encoder_weights(str(gguf), frontend=str(external))
    assert custom != prepared
    assert commands[-1][commands[-1].index("--frontend") + 1] == str(external)
    (external / "tokenizer.model").write_bytes(b"changed")
    assert ingest.ensure_encoder_weights(str(gguf), frontend=str(external)) != custom
    with pytest.raises(SystemExit, match="missing"):
        ingest.ensure_encoder_weights(str(gguf), frontend=str(tmp_path / "missing"))


def test_failed_conversion_does_not_leave_a_reusable_cache(tmp_path, monkeypatch):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"weights fixture")
    cache = tmp_path / "cache"
    monkeypatch.setenv("SUROGATE_SERVE_CACHE", str(cache))
    monkeypatch.setattr(ingest, "_gguf_architecture", lambda _: "gemma-embedding")

    def fail(command, **kwargs):
        Path(command[command.index("--out") + 1]).write_bytes(b"incomplete")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(ingest.subprocess, "run", fail)
    with pytest.raises(SystemExit, match="conversion failed"):
        ingest.ensure_encoder_weights(str(gguf))
    assert not list(cache.glob("*.sinfer*"))


def test_local_embeddinggemma_matches_llama_cpp_token_ids():
    from sentencepiece import SentencePieceProcessor
    from surogate.serve.gguf.lean import LeanGguf

    path = Path(__file__).resolve().parents[2] / "models/embeddinggemma-300M-Q8_0.gguf"
    if not path.is_file():
        pytest.skip("requires the local EmbeddingGemma GGUF")
    with LeanGguf(path) as reader:
        resources = frontend_from_gguf(reader)
    tokenizer = SentencePieceProcessor(model_proto=resources["frontend/tokenizer.model"])
    # llama.cpp /tokenize with add_special=true, parse_special=false. Includes
    # the added space token that cannot be stored literally in a SentencePiece model.
    for text, expected in [
        ("Hello world", [2, 9259, 1902, 1]),
        ("a  b", [2, 236746, 138, 236763, 1]),
        ("\n\n", [2, 108, 1]),
        ("😊 東京", [2, 240782, 86459, 1]),
        ("é e\u0301", [2, 236859, 545, 238288, 1]),
        ("<bos>", [2, 236820, 46757, 236813, 1]),
    ]:
        assert [tokenizer.bos_id(), *tokenizer.encode(text), tokenizer.eos_id()] == expected
