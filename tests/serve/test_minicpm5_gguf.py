"""MiniCPM5 tokenizer metadata, including ordered splits and overlapping controls."""

import json
from types import SimpleNamespace

import pytest

tokenizers = pytest.importorskip("tokenizers")

from surogate.serve.gguf.frontend import extract_generation_config, extract_tokenizer_json
from surogate.serve.gguf.bridge import synthesised_config


def reader():
    fields = {
        "general.architecture": "llama", "tokenizer.ggml.model": "gpt2",
        "llama.embedding_length": 256, "llama.attention.head_count": 4,
        "llama.block_count": 2, "llama.attention.key_length": 128,
        "tokenizer.ggml.pre": "minicpm5",
        "tokenizer.ggml.tokens": ["<s>", "</s>", "1", "2", "3", "4", "Ġ", "ĠĠ",
                                  "123", "<|im_end|>", "<think>"],
        "tokenizer.ggml.token_type": [3, 3, 1, 1, 1, 1, 1, 1, 1, 3, 4],
        "tokenizer.ggml.merges": ["Ġ Ġ"],
        "tokenizer.ggml.eos_token_id": 1, "tokenizer.ggml.bos_token_id": 0,
    }
    return SimpleNamespace(
        get_field=lambda key: (SimpleNamespace(contents=lambda: fields[key])
                               if key in fields else None),
        kv=lambda key, default=None: fields.get(key, default), tensor=lambda _: None,
    )


def test_minicpm5_ordered_splits_and_direct_vocabulary_matches():
    data = extract_tokenizer_json(reader())
    tok = tokenizers.Tokenizer.from_str(json.dumps(data))
    # The first digit split leaves both spaces together; direct vocabulary matches
    # can encode "123" even though the GGUF declares no digit merge rules.
    assert tok.encode("  1234", add_special_tokens=False).ids == [7, 8, 5]
    assert tok.decode([7, 8, 5]) == "  1234"
    assert data["normalizer"] is None
    assert tok.token_to_id("<s>") == 0
    assert tok.encode("<s><think>123<|im_end|>", add_special_tokens=False).ids == [0, 10, 8, 9]


def test_minicpm5_recovers_chat_stop_id_from_vocabulary():
    assert extract_generation_config(reader()) == {"eos_token_id": [1, 9], "bos_token_id": 0}
    config = synthesised_config(reader(), "llama")
    assert config["eos_token_id"] == [1, 9]
    assert config["head_dim"] == 128  # explicit GGUF width differs from hidden / heads
