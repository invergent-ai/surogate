# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# Tests for self-contained GGUF frontend extraction
# (surogate/serve/gguf/frontend.py).
#
# Offline tests exercise the reconstruction against a real GGUF when one is
# present under models/ (skipped otherwise). The equivalence test against the
# official Hugging Face tokenizer additionally needs network and is marked
# accordingly — it is the correctness gate: the reconstructed tokenizer must
# encode identically to the official one.

import json
from pathlib import Path

import pytest

gguf_mod = pytest.importorskip("gguf")
tokenizers = pytest.importorskip("tokenizers")

from gguf import GGUFReader

from surogate.serve.gguf.frontend import (
    extract_chat_template,
    extract_tokenizer_json,
    synthesize_tokenizer_config,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_GGUF = _REPO_ROOT / "models" / "Qwen3.5-0.8B-Q8_0.gguf"
_OFFICIAL_REPO = "Qwen/Qwen3.5-0.8B"  # dev-time comparison source only

_BATTERY = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)",
    "Ana are mere și pere — orthographe française, naïve œuvre.",
    "日本語のテキスト、中文文本，한국어 텍스트",
    "  leading spaces\tand\ttabs\r\nwindows newlines\n\n\n",
    "I'm can't won't SHE'S I'LL 'd 've",
    "<think>reasoning</think><|im_start|>user<|im_end|>",
    "1234567890 3.14159 0xDEADBEEF 1e-9",
    "emoji 🎉🚀 mixed with ASCII and ½ fractions ∑∫∂",
]

needs_gguf = pytest.mark.skipif(not _GGUF.is_file(), reason=f"no test GGUF at {_GGUF}")


@pytest.fixture(scope="module")
def reader():
    return GGUFReader(str(_GGUF), "r")


@pytest.fixture(scope="module")
def reconstructed(reader, tmp_path_factory):
    tok_json = extract_tokenizer_json(reader)
    path = tmp_path_factory.mktemp("frontend") / "tokenizer.json"
    path.write_text(json.dumps(tok_json, ensure_ascii=False))
    return tok_json, tokenizers.Tokenizer.from_file(str(path))


@needs_gguf
def test_reconstruction_loads_and_round_trips(reconstructed):
    tok_json, tok = reconstructed
    assert len(tok_json["model"]["vocab"]) > 100_000
    assert tok_json["added_tokens"], "control/user-defined tokens must be surfaced"
    for s in _BATTERY:
        ids = tok.encode(s, add_special_tokens=False).ids
        assert ids, s
        # Byte-level BPE decode must round-trip exactly (keep special tokens —
        # decode() defaults to skipping them, which is correct behavior but
        # not what a byte-exactness check wants).
        assert tok.decode(ids, skip_special_tokens=False) == s


@needs_gguf
def test_chat_template_and_config_extracted(reader):
    template = extract_chat_template(reader)
    assert template and "{%" in template
    cfg = synthesize_tokenizer_config(reader, "qwen35")
    assert cfg["eos_token"] and cfg["chat_template"] == template
    assert cfg["add_bos_token"] is False


@needs_gguf
@pytest.mark.network
def test_encode_equivalence_vs_official(reconstructed):
    from huggingface_hub import hf_hub_download

    try:
        official_path = hf_hub_download(_OFFICIAL_REPO, "tokenizer.json")
    except Exception as exc:  # offline CI
        pytest.skip(f"official tokenizer unavailable: {exc}")
    _, mine = reconstructed
    official = tokenizers.Tokenizer.from_file(official_path)
    for s in _BATTERY:
        assert (
            mine.encode(s, add_special_tokens=False).ids
            == official.encode(s, add_special_tokens=False).ids
        ), f"encode divergence on {s!r}"


@needs_gguf
@pytest.mark.network
def test_chat_template_byte_identical_to_official(reader):
    from huggingface_hub import hf_hub_download

    try:
        cfg_path = hf_hub_download(_OFFICIAL_REPO, "tokenizer_config.json")
        official = json.loads(Path(cfg_path).read_text()).get("chat_template")
        if official is None:
            official = Path(hf_hub_download(_OFFICIAL_REPO, "chat_template.jinja")).read_text()
    except Exception as exc:
        pytest.skip(f"official template unavailable: {exc}")
    assert extract_chat_template(reader) == official
