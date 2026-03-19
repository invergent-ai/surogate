"""Tests for the native C++ tokenizer (surogate._surogate.Tokenizer).

Compares output against HuggingFace tokenizers to ensure correctness.
"""
import os
import sys
import pytest

# Try to import the native module
try:
    from surogate import _surogate
    HAS_NATIVE = hasattr(_surogate, "Tokenizer")
except ImportError:
    HAS_NATIVE = False

# Try to import HuggingFace tokenizers for comparison
try:
    from transformers import AutoTokenizer
    HAS_HF = True
except ImportError:
    HAS_HF = False


def find_qwen3_model_dir():
    """Find a Qwen3 model in the HuggingFace cache."""
    import glob
    patterns = [
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json"),
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-FP8/snapshots/*/tokenizer.json"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return os.path.dirname(matches[0])
    return None


MODEL_DIR = find_qwen3_model_dir()


@pytest.mark.skipif(not HAS_NATIVE, reason="Native tokenizer not built")
@pytest.mark.skipif(MODEL_DIR is None, reason="No Qwen3 model in HF cache")
class TestNativeTokenizer:
    """Test the native C++ tokenizer."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tok = _surogate.Tokenizer.from_pretrained(MODEL_DIR)

    def test_vocab_size(self):
        assert self.tok.vocab_size > 100_000

    def test_special_token_ids(self):
        assert self.tok.eos_token_id >= 0
        assert self.tok.pad_token_id >= 0

    def test_encode_simple(self):
        ids = self.tok.encode("Hello, world!")
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_decode_roundtrip(self):
        text = "Hello, world! This is a test."
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        assert decoded == text

    def test_encode_empty(self):
        ids = self.tok.encode("")
        assert ids == []

    def test_encode_ordinary(self):
        ids = self.tok.encode_ordinary("Hello")
        assert len(ids) > 0

    def test_encode_with_special_tokens(self):
        ids = self.tok.encode_with_special_tokens("Hello<|im_end|>")
        assert self.tok.eos_token_id in ids

    def test_encode_ordinary_no_special(self):
        ids = self.tok.encode_ordinary("<|im_end|>")
        # Should NOT contain the special token as a single ID
        assert self.tok.eos_token_id not in ids

    def test_is_special_token(self):
        assert self.tok.is_special_token(self.tok.eos_token_id)
        assert not self.tok.is_special_token(0)  # '!' is not special

    def test_encode_batch(self):
        texts = ["Hello", "World", "Test"]
        batch = self.tok.encode_batch(texts)
        assert len(batch) == 3
        for ids in batch:
            assert len(ids) > 0

    def test_decode_single_token(self):
        # Token 0 should be '!' in Qwen3
        decoded = self.tok.decode_single_token(0)
        assert len(decoded) > 0

    def test_special_token_name(self):
        eos = self.tok.special_token("eos_token")
        assert len(eos) > 0  # Should be something like "<|im_end|>"

    def test_unicode_text(self):
        text = "Hello 你好世界 🌍"
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        assert decoded == text

    def test_numbers(self):
        text = "The year is 2025 and pi is 3.14159"
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        assert decoded == text

    def test_whitespace(self):
        text = "  multiple   spaces   "
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        assert decoded == text

    def test_newlines(self):
        text = "line1\nline2\nline3"
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        assert decoded == text

    def test_code(self):
        text = 'def hello():\n    print("Hello, world!")\n'
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        assert decoded == text


@pytest.mark.skipif(not HAS_NATIVE, reason="Native tokenizer not built")
@pytest.mark.skipif(not HAS_HF, reason="HuggingFace transformers not installed")
@pytest.mark.skipif(MODEL_DIR is None, reason="No Qwen3 model in HF cache")
class TestNativeVsHuggingFace:
    """Compare native tokenizer output against HuggingFace reference."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.native = _surogate.Tokenizer.from_pretrained(MODEL_DIR)
        self.hf = AutoTokenizer.from_pretrained(MODEL_DIR)

    TEST_STRINGS = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Hello 你好世界 🌍",
        "def foo():\n    return 42\n",
        "   spaces   and\ttabs\n\nnewlines",
        "It's a test! Don't you think?",
        "123 + 456 = 579",
        "UPPERCASE lowercase MiXeD",
        "",
        " ",
        "a",
        "The year is 2025 and pi is 3.14159265358979323846.",
        "<html><body>Hello</body></html>",
        "user@example.com",
        "https://example.com/path?key=value&foo=bar",
        'She said "Hello" and he replied \'Hi\'.',
    ]

    @pytest.mark.parametrize("text", TEST_STRINGS)
    def test_encode_matches(self, text):
        native_ids = self.native.encode_ordinary(text)
        hf_ids = self.hf.encode(text, add_special_tokens=False)
        assert native_ids == hf_ids, (
            f"Mismatch for {text!r}:\n"
            f"  native ({len(native_ids)}): {native_ids[:20]}...\n"
            f"  hf     ({len(hf_ids)}): {hf_ids[:20]}..."
        )

    @pytest.mark.parametrize("text", TEST_STRINGS)
    def test_decode_matches(self, text):
        if not text:
            return  # Skip empty
        native_ids = self.native.encode_ordinary(text)
        native_decoded = self.native.decode(native_ids)
        hf_decoded = self.hf.decode(native_ids)
        assert native_decoded == hf_decoded, (
            f"Decode mismatch for {text!r}:\n"
            f"  native: {native_decoded!r}\n"
            f"  hf:     {hf_decoded!r}"
        )

    def test_vocab_size_matches(self):
        # Allow some slack for added tokens
        assert abs(self.native.vocab_size - self.hf.vocab_size) <= 100

    def test_special_tokens_match(self):
        if self.hf.eos_token_id is not None:
            assert self.native.eos_token_id == self.hf.eos_token_id
        if self.hf.pad_token_id is not None:
            assert self.native.pad_token_id == self.hf.pad_token_id
