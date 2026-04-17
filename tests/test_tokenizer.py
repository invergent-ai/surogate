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

    @pytest.fixture(autouse=True, scope="class")
    def setup(self, request):
        request.cls.tok = _surogate.Tokenizer.from_pretrained(MODEL_DIR)

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

    @pytest.fixture(autouse=True, scope="class")
    def setup(self, request):
        request.cls.native = _surogate.Tokenizer.from_pretrained(MODEL_DIR)
        request.cls.hf = AutoTokenizer.from_pretrained(MODEL_DIR)

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
        "She said \"Hello\" and he replied 'Hi'.",
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
            f"Decode mismatch for {text!r}:\n  native: {native_decoded!r}\n  hf:     {hf_decoded!r}"
        )

    def test_vocab_size_matches(self):
        # Allow some slack for added tokens
        assert abs(self.native.vocab_size - self.hf.vocab_size) <= 100

    def test_special_tokens_match(self):
        if self.hf.eos_token_id is not None:
            assert self.native.eos_token_id == self.hf.eos_token_id
        if self.hf.pad_token_id is not None:
            assert self.native.pad_token_id == self.hf.pad_token_id

    CHAT_TEMPLATE_TESTS = [
        (
            "single turn gen",
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
            True,
        ),
        (
            "multi turn gen",
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "And 3+3?"},
            ],
            True,
        ),
        (
            "no system",
            [
                {"role": "user", "content": "Hello!"},
            ],
            True,
        ),
        (
            "training",
            [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            False,
        ),
        (
            "multi turn training",
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "Bye"},
                {"role": "assistant", "content": "Goodbye!"},
            ],
            False,
        ),
    ]

    @pytest.mark.parametrize("desc,messages,gen_prompt", CHAT_TEMPLATE_TESTS, ids=[t[0] for t in CHAT_TEMPLATE_TESTS])
    def test_apply_chat_template_matches(self, desc, messages, gen_prompt):
        native_text = self.native.apply_chat_template(messages, add_generation_prompt=gen_prompt)
        hf_text = self.hf.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen_prompt)
        assert native_text == hf_text, (
            f"Chat template mismatch for {desc}:\n  native: {native_text!r}\n  hf:     {hf_text!r}"
        )

    def test_apply_chat_template_and_encode(self):
        messages = [{"role": "user", "content": "Hello!"}]
        native_ids = self.native.apply_chat_template_and_encode(messages, add_generation_prompt=True)
        hf_text = self.hf.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        hf_ids = self.hf.encode(hf_text, add_special_tokens=False)
        assert native_ids == hf_ids


@pytest.mark.skipif(not HAS_NATIVE, reason="Native tokenizer not built")
@pytest.mark.skipif(not HAS_HF, reason="HuggingFace transformers not installed")
@pytest.mark.skipif(MODEL_DIR is None, reason="No Qwen3 model in HF cache")
class TestEncodeForTraining:
    """Test encode_for_training against manual construction from HF tokenizer."""

    @pytest.fixture(autouse=True, scope="class")
    def setup(self, request):
        request.cls.native = _surogate.Tokenizer.from_pretrained(MODEL_DIR)
        request.cls.hf = AutoTokenizer.from_pretrained(MODEL_DIR)

    def _hf_encode_for_training(self, messages, strategy="default"):
        """Reference implementation: build input_ids/labels using HF tokenizer."""
        hf = self.hf

        # Separate system messages
        first_non_system = 0
        while first_non_system < len(messages) and messages[first_non_system]["role"] == "system":
            first_non_system += 1

        # Count rounds
        pairs = []
        for i in range(first_non_system, len(messages) - 1, 2):
            pairs.append((i, i + 1))
        if not pairs:
            return [], []

        total_rounds = len(pairs)
        input_ids = []
        labels = []
        prev_text = ""

        for round_idx, (user_idx, asst_idx) in enumerate(pairs):
            is_last = round_idx == total_rounds - 1

            # Render up to user with gen_prompt=true
            msgs_to_user = messages[: user_idx + 1]
            text_user = hf.apply_chat_template(msgs_to_user, tokenize=False, add_generation_prompt=True)
            chrome = text_user[len(prev_text) :]
            if chrome:
                chrome_ids = hf.encode(chrome, add_special_tokens=False)
                input_ids.extend(chrome_ids)
                if strategy == "all":
                    labels.extend(chrome_ids)
                else:
                    labels.extend([-100] * len(chrome_ids))

            # Render up to assistant with gen_prompt=false
            msgs_to_asst = messages[: asst_idx + 1]
            text_asst = hf.apply_chat_template(msgs_to_asst, tokenize=False, add_generation_prompt=False)
            response = text_asst[len(text_user) :]
            if response:
                resp_ids = hf.encode(response, add_special_tokens=False)
                input_ids.extend(resp_ids)
                trainable = False
                if strategy == "all" or strategy == "default":
                    trainable = True
                elif strategy == "last_round":
                    trainable = is_last
                if trainable:
                    labels.extend(resp_ids)
                else:
                    labels.extend([-100] * len(resp_ids))

            prev_text = text_asst

        # Always mask first token
        if labels:
            labels[0] = -100

        return input_ids, labels

    TRAINING_TESTS = [
        (
            "single turn",
            [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        ),
        (
            "single turn with system",
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        ),
        (
            "multi turn",
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "And 3+3?"},
                {"role": "assistant", "content": "6"},
            ],
        ),
        (
            "long response",
            [
                {"role": "user", "content": "Write a poem."},
                {"role": "assistant", "content": "Roses are red,\nViolets are blue,\nSugar is sweet,\nAnd so are you."},
            ],
        ),
        (
            "unicode",
            [
                {"role": "user", "content": "Hello 你好"},
                {"role": "assistant", "content": "你好世界! 🌍"},
            ],
        ),
        (
            "code",
            [
                {"role": "user", "content": "Write hello world in Python"},
                {"role": "assistant", "content": 'def hello():\n    print("Hello, world!")\n'},
            ],
        ),
        (
            "three turns",
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well."},
                {"role": "user", "content": "Bye"},
                {"role": "assistant", "content": "Goodbye!"},
            ],
        ),
    ]

    @pytest.mark.parametrize("desc,messages", TRAINING_TESTS, ids=[t[0] for t in TRAINING_TESTS])
    def test_default_strategy(self, desc, messages):
        result = self.native.encode_for_training(messages, strategy="default")
        expected_ids, expected_labels = self._hf_encode_for_training(messages, "default")
        assert result["input_ids"] == expected_ids, (
            f"input_ids mismatch for {desc}:\n"
            f"  native ({len(result['input_ids'])}): {result['input_ids'][:20]}\n"
            f"  hf     ({len(expected_ids)}): {expected_ids[:20]}"
        )
        assert result["labels"] == expected_labels, (
            f"labels mismatch for {desc}:\n"
            f"  native ({len(result['labels'])}): {result['labels'][:20]}\n"
            f"  hf     ({len(expected_labels)}): {expected_labels[:20]}"
        )

    @pytest.mark.parametrize("desc,messages", TRAINING_TESTS, ids=[t[0] for t in TRAINING_TESTS])
    def test_last_round_strategy(self, desc, messages):
        result = self.native.encode_for_training(messages, strategy="last_round")
        expected_ids, expected_labels = self._hf_encode_for_training(messages, "last_round")
        assert result["input_ids"] == expected_ids
        assert result["labels"] == expected_labels

    @pytest.mark.parametrize("desc,messages", TRAINING_TESTS, ids=[t[0] for t in TRAINING_TESTS])
    def test_all_strategy(self, desc, messages):
        result = self.native.encode_for_training(messages, strategy="all")
        expected_ids, expected_labels = self._hf_encode_for_training(messages, "all")
        assert result["input_ids"] == expected_ids
        assert result["labels"] == expected_labels

    def test_labels_structure_default(self):
        """Verify that default strategy masks system/user and trains assistant."""
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = self.native.encode_for_training(messages, strategy="default")
        ids = result["input_ids"]
        labels = result["labels"]

        assert len(ids) == len(labels)
        assert ids[0] != -100  # first token is a real token
        assert labels[0] == -100  # but label is masked

        # At least some labels should be trainable (the assistant part)
        trainable = [l for l in labels if l != -100]
        assert len(trainable) > 0

        # At least some labels should be masked (the system/user/chrome part)
        masked = [l for l in labels if l == -100]
        assert len(masked) > 0

        # Full text should decode to the same as apply_chat_template
        full_text = self.native.apply_chat_template(messages, add_generation_prompt=False)
        full_ids = self.native.encode_with_special_tokens(full_text)
        assert ids == full_ids

    def test_last_round_masks_early_turns(self):
        """Verify last_round only trains on the final assistant response."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "content": "6"},
        ]
        result_default = self.native.encode_for_training(messages, strategy="default")
        result_last = self.native.encode_for_training(messages, strategy="last_round")

        # Same input_ids
        assert result_default["input_ids"] == result_last["input_ids"]

        # last_round should have fewer trainable tokens
        default_trainable = sum(1 for l in result_default["labels"] if l != -100)
        last_trainable = sum(1 for l in result_last["labels"] if l != -100)
        assert last_trainable < default_trainable

    def test_all_strategy_trains_everything(self):
        """Verify 'all' strategy marks (almost) all tokens as trainable."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = self.native.encode_for_training(messages, strategy="all")
        # Only labels[0] should be -100
        assert result["labels"][0] == -100
        assert all(l != -100 for l in result["labels"][1:])

    def test_empty_messages(self):
        result = self.native.encode_for_training([], strategy="default")
        assert result["input_ids"] == []
        assert result["labels"] == []

    def test_no_assistant(self):
        """Messages with no assistant should return empty."""
        messages = [{"role": "user", "content": "Hello"}]
        result = self.native.encode_for_training(messages, strategy="default")
        assert result["input_ids"] == []
        assert result["labels"] == []

    def test_batch(self):
        """Test batch encoding."""
        batch = [
            [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi!"},
            ],
            [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "What is 1+1?"},
                {"role": "assistant", "content": "2"},
            ],
            [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "B"},
                {"role": "user", "content": "C"},
                {"role": "assistant", "content": "D"},
            ],
        ]
        results = self.native.encode_for_training_batch(batch, strategy="default")
        assert len(results) == 3

        # Each result should match individual encode_for_training
        for conv, result in zip(batch, results):
            single = self.native.encode_for_training(conv, strategy="default")
            assert result["input_ids"] == single["input_ids"]
            assert result["labels"] == single["labels"]

    def test_invalid_strategy(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        with pytest.raises(Exception):
            self.native.encode_for_training(messages, strategy="invalid")
