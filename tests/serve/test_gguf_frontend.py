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
    _declared_merges,
    _spm_merges,
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
def test_chat_template_is_the_file_s_own_and_is_usable(reader):
    """The template we serve is the one the GGUF carries, verbatim -- `extract_chat_template`
    is a key-value read, so there is nothing of ours between the file and the client. What is
    ours to get wrong is whether it is present and whether it parses, so that is what this
    asserts. Comparing it byte-for-byte against the Hub is the next test, and deliberately not
    a failure."""
    import jinja2

    template = extract_chat_template(reader)
    assert template, "the GGUF carries no chat template"
    jinja2.Environment().parse(template)  # raises TemplateSyntaxError if it is not usable


@needs_gguf
@pytest.mark.network
def test_chat_template_drift_from_the_official_repo_is_reported_not_failed(reader):
    """A GGUF embeds the template as it stood when the file was quantised; the Hub's copy
    moves under it. This once asserted equality and duly broke when Qwen edited a single
    condition (`arguments is defined` -> `arguments is mapping`), reporting a red test for a
    change in someone else's repository. Serving the file's own template is correct -- it is
    what llama.cpp serves from the same file -- so drift is reported and skipped."""
    from huggingface_hub import hf_hub_download

    try:
        cfg_path = hf_hub_download(_OFFICIAL_REPO, "tokenizer_config.json")
        official = json.loads(Path(cfg_path).read_text()).get("chat_template")
        if official is None:
            official = Path(hf_hub_download(_OFFICIAL_REPO, "chat_template.jinja")).read_text()
    except Exception as exc:
        pytest.skip(f"official template unavailable: {exc}")
    mine = extract_chat_template(reader)
    if mine == official:
        return
    import difflib

    first = next(
        (line for line in difflib.unified_diff(
            official.splitlines(), mine.splitlines(), "hub", "gguf", lineterm="", n=0)
         if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))),
        "(no line differs; whitespace only)",
    )
    pytest.skip(
        f"{_OFFICIAL_REPO} has edited its chat template since this GGUF was quantised; "
        f"the file's own template is what gets served. First difference: {first.strip()[:120]}"
    )


# ---------------------------------------------------------------------------
# SentencePiece: the merge list a vocabulary implies
# ---------------------------------------------------------------------------
#
# A sentencepiece GGUF carries pieces and scores and no merges, and the engine ranks its merges
# off the list rather than off the ids. `_spm_merges` recovers it. These check the properties the
# recovery rests on, against a vocabulary small enough to reason about by hand; the equivalence
# that matters — that a reconstruction encodes like the official tokenizer — is checked against
# the real files, which reproduce TinyLlama's 61,249 merges exactly and tokenize Gemma 3
# identically over a corpus.

_NORMAL, _CONTROL, _USER_DEFINED, _BYTE_TYPE = 1, 3, 4, 6


def test_spm_merges_orders_by_score_and_skips_leaves():
    #                 0     1     2      3       4        5         6
    tokens = ["<unk>", "a", "b", "ab", "abb", "<0x41>", "ab</s>"]
    scores = [0.0, -1.0, -2.0, -5.0, -3.0, 0.0, 0.0]
    types = [_CONTROL, _NORMAL, _NORMAL, _NORMAL, _NORMAL, _BYTE_TYPE, _CONTROL]
    merges = _spm_merges(tokens, scores, types)
    # "abb" (-3) outranks "ab" (-5): a higher score was learned earlier.
    assert merges == [["ab", "b"], ["a", "b"]]
    # A byte piece and a control piece are leaves, so neither is split.
    assert not any("<0x41>" in pair or "</s>" in "".join(pair) for pair in merges)


def test_spm_merges_emits_every_split_of_a_piece():
    # "aaa" splits two ways, and both halves are in the vocabulary both times.
    tokens = ["a", "aa", "aaa"]
    scores = [0.0, -1.0, -2.0]
    types = [_NORMAL, _NORMAL, _NORMAL]
    merges = _spm_merges(tokens, scores, types)
    assert merges == [["a", "a"], ["a", "aa"], ["aa", "a"]]


def test_spm_merges_reaches_user_defined_pieces():
    # Gemma 3 carries ~900 merges whose result is a user-defined piece; skipping that type
    # loses them, and the tokenizer then splits those strings differently.
    tokens = ["x", "y", "xy"]
    scores = [0.0, 0.0, -1.0]
    assert _spm_merges(tokens, scores, [_NORMAL, _NORMAL, _USER_DEFINED]) == [["x", "y"]]
    assert _spm_merges(tokens, scores, [_NORMAL, _NORMAL, _CONTROL]) == []


# ---------------------------------------------------------------------------
# Gemma 4: a SentencePiece surface over a stated merge list
# ---------------------------------------------------------------------------
#
# `tokenizer.ggml.model == "gemma4"` is neither of the two schemes above. llama.cpp reads it as
# BPE over the file's own merges while keeping the metaspace surface (llama-vocab.cpp), and the
# scores are a constant -1000, so `_spm_merges` could not recover the order even in principle.

_G4_GGUF = _REPO_ROOT / "models" / "gguf" / "gemma-4-12b-it-qat-q4_0.gguf"
_G4_OFFICIAL = _REPO_ROOT / "models" / "gemma-4-31B-it-frontend" / "tokenizer.json"

needs_gemma4 = pytest.mark.skipif(
    not _G4_GGUF.is_file(), reason=f"no Gemma 4 GGUF at {_G4_GGUF}"
)


def test_declared_merges_split_at_the_joining_space():
    # The pieces spell a space as the word mark, so the one literal space is the join --
    # including when a half is itself a run of marks or of newlines.
    assert _declared_merges(_StubReader(["ab c", "\u2581\u2581 \u2581", "\n\n \n"])) == [
        ["ab", "c"], ["\u2581\u2581", "\u2581"], ["\n\n", "\n"]
    ]


def test_declared_merges_refuse_an_entry_with_no_join():
    # llama.cpp leaves both halves empty for such an entry, giving a rank nothing reaches;
    # a merge list we cannot read is an error rather than a silently dropped merge.
    with pytest.raises(SystemExit, match="no separating space"):
        _declared_merges(_StubReader(["ab", "c d"]))


class _StubReader:
    """Just enough reader for `_declared_merges`: one key, read through `contents()`."""

    def __init__(self, merges):
        self._merges = merges

    def get_field(self, name):
        if name != "tokenizer.ggml.merges":
            return None
        return type("F", (), {"contents": lambda _self, m=self._merges: m})()


@needs_gemma4
def test_gemma4_reconstruction_reads_the_merges_it_is_given():
    reader = GGUFReader(str(_G4_GGUF), "r")
    tok = extract_tokenizer_json(reader)
    stated = list(reader.get_field("tokenizer.ggml.merges").contents())
    model = tok["model"]
    # Read, not recovered: one pair per stated entry, in the file's order.
    assert len(model["merges"]) == len(stated)
    assert [" ".join(pair) for pair in model["merges"][:64]] == [str(e) for e in stated[:64]]
    # The SentencePiece surface, which is what separates this from the byte-level scheme.
    assert model["byte_fallback"] and model["fuse_unk"]
    assert tok["normalizer"] == {
        "type": "Replace", "pattern": {"String": " "}, "content": "\u2581"
    }


@needs_gemma4
@pytest.mark.skipif(not _G4_OFFICIAL.is_file(), reason="no official Gemma 4 tokenizer on disk")
def test_gemma4_reconstruction_matches_the_official_tokenizer(tmp_path):
    """The gate. Google's own file is the standard, and llama.cpp agrees with it: all three
    tokenize this battery identically, which was checked against `llama-tokenize` on the same
    GGUF at the time this arm was written (2,608 tokens, no divergence)."""
    path = tmp_path / "tokenizer.json"
    path.write_text(
        json.dumps(extract_tokenizer_json(GGUFReader(str(_G4_GGUF), "r")), ensure_ascii=False)
    )
    mine = tokenizers.Tokenizer.from_file(str(path))
    official = tokenizers.Tokenizer.from_file(str(_G4_OFFICIAL))
    for s in _BATTERY + ["<|turn>user\nhi<turn|>", "line\r\nwith\rcarriage\rreturns"]:
        assert (
            mine.encode(s, add_special_tokens=False).ids
            == official.encode(s, add_special_tokens=False).ids
        ), f"encode divergence on {s!r}"


def test_generation_config_uses_declared_ids():
    from types import SimpleNamespace
    from surogate.serve.gguf.frontend import extract_generation_config
    values = {"tokenizer.ggml.tokens": ["a", "b", "stop", "turn", "pad"],
              "tokenizer.ggml.eos_token_id": 2, "tokenizer.ggml.eot_token_id": 3,
              "tokenizer.ggml.eom_token_id": 3, "tokenizer.ggml.padding_token_id": 4}
    class Reader:
        def get_field(self, name):
            return None if name not in values else SimpleNamespace(contents=lambda: values[name])
    assert extract_generation_config(Reader()) == {"eos_token_id": [2, 3], "pad_token_id": 4}
    values["tokenizer.ggml.eot_token_id"] = 5
    with pytest.raises(ValueError, match="outside its vocabulary"):
        extract_generation_config(Reader())


def test_native_frontend_comes_from_the_weight_source(tmp_path):
    from types import SimpleNamespace
    from surogate.serve.ingest import _native_gguf_frontend
    values = {"general.architecture": "qwen4exp", "tokenizer.ggml.model": "gpt2",
              "tokenizer.ggml.pre": "qwen35", "tokenizer.ggml.tokens": ["a", "b", "stop"],
              "tokenizer.ggml.token_type": [1, 1, 3], "tokenizer.ggml.merges": [],
              "tokenizer.ggml.eos_token_id": 2, "tokenizer.chat_template": "{{ messages }}"}
    class Reader:
        def kv(self, name):
            return values.get(name)
        def get_field(self, name):
            return None if name not in values else SimpleNamespace(contents=lambda: values[name])
    root = _native_gguf_frontend(Reader(), tmp_path, echo=lambda _: None)
    tokenizer = json.loads((root / "tokenizer.json").read_text())
    assert tokenizer["model"]["vocab"] == {"a": 0, "b": 1}
    assert tokenizer["added_tokens"][0]["id"] == 2
    assert json.loads((root / "generation_config.json").read_text()) == {"eos_token_id": 2}
    assert (root / "chat_template.jinja").read_text() == "{{ messages }}"


def test_gemma3_turn_token_is_a_generation_stop():
    from types import SimpleNamespace
    from surogate.serve.gguf.frontend import extract_generation_config
    values = {"general.architecture": "gemma3",
              "tokenizer.ggml.tokens": ["a", "<eos>", "<end_of_turn>"],
              "tokenizer.ggml.eos_token_id": 1}
    class Reader:
        def get_field(self, name):
            return None if name not in values else SimpleNamespace(contents=lambda: values[name])
    assert extract_generation_config(Reader())["eos_token_id"] == [1, 2]


def test_glm_pre_tokenizer_groups_digits_in_threes():
    from surogate.serve.gguf.frontend import _PRE_SPLIT_REGEX
    split = tokenizers.pre_tokenizers.Split(tokenizers.Regex(_PRE_SPLIT_REGEX["glm4"]),
                                           behavior="isolated")
    pieces = [piece for piece, _ in split.pre_tokenize_str("1234567")]
    assert pieces == ["123", "456", "7"]
