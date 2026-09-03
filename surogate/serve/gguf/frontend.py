# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# Self-contained frontend extraction from GGUF KV metadata.
#
# A GGUF carries its tokenizer (tokenizer.ggml.*: model/pre/tokens/token_type/
# merges/special ids) and chat template (tokenizer.chat_template) in-file.
# This module reconstructs the HF-format frontend files the conversion
# pipeline consumes — tokenizer.json, tokenizer_config.json,
# chat_template.jinja — with no network access, keeping local GGUF serving
# fully offline (llama.cpp parity).
#
# The reconstructed tokenizer.json is semantically equivalent but not
# byte-identical to the official file, so GGUF-sourced conversions run the
# vendored converter with SINFER_ALLOW_DERIVED_FRONTEND=1 (PATCHES.md #12),
# which downgrades the pinned-hash mismatch to a recorded warning.
# Equivalence is enforced by tests/serve/test_gguf_frontend.py: the
# reconstruction must encode/decode identically to the official tokenizer.

from __future__ import annotations

import json
from pathlib import Path

# GGUF token_type values (gguf.TokenType).
_NORMAL, _UNKNOWN, _CONTROL, _USER_DEFINED, _UNUSED, _BYTE = 1, 2, 3, 4, 5, 6

# Pre-tokenizer split regexes, keyed by tokenizer.ggml.pre. Values are the
# ORIGINAL tokenizer.json patterns, taken from llama.cpp's llama-vocab.cpp
# comments (study/llama.cpp/src/llama-vocab.cpp) — llama.cpp preserves the
# upstream regex verbatim in a comment above its own case-folded rewrite.
_PRE_SPLIT_REGEX = {
    "qwen2": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    "qwen35": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
}


def _field(reader, name: str, default=None):
    f = reader.get_field(name)
    return default if f is None else f.contents()


def _spm_merges(tokens: list[str], scores: list[float], types: list[int]) -> list[list[str]]:
    """The merge list a SentencePiece vocabulary implies.

    A sentencepiece model stores pieces and scores and never the merges, but every piece that is
    not a byte or a special was made by joining two pieces that are themselves in the vocabulary.
    So the list is recoverable: split each piece at every position, keep the splits whose halves
    are both vocabulary entries, and order what remains by the merged piece's score, which is the
    order training learned them in. Ties keep vocabulary order, and the splits of one piece are
    emitted by the ids of their halves, so the result is deterministic.

    The merges are the authority for rank here, not the ids: a SentencePiece conversion orders
    ids by score, so `token_id` and `merge rank` disagree, which is why the engine reads this
    list rather than deriving ranks from the vocabulary.

    Checked against the official files: this reproduces TinyLlama's 61,249 merges exactly, in
    order. Gemma 3's published list differs in ordering and in 465 entries that never fire, and
    both reconstructions encode 3,104 corpus strings identically to the official tokenizer.
    """
    vocab = {token: index for index, token in enumerate(tokens)}
    found: list[tuple[float, str, str]] = []
    for index, piece in enumerate(tokens):
        # A byte (`<0xNN>`) and a control token are leaves; a user-defined piece is not, and
        # Gemma 3 reaches ~900 of its merges through them.
        if types[index] not in (_NORMAL, _USER_DEFINED) or len(piece) < 2:
            continue
        local = []
        for cut in range(1, len(piece)):
            left, right = piece[:cut], piece[cut:]
            left_id, right_id = vocab.get(left), vocab.get(right)
            if left_id is not None and right_id is not None:
                local.append((left_id, right_id, left, right))
        local.sort(key=lambda split: (split[0], split[1]))
        found.extend((scores[index], left, right) for _, _, left, right in local)
    found.sort(key=lambda merge: merge[0], reverse=True)
    # The pair form, not "left right": a Gemma piece may contain a space, which the joined
    # spelling cannot represent. The engine reads both.
    return [[left, right] for _, left, right in found]


def _spm_tokenizer_json(reader) -> dict:
    """Reconstruct tokenizer.json for a SentencePiece vocabulary.

    Metaspace and byte fallback, which is a different scheme from the byte-level BPE below:
    no pre-tokenizer regex, no byte-level alphabet, a word mark in place of the space, and
    `<0xNN>` entries for what the merges cannot place. Whether the text is opened with a mark
    is the file's own `add_space_prefix` — Llama 2 and TinyLlama do, Gemma 3 does not and splits
    on the space instead.
    """
    tokens: list[str] = list(_field(reader, "tokenizer.ggml.tokens"))
    scores: list[float] = list(_field(reader, "tokenizer.ggml.scores", []))
    types: list[int] = list(_field(reader, "tokenizer.ggml.token_type"))
    if not scores:
        raise SystemExit(
            "surogate serve: this GGUF carries a SentencePiece vocabulary with no scores, "
            "so its merge order cannot be recovered."
        )
    prefixed = bool(_field(reader, "tokenizer.ggml.add_space_prefix", True))
    unknown = int(_field(reader, "tokenizer.ggml.unknown_token_id", 0) or 0)
    added_ids = [i for i in range(len(tokens)) if types[i] in (_CONTROL, _USER_DEFINED)]
    mark = "\u2581"
    normalizer = (
        {"type": "Sequence", "normalizers": [
            {"type": "Prepend", "prepend": mark},
            {"type": "Replace", "pattern": {"String": " "}, "content": mark}]}
        if prefixed else
        {"type": "Replace", "pattern": {"String": " "}, "content": mark}
    )
    decoders = [
        {"type": "Replace", "pattern": {"String": mark}, "content": " "},
        {"type": "ByteFallback"},
        {"type": "Fuse"},
    ]
    if prefixed:
        decoders.append({"type": "Strip", "content": " ", "start": 1, "stop": 0})
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": i, "content": tokens[i], "single_word": False, "lstrip": False,
             "rstrip": False, "normalized": False, "special": types[i] == _CONTROL}
            for i in added_ids
        ],
        "normalizer": normalizer,
        "pre_tokenizer": (
            None if prefixed else
            {"type": "Split", "pattern": {"String": " "},
             "behavior": "MergedWithPrevious", "invert": False}
        ),
        "post_processor": None,
        "decoder": {"type": "Sequence", "decoders": decoders},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": tokens[unknown] if unknown < len(tokens) else None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": True,
            "byte_fallback": True,
            "ignore_merges": False,
            "vocab": {token: index for index, token in enumerate(tokens)},
            "merges": _spm_merges(tokens, scores, types),
        },
    }


def extract_tokenizer_json(reader) -> dict:
    """Reconstruct an HF `tokenizers`-format tokenizer.json from GGUF KV."""
    model = _field(reader, "tokenizer.ggml.model")
    if model == "llama":
        return _spm_tokenizer_json(reader)
    if model != "gpt2":
        raise SystemExit(
            f"surogate serve: GGUF tokenizer model '{model}' is not supported yet "
            "(byte-level BPE 'gpt2' and SentencePiece 'llama' only)."
        )
    pre = str(_field(reader, "tokenizer.ggml.pre", "qwen2"))
    split_regex = _PRE_SPLIT_REGEX.get(pre)
    if split_regex is None:
        raise SystemExit(
            f"surogate serve: GGUF pre-tokenizer '{pre}' has no known split regex; "
            f"supported: {sorted(_PRE_SPLIT_REGEX)}."
        )

    tokens: list[str] = list(_field(reader, "tokenizer.ggml.tokens"))
    types: list[int] = list(_field(reader, "tokenizer.ggml.token_type"))
    merges: list[str] = list(_field(reader, "tokenizer.ggml.merges"))

    # HF convention (and the engine's loader enforces it): added tokens are
    # NOT part of model.vocab — they overlay it, and the `tokenizers` runtime
    # re-derives their ids as len(vocab)+position, so they must sit
    # CONTIGUOUSLY right after the base vocab. The official files obey this;
    # GGUF appends [PAD...] filler rows after the added block to reach the
    # embedding row count — those fillers must be dropped, exactly as the
    # official tokenizer.json omits them (the model config, not the
    # tokenizer, carries the padded vocab_size).
    added_ids = sorted(i for i in range(len(tokens)) if types[i] in (_CONTROL, _USER_DEFINED))
    base = min(added_ids) if added_ids else len(tokens)
    if added_ids and added_ids != list(range(base, base + len(added_ids))):
        raise SystemExit(
            "surogate serve: GGUF added tokens are not contiguous after the base "
            "vocab; cannot reconstruct an HF tokenizer faithfully."
        )
    vocab = {tokens[idx]: idx for idx in range(base)}
    added = [
        {
            "id": idx,
            "content": tokens[idx],
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": types[idx] == _CONTROL,
        }
        for idx in added_ids
    ]

    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": split_regex},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": False,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": merges,
        },
    }


def extract_chat_template(reader) -> str | None:
    t = _field(reader, "tokenizer.chat_template")
    return str(t) if t else None


def synthesize_tokenizer_config(reader, arch: str) -> dict:
    """Minimal HF tokenizer_config.json from GGUF KV."""
    tokens = list(_field(reader, "tokenizer.ggml.tokens"))

    def tok_str(key):
        idx = _field(reader, f"tokenizer.ggml.{key}")
        return tokens[int(idx)] if idx is not None and int(idx) < len(tokens) else None

    cfg: dict = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "add_bos_token": bool(_field(reader, "tokenizer.ggml.add_bos_token", False)),
        "add_prefix_space": False,
        "clean_up_tokenization_spaces": False,
    }
    for hf_key, kv_key in (
        ("bos_token", "bos_token_id"),
        ("eos_token", "eos_token_id"),
        ("pad_token", "padding_token_id"),
        ("unk_token", "unknown_token_id"),
    ):
        s = tok_str(kv_key)
        if s is not None:
            cfg[hf_key] = s
    # The pad token is stated as the file states it. It is a serving-internal
    # choice with no effect on how real text tokenizes -- the frontend resolves
    # it to an id and no path pads with it -- and exporters disagree about which
    # special to name (unsloth's Qwen3 exports say <|vision_pad|> where the
    # official repository says <|endoftext|>). Rewriting it here would put a
    # token in the artifact that the checkpoint does not claim.
    if not cfg.get("pad_token") and "<|endoftext|>" in tokens:
        cfg["pad_token"] = "<|endoftext|>"
    types = list(_field(reader, "tokenizer.ggml.token_type"))
    cfg["added_tokens_decoder"] = {
        str(i): {
            "content": tokens[i],
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": types[i] == _CONTROL,
        }
        for i in range(len(tokens))
        if types[i] in (_CONTROL, _USER_DEFINED)
    }
    ctx = _field(reader, f"{arch}.context_length")
    if ctx:
        cfg["model_max_length"] = int(ctx)
    template = extract_chat_template(reader)
    if template:
        cfg["chat_template"] = template
    return cfg


def write_frontend(reader, arch: str, out_dir: Path, *, echo=print) -> None:
    """Write tokenizer.json / tokenizer_config.json / chat_template.jinja
    reconstructed from the GGUF into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = extract_tokenizer_json(reader)
    (out_dir / "tokenizer.json").write_text(json.dumps(tok, ensure_ascii=False))
    cfg = synthesize_tokenizer_config(reader, arch)
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2)
    )
    template = extract_chat_template(reader)
    if template:
        (out_dir / "chat_template.jinja").write_text(template)
    echo(
        f"surogate serve: frontend reconstructed from GGUF "
        f"(vocab {len(tok['model']['vocab'])}, {len(tok['added_tokens'])} added tokens, "
        f"chat template {'present' if template else 'absent'})"
    )
