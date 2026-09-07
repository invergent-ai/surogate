"""Pinned official Qwen3.6 frontend resources used by artifact conversion.

This module owns only the checkpoint-invariant resource profile.  Exact-target
converters continue to own config, tensor inventory, recipes, and execution
geometry.
"""

from __future__ import annotations

import hashlib
import json

from pathlib import Path
from typing import Mapping, Sequence

from .conversion import ResourcePayload, load_resources
from .inventory import ResourceSpec


OFFICIAL_RESOURCE_SHA256 = {
    "frontend/tokenizer.json": (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    ),
    "frontend/tokenizer_config.json": (
        "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b"
    ),
    "frontend/chat_template.jinja": (
        "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
    ),
    "frontend/generation_config.json": (
        "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"
    ),
    "frontend/preprocessor_config.json": (
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
    ),
    "frontend/video_preprocessor_config.json": (
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
    ),
}


def _derived_frontend_allowed() -> bool:
    # surogate vendor patch (csrc/src/serve/PATCHES.md #12): GGUF-sourced
    # conversions reconstruct the tokenizer from the GGUF's own KV metadata
    # (semantically equivalent, not byte-identical), so the pinned-hash check
    # downgrades to a recorded warning for the derived files when this
    # environment variable is set by the ingest layer. Safetensors-sourced
    # conversions never set it and keep the strict check.
    import os

    return os.environ.get("SINFER_ALLOW_DERIVED_FRONTEND", "") == "1"


_DERIVABLE_RESOURCES = frozenset(
    {"frontend/tokenizer.json", "frontend/tokenizer_config.json", "frontend/chat_template.jinja"}
)


def validate_official_resource_hashes(
    actual_hashes: Mapping[str, str],
    *,
    accept_source: bool = False,
) -> None:
    """Require the complete official six-resource profile.

    ``accept_source`` is for a checkpoint that is its own authority -- a
    quantized export ships the frontend it was calibrated and served with, and
    a re-serialised ``tokenizer.json`` is not a defect. Every resource is still
    present and hashed; a hash that differs from the pinned official one is
    reported, not refused.
    """

    expected_names = tuple(OFFICIAL_RESOURCE_SHA256)
    actual_names = tuple(actual_hashes)
    if actual_names != expected_names:
        raise ValueError(
            "Qwen3.6 frontend resource set mismatch: "
            f"expected {expected_names!r}, got {actual_names!r}"
        )
    derived_ok = _derived_frontend_allowed()
    for name, expected in OFFICIAL_RESOURCE_SHA256.items():
        actual = actual_hashes[name]
        if actual != expected:
            filename = name.removeprefix("frontend/")
            if accept_source:
                import sys

                print(
                    f"note: {filename} is the checkpoint's own: sha256 {actual} "
                    f"(pinned official {expected})",
                    file=sys.stderr,
                )
                continue
            # A GGUF-sourced conversion carries no official file at all: its tokenizer is
            # reconstructed from the file's metadata and the rest is vendored from one size of
            # the family, so every resource is a recorded warning there, not a refusal.
            if derived_ok:
                import sys

                print(
                    f"warning: derived frontend resource {filename}: "
                    f"sha256 {actual} differs from pinned official {expected} "
                    "(SINFER_ALLOW_DERIVED_FRONTEND=1)",
                    file=sys.stderr,
                )
                continue
            raise ValueError(
                f"official Qwen3.6 resource hash mismatch for {filename}: "
                f"expected {expected}, got {actual}"
            )


def validate_official_resources(
    resources: Sequence[ResourcePayload], *, accept_source: bool = False
) -> None:
    """Hash and validate already loaded resource payloads."""

    hashes = {
        resource.name: hashlib.sha256(resource.data).hexdigest()
        for resource in resources
    }
    if len(hashes) != len(resources):
        raise ValueError("Qwen3.6 frontend resource set contains duplicate names")
    validate_official_resource_hashes(hashes, accept_source=accept_source)


def load_official_resources(
    model_dir: str | Path,
    resource_specs: Sequence[ResourceSpec],
    *,
    accept_source: bool = False,
) -> tuple[ResourcePayload, ...]:
    """Load exactly the pinned official resource set from a source checkpoint."""

    spec_names = tuple(spec.name for spec in resource_specs)
    expected_names = tuple(OFFICIAL_RESOURCE_SHA256)
    if spec_names != expected_names:
        raise ValueError(
            "converter resource inventory does not match the official Qwen3.6 profile: "
            f"expected {expected_names!r}, got {spec_names!r}"
        )
    resources = load_resources(model_dir, resource_specs)
    validate_official_resources(resources, accept_source=accept_source)
    return resources


__all__ = [
    "OFFICIAL_RESOURCE_SHA256",
    "load_official_resources",
    "tokenizer_config_with_template",
    "validate_official_resource_hashes",
    "validate_official_resources",
]


def chat_template_bytes(root: Path) -> bytes | None:
    """The template the artifact serves, from whichever place the release states it.

    Written out verbatim, with no added trailing newline: the engine compares it byte for
    byte against the copy in `tokenizer_config.json`.

    None when the checkpoint publishes none, which is what a base model looks like rather
    than a broken release -- Gemma 3 270M is one. The artifact then carries no
    `frontend/chat_template.jinja`, the engine reads its absence as "this is a base model",
    and the chat-shaped endpoints refuse it by name while `/v1/completions` serves it.
    """

    path = root / "chat_template.jinja"
    if path.exists():
        return path.read_bytes()
    config = json.loads((root / "tokenizer_config.json").read_text(encoding="utf-8"))
    template = config.get("chat_template")
    if not isinstance(template, str) or not template:
        return None
    return template.encode("utf-8")


def tokenizer_config_with_template(raw: bytes, template: bytes) -> bytes:
    """`tokenizer_config.json`, guaranteed to state the template the artifact serves.

    Lives here rather than beside one target because more than one release keeps its
    template in only one of the two places, and a second copy of this insertion is a
    second chance to get a delicate textual edit subtly different.

    Returned unchanged when the config already states it — which is the Qwen and
    TinyLlama case, and where an existing key that *disagrees* with
    `chat_template.jinja` is a checkpoint contradicting itself and is refused
    rather than silently normalized. Where the key is absent, it is inserted
    immediately after the opening brace so that every other byte of the file
    survives; re-serializing 1.1 MB of `added_tokens_decoder` to add one member
    would rewrite the whole file to no purpose.
    """

    config = json.loads(raw.decode("utf-8"))
    existing = config.get("chat_template")
    if isinstance(existing, str):
        if existing.encode("utf-8") != template:
            raise ValueError(
                "tokenizer_config.json.chat_template disagrees with "
                "chat_template.jinja; the engine compares the two and would "
                "refuse the artifact"
            )
        return raw
    if existing is not None:
        raise ValueError(
            "tokenizer_config.json.chat_template is not a string; the engine "
            f"requires one, got {type(existing).__name__}"
        )
    member = f'"chat_template": {json.dumps(template.decode("utf-8"), ensure_ascii=False)}'
    if not config:
        return ("{" + member + "}").encode("utf-8")
    text = raw.decode("utf-8")
    brace = text.index("{")
    return (text[: brace + 1] + member + "," + text[brace + 1 :]).encode("utf-8")
