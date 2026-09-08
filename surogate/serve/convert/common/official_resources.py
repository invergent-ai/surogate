"""Read and reconcile chat templates from the selected checkpoint's frontend files."""

from __future__ import annotations
import json
from pathlib import Path

__all__ = ["chat_template_bytes", "tokenizer_config_with_template"]


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
