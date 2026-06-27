"""Running-transcript state for multi-turn routing.

At each turn the router recomputes its decision feature over the whole transcript, so
it can route per step (e.g. swap a debugger in at a critical moment). ``render`` flattens
the transcript into the single text the featurizer consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import Message


def raw_text(messages: list[Message]) -> str:
    """Flatten messages to the raw ``"role: content\\n"`` form the router reads.

    Critical: Fugu feeds the backbone this raw text, NOT a chat template. Using a chat
    template collapses routing accuracy (the reference inspection measured ~5% vs ~95%
    joint agent/role accuracy), because the decision hidden state is taken from this
    exact surface form.
    """
    return "".join(f"{m['role']}: {m['content']}\n" for m in messages)


def raw_query(prompt: str, system: str | None = None) -> str:
    msgs: list[Message] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return raw_text(msgs)


@dataclass
class Transcript:
    messages: list[Message] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def render(self) -> str:
        return raw_text(self.messages)

    def as_messages(self) -> list[Message]:
        return list(self.messages)
