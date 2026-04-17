"""Message/conversation utility types and functions.

Extracted from the old chat_templates/utils.py for use by dataset
preprocessors and loss-scale logic.
"""

from typing import Any

Tool = dict[str, str | dict]
History = list[tuple[str, str] | list[str]]
Message = dict[str, str | list[dict[str, Any]] | list[int] | None]
Messages = list[Message]


def history_to_messages(
    history: History,
    system: str | None = None,
    roles: list[list[str]] | None = None,
) -> Messages:
    """Convert (query, response) history pairs into a messages list."""
    messages = []
    if not roles:
        roles = [["user", "assistant"]] * len(history)
    else:
        assert len(roles) == len(history)
    if system is not None:
        messages.append({"role": "system", "content": system})
    for role, h in zip(roles, history):
        assert isinstance(h, (list, tuple))
        if h[0] is not None:
            messages.append({"role": role[0], "content": h[0]})
        if h[1] is not None:
            messages.append({"role": role[1], "content": h[1]})
    return messages


def get_last_user_round(messages) -> int:
    """Get the index of the last occurrence of user role."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            return i
    return -1
