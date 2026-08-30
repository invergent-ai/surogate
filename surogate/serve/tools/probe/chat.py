"""One chat request, split the way a thinking model answers it.

Every model this engine serves reasons before it answers, and the server returns the two parts
separately: `reasoning_content` and `content`. That makes two mistakes available to a probe, and
they are the same mistake from opposite sides:

* **Read only `content`.** When the token budget runs out inside the reasoning block the answer
  never begins, `content` comes back empty, and the probe scores zero — on a correct engine. At
  the budgets these probes used to ask for (24 to 128 tokens) that is the *usual* outcome, not an
  edge case, so the probe reports a catastrophe on every model and every artifact alike.
* **Read the reasoning too.** Tempting after the first mistake bites, and worse: a model thinking
  about the capital of France writes "Paris" in its reasoning long before it decides how to
  answer, so a run that never produces an answer at all still matches. That is a false pass, and
  a false pass in a correctness probe is more expensive than a false failure.

So: **score the answer, never the reasoning**, and give a request whose budget expired before the
answer began its own outcome — `truncated`. That is a failure of the budget, not of the model,
and calling it either `ok` or `wrong` loses the distinction that tells you which to fix.

The reasoning text is still returned, because it is what you want to look at when an answer is
wrong, and its token count is worth reporting: it is most of what the engine generated.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Mapping, NamedTuple, Sequence


class Reply(NamedTuple):
    """One completion, with the reasoning kept apart from the answer it precedes."""

    answer: str
    reasoning: str
    finish_reason: str
    completion_tokens: int
    reasoning_tokens: int

    @property
    def truncated_in_reasoning(self) -> bool:
        """The budget ran out before the answer began: reasoning, no answer, stopped on length."""

        return not self.answer.strip() and bool(self.reasoning.strip())


def split_message(message: Mapping[str, Any]) -> tuple[str, str]:
    """The answer and the reasoning of one assistant message.

    `reasoning` is the OpenAI spelling and `reasoning_content` the one this server and vLLM use;
    both are accepted so a probe can be pointed at either.
    """

    answer = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    return answer, reasoning


def ask(
    port: int,
    model: str,
    prompt: str,
    max_tokens: int,
    *,
    timeout: float = 900.0,
    temperature: float = 0.0,
    thinking: bool | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Reply:
    """Send one user turn and return its answer and reasoning.

    `thinking` left as None keeps whatever the server was started with, which is what a probe
    should normally measure; True or False overrides it per request so one run can compare the
    two modes without restarting the engine.
    """

    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if thinking is not None:
        body["enable_thinking"] = thinking
    if extra:
        body.update(extra)
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    choice = payload["choices"][0]
    answer, reasoning = split_message(choice.get("message") or {})
    usage = payload.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return Reply(
        answer=answer,
        reasoning=reasoning,
        finish_reason=choice.get("finish_reason") or "",
        completion_tokens=int(usage.get("completion_tokens") or 0),
        reasoning_tokens=int(details.get("reasoning_tokens") or 0),
    )


def is_first_token(delta: Mapping[str, Any]) -> bool:
    """True for the first streamed chunk that carries generated text of any kind.

    Time to first token means the first token the model produced, which on a thinking model is a
    reasoning token. Waiting for `content` alone would report the time to the *answer* — a
    different and much larger number, and one that would silently change meaning the day a model
    stops reasoning.
    """

    return bool(delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"))


def options(argv_tail: Sequence[str]) -> dict[str, str]:
    """Trailing `key=value` arguments, so a probe can gain an option without moving its
    positional ones."""

    parsed: dict[str, str] = {}
    for item in argv_tail:
        if "=" not in item:
            raise SystemExit(f"expected key=value, got {item!r}")
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def thinking_option(parsed: Mapping[str, str]) -> bool | None:
    """`thinking=on|off` from the trailing options; absent means the server's own default."""

    value = parsed.get("thinking")
    if value is None:
        return None
    if value in ("on", "true", "1"):
        return True
    if value in ("off", "false", "0"):
        return False
    raise SystemExit(f"thinking must be on or off, got {value!r}")


def thinking_label(thinking: bool | None) -> str:
    return "server-default" if thinking is None else ("on" if thinking else "off")
