"""What a GRPO rollout needs back from the engine, checked with the client GRPO uses.

`verifiers` reads a chat response through `parse_tokens`, which returns None -- silently,
with no error anywhere -- whenever the top-level `prompt_token_ids`, the choice's `token_ids`
or `logprobs.content` is missing. A None there is not a degraded rollout, it is no training
sample at all, so nothing downstream complains and a run simply learns from less than it
collected. That failure mode is why this test exists rather than a check on the JSON shape:
it asserts the thing the client actually concludes.

The second turn is the one worth having. TITO falls back to re-rendering the whole
conversation on an empty trajectory, so only a later turn exercises `/tokenize` and
`/v1/chat/completions/tokens` -- and when the stitch fails it says so in a log line and
carries on, which is how an engine that ignored `add_generation_prompt` looked healthy.

Needs a real engine and a real model, so it skips unless SUROGATE_TEST_MODEL names a
checkpoint and the engine binary has been built.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_MODEL = os.environ.get("SUROGATE_TEST_MODEL", "")
# The CLI rather than the engine binary: the binary takes a built .sinfer artifact,
# and the CLI is what turns a GGUF or a checkpoint directory into one first.
_CLI = _ROOT / ".venv" / "bin" / "surogate"

needs_engine = pytest.mark.skipif(
    not _MODEL or not Path(_MODEL).exists() or not _CLI.is_file(),
    reason="set SUROGATE_TEST_MODEL to a checkpoint; needs .venv/bin/surogate and a built engine",
)
verifiers = pytest.importorskip("verifiers", reason="verifiers is not installed")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def engine():
    port = _free_port()
    # The log goes to a file rather than a pipe: nothing drains a pipe while the
    # engine loads, and a full one would stall it exactly where it is slowest.
    with subprocess.Popen(
        [str(_CLI), "serve", _MODEL, "--device", os.environ.get("SUROGATE_TEST_DEVICE", "0"),
         "--port", str(port), "--max-model-len", "4096"],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    ) as proc:
        deadline = time.time() + 900
        try:
            ready = False
            while time.time() < deadline:
                if proc.poll() is not None:
                    pytest.skip(f"the engine exited while starting (code {proc.returncode})")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        ready = True
                        break
                except OSError:
                    time.sleep(2)
            if not ready:
                pytest.skip("the engine did not start in time")
            yield port
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()


def _model_id(port: int) -> str:
    import urllib.request

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30) as response:
        return json.load(response)["data"][0]["id"]


@needs_engine
def test_two_turn_rollout_yields_training_samples(engine):
    import asyncio

    import verifiers as vf
    from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
    from verifiers.clients.openai_chat_completions_token_client import (
        OpenAIChatCompletionsTokenClient,
    )
    from verifiers.types import AssistantMessage, UserMessage

    port = engine
    model = _model_id(port)
    os.environ.setdefault("OPENAI_API_KEY", "unused")

    def config(client_type: str):
        return vf.ClientConfig(
            client_idx=0, client_type=client_type,
            api_base_url=f"http://127.0.0.1:{port}/v1", api_key_var="OPENAI_API_KEY",
            timeout=600, max_connections=8, max_keepalive_connections=8,
            max_retries=2, extra_headers={},
        )

    # exactly what surogate/grpo/orchestrator/utils.py::get_sampling_args builds
    sampling = {
        "temperature": 1.0, "top_p": 1.0, "max_tokens": 600, "logprobs": True,
        "extra_body": {"return_token_ids": True, "top_k": -1, "min_p": 0.0,
                       "min_tokens": 0, "repetition_penalty": 1.0},
    }
    question = "Name one prime number."

    def tokens_of(response):
        got = getattr(getattr(response, "message", None), "tokens", None)
        assert got is not None, (
            "parse_tokens returned None: the response is missing prompt_token_ids, "
            "choices[0].token_ids or choices[0].logprobs.content, so this rollout "
            "yields no training sample"
        )
        read = (lambda k: got[k]) if isinstance(got, dict) else (lambda k: getattr(got, k))
        return read("prompt_ids"), read("completion_ids"), read("completion_logprobs")

    async def run():
        mito = OpenAIChatCompletionsClient(config("openai_chat_completions"))
        tito = OpenAIChatCompletionsTokenClient(config("openai_chat_completions_token"))

        native = await mito.get_native_response(
            [{"role": "user", "content": question}], model, dict(sampling), [])
        first = tokens_of(await mito.from_native_response(native))
        assert len(first[0]) > 0, "the prompt tokenised to nothing"
        assert len(first[1]) == len(first[2]), "one log-probability per generated token"
        assert all(value <= 1e-6 for value in first[2]), "a log-probability above zero"

        text = native.choices[0].message.content or ""
        if native.choices[0].finish_reason != "stop":
            pytest.skip("the first turn was truncated, which the client refuses to stitch")

        # Turn 2 must be built through the client's own conversion: the stitch compares
        # its trajectory against these messages key by key.
        prefix, _ = await tito.to_native_prompt(
            [UserMessage(content=question), AssistantMessage(content=text)])
        step = {
            "prompt": [UserMessage(content=question)],
            "completion": [AssistantMessage(content=text)],
            "tokens": {"prompt_ids": first[0], "completion_ids": first[1],
                       "prompt_mask": [0] * len(first[0]),
                       "completion_mask": [1] * len(first[1]),
                       "completion_logprobs": first[2]},
        }
        state = {"trajectory": [step], "model": model}

        stitched = await tito.get_prompt_ids(state, [*prefix, {"role": "user", "content": "And another."}], [])
        assert stitched is not None, (
            "the client could not stitch turn 2 and fell back to re-rendering it; "
            "an engine that ignores add_generation_prompt does this silently"
        )
        assert stitched[: len(first[0]) + len(first[1])] == list(first[0]) + list(first[1]), (
            "turn 2's prompt does not extend turn 1's prompt and completion, so the "
            "trajectory breaks into separate training samples"
        )

        native2 = await tito.get_native_response(
            [*prefix, {"role": "user", "content": "And another."}], model, dict(sampling), [],
            state=state)
        second = tokens_of(await tito.from_native_response(native2))
        assert len(second[1]) == len(second[2])

    asyncio.run(run())


@needs_engine
def test_unimplemented_sampling_fields_are_refused_not_ignored(engine):
    """A field we cannot honour must fail loudly; the values that ask for nothing must not."""
    import urllib.error
    import urllib.request

    port = engine
    model = _model_id(port)

    def post(extra: dict) -> int:
        body = {"model": model, "prompt": "The", "max_tokens": 1, "temperature": 1.0}
        body.update(extra)
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
            headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                response.read()
                return 200
        except urllib.error.HTTPError as error:
            return int(error.code)

    for accepted in ({"min_tokens": 0}, {"repetition_penalty": 1.0},
                     {"prompt_logprobs": False}, {"min_p": 0.0}, {"top_k": -1}):
        assert post(accepted) == 200, f"a request asking for nothing was refused: {accepted}"

    for refused in ({"min_tokens": 8}, {"repetition_penalty": 1.2}, {"prompt_logprobs": True}):
        assert post(refused) == 400, (
            f"{refused} is not implemented but was accepted; the caller would get a "
            f"completion that silently ignored it"
        )
