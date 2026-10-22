"""Teacher logprobs: the actual token's score, a teacher-only rule, and a teacher per training env.

The orchestrator used to ask for `prompt_logprobs: true` and read the first candidate of every position. Surogate's
engine refuses the boolean (HTTP 400, it takes an integer) and returns a position's candidates as a JSON object
whose keys come back sorted, so with alternatives the first entry was the actual token at only 34, 6 and 3 of 57
positions for k = 1, 5 and 20 in a measurement (57/57 for k = 0, which returns the actual token alone). vLLM
0.31.0 puts the actual token first for every k, so the first entry was right there only by its ordering. Each
position is now read by the token id that was sent, and the request asks for an integer, 0.

The fake teachers below answer every request with one candidate dict per position, scored by `logprob`.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import verifiers as vf

from surogate.grpo.orchestrator.utils import (
    TeacherRoute,
    TeacherRule,
    actual_token_logprobs,
    compute_routed_teacher_logprobs,
    compute_teacher_logprobs,
    set_semaphore,
)
from surogate.grpo.transport import TrainingSample

ALTERNATIVES = (1, 2)  # ids whose keys sort ahead of every student token used here ("1" < "100")


def logprob(position: int, token: int, offset: float = 0.0) -> float:
    """A score unique to the teacher position and the token there, so a misaligned read cannot match."""
    return -(position + token / 1000 + offset)


def candidate(position: int, token: int, offset: float = 0.0) -> dict:
    return {"logprob": logprob(position, token, offset), "rank": 2, "decoded_token": "t"}


def k0_entries(tokens: list[int], offset: float = 0.0) -> list:
    """The engine at prompt_logprobs 0: the actual token alone; position 0 has no context."""
    return [None] + [{str(token): candidate(t, token, offset)} for t, token in enumerate(tokens) if t]


def engine_entries(tokens: list[int]) -> list:
    """The engine at k >= 1: the actual token and more likely alternatives, keys sorted as strings."""
    entries = [None]
    for t, token in enumerate(tokens[1:], start=1):
        found = {str(alt): {"logprob": -0.01, "rank": 1, "decoded_token": "a"} for alt in ALTERNATIVES}
        found[str(token)] = candidate(t, token)
        entries.append(dict(sorted(found.items())))
    return entries


def vllm_entries(tokens: list[int]) -> list:
    """vLLM: the actual token first, then more likely alternatives."""
    entries = [None]
    for t, token in enumerate(tokens[1:], start=1):
        found = {str(token): candidate(t, token)}
        found.update({str(alt): {"logprob": -0.01, "rank": 1, "decoded_token": "a"} for alt in ALTERNATIVES})
        entries.append(found)
    return entries


class FakeTeacher:
    """A `/v1/chat/completions/tokens` endpoint that records request bodies and answers with `score(tokens)`."""

    def __init__(self, score):
        self.bodies: list[dict] = []
        teacher = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                teacher.bodies.append({"path": self.path, **body})
                data = json.dumps(
                    {
                        "id": "score",
                        "object": "chat.completion",
                        "created": 0,
                        "model": body["model"],
                        "choices": [
                            {"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "length"}
                        ],
                        "prompt_logprobs": score(body["tokens"]),
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()
        self.client = vf.ClientConfig(
            api_base_url=f"http://127.0.0.1:{self.server.server_port}/v1",
            api_key_var="SUROGATE_TEST_TEACHER_KEY",
            max_retries=0,
        )

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def teacher():
    started: list[FakeTeacher] = []

    def start(score) -> FakeTeacher:
        started.append(FakeTeacher(score))
        return started[-1]

    yield start
    for server in started:
        server.close()


def make_sample(prompt_ids: list[int], completion_ids: list[int]) -> TrainingSample:
    return TrainingSample(
        prompt_ids=list(prompt_ids),
        prompt_mask=[False] * len(prompt_ids),
        completion_ids=list(completion_ids),
        completion_mask=[True] * len(completion_ids),
        completion_logprobs=[0.0] * len(completion_ids),
        completion_temperatures=[1.0] * len(completion_ids),
    )


def run(coroutine_fn):
    async def main():
        await set_semaphore(-1)
        return await coroutine_fn()

    return asyncio.run(main())


def score(clients, samples, rule=None, model="teacher"):
    return run(lambda: compute_teacher_logprobs(clients, model, samples, rule=rule))


PROMPT = [100, 205, 117, 310, 499]
COMPLETION = [123, 456, 789]
TOKENS = PROMPT + COMPLETION
EXPECTED = [0.0] + [logprob(t, token) for t, token in enumerate(TOKENS) if t]


def test_engine_sorted_candidates_are_read_by_token_id(teacher):
    entries = engine_entries(TOKENS)
    # The fixture is the case that broke: the first key is an alternative at every scored position.
    assert all(next(iter(entry)) != str(token) for entry, token in zip(entries[1:], TOKENS[1:]))
    server = teacher(engine_entries)

    assert score([server.client], [make_sample(PROMPT, COMPLETION)]) == [EXPECTED]


def test_vllm_actual_first_candidates_give_the_same_scores(teacher):
    server = teacher(vllm_entries)

    assert score([server.client], [make_sample(PROMPT, COMPLETION)]) == [EXPECTED]


def test_k0_request_and_single_candidate(teacher):
    server = teacher(k0_entries)

    result = score([server.client], [make_sample(PROMPT, COMPLETION), make_sample([100, 101], [102])])

    assert result == [EXPECTED, [0.0, logprob(1, 101), logprob(2, 102)]]
    body = server.bodies[0]
    assert body["path"] == "/v1/chat/completions/tokens"
    assert body["tokens"] == TOKENS
    # An integer, not a boolean: JSON false would decode to a Python value equal to 0.
    assert type(body["prompt_logprobs"]) is int and body["prompt_logprobs"] == 0


def test_an_int_keyed_candidate_is_the_actual_token_too():
    entries = [None, {205: {"logprob": -1.5}}, {"117": {"logprob": -2.5}}]
    assert actual_token_logprobs([100, 205, 117], entries) == [0.0, -1.5, -2.5]


@pytest.mark.parametrize(
    "scores, error",
    [
        # The actual token is missing at one position: no other candidate may stand in for it.
        (lambda tokens: k0_entries(tokens)[:3] + [{"1": {"logprob": -0.01}}] + k0_entries(tokens)[4:], "position 3"),
        # A null entry past position 0 is a missing score as well.
        (lambda tokens: k0_entries(tokens)[:2] + [None] + k0_entries(tokens)[3:], "position 2"),
        (lambda tokens: k0_entries(tokens)[:-1], f"7 entries for {len(TOKENS)} tokens"),
        (lambda tokens: None, "no prompt_logprobs"),
    ],
    ids=["actual-token-missing", "null-entry", "one-entry-short", "no-prompt-logprobs"],
)
def test_a_missing_actual_token_is_an_error(teacher, scores, error):
    server = teacher(scores)

    with pytest.raises(ValueError, match=error):
        score([server.client], [make_sample(PROMPT, COMPLETION)])


class CharTokenizer:
    """One id per character (its code point) and the two ChatML markers, which is all the rule splice needs."""

    unk_token_id = 0
    special = {"<|im_start|>": 1, "<|im_end|>": 2}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.special.get(token, self.unk_token_id)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


TOKENIZER = CharTokenizer()
IM_START, IM_END = 1, 2


def chatml(*messages: tuple[str, str]) -> list[int]:
    ids = []
    for role, content in messages:
        ids += [IM_START, *TOKENIZER.encode(f"{role}\n{content}", add_special_tokens=False), IM_END]
        ids += TOKENIZER.encode("\n", add_special_tokens=False)
    return ids


ASSISTANT = [IM_START, *TOKENIZER.encode("assistant\n", add_special_tokens=False)]
ANSWER = [*TOKENIZER.encode("Done.", add_special_tokens=False), IM_END]
RULE = TeacherRule.from_tokenizer(TOKENIZER, "Check the account before answering.\n")


def test_rule_goes_to_the_end_of_the_system_message_and_its_positions_are_dropped(teacher):
    prompt = chatml(("system", "You are a support agent."), ("user", "Hi")) + ASSISTANT
    ids = prompt + ANSWER
    end_of_system = ids.index(IM_END)
    inserted = TOKENIZER.encode("\n\nCheck the account before answering.", add_special_tokens=False)
    server = teacher(k0_entries)

    [result] = score([server.client], [make_sample(prompt, ANSWER)], rule=RULE)

    # The teacher saw the rule just before the system message's <|im_end|>, and nothing else changed.
    assert server.bodies[0]["tokens"] == ids[:end_of_system] + inserted + ids[end_of_system:]
    # The result keeps the student's positions: below the span teacher position t, from it on t + len(rule).
    assert len(result) == len(ids)
    assert result[0] == 0.0
    assert result[1:end_of_system] == [logprob(t, ids[t]) for t in range(1, end_of_system)]
    assert result[end_of_system:] == [logprob(t + len(inserted), ids[t]) for t in range(end_of_system, len(ids))]


def test_without_a_system_message_the_rule_gets_one_at_the_start(teacher):
    prompt = chatml(("user", "Hi")) + ASSISTANT
    ids = prompt + ANSWER
    block = chatml(("system", "Check the account before answering."))
    server = teacher(k0_entries)

    [result] = score([server.client], [make_sample(prompt, ANSWER)], rule=RULE)

    assert server.bodies[0]["tokens"] == block + ids
    # Student position 0 now has context, the rule's system message, so it is scored like every other.
    assert result == [logprob(t + len(block), ids[t]) for t in range(len(ids))]


def test_no_rule_sends_the_student_tokens_unchanged(teacher):
    prompt = chatml(("system", "You are a support agent."), ("user", "Hi")) + ASSISTANT
    ids = prompt + ANSWER
    server = teacher(k0_entries)

    [result] = score([server.client], [make_sample(prompt, ANSWER)])

    assert server.bodies[0]["tokens"] == ids
    assert result == [0.0] + [logprob(t, ids[t]) for t in range(1, len(ids))]


def test_a_rule_needs_chatml_markers():
    class PlainTokenizer(CharTokenizer):
        special = {}

    with pytest.raises(ValueError, match="ChatML"):
        TeacherRule.from_tokenizer(PlainTokenizer(), "A rule.")


class FakePool:
    def __init__(self, *servers: FakeTeacher):
        self.clients = [server.client for server in servers]


def test_each_env_is_scored_by_its_own_teacher_and_results_keep_sample_order(teacher):
    plain = teacher(k0_entries)
    ruled = teacher(lambda tokens: k0_entries(tokens, offset=0.5))
    plain_pool, ruled_pool = FakePool(plain), FakePool(ruled)
    routes = {
        "math": TeacherRoute(plain_pool, "teacher-a"),
        "chat": TeacherRoute(plain_pool, "teacher-a"),  # same route as math: scored with it
        "support": TeacherRoute(ruled_pool, "teacher-b", RULE),
    }
    prompt = chatml(("system", "You are a support agent."), ("user", "Hi")) + ASSISTANT
    samples = [make_sample(prompt, ANSWER[i:]) for i in range(5)]
    envs = ["math", "support", "chat", "support", "math"]

    result = run(lambda: compute_routed_teacher_logprobs(samples, envs, routes))

    end_of_system = prompt.index(IM_END)
    inserted = len(RULE.appended_ids)
    for sample, env, sample_result in zip(samples, envs, result):
        ids = sample.prompt_ids + sample.completion_ids
        if env == "support":
            expected = [0.0] + [
                logprob(t if t < end_of_system else t + inserted, ids[t], offset=0.5) for t in range(1, len(ids))
            ]
        else:
            expected = [0.0] + [logprob(t, ids[t]) for t in range(1, len(ids))]
        assert sample_result == expected
    assert {body["model"] for body in plain.bodies} == {"teacher-a"} and len(plain.bodies) == 3
    assert {body["model"] for body in ruled.bodies} == {"teacher-b"} and len(ruled.bodies) == 2


def test_a_sample_of_an_env_without_a_teacher_is_an_error():
    routes = {"math": TeacherRoute(FakePool(), "teacher-a")}

    with pytest.raises(ValueError, match="'chat' has no teacher"):
        run(lambda: compute_routed_teacher_logprobs([make_sample([100], [101])], ["chat"], routes))
