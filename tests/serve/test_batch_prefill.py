"""Prompts from several requests are prefilled together (SUROGATE-CHANGES #14). GPU.

Skipped unless SUROGATE_BATCH_PREFILL_TEST_ARTIFACT names a prepared chat model (.sinfer); select one
free GPU with CUDA_VISIBLE_DEVICES, and use SUROGATE_SERVE_BIN to pick the engine binary under test.

- One-question decisions sent together are prefilled in shared rounds (packed_prefill_rounds_total
  grows) and answer exactly as they do one at a time, bit for bit.
- A chat prompt's first token is the same, bit for bit, prefilled alone or packed with others,
  a prompt longer than the step's token window included.
- A decision takes the queue places its questions will need before any of it is prefilled, and
  waits for them, instead of running its shared prefix and then finding the queue full; one that
  leaves while it waits gives its places back.
"""

import concurrent.futures as cf
import contextlib
import json
import os
import socket
import subprocess
import time

import pytest
import requests

from surogate.cli.serve import _resolve_binary

pytestmark = pytest.mark.skipif(not os.getenv("SUROGATE_BATCH_PREFILL_TEST_ARTIFACT"),
                                reason="needs a prepared chat model and a free GPU")

SENTENCES = [
    "The customer says the invoice was paid, but the service was suspended yesterday.",
    "The order arrived a day late and the packaging was damaged.",
    "The customer asks whether the contract can be cancelled without a penalty.",
    "The agent confirmed the refund, but the amount has not appeared in the account yet.",
    "The invoice lists two identical charges for the same subscription month.",
    "The user cannot log in after resetting the password.",
]
REFUND = {"type": "noul", "instructions": "Does the customer ask for money back?",
          "criteria": {"true": "A refund is requested", "false": "No refund is requested"}}
TOPIC = {"type": "choice", "instructions": "What is the main topic?",
         "criteria": {"billing": "Billing or payment", "delivery": "Delivery or shipping",
                      "account": "Account access", "other": "Something else"}}


@contextlib.contextmanager
def server(tmp_path, *flags):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    log = tmp_path / f"server-{port}.log"
    cmd = [_resolve_binary("server"), os.environ["SUROGATE_BATCH_PREFILL_TEST_ARTIFACT"], "--host", "127.0.0.1",
           "--port", str(port), "--served-model-name", "rune", "--max-model-len", "8192", *flags]
    base = f"http://127.0.0.1:{port}"
    with log.open("w") as output:
        process = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            assert process.poll() is None, log.read_text()[-4000:]
            try:
                if requests.get(base + "/v1/models", timeout=1).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5)
        else:
            pytest.fail(log.read_text()[-4000:])
        yield base
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def counter(base, name):
    for line in requests.get(base + "/metrics", timeout=10).text.splitlines():
        if line.startswith(f"surogate_{name}"):
            return float(line.rsplit(" ", 1)[1])
    raise AssertionError(name)


def decision(i, questions=None):
    state = " ".join(SENTENCES[(i + k) % len(SENTENCES)] for k in range(6)) + f" Ticket {i}."
    return {"model": "rune", "state": state, "questions": questions or {"refund": REFUND}}


def decide(base, body):
    response = requests.post(base + "/v1/decisions", json=body, timeout=300)
    return response.status_code, response.json()


def first_token(base, text):
    response = requests.post(base + "/v1/chat/completions", timeout=120, json={
        "model": "rune", "temperature": 0, "max_tokens": 1, "logprobs": True, "top_logprobs": 5,
        "messages": [{"role": "user", "content": text}]})
    assert response.status_code == 200, response.text
    return response.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]


@contextlib.contextmanager
def sent_decision(base, body):
    """Sends a decision, and hangs up on leaving without reading the answer."""
    host, port = base.removeprefix("http://").split(":")
    payload = json.dumps(body).encode()
    with socket.create_connection((host, int(port))) as connection:
        connection.sendall(b"POST /v1/decisions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                           b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload)
        yield


def wait_for(condition, seconds=30):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.02)


def test_one_question_decisions_are_packed_and_answer_as_alone(tmp_path):
    with server(tmp_path, "--max-num-seqs", "16") as base:
        bodies = [decision(i) for i in range(12)]
        alone = [decide(base, body) for body in bodies]
        assert all(status == 200 for status, _ in alone)
        rounds, prompts = counter(base, "packed_prefill_rounds_total"), counter(base, "packed_prefill_prompts_total")
        with cf.ThreadPoolExecutor(len(bodies)) as pool:
            together = list(pool.map(lambda body: decide(base, body), bodies))
        assert counter(base, "packed_prefill_rounds_total") > rounds
        assert counter(base, "packed_prefill_prompts_total") - prompts >= 6  # most of the 12
        assert [answer["answers"] for _, answer in together] == [answer["answers"] for _, answer in alone]


def test_multi_question_decisions_answer_as_alone_under_concurrency(tmp_path):
    three = {"refund": REFUND, "topic": TOPIC, "billing": {**REFUND, "instructions": "Is this about billing?"}}
    with server(tmp_path, "--max-num-seqs", "16") as base:
        bodies = [decision(i, three) for i in range(8)] + [decision(20 + i) for i in range(8)]

        def packing():
            return counter(base, "packed_prefill_rounds_total"), counter(base, "packed_prefill_prompts_total")

        before = packing()
        alone = [decide(base, body) for body in bodies]
        # Alone, a decision's own questions share a round; sent together, the questions (GPU-prefix
        # readouts) of several decisions do, so the rounds carry more prompts.
        between = packing()
        with cf.ThreadPoolExecutor(8) as pool:
            first = list(pool.map(lambda body: decide(base, body), bodies[:8]))
        after = packing()
        assert [answer["answers"] for _, answer in first] == [answer["answers"] for _, answer in alone[:8]]
        assert between[0] > before[0] and after[0] > between[0]
        assert ((after[1] - between[1]) / (after[0] - between[0]) >
                (between[1] - before[1]) / (between[0] - before[0]))
        with cf.ThreadPoolExecutor(len(bodies)) as pool:
            together = list(pool.map(lambda body: decide(base, body), bodies))
        assert all(status == 200 for status, _ in together)
        assert [answer["answers"] for _, answer in together] == [answer["answers"] for _, answer in alone]


def test_a_chat_prompts_first_token_is_the_same_packed(tmp_path):
    prompts = [f"Describe case {i}: " + " ".join(SENTENCES[: 2 + i % 4]) for i in range(10)]
    # Without the prefix cache, so the second time the prompts are prefilled again, not reused.
    with server(tmp_path, "--max-num-seqs", "16", "--no-enable-prefix-caching") as base:
        alone = [first_token(base, text) for text in prompts]
        rounds = counter(base, "packed_prefill_rounds_total")
        with cf.ThreadPoolExecutor(len(prompts)) as pool:
            packed = list(pool.map(lambda text: first_token(base, text), prompts))
        assert counter(base, "packed_prefill_rounds_total") > rounds
        assert packed == alone


def test_a_prompt_longer_than_the_window_ahead_of_others(tmp_path):
    # A 256-token window and a prompt of about 3,700 tokens sent first: while it is the first staged
    # prompt with more than a window left, a round takes it alone, whole window, with the others
    # staged behind it; its last chunk shares a round with them.
    long_prompt = "Summarize this support log: " + " ".join(SENTENCES * 40)
    prompts = [long_prompt] + [f"Describe case {i}: " + " ".join(SENTENCES[: 2 + i % 4]) for i in range(7)]
    with server(tmp_path, "--max-num-seqs", "16", "--no-enable-prefix-caching",
                "--max-num-batched-tokens", "256") as base:
        alone = [first_token(base, text) for text in prompts]
        rounds = counter(base, "packed_prefill_rounds_total")
        with cf.ThreadPoolExecutor(len(prompts)) as pool:
            head = pool.submit(first_token, base, prompts[0])
            wait_for(lambda: counter(base, 'requests{model="rune",state="prefilling"}') >= 1)
            rest = [pool.submit(first_token, base, text) for text in prompts[1:]]
            packed = [head.result()] + [future.result() for future in rest]
        assert counter(base, "packed_prefill_rounds_total") > rounds
        assert packed == alone


def test_a_decision_waits_for_its_places_before_any_prefill(tmp_path):
    # Four lanes and one queue place: three long chats hold three of the five places, and a
    # four-question decision needs four at once. It used to prefill its shared prefix in a free
    # place and then find the queue full for its questions; now it takes its places first, and
    # waits for them without prefilling anything (surogate_requests{state="reserving"}). A decision
    # ahead of it in the line that hangs up while it waits leaves the line and takes no places.
    four = {"refund": REFUND, "topic": TOPIC, "escalate": {**REFUND, "instructions": "Should a supervisor handle this?"},
            "billing": {**REFUND, "instructions": "Is this about billing?"}}
    # The chats run for seconds: the waiting decision must not time out behind them.
    with server(tmp_path, "--max-num-seqs", "4", "--max-pending-requests", "1",
                "--pending-timeout-ms", "120000") as base:
        assert decide(base, decision(0, four))[0] == 200  # alone it fits
        chat = {"model": "rune", "temperature": 0, "max_tokens": 1500, "ignore_eos": True,
                "messages": [{"role": "user", "content": "Write a long story about a parcel."}]}
        with cf.ThreadPoolExecutor(4) as pool:
            chats = [pool.submit(requests.post, base + "/v1/chat/completions", json=chat, timeout=600) for _ in range(3)]
            deadline = time.monotonic() + 60
            # All three generating, their prompts done.
            while (counter(base, 'requests{model="rune",state="running"}') < 3 or
                   counter(base, 'requests{model="rune",state="prefilling"}') > 0):
                assert time.monotonic() < deadline
                time.sleep(0.05)
            time.sleep(0.2)
            prefilled = counter(base, "prefill_tokens_total")
            reserving = 'requests{model="rune",state="reserving"}'
            with sent_decision(base, decision(2, four)):  # first in line, then gone
                wait_for(lambda: counter(base, reserving) == 1)
            wait_for(lambda: counter(base, reserving) == 0)
            waiting = pool.submit(decide, base, decision(1, four))
            wait_for(lambda: counter(base, reserving) == 1)
            time.sleep(0.5)
            if any(future.done() for future in chats):
                pytest.skip("the chats ended before the decision could be seen waiting")
            assert not waiting.done() and counter(base, "prefill_tokens_total") == prefilled
            for future in chats:
                assert future.result().status_code == 200
            status, body = waiting.result()
            assert status == 200, body
            assert counter(base, "prefill_tokens_total") > prefilled
        # Every place is back: a four-question decision fits at once again.
        started = time.monotonic()
        assert decide(base, decision(3, four))[0] == 200
        assert time.monotonic() - started < 10
