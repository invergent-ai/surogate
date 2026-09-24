"""Decisions calibration end to end (--decision-temperature). GPU; pending until run on a free card.

Set SUROGATE_DECISIONS_TEST_ARTIFACT to a prepared chat model (.sinfer) and select one free GPU with
CUDA_VISIBLE_DEVICES; SUROGATE_DECISIONS_TEST_ARGS may add engine flags as a JSON list. The server is
started twice, at T = 1 and at T = 2.5, and every tempered answer is checked against the untempered
one: softmax(log p / T) over the options is what the calibrated server must return, whichever route
produced the readout -- the shared GPU prefix with per-question suffixes, whole prompts, option
letters or codebook codes -- and on every route alias.
"""

import contextlib
import json
import math
import os
import socket
import subprocess
import time

import pytest
import requests

from surogate.cli.serve import _resolve_binary

pytestmark = pytest.mark.skipif(not os.getenv("SUROGATE_DECISIONS_TEST_ARTIFACT"),
                                reason="needs a prepared chat model and a free GPU")

T = 2.5
ROUTES = ("/api/alpha/decisions", "/v1/decisions", "/api/v1/decisions")

STATE = {
    "ticket": 8812,
    "customer": "Dana Whitfield",
    "message": ("The order arrived nine days late, the box was crushed on one corner and the ceramic "
                "mug inside was chipped. I contacted support twice last week and nobody answered. I want "
                "my money back, and honestly I am not sure I will order from this shop again."),
    "history": [{"order": 7310, "status": "delivered"}, {"order": 8120, "status": "returned"}],
}


def letters(n):
    return {f"o{i}": f"option number {i}" for i in range(n)}


def request(questions):
    return {"model": "test", "state": STATE, "questions": questions}


SENTIMENT = {"type": "choice", "instructions": "What is the customer's sentiment?",
             "criteria": {"positive": "The message is positive", "neutral": "Neither",
                          "negative": "The message is negative"}}
REFUND = {"type": "noul", "instructions": "Does the customer ask for a refund?",
          "criteria": {"true": "A refund is requested", "false": "No refund is requested"}}
URGENCY = {"type": "score", "instructions": "How urgent is this ticket?",
           "criteria": ["Not urgent", "Somewhat urgent", "Urgent", "Critical"]}
MANY = {"type": "choice", "instructions": "Which option number is written first in this list?",
        "criteria": letters(30)}
LEVELS = {"type": "score", "instructions": "Rate the customer's anger from 0 to 39.",
          "criteria": [f"level {i}" for i in range(40)]}

REQUESTS = {
    # Letters, several questions over one state: the shared GPU prefix and per-question suffixes.
    "shared_letters": request({"sentiment": SENTIMENT, "refund": REFUND, "urgency": URGENCY}),
    # Codebook codes past 26 options, two extended questions: shared prefix again, code labels.
    "shared_codes": request({"many": MANY, "levels": LEVELS}),
    # Mixed system prompts: the shared part stops before the state, and is normally shorter than
    # the 47-token prefill floor, so the questions run as whole prompts (tokenizer-dependent; the
    # route checks below do not rely on it).
    "mixed_whole": request({"sentiment": SENTIMENT, "many": MANY}),
    # One question is always a whole prompt.
    "single": request({"refund": REFUND}),
}


@contextlib.contextmanager
def server(temperature, root):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    log = root / f"server-{temperature}.log"
    jsonl = root / f"requests-{temperature}.jsonl"
    cmd = [_resolve_binary("server"), os.environ["SUROGATE_DECISIONS_TEST_ARTIFACT"],
           "--host", "127.0.0.1", "--port", str(port), "--served-model-name", "test",
           "--max-model-len", "4096", "--max-num-seqs", "4", "--request-log-jsonl", str(jsonl)]
    if temperature is not None:
        cmd += ["--decision-temperature", str(temperature)]
    cmd += json.loads(os.getenv("SUROGATE_DECISIONS_TEST_ARGS", "[]"))
    base = f"http://127.0.0.1:{port}"
    with log.open("w") as output:
        process = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                assert process.poll() is None, log.read_text()
                try:
                    if requests.get(base + "/v1/models", timeout=1).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.2)
            else:
                pytest.fail(log.read_text())
            yield base, jsonl, log
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def ask(base, body, route=ROUTES[0]):
    response = requests.post(base + route, json=body, timeout=300)
    assert response.ok, response.text
    return response.json()["answers"]


def records(jsonl, event):
    return [r for r in map(json.loads, jsonl.read_text().splitlines()) if r["event"] == event]


def distribution(answer):
    if answer["type"] == "noul":
        return {"false": 1.0 - answer["noul"], "true": answer["noul"]}
    return answer["probabilities"]


def tempered(probabilities, temperature):
    """softmax(log p / T): what the calibrated server must return, from the untempered answer."""
    logs = {k: (math.log(p) if p > 0 else -math.inf) for k, p in probabilities.items()}
    peak = max(logs.values())
    weights = {k: math.exp((v - peak) / temperature) for k, v in logs.items()}
    total = math.fsum(weights.values())
    return {k: w / total for k, w in weights.items()}


@pytest.fixture(scope="module")
def answers(tmp_path_factory):
    root = tmp_path_factory.mktemp("decision-temperature")
    out = {}
    for temperature in (None, T):  # the default server first, then the calibrated one
        with server(temperature, root) as (base, jsonl, log):
            got = {name: ask(base, body) for name, body in REQUESTS.items()}
            aliases = {route: ask(base, REQUESTS["shared_letters"], route) for route in ROUTES}
            out[temperature] = {"answers": got, "aliases": aliases, "jsonl": jsonl, "log": log}
    return out


def test_tempered_answers_follow_from_the_untempered_readout(answers):
    for name in REQUESTS:
        plain, calibrated = answers[None]["answers"][name], answers[T]["answers"][name]
        assert list(plain) == list(calibrated)
        for question in plain:
            p1, pt = plain[question], calibrated[question]
            expected = tempered(distribution(p1), T)
            got = distribution(pt)
            assert list(got) == list(expected), (name, question)
            for key in expected:
                assert got[key] == pytest.approx(expected[key], abs=1e-6), (name, question, key)
            assert math.fsum(got.values()) == pytest.approx(1.0, abs=1e-12)
            if p1["type"] == "choice":
                assert pt["choice"] == p1["choice"], (name, question)  # the argmax is invariant
                n = len(got)
                assert pt["confidence"] == pytest.approx((max(got.values()) - 1 / n) / (1 - 1 / n), abs=1e-12)
                assert pt["confidence"] <= p1["confidence"] + 1e-12  # T > 1 never sharpens
            if p1["type"] == "score":
                assert pt["score"] == pytest.approx(sum(int(k) * v for k, v in got.items()), abs=1e-9)


def test_every_route_alias_is_calibrated(answers):
    reference = answers[T]["aliases"][ROUTES[0]]
    for route in ROUTES[1:]:
        other = answers[T]["aliases"][route]
        for question in reference:
            for key, value in distribution(reference[question]).items():
                assert distribution(other[question])[key] == pytest.approx(value, abs=1e-9), (route, question)
    plain = answers[None]["aliases"][ROUTES[0]]
    assert any(abs(distribution(plain[q])[k] - v) > 1e-6
               for q in reference for k, v in distribution(reference[q]).items())


def test_temperature_is_recorded_and_the_prefix_routes_were_taken(answers):
    for temperature, expected in ((None, 1.0), (T, T)):
        jsonl = answers[temperature]["jsonl"]
        (start,) = records(jsonl, "server_start")
        assert start["server"]["decision_temperature"] == expected
        done = records(jsonl, "request_done")
        assert len(done) == len(REQUESTS) + len(ROUTES)
        assert all(r["request"]["decisions"]["temperature"] == expected for r in done)
        # Requests were sent one at a time, in REQUESTS order, so the records line up with them.
        shared = {name: r["request"]["decisions"]["shared_prefix_tokens"] for name, r in zip(REQUESTS, done)}
        assert shared["shared_letters"] > 0, "the letters request did not take the shared GPU prefix route"
        assert shared["shared_codes"] > 0, "the codebook request did not take the shared GPU prefix route"
        assert shared["single"] == 0, "a one-question request did not run as a whole prompt"
    assert "decisions calibrated" in answers[T]["log"].read_text()
    assert "decisions calibrated" not in answers[None]["log"].read_text()
