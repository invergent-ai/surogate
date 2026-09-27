"""Decisions option-order averaging, end to end on a served decision model.

GPU. Skipped unless a served decision model is given, either
- SUROGATE_DECISIONS_ORDER_URL=http://127.0.0.1:PORT: a running server, otherwise idle while this runs (its model id
  is read from /v1/models), or
- SUROGATE_DECISIONS_ORDER_ARTIFACT=<prepared .sinfer>: the test starts the server on the GPU that
  CUDA_VISIBLE_DEVICES selects, with SUROGATE_DECISIONS_ORDER_ARGS (a JSON list) added.

SUROGATE_DECISIONS_ORDER_TEMPERATURE is the server's --decision-temperature T (default 1; a server this test starts
is given it when set). The reference readings come back tempered, so each is untempered (p ~ p_T ** T), the two are
averaged, and the mean is tempered once (softmax(log(mean) / T)), as the endpoint does.

`"order_averaging": true` reads every choice and noul question a second time with its options reversed and answers
from the mean of the two readings. The reference is v1 itself: the same request with every mirrored reading sent as
a question of its own (a choice question's criteria in reverse order; a noul question as a choice question with
`true` first, which renders the same prompt), whose rows are the order-averaged request's rows token for token. A
lone question stays on v1's whole-prompt route with its mirrored reading, so its reference is two one-question v1
requests, the question and its mirrored reading, each a whole prompt.

The lone-question comparison is exact only on a model whose prefill does not depend on the round width, such as
a BF16 or W8 mixture (#225). A routed-NVFP4 export picks its expert kernels by width, so the endpoint, which reads
the question and its mirror as two whole prompts in one wave, can differ from the two single-prompt references by
~1e-2 in a probability; for such a model raise SUROGATE_DECISIONS_ORDER_TOLERANCE or use its W8 conversion. Rune v3:
9 passed on the W8 conversion; on NVFP4 the one-question and extended cases differed by up to 0.044.

1. Without the field, with `false` and with `null` the response is v1's: the same answers and usage.
2. With it, every choice and noul answer is the mean of the reference's two readings (choice, confidence, noul and
   probabilities within SUROGATE_DECISIONS_ORDER_TOLERANCE, default 1e-5: the endpoint rounds the mean's log to
   float), every score answer is v1's, `input_tokens` is the reference's and `output_tokens` the question count.
3. With thinking also on the request is refused with HTTP 400, param order_averaging (not supported together yet).
4. A value that is not a boolean is refused with HTTP 400, param order_averaging.

The console lines of a server this test started carry ` order_averaging=on mirrored=N`. Latencies are printed.
"""

import json
import os
import socket
import subprocess
import time

import pytest
import requests

pytestmark = pytest.mark.skipif(
    not (os.getenv("SUROGATE_DECISIONS_ORDER_URL") or os.getenv("SUROGATE_DECISIONS_ORDER_ARTIFACT")),
    reason="needs a served decision model and a free GPU")

TOLERANCE = float(os.getenv("SUROGATE_DECISIONS_ORDER_TOLERANCE", "1e-5"))
TEMPERATURE_SET = os.getenv("SUROGATE_DECISIONS_ORDER_TEMPERATURE")
TEMPERATURE = float(TEMPERATURE_SET or "1.0")
TIMEOUT = 1800
MIRROR = "__mirrored"

STATE = {"ticket": "Order 8812 arrived late and the box was crushed. I asked for a refund twice and nobody answered.",
         "customer_since": 2019, "previous_complaints": 0}
QUESTIONS = {
    "tone": {"type": "choice", "instructions": "What is the tone of the ticket?",
             "criteria": {"calm": "Calm", "annoyed": "Annoyed", "furious": "Furious", "sarcastic": "Sarcastic"}},
    "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?",
               "criteria": {"true": "A refund is requested", "false": "No refund is requested"}},
    "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                "criteria": ["Not urgent", "Somewhat urgent", "Very urgent"]},
    "loyal": {"type": "noul", "instructions": "Is this a long-standing customer?",
              "criteria": {"false": "No", "true": "Yes"}},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Billing", "shipping": "Shipping and logistics", "support": "General support"}},
}
EXTENDED = {"city": {"type": "choice", "instructions": "Which of these cities is the capital of Romania?",
                     "criteria": {f"c{i}": name for i, name in enumerate(
                         ["Cluj", "Iasi", "Timisoara", "Constanta", "Craiova", "Brasov", "Galati", "Ploiesti",
                          "Oradea", "Braila", "Arad", "Pitesti", "Sibiu", "Bacau", "Targu Mures", "Baia Mare",
                          "Buzau", "Botosani", "Satu Mare", "Ramnicu Valcea", "Suceava", "Piatra Neamt",
                          "Drobeta", "Targu Jiu", "Focsani", "Bistrita", "Tulcea", "Bucharest", "Resita"])}}}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    url = os.getenv("SUROGATE_DECISIONS_ORDER_URL")
    if url:
        url = url.rstrip("/")
        yield url, requests.get(url + "/v1/models", timeout=30).json()["data"][0]["id"], None
        return
    from surogate.cli.serve import _resolve_binary
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    log = tmp_path_factory.mktemp("decisions-order") / "server.log"
    cmd = [_resolve_binary("server"), os.environ["SUROGATE_DECISIONS_ORDER_ARTIFACT"],
           "--host", "127.0.0.1", "--port", str(port), "--served-model-name", "rune",
           "--max-model-len", "16384", "--max-num-seqs", "16",
           *(["--decision-temperature", TEMPERATURE_SET] if TEMPERATURE_SET else []),
           *json.loads(os.getenv("SUROGATE_DECISIONS_ORDER_ARGS", "[]"))]
    url = f"http://127.0.0.1:{port}"
    with log.open("w") as output:
        process = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                assert process.poll() is None, log.read_text()
                try:
                    if requests.get(url + "/v1/models", timeout=1).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.5)
            else:
                pytest.fail(log.read_text())
            yield url, "rune", log
        finally:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def decide(url, body):
    started = time.monotonic()
    response = requests.post(url + "/v1/decisions", json=body, timeout=TIMEOUT)
    assert response.status_code == 200, response.text
    return response.json(), time.monotonic() - started


def mirrored_question(q):
    """The mirrored reading as a v1 question: the same prompt, options reversed."""
    if q["type"] == "noul":
        return {"type": "choice", "instructions": q["instructions"],
                "criteria": {"true": q["criteria"]["true"], "false": q["criteria"]["false"]}}
    return {**q, "criteria": dict(reversed(list(q["criteria"].items())))}


def reference_body(body):
    questions = dict(body["questions"])
    for name, q in body["questions"].items():
        if q["type"] in ("choice", "noul"):
            questions[name + MIRROR] = mirrored_question(q)
    return {k: v for k, v in body.items() if k not in ("order_averaging", "questions")} | {"questions": questions}


def reference(url, body):
    """v1's answers for every reading of an order-averaged request, and the input tokens v1 counts for them."""
    ref = reference_body(body)
    if len(body["questions"]) > 1:
        response, _ = decide(url, ref)
        return response["answers"], response["usage"]["input_tokens"]
    # A lone question and its mirrored reading are two whole prompts: one v1 request each.
    answers, tokens = {}, 0
    for name, q in ref["questions"].items():
        response, _ = decide(url, ref | {"questions": {name: q}})
        answers[name] = response["answers"][name]
        tokens += response["usage"]["input_tokens"]
    return answers, tokens


def untempered(probabilities):
    """A reading's T = 1 distribution from its tempered one: p ~ p_T ** T."""
    raised = [p**TEMPERATURE for p in probabilities]
    total = sum(raised)
    return [p / total for p in raised]


def tempered(probabilities):
    """softmax(log(p) / T), as the endpoint tempers the mean (a zero stays zero)."""
    raised = [p ** (1 / TEMPERATURE) for p in probabilities]
    total = sum(raised)
    return [p / total for p in raised]


def expected_answer(q, as_sent, mirrored):
    """v1's answer arithmetic on the mean of the two untempered readings, tempered once."""
    if q["type"] == "noul":
        p = untempered([1 - as_sent["noul"], as_sent["noul"]])
        r = untempered([mirrored["probabilities"]["false"], mirrored["probabilities"]["true"]])
        return {"type": "noul", "noul": tempered([(p[0] + r[0]) / 2, (p[1] + r[1]) / 2])[1]}
    keys = list(q["criteria"])
    p = untempered([as_sent["probabilities"][k] for k in keys])
    r = untempered([mirrored["probabilities"][k] for k in keys])
    mean = dict(zip(keys, tempered([(a + b) / 2 for a, b in zip(p, r)])))
    total = sum(mean.values())
    peak = max(mean.values())
    n = len(keys)
    return {"type": "choice", "choice": next(k for k in keys if mean[k] == peak),
            "confidence": (peak / total - 1 / n) / (1 - 1 / n), "probabilities": mean}


def assert_close(answer, want, where):
    assert answer["type"] == want["type"], where
    for field in ("noul", "confidence", "score"):
        if field in want:
            assert abs(answer[field] - want[field]) <= TOLERANCE, (where, field, answer[field], want[field])
    if "probabilities" in want:
        assert list(answer["probabilities"]) == list(want["probabilities"]), where
        for key, value in want["probabilities"].items():
            assert abs(answer["probabilities"][key] - value) <= TOLERANCE, (where, key, answer, want)
        if "choice" in want:
            ranked = sorted(want["probabilities"].values(), reverse=True)
            if ranked[0] - ranked[1] > TOLERANCE:  # a near-tie may round either way
                assert answer["choice"] == want["choice"], (where, answer, want)


@pytest.mark.parametrize("questions", [QUESTIONS, {"tone": QUESTIONS["tone"]}, EXTENDED],
                         ids=["mixed", "one-question", "extended"])
def test_order_averaging_is_the_mean_of_both_readings(server, questions):
    url, model, log = server
    body = {"model": model, "state": STATE, "questions": questions}
    v1, v1_seconds = decide(url, body)
    for off in (False, None):
        same, _ = decide(url, body | {"order_averaging": off})
        assert same["answers"] == v1["answers"] and same["usage"] == v1["usage"], off

    averaged, averaged_seconds = decide(url, body | {"order_averaging": True})
    answers, input_tokens = reference(url, body)
    mirrored = [name for name, q in questions.items() if q["type"] in ("choice", "noul")]
    print(f"\n{len(questions)} questions, {len(mirrored)} mirrored: v1 {v1_seconds:.3f}s, "
          f"order averaging {averaged_seconds:.3f}s; input tokens {v1['usage']['input_tokens']} -> "
          f"{averaged['usage']['input_tokens']}")
    assert list(averaged["answers"]) == list(questions)
    assert averaged["usage"]["output_tokens"] == len(questions)
    assert averaged["usage"]["input_tokens"] == input_tokens
    for name, q in questions.items():
        answer = averaged["answers"][name]
        if name in mirrored:
            want = expected_answer(q, answers[name], answers[name + MIRROR])
        else:
            want = answers[name]
            assert answer.keys() == want.keys(), name
        assert_close(answer, want, name)
    if log is not None:
        assert f" order_averaging=on mirrored={len(mirrored)}" in log.read_text()


def test_thinking_with_order_averaging_is_refused(server):
    url, model, _ = server
    body = {"model": model, "state": STATE, "questions": QUESTIONS, "order_averaging": True, "thinking": True}
    response = requests.post(url + "/v1/decisions", timeout=60, json=body)
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == "invalid_decisions_request" and error["param"] == "order_averaging", error
    assert "cannot be combined with thinking" in error["message"], error


@pytest.mark.parametrize("value", ["true", 1, 0, {"enabled": True}, [True]])
def test_an_order_averaging_value_that_is_not_a_boolean_is_refused(server, value):
    url, model, _ = server
    response = requests.post(url + "/v1/decisions", timeout=60, json={
        "model": model, "state": "s", "order_averaging": value,
        "questions": {"x": {"type": "noul", "instructions": "i", "criteria": {"true": "t", "false": "f"}}}})
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == "invalid_decisions_request" and error["param"] == "order_averaging", error
