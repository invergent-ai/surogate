"""Responses history belongs to the client; no response-ID state is retained."""

import json
import os

import pytest
import requests

from tests.serve.test_sampling_constraints import server  # noqa: F401

pytestmark = pytest.mark.skipif(
    not os.getenv("SUROGATE_SAMPLING_TEST_ARTIFACT"), reason="requires a prepared GPU artifact"
)


def generate(server, payload):
    response = requests.post(
        server + "/v1/responses",
        json={
            "model": "test",
            "input": "Say hello.",
            "max_output_tokens": 16,
            "temperature": 0,
            **payload,
        },
        timeout=90,
    )
    assert response.ok, response.text
    if payload.get("stream"):
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        assert events[0]["type"] == "response.created"
        assert events[0]["response"]["store"] is False
        result = next(
            event["response"] for event in events if event["type"] in ("response.completed", "response.incomplete")
        )
    else:
        result = response.json()
    assert result["store"] is False
    assert result["previous_response_id"] is None
    return result


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("options", [{}, {"store": False}, {"store": None, "previous_response_id": None}])
def test_responses_are_stateless(server, stream, options):
    result = generate(server, {"stream": stream, **options})
    assert result["output"]
    assert result["usage"]["output_tokens"] > 0


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "field,value,code",
    [
        ("store", True, "response_storage_not_supported"),
        ("previous_response_id", "resp_missing", "previous_response_id_not_supported"),
    ],
)
def test_stateful_requests_fail_before_generation(server, stream, field, value, code):
    response = requests.post(
        server + "/v1/responses",
        json={
            "model": "test",
            "input": "Continue.",
            "stream": stream,
            field: value,
        },
        timeout=30,
    )
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["param"] == field
    assert error["code"] == code


def test_response_storage_routes_removed(server):
    result = generate(server, {})
    path = server + "/v1/responses/" + result["id"]
    for method, suffix in (("GET", ""), ("DELETE", ""), ("GET", "/input_items"), ("POST", "/cancel")):
        response = requests.request(method, path + suffix, timeout=30)
        assert response.status_code == 404, (method, suffix, response.text)


def test_explicit_conversation_history(server):
    question = "Remember the word ORCHID."
    first = generate(server, {"input": question})
    history = [{"role": "user", "content": question}]
    history.extend({key: value for key, value in item.items() if key != "status"} for item in first["output"])
    followup = "What word did I give you?"
    history.append({"role": "user", "content": followup})
    expected = requests.post(
        server + "/v1/responses/input_tokens",
        json={
            "model": "test",
            "input": history,
        },
        timeout=30,
    )
    standalone = requests.post(
        server + "/v1/responses/input_tokens",
        json={
            "model": "test",
            "input": followup,
        },
        timeout=30,
    )
    assert expected.ok and standalone.ok
    assert expected.json()["input_tokens"] > standalone.json()["input_tokens"]
    for stream in (False, True):
        result = generate(server, {"input": history, "stream": stream})
        assert result["usage"]["input_tokens"] == expected.json()["input_tokens"]
