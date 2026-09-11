"""Model tests for vLLM-compatible automatic choice and constrained tools."""
import json
import os

import pytest
import requests

from tests.serve.test_sampling_constraints import chat, server  # noqa: F401

pytestmark = pytest.mark.skipif(not os.getenv("SUROGATE_SAMPLING_TEST_ARTIFACT"), reason="requires a prepared GPU artifact")


def weather(strict=True):
    return {"type": "function", "function": {"name": "weather", "description": "Look up the weather in a city.",
        "strict": strict, "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "enum": ["Paris"]}, "days": {"type": "integer", "minimum": 1, "maximum": 3}},
            "required": ["city", "days"], "additionalProperties": False}}}


def verify_calls(calls):
    assert calls
    for call in calls:
        assert call["function"]["name"] == "weather"
        args = json.loads(call["function"]["arguments"])
        assert set(args) == {"city", "days"}
        assert args["city"] == "Paris" and type(args["days"]) is int and 1 <= args["days"] <= 3


@pytest.mark.parametrize("choice", ["required", {"type": "function", "function": {"name": "weather"}}])
@pytest.mark.parametrize("temperature", [0, 0.8])
def test_required_and_named(server, choice, temperature):
    response = chat(server, tools=[weather()], tool_choice=choice, temperature=temperature,
        top_k=0, top_p=1, parallel_tool_calls=False, messages=[{"role": "user", "content": "Say hello. Do not call any tool."}])
    assert response.ok, response.text
    message = response.json()["choices"][0]
    assert message["finish_reason"] == "tool_calls", response.text
    calls = message["message"]["tool_calls"]
    assert len(calls) == 1
    verify_calls(calls)


@pytest.mark.parametrize("strict", [False, True])
def test_auto_can_answer_normally(server, strict):
    response = chat(server, tools=[weather(strict)], tool_choice="auto", parallel_tool_calls=False,
        messages=[{"role": "user", "content": "What is 2 + 2? Answer directly without tools."}])
    assert response.ok, response.text
    choice = response.json()["choices"][0]
    assert not choice["message"].get("tool_calls"), response.text
    assert "4" in choice["message"]["content"], response.text


def test_auto_strict_tool(server):
    response = chat(server, tools=[weather()], tool_choice="auto", parallel_tool_calls=False,
        messages=[{"role": "user", "content": "Use weather to get the weather for Paris for 2 days."}])
    assert response.ok, response.text
    verify_calls(response.json()["choices"][0]["message"].get("tool_calls", []))


@pytest.mark.parametrize("choice", ["required", {"type": "function", "name": "weather"}])
@pytest.mark.parametrize("stream", [False, True])
def test_responses_tool_parity(server, choice, stream):
    fn = weather()["function"]
    payload = {"model": "test", "input": "Say hello. Do not use tools.", "tools": [{"type": "function", **fn}],
        "tool_choice": choice, "parallel_tool_calls": False, "temperature": 0, "max_output_tokens": 128, "stream": stream}
    response = requests.post(server + "/v1/responses", json=payload, timeout=90)
    assert response.ok, response.text
    if stream:
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        assert not any(event["type"] == "response.output_text.delta" and event.get("delta") for event in events), response.text
        result = next(event["response"] for event in events if event["type"] == "response.completed")
    else:
        result = response.json()
    assert result["tool_choice"] == choice and result["parallel_tool_calls"] is False
    calls = [item for item in result["output"] if item["type"] == "function_call"]
    assert len(calls) == 1, response.text
    verify_calls([{"function": call} for call in calls])
    stored = requests.get(server + "/v1/responses/" + result["id"], timeout=10)
    assert stored.ok and stored.json()["tools"][0]["strict"] is True


def test_none_does_not_enforce_unused_schema(server):
    tool = weather()
    tool["function"]["parameters"] = {"not": {}}
    response = chat(server, tools=[tool], tool_choice="none", messages=[{"role": "user", "content": "Say hello."}])
    assert response.ok and not response.json()["choices"][0]["message"].get("tool_calls"), response.text


def test_auto_nonstrict_does_not_compile_schema(server):
    tool = weather(False)
    tool["function"]["parameters"] = {"not": {}}
    response = chat(server, tools=[tool], tool_choice="auto", parallel_tool_calls=False,
        messages=[{"role": "user", "content": "What is 2 + 2? Answer without tools."}])
    assert response.ok, response.text


def test_strict_unsupported_schema_rejected(server):
    tool = weather()
    tool["function"]["parameters"] = {"not": {}}
    response = chat(server, tools=[tool], tool_choice="auto")
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == "tools"


def test_auto_parallel_false_keeps_generation_and_first_call(server):
    payload = {"tools": [weather()], "tool_choice": "auto", "max_tokens": 256,
        "messages": [{"role": "user", "content": "Call weather twice: once for Paris for 1 day, and once for Paris for 2 days. Make both calls now."}]}
    many = chat(server, parallel_tool_calls=True, **payload)
    one = chat(server, parallel_tool_calls=False, **payload)
    assert many.ok and one.ok, many.text + one.text
    before = many.json()["choices"][0]
    after = one.json()["choices"][0]
    assert before["token_ids"] == after["token_ids"]
    verify_calls(before["message"].get("tool_calls", []))
    assert len(after["message"]["tool_calls"]) == 1
    assert after["message"]["tool_calls"][0]["function"] == before["message"]["tool_calls"][0]["function"]


def test_tool_constraint_precedes_response_format(server):
    response = chat(server, tools=[weather()], tool_choice="auto", parallel_tool_calls=False,
        response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": {"const": "ordinary answer"}}},
        messages=[{"role": "user", "content": "Use weather to get the weather for Paris for 2 days."}])
    assert response.ok, response.text
    verify_calls(response.json()["choices"][0]["message"].get("tool_calls", []))
