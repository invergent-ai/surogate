"""Numerical and protocol coverage for requested native token scores."""
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from tests.serve.test_sampling_constraints import chat, server  # noqa: F401

pytestmark = pytest.mark.skipif(not os.getenv("SUROGATE_SAMPLING_TEST_ARTIFACT"), reason="requires a prepared GPU artifact")


def scores(response):
    assert response.ok, response.text
    return response.json()["choices"][0]["logprobs"]["content"]


@pytest.mark.parametrize("count", [0, 5, 20])
def test_generated_alternatives(server, count):
    response = chat(server, max_tokens=12, logprobs=True, top_logprobs=count, ignore_eos=True)
    entries = scores(response)
    assert len(entries) == 12
    for entry in entries:
        assert math.isfinite(entry["logprob"]) and entry["logprob"] <= 0
        top = entry["top_logprobs"]
        assert len(top) == count
        assert all(a["logprob"] >= b["logprob"] for a, b in zip(top, top[1:]))
        if count:
            assert entry["bytes"] == top[0]["bytes"]
            assert entry["logprob"] == top[0]["logprob"]


@pytest.mark.parametrize("sampling", [{}, {"temperature": 0.7}, {"temperature": 0.7, "logit_bias": {"100": 100}}])
@pytest.mark.parametrize("output_tokens", [8, 32])
def test_prompt_scores_match_next_token_predictions(server, sampling, output_tokens):
    # The longer sequence spans multiple rounds even with the maximum DFlash window.
    first = chat(server, max_tokens=output_tokens, logprobs=True, top_logprobs=5, ignore_eos=True, **sampling)
    generated = scores(first)
    body = first.json()
    prompt = body["prompt_token_ids"]
    tokens = body["choices"][0]["token_ids"]
    # The same sequence evaluated with teacher forcing must assign the same
    # target-model probabilities, including the speculative accept/reject path.
    for _ in range(2):
        replay = chat(server, tokens=prompt + tokens, max_tokens=1, prompt_logprobs=5)
        assert replay.ok, replay.text
        data = replay.json()["prompt_logprobs"]
        assert len(data) == len(prompt) + len(tokens)
        assert data[0] is None
        for index, entry in enumerate(generated):
            selected = data[len(prompt) + index][str(tokens[index])]
            assert selected["logprob"] == pytest.approx(entry["logprob"], abs=0.12)
            assert selected["rank"] >= 1


def test_prompt_zero_and_concurrent_requests(server):
    def run(i):
        response = chat(server, tokens=[100 + i, 105 + i, 110 + i], max_tokens=3,
                        prompt_logprobs=0, logprobs=True, top_logprobs=i, ignore_eos=True)
        assert len(scores(response)) == 3
        data = response.json()["prompt_logprobs"]
        assert data[0] is None
        assert len(data[1]) == len(data[2]) == 1
        assert str(105 + i) in data[1]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, range(4)))


def test_scores_ignore_sampling_transformations(server):
    baseline = chat(server, max_tokens=1, logprobs=True, top_logprobs=5)
    expected = scores(baseline)[0]["top_logprobs"]
    biased = chat(server, max_tokens=1, logprobs=True, top_logprobs=5,
                  temperature=0.5, logit_bias={"100": 100}, top_k=1)
    entry = scores(biased)[0]
    assert biased.json()["choices"][0]["token_ids"] == [100]
    for actual, reference in zip(entry["top_logprobs"], expected):
        assert actual["bytes"] == reference["bytes"]
        assert actual["logprob"] == pytest.approx(reference["logprob"], abs=0.02)
    assert entry["logprob"] < entry["top_logprobs"][0]["logprob"]


def test_streaming_scores(server):
    response = chat(server, max_tokens=32, stream=True, logprobs=True, top_logprobs=3,
                    prompt_logprobs=0, ignore_eos=True)
    assert response.ok, response.text
    chunks = [json.loads(line[6:]) for line in response.content.decode("utf-8").splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    details = [chunk for chunk in chunks if "prompt_logprobs" in chunk]
    assert len(details) == 1
    scored = [chunk for chunk in chunks if chunk["choices"] and chunk["choices"][0].get("logprobs")]
    assert len(scored) > 1
    assert sum(len(chunk["choices"][0]["logprobs"]["content"]) for chunk in scored) == 32
    first = chunks.index(scored[0])
    assert any(chunk["choices"] and chunk["choices"][0].get("delta", {}).get("content") for chunk in chunks[first+1:])
    assert details[0]["prompt_logprobs"][0] is None


@pytest.mark.parametrize("extra", [
    {"top_logprobs": 2}, {"logprobs": 1}, {"logprobs": True, "top_logprobs": 21},
    {"logprobs": True, "top_logprobs": -1}, {"prompt_logprobs": 21},
    {"prompt_logprobs": -1}, {"prompt_logprobs": True}, {"prompt_logprobs": 1.5},
])
def test_invalid_score_requests(server, extra):
    response = chat(server, max_tokens=1, **extra)
    assert response.status_code == 400, response.text


@pytest.mark.parametrize("stream", [False, True])
def test_completion_scores(server, stream):
    response = requests.post(server + "/v1/completions", json={"model": "test", "prompt": "Hello world",
        "logprobs": 3, "prompt_logprobs": 0, "max_tokens": 4, "ignore_eos": True,
        "return_token_ids": True, "stream": stream}, timeout=90)
    assert response.ok, response.text
    if stream:
        chunks = [json.loads(line[6:]) for line in response.content.decode("utf-8").splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        body = next(chunk for chunk in chunks if "prompt_logprobs" in chunk)
    else:
        body = response.json()
    details = body["choices"][0]["logprobs"]
    if stream:
        parts = [chunk["choices"][0]["logprobs"] for chunk in chunks
                 if chunk["choices"] and chunk["choices"][0].get("logprobs")]
        assert len(parts) > 1
        details = {key: [value for part in parts for value in part[key]] for key in parts[0]}
        assert details["text_offset"] == sorted(details["text_offset"])
    assert len(details["tokens"]) == len(details["token_logprobs"]) == 4
    assert all(len(top) == 3 for top in details["top_logprobs"])
    assert body["prompt_logprobs"][0] is None


def test_chunked_prompt(server):
    # Cross the server's prefill chunk boundaries and verify the shifted target
    # at every position, including the first token of each continuation chunk.
    tokens = [100 + i % 20 for i in range(385)]
    response = chat(server, tokens=tokens, prompt_logprobs=0, max_tokens=1)
    assert response.ok, response.text
    data = response.json()["prompt_logprobs"]
    assert len(data) == len(tokens)
    assert data[0] is None
    assert all(str(token) in score for token, score in zip(tokens[1:], data[1:]))


@pytest.mark.parametrize("stream", [False, True])
def test_responses_scores(server, stream):
    response = requests.post(server + "/v1/responses", json={"model": "test", "input": "Say hello.",
        "include": ["message.output_text.logprobs"], "top_logprobs": 3, "max_output_tokens": 16,
        "stream": stream}, timeout=90)
    assert response.ok, response.text
    if stream:
        events = [json.loads(line[6:]) for line in response.content.decode("utf-8").splitlines()
                  if line.startswith("data: ")]
        done = next(event for event in events if event["type"] == "response.output_text.done")
        body = next(event for event in events if event["type"] in ("response.completed", "response.incomplete"))["response"]
    else:
        body = response.json()
    message = next(item for item in body["output"] if item["type"] == "message")
    entries = message["content"][0]["logprobs"]
    assert 0 < len(entries) <= 16
    assert all(len(entry["top_logprobs"]) == 3 for entry in entries)
    if stream:
        assert done["logprobs"] == entries
        deltas = [event for event in events if event["type"] == "response.output_text.delta" and event["logprobs"]]
        assert len(deltas) > 1
        assert [entry for event in deltas for entry in event["logprobs"]] == entries
    assert body["store"] is False


def test_constrained_scores(server):
    response = chat(server, max_tokens=32, logprobs=True, top_logprobs=5, temperature=0.7,
        response_format={"type": "json_schema", "json_schema": {"name": "answer", "strict": True,
            "schema": {"const": {"answer": "yes"}}}})
    entries = scores(response)
    assert entries and all(math.isfinite(entry["logprob"]) for entry in entries)
    assert json.loads(response.json()["choices"][0]["message"]["content"]) == {"answer": "yes"}


@pytest.mark.parametrize("length", [1, 64])
def test_cached_scores_extend_and_upgrade(server, length):
    prompt = [2000] + [100 + i % 20 for i in range(length - 1)]
    first = chat(server, tokens=prompt, max_tokens=1, prompt_logprobs=5, logprobs=True, top_logprobs=5).json()
    for count in (0, 5, 20):
        response = chat(server, tokens=prompt, max_tokens=1, prompt_logprobs=count)
        assert response.ok, response.text
        data = response.json()["prompt_logprobs"]
        records = [json.loads(line) for line in server.records.read_text().splitlines()]
        done = [record for record in records if "result" in record][-1]
        if count <= 5 or length == 1:
            assert done["result"]["prefix_cache_hit_tokens"] == len(prompt)
        else:
            assert done["result"]["prefix_cache_hit_tokens"] == 0
        for i, token in enumerate(prompt[1:], 1):
            assert data[i][str(token)]["logprob"] == pytest.approx(first["prompt_logprobs"][i][str(token)]["logprob"], abs=0.02)
            assert max(1, count) <= len(data[i]) <= count + 1
    extended = prompt + first["choices"][0]["token_ids"] + [123]
    response = chat(server, tokens=extended, prompt_logprobs=5, max_tokens=1)
    assert response.ok, response.text
    data = response.json()["prompt_logprobs"]
    assert len(data) == len(extended)
    boundary = str(first["choices"][0]["token_ids"][0])
    assert data[len(prompt)][boundary]["logprob"] == pytest.approx(
        first["choices"][0]["logprobs"]["content"][0]["logprob"], abs=0.02)
    records = [json.loads(line) for line in server.records.read_text().splitlines()]
    done = [record for record in records if "result" in record][-1]
    assert done["result"]["prefix_cache_hit_tokens"] == len(prompt)
    if length < 3:
        return
    # A divergent prefix must not receive the cached values from the old input.
    changed = prompt.copy()
    changed[2] = 2500
    response = chat(server, tokens=changed, prompt_logprobs=0, max_tokens=1)
    assert response.ok, response.text
    assert "2500" in response.json()["prompt_logprobs"][2]
    assert str(prompt[2]) not in response.json()["prompt_logprobs"][2]
