"""Benchmark failures must not be reported as successful throughput."""

import io
import json

import pytest

from surogate.serve.tools.bench import embeddings_bench as bench


def test_running_model_and_natural_token_windows(monkeypatch):
    inputs = bench.build_inputs(4, 2, [10, 20, 30])
    assert inputs == [[10, 20, 30, 10], [30, 10, 20, 30]]

    def respond(request, timeout):
        body = json.loads(request.data)
        assert body == {"input": inputs, "encoding_format": "float"}
        return io.StringIO(json.dumps({"data": [
            {"index": index, "embedding": [0.6, 0.8]} for index in range(2)
        ], "usage": {"prompt_tokens": 8}}))

    monkeypatch.setattr(bench.urllib.request, "urlopen", respond)
    assert bench.post("http://test", None, inputs, 1) == (2, 2)


@pytest.mark.parametrize("response", [
    {"data": [], "usage": {"prompt_tokens": 1}},
    {"data": [{"index": 0, "embedding": [float("nan")]}], "usage": {"prompt_tokens": 1}},
    {"data": [{"index": 1, "embedding": [1.]}], "usage": {"prompt_tokens": 1}},
    {"data": [{"index": 0, "embedding": [1.]}], "usage": {"prompt_tokens": 0}},
])
def test_invalid_results_count_as_failed_requests(monkeypatch, response):
    monkeypatch.setattr(bench.urllib.request, "urlopen",
                        lambda *args, **kwargs: io.StringIO(json.dumps(response)))
    result = bench.run("http://test", None, [[42]], 1, 2, 3, 1)
    assert result.failures == 3
    assert result.vectors == 0 and not result.latencies


def test_failed_warmup_fails_the_benchmark(monkeypatch):
    monkeypatch.setattr(bench, "run", lambda *args: bench.Result(failures=1))
    assert bench.main(["--url", "http://test", "--warmup", "1"]) == 1
