"""Admission, publication, sampling and model coverage without a GPU."""

import concurrent.futures
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import requests
import torch

from surogate.grpo.decode_scheduler import DecodeCapacityError
from surogate.grpo.shared_model import SharedModelServer, shared_execution
from tests.grpo.shared_model_configs import configurations


@pytest.mark.parametrize("name,config", configurations().items())
def test_supported_training_families_select_shared_execution(name, config):
    assert shared_execution(config) == ("serve" if name in ("qwen3", "qwen3_5") else "training")
    assert shared_execution(config | {"_name_or_path": "renamed"}) == shared_execution(config)


@pytest.mark.parametrize("architecture", ["NemotronHForCausalLM", "nemotron_h"])
def test_nemotron_is_excluded(architecture):
    with pytest.raises(ValueError, match="Nemotron is excluded"):
        shared_execution(dict(architectures=[architecture]))


@pytest.mark.parametrize("architecture", ["DeepseekV4ForCausalLM",
                                         "Qwen4ExpForCausalLM", "MissingForCausalLM"])
def test_incomplete_or_unknown_training_definitions_do_not_imply_support(architecture):
    with pytest.raises(ValueError, match="not a supported training"):
        shared_execution(dict(architectures=[architecture]))


def test_glm_uses_the_training_model_for_shared_rollouts():
    from examples.sft.glm.create_dummy import dummy_config

    assert shared_execution(dummy_config(), ["all"]) == "training"


def test_custom_or_missing_qwen_tool_template_uses_shared_training():
    config = configurations()["qwen3"]
    assert shared_execution(config, tokenizer=SimpleNamespace(chat_template=None)) == "training"
    assert shared_execution(config, tokenizer=SimpleNamespace(chat_template="<tool_call>")) == "serve"
    assert shared_execution(config, tokenizer=SimpleNamespace(chat_template="<arg_key>")) == "training"


def test_quantized_and_bidirectional_models_are_rejected():
    config = configurations()["llama"]
    for change, message in ((dict(quantization_config={"quant_method": "mxfp4"}), "unquantized"),
                             (dict(use_bidirectional_attention=True), "causal")):
        with pytest.raises(ValueError, match=message):
            shared_execution(config | change)
    assert shared_execution(configurations()["qwen3"], ["lm_head"]) == "training"
    with pytest.raises(ValueError, match="causal"):
        shared_execution(dict(architectures=["Gemma3TextModel"], model_type="gemma3_text"))


@pytest.fixture
def service():
    class Tokenizer:
        eos_token_id = 1

        def decode(self, ids, **kwargs):
            return "".join("ABCDE"[i] for i in ids)

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["tokenize"] and kwargs["return_dict"] is False
            return [0, 2]

    weight = torch.zeros(8, dtype=torch.bfloat16)
    trainer = SimpleNamespace(batch_size=1, seq_length=16,
        get_shared_base_weights=lambda: dict(embedding=weight, lm_head=weight),
        next_token_logits=lambda ids, positions: np.array([[0., -2., 2., 1., -1.]], dtype=np.float32))
    server = SharedModelServer(trainer, Tokenizer(), dict(vocab_size=5),
        dict(host="127.0.0.1", port=0, model="base", max_context=16, max_concurrency=4, eos_token_id=[1, 4]))
    url = f"http://127.0.0.1:{server.http.server_port}"
    yield server, trainer, url
    server.close()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("fail_after", [0, 1])
@pytest.mark.parametrize("error,status,error_type", [
    (DecodeCapacityError("decode workspace is full"), 429, "capacity_error"),
    (RuntimeError("execution failed"), 500, "server_error"),
])
def test_generation_errors_are_reported_and_release_admission(service, stream, fail_after, error, status, error_type):
    server, trainer, url = service
    server.publish("policy", [], 0)
    original = trainer.next_token_logits
    calls = 0

    def fail(*args):
        nonlocal calls
        calls += 1
        if calls > fail_after:
            raise error
        return original(*args)

    trainer.next_token_logits = fail
    body = dict(model="policy", tokens=[0, 2], temperature=0, max_tokens=3, stream=stream)
    endpoint = url + "/v1/chat/completions/tokens"
    response = requests.post(endpoint, json=body, timeout=10)
    if stream:
        assert response.status_code == 200
        events = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
        assert events[-1] == "[DONE]"
        payload = json.loads(events[-2])
        assert all(choice.get("finish_reason") is None
                   for event in events[:-2] for choice in json.loads(event).get("choices", []))
    else:
        assert response.status_code == status
        payload = response.json()
    assert payload == {"error": {"message": str(error), "type": error_type, "code": status}}
    assert server.active == 0
    trainer.next_token_logits = original
    assert requests.post(endpoint, json=body | dict(stream=False), timeout=10).status_code == 200


def test_http_tokens_sampling_and_publication(service):
    server, trainer, url = service
    body = dict(model="policy", tokens=[0, 2], temperature=0, max_tokens=3, logprobs=True, top_logprobs=2)
    endpoint = url + "/v1/chat/completions/tokens"
    assert requests.post(endpoint, json=body).status_code == 429
    server.publish("policy", [], 0)
    response = requests.post(endpoint, json=body).json()
    assert response["prompt_token_ids"] == [0, 2]
    choice = response["choices"][0]
    assert choice["token_ids"] == [2, 2, 2]
    assert choice["message"]["content"] == "CCC"
    expected = 2. - np.logaddexp.reduce([0., -2., 2., 1., -1.])
    assert all(s["logprob"] == pytest.approx(expected) for s in choice["logprobs"]["content"])
    assert len(choice["logprobs"]["content"][0]["top_logprobs"]) == 2
    assert response["usage"]["total_tokens"] == 5
    assert server.summary()["shared_base_bytes"] == 16
    tokenized = requests.post(url + "/tokenize", json=dict(messages=[dict(role="user", content="hello")])).json()
    assert tokenized["tokens"] == [0, 2]
    stopped = requests.post(endpoint, json=body | dict(stop="CC")).json()
    assert stopped["choices"][0]["finish_reason"] == "stop"
    assert stopped["choices"][0]["message"]["content"] == ""
    server.begin_training()
    assert requests.post(endpoint, json=body).status_code == 429
    trainer.next_token_logits = lambda *args: np.array([[0., -2., 1., 3., -1.]], dtype=np.float32)
    server.publish("policy", [], 1)
    updated = requests.post(endpoint, json=body).json()
    assert updated["choices"][0]["token_ids"] == [3, 3, 3]
    with pytest.raises(ValueError, match="publication"):
        server.publish("policy", [], 1)
    server.begin_training()
    with pytest.raises(ValueError, match="live adapter"):
        server.publish("policy", [object()], 2)


@pytest.mark.parametrize("fail", [False, True])
def test_persistent_decode_prefills_once_and_resets_between_requests(service, fail):
    server, trainer, url = service
    server.persistent_decode = True
    calls, resets = [], []

    def decode(ids, reset=False):
        calls.append((ids.tolist(), reset))
        if fail and not reset:
            raise RuntimeError("decode failed")
        return np.array([0., -2., 2., 1., -1.], dtype=np.float32)

    trainer.decode_logits = decode
    trainer.reset_decode_state = lambda: resets.append(True)
    trainer.next_token_logits = lambda *args: pytest.fail("GLM generation must use the persistent cache")
    server.publish("policy", [], 0)
    body = dict(model="policy", tokens=[0, 2], temperature=0, max_tokens=3)
    for _ in range(2):
        response = requests.post(url + "/v1/chat/completions/tokens", json=body, timeout=10)
        assert response.status_code == (500 if fail else 200), response.text
    per_request = [([0, 2], True), ([2], False)] + ([] if fail else [([2], False)])
    assert calls == per_request * 2
    assert len(resets) == 2
    server.begin_training()
    assert len(resets) == 3
    assert server.summary()["persistent_decode"]


def test_pause_drains_streams_and_rejects_new_requests(service):
    server, trainer, url = service
    entered, release = threading.Event(), threading.Event()
    old = trainer.next_token_logits

    def blocked(*args):
        entered.set()
        assert release.wait(10)
        return old(*args)

    trainer.next_token_logits = blocked
    server.publish("policy", [], 0)
    body = dict(model="policy", tokens=[0, 2], temperature=0, max_tokens=3, stream=True)
    endpoint = url + "/v1/chat/completions"
    with concurrent.futures.ThreadPoolExecutor() as pool:
        response = pool.submit(requests.post, endpoint, json=body, timeout=15)
        assert entered.wait(5)
        pause = pool.submit(server.begin_training)
        with server.condition:
            assert server.condition.wait_for(lambda: server.sleeping, timeout=5)
        assert not pause.done()
        assert requests.post(endpoint, json=body).status_code == 429
        release.set()
        stream = response.result().text
        pause.result(timeout=10)
    assert "data: [DONE]" in stream
    chunks = [json.loads(line[6:]) for line in stream.splitlines() if line.startswith("data: {")]
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "CCC"


@pytest.mark.parametrize("change", [dict(tokens=[-1]), dict(tokens=[99]), dict(tokens=[0] * 16),
    dict(temperature=-1), dict(top_p=0), dict(min_p=2), dict(max_tokens=0), dict(min_tokens=9),
    dict(stop_token_ids=[5]), dict(logit_bias={"-1": 2}), dict(seed=-1), dict(tools=[{}]),
    dict(messages=[dict(role="user", content=[dict(type="image_url")])], tokens=None)])
def test_invalid_requests_do_not_strand_the_phase(service, change):
    server, _, url = service
    server.publish("policy", [], 0)
    response = requests.post(url + "/v1/chat/completions", json=dict(model="policy", tokens=[0], max_tokens=4) | change)
    assert response.status_code == 400, response.text
    server.begin_training()
    assert server.active == 0


def test_generation_config_eos_tokens_stop_completion(service):
    server, trainer, url = service
    server.publish("policy", [], 0)
    result = requests.post(url + "/v1/chat/completions/tokens", json=dict(
        model="policy", tokens=[0, 2], max_tokens=3, temperature=0, logit_bias={"4": 100})).json()
    assert result["choices"][0]["token_ids"] == [4]
    assert result["choices"][0]["finish_reason"] == "stop"


def test_ignoring_model_eos_preserves_explicit_stop_tokens(service):
    server, trainer, url = service
    server.publish("policy", [], 0)
    result = requests.post(url + "/v1/chat/completions/tokens", json=dict(
        model="policy", tokens=[0], max_tokens=3, temperature=0, ignore_eos=True,
        stop_token_ids=[2])).json()
    assert result["choices"][0]["token_ids"] == [2]
    assert result["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize("stream", [False, True])
def test_successful_turns_are_cached_and_failed_generations_are_not(stream):
    from tests.grpo.test_decode_scheduling import PrefixTrainer

    class CachedTrainer(PrefixTrainer):
        def get_shared_base_weights(self):
            return dict(weight=torch.zeros(8, dtype=torch.bfloat16))

        def decode_batch_logits(self, ids, tokens, offsets, resets):
            super().decode_batch_logits(ids, tokens, offsets, resets)
            return np.tile(np.array([0., -2., 2., 1., -1.], dtype=np.float32), (len(ids), 1))

        def reset_decode_state(self):
            self.states.clear()
            self.prefixes.clear()

    trainer = CachedTrainer()
    tokenizer = SimpleNamespace(eos_token_id=1, decode=lambda ids, **kwargs: "x" * len(ids))
    server = SharedModelServer(trainer, tokenizer, dict(vocab_size=5),
                               dict(host="127.0.0.1", port=0, model="base", max_context=32, max_concurrency=2))
    url = f"http://127.0.0.1:{server.http.server_port}/v1/chat/completions/tokens"
    body = dict(model="policy", tokens=[0, 2], temperature=0, max_tokens=3, stream=stream)
    try:
        server.publish("policy", [], 0)
        response = requests.post(url, json=body, timeout=10)
        assert response.ok and "error" not in response.text
        assert server.summary()["completed_turn_cache_saves"] == 1
        # Raw generated IDs survive even when decoding does not preserve text.
        continuation = body | dict(tokens=[0, 2, 2, 2, 2, 3])
        assert requests.post(url, json=continuation, timeout=10).ok
        assert server.summary()["completed_turn_cache_hits"] == 1
        saves = server.summary()["completed_turn_cache_saves"]
        trainer.fail = True
        response = requests.post(url, json=body, timeout=10)
        assert "error" in response.text
        assert server.summary()["completed_turn_cache_saves"] == saves
        assert not trainer.states and not server.scheduler.histories
        server.begin_training()
        assert not trainer.prefixes and not server.scheduler.completed_prefixes
    finally:
        server.close()
