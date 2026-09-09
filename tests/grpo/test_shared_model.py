"""Admission, publication, sampling and model coverage without a GPU."""

import concurrent.futures
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import requests
import torch

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
