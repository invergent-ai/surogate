"""Real-model regression: set SUROGATE_QWEN3_VL_TEST_URL to a server with --vision,
--max-num-seqs 4, and --max-num-batched-tokens 128. Uses generated in-memory images.
"""

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import os

import pytest
import requests
from PIL import Image


@pytest.mark.skipif(not os.getenv("SUROGATE_QWEN3_VL_TEST_URL"), reason="requires a Qwen3-VL GPU server")
def test_concurrent_media_preserves_each_prompts_visual_features():
    base = os.environ["SUROGATE_QWEN3_VL_TEST_URL"].rstrip("/")

    def ask(color):
        data = io.BytesIO()
        # 256 visual tokens force multiple prefill chunks; the encoded image must survive them.
        Image.new("RGB", (512, 512), color).save(data, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()
        response = requests.post(base + "/v1/chat/completions", json={
            "model": "qwen3_vl", "temperature": 0, "max_tokens": 10,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": uri}},
                {"type": "text", "text": "What is the background color? Reply with one color only."},
            ]}],
        }, timeout=120)
        assert response.status_code == 200, response.text
        content = response.json()["choices"][0]["message"]["content"].strip().lower().strip(".* \n")
        assert content == color, (color, content)

    for color in ("red", "blue"):
        ask(color)
    for _ in range(2):
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(ask, ("red", "blue", "red", "blue")))


@pytest.mark.skipif(
    not os.getenv("SUROGATE_QWEN3_VL_TEST_URL") or not os.getenv("SUROGATE_VISION_LORA_TEST_ADAPTERS"),
    reason="requires a vision server with two distinct vision adapters",
)
def test_vision_adapters_preserve_policies_across_concurrent_prefill():
    """SUROGATE_VISION_LORA_TEST_ADAPTERS names two loaded adapters, separated by a comma."""
    base = os.environ["SUROGATE_QWEN3_VL_TEST_URL"].rstrip("/")
    adapters = os.environ["SUROGATE_VISION_LORA_TEST_ADAPTERS"].split(",")
    assert len(adapters) == 2
    model = os.getenv("SUROGATE_QWEN3_VL_TEST_MODEL", "qwen3_vl")
    data = io.BytesIO()
    Image.new("RGB", (64, 64), (32, 90, 180)).save(data, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()

    def ask(name):
        response = requests.post(base + "/v1/chat/completions", json={
            "model": name, "temperature": 0, "max_tokens": 16, "ignore_eos": True,
            "return_token_ids": True, "logprobs": True,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": uri}},
                {"type": "text", "text": "Describe."},
            ]}],
        }, timeout=120)
        assert response.status_code == 200, response.text
        choice = response.json()["choices"][0]
        return choice["token_ids"], [token["logprob"] for token in choice["logprobs"]["content"]]

    def same_policy(actual, expected):
        return actual[0] == expected[0] and actual[1] == pytest.approx(expected[1], abs=.005)

    expected = {name: ask(name) for name in (model, *adapters)}
    # Probabilities can change even when greedy decoding chooses the same tokens.
    assert not same_policy(expected[adapters[0]], expected[model])
    assert not same_policy(expected[adapters[0]], expected[adapters[1]])
    with ThreadPoolExecutor(max_workers=3) as pool:
        names = [*adapters, model] * 2
        outcomes = list(pool.map(ask, names))
    for name, actual in zip(names, outcomes):
        assert same_policy(actual, expected[name]), name
