"""Real-model regression: set SUROGATE_QWEN3_VL_TEST_URL to a server with --vision,
--max-num-seqs 4, and --max-num-batched-tokens 128. Uses generated in-memory images.
"""

import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor

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


@pytest.mark.skipif(not os.getenv("SUROGATE_QWEN3_VL_TEST_URL"), reason="requires a Qwen3-VL GPU server")
@pytest.mark.parametrize("responses", [False, True])
def test_video_color_order_through_chat_and_responses(tmp_path, responses):
    import shutil
    import subprocess
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required to prepare the test video")
    frames = b"".join(Image.new("RGB", (64, 64), color).tobytes()
                      for color in ("red",) * 4 + ("blue",) * 4)
    video = tmp_path / "colors.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                    "-video_size", "64x64", "-framerate", "4", "-i", "pipe:0",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
                   input=frames, check=True, timeout=30)
    uri = "data:video/mp4;base64," + base64.b64encode(video.read_bytes()).decode()
    base = os.environ["SUROGATE_QWEN3_VL_TEST_URL"].rstrip("/")
    question = ("What is the color at the beginning of the video? What is the color at the end? "
                "Reply with the two colors.")
    model = os.getenv("SUROGATE_QWEN3_VL_TEST_MODEL", "qwen3_vl")
    if responses:
        endpoint = "/v1/responses"
        body = {"model": model, "temperature": 0, "max_output_tokens": 24,
                "input": [{"role": "user", "content": [
                    {"type": "input_video", "video_url": uri},
                    {"type": "input_text", "text": question}]}]}
    else:
        endpoint = "/v1/chat/completions"
        body = {"model": model, "temperature": 0, "max_tokens": 24,
                "messages": [{"role": "user", "content": [
                    {"type": "video_url", "video_url": {"url": uri}},
                    {"type": "text", "text": question}]}]}
    response = requests.post(base + endpoint, json=body, timeout=120)
    assert response.ok, response.text
    data = response.json()
    answer = (" ".join(part.get("text", "") for item in data["output"] for part in item.get("content", []))
              if responses else data["choices"][0]["message"]["content"]).lower()
    assert "red" in answer and "blue" in answer, answer
    assert answer.index("red") < answer.index("blue"), answer


@pytest.mark.skipif(not os.getenv("SUROGATE_QWEN3_VL_TEST_URL") or not os.getenv("SUROGATE_QWEN3_VL_TEST_SLEEP"),
                    reason="requires a vision server started with --enable-sleep-mode")
def test_media_scores_survive_sleep_and_wake():
    base = os.environ["SUROGATE_QWEN3_VL_TEST_URL"].rstrip("/")
    data = io.BytesIO()
    Image.new("RGB", (128, 128), "red").save(data, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()
    body = {"model": os.getenv("SUROGATE_QWEN3_VL_TEST_MODEL", "qwen3_vl"), "temperature": 0,
            "max_tokens": 8, "return_token_ids": True, "logprobs": True,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": uri}},
                {"type": "text", "text": "Name the background color. Reply with one color only."}]}]}
    first = requests.post(base + "/v1/chat/completions", json=body, timeout=120)
    assert first.ok, first.text
    assert requests.post(base + "/sleep", timeout=60).ok
    assert requests.post(base + "/wake_up", timeout=60).ok
    replay = requests.post(base + "/v1/chat/completions", json=body, timeout=120)
    assert replay.ok, replay.text
    a, b = [r.json()["choices"][0] for r in (first, replay)]
    assert a["token_ids"] == b["token_ids"]
    assert [t["logprob"] for t in a["logprobs"]["content"]] == pytest.approx(
        [t["logprob"] for t in b["logprobs"]["content"]], abs=.005)
