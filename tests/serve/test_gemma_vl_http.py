"""Real Gemma vision server regressions (Gemma 3 4B or Gemma 4 E2B).

Set SUROGATE_GEMMA_VL_TEST_URL and optionally SUROGATE_GEMMA_VL_TEST_MODEL.
Start with --vision --max-num-seqs 4 --max-num-batched-tokens 128.
SUROGATE_GEMMA_VL_TEST_ADAPTER enables the vision adapter regression;
SUROGATE_GEMMA_VL_TEST_SLEEP enables sleep/wake verification.
"""

import base64
import io
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
from PIL import Image

pytestmark = pytest.mark.skipif(
    not os.getenv("SUROGATE_GEMMA_VL_TEST_URL"), reason="requires a Gemma vision GPU server"
)


def ask(color, model=None, question="Name the background color. Reply with one color only."):
    encoded = io.BytesIO()
    Image.new("RGB", (128, 128), color).save(encoded, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode()
    response = requests.post(
        os.environ["SUROGATE_GEMMA_VL_TEST_URL"].rstrip("/") + "/v1/chat/completions",
        json={
            "model": model or os.getenv("SUROGATE_GEMMA_VL_TEST_MODEL", "vision"),
            "temperature": 0,
            "max_tokens": 16,
            "return_token_ids": True,
            "logprobs": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": uri}},
                        {"type": "text", "text": question},
                    ],
                }
            ],
        },
        timeout=180,
    )
    assert response.ok, response.text
    return response.json()["choices"][0]


def same_policy(actual, expected):
    assert actual["token_ids"] == expected["token_ids"]
    assert [t["logprob"] for t in actual["logprobs"]["content"]] == pytest.approx(
        [t["logprob"] for t in expected["logprobs"]["content"]], abs=0.005
    )


def test_concurrent_images_and_repeated_turns():
    expected = {color: ask(color) for color in ("red", "blue")}
    for color, answer in expected.items():
        assert color in answer["message"]["content"].lower(), answer
    colors = ["red", "blue"] * 2
    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = list(pool.map(ask, colors))
    for color, answer in zip(colors, answers):
        same_policy(answer, expected[color])


@pytest.mark.skipif(not os.getenv("SUROGATE_GEMMA_VL_TEST_ADAPTER"), reason="requires a vision adapter")
def test_vision_adapter_with_image_larger_than_requested_prefill():
    # Gemma image blocks exceed the requested 128-token prefill chunk. Adapter
    # scratch must cover the whole block, without silently serving the base policy.
    color = (32, 90, 180)
    question = "Describe this picture in five words."
    base = ask(color, question=question)
    adapted = ask(color, os.environ["SUROGATE_GEMMA_VL_TEST_ADAPTER"], question=question)
    assert adapted["logprobs"] != base["logprobs"], "vision adapter was not applied"
    same_policy(ask(color, question=question), base)


@pytest.mark.skipif(not os.getenv("SUROGATE_GEMMA_VL_TEST_SLEEP"), reason="requires --enable-sleep-mode")
def test_image_scores_survive_sleep_and_wake():
    before = ask("red")
    url = os.environ["SUROGATE_GEMMA_VL_TEST_URL"].rstrip("/")
    for endpoint in ("/sleep", "/wake_up"):
        response = requests.post(url + endpoint, timeout=90)
        assert response.ok, response.text
    same_policy(ask("red"), before)


@pytest.mark.parametrize("responses", [False, True])
def test_video_color_order(tmp_path, responses):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required to prepare the test video")
    frames = b"".join(Image.new("RGB", (64, 64), color).tobytes() for color in ("red",) * 4 + ("blue",) * 4)
    video = tmp_path / "colors.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "64x64",
            "-framerate",
            "4",
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        input=frames,
        check=True,
        timeout=30,
    )
    uri = "data:video/mp4;base64," + base64.b64encode(video.read_bytes()).decode()
    question = (
        "What is the color at the beginning of the video? What is the color at the end? Reply with the two colors."
    )
    body = {"model": os.getenv("SUROGATE_GEMMA_VL_TEST_MODEL", "vision"), "temperature": 0}
    if responses:
        endpoint = "/v1/responses"
        body.update(
            max_output_tokens=32,
            reasoning={"effort": "none"},
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_video", "video_url": uri},
                        {"type": "input_text", "text": question},
                    ],
                }
            ],
        )
    else:
        endpoint = "/v1/chat/completions"
        body.update(
            max_tokens=32,
            enable_thinking=False,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": uri}},
                        {"type": "text", "text": question},
                    ],
                }
            ],
        )
    response = requests.post(
        os.environ["SUROGATE_GEMMA_VL_TEST_URL"].rstrip("/") + endpoint,
        json=body,
        timeout=240,
    )
    assert response.ok, response.text
    data = response.json()
    answer = (
        " ".join(part.get("text", "") for item in data["output"] for part in item.get("content", []))
        if responses
        else data["choices"][0]["message"]["content"]
    ).lower()
    assert "red" in answer and "blue" in answer, answer
    assert answer.index("red") < answer.index("blue"), answer
