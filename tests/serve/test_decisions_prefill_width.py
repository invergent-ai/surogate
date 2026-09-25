"""A decision's answer does not depend on how the engine's prefill window cuts its prompts.

GPU. Skipped unless SUROGATE_PREFILL_WIDTH_ARTIFACT names a prepared attention-only
mixture-of-experts chat model with W8 routed experts (.sinfer; the regression was found on Gemma 4
26B-A4B, served from bf16 weights); select one free GPU with CUDA_VISIBLE_DEVICES. Hybrid models
whose linear-attention mixers carry state across prefill chunks are not claimed here.

A prefill round's width is set by what the packing window leaves: a decisions request's shared
prefix, and each question's suffix, are cut wherever their round-mates leave off, and a remainder
can run alone in a round of a few tokens. The sparse-MoE op used to pick its kernels by that width
(decode for one token, small-T below the W8 wide route's 20-token floor), so those tokens got
different arithmetic and the answers moved by a few ulps -- every question of a request when the
remainder was part of its shared prefix. Prefill rounds now pin the wide route at every width
(ops::SparseMoeRouting::WidthInvariant).

The same requests are served by two engines that differ only in --max-num-batched-tokens (the
prefill window): 128, which cuts every prompt into many rounds, and 2048, which cuts almost none.
Every answer must be identical, bit for bit. Both engines run eager (a captured prefill graph pads
a lone chunk to a multiple of 128, which would hide a short remainder) and without the prefix
cache (so a prefix is prefilled from its first token), and the states grow three words at a time
so that some shared prefixes end one to nineteen tokens past a 128-token boundary: a remainder
that took the decode or small-T kernels before. On an engine without the fix this test fails.
"""

import json
import os
import re
import socket
import subprocess
import time

import pytest
import requests

from surogate.cli.serve import _resolve_binary

pytestmark = pytest.mark.skipif(not os.getenv("SUROGATE_PREFILL_WIDTH_ARTIFACT"),
                                reason="needs a prepared mixture-of-experts chat model and a free GPU")

WORDS = ("amber", "basalt", "cobalt", "delta", "ember", "fjord", "garnet", "harbor", "indigo",
         "juniper", "kelp", "lagoon", "meadow", "nectar", "onyx", "pumice", "quartz", "russet")


def serve(port, window, log):
    cmd = [_resolve_binary("server"), os.environ["SUROGATE_PREFILL_WIDTH_ARTIFACT"],
           "--host", "127.0.0.1", "--port", str(port), "--served-model-name", "rune",
           "--max-model-len", "8192", "--max-num-seqs", "8",
           "--max-num-batched-tokens", str(window), "--enforce-eager", "--no-enable-prefix-caching"]
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def wait_ready(url, process, log_path):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        assert process.poll() is None, log_path.read_text()
        try:
            if requests.get(url + "/v1/models", timeout=1).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.2)
    pytest.fail(log_path.read_text())


def body(words):
    state = {"ticket": " ".join(WORDS[i % len(WORDS)] for i in range(words)),
             "channel": "email"}
    # Three questions whose rendered suffixes are well past the endpoint's 47-token floor, so the
    # request shares its prefix; their order is the order of the answers.
    questions = {
        "urgent": {"type": "noul",
                   "instructions": "Does the ticket describe a problem that needs attention within "
                                   "the next hour, judging only by the words it contains?",
                   "criteria": {"true": "It needs attention within the hour",
                                "false": "It can wait longer than an hour"}},
        "topic": {"type": "choice",
                  "instructions": "Which kind of material does the ticket mention most often, "
                                  "counting every word that names a stone, a plant or a place?",
                  "criteria": {"stone": "Mostly stones and minerals", "plant": "Mostly plants",
                               "place": "Mostly places and landforms"}},
        "tone": {"type": "choice",
                 "instructions": "What is the overall tone of the ticket, read as a message sent "
                                 "by a customer to a support desk?",
                 "criteria": {"calm": "Calm", "annoyed": "Annoyed", "neutral": "Neutral"}},
    }
    return {"model": "rune", "state": state, "questions": questions}


def answers(url, request):
    response = requests.post(url + "/api/alpha/decisions", json=request, timeout=300)
    assert response.ok, response.text
    return response.json()["answers"]


def shared_prefixes(log_path):
    return [int(m) for m in re.findall(r"questions=3 shared_prefix=(\d+)", log_path.read_text())]


def test_answers_do_not_depend_on_the_prefill_window(tmp_path):
    requests_ = [body(words) for words in range(40, 40 + 3 * 48, 3)]
    served = {}
    prefixes = {}
    for window in (2048, 128):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        log_path = tmp_path / f"server-{window}.log"
        with log_path.open("w") as log:
            process = serve(port, window, log)
            try:
                url = f"http://127.0.0.1:{port}"
                wait_ready(url, process, log_path)
                served[window] = [answers(url, request) for request in requests_]
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        prefixes[window] = shared_prefixes(log_path)

    # The requests must actually exercise the short remainders: with no prefix cache a request's
    # shared prefix is prefilled from its first token, so under the 128-token window some prefix
    # must end one to nineteen tokens past a window boundary -- a remainder of its own.
    assert prefixes[128] and prefixes[128] == prefixes[2048], (prefixes[128], prefixes[2048])
    tails = sorted({length % 128 for length in prefixes[128]})
    assert any(1 <= tail <= 19 for tail in tails), tails
    for index, (wide, narrow) in enumerate(zip(served[2048], served[128])):
        assert narrow == wide, (index, json.dumps(wide), json.dumps(narrow))
