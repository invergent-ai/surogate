"""Under --batch-invariant a request's numbers do not depend on what shares its rounds. GPU.

Skipped unless SUROGATE_BATCH_INVARIANT_ARTIFACT names a prepared BF16 text model (.sinfer; the
regression was found on Qwen3.5-0.8B converted from its safetensors, a GDN hybrid); select one
free GPU with CUDA_VISIBLE_DEVICES, and use SUROGATE_SERVE_BIN to pick the engine binary.

Without the switch, a prompt scored alone and the same prompt scored among fifteen others came
out different by up to 1 nat on a single token and several nats over a sentence, and greedy
chat answers changed: cuBLASLt picks its BF16 algorithm (split-K among it) by the width of the
round, the fused GDN gating projection splits k by that width too, and a long prompt that shares
a round was cut wherever its company left the prefill window, which moves a recurrent layer's
chunk boundaries. With the switch every one of those is fixed by the request alone.

One engine, with a small prefill window so that prompts of a few hundred tokens already cross
cuts: every request is first served alone, twice, then everything again with sixteen in flight in
two different orders, while short and long prompts are scored and chat answers of different
lengths are generated -- so rounds pack prompts, carry decode rows beside prompt chunks, and
change width from one step to the next. Prompt log-probabilities, sampled-token log-probabilities
and answers must be identical, bit for bit, and prompts must actually have been packed.
"""

import concurrent.futures as cf
import contextlib
import os
import random
import socket
import subprocess
import time

import pytest
import requests

from surogate.cli.serve import _resolve_binary

pytestmark = pytest.mark.skipif(not os.getenv("SUROGATE_BATCH_INVARIANT_ARTIFACT"),
                                reason="needs a prepared BF16 text model and a free GPU")

MODEL = "invariant"
SENTENCES = [
    "The customer says the invoice was paid, but the service was suspended yesterday.",
    "Clientul spune că factura a fost plătită, dar serviciul a fost suspendat ieri.",
    "The order arrived a day late and the packaging was damaged in two places.",
    "Comanda a sosit cu o zi întârziere, iar ambalajul era deteriorat.",
    "The agent confirmed the refund, but the amount has not appeared in the account yet.",
    "Rata dobânzii a crescut cu 0,25 puncte procentuale în ultimul trimestru.",
    "The user cannot log in after resetting the password on the mobile application.",
    "Trenul de la Cluj la București pleacă la ora 7:45 și ajunge la 17:10.",
    "A recurrent layer carries its state from one chunk of the prompt to the next one.",
    "Profesorul a explicat de ce suma unghiurilor unui triunghi este de 180 de grade.",
]
QUESTIONS = [
    "What is 17 + 25? Answer with the number only.",
    "Cât face 144 / 12? Răspunde doar cu numărul.",
    "Write two sentences about the Danube.",
    "Numără de la 1 la 30, separat prin virgule.",
    "Explain in three sentences what a prime number is.",
    "Scrie un paragraf despre Carpați.",
]


@contextlib.contextmanager
def server(tmp_path, *flags):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    log = tmp_path / f"server-{port}.log"
    cmd = [_resolve_binary("server"), os.environ["SUROGATE_BATCH_INVARIANT_ARTIFACT"],
           "--host", "127.0.0.1", "--port", str(port), "--served-model-name", MODEL,
           "--max-model-len", "4096", "--kv-cache-dtype", "bf16", "--greedy", *flags]
    base = f"http://127.0.0.1:{port}"
    with log.open("w") as output:
        process = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            assert process.poll() is None, log.read_text()[-4000:]
            try:
                if requests.get(base + "/v1/models", timeout=1).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5)
        else:
            pytest.fail(log.read_text()[-4000:])
        yield base
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def counter(base, name):
    for line in requests.get(base + "/metrics", timeout=10).text.splitlines():
        if line.startswith(f"surogate_{name}"):
            return float(line.rsplit(" ", 1)[1])
    raise AssertionError(name)


def document(i, sentences):
    return " ".join(SENTENCES[(i + k) % len(SENTENCES)] for k in range(sentences))


def workload():
    jobs = [("score", SENTENCES[i], 0) for i in range(len(SENTENCES))]
    # 20..60 sentences is roughly 350..1100 tokens: several cuts at a 512-token window.
    jobs += [("score", document(i, 20 + 8 * i), 0) for i in range(6)]
    jobs += [("chat", q, 12 + 29 * i) for i, q in enumerate(QUESTIONS)]
    jobs += [("chat", document(i, 24 + 10 * i) + "\n\nSummarize the text above in one sentence.", 40)
             for i in range(4)]
    return jobs


def run(base, job):
    kind, text, max_tokens = job
    if kind == "score":
        response = requests.post(base + "/v1/completions", timeout=300, json={
            "model": MODEL, "prompt": text, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0})
        assert response.status_code == 200, response.text
        body = response.json()
        entries = body.get("prompt_logprobs") or body["choices"][0]["prompt_logprobs"]
        return [None if entry is None else sorted((token, value["logprob"]) for token, value in entry.items())
                for entry in entries]
    response = requests.post(base + "/v1/chat/completions", timeout=300, json={
        "model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
        "temperature": 0, "logprobs": True})
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    return choice["message"]["content"], [item["logprob"] for item in choice["logprobs"]["content"]]


def test_results_do_not_depend_on_what_shares_the_rounds(tmp_path):
    with server(tmp_path, "--batch-invariant", "--max-num-seqs", "16",
                "--max-num-batched-tokens", "512") as base:
        jobs = workload()
        alone = [run(base, job) for job in jobs]
        assert [run(base, job) for job in jobs] == alone
        packed_before = counter(base, "packed_prefill_rounds_total")
        for seed in (1, 2):
            order = list(range(len(jobs)))
            random.Random(seed).shuffle(order)
            with cf.ThreadPoolExecutor(16) as pool:
                together = dict(zip(order, pool.map(lambda i: run(base, jobs[i]), order)))
            for i, job in enumerate(jobs):
                assert together[i] == alone[i], f"{job[0]} request {i} changed when batched"
        assert counter(base, "packed_prefill_rounds_total") > packed_before


def test_batch_invariant_refuses_speculative_decoding(tmp_path):
    result = subprocess.run([_resolve_binary("server"), os.environ["SUROGATE_BATCH_INVARIANT_ARTIFACT"],
                             "--batch-invariant", "--spec", "mtp", "--draft-tokens", "3"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "--batch-invariant cannot be combined with --spec" in result.stderr + result.stdout
