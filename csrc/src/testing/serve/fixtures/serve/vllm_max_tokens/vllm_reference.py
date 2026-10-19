#!/usr/bin/env python3
"""Re-derive parity.jsonl: what vLLM 0.31.0 answers over HTTP when prompt + max_tokens meets max_model_len.

The answers come from vLLM itself: its GPU-less render server (`vllm launch render`), which runs the same request
validation as `vllm serve` -- OnlineRenderer.render_chat / render_completion (TokenizeParams: max_tokens against
max_model_len, the prompt text's character bound, the truncated token count), get_max_tokens and
to_sampling_params -- behind the same FastAPI exception handlers, and answers with the GenerateRequest it would hand
the engine (so the max_tokens it would generate with) or with the error response. Start four of them on the
Surogate3.7-35B-A3B tokenizer and chat template (no GPU; the weights are not read):

    M=.../Surogate3.7-35B-A3B
    CUDA_VISIBLE_DEVICES= vllm launch render $M --served-model-name m --max-model-len 4096   --port 18431 &
    CUDA_VISIBLE_DEVICES= vllm launch render $M1000 --served-model-name m --max-model-len 4096   --port 18432 &
    CUDA_VISIBLE_DEVICES= vllm launch render $M --served-model-name m --max-model-len 262144 --port 18433 &
    CUDA_VISIBLE_DEVICES= vllm launch render $M8192 --served-model-name m --max-model-len 262144 --port 18434 &

where $M1000 / $M8192 are $M with "max_new_tokens": 1000 / 8192 added to generation_config.json (the checkpoint
itself has none), and then run, with the same Python:

    python vllm_reference.py $M > parity.jsonl

Prompts are "a a a ..." (one token per "a"): a chat user message, a completions text prompt, or completions token
ids. Each row carries the prompt's token count and the length in code points of the text vLLM tokenized (the
rendered chat template; computed here with the checkpoint's tokenizer and checked against the ids vLLM returns and
the character counts its refusals quote), and vLLM's answer: {"status": 200, "max_tokens": N} or
{"status": 400, "http_body": "<the exact response bytes>"}. A streaming request is validated by the same steps before
vLLM opens the stream, so stream=true rows answer as stream=false ones do; they are kept to show it.
"""

import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

SERVERS = [  # (max_model_len, generation-config max_new_tokens, port)
    (4096, None, 18431),
    (4096, 1000, 18432),
    (262144, None, 18433),
    (262144, 8192, 18434),
]

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
L = max(len(token) for token in tokenizer.get_vocab())  # vllm/tokenizers/hf.py max_chars_per_token
A = tokenizer.encode(" a", add_special_tokens=False)
assert len(A) == 1, A


def content(k):
    return "a" + " a" * (k - 1)


def chat_text(k):
    return tokenizer.apply_chat_template([{"role": "user", "content": content(k)}], tokenize=False,
                                         add_generation_prompt=True)


def measure(kind, k):
    """(prompt tokens, characters vLLM's text check sees) for prompt kind `kind` of size k."""
    if kind == "chat":
        text = chat_text(k)
        return len(tokenizer.encode(text, add_special_tokens=False)), len(text)
    if kind == "text":
        text = content(k)
        return len(tokenizer.encode(text)), len(text)
    return k, 0  # ids


OVERHEAD = measure("chat", 1)[0] - 1
assert measure("chat", 50)[0] == OVERHEAD + 50 and measure("text", 50)[0] == 50


def k_for(kind, P):
    return P - OVERHEAD if kind == "chat" else P


def post(port, path, body):
    data = json.dumps(body).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def specs(C, P, chars, kind):
    """The request fields to try for one prompt."""
    edge = None
    if chars:
        # The smallest max_tokens whose character bound the text exceeds: chars > (C - M) * L.
        edge = C - (chars - 1) // L
    values = {0, -1, 1, 16, 1000, 32768, C - P - 1, C - P, C - P + 1, C - 1, C, C + 1}
    if edge is not None:
        values |= {edge - 1, edge, edge + 1}
    out = [{}]
    out += [{"max_tokens": m} for m in sorted(values)]
    if kind == "chat":
        out += [{"max_completion_tokens": m} for m in sorted({0, C - P, C - P + 1, C, C + 1})]
        out += [{"max_completion_tokens": C - P, "max_tokens": C + 5}, {"max_completion_tokens": None, "max_tokens": 7}]
    else:
        out += [{"max_tokens": None}]
    out += [{"min_tokens": 10}, {"min_tokens": C - P}, {"min_tokens": C - P + 1},
            {"min_tokens": 10, "max_tokens": 5}, {"min_tokens": 5, "max_tokens": 5},
            {"min_tokens": 64, "max_tokens": C - P}]
    return out


def cases():
    for C, default, port in SERVERS:
        big = C > 100000
        if big:
            targets = [1000, 131072, 229376, 229377, 255970, 262000, 262143, 262144, 262145, 262400]
        else:
            targets = [20, 100, 1000, 2048, C - 1001, C - 129, C - 65, C - 17, C - 16, C - 15, C - 2, C - 1, C, C + 1,
                       C + 2, C + 200]
        for kind in ("chat", "text", "ids"):
            for P in targets:
                if kind == "ids" and (big or P not in (20, 1000, C - 16, C - 1, C, C + 1)):
                    continue
                k = k_for(kind, P)
                if k < 1:
                    continue
                tokens, chars = measure(kind, k)
                assert tokens == P, (kind, k, tokens, P)
                for spec in specs(C, P, chars, kind):
                    streams = (False, True) if not big or spec in ({}, {"max_tokens": 32768}) else (False,)
                    for stream in streams:
                        yield C, default, port, kind, k, P, chars, dict(spec, stream=stream)


def run(case):
    C, default, port, kind, k, P, chars, body = case
    request = dict(body, model="m")
    if kind == "chat":
        path = "/v1/chat/completions/render"
        request["messages"] = [{"role": "user", "content": content(k)}]
    else:
        path = "/v1/completions/render"
        request["prompt"] = content(k) if kind == "text" else A * k
    status, text = post(port, path, request)
    if status == 200:
        rendered = json.loads(text)
        rendered = rendered[0] if isinstance(rendered, list) else rendered
        assert len(rendered["token_ids"]) == P, (case, len(rendered["token_ids"]))
        # SamplingParams serialises without its defaults, and max_tokens defaults to 16.
        answer = {"status": 200, "max_tokens": rendered["sampling_params"].get("max_tokens", 16)}
    else:
        message = json.loads(text)["error"]["message"]
        if "characters (more than" in message:
            assert f"contains {chars} characters" in message, (case, message)
        answer = {"status": status, "http_body": text}
    return {
        "max_model_len": C,
        "default_max_tokens": default,
        "max_chars_per_token": L,
        "endpoint": "chat" if kind == "chat" else "completions",
        "prompt": "tokens" if kind == "ids" else "text",
        "k": k,
        "prompt_tokens": P,
        "prompt_characters": chars,
        "body": body,
        "vllm": answer,
    }


def main():
    all_cases = list(cases())
    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(run, all_cases))
    for n, row in enumerate(rows, 1):
        sys.stdout.write(json.dumps(dict(id=n, **row), ensure_ascii=False) + "\n")
    print(f"{len(rows)} cases, L={L}, chat template overhead {OVERHEAD} tokens", file=sys.stderr)


if __name__ == "__main__":
    main()
