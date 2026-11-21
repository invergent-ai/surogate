#!/usr/bin/env python3
"""Lane T M4 engine client (stdlib only; runs next to the engine on gpubox1).

  roll  <base_url> <out.json> [groups] [samples] [max_tokens]: GSPO-style groups of chat rollouts through the kit's
        request shape (reasoning_effort xhigh = 0.99 (rule 20), T 1.0, top_p 0.95, top_k 20, logprobs, token ids,
        routed experts) on model eff-policy. Prompts: templated integer products (plumbing data; the adapter of this
        test is discarded).
  score <base_url> <rows.json> <out.json>: the engine's prompt log-probs of each row's full ids (prompt + completion)
        under eff-policy (/v1/completions, token-in), position t = log p(ids[t] | ids[:t]).
"""
import json
import random
import sys
import threading
import time
import urllib.request


def post(base, path, body, timeout=1800):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def roll(base, out, groups=2, samples=4, max_tokens=1024):
    rng = random.Random(20261121)
    jobs = []
    for g in range(groups):
        a, b = rng.randint(12, 99), rng.randint(12, 99)
        for s in range(samples):
            jobs.append({"group": g, "sample": s, "a": a, "b": b, "answer": str(a * b),
                         "body": {"model": "eff-policy",
                                  "messages": [{"role": "user", "content": f"Compute {a} * {b}. Give only the final number."}],
                                  "reasoning_effort": "xhigh", "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                                  "max_tokens": max_tokens, "seed": 1000 * g + s, "logprobs": True,
                                  "return_token_ids": True, "return_routed_experts": True}})
    res = [None] * len(jobs)

    def run(i):
        t0 = time.time()
        try:
            res[i] = {"job": jobs[i], "response": post(base, "/v1/chat/completions", jobs[i]["body"]),
                      "seconds": time.time() - t0}
        except Exception as e:  # noqa: BLE001
            res[i] = {"job": jobs[i], "error": repr(e), "seconds": time.time() - t0}
        r = res[i].get("response", {})
        n = len(((r.get("choices") or [{}])[0]).get("token_ids") or [])
        print(f"rollout g{jobs[i]['group']} s{jobs[i]['sample']}: {n} tokens {res[i]['seconds']:.1f}s "
              f"{res[i].get('error', '')}", flush=True)

    th = [threading.Thread(target=run, args=(i,)) for i in range(len(jobs))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    json.dump(res, open(out, "w"))


def score(base, rows_path, out):
    rows = json.load(open(rows_path))
    res = []
    for r in rows:
        body = {"model": "eff-policy", "prompt": "", "tokens": r["ids"], "max_tokens": 1, "temperature": 0.0,
                "prompt_logprobs": 1, "logprobs": 1, "return_token_ids": True}
        t0 = time.time()
        d = post(base, "/v1/completions", body)
        plp = d["prompt_logprobs"]
        lp = [0.0] + [float(plp[t][str(r["ids"][t])]["logprob"]) for t in range(1, len(r["ids"]))]
        res.append({"lp": lp})
        print(f"score row {len(res) - 1}: {len(r['ids'])} tokens {time.time() - t0:.1f}s", flush=True)
    json.dump(res, open(out, "w"))


if __name__ == "__main__":
    if sys.argv[1] == "roll":
        roll(sys.argv[2], sys.argv[3], *(int(x) for x in sys.argv[4:7]))
    else:
        score(sys.argv[2], sys.argv[3], sys.argv[4])
