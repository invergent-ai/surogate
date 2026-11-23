#!/usr/bin/env python3
"""Lane T8: the engine's prompt log-probs of each row's token ids (stdlib only).   t8_score_client.py <base url> <rows.json> <model> <out.json>
model "inkling" = the base model (no adapter slot); any other name = that LoRA module. Position t holds log p(ids[t] | ids[:t]); position 0 holds 0."""
import json
import sys
import urllib.request


def post(base, body):
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def main():
    base, rows, model, out = sys.argv[1:5]
    res = []
    for r in json.load(open(rows)):
        d = post(base, {"model": model, "prompt": "", "tokens": r["ids"], "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 1,
                        "logprobs": 1, "return_token_ids": True})
        plp = d["prompt_logprobs"]
        res.append({"lp": [0.0] + [float(plp[t][str(r["ids"][t])]["logprob"]) for t in range(1, len(r["ids"]))]})
        print(f"{model}: row of {len(r['ids'])} tokens scored", flush=True)
    json.dump(res, open(out, "w"))


if __name__ == "__main__":
    sys.exit(main())
