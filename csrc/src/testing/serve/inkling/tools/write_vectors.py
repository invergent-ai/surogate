# agent-EN-f2: join request bodies with the reference driver's outputs into one vector file per case.
# python3 write_vectors.py KIND OUT_DIR BODIES.jsonl REF_OUT.jsonl [BODIES.jsonl REF_OUT.jsonl ...]
#   KIND renderer: {"case","source","meta","body","expected":{"prompt_token_ids"} | {"error"}}
#   KIND parser:   {"case","source","meta","body","prompt_token_ids","output_token_ids","engine_finish_reason",
#                   "expected":{output_text, reasoning, content, tool_calls, finish_reason, reasoning_tokens} | {"error"}}
import hashlib, json, os, re, sys

kind, out_dir = sys.argv[1], sys.argv[2]
pairs = sys.argv[3:]
os.makedirs(out_dir, exist_ok=True)
index = []
for bodies, ref in zip(pairs[0::2], pairs[1::2]):
    refs = {}
    for line in open(ref):
        r = json.loads(line)
        refs[r["case"]] = r
    for line in open(bodies):
        item = json.loads(line)
        r = refs[item["case"]]
        expected = {k: v for k, v in r.items() if k != "case" and k != "trace"}
        vec = {"case": item["case"], "source": item.get("source"), "meta": item.get("meta"), "body": item["body"]}
        if kind == "parser":
            for k in ("prompt_token_ids", "output_token_ids", "engine_finish_reason"):
                vec[k] = item[k]
        vec["expected"] = expected
        name = re.sub(r"[^A-Za-z0-9_.=+-]", "_", item["case"])[:150]
        path = os.path.join(out_dir, name + ".json")
        data = json.dumps(vec, ensure_ascii=False)
        with open(path, "w") as f:
            f.write(data)
        index.append({"case": item["case"], "file": name + ".json", "error": "error" in expected,
                      "n_ids": len(expected.get("prompt_token_ids") or vec.get("output_token_ids") or []),
                      "sha256": hashlib.sha256(data.encode()).hexdigest()[:16]})
with open(os.path.join(out_dir, "INDEX.json"), "w") as f:
    json.dump({"kind": kind, "n": len(index), "n_error": sum(i["error"] for i in index),
               "reference": "vllm/vllm-openai:inkling-2-mini-x86_64-7cb5dae (vLLM 0.1.dev25855+g7cb5dae22), "
                            "inkling_ref_driver.py", "cases": index}, f, indent=0)
print(kind, len(index), "vectors,", sum(i["error"] for i in index), "error vectors ->", out_dir)
