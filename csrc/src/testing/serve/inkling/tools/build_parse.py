# agent-EN-f2 (ENGINE-INKLING lane f2), 2026-11-21.
# Parser corpus: model outputs (token ids) -> what vLLM's Inkling parser returns, for inkling_ref_driver.py parse.
#
# Outputs are built from REAL Inkling-2-Mini replies (the assistant message each real request of real_bodies.jsonl got:
# the session's llm.response after that request: its reasoning, text and tool calls) encoded in the block format the
# renderer writes for an assistant turn (= what the model samples):
#   <|message_model|><|content_thinking|>R<|end_message|> [<|message_model|><|content_text|>T<|end_message|>]
#   (<|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":ARGS}<|end_message|>)* <|content_model_end_sampling|>
# plus deterministic variants that reach the parser's other branches (length cuts, whitespace, text after tools,
# dropped special tokens, literal marker text, malformed tool JSON, tool_choice / include_reasoning, coercion).
# Token ids come from the model's tokenizer.json (tokenizers, special tokens split in payload text).
# Run with the lane venv: venv/bin/python -I build_parse.py REAL_BODIES REAL_RENDER_OUT OUT.jsonl
import json
import os
import random
import sys

import tokenizers

IK = "/mnt/data/work/rl/inkling-2026-11-21"
TOK = tokenizers.Tokenizer.from_file("/mnt/data/work/rl/engine-inkling-2026-11-21/model/tokenizer.json")
TOK.encode_special_tokens = True
SP = {t.content: i for i, t in TOK.get_added_tokens_decoder().items()}
MM, TX, TH = SP["<|message_model|>"], SP["<|content_text|>"], SP["<|content_thinking|>"]
EM, ES, TJ = SP["<|end_message|>"], SP["<|content_model_end_sampling|>"], SP["<|content_invoke_tool_json|>"]
TT, TE, EOT = SP["<|content_invoke_tool_text|>"], SP["<|content_tool_error|>"], SP["<|endoftext|>"]
MU, MSYS, U11 = SP["<|message_user|>"], SP["<|message_system|>"], SP["<|unused_200011|>"]


def enc(text):
    return TOK.encode(text, add_special_tokens=False).ids if text else []


def blocks_for(msg, args_override=None):
    out = []
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    if reasoning:
        out += [MM, TH] + enc(reasoning) + [EM]
    content = msg.get("content")
    if isinstance(content, str) and content:
        out += [MM, TX] + enc(content) + [EM]
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        args = fn.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args or {}, separators=(",", ":"), ensure_ascii=False)
        if args_override is not None:
            args = args_override(i, args)
        payload = '{"name":' + json.dumps(name, ensure_ascii=False) + ',"args":' + (args or "{}") + "}"
        out += [MM] + enc(name) + [TJ] + enc(payload) + [EM]
    return out


def load_events(path):
    return [json.loads(x) for x in open(os.path.join(IK, path, "events.jsonl")) if x.strip()]


def reply_for(item):
    """The llm.response that answered the call_index-th llm.request of the session."""
    events = load_events(item["source"])
    k = item["meta"]["call_index"]
    seen = -1
    for e in events:
        if e["type"] == "llm.request":
            seen += 1
        elif e["type"] == "llm.response" and seen == k:
            return (e.get("data") or {}).get("message"), (e.get("data") or {}).get("finish_reason")
    return None, None


def main():
    bodies_path, render_path, out_path = sys.argv[1:4]
    prompts = {}
    for line in open(render_path):
        r = json.loads(line)
        if "prompt_token_ids" in r:
            prompts[r["case"]] = r["prompt_token_ids"][-16:]
    rng = random.Random(20261121)
    items = []

    def add(case, body, prompt, out, finish, note):
        items.append({"case": case, "body": body, "prompt_token_ids": prompt, "output_token_ids": out,
                      "engine_finish_reason": finish, "meta": {"variant": note}, "source": None})

    n_real = 0
    for line in open(bodies_path):
        item = json.loads(line)
        if item["meta"].get("kind") != "main":
            continue
        msg, fin = reply_for(item)
        if not msg:
            continue
        body = dict(item["body"])
        body["messages"] = body["messages"][:2]  # the parser never reads the history; keep vectors small
        prompt = prompts.get(item["case"])
        if not prompt:
            continue
        base = blocks_for(msg)
        if not base:
            continue
        case = item["case"].replace("-c", "-r", 1) if False else "p-" + item["case"]
        n_real += 1
        v = n_real % 12
        if fin == "length":
            add(case, body, prompt, base, "length", "real-length")
        else:
            add(case, body, prompt, base + [ES], "stop", "real")
        # one variant per real reply, cycling through the branches
        if v == 0:  # length cut at a random point
            cut = rng.randrange(1, len(base))
            add(case + "-cut", body, prompt, base[:cut], "length", f"cut@{cut}")
        elif v == 1:  # trailing whitespace on reasoning, whitespace-only text block before tools
            ws = [MM, TH] + enc((msg.get("reasoning_content") or "r") + " \n\n ") + [EM]
            add(case + "-ws", body, prompt, ws + [MM, TX] + enc("  \n") + [EM] + base + [ES], "stop", "ws")
        elif v == 2:  # text block after the tool blocks
            add(case + "-textafter", body, prompt, base + [MM, TX] + enc(" Done.\n") + [EM, ES], "stop", "text-after")
        elif v == 3:  # dropped special tokens inside text and reasoning
            out = [MM, TH] + enc("a") + [EOT] + enc("b") + [MU, EM, MM, TX] + enc("x") + [U11, MSYS] + enc("y") + [EM]
            add(case + "-drops", body, prompt, out + base + [ES], "stop", "drops")
        elif v == 4:  # marker spellings as plain text (the parser matches text, not token ids)
            out = [MM, TH] + enc("think <|end_message|> more") + [EM, MM, TX] + enc("see <|content_thinking|>z") + [EM]
            add(case + "-literal", body, prompt, out + [ES], "stop", "literal-markers")
        elif v == 5:  # tool_choice none: tool blocks must not become calls
            b2 = dict(body, tool_choice="none")
            add(case + "-tcnone", b2, prompt, base + [ES], "stop", "tool_choice-none")
        elif v == 6:  # include_reasoning false
            add(case + "-noreason", dict(body, include_reasoning=False), prompt, base + [ES], "stop", "no-reasoning")
        elif v == 7:  # malformed / partial tool JSON forms
            forms = ['{"name":"f","args":{"a":1}', '{"args":{"a":1},"name":"g"}', '{"name":"h"}', '{"name":"i","args":[1]}',
                     'not json', '{"name":"j","args":{"x":"<|end"}}', '{"name": "k" , "args" : { "q" : "v" } }']
            out = [MM, TH] + enc("try") + [EM]
            for f in forms:
                out += [MM, TJ] + enc(f) + [EM]
            add(case + "-badjson", body, prompt, out + [ES], "stop", "bad-json")
        elif v == 8:  # raw-text tool block and tool-error block render as text
            out = [MM, TT] + enc("raw call text") + [EM, MM, TE] + enc("an error") + [EM]
            add(case + "-tooltext", body, prompt, out + base + [ES], "stop", "tool-text/error")
        elif v == 9:  # several thinking blocks, thinking after text, no leading message_model
            out = [TH] + enc("first") + [EM, MM, TH] + enc("second ") + [EM, MM, TX] + enc("t") + [EM, MM, TH] + enc("third") + [EM]
            add(case + "-multithink", body, prompt, out + base + [ES], "stop", "multi-think")
        elif v == 10:  # end-sampling without end_message, mid tool JSON cut
            out = [MM, TH] + enc("x") + [ES]
            add(case + "-endsampling", body, prompt, out, "stop", "end-sampling-closes-thinking")
            if msg.get("tool_calls"):
                full = blocks_for(msg)
                cut = max(1, len(full) - 3)
                add(case + "-toolcut", body, prompt, full[:cut], "length", "tool-json-cut")
        elif v == 11:  # no thinking at all, text only
            out = [MM, TX] + enc((msg.get("content") or "Answer.") + "\n") + [EM]
            add(case + "-textonly", body, prompt, out + [ES], "stop", "text-only")

    # schema coercion: string / number / bool / null / object / array values against the edge tools
    tools = [{"type": "function", "function": {"name": "typed", "description": "d", "parameters": {"type": "object", "properties": {
        "i": {"type": "integer"}, "n": {"type": "number"}, "b": {"type": "boolean"}, "s": {"type": "string"},
        "o": {"type": "object", "properties": {"k": {"type": "integer"}}}, "a": {"type": "array", "items": {"type": "number"}},
        "u": {"type": ["integer", "null"]}, "e": {"enum": [1, 2, "x"]}, "any": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
        "f": {"type": "float"}, "big": {"type": "integer"}}}}}]
    body = {"model": "cand", "messages": [{"role": "user", "content": "x"}], "tools": tools}
    prompt = [200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 13, 24, 200010]
    argsets = ['{"i":"5","n":"2.50","b":"true","s":5,"o":{"k":"7"},"a":["1","2.5"],"u":"null","e":"2","any":"3","f":"1e3"}',
               '{"i":" 12 ","n":"1e400","b":"0","s":true,"o":"{\\"k\\": 1}","a":"[1, 2]","u":"x","e":"x","big":"123456789012345678901234567890"}',
               '{"i":5,"n":2,"b":false,"s":"plain","u":null,"f":"inf","big":1e3}',
               '{"i":"1_000","n":"-0","b":"TRUE","s":null,"a":[1,"x"],"zz":"unknown key"}',
               '{"i":"0x10","n":".5","b":"yes","s":["l"],"o":{"k":"x","extra":{"y":"1"}},"e":"3"}',
               '{"s":"café \\u00e9 \\ud83d\\ude00","i":"٣","n":"1.0"}']
    for j, a in enumerate(argsets):
        out = [MM, TH] + enc("coerce") + [EM, MM] + enc("typed") + [TJ] + enc('{"name":"typed","args":' + a + "}") + [EM, ES]
        add(f"p-coerce-{j}", body, prompt, out, "stop", "coerce")
    for tc in ["required", {"type": "function", "function": {"name": "typed"}}]:
        out = [MM, TH] + enc("x") + [EM, MM] + enc("typed") + [TJ] + enc('{"name":"typed","args":{"i":"1"}}') + [EM, ES]
        add(f"p-toolchoice-{json.dumps(tc, sort_keys=True)}", dict(body, tool_choice=tc), prompt, out, "stop", "toolchoice")
        add(f"p-toolchoice-text-{json.dumps(tc, sort_keys=True)}", dict(body, tool_choice=tc), prompt,
            [MM, TX] + enc("no call") + [EM, ES], "stop", "toolchoice-text")
    # no tools in the request at all
    nb = {"model": "cand", "messages": [{"role": "user", "content": "x"}]}
    add("p-notools-call", nb, prompt, [MM, TJ] + enc('{"name":"f","args":{"a":"1"}}') + [EM, ES], "stop", "no-tools")
    add("p-notools-text", nb, prompt, [MM, TH] + enc("r ") + [EM, MM, TX] + enc(" hi ") + [EM, ES], "stop", "no-tools")
    add("p-empty", nb, prompt, [ES], "stop", "empty")
    add("p-only-header", nb, prompt, [MM] + enc("hdr"), "length", "header-only")
    add("p-utf8-cut", nb, prompt, [MM, TX] + enc("日本")[:1], "length", "utf8-cut")
    add("p-header-then-end", nb, prompt, [MM] + enc("name") + [EM, ES], "stop", "header-end")

    with open(out_path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(len(items), "parser items from", n_real, "real replies", file=sys.stderr)


if __name__ == "__main__":
    main()
