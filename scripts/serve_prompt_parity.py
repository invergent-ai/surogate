#!/usr/bin/env python3
"""Prompt-render parity between the Surogate engine and vLLM's OpenAI-compatible server.

The same Chat Completions request must reach the model as the same token ids on both servers: a
model measured or trained on vLLM's rendering of a request and served another one is being given a
different prompt. This tool renders a set of requests both ways and compares the token ids.

  reference   token ids vLLM's OpenAI server would prefill for each request. With vLLM importable
              (--impl vllm, the default when it is) this runs vLLM's own code: ChatCompletionRequest
              validation, tool dicts (`tool.model_dump()`), parse_chat_messages and
              safe_apply_chat_template. --impl hf reproduces that pipeline on transformers alone
              (vllm_conversation / vllm_tools below, written against vLLM 0.30/0.31).
  engine      the engine's ids for the same requests (POST /tokenize with the request body).
  compare     identical / differing / refused counts per category, with the first mismatch of each
              differing request decoded from both sides.
  synthetic   write the edge-case request set (tool schemas without parameters or description,
              strict, nested objects, unicode, numbers, tool_choice modes, history with reasoning,
              multi-part content, ...).

Requests are JSONL, one {"id", "category", "body"} per line, body = a Chat Completions request.

  python scripts/serve_prompt_parity.py synthetic --out requests.jsonl
  python scripts/serve_prompt_parity.py reference --requests requests.jsonl --tokenizer MODEL_DIR --out ref.jsonl
  python scripts/serve_prompt_parity.py engine --requests requests.jsonl --url http://127.0.0.1:8000 \\
      --api-key-file KEY --out eng.jsonl
  python scripts/serve_prompt_parity.py compare --requests requests.jsonl --reference ref.jsonl \\
      --engine eng.jsonl --tokenizer MODEL_DIR
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# --------------------------------------------------------------------------------------------------
# vLLM's preprocessing of a Chat Completions request, on transformers alone.


def vllm_tools(tools):
    """`[tool.model_dump() for tool in request.tools]`: ChatCompletionToolsParam / FunctionDefinition
    declared fields only, in declaration order, None for an absent description or parameters,
    strict / defer_loading only when set."""
    if tools is None:
        return None
    out = []
    for tool in tools:
        fn = tool["function"]
        function = {"name": fn["name"], "description": fn.get("description"), "parameters": fn.get("parameters")}
        if fn.get("strict") is not None:
            function["strict"] = fn["strict"]
        defer = fn.get("defer_loading")
        if defer is None:
            defer = tool.get("defer_loading")
        if defer is not None:
            function["defer_loading"] = defer
        item = {"type": "function", "function": function}
        if tool.get("defer_loading") is not None:
            item["defer_loading"] = tool["defer_loading"]
        out.append(item)
    return out


_TEXT_PARTS = ("text", "input_text", "output_text", "refusal", "thinking")


def vllm_conversation(messages, content_format="openai", template=""):
    """parse_chat_messages + _postprocess_messages (+ the developer-role conversion of
    safe_apply_chat_template) for text-only messages."""
    conversation = []
    for raw in messages:
        message = copy.deepcopy(raw)
        legacy = message.pop("reasoning_content", None)  # ChatCompletionRequest._normalize_messages_before
        if legacy is not None and message.get("reasoning") is None:
            message["reasoning"] = legacy
        role, content, reasoning = message["role"], message.get("content"), message.get("reasoning")
        parts = [] if content is None else ([{"type": "text", "text": content}] if isinstance(content, str) else content)
        texts = []
        for part in parts:
            if isinstance(part, str):
                texts.append({"type": "text", "text": part})
            elif part.get("type") in _TEXT_PARTS:
                key = "refusal" if part["type"] == "refusal" else ("thinking" if part["type"] == "thinking" else "text")
                extra = {k: v for k, v in part.items() if k not in ("type", "text", "refusal", "thinking")}
                texts.append({"type": "text", "text": part.get(key, part.get("text")), **extra})
            else:
                raise ValueError(f"media part {part.get('type')!r}: the hf reference is text-only")
        out = {"role": role, "content": texts if content_format == "openai" else "\n".join(t["text"] for t in texts)}
        if role == "assistant":
            if message.get("tool_calls") is not None:
                out["tool_calls"] = list(message["tool_calls"])
            if reasoning is not None:
                out["reasoning"] = reasoning
                out["reasoning_content"] = reasoning
        elif role == "tool":
            if "tool_call_id" in message:
                out["tool_call_id"] = message["tool_call_id"]
            if isinstance(out["content"], list):
                out["content"] = "\n".join(t["text"] for t in out["content"])
        if isinstance(message.get("name"), str):
            out["name"] = message["name"]
        if role == "developer":
            out["tools"] = message.get("tools")
        conversation.append(out)
    for message in conversation:  # _postprocess_messages: arguments strings become dicts
        if message["role"] == "assistant" and isinstance(message.get("tool_calls"), list):
            if not message["tool_calls"]:
                message.pop("tool_calls")
                continue
            for call in message["tool_calls"]:
                function = call["function"]
                arguments = function.get("arguments")
                if arguments:
                    if not isinstance(arguments, (dict, list)):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = None
                    function["arguments"] = arguments if isinstance(arguments, dict) else {}
                else:
                    function["arguments"] = {}
    if any(m["role"] == "developer" for m in conversation) and \
            '"developer"' not in template and "'developer'" not in template:
        for m in conversation:
            if m["role"] == "developer":
                m["role"] = "system"
                m.pop("tools", None)
        systems, rest, late = [], [], False
        for i, m in enumerate(conversation):
            if m["role"] != "system":
                rest.append(m)
                continue
            late = late or i > 0 or bool(systems)
            text = m.get("content", "")
            if isinstance(text, list):
                text = "\n".join(p["text"] if isinstance(p, dict) else p for p in text)
            if text:
                systems.append(text)
        if late:
            conversation = [{"role": "system", "content": "\n\n".join(systems)}, *rest]
    return conversation


def hf_reference_tokens(tokenizer, template, body, content_format="openai"):
    kwargs = dict(body.get("chat_template_kwargs") or {})
    kwargs.setdefault("add_generation_prompt", body.get("add_generation_prompt", True))
    kwargs.setdefault("continue_final_message", body.get("continue_final_message", False))
    kwargs.pop("tokenize", None)
    kwargs.pop("chat_template", None)
    ids = tokenizer.apply_chat_template(vllm_conversation(body["messages"], content_format, template),
                                        tools=vllm_tools(body.get("tools")), chat_template=template,
                                        tokenize=True, return_dict=False, **kwargs)
    return list(ids)


def vllm_reference(tokenizer, template, model_dir):
    """vLLM's own pipeline, for a machine that has it installed."""
    import types

    from vllm.entrypoints.chat_utils import parse_chat_messages
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.renderers.hf import _detect_content_format, safe_apply_chat_template

    fmt = _detect_content_format(template, default="string")
    config = types.SimpleNamespace(multimodal_config=None, enable_prompt_embeds=False, hf_config=None,
                                   tokenizer=model_dir, model=model_dir, tokenizer_revision=None,
                                   code_revision=None, trust_remote_code=False, max_model_len=1 << 22)

    def render(body):
        request = ChatCompletionRequest.model_validate(copy.deepcopy(body))
        tools = None if request.tools is None else [t.model_dump() for t in request.tools]
        conversation, _, _ = parse_chat_messages(request.messages, config, fmt)
        kwargs = request.build_chat_params(template, fmt).get_apply_chat_template_kwargs()
        kwargs["chat_template"] = template
        return list(safe_apply_chat_template(config, tokenizer, conversation, tools=tools, tokenize=True, **kwargs))

    return render, fmt


# --------------------------------------------------------------------------------------------------


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def cmd_reference(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    template = open(args.chat_template or f"{args.tokenizer}/chat_template.jinja").read()
    impl = args.impl
    if impl == "auto":
        try:
            import vllm  # noqa: F401
            impl = "vllm"
        except ImportError:
            impl = "hf"
    if impl == "vllm":
        render, fmt = vllm_reference(tokenizer, template, args.tokenizer)
    else:
        fmt = args.content_format

        def render(body):
            return hf_reference_tokens(tokenizer, template, body, fmt)
    print(f"reference: {impl}, content format {fmt}", file=sys.stderr)
    with open(args.out, "w") as out:
        for item in read_jsonl(args.requests):
            try:
                out.write(json.dumps({"id": item["id"], "tokens": render(item["body"])}) + "\n")
            except Exception as e:  # noqa: BLE001 - a request the reference refuses
                out.write(json.dumps({"id": item["id"], "error": f"{type(e).__name__}: {e}"[:500]}) + "\n")


def cmd_engine(args):
    key = open(args.api_key_file).read().strip() if args.api_key_file else None

    def one(item):
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        request = urllib.request.Request(args.url.rstrip("/") + "/tokenize", data=json.dumps(item["body"]).encode(),
                                         headers=headers)
        try:
            return {"id": item["id"], "tokens": json.loads(urllib.request.urlopen(request, timeout=300).read())["tokens"]}
        except urllib.error.HTTPError as e:
            return {"id": item["id"], "error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:400]}"}
        except Exception as e:  # noqa: BLE001
            return {"id": item["id"], "error": f"{type(e).__name__}: {e}"[:400]}

    items = read_jsonl(args.requests)
    with ThreadPoolExecutor(args.concurrency) as pool, open(args.out, "w") as out:
        for result in pool.map(one, items):
            out.write(json.dumps(result) + "\n")


def compare(requests, reference, engine, decode=None):
    """{category: {kind: count}} and per-request details (first mismatch, decoded when possible)."""
    ref = {o["id"]: o for o in reference}
    eng = {o["id"]: o for o in engine}
    stats = defaultdict(lambda: defaultdict(int))
    details = []
    for item in requests:
        rid, cat = item["id"], item["category"]
        r, e = ref.get(rid), eng.get(rid)
        if r is None or e is None:
            kind = "missing"
        elif "error" in r and "error" in e:
            kind = "both_refuse"
        elif "error" in r:
            kind = "ref_refuses"
        elif "error" in e:
            kind = "eng_refuses"
        elif r["tokens"] == e["tokens"]:
            kind = "identical"
        else:
            kind = "differing"
        stats[cat][kind] += 1
        if kind == "identical":
            continue
        detail = {"id": rid, "category": cat, "kind": kind}
        if kind == "differing":
            a, b = r["tokens"], e["tokens"]
            at = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
            detail.update(at=at, len_ref=len(a), len_eng=len(b))
            if decode:
                lo = max(0, at - 12)
                detail.update(ref=decode(a[lo:at + 40]), eng=decode(b[lo:at + 40]))
        else:
            detail.update(ref=(r or {}).get("error"), eng=(e or {}).get("error"))
        details.append(detail)
    return stats, details


def cmd_compare(args):
    decode = None
    if args.tokenizer:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(f"{args.tokenizer}/tokenizer.json")

        def decode(ids):
            return tok.decode(ids, skip_special_tokens=False)
    stats, details = compare(read_jsonl(args.requests), read_jsonl(args.reference), read_jsonl(args.engine), decode)
    kinds = ["identical", "differing", "eng_refuses", "ref_refuses", "both_refuse", "missing"]
    print(f"{'category':22s} " + " ".join(f"{k:>11s}" for k in kinds))
    total = defaultdict(int)
    for cat in sorted(stats):
        print(f"{cat:22s} " + " ".join(f"{stats[cat][k]:11d}" for k in kinds))
        for k in kinds:
            total[k] += stats[cat][k]
    print(f"{'TOTAL':22s} " + " ".join(f"{total[k]:11d}" for k in kinds))
    shown = defaultdict(int)
    for d in details:
        if shown[(d["category"], d["kind"])] >= args.show:
            continue
        shown[(d["category"], d["kind"])] += 1
        print(f"\n--- {d['id']} {d['kind']}" + (f" at token {d['at']} ({d['len_ref']} vs {d['len_eng']})" if "at" in d else ""))
        print("REF:", repr(d.get("ref"))[:600])
        print("ENG:", repr(d.get("eng"))[:600])
    if args.report:
        with open(args.report, "w") as fh:
            json.dump({"stats": {c: dict(v) for c, v in stats.items()}, "total": dict(total), "details": details},
                      fh, indent=1, ensure_ascii=False)
    return 0 if total["differing"] == total["eng_refuses"] == total["ref_refuses"] == 0 else 1


# --------------------------------------------------------------------------------------------------
# Edge cases a request set should carry besides real traffic.


def synthetic_requests(model="surogate"):
    weather = {"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city.",
               "parameters": {"type": "object", "properties": {
                   "zeta_city": {"type": "string", "description": "City"}, "alpha_unit": {"type": "string", "enum": ["c", "f"]},
                   "days": {"type": "integer", "minimum": 1, "maximum": 14, "default": 3}}, "required": ["zeta_city"]}}}
    nested = {"type": "function", "function": {"name": "create_event", "description": "Create an event.",
              "parameters": {"type": "object", "required": ["title", "when"], "properties": {
                  "title": {"type": "string"},
                  "when": {"type": "object", "properties": {"start": {"type": "string", "format": "date-time"},
                           "end": {"type": "string"}, "tz": {"type": "string", "default": "UTC"}}, "required": ["start"]},
                  "attendees": {"type": "array", "items": {"type": "object", "properties": {
                      "name": {"type": "string"}, "email": {"type": "string"}, "optional": {"type": "boolean", "default": False}}}},
                  "priority": {"type": ["integer", "null"], "default": None},
                  "ratio": {"type": "number", "default": 1.0, "minimum": 1e-5, "maximum": 1e16, "multipleOf": 0.25},
                  "big": {"type": "integer", "default": 12345678901234567, "examples": [-0.0, 3.14159, 2.5e-7, 1e300]},
                  "flag": {"type": "boolean", "default": True}}}}}
    unicode = {"type": "function", "function": {"name": "traduire",
               "description": "Traduit un texte — 日本語/中文 ✓ \U0001f600 «quotes» \"dq\" \\ back\\slash \t tab \u0001 ctl   sep",
               "parameters": {"type": "object", "properties": {"texte": {"type": "string", "description": "Ünïcødé ключ"},
                              "langue": {"type": "string", "enum": ["fr", "日本"]}}}}}
    no_params = {"type": "function", "function": {"name": "ping", "description": "Ping the server."}}
    no_desc = {"type": "function", "function": {"name": "list_files", "parameters": {"type": "object", "properties": {}}}}
    bare = {"type": "function", "function": {"name": "noop"}}
    strict = {"type": "function", "function": {"name": "set_mode", "description": "Set mode.", "strict": True,
              "parameters": {"type": "object", "properties": {"mode": {"type": "string"}}, "required": ["mode"],
                             "additionalProperties": False}}}
    reordered = {"type": "function", "function": {"strict": False, "parameters": {"properties": {"b": {"type": "string"},
                 "a": {"type": "string"}}, "type": "object"}, "description": "Keys sent in non-canonical order.", "name": "reordered"}}
    extras = {"type": "function", "function": {"name": "with_extras", "description": "Has unknown fields.",
              "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}, "examples": [{"x": "1"}]}}
    user = lambda text: {"role": "user", "content": text}  # noqa: E731
    system = lambda text: {"role": "system", "content": text}  # noqa: E731

    def call(cid, name, args, content=None, reasoning=None):
        message = {"role": "assistant", "content": content,
                   "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": args}}]}
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        return message

    def result(cid, text):
        return {"role": "tool", "tool_call_id": cid, "content": text}

    out = []

    def case(name, messages, tools=None, kwargs=None, **extra):
        body = {"model": model, "messages": messages}
        if tools is not None:
            body["tools"] = tools
        body["chat_template_kwargs"] = {"enable_thinking": True} if kwargs is None else kwargs
        body.update(extra)
        out.append({"id": f"syn_{name}", "category": "synthetic", "body": body})

    args1 = json.dumps({"zeta_city": "Paris", "alpha_unit": "c", "days": 5})
    args2 = json.dumps({"title": "Sync", "when": {"start": "2026-10-18T10:00", "tz": "Europe/Bucharest", "end": "x"},
                        "attendees": [{"name": "Ана", "email": "a@b.c", "optional": False}], "priority": None,
                        "ratio": 1.0, "big": 12345678901234567, "flag": True, "small": 1e-05, "huge": 1e16,
                        "neg": -0.0, "pi": 3.14159}, ensure_ascii=False)
    case("plain", [user("Hello there")])
    case("plain_system", [system("You are terse."), user("Hi")])
    case("thinking_off", [user("Hi")], kwargs={"enable_thinking": False})
    case("no_kwargs", [user("Hi")], kwargs={})
    case("tools_basic", [system("Be helpful."), user("Weather in Paris?")], [weather])
    case("tools_nested_numbers", [user("Plan a meeting")], [nested])
    case("tools_unicode", [user("Traduis «bonjour» en 日本語 \U0001f600")], [unicode])
    case("tools_no_params", [user("ping")], [no_params])
    case("tools_no_desc", [user("list")], [no_desc])
    case("tools_bare", [user("noop")], [bare])
    case("tools_strict_true", [user("set mode fast")], [strict])
    case("tools_strict_false_reordered", [user("call it")], [reordered])
    case("tools_extra_fields", [user("x")], [extras])
    case("tools_many", [user("Do things")], [weather, nested, unicode, no_params, no_desc, bare, strict, reordered])
    for choice in ("none", "required", "auto", {"type": "function", "function": {"name": "get_weather"}}):
        label = choice if isinstance(choice, str) else "named"
        case(f"tool_choice_{label}", [user("Weather in Paris?")], [weather, no_desc], tool_choice=choice)
    case("history_tool_calls", [system("sys"), user("Weather in Paris then plan a sync"),
                                call("c1", "get_weather", args1, reasoning="I should check the weather first."),
                                result("c1", '{"temp": 21.5, "sky": "clear"}'),
                                call("c2", "create_event", args2, content="Now the event.", reasoning="Weather is fine."),
                                result("c2", "created")], [weather, nested])
    history = [user("first question"), {"role": "assistant", "content": "first answer", "reasoning_content": "thinking about the first"},
               user("second question"), call("c3", "get_weather", args1, reasoning="need weather"), result("c3", "sunny"),
               {"role": "assistant", "content": "It is sunny.", "reasoning_content": "done"}, user("third question")]
    case("history_two_users_reasoning", history, [weather])
    case("history_two_users_reasoning_preserve", history[:3], [weather], kwargs={"enable_thinking": True, "preserve_thinking": True})
    case("history_reasoning_key", [user("q1"), {"role": "assistant", "content": "a1", "reasoning": "via reasoning key"}, user("q2")])
    case("history_reasoning_empty_with_think_tags", [user("q1"), {"role": "assistant", "content": "<think>inline</think>answer",
                                                                   "reasoning_content": ""}, user("q2")])
    case("history_think_tags_in_content", [user("q1"), {"role": "assistant", "content": "<think>\ninline thought\n</think>\n\nanswer"},
                                           user("q2")])
    case("history_parallel_calls", [user("Weather in Paris and Rome"), {"role": "assistant", "content": None, "reasoning_content": "two calls",
         "tool_calls": [{"id": "p1", "type": "function", "function": {"name": "get_weather", "arguments": '{"zeta_city": "Paris"}'}},
                        {"id": "p2", "type": "function", "function": {"name": "get_weather", "arguments": '{"days": 2, "zeta_city": "Rome"}'}}]},
         result("p1", "sunny"), result("p2", "rain")], [weather])
    case("history_empty_args", [user("ping it"), call("e1", "ping", "{}"), result("e1", "pong")], [no_params])
    case("history_content_and_calls", [user("go"), call("e2", "get_weather", args1, content="Checking now.")], [weather])
    case("multipart_user_text", [{"role": "user", "content": [{"type": "text", "text": "part one."}, {"type": "text", "text": "part two."}]}])
    case("multipart_system_text", [{"role": "system", "content": [{"type": "text", "text": "Rule A."}, {"type": "text", "text": "Rule B."}]},
                                   user("hi")], [weather])
    case("multipart_tool_text", [user("go"), call("m1", "get_weather", args1), {"role": "tool", "tool_call_id": "m1",
         "content": [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}]}], [weather])
    case("whitespace_content", [system("  padded system  \n"), user("\n  padded user \n")])
    case("unicode_messages", [user("Ce mai faci? Ăîșțâ — 你好 — \U0001f680 —  nbsp — ​zwsp")])
    # Jinja's `trim` is Python's strip(): Unicode whitespace too (U+3000, U+00A0, U+2028, \v, \f, ...)
    case("unicode_whitespace_edges", [system("\x0bsystem rules\x0c"), user("　 hello  "),
                                      {"role": "assistant", "content": "answer ", "reasoning_content": " thought　"},
                                      user(" next\x1c"),
                                      call("w1", "get_weather", args1, content=" "), result("w1", " result ")], [weather])
    # Python writes floats in [1e15, 1e16) without an exponent, and correctly rounded
    decade = {"type": "function", "function": {"name": "big_numbers", "description": "Floats near 1e15.",
              "parameters": {"type": "object", "properties": {"x": {"type": "number", "default": 1e15,
                             "maximum": 1234567890123456.7, "examples": [1.5e15, 9999999999999998.0, 1e16, 0.0001, 1e-05]}}}}}
    case("numbers_decade_edge", [user("x"), call("d1", "big_numbers", json.dumps({"x": 1e15, "y": 1.5e15, "z": 2.0, "w": [1e15, 0.1]})),
                                 result("d1", "ok")], [decade])
    case("assistant_last_prefill", [user("Say hi"), {"role": "assistant", "content": "Hi"}])
    case("developer_role", [{"role": "developer", "content": "dev rules"}, user("hi")])
    case("developer_after_system", [system("sys rules"), {"role": "developer", "content": "dev rules"}, user("hi")])
    case("tool_name_dots", [user("x")], [{"type": "function", "function": {"name": "ns.tool-name_1", "description": "d",
                                                                         "parameters": {"type": "object", "properties": {}}}}])
    case("params_with_defs", [user("x")], [{"type": "function", "function": {"name": "refs", "description": "uses $defs",
         "parameters": {"$defs": {"Item": {"type": "object", "properties": {"id": {"type": "integer"}}}},
                        "type": "object", "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}}}}}])
    return out


def cmd_synthetic(args):
    with open(args.out, "w") as fh:
        for item in synthetic_requests(args.model):
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("reference")
    p.add_argument("--requests", required=True)
    p.add_argument("--tokenizer", required=True, help="HF model/tokenizer directory")
    p.add_argument("--chat-template", help="template file (default: TOKENIZER/chat_template.jinja)")
    p.add_argument("--impl", choices=["auto", "vllm", "hf"], default="auto")
    p.add_argument("--content-format", choices=["openai", "string"], default="openai",
                   help="--impl hf: vLLM's content format for the template (Qwen's templates: openai)")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_reference)
    p = sub.add_parser("engine")
    p.add_argument("--requests", required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--api-key-file")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_engine)
    p = sub.add_parser("compare")
    p.add_argument("--requests", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--tokenizer", help="decode mismatches with this tokenizer.json directory")
    p.add_argument("--report")
    p.add_argument("--show", type=int, default=3, help="mismatches printed per category and kind")
    p.set_defaults(func=cmd_compare)
    p = sub.add_parser("synthetic")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="surogate")
    p.set_defaults(func=cmd_synthetic)
    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
