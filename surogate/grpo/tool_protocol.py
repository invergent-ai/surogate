"""Checkpoint tool protocols at the HTTP boundary; policy tokens stay intact.

Use the checkpoint's tool-aware chat template, or adapt text-only templates to
JSON tool calls. Parsing never evaluates model output or executes a tool.
"""

from __future__ import annotations

import ast
import json
import re
import uuid
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass

OPEN = "<tool_call>"
CLOSE = "</tool_call>"
DEFAULT_CHAT_TEMPLATE = (
    "{{ bos_token | default('', true) }}{% for m in messages %}"
    "{{ m.role + ':\\n' + m.content + '\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ 'assistant:\\n' }}{% endif %}"
)


def request_tools(body: dict) -> list[dict]:
    tools = body.get("tools")
    if tools is None:
        tools = []
    if not isinstance(tools, list):
        raise ValueError("tools must be a list of function definitions")
    tools = deepcopy(tools)
    choice = body.get("tool_choice", "auto" if tools else "none")
    if choice not in ("auto", "none"):
        raise ValueError("native tool_choice supports 'auto' and 'none'; forced tool decoding is unavailable")
    names = set()
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise ValueError("tools must contain function definitions")
        fn = tool["function"]
        name = fn.get("name")
        if not isinstance(name, str) or not name or re.search(r"[\s<>]", name) or name in names:
            raise ValueError("tool names must be nonempty, unique and contain no whitespace or XML delimiters")
        names.add(name)
        if not isinstance(fn.get("parameters", {}), dict):
            raise ValueError("tool parameters must be a JSON schema object")
        fn.setdefault("description", "")
        fn.setdefault("parameters", dict(type="object", properties={}))
        # Verifiers marks function schemas strict by default. Like the native
        # serving frontend, accept this schema metadata; sampling stays
        # unconstrained and argument errors are handled by the environment.
        if "strict" in fn and not isinstance(fn["strict"], bool):
            raise ValueError("tool strict must be a boolean")
    if not isinstance(body.get("parallel_tool_calls", True), bool):
        raise ValueError("parallel_tool_calls must be a boolean")
    return tools if choice == "auto" else []


def template_messages(messages: list[dict]) -> list[dict]:
    """HF templates expect argument objects; the wire API uses JSON strings."""
    messages = deepcopy(messages)
    for message in messages:
        content = message.get("content")
        if content is None:
            message["content"] = ""
        elif isinstance(content, list):
            # Verifiers/MCP may wrap text-only tool results in content parts.
            if any(
                not isinstance(p, dict) or p.get("type") != "text" or not isinstance(p.get("text"), str)
                for p in content
            ):
                raise ValueError("shared-model GRPO currently accepts text messages only")
            message["content"] = "".join(p["text"] for p in content)
        elif not isinstance(content, str):
            raise ValueError("shared-model GRPO currently accepts text messages only")
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list):
            raise ValueError("message tool_calls must be a list")
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                raise ValueError("message tool_calls must contain function calls")
            fn = call["function"]
            if not isinstance(fn.get("name"), str) or not fn["name"]:
                raise ValueError("tool calls require a function name")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError as exc:
                    raise ValueError("tool-call arguments must be a JSON object") from exc
            if not isinstance(args, dict):
                raise ValueError("tool-call arguments must be a JSON object")
            fn["arguments"] = args
    return messages


@dataclass(frozen=True)
class ToolProtocol:
    name: str
    fallback: bool = False
    chat_template: str | None = None

    def messages(self, messages: list[dict], tools: list[dict]) -> list[dict]:
        messages = template_messages(messages)
        # GPT-OSS's template calls the OpenAI reasoning_content field thinking.
        if self.name == "harmony":
            for message in messages:
                if "reasoning_content" in message:
                    message.setdefault("thinking", message["reasoning_content"])
        if not self.fallback or not (tools or any(m.get("tool_calls") or m.get("role") == "tool" for m in messages)):
            return messages
        # Locate the initial user before adapting tool results into user turns.
        # Verifiers tokenizes [dummy assistant] and [dummy assistant, tool result]
        # separately: both must get an identical schema prefix.
        first_user = None
        for message in messages:
            if message["role"] == "assistant":
                break
            if message["role"] == "user":
                first_user = message
                break
        adapted = []
        for message in messages:
            if message.get("tool_calls"):
                message["content"] += "".join(
                    OPEN + json.dumps(call["function"], ensure_ascii=False) + CLOSE
                    for call in message.pop("tool_calls")
                )
            if message.get("role") == "tool":
                message["role"] = "user"
                message["content"] = (
                    "<tool_response>"
                    + json.dumps(
                        dict(tool_call_id=message.get("tool_call_id"), content=message["content"]), ensure_ascii=False
                    )
                    + "</tool_response>"
                )
                if adapted and adapted[-1]["role"] == "user":
                    adapted[-1]["content"] += "\n" + message["content"]
                    continue
            adapted.append(message)
        if tools:
            # Inject into the first user turn: Gemma's text-only template rejects
            # system/tool roles. Keep this independent of later tool results so
            # /tokenize bridge prefixes remain stable across agent turns.
            instructions = (
                "Available tools: " + json.dumps(tools, ensure_ascii=False) + "\n"
                'To call a tool, emit <tool_call>{"name":"function_name","arguments":{...}}</tool_call>. '
                "Use only the listed names and JSON argument objects. You may emit multiple calls. "
                "After calling tools, end your turn and wait for tool responses. Otherwise answer normally.\n\n"
            )
            if first_user is None:
                adapted.insert(0, dict(role="user", content=instructions))
            else:
                first_user["content"] = instructions + first_user["content"]
        return adapted

    def parse(self, text: str, tools: list[dict], *, prefix: str, final: bool) -> dict:
        if self.name == "harmony":
            return _harmony(text, tools, prefix=prefix) if final else dict(content="", reasoning_content="")
        return parse_output(text, tools, protocol=self.name, prefix=prefix, final=final)


def select_protocol(tokenizer, config: dict, tools: list[dict], kwargs: dict | None = None) -> ToolProtocol:
    """Select from the actual template, including tool_use/custom variants.

    Model types are a fallback for templates that delegate rendering to macros;
    template markers take precedence (MiniCPM5, for example, identifies as Llama).
    """
    override = (kwargs or {}).get("chat_template")
    if not override and not getattr(tokenizer, "chat_template", None):
        template = DEFAULT_CHAT_TEMPLATE
    elif hasattr(tokenizer, "get_chat_template"):
        template = tokenizer.get_chat_template(override, tools=tools)
    else:
        template = override or getattr(tokenizer, "chat_template", "")
    if not isinstance(template, str):
        raise ValueError("a text chat template is required for native tool calling")
    for marker, name in (
        ("<|start|>assistant", "harmony"),
        ("<|tool_call>", "gemma4"),
        ("<|tool_call_start|>", "lfm2"),
        ("<function name=", "minicpm"),
        ("<arg_key>", "arg_xml"),
        ("<function=", "qwen_coder"),
        ("<tool_call>", "json_xml"),
        ("<|python_tag|>", "llama_json"),
    ):
        if marker in template:
            return ToolProtocol(name, chat_template=template)
    if re.search(r"\btools\b", template) and "tool_calls" in template:
        model_type = config.get("text_config", config).get("model_type", "")
        if "python list" in template.lower() or "gemma3" in model_type:
            return ToolProtocol("pythonic", chat_template=template)
        if "llama" in model_type:
            return ToolProtocol("llama_json", chat_template=template)
    return ToolProtocol("json_xml", fallback=True, chat_template=template)


def _types(schema: dict, root: dict, seen: frozenset[str] = frozenset()) -> set[str]:
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/") and ref not in seen:
        target = root
        for part in ref[2:].split("/"):
            target = target.get(part.replace("~1", "/").replace("~0", "~"), {})
            if not isinstance(target, dict):
                return set()
        return _types(target, root, seen | {ref})
    kind = schema.get("type", [])
    kinds = {kind} if isinstance(kind, str) else set(kind) if isinstance(kind, list) else set()
    for key in ("anyOf", "oneOf", "allOf"):
        for variant in schema.get(key, []):
            if isinstance(variant, dict):
                kinds.update(_types(variant, root, seen))
    return kinds


def _argument(value: str, schema: dict, root: dict):
    # Strings are written literally by GLM, including their whitespace, quotes,
    # and strings which happen to look like JSON numbers or booleans.
    kinds = _types(schema, root)
    if "string" in kinds and kinds <= {"string", "null"}:
        if "null" in kinds and value.strip() in ("null", "None"):
            return None
        return value
    if value.strip() in ("True", "False", "None"):
        return {"True": True, "False": False, "None": None}[value.strip()]
    try:
        parsed = json.loads(value)
        # Reject non-JSON NaN/Infinity without changing the raw string value.
        json.dumps(parsed, allow_nan=False)
        return parsed
    except (ValueError, TypeError):
        return value


def _call(name: str, args: dict, functions: dict) -> dict:
    if name not in functions or not isinstance(args, dict):
        raise ValueError("unknown function or non-object arguments")
    return dict(
        id="call_" + uuid.uuid4().hex,
        type="function",
        function=dict(name=name, arguments=json.dumps(args, ensure_ascii=False, allow_nan=False)),
    )


def _json_call(node: dict, functions: dict) -> dict:
    if not isinstance(node, dict):
        raise ValueError("expected a function object")
    args = node.get("arguments", node.get("parameters", {}))
    if isinstance(args, str):
        args = json.loads(args)
    return _call(node.get("name"), args, functions)


def _arg_xml(body: str, functions: dict) -> list[dict]:
    match = re.match(r"\s*([^\s<>]+)\s*", body)
    if match is None or match[1] not in functions:
        raise ValueError("unknown or missing tool name")
    name, offset = match[1], match.end()
    schema = functions[name].get("parameters", {})
    properties = schema.get("properties", {})
    args = {}
    while offset < len(body):
        match = re.match(r"\s*<arg_key>([^<>]+)</arg_key>\s*<arg_value>", body[offset:])
        if match is None:
            raise ValueError("malformed tool argument")
        key = match[1].strip()
        offset += match.end()
        end = body.find("</arg_value>", offset)
        if end < 0 or not key or key in args:
            raise ValueError("incomplete or duplicate tool argument")
        value = body[offset:end]
        args[key] = _argument(value, properties.get(key, {}), schema)
        offset = end + len("</arg_value>")
        while offset < len(body) and body[offset].isspace():
            offset += 1
    return [_call(name, args, functions)]


def _qwen_coder(body: str, functions: dict) -> list[dict]:
    match = re.fullmatch(r"\s*<function=([^<>]+)>(.*)</function>\s*", body, re.S)
    if match is None or match[1] not in functions:
        raise ValueError("malformed function")
    name, body = match[1], match[2]
    schema = functions[name].get("parameters", {})
    args, offset = {}, 0
    while body[offset:].strip():
        match = re.match(r"\s*<parameter=([^<>]+)>(.*?)</parameter>", body[offset:], re.S)
        if match is None or match[1] in args:
            raise ValueError("malformed or duplicate parameter")
        key, value = match[1], match[2]
        # The template adds one framing newline on either side of each value.
        value = value.removeprefix("\n").removesuffix("\n")
        args[key] = _argument(value, schema.get("properties", {}).get(key, {}), schema)
        offset += match.end()
    return [_call(name, args, functions)]


def _minicpm(body: str, functions: dict) -> list[dict]:
    node = ET.fromstring(body)
    name = node.attrib.get("name")
    if node.tag != "function" or name not in functions or (node.text or "").strip():
        raise ValueError("malformed function")
    schema, args = functions[name].get("parameters", {}), {}
    for param in node:
        key = param.attrib.get("name")
        if param.tag != "param" or not key or key in args or len(param) or (param.tail or "").strip():
            raise ValueError("malformed parameter")
        args[key] = _argument(param.text or "", schema.get("properties", {}).get(key, {}), schema)
    return [_call(name, args, functions)]


def _literal(node):
    # LFM templates mix Python scalars with JSON containers. Accept literals
    # only: no eval, variable lookup, attribute access, or nested function calls.
    if isinstance(node, ast.Name) and node.id in ("true", "false", "null"):
        return {"true": True, "false": False, "null": None}[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        keys = [_literal(key) for key in node.keys]
        if any(not isinstance(k, str) for k in keys) or len(set(keys)) != len(keys):
            raise ValueError("invalid object keys")
        return dict(zip(keys, map(_literal, node.values)))
    return ast.literal_eval(node)


def _pythonic(body: str, functions: dict) -> list[dict]:
    root = ast.parse(body.strip(), mode="eval").body
    nodes = root.elts if isinstance(root, ast.List) else [root]
    calls = []
    for node in nodes:
        if not isinstance(node, ast.Call) or node.args:
            raise ValueError("expected named function arguments")
        target, parts = node.func, []
        while isinstance(target, ast.Attribute):
            parts.insert(0, target.attr)
            target = target.value
        if not isinstance(target, ast.Name):
            raise ValueError("invalid function name")
        name = ".".join([target.id] + parts)
        args = {}
        for arg in node.keywords:
            if arg.arg is None or arg.arg in args:
                raise ValueError("invalid keyword argument")
            args[arg.arg] = _literal(arg.value)
        calls.append(_call(name, args, functions))
    if not calls:
        raise ValueError("empty function list")
    return calls


class _GemmaValues:
    """Small recursive reader for Gemma's delimited strings and JSON-like values."""

    def __init__(self, text):
        self.text, self.pos = text, 0

    def ws(self):
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def take(self, marker):
        self.ws()
        if not self.text.startswith(marker, self.pos):
            raise ValueError("malformed Gemma argument")
        self.pos += len(marker)

    def value(self):
        self.ws()
        text, pos = self.text, self.pos
        if text.startswith('<|"|>', pos):
            end = text.find('<|"|>', pos + 5)
            if end < 0:
                raise ValueError("unterminated Gemma string")
            self.pos = end + 5
            return text[pos + 5 : end]
        if pos >= len(text):
            raise ValueError("missing Gemma value")
        if text[pos] in "{[":
            obj = text[pos] == "{"
            end, result = ("}", {}) if obj else ("]", [])
            self.pos += 1
            self.ws()
            if text.startswith(end, self.pos):
                self.pos += 1
                return result
            while True:
                if obj:
                    self.ws()
                    if text.startswith('<|"|>', self.pos):
                        key = self.value()
                    else:
                        match = re.match(r"[^\s:{}\[\],]+", text[self.pos :])
                        if match is None:
                            raise ValueError("missing Gemma key")
                        key = match[0]
                        self.pos += match.end()
                    if key in result:
                        raise ValueError("duplicate Gemma key")
                    self.take(":")
                    result[key] = self.value()
                else:
                    result.append(self.value())
                self.ws()
                if text.startswith(end, self.pos):
                    self.pos += 1
                    return result
                self.take(",")
        match = re.match(r"[^,}\]]+", text[pos:])
        if match is None:
            raise ValueError("invalid Gemma value")
        self.pos += match.end()
        value = match[0].strip()
        if not value:
            raise ValueError("missing Gemma value")
        return _argument(value, {}, {})


def _gemma(body: str, functions: dict) -> list[dict]:
    match = re.match(r"\s*call:([^\s{}]+)", body)
    if match is None:
        raise ValueError("missing Gemma function")
    values = _GemmaValues(body[match.end() :])
    args = values.value()
    values.ws()
    if values.pos != len(values.text):
        raise ValueError("trailing Gemma arguments")
    return [_call(match[1], args, functions)]


def _llama(body: str, functions: dict) -> list[dict]:
    body = body.strip().removeprefix("<|python_tag|>").lstrip()
    decoder, calls = json.JSONDecoder(), []
    while body:
        node, end = decoder.raw_decode(body)
        calls.extend(_json_call(n, functions) for n in (node if isinstance(node, list) else [node]))
        body = body[end:].strip()
        if body:
            if body[0] not in ";,":
                raise ValueError("trailing function text")
            body = body[1:].lstrip()
    if not calls:
        raise ValueError("empty function list")
    return calls


def _harmony(text: str, tools: list[dict], *, prefix: str) -> dict:
    fallback = dict(content=text, reasoning_content="")
    functions = {t["function"]["name"]: t["function"] for t in tools}
    start = prefix.rfind("<|start|>assistant")
    initial = prefix[start:] if start >= 0 else "<|start|>assistant"
    # Normalize token spellings used by different GPT-OSS tokenizer revisions.
    combined = initial + text
    for old, new in (
        ("<|meta_sep|>", "<|channel|>"),
        ("<|im_sep|>", "<|message|>"),
        ("<|im_start|>", "<|start|>"),
        ("<|im_end|>", "<|end|>"),
        ("<|ghissue|>", "<|call|>"),
        ("<|fim_suffix|>", "<|return|>"),
    ):
        combined = combined.replace(old, new)
    content, reasoning, calls = [], [], []
    try:
        for part in combined.split("<|start|>")[1:]:
            header, sep, body = part.partition("<|message|>")
            if not sep or not header.startswith("assistant"):
                raise ValueError("invalid Harmony header")
            end = re.search(r"<\|(end|call|return)\|>", body)
            payload = body[: end.start()] if end else body
            if end and body[end.end() :].strip():
                raise ValueError("trailing Harmony content")
            recipient = re.search(r"\bto=([^\s<]+)", header)
            channel = header.partition("<|channel|>")[2].split()
            if recipient:
                if not end or end[1] != "call":
                    raise ValueError("incomplete Harmony tool call")
                name = recipient[1].removeprefix("functions.")
                calls.append(_call(name, json.loads(payload), functions))
            elif channel and channel[0] == "analysis":
                reasoning.append(payload)
            else:
                content.append(payload)
    except (ValueError, TypeError, KeyError):
        return fallback
    result = dict(content="".join(content), reasoning_content="".join(reasoning))
    if calls:
        result["tool_calls"] = calls
    return result


def _hold_marker_prefix(text: str, marker: str) -> str:
    for length in range(min(len(text), len(marker) - 1), 0, -1):
        if text.endswith(marker[:length]):
            return text[:-length]
    return text


def parse_output(text: str, tools: list[dict], *, protocol: str, prefix: str, final: bool) -> dict:
    """Project raw output onto chat fields, buffering tool bodies until completion.

    Partial projections are prefixes of the final text/reasoning fields. A
    streaming response can emit their suffixes and publish complete tool calls
    at the end, without leaking XML into assistant content. Malformed tool
    regions are returned verbatim as text and never partially executed.
    """
    reasoning = ""
    think_open, think_close = ("<|channel>thought\n", "<channel|>") if protocol == "gemma4" else ("<think>", "</think>")
    initial_thinking = prefix.rfind(think_open) > prefix.rfind(think_close)
    if text.lstrip().startswith(think_open):
        text = text.lstrip()[len(think_open) :]
        initial_thinking = True
    elif not final and think_open.startswith(text.lstrip()):
        return dict(content="", reasoning_content="")
    if initial_thinking:
        end = text.find(think_close)
        if end < 0:
            reasoning = text if final else _hold_marker_prefix(text, think_close)
            return dict(content="", reasoning_content=reasoning)
        reasoning, text = text[:end], text[end + len(think_close) :]
    result = dict(content=text, reasoning_content=reasoning)
    if not tools:
        return result
    opening, closing, parser = {
        "arg_xml": (OPEN, CLOSE, _arg_xml),
        "json_xml": (OPEN, CLOSE, lambda body, f: [_json_call(json.loads(body), f)]),
        "qwen_coder": (OPEN, CLOSE, _qwen_coder),
        "minicpm": ("<function name=", "</function>", _minicpm),
        "lfm2": ("<|tool_call_start|>", "<|tool_call_end|>", _pythonic),
        "gemma4": ("<|tool_call>", "<tool_call|>", _gemma),
        "llama_json": ("", "", _llama),
        "pythonic": ("", "", _pythonic),
    }[protocol]
    functions = {tool["function"]["name"]: tool["function"] for tool in tools}
    if not opening:
        if not final:
            # Bare JSON/Python formats have no unambiguous streaming delimiter.
            result["content"] = ""
            return result
        try:
            result.update(tool_calls=parser(text, functions), content="")
        except (ValueError, TypeError, SyntaxError, AttributeError, RecursionError):
            pass
        return result
    start = text.find(opening)
    if not final:
        result["content"] = text[:start] if start >= 0 else _hold_marker_prefix(text, opening)
        return result
    if start < 0:
        return result
    calls, content, offset = [], [], 0
    try:
        while start >= 0:
            content.append(text[offset:start])
            end = text.find(closing, start + len(opening))
            if end < 0:
                return result
            body = text[start : end + len(closing)] if protocol == "minicpm" else text[start + len(opening) : end]
            calls.extend(parser(body, functions))
            offset = end + len(closing)
            start = text.find(opening, offset)
        content.append(text[offset:])
        # A truncated following call must not turn an earlier complete call
        # into an executable partial response.
        if _hold_marker_prefix(text[offset:], opening) != text[offset:]:
            return result
    except (ValueError, TypeError, AttributeError, SyntaxError, ET.ParseError, RecursionError):
        return result
    result.update(content="".join(content), tool_calls=calls)
    return result
