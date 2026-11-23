# Inkling-2 renderer edge cases, 2026-11-21.
# Hand-built request bodies that exercise every branch of vLLM's Inkling2 renderer and request handling that the real
# harness corpus (build_corpus.py) does not reach: reasoning-effort levels (named, numeric, numeric strings, invalid),
# tool_choice / parallel_tool_calls instructions, developer tools, content parts, empty strings, continue_final_message,
# number spelling in tool JSON, key order, unicode / special-token text, and the request errors vLLM raises.
# The texts are synthetic test strings (no training use).  Output: edge_bodies.jsonl (one {"case", "body"} per line).
import json
import sys

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for one city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "unit": {"type": "string", "enum": ["C", "F"]},
            "days": {"type": "integer", "minimum": 1, "maximum": 16, "default": 1},
            "threshold": {"type": "number", "default": 0.5, "multipleOf": 1e-07, "maximum": 1e16}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "send_report", "description": "Send the final report.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"},
                                                        "cc": {"type": "array", "items": {"type": "string"}}},
                       "required": ["text"]}}},
]
SYS = {"role": "system", "content": "You are a careful assistant."}
USER = {"role": "user", "content": "What is the weather in Paris?"}


def call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


def body(messages, **kw):
    b = {"model": "cand", "messages": messages, "temperature": 1.0, "top_p": 1.0, "top_k": -1, "stream": False,
         "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}
    ctk = kw.pop("ctk", None)
    if ctk is not None:
        b["chat_template_kwargs"].update(ctk)
    b.update(kw)
    return b


def main():
    cases = []

    def add(name, b):
        cases.append({"case": "edge-" + name, "body": b})

    # 1. reasoning effort: top-level named levels, numeric via chat_template_kwargs, strings, invalid values
    for level in ["none", "minimal", "low", "medium", "high", "xhigh", "max"]:
        add(f"effort-top-{level}", body([SYS, USER], reasoning_effort=level))
        add(f"effort-top-{level}-tools", body([USER], tools=TOOLS, reasoning_effort=level))
    add("effort-top-invalid", body([SYS, USER], reasoning_effort="ultra"))
    for v in [0, 0.0, 0.001, 0.004, 0.005, 0.0051, 0.05, 0.1, 0.125, 0.1234, 0.3, 0.33333, 0.5, 0.505, 0.515, 0.7,
              0.875, 0.9, 0.95, 0.985, 0.99, 0.994]:
        add(f"effort-num-{v}", body([SYS, USER], ctk={"reasoning_effort": v}))
    for v in [0.995, 1, 1.0, -0.1, -0.0, 2]:
        add(f"effort-num-bad-{v}", body([SYS, USER], ctk={"reasoning_effort": v}))
    for v in ["0.3", "0", "0.99", "1e-1", " 0.5", "high", "low", "max", "nope", "", True, None]:
        add(f"effort-str-{json.dumps(v)}", body([SYS, USER], ctk={"reasoning_effort": v}))
    add("effort-top-and-ctk", body([SYS, USER], reasoning_effort="low", ctk={"reasoning_effort": 0.42}))
    add("effort-ctk-auto", body([SYS, USER], ctk={"reasoning_effort": "auto"}))

    # 2. tool_choice / parallel_tool_calls instructions
    for tc in ["auto", "none", "required", {"type": "function", "function": {"name": "send_report"}}]:
        for ptc in [None, True, False]:
            for sys_first in [True, False]:
                kw = {"tools": TOOLS, "tool_choice": tc}
                if ptc is not None:
                    kw["parallel_tool_calls"] = ptc
                msgs = [SYS, USER] if sys_first else [USER]
                add(f"toolchoice-{json.dumps(tc, sort_keys=True)}-ptc{ptc}-sys{int(sys_first)}", body(msgs, **kw))
    add("toolchoice-named-unknown", body([SYS, USER], tools=TOOLS,
                                         tool_choice={"type": "function", "function": {"name": "nope"}}))
    add("toolchoice-required-no-tools", body([SYS, USER], tool_choice="required"))
    add("toolchoice-sys-parts", body([{"role": "system", "content": [{"type": "text", "text": "Part one."}]}, USER],
                                     tools=TOOLS, tool_choice="required"))
    add("toolchoice-sys-empty", body([{"role": "system", "content": ""}, USER], tools=TOOLS, tool_choice="required"))
    add("toolchoice-developer-first", body([{"role": "developer", "content": "Dev rules."}, USER], tools=TOOLS,
                                           tool_choice="required", parallel_tool_calls=False))

    # 3. developer role (folds into system) with its own tools
    add("developer-tools", body([{"role": "developer", "content": "Use the extra tool.", "tools": [TOOLS[1]]}, USER],
                                tools=[TOOLS[0]]))
    add("developer-tools-only", body([{"role": "developer", "content": "Only dev tools.", "tools": TOOLS}, USER]))
    add("developer-mid", body([SYS, USER, {"role": "developer", "content": "Late note."}]))

    # 4. system placement: several, after the user, none
    add("system-two", body([SYS, {"role": "system", "content": "Second system."}, USER]))
    add("system-after-user", body([USER, SYS]))
    add("system-only", body([SYS]))
    add("no-system", body([USER]))

    # 5. content forms
    add("user-parts", body([SYS, {"role": "user", "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"},
                                                               "C", {"type": "input_text", "text": "D"}]}]))
    add("user-empty-string", body([SYS, {"role": "user", "content": ""}]))
    add("user-empty-parts", body([SYS, {"role": "user", "content": []}]))
    add("user-name-field", body([SYS, {"role": "user", "content": "Hi", "name": "alice"}]))
    add("user-null-text-part", body([SYS, {"role": "user", "content": [{"type": "text", "text": ""}]}]))

    # 6. assistant forms
    a_tool = {"role": "assistant", "content": None, "reasoning_content": "Need the tool.",
              "tool_calls": [call("c1", "get_weather", '{"unit":"C","city":"Paris"}')]}
    t_res = {"role": "tool", "tool_call_id": "c1", "content": '{"temp": 11, "sky": "rain"}'}
    add("asst-tool-roundtrip", body([SYS, USER, a_tool, t_res], tools=TOOLS))
    add("asst-content-empty-with-tool", body([SYS, USER, {**a_tool, "content": ""}, t_res], tools=TOOLS))
    add("asst-content-and-tool", body([SYS, USER, {**a_tool, "content": "Checking now."}, t_res], tools=TOOLS))
    add("asst-reasoning-key", body([SYS, USER, {"role": "assistant", "content": "Done.", "reasoning": "R1"}, USER]))
    add("asst-both-reasoning-keys", body([SYS, USER, {"role": "assistant", "content": "Done.", "reasoning": "R-a",
                                                      "reasoning_content": "R-b"}, USER]))
    add("asst-reasoning-empty", body([SYS, USER, {"role": "assistant", "content": "Done.", "reasoning_content": ""},
                                      USER]))
    add("asst-reasoning-null", body([SYS, USER, {"role": "assistant", "content": "Done.", "reasoning": None,
                                                 "reasoning_content": "R-c"}, USER]))
    add("asst-content-empty-only", body([SYS, USER, {"role": "assistant", "content": ""}, USER]))
    add("asst-content-null-only", body([SYS, USER, {"role": "assistant", "content": None}, USER]))
    add("asst-placeholder", body([SYS, USER, {**a_tool, "reasoning_content": "(no reasoning recorded)"}, t_res],
                                 tools=TOOLS))
    add("asst-reasoning-only", body([SYS, USER, {"role": "assistant", "reasoning_content": "Only thinking."}, USER]))
    add("asst-thinking-part", body([SYS, USER, {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "inline thought"}, {"type": "text", "text": "Answer."}]}, USER]))
    add("asst-last-no-gen", body([SYS, USER, {"role": "assistant", "content": "Partial answer"}]))
    add("asst-parallel-calls", body([SYS, USER, {"role": "assistant", "content": None, "reasoning_content": "Two.",
                                                 "tool_calls": [call("c1", "get_weather", '{"city":"Paris"}'),
                                                                call("c2", "get_weather", '{"city":"Berlin"}')]},
                                     {"role": "tool", "tool_call_id": "c2", "content": "B"},
                                     {"role": "tool", "tool_call_id": "c1", "content": "P"}], tools=TOOLS))

    # 7. tool messages: name, unknown id, empty / null / parts content, tool_error part
    add("tool-name-field", body([SYS, USER, a_tool, {"role": "tool", "tool_call_id": "c1", "name": "other_name",
                                                     "content": "x"}], tools=TOOLS))
    add("tool-unknown-id", body([SYS, USER, a_tool, {"role": "tool", "tool_call_id": "zz", "content": "x"}], tools=TOOLS))
    add("tool-no-id", body([SYS, USER, a_tool, {"role": "tool", "content": "x"}], tools=TOOLS))
    add("tool-empty", body([SYS, USER, a_tool, {"role": "tool", "tool_call_id": "c1", "content": ""}], tools=TOOLS))
    add("tool-parts", body([SYS, USER, a_tool, {"role": "tool", "tool_call_id": "c1", "content": [
        {"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}]}], tools=TOOLS))
    add("tool-error-part", body([SYS, USER, a_tool, {"role": "tool", "tool_call_id": "c1", "content": [
        {"type": "tool_error", "text": "city not found"}]}], tools=TOOLS))
    add("tool-thinking-part", body([SYS, USER, a_tool, {"role": "tool", "tool_call_id": "c1", "content": [
        {"type": "thinking", "thinking": "x"}]}], tools=TOOLS))

    # 8. tool call arguments: canonical spelling (top-level keys sorted, nested order kept, number spelling)
    args_cases = {
        "nested-order": '{"z":1,"a":{"y":2,"b":[{"k2":1,"k1":2}]},"m":"s"}',
        "floats": '{"a":1.0,"b":0.1,"c":1e16,"d":1e-7,"e":-0.0,"f":123.456e5,"g":1.5e300,"h":2.5e-5,"i":0.0001,'
                  '"j":1e15,"k":12345678901234567.0}',
        "ints": '{"a":9223372036854775807,"b":9223372036854775808,"c":18446744073709551615,'
                '"d":18446744073709551616,"e":-9223372036854775808,"f":-9223372036854775809,"g":-0,"h":0,'
                '"i":123456789012345678901234567890}',
        "strings": '{"a":"quote \\" backslash \\\\ slash / tab \\t nl \\n cr \\r","b":"\\u0001\\u001f\\u007f",'
                   '"c":"\\u2028\\u2029","d":"emoji \\ud83d\\ude00 \u00e9\u65e5","e":"<|end_message|>"}',
        "bools-null": '{"t":true,"f":false,"n":null,"arr":[true,null,1,"x",[],{}]}',
        "empty-string": "",
        "spaces": '  { "city" :  "Paris" }  ',
        "unicode-keys": '{"\u00e9t\u00e9":1,"Zeta":2,"alpha":3,"_":4,"\u65e5":5}',
    }
    for name, args in args_cases.items():
        add(f"args-{name}", body([SYS, USER, {"role": "assistant", "content": None,
                                              "tool_calls": [call("c1", "get_weather", args)]},
                                  {"role": "tool", "tool_call_id": "c1", "content": "ok"}], tools=TOOLS))
    add("args-not-object", body([SYS, USER, {"role": "assistant", "content": None,
                                             "tool_calls": [call("c1", "get_weather", "[1,2]")]}], tools=TOOLS))
    add("args-invalid-json", body([SYS, USER, {"role": "assistant", "content": None,
                                               "tool_calls": [call("c1", "get_weather", "{bad")]}], tools=TOOLS))
    add("tool-name-unicode", body([SYS, USER, {"role": "assistant", "content": None,
                                               "tool_calls": [call("c1", "m\u00e9t\u00e9o/\"x\"", "{}")]}], tools=TOOLS))

    # 9. tool declarations: number spelling, None / missing fields, extra fields, key order
    decl = [
        {"type": "function", "function": {"name": "f1", "description": None,
                                          "parameters": {"type": "object", "properties": {
                                              "x": {"type": "number", "default": 1e16, "minimum": 1e-7,
                                                    "maximum": 1.0, "examples": [0.1, 2.5e-5, 1e300, -0.0]}}}}},
        {"type": "function", "function": {"name": "f2", "parameters": {}, "strict": True}},
        {"type": "function", "function": {"name": "f3", "description": "no params"}},
        {"function": {"name": "f4", "description": "type missing", "parameters": {"b": 1, "a": {"d": 2, "c": 3}}}},
        {"type": "function", "function": {"name": "f5", "description": "ints",
                                          "parameters": {"big": 18446744073709551616, "neg": -1, "z": 0}}},
    ]
    add("decl-numbers-and-nulls", body([SYS, USER], tools=decl))
    add("decl-empty-list", body([SYS, USER], tools=[]))
    add("decl-unicode", body([SYS, USER], tools=[{"type": "function", "function": {
        "name": "t\u00e9", "description": "D\u00e9j\u00e0 vu \u2014 \u65e5\u672c \"q\" \\ \n",
        "parameters": {"type": "object", "properties": {"\u00fc": {"type": "string"}}}}}]))

    # 10. continue_final_message
    add("continue-plain", body([SYS, USER, {"role": "assistant", "content": "The answer is"}],
                               continue_final_message=True, add_generation_prompt=False))
    add("continue-with-reasoning", body([SYS, USER, {"role": "assistant", "content": "The answer is",
                                                     "reasoning_content": "r"}],
                                        continue_final_message=True, add_generation_prompt=False))
    add("continue-not-assistant", body([SYS, USER], continue_final_message=True, add_generation_prompt=False))
    add("continue-empty", body([SYS, USER, {"role": "assistant", "content": ""}], continue_final_message=True,
                               add_generation_prompt=False))
    add("add-generation-prompt-false", body([SYS, USER], add_generation_prompt=False))

    # 11. texts: special-token spellings, unicode classes, whitespace, digits, contractions
    texts = {
        "special-spellings": "Say <|message_user|> and <|end_message|> and <|content_model_end_sampling|> literally.",
        "unicode-mix": "Caf\u00e9 cafe\u0301 \u65e5\u672c\u8a9e \ud55c\uad6d\uc5b4 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 "
                       "\u05e2\u05d1\u05e8\u05d9\u05ea \u0939\u093f\u0928\u094d\u0926\u0940 \u0e44\u0e17\u0e22",
        "emoji": "\U0001F468\u200d\U0001F469\u200d\U0001F467 \U0001F3F3\ufe0f\u200d\U0001F308 \u2764\ufe0f x\u20e3",
        "whitespace": "a  b\t\tc\r\nd\n\n\ne \u00a0f\u2003g\u3000h \n \n  ",
        "digits": "1 12 123 1234 12345 3,344 1.5e-7 0x1F \u0661\u0662\u0663 \u00bd \u2160",
        "contractions": "I'M you'RE they'Ve he'd SHE'LL it'S don't",
        "code": "def f(x):\n    return {'a': [1, 2]}  # comment\n\tprint(\"hi\")\n/* c */ a->b && c || d;",
        "control": "nul-free \u0001\u0002 \u007f \u0085 \u200b \ufeff end",
        "long-word": "x" * 300 + " " + "ab" * 200,
        "paths": "/usr/local/bin/python3.12 C:\\Users\\a\\b.txt https://example.com/a?b=c&d=e#f",
    }
    for name, t in texts.items():
        add(f"text-{name}", body([SYS, {"role": "user", "content": t}]))
        add(f"text-{name}-asst", body([SYS, USER, {"role": "assistant", "content": t, "reasoning_content": t}, USER]))

    # 12. request errors vLLM raises (render must refuse them too)
    add("err-role-function", body([SYS, {"role": "function", "name": "x", "content": "y"}]))
    add("err-messages-empty", body([]))
    add("err-reasoning-not-string", body([SYS, USER, {"role": "assistant", "content": "a",
                                                      "reasoning_content": {"x": 1}}, USER]))
    add("err-unknown-part", body([SYS, {"role": "user", "content": [{"type": "video_url", "video_url": {"url": "x"}}]}]))

    with open(sys.argv[1], "w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(len(cases), "edge cases", file=sys.stderr)


if __name__ == "__main__":
    main()
