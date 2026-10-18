"""Check (or rewrite) the vLLM results in parity.jsonl (consumed by test_vllm_tool_arguments.cpp).

Every case's `vllm` field is what vLLM's own parser returns for its raw model output: the parser the OpenAI-compatible
server builds for `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder` (qwen3_xml is
the same class in vLLM 0.31), called as chat_completion/serving.py calls it. Run this where vLLM 0.31.0 is installed
(no GPU needed):

    python vllm_reference.py TOKENIZER_DIR parity.jsonl [--write]

TOKENIZER_DIR is the served model's tokenizer (Surogate 3.7 / any Qwen3.5-family tokenizer: `<tool_call>` and
`</think>` must be its special tokens). The output is fed to vLLM as "ok\n</think>\n\n" + text, since the model writes
it after its thinking block. Without --write, cases whose stored result differs are listed and the exit code is 1.

The cases: calls the model made on BFCL, IFEval and EnterpriseOps-Gym (rebuilt as the model writes them: string
values as written, others as the template's tojson), synthetic values under every schema shape, and markup edge
cases. A case with `expect` is one where the engine deliberately differs; `exception` says why.
"""
import json
import sys


def case_input(header, c):
    if "schema" in c:
        text = f"<tool_call>\n<function=f>\n<parameter=v>\n{c['value']}\n</parameter>\n</function>\n</tool_call>"
        tools = [{"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {"v": header["schemas"][c["schema"]]}, "$defs": header["defs"]}}}]
        return text, tools
    return c["text"], c["tools"]


def main(tok_dir, path, *flags):
    from transformers import AutoTokenizer
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser.parser_manager import ParserManager

    tok = AutoTokenizer.from_pretrained(tok_dir)
    parser_cls = ParserManager.get_parser(tool_parser_name="qwen3_coder", reasoning_parser_name="qwen3",
                                          enable_auto_tools=True)
    lines = [json.loads(line) for line in open(path)]
    header, cases = lines[0], lines[1:]
    differ = 0
    for c in cases:
        text, tools = case_input(header, c)
        try:
            req = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "x"}], tools=tools)
            ids = tok.encode("ok\n</think>\n\n" + text, add_special_tokens=False)
            parser = parser_cls(tok, req.tools, chat_template_kwargs={"enable_thinking": True}, model_config=None)
            parser.adjust_request(req)
            _, _, calls = parser.parse(tok.decode(ids, skip_special_tokens=False), req, enable_auto_tools=True,
                                       model_output_token_ids=ids)
            calls = [{"name": x.name, "arguments": x.arguments} for x in calls or []]
        except Exception as e:  # vLLM refuses the request (e.g. an empty tools array)
            calls = None
            print(f"{c['id']}: vLLM refuses: {e}", file=sys.stderr)
        result = (calls[0]["arguments"] if calls else None) if "schema" in c else calls
        stored = json.loads(c["vllm_ascii"]) if "vllm_ascii" in c else c.get("vllm")
        if result != stored:
            differ += 1
            print(f"{c['id']}: stored {json.dumps(stored)[:300]}\n    vLLM  {json.dumps(result)[:300]}")
            c.pop("vllm_ascii", None)
            c["vllm"] = result
        try:  # a lone surrogate (from a \ud800 escape) cannot be read by the C++ test's JSON reader
            json.dumps(c.get("vllm"), ensure_ascii=False).encode("utf-8")
        except UnicodeEncodeError:
            c["vllm_ascii"] = json.dumps(c.pop("vllm"))
    print(f"{len(cases)} cases, {differ} differ from the stored vLLM results", file=sys.stderr)
    if "--write" in flags:
        with open(path, "w") as fo:
            for row in [header] + cases:
                fo.write(json.dumps(row) + "\n")
        return 0
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
