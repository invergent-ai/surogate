"""Check (or rewrite) the vLLM results in corpus.jsonl (consumed by test_vllm_qwen3_parser.cpp).

Each row's `vllm` field is what vLLM 0.31.0 returns for its output: the parser the OpenAI-compatible server builds for
`--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder` (qwen3_xml is the same class),
called as chat_completion/serving.py calls it for a non-streaming request: parse(output.text, request,
enable_auto_tools=True, model_output_token_ids=output.token_ids). `text` is output.text (the token ids decoded with
their special tokens, the stop token excluded); `ids` are output.token_ids. A lone surrogate in a result (an argument
value read from a \\u escape) is stored as the escape text the engine writes. Run where vLLM 0.31.0 is installed (no
GPU needed):

    CUDA_VISIBLE_DEVICES= python vllm_reference.py TOKENIZER_DIR corpus.jsonl [--write]

The rows were drawn from a larger corpus (rl/engine-goal-2026-10-19/toolcall on the project disk: 27,667 outputs,
all identical): Surogate3.7-35B-A3B NVFP4 outputs on BFCL v4 (our engine's text answers, the calls vLLM returned,
rebuilt as the model writes them), the tool-argument cases of fixtures/serve/vllm_tool_arguments, and mutations of
them (unknown names, missing closers, text after and between calls, calls inside reasoning, thinking off,
tool_choice none, no tools, special tokens inside, markers spelled by ordinary tokens, bare <function=, JSON bodies,
multibyte text, every token prefix). Without --write, rows whose stored result differs are listed and the exit
code is 1.
"""
import json
import re
import sys

LONE = re.compile("[\ud800-\udfff]")


def wtf(value):
    if isinstance(value, str):
        return LONE.sub(lambda m: "\\u%04x" % ord(m.group()), value)
    if isinstance(value, list):
        return [wtf(v) for v in value]
    if isinstance(value, dict):
        return {k: wtf(v) for k, v in value.items()}
    return value


def main(tok_dir, path, *flags):
    from transformers import AutoTokenizer
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser.parser_manager import ParserManager

    tok = AutoTokenizer.from_pretrained(tok_dir)
    cls = ParserManager.get_parser(tool_parser_name="qwen3_coder", reasoning_parser_name="qwen3",
                                   enable_auto_tools=True)
    lines = [line for line in open(path, encoding="utf-8").read().split("\n") if line]
    header, rows = json.loads(lines[0]), [json.loads(line) for line in lines[1:]]
    differ = 0
    for r in rows:
        kw = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
        if r["tools"]:
            kw["tools"] = r["tools"]
        if r["tool_choice"]:
            kw["tool_choice"] = r["tool_choice"]
        req = ChatCompletionRequest(**kw)
        p = cls(tok, req.tools, chat_template_kwargs={"enable_thinking": r["thinking"]}, model_config=None)
        p.adjust_request(req)
        reasoning, content, calls = p.parse(r["text"], req, enable_auto_tools=True, model_output_token_ids=r["ids"])
        got = wtf({"reasoning": reasoning, "content": content,
                   "calls": [{"name": c.name, "arguments": c.arguments} for c in calls] if calls else []})
        if got != r["vllm"]:
            differ += 1
            print(f"{r['id']}: stored {json.dumps(r['vllm'])[:300]}\n    vLLM  {json.dumps(got)[:300]}")
            r["vllm"] = got
    print(f"{len(rows)} rows, {differ} differ from the stored vLLM results", file=sys.stderr)
    if "--write" in flags:
        with open(path, "w", encoding="utf-8") as fo:
            for row in [header] + rows:
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
        return 0
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
