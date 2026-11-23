# agent-EN-f2 (ENGINE-INKLING lane f2), 2026-11-21.
# Reference driver: runs INSIDE the vLLM image vllm/vllm-openai:inkling-2-mini-x86_64-7cb5dae (CPU only, no card,
# --network none) and reproduces what that vLLM server does with an OpenAI chat request for Inkling-2-Mini, using the
# server's own classes, without the GPU-only ModelConfig:
#   render:   HTTP body -> ChatCompletionRequest (pydantic) -> tool_dicts -> build_chat_params().with_defaults()
#             -> InklingRenderer._render(request.messages, params)         (online_renderer.render_chat / preprocess_chat)
#   parse:    body + prompt ids + output ids + engine finish reason -> parser.adjust_request -> sampling params
#             -> FastIncrementalDetokenizer (stop-terminated: last token not detokenized) -> parser.parse(...)
#             -> assemble_chat_message_fields -> resolve_finish_reason  (serving.chat_completion_full_generator)
#   tokenize: text -> encode(add_special_tokens=False, split_special_tokens=True)   (the renderer's encode_text)
#             and encode(add_special_tokens=False) (specials parsed), decode(ids, skip_special_tokens=False)
# Usage: python3 inkling_ref_driver.py MODE IN.jsonl OUT.jsonl   (one JSON object per line in and out)
import json
import os
import sys
import traceback
from types import SimpleNamespace

os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")

from vllm.tokenizers.registry import get_tokenizer  # noqa: E402

TOK = get_tokenizer("/model")

from vllm.entrypoints.openai.chat_completion.message_assembly import (  # noqa: E402
    assemble_chat_message_fields,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest  # noqa: E402
from vllm.entrypoints.serve.utils.tool_calls_utils import resolve_finish_reason  # noqa: E402
from vllm.parser import ParserManager  # noqa: E402
from vllm.renderers.inkling import InklingRenderer, _HfBackedTmlTokenizer  # noqa: E402
from vllm.renderers.params import merge_kwargs  # noqa: E402
from vllm.v1.engine.detokenizer import FastIncrementalDetokenizer  # noqa: E402

RENDERER = object.__new__(InklingRenderer)
RENDERER._is_inkling2 = True  # config.json model_type inkling2_loop
RENDERER._inkling_tokenizer = _HfBackedTmlTokenizer(TOK, split_special_tokens=True)

PARSER_CLS = ParserManager.get_parser(
    tool_parser_name="inkling",
    reasoning_parser_name="inkling",
    enable_auto_tools=True,
    model_name="/model",
    is_harmony=False,
    tool_strict_level=None,
    tokenizer=TOK,
)


def chat_params(req):
    # online_renderer.render_chat: exclude_tools_when_tool_choice_none defaults to False
    tool_dicts = None if req.tools is None else [tool.model_dump() for tool in req.tools]
    # preprocess_chat: server default_chat_template_kwargs = {} (no --default-chat-template-kwargs)
    default_kwargs = merge_kwargs({}, dict(tools=tool_dicts, tokenize=False))
    return req.build_chat_params(None, "auto", reasoning_effort_map={}).with_defaults(
        default_kwargs,
        default_media_io_kwargs=None,
        default_mm_processor_kwargs=getattr(req, "mm_processor_kwargs", None),
    )


def do_render(item):
    req = ChatCompletionRequest.model_validate(item["body"])
    if not req.messages:
        raise ValueError("messages must be a non-empty list.")
    params = chat_params(req)
    ids = RENDERER._render(req.messages, params)
    return {"prompt_token_ids": [int(t) for t in ids]}


def do_parse(item):
    body = item["body"]
    prompt_ids = item["prompt_token_ids"]
    out_ids = item["output_token_ids"]
    engine_finish = item.get("engine_finish_reason", "stop")
    req = ChatCompletionRequest.model_validate(body)
    # serving._create_chat_completion: the response parser is built first, with the effective kwargs (no tools default)
    eff_kwargs = req.build_chat_params(None, "auto", reasoning_effort_map={}).with_defaults({}).chat_template_kwargs
    parser = PARSER_CLS(TOK, req.tools, chat_template_kwargs=eff_kwargs, model_config=None)
    # preprocess_chat: a second parser instance adjusts the request (skip_special_tokens=False, grammar, ...)
    params = chat_params(req)
    adj = PARSER_CLS(TOK, req.tools, model_config=None, chat_template_kwargs=params.chat_template_kwargs)
    adj.set_prompt_token_ids(prompt_ids)
    req = adj.adjust_request(request=req)
    max_tokens = req.max_completion_tokens if req.max_completion_tokens is not None else (req.max_tokens or 65536)
    sp = req.to_sampling_params(max_tokens, {})
    detok = FastIncrementalDetokenizer(
        TOK, SimpleNamespace(request_id="ref", sampling_params=sp, prompt_token_ids=prompt_ids)
    )
    detok.update(list(out_ids), engine_finish == "stop")
    text = detok.get_next_output_text(finished=True, delta=False)
    parser.set_prompt_token_ids(prompt_ids)
    reasoning, content, tool_calls = parser.parse(
        text, req, enable_auto_tools=True, model_output_token_ids=list(out_ids)
    )
    fields = assemble_chat_message_fields(
        req, reasoning, content, tool_calls, enable_auto_tools=True, parser_cls=PARSER_CLS
    )
    finish = resolve_finish_reason(engine_finish, req, bool(fields.tool_calls))
    return {
        "output_text": text,
        "skip_special_tokens": sp.skip_special_tokens,
        "reasoning": fields.reasoning,
        "content": fields.content,
        "tool_calls": [
            {"name": tc.function.name, "arguments": tc.function.arguments} for tc in fields.tool_calls
        ],
        "finish_reason": finish,
        "reasoning_tokens": parser.count_reasoning_tokens(list(out_ids)),
    }


def do_tokenize(item):
    text = item["text"]
    plain = TOK.encode(text, add_special_tokens=False, split_special_tokens=True)
    with_specials = TOK.encode(text, add_special_tokens=False)
    return {
        "ids_split_special": [int(t) for t in plain],
        "ids_parse_special": [int(t) for t in with_specials],
        "decoded": TOK.decode(plain, skip_special_tokens=False),
    }


MODES = {"render": do_render, "parse": do_parse, "tokenize": do_tokenize}


def main():
    mode, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    fn = MODES[mode]
    n = errors = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            out = {"case": item.get("case")}
            try:
                out.update(fn(item))
            except Exception as e:  # the server answers these with an error response
                out["error"] = {"type": type(e).__name__, "message": str(e)[:2000]}
                if os.environ.get("REF_TRACE"):
                    out["trace"] = traceback.format_exc()[-3000:]
                errors += 1
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    print(f"{mode}: {n} items, {errors} errors", file=sys.stderr)


if __name__ == "__main__":
    main()
