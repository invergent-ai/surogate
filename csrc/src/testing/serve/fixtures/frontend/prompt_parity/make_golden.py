"""Regenerate the expected prompts in cases.json (consumed by family/test_prompt_parity.cpp).

Every case is a Chat Completions request body; its expected prompt, per template, is the text vLLM's
OpenAI-compatible server renders for it -- vLLM's own ChatCompletionRequest validation, tool dicts
(`tool.model_dump()`), parse_chat_messages and safe_apply_chat_template -- so run this where vLLM and
transformers are installed (no GPU needed):

    CUDA_VISIBLE_DEVICES= python make_golden.py TOKENIZER_DIR

TOKENIZER_DIR is any Qwen3.5-family tokenizer (only its chat-template machinery is used; the
templates are the .jinja files named in cases.json). A case whose expected value is {"error": ...}
is one vLLM refuses; the engine must refuse it too. The case requests came from
scripts/serve_prompt_parity.py's synthetic set and a few BFCL / EnterpriseOps-Gym requests.
"""
import json
import os
import sys
import types

import transformers
import vllm
from transformers import AutoTokenizer
from vllm.entrypoints.chat_utils import parse_chat_messages
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.renderers.hf import _detect_content_format, safe_apply_chat_template

here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, "cases.json")
golden = json.load(open(path))
tok_dir = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(tok_dir)
config = types.SimpleNamespace(multimodal_config=None, enable_prompt_embeds=False, hf_config=None, tokenizer=tok_dir,
                               model=tok_dir, tokenizer_revision=None, code_revision=None, trust_remote_code=False,
                               max_model_len=1 << 22)

for name, file in golden["templates"].items():
    template = open(os.path.join(here, file)).read()
    fmt = _detect_content_format(template, default="string")
    for case in golden["cases"]:
        request = ChatCompletionRequest.model_validate(json.loads(json.dumps(case["request"])))
        tools = None if request.tools is None else [t.model_dump() for t in request.tools]
        conversation, _, _ = parse_chat_messages(request.messages, config, fmt)
        kwargs = request.build_chat_params(template, fmt).get_apply_chat_template_kwargs()
        kwargs["chat_template"] = template
        try:
            case["expected"][name] = safe_apply_chat_template(config, tokenizer, conversation, tools=tools,
                                                              tokenize=False, **kwargs)
        except Exception as e:  # noqa: BLE001 - the template refuses the request
            case["expected"][name] = {"error": str(e)[:300]}

golden["generator"] = (f"vLLM {vllm.__version__} OpenAI chat preprocessing (ChatCompletionRequest -> tools model_dump "
                       f"-> parse_chat_messages -> safe_apply_chat_template), transformers {transformers.__version__}; "
                       "see make_golden.py")
with open(path, "w") as fh:
    json.dump(golden, fh, indent=1, ensure_ascii=False)
print(len(golden["cases"]), "cases x", len(golden["templates"]), "templates", file=sys.stderr)
