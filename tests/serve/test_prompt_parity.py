"""Prompt-render parity with vLLM's OpenAI-compatible server.

* The golden prompts the C++ test checks (csrc/src/testing/serve/fixtures/frontend/prompt_parity/cases.json,
  rendered by vLLM itself) are also what transformers renders under the vLLM preprocessing that
  scripts/serve_prompt_parity.py reproduces -- so the reproduction used to check live engines is the
  right one. Needs transformers and jinja2; no GPU, no checkpoint.
* With a running engine (SUROGATE_PROMPT_PARITY_URL, optional SUROGATE_PROMPT_PARITY_KEY_FILE) and the
  model's HF tokenizer directory (SUROGATE_PROMPT_PARITY_TOKENIZER), every synthetic edge-case request
  must tokenize to the same ids on the engine's /tokenize as vLLM would prefill.
"""
import importlib.util
import json
import os
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "csrc/src/testing/serve/fixtures/frontend/prompt_parity"


def _tool():
    spec = importlib.util.spec_from_file_location("serve_prompt_parity", ROOT / "scripts/serve_prompt_parity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render(template, body, tool):
    from transformers.utils.chat_template_utils import render_jinja_template

    kwargs = dict(body.get("chat_template_kwargs") or {})
    rendered = render_jinja_template(conversations=[tool.vllm_conversation(body["messages"], "openai", template)],
                                     tools=tool.vllm_tools(body.get("tools")), chat_template=template,
                                     add_generation_prompt=body.get("add_generation_prompt", True), **kwargs)
    while isinstance(rendered, (tuple, list)):  # (prompts, generation indices) / one prompt per conversation
        rendered = rendered[0]
    return rendered


def _golden():
    return json.loads((FIXTURES / "cases.json").read_text())


@pytest.mark.parametrize("template", sorted(_golden()["templates"]))
def test_golden_prompts_are_the_vllm_preprocessing(template):
    pytest.importorskip("transformers")
    tool = _tool()
    golden = _golden()
    source = (FIXTURES / golden["templates"][template]).read_text()
    mismatched = []
    for case in golden["cases"]:
        expected = case["expected"][template]
        if isinstance(expected, dict):
            with pytest.raises(Exception):
                _render(source, case["request"], tool)
            continue
        if _render(source, case["request"], tool) != expected:
            mismatched.append(case["name"])
    assert not mismatched, f"transformers + the vLLM reproduction disagree with vLLM's golden prompt: {mismatched}"


@pytest.mark.skipif(not (os.getenv("SUROGATE_PROMPT_PARITY_URL") and os.getenv("SUROGATE_PROMPT_PARITY_TOKENIZER")),
                    reason="needs a running engine (SUROGATE_PROMPT_PARITY_URL) and its HF tokenizer directory")
def test_engine_tokenize_matches_vllm_rendering():
    from transformers import AutoTokenizer

    tool = _tool()
    tokenizer_dir = os.environ["SUROGATE_PROMPT_PARITY_TOKENIZER"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    template = Path(os.getenv("SUROGATE_PROMPT_PARITY_TEMPLATE", f"{tokenizer_dir}/chat_template.jinja")).read_text()
    headers = {"Content-Type": "application/json"}
    if os.getenv("SUROGATE_PROMPT_PARITY_KEY_FILE"):
        headers["Authorization"] = "Bearer " + Path(os.environ["SUROGATE_PROMPT_PARITY_KEY_FILE"]).read_text().strip()
    differing = []
    for item in tool.synthetic_requests():
        expected = tool.hf_reference_tokens(tokenizer, template, item["body"])
        request = urllib.request.Request(os.environ["SUROGATE_PROMPT_PARITY_URL"].rstrip("/") + "/tokenize",
                                         data=json.dumps(item["body"]).encode(), headers=headers)
        tokens = json.loads(urllib.request.urlopen(request, timeout=120).read())["tokens"]
        if tokens != expected:
            differing.append(item["id"])
    assert not differing, f"engine prompt differs from vLLM's for {differing}"
