"""Native tool protocols, HTTP streaming, and Verifiers' token-preserving loop."""

import asyncio
import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests
import torch

from examples.sft.glm.create_dummy import dummy_config, dummy_tokenizer
from surogate.grpo.shared_model import SharedModelServer
from surogate.grpo.tool_protocol import ToolProtocol, request_tools, select_protocol, template_messages
from tests.grpo.shared_model_configs import configurations

TOOLS = [
    dict(
        type="function",
        function=dict(
            name="add",
            description="Add two integers.",
            parameters=dict(
                type="object", properties=dict(a=dict(type="integer"), b=dict(type="integer")), required=["a", "b"]
            ),
        ),
    )
]
CALLS = {
    "json_xml": '<tool_call>{"name":"add","arguments":{"a":2,"b":3}}</tool_call>',
    "arg_xml": "<tool_call>add<arg_key>a</arg_key><arg_value>2</arg_value><arg_key>b</arg_key><arg_value>3</arg_value></tool_call>",
    "qwen_coder": "<tool_call>\n<function=add>\n<parameter=a>\n2\n</parameter>\n<parameter=b>\n3\n</parameter>\n</function>\n</tool_call>",
    "minicpm": '<function name="add"><param name="a">2</param><param name="b">3</param></function>',
    "lfm2": "<|tool_call_start|>[add(a=2, b=3)]<|tool_call_end|>",
    "gemma4": "<|tool_call>call:add{a:2,b:3}<tool_call|>",
    "llama_json": '<|python_tag|>{"name":"add","parameters":{"a":2,"b":3}}',
    "pythonic": "[add(a=2,b=3)]",
    "harmony": ' to=functions.add<|channel|>commentary json<|message|>{"a":2,"b":3}<|call|>',
}


@pytest.mark.parametrize("name", CALLS)
def test_native_formats_and_streamed_fields(name):
    protocol, raw = ToolProtocol(name), CALLS[name]
    prefix = "<|start|>assistant" if name == "harmony" else ""
    result = protocol.parse(raw, TOOLS, prefix=prefix, final=True)
    assert result["content"] == ""
    assert result["tool_calls"][0]["function"]["name"] == "add"
    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == dict(a=2, b=3)
    assert result["tool_calls"][0]["id"].startswith("call_")
    previous = dict(content="", reasoning_content="")
    for end in range(1, len(raw)):
        partial = protocol.parse(raw[:end], TOOLS, prefix=prefix, final=False)
        assert not partial.get("tool_calls")
        for field in previous:
            assert partial[field].startswith(previous[field])
            assert result[field].startswith(partial[field])
        previous = {k: partial[k] for k in previous}


@pytest.mark.parametrize("name", CALLS)
def test_unknown_and_incomplete_calls_remain_text(name):
    protocol, raw = ToolProtocol(name), CALLS[name]
    for bad in (raw.replace("add", "unknown"), raw[:-2]):
        result = protocol.parse(bad, TOOLS, prefix="", final=True)
        assert not result.get("tool_calls")
        assert result["content"] == bad


@pytest.mark.parametrize("name", ["json_xml", "arg_xml", "qwen_coder", "minicpm", "lfm2", "gemma4"])
def test_parallel_calls_and_reasoning_do_not_execute_thoughts(name):
    opening, closing = ("<|channel>thought\n", "<channel|>") if name == "gemma4" else ("<think>", "</think>")
    raw = CALLS[name]
    protocol = ToolProtocol(name)
    text = "consider " + raw + closing + "before" + raw + raw + "after"
    result = protocol.parse(text, TOOLS, prefix="assistant" + opening, final=True)
    assert result["reasoning_content"] == "consider " + raw
    assert result["content"] == "beforeafter"
    assert len(result["tool_calls"]) == 2
    malformed = protocol.parse(text[: -len("after")] + raw[:-1], TOOLS, prefix=opening, final=True)
    assert not malformed.get("tool_calls")


@pytest.mark.parametrize(
    "name,raw",
    [
        ("arg_xml", "<tool_call>echo<arg_key>text</arg_key><arg_value>  007\n</arg_value></tool_call>"),
        ("qwen_coder", "<tool_call><function=echo><parameter=text>\n  007\n\n</parameter></function></tool_call>"),
        ("minicpm", '<function name="echo"><param name="text"><![CDATA[  007\n]]></param></function>'),
        ("gemma4", '<|tool_call>call:echo{text:<|"|>  007\n<|"|>}<tool_call|>'),
        ("lfm2", "<|tool_call_start|>[echo(text='  007\\n')]<|tool_call_end|>"),
    ],
)
def test_string_values_are_not_coerced_or_stripped(name, raw):
    schema = dict(
        type="object", properties=dict(text={"$ref": "#/$defs/Text"}), **{"$defs": {"Text": {"type": "string"}}}
    )
    tools = [dict(type="function", function=dict(name="echo", parameters=schema))]
    result = ToolProtocol(name).parse(raw, tools, prefix="", final=True)
    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"text": "  007\n"}


def test_nested_gemma_and_python_literals_are_data_only():
    tools = [dict(type="function", function=dict(name="echo"))]
    cases = [
        ("gemma4", '<|tool_call>call:echo{nested:{s:<|"|>quote " },: [\n<|"|>,n:2},xs:[true,null,3.5]}<tool_call|>'),
        (
            "lfm2",
            """<|tool_call_start|>[echo(nested={'s': 'quote " },: [\\n', 'n': 2},xs=[true,null,3.5])]<|tool_call_end|>""",
        ),
    ]
    for name, raw in cases:
        result = ToolProtocol(name).parse(raw, tools, prefix="", final=True)
        args = json.loads(result["tool_calls"][0]["function"]["arguments"])
        assert args == dict(nested=dict(s='quote " },: [\n', n=2), xs=[True, None, 3.5])
    for raw in ("echo(a=__import__('os').getcwd())", "echo(**{'a': 1})", "echo(a=1,a=2)"):
        result = ToolProtocol("pythonic").parse(raw, tools, prefix="", final=True)
        assert result["content"] == raw and not result.get("tool_calls")


def test_harmony_analysis_tool_handoff_and_final_answer():
    protocol = ToolProtocol("harmony")
    raw = "<|channel|>analysis<|message|>compute<|end|><|start|>assistant" + CALLS["harmony"]
    result = protocol.parse(raw, TOOLS, prefix="<|start|>assistant", final=True)
    assert result["reasoning_content"] == "compute"
    assert len(result["tool_calls"]) == 1
    final = "<|channel|>final<|message|>5<|return|>"
    assert protocol.parse(final, TOOLS, prefix="<|start|>assistant", final=True) == dict(
        content="5", reasoning_content=""
    )


@pytest.mark.parametrize(
    "body",
    [
        dict(tools=[{}]),
        dict(tools=TOOLS, tool_choice="required"),
        dict(tools=TOOLS, tool_choice={"type": "function", "function": {"name": "add"}}),
        dict(tools=[dict(type="function", function=dict(name="add", strict="yes"))]),
        dict(tools=TOOLS * 2),
        dict(tools=TOOLS, parallel_tool_calls="no"),
    ],
)
def test_unsupported_tool_controls_fail_explicitly(body):
    with pytest.raises(ValueError):
        request_tools(body)


def test_template_normalization_preserves_request_and_tool_ids():
    messages = [
        dict(
            role="assistant",
            content=None,
            reasoning_content="compute",
            tool_calls=[dict(id="call_1", type="function", function=dict(name="add", arguments='{"a":2,"b":3}'))],
        ),
        dict(role="tool", tool_call_id="call_1", content=[dict(type="text", text="5")]),
    ]
    original = deepcopy(messages)
    parsed = template_messages(messages)
    assert messages == original
    assert parsed[0]["tool_calls"][0]["function"]["arguments"] == dict(a=2, b=3)
    assert parsed[1] == dict(role="tool", tool_call_id="call_1", content="5")
    assert request_tools(dict(tools=TOOLS, tool_choice="none")) == []


def test_python_scalar_rendering_and_pythonic_template_selection():
    tools = [
        dict(
            type="function",
            function=dict(name="echo", parameters=dict(type="object", properties=dict(enabled=dict(type="boolean")))),
        )
    ]
    raw = "<tool_call><function=echo><parameter=enabled>\nFalse\n</parameter></function></tool_call>"
    result = ToolProtocol("qwen_coder").parse(raw, tools, prefix="", final=True)
    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == dict(enabled=False)
    tokenizer = SimpleNamespace(chat_template="tools tool_calls Python list of function calls")
    assert select_protocol(tokenizer, dict(model_type="llama"), TOOLS).name == "pythonic"


@pytest.mark.parametrize("raw,expected", [("123", "123"), ("null", None)])
def test_nullable_string_arguments(raw, expected):
    tools = [
        dict(
            type="function",
            function=dict(
                name="echo",
                parameters=dict(properties=dict(value=dict(anyOf=[dict(type="string"), dict(type="null")]))),
            ),
        )
    ]
    text = f"<tool_call>echo<arg_key>value</arg_key><arg_value>{raw}</arg_value></tool_call>"
    result = ToolProtocol("arg_xml").parse(text, tools, prefix="", final=True)
    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == dict(value=expected)


@pytest.mark.parametrize("change", [dict(tool_choice="none"), dict(parallel_tool_calls=False), dict(max_tokens=10)])
@pytest.mark.parametrize("stream", [False, True])
def test_disabled_parallel_or_truncated_tools_never_execute(change, stream):
    tokenizer = tokenizer_for("arg_xml")
    raw = "compute</think>" + CALLS["arg_xml"] * 2
    ids = tokenizer.encode(raw, add_special_tokens=False) + [1]
    server = make_server(tokenizer, ids)
    try:
        response = requests.post(
            f"http://127.0.0.1:{server.http.server_port}/v1/chat/completions",
            json=dict(
                model="policy",
                messages=[dict(role="user", content="test")],
                tools=TOOLS,
                temperature=0,
                max_tokens=512,
                stream=stream,
                chat_template_kwargs=dict(tools=TOOLS),
            )
            | change,
            timeout=15,
        )
        assert response.status_code == 200, response.text
        if stream:
            choices = [
                json.loads(line[6:])["choices"][0] for line in response.text.splitlines() if line.startswith("data: {")
            ]
            assert all(not choice["delta"].get("tool_calls") for choice in choices)
        else:
            assert not response.json()["choices"][0]["message"].get("tool_calls")
    finally:
        server.close()


def test_base_checkpoint_without_chat_template_can_use_tools():
    tokenizer = tokenizer_for("json_xml")
    tokenizer.chat_template = None
    ids = (
        tokenizer.encode(CALLS["json_xml"], add_special_tokens=False)
        + [1]
        + tokenizer.encode("5", add_special_tokens=False)
        + [1]
    )
    server = make_server(tokenizer, ids)
    try:
        asyncio.run(tool_rollout(server))
    finally:
        server.close()


def tokenizer_for(protocol, template=None):
    tokenizer = dummy_tokenizer()
    tokenizer.add_special_tokens(
        dict(
            additional_special_tokens=[
                "<|tool_call_start|>",
                "<|tool_call_end|>",
                "<|tool_call>",
                "<tool_call|>",
                '<|"|>',
                "<|channel>",
                "<channel|>",
                "<|start|>",
                "<|message|>",
                "<|channel|>",
                "<|end|>",
                "<|call|>",
                "<|return|>",
                "<|python_tag|>",
                "<|im_start|>",
                "<|im_end|>",
                "<|tool_response>",
            ]
        )
    )
    if template:
        tokenizer.chat_template = template
    elif protocol != "arg_xml":
        # Compact fixtures test field normalization and the token bridge. The
        # optional checkpoint-template run below uses unmodified released Jinja.
        marker = (
            "python list"
            if protocol == "pythonic"
            else CALLS[protocol]
            if protocol != "harmony"
            else "<|start|>assistant"
        )
        tokenizer.chat_template = (
            "{# " + marker + " #}"
            "{% for tool in tools or [] %}{{ tool | tojson }}{% endfor %}"
            "{% for m in messages %}{{ m.role + ': ' + m.content }}"
            "{% if m.tool_calls %}{{ m.tool_calls | tojson }}{% endif %}{{ '\n' }}{% endfor %}"
            "{% if add_generation_prompt %}"
            + ("<|start|>assistant" if protocol == "harmony" else "assistant: ")
            + "{% endif %}"
        )
    return tokenizer


class ScriptedTrainer:
    """Deterministic model output for exercising HTTP/environment control flow."""

    batch_size, seq_length = 1, 8192

    def __init__(self, tokens, vocab=512):
        self.tokens, self.vocab = iter(tokens), vocab
        self.weight = torch.zeros(8, dtype=torch.bfloat16)

    def get_shared_base_weights(self):
        return dict(embedding=self.weight)

    def next_token_logits(self, *args):
        logits = np.zeros((1, self.vocab), dtype=np.float32)
        logits[0, next(self.tokens)] = 8
        return logits

    def decode_logits(self, *args, **kwargs):
        return self.next_token_logits()[0]

    def reset_decode_state(self):
        pass


def make_server(tokenizer, tokens, config=None):
    config = deepcopy(config) if config else dict(vocab_size=512)
    config.get("text_config", config)["vocab_size"] = 512
    trainer = ScriptedTrainer(tokens)
    eos = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|call|>"),
        tokenizer.convert_tokens_to_ids("<|return|>"),
    ]
    server = SharedModelServer(
        trainer,
        tokenizer,
        config,
        dict(
            host="127.0.0.1", port=0, model="base", max_context=trainer.seq_length, max_concurrency=2, eos_token_id=eos
        ),
    )
    server.publish("policy", [], 0)
    return server


@pytest.mark.parametrize("protocol", CALLS)
@pytest.mark.parametrize("stream", [False, True])
def test_http_tool_response_keeps_raw_tokens_and_logprobs(protocol, stream):
    tokenizer = tokenizer_for(protocol)
    raw = ("compute</think>" if protocol == "arg_xml" else "") + CALLS[protocol]
    ids = tokenizer.encode(raw, add_special_tokens=False) + ([] if protocol == "harmony" else [tokenizer.eos_token_id])
    server = make_server(tokenizer, ids)
    try:
        response = requests.post(
            f"http://127.0.0.1:{server.http.server_port}/v1/chat/completions",
            json=dict(
                model="policy",
                messages=[dict(role="user", content="add 2 and 3")],
                tools=TOOLS,
                temperature=0,
                stream=stream,
                max_tokens=512,
                logprobs=True,
                return_token_ids=True,
            ),
            timeout=20,
        )
        assert response.status_code == 200, response.text
        if stream:
            chunks = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
            choices = [chunk["choices"][0] for chunk in chunks]
            calls = [call for choice in choices for call in choice["delta"].get("tool_calls", [])]
            assert [t for choice in choices for t in choice.get("token_ids", [])] == ids
            scores = [s for choice in choices for s in (choice.get("logprobs") or {}).get("content", [])]
            assert choices[-1]["finish_reason"] == "tool_calls"
            assert "<tool_call" not in "".join(choice["delta"].get("content", "") for choice in choices)
        else:
            choice = response.json()["choices"][0]
            assert choice["token_ids"] == ids
            assert choice["finish_reason"] == "tool_calls"
            scores, calls = choice["logprobs"]["content"], choice["message"]["tool_calls"]
        assert len(scores) == len(ids)
        assert all(s["logprob"] == pytest.approx(8 - np.log(np.exp(8) + 511)) for s in scores)
        assert len(calls) == 1 and json.loads(calls[0]["function"]["arguments"]) == dict(a=2, b=3)
    finally:
        server.close()


async def tool_rollout(server, temperature=0, **sampling):
    import verifiers as vf
    from datasets import Dataset
    from openai import AsyncOpenAI

    from surogate.grpo.tool_client import ToolChatCompletionsTokenClient

    used = []

    def add(a: int, b: int) -> str:
        """Add two integers.

        Args:
            a: First integer.
            b: Second integer.
        """
        used.append((a, b))
        return str(a + b)

    def correct(completion, **kwargs):
        return float(completion[-1].content.strip() == "5" and used == [(2, 3)])

    row = dict(
        example_id=0,
        prompt=[
            dict(
                role="user",
                content="First call the add tool with a=2 and b=3. Wait for its result before answering. Then reply with only the resulting number.",
            )
        ],
        answer="5",
    )
    env = vf.ToolEnv(dataset=Dataset.from_list([row]), tools=[add], max_turns=3, rubric=vf.Rubric(funcs=[correct]))
    async with AsyncOpenAI(
        base_url=f"http://127.0.0.1:{server.http.server_port}/v1", api_key="test", max_retries=0
    ) as client:
        output = await env.run_rollout(
            row,
            client=ToolChatCompletionsTokenClient(client),
            model="policy",
            sampling_args=dict(temperature=temperature, max_tokens=512) | sampling,
            state_columns=["trajectory", "sampling_args"],
        )
    assert output["error"] is None, output
    assert used == [(2, 3)], output["completion"]
    assert output["reward"] == 1
    assert len(output["trajectory"]) == 2
    first, second = [step["tokens"] for step in output["trajectory"]]
    prefix = first["prompt_ids"] + first["completion_ids"]
    assert second["prompt_ids"][: len(prefix)] == prefix
    return output


FAMILIES = configurations() | {"glm": dummy_config()}


@pytest.mark.parametrize("family", FAMILIES)
def test_every_training_family_can_use_tools_with_a_text_only_template(family):
    from surogate.grpo.orchestrator.trajectories import interleave_rollout

    tokenizer = tokenizer_for(
        "json_xml",
        "{% for m in messages %}{{ m.role + ': ' + m.content + '\n' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}",
    )
    assert select_protocol(tokenizer, FAMILIES[family], TOOLS).fallback
    ids = (
        tokenizer.encode(CALLS["json_xml"], add_special_tokens=False)
        + [1]
        + tokenizer.encode("5", add_special_tokens=False)
        + [1]
    )
    server = make_server(tokenizer, ids, FAMILIES[family])
    try:
        output = asyncio.run(tool_rollout(server))
        samples = interleave_rollout(output)
        assert len(samples) == 1
        sample = samples[0]
        assert sample.num_turns == 2
        assert sum(sample.completion_mask) == len(ids)
        assert any(not masked for masked in sample.completion_mask)
        assert all(lp == 0 for lp, masked in zip(sample.completion_logprobs, sample.completion_mask) if not masked)
    finally:
        server.close()


RELEASED = {
    "glm": ("arg_xml", None),
    "qwen3": ("json_xml", "Qwen--Qwen3-0.6B"),
    "qwen3_5": ("qwen_coder", "Qwen--Qwen3.5-0.8B"),
    "minicpm5": ("minicpm", "openbmb--MiniCPM5-1B"),
    "spark": ("arg_xml", "XHToken--Spark-X2.5-1.7B"),
    "laguna": ("arg_xml", "poolside--Laguna-S-2.1"),
    "lfm2": ("lfm2", "LiquidAI--LFM2.5-350M"),
    "gemma4": ("gemma4", "google--gemma-4-E2B-it"),
    "gpt_oss": ("harmony", "openai--gpt-oss-20b"),
}


@pytest.mark.parametrize("family", RELEASED)
def test_released_chat_template_tool_loop(family):
    name, checkpoint = RELEASED[family]
    template = None
    if checkpoint:
        if not os.environ.get("SUROGATE_TOOL_TEMPLATES"):
            pytest.skip("set SUROGATE_TOOL_TEMPLATES to downloaded checkpoint tokenizer/template files")
        root = Path(os.environ["SUROGATE_TOOL_TEMPLATES"]) / checkpoint
        path = root / "chat_template.jinja"
        template = (
            path.read_text()
            if path.exists()
            else json.loads((root / "tokenizer_config.json").read_text())["chat_template"]
        )
    tokenizer = tokenizer_for(name, template)
    protocol = select_protocol(tokenizer, FAMILIES[family], TOOLS)
    assert protocol.name == name and not protocol.fallback
    prompt = tokenizer.apply_chat_template(
        [dict(role="user", content="test")], tools=TOOLS, tokenize=False, add_generation_prompt=True
    )
    think = prompt.rfind("<think>") > prompt.rfind("</think>")
    gemma_think = prompt.rfind("<|channel>thought\n") > prompt.rfind("<channel|>")
    reasoning = "compute</think>" if think else "compute<channel|>" if gemma_think else ""
    raw = reasoning + CALLS[name]
    final = "<|channel|>final<|message|>5<|return|>" if name == "harmony" else reasoning + "5"
    ids = tokenizer.encode(raw, add_special_tokens=False) + ([] if name == "harmony" else [1])
    ids += tokenizer.encode(final, add_special_tokens=False) + ([] if name == "harmony" else [1])
    server = make_server(tokenizer, ids, FAMILIES[family])
    try:
        asyncio.run(tool_rollout(server))
    finally:
        server.close()
