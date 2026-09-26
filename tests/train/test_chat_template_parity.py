"""The native training encoder renders exactly what the model's chat template renders.

`Tokenizer.encode_for_training_json_batch` (C++) must give, for every conversation shape the
datasets carry, the text and token ids `apply_chat_template` gives, and train exactly the
assistant turns. CPU only; each cached checkpoint's tokenizer is a case, absent ones skip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")

try:
    from surogate._surogate import Tokenizer as NativeTokenizer
except ImportError:
    pytest.skip("surogate._surogate C++ extension not built", allow_module_level=True)

HUB = Path("~/.cache/huggingface/hub").expanduser()
MODELS = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.8-Flash-Next", "Qwen/Qwen3-0.6B"]


def _snapshot(model_id: str) -> Path | None:
    snaps = HUB / f"models--{model_id.replace('/', '--')}" / "snapshots"
    for snap in sorted(snaps.glob("*"), reverse=True) if snaps.exists() else []:
        if (snap / "tokenizer.json").exists() and (snap / "tokenizer_config.json").exists():
            return snap
    return None


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
]

CASES = {
    "no_system": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4."},
    ],
    "system": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4."},
    ],
    "reasoning_content": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "reasoning_content": "Two plus two is four.", "content": "4."},
    ],
    "reasoning_in_content": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>\nTwo plus two is four.\n</think>\n\n4."},
    ],
    "multi_turn_history": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "reasoning_content": "Easy.", "content": "4."},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "reasoning_content": "Also easy.", "content": "6."},
    ],
    "mixed_think": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "What is 3*3?"},
        {"role": "assistant", "reasoning_content": "Three threes.", "content": "9."},
    ],
    "tool_call": [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}],
        },
        {"role": "tool", "content": '{"temp": 21}'},
        {"role": "assistant", "content": "It is 21 degrees in Paris."},
    ],
}


@pytest.fixture(scope="module", params=MODELS)
def tokenizers(request):
    snap = _snapshot(request.param)
    if snap is None:
        pytest.skip(f"{request.param} tokenizer not cached")
    hf = transformers.AutoTokenizer.from_pretrained(str(snap))
    if not hf.chat_template:
        pytest.skip(f"{request.param} has no chat template")
    return request.param, hf, NativeTokenizer.from_pretrained(str(snap))


def _encode(native, messages, tools=None, strategy="default"):
    [result] = native.encode_for_training_json_batch(
        [json.dumps(messages)], [[json.dumps(t) for t in tools]] if tools else None, strategy=strategy
    )
    return list(result["input_ids"]), list(result["labels"])


def _trained_text(hf, ids, labels):
    return "".join(hf.decode([t], skip_special_tokens=False) for t, l in zip(ids, labels) if l != -100)


@pytest.mark.parametrize("case", sorted(CASES))
def test_text_and_ids_match_the_template(tokenizers, case):
    name, hf, native = tokenizers
    messages = CASES[case]
    tools = TOOLS if case == "tool_call" else None
    try:
        text = hf.apply_chat_template(messages, tools=tools, tokenize=False)
    except Exception as e:  # a template that cannot render this shape at all
        pytest.skip(f"{name} template refuses {case}: {e}")
    ids, labels = _encode(native, messages, tools)
    assert ids, f"{name}/{case}: the native encoder dropped the conversation"
    assert hf.decode(ids, skip_special_tokens=False) == text
    assert ids == hf(text, add_special_tokens=False).input_ids
    assert len(labels) == len(ids) and labels[0] == -100


@pytest.mark.parametrize("case", sorted(CASES))
def test_only_assistant_turns_are_trained(tokenizers, case):
    name, hf, native = tokenizers
    messages = CASES[case]
    tools = TOOLS if case == "tool_call" else None
    try:
        hf.apply_chat_template(messages, tools=tools, tokenize=False)
    except Exception as e:
        pytest.skip(f"{name} template refuses {case}: {e}")
    ids, labels = _encode(native, messages, tools)
    trained = _trained_text(hf, ids, labels)
    for m in messages:
        if m["role"] != "assistant" and m.get("content"):
            assert m["content"] not in trained, f"{name}/{case}: {m['role']} text is trained"
        if m["role"] == "assistant" and m.get("content"):
            final = m["content"].split("</think>")[-1].strip()
            assert final in trained, f"{name}/{case}: assistant answer {final!r} is not trained"
    assert "<|im_start|>" not in trained and "<|im_start|>assistant" not in trained


def test_qwen35_trained_spans_exactly(tokenizers):
    """The exact trained text on Qwen3.5: reasoning of the turn after the last query, history turns
    without reasoning, and never the empty think block a no-think turn opens with."""
    name, hf, native = tokenizers
    if name != "Qwen/Qwen3.5-0.8B":
        pytest.skip("exact spans are written for the Qwen3.5 template")
    expect = {
        "no_system": "4.<|im_end|>\n",
        "reasoning_content": "Two plus two is four.\n</think>\n\n4.<|im_end|>\n",
        "reasoning_in_content": "Two plus two is four.\n</think>\n\n4.<|im_end|>\n",
        "multi_turn_history": "4.<|im_end|>\nAlso easy.\n</think>\n\n6.<|im_end|>\n",
        "mixed_think": "Hello.<|im_end|>\nThree threes.\n</think>\n\n9.<|im_end|>\n",
    }
    for case, want in expect.items():
        ids, labels = _encode(native, CASES[case])
        assert _trained_text(hf, ids, labels) == want, case
    ids, labels = _encode(native, CASES["reasoning_content"], strategy="thinking_only")
    assert _trained_text(hf, ids, labels) == "Two plus two is four.\n</think>"
    ids, labels = _encode(native, CASES["reasoning_content"], strategy="final_only")
    assert _trained_text(hf, ids, labels) == "\n\n4.<|im_end|>\n"
    ids, labels = _encode(native, CASES["multi_turn_history"], strategy="last_round")
    assert _trained_text(hf, ids, labels) == "Also easy.\n</think>\n\n6.<|im_end|>\n"
