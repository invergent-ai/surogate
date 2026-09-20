"""CPU-only exact-token contract tests; never construct a native GPU engine."""

from copy import deepcopy

import pytest

from surogate.serve.candidate_readout import PROTOCOL, TIMINGS, score_tokens
from surogate.serve.engine import Engine


class FakeNative:
    def __init__(self):
        self.calls = []
        self.change = lambda result: result

    def score_tokens(self, inputs, candidates, *, allow_prefix_reuse):
        self.calls.append((list(inputs), list(candidates), allow_prefix_reuse))
        result = {
            "protocol": PROTOCOL,
            "candidate_token_ids": list(candidates),
            "next_token_logits": [float(token) / 10 for token in candidates],
            "prompt_tokens": len(inputs),
            "reused_prompt_tokens": 0,
            "finish_reason": "output_limit",
            "timings": dict.fromkeys(TIMINGS, 0.001),
        }
        # Deliberate mutation verifies copied-input protection.
        inputs.clear()
        candidates.clear()
        return self.change(result)

    def generate(self, *args, **kwargs):
        raise AssertionError("Readout must not call text generation")

    def count_tokens(self, *args, **kwargs):
        raise AssertionError("Readout must not tokenize or rerender chat")


def test_exact_prompt_and_order_survive_without_synthetic_decision():
    native = FakeNative()
    inputs, candidates = [7, 3, 7, 99], [9, 2, 5]
    result = score_tokens(native, inputs, candidates)
    assert native.calls == [(inputs, candidates, False)]
    assert inputs == [7, 3, 7, 99] and candidates == [9, 2, 5]
    assert result["candidate_token_ids"] == [9, 2, 5]
    assert result["next_token_logits"] == [0.9, 0.2, 0.5]
    assert not {"content", "token_ids", "generated_token_ids", "prediction", "probabilities"} & result.keys()
    flipped = score_tokens(native, inputs, candidates[::-1])
    assert flipped["next_token_logits"] == result["next_token_logits"][::-1]


@pytest.mark.parametrize("count", [1, 2, 255, 256])
def test_candidate_boundaries_preserve_distinct_continuations(count):
    candidates = list(range(count))[::-1]
    result = score_tokens(FakeNative(), [1, 2], candidates)
    assert result["candidate_token_ids"] == candidates
    assert len(result["next_token_logits"]) == count


@pytest.mark.parametrize(
    "inputs,candidates",
    [
        ([], [1]),
        ([True], [1]),
        ([1.0], [1]),
        ([-1], [1]),
        ([2**31], [1]),
        ("text", [1]),
        ([1], []),
        ([1], [False]),
        ([1], [1.0]),
        ([1], [-1]),
        ([1], [2**31]),
        ([1], [2, 2]),
        ([1], list(range(257))),
        ([1], "AB"),
    ],
)
def test_invalid_inputs_fail_before_native_invocation(inputs, candidates):
    native = FakeNative()
    with pytest.raises(ValueError):
        score_tokens(native, inputs, candidates)
    assert not native.calls


@pytest.mark.parametrize(
    "damage",
    [
        "missing_logits",
        "short_logits",
        "nan",
        "infinity",
        "bool_logit",
        "wrong_order",
        "wrong_prompt_count",
        "unexpected_reuse",
        "bad_finish",
        "timing_nan",
        "negative_time",
        "synthetic_token",
        "wrong_protocol",
        "duplicate_candidates",
        "bool_prompt_count",
    ],
)
def test_runtime_result_contract_refuses_corruption(damage):
    native = FakeNative()

    def change(result):
        result = deepcopy(result)
        if damage == "missing_logits":
            del result["next_token_logits"]
        elif damage == "short_logits":
            result["next_token_logits"].pop()
        elif damage in ("nan", "infinity", "bool_logit"):
            result["next_token_logits"][0] = {"nan": float("nan"), "infinity": float("inf"), "bool_logit": True}[damage]
        elif damage == "wrong_order":
            result["candidate_token_ids"].reverse()
        elif damage == "wrong_prompt_count":
            result["prompt_tokens"] += 1
        elif damage == "unexpected_reuse":
            result["reused_prompt_tokens"] = 1
        elif damage == "bad_finish":
            result["finish_reason"] = "cancelled"
        elif damage in ("timing_nan", "negative_time"):
            result["timings"]["total_seconds"] = float("nan") if damage == "timing_nan" else -1
        elif damage == "synthetic_token":
            result["generated_token_ids"] = [3]
        elif damage == "wrong_protocol":
            result["protocol"] = "other"
        elif damage == "duplicate_candidates":
            result["candidate_token_ids"] = [3, 3]
        elif damage == "bool_prompt_count":
            result["prompt_tokens"] = True
        return result

    native.change = change
    with pytest.raises(ValueError):
        score_tokens(native, [1, 2], [3, 4])


def test_prefix_reuse_is_explicit_opt_in_and_bool_only():
    native = FakeNative()
    native.change = lambda result: {**result, "reused_prompt_tokens": 1}
    assert score_tokens(native, (1, 2), (3, 4), allow_prefix_reuse=True)["reused_prompt_tokens"] == 1
    with pytest.raises(ValueError, match="boolean"):
        score_tokens(native, [1, 2], [3, 4], allow_prefix_reuse=1)


def test_engine_wrapper_uses_only_raw_native_method():
    engine = object.__new__(Engine)
    engine._engine = FakeNative()
    result = engine.score_tokens([11, 12], [19, 17])
    assert result["next_token_logits"] == [1.9, 1.7]
    assert engine._engine.calls == [([11, 12], [19, 17], False)]


def test_older_binding_does_not_fall_back_to_chat_generation():
    with pytest.raises(RuntimeError, match="rebuild"):
        score_tokens(object(), [1, 2], [3, 4])


def test_artifact_vocabulary_bounds_error_is_not_silently_clipped():
    class BoundedNative:
        def score_tokens(self, *args, **kwargs):
            raise ValueError("invalid candidate token readout")

    with pytest.raises(ValueError, match="invalid candidate"):
        score_tokens(BoundedNative(), [1, 2], [248320])


def test_existing_generate_forwarding_is_unchanged():
    class ExistingNative:
        def generate(self, prompt, **kwargs):
            return {"prompt": prompt, **kwargs}

    engine = object.__new__(Engine)
    engine._engine = ExistingNative()
    result = engine.generate("hello", max_new=3, greedy=True)
    assert result == {
        "prompt": "hello",
        "enable_thinking": True,
        "max_new": 3,
        "greedy": True,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "min_p": None,
        "seed": None,
        "stop": [],
        "on_delta": None,
    }
