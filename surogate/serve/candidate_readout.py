"""CPU-side contract for exact-token, ordered native candidate logits.

No tokenizer, model generation, probability calibration or synthetic-token decision
is performed here. The engine validates vocabulary bounds against its artifact.
"""

from __future__ import annotations

import math

PROTOCOL = "native-candidate-readout-v1"
MAX_CANDIDATES = 256
MAX_TOKEN_ID = 2**31 - 1
TIMINGS = frozenset(("prepare_seconds", "first_token_seconds", "prefill_seconds", "decode_seconds", "total_seconds"))
FIELDS = frozenset(
    (
        "protocol",
        "candidate_token_ids",
        "next_token_logits",
        "prompt_tokens",
        "reused_prompt_tokens",
        "finish_reason",
        "timings",
    )
)


def token_ids(value, *, name, candidates=False):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty list or tuple of integer IDs")
    if any(type(token) is not int or not 0 <= token <= MAX_TOKEN_ID for token in value):
        raise ValueError(f"{name} requires nonnegative int32 IDs, not booleans or floats")
    if candidates and (len(value) > MAX_CANDIDATES or len(set(value)) != len(value)):
        raise ValueError("candidate_token_ids must contain 1..256 distinct IDs")
    return list(value)


def validate_result(result, input_ids, candidate_ids, *, allow_prefix_reuse):
    if not isinstance(result, dict) or set(result) != FIELDS or result["protocol"] != PROTOCOL:
        raise ValueError("Unexpected native readout fields/protocol (no generated decision permitted)")
    if token_ids(result["candidate_token_ids"], name="returned candidates", candidates=True) != candidate_ids:
        raise ValueError("Native candidate IDs or their order changed")
    logits = result["next_token_logits"]
    if (
        not isinstance(logits, (list, tuple))
        or len(logits) != len(candidate_ids)
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in logits)
    ):
        raise ValueError("Native candidate logits must be finite and match every candidate")
    if type(result["prompt_tokens"]) is not int or result["prompt_tokens"] != len(input_ids):
        raise ValueError("Native prompt token count differs from exact supplied IDs")
    reused = result["reused_prompt_tokens"]
    if type(reused) is not int or not 0 <= reused <= len(input_ids) or (reused and not allow_prefix_reuse):
        raise ValueError("Unexpected native prefix reuse")
    if result["finish_reason"] not in ("output_limit", "context_capacity"):
        raise ValueError("Native candidate readout did not finish normally")
    timings = result["timings"]
    if (
        not isinstance(timings, dict)
        or set(timings) != TIMINGS
        or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in timings.values())
    ):
        raise ValueError("Native readout timings must be finite nonnegative values")
    return {
        **result,
        "candidate_token_ids": list(candidate_ids),
        "next_token_logits": [float(value) for value in logits],
        "timings": dict(timings),
    }


def score_tokens(engine, input_ids, candidate_token_ids, *, allow_prefix_reuse=False):
    """Score exact token IDs once; return raw logits in the supplied candidate order.

    Prefix reuse is disabled by default for an independent inference comparison.
    Logits are conditional-score inputs, not full-vocabulary probability statistics.
    """
    if type(allow_prefix_reuse) is not bool:
        raise ValueError("allow_prefix_reuse must be boolean")
    inputs = token_ids(input_ids, name="input_ids")
    candidates = token_ids(candidate_token_ids, name="candidate_token_ids", candidates=True)
    method = getattr(engine, "score_tokens", None)
    if method is None:
        raise RuntimeError("Native serving extension lacks score_tokens; rebuild the separate serving binding")
    # Copies prevent an extension/test backend from mutating caller-owned arrays.
    result = method(list(inputs), list(candidates), allow_prefix_reuse=allow_prefix_reuse)
    return validate_result(result, inputs, candidates, allow_prefix_reuse=allow_prefix_reuse)
