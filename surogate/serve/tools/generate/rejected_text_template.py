"""REJECTED APPROACH — retained as evidence, do not build on it.

Deriving a template by substituting a committed target's numeric values with
placeholders is unsound, and the two-model check below is what proves it. The
source target (qwen3_5_4b) reproduces perfectly, because it is where the values
came from; qwen3_5_0_8b diverges on 28 lines, all of them constants that a
value-based rule cannot distinguish from shapes:

    const std::uint64_t low_group = 32;   <- 32 is a group size, not value_heads
    materialized_weight(..., 2560, 1024)  <- 2560 here is another variant's
                                             branch, not this target's hidden

The lesson generalises: a shape and a constant that happen to share a value are
indistinguishable to text substitution, and the failure is silent for whichever
model the template was derived from. bindings.cpp must therefore be EMITTED
from structure — the weight table, with each entry's rows and columns computed
from the spec — the way emit_config.py writes config.h, rather than patched.

Original docstring follows.

Derives a parameterised template from a committed target, and proves it.

The template is built by substituting a source target's known values with
placeholders. That is only trustworthy if it round-trips on a DIFFERENT model:
a constant wrongly treated as a shape (or a shape missed) renders correctly for
the target it came from and wrongly for any other. So the check renders both
committed targets and diffs each against its own file.

Ordering matters — longer values are substituted first, so 4096 is not eaten by
a rule for 409, and values that coincide between the two models are left alone
because they cannot be told apart from constants by this method.
"""

from __future__ import annotations

import pathlib
import re

from target_spec import TargetSpec


def placeholders(spec: TargetSpec) -> list[tuple[str, str]]:
    """(value, placeholder) pairs, longest value first."""
    l = spec.linear_attention
    assert l is not None
    pairs = {
        "name": spec.name,
        "hidden": spec.hidden,
        "intermediate": spec.intermediate,
        "gate_up_rows": 2 * spec.intermediate,
        "vocab": spec.vocab,
        "query_size": spec.query_size,
        "kv_size": spec.kv_size,
        "attention_parent_rows": 2 * spec.query_size + 2 * spec.kv_size,
        "gdn_value_heads": l.value_heads,
        "gdn_value_dim": l.value_dim,
        "gdn_conv_dim": 2 * l.key_dim + l.value_dim,
        "gdn_parent_rows": 2 * l.key_dim + l.value_dim + l.value_dim,
    }
    out = [(str(v), "{" + k + "}") for k, v in pairs.items()]
    out.sort(key=lambda kv: len(kv[0]), reverse=True)
    return out


def to_template(source: str, spec: TargetSpec) -> str:
    text = source.replace("{", "{{").replace("}", "}}")
    for value, token in placeholders(spec):
        text = re.sub(rf"(?<![\w.]){re.escape(value)}(?![\w.])", token, text)
    return text


def render(template: str, spec: TargetSpec) -> str:
    l = spec.linear_attention
    assert l is not None
    return template.format(
        name=spec.name,
        hidden=spec.hidden,
        intermediate=spec.intermediate,
        gate_up_rows=2 * spec.intermediate,
        vocab=spec.vocab,
        query_size=spec.query_size,
        kv_size=spec.kv_size,
        attention_parent_rows=2 * spec.query_size + 2 * spec.kv_size,
        gdn_value_heads=l.value_heads,
        gdn_value_dim=l.value_dim,
        gdn_conv_dim=2 * l.key_dim + l.value_dim,
        gdn_parent_rows=2 * l.key_dim + l.value_dim + l.value_dim,
    )
