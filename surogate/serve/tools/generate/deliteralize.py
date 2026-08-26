"""Replaces shape literals in a target's bindings.cpp with TextConfig expressions.

Substitution is confined to the argument lists of the calls that carry weight
shapes, so a literal that happens to equal a shape elsewhere (a group size, a
vocabulary constant) is untouched — that ambiguity is exactly what sank the
text-templating attempt.
"""
from __future__ import annotations
import re, sys, pathlib

CALLS = ("materialized_weight", "bind_weight", "bind_nvfp4_weight", "bind_device_tensor",
         "materialized_tensor", "row_view", "bind_mtp")

def config_values(cfg: str, ambiguous_out: dict[int, list[str]] | None = None) -> dict[int, str]:
    def get(name):
        m = re.search(rf"constexpr int {name}\s*=\s*(\d+);", cfg)
        return int(m.group(1)) if m else None
    hidden, layers, inter = get("hidden"), get("layers"), get("intermediate")
    out_rows = get("output_rows")
    qh, kvh, hd = get("query_heads"), get("kv_heads"), get("head_dim")
    kh, khd = get("gdn_key_heads"), get("gdn_key_head_dim")
    vh, vhd = get("gdn_value_heads"), get("gdn_value_head_dim")
    conv_k = get("gdn_conv_kernel")
    key_dim, value_dim = kh * khd, vh * vhd
    conv_dim = 2 * key_dim + value_dim
    qsize, kvsize = qh * hd, kvh * hd
    # Built as PAIRS, not a dict literal. A dict would silently drop a colliding
    # value — the 2B's intermediate and convolution_dim are both 6144, and its
    # query_size and hidden are both 2048 — leaving whichever happened to be
    # written last. Collisions have to survive construction in order to be
    # refused below.
    pairs = [
        (out_rows, "TextConfig::output_rows"),
        (2 * inter, "2 * TextConfig::intermediate"),
        (conv_dim + value_dim, "TextConfig::convolution_dim + TextConfig::value_dim"),
        (inter, "TextConfig::intermediate"),
        (2 * qsize + 2 * kvsize, "TextConfig::mtp_attention_input_rows"),
        (conv_dim, "TextConfig::convolution_dim"),
        (2 * hidden, "TextConfig::mtp_input_rows"),
        (qsize, "TextConfig::query_size"),
        (hidden, "TextConfig::hidden"),
        (kvsize, "TextConfig::kv_size"),
        (hd, "TextConfig::head_dim"),
        (khd, "TextConfig::gdn_key_head_dim"),
        (2 * vh, "2 * TextConfig::gdn_value_heads"),
        (vh, "TextConfig::gdn_value_heads"),
        (conv_k, "TextConfig::gdn_conv_kernel"),
    ]
    # Collisions are refused, not guessed: a value-based rule cannot tell which
    # quantity a literal meant, and substituting the wrong one is correct today
    # and wrong the moment either quantity changes.
    by_value: dict[int, list[str]] = {}
    for value, expr in pairs:
        if value:
            by_value.setdefault(value, []).append(expr)
    resolved = {}
    for value, exprs in sorted(by_value.items()):
        distinct = sorted(set(exprs))
        if len(distinct) == 1:
            resolved[value] = distinct[0]
        else:
            if ambiguous_out is not None:
                ambiguous_out[value] = distinct
            else:
                print(f"  ambiguous: {value} could be " + " or ".join(distinct))
    return resolved


# When a value is ambiguous, the call itself says which quantity was meant: a
# weight named "mlp/down" has intermediate columns, "gdn/convolution" has
# convolution_dim. These rules resolve exactly that, keyed on a token that
# appears in the call, and are consulted only for values the value-based pass
# refused.
CONTEXT_RULES: tuple[tuple[str, str, str], ...] = (
    ("mlp/down", "intermediate", "TextConfig::intermediate"),
    ("plan.down", "intermediate", "TextConfig::intermediate"),
    ("mlp.down", "intermediate", "TextConfig::intermediate"),
    ("mlp/gate_up", "intermediate", "TextConfig::intermediate"),
    ("gdn/convolution", "convolution", "TextConfig::convolution_dim"),
    ("gdn.convolution", "convolution", "TextConfig::convolution_dim"),
    ("attention/output", "query", "TextConfig::query_size"),
    ("attention.output", "query", "TextConfig::query_size"),
    ("gdn/output", "value", "TextConfig::value_dim"),
    ("gdn.output", "value", "TextConfig::value_dim"),
    # A 1-D norm is over the residual stream. query_norm and key_norm are over
    # the head dim, but that value is unambiguous so it never reaches here.
    ("norm", "hidden", "TextConfig::hidden"),
)


# Rows and columns of a weight cannot be told apart by a whole-call rule: the
# 2B's hidden and query_size are both 2048, and `plan.down` has hidden rows and
# intermediate columns. These map a weight's identity to the role of each
# dimension, which is the only thing that actually disambiguates.
ROLE_RULES: tuple[tuple[str, str, str], ...] = (
    ("plan.down", "TextConfig::hidden", "TextConfig::intermediate"),
    ("plan.gate_up", "2 * TextConfig::intermediate", "TextConfig::hidden"),
    ("source.attention.output", "TextConfig::hidden", "TextConfig::query_size"),
    ("source.gdn.output", "TextConfig::hidden", "TextConfig::value_dim"),
)


def apply_role_rules(call_text: str, resolved: dict[int, str]) -> str:
    """Rewrites the trailing `rows, cols` pair of a weight call when the weight's
    identity fixes what each dimension means."""
    for token, rows_expr, cols_expr in ROLE_RULES:
        if token not in call_text:
            continue
        parts = [a.strip() for a in call_text.split(",")]
        if len(parts) < 2:
            continue
        parts[-2], parts[-1] = rows_expr, cols_expr
        return ", ".join(parts)
    return call_text


def apply_trailing_dim_rule(call_text: str, ambiguous: dict[int, list[str]]) -> str:
    """A weight's trailing dimension is its input width, i.e. hidden.

    True for every weight in these targets except the two whose columns are the
    MLP intermediate or the MTP input rows, and those are already resolved by
    the role rules above, so by the time an ambiguous value is still sitting in
    the trailing position, hidden is what it means. Only the trailing token is
    touched — the leading dimension keeps whatever the value-based pass decided.
    """
    # row_view's trailing argument is a ROW COUNT, not an input width. Applying
    # the rule there rewrote a count as hidden — correct by value on the 2B,
    # where hidden and query_size coincide, and wrong in meaning.
    if "row_view" in call_text:
        return call_text
    for value, options in ambiguous.items():
        if "TextConfig::hidden" not in options:
            continue
        pattern = rf"(?<![\w:.]){value}(?![\w.])(\s*[}}\)]?\s*)$"
        call_text = re.sub(pattern, r"TextConfig::hidden\1", call_text.rstrip()) + ""
    return call_text


def resolve_ambiguous(call_text: str, value: int, options: list[str]) -> str | None:
    """Picks one of several equal-valued expressions using the call's own text."""
    for token, _kind, expr in CONTEXT_RULES:
        if token in call_text and expr in options:
            return expr
    return None


def transform(src: str, values: dict[int, str]) -> str:
    order = sorted(values, key=lambda v: -v)
    def fix_args(text: str) -> str:
        for v in order:
            text = re.sub(rf"(?<![\w:.]){v}(?![\w.])", values[v], text)
        for v, options in sorted(AMBIGUOUS.items(), key=lambda kv: -kv[0]):
            expr = resolve_ambiguous(text, v, options)
            if expr is not None:
                text = re.sub(rf"(?<![\w:.]){v}(?![\w.])", expr, text)
        if AMBIGUOUS:
            text = apply_role_rules(text, values)
            text = apply_trailing_dim_rule(text, AMBIGUOUS)
        return text
    out, i = [], 0
    pattern = re.compile(rf"\b({'|'.join(CALLS)})\s*\(")
    while True:
        m = pattern.search(src, i)
        if not m:
            out.append(src[i:]); break
        out.append(src[i:m.end()])
        depth, j = 1, m.end()
        while depth:
            if src[j] == "(": depth += 1
            elif src[j] == ")": depth -= 1
            j += 1
        out.append(fix_args(src[m.end():j-1])); out.append(")")
        i = j
    return "".join(out)

AMBIGUOUS: dict[int, list[str]] = {}


if __name__ == "__main__":
    target = pathlib.Path(sys.argv[1])
    cfg = (target / "impl/config.h").read_text()
    p = target / "impl/load/bindings.cpp"
    AMBIGUOUS.clear()
    resolved = config_values(cfg, AMBIGUOUS)
    p.write_text(transform(p.read_text(), resolved))
    unresolved = {v: o for v, o in AMBIGUOUS.items()}
    print(f"{target.name}: de-literalised"
          + (f"; {len(unresolved)} ambiguous values resolved by context or left literal"
             if unresolved else ""))
