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

def config_values(cfg: str) -> dict[int, str]:
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
    # Longest-value-first so 4096 is not shadowed by a shorter rule.
    m = {
        out_rows: "TextConfig::output_rows",
        2 * inter: "2 * TextConfig::intermediate",
        conv_dim + value_dim: "TextConfig::convolution_dim + TextConfig::value_dim",
        inter: "TextConfig::intermediate",
        2 * qsize + 2 * kvsize: "TextConfig::mtp_attention_input_rows",
        conv_dim: "TextConfig::convolution_dim",
        2 * hidden: "TextConfig::mtp_input_rows",
        qsize: "TextConfig::query_size",
        hidden: "TextConfig::hidden",
        kvsize: "TextConfig::kv_size",
        hd: "TextConfig::head_dim",
        khd: "TextConfig::gdn_key_head_dim",
        2 * vh: "2 * TextConfig::gdn_value_heads",
        vh: "TextConfig::gdn_value_heads",
        conv_k: "TextConfig::gdn_conv_kernel",
    }
    return {k: v for k, v in m.items() if k}

def transform(src: str, values: dict[int, str]) -> str:
    order = sorted(values, key=lambda v: -v)
    def fix_args(text: str) -> str:
        for v in order:
            text = re.sub(rf"(?<![\w:.]){v}(?![\w.])", values[v], text)
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

if __name__ == "__main__":
    target = pathlib.Path(sys.argv[1])
    cfg = (target / "impl/config.h").read_text()
    p = target / "impl/load/bindings.cpp"
    p.write_text(transform(p.read_text(), config_values(cfg)))
    print(f"{target.name}: de-literalised")
