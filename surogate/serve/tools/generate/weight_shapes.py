"""Rewrites weight shapes by the weight's NAME, which is what actually fixes them.

Value-based substitution cannot resolve a literal whose value coincides with
another config quantity, and position-based rules cannot tell rows from columns
without knowing the weight. The artifact names every weight, and a name fixes
both dimensions exactly — "mlp/down" is hidden rows by intermediate columns,
whatever those happen to equal for a given model.

This closes the residue the value pass leaves behind, and it is the form the
generator should ultimately emit from: a weight table keyed by name.
"""

from __future__ import annotations

import pathlib
import re
import sys

# name fragment -> (rows expression, columns expression)
SHAPES: tuple[tuple[str, str, str], ...] = (
    ("text/token_embedding", "TextConfig::output_rows", "TextConfig::hidden"),
    ("text/output_head", "TextConfig::output_rows", "TextConfig::hidden"),
    ("attention/query_key_gate_value", "TextConfig::mtp_attention_input_rows", "TextConfig::hidden"),
    ("attention/output", "TextConfig::hidden", "TextConfig::query_size"),
    ("gdn/query_key_value_z", "TextConfig::convolution_dim + TextConfig::value_dim",
     "TextConfig::hidden"),
    ("gdn/output", "TextConfig::hidden", "TextConfig::value_dim"),
    ("mlp/gate_up", "2 * TextConfig::intermediate", "TextConfig::hidden"),
    ("mlp/down", "TextConfig::hidden", "TextConfig::intermediate"),
)

# Calls whose shape follows the name, in the two spellings the targets use:
#   bind_weight(binder, prefix + "name", FORMAT, {ROWS, COLS})
#   bind_nvfp4_weight(binder, prefix + "name", ROWS, COLS, ...)
BRACED = re.compile(r'("(?:[\w/]*)"[^;{}]*?)\{\s*([^,{}]+?)\s*,\s*([^,{}]+?)\s*\}')
BARE = re.compile(r'(bind_nvfp4_weight\(\s*binder,\s*prefix \+ "[\w/]+",\s*)([^,]+),\s*([^,]+),')


def rewrite(text: str) -> tuple[str, int]:
    changed = 0
    for fragment, rows, cols in SHAPES:
        def braced(m: re.Match[str]) -> str:
            nonlocal changed
            if fragment not in m.group(1):
                return m.group(0)
            if m.group(2) == rows and m.group(3) == cols:
                return m.group(0)
            changed += 1
            return f"{m.group(1)}{{{rows}, {cols}}}"

        def bare(m: re.Match[str]) -> str:
            nonlocal changed
            if fragment not in m.group(1):
                return m.group(0)
            if m.group(2).strip() == rows and m.group(3).strip() == cols:
                return m.group(0)
            changed += 1
            return f"{m.group(1)}{rows}, {cols},"

        text = BRACED.sub(braced, text)
        text = BARE.sub(bare, text)
    return text, changed


if __name__ == "__main__":
    target = pathlib.Path(sys.argv[1])
    p = target / "impl/load/bindings.cpp"
    out, n = rewrite(p.read_text())
    p.write_text(out)
    print(f"{target.name}: {n} weight shapes rewritten by name")
