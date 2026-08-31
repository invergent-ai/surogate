"""Cross-checks a committed serve target's constants against the DSL declaration.

`check_roundtrip.py` regenerates targets that were written to be generated and
demands a byte-exact match. That is the right test for a generated file and the
wrong one for a hand-written target: `qwen4exp/impl/config.h` carries prose that
records *why* values are what they are ("also cuts the n-gram context"), and a
generator forced to reproduce that prose would simply relocate the duplication.

What actually matters is narrower and stronger: the declaration and the serving
engine must agree about the model. So this parses the literal `static constexpr`
values out of a committed header and compares them against the same quantities
read from `surogate/dsl/models/*.py`. It catches drift on a header nobody intends
to generate yet, it does not break when a comment improves, and it is what lets a
hand-written target migrate incrementally: today the values are checked, later the
file is emitted, and the check narrows to a diff.

Run: python check_contract.py <target_name> <model_dir>
     python check_contract.py qwen4exp ../../../../models/Qwen3.8-Flash-Next-frontend
"""

from __future__ import annotations

import pathlib
import re
import sys
from typing import Any, Callable

from from_dsl import _compile, _config  # noqa: PLC2701 - same package, one contract

#: C++ constant name -> DSL runtime-config key, or a callable over that config.
#: Only quantities the declaration genuinely owns belong here; anything the header
#: derives arithmetically (`hc_width = hc_count * hidden`) is checked implicitly by
#: checking its inputs.
CONTRACT: dict[str, str | Callable[[dict[str, Any]], Any]] = {
    "hidden": "d_model",
    "layers": "n_layers",
    "intermediate": "d_ff",
    "output_rows": "vocab_size",
    "rms_epsilon": "eps",
    "full_attention_interval": "full_attention_interval",
    # Full attention
    "query_heads": "num_query_heads",
    "kv_heads": "num_kv_heads",
    "head_dim": "head_size",
    "rotary_dim": "rotary_dim",
    # Gated delta net
    "gdn_conv_kernel": "linear_conv_kernel_dim",
    "gdn_key_heads": "linear_num_key_heads",
    "gdn_key_head_dim": "linear_key_head_dim",
    "gdn_value_heads": "linear_num_value_heads",
    "gdn_value_head_dim": "linear_value_head_dim",
    # Hyper-connections
    "hc_count": "hc_count",
    "hc_low_rank": "hc_lowrank",
    # Sparse MoE
    "experts": "num_experts",
    "experts_per_token": "num_experts_per_tok",
    "shared_intermediate": "shared_expert_intermediate",
    # QSA indexer
    "indexer_heads": "indexer_n_heads",
    "indexer_head_dim": "indexer_head_dim",
    "indexer_top_k": "indexer_budget",
    "indexer_block": "indexer_compress_ratio",
    # n-gram PLE. `ple_layer_ids` is 1-based in the HF config and 0-based in the
    # engine — the conversion is stated here so a future edit to either side has
    # to confront it rather than silently disagree by one layer.
    "ple_layer": lambda c: (c["ple_layer_ids"][0] - 1) if c.get("ple_layer_ids") else None,
    "ple_ngram": "ngram_size",
    "ple_heads_per_gram": "heads_per_ngram",
    "ple_conv_kernel": "ple_conv_kernel_size",
    "ple_embed": "ple_embed_dim",
    "ple_head_dim": lambda c: (
        c["ple_embed_dim"] // ((c["ngram_size"] - 1) * c["heads_per_ngram"])
        if c.get("ple_embed_dim") and c.get("ngram_size") and c.get("heads_per_ngram")
        else None
    ),
    # MTP is declared but not served; the count must still agree.
    "mtp_layers": "mtp_num_hidden_layers",
}


def _q(config, key):
    return config.get(key)


#: The converter is the third description of the same model, and it restates the
#: geometry in its own constants (`LAYERS = 48`, `HIDDEN = 2560`). Those are checked
#: here for the same reason the header's are: nothing else notices when they drift
#: away from what training compiles. Derived entries are checked too, because a
#: fused row count that disagrees with its parts is exactly the bug that surfaces
#: as a load-time shape error on a hundred-gigabyte artifact.
CONVERTER_CONTRACT: dict[str, str | Callable[[dict[str, Any]], Any]] = {
    "LAYERS": "n_layers",
    "HIDDEN": "d_model",
    "VOCAB": "vocab_size",
    "FULL_ATTENTION_INTERVAL": "full_attention_interval",
    "HEAD_DIM": "head_size",
    "QUERY_HEADS": "num_query_heads",
    "KV_HEADS": "num_kv_heads",
    "HC_COUNT": "hc_count",
    "HC_LOW_RANK": "hc_lowrank",
    "GDN_CONV_KERNEL": "linear_conv_kernel_dim",
    "GDN_KEY_HEADS": "linear_num_key_heads",
    "GDN_VALUE_HEADS": "linear_num_value_heads",
    "GDN_HEAD_DIM": "linear_key_head_dim",
    "EXPERTS": "num_experts",
    "TOP_K": "num_experts_per_tok",
    "EXPERT_FFN": "d_ff",
    "SHARED_FFN": "shared_expert_intermediate",
    "INDEXER_HEADS": "indexer_n_heads",
    "INDEXER_DIM": "indexer_head_dim",
    "PLE_NGRAM": "ngram_size",
    "PLE_HEADS_PER_NGRAM": "heads_per_ngram",
    "PLE_CONV_KERNEL": "ple_conv_kernel_size",
    "PLE_EMBED": "ple_embed_dim",
    "PLE_LAYER": lambda c: (c["ple_layer_ids"][0] - 1) if c.get("ple_layer_ids") else None,
    "PLE_HEADS": lambda c: (c["ngram_size"] - 1) * c["heads_per_ngram"],
    "PLE_HEAD_DIM": lambda c: c["ple_embed_dim"] // ((c["ngram_size"] - 1) * c["heads_per_ngram"]),
    # Derived: the fused row counts the binder and the converter must both agree on.
    "HC_WIDTH": lambda c: c["hc_count"] * c["d_model"],
    "QUERY_SIZE": lambda c: c["num_query_heads"] * c["head_size"],
    "KV_SIZE": lambda c: c["num_kv_heads"] * c["head_size"],
    "GDN_KEY_DIM": lambda c: c["linear_num_key_heads"] * c["linear_key_head_dim"],
    "GDN_VALUE_DIM": lambda c: c["linear_num_value_heads"] * c["linear_value_head_dim"],
    "GDN_CONV_DIM": lambda c: (
        2 * c["linear_num_key_heads"] * c["linear_key_head_dim"]
        + c["linear_num_value_heads"] * c["linear_value_head_dim"]
    ),
    "GDN_FUSED_ROWS": lambda c: (
        2 * c["linear_num_key_heads"] * c["linear_key_head_dim"]
        + 2 * c["linear_num_value_heads"] * c["linear_value_head_dim"]
    ),
    "ATTENTION_FUSED_ROWS": lambda c: (
        2 * c["num_query_heads"] * c["head_size"] + 2 * c["num_kv_heads"] * c["head_size"]
    ),
    "ROUTER_ROWS": lambda c: c["num_experts"] + 1,
}

_CONSTANT = re.compile(
    r"static\s+constexpr\s+(?:int|float|std::int64_t|std::uint32_t|bool)\s+"
    r"(\w+)\s*=\s*([^;]+);"
)
_LITERAL = re.compile(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?[FfUuLl]*$")


def text_struct(header: str) -> str:
    """Just the `struct TextConfig { ... }` body.

    Scoping matters: a target header declares several structs and they reuse
    names — `DFlashConfig` has its own `layers`, which flattening the file would
    silently substitute for the text stack's.
    """

    start = header.find("struct TextConfig")
    if start < 0:
        return header
    depth, index = 0, header.index("{", start)
    for position in range(index, len(header)):
        if header[position] == "{":
            depth += 1
        elif header[position] == "}":
            depth -= 1
            if depth == 0:
                return header[index : position + 1]
    return header[index:]


def parse_constants(header: str) -> dict[str, float | int]:
    """Literal `static constexpr` values from the text config. Expressions over
    other constants are skipped — they are checked by checking their inputs."""

    found: dict[str, float | int] = {}
    for name, raw in _CONSTANT.findall(text_struct(header)):
        text = raw.strip()
        if not _LITERAL.match(text):
            continue
        cleaned = text.rstrip("FfUuLl")
        found[name] = float(cleaned) if ("." in cleaned or "e" in cleaned.lower()) else int(cleaned)
    return found


def expected_values(config: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for constant, source in CONTRACT.items():
        value = source(config) if callable(source) else config.get(source)
        if value is not None:
            values[constant] = value
    return values


def check(target: str, model_dir: str, targets_root: pathlib.Path) -> int:
    from surogate.dsl.ir_builder import load_hf_config, resolve_architecture

    hf_config = load_hf_config(model_dir)
    architecture = resolve_architecture(hf_config)
    config = _config(_compile(architecture, hf_config))

    header_path = targets_root / target / "impl" / "config.h"
    constants = parse_constants(header_path.read_text())
    expected = expected_values(config)

    agreed: list[str] = []
    disagreed: list[tuple[str, Any, Any]] = []
    for name, want in expected.items():
        if name not in constants:
            continue
        got = constants[name]
        same = abs(got - want) <= 1e-9 * max(1.0, abs(want)) if isinstance(want, float) else got == want
        (agreed if same else disagreed).append(name if same else (name, got, want))  # type: ignore[arg-type]

    print(f"{target} vs {architecture}")
    print(f"  {len(agreed)} constants agree with the declaration")
    unchecked = sorted(set(constants) - set(expected))
    if unchecked:
        print(f"  {len(unchecked)} header constants not in the contract "
              f"(serving policy or derived): {', '.join(unchecked[:8])}"
              f"{' ...' if len(unchecked) > 8 else ''}")
    body = text_struct(header_path.read_text())
    missing = sorted(set(expected) - set(constants))
    derived = [name for name in missing if re.search(rf"\b{name}\s*=", body)]
    absent = [name for name in missing if name not in derived]
    if derived:
        print(f"  {len(derived)} checked via their inputs (derived in the header): "
              f"{', '.join(derived)}")
    if absent:
        print(f"  {len(absent)} declared quantities absent from the header: {', '.join(absent)}")
    for name, got, want in disagreed:
        print(f"  DISAGREE {name}: header={got!r} declaration={want!r}")
    return len(disagreed) + check_converter(target, config)


def check_converter(target: str, config: dict[str, Any]) -> int:
    """The converter's own geometry constants, against the same declaration."""

    import importlib

    try:
        inventory = importlib.import_module(
            f"surogate.serve.tools.convert.{target}.inventory"
        )
    except ModuleNotFoundError:
        print(f"  (no converter inventory for {target}; skipped)")
        return 0

    constants = {
        name: value
        for name, value in vars(inventory).items()
        if name.isupper() and isinstance(value, int) and not isinstance(value, bool)
    }
    agreed, disagreed = 0, []
    for name, source in CONVERTER_CONTRACT.items():
        if name not in constants:
            continue
        want = source(config) if callable(source) else config.get(source)
        if want is None:
            continue
        if constants[name] == want:
            agreed += 1
        else:
            disagreed.append((name, constants[name], want))

    total = len(getattr(inventory, "TENSOR_SPECS", ()))
    print(f"  converter inventory: {agreed} constants agree "
          f"({total} artifact objects declared)")
    for name, got, want in disagreed:
        print(f"  DISAGREE {name}: converter={got!r} declaration={want!r}")
    return len(disagreed)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    root = pathlib.Path(__file__).resolve().parents[4] / "csrc/src/serve/targets"
    sys.exit(check(sys.argv[1], sys.argv[2], root))
