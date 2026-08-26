"""Regenerates committed targets and reports any divergence.

Run: python check_roundtrip.py [serve_targets_dir]
"""

from __future__ import annotations

import difflib
import pathlib
import sys

from emit_config import emit_config_h
from target_spec import AttentionSpec, LinearAttentionSpec, TargetSpec

# Declarations for the targets that exist today. These are the values the
# training DSL already carries; wiring them to be read from the DSL rather than
# restated here is the next step.
SPECS = {
    "qwen3_5_0_8b": TargetSpec(
        name="qwen3_5_0_8b", hidden=1024, layers=24, intermediate=3584, vocab=248320,
        rms_epsilon=1e-6, rope_theta=1e7,
        attention=AttentionSpec(query_heads=8, kv_heads=2, head_dim=256, rotary_dim=64),
        linear_attention=LinearAttentionSpec(16, 128, 16, 128, 4),
    ),
    "qwen3_5_4b": TargetSpec(
        name="qwen3_5_4b", hidden=2560, layers=32, intermediate=9216, vocab=248320,
        rms_epsilon=1e-6, rope_theta=1e7,
        attention=AttentionSpec(query_heads=16, kv_heads=4, head_dim=256, rotary_dim=64),
        linear_attention=LinearAttentionSpec(16, 128, 32, 128, 4),
    ),
}


def main(root: pathlib.Path) -> int:
    failures = 0
    for name, spec in SPECS.items():
        committed = (root / name / "impl" / "config.h").read_text()
        generated = emit_config_h(spec)
        # The emitter covers the header through rope_theta; compare that prefix.
        prefix = committed[: len(generated)]
        if prefix == generated:
            print(f"{name}: config.h prefix reproduced ({len(generated.splitlines())} lines)")
            continue
        failures += 1
        print(f"{name}: MISMATCH")
        for line in difflib.unified_diff(
            prefix.splitlines(), generated.splitlines(),
            fromfile="committed", tofile="generated", lineterm="", n=1,
        ):
            print("   ", line)
    return failures


if __name__ == "__main__":
    default = pathlib.Path(__file__).resolve().parents[4] / "csrc/src/serve/targets"
    sys.exit(main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else default))
