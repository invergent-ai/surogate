"""Regenerates committed targets from the training DSL and reports any divergence.

This is the check that makes the DSL the single source of truth rather than
merely a second opinion. Each committed target is regenerated from
`surogate/dsl/models/*.py` plus the checkpoint's own `config.json`; the emitted
`config.h` must match the hand-written file byte for byte. A mismatch means the
declaration and the serving engine disagree about the model, and says which one
to go and read.

Targets migrate one at a time: a target listed here is derived, one that is not
is still hand-maintained, and the count below is the honest progress bar.

Run: python check_roundtrip.py [serve_targets_dir]
     QWEN3_5_0_8B_CONFIG=/path/to/config.json python check_roundtrip.py
"""

from __future__ import annotations

import difflib
import glob
import json
import os
import pathlib
import sys
from typing import NamedTuple

from emit_config import emit_config_h
from from_dsl import from_dsl


class Target(NamedTuple):
    """Where one target's checkpoint config comes from, and where it lands.

    `directory` exists because a target's C++ namespace and its source directory
    are not always the same word: `targets/gemma3/` holds `gemma3_270m::detail`,
    since one header describes one size and a 4B Gemma 3 will want its own.
    """

    env_var: str
    pattern: str
    directory: str = ""


#: serve target name (the C++ namespace) -> where to find an instance of it.
#: The declaration supplies the architecture; these supply which instance.
TARGETS = {
    # `qwen3_5` is absent: one target now serves every size of the family, and what it
    # compiles is the 2B as a reference size plus the family's vision tower, whose
    # dimensions appear in no text `config.json`. It is checked value-by-value instead.
    "gemma3_270m": Target(
        "GEMMA3_270M_CONFIG",
        "~/.cache/huggingface/hub/models--google--gemma-3-270m-it/snapshots/*/config.json",
        directory="gemma3",
    ),
}


def resolve_config(name: str) -> pathlib.Path | None:
    target = TARGETS[name]
    override = os.environ.get(target.env_var)
    if override:
        return pathlib.Path(override)
    matches = sorted(glob.glob(os.path.expanduser(target.pattern)))
    return pathlib.Path(matches[0]) if matches else None


def committed_config(root: pathlib.Path, name: str) -> pathlib.Path:
    """The committed header this target must reproduce."""

    return root / (TARGETS[name].directory or name) / "impl" / "config.h"


def main(root: pathlib.Path) -> int:
    failures = 0
    skipped = 0
    for name in TARGETS:
        config_path = resolve_config(name)
        if config_path is None or not config_path.exists():
            print(f"{name}: SKIPPED (no checkpoint config; set {TARGETS[name][0]})")
            skipped += 1
            continue

        hf_config = json.loads(config_path.read_text())
        architecture = (hf_config.get("architectures") or [hf_config.get("model_type")])[0]
        try:
            spec = from_dsl(architecture, hf_config, name=name)
        except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
            print(f"{name}: DECLARATION ERROR ({architecture}): {exc}")
            failures += 1
            continue

        committed = committed_config(root, name).read_text()
        generated = emit_config_h(spec)
        if committed == generated:
            lora = sum(1 for p in spec.params if p.is_lora_target)
            print(
                f"{name}: config.h reproduced in full ({len(generated.splitlines())} lines) "
                f"from {architecture}; contract carries {len(spec.params)} params, "
                f"{lora} adapter-addressable"
            )
            continue

        failures += 1
        print(f"{name}: MISMATCH (declaration and committed target disagree)")
        for line in difflib.unified_diff(
            committed.splitlines(), generated.splitlines(),
            fromfile="committed", tofile="from-dsl", lineterm="", n=1,
        ):
            print("   ", line)

    derived = len(TARGETS) - skipped - failures
    print(f"\n{derived}/{len(TARGETS)} listed targets derive cleanly from the DSL "
          f"({failures} mismatched, {skipped} skipped)")
    return failures


if __name__ == "__main__":
    default = pathlib.Path(__file__).resolve().parents[4] / "csrc/src/serve/targets"
    sys.exit(main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else default))
