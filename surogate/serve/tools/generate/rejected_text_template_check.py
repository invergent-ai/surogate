"""Runs the REJECTED text-template approach; kept to reproduce its failure."""
from __future__ import annotations
import difflib, pathlib, sys
from rejected_text_template import to_template, render
from check_roundtrip import SPECS

ROOT = pathlib.Path(__file__).resolve().parents[4] / "csrc/src/serve/targets"
REL = "impl/load/bindings.cpp"
SOURCE = "qwen3_5_4b"

def main() -> int:
    template = to_template((ROOT / SOURCE / REL).read_text(), SPECS[SOURCE])
    failures = 0
    for name, spec in SPECS.items():
        committed = (ROOT / name / REL).read_text()
        got = render(template, spec)
        if got == committed:
            print(f"{name}: {REL} reproduced ({len(got.splitlines())} lines)")
            continue
        failures += 1
        diff = [l for l in difflib.unified_diff(committed.splitlines(), got.splitlines(),
                                                fromfile="committed", tofile="generated",
                                                lineterm="", n=0)
                if l[:1] in "+-" and l[:3] not in ("---", "+++")]
        print(f"{name}: MISMATCH, {len(diff)} lines")
        for l in diff[:14]:
            print("   ", l[:104])
    return failures

if __name__ == "__main__":
    sys.exit(main())
