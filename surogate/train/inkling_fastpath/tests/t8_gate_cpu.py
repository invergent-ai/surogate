"""the gate's comparison functions on synthetic tensors: a true match passes, and every vacuous or wrong comparison fails."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t8_gate as G  # noqa: E402


def make(segments, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {(s, L, k): torch.randn(3, 4, 2, generator=g) for s in segments for L in G.G7_LAYERS[s] for k in ("B13", "B2")}


def main():
    for segs in (("A", "B", "C", "D1", "D0"), ("A", "B")):
        ref = make(("A", "B", "C", "D1", "D0"))              # the reference always holds all five segments
        mine = {k: v.clone() for k, v in ref.items() if k[0] in segs}
        c = G.g7_compare(mine, ref, segs)
        assert c["pass"] and c["total"]["bitwise"], c
        near = {k: v * (1 + 1e-3) for k, v in mine.items()}
        assert G.g7_compare(near, ref, segs)["pass"], "0.1 % scale is inside the line"
        far = {k: v * 1.02 for k, v in mine.items()}
        assert not G.g7_compare(far, ref, segs)["pass"], "2 % is outside the line"
        assert not G.g7_compare({}, ref, segs)["pass"], "empty step"
        assert not G.g7_compare(mine, {}, segs)["pass"], "empty reference"
        assert not G.g7_compare({}, {}, segs)["pass"], "both empty"
        k0 = next(iter(mine))
        d = dict(mine); del d[k0]
        assert not G.g7_compare(d, ref, segs)["pass"], "missing key"
        d = dict(mine); d[("A", 2, "B13")] = torch.ones(3, 4, 2)
        assert not G.g7_compare(d, ref, segs)["pass"], "extra key"
        d = dict(mine); d[k0] = mine[k0].permute(1, 0, 2)
        assert not G.g7_compare(d, ref, segs)["pass"], "permuted shape"
        z = {k: torch.zeros_like(v) for k, v in mine.items()}
        zr = {k: torch.zeros_like(v) for k, v in ref.items()}
        assert not G.g7_compare(z, zr, segs)["pass"], "all zero on both sides"
        d = dict(mine); d[k0] = torch.zeros_like(mine[k0])
        r2 = dict(ref); r2[k0] = torch.zeros_like(ref[k0])
        assert not G.g7_compare(d, r2, segs)["pass"], "one tensor zero on both sides"
    assert not G.compare({}, {})["pass"]
    x = {"a": torch.ones(2, 3)}
    assert G.compare(x, {"a": x["a"].clone()})["pass"] and G.compare(x, {"a": x["a"].clone()})["bitwise"]
    assert not G.compare(x, {"a": torch.ones(3, 2)})["pass"]
    assert not G.compare({"a": torch.zeros(2)}, {"a": torch.zeros(2)})["pass"]
    assert not G.compare(x, {"a": x["a"] * float("nan")})["pass"]
    assert not G.compare(x, {"b": x["a"]})["pass"], "disjoint keys"
    print("T8 GATE COMPARE CPU TEST: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
