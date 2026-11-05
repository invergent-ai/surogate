"""The block-influence reader over the engine's residual-sample records (--residual-samples-out)."""

import json

import numpy as np
import pytest

from surogate.serve.tools.calibration import residual_influence as ri


def _write(directory, forwards, *, rows, streams, boundaries, drop_last_of=None, flag=None, interleave=False):
    """Records as the engine writes them (ops/calibration/residual_samples.cpp): `flag` = (forward,
    boundary) whose record says its token ids are not its forward's; `interleave` writes the
    forwards' records boundary-major, as a pipeline's stages produce them."""
    order = [(index, boundary) for index in range(len(forwards)) for boundary in range(boundaries)]
    if interleave:
        order.sort(key=lambda item: (item[1] + item[0], item[0]))
    with open(directory / "residual-samples.bin", "wb") as out:
        for index, boundary in order:
                columns = forwards[index]
                if index == drop_last_of and boundary == boundaries - 1:
                    continue
                values = columns[boundary].astype(np.float16)
                header = np.zeros(1, dtype=ri.HEADER)
                header["magic"] = b"RSAMPv1"
                header["forward"], header["boundary"], header["boundaries"] = index, boundary, boundaries
                header["layer"], header["rows"], header["count"] = 4 * boundary, rows, values.shape[0]
                header["flags"] = ri.IDS_DIFFER if flag == (index, boundary) else 0
                out.write(header.tobytes())
                out.write(np.arange(values.shape[0], dtype="<i4").tobytes())
                out.write(np.full(values.shape[0], 7, dtype="<i4").tobytes())
                out.write(values.tobytes())
    (directory / "residual-samples.json").write_text(json.dumps({"format": "surogate-residual-samples-v1"}))


@pytest.mark.parametrize("combine", ["sum", "concat"])
def test_influence_per_group(tmp_path, combine):
    rng = np.random.default_rng(3)
    streams, hidden, boundaries, count = 2, 8, 3, 5
    rows = streams * hidden
    forwards = []
    for _ in range(2):
        a = rng.normal(size=(count, rows))
        b = a.copy()                       # group 1 changes nothing: influence 0
        c = -b                             # group 2 flips the residual: influence 2
        forwards.append([a, b, c])
    _write(tmp_path, forwards, rows=rows, streams=streams, boundaries=boundaries)
    result = ri.influence(tmp_path, streams, combine)
    assert result["complete_forwards"] == 2 and result["samples"] == 2 * count
    assert result["groups"][0]["influence_mean"] == pytest.approx(0.0, abs=1e-3)
    assert result["groups"][1]["influence_mean"] == pytest.approx(2.0, abs=1e-3)
    assert result["lowest_influence_groups"] == [1, 2]


def test_incomplete_forwards_are_dropped(tmp_path):
    rng = np.random.default_rng(4)
    forwards = [[rng.normal(size=(3, 8)) for _ in range(3)] for _ in range(2)]
    _write(tmp_path, forwards, rows=8, streams=1, boundaries=3, drop_last_of=1)
    result = ri.influence(tmp_path, 1, "sum")
    assert result["forwards_seen"] == 2 and result["complete_forwards"] == 1 and result["samples"] == 3


def test_interleaved_and_flagged_forwards(tmp_path):
    """Records of several forwards in flight arrive interleaved; a flagged forward is left out."""
    rng = np.random.default_rng(5)
    forwards = []
    for _ in range(3):
        a = rng.normal(size=(4, 6))
        forwards.append([a, a.copy(), -a])
    _write(tmp_path, forwards, rows=6, streams=1, boundaries=3, interleave=True, flag=(2, 1))
    result = ri.influence(tmp_path, 1, "sum")
    assert result["forwards_seen"] == 3 and result["flagged_forwards"] == 1
    assert result["complete_forwards"] == 2 and result["samples"] == 8 and result["incomplete_forwards"] == 0
    assert result["groups"][0]["influence_mean"] == pytest.approx(0.0, abs=1e-3)
    assert result["groups"][1]["influence_mean"] == pytest.approx(2.0, abs=1e-3)
