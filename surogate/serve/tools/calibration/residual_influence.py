"""Block influence per layer group from the engine's residual samples (--residual-samples-out).

  python -m surogate.serve.tools.calibration.residual_influence DIR [--streams 4] [--combine sum|concat] [--json OUT]

Reads DIR/residual-samples.bin (format in DIR/residual-samples.json): per sampled column, the
residual entering every layer group and the final residual. Group g's block influence is
1 - cos(in_g, out_g) with in_g the residual entering it and out_g the one entering the next group
(the final residual for the last), averaged over the sampled columns (ShortGPT's measure, RECIPE.md
section 9.3). `--combine sum` (default) compares the streams' sum, the hyper-connection residual's
own combination; `concat` compares all streams side by side. Only forwards with a record for every
boundary are used, and none with a record the engine flagged (its token ids are not its forward's).
Forwards are evaluated as soon as their last boundary is read, so memory stays at the forwards
still open in the file, whatever its size.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HEADER = np.dtype([("magic", "S8"), ("forward", "<u8"), ("boundary", "<i4"), ("boundaries", "<i4"),
                   ("layer", "<i4"), ("rows", "<i4"), ("count", "<i4"), ("device", "<i4"),
                   ("column_base", "<i8"), ("flags", "<i4"), ("reserved", "<i4", (3,))])
IDS_DIFFER = 1


def records(path: Path):
    """(header, columns, token_ids, values [count, rows] float32) for every record in the file."""
    data = np.memmap(path, dtype=np.uint8, mode="r")
    offset = 0
    while offset + HEADER.itemsize <= data.size:
        header = np.frombuffer(data[offset:offset + HEADER.itemsize], dtype=HEADER)[0]
        if header["magic"] != b"RSAMPv1":
            raise ValueError(f"{path}: bad record at byte {offset}")
        offset += HEADER.itemsize
        count, rows = int(header["count"]), int(header["rows"])
        columns = np.frombuffer(data[offset:offset + 4 * count], dtype="<i4")
        offset += 4 * count
        tokens = np.frombuffer(data[offset:offset + 4 * count], dtype="<i4")
        offset += 4 * count
        values = np.frombuffer(data[offset:offset + 2 * count * rows], dtype="<f2").reshape(count, rows)
        offset += 2 * count * rows
        yield header, columns, tokens, values.astype(np.float32)


def influence(directory: Path, streams: int, combine: str) -> dict:
    meta = json.loads((directory / "residual-samples.json").read_text())
    open_forwards: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    flagged: set[int] = set()
    layer_of: dict[int, int] = {}
    boundaries = None
    per_group: list[list[np.ndarray]] = []
    complete = totals = 0
    seen: set[int] = set()
    for header, _, _, values in records(directory / "residual-samples.bin"):
        if boundaries is None:
            boundaries = int(header["boundaries"])
            per_group = [[] for _ in range(boundaries - 1)]
        forward, b = int(header["forward"]), int(header["boundary"])
        seen.add(forward)
        layer_of[b] = int(header["layer"])
        if int(header["flags"]) & IDS_DIFFER:
            flagged.add(forward)
        if combine == "sum":
            values = values.reshape(values.shape[0], streams, -1).sum(axis=1)
        f = open_forwards[forward]
        f[b] = values
        if len(f) < boundaries:
            continue
        del open_forwards[forward]
        if forward in flagged:
            continue
        for g in range(boundaries - 1):
            a, c = f[g].astype(np.float64), f[g + 1].astype(np.float64)
            cos = (a * c).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(c, axis=1), 1e-30)
            per_group[g].append(1.0 - cos)
        complete += 1
        totals += f[0].shape[0]
    if boundaries is None:
        raise ValueError("no records")
    result = {"directory": str(directory), "samples": totals, "complete_forwards": complete,
              "forwards_seen": len(seen), "flagged_forwards": len(flagged),
              "incomplete_forwards": len(open_forwards), "combine": combine, "groups": []}
    for g in range(boundaries - 1):
        values = np.concatenate(per_group[g]) if per_group[g] else np.zeros(0)
        result["groups"].append({"group": g + 1, "first_layer": layer_of.get(g), "influence_mean":
                                 float(values.mean()) if values.size else None,
                                 "influence_std": float(values.std()) if values.size else None})
    ranked = sorted((r for r in result["groups"] if r["influence_mean"] is not None),
                    key=lambda r: r["influence_mean"])
    result["lowest_influence_groups"] = [r["group"] for r in ranked]
    result["engine_summary"] = meta
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--combine", choices=("sum", "concat"), default="sum")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    result = influence(args.directory, args.streams, args.combine)
    text = json.dumps(result, indent=1)
    if args.json:
        args.json.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
