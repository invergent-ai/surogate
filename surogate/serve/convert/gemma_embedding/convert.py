"""Convert the EmbeddingGemma GGUF (``gemma-embedding``) into a ``.sinfer`` artifact.

GGUF-native, from ``ggml-org/embeddinggemma-300M-GGUF``. The object list is not
written down here: ``inventory.py`` derives it from the DSL declaration by
``generate/emit_inventory.py``, and ``recipe.py`` says only where each object
comes from. This module is the driver that puts the two together.

Almost all of the work is no work. ``Q8_0`` and the artifact's ``W8G32_F16S``
are the same format -- int8 codes with one binary16 scale per 32-value group --
so any object built by row algebra moves across bit-exactly, straight from a
memmap, with no dequantize and no GPU. That covers 169 of the 315 objects and
99.78% of the parameters. Only two kinds of object need real work:

* the six norms per layer, which lose a folded one (see ``recipe.py``);
* the embedding head, which composes two matrices and so has to dequantize.

Usage::

    python -m surogate.serve.convert.gemma_embedding.convert \\
        --gguf models/embeddinggemma-300M-Q8_0.gguf \\
        --frontend ~/.cache/huggingface/hub/models--google--embeddinggemma-300m/snapshots/<sha> \\
        --out ~/work/models/sinfer/embeddinggemma_300m.sinfer
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from surogate.serve.artifact.container import (
    RAW_BYTES_V1,
    ArtifactIdentity,
    ArtifactWriter,
    ResourceSpec,
    TensorSpec,
)
from surogate.serve.artifact.layouts import encode_direct, encode_row_split
from surogate.serve.artifact.numeric import get_format
from surogate.serve.convert.common.gguf_repack import REPACKABLE_TYPES
from surogate.serve.convert.qwen4exp.convert import GgufSource

from . import inventory, recipe
from .inventory import BF16, MODEL_ID, W8, WEIGHTS_ID
from .recipe import TensorRecipe

RECIPE_ID = "gemma-embedding-w8-v1"

_W8 = get_format(W8)


# --------------------------------------------------------------------------------------------
# Materialisation
# --------------------------------------------------------------------------------------------


def _bf16(x: np.ndarray, shape: tuple[int, ...]) -> bytes:
    tensor = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).reshape(shape)
    return encode_direct(tensor.to(torch.bfloat16), BF16)


def materialize(source: GgufSource, item: TensorRecipe, shape: tuple[int, ...]) -> bytes:
    """One artifact object's bytes."""

    names = item.tensors

    if item.op == "unfold":
        # The runtime re-applies the offset (rmsnorm unit_offset=true), so the
        # artifact holds w where the GGUF holds the folded 1 + w.
        return _bf16(source.float32(names[0]) - 1.0, shape)

    if item.op == "compose_linear":
        # Both sentence-transformers Dense modules declare Identity, so the pair
        # is one linear map. Composed in fp32, stored bf16: it rounds once where
        # the pair would round twice.
        second, first = (source.float32(n).astype(np.float64) for n in names)
        return _bf16((second @ first).astype(np.float32), shape)

    # Q8_0 straight into W8G32_F16S: same format, so repack the planes and never
    # dequantize.
    if source.tensor(names[0]).type_name not in REPACKABLE_TYPES:
        raise ValueError(
            f"{item.object_name}: {source.tensor(names[0]).type_name} "
            "is not exactly repackable"
        )
    codes, scales, (rows, k) = source.planes_exact(names[0])
    return encode_row_split(torch.from_numpy(codes), torch.from_numpy(scales), _W8, (rows, k))


def load_frontend(frontend_dir: Path) -> dict[str, bytes]:
    resources = {}
    for name in inventory.FRONTEND_RESOURCES:
        path = frontend_dir / name.split("/", 1)[1]
        if not path.exists():
            raise FileNotFoundError(f"{path} (needed for artifact object {name})")
        resources[name] = path.read_bytes()
    return resources


def convert(gguf: str | Path, frontend_dir: str | Path, out_path: str | Path) -> Path:
    started = time.perf_counter()
    source = GgufSource(Path(gguf))
    config = inventory.config_from_gguf(source)
    geometry = recipe.geometry_from_config(config)
    objects = inventory.declared_objects(geometry)
    recipes = {item.object_name: item for item in recipe.build_recipes(geometry)}
    resources = load_frontend(Path(frontend_dir))

    specs: list[TensorSpec | ResourceSpec] = inventory.tensor_specs(objects)
    specs += [
        ResourceSpec(name=name, encoding=RAW_BYTES_V1, bytes=len(payload))
        for name, payload in resources.items()
    ]

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    shapes = {obj["name"]: obj["shape"] for obj in objects}

    print(f"{len(objects)} objects from {Path(gguf).name}", flush=True)
    repacked = 0
    with ArtifactWriter(output, ArtifactIdentity(MODEL_ID, WEIGHTS_ID), specs) as writer:
        for index, spec in enumerate(specs, start=1):
            if isinstance(spec, ResourceSpec):
                writer.write(spec.name, resources[spec.name])
                continue
            item = recipes[spec.name]
            writer.write(spec.name, materialize(source, item, shapes[spec.name]))
            repacked += item.repackable
            if index % 50 == 0 or index == len(specs):
                print(f"[{index}/{len(specs)}] {spec.name}", flush=True)

    report = {
        "recipe_id": RECIPE_ID,
        "model_id": MODEL_ID,
        "weights_id": WEIGHTS_ID,
        "gguf": str(gguf),
        "frontend": str(frontend_dir),
        "objects": len(objects),
        "repacked_exactly": repacked,
        "elapsed_seconds": time.perf_counter() - started,
        "bytes": output.stat().st_size,
    }
    Path(str(output) + ".conversion.json").write_text(json.dumps(report, indent=2))
    print(
        f"converted in {report['elapsed_seconds']:.1f} s -> {output} "
        f"({report['bytes'] / 1e6:.0f} MB, {repacked} objects repacked bit-exactly)",
        flush=True,
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--frontend", required=True, help="directory holding tokenizer.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    convert(args.gguf, args.frontend, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
