"""Convert the EmbeddingGemma GGUF (``gemma-embedding``) into a ``.sinfer`` artifact.

GGUF-native, from ``ggml-org/embeddinggemma-300M-GGUF``. The object list is not
written down here: it is derived from the DSL declaration by
``generate/emit_inventory.py``, and ``sources.py`` says only where each object
comes from. This module is the driver that puts the two together.

Almost all of the work is no work. ``Q8_0`` and the artifact's ``W8G32_F16S``
are the same format -- int8 codes with one binary16 scale per 32-value group --
so any object built by row algebra moves across bit-exactly, straight from a
memmap, with no dequantize and no GPU. That covers 121 of the 267 objects and
99.78% of the parameters. Only two kinds of object need real work:

* the six norms per layer, which lose a folded one (see ``sources.py``);
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
import sys
import time
from pathlib import Path
from typing import Any, Sequence

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

from . import CAPABILITIES, MODEL_ID, WEIGHTS_ID, source_for
from .sources import gguf_names

RECIPE_ID = "gemma-embedding-w8-v1"

BF16 = "BF16"
W8 = "W8G32_F16S"
CONTIGUOUS_LAYOUT = "contiguous-le-v1"
ROW_SPLIT_LAYOUT = "row-split-k128-v1"
_W8 = get_format(W8)

#: An embedding request carries no chat template and generates no tokens, so the
#: frontend is a tokenizer and nothing else.
#:
#: The SentencePiece model rather than `tokenizer.json`: upstream SentencePiece
#: reproduces the HF ids exactly on it, and it is 4.7 MB where the JSON is 33 MB.
#: `tokenizer_config.json` still travels for the special-token ids and the
#: sequence limit.
FRONTEND_RESOURCES = ("frontend/tokenizer.model", "frontend/tokenizer_config.json")


# --------------------------------------------------------------------------------------------
# The config the declaration is compiled against
# --------------------------------------------------------------------------------------------

#: What the GGUF does not carry, and what llama.cpp supplies from the
#: architecture identity instead (``models/gemma-embedding.cpp``). Every one is
#: silent when wrong, which is the case for keeping them in a declaration rather
#: than reading them from whatever file happens to be at hand.
DECLARED_NOT_IN_GGUF = {
    # swa_period = 6 there, as a default for an optional key this file omits.
    "_sliding_window_pattern": 6,
    # hparams.causal_attn = false, hardcoded for the architecture.
    "use_bidirectional_attention": True,
    # llama.cpp uses 1/sqrt(n_embd_head_k), which coincides with the real scalar
    # here because head_dim is also 256. It does not for Gemma3-27B.
    "query_pre_attn_scalar": 256,
}


def config_from_gguf(source: GgufSource) -> dict[str, Any]:
    """The HF-shaped config the DSL declaration compiles against.

    Read from the GGUF where the GGUF has it, declared where it does not.
    """

    def kv(key: str) -> Any:
        return source.fields[f"gemma-embedding.{key}"].contents()

    architecture = source.fields["general.architecture"].contents()
    if architecture != "gemma-embedding":
        raise ValueError(f"expected a gemma-embedding GGUF, got {architecture!r}")

    vocab, hidden = source.tensor("token_embd.weight").shape
    config = {
        "architectures": ["Gemma3TextModel"],
        "model_type": "gemma3_text",
        "vocab_size": int(vocab),
        "hidden_size": int(hidden),
        "num_hidden_layers": int(kv("block_count")),
        "num_attention_heads": int(kv("attention.head_count")),
        "num_key_value_heads": int(kv("attention.head_count_kv")),
        "intermediate_size": int(kv("feed_forward_length")),
        "max_position_embeddings": int(kv("context_length")),
        "head_dim": int(kv("attention.key_length")),
        "rms_norm_eps": float(kv("attention.layer_norm_rms_epsilon")),
        "sliding_window": int(kv("attention.sliding_window")),
        "rope_theta": float(kv("rope.freq_base")),
        "rope_local_base_freq": float(kv("rope.freq_base_swa")),
        **DECLARED_NOT_IN_GGUF,
    }
    if int(kv("pooling_type")) != 1:  # LLAMA_POOLING_TYPE_MEAN
        raise ValueError(f"expected mean pooling, got pooling_type {kv('pooling_type')}")
    return config


def inventory(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Every object the artifact stores, from the declaration."""

    generate = Path(__file__).resolve().parents[2] / "generate"
    if str(generate) not in sys.path:
        sys.path.insert(0, str(generate))
    import emit_inventory  # noqa: PLC0415

    return emit_inventory.inventory_for(
        "Gemma3TextModel", config, capabilities=set(CAPABILITIES)
    )


def tensor_specs(objects: Sequence[dict[str, Any]]) -> list[TensorSpec]:
    """Declared objects as artifact specs.

    The declaration says ``quantised`` and leaves the width to the target; every
    quantised object here comes from Q8_0, so W8 is the whole mapping.
    """
    out = []
    for obj in objects:
        fmt = W8 if obj["format"] == "quantised" else BF16
        layout = ROW_SPLIT_LAYOUT if fmt == W8 else CONTIGUOUS_LAYOUT
        out.append(TensorSpec(name=obj["name"], shape=obj["shape"], format=fmt, layout=layout))
    return out


# --------------------------------------------------------------------------------------------
# Materialisation
# --------------------------------------------------------------------------------------------


def _bf16(x: np.ndarray, shape: tuple[int, ...]) -> bytes:
    tensor = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).reshape(shape)
    return encode_direct(tensor.to(torch.bfloat16), BF16)


def materialize(source: GgufSource, name: str, shape: tuple[int, ...]) -> bytes:
    """One artifact object's bytes."""

    spec = source_for(name)
    names = gguf_names(name)

    if spec.op == "unfold":
        # The runtime re-applies the offset (rmsnorm unit_offset=true), so the
        # artifact holds w where the GGUF holds the folded 1 + w.
        return _bf16(source.float32(names[0]) - 1.0, shape)

    if spec.op == "compose_linear":
        # Both sentence-transformers Dense modules declare Identity, so the pair
        # is one linear map. Composed in fp32, stored bf16: it rounds once where
        # the pair would round twice.
        second, first = (source.float32(n).astype(np.float64) for n in names)
        return _bf16((second @ first).astype(np.float32), shape)

    # Q8_0 straight into W8G32_F16S: same format, so repack the planes and never
    # dequantize.
    if source.tensor(names[0]).type_name not in REPACKABLE_TYPES:
        raise ValueError(f"{name}: {source.tensor(names[0]).type_name} is not exactly repackable")
    codes, scales, (rows, k) = source.planes_exact(names[0])
    return encode_row_split(torch.from_numpy(codes), torch.from_numpy(scales), _W8, (rows, k))


def load_frontend(frontend_dir: Path) -> dict[str, bytes]:
    resources = {}
    for name in FRONTEND_RESOURCES:
        path = frontend_dir / name.split("/", 1)[1]
        if not path.exists():
            raise FileNotFoundError(f"{path} (needed for artifact object {name})")
        resources[name] = path.read_bytes()
    return resources


def convert(gguf: str | Path, frontend_dir: str | Path, out_path: str | Path) -> Path:
    started = time.perf_counter()
    source = GgufSource(Path(gguf))
    config = config_from_gguf(source)
    objects = inventory(config)
    resources = load_frontend(Path(frontend_dir))

    specs: list[TensorSpec | ResourceSpec] = tensor_specs(objects)
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
            writer.write(spec.name, materialize(source, spec.name, shapes[spec.name]))
            repacked += source_for(spec.name).repackable
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
