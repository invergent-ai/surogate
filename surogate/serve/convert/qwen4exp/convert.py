"""Convert the Qwen3.8-Flash-Next GGUF (`qwen4exp`) into a `.sinfer` artifact.

GGUF-native, and read in place: the four Unsloth shards are the only weight
source (there is no bridged HF checkpoint — the base model is 131 safetensors
shards on the Hub), and the artifact mostly does not copy them.  `recipe.py`
says where every object's rows come from; the shared planners in
`common/gguf_repack.py` turn those row programs into *runs* — stretches of a
shard the engine maps and reads directly — so the weights stay in the GGUF and
the artifact carries an index to them:

- the routed experts keep the format llama.cpp chose (Q4_K/Q5_K gate/up,
  Q5_1/Q8_0 down) and are served from it, which is both 120 B parameters of
  bytes never written and the end of the 0.55 % requantisation error the old
  dequantise-and-requantise path put on them;
- every Q8_0 projection is a run too: Q8_0 and W8G32_F16S hold the same
  numbers, so the loader rearranges the blocks into row-split planes on the
  device (`q8_0-to-w8g32`) and the kernels see what they always saw;
- the PLE table's 28.8 GB of IQ4_NL rows are one run.

What the artifact still stores is what the row algebra cannot say: the
hyper-connection, PLE and indexer projections (BF16 specs), the norms, and a
handful of genuine value transforms — `log(-x)`, the unfolded RMSNorm gamma,
the PLE hash parameters built from GGUF key-values.  That is ~1.5 GB.

The GGUF carries llama.cpp's *tiled* order of the GDN value heads; the engine
wants HF *grouped* order.  Along rows that inversion is a row permutation at
128-row granularity and rides in the runs; on `ssm_out` it permutes columns
instead and travels as the source's `col_groups` map.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from surogate.serve.artifact.container import ArtifactIdentity, ArtifactWriter
from surogate.serve.artifact.layouts import encode_direct
from surogate.serve.convert.common.gguf_repack import GgufRepackSource
from surogate.serve.convert.common.gguf_source import (
    GgufRecipeReader,
    GgufSource,
    candidate_sources as _candidate_sources,
    check_every_quantised_object_is_planned,
    encode_tensor,
)
from surogate.serve.convert.common import conversion as family_conversion

from . import inventory as inv
from . import recipe as rcp

RECIPE_ID = "qwen4exp-gguf-in-place-v1"
_GROUP = 32


# --------------------------------------------------------------------------------------------
# GGUF access
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# Row algebra (the transforms `recipe.py` cannot say, kept here beside their objects)
# --------------------------------------------------------------------------------------------


def untile_heads(x: np.ndarray, axis: int, k_heads: int, v_per_k: int, head_dim: int) -> np.ndarray:
    """llama.cpp's tiled V order [G0_v0, G1_v0, ..., G0_v1, ...] back to HF grouped order
    [G0_v0, G0_v1, ..., G1_v0, ...] along `axis` (length k_heads * v_per_k * head_dim)."""
    shape = list(x.shape)
    if axis < 0:
        axis += len(shape)
    if shape[axis] != k_heads * v_per_k * head_dim:
        raise ValueError(f"untile_heads: axis {axis} has {shape[axis]} != {k_heads * v_per_k * head_dim}")
    new_shape = shape[:axis] + [v_per_k, k_heads, head_dim] + shape[axis + 1 :]
    y = x.reshape(new_shape)
    y = np.swapaxes(y, axis, axis + 1)  # [k_heads, v_per_k, head_dim]
    return np.ascontiguousarray(y).reshape(shape)


def _untile_v(x: np.ndarray, axis: int) -> np.ndarray:
    """Un-tile an axis of value *channels* (48 heads x 128)."""
    return untile_heads(x, axis, inv.GDN_KEY_HEADS, inv.GDN_VALUE_HEADS // inv.GDN_KEY_HEADS, inv.GDN_HEAD_DIM)


def _untile_v_heads(x: np.ndarray, axis: int) -> np.ndarray:
    """Un-tile an axis of value *heads* (48 scalars: alpha/beta rows, a_log, dt_bias)."""
    return untile_heads(x, axis, inv.GDN_KEY_HEADS, inv.GDN_VALUE_HEADS // inv.GDN_KEY_HEADS, 1)


def value_column_groups() -> tuple[int, ...]:
    """Source quantisation group for each destination group of a value-tiled column axis.

    `ssm_out`'s columns are value heads, and 128 columns are exactly four groups of 32, so the
    un-tiling moves whole groups: it travels as this map beside the runs rather than forcing
    the tensor to be decoded and rebuilt.
    """
    columns = _untile_v(np.arange(inv.GDN_VALUE_DIM, dtype=np.int64).reshape(1, -1), 1)[0]
    groups = columns.reshape(-1, _GROUP)
    if not np.all(groups[:, 1:] == groups[:, :1] + np.arange(1, _GROUP)):
        raise ValueError("the value un-tiling splits a quantisation group")
    return tuple(int(first) // _GROUP for first in groups[:, 0])


def attention_rows_q_k_gate_v() -> tuple[list[int], list[int], list[int], list[int]]:
    """Row indices into the fused source [q_proj rows (12288) | k (512) | v (512)] for the
    engine's q | k | gate | v order. q_proj interleaves [q_h(256) | gate_h(256)] per head."""
    q, gate = [], []
    for head in range(inv.QUERY_HEADS):
        base = head * 2 * inv.HEAD_DIM
        q.extend(range(base, base + inv.HEAD_DIM))
        gate.extend(range(base + inv.HEAD_DIM, base + 2 * inv.HEAD_DIM))
    return q, [], gate, []


# --------------------------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------------------------


def bf16_bytes(x: np.ndarray, shape: tuple[int, ...]) -> bytes:
    tensor = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).reshape(shape)
    return encode_direct(tensor.to(torch.bfloat16), inv.BF16)


def f32_bytes(x: np.ndarray, shape: tuple[int, ...]) -> bytes:
    return encode_direct(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).reshape(shape), inv.FP32)


def i32_bytes(values: Sequence[int]) -> bytes:
    return encode_direct(torch.tensor(list(values), dtype=torch.int32), inv.I32)


# --------------------------------------------------------------------------------------------
# The objects no recipe can say
# --------------------------------------------------------------------------------------------


def _ple_hash_parameters(source: GgufSource, leaf: str) -> bytes:
    if leaf == "multipliers":
        words: list[int] = []
        for m in source.array_field("qwen4exp.ple.layer_multipliers"):
            words.extend((m & 0xFFFFFFFF, (m >> 32) & 0xFFFFFFFF))
        return i32_bytes([w - (1 << 32) if w >= (1 << 31) else w for w in words])
    return i32_bytes(source.array_field(f"qwen4exp.ple.{leaf}"))


def materialize_unspoken(source: GgufSource, name: str) -> bytes:
    """The six objects `recipe.py` lists in `MATERIALIZED_OBJECTS`."""
    leaf = name.rsplit("/", 1)[-1]
    if name.startswith("text/ple/"):
        return _ple_hash_parameters(source, leaf)
    # `text/layers/N/...` names its block; the MTP head is the one past the trunk.
    blk = (f"blk.{inv.LAYERS}." if name.startswith("mtp/")
           else f"blk.{name.split('/')[2]}.")
    if name.endswith(("attention/query_norm", "attention/key_norm")):
        # The runtime applies these with a unit offset (HF gamma w, norm = (1+w)*x); the GGUF
        # carries the folded gamma 1+w. Every other norm keeps the folded form.
        return bf16_bytes(source.float32(blk + f"attn_{leaf[0]}_norm.weight") - 1.0, (inv.HEAD_DIM,))
    if name.endswith("gdn/a_log"):
        values = source.float32(blk + "ssm_a").reshape(inv.GDN_VALUE_HEADS)
        return f32_bytes(np.log(-_untile_v_heads(values, 0)), (inv.GDN_VALUE_HEADS,))
    if name.endswith("gdn/dt_bias"):
        values = source.float32(blk + "ssm_dt.bias").reshape(inv.GDN_VALUE_HEADS)
        return f32_bytes(_untile_v_heads(values, 0), (inv.GDN_VALUE_HEADS,))
    raise KeyError(f"no materialiser covers artifact object {name!r}")


# --------------------------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------------------------


def candidate_sources(source: GgufSource) -> dict[str, dict]:
    """The repack planner's candidate map. One tensor of this model has its columns in another
    order -- the GDN output projection follows llama.cpp's tiled value heads -- and its map
    travels with the candidate so the runtime permutes the activation instead."""
    return _candidate_sources(source, {"ssm_out.weight": value_column_groups()})


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def convert(gguf: str | Path, frontend_dir: str | Path, out_path: str | Path,
            *, device: str = "cuda", mtp: str | Path | None = None) -> Path:
    """Write the artifact. `device` is accepted so the ingest path can call every converter
    the same way; nothing here needs a GPU, because nothing here quantises any more.

    `mtp` is the NextN draft head's own GGUF (`mtp-*-shared-*.gguf`). It is read in
    place like the shards, so passing it costs an index entry, not a copy.
    """
    started = time.perf_counter()
    del device
    inv.validate_inventory()
    rcp.validate_recipe_coverage()
    source = GgufSource(Path(gguf), (Path(mtp),) if mtp is not None else ())
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    table = source.tensor(rcp.PLE_TABLE_SOURCE)
    if table.shape != (inv.PLE_TABLE_ROWS, inv.PLE_HEAD_DIM) or table.type_name != inv.PLE_TABLE_FORMAT:
        raise ValueError(f"unexpected PLE table {table.shape} {table.type_name}")
    # The tower is not reachable from this source. Every published GGUF export of this model
    # drops it (the four-shard Q4_K_XL set has 1,224 tensors and none of them vision), and the
    # recipes name GGUF tensors, so an export that did carry one would need its own mapping
    # rather than a silent text-only artifact.
    if any(name.startswith(("v.", "mm.")) or "vision" in name for name in source.tensors):
        raise NotImplementedError(
            "this GGUF carries a vision tower; the qwen4exp recipes cover the text stack only"
        )
    # The head is present when its GGUF was passed; `blk.48` is the proof, since a trunk
    # shard stops at 47.
    has_mtp = f"blk.{inv.LAYERS}.nextn.eh_proj.weight" in source.tensors
    if mtp is not None and not has_mtp:
        raise ValueError(f"{mtp} carries no blk.{inv.LAYERS} NextN head")
    tensor_specs, object_specs = inv.active_specs(vision=False, mtp=has_mtp)

    recipes = dict(rcp.RECIPES_BY_NAME)
    recipes[inv.PLE_TABLE_RESOURCE] = rcp.PLE_TABLE_RECIPE
    repack = GgufRepackSource.from_sources(source.shards, candidate_sources(source))

    # Two ways to read a weight where it lies. A K-quant or a 32-value block the kernels
    # already decode is served in the file's own format; a Q8_0 whose op wants the row-split
    # W8 planes is read just the same and rearranged on the device, which is why the exclude
    # list and the in-place list are one list.
    native = repack.plan_native(recipes, tensor_specs,
                                exclude_suffixes=rcp.NATIVE_EXCLUDE_SUFFIXES)
    in_place = repack.plan_repack_in_place(recipes, tensor_specs, rcp.NATIVE_EXCLUDE_SUFFIXES)
    check_every_quantised_object_is_planned(tensor_specs, set(native) | set(in_place), inv.W8)

    native_specs = {spec.name: spec for spec in
                    GgufRepackSource.native_specs(tensor_specs, native)}
    native_runs = {name: repack.runs_for_native(native_specs[name], recipes[name], None)
                   for name in native}
    # The PLE table is already declared in the format the file holds it in, so it needs no
    # rewrite -- only its one run.
    native[inv.PLE_TABLE_RESOURCE] = inv.PLE_TABLE_FORMAT
    native_runs[inv.PLE_TABLE_RESOURCE] = repack.runs_for_native(
        inv.PLE_TABLE_SPEC, rcp.PLE_TABLE_RECIPE, None
    )

    object_specs = GgufRepackSource.in_place_specs(object_specs, in_place)
    object_specs = GgufRepackSource.native_specs(object_specs, native, native_runs)
    external = tuple((str(path.resolve()), path.stat().st_size) for path in source.shards)

    read_in_place = sum(
        sum(run[2] for run in getattr(spec, "runs", ())) for spec in object_specs
    )
    run_count = sum(len(getattr(spec, "runs", ())) for spec in object_specs)
    print(f"read in place: {len(native_runs) + len(in_place)} objects, {run_count} runs, "
          f"{read_in_place / 1e9:.1f} GB never copied", flush=True)

    frontend = family_conversion.load_resources(frontend_dir, inv.RESOURCE_SPECS)
    resources = {item.name: item.data for item in frontend}
    plan = family_conversion.build_object_plan(object_specs, resources)
    reader = GgufRecipeReader(source)

    specs = list(object_specs)
    total = len(specs)
    print(f"converting {total} objects from {len(source.shards)} shards", flush=True)
    with ArtifactWriter(
        output,
        ArtifactIdentity(inv.MODEL_ID, inv.WEIGHTS_ID),
        plan.specs,
        external=external,
    ) as writer:
        for index, spec in enumerate(specs, start=1):
            t0 = time.perf_counter()
            if getattr(spec, "runs", ()):
                continue  # served from the GGUF; the artifact carries no bytes for it
            if isinstance(spec, inv.ResourceSpec):
                payload: bytes = resources[spec.name]
            elif spec.name in recipes:
                tensor = rcp.materialize_recipe(recipes[spec.name], reader)
                payload = encode_tensor(tensor, spec)
                del tensor
            else:
                payload = materialize_unspoken(source, spec.name)
            writer.write(spec.name, payload)
            del payload
            if index % 100 == 0 or index == total:
                print(f"[{index}/{total}] {spec.name} ({time.perf_counter() - t0:.1f}s)", flush=True)

    report = {
        "recipe_id": RECIPE_ID,
        "model_id": inv.MODEL_ID,
        "weights_id": inv.WEIGHTS_ID,
        "gguf": [str(path) for path in source.shards],
        "frontend": str(frontend_dir),
        "elapsed_seconds": time.perf_counter() - started,
        "bytes": output.stat().st_size,
        "read_in_place_bytes": int(read_in_place),
        "runs": int(run_count),
        "objects_read_in_place": len(native_runs) + len(in_place),
        "objects_materialized": total - len(native_runs) - len(in_place),
    }
    Path(str(output) + ".conversion.json").write_text(json.dumps(report, indent=2))
    source.close()
    print(f"conversion finished in {report['elapsed_seconds']:.0f} s -> {output} "
          f"({report['bytes'] / 1e9:.2f} GB written, {read_in_place / 1e9:.1f} GB read in place)",
          flush=True)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True, help="first shard of the split GGUF")
    parser.add_argument("--frontend", required=True, help="directory with tokenizer/chat template/configs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mtp", default=None,
                        help="the NextN draft head's GGUF (mtp-*-shared-*.gguf), read in place")
    parser.add_argument("--device", default="cuda",
                        help="accepted for a uniform call across the converters; unused")
    args = parser.parse_args(argv)
    convert(args.gguf, args.frontend, args.out, device=args.device, mtp=args.mtp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
