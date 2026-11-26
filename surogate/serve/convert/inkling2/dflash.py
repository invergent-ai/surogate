"""A DFlash checkpoint drafter for Inkling-2 (``--dflash-model``), stored as the family's ``dflash/*`` objects.

The objects, names and formats are the shared DFlash declaration's (``surogate/dsl/models/dflash.py``, bound by
``family::bind_dflash``): norms, conv base kernels and selector codebooks as BF16 copies, the projections W8. What is
Inkling's is where the drafter reads the target: the checkpoint names DSpark's four features -- the post-residual
features at visits 5 / 23 / 41 and the final-norm output after the last of the 48 visits (``48``) -- exactly as the
SGLang Inkling DFlash integration captures them (its DFlash capture is DSpark's), so its feature positions run to 49,
one past the 48 visits. The artifact states that bound (``dflash_geometry.feature_extent``); the engine relabels the
features to the visits that produce them (``targets/inkling2`` ``dflash_checkpoint_geometry``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from surogate.serve.artifact.container import TensorSpec
from surogate.serve.convert.common import dflash as common
from surogate.serve.convert.common.conversion import encode_tensor_payload
from surogate.serve.convert.common.inventory import W8, tensor_spec

from .checkpoint import Geometry

#: Inkling-2-Mini's DSpark features (visits 5 / 23 / 41, the final-norm output after visit 47): what a DFlash drafter
#: reads when the checkpoint carries no DSpark drafter to name them.
FEATURES = (5, 23, 41, 48)


@dataclass(frozen=True)
class Object:
    spec: TensorSpec          # the container's spec (ArtifactWriter)
    encoding: object          # the shared inventory's spec of the same object (encode_tensor_payload)
    sources: tuple[str, ...]  # checkpoint tensors, concatenated along rows in this order


def geometry(dflash_dir: str | Path, g: Geometry) -> common.Geometry:
    config = json.loads((Path(dflash_dir) / "config.json").read_text())
    visits = len(g.visits)
    target = SimpleNamespace(hidden=g.hidden, vocab=g.vocab_padded, layers=visits)
    d = common.geometry_from_config(config, target, feature_extent=visits + 1)
    features = tuple(g.draft.target_visits) if g.draft is not None else FEATURES
    if d.target_feature_layers != features:
        raise ValueError(f"an Inkling-2 DFlash drafter reads DSpark's features {features}, "
                         f"not {d.target_feature_layers}")
    if d.local_layers != d.layers:
        raise ValueError("an Inkling-2 DFlash drafter is served with sliding-window layers only")
    return d


def objects(d: common.Geometry, g: Geometry) -> list[Object]:
    C = g.hidden

    def obj(name, shape, fmt, *sources):
        encoding = tensor_spec(name, tuple(shape), fmt)
        return Object(TensorSpec(encoding.name, encoding.shape, encoding.format, encoding.layout), encoding,
                      tuple(sources))

    out = [obj("dflash/feature_projection", (C, d.feature_rows), W8, "fc.weight"),
           obj("dflash/context_norm", (C,), "BF16", "hidden_norm.weight")]
    for layer in range(d.layers):
        p, s = f"dflash/layers/{layer}/", f"layers.{layer}."
        out += [
            obj(p + "input_norm", (C,), "BF16", s + "input_layernorm.weight"),
            obj(p + "attention/query_key_value", (d.query_size + 2 * d.kv_size, C), W8,
                s + "self_attn.q_proj.weight", s + "self_attn.k_proj.weight", s + "self_attn.v_proj.weight"),
            obj(p + "attention/query_norm", (d.head_dim,), "BF16", s + "self_attn.q_norm.weight"),
            obj(p + "attention/key_norm", (d.head_dim,), "BF16", s + "self_attn.k_norm.weight"),
            obj(p + "attention/output", (C, d.query_size), W8, s + "self_attn.o_proj.weight"),
            obj(p + "post_attention_norm", (C,), "BF16", s + "post_attention_layernorm.weight"),
            obj(p + "mlp/gate_up", (2 * d.intermediate, C), W8, s + "mlp.gate_proj.weight", s + "mlp.up_proj.weight"),
            obj(p + "mlp/down", (C, d.intermediate), W8, s + "mlp.down_proj.weight"),
        ]
        if d.conv_kernel:
            for conv in ("attention_conv", "mlp_conv"):
                out += [obj(p + conv + "/base_kernel", (2, d.conv_kernel, C), "BF16", s + conv + ".base_kernel"),
                        obj(p + conv + "/kernel_projection", (d.conv_rows, C), W8,
                            s + conv + ".kernel_projection.weight")]
    out.append(obj("dflash/final_norm", (C,), "BF16", "norm.weight"))
    if d.selector_rank:
        out += [obj("dflash/selector/hidden_projection", (d.selector_rank, C), W8,
                    "candidate_selector.hidden_projection.weight"),
                obj("dflash/selector/predecessor_codebook", (g.vocab_padded, d.selector_rank), "BF16",
                    "candidate_selector.predecessor_codebook"),
                obj("dflash/selector/successor_codebook", (g.vocab_padded, d.selector_rank), "BF16",
                    "candidate_selector.successor_codebook")]
    return out


class Reader:
    """The drafter's safetensors files, every tensor read exactly once (``unread`` must end empty)."""

    def __init__(self, dflash_dir: str | Path):
        from safetensors import safe_open
        self._files = {}
        for path in sorted(Path(dflash_dir).glob("*.safetensors")):
            handle = safe_open(str(path), framework="pt", device="cpu")
            for name in handle.keys():
                if name in self._files:
                    raise ValueError(f"{name} is in more than one drafter file")
                self._files[name] = handle
        if not self._files:
            raise FileNotFoundError(f"no .safetensors in {dflash_dir}")
        self.unread = set(self._files)

    def get(self, name: str) -> torch.Tensor:
        if name not in self._files:
            raise KeyError(f"the drafter checkpoint has no {name}")
        self.unread.discard(name)
        return self._files[name].get_tensor(name)


def preflight(reader: Reader, objs: list[Object]) -> None:
    """Every drafter tensor is read by exactly one object, as BF16, with the declared shape."""
    problems, seen = [], set()
    for o in objs:
        rows = 0
        for name in o.sources:
            if name in seen:
                problems.append(f"{name} is read twice")
            seen.add(name)
            handle = reader._files.get(name)
            if handle is None:
                problems.append(f"{o.spec.name}: the drafter has no {name}")
                continue
            sl = handle.get_slice(name)
            if sl.get_dtype() != "BF16":
                problems.append(f"{name} is {sl.get_dtype()}, expected BF16")
            shape = tuple(sl.get_shape())
            if len(o.spec.shape) >= 2 and shape[1:] != tuple(o.spec.shape[1:]):
                problems.append(f"{name} {shape} does not fit {o.spec.name} {tuple(o.spec.shape)}")
            rows += shape[0] if len(o.sources) > 1 else 0
            if len(o.sources) == 1 and shape != tuple(o.spec.shape):
                problems.append(f"{name} {shape} != {o.spec.name} {tuple(o.spec.shape)}")
        if len(o.sources) > 1 and rows != o.spec.shape[0]:
            problems.append(f"{o.spec.name}: sources hold {rows} rows, object needs {o.spec.shape[0]}")
    missing = sorted(set(reader._files) - seen)
    problems += [f"drafter tensor not carried: {n}" for n in missing]
    if problems:
        raise ValueError("the DFlash drafter disagrees with its declaration:\n  " + "\n  ".join(problems[:20]))


def payload(reader: Reader, o: Object) -> bytes:
    parts = [reader.get(name) for name in o.sources]
    tensor = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{o.spec.name}: {tensor.dtype}, expected bfloat16")
    return encode_tensor_payload(tensor.reshape(o.spec.shape).contiguous(), o.encoding, "cpu")


__all__ = ["FEATURES", "Object", "Reader", "geometry", "objects", "payload", "preflight"]
