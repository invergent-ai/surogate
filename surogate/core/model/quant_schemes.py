"""Per-module quantization schemes of a compressed-tensors checkpoint.

A compressed-tensors export declares what it quantized in
``config.json["quantization_config"]``: ``config_groups`` name the schemes
and the ``targets`` they apply to (a class name such as ``Linear``, an exact
module name, or a ``re:`` regex), and ``ignore`` names what was left alone.
That declaration is the authority. Tensor names are not: twenty exports of
one model quantize twenty different subsets, and every one spells the packed
tensor the same way.

Resolution runs the library's own matcher over module names taken from the
checkpoint itself. The names come from the checkpoint rather than from an
instantiated model because transformers 5 represents routed experts as one
fused module while the checkpoint stores them per expert, so a model's
``named_modules()`` never lists the modules the file actually holds. The
class a name needs for a class-name target comes from a meta-device skeleton
of the architecture (no memory, about a second), and for a leaf below a
fused module from the shape of what the file stores for it.

Checked on RedHatAI's Qwen3.6-35B-A3B NVFP4 (343 ``ignore`` entries, two of
them regexes): the resolution names exactly the 30,880 modules the file
packs, none more, none fewer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Iterator, Mapping

import torch
from torch import nn

from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
from compressed_tensors.utils.match import is_match, match_name


#: Parameter names compressed-tensors writes beside a quantized ``Linear``.
#: ``weight_packed`` is the one that says "this module was quantized".
PACKED_WEIGHT = "weight_packed"
_LINEAR_PARAMETERS = frozenset(
    {
        "weight",
        "bias",
        PACKED_WEIGHT,
        "weight_scale",
        "weight_global_scale",
        "weight_zero_point",
        "weight_shape",
        "input_scale",
        "input_global_scale",
        "input_zero_point",
        "output_scale",
        "output_zero_point",
    }
)


@dataclass(frozen=True)
class CheckpointModule:
    """One module as the checkpoint describes it."""

    name: str
    packed: bool
    #: Shape of ``weight`` or ``weight_packed`` if the module stores one.
    weight_shape: tuple[int, ...] | None


@dataclass(frozen=True)
class ModuleQuantization:
    module: CheckpointModule
    #: The scheme that applies, or ``None`` for a module the config leaves alone.
    scheme: QuantizationScheme | None
    #: Which ``config_groups`` entry matched, for the report.
    group: str | None


@dataclass(frozen=True)
class ResolvedQuantization:
    config: QuantizationConfig
    modules: dict[str, ModuleQuantization]

    def quantized(self) -> Iterator[ModuleQuantization]:
        return (item for item in self.modules.values() if item.scheme is not None)

    def check_against_checkpoint(self) -> None:
        """The config must explain every packed tensor and claim no unpacked one.

        A disagreement in either direction means the file is not what its
        config says, and nothing downstream should guess which side to
        believe.
        """

        claimed_not_packed = sorted(
            name for name, item in self.modules.items()
            if item.scheme is not None and item.scheme.weights is not None
            and not item.module.packed
        )
        packed_not_claimed = sorted(
            name for name, item in self.modules.items()
            if item.module.packed and (item.scheme is None or item.scheme.weights is None)
        )
        if not claimed_not_packed and not packed_not_claimed:
            return
        lines = ["quantization_config disagrees with the checkpoint's tensors:"]
        if claimed_not_packed:
            lines.append(
                f"  {len(claimed_not_packed)} module(s) the config quantizes have no "
                f"{PACKED_WEIGHT}, e.g. {claimed_not_packed[:3]}"
            )
        if packed_not_claimed:
            lines.append(
                f"  {len(packed_not_claimed)} module(s) with {PACKED_WEIGHT} match no "
                f"config group (or are ignored), e.g. {packed_not_claimed[:3]}"
            )
        raise ValueError("\n".join(lines))


# ---------------------------------------------------------------------------
# the checkpoint's own account of its modules
# ---------------------------------------------------------------------------


def read_tensor_shapes(model_dir: str | Path) -> dict[str, tuple[int, ...]]:
    """Every tensor name and shape in the checkpoint, from the shard headers.

    Reads headers only -- 8 bytes of length and a JSON directory per shard --
    never a tensor, so a 24 GB checkpoint answers in milliseconds.
    """

    model = Path(model_dir)
    index = model / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted({model / shard for shard in weight_map.values()})
    else:
        shards = sorted(model.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"{model}: no safetensors shards")
    shapes: dict[str, tuple[int, ...]] = {}
    for shard in shards:
        with shard.open("rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(length))
        for name, entry in header.items():
            if name != "__metadata__":
                shapes[name] = tuple(entry["shape"])
    return shapes


def checkpoint_modules(shapes: Mapping[str, tuple[int, ...]]) -> dict[str, CheckpointModule]:
    """Module names as the checkpoint implies them: each tensor's owner.

    A tensor ``a.b.c.weight_packed`` belongs to module ``a.b.c``; so does
    ``a.b.c.A_log``. Grouping by owner is what makes per-expert leaves
    (``experts.5.gate_proj``) visible even though no instantiated model
    lists them.
    """

    packed: set[str] = set()
    weight_shape: dict[str, tuple[int, ...]] = {}
    owners: set[str] = set()
    for name, shape in shapes.items():
        owner, _, parameter = name.rpartition(".")
        if not owner:
            continue
        owners.add(owner)
        if parameter == PACKED_WEIGHT:
            packed.add(owner)
            weight_shape[owner] = shape
        elif parameter == "weight" and owner not in weight_shape:
            weight_shape[owner] = shape
    return {
        owner: CheckpointModule(owner, owner in packed, weight_shape.get(owner))
        for owner in sorted(owners)
    }


# ---------------------------------------------------------------------------
# the architecture's account, for class-name targets
# ---------------------------------------------------------------------------


def build_meta_skeleton(model_dir: str | Path) -> nn.Module:
    """The architecture instantiated on the meta device: names and classes, no memory."""

    import transformers

    config = transformers.AutoConfig.from_pretrained(str(model_dir))
    factories = (
        getattr(transformers, "AutoModelForImageTextToText", None),
        transformers.AutoModelForCausalLM,
        transformers.AutoModel,
    )
    last: Exception | None = None
    with torch.device("meta"):
        for factory in factories:
            if factory is None:
                continue
            try:
                return factory.from_config(config)
            except (ValueError, KeyError, AttributeError) as error:
                last = error
    raise ValueError(f"{model_dir}: no transformers class builds this architecture") from last


def _class_for(
    module: CheckpointModule, skeleton: Mapping[str, nn.Module]
) -> nn.Module | None:
    """The module object a class-name target is matched against.

    From the skeleton where the name exists. A name the skeleton lacks is a
    leaf the architecture fuses (per-expert projections under one experts
    module): a leaf the file packs, or stores a rank-2 weight for, is a
    linear map and is matched as ``Linear``. Anything else has no class and
    can only be matched by name.
    """

    if module.name in skeleton:
        return skeleton[module.name]
    if module.packed or (module.weight_shape is not None and len(module.weight_shape) == 2):
        return _LINEAR_STAND_IN
    return None


_LINEAR_STAND_IN = nn.Linear(1, 1, bias=False)


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def quantization_config_of(config: Mapping[str, object]) -> QuantizationConfig | None:
    """The parsed ``quantization_config`` if it is a compressed-tensors one."""

    raw = config.get("quantization_config")
    if not isinstance(raw, Mapping) or raw.get("quant_method") != "compressed-tensors":
        return None
    return QuantizationConfig.model_validate(dict(raw))


def resolve_schemes(
    quantization: QuantizationConfig,
    modules: Mapping[str, CheckpointModule],
    skeleton: nn.Module,
) -> ResolvedQuantization:
    """Assign every checkpoint module its scheme, by the config's own rules.

    ``ignore`` is applied first and by name alone, because it needs no class:
    the modules a file holds that no skeleton describes (an MTP head stored
    beside the model, say) are still correctly left alone by a regex that
    names them. ``targets`` then needs a class only for class-name entries.
    A packed module the config neither ignores nor targets, and for which
    no class can be found, is an error rather than a guess.
    """

    named = dict(skeleton.named_modules())
    ignore = list(quantization.ignore or [])
    resolved: dict[str, ModuleQuantization] = {}
    for name, module in modules.items():
        if any(match_name(name, pattern) for pattern in ignore):
            resolved[name] = ModuleQuantization(module, None, None)
            continue
        candidate = _class_for(module, named)
        chosen: tuple[str, QuantizationScheme] | None = None
        if candidate is not None:
            for group, scheme in quantization.config_groups.items():
                if is_match(name, candidate, scheme.targets, ignore):
                    chosen = (group, scheme)
                    break
        if chosen is None and module.packed:
            raise ValueError(
                f"{name}: the checkpoint packs this module but quantization_config "
                f"neither ignores nor targets it"
                + ("" if candidate is not None else " (and no class is known for it)")
            )
        resolved[name] = (
            ModuleQuantization(module, chosen[1], chosen[0])
            if chosen is not None
            else ModuleQuantization(module, None, None)
        )
    return ResolvedQuantization(quantization, resolved)


def resolve_checkpoint(model_dir: str | Path) -> ResolvedQuantization | None:
    """Everything above, for a checkpoint on disk. ``None`` if not compressed-tensors."""

    model = Path(model_dir)
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    quantization = quantization_config_of(config)
    if quantization is None:
        return None
    resolved = resolve_schemes(
        quantization,
        checkpoint_modules(read_tensor_shapes(model)),
        build_meta_skeleton(model),
    )
    resolved.check_against_checkpoint()
    return resolved
