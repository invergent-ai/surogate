#!/usr/bin/env python3
"""
Merge LoRA adapter into base model by operating directly on safetensors files.

Copies the original base model files, then applies LoRA deltas in-place,
preserving the exact original key structure for compatibility with vLLM
and other serving frameworks.

Every LoRA pair in the adapter must land somewhere in the base checkpoint. The
whole merge is planned (and every target shape checked) before anything is
written, and the output is staged next to ``output_path`` and moved into place
only once every shard is written, so a failed merge leaves no output behind.
"""

import glob
import json
import math
import os
import re
import shutil
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from surogate.utils.logger import get_logger

logger = get_logger()


class AdapterMergeError(ValueError):
    """The adapter cannot be merged into this base model. Nothing was written."""


def load_adapter_weights(adapter_path: str) -> dict[str, torch.Tensor]:
    """Load LoRA adapter weights from safetensors."""
    adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        raise FileNotFoundError(f"Adapter file not found: {adapter_file}")

    logger.info(f"Loading adapter weights from {adapter_file}...")
    weights = load_file(adapter_file)
    logger.info(f"Loaded {len(weights)} adapter tensors")
    return weights


def _strip_adapter_key(key: str) -> str:
    """Strip PEFT prefix and .weight suffix from an adapter key.

    Input:  base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    Output: model.layers.0.self_attn.q_proj.lora_A
    """
    if key.startswith("base_model.model."):
        key = key[len("base_model.model.") :]
    if key.endswith(".weight"):
        key = key[: -len(".weight")]
    return key


# ---------------------------------------------------------------------------
# Where a LoRA pair lands
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusedExpertLayout:
    """How a checkpoint stores its routed experts when they are fused per layer.

    Some checkpoints have no per-expert ``<moe>.experts.{e}.{gate,up,down}_proj.weight``
    tensors but one ``<moe>.experts.gate_up_proj`` and one ``<moe>.experts.down_proj``
    per layer. The default is the transformers convention, the layout Qwen3.5/3.6-MoE
    checkpoints ship (``surogate/dsl/models/qwen3_5_moe.py``; the serving engine's
    ``lora_bind.h`` binds a BF16 checkpoint's expert gate rows first the same way):

      gate_up_proj  [E, 2*M, C]  gate rows first (``gate, up = linear(x, W[e]).chunk(2, -1)``)
      down_proj     [E, C,   M]

    i.e. each expert is ``[out, in]``, like ``nn.Linear.weight``. GPT-OSS stores each expert
    ``[in, out]`` (``x @ W[e]``, hence the DSL's ``transform(..., fn="transpose")``) and
    interleaves gate and up along the output axis (``gate = h[..., ::2]``, ``up = h[..., 1::2]``):

      gate_up_proj  [E, C, 2*M]
      down_proj     [E, M, C]
    """

    transposed: bool = False  # each expert is stored [in, out]
    interleaved: bool = False  # gate/up alternate along the output axis instead of gate-first halves


#: Fused-expert layouts that differ from the default, by HF ``model_type``.
_FUSED_EXPERT_LAYOUTS: dict[str, FusedExpertLayout] = {
    "gpt_oss": FusedExpertLayout(transposed=True, interleaved=True),
}


def fused_expert_layout(config: dict | None) -> FusedExpertLayout:
    """The fused-expert layout of a checkpoint, from its ``config.json``."""
    config = config or {}
    for cfg in (config, config.get("text_config") or {}):
        layout = _FUSED_EXPERT_LAYOUTS.get(cfg.get("model_type") or "")
        if layout is not None:
            return layout
    return FusedExpertLayout()


@dataclass(frozen=True)
class LoRAMergeTarget:
    """Where one LoRA pair's delta ``scaling * B @ A`` (shape ``[out, in]``) is added.

    ``key`` is the base safetensors tensor. For a fused experts tensor ``expert`` selects
    the expert along dim 0. ``transposed`` means the (expert's) matrix is stored
    ``[in, out]``; ``rows`` then selects the output rows the delta covers.
    ``swap_halves`` exchanges the delta's two row halves first: surogate's fused
    ``gate_up_proj`` adapters are ``[up; gate]`` (the runtime's row order) while a
    gate-first checkpoint is ``[gate; up]``.
    """

    key: str
    expert: int | None = None
    rows: slice | None = None  # None: every output row
    transposed: bool = False
    swap_halves: bool = False

    def describe(self) -> str:
        where = self.key if self.expert is None else f"{self.key}[{self.expert}]"
        if self.rows is not None:
            start, stop, step = (self.rows.start or 0), self.rows.stop, self.rows.step
            span = f"{start}:{'' if stop is None else stop}{'' if step in (None, 1) else f':{step}'}"
            where += f" {'columns' if self.transposed else 'rows'} {span}"
        elif self.transposed:
            where += " (transposed)"
        return where


#: A routed expert's projection in surogate's adapter naming: ``<moe>.experts.{e}.<proj>``.
_EXPERT_MODULE = re.compile(
    r"^(?P<experts>.+\.experts)\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj|gate_up_proj)$"
)

#: The shared expert is ``shared_experts`` in surogate's adapters (and in DeepSeek, GLM and
#: Nemotron-H checkpoints) but ``shared_expert`` in Qwen2/3-MoE-style checkpoints
#: (Qwen3-Next, Qwen3.5/3.6-MoE). The serving engine accepts both (family/impl/lora_bind.h).
_MODULE_ALIASES: tuple[tuple[str, str], ...] = (
    (".shared_experts.", ".shared_expert."),
    (".shared_expert.", ".shared_experts."),
)


@dataclass(frozen=True)
class _Candidate:
    key: str  # base tensor name before any prefix remap
    proj: str | None = None  # fused expert projection, None for a plain linear weight
    expert: int | None = None


def _candidates(module: str) -> list[_Candidate]:
    """Base tensors a LoRA module may target, most specific first."""
    names = [module]
    for old, new in _MODULE_ALIASES:
        if old in module:
            names.append(module.replace(old, new))
    candidates = [_Candidate(name + ".weight") for name in names]
    match = _EXPERT_MODULE.match(module)
    if match:
        proj = match["proj"]
        fused = match["experts"] + (".down_proj" if proj == "down_proj" else ".gate_up_proj")
        candidates.append(_Candidate(fused, proj, int(match["expert"])))
    return candidates


def _make_target(candidate: _Candidate, key: str, shape: tuple[int, ...], layout: FusedExpertLayout) -> LoRAMergeTarget:
    if candidate.proj is None:
        return LoRAMergeTarget(key)
    if candidate.proj == "down_proj":
        return LoRAMergeTarget(key, expert=candidate.expert, transposed=layout.transposed)
    # Output rows of one expert's fused gate+up projection (the shape check rejects a non-3D tensor).
    rows = (shape[2] if layout.transposed else shape[1]) if len(shape) == 3 else 0
    half = rows // 2
    if layout.interleaved:
        selected = {"gate_proj": slice(0, None, 2), "up_proj": slice(1, None, 2), "gate_up_proj": None}
    else:
        selected = {"gate_proj": slice(0, half), "up_proj": slice(half, rows), "gate_up_proj": None}
    return LoRAMergeTarget(
        key,
        expert=candidate.expert,
        rows=selected[candidate.proj],
        transposed=layout.transposed,
        swap_halves=candidate.proj == "gate_up_proj" and not layout.interleaved,
    )


def _target_shape(target: LoRAMergeTarget, shape: tuple[int, ...]) -> tuple[int, int] | None:
    """The ``[out, in]`` shape the delta must have to land on ``target``, or None if it cannot."""
    if target.expert is not None:
        if len(shape) != 3 or not 0 <= target.expert < shape[0]:
            return None
        shape = shape[1:]
    if len(shape) != 2:
        return None
    out_features, in_features = (shape[1], shape[0]) if target.transposed else shape
    rows = range(out_features) if target.rows is None else range(out_features)[target.rows]
    return len(rows), in_features


def _suffix_index(keys) -> dict[str, list[str]]:
    """Every base key under each of its dot-separated suffixes (itself included)."""
    index: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        index[key].append(key)
        pos = key.find(".")
        while pos != -1:
            index[key[pos + 1 :]].append(key)
            pos = key.find(".", pos + 1)
    return index


def _probe_suffix(expected_st_key: str) -> str:
    return expected_st_key.split(".", 1)[1] if "." in expected_st_key else expected_st_key


def _select_prefix_probe(expected_st_key: str, suffix_index: dict[str, list[str]]) -> str | None:
    """Resolve one adapter key under another prefix without confusing the language model with MTP."""
    matches = suffix_index.get(_probe_suffix(expected_st_key), [])
    if expected_st_key in matches:
        return expected_st_key
    if not matches:
        return None

    if expected_st_key.startswith("model.layers."):
        language_model = [key for key in matches if key.startswith("model.language_model.layers.")]
        if len(language_model) == 1:
            return language_model[0]
    if expected_st_key.startswith("mtp.layers."):
        mtp = [key for key in matches if key.startswith("mtp.layers.")]
        if len(mtp) == 1:
            return mtp[0]

    if len(matches) == 1:
        return matches[0]
    raise AdapterMergeError(f"ambiguous LoRA target for {expected_st_key}: {matches}")


def _apply_remap(key: str, remap: tuple[str, str]) -> str | None:
    from_pfx, to_pfx = remap
    if not key.startswith(from_pfx):
        return None
    return to_pfx + key[len(from_pfx) :]


@dataclass
class LoRAMergePlan:
    """Every LoRA pair of an adapter, resolved against a base checkpoint."""

    #: base key -> [(target, lora_A, lora_B, adapter module)]
    targets: dict[str, list[tuple[LoRAMergeTarget, torch.Tensor, torch.Tensor, str]]] = field(default_factory=dict)
    #: adapter modules with no target in the base checkpoint
    missing: list[str] = field(default_factory=list)
    #: (adapter prefix, checkpoint prefix) pairs found while resolving
    prefix_remaps: list[tuple[str, str]] = field(default_factory=list)
    num_pairs: int = 0

    @property
    def num_resolved(self) -> int:
        return sum(len(v) for v in self.targets.values())

    @property
    def num_fused(self) -> int:
        return sum(1 for v in self.targets.values() for t, *_ in v if t.expert is not None)


def _lora_pairs(adapter_weights: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Group the adapter into module -> (lora_A, lora_B); reject what cannot be merged."""
    stripped = {_strip_adapter_key(key): tensor for key, tensor in adapter_weights.items()}
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    unpaired, unsupported, ignored = [], [], []
    for key in stripped:
        if key.endswith(".lora_A"):
            module = key[: -len(".lora_A")]
            if module + ".lora_B" in stripped:
                pairs[module] = (stripped[key], stripped[module + ".lora_B"])
            else:
                unpaired.append(key)
        elif key.endswith(".lora_B"):
            if key[: -len(".lora_B")] + ".lora_A" not in stripped:
                unpaired.append(key)
        elif ".lora_" in key:
            unsupported.append(key)
        else:
            ignored.append(key)
    if unpaired:
        raise AdapterMergeError(
            f"{len(unpaired)} LoRA tensors have no matching lora_A/lora_B half, e.g. {unpaired[:4]}"
        )
    if unsupported:
        raise AdapterMergeError(
            f"{len(unsupported)} adapter tensors are not plain LoRA A/B weights and cannot be merged "
            f"(e.g. {unsupported[:4]})"
        )
    if ignored:
        logger.warning(f"Ignoring {len(ignored)} non-LoRA adapter tensors, e.g. {ignored[:4]}")
    return pairs


def _plan_lora_merge(
    adapter_weights: dict[str, torch.Tensor],
    base_shapes: dict[str, tuple[int, ...]],
    layout: FusedExpertLayout | None = None,
) -> LoRAMergePlan:
    """Resolve every LoRA pair of the adapter to the base tensor (and slice of it) it adapts.

    A pair ``<module>`` lands on ``<module>.weight``, on its shared-expert alias, or, for a
    routed expert ``<moe>.experts.{e}.<proj>`` of a checkpoint that fuses its experts, on
    expert ``e`` of ``<moe>.experts.gate_up_proj`` / ``<moe>.experts.down_proj``.

    The adapter may name layers under another prefix than the checkpoint (surogate writes
    ``model.layers.*``; a Qwen3.5 checkpoint has ``model.language_model.layers.*``). Such a
    remap is detected from whichever pair first fails to resolve as named, and reused for
    the rest; a pair no known remap resolves is probed again, so one unresolvable pair
    (an expert of a fused checkpoint, say) cannot hide the remap from the others.

    Raises AdapterMergeError on an ambiguous target or a delta whose shape does not fit.
    Pairs without any target are listed in ``plan.missing``.
    """
    layout = layout or FusedExpertLayout()
    pairs = _lora_pairs(adapter_weights)
    plan = LoRAMergePlan(num_pairs=len(pairs))
    suffix_index: dict[str, list[str]] | None = None
    bad_shapes: list[str] = []

    for module, (lora_a, lora_b) in pairs.items():
        candidates = _candidates(module)
        resolved: tuple[_Candidate, str] | None = None
        for remap in [("", ""), *plan.prefix_remaps]:
            for candidate in candidates:
                key = _apply_remap(candidate.key, remap)
                if key is not None and key in base_shapes:
                    resolved = (candidate, key)
                    break
            if resolved:
                break
        if resolved is None:
            if suffix_index is None:
                suffix_index = _suffix_index(base_shapes)
            for candidate in candidates:
                matched = _select_prefix_probe(candidate.key, suffix_index)
                if matched is None:
                    continue
                suffix = _probe_suffix(candidate.key)
                remap = (candidate.key[: len(candidate.key) - len(suffix)], matched[: len(matched) - len(suffix)])
                if remap not in plan.prefix_remaps:
                    plan.prefix_remaps.append(remap)
                resolved = (candidate, matched)
                break
        if resolved is None:
            plan.missing.append(module)
            continue

        candidate, key = resolved
        target = _make_target(candidate, key, base_shapes[key], layout)
        expected = _target_shape(target, base_shapes[key])
        delta = (lora_b.shape[0], lora_a.shape[-1]) if lora_a.dim() == 2 and lora_b.dim() == 2 else None
        if (
            delta is None
            or lora_a.shape[0] != lora_b.shape[1]
            or expected != delta
            or (target.swap_halves and delta[0] % 2)
        ):
            a_shape, b_shape = tuple(lora_a.shape), tuple(lora_b.shape)
            bad_shapes.append(
                f"{module} (lora_A {a_shape}, lora_B {b_shape}) -> {target.describe()} "
                f"of {tuple(base_shapes[key])} needs a delta of {expected}"
            )
            continue
        plan.targets.setdefault(key, []).append((target, lora_a, lora_b, module))

    if bad_shapes:
        raise AdapterMergeError(
            f"{len(bad_shapes)} LoRA pairs do not fit the base tensor they target "
            f"(is this adapter for this base model?): " + "; ".join(bad_shapes[:4])
        )
    return plan


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def _lora_delta(lora_A: torch.Tensor, lora_B: torch.Tensor, scaling: float) -> torch.Tensor:
    # lora_A: [rank, in_features], lora_B: [out_features, rank]; computed in float32 for accuracy
    return (lora_B.float() @ lora_A.float()) * scaling


def merge_lora_into_linear(
    base_weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    lora_alpha: float,
    lora_rank: int,
    scaling: float | None = None,
) -> torch.Tensor:
    """Merge LoRA weights into base linear layer: W' = W + (B @ A) * scaling."""
    if scaling is None:
        scaling = lora_alpha / lora_rank

    orig_dtype = base_weight.dtype
    merged = base_weight.float() + _lora_delta(lora_A, lora_B, scaling)
    return merged.to(orig_dtype)


def _add_delta(weight: torch.Tensor, target: LoRAMergeTarget, delta: torch.Tensor) -> None:
    """Add a ``[out, in]`` delta in place to a float32 ``[out, in]`` (or ``[in, out]``) matrix."""
    view = weight.t() if target.transposed else weight
    if target.swap_halves:
        first, second = delta.chunk(2, dim=0)
        delta = torch.cat([second, first], dim=0)
    if target.rows is not None:
        view = view[target.rows]
    view.add_(delta)


def _merge_tensor(
    tensor: torch.Tensor,
    applications: list[tuple[LoRAMergeTarget, torch.Tensor, torch.Tensor, str]],
    scaling: float,
) -> torch.Tensor:
    """``tensor`` with every LoRA delta aimed at it added, in float32 and rounded once."""
    orig_dtype = tensor.dtype
    by_expert: dict[int | None, list[tuple[LoRAMergeTarget, torch.Tensor, torch.Tensor, str]]] = defaultdict(list)
    for application in applications:
        by_expert[application[0].expert].append(application)
    if None in by_expert and len(by_expert) > 1:  # the planner's shape check makes this unreachable
        raise AdapterMergeError(
            f"LoRA pairs {[a[3] for a in applications]} target one tensor both whole and per expert"
        )

    if None in by_expert:  # a plain linear weight: one [out, in] (or [in, out]) matrix
        merged = tensor.to(torch.float32, copy=True)
        for target, lora_a, lora_b, _ in by_expert[None]:
            _add_delta(merged, target, _lora_delta(lora_a, lora_b, scaling))
        return merged.to(orig_dtype)

    merged = tensor.clone()  # a fused [E, ...] experts tensor: merge one expert at a time
    for expert, expert_applications in sorted(by_expert.items()):
        work = merged[expert].to(torch.float32, copy=True)
        for target, lora_a, lora_b, _ in expert_applications:
            _add_delta(work, target, _lora_delta(lora_a, lora_b, scaling))
        merged[expert] = work.to(orig_dtype)
    return merged


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

#: safetensors dtypes a delta can be added to (an FP8/int tensor carries scales this merge ignores).
_MERGEABLE_DTYPES = {"F16", "BF16", "F32", "F64"}


def _lora_scaling(adapter_config: dict) -> float:
    lora_alpha = adapter_config["lora_alpha"]
    lora_rank = adapter_config["r"]
    if adapter_config.get("use_rslora", False):
        return lora_alpha / math.sqrt(lora_rank)
    return lora_alpha / lora_rank


def _read_base_index(st_files: list[str]) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, str] = {}
    for st_file in st_files:
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor_slice = f.get_slice(key)
                shapes[key] = tuple(tensor_slice.get_shape())
                dtypes[key] = tensor_slice.get_dtype()
    return shapes, dtypes


def _read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _publish(staging: str, output_path: str) -> None:
    """Move the finished, staged model into ``output_path``."""
    if not os.path.exists(output_path):
        os.replace(staging, output_path)  # the merged model appears all at once
        return
    # An existing directory (the SFT trainer merges into the one holding the adapter)
    for item in sorted(os.listdir(staging)):
        os.replace(os.path.join(staging, item), os.path.join(output_path, item))


def merge_adapter(
    base_model_path: str, adapter_path: str, output_path: str, max_shard_size: str = "5GB", cpu_offload: bool = True
) -> None:
    """
    Merge LoRA adapter into base model by operating directly on safetensors files.

    Copies the base model's safetensors to output_path, then applies LoRA deltas
    in-place. This preserves the exact original key structure, ensuring
    compatibility with vLLM and other serving frameworks. Tensors no LoRA pair
    targets keep their exact bytes; shards with none are copied as files.

    Raises AdapterMergeError when any LoRA pair has no target in the base model
    (or does not fit it). Nothing is written to output_path in that case.

    Args:
        base_model_path: Path to the base model directory
        adapter_path: Path to the adapter directory
        output_path: Output directory for merged model
        max_shard_size: Unused (kept for API compat)
        cpu_offload: Unused (always operates on CPU)
    """
    # Load adapter config
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(adapter_config_path) as f:
        adapter_config = json.load(f)
    scaling = _lora_scaling(adapter_config)

    # Load adapter weights
    adapter_weights = load_adapter_weights(adapter_path)

    # Find all safetensor files in the base model
    st_files = sorted(glob.glob(os.path.join(base_model_path, "*.safetensors")))
    st_files = [f for f in st_files if "adapter" not in os.path.basename(f)]
    if not st_files:
        raise FileNotFoundError(f"No safetensors files found in {base_model_path}")

    # Resolve every LoRA pair against the base checkpoint before writing anything
    base_shapes, base_dtypes = _read_base_index(st_files)
    layout = fused_expert_layout(_read_json(os.path.join(base_model_path, "config.json")))
    plan = _plan_lora_merge(adapter_weights, base_shapes, layout)
    if plan.num_pairs == 0:
        raise AdapterMergeError(f"No LoRA pairs found in {adapter_path}; nothing to merge")
    if plan.missing:
        examples = ", ".join(plan.missing[:5])
        raise AdapterMergeError(
            f"{len(plan.missing)} of {plan.num_pairs} LoRA pairs have no target in the base model "
            f"{base_model_path} (e.g. {examples}). Is this adapter for this base model? Nothing was written."
        )
    unmergeable = sorted(key for key in plan.targets if base_dtypes[key] not in _MERGEABLE_DTYPES)
    if unmergeable:
        raise AdapterMergeError(
            f"{len(unmergeable)} LoRA targets are stored as {base_dtypes[unmergeable[0]]}, not a float type "
            f"a delta can be added to (e.g. {unmergeable[0]}); merge into an unquantized base model"
        )

    remaps = ", ".join(f"{src or '<none>'} -> {dst or '<none>'}" for src, dst in plan.prefix_remaps)
    logger.info(
        f"Found {plan.num_pairs} LoRA pairs to merge into {len(plan.targets)} base tensors"
        + (f" ({plan.num_fused} into fused expert tensors)" if plan.num_fused else "")
        + (f"; prefix remap {remaps}" if remaps else "")
    )

    # Stage the output next to output_path (same filesystem, so it moves by rename): a failure
    # leaves no partial model behind.
    output_path = os.path.abspath(output_path)
    staging = os.path.join(
        os.path.dirname(output_path), f".{os.path.basename(output_path)}.merging-{uuid.uuid4().hex[:12]}"
    )
    os.makedirs(staging)
    try:
        # Copy all non-safetensor files from base model (config, tokenizer, etc.)
        for item in os.listdir(base_model_path):
            src = os.path.join(base_model_path, item)
            if os.path.isfile(src) and not item.endswith(".safetensors"):
                shutil.copy2(src, os.path.join(staging, item))

        # Process each safetensor shard: copy then merge LoRA in-place
        merged_count = 0
        for st_file in st_files:
            shard_name = os.path.basename(st_file)
            output_shard = os.path.join(staging, shard_name)

            with safe_open(st_file, framework="pt", device="cpu") as f:
                shard_keys = list(f.keys())
                if not any(key in plan.targets for key in shard_keys):
                    shard_tensors = None
                else:
                    metadata = f.metadata()
                    shard_tensors = {key: f.get_tensor(key) for key in shard_keys}

            if shard_tensors is None:
                # Just copy the file as-is (faster than re-serializing)
                shutil.copy2(st_file, output_shard)
                continue

            for key in shard_keys:
                if key in plan.targets:
                    shard_tensors[key] = _merge_tensor(shard_tensors[key], plan.targets[key], scaling)
                    merged_count += len(plan.targets[key])
            save_file(shard_tensors, output_shard, metadata=metadata)
            del shard_tensors

        if merged_count != plan.num_pairs:
            raise AdapterMergeError(f"Merged {merged_count} of {plan.num_pairs} LoRA pairs; nothing was written")
        logger.info(f"Merged {merged_count}/{plan.num_pairs} LoRA weights")

        _publish(staging, output_path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    # Move adapter files into a subdirectory so vLLM doesn't misdetect as PEFT adapter
    adapter_files = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]
    if any(os.path.exists(os.path.join(output_path, f)) for f in adapter_files):
        adapter_subdir = os.path.join(output_path, "adapter")
        os.makedirs(adapter_subdir, exist_ok=True)
        for adapter_file in adapter_files:
            src = os.path.join(output_path, adapter_file)
            if os.path.exists(src):
                shutil.move(src, os.path.join(adapter_subdir, adapter_file))
        logger.info(f"Moved adapter files to {adapter_subdir}/")

    logger.info(f"Merged model saved to {output_path}")
