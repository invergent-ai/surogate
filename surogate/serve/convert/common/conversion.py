"""Mechanical artifact-writing helpers shared by Qwen3.6 converters.

This module deliberately knows neither target identity nor checkpoint geometry.
Target packages own their config validation, complete object inventory, source
recipe, shortlist provenance, and conversion entry point.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import subprocess
from typing import Mapping, Sequence

import torch

from surogate.serve.convert.common import quant_scope
from surogate.serve.artifact.container import (
    ArtifactIdentity,
    ArtifactObject,
    ObjectSpec,
    ResourceObject,
    ResourceSpec as ArtifactResourceSpec,
    TensorObject,
    TensorSpec as ArtifactTensorSpec,
    plan_objects,
    tensor_payload_bytes as stored_tensor_bytes,
)
from surogate.serve.artifact.layouts import align_up, encode_direct, encoded_size
from surogate.serve.convert.common.quantize import quantize_and_encode

from .inventory import DIRECT_FORMATS, ResourceSpec, StoredObjectSpec, TensorSpec


@dataclass(frozen=True, slots=True)
class ResourcePayload:
    name: str
    data: bytes


@dataclass(frozen=True, slots=True)
class ObjectPlan:
    specs: tuple[ObjectSpec, ...]
    objects: tuple[ArtifactObject, ...]

    @property
    def payload_span_bytes(self) -> int:
        last = self.objects[-1]
        return last.offset + last.bytes


def load_json(path: str | Path) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def check_members(
    scope: str,
    actual: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    """Require the target-selected members while allowing unrelated metadata."""

    mismatches = [
        f"{scope}.{name}: expected {value!r}, got {actual.get(name)!r}"
        for name, value in expected.items()
        if actual.get(name) != value
    ]
    if mismatches:
        raise ValueError("checkpoint config mismatch:\n  " + "\n  ".join(mismatches))


def load_resources(
    model_dir: str | Path,
    resource_specs: Sequence[ResourceSpec],
) -> tuple[ResourcePayload, ...]:
    root = Path(model_dir)
    resources: list[ResourcePayload] = []
    for spec in resource_specs:
        filename = spec.name.removeprefix("frontend/")
        path = root / filename
        if not path.exists() and filename == "generation_config.json":
            data = _synthesize_generation_config(root)
        elif (not path.exists() and filename in ("preprocessor_config.json", "video_preprocessor_config.json")
              and (root / "processor_config.json").is_file()):
            # Recent Transformers exports bundle both processors in one resource.
            processor = load_json(root / "processor_config.json")
            key = "image_processor" if filename == "preprocessor_config.json" else "video_processor"
            nested = processor.get(key)
            if nested is None:
                continue
            if not isinstance(nested, dict):
                raise ValueError(f"processor_config.json.{key} must be an object")
            data = json.dumps({**nested, **({"processor_class": processor["processor_class"]}
                                          if "processor_class" in processor else {})}).encode()
        elif not path.exists() and (filename == "chat_template.jinja" or filename.endswith("preprocessor_config.json")):
            # A text-only release of a vision family ships no image processor. The artifact
            # then carries no such resource, and the engine's frontend reads its absence as
            # "this target never asked for a pixel" and refuses --vision at load.
            continue
        else:
            data = path.read_bytes()
        if not data:
            raise ValueError(f"frontend resource {filename} is empty")
        resources.append(ResourcePayload(spec.name, data))
    return tuple(resources)


def _synthesize_generation_config(root: Path) -> bytes:
    """A generation config for checkpoints that publish none.

    Not every model in this family ships `generation_config.json` — `Qwen/Qwen3.5-0.8B`
    and `Qwen3.5-9B` do not, and the hub returns 404 for it — but the engine requires
    one and reads `eos_token_id` out of it to seed its default stop tokens. That field
    lives in config.json, but chat checkpoints may also use a different tokenizer
    EOS for end-of-turn. Retain both so tool calls terminate at the chat boundary.
    """

    config = json.loads((root / "config.json").read_text())
    text = config.get("text_config", config)
    generation: dict[str, object] = {}
    for key in ("eos_token_id", "bos_token_id", "pad_token_id"):
        value = text.get(key, config.get(key))
        if value is not None:
            generation[key] = value
    tokenizer_config = root / "tokenizer_config.json"
    if tokenizer_config.exists():
        tokenizer = json.loads(tokenizer_config.read_text())
        token = tokenizer.get("eos_token")
        if isinstance(token, dict):
            token = token.get("content")
        if isinstance(token, str):
            chat_eos = next((int(i) for i, added in tokenizer.get("added_tokens_decoder", {}).items()
                             if added.get("content") == token), None)
            if chat_eos is None and (root / "tokenizer.json").exists():
                backend = json.loads((root / "tokenizer.json").read_text())
                chat_eos = next((added["id"] for added in backend.get("added_tokens", [])
                                 if added.get("content") == token), None)
                if chat_eos is None:
                    vocab = backend.get("model", {}).get("vocab", {})
                    if isinstance(vocab, dict):
                        chat_eos = vocab.get(token)
            if chat_eos is not None:
                current = generation.get("eos_token_id", [])
                stops = list(current) if isinstance(current, list) else [current]
                if chat_eos not in stops:
                    stops.append(chat_eos)
                generation["eos_token_id"] = stops[0] if len(stops) == 1 else stops
    if "eos_token_id" not in generation:
        raise ValueError(
            "cannot synthesize generation_config.json: config.json declares no eos_token_id, "
            "and the engine needs one to seed its default stop tokens"
        )
    return json.dumps(generation, indent=2).encode()


#: The artifact objects a checkpoint may legitimately not carry.
_CHAT_TEMPLATE = "frontend/chat_template.jinja"
#: The image and video processor configs: a text-only release of a vision family ships
#: neither, and the engine reads their absence as "never asked for a pixel".
_PREPROCESSOR_CONFIGS = ("frontend/preprocessor_config.json", "frontend/video_preprocessor_config.json")
OPTIONAL_RESOURCES = (_CHAT_TEMPLATE, *_PREPROCESSOR_CONFIGS)


def build_object_plan(
    object_specs: Sequence[StoredObjectSpec],
    resources: Mapping[str, bytes],
) -> ObjectPlan:
    expected_resources = tuple(
        spec.name for spec in object_specs if isinstance(spec, ResourceSpec)
    )
    # The chat template and the pixel processor configs are the resources a checkpoint may
    # legitimately not have: a base model publishes no template, a text-only release no
    # processor. An absence drops the object from the artifact, and the engine reads it as
    # "no chat endpoints" or "never asked for a pixel". Every other resource missing is still
    # a converter that lost track of its own contract.
    missing_optional = {name for name in OPTIONAL_RESOURCES if name not in resources}
    if missing_optional:
        expected_resources = tuple(
            name for name in expected_resources if name not in missing_optional
        )
    if tuple(resources) != expected_resources:
        raise ValueError("resource mapping does not match canonical inventory order")

    specs: list[ObjectSpec] = []
    for spec in object_specs:
        if isinstance(spec, ResourceSpec):
            if spec.name not in resources:
                continue
            specs.append(
                ArtifactResourceSpec(spec.name, spec.encoding, len(resources[spec.name]))
            )
        elif isinstance(spec, TensorSpec):
            specs.append(
                ArtifactTensorSpec(
                    spec.name, spec.shape, spec.format, spec.layout, getattr(spec, "runs", ()),
                    getattr(spec, "transform", ""), getattr(spec, "group_map", ()),
                    getattr(spec, "segments", ()),
                )
            )
        else:
            raise TypeError(f"unsupported inventory spec: {type(spec).__name__}")
    frozen_specs = tuple(specs)
    return ObjectPlan(frozen_specs, plan_objects(frozen_specs))


def encode_tensor_payload(
    tensor: torch.Tensor,
    spec: TensorSpec,
    device: str | torch.device,
) -> bytes:
    if spec.format in DIRECT_FORMATS:
        return encode_direct(tensor, spec.format)
    return quantize_and_encode(tensor, spec.format, device=device)


def object_statistics(objects: Sequence[ArtifactObject]) -> dict[str, object]:
    tensors = [obj for obj in objects if isinstance(obj, TensorObject)]
    resources = [obj for obj in objects if isinstance(obj, ResourceObject)]
    object_bytes = sum(obj.bytes for obj in objects)
    payload_span = objects[-1].offset + objects[-1].bytes
    return {
        "count": len(objects),
        "tensors": len(tensors),
        "resources": len(resources),
        "formats": dict(sorted(Counter(obj.format for obj in tensors).items())),
        "layouts": dict(sorted(Counter(obj.layout for obj in tensors).items())),
        "encodings": dict(
            sorted(Counter(obj.encoding for obj in resources).items())
        ),
        "tensor_bytes": sum(obj.bytes for obj in tensors),
        "resource_bytes": sum(obj.bytes for obj in resources),
        "object_bytes": object_bytes,
        "payload_span_bytes": payload_span,
        "alignment_bytes": payload_span - object_bytes,
    }


def tensor_payload_bytes(tensor_specs: Sequence[TensorSpec]) -> int:
    return sum(
        stored_tensor_bytes(spec.layout, spec.format, spec.shape, getattr(spec, "segments", ()))
        for spec in tensor_specs
    )


def device_arena_bytes(tensor_specs: Sequence[TensorSpec], alignment: int = 256) -> int:
    cursor = 0
    for spec in tensor_specs:
        cursor = align_up(cursor, alignment)
        cursor += stored_tensor_bytes(spec.layout, spec.format, spec.shape,
                                      getattr(spec, "segments", ()))
    return cursor


def converter_revision(repo_root: str | Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def environment(device: torch.device) -> dict[str, object]:
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "resolved_device": str(device),
        "gpu": gpu,
    }


def checkpoint_label(model_dir) -> str:
    """A name a reader recognises. A Hugging Face cache path ends in a snapshot hash, which
    names nothing; the repo two levels up is what the user asked for."""
    root = Path(model_dir).resolve()
    for parent in root.parents:
        if parent.name.startswith("models--"):
            return parent.name[len("models--"):].replace("--", "/", 1)
    return root.name


def checkpoint_tensor_names(model_dir) -> tuple[str, ...]:
    """The checkpoint's tensor names from its index, without opening a shard.

    Empty when there is no index and no single-file checkpoint, which only narrows the
    cross-check: it is diagnostic and never load-bearing.
    """
    root = Path(model_dir)
    index = root / "model.safetensors.index.json"
    if index.exists():
        try:
            return tuple(json.loads(index.read_text())["weight_map"])
        except (ValueError, KeyError, TypeError):
            return ()
    single = root / "model.safetensors"
    if not single.exists():
        return ()
    try:
        from safetensors import safe_open
        with safe_open(str(single), framework="pt") as handle:
            return tuple(handle.keys())
    except Exception:  # noqa: BLE001
        return ()


def honour_declared_scope(
    config,
    geometry,
    names=(),
    *,
    what: str = "checkpoint",
) -> str:
    """Refuse what a checkpoint's `quantization_config` asks for and we do not implement, and
    report where its declaration disagrees with its own tensors.

    Every converter runs this, because the failure it rules out is not per-family: a field
    nobody reads is a field a checkpoint is served against. `geometry` supplies the one number
    the KV decision turns on, so a request the engine's default already satisfies is honoured
    rather than refused. `names` is the checkpoint's tensor inventory when the caller has it;
    without it only the refusal half runs.

    Returns the line for the conversion log, empty for an unquantised checkpoint.
    """
    gdn = len(getattr(geometry, "gdn_layers", ()) or ()) if geometry is not None else 0
    quant_scope.require_honourable(config, gdn_layers=gdn, what=what)
    if isinstance(names, (str, Path)):
        names = checkpoint_tensor_names(names)
    return quant_scope.report(config, tuple(names), what=what) if names else ""


def build_conversion_report(
    *,
    identity: ArtifactIdentity,
    target_key: str,
    recipe_id: str,
    repo_root: str | Path,
    model_dir: str | Path,
    out_path: str | Path,
    arguments: Mapping[str, object],
    config_summary: Mapping[str, object],
    source_preflight: object,
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
    ranking_path: str | Path | None,
    revision: str | None = None,
    environment_summary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the shared report envelope without defining target validity."""

    return {
        "identity": {
            "model_id": identity.model_id,
            "weights_id": identity.weights_id,
        },
        "target_key": target_key,
        "recipe_id": recipe_id,
        "source": {
            "model_path": str(Path(model_dir).resolve()),
            "ranking_path": str(Path(ranking_path).resolve()) if ranking_path is not None else None,
            "shortlist_order": "frequency" if ranking_path is not None else "token_id",
        },
        "arguments": dict(arguments),
        "config_summary": dict(config_summary),
        "source_preflight": {
            "recipes": source_preflight.recipe_count,
            "tensors": source_preflight.source_tensor_count,
            "shards": source_preflight.source_shard_count,
            "dtypes": dict(source_preflight.source_dtype_counts),
        },
        "converter": {
            "revision": (
                converter_revision(repo_root) if revision is None else revision
            ),
            "environment": dict(
                environment(device)
                if environment_summary is None
                else environment_summary
            ),
        },
        "objects": object_statistics(objects),
        "elapsed_seconds": elapsed_seconds,
        "artifact": {
            "path": str(Path(out_path)),
            "bytes": final_bytes,
        },
    }


__all__ = [
    "ObjectPlan",
    "ResourcePayload",
    "build_conversion_report",
    "build_object_plan",
    "check_members",
    "device_arena_bytes",
    "encode_tensor_payload",
    "environment",
    "load_json",
    "load_resources",
    "object_statistics",
    "tensor_payload_bytes",
]
