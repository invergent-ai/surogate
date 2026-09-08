"""Convert quantized hybrid checkpoints using resolved dimensions and observed storage."""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import struct
import time

import torch

from surogate.serve.artifact.container import ArtifactIdentity, ArtifactWriter
from surogate.serve.artifact.layouts import (
    encode_nvfp4, encode_fp8_block_scaled, encode_fp8_row_f32, encode_fp8_row_scaled,
)
from surogate.serve.convert.common import conversion, qwen3_5 as checkpoint
from surogate.serve.convert.common.checkpoint import tokenizer_domain
from surogate.serve.convert.common.quant_scope import observed_scope
from surogate.serve.convert.common.recipe import (
    Concat, TensorRecipe, expression_sources, preflight_source_reader,
)
from surogate.serve.convert.common.safetensors import ShardReader
from .. import draft_head, inventory as inv, recipe as base_recipe
from . import recipe_nvfp4_uniform as matrices


@dataclass(frozen=True)
class Encoding:
    format: str
    weight: str
    scale: str = ""
    global_scale: str = ""
    input_scale: str = ""
    reciprocal: bool = False


class Sources:
    """A primary checkpoint, with an optional complete checkpoint for missing modules."""

    def __init__(self, primary, fallback=None):
        self.readers = (primary,) if fallback is None else (primary, fallback)
        self.names = tuple(dict.fromkeys(name for reader in self.readers for name in reader.names))
        self._metadata = {}

    def owner(self, name):
        stem = name.removesuffix(".weight")
        for reader in self.readers:
            if reader.has(name) or reader.has(stem + ".weight_packed"):
                return reader
        raise KeyError(f"checkpoint has no tensor {name}")

    def has(self, name):
        try:
            self.owner(name)
            return True
        except KeyError:
            return False

    def raw_metadata(self, reader, name):
        key = (id(reader), name)
        if key not in self._metadata:
            self._metadata[key] = reader.metadata((name,))[name]
        return self._metadata[key]

    def encoding(self, stem):
        reader = self.owner(stem + ".weight")
        if reader.has(stem + ".weight_zero_point"):
            raise ValueError(f"{stem}: asymmetric quantization is not supported")
        if reader.has(stem + ".weight_packed"):
            return Encoding(inv.NVFP4, "weight_packed", "weight_scale", "weight_global_scale", "input_global_scale")
        meta = self.raw_metadata(reader, stem + ".weight")
        if meta.dtype == "U8" and reader.has(stem + ".weight_scale_2"):
            return Encoding(inv.NVFP4, "weight", "weight_scale", "weight_scale_2", "input_scale", True)
        if meta.dtype == "F8_E4M3":
            if reader.has(stem + ".weight_scale_inv"):
                return Encoding(inv.FP8_BLOCK_FORMAT, "weight", "weight_scale_inv")
            scale = self.raw_metadata(reader, stem + ".weight_scale")
            return Encoding(inv.FP8 if scale.dtype == "BF16" else inv.FP8_ROW_F32_FORMAT,
                            "weight", "weight_scale")
        if meta.dtype not in ("BF16", "F16", "F32"):
            raise ValueError(f"{stem}: unsupported weight dtype {meta.dtype}")
        return Encoding(inv.BF16, "weight")

    def scalar(self, stem, suffix, reciprocal=False):
        reader = self.owner(stem + ".weight")
        value = reader.get(stem + "." + suffix)
        if value.numel() != 1:
            raise ValueError(f"{stem}.{suffix}: expected one scale")
        result = float(value.float().reshape(()))
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"{stem}.{suffix}: scale must be finite and positive")
        return 1.0 / result if reciprocal else result

    def validate_matrix(self, source, encoding):
        reader = self.owner(source.name + ".weight")
        n, k = source.shape
        metadata = self.raw_metadata(reader, source.field(encoding.weight))
        if encoding.format == inv.NVFP4:
            shape, dtype = (n, k // 2), "U8"
            if k % 16:
                raise ValueError(f"{source.name}: NVFP4 columns must be divisible by 16")
            scale = self.raw_metadata(reader, source.field(encoding.scale))
            if scale.shape != (n, k // 16) or scale.dtype != "F8_E4M3":
                raise ValueError(f"{source.name}: invalid NVFP4 block scales")
            self.scalar(source.name, encoding.global_scale, encoding.reciprocal)
            self.scalar(source.name, encoding.input_scale, encoding.reciprocal)
        else:
            shape, dtype = (n, k), "F8_E4M3" if encoding.format != inv.BF16 else metadata.dtype
            if encoding.scale:
                scale = self.raw_metadata(reader, source.field(encoding.scale))
                if encoding.format == inv.FP8_BLOCK_FORMAT:
                    if n % 128 or k % 128 or scale.shape != (n // 128, k // 128) or scale.dtype != "F32":
                        raise ValueError(f"{source.name}: invalid FP8 block geometry")
                elif math.prod(scale.shape) != n:
                    raise ValueError(f"{source.name}: expected one FP8 scale per row")
        if metadata.shape != shape or metadata.dtype != dtype:
            raise ValueError(f"{source.name}: stored weight signature disagrees with config shape {source.shape}")

    def metadata(self, names):
        out = {}
        for name in names:
            reader = self.owner(name)
            if name.endswith(".weight"):
                stem = name.removesuffix(".weight")
                encoding = self.encoding(stem)
                meta = self.raw_metadata(reader, stem + "." + encoding.weight)
                shape = (meta.shape[0], meta.shape[1] * 2) if encoding.format == inv.NVFP4 else meta.shape
                if encoding.format != inv.BF16:
                    self.validate_matrix(matrices.MatrixSource(stem, shape), encoding)
                out[name] = replace(meta, shape=shape, dtype="BF16")
            else:
                out[name] = self.raw_metadata(reader, name)
        return out

    def get(self, name):
        reader = self.owner(name)
        if not name.endswith(".weight"):
            return reader.get(name)
        stem = name.removesuffix(".weight")
        encoding = self.encoding(stem)
        codes = reader.get(stem + "." + encoding.weight)
        if encoding.format == inv.BF16:
            return codes.to(torch.bfloat16)
        scales = reader.get(stem + "." + encoding.scale).float()
        if encoding.format == inv.NVFP4:
            values = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
            dense = torch.stack((values[(codes & 15).long()], values[(codes >> 4).long()]), dim=-1).flatten(1)
            scale = self.scalar(stem, encoding.global_scale, encoding.reciprocal)
            return (dense * scales.repeat_interleave(16, dim=1) / scale).to(torch.bfloat16)
        if encoding.format == inv.FP8_BLOCK_FORMAT:
            scales = scales.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        else:
            scales = scales.reshape(-1, 1)
        return (codes.float() * scales).to(torch.bfloat16)


def _same_divisor(sources, parts, field):
    values = []
    for part in parts:
        encoding = sources.encoding(part.source.name)
        values.append(sources.scalar(part.source.name, getattr(encoding, field), encoding.reciprocal))
    if len(set(values)) != 1:
        raise ValueError(f"fused sources disagree on {field}")
    return values[0]


def _matrix_encoding(entry, sources):
    formats = set()
    for part in entry.parts:
        encoding = sources.encoding(part.source.name)
        sources.validate_matrix(part.source, encoding)
        formats.add(encoding.format)
    if len(formats) != 1:
        raise ValueError(f"{entry.object_name}: cannot fuse components stored in different formats")
    result = formats.pop()
    if result == inv.NVFP4:
        _same_divisor(sources, entry.parts, "global_scale")
        _same_divisor(sources, entry.parts, "input_scale")
    return result


def _split_gdn(entry):
    return tuple(matrices.Nvfp4WeightRecipe(
        entry.object_name.removesuffix("query_key_value_z") + role,
        (part.output_rows, entry.shape[1]), (part,), (part.source,),
    ) for role, part in zip(("query_key_value", "z"), entry.parts))


def build(geometry, profile, sources):
    """Build an artifact plan from config dimensions and actual per-object source formats."""
    g = geometry
    policy = inv.export_for(profile, g)
    template = inv.build_tensor_specs(g, mtp=bool(g.mtp_layers), vision=bool(inv.vision_tower(g)))
    base = {r.object_name: r for r in base_recipe.build_recipes(g)}
    entries = list(matrices.build(g, "model.").nvfp4_weights)
    # Endpoints preserve a checkpoint's FP8 codes when it stores them quantized.
    for name in ("text/token_embedding", "text/output_head"):
        from surogate.serve.convert.common.recipe import resolve_options
        expression = resolve_options(base[name].expression, sources.has)
        tensors = expression_sources(expression)
        if len(tensors) != 1:
            raise ValueError(f"{name}: expected a single vocabulary tensor")
        source = matrices.MatrixSource(tensors[0].name.removesuffix(".weight"), (g.vocab, g.hidden))
        if sources.encoding(source.name).format in (inv.FP8, inv.FP8_ROW_F32_FORMAT, inv.FP8_BLOCK_FORMAT):
            entries.append(matrices.Nvfp4WeightRecipe(name, source.shape, (matrices._all(source),), (source,)))
    matrix_by_name, formats, replacements = {}, {}, {}
    for entry in entries:
        if entry.object_name.endswith("gdn/query_key_value_z"):
            # Split the qkv/z pair when its formats or calibration scales differ.
            try:
                fmt = _matrix_encoding(entry, sources)
            except ValueError:
                parts = _split_gdn(entry)
                for part in parts:
                    formats[part.object_name] = _matrix_encoding(part, sources)
                    matrix_by_name[part.object_name] = part
                replacements[entry.object_name] = tuple(p.object_name for p in parts)
                continue
        else:
            fmt = _matrix_encoding(entry, sources)
        formats[entry.object_name] = fmt
        matrix_by_name[entry.object_name] = entry
    tensors, divisors = [], {}
    for spec in template:
        names = replacements.get(spec.name, (spec.name,))
        for name in names:
            if name in matrix_by_name:
                entry = matrix_by_name[name]
                spec_out = inv.tensor_spec(name, entry.shape, formats[name])
            else:
                numeric_format = (policy.vocabulary if name in ("text/token_embedding", "text/output_head")
                                  else policy.draft_head if name == "text/draft_head" else spec.format)
                if name.startswith(("mtp/", "vision/")) and len(spec.shape) == 2:
                    from surogate.serve.convert.common.recipe import resolve_options
                    expression = resolve_options(base[name].expression, sources.has)
                    source_formats = {
                        sources.encoding(source.name.removesuffix(".weight")).format
                        for source in expression_sources(expression)
                    }
                    if source_formats != {inv.BF16}:
                        component = name.split("/", 1)[0]
                        raise ValueError(
                            f"{name}: this export supports unquantized {component} matrices; "
                            f"found {sorted(source_formats)}. Use --no-{component} to omit this component."
                        )
                    numeric_format = inv.BF16
                spec_out = inv.tensor_spec(name, spec.shape, numeric_format)
            tensors.append(spec_out)
            if spec_out.format == inv.NVFP4:
                prefix, role = name.split("/", 3)[:3], name.split("/", 3)[-1]
                site = inv._DIVISOR_SITES[role]
                divisor_name = "/".join(prefix) + "/" + site + "/input_scale_divisor"
                divisors[divisor_name] = matrix_by_name[name]
                tensors.append(inv.tensor_spec(divisor_name, (), inv.FP32))
    # Dense fallback expressions need no storage cuts: matrix recipes handle split inputs.
    needed = [base[s.name] for s in tensors if s.name not in matrix_by_name and s.name not in divisors]
    source_preflight = preflight_source_reader(sources, needed)
    names = [s.name for s in tensors]
    if len(names) != len(set(names)):
        raise ValueError("quantized inventory contains duplicate objects")
    return RecipePlan(g, profile, tuple(tensors), base, matrix_by_name, formats, divisors, source_preflight)


@dataclass(frozen=True)
class RecipePlan:
    geometry: inv.Geometry
    profile: str
    tensors: tuple
    base_recipes: dict
    matrices: dict
    formats: dict
    divisors: dict
    source_preflight: object

    @property
    def objects(self):
        return inv.RESOURCE_SPECS + self.tensors


def encode_matrix(entry, sources, device):
    numeric_format = _matrix_encoding(entry, sources)
    if numeric_format == inv.BF16:
        parts = [matrices._select_rows(sources.get(part.source.name + ".weight"), part) for part in entry.parts]
        tensor = torch.cat(parts, dim=0)
        return conversion.encode_tensor_payload(tensor, inv.tensor_spec(entry.object_name, entry.shape, inv.BF16), device)
    codes, scales = [], []
    for part in entry.parts:
        encoding = sources.encoding(part.source.name)
        reader = sources.owner(part.source.name + ".weight")
        weight = reader.get(part.source.field(encoding.weight)).view(torch.uint8)
        scale = reader.get(part.source.field(encoding.scale))
        unit = 128 if numeric_format == inv.FP8_BLOCK_FORMAT else 1
        if numeric_format in (inv.FP8, inv.FP8_ROW_F32_FORMAT):
            scale = scale.reshape(-1)
        for rows in part.rows:
            if rows.begin % unit or rows.end % unit:
                raise ValueError(f"{entry.object_name}: fusion must preserve whole scale blocks")
            codes.append(weight[rows.begin:rows.end])
            scales.append(scale[rows.begin // unit:rows.end // unit])
    codes, scales = torch.cat(codes, dim=0), torch.cat(scales, dim=0)
    if numeric_format == inv.NVFP4:
        scale_values = scales.float()
        if not bool(torch.isfinite(scale_values).all()) or bool((scale_values < 0).any()):
            raise ValueError(f"{entry.object_name}: invalid NVFP4 block scales")
        divisor = _same_divisor(sources, entry.parts, "global_scale")
        return encode_nvfp4(codes, scales.view(torch.uint8), struct.pack("<f", divisor), entry.shape)
    if not bool(torch.isfinite(scales.float()).all()):
        raise ValueError(f"{entry.object_name}: invalid FP8 scales")
    if numeric_format == inv.FP8_BLOCK_FORMAT:
        return encode_fp8_block_scaled(codes, scales, entry.shape)
    if numeric_format == inv.FP8_ROW_F32_FORMAT:
        return encode_fp8_row_f32(codes, scales.float(), entry.shape)
    return encode_fp8_row_scaled(codes, scales, entry.shape)


def encode_dense(spec, tensor, device):
    if spec.format == inv.FP8:
        from .fp8_embedding import encode_bf16_rows
        return encode_bf16_rows(tensor.to(torch.bfloat16))
    if spec.format in (inv.BF16, inv.FP32, inv.I32):
        tensor = tensor.to({inv.BF16: torch.bfloat16, inv.FP32: torch.float32, inv.I32: torch.int32}[spec.format])
    return conversion.encode_tensor_payload(tensor, spec, device)


def _geometry(config, root, sources, *, mtp=True, vision=True):
    selected = deepcopy(config)
    text = selected.get("text_config", selected)
    if not mtp or not any(name.startswith("mtp.") for name in sources.names):
        text["mtp_num_hidden_layers"] = 0
    if not vision or not any("visual." in name for name in sources.names):
        selected.pop("vision_config", None)
    g = inv.geometry_from_config(selected, token_domain=tokenizer_domain(root))
    return replace(g, observed_scope=observed_scope(sources.readers[0].names))


def _validate_quantization(config):
    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        raise ValueError("a quantized conversion requires quantization_config")
    if quant.get("quant_method") != "compressed-tensors":
        return
    for name, group in (quant.get("config_groups") or {}).items():
        weights = group.get("weights") or {}
        bits = weights.get("num_bits")
        if weights.get("type") != "float" or bits not in (4, 8) or weights.get("symmetric") is not True:
            raise ValueError(f"quantization group {name} requires symmetric floating-point weights")
        if weights.get("dynamic", False) is not False:
            raise ValueError(f"quantization group {name}: dynamic weight quantization is unsupported")
        if bits == 4 and (weights.get("group_size") != 16 or weights.get("strategy") != "tensor_group"):
            raise ValueError(f"quantization group {name}: NVFP4 requires groups of 16")
        if bits == 8 and weights.get("strategy") != "channel":
            raise ValueError(f"quantization group {name}: compressed FP8 requires per-channel scales")
        activation = group.get("input_activations")
        if activation is not None:
            if not isinstance(activation, dict) or (
                activation.get("type") != "float" or activation.get("num_bits") != bits
                or activation.get("symmetric") is not True
                or activation.get("strategy") not in (("tensor_group",) if bits == 4 else ("tensor", "token"))
                or (bits == 4 and activation.get("group_size") != 16)
                or not isinstance(activation.get("dynamic", False), bool)
            ):
                raise ValueError(f"quantization group {name}: unsupported input activation quantization")
        if group.get("output_activations") is not None:
            raise ValueError(f"quantization group {name}: output activation quantization is unsupported")


def dense_payload(spec, entry, sources, derived, device):
    """Stream vocabulary matrices without materializing the entire checkpoint tensor."""
    if spec.format == inv.FP8 and spec.name in ("text/token_embedding", "text/output_head"):
        from surogate.serve.convert.common.recipe import resolve_options
        from .fp8_embedding import iter_reader_payload
        expression = resolve_options(entry.expression, sources.has)
        source, = expression_sources(expression)
        return iter_reader_payload(sources.owner(source.name), source.name, spec.shape)
    return encode_dense(spec, base_recipe.materialize_recipe(entry, sources, derived), device)


def _frontend(root, resources_from, stack):
    if resources_from is None:
        return root, conversion.load_resources(root, inv.RESOURCE_SPECS)
    from tempfile import TemporaryDirectory
    from surogate.serve.artifact.container import Artifact, ResourceObject

    directory = Path(stack.enter_context(TemporaryDirectory(prefix="sinfer-frontend-")))
    resources = []
    with Artifact.open(resources_from) as artifact:
        for spec in inv.RESOURCE_SPECS:
            obj = artifact.find(spec.name)
            if obj is None:
                continue
            if not isinstance(obj, ResourceObject):
                raise ValueError(f"{spec.name}: expected a frontend resource")
            data = bytes(artifact.payload(obj))
            resources.append(conversion.ResourcePayload(spec.name, data))
            (directory / spec.name.removeprefix("frontend/")).write_bytes(data)
    return directory, tuple(resources)


def convert(model_dir, out_path, *, profile, quantized_model_dir=None, device="cuda", resources_from=None,
            mtp=True, vision=True):
    from surogate.serve.convert.common.quantize import pick_device
    from ..convert import _tools_root
    started = time.perf_counter()
    root = Path(model_dir)
    primary = Path(quantized_model_dir) if quantized_model_dir is not None else root
    config = conversion.load_json(root / "config.json")
    primary_config = conversion.load_json(primary / "config.json")
    _validate_quantization(primary_config)
    with ExitStack() as stack:
        primary_reader = stack.enter_context(ShardReader.for_directory(primary))
        base_reader = stack.enter_context(ShardReader.for_directory(root)) if primary != root else None
        sources = Sources(primary_reader, base_reader)
        frontend_root, resources = _frontend(root, resources_from, stack)
        g = _geometry(config, frontend_root, sources, mtp=mtp, vision=vision)
        if base_reader is not None:
            other = _geometry(primary_config, frontend_root, sources, mtp=mtp, vision=vision)
            if g != other:
                raise ValueError("base and quantized checkpoint dimensions or execution settings differ")
        scope = conversion.honour_declared_scope(primary_config, g, primary, what=conversion.checkpoint_label(primary))
        if scope:
            print(scope, flush=True)
        plan = build(g, profile, sources)
        object_plan = conversion.build_object_plan(plan.objects, {r.name: r.data for r in resources})
        draft = draft_head.compute_shortlist(_tools_root() / draft_head.DEFAULT_RANKING, frontend_root, geometry=g)
        derived = {draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: draft_head.materialize_draft_head_token_ids(draft)}
        resource_map = {r.name: r.data for r in resources}
        output = Path(out_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        resolved_device = pick_device(device)
        print(f"preflight complete: {len(object_plan.objects)} objects, {g.layers} layers, hidden={g.hidden}", flush=True)
        with ArtifactWriter(
            output, ArtifactIdentity(inv.MODEL_ID, inv.weights_id_for(profile), architecture=inv.TARGET_KEY),
            object_plan.specs, geometry=checkpoint.geometry_block(g), layer_types=g.layer_types,
            vision_geometry=checkpoint.vision_geometry_block(g.declared.hf_config),
        ) as writer:
            for spec in plan.objects:
                if isinstance(spec, inv.ResourceSpec):
                    if spec.name not in resource_map:
                        continue
                    payload = resource_map[spec.name]
                elif spec.name in plan.divisors:
                    payload = struct.pack("<f", _same_divisor(sources, plan.divisors[spec.name].parts, "input_scale"))
                elif spec.name in plan.matrices:
                    payload = encode_matrix(plan.matrices[spec.name], sources, resolved_device)
                else:
                    payload = dense_payload(spec, plan.base_recipes[spec.name], sources, derived, resolved_device)
                writer.write(spec.name, payload)
        report = conversion.build_conversion_report(
            identity=ArtifactIdentity(inv.MODEL_ID, inv.weights_id_for(profile), architecture=inv.TARGET_KEY),
            target_key=inv.TARGET_KEY, recipe_id="qwen3_5-checkpoint-quantized-v3", repo_root=_tools_root(),
            model_dir=primary, out_path=output,
            arguments={"model": str(root), "quantized_model": str(primary), "device": device,
                       "mtp": mtp, "vision": vision},
            config_summary=checkpoint.geometry_block(g), source_preflight=plan.source_preflight,
            objects=object_plan.objects, elapsed_seconds=time.perf_counter() - started,
            final_bytes=output.stat().st_size, device=resolved_device, ranking_path=draft.ranking,
        )
        report["source"]["matrix_formats"] = plan.formats
        Path(str(output) + ".conversion.json").write_text(json.dumps(report, indent=2) + "\n")
        return output


def main(argv=None, *, profile=None):
    import argparse
    from ..convert import profile_for_checkpoint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--quantized-model", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--no-vision", action="store_true")
    args = parser.parse_args(argv)
    source = args.quantized_model or args.model
    profile = profile or profile_for_checkpoint(conversion.load_json(source / "config.json"))
    return convert(args.model, args.out, profile=profile, quantized_model_dir=args.quantized_model,
                   device=args.device, mtp=not args.no_mtp, vision=not args.no_vision)
