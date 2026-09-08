"""Verify a quantized hybrid artifact against its checkpoint configuration and source words."""

from contextlib import ExitStack
from pathlib import Path
import struct

from surogate.serve.artifact.container import Artifact, ResourceObject, TensorObject
from surogate.serve.convert.common import conversion, qwen3_5 as checkpoint
from surogate.serve.convert.common.safetensors import ShardReader
from .. import inventory as inv
from .. import recipe as base_recipe
from .. import draft_head
from ..convert import _tools_root
from . import quantized


def verify_artifact(artifact, base_dir, nvfp4_dir=None, *, device="cpu"):
    root = Path(base_dir)
    primary = root if nvfp4_dir is None else Path(nvfp4_dir)
    if artifact.identity.architecture != inv.TARGET_KEY:
        raise ValueError("artifact architecture does not match this checkpoint family")
    profile = inv.profile_for(artifact.identity.weights_id)
    with ExitStack() as stack:
        reader = stack.enter_context(ShardReader.for_directory(primary))
        fallback = stack.enter_context(ShardReader.for_directory(root)) if primary != root else None
        sources = quantized.Sources(reader, fallback)
        g = quantized._geometry(conversion.load_json(root / "config.json"), root, sources,
                                mtp=bool(artifact.geometry["mtp_layers"]), vision=bool(artifact.vision_geometry))
        plan = quantized.build(g, profile, sources)
        if artifact.geometry != checkpoint.geometry_block(g) or tuple(artifact.layer_types) != g.layer_types:
            raise ValueError("artifact geometry does not match the source checkpoint")
        resources = {r.name: r.data for r in conversion.load_resources(root, inv.RESOURCE_SPECS)}
        expected = [s for s in plan.objects if isinstance(s, inv.TensorSpec) or s.name in resources]
        if [obj.name for obj in artifact.objects] != [s.name for s in expected]:
            raise ValueError("artifact inventory does not match the source checkpoint")
        draft = draft_head.compute_shortlist(_tools_root() / draft_head.DEFAULT_RANKING, root, geometry=g)
        derived = {draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: draft_head.materialize_draft_head_token_ids(draft)}
        for spec in expected:
            obj = artifact.find(spec.name)
            if isinstance(spec, inv.ResourceSpec):
                if not isinstance(obj, ResourceObject):
                    raise ValueError(f"{spec.name}: expected a resource")
                payload = resources[spec.name]
            else:
                if not isinstance(obj, TensorObject) or (obj.shape, obj.format, obj.layout) != (spec.shape, spec.format, spec.layout):
                    raise ValueError(f"{spec.name}: tensor signature disagrees with checkpoint")
                if spec.name in plan.divisors:
                    payload = struct.pack("<f", quantized._same_divisor(sources, plan.divisors[spec.name].parts, "input_scale"))
                elif spec.name in plan.matrices:
                    payload = quantized.encode_matrix(plan.matrices[spec.name], sources, device)
                else:
                    payload = quantized.dense_payload(spec, plan.base_recipes[spec.name], sources, derived, device)
            chunks = (payload,) if isinstance(payload, (bytes, bytearray, memoryview)) else payload
            stored = artifact.payload(obj)
            try:
                offset = 0
                for chunk in chunks:
                    if stored[offset:offset + len(chunk)] != chunk:
                        raise ValueError(f"{spec.name}: artifact bytes differ from checkpoint source words")
                    offset += len(chunk)
                if offset != len(stored):
                    raise ValueError(f"{spec.name}: artifact payload size differs from checkpoint")
            finally:
                stored.release()
    return {"objects": len(expected), "architecture": inv.TARGET_KEY, "weights_id": artifact.identity.weights_id}


def main(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--model", "--base", required=True, type=Path)
    parser.add_argument("--quantized-model", "--nvfp4", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    with Artifact.open(args.artifact) as artifact:
        result = verify_artifact(artifact, args.model, args.quantized_model, device=args.device)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
