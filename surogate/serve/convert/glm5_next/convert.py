"""Convert the GLM-5.3-Flash GGUF (`glm5next`) into a `.sinfer` artifact.

GGUF-native, and read in place. gguf-py carries no `glm5next` name map, so there is no bridge
to route through; the six shards are the only weight source and `recipe.py` says where every
object's rows come from. The shared planners in `common/gguf_repack.py` turn those row programs
into *runs* -- stretches of a shard the engine maps and reads directly -- so 288 experts a
layer stay in the file and the artifact carries an index to them.

The mixture is 185 of the checkpoint's 200 GB and keeps the formats llama.cpp chose for it;
every Q8_0 projection is a run too, because Q8_0 and W8G32_F16S hold the same numbers and the
loader rearranges the blocks into row-split planes on the device. What the artifact stores is
what the row algebra cannot say: the norms, the hyper-connection projections, the router, the
convolution taps and the latent attention's expansion -- about 2 GB.

The frontend (tokenizer, chat template, generation config) comes from the model's own Hub
repository: a GGUF holds a token table, and the engine's frontend reads a `tokenizer.json`.
"""

from __future__ import annotations

from surogate.serve.convert.common.checkpoint import tokenizer_domain

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from surogate.serve.artifact.container import ArtifactIdentity, ArtifactWriter
from surogate.serve.convert.common import conversion as family_conversion
from surogate.serve.convert.common.gguf_repack import GgufRepackSource
from surogate.serve.convert.common.gguf_source import (
    GgufRecipeReader,
    GgufSource,
    candidate_sources,
    check_every_quantised_object_is_planned,
    encode_tensor,
)

from . import inventory as inv
from . import recipe as rcp

RECIPE_ID = rcp.RECIPE_ID


def _refuse_what_is_not_bound(source: GgufSource, geometry: inv.Geometry) -> inv.Geometry:
    """What the file carries that this artifact does not, said out loud.

    The indexer is a real part of the published model, and serving without it changes what the
    engine computes past a bound, which the artifact records so the engine can refuse rather
    than quietly attend to more than the model was trained to. A draft head the file declares
    but does not carry (a trunk-only export) is dropped from the geometry, so the artifact says
    it has none and `--spec mtp` refuses by name.
    """
    if any(name.startswith(("v.", "mm.")) or "vision" in name for name in source.tensors):
        raise NotImplementedError(
            "this GGUF carries a vision tower; the glm5_next recipes cover the text stack only"
        )
    head = f"blk.{geometry.layers}.nextn.eh_proj.weight"
    if geometry.nextn_layers and head not in source.tensors:
        print("note: the file declares a NextN draft head and does not carry it (a trunk-only "
              "export); the artifact is written without one and --spec mtp will refuse",
              flush=True)
        geometry = replace(geometry, nextn_layers=0)
    if any(".indexer." in name for name in source.tensors):
        print(f"note: the sparse indexer is present and is not bound. It selects "
              f"{geometry.index_topk} tokens, so up to that context every visible token is "
              f"selected and full attention is exactly what it would have asked for; the "
              f"artifact records the bound and the engine refuses beyond it", flush=True)
    return geometry


def materialize_unspoken(source: GgufSource, name: str, spec) -> bytes:
    """The objects `recipe.materialized_objects` names: the KDA decay's `A_log`.

    llama.cpp's converter folds the exponential -- `ssm_a = -exp(A_log)` -- so that its graph
    multiplies by the stored value directly. The engine's gate takes `A_log`, as the checkpoint
    writes it, so the fold is undone here. The sign is checked because it is the whole
    convention: a stored value that is not negative is not `-exp` of anything, and serving it
    as `A_log` would be silent.
    """
    if not name.endswith("kda/a_log"):
        raise KeyError(f"no materialiser covers artifact object {name!r}")
    stored_name = f"blk.{name.split('/')[2]}.ssm_a"
    stored = source.float32(stored_name).reshape(-1)
    if stored.size == 0 or not np.all(stored < 0.0):
        raise ValueError(
            f"{stored_name}: expected -exp(A_log), every value negative; the largest is "
            f"{stored.max() if stored.size else 'nothing'}"
        )
    return encode_tensor(torch.from_numpy(np.log(-stored)), spec)


def _geometry_block(g: inv.Geometry, *, token_domain: int) -> dict[str, int | float]:
    from surogate.serve.artifact.geometry import validate_resolved_geometry

    return validate_resolved_geometry({
        "hidden": g.hidden, "residual": g.residual, "hc_streams": g.hc_streams,
        "hc_sinkhorn_iterations": g.hc_sinkhorn_iterations, "hc_epsilon": g.hc_epsilon,
        "layers": g.layers, "intermediate": g.expert_intermediate,
        "dense_intermediate": g.dense_intermediate, "leading_dense_layers": len(g.dense_layers),
        "shared_intermediate": g.shared_intermediate, "experts": g.experts,
        "experts_per_token": g.experts_per_token, "routed_scale": g.routed_scale,
        "swiglu_limit": g.swiglu_limit, "output_rows": g.vocab, "token_domain": token_domain,
        "query_heads": g.query_heads, "kv_heads": 1, "head_dim": g.kv_lora_rank,
        "rotary_dim": 0, "rope_theta": 0.0, "qk_head_dim": g.qk_head_dim,
        "v_head_dim": g.v_head_dim, "q_lora_rank": g.q_lora_rank, "kv_lora_rank": g.kv_lora_rank,
        "gdn_conv_kernel": g.kda_conv_kernel, "gdn_key_heads": g.kda_heads,
        "gdn_key_head_dim": g.kda_head_dim, "gdn_value_heads": g.kda_heads,
        "gdn_value_head_dim": g.kda_head_dim, "kda_gate_rank": g.kda_gate_rank,
        "kda_gate_bound": -g.kda_lower_bound, "mtp_layers": g.nextn_layers,
        "rms_epsilon": g.rms_epsilon, "max_context": g.serving_context,
        "attention_scale": g.kv_lora_rank ** -0.5, "gdn_scale": g.kda_head_dim ** -0.5,
    })


def convert(gguf: str | Path, frontend_dir: str | Path, out_path: str | Path,
            *, device: str = "cuda") -> Path:
    """Write the artifact. `device` is accepted so the ingest path can call every converter the
    same way; nothing here needs a GPU, because nothing here quantises."""
    started = time.perf_counter()
    del device
    source = GgufSource(Path(gguf))
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    geometry = _refuse_what_is_not_bound(source, inv.geometry_from_gguf(source.kv))
    tensor_specs = inv.build_tensor_specs(geometry)
    object_specs: tuple = inv.RESOURCE_SPECS + tensor_specs
    recipes = rcp.build_recipes(geometry)
    materialized = rcp.materialized_objects(geometry)

    repack = GgufRepackSource.from_sources(source.shards, candidate_sources(source))
    # Two ways to read a weight where it lies. A K-quant the kernels already decode is served
    # in the file's own format; a Q8_0 whose op wants the row-split W8 planes is read just the
    # same and rearranged on the device, which is why the exclude list and the in-place list
    # are one list.
    native = repack.plan_native(recipes, tensor_specs,
                                exclude_suffixes=rcp.NATIVE_EXCLUDE_SUFFIXES)
    in_place = repack.plan_repack_in_place(recipes, tensor_specs, rcp.NATIVE_EXCLUDE_SUFFIXES)
    check_every_quantised_object_is_planned(tensor_specs, set(native) | set(in_place), inv.W8)

    native_specs = {spec.name: spec
                    for spec in GgufRepackSource.native_specs(tensor_specs, native)}
    native_runs = {name: repack.runs_for_native(native_specs[name], recipes[name], None)
                   for name in native}
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
        ArtifactIdentity(inv.MODEL_ID, inv.WEIGHTS_ID, architecture="glm5_next"),
        plan.specs,
        external=external,
        geometry=_geometry_block(geometry, token_domain=tokenizer_domain(frontend_dir)),
        layer_types=geometry.layer_types,
    ) as writer:
        for index, spec in enumerate(specs, start=1):
            t0 = time.perf_counter()
            if getattr(spec, "runs", ()):
                continue  # served from the GGUF; the artifact carries no bytes for it
            if isinstance(spec, inv.ResourceSpec):
                # A text-only release ships no image or video processor config. `load_resources`
                # leaves them out and the plan drops them; the artifact then carries no such
                # object and the engine reads that as "never asked for a pixel".
                if spec.name not in resources:
                    continue
                payload: bytes = resources[spec.name]
            elif spec.name in materialized:
                payload = materialize_unspoken(source, spec.name, spec)
            else:
                tensor = rcp.materialize_recipe(recipes[spec.name], reader)
                payload = encode_tensor(tensor, spec)
                del tensor
            writer.write(spec.name, payload)
            del payload
            if index % 100 == 0 or index == total:
                print(f"[{index}/{total}] {spec.name} ({time.perf_counter() - t0:.1f}s)",
                      flush=True)

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
        "index_topk": geometry.index_topk,
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
    parser.add_argument("--frontend", required=True,
                        help="directory with tokenizer/chat template/configs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda",
                        help="accepted for a uniform call across the converters; unused")
    args = parser.parse_args(argv)
    convert(args.gguf, args.frontend, args.out, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
