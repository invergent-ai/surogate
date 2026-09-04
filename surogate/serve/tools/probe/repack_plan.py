"""The GGUF repack plan, examined offline.

Rebuilds the bridge's candidate pre-walk for one GGUF, runs the ingest planner over it, and
then replays the converter's own coverage check on what the planner kept -- the check that
raises ``RepackError: repack map names sources still needed by materialized recipes``. A
failed conversion is thereby examined in about a minute instead of after a bridge run, and
without keeping the bridge's dequantised copy around.

    python -m surogate.serve.tools.probe.repack_plan <model.gguf> [converter key]

Prints the candidate and kept counts, the converter's planned/native/halves counts, every
stray source with the objects that read it and their coverage, and -- for the attention
parents of two layers -- both sides' view, which is where a mixed-type UD mixture shows.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from surogate.serve.convert.common.gguf_repack import GgufRepackSource
from surogate.serve.gguf import bridge as B
from surogate.serve.ingest import _gguf_geometry, _repack_planner, _sinfer_root


def main() -> int:
    gguf_path = Path(sys.argv[1])
    target = sys.argv[2] if len(sys.argv) > 2 else B.gguf_converter_key(gguf_path)
    reader = B.open_gguf(gguf_path)
    arch = reader.get_field("general.architecture").contents()
    n_layers = int(B._arch_kv(reader, arch, "block_count", 0))
    fam = importlib.import_module("surogate.serve.gguf.qwen35")
    n_mtp = int(B._arch_kv(reader, arch, "nextn_predict_layers", 0) or 0); n_main = n_layers - n_mtp
    num_v = int(B._arch_kv(reader, arch, "ssm.time_step_rank", 0) or 0); inner = int(B._arch_kv(reader, arch, "ssm.inner_size", 0) or 0)
    geom = fam.GdnGeometry(num_k_heads=int(B._arch_kv(reader, arch, "ssm.group_count", 0) or 0), num_v_heads=num_v,
                           head_k_dim=int(B._arch_kv(reader, arch, "ssm.state_size", 0) or 0), head_v_dim=(inner // num_v) if num_v else 0)
    name_map = B._hf_name_map(arch, n_layers)
    cands = {}
    for t in reader.tensors:
        hf = B._family_or_generic(fam, t.name, n_main, name_map)
        if hf is None or len(t.shape) < 2: continue
        shape = tuple(int(e) for e in reversed(t.shape)); rows = 1
        for e in shape[:-1]: rows *= e
        row_perm = fam.inverse_row_permutation(hf, geom, rows); col_groups = fam.inverse_column_group_map(hf, geom, 32)
        if row_perm is None and col_groups is None and not fam.inverse_is_row_identity(hf, geom): continue
        cands[hf] = {"name": t.name, "shape": list(shape), "rows": rows, "k": shape[-1], "offset": int(t.data_offset), "type": t.type_name,
                     "row_perm": None if row_perm is None else [int(v) for v in row_perm], "col_groups": None if col_groups is None else [int(v) for v in col_groups]}
    for n in list(cands):
        if B._has_export_transform(arch, n): cands.pop(n)
    plan = _repack_planner(_sinfer_root(), target)
    keep = plan(gguf_path, cands)
    print(f"candidates {len(cands)}, kept {len(keep)}")
    # the converter's side
    inv = importlib.import_module(f"surogate.serve.convert.{target}.inventory"); rcp = importlib.import_module(f"surogate.serve.convert.{target}.recipe")
    conv = importlib.import_module(f"surogate.serve.convert.{target}.convert")
    geometry = _gguf_geometry(rcp, inv, gguf_path)
    recipes = conv.active_recipes(mtp=False, vision=False, geometry=geometry)
    recipes_by_name = {r.object_name: r for r in recipes} if not isinstance(recipes, dict) else recipes
    specs = inv.build_tensor_specs(geometry) if hasattr(inv, "build_tensor_specs") else inv.active_specs(mtp=False, vision=False, geometry=geometry)[0]
    src = GgufRepackSource.from_sources(gguf_path, keep)
    planned = set(src.plan(recipes_by_name, specs)); native = set(src.plan_native(recipes_by_name, specs, exclude_suffixes=getattr(rcp, "NATIVE_EXCLUDE_SUFFIXES", ())))
    halves = set(src.plan_native_halves(recipes_by_name, specs))
    covered = planned | native | halves
    stray = {}
    for name, r in recipes_by_name.items():
        if name in covered: continue
        for s_ in rcp.expression_sources(r.expression):
            if s_.name in src.sources: stray.setdefault(s_.name, []).append(name)
    print(f"converter: planned {len(planned)} native {len(native)} halves {len(halves)} stray {len(stray)}")
    for s_, objs in list(stray.items())[:4]:
        readers = [n for n, r in recipes_by_name.items() if any(x.name == s_ for x in rcp.expression_sources(r.expression))]
        print(f"  {s_} ({cands[s_]['type']}): read by {[(n, 'covered' if n in covered else 'UNCOVERED') for n in readers]}")

    # ---- the planner's own view of the same objects
    p_recipes = {r.object_name: r for r in rcp.build_recipes(geometry)}
    p_specs = inv.build_tensor_specs(geometry) if hasattr(inv, "build_tensor_specs") else inv.active_specs(mtp=True, vision=True, geometry=geometry)[0]
    p_src = GgufRepackSource.from_sources(gguf_path, cands)
    p_planned = set(p_src.plan(p_recipes, p_specs)); p_native = set(p_src.plan_native(p_recipes, p_specs, exclude_suffixes=getattr(rcp, "NATIVE_EXCLUDE_SUFFIXES", ())))
    p_halves = set(p_src.plan_native_halves(p_recipes, p_specs)); p_cov = p_planned | p_native | p_halves
    print(f"planner:   planned {len(p_planned)} native {len(p_native)} halves {len(p_halves)}")
    for layer in (3, 7):
        for src_name in (f"model.layers.{layer}.self_attn.q_proj.weight", f"model.layers.{layer}.self_attn.k_proj.weight", f"model.layers.{layer}.self_attn.v_proj.weight"):
            readers = [n for n, r in p_recipes.items() if any(x.name == src_name for x in rcp.expression_sources(r.expression))]
            print(f"  L{layer} {src_name.split('.')[-2]} {cands.get(src_name, {}).get('type')} kept={src_name in keep}: " +
                  ", ".join(f"{n.split('/')[-1]}[planner {'cov' if n in p_cov else 'UNC'} / converter {'cov' if n in covered else ('UNC' if n in recipes_by_name else 'absent')}]" for n in readers))
    # spec formats the two sides assign to the gate_value object of layer 3
    for label, sp in (("planner", p_specs), ("converter", specs)):
        for s_ in sp:
            if s_.name == "text/layers/3/attention/gate_value": print(f"  {label} spec: {s_.name} format={s_.format} shape={tuple(s_.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
