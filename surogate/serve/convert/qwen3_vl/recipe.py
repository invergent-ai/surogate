"""Text weights follow the DSL; vision recipes include every configured deepstack merger."""

from surogate.serve.convert.common.declaration import derive_recipes
from surogate.serve.convert.common.recipe import (
    AnyOf,
    Reshape,
    SourceTensor,
    TensorRecipe,
    Transpose,
    build_vision_recipes,
)
from surogate.serve.convert.qwen3.recipe import open_reader as open_reader
from surogate.serve.convert.qwen3.recipe import preflight_sources as preflight_sources

from .inventory import geometry_from_config as geometry_from_config
from .inventory import merger_tensors


def _packed_expert_expressions(g, reader):
    """Read both banks to distinguish original input-major and newer output-major exports.

    Either bank may be square, but both cannot be, so the pair resolves the layout
    without guessing from a Transformers version or a checkpoint's filename.
    """
    overrides = {}
    for layer in range(g.layers):
        prefix = f"model.layers.{layer}.mlp.experts."
        gate, down = prefix + "gate_up_proj", prefix + "down_proj"
        if reader.has(gate):
            if not reader.has(down):
                raise ValueError(f"{gate}: packed expert bank has no down projection")
            metadata = reader.metadata((gate, down))
            shapes = (metadata[gate].shape, metadata[down].shape)
            output_shapes = ((g.experts, 2 * g.intermediate, g.hidden),
                             (g.experts, g.hidden, g.intermediate))
            input_shapes = tuple((e, cols, rows) for e, rows, cols in output_shapes)
            if shapes not in (input_shapes, output_shapes):
                raise ValueError(f"layer {layer}: incompatible packed expert shapes {shapes}")
            for name, shape, output_shape, target in zip(
                    (gate, down), shapes, output_shapes, ("routed_gate_up", "routed_down")):
                expression = SourceTensor(name, shape)
                if shapes == input_shapes:
                    expression = Transpose(expression, (0, 2, 1))
                overrides[f"text/layers/{layer}/moe/{target}"] = Reshape(
                    expression, (output_shape[0] * output_shape[1], output_shape[2]))
        elif reader.has(down):
            # Separate stacked gate/up tensors use the conventional output-major bank.
            overrides[f"text/layers/{layer}/moe/routed_down"] = Reshape(
                SourceTensor(down, (g.experts, g.hidden, g.intermediate)),
                (g.experts * g.hidden, g.intermediate))
    return overrides


def build_recipes(g, *, reader=None):
    v = g.declared.hf_config["vision_config"]
    text = derive_recipes(g.declared, capabilities={"text"},
                          tied_output_head=g.declared.hf_config.get("tie_word_embeddings", False))
    if g.experts:
        overrides = _packed_expert_expressions(g, reader) if reader is not None else {}
        # Released VL-MoE checkpoints pack each bank as [expert, input, output].
        # Also accept per-expert tensors and the output-major banks used by GGUF.
        converted = []
        for r in text:
            if r.object_name in overrides:
                converted.append(TensorRecipe(r.object_name, overrides[r.object_name]))
                continue
            if r.object_name.endswith(("/moe/routed_gate_up", "/moe/routed_down")):
                layer = r.object_name.split("/")[2]
                gate_up = r.object_name.endswith("routed_gate_up")
                inputs, outputs = (g.hidden, 2 * g.intermediate) if gate_up else (g.intermediate, g.hidden)
                name = f"model.layers.{layer}.mlp.experts.{'gate_up_proj' if gate_up else 'down_proj'}"
                packed = Reshape(Transpose(SourceTensor(name, (g.experts, inputs, outputs)),
                                            (0, 2, 1)), (g.experts * outputs, inputs))
                original = r.expression
                alternatives = (tuple(Reshape(option, original.shape) for option in original.source.options)
                                if isinstance(original, Reshape) and isinstance(original.source, AnyOf)
                                else (original,))
                r = TensorRecipe(r.object_name, AnyOf((packed, *alternatives)))
            converted.append(r)
        text = tuple(converted)
    return (
        *text,
        *build_vision_recipes(
            g.hidden, layers=v["depth"], hidden=v["hidden_size"],
            intermediate=v["intermediate_size"], qkv_rows=3 * v["hidden_size"],
            patch_channels=v["in_channels"], patch_temporal=v["temporal_patch_size"],
            patch_size=v["patch_size"], position_embeddings=v["num_position_embeddings"],
            merger_hidden=v["hidden_size"] * v["spatial_merge_size"] ** 2,
        ),
        *(TensorRecipe(name, SourceTensor(source, shape))
          for name, source, shape, _ in merger_tensors(g)),
    )
