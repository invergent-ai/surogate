"""Text weights follow the DSL; vision recipes include every configured deepstack merger."""

from surogate.serve.convert.common.declaration import derive_recipes
from surogate.serve.convert.common.recipe import build_vision_recipes, TensorRecipe, SourceTensor
from surogate.serve.convert.qwen3.recipe import open_reader, preflight_sources
from .inventory import geometry_from_config, merger_tensors


def build_recipes(g):
    v = g.declared.hf_config["vision_config"]
    return (
        *derive_recipes(g.declared, capabilities={"text"},
                        tied_output_head=g.declared.hf_config.get("tie_word_embeddings", False)),
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
