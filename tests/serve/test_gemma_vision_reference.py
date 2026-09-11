"""Compare the native image encoder with Transformers on a supplied Gemma checkpoint.

SUROGATE_GEMMA_VISION_MODEL points to the HF checkpoint, SUROGATE_GEMMA_VISION_ARTIFACT
 to its converted artifact, and SUROGATE_GEMMA_VISION_TEST_BIN to sinfer_gemma_vision_test.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest


def _weights(model, prefix):
    from safetensors import safe_open

    tensors = {}
    for shard in model.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key.startswith(prefix):
                    tensors[key[len(prefix) :]] = reader.get_tensor(key)
    return tensors


def test_checkpoint_vision_matches_transformers(tmp_path):
    model_path = os.environ.get("SUROGATE_GEMMA_VISION_MODEL")
    artifact = os.environ.get("SUROGATE_GEMMA_VISION_ARTIFACT")
    binary = os.environ.get("SUROGATE_GEMMA_VISION_TEST_BIN")
    if not all((model_path, artifact, binary)):
        pytest.skip("set Gemma checkpoint, artifact and native test executable")
    import torch
    from transformers import AutoConfig

    from surogate.serve.convert.gemma_vl.inventory import geometry_from_config

    model = Path(model_path)
    config = AutoConfig.from_pretrained(model)
    geometry = geometry_from_config(json.loads((model / "config.json").read_text()))
    v = geometry.vision
    torch.manual_seed(732)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Match the default fused attention path, whose scores stay in FP32.
    # Eager attention rounds the QK product and scaled scores to BF16 separately.
    config.vision_config._attn_implementation = "sdpa"
    if v["gemma_version"] == 3:
        from transformers.models.gemma3.modeling_gemma3 import Gemma3MultiModalProjector
        from transformers.models.siglip.modeling_siglip import SiglipVisionModel

        tower = SiglipVisionModel(config.vision_config).to(dtype=torch.bfloat16, device="cuda")
        state = _weights(model, "vision_tower.") or _weights(model, "model.vision_tower.")
        state = {key.removeprefix("vision_model."): value for key, value in state.items()}
        tower.load_state_dict(state)
        projector = Gemma3MultiModalProjector(config).to(dtype=torch.bfloat16, device="cuda")
        state = _weights(model, "multi_modal_projector.") or _weights(model, "model.multi_modal_projector.")
        projector.load_state_dict(state)
        patch = config.vision_config.patch_size
        height = width = config.vision_config.image_size // patch
        pixels = torch.rand(1, 3, height * patch, width * patch, device="cuda") * 2 - 1
        pixels = pixels.bfloat16()
        patches = pixels.reshape(1, 3, height, patch, width, patch).permute(0, 2, 4, 1, 3, 5).reshape(height, width, -1)
        with torch.no_grad():
            expected = projector(tower(pixel_values=pixels).last_hidden_state)
    else:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder, Gemma4VisionModel

        tower = Gemma4VisionModel(config.vision_config).to(dtype=torch.bfloat16, device="cuda")
        tower.load_state_dict(_weights(model, "model.vision_tower."))
        projector = Gemma4MultimodalEmbedder(config.vision_config, config.text_config).to(
            dtype=torch.bfloat16, device="cuda"
        )
        projector.load_state_dict(_weights(model, "model.embed_vision."))
        height, width = 6, 9
        pixels = torch.rand(1, height * width, v["patch_dim"], device="cuda")
        yy, xx = torch.meshgrid(torch.arange(height, device="cuda"), torch.arange(width, device="cuda"), indexing="ij")
        positions = torch.stack((xx, yy), -1).reshape(1, -1, 2)
        patches = (2 * (pixels - 0.5)).bfloat16().reshape(height, width, -1)
        with torch.no_grad():
            expected = projector(tower(pixel_values=pixels, pixel_position_ids=positions).last_hidden_state)
    merge = v["merge"]
    patches = patches.reshape(height // merge, merge, width // merge, merge, -1).permute(0, 2, 1, 3, 4).contiguous()
    input_path, output_path = tmp_path / "patches.bin", tmp_path / "output.bin"
    input_path.write_bytes(patches.cpu().view(torch.uint16).numpy().tobytes())
    expected = expected.float().cpu().reshape(-1)
    del tower, projector, patches, pixels
    torch.cuda.empty_cache()
    subprocess.run([binary, artifact, str(input_path), str(height), str(width), str(output_path), "1"], check=True)
    actual = torch.frombuffer(bytearray(output_path.read_bytes()), dtype=torch.bfloat16).float()
    assert torch.isfinite(actual).all()
    relative = (actual - expected).norm() / expected.norm()
    cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=0)
    print(f"Gemma vision relative L2={relative:.6f}, cosine={cosine:.7f}")
    assert relative < 0.04
    assert cosine > 0.999


@pytest.mark.parametrize("kind", ["clipped", "standardized72", "unified"])
def test_generated_vision_variants_match_transformers(tmp_path, kind):
    binary = os.environ.get("SUROGATE_GEMMA_VISION_TEST_BIN")
    if not binary:
        pytest.skip("set SUROGATE_GEMMA_VISION_TEST_BIN")
    from types import SimpleNamespace

    import torch
    from safetensors.torch import save_file
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig, Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder, Gemma4VisionModel

    from surogate.serve.artifact.container import ArtifactIdentity, ArtifactWriter
    from surogate.serve.convert.common import conversion
    from surogate.serve.convert.common.recipe import materialize_recipe
    from surogate.serve.convert.common.safetensors import ShardReader
    from surogate.serve.convert.gemma_vl.inventory import Geometry, vision_recipes

    torch.manual_seed(993)
    free = kind == "unified"
    h = 144 if kind == "standardized72" else 64
    tc = Gemma4TextConfig(hidden_size=128)
    if free:
        from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedVisionConfig
        from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedVisionEmbedder

        vc = Gemma4UnifiedVisionConfig(mm_embed_dim=h, mm_posemb_size=32, output_proj_dims=h)
        tower = Gemma4UnifiedVisionEmbedder(vc, tc).bfloat16().cuda()
        with torch.no_grad():
            tower.pos_embedding.normal_(std=0.02)
        state = {"model.embed_vision." + k: v.contiguous().cpu() for k, v in tower.state_dict().items()}
    else:
        vc = Gemma4VisionConfig(
            hidden_size=h,
            intermediate_size=2 * h,
            num_hidden_layers=2,
            num_attention_heads=2 if h == 144 else 1,
            num_key_value_heads=2 if h == 144 else 1,
            head_dim=72 if h == 144 else 64,
            position_embedding_size=32,
            standardize=kind == "standardized72",
            use_clipped_linears=kind == "clipped",
        )
        vc._attn_implementation = "eager"
        tower = Gemma4VisionModel(vc).bfloat16().cuda()
        if vc.standardize:
            tower.std_bias = torch.linspace(-0.3, 0.3, h, device="cuda")
            tower.std_scale = torch.linspace(0.5, 1.5, h, device="cuda")
        if vc.use_clipped_linears:
            for module in tower.modules():
                if hasattr(module, "input_min"):
                    module.input_min.fill_(-0.3)
                    module.input_max.fill_(0.4)
                    module.output_min.fill_(-0.1)
                    module.output_max.fill_(0.2)
        projector = Gemma4MultimodalEmbedder(vc, tc).bfloat16().cuda()
        state = {"model.vision_tower." + k: v.contiguous().cpu() for k, v in tower.state_dict().items()}
        state.update({"model.embed_vision." + k: v.contiguous().cpu() for k, v in projector.state_dict().items()})
    merge = 1 if free else 3
    patch_dim = 6912 if free else 768
    height, width = (2, 3) if free else (6, 9)
    pixels = torch.rand(1, height * width, patch_dim, device="cuda")
    yy, xx = torch.meshgrid(torch.arange(height, device="cuda"), torch.arange(width, device="cuda"), indexing="ij")
    positions = torch.stack((xx, yy), -1).reshape(1, -1, 2)
    with torch.no_grad():
        if free:
            expected = tower(pixels, image_position_ids=positions, return_dict=True).pooler_output
        else:
            expected = projector(tower(pixel_values=pixels, pixel_position_ids=positions).last_hidden_state)
    vision = dict(
        gemma_version=4,
        encoder_free=int(free),
        layers=0 if free else 2,
        hidden=h,
        intermediate=h if free else 2 * h,
        heads=1 if free or h == 64 else 2,
        patch_dim=patch_dim,
        merge=merge,
        position_embeddings=32,
        rotary_dim=0 if free else vc.head_dim,
        output_hidden=128,
        rope_theta=0.0 if free else 100.0,
        norm_epsilon=1e-6,
        clipped_linears=int(kind == "clipped"),
        standardize=int(kind == "standardized72"),
        attention_mode=0,
        max_image_tokens=6,
    )
    g = Geometry(SimpleNamespace(hidden=128), vision, "gemma4", {"vision_config": vc.to_dict()})
    specs, recipes = vision_recipes(g)
    save_file(state, str(tmp_path / "model.safetensors"))
    artifact = tmp_path / "vision.sinfer"
    with ShardReader.for_directory(tmp_path) as reader:
        with ArtifactWriter(
            artifact, ArtifactIdentity("gemma4", "groupwise-int", architecture="gemma4"), conversion.build_object_plan(specs, {}).specs, vision_geometry=vision
        ) as writer:
            for spec, recipe in zip(specs, recipes, strict=True):
                value = materialize_recipe(recipe, reader)
                if spec.format == "FP32":
                    value = value.float()
                writer.write(spec.name, conversion.encode_tensor_payload(value, spec, "cpu"))
    patches = pixels if free else 2 * (pixels - 0.5)
    patches = (
        patches.bfloat16()
        .reshape(height // merge, merge, width // merge, merge, -1)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )
    inp, out = tmp_path / "patches.bin", tmp_path / "out.bin"
    inp.write_bytes(patches.cpu().view(torch.uint16).numpy().tobytes())
    subprocess.run([binary, str(artifact), str(inp), str(height), str(width), str(out), "1"], check=True)
    actual = torch.frombuffer(bytearray(out.read_bytes()), dtype=torch.bfloat16).float()
    expected = expected.float().cpu().reshape(-1)
    relative = (actual - expected).norm() / expected.norm()
    print(f"{kind}: relative L2={relative:.6f}")
    assert relative < 0.025
