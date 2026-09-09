"""Every plane the engine's vision tower produces, against the tower the checkpoint ships.

Qwen3-VL taps its encoder partway up -- at vision layers 5, 11 and 17 for the 2B -- and
sends each tap through a merger of its own. Those deepstack planes reach the text stack as
additions at its first few layers, so a plane that is subtly wrong does not fail; it
degrades, in a way that looks like the model being worse at images.

The qwen3.5 counterpart to this compares one plane, because that family's towers declare no
deepstack indexes and produce none. This compares all of them, and also compares them to
each other: three planes that were accidentally the same tensor would pass every per-plane
check, since the reference would be compared against a copy of the right answer.

Run against a checkpoint and its converted artifact:

    python -m surogate.serve.tools.parity.qwen3_vl.vision \\
        --model-dir ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/<rev> \\
        --device 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

#: Grids to encode, as (height, width) in merged blocks. Small enough to run in seconds,
#: various enough that a shape-dependent dispatch bug has somewhere to show.
GRIDS = ((2, 2), (4, 6), (8, 8))


def load_hf_vision(model_dir: Path):
    """The checkpoint's own tower, weights and all, on the CPU in BF16."""
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_config(config.vision_config).to(dtype=torch.bfloat16)

    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        weight_map = json.loads(index_file.read_text())["weight_map"]
    else:  # a small checkpoint fits in one shard, and then there is no index
        with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as source:
            weight_map = {name: "model.safetensors" for name in source.keys()}

    shards: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if name.startswith("model.visual."):
            shards.setdefault(shard, []).append(name)
    state = {}
    for shard, names in sorted(shards.items()):
        with safe_open(model_dir / shard, framework="pt", device="cpu") as source:
            for name in names:
                state[name.removeprefix("model.visual.")] = source.get_tensor(name)
    missing, unexpected = model.load_state_dict(state)
    if missing or unexpected:
        raise RuntimeError(f"HF vision state mismatch: missing={missing}, unexpected={unexpected}")
    return model.eval(), config


def agreement(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    """Cosine and RMSE relative to the reference's own scale."""
    a, b = actual.float().flatten(), expected.float().flatten()
    rmse = (a - b).square().mean().sqrt()
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=0)
    return float(cosine), float(rmse / b.square().mean().sqrt())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="the source checkpoint")
    parser.add_argument("--weights", help="a converted artifact; converts one if omitted")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ordinal")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--floor", type=float, default=0.98,
                        help="fail below this cosine on any plane")
    args = parser.parse_args()

    from surogate import _surogate_serve

    model_dir = Path(args.model_dir).resolve()
    model, config = load_hf_vision(model_dir)
    vision = config.vision_config
    merge = vision.spatial_merge_size
    patch_dim = 3 * vision.temporal_patch_size * vision.patch_size**2
    taps = list(getattr(vision, "deepstack_visual_indexes", ()) or ())
    print(f"vision: depth {vision.depth}, deepstack at {taps or 'none'}, "
          f"hidden {vision.hidden_size} -> {vision.out_hidden_size}")

    if args.weights:
        artifact = Path(args.weights)
    else:
        from surogate.serve.ingest import ensure_engine_weights

        artifact = ensure_engine_weights(str(model_dir), echo=lambda message: None)
    encoder = _surogate_serve.VisionEncoder(str(artifact), args.device)
    print(f"artifact: {artifact.name}\ngeometry: {encoder.geometry}\n")

    rng = np.random.default_rng(args.seed)
    worst = 1.0
    for height_blocks, width_blocks in GRIDS:
        height, width = height_blocks * merge, width_blocks * merge
        patches = torch.from_numpy(
            rng.standard_normal((height * width, patch_dim), dtype=np.float32)).bfloat16()

        with torch.no_grad():
            reference = model(patches, grid_thw=torch.tensor([[1, height, width]], dtype=torch.long))
        expected = [reference.pooler_output, *reference.deepstack_features]

        flat = patches.contiguous().view(torch.uint16).numpy().reshape(-1)
        planes = torch.from_dlpack(encoder.encode(flat, 1, height, width, "image")).cpu()
        if planes.shape[0] != len(expected):
            raise SystemExit(f"tower returned {planes.shape[0]} planes, checkpoint has {len(expected)}")

        print(f"grid 1x{height}x{width} -> {planes.shape[0]} planes of {tuple(planes.shape[1:])}")
        for i, want in enumerate(expected):
            cosine, relative = agreement(planes[i], want)
            worst = min(worst, cosine)
            label = "merged" if i == 0 else f"deepstack {i - 1} (vision layer {taps[i - 1]})"
            print(f"  plane {i} {label:<38} cosine {cosine:.5f}  rel_rmse {relative:.4f}")
        for i in range(1, planes.shape[0]):
            cosine, _ = agreement(planes[i], planes[0])
            print(f"  plane {i} against plane 0, which it must not equal   cosine {cosine:+.5f}")
            if torch.equal(planes[i], planes[0]):
                raise SystemExit(f"plane {i} is a copy of plane 0")

    print(f"\nworst cosine across every plane and size: {worst:.5f}")
    if worst < args.floor:
        raise SystemExit(f"below the {args.floor} floor")


if __name__ == "__main__":
    main()
