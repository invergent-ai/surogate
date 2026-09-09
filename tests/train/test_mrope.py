"""Our MRoPE against the one it replaced.

`surogate.train.mrope` reproduces transformers' `get_rope_index` so the training path need
not build a model to reach it. A reimplementation of a convention is only worth what it is
checked against, and there is exactly one thing worth checking it against, so this compares
the two directly on random prompts -- images, multi-frame video, padding, several rows --
and demands equality, not closeness. These are indices.

The comparison needs a real config for its geometry and a meta-device skeleton to call the
original on; both come from a cached checkpoint, and the module skips when there is none.
"""

import glob
import itertools
import random

import pytest
import torch

from surogate.train.mrope import MROPE_FAMILIES, mrope_position_ids, rope_index_fn

CHECKPOINT = "models--Qwen--Qwen3.5-0.8B"


@pytest.fixture(scope="module")
def reference():
    """transformers' own `get_rope_index`, plus the config the prompts are built from."""
    hits = sorted(glob.glob(
        f"/home/densemax2/.cache/huggingface/hub/{CHECKPOINT}/snapshots/*/config.json"))
    if not hits:
        pytest.skip(f"no cached {CHECKPOINT}")
    from transformers import AutoConfig, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(hits[0].rsplit("/", 1)[0], trust_remote_code=True)
    with torch.device("meta"):
        skeleton = AutoModelForImageTextToText.from_config(config)
    model = skeleton if hasattr(skeleton, "get_rope_index") else skeleton.model
    return model, config


def _prompt(rng, config, *, videos, pad):
    """A batch of prompts: text, some pictures, text -- optionally right-padded.

    A video is laid down as its frames separated by a timestamp token, because that is how
    the processor emits one and why the model splits a T-frame grid into T grids of one.
    """
    merge = config.vision_config.spatial_merge_size
    image_id, video_id = config.image_token_id, config.video_token_id
    rows, images, clips = [], [], []
    for _ in range(rng.randint(1, 3)):
        tokens, kinds = [], []
        for _ in range(rng.randint(0, 2)):
            run = rng.randint(1, 3)
            tokens += [7] * run
            kinds += [0] * run
            height = merge * rng.randint(1, 3)
            width = merge * rng.randint(1, 3)
            per_frame = height * width // (merge * merge)
            if not videos or rng.random() < 0.5:
                tokens += [image_id] * per_frame
                kinds += [1] * per_frame
                images.append([1, height, width])
            else:
                frames = rng.randint(1, 3)
                for frame in range(frames):
                    if frame:
                        tokens += [7]
                        kinds += [0]
                    tokens += [video_id] * per_frame
                    kinds += [2] * per_frame
                clips.append([frames, height, width])
        run = rng.randint(1, 4)
        tokens += [7] * run
        kinds += [0] * run
        rows.append((tokens, kinds))

    width = max(len(t) for t, _ in rows) + (rng.randint(1, 3) if pad else 0)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    modality = torch.zeros(len(rows), width, dtype=torch.int32)
    mask = torch.zeros(len(rows), width, dtype=torch.long)
    for i, (tokens, kinds) in enumerate(rows):
        ids[i, :len(tokens)] = torch.tensor(tokens)
        modality[i, :len(kinds)] = torch.tensor(kinds)
        mask[i, :len(tokens)] = 1
    return (ids, modality, mask,
            torch.tensor(images, dtype=torch.long) if images else None,
            torch.tensor(clips, dtype=torch.long) if clips else None)


@pytest.mark.parametrize("videos,pad,masked", list(itertools.product((False, True), repeat=3)))
def test_positions_match_transformers_exactly(reference, videos, pad, masked):
    """Fifty random batches per arrangement; every index must agree.

    Padding and masking vary separately because they are different questions: whether the
    rows are ragged, and whether the caller tells the function so.
    """
    model, config = reference
    merge = config.vision_config.spatial_merge_size
    rng = random.Random(hash((videos, pad, masked)) & 0xFFFF)

    for _ in range(50):
        ids, modality, mask, images, clips = _prompt(rng, config, videos=videos, pad=pad)
        attention = mask if masked else None
        if pad and not masked:
            continue  # ragged rows without a mask are not a case the caller produces
        expected, _ = model.get_rope_index(
            ids, modality,
            image_grid_thw=None if images is None else images.clone(),
            video_grid_thw=None if clips is None else clips.clone(),
            attention_mask=attention)
        actual = mrope_position_ids(
            ids, modality, spatial_merge_size=merge,
            image_grid_thw=images, video_grid_thw=clips, attention_mask=attention)
        assert torch.equal(expected, actual), (
            f"ids={ids.tolist()} modality={modality.tolist()} "
            f"images={None if images is None else images.tolist()} "
            f"videos={None if clips is None else clips.tolist()}\n"
            f"expected={expected.tolist()}\nactual={actual.tolist()}")


def test_the_caller_is_handed_the_grids_untouched():
    """transformers rewrites a video grid's frame count in place; ours must not.

    The same grids go on to the vision tower, which needs the frame count it was given.
    """
    # Two frames of a 2x2 grid at merge 2: one token each, separated by a timestamp.
    clips = torch.tensor([[2, 2, 2]], dtype=torch.long)
    before = clips.clone()
    ids = torch.full((1, 4), 7, dtype=torch.long)
    modality = torch.tensor([[0, 2, 0, 2]], dtype=torch.int32)
    mrope_position_ids(ids, modality, spatial_merge_size=2, video_grid_thw=clips)
    assert torch.equal(clips, before)


class TestTheRefusal:
    def test_an_unchecked_family_is_refused_by_name(self):
        """Silence is the failure mode here: wrong positions train, they do not crash."""

        class Config:
            model_type = "qwen2_vl"

        with pytest.raises(ValueError, match="no verified MRoPE layout"):
            rope_index_fn(Config())

    def test_a_checked_family_reports_its_merge_size(self, reference):
        _, config = reference
        assert config.model_type in MROPE_FAMILIES
        rope_fn = rope_index_fn(config)
        ids = torch.full((1, 3), 7, dtype=torch.long)
        modality = torch.zeros((1, 3), dtype=torch.int32)
        assert rope_fn(ids, mm_token_type_ids=modality).shape == (3, 1, 3)
