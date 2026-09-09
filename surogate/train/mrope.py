"""MRoPE position indices, computed from the config rather than from a model.

`get_rope_index` is not a model: given token ids, their modality and the grids the
processor produced, it returns where each token sits on the three MRoPE axes, out of
config values alone. Calling it meant building a transformers model skeleton for a
function that reads no weights -- and that skeleton is the last thing on the training path
that instantiates an architecture we already serve ourselves.

The arithmetic below is transformers' own, and deliberately so: this is a reimplementation,
not an improvement. What it must reproduce is not a formula but a convention -- which axis
counts what, where an image's positions start, how much a picture advances the clock for
the text after it. Any of those chosen differently is a model that has been trained to read
positions one way and is being handed them in another, which shows up as worse loss and
nothing else.
"""

from __future__ import annotations

import itertools

import torch

#: The families whose `get_rope_index` this reproduces. All five carry the same
#: implementation in transformers, line for line -- Qwen2-VL and Qwen2.5-VL do not, which
#: is why this is a list and not an assumption. It is also, and not by coincidence, the set
#: whose vision towers the serving engine can bind: a family that cannot reach the engine
#: tower cannot reach this function either.
MROPE_FAMILIES = frozenset(
    {"qwen3_5", "qwen3_5_moe", "qwen3_vl", "qwen3_vl_moe", "qwen4_exp"}
)

#: The modality codes the processor writes into `mm_token_type_ids`.
TEXT, IMAGE, VIDEO = 0, 1, 2


def vision_position_ids(start: int, grid_thw, spatial_merge_size: int) -> torch.Tensor:
    """The three axes for one image or one video frame, offset to `start`.

    Height and width count within the picture, so two tokens on the same row share a
    height; the temporal axis counts frames. All three begin at `start`, which is why an
    image occupies one position per row of merged patches rather than one per token: the
    text that follows resumes at `start + max(height, width)`.
    """
    t = int(grid_thw[0])
    h = int(grid_thw[1]) // spatial_merge_size
    w = int(grid_thw[2]) // spatial_merge_size

    temporal = torch.arange(t)
    height = torch.arange(h) + start
    width = torch.arange(w) + start
    axes = torch.meshgrid(temporal, height, width, indexing="ij")
    positions = torch.stack(axes, dim=0).reshape(3, -1)
    positions[0] += start
    return positions


def mrope_position_ids(
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    *,
    spatial_merge_size: int,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """`[3, batch, sequence]` positions for a batch of prompts.

    The grids are consumed in order across the whole batch, not per row, because that is
    the order the processor concatenated them in: the second row's first image is the
    batch's third grid if the first row held two.

    Padding is skipped rather than numbered -- positions are laid down only where
    `attention_mask` is set, so a right-padded row's tail keeps the zeros it started with
    and a left-padded one still begins its text at zero.
    """
    if video_grid_thw is not None and video_grid_thw.numel():
        # A video arrives as one grid of T frames and is read as T grids of one frame:
        # timestamps separate the frames in the prompt, so each gets its own positions.
        # `repeat_interleave` allocates even when every count is 1, so setting the frame
        # count below writes to a copy and the caller's grids -- which go on to the vision
        # tower, where the count still means T -- are left as they were.
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    grids = {
        IMAGE: iter(image_grid_thw) if image_grid_thw is not None else iter(()),
        VIDEO: iter(video_grid_thw) if video_grid_thw is not None else iter(()),
    }

    batch, sequence = input_ids.shape
    position_ids = torch.zeros(3, batch, sequence, dtype=input_ids.dtype)

    for row in range(batch):
        modalities = mm_token_type_ids[row]
        if attention_mask is not None:
            modalities = modalities[attention_mask[row].bool()]

        cursor = 0
        laid = []
        for modality, group in itertools.groupby(modalities.tolist()):
            span = len(list(group))
            if modality == TEXT:
                laid.append(torch.arange(span).view(1, -1).expand(3, -1) + cursor)
                cursor += span
                continue
            grid = next(grids[modality])
            laid.append(vision_position_ids(cursor, grid, spatial_merge_size))
            cursor += max(int(grid[1]), int(grid[2])) // spatial_merge_size

        if not laid:
            continue
        positions = torch.cat(laid, dim=1).reshape(3, -1)
        if attention_mask is not None:
            position_ids[:, row, attention_mask[row].bool()] = positions.to(position_ids.dtype)
        else:
            position_ids[:, row] = positions.to(position_ids.dtype)

    return position_ids


def rope_index_fn(hf_config):
    """The MRoPE function for `hf_config`, or a refusal naming why there isn't one.

    A family absent from `MROPE_FAMILIES` is not necessarily incompatible -- it is
    unchecked, which is the same thing to act on. Positions computed by the wrong
    convention do not fail; they train.
    """
    model_type = getattr(hf_config, "model_type", None)
    if model_type not in MROPE_FAMILIES:
        raise ValueError(
            f"train_vision: no verified MRoPE layout for model_type {model_type!r}. "
            f"Checked families: {', '.join(sorted(MROPE_FAMILIES))}. Compare this "
            "architecture's get_rope_index against surogate/train/mrope.py and add it "
            "there if they agree -- a mismatch is silent, and costs accuracy, not a crash."
        )
    vision = getattr(hf_config, "vision_config", None)
    merge = int(getattr(vision, "spatial_merge_size", 1)) if vision is not None else 1

    def rope_fn(input_ids, *, mm_token_type_ids, image_grid_thw=None, video_grid_thw=None,
                attention_mask=None):
        return mrope_position_ids(
            input_ids, mm_token_type_ids,
            spatial_merge_size=merge,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )

    return rope_fn
