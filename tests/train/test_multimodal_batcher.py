"""What `OnTheFlyMultimodalBatcher` promises the model about visual tokens.

The batcher is the join between two things that count separately: a prompt, which marks
visual positions with a token id, and the vision tower, which produces one embedding per
merged patch. Nothing downstream re-derives the correspondence -- the model consumes a
packed buffer and a mask, and trusts that the nth marked position wants the nth embedding.

So these tests drive the batcher with a tower whose embeddings say which item and which
token they came from, and check the buffer against the token stream. A stub stands in for
the real tower on purpose: the tower's numbers are checked against transformers elsewhere,
and what is unverified here is the bookkeeping around it -- which is precisely what a real
tower would hide behind plausible-looking floats.
"""

import numpy as np
import pytest
import torch

from surogate.train.vision import OnTheFlyMultimodalBatcher

IMAGE_TOKEN = 151655
VIDEO_TOKEN = 151656
HIDDEN = 8
MERGE = 2


class StubTower:
    """A tower whose every embedding names its origin.

    Row `k` of an item's output is the constant vector `base + k`, where `base` identifies
    the item and the modality; deepstack layer `li` adds `100 * (li + 1)`. Any embedding
    that lands in the wrong slot therefore reads as a different number, not as noise.
    """

    #: Rows to hand back beyond what the grids account for, marked -1 so a trim that
    #: keeps the wrong end is visible rather than merely off by a number.
    extra_rows = 0

    def __init__(self, deepstack_layers=0):
        self.image_token_id = IMAGE_TOKEN
        self.video_token_id = VIDEO_TOKEN
        self.hidden_size = HIDDEN
        self.deepstack_layers = deepstack_layers
        self.merge = MERGE
        self.calls = []

    def features(self, pixel_values, grid_thw, *, modality):
        self.calls.append((modality, grid_thw.tolist(), int(pixel_values.shape[0])))
        offset = 0.0 if modality == "image" else 1000.0
        merged = []
        for item, row in enumerate(grid_thw.tolist()):
            count = int(np.prod(row)) // (self.merge ** 2)
            base = offset + 10.0 * (item + 1)
            merged.append(torch.arange(count, dtype=torch.float32).unsqueeze(1) + base)
        if self.extra_rows:
            merged.append(torch.full((self.extra_rows, 1), -1.0))
        stacked = torch.cat(merged).repeat(1, HIDDEN)
        deepstack = [stacked + 100.0 * (li + 1) for li in range(self.deepstack_layers)]
        return stacked, deepstack


class StubTemplate:
    """Hands back rows already encoded; the real encoder is the HF processor."""

    def encode(self, row, return_length=True):
        return row


def _rope_fn(input_ids, **kwargs):
    """Three MRoPE axes of plain positions -- shape is all the batcher uses.

    The real one is `surogate.train.mrope`, checked against transformers in test_mrope.
    """
    b, t = input_ids.shape
    return torch.arange(t).view(1, 1, t).expand(3, b, t).contiguous()


def _item(*, n_image=0, n_video=0, prefix=2, suffix=2, video_first=False):
    """One encoded row: text, then `n_image` image tokens, then `n_video` video ones.

    `video_first` swaps the two runs. It matters because it is the only arrangement in
    which an image token's position in the batch differs from its position among the
    images -- everywhere else the two indices coincide and a confusion between them is
    invisible.
    """
    visual = ([VIDEO_TOKEN] * n_video + [IMAGE_TOKEN] * n_image if video_first
              else [IMAGE_TOKEN] * n_image + [VIDEO_TOKEN] * n_video)
    tokens = [1] * prefix + visual + [1] * suffix
    row = {"input_ids": np.array(tokens, dtype=np.int32),
           "labels": np.array(tokens, dtype=np.int32)}
    patch = MERGE ** 2
    if n_image:
        row["image_grid_thw"] = torch.tensor([[1, patch, n_image]], dtype=torch.long)
        row["pixel_values"] = torch.zeros((patch * n_image * patch, 3), dtype=torch.float32)
    if n_video:
        row["video_grid_thw"] = torch.tensor([[1, patch, n_video]], dtype=torch.long)
        row["pixel_values_videos"] = torch.zeros((patch * n_video * patch, 3), dtype=torch.float32)
    return row


def _batcher(rows, *, tower, batch_size=2, seq_len=16):
    return OnTheFlyMultimodalBatcher(
        dataset=rows,
        template_processor=StubTemplate(),
        vision=tower,
        vision_device=torch.device("cpu"),
        rope_fn=_rope_fn,
        batch_size=batch_size,
        seq_len=seq_len,
        pad_token_id=0,
        seed=0,
        shuffle=False,
        repeat=True,
    )


def _expected_sequence(batch, tower):
    """The embeddings the marked positions ask for, in the order the buffer packs them."""
    flat = batch["inputs"].reshape(-1)
    image_seen, video_seen, wanted = 0, 0, []
    for token in flat:
        if token == IMAGE_TOKEN:
            wanted.append(("image", image_seen))
            image_seen += 1
        elif token == VIDEO_TOKEN:
            wanted.append(("video", video_seen))
            video_seen += 1
    return wanted


class TestPacking:
    def test_embeddings_are_packed_in_token_order(self):
        tower = StubTower()
        batch = _batcher([_item(n_image=3), _item(n_image=2)], tower=tower).next_batch()

        num_visual = int(batch["visual_pos_masks"].sum())
        assert num_visual == 5
        # Item 1 contributed rows 10,11,12 and item 2 rows 20,21 -- in that order.
        assert [v[0] for v in batch["visual_embeds"][:num_visual]] == [10, 11, 12, 20, 21]

    def test_the_mask_marks_exactly_the_visual_positions(self):
        tower = StubTower()
        batch = _batcher([_item(n_image=3), _item(n_image=2)], tower=tower).next_batch()

        marked = batch["visual_pos_masks"].astype(bool)
        is_visual = (batch["inputs"] == IMAGE_TOKEN) | (batch["inputs"] == VIDEO_TOKEN)
        assert np.array_equal(marked, is_visual)

    def test_the_tail_of_the_buffer_stays_zero(self):
        tower = StubTower()
        batch = _batcher([_item(n_image=3), _item(n_image=2)], tower=tower).next_batch()

        num_visual = int(batch["visual_pos_masks"].sum())
        assert batch["visual_embeds"].shape == (2 * 16, HIDDEN)
        assert not batch["visual_embeds"][num_visual:].any()

    def test_images_and_videos_interleave_by_their_own_cursors(self):
        """Two modalities, two independent counters -- and one output order.

        A row that carries both is where a single shared cursor would show up: the video
        embeddings would be read at the image positions' indices.
        """
        tower = StubTower()
        batch = _batcher([_item(n_image=2, n_video=3), _item(n_video=1)],
                         tower=tower).next_batch()

        num_visual = int(batch["visual_pos_masks"].sum())
        got = [v[0] for v in batch["visual_embeds"][:num_visual]]
        assert got == [10, 11, 1010, 1011, 1012, 1020]
        assert [c[0] for c in tower.calls] == ["image", "video"]

    def test_an_image_after_a_video_is_still_the_first_image(self):
        """The cursors count within a modality; the output counts across both."""
        tower = StubTower()
        batch = _batcher([_item(n_image=2, n_video=3, video_first=True)],
                         tower=tower, batch_size=1).next_batch()

        num_visual = int(batch["visual_pos_masks"].sum())
        assert [v[0] for v in batch["visual_embeds"][:num_visual]] == [1010, 1011, 1012, 10, 11]

    def test_a_batch_without_images_produces_an_empty_buffer(self):
        tower = StubTower()
        batch = _batcher([_item(), _item()], tower=tower).next_batch()

        assert int(batch["visual_pos_masks"].sum()) == 0
        assert not batch["visual_embeds"].any()
        assert tower.calls == []


class TestDeepstack:
    """The deepstack planes are the same tokens read at other depths.

    They are built by a second set of index expressions, so they can drift from plane 0
    without anything downstream noticing -- the shapes agree either way.
    """

    def test_every_layer_lands_where_plane_zero_does(self):
        tower = StubTower(deepstack_layers=3)
        batch = _batcher([_item(n_image=3), _item(n_image=2)], tower=tower).next_batch()

        num_visual = int(batch["visual_pos_masks"].sum())
        assert len(batch["deepstack_visual_embeds"]) == 3
        base = batch["visual_embeds"][:num_visual]
        for li, plane in enumerate(batch["deepstack_visual_embeds"]):
            assert plane.shape == batch["visual_embeds"].shape
            assert np.array_equal(plane[:num_visual], base + 100.0 * (li + 1))
            assert not plane[num_visual:].any()

    @pytest.mark.parametrize("video_first,expected", [
        (False, [10, 11, 1010, 1011, 1012]),
        (True, [1010, 1011, 1012, 10, 11]),
    ])
    def test_deepstack_follows_the_modality_cursors_too(self, video_first, expected):
        """Read at the output index instead of the modality's, and this is what parts.

        Both orderings are here because each hides one half of the mistake: with images
        first, an image's output slot equals its image slot; with videos first, a video's
        does. Only the modality that comes second has the two indices disagree.
        """
        tower = StubTower(deepstack_layers=2)
        batch = _batcher([_item(n_image=2, n_video=3, video_first=video_first)],
                         tower=tower, batch_size=1).next_batch()

        num_visual = int(batch["visual_pos_masks"].sum())
        for li, plane in enumerate(batch["deepstack_visual_embeds"]):
            assert [v[0] for v in plane[:num_visual]] == [e + 100.0 * (li + 1) for e in expected]

    def test_a_visual_free_batch_still_carries_one_plane_per_layer(self):
        """The model indexes the list by layer; a short list is an IndexError mid-step."""
        tower = StubTower(deepstack_layers=2)
        batch = _batcher([_item(), _item()], tower=tower).next_batch()

        assert len(batch["deepstack_visual_embeds"]) == 2
        for plane in batch["deepstack_visual_embeds"]:
            assert plane.shape == batch["visual_embeds"].shape
            assert not plane.any()

    def test_a_tower_with_no_deepstack_carries_no_planes(self):
        tower = StubTower(deepstack_layers=0)
        batch = _batcher([_item(n_image=3), _item(n_image=2)], tower=tower).next_batch()

        assert batch["deepstack_visual_embeds"] == []


class TestDisagreement:
    """Where the prompt's count and the tower's count differ."""

    def test_surplus_embeddings_are_ignored_rather_than_shifting_the_rest(self):
        """A tower that returns more rows than there are tokens: the leading ones win.

        The batcher also slices the surplus away, on plane 0 and the deepstack planes
        alike, and that slice is belt-and-braces: every read is at an index below the token
        count, so removing it changes no output. What this pins is the property that
        survives either way -- surplus at the end displaces nothing.
        """
        tower = StubTower(deepstack_layers=2)
        tower.extra_rows = 2
        batch = _batcher([_item(n_image=3), _item(n_image=2)], tower=tower).next_batch()

        num_visual = int(batch["visual_pos_masks"].sum())
        assert num_visual == 5
        assert [v[0] for v in batch["visual_embeds"][:num_visual]] == [10, 11, 12, 20, 21]
        for li, plane in enumerate(batch["deepstack_visual_embeds"]):
            assert [v[0] for v in plane[:num_visual]] == [10 + 100.0 * (li + 1),
                                                          11 + 100.0 * (li + 1),
                                                          12 + 100.0 * (li + 1),
                                                          20 + 100.0 * (li + 1),
                                                          21 + 100.0 * (li + 1)]

    def test_a_row_whose_counts_disagree_is_dropped_before_the_tower_runs(self):
        """Ten image tokens against a grid that describes three.

        The batcher cannot know which of the two is wrong, and either way it has no
        embedding for the seventh token -- so the row does not reach the tower at all.
        """
        bad = _item(n_image=3)
        bad["input_ids"] = np.array([1, 1] + [IMAGE_TOKEN] * 10 + [1], dtype=np.int32)
        bad["labels"] = bad["input_ids"].copy()

        tower = StubTower()
        batch = _batcher([bad, _item(n_image=2), _item(n_image=1)],
                         tower=tower).next_batch()

        assert int(batch["visual_pos_masks"].sum()) == 3
        assert [c[1] for c in tower.calls] == [[[1, 4, 2], [1, 4, 1]]]

    def test_a_row_with_visual_tokens_past_the_window_is_dropped(self):
        """Truncation would cut tokens the tower has already been asked to encode."""
        long_row = _item(n_image=2, prefix=14, suffix=0)
        tower = StubTower()
        batch = _batcher([long_row, _item(n_image=2), _item(n_image=1)],
                         tower=tower, seq_len=8).next_batch()

        assert int(batch["visual_pos_masks"].sum()) == 3

    def test_too_few_embeddings_is_an_error_not_a_short_batch(self):
        """Silently training on whatever arrived is the one outcome worth refusing."""

        class ShortTower(StubTower):
            def features(self, pixel_values, grid_thw, *, modality):
                embeddings, deepstack = super().features(pixel_values, grid_thw, modality=modality)
                return embeddings[:-1], [d[:-1] for d in deepstack]

        batcher = _batcher([_item(n_image=3), _item(n_image=2)], tower=ShortTower())
        with pytest.raises(ValueError, match="not enough image embeddings"):
            batcher.next_batch()
