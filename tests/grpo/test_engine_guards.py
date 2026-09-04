"""Guards for two latent GRPO defects (E1, E2 in the RL end-to-end findings).

Neither *harmful* case is reachable from a config, a YAML or Studio today, and
both are silent if they ever do fire -- which is why they are worth closing
rather than leaving to be found in a training curve that merely looks
disappointing. The trailing single pad token below is reachable; it is benign,
and pinned so it stays that way.
"""

import numpy as np
import pytest

from surogate.core.config.grpo_orch_config import GRPOAdvantageConfig
from surogate.grpo.orchestrator.advantage import compute_advantages
from surogate.grpo.trainer import _find_sample_boundaries

# ── E1: a falsy advantage config must not become "no centering" ──────


def test_a_missing_advantage_config_is_rejected():
    """A falsy config used to hand the raw rewards back as advantages, with no
    baseline subtracted; see `compute_advantages` for why that trains wrong."""
    with pytest.raises(ValueError, match="advantage_config"):
        compute_advantages(
            rewards=[1.0, 2.0, 3.0, 4.0],
            completion_lengths=[10, 10, 10, 10],
            samples_per_problem=2,
            advantage_config=None,
        )


def test_a_real_advantage_config_still_centers_rewards():
    """The guard must not disturb the normal path."""
    out = compute_advantages(
        rewards=[1.0, 3.0],
        completion_lengths=[10, 10],
        samples_per_problem=2,
        advantage_config=GRPOAdvantageConfig({}),
    )
    assert out != [1.0, 3.0], "raw rewards must not pass through"
    assert sum(out) == pytest.approx(0.0), "a mean baseline leaves the group centered"


# ── E2: packed sample boundaries ─────────────────────────────────────


def test_ordinary_packed_samples_split_where_the_positions_reset():
    """The docstring's own example, and the case every real batch hits."""
    assert _find_sample_boundaries(np.array([0, 1, 2, 0, 1, 0, 1, 2, 3])) == [(0, 3), (3, 5), (5, 9)]


def test_a_single_unpacked_sample_is_one_range():
    assert _find_sample_boundaries(np.array([0, 1, 2, 3])) == [(0, 4)]


def test_a_lone_trailing_pad_token_becomes_its_own_range():
    """The one reachable 1-token case: a padding block of length 1. The old
    lookahead dropped this boundary and merged the pad into the previous sample."""
    assert _find_sample_boundaries(np.array([0, 1, 2, 0])) == [(0, 3), (3, 4)]


def test_a_multi_token_pad_block_was_already_split():
    """Padding of 2+ looks like any other sample, so only length-1 was affected."""
    assert _find_sample_boundaries(np.array([0, 1, 2, 0, 1, 2])) == [(0, 3), (3, 6)]


def test_an_interior_one_token_sample_splits_instead_of_contaminating():
    """The defect E2 names. `[0,1,2,0,0,1]` used to collapse into one unsplit
    range, so the next-token shift ran across both joins and one sample's
    gradient landed on another sample's token.

    The delta rule reads it correctly: a 3-token sample, a 1-token sample, then
    a 2-token sample.
    """
    assert _find_sample_boundaries(np.array([0, 1, 2, 0, 0, 1])) == [(0, 3), (3, 4), (4, 6)]


def test_an_empty_sequence_is_not_an_error():
    assert _find_sample_boundaries(np.array([], dtype=np.int32)) == [(0, 0)]
