"""Guards for two latent GRPO defects (E1, E2 in the RL end-to-end findings).

Neither is reachable from a config, a YAML or Studio today. Both are silent if
they ever do fire, which is the reason to spend a guard on them rather than
leave them to be discovered from a training curve that merely looks wrong.
"""

import numpy as np
import pytest

from surogate.grpo.orchestrator.advantage import compute_advantages
from surogate.grpo.trainer import _find_sample_boundaries

# ── E1: a falsy advantage config must not become "no centering" ──────


def test_a_missing_advantage_config_is_rejected():
    """`if not advantage_config: return rewards` handed the raw rewards back as
    advantages, with no baseline subtracted. That is REINFORCE with all-positive
    advantages: every completion is pushed up, including the bad ones, variance
    reduction is gone and the policy can collapse. Silently.

    Not reachable through config today (the builder always yields a truthy
    `GRPOAdvantageConfig`), so the only way in is a caller passing None directly
    -- which is a bug in that caller, and should say so rather than train wrong.
    """
    with pytest.raises(ValueError, match="advantage_config"):
        compute_advantages(
            rewards=[1.0, 2.0, 3.0, 4.0],
            completion_lengths=[10, 10, 10, 10],
            samples_per_problem=2,
            advantage_config=None,
        )


def test_a_real_advantage_config_still_centers_rewards():
    """The guard must not disturb the normal path: equal-length completions in
    one group come back centered on their own mean."""
    from surogate.core.config.grpo_orch_config import GRPOAdvantageConfig

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
    pos = np.array([0, 1, 2, 0, 1, 0, 1, 2, 3])
    assert _find_sample_boundaries(pos) == [(0, 3), (3, 5), (5, 9)]


def test_a_single_unpacked_sample_is_one_range():
    assert _find_sample_boundaries(np.array([0, 1, 2, 3])) == [(0, 4)]


def test_a_lone_trailing_pad_token_becomes_its_own_range():
    """The one reachable 1-token case. The packer pads with `range(padding_size)`
    (`batch.py:140`), so a padding block of 1 is a single trailing `0`.

    The `pos[i+1] == 1` lookahead dropped this boundary and merged the pad into
    the previous sample. Benign either way -- a padding-only range carries no
    unmasked tokens and `compute_grpo_per_token_grads` skips it -- but splitting
    it is what the position ids actually say.
    """
    assert _find_sample_boundaries(np.array([0, 1, 2, 0])) == [(0, 3), (3, 4)]


def test_a_multi_token_pad_block_was_already_split():
    """Padding of 2 or more looks exactly like another sample, which is why only
    the length-1 pad was ever affected."""
    assert _find_sample_boundaries(np.array([0, 1, 2, 0, 1, 2])) == [(0, 3), (3, 6)]


def test_an_interior_one_token_sample_is_rejected_rather_than_mis_split():
    """Adjacent zeros cannot be represented in this scheme: `[0,0,1]` is
    ambiguous between "a 1-token sample then a 2-token sample" and "one sample
    whose positions start 0,0". No detector can resolve that, so the honest move
    is to refuse it.

    Left alone it was worse than ambiguous. `[0,1,2,0,0,1]` produced a single
    unsplit range covering everything, so the next-token shift ran straight
    across both boundaries and one sample's gradient landed on another's token.
    """
    with pytest.raises(ValueError, match="1-token"):
        _find_sample_boundaries(np.array([0, 1, 2, 0, 0, 1]))


def test_an_empty_sequence_is_not_an_error():
    assert _find_sample_boundaries(np.array([], dtype=np.int32)) == [(0, 0)]
