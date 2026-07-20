from __future__ import annotations

import pytest

from surogate.grpo.orchestrator.direct_scorer import (
    DeterministicAdapterScorer,
    DeterministicScorerError,
)


class StubDirectScorer(DeterministicAdapterScorer):
    def __init__(self, *, nondeterministic_prompt: tuple[int, ...] | None = None):
        self.repeat_count = 2
        self.nondeterministic_prompt = nondeterministic_prompt
        self.calls: dict[tuple[int, ...], int] = {}

    def score_completion(self, *, prompt_ids, completion_ids):
        prompt = tuple(prompt_ids)
        call = self.calls.get(prompt, 0)
        self.calls[prompt] = call + 1
        offset = 0.01 * call if prompt == self.nondeterministic_prompt else 0.0
        base = -0.4 if prompt == (1, 2) else -0.2
        return tuple(base - 0.1 * index - offset for index, _ in enumerate(completion_ids))


def test_matched_direct_scorer_requires_exact_repeats() -> None:
    scorer = StubDirectScorer()

    result = scorer.score_matched(
        ordinary_prompt_ids=[1, 2],
        hindsight_prompt_ids=[1, 2, 3],
        completion_ids=[7, 8],
    )

    assert result.reference_logprobs == (-0.4, -0.5)
    assert result.hindsight_logprobs == (-0.2, -0.30000000000000004)
    assert result.shifts == pytest.approx((0.2, 0.2))
    assert result.reference_repeat_count == 2
    assert result.hindsight_repeat_count == 2


def test_matched_direct_scorer_rejects_nondeterministic_branch() -> None:
    scorer = StubDirectScorer(nondeterministic_prompt=(1, 2, 3))

    with pytest.raises(DeterministicScorerError, match="hindsight branch"):
        scorer.score_matched(
            ordinary_prompt_ids=[1, 2],
            hindsight_prompt_ids=[1, 2, 3],
            completion_ids=[7, 8],
        )


def test_repeatable_completion_rejects_reference_drift() -> None:
    scorer = StubDirectScorer(nondeterministic_prompt=(1, 2))

    with pytest.raises(DeterministicScorerError, match="replay reference"):
        scorer.score_repeatable_completion(
            prompt_ids=[1, 2],
            completion_ids=[7, 8],
            branch_label="replay reference",
        )


def test_matched_direct_scorer_rejects_identical_branches() -> None:
    scorer = StubDirectScorer()

    with pytest.raises(DeterministicScorerError, match="identical"):
        scorer.score_matched(
            ordinary_prompt_ids=[1, 2],
            hindsight_prompt_ids=[1, 2],
            completion_ids=[7],
        )
