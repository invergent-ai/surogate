"""A run that cannot take a single optimizer step must fail, not 'succeed'.

The budget is counted in tokens, so a dataset smaller than one step's worth
yields 0 steps. Unguarded, the loop ran zero times, saved an adapter that
never saw a gradient, and reported "Training completed successfully" —
observed live on a 10-row dataset against the default 2 x 2048 x 4.
"""

import pytest

from surogate.train.step_budget import ZeroStepBudgetError, check_step_budget


def test_a_positive_budget_passes():
    check_step_budget(50)


@pytest.mark.parametrize("budget", [0, -1])
def test_a_nonpositive_budget_raises(budget):
    with pytest.raises(ZeroStepBudgetError):
        check_step_budget(budget)


def test_the_message_names_the_arithmetic_and_the_knobs():
    with pytest.raises(ZeroStepBudgetError) as exc:
        check_step_budget(
            0, dataset_tokens=2048, tokens_per_step=16384,
            batch_size=2, sequence_len=2048, gpus=1,
            gradient_accumulation_steps=4,
        )
    msg = str(exc.value)
    assert "2,048 tokens" in msg          # what you have
    assert "16,384" in msg                # what one step costs
    assert "short by 14,336" in msg       # the gap
    assert "per_device_train_batch_size=2" in msg
    assert "gradient_accumulation_steps=4" in msg
    assert "max_steps" in msg             # the override


def test_the_message_degrades_without_the_numbers():
    # Callers that cannot supply the arithmetic still get the headline.
    with pytest.raises(ZeroStepBudgetError) as exc:
        check_step_budget(0)
    assert "0 optimizer steps" in str(exc.value)
    assert "tokens but one step" not in str(exc.value)
