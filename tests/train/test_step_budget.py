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
    # The message used to end "Set max_steps explicitly to override". That
    # advice was removed: an explicit max_steps never let anyone train on a
    # starved dataset, it only swapped this message for the dataloader's
    # `No more files to load`. Covered by
    # test_the_message_no_longer_advises_setting_max_steps.


def test_the_message_degrades_without_the_numbers():
    # Callers that cannot supply the arithmetic still get the headline.
    with pytest.raises(ZeroStepBudgetError) as exc:
        check_step_budget(0)
    assert "0 optimizer steps" in str(exc.value)
    assert "tokens but one step" not in str(exc.value)


# ── a starved dataset must fail even when max_steps was typed in ────


def test_explicit_max_steps_does_not_excuse_a_starved_dataset():
    """The guard used to ask only "is the step count positive?".

    A typed-in max_steps sailed past it while the data was still absent, and
    training then died in the dataloader with `No more files to load`, naming
    neither the dataset nor the batch settings. These are bug 36's numbers: 8
    examples packed into one 2048-token sequence, against a default step of
    2 x 2048 x 1 x 4.
    """
    with pytest.raises(ZeroStepBudgetError):
        check_step_budget(
            5,
            dataset_tokens=2048,
            tokens_per_step=16384,
            batch_size=2,
            sequence_len=2048,
            gpus=1,
            gradient_accumulation_steps=4,
        )


def test_the_message_names_both_numbers():
    with pytest.raises(ZeroStepBudgetError) as exc:
        check_step_budget(5, dataset_tokens=2048, tokens_per_step=16384)
    msg = str(exc.value)
    assert "2,048" in msg and "16,384" in msg


def test_the_message_no_longer_advises_setting_max_steps():
    """It used to end "Set max_steps explicitly to override", which never let
    anyone train: it swapped a clear error for `No more files to load`."""
    with pytest.raises(ZeroStepBudgetError) as exc:
        check_step_budget(0, dataset_tokens=500, tokens_per_step=16384)
    assert "explicitly to override" not in str(exc.value)


def test_exactly_one_step_of_data_is_allowed():
    """The boundary: enough for one step is enough. The loader wraps for more."""
    check_step_budget(5, dataset_tokens=16384, tokens_per_step=16384)


def test_asking_for_more_steps_than_one_epoch_is_fine():
    """Not "fewer steps than requested" — epochs wrap. Only a dataset that
    cannot fill a single step is a defect."""
    check_step_budget(1000, dataset_tokens=163840, tokens_per_step=16384)


def test_unknown_dataset_size_does_not_raise():
    """A loader that cannot report its token count must not be guessed at."""
    check_step_budget(5, dataset_tokens=None, tokens_per_step=16384)
