"""A run that cannot take a single optimizer step must fail, not 'succeed'.

The budget is counted in tokens, so a dataset smaller than one step's worth
yields 0 steps. Unguarded, the loop ran zero times, saved an adapter that
never saw a gradient, and reported "Training completed successfully" —
observed live on a 10-row dataset against the default 2 x 2048 x 4.
"""

import pytest

from surogate.train.step_budget import ZeroStepBudgetError, check_step_budget


class Cfg:
    """Only the four fields the message reads."""

    def __init__(self, batch=2, seq=2048, gpus=1, ga=4):
        self.per_device_train_batch_size = batch
        self.sequence_len = seq
        self.gpus = gpus
        self.gradient_accumulation_steps = ga


def test_a_positive_budget_passes():
    check_step_budget(50)


@pytest.mark.parametrize("budget", [0, -1])
def test_a_nonpositive_budget_raises(budget):
    with pytest.raises(ZeroStepBudgetError):
        check_step_budget(budget)


def test_the_message_names_the_arithmetic_and_the_knobs():
    with pytest.raises(ZeroStepBudgetError) as exc:
        check_step_budget(
            0,
            dataset_tokens=2048,
            tokens_per_step=16384,
            config=Cfg(),
        )
    msg = str(exc.value)
    assert "2,048 tokens" in msg  # what you have
    assert "16,384" in msg  # what one step costs
    assert "short by 14,336" in msg  # the gap
    assert "per_device_train_batch_size=2" in msg
    assert "gradient_accumulation_steps=4" in msg


def test_the_message_degrades_without_the_numbers():
    # Callers that cannot supply the arithmetic still get the headline.
    with pytest.raises(ZeroStepBudgetError) as exc:
        check_step_budget(0)
    assert "cannot complete a single optimizer step" in str(exc.value)
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
            config=Cfg(),
        )


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


def test_a_healthy_chunk_count_is_not_rejected():
    """The ratio must come from the caller, not be derived from sequence_len.

    The loader is built with `chunk_size` (= bsz x seq x gpus) as its unit, so
    a step needs `total_batch_size // chunk_size` chunks, which is grad_accum.
    Deriving it as `tokens_per_step // sequence_len` overstates it by
    `bsz x gpus` and rejects runs that train fine.
    """
    check_step_budget(
        5,
        config=Cfg(),
        num_chunks=6,
        chunks_per_step=4,
        dataset_tokens=24_577,
        tokens_per_step=16_384,
    )


def test_unknown_dataset_size_is_not_treated_as_empty():
    """The multimodal path reports -1 when it cannot measure the dataset."""
    check_step_budget(5, config=Cfg(), dataset_tokens=-1, tokens_per_step=16_384)


def test_chunk_starvation_reports_chunks_not_a_negative_shortfall():
    """Tokens can look ample when chunks are the thing that ran out, because
    chunks are floored per file."""
    with pytest.raises(ZeroStepBudgetError) as exc:
        check_step_budget(
            5,
            config=Cfg(),
            num_chunks=1,
            chunks_per_step=4,
            dataset_tokens=24_577,
            tokens_per_step=16_384,
        )
    msg = str(exc.value)
    assert "chunk" in msg
    assert "short by -" not in msg


def test_the_factor_list_is_omitted_when_it_would_not_multiply_out():
    """The Ray path scales by num_nodes, so a blindly printed factorisation
    would contradict the total beside it."""
    with pytest.raises(ZeroStepBudgetError) as exc:
        check_step_budget(0, config=Cfg(), dataset_tokens=500, tokens_per_step=32_768)
    assert "per_device_train_batch_size=" not in str(exc.value)


def test_chunks_beat_tokens_when_the_two_disagree():
    """The token count is a proxy and it misses a real case.

    Chunks are counted per file with a floor, so a dataset of many files each
    shorter than one chunk yields zero chunks while its token total looks
    healthy. Before the chunk predicate this passed the guard and then died in
    the loader with `No more files to load`.
    """
    with pytest.raises(ZeroStepBudgetError):
        check_step_budget(
            5,
            config=Cfg(),
            num_chunks=0,  # 100 files, each shorter than one chunk
            chunks_per_step=4,
            dataset_tokens=204_800,  # ... while the token total looks ample
            tokens_per_step=16_384,
        )


def test_enough_chunks_passes_even_with_a_modest_token_count():
    check_step_budget(5, config=Cfg(), num_chunks=8, chunks_per_step=4, tokens_per_step=16_384)


def test_no_negative_shortfall_in_the_message():
    """max_steps can resolve to 0 with ample data (num_epochs: 0). The detail
    block used to run anyway and print `short by -3,616`."""
    with pytest.raises(ZeroStepBudgetError) as exc:
        check_step_budget(0, config=Cfg(), dataset_tokens=20_000, tokens_per_step=16_384)
    assert "short by -" not in str(exc.value)
