"""Guard against a run that cannot take a single optimizer step.

Left unchecked, such a run loops zero times, writes out an adapter that never
saw a gradient, and reports success, or else dies inside the native dataloader
with ``No more files to load`` (``csrc/.../dataloader.cpp``) naming neither the
dataset nor the batch settings. Both are the same config defect: the data is
smaller than one step. Only the Python side knows the knobs, which is why the
check lives here rather than in the loader.
"""


class ZeroStepBudgetError(ValueError):
    """The configured budget cannot complete one optimizer step."""


def _starved(num_chunks, chunks_per_step, dataset_tokens, tokens_per_step) -> bool:
    """Whether the data cannot fill a single optimizer step.

    Chunks are what the loader actually serves, so prefer them: they are
    counted **per file** with a floor, which means a dataset of many files each
    shorter than the loader's chunk size yields zero chunks while its token
    total looks healthy. Token count is the fallback for the Ray path, which
    reaches its loader over RPC and exposes no chunk count today; it is an
    approximation that misses exactly that case.

    *chunks_per_step* must be supplied by the caller, not derived here: the
    loader is constructed with ``chunk_size`` as its unit, not ``sequence_len``
    (``trainer.py`` passes ``self.chunk_size``), and dispatch-pp redefines the
    relationship again. Only the caller knows the ratio it actually built.

    A non-positive *dataset_tokens* means "unknown", not "empty": the
    multimodal path reports ``-1`` when it cannot measure the dataset.
    """
    if num_chunks is not None and chunks_per_step:
        return num_chunks < chunks_per_step
    if dataset_tokens is not None and dataset_tokens > 0 and tokens_per_step:
        return dataset_tokens < tokens_per_step
    return False


def check_step_budget(
    max_steps: int,
    *,
    config=None,
    num_chunks: int | None = None,
    chunks_per_step: int | None = None,
    dataset_tokens: int | None = None,
    tokens_per_step: int | None = None,
) -> None:
    """Raise if the run cannot complete a full optimizer step.

    Two ways that happens and both must be caught here: the budget resolved to
    zero, which is what an epochs-based run yields when the floor divide gives
    0; or the dataset cannot fill one step even though ``max_steps`` was given
    explicitly. A typed-in number is not evidence the data exists.

    Deliberately *not* "fewer steps than requested": asking for more steps than
    one epoch holds is normal, the loader wraps between steps. Only a dataset
    too small for a single step is a defect.

    *config* is used only to shape the message; every count comes from the
    caller, which is the only place that knows what the loader was built with.
    """
    starved = _starved(num_chunks, chunks_per_step, dataset_tokens, tokens_per_step)

    if max_steps > 0 and not starved:
        return

    # Covers both causes without claiming a step count the user did not ask
    # for: someone who set max_steps=5 should not be told they asked for 0.
    detail = ""
    chunk_starved = num_chunks is not None and chunks_per_step and num_chunks < chunks_per_step
    if chunk_starved:
        # Report the unit that actually ran out. Token totals can look ample
        # here, because chunks are floored per file.
        detail = f" The dataset yields {num_chunks:,} loadable chunk(s) but one step needs {chunks_per_step:,}"
        if dataset_tokens is not None and dataset_tokens > 0:
            detail += (
                f"; its {dataset_tokens:,} tokens did not fill them, which usually "
                f"means individual files are shorter than one chunk"
            )
        detail += "."
    elif starved and dataset_tokens is not None and tokens_per_step:
        detail = f" The dataset holds {dataset_tokens:,} tokens but one step consumes {tokens_per_step:,}"
        # Only print the factorisation when it actually multiplies out: the Ray
        # path scales by num_nodes and dispatch-pp redefines the product, so a
        # blindly printed factor list would contradict the total beside it.
        if (
            config is not None
            and (
                config.per_device_train_batch_size
                * config.sequence_len
                * config.gpus
                * config.gradient_accumulation_steps
            )
            == tokens_per_step
        ):
            detail += (
                f" (per_device_train_batch_size={config.per_device_train_batch_size}"
                f" x sequence_len={config.sequence_len}"
                f" x gpus={config.gpus}"
                f" x gradient_accumulation_steps={config.gradient_accumulation_steps})"
            )
        detail += f", short by {tokens_per_step - dataset_tokens:,}."

    raise ZeroStepBudgetError(
        "This run cannot complete a single optimizer step, so it would produce "
        "an untrained model." + detail + " Use a larger dataset, a shorter sequence_len, or a smaller effective "
        "batch (batch size x gradient accumulation)."
    )
